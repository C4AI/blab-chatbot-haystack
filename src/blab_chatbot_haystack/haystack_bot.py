"""Haystack bot for BLAB."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, List, cast

from datasets import Dataset, DatasetDict
from haystack.document_stores import ElasticsearchDocumentStore, KeywordDocumentStore
from haystack.nodes import BaseRetriever, BM25Retriever, PreProcessor, Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.pipelines.base import Pipeline
from haystack.schema import Answer, Document
from haystack.utils import convert_files_to_docs
from transformers import (
    BatchEncoding,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from blab_chatbot_haystack import make_path_absolute

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
)
logging.getLogger("haystack").setLevel(logging.WARNING)


class CustomInputConverter:  # adapted from _BartEli5Converter
    def __call__(
        self,
        tokenizer: PreTrainedTokenizer,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> BatchEncoding:
        conditioned_doc = "<P> " + " <P> ".join([d.content for d in documents])
        query_and_docs = f"question: {query} context: {conditioned_doc}"
        enc: BatchEncoding = tokenizer(
            query_and_docs,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return enc


def custom_preprocess_function(
    tokenizer: PreTrainedTokenizer, max_input_length: int, max_answer_length: int
) -> Callable[[dict[str, Any]], BatchEncoding]:
    def f(instances: dict[str, Any]) -> BatchEncoding:
        inputs = list(
            map(
                lambda j: "question: "  # type: ignore
                + instances["questions"][j]
                + "context: "
                + (" <P> " if instances["supporting_documents"][j] else "")
                + " <P>".join(instances["supporting_documents"][j]),
                range(len(instances["questions"])),
            )
        )
        model_inputs: BatchEncoding = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )
        labels = tokenizer(
            instances["answers"], max_length=max_answer_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return f


class HaystackBot:
    """A bot that uses Haystack."""

    def __init__(
        self,
        doc_dir: str,
        model_dir: str,
        output_model_dir: str | None = None,
        conversion_args: dict[str, Any] | None = None,
        pre_processor_args: dict[str, Any] | None = None,
        es_args: dict[str, Any] | None = None,
        retriever_args: dict[str, Any] | None = None,
        generator_args: dict[str, Any] | None = None,
        generator_train_args: dict[str, Any] | None = None,
        training_qa_file_name: str | None = None,
        evaluation_qa_file_name: str | None = None,
        pipeline_retriever_args: dict[str, Any] | None = None,
        pipeline_generator_args: dict[str, Any] | None = None,
    ):
        """.

        Args:
            doc_dir: the directory that contains the documents
                (full path or relative to project root)
            model_dir: the directory that contains the model
                (full path or relative to project root)
            output_model_dir: the directory that will contain
                the model after training
                (full path or relative to project root)
                - required only when the model will be trained,
                in which case the argument "model_dir"
                represents the directory containing the
                initial model
            conversion_args: arguments to be passed to
                `convert_files_to_docs` (except for the document path,
                which is included automatically)
            pre_processor_args: arguments to be passed to `PreProcessor`
                constructor
            es_args: arguments to be passed to `ElasticsearchDocumentStore`
                constructor
            retriever_args: arguments to be passed to the retriever constructor
            generator_args: arguments to be passed to the generator constructor
            generator_train_args: arguments to be passed to the generator trainer
            training_qa_file_name: JSON file with quesiton-answer pairs to train
                the model
            evaluation_qa_file_name: JSON file with quesiton-answer pairs to
                evaluate the model
            pipeline_retriever_args: arguments to be passed to the retriever
                in the pipeline (to request an answer)
            pipeline_generator_args: arguments to be passed to the generator
                in the pipeline (to request an answer)
        """
        self.doc_dir = make_path_absolute(doc_dir)
        self.output_model_dir = (
            make_path_absolute(output_model_dir) if output_model_dir else None
        )
        self.model_dir = make_path_absolute(model_dir)
        self.docs: list[Document] = []
        self.doc_store: KeywordDocumentStore | None = None
        self.conversion_args: dict[str, Any] = conversion_args or {}
        self.pre_processor_args: dict[str, Any] = pre_processor_args or {}
        self.es_args: dict[str, Any] = es_args or {}
        self.retriever_args: dict[str, Any] = retriever_args or {}
        self.generator_args: dict[str, Any] = generator_args or {}
        self.generator_train_args: dict[str, Any] = generator_train_args or {}
        self.training_qa_file_name: str | None = (
            make_path_absolute(training_qa_file_name) if training_qa_file_name else None
        )
        self.evaluation_qa_file_name: str | None = (
            make_path_absolute(evaluation_qa_file_name)
            if evaluation_qa_file_name
            else None
        )
        self.pipeline_retriever_args: dict[str, Any] = pipeline_retriever_args or {}
        self.pipeline_generator_args: dict[str, Any] = pipeline_generator_args or {}
        self.retriever: BaseRetriever | None = None
        self.generator: Seq2SeqGenerator | None = None
        self.pipeline: Pipeline | None = None

    def connect_to_elastic_search(self) -> None:
        """Create a connection to Elastic Search.

        The arguments in the constructor's parameter "es_args" are used.
        """
        self.doc_store = ElasticsearchDocumentStore(**self.es_args)

    def load_documents(self) -> None:
        """Read documents to be indexed.

        The documents are read from the directory given in the constructor's
        parameter "doc_dir", and the arguments in the constructor's parameter
        "conversion_args" are used.
        """
        self.docs = convert_files_to_docs(self.doc_dir, **self.conversion_args)

    def pre_process(self) -> None:
        """Pre-process documents that will be indexed.

        The documents must have already been loaded.
        The arguments in the constructor's parameter "pre_processor_args"
        are used.
        """
        pre_processor = PreProcessor(**self.pre_processor_args)
        self.docs = pre_processor.process(self.docs)

    def index_documents(self) -> None:
        """Index documents using ElasticSearch.

        If a connection to ElasticSearch has not been created yet, it is
        created by this method.

        Existing documents in the same index are deleted.

        Documents are loaded, pre-processed and written to the
        ElasticSearch document store.
        """
        if not self.doc_store:
            self.connect_to_elastic_search()
        assert self.doc_store
        existing = self.doc_store.get_document_count()
        if existing:
            self.doc_store.delete_documents()
        if not self.docs:
            self.load_documents()
            self.pre_process()
        self.doc_store.write_documents(self.docs)

    def create_retriever(self) -> None:
        """Create a retriever.

        If a connection to ElasticSearch has not been created yet, it is
        created by this method.

        The arguments in the constructor's parameter "retriever_args"
        are used. Note that the ES document store is
        filled automatically and should not be included
        in "retriever_args".
        """
        if not self.doc_store:
            self.connect_to_elastic_search()
        self.retriever = BM25Retriever(
            **(dict(document_store=self.doc_store, **self.retriever_args))
        )

    def create_generator(self) -> None:
        """Create the generator.

        The arguments in the constructor's parameter "generator_args"
        are used. Note that the model directory and the input converter
        are filled automatically and should not be included
        in "generator_args".
        """
        self.generator = Seq2SeqGenerator(
            **(
                dict(
                    model_name_or_path=self.model_dir,
                    input_converter=CustomInputConverter(),
                    **self.generator_args,
                )
            ),
        )

    def _read_qa_data(self, file_name: str) -> Dataset:
        with open(file_name, encoding="utf-8") as fd:
            j = json.load(fd)
        questions: list[str] = []
        answers: list[str] = []
        documents: list[list[str]] = []
        for item in j["items"]:
            questions.append(item["q"])
            answers.append(item["a"])
            documents.append(item.get("d", []))
        return Dataset.from_dict(
            {
                "questions": questions,
                "answers": answers,
                "supporting_documents": documents,
            }
        )

    def train_generator(self) -> None:
        """Train the generator and save the resulting model.

        The model is saved in the directory specified by the
        constructor's parameter "output_model_dir". The directory
        must exist.

        Question-answer pairs with the respective lists of
        supporting documents are read from the files specified
        by the constructor's parameters "training_qa_file_name"
        and "evaluation_qa_file_name".

        The arguments in the constructor's parameter "generator_training_args"
        are used. Note that the output model directory is
        filled automatically and should not be included
        in "generator_training_args".
        """
        if not self.output_model_dir:
            raise ValueError("Output directory not set")
        if not Path(self.output_model_dir).is_dir():
            raise FileNotFoundError("Output directory does not exist")
        if not self.training_qa_file_name:
            raise ValueError("Training QA dataset file not set")
        if not self.generator:
            self.create_generator()
        assert self.generator

        collator = DataCollatorForSeq2Seq(
            tokenizer=self.generator.tokenizer, model=self.model_dir
        )
        training_data = self._read_qa_data(self.training_qa_file_name)
        if self.evaluation_qa_file_name:
            evaluation_data = self._read_qa_data(self.evaluation_qa_file_name)
        else:
            evaluation_data = Dataset.from_dict(
                {
                    "questions": [],
                    "answers": [],
                    "supporting_documents": [],
                }
            )
        data = DatasetDict({"train": training_data, "eval": evaluation_data})
        tokenized_data = data.map(
            custom_preprocess_function(self.generator.tokenizer, 2048, 512),
            batched=True,
        )

        training_args = Seq2SeqTrainingArguments(
            self.output_model_dir, **self.generator_train_args
        )

        trainer = Seq2SeqTrainer(
            model=self.generator.model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["eval"],
            tokenizer=self.generator.tokenizer,
            data_collator=collator,
        )
        trainer.train()  # type: ignore
        trainer.save_model()  # type: ignore

    def create_pipeline(self) -> None:
        """Create a Haystack pipeline.

        The generator and the retriever are created if they have not been created yet.
        """
        if not self.generator:
            self.create_generator()
        if not self.retriever:
            self.create_retriever()
        self.pipeline = GenerativeQAPipeline(self.generator, self.retriever).pipeline

    def answer(self, query: str) -> list[Answer]:
        """Generate answers to a query.

        A Haystack pipeline is created if it has not been created yet.

        The arguments in the constructor's parameters "pipeline_retriever_args"
        and "pipeline_generator_args" are used.
        """
        if not self.pipeline:
            self.create_pipeline()
        assert self.pipeline
        params = {
            "Retriever": self.pipeline_retriever_args,
            "Generator": self.pipeline_generator_args,
        }

        return cast(
            List[Answer],
            self.pipeline.run(
                query=query,
                params={k: v for k, v in params.items() if v},
            ).get("answers", []),
        )


__all__ = ("HaystackBot",)
