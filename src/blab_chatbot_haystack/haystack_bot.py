"""Haystack bot for BLAB."""
from __future__ import annotations

import logging
from typing import Any, cast, List

from haystack.document_stores import ElasticsearchDocumentStore, KeywordDocumentStore
from haystack.nodes import BaseRetriever, BM25Retriever, PreProcessor, Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.pipelines.base import Pipeline
from haystack.schema import Answer, Document
from haystack.utils import convert_files_to_docs

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
        documents: List[Document],
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


class HaystackBot:
    """A bot that uses Haystack."""

    def __init__(
        self,
        doc_dir: str,
        model_dir: str,
        conversion_args: dict[str, Any] | None = None,
        pre_processor_args: dict[str, Any] | None = None,
        es_args: dict[str, Any] | None = None,
        retriever_args: dict[str, Any] | None = None,
        generator_args: dict[str, Any] | None = None,
        generator_train_args: dict[str, Any] | None = None,
        pipeline_retriever_args: dict[str, Any] | None = None,
        pipeline_generator_args: dict[str, Any] | None = None,
    ):
        self.doc_dir = make_path_absolute(doc_dir)
        self.model_dir = make_path_absolute(model_dir)
        self.docs: list[Document] = []
        self.doc_store: KeywordDocumentStore | None = None
        self.conversion_args: dict[str, Any] = conversion_args or {}
        self.pre_processor_args: dict[str, Any] = pre_processor_args or {}
        self.es_args: dict[str, Any] = es_args or {}
        self.retriever_args: dict[str, Any] = retriever_args or {}
        self.generator_args: dict[str, Any] = generator_args or {}
        self.generator_train_args: dict[str, Any] = generator_train_args or {}
        self.retriever: BaseRetriever | None = None
        self.generator: Seq2SeqGenerator | None = None
        self.pipeline_retriever_args: dict[str, Any] = pipeline_retriever_args or {}
        self.pipeline_generator_args: dict[str, Any] = pipeline_generator_args or {}
        self.pipeline: Pipeline | None = None

    def connect_to_elastic_search(self) -> None:
        self.doc_store = ElasticsearchDocumentStore(**self.es_args)

    def load_documents(self) -> None:
        self.docs = convert_files_to_docs(self.doc_dir, **self.conversion_args)

    def pre_process(self) -> None:
        pre_processor = PreProcessor(**self.pre_processor_args)
        self.docs = pre_processor.process(self.docs)

    def index_documents(self) -> None:
        if not self.doc_store:
            self.connect_to_elastic_search()
        assert self.doc_store
        if not self.docs:
            self.load_documents()
            self.pre_process()
        self.doc_store.write_documents(self.docs)

    def create_retriever(self) -> None:
        if not self.doc_store:
            self.connect_to_elastic_search()
        self.retriever = BM25Retriever(
            **(dict(document_store=self.doc_store, **self.retriever_args))
        )

    def create_generator(self) -> None:
        self.generator = Seq2SeqGenerator(
            **(
                dict(
                    model_name_or_path=self.model_dir,
                    input_converter=CustomInputConverter(),
                    **self.generator_args,
                )
            ),
        )

    def train_generator(self) -> None:
        assert self.generator, "Create generator before running train_generator()"
        raise NotImplementedError
        # self.generator.train(**self.generator_train_args)

    def create_pipeline(self) -> None:
        if not self.generator:
            self.create_generator()
        if not self.retriever:
            self.create_retriever()
        self.pipeline = GenerativeQAPipeline(self.generator, self.retriever).pipeline

    def answer(self, query: str) -> list[Answer]:
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
