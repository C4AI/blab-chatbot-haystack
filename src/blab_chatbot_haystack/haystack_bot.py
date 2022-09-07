"""Haystack bot for BLAB."""
from __future__ import annotations

import logging
from typing import Any

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
logging.getLogger("haystack").setLevel(logging.INFO)


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
        reader_args: dict[str, Any] | None = None,
        reader_train_args: dict[str, Any] | None = None,
        pipeline_retriever_args: dict[str, Any] | None = None,
        pipeline_reader_args: dict[str, Any] | None = None,
    ):
        self.doc_dir = make_path_absolute(doc_dir)
        self.model_dir = make_path_absolute(model_dir)
        self.docs: list[Document] = []
        self.doc_store: KeywordDocumentStore | None = None
        self.conversion_args: dict[str, Any] = conversion_args or {}
        self.pre_processor_args: dict[str, Any] = pre_processor_args or {}
        self.es_args: dict[str, Any] = es_args or {}
        self.retriever_args: dict[str, Any] = retriever_args or {}
        self.reader_args: dict[str, Any] = reader_args or {}
        self.reader_train_args: dict[str, Any] = reader_train_args or {}
        self.retriever: BaseRetriever | None = None
        self.reader: Seq2SeqGenerator | None = None
        self.pipeline_retriever_args: dict[str, Any] = pipeline_retriever_args or {}
        self.pipeline_reader_args: dict[str, Any] = pipeline_reader_args or {}
        self.pipeline: Pipeline | None = None

    def connect_to_elastic_search(self) -> None:
        self.doc_store = ElasticsearchDocumentStore(**self.es_args)

    def load_documents(self) -> None:
        convert_files_to_docs(self.doc_dir, **self.conversion_args)

    def pre_process(self) -> None:
        pre_processor = PreProcessor(**self.pre_processor_args)
        self.docs = pre_processor.process(self.docs)

    def create_retriever(self) -> None:
        self.retriever = BM25Retriever(
            **(dict(document_store=self.doc_store, **self.retriever_args))
        )

    def create_reader(self) -> None:
        self.reader = Seq2SeqGenerator(
            **(dict(model_name_or_path=self.model_dir, **self.reader_args))
        )

    def train_reader(self) -> None:
        assert self.reader, "Create reader before running train_reader()"
        raise NotImplementedError
        # self.reader.train(**self.reader_train_args)

    def create_pipeline(self) -> None:
        if not self.reader:
            self.create_reader()
        if not self.retriever:
            self.create_retriever()
        self.pipeline = GenerativeQAPipeline(self.reader, self.retriever).pipeline

    def answer(self, query: str) -> Answer:
        if not self.pipeline:
            self.create_pipeline()
        assert self.pipeline
        params = {
            "Retriever": self.pipeline_retriever_args,
            "Reader": self.pipeline_reader_args,
        }
        return self.pipeline.run(
            query=query,
            params={k: v for k, v in params.items() if v},
        )


__all__ = ("HaystackBot",)
