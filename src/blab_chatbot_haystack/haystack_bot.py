"""Haystack bot for BLAB."""

from typing import Any

from haystack.pipelines.base import Pipeline
from haystack.pipelines import ExtractiveQAPipeline, BaseStandardPipeline
from haystack.schema import Document, Answer
from haystack.document_stores import ElasticsearchDocumentStore, KeywordDocumentStore
from haystack.nodes import (
    PreProcessor,
    BM25Retriever,
    FARMReader,
    BaseRetriever,
    BaseReader,
)
from haystack.utils import convert_files_to_docs

import logging

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
)
logging.getLogger("haystack").setLevel(logging.INFO)


class HaystackBot:
    """A bot that uses Haystack"""

    def __init__(
        self,
        doc_dir: str,
        model_dir: str,
        conversion_args: dict[str, Any] | None = None,
        pre_processor_args: dict[str, Any] | None = None,
        es_args: dict[str, Any] | None = None,
        retriever_params: dict[str, Any] | None = None,
        reader_params: dict[str, Any] | None = None,
    ):
        self.doc_dir = doc_dir
        self.model_dir = model_dir
        self.docs: list[Document] = []
        self.doc_store: KeywordDocumentStore | None = None
        self.conversion_args: dict[str, Any] = conversion_args or {}
        self.pre_processor_args: dict[str, Any] = pre_processor_args or {}
        self.es_args: dict[str, Any] = es_args or {}
        self.retriever: BaseRetriever | None = None
        self.reader: BaseReader | None = None
        self.retriever_params: dict[str, Any] = retriever_params or {}
        self.reader_params: dict[str, Any] = reader_params or {}
        self.pipeline: Pipeline | None = None

    def connect_to_elastic_search(self):
        self.doc_store = ElasticsearchDocumentStore(**self.es_args)

    def load_documents(self):
        convert_files_to_docs(self.doc_dir, **self.conversion_args)

    def pre_process(self):
        pre_processor = PreProcessor(**self.pre_processor_args)
        self.docs = pre_processor.process(self.docs)

    def create_retriever(self):
        self.retriever = BM25Retriever(document_store=self.doc_store)

    def create_reader(self):
        self.reader = FARMReader(model_name_or_path=self.model_dir)

    def create_pipeline(self):
        assert self.reader
        assert self.retriever
        self.pipeline = ExtractiveQAPipeline(self.reader, self.retriever).pipeline

    def answer(self, query: str) -> Answer:
        assert self.pipeline
        return self.pipeline.run(
            query=query,
            params={"Retriever": self.retriever_params, "Reader": self.reader_params},
        )
