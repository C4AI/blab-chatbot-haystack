"""This is a template.

Create a copy of this file and name it "settings.py".
Fill in the arguments and settings.

Paths to files and directories can be either absolute or relative to
the project's root directory (which contains the README.md file).
"""

from __future__ import annotations

from typing import Any

from haystack.utils import clean_wiki_text

# Arguments used to connect with BLAB Controller:
SERVER_SETTINGS: dict[str, str | int] = {
    "HOST": "localhost",
    "PORT": 25228,
    "WS_URL": "",  # "ws://localhost:8000" in development environments
}

HAYSTACK_SETTINGS: dict[str, Any] = {
    # Directory that contains the text documents
    "DOC_DIR": "",
    #
    # Directory that contains the model:
    "MODEL_DIR": "",
    #
    # Directory where the model will be saved
    # (only used to train the model):
    "OUTPUT_MODEL_DIR": "",
    #
    # Arguments to be passed to `convert_files_to_docs()`:
    "CONVERSION_ARGS": {
        "split_paragraphs": True,
        "clean_func": clean_wiki_text,
    },
    #
    # Arguments to be passed to `PreProcessor` constructor:
    "PRE_PROCESSOR_ARGS": {
        "language": "pt",
    },
    #
    # ElasticSearch arguments, passed to `ElasticsearchDocumentStore` constructor:
    "ES_ARGS": {
        "host": "localhost",
        "index": "",
    },
    #
    # Arguments to be passed to `BM25Retriever` constructor
    # ("document_store" is filled automatically and should not be included here):
    "RETRIEVER_ARGS": {
        #
    },
    #
    # Arguments to be passed to `Seq2SeqGenerator` constructor
    # ("model_name_or_path" and "input_converter" are filled automatically
    # and should not be included here):
    "GENERATOR_ARGS": {
        "use_gpu": True,
    },
    #
    # Arguments to be passed to `Seq2SeqTrainingArguments`
    # (only used to train the model;
    # "output_dir" is filled automatically and should not be included here):
    "GENERATOR_TRAIN_ARGS": {
        "evaluation_strategy": "epoch",
        "learning_rate": 2e-05,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
        "save_total_limit": 3,
        "num_train_epochs": 25,
        "gradient_accumulation_steps": 3,
        "fp16": True,  # only with GPU
    },
    #
    # Name of the JSON file that contains question-answer pairs
    # (only used to train the model)
    "TRAINING_QA_FILE_NAME": "",
    #
    # Name of the JSON file that contains question-answer pairs
    # (only used to evaluate the model after training)
    "EVALUATION_QA_FILE_NAME": "",
    #
    # Arguments to be passed to the retriever in the pipeline:
    "PIPELINE_RETRIEVER_ARGS": {
        "top_k": 10,
    },
    #
    # Arguments to be passed to the generator in the pipeline:
    "PIPELINE_GENERATOR_ARGS": {
        #
    },
}
"""Settings for Haystack."""
