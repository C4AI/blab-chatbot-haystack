#!/usr/bin/env python3

"""This script can be used to run the main module."""

from pathlib import Path
from runpy import run_path

MODULE = "blab_chatbot_haystack"

run_path(str(Path(__file__).parent.resolve() / "src" / MODULE))
