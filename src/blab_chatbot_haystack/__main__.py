"""This module is called from the command-line."""

from __future__ import annotations

import argparse
from colorama import Style
from colorama import init as init_colorama

from importlib import util as import_util
from pathlib import Path
from typing import Any

from blab_chatbot_haystack import make_path_absolute
from blab_chatbot_haystack.haystack_bot import HaystackBot
from blab_chatbot_haystack.server import start_server


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="settings.py")
    subparsers = parser.add_subparsers(help="command", dest="command")
    subparsers.add_parser("startserver", help="start server")
    subparsers.add_parser("index", help="index documents")
    subparsers.add_parser("answer", help="answer question typed on terminal")
    subparsers.add_parser("train", help="train the model")
    return parser


def load_config(p: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg_path = make_path_absolute(p)
    spec = import_util.spec_from_file_location(Path(cfg_path).name[:-3], cfg_path)
    assert spec
    assert spec.loader
    settings_module = import_util.module_from_spec(spec)
    spec.loader.exec_module(settings_module)
    haystack_cfg = getattr(settings_module, "HAYSTACK_SETTINGS", None)
    server_cfg = getattr(settings_module, "SERVER_SETTINGS", None)
    if isinstance(haystack_cfg, dict) and isinstance(server_cfg, dict):
        return server_cfg, haystack_cfg
    raise ValueError("Invalid settings file")


parser = create_arg_parser()
args = parser.parse_args()
server_config, haystack_config = load_config(args.config)

bot = HaystackBot(**{k.lower(): v for k, v in haystack_config.items()})

if args.command == "index":
    bot.index_documents()
elif args.command == "answer":
    init_colorama()

    print("TYPE YOUR QUESTION AND PRESS ENTER.")
    while True:
        try:
            question = input(f"{Style.BRIGHT}\n>> YOU: {Style.RESET_ALL}")
        except (EOFError, KeyboardInterrupt):
            question = ""
        if not question:
            break
        for a in bot.answer(question) or []:
            print(
                f"{Style.BRIGHT}\n>> HAYSTACK {Style.RESET_ALL}"
                + f"{Style.DIM}(score={a.score}, context={a.context}){Style.BRIGHT}: {Style.RESET_ALL}"
                + a.answer
            )


elif args.command == "train":
    bot.train_generator()
elif args.command == "startserver":
    start_server(bot=bot, **{k.lower(): v for k, v in server_config.items()})
