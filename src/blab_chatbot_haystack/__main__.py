"""This module is called from the command-line."""

from __future__ import annotations

import argparse
from importlib import util as import_util
from pathlib import Path
from typing import Any

from blab_chatbot_haystack import make_path_absolute

# from blab_chatbot_haystack.server import start_server


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="settings.py")
    subparsers = parser.add_subparsers(help="command", dest="command")
    subparsers.add_parser("startserver", help="start server")
    subparsers.add_parser("answer", help="answer question typed on terminal")
    return parser


def load_config(p: str) -> dict[str, Any]:
    cfg_path = make_path_absolute(p)
    spec = import_util.spec_from_file_location(Path(cfg_path).name[:-3], cfg_path)
    assert spec
    assert spec.loader
    settings_module = import_util.module_from_spec(spec)
    spec.loader.exec_module(settings_module)
    if isinstance(settings_module.HAYSTACK_SETTINGS, dict):
        return settings_module.HAYSTACK_SETTINGS
    raise ValueError("Invalid settings file")


parser = create_arg_parser()
args = parser.parse_args()
config = load_config(args.config)

if args.command == "answer":
    from blab_chatbot_haystack.haystack_bot import HaystackBot

    bot = HaystackBot(**{k.lower(): v for k, v in config.items()})
    print("TYPE YOUR QUESTION AND PRESS ENTER.")
    while True:
        try:
            question = input(">> YOU: ")
        except (EOFError, KeyboardInterrupt):
            question = ""
        if not question:
            break
        for answer in bot.answer(question) or []:
            print(">> HAYSTACK: " + answer)

elif args.command == "startserver":
    # bot = HaystackBot()
    # start_server(
    #     host=config["server_host"],
    #     port=config.getint("server_port"),
    #     bot=bot,
    #     ws_url=config["ws_url"],
    # )
    pass
