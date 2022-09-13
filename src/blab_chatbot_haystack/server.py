"""HTTP server and WebSocket client used to interact with the controller."""

import json
import uuid
from threading import Thread

from flask import Flask, request
from websocket import WebSocketApp

from blab_chatbot_haystack.haystack_bot import HaystackBot

app = Flask(__name__)


class MessageType:
    """Represents a message type."""

    SYSTEM = "S"
    TEXT = "T"


@app.route("/", methods=["POST"])
def conversation_start() -> None:
    """Answer POST requests.

    This function will be called whenever there is a new conversation or
    an old connection is re-established.
    """
    # noinspection PyUnresolvedReferences
    bot: HaystackBot = app._BOT
    ws_url: str = app._WS_URL

    def on_message(ws_app: WebSocketApp, m: str) -> None:
        """Send a message answering the question.

        This function is called when the WebSocket connection receives a message.

        Args:
            ws_app: the WebSocketApp instance
            m: message or event (JSON-encoded)
        """
        contents = json.loads(m)
        if "message" in contents:
            message = contents["message"]
            # ignore system messages and our own messages
            if not message.get("sent_by_human", False):
                return
            # generate answers
            if not message.get("type", None) == MessageType.TEXT:
                answers = ["O Haystack entende apenas mensagens de texto."]
            else:
                answers = list(
                    map(lambda a: a.answer, bot.answer(message["text"]) or [])
                )
            for i, answer in enumerate(answers):
                msg_type = "T"
                local_id = str(uuid.uuid4()).replace("-", "")
                ans = {
                    "type": msg_type,
                    "text": answer,
                    "local_id": local_id,
                    "quoted_message_id": message["id"] if i == 0 else None,
                }
                # send answer
                Thread(target=lambda: ws_app.send(json.dumps(ans))).start()

    def on_open(ws_app: WebSocketApp) -> None:
        """Send a greeting message.

        This function is called when the WebSocket connection is opened.

        Args:
            ws_app: the WebSocketApp instance
        """
        # generate greeting message
        text = "HAYSTACK ESTÁ PRONTO"
        msg_type = "T"
        local_id = str(uuid.uuid4()).replace("-", "")
        greeting = {
            "type": msg_type,
            "text": text,
            "local_id": local_id,
        }
        # send greeting message
        ws_app.send(json.dumps(greeting))

    ws_url = ws_url + "/ws/chat/" + request.json["conversation_id"] + "/"
    ws = WebSocketApp(
        ws_url,
        on_message=on_message,
        cookie="sessionid=" + request.json["session"],
        on_open=on_open,
    )
    Thread(target=ws.run_forever).start()
    return ""


def start_server(host: str, port: int, bot: HaystackBot, ws_url: str) -> None:
    """
    Start the HTTP server.

    Args:
        host:
            host to listen on (127.0.0.1 to accept only local connections,
            0.0.0.0 to accept all connections)
        port: port to listen on
        bot: Haystack bot
        ws_url: URL of the WebSocket server which to connect to
            once a conversation starts
    """
    app._BOT = bot
    app._WS_URL = ws_url
    app.run(host=host, port=port)
