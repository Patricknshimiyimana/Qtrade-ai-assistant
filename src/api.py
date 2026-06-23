import json
import logging
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

from src.pipeline import SupportPipeline
from src.schema import SessionCreateResponse, SessionMessageResponse
from src.session_store import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class SupportAPIState:
    pipeline: SupportPipeline
    sessions: SessionStore = field(default_factory=SessionStore)


def create_support_session(state: SupportAPIState) -> SessionCreateResponse:
    session = state.sessions.create_session()
    return SessionCreateResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        message_count=len(session.history),
    )


def send_message_to_session(
    state: SupportAPIState, session_id: str, message: str
) -> SessionMessageResponse:
    session = state.sessions.get_session(session_id)
    if session is None:
        raise KeyError(f"Unknown session_id: {session_id}")

    response = state.pipeline.process_query(message, session.history)
    state.sessions.append_message(session_id, "user", message)
    state.sessions.append_message(session_id, "assistant", response.answer)

    session_after = state.sessions.get_session(session_id)
    return SessionMessageResponse(
        session_id=session_id,
        response=response,
        history=list(session_after.history if session_after else []),
    )


class SupportAPIHandler(BaseHTTPRequestHandler):
    state: Optional[SupportAPIState] = None

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc

    def _send_error(self, status_code: int, message: str) -> None:
        self._write_json(status_code, {"error": message})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(200, {"status": "ok"})
            return

        self._send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.state is None:
            self._send_error(500, "API state is not configured.")
            return

        parsed = urlparse(self.path)
        path_parts = [part for part in parsed.path.split("/") if part]

        try:
            if parsed.path == "/sessions":
                payload = create_support_session(self.state)
                self._write_json(201, payload.model_dump())
                return

            if len(path_parts) == 3 and path_parts[0] == "sessions" and path_parts[2] == "messages":
                session_id = path_parts[1]
                body = self._read_json_body()
                message = str(body.get("message", "")).strip()
                if not message:
                    self._send_error(400, "Field 'message' is required.")
                    return

                payload = send_message_to_session(self.state, session_id, message)
                self._write_json(200, payload.model_dump())
                return

            self._send_error(404, "Not found")
        except KeyError as exc:
            self._send_error(404, str(exc))
        except ValueError as exc:
            self._send_error(400, str(exc))
        except Exception as exc:
            logger.exception("Unhandled API error")
            self._send_error(500, f"Unexpected API error: {exc}")


def run_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the HTTP API alongside the existing CLI."""
    print("Initializing knowledge base pipeline...")
    state = SupportAPIState(pipeline=SupportPipeline())
    SupportAPIHandler.state = state

    server = ThreadingHTTPServer((host, port), SupportAPIHandler)
    logger.info(f"Starting QTrade API on http://{host}:{port}")
    print(f"QTrade API running on http://{host}:{port} — press Ctrl+C to stop.")
    print("Endpoints: GET /health | POST /sessions | POST /sessions/{id}/messages")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("API shutdown requested.")
        print("\nShutting down QTrade API. Goodbye!")
    finally:
        server.server_close()