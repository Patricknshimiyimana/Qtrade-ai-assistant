from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional
from uuid import uuid4


@dataclass
class SupportSession:
    session_id: str
    created_at: datetime
    history: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.history),
            "history": list(self.history),
        }


class SessionStore:
    """Thread-safe in-memory store for chat sessions."""

    def __init__(self):
        self._lock = Lock()
        self._sessions: dict[str, SupportSession] = {}

    def create_session(self) -> SupportSession:
        session = SupportSession(
            session_id=uuid4().hex,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SupportSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def append_message(self, session_id: str, role: str, content: str) -> SupportSession:
        with self._lock:
            session = self._sessions[session_id]
            session.history.append({"role": role, "content": content})
            return session