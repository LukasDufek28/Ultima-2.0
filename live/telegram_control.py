"""Lightweight Telegram polling helper for Ultima live control."""
from __future__ import annotations

import json
import queue
import threading
from typing import Iterable, Optional


class TelegramController:
    """Poll Telegram for bot commands and push them to a queue."""

    def __init__(
        self,
        token: str,
        allowed_user_ids: Iterable[int],
        command_queue: "queue.Queue[dict[str, object]]",
        poll_interval: float = 2.5,
    ) -> None:
        try:
            import requests as _requests  # Lazy import so dependency is optional until used
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError(
                "Telegram integration requires the 'requests' package. Install it with 'pip install requests'."
            ) from exc

        self._requests = _requests
        self._api_base = f"https://api.telegram.org/bot{token.strip()}"
        self._allowed_user_ids: set[int] = {int(uid) for uid in allowed_user_ids if uid is not None}
        self._command_queue = command_queue
        self._poll_interval = max(1.0, float(poll_interval))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._update_offset: Optional[int] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="TelegramPolling", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def send_message(self, chat_id: int, text: str) -> None:
        payload = {"chat_id": chat_id, "text": text}
        try:
            response = self._requests.post(
                f"{self._api_base}/sendMessage",
                data=payload,
                timeout=10,
            )
            if response.status_code >= 400:
                print(f"⚠️  Telegram sendMessage failed ({response.status_code}): {response.text}")
        except self._requests.exceptions.RequestException as exc:
            print(f"⚠️  Telegram sendMessage error: {exc}")

    def broadcast_message(self, text: str, chat_ids: Optional[Iterable[int]] = None) -> None:
        targets = list(chat_ids) if chat_ids else list(self._allowed_user_ids)
        if not targets:
            return
        for chat_id in targets:
            try:
                self.send_message(int(chat_id), text)
            except Exception as exc:  # pragma: no cover - best-effort broadcast
                print(f"⚠️  Telegram broadcast to {chat_id} failed: {exc}")

    # Internal helpers -----------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            # Use wait() so we can break early if stop_event is set mid-sleep
            self._stop_event.wait(self._poll_interval)

    def _poll_once(self) -> None:
        params: dict[str, object] = {"timeout": 10}
        if self._update_offset is not None:
            params["offset"] = self._update_offset
        try:
            response = self._requests.get(
                f"{self._api_base}/getUpdates",
                params=params,
                timeout=15,
            )
        except self._requests.exceptions.RequestException as exc:
            print(f"⚠️  Telegram polling error: {exc}")
            return

        data = self._safe_json(response)
        if not data or not data.get("ok", False):
            if data is not None:
                print(f"⚠️  Telegram getUpdates returned error payload: {data}")
            return

        results = data.get("result", [])
        for update in results:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._update_offset = update_id + 1

            message = update.get("message") or update.get("edited_message") or update.get("channel_post")
            if not message:
                continue

            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            sender = message.get("from") or {}
            user_id = sender.get("id")
            if chat_id is None or user_id is None:
                continue

            if self._allowed_user_ids and user_id not in self._allowed_user_ids:
                print(f"🚫 Telegram message from unauthorized id {user_id}; ignoring.")
                continue

            text = (message.get("text") or "").strip()
            if not text:
                continue

            command_payload = self._parse_command(text)
            if command_payload is None:
                continue

            command_payload.update(
                {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "message_id": message.get("message_id"),
                    "text": text,
                    "timestamp": message.get("date"),
                }
            )
            self._command_queue.put(command_payload)

    def _safe_json(self, response) -> Optional[dict]:
        try:
            return response.json()
        except ValueError:
            print(f"⚠️  Telegram response was not valid JSON: {response.text[:200]}")
            return None

    def _parse_command(self, text: str) -> Optional[dict[str, object]]:
        if not text.startswith("/"):
            return None

        parts = text.split()
        raw_command = parts[0][1:]
        if "@" in raw_command:
            raw_command = raw_command.split("@", 1)[0]
        command = raw_command.lower()
        args = parts[1:]

        if command in {
            "pause",
            "resume",
            "performance",
            "risk",
            "risklist",
            "status",
            "guards",
            "exposure",
            "flatten",
            "scaleout",
            "help",
            "kill",
            "confirmkill",
            "cancelkill",
            "start",
            "startbot",
        }:
            return {"name": command, "args": args}

        # Any other slash command bubbles up as unknown so the caller can respond politely
        return {"name": "unknown", "command": command, "args": args}
