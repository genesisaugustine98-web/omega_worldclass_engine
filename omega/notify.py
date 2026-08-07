from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


class TelegramNotifier:
    """Telegram notifier whose credentials come only from environment secrets."""

    def __init__(self, token: str | None = None, chat_id: str | None = None, enabled: bool = True):
        self.token = token or os.getenv("OMEGA_TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("OMEGA_TELEGRAM_CHAT_ID")
        self.enabled = enabled

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str) -> dict:
        if not self.enabled:
            return {"ok": False, "skipped": "disabled"}
        if not self.configured:
            return {"ok": False, "skipped": "credentials_missing"}
        text = message[:4000]
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        body = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text}).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload.get("ok"):
            raise RuntimeError("Telegram API rejected the notification")
        return {"ok": True, "message_id": payload["result"]["message_id"]}
