from __future__ import annotations

import os
from collections.abc import Callable, Iterable


SECRET_NAMES = (
    "OMEGA_TELEGRAM_BOT_TOKEN",
    "OMEGA_TELEGRAM_CHAT_ID",
    "OMEGA_OANDA_TOKEN",
    "OMEGA_TWELVEDATA_API_KEY",
    "OMEGA_POLYGON_API_KEY",
)


def load_platform_secrets(
    names: Iterable[str] = SECRET_NAMES,
    getter: Callable[[str], str | None] | None = None,
) -> dict[str, bool]:
    """Load missing secrets from Colab/Kaggle without returning secret values."""
    if getter is None:
        getter = _detect_secret_getter()
    status: dict[str, bool] = {}
    for name in names:
        value = os.getenv(name)
        if not value and getter is not None:
            try:
                value = getter(name)
            except Exception:
                value = None
            if value:
                os.environ[name] = value
        status[name] = bool(value)
    return status


def _detect_secret_getter() -> Callable[[str], str | None] | None:
    try:
        from google.colab import userdata

        return userdata.get
    except ImportError:
        pass
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        return client.get_secret
    except ImportError:
        return None


# CONVERSATION_HOOK: add provider secret names only; never serialize their values.