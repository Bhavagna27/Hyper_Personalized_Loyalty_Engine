from __future__ import annotations

import html
import re
from typing import Any

_CSS_TOKEN_PATTERN = re.compile(r"[^a-z0-9_-]+")


def esc(value: Any, default: str = "") -> str:
    """Escape a value before interpolating it into a raw HTML block."""
    if value is None:
        return default
    return html.escape(str(value), quote=True)


def css_token(value: Any, default: str = "") -> str:
    """Reduce a value to a safe CSS class token."""
    if value is None:
        return default
    token = _CSS_TOKEN_PATTERN.sub("", str(value).strip().lower())
    return token or default
