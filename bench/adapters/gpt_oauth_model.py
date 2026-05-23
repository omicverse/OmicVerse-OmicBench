"""Codex OAuth model class for mini-swe-agent.

Lets mini-swe-agent talk to OpenAI's GPT-5.x family via the user's
ChatGPT subscription (free with active plan) instead of via a paid
OpenAI API key. The Codex CLI's OAuth tokens at ``~/.codex/auth.json``
are loaded, refreshed when expired, and used as Bearer credentials
against ``https://chatgpt.com/backend-api/codex/responses`` (OpenAI's
*Responses API*, not Chat Completions).

The class subclasses ``minisweagent.models.litellm_model.LitellmModel``
and overrides only ``_query`` — everything else (action parsing, cost
tracking with ``ignore_errors``, observation formatting, retry loop) is
inherited from the upstream model.

Adapted from PantheonOS's ``gpt_adapter.py`` and ``oauth/codex.py``
(``aristoteleo/PantheonOS``, MIT licensed) — token refresh logic and
message/tool converters are direct ports of that implementation.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    parse_toolcall_actions,
)

logger = logging.getLogger("gpt_oauth_model")

# ============ Constants ============

AUTH_ISSUER = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ORIGINATOR = "pi"
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_CLI_AUTH = Path.home() / ".codex" / "auth.json"
TOKEN_REFRESH_SKEW_SEC = 300


# ============ JWT helpers ============


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = (token or "").split(".")
    if len(parts) != 3 or not parts[1]:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
    except Exception:
        return {}


def _jwt_org_context(token: str) -> dict[str, str]:
    payload = _decode_jwt_payload(token)
    nested = payload.get("https://api.openai.com/auth")
    claims = nested if isinstance(nested, dict) else {}
    out: dict[str, str] = {}
    for key in ("organization_id", "project_id", "chatgpt_account_id"):
        v = str(claims.get(key) or "").strip()
        if v:
            out[key] = v
    return out


def _token_expired(token: str, skew: int = TOKEN_REFRESH_SKEW_SEC) -> bool:
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)):
        return True
    return time.time() >= float(exp) - skew


# ============ Token storage ============


def _load_gpt_auth() -> dict[str, Any]:
    """Load tokens from ~/.codex/auth.json (Codex CLI's auth file)."""
    if not CODEX_CLI_AUTH.exists():
        raise RuntimeError(
            f"Codex CLI auth file not found at {CODEX_CLI_AUTH}. "
            "Run `codex login` first."
        )
    try:
        data = json.loads(CODEX_CLI_AUTH.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse {CODEX_CLI_AUTH}: {e}") from e
    tokens = data.get("tokens") or {}
    if not tokens.get("refresh_token"):
        raise RuntimeError(
            f"{CODEX_CLI_AUTH} has no refresh_token; please re-run `codex login`."
        )
    return data


def _save_gpt_auth(data: dict[str, Any]) -> None:
    """Persist refreshed tokens back to ~/.codex/auth.json so the next
    process can reuse them. We deliberately write to the SAME file the
    Codex CLI uses to avoid divergence (single source of truth)."""
    CODEX_CLI_AUTH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CODEX_CLI_AUTH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(CODEX_CLI_AUTH)


def _refresh_tokens(refresh_token: str) -> dict[str, str]:
    """Exchange refresh_token for new access_token + id_token + (rotated)
    refresh_token. Note: OpenAI's refresh tokens are single-use, so we
    persist the rotated value immediately and the caller writes it back."""
    resp = httpx.post(
        f"{AUTH_ISSUER}/oauth/token",
        data={
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    if not resp.is_success:
        raise RuntimeError(
            f"OAuth refresh failed: HTTP {resp.status_code} {resp.text[:300]}"
        )
    data = resp.json()
    access_token = str(data.get("access_token") or "").strip()
    id_token = str(data.get("id_token") or "").strip()
    next_refresh = str(data.get("refresh_token") or refresh_token).strip()
    if not access_token or not id_token:
        raise RuntimeError("OAuth refresh returned incomplete credentials")
    return {
        "id_token": id_token,
        "access_token": access_token,
        "refresh_token": next_refresh,
    }


# ============ Chat Completions ↔ Responses API converters ============
# Ported from PantheonOS's pantheon/utils/llm.py.


def _content_to_text(content: Any) -> str:
    """Flatten possibly-multimodal content into a string (we don't pass
    images in this bench, so a simple flatten is enough)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _convert_messages_to_responses_input(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Chat Completions messages → Responses API (instructions, input_items)."""
    instructions: str | None = None
    input_items: list[dict] = []
    first_system_seen = False

    for msg in messages:
        role = msg.get("role")
        content_raw = msg.get("content")

        if role == "system":
            text = _content_to_text(content_raw)
            if not first_system_seen:
                instructions = text
                first_system_seen = True
            else:
                input_items.append({
                    "role": "developer",
                    "content": [{"type": "input_text", "text": text}],
                })

        elif role == "user":
            input_items.append({
                "role": "user",
                "content": [{"type": "input_text", "text": _content_to_text(content_raw)}],
            })

        elif role == "assistant":
            text = _content_to_text(content_raw)
            if text:
                input_items.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                input_items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                })

        elif role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": _content_to_text(content_raw),
            })

    return instructions, input_items


def _convert_tools_for_responses(tools: list[dict] | None) -> list[dict] | None:
    """{type, function: {name, description, parameters}} → flat {type, name, ...}."""
    if not tools:
        return None
    out: list[dict] = []
    for tool in tools:
        func = tool.get("function", {})
        item: dict[str, Any] = {"type": "function", "name": func.get("name", "")}
        if "description" in func:
            item["description"] = func["description"]
        if "parameters" in func:
            item["parameters"] = func["parameters"]
        if "strict" in func:
            item["strict"] = func["strict"]
        out.append(item)
    return out


# ============ Pseudo litellm response ============
# LitellmModel.query() expects a litellm-style response. Build a minimal
# duck-typed object so the inherited code (_calculate_cost, _parse_actions,
# response.model_dump()) works unchanged. cost_tracking="ignore_errors"
# keeps cost calc happy.


class _DuckMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[dict] | None = None

    def model_dump(self, **_kw) -> dict:  # type: ignore[override]
        return {"role": self.role, "content": self.content,
                "tool_calls": self.tool_calls}


class _DuckChoice(BaseModel):
    finish_reason: str = "stop"
    message: _DuckMessage = _DuckMessage()


class _DuckResponse(BaseModel):
    choices: list[_DuckChoice] = []
    usage: dict[str, int] = {}
    model: str = ""

    def model_dump(self, **_kw) -> dict:  # type: ignore[override]
        return {
            "choices": [{
                "finish_reason": c.finish_reason,
                "message": c.message.model_dump(),
            } for c in self.choices],
            "usage": self.usage,
            "model": self.model,
        }


# ============ The model class ============


class GptOauthModelConfig(LitellmModelConfig):
    """Codex OAuth uses the Responses API on the user's ChatGPT
    backend; cost tracking via litellm's pricing tables doesn't apply
    (the calls are billed against the ChatGPT subscription, not per-token).
    Default cost_tracking to ``"ignore_errors"``."""

    cost_tracking: Any = "ignore_errors"
    reasoning_effort: str = "medium"
    """`low` / `medium` / `high` — passed through as ``reasoning.effort``."""


class GptOauthModel(LitellmModel):
    """mini-swe-agent ``LitellmModel`` subclass that talks to Codex via
    OAuth instead of via a paid OpenAI API key.

    Configuration: pass ``model_name`` like ``"gpt-5"``; tokens are read
    automatically from ``~/.codex/auth.json``. No ``api_key`` /
    ``api_base`` plumbing — everything is hardcoded to the Codex backend.
    """

    def __init__(self, *, config_class: Callable = GptOauthModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        # Lazy-loaded: don't crash at import time if the auth file is missing.
        self._auth: dict[str, Any] | None = None

    # ---- token management ------------------------------------------------

    def _ensure_token(self) -> tuple[str, str | None]:
        """Return (access_token, account_id), refreshing if expired."""
        if self._auth is None:
            self._auth = _load_gpt_auth()

        tokens = self._auth.get("tokens") or {}
        access_token = str(tokens.get("access_token") or "").strip()
        refresh_token = str(tokens.get("refresh_token") or "").strip()
        account_id = (tokens.get("account_id") or "").strip() or None

        if not access_token or _token_expired(access_token):
            if not refresh_token:
                raise RuntimeError(
                    "Codex access_token expired and no refresh_token available; "
                    "run `codex login`."
                )
            logger.info("Codex OAuth: access_token expired, refreshing...")
            refreshed = _refresh_tokens(refresh_token)
            claims = _jwt_org_context(refreshed["id_token"])
            self._auth["tokens"] = {
                **refreshed,
                "account_id": claims.get("chatgpt_account_id") or account_id,
                "organization_id": claims.get("organization_id"),
                "project_id": claims.get("project_id"),
            }
            self._auth["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                _save_gpt_auth(self._auth)
            except Exception as e:
                logger.warning(f"Codex OAuth: could not persist refreshed token: {e}")
            access_token = self._auth["tokens"]["access_token"]
            account_id = self._auth["tokens"].get("account_id") or account_id
        return access_token, account_id

    # ---- request --------------------------------------------------------

    def _build_headers(self, access_token: str, account_id: str | None) -> dict[str, str]:
        import platform
        h = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "originator": ORIGINATOR,
            "User-Agent": f"pi ({platform.system()} {platform.release()}; {platform.machine()})",
        }
        if account_id:
            h["chatgpt-account-id"] = account_id
        return h

    def _build_body(self, instructions: str | None, input_items: list[dict],
                    tools: list[dict] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "input": input_items,
            "instructions": instructions or "You are a helpful assistant.",
            "stream": True,
            "store": False,
            "parallel_tool_calls": True,
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            body["tools"] = tools
        effort = getattr(self.config, "reasoning_effort", None)
        if effort:
            body["reasoning"] = {"effort": effort}
        return body

    def _stream_one_call(self, body: dict[str, Any], headers: dict[str, str]) -> _DuckResponse:
        """POST to /codex/responses, parse the SSE stream, return a duck-typed
        response with content + tool_calls aggregated."""
        endpoint = f"{CODEX_BASE_URL}/codex/responses"
        text_parts: list[str] = []
        tool_calls_by_id: dict[str, dict[str, Any]] = {}
        item_to_call: dict[str, str] = {}
        usage: dict[str, int] = {}

        with httpx.Client(timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30)) as client:
            with client.stream("POST", endpoint, headers=headers, json=body) as resp:
                if resp.status_code == 401:
                    raise PermissionError("Codex OAuth token rejected (401); re-run `codex login`")
                if resp.status_code >= 400:
                    body_text = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Codex backend HTTP {resp.status_code}: {body_text[:400]}")
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    et = event.get("type", "")
                    if et == "response.output_text.delta":
                        text_parts.append(event.get("delta", ""))
                    elif et == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            cid = item.get("call_id", "")
                            iid = item.get("id", "")
                            item_to_call[iid] = cid
                            tool_calls_by_id[cid] = {
                                "name": item.get("name", ""),
                                "arguments": "",
                            }
                    elif et == "response.function_call_arguments.done":
                        iid = event.get("item_id", "")
                        cid = item_to_call.get(iid, "")
                        if cid and cid in tool_calls_by_id:
                            tool_calls_by_id[cid]["arguments"] = event.get("arguments", "")
                            if event.get("name"):
                                tool_calls_by_id[cid]["name"] = event["name"]
                    elif et == "response.completed":
                        u = event.get("response", {}).get("usage") or {}
                        if u:
                            usage = {
                                "prompt_tokens": int(u.get("input_tokens", 0)),
                                "completion_tokens": int(u.get("output_tokens", 0)),
                                "total_tokens": int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0)),
                            }
                    elif et == "response.failed":
                        err = event.get("response", {}).get("error", {})
                        raise RuntimeError(f"Codex response.failed: {err}")

        content = "".join(text_parts) if text_parts else None
        tool_calls = (
            [{"id": cid, "type": "function",
              "function": {"name": v["name"], "arguments": v["arguments"]}}
             for cid, v in tool_calls_by_id.items()]
            if tool_calls_by_id else None
        )
        return _DuckResponse(
            choices=[_DuckChoice(message=_DuckMessage(
                role="assistant", content=content, tool_calls=tool_calls,
            ))],
            usage=usage,
            model=self.config.model_name,
        )

    # ---- override LitellmModel hooks ------------------------------------

    def _query(self, messages: list[dict[str, str]], **kwargs):  # type: ignore[override]
        access_token, account_id = self._ensure_token()
        instructions, input_items = _convert_messages_to_responses_input(messages)
        tools = _convert_tools_for_responses([BASH_TOOL])
        body = self._build_body(instructions, input_items, tools)
        headers = self._build_headers(access_token, account_id)
        try:
            return self._stream_one_call(body, headers)
        except PermissionError:
            # 401: try one refresh then retry once.
            logger.info("Codex OAuth: forced refresh after 401")
            self._auth = _load_gpt_auth()
            access_token, account_id = self._ensure_token()
            headers = self._build_headers(access_token, account_id)
            return self._stream_one_call(body, headers)

    def _parse_actions(self, response) -> list[dict]:
        """Subclass hook — toolcalls in our duck response are already in the
        ``{id, type, function: {name, arguments}}`` Chat Completions shape."""
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []
        # parse_toolcall_actions expects objects with .id / .function.name /
        # .function.arguments; wrap dicts into a SimpleNamespace tree.
        from types import SimpleNamespace
        wrapped = []
        for tc in tool_calls:
            wrapped.append(SimpleNamespace(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            ))
        return parse_toolcall_actions(
            wrapped, format_error_template=self.config.format_error_template,
        )

    def _calculate_cost(self, response) -> dict[str, float]:
        # ChatGPT subscription, not per-token. Always zero, never error out.
        return {"cost": 0.0}

    def query(self, messages, **kwargs):  # type: ignore[override]
        # Skip the upstream's `_prepare_messages_for_api` (anthropic-specific
        # cache control / thinking-block reordering doesn't apply to Codex).
        prepared = [{k: v for k, v in m.items() if k != "extra"} for m in messages]
        # Keep retry behaviour from parent.
        from minisweagent.models.utils.retry import retry
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(prepared, **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message
