"""HTTP client for the firm's LitKit instance, as the matter host calls it.

Authentication (LitKit agent-token lane):

* ``Authorization: Bearer lkm_...`` on every request. The token is pinned to one matter;
  a request about any other matter answers ``403 agent_token_matter_mismatch``.
* During a turn, ``X-LitKit-Acting-User`` and ``X-LitKit-User-Assertion`` name the lawyer
  the turn acts for, so LitKit applies that lawyer's roles, walls and audit. The assertion
  is minted fresh for every request (HMAC over ``v1.<user>.<matter>.<iatMs>.<ttlMs>``,
  keyed on the host secret; see :mod:`litco.assertion`). Without a turn (cron) no
  assertion is sent and the Matter Agent user's own viewer role applies.
* ``X-LitKit-Turn-Grant`` goes only on the cross-matter search, and only when the app
  minted a grant for the turn (FIRM_AGENT_HOST 6.3).
* No ``Origin`` header is sent.

Errors are honest: 401/403 raise :class:`LitKitPermissionError` and are never retried. 429
and 5xx are retried with exponential backoff for reads; writes are retried only where the
server cannot have acted (429, 503, or a connection that never opened). The token and the
host secret never appear in logs, reprs, or error messages.

Every refusal and server error is classified by whom it is about (:func:`refusal_kind`), and
:meth:`LitKitError.to_dict` gives the model one ``message`` and one ``next`` for that class
(:func:`explain_refusal`). This is the only place that text lives. A LitKit gate on the agent
host itself (``mfa_required`` on an agent-token request) is never the person's to fix: the
person never signs in, enrolls MFA or changes a setting for the agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import httpx

from litco import __version__
from litco.assertion import DEFAULT_TTL_MS, mint_user_assertion
from litco.litkit.context import current_acting_user

logger = logging.getLogger("litco.litkit.client")

ACTING_USER_HEADER = "X-LitKit-Acting-User"
ASSERTION_HEADER = "X-LitKit-User-Assertion"
TURN_GRANT_HEADER = "X-LitKit-Turn-Grant"
TOKEN_PREFIX = "lkm_"
RETRY_STATUSES_READ = frozenset({429, 500, 502, 503, 504})
RETRY_STATUSES_WRITE = frozenset({429, 503})
PERMISSION_MESSAGE = "not permitted for this user on this matter"
PLATFORM_MESSAGE = "LitKit refused this action for the agent host"
HOST_MESSAGE = "LitKit rejected this agent host's credentials"

# Who a refusal is about. The person: the lawyer on the turn may not see or do this. The
# platform: LitKit gates the agent host itself, whoever is on the turn. The host: this host's
# token or assertion is wrong, which the LitCo operator fixes. LitKit: the server failed.
REFUSAL_PERSON = "person"
REFUSAL_PLATFORM = "platform"
REFUSAL_HOST = "host"
REFUSAL_LITKIT = "litkit_error"
# Every request carries the agent token, so these 403s gate the agent host, never the person.
_PLATFORM_CODES = frozenset({"mfa_required", "agent_token_forbidden", "agent_token_required",
                             "acting_user_required"})
_HOST_CODES = frozenset({"agent_token_matter_mismatch", "user_assertion_required", "acting_user_invalid",
                         "acting_user_not_found", "acting_user_is_service"})

_CURRENT = object()  # sentinel: "use the current turn's acting user"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

def refusal_kind(status: int, code: Optional[str], *, acting: bool) -> Optional[str]:
    """Whom a failed request is about; ``None`` for answers that are not refusals or server errors."""
    code = str(code or "")
    if status == 401 or code in _HOST_CODES or code.startswith("assertion_"):
        return REFUSAL_HOST
    if status == 403:
        # Without a lawyer on the turn, LitKit judged the Matter Agent user, not a person.
        return REFUSAL_PLATFORM if code in _PLATFORM_CODES or not acting else REFUSAL_PERSON
    if status >= 500:
        return REFUSAL_LITKIT
    return None


def explain_refusal(kind: str, *, tool: Optional[str] = None, status: int = 0,
                    code: Optional[str] = None) -> Tuple[str, str]:
    """``(message, next)`` for the model: what LitKit said about the agent's action, and the step the agent takes."""
    name = f"`{tool}`" if tool else "this call"
    said = code or f"HTTP {status}"
    never = ("Never ask the person to sign in, enroll MFA, refresh a session, check their settings, or otherwise "
             "fix their LitKit account, and never describe this as about their session or identity.")
    if kind == REFUSAL_PLATFORM:
        return (f"LitKit refused {name} for the agent host ({said}). This is a limit LitKit places on the agent; "
                "nothing about the person's sign-in or account is involved.",
                f"Report it as a LitKit limit on {name} and offer the other path: a narrower request, another "
                "litkit_* tool that does the job, or a review run. If none fits, say LitCo needs to fix this and "
                f"that you have noted it. {never}")
    if kind == REFUSAL_HOST:
        return (f"LitKit rejected this agent host's credentials on {name} ({said}). The LitCo operator fixes "
                "this; the person is not involved.",
                f"Tell the person LitCo needs to fix the agent host's connection to LitKit and that you have "
                f"noted it. {never} Do not look for credentials.")
    if kind == REFUSAL_PERSON:
        return (f"LitKit says the person on this turn may not see or do what {name} asked ({said}).",
                "Tell the person plainly that their LitKit access does not cover this and that a matter admin "
                "can grant it. Do not try another route.")
    return (f"LitKit failed on {name} ({said}).",
            f"Retry {name} once, with a smaller request if it was large. If it fails again, report it as a "
            "LitKit error on that tool, quoting this message, and say what you will do instead.")


class LitKitError(Exception):
    """A LitKit call failed. ``status`` is the HTTP status (0 for transport failures); ``acting`` says
    whether a lawyer's assertion went with the request."""

    def __init__(self, message: str, *, status: int = 0, code: Optional[str] = None,
                 body: Any = None, method: str = "", path: str = "", acting: bool = False):
        super().__init__(message)
        self.status = status
        self.code = code
        self.body = body
        self.method = method
        self.path = path
        self.acting = acting

    @property
    def kind(self) -> Optional[str]:
        return refusal_kind(self.status, self.code, acting=self.acting)

    def to_dict(self, tool: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"error": str(self), "status": self.status}
        if self.code:
            out["code"] = self.code
        if isinstance(self.body, dict):
            for key in ("errorKind", "token", "detail"):
                if key in self.body and key not in out:
                    out[key] = self.body[key]
        kind = self.kind
        if kind:
            out["refusal"] = kind
            out["message"], out["next"] = explain_refusal(kind, tool=tool, status=self.status, code=self.code)
        return out


class LitKitPermissionError(LitKitError):
    """401 or 403. :attr:`kind` says whether the person, the platform, or this host's configuration
    is refused; only the person's refusal is ``permission_denied``."""

    def to_dict(self, tool: Optional[str] = None) -> Dict[str, Any]:
        out = super().to_dict(tool)
        if self.kind == REFUSAL_PERSON:
            out["permission_denied"] = True
        return out


class LitKitNotConfigured(LitKitError):
    pass


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def _secret(name: str) -> str:
    try:
        from agent.secret_scope import get_secret
        value = get_secret(name)
    except ImportError:
        value = os.environ.get(name)
    return str(value or "").strip()


@dataclass(frozen=True)
class LitKitConfig:
    instance_url: str
    token: str
    host_secret: str = ""
    matter_id: str = ""

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "LitKitConfig":
        if env is not None:
            get = lambda k: str(env.get(k) or "").strip()  # noqa: E731
        else:
            get = _secret
        return cls(instance_url=get("LITCO_INSTANCE_URL").rstrip("/"), token=get("LITCO_AGENT_TOKEN"),
                   host_secret=get("LITCO_HOST_SECRET"), matter_id=get("LITCO_MATTER_ID"))

    @property
    def configured(self) -> bool:
        return bool(self.instance_url and self.token)

    def __repr__(self) -> str:  # never print the token or the secret
        return (f"LitKitConfig(instance_url={self.instance_url!r}, token={'set' if self.token else 'unset'}, "
                f"host_secret={'set' if self.host_secret else 'unset'}, matter_id={self.matter_id!r})")

    __str__ = __repr__


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

def _error_code(body: Any) -> Optional[str]:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, str):
            return err
        if isinstance(err, dict):
            code = err.get("code") or err.get("message")
            return str(code) if code else None
        for key in ("errorKind", "code"):
            if isinstance(body.get(key), str):
                return body[key]
    return None


def _parse_body(resp: httpx.Response) -> Any:
    ctype = resp.headers.get("content-type", "")
    if "json" in ctype and "ndjson" not in ctype:
        try:
            return resp.json()
        except Exception:
            pass
    try:
        text = resp.text
    except Exception:
        return None
    return text[:2000] if text else None


def _retry_after_seconds(resp: Optional[httpx.Response]) -> Optional[float]:
    if resp is None:
        return None
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 60.0))
    except ValueError:
        return None


class LitKitClient:
    """Synchronous LitKit client. One instance can serve many turns; the acting user is read
    from the turn context on every request unless ``acting_user=`` is passed explicitly."""

    def __init__(self, config: Optional[LitKitConfig] = None, *, transport: Optional[httpx.BaseTransport] = None,
                 max_retries: int = 4, backoff_base: float = 0.5, backoff_max: float = 20.0,
                 timeout: float = 120.0, sleep: Callable[[float], None] = time.sleep,
                 now_ms: Optional[Callable[[], int]] = None, assertion_ttl_ms: int = DEFAULT_TTL_MS):
        self.config = config or LitKitConfig.from_env()
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self._sleep = sleep
        self._now_ms = now_ms
        self._ttl_ms = assertion_ttl_ms
        self._transport = transport
        self._http: Optional[httpx.Client] = None
        self._matter_id: Optional[str] = self.config.matter_id or None

    # -- plumbing ----------------------------------------------------------------
    def __repr__(self) -> str:
        return f"LitKitClient({self.config!r})"

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(transport=self._transport, timeout=httpx.Timeout(self.timeout, connect=15.0),
                                      follow_redirects=False)
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def _require_config(self) -> None:
        if not self.config.configured:
            raise LitKitNotConfigured("LitKit is not configured on this host (LITCO_INSTANCE_URL and "
                                      "LITCO_AGENT_TOKEN are required)")

    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            raise ValueError("LitKit paths are relative to the instance")
        return self.config.instance_url + ("" if path.startswith("/") else "/") + path

    def headers(self, acting_user: Any = _CURRENT) -> Dict[str, str]:
        """Request headers: bearer token plus, when a user is acting, a freshly minted assertion."""
        self._require_config()
        user = current_acting_user() if acting_user is _CURRENT else acting_user
        out = {"Authorization": f"Bearer {self.config.token}", "Accept": "application/json",
               "User-Agent": f"litco-agent/{__version__}"}
        if user:
            matter = self.matter_id
            if not self.config.host_secret:
                raise LitKitNotConfigured("LITCO_HOST_SECRET is required to act for a user")
            now = self._now_ms() if self._now_ms else None
            out[ACTING_USER_HEADER] = str(user)
            out[ASSERTION_HEADER] = mint_user_assertion(str(user), matter, self.config.host_secret,
                                                        ttl_ms=self._ttl_ms, now_ms=now)
        return out

    @property
    def matter_id(self) -> str:
        """The one matter this host serves: ``LITCO_MATTER_ID``, else the only matter the token sees."""
        if self._matter_id:
            return self._matter_id
        body = self._request_json("GET", "/api/matters", acting_user=None)
        matters = body.get("matters") if isinstance(body, dict) else None
        if not isinstance(matters, list) or len(matters) != 1 or not isinstance(matters[0], dict):
            raise LitKitError("could not resolve this host's matter: the agent token should see exactly one "
                              "matter; set LITCO_MATTER_ID", status=0)
        self._matter_id = str(matters[0].get("id") or "")
        return self._matter_id

    def _delay(self, attempt: int, resp: Optional[httpx.Response]) -> float:
        hinted = _retry_after_seconds(resp)
        if hinted is not None:
            return hinted
        base = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        return base * (0.75 + random.random() * 0.5)

    def _raise_for(self, resp: httpx.Response, method: str, path: str, *, acting: bool = False) -> None:
        body = _parse_body(resp)
        code = _error_code(body)
        status = resp.status_code
        if status in (401, 403):
            detail = f" (HTTP {status}{': ' + code if code else ''})"
            kind = refusal_kind(status, code, acting=acting)
            if code == "agent_token_matter_mismatch":
                msg = f"{HOST_MESSAGE}: the agent token is pinned to a different matter{detail}"
            elif kind == REFUSAL_HOST:
                msg = f"{HOST_MESSAGE}{detail}"
            elif kind == REFUSAL_PLATFORM:
                msg = f"{PLATFORM_MESSAGE}{detail}"
            else:
                msg = f"{PERMISSION_MESSAGE}{detail}"
            raise LitKitPermissionError(msg, status=status, code=code, body=body, method=method, path=path,
                                        acting=acting)
        if status == 404:
            msg = f"not found, or not visible to this user (HTTP 404{': ' + code if code else ''})"
        elif status == 504:
            msg = f"LitKit timed out (HTTP 504{': ' + code if code else ''})"
        elif status == 429:
            msg = "LitKit is rate limiting this host (HTTP 429); retries exhausted"
        elif status >= 500:
            msg = f"LitKit server error (HTTP {status}{': ' + code if code else ''})"
        else:
            msg = f"HTTP {status}{': ' + code if code else ''}"
        raise LitKitError(msg, status=status, code=code, body=body, method=method, path=path, acting=acting)

    def _send(self, method: str, path: str, *, params: Optional[Mapping[str, Any]] = None, json_body: Any = None,
              files: Any = None, data: Optional[Mapping[str, Any]] = None, acting_user: Any = _CURRENT,
              idempotent: Optional[bool] = None, ok_statuses: Iterable[int] = (), stream: bool = False,
              timeout: Optional[float] = None, extra_headers: Optional[Mapping[str, str]] = None) -> httpx.Response:
        """Send with retries. Returns the response (streamed and still open when ``stream``).

        ``extra_headers`` ride beside the auth headers and, like them, are never logged."""
        self._require_config()
        method = method.upper()
        if idempotent is None:
            idempotent = method in ("GET", "HEAD")
        retry_statuses = RETRY_STATUSES_READ if idempotent else RETRY_STATUSES_WRITE
        ok = set(ok_statuses)
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        attempt = 0
        while True:
            headers = {**(extra_headers or {}), **self.headers(acting_user)}  # fresh assertion per attempt
            if files is not None:
                for _name, spec in (files.items() if isinstance(files, dict) else files):
                    if isinstance(spec, tuple) and len(spec) >= 2 and hasattr(spec[1], "seek"):
                        spec[1].seek(0)
            req = self.http.build_request(method, self.url(path), params=clean_params or None, json=json_body,
                                          files=files, data=data, headers=headers,
                                          timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT)
            try:
                resp = self.http.send(req, stream=stream)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt < self.max_retries:
                    logger.info("litkit %s %s: connect failed (%s); retrying", method, path, exc.__class__.__name__)
                    self._sleep(self._delay(attempt, None))
                    attempt += 1
                    continue
                raise LitKitError(f"could not reach LitKit ({exc.__class__.__name__})", method=method,
                                  path=path) from None
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if idempotent and attempt < self.max_retries:
                    self._sleep(self._delay(attempt, None))
                    attempt += 1
                    continue
                raise LitKitError(f"LitKit request failed ({exc.__class__.__name__})", method=method,
                                  path=path) from None
            status = resp.status_code
            if status < 400 or status in ok:
                return resp
            if status in retry_statuses and attempt < self.max_retries:
                delay = self._delay(attempt, resp)
                resp.close()
                logger.info("litkit %s %s: HTTP %s; retry %d in %.1fs", method, path, status, attempt + 1, delay)
                self._sleep(delay)
                attempt += 1
                continue
            try:
                if stream:
                    resp.read()
                self._raise_for(resp, method, path, acting=ACTING_USER_HEADER in headers)
            finally:
                resp.close()

    # -- public API --------------------------------------------------------------
    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """A buffered request; raises :class:`LitKitError` on failure."""
        return self._send(method, path, **kwargs)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._send(method, path, **kwargs)
        body = _parse_body(resp)
        if isinstance(body, str) and resp.headers.get("content-type", "").startswith("application/json"):
            try:
                body = json.loads(body)
            except Exception:
                pass
        return body

    def get(self, path: str, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Any:
        return self._request_json("GET", path, params=params, **kwargs)

    def post(self, path: str, json_body: Any = None, **kwargs: Any) -> Any:
        return self._request_json("POST", path, json_body=json_body, **kwargs)

    def patch(self, path: str, json_body: Any = None, **kwargs: Any) -> Any:
        return self._request_json("PATCH", path, json_body=json_body, **kwargs)

    def delete(self, path: str, json_body: Any = None, **kwargs: Any) -> Any:
        return self._request_json("DELETE", path, json_body=json_body, **kwargs)

    def json_with_status(self, method: str, path: str, **kwargs: Any) -> Tuple[int, Any]:
        """Like :meth:`get`/:meth:`post` but returns ``(status, body)``; ``ok_statuses`` pass through."""
        resp = self._send(method, path, **kwargs)
        return resp.status_code, _parse_body(resp)

    def stream_ndjson(self, method: str, path: str, *, json_body: Any = None,
                      params: Optional[Mapping[str, Any]] = None, idempotent: bool = True,
                      acting_user: Any = _CURRENT, timeout: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        """Yield one dict per NDJSON line. Retries happen only before the first byte arrives."""
        resp = self._send(method, path, json_body=json_body, params=params, idempotent=idempotent,
                          acting_user=acting_user, stream=True, timeout=timeout)
        try:
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    yield {"error": "unparseable_line", "raw": line[:200]}
                    continue
                if isinstance(item, dict):
                    yield item
        finally:
            resp.close()

    def download(self, path: str, dest: Path, *, params: Optional[Mapping[str, Any]] = None,
                 acting_user: Any = _CURRENT, max_bytes: int = 2 * 1024 ** 3,
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        """Stream a byte route to ``dest`` (written via a temp file, then renamed)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        resp = self._send("GET", path, params=params, acting_user=acting_user, stream=True, timeout=timeout)
        digest = hashlib.sha256()
        size = 0
        head = b""
        try:
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 16):
                    size += len(chunk)
                    if size > max_bytes:
                        raise LitKitError(f"download exceeds {max_bytes} bytes", method="GET", path=path)
                    if len(head) < 8:
                        head += chunk[: 8 - len(head)]
                    digest.update(chunk)
                    fh.write(chunk)
            tmp.replace(dest)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        finally:
            resp.close()
        return {"path": str(dest), "bytes": size, "sha256": digest.hexdigest(),
                "contentType": resp.headers.get("content-type", ""),
                "contentDisposition": resp.headers.get("content-disposition", ""), "head": head}

    def upload(self, path: str, file_path: Path, *, fields: Optional[Mapping[str, Any]] = None,
               filename: Optional[str] = None, content_type: Optional[str] = None, field_name: str = "file",
               ok_statuses: Sequence[int] = (), acting_user: Any = _CURRENT,
               timeout: Optional[float] = None) -> Tuple[int, Any]:
        """Multipart POST of one file plus string fields. Returns ``(status, body)``."""
        import mimetypes
        file_path = Path(file_path)
        name = filename or file_path.name
        ctype = content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
        data = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in (fields or {}).items() if v is not None}
        with open(file_path, "rb") as fh:
            files = {field_name: (name, fh, ctype)}
            resp = self._send("POST", path, files=files, data=data, ok_statuses=ok_statuses,
                              acting_user=acting_user, idempotent=False, timeout=timeout)
        return resp.status_code, _parse_body(resp)


_default_client: Optional[LitKitClient] = None
_pinned_client: Optional[LitKitClient] = None


def default_client() -> LitKitClient:
    """Process-wide client built from the environment (rebuilt if the configuration changes)."""
    global _default_client
    if _pinned_client is not None:
        return _pinned_client
    config = LitKitConfig.from_env()
    if _default_client is None or _default_client.config != config:
        if _default_client is not None:
            _default_client.close()
        _default_client = LitKitClient(config)
    return _default_client


def set_default_client(client: Optional[LitKitClient]) -> None:
    """Pin a client for every tool call (tests); ``None`` returns to the environment-built one."""
    global _pinned_client
    _pinned_client = client
