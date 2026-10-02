"""Reading one document's text and metadata from LitKit, and the fallback to it when the batch export is refused.

``POST /api/matters/{m}/export/text`` returns many documents' texts in one NDJSON stream. LitKit can refuse
that route to the agent host (``403 mfa_required`` on the Adobe host, 2026-10-02) while still answering the
per-document ``GET /api/documents/{id}/text``. :func:`texts_one_by_one` reads each document that way and returns
rows shaped like the batch export's, so ``litkit_export_text`` and ``litkit_jev`` keep working, and
:func:`batch_refusal_note` is what their results say about it.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from litco.litkit.client import REFUSAL_HOST, LitKitClient, LitKitError, LitKitPermissionError

TEXT_FALLBACK_MAX = 500  # documents per call when the batch export is refused and texts come one at a time
TEXT_FALLBACK_CONCURRENCY = 8


def date_of(row: Dict[str, Any]) -> Optional[str]:
    for key in ("date", "documentDate", "dateSent", "dateCreated", "dateModified"):
        if row.get(key):
            return str(row[key])
    return None


def metadata(client: LitKitClient, doc_id: str) -> Dict[str, Any]:
    body = client.get(f"/api/documents/{doc_id}")
    doc = body.get("doc") if isinstance(body, dict) else None
    return doc if isinstance(doc, dict) else {}


def text_from_payload(body: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    chunks = body.get("chunks") if isinstance(body.get("chunks"), list) else []
    if chunks:
        parts: List[str] = []
        last_page = None
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            page = ch.get("pageStart")
            if page is not None and page != last_page:
                parts.append(f"[page {page}]")
                last_page = page
            parts.append(str(ch.get("text") or ""))
        return "\n\n".join(parts), None
    text = str(body.get("extractedText") or "")
    capped = body.get("extractedTextCapped")
    trunc = ({"served": capped.get("servedLength"), "full": capped.get("fullLength")}
             if isinstance(capped, dict) else None)
    return text, trunc


def batch_text_refused(exc: LitKitError) -> bool:
    """A 403 on the batch export that the per-document route may still answer (not this host's credentials)."""
    return exc.status == 403 and exc.kind != REFUSAL_HOST


def batch_refusal_note(exc: LitKitError, mid: str, tool: str) -> Dict[str, Any]:
    """What a result says when LitKit refused the batch export and the texts came one document at a time."""
    message = exc.to_dict(tool)["message"]
    return {"batchRouteRefused": {"route": f"POST /api/matters/{mid}/export/text", "status": exc.status,
                                  "code": exc.code, "message": message},
            "textRoute": "per-document",
            "fallbackNote": (f"{message} So the texts came one document at a time through GET "
                             "/api/documents/{id}/text, which is slower. Report the counts as they are; "
                             "there is nothing for the person to do.")}


def texts_one_by_one(client: LitKitClient, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Each document's text and Bates metadata through the per-document routes, as rows shaped like the batch
    export's NDJSON. A refusal on the first document stops the run, since it would refuse every one."""
    def one(doc_id: str, first: bool = False) -> Dict[str, Any]:
        try:
            meta = metadata(client, doc_id)
            body = client.get(f"/api/documents/{doc_id}/text")
        except LitKitPermissionError:
            if first:
                raise
            return {"docId": doc_id, "error": "refused"}
        except LitKitError as exc:
            return {"docId": doc_id, "error": "not_found" if exc.status == 404 else f"HTTP {exc.status}"}
        text, trunc = text_from_payload(body if isinstance(body, dict) else {})
        return {"docId": doc_id, "batesStart": meta.get("batesStart"), "batesEnd": meta.get("batesEnd"),
                "custodian": meta.get("custodian"), "date": date_of(meta), "author": meta.get("author"),
                "subject": meta.get("subject"), "text": text, "truncated": bool(trunc),
                "fullLength": (trunc or {}).get("full"), "route": f"GET /api/documents/{doc_id}/text"}

    if not ids:
        return {}
    rows = {ids[0]: one(ids[0], True)}
    # Each task runs in a copy of the turn's context, so every request carries the acting lawyer's assertion.
    with ThreadPoolExecutor(max_workers=TEXT_FALLBACK_CONCURRENCY) as pool:
        futures = {d: pool.submit(contextvars.copy_context().run, one, d) for d in ids[1:]}
        rows.update({d: f.result() for d, f in futures.items()})
    return rows
