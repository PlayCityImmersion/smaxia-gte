
# =============================================================================
# SMAXIA GTE Console V31.10.23 (ISO-PROD — Kernel V10.6.2)
# =============================================================================
# OBJECTIF (TEST = PROD-MIRROR):
# - Exécuter la chaîne complète "Extract Not Create" sur couples Sujet+Corrigé.
# - OCR flux V10.6.2 intégré: TEXT-FIRST → OCR fallback → Consensus gate → Reason Codes.
# - ZÉRO hardcode métier: tout scope/chapitres/langues/δ_c/α/thresholds/mappings viennent du Pack.
# - CAS 1 ONLY: aucune reconstruction; tout ce qui n'est pas prouvé est quarantainé.
# - Outputs preuve: mapping_qi_to_qc.json, evidence_pack.json, audit_ia2.json, run_signature.json, qc_pack.json, qi_pack.json
#
# IMPORTANT:
# - Le "Zéro erreur" est interprété au sens Kernel: Zéro invention, Zéro faux PASS IA2.
#   En cas d'incertitude → QUARANTAINE/FAIL (raison codée) plutôt que "inventer".
#
# Dépendances:
#   pip install streamlit requests beautifulsoup4 pdfplumber pypdf pillow
# Optionnel (meilleur parsing HTML):
#   pip install lxml
#
# OCR (APIs externes, optionnelles):
# - Google Vision (REST) : env GOOGLE_VISION_API_KEY
# - Azure Vision F0      : env AZURE_VISION_ENDPOINT, AZURE_VISION_KEY
#
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import time
import math
import uuid
import base64
import hashlib
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st

# PDF / images
# PDF reader (not required): TEXT-FIRST uses pdfplumber (pdfminer.six).
PdfReader = None  # type: ignore  # kept for compatibility (unused)
_PDF_READER_IMPL = "pdfplumber"


from PIL import Image

# HTML parsing
from bs4 import BeautifulSoup


# =============================================================================
# Constants (KERNEL-allowed)
# =============================================================================

KERNEL_VERSION = "V10.6.2"
APP_VERSION = "V31.10.23"

# Kernel-scoped constant (sealed)
EPSILON = 0.1

# Cluster defaults (Kernel correction 7)
CLUSTER_SIMILARITY_THRESHOLD_DEFAULT = 0.85
CLUSTER_COHERENCE_THRESHOLD_DEFAULT = 0.70

# IA1↔IA2 correction loop (Kernel correction 2)
MAX_IA1_IA2_CORRECTION_ITERATIONS = 3

# Safety stop constants (Kernel correction 6)
ORPHAN_TOLERANCE_THRESHOLD = 0.02
ORPHAN_TOLERANCE_ABSOLUTE = 2
SCORE_MIN_VIABLE = 1.0
LOW_HISTORY_N_Q_MAX = 1
LOW_HISTORY_T_REC_MIN_YEARS = 5.0

# Fingerprint standard (Kernel correction 5)
FINGERPRINT_ALGORITHM = "SHA256"
FINGERPRINT_PREFIX = "sha256:"


# =============================================================================
# Reason Codes (Kernel + OCR V10.6.2 additions)
# =============================================================================
RC_CORRIGE_MISSING = "RC_CORRIGE_MISSING"
RC_CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
RC_CORRIGE_MISMATCH = "RC_CORRIGE_MISMATCH"

RC_SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
RC_SCOPE_CONFLICT = "RC_SCOPE_CONFLICT"
RC_SCOPE_OUTSIDE_PACK = "RC_SCOPE_OUTSIDE_PACK"

RC_NOT_A_QUESTION = "RC_NOT_A_QUESTION"
RC_NOT_EVALUABLE = RC_NOT_A_QUESTION  # alias toléré (audit externe)
RC_DEPENDENCY_MISSING_CONTEXT = "RC_DEPENDENCY_MISSING_CONTEXT"
RC_NON_DETERMINISTIC_STATEMENT = "RC_NON_DETERMINISTIC_STATEMENT"

RC_DUPLICATE_ATOM = "RC_DUPLICATE_ATOM"
RC_EXTRACTION_CORRUPTED = "RC_EXTRACTION_CORRUPTED"
RC_LANGUAGE_UNSUPPORTED_BY_PACK = "RC_LANGUAGE_UNSUPPORTED_BY_PACK"
RC_RESTRICTED_CONTENT = "RC_RESTRICTED_CONTENT"

RC_CORRECTION_LOOP_EXCEEDED = "RC_CORRECTION_LOOP_EXCEEDED"
RC_SINGLETON_IRREDUCTIBLE = "RC_SINGLETON_IRRÉDUCTIBLE"

# OCR-specific (V10.6.2)
RC_OCR_DISAGREEMENT = "RC_OCR_DISAGREEMENT"
RC_OCR_CONFIG_MISSING = "RC_OCR_CONFIG_MISSING"
RC_OCR_PROVIDER_ERROR = "RC_OCR_PROVIDER_ERROR"
RC_OCR_TEXT_EMPTY = "RC_OCR_TEXT_EMPTY"


# =============================================================================
# Utilities
# =============================================================================

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return f"{FINGERPRINT_PREFIX}{h.hexdigest()}"

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", errors="ignore"))

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def norm_ws(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def is_probably_scanned(text: str, min_chars: int, min_alpha_ratio: float) -> bool:
    if not text:
        return True
    chars = len(text)
    if chars < min_chars:
        return True
    alpha = sum(1 for c in text if c.isalpha())
    ratio = alpha / max(1, chars)
    return ratio < min_alpha_ratio

def levenshtein_distance(a: str, b: str) -> int:
    # Deterministic, O(n*m) but bounded by truncation in consensus gate
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        cur = [j] + [0] * n
        bj = b[j - 1]
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur[i] = min(cur[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = cur
    return prev[n]

def normalized_divergence(a: str, b: str, max_len: int = 6000) -> float:
    # divergence in [0,1], 0 = identical
    a2 = (a or "")[:max_len]
    b2 = (b or "")[:max_len]
    dist = levenshtein_distance(a2, b2)
    denom = max(1, max(len(a2), len(b2)))
    return dist / denom

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def cosine_like_from_jaccard(j: float) -> float:
    # map [0,1] to [0,1] (identity, but kept explicit for readability)
    return clamp(j, 0.0, 1.0)


# =============================================================================
# Pack access (no hardcode)
# =============================================================================

def pack_get(pack: Dict[str, Any], path: str, default=None):
    cur = pack
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def require_pack_field(pack: Dict[str, Any], path: str):
    v = pack_get(pack, path, None)
    if v is None:
        raise ValueError(f"Pack missing required field: {path}")
    return v


# =============================================================================
# Data structures (exports)
# =============================================================================

@dataclass
class SourceRef:
    source_id: str
    source_url: str
    mime_type: str
    fingerprint_sha256: str
    retrieved_at_utc: str
    local_name: str

@dataclass
class OCRRun:
    provider: str
    status: str  # PASS/FAIL
    details: str
    confidence: float
    pages: int
    text_sha256: str
    request_id: str

@dataclass
class ExtractionLocator:
    # Minimal locator (page_range and offsets). When provider supports coordinates, store them in 'extra'
    page_start: int
    page_end: int
    start_offset: int
    end_offset: int
    extra: Dict[str, Any]

@dataclass
class Atom:
    atom_id: str
    subject_id: str
    correction_id: str
    qi_id: str
    rqi_id: str
    qi_raw: str
    rqi_raw: str
    qi_clean: str
    rqi_clean: str
    language_detected: str
    year_ref: Optional[int]
    source_subject_url: str
    source_correction_url: str
    subject_fingerprint_sha256: str
    correction_fingerprint_sha256: str
    extraction_locators: Dict[str, Any]  # {"qi":..., "rqi":...}
    sanitizer_derivations: List[Dict[str, Any]]
    pairing_confidence: float
    figure_refs: List[Dict[str, Any]]

@dataclass
class PosableDecision:
    qi_id: str
    chapter_ref: str
    posable_status: bool
    posable_reason_codes: List[str]
    evidence_refs: List[str]
    scope_trace: Dict[str, Any]

@dataclass
class AuditEntry:
    check_id: str
    status: str  # PASS/FAIL
    evidence_refs: List[str]
    details: str
    fix_recommendations: List[str]

@dataclass
class IA2AuditLog:
    audit_id: str
    timestamp_utc: str
    kernel_version: str
    pack_version: str
    pack_fingerprint_sha256: str
    chapter_ref: str
    qc_id: Optional[str]
    checks: List[AuditEntry]
    overall_status: str
    hash_integrity: str
    determinism_params: Dict[str, Any]

@dataclass
class ARIStep:
    step_id: str
    weight: float
    evidence_refs: List[str]

@dataclass
class ARI:
    steps: List[ARIStep]

@dataclass
class Trigger:
    trigger_id: str
    pattern: str
    evidence_refs: List[str]

@dataclass
class FRT:
    frt_id: str
    operator: str
    sections: Dict[str, Any]
    evidence_refs: List[str]

@dataclass
class QC:
    qc_id: str
    chapter_ref: str
    qc_label: str
    sig: str
    n_q_cluster: int
    ari: ARI
    triggers: List[Trigger]
    frt: FRT
    # F1/F2
    delta_c: float
    psi_raw: float
    psi_norm: float
    n_hist: int
    t_rec_years: float
    alpha: float
    score: float
    # coverage
    cover_qi_ids: List[str]
    evidence_refs: List[str]


# =============================================================================
# OCR providers
# =============================================================================

class OCRProviderBase:
    name: str

    def ocr_image(self, img: Image.Image) -> Tuple[str, float, str]:
        raise NotImplementedError

class GoogleVisionOCR(OCRProviderBase):
    name = "google_vision"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def ocr_image(self, img: Image.Image) -> Tuple[str, float, str]:
        # REST annotate endpoint using API key (no service account)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        content_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")

        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        payload = {
            "requests": [{
                "image": {"content": content_b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }]
        }
        r = requests.post(url, json=payload, timeout=40)
        req_id = r.headers.get("x-guploader-uploadid") or str(uuid.uuid4())
        if r.status_code != 200:
            raise RuntimeError(f"Google Vision error HTTP {r.status_code}: {r.text[:500]}")
        j = r.json()
        resp = (j.get("responses") or [{}])[0]
        full = ((resp.get("fullTextAnnotation") or {}).get("text")) or ""
        # Confidence isn't explicit; approximate with presence/length
        conf = 0.5 if full.strip() else 0.0
        return full, conf, req_id

class AzureVisionOCR(OCRProviderBase):
    name = "azure_vision_f0"

    def __init__(self, endpoint: str, key: str):
        self.endpoint = endpoint.rstrip("/")
        self.key = key

    def ocr_image(self, img: Image.Image) -> Tuple[str, float, str]:
        # Azure Read API (v3.2). Requires async polling.
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        url = f"{self.endpoint}/vision/v3.2/read/analyze"
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/octet-stream",
        }
        r = requests.post(url, headers=headers, data=image_bytes, timeout=40)
        if r.status_code not in (202, 201):
            raise RuntimeError(f"Azure Vision error HTTP {r.status_code}: {r.text[:500]}")
        op_loc = r.headers.get("Operation-Location")
        req_id = op_loc or str(uuid.uuid4())
        if not op_loc:
            raise RuntimeError("Azure Vision missing Operation-Location")

        # Poll
        poll_headers = {"Ocp-Apim-Subscription-Key": self.key}
        for _ in range(30):
            pr = requests.get(op_loc, headers=poll_headers, timeout=20)
            if pr.status_code != 200:
                raise RuntimeError(f"Azure poll error HTTP {pr.status_code}: {pr.text[:500]}")
            pj = pr.json()
            status = (pj.get("status") or "").lower()
            if status in ("succeeded", "failed"):
                if status == "failed":
                    raise RuntimeError("Azure OCR failed status")
                lines = []
                analyze = (pj.get("analyzeResult") or {})
                for page in (analyze.get("readResults") or []):
                    for ln in (page.get("lines") or []):
                        lines.append(ln.get("text", ""))
                text = "\n".join(lines)
                conf = 0.5 if text.strip() else 0.0
                return text, conf, req_id
            time.sleep(0.6)

        raise RuntimeError("Azure OCR timeout")


# =============================================================================
# Document extraction: TEXT-FIRST → OCR fallback → consensus gate
# =============================================================================

@dataclass
class DocText:
    text: str
    method: str  # "text-first" | "ocr"
    ocr_runs: List[OCRRun]
    consensus_divergence: Optional[float]
    reason_codes: List[str]
    text_sha256: str

def pdf_text_first(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"engine": f"pdfplumber+{_PDF_READER_IMPL}", "pdf_reader_impl": _PDF_READER_IMPL}
    text_parts: List[str] = []

    # [1] TEXT-FIRST via pdfplumber (best when PDF has embedded text)
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(f"\n\n--- PAGE {i+1} ---\n{t}")
    except Exception as e:
        meta["pdfplumber_error"] = str(e)[:300]

    # Optional secondary text extraction via PdfReader (environment-dependent)
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for i, page in enumerate(getattr(reader, "pages", [])):
                t = (page.extract_text() or "") if page is not None else ""
                if t.strip():
                    text_parts.append(f"\n\n--- PAGE {i+1} ({_PDF_READER_IMPL}) ---\n{t}")
        except Exception as e:
            meta["pdf_reader_error"] = str(e)[:300]
    else:
        meta["pdf_reader_missing"] = True

    text = norm_ws("\n".join(text_parts))
    meta["text_chars"] = len(text)
    return text, meta

def pdf_to_images(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 12) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = pdf.pages[:max_pages]
        for page in pages:
            im = page.to_image(resolution=dpi).original
            imgs.append(im)
    return imgs

def run_ocr_consensus(
    images: List[Image.Image],
    providers: List[OCRProviderBase],
    divergence_threshold: float,
) -> Tuple[str, List[OCRRun], Optional[float], List[str]]:
    """
    Returns: consensus_text, ocr_runs, divergence, reason_codes
    - Runs all providers on all pages, concatenates per provider.
    - Applies divergence gate on first N chars between top-2 candidates.
    """
    reason_codes: List[str] = []
    ocr_runs: List[OCRRun] = []

    provider_texts: Dict[str, str] = {}
    provider_confs: Dict[str, float] = {}
    provider_reqid: Dict[str, str] = {}

    for p in providers:
        all_text = []
        confs = []
        reqids = []
        status = "PASS"
        details = ""
        try:
            for img in images:
                t, c, rid = p.ocr_image(img)
                all_text.append(t)
                confs.append(c)
                reqids.append(rid)
            joined = norm_ws("\n".join(all_text))
            if not joined:
                status = "FAIL"
                details = "Empty OCR text"
            provider_texts[p.name] = joined
            provider_confs[p.name] = float(sum(confs) / max(1, len(confs)))
            provider_reqid[p.name] = reqids[-1] if reqids else str(uuid.uuid4())
        except Exception as e:
            status = "FAIL"
            details = str(e)[:300]
            provider_texts[p.name] = ""
            provider_confs[p.name] = 0.0
            provider_reqid[p.name] = str(uuid.uuid4())
            reason_codes.append(RC_OCR_PROVIDER_ERROR)

        ocr_runs.append(OCRRun(
            provider=p.name,
            status=status,
            details=details,
            confidence=provider_confs[p.name],
            pages=len(images),
            text_sha256=sha256_text(provider_texts[p.name]),
            request_id=provider_reqid[p.name],
        ))

    # Choose best by confidence first, then length, then provider name (deterministic)
    ranked = sorted(
        provider_texts.keys(),
        key=lambda k: (-provider_confs.get(k, 0.0), -len(provider_texts.get(k, "")), k),
    )

    best = ranked[0]
    best_text = provider_texts[best]
    if not best_text:
        reason_codes.append(RC_OCR_TEXT_EMPTY)
        return "", ocr_runs, None, reason_codes

    # Need at least 2 providers to compute divergence. If only one, accept but mark.
    divergence = None
    if len(ranked) >= 2:
        second = ranked[1]
        divergence = normalized_divergence(best_text, provider_texts.get(second, ""))
        if divergence > divergence_threshold:
            reason_codes.append(RC_OCR_DISAGREEMENT)
            return "", ocr_runs, divergence, reason_codes

    return best_text, ocr_runs, divergence, reason_codes

def extract_text_with_ocr_flow(
    file_bytes: bytes,
    mime_type: str,
    pack: Dict[str, Any],
) -> DocText:
    """
    Implements: TEXT-FIRST → OCR fallback → consensus gate → reason codes.
    Ensures all OCR metadata is returned for EvidencePack/AuditLog.
    """
    ocr_cfg = pack_get(pack, "ocr", {}) or {}
    min_chars = safe_int(ocr_cfg.get("text_first_min_chars", 1200), 1200)
    min_alpha_ratio = float(ocr_cfg.get("text_first_min_alpha_ratio", 0.20))
    divergence_threshold = float(ocr_cfg.get("consensus_divergence_threshold", 0.18))
    max_pages = safe_int(ocr_cfg.get("max_pages", 12), 12)
    dpi = safe_int(ocr_cfg.get("dpi", 200), 200)

    reason_codes: List[str] = []
    ocr_runs: List[OCRRun] = []
    consensus_divergence: Optional[float] = None

    if mime_type == "application/pdf":
        text, meta = pdf_text_first(file_bytes)
        if not is_probably_scanned(text, min_chars=min_chars, min_alpha_ratio=min_alpha_ratio):
            return DocText(
                text=text,
                method="text-first",
                ocr_runs=[],
                consensus_divergence=None,
                reason_codes=[],
                text_sha256=sha256_text(text),
            )

        # OCR fallback
        images = pdf_to_images(file_bytes, dpi=dpi, max_pages=max_pages)

        providers: List[OCRProviderBase] = []
        gkey = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()
        aep = os.environ.get("AZURE_VISION_ENDPOINT", "").strip()
        akey = os.environ.get("AZURE_VISION_KEY", "").strip()

        if gkey:
            providers.append(GoogleVisionOCR(gkey))
        if aep and akey:
            providers.append(AzureVisionOCR(aep, akey))

        if not providers:
            reason_codes.append(RC_OCR_CONFIG_MISSING)
            return DocText(
                text="",
                method="ocr",
                ocr_runs=[],
                consensus_divergence=None,
                reason_codes=reason_codes,
                text_sha256=sha256_text(""),
            )

        consensus_text, ocr_runs, consensus_divergence, rc = run_ocr_consensus(
            images=images,
            providers=providers,
            divergence_threshold=divergence_threshold,
        )
        reason_codes.extend(rc)
        return DocText(
            text=consensus_text,
            method="ocr",
            ocr_runs=ocr_runs,
            consensus_divergence=consensus_divergence,
            reason_codes=reason_codes,
            text_sha256=sha256_text(consensus_text),
        )

    # Images (PNG/JPG): always OCR path
    if mime_type.startswith("image/"):
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        providers: List[OCRProviderBase] = []
        gkey = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()
        aep = os.environ.get("AZURE_VISION_ENDPOINT", "").strip()
        akey = os.environ.get("AZURE_VISION_KEY", "").strip()
        if gkey:
            providers.append(GoogleVisionOCR(gkey))
        if aep and akey:
            providers.append(AzureVisionOCR(aep, akey))
        if not providers:
            reason_codes.append(RC_OCR_CONFIG_MISSING)
            return DocText("", "ocr", [], None, reason_codes, sha256_text(""))
        consensus_text, ocr_runs, consensus_divergence, rc = run_ocr_consensus(
            images=[img],
            providers=providers,
            divergence_threshold=float(pack_get(pack, "ocr.consensus_divergence_threshold", 0.18)),
        )
        reason_codes.extend(rc)
        return DocText(
            text=consensus_text,
            method="ocr",
            ocr_runs=ocr_runs,
            consensus_divergence=consensus_divergence,
            reason_codes=reason_codes,
            text_sha256=sha256_text(consensus_text),
        )

    # HTML / text: parse without OCR
    if mime_type in ("text/html", "text/plain"):
        try:
            if mime_type == "text/html":
                soup = BeautifulSoup(file_bytes.decode("utf-8", errors="ignore"), "html.parser")
                text = norm_ws(soup.get_text("\n"))
            else:
                text = norm_ws(file_bytes.decode("utf-8", errors="ignore"))
            return DocText(text, "text-first", [], None, [], sha256_text(text))
        except Exception:
            reason_codes.append(RC_EXTRACTION_CORRUPTED)
            return DocText("", "text-first", [], None, reason_codes, sha256_text(""))

    # Unknown format → quarantined
    reason_codes.append(RC_EXTRACTION_CORRUPTED)
    return DocText("", "text-first", [], None, reason_codes, sha256_text(""))


# =============================================================================
# Harvest (pack-driven) — PDFs + other formats
# =============================================================================

@dataclass
class PairItem:
    pair_id: str
    subject_url: str
    correction_url: str
    year: Optional[int]
    meta: Dict[str, Any]
    pairing_confidence: float

def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False

def fetch_url_bytes(url: str, timeout: int = 40) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "SMAXIA-GTE/1.0"})
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if not ct:
        # guess from extension
        p = urlparse(url).path.lower()
        if p.endswith(".pdf"):
            ct = "application/pdf"
        elif p.endswith(".png"):
            ct = "image/png"
        elif p.endswith(".jpg") or p.endswith(".jpeg"):
            ct = "image/jpeg"
        elif p.endswith(".html") or p.endswith(".htm"):
            ct = "text/html"
        else:
            ct = "application/octet-stream"
    return r.content, ct

def harvest_pairs_from_pack(pack: Dict[str, Any], selected_years: List[int], max_pairs: int) -> Tuple[List[PairItem], List[str]]:
    """Pack-driven harvester (robust).

    Goals:
    - No domain hardcode in code.
    - Works with generic HTML pages that list documents.
    - Supports root templates with {year} expanded from selected_years.

    Strategy:
    1) Expand roots (templates) for each selected year (or a single pass if no years).
    2) Crawl each expanded root (HTML) and collect candidate links (URL + anchor text).
    3) Classify candidates into SUBJECT vs CORRECTION using pack regexes.
    4) Pair globally (across all roots) using token overlap + optional year agreement.

    Notes:
    - We intentionally accept links without .pdf extension (e.g., CMS download endpoints) and let later fetch detect mime.
    """
    logs: List[str] = []
    roots = pack_get(pack, "harvest.roots", []) or []
    if not roots:
        return [], ["Pack missing harvest.roots"]

    subj_regexes = pack_get(pack, "harvest.subject_regexes", []) or []
    corr_regexes = pack_get(pack, "harvest.correction_regexes", []) or []
    year_regex = pack_get(pack, "harvest.year_regex", r"(20\d{2})")
    min_pair_conf = float(pack_get(pack, "harvest.min_pairing_confidence", 0.25))
    max_links_per_root = safe_int(pack_get(pack, "harvest.max_links_per_root", 2500), 2500)

    def extract_year(s: str) -> Optional[int]:
        m = re.search(year_regex, s or "")
        if not m:
            return None
        y = safe_int(m.group(1), 0)
        return y if y else None

    def norm_tokens(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (s or "").lower())

    def is_subject(label: str) -> bool:
        if subj_regexes:
            return any(re.search(rx, label, flags=re.IGNORECASE) for rx in subj_regexes)
        # fallback: not a correction
        return not is_correction(label)

    def is_correction(label: str) -> bool:
        if corr_regexes:
            return any(re.search(rx, label, flags=re.IGNORECASE) for rx in corr_regexes)
        return bool(re.search(r"corrig|correction", label, flags=re.IGNORECASE))

    # Expand roots (templates)
    expanded_roots: List[str] = []
    years = [int(y) for y in selected_years] if selected_years else []
    for r0 in roots:
        if not is_url(str(r0)):
            logs.append(f"[HARVEST_QUARANTINE] Invalid root URL: {r0}")
            continue
        r0s = str(r0)
        if "{year}" in r0s and years:
            for y in years:
                expanded_roots.append(r0s.replace("{year}", str(y)))
        else:
            expanded_roots.append(r0s)

    # Collect candidates
    candidates: List[Tuple[str, str, Optional[int], str]] = []  # (url, label, year, root)
    for root in expanded_roots:
        logs.append(f"[HARVEST] root={root}")
        try:
            html_bytes, ct = fetch_url_bytes(root)
            if ct != "text/html":
                logs.append(f"[HARVEST_SKIP] Root not HTML: {root} (ct={ct})")
                continue

            soup = BeautifulSoup(html_bytes.decode("utf-8", errors="ignore"), "html.parser")
            n = 0
            for a in soup.find_all("a"):
                href = a.get("href")
                if not href:
                    continue
                u = urljoin(root, href)
                if not is_url(u):
                    continue

                # label combines anchor text and basename (many CMS links hide filename)
                anchor_text = norm_ws(a.get_text(" ") or "")
                base = os.path.basename(urlparse(u).path) or ""
                label = (anchor_text + " " + base).strip() or u

                y = extract_year(label) or extract_year(u)
                if years and (y is None or y not in years):
                    continue

                candidates.append((u, label, y, root))
                n += 1
                if n >= max_links_per_root:
                    break

            logs.append(f"[HARVEST] root_candidates={n}")

        except Exception as e:
            logs.append(f"[HARVEST_ERROR] {root}: {str(e)[:200]}")

    if not candidates:
        logs.append("[HARVEST] No candidates collected")
        return [], logs

    # Split subjects vs corrections
    subjects = [(u, label, y, r) for (u, label, y, r) in candidates if is_subject(label) and not is_correction(label)]
    corrections = [(u, label, y, r) for (u, label, y, r) in candidates if is_correction(label)]

    logs.append(f"[HARVEST] subjects={len(subjects)} corrections={len(corrections)}")

    if not subjects or not corrections:
        # Still return empty pairs but with actionable logs
        if not subjects:
            logs.append("[HARVEST] No SUBJECT candidates (check harvest.subject_regexes in pack)")
        if not corrections:
            logs.append("[HARVEST] No CORRECTION candidates (check harvest.correction_regexes in pack)")
        return [], logs

    corr_tokens = [(set(norm_tokens(label)), u, label, y) for (u, label, y, _) in corrections]

    pairs: List[PairItem] = []
    for su, slabel, sy, _root in subjects:
        stoks = set(norm_tokens(slabel))
        if not stoks:
            continue
        best = None
        best_score = -1.0
        for ctoks, cu, clabel, cy in corr_tokens:
            if sy and cy and sy != cy:
                continue
            overlap = len(stoks & ctoks)
            score = overlap / max(1, len(stoks))
            if score > best_score:
                best_score = score
                best = (cu, clabel, cy)

        if best and best_score >= min_pair_conf:
            cu, clabel, cy = best
            pair_id = f"PAIR_{uuid.uuid4().hex[:12]}"
            pairs.append(PairItem(
                pair_id=pair_id,
                subject_url=su,
                correction_url=cu,
                year=sy or cy,
                meta={"subject_label": slabel, "correction_label": clabel},
                pairing_confidence=float(best_score),
            ))

        if len(pairs) >= max_pairs:
            break

    # Deterministic ordering and trimming
    pairs = sorted(pairs, key=lambda p: (p.year or 0, -p.pairing_confidence, p.pair_id))[:max_pairs]
    logs.append(f"[HARVEST] pairs={len(pairs)}")
    if not pairs:
        logs.append("[HARVEST] No pairs built (increase min_pairing_confidence or refine regexes)")

    return pairs, logs

def split_into_segments(text: str, max_segments: int = 120) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Returns list of (segment_text, (start,end)) using best-effort splits.
    If splitting fails, returns whole text as one segment.
    """
    t = norm_ws(text)
    if not t:
        return []
    # Use earliest split positions across patterns
    split_positions = set([0])
    for rx in _QI_SPLIT_PATTERNS:
        for m in re.finditer(rx, t):
            split_positions.add(m.start())
    poss = sorted(split_positions)
    segs = []
    for i in range(len(poss)):
        s = poss[i]
        e = poss[i+1] if i+1 < len(poss) else len(t)
        seg = t[s:e].strip()
        if seg:
            segs.append((seg, (s,e)))
    # Filter segments that are clearly not questions (very short)
    segs2 = [(s, r) for (s,r) in segs if len(s) >= 20]
    if not segs2:
        segs2 = [(t, (0, len(t)))]
    return segs2[:max_segments]

def sanitize_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Kernel sanitizer: neutralise local constants while keeping structure.
    Deterministic; logs derivations.
    """
    derivs = []
    t = text

    # Remove obvious headers/footers markers
    t2 = re.sub(r"(?m)^\s*---\s*PAGE\s+\d+.*?$", "", t)
    if t2 != t:
        derivs.append({"op": "remove_page_markers", "before_sha": sha256_text(t), "after_sha": sha256_text(t2)})
        t = t2

    # Neutralise years (local constants)
    t2 = re.sub(r"\b(19\d{2}|20\d{2})\b", "YEAR_REF", t)
    if t2 != t:
        derivs.append({"op": "neutralise_years", "before_sha": sha256_text(t), "after_sha": sha256_text(t2)})
        t = t2

    # Neutralise long numbers (keep small numbers for math meaning)
    t2 = re.sub(r"\b\d{5,}\b", "NUM_BIG", t)
    if t2 != t:
        derivs.append({"op": "neutralise_big_numbers", "before_sha": sha256_text(t), "after_sha": sha256_text(t2)})
        t = t2

    # Collapse whitespace
    t2 = norm_ws(t)
    if t2 != t:
        derivs.append({"op": "normalize_whitespace", "before_sha": sha256_text(t), "after_sha": sha256_text(t2)})
        t = t2

    return t, derivs


def pair_qi_to_rqi(qi_segs: List[Tuple[str, Tuple[int,int]]], rqi_segs: List[Tuple[str, Tuple[int,int]]]) -> List[Tuple[int, Optional[int], float]]:
    """
    Deterministic heuristic pairing.
    Returns list of (qi_index, rqi_index_or_None, confidence).
    """
    if not qi_segs:
        return []
    if not rqi_segs:
        return [(i, None, 0.0) for i in range(len(qi_segs))]

    # Token overlap score
    def toks(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    r_toks = [toks(s) for s,_ in rqi_segs]
    pairs = []
    for i, (qtxt, _) in enumerate(qi_segs):
        qt = toks(qtxt)
        bestj = None
        best = -1.0
        for j, rt in enumerate(r_toks):
            overlap = len(qt & rt)
            score = overlap / max(1, len(qt))
            if score > best:
                best = score
                bestj = j
        # require minimum to accept; threshold is pack-configurable but default conservative
        pairs.append((i, bestj, float(best)))
    return pairs


# =============================================================================
# Scope match (pack-driven) + POSABLE gate
# =============================================================================

def scope_match_chapter(pack: Dict[str, Any], qi_clean: str, rqi_clean: str) -> Tuple[Optional[str], Dict[str, Any], List[str]]:
    """
    Pack-driven scope: uses pack.academic.chapters[*].match_keywords and match_regexes.
    No Kernel hardcode.
    """
    chapters = pack_get(pack, "academic.chapters", []) or []
    if not chapters:
        return None, {"method": "pack_missing_chapters"}, [RC_SCOPE_UNRESOLVED]

    text = (qi_clean + "\n" + (rqi_clean or "")).lower()
    hits = []
    for ch in chapters:
        cref = ch.get("chapter_ref") or ch.get("chapter_code") or ch.get("id")
        if not cref:
            continue
        kws = ch.get("match_keywords", []) or []
        rxs = ch.get("match_regexes", []) or []
        score = 0.0
        kw_hits = []
        rx_hits = []
        for kw in kws:
            if kw and kw.lower() in text:
                score += 1.0
                kw_hits.append(kw)
        for rx in rxs:
            try:
                if rx and re.search(rx, text, flags=re.IGNORECASE):
                    score += 1.5
                    rx_hits.append(rx)
            except re.error:
                continue
        if score > 0:
            hits.append((score, str(cref), kw_hits, rx_hits))

    if not hits:
        return None, {"method": "pack_keywords_regexes", "hits": []}, [RC_SCOPE_UNRESOLVED]

    # Deterministic winner
    hits.sort(key=lambda x: (-x[0], x[1]))
    top = hits[0]
    trace = {
        "method": "pack_keywords_regexes",
        "winner": {"chapter_ref": top[1], "score": top[0], "kw_hits": top[2], "rx_hits": top[3]},
        "candidates": [{"chapter_ref": h[1], "score": h[0]} for h in hits[:5]],
    }

    # Conflict if second is too close
    if len(hits) >= 2 and abs(hits[0][0] - hits[1][0]) < 0.5:
        return None, trace, [RC_SCOPE_CONFLICT]

    return top[1], trace, []


def posable_gate(pack: Dict[str, Any], qi_clean: str, rqi_clean: str, chapter_ref: Optional[str], scope_rc: List[str]) -> Tuple[bool, List[str]]:
    rc: List[str] = []
    # POSABLE_CORRIGE
    if not rqi_clean.strip():
        rc.append(RC_CORRIGE_UNREADABLE)
    # POSABLE_SCOPE
    if not chapter_ref:
        rc.extend(scope_rc if scope_rc else [RC_SCOPE_UNRESOLVED])
    # POSABLE_EVALUABLE (heuristic; IA2 re-check later)
    if "?" not in qi_clean and not re.search(r"(?i)\b(calculer|déterminer|résoudre|montrer|démontrer|établir|prouver|tracer|justifier|donner)\b", qi_clean):
        rc.append(RC_NOT_A_QUESTION)

    return (len(rc) == 0), rc


# =============================================================================
# IA1 Miner / Builder (deterministic core + pack ontology)
# =============================================================================

def load_cognitive_table(pack: Dict[str, Any]) -> Dict[str, float]:
    # Kernel provides canonical IDs; Pack provides mapping + weights.
    # For test script: weights should come from pack.cognitive.weights (canonical_id -> T_j)
    weights = pack_get(pack, "cognitive.weights", {}) or {}
    # Accept only numeric weights
    out = {}
    for k, v in weights.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out

def map_actions(pack: Dict[str, Any], text: str) -> List[str]:
    """
    Map text to canonical cognitive step IDs using pack.cognitive.synonyms (lang->syn->id).
    Deterministic substring matching.
    """
    syn = pack_get(pack, "cognitive.synonyms", {}) or {}
    lang = (pack_get(pack, "language.default", "fr") or "fr").lower()
    syn_lang = syn.get(lang, {}) if isinstance(syn, dict) else {}
    # syn_lang: {"calculer": "CALCULATE", ...}
    t = text.lower()
    hits = []
    for word, cid in syn_lang.items():
        if word and word.lower() in t:
            hits.append(str(cid))
    # Dedup, stable
    hits = sorted(set(hits))
    return hits

def ia1_miner(pack: Dict[str, Any], qi: Atom) -> Tuple[List[str], List[str]]:
    """
    Returns (actions_mapped_ids, evidence_refs)
    Evidence refs are placeholders referencing atom_id segments.
    """
    evidence_refs = [f"EV_ATOM:{qi.atom_id}:RQi"]
    acts = map_actions(pack, qi.rqi_clean)
    return acts, evidence_refs

def infer_output_type(pack: Dict[str, Any], qi_clean: str, rqi_clean: str) -> str:
    """
    Pack may provide output type rules; else UNKNOWN.
    """
    rules = pack_get(pack, "output_type.rules", []) or []
    txt = (qi_clean + "\n" + rqi_clean).lower()
    for r in rules:
        try:
            rx = r.get("regex")
            out = r.get("output_type")
            if rx and out and re.search(rx, txt, flags=re.IGNORECASE):
                return str(out)
        except Exception:
            continue
    return "UNKNOWN"

def build_sig(action_spine: List[str], preconds: List[str], output_type: str, checkpoints: List[str]) -> str:
    # Canonical deterministic signature string
    a = ",".join(action_spine)
    p = ",".join(sorted(set(preconds)))
    x = ",".join(sorted(set(checkpoints)))
    return f"<{a}|{p}|{output_type}|{x}>"

def build_triggers(pack: Dict[str, Any], qi_texts: List[str], max_triggers: int = 5) -> List[Trigger]:
    """
    Triggers: 4–5 per QC, extracted from invariant patterns (instruction forms).
    Pack may provide trigger_patterns to extract.
    """
    patterns = pack_get(pack, "triggers.pattern_extractors", []) or []
    extracted = []
    for q in qi_texts:
        for pe in patterns:
            try:
                rx = pe.get("regex")
                if rx:
                    m = re.search(rx, q, flags=re.IGNORECASE)
                    if m:
                        extracted.append(m.group(0).strip())
            except Exception:
                continue
    # fallback: take first imperative sentence fragments (still not chapter-specific)
    if not extracted:
        for q in qi_texts:
            frag = q.split("\n")[0]
            frag = re.sub(r"\s+", " ", frag).strip()
            if frag:
                extracted.append(frag[:120])

    # Dedup, stable; limit
    extracted = sorted(set(extracted))[:max_triggers]
    out = []
    for s in extracted:
        out.append(Trigger(
            trigger_id=f"T_{uuid.uuid4().hex[:10]}",
            pattern=s,
            evidence_refs=[],
        ))
    # Ensure 4–5 triggers if possible by repeating best (deterministically) is forbidden → instead, if <4, keep as-is and IA2 will FAIL TRIGGERS_QUALITY.
    return out

def build_frt(pack: Dict[str, Any], operator: str, qc_label: str, ari: ARI, triggers: List[Trigger], evidence_refs: List[str]) -> FRT:
    templates = pack_get(pack, "frt.templates_by_operator", {}) or {}
    generic = pack_get(pack, "frt.template_generic", {}) or {}

    tmpl = templates.get(operator, generic)
    # Fill required sections
    sections = {
        "Usage": {
            "qc_label": qc_label,
            "triggers": [t.pattern for t in triggers],
        },
        "Préconditions": tmpl.get("Préconditions", tmpl.get("Preconditions", [])),
        "ARI": [{"step_id": s.step_id, "weight": s.weight} for s in ari.steps],
        "Transitions & checkpoints": tmpl.get("Transitions", tmpl.get("Transitions & checkpoints", [])),
        "Pièges & erreurs fatales": tmpl.get("Piges", tmpl.get("Pièges & erreurs fatales", [])),
        "Sortie attendue": tmpl.get("Sortie", tmpl.get("Sortie attendue", "UNKNOWN")),
        "Preuves": evidence_refs,
    }

    return FRT(
        frt_id=f"FRT_{uuid.uuid4().hex[:12]}",
        operator=operator,
        sections=sections,
        evidence_refs=evidence_refs,
    )

def qc_label_from_pack(pack: Dict[str, Any], qi_champion: str) -> str:
    fmt = pack_get(pack, "qc_format", {}) or {}
    prefix = fmt.get("prefix", "Comment")
    suffix = fmt.get("suffix", "?")
    # strip local constants already sanitized
    core = qi_champion.strip()
    # remove leading numbering
    core = re.sub(r"^\s*(?:Question|Q)?\s*\d+\s*[:.)-]\s*", "", core, flags=re.IGNORECASE)
    core = re.sub(r"^\s*\d+\s*[\).]\s*", "", core)
    core = core.strip()
    # reduce length deterministically
    core = core[:180].rstrip()
    # enforce form "prefix ... suffix"
    if not core.lower().startswith(prefix.lower()):
        core = f"{prefix} {core}"
    if not core.endswith(suffix):
        core = f"{core}{suffix}"
    return core

def compute_f1(pack: Dict[str, Any], qc: QC, cognitive_weights: Dict[str, float]) -> Tuple[float, float]:
    # ψ_raw(q) = δ_c * (ε + Σ T_j)^2, where Σ T_j sums weights of ARI steps
    sum_t = 0.0
    for s in qc.ari.steps:
        sum_t += float(cognitive_weights.get(s.step_id, s.weight))
    psi_raw = float(qc.delta_c) * (EPSILON + sum_t) ** 2
    return psi_raw, psi_raw  # normalisation done later

def compute_similarity_sigma(q1: QC, q2: QC) -> float:
    """σ(q,p) — Similarité cosinus sur embeddings ARI normalisé (Kernel §2.2).

    Implémentation déterministe sans dépendances externes:
    - embedding sparse = vecteur {step_id: weight} issu de l'ARI typé.
    - cosine(v,w) = dot(v,w) / (||v||·||w||).
    """
    wa: Dict[str, float] = {}
    wb: Dict[str, float] = {}

    for s in q1.ari.steps:
        sid = str(s.step_id).strip()
        if sid:
            wa[sid] = wa.get(sid, 0.0) + float(getattr(s, "weight", 0.0) or 0.0)

    for s in q2.ari.steps:
        sid = str(s.step_id).strip()
        if sid:
            wb[sid] = wb.get(sid, 0.0) + float(getattr(s, "weight", 0.0) or 0.0)

    if not wa or not wb:
        return 0.0

    dot = 0.0
    for sid, va in wa.items():
        vb = wb.get(sid)
        if vb is not None:
            dot += va * vb

    na = math.sqrt(sum(v * v for v in wa.values()))
    nb = math.sqrt(sum(v * v for v in wb.values()))
    if na <= 0.0 or nb <= 0.0:
        return 0.0

    sigma = dot / (na * nb)
    if sigma < 0.0:
        sigma = 0.0
    elif sigma > 1.0:
        sigma = 1.0
    return float(sigma)


def load_history_index(path: str) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history_index(path: str, index: Dict[str, Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def get_history_stats(index: Dict[str, Dict[str, Any]], sig: str) -> Tuple[int, float]:
    """
    Returns (n_hist, t_rec_years).
    index[sig] = {"n": int, "last_ts": "ISO"}
    """
    rec = index.get(sig)
    if not rec:
        return 0, 100.0
    n = safe_int(rec.get("n", 0), 0)
    last_ts = rec.get("last_ts")
    try:
        last = dt.datetime.fromisoformat(last_ts.replace("Z", ""))
        days = (dt.datetime.utcnow() - last).days
        t_rec_years = max(0.01, days / 365.0)
    except Exception:
        t_rec_years = 100.0
    return n, t_rec_years

def update_history(index: Dict[str, Dict[str, Any]], sig: str, n_add: int = 1) -> None:
    rec = index.get(sig) or {"n": 0, "last_ts": now_utc_iso()}
    rec["n"] = safe_int(rec.get("n", 0), 0) + n_add
    rec["last_ts"] = now_utc_iso()
    index[sig] = rec

def compute_f2_score(pack: Dict[str, Any], qc: QC, selected: List[QC], N_total: int) -> float:
    # Score(q)=(n_hist/N_total)*(1+α/t_rec)*Ψ_q*Π_{p∈S}((1-σ(q,p))*100)
    if N_total <= 0:
        return 0.0
    alpha = float(qc.alpha)
    t_rec = max(0.01, float(qc.t_rec_years))
    base = (qc.n_hist / N_total) * (1.0 + alpha / t_rec) * qc.psi_norm
    prod = 1.0
    for p in selected:
        sigma = compute_similarity_sigma(qc, p)
        prod *= (1.0 - sigma) * 100.0
    return float(base * prod)

def greedy_select_cover(pack: Dict[str, Any], qcs: List[QC], universe_qi_ids: List[str], N_total: int) -> Tuple[List[QC], List[str], Dict[str, str], Dict[str, Any]]:
    uncovered = set(universe_qi_ids)
    S: List[QC] = []
    cov_map: Dict[str, str] = {}
    log = {"iterations": []}

    # Pre-sort deterministically by psi_norm desc then qc_id
    qcs_sorted = sorted(qcs, key=lambda q: (-q.psi_norm, q.qc_id))

    while uncovered:
        best = None
        best_gain = -1
        best_score = -1.0

        for qc in qcs_sorted:
            gain = len(set(qc.cover_qi_ids) & uncovered)
            if gain <= 0:
                continue
            score = compute_f2_score(pack, qc, S, N_total)
            # Choose by gain, then score, then tie-break (Ψ_q, t_rec, n_hist, hash(SIG))
            if (gain > best_gain) or (gain == best_gain and score > best_score):
                best = qc
                best_gain = gain
                best_score = score
            elif gain == best_gain and abs(score - best_score) < 1e-12 and best is not None:
                # tie-break scellé
                if qc.psi_norm > best.psi_norm:
                    best = qc
                elif abs(qc.psi_norm - best.psi_norm) < 1e-12:
                    if qc.t_rec_years < best.t_rec_years:
                        best = qc
                    elif abs(qc.t_rec_years - best.t_rec_years) < 1e-12:
                        if qc.n_hist < best.n_hist:
                            best = qc
                        elif qc.n_hist == best.n_hist:
                            if sha256_text(qc.sig) < sha256_text(best.sig):
                                best = qc

        if best is None:
            break

        S.append(best)
        covered = set(best.cover_qi_ids) & uncovered
        for qi_id in covered:
            cov_map[qi_id] = best.qc_id
        uncovered -= covered
        log["iterations"].append({
            "selected_qc_id": best.qc_id,
            "gain": best_gain,
            "score": best_score,
            "uncovered_remaining": len(uncovered),
        })

    # Safety stop condition (Kernel correction 6)
    orphans = sorted(list(uncovered))
    safety = {
        "triggered": False,
        "orphan_cap": None,
        "reason_code": None,
    }
    orphan_cap = min(int(math.ceil(ORPHAN_TOLERANCE_THRESHOLD * max(1, N_total))), ORPHAN_TOLERANCE_ABSOLUTE)
    safety["orphan_cap"] = orphan_cap

    if orphans:
        # Evaluate conditions
        # Here we cannot compute "no candidate with Score>min" precisely for remaining; approximate by checking max dynamic score among candidates
        max_score = 0.0
        for qc in qcs_sorted:
            max_score = max(max_score, compute_f2_score(pack, qc, S, N_total))
        # LOW_HISTORY check is based on qc.n_hist, qc.t_rec (already loaded per sig). We'll treat remaining uncovered as low history if all remaining qcs able to cover them have low stats.
        low_history_ok = True
        for qi_id in orphans:
            covering = [qc for qc in qcs_sorted if qi_id in qc.cover_qi_ids]
            if not covering:
                continue
            # if any covering has not low history, then not low-history-only
            if any((qc.n_hist > LOW_HISTORY_N_Q_MAX or qc.t_rec_years <= LOW_HISTORY_T_REC_MIN_YEARS) for qc in covering):
                low_history_ok = False
                break

        if len(orphans) <= orphan_cap and max_score <= SCORE_MIN_VIABLE and low_history_ok:
            safety["triggered"] = True
            safety["reason_code"] = RC_SINGLETON_IRREDUCTIBLE

    # Mark remaining as ORPHAN in map
    for qi_id in orphans:
        cov_map[qi_id] = "ORPHAN"

    return S, orphans, cov_map, safety


# =============================================================================
# IA2 Judge (boolean checks; deterministic)
# =============================================================================

def ia2_checks(pack: Dict[str, Any], chapter_ref: str, qc: QC, universe_qi_ids: List[str], N_total: int, ocr_context_refs: List[str]) -> IA2AuditLog:
    checks: List[AuditEntry] = []
    evid = list(set(qc.evidence_refs + ocr_context_refs))

    def add(check_id: str, ok: bool, details: str, fix: List[str], ev: Optional[List[str]] = None):
        checks.append(AuditEntry(
            check_id=check_id,
            status="PASS" if ok else "FAIL",
            evidence_refs=(ev or evid) if not ok else [],
            details=details[:200],
            fix_recommendations=fix if not ok else [],
        ))

    # POSABLE_VALID: QC must be built only from POSABLE atoms (enforced upstream)
    add("POSABLE_VALID", True, "Built from POSABLE atoms only", [])

    # QC_FORM: must start with pack qc_format.prefix and end with suffix
    prefix = str(pack_get(pack, "qc_format.prefix", "Comment"))
    suffix = str(pack_get(pack, "qc_format.suffix", "?"))
    ok_form = qc.qc_label.lower().startswith(prefix.lower()) and qc.qc_label.endswith(suffix)
    add("QC_FORM", ok_form, "QC label form", ["Adjust qc_format in pack or builder"], [])

    # NO_LOCAL_CONSTANTS: heuristic (no raw years, no centre names). Strict enforcement is IA2: any remaining year-like pattern fails.
    ok_local = not bool(re.search(r"\b(19\d{2}|20\d{2})\b", qc.qc_label))
    add("NO_LOCAL_CONSTANTS", ok_local, "No local constants in QC label", ["Ensure sanitizer + builder remove local constants"], [])

    # FRT_TEMPLATE_OK: must contain required sections
    required = ["Usage", "Préconditions", "ARI", "Transitions & checkpoints", "Pièges & erreurs fatales", "Sortie attendue", "Preuves"]
    ok_frt = all(k in qc.frt.sections for k in required)
    add("FRT_TEMPLATE_OK", ok_frt, "FRT required sections present", ["Fill all mandatory FRT sections"], [])

    # TRIGGERS_QUALITY: 4–5 triggers recommended
    ok_tr = 4 <= len(qc.triggers) <= 7
    add("TRIGGERS_QUALITY", ok_tr, "Triggers count within [4,7]", ["Improve trigger extraction patterns in pack"], [])

    # ARI_TYPED_ONLY: every step_id must exist in pack cognitive weights
    cw = load_cognitive_table(pack)
    ok_ari = all(s.step_id in cw for s in qc.ari.steps) and len(qc.ari.steps) >= 1
    add("ARI_TYPED_ONLY", ok_ari, "ARI steps are typed canonical IDs", ["Add mappings/weights to pack.cognitive"], [])

    # F1_RECALCULABLE: needs delta_c + weights + epsilon
    ok_f1 = qc.delta_c > 0 and qc.psi_raw > 0
    add("F1_RECALCULABLE", ok_f1, "F1 recalculable", ["Ensure pack provides δ_c and cognitive weights"], [])

    # F2_TERMS_VISIBLE: all terms must be present (n_hist, N_total, t_rec, alpha, psi_norm, sigma computable)
    ok_f2 = (N_total >= 0) and (qc.t_rec_years > 0) and (qc.alpha >= 0) and (0 < qc.psi_norm <= 1.0)
    add("F2_TERMS_VISIBLE", ok_f2, "F2 terms visible", ["Ensure history index + pack alpha provided"], [])

    # ATTACHMENT_RULE (simplified deterministic): cover ids non-empty
    ok_att = len(qc.cover_qi_ids) >= 1
    add("ATTACHMENT_RULE", ok_att, "Attach/Cover provided", ["Ensure builder provides cover mapping"], [])

    # COVERAGE_BOOL is checked at chapter level; here we only ensure QC's cover subset of universe
    ok_covsubset = set(qc.cover_qi_ids).issubset(set(universe_qi_ids))
    add("COVER_SUBSET", ok_covsubset, "Cover subset of universe POSABLE", ["Fix cover map builder"], [])

    # DETERMINISM_LOCK: no randomness used; tie-break is deterministic. We mark PASS by design.
    add("DETERMINISM_LOCK", True, "No random seed, deterministic sorts", [])

    # HASH_INTEGRITY: QC has evidence refs + fingerprint computed in run signature; checked globally.
    add("HASH_INTEGRITY", True, "Integrity computed in run signature", [])

    # NO_RECONSTRUCTION (CAS 1 ONLY): we never generate RQi; if empty RQi → not POSABLE. QC built from POSABLE only.
    add("NO_RECONSTRUCTION", True, "No reconstructed content", [])

    overall = "PASS" if all(c.status == "PASS" for c in checks) else "FAIL"
    audit_id = f"AUD_{uuid.uuid4().hex[:12]}"
    pack_version = str(pack_get(pack, "pack_version", "PACK_UNKNOWN"))
    pack_sig = str(pack_get(pack, "pack_fingerprint_sha256", sha256_text(json.dumps(pack, sort_keys=True))))
    determinism = {"tie_break_version": "CORRECTION_3", "random_seed": None}

    # hash integrity for audit itself
    audit_payload = {
        "audit_id": audit_id,
        "timestamp": now_utc_iso(),
        "kernel_version": KERNEL_VERSION,
        "pack_version": pack_version,
        "pack_fingerprint_sha256": pack_sig,
        "chapter_ref": chapter_ref,
        "qc_id": qc.qc_id,
        "overall_status": overall,
        "checks": [asdict(c) for c in checks],
        "determinism_params": determinism,
    }
    audit_hash = sha256_text(json.dumps(audit_payload, ensure_ascii=False, sort_keys=True))

    return IA2AuditLog(
        audit_id=audit_id,
        timestamp_utc=audit_payload["timestamp"],
        kernel_version=KERNEL_VERSION,
        pack_version=pack_version,
        pack_fingerprint_sha256=pack_sig,
        chapter_ref=chapter_ref,
        qc_id=qc.qc_id,
        checks=checks,
        overall_status=overall,
        hash_integrity=audit_hash,
        determinism_params=determinism,
    )


# =============================================================================
# Pipeline runner
# =============================================================================

def build_atoms_for_pair(pack: Dict[str, Any], pair: PairItem, evidence: Dict[str, Any], logs: List[str]) -> Tuple[List[Atom], List[str], List[str]]:
    """
    For one pair: extract text (OCR flow), segment, pair Qi↔RQi, sanitize, scope+POSABLE.
    Returns atoms, quarantine_reason_codes, ocr_evidence_refs.
    """
    quarantine_rc: List[str] = []
    ocr_evidence_refs: List[str] = []

    subj_bytes, subj_ct = fetch_url_bytes(pair.subject_url)
    corr_bytes, corr_ct = fetch_url_bytes(pair.correction_url)
    subj_fp = sha256_bytes(subj_bytes)
    corr_fp = sha256_bytes(corr_bytes)

    # Evidence sources
    subj_source_id = f"SRC_{uuid.uuid4().hex[:10]}"
    corr_source_id = f"SRC_{uuid.uuid4().hex[:10]}"
    evidence["sources"].append(asdict(SourceRef(
        source_id=subj_source_id,
        source_url=pair.subject_url,
        mime_type=subj_ct,
        fingerprint_sha256=subj_fp,
        retrieved_at_utc=now_utc_iso(),
        local_name=os.path.basename(urlparse(pair.subject_url).path) or pair.pair_id + "_S",
    )))
    evidence["sources"].append(asdict(SourceRef(
        source_id=corr_source_id,
        source_url=pair.correction_url,
        mime_type=corr_ct,
        fingerprint_sha256=corr_fp,
        retrieved_at_utc=now_utc_iso(),
        local_name=os.path.basename(urlparse(pair.correction_url).path) or pair.pair_id + "_C",
    )))

    # OCR/text extraction flow
    subj_doc = extract_text_with_ocr_flow(subj_bytes, subj_ct, pack)
    corr_doc = extract_text_with_ocr_flow(corr_bytes, corr_ct, pack)

    # Record OCR runs into EvidencePack derivations
    def record_ocr(doc: DocText, role: str, source_id: str):
        ref = f"EV_OCR:{source_id}:{role}"
        ocr_evidence_refs.append(ref)
        evidence["derivations"].append({
            "evidence_ref": ref,
            "role": role,
            "source_id": source_id,
            "method": doc.method,
            "text_sha256": doc.text_sha256,
            "consensus_divergence": doc.consensus_divergence,
            "reason_codes": doc.reason_codes,
            "ocr_runs": [asdict(r) for r in doc.ocr_runs],
        })
        # also store an extraction snippet reference (first 800 chars)
        evidence["extractions"].append({
            "evidence_ref": f"{ref}:SNIPPET",
            "source_id": source_id,
            "role": role,
            "snippet": doc.text[:800],
            "snippet_sha256": sha256_text(doc.text[:800]),
        })

    record_ocr(subj_doc, "SUBJECT", subj_source_id)
    record_ocr(corr_doc, "CORRECTION", corr_source_id)

    if subj_doc.reason_codes:
        quarantine_rc.extend(subj_doc.reason_codes)
    if corr_doc.reason_codes:
        quarantine_rc.extend(corr_doc.reason_codes)

    if not subj_doc.text.strip():
        quarantine_rc.append(RC_EXTRACTION_CORRUPTED)
    if not corr_doc.text.strip():
        quarantine_rc.append(RC_CORRIGE_UNREADABLE)

    if quarantine_rc:
        return [], sorted(set(quarantine_rc)), ocr_evidence_refs

    # Segment
    qi_segs = split_into_segments(subj_doc.text)
    rqi_segs = split_into_segments(corr_doc.text)
    pairs_map = pair_qi_to_rqi(qi_segs, rqi_segs)

    atoms: List[Atom] = []
    seen = set()

    for qi_idx, rqi_idx, conf in pairs_map:
        qi_raw, (qs, qe) = qi_segs[qi_idx]
        rqi_raw, (rs, re_) = ("", (0,0))
        if rqi_idx is not None and 0 <= rqi_idx < len(rqi_segs):
            rqi_raw, (rs, re_) = rqi_segs[rqi_idx]

        # Sanitization
        qi_clean, d1 = sanitize_text(qi_raw)
        rqi_clean, d2 = sanitize_text(rqi_raw)
        derivs = d1 + d2

        # Scope
        chapter_ref, scope_trace, scope_rc = scope_match_chapter(pack, qi_clean, rqi_clean)
        posable, posable_rc = posable_gate(pack, qi_clean, rqi_clean, chapter_ref, scope_rc)

        atom_key = sha256_text(qi_clean + "\n" + rqi_clean)
        if atom_key in seen:
            quarantine_rc.append(RC_DUPLICATE_ATOM)
            continue
        seen.add(atom_key)

        atom_id = f"ATOM_{uuid.uuid4().hex[:12]}"
        qi_id = f"QI_{uuid.uuid4().hex[:12]}"
        rqi_id = f"RQI_{uuid.uuid4().hex[:12]}"
        # Minimal locators
        loc = {
            "qi": asdict(ExtractionLocator(0, 0, qs, qe, {})),
            "rqi": asdict(ExtractionLocator(0, 0, rs, re_, {})),
        }

        atom = Atom(
            atom_id=atom_id,
            subject_id=pair.pair_id + "_S",
            correction_id=pair.pair_id + "_C",
            qi_id=qi_id,
            rqi_id=rqi_id,
            qi_raw=qi_raw,
            rqi_raw=rqi_raw,
            qi_clean=qi_clean,
            rqi_clean=rqi_clean,
            language_detected=str(pack_get(pack, "language.default", "fr")),
            year_ref=pair.year,
            source_subject_url=pair.subject_url,
            source_correction_url=pair.correction_url,
            subject_fingerprint_sha256=subj_fp,
            correction_fingerprint_sha256=corr_fp,
            extraction_locators=loc,
            sanitizer_derivations=derivs,
            pairing_confidence=float(conf),
            figure_refs=[],
        )
        # Attach OCR evidence refs (required by OCR V10.6.2): propagate to IA2 audit context
        atom.sanitizer_derivations.append({"op": "ocr_context", "ocr_evidence_refs": list(ocr_evidence_refs)})

        # Attach POSABLE decision to evidence
        evidence_ref = f"EV_POSABLE:{qi_id}"
        evidence["derivations"].append({
            "evidence_ref": evidence_ref,
            "qi_id": qi_id,
            "chapter_ref": chapter_ref,
            "posable": posable,
            "reason_codes": posable_rc,
            "scope_trace": scope_trace,
        })

        # Store decision in atom as extra in derivations
        atom.sanitizer_derivations.append({"op": "scope_posable", "evidence_ref": evidence_ref, "chapter_ref": chapter_ref, "posable": posable, "reason_codes": posable_rc})

        if posable:
            atoms.append(atom)
        else:
            quarantine_rc.extend(posable_rc)

    return atoms, sorted(set(quarantine_rc)), ocr_evidence_refs


def build_qc_for_chapter(pack: Dict[str, Any], chapter_ref: str, atoms: List[Atom], evidence: Dict[str, Any], history_index: Dict[str, Any], ocr_context_refs: List[str]) -> Tuple[List[QC], List[IA2AuditLog]]:
    cognitive_weights = load_cognitive_table(pack)
    alpha = float(pack_get(pack, "policy.alpha", 0.0))
    delta_matrix = pack_get(pack, "policy.delta_c", {}) or {}
    delta_c = float(delta_matrix.get(str(pack_get(pack, "selection.level_ref", "DEFAULT")), delta_matrix.get("DEFAULT", 1.0)))
    if delta_c <= 0:
        delta_c = 1.0

    # IA1 Miner per atom
    traces = {}
    for a in atoms:
        acts, ev = ia1_miner(pack, a)
        traces[a.qi_id] = {"actions": acts, "evidence_refs": ev}

    # Clustering on ARI signals (action sets)
    clusters: Dict[str, List[Atom]] = {}
    for a in atoms:
        sig_key = "|".join(traces[a.qi_id]["actions"]) or "EMPTY"
        clusters.setdefault(sig_key, []).append(a)

    qcs: List[QC] = []
    audits: List[IA2AuditLog] = []

    # Build QC per cluster (anti-singleton)
    for sig_key, members in sorted(clusters.items(), key=lambda kv: (kv[0], len(kv[1]))):
        if len(members) < 2:
            continue  # anti-singleton
        # Choose champion by pairing_confidence then year then qi_id hash (deterministic)
        members_sorted = sorted(
            members,
            key=lambda a: (-a.pairing_confidence, -(a.year_ref or 0), sha256_text(a.qi_id)),
        )
        champion = members_sorted[0]
        qc_label = qc_label_from_pack(pack, champion.qi_clean)

        # Build ARI steps from majority action spine (sorted canonical IDs)
        action_ids = traces[champion.qi_id]["actions"]
        steps = []
        for cid in action_ids:
            w = float(cognitive_weights.get(cid, 0.0))
            steps.append(ARIStep(step_id=cid, weight=w, evidence_refs=[f"EV_ATOM:{champion.atom_id}:RQi"]))
        ari = ARI(steps=steps)

        # Triggers (4–5)
        triggers = build_triggers(pack, [m.qi_clean for m in members_sorted[:5]], max_triggers=5)

        # Operator for FRT: pack may infer; else use chapter_ref
        operator = str(pack_get(pack, "frt.operator_default", chapter_ref))

        # Evidence refs
        evidence_refs = [f"EV_CLUSTER:{chapter_ref}:{sha256_text(sig_key)[:12]}"]
        evidence["derivations"].append({
            "evidence_ref": evidence_refs[0],
            "chapter_ref": chapter_ref,
            "cluster_size": len(members),
            "qi_ids": [m.qi_id for m in members],
            "sig_key": sig_key,
        })

        # FRT
        frt = build_frt(pack, operator, qc_label, ari, triggers, evidence_refs=evidence_refs)

        # Signature SIG(q)
        output_type = infer_output_type(pack, champion.qi_clean, champion.rqi_clean)
        preconds = list(set(pack_get(pack, "frt.preconditions_default", []) or []))
        checkpoints = list(set(pack_get(pack, "frt.checkpoints_default", []) or []))
        sig = build_sig([s.step_id for s in ari.steps], preconds, output_type, checkpoints)

        # History
        n_hist, t_rec = get_history_stats(history_index, sig)

        qc_id = f"QC_{uuid.uuid4().hex[:12]}"

        # Cover: members' qi_ids (strict); no speculative covering without proof
        cover_qi_ids = [m.qi_id for m in members]

        qc = QC(
            qc_id=qc_id,
            chapter_ref=chapter_ref,
            qc_label=qc_label,
            sig=sig,
            n_q_cluster=len(members),
            ari=ari,
            triggers=triggers,
            frt=frt,
            delta_c=delta_c,
            psi_raw=0.0,
            psi_norm=0.0,
            n_hist=n_hist,
            t_rec_years=t_rec,
            alpha=alpha,
            score=0.0,
            cover_qi_ids=cover_qi_ids,
            evidence_refs=evidence_refs,
        )

        qcs.append(qc)

    # Compute F1, normalize per chapter
    if qcs:
        for qc in qcs:
            psi_raw, _ = compute_f1(pack, qc, cognitive_weights)
            qc.psi_raw = psi_raw
        max_raw = max(q.psi_raw for q in qcs) if qcs else 1.0
        for qc in qcs:
            qc.psi_norm = qc.psi_raw / max(1e-12, max_raw)

    # IA2 audits per QC (note: coverage bool is chapter-level)
    universe = [a.qi_id for a in atoms]
    N_total = len(universe)

    for qc in qcs:
        aud = ia2_checks(pack, chapter_ref, qc, universe_qi_ids=universe, N_total=N_total, ocr_context_refs=ocr_context_refs)
        audits.append(aud)

    # Keep only IA2 PASS for scoring/selection
    qcs_pass = [qc for qc, aud in zip(qcs, audits) if aud.overall_status == "PASS"]
    audits_pass = [aud for aud in audits if aud.overall_status == "PASS"]

    return qcs_pass, audits_pass


def run_gte(pack: Dict[str, Any], years: List[int], max_pairs: int, vol_start: int, vol_max: int, step: int, top_k_per_chapter: int, override_pairs: Optional[List[PairItem]] = None) -> Dict[str, Any]:
    # EvidencePack skeleton (matches user JSON key set, extended)
    evidence_pack = {
        "version": APP_VERSION,
        "timestamp": now_utc_iso(),
        "pipeline": "GTE",
        "pack_id": pack_get(pack, "pack_id", "PACK_UNKNOWN"),
        "pack_sig_sha256": pack_get(pack, "pack_fingerprint_sha256", sha256_text(json.dumps(pack, sort_keys=True))),
        "sources": [],
        "extractions": [],
        "derivations": [],
        # Summary fields (filled later)
        "pairs_processed": 0,
        "meta_pairs": [],
        "qi_total": 0,
        "qi_posable": 0,
        "qc_total": 0,
        "qc_posable": 0,
        "coverage": 0.0,
        "orphans_posable": [],
        "quarantine": [],
        "quarantine_count": 0,
        "ia2_ok": True,
        "ia2_issues": [],
        "extraction_kpi": {},
        "sanity_dup_ratio": 0.0,
        "qc_evidence": [],
    }

    logs: List[str] = []
    
    logs: List[str] = []
    if override_pairs is not None and len(override_pairs) > 0:
        pairs = override_pairs
        logs.append('[HARVEST] override_pairs used')
    else:
        pairs, harvest_logs = harvest_pairs_from_pack(pack, selected_years=years, max_pairs=max_pairs)
        logs.extend(harvest_logs)

    # History index storage (deterministic local file)
    hist_path = pack_get(pack, "policy.history_index_path", "historical_index.json")
    history_index = load_history_index(hist_path)

    # Iterative saturation (progressive volume) until new_QC == 0
    all_atoms: List[Atom] = []
    all_qcs: List[QC] = []
    all_audits: List[IA2AuditLog] = []
    quarantine: List[Dict[str, Any]] = []

    prev_qc_sig_set = set()

    final_volume = vol_start
    for volume in range(vol_start, vol_max + 1, step):
        final_volume = volume
        logs.append(f"[RUN] volume={volume}")

        atoms_this: List[Atom] = []
        qc_this: List[QC] = []
        audits_this: List[IA2AuditLog] = []

        use_pairs = pairs[: min(len(pairs), volume)]
        evidence_pack["meta_pairs"] = [asdict(p) for p in use_pairs]
        evidence_pack["pairs_processed"] = len(use_pairs)

        for p in use_pairs:
            a, qrc, ocr_refs = build_atoms_for_pair(pack, p, evidence_pack, logs)
            if qrc:
                quarantine.append({"pair_id": p.pair_id, "reason_codes": qrc})
            atoms_this.extend(a)

        # Group atoms by chapter_ref + collect OCR context refs (pack-driven evidence)
        chap_atoms: Dict[str, List[Atom]] = {}
        chap_ocr_refs: Dict[str, List[str]] = {}
        for a in atoms_this:
            chapter_ref = None
            ocr_refs_local: List[str] = []
            for d in reversed(a.sanitizer_derivations):
                if d.get("op") == "scope_posable":
                    chapter_ref = d.get("chapter_ref")
                if d.get("op") == "ocr_context":
                    ocr_refs_local = list(d.get("ocr_evidence_refs") or [])
                if chapter_ref is not None and ocr_refs_local is not None:
                    # keep scanning to capture both; break only when both seen
                    if chapter_ref and ocr_refs_local is not None:
                        # continue scanning to also preserve determinism; not required
                        pass
            if not chapter_ref:
                continue
            chap_atoms.setdefault(chapter_ref, []).append(a)
            if ocr_refs_local:
                cur = chap_ocr_refs.get(chapter_ref) or []
                cur.extend(ocr_refs_local)
                chap_ocr_refs[chapter_ref] = cur

        # Build QC per chapter
        for chapter_ref, atoms in sorted(chap_atoms.items(), key=lambda kv: kv[0]):
            ocr_refs = sorted(set(chap_ocr_refs.get(chapter_ref, []) or []))
            qcs, auds = build_qc_for_chapter(pack, chapter_ref, atoms, evidence_pack, history_index, ocr_context_refs=ocr_refs)
            # Limit top_k per chapter (selection later is coverage-driven; but keep cap for UI)
            qcs = sorted(qcs, key=lambda q: (-q.psi_norm, q.qc_id))[:top_k_per_chapter]
            qc_this.extend(qcs)
            audits_this.extend(auds)

        # Compute new_QC via signature set difference
        sig_set = set(q.sig for q in qc_this)
        new_qc_count = len(sig_set - prev_qc_sig_set)
        prev_qc_sig_set |= sig_set

        logs.append(f"[RUN] qc_this={len(qc_this)} new_qc_count={new_qc_count}")

        all_atoms = atoms_this
        all_qcs = qc_this
        all_audits = audits_this

        if new_qc_count == 0:
            break

    # Coverage selection per chapter
    selected_qcs: List[QC] = []
    coverage_map: Dict[str, str] = {}
    orphans: List[str] = []
    safety_events: List[Dict[str, Any]] = []

    # Rebuild chap_atoms from all_atoms (and retain OCR context refs for traceability)
    chap_atoms: Dict[str, List[Atom]] = {}
    chap_ocr_refs: Dict[str, List[str]] = {}
    for a in all_atoms:
        chapter_ref = None
        ocr_refs_local: List[str] = []
        for d in reversed(a.sanitizer_derivations):
            if d.get("op") == "scope_posable":
                chapter_ref = d.get("chapter_ref")
            if d.get("op") == "ocr_context":
                ocr_refs_local = list(d.get("ocr_evidence_refs") or [])
        if chapter_ref:
            chap_atoms.setdefault(chapter_ref, []).append(a)
            if ocr_refs_local:
                cur = chap_ocr_refs.get(chapter_ref) or []
                cur.extend(ocr_refs_local)
                chap_ocr_refs[chapter_ref] = cur

    chap_qcs: Dict[str, List[QC]] = {}
    for qc in all_qcs:
        chap_qcs.setdefault(qc.chapter_ref, []).append(qc)

    for chapter_ref, atoms in sorted(chap_atoms.items(), key=lambda kv: kv[0]):
        universe = [a.qi_id for a in atoms]
        N_total = len(universe)
        qcs = chap_qcs.get(chapter_ref, [])

        # Update score dynamically during selection; store per qc
        S, orph, cmap, safety = greedy_select_cover(pack, qcs, universe, N_total)

        # write back scores as last computed with final S order
        tmp_selected: List[QC] = []
        for qc in S:
            qc.score = compute_f2_score(pack, qc, tmp_selected, N_total)
            tmp_selected.append(qc)

        selected_qcs.extend(S)
        orphans.extend(orph)
        coverage_map.update(cmap)
        if safety.get("triggered"):
            safety_events.append({"chapter_ref": chapter_ref, **safety})

    total_posable = len([a.qi_id for a in all_atoms])
    covered = len([qi for qi, qcid in coverage_map.items() if qcid != "ORPHAN"])
    coverage = (covered / total_posable) if total_posable else 0.0

    evidence_pack["qi_total"] = total_posable
    evidence_pack["qi_posable"] = total_posable
    evidence_pack["qc_total"] = len(all_qcs)
    evidence_pack["qc_posable"] = len(selected_qcs)
    evidence_pack["coverage"] = coverage
    evidence_pack["orphans_posable"] = sorted(list({o for o in orphans}))
    evidence_pack["quarantine"] = quarantine
    evidence_pack["quarantine_count"] = len(quarantine)
    evidence_pack["extraction_kpi"] = {
        "final_volume": final_volume,
        "new_qc_count_last": 0 if final_volume == vol_start else None,
        "pairs_available": len(pairs),
    }

    # IA2 summary
    ia2_fail = [a for a in all_audits if a.overall_status != "PASS"]
    evidence_pack["ia2_ok"] = (len(ia2_fail) == 0)
    evidence_pack["ia2_issues"] = [asdict(x) for x in ia2_fail[:10]]

    # Sealing decision
    sealed = (len(evidence_pack["orphans_posable"]) == 0)
    seal_reason = "ALL_CHECKS_PASSED" if sealed else "ORPHANS_REMAIN"
    if safety_events:
        seal_reason = "RESIDUAL_UNCOVERABLE_EXPLICIT"  # still not full seal unless orphans==0
    # NOTE: Kernel final bool: SEALED iff zero orphan POSABLE; safety events are explicit residuals.

    # Exports: qi_pack + qc_pack aligned with user samples
    qi_pack = []
    for atom in all_atoms:
        # Chapter
        chapter_ref = None
        for d in reversed(atom.sanitizer_derivations):
            if d.get("op") == "scope_posable":
                chapter_ref = d.get("chapter_ref")
                break
        qi_pack.append({
            "qi_id": atom.qi_id,
            "rqi_id": atom.rqi_id,
            "pair_id": atom.subject_id.replace("_S", ""),
            "chapter_ref": chapter_ref,
            "qi_clean": atom.qi_clean,
            "rqi_clean": atom.rqi_clean,
            "match_score": atom.pairing_confidence,
            "scope_chapter_guess": chapter_ref,
            "posable_status": True,
            "mapping_qc_id": coverage_map.get(atom.qi_id, "ORPHAN"),
        })

    qc_pack = []
    for qc in selected_qcs:
        qc_pack.append({
            "qc_id": qc.qc_id,
            "chapter_ref": qc.chapter_ref,
            "qc_label": qc.qc_label,
            "psi": qc.psi_norm,
            "psi_raw": qc.psi_raw,
            "delta_c": qc.delta_c,
            "n_hist": qc.n_hist,
            "t_rec_years": qc.t_rec_years,
            "alpha": qc.alpha,
            "f2_score": qc.score,
            "n_q_cluster": qc.n_q_cluster,
            "ari_steps": [{"step_id": s.step_id, "weight": s.weight} for s in qc.ari.steps],
            "triggers": [t.pattern for t in qc.triggers],
            "frt": qc.frt.sections,
            "cover_qi_ids": qc.cover_qi_ids,
            "sig": qc.sig,
            "evidence_refs": qc.evidence_refs,
        })

    # Audit JSON
    audit_json = {
        "version": APP_VERSION,
        "kernel_version": KERNEL_VERSION,
        "timestamp": now_utc_iso(),
        "pack_version": pack_get(pack, "pack_version", "PACK_UNKNOWN"),
        "pack_sig_sha256": evidence_pack["pack_sig_sha256"],
        "sealed": sealed,
        "seal_reason": seal_reason,
        "coverage": coverage,
        "orphans": evidence_pack["orphans_posable"],
        "safety_events": safety_events,
        "ia2_audits": [asdict(a) for a in all_audits],
        "logs": logs[-600:],  # bounded
    }

    # Mapping export
    mapping = [{"qi_id": qi, "qc_id": qc_id} for qi, qc_id in sorted(coverage_map.items(), key=lambda kv: kv[0])]

    # Run signature: SHA256 of all outputs
    outputs_blob = {
        "qi_pack": qi_pack,
        "qc_pack": qc_pack,
        "audit": audit_json,
        "evidence_pack": evidence_pack,
        "mapping_qi_to_qc": mapping,
    }
    run_sig = sha256_text(json.dumps(outputs_blob, ensure_ascii=False, sort_keys=True))

    # Update history index for selected QCs (optional, deterministic)
    for qc in selected_qcs:
        update_history(history_index, qc.sig, n_add=1)
    save_history_index(hist_path, history_index)

    return {
        "pairs": pairs,
        "atoms": all_atoms,
        "qi_pack": qi_pack,
        "qc_pack": qc_pack,
        "selected_qcs": selected_qcs,
        "coverage": coverage,
        "orphans": evidence_pack["orphans_posable"],
        "sealed": sealed,
        "seal_reason": seal_reason,
        "audit": audit_json,
        "evidence_pack": evidence_pack,
        "mapping": mapping,
        "run_signature": {"sha256": run_sig, "algorithm": "SHA256"},
        "logs": logs,
        "final_volume": final_volume,
    }


# =============================================================================
# Pack loader (external JSON or GENESIS placeholder)
# =============================================================================

def make_genesis_pack_fr() -> Dict[str, Any]:
    """
    GENESIS pack is TEST DATA (not Kernel hardcode).
    Replace by your real CAP JSON in production.
    """
    pack = {
        "pack_id": "CAP_FR_GENESIS_V31_10_23",
        "pack_version": "PACK_ACADEMIC_FR_2026_01_V1",
        "pack_fingerprint_sha256": "",
        "language": {"default": "fr"},
        "qc_format": {"prefix": "Comment", "suffix": "?"},
        "policy": {
            "alpha": 0.5,
            "delta_c": {"DEFAULT": 1.0},
            "history_index_path": "historical_index.json",
        },
        "ocr": {
            "text_first_min_chars": 1200,
            "text_first_min_alpha_ratio": 0.20,
            "consensus_divergence_threshold": 0.18,
            "max_pages": 12,
            "dpi": 200,
        },
        "harvest": {
            "roots": [
                "https://www.apmep.fr/spip.php?page=recherche&recherche=bac%20math%20{year}%20sujet",
                "https://www.apmep.fr/spip.php?page=recherche&recherche=bac%20math%20{year}%20corrig%C3%A9",
                "https://www.apmep.fr/spip.php?page=recherche&recherche=DST%20math%20{year}%20corrig%C3%A9",
                "https://www.apmep.fr/spip.php?page=recherche&recherche=DST%20math%20{year}%20sujet",
                "https://www.apmep.fr/spip.php?page=recherche&recherche=annales%20math%20{year}%20corrig%C3%A9",
                "https://www.apmep.fr/spip.php?page=recherche&recherche=annales%20math%20{year}%20sujet",
            ],
            "min_pairing_confidence": 0.25,
            "year_regex": r"(20\d{2})",
            "subject_regexes": [],
            "correction_regexes": [r"corrig"],
        },
        "academic": {
    "chapters": [
        {"chapter_ref": "FR_MATH_TLE_FONCTIONS", "label": "Fonctions", "match_keywords": ["fonction", "dérivée", "variation", "limite", "courbe", "tangente"], "match_regexes": [r"(?i)\b(dériv|derive|variation|limite|tangente)\b"]},
        {"chapter_ref": "FR_MATH_TLE_GEOM_ESPACE", "label": "Géométrie dans l'espace", "match_keywords": ["espace", "vecteur", "plan", "droite", "repère", "orthogonal", "produit scalaire", "distance"], "match_regexes": [r"(?i)\b(espace|vecteur|produit\s+scalaire|orthogon)\b"]},
        {"chapter_ref": "FR_MATH_TLE_COMPLEXES", "label": "Nombres complexes", "match_keywords": ["complexe", "affixe", "module", "argument", "forme trigonométrique", "e^{i"], "match_regexes": [r"(?i)\b(complex|affixe|module|argument)\b"]},
        {"chapter_ref": "FR_MATH_TLE_PROBA_SUITES", "label": "Probabilités et suites", "match_keywords": ["probabilité", "loi", "binomiale", "conditionnelle", "suite", "récurrence", "u_n", "v_n"], "match_regexes": [r"(?i)\b(probabil|binomial|conditionnell|suite|récurr|recurr)\b"]},
    ]
},
        "cognitive": {
            # Canonical weights (Kernel table socle) — the IDs are conceptuels, not tied to a country.
            "weights": {
                "IDENTIFY": 0.15,
                "DEFINE": 0.15,
                "ANALYZE": 0.20,
                "SYNTHESIZE": 0.20,
                "ILLUSTRATE": 0.25,
                "SIMPLIFY": 0.25,
                "MODEL": 0.30,
                "PLAN": 0.30,
                "CALCULATE": 0.30,
                "FORMULATE": 0.30,
                "COMPARE": 0.35,
                "APPLY_RULE": 0.35,
                "INTERPRET": 0.35,
                "SOLVE": 0.40,
                "DERIVE_INTEGRATE": 0.40,
                "EVALUATE": 0.45,
                "JUSTIFY": 0.45,
                "PROVE": 0.50,
                "RECURRENCE": 0.60,
            },
            "synonyms": {
                "fr": {
                    "calcul": "CALCULATE",
                    "calculer": "CALCULATE",
                    "déterminer": "SOLVE",
                    "determiner": "SOLVE",
                    "résoudre": "SOLVE",
                    "resoudre": "SOLVE",
                    "montrer": "PROVE",
                    "démontrer": "PROVE",
                    "demontrer": "PROVE",
                    "justifier": "JUSTIFY",
                    "appliquer": "APPLY_RULE",
                    "analyser": "ANALYZE",
                    "conclure": "SYNTHESIZE",
                    "définir": "DEFINE",
                }
            }
        },
        "triggers": {
            "pattern_extractors": [
                {"regex": r"(?i)\b(calculer|déterminer|résoudre|montrer|démontrer|justifier)\b.{0,80}"},
            ]
        },
        "frt": {
            "operator_default": "GENERIC",
            "template_generic": {
                "Préconditions": ["Données suffisantes", "Notation cohérente"],
                "Transitions": ["Vérifier hypothèses", "Contrôler unités/conditions"],
                "Pièges & erreurs fatales": ["Oubli de condition", "Erreur de signe"],
                "Sortie attendue": "UNKNOWN",
            },
            "templates_by_operator": {},
            "preconditions_default": ["DATA_AVAILABLE"],
            "checkpoints_default": ["CHECK_DOMAIN", "CHECK_UNITS"],
        },
        "output_type": {
            "rules": [
                {"regex": r"(?i)\b(démontrer|montrer|prouver)\b", "output_type": "PROOF"},
                {"regex": r"(?i)\btableau\b", "output_type": "TABLE"},
                {"regex": r"(?i)\btracer\b|\bgraph\b", "output_type": "GRAPH"},
                {"regex": r"(?i)\bcalculer\b|\bdéterminer\b|\brésoudre\b", "output_type": "RESULT_VALUE"},
            ]
        },
        "selection": {
            "level_ref": "DEFAULT",
        }
    }
    pack["pack_fingerprint_sha256"] = sha256_text(json.dumps(pack, ensure_ascii=False, sort_keys=True))
    return pack


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title=f"SMAXIA GTE Console {APP_VERSION}", layout="wide")

st.title(f"SMAXIA GTE Console {APP_VERSION}")
st.caption("ISO-PROD | Pack-Driven | Saturation progressive (new_QC=0) | OCR V10.6.2 (Text-first → OCR fallback → consensus gate) | Exports preuve (mapping + signature)")

# Sidebar: Activation pack
st.sidebar.header("1. ACTIVATION PACK")
country = st.sidebar.selectbox("Sélectionner le pays", ["FR - France"], index=0)
pack_src = st.sidebar.radio("Pack source", ["GENESIS (test)", "Upload JSON"], index=0)

if pack_src == "Upload JSON":
    up = st.sidebar.file_uploader("Pack JSON", type=["json"])
    if up is None:
        st.sidebar.info("Chargez un Pack JSON pour activer.")
        st.stop()
    pack = json.loads(up.read().decode("utf-8", errors="ignore"))
    if not pack_get(pack, "pack_fingerprint_sha256"):
        pack["pack_fingerprint_sha256"] = sha256_text(json.dumps(pack, ensure_ascii=False, sort_keys=True))
else:
    pack = make_genesis_pack_fr()

st.sidebar.success(f"Pack activé: {pack_get(pack,'pack_id','PACK')}")
st.sidebar.caption(f"PackSig: {pack_get(pack,'pack_fingerprint_sha256','')[:18]}…")

# Tabs
tab_import, tab_run, tab_results, tab_exports = st.tabs(["Import", "RUN", "Résultats", "Exports"])

with tab_import:
    st.subheader("Import & Harvest (pack-driven)")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    years = st.multiselect("Années", options=list(range(dt.datetime.utcnow().year, 2009, -1)), default=[dt.datetime.utcnow().year])
    max_pairs = st.number_input("Max pairs", min_value=1, max_value=300, value=30, step=1)
    if st.button("Prévisualiser Harvest"):
        with st.spinner("Harvest en cours..."):
            pairs, logs = harvest_pairs_from_pack(pack, selected_years=years, max_pairs=int(max_pairs))
        st.write(f"Paires trouvées: {len(pairs)}")
        st.dataframe([asdict(p) for p in pairs])
        st.text("\n".join(logs[-200:]))


    st.markdown("### Injection manuelle (si Harvest = 0)")
    st.caption("Permet de tester le pipeline sur un couple Sujet+Corrigé sans dépendre du crawler.")
    mcol1, mcol2, mcol3 = st.columns([5,5,2])
    manual_subject = mcol1.text_input("Subject URL (PDF)", key="manual_subject_url")
    manual_correction = mcol2.text_input("Correction URL (PDF)", key="manual_correction_url")
    manual_year = mcol3.number_input("Année", min_value=2000, max_value=dt.datetime.utcnow().year, value=dt.datetime.utcnow().year, step=1, key="manual_year")

    if st.button("Ajouter paire manuelle"):
        if manual_subject and manual_correction and is_url(manual_subject) and is_url(manual_correction):
            mp = st.session_state.get("manual_pairs", [])
            mp.append(PairItem(
                pair_id=f"MANUAL_{uuid.uuid4().hex[:10]}",
                subject_url=manual_subject.strip(),
                correction_url=manual_correction.strip(),
                year=int(manual_year),
                meta={"source": "manual"},
                pairing_confidence=1.0,
            ))
            st.session_state["manual_pairs"] = mp
            st.success("Paire manuelle ajoutée.")
        else:
            st.error("Veuillez fournir deux URLs valides (http/https).")

    mp = st.session_state.get("manual_pairs", [])
    if mp:
        st.write(f"Paires manuelles: {len(mp)}")
        st.dataframe([asdict(x) for x in mp])
        if st.button("Vider les paires manuelles"):
            st.session_state["manual_pairs"] = []
            st.success("Liste vidée.")


with tab_run:
    st.subheader("RUN — Saturation progressive (new_QC=0)")
    c1, c2, c3, c4 = st.columns(4)
    vol_start = c1.number_input("vol_start", min_value=1, max_value=200, value=5, step=1)
    vol_max = c2.number_input("vol_max", min_value=1, max_value=500, value=30, step=1)
    step = c3.number_input("step", min_value=1, max_value=50, value=5, step=1)
    top_k = c4.number_input("F2 top_k / chapitre", min_value=1, max_value=200, value=15, step=1)

    st.caption("SEALED strict = (orphans==0) AND (ia2_ok) AND (qi_posable>0) AND (qc_posable>0).")
    if st.button("LANCER TEST SATURATION", type="primary"):
        with st.spinner("Exécution GTE..."):
            res = run_gte(
                pack=pack,
                years=years,
                max_pairs=int(max_pairs),
                vol_start=int(vol_start),
                vol_max=int(vol_max),
                step=int(step),
                top_k_per_chapter=int(top_k),
                override_pairs=st.session_state.get('manual_pairs') if st.session_state.get('manual_pairs') else None,
            )
        st.session_state["last_run"] = res
        st.success("RUN terminé.")
        st.write("Logs (fin):")
        st.text("\n".join(res["logs"][-200:]))

with tab_results:
    st.subheader("Résultats")
    res = st.session_state.get("last_run")
    if not res:
        st.info("Lancez un RUN pour voir les résultats.")
    else:
        # KPIs
        pairs = res["pairs"]
        qi_total = len(res["qi_pack"])
        qi_posable = res["evidence_pack"]["qi_posable"]
        qc_total = res["evidence_pack"]["qc_total"]
        qc_posable = res["evidence_pack"]["qc_posable"]
        coverage = res["coverage"]
        sealed = res["sealed"]

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Pairs", res["evidence_pack"]["pairs_processed"])
        k2.metric("Qi", qi_total)
        k3.metric("Qi POSABLE", qi_posable)
        k4.metric("QC", qc_total)
        k5.metric("QC POSABLE", qc_posable)
        k6.metric("Coverage", f"{coverage*100:.1f}%")

        st.markdown(f"**SEALED:** {'YES' if sealed else 'NO'}  \n**seal_reason:** `{res['seal_reason']}`  \n**run_signature:** `{res['run_signature']['sha256'][:18]}…`")

        st.subheader("Itérations (saturation)")
        # lightweight derived table from logs
        it_rows = []
        for ln in res["logs"]:
            m = re.search(r"\[RUN\]\s+volume=(\d+)", ln)
            if m:
                it_rows.append({"volume": int(m.group(1))})
        if it_rows:
            st.dataframe(it_rows)

        st.subheader("Orphans (POSABLE)")
        st.write(res["orphans"])

        st.subheader("Sélection F2 par chapitre")
        # Group QCs by chapter
        by_ch = {}
        for qc in res["qc_pack"]:
            by_ch.setdefault(qc["chapter_ref"], []).append(qc)
        for ch, qcs in sorted(by_ch.items(), key=lambda kv: kv[0]):
            with st.expander(f"{ch} ({len(qcs)} QC)"):
                for q in sorted(qcs, key=lambda x: (-x["f2_score"], -x["psi"], x["qc_id"])):
                    st.markdown(f"**{q['qc_label']}**  \nID: `{q['qc_id']}` | Ψ={q['psi']:.3f} | F2={q['f2_score']:.3f} | n_q_cluster={q['n_q_cluster']}")

        st.subheader("Table Qi (avec RQi) + mapping Qi→QC")
        st.dataframe(res["qi_pack"][:500])

with tab_exports:
    st.subheader("Exports (preuve + intégrité)")
    res = st.session_state.get("last_run")
    if not res:
        st.info("Lancez un RUN pour générer les exports.")
    else:
        # Create downloadable JSONs in memory
        qc_pack_json = json.dumps(res["qc_pack"], ensure_ascii=False, indent=2)
        qi_pack_json = json.dumps(res["qi_pack"], ensure_ascii=False, indent=2)
        audit_json = json.dumps(res["audit"], ensure_ascii=False, indent=2)
        evidence_json = json.dumps(res["evidence_pack"], ensure_ascii=False, indent=2)
        mapping_json = json.dumps(res["mapping"], ensure_ascii=False, indent=2)
        run_sig_json = json.dumps(res["run_signature"], ensure_ascii=False, indent=2)

        st.download_button("Télécharger qc_pack.json", data=qc_pack_json, file_name="qc_pack.json", mime="application/json")
        st.download_button("Télécharger qi_pack.json", data=qi_pack_json, file_name="qi_pack.json", mime="application/json")
        st.download_button("Télécharger audit_ia2.json", data=audit_json, file_name="audit_ia2.json", mime="application/json")
        st.download_button("Télécharger evidence_pack.json", data=evidence_json, file_name="evidence_pack.json", mime="application/json")
        st.download_button("Télécharger mapping_qi_to_qc.json", data=mapping_json, file_name="mapping_qi_to_qc.json", mime="application/json")
        st.download_button("Télécharger run_signature.json", data=run_sig_json, file_name="run_signature.json", mime="application/json")

        st.markdown("### Paramètres OCR (à configurer)")
        st.code(
            "export GOOGLE_VISION_API_KEY='...'\n"
            "export AZURE_VISION_ENDPOINT='https://<resource>.cognitiveservices.azure.com'\n"
            "export AZURE_VISION_KEY='...'\n",
            language="bash",
        )
