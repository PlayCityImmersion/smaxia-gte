# =============================================================================
# SMAXIA GTE Console V31.10.23 — ISO-PROD (FICHIER UNIQUE)
# =============================================================================
# BASE: V31.10.22 (user-provided)
#
# PATCHES (strict invariants / anti-hardcode):
# 1) harvest pair_id: remove hardcoded "MATH" => uses selected subject
# 2) split_into_a_segments: remove FR hardcode ("exercice/partie") => pack-driven markers
# 3) OCR lang: pack-driven (atomization.ocr.langs), no hardcoded "fra+eng"
# 4) HTTP pdf cache: thread-safe (lock) to avoid race conditions under ThreadPoolExecutor
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import time
import math
import hashlib
import unicodedata
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st

# ---------------- Optional deps ----------------
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

# OCR optional (depends on environment)
try:
    import fitz  # PyMuPDF  # type: ignore
except Exception:
    fitz = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

# =============================================================================
# KERNEL CONSTANTS (invariants non-métier)
# =============================================================================
APP_VERSION = "V31.10.23"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 35
MAX_PDF_MB = 40

# Thread-safety: Streamlit session_state + threads => lock mandatory for shared cache
_HTTP_CACHE_LOCK = threading.Lock()

# =============================================================================
# RC_* QUARANTAINE (Kernel - codes invariants)
# =============================================================================
RC_CORRIGE_MISSING = "RC_CORRIGE_MISSING"
RC_CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
RC_SUJET_UNREADABLE = "RC_SUJET_UNREADABLE"
RC_RECONSTRUCTED_FORBIDDEN = "RC_RECONSTRUCTED_FORBIDDEN"  # always used if someone tries (we never do)
RC_SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
RC_RQI_MISSING = "RC_RQI_MISSING"
RC_LOW_CONFIDENCE = "RC_LOW_CONFIDENCE"
RC_ATOMIZATION_LOW = "RC_ATOMIZATION_LOW"
RC_ALIGN_LOW = "RC_ALIGN_LOW"

# =============================================================================
# Dataclasses (proof-grade structures)
# =============================================================================
@dataclass(frozen=True)
class PdfExtractMeta:
    method: str
    pages: int
    chars: int
    seconds: float
    ocr_used: bool

@dataclass
class QuarantineEntry:
    pair_id: str
    sujet_url: str
    corrige_url: str
    reason_codes: List[str]
    details: Dict[str, Any]

@dataclass
class PosableDecision:
    is_posable: bool
    reason_codes: List[str]
    confidence: float

@dataclass
class AriOpEvidence:
    op: str
    confidence: float
    evidence: List[str]
    match_count: int

@dataclass
class AriTrace:
    primary_op: str
    ops: List[AriOpEvidence]
    all_ops: List[str]
    confidence_global: float

@dataclass
class QiItem:
    qi_id: str
    pair_id: str
    source_year: int
    source_exam_date: Optional[str]
    qi: str
    rqi: str
    has_rqi: bool
    chapter_code: str
    chapter_match_score: float
    ari: Dict[str, Any]
    triggers: List[str]
    posable: Dict[str, Any]
    qc_id: Optional[str] = None
    is_orphan: bool = False

@dataclass
class FRT:
    frt_id: str
    primary_op: str
    title: str
    sections: Dict[str, Any]
    generated_at: str

@dataclass
class QCItem:
    qc_id: str
    qc: str
    qc_label: str
    chapter_code: str
    primary_op: str
    all_ops: List[str]
    cluster_size: int
    posable_in_cluster: int
    qc_state: str
    qi_ids: List[str]
    triggers: List[str]
    frt: Dict[str, Any]
    # Kernel scoring
    delta_c: float = 1.0
    sum_Tj: float = 0.0
    f1_raw: float = 0.0
    Psi_q: float = 0.0
    n_hist: int = 0
    t_rec: float = 1.0
    f2_score: float = 0.0

# =============================================================================
# Time / logging / deterministic helpers
# =============================================================================
def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def ss_init():
    defaults = {
        "pack_active": None,
        "pack_id": None,
        "pack_sig_sha256": None,
        "country": None,
        "level": None,
        "subject": None,
        "library": [],
        "harvest_manifest": None,
        "qi_pack": None,
        "qc_pack": None,
        "frt_pack": None,
        "mapping_qi_to_qc": None,
        "selection_report": None,
        "evidence_pack": None,
        "quarantine": None,
        "sealed": False,
        "last_run_audit": None,
        "logs": [],
        "run_stats": {},
        "_http_pdf_cache": {},
        "_run_ts": None,  # fixed timestamp per run to keep deterministic within-run
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def log(msg: str):
    st.session_state.logs.append(f"[{_utc_ts()}] {msg}")

def logs_text() -> str:
    return "\n".join(st.session_state.logs[-12000:])

def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:12]

def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace("\r", "\n")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# =============================================================================
# PACK GENESIS (données test) — core reste pack-driven
# =============================================================================
def generate_pack(country_code: str) -> Dict[str, Any]:
    # IMPORTANT: ces données sont TEST/DÉMO.
    # En ISO-PROD réel, le même core fonctionne avec upload de pack scellé.
    base = {
        "pack_id": f"CAP_{country_code}_GENESIS_{APP_VERSION.replace('.','_')}",
        "pack_version": APP_VERSION,
        "country_code": country_code,
        "language": "fr",
        "status": "GENESIS_TEST",
        "created_at": _utc_ts(),
        "_source": "GENESIS_AUTO",

        # ---------------- Harvest source (APMEP) ----------------
        "harvest_sources": [{
            "source_id": "APMEP",
            "source_name": "APMEP - Annales",
            "base_url": "https://www.apmep.fr",
            # NOTE: l’URL exacte est une donnée de pack (pas core)
            "levels": {
                "TERMINALE": "/Annales-Terminale-Generale",
                "PREMIERE": "/Annales-du-Bac-Premiere"
            },
            "year_pattern": r"(20\d{2})",
            "pdf_patterns": {
                "corrige": ["corrig", "correction", "solution", "corrige"],
            },
            "meta_exclude": ["index", "sommaire", "liste", "annexe", "grille", "formulaire"],
            "geographic_zones": ["metro", "metropole", "ameri]()_
