# =============================================================================
# SMAXIA GTE Console V31.10.22 (ISO-PROD TEST — Kernel V10.6.1 aligned)
# =============================================================================
# KEY FIX V31.10.22:
# - Atomisation robuste Sujet->Exos->Questions->Sous-questions (multi-pages)
# - Extraction RQi (corrigés) + affichage UI + exports dédiés
# - CAS 1 ONLY strict: aucune reconstruction; quarantaine RC_* + preuves
# - POSABLE Gate (corrigé + scope + évaluable) avant toute étape aval
# - Pack-driven (qc_format, scope rules, verb ontology mappings). Si absent => quarantine.
# - Saturation progressive new_QC=0 + SHA256 signature outputs
# - Determinisme total (tri + seeds fixes)
#
# RUN:
#   streamlit run smaxia_gte_console_v31_10_22.py
#
# Dependencies (typical):
#   pip install streamlit requests beautifulsoup4 pdfplumber pandas numpy scikit-learn
#
# Notes:
# - OCR fallback volontairement "soft": si pytesseract/pdf2image non dispo => quarantaine UNREADABLE
# - Aucun hardcode chapitre/pays: uniquement via Pack JSON. Sans scope rules => POSABLE_SCOPE fail.
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import time
import math
import hashlib
import difflib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Iterable

import streamlit as st
import pandas as pd
import numpy as np

import requests
import pdfplumber
from bs4 import BeautifulSoup

# Optional ML (clustering/features). If missing, fallback to deterministic signature bucketing.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# -----------------------------
# Global constants (Kernel-style)
# -----------------------------
VERSION = "V31.10.22"
KERNEL_VERSION = "V10.6.1"
FINGERPRINT_ALGO = "SHA256"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 25
MAX_PDF_MB = 35

# Determinism
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# -----------------------------
# Reason Codes (Kernel-aligned spirit)
# -----------------------------
RC_CORRIGE_MISSING = "RC_CORRIGE_MISSING"
RC_CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
RC_CORRIGE_MISMATCH = "RC_CORRIGE_MISMATCH"
RC_SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
RC_SCOPE_AMBIGUOUS = "RC_SCOPE_AMBIGUOUS"
RC_NOT_EVALUABLE = "RC_NOT_EVALUABLE"
RC_SUBJECT_UNREADABLE = "RC_SUBJECT_UNREADABLE"
RC_PAIRING_LOW_CONF = "RC_PAIRING_LOW_CONF"
RC_SINGLETON = "RC_SINGLETON_IRREDUCTIBLE"
RC_RECONSTRUCTED_FORBIDDEN = "RC_RECONSTRUCTED_FORBIDDEN"

ATT_PRECOND_FAIL = "ATT_PRECOND_FAIL"
ATT_TRIGGER_MISS = "ATT_TRIGGER_MISS"
ATT_SIGNATURE_MISMATCH = "ATT_SIGNATURE_MISMATCH"
ATT_NEEDS_EXTRA_STEP = "ATT_NEEDS_EXTRA_STEP"
ATT_OUTPUT_TYPE_MISMATCH = "ATT_OUTPUT_TYPE_MISMATCH"

# -----------------------------
# Utility
# -----------------------------
def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return f"sha256:{h.hexdigest()}"

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", errors="ignore"))

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def stable_sort(items: Iterable[Any], key_fn):
    return sorted(list(items), key=key_fn)

# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Locator:
    page_start: int
    page_end: int
    line_start: Optional[int] = None
    line_end: Optional[int] = None

@dataclass
class SourceDoc:
    doc_id: str
    kind: str  # "subject" or "correction"
    url: str
    filename: str
    bytes_sha256: str
    fetched_at: str

@dataclass
class Pair:
    pair_id: str
    subject: SourceDoc
    correction: Optional[SourceDoc]
    pairing_confidence: float
    pairing_evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Atom:
    atom_id: str
    subject_id: str
    correction_id: Optional[str]
    qi_id: str
    rqi_id: Optional[str]
    exo_ref: Optional[str]
    q_ref: Optional[str]
    qi_raw: str
    rqi_raw: Optional[str]
    qi_clean: str
    rqi_clean: Optional[str]
    language_detected: str
    year_ref: Optional[str]
    source_url: str
    source_fingerprint: str
    extraction_locators: Dict[str, Locator]
    sanitizer_derivations: List[Dict[str, Any]]
    pairing_confidence: float
    figure_refs: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PosableDecision:
    qi_id: str
    chapter_ref: Optional[str]
    posable_status: bool
    posable_reason_codes: List[str]
    evidence_refs: List[str]
    scope_trace: Dict[str, Any]

@dataclass
class ChapterMetrics:
    chapter_ref: str
    N_total_posable: int
    quarantined_count: int
    posable_rate: float

@dataclass
class TraceARI:
    qi_id: str
    chapter_ref: str
    actions: List[Dict[str, Any]]  # ordered steps with evidence
    triggers: List[Dict[str, Any]]
    output_type: str
    evidence_refs: List[str]

@dataclass
class ClusterCandidate:
    cluster_id: str
    chapter_ref: str
    qi_ids: List[str]
    n_q_cluster: int
    qi_champion_id: str
    cluster_coherence_score: float
    signature_variance_flag: bool

@dataclass
class AuditEntry:
    check_id: str
    status: str  # PASS/FAIL
    evidence_refs: List[str]
    details: str
    fix_recommendations: List[str]

@dataclass
class QCCandidate:
    qc_id: str
    chapter_ref: str
    qc_text: str
    language: str
    ari_typed: Dict[str, Any]
    frt: Dict[str, Any]
    triggers: List[Dict[str, Any]]
    sig: str
    n_q_cluster: int
    evidence_pack: Dict[str, Any]
    coverage_map_provisional: Dict[str, Any]
    f1: float
    f2: float
    ia2_audit: List[AuditEntry]
    ia2_pass: bool

@dataclass
class SelectionResult:
    chapter_ref: str
    selected_qc_ids: List[str]
    coverage_map_final: Dict[str, Any]  # qi_id -> {qc_id or ORPHAN, evidence_refs, motifs}
    orphans: List[str]
    coverage_kpi: float

# =============================================================================
# Pack loading (no hardcode chapter/pays)
# =============================================================================
def load_pack_from_upload(uploaded_file) -> Dict[str, Any]:
    raw = uploaded_file.read()
    pack = json.loads(raw.decode("utf-8", errors="ignore"))
    pack["_pack_bytes_sha256"] = sha256_bytes(raw)
    pack["_loaded_at"] = now_iso()
    return pack

def pack_get(pack: Dict[str, Any], path: str, default=None):
    cur = pack
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# =============================================================================
# Harvesting (APMEP example) — pack-driven root URL
# =============================================================================
def http_get(url: str) -> requests.Response:
    headers = {"User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r

def is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf")

def normalize_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def harvest_apmep_pairs(pack: Dict[str, Any], year_min: Optional[int], year_max: Optional[int], limit_pairs: int) -> Tuple[List[Pair], Dict[str, Any]]:
    """
    Harvest APMEP annales pages (or any similar index) using pack policy:
      pack.policy.harvest.root_url
      pack.policy.harvest.pdf_link_regex (optional)
      pack.policy.harvest.corrige_keywords (optional)
    """
    root = pack_get(pack, "policy.harvest.root_url")
    if not root:
        raise ValueError("Pack policy.harvest.root_url manquant")

    corrige_kw = pack_get(pack, "policy.harvest.corrige_keywords", ["corrig", "corrigé", "corrige"])
    pdf_regex = pack_get(pack, "policy.harvest.pdf_link_regex")  # optional

    audit = {"root": root, "pages_visited": [], "links_found": 0, "pairs_built": 0, "quarantine": []}

    # APMEP may have multiple year subpages; simplest: fetch root and scan pdf links.
    r = http_get(root)
    audit["pages_visited"].append(root)
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        text = (a.get_text() or "").strip()
        if not href:
            continue
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/"):
            # best-effort
            base = re.match(r"^(https?://[^/]+)", root)
            href = (base.group(1) if base else "") + href

        if ".pdf" in href.lower():
            if pdf_regex:
                if not re.search(pdf_regex, href, flags=re.I):
                    continue
            links.append((href, text))

    audit["links_found"] = len(links)

    # Group likely subject/correction by normalized base name + token similarity.
    # Deterministic: sort by URL.
    links = stable_sort(links, key_fn=lambda x: x[0])

    # Build candidates
    docs = []
    for url, text in links:
        fname = url.split("/")[-1].split("?")[0]
        kind_guess = "correction" if any(k in (text.lower() + " " + fname.lower()) for k in corrige_kw) else "subject"
        # year extraction (best-effort)
        y = None
        m = re.search(r"(19|20)\d{2}", fname)
        if m:
            y = int(m.group(0))
        docs.append({"url": url, "text": text, "filename": fname, "kind_guess": kind_guess, "year": y})

    # Filter by years if provided
    if year_min or year_max:
        docs2 = []
        for d in docs:
            if d["year"] is None:
                docs2.append(d)  # keep unknown year; will be evaluated at pairing
            else:
                if year_min and d["year"] < year_min:
                    continue
                if year_max and d["year"] > year_max:
                    continue
                docs2.append(d)
        docs = docs2

    # Pairing strategy:
    # 1) Create buckets by "base" (remove corrige tokens), then within bucket pick best subject<->corrige.
    def base_key(filename: str) -> str:
        s = filename.lower()
        s = re.sub(r"\.pdf.*$", "", s)
        s = re.sub(r"(corrig[eé]?)", "", s)
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for d in docs:
        buckets.setdefault(base_key(d["filename"]), []).append(d)

    pairs: List[Pair] = []
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})

    # Download until limit reached. Deterministic bucket order.
    for bkey in stable_sort(buckets.keys(), key_fn=lambda x: x):
        if len(pairs) >= limit_pairs:
            break
        items = buckets[bkey]
        subjects = [x for x in items if x["kind_guess"] == "subject"]
        corr = [x for x in items if x["kind_guess"] == "correction"]

        if not subjects:
            continue

        # Choose one subject deterministically (first URL), and best correction by similarity if available.
        subjects = stable_sort(subjects, key_fn=lambda x: x["url"])
        subj = subjects[0]

        corr_best = None
        conf = 0.0
        evidence = {"bucket": bkey, "subject_candidates": len(subjects), "correction_candidates": len(corr)}
        if corr:
            corr = stable_sort(corr, key_fn=lambda x: x["url"])
            # similarity on filenames
            best = None
            for c in corr:
                sim = difflib.SequenceMatcher(None, normalize_name(subj["filename"]), normalize_name(c["filename"])).ratio()
                if best is None or sim > best[0]:
                    best = (sim, c)
            conf = float(best[0])
            corr_best = best[1]
            evidence["filename_similarity"] = conf
        else:
            conf = 0.0

        # Download subject
        def download_doc(d: Dict[str, Any], kind: str) -> SourceDoc:
            rr = sess.get(d["url"], timeout=REQ_TIMEOUT)
            rr.raise_for_status()
            b = rr.content
            if len(b) > MAX_PDF_MB * 1024 * 1024:
                raise ValueError(f"PDF trop volumineux: {d['url']}")
            return SourceDoc(
                doc_id=f"{kind}_{sha256_bytes(b)[7:19]}",
                kind=kind,
                url=d["url"],
                filename=d["filename"],
                bytes_sha256=sha256_bytes(b),
                fetched_at=now_iso()
            ), b

        try:
            subj_doc, subj_bytes = download_doc(subj, "subject")
        except Exception as e:
            audit["quarantine"].append({"url": subj["url"], "reason": str(e)})
            continue

        corr_doc = None
        corr_bytes = None
        if corr_best is not None:
            try:
                corr_doc, corr_bytes = download_doc(corr_best, "correction")
            except Exception as e:
                # keep pair but correction missing
                evidence["corr_download_error"] = str(e)
                corr_doc = None
                corr_bytes = None
                conf = float(conf) * 0.5

        pair_id = f"PAIR_{normalize_name(subj_doc.filename)}_{subj_doc.doc_id[-6:]}"
        pairs.append(Pair(
            pair_id=pair_id,
            subject=subj_doc,
            correction=corr_doc,
            pairing_confidence=conf,
            pairing_evidence=evidence
        ))

        # store bytes in session for later extraction
        st.session_state.setdefault("_doc_bytes", {})[subj_doc.doc_id] = subj_bytes
        if corr_doc and corr_bytes:
            st.session_state["_doc_bytes"][corr_doc.doc_id] = corr_bytes

    audit["pairs_built"] = len(pairs)
    return pairs, audit

# =============================================================================
# PDF text extraction + Atomisation
# =============================================================================
def extract_pdf_pages_text(pdf_bytes: bytes) -> Tuple[List[List[str]], Dict[str, Any]]:
    """
    Returns: pages_lines: list of list-of-lines (per page)
    Audit: extraction stats
    """
    audit = {"pages": 0, "pages_with_text": 0, "empty_pages": 0, "mode": "pdfplumber", "warnings": []}
    pages_lines: List[List[str]] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            audit["pages"] = len(pdf.pages)
            for p in pdf.pages:
                txt = p.extract_text(x_tolerance=2, y_tolerance=2, layout=True) or ""
                lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
                if lines:
                    audit["pages_with_text"] += 1
                else:
                    audit["empty_pages"] += 1
                pages_lines.append(lines)
    except Exception as e:
        audit["warnings"].append(f"pdfplumber_open_failed: {e}")
        return [], audit

    # If mostly empty => likely scanned. We do not hard-depend on OCR libs.
    if audit["pages"] > 0 and audit["pages_with_text"] / max(1, audit["pages"]) < 0.25:
        audit["warnings"].append("low_text_density_likely_scanned")
        # Optional OCR if available
        try:
            import pytesseract  # type: ignore
            from pdf2image import convert_from_bytes  # type: ignore
            audit["mode"] = "ocr_fallback"
            pages_lines = []
            images = convert_from_bytes(pdf_bytes, dpi=250)
            for img in images:
                ocr = pytesseract.image_to_string(img, lang="fra+eng")
                lines = [ln.rstrip() for ln in ocr.splitlines() if ln.strip()]
                pages_lines.append(lines)
            audit["pages_with_text"] = sum(1 for pl in pages_lines if pl)
            audit["empty_pages"] = audit["pages"] - audit["pages_with_text"]
        except Exception as e:
            audit["warnings"].append(f"ocr_unavailable_or_failed: {e}")

    return pages_lines, audit

EXO_RE = re.compile(r"(?i)\bexercice\s+([0-9ivx]+)\b")
QNUM_RE = re.compile(r"^\s*(?:question\s*)?(\d{1,2})\s*[\)\.\:\-]\s+(.+)\s*$", re.I)
SUBQ_RE = re.compile(r"^\s*([a-h])\)\s+(.+)\s*$", re.I)
SUBQ2_RE = re.compile(r"^\s*\(([a-h])\)\s+(.+)\s*$", re.I)

def looks_like_evaluable(line: str) -> bool:
    # Minimal kernel-safe heuristic: task/request indicators (language-level, not chapter hardcode)
    # If pack provides better rule, it will be used at POSABLE_EVALUABLE stage.
    task_verbs = [
        "calcul", "détermin", "resoud", "résoud", "montr", "justif", "démontr", "prouv",
        "établ", "donner", "tracer", "étudier", "simplif", "factoris", "approx",
        "interpr", "conclu", "vérif", "valider"
    ]
    l = line.lower()
    if "?" in line:
        return True
    return any(v in l for v in task_verbs)

def join_lines(lines: List[str]) -> str:
    # Preserve paragraph-ish spacing, but keep it deterministic
    s = "\n".join(lines)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sanitize_text(text: str, lang: str) -> Tuple[str, List[Dict[str, Any]]]:
    deriv = []
    s = text

    # Remove obvious headers/footers artifacts (deterministic regex)
    s2 = re.sub(r"\bpage\s+\d+\s*/\s*\d+\b", " ", s, flags=re.I)
    if s2 != s:
        deriv.append({"op": "rm_page_counter", "before_sha": sha256_text(s), "after_sha": sha256_text(s2)})
        s = s2

    # Neutralize years
    s2 = re.sub(r"\b(19|20)\d{2}\b", "<YEAR>", s)
    if s2 != s:
        deriv.append({"op": "neutralize_year", "before_sha": sha256_text(s), "after_sha": sha256_text(s2)})
        s = s2

    # Neutralize long IDs / centers patterns
    s2 = re.sub(r"\b[A-Z]{2,}\-[A-Z0-9]{2,}\b", "<ID>", s)
    if s2 != s:
        deriv.append({"op": "neutralize_id", "before_sha": sha256_text(s), "after_sha": sha256_text(s2)})
        s = s2

    # Neutralize standalone numbers (keep math structure moderately)
    s2 = re.sub(r"(?<![A-Za-z])(\d+([.,]\d+)?)(?![A-Za-z])", "<NUM>", s)
    if s2 != s:
        deriv.append({"op": "neutralize_numbers", "before_sha": sha256_text(s), "after_sha": sha256_text(s2)})
        s = s2

    # Whitespace normalize
    s2 = re.sub(r"[ \t]+", " ", s).strip()
    if s2 != s:
        deriv.append({"op": "ws_normalize", "before_sha": sha256_text(s), "after_sha": sha256_text(s2)})
        s = s2

    return s, deriv

def atomise_subject(pages_lines: List[List[str]], doc_id: str, lang: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Produce raw question segments with locators:
      {exo_ref, q_ref, qi_raw, locator}
    Captures numbered questions + lettered sub-questions as separate Qi units.
    """
    audit = {
        "doc_id": doc_id,
        "pages": len(pages_lines),
        "exo_detected": 0,
        "q_units": 0,
        "warnings": []
    }

    units: List[Dict[str, Any]] = []

    current_exo = None
    current_qnum = None
    current_unit_lines: List[str] = []
    current_locator = None
    current_qref = None

    def flush_unit():
        nonlocal current_unit_lines, current_locator, current_qref, current_exo
        if not current_unit_lines or not current_locator:
            current_unit_lines = []
            current_locator = None
            return
        raw = join_lines(current_unit_lines)
        # Conservative filter: must look evaluable OR be non-trivial length
        if looks_like_evaluable(raw) or len(raw) >= 60:
            units.append({
                "exo_ref": current_exo,
                "q_ref": current_qref,
                "qi_raw": raw,
                "locator": current_locator
            })
        current_unit_lines = []
        current_locator = None

    for pi, lines in enumerate(pages_lines):
        for li, line in enumerate(lines):
            # Exercice detection
            m_exo = EXO_RE.search(line)
            if m_exo:
                current_exo = str(m_exo.group(1)).upper()
                audit["exo_detected"] += 1

            # Question start (numbered)
            m_q = QNUM_RE.match(line)
            m_sub = SUBQ_RE.match(line) or SUBQ2_RE.match(line)

            if m_q:
                # New numbered question => flush previous unit
                flush_unit()
                current_qnum = m_q.group(1)
                rest = m_q.group(2)
                current_qref = f"{current_qnum}"
                current_unit_lines = [rest] if rest else []
                current_locator = Locator(page_start=pi, page_end=pi, line_start=li, line_end=li)
                continue

            if m_sub and current_qnum is not None:
                # New sub-question => flush previous unit, start new unit with sub label
                flush_unit()
                sub = m_sub.group(1).lower()
                rest = m_sub.group(2)
                current_qref = f"{current_qnum}{sub}"
                current_unit_lines = [rest] if rest else []
                current_locator = Locator(page_start=pi, page_end=pi, line_start=li, line_end=li)
                continue

            # Accumulate into current unit if active
            if current_locator is not None:
                current_unit_lines.append(line)
                # extend locator (multi-line, multi-page)
                current_locator.page_end = pi
                current_locator.line_end = li

    flush_unit()

    audit["q_units"] = len(units)
    if len(units) == 0:
        audit["warnings"].append("no_question_units_detected")

    return units, audit

def atomise_correction(pages_lines: List[List[str]], doc_id: str, lang: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Correction segmentation indexed by (exo_ref, q_ref) where possible.
    Returns dict key -> {rqi_raw, locator}
    """
    audit = {"doc_id": doc_id, "pages": len(pages_lines), "segments": 0, "warnings": []}
    idx: Dict[str, Dict[str, Any]] = {}

    current_exo = None
    current_qnum = None
    current_key = None
    cur_lines: List[str] = []
    cur_loc = None

    def flush():
        nonlocal current_key, cur_lines, cur_loc
        if current_key and cur_lines and cur_loc:
            idx[current_key] = {"rqi_raw": join_lines(cur_lines), "locator": cur_loc}
        current_key = None
        cur_lines = []
        cur_loc = None

    for pi, lines in enumerate(pages_lines):
        for li, line in enumerate(lines):
            m_exo = EXO_RE.search(line)
            if m_exo:
                current_exo = str(m_exo.group(1)).upper()

            m_q = QNUM_RE.match(line)
            m_sub = SUBQ_RE.match(line) or SUBQ2_RE.match(line)

            if m_q:
                flush()
                current_qnum = m_q.group(1)
                rest = m_q.group(2)
                q_ref = f"{current_qnum}"
                current_key = f"{current_exo or 'X'}::{q_ref}"
                cur_lines = [rest] if rest else []
                cur_loc = Locator(page_start=pi, page_end=pi, line_start=li, line_end=li)
                continue

            if m_sub and current_qnum is not None:
                flush()
                sub = (m_sub.group(1) or "").lower()
                rest = m_sub.group(2)
                q_ref = f"{current_qnum}{sub}"
                current_key = f"{current_exo or 'X'}::{q_ref}"
                cur_lines = [rest] if rest else []
                cur_loc = Locator(page_start=pi, page_end=pi, line_start=li, line_end=li)
                continue

            if cur_loc is not None:
                cur_lines.append(line)
                cur_loc.page_end = pi
                cur_loc.line_end = li

    flush()
    audit["segments"] = len(idx)
    if len(idx) == 0:
        audit["warnings"].append("no_correction_segments_detected")

    return idx, audit

def best_effort_match_rqi(qi_raw: str, corr_idx: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], float, Optional[Dict[str, Any]]]:
    """
    Fallback matching when numbering keys fail: pick segment with highest similarity on sanitized surface.
    This does NOT create content; it only selects an existing segment.
    """
    if not corr_idx:
        return None, 0.0, None
    q = re.sub(r"\s+", " ", qi_raw).strip().lower()
    best = (0.0, None, None)
    for k, v in corr_idx.items():
        s = re.sub(r"\s+", " ", v["rqi_raw"]).strip().lower()
        sim = difflib.SequenceMatcher(None, q[:1200], s[:1200]).ratio()
        if sim > best[0]:
            best = (sim, k, v)
    if best[1] is None:
        return None, 0.0, None
    return best[2]["rqi_raw"], float(best[0]), {"match_key": best[1], "sim": float(best[0])}

# =============================================================================
# POSABLE Gate (corrigé + scope + évaluable)
# =============================================================================
def scope_match_chapter(pack: Dict[str, Any], qi_clean: str, rqi_clean: Optional[str]) -> Tuple[Optional[str], Dict[str, Any], List[str]]:
    """
    Pack-driven scope rules.
    Expected pack.academic.chapters = [{chapter_code, label, match_keywords[], match_regexes[]}]
    Returns: chapter_ref, scope_trace, reason_codes(if fail)
    """
    chapters = pack_get(pack, "academic.chapters", [])
    if not chapters:
        return None, {"method": "none", "details": "pack academic.chapters missing"}, [RC_SCOPE_UNRESOLVED]

    text = (qi_clean + "\n" + (rqi_clean or "")).lower()

    scores = []
    for ch in chapters:
        code = ch.get("chapter_code") or ch.get("code")
        if not code:
            continue
        kw = ch.get("match_keywords", []) or []
        rg = ch.get("match_regexes", []) or []
        hit_kw = [k for k in kw if isinstance(k, str) and k.lower() in text]
        hit_rg = []
        for r in rg:
            try:
                if re.search(r, text, flags=re.I):
                    hit_rg.append(r)
            except Exception:
                continue
        score = len(hit_kw) + 2 * len(hit_rg)
        if score > 0:
            scores.append((score, code, hit_kw, hit_rg))

    scores = stable_sort(scores, key_fn=lambda x: (-x[0], x[1]))

    if not scores:
        return None, {"method": "keywords/regex", "hits": []}, [RC_SCOPE_UNRESOLVED]

    # Ambiguity: top two close
    if len(scores) >= 2 and scores[0][0] == scores[1][0]:
        return None, {"method": "keywords/regex", "top": scores[:3]}, [RC_SCOPE_AMBIGUOUS]

    top = scores[0]
    trace = {"method": "keywords/regex", "chapter_ref": top[1], "score": top[0], "hit_keywords": top[2], "hit_regexes": top[3]}
    return top[1], trace, []

def posable_gate(pack: Dict[str, Any], atoms: List[Atom]) -> Tuple[Dict[str, PosableDecision], Dict[str, ChapterMetrics]]:
    """
    Determine POSABLE per Atom/Qi:
      - POSABLE_CORRIGE: rqi_clean exists
      - POSABLE_SCOPE: chapter resolvable (pack rules)
      - POSABLE_EVALUABLE: minimal task criterion (pack may override)
    """
    decisions: Dict[str, PosableDecision] = {}
    chapter_counts: Dict[str, Dict[str, int]] = {}

    # Pack may provide a regex list for evaluable detection
    eval_rules = pack_get(pack, "policy.posable.evaluable_regexes", None)

    for a in atoms:
        reason = []
        evidence_refs = [a.source_fingerprint]  # minimal; per-atom evidence pack will add more
        chapter_ref = None
        scope_trace = {}

        # POSABLE_CORRIGE
        if not a.rqi_clean:
            reason.append(RC_CORRIGE_MISSING)

        # POSABLE_EVALUABLE
        is_eval = looks_like_evaluable(a.qi_raw)
        if eval_rules:
            try:
                is_eval = any(re.search(rgx, a.qi_raw, flags=re.I) for rgx in eval_rules)
            except Exception:
                pass
        if not is_eval:
            reason.append(RC_NOT_EVALUABLE)

        # POSABLE_SCOPE (only if corrigé exists; still compute trace)
        chapter_ref, scope_trace, scope_rc = scope_match_chapter(pack, a.qi_clean, a.rqi_clean)
        reason.extend(scope_rc)

        posable = (len(reason) == 0)

        decisions[a.qi_id] = PosableDecision(
            qi_id=a.qi_id,
            chapter_ref=chapter_ref,
            posable_status=posable,
            posable_reason_codes=reason,
            evidence_refs=evidence_refs,
            scope_trace=scope_trace
        )

        ch = chapter_ref or "UNSCOPED"
        chapter_counts.setdefault(ch, {"posable": 0, "quarantined": 0})
        if posable and chapter_ref:
            chapter_counts[ch]["posable"] += 1
        else:
            chapter_counts[ch]["quarantined"] += 1

    chapter_metrics: Dict[str, ChapterMetrics] = {}
    for ch, cnt in chapter_counts.items():
        total = cnt["posable"] + cnt["quarantined"]
        chapter_metrics[ch] = ChapterMetrics(
            chapter_ref=ch,
            N_total_posable=int(cnt["posable"]),
            quarantined_count=int(cnt["quarantined"]),
            posable_rate=float(cnt["posable"]) / max(1, total)
        )

    return decisions, chapter_metrics

# =============================================================================
# Cognitive Verb Ontology (Kernel minimal) + Pack mappings
# =============================================================================
KERNEL_VERBS = [
    ("IDENTIFY", ["identifier", "repérer", "nommer"], 0.15),
    ("DEFINE", ["définir", "rappeler"], 0.15),
    ("ANALYZE", ["analyser", "observer", "examiner"], 0.20),
    ("SYNTHESIZE", ["synthétiser", "conclure"], 0.20),
    ("ILLUSTRATE", ["illustrer", "exemplifier"], 0.25),
    ("SIMPLIFY", ["simplifier", "factoriser", "réduire"], 0.25),
    ("MODEL", ["schématiser", "modéliser"], 0.30),
    ("PLAN", ["planifier", "organiser"], 0.30),
    ("CALCULATE", ["calculer", "mesurer", "quantifier"], 0.30),
    ("FORMULATE", ["exprimer", "formuler"], 0.30),
    ("COMPARE", ["comparer", "opposer", "distinguer"], 0.35),
    ("APPLY", ["appliquer", "utiliser"], 0.35),
    ("INTERPRET", ["interpréter"], 0.35),
    ("SOLVE", ["résoudre", "déterminer"], 0.40),
    ("DIFF_INTEG", ["dériver", "intégrer"], 0.40),
    ("ARGUE", ["argumenter", "justifier"], 0.45),
    ("PROVE", ["démontrer", "prouver"], 0.50),
    ("RECURRENCE", ["récurrence"], 0.60),
]

def build_verb_map(pack: Dict[str, Any], lang: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping from surface token -> {verb_id, weight}
    Pack may provide:
      pack.policy.verb_ontology.synonyms = { "differencier": "DIFF_INTEG", ... }
      pack.policy.verb_ontology.weights_override = { "DIFF_INTEG": 0.42, ... }
    """
    weights_override = pack_get(pack, "policy.verb_ontology.weights_override", {}) or {}
    synonyms = pack_get(pack, "policy.verb_ontology.synonyms", {}) or {}

    verb_map: Dict[str, Dict[str, Any]] = {}
    for vid, forms, w in KERNEL_VERBS:
        w2 = float(weights_override.get(vid, w))
        for f in forms:
            verb_map[f.lower()] = {"verb_id": vid, "weight": w2}

    # Add pack synonyms
    for surf, vid in synonyms.items():
        if isinstance(surf, str) and isinstance(vid, str):
            base_w = next((x[2] for x in KERNEL_VERBS if x[0] == vid), 0.25)
            w2 = float(weights_override.get(vid, base_w))
            verb_map[surf.lower()] = {"verb_id": vid, "weight": w2}

    return verb_map

def mine_traces(pack: Dict[str, Any], atoms: List[Atom], posable: Dict[str, PosableDecision], lang: str) -> Dict[str, TraceARI]:
    """
    IA1 Miner (factuel): actions observed + evidence refs, no synthesis.
    """
    verb_map = build_verb_map(pack, lang)
    traces: Dict[str, TraceARI] = {}

    for a in atoms:
        dec = posable.get(a.qi_id)
        if not dec or not dec.posable_status or not dec.chapter_ref:
            continue

        text = (a.rqi_clean or "").lower()

        actions = []
        used = set()
        for surf, meta in verb_map.items():
            # deterministic: scan for word boundaries
            if re.search(rf"\b{re.escape(surf)}\b", text, flags=re.I):
                vid = meta["verb_id"]
                if vid in used:
                    continue
                # Evidence snippet (limited)
                m = re.search(rf"(.{{0,40}})\b{re.escape(surf)}\b(.{{0,40}})", text, flags=re.I)
                snippet = (m.group(0) if m else surf)[:120]
                actions.append({
                    "verb_id": vid,
                    "surface": surf,
                    "weight": meta["weight"],
                    "evidence": snippet
                })
                used.add(vid)

        # Ensure deterministic order: by first appearance index in text
        def first_pos(surf: str) -> int:
            m = re.search(rf"\b{re.escape(surf)}\b", text, flags=re.I)
            return m.start() if m else 10**9
        actions = stable_sort(actions, key_fn=lambda x: (first_pos(x["surface"]), x["verb_id"]))

        # Triggers (simple: based on actions)
        triggers = [{"trigger_id": f"TRG_{ac['verb_id']}", "verb_id": ac["verb_id"], "evidence": ac["evidence"]} for ac in actions]

        output_type = pack_get(pack, "policy.outputs.default_output_type", "GENERIC")

        traces[a.qi_id] = TraceARI(
            qi_id=a.qi_id,
            chapter_ref=dec.chapter_ref,
            actions=actions,
            triggers=triggers,
            output_type=str(output_type),
            evidence_refs=[a.source_fingerprint]
        )

    return traces

# =============================================================================
# Clustering intra-chapitre (features on ARI signals, not Qi raw text)
# =============================================================================
def trace_signature(tr: TraceARI) -> str:
    # Deterministic signature: ordered verb_ids
    vids = [a["verb_id"] for a in tr.actions]
    return "SIG:" + "-".join(vids) if vids else "SIG:EMPTY"

def build_clusters(pack: Dict[str, Any], traces: Dict[str, TraceARI]) -> List[ClusterCandidate]:
    """
    Cluster per chapter using ARI signals.
    If sklearn available, use TF-IDF on verb_id sequences; else bucket by signature.
    """
    by_ch: Dict[str, List[TraceARI]] = {}
    for tr in traces.values():
        by_ch.setdefault(tr.chapter_ref, []).append(tr)

    clusters: List[ClusterCandidate] = []
    for ch in stable_sort(by_ch.keys(), key_fn=lambda x: x):
        trs = stable_sort(by_ch[ch], key_fn=lambda t: t.qi_id)
        if len(trs) == 0:
            continue

        # Represent each trace as a "document" of verb_ids
        docs = [" ".join([a["verb_id"] for a in tr.actions]) for tr in trs]
        qi_ids = [tr.qi_id for tr in trs]

        if SKLEARN_OK and len(trs) >= 4:
            # Params from pack
            dist_thr = float(pack_get(pack, "policy.clustering.distance_threshold", 0.55))
            vect = TfidfVectorizer(token_pattern=r"(?u)\b[A-Z_]+\b")
            X = vect.fit_transform(docs)
            # Agglomerative clustering on cosine distance
            sim = cosine_similarity(X)
            # distance = 1 - sim
            dist = 1.0 - sim
            # sklearn expects condensed? We'll use precomputed affinity by converting to linkage on distance threshold via average linkage
            # AgglomerativeClustering supports metric='precomputed' in newer versions; handle both.
            try:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    metric="precomputed",
                    linkage="average",
                    distance_threshold=dist_thr,
                )
                labels = model.fit_predict(dist)
            except Exception:
                # fallback: bucket by raw signature (still deterministic)
                labels = None

            if labels is None:
                # signature buckets
                buckets: Dict[str, List[str]] = {}
                for tr in trs:
                    buckets.setdefault(trace_signature(tr), []).append(tr.qi_id)
                for bi, sig in enumerate(stable_sort(buckets.keys(), key_fn=lambda x: x)):
                    ids = stable_sort(buckets[sig], key_fn=lambda x: x)
                    clusters.append(ClusterCandidate(
                        cluster_id=f"{ch}::BKT::{bi:03d}",
                        chapter_ref=ch,
                        qi_ids=ids,
                        n_q_cluster=len(ids),
                        qi_champion_id=ids[0],
                        cluster_coherence_score=1.0,
                        signature_variance_flag=False
                    ))
            else:
                # build cluster candidates
                label_set = stable_sort(set(labels.tolist()), key_fn=lambda x: int(x))
                for lab in label_set:
                    members = [qi_ids[i] for i in range(len(qi_ids)) if int(labels[i]) == int(lab)]
                    members = stable_sort(members, key_fn=lambda x: x)
                    # coherence heuristic: average intra-cluster similarity
                    idxs = [qi_ids.index(m) for m in members]
                    if len(idxs) >= 2:
                        sub = sim[np.ix_(idxs, idxs)]
                        coh = float((sub.sum() - len(idxs)) / max(1, (len(idxs) * (len(idxs) - 1))))
                    else:
                        coh = 0.0
                    clusters.append(ClusterCandidate(
                        cluster_id=f"{ch}::CL::{int(lab):03d}",
                        chapter_ref=ch,
                        qi_ids=members,
                        n_q_cluster=len(members),
                        qi_champion_id=members[0],
                        cluster_coherence_score=coh,
                        signature_variance_flag=False
                    ))
        else:
            # Pure deterministic buckets by signature
            buckets: Dict[str, List[str]] = {}
            for tr in trs:
                buckets.setdefault(trace_signature(tr), []).append(tr.qi_id)
            for bi, sig in enumerate(stable_sort(buckets.keys(), key_fn=lambda x: x)):
                ids = stable_sort(buckets[sig], key_fn=lambda x: x)
                clusters.append(ClusterCandidate(
                    cluster_id=f"{ch}::BKT::{bi:03d}",
                    chapter_ref=ch,
                    qi_ids=ids,
                    n_q_cluster=len(ids),
                    qi_champion_id=ids[0],
                    cluster_coherence_score=1.0,
                    signature_variance_flag=False
                ))

    return clusters

# =============================================================================
# IA1 Builder: QC + ARI + FRT + EvidencePack + SIG
# =============================================================================
def qc_format(pack: Dict[str, Any], lang: str, core: str) -> str:
    prefix = pack_get(pack, "policy.qc_format.prefix", "Comment ")
    suffix = pack_get(pack, "policy.qc_format.suffix", " ?")
    s = f"{prefix}{core}".strip()
    if not s.endswith("?") and suffix.strip().endswith("?"):
        s = s.rstrip(" ?") + suffix
    else:
        s = s + (suffix if not s.endswith(suffix.strip()) else "")
    return s.strip()

def pick_qc_core(pack: Dict[str, Any], lang: str, actions: List[Dict[str, Any]]) -> str:
    """
    Pack may provide mapping from verb_id -> phrase, else minimal deterministic mapping.
    """
    m = pack_get(pack, "policy.qc_phrases.by_verb_id", {}) or {}
    if actions:
        top = actions[0]["verb_id"]
        phrase = m.get(top)
        if isinstance(phrase, str) and phrase.strip():
            return phrase.strip()
        # fallback phrases (language-level, not chapter-level)
        fallback = {
            "DIFF_INTEG": "dériver ou intégrer une expression",
            "SOLVE": "résoudre une équation ou un problème",
            "CALCULATE": "effectuer un calcul",
            "PROVE": "démontrer un résultat",
            "APPLY": "appliquer une règle ou un théorème",
            "ANALYZE": "analyser une situation et conclure",
        }
        return fallback.get(top, "résoudre un exercice standard")
    return "résoudre un exercice standard"

def build_frt(pack: Dict[str, Any], lang: str, trace_list: List[TraceARI], atoms_by_qi: Dict[str, Atom]) -> Dict[str, Any]:
    """
    FRT template mandatory sections:
      Usage / Preconditions / ARI typed / Transitions & checkpoints / Pitfalls / Expected output / Evidence
    Pack may provide operator templates by output_type or by dominant verb_id.
    """
    # Evidence: list of source fingerprints + snippets
    evid = []
    for tr in trace_list:
        a = atoms_by_qi.get(tr.qi_id)
        if not a:
            continue
        evid.append({
            "qi_id": tr.qi_id,
            "source_fp": a.source_fingerprint,
            "rqi_snippet": (a.rqi_clean or "")[:220]
        })

    # Build typed ARI (normalized IDs)
    # Deterministic: keep union of steps in majority order; use first trace as spine.
    spine = trace_list[0].actions if trace_list and trace_list[0].actions else []
    ari_steps = []
    for i, ac in enumerate(spine):
        ari_steps.append({
            "step": i + 1,
            "verb_id": ac["verb_id"],
            "label": ac["verb_id"],  # label can be localized by pack at UI time
            "evidence": ac.get("evidence", "")[:120]
        })

    frt = {
        "usage": "Procédure type applicable aux questions partageant la même structure de résolution (ARI).",
        "preconditions": [
            "Énoncé compris après sanitization (aucune constante locale nécessaire).",
            "Données nécessaires disponibles dans l’énoncé.",
        ],
        "ari_typed": ari_steps,
        "transitions_checkpoints": [
            "Après chaque étape, vérifier la cohérence des hypothèses et des transformations.",
            "Valider les conditions d’application (domaines, contraintes, existence)."
        ],
        "pitfalls_fatal": [
            "Oublier une condition d’application.",
            "Introduire une constante locale (interdit).",
            "Ajouter une étape non supportée par les corrigés (RECONSTRUCTED interdit en CAS 1)."
        ],
        "expected_output": "Résultat final conforme au corrigé de référence (après sanitization).",
        "evidence": evid
    }

    # Template override by pack (operator-specific)
    tpl = pack_get(pack, "policy.frt_templates.by_output_type", {}) or {}
    if trace_list:
        ot = trace_list[0].output_type
        if isinstance(tpl.get(ot), dict):
            # shallow merge deterministically
            frt = {**frt, **tpl[ot]}

    return frt

def build_qc_candidates(
    pack: Dict[str, Any],
    atoms: List[Atom],
    posable: Dict[str, PosableDecision],
    traces: Dict[str, TraceARI],
    clusters: List[ClusterCandidate],
    lang: str
) -> List[QCCandidate]:
    atoms_by_qi = {a.qi_id: a for a in atoms}
    candidates: List[QCCandidate] = []

    # level coefficient delta_c from pack (secret; if missing, default 1.0)
    delta_c = float(pack_get(pack, "policy.level.delta_c", 1.0))
    eps = float(pack_get(pack, "policy.scoring.eps", 1e-6))

    # historical occurrence index optional
    hist = pack_get(pack, "history.sig_occurrences", {}) or {}
    alpha = float(pack_get(pack, "policy.scoring.alpha_recency", 0.0))

    for cl in clusters:
        # anti-singleton strictly enforced at IA2 stage, but we can pre-mark
        qi_ids = [qid for qid in cl.qi_ids if qid in traces]
        if not qi_ids:
            continue
        trace_list = [traces[qid] for qid in qi_ids]
        trace_list = stable_sort(trace_list, key_fn=lambda t: t.qi_id)

        # Champion: deterministic (cluster's champion id)
        champ = traces.get(cl.qi_champion_id) or trace_list[0]

        # QC core phrase from actions
        core = pick_qc_core(pack, lang, champ.actions)
        qc_text = qc_format(pack, lang, core)

        # ARI typed (use champion spine)
        ari = {
            "typed_only": True,
            "steps": [{
                "step": i + 1,
                "verb_id": ac["verb_id"],
                "weight": float(ac.get("weight", 0.25)),
                "evidence": ac.get("evidence", "")[:120]
            } for i, ac in enumerate(champ.actions)]
        }

        # Triggers union (deterministic)
        trg = []
        seen = set()
        for tr in trace_list:
            for t in tr.triggers:
                tid = t.get("trigger_id")
                if not tid or tid in seen:
                    continue
                trg.append(t)
                seen.add(tid)
        trg = stable_sort(trg, key_fn=lambda x: x.get("trigger_id", ""))

        frt = build_frt(pack, lang, trace_list, atoms_by_qi)

        # SIG(q): signature of champion
        sig = trace_signature(champ)

        # EvidencePack minimal
        sources = []
        extractions = []
        derivations = []
        for qid in qi_ids:
            a = atoms_by_qi.get(qid)
            if not a:
                continue
            sources.append({
                "url": a.source_url,
                "fingerprint": a.source_fingerprint,
                "doc_kind": "pair_subject+correction",
            })
            extractions.append({
                "qi_id": a.qi_id,
                "qi_locator": asdict(a.extraction_locators.get("qi")) if a.extraction_locators.get("qi") else None,
                "rqi_locator": asdict(a.extraction_locators.get("rqi")) if a.extraction_locators.get("rqi") else None,
            })
            derivations.extend(a.sanitizer_derivations)

        evidence_pack = {
            "kernel_version": KERNEL_VERSION,
            "pack_fingerprint": pack.get("_pack_bytes_sha256", ""),
            "sources": sources,
            "extractions": extractions,
            "derivations": derivations,
            "fingerprints": [x.get("fingerprint") for x in sources if x.get("fingerprint")],
            "timestamps": [now_iso()],
        }

        # Provisional coverage map: attach only cluster members for now (expanded later at selection)
        coverage_prov = {
            "cover_set": qi_ids,
            "attach_rules": "cluster_members_initial"
        }

        # ---- Scoring (deterministic, kernel-consistent in spirit; exact parameters pack-driven)
        # F1: normalized cognitive weight sum (intra-chapitre normalization done later if needed)
        raw = sum(float(s["weight"]) for s in ari["steps"])
        f1_raw = delta_c * raw
        f1 = f1_raw  # normalization done per chapter after IA2 PASS if desired

        # F2: popularity/recency/anti-redundancy (anti-redondance applied during greedy selection)
        n_hist = float(hist.get(sig, 0.0))
        rec = 0.0  # if pack provides last_seen timestamps, can compute
        f2 = (n_hist + alpha * rec + eps) * (f1 + eps)

        # IA2 audit placeholder; actual audit computed next
        candidates.append(QCCandidate(
            qc_id=f"QC_{sha256_text(cl.cluster_id + sig)[7:19]}",
            chapter_ref=cl.chapter_ref,
            qc_text=qc_text,
            language=lang,
            ari_typed=ari,
            frt=frt,
            triggers=trg,
            sig=sig,
            n_q_cluster=cl.n_q_cluster,
            evidence_pack=evidence_pack,
            coverage_map_provisional=coverage_prov,
            f1=float(f1),
            f2=float(f2),
            ia2_audit=[],
            ia2_pass=False
        ))

    return candidates

# =============================================================================
# IA2 Judge (boolean checks, CAS 1 ONLY)
# =============================================================================
def ia2_checks(pack: Dict[str, Any], qc: QCCandidate) -> Tuple[bool, List[AuditEntry]]:
    audits: List[AuditEntry] = []

    def add(check_id: str, ok: bool, details: str, fixes: List[str], evidence: Optional[List[str]] = None):
        audits.append(AuditEntry(
            check_id=check_id,
            status="PASS" if ok else "FAIL",
            evidence_refs=evidence or [],
            details=details,
            fix_recommendations=fixes if not ok else []
        ))

    # QC_FORM: must end with ? and (for FR) start with "Comment"
    qc_prefix = pack_get(pack, "policy.qc_format.prefix", "Comment ")
    ok_form = qc.qc_text.strip().endswith("?") and qc.qc_text.strip().startswith(qc_prefix)
    add("QC_FORM", ok_form, "QC UI format check", ["Ensure pack qc_format prefix/suffix applied correctly"])

    # NO_LOCAL_CONSTANTS (heuristic): reject if obvious year/center remains
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", qc.qc_text))
    has_num = bool(re.search(r"\b\d+\b", qc.qc_text))
    ok_local = (not has_year) and (not has_num)
    add("NO_LOCAL_CONSTANTS", ok_local, "QC must not contain local constants", ["Improve sanitization or QC core phrase selection"])

    # FRT_TEMPLATE_OK: mandatory keys
    mandatory = ["usage", "preconditions", "ari_typed", "transitions_checkpoints", "pitfalls_fatal", "expected_output", "evidence"]
    ok_frt = all(k in qc.frt for k in mandatory)
    add("FRT_TEMPLATE_OK", ok_frt, "FRT mandatory sections present", ["Ensure FRT builder includes all required sections"])

    # ARI_TYPED_ONLY
    ok_ari = bool(qc.ari_typed.get("typed_only", False)) and isinstance(qc.ari_typed.get("steps", []), list)
    add("ARI_TYPED_ONLY", ok_ari, "ARI typed-only check", ["Ensure ARI uses canonical verb IDs only"])

    # NO_RECONSTRUCTION (CAS 1): we cannot prove reconstruction here; enforce cluster evidence exists
    ok_evid = bool(qc.evidence_pack.get("extractions")) and bool(qc.evidence_pack.get("sources"))
    add("NO_RECONSTRUCTION", ok_evid, "EvidencePack must contain sources/extractions", ["Attach proper locators and extracted segments to EvidencePack"])

    # Anti-singleton
    ok_cluster = qc.n_q_cluster >= 2
    add("ANTI_SINGLETON", ok_cluster, "n_q_cluster >= 2 required to canonize QC", ["Increase volume or clustering coherence; singletons must go to residual/quarantine"])

    passed = all(a.status == "PASS" for a in audits)
    return passed, audits

# =============================================================================
# Attach/Cover/Coverage + selection (coverage-driven)
# =============================================================================
def attach_decision(tr_qi: TraceARI, qc: QCCandidate) -> Tuple[bool, str]:
    # Attach = TRUE iff triggers compatible AND ARI spine compatible (no extra steps)
    qc_steps = [s["verb_id"] for s in qc.ari_typed.get("steps", [])]
    qi_steps = [a["verb_id"] for a in tr_qi.actions]

    # triggers: qi must be subset of qc triggers (by verb_id)
    qc_trg_verbs = set([t.get("verb_id") for t in qc.triggers if t.get("verb_id")])
    if any(v not in qc_trg_verbs for v in qi_steps):
        return False, ATT_TRIGGER_MISS

    # ARI spine compatibility: qi_steps must be subset (order-preserving) of qc_steps
    it = iter(qc_steps)
    ok_order = True
    for v in qi_steps:
        found = False
        for vv in it:
            if vv == v:
                found = True
                break
        if not found:
            ok_order = False
            break
    if not ok_order:
        return False, ATT_NEEDS_EXTRA_STEP

    return True, "OK"

def greedy_select_cover(pack: Dict[str, Any], chapter_ref: str, qcs: List[QCCandidate], traces: Dict[str, TraceARI]) -> SelectionResult:
    # Universe POSABLE for chapter
    universe = [tr.qi_id for tr in traces.values() if tr.chapter_ref == chapter_ref]
    universe = stable_sort(universe, key_fn=lambda x: x)
    uncovered = set(universe)

    # Precompute cover sets deterministically
    cover_sets: Dict[str, List[str]] = {}
    motifs: Dict[Tuple[str, str], str] = {}  # (qi_id, qc_id) -> motif
    for qc in qcs:
        cov = []
        for qi_id in universe:
            ok, reason = attach_decision(traces[qi_id], qc)
            if ok:
                cov.append(qi_id)
            else:
                motifs[(qi_id, qc.qc_id)] = reason
        cover_sets[qc.qc_id] = stable_sort(cov, key_fn=lambda x: x)

    selected: List[str] = []
    coverage_map: Dict[str, Any] = {qi: {"status": "ORPHAN", "qc_id": None, "evidence_refs": [], "motifs": []} for qi in universe}

    # Anti-redundancy uses cosine on step vectors (if sklearn ok)
    def step_vector(qc: QCCandidate) -> Dict[str, float]:
        vec = {}
        for s in qc.ari_typed.get("steps", []):
            vid = s["verb_id"]
            vec[vid] = vec.get(vid, 0.0) + 1.0
        return vec

    qc_vec = {qc.qc_id: step_vector(qc) for qc in qcs}

    def cosine_vec(a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a.keys()) | set(b.keys())
        va = np.array([a.get(k, 0.0) for k in keys], dtype=float)
        vb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na <= 1e-9 or nb <= 1e-9:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def anti_redondance(qc_id: str, selected_ids: List[str]) -> float:
        # product (1 - sim) with already selected
        prod = 1.0
        for sid in selected_ids:
            sim = cosine_vec(qc_vec[qc_id], qc_vec[sid])
            prod *= (1.0 - clamp(sim, 0.0, 0.95))
        return float(prod)

    # Selection loop
    max_k = int(pack_get(pack, "policy.selection.max_k_per_chapter", 9999))
    while uncovered and len(selected) < max_k:
        best = None
        for qc in qcs:
            if not qc.ia2_pass:
                continue
            if qc.qc_id in selected:
                continue
            newly = [qi for qi in cover_sets[qc.qc_id] if qi in uncovered]
            gain = len(newly)
            if gain <= 0:
                continue
            ar = anti_redondance(qc.qc_id, selected)
            score = (gain, qc.f2 * ar, qc.qc_id)  # deterministic tie-break
            if best is None or score > best[0]:
                best = (score, qc, newly)

        if best is None:
            break

        _, qc_best, newly = best
        selected.append(qc_best.qc_id)
        for qi in newly:
            coverage_map[qi] = {"status": "COVERED", "qc_id": qc_best.qc_id, "evidence_refs": qc_best.evidence_pack.get("fingerprints", [])[:3], "motifs": []}
            uncovered.discard(qi)

    # Orphans
    orphans = stable_sort(list(uncovered), key_fn=lambda x: x)

    # Coverage KPI
    covered = sum(1 for qi in universe if coverage_map[qi]["status"] == "COVERED")
    kpi = float(covered) / max(1, len(universe))

    # Add motifs for uncovered (optional, deterministic)
    for qi in orphans:
        # record top 3 failure motifs against best 3 qc by f2
        ranked = stable_sort([qc for qc in qcs if qc.ia2_pass], key_fn=lambda q: (-q.f2, q.qc_id))
        top = []
        for qc in ranked[:3]:
            top.append({"qc_id": qc.qc_id, "reason": motifs.get((qi, qc.qc_id), "NO_MATCH")})
        coverage_map[qi]["motifs"] = top

    return SelectionResult(
        chapter_ref=chapter_ref,
        selected_qc_ids=selected,
        coverage_map_final=coverage_map,
        orphans=orphans,
        coverage_kpi=kpi
    )

# =============================================================================
# End-to-end pipeline for a given volume
# =============================================================================
def build_atoms_from_pair(pack: Dict[str, Any], pair: Pair, lang: str) -> Tuple[List[Atom], Dict[str, Any]]:
    """
    Atomisation + Sanitization + Qi<->RQi matching (no creation).
    """
    audit = {"pair_id": pair.pair_id, "pairing_confidence": pair.pairing_confidence, "steps": [], "quarantine": []}
    atoms: List[Atom] = []

    # Must have correction in CAS 1 ONLY, else quarantine all
    if pair.correction is None:
        audit["quarantine"].append({"pair_id": pair.pair_id, "reason": RC_CORRIGE_MISSING})
        return [], audit

    doc_bytes = st.session_state.get("_doc_bytes", {})
    subj_b = doc_bytes.get(pair.subject.doc_id)
    corr_b = doc_bytes.get(pair.correction.doc_id) if pair.correction else None

    if not subj_b:
        audit["quarantine"].append({"subject_doc_id": pair.subject.doc_id, "reason": RC_SUBJECT_UNREADABLE})
        return [], audit
    if not corr_b:
        audit["quarantine"].append({"correction_doc_id": pair.correction.doc_id, "reason": RC_CORRIGE_UNREADABLE})
        return [], audit

    subj_pages, subj_a = extract_pdf_pages_text(subj_b)
    corr_pages, corr_a = extract_pdf_pages_text(corr_b)
    audit["steps"].append({"subject_extraction": subj_a})
    audit["steps"].append({"correction_extraction": corr_a})

    if len(subj_pages) == 0 or ("low_text_density_likely_scanned" in subj_a.get("warnings", []) and subj_a.get("mode") != "ocr_fallback"):
        audit["quarantine"].append({"pair_id": pair.pair_id, "reason": RC_SUBJECT_UNREADABLE, "details": subj_a})
        return [], audit
    if len(corr_pages) == 0 or ("low_text_density_likely_scanned" in corr_a.get("warnings", []) and corr_a.get("mode") != "ocr_fallback"):
        audit["quarantine"].append({"pair_id": pair.pair_id, "reason": RC_CORRIGE_UNREADABLE, "details": corr_a})
        return [], audit

    subj_units, subj_unit_a = atomise_subject(subj_pages, pair.subject.doc_id, lang)
    corr_idx, corr_unit_a = atomise_correction(corr_pages, pair.correction.doc_id, lang)
    audit["steps"].append({"subject_atomisation": subj_unit_a})
    audit["steps"].append({"correction_atomisation": corr_unit_a})

    # Year ref best-effort from filename
    year_ref = None
    m = re.search(r"(19|20)\d{2}", pair.subject.filename)
    if m:
        year_ref = m.group(0)

    # Build atoms: for each subject Qi unit, try exact key match (exo_ref::q_ref), else similarity fallback
    for i, u in enumerate(subj_units):
        exo_ref = u.get("exo_ref") or "X"
        q_ref = u.get("q_ref") or f"Q{i+1}"
        qi_raw = u["qi_raw"]
        key = f"{exo_ref}::{q_ref}"
        rqi_raw = None
        rqi_loc = None
        match_meta = None
        pairing_conf = pair.pairing_confidence

        if key in corr_idx:
            rqi_raw = corr_idx[key]["rqi_raw"]
            rqi_loc = corr_idx[key]["locator"]
            match_meta = {"method": "key_match", "key": key}
        else:
            # fallback selection
            rqi_raw2, sim, meta = best_effort_match_rqi(qi_raw, corr_idx)
            if rqi_raw2 and sim >= float(pack_get(pack, "policy.pairing.min_similarity_fallback", 0.45)):
                rqi_raw = rqi_raw2
                match_meta = {"method": "similarity_fallback", **(meta or {})}
                pairing_conf = min(pairing_conf, sim)
                # locator: use matched key locator if possible
                mk = (meta or {}).get("match_key")
                if mk and mk in corr_idx:
                    rqi_loc = corr_idx[mk]["locator"]

        # CAS 1 ONLY: if no RQi segment => quarantine via POSABLE later (rqi_clean=None)
        qi_clean, qi_der = sanitize_text(qi_raw, lang)
        rqi_clean = None
        rqi_der = []
        if rqi_raw:
            rqi_clean, rqi_der = sanitize_text(rqi_raw, lang)

        deriv = []
        deriv.extend([{"target": "qi", **d} for d in qi_der])
        deriv.extend([{"target": "rqi", **d} for d in rqi_der])
        if match_meta:
            deriv.append({"op": "qi_rqi_match", "meta": match_meta})

        qi_id = f"QI_{sha256_text(pair.subject.doc_id + '|' + exo_ref + '|' + q_ref + '|' + qi_clean)[7:19]}"
        rqi_id = f"RQI_{sha256_text(pair.correction.doc_id + '|' + exo_ref + '|' + q_ref)[7:19]}" if rqi_raw else None

        atoms.append(Atom(
            atom_id=f"ATOM_{sha256_text(qi_id + '|' + (rqi_id or 'NONE'))[7:19]}",
            subject_id=pair.subject.doc_id,
            correction_id=pair.correction.doc_id if pair.correction else None,
            qi_id=qi_id,
            rqi_id=rqi_id,
            exo_ref=str(exo_ref),
            q_ref=str(q_ref),
            qi_raw=qi_raw,
            rqi_raw=rqi_raw,
            qi_clean=qi_clean,
            rqi_clean=rqi_clean,
            language_detected=lang,
            year_ref=year_ref,
            source_url=pair.subject.url,
            source_fingerprint=pair.subject.bytes_sha256,
            extraction_locators={
                "qi": u["locator"],
                "rqi": rqi_loc if rqi_loc else Locator(page_start=-1, page_end=-1)
            },
            sanitizer_derivations=deriv,
            pairing_confidence=float(pairing_conf),
            figure_refs=[]
        ))

    return atoms, audit

def run_pipeline(pack: Dict[str, Any], pairs: List[Pair], lang: str) -> Dict[str, Any]:
    """
    Full pipeline:
      Pairs -> ATOMS -> POSABLE -> TRACES -> CLUSTERS -> QC candidates -> IA2 -> Selection/Coverage
    """
    # Build atoms for all pairs
    all_atoms: List[Atom] = []
    pair_audits = []
    for pr in pairs:
        atoms, aud = build_atoms_from_pair(pack, pr, lang)
        pair_audits.append(aud)
        all_atoms.extend(atoms)

    # POSABLE gate
    posable, chapter_metrics = posable_gate(pack, all_atoms)

    # Miner traces (POSABLE only)
    traces = mine_traces(pack, all_atoms, posable, lang)

    # Clusters
    clusters = build_clusters(pack, traces)

    # Candidates
    qcs = build_qc_candidates(pack, all_atoms, posable, traces, clusters, lang)

    # IA2
    for qc in qcs:
        ok, aud = ia2_checks(pack, qc)
        qc.ia2_audit = aud
        qc.ia2_pass = ok

    # Selection & coverage per chapter
    by_ch: Dict[str, List[QCCandidate]] = {}
    for qc in qcs:
        by_ch.setdefault(qc.chapter_ref, []).append(qc)

    selections: Dict[str, SelectionResult] = {}
    for ch in stable_sort(by_ch.keys(), key_fn=lambda x: x):
        selections[ch] = greedy_select_cover(pack, ch, by_ch[ch], traces)

    # Global KPIs
    total_posable = sum(cm.N_total_posable for cm in chapter_metrics.values() if cm.chapter_ref not in ["UNSCOPED"])
    total_orphans = sum(len(sel.orphans) for sel in selections.values())
    coverage_global = 1.0 if total_posable == 0 else float(total_posable - total_orphans) / float(total_posable)

    return {
        "atoms": all_atoms,
        "pair_audits": pair_audits,
        "posable": posable,
        "chapter_metrics": chapter_metrics,
        "traces": traces,
        "clusters": clusters,
        "qc_candidates": qcs,
        "selections": selections,
        "kpi": {
            "total_pairs": len(pairs),
            "total_atoms": len(all_atoms),
            "total_qi": len(all_atoms),
            "total_rqi": sum(1 for a in all_atoms if a.rqi_clean),
            "total_posable": total_posable,
            "total_qc": sum(len(sel.selected_qc_ids) for sel in selections.values()),
            "total_orphans": total_orphans,
            "coverage_global": coverage_global
        }
    }

# =============================================================================
# Saturation loop (new_QC=0)
# =============================================================================
def qc_set_signature(selections: Dict[str, SelectionResult]) -> str:
    items = []
    for ch in stable_sort(selections.keys(), key_fn=lambda x: x):
        items.extend([f"{ch}:{qc_id}" for qc_id in selections[ch].selected_qc_ids])
    return sha256_text("|".join(items))

def run_saturation(pack: Dict[str, Any], all_pairs: List[Pair], lang: str, vol_start: int, vol_max: int, step: int) -> Dict[str, Any]:
    logs = []
    iters = []
    prev_sig = None
    last_result = None

    # deterministic ordering of pairs
    ordered = stable_sort(all_pairs, key_fn=lambda p: p.pair_id)

    for vol in range(vol_start, vol_max + 1, step):
        subset = ordered[:vol]
        t0 = time.time()
        res = run_pipeline(pack, subset, lang)
        dt = time.time() - t0

        sig = qc_set_signature(res["selections"])
        new_qc_count = 0
        if prev_sig is None:
            new_qc_count = res["kpi"]["total_qc"]
        else:
            new_qc_count = 0 if sig == prev_sig else 1  # strict: detect change; detailed diff can be added

        prev_sig = sig
        last_result = res

        iters.append({
            "volume": vol,
            "qc_total": res["kpi"]["total_qc"],
            "qi_total": res["kpi"]["total_qi"],
            "rqi_total": res["kpi"]["total_rqi"],
            "qi_posable": res["kpi"]["total_posable"],
            "orphans": res["kpi"]["total_orphans"],
            "coverage": res["kpi"]["coverage_global"],
            "new_qc_count": new_qc_count,
            "runtime_s": round(dt, 2),
            "sig": sig
        })

        logs.append(f"[{now_iso()}] vol={vol} qi={res['kpi']['total_qi']} rqi={res['kpi']['total_rqi']} posable={res['kpi']['total_posable']} qc={res['kpi']['total_qc']} orphans={res['kpi']['total_orphans']} coverage={res['kpi']['coverage_global']:.3f} new_qc={new_qc_count}")

        # stop condition: one iteration with new_QC=0
        if new_qc_count == 0 and vol > vol_start:
            break

    # SEALED strict:
    # - new_QC==0 on last iter
    # - orphans==0
    # - sanity_ok: at least 1 posable and 1 qc if posable>0
    final = iters[-1] if iters else None
    sealed = False
    seal_reason = "NOT_RUN"
    if final:
        sanity_ok = True
        if final["qi_posable"] > 0 and final["qc_total"] <= 0:
            sanity_ok = False
        sealed = (final["new_qc_count"] == 0) and (final["orphans"] == 0) and sanity_ok
        seal_reason = "ALL_CHECKS_PASSED" if sealed else "FAILED_STRICT_CHECKS"

    return {
        "iterations": iters,
        "logs": logs,
        "sealed": sealed,
        "seal_reason": seal_reason,
        "last_result": last_result
    }

# =============================================================================
# Exports + signature
# =============================================================================
def make_exports(pack: Dict[str, Any], sat: Dict[str, Any]) -> Dict[str, Any]:
    res = sat.get("last_result") or {}
    atoms: List[Atom] = res.get("atoms", [])
    posable: Dict[str, PosableDecision] = res.get("posable", {})
    qcs: List[QCCandidate] = res.get("qc_candidates_
