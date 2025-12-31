# =============================================================================
# SMAXIA GTE Console V31.10.12 — ISO-PROD TEST (MONO-FICHIER)
# =============================================================================
# OBJECTIF (ISO-PROD):
# - 1 seul fichier: UI + moteur (aucun import moteur externe)
# - Flux: Activation (Pack FILE signé) -> Sélection -> Harvest -> RUN -> Preuve -> Exports
# - Variables observables:
#   (Qi, RQi), Triggers (Tj), F1 (psi_raw, Psi_q), ARI, FRT, QC, saturation (Set_B-Set_A), sélection F2
#
# DOCTRINE SMAXIA (rappels):
# - Interdit: hardcode métier (pays/langue/matière/chapitres) DANS le CORE.
# - Pack doit être un FICHIER (UPLOADED_FILE ou GENERATED_FILE) avec signature SHA256.
# - Aucune Qi orpheline (Qi sans QC) pour SEALED.
# - SEALED basé sur convergence: Set_B - Set_A = ∅ (saturation)
#
# NOTE TEST:
# - Pack Genesis = générateur de pack minimal, écrit sur disque temp, signé, puis rechargé depuis fichier.
#   -> Zéro embedded pack "en mémoire" comme fallback silencieux.
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import glob
import time
import math
import hashlib
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
APP_VERSION = "V31.10.12"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 35
MAX_PDF_MB = 40

DEFAULT_COUNTRY = "FR"
DEFAULT_LEVEL = "TERMINALE"
DEFAULT_SUBJECT = "MATH"

APMEP_ROOT_BY_LEVEL = {
    "TERMINALE": "https://www.apmep.fr/Annales-du-Bac-Terminale",
    "PREMIERE": "https://www.apmep.fr/Annales-du-Bac-Premiere",
    "SECONDE": "https://www.apmep.fr/Annales-du-Bac-Seconde",
}

# -----------------------------------------------------------------------------
# SESSION / LOGS
# -----------------------------------------------------------------------------
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def ss_init():
    st.session_state.setdefault("pack_active", None)
    st.session_state.setdefault("pack_meta", {})
    st.session_state.setdefault("country", DEFAULT_COUNTRY)
    st.session_state.setdefault("level", DEFAULT_LEVEL)
    st.session_state.setdefault("subject", DEFAULT_SUBJECT)

    st.session_state.setdefault("library", [])
    st.session_state.setdefault("harvest_manifest", None)

    st.session_state.setdefault("qi_pack", None)
    st.session_state.setdefault("qc_pack", None)
    st.session_state.setdefault("qc_selection", None)
    st.session_state.setdefault("chapter_report", None)

    st.session_state.setdefault("sealed", False)
    st.session_state.setdefault("run_stats", {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0})
    st.session_state.setdefault("last_run_audit", None)

    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("_uploads", {})          # local://id -> bytes
    st.session_state.setdefault("_http_pdf_cache", {})   # url -> bytes
    st.session_state.setdefault("_pack_file_path", None) # physical pack path

def log(msg: str):
    st.session_state.logs.append(f"[{_utc_ts()}] {msg}")

def logs_text() -> str:
    return "\n".join(st.session_state.logs[-12000:])


# -----------------------------------------------------------------------------
# TEXT / HASH UTILS
# -----------------------------------------------------------------------------
def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", norm_text(s))[:1600]

def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 0.0


# -----------------------------------------------------------------------------
# PACK GENESIS (FILE-BASED, SIGNED)
# -----------------------------------------------------------------------------
def pack_validate(pack: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(pack, dict):
        return False, "Pack: JSON non-dict"
    pack_id = str(pack.get("pack_id") or "").strip()
    if not pack_id:
        return False, "Pack: pack_id manquant"
    chapters = pack.get("chapters")
    if not isinstance(chapters, list) or len(chapters) == 0:
        return False, "Pack: chapters vides"
    for ch in chapters:
        if not isinstance(ch, dict):
            return False, "Pack: chapitre non-dict"
        code = str(ch.get("chapter_code") or "").strip()
        if not code:
            return False, "Pack: chapter_code manquant"
    return True, "OK"

def pack_canonical_json(pack: Dict[str, Any]) -> str:
    # Canonicalisation simple (stable) pour signature
    return json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def pack_sign(pack: Dict[str, Any]) -> Dict[str, Any]:
    canon = pack_canonical_json(pack)
    sig = sha256_text(canon)
    out = dict(pack)
    out["pack_signature_sha256"] = sig
    out["_signed_at"] = _utc_ts()
    return out

def pack_chapters(pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters = pack.get("chapters") or []
    out = []
    for it in chapters:
        if not isinstance(it, dict):
            continue
        code = str(it.get("chapter_code") or "").strip()
        label = str(it.get("chapter_label") or it.get("label") or code).strip()
        if not code:
            continue
        out.append({
            "chapter_code": code,
            "chapter_label": label,
            "keywords": it.get("keywords", []),
        })
    # unique by code
    seen, uniq = set(), []
    for c in out:
        if c["chapter_code"] in seen:
            continue
        seen.add(c["chapter_code"])
        uniq.append(c)
    return uniq

def pack_genesis_build(country: str, n_chapters: int) -> Tuple[str, Dict[str, Any]]:
    """
    Génère un pack minimal *non-métier* (codes CH_001..), écrit sur disque temp, signé,
    puis retourne (path, pack_dict).
    """
    n = max(3, min(int(n_chapters), 60))
    pack = {
        "pack_id": f"CAP_{country}_GENESIS_V1",
        "country_code": country,
        "version": "1.0.0",
        "_source": "GENERATED_FILE",
        "chapters": [
            {
                "chapter_code": f"CH_{i:03d}",
                "chapter_label": f"CH_{i:03d}",
                "keywords": []
            }
            for i in range(1, n + 1)
        ],
    }
    pack = pack_sign(pack)

    # Write to temp file
    tmp_dir = tempfile.gettempdir()
    fname = f"smaxia_pack_{pack['pack_id']}_{stable_id(pack['pack_signature_sha256'])}.json"
    path = os.path.join(tmp_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(pack, ensure_ascii=False, indent=2))
    return path, pack

def pack_load_from_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        pack = json.load(f)

    # If signature absent, sign now but keep audit field
    if not pack.get("pack_signature_sha256"):
        pack["_signature_note"] = "signature_absente_dans_fichier; signature_calculée_en_runtime"
        pack = pack_sign(pack)

    ok, msg = pack_validate(pack)
    if not ok:
        raise RuntimeError(msg)

    # Verify signature matches canonical content (excluding pack_signature_sha256 and runtime underscored keys)
    # To be strict, we recompute on a copy without runtime-only keys.
    pack_copy = {k: v for k, v in pack.items() if not str(k).startswith("_") and k != "pack_signature_sha256"}
    canon = pack_canonical_json(pack_copy)
    sig_calc = sha256_text(canon)
    sig_file = str(pack.get("pack_signature_sha256") or "")
    if sig_file and sig_calc != sig_file:
        raise RuntimeError("Pack: signature_sha256 invalide (contenu != signature)")

    return pack

def pack_activate(country: str, uploaded_pack_file: Optional[bytes], genesis_enabled: bool, genesis_n: int) -> Dict[str, Any]:
    """
    Active un pack en respectant doctrine FILE-based:
    - si upload fourni: écrire sur disque temp, signer/valider, source=UPLOADED_FILE
    - sinon si genesis enabled: générer un pack genesis FILE signé, source=GENERATED_FILE
    - sinon: FAIL (pas de fallback silencieux)
    """
    if uploaded_pack_file:
        tmp_dir = tempfile.gettempdir()
        raw_sig = sha256_bytes(uploaded_pack_file)
        path = os.path.join(tmp_dir, f"smaxia_pack_uploaded_{country}_{raw_sig[:16]}.json")
        with open(path, "wb") as f:
            f.write(uploaded_pack_file)

        pack = pack_load_from_file(path)
        pack["_source"] = "UPLOADED_FILE"
        pack["_pack_file_path"] = path
        log(f"[PACK] Activated UPLOADED_FILE path={path} pack_id={pack.get('pack_id')}")
        return pack

    if genesis_enabled:
        path, pack = pack_genesis_build(country, genesis_n)
        pack["_pack_file_path"] = path
        log(f"[PACK] Activated GENERATED_FILE (Genesis) path={path} pack_id={pack.get('pack_id')}")
        return pack

    raise RuntimeError("Activation impossible: aucun pack uploadé et Genesis désactivé (ISO-PROD = FAIL)")


# -----------------------------------------------------------------------------
# HTTP + PDF
# -----------------------------------------------------------------------------
def _http_get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r

def download_pdf_bytes(url: str) -> bytes:
    r = _http_get(url)
    if (len(r.content) / (1024 * 1024)) > MAX_PDF_MB:
        raise ValueError("PDF trop volumineux")
    return r.content

def fetch_pdf_bytes(url: str) -> bytes:
    if url.startswith("local://"):
        data = st.session_state.get("_uploads", {}).get(url)
        if not data:
            raise RuntimeError(f"Upload local introuvable: {url}")
        return data
    cache = st.session_state.get("_http_pdf_cache", {})
    if url in cache:
        return cache[url]
    b = download_pdf_bytes(url)
    cache[url] = b
    st.session_state._http_pdf_cache = cache
    return b

def _dehyphenate(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def _looks_like_page_number(line: str) -> bool:
    s = norm_text(line)
    return bool(re.fullmatch(r"\d{1,3}", s) or re.fullmatch(r"page\s*\d{1,3}", s))

def _extract_pages_text(pdf_bytes: bytes, max_pages: int = 90) -> List[str]:
    pages = []
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages[:max_pages]:
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pages.append("")
            if any(pages):
                return pages
        except Exception:
            pass

    if PdfReader:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages[:max_pages]:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
        except Exception:
            pass

    return pages

def clean_pdf_text(pages: List[str]) -> str:
    if not pages:
        return ""

    page_lines = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = _dehyphenate(p)
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        page_lines.append(lines)

    # Detect repeated headers/footers
    top_counts, bot_counts = {}, {}
    n_pages = len(page_lines)

    for lines in page_lines:
        for ln in lines[:3]:
            k = norm_text(ln)[:140]
            if k:
                top_counts[k] = top_counts.get(k, 0) + 1
        for ln in lines[-3:]:
            k = norm_text(ln)[:140]
            if k:
                bot_counts[k] = bot_counts.get(k, 0) + 1

    header_keys = {k for k, c in top_counts.items() if c >= max(3, int(0.65 * n_pages))}
    footer_keys = {k for k, c in bot_counts.items() if c >= max(3, int(0.65 * n_pages))}

    cleaned_pages = []
    for lines in page_lines:
        keep = []
        for ln in lines:
            k = norm_text(ln)[:140]
            if k in header_keys or k in footer_keys:
                continue
            if _looks_like_page_number(ln):
                continue
            keep.append(ln)
        cleaned_pages.append("\n".join(keep))

    text = "\n\n".join([p for p in cleaned_pages if p.strip()])
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    pages = _extract_pages_text(pdf_bytes)
    return clean_pdf_text(pages)


# -----------------------------------------------------------------------------
# QI EXTRACTION (non-arbitraire, anti-fake par stabilité)
# -----------------------------------------------------------------------------
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦±×÷]")

def _has_question_shape(s: str) -> bool:
    t = s or ""
    if "?" in t:
        return True
    if _MATH_SYMBOL_RE.search(t):
        return True
    if re.search(r"\b(demontrer|montrer|calculer|determiner|resoudre|etudier|justifier)\b", norm_text(t)):
        return True
    if re.search(r"\b\d+\b", t) and re.search(r"[=<>≤≥]", t):
        return True
    return False

def _merge_wrapped_lines(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").split("\n")]
    out, buf = [], ""
    for ln in lines:
        if not ln.strip():
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append("")
            continue

        if not buf:
            buf = ln.strip()
            continue

        # If previous ends with strong punctuation, new sentence
        if re.search(r"[.:;!?]$", buf):
            out.append(buf.strip())
            buf = ln.strip()
        else:
            buf = (buf + " " + ln.strip()).strip()

    if buf:
        out.append(buf.strip())

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()

_Q_START_RE = re.compile(r"(?m)^\s*(\d{1,2}|[a-h]|[ivxIVX]{1,6})\s*[\)\.\-:]\s+")

def split_questions(raw_text: str, max_qi: int = 260) -> List[str]:
    t = (raw_text or "").replace("\r", "\n")
    t = _dehyphenate(t)
    t = _merge_wrapped_lines(t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []

    starts = {0}
    for m in _Q_START_RE.finditer(t):
        starts.add(m.start())
    idxs = sorted(starts)

    chunks = []
    for a, b in zip(idxs, idxs[1:] + [len(t)]):
        c = t[a:b].strip()
        c = re.sub(r"\s+", " ", c).strip()
        if len(c) < 45:
            continue
        if not _has_question_shape(c):
            continue
        chunks.append(c)

    # Fallback if too few segments (question marks)
    if len(chunks) <= 2:
        parts = re.split(r"(?<=\?)\s+", re.sub(r"\s+", " ", t))
        chunks = [p.strip() for p in parts if len(p.strip()) >= 70 and _has_question_shape(p)]

    # De-dup
    out, seen = [], set()
    for c in chunks:
        k = stable_id(norm_text(c)[:420])
        if k in seen:
            continue
        seen.add(k)
        if len(c) > 2500:
            c = c[:2500] + "…"
        out.append(c)

    return out[:max_qi]

def align_qi_rqi(qs: List[str], rs: List[str]) -> List[Optional[int]]:
    if not qs or not rs:
        return [None for _ in qs]

    qtok = [tokenize(q) for q in qs]
    rtok = [tokenize(r) for r in rs]

    out = [None] * len(qs)

    # local window first
    for i in range(len(qs)):
        best_j, best_s = None, 0.0
        lo, hi = max(0, i - 6), min(len(rs), i + 7)
        for j in range(lo, hi):
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.10:
            out[i] = best_j

    # global for remaining
    for i in range(len(qs)):
        if out[i] is not None:
            continue
        best_j, best_s = None, 0.0
        for j in range(len(rs)):
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.14:
            out[i] = best_j

    return out


# -----------------------------------------------------------------------------
# TRIGGERS (Tj) + F1 IMPLEMENTATION (auditables)
# -----------------------------------------------------------------------------
def build_triggers(qi_text: str) -> List[str]:
    t = qi_text or ""
    out = []
    if "?" in t:
        out.append("TRG_QMARK")
    if _MATH_SYMBOL_RE.search(t):
        out.append("TRG_MATHSYM")
    if re.search(r"\b\d+\b", t):
        out.append("TRG_NUMBER")
    if re.search(r"[∑∏∫]", t):
        out.append("TRG_AGGREGATE")
    if re.search(r"[≤≥<>]", t):
        out.append("TRG_RELATION")
    if re.search(r"\b(demontrer|montrer|justifier)\b", norm_text(t)):
        out.append("TRG_PROOF")
    if re.search(r"\b(resoudre|equation|systeme)\b", norm_text(t)):
        out.append("TRG_SOLVE")
    if not out:
        out = ["TRG_GENERIC"]
    return out[:10]

def trigger_weight(trigger_id: str) -> float:
    """
    Tj déterministe, auditable: hash(trigger_id) -> [0.10..1.00]
    (pas de hardcode métier ; stable ; recalculable)
    """
    h = hashlib.sha256(trigger_id.encode("utf-8")).hexdigest()
    # take 8 hex => int
    x = int(h[:8], 16) / float(16**8 - 1)
    return 0.10 + 0.90 * x

def F1_compute(qi_text: str, triggers: List[str], delta_c: float = 1.0) -> Dict[str, Any]:
    """
    F1 (doc v10.6.1):
      psi_raw(q) = δ_c * Σ_{j=1..m(q)} Tj
      Ψ_q = max(0, max_{j=1..m(q)} Tj)
    """
    Tj = {t: round(trigger_weight(t), 6) for t in triggers}
    if Tj:
        psi_raw = float(delta_c) * sum(Tj.values())
        Psi_q = max(0.0, max(Tj.values()))
    else:
        psi_raw, Psi_q = 0.0, 0.0
    return {
        "delta_c": float(delta_c),
        "m_q": len(triggers),
        "triggers_Tj": Tj,
        "psi_raw": round(psi_raw, 8),
        "Psi_q": round(Psi_q, 8),
    }


# -----------------------------------------------------------------------------
# ARI / FRT (templates invariants + evidence)
# -----------------------------------------------------------------------------
def method_signature(qi_text: str, triggers: List[str]) -> Tuple[str, ...]:
    """
    Signature méthode (invariant):
    - basée sur structure + triggers (pas de mots métier/chapitre)
    """
    feats = []
    t = qi_text or ""
    if "?" in t:
        feats.append("F_QMARK")
    if _MATH_SYMBOL_RE.search(t):
        feats.append("F_MATHSYM")
    if re.search(r"\b\d+[,\.]\d+\b", t):
        feats.append("F_DECIMAL")
    if re.search(r"[∑∏∫]", t):
        feats.append("F_AGGREGATE")
    if re.search(r"[≤≥<>]", t):
        feats.append("F_RELATION")
    if re.search(r"\b(demontrer|montrer|justifier)\b", norm_text(t)):
        feats.append("F_PROOF")
    if re.search(r"\b(resoudre|equation|systeme)\b", norm_text(t)):
        feats.append("F_SOLVE")

    # include triggers ids (already invariant)
    feats.extend([f"T::{x}" for x in triggers])

    return tuple(sorted(set(feats))) if feats else ("F_GENERIC",)

def build_ari(qi_text: str, rqi_text: str, sig: Tuple[str, ...], F1: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "template": "ARI_V2_INVARIANT",
        "sig_method": list(sig),
        "steps": [
            {"step_id": "S1", "op": "OP_READ", "action": "Extract data and constraints."},
            {"step_id": "S2", "op": "OP_PLAN", "action": "Select invariant resolution strategy."},
            {"step_id": "S3", "op": "OP_EXECUTE", "action": "Apply transformations step-by-step."},
            {"step_id": "S4", "op": "OP_CHECK", "action": "Validate intermediate and final results."},
            {"step_id": "S5", "op": "OP_CONCLUDE", "action": "Formulate final answer."},
        ],
        "evidence": {
            "has_rqi": bool(rqi_text),
            "qi_chars": len(qi_text or ""),
            "rqi_chars": len(rqi_text or ""),
            "F1": F1,
        },
    }

def build_frt(triggers: List[str], sig: Tuple[str, ...]) -> Dict[str, Any]:
    return {
        "template": "FRT_V2_INVARIANT",
        "sig_method": list(sig),
        "sections": [
            {"section_id": "SEC_OBJ", "name": "OBJECTIVE"},
            {"section_id": "SEC_DAT", "name": "DATA"},
            {"section_id": "SEC_MTH", "name": "METHOD"},
            {"section_id": "SEC_EXE", "name": "EXECUTION"},
            {"section_id": "SEC_RES", "name": "RESULT"},
            {"section_id": "SEC_VAL", "name": "VALIDATION"},
        ],
        "triggers_applied": triggers,
    }


# -----------------------------------------------------------------------------
# QC GENERATION (coverage-driven; QC = "Comment ... ?" + signature méthode)
# -----------------------------------------------------------------------------
def build_qc_text(sig_method: Tuple[str, ...]) -> str:
    # QC MUST start with "Comment" and end with "?"
    core = " ".join(list(sig_method)[:6]) if sig_method else "F_GENERIC"
    return f"Comment résoudre une question caractérisée par: {core} ?"

def qc_id_from_sig(sig_method: Tuple[str, ...]) -> str:
    return "QC_" + stable_id("SIG", "||".join(sig_method))

def map_qc_to_chapter(qc_text: str, chapters: List[Dict[str, Any]]) -> str:
    # pack-driven only; if keywords empty, UNMAPPED.
    if not chapters:
        return "UNMAPPED"
    qc_toks = set(tokenize(qc_text))
    best = ("UNMAPPED", 0.0)
    for ch in chapters:
        kws = ch.get("keywords") or []
        if not kws:
            continue
        ch_toks = set(tokenize(" ".join([str(x) for x in kws])))
        if not ch_toks:
            continue
        score = len(qc_toks & ch_toks) / max(1, len(ch_toks))
        if score > best[1]:
            best = (str(ch["chapter_code"]), score)
    return best[0] if best[1] >= 0.10 else "UNMAPPED"

def sigma_similarity(sig_a: Tuple[str, ...], sig_b: Tuple[str, ...]) -> float:
    # σ(q,p) in [0,1] based on Jaccard over signature tokens
    a, b = list(sig_a), list(sig_b)
    return float(jaccard(a, b))


# -----------------------------------------------------------------------------
# F2 SELECTION SCORE (QC selection coverage-driven)
# -----------------------------------------------------------------------------
def alpha_trec_deterministic(qc_id: str) -> float:
    """
    α_trec: deterministic recency factor (>=1) without external time series.
    For ISO-PROD test we produce a stable value from qc_id.
    """
    h = hashlib.sha256(qc_id.encode("utf-8")).hexdigest()
    x = int(h[8:16], 16) / float(16**8 - 1)
    return 1.0 + 3.0 * x   # [1..4]

def F2_score_qc(
    qc_id: str,
    Psi_qc: float,
    n_historical: int,
    N_total: int,
    selected_sig_list: List[Tuple[str, ...]],
    qc_sig: Tuple[str, ...]
) -> float:
    """
    F2 (doc v10.6.1):
      Score(q)= (n_historical / N_total) * (1/alpha_trec) * Psi_q * Π_{p∈S}(1 - σ(q,p)) * 100
    """
    if N_total <= 0:
        return 0.0
    base = (n_historical / float(N_total))
    alpha = alpha_trec_deterministic(qc_id)
    diversity = 1.0
    for sig_p in selected_sig_list:
        diversity *= (1.0 - sigma_similarity(qc_sig, sig_p))
    score = base * (1.0 / alpha) * float(Psi_qc) * diversity * 100.0
    return float(max(0.0, score))

def select_qc_coverage_driven(
    qc_pack: List[Dict[str, Any]],
    qi_pack: List[Dict[str, Any]],
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Simple selection: ranks QC by F2 score.
    - n_historical approximated by cluster_size
    - N_total = sum cluster_size
    - Psi_qc = max Psi_q among Qi in cluster
    """
    qc_by_id = {q["qc_id"]: q for q in qc_pack}
    qi_by_qc: Dict[str, List[Dict[str, Any]]] = {}
    for qi in qi_pack:
        qcid = qi.get("qc_id") or ""
        if not qcid:
            continue
        qi_by_qc.setdefault(qcid, []).append(qi)

    N_total = sum(int(q.get("cluster_size") or 0) for q in qc_pack) or 1
    selected: List[Dict[str, Any]] = []
    selected_sigs: List[Tuple[str, ...]] = []

    # greedy selection for diversity
    remaining = list(qc_pack)
    for rank in range(1, max(1, int(top_k)) + 1):
        best = None
        best_score = -1.0
        for qc in remaining:
            qcid = qc["qc_id"]
            qsig = tuple(qc.get("sig_method") or [])
            items = qi_by_qc.get(qcid, [])
            Psi_qc = 0.0
            for qi in items:
                Psi_qc = max(Psi_qc, float(qi.get("F1", {}).get("Psi_q", 0.0)))
            score = F2_score_qc(
                qc_id=qcid,
                Psi_qc=Psi_qc,
                n_historical=int(qc.get("cluster_size") or 0),
                N_total=N_total,
                selected_sig_list=selected_sigs,
                qc_sig=qsig
            )
            if score > best_score:
                best_score = score
                best = (qc, score, Psi_qc)

        if not best:
            break

        qc, score, Psi_qc = best
        remaining = [q for q in remaining if q["qc_id"] != qc["qc_id"]]
        selected_sigs.append(tuple(qc.get("sig_method") or []))

        selected.append({
            "selected_rank": rank,
            "qc_id": qc["qc_id"],
            "qc": qc["qc"],
            "cluster_size": qc.get("cluster_size"),
            "Psi_qc": round(float(Psi_qc), 8),
            "score_F2": round(float(score), 6),
            "sig_method": qc.get("sig_method"),
        })

    return selected


# -----------------------------------------------------------------------------
# HARVEST APMEP (volume réel)
# -----------------------------------------------------------------------------
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé")
    return BeautifulSoup(html, "html.parser")

def _abs_url(base: str, href: str) -> str:
    return href if href.startswith("http") else requests.compat.urljoin(base, href)

def _is_meta_pdf(url: str, label: str) -> bool:
    s = norm_text(url + " " + (label or ""))
    return any(w in s for w in ["index", "sommaire", "liste", "annexe", "grille", "formulaire"])

def _is_corrige_label(url: str, label: str) -> bool:
    s = norm_text(os.path.basename(url) + " " + (label or ""))
    return "corrig" in s or "correction" in s or "solution" in s

def _filekey(name: str) -> str:
    s = norm_text(name)
    s = re.sub(r"\.pdf$", "", s)
    s = re.sub(r"\b(corrig(e|é|es)?|correction|solutions?)\b", "", s)
    return re.sub(r"[_\-]+", " ", s).strip()

def harvest_apmep(level: str, subject: str, years_back: int, volume_max: int) -> Dict[str, Any]:
    if level not in APMEP_ROOT_BY_LEVEL:
        raise ValueError(f"Niveau non supporté: {level}")

    root = APMEP_ROOT_BY_LEVEL[level]
    log(f"[HARVEST] scope={level}|{subject} root={root}")

    html = _http_get(root).text
    sp = _soup(html)

    year_links = []
    for a in sp.find_all("a"):
        href = a.get("href") or ""
        m = re.search(r"Annee-(20\d{2})", href)
        if m:
            year_links.append((int(m.group(1)), _abs_url(root, href)))

    year_links = sorted(set(year_links), key=lambda x: -x[0])
    if not year_links:
        raise RuntimeError("Aucune page Année trouvée")

    current_year = year_links[0][0]
    min_year = current_year - max(1, int(years_back)) + 1
    selected = [(y, u) for (y, u) in year_links if y >= min_year]
    log(f"[HARVEST] years={len(selected)} (min={min_year})")

    pairs = []
    corrige_ok = 0

    for (y, url) in selected:
        if len(pairs) >= int(volume_max):
            break

        try:
            y_html = _http_get(url).text
            y_sp = _soup(y_html)

            pdf_links = []
            for a in y_sp.find_all("a"):
                href = (a.get("href") or "").strip()
                if not href.lower().endswith(".pdf"):
                    continue
                absu = _abs_url(url, href)
                label = (a.get_text() or "").strip()
                if _is_meta_pdf(absu, label):
                    continue
                pdf_links.append({
                    "url": absu,
                    "name": os.path.basename(absu),
                    "label": label,
                    "is_corrige": _is_corrige_label(absu, label),
                })

            if not pdf_links:
                continue

            subjects = [p for p in pdf_links if not p["is_corrige"]]
            corriges = [p for p in pdf_links if p["is_corrige"]]

            corr_index = [(p, _filekey(p["name"]), tokenize(_filekey(p["name"]))) for p in corriges]
            used_corr = set()

            for s in subjects:
                if len(pairs) >= int(volume_max):
                    break

                skey = _filekey(s["name"])
                stok = tokenize(skey)

                best = (None, 0.0)
                for (c, ckey, ctok) in corr_index:
                    if c["url"] in used_corr:
                        continue
                    score = jaccard(stok, ctok)
                    if score > best[1]:
                        best = (c, score)

                corrige = best[0] if best[1] >= 0.15 else None
                if corrige:
                    used_corr.add(corrige["url"])

                pair_id = f"PAIR_{level}|{subject}_{y}_{stable_id(s['name'], str(corrige['name'] if corrige else ''))}"
                item = {
                    "pair_id": pair_id,
                    "scope": f"{level}|{subject}",
                    "source": f"APMEP {y}",
                    "year": y,
                    "sujet": s["name"],
                    "corrige?": bool(corrige),
                    "corrige_name": corrige["name"] if corrige else "",
                    "reason": "" if corrige else "corrigé absent",
                    "sujet_url": s["url"],
                    "corrige_url": corrige["url"] if corrige else "",
                }

                if not any(x["sujet_url"] == item["sujet_url"] for x in pairs):
                    pairs.append(item)
                    if item["corrige?"]:
                        corrige_ok += 1

            log(f"[HARVEST] year={y} pdfs={len(pdf_links)} sujets={len(subjects)} pairs_now={len(pairs)}")

        except Exception as e:
            log(f"[HARVEST] year={y} FAILED: {e}")

    return {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "country": st.session_state.country,
        "level": level,
        "subject": subject,
        "items_total": len(pairs),
        "items_corrige_ok": corrige_ok,
        "library": pairs,
    }


# -----------------------------------------------------------------------------
# SANITY (non arbitraire): detecte instabilité de segmentation (fake fragments)
# -----------------------------------------------------------------------------
def segmentation_stability_check(qs: List[str]) -> Dict[str, Any]:
    n = len(qs)
    if n == 0:
        return {"ok": False, "reason": "no_qi", "n": 0}

    lengths = [len(q or "") for q in qs]
    avg_len = sum(lengths) / max(1, n)
    short_ratio = sum(1 for L in lengths if L < 60) / float(n)

    math_ratio = sum(1 for q in qs if _MATH_SYMBOL_RE.search(q or "")) / float(n)
    shape_ratio = sum(1 for q in qs if _has_question_shape(q or "")) / float(n)

    # duplicate ratio (signals of fragmented repetition)
    keys = [stable_id(norm_text(q)[:220]) for q in qs]
    dup_ratio = 1.0 - (len(set(keys)) / float(n))

    # Instability if multiple symptoms together (NOT only quantity)
    # - too many + too short + poor structure OR high duplicates
    instable = False
    reasons = []

    if n >= 120 and avg_len < 85 and shape_ratio < 0.75:
        instable = True
        reasons.append("high_n_low_avglen_low_shape")

    if n >= 120 and short_ratio > 0.40:
        instable = True
        reasons.append("high_short_ratio")

    if n >= 120 and dup_ratio > 0.25:
        instable = True
        reasons.append("high_dup_ratio")

    if n >= 150 and math_ratio < 0.25 and shape_ratio < 0.70:
        instable = True
        reasons.append("very_high_n_low_math_low_shape")

    return {
        "ok": (not instable),
        "n": n,
        "avg_len": round(avg_len, 2),
        "short_ratio": round(short_ratio, 4),
        "math_ratio": round(math_ratio, 4),
        "shape_ratio": round(shape_ratio, 4),
        "dup_ratio": round(dup_ratio, 4),
        "instability_reasons": reasons,
    }


# -----------------------------------------------------------------------------
# RUN GRANULO TEST (F1/F2 + saturation Set_B-Set_A)
# -----------------------------------------------------------------------------
def build_qc_from_qi(qi_items: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Cluster by signature méthode (sig_method) => QC per signature.
    Returns qc_pack, qc_map (qi_id -> qc_id)
    """
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for qi in qi_items:
        sig = tuple(qi.get("sig_method") or ("F_GENERIC",))
        buckets.setdefault(sig, []).append(qi)

    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}

    for sig, bucket in buckets.items():
        qcid = qc_id_from_sig(sig)
        qc_text = build_qc_text(sig)
        ch_code = map_qc_to_chapter(qc_text, chapters)
        qc_pack.append({
            "qc_id": qcid,
            "qc": qc_text,
            "chapter_code": ch_code,
            "cluster_size": len(bucket),
            "sig_method": list(sig),
            "qi_ids": [x["qi_id"] for x in bucket],
        })
        for x in bucket:
            qc_map[x["qi_id"]] = qcid

    # stable order
    qc_pack.sort(key=lambda x: (-int(x.get("cluster_size") or 0), x["qc_id"]))
    return qc_pack, qc_map

def run_granulo_test(library: List[Dict[str, Any]], volume_A: int, volume_B: int, pack: Dict[str, Any]) -> Dict[str, Any]:
    chapters = pack_chapters(pack)
    if not chapters:
        raise RuntimeError("Pack invalide: chapitres vides")

    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    if not exploitable:
        raise RuntimeError("Aucun corrigé exploitable (ISO-PROD recommande Sujet+Corrigé)")

    # Phase A subset, Phase B superset
    volume_A = max(1, int(volume_A))
    volume_B = max(volume_A, int(volume_B))
    to_process_B = exploitable[:min(volume_B, len(exploitable))]
    to_process_A = to_process_B[:min(volume_A, len(to_process_B))]

    log(f"[RUN] pairs_A={len(to_process_A)} pairs_B={len(to_process_B)}")

    def extract_pairs(pairs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        qi_items: List[Dict[str, Any]] = []
        rqi_items: List[Dict[str, Any]] = []
        doc_stats: List[Dict[str, Any]] = []
        suspicious_docs: List[Dict[str, Any]] = []

        for pair in pairs:
            pid = pair["pair_id"]
            try:
                su_pdf = fetch_pdf_bytes(pair["sujet_url"])
                co_pdf = fetch_pdf_bytes(pair["corrige_url"])
            except Exception as e:
                log(f"[DL] FAILED {pid}: {e}")
                continue

            su_text = extract_text_from_pdf_bytes(su_pdf)
            co_text = extract_text_from_pdf_bytes(co_pdf)
            if not su_text or not co_text:
                log(f"[PDF] EMPTY {pid}")
                continue

            qs = split_questions(su_text)
            rs = split_questions(co_text)
            link = align_qi_rqi(qs, rs)

            stab = segmentation_stability_check(qs)
            if not stab["ok"]:
                suspicious_docs.append({
                    "pair_id": pid,
                    "qi_count": len(qs),
                    "stability": stab,
                })

            doc_stats.append({
                "pair_id": pid,
                "qi_count": len(qs),
                "rqi_count": len(rs),
                "stability": stab,
            })

            for i, q in enumerate(qs):
                if not q.strip():
                    continue
                j = link[i] if i < len(link) else None
                r = rs[j] if (j is not None and 0 <= j < len(rs)) else ""

                qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
                rqi_id = f"RQI_{stable_id(pid, str(j), norm_text(r)[:180])}" if r else ""

                rqi_items.append({"rqi_id": rqi_id, "pair_id": pid, "k": i + 1, "text": r}) if r else None
                qi_items.append({
                    "qi_id": qi_id,
                    "pair_id": pid,
                    "k": i + 1,
                    "text": q,
                    "rqi_id": rqi_id,
                })

        return qi_items, rqi_items, (doc_stats, suspicious_docs)

    # Extract for B (main run)
    qi_items_B, rqi_items_B, (doc_stats_B, suspicious_B) = extract_pairs(to_process_B)
    if not qi_items_B:
        raise RuntimeError("Aucune Qi extraite (B)")

    # Extract for A (for saturation comparison) – reuse cache
    qi_items_A, rqi_items_A, (doc_stats_A, suspicious_A) = extract_pairs(to_process_A)
    if not qi_items_A:
        raise RuntimeError("Aucune Qi extraite (A)")

    # Build qi_pack (B) with F1/ARI/FRT
    rqi_by_id = {r["rqi_id"]: r for r in rqi_items_B if r.get("rqi_id")}
    qi_pack_B: List[Dict[str, Any]] = []
    qi_posable = 0

    for qi in qi_items_B:
        rtxt = rqi_by_id.get(qi.get("rqi_id") or "", {}).get("text", "")
        triggers = build_triggers(qi["text"])
        f1 = F1_compute(qi["text"], triggers, delta_c=1.0)
        sig = method_signature(qi["text"], triggers)
        ari = build_ari(qi["text"], rtxt, sig, f1)
        frt = build_frt(triggers, sig)

        if rtxt:
            qi_posable += 1

        qi_pack_B.append({
            "qi_id": qi["qi_id"],
            "pair_id": qi["pair_id"],
            "k": qi["k"],
            "qi": qi["text"],
            "rqi": rtxt,
            "has_rqi": bool(rtxt),
            "triggers": triggers,
            "F1": f1,
            "sig_method": list(sig),
            "ari": ari,
            "frt": frt,
        })

    # QC from B
    qc_pack_B, qc_map_B = build_qc_from_qi(qi_pack_B, chapters)

    # Attach qc_id + orphan test
    orphan_ids = []
    for qi in qi_pack_B:
        qcid = qc_map_B.get(qi["qi_id"], "")
        qi["qc_id"] = qcid
        qi["is_orphan"] = (not qcid)
        if not qcid:
            orphan_ids.append(qi["qi_id"])

    # QC from A (saturation set)
    # For A, we can compute signatures quickly from qi_items_A:
    rqi_by_id_A = {r["rqi_id"]: r for r in rqi_items_A if r.get("rqi_id")}
    qi_pack_A: List[Dict[str, Any]] = []
    for qi in qi_items_A:
        rtxt = rqi_by_id_A.get(qi.get("rqi_id") or "", {}).get("text", "")
        triggers = build_triggers(qi["text"])
        f1 = F1_compute(qi["text"], triggers, delta_c=1.0)
        sig = method_signature(qi["text"], triggers)
        qi_pack_A.append({
            "qi_id": qi["qi_id"],
            "qi": qi["text"],
            "rqi": rtxt,
            "has_rqi": bool(rtxt),
            "triggers": triggers,
            "F1": f1,
            "sig_method": list(sig),
        })
    qc_pack_A, _ = build_qc_from_qi(qi_pack_A, chapters)

    set_A = set(q["qc_id"] for q in qc_pack_A)
    set_B = set(q["qc_id"] for q in qc_pack_B)
    delta_new = sorted(list(set_B - set_A))

    saturation_ok = (len(delta_new) == 0)

    # sanity_ok: segmentation stability must not show systemic instability across many docs
    suspicious_all = suspicious_B[:]  # keep B as reference
    suspicious_ratio = (len(suspicious_all) / float(max(1, len(to_process_B))))
    sanity_ok = (suspicious_ratio <= 0.20)  # if too many docs unstable => FAIL
    if not sanity_ok:
        log(f"[SANITY] FAIL suspicious_docs={len(suspicious_all)} ratio={round(suspicious_ratio,3)}")

    # selection F2
    qc_selection = select_qc_coverage_driven(qc_pack_B, qi_pack_B, top_k=min(30, len(qc_pack_B)))

    sealed = bool(
        sanity_ok
        and saturation_ok
        and (len(orphan_ids) == 0)
        and (qi_posable > 0)
        and (len(qc_pack_B) > 0)
    )

    audit = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pairs_A": len(to_process_A),
        "pairs_B": len(to_process_B),

        "qi_total": len(qi_pack_B),
        "qi_posable": qi_posable,
        "rqi_total": sum(1 for q in qi_pack_B if q.get("rqi")),
        "qc_total": len(qc_pack_B),

        "qi_orphans": len(orphan_ids),
        "orphan_ids_sample": orphan_ids[:30],

        "sanity_ok": sanity_ok,
        "suspicious_docs_count": len(suspicious_all),
        "suspicious_docs": suspicious_all[:20],

        "saturation_ok": saturation_ok,
        "Set_A_qc_count": len(set_A),
        "Set_B_qc_count": len(set_B),
        "Set_B_minus_Set_A": delta_new[:80],

        "sealed": sealed,
        "pack_id": pack.get("pack_id"),
        "pack_signature_sha256": pack.get("pack_signature_sha256"),
        "pack_source": pack.get("_source"),
        "pack_path": pack.get("_pack_file_path"),
    }

    # chapter report (basic counts)
    qc_by_id = {q["qc_id"]: q for q in qc_pack_B}
    qi_by_chapter: Dict[str, List[str]] = {}
    for qi in qi_pack_B:
        qcid = qi.get("qc_id") or ""
        cc = qc_by_id.get(qcid, {}).get("chapter_code", "UNMAPPED") if qcid else "UNMAPPED"
        qi_by_chapter.setdefault(cc, []).append(qi["qi_id"])

    chapter_report = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": pack.get("pack_id"),
        "chapters": [
            {"chapter_code": cc, "qi_total": len(ids), "qc_total": len([q for q in qc_pack_B if q.get("chapter_code")==cc])}
            for cc, ids in sorted(qi_by_chapter.items(), key=lambda x: -len(x[1]))
        ]
    }

    return {
        "qi_pack": qi_pack_B,
        "qc_pack": qc_pack_B,
        "qc_selection": qc_selection,
        "chapter_report": chapter_report,
        "audit": audit,
        "doc_stats": doc_stats_B,
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def metric_row(pairs, corr_ok, qi, qi_posable, qc, sealed):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pairs", pairs)
    c2.metric("Corrigés", corr_ok)
    c3.metric("Qi", qi)
    c4.metric("Qi POSABLE", qi_posable)
    c5.metric("QC", qc)
    c6.metric("SEALED", "YES" if sealed else "NO")

def df_table_library(rows):
    if not rows:
        st.info("Bibliothèque vide. Lance HARVEST AUTO ou upload manuel.")
        return
    cols = ["pair_id", "year", "sujet", "corrige?", "corrige_name", "sujet_url", "corrige_url"]
    view = [{k: r.get(k, "") for k in cols} for r in rows]
    st.dataframe(view, use_container_width=True, hide_index=True)

def main():
    st.set_page_config(page_title=f"SMAXIA GTE Console {APP_VERSION}", layout="wide")
    ss_init()

    st.markdown(f"# SMAXIA GTE Console {APP_VERSION} — ISO-PROD TEST")
    st.caption("UI + Moteur en mono-fichier. Pack FILE signé. F1/F2 auditable. Saturation Set_B-Set_A.")

    with st.sidebar:
        st.markdown("## ÉTAPE 1 — ACTIVATION (Pack FILE signé)")

        country = st.selectbox("Pays", options=[DEFAULT_COUNTRY], index=0)
        st.session_state.country = country

        uploaded_pack = st.file_uploader("Upload Pack JSON (optionnel)", type=["json"])
        genesis_enabled = st.checkbox("Autoriser Pack Genesis (TEST)", value=True)
        genesis_n = st.number_input("Genesis: nombre de chapitres", min_value=3, max_value=60, value=12)

        if st.button("ACTIVER", use_container_width=True):
            try:
                pack_bytes = uploaded_pack.getvalue() if uploaded_pack else None
                pack = pack_activate(country, pack_bytes, genesis_enabled, int(genesis_n))
                st.session_state.pack_active = pack
                st.session_state.pack_meta = {
                    "pack_id": pack.get("pack_id"),
                    "source": pack.get("_source"),
                    "pack_path": pack.get("_pack_file_path"),
                    "signature": pack.get("pack_signature_sha256"),
                }
                st.session_state._pack_file_path = pack.get("_pack_file_path")
                st.success("Pack actif (FILE-based)")
            except Exception as e:
                st.session_state.pack_active = None
                st.session_state.pack_meta = {}
                st.error(f"Activation impossible: {e}")

        st.markdown("---")
        if st.session_state.pack_active:
            pm = st.session_state.pack_meta
            st.success(f"Pack: {pm.get('pack_id')}")
            st.write(f"Source: {pm.get('source')}")
            st.write(f"Signature SHA256: `{pm.get('signature')}`")
            st.write(f"Path: `{pm.get('pack_path')}`")
        else:
            st.warning("Pack inactif")

        st.markdown("---")
        st.markdown("## ÉTAPE 2 — SÉLECTION (scope)")
        level = st.radio("Niveau", ["Seconde", "Première", "Terminale"], index=2)
        level_code = {"Seconde": "SECONDE", "Première": "PREMIERE", "Terminale": "TERMINALE"}[level]
        st.session_state.level = level_code

        subject = st.selectbox("Matière (test)", options=[DEFAULT_SUBJECT], index=0)
        st.session_state.subject = subject

        st.markdown("### Chapitres (pack-driven)")
        if st.session_state.pack_active:
            chs = pack_chapters(st.session_state.pack_active)
            st.caption(f"{len(chs)} chapitres")
            for c in chs[:25]:
                st.write(f"• {c['chapter_code']}")

    tab1, tab2, tab3 = st.tabs(["Import", "RUN", "Exports"])

    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?") and x.get("corrige_url"))

    # Header metrics
    metric_row(
        pairs=len(lib),
        corr_ok=corr_ok,
        qi=st.session_state.run_stats.get("qi", 0),
        qi_posable=st.session_state.run_stats.get("qi_posable", 0),
        qc=st.session_state.run_stats.get("qc", 0),
        sealed=st.session_state.sealed
    )

    with tab1:
        st.markdown("## Import / Bibliothèque")

        st.markdown("### Bibliothèque")
        df_table_library(lib)

        st.markdown("---")
        st.markdown("### HARVEST AUTO (APMEP)")

        c1, c2 = st.columns(2)
        years_back = c1.number_input("Années", min_value=1, max_value=15, value=10)
        volume_max = c2.number_input("Volume max (pairs)", min_value=5, max_value=200, value=50)

        if st.button("HARVEST", use_container_width=True):
            if not st.session_state.pack_active:
                st.error("Active le pack d'abord (ISO-PROD).")
            else:
                try:
                    manifest = harvest_apmep(st.session_state.level, st.session_state.subject, int(years_back), int(volume_max))
                    st.session_state.harvest_manifest = manifest
                    st.session_state.library = manifest["library"]

                    # reset run
                    st.session_state.sealed = False
                    st.session_state.qi_pack = None
                    st.session_state.qc_pack = None
                    st.session_state.qc_selection = None
                    st.session_state.chapter_report = None
                    st.session_state.last_run_audit = None
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0}

                    st.success(f"HARVEST OK: {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                except Exception as e:
                    st.error(f"HARVEST échoué: {e}")

        st.markdown("---")
        st.markdown("### Upload manuel (Sujets/Corrigés)")

        up1, up2 = st.columns(2)
        sujet_file = up1.file_uploader("PDF Sujet", type=["pdf"], key="upl_sujet")
        corr_file = up2.file_uploader("PDF Corrigé", type=["pdf"], key="upl_corr")

        if st.button("Ajouter à la bibliothèque"):
            if not sujet_file:
                st.error("PDF Sujet requis.")
            else:
                su_id = stable_id(sujet_file.name, str(len(sujet_file.getvalue())))
                co_id = stable_id(corr_file.name, str(len(corr_file.getvalue()))) if corr_file else ""

                st.session_state._uploads[f"local://{su_id}"] = sujet_file.getvalue()
                if corr_file:
                    st.session_state._uploads[f"local://{co_id}"] = corr_file.getvalue()

                item = {
                    "pair_id": f"PAIR_{st.session_state.level}|{st.session_state.subject}_{stable_id(su_id, co_id)}",
                    "scope": f"{st.session_state.level}|{st.session_state.subject}",
                    "source": "UPLOAD",
                    "year": "",
                    "sujet": sujet_file.name,
                    "corrige?": bool(corr_file),
                    "corrige_name": corr_file.name if corr_file else "",
                    "reason": "" if corr_file else "corrigé absent",
                    "sujet_url": f"local://{su_id}",
                    "corrige_url": f"local://{co_id}" if corr_file else "",
                }
                st.session_state.library.insert(0, item)
                st.success("Ajouté.")

    with tab2:
        st.markdown("## RUN (preuve F1/F2 + saturation)")

        lib2 = st.session_state.library
        corr_ok2 = sum(1 for x in lib2 if x.get("corrige?") and x.get("corrige_url"))

        if not st.session_state.pack_active:
            st.error("Pack inactif (ISO-PROD).")
        elif not lib2:
            st.warning("Bibliothèque vide.")
        elif corr_ok2 <= 0:
            st.error("Aucun corrigé exploitable (pour ISO-PROD, privilégier Sujet+Corrigé).")
        else:
            max_vol = max(2, min(200, corr_ok2))
            c1, c2 = st.columns(2)
            volume_A = c1.slider("Phase A (pairs)", 1, max_vol, min(10, max_vol))
            volume_B = c2.slider("Phase B (pairs)", 2, max_vol, min(20, max_vol))

            if volume_B < volume_A:
                st.warning("Phase B doit être >= Phase A.")

            if st.button("LANCER", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_granulo_test(
                        library=lib2,
                        volume_A=int(volume_A),
                        volume_B=int(volume_B),
                        pack=st.session_state.pack_active
                    )

                    st.session_state.qi_pack = res["qi_pack"]
                    st.session_state.qc_pack = res["qc_pack"]
                    st.session_state.qc_selection = res["qc_selection"]
                    st.session_state.chapter_report = res["chapter_report"]
                    st.session_state.last_run_audit = res["audit"]
                    st.session_state.sealed = bool(res["audit"]["sealed"])

                    st.session_state.run_stats = {
                        "qi": res["audit"]["qi_total"],
                        "rqi": res["audit"]["rqi_total"],
                        "qc": res["audit"]["qc_total"],
                        "qi_posable": res["audit"]["qi_posable"],
                    }

                    if res["audit"]["sealed"]:
                        st.success("SEALED = YES (saturation OK + no orphans + sanity OK)")
                    else:
                        st.warning(
                            f"SEALED = NO | "
                            f"saturation_ok={res['audit'].get('saturation_ok')} | "
                            f"sanity_ok={res['audit'].get('sanity_ok')} | "
                            f"orphans={res['audit'].get('qi_orphans')} | "
                            f"posable={res['audit'].get('qi_posable')}"
                        )
                except Exception as e:
                    st.error(f"RUN échoué: {e}")

        st.markdown("### Logs")
        st.text_area("", logs_text(), height=320)

    with tab3:
        st.markdown("## Exports + Explorateur de preuve")

        # Exports
        st.markdown("### Exports")
        hm = st.session_state.harvest_manifest or {"version": APP_VERSION, "library": st.session_state.library}
        st.download_button("harvest_manifest.json", json.dumps(hm, ensure_ascii=False, indent=2), "harvest_manifest.json")
        st.download_button("logs.txt", logs_text(), "logs.txt")

        if st.session_state.pack_active:
            pack = st.session_state.pack_active
            st.download_button("pack.json", json.dumps(pack, ensure_ascii=False, indent=2), "pack.json")

        if st.session_state.qi_pack:
            st.download_button("qi_pack.json", json.dumps(st.session_state.qi_pack, ensure_ascii=False, indent=2), "qi_pack.json")
        if st.session_state.qc_pack:
            st.download_button("qc_pack.json", json.dumps(st.session_state.qc_pack, ensure_ascii=False, indent=2), "qc_pack.json")
        if st.session_state.qc_selection:
            st.download_button("qc_selection.json", json.dumps(st.session_state.qc_selection, ensure_ascii=False, indent=2), "qc_selection.json")
        if st.session_state.chapter_report:
            st.download_button("chapter_report.json", json.dumps(st.session_state.chapter_report, ensure_ascii=False, indent=2), "chapter_report.json")
        if st.session_state.last_run_audit:
            st.download_button("run_audit.json", json.dumps(st.session_state.last_run_audit, ensure_ascii=False, indent=2), "run_audit.json")

        st.markdown("---")
        st.markdown("### Explorateur QC → Qi → (RQi, Triggers, F1, ARI, FRT)")

        if st.session_state.qc_pack and st.session_state.qi_pack:
            qc_pack = st.session_state.qc_pack
            qi_pack = st.session_state.qi_pack

            ch_options = ["ALL"] + sorted({qc["chapter_code"] for qc in qc_pack})
            sel_ch = st.selectbox("Chapitre", ch_options)

            qcs = qc_pack if sel_ch == "ALL" else [q for q in qc_pack if q["chapter_code"] == sel_ch]
            if qcs:
                labels = [f"{q['qc_id']} | {q['chapter_code']} | n={q['cluster_size']}" for q in qcs]
                sel_idx = st.selectbox("QC", range(len(labels)), format_func=lambda i: labels[i])
                qc = qcs[sel_idx]

                st.markdown(f"#### QC: {qc['qc']}")
                st.caption(f"qc_id={qc['qc_id']} | chapter={qc['chapter_code']} | cluster_size={qc['cluster_size']}")

                qi_by_id = {q["qi_id"]: q for q in qi_pack}
                for qi_id in qc["qi_ids"][:40]:
                    q = qi_by_id.get(qi_id)
                    if not q:
                        continue
                    with st.expander(f"{qi_id} | has_rqi={q.get('has_rqi')} | orphan={q.get('is_orphan')}"):
                        st.markdown("**Qi**")
                        st.write((q.get("qi") or "")[:1400])

                        st.markdown("**RQi**")
                        st.write((q.get("rqi") or "—")[:1400])

                        st.markdown("**Triggers**")
                        st.write(q.get("triggers"))

                        st.markdown("**F1 (audit)**")
                        st.json(q.get("F1"))

                        st.markdown("**ARI**")
                        st.json(q.get("ari"))

                        st.markdown("**FRT**")
                        st.json(q.get("frt"))

        st.markdown("---")
        st.markdown("### QC selection (F2 coverage-driven)")
        if st.session_state.qc_selection:
            st.dataframe(st.session_state.qc_selection, use_container_width=True, hide_index=True)
        else:
            st.caption("Aucune sélection (F2) disponible (lancer RUN).")

        st.markdown("---")
        st.markdown("### Audit")
        if st.session_state.last_run_audit:
            st.json(st.session_state.last_run_audit)
        else:
            st.caption("Pas d'audit (lancer RUN).")


if __name__ == "__main__":
    main()
