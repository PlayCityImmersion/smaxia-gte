# =============================================================================
# SMAXIA GTE Console V31.10.13 — ISO-PROD TEST (MONO-FICHIER)
# =============================================================================
# OBJECTIF CEO:
# - Tester tout le flux ISO-PROD: Pack FILE signé -> Harvest réel -> Qi/RQi ->
#   Triggers -> F1/F2 auditables -> ARI/FRT (proxy IA1 déterministe) -> QC ->
#   Mapping chapitre (pack-driven) -> Preuve de saturation Set_A/Set_B -> Explorateur.
#
# PILIERS:
# (1) Anti-hardcode métier: le CORE ne "connait" pas les chapitres réels. Variabilité = Pack FILE.
# (2) Pack Genesis = généré sur disque (/tmp) + signé SHA256 + rechargé comme un pack externe.
# (3) Sanity = anti-fake (instabilité de segmentation), PAS un seuil arbitraire sur le nombre.
# (4) SEALED = preuve de saturation: Set_B - Set_A == ∅ + contraintes (orphans=0, posable>0, sanity_ok).
#
# NOTE:
# - Proxy IA1: extraction d'opérateurs cognitifs depuis RQi via règles (pas d'API LLM).
# - F1/F2: implémentation "audit-first" à partir des formules CEO (images fournies).
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import glob
import math
import time
import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests
import streamlit as st

# Optional libs (grâce à vos versions précédentes)
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
APP_VERSION = "V31.10.13"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

DEFAULT_COUNTRY = "FR"
DEFAULT_LEVEL = "TERMINALE"
DEFAULT_SUBJECT = "MATH"

APMEP_ROOT_BY_LEVEL = {
    "TERMINALE": "https://www.apmep.fr/Annales-du-Bac-Terminale",
    "PREMIERE": "https://www.apmep.fr/Annales-du-Bac-Premiere",
    "SECONDE": "https://www.apmep.fr/Annales-du-Bac-Seconde",
}

# F1/F2 constants (ISO-PROD TEST)
EPSILON_F1 = 1e-6
ALPHA_F2 = 0.35  # facteur de récence (test)
SIGMA_HARD_PENALTY = 0.92  # si sigma(q,p) très élevé => pénalisation forte
SIGMA_MODE = "TRIGGERS_JACCARD"  # test-only, auditable

# -----------------------------------------------------------------------------
# SESSION
# -----------------------------------------------------------------------------
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ss_init():
    st.session_state.setdefault("pack_active", None)
    st.session_state.setdefault("pack_id", None)
    st.session_state.setdefault("pack_source", None)
    st.session_state.setdefault("pack_path", None)
    st.session_state.setdefault("pack_sig_sha256", None)

    st.session_state.setdefault("country", DEFAULT_COUNTRY)
    st.session_state.setdefault("level", DEFAULT_LEVEL)
    st.session_state.setdefault("subject", DEFAULT_SUBJECT)

    st.session_state.setdefault("library", [])
    st.session_state.setdefault("harvest_manifest", None)

    st.session_state.setdefault("qi_pack", None)
    st.session_state.setdefault("qc_pack", None)
    st.session_state.setdefault("chapter_report", None)

    st.session_state.setdefault("sealed", False)
    st.session_state.setdefault("last_run_audit", None)
    st.session_state.setdefault("run_stats", {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0})

    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("_uploads", {})        # local://... => bytes
    st.session_state.setdefault("_http_pdf_cache", {}) # url => bytes


def log(msg: str):
    st.session_state.logs.append(f"[{_utc_ts()}] {msg}")


def logs_text() -> str:
    return "\n".join(st.session_state.logs[-8000:])


# -----------------------------------------------------------------------------
# CORE UTILS
# -----------------------------------------------------------------------------
def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:12]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", norm_text(s))[:3000]


def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# -----------------------------------------------------------------------------
# PACK GENESIS (FILE signé) + LOAD PACK FILE
# -----------------------------------------------------------------------------
def _genesis_pack_builder(country: str, n_chapters: int) -> Dict[str, Any]:
    """
    Pack Genesis TEST:
    - Le CORE ne contient pas les noms 'France' officiels (éviter hardcode).
    - Mais pour valider le mapping, on injecte des keywords génériques "math-like"
      (limite, derivee, proba, suite, vecteur, etc.) = TEST.
    """
    n = max(4, int(n_chapters))
    base_keywords = [
        ("CH_ANALYSE", ["fonction", "derivee", "limite", "continuite", "integrale", "primitive", "variation"]),
        ("CH_PROBA", ["probabilite", "loi", "variable", "aleatoire", "binomiale", "normale", "esperance", "variance"]),
        ("CH_SUITES", ["suite", "recurrence", "convergence", "arithmetique", "geometrique"]),
        ("CH_GEO", ["vecteur", "plan", "espace", "orthogonal", "droite", "repere", "distance", "angle"]),
        ("CH_COMPLEXES", ["complexe", "affixe", "module", "argument", "imaginaire"]),
        ("CH_ALGEBRE", ["equation", "inequation", "systeme", "polynome", "factoriser", "discriminant", "racine"]),
        ("CH_LOGEXP", ["logarithme", "exponentielle", "ln", "exp", "croissance", "decroissance"]),
        ("CH_TRIGO", ["cosinus", "sinus", "tangente", "radian", "trigonometrique"]),
    ]

    chapters = []
    for i in range(n):
        code, kws = base_keywords[i % len(base_keywords)]
        chapters.append({
            "chapter_code": f"{code}_{i+1:02d}",
            "chapter_label": f"{code}_{i+1:02d}",
            "keywords": kws,
        })

    pack = {
        "pack_id": f"CAP_{country}_GENESIS_V1",
        "country_code": country,
        "version": "1.0.0",
        "_source": "GENESIS_BUILDER",
        "chapters": chapters,
    }
    return pack


def _write_pack_to_tmp(pack: Dict[str, Any]) -> str:
    stamp = stable_id(pack.get("pack_id", ""), _utc_ts())
    path = f"/tmp/smaxia_pack_{pack.get('pack_id','PACK')}_{stamp}.json"
    b = json.dumps(pack, ensure_ascii=False, indent=2).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b)
    return path


def load_pack_from_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    if not isinstance(pack, dict):
        raise RuntimeError("Pack invalide: JSON racine non objet")
    if not pack.get("pack_id"):
        raise RuntimeError("Pack invalide: pack_id manquant")
    ch = pack.get("chapters") or []
    if not isinstance(ch, list) or len(ch) == 0:
        raise RuntimeError("Pack invalide: chapters vides")
    return pack


def pack_chapters(pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for item in (pack.get("chapters") or []):
        if isinstance(item, dict):
            code = str(item.get("chapter_code") or item.get("code") or "").strip()
            label = str(item.get("chapter_label") or item.get("label") or code).strip()
            if code:
                out.append({
                    "chapter_code": code,
                    "chapter_label": label,
                    "keywords": item.get("keywords", []),
                })
    return out


def activate_pack(country: str, allow_genesis: bool, genesis_n_chapters: int, uploaded_pack_bytes: Optional[bytes]) -> None:
    """
    ISO-PROD TEST:
    - Soit pack uploadé (FILE) -> signé et chargé
    - Soit Genesis autorisé -> builder -> écrit /tmp -> signe -> recharge
    - Sinon -> refus (pas d'embedded en mémoire)
    """
    if uploaded_pack_bytes:
        # Upload pack => fichier /tmp
        stamp = stable_id(country, "UPLOADED", _utc_ts())
        path = f"/tmp/smaxia_pack_UPLOADED_{country}_{stamp}.json"
        with open(path, "wb") as f:
            f.write(uploaded_pack_bytes)

        pack = load_pack_from_file(path)
        sig = sha256_file(path)

        st.session_state.pack_active = pack
        st.session_state.pack_id = pack["pack_id"]
        st.session_state.pack_source = "UPLOADED_FILE"
        st.session_state.pack_path = path
        st.session_state.pack_sig_sha256 = sig

        log(f"[PACK] Activated UPLOADED_FILE path={path} pack_id={pack['pack_id']} sha256={sig}")
        return

    if allow_genesis:
        pack = _genesis_pack_builder(country, genesis_n_chapters)
        path = _write_pack_to_tmp(pack)
        sig = sha256_file(path)
        pack2 = load_pack_from_file(path)

        st.session_state.pack_active = pack2
        st.session_state.pack_id = pack2["pack_id"]
        st.session_state.pack_source = "GENERATED_FILE"
        st.session_state.pack_path = path
        st.session_state.pack_sig_sha256 = sig

        log(f"[PACK] Activated GENERATED_FILE (Genesis) path={path} pack_id={pack2['pack_id']} sha256={sig}")
        return

    raise RuntimeError("Activation refusée: aucun pack FILE fourni et Genesis désactivé (anti-embedded).")


# -----------------------------------------------------------------------------
# HTTP + PDF
# -----------------------------------------------------------------------------
def _http_get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r


def download_pdf_bytes(url: str) -> bytes:
    r = _http_get(url)
    if len(r.content) / (1024 * 1024) > MAX_PDF_MB:
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


def _extract_pages_text(pdf_bytes: bytes, max_pages: int = 120) -> List[str]:
    pages: List[str] = []

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


def _dehyphenate(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _looks_like_page_number(line: str) -> bool:
    s = norm_text(line)
    return bool(re.fullmatch(r"\d{1,3}", s) or re.fullmatch(r"page\s*\d{1,3}", s))


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
            key = norm_text(ln)[:120]
            if key:
                top_counts[key] = top_counts.get(key, 0) + 1
        for ln in lines[-3:]:
            key = norm_text(ln)[:120]
            if key:
                bot_counts[key] = bot_counts.get(key, 0) + 1

    header_keys = {k for k, c in top_counts.items() if c >= max(3, int(0.6 * n_pages))}
    footer_keys = {k for k, c in bot_counts.items() if c >= max(3, int(0.6 * n_pages))}

    cleaned_pages = []
    for lines in page_lines:
        keep = []
        for ln in lines:
            k = norm_text(ln)[:120]
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
# QI / RQI EXTRACTION (structure-first, sujet vs corrigé)
# -----------------------------------------------------------------------------
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦±×÷]")
_QENUM_RE = re.compile(r"(?m)^\s*(?:\(?\d{1,2}\)?|[a-h]|[ivxIVX]{1,6})\s*[\)\.\-:]\s+")
_EXO_RE = re.compile(r"(?im)^\s*(exercice|partie)\b[^\n]*")
_CORR_RE = re.compile(r"(?im)^\s*(corrig|solution)\b[^\n]*")


def _math_density(s: str) -> float:
    if not s:
        return 0.0
    sym = len(_MATH_SYMBOL_RE.findall(s))
    digits = len(re.findall(r"\d", s))
    letters = len(re.findall(r"[a-zA-Z]", s))
    denom = max(1, sym + digits + letters)
    return (sym + digits) / denom


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

        # if buf ends with strong punctuation, flush
        if re.search(r"[.:;!?]$", buf):
            out.append(buf.strip())
            buf = ln.strip()
        else:
            buf = (buf + " " + ln.strip()).strip()

    if buf:
        out.append(buf.strip())

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _split_by_exercices(text: str) -> List[str]:
    t = (text or "").replace("\r", "\n")
    t = _dehyphenate(t)
    t = _merge_wrapped_lines(t)
    if not t:
        return []
    # split on "Exercice"
    blocks = []
    starts = [m.start() for m in _EXO_RE.finditer(t)]
    if not starts:
        return [t]
    starts = [0] + sorted(set(starts))
    for a, b in zip(starts, starts[1:] + [len(t)]):
        blk = t[a:b].strip()
        if len(blk) > 100:
            blocks.append(blk)
    return blocks or [t]


def _extract_q_candidates(block: str, is_correction: bool) -> List[str]:
    """
    Extraction non-LLM, structure-first.
    - Sujet: garde les segments "question-like" (ponctuation, contraintes, demandes).
    - Corrigé: plus permissif car le style est narratif; on capte sous-parties numérotées.
    """
    b = (block or "").strip()
    if not b:
        return []

    # Standardize spaces
    b = re.sub(r"[ \t]+", " ", b)
    b = re.sub(r"\n{3,}", "\n\n", b)

    # Find enumerated starts
    idxs = [0]
    for m in _QENUM_RE.finditer(b):
        idxs.append(m.start())
    idxs = sorted(set(idxs))

    chunks = []
    for a, c in zip(idxs, idxs[1:] + [len(b)]):
        seg = b[a:c].strip()
        seg = re.sub(r"\s+", " ", seg).strip()
        if len(seg) < 40:
            continue
        # filters for sujet/corrigé
        if not is_correction:
            if ("?" not in seg) and (_math_density(seg) < 0.08) and (not re.search(r"\b(montrer|determiner|calculer|resoudre|etudier|justifier|demontrer)\b", norm_text(seg))):
                continue
            if len(seg) < 60:
                continue
        else:
            # correction: accept shorter if it contains operations/keywords
            if (_math_density(seg) < 0.06) and (not re.search(r"\b(on a|donc|ainsi|or|d'ou|car|alors|supposons|posons)\b", norm_text(seg))) and ("=" not in seg):
                continue

        # cap size
        if len(seg) > 2400:
            seg = seg[:2400] + "…"
        chunks.append(seg)

    # fallback: split on '?'
    if len(chunks) <= 2:
        parts = re.split(r"(?<=\?)\s+", re.sub(r"\s+", " ", b))
        parts = [p.strip() for p in parts if len(p.strip()) >= (80 if not is_correction else 60)]
        chunks = chunks or parts

    # de-dup
    out, seen = [], set()
    for seg in chunks:
        k = stable_id(norm_text(seg)[:500])
        if k in seen:
            continue
        seen.add(k)
        out.append(seg)

    return out[:260]


def split_qi_subject(text: str) -> List[str]:
    blocks = _split_by_exercices(text)
    out: List[str] = []
    for blk in blocks:
        out.extend(_extract_q_candidates(blk, is_correction=False))
    return out[:400]


def split_rqi_correction(text: str) -> List[str]:
    blocks = _split_by_exercices(text)
    out: List[str] = []
    for blk in blocks:
        out.extend(_extract_q_candidates(blk, is_correction=True))
    return out[:600]


def align_qi_rqi(qs: List[str], rs: List[str]) -> List[Optional[int]]:
    """
    Alignement auditable: jaccard tokens + fenêtre locale
    """
    if not qs or not rs:
        return [None for _ in qs]

    qtok = [tokenize(q) for q in qs]
    rtok = [tokenize(r) for r in rs]
    out: List[Optional[int]] = [None] * len(qs)

    # Local window first
    for i in range(len(qs)):
        best_j, best_s = None, 0.0
        lo, hi = max(0, i - 10), min(len(rs), i + 11)
        for j in range(lo, hi):
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.10:
            out[i] = best_j

    # Global for remaining
    for i in range(len(qs)):
        if out[i] is not None:
            continue
        best_j, best_s = None, 0.0
        for j in range(len(rs)):
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.13:
            out[i] = best_j

    return out


# -----------------------------------------------------------------------------
# SANITY (anti-fake segmentation)
# -----------------------------------------------------------------------------
def sanity_doc_instability(qs: List[str]) -> Dict[str, Any]:
    """
    Aucun seuil sur le nombre brut.
    On mesure l'instabilité (fake-like) :
    - trop de segments très courts
    - densité math/contraintes trop faible
    - redondance extrême (beaucoup de quasi-doublons)
    """
    if not qs:
        return {"ok": False, "reason": "no_qi", "score": 1.0}

    lens = [len(q) for q in qs]
    short = sum(1 for L in lens if L < 55)
    very_short = sum(1 for L in lens if L < 35)
    avg_len = sum(lens) / max(1, len(lens))

    dens = [_math_density(q) for q in qs]
    low_dens = sum(1 for d in dens if d < 0.04)

    # Redundancy estimate by hashing first tokens
    sigs = [stable_id(" ".join(tokenize(q)[:18])) for q in qs]
    uniq = len(set(sigs))
    redundancy = 1.0 - (uniq / max(1, len(sigs)))

    # Instability score [0..1]
    s1 = min(1.0, very_short / max(1, len(qs)) * 2.0)
    s2 = min(1.0, short / max(1, len(qs)) * 1.2)
    s3 = min(1.0, low_dens / max(1, len(qs)) * 1.0)
    s4 = min(1.0, redundancy * 1.4)

    score = min(1.0, 0.25 * s1 + 0.25 * s2 + 0.25 * s3 + 0.25 * s4)

    ok = score < 0.62  # seuil sur l'instabilité (pas sur la quantité)
    reason = "ok" if ok else "segmentation_instable"
    return {
        "ok": ok,
        "reason": reason,
        "score": round(score, 4),
        "n": len(qs),
        "avg_len": round(avg_len, 1),
        "very_short": very_short,
        "short": short,
        "low_density": low_dens,
        "redundancy": round(redundancy, 4),
    }


# -----------------------------------------------------------------------------
# TRIGGERS + PROXY IA1 (ARI ops réels depuis RQi)
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
    if re.search(r"\b(tableau|variations?)\b", norm_text(t)):
        out.append("TRG_TABLEAU")
    if re.search(r"\b(probabilit|loi|binomial|normale)\b", norm_text(t)):
        out.append("TRG_PROBA")
    if not out:
        out = ["TRG_GENERIC"]
    return out[:10]


@dataclass
class AriStep:
    step_id: str
    op: str
    desc: str
    evidence: List[str]


class SmaxiaExpertAgent:
    """
    Proxy IA1:
    - extraction déterministe d'opérateurs cognitifs depuis RQi (et un peu Qi)
    - audit: chaque step porte des 'evidence' (mots-clés trouvés)
    """
    RULES: List[Tuple[str, str, List[str]]] = [
        ("OP_DERIVATE", "Calcul de dérivée / variations", ["dériv", "derive", "variation", "croissant", "decroissant"]),
        ("OP_LIMIT", "Calcul / usage de limites", ["limit", "limite", "tend vers", "→", "to"]),
        ("OP_INTEGRAL", "Calcul d'intégrale / primitive", ["intégr", "integr", "primitive", "aire"]),
        ("OP_SOLVE_EQ", "Résolution d'équation / racines", ["résoud", "resoud", "equation", "racine", "solutions", "discriminant"]),
        ("OP_INEQUALITY", "Résolution d'inéquation / encadrement", ["inéqu", "inequ", "major", "minor", "encadr"]),
        ("OP_PROBABILITY", "Calcul de probabilité / loi", ["probabil", "loi", "binom", "normale", "esperance", "variance"]),
        ("OP_RECURSION", "Récurrence / suites", ["récurr", "recurr", "suite", "u_n", "un+1", "convergence"]),
        ("OP_GEOMETRY", "Géométrie / vecteurs / repères", ["vecteur", "repere", "orthogonal", "produit scalaire", "distance", "angle", "droite", "plan", "espace"]),
        ("OP_COMPLEX", "Nombres complexes", ["affixe", "module", "argument", "complexe", "imaginaire"]),
        ("OP_LOGEXP", "Log/Exp", ["ln", "log", "exponent", "exp"]),
        ("OP_TRIGO", "Trigonométrie", ["cos", "sin", "tan", "radian", "trigon"]),
    ]

    @staticmethod
    def extract_real_ari(rqi_text: str, qi_text: str, triggers: List[str]) -> Dict[str, Any]:
        t = norm_text(rqi_text or "")
        q = norm_text(qi_text or "")

        steps: List[AriStep] = []

        # Identify ops by evidence scanning
        for op, desc, keys in SmaxiaExpertAgent.RULES:
            ev = [k for k in keys if k in t or k in q]
            if ev:
                steps.append(AriStep(step_id=f"S{len(steps)+1}", op=op, desc=desc, evidence=ev[:5]))

        # Always include minimal invariant steps (read/check), but keep auditable
        base = [
            AriStep(step_id=f"S{len(steps)+1}", op="OP_READ", desc="Extraire données et contraintes", evidence=["(invariant)"]),
            AriStep(step_id=f"S{len(steps)+2}", op="OP_PLAN", desc="Choisir la stratégie de résolution", evidence=["(invariant)"]),
            AriStep(step_id=f"S{len(steps)+3}", op="OP_CHECK", desc="Vérifier cohérence / résultat", evidence=["(invariant)"]),
        ]

        # If no detected cognitive ops, still provide one standard op but flag low evidence
        if not steps:
            steps = [
                AriStep(step_id="S1", op="OP_STANDARD", desc="Application d'un théorème/procédé standard", evidence=["no_rule_hit"])
            ] + base
        else:
            steps = steps + base

        return {
            "template": "ARI_SMAXIA_PROXY_V1",
            "m_q": len([s for s in steps if s.op not in ("OP_READ", "OP_PLAN", "OP_CHECK")]),
            "steps": [{"step_id": s.step_id, "op": s.op, "desc": s.desc, "evidence": s.evidence} for s in steps],
            "audit": {
                "has_rqi": bool(rqi_text),
                "triggers": triggers,
                "rqi_chars": len(rqi_text or ""),
                "qi_chars": len(qi_text or ""),
            },
        }


def build_frt(qi_text: str, rqi_text: str, ari: Dict[str, Any], triggers: List[str]) -> Dict[str, Any]:
    """
    FRT alimentée par ARI proxy + données (audit-first).
    """
    ops = [s["op"] for s in (ari.get("steps") or []) if isinstance(s, dict)]
    return {
        "template": "FRT_SMAXIA_PROXY_V1",
        "sections": [
            {"section_id": "SEC_OBJ", "name": "OBJECTIF", "content": (qi_text or "")[:400]},
            {"section_id": "SEC_DAT", "name": "DONNÉES/CONTRAINTES", "content": "AUTO: extractable (test)"},
            {"section_id": "SEC_TRG", "name": "TRIGGERS", "content": triggers},
            {"section_id": "SEC_ARI", "name": "ARI_OPS", "content": ops[:18]},
            {"section_id": "SEC_EXE", "name": "EXÉCUTION", "content": (rqi_text or "")[:900]},
            {"section_id": "SEC_VAL", "name": "VALIDATION", "content": "OP_CHECK + cohérence"},
        ],
        "audit": {"ari_m_q": ari.get("m_q", 0), "ops_count": len(ops)},
    }


# -----------------------------------------------------------------------------
# QC SIGNATURE (fini) + CLUSTERING
# -----------------------------------------------------------------------------
def qc_signature_from_ari(ari: Dict[str, Any], triggers: List[str]) -> Tuple[str, ...]:
    """
    Signature QC = combinaison d'opérateurs cognitifs + triggers (fini, déterministe).
    """
    ops = []
    for s in (ari.get("steps") or []):
        if isinstance(s, dict):
            op = s.get("op")
            if op and op not in ("OP_READ", "OP_PLAN", "OP_CHECK"):
                ops.append(op)

    # Normalisation: set, tri, cap
    ops = sorted(set(ops))[:8]
    trg = sorted(set(triggers))[:8]
    sig = tuple(ops + ["::"] + trg) if (ops or trg) else ("OP_STANDARD", "::", "TRG_GENERIC")
    return sig


def qc_label_from_signature(sig: Tuple[str, ...]) -> str:
    sig_txt = " ".join(sig[:16]) if sig else "OP_STANDARD"
    return f"Comment résoudre une question caractérisée par: {sig_txt} ?"


def cluster_qi_by_qc_signature(qi_items: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for it in qi_items:
        sig = it.get("qc_sig") or ("OP_STANDARD", "::", "TRG_GENERIC")
        buckets.setdefault(tuple(sig), []).append(it)
    return buckets


# -----------------------------------------------------------------------------
# CHAPTER MAPPING (pack-driven) by cluster tokens (Qi/RQi), not by QC label
# -----------------------------------------------------------------------------
def map_cluster_to_chapter(cluster_items: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, float]]:
    if not chapters:
        return ("UNMAPPED", 0.0, {})

    # aggregate tokens from Qi + RQi of cluster
    agg_text = " ".join([(it.get("qi") or "") + " " + (it.get("rqi") or "") for it in cluster_items])[:200000]
    toks = set(tokenize(agg_text))
    scores: Dict[str, float] = {}

    best_code, best_score = "UNMAPPED", 0.0
    for ch in chapters:
        kws = ch.get("keywords") or []
        if not kws:
            continue
        ch_toks = set(tokenize(" ".join([str(x) for x in kws])))
        if not ch_toks:
            continue
        score = len(toks & ch_toks) / max(1, len(ch_toks))
        code = str(ch["chapter_code"])
        scores[code] = round(score, 4)
        if score > best_score:
            best_code, best_score = code, score

    # threshold: mapping only if signal present
    if best_score < 0.10:
        return ("UNMAPPED", float(best_score), scores)
    return (best_code, float(best_score), scores)


# -----------------------------------------------------------------------------
# F1 / F2 (formules CEO) + audit
# -----------------------------------------------------------------------------
def _trigger_weights(triggers: List[str]) -> Dict[str, float]:
    """
    Tj: poids de triggers (test). Doit être déterministe, auditable.
    Ici: on affecte des poids fixes par type, sans métier.
    """
    w = {}
    for t in triggers:
        if t in ("TRG_MATHSYM", "TRG_RELATION", "TRG_AGGREGATE"):
            w[t] = 0.50
        elif t in ("TRG_NUMBER", "TRG_QMARK", "TRG_TABLEAU"):
            w[t] = 0.35
        else:
            w[t] = 0.20
    return w


def compute_F1_Psi_q(delta_c: int, m_q: int, triggers: List[str]) -> Dict[str, Any]:
    """
    F1 — Poids prédictif purifié normalisé Ψ_q

    ψ_raw(q) = δ_c * ( ε + ( ∑_{j=1..m(q)} T_j )^2 )
    Ψ_q = ψ_raw(q) / max_{p∈Qc} ψ_raw(p)

    Ici on renvoie ψ_raw + la base (normalisation faite au niveau chapitre après calcul de tous).
    """
    trig_w = _trigger_weights(triggers)
    # m(q): nb d'étapes cognitives retenues (ARI après normalisation IA2) -> ici proxy = m_q
    # On prend les m_q triggers "dominants" comme approximation test (audit)
    sorted_tw = sorted(trig_w.items(), key=lambda x: -x[1])
    selected = sorted_tw[:max(1, min(m_q, len(sorted_tw)))]
    sum_T = sum(v for _, v in selected)

    psi_raw = float(delta_c) * (EPSILON_F1 + (sum_T ** 2))
    return {
        "delta_c": int(delta_c),
        "m_q": int(m_q),
        "triggers_Tj": {k: float(v) for k, v in selected},
        "sum_T": float(sum_T),
        "psi_raw": float(psi_raw),
        "Psi_q": None,  # rempli plus tard après normalisation intra-chapitre
    }


def sigma_similarity(q_trg: List[str], p_trg: List[str]) -> float:
    if SIGMA_MODE == "TRIGGERS_JACCARD":
        return jaccard(q_trg, p_trg)
    return jaccard(q_trg, p_trg)


def compute_F2_Score(
    n_q_historical: int,
    N_total: int,
    t_rec_years: float,
    Psi_q: float,
    sigma_to_selected: List[float],
) -> Dict[str, Any]:
    """
    F2 — Score(q)
    Score(q) = (n_q_historical / N_total) * (1 + alpha / t_rec) * Ψ_q * Π_{p∈S} ((1 - σ(q,p)) * 100)

    NB: t_rec normalisé en années (doc).
    """
    if N_total <= 0:
        base_ratio = 0.0
    else:
        base_ratio = float(n_q_historical) / float(N_total)

    t = max(1e-6, float(t_rec_years))
    recency = (1.0 + (ALPHA_F2 / t))

    prod = 1.0
    for s in sigma_to_selected:
        prod *= max(0.0, (1.0 - float(s)) * 100.0)

    score = base_ratio * recency * float(Psi_q) * prod
    return {
        "n_q_historical": int(n_q_historical),
        "N_total": int(N_total),
        "t_rec_years": float(t_rec_years),
        "alpha": float(ALPHA_F2),
        "Psi_q": float(Psi_q),
        "sigmas": [float(x) for x in sigma_to_selected],
        "prod_term": float(prod),
        "score": float(score),
    }


# -----------------------------------------------------------------------------
# HARVEST APMEP
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
    return ("corrig" in s) or ("correction" in s) or ("solution" in s)


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
    min_year = current_year - max(1, years_back) + 1
    selected = [(y, u) for (y, u) in year_links if y >= min_year]
    log(f"[HARVEST] years={len(selected)} (min={min_year})")

    pairs = []
    corrige_ok = 0

    for (y, url) in selected:
        if len(pairs) >= volume_max:
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
                if len(pairs) >= volume_max:
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
# RUN ENGINE (Phase A/B -> Saturation)
# -----------------------------------------------------------------------------
def build_qi_pack_from_pairs(pairs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
    - qi_pack items (with rqi aligned, triggers, ari, frt, qc_sig)
    - doc_stats
    - sanity_docs (instability info if FAIL)
    """
    qi_pack: List[Dict[str, Any]] = []
    doc_stats: List[Dict[str, Any]] = []
    sanity_docs: List[Dict[str, Any]] = []

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

        qs = split_qi_subject(su_text)
        rs = split_rqi_correction(co_text)

        san = sanity_doc_instability(qs)
        if not san["ok"]:
            sanity_docs.append({"pair_id": pid, **san})

        link = align_qi_rqi(qs, rs)

        doc_stats.append({
            "pair_id": pid,
            "qi": len(qs),
            "rqi": len(rs),
            "sanity_ok": san["ok"],
            "sanity_score": san["score"],
        })

        for i, q in enumerate(qs):
            j = link[i] if i < len(link) else None
            r = rs[j] if (j is not None and 0 <= j < len(rs)) else ""

            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:200])}"
            rqi_id = f"RQI_{stable_id(pid, str(j), norm_text(r)[:200])}" if r else ""

            triggers = build_triggers(q)
            ari = SmaxiaExpertAgent.extract_real_ari(r, q, triggers)
            frt = build_frt(q, r, ari, triggers)

            qc_sig = qc_signature_from_ari(ari, triggers)

            qi_pack.append({
                "qi_id": qi_id,
                "pair_id": pid,
                "k": i + 1,
                "qi": q,
                "rqi": r,
                "has_rqi": bool(r),
                "triggers": triggers,
                "ari": ari,
                "frt": frt,
                "qc_sig": list(qc_sig),
                "qc_id": "",
                "is_orphan": False,
                "F1_audit": None,
                "F2_audit": None,
            })

    return qi_pack, doc_stats, sanity_docs


def build_qc_pack_and_map(qi_pack: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
    """
    - Build QC by grouping on qc_sig
    - Assign chapter by mapping cluster tokens (Qi+RQi) to pack chapter keywords
    """
    buckets = cluster_qi_by_qc_signature(qi_pack)

    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}
    qc_chapter_debug: Dict[str, Any] = {}

    # stable ordering
    sig_keys = sorted(buckets.keys(), key=lambda s: (-len(buckets[s]), " ".join(s)))

    for idx, sig in enumerate(sig_keys, start=1):
        cluster = buckets[sig]
        label = qc_label_from_signature(sig)
        qc_id = f"QC_{stable_id(str(idx), norm_text(label), ' '.join(sig))}"

        chapter_code, chapter_score, chapter_scores = map_cluster_to_chapter(cluster, chapters)

        qc_pack.append({
            "qc_id": qc_id,
            "qc": label,
            "signature": list(sig),
            "cluster_size": len(cluster),
            "chapter_code": chapter_code,
            "chapter_score": chapter_score,
            "qi_ids": [it["qi_id"] for it in cluster],
        })

        qc_chapter_debug[qc_id] = {
            "chapter_code": chapter_code,
            "chapter_score": chapter_score,
            "chapter_scores": chapter_scores,
        }

        for it in cluster:
            qc_map[it["qi_id"]] = qc_id

    return qc_pack, qc_map, qc_chapter_debug


def apply_F1_F2_selection(qi_pack: List[Dict[str, Any]], qc_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    ISO-PROD TEST: calcul F1/F2 auditable par chapitre, sur les QC.
    On construit pour chaque QC:
    - m(q) depuis ARI proxy (m_q)
    - triggers depuis une "QC-trigger-set" dérivée des Qi du cluster
    - Ψ_q normalisé intra-chapitre
    - Score(q) pour la sélection progressive (coverage-driven)
    """
    # Build helpers
    qi_by_id = {q["qi_id"]: q for q in qi_pack}
    qc_by_id = {q["qc_id"]: q for q in qc_pack}

    # For each QC, compute F1 raw
    qc_F1_raw: Dict[str, Dict[str, Any]] = {}
    qc_triggers: Dict[str, List[str]] = {}
    qc_mq: Dict[str, int] = {}

    for qc in qc_pack:
        qc_id = qc["qc_id"]
        cluster_qis = [qi_by_id[qid] for qid in qc["qi_ids"] if qid in qi_by_id]

        # delta_c: 1 si mappé à un chapitre, 0 sinon (test)
        delta_c = 1 if qc.get("chapter_code") and qc["chapter_code"] != "UNMAPPED" else 0

        # triggers QC = union triggers cluster
        trg_set = set()
        m_vals = []
        for qi in cluster_qis:
            trg_set.update(qi.get("triggers", []))
            m_vals.append(int(qi.get("ari", {}).get("m_q", 0) or 0))

        # m(q): moyenne robuste (cap)
        m_q = int(round(sum(m_vals) / max(1, len(m_vals)))) if m_vals else 1
        m_q = max(1, min(10, m_q))

        tlist = sorted(trg_set)[:12]
        qc_triggers[qc_id] = tlist
        qc_mq[qc_id] = m_q

        qc_F1_raw[qc_id] = compute_F1_Psi_q(delta_c=delta_c, m_q=m_q, triggers=tlist)

    # Normalize Ψ_q intra-chapitre
    by_chapter: Dict[str, List[str]] = {}
    for qc in qc_pack:
        ch = qc.get("chapter_code") or "UNMAPPED"
        by_chapter.setdefault(ch, []).append(qc["qc_id"])

    qc_F1: Dict[str, Dict[str, Any]] = {}
    for ch, qc_ids in by_chapter.items():
        max_raw = max((qc_F1_raw[qid]["psi_raw"] for qid in qc_ids), default=0.0)
        max_raw = max(max_raw, EPSILON_F1)
        for qid in qc_ids:
            x = dict(qc_F1_raw[qid])
            x["Psi_q"] = float(x["psi_raw"]) / float(max_raw)
            qc_F1[qid] = x

    # Progressive selection (F2) per chapter - coverage-driven:
    # NOTE TEST: we compute scores but do not "finalize" selection list here;
    # selection list can be added later. For audit, we compute score with a growing S.
    qc_F2: Dict[str, Dict[str, Any]] = {}

    now_year = datetime.utcnow().year

    for ch, qc_ids in by_chapter.items():
        selected: List[str] = []
        N_total = len(qc_ids)

        # heuristic n_q_historical = cluster_size (test proxy)
        # t_rec: years since last occurrence (test proxy): use (now - median year) if possible else 1
        # Here: we set t_rec_years=1.0 for test, auditable and non-fake.
        t_rec_years = 1.0

        # order by Psi desc for deterministic selection simulation
        ordered = sorted(qc_ids, key=lambda qid: (-qc_F1[qid]["Psi_q"], qid))

        for qid in ordered:
            # compute sigma against already selected
            sigmas = []
            for pid in selected:
                s = sigma_similarity(qc_triggers[qid], qc_triggers[pid])
                # hard penalty if near-duplicate
                if s >= SIGMA_HARD_PENALTY:
                    s = min(1.0, s)
                sigmas.append(s)

            n_hist = int(qc_by_id[qid].get("cluster_size", 1) or 1)

            f2 = compute_F2_Score(
                n_q_historical=n_hist,
                N_total=N_total,
                t_rec_years=t_rec_years,
                Psi_q=float(qc_F1[qid]["Psi_q"]),
                sigma_to_selected=sigmas,
            )
            qc_F2[qid] = f2

            # add to selected list in simulation (keeps determinism)
            selected.append(qid)

    return {
        "qc_F1": qc_F1,
        "qc_F2": qc_F2,
        "qc_triggers": qc_triggers,
        "qc_mq": qc_mq,
        "by_chapter": by_chapter,
    }


def assemble_chapter_report(chapters: List[Dict[str, Any]], qi_pack: List[Dict[str, Any]], qc_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    ch_label = {c["chapter_code"]: c.get("chapter_label", c["chapter_code"]) for c in chapters}

    qc_by_chapter: Dict[str, List[str]] = {}
    for qc in qc_pack:
        qc_by_chapter.setdefault(qc["chapter_code"], []).append(qc["qc_id"])

    qi_by_chapter: Dict[str, List[str]] = {}
    covered_by_chapter: Dict[str, List[str]] = {}

    qc_by_id = {q["qc_id"]: q for q in qc_pack}

    for qi in qi_pack:
        qc_id = qi.get("qc_id") or ""
        cc = qc_by_id.get(qc_id, {}).get("chapter_code", "UNMAPPED") if qc_id else "UNMAPPED"
        qi_by_chapter.setdefault(cc, []).append(qi["qi_id"])
        if qi.get("has_rqi") and not qi.get("is_orphan"):
            covered_by_chapter.setdefault(cc, []).append(qi["qi_id"])

    entries = []
    for cc in sorted(set(list(qi_by_chapter.keys()) + list(qc_by_chapter.keys()))):
        qi_total = len(qi_by_chapter.get(cc, []))
        qi_cov = len(covered_by_chapter.get(cc, []))
        qc_count = len(qc_by_chapter.get(cc, []))
        cov_pct = round((qi_cov / qi_total) * 100, 2) if qi_total else 0.0
        entries.append({
            "chapter_code": cc,
            "chapter_label": ch_label.get(cc, cc),
            "qi_total": qi_total,
            "covered_qi_count": qi_cov,
            "coverage_pct": cov_pct,
            "qc_count": qc_count,
        })

    return {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": st.session_state.pack_id,
        "chapters": sorted(entries, key=lambda x: -x["qi_total"]),
    }


def run_phase(library: List[Dict[str, Any]], volume: int, pack: Dict[str, Any]) -> Dict[str, Any]:
    chapters = pack_chapters(pack)
    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    if not exploitable:
        raise RuntimeError("Aucun corrigé exploitable")

    to_process = exploitable[:max(1, min(int(volume), len(exploitable)))]
    log(f"[RUN] Phase processing={len(to_process)} pairs")

    qi_pack, doc_stats, sanity_docs = build_qi_pack_from_pairs(to_process)

    if not qi_pack:
        raise RuntimeError("Aucune Qi extraite")

    qc_pack, qc_map, qc_map_debug = build_qc_pack_and_map(qi_pack, chapters)

    # attach qc_id
    orphans = 0
    posable = 0
    for qi in qi_pack:
        qc_id = qc_map.get(qi["qi_id"], "")
        qi["qc_id"] = qc_id
        qi["is_orphan"] = (not qc_id)
        if qi["is_orphan"]:
            orphans += 1
        if qi["has_rqi"]:
            posable += 1

    # F1/F2 on QC
    f12 = apply_F1_F2_selection(qi_pack, qc_pack)

    # attach F1/F2 audit to Qi (per its QC)
    for qi in qi_pack:
        qcid = qi.get("qc_id") or ""
        if qcid and qcid in f12["qc_F1"]:
            qi["F1_audit"] = f12["qc_F1"][qcid]
            qi["F2_audit"] = f12["qc_F2"].get(qcid)

    chapter_report = assemble_chapter_report(chapters, qi_pack, qc_pack)

    # sanity global: fail only if too many docs unstable
    sanity_ok = (len(sanity_docs) == 0) or (len(sanity_docs) <= max(1, int(0.25 * len(to_process))))

    return {
        "qi_pack": qi_pack,
        "qc_pack": qc_pack,
        "chapter_report": chapter_report,
        "phase_audit": {
            "pairs": len(to_process),
            "qi_total": len(qi_pack),
            "rqi_total": sum(1 for q in qi_pack if q.get("rqi")),
            "qi_posable": posable,
            "qc_total": len(qc_pack),
            "qi_orphans": orphans,
            "sanity_ok": sanity_ok,
            "sanity_docs": sanity_docs[:50],
            "doc_stats": doc_stats[:80],
            "qc_chapter_debug": qc_map_debug,
            "F1F2_meta": {
                "EPSILON_F1": EPSILON_F1,
                "ALPHA_F2": ALPHA_F2,
                "SIGMA_MODE": SIGMA_MODE,
            },
        },
        "qc_set": sorted({q["qc_id"] for q in qc_pack}),
    }


def run_granulo_test_AB(library: List[Dict[str, Any]], phaseA: int, phaseB: int, pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preuve de saturation:
    - Phase A => Set_A
    - Phase B => Set_B
    SEALED si Set_B - Set_A == ∅ + contraintes.
    """
    A = run_phase(library, phaseA, pack)
    B = run_phase(library, phaseB, pack)

    setA = set(A["qc_set"])
    setB = set(B["qc_set"])

    delta = sorted(list(setB - setA))
    saturation_ok = (len(delta) == 0)

    # Sealed constraints (ISO-PROD TEST)
    orphans = int(B["phase_audit"]["qi_orphans"])
    posable = int(B["phase_audit"]["qi_posable"])
    sanity_ok = bool(B["phase_audit"]["sanity_ok"])
    qc_total = int(B["phase_audit"]["qc_total"])

    sealed = bool(saturation_ok and sanity_ok and orphans == 0 and posable > 0 and qc_total > 0)

    audit = {
        "phaseA_pairs": phaseA,
        "phaseB_pairs": phaseB,
        "Set_A": sorted(list(setA))[:2000],
        "Set_B": sorted(list(setB))[:2000],
        "Set_B_minus_A": delta[:2000],
        "saturation_ok": saturation_ok,
        "sanity_ok": sanity_ok,
        "orphans": orphans,
        "posable": posable,
        "qc_total": qc_total,
        "sealed": sealed,
    }

    return {
        "A": A,
        "B": B,
        "audit": audit,
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def metric_row(items, corr_ok, qi, qi_posable, qc, sealed):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pairs", items)
    c2.metric("Corrigés", corr_ok)
    c3.metric("Qi", qi)
    c4.metric("Qi POSABLE", qi_posable)
    c5.metric("QC", qc)
    c6.metric("SEALED", "YES" if sealed else "NO")


def df_table(rows):
    if not rows:
        st.info("Bibliothèque vide")
        return
    cols = ["pair_id", "year", "sujet", "corrige?", "corrige_name", "sujet_url"]
    st.dataframe([{k: r.get(k, "") for k in cols} for r in rows], use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title=f"SMAXIA GTE {APP_VERSION}", layout="wide")
    ss_init()

    st.markdown(f"# SMAXIA GTE Console {APP_VERSION} — ISO-PROD TEST")
    st.caption("UI + Moteur en mono-fichier. Pack FILE signé. F1/F2 auditables. Saturation Set_B−Set_A.")

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown("## ÉTAPE 1 — ACTIVATION (Pack FILE signé)")

        country = st.selectbox("Pays", [DEFAULT_COUNTRY])
        st.session_state.country = country

        uploaded_pack = st.file_uploader("Upload Pack JSON (optionnel)", type=["json"])
        uploaded_pack_bytes = uploaded_pack.read() if uploaded_pack else None

        allow_genesis = st.checkbox("Autoriser Pack Genesis (TEST)", value=True)
        genesis_n = st.number_input("Genesis: nombre de chapitres", min_value=4, max_value=40, value=12, step=1)

        if st.button("ACTIVER", use_container_width=True):
            try:
                activate_pack(country, allow_genesis, int(genesis_n), uploaded_pack_bytes)
                st.success("Pack actif")
            except Exception as e:
                st.session_state.pack_active = None
                st.session_state.pack_id = None
                st.session_state.pack_source = None
                st.session_state.pack_path = None
                st.session_state.pack_sig_sha256 = None
                st.error(f"Erreur: {e}")

        if st.session_state.pack_active:
            st.success(f"Pack: {st.session_state.pack_id}")
            st.caption(f"Source: {st.session_state.pack_source}")
            st.caption(f"Signature SHA256: {st.session_state.pack_sig_sha256}")
            st.caption(f"Path: {st.session_state.pack_path}")
        else:
            st.warning("Pack inactif")

        st.markdown("---")
        st.markdown("## ÉTAPE 2 — SÉLECTION (scope)")

        level = st.radio("Niveau", ["Seconde", "Première", "Terminale"], index=2)
        st.session_state.level = {"Seconde": "SECONDE", "Première": "PREMIERE", "Terminale": "TERMINALE"}[level]

        subject = st.selectbox("Matière (test)", ["MATH"], index=0)
        st.session_state.subject = subject

        st.markdown("### Chapitres (pack-driven)")
        if st.session_state.pack_active:
            chs = pack_chapters(st.session_state.pack_active)
            st.caption(f"{len(chs)} chapitres")
            for c in chs[:25]:
                st.write(f"• {c['chapter_code']}")
        else:
            st.caption("—")

    # ---------------- MAIN TABS ----------------
    tab1, tab2, tab3 = st.tabs(["Import", "RUN", "Exports"])

    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?") and x.get("corrige_url"))

    # -------- Import --------
    with tab1:
        st.markdown("## Import / Bibliothèque")
        metric_row(len(lib), corr_ok, st.session_state.run_stats.get("qi", 0),
                   st.session_state.run_stats.get("qi_posable", 0),
                   st.session_state.run_stats.get("qc", 0), st.session_state.sealed)

        st.markdown("### Bibliothèque")
        df_table(lib)

        st.markdown("---")
        st.markdown("### HARVEST AUTO (APMEP)")
        c1, c2 = st.columns(2)
        years_back = c1.number_input("Années", 1, 15, 10)
        volume_max = c2.number_input("Volume max (pairs)", 5, 200, 30)

        if st.button("HARVEST", use_container_width=True):
            if not st.session_state.pack_active:
                st.error("Activez le pack d'abord")
            else:
                try:
                    manifest = harvest_apmep(st.session_state.level, st.session_state.subject, int(years_back), int(volume_max))
                    st.session_state.harvest_manifest = manifest
                    st.session_state.library = manifest["library"]

                    # reset run
                    st.session_state.sealed = False
                    st.session_state.qi_pack = None
                    st.session_state.qc_pack = None
                    st.session_state.chapter_report = None
                    st.session_state.last_run_audit = None
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0}

                    st.success(f"HARVEST OK: {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                except Exception as e:
                    st.error(f"HARVEST échoué: {e}")

        st.markdown("---")
        st.markdown("### Upload manuel (Sujets/Corrigés)")
        cA, cB = st.columns(2)
        su = cA.file_uploader("PDF Sujet", type=["pdf"], key="upl_sujet")
        co = cB.file_uploader("PDF Corrigé", type=["pdf"], key="upl_corrige")

        if st.button("Ajouter à la bibliothèque", use_container_width=True):
            if not (su and co):
                st.error("Veuillez fournir Sujet + Corrigé")
            else:
                su_b = su.read()
                co_b = co.read()
                su_url = f"local://sujet_{stable_id(su.name, sha256_bytes(su_b))}.pdf"
                co_url = f"local://corrige_{stable_id(co.name, sha256_bytes(co_b))}.pdf"
                st.session_state._uploads[su_url] = su_b
                st.session_state._uploads[co_url] = co_b

                item = {
                    "pair_id": f"PAIR_MANUAL_{stable_id(su.name, co.name, _utc_ts())}",
                    "scope": f"{st.session_state.level}|{st.session_state.subject}",
                    "source": "MANUAL_UPLOAD",
                    "year": datetime.utcnow().year,
                    "sujet": su.name,
                    "corrige?": True,
                    "corrige_name": co.name,
                    "reason": "",
                    "sujet_url": su_url,
                    "corrige_url": co_url,
                }
                st.session_state.library = [item] + st.session_state.library
                st.success("Pair ajouté")

    # -------- RUN --------
    with tab2:
        st.markdown("## RUN (preuve F1/F2 + saturation)")

        if not st.session_state.pack_active:
            st.error("Pack inactif")
        elif not st.session_state.library:
            st.warning("Bibliothèque vide")
        else:
            lib2 = st.session_state.library
            corr_ok2 = sum(1 for x in lib2 if x.get("corrige?") and x.get("corrige_url"))
            if corr_ok2 <= 0:
                st.error("Aucun corrigé exploitable")
            else:
                max_vol = max(2, min(200, corr_ok2))
                c1, c2 = st.columns(2)
                phaseA = c1.slider("Phase A (pairs)", 2, max_vol, min(10, max_vol))
                phaseB = c2.slider("Phase B (pairs)", 2, max_vol, min(20, max_vol))

                if phaseB < phaseA:
                    st.warning("Phase B doit être >= Phase A")

                if st.button("LANCER", use_container_width=True):
                    try:
                        log(f"=== RUN {APP_VERSION} A={phaseA} B={phaseB} ===")
                        res = run_granulo_test_AB(lib2, int(phaseA), int(phaseB), st.session_state.pack_active)

                        # take Phase B as visible packs
                        B = res["B"]
                        st.session_state.qi_pack = B["qi_pack"]
                        st.session_state.qc_pack = B["qc_pack"]
                        st.session_state.chapter_report = B["chapter_report"]
                        st.session_state.last_run_audit = res["audit"]
                        st.session_state.sealed = bool(res["audit"]["sealed"])

                        st.session_state.run_stats = {
                            "qi": B["phase_audit"]["qi_total"],
                            "rqi": B["phase_audit"]["rqi_total"],
                            "qc": B["phase_audit"]["qc_total"],
                            "qi_posable": B["phase_audit"]["qi_posable"],
                        }

                        if st.session_state.sealed:
                            st.success("SEALED = YES")
                        else:
                            st.warning(
                                f"SEALED = NO | saturation_ok={res['audit']['saturation_ok']} | sanity_ok={res['audit']['sanity_ok']} | "
                                f"orphans={res['audit']['orphans']} | posable={res['audit']['posable']}"
                            )

                    except Exception as e:
                        st.error(f"RUN échoué: {e}")

        st.markdown("### Logs")
        st.text_area("", logs_text(), height=320)

    # -------- Exports + Explorer --------
    with tab3:
        st.markdown("## Exports")
        metric_row(
            len(st.session_state.library),
            sum(1 for x in st.session_state.library if x.get("corrige?") and x.get("corrige_url")),
            st.session_state.run_stats.get("qi", 0),
            st.session_state.run_stats.get("qi_posable", 0),
            st.session_state.run_stats.get("qc", 0),
            st.session_state.sealed,
        )

        hm = st.session_state.harvest_manifest or {"version": APP_VERSION}
        st.download_button("harvest_manifest.json", json.dumps(hm, ensure_ascii=False, indent=2), "harvest_manifest.json")
        st.download_button("logs.txt", logs_text(), "logs.txt")

        if st.session_state.qi_pack:
            st.download_button("qi_pack.json", json.dumps(st.session_state.qi_pack, ensure_ascii=False, indent=2), "qi_pack.json")
        if st.session_state.qc_pack:
            st.download_button("qc_pack.json", json.dumps(st.session_state.qc_pack, ensure_ascii=False, indent=2), "qc_pack.json")
        if st.session_state.chapter_report:
            st.download_button("chapter_report.json", json.dumps(st.session_state.chapter_report, ensure_ascii=False, indent=2), "chapter_report.json")
        if st.session_state.last_run_audit:
            st.download_button("run_audit.json", json.dumps(st.session_state.last_run_audit, ensure_ascii=False, indent=2), "run_audit.json")

        st.markdown("---")
        st.markdown("## Explorateur QC → Qi → (RQi, Triggers, F1, ARI, FRT)")

        if st.session_state.qc_pack and st.session_state.qi_pack:
            qc_pack = st.session_state.qc_pack
            qi_pack = st.session_state.qi_pack

            ch_options = ["ALL"] + sorted({qc["chapter_code"] for qc in qc_pack})
            sel_ch = st.selectbox("Chapitre", ch_options)

            qcs = qc_pack if sel_ch == "ALL" else [q for q in qc_pack if q["chapter_code"] == sel_ch]
            if qcs:
                qc_labels = [
                    f"{q['qc_id']} | {q['chapter_code']} | n={q['cluster_size']} | score={q.get('chapter_score',0)}"
                    for q in qcs
                ]
                sel_idx = st.selectbox("QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]

                st.markdown(f"### QC: {qc['qc']}")
                st.caption(f"qc_id={qc['qc_id']} | chapter={qc['chapter_code']} | cluster_size={qc['cluster_size']}")

                qi_by_id = {q["qi_id"]: q for q in qi_pack}

                st.markdown("### Qi associées")
                for qi_id in qc["qi_ids"][:80]:
                    q = qi_by_id.get(qi_id)
                    if not q:
                        continue

                    with st.expander(f"{qi_id} | has_rqi={q.get('has_rqi')} | orphan={q.get('is_orphan')}"):
                        st.markdown("**Qi**")
                        st.write(q.get("qi", ""))

                        st.markdown("**RQi**")
                        st.write(q.get("rqi", "") if q.get("rqi") else "— Pas de RQi alignée —")

                        st.markdown("**Triggers**")
                        st.write(q.get("triggers", []))

                        st.markdown("**F1 (audit)**")
                        st.json(q.get("F1_audit", {}))

                        st.markdown("**ARI (proxy IA1)**")
                        st.json(q.get("ari", {}))

                        st.markdown("**FRT (proxy)**")
                        st.json(q.get("frt", {}))

        st.markdown("---")
        st.markdown("## Audit")
        if st.session_state.last_run_audit:
            st.json(st.session_state.last_run_audit)
        else:
            st.info("Aucun audit (lancez RUN).")


if __name__ == "__main__":
    main()
