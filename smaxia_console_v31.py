# =============================================================================
# SMAXIA GTE Console V31.10.24 — ISO-PROD (FICHIER UNIQUE)
# =============================================================================
# OBJECTIF V31.10.24 (KERNEL V10.6.1 / compat V10.6.2 sur F1/F2 gates) :
# - CAS 1 ONLY (zéro reconstruction / zéro invention) : Qi/RQi proviennent UNIQUEMENT des PDFs Sujet+Corrigé
# - Zéro hardcode métier dans le CORE : toute variabilité est Pack-Driven
# - Atomisation robuste Qi/RQi (multi-stratégies) + OCR fallback optionnel
# - QC / ARI / FRT / Triggers générés sans erreur + preuves exportées
# - F1 et F2 conformes aux formules Kernel rappelées :
#     F1 = δc · ( ε + Σ Tj )²
#     F2 = (n_hist/N) · (1 + α/t_rec) · Ψq · Π(1-σ)
# - Saturation progressive : vol_start → vol_max jusqu'à new_QC=0
# - Quarantaine RC_* + Audit IA2 booléen + Signature SHA256 de tous outputs
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
APP_VERSION = "V31.10.24"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 35
MAX_PDF_MB = 40

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
            # IMPORTANT: liste “safe” (évite corruption de quote lors des copies)
            "geographic_zones": [
                "metro", "metropole", "amerique", "asie", "polynesie",
                "etranger", "antilles", "liban"
            ],
            # date parsing from filename/text (pack-driven)
            "date_regexes": [
                r"(\d{2})[_-](\d{2})[_-](20\d{2})",  # dd_mm_yyyy
                r"(20\d{2})[_-](\d{2})[_-](\d{2})",  # yyyy_mm_dd
            ],
        }],

        # ---------------- Text processing ----------------
        "text_processing": {
            "glued_patterns": [
                {"pattern": r"(\d)([a-zA-Z])", "replacement": r"\1 \2"},
                {"pattern": r"([a-zA-Z])(\d)", "replacement": r"\1 \2"},
                {"pattern": r"([a-z])([A-Z])", "replacement": r"\1 \2"},
            ],
            "intent_verbs": [
                "montrer", "demontrer", "prouver", "justifier", "determiner", "calculer",
                "resoudre", "etudier", "donner", "exprimer", "simplifier", "trouver",
                "verifier", "deduire", "conclure", "etablir", "tracer", "representer",
                "interpreter", "comparer", "encadrer"
            ],
            "trigger_keywords": [
                "limite", "derivee", "integrale", "probabilite", "suite", "vecteur",
                "complexe", "equation", "fonction", "recurrence", "matrice", "logarithme",
                "exponentielle", "primitive", "continuite", "asymptote", "variance", "esperance"
            ],
        },

        # ---------------- Atomization rules (pack-driven) ----------------
        "atomization": {
            "max_pages": 140,
            "min_segment_len": 20,
            "max_segment_len": 2600,

            # Multi-strategies (regexes are pack data)
            "markers": {
                "exercise": [r"(?im)^\s*(exercice|exo\.?)\s*([0-9ivx]+)\b"],
                "part": [r"(?im)^\s*(partie)\s*([a-z0-9]+)\b"],
                "question": [
                    r"(?m)^\s*(\d{1,2})\s*[\)\.\-:]\s+",
                    r"(?m)^\s*([a-h])\s*[\)\.\-:]\s+",
                    r"(?m)^\s*\(\s*([a-h])\s*\)\s+",
                ],
                "qcm": [r"(?im)^\s*qcm\b", r"(?im)^\s*questionnaire\b"],
            },

            # If PDFs lose line breaks, we inject breaks before markers
            "inject_newlines": [
                r"(?i)\bexercice\s*(\d+|[ivx]+)\b",
                r"(?i)\bpartie\s*[a-z0-9]+\b",
                r"(?m)\s(\d{1,2})\)",
                r"(?m)\s([a-h])\)",
            ],

            # Heuristics thresholds (pack-driven)
            "question_keep": {
                "min_digits_or_symbols": 1,
                "min_len": 25,
                "max_len": 2200,
            },
            "answer_keep": {
                "min_len": 30,
                "max_len": 2600,
            },

            # Alignment thresholds (pack-driven)
            "alignment": {
                "min_score": 0.18,
                "min_char_ngram": 0.10,
                "max_candidates": 900,
            },

            # OCR policy (pack-driven)
            "ocr": {
                "enabled": True,
                "min_chars_before_ocr": 1200,
                "max_ocr_pages": 8,
                "ocr_dpi": 220,
            }
        },

        # ---------------- ARI config (pack-driven) ----------------
        "ari_config": {
            "op_patterns": [
                {"op": "OP_PROBABILITY", "pattern": r"\b(probabilit|proba|loi\s+binomiale|loi\s+normale|esperance|variance|aleatoire)\b", "weight": 1.0},
                {"op": "OP_DERIVE", "pattern": r"\b(deriv|f'\s*\(|tangente|taux\s+de\s+variation)\b", "weight": 1.0},
                {"op": "OP_INTEGRATE", "pattern": r"\b(integr|primitive|aire\s+sous|calcul\s+d'aire)\b", "weight": 1.0},
                {"op": "OP_LIMIT", "pattern": r"\b(limit|tend\s+vers|infini|convergence)\b", "weight": 1.0},
                {"op": "OP_INDUCTION", "pattern": r"\b(recurr|induction|heredite|initialisation|rang)\b", "weight": 1.0},
                {"op": "OP_COMPLEX", "pattern": r"\b(complex|imaginaire|module|argument|affixe|conjugue)\b", "weight": 1.0},
                {"op": "OP_VECTOR_GEOM", "pattern": r"\b(vecteur|scalaire|orthogonal|colineaire|produit\s+scalaire)\b", "weight": 1.0},
                {"op": "OP_LOGEXP", "pattern": r"\b(ln|log|exp|exponentiel|logarithme)\b", "weight": 1.0},
                {"op": "OP_SOLVE_EQUATION", "pattern": r"\b(equat|resou|racine|solution|discriminant|inequation)\b", "weight": 1.0},
                {"op": "OP_VARIATION_TABLE", "pattern": r"\b(tableau\s+de\s+variation|signe|croissant|decroissant|extremum)\b", "weight": 1.0},
                {"op": "OP_PROVE", "pattern": r"\b(demontr|prouv|justifi|montr|etablir)\b", "weight": 0.8},
            ],
            "primary_ops_order": [
                "OP_PROBABILITY", "OP_INTEGRATE", "OP_DERIVE", "OP_LIMIT",
                "OP_INDUCTION", "OP_COMPLEX", "OP_VECTOR_GEOM",
                "OP_LOGEXP", "OP_SOLVE_EQUATION", "OP_VARIATION_TABLE", "OP_PROVE",
                "OP_STANDARD"
            ],
            "op_labels": {
                "OP_PROBABILITY": "calculer une probabilité",
                "OP_DERIVE": "dériver une fonction",
                "OP_INTEGRATE": "calculer une intégrale",
                "OP_LIMIT": "calculer une limite",
                "OP_INDUCTION": "démontrer par récurrence",
                "OP_COMPLEX": "manipuler des nombres complexes",
                "OP_VECTOR_GEOM": "résoudre un problème de géométrie vectorielle",
                "OP_LOGEXP": "utiliser logarithme et exponentielle",
                "OP_SOLVE_EQUATION": "résoudre une équation ou inéquation",
                "OP_VARIATION_TABLE": "établir un tableau de variation",
                "OP_PROVE": "démontrer une propriété",
                "OP_STANDARD": "résoudre un exercice standard",
            },

            # FRT schema and templates are PACK DATA (no hardcode in core)
            "frt_schema_required_sections": ["context", "triggers", "ari", "procedure", "checks", "pitfalls", "output"],
            "frt_templates": {
                "OP_PROBABILITY": {
                    "procedure": [
                        "Identifier l’expérience aléatoire et l’univers Ω",
                        "Définir précisément les événements",
                        "Identifier la loi et les paramètres",
                        "Appliquer la formule / outil adapté",
                        "Calculer, vérifier, conclure"
                    ],
                    "checks": ["Cohérence des événements", "Valeur dans [0,1]", "Interprétation correcte"],
                    "pitfalls": ["Confusion événements", "Mauvaise loi", "Arrondis prématurés"]
                },
                "OP_DERIVE": {
                    "procedure": [
                        "Identifier f et le domaine",
                        "Reconnaître la forme (produit/quotient/composition)",
                        "Appliquer les règles de dérivation",
                        "Simplifier f'(x)",
                        "Exploiter (variation/tangente) si demandé"
                    ],
                    "checks": ["Domaine", "Simplification", "Signe de f' si tableau"],
                    "pitfalls": ["Erreur de règle", "Oubli domaine", "Signe mal étudié"]
                },
                "OP_STANDARD": {
                    "procedure": [
                        "Lire et reformuler la demande",
                        "Lister données / inconnues",
                        "Choisir la méthode",
                        "Exécuter les calculs",
                        "Vérifier et conclure"
                    ],
                    "checks": ["Données utilisées", "Résultat plausible", "Conclusion explicite"],
                    "pitfalls": ["Mauvaise méthode", "Erreurs algébriques", "Conclusion absente"]
                }
            }
        },

        # ---------------- Chapter taxonomy (pack-driven) ----------------
        "chapter_taxonomy": {
            "MATH": {
                "TERMINALE": [
                    {"code": "CH_ANALYSE", "label": "Analyse", "keywords": ["limite", "derivee", "asymptote", "variation", "tangente", "convexite", "continuite", "fonction"], "delta_c": 1.0},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "keywords": ["probabilite", "loi", "binomiale", "normale", "esperance", "variance", "aleatoire"], "delta_c": 1.0},
                    {"code": "CH_SUITES", "label": "Suites", "keywords": ["suite", "recurrence", "arithmetique", "geometrique", "convergence", "rang"], "delta_c": 1.0},
                    {"code": "CH_INTEGRATION", "label": "Intégration", "keywords": ["integrale", "primitive", "aire", "valeur moyenne"], "delta_c": 1.0},
                    {"code": "CH_GEOMETRIE", "label": "Géométrie espace", "keywords": ["vecteur", "plan", "droite", "espace", "orthogonal", "colineaire", "scalaire"], "delta_c": 1.0},
                    {"code": "CH_COMPLEXES", "label": "Nombres complexes", "keywords": ["complexe", "imaginaire", "module", "argument", "affixe", "conjugue"], "delta_c": 1.0},
                    {"code": "CH_LOGEXP", "label": "Log & Exp", "keywords": ["exponentielle", "logarithme", "ln", "exp"], "delta_c": 1.0},
                ]
            }
        },

        # ---------------- QC format (pack-driven) ----------------
        "qc_format": {
            "prefix": "Comment",
            "suffix": "?",
            "template": "{prefix} {label} {suffix}"
        },

        # ---------------- Trigger weighting for F1 (pack-driven) ----------------
        "trigger_weights": {
            "default": 0.6,
            "by_prefix": {
                "ARI": 1.0,
                "KW": 0.7,
                "INTENT": 0.6,
                "SCOPE": 0.5
            }
        },

        # ---------------- POSABLE rules (pack-driven) ----------------
        "posable_rules": {
            "require_rqi": True,
            "require_scope": True,
            "require_evaluable": True,
            "min_ari_confidence": 0.55,
            "min_answer_len": 20
        },

        # ---------------- F1/F2 params (Kernel inputs are pack data) ----------------
        "f1_f2_params": {
            "epsilon": 1e-6,
            "alpha": 1.0,
            "t_rec_min": 1.0,

            # seal gates (pack-driven)
            "seal_coverage_threshold": 0.80,
            "seal_sanity_dup_ratio_max": 0.35,

            # anti-faux SEALED: extraction KPI
            "seal_min_avg_qi_per_pair": 8.0,
            "seal_min_total_qi": 160,  # ~10 sujets * 16 Qi
        },

        # ---------------- Saturation (pack-driven) ----------------
        "saturation": {
            "stop_on_new_qc_zero": True
        }
    }

    # country naming is pack data (not core)
    if country_code == "FR":
        base["country_name"] = "France"
    elif country_code == "CI":
        base["country_name"] = "Côte d'Ivoire"
    elif country_code == "SN":
        base["country_name"] = "Sénégal"
    else:
        base["country_name"] = country_code

    base["_pack_sig_sha256"] = sha256_text(safe_json_dumps(base))
    return base

GENESIS_PACKS = {
    "FR": "France",
    "CI": "Côte d'Ivoire",
    "SN": "Sénégal",
}

# =============================================================================
# Pack accessors (core invariant)
# =============================================================================
def pack_get(pack: Dict[str, Any], path: str, default=None):
    cur = pack
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def pack_chapters(pack: Dict[str, Any], subject: str, level: str) -> List[Dict[str, Any]]:
    return pack_get(pack, f"chapter_taxonomy.{subject}.{level}", []) or []

def pack_source(pack: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    srcs = pack.get("harvest_sources", [])
    return srcs[0] if srcs else None

# =============================================================================
# HTTP fetch / caching
# =============================================================================
def _http_get(url: str) -> requests.Response:
    res = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    res.raise_for_status()
    return res

def fetch_pdf_bytes(url: str) -> bytes:
    cache = st.session_state.get("_http_pdf_cache", {})
    if url in cache:
        return cache[url]
    res = _http_get(url)
    pdf_bytes = res.content
    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError("PDF trop volumineux")
    cache[url] = pdf_bytes
    st.session_state["_http_pdf_cache"] = cache
    return pdf_bytes

# =============================================================================
# PDF text extraction (multi-backend + optional OCR)
# =============================================================================
def _fix_missing_spaces(text: str, text_proc: Dict[str, Any]) -> str:
    if not text:
        return ""
    for item in (text_proc or {}).get("glued_patterns", []):
        try:
            text = re.sub(item["pattern"], item["replacement"], text)
        except Exception:
            pass
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _dehyphenate(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text

def _extract_pdf_pdfplumber(pdf_bytes: bytes, max_pages: int) -> List[str]:
    if not pdfplumber:
        return []
    out = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:max_pages]:
                try:
                    out.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
                except Exception:
                    out.append("")
    except Exception:
        return []
    return out

def _extract_pdf_pypdf(pdf_bytes: bytes, max_pages: int) -> List[str]:
    if not PdfReader:
        return []
    out = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages[:max_pages]:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                out.append("")
    except Exception:
        return []
    return out

def _extract_pdf_pymupdf_text(pdf_bytes: bytes, max_pages: int) -> List[str]:
    if not fitz:
        return []
    out = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(min(max_pages, doc.page_count)):
            try:
                out.append(doc.load_page(i).get_text("text") or "")
            except Exception:
                out.append("")
    except Exception:
        return []
    return out

def _ocr_pdf_pymupdf(pdf_bytes: bytes, max_pages: int, dpi: int) -> List[str]:
    # Requires fitz + pytesseract + tesseract binary in environment
    if not fitz or not pytesseract:
        return []
    out = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(min(max_pages, doc.page_count)):
            page = doc.load_page(i)
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            try:
                from PIL import Image  # type: ignore
                img = Image.open(io.BytesIO(img_bytes))
                txt = pytesseract.image_to_string(img, lang="fra+eng")
            except Exception:
                txt = ""
            out.append(txt or "")
    except Exception:
        return []
    return out

def clean_pdf_text(pages: List[str], text_proc: Dict[str, Any]) -> str:
    if not pages:
        return ""
    lines_by_page: List[List[str]] = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = _dehyphenate(p)
        p = _fix_missing_spaces(p, text_proc)
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        lines_by_page.append(lines)

    # header/footer removal (invariant)
    top_counts = Counter()
    bot_counts = Counter()
    for lines in lines_by_page:
        if lines:
            top_counts[lines[0]] += 1
            bot_counts[lines[-1]] += 1
    threshold = max(2, len(lines_by_page) // 3)
    skip_top = {k for k, v in top_counts.items() if v >= threshold}
    skip_bot = {k for k, v in bot_counts.items() if v >= threshold}

    out_lines: List[str] = []
    for lines in lines_by_page:
        for i, ln in enumerate(lines):
            if i == 0 and ln in skip_top:
                continue
            if i == len(lines) - 1 and ln in skip_bot:
                continue
            if re.fullmatch(r"\d{1,3}", ln):
                continue
            out_lines.append(ln)
        out_lines.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()

def extract_text_from_pdf(pdf_bytes: bytes, pack: Dict[str, Any]) -> Tuple[str, PdfExtractMeta]:
    text_proc = pack_get(pack, "text_processing", {}) or {}
    atom = pack_get(pack, "atomization", {}) or {}
    max_pages = int(atom.get("max_pages", 140))
    ocr_cfg = atom.get("ocr", {}) or {}
    ocr_enabled = bool(ocr_cfg.get("enabled", True))
    ocr_min_chars = int(ocr_cfg.get("min_chars_before_ocr", 1200))
    ocr_pages = int(ocr_cfg.get("max_ocr_pages", 8))
    ocr_dpi = int(ocr_cfg.get("ocr_dpi", 220))

    t0 = time.time()
    ocr_used = False

    # strategy 1: pdfplumber
    pages = _extract_pdf_pdfplumber(pdf_bytes, max_pages)
    txt = clean_pdf_text(pages, text_proc)
    method = "pdfplumber"

    # strategy 2: pypdf
    if len(txt) < 500:
        pages2 = _extract_pdf_pypdf(pdf_bytes, max_pages)
        txt2 = clean_pdf_text(pages2, text_proc)
        if len(txt2) > len(txt):
            txt, pages, method = txt2, pages2, "pypdf"

    # strategy 3: pymupdf text
    if len(txt) < 700:
        pages3 = _extract_pdf_pymupdf_text(pdf_bytes, max_pages)
        txt3 = clean_pdf_text(pages3, text_proc)
        if len(txt3) > len(txt):
            txt, pages, method = txt3, pages3, "pymupdf_text"

    # strategy 4: OCR fallback (optional)
    if ocr_enabled and len(txt) < ocr_min_chars:
        pages4 = _ocr_pdf_pymupdf(pdf_bytes, ocr_pages, ocr_dpi)
        txt4 = clean_pdf_text(pages4, text_proc)
        if len(txt4) > len(txt):
            txt, pages, method = txt4, pages4, "pymupdf_ocr"
            ocr_used = True

    dt = time.time() - t0
    meta = PdfExtractMeta(method=method, pages=len(pages), chars=len(txt), seconds=round(dt, 3), ocr_used=ocr_used)
    return txt, meta

# =============================================================================
# Harvest (pack-driven)
# =============================================================================
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé")
    return BeautifulSoup(html, "html.parser")

def _abs_url(base: str, href: str) -> str:
    return href if href.startswith("http") else requests.compat.urljoin(base, href)

def harvest_from_pack(pack: Dict[str, Any], level: str, years_back: int, volume_max: int) -> Dict[str, Any]:
    src = pack_source(pack)
    if not src:
        raise ValueError("Aucune source harvest dans le Pack")
    base_url = src.get("base_url", "")
    level_path = (src.get("levels", {}) or {}).get(level)
    if not level_path:
        raise ValueError(f"Niveau '{level}' non configuré")
    root = f"{base_url}{level_path}"
    log(f"[HARVEST] root={root}")

    year_pattern = src.get("year_pattern", r"(20\d{2})")
    meta_exclude = src.get("meta_exclude", [])
    corrige_patterns = ((src.get("pdf_patterns", {}) or {}).get("corrige", [])) or []
    geo_zones = src.get("geographic_zones", []) or []

    html = _http_get(root).text
    sp = _soup(html)

    year_links: List[Tuple[int, str]] = []
    for a in sp.find_all("a"):
        href = a.get("href") or ""
        m = re.search(year_pattern, href)
        if m:
            try:
                year_links.append((int(m.group(1)), _abs_url(root, href)))
            except Exception:
                pass

    year_links = sorted(set(year_links), key=lambda x: -x[0])
    if not year_links:
        raise RuntimeError("Aucune page Année trouvée")

    current_year = year_links[0][0]
    min_year = current_year - max(1, years_back) + 1
    selected = [(y, u) for (y, u) in year_links if y >= min_year]
    log(f"[HARVEST] années sélectionnées={len(selected)} (>= {min_year})")

    pairs: List[Dict[str, Any]] = []
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
                s = norm_text(absu + " " + label)

                if any(w in s for w in meta_exclude):
                    continue

                is_corrige = any(p in s for p in corrige_patterns)
                pdf_links.append({"url": absu, "name": os.path.basename(absu), "is_corrige": is_corrige})

            sujets = [p for p in pdf_links if not p["is_corrige"]]
            corriges = [p for p in pdf_links if p["is_corrige"]]
            used_corr = set()

            for suj in sujets:
                if len(pairs) >= volume_max:
                    break

                best = (None, 0.0)
                s_name = norm_text(suj["name"])
                s_geo = next((z for z in geo_zones if z in s_name), "")

                for corr in corriges:
                    if corr["url"] in used_corr:
                        continue
                    c_name = norm_text(corr["name"])
                    c_geo = next((z for z in geo_zones if z in c_name), "")

                    score = 0.0
                    if s_geo and c_geo:
                        if s_geo == c_geo:
                            score += 0.6
                        else:
                            continue

                    s_tok = set(re.findall(r"[a-z0-9]{2,}", s_name))
                    c_tok = set(re.findall(r"[a-z0-9]{2,}", c_name))
                    if s_tok and c_tok:
                        score += 0.4 * (len(s_tok & c_tok) / max(1, len(s_tok | c_tok)))

                    if score > best[1]:
                        best = (corr, score)

                corrige = best[0] if best[1] >= 0.25 else None
                if corrige:
                    used_corr.add(corrige["url"])

                item = {
                    "pair_id": f"PAIR_{level}_MATH_{y}_{stable_id(suj['name'])}",
                    "year": y,
                    "sujet": suj["name"],
                    "corrige?": bool(corrige),
                    "corrige_name": corrige["name"] if corrige else "",
                    "match_score": round(best[1], 2) if corrige else 0.0,
                    "sujet_url": suj["url"],
                    "corrige_url": corrige["url"] if corrige else "",
                }

                if not any(x["sujet_url"] == item["sujet_url"] for x in pairs):
                    pairs.append(item)
                    if item["corrige?"]:
                        corrige_ok += 1

            log(f"[HARVEST] year={y} pairs_cum={len(pairs)}")
        except Exception as e:
            log(f"[HARVEST] year={y} ERREUR: {e}")

    return {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": pack.get("pack_id"),
        "items_total": len(pairs),
        "items_corrige_ok": corrige_ok,
        "library": pairs,
    }

# =============================================================================
# Atomization: Qi/RQi robust multi-strategies
# =============================================================================
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√]")

def _inject_newlines(text: str, patterns: List[str]) -> str:
    t = text
    for pat in patterns:
        try:
            t = re.sub(pat, lambda m: "\n" + m.group(0), t)
        except Exception:
            continue
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def _tokenize_basic(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", norm_text(s))[:2500]

def _char_ngrams(s: str, n: int = 3, limit: int = 3000) -> List[str]:
    s = norm_text(s).replace("\n", " ")
    s = re.sub(r"\s+", " ", s)[:limit]
    if len(s) < n:
        return []
    return [s[i:i+n] for i in range(0, len(s)-n+1)]

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _compile_any(patterns: List[str]) -> List[re.Pattern]:
    out = []
    for p in patterns:
        try:
            out.append(re.compile(p))
        except Exception:
            pass
    return out

def _cut_positions(text: str, regs: List[re.Pattern]) -> List[int]:
    pos = set([0])
    for rg in regs:
        for m in rg.finditer(text):
            pos.add(m.start())
    for m in re.finditer(r"\n\n+", text):
        pos.add(m.start())
        pos.add(m.end())
    return sorted([p for p in pos if 0 <= p <= len(text)])

def _slice_segments(text: str, positions: List[int], min_len: int, max_len: int) -> List[str]:
    segs = []
    for a, b in zip(positions, positions[1:] + [len(text)]):
        s = re.sub(r"[ \t]+", " ", text[a:b]).strip()
        if len(s) < min_len:
            continue
        if len(s) > max_len:
            s = s[:max_len]
        segs.append(s)
    return segs

def _looks_like_question(seg: str, intent_verbs: List[str], keep_cfg: Dict[str, Any]) -> bool:
    if not seg:
        return False
    L = len(seg)
    if L < int(keep_cfg.get("min_len", 25)) or L > int(keep_cfg.get("max_len", 2200)):
        return False

    s = norm_text(seg)
    has_intent = any(v in s for v in intent_verbs) or ("?" in seg)
    digits = len(re.findall(r"\d", seg))
    has_math = bool(_MATH_SYMBOL_RE.search(seg)) or digits >= int(keep_cfg.get("min_digits_or_symbols", 1))
    return has_intent or has_math

def _looks_like_answer(seg: str, keep_cfg: Dict[str, Any]) -> bool:
    if not seg:
        return False
    L = len(seg)
    if L < int(keep_cfg.get("min_len", 30)) or L > int(keep_cfg.get("max_len", 2600)):
        return False
    digits = len(re.findall(r"\d", seg))
    has_math = bool(_MATH_SYMBOL_RE.search(seg)) or digits >= 1
    return has_math or L > 120

def split_into_q_segments(text: str, pack: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    atom = pack_get(pack, "atomization", {}) or {}
    markers = atom.get("markers", {}) or {}
    inject = atom.get("inject_newlines", []) or []
    min_len = int(atom.get("min_segment_len", 20))
    max_len = int(atom.get("max_segment_len", 2600))

    t = (text or "").replace("\r", "\n")
    t = _inject_newlines(t, inject)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return [], {"raw": 0, "kept": 0, "strategy": "empty"}

    regs = []
    regs += _compile_any(markers.get("exercise", []))
    regs += _compile_any(markers.get("part", []))
    regs += _compile_any(markers.get("question", []))
    regs += _compile_any(markers.get("qcm", []))

    positions = _cut_positions(t, regs)
    segs = _slice_segments(t, positions, min_len=min_len, max_len=max_len)

    intent_verbs = pack_get(pack, "text_processing.intent_verbs", []) or []
    keep_cfg = atom.get("question_keep", {}) or {}

    kept = []
    seen = set()
    for s in segs:
        key = stable_id(norm_text(s)[:400])
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_question(s, intent_verbs, keep_cfg):
            kept.append(s)

    return kept, {"raw": len(segs), "kept": len(kept), "strategy": "markers+paragraphs"}

def split_into_a_segments(text: str, pack: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    atom = pack_get(pack, "atomization", {}) or {}
    inject = atom.get("inject_newlines", []) or []
    min_len = int(atom.get("min_segment_len", 20))
    max_len = int(atom.get("max_segment_len", 2600))
    keep_cfg = atom.get("answer_keep", {}) or {}

    t = (text or "").replace("\r", "\n")
    t = _inject_newlines(t, inject)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return [], {"raw": 0, "kept": 0, "strategy": "empty"}

    regs = _compile_any([
        r"(?m)^\s*(\d{1,2}|[a-h])\s*[\)\.\-:]\s+",
        r"(?im)^\s*(exercice|exo\.?)\s*([0-9ivx]+)\b",
        r"(?im)^\s*(partie)\s*([a-z0-9]+)\b",
    ])
    positions = _cut_positions(t, regs)
    segs = _slice_segments(t, positions, min_len=min_len, max_len=max_len)

    kept = []
    seen = set()
    for s in segs:
        key = stable_id(norm_text(s)[:400])
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_answer(s, keep_cfg):
            kept.append(s)

    return kept, {"raw": len(segs), "kept": len(kept), "strategy": "answer_segments"}

def align_q_to_a(qs: List[str], ans: List[str], pack: Dict[str, Any]) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    atom = pack_get(pack, "atomization", {}) or {}
    al = atom.get("alignment", {}) or {}
    min_score = float(al.get("min_score", 0.18))
    min_ng = float(al.get("min_char_ngram", 0.10))

    if not qs or not ans:
        return [None] * len(qs), {"matched": 0, "min_score": min_score}

    q_tok = [_tokenize_basic(q) for q in qs]
    a_tok = [_tokenize_basic(a) for a in ans]

    q_ng = [_char_ngrams(q, 3) for q in qs]
    a_ng = [_char_ngrams(a, 3) for a in ans]

    used = set()
    link: List[Optional[int]] = [None] * len(qs)
    matched = 0

    for i in range(len(qs)):
        best_j, best_s = None, 0.0
        best_ng = 0.0
        for j in range(len(ans)):
            if j in used:
                continue
            s = _jaccard(q_tok[i], a_tok[j])
            if s < 0.02:
                continue
            ng = _jaccard(q_ng[i], a_ng[j])
            comb = 0.65 * s + 0.35 * ng
            if comb > best_s:
                best_s = comb
                best_j = j
                best_ng = ng

        if best_j is not None and best_s >= min_score and best_ng >= min_ng:
            link[i] = best_j
            used.add(best_j)
            matched += 1

    return link, {"matched": matched, "min_score": min_score, "min_ng": min_ng}

# =============================================================================
# Scope mapping (pack-driven)
# =============================================================================
def map_to_chapter(q_text: str, chapters: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not chapters:
        return "UNMAPPED", 0.0
    qn = norm_text(q_text)
    best_code = "UNMAPPED"
    best = 0.0
    for ch in chapters:
        kws = ch.get("keywords", []) or []
        score = 0
        for kw in kws:
            if kw and kw in qn:
                score += 1
        denom = max(6, len(kws))
        sc = score / denom
        if sc > best:
            best = sc
            best_code = ch.get("code", "UNMAPPED")
    return best_code, round(best, 4)

# =============================================================================
# ARI extraction (pack-driven, with evidence)
# =============================================================================
def extract_ari(q: str, r: str, pack: Dict[str, Any]) -> AriTrace:
    ari_cfg = pack_get(pack, "ari_config", {}) or {}
    patterns = ari_cfg.get("op_patterns", []) or []
    primary_order = ari_cfg.get("primary_ops_order", []) or []

    combined = norm_text(f"{q}\n{r}")
    ops: List[AriOpEvidence] = []
    seen = set()

    for item in patterns:
        op_code = item.get("op", "")
        pat = item.get("pattern", "")
        weight = float(item.get("weight", 1.0))
        if not op_code or not pat:
            continue
        try:
            matches = list(re.finditer(pat, combined, flags=re.IGNORECASE))
        except Exception:
            matches = []
        if matches and op_code not in seen:
            ev = []
            for m in matches[:3]:
                g = (m.group(0) or "").strip()
                if g:
                    ev.append(g[:40])
            conf = min(0.95, 0.55 + 0.08 * len(matches)) * weight
            ops.append(AriOpEvidence(op=op_code, confidence=round(conf, 4), evidence=ev, match_count=len(matches)))
            seen.add(op_code)

    if not ops:
        ops = [AriOpEvidence(op="OP_STANDARD", confidence=0.55, evidence=[], match_count=0)]
        seen = {"OP_STANDARD"}

    primary = "OP_STANDARD"
    for op in primary_order:
        if op in seen:
            primary = op
            break

    conf_global = max(o.confidence for o in ops) if ops else 0.55
    return AriTrace(
        primary_op=primary,
        ops=ops,
        all_ops=sorted(list(seen)),
        confidence_global=round(conf_global, 4),
    )

def ari_from_dict(d: Dict[str, Any]) -> AriTrace:
    ops = []
    for x in (d.get("ops") or []):
        if isinstance(x, AriOpEvidence):
            ops.append(x)
        else:
            ops.append(AriOpEvidence(
                op=str(x.get("op", "OP_STANDARD")),
                confidence=float(x.get("confidence", 0.55)),
                evidence=list(x.get("evidence", []) or []),
                match_count=int(x.get("match_count", 0)),
            ))
    return AriTrace(
        primary_op=str(d.get("primary_op", "OP_STANDARD")),
        ops=ops,
        all_ops=list(d.get("all_ops", []) or []),
        confidence_global=float(d.get("confidence_global", 0.55)),
    )

# =============================================================================
# Triggers (pack-driven)
# =============================================================================
def build_triggers(q: str, ari: AriTrace, chapter_code: str, pack: Dict[str, Any]) -> List[str]:
    tp = pack_get(pack, "text_processing", {}) or {}
    kws = tp.get("trigger_keywords", []) or []
    verbs = tp.get("intent_verbs", []) or []

    out = []
    for op in ari.ops:
        out.append(f"ARI:{op.op}")

    qn = norm_text(q)
    for kw in kws:
        if kw and kw in qn:
            out.append(f"KW:{kw.upper()}")

    for v in verbs:
        if v and v in qn:
            out.append(f"INTENT:{v.upper()}")
            break

    if chapter_code and chapter_code != "UNMAPPED":
        out.append(f"SCOPE:{chapter_code}")

    out2 = list(dict.fromkeys(out))
    return out2[:18]

# =============================================================================
# POSABLE gate (3 criteria pack-driven)
# =============================================================================
def posable_gate(q: str, r: str, chapter_code: str, ari: AriTrace, pack: Dict[str, Any]) -> PosableDecision:
    rules = pack_get(pack, "posable_rules", {}) or {}
    require_rqi = bool(rules.get("require_rqi", True))
    require_scope = bool(rules.get("require_scope", True))
    require_eval = bool(rules.get("require_evaluable", True))
    min_conf = float(rules.get("min_ari_confidence", 0.55))
    min_ans_len = int(rules.get("min_answer_len", 20))

    rc: List[str] = []
    has_rqi = bool(r and r.strip())
    if require_rqi and not has_rqi:
        rc.append(RC_RQI_MISSING)
    if require_scope and (not chapter_code or chapter_code == "UNMAPPED"):
        rc.append(RC_SCOPE_UNRESOLVED)
    if require_eval:
        if ari.confidence_global < min_conf:
            rc.append(RC_LOW_CONFIDENCE)
        if has_rqi and len(norm_text(r)) < min_ans_len:
            rc.append(RC_RQI_MISSING)

    ok = len(rc) == 0
    return PosableDecision(is_posable=ok, reason_codes=rc, confidence=float(ari.confidence_global))

# =============================================================================
# FRT generation (pack-driven) — required sections enforced
# =============================================================================
def generate_frt(primary_op: str, triggers: List[str], ari: AriTrace, pack: Dict[str, Any]) -> FRT:
    ari_cfg = pack_get(pack, "ari_config", {}) or {}
    op_labels = ari_cfg.get("op_labels", {}) or {}
    templates = ari_cfg.get("frt_templates", {}) or {}
    required = ari_cfg.get("frt_schema_required_sections", []) or []

    label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())
    title = f"Comment {label} ?"

    tpl = templates.get(primary_op, templates.get("OP_STANDARD", {})) or {}
    procedure = tpl.get("procedure", templates.get("OP_STANDARD", {}).get("procedure", []))
    checks = tpl.get("checks", templates.get("OP_STANDARD", {}).get("checks", []))
    pitfalls = tpl.get("pitfalls", templates.get("OP_STANDARD", {}).get("pitfalls", []))

    sections = {
        "context": {"primary_op": primary_op, "label": label},
        "triggers": triggers,
        "ari": asdict(ari),
        "procedure": procedure,
        "checks": checks,
        "pitfalls": pitfalls,
        "output": {"expected": "Réponse structurée conforme au modèle FRT", "format": "texte + étapes"},
    }

    for k in required:
        sections.setdefault(k, {})

    frt_id = f"FRT_{stable_id(primary_op, sha256_text(safe_json_dumps(sections))[:12])}"
    return FRT(frt_id=frt_id, primary_op=primary_op, title=title, sections=sections, generated_at=_utc_ts())

# =============================================================================
# QC generation + Kernel F1/F2
# =============================================================================
def build_qc_text(primary_op: str, pack: Dict[str, Any]) -> Tuple[str, str]:
    ari_cfg = pack_get(pack, "ari_config", {}) or {}
    op_labels = ari_cfg.get("op_labels", {}) or {}
    qc_fmt = pack_get(pack, "qc_format", {}) or {}
    prefix = qc_fmt.get("prefix", "Comment")
    suffix = qc_fmt.get("suffix", "?")
    template = qc_fmt.get("template", "{prefix} {label} {suffix}")

    label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())
    qc = template.format(prefix=prefix, label=label, suffix=suffix).strip()
    return qc, label

def trigger_weight_sum(triggers: List[str], pack: Dict[str, Any]) -> float:
    wcfg = pack_get(pack, "trigger_weights", {}) or {}
    default_w = float(wcfg.get("default", 0.6))
    by_prefix = wcfg.get("by_prefix", {}) or {}

    uniq = list(dict.fromkeys(triggers or []))
    s = 0.0
    for t in uniq:
        pref = t.split(":", 1)[0] if ":" in t else ""
        s += float(by_prefix.get(pref, default_w))
    return float(round(s, 6))

def f1_kernel(delta_c: float, epsilon: float, sum_Tj: float) -> float:
    return float(delta_c) * float((epsilon + sum_Tj) ** 2)

def sigma_similarity(qc_a: QCItem, qc_b: QCItem) -> float:
    a = set((qc_a.all_ops or []) + (qc_a.triggers or []))
    b = set((qc_b.all_ops or []) + (qc_b.triggers or []))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def parse_exam_date_from_name(name: str, pack: Dict[str, Any]) -> Optional[str]:
    src = pack_source(pack) or {}
    regs = src.get("date_regexes", []) or []
    s = name or ""
    for rg in regs:
        try:
            m = re.search(rg, s)
        except Exception:
            m = None
        if not m:
            continue
        g = m.groups()
        if len(g) == 3 and len(g[2]) == 4 and int(g[2]) >= 2000 and int(g[0]) <= 31:
            dd, mm, yy = int(g[0]), int(g[1]), int(g[2])
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return f"{yy:04d}-{mm:02d}-{dd:02d}"
        if len(g) == 3 and len(g[0]) == 4 and int(g[0]) >= 2000:
            yy, mm, dd = int(g[0]), int(g[1]), int(g[2])
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return f"{yy:04d}-{mm:02d}-{dd:02d}"
    return None

def t_rec_from_exam_date(exam_date: Optional[str], run_ts: str, t_rec_min: float) -> float:
    try:
        run_dt = datetime.fromisoformat(run_ts.replace("Z", "+00:00"))
    except Exception:
        run_dt = datetime.now(timezone.utc)
    if not exam_date:
        return float(t_rec_min)
    try:
        ex_dt = datetime.fromisoformat(exam_date + "T00:00:00+00:00")
        days = max(0, (run_dt - ex_dt).days)
        months = max(t_rec_min, days / 30.0)
        return float(round(months, 6))
    except Exception:
        return float(t_rec_min)

def f2_kernel(n_hist: int, N: int, alpha: float, t_rec: float, Psi_q: float, selected: List[QCItem], qc: QCItem) -> float:
    base = (float(n_hist) / max(1.0, float(N))) * (1.0 + float(alpha) / max(1e-9, float(t_rec))) * float(Psi_q)
    prod = 1.0
    for prev in selected:
        sig = sigma_similarity(qc, prev)
        prod *= (1.0 - sig)
    return float(round(base * prod, 8))

def progressive_select_per_chapter(qcs: List[QCItem], N_total: int, pack: Dict[str, Any], top_k: int) -> List[QCItem]:
    params = pack_get(pack, "f1_f2_params", {}) or {}
    alpha = float(params.get("alpha", 1.0))
    t_rec_min = float(params.get("t_rec_min", 1.0))

    remaining = [q for q in qcs if q.qc_state in ("POSABLE", "POSABLE_WEAK")]
    selected: List[QCItem] = []

    for _ in range(min(top_k, len(remaining))):
        best = None
        best_score = -1.0
        for qc in remaining:
            score = f2_kernel(qc.n_hist, N_total, alpha, qc.t_rec, qc.Psi_q, selected, qc)
            if score > best_score:
                best_score = score
                best = qc
        if best is None:
            break
        best.f2_score = best_score
        selected.append(best)
        remaining = [x for x in remaining if x.qc_id != best.qc_id]

    return selected

# =============================================================================
# Sanity, coverage, anti-faux-SEALED extraction KPI
# =============================================================================
def sanity_dup_ratio(qi_items: List[QiItem]) -> float:
    seen = set()
    dups = 0
    for q in qi_items:
        k = stable_id(norm_text(q.qi)[:400])
        if k in seen:
            dups += 1
        else:
            seen.add(k)
    return dups / max(1, len(qi_items))

def coverage_posable(qi_items: List[QiItem]) -> float:
    posable = [q for q in qi_items if q.has_rqi and q.posable.get("is_posable")]
    if not posable:
        return 0.0
    covered = sum(1 for q in posable if q.qc_id)
    return covered / max(1, len(posable))

def extraction_kpi(qi_items: List[QiItem], processed_pairs: int) -> Dict[str, float]:
    qi_total = len(qi_items)
    avg = qi_total / max(1, processed_pairs)
    return {"qi_total": float(qi_total), "avg_qi_per_pair": float(round(avg, 4))}

# =============================================================================
# IA2 Judge (bool checks) — invariant checks, pack-driven formatting
# =============================================================================
def ia2_checks(pack: Dict[str, Any], qc: QCItem, frt: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    issues = []
    qc_fmt = pack_get(pack, "qc_format", {}) or {}
    prefix = qc_fmt.get("prefix", "")
    suffix = qc_fmt.get("suffix", "")
    if prefix and not qc.qc.lower().startswith(prefix.lower()):
        issues.append({"check": "QC_FORM_PREFIX", "ok": False})
    if suffix and not qc.qc.strip().endswith(str(suffix)):
        issues.append({"check": "QC_FORM_SUFFIX", "ok": False})
    if qc.cluster_size < 2 and qc.posable_in_cluster == 0:
        issues.append({"check": "ANTI_SINGLETON", "ok": False})
    required = pack_get(pack, "ari_config.frt_schema_required_sections", []) or []
    for k in required:
        if k not in (frt.get("sections") or {}):
            issues.append({"check": f"FRT_SECTION_{k}", "ok": False})
    ok = len(issues) == 0
    return ok, issues

# =============================================================================
# Pipeline CAS1 ONLY
# =============================================================================
def run_pipeline(pack: Dict[str, Any], library: List[Dict[str, Any]], volume: int, threads: int) -> Dict[str, Any]:
    subject = st.session_state.get("subject", "MATH")
    level = st.session_state.get("level", "TERMINALE")
    chapters = pack_chapters(pack, subject, level)

    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    to_process = exploitable[:min(volume, len(exploitable))]
    processed_pairs = len(to_process)

    quarantine: List[QuarantineEntry] = []
    qi_items: List[QiItem] = []

    if not st.session_state.get("_run_ts"):
        st.session_state["_run_ts"] = _utc_ts()
    run_ts = st.session_state["_run_ts"]

    params = pack_get(pack, "f1_f2_params", {}) or {}
    epsilon = float(params.get("epsilon", 1e-6))
    t_rec_min = float(params.get("t_rec_min", 1.0))

    def process_pair(pair: Dict[str, Any]) -> Tuple[List[QiItem], Optional[QuarantineEntry], Dict[str, Any]]:
        pid = pair["pair_id"]
        info = {"pair_id": pid, "meta": {}}

        if not pair.get("corrige_url"):
            q = QuarantineEntry(
                pair_id=pid,
                sujet_url=pair.get("sujet_url", ""),
                corrige_url="",
                reason_codes=[RC_CORRIGE_MISSING],
                details={"why": "corrige_url absent"}
            )
            return [], q, info

        try:
            su_pdf = fetch_pdf_bytes(pair["sujet_url"])
        except Exception as e:
            q = QuarantineEntry(
                pair_id=pid,
                sujet_url=pair.get("sujet_url", ""),
                corrige_url=pair.get("corrige_url", ""),
                reason_codes=[RC_SUJET_UNREADABLE],
                details={"error": str(e)}
            )
            return [], q, info

        try:
            co_pdf = fetch_pdf_bytes(pair["corrige_url"])
        except Exception as e:
            q = QuarantineEntry(
                pair_id=pid,
                sujet_url=pair.get("sujet_url", ""),
                corrige_url=pair.get("corrige_url", ""),
                reason_codes=[RC_CORRIGE_UNREADABLE],
                details={"error": str(e)}
            )
            return [], q, info

        su_text, su_meta = extract_text_from_pdf(su_pdf, pack)
        co_text, co_meta = extract_text_from_pdf(co_pdf, pack)

        info["meta"]["sujet_extract"] = asdict(su_meta)
        info["meta"]["corrige_extract"] = asdict(co_meta)

        if len(su_text) < 200:
            q = QuarantineEntry(
                pair_id=pid,
                sujet_url=pair.get("sujet_url", ""),
                corrige_url=pair.get("corrige_url", ""),
                reason_codes=[RC_SUJET_UNREADABLE],
                details={"chars": len(su_text), "method": su_meta.method}
            )
            return [], q, info

        if len(co_text) < 200:
            q = QuarantineEntry(
                pair_id=pid,
                sujet_url=pair.get("sujet_url", ""),
                corrige_url=pair.get("corrige_url", ""),
                reason_codes=[RC_CORRIGE_UNREADABLE],
                details={"chars": len(co_text), "method": co_meta.method}
            )
            return [], q, info

        qs, q_audit = split_into_q_segments(su_text, pack)
        ans, a_audit = split_into_a_segments(co_text, pack)

        info["meta"]["atom_q"] = q_audit
        info["meta"]["atom_a"] = a_audit

        if len(qs) < 6:
            info["meta"]["atom_low"] = True

        link, al_audit = align_q_to_a(qs, ans, pack)
        info["meta"]["align"] = al_audit

        exam_date = parse_exam_date_from_name(pair.get("sujet", ""), pack) or parse_exam_date_from_name(pair.get("corrige_name", ""), pack)

        local_qi: List[QiItem] = []
        for i, qtxt in enumerate(qs):
            j = link[i] if i < len(link) else None
            rtxt = ans[j] if (j is not None and j < len(ans)) else ""
            ari = extract_ari(qtxt, rtxt, pack)
            chapter_code, ch_sc = map_to_chapter(qtxt, chapters)
            triggers = build_triggers(qtxt, ari, chapter_code, pack)
            pd = posable_gate(qtxt, rtxt, chapter_code, ari, pack)

            rc_extra = list(pd.reason_codes)
            if info["meta"].get("atom_low"):
                rc_extra = list(dict.fromkeys(rc_extra + [RC_ATOMIZATION_LOW]))
            if j is None and rtxt.strip() == "":
                rc_extra = list(dict.fromkeys(rc_extra + [RC_ALIGN_LOW]))

            pd2 = PosableDecision(is_posable=(len(rc_extra) == 0), reason_codes=rc_extra, confidence=pd.confidence)

            qi_id = f"QI_{stable_id(pid, str(i), norm_text(qtxt)[:160])}"
            local_qi.append(QiItem(
                qi_id=qi_id,
                pair_id=pid,
                source_year=int(pair.get("year", 0) or 0),
                source_exam_date=exam_date,
                qi=qtxt,
                rqi=rtxt,
                has_rqi=bool(rtxt.strip()),
                chapter_code=chapter_code,
                chapter_match_score=float(ch_sc),
                ari=asdict(ari),
                triggers=triggers,
                posable=asdict(pd2),
                qc_id=None,
                is_orphan=False
            ))

        return local_qi, None, info

    log(f"[PIPELINE] CAS1 pairs={processed_pairs} threads={threads}")
    metas = []
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        futures = [ex.submit(process_pair, p) for p in to_process]
        for fut in as_completed(futures):
            local_qi, qent, meta = fut.result()
            metas.append(meta)
            if qent:
                quarantine.append(qent)
            qi_items.extend(local_qi)

    qi_items.sort(key=lambda x: (x.pair_id, x.qi_id))
    quarantine.sort(key=lambda x: x.pair_id)

    buckets: Dict[Tuple[str, str], List[QiItem]] = defaultdict(list)
    for qi in qi_items:
        primary_op = (qi.ari or {}).get("primary_op", "OP_STANDARD")
        key = (qi.chapter_code or "UNMAPPED", primary_op)
        buckets[key].append(qi)

    chapter_bucket_counts = Counter([cc for (cc, _) in buckets.keys()])

    qc_pack: List[QCItem] = []
    mapping: Dict[str, str] = {}
    frt_pack: Dict[str, Any] = {}
    evidence_per_qc: List[Dict[str, Any]] = []

    qi_posable_items = [q for q in qi_items if q.has_rqi and q.posable.get("is_posable")]
    N_total = len(qi_posable_items)

    for (cc, primary_op), items in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        cluster_size = len(items)
        posable_items = [x for x in items if x.has_rqi and x.posable.get("is_posable")]
        posable_count = len(posable_items)

        is_singleton = cluster_size < 2
        is_only_in_chapter = chapter_bucket_counts.get(cc, 0) <= 1

        if is_singleton and (not is_only_in_chapter) and posable_count == 0:
            continue

        qc_text, qc_label = build_qc_text(primary_op, pack)
        qc_id = f"QC_{stable_id(cc, primary_op, qc_text)}"

        if posable_count >= 2:
            qc_state = "POSABLE"
        elif posable_count == 1:
            qc_state = "POSABLE_WEAK"
        else:
            qc_state = "UNPOSABLE"

        trig_counts = Counter()
        all_ops = set()
        for it in items:
            trig_counts.update(it.triggers or [])
            all_ops.update((it.ari or {}).get("all_ops", []))

        qc_triggers = [k for k, _ in trig_counts.most_common(12)]
        qi_ids = [x.qi_id for x in items]

        rep = max(items, key=lambda z: float((z.ari or {}).get("confidence_global", 0.0)))
        frt = generate_frt(primary_op, qc_triggers, ari_from_dict(rep.ari), pack)
        frt_pack[frt.frt_id] = asdict(frt)

        ch_info = next((c for c in chapters if c.get("code") == cc), {})
        delta_c = float(ch_info.get("delta_c", 1.0))

        sum_Tj = trigger_weight_sum(qc_triggers, pack)
        f1_raw = f1_kernel(delta_c, float(pack_get(pack, "f1_f2_params.epsilon", 1e-6)), sum_Tj)

        dates = [x.source_exam_date for x in items if x.source_exam_date]
        exam_date = max(dates) if dates else None
        t_rec = t_rec_from_exam_date(exam_date, run_ts, float(pack_get(pack, "f1_f2_params.t_rec_min", 1.0)))

        qc_item = QCItem(
            qc_id=qc_id,
            qc=qc_text,
            qc_label=qc_label,
            chapter_code=cc,
            primary_op=primary_op,
            all_ops=sorted(list(all_ops)) if all_ops else ["OP_STANDARD"],
            cluster_size=cluster_size,
            posable_in_cluster=posable_count,
            qc_state=qc_state,
            qi_ids=qi_ids,
            triggers=qc_triggers,
            frt=asdict(frt),
            delta_c=delta_c,
            sum_Tj=sum_Tj,
            f1_raw=float(round(f1_raw, 8)),
            Psi_q=0.0,
            n_hist=cluster_size,
            t_rec=float(t_rec),
            f2_score=0.0
        )
        qc_pack.append(qc_item)

        for it in items:
            mapping[it.qi_id] = qc_id

    by_ch = defaultdict(list)
    for qc in qc_pack:
        by_ch[qc.chapter_code].append(qc)
    for cc, qcs in by_ch.items():
        mx = max((q.f1_raw for q in qcs), default=0.0)
        for q in qcs:
            q.Psi_q = float(round((q.f1_raw / mx) if mx > 0 else 0.0, 8))

    orphans = 0
    for qi in qi_items:
        if qi.qi_id in mapping:
            qi.qc_id = mapping[qi.qi_id]
            qi.is_orphan = False
        else:
            qi.qc_id = None
            qi.is_orphan = True
            if qi.has_rqi and qi.posable.get("is_posable"):
                orphans += 1

    ia2_ok = True
    ia2_issues = []
    for qc in qc_pack:
        ok, issues = ia2_checks(pack, qc, qc.frt)
        if not ok:
            ia2_ok = False
            ia2_issues.extend([{"qc_id": qc.qc_id, **x} for x in issues])
        evidence_per_qc.append({
            "qc_id": qc.qc_id,
            "chapter_code": qc.chapter_code,
            "primary_op": qc.primary_op,
            "inputs": {
                "delta_c": qc.delta_c,
                "epsilon": float(pack_get(pack, "f1_f2_params.epsilon", 1e-6)),
                "sum_Tj": qc.sum_Tj,
                "f1_raw": qc.f1_raw,
                "Psi_q": qc.Psi_q,
                "n_hist": qc.n_hist,
                "t_rec": qc.t_rec,
                "N_total": N_total,
            }
        })

    cov = float(round(coverage_posable(qi_items), 6))
    dup = float(round(sanity_dup_ratio(qi_items), 6))
    sanity_max = float(pack_get(pack, "f1_f2_params.seal_sanity_dup_ratio_max", 0.35))
    sanity_ok = dup <= sanity_max

    kpi = extraction_kpi(qi_items, processed_pairs)
    min_avg = float(pack_get(pack, "f1_f2_params.seal_min_avg_qi_per_pair", 8.0))
    min_total = float(pack_get(pack, "f1_f2_params.seal_min_total_qi", 160))
    extraction_ok = (kpi["avg_qi_per_pair"] >= min_avg) and (kpi["qi_total"] >= min_total)

    evidence_pack = {
        "version": APP_VERSION,
        "timestamp": run_ts,
        "pack_id": pack.get("pack_id"),
        "pack_sig_sha256": pack.get("_pack_sig_sha256"),
        "pipeline": "CAS1_ONLY",
        "pairs_processed": processed_pairs,
        "quarantine_count": len(quarantine),
        "qi_total": len(qi_items),
        "qi_posable": len([q for q in qi_items if q.has_rqi and q.posable.get("is_posable")]),
        "orphans_posable": orphans,
        "qc_total": len(qc_pack),
        "qc_posable": len([q for q in qc_pack if q.qc_state in ("POSABLE", "POSABLE_WEAK")]),
        "coverage": cov,
        "sanity_dup_ratio": dup,
        "ia2_ok": ia2_ok,
        "ia2_issues": ia2_issues[:60],
        "extraction_kpi": kpi,
        "meta_pairs": metas[:80],
        "qc_evidence": evidence_per_qc[:200],
        "quarantine": [asdict(x) for x in quarantine[:200]],
    }

    out = {
        "qi_pack": [asdict(x) for x in qi_items],
        "qc_pack": [asdict(x) for x in qc_pack],
        "frt_pack": frt_pack,
        "mapping_qi_to_qc": mapping,
        "quarantine": [asdict(x) for x in quarantine],
        "evidence_pack": evidence_pack,
        "audit": {
            "pairs_processed": processed_pairs,
            "qi_total": len(qi_items),
            "rqi_total": len([q for q in qi_items if q.has_rqi]),
            "qi_posable": len([q for q in qi_items if q.has_rqi and q.posable.get("is_posable")]),
            "orphans_posable": orphans,
            "qc_total": len(qc_pack),
            "qc_posable": len([q for q in qc_pack if q.qc_state in ("POSABLE", "POSABLE_WEAK")]),
            "coverage": cov,
            "sanity_ok": sanity_ok,
            "sanity_dup_ratio": dup,
            "ia2_ok": ia2_ok,
            "extraction_ok": extraction_ok,
            "extraction_kpi": kpi,
        }
    }
    return out

# =============================================================================
# Saturation progressive (new_QC == 0)
# =============================================================================
def seal_reason(audit: Dict[str, Any], pack: Dict[str, Any], new_qc_count: int) -> str:
    cov_th = float(pack_get(pack, "f1_f2_params.seal_coverage_threshold", 0.80))

    reasons = []
    if new_qc_count != 0:
        reasons.append(f"SATURATION_NOT_REACHED_newQC={new_qc_count}")
    if not audit.get("sanity_ok", False):
        reasons.append("SANITY_FAILED")
    if audit.get("coverage", 0.0) < cov_th:
        reasons.append(f"COVERAGE_LOW_{audit.get('coverage', 0.0):.2%}")
    if audit.get("orphans_posable", 0) != 0:
        reasons.append(f"ORPHANS_POSABLE_{audit.get('orphans_posable', 0)}")
    if audit.get("qc_posable", 0) <= 0 or audit.get("qi_posable", 0) <= 0:
        reasons.append("NO_POSABLE_CONTENT")
    if not audit.get("ia2_ok", False):
        reasons.append("IA2_FAILED")
    if not audit.get("extraction_ok", False):
        reasons.append("EXTRACTION_KPI_LOW")
    return "ALL_CHECKS_PASSED" if not reasons else " | ".join(reasons)

def run_saturation(pack: Dict[str, Any], library: List[Dict[str, Any]], vol_start: int, vol_max: int, step: int, threads: int, top_k: int) -> Dict[str, Any]:
    iterations = []
    prev_set = None
    final = None

    for vol in range(vol_start, vol_max + 1, max(1, step)):
        log(f"=== RUN {APP_VERSION} vol={vol} ===")
        out = run_pipeline(pack, library, vol, threads)
        qc_ids = [q["qc_id"] for q in out["qc_pack"]]
        cur_set = set(qc_ids)

        if prev_set is None:
            new_qc_count = len(cur_set)
        else:
            new_qc_count = len(cur_set - prev_set)

        audit = out["audit"]
        iterations.append({
            "volume": vol,
            "qc_total": audit["qc_total"],
            "qc_posable": audit["qc_posable"],
            "qi_total": audit["qi_total"],
            "rqi_total": audit["rqi_total"],
            "qi_posable": audit["qi_posable"],
            "orphans": audit["orphans_posable"],
            "coverage": audit["coverage"],
            "sanity_ok": audit["sanity_ok"],
            "ia2_ok": audit["ia2_ok"],
            "extraction_ok": audit["extraction_ok"],
            "avg_qi_per_pair": audit["extraction_kpi"]["avg_qi_per_pair"],
            "new_qc_count": new_qc_count,
        })

        prev_set = cur_set
        final = out

        if new_qc_count == 0:
            break

    assert final is not None

    qcs = [QCItem(**qc) for qc in final["qc_pack"]]
    subject = st.session_state.get("subject", "MATH")
    level = st.session_state.get("level", "TERMINALE")
    chapters = pack_chapters(pack, subject, level)

    N_total = final["audit"]["qi_posable"]
    selection = []
    by_ch = defaultdict(list)
    for qc in qcs:
        by_ch[qc.chapter_code].append(qc)

    for cc, qlist in sorted(by_ch.items()):
        ch_info = next((c for c in chapters if c.get("code") == cc), {"label": cc})
        selected = progressive_select_per_chapter(qlist, N_total, pack, top_k=top_k)
        selection.append({
            "chapter_code": cc,
            "chapter_label": ch_info.get("label", cc),
            "qc_count": len(qlist),
            "selected_count": len(selected),
            "selected_qc": [{
                "qc_id": x.qc_id,
                "qc": x.qc,
                "primary_op": x.primary_op,
                "Psi_q": x.Psi_q,
                "t_rec": x.t_rec,
                "f2_score": x.f2_score,
                "frt_id": (x.frt or {}).get("frt_id", ""),
                "posable_in_cluster": x.posable_in_cluster,
                "cluster_size": x.cluster_size,
            } for x in selected]
        })

    last_iter = iterations[-1]
    new_qc_count_last = int(last_iter["new_qc_count"])
    audit = final["audit"]
    sealed = (seal_reason(audit, pack, new_qc_count_last) == "ALL_CHECKS_PASSED")
    reason = seal_reason(audit, pack, new_qc_count_last)

    signature_payload = {
        "version": APP_VERSION,
        "timestamp": st.session_state.get("_run_ts"),
        "pack_sig_sha256": pack.get("_pack_sig_sha256"),
        "harvest_sig": sha256_text(safe_json_dumps(st.session_state.get("harvest_manifest") or {})),
        "qi_pack_sha": sha256_text(safe_json_dumps(final["qi_pack"])),
        "qc_pack_sha": sha256_text(safe_json_dumps(final["qc_pack"])),
        "frt_pack_sha": sha256_text(safe_json_dumps(final["frt_pack"])),
        "mapping_sha": sha256_text(safe_json_dumps(final["mapping_qi_to_qc"])),
        "selection_sha": sha256_text(safe_json_dumps({"chapters": selection})),
        "evidence_sha": sha256_text(safe_json_dumps(final["evidence_pack"])),
        "quarantine_sha": sha256_text(safe_json_dumps(final["quarantine"])),
        "audit_sha": sha256_text(safe_json_dumps(audit)),
    }
    signature_payload["run_signature_sha256"] = sha256_text(safe_json_dumps(signature_payload))

    return {
        "final": final,
        "iterations": iterations,
        "selection_report": {"chapters": selection},
        "sealed": sealed,
        "seal_reason": reason,
        "run_signature": signature_payload
    }

# =============================================================================
# UI Streamlit (4 onglets + RQi)
# =============================================================================
def metric_row(stats: Dict[str, Any], sealed: bool, coverage: float):
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Pairs", stats.get("pairs", 0))
    c2.metric("Corrigés", stats.get("corr_ok", 0))
    c3.metric("Qi", stats.get("qi", 0))
    c4.metric("RQi", stats.get("rqi", 0))
    c5.metric("Qi POSABLE", stats.get("qi_posable", 0))
    c6.metric("QC", stats.get("qc", 0))
    c7.metric("Coverage", f"{coverage:.0%}")
    c8.metric("SEALED", "✅ YES" if sealed else "❌ NO")

def main():
    # IMPORTANT Streamlit: set_page_config must be the first Streamlit call
    st.set_page_config(page_title=f"SMAXIA GTE {APP_VERSION}", layout="wide")

    ss_init()

    st.markdown(f"# SMAXIA GTE Console {APP_VERSION}")
    st.caption("ISO-PROD | Pack-Driven | CAS1 ONLY | Saturation progressive (new_QC=0) | Preuves (mapping+signature)")

    with st.sidebar:
        st.markdown("## 1. ACTIVATION PACK")

        codes = list(GENESIS_PACKS.keys())
        labels = [f"{c} - {GENESIS_PACKS[c]}" for c in codes]
        idx = st.selectbox("Sélectionner le pays", range(len(codes)), format_func=lambda i: labels[i])
        country = codes[idx]

        st.markdown("---")
        st.caption("Pack externe (optionnel)")
        uploaded = st.file_uploader("Pack JSON", type=["json"])

        if st.button("🚀 ACTIVER", use_container_width=True, type="primary"):
            try:
                if uploaded:
                    pack = json.loads(uploaded.read().decode("utf-8"))
                    pack["_source"] = "UPLOAD"
                    pack["_pack_sig_sha256"] = sha256_text(safe_json_dumps(pack))
                    log(f"[PACK] Upload pack_id={pack.get('pack_id')}")
                else:
                    pack = generate_pack(country)
                    log(f"[PACK] Genesis country={country}")

                st.session_state.pack_active = pack
                st.session_state.pack_id = pack.get("pack_id", "")
                st.session_state.pack_sig_sha256 = pack.get("_pack_sig_sha256", "")
                st.session_state.country = pack.get("country_code", "")

                taxonomy = pack.get("chapter_taxonomy", {}) or {}
                subjects = list(taxonomy.keys())
                if subjects:
                    st.session_state.subject = subjects[0]
                    levels = list((taxonomy.get(subjects[0], {}) or {}).keys())
                    if levels:
                        st.session_state.level = levels[0]

                st.session_state.library = []
                st.session_state.harvest_manifest = None
                st.session_state.qi_pack = None
                st.session_state.qc_pack = None
                st.session_state.frt_pack = None
                st.session_state.mapping_qi_to_qc = None
                st.session_state.selection_report = None
                st.session_state.evidence_pack = None
                st.session_state.quarantine = None
                st.session_state.sealed = False
                st.session_state.last_run_audit = None
                st.session_state.run_stats = {}
                st.session_state._run_ts = None

                st.success(f"✅ Pack activé: {st.session_state.pack_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur: {e}")

        if st.session_state.pack_active:
            pack = st.session_state.pack_active
            st.success(f"✅ {st.session_state.pack_id}")
            st.caption(f"Pays: {pack.get('country_name')}")
            st.caption(f"Source: {pack.get('_source')}")
            st.caption(f"PackSig: {st.session_state.pack_sig_sha256[:12]}...")

            st.markdown("---")
            st.markdown("## 2. SÉLECTION")

            taxonomy = pack.get("chapter_taxonomy", {}) or {}
            subjects = list(taxonomy.keys())
            if subjects:
                subject = st.selectbox("Matière", subjects, index=subjects.index(st.session_state.subject) if st.session_state.subject in subjects else 0)
                st.session_state.subject = subject
                levels = list((taxonomy.get(subject, {}) or {}).keys())
                if levels:
                    level = st.selectbox("Niveau", levels, index=levels.index(st.session_state.level) if st.session_state.level in levels else 0)
                    st.session_state.level = level

            chapters = pack_chapters(pack, st.session_state.subject, st.session_state.level)
            if chapters:
                st.markdown("### Chapitres (preview)")
                for ch in chapters[:8]:
                    st.caption(f"• {ch.get('label')}")

    if not st.session_state.pack_active:
        st.warning("Sélectionne un pays et clique sur ACTIVER.")
        return

    pack = st.session_state.pack_active
    tab1, tab2, tab3, tab4 = st.tabs(["Import", "RUN", "Résultats", "Exports"])

    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?"))
    stats = st.session_state.run_stats or {}
    coverage = float(stats.get("coverage", 0.0) or 0.0)

    with tab1:
        st.markdown("## Import & Harvest")
        metric_row(
            {"pairs": len(lib), "corr_ok": corr_ok, "qi": stats.get("qi", 0), "rqi": stats.get("rqi", 0),
             "qi_posable": stats.get("qi_posable", 0), "qc": stats.get("qc", 0)},
            st.session_state.sealed,
            coverage
        )

        if lib:
            st.dataframe(
                [{k: r.get(k, "") for k in ["pair_id", "year", "sujet", "corrige?", "match_score"]} for r in lib[:120]],
                use_container_width=True,
                hide_index=True
            )

        st.markdown("### HARVEST")
        src = pack_source(pack)
        if src:
            st.info(f"Source: {src.get('source_name')}")
            c1, c2 = st.columns(2)
            years_back = c1.number_input("Années (back)", 1, 15, 5)
            volume_max = c2.number_input("Volume max pairs", 5, 200, 60)

            if st.button("🔄 HARVEST", use_container_width=True):
                with st.spinner("Harvest en cours..."):
                    try:
                        manifest = harvest_from_pack(pack, st.session_state.level, int(years_back), int(volume_max))
                        st.session_state.harvest_manifest = manifest
                        st.session_state.library = manifest["library"]
                        st.session_state.sealed = False
                        st.session_state._run_ts = None

                        st.session_state.qi_pack = None
                        st.session_state.qc_pack = None
                        st.session_state.frt_pack = None
                        st.session_state.mapping_qi_to_qc = None
                        st.session_state.selection_report = None
                        st.session_state.evidence_pack = None
                        st.session_state.quarantine = None
                        st.session_state.last_run_audit = None
                        st.session_state.run_stats = {}

                        st.success(f"✅ {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")

    with tab2:
        st.markdown("## RUN — Saturation progressive (new_QC=0)")
        if not lib:
            st.warning("Bibliothèque vide : HARVEST d'abord.")
        elif corr_ok <= 0:
            st.error("Aucun corrigé disponible (CAS1 ONLY).")
        else:
            max_vol = max(1, min(160, corr_ok))
            c1, c2, c3, c4 = st.columns(4)
            vol_start = c1.number_input("vol_start", 1, max_vol, min(5, max_vol))
            vol_max = c2.number_input("vol_max", int(vol_start), max_vol, min(30, max_vol))
            step = c3.number_input("step", 1, 10, 1)
            threads = c4.number_input("threads", 1, 16, 6)

            top_k = st.number_input("F2 top_k / chapitre", 5, 40, 15)

            st.caption("SEALED strict = (new_QC==0) AND (orphans_posable==0) AND sanity_ok AND IA2_ok AND extraction_kpi_ok AND qc_posable>0 AND qi_posable>0 AND coverage>=threshold")

            if st.button("🚀 LANCER TEST SATURATION", use_container_width=True, type="primary"):
                with st.spinner("Pipeline en cours..."):
                    try:
                        st.session_state.logs = st.session_state.logs[-4000:]
                        res = run_saturation(pack, lib, int(vol_start), int(vol_max), int(step), int(threads), int(top_k))

                        final = res["final"]
                        audit = final["audit"]

                        st.session_state.qi_pack = final["qi_pack"]
                        st.session_state.qc_pack = final["qc_pack"]
                        st.session_state.frt_pack = final["frt_pack"]
                        st.session_state.mapping_qi_to_qc = final["mapping_qi_to_qc"]
                        st.session_state.selection_report = res["selection_report"]
                        st.session_state.evidence_pack = final["evidence_pack"]
                        st.session_state.quarantine = final["quarantine"]
                        st.session_state.last_run_audit = {
                            "sealed": res["sealed"],
                            "seal_reason": res["seal_reason"],
                            "iterations": res["iterations"],
                            "run_signature": res["run_signature"],
                            "audit": audit
                        }

                        st.session_state.sealed = bool(res["sealed"])
                        st.session_state.run_stats = {
                            "qi": audit["qi_total"],
                            "rqi": audit["rqi_total"],
                            "qi_posable": audit["qi_posable"],
                            "qc": audit["qc_total"],
                            "coverage": audit["coverage"]
                        }

                        if res["sealed"]:
                            st.success(f"✅ SEALED = YES | Coverage={audit['coverage']:.0%} | Qi={audit['qi_total']} | avgQi/pair={audit['extraction_kpi']['avg_qi_per_pair']}")
                        else:
                            st.warning(f"⚠️ SEALED = NO | Raison: {res['seal_reason']}")

                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
                        import traceback
                        st.text(traceback.format_exc())

        st.markdown("### Logs")
        st.text_area("", logs_text(), height=220)

    with tab3:
        st.markdown("## Résultats")
        run = st.session_state.last_run_audit
        if not run:
            st.info("Lance le pipeline pour voir les résultats.")
        else:
            audit = run["audit"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("SEALED", "✅ YES" if run["sealed"] else "❌ NO")
            c2.metric("Qi total", audit.get("qi_total", 0))
            c3.metric("RQi total", audit.get("rqi_total", 0))
            c4.metric("Coverage", f"{audit.get('coverage', 0):.0%}")
            c5.metric("Orphans (posable)", audit.get("orphans_posable", 0))

            st.markdown("### Verdict Kernel")
            st.write(f"- seal_reason: `{run['seal_reason']}`")
            st.write(f"- sanity_ok: `{audit.get('sanity_ok')}` | dup_ratio={audit.get('sanity_dup_ratio')}")
            st.write(f"- ia2_ok: `{audit.get('ia2_ok')}`")
            st.write(f"- extraction_ok: `{audit.get('extraction_ok')}` | avgQi/pair={audit.get('extraction_kpi', {}).get('avg_qi_per_pair')} | qi_total={audit.get('extraction_kpi', {}).get('qi_total')}")

            st.markdown("### Itérations de saturation")
            st.dataframe(run["iterations"], use_container_width=True, hide_index=True)

            st.markdown("### Sélection F2 par chapitre")
            sel = st.session_state.selection_report or {}
            for ch in sel.get("chapters", []):
                with st.expander(f"{ch['chapter_label']} ({ch['selected_count']}/{ch['qc_count']} QC)"):
                    for qc in ch.get("selected_qc", []):
                        st.markdown(f"**{qc['qc']}**")
                        st.caption(f"ID: {qc['qc_id']} | Ψ={qc['Psi_q']:.4f} | t_rec={qc['t_rec']:.2f} | F2={qc['f2_score']:.6f} | cluster={qc['cluster_size']} | posable={qc['posable_in_cluster']}")

            st.markdown("---")
            st.markdown("## Explorateur QC → Qi → RQi → ARI → FRT")
            if st.session_state.qc_pack and st.session_state.qi_pack:
                qcs = st.session_state.qc_pack
                qi = st.session_state.qi_pack
                qi_by_id = {x["qi_id"]: x for x in qi}

                ch_opts = ["ALL"] + sorted({q["chapter_code"] for q in qcs})
                sel_ch = st.selectbox("Filtrer par chapitre", ch_opts)

                qcs2 = qcs if sel_ch == "ALL" else [q for q in qcs if q["chapter_code"] == sel_ch]
                qcs2 = sorted(qcs2, key=lambda x: (-x.get("posable_in_cluster", 0), x.get("qc_id", "")))

                if qcs2:
                    labels = [f"{q['qc_id']} | {q['chapter_code']} | {q.get('qc_label','')} | n={q['cluster_size']} | posable={q['posable_in_cluster']}" for q in qcs2]
                    sel_idx = st.selectbox("Sélectionner une QC", range(len(labels)), format_func=lambda i: labels[i])
                    qc = qcs2[sel_idx]

                    st.markdown(f"### {qc['qc']}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Cluster", qc["cluster_size"])
                    c2.metric("Posable", qc["posable_in_cluster"])
                    c3.metric("État", qc["qc_state"])
                    c4.metric("Ψq", qc.get("Psi_q", 0.0))

                    st.markdown("#### Triggers")
                    st.write(qc.get("triggers", []))

                    st.markdown("#### FRT")
                    st.json(qc.get("frt", {}))

                    st.markdown("#### Qi liées (avec RQi)")
                    for qid in qc.get("qi_ids", [])[:18]:
                        item = qi_by_id.get(qid)
                        if not item:
                            continue
                        tag = "✅" if item.get("has_rqi") else "❌"
                        pos = "POSABLE" if (item.get("posable") or {}).get("is_posable") else "UNPOSABLE"
                        with st.expander(f"{qid} | RQi={tag} | {pos} | {item.get('chapter_code','')} | {item.get('ari',{}).get('primary_op','')}"):
                            st.markdown("**Qi**")
                            st.text(item.get("qi","")[:1200])
                            st.markdown("**RQi**")
                            st.text(item.get("rqi","")[:1200])
                            st.markdown("**ARI**")
                            st.json(item.get("ari", {}))
                            st.markdown("**Triggers**")
                            st.write(item.get("triggers", []))
                            st.markdown("**Posable decision**")
                            st.json(item.get("posable", {}))

    with tab4:
        st.markdown("## Exports")
        if st.session_state.last_run_audit:
            audit_obj = st.session_state.last_run_audit
            sig = audit_obj.get("run_signature", {})
            st.caption(f"RunSignature: {sig.get('run_signature_sha256','')[:18]}...")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("logs.txt", logs_text(), "smaxia_logs.txt")
            if st.session_state.qi_pack:
                st.download_button("qi_pack.json", safe_json_dumps(st.session_state.qi_pack), "qi_pack.json")
            if st.session_state.qc_pack:
                st.download_button("qc_pack.json", safe_json_dumps(st.session_state.qc_pack), "qc_pack.json")
            if st.session_state.frt_pack:
                st.download_button("frt_pack.json", safe_json_dumps(st.session_state.frt_pack), "frt_pack.json")

        with c2:
            if st.session_state.mapping_qi_to_qc:
                st.download_button("mapping_qi_to_qc.json", safe_json_dumps(st.session_state.mapping_qi_to_qc), "mapping_qi_to_qc.json")
            if st.session_state.selection_report:
                st.download_button("selection_report.json", safe_json_dumps(st.session_state.selection_report), "selection_report.json")
            if st.session_state.evidence_pack:
                st.download_button("evidence_pack.json", safe_json_dumps(st.session_state.evidence_pack), "evidence_pack.json")
            if st.session_state.quarantine is not None:
                st.download_button("quarantine.json", safe_json_dumps(st.session_state.quarantine), "quarantine.json")
            if st.session_state.last_run_audit:
                st.download_button("audit.json", safe_json_dumps(st.session_state.last_run_audit), "audit.json")
            if st.session_state.last_run_audit and st.session_state.last_run_audit.get("run_signature"):
                st.download_button("run_signature.json", safe_json_dumps(st.session_state.last_run_audit["run_signature"]), "run_signature.json")

# =============================================================================
# ENTRYPOINT (Streamlit-safe): exécute toujours main()
# =============================================================================
main()
