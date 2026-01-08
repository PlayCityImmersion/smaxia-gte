# =============================================================================
# SMAXIA GTE Console V31.10.24 — ISO-PROD (FICHIER UNIQUE)
# =============================================================================
# OBJECTIF V31.10.24 (KERNEL V10.6.2) :
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

            # ✅ FIX: ligne valide (plus de chaîne cassée)
            "geographic_zones": ["metro", "metropole", "amerique", "asie", "polynesie", "etranger", "antilles", "liban"],

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
                import PIL.Image  # noqa
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
# (…)
# =============================================================================
# Le reste du script reste inchangé fonctionnellement ; la correction bloquante
# était la chaîne cassée dans generate_pack() → geographic_zones.
#
# IMPORTANT : Si vous voulez que je recolle ici la totalité du fichier jusqu’au
# dernier caractère (main inclus), dites “colle la suite” et je vous renvoie
# la suite en continu, sans découpage logique.
