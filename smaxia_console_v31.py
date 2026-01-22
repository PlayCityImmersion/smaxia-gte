# =============================================================================
# SMAXIA GTE Console V31.10.25 — ISO-PROD (FICHIER UNIQUE)
# =============================================================================
# KERNEL V10.6.3 — DOCTRINE STRICTE
# - Action humaine unique : choisir un pays (France | Côte d’Ivoire) puis ACTIVER
# - Ensuite : pipeline 100% automatique (ISO-PROD logique) sans autre action utilisateur
# - ZÉRO HARDCODE MÉTIER : aucune logique pays/matière/chapitre en dur dans le CORE
#   Toute variabilité provient du CAP (Country Academic Pack) chargé/validé.
# - Mode déconnecté : si aucun PDF local, génération de STUBS 100% Pack-Driven (depuis CAP)
# - Déterminisme : pas de timestamps non déterministes ; référence temporelle fixe via CAP
# - Sortie UI (finale) : UNIQUEMENT QC / FRT / ARI / TRIGGERS par chapitre + Qi associées
#
# Correctif critique V31.10.25:
# - Plus d'assertion "final is not None" : si 0 sources, fallback STUB Pack-Driven garanti
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import math
import hashlib
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import streamlit as st

# ---------------- Optional deps (local-only; no network) ----------------
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    import fitz  # PyMuPDF  # type: ignore
except Exception:
    fitz = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None


# =============================================================================
# Invariants (non-métier)
# =============================================================================
APP_NAME = "SMAXIA GTE Console V31.10.25"
APP_VERSION = "V31.10.25"

# =============================================================================
# RC_* QUARANTAINE (Kernel — codes invariants)
# =============================================================================
RC_CORRIGE_MISSING = "RC_CORRIGE_MISSING"
RC_CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
RC_SUJET_UNREADABLE = "RC_SUJET_UNREADABLE"
RC_SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
RC_RQI_MISSING = "RC_RQI_MISSING"
RC_LOW_CONFIDENCE = "RC_LOW_CONFIDENCE"
RC_ATOMIZATION_LOW = "RC_ATOMIZATION_LOW"
RC_ALIGN_LOW = "RC_ALIGN_LOW"
RC_CLUSTER_SINGLETON_FORBIDDEN = "RC_CLUSTER_SINGLETON_FORBIDDEN"
RC_SAFETY_STOP = "RC_SAFETY_STOP"
RC_NO_SOURCES = "RC_NO_SOURCES"


# =============================================================================
# Deterministic helpers
# =============================================================================
def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

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

def pack_get(pack: Dict[str, Any], path: str, default=None):
    cur: Any = pack
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def cap_sig(pack: Dict[str, Any]) -> str:
    # scellabilité : hash stable (tri JSON)
    return sha256_text(safe_json_dumps(pack))

def det_now(cap: Dict[str, Any]) -> str:
    # 100% déterministe : "timestamp" dérivé du CAP (pas d'horloge système)
    # format stable (string)
    return f"DET_{str(pack_get(cap,'_pack_sig_sha256',''))[:16]}"

def det_reference_date_iso(cap: Dict[str, Any]) -> str:
    # Référence temporelle FIXE pour calcul t_rec (anti-aléa)
    # CAP peut fournir f_params.reference_date_iso (ex: "2026-01-01")
    return str(pack_get(cap, "f_params.reference_date_iso", "2026-01-01"))


# =============================================================================
# Structures (proof-grade)
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
    source_id: str
    sujet_ref: str
    corrige_ref: str
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
    step_types: List[str]

@dataclass
class QiItem:
    qi_id: str
    pair_id: str
    source_id: str
    source_year: int
    source_exam_date: Optional[str]
    sujet_ref: str
    corrige_ref: str
    qi: str
    rqi: str
    has_rqi: bool
    chapter_code: str
    chapter_match_score: float
    ari: Dict[str, Any]
    triggers: List[str]
    posable: Dict[str, Any]
    sig_q: str
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
    chapter_label: str
    primary_op: str
    all_ops: List[str]
    cluster_size: int
    posable_in_cluster: int
    qi_ids: List[str]
    triggers: List[str]
    frt: Dict[str, Any]
    # F1/F2 terms
    delta_c: float
    epsilon: float
    T_steps: List[Dict[str, Any]]
    sum_Tj: float
    psi_brut: float
    psi_norm: float
    n_q: int
    N_total_chap: int
    alpha: float
    t_rec: float
    sigma_terms: List[Dict[str, Any]]
    score_f2: float
    evidence_min: Dict[str, Any]


# =============================================================================
# CAP (EMBEDDED) — Données uniquement (PAS de logique métier dans CORE)
# =============================================================================
def _embedded_caps() -> Dict[str, Dict[str, Any]]:
    # Chapter taxonomy (données CAP)
    base_math_tax = {
        "MATH": {
            "TERMINALE": [
                {"code": "CH_ANALYSE", "label": "Analyse", "keywords": ["limite", "dérivée", "derivee", "asymptote", "variation", "tangente", "convexite", "continuité", "continuite", "fonction"], "delta_c": 1.0},
                {"code": "CH_PROBAS", "label": "Probabilités", "keywords": ["probabilite", "proba", "loi", "binomiale", "normale", "espérance", "esperance", "variance", "aléatoire", "aleatoire"], "delta_c": 1.0},
                {"code": "CH_SUITES", "label": "Suites", "keywords": ["suite", "récurrence", "recurrence", "arithmétique", "arithmetique", "géométrique", "geometrique", "convergence", "rang"], "delta_c": 1.0},
                {"code": "CH_INTEG", "label": "Intégration", "keywords": ["intégrale", "integrale", "primitive", "aire", "valeur moyenne"], "delta_c": 1.0},
            ]
        }
    }

    cognitive_table = {
        "step_weights": {
            "STEP_LIMIT": 1.2,
            "STEP_DERIVE": 1.1,
            "STEP_INTEGRATE": 1.3,
            "STEP_PROBABILITY": 1.25,
            "STEP_SEQUENCE": 1.15,
            "STEP_SOLVE_EQUATION": 1.0,
            "STEP_PROVE": 0.9,
            "STEP_STANDARD": 0.8,
        },
        "op_to_steps": {
            "OP_LIMIT": ["STEP_LIMIT"],
            "OP_DERIVE": ["STEP_DERIVE"],
            "OP_INTEGRATE": ["STEP_INTEGRATE"],
            "OP_PROBABILITY": ["STEP_PROBABILITY"],
            "OP_INDUCTION": ["STEP_SEQUENCE", "STEP_PROVE"],
            "OP_SOLVE_EQUATION": ["STEP_SOLVE_EQUATION"],
            "OP_PROVE": ["STEP_PROVE"],
            "OP_STANDARD": ["STEP_STANDARD"],
        },
    }

    ari_config = {
        "op_patterns": [
            {"op": "OP_PROBABILITY", "pattern": r"\b(probabilit|proba|loi\s+binomiale|loi\s+normale|esperance|variance|aleatoire)\b", "weight": 1.0},
            {"op": "OP_DERIVE", "pattern": r"\b(deriv|f'\s*\(|tangente|taux\s+de\s+variation)\b", "weight": 1.0},
            {"op": "OP_INTEGRATE", "pattern": r"\b(integr|primitive|aire\s+sous|calcul\s+d'aire)\b", "weight": 1.0},
            {"op": "OP_LIMIT", "pattern": r"\b(limit|tend\s+vers|infini|convergence)\b", "weight": 1.0},
            {"op": "OP_INDUCTION", "pattern": r"\b(recurr|induction|heredite|initialisation|rang)\b", "weight": 1.0},
            {"op": "OP_SOLVE_EQUATION", "pattern": r"\b(equat|resou|racine|solution|discriminant|inequation)\b", "weight": 1.0},
            {"op": "OP_PROVE", "pattern": r"\b(demontr|prouv|justifi|montr|etablir)\b", "weight": 0.8},
        ],
        "primary_ops_order": ["OP_PROBABILITY", "OP_INTEGRATE", "OP_DERIVE", "OP_LIMIT", "OP_INDUCTION", "OP_SOLVE_EQUATION", "OP_PROVE", "OP_STANDARD"],
        "op_labels": {
            "OP_PROBABILITY": "calculer une probabilité",
            "OP_DERIVE": "dériver une fonction",
            "OP_INTEGRATE": "calculer une intégrale",
            "OP_LIMIT": "calculer une limite",
            "OP_INDUCTION": "démontrer par récurrence",
            "OP_SOLVE_EQUATION": "résoudre une équation",
            "OP_PROVE": "démontrer une propriété",
            "OP_STANDARD": "résoudre un exercice standard",
        },
        "frt_schema_required_sections": ["context", "triggers", "ari", "procedure", "checks", "pitfalls", "output"],
        "frt_templates": {
            "OP_PROBABILITY": {
                "procedure": [
                    "Identifier l’expérience aléatoire et l’univers Ω",
                    "Définir précisément les événements",
                    "Identifier la loi et les paramètres",
                    "Appliquer la formule / outil adapté",
                    "Calculer, vérifier, conclure",
                ],
                "checks": ["Valeur dans [0,1]", "Interprétation correcte"],
                "pitfalls": ["Mauvaise loi", "Arrondis prématurés"],
            },
            "OP_DERIVE": {
                "procedure": [
                    "Identifier f et le domaine",
                    "Appliquer les règles de dérivation",
                    "Simplifier f'(x)",
                    "Exploiter (variations / tangente) si demandé",
                ],
                "checks": ["Domaine", "Signe de f' si tableau"],
                "pitfalls": ["Erreur de règle", "Oubli domaine"],
            },
            "OP_STANDARD": {
                "procedure": [
                    "Lire et reformuler la demande",
                    "Lister données / inconnues",
                    "Choisir la méthode",
                    "Exécuter les calculs",
                    "Vérifier et conclure",
                ],
                "checks": ["Résultat plausible", "Conclusion explicite"],
                "pitfalls": ["Mauvaise méthode", "Erreurs algébriques"],
            },
        },
    }

    qc_format = {"prefix": "Comment", "suffix": "?", "template": "{prefix} {label} {suffix}"}

    text_processing = {
        "intent_verbs": ["montrer", "demontrer", "prouver", "justifier", "determiner", "calculer", "resoudre", "etudier", "donner", "exprimer", "simplifier", "trouver", "verifier", "deduire", "conclure", "etablir"],
        "trigger_keywords": ["limite", "derivee", "integrale", "probabilite", "suite", "equation", "fonction", "recurrence", "variance", "esperance"],
        "glued_patterns": [
            {"pattern": r"(\d)([a-zA-Z])", "replacement": r"\1 \2"},
            {"pattern": r"([a-zA-Z])(\d)", "replacement": r"\1 \2"},
            {"pattern": r"([a-z])([A-Z])", "replacement": r"\1 \2"},
        ],
    }

    atomization = {
        "max_pages": 120,
        "min_segment_len": 25,
        "max_segment_len": 2400,
        "markers": {
            "exercise": [r"(?im)^\s*(exercice|exo\.?)\s*([0-9ivx]+)\b"],
            "question": [r"(?m)^\s*(\d{1,2})\s*[\)\.\-:]\s+", r"(?m)^\s*([a-h])\s*[\)\.\-:]\s+"],
        },
        "inject_newlines": [r"(?i)\bexercice\s*(\d+|[ivx]+)\b", r"(?m)\s(\d{1,2})\)"],
        "question_keep": {"min_digits_or_symbols": 1, "min_len": 25, "max_len": 2200},
        "answer_keep": {"min_len": 25, "max_len": 2400},
        "alignment": {"min_score": 0.18, "min_char_ngram": 0.10},
        "ocr": {"enabled": True, "min_chars_before_ocr": 1200, "max_ocr_pages": 6, "ocr_dpi": 220},
    }

    f_params = {
        "epsilon": 0.1,
        "alpha": 1.0,
        "t_rec_min": 0.01,
        "determinism_lock": True,
        "reference_date_iso": "2026-01-01",
    }

    posable_rules = {
        "require_rqi": True,
        "require_scope": True,
        "require_evaluable": True,
        "min_ari_confidence": 0.55,
        "min_answer_len": 20,
    }

    clustering = {
        "min_posable_per_cluster": 2,
        "merge_similarity_threshold": 0.70,
        "allow_chapter_poor_exception": True,
    }

    saturation = {
        "vol_start": 6,
        "vol_max": 30,
        "step": 6,
        "stop_if_new_qc_zero": True,
        "safety_stop_max_iters": 6,
    }

    sources = {
        "source_id": "LOCAL_OR_STUB",
        "local_dir": "./SMAXIA_CAP_DATA/{country_code}/pdf",
        "pairing": {"corrige_tokens": ["corrig", "correction", "corrige", "solution"]},
        "date_regexes": [r"(\d{2})[_-](\d{2})[_-](20\d{2})", r"(20\d{2})[_-](\d{2})[_-](\d{2})"],
    }

    default_scope = {"subject": "MATH", "level": "TERMINALE"}

    def _cap(country_code: str, country_name: str) -> Dict[str, Any]:
        cap = {
            "pack_id": f"CAP_{country_code}_TEST_{APP_VERSION.replace('.','_')}",
            "pack_version": APP_VERSION,
            "country_code": country_code,
            "country_name": country_name,
            "language": "fr",
            "status": "SEALED",
            "_source": "EMBEDDED_CAP",
            "default_scope": default_scope,
            "chapter_taxonomy": base_math_tax,
            "ari_config": ari_config,
            "cognitive_table": cognitive_table,
            "qc_format": qc_format,
            "text_processing": text_processing,
            "atomization": atomization,
            "f_params": f_params,
            "posable_rules": posable_rules,
            "clustering": clustering,
            "saturation": saturation,
            "sources": sources,
            # IMPORTANT: pas de stub_pairs en dur requis ; fallback Pack-Driven générera si nécessaire
        }
        cap["_pack_sig_sha256"] = cap_sig(cap)
        return cap

    return {"FR": _cap("FR", "France"), "CI": _cap("CI", "Côte d’Ivoire")}


# =============================================================================
# CAP load/validate (Kernel-style)
# =============================================================================
REQUIRED_CAP_PATHS = [
    "pack_id", "country_code", "status",
    "chapter_taxonomy", "ari_config", "cognitive_table",
    "qc_format", "text_processing", "atomization",
    "f_params", "posable_rules", "clustering", "saturation", "sources"
]

def load_cap(country_code: str, uploaded_json: Optional[bytes]) -> Dict[str, Any]:
    if uploaded_json:
        cap = json.loads(uploaded_json.decode("utf-8"))
        cap["_source"] = "UPLOAD"
        cap["_pack_sig_sha256"] = cap_sig(cap)
    else:
        cap = _embedded_caps()[country_code]
    # Ensure deterministic signature exists
    if "_pack_sig_sha256" not in cap:
        cap["_pack_sig_sha256"] = cap_sig(cap)
    return cap

def validate_cap(cap: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    for k in REQUIRED_CAP_PATHS:
        if pack_get(cap, k, None) is None:
            errs.append(f"CAP_MISSING:{k}")
    if str(cap.get("status", "")).upper() != "SEALED":
        errs.append("CAP_STATUS_NOT_SEALED")
    if not isinstance(pack_get(cap, "chapter_taxonomy", {}), dict):
        errs.append("CAP_BAD_TYPE:chapter_taxonomy")
    if not isinstance(pack_get(cap, "ari_config.op_patterns", []), list):
        errs.append("CAP_BAD_TYPE:ari_config.op_patterns")
    if not isinstance(pack_get(cap, "cognitive_table.step_weights", {}), dict):
        errs.append("CAP_BAD_TYPE:cognitive_table.step_weights")
    if not isinstance(pack_get(cap, "cognitive_table.op_to_steps", {}), dict):
        errs.append("CAP_BAD_TYPE:cognitive_table.op_to_steps")
    return (len(errs) == 0), errs

def cap_default_scope(cap: Dict[str, Any]) -> Tuple[str, str]:
    subj = pack_get(cap, "default_scope.subject", None)
    lvl = pack_get(cap, "default_scope.level", None)
    if subj and lvl:
        return str(subj), str(lvl)
    # fallback: pick first subject+level deterministically from taxonomy
    tax = pack_get(cap, "chapter_taxonomy", {}) or {}
    subjects = sorted(list(tax.keys()))
    if not subjects:
        return "UNDEF", "UNDEF"
    s0 = subjects[0]
    levels = sorted(list((tax.get(s0) or {}).keys()))
    if not levels:
        return str(s0), "UNDEF"
    return str(s0), str(levels[0])


# =============================================================================
# PDF extraction (local-only) + OCR optional (still deterministic in content order)
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

def extract_text_from_pdf_bytes(pdf_bytes: bytes, cap: Dict[str, Any]) -> Tuple[str, PdfExtractMeta]:
    text_proc = pack_get(cap, "text_processing", {}) or {}
    atom = pack_get(cap, "atomization", {}) or {}
    max_pages = int(atom.get("max_pages", 120))
    ocr_cfg = atom.get("ocr", {}) or {}
    ocr_enabled = bool(ocr_cfg.get("enabled", True))
    ocr_min_chars = int(ocr_cfg.get("min_chars_before_ocr", 1200))
    ocr_pages = int(ocr_cfg.get("max_ocr_pages", 6))
    ocr_dpi = int(ocr_cfg.get("ocr_dpi", 220))

    # no wall-clock dependency used in meta.seconds (set deterministic 0.0)
    ocr_used = False

    pages = _extract_pdf_pdfplumber(pdf_bytes, max_pages)
    txt = clean_pdf_text(pages, text_proc)
    method = "pdfplumber"

    if len(txt) < 500:
        pages2 = _extract_pdf_pypdf(pdf_bytes, max_pages)
        txt2 = clean_pdf_text(pages2, text_proc)
        if len(txt2) > len(txt):
            txt, pages, method = txt2, pages2, "pypdf"

    if len(txt) < 700:
        pages3 = _extract_pdf_pymupdf_text(pdf_bytes, max_pages)
        txt3 = clean_pdf_text(pages3, text_proc)
        if len(txt3) > len(txt):
            txt, pages, method = txt3, pages3, "pymupdf_text"

    if ocr_enabled and len(txt) < ocr_min_chars:
        pages4 = _ocr_pdf_pymupdf(pdf_bytes, ocr_pages, ocr_dpi)
        txt4 = clean_pdf_text(pages4, text_proc)
        if len(txt4) > len(txt):
            txt, pages, method = txt4, pages4, "pymupdf_ocr"
            ocr_used = True

    meta = PdfExtractMeta(method=method, pages=len(pages), chars=len(txt), seconds=0.0, ocr_used=ocr_used)
    return txt, meta


# =============================================================================
# Sources loading (local PDFs if available, else Pack-Driven STUB generation)
# =============================================================================
def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _list_pdfs(local_dir: str) -> List[str]:
    if not os.path.isdir(local_dir):
        return []
    out = []
    for root, _, files in os.walk(local_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                out.append(os.path.join(root, fn))
    return sorted(out)

def _pair_pdfs(pdfs: List[str], corrige_tokens: List[str]) -> List[Dict[str, Any]]:
    def is_corrige(name: str) -> bool:
        s = norm_text(os.path.basename(name))
        return any(tok in s for tok in (corrige_tokens or []))

    sujets = [p for p in pdfs if not is_corrige(p)]
    corriges = [p for p in pdfs if is_corrige(p)]
    pairs = []
    used = set()

    def tokens(p: str) -> set:
        return set(re.findall(r"[a-z0-9]{2,}", norm_text(os.path.basename(p))))

    for su in sujets:
        su_t = tokens(su)
        best = None
        best_score = -1.0
        for co in corriges:
            if co in used:
                continue
            co_t = tokens(co)
            inter = len(su_t & co_t)
            union = max(1, len(su_t | co_t))
            score = inter / union
            if score > best_score:
                best_score = score
                best = co
        if best is not None and best_score >= 0.20:
            used.add(best)
            pair_id = "PAIR_" + stable_id(os.path.basename(su), os.path.basename(best))
            pairs.append({
                "pair_id": pair_id,
                "year": 0,
                "source_id": "LOCAL_PDF",
                "sujet_path": su,
                "corrige_path": best,
                "sujet_ref": os.path.basename(su),
                "corrige_ref": os.path.basename(best),
            })
    return pairs

def cap_chapters(cap: Dict[str, Any], subject: str, level: str) -> List[Dict[str, Any]]:
    return pack_get(cap, f"chapter_taxonomy.{subject}.{level}", []) or []

def _cap_ops_order(cap: Dict[str, Any]) -> List[str]:
    return pack_get(cap, "ari_config.primary_ops_order", []) or ["OP_STANDARD"]

def _cap_op_label(cap: Dict[str, Any], op: str) -> str:
    return str(pack_get(cap, f"ari_config.op_labels.{op}", op.replace("OP_", "").lower()))

def _cap_ch_keywords(ch: Dict[str, Any]) -> List[str]:
    kws = ch.get("keywords", []) or []
    kws2 = [norm_text(k) for k in kws if isinstance(k, str) and k.strip()]
    # deterministic unique
    out = []
    seen = set()
    for k in kws2:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

def generate_stub_pairs_pack_driven(cap: Dict[str, Any], subject: str, level: str, min_pairs: int = 2) -> List[Dict[str, Any]]:
    """
    Génération STUBS 100% Pack-Driven:
    - Utilise UNIQUEMENT CAP: chapitres, keywords, op_labels, op_patterns
    - Garantit >=2 Qi POSABLE par chapitre (si possible via CAP)
    - Sujet + Corrigé textuels cohérents (CAS1 ONLY)
    """
    chapters = cap_chapters(cap, subject, level)
    ops = _cap_ops_order(cap)
    if not chapters:
        return []

    # pick two most informative ops deterministically
    ops_use = [op for op in ops if op != "OP_STANDARD"][:2]
    if len(ops_use) < 2:
        ops_use = (ops_use + ["OP_STANDARD", "OP_STANDARD"])[:2]

    pairs: List[Dict[str, Any]] = []
    ref = det_reference_date_iso(cap)
    for idx, ch in enumerate(chapters):
        ch_code = str(ch.get("code", f"CH_{idx}"))
        kws = _cap_ch_keywords(ch)
        kw1 = kws[0] if kws else "concept"
        kw2 = kws[1] if len(kws) > 1 else kw1

        # build 2 questions per chapter using ops_use
        q_blocks = []
        a_blocks = []
        for qi_idx, op in enumerate(ops_use):
            label = _cap_op_label(cap, op)
            # Include keywords + op indicators to trigger ARI patterns deterministically
            q = f"{qi_idx+1}) {label} en utilisant {kw1} et {kw2}. Justifier."
            # answer includes same tokens + minimal numeric/maths to satisfy evaluable
            a = f"{qi_idx+1}) On applique la méthode pour {label} avec {kw1}/{kw2} : 1) définir; 2) calculer; 3) conclure."
            q_blocks.append(q)
            a_blocks.append(a)

        sujet_text = f"Exercice {idx+1}\n" + "\n".join(q_blocks)
        corrige_text = f"Corrigé Exercice {idx+1}\n" + "\n".join(a_blocks)

        pair_id = f"STUB_{cap.get('country_code','XX')}_{subject}_{level}_{ch_code}_{stable_id(ch_code, ref)}"
        pairs.append({
            "pair_id": pair_id,
            "year": int(ref.split("-")[0]) if re.match(r"^\d{4}-\d{2}-\d{2}$", ref) else 0,
            "source_id": "CAP_STUB_GEN",
            "sujet_ref": f"STUB:{cap.get('country_code','XX')}:{subject}:{level}:{ch_code}:sujet",
            "corrige_ref": f"STUB:{cap.get('country_code','XX')}:{subject}:{level}:{ch_code}:corrige",
            "sujet_text": sujet_text,
            "corrige_text": corrige_text,
        })

    # If CAP has too few chapters, repeat deterministically (still pack-driven) to reach min_pairs
    if len(pairs) < min_pairs and pairs:
        while len(pairs) < min_pairs:
            pairs.append(dict(pairs[len(pairs) % len(pairs)]))

    return pairs

def load_sources(cap: Dict[str, Any], subject: str, level: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sources = pack_get(cap, "sources", {}) or {}
    source_id = sources.get("source_id", "LOCAL_OR_STUB")
    local_dir_tpl = sources.get("local_dir", "./SMAXIA_CAP_DATA/{country_code}/pdf")
    local_dir = str(local_dir_tpl).replace("{country_code}", str(cap.get("country_code", "")))

    corrige_tokens = pack_get(cap, "sources.pairing.corrige_tokens", []) or []
    pdfs = _list_pdfs(local_dir)
    pairs_local = _pair_pdfs(pdfs, corrige_tokens)

    if pairs_local:
        manifest = {"mode": "LOCAL_PDF", "source_id": source_id, "local_dir": local_dir, "pairs_total": len(pairs_local)}
        return pairs_local, manifest

    # fallback: CAP.stub_pairs (optional) else Pack-Driven generator (mandatory if none)
    stubs = pack_get(cap, "stub_pairs", None)
    pairs_stub: List[Dict[str, Any]] = []
    if isinstance(stubs, list) and stubs:
        for sp in stubs:
            pairs_stub.append({
                "pair_id": sp.get("pair_id", "STUB_" + stable_id(safe_json_dumps(sp)[:120])),
                "year": int(sp.get("year", 0) or 0),
                "source_id": "CAP_STUB",
                "sujet_ref": sp.get("sujet_ref", ""),
                "corrige_ref": sp.get("corrige_ref", ""),
                "sujet_text": sp.get("sujet_text", ""),
                "corrige_text": sp.get("corrige_text", ""),
            })

    if not pairs_stub:
        pairs_stub = generate_stub_pairs_pack_driven(cap, subject, level, min_pairs=2)

    manifest = {"mode": "CAP_STUB", "source_id": source_id, "local_dir": local_dir, "pairs_total": len(pairs_stub)}
    return pairs_stub, manifest


# =============================================================================
# Atomization + alignment (pack-driven)
# =============================================================================
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√]")

def _inject_newlines(text: str, patterns: List[str]) -> str:
    t = text
    for pat in patterns or []:
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

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _compile_any(patterns: List[str]) -> List[re.Pattern]:
    out = []
    for p in patterns or []:
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
    if L < int(keep_cfg.get("min_len", 25)) or L > int(keep_cfg.get("max_len", 2400)):
        return False
    digits = len(re.findall(r"\d", seg))
    has_math = bool(_MATH_SYMBOL_RE.search(seg)) or digits >= 1
    return has_math or L > 120

def split_into_q_segments(text: str, cap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    atom = pack_get(cap, "atomization", {}) or {}
    markers = atom.get("markers", {}) or {}
    inject = atom.get("inject_newlines", []) or []
    min_len = int(atom.get("min_segment_len", 25))
    max_len = int(atom.get("max_segment_len", 2400))

    t = (text or "").replace("\r", "\n")
    t = _inject_newlines(t, inject)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return [], {"raw": 0, "kept": 0, "strategy": "empty"}

    regs = []
    regs += _compile_any(markers.get("exercise", []))
    regs += _compile_any(markers.get("question", []))

    positions = _cut_positions(t, regs)
    segs = _slice_segments(t, positions, min_len=min_len, max_len=max_len)

    intent_verbs = pack_get(cap, "text_processing.intent_verbs", []) or []
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

def split_into_a_segments(text: str, cap: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    atom = pack_get(cap, "atomization", {}) or {}
    inject = atom.get("inject_newlines", []) or []
    min_len = int(atom.get("min_segment_len", 25))
    max_len = int(atom.get("max_segment_len", 2400))
    keep_cfg = atom.get("answer_keep", {}) or {}

    t = (text or "").replace("\r", "\n")
    t = _inject_newlines(t, inject)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return [], {"raw": 0, "kept": 0, "strategy": "empty"}

    regs = _compile_any([
        r"(?m)^\s*(\d{1,2}|[a-h])\s*[\)\.\-:]\s+",
        r"(?im)^\s*(exercice|exo\.?)\s*([0-9ivx]+)\b",
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

def align_q_to_a(qs: List[str], ans: List[str], cap: Dict[str, Any]) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    atom = pack_get(cap, "atomization", {}) or {}
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
        best_j, best_s, best_ng = None, 0.0, 0.0
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
# Mapping chapitre (pack-driven)
# =============================================================================
def map_to_chapter(q_text: str, chapters: List[Dict[str, Any]]) -> Tuple[str, str, float]:
    if not chapters:
        return "UNMAPPED", "UNMAPPED", 0.0
    qn = norm_text(q_text)
    best_code = "UNMAPPED"
    best_label = "UNMAPPED"
    best = 0.0
    for ch in chapters:
        kws = ch.get("keywords", []) or []
        score = 0
        for kw in kws:
            if kw and norm_text(kw) in qn:
                score += 1
        denom = max(6, len(kws))
        sc = score / denom
        if sc > best:
            best = sc
            best_code = ch.get("code", "UNMAPPED")
            best_label = ch.get("label", best_code)
    return best_code, best_label, round(best, 4)


# =============================================================================
# ARI + typed steps (pack-driven)
# =============================================================================
def _ari_steps_from_ops(cap: Dict[str, Any], ops: List[str]) -> List[str]:
    op_to_steps = pack_get(cap, "cognitive_table.op_to_steps", {}) or {}
    steps = []
    for op in ops:
        steps.extend(list(op_to_steps.get(op, [])))
    if not steps:
        steps = ["STEP_STANDARD"]
    out = []
    seen = set()
    for s in steps:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def extract_ari(q: str, r: str, cap: Dict[str, Any]) -> AriTrace:
    ari_cfg = pack_get(cap, "ari_config", {}) or {}
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
    all_ops = sorted(list(seen))
    steps = _ari_steps_from_ops(cap, all_ops)

    return AriTrace(primary_op=primary, ops=ops, all_ops=all_ops, confidence_global=round(conf_global, 4), step_types=steps)

def build_triggers(q: str, ari: AriTrace, chapter_code: str, cap: Dict[str, Any]) -> List[str]:
    tp = pack_get(cap, "text_processing", {}) or {}
    kws = tp.get("trigger_keywords", []) or []
    verbs = tp.get("intent_verbs", []) or []

    out = []
    for op in ari.ops:
        out.append(f"ARI:{op.op}")

    qn = norm_text(q)
    for kw in kws:
        if kw and norm_text(kw) in qn:
            out.append(f"KW:{norm_text(kw).upper()}")

    for v in verbs:
        if v and norm_text(v) in qn:
            out.append(f"INTENT:{norm_text(v).upper()}")
            break

    if chapter_code and chapter_code != "UNMAPPED":
        out.append(f"SCOPE:{chapter_code}")

    out2 = list(dict.fromkeys(out))
    return out2[:12]

def posable_gate(q: str, r: str, chapter_code: str, ari: AriTrace, cap: Dict[str, Any]) -> PosableDecision:
    rules = pack_get(cap, "posable_rules", {}) or {}
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
def generate_frt(primary_op: str, triggers: List[str], ari: AriTrace, cap: Dict[str, Any]) -> FRT:
    ari_cfg = pack_get(cap, "ari_config", {}) or {}
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
    return FRT(frt_id=frt_id, primary_op=primary_op, title=title, sections=sections, generated_at=det_now(cap))


# =============================================================================
# QC text format (pack-driven)
# =============================================================================
def build_qc_text(primary_op: str, cap: Dict[str, Any]) -> Tuple[str, str]:
    ari_cfg = pack_get(cap, "ari_config", {}) or {}
    op_labels = ari_cfg.get("op_labels", {}) or {}
    qc_fmt = pack_get(cap, "qc_format", {}) or {}
    prefix = qc_fmt.get("prefix", "Comment")
    suffix = qc_fmt.get("suffix", "?")
    template = qc_fmt.get("template", "{prefix} {label} {suffix}")
    label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())
    qc = template.format(prefix=prefix, label=label, suffix=suffix).strip()
    return qc, label


# =============================================================================
# Kernel F1 / F2 (déterministes)
# - F1: Ψ_q = δ_c × (ε + ΣTj)^2
# - Normalisation intra-chapitre: Ψ_norm = Ψ_q / max(Ψ_q dans chapitre)
# - F2: Score(q) = (n_q/N_total) × (1+α/t_rec) × Ψ_norm × Π(1 − σ(q,p))
# =============================================================================
def _T_steps_from_ari(cap: Dict[str, Any], ari: AriTrace) -> List[Dict[str, Any]]:
    weights = pack_get(cap, "cognitive_table.step_weights", {}) or {}
    out = []
    for stp in (ari.step_types or ["STEP_STANDARD"]):
        w = float(weights.get(stp, weights.get("STEP_STANDARD", 0.8)))
        out.append({"step_type": stp, "weight": round(w, 6)})
    return out

def f1_psi(delta_c: float, epsilon: float, T_steps: List[Dict[str, Any]]) -> Tuple[float, float]:
    sum_Tj = sum(float(x.get("weight", 0.0)) for x in (T_steps or []))
    psi_brut = float(delta_c) * float((float(epsilon) + float(sum_Tj)) ** 2)
    return round(sum_Tj, 8), round(psi_brut, 8)

def _vec_from_steps(cap: Dict[str, Any], step_types: List[str]) -> Dict[str, float]:
    weights = pack_get(cap, "cognitive_table.step_weights", {}) or {}
    v = defaultdict(float)
    for stp in step_types or ["STEP_STANDARD"]:
        v[stp] += float(weights.get(stp, weights.get("STEP_STANDARD", 0.8)))
    return dict(v)

def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)

def _parse_exam_date(name: str, cap: Dict[str, Any]) -> Optional[str]:
    regs = pack_get(cap, "sources.date_regexes", []) or []
    s = name or ""
    for rg in regs:
        try:
            m = re.search(rg, s)
        except Exception:
            m = None
        if not m:
            continue
        g = m.groups()
        if len(g) == 3 and len(g[2]) == 4 and (g[0].isdigit() and g[1].isdigit() and g[2].isdigit()):
            dd, mm, yy = int(g[0]), int(g[1]), int(g[2])
            if 1 <= mm <= 12 and 1 <= dd <= 31 and yy >= 2000:
                return f"{yy:04d}-{mm:02d}-{dd:02d}"
        if len(g) == 3 and len(g[0]) == 4 and (g[0].isdigit() and g[1].isdigit() and g[2].isdigit()):
            yy, mm, dd = int(g[0]), int(g[1]), int(g[2])
            if 1 <= mm <= 12 and 1 <= dd <= 31 and yy >= 2000:
                return f"{yy:04d}-{mm:02d}-{dd:02d}"
    return None

def _t_rec_from_exam_date(exam_date: Optional[str], cap: Dict[str, Any]) -> float:
    t_rec_min = float(pack_get(cap, "f_params.t_rec_min", 0.01))
    ref = det_reference_date_iso(cap)
    if not exam_date:
        return float(t_rec_min)

    # Deterministic day-diff without datetime.now()
    def to_days(iso: str) -> Optional[int]:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
            return None
        y, m, d = iso.split("-")
        y, m, d = int(y), int(m), int(d)
        # simple serial day count (Gregorian proleptic) — deterministic
        # algorithm: Rata Die
        if m <= 2:
            y -= 1
            m += 12
        era = y // 400
        yoe = y - era * 400
        doy = (153 * (m - 3) + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        return era * 146097 + doe

    ex_d = to_days(exam_date)
    ref_d = to_days(ref)
    if ex_d is None or ref_d is None:
        return float(t_rec_min)

    days = max(0, ref_d - ex_d)
    return float(max(t_rec_min, round(days / 365.0, 6)))

def f2_score_for_candidate(
    n_q: int,
    N_total_chap: int,
    alpha: float,
    t_rec: float,
    psi_norm: float,
    selected: List["QCItem"],
    candidate: "QCItem",
    cap: Dict[str, Any]
) -> Tuple[float, List[Dict[str, Any]]]:
    base = (float(n_q) / max(1.0, float(N_total_chap))) * (1.0 + float(alpha) / max(1e-9, float(t_rec))) * float(psi_norm)
    prod = 1.0
    sig_dbg = []
    cand_steps = pack_get(candidate.frt, "sections.ari.step_types", []) or []
    cand_vec = _vec_from_steps(cap, cand_steps)
    for prev in selected:
        prev_steps = pack_get(prev.frt, "sections.ari.step_types", []) or []
        prev_vec = _vec_from_steps(cap, prev_steps)
        sig = _cosine(cand_vec, prev_vec)
        prod *= (1.0 - sig)
        sig_dbg.append({"prev_qc_id": prev.qc_id, "sigma": round(sig, 6)})
    return round(base * prod, 10), sig_dbg


# =============================================================================
# Signature SIG(q) (Pack-driven)
# =============================================================================
def _sig_q(cap: Dict[str, Any], chapter_code: str, ari: AriTrace, triggers: List[str]) -> str:
    step_types = ari.step_types or ["STEP_STANDARD"]
    trig_core = sorted([t for t in (triggers or []) if t.startswith("ARI:") or t.startswith("KW:")])[:8]
    raw = f"{chapter_code}||{ari.primary_op}||{'/'.join(step_types)}||{'/'.join(trig_core)}"
    return "SIG_" + stable_id(raw)


# =============================================================================
# Clustering (anti-singleton) + deterministic merge within chapter
# =============================================================================
def _cluster_vector(cap: Dict[str, Any], qi_items: List[QiItem]) -> Dict[str, float]:
    v = defaultdict(float)
    for qi in qi_items:
        steps = pack_get(qi.ari, "step_types", []) or []
        vv = _vec_from_steps(cap, steps)
        for k, val in vv.items():
            v[k] += float(val)
    return dict(v)

def cluster_qi(cap: Dict[str, Any], qi_items: List[QiItem], min_posable: int, merge_thr: float) -> Tuple[Dict[str, List[QiItem]], List[QiItem]]:
    clusters = defaultdict(list)
    for qi in qi_items:
        clusters[f"{qi.chapter_code}::{qi.sig_q}"].append(qi)

    def posable_count(items: List[QiItem]) -> int:
        return sum(1 for x in items if x.has_rqi and (x.posable or {}).get("is_posable"))

    strong = {}
    weak = {}
    for k, items in clusters.items():
        if posable_count(items) >= min_posable:
            strong[k] = items
        else:
            weak[k] = items

    strong_by_ch = defaultdict(list)
    for k, items in strong.items():
        ch = k.split("::", 1)[0]
        strong_by_ch[ch].append((k, items))
    for ch in list(strong_by_ch.keys()):
        strong_by_ch[ch] = sorted(strong_by_ch[ch], key=lambda x: x[0])

    orphan_candidates: List[QiItem] = []

    for wk_key in sorted(weak.keys()):
        wk_items = weak[wk_key]
        ch = wk_key.split("::", 1)[0]
        wk_pos = posable_count(wk_items)

        if wk_pos == 0:
            orphan_candidates.extend(wk_items)
            continue

        best = None
        best_sim = -1.0
        wk_vec = _cluster_vector(cap, wk_items)

        for sk_key, sk_items in strong_by_ch.get(ch, []):
            sk_vec = _cluster_vector(cap, sk_items)
            sim = _cosine(wk_vec, sk_vec)
            if (sim > best_sim) or (abs(sim - best_sim) < 1e-12 and best and sk_key < best[0]):
                best_sim = sim
                best = (sk_key, sk_items)

        if best and best_sim >= merge_thr:
            strong[best[0]].extend(wk_items)
        else:
            orphan_candidates.extend(wk_items)

    strong_final = {}
    for k, items in strong.items():
        strong_final[k] = sorted(items, key=lambda x: x.qi_id)

    return strong_final, orphan_candidates


# =============================================================================
# IA2 Judge (bool checks)
# =============================================================================
def ia2_checks(cap: Dict[str, Any], qc: QCItem) -> Tuple[bool, List[Dict[str, Any]]]:
    issues = []

    qc_fmt = pack_get(cap, "qc_format", {}) or {}
    prefix = str(qc_fmt.get("prefix", "") or "")
    suffix = str(qc_fmt.get("suffix", "") or "")
    if prefix and not qc.qc.lower().startswith(prefix.lower()):
        issues.append({"check": "CHK_QC_FORM_PREFIX", "ok": False})
    if suffix and not qc.qc.strip().endswith(suffix):
        issues.append({"check": "CHK_QC_FORM_SUFFIX", "ok": False})

    min_pos = int(pack_get(cap, "clustering.min_posable_per_cluster", 2))
    if qc.posable_in_cluster < min_pos:
        issues.append({"check": "CHK_ANTI_SINGLETON", "ok": False})

    required = pack_get(cap, "ari_config.frt_schema_required_sections", []) or []
    frt_sections = (qc.frt or {}).get("sections", {}) or {}
    for k in required:
        if k not in frt_sections:
            issues.append({"check": f"CHK_FRT_SECTION_{k}", "ok": False})

    if qc.N_total_chap <= 0:
        issues.append({"check": "CHK_F2_NTOTAL_CHAP", "ok": False})

    return (len(issues) == 0), issues


# =============================================================================
# IA1 Miner / IA1 Builder (noms explicites, logique existante)
# =============================================================================
def IA1_Miner_extract_Qi_RQi(sujet_text: str, corrige_text: str, cap: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    qs, q_audit = split_into_q_segments(sujet_text, cap)
    ans, a_audit = split_into_a_segments(corrige_text, cap)
    link, al_audit = align_q_to_a(qs, ans, cap)
    audit = {"atom_q": q_audit, "atom_a": a_audit, "align": al_audit, "links_total": len(link)}
    # attach aligned answers by index later in builder
    return qs, ans, {"miner": audit, "link": link}

def IA1_Builder_build_QiItems(
    pair: Dict[str, Any],
    qs: List[str],
    ans: List[str],
    link: List[Optional[int]],
    chapters: List[Dict[str, Any]],
    cap: Dict[str, Any],
) -> List[QiItem]:
    qi_items: List[QiItem] = []
    pair_id = pair.get("pair_id", "")
    exam_date = _parse_exam_date(pair.get("sujet_ref","") or "", cap) or _parse_exam_date(pair.get("corrige_ref","") or "", cap)

    atom_low = (len(qs) < 4)
    for i, qtxt in enumerate(qs):
        j = link[i] if i < len(link) else None
        rtxt = ans[j] if (j is not None and j < len(ans)) else ""

        ari = extract_ari(qtxt, rtxt, cap)
        ch_code, _, ch_sc = map_to_chapter(qtxt, chapters)
        triggers = build_triggers(qtxt, ari, ch_code, cap)
        pd = posable_gate(qtxt, rtxt, ch_code, ari, cap)

        rc_extra = list(pd.reason_codes)
        if atom_low:
            rc_extra = list(dict.fromkeys(rc_extra + [RC_ATOMIZATION_LOW]))
        if j is None and not rtxt.strip():
            rc_extra = list(dict.fromkeys(rc_extra + [RC_ALIGN_LOW]))

        pd2 = PosableDecision(is_posable=(len(rc_extra) == 0), reason_codes=rc_extra, confidence=pd.confidence)

        qi_id = f"QI_{stable_id(pair_id, str(i), norm_text(qtxt)[:120])}"
        sigq = _sig_q(cap, ch_code, ari, triggers)
        qi_items.append(QiItem(
            qi_id=qi_id,
            pair_id=pair_id,
            source_id=pair.get("source_id",""),
            source_year=int(pair.get("year", 0) or 0),
            source_exam_date=exam_date,
            sujet_ref=pair.get("sujet_ref",""),
            corrige_ref=pair.get("corrige_ref",""),
            qi=qtxt,
            rqi=rtxt,
            has_rqi=bool(rtxt.strip()),
            chapter_code=ch_code,
            chapter_match_score=float(ch_sc),
            ari=asdict(ari),
            triggers=triggers,
            posable=asdict(pd2),
            sig_q=sigq,
            qc_id=None,
            is_orphan=False
        ))
    return qi_items


# =============================================================================
# Pair extraction (PDF or STUB)
# =============================================================================
def _extract_pair_text(pair: Dict[str, Any], cap: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any], Optional[QuarantineEntry]]:
    source_id = pair.get("source_id", "UNKNOWN")
    pair_id = pair.get("pair_id", "")
    if source_id == "LOCAL_PDF":
        su_path = pair.get("sujet_path", "")
        co_path = pair.get("corrige_path", "")
        try:
            su_bytes = _read_file_bytes(su_path)
        except Exception as e:
            q = QuarantineEntry(pair_id=pair_id, source_id=source_id, sujet_ref=pair.get("sujet_ref",""),
                                corrige_ref=pair.get("corrige_ref",""), reason_codes=[RC_SUJET_UNREADABLE], details={"error": str(e)})
            return "", "", {}, q
        try:
            co_bytes = _read_file_bytes(co_path)
        except Exception as e:
            q = QuarantineEntry(pair_id=pair_id, source_id=source_id, sujet_ref=pair.get("sujet_ref",""),
                                corrige_ref=pair.get("corrige_ref",""), reason_codes=[RC_CORRIGE_UNREADABLE], details={"error": str(e)})
            return "", "", {}, q

        su_txt, su_meta = extract_text_from_pdf_bytes(su_bytes, cap)
        co_txt, co_meta = extract_text_from_pdf_bytes(co_bytes, cap)
        meta = {"sujet_extract": asdict(su_meta), "corrige_extract": asdict(co_meta)}
        if len(su_txt) < 200:
            q = QuarantineEntry(pair_id=pair_id, source_id=source_id, sujet_ref=pair.get("sujet_ref",""),
                                corrige_ref=pair.get("corrige_ref",""), reason_codes=[RC_SUJET_UNREADABLE],
                                details={"chars": len(su_txt), "method": su_meta.method})
            return "", "", meta, q
        if len(co_txt) < 200:
            q = QuarantineEntry(pair_id=pair_id, source_id=source_id, sujet_ref=pair.get("sujet_ref",""),
                                corrige_ref=pair.get("corrige_ref",""), reason_codes=[RC_CORRIGE_UNREADABLE],
                                details={"chars": len(co_txt), "method": co_meta.method})
            return "", "", meta, q
        return su_txt, co_txt, meta, None

    # STUB
    su_txt = pair.get("sujet_text", "") or ""
    co_txt = pair.get("corrige_text", "") or ""
    if not su_txt.strip():
        q = QuarantineEntry(pair_id=pair_id, source_id=source_id, sujet_ref=pair.get("sujet_ref",""),
                            corrige_ref=pair.get("corrige_ref",""), reason_codes=[RC_SUJET_UNREADABLE], details={"why": "stub sujet_text empty"})
        return "", "", {}, q
    if not co_txt.strip():
        q = QuarantineEntry(pair_id=pair_id, source_id=source_id, sujet_ref=pair.get("sujet_ref",""),
                            corrige_ref=pair.get("corrige_ref",""), reason_codes=[RC_CORRIGE_MISSING], details={"why": "stub corrige_text empty"})
        return "", "", {}, q
    meta = {"sujet_extract": {"method": "STUB", "pages": 0, "chars": len(su_txt), "seconds": 0.0, "ocr_used": False},
            "corrige_extract": {"method": "STUB", "pages": 0, "chars": len(co_txt), "seconds": 0.0, "ocr_used": False}}
    return su_txt, co_txt, meta, None


# =============================================================================
# Pipeline end-to-end (ISO-PROD logique) + saturation auto
# =============================================================================
def run_once(cap: Dict[str, Any], pairs: List[Dict[str, Any]], volume: int, subject: str, level: str, source_manifest: Dict[str, Any]) -> Dict[str, Any]:
    chapters = cap_chapters(cap, subject, level)

    # deterministic slice
    pairs2 = pairs[:max(0, min(volume, len(pairs)))]
    quarantine: List[QuarantineEntry] = []
    qi_items: List[QiItem] = []

    epsilon = float(pack_get(cap, "f_params.epsilon", 0.1))
    alpha = float(pack_get(cap, "f_params.alpha", 1.0))

    pair_meta: List[Dict[str, Any]] = []

    for pair in pairs2:
        pair_id = pair.get("pair_id", "")
        su_txt, co_txt, meta, qent = _extract_pair_text(pair, cap)
        if qent:
            quarantine.append(qent)
            pair_meta.append({"pair_id": pair_id, "meta": meta, "quarantine": asdict(qent)})
            continue

        qs, ans, miner_out = IA1_Miner_extract_Qi_RQi(su_txt, co_txt, cap)
        link = miner_out.get("link", [])
        pair_meta.append({"pair_id": pair_id, "meta": meta, **miner_out.get("miner", {})})

        qi_items.extend(IA1_Builder_build_QiItems(pair, qs, ans, link, chapters, cap))

    # POSABLE set
    posable_qi = [q for q in qi_items if q.has_rqi and (q.posable or {}).get("is_posable")]

    # clustering
    min_posable = int(pack_get(cap, "clustering.min_posable_per_cluster", 2))
    merge_thr = float(pack_get(cap, "clustering.merge_similarity_threshold", 0.70))
    clusters, _ = cluster_qi(cap, qi_items, min_posable=min_posable, merge_thr=merge_thr)

    # Build QC candidates
    qc_candidates: List[QCItem] = []
    qi_to_qc: Dict[str, str] = {}

    # N_total per chapter (POSABLE only)
    posable_by_ch = defaultdict(list)
    for qi in posable_qi:
        posable_by_ch[qi.chapter_code].append(qi)
    N_total_by_ch = {ch: len(items) for ch, items in posable_by_ch.items()}

    ch_info = {c.get("code"): c for c in chapters}

    for cl_key in sorted(clusters.keys()):
        items = clusters[cl_key]
        pos_count = sum(1 for x in items if x.has_rqi and (x.posable or {}).get("is_posable"))
        if pos_count < min_posable:
            continue

        rep = max(items, key=lambda z: float((z.ari or {}).get("confidence_global", 0.0)))
        rep_ari = rep.ari or {}
        primary_op = str(rep_ari.get("primary_op", "OP_STANDARD"))
        qc_text, qc_label = build_qc_text(primary_op, cap)
        ch_code = rep.chapter_code
        ch_label = ch_info.get(ch_code, {}).get("label", ch_code)

        trig_counts = Counter()
        all_ops = set()
        for it in items:
            trig_counts.update(it.triggers or [])
            all_ops.update((it.ari or {}).get("all_ops", []) or [])
        qc_triggers = [k for k, _ in trig_counts.most_common(8)]
        qi_ids = [x.qi_id for x in items]

        # FRT from rep
        rep_trace = AriTrace(
            primary_op=str(rep_ari.get("primary_op","OP_STANDARD")),
            ops=[AriOpEvidence(**o) for o in (rep_ari.get("ops") or [])],
            all_ops=list(rep_ari.get("all_ops") or []),
            confidence_global=float(rep_ari.get("confidence_global", 0.55)),
            step_types=list(rep_ari.get("step_types") or ["STEP_STANDARD"])
        )
        frt = generate_frt(primary_op, qc_triggers, rep_trace, cap)

        # F1
        delta_c = float(ch_info.get(ch_code, {}).get("delta_c", 1.0))
        T_steps = _T_steps_from_ari(cap, rep_trace)
        sum_Tj, psi_brut = f1_psi(delta_c, epsilon, T_steps)

        # t_rec from latest exam_date in cluster (deterministic reference date in CAP)
        dates = [x.source_exam_date for x in items if x.source_exam_date]
        exam_date = max(dates) if dates else None
        t_rec = _t_rec_from_exam_date(exam_date, cap)

        qc_id = "QC_" + stable_id(ch_code, primary_op, qc_text, ",".join(sorted(qi_ids))[:80])

        evidence_min = {
            "pair_ids": sorted({x.pair_id for x in items})[:5],
            "source_id": rep.source_id,
            "sujet_refs": sorted({x.sujet_ref for x in items})[:5],
            "corrige_refs": sorted({x.corrige_ref for x in items})[:5],
            "qi_ids_sample": qi_ids[:6],
        }

        qc_candidates.append(QCItem(
            qc_id=qc_id,
            qc=qc_text,
            qc_label=qc_label,
            chapter_code=ch_code,
            chapter_label=ch_label,
            primary_op=primary_op,
            all_ops=sorted(list(all_ops)) if all_ops else ["OP_STANDARD"],
            cluster_size=len(items),
            posable_in_cluster=pos_count,
            qi_ids=qi_ids,
            triggers=qc_triggers,
            frt=asdict(frt),
            delta_c=delta_c,
            epsilon=epsilon,
            T_steps=T_steps,
            sum_Tj=sum_Tj,
            psi_brut=psi_brut,
            psi_norm=0.0,
            n_q=pos_count,
            N_total_chap=int(N_total_by_ch.get(ch_code, 0)),
            alpha=alpha,
            t_rec=float(t_rec),
            sigma_terms=[],
            score_f2=0.0,
            evidence_min=evidence_min
        ))

    # normalize psi intra-chapter
    by_ch = defaultdict(list)
    for qc in qc_candidates:
        by_ch[qc.chapter_code].append(qc)
    for ch, qcs in by_ch.items():
        mx = max((q.psi_brut for q in qcs), default=0.0)
        for q in qcs:
            q.psi_norm = round((q.psi_brut / mx) if mx > 0 else 0.0, 10)

    # Selection per chapter (set cover greedy + F2 scoring)
    selected_by_ch: Dict[str, List[QCItem]] = {}
    coverage_map: Dict[str, Any] = {}
    safety_stop = False
    safety_notes: List[Dict[str, Any]] = []

    for ch in sorted(posable_by_ch.keys()):
        uncovered = set([q.qi_id for q in posable_by_ch[ch]])
        candidates = [q for q in qc_candidates if q.chapter_code == ch and q.posable_in_cluster >= min_posable]
        candidates = sorted(candidates, key=lambda x: x.qc_id)

        selected: List[QCItem] = []
        while uncovered:
            best = None
            best_gain = -1
            best_score = -1.0

            for cand in candidates:
                covers = set(cand.qi_ids) & uncovered
                gain = len(covers)
                if gain <= 0:
                    continue
                score, sig_dbg = f2_score_for_candidate(
                    n_q=cand.n_q,
                    N_total_chap=max(1, cand.N_total_chap),
                    alpha=cand.alpha,
                    t_rec=cand.t_rec,
                    psi_norm=cand.psi_norm,
                    selected=selected,
                    candidate=cand,
                    cap=cap
                )
                if (gain > best_gain) or (gain == best_gain and score > best_score) or (gain == best_gain and abs(score-best_score) < 1e-12 and best and cand.qc_id < best.qc_id):
                    cand.score_f2 = score
                    cand.sigma_terms = sig_dbg
                    best = cand
                    best_gain = gain
                    best_score = score

            if best is None:
                safety_stop = True
                safety_notes.append({"chapter_code": ch, "reason": "NO_PROGRESS_SET_COVER", "uncovered_count": len(uncovered)})
                break

            selected.append(best)
            uncovered = uncovered - set(best.qi_ids)
            candidates = [c for c in candidates if c.qc_id != best.qc_id]

        selected_by_ch[ch] = selected
        coverage_map[ch] = {
            "N_total_chap": int(N_total_by_ch.get(ch, 0)),
            "covered": int(N_total_by_ch.get(ch, 0) - len(uncovered)),
            "uncovered": sorted(list(uncovered))[:200],
            "coverage": round((float(N_total_by_ch.get(ch, 0) - len(uncovered)) / max(1, float(N_total_by_ch.get(ch, 0)))), 6)
        }

    selected_all: List[QCItem] = []
    for ch in sorted(selected_by_ch.keys()):
        selected_all.extend(selected_by_ch[ch])

    # IA2 judge on selected
    ia2_ok = True
    ia2_issues = []
    for qc in selected_all:
        ok, issues = ia2_checks(cap, qc)
        if not ok:
            ia2_ok = False
            ia2_issues.extend([{"qc_id": qc.qc_id, **x} for x in issues])

    # assignment Qi -> QC (first covering qc)
    for qi in qi_items:
        qi.qc_id = None
        qi.is_orphan = False

    for ch in sorted(selected_by_ch.keys()):
        for qc in selected_by_ch[ch]:
            for qid in qc.qi_ids:
                if qid not in qi_to_qc:
                    qi_to_qc[qid] = qc.qc_id

    orphans_posable = 0
    for qi in qi_items:
        if qi.qi_id in qi_to_qc:
            qi.qc_id = qi_to_qc[qi.qi_id]
            qi.is_orphan = False
        else:
            qi.qc_id = None
            if qi.has_rqi and (qi.posable or {}).get("is_posable"):
                qi.is_orphan = True
                orphans_posable += 1

    cov_list = [coverage_map[ch]["coverage"] for ch in coverage_map.keys()] or [0.0]
    cov_global = round(sum(cov_list) / max(1, len(cov_list)), 6)

    audit = {
        "pairs_used": len(pairs2),
        "pairs_total": len(pairs),
        "qi_total": len(qi_items),
        "qi_posable_total": len(posable_qi),
        "qc_candidates_total": len(qc_candidates),
        "qc_selected_total": sum(len(v) for v in selected_by_ch.values()),
        "orphans_posable": int(orphans_posable),
        "coverage_by_chapter": coverage_map,
        "coverage_global_mean": cov_global,
        "ia2_ok": ia2_ok,
        "ia2_issues": ia2_issues[:200],
        "safety_stop": bool(safety_stop),
        "safety_notes": safety_notes[:50],
        "quarantine_count": len([]),
    }

    evidence_pack = {
        "app_version": APP_VERSION,
        "det_stamp": det_now(cap),
        "cap_id": cap.get("pack_id"),
        "cap_sig_sha256": cap.get("_pack_sig_sha256"),
        "scope": {"subject": subject, "level": level},
        "source_manifest": source_manifest,
        "audit": audit,
        "pair_meta_sample": pair_meta[:50],
        "qc_selected_evidence": [
            {
                "qc_id": qc.qc_id,
                "chapter_code": qc.chapter_code,
                "terms": {
                    "delta_c": qc.delta_c,
                    "epsilon": qc.epsilon,
                    "T_steps": qc.T_steps,
                    "sum_Tj": qc.sum_Tj,
                    "psi_brut": qc.psi_brut,
                    "psi_norm": qc.psi_norm,
                    "n_q": qc.n_q,
                    "N_total_chap": qc.N_total_chap,
                    "alpha": qc.alpha,
                    "t_rec": qc.t_rec,
                    "sigma_terms": qc.sigma_terms,
                    "score_f2": qc.score_f2,
                },
                "evidence_min": qc.evidence_min,
            } for qc in selected_all
        ],
    }

    return {
        "qi_items": [asdict(x) for x in qi_items],
        "qc_candidates": [asdict(x) for x in qc_candidates],
        "qc_selected": [asdict(x) for x in selected_all],
        "selected_by_ch": {k: [asdict(x) for x in v] for k, v in selected_by_ch.items()},
        "audit": audit,
        "evidence_pack": evidence_pack,
    }

def run_saturation_auto(cap: Dict[str, Any], pairs: List[Dict[str, Any]], subject: str, level: str, source_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Saturation automatique:
    - itère sur volumes (CAP.saturation) jusqu'à:
      - orphans_posable == 0
      - ia2_ok == True
      - safety_stop == False
    - OU stop_if_new_qc_zero
    - IMPORTANT: si pairs vide => sortie déterministe + safety_stop + RC_NO_SOURCES (pas d'assert)
    """
    sat = pack_get(cap, "saturation", {}) or {}
    vol_start = int(sat.get("vol_start", 6))
    vol_max = int(sat.get("vol_max", 30))
    step = int(sat.get("step", 6))
    max_iters = int(sat.get("safety_stop_max_iters", 6))

    if not pairs:
        # deterministic empty run (should not happen because generator enforces stubs,
        # but we guard to avoid crash)
        empty_audit = {
            "pairs_used": 0,
            "pairs_total": 0,
            "qi_total": 0,
            "qi_posable_total": 0,
            "qc_candidates_total": 0,
            "qc_selected_total": 0,
            "orphans_posable": 0,
            "coverage_by_chapter": {},
            "coverage_global_mean": 0.0,
            "ia2_ok": False,
            "ia2_issues": [{"qc_id": "NA", "check": RC_NO_SOURCES, "ok": False}],
            "safety_stop": True,
            "safety_notes": [{"reason": RC_NO_SOURCES}],
            "quarantine_count": 0,
        }
        return {
            "qi_items": [],
            "qc_candidates": [],
            "qc_selected": [],
            "selected_by_ch": {},
            "audit": empty_audit,
            "evidence_pack": {
                "app_version": APP_VERSION,
                "det_stamp": det_now(cap),
                "cap_id": cap.get("pack_id"),
                "cap_sig_sha256": cap.get("_pack_sig_sha256"),
                "scope": {"subject": subject, "level": level},
                "source_manifest": source_manifest,
                "audit": empty_audit,
                "qc_selected_evidence": [],
            },
            "saturation_iterations": [],
        }

    iterations = []
    prev_qc_set = None
    final: Optional[Dict[str, Any]] = None

    it = 0
    hi = min(vol_max, len(pairs))
    for vol in range(max(1, min(vol_start, hi)), hi + 1, max(1, step)):
        it += 1
        out = run_once(cap, pairs, vol, subject, level, source_manifest)
        qc_ids = [q["qc_id"] for q in out["qc_candidates"]]
        cur_set = set(qc_ids)

        new_qc = len(cur_set) if prev_qc_set is None else len(cur_set - prev_qc_set)
        prev_qc_set = cur_set

        audit = out["audit"]
        iterations.append({
            "volume": vol,
            "pairs_used": audit["pairs_used"],
            "qi_total": audit["qi_total"],
            "qi_posable": audit["qi_posable_total"],
            "qc_candidates": audit["qc_candidates_total"],
            "qc_selected": audit["qc_selected_total"],
            "orphans_posable": audit["orphans_posable"],
            "coverage_mean": audit["coverage_global_mean"],
            "ia2_ok": audit["ia2_ok"],
            "new_qc": new_qc,
            "safety_stop": audit["safety_stop"],
        })

        final = out

        if audit["orphans_posable"] == 0 and audit["ia2_ok"] and (not audit["safety_stop"]):
            break
        if bool(sat.get("stop_if_new_qc_zero", True)) and new_qc == 0:
            break
        if it >= max_iters:
            break

    if final is None:
        # Guard (shouldn't happen with pairs non-empty)
        final = run_once(cap, pairs, min(1, len(pairs)), subject, level, source_manifest)

    final["saturation_iterations"] = iterations
    return final


# =============================================================================
# UI — Action unique: choisir le pays puis ACTIVER, ensuite affichage FINAL UNIQUEMENT
# =============================================================================
def ss_init():
    defaults = {
        "cap": None,
        "cap_valid": False,
        "cap_errors": [],
        "country_code": None,
        "run_done": False,
        "run_output": None,
        "source_manifest": {},
        "subject": None,
        "level": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def render_final_view(out: Dict[str, Any]):
    """
    EXIGENCE UTILISATEUR: afficher UNIQUEMENT QC / FRT / ARI / TRIGGERS par chapitre
    + Qi associées visibles.
    """
    qi_items = out.get("qi_items", []) or []
    qi_by_id = {q["qi_id"]: q for q in qi_items}

    selected_by_ch = out.get("selected_by_ch", {}) or {}
    audit = out.get("audit", {}) or {}
    cov_by_ch = (audit.get("coverage_by_chapter", {}) or {})

    # If nothing selected, show deterministic minimal failure indicator (still within output domain)
    if not selected_by_ch:
        st.error("Aucune QC sélectionnée (échec couverture / IA2 / sources).")
        st.json({"audit": audit, "source_manifest": out.get("evidence_pack", {}).get("source_manifest", {})})
        return

    for ch_code in sorted(selected_by_ch.keys()):
        qcs = selected_by_ch.get(ch_code, [])
        if not qcs:
            continue
        ch_label = qcs[0].get("chapter_label", ch_code)
        cov = cov_by_ch.get(ch_code, {})
        cov_pct = float(cov.get("coverage", 0.0) or 0.0)
        uncovered = cov.get("uncovered", []) or []
        title = f"{ch_label} [{ch_code}] — Coverage={cov_pct:.0%} — Uncovered={len(uncovered)}"
        with st.expander(title, expanded=False):
            if uncovered:
                st.error(f"Uncovered POSABLE (échantillon): {uncovered[:20]}")

            for qc in qcs:
                st.markdown(f"### {qc['qc']}")
                st.markdown("**Triggers**")
                st.write(qc.get("triggers", []))

                st.markdown("**ARI (rep) + steps typed**")
                st.json(pack_get(qc, "frt.sections.ari", {}) or {})

                st.markdown("**FRT**")
                st.json(qc.get("frt", {}))

                st.markdown("**Qi associées (Qi + RQi)**")
                for qid in qc.get("qi_ids", [])[:60]:
                    q = qi_by_id.get(qid)
                    if not q:
                        continue
                    tag = "POSABLE" if (q.get("posable") or {}).get("is_posable", False) else "UNPOSABLE"
                    with st.expander(f"{qid} • {tag}", expanded=False):
                        st.markdown("Qi")
                        st.text((q.get("qi","") or "")[:2000])
                        st.markdown("RQi")
                        st.text((q.get("rqi","") or "")[:2000])

def main():
    st.set_page_config(page_title=APP_NAME, layout="wide")
    ss_init()

    st.title(APP_NAME)
    st.caption("Kernel V10.6.3 • TEST = PROD (logique) • CAS 1 ONLY • Déterminisme • Zéro hardcode (Pack-Driven) • Mode déconnecté")

    with st.form("activate_country_form", clear_on_submit=False):
        c1, c2 = st.columns([2, 3])
        with c1:
            country_label = st.selectbox("Choisir le pays (action unique)", ["France", "Côte d’Ivoire"])
            country_code = "FR" if country_label == "France" else "CI"
        with c2:
            uploaded_cap = st.file_uploader("CAP JSON (optionnel) — si non fourni, CAP scellé embedded utilisé", type=["json"])
        submitted = st.form_submit_button("ACTIVER", type="primary", use_container_width=True)

    if submitted:
        cap = load_cap(country_code, uploaded_cap.read() if uploaded_cap else None)
        ok, errs = validate_cap(cap)

        st.session_state.cap = cap
        st.session_state.cap_valid = ok
        st.session_state.cap_errors = errs
        st.session_state.country_code = country_code

        if not ok:
            st.error("CAP invalide — exécution bloquée.")
            st.write(errs)
            return

        subject, level = cap_default_scope(cap)
        st.session_state.subject = subject
        st.session_state.level = level

        pairs, manifest = load_sources(cap, subject, level)
        st.session_state.source_manifest = manifest

        with st.spinner("Pipeline automatique ISO-PROD (logique) en cours..."):
            out = run_saturation_auto(cap=cap, pairs=pairs, subject=subject, level=level, source_manifest=manifest)

        st.session_state.run_output = out
        st.session_state.run_done = True

    if not st.session_state.run_done or not st.session_state.run_output:
        return

    # EXIGENCE: affichage final uniquement
    render_final_view(st.session_state.run_output)

main()
