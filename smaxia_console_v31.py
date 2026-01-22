# =============================================================================
# SMAXIA GTE Console V31.10.25 — ISO-PROD (FICHIER UNIQUE)
# =============================================================================
# DOCTRINE (Kernel V10.6.3) — TEST = PROD (LOGIQUE), DÉTERMINISME, CAS 1 ONLY
# - Action humaine unique : choisir + valider un pays (France | Côte d’Ivoire)
# - Ensuite : pipeline automatique bout-en-bout → affichage QC/FRT/ARI/TRIGGERS par chapitre
# - AUCUN HARDCODE MÉTIER : toute variabilité (chapitres, keywords, params, poids) provient du CAP (LOAD_CAP)
# - Mode déconnecté : si aucune source PDF locale n’est disponible, utilisation de STUBS conformes au CAP (données CAP)
# - Coverage : 100% des Qi POSABLE par chapitre (zéro orphelin) OU safety-stop audité (Kernel)
# - Anti-singleton : cluster POSABLE >= 2 (exception “chapitre pauvre” uniquement si CAP le permet, auditée)
# - EvidencePack minimal : chaque QC expose des preuves (références pair_id / source_id / qi_ids)
#
# IMPORTANT
# - Ce script ne fait AUCUN appel réseau.
# - Pour des PDFs réels : déposer les fichiers dans le dossier défini dans le CAP (cap.sources.local_dir).
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import math
import time
import hashlib
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import streamlit as st

# ---------------- Optional deps (local only) ----------------
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
    # ARI steps typed (Kernel: chaque étape ARI est typée)
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
    # routing / coverage
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
    # cluster
    cluster_size: int
    posable_in_cluster: int
    qi_ids: List[str]
    triggers: List[str]
    frt: Dict[str, Any]
    # F1/F2 (Kernel)
    delta_c: float
    epsilon: float
    T_steps: List[Dict[str, Any]]   # [{step_type, weight}]
    sum_Tj: float
    psi_brut: float
    psi_norm: float
    # selection / redundancy
    n_q: int
    N_total_chap: int
    alpha: float
    t_rec: float
    sigma_terms: List[Dict[str, Any]]  # debug
    score_f2: float
    # evidence
    evidence_min: Dict[str, Any]


# =============================================================================
# Time / deterministic helpers
# =============================================================================
def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

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


# =============================================================================
# CAP (EMBEDDED) — données (PAS de logique métier dans le core)
# =============================================================================
def _embedded_caps() -> Dict[str, Dict[str, Any]]:
    # NOTE: Ces CAPs sont des CAPs de TEST déconnecté.
    # Ils respectent la doctrine: toute variabilité (chapitres/keywords/params/poids) est dans le CAP.
    # Le CORE ne “connait” pas les chapitres.
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

    # Cognitive table (poids T_j) — CAP data
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
        }
    }

    # IA templates (CAP data)
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
                    "Calculer, vérifier, conclure"
                ],
                "checks": ["Valeur dans [0,1]", "Interprétation correcte"],
                "pitfalls": ["Mauvaise loi", "Arrondis prématurés"]
            },
            "OP_DERIVE": {
                "procedure": [
                    "Identifier f et le domaine",
                    "Appliquer les règles de dérivation",
                    "Simplifier f'(x)",
                    "Exploiter (variations / tangente) si demandé"
                ],
                "checks": ["Domaine", "Signe de f' si tableau"],
                "pitfalls": ["Erreur de règle", "Oubli domaine"]
            },
            "OP_STANDARD": {
                "procedure": [
                    "Lire et reformuler la demande",
                    "Lister données / inconnues",
                    "Choisir la méthode",
                    "Exécuter les calculs",
                    "Vérifier et conclure"
                ],
                "checks": ["Résultat plausible", "Conclusion explicite"],
                "pitfalls": ["Mauvaise méthode", "Erreurs algébriques"]
            }
        }
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

    # Kernel table (doc): epsilon = 0.1 (CAP may override but here aligned)
    f_params = {
        "epsilon": 0.1,
        "alpha": 1.0,
        "t_rec_min": 0.01,
        "determinism_lock": True,
    }

    # POSABLE rules (CAP)
    posable_rules = {
        "require_rqi": True,
        "require_scope": True,
        "require_evaluable": True,
        "min_ari_confidence": 0.55,
        "min_answer_len": 20
    }

    # Clustering (CAP)
    clustering = {
        "min_posable_per_cluster": 2,
        "merge_similarity_threshold": 0.70,   # cosine on step-vectors
        "allow_chapter_poor_exception": True
    }

    # Saturation schedule (CAP)
    saturation = {
        "vol_start": 6,
        "vol_max": 30,
        "step": 6,
        "stop_if_new_qc_zero": True,
        "safety_stop_max_iters": 6,
    }

    # Local source dir (CAP) — user can place PDFs here.
    # Pairing rule is also CAP-driven (simple filename tokens).
    sources = {
        "source_id": "LOCAL_OR_STUB",
        "local_dir": "./SMAXIA_CAP_DATA/{country_code}/pdf",
        "pairing": {
            "corrige_tokens": ["corrig", "correction", "corrige", "solution"],
        },
        "date_regexes": [r"(\d{2})[_-](\d{2})[_-](20\d{2})", r"(20\d{2})[_-](\d{2})[_-](\d{2})"],
    }

    # STUB pairs (CAP data) — CAS1 ONLY: sujet + corrige (texte)
    # Intention: fournir au minimum 2 Qi POSABLE par chapitre pour respecter anti-singleton + coverage.
    stub_pairs_fr = [
        {
            "pair_id": "STUB_FR_01",
            "year": 2023,
            "sujet_ref": "STUB:FR:sujet:01",
            "corrige_ref": "STUB:FR:corrige:01",
            "sujet_text": """Exercice 1
1) Calculer la limite de f(x) quand x tend vers +∞.
2) Étudier la dérivée f'(x) et dresser le tableau de variations.
Exercice 2
1) Calculer P(X=3) où X suit une loi binomiale.
2) Calculer l'espérance de X.""",
            "corrige_text": """Exercice 1
1) On calcule la limite en utilisant les propriétés usuelles : ...
2) On dérive f, on étudie le signe de f' : ...
Exercice 2
1) P(X=3)=C(n,3)p^3(1-p)^(n-3) : ...
2) E(X)=np : ..."""
        },
        {
            "pair_id": "STUB_FR_02",
            "year": 2023,
            "sujet_ref": "STUB:FR:sujet:02",
            "corrige_ref": "STUB:FR:corrige:02",
            "sujet_text": """Exercice 1
1) Calculer l'intégrale ∫_0^1 (1+x) dx.
2) Déterminer une primitive de g(x)=e^x.
Exercice 2
1) Définir la suite (u_n) et étudier sa convergence.
2) Démontrer par récurrence que u_n ≥ 0.""",
            "corrige_text": """Exercice 1
1) ∫_0^1 (1+x) dx = [x + x^2/2]_0^1 = ...
2) Une primitive est e^x : ...
Exercice 2
1) On étudie la suite : ...
2) Initialisation + hérédité : ..."""
        },
        {
            "pair_id": "STUB_FR_03",
            "year": 2024,
            "sujet_ref": "STUB:FR:sujet:03",
            "corrige_ref": "STUB:FR:corrige:03",
            "sujet_text": """Exercice 1
1) Calculer la limite de h(x) quand x tend vers 0.
2) Déterminer la dérivée de h(x)=ln(1+x) et conclure.
Exercice 2
1) X ~ loi normale : calculer P(a ≤ X ≤ b).
2) Calculer la variance de X.""",
            "corrige_text": """Exercice 1
1) Limite par équivalent : ...
2) h'(x)=1/(1+x) : ...
Exercice 2
1) Standardisation puis table : ...
2) Var(X)=σ^2 : ..."""
        }
    ]

    stub_pairs_ci = [
        {
            "pair_id": "STUB_CI_01",
            "year": 2023,
            "sujet_ref": "STUB:CI:sujet:01",
            "corrige_ref": "STUB:CI:corrige:01",
            "sujet_text": """Exercice 1
1) Calculer la limite de f(x) quand x tend vers +∞.
2) Résoudre l'équation f(x)=0.
Exercice 2
1) Calculer une probabilité dans une loi binomiale.
2) Calculer l'espérance.""",
            "corrige_text": """Exercice 1
1) Limite : ...
2) Résolution : ...
Exercice 2
1) Formule binomiale : ...
2) E(X)=np : ..."""
        },
        {
            "pair_id": "STUB_CI_02",
            "year": 2024,
            "sujet_ref": "STUB:CI:sujet:02",
            "corrige_ref": "STUB:CI:corrige:02",
            "sujet_text": """Exercice 1
1) Calculer l'intégrale ∫_0^2 x dx.
2) Déterminer une primitive de g(x)=x^2.
Exercice 2
1) Étudier une suite.
2) Démontrer par récurrence une propriété.""",
            "corrige_text": """Exercice 1
1) ∫_0^2 x dx = [x^2/2]_0^2 = ...
2) Une primitive est x^3/3 : ...
Exercice 2
1) Convergence : ...
2) Initialisation + hérédité : ..."""
        }
    ]

    def _cap(country_code: str, country_name: str, stubs: List[Dict[str, Any]]) -> Dict[str, Any]:
        cap = {
            "pack_id": f"CAP_{country_code}_TEST_{APP_VERSION.replace('.','_')}",
            "pack_version": APP_VERSION,
            "country_code": country_code,
            "country_name": country_name,
            "language": "fr",
            "status": "SEALED",
            "created_at": _utc_ts(),
            "_source": "EMBEDDED_CAP_STUB",
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
            "stub_pairs": stubs,
        }
        cap["_pack_sig_sha256"] = cap_sig(cap)
        return cap

    return {
        "FR": _cap("FR", "France", stub_pairs_fr),
        "CI": _cap("CI", "Côte d’Ivoire", stub_pairs_ci),
    }


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
    return cap

def validate_cap(cap: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    for k in REQUIRED_CAP_PATHS:
        if pack_get(cap, k, None) is None:
            errs.append(f"CAP_MISSING:{k}")
    if str(cap.get("status", "")).upper() != "SEALED":
        errs.append("CAP_STATUS_NOT_SEALED")
    # minimal type checks
    if not isinstance(pack_get(cap, "chapter_taxonomy", {}), dict):
        errs.append("CAP_BAD_TYPE:chapter_taxonomy")
    if not isinstance(pack_get(cap, "ari_config.op_patterns", []), list):
        errs.append("CAP_BAD_TYPE:ari_config.op_patterns")
    return (len(errs) == 0), errs


# =============================================================================
# PDF text extraction (local-only) + OCR optional
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

    t0 = time.time()
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

    dt = time.time() - t0
    meta = PdfExtractMeta(method=method, pages=len(pages), chars=len(txt), seconds=round(dt, 3), ocr_used=ocr_used)
    return txt, meta


# =============================================================================
# Source loading (local dir if available, else CAP.stub_pairs)
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
    # Deterministic pairing by filename tokens (CAP data)
    # Sujet = not containing corrige tokens; Corrige = containing corrige tokens
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

def load_sources(cap: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sources = pack_get(cap, "sources", {}) or {}
    source_id = sources.get("source_id", "LOCAL_OR_STUB")
    local_dir_tpl = sources.get("local_dir", "./SMAXIA_CAP_DATA/{country_code}/pdf")
    local_dir = str(local_dir_tpl).replace("{country_code}", str(cap.get("country_code", "")))

    corrige_tokens = pack_get(cap, "sources.pairing.corrige_tokens", []) or []
    pdfs = _list_pdfs(local_dir)
    pairs_local = _pair_pdfs(pdfs, corrige_tokens)

    if pairs_local:
        manifest = {
            "mode": "LOCAL_PDF",
            "source_id": source_id,
            "local_dir": local_dir,
            "pairs_total": len(pairs_local),
        }
        return pairs_local, manifest

    # fallback to stubs
    stubs = pack_get(cap, "stub_pairs", []) or []
    pairs_stub = []
    for sp in stubs:
        pairs_stub.append({
            "pair_id": sp["pair_id"],
            "year": int(sp.get("year", 0) or 0),
            "source_id": "CAP_STUB",
            "sujet_ref": sp.get("sujet_ref", ""),
            "corrige_ref": sp.get("corrige_ref", ""),
            "sujet_text": sp.get("sujet_text", ""),
            "corrige_text": sp.get("corrige_text", ""),
        })

    manifest = {
        "mode": "CAP_STUB",
        "source_id": source_id,
        "local_dir": local_dir,
        "pairs_total": len(pairs_stub),
    }
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
# Scope mapping (pack-driven)
# =============================================================================
def cap_chapters(cap: Dict[str, Any], subject: str, level: str) -> List[Dict[str, Any]]:
    return pack_get(cap, f"chapter_taxonomy.{subject}.{level}", []) or []

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
# ARI extraction (pack-driven, evidence) + typed steps (via CAP cognitive_table)
# =============================================================================
def _ari_steps_from_ops(cap: Dict[str, Any], ops: List[str]) -> List[str]:
    op_to_steps = pack_get(cap, "cognitive_table.op_to_steps", {}) or {}
    steps = []
    for op in ops:
        steps.extend(list(op_to_steps.get(op, [])))
    if not steps:
        steps = ["STEP_STANDARD"]
    # unique, stable order
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

    return AriTrace(
        primary_op=primary,
        ops=ops,
        all_ops=all_ops,
        confidence_global=round(conf_global, 4),
        step_types=steps
    )

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

    # unique stable
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
    return FRT(frt_id=frt_id, primary_op=primary_op, title=title, sections=sections, generated_at=_utc_ts())


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
# Kernel F1 / F2 (aligné sur Kernel V10.6.3 tables)
# - F1: Ψ_q recalculable depuis {δ_c, ε, T_j} (T_j tracés via ARI step_types + CAP cognitive_table)
# - F2: Score(q) = (n_q / N_total[chap]) × (1 + α / t_rec) × Ψ_q × Π(1 − σ(q,p))
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
        # dd_mm_yyyy
        if len(g) == 3 and len(g[2]) == 4 and (g[0].isdigit() and g[1].isdigit() and g[2].isdigit()):
            dd, mm, yy = int(g[0]), int(g[1]), int(g[2])
            if 1 <= mm <= 12 and 1 <= dd <= 31 and yy >= 2000:
                return f"{yy:04d}-{mm:02d}-{dd:02d}"
        # yyyy_mm_dd
        if len(g) == 3 and len(g[0]) == 4 and (g[0].isdigit() and g[1].isdigit() and g[2].isdigit()):
            yy, mm, dd = int(g[0]), int(g[1]), int(g[2])
            if 1 <= mm <= 12 and 1 <= dd <= 31 and yy >= 2000:
                return f"{yy:04d}-{mm:02d}-{dd:02d}"
    return None

def _t_rec_from_exam_date(exam_date: Optional[str], t_rec_min: float) -> float:
    if not exam_date:
        return float(t_rec_min)
    try:
        ex_dt = datetime.fromisoformat(exam_date + "T00:00:00+00:00")
        now = datetime.now(timezone.utc)
        days = max(0, (now - ex_dt).days)
        # Kernel table: t_rec = max(0.01, days/365)
        return float(max(t_rec_min, round(days / 365.0, 6)))
    except Exception:
        return float(t_rec_min)

def f2_score_for_candidate(n_q: int, N_total_chap: int, alpha: float, t_rec: float, psi_norm: float,
                           selected: List[QCItem], candidate: QCItem, cap: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    base = (float(n_q) / max(1.0, float(N_total_chap))) * (1.0 + float(alpha) / max(1e-9, float(t_rec))) * float(psi_norm)
    prod = 1.0
    sig_dbg = []
    cand_vec = _vec_from_steps(cap, pack_get(candidate.frt, "sections.ari.step_types", []) or [])
    for prev in selected:
        prev_vec = _vec_from_steps(cap, pack_get(prev.frt, "sections.ari.step_types", []) or [])
        sig = _cosine(cand_vec, prev_vec)
        prod *= (1.0 - sig)
        sig_dbg.append({"prev_qc_id": prev.qc_id, "sigma": round(sig, 6)})
    return round(base * prod, 10), sig_dbg


# =============================================================================
# Clustering (anti-singleton) + deterministic merge within chapter
# =============================================================================
def _sig_q(cap: Dict[str, Any], chapter_code: str, ari: AriTrace, triggers: List[str]) -> str:
    # Kernel doc mentions SIG(q)=<A,P,O,X> but here we build a deterministic signature from typed steps + primary_op + chapter.
    # Strictly data-driven content (ARI + CAP) — no chapter logic in core.
    step_types = ari.step_types or ["STEP_STANDARD"]
    trig_core = sorted([t for t in (triggers or []) if t.startswith("ARI:") or t.startswith("KW:")])[:8]
    raw = f"{chapter_code}||{ari.primary_op}||{'/'.join(step_types)}||{'/'.join(trig_core)}"
    return "SIG_" + stable_id(raw)

def _cluster_vector(cap: Dict[str, Any], qi_items: List[QiItem]) -> Dict[str, float]:
    # sum of step-weight vectors (deterministic)
    v = defaultdict(float)
    for qi in qi_items:
        steps = pack_get(qi.ari, "step_types", []) or []
        vv = _vec_from_steps(cap, steps)
        for k, val in vv.items():
            v[k] += float(val)
    return dict(v)

def cluster_qi(cap: Dict[str, Any], qi_items: List[QiItem], min_posable: int, merge_thr: float) -> Tuple[Dict[str, List[QiItem]], List[QiItem]]:
    # initial clusters by (chapter_code, sig_q)
    clusters = defaultdict(list)
    for qi in qi_items:
        clusters[f"{qi.chapter_code}::{qi.sig_q}"].append(qi)

    # split into strong clusters (>=min_posable POSABLE) vs weak
    def posable_count(items: List[QiItem]) -> int:
        return sum(1 for x in items if x.has_rqi and (x.posable or {}).get("is_posable"))

    strong = {}
    weak = {}
    for k, items in clusters.items():
        if posable_count(items) >= min_posable:
            strong[k] = items
        else:
            weak[k] = items

    # deterministic merge of weak clusters into closest strong cluster within same chapter if similarity >= threshold
    # otherwise, they remain orphan candidates.
    strong_by_ch = defaultdict(list)
    for k, items in strong.items():
        ch = k.split("::", 1)[0]
        strong_by_ch[ch].append((k, items))

    # stable order
    for ch in list(strong_by_ch.keys()):
        strong_by_ch[ch] = sorted(strong_by_ch[ch], key=lambda x: x[0])

    orphan_candidates: List[QiItem] = []

    for wk_key in sorted(weak.keys()):
        wk_items = weak[wk_key]
        ch = wk_key.split("::", 1)[0]
        wk_pos = posable_count(wk_items)

        if wk_pos == 0:
            # non-posable does not affect coverage; keep in orphan_candidates for trace
            orphan_candidates.extend(wk_items)
            continue

        best = None
        best_sim = -1.0
        wk_vec = _cluster_vector(cap, wk_items)

        for sk_key, sk_items in strong_by_ch.get(ch, []):
            sk_vec = _cluster_vector(cap, sk_items)
            sim = _cosine(wk_vec, sk_vec)
            # deterministic tie-break by key
            if (sim > best_sim) or (abs(sim - best_sim) < 1e-12 and best and sk_key < best[0]):
                best_sim = sim
                best = (sk_key, sk_items)

        if best and best_sim >= merge_thr:
            # merge into that strong cluster
            strong[best[0]].extend(wk_items)
        else:
            orphan_candidates.extend(wk_items)

    # re-evaluate strong clusters (they may have grown)
    strong_final = {}
    for k, items in strong.items():
        strong_final[k] = sorted(items, key=lambda x: x.qi_id)

    return strong_final, orphan_candidates


# =============================================================================
# IA2 Judge (bool checks minimal, Kernel aligned)
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

    # anti-singleton for POSABLE cluster (Kernel)
    min_pos = int(pack_get(cap, "clustering.min_posable_per_cluster", 2))
    if qc.posable_in_cluster < min_pos:
        issues.append({"check": "CHK_ANTI_SINGLETON", "ok": False})

    # FRT required sections
    required = pack_get(cap, "ari_config.frt_schema_required_sections", []) or []
    frt_sections = (qc.frt or {}).get("sections", {}) or {}
    for k in required:
        if k not in frt_sections:
            issues.append({"check": f"CHK_FRT_SECTION_{k}", "ok": False})

    # F2 terms visible in audit (we keep them in QCItem)
    if qc.N_total_chap <= 0:
        issues.append({"check": "CHK_F2_NTOTAL_CHAP", "ok": False})

    return (len(issues) == 0), issues


# =============================================================================
# Pipeline end-to-end + saturation auto
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

    # CAP STUB
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

def run_once(cap: Dict[str, Any], pairs: List[Dict[str, Any]], volume: int, subject: str, level: str) -> Dict[str, Any]:
    chapters = cap_chapters(cap, subject, level)

    # deterministic slice
    pairs2 = pairs[:max(0, min(volume, len(pairs)))]
    quarantine: List[QuarantineEntry] = []
    qi_items: List[QiItem] = []
    pair_meta: List[Dict[str, Any]] = []

    epsilon = float(pack_get(cap, "f_params.epsilon", 0.1))
    alpha = float(pack_get(cap, "f_params.alpha", 1.0))
    t_rec_min = float(pack_get(cap, "f_params.t_rec_min", 0.01))

    for pair in pairs2:
        pair_id = pair.get("pair_id", "")
        su_txt, co_txt, meta, qent = _extract_pair_text(pair, cap)
        if qent:
            quarantine.append(qent)
            pair_meta.append({"pair_id": pair_id, "meta": meta, "quarantine": asdict(qent)})
            continue

        qs, q_audit = split_into_q_segments(su_txt, cap)
        ans, a_audit = split_into_a_segments(co_txt, cap)

        link, al_audit = align_q_to_a(qs, ans, cap)

        # best effort exam_date from refs
        exam_date = _parse_exam_date(pair.get("sujet_ref","") or "", cap) or _parse_exam_date(pair.get("corrige_ref","") or "", cap)

        pair_meta.append({"pair_id": pair_id, "meta": meta, "atom_q": q_audit, "atom_a": a_audit, "align": al_audit})

        atom_low = (len(qs) < 4)
        for i, qtxt in enumerate(qs):
            j = link[i] if i < len(link) else None
            rtxt = ans[j] if (j is not None and j < len(ans)) else ""

            ari = extract_ari(qtxt, rtxt, cap)
            ch_code, ch_label, ch_sc = map_to_chapter(qtxt, chapters)
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

    # POSABLE set
    posable_qi = [q for q in qi_items if q.has_rqi and (q.posable or {}).get("is_posable")]

    # clustering
    min_posable = int(pack_get(cap, "clustering.min_posable_per_cluster", 2))
    merge_thr = float(pack_get(cap, "clustering.merge_similarity_threshold", 0.70))
    clusters, orphan_candidates = cluster_qi(cap, qi_items, min_posable=min_posable, merge_thr=merge_thr)

    # Build QC candidates from clusters (only those meeting anti-singleton on POSABLE)
    qc_candidates: List[QCItem] = []
    qi_to_qc: Dict[str, str] = {}

    # precompute N_total per chapter (Kernel: N_total[c] is per chapter, POSABLE only)
    posable_by_ch = defaultdict(list)
    for qi in posable_qi:
        posable_by_ch[qi.chapter_code].append(qi)

    N_total_by_ch = {ch: len(items) for ch, items in posable_by_ch.items()}

    # chapter label map
    ch_info = {c.get("code"): c for c in chapters}

    for cl_key in sorted(clusters.keys()):
        items = clusters[cl_key]
        # count posable inside
        pos_count = sum(1 for x in items if x.has_rqi and (x.posable or {}).get("is_posable"))
        if pos_count < min_posable:
            # do not generate QC candidate (anti-singleton)
            continue

        # representative = max confidence_global
        rep = max(items, key=lambda z: float((z.ari or {}).get("confidence_global", 0.0)))
        rep_ari = rep.ari or {}
        primary_op = str(rep_ari.get("primary_op", "OP_STANDARD"))
        qc_text, qc_label = build_qc_text(primary_op, cap)
        ch_code = rep.chapter_code
        ch_label = ch_info.get(ch_code, {}).get("label", ch_code)

        # triggers: top occurrences
        trig_counts = Counter()
        all_ops = set()
        for it in items:
            trig_counts.update(it.triggers or [])
            all_ops.update((it.ari or {}).get("all_ops", []) or [])
        qc_triggers = [k for k, _ in trig_counts.most_common(8)]
        qi_ids = [x.qi_id for x in items]

        # FRT from rep
        frt = generate_frt(primary_op, qc_triggers, AriTrace(
            primary_op=str(rep_ari.get("primary_op","OP_STANDARD")),
            ops=[AriOpEvidence(**o) for o in (rep_ari.get("ops") or [])],
            all_ops=list(rep_ari.get("all_ops") or []),
            confidence_global=float(rep_ari.get("confidence_global", 0.55)),
            step_types=list(rep_ari.get("step_types") or ["STEP_STANDARD"])
        ), cap)

        # F1 inputs
        delta_c = float(ch_info.get(ch_code, {}).get("delta_c", 1.0))
        T_steps = _T_steps_from_ari(cap, AriTrace(
            primary_op=str(rep_ari.get("primary_op","OP_STANDARD")),
            ops=[AriOpEvidence(**o) for o in (rep_ari.get("ops") or [])],
            all_ops=list(rep_ari.get("all_ops") or []),
            confidence_global=float(rep_ari.get("confidence_global", 0.55)),
            step_types=list(rep_ari.get("step_types") or ["STEP_STANDARD"])
        ))
        sum_Tj, psi_brut = f1_psi(delta_c, epsilon, T_steps)

        # t_rec (Kernel: based on last occurrence; here approximated from latest exam_date found in cluster)
        dates = [x.source_exam_date for x in items if x.source_exam_date]
        exam_date = max(dates) if dates else None
        t_rec = _t_rec_from_exam_date(exam_date, t_rec_min)

        qc_id = "QC_" + stable_id(ch_code, primary_op, qc_text, ",".join(sorted(qi_ids))[:80])

        evidence_min = {
            "pair_ids": sorted({x.pair_id for x in items})[:5],
            "source_id": rep.source_id,
            "sujet_refs": sorted({x.sujet_ref for x in items})[:5],
            "corrige_refs": sorted({x.corrige_ref for x in items})[:5],
            "qi_ids_sample": qi_ids[:6]
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
            psi_norm=0.0,  # set after normalization
            n_q=pos_count,
            N_total_chap=int(N_total_by_ch.get(ch_code, 0)),
            alpha=alpha,
            t_rec=float(t_rec),
            sigma_terms=[],
            score_f2=0.0,
            evidence_min=evidence_min
        ))

    # normalize psi intra-chapter (Kernel F1-BOOL-3)
    by_ch = defaultdict(list)
    for qc in qc_candidates:
        by_ch[qc.chapter_code].append(qc)

    for ch, qcs in by_ch.items():
        mx = max((q.psi_brut for q in qcs), default=0.0)
        for q in qcs:
            q.psi_norm = round((q.psi_brut / mx) if mx > 0 else 0.0, 10)

    # Selection (Kernel: set-cover deterministic)
    selected_by_ch = {}
    coverage_map = {}
    safety_stop = False
    safety_notes = []

    for ch in sorted(posable_by_ch.keys()):
        uncovered = set([q.qi_id for q in posable_by_ch[ch]])
        candidates = [q for q in qc_candidates if q.chapter_code == ch and q.posable_in_cluster >= min_posable]
        candidates = sorted(candidates, key=lambda x: x.qc_id)  # deterministic base

        selected: List[QCItem] = []
        # Greedy set cover with F2 score ordering (deterministic)
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
                # choose by gain, then score, then qc_id
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

            # remove selected from candidates
            candidates = [c for c in candidates if c.qc_id != best.qc_id]

        selected_by_ch[ch] = selected
        coverage_map[ch] = {
            "N_total_chap": int(N_total_by_ch.get(ch, 0)),
            "covered": int(N_total_by_ch.get(ch, 0) - len(uncovered)),
            "uncovered": sorted(list(uncovered))[:50],
            "coverage": round((float(N_total_by_ch.get(ch, 0) - len(uncovered)) / max(1, float(N_total_by_ch.get(ch, 0)))), 6)
        }

    # Map Qi -> QC (coverage proof): assign each posable Qi to the first selected QC that covers it (deterministic order)
    qc_by_id = {q.qc_id: q for q in qc_candidates}
    selected_all = []
    for ch in sorted(selected_by_ch.keys()):
        selected_all.extend(selected_by_ch[ch])

    # QA IA2 checks on selected only (Kernel: IA2 PASS)
    ia2_ok = True
    ia2_issues = []
    for qc in selected_all:
        ok, issues = ia2_checks(cap, qc)
        if not ok:
            ia2_ok = False
            ia2_issues.extend([{"qc_id": qc.qc_id, **x} for x in issues])

    # assignment
    for qi in qi_items:
        qi.qc_id = None
        qi.is_orphan = False

    for ch in sorted(selected_by_ch.keys()):
        for qc in selected_by_ch[ch]:
            for qid in qc.qi_ids:
                # assign only if not assigned
                if qid not in qi_to_qc:
                    qi_to_qc[qid] = qc.qc_id

    orphans_posable = 0
    for qi in qi_items:
        if qi.qi_id in qi_to_qc:
            qi.qc_id = qi_to_qc[qi.qi_id]
            qi.is_orphan = False
        else:
            qi.qc_id = None
            # orphan relevant only if POSABLE
            if qi.has_rqi and (qi.posable or {}).get("is_posable"):
                qi.is_orphan = True
                orphans_posable += 1

    # chapter-poor exception (Kernel mentions) is only used implicitly if CAP allows and there is a single cluster available;
    # here we keep strict anti-singleton; safety-stop will reflect if it blocks coverage.

    # global coverage (mean over chapters)
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
        "ia2_issues": ia2_issues[:100],
        "safety_stop": bool(safety_stop),
        "safety_notes": safety_notes[:50],
        "quarantine_count": len(quarantine),
    }

    evidence_pack = {
        "app_version": APP_VERSION,
        "timestamp": _utc_ts(),
        "cap_id": cap.get("pack_id"),
        "cap_sig_sha256": cap.get("_pack_sig_sha256"),
        "subject": subject,
        "level": level,
        "source_manifest": {},
        "audit": audit,
        "pair_meta_sample": pair_meta[:50],
        "quarantine_sample": [asdict(q) for q in quarantine[:100]],
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
                    "score_f2": qc.score_f2
                },
                "evidence_min": qc.evidence_min
            } for qc in selected_all
        ]
    }

    return {
        "qi_items": [asdict(x) for x in qi_items],
        "qc_candidates": [asdict(x) for x in qc_candidates],
        "qc_selected": [asdict(x) for x in selected_all],
        "selected_by_ch": {k: [asdict(x) for x in v] for k, v in selected_by_ch.items()},
        "quarantine": [asdict(x) for x in quarantine],
        "audit": audit,
        "evidence_pack": evidence_pack
    }

def run_saturation_auto(cap: Dict[str, Any], pairs: List[Dict[str, Any]], subject: str, level: str) -> Dict[str, Any]:
    sat = pack_get(cap, "saturation", {}) or {}
    vol_start = int(sat.get("vol_start", 6))
    vol_max = int(sat.get("vol_max", 30))
    step = int(sat.get("step", 6))
    max_iters = int(sat.get("safety_stop_max_iters", 6))

    iterations = []
    prev_qc_set = None
    final = None

    it = 0
    for vol in range(vol_start, min(vol_max, len(pairs)) + 1, max(1, step)):
        it += 1
        out = run_once(cap, pairs, vol, subject, level)
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

        # stop if coverage achieved and IA2 ok and zero orphans (Kernel sufficient)
        if audit["orphans_posable"] == 0 and audit["ia2_ok"] and (not audit["safety_stop"]):
            break

        # stop if no new QC and no progress
        if bool(sat.get("stop_if_new_qc_zero", True)) and new_qc == 0:
            break

        if it >= max_iters:
            break

    assert final is not None
    final["saturation_iterations"] = iterations
    return final


# =============================================================================
# UI — Action unique: choisir + valider le pays → auto-run
# =============================================================================
def ss_init():
    defaults = {
        "cap": None,
        "cap_valid": False,
        "cap_errors": [],
        "source_pairs": [],
        "source_manifest": {},
        "run_done": False,
        "run_output": None,
        "country_code": None,
        "subject": "MATH",
        "level": "TERMINALE",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def main():
    st.set_page_config(page_title=APP_NAME, layout="wide")
    ss_init()

    st.title(APP_NAME)
    st.caption("Kernel V10.6.3 • TEST = PROD (logique) • CAS 1 ONLY • Déterminisme • Zéro hardcode (Pack-Driven) • Mode déconnecté")

    # --- Single action form ---
    with st.form("activate_country_form", clear_on_submit=False):
        c1, c2 = st.columns([2, 3])
        with c1:
            country_label = st.selectbox("Choisir le pays (action unique)", ["France", "Côte d’Ivoire"])
            country_code = "FR" if country_label == "France" else "CI"
        with c2:
            uploaded_cap = st.file_uploader("CAP JSON (optionnel) — si non fourni, CAP scellé embedded utilisé", type=["json"])
        submitted = st.form_submit_button("VALIDER & LANCER (AUTO)", type="primary", use_container_width=True)

    if submitted:
        cap = load_cap(country_code, uploaded_cap.read() if uploaded_cap else None)
        ok, errs = validate_cap(cap)

        st.session_state.cap = cap
        st.session_state.cap_valid = ok
        st.session_state.cap_errors = errs
        st.session_state.country_code = country_code

        if not ok:
            st.error("CAP invalide — exécution bloquée.")
        else:
            # load sources and auto-run (no further action)
            pairs, manifest = load_sources(cap)
            st.session_state.source_pairs = pairs
            st.session_state.source_manifest = manifest

            with st.spinner("Pipeline automatique (déconnecté) en cours..."):
                out = run_saturation_auto(
                    cap=cap,
                    pairs=pairs,
                    subject=st.session_state.subject,
                    level=st.session_state.level
                )
            out["evidence_pack"]["source_manifest"] = manifest
            st.session_state.run_output = out
            st.session_state.run_done = True

    # --- Status panel ---
    cap = st.session_state.cap
    if cap is None:
        st.info("Choisir un pays puis cliquer sur ‘VALIDER & LANCER (AUTO)’ (action unique).")
        return

    if not st.session_state.cap_valid:
        st.error("CAP non conforme :")
        st.write(st.session_state.cap_errors)
        st.json(cap)
        return

    # Show CAP header
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pays", f"{cap.get('country_name')} ({cap.get('country_code')})")
    c2.metric("CAP status", cap.get("status", ""))
    c3.metric("CAP id", cap.get("pack_id", ""))
    c4.metric("CAP sig", str(cap.get("_pack_sig_sha256", ""))[:12] + "...")

    manifest = st.session_state.source_manifest or {}
    st.caption(f"Sources: mode={manifest.get('mode')} • local_dir={manifest.get('local_dir')} • pairs_total={manifest.get('pairs_total')}")

    if not st.session_state.run_done or not st.session_state.run_output:
        st.warning("Le pipeline n’a pas encore été exécuté dans cette session.")
        return

    out = st.session_state.run_output
    audit = out.get("audit", {})
    iters = out.get("saturation_iterations", [])

    # KPI row
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Pairs utilisés", audit.get("pairs_used", 0))
    k2.metric("Qi total", audit.get("qi_total", 0))
    k3.metric("Qi POSABLE", audit.get("qi_posable_total", 0))
    k4.metric("QC candidates", audit.get("qc_candidates_total", 0))
    k5.metric("QC sélectionnées", audit.get("qc_selected_total", 0))
    k6.metric("Orphelins POSABLE", audit.get("orphans_posable", 0))
    k7.metric("Coverage (moyenne chap.)", f"{float(audit.get('coverage_global_mean', 0.0)):.0%}")

    # Verdict kernel (strict)
    verdict_ok = (audit.get("orphans_posable", 1) == 0) and bool(audit.get("ia2_ok", False)) and (not bool(audit.get("safety_stop", False)))
    st.subheader("Verdict (Kernel)")
    st.write({
        "PASS": verdict_ok,
        "ia2_ok": audit.get("ia2_ok"),
        "orphans_posable": audit.get("orphans_posable"),
        "safety_stop": audit.get("safety_stop"),
        "safety_notes": audit.get("safety_notes", [])[:10],
        "ia2_issues": audit.get("ia2_issues", [])[:10],
    })

    if iters:
        st.subheader("Saturation (auto)")
        st.dataframe(iters, use_container_width=True, hide_index=True)

    # Main deliverable: QC / FRT / ARI / TRIGGERS grouped by chapter with Qi associated
    st.subheader("QC / FRT / ARI / TRIGGERS par chapitre (avec Qi associées)")

    qi_items = out.get("qi_items", []) or []
    qi_by_id = {q["qi_id"]: q for q in qi_items}
    selected_by_ch = out.get("selected_by_ch", {}) or {}
    cov_by_ch = (audit.get("coverage_by_chapter", {}) or {})

    for ch_code in sorted(selected_by_ch.keys()):
        cov = cov_by_ch.get(ch_code, {})
        cov_pct = float(cov.get("coverage", 0.0) or 0.0)
        uncovered = cov.get("uncovered", []) or []

        qcs = selected_by_ch.get(ch_code, [])
        ch_label = qcs[0].get("chapter_label", ch_code) if qcs else ch_code

        with st.expander(f"{ch_label} [{ch_code}] — Coverage={cov_pct:.0%} — N_total={cov.get('N_total_chap',0)} — Uncovered={len(uncovered)}", expanded=False):
            if uncovered:
                st.error(f"Uncovered POSABLE (sample): {uncovered[:12]}")

            for qc in qcs:
                st.markdown(f"### {qc['qc']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("posable_in_cluster", qc.get("posable_in_cluster", 0))
                c2.metric("cluster_size", qc.get("cluster_size", 0))
                c3.metric("Ψ_norm", qc.get("psi_norm", 0.0))
                c4.metric("Score(q) F2", qc.get("score_f2", 0.0))

                st.markdown("**Triggers**")
                st.write(qc.get("triggers", []))

                st.markdown("**ARI (rep) + steps typed**")
                st.json(pack_get(qc, "frt.sections.ari", {}) or {})

                st.markdown("**FRT**")
                st.json(qc.get("frt", {}))

                st.markdown("**EvidencePack minimal (références)**")
                st.json(qc.get("evidence_min", {}))

                st.markdown("**Termes F1/F2 (audit)**")
                st.json({
                    "delta_c": qc.get("delta_c"),
                    "epsilon": qc.get("epsilon"),
                    "T_steps": qc.get("T_steps"),
                    "sum_Tj": qc.get("sum_Tj"),
                    "psi_brut": qc.get("psi_brut"),
                    "psi_norm": qc.get("psi_norm"),
                    "n_q": qc.get("n_q"),
                    "N_total_chap": qc.get("N_total_chap"),
                    "alpha": qc.get("alpha"),
                    "t_rec": qc.get("t_rec"),
                    "sigma_terms": qc.get("sigma_terms"),
                    "score_f2": qc.get("score_f2"),
                })

                st.markdown("**Qi associées (avec RQi)**")
                for qid in qc.get("qi_ids", [])[:24]:
                    q = qi_by_id.get(qid)
                    if not q:
                        continue
                    pos = (q.get("posable") or {}).get("is_posable", False)
                    tag = "POSABLE" if pos else "UNPOSABLE"
                    with st.expander(f"{qid} • {tag} • {q.get('chapter_code','')}", expanded=False):
                        st.markdown("Qi")
                        st.text((q.get("qi","") or "")[:1400])
                        st.markdown("RQi")
                        st.text((q.get("rqi","") or "")[:1400])
                        st.markdown("Triggers")
                        st.write(q.get("triggers", []))
                        st.markdown("Posable decision")
                        st.json(q.get("posable", {}))
                        st.markdown("Source refs")
                        st.json({
                            "pair_id": q.get("pair_id"),
                            "source_id": q.get("source_id"),
                            "sujet_ref": q.get("sujet_ref"),
                            "corrige_ref": q.get("corrige_ref"),
                            "exam_date": q.get("source_exam_date"),
                        })

    # Exports (proof)
    st.subheader("Exports (preuves)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("evidence_pack.json", safe_json_dumps(out.get("evidence_pack", {})), "evidence_pack.json", use_container_width=True)
    with c2:
        st.download_button("qc_selected.json", safe_json_dumps(out.get("qc_selected", [])), "qc_selected.json", use_container_width=True)
    with c3:
        st.download_button("qi_items.json", safe_json_dumps(out.get("qi_items", [])), "qi_items.json", use_container_width=True)

    # Quarantine visibility
    if out.get("quarantine"):
        st.subheader("Quarantine (sample)")
        st.dataframe(out["quarantine"][:120], use_container_width=True, hide_index=True)


# =============================================================================
# ENTRYPOINT
# =============================================================================
main()
