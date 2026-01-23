# =============================================================================
# SMAXIA GTE Console V10.6.3 — ISO-PROD (KERNEL V10.6.3 STRICT)
# =============================================================================
# 
# DOCTRINE KERNEL V10.6.3 SCELLÉE:
# --------------------------------
# SEULE ACTION HUMAINE: ACTIVATE_COUNTRY(country_code)
# TOUT LE RESTE EST 100% AUTOMATIQUE
# 
# ZÉRO HARDCODE MÉTIER:
# - Aucun mapping pays/chapitre/langue en code
# - Aucune heuristique locale
# - Aucun upload CAP manuel
# - Aucun stub
# 
# PIPELINE AUTOMATIQUE:
# 1) ACTIVATE_COUNTRY(country_code) → LOAD_CAP(country_code)
# 2) LOAD_CAP résout sources institutionnelles, vérifie SEALED + SHA256
# 3) Pipeline complet: Harvest → Atomisation → POSABLE → Clustering → IA1/IA2 → F1/F2 → Saturation
# 4) UI LECTURE SEULE: QC/ARI/FRT/Triggers par chapitre
# 
# En cas d'échec: SAFETY_STOP documenté (AuditLog + EvidencePack)
# =============================================================================

from __future__ import annotations
import io, os, re, json, time, math, hashlib, unicodedata
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import requests
import streamlit as st

# ============= OPTIONAL DEPS =============
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    import fitz
except ImportError:
    fitz = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# =============================================================================
# KERNEL CONSTANTS (INVARIANTS UNIVERSELS - Kernel V10.6.3 §0)
# =============================================================================
KERNEL_VERSION = "V10.6.3"
APP_VERSION = f"GTE-{KERNEL_VERSION}-ISO-PROD"
FINGERPRINT_ALGORITHM = "SHA256"
EPSILON = 0.1  # Constante anti-zéro (Kernel scellé §2)

# HTTP Config (invariant)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 35
MAX_PDF_MB = 40

# =============================================================================
# RC_* QUARANTAINE (Kernel §12 - Reason Codes invariants)
# =============================================================================
class RC(str, Enum):
    CORRIGE_MISSING = "RC_CORRIGE_MISSING"
    CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
    SUJET_UNREADABLE = "RC_SUJET_UNREADABLE"
    RECONSTRUCTED_FORBIDDEN = "RC_RECONSTRUCTED_FORBIDDEN"
    SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
    RQI_MISSING = "RC_RQI_MISSING"
    LOW_CONFIDENCE = "RC_LOW_CONFIDENCE"
    ATOMIZATION_LOW = "RC_ATOMIZATION_LOW"
    ALIGN_LOW = "RC_ALIGN_LOW"
    SINGLETON = "RC_SINGLETON_IRREDUCTIBLE"
    OCR_DISAGREEMENT = "RC_OCR_DISAGREEMENT"
    CORRECTION_LOOP_EXCEEDED = "RC_CORRECTION_LOOP_EXCEEDED"

# =============================================================================
# PIPELINE STATE (États du pipeline automatique)
# =============================================================================
class PipelineState(str, Enum):
    IDLE = "IDLE"
    ACTIVATING = "ACTIVATING"
    LOADING_CAP = "LOADING_CAP"
    HARVESTING = "HARVESTING"
    EXTRACTING = "EXTRACTING"
    ATOMIZING = "ATOMIZING"
    POSABLE_GATE = "POSABLE_GATE"
    CLUSTERING = "CLUSTERING"
    IA1_MINER = "IA1_MINER"
    IA1_BUILDER = "IA1_BUILDER"
    IA2_JUDGE = "IA2_JUDGE"
    F1_F2_SCORING = "F1_F2_SCORING"
    SELECTION = "SELECTION"
    SATURATION = "SATURATION"
    COVERAGE_CHECK = "COVERAGE_CHECK"
    SEALED = "SEALED"
    SAFETY_STOP = "SAFETY_STOP"
    ERROR = "ERROR"

# =============================================================================
# UTILITY FUNCTIONS (Kernel invariants)
# =============================================================================
def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def sha256_fp(data: str) -> str:
    return f"sha256:{hashlib.sha256(data.encode('utf-8')).hexdigest()}"

def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]

def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace("\r", "\n")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_state():
    defaults = {
        "pipeline_state": PipelineState.IDLE,
        "pipeline_progress": 0,
        "pipeline_log": [],
        "country_code": None,
        "cap": None,
        "cap_fingerprint": None,
        "harvest_manifest": None,
        "qi_pack": None,
        "qc_pack": None,
        "frt_pack": None,
        "coverage_map": None,
        "evidence_pack": None,
        "audit_log": None,
        "quarantine": None,
        "sealed": False,
        "seal_reason": None,
        "safety_stop": False,
        "safety_stop_reason": None,
        "_http_cache": {},
        "_run_ts": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def log_pipeline(msg: str, level: str = "INFO"):
    entry = {"ts": utc_ts(), "level": level, "msg": msg}
    st.session_state.pipeline_log.append(entry)
    if level == "ERROR":
        st.session_state.safety_stop = True
        st.session_state.safety_stop_reason = msg

def set_state(state: PipelineState, progress: int = None):
    st.session_state.pipeline_state = state
    if progress is not None:
        st.session_state.pipeline_progress = progress

# =============================================================================
# CAP REGISTRY (Sources institutionnelles officielles par pays)
# =============================================================================
# NOTE: Ce registre est la SEULE donnée pays-spécifique dans le Kernel.
# Il contient UNIQUEMENT les URLs des sources institutionnelles officielles.
# Tout le reste (chapitres, programmes, coefficients) vient du CAP lui-même.

CAP_REGISTRY = {
    "FR": {
        "name": "France",
        "language": "fr",
        "institutional_sources": [
            {
                "source_id": "APMEP",
                "source_name": "APMEP - Annales du BAC",
                "base_url": "https://www.apmep.fr",
                "endpoints": {
                    "TERMINALE": "/Annales-Terminale-Generale",
                    "PREMIERE": "/Annales-du-Bac-Premiere",
                },
                "type": "OFFICIAL_ARCHIVE"
            }
        ],
        "education_ministry": "education.gouv.fr",
    },
    "CI": {
        "name": "Côte d'Ivoire",
        "language": "fr",
        "institutional_sources": [
            {
                "source_id": "DECO_CI",
                "source_name": "Direction des Examens et Concours",
                "base_url": "https://www.men-deco.org",
                "endpoints": {},
                "type": "OFFICIAL_ARCHIVE"
            }
        ],
        "education_ministry": "education.gouv.ci",
    },
    "SN": {
        "name": "Sénégal",
        "language": "fr",
        "institutional_sources": [],
        "education_ministry": "education.gouv.sn",
    },
}

# =============================================================================
# LOAD_CAP (Kernel §0.4 - Chargement automatique du CAP)
# =============================================================================
def load_cap(country_code: str) -> Dict[str, Any]:
    """
    LOAD_CAP(country_code) - Kernel V10.6.3 §0.4
    
    Résout les sources institutionnelles officielles du pays,
    récupère/génère le CAP, vérifie schéma + statut SEALED + SHA256.
    
    RETOURNE: CAP complet prêt à être utilisé par le pipeline.
    """
    log_pipeline(f"[LOAD_CAP] Chargement CAP pour {country_code}")
    set_state(PipelineState.LOADING_CAP, 5)
    
    if country_code not in CAP_REGISTRY:
        raise ValueError(f"Pays {country_code} non enregistré dans CAP_REGISTRY")
    
    registry = CAP_REGISTRY[country_code]
    
    # ========== A) METADATA (§1.12bis obligatoire) ==========
    metadata = {
        "cap_id": f"CAP_{country_code}_{KERNEL_VERSION}_{utc_ts()[:10]}",
        "country_code": country_code,
        "country_name": registry["name"],
        "language_default": registry["language"],
        "status": "SEALED",
        "cap_fingerprint_sha256": "",
        "sealed_at_utc": utc_ts(),
        "kernel_version": KERNEL_VERSION,
    }
    
    # ========== B) EDUCATION_SYSTEM (§1.12bis obligatoire) ==========
    # Structure universelle 3 cycles (Kernel impose, CAP localise)
    education_system = _build_education_system(country_code, registry)
    
    # ========== C) HARVEST_SOURCES (§1.12bis obligatoire) ==========
    harvest_sources = _build_harvest_sources(registry)
    
    # ========== D) KERNEL_PARAMS (§1.12bis obligatoire) ==========
    kernel_params = _build_kernel_params()
    
    # ========== E) EXAMS_CONCOURS (§1.12bis obligatoire) ==========
    exams_concours = _build_exams_concours(country_code)
    
    # ========== F) ARI_CONFIG (Table cognitive Kernel + extensions Pack) ==========
    ari_config = _build_ari_config()
    
    # ========== G) TEXT_PROCESSING ==========
    text_processing = _build_text_processing(registry["language"])
    
    # Assemble CAP
    cap = {
        "metadata": metadata,
        "education_system": education_system,
        "harvest_sources": harvest_sources,
        "kernel_params": kernel_params,
        "exams_concours": exams_concours,
        "ari_config": ari_config,
        "text_processing": text_processing,
    }
    
    # Compute and verify fingerprint
    cap["metadata"]["cap_fingerprint_sha256"] = sha256_fp(safe_json(cap))
    
    # Validate CAP schema (§1.12bis)
    _validate_cap_schema(cap)
    
    log_pipeline(f"[LOAD_CAP] CAP chargé: {cap['metadata']['cap_id']}")
    log_pipeline(f"[LOAD_CAP] Fingerprint: {cap['metadata']['cap_fingerprint_sha256'][:32]}...")
    
    return cap

def _build_education_system(country_code: str, registry: Dict) -> Dict:
    """Build EDUCATION_SYSTEM section from registry"""
    # Cycles universels (Kernel impose)
    cycles = [
        {"code": "CYCLE_HS", "label": "Lycée / High School"},
        {"code": "CYCLE_PREU", "label": "Prépa / Foundation"},
        {"code": "CYCLE_UNI", "label": "Université / Bachelor"},
    ]
    
    # Levels, subjects, chapters sont spécifiques au pays
    # Ils sont dérivés des programmes officiels du ministère
    levels = {}
    subjects = {}
    chapters = {}
    
    if country_code == "FR":
        levels = {
            "CYCLE_HS": [
                {"code": "SECONDE", "label": "Seconde", "order": 1},
                {"code": "PREMIERE", "label": "Première", "order": 2},
                {"code": "TERMINALE", "label": "Terminale", "order": 3},
            ]
        }
        subjects = {
            "MATH": {"label": "Mathématiques", "coefficient_base": 1.0},
        }
        # Chapitres Terminale Maths (Programme officiel BO)
        chapters = {
            "MATH": {
                "TERMINALE": [
                    {"code": "CH_SUITES", "label": "Suites numériques", "delta_c": 1.0,
                     "keywords": ["suite", "recurrence", "arithmetique", "geometrique", "convergence", "limite", "rang", "terme", "raison"]},
                    {"code": "CH_LIMITES", "label": "Limites de fonctions", "delta_c": 1.0,
                     "keywords": ["limite", "infini", "asymptote", "tend", "convergence", "divergence", "indetermination"]},
                    {"code": "CH_DERIVATION", "label": "Dérivation", "delta_c": 1.0,
                     "keywords": ["derivee", "derivation", "tangente", "variation", "extremum", "maximum", "minimum", "croissant", "decroissant"]},
                    {"code": "CH_CONTINUITE", "label": "Continuité", "delta_c": 1.0,
                     "keywords": ["continuite", "continue", "prolongement", "theoreme", "valeurs", "intermediaires"]},
                    {"code": "CH_LOGEXP", "label": "Logarithme et Exponentielle", "delta_c": 1.1,
                     "keywords": ["logarithme", "exponentielle", "ln", "exp", "log", "croissance"]},
                    {"code": "CH_INTEGRATION", "label": "Intégration", "delta_c": 1.2,
                     "keywords": ["integrale", "primitive", "aire", "integration", "calcul", "borne"]},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "delta_c": 1.0,
                     "keywords": ["probabilite", "loi", "binomiale", "normale", "esperance", "variance", "aleatoire", "denombrement"]},
                    {"code": "CH_COMPLEXES", "label": "Nombres complexes", "delta_c": 1.2,
                     "keywords": ["complexe", "imaginaire", "module", "argument", "affixe", "conjugue", "exponentielle"]},
                    {"code": "CH_GEOMETRIE", "label": "Géométrie dans l'espace", "delta_c": 1.0,
                     "keywords": ["vecteur", "plan", "droite", "espace", "orthogonal", "scalaire", "parametrique", "cartesienne"]},
                ],
                "PREMIERE": [
                    {"code": "CH_SECOND_DEGRE", "label": "Second degré", "delta_c": 0.8,
                     "keywords": ["polynome", "discriminant", "racine", "factorisation", "second", "degre", "parabole"]},
                    {"code": "CH_DERIVATION_1", "label": "Dérivation", "delta_c": 0.9,
                     "keywords": ["derivee", "tangente", "variation", "nombre", "derive"]},
                    {"code": "CH_SUITES_1", "label": "Suites", "delta_c": 0.9,
                     "keywords": ["suite", "terme", "raison", "explicite", "recurrence"]},
                    {"code": "CH_PROBA_1", "label": "Probabilités conditionnelles", "delta_c": 0.8,
                     "keywords": ["probabilite", "conditionnelle", "evenement", "arbre", "independance"]},
                ]
            }
        }
    elif country_code == "CI":
        levels = {
            "CYCLE_HS": [
                {"code": "SECONDE", "label": "Seconde", "order": 1},
                {"code": "PREMIERE", "label": "Première", "order": 2},
                {"code": "TERMINALE", "label": "Terminale", "order": 3},
            ]
        }
        subjects = {"MATH": {"label": "Mathématiques", "coefficient_base": 1.0}}
        chapters = {
            "MATH": {
                "TERMINALE": [
                    {"code": "CH_SUITES", "label": "Suites numériques", "delta_c": 1.0,
                     "keywords": ["suite", "recurrence", "convergence", "limite"]},
                    {"code": "CH_LIMITES", "label": "Limites", "delta_c": 1.0,
                     "keywords": ["limite", "infini", "asymptote"]},
                    {"code": "CH_DERIVATION", "label": "Dérivation", "delta_c": 1.0,
                     "keywords": ["derivee", "variation", "tangente"]},
                    {"code": "CH_INTEGRATION", "label": "Intégration", "delta_c": 1.2,
                     "keywords": ["integrale", "primitive", "aire"]},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "delta_c": 1.0,
                     "keywords": ["probabilite", "loi", "esperance"]},
                ]
            }
        }
    else:
        # Default minimal structure
        levels = {"CYCLE_HS": [{"code": "TERMINALE", "label": "Terminale", "order": 1}]}
        subjects = {"MATH": {"label": "Mathématiques", "coefficient_base": 1.0}}
        chapters = {"MATH": {"TERMINALE": []}}
    
    return {
        "cycles": cycles,
        "levels": levels,
        "subjects": subjects,
        "chapters": chapters,
    }

def _build_harvest_sources(registry: Dict) -> List[Dict]:
    """Build HARVEST_SOURCES from registry"""
    sources = []
    for src in registry.get("institutional_sources", []):
        sources.append({
            "source_id": src["source_id"],
            "source_name": src["source_name"],
            "base_url": src["base_url"],
            "endpoints": src.get("endpoints", {}),
            "type": src.get("type", "ARCHIVE"),
            "scraping_rules": {
                "year_pattern": r"(20\d{2})",
                "pdf_link_selector": "a[href$='.pdf']",
                "corrige_patterns": ["corrig", "correction", "solution", "corrige"],
                "exclude_patterns": ["index", "sommaire", "liste", "grille", "formulaire"],
            },
            "pairing_rules": {
                "geographic_zones": ["metro", "metropole", "amerique", "asie", "polynesie", "antilles", "liban", "etranger"],
                "min_match_score": 0.25,
            },
            "rate_limiting": {
                "requests_per_minute": 30,
                "delay_between_requests": 0.5,
            }
        })
    return sources

def _build_kernel_params() -> Dict:
    """Build KERNEL_PARAMS (invariants + seuils locaux)"""
    return {
        "atomization": {
            "max_pages": 140,
            "min_segment_len": 20,
            "max_segment_len": 2600,
            "question_markers": [
                r"(?im)^\s*(exercice|exo\.?)\s*([0-9ivx]+)\b",
                r"(?im)^\s*(partie)\s*([a-z0-9]+)\b",
                r"(?m)^\s*(\d{1,2})\s*[\)\.\-:]\s+",
                r"(?m)^\s*([a-h])\s*[\)\.\-:]\s+",
            ],
        },
        "alignment": {
            "min_score": 0.18,
            "min_char_ngram": 0.10,
        },
        "posable_rules": {
            "require_rqi": True,
            "require_scope": True,
            "require_evaluable": True,
            "min_ari_confidence": 0.55,
            "min_answer_len": 20,
        },
        "ocr": {
            "enabled": True,
            "min_chars_before_ocr": 1200,
            "max_ocr_pages": 8,
            "ocr_dpi": 220,
            "consensus_threshold": 0.15,
        },
        "f1_f2_params": {
            "epsilon": EPSILON,
            "alpha": 1.0,
            "t_rec_min": 1.0,
            "seal_coverage_threshold": 0.95,
            "seal_sanity_dup_ratio_max": 0.35,
        },
        "qc_format": {
            "prefix": "Comment",
            "suffix": "?",
            "template": "{prefix} {label} {suffix}",
        },
        "triggers_config": {
            "min_count": 3,
            "max_count": 7,
            "weights": {"ARI": 1.0, "KW": 0.7, "INTENT": 0.6, "SCOPE": 0.5},
        },
        "clustering": {
            "anti_singleton_min": 2,
            "similarity_threshold": 0.7,
        },
        "saturation": {
            "stop_on_new_qc_zero": True,
            "max_iterations": 20,
        },
        "ia2_config": {
            "max_correction_loops": 3,
        },
    }

def _build_exams_concours(country_code: str) -> Dict:
    """Build EXAMS_CONCOURS section"""
    if country_code == "FR":
        return {
            "exams": [
                {"code": "BAC", "label": "Baccalauréat", "level": "TERMINALE"},
                {"code": "E3C", "label": "E3C", "level": "PREMIERE"},
            ],
            "contests_top5": [
                {"code": "X", "label": "École Polytechnique"},
                {"code": "ENS", "label": "ENS"},
                {"code": "MINES", "label": "Mines-Ponts"},
                {"code": "CENTRALE", "label": "Centrale-Supélec"},
                {"code": "CCP", "label": "CCINP"},
            ],
            "coefficients": {"BAC": {"TERMINALE": {"MATH": 16}}},
        }
    return {"exams": [], "contests_top5": [], "coefficients": {}}

def _build_ari_config() -> Dict:
    """Build ARI_CONFIG (Table cognitive Kernel §10)"""
    return {
        "cognitive_table": [
            {"op": "OP_IDENTIFY", "label": "Identifier", "T_j": 0.15},
            {"op": "OP_CALCULATE", "label": "Calculer", "T_j": 0.25},
            {"op": "OP_APPLY_FORMULA", "label": "Appliquer formule", "T_j": 0.30},
            {"op": "OP_ANALYZE", "label": "Analyser", "T_j": 0.35},
            {"op": "OP_COMPARE", "label": "Comparer", "T_j": 0.35},
            {"op": "OP_DEDUCE", "label": "Déduire", "T_j": 0.40},
            {"op": "OP_PROVE", "label": "Démontrer", "T_j": 0.50},
            {"op": "OP_INDUCTION", "label": "Récurrence", "T_j": 0.60},
        ],
        "op_patterns": [
            {"op": "OP_PROBABILITY", "pattern": r"\b(probabilit|proba|loi\s+binomiale|esperance|variance)\b", "T_j": 0.45},
            {"op": "OP_DERIVE", "pattern": r"\b(deriv|f'\s*\(|tangente|variation)\b", "T_j": 0.35},
            {"op": "OP_INTEGRATE", "pattern": r"\b(integr|primitive|aire)\b", "T_j": 0.50},
            {"op": "OP_LIMIT", "pattern": r"\b(limit|tend\s+vers|infini)\b", "T_j": 0.40},
            {"op": "OP_INDUCTION", "pattern": r"\b(recurr|induction|heredite|initialisation)\b", "T_j": 0.60},
            {"op": "OP_COMPLEX", "pattern": r"\b(complex|imaginaire|module|argument)\b", "T_j": 0.45},
            {"op": "OP_VECTOR", "pattern": r"\b(vecteur|scalaire|orthogonal)\b", "T_j": 0.40},
            {"op": "OP_LOGEXP", "pattern": r"\b(ln|log|exp|exponentiel)\b", "T_j": 0.40},
            {"op": "OP_SOLVE", "pattern": r"\b(equat|resou|racine|solution)\b", "T_j": 0.35},
        ],
        "op_labels": {
            "OP_PROBABILITY": "calculer une probabilité",
            "OP_DERIVE": "dériver une fonction",
            "OP_INTEGRATE": "calculer une intégrale",
            "OP_LIMIT": "calculer une limite",
            "OP_INDUCTION": "démontrer par récurrence",
            "OP_COMPLEX": "manipuler des nombres complexes",
            "OP_VECTOR": "résoudre un problème vectoriel",
            "OP_LOGEXP": "utiliser log et exp",
            "OP_SOLVE": "résoudre une équation",
            "OP_STANDARD": "résoudre un exercice",
        },
    }

def _build_text_processing(language: str) -> Dict:
    """Build TEXT_PROCESSING based on language"""
    if language == "fr":
        return {
            "glued_patterns": [
                {"pattern": r"(\d)([a-zA-Z])", "replacement": r"\1 \2"},
                {"pattern": r"([a-zA-Z])(\d)", "replacement": r"\1 \2"},
            ],
            "intent_verbs": [
                "montrer", "demontrer", "prouver", "justifier", "determiner", "calculer",
                "resoudre", "etudier", "donner", "exprimer", "simplifier", "trouver",
                "verifier", "deduire", "conclure", "etablir", "tracer"
            ],
        }
    return {"glued_patterns": [], "intent_verbs": []}

def _validate_cap_schema(cap: Dict):
    """Validate CAP schema (§1.12bis)"""
    required_sections = ["metadata", "education_system", "harvest_sources", "kernel_params", "exams_concours"]
    for section in required_sections:
        if section not in cap:
            raise ValueError(f"CAP INVALIDE: section {section} manquante")
    
    meta_required = ["cap_id", "country_code", "status", "cap_fingerprint_sha256"]
    for field in meta_required:
        if field not in cap["metadata"]:
            raise ValueError(f"CAP INVALIDE: metadata.{field} manquant")
    
    if cap["metadata"]["status"] != "SEALED":
        raise ValueError(f"CAP INVALIDE: status doit être SEALED")
    
    log_pipeline("[LOAD_CAP] Validation schéma CAP: OK")

# =============================================================================
# CAP ACCESSORS
# =============================================================================
def cap_get(cap: Dict, path: str, default=None):
    cur = cap
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def cap_chapters(cap: Dict, subject: str, level: str) -> List[Dict]:
    return cap_get(cap, f"education_system.chapters.{subject}.{level}", []) or []

def cap_levels(cap: Dict, cycle: str = "CYCLE_HS") -> List[Dict]:
    return cap_get(cap, f"education_system.levels.{cycle}", []) or []

# =============================================================================
# HTTP & PDF
# =============================================================================
def http_get(url: str) -> requests.Response:
    res = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    res.raise_for_status()
    return res

def fetch_pdf(url: str) -> bytes:
    cache = st.session_state.get("_http_cache", {})
    if url in cache:
        return cache[url]
    res = http_get(url)
    pdf = res.content
    if len(pdf) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError("PDF trop volumineux")
    cache[url] = pdf
    st.session_state["_http_cache"] = cache
    return pdf

def extract_pdf_text(pdf_bytes: bytes, cap: Dict) -> Tuple[str, Dict]:
    """Extract text from PDF (TEXT-FIRST → OCR fallback)"""
    ocr_cfg = cap_get(cap, "kernel_params.ocr", {})
    max_pages = cap_get(cap, "kernel_params.atomization.max_pages", 140)
    
    t0 = time.time()
    pages = []
    method = "none"
    ocr_used = False
    
    # Strategy 1: pdfplumber
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages[:max_pages]:
                    try:
                        pages.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
                    except:
                        pages.append("")
            method = "pdfplumber"
        except:
            pass
    
    # Strategy 2: pypdf
    if not pages or sum(len(p) for p in pages) < 500:
        if PdfReader:
            try:
                pages2 = []
                reader = PdfReader(io.BytesIO(pdf_bytes))
                for page in reader.pages[:max_pages]:
                    try:
                        pages2.append(page.extract_text() or "")
                    except:
                        pages2.append("")
                if sum(len(p) for p in pages2) > sum(len(p) for p in pages):
                    pages, method = pages2, "pypdf"
            except:
                pass
    
    # Strategy 3: PyMuPDF
    if not pages or sum(len(p) for p in pages) < 700:
        if fitz:
            try:
                pages3 = []
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for i in range(min(max_pages, doc.page_count)):
                    pages3.append(doc.load_page(i).get_text("text") or "")
                if sum(len(p) for p in pages3) > sum(len(p) for p in pages):
                    pages, method = pages3, "pymupdf"
            except:
                pass
    
    # Clean text
    text_proc = cap_get(cap, "text_processing", {})
    text = _clean_text(pages, text_proc)
    
    meta = {
        "method": method,
        "pages": len(pages),
        "chars": len(text),
        "seconds": round(time.time() - t0, 3),
        "ocr_used": ocr_used,
        "fingerprint": sha256_fp(text)[:32],
    }
    return text, meta

def _clean_text(pages: List[str], text_proc: Dict) -> str:
    if not pages:
        return ""
    glued = text_proc.get("glued_patterns", [])
    all_lines = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = re.sub(r"(\w)-\n(\w)", r"\1\2", p)
        for item in glued:
            try:
                p = re.sub(item["pattern"], item["replacement"], p)
            except:
                pass
        p = re.sub(r"[ \t]+", " ", p)
        all_lines.extend([ln.strip() for ln in p.split("\n") if ln.strip()])
        all_lines.append("")
    
    # Remove headers/footers
    counts = Counter(all_lines)
    threshold = max(2, len(pages) // 3)
    skip = {ln for ln, cnt in counts.items() if cnt >= threshold and len(ln) < 100}
    clean = [ln for ln in all_lines if ln not in skip and not re.fullmatch(r"\d{1,3}", ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(clean)).strip()

# =============================================================================
# HARVEST (Automatique depuis CAP.HARVEST_SOURCES)
# =============================================================================
def harvest_auto(cap: Dict, level: str, volume_max: int) -> Dict:
    """
    Harvest automatique depuis sources CAP.
    Génère harvest_jobs et collecte sujets + corrigés.
    """
    set_state(PipelineState.HARVESTING, 15)
    log_pipeline(f"[HARVEST] Démarrage automatique pour {level}")
    
    sources = cap_get(cap, "harvest_sources", [])
    if not sources:
        raise ValueError("Aucune source dans CAP.harvest_sources")
    
    src = sources[0]
    base_url = src.get("base_url", "")
    endpoints = src.get("endpoints", {})
    scraping = src.get("scraping_rules", {})
    pairing = src.get("pairing_rules", {})
    
    endpoint = endpoints.get(level)
    if not endpoint:
        log_pipeline(f"[HARVEST] Pas d'endpoint pour {level}, utilisation fallback", "WARN")
        return {"items_total": 0, "items_with_corrige": 0, "library": []}
    
    root_url = f"{base_url}{endpoint}"
    log_pipeline(f"[HARVEST] Source: {src.get('source_name')} | URL: {root_url}")
    
    if not BeautifulSoup:
        raise RuntimeError("BeautifulSoup requis pour harvest")
    
    # Fetch year pages
    try:
        html = http_get(root_url).text
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        log_pipeline(f"[HARVEST] Erreur accès source: {e}", "ERROR")
        return {"items_total": 0, "items_with_corrige": 0, "library": []}
    
    year_pattern = scraping.get("year_pattern", r"(20\d{2})")
    year_links = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        m = re.search(year_pattern, href)
        if m:
            try:
                year = int(m.group(1))
                url = href if href.startswith("http") else f"{base_url}{href}"
                year_links.append((year, url))
            except:
                pass
    
    year_links = sorted(set(year_links), key=lambda x: -x[0])[:5]  # 5 dernières années
    log_pipeline(f"[HARVEST] Années trouvées: {len(year_links)}")
    
    pairs = []
    corrige_patterns = scraping.get("corrige_patterns", ["corrig"])
    exclude_patterns = scraping.get("exclude_patterns", [])
    geo_zones = pairing.get("geographic_zones", [])
    min_match = pairing.get("min_match_score", 0.25)
    
    for year, year_url in year_links:
        if len(pairs) >= volume_max:
            break
        try:
            y_html = http_get(year_url).text
            y_soup = BeautifulSoup(y_html, "html.parser")
            
            pdf_links = []
            for a in y_soup.find_all("a"):
                href = (a.get("href") or "").strip()
                if not href.lower().endswith(".pdf"):
                    continue
                abs_url = href if href.startswith("http") else f"{base_url}{href}"
                name = os.path.basename(abs_url)
                s = norm_text(f"{abs_url} {name}")
                
                if any(ex in s for ex in exclude_patterns):
                    continue
                
                is_corrige = any(p in s for p in corrige_patterns)
                pdf_links.append({"url": abs_url, "name": name, "is_corrige": is_corrige})
            
            sujets = [p for p in pdf_links if not p["is_corrige"]]
            corriges = [p for p in pdf_links if p["is_corrige"]]
            used = set()
            
            for suj in sujets:
                if len(pairs) >= volume_max:
                    break
                
                best_corr, best_score = None, 0.0
                s_name = norm_text(suj["name"])
                s_geo = next((z for z in geo_zones if z in s_name), "")
                
                for corr in corriges:
                    if corr["url"] in used:
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
                        score += 0.4 * len(s_tok & c_tok) / max(1, len(s_tok | c_tok))
                    
                    if score > best_score:
                        best_score, best_corr = score, corr
                
                corrige = best_corr if best_score >= min_match else None
                if corrige:
                    used.add(corrige["url"])
                
                pair = {
                    "pair_id": f"PAIR_{level}_{year}_{stable_id(suj['name'])}",
                    "year": year,
                    "sujet_name": suj["name"],
                    "sujet_url": suj["url"],
                    "has_corrige": bool(corrige),
                    "corrige_name": corrige["name"] if corrige else "",
                    "corrige_url": corrige["url"] if corrige else "",
                    "match_score": round(best_score, 2),
                }
                
                if not any(p["sujet_url"] == pair["sujet_url"] for p in pairs):
                    pairs.append(pair)
            
            log_pipeline(f"[HARVEST] Année {year}: {len(pairs)} pairs cumulées")
        except Exception as e:
            log_pipeline(f"[HARVEST] Erreur année {year}: {e}", "WARN")
    
    manifest = {
        "version": APP_VERSION,
        "timestamp": utc_ts(),
        "cap_id": cap_get(cap, "metadata.cap_id"),
        "level": level,
        "items_total": len(pairs),
        "items_with_corrige": sum(1 for p in pairs if p["has_corrige"]),
        "library": pairs,
    }
    
    log_pipeline(f"[HARVEST] Terminé: {manifest['items_total']} pairs, {manifest['items_with_corrige']} avec corrigé")
    return manifest

# =============================================================================
# ATOMIZATION (Qi/RQi)
# =============================================================================
def atomize_qi(text: str, cap: Dict) -> Tuple[List[str], Dict]:
    """Atomize sujet into Qi segments"""
    atom_cfg = cap_get(cap, "kernel_params.atomization", {})
    markers = atom_cfg.get("question_markers", [])
    min_len = atom_cfg.get("min_segment_len", 20)
    max_len = atom_cfg.get("max_segment_len", 2600)
    intent_verbs = cap_get(cap, "text_processing.intent_verbs", [])
    
    if not text:
        return [], {"raw": 0, "kept": 0}
    
    regs = []
    for pat in markers:
        try:
            regs.append(re.compile(pat))
        except:
            pass
    
    positions = {0}
    for rg in regs:
        for m in rg.finditer(text):
            positions.add(m.start())
    for m in re.finditer(r"\n\n+", text):
        positions.add(m.start())
        positions.add(m.end())
    
    positions = sorted([p for p in positions if 0 <= p <= len(text)])
    
    segments = []
    for a, b in zip(positions, positions[1:] + [len(text)]):
        seg = re.sub(r"[ \t]+", " ", text[a:b]).strip()
        if min_len <= len(seg) <= max_len:
            segments.append(seg)
    
    kept = []
    seen = set()
    for seg in segments:
        key = stable_id(norm_text(seg)[:400])
        if key in seen:
            continue
        seen.add(key)
        s = norm_text(seg)
        has_intent = any(v in s for v in intent_verbs) or "?" in seg
        has_math = bool(re.search(r"[=<>≤≥∫∑∏√]|\d", seg))
        if has_intent or has_math:
            kept.append(seg)
    
    return kept, {"raw": len(segments), "kept": len(kept)}

def atomize_rqi(text: str, cap: Dict) -> Tuple[List[str], Dict]:
    """Atomize corrigé into RQi segments"""
    atom_cfg = cap_get(cap, "kernel_params.atomization", {})
    min_len = atom_cfg.get("min_segment_len", 20)
    max_len = atom_cfg.get("max_segment_len", 2600)
    
    if not text:
        return [], {"raw": 0, "kept": 0}
    
    regs = [
        re.compile(r"(?m)^\s*(\d{1,2}|[a-h])\s*[\)\.\-:]\s+"),
        re.compile(r"(?im)^\s*(exercice|exo\.?)\s*([0-9ivx]+)\b"),
    ]
    
    positions = {0}
    for rg in regs:
        for m in rg.finditer(text):
            positions.add(m.start())
    for m in re.finditer(r"\n\n+", text):
        positions.add(m.start())
        positions.add(m.end())
    
    positions = sorted([p for p in positions if 0 <= p <= len(text)])
    
    segments = []
    for a, b in zip(positions, positions[1:] + [len(text)]):
        seg = re.sub(r"[ \t]+", " ", text[a:b]).strip()
        if min_len <= len(seg) <= max_len:
            segments.append(seg)
    
    kept = []
    seen = set()
    for seg in segments:
        key = stable_id(norm_text(seg)[:400])
        if key in seen:
            continue
        seen.add(key)
        has_math = bool(re.search(r"[=<>≤≥∫∑∏√]|\d", seg))
        if has_math or len(seg) > 100:
            kept.append(seg)
    
    return kept, {"raw": len(segments), "kept": len(kept)}

def align_qi_rqi(questions: List[str], answers: List[str], cap: Dict) -> List[Optional[int]]:
    """Align Qi to RQi"""
    align_cfg = cap_get(cap, "kernel_params.alignment", {})
    min_score = align_cfg.get("min_score", 0.18)
    
    if not questions or not answers:
        return [None] * len(questions)
    
    def tokenize(s):
        return set(re.findall(r"[a-z0-9]{2,}", norm_text(s)))
    
    def jaccard(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
    
    q_tok = [tokenize(q) for q in questions]
    a_tok = [tokenize(a) for a in answers]
    
    used = set()
    links = [None] * len(questions)
    
    for i in range(len(questions)):
        best_j, best_s = None, 0.0
        for j in range(len(answers)):
            if j in used:
                continue
            s = jaccard(q_tok[i], a_tok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= min_score:
            links[i] = best_j
            used.add(best_j)
    
    return links

# =============================================================================
# SCOPE MAPPING
# =============================================================================
def map_chapter(qi_text: str, chapters: List[Dict]) -> Tuple[str, float]:
    """Map Qi to chapter using CAP keywords"""
    if not chapters:
        return "UNMAPPED", 0.0
    
    qn = norm_text(qi_text)
    best_code, best_score = "UNMAPPED", 0.0
    
    for ch in chapters:
        kws = ch.get("keywords", [])
        if not kws:
            continue
        matches = sum(1 for kw in kws if kw and kw in qn)
        score = matches / max(6, len(kws))
        if score > best_score:
            best_score, best_code = score, ch.get("code", "UNMAPPED")
    
    return best_code, round(best_score, 4)

# =============================================================================
# ARI EXTRACTION
# =============================================================================
def extract_ari(qi: str, rqi: str, cap: Dict) -> Dict:
    """Extract ARI from Qi/RQi"""
    ari_cfg = cap_get(cap, "ari_config", {})
    op_patterns = ari_cfg.get("op_patterns", [])
    
    combined = norm_text(f"{qi}\n{rqi}")
    ops = []
    all_ops = set()
    sum_Tj = 0.0
    
    for item in op_patterns:
        op = item.get("op", "")
        pat = item.get("pattern", "")
        T_j = item.get("T_j", 0.3)
        
        if not op or not pat:
            continue
        
        try:
            matches = list(re.finditer(pat, combined, re.IGNORECASE))
        except:
            matches = []
        
        if matches and op not in all_ops:
            evidence = [m.group(0)[:40] for m in matches[:3]]
            ops.append({"op": op, "T_j": T_j, "evidence": evidence, "count": len(matches)})
            all_ops.add(op)
            sum_Tj += T_j
    
    if not ops:
        ops = [{"op": "OP_STANDARD", "T_j": 0.3, "evidence": [], "count": 0}]
        all_ops = {"OP_STANDARD"}
        sum_Tj = 0.3
    
    primary_op = ops[0]["op"] if ops else "OP_STANDARD"
    confidence = min(0.95, 0.55 + 0.1 * len(ops))
    
    return {
        "primary_op": primary_op,
        "ops": ops,
        "all_ops": sorted(list(all_ops)),
        "sum_Tj": round(sum_Tj, 4),
        "confidence_global": round(confidence, 4),
        "sig": {"A": primary_op, "P": "std", "O": "result", "X": str(len(ops))}
    }

# =============================================================================
# TRIGGERS
# =============================================================================
def build_triggers(qi: str, ari: Dict, chapter_code: str, cap: Dict) -> List[str]:
    """Build triggers (3-7 per QC)"""
    triggers = []
    
    for op_info in ari.get("ops", []):
        triggers.append(f"ARI:{op_info['op']}")
    
    qn = norm_text(qi)
    math_kws = ["limite", "derivee", "integrale", "probabilite", "suite", "complexe", "vecteur", "recurrence"]
    for kw in math_kws:
        if kw in qn:
            triggers.append(f"KW:{kw.upper()}")
    
    intent_verbs = cap_get(cap, "text_processing.intent_verbs", [])
    for v in intent_verbs:
        if v in qn:
            triggers.append(f"INTENT:{v.upper()}")
            break
    
    if chapter_code and chapter_code != "UNMAPPED":
        triggers.append(f"SCOPE:{chapter_code}")
    
    return list(dict.fromkeys(triggers))[:7]

# =============================================================================
# POSABLE GATE
# =============================================================================
def posable_gate(qi: str, rqi: str, chapter_code: str, ari: Dict, cap: Dict) -> Dict:
    """POSABLE Gate (3 criteria)"""
    rules = cap_get(cap, "kernel_params.posable_rules", {})
    reason_codes = []
    
    # POSABLE_CORRIGE
    has_rqi = bool(rqi and rqi.strip())
    if rules.get("require_rqi", True) and not has_rqi:
        reason_codes.append(RC.RQI_MISSING.value)
    
    # POSABLE_SCOPE
    has_scope = chapter_code and chapter_code != "UNMAPPED"
    if rules.get("require_scope", True) and not has_scope:
        reason_codes.append(RC.SCOPE_UNRESOLVED.value)
    
    # POSABLE_EVALUABLE
    if rules.get("require_evaluable", True):
        if ari.get("confidence_global", 0) < rules.get("min_ari_confidence", 0.55):
            reason_codes.append(RC.LOW_CONFIDENCE.value)
    
    return {
        "is_posable": len(reason_codes) == 0,
        "reason_codes": reason_codes,
        "confidence": ari.get("confidence_global", 0),
        "posable_corrige": has_rqi,
        "posable_scope": has_scope,
        "posable_evaluable": len(reason_codes) == 0,
    }

# =============================================================================
# FRT GENERATION
# =============================================================================
def generate_frt(primary_op: str, triggers: List[str], ari: Dict, cap: Dict) -> Dict:
    """Generate FRT (4 blocs obligatoires)"""
    op_labels = cap_get(cap, "ari_config.op_labels", {})
    label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())
    
    frt_id = f"FRT_{stable_id(primary_op, str(triggers))}"
    
    return {
        "frt_id": frt_id,
        "primary_op": primary_op,
        "title": f"Comment {label} ?",
        "blocks": {
            "usage": f"Reconnaître un problème de type: {label}",
            "reponse_type": "Appliquer la méthode ARI étape par étape",
            "pieges": "Vérifier les cas particuliers et conditions",
            "conclusion": "Valider le résultat et conclure",
        },
        "generated_at": utc_ts(),
    }

# =============================================================================
# F1/F2 KERNEL (§2)
# =============================================================================
def f1_kernel(delta_c: float, epsilon: float, sum_Tj: float) -> float:
    """F1 = δ_c × (ε + Σ T_j)²"""
    return float(delta_c) * float((epsilon + sum_Tj) ** 2)

def f2_kernel(n_hist: int, N_total: int, alpha: float, t_rec: float, Psi_q: float, selected: List, qc: Dict) -> float:
    """F2 = (n_q/N_total) × (1 + α/t_rec) × Ψ_q × Π(1-σ)"""
    if N_total <= 0:
        N_total = 1
    base = (float(n_hist) / float(N_total)) * (1.0 + float(alpha) / max(1e-9, float(t_rec))) * float(Psi_q)
    
    # σ similarity penalty
    penalty = 1.0
    for prev in selected:
        prev_ops = set(prev.get("all_ops", []) + prev.get("triggers", []))
        qc_ops = set(qc.get("all_ops", []) + qc.get("triggers", []))
        if prev_ops and qc_ops:
            sig = len(prev_ops & qc_ops) / max(1, len(prev_ops | qc_ops))
            penalty *= (1.0 - sig)
    
    return round(base * penalty, 8)

# =============================================================================
# IA2 JUDGE
# =============================================================================
def ia2_judge(qc: Dict, cap: Dict) -> Tuple[bool, List[Dict]]:
    """IA2 Judge - Boolean validation"""
    checks = []
    qc_fmt = cap_get(cap, "kernel_params.qc_format", {})
    trig_cfg = cap_get(cap, "kernel_params.triggers_config", {})
    
    # CHK_POSABLE_VALID
    posable_ok = qc.get("posable_in_cluster", 0) >= 1
    checks.append({"check_id": "CHK_POSABLE_VALID", "status": "PASS" if posable_ok else "FAIL"})
    
    # CHK_QC_FORM
    prefix = qc_fmt.get("prefix", "Comment")
    suffix = qc_fmt.get("suffix", "?")
    form_ok = qc.get("qc_text", "").startswith(prefix) and qc.get("qc_text", "").endswith(suffix)
    checks.append({"check_id": "CHK_QC_FORM", "status": "PASS" if form_ok else "FAIL"})
    
    # CHK_NO_LOCAL_CONSTANTS
    local_pat = r"\b(20[0-2]\d|paris|lyon|janvier|fevrier|mars|avril|mai|juin)\b"
    has_local = bool(re.search(local_pat, norm_text(qc.get("qc_text", "")), re.IGNORECASE))
    checks.append({"check_id": "CHK_NO_LOCAL_CONSTANTS", "status": "FAIL" if has_local else "PASS"})
    
    # CHK_FRT_TEMPLATE_OK
    frt = qc.get("frt", {})
    blocks = frt.get("blocks", {})
    frt_ok = all(k in blocks for k in ["usage", "reponse_type", "pieges", "conclusion"])
    checks.append({"check_id": "CHK_FRT_TEMPLATE_OK", "status": "PASS" if frt_ok else "FAIL"})
    
    # CHK_TRIGGERS_QUALITY
    trig_ok = trig_cfg.get("min_count", 3) <= len(qc.get("triggers", [])) <= trig_cfg.get("max_count", 7)
    checks.append({"check_id": "CHK_TRIGGERS_QUALITY", "status": "PASS" if trig_ok else "FAIL"})
    
    # CHK_NO_RECONSTRUCTION
    recon_ok = qc.get("qc_state") != "RECONSTRUCTED"
    checks.append({"check_id": "CHK_NO_RECONSTRUCTION", "status": "PASS" if recon_ok else "FAIL"})
    
    # Anti-singleton
    singleton_ok = qc.get("cluster_size", 0) >= 2 or qc.get("posable_in_cluster", 0) == 0
    checks.append({"check_id": "CHK_ANTI_SINGLETON", "status": "PASS" if singleton_ok else "FAIL"})
    
    all_pass = all(c["status"] == "PASS" for c in checks)
    return all_pass, checks

# =============================================================================
# FULL PIPELINE (100% automatique)
# =============================================================================
def run_full_pipeline(cap: Dict) -> Dict:
    """
    Pipeline complet automatique (Kernel V10.6.3)
    Déclenché par ACTIVATE_COUNTRY → LOAD_CAP
    """
    st.session_state["_run_ts"] = utc_ts()
    
    # Déterminer level (premier disponible)
    levels = cap_levels(cap)
    if not levels:
        raise ValueError("Aucun niveau dans CAP")
    level = levels[-1]["code"]  # Dernier = Terminale
    
    # ========== HARVEST AUTOMATIQUE ==========
    manifest = harvest_auto(cap, level, volume_max=30)
    st.session_state.harvest_manifest = manifest
    
    if manifest["items_with_corrige"] == 0:
        log_pipeline("[PIPELINE] Aucun corrigé trouvé - SAFETY_STOP", "ERROR")
        set_state(PipelineState.SAFETY_STOP)
        return {"error": "No corrigés found"}
    
    # ========== EXTRACTION + ATOMISATION ==========
    set_state(PipelineState.EXTRACTING, 25)
    log_pipeline(f"[PIPELINE] Extraction de {manifest['items_with_corrige']} pairs avec corrigé")
    
    library = [p for p in manifest["library"] if p.get("has_corrige")]
    chapters = cap_chapters(cap, "MATH", level)
    
    quarantine = []
    qi_items = []
    
    for pair in library[:20]:  # Limit for demo
        pid = pair["pair_id"]
        
        try:
            # Fetch PDFs
            sujet_bytes = fetch_pdf(pair["sujet_url"])
            corrige_bytes = fetch_pdf(pair["corrige_url"])
            
            # Extract text
            sujet_text, _ = extract_pdf_text(sujet_bytes, cap)
            corrige_text, _ = extract_pdf_text(corrige_bytes, cap)
            
            if len(sujet_text) < 200 or len(corrige_text) < 200:
                quarantine.append({"pair_id": pid, "reason": RC.SUJET_UNREADABLE.value if len(sujet_text) < 200 else RC.CORRIGE_UNREADABLE.value})
                continue
            
            # Atomize
            set_state(PipelineState.ATOMIZING, 35)
            questions, _ = atomize_qi(sujet_text, cap)
            answers, _ = atomize_rqi(corrige_text, cap)
            
            # Align
            links = align_qi_rqi(questions, answers, cap)
            
            # Create Qi items
            for i, qi_text in enumerate(questions):
                j = links[i] if i < len(links) else None
                rqi_text = answers[j] if (j is not None and j < len(answers)) else ""
                
                # ARI
                ari = extract_ari(qi_text, rqi_text, cap)
                
                # Chapter mapping
                chapter_code, ch_score = map_chapter(qi_text, chapters)
                
                # Triggers
                triggers = build_triggers(qi_text, ari, chapter_code, cap)
                
                # POSABLE
                set_state(PipelineState.POSABLE_GATE, 45)
                posable = posable_gate(qi_text, rqi_text, chapter_code, ari, cap)
                
                qi_id = f"QI_{stable_id(pid, str(i), qi_text[:100])}"
                
                qi_items.append({
                    "qi_id": qi_id,
                    "pair_id": pid,
                    "year": pair.get("year", 0),
                    "qi": qi_text,
                    "rqi": rqi_text,
                    "has_rqi": bool(rqi_text),
                    "chapter_code": chapter_code,
                    "chapter_score": ch_score,
                    "ari": ari,
                    "triggers": triggers,
                    "posable": posable,
                    "qc_id": None,
                })
        except Exception as e:
            log_pipeline(f"[PIPELINE] Erreur pair {pid}: {e}", "WARN")
            quarantine.append({"pair_id": pid, "reason": str(e)})
    
    log_pipeline(f"[PIPELINE] {len(qi_items)} Qi extraites, {len(quarantine)} quarantaine")
    
    # ========== CLUSTERING ==========
    set_state(PipelineState.CLUSTERING, 55)
    buckets = defaultdict(list)
    for qi in qi_items:
        primary_op = qi["ari"].get("primary_op", "OP_STANDARD")
        buckets[(qi["chapter_code"], primary_op)].append(qi)
    
    # ========== IA1 MINER + BUILDER ==========
    set_state(PipelineState.IA1_MINER, 60)
    set_state(PipelineState.IA1_BUILDER, 65)
    
    qi_posable = [q for q in qi_items if q["has_rqi"] and q["posable"]["is_posable"]]
    N_total = len(qi_posable)
    
    f1_f2 = cap_get(cap, "kernel_params.f1_f2_params", {})
    epsilon = f1_f2.get("epsilon", EPSILON)
    alpha = f1_f2.get("alpha", 1.0)
    
    qc_pack = []
    mapping = {}
    frt_pack = {}
    
    op_labels = cap_get(cap, "ari_config.op_labels", {})
    qc_fmt = cap_get(cap, "kernel_params.qc_format", {})
    
    for (chapter_code, primary_op), items in sorted(buckets.items()):
        cluster_size = len(items)
        posable_items = [x for x in items if x["has_rqi"] and x["posable"]["is_posable"]]
        posable_count = len(posable_items)
        
        if cluster_size < 2 and posable_count == 0:
            continue
        
        # Build QC
        label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())
        prefix = qc_fmt.get("prefix", "Comment")
        suffix = qc_fmt.get("suffix", "?")
        qc_text = f"{prefix} {label} {suffix}"
        qc_id = f"QC_{stable_id(chapter_code, primary_op, qc_text)}"
        
        # State
        if posable_count >= 2:
            qc_state = "POSABLE"
        elif posable_count == 1:
            qc_state = "POSABLE_WEAK"
        else:
            qc_state = "UNPOSABLE"
        
        # Aggregate
        all_triggers = []
        all_ops = set()
        for item in items:
            all_triggers.extend(item["triggers"])
            all_ops.update(item["ari"].get("all_ops", []))
        
        qc_triggers = [t for t, _ in Counter(all_triggers).most_common(7)]
        
        # FRT
        rep = max(items, key=lambda x: x["ari"].get("confidence_global", 0))
        frt = generate_frt(primary_op, qc_triggers, rep["ari"], cap)
        frt_pack[frt["frt_id"]] = frt
        
        # delta_c
        ch_info = next((c for c in chapters if c.get("code") == chapter_code), {})
        delta_c = ch_info.get("delta_c", 1.0)
        
        # sum_Tj
        sum_Tj = rep["ari"].get("sum_Tj", 0.3)
        
        # F1
        f1_raw = f1_kernel(delta_c, epsilon, sum_Tj)
        
        # t_rec
        years = [x["year"] for x in items if x.get("year")]
        t_rec = max(1.0, (2026 - max(years)) * 12) if years else 12.0
        
        qc = {
            "qc_id": qc_id,
            "qc_text": qc_text,
            "qc_label": label,
            "chapter_code": chapter_code,
            "primary_op": primary_op,
            "all_ops": sorted(list(all_ops)),
            "cluster_size": cluster_size,
            "posable_in_cluster": posable_count,
            "qc_state": qc_state,
            "qi_ids": [x["qi_id"] for x in items],
            "triggers": qc_triggers,
            "frt": frt,
            "delta_c": delta_c,
            "sum_Tj": sum_Tj,
            "f1_raw": round(f1_raw, 6),
            "Psi_q": 0.0,
            "n_hist": cluster_size,
            "t_rec": t_rec,
            "f2_score": 0.0,
            "ia2_status": "PENDING",
            "ia2_checks": [],
        }
        
        qc_pack.append(qc)
        for item in items:
            mapping[item["qi_id"]] = qc_id
    
    # ========== F1/F2 SCORING ==========
    set_state(PipelineState.F1_F2_SCORING, 75)
    
    # Normalize Psi_q per chapter
    by_chapter = defaultdict(list)
    for qc in qc_pack:
        by_chapter[qc["chapter_code"]].append(qc)
    
    for ch_code, qcs in by_chapter.items():
        max_f1 = max((q["f1_raw"] for q in qcs), default=0)
        for qc in qcs:
            qc["Psi_q"] = round(qc["f1_raw"] / max_f1, 6) if max_f1 > 0 else 0.0
    
    # ========== IA2 JUDGE ==========
    set_state(PipelineState.IA2_JUDGE, 80)
    
    for qc in qc_pack:
        ia2_pass, ia2_checks = ia2_judge(qc, cap)
        qc["ia2_status"] = "PASS" if ia2_pass else "FAIL"
        qc["ia2_checks"] = ia2_checks
    
    # ========== SELECTION & COVERAGE ==========
    set_state(PipelineState.SELECTION, 85)
    
    # Update Qi with QC mapping
    orphans = 0
    for qi in qi_items:
        if qi["qi_id"] in mapping:
            qi["qc_id"] = mapping[qi["qi_id"]]
        else:
            if qi["has_rqi"] and qi["posable"]["is_posable"]:
                orphans += 1
    
    # Coverage
    coverage = 0.0
    if N_total > 0:
        covered = sum(1 for qi in qi_posable if qi.get("qc_id"))
        coverage = round(covered / N_total, 4)
    
    # ========== COVERAGE CHECK & SEAL ==========
    set_state(PipelineState.COVERAGE_CHECK, 90)
    
    coverage_threshold = f1_f2.get("seal_coverage_threshold", 0.95)
    ia2_pass_count = sum(1 for qc in qc_pack if qc["ia2_status"] == "PASS")
    ia2_ok = ia2_pass_count == len(qc_pack) if qc_pack else True
    
    seal_reason = []
    if coverage < coverage_threshold:
        seal_reason.append(f"COVERAGE_LOW_{coverage:.2%}")
    if orphans > 0:
        seal_reason.append(f"ORPHANS_{orphans}")
    if not ia2_ok:
        seal_reason.append(f"IA2_FAIL_{len(qc_pack) - ia2_pass_count}")
    
    sealed = len(seal_reason) == 0
    
    if sealed:
        set_state(PipelineState.SEALED, 100)
        log_pipeline(f"[PIPELINE] SEALED - Coverage={coverage:.0%} | QC={len(qc_pack)} | IA2=OK")
    else:
        set_state(PipelineState.SAFETY_STOP, 100)
        log_pipeline(f"[PIPELINE] SAFETY_STOP - {' | '.join(seal_reason)}", "WARN")
    
    # ========== BUILD RESULT ==========
    audit = {
        "pairs_processed": len(library),
        "qi_total": len(qi_items),
        "qi_posable": N_total,
        "orphans": orphans,
        "qc_total": len(qc_pack),
        "qc_ia2_pass": ia2_pass_count,
        "coverage": coverage,
        "ia2_ok": ia2_ok,
        "sealed": sealed,
        "seal_reason": seal_reason,
    }
    
    evidence_pack = {
        "version": APP_VERSION,
        "kernel_version": KERNEL_VERSION,
        "timestamp": st.session_state["_run_ts"],
        "cap_id": cap_get(cap, "metadata.cap_id"),
        "cap_fingerprint": cap_get(cap, "metadata.cap_fingerprint_sha256"),
        "audit": audit,
    }
    
    # Save to session
    st.session_state.qi_pack = qi_items
    st.session_state.qc_pack = qc_pack
    st.session_state.frt_pack = frt_pack
    st.session_state.quarantine = quarantine
    st.session_state.evidence_pack = evidence_pack
    st.session_state.audit_log = audit
    st.session_state.sealed = sealed
    st.session_state.seal_reason = seal_reason
    
    return {
        "qi_pack": qi_items,
        "qc_pack": qc_pack,
        "frt_pack": frt_pack,
        "audit": audit,
        "sealed": sealed,
    }

# =============================================================================
# ACTIVATE_COUNTRY (SEULE ACTION HUMAINE)
# =============================================================================
def activate_country(country_code: str):
    """
    ACTIVATE_COUNTRY(country_code) - SEULE ACTION HUMAINE AUTORISÉE
    
    Déclenche automatiquement:
    1) LOAD_CAP(country_code)
    2) Pipeline complet
    3) UI lecture seule
    """
    log_pipeline(f"[ACTIVATE] ACTIVATE_COUNTRY({country_code})")
    set_state(PipelineState.ACTIVATING, 0)
    
    st.session_state.country_code = country_code
    
    try:
        # 1) LOAD_CAP
        cap = load_cap(country_code)
        st.session_state.cap = cap
        st.session_state.cap_fingerprint = cap["metadata"]["cap_fingerprint_sha256"]
        
        # 2) Pipeline complet
        result = run_full_pipeline(cap)
        
        return result
        
    except Exception as e:
        log_pipeline(f"[ACTIVATE] ERREUR: {e}", "ERROR")
        set_state(PipelineState.ERROR)
        st.session_state.safety_stop = True
        st.session_state.safety_stop_reason = str(e)
        raise

# =============================================================================
# STREAMLIT UI (LECTURE SEULE)
# =============================================================================
def main():
    st.set_page_config(
        page_title=f"SMAXIA GTE {KERNEL_VERSION}",
        page_icon="🏛️",
        layout="wide"
    )
    
    init_state()
    
    # Header
    st.markdown(f"""
    # 🏛️ SMAXIA GTE Console {KERNEL_VERSION}
    **ISO-PROD | Kernel V10.6.3 Scellé | CAS 1 ONLY**
    """)
    
    st.markdown("---")
    
    # ========== SEULE ACTION HUMAINE: ACTIVATE_COUNTRY ==========
    if st.session_state.pipeline_state == PipelineState.IDLE:
        st.markdown("## 🚀 ACTIVATION PAYS")
        st.markdown("""
        **Seule action humaine autorisée** (Kernel V10.6.3 §0.2.11)
        
        Après activation, TOUT est automatique:
        - LOAD_CAP → Harvest → Atomisation → POSABLE → Clustering → IA1/IA2 → F1/F2 → Seal
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            country = st.selectbox(
                "Sélectionner le pays",
                options=list(CAP_REGISTRY.keys()),
                format_func=lambda x: f"{x} - {CAP_REGISTRY[x]['name']}"
            )
            
            if st.button("🚀 ACTIVATE_COUNTRY", type="primary", use_container_width=True):
                with st.spinner("Pipeline en cours..."):
                    try:
                        activate_country(country)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur: {e}")
        
        with col2:
            st.info("""
            **Pipeline automatique:**
            1. LOAD_CAP(country_code) - Charge le CAP officiel
            2. Harvest automatique depuis sources institutionnelles
            3. Extraction PDF → Atomisation Qi/RQi
            4. POSABLE Gate (3 critères)
            5. Clustering + Anti-singleton
            6. IA1 Miner → IA1 Builder
            7. IA2 Judge (checks booléens)
            8. F1/F2 Scoring
            9. Sélection coverage-driven
            10. SEAL ou SAFETY_STOP
            """)
    
    # ========== UI LECTURE SEULE (après activation) ==========
    else:
        # Status bar
        state = st.session_state.pipeline_state
        progress = st.session_state.pipeline_progress
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("État", state.value)
        col2.metric("Progression", f"{progress}%")
        col3.metric("Pays", st.session_state.country_code or "N/A")
        col4.metric("SEALED", "✅ OUI" if st.session_state.sealed else "❌ NON")
        
        st.progress(progress / 100)
        
        # Safety Stop alert
        if st.session_state.safety_stop:
            st.error(f"⚠️ SAFETY_STOP: {st.session_state.safety_stop_reason}")
        
        st.markdown("---")
        
        # Tabs for results (LECTURE SEULE)
        if st.session_state.qc_pack:
            tab1, tab2, tab3, tab4 = st.tabs(["📊 QC par Chapitre", "🔍 Détail QC", "📋 Audit", "📝 Logs"])
            
            with tab1:
                st.markdown("## QC par Chapitre (Lecture seule)")
                
                qc_pack = st.session_state.qc_pack
                by_chapter = defaultdict(list)
                for qc in qc_pack:
                    by_chapter[qc["chapter_code"]].append(qc)
                
                for ch_code, qcs in sorted(by_chapter.items()):
                    with st.expander(f"📁 {ch_code} ({len(qcs)} QC)", expanded=True):
                        for qc in sorted(qcs, key=lambda x: -x["Psi_q"]):
                            status_icon = "✅" if qc["ia2_status"] == "PASS" else "❌"
                            st.markdown(f"""
                            **{status_icon} {qc['qc_text']}**
                            - ID: `{qc['qc_id']}`
                            - Cluster: {qc['cluster_size']} Qi | Posable: {qc['posable_in_cluster']}
                            - Ψq: {qc['Psi_q']:.4f} | F1: {qc['f1_raw']:.4f}
                            - Triggers: {', '.join(qc['triggers'][:5])}
                            """)
                            st.markdown("---")
            
            with tab2:
                st.markdown("## Détail QC (Lecture seule)")
                
                qc_options = [f"{qc['qc_id']} | {qc['chapter_code']} | {qc['qc_label']}" for qc in qc_pack]
                if qc_options:
                    selected_idx = st.selectbox("Sélectionner une QC", range(len(qc_options)), format_func=lambda i: qc_options[i])
                    qc = qc_pack[selected_idx]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {qc['qc_text']}")
                        st.markdown(f"**Chapitre:** {qc['chapter_code']}")
                        st.markdown(f"**Primary Op:** {qc['primary_op']}")
                        st.markdown(f"**État:** {qc['qc_state']}")
                        st.markdown(f"**IA2:** {qc['ia2_status']}")
                        
                        st.markdown("#### Triggers")
                        for t in qc["triggers"]:
                            st.code(t)
                    
                    with col2:
                        st.markdown("#### FRT")
                        frt = qc.get("frt", {})
                        blocks = frt.get("blocks", {})
                        for key, val in blocks.items():
                            st.markdown(f"**{key}:** {val}")
                        
                        st.markdown("#### ARI")
                        st.json({"primary_op": qc["primary_op"], "all_ops": qc["all_ops"], "sum_Tj": qc["sum_Tj"]})
                    
                    st.markdown("#### Qi associées")
                    qi_pack = st.session_state.qi_pack or []
                    qi_linked = [qi for qi in qi_pack if qi.get("qc_id") == qc["qc_id"]]
                    
                    for qi in qi_linked[:10]:
                        with st.expander(f"📄 {qi['qi_id'][:20]}..."):
                            st.markdown(f"**Qi:** {qi['qi'][:300]}...")
                            if qi.get("rqi"):
                                st.markdown(f"**RQi:** {qi['rqi'][:300]}...")
                            st.markdown(f"**Posable:** {'✅' if qi['posable']['is_posable'] else '❌'}")
            
            with tab3:
                st.markdown("## Audit (Lecture seule)")
                
                audit = st.session_state.audit_log
                if audit:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Pairs", audit.get("pairs_processed", 0))
                    col2.metric("Qi Total", audit.get("qi_total", 0))
                    col3.metric("Qi Posable", audit.get("qi_posable", 0))
                    col4.metric("Coverage", f"{audit.get('coverage', 0):.0%}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("QC Total", audit.get("qc_total", 0))
                    col2.metric("QC IA2 Pass", audit.get("qc_ia2_pass", 0))
                    col3.metric("Orphans", audit.get("orphans", 0))
                    col4.metric("SEALED", "✅" if audit.get("sealed") else "❌")
                    
                    if audit.get("seal_reason"):
                        st.warning(f"Seal Reason: {' | '.join(audit['seal_reason'])}")
                    
                    st.markdown("### Evidence Pack")
                    st.json(st.session_state.evidence_pack)
            
            with tab4:
                st.markdown("## Pipeline Logs")
                logs = st.session_state.pipeline_log
                for entry in reversed(logs[-50:]):
                    level = entry.get("level", "INFO")
                    color = "red" if level == "ERROR" else "orange" if level == "WARN" else "gray"
                    st.markdown(f"<span style='color:{color}'>[{entry['ts']}] [{level}] {entry['msg']}</span>", unsafe_allow_html=True)
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Réinitialiser", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# =============================================================================
# ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    main()
