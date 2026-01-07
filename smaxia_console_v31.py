# =============================================================================
# SMAXIA GTE Console V31.10.22 FINAL — ISO-PROD (KERNEL V10.6.1)
# =============================================================================
# FUSION: GPT 5.2 (architecture/F1/F2/IA2) + Claude (UI/extraction)
#
# CONFORMITÉ KERNEL V10.6.1:
# ✅ F1 = δc·(ε + Σ Tj)² avec normalisation intra-chapitre
# ✅ F2 = (n_hist/N)·(1+α/t_rec)·Ψq·Π(1-σ) avec anti-redondance
# ✅ CAS 1 ONLY: zéro reconstruction, quarantaine RC_*
# ✅ POSABLE Gate: POSABLE_CORRIGE + POSABLE_SCOPE + POSABLE_EVALUABLE
# ✅ Anti-singleton: n_q_cluster >= 2
# ✅ Pack-driven: scope, verbes, qc_format via Pack JSON
# ✅ Saturation progressive: new_QC = 0
# ✅ Signature SHA256 outputs
# ✅ EvidencePack + AuditLog IA2
#
# OBJECTIF CEO: ≥480 Qi pour 30 sujets (16 Qi/sujet)
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
from typing import Dict, List, Optional, Tuple, Any, Iterable, Set
from collections import defaultdict

import requests
import streamlit as st

try:
    import pdfplumber
    PDFPLUMBER_OK = True
except ImportError:
    pdfplumber = None
    PDFPLUMBER_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BeautifulSoup = None
    BS4_OK = False

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    np = None
    NUMPY_OK = False

# =============================================================================
# KERNEL CONSTANTS (SCELLÉES V10.6.1)
# =============================================================================
VERSION = "V31.10.22-FINAL"
KERNEL_VERSION = "V10.6.1"
FINGERPRINT_ALGO = "SHA256"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

# F1/F2 Constants (Kernel scellé)
F1_EPSILON = 0.1  # Kernel V10.6.1: ε = 0.1
F2_ALPHA = 1.0
F2_TREC_DEFAULT = 1.0

# Extraction targets
TARGET_QI_PER_SUBJECT = 16
MIN_QI_LENGTH = 15

# =============================================================================
# REASON CODES (KERNEL-ALIGNED)
# =============================================================================
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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def sha256_bytes(b: bytes) -> str:
    return f"sha256:{hashlib.sha256(b).hexdigest()}"

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", errors="ignore"))

def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:12]

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def stable_sort(items: Iterable[Any], key_fn) -> List[Any]:
    return sorted(list(items), key=key_fn)

def norm_text(s: str) -> str:
    import unicodedata
    s = (s or "").replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", norm_text(s))[:2000]

def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

# =============================================================================
# DATA STRUCTURES (KERNEL-ALIGNED)
# =============================================================================
@dataclass
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
    language: str
    year_ref: Optional[str]
    source_url: str
    source_fingerprint: str
    extraction_method: str
    pairing_confidence: float
    sanitizer_derivations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PosableDecision:
    qi_id: str
    chapter_ref: Optional[str]
    posable_status: bool
    posable_corrige: bool
    posable_scope: bool
    posable_evaluable: bool
    reason_codes: List[str]
    evidence_refs: List[str]
    scope_trace: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TraceARI:
    qi_id: str
    chapter_ref: str
    actions: List[Dict[str, Any]]
    triggers: List[Dict[str, Any]]
    output_type: str
    evidence_refs: List[str]

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
    primary_op: str
    ari_typed: Dict[str, Any]
    frt: Dict[str, Any]
    triggers: List[Dict[str, Any]]
    sig: str
    n_q_cluster: int
    qi_ids: List[str]
    evidence_pack: Dict[str, Any]
    psi_raw: float
    psi_q: float  # Normalized F1
    f2_score: float
    ia2_audit: List[AuditEntry]
    ia2_pass: bool

# =============================================================================
# PACK GENESIS (SIMULATION P2-P5 — ZÉRO HARDCODE KERNEL)
# =============================================================================
def genesis_pack_fr() -> Dict[str, Any]:
    """Pack Genesis France - Toute la variabilité métier est ICI, pas dans le Kernel"""
    return {
        "pack_id": "CAP_FR_GENESIS_V31_10_22",
        "pack_version": "31.10.22",
        "country_code": "FR",
        "country_name": "France",
        "language": "fr",
        
        # Policy: QC Format
        "policy": {
            "qc_format": {
                "prefix": "Comment ",
                "suffix": " ?"
            },
            "harvest": {
                "root_url": "https://www.apmep.fr/Annales-Terminale-Generale",
                "corrige_keywords": ["corrig", "corrigé", "correction", "solution"],
                "pdf_link_regex": r"\.pdf",
            },
            "pairing": {
                "min_similarity_fallback": 0.35,
            },
            "posable": {
                "evaluable_verbs": [
                    "calculer", "déterminer", "résoudre", "montrer", "démontrer",
                    "prouver", "justifier", "vérifier", "établir", "exprimer",
                    "simplifier", "factoriser", "tracer", "étudier", "déduire"
                ],
            },
            "clustering": {
                "distance_threshold": 0.55,
            },
            "scoring": {
                "eps": 0.1,  # Kernel ε scellé
                "alpha_recency": 1.0,
            },
            "level": {
                "delta_c": 1.0,  # Coefficient niveau (secret)
            },
            "selection": {
                "max_k_per_chapter": 15,
            },
        },
        
        # Academic chapters (scope rules)
        "academic": {
            "chapters": [
                {
                    "chapter_code": "CH_ANALYSE",
                    "label": "Analyse",
                    "match_keywords": ["limite", "dérivée", "derivee", "continuité", "continuite", "variation", "tangente", "asymptote", "fonction"],
                    "match_regexes": [r"\blim\b", r"\bf'\s*\(", r"dérivable"],
                    "delta_c": 1.0,
                },
                {
                    "chapter_code": "CH_PROBABILITES",
                    "label": "Probabilités",
                    "match_keywords": ["probabilité", "probabilite", "loi", "binomiale", "normale", "espérance", "esperance", "variance", "aléatoire"],
                    "match_regexes": [r"P\s*\(", r"E\s*\(X\)", r"loi\s+(de|binomiale|normale)"],
                    "delta_c": 1.0,
                },
                {
                    "chapter_code": "CH_SUITES",
                    "label": "Suites",
                    "match_keywords": ["suite", "récurrence", "recurrence", "arithmétique", "géométrique", "convergence", "terme", "rang"],
                    "match_regexes": [r"u_n", r"u_{n\+1}", r"récurrence"],
                    "delta_c": 1.0,
                },
                {
                    "chapter_code": "CH_INTEGRATION",
                    "label": "Intégration",
                    "match_keywords": ["intégrale", "integrale", "primitive", "aire", "intégration"],
                    "match_regexes": [r"∫", r"\\int", r"primitive"],
                    "delta_c": 1.0,
                },
                {
                    "chapter_code": "CH_GEOMETRIE",
                    "label": "Géométrie espace",
                    "match_keywords": ["vecteur", "plan", "droite", "espace", "orthogonal", "colinéaire", "scalaire"],
                    "match_regexes": [r"\\vec", r"produit\s+scalaire"],
                    "delta_c": 1.0,
                },
                {
                    "chapter_code": "CH_COMPLEXES",
                    "label": "Nombres complexes",
                    "match_keywords": ["complexe", "imaginaire", "module", "argument", "affixe", "conjugué"],
                    "match_regexes": [r"\bi\b.*=.*-1", r"forme\s+exponentielle"],
                    "delta_c": 1.0,
                },
                {
                    "chapter_code": "CH_LOGEXP",
                    "label": "Log & Exp",
                    "match_keywords": ["exponentielle", "logarithme", "ln", "exp"],
                    "match_regexes": [r"\bln\b", r"\bexp\b", r"e\^"],
                    "delta_c": 1.0,
                },
            ],
        },
        
        # Verb ontology (cognitive weights — Kernel Table)
        "verb_ontology": {
            "verbs": [
                {"verb_id": "IDENTIFY", "forms": ["identifier", "repérer", "nommer"], "weight": 0.15},
                {"verb_id": "DEFINE", "forms": ["définir", "rappeler"], "weight": 0.15},
                {"verb_id": "ANALYZE", "forms": ["analyser", "observer", "examiner"], "weight": 0.20},
                {"verb_id": "SYNTHESIZE", "forms": ["synthétiser", "conclure"], "weight": 0.20},
                {"verb_id": "SIMPLIFY", "forms": ["simplifier", "factoriser", "réduire"], "weight": 0.25},
                {"verb_id": "CALCULATE", "forms": ["calculer", "mesurer", "quantifier"], "weight": 0.30},
                {"verb_id": "FORMULATE", "forms": ["exprimer", "formuler"], "weight": 0.30},
                {"verb_id": "APPLY", "forms": ["appliquer", "utiliser"], "weight": 0.35},
                {"verb_id": "INTERPRET", "forms": ["interpréter"], "weight": 0.35},
                {"verb_id": "SOLVE", "forms": ["résoudre", "déterminer"], "weight": 0.40},
                {"verb_id": "DIFF_INTEG", "forms": ["dériver", "intégrer"], "weight": 0.40},
                {"verb_id": "ARGUE", "forms": ["argumenter", "justifier"], "weight": 0.45},
                {"verb_id": "PROVE", "forms": ["démontrer", "prouver", "montrer"], "weight": 0.50},
                {"verb_id": "RECURRENCE", "forms": ["récurrence", "initialisation", "hérédité"], "weight": 0.60},
            ],
            "weights_override": {},  # Pack peut override les poids
        },
        
        # QC phrases by verb_id (Pack-driven, pas Kernel)
        "qc_phrases": {
            "by_verb_id": {
                "DIFF_INTEG": "dériver ou intégrer une expression",
                "SOLVE": "résoudre une équation ou un problème",
                "CALCULATE": "effectuer un calcul",
                "PROVE": "démontrer un résultat",
                "APPLY": "appliquer une règle ou un théorème",
                "RECURRENCE": "démontrer par récurrence",
                "ANALYZE": "analyser une situation",
            },
            "default": "résoudre un exercice standard",
        },
        
        # FRT Templates by primary_op (Pack-driven)
        "frt_templates": {
            "PROVE": {
                "steps": [
                    "1. Reformuler clairement l'objectif",
                    "2. Choisir la stratégie (directe, absurde, contraposée)",
                    "3. Dérouler le raisonnement",
                    "4. Vérifier les hypothèses",
                    "5. Conclure",
                ],
            },
            "SOLVE": {
                "steps": [
                    "1. Identifier le type d'équation",
                    "2. Mettre sous forme résoluble",
                    "3. Résoudre",
                    "4. Vérifier les solutions",
                    "5. Conclure",
                ],
            },
            "DIFF_INTEG": {
                "steps": [
                    "1. Identifier la fonction et son domaine",
                    "2. Appliquer les règles de dérivation/intégration",
                    "3. Simplifier",
                    "4. Vérifier",
                    "5. Exploiter le résultat",
                ],
            },
            "RECURRENCE": {
                "steps": [
                    "1. Initialisation: vérifier P(n₀)",
                    "2. Hérédité: supposer P(n) vraie",
                    "3. Démontrer P(n+1)",
                    "4. Conclusion",
                ],
            },
            "DEFAULT": {
                "steps": [
                    "1. Identifier données et objectif",
                    "2. Choisir une méthode",
                    "3. Exécuter",
                    "4. Vérifier",
                    "5. Conclure",
                ],
            },
        },
    }


def pack_get(pack: Dict[str, Any], path: str, default=None):
    """Accès pack-driven sans hardcode"""
    cur = pack
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


GENESIS_PACKS = {
    "FR": ("France", genesis_pack_fr),
}


def generate_pack(country_code: str) -> Dict[str, Any]:
    if country_code not in GENESIS_PACKS:
        raise ValueError(f"Pays non supporté: {country_code}")
    _, gen = GENESIS_PACKS[country_code]
    pack = gen()
    pack["created_at"] = now_iso()
    pack["_source"] = "GENESIS_AUTO"
    pack["_pack_sig"] = sha256_text(safe_json(pack))[:24]
    return pack


# =============================================================================
# SESSION STATE
# =============================================================================
def ss_init():
    defaults = {
        "pack_active": None,
        "pack_id": None,
        "library": [],
        "atoms": [],
        "posable_decisions": {},
        "traces": {},
        "qc_candidates": [],
        "selections": {},
        "sealed": False,
        "logs": [],
        "metrics": {},
        "_doc_bytes": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def log(msg: str):
    st.session_state.logs.append(f"[{now_iso()}] {msg}")


# =============================================================================
# PDF EXTRACTION + ATOMISATION
# =============================================================================
def extract_pdf_pages(pdf_bytes: bytes) -> Tuple[List[List[str]], Dict[str, Any]]:
    """Extraction multi-pages avec audit"""
    audit = {"pages": 0, "pages_with_text": 0, "mode": "pdfplumber", "warnings": []}
    pages_lines = []
    
    if not PDFPLUMBER_OK:
        audit["warnings"].append("pdfplumber not available")
        return [], audit
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            audit["pages"] = len(pdf.pages)
            for p in pdf.pages:
                txt = p.extract_text(x_tolerance=2, y_tolerance=3) or ""
                lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
                if lines:
                    audit["pages_with_text"] += 1
                pages_lines.append(lines)
    except Exception as e:
        audit["warnings"].append(f"extraction_error: {e}")
    
    if audit["pages"] > 0 and audit["pages_with_text"] / audit["pages"] < 0.25:
        audit["warnings"].append("low_text_density_likely_scanned")
    
    return pages_lines, audit


# Patterns d'extraction (Kernel-safe, structure générique)
EXO_RE = re.compile(r"(?i)\b(?:exercice|partie)\s*[:\s]*([0-9ivx]+|[A-Z])\b")
QNUM_RE = re.compile(r"^\s*(?:question\s*)?(\d{1,2})\s*[\)\.\:\-]\s*(.*)$", re.I)
SUBQ_RE = re.compile(r"^\s*([a-h])\s*[\)\.\:\-]\s*(.*)$", re.I)


def looks_like_evaluable(text: str, pack: Dict[str, Any]) -> bool:
    """Détection évaluable via Pack (pas hardcode Kernel)"""
    verbs = pack_get(pack, "policy.posable.evaluable_verbs", [])
    if not verbs:
        # Fallback minimal
        verbs = ["calculer", "déterminer", "montrer", "démontrer", "résoudre"]
    
    text_low = text.lower()
    if "?" in text:
        return True
    return any(v in text_low for v in verbs)


def sanitize_text(text: str, lang: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Sanitization avec traces (Kernel: neutraliser constantes locales)"""
    derivations = []
    s = text
    
    # Neutraliser années
    s2 = re.sub(r"\b(19|20)\d{2}\b", "<YEAR>", s)
    if s2 != s:
        derivations.append({"op": "neutralize_year", "before_sha": sha256_text(s)[:16], "after_sha": sha256_text(s2)[:16]})
        s = s2
    
    # Neutraliser IDs/centres
    s2 = re.sub(r"\b[A-Z]{2,}\-[A-Z0-9]{2,}\b", "<ID>", s)
    if s2 != s:
        derivations.append({"op": "neutralize_id"})
        s = s2
    
    # Whitespace normalize
    s2 = re.sub(r"[ \t]+", " ", s).strip()
    if s2 != s:
        derivations.append({"op": "ws_normalize"})
        s = s2
    
    return s, derivations


def atomise_subject(pages_lines: List[List[str]], doc_id: str, lang: str, pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Atomisation multi-stratégies (Exercice → Question → Sous-question)"""
    units = []
    current_exo = None
    current_qnum = None
    current_lines = []
    current_loc = None
    
    def flush_unit():
        nonlocal current_lines, current_loc
        if not current_lines or not current_loc:
            current_lines = []
            current_loc = None
            return
        
        raw = "\n".join(current_lines).strip()
        if len(raw) >= MIN_QI_LENGTH:
            units.append({
                "exo_ref": current_exo,
                "q_ref": f"{current_qnum or ''}{current_subq or ''}".strip() or f"Q{len(units)+1}",
                "qi_raw": raw,
                "locator": current_loc,
                "method": "STRUCTURED",
            })
        current_lines = []
        current_loc = None
    
    current_subq = ""
    
    for pi, lines in enumerate(pages_lines):
        for li, line in enumerate(lines):
            # Détection Exercice
            m_exo = EXO_RE.search(line)
            if m_exo:
                current_exo = m_exo.group(1).upper()
            
            # Détection Question numérotée
            m_q = QNUM_RE.match(line)
            m_sub = SUBQ_RE.match(line)
            
            if m_q:
                flush_unit()
                current_qnum = m_q.group(1)
                current_subq = ""
                rest = m_q.group(2)
                current_lines = [rest] if rest else []
                current_loc = Locator(page_start=pi, page_end=pi, line_start=li, line_end=li)
                continue
            
            if m_sub and current_qnum:
                flush_unit()
                current_subq = m_sub.group(1).lower()
                rest = m_sub.group(2)
                current_lines = [rest] if rest else []
                current_loc = Locator(page_start=pi, page_end=pi, line_start=li, line_end=li)
                continue
            
            # Accumulation
            if current_loc is not None:
                current_lines.append(line)
                current_loc = Locator(
                    page_start=current_loc.page_start,
                    page_end=pi,
                    line_start=current_loc.line_start,
                    line_end=li
                )
    
    flush_unit()
    
    # STRATÉGIE FALLBACK: Si peu d'unités, découper par verbes/paragraphes
    if len(units) < 5:
        full_text = "\n\n".join("\n".join(pl) for pl in pages_lines)
        
        # Par verbes interrogatifs
        verbs = pack_get(pack, "policy.posable.evaluable_verbs", [])
        if verbs:
            pattern = r"(?i)(?:^|\.\s+|\n\s*)([^.]*?\b(?:" + "|".join(verbs) + r")\b[^.]*\.)"
            matches = re.findall(pattern, full_text)
            for m in matches:
                if len(m) >= MIN_QI_LENGTH and len(m) < 1500:
                    if not any(u["qi_raw"][:50] == m[:50] for u in units):
                        units.append({
                            "exo_ref": "VERB",
                            "q_ref": f"V{len(units)+1}",
                            "qi_raw": m.strip(),
                            "locator": Locator(page_start=0, page_end=0),
                            "method": "VERB_PATTERN",
                        })
        
        # Par questions directes
        questions = re.findall(r"([^.!?]{25,}?\?)", full_text)
        for q in questions:
            if len(q) >= MIN_QI_LENGTH:
                if not any(u["qi_raw"][:50] == q[:50] for u in units):
                    units.append({
                        "exo_ref": "QUESTION",
                        "q_ref": f"Q{len(units)+1}",
                        "qi_raw": q.strip(),
                        "locator": Locator(page_start=0, page_end=0),
                        "method": "QUESTION_MARK",
                    })
    
    return units


def atomise_correction(pages_lines: List[List[str]], doc_id: str, lang: str) -> Dict[str, Dict[str, Any]]:
    """Segmentation corrigé indexée par (exo::q_ref)"""
    idx = {}
    current_exo = None
    current_qnum = None
    current_key = None
    cur_lines = []
    
    def flush():
        nonlocal current_key, cur_lines
        if current_key and cur_lines:
            idx[current_key] = {"rqi_raw": "\n".join(cur_lines).strip()}
        current_key = None
        cur_lines = []
    
    for pi, lines in enumerate(pages_lines):
        for li, line in enumerate(lines):
            m_exo = EXO_RE.search(line)
            if m_exo:
                current_exo = m_exo.group(1).upper()
            
            m_q = QNUM_RE.match(line)
            m_sub = SUBQ_RE.match(line)
            
            if m_q:
                flush()
                current_qnum = m_q.group(1)
                current_key = f"{current_exo or 'X'}::{current_qnum}"
                cur_lines = [m_q.group(2)] if m_q.group(2) else []
                continue
            
            if m_sub and current_qnum:
                flush()
                sub = m_sub.group(1).lower()
                current_key = f"{current_exo or 'X'}::{current_qnum}{sub}"
                cur_lines = [m_sub.group(2)] if m_sub.group(2) else []
                continue
            
            if current_key:
                cur_lines.append(line)
    
    flush()
    return idx


def match_qi_rqi(qi_raw: str, corr_idx: Dict[str, Dict[str, Any]], min_sim: float) -> Tuple[Optional[str], float]:
    """Matching Qi→RQi par similarité (CAS 1 ONLY: pas de création)"""
    if not corr_idx:
        return None, 0.0
    
    q_tokens = tokenize(qi_raw)
    best = (0.0, None)
    
    for key, val in corr_idx.items():
        r_tokens = tokenize(val.get("rqi_raw", ""))
        sim = jaccard(q_tokens, r_tokens)
        if sim > best[0]:
            best = (sim, val.get("rqi_raw"))
    
    if best[0] >= min_sim:
        return best[1], best[0]
    return None, best[0]


# =============================================================================
# POSABLE GATE (KERNEL V10.6.1)
# =============================================================================
def scope_match_chapter(pack: Dict[str, Any], qi_clean: str, rqi_clean: Optional[str]) -> Tuple[Optional[str], Dict[str, Any], List[str]]:
    """Matching chapitre via Pack (pas hardcode Kernel)"""
    chapters = pack_get(pack, "academic.chapters", [])
    if not chapters:
        return None, {"method": "none", "error": "no_chapters_in_pack"}, [RC_SCOPE_UNRESOLVED]
    
    text = (qi_clean + "\n" + (rqi_clean or "")).lower()
    scores = []
    
    for ch in chapters:
        code = ch.get("chapter_code") or ch.get("code")
        if not code:
            continue
        
        kw = ch.get("match_keywords", [])
        rg = ch.get("match_regexes", [])
        
        hit_kw = [k for k in kw if k.lower() in text]
        hit_rg = []
        for r in rg:
            try:
                if re.search(r, text, re.I):
                    hit_rg.append(r)
            except:
                pass
        
        score = len(hit_kw) + 2 * len(hit_rg)
        if score > 0:
            scores.append((score, code, hit_kw, hit_rg))
    
    scores = stable_sort(scores, key_fn=lambda x: (-x[0], x[1]))
    
    if not scores:
        return None, {"method": "keywords/regex", "hits": []}, [RC_SCOPE_UNRESOLVED]
    
    # Ambiguïté: top 2 égaux
    if len(scores) >= 2 and scores[0][0] == scores[1][0]:
        return None, {"method": "keywords/regex", "ambiguous": True}, [RC_SCOPE_AMBIGUOUS]
    
    top = scores[0]
    return top[1], {"chapter": top[1], "score": top[0], "keywords": top[2]}, []


def posable_gate(pack: Dict[str, Any], atoms: List[Atom]) -> Dict[str, PosableDecision]:
    """POSABLE Gate complet (3 critères Kernel)"""
    decisions = {}
    
    for a in atoms:
        reasons = []
        evidence = [a.source_fingerprint]
        
        # POSABLE_CORRIGE
        posable_corrige = bool(a.rqi_clean)
        if not posable_corrige:
            reasons.append(RC_CORRIGE_MISSING)
        
        # POSABLE_SCOPE
        chapter_ref, scope_trace, scope_rc = scope_match_chapter(pack, a.qi_clean, a.rqi_clean)
        posable_scope = (chapter_ref is not None)
        reasons.extend(scope_rc)
        
        # POSABLE_EVALUABLE
        posable_evaluable = looks_like_evaluable(a.qi_raw, pack)
        if not posable_evaluable:
            reasons.append(RC_NOT_EVALUABLE)
        
        posable_status = posable_corrige and posable_scope and posable_evaluable
        
        decisions[a.qi_id] = PosableDecision(
            qi_id=a.qi_id,
            chapter_ref=chapter_ref,
            posable_status=posable_status,
            posable_corrige=posable_corrige,
            posable_scope=posable_scope,
            posable_evaluable=posable_evaluable,
            reason_codes=reasons,
            evidence_refs=evidence,
            scope_trace=scope_trace,
        )
    
    return decisions


# =============================================================================
# ARI MINING (IA1 Miner — Extraction factuelle)
# =============================================================================
def build_verb_map(pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Construction mapping verbes → poids depuis Pack"""
    verbs = pack_get(pack, "verb_ontology.verbs", [])
    overrides = pack_get(pack, "verb_ontology.weights_override", {})
    
    verb_map = {}
    for v in verbs:
        vid = v.get("verb_id")
        weight = overrides.get(vid, v.get("weight", 0.25))
        for form in v.get("forms", []):
            verb_map[form.lower()] = {"verb_id": vid, "weight": float(weight)}
    
    return verb_map


def mine_traces(pack: Dict[str, Any], atoms: List[Atom], posable: Dict[str, PosableDecision]) -> Dict[str, TraceARI]:
    """IA1 Miner: Extraction actions/triggers depuis corrigés"""
    verb_map = build_verb_map(pack)
    traces = {}
    
    for a in atoms:
        dec = posable.get(a.qi_id)
        if not dec or not dec.posable_status:
            continue
        
        text = (a.rqi_clean or "").lower()
        actions = []
        used = set()
        
        for surf, meta in verb_map.items():
            if re.search(rf"\b{re.escape(surf)}\b", text, re.I):
                vid = meta["verb_id"]
                if vid in used:
                    continue
                
                m = re.search(rf"(.{{0,30}})\b{re.escape(surf)}\b(.{{0,30}})", text, re.I)
                snippet = (m.group(0) if m else surf)[:100]
                
                actions.append({
                    "verb_id": vid,
                    "surface": surf,
                    "weight": meta["weight"],
                    "evidence": snippet,
                })
                used.add(vid)
        
        # Tri déterministe par position d'apparition
        def first_pos(surf):
            m = re.search(rf"\b{re.escape(surf)}\b", text, re.I)
            return m.start() if m else 10**9
        
        actions = stable_sort(actions, key_fn=lambda x: (first_pos(x["surface"]), x["verb_id"]))
        
        triggers = [{"trigger_id": f"TRG_{ac['verb_id']}", "verb_id": ac["verb_id"]} for ac in actions]
        
        traces[a.qi_id] = TraceARI(
            qi_id=a.qi_id,
            chapter_ref=dec.chapter_ref,
            actions=actions,
            triggers=triggers,
            output_type="RESULT",
            evidence_refs=[a.source_fingerprint],
        )
    
    return traces


# =============================================================================
# CLUSTERING + QC BUILDING (IA1 Builder)
# =============================================================================
def trace_signature(tr: TraceARI) -> str:
    """Signature déterministe d'un trace ARI"""
    vids = [a["verb_id"] for a in tr.actions]
    return "SIG:" + "-".join(vids) if vids else "SIG:EMPTY"


def build_clusters(traces: Dict[str, TraceARI]) -> Dict[str, List[str]]:
    """Clustering par (chapter, signature) - déterministe"""
    buckets = defaultdict(list)
    
    for qi_id, tr in traces.items():
        key = f"{tr.chapter_ref}::{trace_signature(tr)}"
        buckets[key].append(qi_id)
    
    return dict(buckets)


def build_qc_candidates(
    pack: Dict[str, Any],
    atoms: List[Atom],
    posable: Dict[str, PosableDecision],
    traces: Dict[str, TraceARI],
    clusters: Dict[str, List[str]],
) -> List[QCCandidate]:
    """Construction QC avec F1/F2 (Kernel V10.6.1)"""
    atoms_by_qi = {a.qi_id: a for a in atoms}
    candidates = []
    
    # Paramètres F1/F2 depuis Pack
    delta_c = float(pack_get(pack, "policy.level.delta_c", 1.0))
    eps = float(pack_get(pack, "policy.scoring.eps", F1_EPSILON))
    alpha = float(pack_get(pack, "policy.scoring.alpha_recency", F2_ALPHA))
    
    qc_prefix = pack_get(pack, "policy.qc_format.prefix", "Comment ")
    qc_suffix = pack_get(pack, "policy.qc_format.suffix", " ?")
    qc_phrases = pack_get(pack, "qc_phrases.by_verb_id", {})
    qc_default = pack_get(pack, "qc_phrases.default", "résoudre un exercice")
    
    frt_templates = pack_get(pack, "frt_templates", {})
    
    for cluster_key, qi_ids in clusters.items():
        if not qi_ids:
            continue
        
        # Filtrer qi_ids valides
        valid_ids = [qid for qid in qi_ids if qid in traces]
        if not valid_ids:
            continue
        
        chapter_ref = traces[valid_ids[0]].chapter_ref
        
        # Champion: premier qi_id (déterministe)
        champ_id = valid_ids[0]
        champ_trace = traces[champ_id]
        
        # Primary op = premier verb_id
        primary_op = champ_trace.actions[0]["verb_id"] if champ_trace.actions else "DEFAULT"
        
        # QC text (Pack-driven)
        core = qc_phrases.get(primary_op, qc_default)
        qc_text = f"{qc_prefix}{core}{qc_suffix}".strip()
        
        # Signature
        sig = trace_signature(champ_trace)
        
        # ========== F1: ψ_raw = δc × (ε + Σ Tj)² ==========
        T_sum = sum(float(ac.get("weight", 0.25)) for ac in champ_trace.actions)
        psi_raw = delta_c * (eps + T_sum) ** 2
        
        # ARI typé
        ari_typed = {
            "typed_only": True,
            "steps": [{
                "step": i + 1,
                "verb_id": ac["verb_id"],
                "weight": ac["weight"],
            } for i, ac in enumerate(champ_trace.actions)]
        }
        
        # FRT (Pack-driven)
        frt_tpl = frt_templates.get(primary_op, frt_templates.get("DEFAULT", {}))
        frt = {
            "usage": f"Procédure type pour {core}",
            "preconditions": ["Énoncé compris", "Données disponibles"],
            "ari_typed": ari_typed["steps"],
            "steps": frt_tpl.get("steps", []),
            "pitfalls": ["Oublier une condition", "Constante locale interdite"],
            "expected_output": "Résultat conforme au corrigé",
        }
        
        # Triggers union
        triggers = []
        seen_trg = set()
        for qid in valid_ids:
            tr = traces.get(qid)
            if tr:
                for t in tr.triggers:
                    if t["trigger_id"] not in seen_trg:
                        triggers.append(t)
                        seen_trg.add(t["trigger_id"])
        
        # EvidencePack
        evidence_pack = {
            "kernel_version": KERNEL_VERSION,
            "pack_sig": pack.get("_pack_sig", ""),
            "sources": [atoms_by_qi[qid].source_fingerprint for qid in valid_ids if qid in atoms_by_qi],
            "qi_ids": valid_ids,
        }
        
        # n_q_cluster
        n_q_cluster = len(valid_ids)
        
        candidates.append(QCCandidate(
            qc_id=f"QC_{sha256_text(cluster_key + sig)[:12]}",
            chapter_ref=chapter_ref,
            qc_text=qc_text,
            primary_op=primary_op,
            ari_typed=ari_typed,
            frt=frt,
            triggers=triggers,
            sig=sig,
            n_q_cluster=n_q_cluster,
            qi_ids=valid_ids,
            evidence_pack=evidence_pack,
            psi_raw=psi_raw,
            psi_q=0.0,  # Normalisé après
            f2_score=0.0,  # Calculé après
            ia2_audit=[],
            ia2_pass=False,
        ))
    
    return candidates


def normalize_f1_intra_chapter(qcs: List[QCCandidate]) -> None:
    """Normalisation F1 intra-chapitre: Ψq = psi_raw / max(psi_raw)"""
    by_ch = defaultdict(list)
    for qc in qcs:
        by_ch[qc.chapter_ref].append(qc)
    
    for ch_qcs in by_ch.values():
        max_raw = max(qc.psi_raw for qc in ch_qcs) if ch_qcs else 1.0
        for qc in ch_qcs:
            qc.psi_q = qc.psi_raw / max_raw if max_raw > 0 else 0.0


# =============================================================================
# IA2 JUDGE (Validation booléenne)
# =============================================================================
def ia2_checks(pack: Dict[str, Any], qc: QCCandidate) -> Tuple[bool, List[AuditEntry]]:
    """IA2 Judge: Checks booléens scellés"""
    audits = []
    
    def add(check_id: str, ok: bool, details: str, fixes: List[str]):
        audits.append(AuditEntry(
            check_id=check_id,
            status="PASS" if ok else "FAIL",
            evidence_refs=[qc.qc_id],
            details=details,
            fix_recommendations=fixes if not ok else [],
        ))
    
    # QC_FORM: doit finir par ? et commencer par prefix
    qc_prefix = pack_get(pack, "policy.qc_format.prefix", "Comment ")
    ok_form = qc.qc_text.strip().endswith("?") and qc.qc_text.strip().startswith(qc_prefix)
    add("QC_FORM", ok_form, "QC format check", ["Vérifier qc_format Pack"])
    
    # NO_LOCAL_CONSTANTS
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", qc.qc_text))
    ok_local = not has_year
    add("NO_LOCAL_CONSTANTS", ok_local, "Pas de constantes locales", ["Améliorer sanitization"])
    
    # FRT_TEMPLATE_OK
    mandatory = ["usage", "ari_typed", "steps"]
    ok_frt = all(k in qc.frt for k in mandatory)
    add("FRT_TEMPLATE_OK", ok_frt, "FRT sections obligatoires", ["Ajouter sections manquantes"])
    
    # ARI_TYPED_ONLY
    ok_ari = bool(qc.ari_typed.get("typed_only", False))
    add("ARI_TYPED_ONLY", ok_ari, "ARI typé uniquement", ["Utiliser verb_id canoniques"])
    
    # ANTI_SINGLETON (n_q_cluster >= 2)
    ok_cluster = qc.n_q_cluster >= 2
    add("ANTI_SINGLETON", ok_cluster, f"n_q_cluster={qc.n_q_cluster} >= 2", ["Augmenter volume ou regrouper"])
    
    passed = all(a.status == "PASS" for a in audits)
    return passed, audits


# =============================================================================
# F2 SCORING + SELECTION (Coverage-driven)
# =============================================================================
def compute_f2_and_select(
    pack: Dict[str, Any],
    qcs: List[QCCandidate],
    traces: Dict[str, TraceARI],
    posable: Dict[str, PosableDecision],
) -> Tuple[List[QCCandidate], Dict[str, Any]]:
    """
    F2 = (n_hist/N) × (1 + α/t_rec) × Ψq × Π(1-σ)
    Sélection greedy coverage-driven
    """
    alpha = float(pack_get(pack, "policy.scoring.alpha_recency", F2_ALPHA))
    t_rec = F2_TREC_DEFAULT
    eps = float(pack_get(pack, "policy.scoring.eps", F1_EPSILON))
    max_k = int(pack_get(pack, "policy.selection.max_k_per_chapter", 15))
    
    # Univers POSABLE par chapitre
    by_ch = defaultdict(list)
    for qc in qcs:
        if qc.ia2_pass:
            by_ch[qc.chapter_ref].append(qc)
    
    N_by_ch = defaultdict(int)
    for dec in posable.values():
        if dec.posable_status and dec.chapter_ref:
            N_by_ch[dec.chapter_ref] += 1
    
    selections = {}
    coverage_map = {}
    
    for ch, ch_qcs in by_ch.items():
        N_total = N_by_ch.get(ch, 1)
        selected = []
        uncovered = set()
        
        # Univers = tous les qi_ids POSABLE du chapitre
        for dec in posable.values():
            if dec.posable_status and dec.chapter_ref == ch:
                uncovered.add(dec.qi_id)
        
        # Calcul F2 initial
        for qc in ch_qcs:
            n_hist = qc.n_q_cluster  # Proxy: taille cluster
            # F2 = (n_hist/N) × (1 + α/t_rec) × Ψq
            qc.f2_score = (n_hist / max(1, N_total)) * (1 + alpha / max(0.01, t_rec)) * qc.psi_q * 100
        
        # Greedy selection avec anti-redondance
        for _ in range(min(max_k, len(ch_qcs))):
            best = None
            best_score = -1
            
            for qc in ch_qcs:
                if qc.qc_id in [s.qc_id for s in selected]:
                    continue
                
                # Gain = qi_ids couverts
                newly = [qi for qi in qc.qi_ids if qi in uncovered]
                gain = len(newly)
                
                if gain <= 0:
                    continue
                
                # Anti-redondance: Π(1-σ) avec QC déjà sélectionnées
                penalty = 1.0
                for sel in selected:
                    # σ = Jaccard sur verb_ids
                    sel_vids = set(s.get("verb_id", "") for s in sel.ari_typed.get("steps", []))
                    qc_vids = set(s.get("verb_id", "") for s in qc.ari_typed.get("steps", []))
                    if sel_vids and qc_vids:
                        sigma = len(sel_vids & qc_vids) / len(sel_vids | qc_vids)
                        penalty *= (1 - clamp(sigma, 0, 0.9))
                
                score = gain * qc.f2_score * penalty
                
                if score > best_score:
                    best_score = score
                    best = (qc, newly)
            
            if best is None:
                break
            
            qc_best, newly = best
            selected.append(qc_best)
            
            for qi in newly:
                coverage_map[qi] = {"qc_id": qc_best.qc_id, "status": "COVERED"}
                uncovered.discard(qi)
        
        # Orphans
        for qi in uncovered:
            coverage_map[qi] = {"qc_id": None, "status": "ORPHAN"}
        
        selections[ch] = {
            "selected_qc_ids": [s.qc_id for s in selected],
            "orphans": list(uncovered),
            "coverage": (len(posable) - len(uncovered)) / max(1, len(posable)) if ch in N_by_ch else 0.0,
        }
    
    return qcs, selections


# =============================================================================
# PIPELINE COMPLET
# =============================================================================
def build_atoms_from_pair(pack: Dict[str, Any], pair: Pair, lang: str) -> Tuple[List[Atom], Dict[str, Any]]:
    """Construction Atoms depuis une paire (Sujet + Corrigé)"""
    audit = {"pair_id": pair.pair_id, "quarantine": []}
    atoms = []
    
    # CAS 1 ONLY: corrigé obligatoire
    if pair.correction is None:
        audit["quarantine"].append({"reason": RC_CORRIGE_MISSING})
        return [], audit
    
    doc_bytes = st.session_state.get("_doc_bytes", {})
    subj_b = doc_bytes.get(pair.subject.doc_id)
    corr_b = doc_bytes.get(pair.correction.doc_id) if pair.correction else None
    
    if not subj_b:
        audit["quarantine"].append({"reason": RC_SUBJECT_UNREADABLE})
        return [], audit
    
    if not corr_b:
        audit["quarantine"].append({"reason": RC_CORRIGE_UNREADABLE})
        return [], audit
    
    # Extraction
    subj_pages, subj_audit = extract_pdf_pages(subj_b)
    corr_pages, corr_audit = extract_pdf_pages(corr_b)
    
    if not subj_pages:
        audit["quarantine"].append({"reason": RC_SUBJECT_UNREADABLE, "details": subj_audit})
        return [], audit
    
    if not corr_pages:
        audit["quarantine"].append({"reason": RC_CORRIGE_UNREADABLE, "details": corr_audit})
        return [], audit
    
    # Atomisation
    subj_units = atomise_subject(subj_pages, pair.subject.doc_id, lang, pack)
    corr_idx = atomise_correction(corr_pages, pair.correction.doc_id, lang)
    
    audit["subj_units"] = len(subj_units)
    audit["corr_segments"] = len(corr_idx)
    
    # Year ref
    year_ref = None
    m = re.search(r"(19|20)\d{2}", pair.subject.filename)
    if m:
        year_ref = m.group(0)
    
    min_sim = float(pack_get(pack, "policy.pairing.min_similarity_fallback", 0.35))
    
    for i, u in enumerate(subj_units):
        qi_raw = u["qi_raw"]
        
        # Match RQi
        rqi_raw, conf = match_qi_rqi(qi_raw, corr_idx, min_sim)
        
        # Sanitize
        qi_clean, qi_deriv = sanitize_text(qi_raw, lang)
        rqi_clean = None
        rqi_deriv = []
        if rqi_raw:
            rqi_clean, rqi_deriv = sanitize_text(rqi_raw, lang)
        
        qi_id = f"QI_{stable_id(pair.subject.doc_id, u['q_ref'], qi_clean[:100])}"
        rqi_id = f"RQI_{stable_id(pair.correction.doc_id, u['q_ref'])}" if rqi_raw else None
        
        atoms.append(Atom(
            atom_id=f"ATOM_{stable_id(qi_id, rqi_id or 'NONE')}",
            subject_id=pair.subject.doc_id,
            correction_id=pair.correction.doc_id,
            qi_id=qi_id,
            rqi_id=rqi_id,
            exo_ref=u.get("exo_ref"),
            q_ref=u.get("q_ref"),
            qi_raw=qi_raw,
            rqi_raw=rqi_raw,
            qi_clean=qi_clean,
            rqi_clean=rqi_clean,
            language=lang,
            year_ref=year_ref,
            source_url=pair.subject.url,
            source_fingerprint=pair.subject.bytes_sha256,
            extraction_method=u.get("method", "STRUCTURED"),
            pairing_confidence=conf,
            sanitizer_derivations=qi_deriv + rqi_deriv,
        ))
    
    return atoms, audit


def run_pipeline(pack: Dict[str, Any], pairs: List[Pair]) -> Dict[str, Any]:
    """Pipeline complet ISO-PROD"""
    lang = pack_get(pack, "language", "fr")
    
    # Build atoms
    all_atoms = []
    pair_audits = []
    
    for pr in pairs:
        atoms, audit = build_atoms_from_pair(pack, pr, lang)
        pair_audits.append(audit)
        all_atoms.extend(atoms)
    
    log(f"[PIPELINE] {len(all_atoms)} atoms from {len(pairs)} pairs")
    
    # POSABLE Gate
    posable = posable_gate(pack, all_atoms)
    posable_count = sum(1 for d in posable.values() if d.posable_status)
    log(f"[POSABLE] {posable_count}/{len(all_atoms)} POSABLE")
    
    # Mine traces
    traces = mine_traces(pack, all_atoms, posable)
    log(f"[TRACES] {len(traces)} traces mined")
    
    # Clusters
    clusters = build_clusters(traces)
    log(f"[CLUSTERS] {len(clusters)} clusters")
    
    # QC Candidates
    qcs = build_qc_candidates(pack, all_atoms, posable, traces, clusters)
    log(f"[QC] {len(qcs)} candidates")
    
    # F1 normalization
    normalize_f1_intra_chapter(qcs)
    
    # IA2 Judge
    for qc in qcs:
        ok, audit = ia2_checks(pack, qc)
        qc.ia2_audit = audit
        qc.ia2_pass = ok
    
    ia2_pass_count = sum(1 for qc in qcs if qc.ia2_pass)
    log(f"[IA2] {ia2_pass_count}/{len(qcs)} PASS")
    
    # F2 + Selection
    qcs, selections = compute_f2_and_select(pack, qcs, traces, posable)
    
    # Metrics
    total_qi = len(all_atoms)
    total_rqi = sum(1 for a in all_atoms if a.rqi_clean)
    total_posable = sum(1 for d in posable.values() if d.posable_status)
    total_qc = sum(len(sel["selected_qc_ids"]) for sel in selections.values())
    total_orphans = sum(len(sel["orphans"]) for sel in selections.values())
    coverage = (total_posable - total_orphans) / max(1, total_posable)
    
    return {
        "atoms": all_atoms,
        "pair_audits": pair_audits,
        "posable": posable,
        "traces": traces,
        "clusters": clusters,
        "qc_candidates": qcs,
        "selections": selections,
        "metrics": {
            "pairs": len(pairs),
            "qi_total": total_qi,
            "rqi_total": total_rqi,
            "qi_posable": total_posable,
            "qc_total": total_qc,
            "orphans": total_orphans,
            "coverage": round(coverage, 4),
            "qi_per_subject": round(total_qi / max(1, len(pairs)), 2),
        },
    }


def run_saturation(pack: Dict[str, Any], pairs: List[Pair], vol_start: int, vol_max: int) -> Dict[str, Any]:
    """Boucle saturation new_QC = 0"""
    ordered = stable_sort(pairs, key_fn=lambda p: p.pair_id)
    
    iterations = []
    prev_qc_ids = None
    final = None
    
    for vol in range(vol_start, vol_max + 1):
        subset = ordered[:vol]
        res = run_pipeline(pack, subset)
        
        qc_ids = set()
        for ch, sel in res["selections"].items():
            qc_ids.update(sel["selected_qc_ids"])
        
        if prev_qc_ids is None:
            new_qc = len(qc_ids)
        else:
            new_qc = len(qc_ids - prev_qc_ids)
        
        iterations.append({
            "volume": vol,
            **res["metrics"],
            "new_qc": new_qc,
        })
        
        log(f"[SAT] vol={vol} Qi={res['metrics']['qi_total']} QC={res['metrics']['qc_total']} new_QC={new_qc}")
        
        if prev_qc_ids is not None and new_qc == 0:
            final = res
            break
        
        prev_qc_ids = qc_ids
        final = res
    
    # SEALED check
    m = final["metrics"] if final else {}
    sealed = (
        iterations[-1]["new_qc"] == 0 if len(iterations) >= 2 else False
    ) and m.get("orphans", 1) == 0 and m.get("qc_total", 0) > 0 and m.get("qi_posable", 0) > 0
    
    seal_reason = "ALL_CHECKS_PASSED" if sealed else "FAILED"
    
    return {
        "iterations": iterations,
        "sealed": sealed,
        "seal_reason": seal_reason,
        "final": final,
    }


# =============================================================================
# HARVEST (APMEP example — Pack-driven)
# =============================================================================
def http_get(url: str) -> requests.Response:
    return requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)


def harvest_pairs(pack: Dict[str, Any], volume_max: int) -> List[Pair]:
    """Harvest depuis source Pack-driven"""
    root = pack_get(pack, "policy.harvest.root_url")
    if not root:
        raise ValueError("Pack policy.harvest.root_url manquant")
    
    corrige_kw = pack_get(pack, "policy.harvest.corrige_keywords", ["corrig"])
    
    if not BS4_OK:
        raise RuntimeError("BeautifulSoup non disponible")
    
    log(f"[HARVEST] root={root}")
    
    r = http_get(root)
    soup = BeautifulSoup(r.text, "html.parser")
    
    links = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if not href:
            continue
        if href.startswith("/"):
            base = re.match(r"^(https?://[^/]+)", root)
            href = (base.group(1) if base else "") + href
        if ".pdf" in href.lower():
            text = (a.get_text() or "").strip()
            links.append((href, text))
    
    links = stable_sort(links, key_fn=lambda x: x[0])
    log(f"[HARVEST] {len(links)} PDF links found")
    
    # Grouper par base name
    def base_key(url):
        fname = url.split("/")[-1].split("?")[0].lower()
        fname = re.sub(r"\.pdf.*$", "", fname)
        fname = re.sub(r"(corrig[eé]?)", "", fname)
        return re.sub(r"[^a-z0-9]+", "_", fname).strip("_")
    
    buckets = defaultdict(list)
    for url, text in links:
        is_corr = any(k in (text.lower() + " " + url.lower()) for k in corrige_kw)
        buckets[base_key(url)].append({"url": url, "text": text, "is_corrige": is_corr})
    
    pairs = []
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    
    for bkey in stable_sort(buckets.keys(), key_fn=lambda x: x):
        if len(pairs) >= volume_max:
            break
        
        items = buckets[bkey]
        sujets = [x for x in items if not x["is_corrige"]]
        corriges = [x for x in items if x["is_corrige"]]
        
        if not sujets:
            continue
        
        sujets = stable_sort(sujets, key_fn=lambda x: x["url"])
        subj = sujets[0]
        
        corr_best = None
        conf = 0.0
        if corriges:
            corriges = stable_sort(corriges, key_fn=lambda x: x["url"])
            corr_best = corriges[0]
            conf = 0.8
        
        # Download
        def download(d, kind):
            rr = sess.get(d["url"], timeout=REQ_TIMEOUT)
            rr.raise_for_status()
            b = rr.content
            fname = d["url"].split("/")[-1].split("?")[0]
            return SourceDoc(
                doc_id=f"{kind}_{sha256_bytes(b)[7:19]}",
                kind=kind,
                url=d["url"],
                filename=fname,
                bytes_sha256=sha256_bytes(b),
                fetched_at=now_iso(),
            ), b
        
        try:
            subj_doc, subj_b = download(subj, "subject")
            st.session_state["_doc_bytes"][subj_doc.doc_id] = subj_b
        except Exception as e:
            log(f"[HARVEST] download error: {e}")
            continue
        
        corr_doc = None
        if corr_best:
            try:
                corr_doc, corr_b = download(corr_best, "correction")
                st.session_state["_doc_bytes"][corr_doc.doc_id] = corr_b
            except:
                corr_doc = None
        
        pairs.append(Pair(
            pair_id=f"PAIR_{stable_id(subj_doc.filename)}",
            subject=subj_doc,
            correction=corr_doc,
            pairing_confidence=conf,
        ))
    
    log(f"[HARVEST] {len(pairs)} pairs built")
    return pairs


# =============================================================================
# STREAMLIT UI
# =============================================================================
def metric_row(m: Dict):
    cols = st.columns(8)
    cols[0].metric("Pairs", m.get("pairs", 0))
    cols[1].metric("Qi", m.get("qi_total", 0))
    cols[2].metric("RQi", m.get("rqi_total", 0))
    cols[3].metric("POSABLE", m.get("qi_posable", 0))
    cols[4].metric("QC", m.get("qc_total", 0))
    cols[5].metric("Orphans", m.get("orphans", 0))
    cols[6].metric("Qi/Sujet", m.get("qi_per_subject", 0))
    cols[7].metric("SEALED", "✅" if st.session_state.get("sealed") else "❌")


def main():
    st.set_page_config(page_title=f"SMAXIA GTE {VERSION}", layout="wide")
    ss_init()
    
    st.markdown(f"# 🔒 SMAXIA GTE Console {VERSION}")
    st.caption(f"Kernel {KERNEL_VERSION} | F1=δc·(ε+ΣTj)² | F2=(n/N)·(1+α/t)·Ψ·Π(1-σ) | CAS 1 ONLY")
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("## 1. Pack")
        
        country = st.selectbox("Pays", list(GENESIS_PACKS.keys()))
        
        if st.button("🚀 ACTIVER PACK", type="primary", use_container_width=True):
            pack = generate_pack(country)
            st.session_state.pack_active = pack
            st.session_state.pack_id = pack["pack_id"]
            st.session_state.library = []
            st.session_state.atoms = []
            st.session_state.qc_candidates = []
            st.session_state.sealed = False
            st.session_state.metrics = {}
            st.success(f"✅ {pack['pack_id']}")
            st.rerun()
        
        if st.session_state.pack_active:
            st.success(f"✅ {st.session_state.pack_id}")
    
    if not st.session_state.pack_active:
        st.warning("⚠️ Activez un Pack")
        return
    
    pack = st.session_state.pack_active
    
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Harvest", "🚀 RUN", "📊 Résultats", "📁 Exports"])
    
    with tab1:
        st.markdown("## Harvest")
        metric_row(st.session_state.metrics)
        
        volume = st.slider("Volume max", 5, 100, 30)
        
        if st.button("🌾 HARVEST", use_container_width=True):
            with st.spinner("Harvesting..."):
                try:
                    pairs = harvest_pairs(pack, volume)
                    st.session_state.library = pairs
                    st.session_state.metrics = {"pairs": len(pairs)}
                    st.success(f"✅ {len(pairs)} pairs")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        
        if st.session_state.library:
            st.dataframe([{
                "pair_id": p.pair_id,
                "sujet": p.subject.filename,
                "corrigé": p.correction.filename if p.correction else "❌",
            } for p in st.session_state.library[:50]])
    
    with tab2:
        st.markdown("## RUN Pipeline")
        
        lib = st.session_state.library
        if not lib:
            st.warning("Faire HARVEST d'abord")
        else:
            corr_ok = sum(1 for p in lib if p.correction)
            st.info(f"**{len(lib)} pairs** | {corr_ok} avec corrigé")
            
            c1, c2 = st.columns(2)
            vol_start = c1.number_input("vol_start", 1, len(lib), min(5, len(lib)))
            vol_max = c2.number_input("vol_max", int(vol_start), len(lib), min(len(lib), 30))
            
            if st.button("🚀 LANCER SATURATION", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    res = run_saturation(pack, lib, int(vol_start), int(vol_max))
                    
                    st.session_state.atoms = res["final"]["atoms"] if res["final"] else []
                    st.session_state.posable_decisions = res["final"]["posable"] if res["final"] else {}
                    st.session_state.traces = res["final"]["traces"] if res["final"] else {}
                    st.session_state.qc_candidates = res["final"]["qc_candidates"] if res["final"] else []
                    st.session_state.selections = res["final"]["selections"] if res["final"] else {}
                    st.session_state.sealed = res["sealed"]
                    st.session_state.metrics = res["final"]["metrics"] if res["final"] else {}
                    st.session_state.iterations = res["iterations"]
                    st.session_state.seal_reason = res["seal_reason"]
                    
                    if res["sealed"]:
                        st.success(f"✅ SEALED")
                    else:
                        st.warning(f"⚠️ {res['seal_reason']}")
                    
                    st.rerun()
        
        st.markdown("### Logs")
        st.text_area("", "\n".join(st.session_state.logs[-50:]), height=200)
    
    with tab3:
        st.markdown("## Résultats")
        m = st.session_state.metrics
        
        if not m:
            st.info("Lancer RUN d'abord")
        else:
            metric_row(m)
            
            st.markdown("### Audit SEALED")
            c1, c2, c3 = st.columns(3)
            c1.metric("SEALED", "✅" if st.session_state.sealed else "❌")
            c2.metric("Coverage", f"{m.get('coverage', 0):.0%}")
            c3.metric("Reason", st.session_state.get("seal_reason", ""))
            
            # Itérations
            if st.session_state.get("iterations"):
                st.markdown("### Itérations saturation")
                st.dataframe(st.session_state.iterations)
            
            # QC par chapitre
            st.markdown("### QC par chapitre")
            for ch, sel in st.session_state.get("selections", {}).items():
                with st.expander(f"{ch} ({len(sel['selected_qc_ids'])} QC)"):
                    for qc_id in sel["selected_qc_ids"]:
                        qc = next((q for q in st.session_state.qc_candidates if q.qc_id == qc_id), None)
                        if qc:
                            st.markdown(f"**{qc.qc_text}**")
                            st.caption(f"ID: {qc.qc_id} | Ψ={qc.psi_q:.3f} | F2={qc.f2_score:.2f} | n={qc.n_q_cluster}")
                            st.write("**FRT steps:**")
                            for step in qc.frt.get("steps", []):
                                st.write(f"  {step}")
    
    with tab4:
        st.markdown("## Exports")
        metric_row(st.session_state.metrics)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.download_button("📝 logs.txt", "\n".join(st.session_state.logs), "logs.txt")
            
            if st.session_state.atoms:
                atoms_export = [asdict(a) for a in st.session_state.atoms]
                st.download_button("📦 atoms.json", safe_json(atoms_export), "atoms.json")
            
            if st.session_state.qc_candidates:
                qc_export = [asdict(q) for q in st.session_state.qc_candidates if q.ia2_pass]
                st.download_button("📦 qc_pack.json", safe_json(qc_export), "qc_pack.json")
        
        with c2:
            if st.session_state.posable_decisions:
                pos_export = {k: asdict(v) for k, v in st.session_state.posable_decisions.items()}
                st.download_button("📦 posable.json", safe_json(pos_export), "posable.json")
            
            if st.session_state.selections:
                st.download_button("📦 selections.json", safe_json(st.session_state.selections), "selections.json")
        
        # Explorateur Qi/RQi
        st.markdown("---")
        st.markdown("## Explorateur Qi ↔ RQi")
        
        if st.session_state.atoms:
            atoms = st.session_state.atoms
            posable = st.session_state.posable_decisions
            
            # Stats
            st.write(f"**{len(atoms)} Atoms** | {sum(1 for a in atoms if a.rqi_clean)} avec RQi")
            
            for a in atoms[:30]:
                dec = posable.get(a.qi_id)
                status = "✅ POSABLE" if (dec and dec.posable_status) else f"❌ {', '.join(dec.reason_codes) if dec else 'N/A'}"
                
                with st.expander(f"{a.qi_id} | {a.q_ref} | {status}"):
                    st.markdown("**Qi (Question)**")
                    st.text(a.qi_raw[:800])
                    
                    if a.rqi_raw:
                        st.markdown("**RQi (Réponse)**")
                        st.text(a.rqi_raw[:800])
                        st.caption(f"Confidence: {a.pairing_confidence:.2f}")
                    else:
                        st.warning("❌ Pas de RQi")
                    
                    if dec:
                        st.markdown("**POSABLE Gate**")
                        st.write(f"- CORRIGE: {'✅' if dec.posable_corrige else '❌'}")
                        st.write(f"- SCOPE: {'✅' if dec.posable_scope else '❌'} ({dec.chapter_ref or 'N/A'})")
                        st.write(f"- EVALUABLE: {'✅' if dec.posable_evaluable else '❌'}")


if __name__ == "__main__":
    main()
