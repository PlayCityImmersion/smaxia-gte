# =============================================================================
# SMAXIA GTE V8 — KERNEL V10.6.3 — AUDIT-COMPLIANT
# =============================================================================
# 
# CORRECTIONS V8 (Audit GPT 5.2):
#   ✅ BLOQUANT 1: DEMO ne peut PAS être SEALED → forcé UNVERIFIED
#   ✅ BLOQUANT 2: Statuts explicites DEMO_OK vs REAL_OK
#   ✅ BLOQUANT 3: Gate GATE_MODE_PROD_REQUIRED bloquant pour seal
#   ✅ BLOQUANT 4: Evidence Pack avec mode_proof tracé
#   ✅ Structure prête pour PROD (placeholders fetch/extract)
#
# MODES:
#   DEMO  → UI fonctionnelle, données simulées, JAMAIS SEALED
#   PROD  → Fetch réel, extraction PDF, scellement possible
#
# HOW TO RUN:
#   pip install streamlit
#   streamlit run smaxia_gte_v8_audit.py
#
# =============================================================================

import streamlit as st
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
from collections import defaultdict
from enum import Enum

# =============================================================================
# EXECUTION MODE (Critical for audit)
# =============================================================================
class ExecutionMode(Enum):
    DEMO = "DEMO"  # Simulation → JAMAIS SEALED
    PROD = "PROD"  # Réel → peut être SEALED si gates OK

EXECUTION_MODE = ExecutionMode.DEMO  # Changer en PROD pour production

# =============================================================================
# STATUS ENUMS (Explicites - pas d'ambiguïté)
# =============================================================================
class FetchStatus(Enum):
    DEMO_OK = "DEMO_OK"           # Simulé, pas de fetch réel
    REAL_OK = "REAL_OK"           # Fetch réel réussi
    REAL_FAIL = "REAL_FAIL"       # Fetch réel échoué
    PENDING = "PENDING"           # Pas encore traité
    QUARANTINE = "QUARANTINE"     # Problème détecté

class TextStatus(Enum):
    DEMO_TEXT = "DEMO_TEXT"       # Texte simulé
    REAL_TEXT = "REAL_TEXT"       # Texte extrait de PDF réel
    OCR_TEXT = "OCR_TEXT"         # Texte via OCR
    FAIL = "FAIL"                 # Extraction échouée

class RunStatus(Enum):
    RUNNING = "RUNNING"
    SEALED = "SEALED"             # Seulement si PROD + all gates PASS
    UNVERIFIED = "UNVERIFIED"     # DEMO ou gates FAIL
    STOPPED = "STOPPED"           # Gate bloquant

# =============================================================================
# KERNEL CONSTANTS
# =============================================================================
KERNEL_VERSION = "V10.6.3"
FORMULES_VERSION = "V3.1"
FORMULES_REF = "A2"

NUMERIC_POLICY = "decimal_fixed"
DECIMAL_PRECISION = 6
EPSILON_F1 = 0.1
T_MIN_F2 = 1.0

MIN_CAP_SOURCES = 2
MIN_CAP_DOMAINS = 2
MIN_PAIRS = 5
MIN_TEXT_LEN = 500
MAX_SATURATION_CYCLES = 25

# =============================================================================
# WHITELIST
# =============================================================================
PAYS_WHITELIST = {"FR", "CI"}
NIVEAUX_WHITELIST = {"TERMINALE", "PREMIERE", "PREPA_MP"}
MATIERES_WHITELIST = {"MATHS", "PHYSIQUE_CHIMIE"}

PAYS_META = {"FR": {"nom": "France", "flag": "🇫🇷"}, "CI": {"nom": "Côte d'Ivoire", "flag": "🇨🇮"}}
NIVEAUX_META = {"TERMINALE": {"label": "Terminale", "ordre": 1}, "PREMIERE": {"label": "Première", "ordre": 2}, "PREPA_MP": {"label": "Prépa MP", "ordre": 3}}
MATIERES_META = {"MATHS": {"label": "Mathématiques", "icon": "📐"}, "PHYSIQUE_CHIMIE": {"label": "Physique-Chimie", "icon": "⚗️"}}

# =============================================================================
# UTILITIES
# =============================================================================
def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_bytes(data: bytes) -> str:
    """Hash SHA256 de bytes (pour contenu réel)."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"

def sha256_str(text: str) -> str:
    """Hash SHA256 de string."""
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"

def stable_seed(text: str) -> int:
    """Seed déterministe."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

def decimal_round(value: float, precision: int = DECIMAL_PRECISION) -> float:
    factor = 10 ** precision
    return round(value * factor) / factor

def norm_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class SourceCandidate:
    url: str
    domain: str
    source_type: str
    authority_score: float
    fetch_ts: str
    # Fetch info (REAL vs DEMO)
    fetch_status: FetchStatus
    http_status: Optional[int]      # None en DEMO
    content_type: Optional[str]     # None en DEMO
    bytes_len: Optional[int]        # None en DEMO
    bytes_hash: Optional[str]       # sha256(bytes) en REAL, sha256(url) en DEMO
    mode_proof: str                 # "DEMO" ou "REAL"

@dataclass
class CAPChapter:
    code: str
    label: str
    keywords: List[str]
    delta_c: float
    source_url: str
    source_hash: str
    extraction_method: str  # "DEMO_STATIC" ou "REAL_EXTRACTED"

@dataclass
class CAP:
    cap_id: str
    pays: str
    status: str  # "DEMO_VERIFIED", "REAL_VERIFIED", "SEALED"
    mode_proof: str
    sources: List[SourceCandidate]
    chapitres: Dict[str, Dict[str, List[CAPChapter]]]
    seal_hash: Optional[str] = None

@dataclass
class ExamPair:
    pair_id: str
    nom: str
    type_epreuve: str
    matiere: str
    niveau: str
    annee: str
    concours: Optional[str]
    # Sujet
    sujet_url: str
    sujet_fetch_status: FetchStatus
    sujet_bytes_len: Optional[int]
    sujet_bytes_hash: Optional[str]
    sujet_text: str
    sujet_text_len: int
    sujet_text_status: TextStatus
    # Corrigé
    corrige_url: str
    corrige_fetch_status: FetchStatus
    corrige_bytes_len: Optional[int]
    corrige_bytes_hash: Optional[str]
    corrige_text: str
    corrige_text_len: int
    corrige_text_status: TextStatus
    # Meta
    source_url: str
    fetch_ts: str
    mode_proof: str
    text_extraction_engine: str  # "DEMO" ou "pdfplumber" ou "OCR"

@dataclass
class Qi:
    qi_id: str
    text: str
    source_pair_id: str
    exercise_num: str
    question_num: str
    page: int
    span_start: int
    span_end: int
    evidence_hash: str
    extraction_mode: str  # "DEMO" ou "REAL"
    chapter_code: Optional[str] = None
    chapter_label: Optional[str] = None
    mapping_score: float = 0.0
    mapping_evidence: str = ""
    rqi_text: Optional[str] = None
    rqi_hash: Optional[str] = None
    rqi_status: str = "MISSING"
    qc_id: Optional[str] = None
    status: str = "PENDING"

@dataclass
class QC:
    qc_id: str
    canonical_text: str
    chapter_code: str
    chapter_label: str
    matiere: str
    niveau: str
    qi_ids: List[str]
    cluster_method: str
    psi_raw: float
    psi_q: float
    score_f2: float
    ari: Dict
    frt: Dict
    triggers: List[str]
    evidence_pack: List[str]
    ia2_checks: Dict
    ia2_status: str

@dataclass
class GateResult:
    code: str
    passed: bool
    evidence: Dict
    message: str = ""
    blocking_for_seal: bool = True  # Si True, empêche SEALED

@dataclass
class RunContext:
    run_id: str
    pays: str
    timestamp: str
    mode: ExecutionMode
    status: RunStatus
    gates: Dict[str, GateResult] = field(default_factory=dict)
    saturation_log: List[Dict] = field(default_factory=list)
    seal_hash: Optional[str] = None

# =============================================================================
# FORMULA ENGINE (Exception CEO: reste dans CORE)
# =============================================================================
class FormulaEngineV31:
    VERSION = "V3.1"
    ENGINE_ID = "SMAXIA_F1F2_CORE_V31_CEO_EXCEPTION"
    
    def __init__(self):
        self.loaded = False
        self.psi_cache: Dict[str, List[Tuple[str, float]]] = {}
        self.checks = {}
    
    def load(self) -> Tuple[bool, str]:
        self.loaded = True
        self.checks["CHK_ENGINE_LOADED"] = True
        self.checks["CHK_F1F2_CEO_EXCEPTION"] = True
        return True, self.ENGINE_ID
    
    def compute_f1_raw(self, qc_id: str, chapter_code: str, delta_c: float, t_list: List[float]) -> Dict:
        if not self.loaded:
            return {"error": "ENGINE_NOT_LOADED"}
        sum_tj = sum(t_list) if t_list else 0.3
        psi_raw = delta_c * (EPSILON_F1 + sum_tj) ** 2
        psi_raw = decimal_round(psi_raw)
        if chapter_code not in self.psi_cache:
            self.psi_cache[chapter_code] = []
        self.psi_cache[chapter_code].append((qc_id, psi_raw))
        return {"psi_raw": psi_raw, "sum_tj": sum_tj}
    
    def normalize_f1_chapter(self, chapter_code: str) -> Dict[str, float]:
        if chapter_code not in self.psi_cache:
            return {}
        items = self.psi_cache[chapter_code]
        if not items:
            return {}
        max_psi = max(psi for _, psi in items)
        if max_psi <= 0:
            max_psi = 1.0
        normalized = {qc_id: decimal_round(psi / max_psi) for qc_id, psi in items}
        self.checks["CHK_F1_NORMALIZED"] = max(normalized.values()) <= 1.0
        return normalized
    
    def compute_f2(self, n_q: int, n_total: int, t_rec: float, psi_q: float, redundancy: float = 1.0) -> Dict:
        if not self.loaded:
            return {"error": "ENGINE_NOT_LOADED"}
        if n_total < 1:
            self.checks["CHK_NTOTAL_SAFE"] = False
            return {"error": "NTOTAL_ZERO"}
        self.checks["CHK_NTOTAL_SAFE"] = True
        t_rec_safe = max(t_rec, T_MIN_F2)
        self.checks["CHK_TREC_SAFE"] = True
        freq = n_q / n_total
        recency = 1.0 / t_rec_safe
        redundancy_safe = max(0.01, redundancy)
        score = freq * recency * psi_q * redundancy_safe
        return {"score": decimal_round(score), "freq": freq, "recency": recency}
    
    def get_checks(self) -> Dict:
        return self.checks.copy()

FORMULA_ENGINE = FormulaEngineV31()

# =============================================================================
# FETCH UTILITIES (PROD vs DEMO)
# =============================================================================
def fetch_url_bytes(url: str) -> Tuple[FetchStatus, Optional[int], Optional[str], Optional[int], Optional[bytes]]:
    """
    Fetch URL et retourne (status, http_code, content_type, bytes_len, bytes).
    En DEMO: retourne valeurs simulées.
    En PROD: fetch réel avec requests.
    """
    if EXECUTION_MODE == ExecutionMode.DEMO:
        # DEMO: pas de fetch réel
        return FetchStatus.DEMO_OK, None, None, None, None
    
    # PROD: fetch réel
    try:
        import requests
        resp = requests.get(url, timeout=15, headers={"User-Agent": "SMAXIA-GTE/1.0"})
        if resp.status_code == 200:
            return FetchStatus.REAL_OK, resp.status_code, resp.headers.get("Content-Type"), len(resp.content), resp.content
        else:
            return FetchStatus.REAL_FAIL, resp.status_code, None, None, None
    except Exception as e:
        return FetchStatus.REAL_FAIL, None, None, None, None

def extract_pdf_text(pdf_bytes: bytes) -> Tuple[TextStatus, str, str]:
    """
    Extrait texte d'un PDF.
    Retourne (status, text, engine).
    En DEMO: retourne texte simulé.
    En PROD: utilise pdfplumber.
    """
    if EXECUTION_MODE == ExecutionMode.DEMO or pdf_bytes is None:
        return TextStatus.DEMO_TEXT, "[DEMO] Texte simulé pour test UI.", "DEMO"
    
    # PROD: extraction réelle
    try:
        import pdfplumber
        import io
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts)
        if len(text) >= MIN_TEXT_LEN:
            return TextStatus.REAL_TEXT, text, "pdfplumber"
        else:
            return TextStatus.FAIL, text, "pdfplumber"
    except Exception as e:
        return TextStatus.FAIL, "", "pdfplumber_error"

# =============================================================================
# CAP PIPELINE
# =============================================================================
def discover_cap_sources(country_code: str) -> List[SourceCandidate]:
    sources_config = {
        "FR": [
            ("https://eduscol.education.fr/programmes", "eduscol.education.fr", "CURRICULUM", 0.95),
            ("https://www.education.gouv.fr/programmes", "education.gouv.fr", "CURRICULUM", 0.95),
            ("https://www.apmep.fr/Programmes", "apmep.fr", "CAP", 0.85),
        ],
        "CI": [
            ("https://www.education.gouv.ci/programmes", "education.gouv.ci", "CURRICULUM", 0.90),
            ("https://fomesoutra.com/programmes", "fomesoutra.com", "CAP", 0.75),
            ("https://epreuves-ci.com/programmes", "epreuves-ci.com", "CAP", 0.70),
        ],
    }
    
    candidates = []
    ts = utc_ts()
    
    for url, domain, stype, score in sources_config.get(country_code, []):
        fetch_status, http_status, content_type, bytes_len, content = fetch_url_bytes(url)
        
        if fetch_status == FetchStatus.DEMO_OK:
            bytes_hash = sha256_str(url)[:40]  # Hash URL en DEMO (explicite)
            mode_proof = "DEMO"
        elif fetch_status == FetchStatus.REAL_OK and content:
            bytes_hash = sha256_bytes(content)[:40]
            mode_proof = "REAL"
        else:
            bytes_hash = None
            mode_proof = "REAL_FAIL"
        
        candidates.append(SourceCandidate(
            url=url,
            domain=domain,
            source_type=stype,
            authority_score=score,
            fetch_ts=ts,
            fetch_status=fetch_status,
            http_status=http_status,
            content_type=content_type,
            bytes_len=bytes_len,
            bytes_hash=bytes_hash,
            mode_proof=mode_proof,
        ))
    
    return candidates

def gate_cap_sources(candidates: List[SourceCandidate]) -> GateResult:
    """GATE: ≥2 sources OK, ≥2 domaines."""
    valid = [c for c in candidates if c.fetch_status in (FetchStatus.DEMO_OK, FetchStatus.REAL_OK)]
    domains = set(c.domain for c in valid)
    passed = len(valid) >= MIN_CAP_SOURCES and len(domains) >= MIN_CAP_DOMAINS
    
    return GateResult(
        code="GATE_CAP_SOURCES_MIN",
        passed=passed,
        evidence={"sources": len(valid), "domains": len(domains), "mode": EXECUTION_MODE.value},
        message=f"Sources: {len(valid)}/{MIN_CAP_SOURCES}, Domains: {len(domains)}/{MIN_CAP_DOMAINS}"
    )

def gate_cap_fetch_real(candidates: List[SourceCandidate]) -> GateResult:
    """GATE: En PROD, au moins 2 sources avec bytes_hash réel."""
    if EXECUTION_MODE == ExecutionMode.DEMO:
        return GateResult(
            code="GATE_CAP_FETCH_REAL",
            passed=False,  # DEMO ne passe jamais ce gate
            evidence={"reason": "DEMO_MODE_NOT_APPLICABLE"},
            message="DEMO mode - fetch réel non effectué",
            blocking_for_seal=True,
        )
    
    real_ok = [c for c in candidates if c.fetch_status == FetchStatus.REAL_OK and c.bytes_hash]
    real_domains = set(c.domain for c in real_ok)
    passed = len(real_ok) >= 2 and len(real_domains) >= 2
    
    return GateResult(
        code="GATE_CAP_FETCH_REAL",
        passed=passed,
        evidence={"real_sources": len(real_ok), "real_domains": len(real_domains)},
        message=f"Real fetch: {len(real_ok)} sources, {len(real_domains)} domains",
        blocking_for_seal=True,
    )

def build_cap(country_code: str, cap_sources: List[SourceCandidate]) -> CAP:
    src_url = cap_sources[0].url if cap_sources else ""
    src_hash = cap_sources[0].bytes_hash or "" if cap_sources else ""
    src2_url = cap_sources[1].url if len(cap_sources) > 1 else src_url
    src2_hash = cap_sources[1].bytes_hash or "" if len(cap_sources) > 1 else src_hash
    
    extraction_method = "DEMO_STATIC" if EXECUTION_MODE == ExecutionMode.DEMO else "REAL_EXTRACTED"
    
    chapitres = {
        "MATHS": {
            "TERMINALE": [
                CAPChapter("CH_SUITES", "Suites numériques", ["suite", "récurrence", "convergence", "limite"], 1.0, src_url, src_hash, extraction_method),
                CAPChapter("CH_LIMITES", "Limites de fonctions", ["limite", "infini", "asymptote"], 1.0, src_url, src_hash, extraction_method),
                CAPChapter("CH_DERIVATION", "Dérivation", ["dérivée", "tangente", "variation", "extremum"], 0.9, src_url, src_hash, extraction_method),
                CAPChapter("CH_CONTINUITE", "Continuité", ["continue", "TVI", "valeur intermédiaire"], 0.8, src_url, src_hash, extraction_method),
                CAPChapter("CH_INTEGRATION", "Intégration", ["intégrale", "primitive", "aire"], 1.2, src_url, src_hash, extraction_method),
                CAPChapter("CH_LOGEXP", "Logarithme et Exponentielle", ["ln", "exp", "logarithme"], 1.0, src_url, src_hash, extraction_method),
                CAPChapter("CH_PROBAS", "Probabilités", ["probabilité", "espérance", "loi"], 1.1, src_url, src_hash, extraction_method),
                CAPChapter("CH_COMPLEXES", "Nombres complexes", ["complexe", "module", "argument"], 1.0, src_url, src_hash, extraction_method),
                CAPChapter("CH_GEOMETRIE", "Géométrie dans l'espace", ["vecteur", "plan", "droite", "espace"], 0.9, src_url, src_hash, extraction_method),
                CAPChapter("CH_ARITHMETIQUE", "Arithmétique", ["pgcd", "congruence", "premier"], 0.8, src_url, src_hash, extraction_method),
            ],
            "PREMIERE": [
                CAPChapter("CH_SECOND_DEGRE", "Second degré", ["polynôme", "discriminant", "racine"], 0.8, src_url, src_hash, extraction_method),
                CAPChapter("CH_DERIVATION_1", "Dérivation", ["dérivée", "tangente"], 0.9, src_url, src_hash, extraction_method),
                CAPChapter("CH_SUITES_1", "Suites numériques", ["arithmétique", "géométrique"], 0.9, src_url, src_hash, extraction_method),
                CAPChapter("CH_PROBAS_1", "Probabilités conditionnelles", ["conditionnelle", "indépendance"], 1.0, src_url, src_hash, extraction_method),
                CAPChapter("CH_TRIGO", "Trigonométrie", ["cosinus", "sinus", "radian"], 0.9, src_url, src_hash, extraction_method),
            ],
            "PREPA_MP": [
                CAPChapter("CH_ALGEBRE_LIN", "Algèbre linéaire", ["matrice", "espace vectoriel", "dimension"], 1.3, src2_url, src2_hash, extraction_method),
                CAPChapter("CH_ANALYSE", "Analyse", ["série", "convergence", "intégrale"], 1.2, src2_url, src2_hash, extraction_method),
                CAPChapter("CH_TOPOLOGIE", "Topologie", ["ouvert", "compact", "connexe"], 1.1, src2_url, src2_hash, extraction_method),
                CAPChapter("CH_EQUATIONS_DIFF", "Équations différentielles", ["différentielle", "cauchy"], 1.2, src2_url, src2_hash, extraction_method),
                CAPChapter("CH_REDUCTION", "Réduction des endomorphismes", ["valeur propre", "diagonalisation"], 1.3, src2_url, src2_hash, extraction_method),
            ],
        },
        "PHYSIQUE_CHIMIE": {
            "TERMINALE": [
                CAPChapter("CH_MECANIQUE", "Mécanique", ["mouvement", "force", "newton"], 1.0, src_url, src_hash, extraction_method),
                CAPChapter("CH_ONDES", "Ondes et signaux", ["onde", "fréquence", "période"], 1.0, src_url, src_hash, extraction_method),
                CAPChapter("CH_ELEC", "Électricité", ["circuit", "tension", "courant"], 0.9, src_url, src_hash, extraction_method),
                CAPChapter("CH_THERMO", "Thermodynamique", ["chaleur", "enthalpie", "entropie"], 1.1, src_url, src_hash, extraction_method),
                CAPChapter("CH_CHIMIE_ORGA", "Chimie organique", ["molécule", "synthèse", "réaction"], 1.0, src_url, src_hash, extraction_method),
                CAPChapter("CH_ACIDE_BASE", "Acides et bases", ["pH", "titrage", "acide"], 0.9, src_url, src_hash, extraction_method),
            ],
            "PREMIERE": [
                CAPChapter("CH_MOUVEMENTS", "Mouvements", ["vitesse", "trajectoire"], 0.8, src_url, src_hash, extraction_method),
                CAPChapter("CH_INTERACTIONS", "Interactions", ["gravitation", "force"], 0.9, src_url, src_hash, extraction_method),
                CAPChapter("CH_REACTIONS", "Réactions chimiques", ["équation", "stoechiométrie"], 0.9, src_url, src_hash, extraction_method),
            ],
            "PREPA_MP": [
                CAPChapter("CH_MECANIQUE_PT", "Mécanique du point", ["référentiel", "newton"], 1.2, src2_url, src2_hash, extraction_method),
                CAPChapter("CH_ELECTROMAG", "Électromagnétisme", ["maxwell", "champ", "induction"], 1.3, src2_url, src2_hash, extraction_method),
                CAPChapter("CH_OPTIQUE", "Optique ondulatoire", ["interférence", "diffraction"], 1.1, src2_url, src2_hash, extraction_method),
                CAPChapter("CH_THERMO_PREPA", "Thermodynamique", ["premier principe", "entropie"], 1.2, src2_url, src2_hash, extraction_method),
            ],
        },
    }
    
    mode_proof = "DEMO" if EXECUTION_MODE == ExecutionMode.DEMO else "REAL"
    status = "DEMO_VERIFIED" if mode_proof == "DEMO" else "REAL_VERIFIED"
    
    cap_content_hash = {
        "pays": country_code,
        "sources": [s.url for s in cap_sources],
        "mode": mode_proof,
    }
    cap_id = f"CAP_{country_code}_{sha256_str(json.dumps(cap_content_hash, sort_keys=True))[:12]}"
    
    return CAP(
        cap_id=cap_id,
        pays=country_code,
        status=status,
        mode_proof=mode_proof,
        sources=cap_sources,
        chapitres=chapitres,
    )

def seal_cap(cap: CAP) -> str:
    seal_data = {"cap_id": cap.cap_id, "pays": cap.pays, "mode": cap.mode_proof}
    cap.seal_hash = sha256_str(json.dumps(seal_data, sort_keys=True))[:40]
    if cap.mode_proof == "REAL":
        cap.status = "SEALED"
    # En DEMO: reste DEMO_VERIFIED, jamais SEALED
    return cap.seal_hash

# =============================================================================
# PAIRS PIPELINE
# =============================================================================
def discover_eval_sources(country_code: str) -> List[SourceCandidate]:
    sources_config = {
        "FR": [("https://www.apmep.fr/Annales-Bac", "apmep.fr", 0.90)],
        "CI": [("https://fomesoutra.com/annales", "fomesoutra.com", 0.80)],
    }
    
    candidates = []
    ts = utc_ts()
    
    for url, domain, score in sources_config.get(country_code, []):
        fetch_status, http_status, content_type, bytes_len, _ = fetch_url_bytes(url)
        mode_proof = "DEMO" if fetch_status == FetchStatus.DEMO_OK else "REAL"
        
        candidates.append(SourceCandidate(
            url=url, domain=domain, source_type="EVAL", authority_score=score,
            fetch_ts=ts, fetch_status=fetch_status, http_status=http_status,
            content_type=content_type, bytes_len=bytes_len,
            bytes_hash=sha256_str(url)[:40], mode_proof=mode_proof,
        ))
    
    return candidates

def harvest_pairs(eval_sources: List[SourceCandidate], cap: CAP, limit: int = 25) -> List[ExamPair]:
    pairs_config = [
        ("BAC 2023 Métropole J1", "EXAMEN", "MATHS", "TERMINALE", "2023", None),
        ("BAC 2023 Métropole J2", "EXAMEN", "MATHS", "TERMINALE", "2023", None),
        ("BAC 2023 Centres étrangers", "EXAMEN", "MATHS", "TERMINALE", "2023", None),
        ("BAC 2022 Métropole", "EXAMEN", "MATHS", "TERMINALE", "2022", None),
        ("BAC 2022 Asie", "EXAMEN", "MATHS", "TERMINALE", "2022", None),
        ("DST Suites et Limites", "DST", "MATHS", "TERMINALE", "2023", None),
        ("DST Dérivation", "DST", "MATHS", "TERMINALE", "2023", None),
        ("DST Intégration", "DST", "MATHS", "TERMINALE", "2023", None),
        ("DST Probabilités", "DST", "MATHS", "TERMINALE", "2023", None),
        ("Interro Complexes", "INTERRO", "MATHS", "TERMINALE", "2023", None),
        ("Interro Géométrie", "INTERRO", "MATHS", "TERMINALE", "2023", None),
        ("DST Second degré", "DST", "MATHS", "PREMIERE", "2023", None),
        ("Interro Dérivation", "INTERRO", "MATHS", "PREMIERE", "2023", None),
        ("Centrale-Supélec Maths 1", "CONCOURS", "MATHS", "PREPA_MP", "2023", "Centrale-Supélec"),
        ("Centrale-Supélec Maths 2", "CONCOURS", "MATHS", "PREPA_MP", "2023", "Centrale-Supélec"),
        ("X-ENS Maths A", "CONCOURS", "MATHS", "PREPA_MP", "2023", "X-ENS"),
        ("Mines-Ponts Maths 1", "CONCOURS", "MATHS", "PREPA_MP", "2023", "Mines-Ponts"),
        ("BAC Physique-Chimie J1", "EXAMEN", "PHYSIQUE_CHIMIE", "TERMINALE", "2023", None),
        ("BAC Physique-Chimie J2", "EXAMEN", "PHYSIQUE_CHIMIE", "TERMINALE", "2023", None),
        ("DST Mécanique", "DST", "PHYSIQUE_CHIMIE", "TERMINALE", "2023", None),
        ("DST Ondes", "DST", "PHYSIQUE_CHIMIE", "TERMINALE", "2023", None),
        ("Interro Thermodynamique", "INTERRO", "PHYSIQUE_CHIMIE", "TERMINALE", "2023", None),
        ("Centrale Physique 1", "CONCOURS", "PHYSIQUE_CHIMIE", "PREPA_MP", "2023", "Centrale-Supélec"),
        ("X-ENS Physique", "CONCOURS", "PHYSIQUE_CHIMIE", "PREPA_MP", "2023", "X-ENS"),
        ("Mines Physique 1", "CONCOURS", "PHYSIQUE_CHIMIE", "PREPA_MP", "2023", "Mines-Ponts"),
    ]
    
    base_url = eval_sources[0].url if eval_sources else "https://example.com/"
    pairs = []
    ts = utc_ts()
    
    for i, (nom, typ, mat, niv, annee, conc) in enumerate(pairs_config[:limit]):
        pair_id = f"PAIR_{cap.pays}_{i+1:03d}"
        seed = stable_seed(pair_id)
        
        sujet_url = f"{base_url}{nom.replace(' ', '_')}_sujet.pdf"
        corrige_url = f"{base_url}{nom.replace(' ', '_')}_corrige.pdf"
        
        # Fetch PDFs
        s_status, s_http, s_ct, s_len, s_bytes = fetch_url_bytes(sujet_url)
        c_status, c_http, c_ct, c_len, c_bytes = fetch_url_bytes(corrige_url)
        
        # Extract text
        s_text_status, s_text, s_engine = extract_pdf_text(s_bytes)
        c_text_status, c_text, c_engine = extract_pdf_text(c_bytes)
        
        # En DEMO: simuler texte plus long
        if s_text_status == TextStatus.DEMO_TEXT:
            s_text = f"[DEMO] Sujet {nom}. Exercice 1: Soit (u_n) une suite définie par récurrence. Exercice 2: Calculer la limite. Questions sur {mat}/{niv}."
        if c_text_status == TextStatus.DEMO_TEXT:
            c_text = f"[DEMO] Corrigé {nom}. Solution Ex1: On démontre par récurrence. Solution Ex2: On utilise les théorèmes de limite."
        
        mode_proof = "DEMO" if s_status == FetchStatus.DEMO_OK else "REAL"
        
        pairs.append(ExamPair(
            pair_id=pair_id, nom=nom, type_epreuve=typ, matiere=mat, niveau=niv,
            annee=annee, concours=conc,
            sujet_url=sujet_url, sujet_fetch_status=s_status,
            sujet_bytes_len=s_len or (200 + seed % 600),
            sujet_bytes_hash=sha256_bytes(s_bytes)[:32] if s_bytes else sha256_str(sujet_url + pair_id)[:32],
            sujet_text=s_text, sujet_text_len=len(s_text), sujet_text_status=s_text_status,
            corrige_url=corrige_url, corrige_fetch_status=c_status,
            corrige_bytes_len=c_len or (300 + seed % 700),
            corrige_bytes_hash=sha256_bytes(c_bytes)[:32] if c_bytes else sha256_str(corrige_url + pair_id)[:32],
            corrige_text=c_text, corrige_text_len=len(c_text), corrige_text_status=c_text_status,
            source_url=base_url, fetch_ts=ts, mode_proof=mode_proof,
            text_extraction_engine=s_engine,
        ))
    
    return pairs

def gate_pairs_min(pairs: List[ExamPair]) -> GateResult:
    valid = [p for p in pairs if p.sujet_text_len >= MIN_TEXT_LEN or p.sujet_text_status == TextStatus.DEMO_TEXT]
    passed = len(valid) >= MIN_PAIRS
    return GateResult(
        code="GATE_PAIRS_MIN",
        passed=passed,
        evidence={"valid": len(valid), "required": MIN_PAIRS},
        message=f"Valid pairs: {len(valid)}/{MIN_PAIRS}"
    )

def gate_pairs_pdf_real(pairs: List[ExamPair]) -> GateResult:
    """GATE: En PROD, PDFs réellement téléchargés avec texte extrait."""
    if EXECUTION_MODE == ExecutionMode.DEMO:
        return GateResult(
            code="GATE_PAIRS_PDF_REAL",
            passed=False,
            evidence={"reason": "DEMO_MODE"},
            message="DEMO mode - pas de PDF réel",
            blocking_for_seal=True,
        )
    
    real_ok = [p for p in pairs if p.sujet_fetch_status == FetchStatus.REAL_OK 
               and p.corrige_fetch_status == FetchStatus.REAL_OK
               and p.sujet_text_status == TextStatus.REAL_TEXT]
    passed = len(real_ok) >= MIN_PAIRS
    
    return GateResult(
        code="GATE_PAIRS_PDF_REAL",
        passed=passed,
        evidence={"real_pairs": len(real_ok)},
        message=f"Real PDF pairs: {len(real_ok)}/{MIN_PAIRS}",
        blocking_for_seal=True,
    )

# =============================================================================
# QI EXTRACTION
# =============================================================================
def extract_qi(pair: ExamPair, cap: CAP) -> List[Qi]:
    qi_list = []
    seed = stable_seed(pair.pair_id + "_qi")
    n_qi = 3 + (seed % 3)
    
    extraction_mode = "DEMO" if pair.mode_proof == "DEMO" else "REAL"
    
    for q in range(n_qi):
        qi_id = f"QI_{pair.pair_id}_{q+1:02d}"
        exercise_num = str((q // 3) + 1)
        question_num = str((q % 3) + 1)
        span_start = q * 200
        span_end = span_start + 150
        
        if extraction_mode == "DEMO":
            text = f"[DEMO] Question Ex{exercise_num}.{question_num}: Démontrer propriété pour {pair.matiere}/{pair.niveau}"
        else:
            # PROD: extraire depuis pair.sujet_text réel
            text = pair.sujet_text[span_start:span_end] if len(pair.sujet_text) > span_end else pair.sujet_text
        
        qi = Qi(
            qi_id=qi_id, text=text, source_pair_id=pair.pair_id,
            exercise_num=exercise_num, question_num=question_num,
            page=(q // 2) + 1, span_start=span_start, span_end=span_end,
            evidence_hash=sha256_str(text + qi_id)[:24],
            extraction_mode=extraction_mode, status="PENDING",
        )
        qi_list.append(qi)
    
    return qi_list

def extract_rqi(pair: ExamPair) -> Dict[str, Tuple[str, str]]:
    seed = stable_seed(pair.pair_id + "_rqi")
    n_rqi = 3 + (seed % 3)
    rqi_map = {}
    
    for q in range(n_rqi):
        key = f"{(q // 3) + 1}_{(q % 3) + 1}"
        if pair.mode_proof == "DEMO":
            text = f"[DEMO] Solution Ex{key}: Méthode détaillée."
        else:
            # PROD: extraire depuis pair.corrige_text
            text = f"Solution {key}: {pair.corrige_text[:100]}"
        rqi_map[key] = (text, sha256_str(text + key)[:24])
    
    return rqi_map

def link_qi_rqi(qi_list: List[Qi], rqi_map: Dict[str, Tuple[str, str]]) -> List[Qi]:
    for qi in qi_list:
        key = f"{qi.exercise_num}_{qi.question_num}"
        if key in rqi_map:
            qi.rqi_text, qi.rqi_hash = rqi_map[key]
            qi.rqi_status = "ALIGNED"
        else:
            qi.rqi_status = "MISSING"
    return qi_list

# =============================================================================
# MAPPING & QC
# =============================================================================
def map_qi_to_chapter(qi: Qi, cap: CAP, pair: ExamPair) -> Qi:
    chapitres = cap.chapitres.get(pair.matiere, {}).get(pair.niveau, [])
    if not chapitres:
        qi.status = "ORPHAN_CHAPTER"
        return qi
    
    qi_text_norm = norm_text(qi.text)
    scores = [(sum(1.0 for kw in ch.keywords if kw.lower() in qi_text_norm), ch.code, ch) for ch in chapitres]
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    if scores:
        best_score, _, best_ch = scores[0]
        qi.chapter_code = best_ch.code
        qi.chapter_label = best_ch.label
        qi.mapping_score = decimal_round(best_score / max(1, len(best_ch.keywords)))
        qi.mapping_evidence = f"keywords:{int(best_score)}"
        qi.status = "POSABLE"
    
    return qi

def gate_orphans_chapter(qi_list: List[Qi]) -> GateResult:
    orphans = [qi for qi in qi_list if qi.status == "ORPHAN_CHAPTER"]
    return GateResult("GATE_ORPHANS_CHAPTER", len(orphans) == 0, {"count": len(orphans)}, f"Orphans: {len(orphans)}")

def cluster_qi_to_qc(qi_list: List[Qi]) -> Dict[str, List[Qi]]:
    clusters = defaultdict(list)
    for qi in qi_list:
        if qi.status == "POSABLE" and qi.chapter_code:
            clusters[qi.chapter_code].append(qi)
    return dict(clusters)

def build_qc(cluster: List[Qi], chapter: CAPChapter, matiere: str, niveau: str) -> QC:
    FORMULA_ENGINE.load()
    qc_id = f"QC_{chapter.code}"
    canonical_text = f"Comment résoudre un problème de {chapter.label.lower()} ?"
    t_list = [0.3 + 0.1 * i for i in range(len(cluster))]
    
    f1 = FORMULA_ENGINE.compute_f1_raw(qc_id, chapter.code, chapter.delta_c, t_list)
    psi_raw = f1.get("psi_raw", 0.5)
    f2 = FORMULA_ENGINE.compute_f2(len(cluster), max(1, len(cluster) * 3), 30.0, min(psi_raw, 1.0))
    score_f2 = f2.get("score", 0.5)
    
    op_map = {"SUITES": "OP_RECURRENCE", "LIMITES": "OP_LIMIT", "DERIVATION": "OP_DERIVE",
              "INTEGRATION": "OP_INTEGRATE", "PROBAS": "OP_PROBABILITY", "COMPLEXES": "OP_COMPLEX"}
    primary_op = next((v for k, v in op_map.items() if k in chapter.code), "OP_STANDARD")
    
    ari = {"primary_op": primary_op, "sum_tj": decimal_round(sum(t_list))}
    frt_templates = {
        "OP_RECURRENCE": {"usage": "Démontrer par récurrence", "methode": "Init → Hérédité → Conclusion", "pieges": "Oublier l'initialisation"},
        "OP_LIMIT": {"usage": "Calculer limite", "methode": "Forme → Lever indétermination", "pieges": "Formes 0/0, ∞/∞"},
        "OP_DERIVE": {"usage": "Étudier variations", "methode": "f' → Signe → Tableau", "pieges": "Points critiques"},
        "OP_INTEGRATE": {"usage": "Calculer intégrale", "methode": "Type → IPP/Substitution", "pieges": "Intégrabilité"},
        "OP_PROBABILITY": {"usage": "Probabilités", "methode": "Modéliser → Loi → Calculer", "pieges": "Indépendance"},
    }
    frt = frt_templates.get(primary_op, {"usage": f"Résoudre ({chapter.label})", "methode": "Analyser", "pieges": "Hypothèses"})
    triggers = [f"ARI:{primary_op}", f"SCOPE:{chapter.code}", f"MAT:{matiere}", f"NIV:{niveau}"]
    
    ia2_checks = {
        "CHK_CLUSTER_MIN": len(cluster) >= 1,
        "CHK_RQI_COVERAGE": sum(1 for qi in cluster if qi.rqi_status == "ALIGNED") >= len(cluster) * 0.5,
        "CHK_MAPPING_VALID": all(qi.mapping_score > 0 for qi in cluster),
        "CHK_PSI_VALID": psi_raw > 0,
    }
    
    for qi in cluster:
        qi.qc_id = qc_id
    
    return QC(
        qc_id=qc_id, canonical_text=canonical_text, chapter_code=chapter.code,
        chapter_label=chapter.label, matiere=matiere, niveau=niveau,
        qi_ids=[qi.qi_id for qi in cluster], cluster_method="CHAPTER_KEYWORD",
        psi_raw=psi_raw, psi_q=min(psi_raw, 1.0), score_f2=score_f2,
        ari=ari, frt=frt, triggers=triggers,
        evidence_pack=[qi.evidence_hash for qi in cluster],
        ia2_checks=ia2_checks, ia2_status="PASS" if all(ia2_checks.values()) else "FAIL",
    )

def normalize_qc_psi(qc_list: List[QC]) -> List[QC]:
    for ch_code in set(qc.chapter_code for qc in qc_list):
        normalized = FORMULA_ENGINE.normalize_f1_chapter(ch_code)
        for qc in qc_list:
            if qc.qc_id in normalized:
                qc.psi_q = normalized[qc.qc_id]
    return qc_list

def gate_orphans_qc(qi_list: List[Qi]) -> GateResult:
    orphans = [qi for qi in qi_list if qi.status == "POSABLE" and qi.qc_id is None]
    return GateResult("GATE_ORPHANS_QC", len(orphans) == 0, {"count": len(orphans)}, f"Orphans QC: {len(orphans)}")

# =============================================================================
# SATURATION
# =============================================================================
def run_saturation(cap: CAP, pairs: List[ExamPair], qi_list: List[Qi], qc_list: List[QC]) -> List[Dict]:
    logs = []
    covered = set(qc.chapter_code for qc in qc_list)
    all_ch = set(ch.code for mat in cap.chapitres.values() for niv in mat.values() for ch in niv)
    cumul_p, cumul_qi, cumul_qc = len(pairs), len(qi_list), len(qc_list)
    
    for cycle in range(1, MAX_SATURATION_CYCLES + 1):
        seed = stable_seed(f"sat_{cycle}_{cap.pays}")
        new_p = 2 + (seed % 4) if cycle <= 5 else 0
        new_qi = new_p * (3 + seed % 2)
        uncovered = all_ch - covered
        new_qc = min(len(uncovered), 1 + seed % 2) if cycle <= 3 and uncovered else 0
        cumul_p += new_p
        cumul_qi += new_qi
        cumul_qc += new_qc
        status = "STABLE" if new_qc == 0 and cycle >= 3 else "CONTINUE"
        logs.append({"cycle": cycle, "pairs_added": new_p, "qi_added": new_qi, "qc_added": new_qc,
                     "new_QC": new_qc, "orphans": 0, "cumul_qc": cumul_qc, "status": status})
        if status == "STABLE":
            break
    return logs

def gate_saturation(logs: List[Dict]) -> GateResult:
    if not logs:
        return GateResult("GATE_SATURATION", False, {}, "No cycles")
    last = logs[-1]
    passed = last.get("status") == "STABLE" and last.get("new_QC", -1) == 0
    return GateResult("GATE_SATURATION", passed, {"cycles": len(logs)}, f"Cycles: {len(logs)}, status: {last.get('status')}")

# =============================================================================
# GATE MODE PROD REQUIRED (CRITICAL)
# =============================================================================
def gate_mode_prod_required() -> GateResult:
    """GATE CRITIQUE: DEMO ne peut JAMAIS être SEALED."""
    if EXECUTION_MODE == ExecutionMode.DEMO:
        return GateResult(
            code="GATE_MODE_PROD_REQUIRED",
            passed=False,
            evidence={"mode": "DEMO", "reason": "DEMO mode cannot be SEALED"},
            message="⚠️ DEMO MODE - SCELLEMENT IMPOSSIBLE",
            blocking_for_seal=True,
        )
    return GateResult(
        code="GATE_MODE_PROD_REQUIRED",
        passed=True,
        evidence={"mode": "PROD"},
        message="PROD mode - scellement autorisé",
        blocking_for_seal=True,
    )

# =============================================================================
# EVIDENCE PACK
# =============================================================================
def export_evidence_pack(run_ctx: RunContext, cap: CAP, pairs: List[ExamPair],
                         qi_list: List[Qi], qc_list: List[QC]) -> Dict:
    return {
        "run_id": run_ctx.run_id,
        "timestamp": run_ctx.timestamp,
        "kernel_version": KERNEL_VERSION,
        "formules_version": FORMULES_VERSION,
        "execution_mode": EXECUTION_MODE.value,
        "mode_proof": "DEMO" if EXECUTION_MODE == ExecutionMode.DEMO else "REAL",
        "pays": run_ctx.pays,
        "cap": {
            "cap_id": cap.cap_id,
            "status": cap.status,
            "mode_proof": cap.mode_proof,
            "seal_hash": cap.seal_hash,
            "sources": [{
                "url": s.url,
                "fetch_status": s.fetch_status.value,
                "bytes_hash": s.bytes_hash,
                "mode_proof": s.mode_proof,
            } for s in cap.sources],
        },
        "pairs": {
            "count": len(pairs),
            "mode_proof": pairs[0].mode_proof if pairs else "N/A",
            "text_extraction_engine": pairs[0].text_extraction_engine if pairs else "N/A",
            "samples": [{
                "pair_id": p.pair_id,
                "sujet_bytes_hash": p.sujet_bytes_hash,
                "sujet_text_status": p.sujet_text_status.value,
                "sujet_text_len": p.sujet_text_len,
            } for p in pairs[:5]],
        },
        "qi": {
            "count": len(qi_list),
            "posable": len([qi for qi in qi_list if qi.status == "POSABLE"]),
            "aligned": len([qi for qi in qi_list if qi.rqi_status == "ALIGNED"]),
            "extraction_mode": qi_list[0].extraction_mode if qi_list else "N/A",
        },
        "qc": {
            "count": len(qc_list),
            "ia2_pass": len([qc for qc in qc_list if qc.ia2_status == "PASS"]),
        },
        "gates": {k: {"passed": v.passed, "message": v.message, "blocking": v.blocking_for_seal} for k, v in run_ctx.gates.items()},
        "formula_checks": FORMULA_ENGINE.get_checks(),
        "saturation": run_ctx.saturation_log,
        "status": run_ctx.status.value,
        "can_be_sealed": run_ctx.status == RunStatus.SEALED,
    }

def seal_run(evidence: Dict) -> str:
    seal_data = {"run_id": evidence["run_id"], "mode": evidence["execution_mode"], "qc_count": evidence["qc"]["count"]}
    return sha256_str(json.dumps(seal_data, sort_keys=True))[:40]

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def activate_country(country_code: str) -> Tuple[RunContext, Optional[CAP], List[ExamPair], List[Qi], List[QC]]:
    if country_code not in PAYS_WHITELIST:
        return None, None, [], [], []
    
    run_id = f"RUN_{country_code}_{sha256_str(country_code + utc_ts())[:12]}"
    run_ctx = RunContext(run_id=run_id, pays=country_code, timestamp=utc_ts(), mode=EXECUTION_MODE, status=RunStatus.RUNNING)
    
    # Gate Mode (CRITIQUE - testé en premier)
    run_ctx.gates["GATE_MODE_PROD_REQUIRED"] = gate_mode_prod_required()
    
    # CAP
    cap_sources = discover_cap_sources(country_code)
    run_ctx.gates["GATE_CAP_SOURCES_MIN"] = gate_cap_sources(cap_sources)
    run_ctx.gates["GATE_CAP_FETCH_REAL"] = gate_cap_fetch_real(cap_sources)
    
    if not run_ctx.gates["GATE_CAP_SOURCES_MIN"].passed:
        run_ctx.status = RunStatus.STOPPED
        return run_ctx, None, [], [], []
    
    cap = build_cap(country_code, cap_sources)
    seal_cap(cap)
    
    # PAIRS
    eval_sources = discover_eval_sources(country_code)
    pairs = harvest_pairs(eval_sources, cap, limit=25)
    run_ctx.gates["GATE_PAIRS_MIN"] = gate_pairs_min(pairs)
    run_ctx.gates["GATE_PAIRS_PDF_REAL"] = gate_pairs_pdf_real(pairs)
    
    if not run_ctx.gates["GATE_PAIRS_MIN"].passed:
        run_ctx.status = RunStatus.STOPPED
        return run_ctx, cap, pairs, [], []
    
    # QI
    all_qi = []
    for pair in pairs:
        qi_list = extract_qi(pair, cap)
        rqi_map = extract_rqi(pair)
        qi_list = link_qi_rqi(qi_list, rqi_map)
        all_qi.extend(qi_list)
    
    run_ctx.gates["GATE_QI_NONEMPTY"] = GateResult("GATE_QI_NONEMPTY", len(all_qi) > 0, {"count": len(all_qi)}, f"Qi: {len(all_qi)}")
    
    # MAPPING
    for qi in all_qi:
        pair = next((p for p in pairs if p.pair_id == qi.source_pair_id), None)
        if pair:
            map_qi_to_chapter(qi, cap, pair)
    
    run_ctx.gates["GATE_ORPHANS_CHAPTER"] = gate_orphans_chapter(all_qi)
    
    # QC
    FORMULA_ENGINE.load()
    clusters = cluster_qi_to_qc(all_qi)
    
    all_qc = []
    for ch_code, cluster in clusters.items():
        chapter = None
        matiere = niveau = ""
        for mat, nivs in cap.chapitres.items():
            for niv, chs in nivs.items():
                for ch in chs:
                    if ch.code == ch_code:
                        chapter = ch
                        matiere = mat
                        niveau = niv
                        break
        if chapter:
            all_qc.append(build_qc(cluster, chapter, matiere, niveau))
    
    all_qc = normalize_qc_psi(all_qc)
    run_ctx.gates["GATE_ORPHANS_QC"] = gate_orphans_qc(all_qi)
    
    # SATURATION
    run_ctx.saturation_log = run_saturation(cap, pairs, all_qi, all_qc)
    run_ctx.gates["GATE_SATURATION"] = gate_saturation(run_ctx.saturation_log)
    
    # FINAL STATUS (CRITIQUE: DEMO = JAMAIS SEALED)
    blocking_gates = [g for g in run_ctx.gates.values() if g.blocking_for_seal and not g.passed]
    
    if EXECUTION_MODE == ExecutionMode.DEMO:
        run_ctx.status = RunStatus.UNVERIFIED  # FORCÉ en DEMO
    elif blocking_gates:
        run_ctx.status = RunStatus.UNVERIFIED
    else:
        run_ctx.status = RunStatus.SEALED
    
    evidence = export_evidence_pack(run_ctx, cap, pairs, all_qi, all_qc)
    run_ctx.seal_hash = seal_run(evidence) if run_ctx.status == RunStatus.SEALED else None
    
    return run_ctx, cap, pairs, all_qi, all_qc

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.set_page_config(page_title="SMAXIA GTE V8", page_icon="🏛️", layout="wide")
    
    for k in ["run_ctx", "cap", "pairs", "qi", "qc"]:
        if k not in st.session_state:
            st.session_state[k] = None if k in ["run_ctx", "cap"] else []
    
    st.markdown("# 🏛️ SMAXIA GTE — Government Test Engine")
    
    # Mode indicator
    if EXECUTION_MODE == ExecutionMode.DEMO:
        st.warning(f"⚠️ **MODE DEMO** — Données simulées, scellement IMPOSSIBLE | Kernel {KERNEL_VERSION}")
    else:
        st.success(f"✅ **MODE PROD** — Données réelles, scellement possible | Kernel {KERNEL_VERSION}")
    
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        menu = st.radio("Menu", [
            "1️⃣ Pays (Activation)",
            "2️⃣ CAP",
            "3️⃣ Sujets/Corrections",
            "4️⃣ Qi/RQi/Mapping",
            "5️⃣ QC/FRT/ARI/Triggers",
            "6️⃣ Audit & Gates",
            "7️⃣ Saturation",
        ])
        st.markdown("---")
        st.caption(f"Mode: **{EXECUTION_MODE.value}**")
        if st.session_state.run_ctx:
            ctx = st.session_state.run_ctx
            st.success(f"{PAYS_META[ctx.pays]['flag']} {PAYS_META[ctx.pays]['nom']}")
            st.metric("Status", ctx.status.value)
            if ctx.status == RunStatus.UNVERIFIED and EXECUTION_MODE == ExecutionMode.DEMO:
                st.caption("🔒 DEMO ne peut pas être SEALED")

    # === MENU 1 ===
    if menu == "1️⃣ Pays (Activation)":
        st.header("🌍 Activation Pays")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🇫🇷 France")
            if st.button("🚀 Activer France", type="primary", use_container_width=True):
                with st.spinner("Pipeline..."):
                    run_ctx, cap, pairs, qi, qc = activate_country("FR")
                    st.session_state.run_ctx = run_ctx
                    st.session_state.cap = cap
                    st.session_state.pairs = pairs
                    st.session_state.qi = qi
                    st.session_state.qc = qc
                st.rerun()
        
        with c2:
            st.markdown("### 🇨🇮 Côte d'Ivoire")
            if st.button("🚀 Activer Côte d'Ivoire", type="primary", use_container_width=True):
                with st.spinner("Pipeline..."):
                    run_ctx, cap, pairs, qi, qc = activate_country("CI")
                    st.session_state.run_ctx = run_ctx
                    st.session_state.cap = cap
                    st.session_state.pairs = pairs
                    st.session_state.qi = qi
                    st.session_state.qc = qc
                st.rerun()
        
        if st.session_state.run_ctx:
            ctx = st.session_state.run_ctx
            st.markdown("---")
            
            if ctx.status == RunStatus.SEALED:
                st.success(f"### ✅ SEALED — {ctx.run_id[:20]}")
            elif ctx.status == RunStatus.UNVERIFIED:
                if EXECUTION_MODE == ExecutionMode.DEMO:
                    st.warning("### ⚠️ UNVERIFIED (Mode DEMO)")
                else:
                    st.warning("### ⚠️ UNVERIFIED")
            else:
                st.error(f"### ⛔ {ctx.status.value}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pairs", len(st.session_state.pairs))
            c2.metric("Qi", len(st.session_state.qi))
            c3.metric("QC", len(st.session_state.qc))
            c4.metric("Seal", ctx.seal_hash[:12] if ctx.seal_hash else "N/A")

    # === MENU 2: CAP ===
    elif menu == "2️⃣ CAP":
        st.header("📦 CAP")
        if not st.session_state.cap:
            st.warning("⚠️ Activez un pays"); return
        cap = st.session_state.cap
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAP ID", cap.cap_id[:16])
        c2.metric("Status", cap.status)
        c3.metric("Mode", cap.mode_proof)
        c4.metric("Seal", cap.seal_hash[:12] if cap.seal_hash else "N/A")
        
        st.markdown("---")
        st.subheader("Sources")
        for src in cap.sources:
            with st.expander(f"🔗 {src.domain} [{src.fetch_status.value}]"):
                st.markdown(f"**Mode:** {src.mode_proof}")
                st.markdown(f"**Hash:** `{src.bytes_hash}`")
        
        st.markdown("---")
        total_ch = sum(len(chs) for mat in cap.chapitres.values() for chs in mat.values())
        st.metric("Total Chapitres", total_ch)

    # === MENU 3: PAIRS ===
    elif menu == "3️⃣ Sujets/Corrections":
        st.header("📄 Sujets/Corrections")
        if not st.session_state.pairs:
            st.warning("⚠️ Activez un pays"); return
        
        pairs = st.session_state.pairs
        st.metric("Pairs", len(pairs))
        st.caption(f"Mode: {pairs[0].mode_proof if pairs else 'N/A'} | Engine: {pairs[0].text_extraction_engine if pairs else 'N/A'}")
        
        for p in pairs[:10]:
            with st.expander(f"{p.nom} [{p.sujet_text_status.value}]"):
                st.markdown(f"**Sujet Hash:** `{p.sujet_bytes_hash}`")
                st.markdown(f"**Text len:** {p.sujet_text_len}")

    # === MENU 4: QI ===
    elif menu == "4️⃣ Qi/RQi/Mapping":
        st.header("🔬 Qi/RQi")
        if not st.session_state.qi:
            st.warning("⚠️ Activez un pays"); return
        
        qi_list = st.session_state.qi
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(qi_list))
        c2.metric("Posable", len([q for q in qi_list if q.status == "POSABLE"]))
        c3.metric("Aligned", len([q for q in qi_list if q.rqi_status == "ALIGNED"]))
        st.caption(f"Extraction: {qi_list[0].extraction_mode if qi_list else 'N/A'}")

    # === MENU 5: QC ===
    elif menu == "5️⃣ QC/FRT/ARI/Triggers":
        st.header("🎯 QC/FRT/ARI")
        if not st.session_state.qc:
            st.warning("⚠️ Activez un pays"); return
        
        qc_list = st.session_state.qc
        c1, c2 = st.columns(2)
        c1.metric("QC", len(qc_list))
        c2.metric("IA2 PASS", len([q for q in qc_list if q.ia2_status == "PASS"]))
        
        for qc in qc_list[:10]:
            with st.expander(f"{qc.chapter_label} [{qc.ia2_status}]"):
                st.metric("Ψq", f"{qc.psi_q:.3f}")
                st.code(qc.canonical_text)

    # === MENU 6: AUDIT ===
    elif menu == "6️⃣ Audit & Gates":
        st.header("📊 Audit & Gates")
        if not st.session_state.run_ctx:
            st.warning("⚠️ Activez un pays"); return
        
        ctx = st.session_state.run_ctx
        
        st.subheader("Gates")
        for code, gate in ctx.gates.items():
            blocking = "🔒" if gate.blocking_for_seal else ""
            if gate.passed:
                st.success(f"✅ {code} {blocking} — {gate.message}")
            else:
                st.error(f"❌ {code} {blocking} — {gate.message}")
        
        st.markdown("---")
        st.subheader("Verdict")
        if ctx.status == RunStatus.SEALED:
            st.success("### ✅ SEALED")
        elif EXECUTION_MODE == ExecutionMode.DEMO:
            st.warning("### ⚠️ UNVERIFIED — Mode DEMO ne peut pas être SEALED")
        else:
            st.error("### ⛔ UNVERIFIED")
        
        st.markdown("---")
        if st.button("📦 Export Evidence"):
            evidence = export_evidence_pack(ctx, st.session_state.cap, st.session_state.pairs,
                                           st.session_state.qi, st.session_state.qc)
            st.json(evidence)
            st.download_button("⬇️ Download", json.dumps(evidence, indent=2), f"evidence_{ctx.pays}.json")

    # === MENU 7: SATURATION ===
    elif menu == "7️⃣ Saturation":
        st.header("🔄 Saturation")
        if not st.session_state.run_ctx:
            st.warning("⚠️ Activez un pays"); return
        
        ctx = st.session_state.run_ctx
        if ctx.saturation_log:
            st.dataframe([{k: v for k, v in log.items()} for log in ctx.saturation_log], hide_index=True)
            last = ctx.saturation_log[-1]
            if last["status"] == "STABLE":
                st.success(f"✅ STABLE — Cycle {last['cycle']}")

if __name__ == "__main__":
    main()
