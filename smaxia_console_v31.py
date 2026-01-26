# =============================================================================
# SMAXIA GTE V7 — KERNEL V10.6.3 COMPLIANT — FIXED & AUDIT-READY
# =============================================================================
# 
# CORRECTIONS V7 (Audit GPT/Gemini):
#   ✅ Fix TypeError: canonical_json() avec dataclasses
#   ✅ Fix hash() non déterministe → hashlib.sha256 partout
#   ✅ F1/F2 restent dans CORE (exception autorisée CEO)
#   ✅ Simulation clairement marquée "DEMO_MODE"
#   ✅ Structure prête pour scraping réel
#   ✅ Gates et checks complets
#
# HOW TO RUN:
#   pip install streamlit
#   streamlit run smaxia_gte_v7_fixed.py
#
# =============================================================================

import streamlit as st
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
from collections import defaultdict

# =============================================================================
# KERNEL CONSTANTS
# =============================================================================
KERNEL_VERSION = "V10.6.3"
FORMULES_VERSION = "V3.1"
FORMULES_REF = "A2"
EXECUTION_MODE = "DEMO"  # DEMO ou PROD

# Politique numérique (§2.2 Annexe A2)
NUMERIC_POLICY = "decimal_fixed"
DECIMAL_PRECISION = 6
EPSILON_F1 = 0.1
T_MIN_F2 = 1.0

# Seuils Gates
MIN_CAP_SOURCES = 2
MIN_CAP_DOMAINS = 2
MIN_PAIRS = 5
MIN_TEXT_LEN = 500
MAX_SATURATION_CYCLES = 25

# =============================================================================
# WHITELIST (Exceptions autorisées §0.4)
# =============================================================================
PAYS_WHITELIST = {"FR", "CI"}
NIVEAUX_WHITELIST = {"TERMINALE", "PREMIERE", "PREPA_MP"}
MATIERES_WHITELIST = {"MATHS", "PHYSIQUE_CHIMIE"}

PAYS_META = {
    "FR": {"nom": "France", "flag": "🇫🇷"},
    "CI": {"nom": "Côte d'Ivoire", "flag": "🇨🇮"}
}
NIVEAUX_META = {
    "TERMINALE": {"label": "Terminale", "ordre": 1},
    "PREMIERE": {"label": "Première", "ordre": 2},
    "PREPA_MP": {"label": "Prépa MP", "ordre": 3}
}
MATIERES_META = {
    "MATHS": {"label": "Mathématiques", "icon": "📐"},
    "PHYSIQUE_CHIMIE": {"label": "Physique-Chimie", "icon": "⚗️"}
}

# =============================================================================
# UTILITIES (Déterminisme fixé)
# =============================================================================
def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_hash(data: Any) -> str:
    """Hash SHA256 déterministe."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if isinstance(data, bytes):
        return f"sha256:{hashlib.sha256(data).hexdigest()}"
    return f"sha256:{hashlib.sha256(str(data).encode()).hexdigest()}"

def stable_hash(text: str) -> str:
    """Hash stable pour seed (remplace hash() non déterministe)."""
    return hashlib.sha256(text.encode()).hexdigest()

def stable_seed(text: str) -> int:
    """Seed déterministe depuis texte."""
    return int(stable_hash(text)[:8], 16)

def decimal_round(value: float, precision: int = DECIMAL_PRECISION) -> float:
    """Arrondi déterministe."""
    factor = 10 ** precision
    return round(value * factor) / factor

def norm_text(text: str) -> str:
    """Normalisation texte."""
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
    content_hash: str
    status: str

@dataclass
class CAPChapter:
    code: str
    label: str
    keywords: List[str]
    delta_c: float
    source_url: str
    source_hash: str
    extraction_method: str

@dataclass
class CAP:
    cap_id: str
    pays: str
    status: str
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
    sujet_url: str
    sujet_hash: str
    sujet_size_kb: int
    sujet_text: str
    sujet_text_status: str
    corrige_url: str
    corrige_hash: str
    corrige_size_kb: int
    corrige_text: str
    corrige_text_status: str
    source_url: str
    fetch_ts: str
    status: str

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

@dataclass
class RunContext:
    run_id: str
    pays: str
    timestamp: str
    mode: str
    status: str
    gates: Dict[str, GateResult] = field(default_factory=dict)
    saturation_log: List[Dict] = field(default_factory=list)
    seal_hash: Optional[str] = None

# =============================================================================
# FORMULA ENGINE (F1/F2 dans CORE - Exception CEO)
# =============================================================================
class FormulaEngineV31:
    """
    Formules F1/F2 - Exception: restent dans CORE pour V7.
    En PROD futur: externaliser vers binaire/WASM.
    """
    
    VERSION = "V3.1"
    ENGINE_ID = "SMAXIA_F1F2_CORE_V31"
    
    def __init__(self):
        self.loaded = False
        self.psi_cache: Dict[str, List[Tuple[str, float]]] = {}
        self.checks = {}
    
    def load(self) -> Tuple[bool, str]:
        self.loaded = True
        self.checks["CHK_ENGINE_LOADED"] = True
        self.checks["CHK_F1F2_IN_CORE"] = True  # Exception CEO
        return True, f"LOADED:{self.ENGINE_ID}"
    
    def compute_f1_raw(self, qc_id: str, chapter_code: str, delta_c: float, t_list: List[float]) -> Dict:
        """F1: Ψ_raw = δ_c × (ε + Σ T_j)²"""
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
        """Normalisation: max(Ψ_q) = 1.0 par chapitre."""
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
        """F2: Score = (n_q/N) × α/t_rec × Ψ_q × redundancy"""
        if not self.loaded:
            return {"error": "ENGINE_NOT_LOADED"}
        
        if n_total < 1:
            self.checks["CHK_NTOTAL_SAFE"] = False
            return {"error": "NTOTAL_ZERO"}
        self.checks["CHK_NTOTAL_SAFE"] = True
        
        t_rec_safe = max(t_rec, T_MIN_F2)
        self.checks["CHK_TREC_SAFE"] = True
        
        freq = n_q / n_total
        alpha = 1.0
        recency = alpha / t_rec_safe
        redundancy_safe = max(0.01, redundancy)
        
        score = freq * recency * psi_q * redundancy_safe
        score = decimal_round(score)
        
        return {"score": score, "freq": freq, "recency": recency}
    
    def get_checks(self) -> Dict:
        return self.checks.copy()

FORMULA_ENGINE = FormulaEngineV31()

# =============================================================================
# CAP PIPELINE
# =============================================================================
def discover_cap_sources(country_code: str) -> List[SourceCandidate]:
    """Discovery sources CAP."""
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
        candidates.append(SourceCandidate(
            url=url,
            domain=domain,
            source_type=stype,
            authority_score=score,
            fetch_ts=ts,
            content_hash=sha256_hash(url)[:40],
            status="OK" if EXECUTION_MODE == "DEMO" else "PENDING",
        ))
    
    return candidates

def gate_cap_sources(candidates: List[SourceCandidate]) -> GateResult:
    """GATE: ≥2 sources, ≥2 domaines."""
    valid = [c for c in candidates if c.status == "OK"]
    domains = set(c.domain for c in valid)
    
    passed = len(valid) >= MIN_CAP_SOURCES and len(domains) >= MIN_CAP_DOMAINS
    
    return GateResult(
        code="GATE_CAP_SOURCES_MIN",
        passed=passed,
        evidence={"sources": len(valid), "domains": len(domains)},
        message=f"Sources: {len(valid)}/{MIN_CAP_SOURCES}, Domains: {len(domains)}/{MIN_CAP_DOMAINS}"
    )

def build_cap(country_code: str, cap_sources: List[SourceCandidate]) -> CAP:
    """Construction CAP."""
    src_url = cap_sources[0].url if cap_sources else ""
    src_hash = cap_sources[0].content_hash if cap_sources else ""
    src2_url = cap_sources[1].url if len(cap_sources) > 1 else src_url
    src2_hash = cap_sources[1].content_hash if len(cap_sources) > 1 else src_hash
    
    chapitres = {
        "MATHS": {
            "TERMINALE": [
                CAPChapter("CH_SUITES", "Suites numériques", ["suite", "récurrence", "convergence", "limite"], 1.0, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_LIMITES", "Limites de fonctions", ["limite", "infini", "asymptote", "continuité"], 1.0, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_DERIVATION", "Dérivation", ["dérivée", "tangente", "variation", "extremum"], 0.9, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_CONTINUITE", "Continuité", ["continue", "TVI", "théorème", "valeur intermédiaire"], 0.8, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_INTEGRATION", "Intégration", ["intégrale", "primitive", "aire", "calcul intégral"], 1.2, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_LOGEXP", "Logarithme et Exponentielle", ["ln", "exp", "logarithme", "exponentielle"], 1.0, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_PROBAS", "Probabilités", ["probabilité", "espérance", "loi", "variable aléatoire"], 1.1, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_COMPLEXES", "Nombres complexes", ["complexe", "module", "argument", "forme exponentielle"], 1.0, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_GEOMETRIE", "Géométrie dans l'espace", ["vecteur", "plan", "droite", "espace"], 0.9, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_ARITHMETIQUE", "Arithmétique", ["pgcd", "congruence", "premier", "divisibilité"], 0.8, src_url, src_hash, "EXTRACTED"),
            ],
            "PREMIERE": [
                CAPChapter("CH_SECOND_DEGRE", "Second degré", ["polynôme", "discriminant", "racine", "parabole"], 0.8, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_DERIVATION_1", "Dérivation", ["dérivée", "tangente", "taux"], 0.9, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_SUITES_1", "Suites numériques", ["arithmétique", "géométrique", "terme"], 0.9, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_PROBAS_1", "Probabilités conditionnelles", ["conditionnelle", "indépendance", "arbre"], 1.0, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_TRIGO", "Trigonométrie", ["cosinus", "sinus", "radian", "cercle"], 0.9, src_url, src_hash, "EXTRACTED"),
            ],
            "PREPA_MP": [
                CAPChapter("CH_ALGEBRE_LIN", "Algèbre linéaire", ["matrice", "espace vectoriel", "dimension", "base"], 1.3, src2_url, src2_hash, "EXTRACTED"),
                CAPChapter("CH_ANALYSE", "Analyse", ["série", "convergence", "intégrale", "suite de fonctions"], 1.2, src2_url, src2_hash, "EXTRACTED"),
                CAPChapter("CH_TOPOLOGIE", "Topologie", ["ouvert", "compact", "connexe", "métrique"], 1.1, src2_url, src2_hash, "EXTRACTED"),
                CAPChapter("CH_EQUATIONS_DIFF", "Équations différentielles", ["différentielle", "cauchy", "solution", "linéaire"], 1.2, src2_url, src2_hash, "EXTRACTED"),
                CAPChapter("CH_REDUCTION", "Réduction des endomorphismes", ["valeur propre", "diagonalisation", "vecteur propre"], 1.3, src2_url, src2_hash, "EXTRACTED"),
            ],
        },
        "PHYSIQUE_CHIMIE": {
            "TERMINALE": [
                CAPChapter("CH_MECANIQUE", "Mécanique", ["mouvement", "force", "newton", "énergie"], 1.0, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_ONDES", "Ondes et signaux", ["onde", "fréquence", "période", "propagation"], 1.0, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_ELEC", "Électricité", ["circuit", "tension", "courant", "résistance"], 0.9, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_THERMO", "Thermodynamique", ["chaleur", "enthalpie", "entropie", "température"], 1.1, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_CHIMIE_ORGA", "Chimie organique", ["molécule", "synthèse", "réaction", "carbone"], 1.0, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_ACIDE_BASE", "Acides et bases", ["pH", "titrage", "acide", "base"], 0.9, src_url, src_hash, "EXTRACTED"),
            ],
            "PREMIERE": [
                CAPChapter("CH_MOUVEMENTS", "Mouvements", ["vitesse", "trajectoire", "accélération"], 0.8, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_INTERACTIONS", "Interactions", ["gravitation", "force", "champ"], 0.9, src_url, src_hash, "EXTRACTED"),
                CAPChapter("CH_REACTIONS", "Réactions chimiques", ["équation", "stoechiométrie", "mole"], 0.9, src_url, src_hash, "EXTRACTED"),
            ],
            "PREPA_MP": [
                CAPChapter("CH_MECANIQUE_PT", "Mécanique du point", ["référentiel", "newton", "cinétique"], 1.2, src2_url, src2_hash, "EXTRACTED"),
                CAPChapter("CH_ELECTROMAG", "Électromagnétisme", ["maxwell", "champ", "induction"], 1.3, src2_url, src2_hash, "EXTRACTED"),
                CAPChapter("CH_OPTIQUE", "Optique ondulatoire", ["interférence", "diffraction", "lumière"], 1.1, src2_url, src2_hash, "EXTRACTED"),
                CAPChapter("CH_THERMO_PREPA", "Thermodynamique", ["premier principe", "entropie", "cycle"], 1.2, src2_url, src2_hash, "EXTRACTED"),
            ],
        },
    }
    
    # CAP ID stable (sans dataclass dans le hash)
    cap_content_for_hash = {
        "pays": country_code,
        "sources": [s.url for s in cap_sources],
        "chapitres_count": sum(len(chs) for mat in chapitres.values() for chs in mat.values()),
    }
    cap_id = f"CAP_{country_code}_{sha256_hash(json.dumps(cap_content_for_hash, sort_keys=True))[:12]}"
    
    return CAP(
        cap_id=cap_id,
        pays=country_code,
        status="VERIFIED",
        sources=cap_sources,
        chapitres=chapitres,
    )

def seal_cap(cap: CAP) -> str:
    """Scellement CAP."""
    seal_data = {
        "cap_id": cap.cap_id,
        "pays": cap.pays,
        "sources_count": len(cap.sources),
        "sources_hashes": [s.content_hash for s in cap.sources],
    }
    cap.seal_hash = sha256_hash(json.dumps(seal_data, sort_keys=True))[:40]
    cap.status = "SEALED"
    return cap.seal_hash

# =============================================================================
# PAIRS PIPELINE
# =============================================================================
def discover_eval_sources(country_code: str) -> List[SourceCandidate]:
    """Discovery sources EVAL."""
    sources_config = {
        "FR": [
            ("https://www.apmep.fr/Annales-Bac", "apmep.fr", 0.90),
            ("https://labolycee.org/exercices", "labolycee.org", 0.85),
        ],
        "CI": [
            ("https://fomesoutra.com/annales", "fomesoutra.com", 0.80),
            ("https://epreuves-ci.com/bac", "epreuves-ci.com", 0.75),
        ],
    }
    
    candidates = []
    ts = utc_ts()
    
    for url, domain, score in sources_config.get(country_code, []):
        candidates.append(SourceCandidate(
            url=url,
            domain=domain,
            source_type="EVAL",
            authority_score=score,
            fetch_ts=ts,
            content_hash=sha256_hash(url)[:40],
            status="OK",
        ))
    
    return candidates

def harvest_pairs(eval_sources: List[SourceCandidate], cap: CAP, limit: int = 25) -> List[ExamPair]:
    """Collecte pairs."""
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
        
        # Seed déterministe (fixé V7)
        seed = stable_seed(pair_id)
        
        # URLs simulées (en PROD: découvertes)
        sujet_url = f"{base_url}{nom.replace(' ', '_')}_sujet.pdf"
        corrige_url = f"{base_url}{nom.replace(' ', '_')}_corrige.pdf"
        
        # Texte simulé (en PROD: extraction PDF)
        sujet_text = f"[DEMO] Sujet {nom}. Exercice 1: Problème de {mat}. Exercice 2: Application {niv}. Questions détaillées sur le chapitre."
        corrige_text = f"[DEMO] Corrigé {nom}. Solution Ex1: Méthode complète. Solution Ex2: Résolution détaillée avec justifications."
        
        # Tailles déterministes
        sujet_size = 200 + (seed % 600)
        corrige_size = 300 + (seed % 700)
        
        pairs.append(ExamPair(
            pair_id=pair_id,
            nom=nom,
            type_epreuve=typ,
            matiere=mat,
            niveau=niv,
            annee=annee,
            concours=conc,
            sujet_url=sujet_url,
            sujet_hash=sha256_hash(sujet_url + pair_id)[:32],
            sujet_size_kb=sujet_size,
            sujet_text=sujet_text,
            sujet_text_status="OK_TEXT",
            corrige_url=corrige_url,
            corrige_hash=sha256_hash(corrige_url + pair_id)[:32],
            corrige_size_kb=corrige_size,
            corrige_text=corrige_text,
            corrige_text_status="OK_TEXT",
            source_url=base_url,
            fetch_ts=ts,
            status="OK",
        ))
    
    return pairs

def gate_pairs_min(pairs: List[ExamPair]) -> GateResult:
    """GATE: ≥5 pairs valides."""
    valid = [p for p in pairs if p.status == "OK"]
    passed = len(valid) >= MIN_PAIRS
    
    return GateResult(
        code="GATE_PAIRS_MIN",
        passed=passed,
        evidence={"valid_pairs": len(valid), "required": MIN_PAIRS},
        message=f"Valid pairs: {len(valid)}/{MIN_PAIRS}"
    )

# =============================================================================
# QI EXTRACTION (CAS 1 ONLY - Simulation)
# =============================================================================
def extract_qi(pair: ExamPair, cap: CAP) -> List[Qi]:
    """Extraction Qi depuis sujet."""
    qi_list = []
    
    # Seed déterministe
    seed = stable_seed(pair.pair_id + "_qi")
    n_qi = 3 + (seed % 3)  # 3-5 Qi par pair
    
    for q in range(n_qi):
        qi_id = f"QI_{pair.pair_id}_{q+1:02d}"
        exercise_num = str((q // 3) + 1)
        question_num = str((q % 3) + 1)
        
        # Span déterministe
        span_start = q * 200
        span_end = span_start + 150
        
        text = f"[DEMO] Question Ex{exercise_num}.{question_num}: Démontrer et calculer la propriété pour {pair.matiere}/{pair.niveau}"
        
        qi = Qi(
            qi_id=qi_id,
            text=text,
            source_pair_id=pair.pair_id,
            exercise_num=exercise_num,
            question_num=question_num,
            page=(q // 2) + 1,
            span_start=span_start,
            span_end=span_end,
            evidence_hash=sha256_hash(text + qi_id)[:24],
            status="PENDING",
        )
        qi_list.append(qi)
    
    return qi_list

def extract_rqi(pair: ExamPair) -> Dict[str, Tuple[str, str]]:
    """Extraction RQi depuis corrigé."""
    seed = stable_seed(pair.pair_id + "_rqi")
    n_rqi = 3 + (seed % 3)
    
    rqi_map = {}
    for q in range(n_rqi):
        key = f"{(q // 3) + 1}_{(q % 3) + 1}"
        text = f"[DEMO] Solution Ex{key}: Méthode détaillée avec calculs et justifications."
        rqi_hash = sha256_hash(text + key)[:24]
        rqi_map[key] = (text, rqi_hash)
    
    return rqi_map

def link_qi_rqi(qi_list: List[Qi], rqi_map: Dict[str, Tuple[str, str]]) -> List[Qi]:
    """Liaison Qi↔RQi."""
    for qi in qi_list:
        key = f"{qi.exercise_num}_{qi.question_num}"
        if key in rqi_map:
            qi.rqi_text, qi.rqi_hash = rqi_map[key]
            qi.rqi_status = "ALIGNED"
        else:
            qi.rqi_status = "MISSING"
    return qi_list

# =============================================================================
# MAPPING QI → CHAPITRE
# =============================================================================
def map_qi_to_chapter(qi: Qi, cap: CAP, pair: ExamPair) -> Qi:
    """Mapping Qi vers chapitre."""
    chapitres = cap.chapitres.get(pair.matiere, {}).get(pair.niveau, [])
    
    if not chapitres:
        qi.status = "ORPHAN_CHAPTER"
        return qi
    
    qi_text_norm = norm_text(qi.text)
    scores = []
    
    for ch in chapitres:
        score = sum(1.0 for kw in ch.keywords if kw.lower() in qi_text_norm)
        scores.append((score, ch.code, ch))
    
    # Tri stable: score desc, code asc
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    if scores:
        best_score, best_code, best_ch = scores[0]
        qi.chapter_code = best_ch.code
        qi.chapter_label = best_ch.label
        qi.mapping_score = decimal_round(best_score / max(1, len(best_ch.keywords)))
        qi.mapping_evidence = f"keywords_matched:{int(best_score)}"
        qi.status = "POSABLE"
    
    return qi

def gate_orphans_chapter(qi_list: List[Qi]) -> GateResult:
    """GATE: 0 Qi orpheline (sans chapitre)."""
    orphans = [qi for qi in qi_list if qi.status == "ORPHAN_CHAPTER"]
    passed = len(orphans) == 0
    
    return GateResult(
        code="GATE_ORPHANS_CHAPTER",
        passed=passed,
        evidence={"orphans_count": len(orphans)},
        message=f"Orphans (no chapter): {len(orphans)}"
    )

# =============================================================================
# QC GENERATION
# =============================================================================
def cluster_qi_to_qc(qi_list: List[Qi]) -> Dict[str, List[Qi]]:
    """Clustering Qi par chapitre."""
    clusters = defaultdict(list)
    for qi in qi_list:
        if qi.status == "POSABLE" and qi.chapter_code:
            clusters[qi.chapter_code].append(qi)
    return dict(clusters)

def build_qc(cluster: List[Qi], chapter: CAPChapter, matiere: str, niveau: str) -> QC:
    """Construction QC depuis cluster Qi."""
    FORMULA_ENGINE.load()
    
    qc_id = f"QC_{chapter.code}"
    canonical_text = f"Comment résoudre un problème de {chapter.label.lower()} ?"
    
    # T_j déterministes
    t_list = [0.3 + 0.1 * i for i in range(len(cluster))]
    
    # F1
    f1_result = FORMULA_ENGINE.compute_f1_raw(qc_id, chapter.code, chapter.delta_c, t_list)
    psi_raw = f1_result.get("psi_raw", 0.5)
    
    # F2
    f2_result = FORMULA_ENGINE.compute_f2(len(cluster), max(1, len(cluster) * 3), 30.0, min(psi_raw, 1.0))
    score_f2 = f2_result.get("score", 0.5)
    
    # ARI
    op_map = {"SUITES": "OP_RECURRENCE", "LIMITES": "OP_LIMIT", "DERIVATION": "OP_DERIVE",
              "INTEGRATION": "OP_INTEGRATE", "PROBAS": "OP_PROBABILITY", "COMPLEXES": "OP_COMPLEX"}
    primary_op = next((v for k, v in op_map.items() if k in chapter.code), "OP_STANDARD")
    
    ari = {"primary_op": primary_op, "sum_tj": decimal_round(sum(t_list))}
    
    # FRT
    frt_templates = {
        "OP_RECURRENCE": {"usage": "Démontrer par récurrence", "methode": "Init → Hérédité → Conclusion", "pieges": "Oublier l'initialisation"},
        "OP_LIMIT": {"usage": "Calculer limite", "methode": "Forme → Lever indétermination", "pieges": "Formes 0/0, ∞/∞"},
        "OP_DERIVE": {"usage": "Étudier variations", "methode": "f' → Signe → Tableau", "pieges": "Points critiques"},
        "OP_INTEGRATE": {"usage": "Calculer intégrale", "methode": "Type → IPP/Substitution", "pieges": "Intégrabilité"},
        "OP_PROBABILITY": {"usage": "Calculer probabilités", "methode": "Modéliser → Loi → Calculer", "pieges": "Indépendance"},
    }
    frt = frt_templates.get(primary_op, {"usage": f"Résoudre ({chapter.label})", "methode": "Analyser → Appliquer", "pieges": "Hypothèses"})
    
    triggers = [f"ARI:{primary_op}", f"SCOPE:{chapter.code}", f"MAT:{matiere}", f"NIV:{niveau}"]
    
    # IA2 Checks
    ia2_checks = {
        "CHK_CLUSTER_MIN": len(cluster) >= 1,
        "CHK_RQI_ALIGNED": sum(1 for qi in cluster if qi.rqi_status == "ALIGNED") >= len(cluster) * 0.5,
        "CHK_MAPPING_VALID": all(qi.mapping_score > 0 for qi in cluster),
        "CHK_PSI_VALID": psi_raw > 0,
    }
    ia2_pass = all(ia2_checks.values())
    
    # Lier Qi
    for qi in cluster:
        qi.qc_id = qc_id
    
    return QC(
        qc_id=qc_id,
        canonical_text=canonical_text,
        chapter_code=chapter.code,
        chapter_label=chapter.label,
        matiere=matiere,
        niveau=niveau,
        qi_ids=[qi.qi_id for qi in cluster],
        cluster_method="CHAPTER_KEYWORD",
        psi_raw=psi_raw,
        psi_q=min(psi_raw, 1.0),  # Sera normalisé
        score_f2=score_f2,
        ari=ari,
        frt=frt,
        triggers=triggers,
        evidence_pack=[qi.evidence_hash for qi in cluster],
        ia2_checks=ia2_checks,
        ia2_status="PASS" if ia2_pass else "FAIL",
    )

def normalize_qc_psi(qc_list: List[QC]) -> List[QC]:
    """Normalisation Ψ_q: max=1.0 par chapitre."""
    for chapter_code in set(qc.chapter_code for qc in qc_list):
        normalized = FORMULA_ENGINE.normalize_f1_chapter(chapter_code)
        for qc in qc_list:
            if qc.qc_id in normalized:
                qc.psi_q = normalized[qc.qc_id]
    return qc_list

def gate_orphans_qc(qi_list: List[Qi]) -> GateResult:
    """GATE: 0 Qi orpheline (sans QC)."""
    orphans = [qi for qi in qi_list if qi.status == "POSABLE" and qi.qc_id is None]
    passed = len(orphans) == 0
    
    return GateResult(
        code="GATE_ORPHANS_QC",
        passed=passed,
        evidence={"orphans_count": len(orphans)},
        message=f"Orphans (no QC): {len(orphans)}"
    )

# =============================================================================
# SATURATION
# =============================================================================
def run_saturation(cap: CAP, pairs: List[ExamPair], qi_list: List[Qi], qc_list: List[QC]) -> List[Dict]:
    """Boucle saturation (déterministe)."""
    logs = []
    
    covered_chapters = set(qc.chapter_code for qc in qc_list)
    all_chapters = set()
    for mat in cap.chapitres.values():
        for niv in mat.values():
            for ch in niv:
                all_chapters.add(ch.code)
    
    cumul_pairs = len(pairs)
    cumul_qi = len(qi_list)
    cumul_qc = len(qc_list)
    
    for cycle in range(1, MAX_SATURATION_CYCLES + 1):
        # Seed déterministe par cycle
        seed = stable_seed(f"saturation_cycle_{cycle}_{cap.pays}")
        
        # Simuler nouveaux éléments (déterministe)
        new_pairs = 2 + (seed % 4) if cycle <= 5 else 0
        new_qi = new_pairs * (3 + (seed % 2))
        
        uncovered = all_chapters - covered_chapters
        new_qc = min(len(uncovered), 1 + (seed % 2)) if cycle <= 3 and uncovered else 0
        
        # Mettre à jour
        cumul_pairs += new_pairs
        cumul_qi += new_qi
        cumul_qc += new_qc
        
        # Status
        if new_qc == 0 and cycle >= 3:
            status = "STABLE"
        else:
            status = "CONTINUE"
        
        logs.append({
            "cycle": cycle,
            "pairs_added": new_pairs,
            "qi_added": new_qi,
            "qc_added": new_qc,
            "new_QC": new_qc,
            "orphans": 0,
            "cumul_pairs": cumul_pairs,
            "cumul_qi": cumul_qi,
            "cumul_qc": cumul_qc,
            "status": status,
        })
        
        if status == "STABLE":
            break
    
    return logs

def gate_saturation(logs: List[Dict]) -> GateResult:
    """GATE: new_QC=0 atteint."""
    if not logs:
        return GateResult("GATE_SATURATION", False, {}, "No cycles")
    
    last = logs[-1]
    passed = last.get("status") == "STABLE" and last.get("new_QC", -1) == 0
    
    return GateResult(
        code="GATE_SATURATION",
        passed=passed,
        evidence={"cycles": len(logs), "last_new_qc": last.get("new_QC")},
        message=f"Cycles: {len(logs)}, last new_QC={last.get('new_QC')}"
    )

# =============================================================================
# EVIDENCE PACK
# =============================================================================
def export_evidence_pack(run_ctx: RunContext, cap: CAP, pairs: List[ExamPair],
                         qi_list: List[Qi], qc_list: List[QC]) -> Dict:
    """Export Evidence Pack."""
    return {
        "run_id": run_ctx.run_id,
        "timestamp": run_ctx.timestamp,
        "kernel_version": KERNEL_VERSION,
        "formules_version": FORMULES_VERSION,
        "execution_mode": EXECUTION_MODE,
        "pays": run_ctx.pays,
        "cap": {
            "cap_id": cap.cap_id,
            "seal_hash": cap.seal_hash,
            "sources_count": len(cap.sources),
        },
        "pairs": {
            "count": len(pairs),
            "ok_count": len([p for p in pairs if p.status == "OK"]),
        },
        "qi": {
            "count": len(qi_list),
            "posable": len([qi for qi in qi_list if qi.status == "POSABLE"]),
            "aligned": len([qi for qi in qi_list if qi.rqi_status == "ALIGNED"]),
        },
        "qc": {
            "count": len(qc_list),
            "ia2_pass": len([qc for qc in qc_list if qc.ia2_status == "PASS"]),
        },
        "gates": {k: {"passed": v.passed, "message": v.message} for k, v in run_ctx.gates.items()},
        "formula_checks": FORMULA_ENGINE.get_checks(),
        "saturation": run_ctx.saturation_log,
        "status": run_ctx.status,
    }

def seal_run(evidence: Dict) -> str:
    """Scellement run."""
    seal_data = {
        "run_id": evidence["run_id"],
        "cap_seal": evidence["cap"]["seal_hash"],
        "qc_count": evidence["qc"]["count"],
        "gates_all_pass": all(g["passed"] for g in evidence["gates"].values()),
    }
    return sha256_hash(json.dumps(seal_data, sort_keys=True))[:40]

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def activate_country(country_code: str) -> Tuple[RunContext, Optional[CAP], List[ExamPair], List[Qi], List[QC]]:
    """Pipeline complet."""
    if country_code not in PAYS_WHITELIST:
        return None, None, [], [], []
    
    # Run Context
    run_id = f"RUN_{country_code}_{sha256_hash(country_code + utc_ts())[:12]}"
    run_ctx = RunContext(
        run_id=run_id,
        pays=country_code,
        timestamp=utc_ts(),
        mode=EXECUTION_MODE,
        status="RUNNING",
    )
    
    # CAP
    cap_sources = discover_cap_sources(country_code)
    gate_cap = gate_cap_sources(cap_sources)
    run_ctx.gates["GATE_CAP_SOURCES_MIN"] = gate_cap
    
    if not gate_cap.passed:
        run_ctx.status = "STOPPED"
        return run_ctx, None, [], [], []
    
    cap = build_cap(country_code, cap_sources)
    seal_cap(cap)
    
    # PAIRS
    eval_sources = discover_eval_sources(country_code)
    pairs = harvest_pairs(eval_sources, cap, limit=25)
    
    gate_pairs = gate_pairs_min(pairs)
    run_ctx.gates["GATE_PAIRS_MIN"] = gate_pairs
    
    if not gate_pairs.passed:
        run_ctx.status = "STOPPED"
        return run_ctx, cap, pairs, [], []
    
    # QI EXTRACTION
    all_qi = []
    for pair in pairs:
        if pair.status == "OK":
            qi_list = extract_qi(pair, cap)
            rqi_map = extract_rqi(pair)
            qi_list = link_qi_rqi(qi_list, rqi_map)
            all_qi.extend(qi_list)
    
    gate_qi = GateResult("GATE_QI_NONEMPTY", len(all_qi) > 0, {"count": len(all_qi)}, f"Qi: {len(all_qi)}")
    run_ctx.gates["GATE_QI_NONEMPTY"] = gate_qi
    
    # MAPPING
    for qi in all_qi:
        pair = next((p for p in pairs if p.pair_id == qi.source_pair_id), None)
        if pair:
            map_qi_to_chapter(qi, cap, pair)
    
    gate_orph_ch = gate_orphans_chapter(all_qi)
    run_ctx.gates["GATE_ORPHANS_CHAPTER"] = gate_orph_ch
    
    # QC GENERATION
    FORMULA_ENGINE.load()
    clusters = cluster_qi_to_qc(all_qi)
    
    all_qc = []
    for chapter_code, cluster in clusters.items():
        chapter = None
        matiere = niveau = ""
        for mat, nivs in cap.chapitres.items():
            for niv, chs in nivs.items():
                for ch in chs:
                    if ch.code == chapter_code:
                        chapter = ch
                        matiere = mat
                        niveau = niv
                        break
        
        if chapter:
            qc = build_qc(cluster, chapter, matiere, niveau)
            all_qc.append(qc)
    
    all_qc = normalize_qc_psi(all_qc)
    
    gate_orph_qc = gate_orphans_qc(all_qi)
    run_ctx.gates["GATE_ORPHANS_QC"] = gate_orph_qc
    
    # SATURATION
    run_ctx.saturation_log = run_saturation(cap, pairs, all_qi, all_qc)
    gate_sat = gate_saturation(run_ctx.saturation_log)
    run_ctx.gates["GATE_SATURATION"] = gate_sat
    
    # FINAL
    all_gates_pass = all(g.passed for g in run_ctx.gates.values())
    run_ctx.status = "SEALED" if all_gates_pass else "UNVERIFIED"
    
    evidence = export_evidence_pack(run_ctx, cap, pairs, all_qi, all_qc)
    run_ctx.seal_hash = seal_run(evidence)
    
    return run_ctx, cap, pairs, all_qi, all_qc

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.set_page_config(page_title="SMAXIA GTE V7", page_icon="🏛️", layout="wide")
    
    for k in ["run_ctx", "cap", "pairs", "qi", "qc"]:
        if k not in st.session_state:
            st.session_state[k] = None if k in ["run_ctx", "cap"] else []
    
    st.markdown("# 🏛️ SMAXIA GTE — Government Test Engine")
    st.markdown(f"**Kernel {KERNEL_VERSION} | Formules {FORMULES_VERSION} (A2) | CAS 1 ONLY | Mode: {EXECUTION_MODE}**")
    
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
        if st.session_state.run_ctx:
            ctx = st.session_state.run_ctx
            st.success(f"✅ {PAYS_META[ctx.pays]['flag']} {PAYS_META[ctx.pays]['nom']}")
            st.metric("Status", ctx.status)
            st.metric("Pairs", len(st.session_state.pairs))
            st.metric("Qi", len(st.session_state.qi))
            st.metric("QC", len(st.session_state.qc))

    # === MENU 1: PAYS ===
    if menu == "1️⃣ Pays (Activation)":
        st.header("🌍 Activation Pays — Full Auto Pipeline")
        
        if EXECUTION_MODE == "DEMO":
            st.info("⚠️ **Mode DEMO** — Données simulées pour test UI. En PROD: scraping réel.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🇫🇷 France")
            st.markdown("BAC, Prépa, Concours")
            if st.button("🚀 Activer France", type="primary", use_container_width=True):
                with st.spinner("Pipeline en cours..."):
                    run_ctx, cap, pairs, qi, qc = activate_country("FR")
                    st.session_state.run_ctx = run_ctx
                    st.session_state.cap = cap
                    st.session_state.pairs = pairs
                    st.session_state.qi = qi
                    st.session_state.qc = qc
                st.rerun()
        
        with c2:
            st.markdown("### 🇨🇮 Côte d'Ivoire")
            st.markdown("BAC Séries C/D, CAMES")
            if st.button("🚀 Activer Côte d'Ivoire", type="primary", use_container_width=True):
                with st.spinner("Pipeline en cours..."):
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
            
            if ctx.status == "SEALED":
                st.success(f"### ✅ RUN SCELLÉ — {ctx.run_id[:24]}")
            elif ctx.status == "STOPPED":
                st.error("### ⛔ RUN ARRÊTÉ")
            else:
                st.warning(f"### ⚠️ {ctx.status}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pairs", len(st.session_state.pairs))
            c2.metric("Qi", len(st.session_state.qi))
            c3.metric("QC", len(st.session_state.qc))
            c4.metric("Seal", ctx.seal_hash[:12] if ctx.seal_hash else "N/A")

    # === MENU 2: CAP ===
    elif menu == "2️⃣ CAP":
        st.header("📦 CAP — Country Academic Pack")
        if not st.session_state.cap:
            st.warning("⚠️ Activez d'abord un pays"); return
        
        cap = st.session_state.cap
        p = PAYS_META[cap.pays]
        
        st.markdown(f"### {p['flag']} CAP {p['nom']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("CAP ID", cap.cap_id[:20])
        c2.metric("Status", cap.status)
        c3.metric("Seal", cap.seal_hash[:12] if cap.seal_hash else "N/A")
        
        st.markdown("---")
        st.subheader("📋 Sources CAP")
        for src in cap.sources:
            with st.expander(f"🔗 {src.domain}", expanded=False):
                st.markdown(f"**URL:** `{src.url}`")
                st.markdown(f"**Hash:** `{src.content_hash}`")
                st.markdown(f"**Type:** {src.source_type} | **Score:** {src.authority_score}")
        
        st.markdown("---")
        st.subheader("📚 Chapitres")
        total_ch = sum(len(chs) for mat in cap.chapitres.values() for chs in mat.values())
        st.metric("Total Chapitres", total_ch)
        
        for niv_code in sorted(NIVEAUX_WHITELIST, key=lambda x: NIVEAUX_META[x]["ordre"]):
            with st.expander(f"📘 {NIVEAUX_META[niv_code]['label']}", expanded=False):
                for mat_code in MATIERES_WHITELIST:
                    chs = cap.chapitres.get(mat_code, {}).get(niv_code, [])
                    if chs:
                        st.markdown(f"#### {MATIERES_META[mat_code]['icon']} {MATIERES_META[mat_code]['label']}")
                        data = [{"Code": ch.code, "Label": ch.label, "δc": ch.delta_c, "Keywords": ", ".join(ch.keywords[:3])} for ch in chs]
                        st.dataframe(data, use_container_width=True, hide_index=True)

    # === MENU 3: SUJETS ===
    elif menu == "3️⃣ Sujets/Corrections":
        st.header("📄 Sujets et Corrections")
        if not st.session_state.pairs:
            st.warning("⚠️ Activez d'abord un pays"); return
        
        pairs = st.session_state.pairs
        c1, c2 = st.columns(2)
        c1.metric("Total Pairs", len(pairs))
        c2.metric("OK", len([p for p in pairs if p.status == "OK"]))
        
        st.markdown("---")
        for pair in pairs[:15]:
            with st.expander(f"{'✅' if pair.status == 'OK' else '⚠️'} {pair.nom} — {pair.matiere}/{pair.niveau}", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📄 Sujet**")
                    st.markdown(f"Hash: `{pair.sujet_hash}`")
                    st.markdown(f"Size: {pair.sujet_size_kb} KB")
                with c2:
                    st.markdown("**📝 Corrigé**")
                    st.markdown(f"Hash: `{pair.corrige_hash}`")
                    st.markdown(f"Size: {pair.corrige_size_kb} KB")

    # === MENU 4: QI ===
    elif menu == "4️⃣ Qi/RQi/Mapping":
        st.header("🔬 Qi / RQi / Mapping")
        if not st.session_state.qi:
            st.warning("⚠️ Activez d'abord un pays"); return
        
        qi_list = st.session_state.qi
        posable = len([qi for qi in qi_list if qi.status == "POSABLE"])
        aligned = len([qi for qi in qi_list if qi.rqi_status == "ALIGNED"])
        orphans = len([qi for qi in qi_list if "ORPHAN" in qi.status])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Qi", len(qi_list))
        c2.metric("Posable", posable)
        c3.metric("RQi Aligned", aligned)
        c4.metric("Orphans", orphans)
        
        if orphans > 0:
            st.error(f"⛔ {orphans} Qi orphelines!")
        
        st.markdown("---")
        for qi in qi_list[:15]:
            icon = "✅" if qi.status == "POSABLE" else "⚠️"
            with st.expander(f"{icon} {qi.qi_id} — {qi.chapter_label or 'N/A'}", expanded=False):
                st.markdown(f"**Text:** {qi.text[:80]}...")
                st.markdown(f"**Pair:** {qi.source_pair_id} | **Score:** {qi.mapping_score:.2f}")
                st.markdown(f"**RQi:** {qi.rqi_status}")

    # === MENU 5: QC ===
    elif menu == "5️⃣ QC/FRT/ARI/Triggers":
        st.header("🎯 QC / FRT / ARI / Triggers")
        if not st.session_state.qc:
            st.warning("⚠️ Activez d'abord un pays"); return
        
        qc_list = st.session_state.qc
        qi_list = st.session_state.qi
        
        ia2_pass = len([qc for qc in qc_list if qc.ia2_status == "PASS"])
        total_qi = len(qi_list)
        orphans = len([qi for qi in qi_list if qi.qc_id is None and qi.status == "POSABLE"])
        coverage = (total_qi - orphans) / max(1, total_qi) * 100
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QC", len(qc_list))
        c2.metric("IA2 PASS", f"{ia2_pass}/{len(qc_list)}")
        c3.metric("Coverage", f"{coverage:.1f}%")
        c4.metric("Orphans", orphans)
        
        st.markdown("---")
        for niv_code in sorted(NIVEAUX_WHITELIST, key=lambda x: NIVEAUX_META[x]["ordre"]):
            niv_qcs = [qc for qc in qc_list if qc.niveau == niv_code]
            if not niv_qcs: continue
            
            st.markdown(f"## 📘 {NIVEAUX_META[niv_code]['label']}")
            
            for mat_code in MATIERES_WHITELIST:
                mat_qcs = [qc for qc in niv_qcs if qc.matiere == mat_code]
                if not mat_qcs: continue
                
                st.markdown(f"### {MATIERES_META[mat_code]['icon']} {MATIERES_META[mat_code]['label']}")
                
                for qc in mat_qcs:
                    icon = "✅" if qc.ia2_status == "PASS" else "❌"
                    with st.expander(f"{icon} {qc.chapter_label} — {qc.qc_id}", expanded=False):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Ψ_raw", f"{qc.psi_raw:.3f}")
                        c2.metric("Ψq", f"{qc.psi_q:.3f}")
                        c3.metric("F2", f"{qc.score_f2:.4f}")
                        c4.metric("Qi", len(qc.qi_ids))
                        
                        st.markdown("**📝 Question Canonique**")
                        st.info(qc.canonical_text)
                        
                        st.markdown("**📋 FRT**")
                        st.code(f"Usage: {qc.frt['usage']}\nMéthode: {qc.frt['methode']}\nPièges: {qc.frt['pieges']}")
                        
                        st.markdown("**🧠 ARI**")
                        st.code(f"Op: {qc.ari['primary_op']} | ΣTj: {qc.ari['sum_tj']}")
                        
                        st.markdown("**🎯 Triggers**")
                        st.code(" | ".join(qc.triggers))

    # === MENU 6: AUDIT ===
    elif menu == "6️⃣ Audit & Gates":
        st.header("📊 Audit & Gates")
        if not st.session_state.run_ctx:
            st.warning("⚠️ Activez d'abord un pays"); return
        
        ctx = st.session_state.run_ctx
        
        st.subheader("✅ Gates Binaires")
        all_pass = True
        for code, gate in ctx.gates.items():
            if gate.passed:
                st.success(f"✅ **{code}** — {gate.message}")
            else:
                st.error(f"❌ **{code}** — {gate.message}")
                all_pass = False
        
        st.markdown("---")
        st.subheader("🔬 Formula Engine Checks")
        for chk, passed in FORMULA_ENGINE.get_checks().items():
            st.markdown(f"{'✅' if passed else '❌'} {chk}")
        
        st.markdown("---")
        if all_pass:
            st.success("### ✅ ÉLIGIBLE AU SCELLEMENT")
        else:
            st.error("### ⛔ NON SCELLABLE")
        
        st.markdown("---")
        st.subheader("📥 Export Evidence Pack")
        if st.button("📦 Générer JSON"):
            evidence = export_evidence_pack(ctx, st.session_state.cap, st.session_state.pairs,
                                           st.session_state.qi, st.session_state.qc)
            st.json(evidence)
            st.download_button("⬇️ Download", json.dumps(evidence, indent=2),
                             f"smaxia_evidence_{ctx.pays}.json")

    # === MENU 7: SATURATION ===
    elif menu == "7️⃣ Saturation":
        st.header("🔄 Boucle de Saturation")
        if not st.session_state.run_ctx:
            st.warning("⚠️ Activez d'abord un pays"); return
        
        ctx = st.session_state.run_ctx
        
        if not ctx.saturation_log:
            st.warning("Aucune donnée"); return
        
        st.subheader("📊 Métriques par Cycle")
        data = [{
            "Cycle": log["cycle"],
            "Pairs+": log["pairs_added"],
            "Qi+": log["qi_added"],
            "QC+": log["qc_added"],
            "new_QC": log["new_QC"],
            "Cumul QC": log["cumul_qc"],
            "Status": log["status"],
        } for log in ctx.saturation_log]
        st.dataframe(data, use_container_width=True, hide_index=True)
        
        last = ctx.saturation_log[-1]
        if last["status"] == "STABLE":
            st.success(f"### ✅ SATURATION — Cycle {last['cycle']}")
        else:
            st.warning(f"### ⏳ En cours — {last['cycle']} cycles")

if __name__ == "__main__":
    main()
