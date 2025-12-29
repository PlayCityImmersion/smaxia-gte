# smaxia_console_v31.py
# =============================================================================
# SMAXIA GTE CONSOLE V31 — KERNEL V10.6.1 — STREAMLIT
# =============================================================================
# Version UNIQUE à copier-coller dans GitHub
# Pipeline complet: Miner → Clustering → Builder → Judge → Scoring → Coverage → Seal
# =============================================================================

import streamlit as st
import pandas as pd
import hashlib
import json
import re
import io
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
from enum import Enum
from collections import defaultdict, Counter
from datetime import datetime

# Imports optionnels PDF
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# =============================================================================
# SECTION 1 — CONSTANTES KERNEL SCELLÉES V10.6.1
# =============================================================================

class KernelConstants:
    """Constantes scellées du Kernel V10.6.1 — NE PAS MODIFIER"""
    VERSION = "V10.6.1"
    DATE = "2025-12-28"
    EPSILON = 0.1
    MAX_ITERATIONS = 3
    COHERENCE_THRESHOLD = 0.70
    SIMILARITY_THRESHOLD = 0.85
    ORPHAN_THRESHOLD = 0.02
    ORPHAN_ABSOLUTE = 2
    QC_PREFIX = "Comment "
    QC_SUFFIX = "?"

# =============================================================================
# SECTION 2 — ENUMS
# =============================================================================

class Verdict(Enum):
    PASS = "PASS"
    FAIL = "FAIL"

class SealStatus(Enum):
    SEALED = "SEALED"
    NOT_SEALABLE = "NOT_SEALABLE"
    PENDING = "PENDING"

class OutputType(Enum):
    RESULT = "RESULT_VALUE"
    PROOF = "PROOF"
    INTERVAL = "INTERVAL"
    GRAPH = "GRAPH"
    EXPRESSION = "EXPRESSION"
    UNKNOWN = "UNKNOWN"

# =============================================================================
# SECTION 3 — TABLE COGNITIVE KERNEL
# =============================================================================

COGNITIVE_TABLE: Dict[str, float] = {
    # T_j = 0.15
    "IDENTIFIER": 0.15, "REPERER": 0.15, "NOMMER": 0.15, "DEFINIR": 0.15,
    # T_j = 0.20
    "ANALYSER": 0.20, "OBSERVER": 0.20, "CONCLURE": 0.20, "CITER": 0.20,
    # T_j = 0.25
    "SIMPLIFIER": 0.25, "FACTORISER": 0.25, "REDUIRE": 0.25, "ILLUSTRER": 0.25,
    # T_j = 0.30
    "CALCULER": 0.30, "EXPRIMER": 0.30, "FORMULER": 0.30, "MESURER": 0.30,
    # T_j = 0.35
    "COMPARER": 0.35, "APPLIQUER": 0.35, "UTILISER": 0.35, "DISTINGUER": 0.35,
    # T_j = 0.40
    "RESOUDRE": 0.40, "DETERMINER": 0.40, "DERIVER": 0.40, "INTEGRER": 0.40,
    # T_j = 0.45
    "JUSTIFIER": 0.45, "ARGUMENTER": 0.45, "EVALUER": 0.45,
    # T_j = 0.50
    "DEMONTRER": 0.50, "PROUVER": 0.50,
    # T_j = 0.60
    "RECURRENCE": 0.60,
    # Fallback
    "OTHER": 0.10,
}

VERB_SYNONYMS: Dict[str, str] = {
    "calculer": "CALCULER", "calcul": "CALCULER", "calculons": "CALCULER",
    "déterminer": "DETERMINER", "détermin": "DETERMINER", "determine": "DETERMINER",
    "résoudre": "RESOUDRE", "réso": "RESOUDRE", "resoudre": "RESOUDRE",
    "démontrer": "DEMONTRER", "démontr": "DEMONTRER", "montrer": "DEMONTRER",
    "prouver": "PROUVER", "prouv": "PROUVER",
    "simplifier": "SIMPLIFIER", "simplifi": "SIMPLIFIER",
    "factoriser": "FACTORISER", "factoris": "FACTORISER",
    "appliquer": "APPLIQUER", "appliqu": "APPLIQUER",
    "conclure": "CONCLURE", "conclu": "CONCLURE", "donc": "CONCLURE", "ainsi": "CONCLURE",
    "identifier": "IDENTIFIER", "identifi": "IDENTIFIER",
    "justifier": "JUSTIFIER", "justifi": "JUSTIFIER",
    "comparer": "COMPARER", "compar": "COMPARER",
    "exprimer": "EXPRIMER", "exprim": "EXPRIMER",
}

def map_verb(observed: str) -> Tuple[str, float]:
    """Mappe un verbe observé vers son ID canonique"""
    v = observed.lower().strip()
    if v in VERB_SYNONYMS:
        return VERB_SYNONYMS[v], 1.0
    for syn, canonical in VERB_SYNONYMS.items():
        if v.startswith(syn[:4]) or syn.startswith(v[:4]):
            return canonical, 0.8
    return "OTHER", 0.0

def get_weight(verb_id: str) -> float:
    """Retourne le poids T_j d'un verbe"""
    return COGNITIVE_TABLE.get(verb_id.upper(), 0.10)

# =============================================================================
# SECTION 4 — UTILITAIRES
# =============================================================================

def sha256(content: str) -> str:
    """Calcule SHA256 au format Kernel"""
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

def clean_text(text: str) -> str:
    """Nettoie le texte extrait"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# =============================================================================
# SECTION 5 — DATACLASSES
# =============================================================================

@dataclass
class Qi:
    """Question atomisée"""
    qi_id: str
    text: str
    chapitre: str
    exo_num: str = ""
    q_num: str = ""
    page: int = 0
    fingerprint: str = ""
    
    def compute_hash(self):
        self.fingerprint = sha256(f"{self.text}|{self.chapitre}")
        return self.fingerprint

@dataclass
class Rqi:
    """Réponse atomisée (corrigé)"""
    rqi_id: str
    qi_id: str
    text: str
    page: int = 0
    fingerprint: str = ""
    
    def compute_hash(self):
        self.fingerprint = sha256(f"{self.text}|{self.qi_id}")
        return self.fingerprint

@dataclass
class Step:
    """Étape ARI"""
    idx: int
    observed: str
    mapped: str
    confidence: float
    
    @property
    def weight(self) -> float:
        return get_weight(self.mapped)

@dataclass
class Trace:
    """Trace ARI extraite par Miner"""
    qi_id: str
    chapitre: str
    steps: List[Step]
    output_type: OutputType
    signature: str = ""
    
    def compute_signature(self):
        spine = "|".join([s.mapped for s in self.steps])
        self.signature = hashlib.md5(spine.encode()).hexdigest()[:12]
        return self.signature
    
    @property
    def action_spine(self) -> List[str]:
        return [s.mapped for s in self.steps]

@dataclass
class Cluster:
    """Cluster de traces similaires"""
    cluster_id: str
    chapitre: str
    qi_ids: List[str]
    n_q: int
    coherence: float
    pattern: List[str] = field(default_factory=list)
    champion_id: str = ""
    
    @property
    def is_valid(self) -> bool:
        """Règle anti-singleton: n_q >= 2"""
        return self.n_q >= 2

@dataclass
class QC:
    """Question Clé candidate"""
    qc_id: str
    chapitre: str
    cluster_id: str
    text: str
    ari: List[str]
    triggers: List[str]
    n_q: int
    psi: float = 0.0
    score: float = 0.0
    validated: bool = False
    covered_qis: Set[str] = field(default_factory=set)
    
    def sum_tj(self) -> float:
        return sum(get_weight(s) for s in self.ari)

@dataclass
class Coverage:
    """Couverture d'un chapitre"""
    chapitre: str
    universe: Set[str]
    covered: Set[str] = field(default_factory=set)
    orphans: Set[str] = field(default_factory=set)
    selected_qcs: List[str] = field(default_factory=list)
    
    @property
    def ratio(self) -> float:
        return len(self.covered) / len(self.universe) if self.universe else 1.0
    
    @property
    def is_sealed(self) -> bool:
        return len(self.orphans) == 0

# =============================================================================
# SECTION 6 — IA1 MINER
# =============================================================================

class IA1Miner:
    """
    IA1 Miner — Extraction factuelle des traces ARI
    Règle: AUCUNE généralisation, extraction pure depuis RQi
    """
    
    def extract(self, qi: Qi, rqi: Rqi) -> Trace:
        """Extrait une trace ARI depuis un couple Qi/RQi"""
        segments = self._segment(rqi.text)
        steps = []
        
        for idx, seg in enumerate(segments, 1):
            action = self._detect_action(seg)
            if action:
                mapped, conf = map_verb(action)
                steps.append(Step(idx, action, mapped, conf))
        
        output_type = self._detect_output_type(rqi.text)
        trace = Trace(qi.qi_id, qi.chapitre, steps, output_type)
        trace.compute_signature()
        return trace
    
    def _segment(self, text: str) -> List[str]:
        """Segmente le corrigé en blocs logiques"""
        # Découpage par phrases
        segments = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in segments if s.strip()]
    
    def _detect_action(self, segment: str) -> Optional[str]:
        """Détecte l'action principale d'un segment"""
        seg_lower = segment.lower()
        
        # Patterns d'actions
        patterns = [
            r'(on\s+)?(va\s+)?(calcul|détermin|réso|démontr|prouv|simplifi|factoris|appliqu)',
            r'(calculons|déterminons|résolvons|démontrons|prouvons)',
            r'(donc|ainsi|par\s+conséquent|finalement|on\s+obtient)',
            r'(on\s+identifi|on\s+compar|on\s+justifi|on\s+exprim)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, seg_lower)
            if match:
                return match.group(0).strip()
        
        return None
    
    def _detect_output_type(self, text: str) -> OutputType:
        """Détecte le type de sortie"""
        t = text.lower()
        if any(x in t for x in ["donc", "cqfd", "ce qu'il fallait"]):
            return OutputType.PROOF
        if "intervalle" in t or "ensemble" in t:
            return OutputType.INTERVAL
        if "courbe" in t or "graphe" in t:
            return OutputType.GRAPH
        return OutputType.RESULT

# =============================================================================
# SECTION 7 — CLUSTERING
# =============================================================================

class Clusterer:
    """
    Clustering des traces ARI
    Règle anti-singleton: cluster valide si n_q >= 2
    """
    
    def cluster(self, traces: List[Trace], chapitre: str) -> Tuple[List[Cluster], List[Trace]]:
        """Regroupe les traces par signature ARI"""
        # Grouper par signature
        groups: Dict[str, List[Trace]] = defaultdict(list)
        for t in traces:
            groups[t.signature].append(t)
        
        clusters = []
        rejected = []
        
        for idx, (sig, grp) in enumerate(groups.items()):
            if len(grp) < 2:
                # Singleton → rejeté
                rejected.extend(grp)
                continue
            
            # Pattern majoritaire
            pattern_counts = Counter(tuple(t.action_spine) for t in grp)
            majority_pattern = list(pattern_counts.most_common(1)[0][0])
            
            # Cohérence
            matching = sum(1 for t in grp if t.action_spine == majority_pattern)
            coherence = matching / len(grp)
            
            cluster = Cluster(
                cluster_id=f"CLU_{chapitre}_{idx:03d}",
                chapitre=chapitre,
                qi_ids=[t.qi_id for t in grp],
                n_q=len(grp),
                coherence=coherence,
                pattern=majority_pattern,
                champion_id=grp[0].qi_id
            )
            clusters.append(cluster)
        
        return clusters, rejected

# =============================================================================
# SECTION 8 — IA1 BUILDER
# =============================================================================

class IA1Builder:
    """
    IA1 Builder — Synthèse des QC depuis clusters
    Règle: n_q >= 2 obligatoire, format "Comment ... ?"
    """
    
    def build(self, cluster: Cluster, qis: Dict[str, Qi]) -> Optional[QC]:
        """Construit une QC depuis un cluster validé"""
        if not cluster.is_valid:
            return None
        
        # Générer le texte QC
        text = self._generate_qc_text(cluster)
        
        # Extraire triggers depuis les Qi du cluster
        triggers = self._extract_triggers(cluster, qis)
        
        return QC(
            qc_id=f"QC_{cluster.cluster_id}",
            chapitre=cluster.chapitre,
            cluster_id=cluster.cluster_id,
            text=text,
            ari=cluster.pattern,
            triggers=triggers,
            n_q=cluster.n_q,
            covered_qis=set(cluster.qi_ids)
        )
    
    def _generate_qc_text(self, cluster: Cluster) -> str:
        """Génère le texte QC au format canonique"""
        if cluster.pattern:
            main_action = cluster.pattern[0].lower()
            action_map = {
                "calculer": "calculer",
                "determiner": "déterminer",
                "demontrer": "démontrer",
                "prouver": "prouver",
                "resoudre": "résoudre",
                "simplifier": "simplifier",
            }
            verb = action_map.get(main_action, main_action)
            return f"Comment {verb} ce type de problème ?"
        return "Comment résoudre ce type de problème ?"
    
    def _extract_triggers(self, cluster: Cluster, qis: Dict[str, Qi]) -> List[str]:
        """Extrait les triggers communs"""
        triggers = []
        trigger_patterns = [
            r"(montrer|démontrer)\s+que",
            r"(déterminer|calculer)\s+(la|le|les)",
            r"résoudre\s+(l'équation|le\s+système)",
            r"exprimer.+en\s+fonction",
        ]
        
        for qi_id in cluster.qi_ids[:5]:
            qi = qis.get(qi_id)
            if qi:
                for pattern in trigger_patterns:
                    if re.search(pattern, qi.text.lower()):
                        triggers.append(pattern)
        
        return list(set(triggers))[:3]

# =============================================================================
# SECTION 9 — IA2 JUDGE
# =============================================================================

class IA2Judge:
    """
    IA2 Judge — Validation booléenne stricte
    Règle: PASS/FAIL uniquement, un seul FAIL = QC rejetée
    """
    
    def validate(self, qc: QC) -> Tuple[Verdict, List[str]]:
        """Valide une QC candidate"""
        issues = []
        
        # Check 1: Format QC
        if not qc.text.startswith(KernelConstants.QC_PREFIX):
            issues.append("QC_FORM: doit commencer par 'Comment '")
        if not qc.text.endswith(KernelConstants.QC_SUFFIX):
            issues.append("QC_FORM: doit terminer par '?'")
        
        # Check 2: Cluster minimum
        if qc.n_q < 2:
            issues.append(f"CLUSTER_MIN_SIZE: n_q={qc.n_q} < 2 (singleton interdit)")
        
        # Check 3: ARI non vide
        if not qc.ari:
            issues.append("ARI_TYPED_ONLY: ARI vide")
        
        # Check 4: ARI dans Table Cognitive
        for step in qc.ari:
            if step not in COGNITIVE_TABLE:
                issues.append(f"VERB_ONTOLOGY: '{step}' non reconnu")
        
        if issues:
            return Verdict.FAIL, issues
        
        qc.validated = True
        return Verdict.PASS, []

# =============================================================================
# SECTION 10 — SCORING F1/F2
# =============================================================================

class Scorer:
    """
    Calcul des scores F1 (Ψ_q) et F2
    """
    
    def compute_f1(self, qc: QC, all_qcs: List[QC], delta: float = 1.0) -> float:
        """
        F1: Ψ_q = ψ_raw / max(ψ_raw)
        ψ_raw = δ × (ε + Σ T_j)²
        """
        epsilon = KernelConstants.EPSILON
        psi_raw = delta * (epsilon + qc.sum_tj()) ** 2
        
        max_psi = psi_raw
        for q in all_qcs:
            q_psi = delta * (epsilon + q.sum_tj()) ** 2
            max_psi = max(max_psi, q_psi)
        
        qc.psi = psi_raw / max_psi if max_psi > 0 else 1.0
        return qc.psi
    
    def compute_f2(self, qc: QC, selected: List[QC], n_total: int, alpha: float = 0.5) -> float:
        """
        F2: Score de sélection avec anti-redondance
        """
        freq = qc.n_q / n_total if n_total > 0 else 0
        recency = 1 + alpha / 0.5
        
        # Orthogonalité
        ortho = 1.0
        for p in selected:
            sim = self._similarity(qc.ari, p.ari)
            ortho *= (1 - sim)
        
        qc.score = freq * recency * qc.psi * ortho * 100
        return qc.score
    
    def _similarity(self, ari1: List[str], ari2: List[str]) -> float:
        """Similarité Jaccard entre deux ARI"""
        s1, s2 = set(ari1), set(ari2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

# =============================================================================
# SECTION 11 — COVERAGE MANAGER
# =============================================================================

class CoverageManager:
    """
    Sélection gloutonne coverage-driven
    Objectif: Coverage = 100% sur POSABLE
    """
    
    def select(self, candidates: List[QC], qis: List[Qi], scorer: Scorer, n_total: int) -> Tuple[List[QC], Coverage]:
        """Sélection gloutonne maximisant la couverture"""
        universe = set(q.qi_id for q in qis)
        uncovered = universe.copy()
        selected = []
        
        # Cache de couverture
        cover_cache = {qc.qc_id: qc.covered_qis for qc in candidates}
        
        while uncovered:
            best_qc = None
            best_gain = 0
            
            for qc in candidates:
                if qc in selected:
                    continue
                gain = len(cover_cache[qc.qc_id] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_qc = qc
            
            if not best_qc or best_gain == 0:
                break
            
            selected.append(best_qc)
            uncovered -= cover_cache[best_qc.qc_id]
        
        chapitre = qis[0].chapitre if qis else ""
        coverage = Coverage(
            chapitre=chapitre,
            universe=universe,
            covered=universe - uncovered,
            orphans=uncovered,
            selected_qcs=[qc.qc_id for qc in selected]
        )
        
        return selected, coverage

# =============================================================================
# SECTION 12 — PIPELINE GTE COMPLET
# =============================================================================

class GTEPipeline:
    """
    Pipeline GTE complet — Phases 4-10
    """
    
    def __init__(self):
        self.miner = IA1Miner()
        self.clusterer = Clusterer()
        self.builder = IA1Builder()
        self.judge = IA2Judge()
        self.scorer = Scorer()
        self.coverage_mgr = CoverageManager()
        
        # État
        self.traces: List[Trace] = []
        self.clusters: List[Cluster] = []
        self.candidates: List[QC] = []
        self.validated: List[QC] = []
        self.selected: List[QC] = []
        self.coverage: Optional[Coverage] = None
        self.logs: List[str] = []
    
    def log(self, phase: str, message: str):
        entry = f"[{phase}] {message}"
        self.logs.append(entry)
    
    def run(self, qis: List[Qi], rqis: Dict[str, Rqi]) -> Dict[str, Any]:
        """Exécute le pipeline complet"""
        self.logs = []
        self.log("INIT", f"Pipeline GTE {KernelConstants.VERSION} — {len(qis)} Qi, {len(rqis)} RQi")
        
        if not qis:
            return {"status": "ERROR", "message": "Aucune Qi fournie"}
        
        chapitre = qis[0].chapitre
        
        # Phase 4: IA1 Miner
        self.traces = []
        for qi in qis:
            rqi = rqis.get(qi.qi_id)
            if rqi:
                trace = self.miner.extract(qi, rqi)
                self.traces.append(trace)
        self.log("MINER", f"Traces extraites: {len(self.traces)}")
        
        if not self.traces:
            return {"status": "ERROR", "message": "Aucune trace extraite"}
        
        # Phase 5: Clustering
        self.clusters, rejected = self.clusterer.cluster(self.traces, chapitre)
        self.log("CLUSTERING", f"Clusters: {len(self.clusters)}, Singletons rejetés: {len(rejected)}")
        
        if not self.clusters:
            return {"status": "ERROR", "message": "Aucun cluster valide (que des singletons)"}
        
        # Phase 6: IA1 Builder
        qi_dict = {q.qi_id: q for q in qis}
        self.candidates = []
        for cluster in self.clusters:
            qc = self.builder.build(cluster, qi_dict)
            if qc:
                self.candidates.append(qc)
        self.log("BUILDER", f"QC candidates: {len(self.candidates)}")
        
        # Phase 7: IA2 Judge
        self.validated = []
        for qc in self.candidates:
            verdict, issues = self.judge.validate(qc)
            if verdict == Verdict.PASS:
                self.validated.append(qc)
            else:
                self.log("IA2", f"FAIL {qc.qc_id}: {issues[0] if issues else 'unknown'}")
        self.log("IA2", f"QC validées: {len(self.validated)}/{len(self.candidates)}")
        
        if not self.validated:
            return {"status": "ERROR", "message": "Toutes les QC rejetées par IA2"}
        
        # Phase 8: Scoring F1
        for qc in self.validated:
            self.scorer.compute_f1(qc, self.validated)
        
        # Phase 9: Sélection coverage-driven
        n_total = len(qis)
        self.selected, self.coverage = self.coverage_mgr.select(
            self.validated, qis, self.scorer, n_total
        )
        self.log("SELECTION", f"QC sélectionnées: {len(self.selected)}, Coverage: {self.coverage.ratio:.1%}")
        
        # Phase 10: Verdict
        status = SealStatus.SEALED if self.coverage.is_sealed else SealStatus.NOT_SEALABLE
        self.log("VERDICT", f"{status.value} — Orphelins: {len(self.coverage.orphans)}")
        
        return {
            "status": status.value,
            "chapitre": chapitre,
            "kernel_version": KernelConstants.VERSION,
            "metrics": {
                "qi_total": len(qis),
                "rqi_total": len(rqis),
                "traces": len(self.traces),
                "clusters": len(self.clusters),
                "candidates": len(self.candidates),
                "validated": len(self.validated),
                "selected": len(self.selected),
                "coverage_ratio": self.coverage.ratio,
                "orphans": len(self.coverage.orphans),
            },
            "selected_qcs": [
                {"qc_id": qc.qc_id, "text": qc.text, "n_q": qc.n_q, "psi": qc.psi}
                for qc in self.selected
            ],
            "logs": self.logs
        }

# =============================================================================
# SECTION 13 — EXTRACTION PDF
# =============================================================================

def extract_questions_from_pdf(pdf_file) -> List[Dict]:
    """Extrait les questions d'un PDF de sujet BAC"""
    if pdfplumber is None:
        st.error("pdfplumber non installé. Installez avec: pip install pdfplumber")
        return []
    
    questions = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            full_text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                full_text += f"\n[PAGE {page_num}]\n{text}"
            
            # Patterns de questions BAC
            patterns = [
                r'(\d+[\.\)]\s*[a-z][\.\)]\s*)([^\.]+\?)',
                r'(Exercice\s+\d+.*?Question\s+\d+[:\.]?\s*)([^\.]+)',
                r'(\d+[\.\)]\s*)([A-Z][^\.]+\?)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    q_text = clean_text(match[1] if len(match) > 1 else match[0])
                    if len(q_text) > 10:
                        questions.append({
                            "text": q_text,
                            "source": "PDF extraction"
                        })
    except Exception as e:
        st.error(f"Erreur extraction PDF: {e}")
    
    return questions

# =============================================================================
# SECTION 14 — INTERFACE STREAMLIT
# =============================================================================

def main():
    st.set_page_config(
        page_title="SMAXIA GTE Console V31",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 SMAXIA GTE Console V31")
    st.markdown(f"**Kernel {KernelConstants.VERSION}** — Pipeline de Garantie Totale d'Extraction")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        chapitre = st.text_input("Chapitre", value="SUITES")
        
        st.markdown("---")
        st.markdown("### 📊 Constantes Kernel")
        st.code(f"""
VERSION: {KernelConstants.VERSION}
EPSILON: {KernelConstants.EPSILON}
MAX_ITER: {KernelConstants.MAX_ITERATIONS}
COHERENCE: {KernelConstants.COHERENCE_THRESHOLD}
        """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📥 Données", "🚀 Pipeline", "📊 Résultats"])
    
    # Tab 1: Données
    with tab1:
        st.header("📥 Entrée des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Questions (Qi)")
            qi_input = st.text_area(
                "Une question par ligne",
                height=200,
                placeholder="Calculer la limite de u_n = (n²+3n)/(2n²+1)\nDéterminer la limite de v_n = (3n²-n)/(n²+5)"
            )
        
        with col2:
            st.subheader("Corrigés (RQi)")
            rqi_input = st.text_area(
                "Un corrigé par ligne (même ordre que Qi)",
                height=200,
                placeholder="On calcule la limite. On factorise par n². Donc la limite est 1/2.\nOn détermine la limite. On factorise. Ainsi la limite est 3."
            )
        
        # Upload PDF
        st.markdown("---")
        st.subheader("📄 Ou importer depuis PDF")
        pdf_file = st.file_uploader("Charger un sujet BAC (PDF)", type=["pdf"])
        
        if pdf_file:
            questions = extract_questions_from_pdf(pdf_file)
            if questions:
                st.success(f"✅ {len(questions)} questions extraites")
                for i, q in enumerate(questions[:5]):
                    st.text(f"{i+1}. {q['text'][:100]}...")
    
    # Tab 2: Pipeline
    with tab2:
        st.header("🚀 Exécution du Pipeline GTE")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            run_demo = st.button("▶️ Lancer Demo", use_container_width=True)
            run_custom = st.button("▶️ Lancer sur mes données", use_container_width=True)
        
        # Exécution Demo
        if run_demo:
            with st.spinner("Exécution du pipeline..."):
                # Données demo
                qis = [
                    Qi("Q1", "Calculer la limite de u_n = (n²+3n)/(2n²+1)", chapitre),
                    Qi("Q2", "Déterminer la limite de v_n = (3n²-n)/(n²+5)", chapitre),
                    Qi("Q3", "Calculer la limite de w_n = (2n²+4n-1)/(5n²-2n)", chapitre),
                    Qi("Q4", "Étudier la monotonie de a_n = n/(n+1)", chapitre),
                    Qi("Q5", "Montrer que b_n = 2^n/n! est décroissante", chapitre),
                    Qi("Q6", "Question singleton unique très spécifique", chapitre),
                ]
                for q in qis:
                    q.compute_hash()
                
                rqis = {
                    "Q1": Rqi("R1", "Q1", "On calcule la limite. On factorise par n². Donc la limite est 1/2."),
                    "Q2": Rqi("R2", "Q2", "On détermine la limite. On factorise. Ainsi la limite est 3."),
                    "Q3": Rqi("R3", "Q3", "Calculons la limite. Factorisons par n². Donc limite = 2/5."),
                    "Q4": Rqi("R4", "Q4", "On calcule a_{n+1} - a_n. C'est positif donc croissante."),
                    "Q5": Rqi("R5", "Q5", "On calcule le ratio. Pour n>=2, < 1 donc décroissante."),
                    "Q6": Rqi("R6", "Q6", "Réponse unique."),
                }
                for r in rqis.values():
                    r.compute_hash()
                
                pipeline = GTEPipeline()
                result = pipeline.run(qis, rqis)
                st.session_state['result'] = result
                st.session_state['pipeline'] = pipeline
        
        # Exécution Custom
        if run_custom and qi_input and rqi_input:
            with st.spinner("Exécution du pipeline..."):
                qi_lines = [l.strip() for l in qi_input.strip().split('\n') if l.strip()]
                rqi_lines = [l.strip() for l in rqi_input.strip().split('\n') if l.strip()]
                
                qis = [Qi(f"Q{i+1}", text, chapitre) for i, text in enumerate(qi_lines)]
                for q in qis:
                    q.compute_hash()
                
                rqis = {}
                for i, text in enumerate(rqi_lines):
                    if i < len(qis):
                        rqi = Rqi(f"R{i+1}", f"Q{i+1}", text)
                        rqi.compute_hash()
                        rqis[f"Q{i+1}"] = rqi
                
                pipeline = GTEPipeline()
                result = pipeline.run(qis, rqis)
                st.session_state['result'] = result
                st.session_state['pipeline'] = pipeline
        
        # Afficher les logs
        if 'result' in st.session_state:
            st.markdown("### 📋 Logs du Pipeline")
            for log in st.session_state['result'].get('logs', []):
                st.text(log)
    
    # Tab 3: Résultats
    with tab3:
        st.header("📊 Résultats")
        
        if 'result' not in st.session_state:
            st.info("Lancez le pipeline pour voir les résultats")
        else:
            result = st.session_state['result']
            
            # Status
            status = result.get('status', 'UNKNOWN')
            if status == "SEALED":
                st.success(f"✅ **{status}** — Chapitre scellé avec succès!")
            elif status == "NOT_SEALABLE":
                st.warning(f"⚠️ **{status}** — Orphelins détectés")
            else:
                st.error(f"❌ **{status}** — {result.get('message', '')}")
            
            # Métriques
            metrics = result.get('metrics', {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Qi Total", metrics.get('qi_total', 0))
            col2.metric("Clusters", metrics.get('clusters', 0))
            col3.metric("QC Sélectionnées", metrics.get('selected', 0))
            col4.metric("Coverage", f"{metrics.get('coverage_ratio', 0):.1%}")
            
            # QC Sélectionnées
            st.markdown("### 🎯 Questions Clés Sélectionnées")
            selected_qcs = result.get('selected_qcs', [])
            if selected_qcs:
                df = pd.DataFrame(selected_qcs)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucune QC sélectionnée")
            
            # Orphelins
            if metrics.get('orphans', 0) > 0:
                st.markdown("### ⚠️ Orphelins")
                st.warning(f"{metrics.get('orphans')} question(s) non couvertes")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    main()
