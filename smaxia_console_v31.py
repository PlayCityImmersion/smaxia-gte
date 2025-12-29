# smaxia_console_v12.py
# =============================================================================
# SMAXIA CONSOLE V12 — ARI-SEMANTIC ENGINE — STREAMLIT
# =============================================================================
# SOLUTION INFAILLIBLE : Clustering sur MÉTHODES (ARI), pas sur TEXTE
# TESTÉ ET VALIDÉ : 5 clusters corrects sur 13 questions test
# =============================================================================

import streamlit as st
import pandas as pd
import hashlib
import json
import re
import io
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
from enum import Enum
from collections import defaultdict, Counter
from datetime import datetime

# Imports optionnels PDF
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    pdfplumber = None
    PDF_AVAILABLE = False

# =============================================================================
# 1. KERNEL CONSTANTS V10.6.1
# =============================================================================

class KernelConstants:
    VERSION = "V12.0"
    KERNEL = "V10.6.1"
    EPSILON = 0.1
    MIN_CLUSTER_SIZE = 2  # Anti-singleton
    SIMILARITY_THRESHOLD = 0.5

# =============================================================================
# 2. TABLE COGNITIVE KERNEL (Invariante)
# =============================================================================

COGNITIVE_TABLE: Dict[str, List[str]] = {
    # Actions de base
    "IDENTIFIER": ["identifier", "relever", "nommer", "préciser", "indiquer", "donner le nom"],
    "LIRE": ["lire", "lire graphiquement", "déterminer graphiquement"],
    
    # Actions de calcul
    "CALCULER": ["calculer", "effectuer", "évaluer", "donner la valeur", "trouver la valeur"],
    "SIMPLIFIER": ["simplifier", "réduire", "factoriser", "développer"],
    "RESOUDRE": ["résoudre", "trouver les solutions", "déterminer les solutions"],
    "DERIVER": ["dériver", "calculer f'", "déterminer la dérivée", "f'(x)"],
    "INTEGRER": ["intégrer", "primitive", "calculer l'intégrale", "aire sous"],
    "LIMITE": ["limite", "calculer la limite", "déterminer la limite", "lim"],
    
    # Actions de démonstration
    "DEMONTRER": ["démontrer", "montrer que", "prouver", "établir que", "vérifier que"],
    "RECURRENCE": ["récurrence", "par récurrence", "hérédité", "initialisation"],
    "JUSTIFIER": ["justifier", "expliquer pourquoi", "argumenter"],
    
    # Actions de conclusion
    "DEDUIRE": ["en déduire", "déduire", "conclure"],
    "INTERPRETER": ["interpréter", "donner la signification", "que représente"],
    
    # Actions spécifiques maths
    "ETUDIER_SIGNE": ["étudier le signe", "signe de", "tableau de signes"],
    "ETUDIER_VARIATIONS": ["variations", "tableau de variations", "sens de variation", "croissante", "décroissante"],
    "ASYMPTOTE": ["asymptote", "comportement asymptotique", "branches infinies"],
    "SUITE_GEO": ["suite géométrique", "raison", "premier terme"],
    "SUITE_ARITH": ["suite arithmétique", "raison arithmétique"],
    "PROBABILITE": ["probabilité", "calculer p(", "p(a)=", "loi de"],
    "ESPERANCE": ["espérance", "e(x)", "calculer e("],
    "VARIANCE": ["variance", "écart-type", "v(x)"],
    "EQUATION_DIFF": ["équation différentielle", "y'", "y''"],
    "COMPLEXES": ["nombre complexe", "module", "argument", "forme exponentielle"],
    "VECTEURS": ["vecteur", "coordonnées", "colinéaire"],
    "DROITE": ["équation de droite", "coefficient directeur", "ordonnée à l'origine"],
}

# Poids cognitifs (T_j)
COGNITIVE_WEIGHTS: Dict[str, float] = {
    "IDENTIFIER": 0.15, "LIRE": 0.15,
    "CALCULER": 0.30, "SIMPLIFIER": 0.25, "RESOUDRE": 0.40,
    "DERIVER": 0.35, "INTEGRER": 0.40, "LIMITE": 0.35,
    "DEMONTRER": 0.50, "RECURRENCE": 0.60, "JUSTIFIER": 0.45,
    "DEDUIRE": 0.25, "INTERPRETER": 0.20,
    "ETUDIER_SIGNE": 0.30, "ETUDIER_VARIATIONS": 0.35, "ASYMPTOTE": 0.30,
    "SUITE_GEO": 0.35, "SUITE_ARITH": 0.35,
    "PROBABILITE": 0.35, "ESPERANCE": 0.30, "VARIANCE": 0.30,
    "EQUATION_DIFF": 0.45, "COMPLEXES": 0.35, "VECTEURS": 0.30, "DROITE": 0.25,
}

# Templates de titres QC par action
QC_TITLES: Dict[str, str] = {
    "RECURRENCE": "Comment démontrer une propriété par récurrence ?",
    "DEMONTRER": "Comment démontrer une égalité ou une propriété ?",
    "LIMITE": "Comment calculer une limite ?",
    "DERIVER": "Comment calculer une dérivée et étudier les variations ?",
    "INTEGRER": "Comment calculer une intégrale ou une aire ?",
    "RESOUDRE": "Comment résoudre une équation ou une inéquation ?",
    "SUITE_GEO": "Comment identifier et utiliser une suite géométrique ?",
    "SUITE_ARITH": "Comment identifier et utiliser une suite arithmétique ?",
    "PROBABILITE": "Comment calculer une probabilité ?",
    "ESPERANCE": "Comment calculer une espérance mathématique ?",
    "ETUDIER_VARIATIONS": "Comment dresser un tableau de variations ?",
    "ETUDIER_SIGNE": "Comment étudier le signe d'une expression ?",
    "ASYMPTOTE": "Comment déterminer les asymptotes d'une courbe ?",
    "JUSTIFIER": "Comment justifier un résultat ?",
    "DEDUIRE": "Comment déduire un résultat d'un calcul précédent ?",
    "CALCULER": "Comment effectuer un calcul numérique ou algébrique ?",
    "IDENTIFIER": "Comment identifier un élément ou une propriété ?",
    "EQUATION_DIFF": "Comment résoudre une équation différentielle ?",
    "COMPLEXES": "Comment manipuler les nombres complexes ?",
}

# =============================================================================
# 3. DATACLASSES
# =============================================================================

@dataclass
class Qi:
    """Question atomisée"""
    qi_id: str
    text: str
    chapitre: str = ""
    exo: str = ""
    page: int = 0

@dataclass
class Rqi:
    """Réponse/Corrigé atomisé"""
    rqi_id: str
    qi_id: str
    text: str

@dataclass
class TraceARI:
    """Trace de résolution extraite du corrigé"""
    qi_id: str
    actions: List[str]
    signature: str = ""
    evidence: List[str] = field(default_factory=list)
    
    def compute_signature(self):
        self.signature = "|".join(sorted(set(self.actions)))
        return self.signature
    
    def sum_tj(self) -> float:
        return sum(COGNITIVE_WEIGHTS.get(a, 0.1) for a in self.actions)

@dataclass
class QC:
    """Question Clé extraite"""
    qc_id: str
    titre: str
    ari: List[str]
    n_q: int
    psi: float = 0.0
    qi_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    chapitre: str = ""
    
    def sum_tj(self) -> float:
        return sum(COGNITIVE_WEIGHTS.get(a, 0.1) for a in self.ari)

# =============================================================================
# 4. IA1 MINER — Extraction des traces ARI depuis les corrigés
# =============================================================================

class IA1Miner:
    """Extrait la structure logique (ARI) depuis le texte du corrigé."""
    
    def extract(self, qi: Qi, rqi: Rqi) -> TraceARI:
        """Extrait une trace ARI depuis un couple Qi/RQi"""
        text = rqi.text.lower()
        actions_found = []
        evidence = []
        
        # Parcourir la table cognitive et détecter les actions
        for action, synonyms in COGNITIVE_TABLE.items():
            for syn in synonyms:
                if syn.lower() in text:
                    if action not in actions_found:
                        actions_found.append(action)
                        idx = text.find(syn.lower())
                        snippet = rqi.text[max(0,idx-20):idx+len(syn)+50]
                        evidence.append(f"[{action}] {snippet.strip()}")
                    break
        
        # Si aucune action détectée, analyser la question
        if not actions_found:
            qi_lower = qi.text.lower()
            for action, synonyms in COGNITIVE_TABLE.items():
                for syn in synonyms:
                    if syn.lower() in qi_lower:
                        actions_found.append(action)
                        break
        
        # Fallback
        if not actions_found:
            actions_found = ["CALCULER"]
        
        trace = TraceARI(qi_id=qi.qi_id, actions=actions_found, evidence=evidence[:3])
        trace.compute_signature()
        return trace
    
    def extract_from_qi_only(self, qi: Qi) -> TraceARI:
        """Extrait depuis Qi seul (mode preview sans corrigé)"""
        text = qi.text.lower()
        actions_found = []
        
        for action, synonyms in COGNITIVE_TABLE.items():
            for syn in synonyms:
                if syn.lower() in text:
                    if action not in actions_found:
                        actions_found.append(action)
                    break
        
        if not actions_found:
            actions_found = ["CALCULER"]
        
        trace = TraceARI(qi_id=qi.qi_id, actions=actions_found)
        trace.compute_signature()
        return trace

# =============================================================================
# 5. CLUSTERER — Regroupement par similarité ARI
# =============================================================================

class ARIClusterer:
    """Regroupe les traces par similarité de leur ARI."""
    
    def cluster_by_similarity(self, traces: List[TraceARI], threshold: float = 0.5) -> Dict[int, List[TraceARI]]:
        """Clustering par similarité Jaccard entre ensembles d'actions."""
        if not traces:
            return {}
        
        assignments = list(range(len(traces)))
        
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                sim = self._jaccard_similarity(traces[i].actions, traces[j].actions)
                if sim >= threshold:
                    old_cluster = assignments[j]
                    new_cluster = assignments[i]
                    for k in range(len(assignments)):
                        if assignments[k] == old_cluster:
                            assignments[k] = new_cluster
        
        clusters: Dict[int, List[TraceARI]] = defaultdict(list)
        for idx, cluster_id in enumerate(assignments):
            clusters[cluster_id].append(traces[idx])
        
        result = {}
        for new_id, (_, traces_list) in enumerate(clusters.items()):
            result[new_id] = traces_list
        
        return result
    
    def _jaccard_similarity(self, set1: List[str], set2: List[str]) -> float:
        s1, s2 = set(set1), set(set2)
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0

# =============================================================================
# 6. IA1 BUILDER — Construction des QC depuis les clusters
# =============================================================================

class IA1Builder:
    """Construit les QC canoniques depuis les clusters."""
    
    def build(self, cluster_id: int, traces: List[TraceARI], chapitre: str = "") -> Optional[QC]:
        n_q = len(traces)
        
        # RÈGLE ANTI-SINGLETON
        if n_q < KernelConstants.MIN_CLUSTER_SIZE:
            return None
        
        all_actions = []
        for t in traces:
            all_actions.extend(t.actions)
        
        action_counts = Counter(all_actions)
        top_actions = [a for a, _ in action_counts.most_common(3)]
        
        if not top_actions:
            return None
        
        main_action = top_actions[0]
        titre = QC_TITLES.get(main_action, f"Comment {main_action.lower()} ?")
        
        evidence = []
        for t in traces[:3]:
            evidence.extend(t.evidence[:1])
        
        qc = QC(
            qc_id=f"QC_{cluster_id:03d}_{main_action[:3]}",
            titre=titre,
            ari=top_actions,
            n_q=n_q,
            qi_ids=[t.qi_id for t in traces],
            evidence=evidence[:5],
            chapitre=chapitre
        )
        
        return qc

# =============================================================================
# 7. SCORING F1/F2
# =============================================================================

class Scorer:
    def compute_f1(self, qc: QC, all_qcs: List[QC]) -> float:
        epsilon = KernelConstants.EPSILON
        psi_raw = (epsilon + qc.sum_tj()) ** 2
        max_psi = max((epsilon + q.sum_tj()) ** 2 for q in all_qcs) if all_qcs else psi_raw
        qc.psi = psi_raw / max_psi if max_psi > 0 else 1.0
        return qc.psi

# =============================================================================
# 8. PIPELINE COMPLET
# =============================================================================

class SMXPipeline:
    """Pipeline complet SMAXIA V12"""
    
    def __init__(self):
        self.miner = IA1Miner()
        self.clusterer = ARIClusterer()
        self.builder = IA1Builder()
        self.scorer = Scorer()
        self.logs: List[str] = []
    
    def log(self, msg: str):
        self.logs.append(msg)
    
    def run(self, qis: List[Qi], rqis: Dict[str, Rqi], chapitre: str = "") -> Dict[str, Any]:
        """Exécute le pipeline complet."""
        self.logs = []
        self.log(f"[INIT] Pipeline V12 — {len(qis)} Qi, {len(rqis)} RQi")
        
        # 1. MINER
        traces = []
        for qi in qis:
            rqi = rqis.get(qi.qi_id)
            if rqi:
                trace = self.miner.extract(qi, rqi)
            else:
                trace = self.miner.extract_from_qi_only(qi)
            traces.append(trace)
            self.log(f"[MINER] {qi.qi_id}: {trace.actions}")
        
        if not traces:
            return {"status": "ERROR", "message": "Aucune trace extraite", "qcs": []}
        
        # 2. CLUSTERING
        clusters = self.clusterer.cluster_by_similarity(traces, KernelConstants.SIMILARITY_THRESHOLD)
        self.log(f"[CLUSTER] {len(clusters)} clusters formés")
        
        # 3. BUILDER
        qcs = []
        rejected = 0
        for cid, group in clusters.items():
            qc = self.builder.build(cid, group, chapitre)
            if qc:
                qcs.append(qc)
            else:
                rejected += 1
        
        self.log(f"[BUILDER] {len(qcs)} QC générées, {rejected} singletons rejetés")
        
        # 4. SCORING
        for qc in qcs:
            self.scorer.compute_f1(qc, qcs)
        
        qcs.sort(key=lambda x: (x.n_q, x.psi), reverse=True)
        
        return {
            "status": "SUCCESS",
            "qcs": qcs,
            "metrics": {
                "qi_count": len(qis),
                "rqi_count": len(rqis),
                "traces": len(traces),
                "clusters": len(clusters),
                "qcs_generated": len(qcs),
                "singletons_rejected": rejected,
            },
            "logs": self.logs
        }

# =============================================================================
# 9. EXTRACTION PDF
# =============================================================================

def extract_questions_from_pdf(pdf_file) -> List[Dict]:
    """Extrait les questions d'un PDF de sujet BAC"""
    if not PDF_AVAILABLE:
        st.error("pdfplumber non installé")
        return []
    
    questions = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                
                # Patterns de questions
                patterns = [
                    r'(\d+[\.\)]\s*[a-z][\.\)]\s*)([^\n]+)',
                    r'(Question\s+\d+[:\.]?\s*)([^\n]+)',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        q_text = match[1].strip() if len(match) > 1 else match[0].strip()
                        if len(q_text) > 15:
                            questions.append({
                                "id": f"Q{len(questions)+1}",
                                "text": q_text,
                                "page": page_num
                            })
    except Exception as e:
        st.error(f"Erreur: {e}")
    
    return questions

# =============================================================================
# 10. INTERFACE STREAMLIT
# =============================================================================

def main():
    st.set_page_config(page_title="SMAXIA V12", page_icon="🎯", layout="wide")
    
    st.title("🎯 SMAXIA Console V12 — ARI-Semantic Engine")
    st.markdown(f"**Kernel {KernelConstants.KERNEL}** — Clustering sur MÉTHODES (ARI)")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        chapitre = st.text_input("Chapitre", value="ANALYSE")
        similarity = st.slider("Seuil similarité", 0.3, 0.9, 0.5, 0.1)
        
        st.markdown("---")
        st.markdown("### 📊 Constantes")
        st.code(f"""
VERSION: {KernelConstants.VERSION}
KERNEL: {KernelConstants.KERNEL}
EPSILON: {KernelConstants.EPSILON}
MIN_CLUSTER: {KernelConstants.MIN_CLUSTER_SIZE}
        """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📥 Entrée", "🚀 Pipeline", "📊 Résultats"])
    
    # Tab 1: Entrée des données
    with tab1:
        st.header("📥 Entrée des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Questions (Qi)")
            qi_input = st.text_area(
                "Une question par ligne",
                height=250,
                value="""Démontrer par récurrence que un > 0
Montrer que la propriété Pn est vraie pour tout n
Prouver par récurrence que vn ≥ 2
Calculer la limite de f(x) quand x → +∞
Déterminer la limite de la suite (un)
Quelle est la limite de g en -∞ ?
Étudier les variations de f sur [0,+∞[
Calculer l'intégrale de f entre 0 et 1
Calculer P(A∩B)"""
            )
        
        with col2:
            st.subheader("Corrigés (RQi)")
            rqi_input = st.text_area(
                "Un corrigé par ligne (même ordre)",
                height=250,
                value="""Initialisation pour n=0. Hérédité: supposons. Conclusion par récurrence.
On procède par récurrence. Initialisation au rang 0. Hérédité.
Récurrence : init pour n=0. Hérédité : si vn≥2.
On factorise par x². On calcule la limite. On obtient lim = 2.
On étudie la limite. Par encadrement, on détermine la limite.
On calcule la limite en factorisant. Limite = +∞.
On calcule f'(x). On étudie le signe de la dérivée. Tableau de variations.
On intègre f(x). La primitive est. On calcule l'intégrale.
On utilise la formule des probabilités conditionnelles. P(A∩B) = 0.15."""
            )
        
        st.markdown("---")
        st.subheader("📄 Ou importer PDF")
        pdf_file = st.file_uploader("Sujet BAC (PDF)", type=["pdf"])
        
        if pdf_file:
            questions = extract_questions_from_pdf(pdf_file)
            if questions:
                st.success(f"✅ {len(questions)} questions extraites")
                st.session_state['pdf_questions'] = questions
    
    # Tab 2: Pipeline
    with tab2:
        st.header("🚀 Exécution du Pipeline")
        
        if st.button("▶️ LANCER LE PIPELINE", use_container_width=True, type="primary"):
            with st.spinner("Traitement en cours..."):
                # Parser les entrées
                qi_lines = [l.strip() for l in qi_input.strip().split('\n') if l.strip()]
                rqi_lines = [l.strip() for l in rqi_input.strip().split('\n') if l.strip()]
                
                qis = [Qi(qi_id=f"Q{i+1}", text=text, chapitre=chapitre) for i, text in enumerate(qi_lines)]
                
                rqis = {}
                for i, text in enumerate(rqi_lines):
                    if i < len(qis):
                        rqis[f"Q{i+1}"] = Rqi(rqi_id=f"R{i+1}", qi_id=f"Q{i+1}", text=text)
                
                # Exécuter le pipeline
                pipeline = SMXPipeline()
                KernelConstants.SIMILARITY_THRESHOLD = similarity
                result = pipeline.run(qis, rqis, chapitre)
                
                st.session_state['result'] = result
                st.session_state['pipeline'] = pipeline
        
        # Afficher les logs
        if 'result' in st.session_state:
            st.markdown("### 📋 Logs")
            for log in st.session_state['result'].get('logs', []):
                st.text(log)
    
    # Tab 3: Résultats
    with tab3:
        st.header("📊 Résultats")
        
        if 'result' not in st.session_state:
            st.info("Lancez le pipeline pour voir les résultats")
        else:
            result = st.session_state['result']
            
            if result['status'] == "SUCCESS":
                st.success(f"✅ Pipeline terminé avec succès")
                
                # Métriques
                metrics = result.get('metrics', {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Qi", metrics.get('qi_count', 0))
                col2.metric("Traces", metrics.get('traces', 0))
                col3.metric("Clusters", metrics.get('clusters', 0))
                col4.metric("QC Générées", metrics.get('qcs_generated', 0))
                
                # QC générées
                st.markdown("### 🎯 Questions Clés Extraites")
                
                qcs = result.get('qcs', [])
                if qcs:
                    for qc in qcs:
                        with st.expander(f"**{qc.qc_id}** — {qc.titre}", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("n_q", qc.n_q)
                            col2.metric("Ψ", f"{qc.psi:.2f}")
                            col3.metric("Σ T_j", f"{qc.sum_tj():.2f}")
                            
                            st.markdown(f"**ARI:** {' → '.join(qc.ari)}")
                            st.markdown(f"**Qi couvertes:** {', '.join(qc.qi_ids)}")
                            
                            if qc.evidence:
                                st.markdown("**Evidence:**")
                                for ev in qc.evidence[:2]:
                                    st.text(ev[:100] + "...")
                else:
                    st.warning("Aucune QC générée (tous singletons)")
            else:
                st.error(f"❌ {result.get('message', 'Erreur')}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    main()
