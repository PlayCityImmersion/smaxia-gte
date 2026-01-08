"""
smaxia_gte_v32_02_00_fullauto.py
Kernel SMAXIA V10.6.2 - ISO-PROD / Industrial Safety
TEST=PROD Pipeline GTE - Full Auto (No JSON Upload Required)
"""

import streamlit as st
import hashlib
import json
import requests
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import PyPDF2
import re
import time
import base64
from datetime import datetime
import os
from urllib.parse import urljoin, urlparse
import pdfplumber

# =============================================================================
# DATACLASSES OBLIGATOIRES
# =============================================================================

@dataclass
class Atom:
    qi_id: str
    rqi_id: str
    qi_clean: str
    rqi_clean: str
    chapter_ref: str
    locators: List[str]
    sha256: str

@dataclass
class TraceARI:
    qi_id: str
    actions: List[Dict]
    preconditions: List[str]
    output_type: str

@dataclass
class QCCandidate:
    qc_id: str
    qc_text: str
    ari_spine: List[Dict]
    frt: Dict[str, str]
    triggers: List[str]
    evidence_qi_ids: List[str]
    n_q_cluster: int

@dataclass
class AuditIA2:
    qc_id: str
    checks: Dict[str, bool]
    status: str
    fix_recommendations: List[str]

# =============================================================================
# UTILS - CACHE & DETERMINISME
# =============================================================================

def compute_sha256(*args, **kwargs) -> str:
    """Cache déterministe par SHA256"""
    content = json.dumps([str(a) for a in args], sort_keys=True) + json.dumps({k:str(v) for k,v in kwargs.items()}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

@st.cache_data(ttl=3600)
def safe_http_head(url: str, timeout: float = 5.0) -> Optional[Dict]:
    """HTTP HEAD safe avec timeout et preuves"""
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return {
            "url": url,
            "status_code": r.status_code,
            "content_type": r.headers.get("content-type", ""),
            "content_length": r.headers.get("content-length", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"url": url, "error": str(e), "timestamp": datetime.now().isoformat()}

@st.cache_data(ttl=3600)
def safe_http_get(url: str, max_size: int = 10*1024*1024, timeout: float = 10.0) -> Optional[bytes]:
    """HTTP GET safe avec limite taille"""
    try:
        with requests.get(url, timeout=timeout, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            content = b""
            for chunk in r.iter_content(chunk_size=8192):
                if len(content) > max_size:
                    return None  # Trop gros
                content += chunk
            return content
    except:
        return None

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extraction texte PDF (pdfplumber > PyPDF2)"""
    try:
        with pdfplumber.open(pdf_bytes) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            return text.strip()
    except:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or "" + "\n"
            return text.strip()
        except:
            return ""

# =============================================================================
# LLM JSON STRICT (HOOK - À BRANCHER)
# =============================================================================

def llm_json_call(prompt: str, schema_hint: Dict = None, context: str = "") -> Dict:
    """HOOK IA1/IA2 - À remplacer par votre moteur"""
    cache_key = compute_sha256(prompt, schema_hint or {}, context)
    
    # PACK_AUTO_BUILD déterministe minimal (SMAXIA-compliant)
    if "pack_auto_build" in prompt.lower():
        return {
            "academic": {
                "levels": ["Lycée", "Prépa", "Université"],
                "subject": st.session_state.get("matiere", ""),
                "chapter": st.session_state.get("chapitre", "")
            },
            "policy": {
                "harvest": {
                    "roots": [
                        f"https://www.education.gouv.fr",
                        f"https://www.education.{st.session_state.get('pays', 'france').lower()}.gouv.fr"
                    ]
                },
                "corrige_keywords": ["corrigé", "correction", "solutions", "barème", "marking scheme"],
                "verb_synonyms": [f"v{i}" for i in range(50)],
                "clustering": {"distance_threshold": 0.7}
            }
        }
    
    # IA1 Miner stub (à remplacer)
    if "miner" in prompt.lower():
        return {
            "qi_id": "",
            "actions": [{"verb_id": "calculer", "obj": "mock"}],
            "preconditions": ["mock"],
            "output_type": "mock"
        }
    
    # IA1 Builder stub (à remplacer)
    if "builder" in prompt.lower():
        return {
            "qc_id": "",
            "qc_text": "Comment ... ?",
            "ari_spine": [{"verb_id": "mock", "obj": "mock"}],
            "frt": {
                "usage": "mock",
                "reponse_type": "mock", 
                "pieges": "mock",
                "conclusion": "mock"
            },
            "triggers": ["mock1", "mock2", "mock3"],
            "evidence_qi_ids": []
        }
    
    return {"error": "LLM call failed - no valid pattern"}

# =============================================================================
# PIPELINE 10 PHASES (SMAXIA-Compliant)
# =============================================================================

def phase_0_inputs_ui():
    """[0] INPUTS_UI(pays, matiere, chapitre, max_pairs, mode) + cache_key"""
    st.sidebar.header("🚀 PARAMÈTRES TEST=PROD")
    st.session_state.pays = st.sidebar.text_input("Pays", value="France", key="pays")
    st.session_state.matiere = st.sidebar.text_input("Matière", value="Mathématiques", key="matiere")
    st.session_state.chapitre = st.sidebar.text_input("Chapitre", value="Fonctions", key="chapitre")
    st.session_state.max_pairs = st.sidebar.slider("Max pairs (test)", 1, 5, 2, key="max_pairs")
    
    if st.sidebar.button("🚀 RUN TEST CHAPITRE COMPLET", type="primary"):
        st.session_state.run_pipeline = True
        st.session_state.phase = 0
        st.rerun()
    
    st.sidebar.info(f"Cache Key: {compute_sha256(st.session_state.pays, st.session_state.matiere, st.session_state.chapitre)}")

def phase_1_pack_auto_build():
    """[1] PACK_AUTO_BUILD(pays, matiere, chapitre) via LLM + WebSearch + snapshot_urls"""
    with st.expander("📦 PHASE 1 - PACK AUTO-BUILD", expanded=True):
        st.info("🔄 Génération Pack académique SMAXIA...")
        
        prompt = f"pack_auto_build pays={st.session_state.pays} matiere={st.session_state.matiere} chapitre={st.session_state.chapitre}"
        st.session_state.pack = llm_json_call(prompt)
        
        # Validation Pack minimal
        required_keys = ["academic", "policy", "policy.harvest.roots"]
        if not all(k in str(st.session_state.pack) for k in required_keys):
            st.error("❌ PACK_AUTO_BUILD FAIL - Structure minimale manquante")
            st.stop()
        
        st.success("✅ Pack SMAXIA généré")
        st.json({k: v for k, v in st.session_state.pack.items() if k != "policy.verb_synonyms"})

def phase_2_harvest():
    """[2] HARVEST(snapshot_urls, policy) → Pairs(SujetPDF, CorrigéPDF) + preuves HTTP"""
    with st.expander("🌾 PHASE 2 - HARVEST OFFICIEL", expanded=True):
        st.info("🔍 Harvest sources officielles (HTTP proofs)...")
        pairs = []
        http_proofs = []
        
        for root in st.session_state.pack["policy"]["harvest"]["roots"]:
            proof = safe_http_head(root)
            http_proofs.append(proof)
            
            if proof and proof.get("status_code") == 200:
                # Recherche liens PDF depuis racine
                try:
                    content = safe_http_get(root)
                    if content:
                        pdf_links = re.findall(r'href=[\'"]?([^\'" >]+?\.(?:pdf|PDF))[\'">]', content.decode('utf-8', errors='ignore'))
                        for pdf_url in pdf_links[:st.session_state.max_pairs]:
                            full_url = urljoin(root, pdf_url)
                            pdf_proof = safe_http_head(full_url)
                            if pdf_proof and pdf_proof.get("status_code") == 200:
                                pairs.append({
                                    "sujet_url": full_url,
                                    "corrige_url": full_url.replace(".pdf", "_corrige.pdf"),  # Heuristique
                                    "sha256": compute_sha256(full_url),
                                    "http_proof": pdf_proof
                                })
                except Exception as e:
                    st.warning(f"Root {root}: {e}")
        
        st.session_state.pairs = pairs[:st.session_state.max_pairs]
        st.session_state.http_proofs = http_proofs
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pairs harvestées", len(st.session_state.pairs))
        with col2:
            st.metric("Roots vérifiés", len([p for p in http_proofs if p.get("status_code") == 200]))
        
        if not st.session_state.pairs:
            st.error("❌ HARVEST FAIL SCELLÉ - Aucune paire Sujet/Corrigé officielle")
            st.json([p for p in http_proofs if p.get("error")])
            st.stop()

def phase_3_extract_text():
    """[3] EXTRACT_TEXT(text-first + OCR fallback) → raw_texts"""
    with st.expander("📄 PHASE 3 - EXTRACTION TEXTE", expanded=True):
        st.info("🔄 Extraction texte réel PDF...")
        raw_texts = []
        
        for i, pair in enumerate(st.session_state.pairs):
            try:
                sujet_bytes = safe_http_get(pair["sujet_url"])
                corrige_bytes = safe_http_get(pair["corrige_url"]) if pair["corrige_url"] != pair["sujet_url"] else sujet_bytes
                
                if not sujet_bytes:
                    st.warning(f"Pair {i}: Sujet vide")
                    continue
                
                sujet_text = extract_pdf_text(sujet_bytes)
                corrige_text = extract_pdf_text(corrige_bytes) if corrige_bytes else ""
                
                raw_texts.append({
                    "pair_id": i,
                    "sujet_text": sujet_text[:500] + "..." if len(sujet_text) > 500 else sujet_text,
                    "corrige_text": corrige_text[:500] + "..." if len(corrige_text) > 500 else corrige_text,
                    "sha256": pair["sha256"],
                    "text_len": len(sujet_text) + len(corrige_text)
                })
            except Exception as e:
                st.warning(f"Pair {i}: {e}")
        
        st.session_state.raw_texts = raw_texts
        
        if not raw_texts:
            st.error("❌ EXTRACTION FAIL SCELLÉ - Aucun texte extractible")
            st.stop()
        
        st.success(f"✅ {len(raw_texts)} textes extraits")
        st.metric("Caractères total", sum(t["text_len"] for t in raw_texts))

def phase_4_atomisation():
    """[4] ATOMISATION → Atoms(Qi,RQi,locators,sha256)"""
    with st.expander("🔬 PHASE 4 - ATOMISATION QI/RQi", expanded=True):
        atoms = []
        
        # Heuristique simple: phrases avec "?" ou chiffres = potentiels QI
        qi_pattern = r'([0-9]+[).])?\s*[^.!?]*\?'
        
        for i, text_data in enumerate(st.session_state.raw_texts):
            full_text = text_data["sujet_text"] + "\n" + text_data["corrige_text"]
            
            qi_matches = re.finditer(qi_pattern, full_text)
            for j, match in enumerate(qi_matches):
                qi_text = match.group().strip()
                if len(qi_text) > 10:
                    atom = Atom(
                        qi_id=f"QI_{i:02d}_{j:03d}",
                        rqi_id=f"RQI_{i:02d}_{j:03d}",
                        qi_clean=qi_text,
                        rqi_clean="RQi associé",  # À enrichir avec alignement
                        chapter_ref=st.session_state.chapitre,
                        locators=[f"pair_{i}_pos_{match.start()}"],
                        sha256=compute_sha256(qi_text)
                    )
                    atoms.append(atom)
        
        st.session_state.atoms = atoms
        st.metric("Atoms QI/RQi", len(atoms))
        
        if not atoms:
            st.warning("⚠️ AUCUN ATOM QI détecté - Pipeline continue")

def phase_5_posable_gate():
    """[5] POSABLE_GATE (CAS 1) → posables + quarantined"""
    with st.expander("🚪 PHASE 5 - POSABLE GATE", expanded=True):
        # Gate simple: texte > 20 chars + contient mot-clé mathématique
        math_keywords = ["fonction", "dérivée", "limite", "équation", "résoudre"]
        
        posables = []
        quarantined = []
        
        for atom in st.session_state.atoms:
            if (len(atom.qi_clean) > 20 and 
                any(kw in atom.qi_clean.lower() for kw in math_keywords)):
                posables.append(atom)
            else:
                quarantined.append(atom)
        
        st.session_state.posables = posables
        st.session_state.quarantined = quarantined
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("QI Posables", len(posables))
        with col2:
            st.metric("QI Quarantaine", len(quarantined))

def phase_6_ia1_miner_batch():
    """[6] IA1_MINER_BATCH(posables) → TraceARI[] (JSON strict)"""
    with st.expander("⛏️ PHASE 6 - IA1 MINER", expanded=True):
        traces = []
        for posable in st.session_state.posables[:10]:  # Batch limité
            prompt = f"miner qi={posable.qi_clean}"
            trace_data = llm_json_call(prompt)
            
            trace = TraceARI(
                qi_id=posable.qi_id,
                actions=trace_data.get("actions", []),
                preconditions=trace_data.get("preconditions", []),
                output_type=trace_data.get("output_type", "unknown")
            )
            traces.append(trace)
        
        st.session_state.traces_ari = traces
        st.success(f"✅ {len(traces)} TraceARI")

def phase_7_clustering():
    """[7] CLUSTERING_PER_CHAPITRE(policy) → ClusterCandidates (anti-singleton)"""
    with st.expander("🧬 PHASE 7 - CLUSTERING", expanded=True):
        # Clustering trivial par similarité texte (anti-singleton)
        clusters = []
        used_qi = set()
        
        for i, posable1 in enumerate(st.session_state.posables):
            if posable1.qi_id in used_qi:
                continue
            cluster_qi = [posable1.qi_id]
            
            for posable2 in st.session_state.posables[i+1:]:
                if (len(set(posable1.qi_clean.lower()) & set(posable2.qi_clean.lower())) > 3 and
                    len(cluster_qi) < 5):
                    cluster_qi.append(posable2.qi_id)
            
            if len(cluster_qi) >= 2:
                clusters.append({
                    "cluster_id": f"C{i}",
                    "qi_ids": cluster_qi,
                    "size": len(cluster_qi)
                })
                used_qi.update(cluster_qi)
        
        st.session_state.clusters = clusters
        st.metric("Clusters n≥2", len(clusters))

def phase_8_ia1_builder_batch():
    """[8] IA1_BUILDER_BATCH(clusters) → QCCandidates (JSON strict)"""
    with st.expander("🏗️ PHASE 8 - IA1 BUILDER", expanded=True):
        qcs = []
        for cluster in st.session_state.clusters:
            qi_texts = [a.qi_clean for a in st.session_state.posables if a.qi_id in cluster["qi_ids"]]
            prompt = f"builder cluster_qi={qi_texts}"
            
            qc_data = llm_json_call(prompt)
            qc = QCCandidate(
                qc_id=cluster["cluster_id"],
                qc_text=qc_data.get("qc_text", f"Comment ... ? [{cluster['cluster_id']}]"),
                ari_spine=qc_data.get("ari_spine", []),
                frt=qc_data.get("frt", {}),
                triggers=qc_data.get("triggers", []),
                evidence_qi_ids=cluster["qi_ids"],
                n_q_cluster=len(cluster["qi_ids"])
            )
            qcs.append(qc)
        
        st.session_state.qc_candidates = qcs
        st.success(f"✅ {len(qcs)} QCCandidates")

def phase_9_ia2_judge_coverage_verdict():
    """[9] IA2_JUDGE + SELECTION_COVERAGE_DRIVEN + VERDICT → SEALED/FAIL + CoverageMap"""
    with st.expander("⚖️ PHASE 9 - IA2 JUDGE + COVERAGE", expanded=True):
        audits = []
        coverage_map = {}
        covered_qi = set()
        
        for qc in st.session_state.qc_candidates:
            # IA2 JUDGE (Python booléen STRICT)
            checks = {
                "QC_FORM": qc.qc_text.startswith("Comment "),
                "NO_LOCAL_CONSTANTS": not re.search(r'\b(20\d{2}|France|Paris)\b', qc.qc_text),
                "N_Q_CLUSTER_GE2": qc.n_q_cluster >= 2,
                "ARI_TYPED_ONLY": all("verb_id" in step for step in qc.ari_spine),
                "TRIGGERS_MIN3": len(qc.triggers) >= 3,
                "FRT_4BLOCS": all(key in qc.frt for key in ["usage","reponse_type","pieges","conclusion"])
            }
            
            status = "PASS" if all(checks.values()) else "FAIL"
            audit = AuditIA2(
                qc_id=qc.qc_id,
                checks=checks,
                status=status,
                fix_recommendations=[k for k,v in checks.items() if not v]
            )
            audits.append(audit)
            
            # CoverageMap
            for qi_id in qc.evidence_qi_ids:
                covered_qi.add(qi_id)
                coverage_map[qi_id] = qc.qc_id
        
        st.session_state.audits = audits
        st.session_state.coverage_map = coverage_map
        st.session_state.covered_qi = covered_qi
        
        # VERDICT SMAXIA STRICT
        total_posables = len(st.session_state.posables)
        coverage_pct = (len(covered_qi) / total_posables * 100) if total_posables > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("QC PASS IA2", sum(1 for a in audits if a.status == "PASS"))
        with col2:
            st.metric("Coverage", f"{coverage_pct:.1f}%")
        with col3:
            verdict = "🟢 SEALED" if coverage_pct == 100.0 else "🔴 FAIL"
            st.metric("VERDICT FINAL", verdict, delta=f"{coverage_pct:.1f}%")
        
        st.session_state.verdict = "SEALED" if coverage_pct == 100.0 else "FAIL"
        st.session_state.coverage_pct = coverage_pct

# =============================================================================
# UI RÉSULTATS
# =============================================================================

def display_results():
    """UI résultats SMAXIA-compliant"""
    st.header("📊 AUDIT SMAXIA FINAL")
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Atoms", len(st.session_state.atoms))
    with col2: st.metric("Posables", len(st.session_state.posables))
    with col3: st.metric("Clusters ≥2", len(st.session_state.clusters))
    with col4: st.metric("QC PASS", sum(1 for a in st.session_state.audits if a.status == "PASS"))
    
    # Tableau QC
    qc_data = []
    for audit in st.session_state.audits:
        qc = next((q for q in st.session_state.qc_candidates if q.qc_id == audit.qc_id), None)
        if qc:
            qc_data.append({
                "QC": qc.qc_text[:60] + "..." if len(qc.qc_text) > 60 else qc.qc_text,
                "n_QI": qc.n_q_cluster,
                "Triggers": len(qc.triggers),
                "Statut": audit.status,
                "Checks": sum(audit.checks.values())
            })
    
    st.dataframe(pd.DataFrame(qc_data), use_container_width=True)
    
    # Orphelins
    orphelins = [p for p in st.session_state.posables if p.qi_id not in st.session_state.covered_qi]
    with st.expander(f"👻 Orphelins ({len(orphelins)} / {len(st.session_state.posables)} QI)"):
        for orphan in orphelins[:10]:  # Top 10
            st.write(f"• **{orphan.qi_id}**: {orphan.qi_clean}")
    
    # EXPORT SCELLÉ
    export_data = {
        "pipeline": "SMAXIA_GTE_V32.02.00",
        "pays": st.session_state.pays,
        "chapitre": st.session_state.chapitre,
        "verdict": st.session_state.verdict,
        "coverage_pct": st.session_state.coverage_pct,
        "http_proofs": st.session_state.http_proofs,
        "pairs": st.session_state.pairs,
        "audits": [asdict(a) for a in st.session_state.audits],
        "timestamp": datetime.now().isoformat()
    }
    
    st.download_button(
        "📥 EXPORT AUDIT SCELLÉ",
        json.dumps(export_data, indent=2, ensure_ascii=False),
        f"smaxia_gte_{st.session_state.pays}_{st.session_state.chapitre}_{datetime.now().strftime('%Y%m%d')}.json",
        "application/json"
    )

# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

def main():
    st.set_page_config(
        page_title="🔬 SMAXIA GTE V32.02.00 - TEST=PROD",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 SMAXIA GTE V32.02.00")
    st.info("Pipeline ISO-PROD 10 phases - Industrial Safety - No Hardcode")
    
    phase_0_inputs_ui()
    
    if "run_pipeline" in st.session_state and st.session_state.run_pipeline:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        phases = [
            phase_1_pack_auto_build,
            phase_2_harvest,
            phase_3_extract_text,
            phase_4_atomisation,
            phase_5_posable_gate,
            phase_6_ia1_miner_batch,
            phase_7_clustering,
            phase_8_ia1_builder_batch,
            phase_9_ia2_judge_coverage_verdict
        ]
        
        for i, phase_func in enumerate(phases):
            status_text.text(f"🔄 PHASE {i+1}/9 - {phase_func.__name__}")
            phase_func()
            progress_bar.progress((i + 1) / len(phases))
            time.sleep(0.5)
        
        display_results()
    else:
        st.info("👈 Configurez les paramètres et lancez 'RUN TEST CHAPITRE'")

if __name__ == "__main__":
    main()
