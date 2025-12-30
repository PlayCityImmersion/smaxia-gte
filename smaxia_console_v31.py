# =============================================================================
# SMAXIA GTE Console V31.9.10 (ISO-PROD — FLUX DÉFINITIF CEO + GEMINI)
# =============================================================================
#
# FLUX VALIDÉ:
# [1] Activation FR (1 clic) → Moteur: load_academic_pack("FR")
# [2] Pack chargé (invisible) → UI: Affiche Niveaux/Matières/Chapitres
# [3] Cocher Niveau + Matière
# [4] Harvest AUTO (pêche aveugle) → UI: Barre progression visible
# [5] Pipeline: Qi/RQi → (ARI, FRT, Triggers) → QC
# [6] Ranger QC par Chapitre via intent_code → Pack.chapter_code
# [7] Boucle saturation: 1 itération new_QC=0 → SEALED
# [8] Explorateur: QC → ARI → FRT → Triggers → Qi
# [9] Exports
#
# RÈGLES SCELLÉES:
# - Ordre: Qi/RQi → (ARI, FRT, Triggers ENSEMBLE) → QC (conteneur final)
# - Mapping chapitre: Option A (Pack-driven, zéro sémantique code)
# - Saturation: 1 itération new_QC=0 suffit (Axiome Saturation)
# - Anti-singleton: n_q_cluster ≥ 2
# =============================================================================

from __future__ import annotations
import hashlib, io, json, random, re, urllib.parse, urllib.request
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

try:
    import pdfplumber
except:
    pdfplumber = None

# =============================================================================
# ZONE A : COUNTRY ACADEMIC PACKS (CAP) — DONNÉES P3
# =============================================================================
# Simule l'API Prompt 5 : load_academic_pack("FR") retourne ce Pack
# En PROD, ceci viendra d'une API/DB externe

def load_academic_pack(country_code: str) -> Optional[Dict]:
    """Moteur invisible : charge le Pack académique pour un pays"""
    PACKS = {
        "FR": {
            "pack_id": "CAP_FR_BAC_2024_V1",
            "country_code": "FR",
            "country_label": "France",
            "version": "1.0.0",
            "signature": {"hash": "sha256:abc123def456", "signed_at": "2024-12-30"},
            
            "academic_structure": {
                "levels": [
                    {"id": "SECONDE", "label": "Seconde", "order": 1},
                    {"id": "PREMIERE", "label": "Première", "order": 2},
                    {"id": "TERMINALE", "label": "Terminale", "order": 3},
                    {"id": "L1", "label": "Licence 1", "order": 4},
                    {"id": "L2", "label": "Licence 2", "order": 5},
                    {"id": "L3", "label": "Licence 3", "order": 6},
                    {"id": "M1", "label": "Master 1", "order": 7},
                    {"id": "M2", "label": "Master 2", "order": 8},
                    {"id": "PREPA", "label": "Prépa (CPGE)", "order": 9},
                ],
                "subjects": [
                    {"id": "MATH", "label": "Mathématiques"},
                    {"id": "PHYS", "label": "Physique-Chimie"},
                    {"id": "SVT", "label": "Sciences de la Vie et de la Terre"},
                    {"id": "NSI", "label": "Numérique et Sciences Informatiques"},
                ],
                "level_subjects": {
                    "TERMINALE": ["MATH", "PHYS", "SVT", "NSI"],
                    "PREMIERE": ["MATH", "PHYS", "SVT", "NSI"],
                    "SECONDE": ["MATH", "PHYS"],
                    "L1": ["MATH", "PHYS"],
                    "PREPA": ["MATH", "PHYS"],
                },
            },
            
            # Chapitres avec intent_allowlist pour mapping QC → Chapitre
            "chapters": [
                {
                    "chapter_code": "CH_ANALYSE",
                    "label": "Analyse - Fonctions",
                    "qc_target": 15,
                    "intent_allowlist": ["STRUCT_EX1_Q1", "STRUCT_EX1_Q2", "STRUCT_EX1_Q3", 
                                        "STRUCT_EX1_Q4", "STRUCT_EX1_Q5", "STRUCT_EX2_Q1",
                                        "STRUCT_EX2_Q2", "STRUCT_EX2_Q3"],
                },
                {
                    "chapter_code": "CH_PROBAS",
                    "label": "Probabilités - Variables aléatoires",
                    "qc_target": 15,
                    "intent_allowlist": ["STRUCT_EX3_Q1", "STRUCT_EX3_Q2", "STRUCT_EX3_Q3",
                                        "STRUCT_EX3_Q4", "STRUCT_EX3_Q5"],
                },
                {
                    "chapter_code": "CH_GEOMETRIE",
                    "label": "Géométrie - Nombres complexes",
                    "qc_target": 15,
                    "intent_allowlist": ["STRUCT_EX4_Q1", "STRUCT_EX4_Q2", "STRUCT_EX4_Q3",
                                        "STRUCT_EX4_Q4", "STRUCT_EX4_Q5"],
                },
                {
                    "chapter_code": "CH_SUITES",
                    "label": "Suites et Récurrence",
                    "qc_target": 15,
                    "intent_allowlist": ["STRUCT_EX2_Q4", "STRUCT_EX2_Q5", "STRUCT_EX2_Q6"],
                },
            ],
            
            # Configuration Harvest (simule API P5)
            "harvest_config": {
                "http": {"user_agent": "SMAXIA-Harvester/1.0", "timeout_sec": 30, "max_pdf_mb": 25},
                "sources": {
                    "TERMINALE|MATH": [
                        {"name": "APMEP", "index_url": "https://www.apmep.fr/spip.php?rubrique387",
                         "pdf_regex": r"\.pdf($|\?)", "corr_regex": r"corrig[eé]|correction"},
                    ],
                    "TERMINALE|PHYS": [
                        {"name": "Labolycee", "index_url": "https://labolycee.org/",
                         "pdf_regex": r"\.pdf($|\?)", "corr_regex": r"corrig[eé]|correction"},
                    ],
                },
            },
        },
        
        "CI": {
            "pack_id": "CAP_CI_BAC_2024_V1",
            "country_code": "CI",
            "country_label": "Côte d'Ivoire",
            "version": "1.0.0",
            "signature": {"hash": "sha256:ci_hash", "signed_at": "2024-12-30"},
            "academic_structure": {
                "levels": [{"id": "TERMINALE", "label": "Terminale", "order": 1}],
                "subjects": [{"id": "MATH", "label": "Mathématiques"}],
                "level_subjects": {"TERMINALE": ["MATH"]},
            },
            "chapters": [{"chapter_code": "CH_ANALYSE", "label": "Analyse", "qc_target": 15, "intent_allowlist": []}],
            "harvest_config": {"http": {}, "sources": {}},
        },
    }
    return PACKS.get(country_code)

# =============================================================================
# ZONE B : CORE HELPERS (métier-neutre)
# =============================================================================

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",",":"), default=str)

def sha8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# =============================================================================
# ZONE C : HARVEST ENGINE — Pêche Aveugle (P5)
# =============================================================================

def http_get(url: str, ua: str = "Mozilla/5.0", timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

def extract_pdf_links(html: bytes, base_url: str, pdf_regex: str) -> List[str]:
    text = html.decode("utf-8", errors="ignore")
    hrefs = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', text, re.I)
    rx = re.compile(pdf_regex, re.I)
    pdfs = [urllib.parse.urljoin(base_url, h) for h in hrefs if rx.search(h)]
    return list(dict.fromkeys(pdfs))

def pair_subject_correction(pdfs: List[str], corr_regex: str) -> List[Tuple[str, Optional[str]]]:
    rx = re.compile(corr_regex, re.I) if corr_regex else None
    corrs = [p for p in pdfs if rx and rx.search(p)]
    subjs = [p for p in pdfs if not rx or not rx.search(p)]
    
    def normalize(u):
        base = urllib.parse.urlparse(u).path.split("/")[-1]
        base = re.sub(r"\.pdf.*$", "", base, flags=re.I)
        if rx: base = rx.sub("", base)
        return re.sub(r"[^a-z0-9]", "", base.lower())
    
    corr_map = {normalize(c): c for c in corrs}
    return [(s, corr_map.get(normalize(s))) for s in subjs]

def harvest_subjects(pack: Dict, level: str, subject: str, volume: int, 
                     progress_callback=None) -> List[Dict]:
    """
    HARVEST AUTO — Pêche aveugle sur les sources du Pack
    Retourne liste de paires (sujet_bytes, correction_bytes)
    """
    cfg = pack.get("harvest_config", {})
    http_cfg = cfg.get("http", {})
    ua = http_cfg.get("user_agent", "Mozilla/5.0")
    timeout = http_cfg.get("timeout_sec", 30)
    
    key = f"{level}|{subject}"
    sources = cfg.get("sources", {}).get(key, [])
    
    if not sources:
        return []
    
    all_pairs = []
    for src in sources:
        try:
            html = http_get(src["index_url"], ua, timeout)
            pdfs = extract_pdf_links(html, src["index_url"], src.get("pdf_regex", r"\.pdf"))
            pairs = pair_subject_correction(pdfs, src.get("corr_regex", ""))
            all_pairs.extend(pairs)
        except:
            pass
    
    # Dedupe
    seen = set()
    unique = [(s, c) for s, c in all_pairs if s not in seen and not seen.add(s)]
    
    # Limit to volume
    random.seed(42)
    if len(unique) > volume:
        unique = random.sample(unique, volume)
    
    # Download avec progression
    items = []
    total = len(unique)
    for i, (subj_url, corr_url) in enumerate(unique):
        if progress_callback:
            progress_callback(i + 1, total, subj_url)
        try:
            sb = http_get(subj_url, ua, timeout)
            cb = http_get(corr_url, ua, timeout) if corr_url else None
            items.append({
                "pair_id": f"PAIR_{i+1:04d}_{sha8(sb)}",
                "label": urllib.parse.urlparse(subj_url).path.split("/")[-1],
                "subject_bytes": sb,
                "correction_bytes": cb,
            })
        except:
            pass
    
    return items

# =============================================================================
# ZONE D : PDF → Qi/RQi EXTRACTION
# =============================================================================

def read_pdf_text(pdf_bytes: bytes) -> str:
    if not pdfplumber:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except:
        return ""

def segment_questions(text: str) -> List[Dict]:
    """Segmente le texte PDF en questions (Qi)"""
    segments = []
    current_ex = "0"
    current_q = None
    current_text = []
    
    for line in text.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        
        # Détecter exercice
        up = line_strip.upper().replace(" ", "")
        if up.startswith("EXERCICE"):
            m = re.search(r"(\d+)", up)
            if m:
                current_ex = m.group(1)
                current_q = None
            continue
        
        # Détecter question principale (1), 2), 3)...)
        m = re.match(r"^\s*(\d{1,2})\s*[\)\.\-]\s*", line_strip)
        if m:
            # Flush previous
            if current_q and current_text:
                txt = "\n".join(current_text).strip()
                if len(txt) >= 30:
                    segments.append({"marker": current_q, "text": txt, "ex": current_ex})
            current_q = f"EX{current_ex}-Q{m.group(1)}"
            current_text = [line]
            continue
        
        # Détecter sous-question a), b), c)...)
        m = re.match(r"^\s*([a-h])\s*[\)\.\-]\s*", line_strip, re.I)
        if m and current_q:
            # Flush previous
            if current_text:
                txt = "\n".join(current_text).strip()
                if len(txt) >= 30:
                    segments.append({"marker": current_q, "text": txt, "ex": current_ex})
            base_q = re.match(r"(EX\d+-Q\d+)", current_q)
            if base_q:
                current_q = f"{base_q.group(1)}{m.group(1).lower()}"
            current_text = [line]
            continue
        
        # Ajouter au bloc courant
        if current_q:
            current_text.append(line)
    
    # Flush final
    if current_q and current_text:
        txt = "\n".join(current_text).strip()
        if len(txt) >= 30:
            segments.append({"marker": current_q, "text": txt, "ex": current_ex})
    
    return segments

def extract_qi_rqi(pairs: List[Dict]) -> List[Dict]:
    """Extrait Qi/RQi depuis les paires (sujet, correction)"""
    qi_list = []
    
    for pair in pairs:
        subj_text = read_pdf_text(pair["subject_bytes"])
        corr_text = read_pdf_text(pair.get("correction_bytes") or b"")
        
        subj_segments = segment_questions(subj_text)
        corr_segments = segment_questions(corr_text)
        
        # Index corrections par marker
        corr_idx = {s["marker"]: s["text"] for s in corr_segments}
        
        for seg in subj_segments:
            marker = seg["marker"]
            qi_id = f"QI_{sha8((pair['pair_id'] + marker).encode())}"
            
            # Générer intent_code depuis marker (EXn-Qm → STRUCT_EXn_Qm)
            m = re.match(r"EX(\d+)-Q(\d+)", marker)
            intent_code = f"STRUCT_EX{m.group(1)}_Q{m.group(2)}" if m else "STRUCT_UNK"
            
            qi_list.append({
                "qi_id": qi_id,
                "pair_id": pair["pair_id"],
                "marker": marker,
                "intent_code": intent_code,
                "statement": seg["text"],
                "rqi": corr_idx.get(marker, ""),
                "has_rqi": marker in corr_idx,
            })
    
    return qi_list

# =============================================================================
# ZONE E : PIPELINE GTE — Qi/RQi → (ARI, FRT, Triggers) → QC
# =============================================================================

def cluster_by_intent(qi_list: List[Dict]) -> Dict[str, List[Dict]]:
    """Groupe Qi par intent_code"""
    clusters = defaultdict(list)
    for qi in qi_list:
        if qi.get("has_rqi"):  # Seulement POSABLE (avec RQi)
            clusters[qi["intent_code"]].append(qi)
    return dict(clusters)

def generate_ari_frt_triggers(cluster: List[Dict], intent_code: str) -> Dict:
    """
    ORDRE VALIDÉ: Générer (ARI, FRT, Triggers) ENSEMBLE depuis le cluster
    Avant de créer la QC (conteneur final)
    """
    h = sha8(intent_code.encode())
    
    # Analyser les RQi pour extraire l'ARI
    all_rqi = " ".join(qi.get("rqi", "") for qi in cluster).lower()
    
    # Détection des opérations cognitives (simplifié pour TEST)
    operations = []
    if any(kw in all_rqi for kw in ["calculer", "calcul", "on obtient"]):
        operations.append({"step_id": "S1", "operator": "CALCULATE", "weight": 3})
    if any(kw in all_rqi for kw in ["donc", "ainsi", "conclusion"]):
        operations.append({"step_id": "S2", "operator": "CONCLUDE", "weight": 2})
    if any(kw in all_rqi for kw in ["soit", "posons", "on a"]):
        operations.append({"step_id": "S3", "operator": "STATE", "weight": 1})
    if any(kw in all_rqi for kw in ["démontrer", "montrer", "prouver"]):
        operations.append({"step_id": "S4", "operator": "PROVE", "weight": 4})
    if any(kw in all_rqi for kw in ["appliquer", "utiliser"]):
        operations.append({"step_id": "S5", "operator": "APPLY", "weight": 2})
    
    if not operations:
        operations = [{"step_id": "S1", "operator": "SOLVE", "weight": 2}]
    
    # ARI — Algorithme de Résolution Invariant
    ari = {
        "ari_id": f"ARI_{intent_code}_{h}",
        "intent_code": intent_code,
        "steps": sorted(operations, key=lambda x: x.get("weight", 0), reverse=True),
        "total_weight": sum(op.get("weight", 0) for op in operations),
    }
    
    # FRT — Fiche de Réponse Type
    frt = {
        "frt_id": f"FRT_{intent_code}_{h}",
        "ari_ref": ari["ari_id"],
        "usage": "Résoudre ce type de question",
        "preconditions": ["Données de l'énoncé identifiées", "Formules connues"],
        "transitions": [f"Après {op['operator']}, vérifier le résultat" for op in operations[:2]],
        "traps": ["Oubli d'hypothèse", "Erreur de calcul", "Mauvaise interprétation"],
        "expected_output": "Réponse justifiée et complète",
    }
    
    # Triggers — Conditions d'activation
    triggers = [{
        "trigger_id": f"TRG_{intent_code}_{h}",
        "pattern": intent_code,
        "activation": f"Question de type {intent_code}",
        "confidence": 0.9,
    }]
    
    return {"ari": ari, "frt": frt, "triggers": triggers}

def build_qc(cluster: List[Dict], intent_code: str, ari_frt_trg: Dict) -> Optional[Dict]:
    """
    Construit la QC (conteneur final) après génération de ARI/FRT/Triggers
    Règle anti-singleton: n_q_cluster ≥ 2
    """
    if len(cluster) < 2:
        return None  # ANTI-SINGLETON
    
    h = sha8(intent_code.encode())
    qi_ids = [qi["qi_id"] for qi in cluster]
    
    qc = {
        "qc_id": f"QC_{intent_code}_{h}",
        "intent_code": intent_code,
        "formulation": "Comment procéder ?",  # Format obligatoire
        "n_q_cluster": len(cluster),
        "qi_ids": qi_ids,
        # Liens vers ARI/FRT/Triggers (générés avant)
        "ari": ari_frt_trg["ari"],
        "frt": ari_frt_trg["frt"],
        "triggers": ari_frt_trg["triggers"],
        # Métadonnées
        "created_at": datetime.now().isoformat(),
    }
    
    return qc

def map_qc_to_chapter(qc: Dict, chapters: List[Dict]) -> Optional[str]:
    """
    Mapping QC → Chapitre via intent_code + Pack.chapter_code
    Option A (Pack-driven) — Aucune sémantique dans le code
    """
    intent = qc.get("intent_code", "")
    
    for ch in chapters:
        if intent in ch.get("intent_allowlist", []):
            return ch["chapter_code"]
    
    return None  # Orphelin

def run_pipeline_gte(qi_list: List[Dict], chapters: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Pipeline complet: Qi/RQi → (ARI, FRT, Triggers) → QC → Mapping chapitre
    """
    # 1. Clustering par intent
    clusters = cluster_by_intent(qi_list)
    
    # 2. Pour chaque cluster: générer ARI/FRT/Triggers PUIS QC
    qc_list = []
    singletons = []
    
    for intent_code, cluster in clusters.items():
        if len(cluster) < 2:
            singletons.append({"intent": intent_code, "count": len(cluster)})
            continue
        
        # ORDRE: (ARI, FRT, Triggers) d'abord
        ari_frt_trg = generate_ari_frt_triggers(cluster, intent_code)
        
        # PUIS: QC (conteneur)
        qc = build_qc(cluster, intent_code, ari_frt_trg)
        if qc:
            # Mapping au chapitre
            chapter = map_qc_to_chapter(qc, chapters)
            qc["chapter_code"] = chapter
            qc_list.append(qc)
    
    stats = {
        "total_qi": len(qi_list),
        "qi_posable": sum(1 for q in qi_list if q.get("has_rqi")),
        "clusters": len(clusters),
        "singletons": len(singletons),
        "qc_generated": len(qc_list),
    }
    
    return qc_list, stats

# =============================================================================
# ZONE F : SATURATION CHECK
# =============================================================================

def check_saturation(prev_qcs: List[Dict], new_qcs: List[Dict]) -> Tuple[int, bool]:
    """
    Vérifie saturation: new_QC = 0 → SEALED
    Retourne (nb_new_qc, is_saturated)
    """
    prev_ids = {qc["qc_id"] for qc in prev_qcs}
    new_only = [qc for qc in new_qcs if qc["qc_id"] not in prev_ids]
    
    is_saturated = len(new_only) == 0
    return len(new_only), is_saturated

def group_by_chapter(qc_list: List[Dict]) -> Dict[str, List[Dict]]:
    """Ranger QC par chapitre"""
    by_chapter = defaultdict(list)
    for qc in qc_list:
        ch = qc.get("chapter_code") or "ORPHAN"
        by_chapter[ch].append(qc)
    return dict(by_chapter)

# =============================================================================
# ZONE G : STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(page_title="SMAXIA V31.9.10", page_icon="🚀", layout="wide")
    
    st.title("🚀 SMAXIA GTE V31.9.10 — ISO-PROD")
    st.caption("Flux définitif: Activation → Pack → Sélection → Harvest → Pipeline → Saturation")
    
    # Diagramme flux
    with st.expander("📊 DIAGRAMME FLUX VALIDÉ", expanded=False):
        st.code("""
[1] ACTIVATION FR ──────────────────────────────────────────────┐
                                                                │
[2] Pack FR chargé (moteur invisible)                           │
    └── UI: Affiche Niveaux / Matières / Chapitres              │
                                                                │
[3] COCHER: ☑ Terminale + ☑ Mathématiques                       │
                                                                │
[4] HARVEST AUTO (Pêche aveugle)                                │
    └── UI: Barre progression "80/100 sujets"                   │
                                                                │
[5] PIPELINE GTE                                                │
    └── Qi/RQi → (ARI, FRT, Triggers) → QC                      │
                                                                │
[6] RANGER QC PAR CHAPITRE                                      │
    └── Via intent_code → Pack.chapter_code                     │
                                                                │
[7] BOUCLE SATURATION ──────────────────────────────────────────┤
    └── SI new_QC = 0 → 🔒 SEALED                               │
    └── SINON → +volume, retour [4] ────────────────────────────┘
                                                                
[8] EXPLORATEUR: QC → ARI → FRT → Triggers → Qi
[9] EXPORTS
        """)
    
    # Session state
    defaults = {
        "pack": None, "level": None, "subjects": [],
        "qi_global": [], "qcs_global": [], "iterations": [],
        "harvested": [], "saturated": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ═══════════════════════════════════════════════════════════════════════════
    # [1] ACTIVATION PAYS
    # ═══════════════════════════════════════════════════════════════════════════
    st.header("🌍 [1] Activation Pays")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        available_countries = ["FR", "CI"]
        country = st.selectbox("Pays", available_countries)
        
        if st.button("🔓 ACTIVER", type="primary", use_container_width=True):
            # Moteur invisible: load_academic_pack
            pack = load_academic_pack(country)
            if pack:
                st.session_state.pack = pack
                st.session_state.qi_global = []
                st.session_state.qcs_global = []
                st.session_state.iterations = []
                st.session_state.harvested = []
                st.session_state.saturated = False
                st.success(f"✅ Pack {country} activé")
            else:
                st.error(f"❌ Pack {country} non disponible")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2] PACK CHARGÉ — Affichage métadonnées
    # ═══════════════════════════════════════════════════════════════════════════
    with col2:
        if st.session_state.pack:
            pack = st.session_state.pack
            acs = pack.get("academic_structure", {})
            
            st.success(f"**Pack:** {pack['pack_id']} | Version: {pack.get('version')}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**📚 Niveaux**")
                for lv in acs.get("levels", [])[:5]:
                    st.caption(f"• {lv['label']}")
                if len(acs.get("levels", [])) > 5:
                    st.caption(f"... +{len(acs['levels']) - 5} autres")
            with c2:
                st.markdown("**📖 Matières**")
                for sb in acs.get("subjects", []):
                    st.caption(f"• {sb['label']}")
            with c3:
                st.markdown("**📑 Chapitres**")
                for ch in pack.get("chapters", [])[:4]:
                    st.caption(f"• {ch['label']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # [3] SÉLECTION NIVEAU + MATIÈRES
    # ═══════════════════════════════════════════════════════════════════════════
    if st.session_state.pack:
        st.markdown("---")
        st.header("📚 [3] Sélection (Niveau + Matières)")
        
        pack = st.session_state.pack
        acs = pack.get("academic_structure", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            levels = acs.get("levels", [])
            level_ids = [l["id"] for l in levels]
            level_labels = {l["id"]: l["label"] for l in levels}
            
            selected_level = st.radio(
                "Niveau",
                level_ids,
                format_func=lambda x: level_labels.get(x, x),
                horizontal=True
            )
            st.session_state.level = selected_level
        
        with col2:
            available = acs.get("level_subjects", {}).get(selected_level, [])
            subject_labels = {s["id"]: s["label"] for s in acs.get("subjects", [])}
            
            selected = st.multiselect(
                "Matières (cocher)",
                available,
                default=available[:1] if available else [],
                format_func=lambda x: subject_labels.get(x, x)
            )
            st.session_state.subjects = selected
        
        if selected_level and selected:
            st.success(f"✅ **{level_labels[selected_level]}** / {', '.join(subject_labels.get(s, s) for s in selected)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # [4-7] LANCER PIPELINE + SATURATION
    # ═══════════════════════════════════════════════════════════════════════════
    if st.session_state.pack and st.session_state.level and st.session_state.subjects:
        st.markdown("---")
        st.header("🚀 [4-7] Pipeline + Saturation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            volume = st.number_input("Volume initial", min_value=5, value=20, step=5)
        with col2:
            max_iter = st.number_input("Max itérations", min_value=1, value=5)
        with col3:
            volume_incr = st.number_input("Incrément", min_value=10, value=30)
        
        # Zones d'affichage
        progress_area = st.empty()
        status_area = st.empty()
        metrics_area = st.empty()
        
        if st.button("⚡ LANCER PIPELINE COMPLET", type="primary", use_container_width=True):
            pack = st.session_state.pack
            chapters = pack.get("chapters", [])
            
            current_volume = volume
            iteration = 0
            
            while iteration < max_iter and not st.session_state.saturated:
                iteration += 1
                
                status_area.info(f"🔄 **Itération {iteration}** — Volume: {current_volume} sujets/matière")
                
                # [4] HARVEST avec progression
                all_pairs = []
                for subj in st.session_state.subjects:
                    def progress_cb(current, total, url):
                        progress_area.progress(
                            current / total,
                            f"Harvest {subj}: {current}/{total} — {url.split('/')[-1][:30]}"
                        )
                    
                    pairs = harvest_subjects(
                        pack, 
                        st.session_state.level, 
                        subj, 
                        current_volume,
                        progress_cb
                    )
                    all_pairs.extend(pairs)
                
                progress_area.empty()
                
                if not all_pairs:
                    status_area.warning("⚠️ Aucun sujet récupéré")
                    break
                
                st.session_state.harvested.extend(all_pairs)
                
                # [5] Extraction Qi/RQi
                new_qi = extract_qi_rqi(all_pairs)
                
                # Merge
                existing_ids = {q["qi_id"] for q in st.session_state.qi_global}
                truly_new = [q for q in new_qi if q["qi_id"] not in existing_ids]
                st.session_state.qi_global.extend(truly_new)
                
                # [5-6] Pipeline GTE
                qc_list, stats = run_pipeline_gte(st.session_state.qi_global, chapters)
                
                # [7] Check saturation
                new_qc_count, is_saturated = check_saturation(st.session_state.qcs_global, qc_list)
                st.session_state.qcs_global = qc_list
                
                # Enregistrer itération
                st.session_state.iterations.append({
                    "iter": iteration,
                    "volume": current_volume,
                    "pairs": len(all_pairs),
                    "new_qi": len(truly_new),
                    "new_qc": new_qc_count,
                    "total_qi": len(st.session_state.qi_global),
                    "total_qc": len(qc_list),
                })
                
                # Afficher métriques
                metrics_area.dataframe(st.session_state.iterations, use_container_width=True)
                
                if is_saturated:
                    st.session_state.saturated = True
                    status_area.success("🎯 **SATURATION ATTEINTE** — new_QC = 0 → 🔒 SEALED")
                    st.balloons()
                    break
                
                current_volume += volume_incr
            
            if not st.session_state.saturated:
                status_area.warning(f"⏳ Non saturé après {iteration} itérations")

    # ═══════════════════════════════════════════════════════════════════════════
    # [8] EXPLORATEUR
    # ═══════════════════════════════════════════════════════════════════════════
    if st.session_state.qcs_global:
        st.markdown("---")
        st.header("🔍 [8] Explorateur: QC → ARI → FRT → Triggers → Qi")
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Qi POSABLE", sum(1 for q in st.session_state.qi_global if q.get("has_rqi")))
        col2.metric("QC générées", len(st.session_state.qcs_global))
        col3.metric("Itérations", len(st.session_state.iterations))
        col4.metric("Statut", "🔒 SEALED" if st.session_state.saturated else "⏳")
        
        # QC par chapitre
        st.subheader("📚 QC par Chapitre")
        qc_by_chapter = group_by_chapter(st.session_state.qcs_global)
        
        for chapter_code, qcs in sorted(qc_by_chapter.items()):
            # Trouver label du chapitre
            label = chapter_code
            if st.session_state.pack:
                for ch in st.session_state.pack.get("chapters", []):
                    if ch["chapter_code"] == chapter_code:
                        label = ch["label"]
                        break
            
            with st.expander(f"**{chapter_code}** — {label} ({len(qcs)} QC)", expanded=False):
                for qc in qcs:
                    st.markdown(f"### {qc['qc_id']}")
                    st.markdown(f"**Formulation:** {qc['formulation']}")
                    st.markdown(f"**Cluster:** {qc['n_q_cluster']} Qi")
                    
                    # ARI
                    ari = qc.get("ari", {})
                    st.markdown("**ARI (Algorithme de Résolution):**")
                    for step in ari.get("steps", []):
                        st.caption(f"  • {step['step_id']}: {step['operator']}")
                    
                    # FRT
                    frt = qc.get("frt", {})
                    st.markdown("**FRT (Fiche Réponse Type):**")
                    st.caption(f"  Usage: {frt.get('usage')}")
                    st.caption(f"  Pièges: {', '.join(frt.get('traps', []))}")
                    
                    # Triggers
                    st.markdown("**Triggers:**")
                    for trg in qc.get("triggers", []):
                        st.caption(f"  • {trg['trigger_id']}: {trg['pattern']}")
                    
                    # Qi liées
                    st.markdown("**Qi liées:**")
                    st.caption(f"  {', '.join(qc.get('qi_ids', [])[:5])}...")
                    
                    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # [9] EXPORTS
    # ═══════════════════════════════════════════════════════════════════════════
    if st.session_state.iterations:
        st.markdown("---")
        st.header("📤 [9] Exports")
        
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.download_button(
                "📥 qi_pack.json",
                canonical_json(st.session_state.qi_global),
                "qi_pack.json",
                use_container_width=True
            )
        
        with c2:
            st.download_button(
                "📥 qc_pack.json",
                canonical_json(st.session_state.qcs_global),
                "qc_pack.json",
                use_container_width=True
            )
        
        with c3:
            report = {
                "version": "V31.9.10",
                "pack_id": st.session_state.pack.get("pack_id") if st.session_state.pack else None,
                "level": st.session_state.level,
                "subjects": st.session_state.subjects,
                "iterations": st.session_state.iterations,
                "saturated": st.session_state.saturated,
                "qc_by_chapter": {k: len(v) for k, v in group_by_chapter(st.session_state.qcs_global).items()},
            }
            st.download_button(
                "📥 report.json",
                canonical_json(report),
                "saturation_report.json",
                use_container_width=True
            )
        
        with c4:
            if st.session_state.pack:
                st.download_button(
                    "📥 pack.json",
                    canonical_json(st.session_state.pack),
                    "academic_pack.json",
                    use_container_width=True
                )

    # Upload manuel (fallback)
    if st.session_state.pack:
        st.markdown("---")
        with st.expander("📤 Upload manuel (si Harvest indisponible)", expanded=False):
            subj_files = st.file_uploader("PDF Sujets", type=["pdf"], accept_multiple_files=True, key="up_subj")
            corr_files = st.file_uploader("PDF Corrigés", type=["pdf"], accept_multiple_files=True, key="up_corr")
            
            if subj_files and st.button("Traiter manuellement"):
                pairs = []
                for sf in subj_files:
                    sb = sf.read()
                    sf.seek(0)
                    cb = None
                    if corr_files:
                        for cf in corr_files:
                            if sf.name.replace("enonce", "") in cf.name or cf.name.replace("corrige", "") in sf.name:
                                cb = cf.read()
                                cf.seek(0)
                                break
                    pairs.append({
                        "pair_id": f"MANUAL_{sha8(sb)}",
                        "label": sf.name,
                        "subject_bytes": sb,
                        "correction_bytes": cb,
                    })
                
                qi = extract_qi_rqi(pairs)
                st.session_state.qi_global.extend(qi)
                qcs, _ = run_pipeline_gte(st.session_state.qi_global, st.session_state.pack.get("chapters", []))
                st.session_state.qcs_global = qcs
                
                st.success(f"✅ Traité {len(pairs)} fichiers → {len(qi)} Qi, {len(qcs)} QC")

if __name__ == "__main__":
    main()
