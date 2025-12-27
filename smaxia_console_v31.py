# smaxia_console_v31.py
# =============================================================================
# SMAXIA - Console V31 (Saturation Proof)
# =============================================================================
# UI scellée V31 — ZÉRO RÉGRESSION
# Moteur: Granulo V10.4 via adapter
# =============================================================================

import streamlit as st
import pandas as pd
from collections import defaultdict
from datetime import datetime
import io

# Import de l'adapter V10.4 (mêmes signatures que l'ancien moteur)
from smaxia_granulo_console_adapter import (
    ingest_from_pdfs,
    compute_qc_real,
    compute_saturation_real,
    audit_internal_real,
    audit_external_real,
    get_coverage_details,
    get_audit_log,
    CHAPTER_KEYWORDS
)

from smaxia_granulo_engine_v104 import KernelConstants

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
    <div style="background: linear-gradient(135deg, #2563eb, #1d4ed8); width: 50px; height: 50px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
        <span style="font-size: 24px;">🛡️</span>
    </div>
    <span style="font-size: 2.2rem; font-weight: 800; color: #1e293b;">SMAXIA - Console V31 (Saturation Proof)</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 🎨 STYLES CSS (GABARIT SMAXIA V31 - INCHANGÉ)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
    .qc-header-box {
        background-color: #f8f9fa; border-left: 6px solid #2563eb; 
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .qc-id-text { color: #d97706; font-weight: 900; font-size: 1.2em; margin-right: 10px; }
    .qc-title-text { color: #1f2937; font-weight: 700; font-size: 1.15em; }
    .qc-meta-text { 
        font-family: 'Courier New', monospace; font-size: 0.85em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 4px 8px; border-radius: 4px; margin-top: 5px; display: inline-block;
    }

    /* DETAILS */
    .trigger-item {
        background-color: #fff1f2; color: #991b1b; padding: 5px 10px; margin-bottom: 4px; 
        border-radius: 4px; border-left: 4px solid #f87171; font-weight: 600; font-size: 0.9em; display: block;
    }
    .ari-step {
        background-color: #f3f4f6; color: #374151; padding: 4px 8px; margin-bottom: 3px; 
        border-radius: 3px; font-family: monospace; font-size: 0.85em; border: 1px dashed #d1d5db; display: block;
    }

    /* FRT */
    .frt-block { padding: 12px; border-bottom: 1px solid #e2e8f0; background: white; margin-bottom: 5px; border-radius: 4px; border: 1px solid #e2e8f0;}
    .frt-title { font-weight: 800; text-transform: uppercase; font-size: 0.75em; display: block; margin-bottom: 6px; letter-spacing: 0.5px; }
    .frt-content { font-family: 'Segoe UI', sans-serif; font-size: 0.95em; color: #334155; line-height: 1.6; white-space: pre-wrap; }
    
    .c-usage { color: #d97706; border-left: 4px solid #d97706; }
    .c-method { color: #059669; border-left: 4px solid #059669; background-color: #f0fdf4; }
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; }
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; }

    /* QI CARDS */
    .file-block { margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
    .file-header { background-color: #f1f5f9; padding: 8px 12px; font-weight: 700; font-size: 0.85em; color: #475569; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; }
    .qi-item { background-color: white; padding: 10px 12px; border-bottom: 1px solid #f8fafc; font-family: 'Georgia', serif; font-size: 0.95em; color: #1e293b; border-left: 3px solid #9333ea; margin: 0; }

    /* SATURATION */
    .sat-box { background-color: #f0f9ff; border: 1px solid #bae6fd; padding: 20px; border-radius: 8px; margin-top: 20px; }
    
    /* STATUS */
    .status-kernel { background-color: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-gte { background-color: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-info { background-color: #dbeafe; color: #1e40af; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; }
    
    /* TABLE SUJETS */
    .sujets-table { width: 100%; border-collapse: collapse; }
    .sujets-table th { background: #f1f5f9; padding: 10px; text-align: left; font-weight: 600; color: #475569; }
    .sujets-table td { padding: 10px; border-bottom: 1px solid #e2e8f0; }
    .pdf-link { color: #dc2626; text-decoration: none; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR - PARAMÈTRES ACADÉMIQUES
# ==============================================================================
LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE DANS L'ESPACE", 
              "INTÉGRALES", "NOMBRES COMPLEXES", "MATRICES", "ARITHMÉTIQUE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

with st.sidebar:
    st.header("📚 Paramètres Académiques")
    
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps = LISTE_CHAPITRES.get(sel_matiere, [])
    sel_chapitres = st.multiselect("Chapitres", chaps, default=chaps[:1] if chaps else [])
    
    st.divider()
    
    # Mode Kernel
    st.markdown(f'<span class="status-kernel">Kernel {KernelConstants.KERNEL_VERSION}</span>', unsafe_allow_html=True)
    st.caption(f"Status: {KernelConstants.KERNEL_STATUS}")
    
    gte_mode = st.checkbox(
        "Mode GTE (prévisualisation sans corrigé)",
        value=True,
        help="Active le mode prévisualisation. Les QC générées ne sont pas scellables."
    )
    
    if gte_mode:
        st.markdown('<span class="status-gte">⚠️ Mode GTE: QC non scellables</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Stats en temps réel
    if 'all_qis' in st.session_state and st.session_state['all_qis']:
        st.markdown("### 📊 Statistiques")
        st.metric("Qi extraites", len(st.session_state['all_qis']))
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            st.metric("QC générées", len(st.session_state['df_qc']))
        if 'v104_results' in st.session_state:
            v104 = st.session_state['v104_results']
            st.metric("POSABLES", v104.get('total_posable', 0))

# ==============================================================================
# ONGLETS
# ==============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# ==============================================================================
# ONGLET 1 - USINE
# ==============================================================================
with tab_usine:
    
    # --- ZONE INJECTION ---
    c1, c2 = st.columns([3, 1])
    with c1:
        urls = st.text_area(
            "URLs Sources",
            "https://apmep.fr",
            height=80,
            help="URLs de référence (actuellement en mode upload PDF direct)"
        )
    with c2:
        vol = st.number_input("Volume", 1, 50, 5, step=1)
        run = st.button("🚀 LANCER L'USINE", type="primary", use_container_width=True)

    st.divider()
    
    # --- UPLOAD PDFs ---
    st.markdown("### 📄 Sujets à traiter")
    
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        subject_files = st.file_uploader(
            "📥 Sujets (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True,
            key="subjects",
            help="Uploadez les sujets d'examen au format PDF"
        )
    with col_upload2:
        correction_files = st.file_uploader(
            "📝 Corrigés (PDF, optionnel)", 
            type=["pdf"], 
            accept_multiple_files=True,
            key="corrections",
            help="Uploadez les corrigés correspondants pour activer le mode SEALED"
        )

    # --- EXÉCUTION ---
    if run and subject_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Extraction et analyse des PDFs...")
        
        try:
            # Préparer les fichiers
            pdf_list = [(f.name, f.read()) for f in subject_files]
            for f in subject_files:
                f.seek(0)
            
            corr_list = None
            if correction_files:
                corr_list = [(f.name, f.read()) for f in correction_files]
                for f in correction_files:
                    f.seek(0)
            
            chapter_filter = sel_chapitres[0] if sel_chapitres else None
            
            # Appel adapter V10.4
            df_src, df_atoms, all_qis, v104_results = ingest_from_pdfs(
                pdf_files=pdf_list,
                matiere=sel_matiere,
                chapter_filter=chapter_filter,
                correction_files=corr_list,
                gte_mode=gte_mode,
                progress_callback=lambda p: progress_bar.progress(p)
            )
            
            if all_qis:
                status_text.text("🧠 Génération des QC...")
                df_qc = compute_qc_real(all_qis, v104_results)
                
                st.session_state['df_src'] = df_src
                st.session_state['df_qc'] = df_qc
                st.session_state['all_qis'] = all_qis
                st.session_state['v104_results'] = v104_results
                st.session_state['chapter_filter'] = chapter_filter
                
                status_text.empty()
                progress_bar.empty()
                
                mode_str = "GTE (preview)" if gte_mode else "KERNEL"
                st.success(f"✅ Traitement terminé [{mode_str}]: {len(df_src)} sujets, {len(all_qis)} Qi, {len(df_qc)} QC")
            else:
                status_text.empty()
                progress_bar.empty()
                st.warning("⚠️ Aucune Qi extraite. Vérifiez les PDFs.")
                
        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.error(f"❌ Erreur : {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    st.divider()

    # --- TABLEAU SUJETS ---
    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        df_view = st.session_state['df_src'].copy()
        if 'Annee' in df_view.columns:
            df_view = df_view.rename(columns={"Annee": "Année"})
        
        display_cols = [c for c in ["Fichier", "Nature", "Année"] if c in df_view.columns]
        if display_cols:
            st.dataframe(df_view[display_cols], hide_index=True, use_container_width=True)

        st.divider()

        # --- BASE QC ---
        st.markdown("### 🧠 Base de Connaissance (QC)")
        
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            df_qc = st.session_state['df_qc']
            
            # Filtrer par chapitres sélectionnés
            if sel_chapitres:
                qc_view = df_qc[df_qc["Chapitre"].isin(sel_chapitres)]
            else:
                qc_view = df_qc
            
            if qc_view.empty:
                st.info("Aucune QC pour les chapitres sélectionnés.")
            else:
                chapters = qc_view["Chapitre"].unique()
                
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    
                    # Badge sealed si au moins une QC scellée
                    any_sealed = subset.get("Sealed", pd.Series([False])).any()
                    seal_badge = " ✅ SEALED" if any_sealed else ""
                    
                    st.markdown(f"#### 📘 {chap} ({len(subset)} QC){seal_badge}")
                    
                    for idx, row in subset.iterrows():
                        # Mode badge
                        mode = row.get('Mode', 'KERNEL')
                        mode_badge = ""
                        if mode == "GTE_PREVIEW":
                            mode_badge = '<span style="background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:4px;font-size:0.7em;margin-left:8px;">GTE</span>'
                        elif row.get('Sealed', False):
                            mode_badge = '<span style="background:#dcfce7;color:#166534;padding:2px 6px;border-radius:4px;font-size:0.7em;margin-left:8px;">SEALED</span>'
                        
                        # Header QC
                        st.markdown(f"""
                        <div class="qc-header-box">
                            <span class="qc-id-text">{row['QC_ID']}</span>{mode_badge}
                            <span class="qc-title-text">{row['Titre']}</span><br>
                            <span class="qc-meta-text">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_réc={row['t_rec']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 4 colonnes
                        c1, c2, c3, c4 = st.columns(4)
                        
                        with c1:
                            with st.expander("🔥 Déclencheurs"):
                                triggers = row['Triggers'] if isinstance(row['Triggers'], list) else []
                                if triggers:
                                    for t in triggers:
                                        st.markdown(f"<span class='trigger-item'>\"{t}\"</span>", unsafe_allow_html=True)
                                else:
                                    st.caption("Aucun déclencheur identifié")
                        
                        with c2:
                            with st.expander("⚙️ ARI"):
                                ari = row['ARI'] if isinstance(row['ARI'], list) else []
                                if ari:
                                    for s in ari:
                                        st.markdown(f"<span class='ari-step'>{s}</span>", unsafe_allow_html=True)
                                else:
                                    st.caption("ARI non disponible")
                        
                        with c3:
                            with st.expander("🧾 FRT"):
                                frt_data = row['FRT_DATA'] if isinstance(row['FRT_DATA'], list) else []
                                for block in frt_data:
                                    cls = {"usage": "c-usage", "method": "c-method", "trap": "c-trap", "conc": "c-conc"}.get(block.get('type', ''), "")
                                    st.markdown(f"<div class='frt-block {cls}'><span class='frt-title'>{block.get('title', '')}</span><div class='frt-content'>{block.get('text', '')}</div></div>", unsafe_allow_html=True)
                        
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                evidence = row['Evidence'] if isinstance(row['Evidence'], list) else []
                                qi_by_file = defaultdict(list)
                                for item in evidence:
                                    qi_by_file[item.get('Fichier', 'unknown')].append(item.get('Qi', ''))
                                
                                html = ""
                                for f, qlist in qi_by_file.items():
                                    html += f"<div class='file-block'><div class='file-header'>📁 {f}</div>"
                                    for q in qlist[:5]:
                                        q_display = q[:100] + "..." if len(q) > 100 else q
                                        html += f"<div class='qi-item'>\"{q_display}\"</div>"
                                    if len(qlist) > 5:
                                        html += f"<div class='qi-item'>… +{len(qlist)-5} autres</div>"
                                    html += "</div>"
                                st.markdown(html, unsafe_allow_html=True)
                        
                        st.write("")
        else:
            st.warning("Aucune QC générée. Lancez l'usine d'abord.")
        
        # --- SATURATION ---
        st.divider()
        st.markdown("### 📈 Analyse de Saturation (Preuve de Complétude)")
        st.caption("Ce graphique montre l'évolution du nombre de QC en fonction des sujets traités.")
        
        if 'all_qis' in st.session_state and st.session_state['all_qis']:
            if st.button("📊 Calculer la courbe de saturation"):
                with st.spinner("Calcul de la saturation..."):
                    df_sat = compute_saturation_real(st.session_state['all_qis'])
                    
                    if not df_sat.empty:
                        st.line_chart(df_sat, x="Sujets (N)", y="QC Découvertes")
                        
                        st.markdown("#### 🔢 Données de Convergence")
                        step = max(1, len(df_sat) // 10)
                        df_display = df_sat.iloc[::step].reset_index(drop=True)
                        st.dataframe(df_display, use_container_width=True)
                        
                        max_qc = df_sat["QC Découvertes"].max()
                        last_values = df_sat["QC Découvertes"].tail(3).tolist()
                        
                        if len(set(last_values)) == 1:
                            st.success(f"✅ **Saturation atteinte !** ({max_qc} QC max)")
                        else:
                            sat_90 = df_sat[df_sat["QC Découvertes"] >= max_qc * 0.9]
                            if not sat_90.empty:
                                sat_point = sat_90.iloc[0]["Sujets (N)"]
                                st.info(f"📈 **Saturation ~90% atteinte à {sat_point} sujets.**")
                            else:
                                st.warning("⚠️ Saturation non atteinte. Ajoutez plus de sujets.")
                    else:
                        st.warning("Pas assez de données pour la saturation.")
        else:
            st.info("Lancez l'usine pour voir la courbe de saturation.")
    else:
        st.info("⏳ Uploadez des sujets PDF et lancez l'usine pour commencer.")

# ==============================================================================
# ONGLET 2 - AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("🔍 Validation Booléenne")
    
    st.markdown("""
    **Objectifs Kernel V10.4 :**
    - **POSABLE** : corrigé exploitable ∧ scope ∧ évaluable
    - **Audit Interne** : Chaque Qi d'un sujet traité → QC = **100%**
    - **Audit Externe** : Sujet inconnu → QC = **≥ 95%**
    """)
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        # --- COVERAGE DETAILS (V10.4) ---
        if 'v104_results' in st.session_state:
            st.divider()
            st.markdown("#### 📊 Coverage par Chapitre (Kernel V10.4)")
            
            coverage_details = get_coverage_details(st.session_state['v104_results'])
            
            if coverage_details:
                for ch, details in coverage_details.items():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        seal_badge = "✅ SEALED" if details['sealed'] else "❌ NOT SEALED"
                        st.markdown(f"**{ch}** {seal_badge}")
                    
                    with col2:
                        st.metric("Coverage", f"{details['coverage_ratio']*100:.0f}%")
                    
                    with col3:
                        st.metric("Couvertes", f"{details['n_covered']}/{details['n_total']}")
                    
                    with col4:
                        if details['orphans'] > 0:
                            st.error(f"❌ {details['orphans']} orphelins")
                        else:
                            st.success("✅ 0 orphelin")
            else:
                st.info("Aucune donnée de coverage disponible.")
            
            # Audit Log
            st.divider()
            st.markdown("#### 📋 Audit Log")
            
            logs = get_audit_log(st.session_state['v104_results'])
            
            with st.expander("Voir le log complet", expanded=False):
                for log in logs[-20:]:  # Derniers 20 logs
                    if "PASS" in log:
                        st.success(log)
                    elif "FAIL" in log or "BLOCKED" in log:
                        st.error(log)
                    else:
                        st.info(log)
        
        # --- AUDIT INTERNE ---
        st.divider()
        st.markdown("#### ✅ 1. Test Interne (Sujet Traité)")
        
        if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
            fichiers = st.session_state['df_src']["Fichier"].tolist()
            t1_file = st.selectbox("Choisir un sujet traité", fichiers)
            
            if st.button("🔬 AUDIT INTERNE"):
                row = st.session_state['df_src'][st.session_state['df_src']["Fichier"] == t1_file].iloc[0]
                qi_data = row.get("Qi_Data", [])
                
                results = audit_internal_real(qi_data, st.session_state['df_qc'])
                
                if results:
                    matched = sum(1 for r in results if r["Statut"] == "✅ MATCH")
                    coverage = (matched / len(results)) * 100 if results else 0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Couverture", f"{coverage:.0f}%")
                    col2.metric("Qi mappées", f"{matched}/{len(results)}")
                    
                    if coverage >= 100:
                        st.success("✅ 100% de couverture - SUCCÈS")
                    elif coverage >= 80:
                        st.warning(f"⚠️ {coverage:.0f}% de couverture - À améliorer")
                    else:
                        st.error(f"❌ {coverage:.0f}% de couverture - INSUFFISANT")
                    
                    df_results = pd.DataFrame(results)
                    
                    def highlight_status(row):
                        if row['Statut'] == "✅ MATCH":
                            return ['background-color: #dcfce7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(
                        df_results.style.apply(highlight_status, axis=1),
                        use_container_width=True
                    )
                else:
                    st.warning("Aucune Qi trouvée dans ce sujet.")
        
        # --- AUDIT EXTERNE ---
        st.divider()
        st.markdown("#### 🌍 2. Test Externe (Sujet Inconnu)")
        
        uploaded = st.file_uploader("Charger un PDF externe", type="pdf", key="audit_external")
        
        if uploaded:
            if st.button("🔬 AUDIT EXTERNE"):
                pdf_bytes = uploaded.read()
                chapter_filter = st.session_state.get('chapter_filter', sel_chapitres[0] if sel_chapitres else None)
                
                with st.spinner("Analyse du sujet externe..."):
                    coverage, results = audit_external_real(pdf_bytes, st.session_state['df_qc'], chapter_filter)
                
                if results:
                    matched = sum(1 for r in results if r["Statut"] == "✅ MATCH")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Couverture", f"{coverage:.0f}%")
                    col2.metric("Qi couvertes", f"{matched}/{len(results)}")
                    
                    if coverage >= 95:
                        st.success(f"✅ {coverage:.0f}% de couverture - OBJECTIF ATTEINT (≥95%)")
                    elif coverage >= 80:
                        st.warning(f"⚠️ {coverage:.0f}% de couverture - PROCHE DE L'OBJECTIF")
                    else:
                        st.error(f"❌ {coverage:.0f}% de couverture - INSUFFISANT")
                    
                    df_results = pd.DataFrame(results)
                    
                    def highlight_status(row):
                        if row['Statut'] == "✅ MATCH":
                            return ['background-color: #dcfce7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(
                        df_results.style.apply(highlight_status, axis=1),
                        use_container_width=True
                    )
                else:
                    st.warning("Aucune Qi extraite du PDF externe.")
    else:
        st.info("⏳ Lancez d'abord l'usine pour générer des QC.")

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.caption(f"SMAXIA Console V31 | Kernel {KernelConstants.KERNEL_VERSION} ({KernelConstants.KERNEL_STATUS}) | {datetime.now().strftime('%Y-%m-%d')}")
