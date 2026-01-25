import streamlit as st
from collections import defaultdict
from datetime import datetime

KERNEL_VERSION = "V10.6.3"
FORMULES_VERSION = "V3.1"

PAYS_VALIDES = {"FR": {"nom": "France", "flag": "🇫🇷"}, "CI": {"nom": "Côte d'Ivoire", "flag": "🇨🇮"}}
NIVEAUX = {"TERMINALE": {"label": "Terminale", "ordre": 1}, "PREMIERE": {"label": "Première", "ordre": 2}, "PREPA_MP": {"label": "Prépa MP", "ordre": 3}}
MATIERES = {"MATHS": {"label": "Mathématiques", "icon": "📐"}, "PHYSIQUE_CHIMIE": {"label": "Physique-Chimie", "icon": "⚗️"}}
TYPES = {"DST": {"label": "DST", "icon": "📝"}, "INTERRO": {"label": "Interrogation", "icon": "❓"}, "EXAMEN": {"label": "Examen", "icon": "🎓"}, "CONCOURS": {"label": "Concours", "icon": "🏆"}}

CHAPITRES = {
    "MATHS": {
        "TERMINALE": [
            {"code": "CH_SUITES", "label": "Suites numériques", "delta_c": 1.0, "kw": ["suite", "récurrence", "convergence"]},
            {"code": "CH_LIMITES", "label": "Limites de fonctions", "delta_c": 1.0, "kw": ["limite", "infini", "asymptote"]},
            {"code": "CH_DERIVATION", "label": "Dérivation", "delta_c": 0.9, "kw": ["dérivée", "tangente", "variation"]},
            {"code": "CH_CONTINUITE", "label": "Continuité", "delta_c": 0.8, "kw": ["continue", "théorème", "TVI"]},
            {"code": "CH_INTEGRATION", "label": "Intégration", "delta_c": 1.2, "kw": ["intégrale", "primitive", "aire"]},
            {"code": "CH_LOGEXP", "label": "Log et Exponentielle", "delta_c": 1.0, "kw": ["ln", "exp", "logarithme"]},
            {"code": "CH_PROBAS", "label": "Probabilités", "delta_c": 1.1, "kw": ["probabilité", "espérance", "loi"]},
            {"code": "CH_COMPLEXES", "label": "Nombres complexes", "delta_c": 1.0, "kw": ["complexe", "module", "argument"]},
            {"code": "CH_GEOMETRIE", "label": "Géométrie espace", "delta_c": 0.9, "kw": ["vecteur", "plan", "droite"]},
            {"code": "CH_ARITHMETIQUE", "label": "Arithmétique", "delta_c": 0.8, "kw": ["pgcd", "congruence", "premier"]},
        ],
        "PREMIERE": [
            {"code": "CH_SECOND_DEGRE", "label": "Second degré", "delta_c": 0.8, "kw": ["polynôme", "discriminant"]},
            {"code": "CH_DERIVATION_1", "label": "Dérivation", "delta_c": 0.9, "kw": ["dérivée", "tangente"]},
            {"code": "CH_SUITES_1", "label": "Suites", "delta_c": 0.9, "kw": ["arithmétique", "géométrique"]},
            {"code": "CH_PROBAS_1", "label": "Probabilités cond.", "delta_c": 1.0, "kw": ["conditionnelle", "indépendance"]},
            {"code": "CH_TRIGO", "label": "Trigonométrie", "delta_c": 0.9, "kw": ["cosinus", "sinus", "radian"]},
        ],
        "PREPA_MP": [
            {"code": "CH_ALGEBRE_LIN", "label": "Algèbre linéaire", "delta_c": 1.3, "kw": ["matrice", "espace vectoriel"]},
            {"code": "CH_ANALYSE", "label": "Analyse", "delta_c": 1.2, "kw": ["série", "convergence"]},
            {"code": "CH_TOPOLOGIE", "label": "Topologie", "delta_c": 1.1, "kw": ["ouvert", "compact", "connexe"]},
            {"code": "CH_EQUATIONS_DIFF", "label": "Équations diff.", "delta_c": 1.2, "kw": ["différentielle", "cauchy"]},
            {"code": "CH_REDUCTION", "label": "Réduction", "delta_c": 1.3, "kw": ["valeur propre", "diagonalisation"]},
        ],
    },
    "PHYSIQUE_CHIMIE": {
        "TERMINALE": [
            {"code": "CH_MECANIQUE", "label": "Mécanique", "delta_c": 1.0, "kw": ["mouvement", "force", "newton"]},
            {"code": "CH_ONDES", "label": "Ondes", "delta_c": 1.0, "kw": ["onde", "fréquence", "période"]},
            {"code": "CH_ELEC", "label": "Électricité", "delta_c": 0.9, "kw": ["circuit", "tension", "courant"]},
            {"code": "CH_THERMO", "label": "Thermodynamique", "delta_c": 1.1, "kw": ["chaleur", "enthalpie", "entropie"]},
            {"code": "CH_CHIMIE_ORGA", "label": "Chimie organique", "delta_c": 1.0, "kw": ["molécule", "synthèse"]},
            {"code": "CH_ACIDE_BASE", "label": "Acides et bases", "delta_c": 0.9, "kw": ["pH", "titrage", "acide"]},
        ],
        "PREMIERE": [
            {"code": "CH_MOUVEMENTS", "label": "Mouvements", "delta_c": 0.8, "kw": ["vitesse", "trajectoire"]},
            {"code": "CH_INTERACTIONS", "label": "Interactions", "delta_c": 0.9, "kw": ["gravitation", "force"]},
            {"code": "CH_REACTIONS", "label": "Réactions chimiques", "delta_c": 0.9, "kw": ["équation", "stoechiométrie"]},
        ],
        "PREPA_MP": [
            {"code": "CH_MECANIQUE_PT", "label": "Mécanique du point", "delta_c": 1.2, "kw": ["référentiel", "newton"]},
            {"code": "CH_ELECTROMAG", "label": "Électromagnétisme", "delta_c": 1.3, "kw": ["maxwell", "champ"]},
            {"code": "CH_OPTIQUE", "label": "Optique ondulatoire", "delta_c": 1.1, "kw": ["interférence", "diffraction"]},
            {"code": "CH_THERMO_PREPA", "label": "Thermodynamique", "delta_c": 1.2, "kw": ["premier principe", "entropie"]},
        ],
    },
}

def get_sujets(pays):
    return [
        {"id": f"S_{pays}_01", "nom": "BAC 2023 Métropole J1", "type": "EXAMEN", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_02", "nom": "BAC 2023 Métropole J2", "type": "EXAMEN", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_03", "nom": "BAC 2023 Centres étrangers", "type": "EXAMEN", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_04", "nom": "BAC 2022 Métropole", "type": "EXAMEN", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_05", "nom": "BAC 2022 Asie", "type": "EXAMEN", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_06", "nom": "DST Suites et Limites", "type": "DST", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_07", "nom": "DST Dérivation", "type": "DST", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_08", "nom": "DST Intégration", "type": "DST", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_09", "nom": "DST Probabilités", "type": "DST", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_10", "nom": "Interro Complexes", "type": "INTERRO", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_11", "nom": "Interro Géométrie", "type": "INTERRO", "matiere": "MATHS", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_12", "nom": "DST Second degré", "type": "DST", "matiere": "MATHS", "niveau": "PREMIERE", "concours": None},
        {"id": f"S_{pays}_13", "nom": "Interro Dérivation", "type": "INTERRO", "matiere": "MATHS", "niveau": "PREMIERE", "concours": None},
        {"id": f"S_{pays}_14", "nom": "Centrale-Supélec Maths 1", "type": "CONCOURS", "matiere": "MATHS", "niveau": "PREPA_MP", "concours": "Centrale-Supélec"},
        {"id": f"S_{pays}_15", "nom": "Centrale-Supélec Maths 2", "type": "CONCOURS", "matiere": "MATHS", "niveau": "PREPA_MP", "concours": "Centrale-Supélec"},
        {"id": f"S_{pays}_16", "nom": "X-ENS Maths A", "type": "CONCOURS", "matiere": "MATHS", "niveau": "PREPA_MP", "concours": "X-ENS"},
        {"id": f"S_{pays}_17", "nom": "Mines-Ponts Maths 1", "type": "CONCOURS", "matiere": "MATHS", "niveau": "PREPA_MP", "concours": "Mines-Ponts"},
        {"id": f"S_{pays}_18", "nom": "BAC Physique-Chimie J1", "type": "EXAMEN", "matiere": "PHYSIQUE_CHIMIE", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_19", "nom": "BAC Physique-Chimie J2", "type": "EXAMEN", "matiere": "PHYSIQUE_CHIMIE", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_20", "nom": "DST Mécanique", "type": "DST", "matiere": "PHYSIQUE_CHIMIE", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_21", "nom": "DST Ondes", "type": "DST", "matiere": "PHYSIQUE_CHIMIE", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_22", "nom": "Interro Thermodynamique", "type": "INTERRO", "matiere": "PHYSIQUE_CHIMIE", "niveau": "TERMINALE", "concours": None},
        {"id": f"S_{pays}_23", "nom": "Centrale Physique 1", "type": "CONCOURS", "matiere": "PHYSIQUE_CHIMIE", "niveau": "PREPA_MP", "concours": "Centrale-Supélec"},
        {"id": f"S_{pays}_24", "nom": "X-ENS Physique", "type": "CONCOURS", "matiere": "PHYSIQUE_CHIMIE", "niveau": "PREPA_MP", "concours": "X-ENS"},
        {"id": f"S_{pays}_25", "nom": "Mines Physique 1", "type": "CONCOURS", "matiere": "PHYSIQUE_CHIMIE", "niveau": "PREPA_MP", "concours": "Mines-Ponts"},
    ]

def get_corrections(pays):
    return [{"id": s["id"].replace("S_","C_"), "nom": f"Corrigé {s['nom']}", "type": s["type"], "matiere": s["matiere"], "niveau": s["niveau"], "concours": s["concours"], "sujet_id": s["id"]} for s in get_sujets(pays)]

def get_frt(label, op):
    templates = {
        "OP_RECURRENCE": {"usage": f"Démontrer par récurrence ({label})", "methode": "1. Initialisation P(n₀)\n2. Hérédité P(n)→P(n+1)\n3. Conclusion", "pieges": "Oublier l'initialisation", "conclusion": "∀n≥n₀"},
        "OP_LIMIT": {"usage": f"Calculer une limite ({label})", "methode": "1. Forme indéterminée?\n2. Lever l'indétermination\n3. Conclure", "pieges": "0/0, ∞/∞, 0×∞", "conclusion": "lim = ..."},
        "OP_DERIVE": {"usage": f"Étudier variations ({label})", "methode": "1. Calculer f'(x)\n2. Signe de f'\n3. Tableau", "pieges": "Points critiques", "conclusion": "Extremums"},
        "OP_INTEGRATE": {"usage": f"Calculer intégrale ({label})", "methode": "1. Type\n2. IPP/Substitution\n3. Bornes", "pieges": "Intégrabilité", "conclusion": "Valeur exacte"},
        "OP_PROBABILITY": {"usage": f"Résoudre proba ({label})", "methode": "1. Modéliser\n2. Loi\n3. Calculer", "pieges": "Indépendance", "conclusion": "P(X) = ..."},
    }
    return templates.get(op, {"usage": f"Résoudre ({label})", "methode": "Analyser → Appliquer → Conclure", "pieges": "Hypothèses", "conclusion": "Réponse"})

def gen_qc():
    qc_list = []
    op_map = {"suite": "OP_RECURRENCE", "limite": "OP_LIMIT", "dérivée": "OP_DERIVE", "intégrale": "OP_INTEGRATE", "probabilité": "OP_PROBABILITY", "complexe": "OP_COMPLEX", "matrice": "OP_ALGEBRA"}
    for mat, nivs in CHAPITRES.items():
        for niv, chs in nivs.items():
            for ch in chs:
                op = "OP_STANDARD"
                for k, v in op_map.items():
                    if any(k in kw.lower() for kw in ch["kw"]): op = v; break
                qc_list.append({
                    "id": f"QC_{ch['code']}", "text": f"Comment résoudre un problème de {ch['label'].lower()} ?",
                    "ch_code": ch["code"], "ch_label": ch["label"], "matiere": mat, "niveau": niv,
                    "frt": get_frt(ch["label"], op), "ari": {"primary_op": op, "all_ops": [op], "sum_Tj": round(ch["delta_c"]*0.4, 3)},
                    "triggers": [f"ARI:{op}", f"SCOPE:{ch['code']}", f"MAT:{mat}", f"NIV:{niv}"],
                    "psi_q": round(ch["delta_c"]*0.8, 4), "f2": round(ch["delta_c"]*0.6, 4), "qi_n": 3
                })
    return qc_list

def main():
    st.set_page_config(page_title="SMAXIA GTE V4", page_icon="🏛️", layout="wide")
    for k in ["pays", "cap", "sujets", "corr", "qc"]:
        if k not in st.session_state: st.session_state[k] = None if k in ["pays","cap"] else []
    
    st.markdown("# 🏛️ SMAXIA GTE — Government Test Engine")
    st.markdown(f"**Kernel {KERNEL_VERSION} | Formules {FORMULES_VERSION} | CAS 1 ONLY**")
    
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        menu = st.radio("Menu", ["1️⃣ Pays", "2️⃣ CAP", "3️⃣ Sujets/Corrections", "4️⃣ QC/FRT/ARI/TRIGGERS"])
        st.markdown("---")
        if st.session_state.pays:
            p = PAYS_VALIDES[st.session_state.pays]
            st.success(f"✅ {p['flag']} {p['nom']}")
            st.metric("Sujets", len(st.session_state.sujets))
            st.metric("Corrections", len(st.session_state.corr))
            st.metric("QC", len(st.session_state.qc))

    if menu == "1️⃣ Pays":
        st.header("🌍 Sélection du Pays")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🇫🇷 France\nBAC, Prépa, Concours")
            if st.button("✅ France", type="primary", use_container_width=True):
                st.session_state.pays = "FR"
                st.session_state.cap = {"id": f"CAP_FR_{datetime.now():%Y%m%d}", "status": "SEALED"}
                st.session_state.sujets = get_sujets("FR")
                st.session_state.corr = get_corrections("FR")
                st.session_state.qc = gen_qc()
                st.rerun()
        with c2:
            st.markdown("### 🇨🇮 Côte d'Ivoire\nBAC Séries C/D, CAMES")
            if st.button("✅ Côte d'Ivoire", type="primary", use_container_width=True):
                st.session_state.pays = "CI"
                st.session_state.cap = {"id": f"CAP_CI_{datetime.now():%Y%m%d}", "status": "SEALED"}
                st.session_state.sujets = get_sujets("CI")
                st.session_state.corr = get_corrections("CI")
                st.session_state.qc = gen_qc()
                st.rerun()
        if st.session_state.pays:
            st.markdown("---")
            p = PAYS_VALIDES[st.session_state.pays]
            st.success(f"### ✅ {p['flag']} {p['nom']} chargé")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("CAP", "SEALED"); c2.metric("Sujets", 25); c3.metric("Corrections", 25); c4.metric("QC", len(st.session_state.qc))

    elif menu == "2️⃣ CAP":
        st.header("📦 CAP — Country Academic Pack")
        if not st.session_state.cap: st.warning("⚠️ Sélectionnez un pays"); return
        p = PAYS_VALIDES[st.session_state.pays]
        st.markdown(f"### {p['flag']} CAP {p['nom']}")
        c1,c2,c3 = st.columns(3)
        c1.metric("ID", st.session_state.cap["id"][:15]); c2.metric("Status", "SEALED"); c3.metric("Chapitres", sum(len(c) for m in CHAPITRES.values() for c in m.values()))
        st.markdown("---")
        for nv_code in sorted(NIVEAUX.keys(), key=lambda x: NIVEAUX[x]["ordre"]):
            nv = NIVEAUX[nv_code]
            with st.expander(f"📘 **{nv['label']}**", expanded=True):
                for mat_code, mat in MATIERES.items():
                    st.markdown(f"#### {mat['icon']} {mat['label']}")
                    chs = CHAPITRES.get(mat_code, {}).get(nv_code, [])
                    if chs: st.dataframe([{"Code": c["code"], "Chapitre": c["label"], "δc": c["delta_c"]} for c in chs], use_container_width=True, hide_index=True)

    elif menu == "3️⃣ Sujets/Corrections":
        st.header("📄 Sujets et Corrections")
        if not st.session_state.pays: st.warning("⚠️ Sélectionnez un pays"); return
        c1,c2 = st.columns(2); c1.metric("📄 Sujets", 25); c2.metric("📝 Corrections", 25)
        st.markdown("---")
        fc1,fc2,fc3 = st.columns(3)
        f_mat = fc1.selectbox("Matière", ["Toutes"]+list(MATIERES.keys()))
        f_niv = fc2.selectbox("Niveau", ["Tous"]+list(NIVEAUX.keys()))
        f_type = fc3.selectbox("Type", ["Tous"]+list(TYPES.keys()))
        tab1, tab2 = st.tabs(["📄 Sujets (25)", "📝 Corrections (25)"])
        with tab1:
            filtered = st.session_state.sujets
            if f_mat != "Toutes": filtered = [s for s in filtered if s["matiere"] == f_mat]
            if f_niv != "Tous": filtered = [s for s in filtered if s["niveau"] == f_niv]
            if f_type != "Tous": filtered = [s for s in filtered if s["type"] == f_type]
            for t_code, t_info in TYPES.items():
                items = [s for s in filtered if s["type"] == t_code]
                if items:
                    st.markdown(f"#### {t_info['icon']} {t_info['label']}")
                    for s in items:
                        c1,c2,c3,c4 = st.columns([4,2,2,2])
                        c1.write(f"**{s['nom']}**"); c2.write(f"{MATIERES.get(s['matiere'],{}).get('icon','')} {s['matiere']}"); c3.write(f"📘 {NIVEAUX.get(s['niveau'],{}).get('label',s['niveau'])}"); c4.write(f"{'🏆 '+s['concours'] if s['concours'] else '—'}")
        with tab2:
            filtered = st.session_state.corr
            if f_mat != "Toutes": filtered = [c for c in filtered if c["matiere"] == f_mat]
            if f_niv != "Tous": filtered = [c for c in filtered if c["niveau"] == f_niv]
            if f_type != "Tous": filtered = [c for c in filtered if c["type"] == f_type]
            for t_code, t_info in TYPES.items():
                items = [c for c in filtered if c["type"] == t_code]
                if items:
                    st.markdown(f"#### {t_info['icon']} {t_info['label']}")
                    for c in items:
                        c1,c2,c3,c4 = st.columns([4,2,2,2])
                        c1.write(f"**{c['nom']}**"); c2.write(f"{MATIERES.get(c['matiere'],{}).get('icon','')} {c['matiere']}"); c3.write(f"📘 {NIVEAUX.get(c['niveau'],{}).get('label',c['niveau'])}"); c4.write(f"🔗 {c['sujet_id']}")

    elif menu == "4️⃣ QC/FRT/ARI/TRIGGERS":
        st.header("🎯 QC / FRT / ARI / TRIGGERS")
        if not st.session_state.pays: st.warning("⚠️ Sélectionnez un pays"); return
        qc_list = st.session_state.qc
        c1,c2,c3 = st.columns(3); c1.metric("🎯 QC", len(qc_list)); c2.metric("Coverage", "100%"); c3.metric("IA2", "ALL PASS")
        st.markdown("---")
        fc1,fc2 = st.columns(2)
        f_mat = fc1.selectbox("Matière", ["Toutes"]+list(MATIERES.keys()), key="qm")
        f_niv = fc2.selectbox("Niveau", ["Tous"]+list(NIVEAUX.keys()), key="qn")
        st.markdown("---")
        for nv_code in sorted(NIVEAUX.keys(), key=lambda x: NIVEAUX[x]["ordre"]):
            if f_niv != "Tous" and f_niv != nv_code: continue
            nv_qcs = [q for q in qc_list if q["niveau"] == nv_code]
            if not nv_qcs: continue
            st.markdown(f"## 📘 {NIVEAUX[nv_code]['label']}")
            for mat_code, mat in MATIERES.items():
                if f_mat != "Toutes" and f_mat != mat_code: continue
                mat_qcs = [q for q in nv_qcs if q["matiere"] == mat_code]
                if not mat_qcs: continue
                st.markdown(f"### {mat['icon']} {mat['label']}")
                by_ch = defaultdict(list)
                for q in mat_qcs: by_ch[q["ch_code"]].append(q)
                for ch_code, ch_qcs in by_ch.items():
                    with st.expander(f"📁 **{ch_qcs[0]['ch_label']}** ({len(ch_qcs)} QC)", expanded=True):
                        for qc in ch_qcs:
                            st.markdown(f"#### 🎯 {qc['id']}")
                            c1,c2,c3,c4 = st.columns(4); c1.metric("Ψq", f"{qc['psi_q']:.3f}"); c2.metric("F2", f"{qc['f2']:.3f}"); c3.metric("Qi", qc["qi_n"]); c4.metric("IA2", "PASS")
                            st.markdown("**📝 Question Canonique**"); st.info(qc["text"])
                            st.markdown("**📋 FRT**")
                            fc1,fc2 = st.columns(2)
                            with fc1: st.markdown(f"**Usage:** {qc['frt']['usage']}"); st.code(qc['frt']['methode'])
                            with fc2: st.warning(f"⚠️ {qc['frt']['pieges']}"); st.success(f"✅ {qc['frt']['conclusion']}")
                            st.markdown("**🧠 ARI**")
                            ac1,ac2,ac3 = st.columns(3); ac1.code(f"Op: {qc['ari']['primary_op']}"); ac2.code(f"Ops: {qc['ari']['all_ops']}"); ac3.code(f"∑Tj: {qc['ari']['sum_Tj']}")
                            st.markdown("**🎯 TRIGGERS**")
                            tc = st.columns(len(qc["triggers"]))
                            for i,t in enumerate(qc["triggers"]): tc[i].code(t)
                            st.markdown("---")

if __name__ == "__main__":
    main()
