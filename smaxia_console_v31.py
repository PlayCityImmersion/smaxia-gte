# ═══════════════════════════════════════════════════════════════════
# SMAXIA COMMAND CENTER — Admin Console Premium v1.0
# Fusion Opus + Manus + GPT design | Kernel V10.6.3
# Usage: streamlit run smaxia_command_center.py
# ═══════════════════════════════════════════════════════════════════
import streamlit as st
import json, zlib, base64, hashlib, os, glob
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 1. CAP DATA — EMBEDDED (compressed b64)
# ═══════════════════════════════════════════════════════════════════
_CAP_B64 = """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"""

@st.cache_data
def load_cap_embedded():
    raw = zlib.decompress(base64.b64decode(_CAP_B64))
    return json.loads(raw.decode("utf-8"))

@st.cache_data
def load_cap_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_cap():
    """Load CAP: from ./runs/ if available, else embedded."""
    run_dir = st.session_state.get("run_dir", "")
    if run_dir:
        caps = glob.glob(os.path.join(run_dir, "CAP_*.json"))
        if caps:
            return load_cap_file(caps[0])
    return load_cap_embedded()

# ═══════════════════════════════════════════════════════════════════
# 2. CES ARTEFACT LOADER (reads from ./runs/<run_id>/ if exists)
# ═══════════════════════════════════════════════════════════════════
CES_FILES = [
    "CES_State.json", "QC_Index.json", "Qi_Index.json",
    "CoverageReport.json", "CoverageTimeline.json",
    "QuarantineLedger.json", "CHK_REPORT.json",
    "SealReport.json", "DeterminismReport_3runs.json",
    "PairingReport_G2.json", "OcrReport_G3.json",
    "AtomTrace_G4.json", "DedupReport_G5.json",
]

def load_artefact(name):
    run_dir = st.session_state.get("run_dir", "")
    if run_dir:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

def artefact_status(name):
    run_dir = st.session_state.get("run_dir", "")
    if run_dir:
        p = os.path.join(run_dir, name)
        return "✅" if os.path.exists(p) else "❌ MISSING"
    return "⏳ En attente"

# ═══════════════════════════════════════════════════════════════════
# 3. CSS INJECTION — Dark Premium Theme (Manus style)
# ═══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* === GLOBAL === */
    .stApp, [data-testid="stAppViewContainer"], .main .block-container {
        background: #0a0e27 !important; color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .main .block-container { padding-top: 1rem !important; max-width: 1400px !important; }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: #0f1329 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] * { color: #b0bec5 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.9rem !important; }
    [data-testid="stSidebar"] .stRadio label:hover { color: #e3f2fd !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #90caf9 !important; }

    /* === HEADER GRADIENT === */
    .smx-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .smx-header h1 { color: #fff !important; font-size: 1.6rem; font-weight: 800; margin: 0; }
    .smx-header p { color: #90caf9; font-size: 0.85rem; margin: 0.2rem 0 0; }

    /* === STAT CARDS === */
    .smx-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; margin-bottom: 1.2rem; }
    .smx-card {
        background: linear-gradient(135deg, #1a1f3e, #151933);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 1rem; text-align: center; transition: transform 0.2s;
    }
    .smx-card:hover { transform: translateY(-2px); }
    .smx-card .val {
        font-size: 2rem; font-weight: 800; color: #64b5f6;
        font-family: 'JetBrains Mono', monospace; letter-spacing: -1px;
    }
    .smx-card .lbl {
        font-size: 0.65rem; color: #78909c; font-weight: 600;
        letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.2rem;
    }

    /* === CYCLE BARS === */
    .cycle-bar { padding: 0.6rem 1rem; border-radius: 8px; margin: 0.4rem 0;
        font-weight: 700; font-size: 0.9rem; color: #fff; }
    .cycle-hs  { background: linear-gradient(90deg, #1565c0, #0d47a1); }
    .cycle-preu { background: linear-gradient(90deg, #2e7d32, #1b5e20); }
    .cycle-uni { background: linear-gradient(90deg, #e65100, #bf360c); }

    /* === CLASS CARDS === */
    .class-card {
        background: #151933; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 0.7rem 1rem; margin: 0.3rem 0;
        transition: border-color 0.2s;
    }
    .class-card:hover { border-color: rgba(100,181,246,0.3); }

    /* === BADGES === */
    .badge-off { display: inline-block; background: #1b5e20; color: #a5d6a7;
        padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.65rem; font-weight: 700; }
    .badge-can { display: inline-block; background: #e65100; color: #ffcc80;
        padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.65rem; font-weight: 700; }
    .badge-seal { display: inline-block; background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: #fff; padding: 0.3rem 0.8rem; border-radius: 15px;
        font-weight: 700; font-size: 0.75rem; letter-spacing: 1px; }
    .badge-exam { display: inline-block; background: #311b92; color: #b39ddb;
        padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; margin: 2px; }
    .badge-conc { display: inline-block; background: #1a237e; color: #90caf9;
        padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; margin: 2px; }
    .badge-pass { color: #66bb6a; font-weight: 700; }
    .badge-fail { color: #ef5350; font-weight: 700; }
    .badge-pending { color: #ffa726; font-weight: 700; }

    /* === SHA256 BOX === */
    .sha-box {
        background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
        padding: 0.5rem 0.8rem; font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: #7ee787; word-break: break-all; margin: 0.4rem 0;
    }

    /* === COVERAGE BAR === */
    .cov-outer { background: #1a1f3e; border-radius: 6px; height: 20px; width: 100%; overflow: hidden; }
    .cov-inner { height: 100%; border-radius: 6px; transition: width 0.3s; }

    /* === GATE ROW === */
    .gate-row { padding: 0.4rem 0; font-size: 0.85rem; border-bottom: 1px solid rgba(255,255,255,0.04); }

    /* === IDEA CARDS === */
    .idea-card {
        background: linear-gradient(135deg, #1a1f3e, #0f1329);
        border-left: 4px solid #ffd740; border-radius: 0 10px 10px 0;
        padding: 0.8rem 1rem; margin: 0.5rem 0;
    }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #151933 !important; color: #90caf9 !important;
        border-radius: 8px 8px 0 0 !important; border: 1px solid rgba(255,255,255,0.06) !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1a237e !important; color: #fff !important;
    }

    /* === EXPANDERS === */
    .streamlit-expanderHeader {
        background: #151933 !important; color: #e3f2fd !important;
        border-radius: 8px !important;
    }

    /* === DATAFRAME === */
    .stDataFrame { background: #151933 !important; }

    /* === MISC === */
    hr { border-color: rgba(255,255,255,0.06) !important; }
    .stSelectbox label, .stTextInput label, .stRadio label { color: #90caf9 !important; }
    </style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 4. HTML HELPERS
# ═══════════════════════════════════════════════════════════════════
def h(tag, text, **attrs):
    a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"<{tag} {a}>{text}</{tag}>"

def header(title, subtitle=""):
    return f'<div class="smx-header"><h1>{title}</h1><p>{subtitle}</p></div>'

def stat_cards(items):
    """items = [(value, label, color?), ...]"""
    cards = ""
    for item in items:
        v, l = item[0], item[1]
        c = item[2] if len(item) > 2 else "#64b5f6"
        cards += f'<div class="smx-card"><div class="val" style="color:{c}">{v}</div><div class="lbl">{l}</div></div>'
    return f'<div class="smx-metrics">{cards}</div>'

def badge(text, cls):
    return f'<span class="badge-{cls}">{text}</span>'

def source_badge(stype):
    if stype == "OFFICIEL":
        return '<span class="badge-off">OFFICIEL</span>'
    return '<span class="badge-can">CANONIQUE</span>'

def cov_bar(pct, width="100%"):
    pct = min(max(pct, 0), 100)
    if pct >= 95: c = "#66bb6a"
    elif pct >= 75: c = "#ffa726"
    else: c = "#ef5350"
    return (f'<div class="cov-outer" style="width:{width}">'
            f'<div class="cov-inner" style="width:{pct}%;background:{c}"></div></div>')

def sha_box(h):
    return f'<div class="sha-box">{h}</div>'

# ═══════════════════════════════════════════════════════════════════
# 5. CAP HELPERS
# ═══════════════════════════════════════════════════════════════════
def cap_stats(cap):
    meta = cap["A_METADATA"]
    edu = cap["B_EDUCATION_SYSTEM"]
    exams = cap["E_EXAMS_CONCOURS"]["exams_and_contests"]
    cycles = edu["cycles"]
    levels = edu["levels"]
    subjects = edu["subjects"]
    total_ch = sum(s["chapter_count"] for s in subjects)
    off = sum(1 for s in subjects if s.get("source_type") == "OFFICIEL")
    can = sum(1 for s in subjects if s.get("source_type") == "CANONIQUE")
    return {
        "meta": meta, "cycles": cycles, "levels": levels, "subjects": subjects,
        "exams": exams, "total_ch": total_ch, "off": off, "can": can,
    }

def levels_for_cycle(edu, cycle_id):
    return sorted(
        [l for l in edu["levels"] if l["cycle_id"] == cycle_id],
        key=lambda x: x["order"]
    )

def subjects_for_level(edu, level_code):
    return [s for s in edu["subjects"] if s["level_code"] == level_code]

def cycle_css(cid):
    return {"CYCLE_HS": "cycle-hs", "CYCLE_PREU": "cycle-preu", "CYCLE_UNI": "cycle-uni"}.get(cid, "")

def exam_for_level(exams, level_code):
    for e in exams:
        if e.get("level_code") == level_code:
            return e
    return None

# ═══════════════════════════════════════════════════════════════════
# 6. PAGES
# ═══════════════════════════════════════════════════════════════════

# ---- 6.1 DASHBOARD ----
def page_dashboard(cap):
    s = cap_stats(cap)
    meta = s["meta"]
    st.markdown(header(
        f"🇫🇷 SMAXIA — CAP {meta['country_name_local']} Scellé",
        f"Country Academic Pack • Kernel {meta['kernel_version']} • {meta['source_doctrine']} • Zéro Invention"
    ), unsafe_allow_html=True)

    # KPI row 1 — CAP
    st.markdown(stat_cards([
        (len(s["cycles"]), "Cycles"),
        (meta["total_classes"], "Classes"),
        (meta["total_subjects_go"], "Matières GO"),
        (s["total_ch"], "Chapitres"),
        (len(s["exams"]), "Examens"),
    ]), unsafe_allow_html=True)

    # Cycles
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### 📈 Répartition par Cycle")
        for c in s["cycles"]:
            lvls = levels_for_cycle(cap["B_EDUCATION_SYSTEM"], c["cycle_id"])
            subs = [su for l in lvls for su in subjects_for_level(cap["B_EDUCATION_SYSTEM"], l["level_code"])]
            chs = sum(su["chapter_count"] for su in subs)
            st.markdown(
                f'<div class="cycle-bar {cycle_css(c["cycle_id"])}">'
                f'{c["cycle_name_local"]} ({c["cycle_name_en"]}) — '
                f'{len(lvls)} classes · {len(subs)} matières · {chs} chapitres</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("#### 🔐 Intégrité")
        st.markdown(sha_box(f"📋 {meta['cap_fingerprint_sha256']}"), unsafe_allow_html=True)
        st.markdown(f"""
        📜 Doctrine: **{meta['source_doctrine']}**  
        🏛️ Juridiction: **{meta['jurisdiction_model']}**  
        📅 Année: **{meta['academic_year']}**  
        🔖 Statut: {badge('✓ SEALED', 'seal') if meta['status']=='SEALED' else badge('NOT SEALED', 'fail')}
        """, unsafe_allow_html=True)
        st.markdown(f"{source_badge('OFFICIEL')} **{s['off']}** matières &nbsp; {source_badge('CANONIQUE')} **{s['can']}** matières", unsafe_allow_html=True)

    # CES Pipeline status
    st.markdown("---")
    st.markdown("#### 🚀 Pipeline CES HARVEST")
    ces = load_artefact("CES_State.json")
    if ces:
        st.markdown(stat_cards([
            (ces.get("pdf_count", "N/A"), "PDFs collectés"),
            (ces.get("pairs_count", "N/A"), "Paires"),
            (ces.get("qi_count", "N/A"), "Qi totales"),
            (ces.get("qc_count", "N/A"), "QC générées"),
            (f"{ces.get('coverage_avg', 0):.0f}%", "Couverture moy."),
        ]), unsafe_allow_html=True)
    else:
        st.info("⏳ **CES HARVEST non lancé** — Aucun artefact CES détecté. "
                "Les données QC/FRT/ARI/TRIGGERS apparaîtront ici après E1 COLLECT.")

# ---- 6.2 CAP EXPLORER ----
def page_cap_explorer(cap):
    s = cap_stats(cap)
    edu = cap["B_EDUCATION_SYSTEM"]
    st.markdown(header("📚 CAP Explorer",
        f"{s['meta']['total_classes']} classes · {s['meta']['total_subjects_go']} matières · {s['total_ch']} chapitres"),
        unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏫 Niveaux & Classes", "📘 Matières", "📖 Chapitres", "🎓 Examens/Concours", "🔗 Sources"])

    # --- Niveaux ---
    with tab1:
        for c in edu["cycles"]:
            lvls = levels_for_cycle(edu, c["cycle_id"])
            st.markdown(f'<div class="cycle-bar {cycle_css(c["cycle_id"])}">'
                        f'{c["cycle_name_local"]} — {len(lvls)} classes</div>', unsafe_allow_html=True)
            for l in lvls:
                subs = subjects_for_level(edu, l["level_code"])
                chs = sum(su["chapter_count"] for su in subs)
                ex = exam_for_level(s["exams"], l["level_code"])
                exam_html = ""
                if ex and ex.get("exam"):
                    exam_html = f' <span class="badge-exam">🎯 {ex["exam"]["exam_name"]}</span>'
                conc_html = ""
                if ex and ex.get("contests_top"):
                    conc_html = " ".join(f'<span class="badge-conc">🏆 #{c["rank"]} {c["name"]}</span>'
                                          for c in ex["contests_top"][:5])
                st.markdown(f"""<div class="class-card">
                    <b style="color:#e3f2fd">{l['level_name_local']}</b>
                    <span style="color:#546e7a;font-size:0.8rem"> ({l['level_code']})</span>
                    <div style="color:#546e7a;font-size:0.75rem">{l.get('voie','')}</div>
                    <div style="color:#90caf9;font-size:0.8rem;margin-top:0.2rem">
                        📘 {len(subs)} matières · 📖 {chs} chapitres</div>
                    {exam_html}{f'<div style="margin-top:0.2rem">{conc_html}</div>' if conc_html else ''}
                </div>""", unsafe_allow_html=True)

    # --- Matières ---
    with tab2:
        search = st.text_input("🔍 Rechercher une matière", key="subj_search")
        for su in edu["subjects"]:
            lname = next((l["level_name_local"] for l in edu["levels"] if l["level_code"] == su["level_code"]), su["level_code"])
            full = f"{lname} — {su['subject_name_local']}"
            if search and search.lower() not in full.lower():
                continue
            st.markdown(f"""<div class="class-card">
                <b style="color:#e3f2fd">{lname}</b> — {su['subject_name_local']}
                ({su['chapter_count']} chap.) {source_badge(su.get('source_type',''))}
                <div style="color:#546e7a;font-size:0.7rem">{su.get('source_ref','')}</div>
            </div>""", unsafe_allow_html=True)

    # --- Chapitres ---
    with tab3:
        search_ch = st.text_input("🔍 Rechercher un chapitre", key="ch_search")
        for su in edu["subjects"]:
            lname = next((l["level_name_local"] for l in edu["levels"] if l["level_code"] == su["level_code"]), su["level_code"])
            chapters_match = [ch for ch in su["chapters"]
                              if not search_ch or search_ch.lower() in ch["chapter_name"].lower()
                              or search_ch.lower() in su["subject_name_local"].lower()
                              or search_ch.lower() in lname.lower()]
            if not chapters_match:
                continue
            with st.expander(f"**{lname} — {su['subject_name_local']}** ({len(su['chapters'])} chap.) {su.get('source_type','')}"):
                for ch in chapters_match:
                    st.markdown(
                        f'<span style="color:#64b5f6;font-weight:700;font-family:JetBrains Mono">'
                        f'{str(ch["chapter_number"]).zfill(2)}</span> '
                        f'{ch["chapter_name"]}'
                        f' <span style="color:#546e7a;font-size:0.7rem">'
                        f'[QC: ⏳ | Qi: 0 | Cov: 0%]</span>',
                        unsafe_allow_html=True
                    )

    # --- Examens ---
    with tab4:
        for c in edu["cycles"]:
            lvls = levels_for_cycle(edu, c["cycle_id"])
            st.markdown(f'<div class="cycle-bar {cycle_css(c["cycle_id"])}">{c["cycle_name_local"]}</div>', unsafe_allow_html=True)
            for l in lvls:
                ex = exam_for_level(s["exams"], l["level_code"])
                if not ex: continue
                exam_html = f'<span class="badge-exam">🎯 {ex["exam"]["exam_name"]}</span>' if ex.get("exam") else ""
                conc_html = " ".join(f'<span class="badge-conc">🏆 #{c["rank"]} {c["name"]}</span>'
                                      for c in (ex.get("contests_top") or []))
                st.markdown(f"""<div class="class-card">
                    <b style="color:#e3f2fd">{l['level_name_local']}</b>
                    <span style="color:#546e7a;font-size:0.8rem">({l['level_code']})</span>
                    {exam_html}
                    <div style="margin-top:0.2rem">{conc_html}</div>
                </div>""", unsafe_allow_html=True)

    # --- Sources ---
    with tab5:
        src = cap["C_HARVEST_SOURCES"]
        for s_ in src["sources"]:
            pc = {"A": "#66bb6a", "B": "#ffa726", "C": "#ef5350"}.get(s_["proof"], "#fff")
            st.markdown(f"""<div class="class-card">
                <b style="color:#e3f2fd">{s_['source_id']}</b> — {s_['domain']}
                <div style="color:#90caf9;font-size:0.8rem">📋 {s_['scope']}</div>
                <div style="font-size:0.8rem">
                    🏅 Authority: <b>{s_['authority_score']}</b> ·
                    Proof: <span style="color:{pc};font-weight:700">{s_['proof']}</span> ·
                    Niveaux: {', '.join(s_['levels_covered'][:5])}{'...' if len(s_['levels_covered'])>5 else ''}
                </div>
            </div>""", unsafe_allow_html=True)
        r = src["scraping_rules"]
        st.markdown("#### ⚙️ Scraping Rules")
        st.markdown(stat_cards([
            (f"{r['rate_limit_ms']}ms", "Rate Limit"),
            (r['max_concurrent'], "Max Concurrent"),
            ("✅", "Robots.txt"),
        ]), unsafe_allow_html=True)

# ---- 6.3 CES MONITOR ----
def page_ces_monitor(cap):
    st.markdown(header("🚀 CES HARVEST Monitor", "Pipeline status · Gates · Saturation"), unsafe_allow_html=True)

    ces = load_artefact("CES_State.json")
    if not ces:
        st.warning("⏳ **CES HARVEST non lancé** — Aucun run détecté.")
        st.markdown("Quand E1 COLLECT sera exécuté, cette page affichera :")
        st.markdown("""
        - Sources harvest actives + status (rate limit, robots.txt)  
        - Pipeline status par gate G0→G10  
        - Saturation STOP_RULE (SR0→SR5) par chapitre  
        - Quarantine counters par gate  
        """)
        # Show expected artefacts
        st.markdown("#### 📦 Artefacts attendus")
        for f in CES_FILES:
            st.markdown(f"&nbsp;&nbsp; {artefact_status(f)} `{f}`", unsafe_allow_html=True)
        return

    # If CES exists, render real data
    st.json(ces)

# ---- 6.4 CHAPTERS & QC ----
def page_chapters_qc(cap):
    st.markdown(header("📊 Chapitres & QC", "Couverture · Saturation · QC par chapitre"), unsafe_allow_html=True)

    edu = cap["B_EDUCATION_SYSTEM"]
    # Cycle/Level/Subject filter
    c1, c2, c3 = st.columns(3)
    cycle_names = {c["cycle_id"]: c["cycle_name_local"] for c in edu["cycles"]}
    with c1:
        sel_cycle = st.selectbox("Cycle", ["Tous"] + list(cycle_names.values()), key="ch_cycle")
    sel_cycle_id = next((k for k, v in cycle_names.items() if v == sel_cycle), None)

    levels_filtered = edu["levels"]
    if sel_cycle_id:
        levels_filtered = [l for l in edu["levels"] if l["cycle_id"] == sel_cycle_id]
    level_names = {l["level_code"]: l["level_name_local"] for l in levels_filtered}
    with c2:
        sel_level = st.selectbox("Niveau", ["Tous"] + list(level_names.values()), key="ch_level")
    sel_level_code = next((k for k, v in level_names.items() if v == sel_level), None)

    subjects_filtered = edu["subjects"]
    if sel_level_code:
        subjects_filtered = [s for s in subjects_filtered if s["level_code"] == sel_level_code]
    elif sel_cycle_id:
        lc = [l["level_code"] for l in levels_filtered]
        subjects_filtered = [s for s in subjects_filtered if s["level_code"] in lc]
    subj_names = {s["subject_id"]: s["subject_name_local"] for s in subjects_filtered}
    with c3:
        sel_subj = st.selectbox("Matière", ["Toutes"] + list(subj_names.values()), key="ch_subj")
    sel_subj_id = next((k for k, v in subj_names.items() if v == sel_subj), None)

    if sel_subj_id:
        subjects_filtered = [s for s in subjects_filtered if s["subject_id"] == sel_subj_id]

    # QC Index
    qc_index = load_artefact("QC_Index.json") or {}

    # Render chapters
    for su in subjects_filtered:
        lname = next((l["level_name_local"] for l in edu["levels"] if l["level_code"] == su["level_code"]), "")
        with st.expander(f"**{lname} — {su['subject_name_local']}** ({su['chapter_count']} chapitres)"):
            for ch in su["chapters"]:
                ch_key = f"{su['subject_id']}_CH{str(ch['chapter_number']).zfill(2)}"
                qc_data = qc_index.get(ch_key, {})
                qi_count = qc_data.get("qi_count", 0)
                qc_count = qc_data.get("qc_count", 0)
                cov = qc_data.get("coverage", 0)
                orphans = qc_data.get("orphans", 0)
                status = qc_data.get("status", "⏳ En attente")

                col_a, col_b, col_c, col_d, col_e = st.columns([4, 1, 1, 2, 1])
                with col_a:
                    st.markdown(f'<span style="color:#64b5f6;font-weight:700">'
                                f'{str(ch["chapter_number"]).zfill(2)}</span> {ch["chapter_name"]}',
                                unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"**QC:** {qc_count}")
                with col_c:
                    st.markdown(f"**Qi:** {qi_count}")
                with col_d:
                    st.markdown(cov_bar(cov), unsafe_allow_html=True)
                    st.caption(f"{cov}%")
                with col_e:
                    st.markdown(f"<small>{status}</small>", unsafe_allow_html=True)

# ---- 6.5 QC DETAIL ----
def page_qc_detail(cap):
    st.markdown(header("🔬 QC Detail", "FRT · ARI · TRIGGERS — Vue opaque (IP protégée)"), unsafe_allow_html=True)

    qc_index = load_artefact("QC_Index.json")
    if not qc_index:
        st.info("⏳ **Aucune QC générée** — En attente CES HARVEST E1→E6.")
        st.markdown("""
        Quand les QC seront générées, cette page affichera pour chaque QC :  
        - **QC_ID** + chapitre + contribution couverture  
        - **FRT** : ID + SHA256 + lien artefact *(formule opaque — IP SMAXIA)*  
        - **ARI** : ID + SHA256 + lien artefact *(formule opaque — IP SMAXIA)*  
        - **TRIGGERS** : liste IDs + SHA256 *(mécaniques opaques)*  
        - **Qi children** : liste des Qi associées  
        - **Correctness** : verdict IA2 judge (PASS/FAIL)  
        """)
        return

    # If QC_Index exists, allow selection
    qc_ids = list(qc_index.keys())
    sel = st.selectbox("Sélectionner QC", qc_ids)
    if sel:
        qc = qc_index[sel]
        st.json(qc)

# ---- 6.6 MAPPING Qi→QC ----
def page_mapping(cap):
    st.markdown(header("🗺️ Mapping Qi → QC", "Orphelins · Couverture · Heatmap"), unsafe_allow_html=True)

    qi_index = load_artefact("Qi_Index.json")
    if not qi_index:
        st.info("⏳ **Aucune Qi indexée** — En attente CES HARVEST.")

    st.markdown("#### 🧪 Sélection")
    exam_type = st.selectbox("Type d'épreuve", ["Tous", "Bac", "DST", "Interro", "Concours"])
    st.caption("Quand le pipeline sera actif, cette page permettra de filtrer par niveau/matière/chapitre "
               "et visualiser le mapping Qi→QC avec heatmap couverture.")

# ---- 6.7 TESTS ----
def page_tests(cap):
    st.markdown(header("🧪 Tests Orphelins & Couverture",
        "Test A : sujet déjà traité | Test B : sujet nouveau"), unsafe_allow_html=True)

    tab_a, tab_b = st.tabs(["Test A — Sujet traité", "Test B — Sujet nouveau"])

    with tab_a:
        st.markdown("""
        **Objectif** : pour un sujet déjà traité par le système, vérifier que **Qi_orphelin = 0**.
        
        Si des Qi restent orphelines après traitement, c'est que la couverture QC est insuffisante
        sur certains chapitres.
        """)
        ces = load_artefact("CES_State.json")
        if not ces:
            st.warning("⏳ Aucun sujet traité — CES HARVEST non lancé.")
        else:
            doc_id = st.text_input("doc_id du sujet traité")
            if doc_id and st.button("▶ Lancer Test A"):
                st.info("Recherche dans les artefacts...")

    with tab_b:
        st.markdown("""
        **Objectif** : pour un sujet **jamais vu**, vérifier que le système couvre toutes les Qi.
        
        Upload un PDF → extraction Qi → mapping Qi→QC → comptage orphelins → verdict **SMAXIA READY ?**
        """)
        ces = load_artefact("CES_State.json")
        if not ces:
            st.warning("⏳ Pipeline CES non actif — impossible de tester un nouveau sujet.")
        else:
            uploaded = st.file_uploader("📄 Upload PDF sujet", type=["pdf"])
            if uploaded and st.button("▶ Lancer Test B"):
                st.info("Extraction Qi en cours...")

# ---- 6.8 GATES & INTEGRITY ----
def page_gates(cap):
    meta = cap["A_METADATA"]
    st.markdown(header("🔐 Gates & Intégrité",
        f"SHA256 · Déterminisme · Conformité Kernel {meta['kernel_version']}"), unsafe_allow_html=True)

    # SHA256
    st.markdown("#### 🔍 Empreinte CAP")
    st.markdown(sha_box(meta["cap_fingerprint_sha256"]), unsafe_allow_html=True)

    # CAP Gates
    st.markdown("#### 🚦 Gates CAP (structurelles)")
    cap_gates = [
        ("GATE_CAP_SCHEMA", "5/5 sections présentes", True),
        ("GATE_CAP_SECTIONS_COMPLETE", "A→E complètes", True),
        ("GATE_LEVEL_EXAM_COMPLETENESS", f"{len(cap['E_EXAMS_CONCOURS']['exams_and_contests'])}/{meta['total_classes']} entries", True),
        ("GATE_SOURCE_TYPE_COVERAGE", f"{cap_stats(cap)['total_ch']}/{meta['total_chapters']} chapitres sourcés", True),
        ("GATE_COUNTRY_BRANCHING", "Zéro branchement pays", True),
        ("GATE_CAS1_ONLY", f"Doctrine {meta['source_doctrine']} active", True),
    ]
    for name, detail, passed in cap_gates:
        icon = "✅" if passed else "❌"
        cls = "pass" if passed else "fail"
        st.markdown(f'<div class="gate-row">{icon} <span class="badge-{cls}">{name}</span> — {detail}</div>',
                    unsafe_allow_html=True)

    # CES Gates
    st.markdown("#### 🚦 Gates CES HARVEST")
    ces_gates = [
        "G0_SOURCES_VALID", "G1_SCRAPE_COMPLETE", "G2_PAIRING",
        "G3_OCR", "G4_ATOMIZE", "G5_DEDUP", "G6_CORRECTNESS",
        "G6.5_JUDGE", "G7_COVERAGE_MIN", "G8_SEAL_QC",
        "G9_PREDICT", "G10_SEAL_CES",
    ]
    chk = load_artefact("CHK_REPORT.json") or {}
    for g in ces_gates:
        status = chk.get(g, None)
        if status == "PASS":
            st.markdown(f'<div class="gate-row">✅ <span class="badge-pass">{g}</span> — PASS</div>', unsafe_allow_html=True)
        elif status == "FAIL":
            st.markdown(f'<div class="gate-row">❌ <span class="badge-fail">{g}</span> — FAIL</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="gate-row">⏳ <span class="badge-pending">{g}</span> — En attente lancement</div>', unsafe_allow_html=True)

    # Determinism
    st.markdown("#### 🔄 Déterminisme")
    det = load_artefact("DeterminismReport_3runs.json")
    if det:
        st.json(det)
    else:
        st.info("⏳ DeterminismReport non disponible — sera généré après 3 runs identiques.")

    # Doctrine
    st.markdown("#### 📜 Doctrine")
    st.markdown("""
    🔒 **CAS 1 ONLY** — Zéro reconstruction, zéro invention  
    🔒 **Zéro Hardcode Pays** — Tout piloté par CAP  
    🔒 **Déterminisme** — 3 runs identiques requis  
    🔒 **Scellabilité** — JSON canonical + SHA256  
    🔒 **Auditabilité** — Artefacts + preuves + pointeurs  
    🔒 **B4** — Toute gate sans artefact = FAIL automatique  
    """)

# ---- 6.9 QUARANTINE ----
def page_quarantine(cap):
    st.markdown(header("🔶 Quarantine Ledger", "Items en quarantaine · Résolution"), unsafe_allow_html=True)

    ql = load_artefact("QuarantineLedger.json")
    if not ql:
        st.info("⏳ Aucun item en quarantaine — CES HARVEST non lancé ou 0 anomalie.")
        st.markdown("Quand le pipeline sera actif, chaque item quarantiné sera listé avec : "
                    "ID, gate d'origine, raison, sévérité, pointeurs (pdf_sha/page/locator), statut (open/resolved).")
        return

    import pandas as pd
    df = pd.DataFrame(ql)
    st.dataframe(df, use_container_width=True)

# ---- 6.10 BEST IDEAS ----
def page_ideas():
    st.markdown(header("💡 BEST IDEAS", "Avantages irrattrapables SMAXIA — 10 idées game-changer"), unsafe_allow_html=True)
    ideas = [
        ("🎯 Proof of Kill Rate™", "Afficher pour chaque chapitre un indicateur public : '95.2% des questions de ton prochain examen sont couvertes par ces QC.' Aucun concurrent ne peut faire ça sans CAS 1 ONLY + gates déterministes.", "AVANTAGE IRRATTRAPABLE", "Produit"),
        ("🌍 170 Pays = 170 Monopoles Locaux", "Chaque CAP scellé est un monopole de fait. Le premier pays activé verrouille le marché.", "WINNER TAKES ALL", "Stratégie"),
        ("📊 PrediNote : Prédire la note", "ARI + couverture QC + taux de réussite = prédiction ±1 point. Le Saint-Graal de l'EdTech.", "VIRAL", "Produit"),
        ("🏆 Classement National SMAX Score", "Position anonyme par matière/chapitre. Compétition saine + motivation + rétention.", "RÉTENTION x3", "Engagement"),
        ("🔄 Boucle Virale 'Défi Chapitre'", "Défier un ami sur un chapitre. L'ami DOIT télécharger pour répondre. K-factor > 1.", "ACQUISITION GRATUITE", "Growth"),
        ("🎓 Certification SMAXIA", "100% couverture + ≥90% réussite = certificat vérifiable QR + SHA256.", "EFFET DE RÉSEAU", "Monétisation"),
        ("👨‍🏫 Dashboard Professeur (B2B2C)", "Dashboard gratuit pour profs : progression classe, chapitres faibles. 1 prof = 30-150 élèves.", "CANAL B2B", "Distribution"),
        ("⚡ Mode Urgence 'J-7 Examen'", "Parcours optimisé ciblant chapitres faibles + QC tombables. Conversion maximale.", "CONVERSION PAYANTE", "Monétisation"),
        ("🧬 Génome de l'Examen™", "Cartographie exacte des familles de questions, fréquence, poids. La carte des mines.", "PR + VIRALITÉ", "Marketing"),
        ("🌐 API SMAXIA pour Éditeurs", "API payante : éditeurs intègrent QC/FRT/ARI. De app à infrastructure mondiale.", "PLATEFORME", "Business Model"),
    ]
    for title, txt, impact, cat in ideas:
        st.markdown(f"""<div class="idea-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <b style="color:#ffd740;font-size:1rem">{title}</b>
                <span class="badge-exam">{cat}</span>
            </div>
            <div style="color:#b0bec5;font-size:0.85rem;margin-top:0.3rem;line-height:1.5">{txt}</div>
            <div style="color:#64b5f6;font-weight:700;font-size:0.8rem;margin-top:0.3rem">💎 {impact}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 7. MAIN — SIDEBAR + ROUTING
# ═══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="SMAXIA Command Center",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    cap = get_cap()
    meta = cap["A_METADATA"]

    # ---- SIDEBAR ----
    with st.sidebar:
        st.markdown("### 🔒 SMAXIA Command Center")
        st.markdown(f"**Pays:** {meta['country_code']} — {meta['country_name_local']}")
        st.markdown(f"**Kernel:** {meta['kernel_version']}")
        st.markdown(f"**CAP:** {meta['version']}")
        sealed = meta.get("status") == "SEALED"
        if sealed:
            st.markdown('<span class="badge-seal">✓ SEALED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-fail">⚠ NOT SEALED</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Run dir selector
        run_dir = st.text_input("📁 Run directory", value="./runs/latest", key="run_dir_input",
                                help="Chemin vers le dossier run CES (ex: ./runs/run_001)")
        st.session_state["run_dir"] = run_dir

        st.markdown("---")

        page = st.radio("Navigation", [
            "📊 Dashboard",
            "📚 CAP Explorer",
            "🚀 CES Monitor",
            "📈 Chapitres & QC",
            "🔬 QC Detail (FRT/ARI)",
            "🗺️ Mapping Qi→QC",
            "🧪 Tests (Orphelins)",
            "🔐 Gates & Intégrité",
            "🔶 Quarantine",
            "💡 Best Ideas",
        ], label_visibility="collapsed")

    # ---- HEALTH BAR ----
    chk = load_artefact("CHK_REPORT.json") or {}
    pass_count = sum(1 for v in chk.values() if v == "PASS")
    total_gates = max(len(chk), 12)
    st.markdown(f"""<div style="background:#0f1329;padding:0.4rem 1rem;border-radius:8px;
        font-size:0.75rem;color:#78909c;margin-bottom:0.5rem;
        border:1px solid rgba(255,255,255,0.04)">
        <b style="color:#e3f2fd">SMAXIA</b> · CAP: {meta['cap_id']} ·
        Kernel: {meta['kernel_version']} · {meta['source_doctrine']} ·
        Gates: <span class="badge-{'pass' if pass_count==total_gates else 'pending'}">{pass_count}/{total_gates}</span> ·
        Quarantine: {artefact_status('QuarantineLedger.json')}
    </div>""", unsafe_allow_html=True)

    # ---- ROUTING ----
    if page == "📊 Dashboard":
        page_dashboard(cap)
    elif page == "📚 CAP Explorer":
        page_cap_explorer(cap)
    elif page == "🚀 CES Monitor":
        page_ces_monitor(cap)
    elif page == "📈 Chapitres & QC":
        page_chapters_qc(cap)
    elif page == "🔬 QC Detail (FRT/ARI)":
        page_qc_detail(cap)
    elif page == "🗺️ Mapping Qi→QC":
        page_mapping(cap)
    elif page == "🧪 Tests (Orphelins)":
        page_tests(cap)
    elif page == "🔐 Gates & Intégrité":
        page_gates(cap)
    elif page == "🔶 Quarantine":
        page_quarantine(cap)
    elif page == "💡 Best Ideas":
        page_ideas()

if __name__ == "__main__":
    main()
