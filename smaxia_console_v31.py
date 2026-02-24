# ═══════════════════════════════════════════════════════════════════
# SMAXIA COMMAND CENTER — Admin Console Premium v2.0
# 7 corrections GPT + Design Manus
# Usage: streamlit run smaxia_command_center_v2.py
# ═══════════════════════════════════════════════════════════════════
import streamlit as st
import json, os, glob, datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADERS — CAP = FILE SCELLÉ ONLY (FIX #1: zéro embedded)
# ═══════════════════════════════════════════════════════════════════
def scan_run_dirs(base="./runs"):
    if not os.path.isdir(base):
        return []
    return sorted([d for d in os.listdir(base)
                   if os.path.isdir(os.path.join(base, d))], reverse=True)

def load_json(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def find_cap(run_dir):
    if not run_dir or not os.path.isdir(run_dir):
        return None
    caps = glob.glob(os.path.join(run_dir, "CAP_*.json"))
    return caps[0] if caps else None

def load_artefact(run_dir, name):
    if not run_dir:
        return None
    p = os.path.join(run_dir, name)
    return load_json(p)

def load_qc_detail(run_dir, qc_id):
    if not run_dir:
        return None
    p = os.path.join(run_dir, "QC_Details", f"{qc_id}.json")
    return load_json(p)

def load_qi_text(run_dir, qi_id):
    if not run_dir:
        return None
    p = os.path.join(run_dir, "Qi_Text", f"{qi_id}.json")
    return load_json(p)

def artefact_exists(run_dir, name):
    if not run_dir:
        return False
    return os.path.exists(os.path.join(run_dir, name))

# ═══════════════════════════════════════════════════════════════════
# 2. CSS — Dark Premium (Manus fusion)
# ═══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

    .stApp, [data-testid="stAppViewContainer"], .main .block-container {
        background: #0a0e27 !important; color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .main .block-container { padding-top: 0.8rem !important; max-width: 1400px !important; }
    [data-testid="stSidebar"] {
        background: #0f1329 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] * { color: #b0bec5 !important; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #90caf9 !important; }

    .smx-hdr {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 1.2rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .smx-hdr h1 { color:#fff!important; font-size:1.5rem; font-weight:800; margin:0; }
    .smx-hdr p  { color:#90caf9; font-size:0.82rem; margin:0.15rem 0 0; }

    .smx-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:0.7rem; margin-bottom:1rem; }
    .smx-card {
        background: linear-gradient(135deg, #1a1f3e, #151933);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 0.9rem; text-align: center; transition: transform 0.2s;
    }
    .smx-card:hover { transform: translateY(-2px); }
    .smx-card .v {
        font-size:1.9rem; font-weight:800; color:#64b5f6;
        font-family:'JetBrains Mono',monospace; letter-spacing:-1px;
    }
    .smx-card .l {
        font-size:0.6rem; color:#78909c; font-weight:600;
        letter-spacing:1.5px; text-transform:uppercase; margin-top:0.15rem;
    }

    .cyc { padding:0.5rem 0.9rem; border-radius:8px; margin:0.35rem 0;
        font-weight:700; font-size:0.88rem; color:#fff; }
    .cyc-hs  { background:linear-gradient(90deg,#1565c0,#0d47a1); }
    .cyc-preu { background:linear-gradient(90deg,#2e7d32,#1b5e20); }
    .cyc-uni { background:linear-gradient(90deg,#e65100,#bf360c); }

    .cc {
        background:#151933; border:1px solid rgba(255,255,255,0.06);
        border-radius:10px; padding:0.6rem 0.9rem; margin:0.25rem 0;
        transition: border-color 0.2s;
    }
    .cc:hover { border-color:rgba(100,181,246,0.3); }

    .b-off { display:inline-block; background:#1b5e20; color:#a5d6a7;
        padding:0.1rem 0.5rem; border-radius:4px; font-size:0.63rem; font-weight:700; }
    .b-can { display:inline-block; background:#e65100; color:#ffcc80;
        padding:0.1rem 0.5rem; border-radius:4px; font-size:0.63rem; font-weight:700; }
    .b-seal { display:inline-block; background:linear-gradient(135deg,#1b5e20,#2e7d32);
        color:#fff; padding:0.25rem 0.7rem; border-radius:15px;
        font-weight:700; font-size:0.72rem; letter-spacing:1px; }
    .b-exam { display:inline-block; background:#311b92; color:#b39ddb;
        padding:0.1rem 0.5rem; border-radius:4px; font-size:0.68rem; font-weight:600; margin:2px; }
    .b-conc { display:inline-block; background:#1a237e; color:#90caf9;
        padding:0.1rem 0.5rem; border-radius:4px; font-size:0.68rem; font-weight:600; margin:2px; }

    .sha {
        background:#0d1117; border:1px solid #30363d; border-radius:6px;
        padding:0.45rem 0.7rem; font-family:'JetBrains Mono',monospace;
        font-size:0.68rem; color:#7ee787; word-break:break-all; margin:0.35rem 0;
    }

    .g-pass { color:#66bb6a; font-weight:700; }
    .g-fail { color:#ef5350; font-weight:700; }
    .g-unk  { color:#ffa726; font-weight:700; }
    .g-row  { padding:0.35rem 0; font-size:0.83rem; border-bottom:1px solid rgba(255,255,255,0.04); }

    .cov-o { background:#1a1f3e; border-radius:6px; height:18px; width:100%; overflow:hidden; }
    .cov-i { height:100%; border-radius:6px; transition:width 0.3s; }

    .idea-card {
        background:linear-gradient(135deg,#1a1f3e,#0f1329);
        border-left:4px solid #ffd740; border-radius:0 10px 10px 0;
        padding:0.7rem 1rem; margin:0.4rem 0;
    }

    .stTabs [data-baseweb="tab-list"] { gap:4px; }
    .stTabs [data-baseweb="tab"] {
        background:#151933!important; color:#90caf9!important;
        border-radius:8px 8px 0 0!important; border:1px solid rgba(255,255,255,0.06)!important;
    }
    .stTabs [aria-selected="true"] { background:#1a237e!important; color:#fff!important; }
    .streamlit-expanderHeader { background:#151933!important; color:#e3f2fd!important; border-radius:8px!important; }
    hr { border-color:rgba(255,255,255,0.06)!important; }
    .stSelectbox label,.stTextInput label,.stRadio label,.stFileUploader label { color:#90caf9!important; }
    .warn-box { background:rgba(255,167,38,0.08); border:1px solid rgba(255,167,38,0.25);
        border-radius:8px; padding:0.6rem 0.9rem; margin:0.5rem 0; font-size:0.83rem; }
    .sat-saturated { color:#66bb6a; font-weight:700; }
    .sat-continue  { color:#ffa726; font-weight:700; }
    </style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 3. HTML HELPERS
# ═══════════════════════════════════════════════════════════════════
def hdr(t, sub=""):
    return f'<div class="smx-hdr"><h1>{t}</h1><p>{sub}</p></div>'

def cards(items):
    h = ""
    for it in items:
        v, l = it[0], it[1]
        c = it[2] if len(it) > 2 else "#64b5f6"
        h += f'<div class="smx-card"><div class="v" style="color:{c}">{v}</div><div class="l">{l}</div></div>'
    return f'<div class="smx-metrics">{h}</div>'

def src_badge(t):
    return '<span class="b-off">OFFICIEL</span>' if t == "OFFICIEL" else '<span class="b-can">CANONIQUE</span>'

def cov_bar(pct, w="100%"):
    pct = min(max(pct, 0), 100)
    c = "#66bb6a" if pct >= 95 else ("#ffa726" if pct >= 75 else "#ef5350")
    return f'<div class="cov-o" style="width:{w}"><div class="cov-i" style="width:{pct}%;background:{c}"></div></div>'

def gate_icon(status):
    if status == "PASS": return "✅", "g-pass"
    if status == "FAIL": return "❌", "g-fail"
    return "⏳", "g-unk"

# CAP helpers
def levels_for(edu, cid):
    return sorted([l for l in edu["levels"] if l["cycle_id"] == cid], key=lambda x: x["order"])

def subjects_for(edu, lc):
    return [s for s in edu["subjects"] if s["level_code"] == lc]

def exam_for(exams, lc):
    return next((e for e in exams if e.get("level_code") == lc), None)

def cyc_cls(cid):
    return {"CYCLE_HS": "cyc-hs", "CYCLE_PREU": "cyc-preu", "CYCLE_UNI": "cyc-uni"}.get(cid, "")

# ═══════════════════════════════════════════════════════════════════
# 4. PAGES
# ═══════════════════════════════════════════════════════════════════

# ──── 4.1 DASHBOARD ────
def pg_dashboard(cap, rd):
    m = cap["A_METADATA"]; edu = cap["B_EDUCATION_SYSTEM"]
    exams = cap["E_EXAMS_CONCOURS"]["exams_and_contests"]
    tot_ch = sum(s["chapter_count"] for s in edu["subjects"])
    off = sum(1 for s in edu["subjects"] if s.get("source_type") == "OFFICIEL")
    can = sum(1 for s in edu["subjects"] if s.get("source_type") == "CANONIQUE")

    st.markdown(hdr(
        f"🇫🇷 SMAXIA — CAP {m['country_name_local']} Scellé",
        f"Kernel {m['kernel_version']} • {m['source_doctrine']} • Zéro Invention"
    ), unsafe_allow_html=True)

    st.markdown(cards([
        (len(edu["cycles"]), "Cycles"), (m["total_classes"], "Classes"),
        (m["total_subjects_go"], "Matières GO"), (tot_ch, "Chapitres"),
        (len(exams), "Examens"),
    ]), unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### 📈 Répartition par Cycle")
        for c in edu["cycles"]:
            lv = levels_for(edu, c["cycle_id"])
            su = [s for l in lv for s in subjects_for(edu, l["level_code"])]
            ch = sum(s["chapter_count"] for s in su)
            st.markdown(f'<div class="cyc {cyc_cls(c["cycle_id"])}">'
                        f'{c["cycle_name_local"]} ({c["cycle_name_en"]}) — '
                        f'{len(lv)} classes · {len(su)} matières · {ch} chap.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("#### 🔐 Intégrité")
        st.markdown(f'<div class="sha">📋 {m["cap_fingerprint_sha256"]}</div>', unsafe_allow_html=True)
        seal = '<span class="b-seal">✓ SEALED</span>' if m["status"] == "SEALED" else '<span class="g-fail">⚠ NOT SEALED</span>'
        st.markdown(f"📜 **{m['source_doctrine']}** · 🏛️ **{m['jurisdiction_model']}** · "
                    f"📅 **{m['academic_year']}** · {seal}", unsafe_allow_html=True)
        st.markdown(f"{src_badge('OFFICIEL')} **{off}** &nbsp; {src_badge('CANONIQUE')} **{can}**", unsafe_allow_html=True)

    # ── CES Pipeline (FIX #5: CoverageTimeline chart) ──
    st.markdown("---")
    st.markdown("#### 🚀 Pipeline CES HARVEST")
    ces = load_artefact(rd, "CES_State.json")
    if ces:
        st.markdown(cards([
            (ces.get("pdf_count", "N/A"), "PDFs"), (ces.get("pairs_count", "N/A"), "Paires"),
            (ces.get("qi_count", "N/A"), "Qi"), (ces.get("qc_count", "N/A"), "QC"),
            (f"{ces.get('coverage_avg', 0):.0f}%", "Couverture moy."),
            (ces.get("orphans_total", "N/A"), "Orphelins"),
        ]), unsafe_allow_html=True)

        # Coverage Timeline chart
        timeline = load_artefact(rd, "CoverageTimeline.json")
        if timeline and isinstance(timeline, list) and len(timeline) > 0:
            import pandas as pd
            df = pd.DataFrame(timeline)
            if "date" in df.columns and "coverage" in df.columns:
                st.markdown("##### 📊 Évolution Couverture")
                st.line_chart(df.set_index("date")["coverage"], use_container_width=True)
            if "orphans" in df.columns:
                st.markdown("##### 📉 Évolution Orphelins")
                st.line_chart(df.set_index("date")["orphans"], use_container_width=True)
        else:
            st.caption("📊 CoverageTimeline.json non trouvé — les graphes apparaîtront après les premiers runs.")
    else:
        st.info("⏳ **CES HARVEST non lancé** — Les KPIs et graphes apparaîtront après E1 COLLECT.")

# ──── 4.2 CAP EXPLORER ────
def pg_cap(cap, rd):
    m = cap["A_METADATA"]; edu = cap["B_EDUCATION_SYSTEM"]
    exams = cap["E_EXAMS_CONCOURS"]["exams_and_contests"]
    tot_ch = sum(s["chapter_count"] for s in edu["subjects"])
    st.markdown(hdr("📚 CAP Explorer",
        f"{m['total_classes']} classes · {m['total_subjects_go']} matières · {tot_ch} chapitres"), unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["🏫 Niveaux", "📘 Matières", "📖 Chapitres", "🎓 Examens", "🔗 Sources"])

    with t1:
        for c in edu["cycles"]:
            lv = levels_for(edu, c["cycle_id"])
            st.markdown(f'<div class="cyc {cyc_cls(c["cycle_id"])}">{c["cycle_name_local"]} — {len(lv)} classes</div>', unsafe_allow_html=True)
            for l in lv:
                su = subjects_for(edu, l["level_code"])
                ch = sum(s["chapter_count"] for s in su)
                ex = exam_for(exams, l["level_code"])
                eh = f' <span class="b-exam">🎯 {ex["exam"]["exam_name"]}</span>' if ex and ex.get("exam") else ""
                ch_html = " ".join(f'<span class="b-conc">🏆#{c["rank"]} {c["name"]}</span>'
                                   for c in (ex.get("contests_top") or [])[:5]) if ex else ""
                st.markdown(f"""<div class="cc">
                    <b style="color:#e3f2fd">{l['level_name_local']}</b>
                    <span style="color:#546e7a;font-size:0.78rem"> ({l['level_code']})</span>
                    <div style="color:#546e7a;font-size:0.72rem">{l.get('voie','')}</div>
                    <div style="color:#90caf9;font-size:0.78rem;margin-top:0.15rem">📘 {len(su)} mat. · 📖 {ch} chap.</div>
                    {eh}{f'<div style="margin-top:0.15rem">{ch_html}</div>' if ch_html else ''}
                </div>""", unsafe_allow_html=True)

    with t2:
        q = st.text_input("🔍 Rechercher matière", key="s_s")
        for su in edu["subjects"]:
            ln = next((l["level_name_local"] for l in edu["levels"] if l["level_code"] == su["level_code"]), su["level_code"])
            full = f"{ln} — {su['subject_name_local']}"
            if q and q.lower() not in full.lower():
                continue
            st.markdown(f'<div class="cc"><b style="color:#e3f2fd">{ln}</b> — {su["subject_name_local"]} '
                        f'({su["chapter_count"]} chap.) {src_badge(su.get("source_type",""))}'
                        f'<div style="color:#546e7a;font-size:0.68rem">{su.get("source_ref","")}</div></div>',
                        unsafe_allow_html=True)

    with t3:
        q2 = st.text_input("🔍 Rechercher chapitre", key="c_s")
        for su in edu["subjects"]:
            ln = next((l["level_name_local"] for l in edu["levels"] if l["level_code"] == su["level_code"]), su["level_code"])
            chs = [ch for ch in su["chapters"]
                   if not q2 or q2.lower() in ch["chapter_name"].lower()
                   or q2.lower() in su["subject_name_local"].lower() or q2.lower() in ln.lower()]
            if not chs:
                continue
            with st.expander(f"**{ln} — {su['subject_name_local']}** ({len(su['chapters'])} chap.) {su.get('source_type','')}"):
                for ch in chs:
                    st.markdown(f'<span style="color:#64b5f6;font-weight:700;font-family:JetBrains Mono">'
                                f'{str(ch["chapter_number"]).zfill(2)}</span> {ch["chapter_name"]}',
                                unsafe_allow_html=True)

    with t4:
        for c in edu["cycles"]:
            lv = levels_for(edu, c["cycle_id"])
            st.markdown(f'<div class="cyc {cyc_cls(c["cycle_id"])}">{c["cycle_name_local"]}</div>', unsafe_allow_html=True)
            for l in lv:
                ex = exam_for(exams, l["level_code"])
                if not ex:
                    continue
                eh = f'<span class="b-exam">🎯 {ex["exam"]["exam_name"]}</span>' if ex.get("exam") else ""
                ch = " ".join(f'<span class="b-conc">🏆#{c["rank"]} {c["name"]}</span>' for c in (ex.get("contests_top") or []))
                st.markdown(f'<div class="cc"><b style="color:#e3f2fd">{l["level_name_local"]}</b> '
                            f'<span style="color:#546e7a;font-size:0.78rem">({l["level_code"]})</span> '
                            f'{eh}<div style="margin-top:0.15rem">{ch}</div></div>', unsafe_allow_html=True)

    with t5:
        src = cap["C_HARVEST_SOURCES"]
        for s in src["sources"]:
            pc = {"A": "#66bb6a", "B": "#ffa726", "C": "#ef5350"}.get(s["proof"], "#fff")
            st.markdown(f'<div class="cc"><b style="color:#e3f2fd">{s["source_id"]}</b> — {s["domain"]}'
                        f'<div style="color:#90caf9;font-size:0.78rem">📋 {s["scope"]}</div>'
                        f'<div style="font-size:0.78rem">🏅 {s["authority_score"]} · '
                        f'Proof: <span style="color:{pc};font-weight:700">{s["proof"]}</span> · '
                        f'{", ".join(s["levels_covered"][:5])}{"..." if len(s["levels_covered"])>5 else ""}</div></div>',
                        unsafe_allow_html=True)
        r = src["scraping_rules"]
        st.markdown(cards([(f'{r["rate_limit_ms"]}ms', "Rate Limit"), (r["max_concurrent"], "Max Conc."), ("✅", "Robots.txt")]),
                    unsafe_allow_html=True)

# ──── 4.3 CES MONITOR ────
def pg_ces(cap, rd):
    st.markdown(hdr("🚀 CES HARVEST Monitor", "Pipeline · Gates · Saturation"), unsafe_allow_html=True)
    ces = load_artefact(rd, "CES_State.json")
    expected = ["CES_State.json", "QC_Index.json", "Qi_Index.json", "CoverageReport.json",
                "CoverageTimeline.json", "QuarantineLedger.json", "CHK_REPORT.json",
                "SealReport.json", "DeterminismReport_3runs.json",
                "PairingReport_G2.json", "OcrReport_G3.json", "AtomTrace_G4.json", "DedupReport_G5.json"]

    if not ces:
        st.warning("⏳ **CES HARVEST non lancé** — Aucun artefact détecté.")
    else:
        st.json(ces)

    st.markdown("#### 📦 Artefacts")
    for f in expected:
        icon = "✅" if artefact_exists(rd, f) else "❌"
        st.markdown(f"&nbsp;&nbsp; {icon} `{f}`", unsafe_allow_html=True)

# ──── 4.4 CHAPTERS & QC + STOP_RULE (FIX #4) ────
def pg_chapters(cap, rd):
    st.markdown(hdr("📈 Chapitres & QC", "Couverture · Saturation · STOP_RULE"), unsafe_allow_html=True)
    edu = cap["B_EDUCATION_SYSTEM"]

    # Filters
    c1, c2, c3 = st.columns(3)
    cn = {c["cycle_id"]: c["cycle_name_local"] for c in edu["cycles"]}
    with c1:
        sc = st.selectbox("Cycle", ["Tous"] + list(cn.values()), key="f_c")
    scid = next((k for k, v in cn.items() if v == sc), None)
    lf = edu["levels"] if not scid else [l for l in edu["levels"] if l["cycle_id"] == scid]
    ln = {l["level_code"]: l["level_name_local"] for l in lf}
    with c2:
        sl = st.selectbox("Niveau", ["Tous"] + list(ln.values()), key="f_l")
    slc = next((k for k, v in ln.items() if v == sl), None)
    sf = edu["subjects"]
    if slc:
        sf = [s for s in sf if s["level_code"] == slc]
    elif scid:
        lcs = [l["level_code"] for l in lf]
        sf = [s for s in sf if s["level_code"] in lcs]
    sn = {s["subject_id"]: s["subject_name_local"] for s in sf}
    with c3:
        ss = st.selectbox("Matière", ["Toutes"] + list(sn.values()), key="f_s")
    ssid = next((k for k, v in sn.items() if v == ss), None)
    if ssid:
        sf = [s for s in sf if s["subject_id"] == ssid]

    qc_idx = load_artefact(rd, "QC_Index.json") or {}
    sat_report = load_artefact(rd, "SaturationReport.json") or {}

    for su in sf:
        lname = next((l["level_name_local"] for l in edu["levels"] if l["level_code"] == su["level_code"]), "")
        with st.expander(f"**{lname} — {su['subject_name_local']}** ({su['chapter_count']} chap.)"):
            for ch in su["chapters"]:
                ck = f"{su['subject_id']}_CH{str(ch['chapter_number']).zfill(2)}"
                qd = qc_idx.get(ck, {})
                sd = sat_report.get(ck, {})
                qi = qd.get("qi_count", 0)
                qc = qd.get("qc_count", 0)
                cv = qd.get("coverage", 0)
                orph = qd.get("orphans", 0)
                # STOP_RULE fields (FIX #4)
                dqc = sd.get("delta_qc", "—")
                orph_new = sd.get("orphans_new", "—")
                win_n = sd.get("window_pairs_N", "—")
                sat_status = sd.get("status", "UNKNOWN")

                ca, cb, cc_, cd, ce, cf = st.columns([3, 0.8, 0.8, 1.5, 0.8, 1.2])
                with ca:
                    st.markdown(f'<span style="color:#64b5f6;font-weight:700">'
                                f'{str(ch["chapter_number"]).zfill(2)}</span> {ch["chapter_name"]}',
                                unsafe_allow_html=True)
                with cb:
                    st.metric("QC", qc)
                with cc_:
                    st.metric("Qi", qi)
                with cd:
                    st.markdown(cov_bar(cv), unsafe_allow_html=True)
                    st.caption(f"{cv}%")
                with ce:
                    st.metric("Orph.", orph)
                with cf:
                    if sat_status == "SATURATED":
                        st.markdown('<span class="sat-saturated">✅ SATURÉ</span>', unsafe_allow_html=True)
                    elif sat_status == "CONTINUE":
                        st.markdown('<span class="sat-continue">🔄 CONTINUE</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="g-unk">⏳</span>', unsafe_allow_html=True)

                # STOP_RULE detail (FIX #4)
                if sd:
                    st.caption(f"ΔQC={dqc} · Orph.new={orph_new} · Window={win_n}")

# ──── 4.5 QC DETAIL — structured (FIX #6) ────
def pg_qc_detail(cap, rd):
    st.markdown(hdr("🔬 QC Detail", "FRT · ARI · TRIGGERS — Opaque (IP protégée)"), unsafe_allow_html=True)

    qc_idx = load_artefact(rd, "QC_Index.json")
    if not qc_idx:
        st.info("⏳ **Aucune QC** — En attente CES HARVEST.")
        st.markdown("""
        Quand les QC seront générées, cette page affichera pour chaque QC :
        - **QC_ID** + chapitre + couverture
        - **FRT** : ID + SHA256 + lien artefact *(formule opaque — IP SMAXIA)*
        - **ARI** : ID + SHA256 + lien artefact *(formule opaque)*
        - **TRIGGERS** : IDs + SHA256 *(mécaniques opaques)*
        - **Qi children** : liste + preview texte
        - **Correctness** : verdict judge (PASS/FAIL)
        """)
        return

    # List all QCs from all chapters
    all_qcs = []
    for ck, data in qc_idx.items():
        for qid in data.get("qc_ids", []):
            all_qcs.append((ck, qid))
    if not all_qcs:
        st.warning("QC_Index chargé mais aucun qc_id trouvé.")
        return

    sel_ch = st.selectbox("Chapitre", list(set(ck for ck, _ in all_qcs)), key="qcd_ch")
    ch_qcs = [qid for ck, qid in all_qcs if ck == sel_ch]
    sel_qc = st.selectbox("QC", ch_qcs, key="qcd_qc")

    if sel_qc:
        detail = load_qc_detail(rd, sel_qc)
        if detail:
            # Structured display
            st.markdown(f"##### QC: `{sel_qc}`")
            c1, c2, c3 = st.columns(3)
            with c1:
                frt = detail.get("frt", {})
                st.markdown(f'<div class="cc" style="border-left:3px solid #42a5f5">'
                            f'<b style="color:#42a5f5">FRT</b><br>'
                            f'ID: <code>{frt.get("id","N/A")}</code><br>'
                            f'SHA: <code style="font-size:0.6rem">{frt.get("sha256","N/A")[:24]}...</code><br>'
                            f'<span style="color:#78909c;font-size:0.7rem">Artefact: {frt.get("artefact","N/A")}</span>'
                            f'</div>', unsafe_allow_html=True)
            with c2:
                ari = detail.get("ari", {})
                st.markdown(f'<div class="cc" style="border-left:3px solid #66bb6a">'
                            f'<b style="color:#66bb6a">ARI</b><br>'
                            f'ID: <code>{ari.get("id","N/A")}</code><br>'
                            f'SHA: <code style="font-size:0.6rem">{ari.get("sha256","N/A")[:24]}...</code><br>'
                            f'<span style="color:#78909c;font-size:0.7rem">Artefact: {ari.get("artefact","N/A")}</span>'
                            f'</div>', unsafe_allow_html=True)
            with c3:
                trigs = detail.get("triggers", [])
                trig_html = "<br>".join(f'<code>{t.get("id","?")}</code>' for t in trigs[:5]) if trigs else "N/A"
                st.markdown(f'<div class="cc" style="border-left:3px solid #ffa726">'
                            f'<b style="color:#ffa726">TRIGGERS ({len(trigs)})</b><br>'
                            f'{trig_html}'
                            f'</div>', unsafe_allow_html=True)

            # Qi children (FIX #6)
            qi_ids = detail.get("qi_children", [])
            if qi_ids:
                st.markdown(f"##### Qi associées ({len(qi_ids)})")
                for qiid in qi_ids[:20]:
                    qi_data = load_qi_text(rd, qiid)
                    preview = qi_data.get("text", "")[:120] + "..." if qi_data else "⏳ fichier non trouvé"
                    st.markdown(f'<div class="cc"><code style="color:#64b5f6">{qiid}</code>'
                                f'<div style="color:#b0bec5;font-size:0.78rem">{preview}</div></div>',
                                unsafe_allow_html=True)

            # Correctness
            corr = detail.get("correctness", {})
            if corr:
                v = corr.get("verdict", "UNKNOWN")
                ic, cl = gate_icon(v)
                st.markdown(f"**Correctness judge:** {ic} <span class='{cl}'>{v}</span> "
                            f"(confiance: {corr.get('confidence','N/A')})", unsafe_allow_html=True)
        else:
            st.warning(f"Fichier `QC_Details/{sel_qc}.json` non trouvé.")

# ──── 4.6 MAPPING ────
def pg_mapping(cap, rd):
    st.markdown(hdr("🗺️ Mapping Qi → QC", "Orphelins · Couverture"), unsafe_allow_html=True)
    qi_idx = load_artefact(rd, "Qi_Index.json")
    cov = load_artefact(rd, "CoverageReport.json")
    if not qi_idx and not cov:
        st.info("⏳ **Aucune Qi indexée** — En attente CES HARVEST.")
    if cov and isinstance(cov, dict):
        st.markdown("#### Couverture par chapitre")
        for ck, data in sorted(cov.items()):
            cv = data.get("coverage", 0)
            orph = data.get("orphans", 0)
            st.markdown(f'`{ck}` — {cov_bar(cv, "60%")} {cv}% · Orphelins: {orph}', unsafe_allow_html=True)

# ──── 4.7 TESTS A/B (FIX #7) ────
def pg_tests(cap, rd):
    st.markdown(hdr("🧪 Tests Orphelins & Couverture",
        "Test A : sujet traité | Test B : sujet nouveau"), unsafe_allow_html=True)

    ta, tb = st.tabs(["Test A — Sujet traité", "Test B — Sujet nouveau"])

    with ta:
        st.markdown("**Objectif** : pour un sujet déjà traité, vérifier **Qi_orphelin = 0**.")
        st.markdown('<div class="warn-box">⚠️ Lit depuis <code>MappingReport_&lt;doc_id&gt;.json</code> '
                    'dans le run directory. Aucune exécution locale.</div>', unsafe_allow_html=True)
        doc_id = st.text_input("doc_id du sujet traité", key="ta_doc")
        if doc_id and st.button("▶ Charger MappingReport", key="ta_btn"):
            mr = load_artefact(rd, f"MappingReport_{doc_id}.json")
            if mr:
                orphans = mr.get("orphans", [])
                total_qi = mr.get("qi_total", 0)
                mapped = mr.get("qi_mapped", 0)
                st.markdown(cards([
                    (total_qi, "Qi totales"), (mapped, "Qi mappées"),
                    (len(orphans), "Qi orphelines", "#ef5350" if orphans else "#66bb6a"),
                    (f"{mr.get('coverage',0):.0f}%", "Couverture"),
                ]), unsafe_allow_html=True)
                if orphans:
                    st.error(f"❌ **{len(orphans)} Qi orphelines** — couverture insuffisante")
                    for o in orphans[:20]:
                        st.markdown(f'<div class="cc"><code>{o.get("qi_id","?")}</code> — '
                                    f'{o.get("text_preview","N/A")[:100]}</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ **Qi_orphelin = 0** — Couverture complète !")
                # Show chapters covered
                chapters = mr.get("chapters_covered", [])
                if chapters:
                    st.markdown("**Chapitres couverts :**")
                    for ch in chapters:
                        st.markdown(f'<span class="b-conc">{ch}</span>', unsafe_allow_html=True)
            else:
                st.error(f"❌ `MappingReport_{doc_id}.json` non trouvé dans {rd}")

    with tb:
        st.markdown("**Objectif** : pour un sujet **jamais vu**, tester la couverture QC.")
        st.markdown('<div class="warn-box">⚠️ Le pipeline CES n\'est pas exécuté localement. '
                    'L\'upload crée une requête <code>TestRequest.json</code> dans le run directory '
                    'pour traitement asynchrone par le pipeline.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("📄 Upload PDF sujet", type=["pdf"], key="tb_up")
        if uploaded and st.button("▶ Créer TestRequest", key="tb_btn"):
            if not rd or not os.path.isdir(rd):
                st.error("❌ Run directory invalide — impossible de créer TestRequest.")
            else:
                import hashlib
                pdf_bytes = uploaded.read()
                pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
                req = {
                    "type": "TEST_B_NEW_SUBJECT",
                    "filename": uploaded.name,
                    "pdf_sha256": pdf_sha,
                    "pdf_size_bytes": len(pdf_bytes),
                    "requested_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "status": "PENDING",
                    "requested_by": "CEO_UI",
                }
                out_path = os.path.join(rd, f"TestRequest_{pdf_sha[:12]}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(req, f, indent=2, ensure_ascii=False)
                # Save PDF
                pdf_dir = os.path.join(rd, "test_pdfs")
                os.makedirs(pdf_dir, exist_ok=True)
                with open(os.path.join(pdf_dir, uploaded.name), "wb") as f:
                    f.write(pdf_bytes)
                st.success(f"✅ TestRequest créé : `{out_path}`")
                st.json(req)
                st.info("Le pipeline CES traitera cette requête au prochain run. "
                        "Résultat dans `MappingReport_{doc_id}.json`.")

# ──── 4.8 GATES & INTEGRITY (FIX #2: zéro PASS hardcodé) ────
def pg_gates(cap, rd):
    m = cap["A_METADATA"]
    st.markdown(hdr("🔐 Gates & Intégrité",
        f"SHA256 · Conformité Kernel {m['kernel_version']}"), unsafe_allow_html=True)

    st.markdown("#### 🔍 Empreinte CAP")
    st.markdown(f'<div class="sha">{m["cap_fingerprint_sha256"]}</div>', unsafe_allow_html=True)

    # FIX #2: ALL gates read from CHK_REPORT.json — no hardcoded PASS
    chk = load_artefact(rd, "CHK_REPORT.json") or {}

    st.markdown("#### 🚦 Gates (depuis CHK_REPORT.json)")
    if not chk:
        st.markdown('<div class="warn-box">⚠️ <code>CHK_REPORT.json</code> non trouvé. '
                    'Toutes les gates = <b>UNKNOWN</b>. Règle B4 : toute gate sans artefact = FAIL automatique.</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="g-row">Aucune gate chargée — fournir CHK_REPORT.json dans le run directory.</div>',
                    unsafe_allow_html=True)
    else:
        pass_c = fail_c = unk_c = 0
        for name, status in chk.items():
            ic, cl = gate_icon(status)
            st.markdown(f'<div class="g-row">{ic} <span class="{cl}">{name}</span> — {status}</div>',
                        unsafe_allow_html=True)
            if status == "PASS": pass_c += 1
            elif status == "FAIL": fail_c += 1
            else: unk_c += 1
        st.markdown(cards([
            (pass_c, "PASS", "#66bb6a"), (fail_c, "FAIL", "#ef5350"), (unk_c, "UNKNOWN", "#ffa726"),
        ]), unsafe_allow_html=True)

    # Determinism
    st.markdown("#### 🔄 Déterminisme")
    det = load_artefact(rd, "DeterminismReport_3runs.json")
    if det:
        st.json(det)
    else:
        st.info("⏳ DeterminismReport non disponible — généré après 3 runs identiques.")

    # Doctrine reminder
    st.markdown("#### 📜 Doctrine")
    st.markdown("""
    🔒 **CAS 1 ONLY** — Zéro reconstruction, zéro invention  
    🔒 **Zéro Hardcode Pays** — Tout piloté par CAP  
    🔒 **Déterminisme** — 3 runs identiques requis  
    🔒 **B4** — Toute gate sans artefact = FAIL automatique  
    """)

# ──── 4.9 QUARANTINE ────
def pg_quarantine(cap, rd):
    st.markdown(hdr("🔶 Quarantine Ledger", "Items en quarantaine · Résolution"), unsafe_allow_html=True)
    ql = load_artefact(rd, "QuarantineLedger.json")
    if not ql:
        st.info("⏳ Aucun item en quarantaine.")
        return
    if isinstance(ql, list):
        import pandas as pd
        df = pd.DataFrame(ql)
        st.dataframe(df, use_container_width=True)
    else:
        st.json(ql)

# ──── 4.10 DEV NOTES (FIX #3: Best Ideas → LOCKED) ────
def pg_devnotes():
    st.markdown(hdr("🔒 Dev Notes", "Espace verrouillé — Aucun claim sans KPI scellé"), unsafe_allow_html=True)
    st.markdown('<div class="warn-box">⚠️ <b>Page VERROUILLÉE</b> — Les idées stratégiques et claims marketing '
                'ne peuvent être affichés que s\'ils sont adossés à un <code>KPI_Report.json</code> scellé avec preuves. '
                'Aucun pourcentage, aucune prédiction, aucun claim sans artefact.</div>', unsafe_allow_html=True)
    st.markdown("Cette page sera déverrouillée quand :")
    st.markdown("""
    - `KPI_Report.json` existe dans le run directory
    - Chaque claim est adossé à un gate PASS + artefact SHA256
    - Le panel a validé (GO unanime)
    """)

# ═══════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="SMAXIA Command Center", page_icon="🔒",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### 🔒 SMAXIA Command Center")

        # Run directory
        st.markdown("---")
        base = st.text_input("📁 Runs base", value="./runs", key="base_dir")
        available = scan_run_dirs(base)
        if available:
            sel_run = st.selectbox("Run", available, key="sel_run")
            rd = os.path.join(base, sel_run)
        else:
            rd = st.text_input("📁 Run path", value="./runs/latest", key="run_manual")

        st.session_state["run_dir"] = rd

        # CAP loading (FIX #1: file only, no embedded)
        cap_path = find_cap(rd)
        cap_upload = None
        if not cap_path:
            st.warning("⚠️ Aucun CAP_*.json dans le run dir.")
            cap_upload = st.file_uploader("📄 Upload CAP JSON", type=["json"], key="cap_up")

        cap = None
        if cap_path:
            cap = load_json(cap_path)
            st.success(f"✅ CAP: `{os.path.basename(cap_path)}`")
        elif cap_upload:
            cap = json.loads(cap_upload.read().decode("utf-8"))
            st.success(f"✅ CAP uploadé: `{cap_upload.name}`")

        if cap:
            m = cap.get("A_METADATA", {})
            st.markdown(f"**Pays:** {m.get('country_code','?')} — {m.get('country_name_local','?')}")
            st.markdown(f"**Kernel:** {m.get('kernel_version','?')}")
            st.markdown(f"**CAP:** {m.get('version','?')}")
            if m.get("status") == "SEALED":
                st.markdown('<span class="b-seal">✓ SEALED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="g-fail">⚠ NOT SEALED</span>', unsafe_allow_html=True)

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
            "🔒 Dev Notes",
        ], label_visibility="collapsed")

    # ── NO CAP → STOP ──
    if not cap:
        st.markdown(hdr("⚠️ Aucun CAP chargé", "Placez CAP_*.json dans le run directory ou uploadez-le"), unsafe_allow_html=True)
        st.markdown("""
        **Pour démarrer :**  
        1. Créez le dossier `./runs/latest/`  
        2. Placez-y `CAP_FR.json` (ou tout `CAP_*.json` scellé)  
        3. Rafraîchissez la page  
        
        *Ou utilisez l'uploader dans la sidebar.*
        """)
        return

    # ── HEALTH BAR ──
    m = cap["A_METADATA"]
    chk = load_artefact(rd, "CHK_REPORT.json") or {}
    pc = sum(1 for v in chk.values() if v == "PASS")
    tc = len(chk) if chk else "?"
    qs = "✅" if artefact_exists(rd, "QuarantineLedger.json") else "❌"
    st.markdown(f'<div style="background:#0f1329;padding:0.35rem 0.9rem;border-radius:8px;'
                f'font-size:0.72rem;color:#78909c;margin-bottom:0.5rem;'
                f'border:1px solid rgba(255,255,255,0.04)">'
                f'<b style="color:#e3f2fd">SMAXIA</b> · '
                f'CAP: {m.get("cap_id","?")} · Kernel: {m.get("kernel_version","?")} · '
                f'{m.get("source_doctrine","?")} · '
                f'Gates: <span class="{"g-pass" if pc==tc and tc!="?" else "g-unk"}">{pc}/{tc}</span> · '
                f'Quarantine: {qs}</div>', unsafe_allow_html=True)

    # ── ROUTING ──
    routes = {
        "📊 Dashboard": pg_dashboard,
        "📚 CAP Explorer": pg_cap,
        "🚀 CES Monitor": pg_ces,
        "📈 Chapitres & QC": pg_chapters,
        "🔬 QC Detail (FRT/ARI)": pg_qc_detail,
        "🗺️ Mapping Qi→QC": pg_mapping,
        "🧪 Tests (Orphelins)": pg_tests,
        "🔐 Gates & Intégrité": pg_gates,
        "🔶 Quarantine": pg_quarantine,
    }
    if page == "🔒 Dev Notes":
        pg_devnotes()
    elif page in routes:
        routes[page](cap, rd)

if __name__ == "__main__":
    main()
