# ═══════════════════════════════════════════════════════════════════
# SMAXIA COMMAND CENTER — World Wide Admin Cockpit v3.0
# Multi-country · QC Generation Progress · CEO-first UX
# Usage: streamlit run smaxia_command_center_v3.py
# ═══════════════════════════════════════════════════════════════════
import streamlit as st
import json, os, glob, datetime, hashlib
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADERS
# ═══════════════════════════════════════════════════════════════════
def scan_runs(base="./runs"):
    if not os.path.isdir(base):
        return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))], reverse=True)

def scan_caps(run_dir):
    if not run_dir or not os.path.isdir(run_dir):
        return []
    return sorted(glob.glob(os.path.join(run_dir, "CAP_*.json")))

def load_json(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_art(rd, name):
    return load_json(os.path.join(rd, name)) if rd else None

def art_exists(rd, name):
    return os.path.exists(os.path.join(rd, name)) if rd else False

def load_qc_detail_file(rd, qc_id):
    return load_json(os.path.join(rd, "QC_Details", f"{qc_id}.json")) if rd else None

def load_qi_text_file(rd, qi_id):
    return load_json(os.path.join(rd, "Qi_Text", f"{qi_id}.json")) if rd else None

# ═══════════════════════════════════════════════════════════════════
# 2. CSS — Ultra Premium Dark
# ═══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

    :root { --bg: #0a0e27; --bg2: #0f1329; --card: #151933; --border: rgba(255,255,255,0.06);
            --txt: #e0e0e0; --txt2: #78909c; --accent: #64b5f6; --blue: #90caf9; }

    .stApp, [data-testid="stAppViewContainer"], .main .block-container {
        background: var(--bg) !important; color: var(--txt) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .main .block-container { padding-top: 0.5rem !important; max-width: 1440px !important; }

    /* SIDEBAR */
    [data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
    [data-testid="stSidebar"] * { color: #b0bec5 !important; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color: var(--blue) !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.88rem !important; padding: 2px 0 !important; }
    [data-testid="stSidebar"] .stRadio label:hover { color: #e3f2fd !important; }

    /* HEADER */
    .hdr { background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 1rem 1.5rem; border-radius: 14px; margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
    .hdr h1 { color:#fff!important; font-size:1.4rem; font-weight:800; margin:0; letter-spacing:-0.5px; }
    .hdr p { color:var(--blue); font-size:0.8rem; margin:0.1rem 0 0; }

    /* CEO STRIP */
    .ceo-strip { background: linear-gradient(90deg, #0f1329 0%, #151933 100%);
        padding: 0.5rem 1.2rem; border-radius: 10px; margin-bottom: 0.8rem;
        border: 1px solid var(--border); display: flex; align-items: center;
        gap: 1.2rem; flex-wrap: wrap; }
    .ceo-chip { display: inline-flex; align-items: center; gap: 0.3rem;
        font-size: 0.72rem; font-weight: 600; }
    .ceo-chip .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .dot-green { background: #66bb6a; }
    .dot-red { background: #ef5350; }
    .dot-orange { background: #ffa726; }
    .dot-blue { background: #42a5f5; }

    /* CARDS */
    .mx { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:0.6rem; margin-bottom:0.8rem; }
    .mc { background: linear-gradient(135deg, #1a1f3e, var(--card));
        border: 1px solid var(--border); border-radius: 12px;
        padding: 0.8rem; text-align: center; transition: transform 0.15s; }
    .mc:hover { transform: translateY(-2px); }
    .mc .v { font-size:1.8rem; font-weight:800; color:var(--accent);
        font-family:'JetBrains Mono',monospace; letter-spacing:-1px; }
    .mc .l { font-size:0.58rem; color:var(--txt2); font-weight:600;
        letter-spacing:1.5px; text-transform:uppercase; margin-top:0.1rem; }

    /* CYCLE BARS */
    .cyc { padding:0.45rem 0.8rem; border-radius:8px; margin:0.3rem 0;
        font-weight:700; font-size:0.85rem; color:#fff; }
    .cyc-hs { background:linear-gradient(90deg,#1565c0,#0d47a1); }
    .cyc-preu { background:linear-gradient(90deg,#2e7d32,#1b5e20); }
    .cyc-uni { background:linear-gradient(90deg,#e65100,#bf360c); }

    /* COMMON CARD */
    .cc { background:var(--card); border:1px solid var(--border);
        border-radius:10px; padding:0.55rem 0.85rem; margin:0.2rem 0;
        transition: border-color 0.2s; }
    .cc:hover { border-color:rgba(100,181,246,0.3); }

    /* BADGES */
    .b-off { display:inline-block; background:#1b5e20; color:#a5d6a7;
        padding:0.08rem 0.45rem; border-radius:4px; font-size:0.62rem; font-weight:700; }
    .b-can { display:inline-block; background:#e65100; color:#ffcc80;
        padding:0.08rem 0.45rem; border-radius:4px; font-size:0.62rem; font-weight:700; }
    .b-seal { display:inline-block; background:linear-gradient(135deg,#1b5e20,#2e7d32);
        color:#fff; padding:0.2rem 0.65rem; border-radius:15px;
        font-weight:700; font-size:0.7rem; letter-spacing:1px; }
    .b-exam { display:inline-block; background:#311b92; color:#b39ddb;
        padding:0.08rem 0.45rem; border-radius:4px; font-size:0.66rem; font-weight:600; margin:1px; }
    .b-conc { display:inline-block; background:#1a237e; color:var(--blue);
        padding:0.08rem 0.45rem; border-radius:4px; font-size:0.66rem; font-weight:600; margin:1px; }

    /* SHA */
    .sha { background:#0d1117; border:1px solid #30363d; border-radius:6px;
        padding:0.4rem 0.65rem; font-family:'JetBrains Mono',monospace;
        font-size:0.65rem; color:#7ee787; word-break:break-all; margin:0.3rem 0; }

    /* GATES */
    .gp { color:#66bb6a; font-weight:700; } .gf { color:#ef5350; font-weight:700; }
    .gu { color:#ffa726; font-weight:700; }
    .gr { padding:0.3rem 0; font-size:0.82rem; border-bottom:1px solid var(--border); }

    /* COVERAGE BAR */
    .co { background:#1a1f3e; border-radius:6px; height:16px; width:100%; overflow:hidden; }
    .ci { height:100%; border-radius:6px; transition:width 0.3s; }

    /* PROGRESS BAR (QC generation) */
    .prog-wrap { margin: 0.15rem 0; }
    .prog-label { font-size: 0.72rem; color: var(--txt2); margin-bottom: 2px; display:flex; justify-content:space-between; }
    .prog-bar { background: #1a1f3e; border-radius: 5px; height: 10px; overflow: hidden; }
    .prog-fill { height: 100%; border-radius: 5px; transition: width 0.4s ease; }
    .prog-fill-done { background: linear-gradient(90deg, #2e7d32, #66bb6a); }
    .prog-fill-wip { background: linear-gradient(90deg, #e65100, #ffa726); }
    .prog-fill-wait { background: #37474f; }

    /* WIZARD */
    .wiz { background: var(--card); border: 1px solid var(--border); border-radius: 14px;
        padding: 1.5rem 2rem; margin: 1rem auto; max-width: 700px; text-align: center; }
    .wiz h2 { color: var(--blue) !important; font-size: 1.3rem; font-weight: 700; }
    .wiz-steps { display: flex; justify-content: center; gap: 2rem; margin: 1.2rem 0; }
    .wiz-step { text-align: center; }
    .wiz-num { width: 36px; height: 36px; border-radius: 50%; display: inline-flex;
        align-items: center; justify-content: center; font-weight: 800; font-size: 1rem;
        margin-bottom: 0.3rem; }
    .wiz-active { background: #1565c0; color: #fff; }
    .wiz-done { background: #2e7d32; color: #fff; }
    .wiz-wait { background: #37474f; color: #78909c; }
    .wiz-label { font-size: 0.7rem; color: var(--txt2); }

    /* WARN */
    .wb { background:rgba(255,167,38,0.08); border:1px solid rgba(255,167,38,0.25);
        border-radius:8px; padding:0.5rem 0.8rem; margin:0.4rem 0; font-size:0.82rem; }

    /* NAV SECTION */
    .nav-section { font-size: 0.62rem; color: #546e7a; font-weight: 700;
        letter-spacing: 2px; text-transform: uppercase; margin: 0.6rem 0 0.15rem; padding-left: 4px; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap:3px; }
    .stTabs [data-baseweb="tab"] { background:var(--card)!important; color:var(--blue)!important;
        border-radius:8px 8px 0 0!important; border:1px solid var(--border)!important; }
    .stTabs [aria-selected="true"] { background:#1a237e!important; color:#fff!important; }
    .streamlit-expanderHeader { background:var(--card)!important; color:#e3f2fd!important; border-radius:8px!important; }
    hr { border-color:var(--border)!important; }
    .stSelectbox label,.stTextInput label,.stRadio label,.stFileUploader label { color:var(--blue)!important; }
    .sat-ok { color:#66bb6a; font-weight:700; } .sat-cont { color:#ffa726; font-weight:700; }
    </style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 3. HTML HELPERS
# ═══════════════════════════════════════════════════════════════════
def hdr(t, sub=""):
    return f'<div class="hdr"><h1>{t}</h1><p>{sub}</p></div>'

def mcards(items):
    h = ""
    for it in items:
        v, l = it[0], it[1]; c = it[2] if len(it) > 2 else "var(--accent)"
        h += f'<div class="mc"><div class="v" style="color:{c}">{v}</div><div class="l">{l}</div></div>'
    return f'<div class="mx">{h}</div>'

def src_b(t):
    return '<span class="b-off">OFFICIEL</span>' if t == "OFFICIEL" else '<span class="b-can">CANONIQUE</span>'

def cov_bar(pct, w="100%"):
    pct = min(max(pct, 0), 100)
    c = "#66bb6a" if pct >= 95 else ("#ffa726" if pct >= 75 else "#ef5350")
    return f'<div class="co" style="width:{w}"><div class="ci" style="width:{pct}%;background:{c}"></div></div>'

def prog_bar(pct, label="", status="done"):
    pct = min(max(pct, 0), 100)
    cls = {"done": "prog-fill-done", "wip": "prog-fill-wip", "wait": "prog-fill-wait"}.get(status, "prog-fill-wait")
    return (f'<div class="prog-wrap">'
            f'<div class="prog-label"><span>{label}</span><span style="color:{"#66bb6a" if status=="done" else "#ffa726" if status=="wip" else "#546e7a"}">{pct:.0f}%</span></div>'
            f'<div class="prog-bar"><div class="prog-fill {cls}" style="width:{pct}%"></div></div></div>')

def gate_i(s):
    if s == "PASS": return "✅", "gp"
    if s == "FAIL": return "❌", "gf"
    return "⏳", "gu"

def lvls_for(edu, cid):
    return sorted([l for l in edu["levels"] if l["cycle_id"] == cid], key=lambda x: x["order"])

def subjs_for(edu, lc):
    return [s for s in edu["subjects"] if s["level_code"] == lc]

def exam_for(exams, lc):
    return next((e for e in exams if e.get("level_code") == lc), None)

def cyc_cls(cid):
    return {"CYCLE_HS": "cyc-hs", "CYCLE_PREU": "cyc-preu", "CYCLE_UNI": "cyc-uni"}.get(cid, "")

# ═══════════════════════════════════════════════════════════════════
# 4. CEO STRIP
# ═══════════════════════════════════════════════════════════════════
def render_ceo_strip(cap, rd):
    m = cap["A_METADATA"]
    chk = load_art(rd, "CHK_REPORT.json") or {}
    ces = load_art(rd, "CES_State.json")
    sat = load_art(rd, "SaturationReport.json") or {}
    pc = sum(1 for v in chk.values() if v == "PASS")
    tc = len(chk) if chk else 0

    seal_dot = "dot-green" if m["status"] == "SEALED" else "dot-red"
    ces_dot = "dot-green" if ces else "dot-orange"
    gates_dot = "dot-green" if pc == tc and tc > 0 else ("dot-red" if any(v == "FAIL" for v in chk.values()) else "dot-orange")

    cov_avg = ces.get("coverage_avg", 0) if ces else 0
    orph_total = ces.get("orphans_total", 0) if ces else 0
    saturated = sum(1 for v in sat.values() if isinstance(v, dict) and v.get("status") == "SATURATED")

    chips = (
        f'<span class="ceo-chip"><span class="dot {seal_dot}"></span> CAP {m["status"]}</span>'
        f'<span class="ceo-chip"><span class="dot {ces_dot}"></span> CES {"ACTIVE" if ces else "NO RUN"}</span>'
        f'<span class="ceo-chip"><span class="dot {gates_dot}"></span> Gates {pc}/{tc if tc else "?"}</span>'
        f'<span class="ceo-chip"><span class="dot dot-blue"></span> Cov. {cov_avg:.0f}%</span>'
        f'<span class="ceo-chip"><span class="dot {"dot-green" if orph_total==0 else "dot-red"}"></span> Orph. {orph_total}</span>'
        f'<span class="ceo-chip"><span class="dot dot-blue"></span> Saturés {saturated}/{len(sat) if sat else "?"}</span>'
        f'<span class="ceo-chip" style="margin-left:auto;color:#546e7a">🌍 {m["country_code"]} · K{m["kernel_version"]}</span>'
    )
    st.markdown(f'<div class="ceo-strip">{chips}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 5. WIZARD (empty state)
# ═══════════════════════════════════════════════════════════════════
def render_wizard(rd, has_cap, has_ces):
    s1 = "wiz-done" if has_cap else "wiz-active"
    s2 = "wiz-done" if has_ces else ("wiz-active" if has_cap else "wiz-wait")
    s3 = "wiz-active" if has_cap and has_ces else "wiz-wait"

    st.markdown(f"""<div class="wiz">
        <h2>🔒 SMAXIA World Wide Admin</h2>
        <p style="color:#78909c;font-size:0.85rem">Configurez votre environnement pour commencer</p>
        <div class="wiz-steps">
            <div class="wiz-step"><div class="wiz-num {s1}">{"✓" if has_cap else "1"}</div><div class="wiz-label">CAP scellé</div></div>
            <div class="wiz-step"><div class="wiz-num {s2}">{"✓" if has_ces else "2"}</div><div class="wiz-label">CES Run</div></div>
            <div class="wiz-step"><div class="wiz-num {s3}">3</div><div class="wiz-label">Dashboard</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    if not has_cap:
        st.markdown(hdr("📋 Étape 1 — Charger un CAP scellé",
            "Placez CAP_*.json dans le run directory ou uploadez-le via la sidebar"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="cc" style="padding:1rem">
                <b style="color:#e3f2fd">📁 Via fichier</b>
                <div style="color:#78909c;font-size:0.82rem;margin-top:0.3rem">
                    Créez <code>{rd or './runs/latest'}/</code><br>
                    Déposez <code>CAP_FR.json</code><br>
                    Rafraîchissez la page
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="cc" style="padding:1rem">
                <b style="color:#e3f2fd">📤 Via upload</b>
                <div style="color:#78909c;font-size:0.82rem;margin-top:0.3rem">
                    Utilisez l'uploader dans la sidebar<br>
                    Le CAP sera sauvegardé dans le run dir
                </div>
            </div>""", unsafe_allow_html=True)
    elif not has_ces:
        st.markdown(hdr("🚀 Étape 2 — Lancer CES HARVEST",
            "Le CAP est chargé. Lancez E1 COLLECT pour générer les artefacts CES."), unsafe_allow_html=True)
        st.info("Quand les artefacts CES seront générés dans le run directory, le Dashboard se remplira automatiquement.")

# ═══════════════════════════════════════════════════════════════════
# 6. QC GENERATION PROGRESS (NEW — CEO killer feature)
# ═══════════════════════════════════════════════════════════════════
def render_qc_progress(cap, rd):
    """Show QC/FRT/ARI/TRIGGERS generation progress per subject."""
    edu = cap["B_EDUCATION_SYSTEM"]
    qc_idx = load_art(rd, "QC_Index.json") or {}
    sat = load_art(rd, "SaturationReport.json") or {}

    if not qc_idx:
        return  # No CES data

    st.markdown("#### ⚡ Progression Génération QC/FRT/ARI/TRIGGERS")

    for cycle in edu["cycles"]:
        lvls = lvls_for(edu, cycle["cycle_id"])
        if not lvls:
            continue
        st.markdown(f'<div class="cyc {cyc_cls(cycle["cycle_id"])}">{cycle["cycle_name_local"]}</div>', unsafe_allow_html=True)

        for lv in lvls:
            subs = subjs_for(edu, lv["level_code"])
            for su in subs:
                total_ch = su["chapter_count"]
                if total_ch == 0:
                    continue

                # Calculate progress for this subject
                ch_done = 0; ch_wip = 0; total_qc = 0; total_qi = 0
                for ch in su["chapters"]:
                    ck = f"{su['subject_id']}_CH{str(ch['chapter_number']).zfill(2)}"
                    qd = qc_idx.get(ck, {})
                    sd = sat.get(ck, {})
                    if sd.get("status") == "SATURATED":
                        ch_done += 1
                    elif qd.get("qc_count", 0) > 0:
                        ch_wip += 1
                    total_qc += qd.get("qc_count", 0)
                    total_qi += qd.get("qi_count", 0)

                pct = (ch_done / total_ch * 100) if total_ch > 0 else 0
                status = "done" if ch_done == total_ch else ("wip" if ch_done + ch_wip > 0 else "wait")
                status_txt = "✅ 100%" if status == "done" else (f"🔄 {ch_done}/{total_ch}" if status == "wip" else "⏳")
                label = f"{lv['level_name_local']} — {su['subject_name_local']}  ·  QC:{total_qc}  Qi:{total_qi}  {status_txt}"

                st.markdown(prog_bar(pct, label, status), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 7. PAGES
# ═══════════════════════════════════════════════════════════════════

# ──── DASHBOARD ────
def pg_dash(cap, rd):
    m = cap["A_METADATA"]; edu = cap["B_EDUCATION_SYSTEM"]
    exams = cap["E_EXAMS_CONCOURS"]["exams_and_contests"]
    tot_ch = sum(s["chapter_count"] for s in edu["subjects"])

    st.markdown(hdr(
        f"🌍 SMAXIA World Wide — {m['country_name_local']}",
        f"Kernel {m['kernel_version']} • {m['source_doctrine']} • CAP {m['version']}"
    ), unsafe_allow_html=True)

    st.markdown(mcards([
        (len(edu["cycles"]), "Cycles"), (m["total_classes"], "Classes"),
        (m["total_subjects_go"], "Matières GO"), (tot_ch, "Chapitres"), (len(exams), "Examens"),
    ]), unsafe_allow_html=True)

    # Cycles
    c1, c2 = st.columns([3, 2])
    with c1:
        for c in edu["cycles"]:
            lv = lvls_for(edu, c["cycle_id"])
            su = [s for l in lv for s in subjs_for(edu, l["level_code"])]
            ch = sum(s["chapter_count"] for s in su)
            st.markdown(f'<div class="cyc {cyc_cls(c["cycle_id"])}">'
                        f'{c["cycle_name_local"]} — {len(lv)} classes · {len(su)} mat. · {ch} chap.</div>',
                        unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="sha">📋 {m["cap_fingerprint_sha256"]}</div>', unsafe_allow_html=True)
        seal = '<span class="b-seal">✓ SEALED</span>' if m["status"] == "SEALED" else '<span class="gf">⚠ NOT SEALED</span>'
        off = sum(1 for s in edu["subjects"] if s.get("source_type") == "OFFICIEL")
        can = sum(1 for s in edu["subjects"] if s.get("source_type") == "CANONIQUE")
        st.markdown(f"{seal} &nbsp; {src_b('OFFICIEL')} **{off}** &nbsp; {src_b('CANONIQUE')} **{can}**", unsafe_allow_html=True)

    # CES section
    st.markdown("---")
    ces = load_art(rd, "CES_State.json")
    if ces:
        st.markdown(mcards([
            (ces.get("pdf_count", "—"), "PDFs"), (ces.get("pairs_count", "—"), "Paires"),
            (ces.get("qi_count", "—"), "Qi"), (ces.get("qc_count", "—"), "QC"),
            (f"{ces.get('coverage_avg', 0):.0f}%", "Couverture", "#66bb6a" if ces.get("coverage_avg",0)>=95 else "#ffa726"),
            (ces.get("orphans_total", "—"), "Orphelins", "#ef5350" if ces.get("orphans_total",0)>0 else "#66bb6a"),
        ]), unsafe_allow_html=True)

        # QC Generation Progress
        render_qc_progress(cap, rd)

        # Timeline charts
        tl = load_art(rd, "CoverageTimeline.json")
        if tl and isinstance(tl, list) and len(tl) > 0:
            import pandas as pd
            df = pd.DataFrame(tl)
            tc1, tc2 = st.columns(2)
            with tc1:
                if "date" in df.columns and "coverage" in df.columns:
                    st.markdown("##### 📊 Couverture")
                    st.line_chart(df.set_index("date")["coverage"], use_container_width=True)
            with tc2:
                if "date" in df.columns and "orphans" in df.columns:
                    st.markdown("##### 📉 Orphelins")
                    st.line_chart(df.set_index("date")["orphans"], use_container_width=True)

        # Top risks
        sat = load_art(rd, "SaturationReport.json") or {}
        qc_idx = load_art(rd, "QC_Index.json") or {}
        risks = []
        for ck, qd in qc_idx.items():
            sd = sat.get(ck, {})
            if sd.get("status") != "SATURATED":
                risks.append((ck, qd.get("coverage", 0), qd.get("orphans", 0)))
        if risks:
            risks.sort(key=lambda x: (x[1], -x[2]))  # coverage asc, orphans desc
            st.markdown("#### ⚠️ Top 10 Chapitres à Risque")
            for ck, cv, orph in risks[:10]:
                st.markdown(f'<div class="cc" style="display:flex;align-items:center;gap:0.8rem">'
                            f'<code style="color:var(--accent);min-width:200px">{ck}</code>'
                            f'<div style="flex:1">{cov_bar(cv)}</div>'
                            f'<span style="font-size:0.78rem">{cv}%</span>'
                            f'<span style="color:#ef5350;font-size:0.78rem">🔴 {orph} orph.</span>'
                            f'</div>', unsafe_allow_html=True)
    else:
        st.info("⏳ **CES HARVEST non lancé** — Lancez E1 COLLECT pour activer le monitoring.")

# ──── CAP EXPLORER ────
def pg_cap(cap, rd):
    m = cap["A_METADATA"]; edu = cap["B_EDUCATION_SYSTEM"]
    exams = cap["E_EXAMS_CONCOURS"]["exams_and_contests"]
    tot = sum(s["chapter_count"] for s in edu["subjects"])
    st.markdown(hdr("📚 CAP Explorer", f"{m['total_classes']} classes · {m['total_subjects_go']} mat. · {tot} chap."), unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["🏫 Niveaux", "📘 Matières", "📖 Chapitres", "🎓 Examens", "🔗 Sources"])

    with t1:
        for c in edu["cycles"]:
            lv = lvls_for(edu, c["cycle_id"])
            st.markdown(f'<div class="cyc {cyc_cls(c["cycle_id"])}">{c["cycle_name_local"]} — {len(lv)} classes</div>', unsafe_allow_html=True)
            for l in lv:
                su = subjs_for(edu, l["level_code"]); ch = sum(s["chapter_count"] for s in su)
                ex = exam_for(exams, l["level_code"])
                eh = f' <span class="b-exam">🎯 {ex["exam"]["exam_name"]}</span>' if ex and ex.get("exam") else ""
                ct = " ".join(f'<span class="b-conc">🏆#{c["rank"]} {c["name"]}</span>' for c in (ex.get("contests_top") or [])[:5]) if ex else ""
                st.markdown(f'<div class="cc"><b style="color:#e3f2fd">{l["level_name_local"]}</b>'
                            f' <span style="color:#546e7a;font-size:0.75rem">({l["level_code"]})</span>'
                            f'<div style="color:#546e7a;font-size:0.7rem">{l.get("voie","")}</div>'
                            f'<div style="color:var(--blue);font-size:0.76rem">📘 {len(su)} mat. · 📖 {ch} chap.</div>'
                            f'{eh}{f"<div>{ct}</div>" if ct else ""}</div>', unsafe_allow_html=True)

    with t2:
        q = st.text_input("🔍 Filtrer", key="s_s2")
        for su in edu["subjects"]:
            ln = next((l["level_name_local"] for l in edu["levels"] if l["level_code"] == su["level_code"]), su["level_code"])
            if q and q.lower() not in f"{ln} {su['subject_name_local']}".lower(): continue
            st.markdown(f'<div class="cc"><b style="color:#e3f2fd">{ln}</b> — {su["subject_name_local"]} '
                        f'({su["chapter_count"]} chap.) {src_b(su.get("source_type",""))}'
                        f'<div style="color:#546e7a;font-size:0.66rem">{su.get("source_ref","")}</div></div>', unsafe_allow_html=True)

    with t3:
        q2 = st.text_input("🔍 Filtrer", key="c_s2")
        for su in edu["subjects"]:
            ln = next((l["level_name_local"] for l in edu["levels"] if l["level_code"] == su["level_code"]), su["level_code"])
            chs = [ch for ch in su["chapters"]
                   if not q2 or q2.lower() in ch["chapter_name"].lower() or q2.lower() in su["subject_name_local"].lower() or q2.lower() in ln.lower()]
            if not chs: continue
            with st.expander(f"**{ln} — {su['subject_name_local']}** ({len(su['chapters'])} chap.)"):
                for ch in chs:
                    st.markdown(f'<span style="color:var(--accent);font-weight:700;font-family:JetBrains Mono">'
                                f'{str(ch["chapter_number"]).zfill(2)}</span> {ch["chapter_name"]}', unsafe_allow_html=True)

    with t4:
        for c in edu["cycles"]:
            st.markdown(f'<div class="cyc {cyc_cls(c["cycle_id"])}">{c["cycle_name_local"]}</div>', unsafe_allow_html=True)
            for l in lvls_for(edu, c["cycle_id"]):
                ex = exam_for(exams, l["level_code"])
                if not ex: continue
                eh = f'<span class="b-exam">🎯 {ex["exam"]["exam_name"]}</span>' if ex.get("exam") else ""
                ct = " ".join(f'<span class="b-conc">🏆#{c["rank"]} {c["name"]}</span>' for c in (ex.get("contests_top") or []))
                st.markdown(f'<div class="cc"><b style="color:#e3f2fd">{l["level_name_local"]}</b> {eh}<div>{ct}</div></div>', unsafe_allow_html=True)

    with t5:
        src = cap["C_HARVEST_SOURCES"]
        for s in src["sources"]:
            pc = {"A": "#66bb6a", "B": "#ffa726", "C": "#ef5350"}.get(s["proof"], "#fff")
            st.markdown(f'<div class="cc"><b style="color:#e3f2fd">{s["source_id"]}</b> — {s["domain"]}'
                        f'<div style="font-size:0.76rem">🏅 {s["authority_score"]} · Proof: <span style="color:{pc};font-weight:700">{s["proof"]}</span></div></div>', unsafe_allow_html=True)

# ──── CES MONITOR ────
def pg_ces(cap, rd):
    st.markdown(hdr("🚀 CES Monitor", "Pipeline · Artefacts"), unsafe_allow_html=True)
    ces = load_art(rd, "CES_State.json")
    expected = ["CES_State.json","QC_Index.json","Qi_Index.json","CoverageReport.json","CoverageTimeline.json",
                "SaturationReport.json","QuarantineLedger.json","CHK_REPORT.json","SealReport.json",
                "DeterminismReport_3runs.json","PairingReport_G2.json","OcrReport_G3.json","AtomTrace_G4.json","DedupReport_G5.json"]
    if ces: st.json(ces)
    else: st.warning("⏳ Aucun artefact CES.")
    st.markdown("#### 📦 Artefacts")
    for f in expected:
        st.markdown(f"&nbsp; {'✅' if art_exists(rd,f) else '❌'} `{f}`", unsafe_allow_html=True)

# ──── CHAPTERS & QC + STOP_RULE ────
def pg_chapters(cap, rd):
    st.markdown(hdr("📈 Chapitres & QC", "Couverture · STOP_RULE · Saturation"), unsafe_allow_html=True)
    edu = cap["B_EDUCATION_SYSTEM"]
    c1, c2, c3 = st.columns(3)
    cn = {c["cycle_id"]: c["cycle_name_local"] for c in edu["cycles"]}
    with c1: sc = st.selectbox("Cycle", ["Tous"]+list(cn.values()), key="fc")
    scid = next((k for k,v in cn.items() if v==sc), None)
    lf = edu["levels"] if not scid else [l for l in edu["levels"] if l["cycle_id"]==scid]
    ln = {l["level_code"]: l["level_name_local"] for l in lf}
    with c2: sl = st.selectbox("Niveau", ["Tous"]+list(ln.values()), key="fl")
    slc = next((k for k,v in ln.items() if v==sl), None)
    sf = edu["subjects"]
    if slc: sf = [s for s in sf if s["level_code"]==slc]
    elif scid: sf = [s for s in sf if s["level_code"] in [l["level_code"] for l in lf]]
    sn = {s["subject_id"]: s["subject_name_local"] for s in sf}
    with c3: ss = st.selectbox("Matière", ["Toutes"]+list(sn.values()), key="fs")
    ssid = next((k for k,v in sn.items() if v==ss), None)
    if ssid: sf = [s for s in sf if s["subject_id"]==ssid]

    qc_idx = load_art(rd, "QC_Index.json") or {}
    sat = load_art(rd, "SaturationReport.json") or {}

    # Sort option
    sort_by = st.selectbox("Trier par", ["Défaut", "Couverture ↑", "Orphelins ↓", "Statut"], key="sort_ch")

    for su in sf:
        lname = next((l["level_name_local"] for l in edu["levels"] if l["level_code"]==su["level_code"]), "")
        chapters = su["chapters"]

        # Enrich with data for sorting
        enriched = []
        for ch in chapters:
            ck = f"{su['subject_id']}_CH{str(ch['chapter_number']).zfill(2)}"
            qd = qc_idx.get(ck, {}); sd = sat.get(ck, {})
            enriched.append((ch, ck, qd, sd))

        if sort_by == "Couverture ↑":
            enriched.sort(key=lambda x: x[2].get("coverage", 0))
        elif sort_by == "Orphelins ↓":
            enriched.sort(key=lambda x: -x[2].get("orphans", 0))
        elif sort_by == "Statut":
            enriched.sort(key=lambda x: (0 if x[3].get("status")=="SATURATED" else 1))

        with st.expander(f"**{lname} — {su['subject_name_local']}** ({su['chapter_count']} chap.)"):
            for ch, ck, qd, sd in enriched:
                qi = qd.get("qi_count", 0); qc = qd.get("qc_count", 0)
                cv = qd.get("coverage", 0); orph = qd.get("orphans", 0)
                dqc = sd.get("delta_qc", "—"); on = sd.get("orphans_new", "—"); ss_ = sd.get("status", "⏳")
                ca, cb, cc_, cd, ce = st.columns([3.5, 0.7, 0.7, 1.5, 1.2])
                with ca:
                    st.markdown(f'<span style="color:var(--accent);font-weight:700">{str(ch["chapter_number"]).zfill(2)}</span> {ch["chapter_name"]}', unsafe_allow_html=True)
                with cb: st.metric("QC", qc)
                with cc_: st.metric("Qi", qi)
                with cd:
                    st.markdown(cov_bar(cv), unsafe_allow_html=True)
                    st.caption(f"{cv}% · Orph:{orph}")
                with ce:
                    if ss_ == "SATURATED": st.markdown('<span class="sat-ok">✅ SATURÉ</span>', unsafe_allow_html=True)
                    elif ss_ == "CONTINUE": st.markdown('<span class="sat-cont">🔄 CONTINUE</span>', unsafe_allow_html=True)
                    else: st.markdown('<span class="gu">⏳</span>', unsafe_allow_html=True)
                    if sd: st.caption(f"ΔQC={dqc} · New={on}")

# ──── QC DETAIL ────
def pg_qc_detail(cap, rd):
    st.markdown(hdr("🔬 QC Detail", "FRT · ARI · TRIGGERS — Opaque (IP)"), unsafe_allow_html=True)
    qc_idx = load_art(rd, "QC_Index.json")
    if not qc_idx:
        st.info("⏳ Aucune QC — En attente CES HARVEST.")
        return
    all_qcs = [(ck, qid) for ck, d in qc_idx.items() for qid in d.get("qc_ids", [])]
    if not all_qcs:
        st.warning("QC_Index chargé mais 0 qc_id.")
        return
    sc = st.selectbox("Chapitre", list(set(ck for ck,_ in all_qcs)), key="qcd_c")
    cq = [q for c,q in all_qcs if c==sc]
    sq = st.selectbox("QC", cq, key="qcd_q")
    if sq:
        d = load_qc_detail_file(rd, sq)
        if d:
            c1, c2, c3 = st.columns(3)
            for col, key, color, lbl in [(c1,"frt","#42a5f5","FRT"),(c2,"ari","#66bb6a","ARI"),(c3,"triggers","#ffa726","TRIGGERS")]:
                with col:
                    obj = d.get(key, {})
                    if isinstance(obj, list):
                        ids = "<br>".join(f'<code>{t.get("id","?")}</code>' for t in obj[:5])
                        st.markdown(f'<div class="cc" style="border-left:3px solid {color}"><b style="color:{color}">{lbl} ({len(obj)})</b><br>{ids}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="cc" style="border-left:3px solid {color}"><b style="color:{color}">{lbl}</b><br>ID: <code>{obj.get("id","N/A")}</code><br>SHA: <code style="font-size:0.6rem">{obj.get("sha256","N/A")[:24]}...</code></div>', unsafe_allow_html=True)
            qi_ids = d.get("qi_children", [])
            if qi_ids:
                st.markdown(f"##### Qi ({len(qi_ids)})")
                for qiid in qi_ids[:15]:
                    qd = load_qi_text_file(rd, qiid)
                    prev = qd.get("text","")[:100]+"..." if qd else "⏳"
                    st.markdown(f'<div class="cc"><code style="color:var(--accent)">{qiid}</code> <span style="color:#b0bec5;font-size:0.76rem">{prev}</span></div>', unsafe_allow_html=True)
        else:
            st.warning(f"`QC_Details/{sq}.json` non trouvé.")

# ──── TESTS ────
def pg_tests(cap, rd):
    st.markdown(hdr("🧪 Tests", "Orphelins · Couverture"), unsafe_allow_html=True)
    ta, tb = st.tabs(["Test A — Sujet traité", "Test B — Sujet nouveau"])
    with ta:
        st.markdown("**Vérifier Qi_orphelin = 0 pour un sujet déjà traité.**")
        doc_id = st.text_input("doc_id", key="ta_d")
        if doc_id and st.button("▶ Charger", key="ta_b"):
            mr = load_art(rd, f"MappingReport_{doc_id}.json")
            if mr:
                orph = mr.get("orphans", [])
                st.markdown(mcards([
                    (mr.get("qi_total",0), "Qi total"), (mr.get("qi_mapped",0), "Mappées"),
                    (len(orph), "Orphelines", "#ef5350" if orph else "#66bb6a"),
                    (f"{mr.get('coverage',0):.0f}%", "Couverture"),
                ]), unsafe_allow_html=True)
                if orph:
                    st.error(f"❌ {len(orph)} orphelines")
                    for o in orph[:15]: st.markdown(f'`{o.get("qi_id","?")}` — {o.get("text_preview","")[:80]}')
                else:
                    st.success("✅ Qi_orphelin = 0")
            else:
                st.error(f"❌ `MappingReport_{doc_id}.json` non trouvé")
    with tb:
        st.markdown("**Upload PDF → TestRequest (traitement asynchrone).**")
        up = st.file_uploader("📄 PDF", type=["pdf"], key="tb_u")
        if up and st.button("▶ Créer TestRequest", key="tb_b"):
            if not rd or not os.path.isdir(rd):
                st.error("❌ Run dir invalide"); return
            pdf_bytes = up.read()
            sha = hashlib.sha256(pdf_bytes).hexdigest()
            req = {"type":"TEST_B","filename":up.name,"pdf_sha256":sha,
                   "size":len(pdf_bytes),"at":datetime.datetime.utcnow().isoformat()+"Z","status":"PENDING"}
            p = os.path.join(rd, f"TestRequest_{sha[:12]}.json")
            with open(p,"w") as f: json.dump(req, f, indent=2, ensure_ascii=False)
            os.makedirs(os.path.join(rd,"test_pdfs"), exist_ok=True)
            with open(os.path.join(rd,"test_pdfs",up.name),"wb") as f: f.write(pdf_bytes)
            st.success(f"✅ TestRequest créé"); st.json(req)

# ──── GATES ────
def pg_gates(cap, rd):
    m = cap["A_METADATA"]
    st.markdown(hdr("🔐 Gates & Intégrité", f"Kernel {m['kernel_version']}"), unsafe_allow_html=True)
    st.markdown(f'<div class="sha">{m["cap_fingerprint_sha256"]}</div>', unsafe_allow_html=True)
    chk = load_art(rd, "CHK_REPORT.json") or {}
    if not chk:
        st.markdown('<div class="wb">⚠️ <code>CHK_REPORT.json</code> absent — toutes gates = UNKNOWN. B4: sans artefact = FAIL.</div>', unsafe_allow_html=True)
    else:
        pc=fc=uc=0
        for n, s in chk.items():
            ic, cl = gate_i(s)
            st.markdown(f'<div class="gr">{ic} <span class="{cl}">{n}</span> — {s}</div>', unsafe_allow_html=True)
            if s=="PASS": pc+=1
            elif s=="FAIL": fc+=1
            else: uc+=1
        st.markdown(mcards([(pc,"PASS","#66bb6a"),(fc,"FAIL","#ef5350"),(uc,"?","#ffa726")]), unsafe_allow_html=True)
    det = load_art(rd, "DeterminismReport_3runs.json")
    if det: st.json(det)
    else: st.info("⏳ DeterminismReport non disponible.")

# ──── QUARANTINE ────
def pg_quarantine(cap, rd):
    st.markdown(hdr("🔶 Quarantine", "Items en quarantaine"), unsafe_allow_html=True)
    ql = load_art(rd, "QuarantineLedger.json")
    if not ql: st.info("⏳ 0 item."); return
    if isinstance(ql, list):
        import pandas as pd; st.dataframe(pd.DataFrame(ql), use_container_width=True)
    else: st.json(ql)

# ──── MAPPING ────
def pg_mapping(cap, rd):
    st.markdown(hdr("🗺️ Mapping Qi → QC", "Couverture par chapitre"), unsafe_allow_html=True)
    cov = load_art(rd, "CoverageReport.json")
    if not cov: st.info("⏳ En attente CES."); return
    for ck, d in sorted(cov.items()):
        cv = d.get("coverage",0); orph = d.get("orphans",0)
        st.markdown(f'<div class="cc" style="display:flex;align-items:center;gap:0.6rem">'
                    f'<code style="color:var(--accent);min-width:180px">{ck}</code>'
                    f'<div style="flex:1">{cov_bar(cv)}</div>'
                    f'<span style="font-size:0.76rem">{cv}% · {orph} orph.</span></div>', unsafe_allow_html=True)

# ──── DEV NOTES ────
def pg_dev():
    st.markdown(hdr("🔒 Dev Notes", "Verrouillé — Aucun claim sans KPI scellé"), unsafe_allow_html=True)
    st.markdown('<div class="wb">⚠️ <b>LOCKED</b> — Claims débloqués uniquement avec <code>KPI_Report.json</code> scellé + panel GO.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="SMAXIA World Wide Admin", page_icon="🌍",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### 🌍 SMAXIA Admin")

        # Run selector
        runs_base = "./runs"
        available = scan_runs(runs_base)
        if available:
            sel_run = st.selectbox("📂 Run", available, key="sel_r")
            rd = os.path.join(runs_base, sel_run)
        else:
            rd = os.path.join(runs_base, "latest")

        with st.expander("⚙️ Avancé"):
            custom_base = st.text_input("Runs base", value=runs_base, key="cb")
            if custom_base != runs_base:
                av2 = scan_runs(custom_base)
                if av2:
                    sel2 = st.selectbox("Run", av2, key="sel_r2")
                    rd = os.path.join(custom_base, sel2)
            custom_path = st.text_input("Chemin custom", value="", key="cp")
            if custom_path: rd = custom_path

        # ── CAP LOADING (3 sources: session_state > filesystem > upload) ──
        cap = None

        # Source 1: session_state (persists on Streamlit Cloud across reruns)
        if "cap_data" in st.session_state and st.session_state["cap_data"]:
            cap = st.session_state["cap_data"]

        # Source 2: filesystem (local mode)
        if cap is None:
            caps = scan_caps(rd)
            if caps:
                if len(caps) > 1:
                    sel_cap = st.selectbox("🌍 Pays", [os.path.basename(c) for c in caps], key="sc")
                    cap = load_json(os.path.join(rd, sel_cap))
                else:
                    cap = load_json(caps[0])
                if cap:
                    st.session_state["cap_data"] = cap

        # Source 3: upload
        has_cap = cap is not None
        has_ces = art_exists(rd, "CES_State.json")
        cap_badge = "✅ CAP" if has_cap else "❌ CAP"
        ces_badge = "✅ CES" if has_ces else "⏳ CES"
        st.markdown(f"<div style='display:flex;gap:0.5rem;margin:0.3rem 0'>"
                    f"<span style='font-size:0.75rem;font-weight:700'>{cap_badge}</span>"
                    f"<span style='font-size:0.75rem;font-weight:700'>{ces_badge}</span></div>",
                    unsafe_allow_html=True)

        cap_up = st.file_uploader("📤 Upload CAP JSON", type=["json"], key="cup")
        if cap_up:
            try:
                cap_bytes = cap_up.read()
                new_cap = json.loads(cap_bytes.decode("utf-8"))
                st.session_state["cap_data"] = new_cap
                cap = new_cap
                has_cap = True
                # Also try to save to disk (works local, may fail cloud — that's OK)
                try:
                    os.makedirs(rd, exist_ok=True)
                    with open(os.path.join(rd, cap_up.name), "wb") as f:
                        f.write(cap_bytes)
                except Exception:
                    pass  # Cloud filesystem — expected
                st.success("✅ CAP chargé !")
            except Exception as e:
                st.error(f"❌ JSON invalide: {e}")

        if cap:
            m = cap.get("A_METADATA", {})
            flag = {"FR":"🇫🇷","MA":"🇲🇦","SN":"🇸🇳","CI":"🇨🇮","TN":"🇹🇳","DZ":"🇩🇿","CM":"🇨🇲"}.get(m.get("country_code",""), "🌍")
            st.markdown(f"**{flag} {m.get('country_name_local','')}**")
            st.markdown(f"<span style='font-size:0.72rem;color:#546e7a'>K{m.get('kernel_version','')} · {m.get('version','')}</span>", unsafe_allow_html=True)
            if m.get("status") == "SEALED":
                st.markdown('<span class="b-seal">✓ SEALED</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Grouped navigation
        st.markdown('<div class="nav-section">OVERVIEW</div>', unsafe_allow_html=True)
        page = st.radio("Nav", [
            "📊 Dashboard",
            "🔐 Gates & Intégrité",
        ], key="nav1", label_visibility="collapsed")

        st.markdown('<div class="nav-section">CAP</div>', unsafe_allow_html=True)
        p2 = st.radio("Nav2", [
            "📚 CAP Explorer",
        ], key="nav2", label_visibility="collapsed")

        st.markdown('<div class="nav-section">CES / QC</div>', unsafe_allow_html=True)
        p3 = st.radio("Nav3", [
            "🚀 CES Monitor",
            "📈 Chapitres & QC",
            "🔬 QC Detail",
            "🗺️ Mapping",
            "🔶 Quarantine",
        ], key="nav3", label_visibility="collapsed")

        st.markdown('<div class="nav-section">TESTS</div>', unsafe_allow_html=True)
        p4 = st.radio("Nav4", [
            "🧪 Tests (Orphelins)",
        ], key="nav4", label_visibility="collapsed")

        st.markdown('<div class="nav-section">SYSTEM</div>', unsafe_allow_html=True)
        p5 = st.radio("Nav5", ["🔒 Dev Notes"], key="nav5", label_visibility="collapsed")

    # Determine active page (last clicked radio)
    # Streamlit limitation: multiple radios = all have selection.
    # We use session state to track which was last clicked
    all_pages = {"nav1": page, "nav2": p2, "nav3": p3, "nav4": p4, "nav5": p5}
    active = None
    for k, v in all_pages.items():
        if f"_last_{k}" not in st.session_state:
            st.session_state[f"_last_{k}"] = v
        if st.session_state[f"_last_{k}"] != v:
            active = v
            st.session_state[f"_last_{k}"] = v

    if active is None:
        active = page  # default to Dashboard

    # ── NO CAP ──
    if not cap:
        render_wizard(rd, False, False)
        return

    # ── CEO STRIP ──
    render_ceo_strip(cap, rd)

    # ── WIZARD if no CES ──
    if not has_ces and active == "📊 Dashboard":
        render_wizard(rd, True, False)

    # ── ROUTING ──
    routes = {
        "📊 Dashboard": pg_dash, "📚 CAP Explorer": pg_cap,
        "🚀 CES Monitor": pg_ces, "📈 Chapitres & QC": pg_chapters,
        "🔬 QC Detail": pg_qc_detail, "🗺️ Mapping": pg_mapping,
        "🧪 Tests (Orphelins)": pg_tests, "🔐 Gates & Intégrité": pg_gates,
        "🔶 Quarantine": pg_quarantine, "🔒 Dev Notes": pg_dev,
    }
    if active in routes:
        routes[active](cap, rd) if active != "🔒 Dev Notes" else pg_dev()

if __name__ == "__main__":
    main()
