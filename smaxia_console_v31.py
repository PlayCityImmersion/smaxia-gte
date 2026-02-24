# ═══════════════════════════════════════════════════════════════════
# SMAXIA COMMAND CENTER v4.0 — Ultra Premium CEO Console
# Structure: SMAXIA PIPELINE (4p) + VALIDATION CEO (6p)
# Usage: streamlit run smaxia_command_center_v4.py
# ═══════════════════════════════════════════════════════════════════
import streamlit as st, json, os, glob, datetime, hashlib
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 1. LOADERS
# ═══════════════════════════════════════════════════════════════════
def scan_runs(b="./runs"):
    return sorted([d for d in os.listdir(b) if os.path.isdir(os.path.join(b,d))],reverse=True) if os.path.isdir(b) else []
def load_json(p):
    if p and os.path.exists(p):
        with open(p,"r",encoding="utf-8") as f: return json.load(f)
    return None
def scan_caps(rd):
    return sorted(glob.glob(os.path.join(rd,"CAP_*.json"))) if rd and os.path.isdir(rd) else []
def load_art(rd,n): return load_json(os.path.join(rd,n)) if rd else None
def art_ok(rd,n): return os.path.exists(os.path.join(rd,n)) if rd else False

# ═══════════════════════════════════════════════════════════════════
# 2. CSS
# ═══════════════════════════════════════════════════════════════════
def css():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
:root{--bg:#0a0e27;--bg2:#0f1329;--c:#151933;--bd:rgba(255,255,255,0.06);--t:#d0d0d0;--t2:#6b7b8d;--a:#64b5f6;--bl:#90caf9}
.stApp,[data-testid="stAppViewContainer"],.main .block-container{background:var(--bg)!important;color:var(--t)!important;font-family:'Inter',sans-serif!important}
.main .block-container{padding-top:.5rem!important;max-width:1440px!important}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--bd)!important}
[data-testid="stSidebar"] *{color:#99a8b5!important;font-size:.82rem!important}
[data-testid="stSidebar"] h3{color:var(--bl)!important;font-size:.88rem!important}
.hdr{background:linear-gradient(135deg,#1a237e,#0d47a1 50%,#01579b);padding:.85rem 1.3rem;border-radius:12px;margin-bottom:.7rem;border:1px solid rgba(255,255,255,.1);box-shadow:0 6px 24px rgba(0,0,0,.35)}
.hdr h1{color:#fff!important;font-size:1.2rem;font-weight:800;margin:0;letter-spacing:-.3px}
.hdr p{color:var(--bl);font-size:.72rem;margin:.08rem 0 0}
.mx{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.5rem;margin-bottom:.6rem}
.mc{background:linear-gradient(135deg,#1a1f3e,var(--c));border:1px solid var(--bd);border-radius:10px;padding:.65rem;text-align:center}
.mc .v{font-size:1.5rem;font-weight:800;color:var(--a);font-family:'JetBrains Mono',monospace;letter-spacing:-1px}
.mc .l{font-size:.52rem;color:var(--t2);font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-top:.05rem}
.cyc{padding:.38rem .7rem;border-radius:7px;margin:.25rem 0;font-weight:700;font-size:.8rem;color:#fff}
.cyc-hs{background:linear-gradient(90deg,#1565c0,#0d47a1)}.cyc-pr{background:linear-gradient(90deg,#2e7d32,#1b5e20)}.cyc-un{background:linear-gradient(90deg,#e65100,#bf360c)}
.cc{background:var(--c);border:1px solid var(--bd);border-radius:9px;padding:.45rem .75rem;margin:.18rem 0;transition:border-color .15s}
.cc:hover{border-color:rgba(100,181,246,.25)}
.b-of{display:inline-block;background:#1b5e20;color:#a5d6a7;padding:.05rem .4rem;border-radius:3px;font-size:.58rem;font-weight:700}
.b-ca{display:inline-block;background:#e65100;color:#ffcc80;padding:.05rem .4rem;border-radius:3px;font-size:.58rem;font-weight:700}
.b-se{display:inline-block;background:linear-gradient(135deg,#1b5e20,#2e7d32);color:#fff;padding:.18rem .55rem;border-radius:12px;font-weight:700;font-size:.64rem;letter-spacing:.8px}
.b-ex{display:inline-block;background:#311b92;color:#b39ddb;padding:.05rem .4rem;border-radius:3px;font-size:.6rem;font-weight:600;margin:1px}
.b-co{display:inline-block;background:#1a237e;color:var(--bl);padding:.05rem .4rem;border-radius:3px;font-size:.6rem;font-weight:600;margin:1px}
.sha{background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:.35rem .55rem;font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#7ee787;word-break:break-all;margin:.25rem 0}
.gp{color:#66bb6a;font-weight:700}.gf{color:#ef5350;font-weight:700}.gu{color:#ffa726;font-weight:700}
.co{background:#1a1f3e;border-radius:5px;height:14px;width:100%;overflow:hidden}.ci{height:100%;border-radius:5px;transition:width .3s}
.wb{background:rgba(255,167,38,.08);border:1px solid rgba(255,167,38,.25);border-radius:7px;padding:.4rem .7rem;margin:.35rem 0;font-size:.76rem}
.ns{font-size:.56rem;color:#546e7a;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:.5rem 0 .1rem;padding-left:3px}
.tbl-row{display:flex;align-items:center;padding:.35rem .5rem;border-bottom:1px solid var(--bd);font-size:.78rem}
.tbl-row:hover{background:rgba(100,181,246,.04)}
.tbl-h{font-size:.6rem;color:var(--t2);font-weight:700;letter-spacing:1px;text-transform:uppercase;border-bottom:2px solid var(--bd)}
.qc-card{background:var(--c);border:1px solid var(--bd);border-radius:10px;padding:.7rem .9rem;margin:.4rem 0}
.qc-frt{border-left:3px solid #42a5f5;padding:.3rem .6rem;margin:.2rem 0;background:rgba(66,165,245,.04);border-radius:0 5px 5px 0;font-size:.74rem}
.qc-ari{border-left:3px solid #66bb6a;padding:.3rem .6rem;margin:.2rem 0;background:rgba(102,187,106,.04);border-radius:0 5px 5px 0;font-size:.74rem}
.qc-trg{border-left:3px solid #ef5350;padding:.3rem .6rem;margin:.2rem 0;background:rgba(239,83,80,.04);border-radius:0 5px 5px 0;font-size:.74rem}
.trig-chip{display:inline-block;background:#b71c1c;color:#ffcdd2;padding:.05rem .35rem;border-radius:3px;font-size:.58rem;font-weight:600;margin:1px}
.idea-card{background:var(--c);border-radius:0 9px 9px 0;padding:.6rem .9rem;margin:.3rem 0}
.stTabs [data-baseweb="tab-list"]{gap:3px}.stTabs [data-baseweb="tab"]{background:var(--c)!important;color:var(--bl)!important;border-radius:7px 7px 0 0!important;border:1px solid var(--bd)!important;font-size:.78rem!important}
.stTabs [aria-selected="true"]{background:#1a237e!important;color:#fff!important}
.streamlit-expanderHeader{background:var(--c)!important;color:#e3f2fd!important;border-radius:7px!important;font-size:.82rem!important}
hr{border-color:var(--bd)!important}
.stSelectbox label,.stTextInput label{color:var(--bl)!important;font-size:.74rem!important}
</style>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 3. HELPERS
# ═══════════════════════════════════════════════════════════════════
def H(t,s=""): return f'<div class="hdr"><h1>{t}</h1><p>{s}</p></div>'
def MC(items):
    h="".join(f'<div class="mc"><div class="v" style="color:{it[2] if len(it)>2 else "var(--a)"}">{it[0]}</div><div class="l">{it[1]}</div></div>' for it in items)
    return f'<div class="mx">{h}</div>'
def CB(p,w="100%"):
    p=min(max(p,0),100);c="#66bb6a" if p>=95 else("#ffa726" if p>=75 else "#ef5350")
    return f'<div class="co" style="width:{w}"><div class="ci" style="width:{p}%;background:{c}"></div></div>'
def SB(t): return '<span class="b-of">OFFICIEL</span>' if t=="OFFICIEL" else '<span class="b-ca">CANONIQUE</span>'
def LF(e,cid): return sorted([l for l in e["levels"] if l["cycle_id"]==cid],key=lambda x:x["order"])
def SF(e,lc): return [s for s in e["subjects"] if s["level_code"]==lc]
def EF(ex,lc): return next((e for e in ex if e.get("level_code")==lc),None)
def CC(c): return{"CYCLE_HS":"cyc-hs","CYCLE_PREU":"cyc-pr","CYCLE_UNI":"cyc-un"}.get(c,"")
def GI(s):
    if s=="PASS":return"✅","gp"
    if s=="FAIL":return"❌","gf"
    return"⏳","gu"

# ═══════════════════════════════════════════════════════════════════
# 4. PAGES — SMAXIA PIPELINE
# ═══════════════════════════════════════════════════════════════════
def pg_dashboard(cap,rd):
    m=cap["A_METADATA"];edu=cap["B_EDUCATION_SYSTEM"];ex=cap["E_EXAMS_CONCOURS"]["exams_and_contests"]
    tot=sum(s["chapter_count"] for s in edu["subjects"])
    off=sum(1 for s in edu["subjects"] if s.get("source_type")=="OFFICIEL")
    can=sum(1 for s in edu["subjects"] if s.get("source_type")=="CANONIQUE")
    fl={"FR":"🇫🇷","MA":"🇲🇦","SN":"🇸🇳","CI":"🇨🇮","TN":"🇹🇳","DZ":"🇩🇿","CM":"🇨🇲"}.get(m.get("country_code",""),"🌍")
    st.markdown(H(f'{fl} SMAXIA — CAP {m["country_name_local"]} Scellé',
        f'Country Academic Pack • Kernel {m["kernel_version"]} • {m["source_doctrine"]} • Zéro Invention'),unsafe_allow_html=True)
    st.markdown(MC([(len(edu["cycles"]),"Cycles"),(m["total_classes"],"Classes"),(m["total_subjects_go"],"Matières GO"),
        (tot,"Chapitres"),(len(ex),"Examens")]),unsafe_allow_html=True)
    c1,c2=st.columns([3,2])
    with c1:
        st.markdown("#### Répartition par Cycle")
        for c in edu["cycles"]:
            lv=LF(edu,c["cycle_id"]);su=[s for l in lv for s in SF(edu,l["level_code"])];ch=sum(s["chapter_count"] for s in su)
            st.markdown(f'<div class="cyc {CC(c["cycle_id"])}">{c["cycle_name_local"]} ({c["cycle_name_en"]}) — {len(lv)} classes · {len(su)} matières · {ch} chapitres</div>',unsafe_allow_html=True)
    with c2:
        st.markdown("#### 🔐 Intégrité")
        st.markdown(f'<div class="sha">📋 {m["cap_fingerprint_sha256"]}</div>',unsafe_allow_html=True)
        st.markdown(f'📜 Doctrine: **{m["source_doctrine"]}**',unsafe_allow_html=True)
        st.markdown(f'🏛️ Juridiction: **{m["jurisdiction_model"]}**')
        st.markdown(f'📅 Année: **{m["academic_year"]}**')
        st.markdown(f'{SB("OFFICIEL")} **{off}** matières &nbsp; {SB("CANONIQUE")} **{can}** matières',unsafe_allow_html=True)
    # CES section
    st.markdown("---")
    ces=load_art(rd,"CES_State.json")
    if ces:
        st.markdown(MC([(ces.get("chapters_count","—"),"Chapitres"),(f'{ces.get("saturated",0)}/{ces.get("chapters_count","?")}',
            "Saturés (100%)","#66bb6a"),(ces.get("qc_count","—"),"QC Générées"),(ces.get("qi_count","—"),"Qi Extraites"),
            (f'{ces.get("coverage_avg",0):.0f}%',"Couverture moy.","#66bb6a" if ces.get("coverage_avg",0)>=95 else "#ffa726")]),unsafe_allow_html=True)
        st.markdown(MC([(ces.get("pdf_count","—"),"PDFs Collectés"),(ces.get("pairs_count","—"),"Paires Sujet/Corrigé"),
            (f'{ces.get("gates_pass",0)}/{ces.get("gates_total","?")}',f"Gates PASS"),
            (ces.get("quarantine_count",0),"Items Quarantaine","#ef5350" if ces.get("quarantine_count",0)>0 else "#66bb6a")]),unsafe_allow_html=True)
        # Saturation per subject
        qc_idx=load_art(rd,"QC_Index.json") or {};sat=load_art(rd,"SaturationReport.json") or {}
        if qc_idx:
            sel_lv=st.selectbox("Filtrer par niveau",["Tous"]+[l["level_name_local"] for l in edu["levels"]],key="d_lv")
            sfs=edu["subjects"]
            if sel_lv!="Tous":
                lc=next((l["level_code"] for l in edu["levels"] if l["level_name_local"]==sel_lv),None)
                sfs=[s for s in sfs if s["level_code"]==lc] if lc else sfs
            for su in sfs:
                ln=next((l["level_name_local"] for l in edu["levels"] if l["level_code"]==su["level_code"]),su["level_code"])
                has_data=any(qc_idx.get(f"{su['subject_id']}_CH{str(ch['chapter_number']).zfill(2)}") for ch in su["chapters"])
                if not has_data:continue
                st.markdown(f"**Couverture QC par Chapitre — {ln} {su['subject_name_local']}**")
                for ch in su["chapters"]:
                    ck=f"{su['subject_id']}_CH{str(ch['chapter_number']).zfill(2)}"
                    qd=qc_idx.get(ck,{});sd=sat.get(ck,{})
                    cv=qd.get("coverage",0);ss=sd.get("status","⏳")
                    sb='<span class="gp" style="font-size:.68rem">SATURÉ</span>' if ss=="SATURATED" else '<span class="gu" style="font-size:.68rem">EN COURS</span>'
                    st.markdown(f'<div style="display:flex;align-items:center;gap:.5rem;padding:.15rem 0;font-size:.76rem">'
                        f'<span style="min-width:180px;color:var(--t2)">CH{str(ch["chapter_number"]).zfill(2)} {ch["chapter_name"][:40]}</span>'
                        f'<div style="flex:1">{CB(cv)}</div>'
                        f'<span style="min-width:35px;text-align:right;color:{"#66bb6a" if cv>=95 else "#ffa726" if cv>=75 else "#ef5350"};font-weight:700">{cv}%</span>'
                        f'{sb}</div>',unsafe_allow_html=True)
    else:
        st.info("⏳ **CES HARVEST non lancé** — Les KPIs apparaîtront après E1 COLLECT.")

def pg_niveaux(cap,rd):
    edu=cap["B_EDUCATION_SYSTEM"];ex=cap["E_EXAMS_CONCOURS"]["exams_and_contests"]
    st.markdown(H("🏫 Niveaux & Classes",f'{cap["A_METADATA"]["total_classes"]} classes réparties sur {len(edu["cycles"])} cycles académiques'),unsafe_allow_html=True)
    for c in edu["cycles"]:
        lv=LF(edu,c["cycle_id"])
        st.markdown(f'<div class="cyc {CC(c["cycle_id"])}">{c["cycle_name_local"]} — {len(lv)} classes</div>',unsafe_allow_html=True)
        for l in lv:
            su=SF(edu,l["level_code"]);ch=sum(s["chapter_count"] for s in su);e=EF(ex,l["level_code"])
            eb=f'<span class="b-ex">🎯 {e["exam"]["exam_name"]}</span>' if e and e.get("exam") else ""
            ct=" ".join(f'<span class="b-co">🏆#{c["rank"]} {c["name"]}</span>' for c in(e.get("contests_top")or[])[:7]) if e else ""
            st.markdown(f'<div class="cc"><b style="color:#e3f2fd;font-size:.88rem">{l["level_name_local"]}</b>'
                f' <span style="color:#546e7a;font-size:.68rem">({l["level_code"]})</span>'
                f'<div style="color:#546e7a;font-size:.66rem">{l.get("voie","")}</div>'
                f'<div style="color:var(--bl);font-size:.72rem">📘 {len(su)} matières · 📖 {ch} chapitres</div>'
                f'{eb}{f"<div style=margin-top:.1rem>{ct}</div>" if ct else ""}</div>',unsafe_allow_html=True)

def pg_chapitres(cap,rd):
    edu=cap["B_EDUCATION_SYSTEM"];tot=sum(s["chapter_count"] for s in edu["subjects"])
    st.markdown(H("📖 Chapitres Explorer",f'{tot} chapitres sourcés — cliquez pour déplier'),unsafe_allow_html=True)
    q=st.text_input("🔍 Rechercher un chapitre...",key="ch_q")
    for su in edu["subjects"]:
        ln=next((l["level_name_local"] for l in edu["levels"] if l["level_code"]==su["level_code"]),su["level_code"])
        chs=[ch for ch in su["chapters"] if not q or q.lower() in ch["chapter_name"].lower() or q.lower() in su["subject_name_local"].lower() or q.lower() in ln.lower()]
        if not chs:continue
        hdr_txt=f"{ln} — {su['subject_name_local']} ({su['chapter_count']} chap.)"
        with st.expander(f"**{hdr_txt}** {SB(su.get('source_type',''))}"):
            st.markdown(f'<div style="color:#546e7a;font-size:.66rem;margin-bottom:.3rem">{su.get("source_ref","")}</div>',unsafe_allow_html=True)
            for ch in chs:
                st.markdown(f'<span style="color:var(--a);font-weight:700;font-family:JetBrains Mono;font-size:.78rem">'
                    f'{str(ch["chapter_number"]).zfill(2)}</span>&nbsp;&nbsp;'
                    f'<span style="font-size:.78rem">{ch["chapter_name"]}</span>',unsafe_allow_html=True)

def pg_examens(cap,rd):
    ex=cap["E_EXAMS_CONCOURS"]["exams_and_contests"];edu=cap["B_EDUCATION_SYSTEM"]
    st.markdown(H("🎓 Examens & Concours",f'{len(ex)} entrées — chaque classe a son examen + concours associés'),unsafe_allow_html=True)
    for c in edu["cycles"]:
        st.markdown(f'<div class="cyc {CC(c["cycle_id"])}">{c["cycle_name_local"]}</div>',unsafe_allow_html=True)
        for l in LF(edu,c["cycle_id"]):
            e=EF(ex,l["level_code"])
            if not e:continue
            eb=f'<span class="b-ex">🎯 {e["exam"]["exam_name"]}</span>' if e.get("exam") else ""
            ct=" ".join(f'<span class="b-co">🏆#{c["rank"]} {c["name"]}</span>' for c in(e.get("contests_top")or[])) if e else ""
            st.markdown(f'<div class="cc"><b style="color:#e3f2fd;font-size:.85rem">{l["level_name_local"]}</b>'
                f' <span style="color:#546e7a;font-size:.66rem">({l["level_code"]})</span>'
                f'<div style="margin-top:.1rem">{eb}</div>{f"<div style=margin-top:.1rem>{ct}</div>" if ct else ""}</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 5. PAGES — VALIDATION CEO
# ═══════════════════════════════════════════════════════════════════
def pg_ch_status(cap,rd):
    edu=cap["B_EDUCATION_SYSTEM"];qc_idx=load_art(rd,"QC_Index.json") or {};sat=load_art(rd,"SaturationReport.json") or {}
    st.markdown(H("📊 Chapitres Status","Couverture QC · Saturation · STOP_RULE"),unsafe_allow_html=True)
    if not qc_idx:
        st.info("⏳ Aucune donnée CES — En attente du pipeline.");return
    # Filters
    cn={c["cycle_id"]:c["cycle_name_local"] for c in edu["cycles"]}
    c1,c2=st.columns(2)
    with c1:sc=st.selectbox("Cycle",["Tous"]+list(cn.values()),key="cs_c")
    scid=next((k for k,v in cn.items() if v==sc),None)
    lf=edu["levels"] if not scid else [l for l in edu["levels"] if l["cycle_id"]==scid]
    ln={l["level_code"]:l["level_name_local"] for l in lf}
    with c2:sl=st.selectbox("Niveau",["Tous"]+list(ln.values()),key="cs_l")
    slc=next((k for k,v in ln.items() if v==sl),None)
    sf=edu["subjects"]
    if slc:sf=[s for s in sf if s["level_code"]==slc]
    elif scid:sf=[s for s in sf if s["level_code"] in [l["level_code"] for l in lf]]
    # Table header
    st.markdown('<div class="tbl-row tbl-h"><span style="min-width:55px">ID</span><span style="flex:2">CHAPITRE</span>'
        '<span style="min-width:70px;text-align:center">QC</span><span style="min-width:60px;text-align:center">QI</span>'
        '<span style="flex:1.5">COUVERTURE</span><span style="min-width:45px;text-align:right">%</span>'
        '<span style="min-width:75px;text-align:center">STATUT</span></div>',unsafe_allow_html=True)
    for su in sf:
        for ch in su["chapters"]:
            ck=f"{su['subject_id']}_CH{str(ch['chapter_number']).zfill(2)}"
            qd=qc_idx.get(ck,{});sd=sat.get(ck,{})
            qc=qd.get("qc_count",0);qi=qd.get("qi_count",0);cv=qd.get("coverage",0)
            tgt=qd.get("target",15);ss=sd.get("status","⏳")
            qc_col="#66bb6a" if qc>=tgt else "#ffa726"
            cv_col="#66bb6a" if cv>=95 else("#ffa726" if cv>=75 else "#ef5350")
            sb='<span style="background:#1b5e20;color:#a5d6a7;padding:.05rem .35rem;border-radius:3px;font-size:.6rem;font-weight:700">SATURÉ</span>' if ss=="SATURATED" else '<span style="background:#e65100;color:#ffcc80;padding:.05rem .35rem;border-radius:3px;font-size:.6rem;font-weight:700">EN COURS</span>'
            st.markdown(f'<div class="tbl-row">'
                f'<span style="min-width:55px;color:var(--a);font-weight:700;font-family:JetBrains Mono;font-size:.74rem">CH{str(ch["chapter_number"]).zfill(2)}</span>'
                f'<span style="flex:2;font-size:.76rem">{ch["chapter_name"][:45]}</span>'
                f'<span style="min-width:70px;text-align:center"><span style="background:#1a237e;color:{qc_col};padding:.05rem .3rem;border-radius:3px;font-size:.68rem;font-weight:600">{qc}/{tgt} QC</span></span>'
                f'<span style="min-width:60px;text-align:center;font-size:.74rem;color:var(--t2)">{qi} Qi</span>'
                f'<span style="flex:1.5">{CB(cv)}</span>'
                f'<span style="min-width:45px;text-align:right;color:{cv_col};font-weight:700;font-size:.76rem">{cv}%</span>'
                f'<span style="min-width:75px;text-align:center">{sb}</span></div>',unsafe_allow_html=True)

def pg_qc_explorer(cap,rd):
    edu=cap["B_EDUCATION_SYSTEM"];qc_idx=load_art(rd,"QC_Index.json")
    st.markdown(H("🔬 QC Explorer","QC · FRT · ARI · TRIGGERS — Données opaques (IP SMAXIA)"),unsafe_allow_html=True)
    if not qc_idx:
        st.info("⏳ Aucune QC — En attente CES HARVEST.")
        st.markdown('<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin:.4rem 0">'
            '<span style="background:#1565c0;color:#fff;padding:.1rem .4rem;border-radius:3px;font-size:.62rem">QC = Question Clé (parent)</span>'
            '<span style="background:#2e7d32;color:#fff;padding:.1rem .4rem;border-radius:3px;font-size:.62rem">FRT = Fiche Réponse Type</span>'
            '<span style="background:#e65100;color:#fff;padding:.1rem .4rem;border-radius:3px;font-size:.62rem">ARI = Algorithme Résolution Invariant</span>'
            '<span style="background:#b71c1c;color:#fff;padding:.1rem .4rem;border-radius:3px;font-size:.62rem">TRIGGER = Mots-clés de mapping Qi→QC</span>'
            '</div>',unsafe_allow_html=True)
        return
    # Chapter selector
    all_chs=list(set(ck for ck in qc_idx.keys()))
    sel_ch=st.selectbox("Chapitre",sorted(all_chs),key="qce_ch")
    if not sel_ch:return
    qd=qc_idx[sel_ch];qc_ids=qd.get("qc_ids",[])
    cv=qd.get("coverage",0);ss=qd.get("status","⏳")
    st.markdown(MC([(len(qc_ids),"QC"),(qd.get("qi_count",0),"Qi rattachées"),
        (f"{cv}%","Couverture"),(ss,"Statut")]),unsafe_allow_html=True)
    st.markdown('<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin:.3rem 0">'
        '<span style="background:#1565c0;color:#fff;padding:.1rem .4rem;border-radius:3px;font-size:.62rem">QC = Question Clé (parent)</span>'
        '<span style="background:#2e7d32;color:#fff;padding:.1rem .4rem;border-radius:3px;font-size:.62rem">FRT = Fiche Réponse Type</span>'
        '<span style="background:#e65100;color:#fff;padding:.1rem .4rem;border-radius:3px;font-size:.62rem">ARI = Algorithme Résolution Invariant</span>'
        '<span style="background:#b71c1c;color:#fff;padding:.1rem .4rem;border-radius:3px;font-size:.62rem">TRIGGER = Mots-clés de mapping Qi→QC</span>'
        '</div>',unsafe_allow_html=True)
    q=st.text_input("Rechercher une QC...",key="qce_q")
    # Display QC cards
    for qid in qc_ids:
        det=load_json(os.path.join(rd,"QC_Details",f"{qid}.json")) if rd and os.path.isdir(os.path.join(rd,"QC_Details")) else None
        if q and det and q.lower() not in json.dumps(det).lower():continue
        if det:
            qi_n=len(det.get("qi_children",[]));qi_cv=det.get("coverage",100)
            st.markdown(f'<div class="qc-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<div><span style="color:#ef5350;font-weight:700;font-family:JetBrains Mono;font-size:.76rem">{qid}</span>'
                f'&nbsp;&nbsp;<b style="font-size:.82rem">{det.get("title","")}</b></div>'
                f'<div><span style="background:#1b5e20;color:#a5d6a7;padding:.05rem .35rem;border-radius:3px;font-size:.62rem;font-weight:700">{qi_n} Qi</span>'
                f'&nbsp;<span style="background:#0d47a1;color:#90caf9;padding:.05rem .35rem;border-radius:3px;font-size:.62rem;font-weight:700">COV {qi_cv}%</span></div></div>'
                f'<div class="qc-frt"><span style="color:#42a5f5;font-weight:700;font-size:.72rem">FRT</span> — {det.get("frt",{}).get("text","N/A")}</div>'
                f'<div class="qc-ari"><span style="color:#66bb6a;font-weight:700;font-size:.72rem">ARI</span> — {det.get("ari",{}).get("text","N/A")}</div>'
                f'<div class="qc-trg"><span style="color:#ef5350;font-weight:700;font-size:.72rem">TRIGGERS</span> — '
                f'{"".join(f"<span class=trig-chip>{t}</span> " for t in det.get("triggers",[])[:6])}</div></div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="qc-card"><span style="color:#ef5350;font-weight:700;font-family:JetBrains Mono;font-size:.76rem">{qid}</span>'
                f'&nbsp;&nbsp;<span style="color:var(--t2);font-size:.74rem">⏳ Détails non disponibles (QC_Details/{qid}.json)</span></div>',unsafe_allow_html=True)

def pg_mapping(cap,rd):
    st.markdown(H("🗺️ Mapping Qi → QC","Couverture par chapitre"),unsafe_allow_html=True)
    cov=load_art(rd,"CoverageReport.json")
    if not cov:st.info("⏳ En attente CES HARVEST.");return
    for ck,d in sorted(cov.items()):
        cv=d.get("coverage",0);orph=d.get("orphans",0)
        st.markdown(f'<div class="tbl-row"><code style="color:var(--a);min-width:180px;font-size:.72rem">{ck}</code>'
            f'<div style="flex:1">{CB(cv)}</div>'
            f'<span style="min-width:40px;text-align:right;font-size:.74rem">{cv}%</span>'
            f'<span style="min-width:60px;text-align:right;color:{"#ef5350" if orph>0 else "#66bb6a"};font-size:.74rem">{orph} orph.</span></div>',unsafe_allow_html=True)

def pg_test_sujet(cap,rd):
    st.markdown(H("🧪 Test Sujet","Vérifiez la couverture QC sur n'importe quel sujet (Interro, DST, Bac, Concours)"),unsafe_allow_html=True)
    # Load test results
    tests=[]
    if rd and os.path.isdir(rd):
        for f in glob.glob(os.path.join(rd,"MappingReport_*.json")):
            d=load_json(f)
            if d:d["_file"]=os.path.basename(f);tests.append(d)
    if tests:
        ok=sum(1 for t in tests if t.get("coverage",0)>=95)
        ko=sum(1 for t in tests if t.get("coverage",0)<95)
        st.markdown(MC([(len(tests),"Sujets testés"),(ok,"Taux ≥ 95%","#66bb6a"),(ko,"Taux < 95%","#ef5350")]),unsafe_allow_html=True)
        q=st.text_input("Rechercher un sujet...",key="ts_q")
        for t in tests:
            if q and q.lower() not in json.dumps(t).lower():continue
            cv=t.get("coverage",0);orph=t.get("orphans",[]);qi_t=t.get("qi_total",0);qi_m=t.get("qi_mapped",0)
            nm=t.get("title",t["_file"]);dt=t.get("date","");tp=t.get("type","")
            cv_col="#66bb6a" if cv>=95 else("#ffa726" if cv>=75 else "#ef5350")
            tp_b={"BAC":"#1565c0","DST":"#6a1b9a","INTERRO":"#2e7d32","CONCOURS":"#b71c1c"}.get(tp.upper(),"#37474f")
            chs=" ".join(f'<span style="background:#1a237e;color:#90caf9;padding:.04rem .3rem;border-radius:3px;font-size:.58rem;font-weight:600">{c}</span>' for c in t.get("chapters_covered",[]))
            st.markdown(f'<div class="cc" style="padding:.6rem .85rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                f'<div><b style="font-size:.85rem">{nm}</b><br>'
                f'<span style="color:var(--t2);font-size:.66rem">{dt}</span>'
                f'&nbsp;<span style="background:{tp_b};color:#fff;padding:.04rem .3rem;border-radius:3px;font-size:.58rem;font-weight:700">{tp}</span>'
                f'<div style="font-size:.72rem;margin-top:.15rem">{qi_t} Qi total&nbsp;&nbsp;{qi_m} mappées'
                f'&nbsp;&nbsp;<span style="color:{"#ef5350" if len(orph)>0 else "#66bb6a"}">{len(orph)} non mappées</span></div>'
                f'<div style="margin-top:.1rem">Chapitres : {chs}</div></div>'
                f'<div style="text-align:right"><span style="font-size:1.6rem;font-weight:800;color:{cv_col}">{cv}%</span>'
                f'<div style="font-size:.56rem;color:var(--t2);font-weight:700;letter-spacing:1px">TAUX QC</div></div></div>'
                f'{CB(cv)}</div>',unsafe_allow_html=True)
    else:
        st.info("⏳ Aucun sujet testé. Déposez MappingReport_*.json dans le run dir.")
    # Upload test
    st.markdown("---")
    st.markdown("**📤 Tester un nouveau sujet**")
    up=st.file_uploader("Upload PDF",type=["pdf"],key="ts_up")
    if up and st.button("▶ Créer TestRequest",key="ts_btn"):
        if rd:
            b=up.read();sha=hashlib.sha256(b).hexdigest()
            req={"type":"TEST_B","filename":up.name,"pdf_sha256":sha,"size":len(b),
                "at":datetime.datetime.utcnow().isoformat()+"Z","status":"PENDING"}
            os.makedirs(rd,exist_ok=True)
            with open(os.path.join(rd,f"TestRequest_{sha[:12]}.json"),"w") as f:json.dump(req,f,indent=2)
            os.makedirs(os.path.join(rd,"test_pdfs"),exist_ok=True)
            with open(os.path.join(rd,"test_pdfs",up.name),"wb") as f:f.write(b)
            st.success("✅ TestRequest créé");st.json(req)

def pg_predictions(cap,rd):
    st.markdown(H("🎯 Prédiction Chapitres Tombables","PREDICTION_PACK_REF opaque — IP SMAXIA"),unsafe_allow_html=True)
    pred=load_art(rd,"PredictionReport.json")
    if not pred:
        st.info("⏳ Module de prédiction non activé. Déposez PredictionReport.json dans le run dir.")
        st.markdown('<div class="wb">Ce module est opaque : il référence un PREDICTION_PACK_REF + SHA256. '
            'Aucun détail mécanique exposé (D6 — IP SMAXIA).</div>',unsafe_allow_html=True)
        return
    topk=pred.get("top_k_chapters",[]);recall=pred.get("recall_at_k",0);years=pred.get("backtest_years",0)
    status="VALIDÉ" if recall>=95 else "EN COURS"
    st.markdown(MC([(len(topk),"Chapitres TOP-K"),(f"{recall}%","Recall@K actuel","#66bb6a" if recall>=95 else "#ffa726"),
        (years,"Années backtest"),(status,"Statut","#66bb6a" if status=="VALIDÉ" else "#ffa726")]),unsafe_allow_html=True)
    if recall<95:
        st.markdown(f'<div class="wb">recall@K = {recall}% < 95% — UI affichera « EN COURS DE PREUVE » (honnêteté intégrée)</div>',unsafe_allow_html=True)
    # Table
    st.markdown('<div class="tbl-row tbl-h"><span style="min-width:50px">RANG</span><span style="min-width:55px">ID</span>'
        '<span style="flex:2">CHAPITRE</span><span style="flex:1.5">SCORE TOMBABLE</span><span style="min-width:45px;text-align:right">%</span>'
        '<span style="min-width:70px;text-align:center">SET</span></div>',unsafe_allow_html=True)
    for i,ch in enumerate(topk):
        sc=ch.get("score",0);is_topk=ch.get("in_top_k",True)
        sc_col="#66bb6a" if sc>=85 else("#42a5f5" if sc>=70 else("#ffa726" if sc>=50 else "#ef5350"))
        sb='<span style="background:#1a237e;color:#90caf9;padding:.05rem .35rem;border-radius:3px;font-size:.6rem;font-weight:700">TOP-K</span>' if is_topk else '<span style="background:#b71c1c;color:#ffcdd2;padding:.05rem .35rem;border-radius:3px;font-size:.6rem;font-weight:700">HORS SET</span>'
        rk=f'<span style="color:{"#ffa726" if i<3 else "var(--t2)"};font-weight:{"800" if i<3 else "400"};font-size:.82rem">#{i+1}</span>'
        st.markdown(f'<div class="tbl-row">'
            f'<span style="min-width:50px">{rk}</span>'
            f'<span style="min-width:55px;color:var(--a);font-family:JetBrains Mono;font-size:.72rem">{ch.get("id","")}</span>'
            f'<span style="flex:2;font-size:.76rem">{ch.get("name","")}</span>'
            f'<span style="flex:1.5">{CB(sc)}</span>'
            f'<span style="min-width:45px;text-align:right;color:{sc_col};font-weight:700;font-size:.76rem">{sc}%</span>'
            f'<span style="min-width:70px;text-align:center">{sb}</span></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="cc" style="margin-top:.5rem;border-left:3px solid #42a5f5"><b style="color:#42a5f5;font-size:.8rem">Méthode</b>'
        f'<div style="font-size:.72rem;color:var(--t2);margin-top:.15rem">'
        f'Module opaque référencé par PREDICTION_PACK_REF + SHA256.<br>'
        f'Validation : backtest rolling leave-one-year-out sur {years} années.<br>'
        f'KPI cible : recall@K ≥ 0.95. Si non atteint, l\'UI l\'indique explicitement.<br>'
        f'Aucun détail mécanique exposé (D6 — IP SMAXIA).</div></div>',unsafe_allow_html=True)

def pg_best_ideas():
    st.markdown(H("💡 BEST IDEAS — Avantages Irrattrapables","10 idées game-changer pour SMAXIA n°1 mondial"),unsafe_allow_html=True)
    ideas=[
        ("🎯","Proof of Kill Rate™","Afficher pour chaque chapitre un indicateur public : 'X% des questions de ton prochain examen sont couvertes par ces QC.' Aucun concurrent ne peut faire ça sans CAS 1 ONLY + gates déterministes.",
         "💎 AVANTAGE IRRATTRAPABLE — Seul SMAXIA peut prouver mathématiquement sa couverture","Produit","#42a5f5"),
        ("🌍","170 Pays = 170 Monopoles Locaux","Chaque CAP scellé est un monopole de fait. Le premier pays activé verrouille le marché. Notre pipeline parallèle permet d'activer un pays en jours, pas en mois.",
         "💎 WINNER TAKES ALL — Premier CAP scellé = barrière permanente","Stratégie","#66bb6a"),
        ("📊","PrediNote : Prédire la note","ARI + couverture QC + taux de réussite = prédiction ±1 point. 'Tu auras entre 14 et 16 au Bac Maths.' Le Saint-Graal de l'EdTech, mathématiquement fondé.",
         "💎 VIRAL — Chaque élève partage sa PrediNote = acquisition gratuite","Produit","#42a5f5"),
        ("🏆","Classement National SMAX Score","Position anonyme par matière/chapitre. 'Tu es dans le top 12% en Intégration.' Compétition saine + motivation + rétention. Seul SMAXIA a les données par chapitre.",
         "💎 RÉTENTION x3 — Gamification par données réelles > badges artificiels","Engagement","#ffa726"),
        ("🔄","Boucle Virale 'Défi Chapitre'","Défier un ami sur un chapitre. L'ami DOIT télécharger pour répondre. K-factor > 1 = croissance exponentielle.",
         "💎 ACQUISITION GRATUITE — K-factor > 1","Growth","#66bb6a"),
        ("🎖️","Certification SMAXIA (Badge LinkedIn)","100% couverture + ≥90% réussite = certificat vérifiable QR + SHA256. Intégrable LinkedIn/Parcoursup.",
         "💎 EFFET DE RÉSEAU — Plus de certifiés = plus de crédibilité","Monétisation","#b39ddb"),
        ("👨‍🏫","Dashboard Prof B2B","Vue agrégée classe/établissement. Le prof voit les chapitres faibles, adapte son cours. Aucun concurrent n'a cette granularité chapitre-par-chapitre.",
         "💎 B2B PREMIUM — Licence établissement = revenus récurrents","B2B","#ef5350"),
        ("⚡","Mode Urgence J-7","7 jours avant l'examen : SMAXIA concentre l'élève sur les chapitres TOP-K prédits + les QC non maîtrisées. Impact maximal, effort minimal.",
         "💎 CONVERSION — Urgence = achat impulsif","Produit","#42a5f5"),
        ("🧬","Génome Examen","Chaque examen a un 'ADN' : distribution des QC sur les chapitres. SMAXIA le cartographie année après année, révélant les patterns cachés.",
         "💎 DATA MOAT — Plus d'années = plus de patterns = meilleure prédiction","Data","#ffa726"),
        ("🔌","API Éditeurs","Ouvrir SMAXIA aux éditeurs scolaires : ils intègrent nos QC/FRT dans leurs manuels. Revenus de licence + distribution massive.",
         "💎 PLATEFORME — De produit à infrastructure éducative","Plateforme","#b39ddb"),
    ]
    for icon,title,desc,adv,cat,col in ideas:
        st.markdown(f'<div class="idea-card" style="border-left:4px solid {col}">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
            f'<div><span style="font-size:.9rem">{icon}</span> <b style="color:{col};font-size:.85rem">{title}</b>'
            f'<div style="font-size:.74rem;color:var(--t);margin-top:.15rem">{desc}</div>'
            f'<div style="color:{col};font-size:.7rem;font-weight:600;margin-top:.15rem">{adv}</div></div>'
            f'<span style="background:rgba(255,255,255,.06);color:var(--t2);padding:.1rem .4rem;border-radius:3px;font-size:.6rem;font-weight:700;white-space:nowrap">{cat}</span></div></div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="SMAXIA Command Center",page_icon="🔒",layout="wide",initial_sidebar_state="expanded")
    css()
    PAGES_PIPE=["📊 Dashboard","🏫 Niveaux & Classes","📖 Chapitres Explorer","🎓 Examens & Concours"]
    PAGES_CEO=["📊 Chapitres Status","🔬 QC Explorer","🗺️ Mapping Qi→QC","🧪 Test Sujet","🎯 Prédictions","💡 Best Ideas"]
    with st.sidebar:
        st.markdown("### 🔒 SMAXIA")
        # Run
        runs_base="./runs";av=scan_runs(runs_base)
        if av:
            sel=st.selectbox("📂 Run",av,key="sr")
            rd=os.path.join(runs_base,sel)
        else:rd=os.path.join(runs_base,"latest")
        with st.expander("⚙️ Avancé"):
            cp=st.text_input("Chemin custom","",key="cp")
            if cp:rd=cp
        # CAP
        cap=st.session_state.get("cap_data")
        if not cap:
            for p in scan_caps(rd):
                cap=load_json(p)
                if cap:st.session_state["cap_data"]=cap;break
        cap_up=st.file_uploader("📤 CAP JSON",type=["json"],key="cup")
        if cap_up:
            try:
                b=cap_up.read();cap=json.loads(b.decode("utf-8"));st.session_state["cap_data"]=cap
                try:os.makedirs(rd,exist_ok=True);open(os.path.join(rd,cap_up.name),"wb").write(b)
                except:pass
                st.success("✅ CAP chargé")
            except Exception as e:st.error(f"❌ {e}")
        if cap:
            m=cap.get("A_METADATA",{})
            fl={"FR":"🇫🇷","MA":"🇲🇦","SN":"🇸🇳","CI":"🇨🇮","TN":"🇹🇳"}.get(m.get("country_code",""),"🌍")
            st.markdown(f"**{fl} {m.get('country_name_local','')}** · K{m.get('kernel_version','')}")
            if m.get("status")=="SEALED":st.markdown('<span class="b-se">✓ SEALED</span>',unsafe_allow_html=True)
        st.markdown("---")
        # Navigation
        st.markdown('<div class="ns">SMAXIA PIPELINE</div>',unsafe_allow_html=True)
        p1=st.radio("p1",PAGES_PIPE,key="n1",label_visibility="collapsed")
        st.markdown('<div class="ns">VALIDATION CEO</div>',unsafe_allow_html=True)
        p2=st.radio("p2",PAGES_CEO,key="n2",label_visibility="collapsed")
    # Track active
    for k in["n1","n2"]:
        pk=f"_p_{k}"
        cur=st.session_state.get(k)
        if pk not in st.session_state:st.session_state[pk]=cur
        if st.session_state[pk]!=cur:st.session_state["_active"]=cur;st.session_state[pk]=cur
    active=st.session_state.get("_active",PAGES_PIPE[0])
    if not cap:
        st.markdown(H("⚠️ Aucun CAP chargé","Uploadez CAP_*.json via la sidebar"),unsafe_allow_html=True)
        st.markdown("Uploadez votre CAP scellé via le bouton dans la sidebar pour commencer.")
        return
    # Route
    R={
        PAGES_PIPE[0]:pg_dashboard,PAGES_PIPE[1]:pg_niveaux,PAGES_PIPE[2]:pg_chapitres,PAGES_PIPE[3]:pg_examens,
        PAGES_CEO[0]:pg_ch_status,PAGES_CEO[1]:pg_qc_explorer,PAGES_CEO[2]:pg_mapping,PAGES_CEO[3]:pg_test_sujet,
        PAGES_CEO[4]:pg_predictions,
    }
    if active=="💡 Best Ideas":pg_best_ideas()
    elif active in R:R[active](cap,rd)

if __name__=="__main__":main()
