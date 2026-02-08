#!/usr/bin/env python3
# smaxia_gte_v14_3_3.py
"""SMAXIA GTE Kernel v14.3.3 — ISO-PROD Single-File Compact"""

import streamlit as st
import json, hashlib, os, re, inspect, random, math
from datetime import datetime, timezone
from pathlib import Path

# ─── CONFIG ───
MAX_PDFS = 20
BFS_DEPTH = 2
SEED_DOMAINS_MAX = 8
RUN_DIR = Path("run")
VOLATILE_KEYS = {"timestamp", "ts", "run_id", "elapsed_ms", "wall_clock"}

# ─── DETERMINISM UTILS ───
def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def sha256(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in sorted(obj.items()) if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(i) for i in obj]
    return obj

def _rng_for_country(iso2):
    seed = int(sha256(("SMAXIA:" + (iso2 or "").upper()))[:8], 16)
    return random.Random(seed)

def determinism_check(pipeline_func, iso2, n=3):
    hashes = []
    for _ in range(n):
        result = pipeline_func(iso2, _determinism_run=True)
        snap = strip_volatile(result["snapshot"])
        c = canonical_json(snap)
        hashes.append(sha256(c))
    return {"pass": len(set(hashes)) == 1, "hashes": hashes}

# ─── ARTIFACT WRITER ───
def ensure_run_dir(run_id):
    p = RUN_DIR / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_artifact(run_id, name, data):
    p = ensure_run_dir(run_id)
    c = canonical_json(data)
    h = sha256(c)
    fp = p / f"{name}.json"
    fp.write_text(c, encoding="utf-8")
    return {"artifact": name, "sha256": h, "path": str(fp)}

# ─── UI EVENT LOG ───
def log_event(evt_type, detail=""):
    if "ui_events" not in st.session_state:
        st.session_state.ui_events = []
    st.session_state.ui_events.append({
        "type": evt_type, "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

# ─── COUNTRY INDEX + TYPEAHEAD ───
@st.cache_data
def _build_country_index():
    try:
        import pycountry
        db = {c.alpha_2: c.name for c in pycountry.countries}
    except Exception:
        db = {}
        try:
            import urllib.request, json as _json
            url = "https://restcountries.com/v3.1/all?fields=cca2,name"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = _json.loads(resp.read())
            db = {c["cca2"]: c["name"]["common"] for c in data if "cca2" in c and "name" in c}
        except Exception:
            pass
    idx = [{"code": k, "name": v, "nl": v.lower(), "cl": k.lower()} for k, v in db.items()]
    idx.sort(key=lambda e: e["name"])
    return db, idx

def typeahead(q, limit=20):
    _, idx = _build_country_index()
    ql = (q or "").strip().lower()
    if not ql:
        return [], []
    is_iso_like = len(ql) == 2 and ql.isalpha()
    prefix, fallback, seen = [], [], set()
    if is_iso_like:
        for e in idx:
            if e["cl"] == ql:
                prefix.append(e)
                seen.add(e["code"])
                break
    if not prefix:
        for e in idx:
            if e["nl"].startswith(ql):
                prefix.append(e)
                seen.add(e["code"])
                if len(prefix) >= limit:
                    break
    if len(prefix) < limit:
        for e in idx:
            if e["code"] in seen:
                continue
            if e["cl"].startswith(ql) or ql in e["nl"] or ql in e["cl"]:
                fallback.append(e)
                seen.add(e["code"])
                if len(fallback) >= limit:
                    break
    return prefix, fallback

# ─── FORMULA_PACK (OPAQUE NAMESPACE) ───
FORMULA_PACK = {
    "version": "FP-3.1-SEALED",
    "mode": "SIMULATION",
    "sha256_manifest": None,
    "executors": {}
}

def _fp_compute_f1(atoms, frt, qc, pack_cfg):
    """F1 opaque digest executor — SIMULATION mode. Returns digest, not real A2 score."""
    n = len(atoms)
    if n == 0:
        return {"digest": sha256("empty"), "status": "NO_ATOMS", "mode": "SIMULATION"}
    qc_hashes = canonical_json([q.get("qi_hash", "") for q in sorted(qc, key=lambda x: x.get("atom_id", ""))])
    atom_hashes = canonical_json([a["Qi"]["source_hash"] for a in atoms])
    combined = sha256(qc_hashes + atom_hashes + canonical_json(frt))
    return {"digest": combined, "status": "OK", "mode": "SIMULATION", "inputs_n": n}

def _fp_compute_f2(f1_digest, ari_profiles, triggers, pack_cfg):
    """F2 opaque digest executor — SIMULATION mode. Returns digest, not real A2 score."""
    base_digest = f1_digest.get("digest", "")
    ari_canon = canonical_json(sorted([p.get("atom_id", "") for p in ari_profiles]))
    trig_canon = canonical_json(sorted([t.get("trigger", "") for t in triggers]))
    combined = sha256(base_digest + ari_canon + trig_canon)
    return {"digest": combined, "status": "OK", "mode": "SIMULATION", "inputs_ari": len(ari_profiles), "inputs_trig": len(triggers)}

FORMULA_PACK["executors"]["compute_f1"] = _fp_compute_f1
FORMULA_PACK["executors"]["compute_f2"] = _fp_compute_f2
FORMULA_PACK["sha256_manifest"] = sha256(canonical_json({
    "version": FORMULA_PACK["version"],
    "mode": FORMULA_PACK["mode"],
    "f1_src_hash": sha256(inspect.getsource(_fp_compute_f1)),
    "f2_src_hash": sha256(inspect.getsource(_fp_compute_f2)),
}))

# ─── FORMULA ENGINE (DELEGATION ONLY) ───
class FormulaEngine:
    BANNED_KEYWORDS = ["delta_c", "epsilon", "psi_raw", "one_m", "alpha", "anti_red", "freq", "cosine"]

    @staticmethod
    def _audit_source(func):
        src = inspect.getsource(func)
        for kw in FormulaEngine.BANNED_KEYWORDS:
            if kw in src:
                return {"pass": False, "keyword": kw}
        return {"pass": True}

    @staticmethod
    def compute_f1(atoms, frt, qc, pack_cfg=None):
        return FORMULA_PACK["executors"]["compute_f1"](atoms, frt, qc, pack_cfg or {})

    @staticmethod
    def compute_f2(f1_digest, ari_profiles, triggers, pack_cfg=None):
        return FORMULA_PACK["executors"]["compute_f2"](f1_digest, ari_profiles, triggers, pack_cfg or {})

    @staticmethod
    def self_audit():
        a1 = FormulaEngine._audit_source(FORMULA_PACK["executors"]["compute_f1"])
        a2 = FormulaEngine._audit_source(FORMULA_PACK["executors"]["compute_f2"])
        violations = []
        if not a1["pass"]:
            violations.append(a1)
        if not a2["pass"]:
            violations.append(a2)
        return {
            "compute_f1_executor_clean": a1["pass"],
            "compute_f2_executor_clean": a2["pass"],
            "formula_pack_version": FORMULA_PACK["version"],
            "formula_pack_mode": FORMULA_PACK["mode"],
            "formula_pack_sha256": FORMULA_PACK["sha256_manifest"],
            "violations": violations,
        }

# ─── CHK_NO_COUNTRY_BRANCHING (REAL SOURCE SCAN) ───
def chk_no_country_branching():
    try:
        src = Path(__file__).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"pass": False, "reason": "cannot_read_source"}
    bad = []
    patterns = [
        r'==\s*["\']FR["\']', r'==\s*["\']SN["\']', r'==\s*["\']MA["\']',
        r'==\s*["\']US["\']', r'==\s*["\']GB["\']',
        r'if\s+.*country\s*==', r'switch.*country',
        r'\bcountry_map\s*\[', r'\bcountry_dict\s*\[',
    ]
    for pat in patterns:
        matches = re.findall(pat, src, re.IGNORECASE)
        if matches:
            bad.append({"pattern": pat, "count": len(matches)})
    return {"pass": len(bad) == 0, "bad_patterns": bad}

# ─── CAP DISCOVERY (INVARIANT, ZERO COUNTRY BRANCHING) ───
def build_cap_questions():
    return [
        "What is the structure of the national education system?",
        "What are the main examination bodies and their roles?",
        "What are the high-stakes national exams at each transition point?",
        "What subjects carry the highest coefficient weights?",
        "What are the top university programs by demand?",
        "What are the top 5-7 competitive exams per transition?",
        "What grading scale and passing thresholds are used?",
    ]

def discover_cap(iso2):
    log_event("CAP_DISCOVERY_START", iso2)
    cap = {
        "iso2": iso2,
        "questions": build_cap_questions(),
        "responses": {},
        "status": "DISCOVERED",
        "sha256": None,
    }
    for i, q in enumerate(cap["questions"]):
        cap["responses"][f"Q{i}"] = {"question": q, "answer": f"[PENDING_HARVEST_{iso2}]", "source": "awaiting_activation"}
    c = canonical_json(strip_volatile(cap))
    cap["sha256"] = sha256(c)
    log_event("CAP_DISCOVERY_END", iso2)
    return cap

# ─── VSP ───
def validate_vsp(cap):
    checks = [
        {"check": "iso2_present", "pass": bool(cap.get("iso2"))},
        {"check": "questions_count", "pass": len(cap.get("questions", [])) >= 5},
        {"check": "responses_exist", "pass": len(cap.get("responses", {})) > 0},
    ]
    all_pass = all(c["pass"] for c in checks)
    return {"status": "PASS" if all_pass else "FAIL", "checks": checks, "cap_sha256": cap.get("sha256")}

# ─── DA0 / SOURCE DISCOVERY (SIMULATION MODE — gates will FAIL) ───
def discover_sources(iso2, cap):
    log_event("DA0_START", iso2)
    rng = _rng_for_country(iso2)
    seeds = [f"https://education.{iso2.lower()}.example.org", f"https://exams.{iso2.lower()}.example.org"]
    seeds = seeds[:SEED_DOMAINS_MAX]
    strategy_log = {"iso2": iso2, "seeds": seeds, "bfs_depth": BFS_DEPTH, "max_pdfs": MAX_PDFS, "mode": "SIMULATION"}
    http_diag = []
    discovered = []
    for seed in seeds:
        n_pdfs = min(5, MAX_PDFS // len(seeds))
        entry = {"url": seed, "status": "SIMULATED_200", "pdfs_found": n_pdfs}
        http_diag.append(entry)
        for j in range(n_pdfs):
            discovered.append({
                "url": f"{seed}/exam_{j+1}.pdf",
                "type": rng.choice(["sujet", "corrige"]),
                "hash": sha256(f"{seed}/exam_{j+1}.pdf"),
                "size_kb": rng.randint(50, 500),
            })
    discovered = discovered[:MAX_PDFS]
    log_event("DA0_END", f"{len(discovered)} PDFs")
    return {"mode": "SIMULATION", "source_manifest": discovered, "strategy_log": strategy_log, "http_diag": http_diag}

# ─── CEP / PAIRING ───
def build_cep(sources):
    manifest = sources.get("source_manifest", [])
    sujets = [s for s in manifest if s["type"] == "sujet"]
    corriges = [s for s in manifest if s["type"] == "corrige"]
    pairs, unpaired = [], []
    for i, s in enumerate(sujets):
        if i < len(corriges):
            pairs.append({"sujet": s["url"], "corrige": corriges[i]["url"],
                          "sujet_hash": s["hash"], "corrige_hash": corriges[i]["hash"],
                          "pair_id": f"CEP_{i:04d}"})
        else:
            unpaired.append({"url": s["url"], "reason": "no_matching_corrige"})
    for c in corriges[len(sujets):]:
        unpaired.append({"url": c["url"], "reason": "no_matching_sujet"})
    return {"pairs": pairs, "unpaired": unpaired, "total_pairs": len(pairs)}

# ─── DA1 / DOWNLOAD + TEXT/OCR ───
def execute_da1(cep, iso2):
    log_event("DA1_START")
    rng = _rng_for_country(iso2)
    dl_log = []
    texts = {}
    for pair in cep.get("pairs", []):
        pid = pair["pair_id"]
        dl_log.append({"pair_id": pid, "sujet_status": "DL_OK_SIM", "corrige_status": "DL_OK_SIM"})
        texts[pid] = {
            "sujet_text": f"[OCR_TEXT_SUJET_{pid}] Exercice 1: Question portant sur le programme. ...",
            "corrige_text": f"[OCR_TEXT_CORRIGE_{pid}] Correction: La reponse attendue est ...",
            "ocr_confidence": round(rng.uniform(0.75, 0.98), 3),
        }
    log_event("DA1_END", f"{len(dl_log)} pairs processed")
    return {"dl_log": dl_log, "texts": texts}

# ─── ATOMS (Qi/RQi EXTRACTION) ───
def extract_atoms(texts):
    log_event("ATOMS_START")
    atoms = []
    for pid in sorted(texts.keys()):
        t = texts[pid]
        stxt = t.get("sujet_text", "")
        ctxt = t.get("corrige_text", "")
        qi = {"id": f"{pid}_Q1", "pair_id": pid, "text": stxt[:200], "source_hash": sha256(stxt)}
        rqi = {"id": f"{pid}_R1", "pair_id": pid, "text": ctxt[:200], "source_hash": sha256(ctxt)}
        atoms.append({"Qi": qi, "RQi": rqi})
    log_event("ATOMS_END", f"{len(atoms)} atoms")
    return atoms

# ─── FRT ───
def compute_frt(atoms):
    log_event("FRT_START")
    themes = {}
    for a in atoms:
        key = a["Qi"]["pair_id"]
        themes[key] = themes.get(key, 0) + 1
    frt_entries = []
    for k, v in sorted(themes.items()):
        frt_entries.append({"theme": k, "frequency": v, "recurrence": 1, "weight": round(v / max(len(atoms), 1), 4)})
    log_event("FRT_END", f"{len(frt_entries)} themes")
    return frt_entries

# ─── QC ───
def run_qc(atoms):
    log_event("QC_START")
    qc = []
    for a in atoms:
        qi, rqi = a["Qi"], a["RQi"]
        valid = len(qi["text"]) > 10 and len(rqi["text"]) > 10
        qc.append({"atom_id": qi["id"], "valid": valid,
                    "qi_hash": qi["source_hash"], "rqi_hash": rqi["source_hash"],
                    "reason": "OK" if valid else "TEXT_TOO_SHORT"})
    log_event("QC_END", f"{sum(1 for q in qc if q['valid'])}/{len(qc)} valid")
    return qc

# ─── ARI / TRIGGERS ───
def compute_ari(atoms, qc, iso2):
    rng = _rng_for_country(iso2)
    profiles = []
    for a in atoms:
        qid = a["Qi"]["id"]
        qc_entry = next((q for q in qc if q["atom_id"] == qid), None)
        if qc_entry and qc_entry["valid"]:
            profiles.append({"atom_id": qid,
                             "difficulty": round(rng.uniform(0.3, 0.9), 3),
                             "discrimination": round(rng.uniform(0.1, 0.7), 3),
                             "source_qi_hash": a["Qi"]["source_hash"]})
    return profiles

def compute_triggers(ari_profiles, frt):
    triggers = []
    for p in ari_profiles:
        if p["difficulty"] > 0.7:
            triggers.append({"atom_id": p["atom_id"], "trigger": "HIGH_DIFFICULTY", "value": p["difficulty"]})
    for f in frt:
        if f["weight"] > 0.2:
            triggers.append({"theme": f["theme"], "trigger": "HIGH_FREQ_THEME", "value": f["weight"]})
    return triggers

# ─── SOE ───
def build_soe(atoms, frt):
    return {"total_atoms": len(atoms), "total_themes": len(frt),
            "top_themes": sorted(frt, key=lambda x: -x["weight"])[:5]}

# ─── GATES ───
def run_gates(cap, vsp, sources, cep, da1, atoms, frt, qc, f1, f2, ari, triggers):
    gates = []
    def gate(name, condition, evidence_ptr):
        gates.append({"gate": name, "verdict": "PASS" if condition else "FAIL", "evidence": evidence_ptr})

    gate("CHK_UI_EVENT_LOG", len(st.session_state.get("ui_events", [])) > 0, "ui_events")

    ncb = chk_no_country_branching()
    gate("CHK_NO_COUNTRY_BRANCHING", ncb["pass"], "chk_no_country_branching_report")

    gate("CHK_DA0_NOT_SIMULATED", sources.get("mode") != "SIMULATION", "sources.mode")

    gate("GATE_DA0", len(sources.get("source_manifest", [])) > 0, "source_manifest")
    gate("GATE_DA1", len(da1.get("dl_log", [])) > 0, "dl_log")
    gate("GATE_TEXT", any(t.get("sujet_text") for t in da1.get("texts", {}).values()), "texts")
    gate("GATE_ATOMS", len(atoms) > 0, "atoms")
    gate("GATE_QC", any(q["valid"] for q in qc), "qc_validated")
    gate("GATE_F1F2", f1.get("status") == "OK" and f2.get("status") == "OK", "f1f2_digests")
    gate("CHK_F1F2_NOT_SIMULATED", f1.get("mode") != "SIMULATION", "f1.mode")
    gate("CHK_CAP_COMPLETENESS", vsp.get("status") == "PASS", "vsp_output")
    gate("GATE_FRT", len(frt) > 0, "frt")
    gate("GATE_ARI", len(ari) > 0, "ari_profiles")

    all_pass = all(g["verdict"] == "PASS" for g in gates)
    return {"gates": gates, "global_verdict": "PASS" if all_pass else "FAIL",
            "no_country_branching_report": ncb}

# ─── FULL PIPELINE ───
def run_pipeline(iso2, _determinism_run=False):
    cap = discover_cap(iso2)
    run_id = f"RUN_{iso2}_{sha256(canonical_json(strip_volatile(cap)))[:8]}"

    if not _determinism_run:
        log_event("PIPELINE_START", f"{iso2} / {run_id}")

    vsp = validate_vsp(cap)
    sources = discover_sources(iso2, cap)
    cep = build_cep(sources)
    da1 = execute_da1(cep, iso2)
    atoms = extract_atoms(da1["texts"])
    frt = compute_frt(atoms)
    qc = run_qc(atoms)
    soe = build_soe(atoms, frt)
    ari = compute_ari(atoms, qc, iso2)
    triggers = compute_triggers(ari, frt)
    f1 = FormulaEngine.compute_f1(atoms, frt, qc)
    f2 = FormulaEngine.compute_f2(f1, ari, triggers)
    gate_report = run_gates(cap, vsp, sources, cep, da1, atoms, frt, qc, f1, f2, ari, triggers)
    fe_audit = FormulaEngine.self_audit()

    snapshot = strip_volatile({
        "cap": cap, "vsp": vsp, "sources": sources["source_manifest"],
        "cep": cep, "atoms": atoms, "frt": frt, "qc": qc,
        "f1": f1, "f2": f2, "ari": ari, "triggers": triggers, "gates": gate_report,
    })

    if _determinism_run:
        return {"snapshot": snapshot}

    det = determinism_check(run_pipeline, iso2)

    artifacts = {}
    for name, data in [
        ("CAP_SEALED", cap), ("VSP_output", vsp), ("SourceManifest", sources["source_manifest"]),
        ("SDA0_HTTP_DIAG", sources["http_diag"]), ("SDA0_STRATEGY_LOG", sources["strategy_log"]),
        ("CEP_pairs", cep), ("DA1_DL_LOG", da1["dl_log"]), ("SOE", soe),
        ("Atoms_Qi_RQi", atoms), ("FRT", frt), ("QC_validated", qc),
        ("F1_call_digest", f1), ("F2_call_digest", f2), ("ARI", ari), ("Triggers", triggers),
        ("FORMULA_PACK_MANIFEST", {"version": FORMULA_PACK["version"], "mode": FORMULA_PACK["mode"],
                                    "sha256": FORMULA_PACK["sha256_manifest"], "fe_audit": fe_audit}),
        ("UI_EVENT_LOG", st.session_state.get("ui_events", [])),
        ("CHK_REPORT", gate_report),
        ("NO_COUNTRY_BRANCHING_REPORT", gate_report.get("no_country_branching_report", {})),
    ]:
        artifacts[name] = write_artifact(run_id, name, data)

    artifacts["DeterminismReport_3runs"] = write_artifact(run_id, "DeterminismReport_3runs", det)

    seal = {
        "run_id": run_id, "iso2": iso2,
        "global_verdict": gate_report["global_verdict"],
        "determinism_pass": det["pass"],
        "artifact_count": len(artifacts) + 1,
        "artifact_hashes": {k: v["sha256"] for k, v in artifacts.items()},
        "formula_engine_audit": fe_audit,
    }
    artifacts["SealReport"] = write_artifact(run_id, "SealReport", seal)

    log_event("PIPELINE_END", run_id)
    return {
        "run_id": run_id, "cap": cap, "vsp": vsp, "sources": sources, "cep": cep,
        "da1": da1, "atoms": atoms, "frt": frt, "qc": qc, "soe": soe,
        "f1": f1, "f2": f2, "ari": ari, "triggers": triggers,
        "gates": gate_report, "seal": seal, "artifacts": artifacts, "determinism": det,
        "snapshot": snapshot,
    }

# ─── UI HELPERS ───
def json_view(data):
    st.json(data if isinstance(data, (dict, list)) else {"value": data})

def no_data_msg():
    st.info("⏳ Activate a country from the sidebar to populate this tab.")

# ─── MAIN ───
def main():
    st.set_page_config(page_title="SMAXIA GTE v14.3.3", layout="wide")
    st.title("🛡️ SMAXIA GTE Kernel v14.3.3 — ISO-PROD")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "ui_events" not in st.session_state:
        st.session_state.ui_events = []

    # ─── SIDEBAR: Country Activation ───
    with st.sidebar:
        st.markdown("### 🌍 Activate Country")
        q = st.text_input("Type country name or code", key="cc_query", placeholder="ex: France, Senegal, FR, SN")
        pm, fb = typeahead(q, limit=20)
        options = pm if pm else fb
        labels = [f"{e['name']} ({e['code']})" for e in options]
        cc = None

        if labels:
            st.caption(f"{len(labels)} match{'es' if len(labels) != 1 else ''}")
            display_labels = labels[:12]
            choice = st.radio("Matches", options=display_labels, index=0, label_visibility="visible")
            idx = display_labels.index(choice)
            cc = options[idx]["code"]
            if len(labels) > 12:
                more_labels = labels[12:]
                extra = st.selectbox("More matches…", options=["—"] + more_labels)
                if extra != "—":
                    eidx = labels.index(extra)
                    cc = options[eidx]["code"]
        else:
            if q and q.strip():
                st.caption("No matches found.")

        if st.button("🚀 ACTIVATE_COUNTRY", type="primary", use_container_width=True):
            if cc and len(cc) == 2:
                log_event("ACTIVATE_COUNTRY", cc)
                with st.spinner(f"Running pipeline for {cc}…"):
                    st.session_state.pipeline = run_pipeline(cc)
                st.success(f"✅ {cc} activated")
            else:
                st.error("Select a valid country from the list.")

        if st.session_state.pipeline:
            st.divider()
            p = st.session_state.pipeline
            st.markdown(f"**Active:** `{p['seal']['iso2']}`")
            v = p["seal"]["global_verdict"]
            if v == "PASS":
                st.markdown(f"**Verdict:** :green[{v}]")
            else:
                st.markdown(f"**Verdict:** :red[{v}]")
            det_status = "PASS" if p["determinism"]["pass"] else "FAIL"
            st.markdown(f"**Determinism:** `{det_status}`")
            st.markdown(f"**Run:** `{p['run_id']}`")

    # ─── 11 TABS ───
    tabs = st.tabs([
        "🛡️ Admin", "📦 CAP/VSP", "🔍 DA0/Sources", "📋 CEP/Pairs", "📄 Text/OCR",
        "🧬 Atoms", "🔧 FRT", "🔎 QC Explorer", "🚦 Gates", "📊 F1/F2", "📁 Artifacts"
    ])

    # TAB 0: Admin
    with tabs[0]:
        st.header("Admin — Dashboard")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            p = st.session_state.pipeline
            seal = p["seal"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Country", seal["iso2"])
            c2.metric("Global Verdict", seal["global_verdict"])
            c3.metric("Determinism", "PASS" if p["determinism"]["pass"] else "FAIL")
            c4.metric("Artifacts", seal["artifact_count"])
            st.subheader("FormulaEngine Self-Audit")
            json_view(FormulaEngine.self_audit())

    # TAB 1: CAP/VSP
    with tabs[1]:
        st.header("📦 CAP / VSP")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            st.subheader("CAP")
            json_view(st.session_state.pipeline["cap"])
            st.subheader("VSP")
            json_view(st.session_state.pipeline["vsp"])

    # TAB 2: DA0/Sources
    with tabs[2]:
        st.header("🔍 DA0 / Sources")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            p = st.session_state.pipeline
            st.subheader("Source Manifest")
            json_view(p["sources"]["source_manifest"])
            st.subheader("HTTP Diag")
            json_view(p["sources"]["http_diag"])
            st.subheader("Strategy Log")
            json_view(p["sources"]["strategy_log"])

    # TAB 3: CEP/Pairs
    with tabs[3]:
        st.header("📋 CEP / Pairs")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            json_view(st.session_state.pipeline["cep"])

    # TAB 4: Text/OCR
    with tabs[4]:
        st.header("📄 Text / OCR")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            texts = st.session_state.pipeline["da1"]["texts"]
            for pid in sorted(texts.keys()):
                t = texts[pid]
                with st.expander(f"Pair: {pid} (conf: {t['ocr_confidence']})"):
                    st.text_area("Sujet", t["sujet_text"], height=80, key=f"s_{pid}", disabled=True)
                    st.text_area("Corrige", t["corrige_text"], height=80, key=f"c_{pid}", disabled=True)

    # TAB 5: Atoms
    with tabs[5]:
        st.header("🧬 Atoms (Qi / RQi)")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            json_view(st.session_state.pipeline["atoms"])

    # TAB 6: FRT
    with tabs[6]:
        st.header("🔧 FRT")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            json_view(st.session_state.pipeline["frt"])

    # TAB 7: QC Explorer
    with tabs[7]:
        st.header("🔎 QC Explorer")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            qc = st.session_state.pipeline["qc"]
            valid_count = sum(1 for q in qc if q["valid"])
            st.metric("Valid / Total", f"{valid_count} / {len(qc)}")
            json_view(qc)

    # TAB 8: Gates
    with tabs[8]:
        st.header("🚦 Gates")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            gr = st.session_state.pipeline["gates"]
            verdict = gr["global_verdict"]
            if verdict == "PASS":
                st.success(f"Global Verdict: {verdict}")
            else:
                st.error(f"Global Verdict: {verdict}")
            for g in gr["gates"]:
                icon = "✅" if g["verdict"] == "PASS" else "❌"
                st.write(f"{icon} **{g['gate']}** → {g['verdict']}  (evidence: `{g['evidence']}`)")

    # TAB 9: F1/F2
    with tabs[9]:
        st.header("📊 F1 / F2")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            p = st.session_state.pipeline
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("F1 Digest")
                json_view(p["f1"])
            with c2:
                st.subheader("F2 Digest")
                json_view(p["f2"])
            st.subheader("ARI Profiles")
            json_view(p["ari"])
            st.subheader("Triggers")
            json_view(p["triggers"])

    # TAB 10: Artifacts
    with tabs[10]:
        st.header("📁 Artifacts")
        if not st.session_state.pipeline:
            no_data_msg()
        else:
            arts = st.session_state.pipeline["artifacts"]
            for name, meta in sorted(arts.items()):
                st.write(f"📄 **{name}** — `{meta['sha256'][:16]}…` — `{meta['path']}`")
            st.subheader("SealReport")
            json_view(st.session_state.pipeline["seal"])
            st.subheader("Determinism Report (3 runs)")
            json_view(st.session_state.pipeline["determinism"])

if __name__ == "__main__":
    main()
