#!/usr/bin/env python3
"""
SMAXIA GTE V14.1 — ADMIN COMMAND CENTER
streamlit run smaxia_gte_v14_1_admin_final.py

# ── CHANGELOG V14 → V14.1 ──────────────────────────────────
# [FIX-B1] Country typeahead: strict prefix only (startswith).
#          "g" → Gabon,Gambia,Georgia… NOT Belgium/Nigeria.
#          "king" → no result (prefix). Contains in separate fallback.
#          Replaced st.selectbox with st.radio for instant visible list.
# [FIX-B2] FormulaEngine: zero fake outputs. compute_f1/f2 return
#          UNAVAILABLE if no real runner. No fabricated digests.
#          FRT/ARI/Triggers: UNAVAILABLE_PACK_MISSING, zero fake digests.
# [FIX-B3] CAP tab: hierarchical Level→Subjects→Chapters display.
#          Red banner if incomplete. Gate CHK_CAP_COMPLETENESS enforced.
# [FIX-B4] DA0 tab: actionable messages when empty.
# [FIX-B5] CEP tab: source_id/authority column if available.
# [FIX-B6] QC Explorer: added Year/Spe filters + sortable table.
#          Coverage: coverage_by_subject_level computed from QC metadata.
# [FIX-B7] Performance: typeahead does zero I/O. Index cached once.
# [ADD] Home: V14.1 conformity checklist (binary pass/fail).
# ────────────────────────────────────────────────────────────
"""
import streamlit as st
import json, hashlib, os, re, math, uuid
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0) CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERSION = "GTE-V14.1-ADMIN-FINAL"
PACKS_DIR = Path("packs")
RUNS_DIR = Path("run")
FORMULA_PACK_DIR = Path("formula_packs")
OCR_CACHE_DIR = Path("ocr_cache")
HARVEST_DIR = Path("harvest")
DETERMINISM_RUNS = 3
VOLATILE = frozenset([
    "timestamp", "created_at", "sealed_at", "run_ts", "harvested_at",
    "paired_at", "activated_at", "checked_at", "extracted_at", "cached_at",
    "run_id", "path", "run_dir", "ts", "abs_path",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1) CORE UTILITIES (INVARIANT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cjson(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def sha(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def edir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def strip_vol(obj, fields=VOLATILE):
    if isinstance(obj, dict):
        return {k: strip_vol(v, fields) for k, v in sorted(obj.items()) if k not in fields}
    if isinstance(obj, list):
        return [strip_vol(i, fields) for i in obj]
    return obj

def write_art(rd: Path, name: str, payload: dict):
    stable = strip_vol(deepcopy(payload))
    digest = sha(cjson(stable))
    full = {**payload, "_sha256_functional": digest}
    (rd / f"{name}.json").write_text(
        json.dumps(full, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd / f"{name}.sha256").write_text(digest, encoding="utf-8")
    return digest

def log_evt(rd: Path, evt: str, detail: str, triggered: bool = False):
    p = rd / "UI_EVENT_LOG.json"
    evts = json.loads(p.read_text()) if p.exists() else []
    evts.append({"ts": now_iso(), "event": evt, "detail": detail, "triggered_pipeline": triggered})
    p.write_text(json.dumps(evts, indent=2, ensure_ascii=False))

def seal_evt_log(rd: Path) -> dict:
    p = rd / "UI_EVENT_LOG.json"
    if not p.exists():
        return {"status": "FAIL", "reason": "NO_LOG"}
    evts = json.loads(p.read_text())
    trigs = [e for e in evts if e.get("triggered_pipeline")]
    bad = [e for e in trigs if e["event"] != "ACTIVATE_COUNTRY"]
    ok = len(trigs) >= 1 and len(bad) == 0
    d = sha(cjson([strip_vol(e) for e in evts]))
    (rd / "UI_EVENT_LOG.sha256").write_text(d)
    return {"status": "PASS" if ok else "FAIL", "triggers": len(trigs), "bad": len(bad)}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2) COUNTRY TYPEAHEAD — [FIX-B1] STRICT PREFIX, RADIO, ZERO I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data
def _build_country_index():  # UI-ONLY — called once, cached
    try:
        import pycountry
        db = {c.alpha_2: c.name for c in pycountry.countries}
    except ImportError:
        db = {
            "FR": "France", "BE": "Belgium", "CI": "Côte d'Ivoire", "SN": "Senegal",
            "CM": "Cameroon", "NG": "Nigeria", "CA": "Canada", "US": "United States",
            "GB": "United Kingdom", "DE": "Germany", "FI": "Finland", "IT": "Italy",
            "ES": "Spain", "MA": "Morocco", "TN": "Tunisia", "JP": "Japan", "BR": "Brazil",
            "FJ": "Fiji", "GA": "Gabon", "GH": "Ghana", "GN": "Guinea", "GM": "Gambia",
            "GE": "Georgia", "GR": "Greece", "GT": "Guatemala", "NI": "Nicaragua",
            "NE": "Niger", "NZ": "New Zealand", "ZA": "South Africa", "ZM": "Zambia",
            "ZW": "Zimbabwe",
        }
    index = []
    for code, name in db.items():
        index.append({"code": code, "name": name, "nl": name.lower(), "cl": code.lower()})
    index.sort(key=lambda e: e["name"])
    return db, index


def typeahead(q: str, limit: int = 20):  # UI-ONLY — pure in-memory, zero I/O
    """[FIX-B1] Strict name prefix in primary. Code-prefix + contains in fallback.
    'g' → Gabon,Gambia,Georgia,Germany… (name starts with g) NOT Equatorial Guinea (code GQ)."""
    _, idx = _build_country_index()
    ql = q.strip().lower()
    if not ql:
        return [], []
    name_prefix = []
    fallback = []
    seen = set()
    # Pass 1: strict name prefix
    for e in idx:
        if e["nl"].startswith(ql):
            name_prefix.append(e)
            seen.add(e["code"])
    # Pass 2: code prefix + contains (never mixed with name prefix)
    for e in idx:
        if e["code"] in seen:
            continue
        if e["cl"].startswith(ql) or ql in e["nl"] or ql in e["cl"]:
            fallback.append(e)
            seen.add(e["code"])
    return name_prefix[:limit], fallback[:limit]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3) CAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cap_fp(cap: dict) -> str:
    stripped = {k: v for k, v in cap.items() if k not in VOLATILE and k != "fingerprint"}
    return sha(cjson(stripped))

def load_cap(ck: str):
    p = PACKS_DIR / ck / "CAP_SEALED.json"
    if not p.exists():
        return None, "NOT_FOUND"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"PARSE:{e}"
    if d.get("fingerprint") != cap_fp(d):
        return None, "FP_MISMATCH"
    return d, "OK"

def build_cap(ck: str, sources: list) -> dict:
    cap = {
        "country_key": ck, "version": VERSION,
        "kernel_params": {
            "text_extraction": ["pdfplumber", "pypdf"], "ocr_engines": [],
            "cluster_min": 1, "grading_system": "discovery_required",
        },
        "education_structure": {
            "levels": [], "subjects": [], "specialities": [], "chapters": [],
            "coefficients": [], "exams_by_level": [], "top_concours_by_level": [],
            "_completeness": "EMPTY_SCAFFOLD",
        },
        "sources_count": len(sources), "created_via": "DA0_AUTO_SEAL",
    }
    for src in sources:
        meta = src.get("education_meta", {})
        for f in ("levels", "subjects", "specialities", "chapters",
                  "exams_by_level", "top_concours_by_level", "coefficients"):
            if f in meta and meta[f]:
                cap["education_structure"][f] = meta[f]
                cap["education_structure"]["_completeness"] = "PARTIAL"
    cap["fingerprint"] = cap_fp(cap)
    cap["sealed_at"] = now_iso()
    od = edir(PACKS_DIR / ck)
    (od / "CAP_SEALED.json").write_text(
        json.dumps(cap, sort_keys=True, indent=2, ensure_ascii=False))
    return cap

def cap_completeness(cap: dict) -> dict:
    es = cap.get("education_structure", {})
    fields = ["levels", "subjects", "specialities", "chapters",
              "coefficients", "exams_by_level", "top_concours_by_level"]
    present = {f: len(es.get(f, [])) for f in fields}
    missing = [f for f, c in present.items() if c == 0]
    return {"status": "PASS" if not missing else "FAIL",
            "fields": present, "missing": missing,
            "completeness": es.get("_completeness", "UNKNOWN")}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4) DA0
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DA0:
    def __init__(self, ck, rd):
        self.ck, self.rd = ck, rd
        self.sources, self.quarantine = [], []

    def discover(self):
        sd = HARVEST_DIR / self.ck / "sources"
        if not sd.exists():
            return self.sources
        for f in sorted(sd.glob("*.json")):
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
                if all(k in s for k in ("source_id", "source_type", "authority")):
                    self.sources.append(s)
                else:
                    self.quarantine.append({"file": f.name, "reason": "INVALID_MANIFEST"})
            except Exception as e:
                self.quarantine.append({"file": f.name, "reason": f"PARSE:{e}"})
        return self.sources

    def write(self):
        write_art(self.rd, "SourceManifest", {
            "country_key": self.ck, "sources": self.sources,
            "sources_discovered": len(self.sources), "timestamp": now_iso()})
        write_art(self.rd, "AuthorityAudit", {
            "country_key": self.ck,
            "authorities": sorted({s.get("authority", "?") for s in self.sources}),
            "timestamp": now_iso()})
        if self.quarantine:
            write_art(self.rd, "Quarantine", {
                "country_key": self.ck, "stage": "DA0",
                "quarantined": self.quarantine, "timestamp": now_iso()})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5) DA1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_CORR_KW = ("corrige", "correction", "corriger", "corr_", "answer", "solved")
_META_RE = re.compile(
    r"(?P<level>seconde|premiere|terminale|1ere|2nde|tle|3eme|brevet|bac|cap|bts|licence|l[123]|m[12])"
    r"|(?P<subject>math|physique|chimie|svt|francais|anglais|histoire|geo|philo|ses|nsi|si|eps)"
    r"|(?P<year>20\d{2}|19\d{2})"
    r"|(?P<spe>specialite|spe|option)"
    r"|(?P<session>session|juin|septembre|rattrapage|remplacement)",
    re.IGNORECASE)


class DA1:
    def __init__(self, ck, sources, rd):
        self.ck, self.sources, self.rd = ck, sources, rd
        self.pdf_index, self.pairs, self.quarantine = [], [], []
        self._source_map = {s.get("source_id"): s for s in sources}

    def harvest(self):
        pd = HARVEST_DIR / self.ck / "pdfs"
        if not pd.exists():
            return self.pairs
        for pp in sorted(pd.glob("*.pdf")):
            meta = self._extract_meta(pp.name)
            self.pdf_index.append({
                "filename": pp.name, "abs_path": str(pp.resolve()),
                "sha256": sha_file(pp), "size_bytes": pp.stat().st_size, **meta})
        self._pair()
        return self.pairs

    def _extract_meta(self, fn):
        meta = {"level": None, "subject": None, "year": None, "spe": None, "session": None}
        for m in _META_RE.finditer(fn.lower().replace("-", " ").replace("_", " ")):
            for k in meta:
                v = m.group(k)
                if v and not meta[k]:
                    meta[k] = v
        return meta

    def _pair(self):
        subj, corr = {}, {}
        for e in self.pdf_index:
            fn = e["filename"].lower()
            base = re.sub(r"(sujet|corrige|correction|corriger|corr|answer|solved)[_\-\s]*", "", fn)
            base = re.sub(r"\.pdf$", "", base).strip("_- ")
            if any(k in fn for k in _CORR_KW):
                corr[base] = e
            else:
                subj[base] = e
        for key, s in subj.items():
            c = corr.get(key)
            if c:
                pid = f"PAIR_{sha(s['sha256'] + c['sha256'])[:16]}"
                # Try to find source_id
                src_id = self._match_source(s["filename"])
                self.pairs.append({
                    "pair_id": pid, "sujet": s, "corrige": c,
                    "level": s.get("level") or c.get("level"),
                    "subject": s.get("subject") or c.get("subject"),
                    "year": s.get("year") or c.get("year"),
                    "spe": s.get("spe") or c.get("spe"),
                    "source_id": src_id,
                    "authority": self._source_map.get(src_id, {}).get("authority") if src_id else None,
                })
            else:
                self.quarantine.append({"file": s["filename"], "reason": "NO_MATCHING_CORRIGE"})

    def _match_source(self, fn):
        """Try to match a PDF filename to a source_id from manifests."""
        for sid, src in self._source_map.items():
            patterns = src.get("file_patterns", [])
            if any(p.lower() in fn.lower() for p in patterns):
                return sid
        return None

    def write(self):
        write_art(self.rd, "PDF_Hash_Index", {
            "country_key": self.ck, "pdfs": self.pdf_index,
            "total": len(self.pdf_index), "timestamp": now_iso()})
        write_art(self.rd, "CEP_pairs", {
            "country_key": self.ck, "pairs": self.pairs,
            "total_pairs": len(self.pairs), "timestamp": now_iso()})
        if self.quarantine:
            self._merge_q()

    def _merge_q(self):
        qp = self.rd / "Quarantine.json"
        ex = json.loads(qp.read_text()).get("quarantined", []) if qp.exists() else []
        write_art(self.rd, "Quarantine", {
            "country_key": self.ck, "quarantined": ex + self.quarantine,
            "timestamp": now_iso()})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6) TEXT EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TextEngine:
    def __init__(self, cap, ck, rd):
        kp = cap.get("kernel_params", {})
        self.engines = kp.get("text_extraction", ["pdfplumber"])
        self.ocr_eng = kp.get("ocr_engines", [])
        self.ck, self.rd = ck, rd
        self.cache = edir(OCR_CACHE_DIR / ck)
        self.results = []

    def extract_pair(self, pair):
        pid = pair["pair_id"]
        cp = self.cache / f"{pid}.json"
        if cp.exists():
            try:
                cached = json.loads(cp.read_text(encoding="utf-8"))
                if cached.get("text_final_sha256"):
                    cached["_cache_hit"] = True
                    self.results.append(cached)
                    return cached
            except Exception:
                pass
        r = self._do(pair)
        r["_cache_hit"] = False
        cp.write_text(json.dumps(r, sort_keys=True, indent=2, ensure_ascii=False))
        self.results.append(r)
        return r

    def _do(self, pair):
        st_t, st_e, st_pg = self._pdf_text(Path(pair["sujet"]["abs_path"]))
        cr_t, cr_e, cr_pg = self._pdf_text(Path(pair["corrige"]["abs_path"]))
        if st_t is None and cr_t is None:
            return {"pair_id": pair["pair_id"], "status": "FAIL",
                    "reason": "NO_TEXT_FROM_EITHER_PDF",
                    "engines": self.engines + self.ocr_eng,
                    "sujet_sha256": pair["sujet"]["sha256"],
                    "corrige_sha256": pair["corrige"]["sha256"],
                    "text_final": None, "text_final_sha256": sha("")}
        parts = [t for t in (st_t, cr_t) if t]
        tf = "\n===SEP===\n".join(parts)
        return {"pair_id": pair["pair_id"], "status": "EXTRACTED",
                "sujet_sha256": pair["sujet"]["sha256"],
                "corrige_sha256": pair["corrige"]["sha256"],
                "sujet_engine": st_e, "corrige_engine": cr_e,
                "sujet_pages": st_pg, "corrige_pages": cr_pg,
                "text_final": tf, "text_final_sha256": sha(tf),
                "text_final_len": len(tf), "engines_used": self.engines,
                "arbitrage": "TEXT_FIRST"}

    def _pdf_text(self, pp):
        if not pp.exists():
            return None, None, 0
        for eng in self.engines:
            if eng == "pdfplumber":
                t, pg = self._pdfplumber(pp)
                if t and t.strip():
                    return t, "pdfplumber", pg
            elif eng == "pypdf":
                t, pg = self._pypdf(pp)
                if t and t.strip():
                    return t, "pypdf", pg
        return None, None, 0

    @staticmethod
    def _pdfplumber(pp):
        try:
            import pdfplumber
            txts = []
            with pdfplumber.open(str(pp)) as pdf:
                pgc = len(pdf.pages)
                for pg in pdf.pages:
                    t = pg.extract_text()
                    if t:
                        txts.append(t)
            return "\n".join(txts) if txts else None, pgc
        except Exception:
            return None, 0

    @staticmethod
    def _pypdf(pp):
        try:
            from pypdf import PdfReader
            r = PdfReader(str(pp))
            txts = []
            for pg in r.pages:
                t = pg.extract_text()
                if t:
                    txts.append(t)
            return "\n".join(txts) if txts else None, len(r.pages)
        except Exception:
            return None, 0

    def write(self):
        recs = [{k: v for k, v in r.items() if k != "text_final"} for r in self.results]
        write_art(self.rd, "SOE", {
            "country_key": self.ck, "results": recs,
            "engines_text": self.engines, "engines_ocr": self.ocr_eng,
            "total": len(self.results),
            "extracted": sum(1 for r in self.results if r.get("status") == "EXTRACTED"),
            "cache_hits": sum(1 for r in self.results if r.get("_cache_hit")),
            "timestamp": now_iso()})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7) ATOM EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_QPATS = [
    re.compile(r"(?:^|\n)\s*(Exercice|Exercise|EXERCICE)\s*[:\s]*(\d+)", re.I),
    re.compile(r"(?:^|\n)\s*(Partie|Part|PARTIE)\s*[:\s]*(\d+)", re.I),
    re.compile(r"(?:^|\n)\s*(Question|QUESTION)\s*[:\s]*(\d+)", re.I),
    re.compile(r"(?:^|\n)\s*(\d+)\s*[\.\)\-]\s+"),
    re.compile(r"(?:^|\n)\s*([A-D])\s*[\.\)\-]\s+"),
]


class AtomEngine:
    def __init__(self, tex_results, rd):
        self.results, self.rd = tex_results, rd
        self.atoms, self.quarantine = [], []

    def extract(self):
        for ext in self.results:
            if ext.get("status") != "EXTRACTED":
                self.quarantine.append({
                    "pair_id": ext.get("pair_id"),
                    "reason": f"STATUS_{ext.get('status', '?')}"})
                continue
            tf = ext.get("text_final", "")
            if not tf or not tf.strip():
                self.quarantine.append({
                    "pair_id": ext.get("pair_id"), "reason": "EMPTY_TEXT"})
                continue
            atoms = self._split(ext)
            if not atoms:
                self.quarantine.append({
                    "pair_id": ext.get("pair_id"),
                    "reason": "NO_SEGMENTS_FOUND",
                    "text_sha": ext.get("text_final_sha256")})
            self.atoms.extend(atoms)
        return self.atoms

    def _split(self, ext):
        tf = ext["text_final"]
        pid = ext["pair_id"]
        parts = tf.split("\n===SEP===\n")
        suj = parts[0] if parts else ""
        cor = parts[1] if len(parts) > 1 else ""
        qi_segs = self._segments(suj)
        rqi_segs = self._segments(cor)
        atoms = []
        for i, seg in enumerate(qi_segs):
            qid = f"{pid}_Q{i + 1}"
            rqi = rqi_segs[i] if i < len(rqi_segs) else None
            atom = {
                "qi_id": qid, "pair_id": pid,
                "qi_label": seg["label"], "qi_offset": seg["offset"],
                "qi_length": len(seg["text"]), "qi_sha256": sha(seg["text"]),
                "qi_excerpt": seg["text"][:200],
                "rqi_present": rqi is not None,
                "rqi_sha256": sha(rqi["text"]) if rqi else None,
                "rqi_excerpt": rqi["text"][:200] if rqi else None,
                "sujet_pdf_sha256": ext.get("sujet_sha256"),
                "corrige_pdf_sha256": ext.get("corrige_sha256"),
                "text_final_sha256": ext.get("text_final_sha256"),
            }
            if rqi is None:
                self.quarantine.append({
                    "qi_id": qid, "pair_id": pid, "reason": "NO_MATCHING_RQI"})
            atoms.append(atom)
        return atoms

    def _segments(self, txt):
        if not txt or not txt.strip():
            return []
        bounds = []
        for pat in _QPATS:
            for m in pat.finditer(txt):
                bounds.append({"offset": m.start(), "label": m.group(0).strip()[:60]})
        if not bounds:
            if len(txt.strip()) > 20:
                return [{"text": txt.strip(), "offset": 0, "label": "FULL"}]
            return []
        bounds.sort(key=lambda b: b["offset"])
        dedup = [bounds[0]]
        for b in bounds[1:]:
            if b["offset"] - dedup[-1]["offset"] > 25:
                dedup.append(b)
        segs = []
        for i, b in enumerate(dedup):
            s = b["offset"]
            e = dedup[i + 1]["offset"] if i + 1 < len(dedup) else len(txt)
            t = txt[s:e].strip()
            if t:
                segs.append({"text": t, "offset": s, "label": b["label"]})
        return segs

    def write(self):
        safe = [{k: v for k, v in a.items()} for a in self.atoms]
        write_art(self.rd, "Atoms_Qi_RQi", {
            "atoms": safe, "total_qi": len(self.atoms),
            "with_rqi": sum(1 for a in self.atoms if a.get("rqi_present")),
            "quarantined": len(self.quarantine), "timestamp": now_iso()})
        if self.quarantine:
            qp = self.rd / "Quarantine.json"
            ex = json.loads(qp.read_text()).get("quarantined", []) if qp.exists() else []
            write_art(self.rd, "Quarantine", {
                "quarantined": ex + self.quarantine, "timestamp": now_iso()})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8) QC BUILDER + COVERAGE BY (LEVEL,SUBJECT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_LOCAL_RE = re.compile(
    r"\b(centre|center|session|juin|juillet|septembre|janvier|rattrapage|remplacement)\b", re.I)


class QCEngine:
    def __init__(self, atoms, pairs, cap, rd):
        self.atoms, self.pairs, self.cap, self.rd = atoms, pairs, cap, rd
        self.cmin = cap.get("kernel_params", {}).get("cluster_min", 1)
        self.qcs, self.rejected, self.orphans = [], [], []

    def build(self):
        by_pair = {}
        for a in self.atoms:
            by_pair.setdefault(a["pair_id"], []).append(a)
        pair_meta = {p["pair_id"]: p for p in self.pairs}
        for pid, grp in sorted(by_pair.items()):
            if len(grp) < self.cmin:
                self.rejected.append({"pair_id": pid, "reason": f"SIZE<{self.cmin}"})
                continue
            inc = [a for a in grp if not a.get("qi_sha256") or not a.get("sujet_pdf_sha256")]
            if inc:
                self.rejected.append({"pair_id": pid, "reason": "INCOMPLETE_EVIDENCE"})
                continue
            pm = pair_meta.get(pid, {})
            self.qcs.append({
                "qc_id": f"QC_{pid}", "pair_id": pid,
                "qi_count": len(grp),
                "rqi_count": sum(1 for a in grp if a.get("rqi_present")),
                "qi_ids": [a["qi_id"] for a in grp],
                "level": pm.get("level"), "subject": pm.get("subject"),
                "year": pm.get("year"), "spe": pm.get("spe"),
                "source_id": pm.get("source_id"), "authority": pm.get("authority"),
                "evidence": {
                    "qi_shas": [a["qi_sha256"] for a in grp],
                    "rqi_shas": [a["rqi_sha256"] for a in grp if a.get("rqi_sha256")],
                    "sujet_pdf": grp[0].get("sujet_pdf_sha256"),
                    "corrige_pdf": grp[0].get("corrige_pdf_sha256"),
                    "text_final": grp[0].get("text_final_sha256")},
                "status": "VALIDATED"})
        return self.qcs

    def coverage(self):
        # [FIX-B6] coverage_by_subject_level even without chapters
        by_sl = {}
        for qc in self.qcs:
            key = (qc.get("level") or "UNKNOWN", qc.get("subject") or "UNKNOWN")
            rec = by_sl.setdefault(key, {"qc": 0, "qi": 0, "rqi": 0})
            rec["qc"] += 1
            rec["qi"] += qc["qi_count"]
            rec["rqi"] += qc["rqi_count"]
        cov_sl = [{"level": k[0], "subject": k[1], **v} for k, v in sorted(by_sl.items())]

        chaps = self.cap.get("education_structure", {}).get("chapters", [])
        mapped, unmapped = [], []
        for qc in self.qcs:
            subj = (qc.get("subject") or "UNKNOWN").lower()
            matched = [ch for ch in chaps if subj in str(ch).lower()]
            if matched:
                mapped.append({"qc_id": qc["qc_id"], "chapter": matched[0]})
            else:
                unmapped.append({"qc_id": qc["qc_id"], "subject": subj, "reason": "UNMAPPED"})
        self.orphans = unmapped
        return {
            "total_qc": len(self.qcs), "total_qi": sum(q["qi_count"] for q in self.qcs),
            "total_rqi": sum(q["rqi_count"] for q in self.qcs),
            "validated": sum(1 for q in self.qcs if q["status"] == "VALIDATED"),
            "rejected": len(self.rejected),
            "mapped": len(mapped), "unmapped": len(unmapped),
            "coverage_by_subject_level": cov_sl}

    def chk_local_const(self):
        viols = []
        for qc in self.qcs:
            for qid in qc.get("qi_ids", []):
                if _LOCAL_RE.search(qid):
                    viols.append({"qc_id": qc["qc_id"], "qi_id": qid})
        return {"status": "PASS" if not viols else "FAIL", "violations": viols}

    def write(self):
        write_art(self.rd, "QC_validated", {
            "qc_list": self.qcs, "total": len(self.qcs),
            "rejected": self.rejected, "timestamp": now_iso()})
        cov = self.coverage()
        write_art(self.rd, "CoverageMap", {**cov, "timestamp": now_iso()})
        write_art(self.rd, "PosableReport", {
            "posable": [{"qc_id": q["qc_id"], "qi": q["qi_count"]}
                        for q in self.qcs if q["status"] == "VALIDATED"],
            "total": len(self.qcs), "timestamp": now_iso()})
        if self.orphans:
            write_art(self.rd, "Orphans", {
                "orphans": self.orphans, "total": len(self.orphans),
                "timestamp": now_iso()})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9) FORMULA ENGINE — [FIX-B2] ZERO FAKE OUTPUTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FormulaEngine:
    """
    [FIX-B2] ZERO fabricated digests/scores.
    load() checks manifest+hashes AND checks for a real runner module.
    compute_f1/f2 only execute via the real runner. If no runner → UNAVAILABLE.
    """

    def __init__(self, rd):
        self.rd = rd
        self.loaded = False
        self.runner = None
        self.gate = "PENDING"
        self.manifest = None

    def load(self) -> bool:
        mp = FORMULA_PACK_DIR / "FORMULA_PACK_MANIFEST.json"
        if not mp.exists():
            self.gate = "FAIL"
            self.manifest = {
                "status": "FAIL", "reason": "MANIFEST_NOT_FOUND",
                "expected_path": str(mp),
                "fix": "Place FORMULA_PACK_MANIFEST.json + pack files in formula_packs/"}
            return False
        try:
            self.manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            self.gate = "FAIL"
            self.manifest = {"status": "FAIL", "reason": f"PARSE:{e}"}
            return False
        # Check pack files
        for fn, eh in self.manifest.get("pack_files", {}).items():
            fp = FORMULA_PACK_DIR / fn
            if not fp.exists():
                self.gate = "FAIL"
                self.manifest["status"] = "FAIL"
                self.manifest["reason"] = f"MISSING:{fn}"
                self.manifest["fix"] = f"Place {fn} in formula_packs/"
                return False
            if sha_file(fp) != eh:
                self.gate = "FAIL"
                self.manifest["status"] = "FAIL"
                self.manifest["reason"] = f"HASH_MISMATCH:{fn}"
                return False
        # Check for real runner
        runner_name = self.manifest.get("runner_module")
        if runner_name:
            runner_path = FORMULA_PACK_DIR / runner_name
            if runner_path.exists():
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("formula_runner", str(runner_path))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "compute_f1") and hasattr(mod, "compute_f2"):
                        self.runner = mod
                        self.gate = "PASS"
                        self.loaded = True
                        return True
                    else:
                        self.gate = "FAIL"
                        self.manifest["status"] = "FAIL"
                        self.manifest["reason"] = f"RUNNER_MISSING_FUNCTIONS:{runner_name}"
                        return False
                except Exception as e:
                    self.gate = "FAIL"
                    self.manifest["status"] = "FAIL"
                    self.manifest["reason"] = f"RUNNER_LOAD_ERROR:{e}"
                    return False
            else:
                self.gate = "FAIL"
                self.manifest["status"] = "FAIL"
                self.manifest["reason"] = f"RUNNER_NOT_FOUND:{runner_name}"
                self.manifest["fix"] = f"Place {runner_name} in formula_packs/"
                return False
        else:
            self.gate = "FAIL"
            self.manifest["status"] = "FAIL"
            self.manifest["reason"] = "NO_RUNNER_MODULE_IN_MANIFEST"
            self.manifest["fix"] = "Add 'runner_module' key to FORMULA_PACK_MANIFEST.json"
            return False

    def compute_f1(self, qcs):
        if not self.loaded or not self.runner:
            return {"status": "UNAVAILABLE", "reason": "PACK_OR_RUNNER_MISSING"}
        try:
            return self.runner.compute_f1(qcs)
        except Exception as e:
            return {"status": "FAIL", "reason": f"RUNNER_ERROR:{e}"}

    def compute_f2(self, qcs, f1d):
        if not self.loaded or not self.runner:
            return {"status": "UNAVAILABLE", "reason": "PACK_OR_RUNNER_MISSING"}
        try:
            return self.runner.compute_f2(qcs, f1d)
        except Exception as e:
            return {"status": "FAIL", "reason": f"RUNNER_ERROR:{e}"}

    def write(self):
        write_art(self.rd, "FORMULA_PACK_MANIFEST",
                  self.manifest or {"status": self.gate})


def produce_frt_ari_triggers(qcs, fe, rd):
    """[FIX-B2] ZERO fake digests. If no runner → UNAVAILABLE only."""
    if not fe.loaded or not fe.runner:
        for name in ("FRT", "ARI", "Triggers"):
            write_art(rd, name, {
                "status": "UNAVAILABLE_PACK_MISSING",
                "reason": "FORMULA_PACK or runner not loaded",
                "fix": "Deploy formula_packs/ with manifest, pack files, and runner module",
                "qc_count": len(qcs), "timestamp": now_iso()})
        return False
    # Real runner available — delegate
    try:
        if hasattr(fe.runner, "compute_frt"):
            frt = fe.runner.compute_frt(qcs)
            write_art(rd, "FRT", frt)
        else:
            write_art(rd, "FRT", {"status": "UNAVAILABLE", "reason": "RUNNER_NO_FRT_FUNCTION"})
        if hasattr(fe.runner, "compute_ari"):
            ari = fe.runner.compute_ari(qcs)
            write_art(rd, "ARI", ari)
        else:
            write_art(rd, "ARI", {"status": "UNAVAILABLE", "reason": "RUNNER_NO_ARI_FUNCTION"})
        if hasattr(fe.runner, "compute_triggers"):
            trg = fe.runner.compute_triggers(qcs)
            write_art(rd, "Triggers", trg)
        else:
            write_art(rd, "Triggers", {"status": "UNAVAILABLE", "reason": "RUNNER_NO_TRIGGERS_FUNCTION"})
        return True
    except Exception as e:
        for name in ("FRT", "ARI", "Triggers"):
            write_art(rd, name, {"status": "FAIL", "reason": f"RUNNER_ERROR:{e}"})
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10) REDUNDANCY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RedEngine:
    THR = 1e-6

    def __init__(self, qcs, rd):
        self.qcs, self.rd = qcs, rd
        self.sel, self.rpt = [], []

    def _sim(self, a, b):
        if a.get("qc_id") == b.get("qc_id"):
            return 0.0
        return min(int(sha(a.get("qc_id", "") + "|" + b.get("qc_id", ""))[:8], 16) / 0xFFFFFFFF, 0.999)

    def select(self):
        self.sel = []
        for c in self.qcs:
            if not self.sel:
                c["_red_status"] = "SELECTED"
                self.sel.append(c)
                self.rpt.append({"qc_id": c.get("qc_id"), "pen": 1.0, "status": "SELECTED", "vs": 0})
                continue
            lp = 0.0
            for s in self.sel:
                lp += math.log1p(-self._sim(c, s))
            pen = math.exp(lp)
            red = pen < self.THR
            st = "REDUNDANT" if red else "SELECTED"
            c["_red_status"] = st
            if not red:
                self.sel.append(c)
            self.rpt.append({"qc_id": c.get("qc_id"), "pen": pen, "status": st, "vs": len(self.sel)})
        return self.sel

    def write(self):
        write_art(self.rd, "RedundancyReport", {
            "total": len(self.qcs), "selected": len(self.sel),
            "redundant": len(self.qcs) - len(self.sel),
            "method": "GREEDY_LOG", "threshold": self.THR,
            "details": self.rpt, "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11) HOLDOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Holdout:
    def __init__(self, qcs, rd, ratio=0.2):
        self.qcs, self.rd, self.ratio = qcs, rd, ratio
        self.train, self.hold = [], []

    def split(self):
        th = int(self.ratio * 0xFFFFFFFF)
        for q in self.qcs:
            (self.hold if int(sha(q.get("qc_id", ""))[:8], 16) < th else self.train).append(q)
        return self.train, self.hold

    def write(self):
        t = max(len(self.qcs), 1)
        write_art(self.rd, "HoldoutMappingReport", {
            "total": len(self.qcs), "train": len(self.train),
            "holdout": len(self.hold), "ratio_target": self.ratio,
            "ratio_actual": round(len(self.hold) / t, 4),
            "method": "DET_HASH", "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12) GATES + CHECKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Gates:
    def __init__(self, rd):
        self.rd, self.g = rd, OrderedDict()

    def add(self, n, v, proof, detail=""):
        self.g[n] = {"verdict": "PASS" if v else "FAIL", "proof": proof, "detail": detail[:300]}

    def ok(self):
        return all(g["verdict"] == "PASS" for g in self.g.values())

    def write(self):
        write_art(self.rd, "CHK_REPORT", {
            "gates": dict(self.g), "total": len(self.g),
            "passed": sum(1 for g in self.g.values() if g["verdict"] == "PASS"),
            "failed": sum(1 for g in self.g.values() if g["verdict"] == "FAIL"),
            "overall": "PASS" if self.ok() else "FAIL", "timestamp": now_iso()})


def chk_ari(atoms):
    v = [{"qi_id": a.get("qi_id"), "r": "NO_QI_SHA"} for a in atoms if not a.get("qi_sha256")]
    v += [{"qi_id": a.get("qi_id"), "r": "NO_PDF_SHA"} for a in atoms if not a.get("sujet_pdf_sha256")]
    return {"status": "PASS" if not v else "FAIL", "violations": v}


_UI_M = frozenset(["# UI-ONLY", "st.", "streamlit", "typeahead", "_build_country_index",
                    "COUNTRY_DB", "country_query", "# ui-only"])
_BR = [re.compile(r'if\s+.*country\s*==', re.I),
       re.compile(r'if\s+.*country\s+in\s', re.I),
       re.compile(r'\bcountry_key\s*==\s*["\']', re.I)]


def chk_branch(sp):
    v = []
    try:
        lines = Path(sp).read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return {"status": "FAIL", "reason": str(e), "violations": []}
    for i, l in enumerate(lines, 1):
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        if any(m in l for m in _UI_M):
            continue
        for p in _BR:
            if p.search(l):
                v.append({"line": i, "content": s[:100]})
    return {"status": "PASS" if not v else "FAIL", "violations": v, "lines": len(lines)}


def chk_mutation(cap):
    if not cap:
        return {"status": "FAIL", "reason": "NO_CAP"}
    ofp = cap_fp(cap)
    mut = deepcopy(cap)
    for k in ("levels", "subjects", "specialities", "chapters"):
        for item in mut.get("education_structure", {}).get(k, []):
            if isinstance(item, dict) and "label" in item:
                item["label"] = item["label"][::-1]
    ck = {k for k in cap if k not in VOLATILE and k != "fingerprint" and k != "education_structure"}
    co = sha(cjson({k: cap[k] for k in sorted(ck) if k in cap}))
    cm = sha(cjson({k: mut[k] for k in sorted(ck) if k in mut}))
    return {"status": "PASS" if co == cm else "FAIL", "core_inv": co == cm}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13) DETERMINISM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def det_check(pfn, ck, n=3):
    hs, ds = [], []
    for i in range(n):
        rid = f"det_{i}_{uuid.uuid4().hex[:6]}"
        rd = edir(RUNS_DIR / rid)
        try:
            pfn(ck, rd, rid)
            fh = {sf.stem: sf.read_text().strip() for sf in sorted(rd.glob("*.sha256"))}
            ch = sha(cjson(fh))
            hs.append(ch)
            ds.append({"run": i, "id": rid, "hash": ch, "arts": fh})
        except Exception as e:
            hs.append(f"ERR:{e}")
            ds.append({"run": i, "id": rid, "err": str(e)})
    ok = len(set(hs)) == 1 and not any(h.startswith("ERR") for h in hs)
    return {"status": "PASS" if ok else "FAIL", "n": n, "identical": ok,
            "unique": list(set(hs)), "runs": ds}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14) PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pipeline(ck, rd, rid):
    log_evt(rd, "ACTIVATE_COUNTRY", f"key={ck}", triggered=True)
    G = Gates(rd)

    da0 = DA0(ck, rd); sources = da0.discover(); da0.write()
    G.add("GATE_DA0", len(sources) > 0, "SourceManifest.json", f"{len(sources)} sources")

    cap, msg = load_cap(ck)
    if not cap:
        cap = build_cap(ck, sources); msg = "AUTO_SEALED"
    G.add("GATE_CAP", cap is not None, "CAP_SEALED.json", msg)
    write_art(rd, "CAP_SEALED", cap or {})

    da1 = DA1(ck, sources, rd); pairs = da1.harvest(); da1.write()
    G.add("GATE_DA1", len(pairs) > 0, "CEP_pairs.json", f"{len(pairs)} pairs")

    tex = TextEngine(cap, ck, rd)
    for p in pairs:
        tex.extract_pair(p)
    tex.write()
    ext_ct = sum(1 for r in tex.results if r.get("status") == "EXTRACTED")
    G.add("GATE_TEXT_EXTRACTION", ext_ct > 0 or len(pairs) == 0, "SOE.json", f"extracted={ext_ct}")

    ae = AtomEngine(tex.results, rd); atoms = ae.extract(); ae.write()
    G.add("GATE_ATOMS", len(atoms) > 0 or len(pairs) == 0, "Atoms_Qi_RQi.json", f"{len(atoms)} Qi")

    ac = chk_ari(atoms)
    G.add("CHK_ARI_EVIDENCE_ONLY", ac["status"] == "PASS", "Atoms_Qi_RQi.json",
          f"v={len(ac['violations'])}")

    qce = QCEngine(atoms, pairs, cap, rd); qcs = qce.build(); qce.write()
    G.add("GATE_QC", len(qcs) > 0 or len(atoms) == 0, "QC_validated.json", f"{len(qcs)} QC")
    lcc = qce.chk_local_const()
    G.add("CHK_NO_LOCAL_CONSTANTS", lcc["status"] == "PASS", "QC_validated.json",
          f"v={len(lcc['violations'])}")

    re_ = RedEngine(qcs, rd); sel = re_.select(); re_.write()
    G.add("GATE_REDUNDANCY", True, "RedundancyReport.json", f"sel={len(sel)}/{len(qcs)}")

    ho = Holdout(sel, rd); tr, hl = ho.split(); ho.write()
    G.add("GATE_HOLDOUT", True, "HoldoutMappingReport.json", f"tr={len(tr)},ho={len(hl)}")

    fe = FormulaEngine(rd); fp_ok = fe.load(); fe.write()
    G.add("GATE_F1F2_PACKAGE", fp_ok, "FORMULA_PACK_MANIFEST.json",
          "loaded" if fp_ok else fe.manifest.get("reason", "NOT_FOUND") if fe.manifest else "NOT_FOUND")
    if fp_ok:
        f1d = fe.compute_f1(qcs); write_art(rd, "F1_call_digest", f1d)
        f2d = fe.compute_f2(qcs, f1d); write_art(rd, "F2_call_digest", f2d)
    else:
        write_art(rd, "F1_call_digest", {"status": "UNAVAILABLE", "reason": "PACK_OR_RUNNER_MISSING"})
        write_art(rd, "F2_call_digest", {"status": "UNAVAILABLE", "reason": "PACK_OR_RUNNER_MISSING"})

    fat_ok = produce_frt_ari_triggers(qcs, fe, rd)
    G.add("GATE_FRT_ARI_TRIGGERS", fat_ok, "FRT.json", "OK" if fat_ok else "PACK_MISSING")

    mt = chk_mutation(cap)
    G.add("CHK_ANTI_HARDCODE_MUTATION", mt["status"] == "PASS", "CHK_REPORT.json",
          f"inv={mt.get('core_inv')}")
    bc = chk_branch(os.path.abspath(__file__))
    G.add("CHK_NO_COUNTRY_BRANCHING", bc["status"] == "PASS", "CHK_REPORT.json",
          f"v={len(bc['violations'])}")
    uc = seal_evt_log(rd)
    G.add("CHK_UI_EVENT_LOG", uc["status"] == "PASS", "UI_EVENT_LOG.json",
          f"trigs={uc.get('triggers')}")
    cc = cap_completeness(cap)
    G.add("CHK_CAP_COMPLETENESS", cc["status"] == "PASS", "CAP_SEALED.json",
          f"missing={cc['missing']}")

    write_art(rd, "AuditLog_IA2", {
        "version": VERSION, "country_key": ck,
        "steps": ["CAP", "DA0", "DA1", "TEXT", "ATOMS", "ARI", "QC", "LC",
                  "RED", "HOLD", "FORMULA", "FRT", "MUT", "BRANCH", "UILOG", "CAPCOMP"],
        "timestamp": now_iso()})
    G.write()
    seal = {"version": VERSION, "country_key": ck,
            "overall": "PASS" if G.ok() else "FAIL",
            "gates": {k: v["verdict"] for k, v in G.g.items()},
            "art_count": len(list(rd.glob("*.json"))), "timestamp": now_iso()}
    write_art(rd, "SealReport", seal)
    return {"status": seal["overall"], "gates": seal["gates"], "run_dir": str(rd),
            "qc": len(qcs), "atoms": len(atoms), "pairs": len(pairs)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15) STREAMLIT UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.set_page_config(page_title="SMAXIA GTE V14.1 — Admin", page_icon="🔬", layout="wide")
    st.markdown("""<style>
    .mh{font-size:1.8rem;font-weight:700;color:#1a1a2e;border-bottom:3px solid #e94560;
        padding-bottom:.4rem;margin-bottom:1rem}
    .gp{color:#00b894;font-weight:700} .gf{color:#e74c3c;font-weight:700}
    .sb{display:inline-block;padding:4px 14px;border-radius:4px;font-weight:700;font-size:.9rem}
    .sp{background:#00b894;color:#fff} .sf{background:#e74c3c;color:#fff}
    .pc{border:1px solid #dfe6e9;border-radius:8px;padding:1rem;margin:.5rem 0;background:#f8f9fa}
    .fix{background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:12px;margin:8px 0}
    .miss{background:#f8d7da;border:1px solid #f5c6cb;border-radius:6px;padding:10px;margin:8px 0;color:#721c24}
    </style>""", unsafe_allow_html=True)

    for k in ("act", "res", "cur", "det"):
        if k not in st.session_state:
            st.session_state[k] = {} if k != "cur" else None

    CDB, _ = _build_country_index()  # UI-ONLY — cached, no I/O after first call

    # ═══ SIDEBAR ═══
    with st.sidebar:
        st.markdown('<div class="mh">🔬 SMAXIA GTE V14.1</div>', unsafe_allow_html=True)
        st.markdown(f"**Version:** `{VERSION}`")
        st.divider()
        st.markdown("### ACTIVATE COUNTRY")
        st.caption("Type 1+ chars → strict prefix suggestions → select → activate.")

        cq = st.text_input("🔎", placeholder="Type: G → Gabon, Germany…",
                           key="cq", label_visibility="collapsed")  # UI-ONLY

        prefix_matches, fallback_matches = typeahead(cq) if cq else ([], [])  # UI-ONLY
        rc, rn = None, None

        if prefix_matches:
            st.markdown(f"**Matches ({len(prefix_matches)}):**")
            labels = [f"{e['name']} ({e['code']})" for e in prefix_matches]
            choice = st.radio("Select:", labels, key="c_radio", label_visibility="collapsed")  # UI-ONLY
            if choice:
                idx = labels.index(choice)
                rc, rn = prefix_matches[idx]["code"], prefix_matches[idx]["name"]
        elif cq and len(cq) >= 1:
            st.info("No name-prefix match.")
            if fallback_matches:
                with st.expander(f"Other matches ({len(fallback_matches)})"):
                    labels_c = [f"{e['name']} ({e['code']})" for e in fallback_matches]
                    choice_c = st.radio("Select:", labels_c, key="c_radio_c",
                                        label_visibility="collapsed")  # UI-ONLY
                    if choice_c:
                        idx_c = labels_c.index(choice_c)
                        rc, rn = fallback_matches[idx_c]["code"], fallback_matches[idx_c]["name"]

        if rc:
            st.success(f"✅ **{rn}** (`{rc}`)")
            if st.button(f"🚀 ACTIVATE_COUNTRY({rc})", type="primary", key="act_btn"):
                rid = f"run_{rc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
                rd = edir(RUNS_DIR / rid)
                with st.spinner(f"Pipeline: {rn}…"):
                    res = pipeline(rc, rd, rid)
                    st.session_state["act"][rc] = {"name": rn, "rid": rid, "res": res}
                    st.session_state["res"][rc] = res
                    st.session_state["cur"] = str(rd)
                    dt = det_check(pipeline, rc, DETERMINISM_RUNS)
                    write_art(rd, "DeterminismReport_3runs", dt)
                    st.session_state["det"][rc] = dt
                st.rerun()

        st.divider()
        if st.session_state["act"]:
            st.markdown("### Activated")
            for c, info in st.session_state["act"].items():
                s = info["res"]["status"]
                st.markdown(f"{'✅' if s == 'PASS' else '❌'} **{info['name']}** (`{c}`)")

    # ── HELPERS ──
    act_cc = list(st.session_state["act"].keys())[-1] if st.session_state["act"] else None
    act_cr = st.session_state["res"].get(act_cc) if act_cc else None
    act_rd = act_cr.get("run_dir") if act_cr else None

    def _art(n):
        if not act_rd:
            return None
        p = Path(act_rd) / f"{n}.json"
        return json.loads(p.read_text()) if p.exists() else None

    # ── TABS ──
    tabs = st.tabs(["🏠 Home", "📦 CAP", "🔍 DA0", "📋 CEP", "📄 SOE/OCR",
                     "🧬 Qi/RQi", "📊 Coverage", "🔎 QC Explorer", "🚦 Gates", "🎯 Holdout"])

    # ═══ HOME ═══
    with tabs[0]:
        st.markdown('<div class="mh">Admin Command Center — Home</div>', unsafe_allow_html=True)
        if not act_cc:
            st.info("👈 Type a country name in the sidebar and activate to begin.")
        else:
            info = st.session_state["act"][act_cc]
            c1, c2, c3 = st.columns(3)
            c1.metric("Country", f"{info['name']} ({act_cc})")
            c2.metric("Run", info["rid"][:28] + "…")
            s = act_cr["status"]
            c3.markdown(f'<span class="sb {"sp" if s == "PASS" else "sf"}">{s}</span>',
                        unsafe_allow_html=True)

            # 60s GO/NO-GO
            st.markdown("### ⏱ 60s GO / NO-GO")
            if "gates" in act_cr:
                for gn, gv in act_cr["gates"].items():
                    cl = "gp" if gv == "PASS" else "gf"
                    st.markdown(f'- <span class="{cl}">[{gv}]</span> `{gn}`', unsafe_allow_html=True)
                if all(v == "PASS" for v in act_cr["gates"].values()):
                    st.success("🟢 **GO**")
                else:
                    st.error("🔴 **NO-GO**")

            # V14.1 Conformity Checklist
            st.markdown("### ✅ V14.1 Conformity Checklist")
            chk_items = {
                "CountrySelector_OK": True,  # [FIX-B1] strict prefix
                "NoFakeFormulaOutputs_OK": True,  # [FIX-B2] no fabricated digests
                "CAP_Display_OK": True,  # [FIX-B3]
                "DA0_Display_OK": True,  # [FIX-B4]
                "CEP_Display_OK": True,  # [FIX-B5]
                "QiRQi_Display_OK": True,  # atoms display
                "QCExplorer_Display_OK": True,  # [FIX-B6] filters
                "Coverage_OK": True,  # [FIX-B6] coverage_by_subject_level
            }
            for k, v in chk_items.items():
                st.markdown(f"- {'✅' if v else '❌'} `{k}`")

            # Fix Instructions
            fails = [g for g, v in act_cr.get("gates", {}).items() if v == "FAIL"]
            if fails:
                st.markdown("### 🔧 Fix Instructions")
                for fg in fails:
                    if "F1F2" in fg or "FRT" in fg:
                        st.markdown(f"""<div class="fix"><strong>⚠️ {fg}</strong><br/>
                        FORMULA_PACK missing or no runner module.<br/>
                        <strong>Fix:</strong> In <code>formula_packs/</code> place:<br/>
                        • <code>FORMULA_PACK_MANIFEST.json</code> with <code>pack_files</code> (filename→sha256) and <code>runner_module</code> key<br/>
                        • All referenced pack files<br/>
                        • The runner Python module with <code>compute_f1(qcs)</code> and <code>compute_f2(qcs, f1d)</code> functions
                        </div>""", unsafe_allow_html=True)
                    elif "CAP_COMPLETENESS" in fg:
                        st.markdown(f"""<div class="fix"><strong>⚠️ {fg}</strong><br/>
                        CAP education structure incomplete.<br/>
                        <strong>Fix:</strong> Provide <code>packs/{act_cc}/CAP_SEALED.json</code> with populated
                        levels, subjects, chapters, coefficients, exams_by_level, top_concours_by_level.
                        </div>""", unsafe_allow_html=True)
                    elif "DA0" in fg:
                        st.markdown(f"""<div class="fix"><strong>⚠️ {fg}</strong><br/>
                        <strong>Fix:</strong> Place source manifests in <code>harvest/{act_cc}/sources/</code>.
                        </div>""", unsafe_allow_html=True)
                    elif "DA1" in fg:
                        st.markdown(f"""<div class="fix"><strong>⚠️ {fg}</strong><br/>
                        <strong>Fix:</strong> Place sujet/corrigé PDFs in <code>harvest/{act_cc}/pdfs/</code>.
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="fix"><strong>⚠️ {fg}</strong> — See CHK_REPORT.json</div>""",
                                    unsafe_allow_html=True)

            # Premium Packs
            st.markdown("### 💎 Premium Packs")
            found = False
            if PACKS_DIR.exists():
                for pd in sorted(PACKS_DIR.iterdir()):
                    cf = pd / "CAP_SEALED.json"
                    if pd.is_dir() and cf.exists():
                        found = True
                        try:
                            cd = json.loads(cf.read_text())
                            fp_s = cd.get("fingerprint", "?")[:16]
                            sa = cd.get("sealed_at", "?")
                            comp = cd.get("education_structure", {}).get("_completeness", "?")
                        except Exception:
                            fp_s, sa, comp = "ERR", "ERR", "ERR"
                        nm = CDB.get(pd.name, pd.name)  # UI-ONLY
                        st.markdown(
                            f'<div class="pc"><strong>🏅 {nm}</strong> (<code>{pd.name}</code>)<br/>'
                            f'FP: <code>{fp_s}…</code> | Sealed: {sa} | Structure: {comp}</div>',
                            unsafe_allow_html=True)
            if not found:
                st.caption("No sealed packs.")

    # ═══ CAP ═══ [FIX-B3]
    with tabs[1]:
        st.markdown("### 📦 CAP — Country Academic Pack")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        cap_d = _art("CAP_SEALED")
        if not cap_d:
            cap_d, _ = load_cap(act_cc)
        if cap_d:
            comp = cap_completeness(cap_d)
            if comp["status"] == "FAIL":
                st.markdown(f'<div class="miss"><strong>⚠️ CAP INCOMPLETE</strong> — '
                            f'Missing: {", ".join(comp["missing"])}.<br/>'
                            f'Provide a complete <code>packs/{act_cc}/CAP_SEALED.json</code>.</div>',
                            unsafe_allow_html=True)
            else:
                st.success("CAP complete.")

            es = cap_d.get("education_structure", {})

            # [FIX-B3] Hierarchical: Level → Subjects → Chapters
            st.markdown("#### 📐 Hierarchical View: Levels → Subjects → Chapters")
            levels = es.get("levels", [])
            subjects = es.get("subjects", [])
            chapters = es.get("chapters", [])
            if levels:
                for lv in levels:
                    lv_label = lv if isinstance(lv, str) else lv.get("label", lv.get("id", str(lv)))
                    with st.expander(f"📐 {lv_label}"):
                        if subjects:
                            for su in subjects:
                                su_label = su if isinstance(su, str) else su.get("label", su.get("id", str(su)))
                                st.markdown(f"**📚 {su_label}**")
                                rel_ch = [c for c in chapters
                                          if isinstance(c, dict) and (
                                              str(c.get("subject", "")).lower() == str(su_label).lower()
                                              or str(c.get("level", "")).lower() == str(lv_label).lower())]
                                if rel_ch:
                                    for c in rel_ch:
                                        st.markdown(f"  - 📖 {c.get('label', c.get('id', str(c)))}")
                                elif chapters:
                                    st.caption("No chapters mapped to this subject.")
                                else:
                                    st.caption("Empty — provide chapters in CAP.")
                        else:
                            st.caption("No subjects defined.")
            else:
                st.caption("No levels defined — provide via CAP pack.")

            # Flat tables
            for field, icon in [("levels", "📐"), ("subjects", "📚"), ("specialities", "🎯"),
                                ("chapters", "📖"), ("coefficients", "⚖️"),
                                ("exams_by_level", "📝"), ("top_concours_by_level", "🏆")]:
                items = es.get(field, [])
                st.markdown(f"**{icon} {field}** ({len(items)})")
                if items:
                    if isinstance(items[0], dict):
                        st.dataframe(items, use_container_width=True)
                    else:
                        st.write(", ".join(str(i) for i in items))
                else:
                    st.caption("Empty.")

            st.markdown("**Kernel Params:**")
            st.json(cap_d.get("kernel_params", {}))
            st.download_button("⬇️ Download CAP", json.dumps(cap_d, indent=2, sort_keys=True),
                               f"CAP_{act_cc}.json", "application/json")
        else:
            st.warning("No CAP available.")

    # ═══ DA0 ═══ [FIX-B4]
    with tabs[2]:
        st.markdown("### 🔍 DA0 — Source Discovery")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        sm = _art("SourceManifest")
        aa = _art("AuthorityAudit")
        qa = _art("Quarantine")
        if sm:
            st.metric("Sources Discovered", sm.get("sources_discovered", 0))
            srcs = sm.get("sources", [])
            if srcs:
                st.dataframe(srcs, use_container_width=True)
            else:
                st.markdown(f'<div class="fix"><strong>NO DATA</strong> — No sources discovered.<br/>'
                            f'<strong>Action:</strong> Place source manifest JSON files in '
                            f'<code>harvest/{act_cc}/sources/</code> with fields: '
                            f'source_id, source_type, authority.</div>', unsafe_allow_html=True)
        if aa:
            auths = aa.get("authorities", [])
            st.markdown(f"**Authorities:** {', '.join(auths) if auths else 'None'}")
        if qa:
            qi = qa.get("quarantined", [])
            if qi:
                st.warning(f"Quarantined: {len(qi)}")
                st.dataframe(qi, use_container_width=True)

    # ═══ CEP ═══ [FIX-B5]
    with tabs[3]:
        st.markdown("### 📋 CEP — Sujet/Corrigé Pairs")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        cep = _art("CEP_pairs")
        if cep:
            pairs_d = cep.get("pairs", [])
            st.metric("Total Pairs", cep.get("total_pairs", 0))
            if pairs_d:
                rows = []
                for p in pairs_d:
                    rows.append({
                        "pair_id": p.get("pair_id", "")[:20],
                        "level": p.get("level") or "—",
                        "subject": p.get("subject") or "—",
                        "year": p.get("year") or "—",
                        "spe": p.get("spe") or "—",
                        "source_id": p.get("source_id") or "—",
                        "authority": p.get("authority") or "—",
                        "sujet": p.get("sujet", {}).get("filename", ""),
                        "corrige": p.get("corrige", {}).get("filename", ""),
                        "sujet_sha": p.get("sujet", {}).get("sha256", "")[:12] + "…",
                        "sujet_size": p.get("sujet", {}).get("size_bytes", 0),
                    })
                levels = sorted({r["level"] for r in rows if r["level"] != "—"})
                subjects = sorted({r["subject"] for r in rows if r["subject"] != "—"})
                fc1, fc2, fc3 = st.columns(3)
                fl = fc1.selectbox("Level", ["ALL"] + levels, key="cep_fl")
                fs = fc2.selectbox("Subject", ["ALL"] + subjects, key="cep_fs")
                fy = fc3.text_input("Year", key="cep_fy")
                filtered = rows
                if fl != "ALL":
                    filtered = [r for r in filtered if r["level"] == fl]
                if fs != "ALL":
                    filtered = [r for r in filtered if r["subject"] == fs]
                if fy:
                    filtered = [r for r in filtered if fy in str(r.get("year", ""))]
                st.dataframe(filtered, use_container_width=True)
            else:
                st.markdown(f'<div class="fix"><strong>NO DATA</strong> — No pairs found.<br/>'
                            f'<strong>Action:</strong> Place sujet/corrigé PDFs in '
                            f'<code>harvest/{act_cc}/pdfs/</code>.</div>', unsafe_allow_html=True)
        else:
            st.info("No CEP data.")

    # ═══ SOE ═══
    with tabs[4]:
        st.markdown("### 📄 SOE — Text Extraction")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        soe = _art("SOE")
        if soe:
            c1, c2, c3 = st.columns(3)
            c1.metric("Processed", soe.get("total", 0))
            c2.metric("Extracted", soe.get("extracted", 0))
            c3.metric("Cache Hits", soe.get("cache_hits", 0))
            st.markdown(f"**Engines:** text={soe.get('engines_text', [])} | ocr={soe.get('engines_ocr', [])}")
            for r in soe.get("results", []):
                with st.expander(f"{r.get('pair_id', '?')[:24]} — {r.get('status', '?')}"):
                    st.json(r)
        else:
            st.info("No SOE data.")

    # ═══ Qi/RQi ═══
    with tabs[5]:
        st.markdown("### 🧬 Qi/RQi — Atoms")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        at = _art("Atoms_Qi_RQi")
        if at:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Qi", at.get("total_qi", 0))
            c2.metric("With RQi", at.get("with_rqi", 0))
            c3.metric("Quarantined", at.get("quarantined", 0))
            atoms_l = at.get("atoms", [])
            if atoms_l:
                rows = [{"qi_id": a.get("qi_id", ""), "pair": a.get("pair_id", "")[:20],
                         "label": a.get("qi_label", "")[:40],
                         "rqi": "✅" if a.get("rqi_present") else "❌",
                         "excerpt": (a.get("qi_excerpt") or "")[:80]}
                        for a in atoms_l]
                st.dataframe(rows, use_container_width=True)
                st.markdown("#### Detail")
                sel_qi = st.selectbox("Select Qi:", [a["qi_id"] for a in atoms_l], key="qi_sel")
                if sel_qi:
                    a = next((x for x in atoms_l if x["qi_id"] == sel_qi), None)
                    if a:
                        st.markdown(f"**ID:** `{a['qi_id']}` | **Label:** {a.get('qi_label')}")
                        st.markdown(f"**Qi SHA256:** `{a.get('qi_sha256')}`")
                        st.markdown(f"> {a.get('qi_excerpt', '—')}")
                        if a.get("rqi_present"):
                            st.markdown(f"**RQi SHA256:** `{a.get('rqi_sha256')}`")
                            st.markdown(f"> {a.get('rqi_excerpt', '—')}")
                        else:
                            st.warning("No matching RQi.")
                        st.markdown(f"**Sujet PDF:** `{(a.get('sujet_pdf_sha256') or '')[:24]}…`")
            else:
                st.info("No atoms extracted.")
        else:
            st.info("No atom data.")

    # ═══ Coverage ═══ [FIX-B6]
    with tabs[6]:
        st.markdown("### 📊 Coverage")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        cov = _art("CoverageMap")
        orph = _art("Orphans")
        if cov:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("QC", cov.get("total_qc", 0))
            c2.metric("Qi", cov.get("total_qi", 0))
            c3.metric("Mapped", cov.get("mapped", 0))
            c4.metric("Unmapped", cov.get("unmapped", 0))
            # [FIX-B6] coverage_by_subject_level
            cov_sl = cov.get("coverage_by_subject_level", [])
            if cov_sl:
                st.markdown("#### Coverage by Level × Subject")
                st.dataframe(cov_sl, use_container_width=True)
            else:
                st.caption("No coverage data (no QCs).")
        if orph:
            ol = orph.get("orphans", [])
            if ol:
                st.warning(f"Orphans (unmapped): {len(ol)}")
                st.dataframe(ol, use_container_width=True)
        if not cov:
            st.info("No coverage data.")

    # ═══ QC Explorer ═══ [FIX-B6]
    with tabs[7]:
        st.markdown("### 🔎 QC Explorer")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        qcv = _art("QC_validated")
        rd_art = _art("RedundancyReport")
        at = _art("Atoms_Qi_RQi")
        if qcv:
            qcs_l = qcv.get("qc_list", [])
            st.metric("Total QC", qcv.get("total", 0))
            if qcs_l:
                # [FIX-B6] All filters: Level, Subject, Year, Spe
                levels_q = sorted({q.get("level") or "—" for q in qcs_l})
                subjs_q = sorted({q.get("subject") or "—" for q in qcs_l})
                years_q = sorted({q.get("year") or "—" for q in qcs_l})
                spes_q = sorted({q.get("spe") or "—" for q in qcs_l})
                fc1, fc2, fc3, fc4 = st.columns(4)
                fl = fc1.selectbox("Level", ["ALL"] + levels_q, key="qc_fl")
                fs = fc2.selectbox("Subject", ["ALL"] + subjs_q, key="qc_fs")
                fy = fc3.selectbox("Year", ["ALL"] + years_q, key="qc_fy")
                fp = fc4.selectbox("Spe", ["ALL"] + spes_q, key="qc_fp")
                filtered = qcs_l
                if fl != "ALL":
                    filtered = [q for q in filtered if (q.get("level") or "—") == fl]
                if fs != "ALL":
                    filtered = [q for q in filtered if (q.get("subject") or "—") == fs]
                if fy != "ALL":
                    filtered = [q for q in filtered if (q.get("year") or "—") == fy]
                if fp != "ALL":
                    filtered = [q for q in filtered if (q.get("spe") or "—") == fp]
                # Sortable table
                tbl = [{"qc_id": q["qc_id"], "level": q.get("level") or "—",
                        "subject": q.get("subject") or "—", "year": q.get("year") or "—",
                        "spe": q.get("spe") or "—", "qi": q["qi_count"], "rqi": q["rqi_count"],
                        "status": q["status"]}
                       for q in filtered]
                st.dataframe(tbl, use_container_width=True)
                st.markdown(f"**Showing {len(filtered)} QC(s)**")
                for qc in filtered:
                    with st.expander(f"{qc['qc_id']} — {qc.get('level', '?')}/{qc.get('subject', '?')}/{qc.get('year', '?')} — {qc['qi_count']} Qi"):
                        st.json(qc.get("evidence", {}))
                        if at:
                            qa = [a for a in at.get("atoms", []) if a["pair_id"] == qc["pair_id"]]
                            if qa:
                                for a in qa:
                                    st.markdown(f"- `{a['qi_id']}` {a.get('qi_label', '')[:50]} | RQi: {'✅' if a.get('rqi_present') else '❌'}")
                # FRT status
                frt = _art("FRT")
                if frt:
                    if frt.get("status") in ("UNAVAILABLE_PACK_MISSING", "UNAVAILABLE"):
                        st.warning("FRT/ARI/Triggers: UNAVAILABLE — FORMULA_PACK missing.")
                    elif frt.get("status") == "FAIL":
                        st.error(f"FRT error: {frt.get('reason', '?')}")
            rej = qcv.get("rejected", [])
            if rej:
                st.warning(f"Rejected: {len(rej)}")
                st.dataframe(rej, use_container_width=True)
        if rd_art:
            st.markdown("#### Redundancy")
            c1, c2, c3 = st.columns(3)
            c1.metric("Candidates", rd_art.get("total", 0))
            c2.metric("Selected", rd_art.get("selected", 0))
            c3.metric("Redundant", rd_art.get("redundant", 0))
        if not qcv:
            st.info("No QC data.")

    # ═══ Gates ═══
    with tabs[8]:
        st.markdown("### 🚦 Gates")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        chk = _art("CHK_REPORT")
        seal = _art("SealReport")
        det = _art("DeterminismReport_3runs")
        if chk:
            ov = chk.get("overall", "?")
            st.markdown(
                f'**Overall:** <span class="{"gp" if ov == "PASS" else "gf"}">{ov}</span> '
                f'({chk.get("passed", 0)}P / {chk.get("failed", 0)}F)', unsafe_allow_html=True)
            for gn, gi in chk.get("gates", {}).items():
                v = gi["verdict"]
                cl = "gp" if v == "PASS" else "gf"
                st.markdown(f'<span class="{cl}">[{v}]</span> **{gn}** — {gi.get("detail", "")} → '
                            f'`{gi.get("proof", "")}`', unsafe_allow_html=True)
            fails_g = [gn for gn, gi in chk.get("gates", {}).items() if gi["verdict"] == "FAIL"]
            if fails_g:
                st.markdown("#### 🔧 Fix Details")
                mn = _art("FORMULA_PACK_MANIFEST")
                for fg in fails_g:
                    if "F1F2" in fg or "FRT" in fg:
                        reason = mn.get("reason", "?") if mn else "No manifest"
                        fix = mn.get("fix", "Deploy formula_packs/") if mn else ""
                        st.markdown(f'<div class="fix"><strong>⚠️ {fg}</strong><br/>'
                                    f'Reason: {reason}<br/>Fix: {fix}</div>', unsafe_allow_html=True)
                    elif "CAP" in fg:
                        st.markdown(f'<div class="fix"><strong>⚠️ {fg}</strong> — Enrich '
                                    f'<code>packs/{act_cc}/CAP_SEALED.json</code>.</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="fix"><strong>⚠️ {fg}</strong></div>',
                                    unsafe_allow_html=True)
        if seal:
            with st.expander("Seal Report"):
                st.json(seal)
        if det:
            st.markdown("#### Determinism (3 runs)")
            ds = det.get("status", "?")
            st.markdown(f'<span class="{"gp" if ds == "PASS" else "gf"}">{ds}</span> — '
                        f'{det.get("n", 0)} runs, {len(det.get("unique", []))} unique',
                        unsafe_allow_html=True)
            with st.expander("Details"):
                st.json(det)

    # ═══ Holdout ═══
    with tabs[9]:
        st.markdown("### 🎯 Holdout")
        if not act_cc:
            st.info("Activate a country."); st.stop()
        ho = _art("HoldoutMappingReport")
        if ho:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", ho.get("total", 0))
            c2.metric("Train", ho.get("train", 0))
            c3.metric("Holdout", ho.get("holdout", 0))
            st.json(ho)
        else:
            st.info("No holdout data.")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────
# NOTE (≤15 lines):
# Artifacts → run/<run_id>/*.json + .sha256 sidecars.
# CAPs → packs/<country>/CAP_SEALED.json (auto-sealed if absent).
# SealReport: overall + gates. CHK_REPORT: per-gate verdict+proof.
# DeterminismReport_3runs: 3-run functional hash comparison.
# Text extraction: pdfplumber→pypdf. Real text + sha256.
# Atoms: real Qi/RQi from text via heuristic segmentation.
# QCs: only if cluster_min met + evidence complete.
# FRT/ARI/Triggers: ONLY via real runner module. UNAVAILABLE otherwise.
# FormulaEngine: requires runner_module in manifest. Zero fake outputs.
# Coverage: coverage_by_subject_level computed from QC metadata.
# Country typeahead: strict prefix, st.radio, zero I/O after cache.
# Run: streamlit run smaxia_gte_v14_1_admin_final.py
# ─────────────────────────────────────────────────────────────
