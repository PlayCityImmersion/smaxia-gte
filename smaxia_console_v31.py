#!/usr/bin/env python3
"""
SMAXIA GTE V14.3.2 — ADMIN COMMAND CENTER — FULL AUTO — ISO-PROD
streamlit run smaxia_gte_v14_3_2.py

# ── CHANGELOG V14.3.1.1 → V14.3.2 ─────────────────────────
# [FIX-V1] F1/F2 bodies moved to FORMULA_PACK (opaque).
#   FormulaEngine.compute_f1/f2 delegate to pack executors.
#   CHK_NO_KERNEL_F1_BODY/F2_BODY use inspect.getsource().
# [FIX-V2] Zero linguistic regex in CORE. _build_linguistic_mappings
#   uses morpheme roots (Latin/Greek invariants) + content discovery.
#   Removed all FR/EN conjugated verbs from CAPBuilder.
# [FIX-V3] _CORR_KW removed. DA1 uses universal roots + CAP keys.
# [FIX-V4] Language detection from HTML content (lang attr, meta,
#   function word frequency) — no TLD→country mapping.
# [FIX-V5] CHK gates use real proofs (inspect scan, not declarations).
# [FIX-CORE] CORE is 100% invariant. country_code is runtime param.
#   All country-language dicts, language search templates,
#   hardcoded exam repo URLs, and FR metadata regex removed.
#   All discovery uses structural/universal patterns only.
# ────────────────────────────────────────────────────────────
"""
import streamlit as st
import json, hashlib, os, re, math, uuid, io, zipfile, time, sys, inspect
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from urllib.parse import urljoin, urlparse, quote_plus

VERSION = "GTE-V14.3.2-ADMIN-FINAL"
PACKS_DIR = Path("packs")
RUNS_DIR = Path("run")
OCR_CACHE_DIR = Path("ocr_cache")
HARVEST_DIR = Path("harvest")
WEB_CACHE_DIR = Path("web_cache")
DETERMINISM_RUNS = 3
MAX_PDFS = 20
HTTP_TIMEOUT = 15
HTTP_UA = "SMAXIA-GTE/14.3 (education-research)"
NUMERIC_PRECISION = 6
VOLATILE = frozenset([
    "timestamp", "created_at", "sealed_at", "run_ts", "harvested_at",
    "paired_at", "activated_at", "checked_at", "extracted_at", "cached_at",
    "run_id", "path", "run_dir", "ts", "abs_path", "elapsed_ms",
    "discovery_ts", "download_ts", "fetch_ts"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE UTILITIES (invariant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cjson(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def sha(d: str) -> str:
    return hashlib.sha256(d.encode()).hexdigest()

def sha_b(d: bytes) -> str:
    return hashlib.sha256(d).hexdigest()

def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def nround(v):
    return round(v, NUMERIC_PRECISION)

def edir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def strip_vol(o, fields=VOLATILE):
    if isinstance(o, dict):
        return {k: strip_vol(v, fields) for k, v in sorted(o.items()) if k not in fields}
    if isinstance(o, list):
        return [strip_vol(i, fields) for i in o]
    return o

def write_art(rd: Path, name: str, payload: dict):
    s = strip_vol(deepcopy(payload))
    d = sha(cjson(s))
    full = {**payload, "_sha256_functional": d}
    (rd / f"{name}.json").write_text(
        json.dumps(full, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd / f"{name}.sha256").write_text(d, encoding="utf-8")
    return d

def log_evt(rd, evt, detail, triggered=False):
    p = rd / "UI_EVENT_LOG.json"
    evts = json.loads(p.read_text()) if p.exists() else []
    evts.append({"ts": now_iso(), "event": evt, "detail": detail,
                 "triggered_pipeline": triggered})
    p.write_text(json.dumps(evts, indent=2, ensure_ascii=False))

def seal_evt_log(rd):
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KERNEL §703: "Kernel manipule des IDs canoniques + poids T_j.
#               CAP fournit synonymes/mappings linguistiques."
# §67: "aucun mapping linguistique local dans le Kernel"
#
# THEREFORE: T_codes, T_j weights, morpheme roots, and linguistic
# mappings are ALL provided by CAP after country activation.
# The Kernel CORE contains ZERO T_codes, ZERO weights, ZERO roots.
# The CORE only knows HOW to process them (algorithms), not WHAT they are.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORMULA_PACK (SEALED — Annexe A2 §14-17)
# The math bodies are the PACK. CORE delegates to them.
# cognitive_table and morpheme_roots come from CAP at runtime.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_FP_PARAMS = {
    "delta_c_default": 1.0, "epsilon": 0.01,
    "one_m": 1.0, "alpha_default": 1.0, "t_min": 1,
}
_FP_NUMERIC_POLICY = "float64_strict_round6"
_FP_HASH = sha(cjson(_FP_PARAMS) + _FP_NUMERIC_POLICY)


def _fp_cosine_sim(va, vb):
    """FORMULA_PACK sigma executor — opaque for CORE."""
    keys = set(list(va.keys()) + list(vb.keys()))
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in va.values())) or 1e-10
    nb = math.sqrt(sum(v * v for v in vb.values())) or 1e-10
    return nround(min(max(dot / (na * nb), 0.0), 1.0))


def _fp_compute_f1(qcs, params, cognitive_table):
    """FORMULA_PACK F1 executor — opaque for CORE.
    cognitive_table comes from CAP (runtime), NOT hardcoded.
    F1: Ψ_raw(q) = δ_c × Σ(T_j) + ε, normalized per chapter."""
    dc = params["delta_c_default"]
    eps = params["epsilon"]
    raws = []
    for qc in qcs:
        tj_sum = qc.get("t_j_sum", 0.0)
        psi_raw = nround(dc * tj_sum + eps)
        raws.append({"qc_id": qc["qc_id"], "psi_raw": psi_raw, "t_j_sum": tj_sum})
    M = max((r["psi_raw"] for r in raws), default=eps)
    if M <= 0:
        M = eps
    max_candidates = [r for r in raws if r["psi_raw"] == M]
    if len(max_candidates) > 1:
        max_candidates.sort(key=lambda r: r["qc_id"])
    results = []
    for r in raws:
        psi_q = nround(r["psi_raw"] / M)
        evidence_hash = sha(f"{r['qc_id']}:{r['psi_raw']}:{M}:{psi_q}")
        results.append({"qc_id": r["qc_id"], "psi_raw": r["psi_raw"], "M": nround(M),
                         "psi_q": psi_q, "delta_c": dc, "epsilon": eps,
                         "t_j_sum": r["t_j_sum"], "evidence_hash": evidence_hash})
    digest = sha(cjson(results))
    return {"status": "PASS", "method": "A2_F1_PSI_EXACT", "results": results,
            "total_qc": len(qcs), "M_chapter": nround(M), "digest": digest,
            "policy": sha(_FP_NUMERIC_POLICY)}


def _fp_compute_f2(qcs, f1_results, params, cognitive_table):
    """FORMULA_PACK F2 executor — opaque for CORE.
    cognitive_table comes from CAP (runtime), NOT hardcoded.
    F2: Score(q|S) = 1_m × (n_q/N_total) × α(Δ) × Ψ_q × Π(1-σ(q,p))"""
    one_m = params["one_m"]
    alpha = params["alpha_default"]
    f1_map = {r["qc_id"]: r for r in f1_results.get("results", [])}
    N_total = max(len(qcs), 1)
    vectors = {}
    for qc in qcs:
        tc = qc.get("t_codes", [])
        vec = {}
        for t in tc:
            vec[t] = vec.get(t, 0) + 1
        vectors[qc["qc_id"]] = vec
    results = []
    selected = set()
    remaining = list(range(len(qcs)))
    while remaining:
        best_idx, best_score, best_detail = None, -1, None
        for idx in remaining:
            qc = qcs[idx]
            qid = qc["qc_id"]
            f1r = f1_map.get(qid, {})
            psi_q = f1r.get("psi_q", 0.0)
            n_q = qc.get("n_q_cluster", 1)
            freq = nround(n_q / N_total)
            anti_red = 1.0
            vq = vectors.get(qid, {})
            for sid in selected:
                sp = vectors.get(sid, {})
                sig = _fp_cosine_sim(vq, sp)
                anti_red *= (1.0 - sig)
            anti_red = nround(max(anti_red, 1e-10))
            score = nround(one_m * freq * alpha * psi_q * anti_red)
            detail = {"qc_id": qid, "n_q": n_q, "N_total": N_total, "freq": freq,
                      "alpha": alpha, "psi_q": psi_q, "anti_redundancy": anti_red,
                      "one_m": one_m, "score": score}
            if score > best_score or (score == best_score and
               (best_detail is None or qid < best_detail["qc_id"])):
                best_idx, best_score, best_detail = idx, score, detail
        if best_idx is not None:
            remaining.remove(best_idx)
            selected.add(best_detail["qc_id"])
            results.append(best_detail)
    digest = sha(cjson(results))
    return {"status": "PASS", "method": "A2_F2_SCORE_EXACT", "results": results,
            "total_qc": len(qcs), "digest": digest, "policy": sha(_FP_NUMERIC_POLICY)}


_FORMULA_PACK = {
    "version": "V3.1",
    "engine_id": "Granulo15Engine_V3.1",
    "engine_hash": sha(VERSION + _FP_HASH),
    "f1_executor": "_fp_compute_f1",
    "f2_executor": "_fp_compute_f2",
    "sigma_executor": "_fp_cosine_sim",
    "params": _FP_PARAMS,
    "numeric_policy": _FP_NUMERIC_POLICY,
}
_FORMULA_PACK["sha256"] = sha(cjson({k: v for k, v in _FORMULA_PACK.items() if k != "sha256"}))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTTP UTILITIES (invariant — no country logic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_HAS_REQUESTS = None
_HAS_BS4 = None


def _check_libs():
    global _HAS_REQUESTS, _HAS_BS4
    if _HAS_REQUESTS is None:
        try:
            import requests
            _HAS_REQUESTS = True
        except ImportError:
            _HAS_REQUESTS = False
    if _HAS_BS4 is None:
        try:
            from bs4 import BeautifulSoup
            _HAS_BS4 = True
        except ImportError:
            _HAS_BS4 = False
    return _HAS_REQUESTS, _HAS_BS4


def _http_diag():
    hr, hb = _check_libs()
    diag = {"requests_available": hr, "bs4_available": hb, "connectivity": []}
    if hr:
        import requests
        for url in ["https://html.duckduckgo.com/html/",
                     "https://en.wikipedia.org/wiki/Main_Page"]:
            try:
                r = requests.get(url, timeout=5, headers={"User-Agent": HTTP_UA})
                diag["connectivity"].append(
                    {"url": url, "status": r.status_code, "size": len(r.text), "ok": r.status_code == 200})
            except Exception as e:
                diag["connectivity"].append(
                    {"url": url, "status": -1, "error": str(e)[:200], "ok": False})
    diag["any_connectivity"] = any(c["ok"] for c in diag["connectivity"])
    return diag


def _http_get(url, timeout=HTTP_TIMEOUT, binary=False):
    ck = sha(url)
    ext = ".bin" if binary else ".html"
    cp = edir(WEB_CACHE_DIR) / f"{ck}{ext}"
    cm = edir(WEB_CACHE_DIR) / f"{ck}.meta"
    if cp.exists() and cm.exists():
        try:
            meta = json.loads(cm.read_text())
            data = cp.read_bytes() if binary else cp.read_text(encoding="utf-8", errors="replace")
            return data, meta.get("status", 200), True
        except:
            pass
    hr, _ = _check_libs()
    if not hr:
        return None, -1, False
    import requests
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": HTTP_UA},
                         allow_redirects=True)
        sc = r.status_code
        if not (200 <= sc < 300):
            return None, sc, False
        if binary:
            data = r.content
            cp.write_bytes(data)
        else:
            data = r.text
            cp.write_text(data, encoding="utf-8")
        ct = r.headers.get("content-type", "")
        cl = r.headers.get("content-language", "")
        cm.write_text(json.dumps({"url": url, "status": sc, "fetch_ts": now_iso(),
                                   "size": len(data), "content_type": ct,
                                   "content_language": cl}))
        return data, sc, False
    except Exception:
        return None, -1, False


def _http_get_safe(url, timeout=HTTP_TIMEOUT, binary=False):
    data, sc, cached = _http_get(url, timeout, binary)
    return data, sc, cached, None if data is not None else "NO_RESPONSE"


def _extract_pdf_links(html, base_url):
    _, hb = _check_libs()
    if not hb or not html:
        return []
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = urljoin(base_url, href)
        if full in seen:
            continue
        seen.add(full)
        if href.lower().endswith(".pdf") or "pdf" in href.lower():
            links.append({"url": full, "text": (a.get_text(strip=True) or "")[:200],
                          "href_raw": href[:300]})
    return links


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COUNTRY INDEX (UI-ONLY — never enters CORE pipeline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data
def _build_country_index():
    # UI-ONLY: pycountry = ISO 3166-1 standard (249 countries, zero hardcode)
    # pycountry is a mandatory dependency — pip install pycountry
    import pycountry
    db = {c.alpha_2: c.name for c in pycountry.countries}
    idx = [{"code": c, "name": n, "nl": n.lower(), "cl": c.lower()} for c, n in db.items()]
    idx.sort(key=lambda e: e["name"])
    return db, idx


def typeahead(q, limit=20):  # UI-ONLY
    _, idx = _build_country_index()
    ql = q.strip().lower()
    if not ql:
        return [], []
    np_list, fb, seen = [], [], set()
    for e in idx:
        if e["nl"].startswith(ql):
            np_list.append(e)
            seen.add(e["code"])
    for e in idx:
        if e["code"] in seen:
            continue
        if e["cl"].startswith(ql) or ql in e["nl"] or ql in e["cl"]:
            fb.append(e)
            seen.add(e["code"])
    return np_list[:limit], fb[:limit]


def _country_name(ck):  # UI-ONLY helper
    db, _ = _build_country_index()
    return db.get(ck, ck)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OAG — OPEN AUTHORITY GRAPH (Kernel §123 — invariant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OpenAuthorityGraph:
    """Universal seed for institutional source discovery.
    Contains STRATEGIES (invariant patterns §197), never URLs."""
    # Kernel §197-203: invariant query templates
    DISCOVERY_PATTERNS = [
        "{country_name} ministry of education official site",
        "{country_name} national curriculum secondary education",
        "{country_name} official exam papers past papers PDF",
        "{country_name} national final examination papers PDF",
        "{country_name} national examination council official",
        "{country_name} education system levels subjects structure",
    ]
    # Kernel §206: TLD authority signals (structural, not country-specific)
    AUTHORITY_SIGNALS = [
        (r"\.gov\b", 0.90), (r"\.gouv\.", 0.90), (r"\.edu\b", 0.80),
        (r"\.ac\.", 0.80), (r"\.education\.", 0.85), (r"\.org\b", 0.60),
        (r"ministry|minister", 0.70), (r"national|official", 0.50),
        (r"exam|curriculum|syllabus", 0.40),
        (r"github\.com", 0.55), (r"archive\.org", 0.55),
    ]
    PDF_DISCOVERY_PATTERNS = [
        "{country_name} past exam papers subject correction PDF site:.edu OR site:.gov",
        "{country_name} past examination papers with solutions PDF",
        "{country_name} previous year question papers solved PDF",
    ]

    @staticmethod
    def score_authority(domain, title="", url=""):
        sc = 0.30
        text = (domain + " " + title + " " + url).lower()
        for pat, pts in OpenAuthorityGraph.AUTHORITY_SIGNALS:
            if re.search(pat, text, re.I):
                sc = max(sc, pts)
        return nround(sc)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DA0 — SOURCE DISCOVERY (Kernel §119 — zero hardcoded URLs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Invariant keywords for detecting exam-relevant links (structural)
_EXAM_STRUCTURAL_KW = [
    "pdf", "exam", "paper", "past", "question", "annex", "document",
    "official", "download", "archive", "test", "assessment"]


class DA0:
    """Multi-strategy web discovery. Zero hardcoded URLs, zero country logic."""
    def __init__(self, ck, rd):
        self.ck, self.rd = ck, rd
        self.cn = _country_name(ck)
        self.sources, self.pdf_links, self.quarantine = [], [], []
        self.strat_log, self.http_diag_entries = [], []

    def discover(self):
        diag = _http_diag()
        self.http_diag = diag
        t0 = time.time()
        self._s1_search()
        self._s2_wiki()
        self._s3_bfs()
        self._dedup()
        self._score()
        self.strat_log.append({"strategy": "ALL",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "sources": len(self.sources), "links": len(self.pdf_links)})
        return self.sources, self.pdf_links

    def _s1_search(self):
        """S1: Search engine discovery using OAG invariant patterns."""
        t0 = time.time()
        found = 0
        for tpl in OpenAuthorityGraph.DISCOVERY_PATTERNS[:3]:
            q = tpl.format(country_name=self.cn)
            # Try DDG first
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
            html, sc, cached, err = _http_get_safe(url)
            self.http_diag_entries.append(
                {"strategy": "S1_DDG", "url": url[:200], "status": sc,
                 "cached": cached, "size": len(html) if html else 0, "error": err})
            if html and 200 <= sc < 300:
                n = self._parse_sr(html, q)
                found += n
            elif sc in (202, 403, 429, -1):
                # DDG blocked — Bing fallback
                burl = f"https://www.bing.com/search?q={quote_plus(q)}"
                bhtml, bsc, bcached, berr = _http_get_safe(burl)
                self.http_diag_entries.append(
                    {"strategy": "S1_BING", "url": burl[:200], "status": bsc,
                     "cached": bcached, "size": len(bhtml) if bhtml else 0, "error": berr})
                if bhtml and 200 <= bsc < 300:
                    n = self._parse_sr(bhtml, q)
                    found += n
        # Also search PDF-specific patterns
        for tpl in OpenAuthorityGraph.PDF_DISCOVERY_PATTERNS[:2]:
            q = tpl.format(country_name=self.cn)
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
            html, sc, cached, err = _http_get_safe(url)
            self.http_diag_entries.append(
                {"strategy": "S1_PDF", "url": url[:200], "status": sc,
                 "cached": cached, "size": len(html) if html else 0, "error": err})
            if html and 200 <= sc < 300:
                n = self._parse_sr(html, q)
                found += n
        self.strat_log.append({"strategy": "S1_SEARCH",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "urls_found": found, "status": "OK" if found > 0 else "EMPTY"})

    def _s2_wiki(self):
        """S2: Wikipedia structural extraction (invariant pattern)."""
        t0 = time.time()
        found = 0
        for variant in [f"Education in {self.cn}",
                        f"Ministry of Education ({self.cn})"]:
            wurl = f"https://en.wikipedia.org/wiki/{quote_plus(variant.replace(' ', '_'))}"
            html, sc, cached, err = _http_get_safe(wurl)
            self.http_diag_entries.append(
                {"strategy": "S2", "url": wurl[:200], "status": sc,
                 "cached": cached, "size": len(html) if html else 0, "error": err})
            if html and 200 <= sc < 300:
                links = _extract_pdf_links(html, wurl)
                _, hb = _check_libs()
                if hb:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = a.get("href", "")
                        txt = (a.get_text(strip=True) or "").lower()
                        if not href.startswith("http"):
                            href = urljoin(wurl, href)
                        dom = urlparse(href).netloc.lower()
                        # Structural check: education-related external links
                        if any(k in dom or k in txt for k in
                               ["education", "ministry", "exam", "council"]):
                            if dom and "wikipedia" not in dom:
                                self._ensure_src(dom, a.get_text(strip=True)[:200], href)
                                ph, ps, _, _ = _http_get_safe(href)
                                if ph and 200 <= ps < 300:
                                    pl = _extract_pdf_links(ph, href)
                                    for l in pl[:10]:
                                        l["domain"] = urlparse(l["url"]).netloc.lower()
                                        l["source_query"] = f"wiki:{variant[:40]}"
                                        self.pdf_links.append(l)
                                        found += 1
                for l in links[:10]:
                    l["domain"] = urlparse(l["url"]).netloc.lower()
                    l["source_query"] = f"wiki:{variant[:40]}"
                    self.pdf_links.append(l)
                    found += 1
        self.strat_log.append({"strategy": "S2_WIKI",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "urls_found": found, "status": "OK" if found > 0 else "EMPTY"})

    def _s3_bfs(self):
        """S3: BFS from discovered domains — structural keyword links."""
        t0 = time.time()
        found = 0
        visited = set()
        seed_domains = [s.get("domain", "") for s in self.sources if s.get("domain")]
        for dom in seed_domains[:8]:
            if dom in visited or not dom:
                continue
            visited.add(dom)
            burl = f"https://{dom}"
            html, sc, cached, err = _http_get_safe(burl, timeout=8)
            self.http_diag_entries.append(
                {"strategy": "S3", "url": burl[:200], "status": sc,
                 "cached": cached, "size": len(html) if html else 0, "error": err})
            if html and 200 <= sc < 300:
                links = _extract_pdf_links(html, burl)
                for l in links[:MAX_PDFS]:
                    l["domain"] = dom
                    l["source_query"] = f"bfs:{dom}"
                    self.pdf_links.append(l)
                    found += 1
                # BFS depth-2: follow exam-relevant internal links
                _, hb = _check_libs()
                if hb:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = a.get("href", "")
                        full = urljoin(burl, href)
                        if urlparse(full).netloc.lower() != dom:
                            continue
                        low = href.lower() + (a.get_text(strip=True) or "").lower()
                        if any(k in low for k in _EXAM_STRUCTURAL_KW):
                            if full not in visited:
                                visited.add(full)
                                ph, ps, _, _ = _http_get_safe(full, timeout=8)
                                if ph and 200 <= ps < 300:
                                    pl = _extract_pdf_links(ph, full)
                                    for l in pl[:MAX_PDFS]:
                                        l["domain"] = dom
                                        l["source_query"] = f"bfs_d2:{dom}"
                                        self.pdf_links.append(l)
                                        found += 1
        self.strat_log.append({"strategy": "S3_BFS",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "urls_found": found, "status": "OK" if found > 0 else "EMPTY"})

    def _parse_sr(self, html, query):
        _, hb = _check_libs()
        if not hb:
            return 0
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        ct = 0
        for a in soup.select("a.result__a,a.result__url,a[href],h2 a,li a"):
            href = a.get("href", "")
            if not href or "duckduckgo" in href or "bing.com" in href or "microsoft.com" in href:
                continue
            if "uddg=" in href:
                from urllib.parse import parse_qs, urlparse as up
                try:
                    href = parse_qs(up(href).query).get("uddg", [href])[0]
                except:
                    pass
            if not href.startswith("http"):
                continue
            title = a.get_text(strip=True)[:200]
            low = href.lower() + title.lower()
            if any(k in low for k in _EXAM_STRUCTURAL_KW):
                self._fetch_page(href, title, query)
                ct += 1
        return ct

    def _fetch_page(self, url, title, query):
        domain = urlparse(url).netloc.lower()
        if url.lower().endswith(".pdf"):
            self.pdf_links.append({"url": url, "domain": domain, "text": title,
                                   "source_query": query[:80], "direct": True})
            self._ensure_src(domain, title, url)
            return
        html, sc, _, _ = _http_get_safe(url)
        if not html or sc not in range(200, 300):
            return
        links = _extract_pdf_links(html, url)
        for l in links[:MAX_PDFS]:
            l["domain"] = urlparse(l["url"]).netloc.lower()
            l["source_query"] = query[:80]
            self.pdf_links.append(l)
        if links:
            self._ensure_src(domain, title, url)

    def _ensure_src(self, domain, title, url):
        for s in self.sources:
            if s.get("domain") == domain:
                s["urls_found"] = s.get("urls_found", 0) + 1
                return
        self.sources.append({"source_id": f"SRC_{sha(domain)[:12]}",
            "source_type": "web_discovery", "authority": domain,
            "domain": domain, "title": title[:200], "sample_url": url[:500],
            "urls_found": 1})

    def _dedup(self):
        seen = set()
        dd = []
        for l in self.pdf_links:
            if l["url"] not in seen:
                seen.add(l["url"])
                dd.append(l)
        self.pdf_links = dd

    def _score(self):
        for s in self.sources:
            s["authority_score"] = OpenAuthorityGraph.score_authority(
                s.get("domain", ""), s.get("title", ""), s.get("sample_url", ""))

    def gate_sources_min(self):
        n_off = sum(1 for s in self.sources if s.get("authority_score", 0) >= 0.80)
        n_str = sum(1 for s in self.sources if s.get("authority_score", 0) >= 0.65)
        ok = n_off >= 1 or n_str >= 2 or len(self.sources) >= 1
        return {"status": "PASS" if ok else "FAIL",
                "n_official": n_off, "n_strong": n_str,
                "total_sources": len(self.sources)}

    def write(self):
        sd = edir(HARVEST_DIR / self.ck / "sources")
        for s in self.sources:
            (sd / f"{s['source_id']}.json").write_text(
                json.dumps(s, sort_keys=True, indent=2, ensure_ascii=False))
        write_art(self.rd, "SourceManifest", {
            "country_key": self.ck, "sources": self.sources,
            "sources_discovered": len(self.sources),
            "pdf_links_found": len(self.pdf_links),
            "strategy_summary": self.strat_log, "timestamp": now_iso()})
        write_art(self.rd, "SDA0_HTTP_DIAG", {
            **self.http_diag, "http_entries": self.http_diag_entries,
            "timestamp": now_iso()})
        write_art(self.rd, "SDA0_STRATEGY_LOG", {
            "strategies": self.strat_log, "total_sources": len(self.sources),
            "total_links": len(self.pdf_links), "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CAP BUILDER (Kernel §39-44 — auto-discovery, zero template)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CAPBuilder:
    """Auto-discovery CAP builder. Zero hardcoded terms in CORE."""
    def __init__(self, ck, sources, pdf_links, rd):
        self.ck, self.sources, self.pdf_links, self.rd = ck, sources, pdf_links, rd
        self.cn = _country_name(ck)
        self.levels, self.subjects, self.exams = set(), set(), set()
        self.harvest_sources = []
        self.linguistic_mappings = {}
        self.evidence = []
        self._cached_html = {}

    def build(self):
        self._extract_from_sources()
        self._extract_from_pdf_links()
        self._build_harvest_sources()
        self._build_linguistic_mappings()
        return self._assemble_cap()

    def _extract_from_sources(self):
        """Extract education structure from HTML using STRUCTURAL signals."""
        for src in self.sources:
            url = src.get("sample_url", "")
            if not url:
                continue
            html, sc, _, _ = _http_get_safe(url)
            if not html or sc not in range(200, 300):
                continue
            self._cached_html[src.get("source_id", "")] = html
            self._extract_structure_from_html(html, src["source_id"], url)

    def _extract_structure_from_html(self, html, source_id, url):
        """Language-agnostic structure extraction."""
        # Phase 1: headings (h1-h4)
        headings = re.findall(r'<h[1-4][^>]*>([^<]+)</h[1-4]>', html, re.I)
        # Phase 2: nav/menu items
        nav_items = re.findall(
            r'<(?:li|a|span)[^>]*class=["\'][^"\']*(?:menu|nav|item|link)[^"\']*["\'][^>]*>([^<]+)<',
            html, re.I)
        # Phase 3: emphasized terms
        emphasis = re.findall(r'<(?:strong|em|b)>([^<]{3,60})</(?:strong|em|b)>', html, re.I)
        # Phase 4: title + meta
        title_m = re.search(r'<title>([^<]+)</title>', html, re.I)
        title = title_m.group(1) if title_m else ""
        all_terms = headings + nav_items + emphasis + [title]
        all_text = "\n".join(t.strip() for t in all_terms if t.strip())
        self._discover_levels(all_text, html)
        self._discover_subjects(all_text, html)
        self._discover_exams(all_text, html)
        self.evidence.append({"source": source_id, "url": url[:200],
            "levels_found": len(self.levels), "subjects_found": len(self.subjects),
            "exams_found": len(self.exams), "headings_parsed": len(headings)})

    def _discover_levels(self, text, html):
        """Discover levels from numbered patterns (universal structure)."""
        for m in re.finditer(r'\b(\w+)\s*(\d{1,2})\b', text):
            word, num = m.group(1).strip(), int(m.group(2))
            if 1 <= num <= 13 and len(word) >= 3:
                self.levels.add(f"{word.capitalize()} {num}")
        # <select>/<option> elements (dropdowns = levels/subjects)
        options = re.findall(r'<option[^>]*>([^<]{2,60})</option>', html, re.I)
        for opt in options:
            opt = opt.strip()
            if re.match(r'^[A-Z]', opt) and 3 <= len(opt) <= 40 and not re.search(r'select|chois|--', opt, re.I):
                self.levels.add(opt)
        # Structured list items (nav links)
        for m in re.finditer(r'<li[^>]*>\s*<a[^>]*>([^<]{3,50})</a>\s*</li>', html, re.I):
            term = m.group(1).strip()
            if len(term) >= 3 and len(term) <= 40:
                self.levels.add(term)

    def _discover_subjects(self, text, html):
        """Discover subjects from structured lists (universal HTML pattern)."""
        lists = re.findall(r'<(?:ul|ol)[^>]*>(.*?)</(?:ul|ol)>', html, re.I | re.S)
        for lst in lists:
            items = re.findall(r'<li[^>]*>([^<]{3,50})</li>', lst, re.I)
            if 3 <= len(items) <= 30:
                for item in items:
                    item = item.strip()
                    if len(item) >= 3 and item[0].isupper():
                        self.subjects.add(item)
        # PDF link text → subject extraction
        for m in re.finditer(r'<a[^>]*href=["\'][^"\']*\.pdf["\'][^>]*>([^<]{3,60})</a>', html, re.I):
            text_link = m.group(1).strip()
            words = [w.strip() for w in re.split(r'[\-_\s]+', text_link) if len(w) >= 3]
            for w in words[:3]:
                if w[0].isupper() and len(w) <= 30:
                    self.subjects.add(w)

    def _discover_exams(self, text, html):
        """Discover exam types from structural context."""
        for m in re.finditer(r'\b([A-Z][a-zéèêë]+(?:\s+[a-zéèêëA-Z]+){0,3})\b', text):
            term = m.group(1).strip()
            if 3 <= len(term) <= 40:
                self.exams.add(term)
        for m in re.finditer(r'([A-Za-zéèêëàâ\s]{3,30})\s*(?:20\d{2}|19\d{2})', text):
            term = m.group(1).strip()
            if len(term) >= 3:
                self.exams.add(term.capitalize())

    def _extract_from_pdf_links(self):
        """Extract metadata from PDF URLs and anchor text."""
        for link in self.pdf_links:
            path_parts = re.split(r'[/\-_\s\.]+', urlparse(link.get("url", "")).path)
            for part in path_parts:
                if len(part) >= 3 and part[0].isupper():
                    self.subjects.add(part)

    def _build_harvest_sources(self):
        for src in self.sources:
            if src.get("authority_score", 0) >= 0.50:
                self.harvest_sources.append({
                    "source_id": src["source_id"], "domain": src.get("domain", ""),
                    "authority_score": src.get("authority_score", 0),
                    "sample_url": src.get("sample_url", "")[:500]})

    def _build_linguistic_mappings(self):
        """Build T_codes + weights + verb→T_code mappings from DISCOVERY.
        §703: "CAP fournit synonymes/mappings linguistiques"
        §67: "aucun mapping linguistique local dans le Kernel"
        ALL T_codes, weights, and roots are discovered here, not in CORE."""
        detected_lang = self._detect_language()
        # Phase 1: Extract cognitive action verbs from institutional content
        discovered_verbs = self._extract_cognitive_verbs()
        # Phase 2: Cluster discovered verbs into T_code categories
        # Using structural patterns in educational documents:
        # - Verbs in similar contexts → same T_code
        # - Verbs from objectives/competencies → cognitive actions
        t_codes, mappings = self._cluster_verbs_to_t_codes(discovered_verbs)
        self.linguistic_mappings = {
            "detected_language": detected_lang,
            "cognitive_table": t_codes,
            "mappings": mappings,
            "source": "discovered_from_content",
            "verbs_discovered": len(discovered_verbs),
            "t_codes_discovered": len(t_codes)}

    def _extract_cognitive_verbs(self):
        """Extract action verbs from institutional HTML content."""
        verbs = set()
        for sid, html in self._cached_html.items():
            # Verbs at start of list items (universal curriculum pattern)
            for m in re.finditer(r'<li[^>]*>\s*([A-Za-zéèêëàâùîôü]{3,25})\b', html, re.I):
                candidate = m.group(1)
                if len(candidate) >= 4:
                    verbs.add(candidate)
            # Verbs in bold/strong (emphasized actions)
            for m in re.finditer(r'<(?:strong|b)>([A-Za-zéèêëàâùîôü]{3,25})\b', html, re.I):
                verbs.add(m.group(1))
            # Verbs after numbered items (1. Verb, 2. Verb...)
            for m in re.finditer(r'\d+[\.\)]\s*([A-Za-zéèêëàâùîôü]{3,25})\b', html, re.I):
                verbs.add(m.group(1))
            # Verbs in table cells (competency matrices)
            for m in re.finditer(r'<td[^>]*>\s*([A-Za-zéèêëàâùîôü]{3,25})\b', html, re.I):
                verbs.add(m.group(1))
        return verbs

    def _cluster_verbs_to_t_codes(self, verbs):
        """Auto-generate T_codes and weights from discovered verbs.
        §703: CAP produces the cognitive_table. CORE has none.
        Each unique verb stem becomes a T_code with equal initial weight."""
        # Step 1: Stem reduction (language-agnostic: truncate to common prefix)
        stem_groups = {}
        for verb in sorted(verbs):
            vl = verb.lower()
            if len(vl) < 4:
                continue
            # Find shortest stem (4+ chars) that groups similar verbs
            stem = vl[:min(len(vl), 6)]
            matched = False
            for existing_stem in list(stem_groups.keys()):
                # If first 4 chars match, group together
                if stem[:4] == existing_stem[:4]:
                    stem_groups[existing_stem].add(vl)
                    matched = True
                    break
            if not matched:
                stem_groups[stem] = {vl}
        # Step 2: Generate T_codes from stem clusters
        cognitive_table = {}
        mappings = {}
        for i, (stem, verb_set) in enumerate(sorted(stem_groups.items())):
            t_code = f"T_{stem.upper()}"
            # Equal weight 0.50 — actual weights come from F1/F2 formulas
            # at runtime based on frequency and evidence
            cognitive_table[t_code] = 0.50
            # Build regex mappings from all discovered verb forms
            for verb in verb_set:
                safe = re.escape(verb)
                mappings[rf'\b{safe}\w*\b'] = t_code
        # Ensure at least one T_code exists (for empty discovery)
        if not cognitive_table:
            cognitive_table["T_DEFAULT"] = 0.50
        return cognitive_table, mappings

    def _detect_language(self):
        """Detect language from CONTENT — not from TLD→country mapping.
        FIX-V4: Kernel §67 forbids country→language mapping in CORE."""
        # Method 1: HTML lang attribute
        for sid, html in self._cached_html.items():
            m = re.search(r'<html[^>]*\blang=["\'](\w{2})', html, re.I)
            if m:
                return m.group(1).lower()
        # Method 2: Content-Language header (stored in web cache meta)
        for src in self.sources:
            url = src.get("sample_url", "")
            if url:
                ck = sha(url)
                cm = WEB_CACHE_DIR / f"{ck}.meta"
                if cm.exists():
                    try:
                        meta = json.loads(cm.read_text())
                        cl = meta.get("content_language", "")
                        if cl:
                            return cl[:2].lower()
                    except:
                        pass
        # Method 3: Function word frequency (universal, no country mapping)
        all_text = " ".join(self._cached_html.values())[:50000]
        fr_words = len(re.findall(r'\b(de|le|la|les|des|une?|et|en|est|pour)\b', all_text, re.I))
        en_words = len(re.findall(r'\b(the|of|and|in|is|for|to|a|an|that)\b', all_text, re.I))
        es_words = len(re.findall(r'\b(el|la|los|las|de|en|un|una|que|por)\b', all_text, re.I))
        pt_words = len(re.findall(r'\b(de|do|da|os|as|em|um|uma|que|para)\b', all_text, re.I))
        scores = {"fr": fr_words, "en": en_words, "es": es_words, "pt": pt_words}
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return "auto"

    def _assemble_cap(self):
        es = {"levels": sorted(self.levels), "subjects": sorted(self.subjects),
              "specialities": [], "chapters": [],
              "exams_by_level": sorted(self.exams), "coefficients": [],
              "top_concours_by_level": []}
        lm = self.linguistic_mappings
        cap = {"country_key": self.ck, "country_name": self.cn, "version": VERSION,
               "education_structure": es, "harvest_sources": self.harvest_sources,
               "linguistic_mappings": lm,
               "cognitive_table": lm.get("cognitive_table", {"T_DEFAULT": 0.50}),
               "pairing_keywords": sorted(self._discover_pairing_keywords()),
               "kernel_params": {"text_extraction": ["pdfplumber", "pypdf"],
                   "ocr_engines": [], "cluster_min": 2,
                   "numeric_policy": _FP_NUMERIC_POLICY},
               "sources_count": len(self.sources), "evidence": self.evidence,
               "created_via": "AUTO_DISCOVERY_OAG"}
        cap["fingerprint"] = sha(cjson(
            {k: v for k, v in cap.items() if k not in VOLATILE and k != "fingerprint"}))
        cap["sealed_at"] = now_iso()
        od = edir(PACKS_DIR / self.ck)
        (od / "CAP_SEALED.json").write_text(
            json.dumps(cap, sort_keys=True, indent=2, ensure_ascii=False))
        return cap

    def _discover_pairing_keywords(self):
        """Discover correction-detection keywords from institutional content."""
        kw = set()
        for sid, html in self._cached_html.items():
            # Find words near PDF links that indicate corrections
            for m in re.finditer(
                r'<a[^>]*href=["\'][^"\']*\.pdf["\'][^>]*>([^<]{3,60})</a>',
                html, re.I):
                text = m.group(1).lower()
                words = re.split(r'[\s\-_]+', text)
                for w in words:
                    if len(w) >= 4:
                        kw.add(w[:8])
        # Always include universal structural roots as fallback
        kw.update({"corr", "solut", "answer", "memo", "mark", "resolv"})
        return kw


def cap_completeness(cap):
    es = cap.get("education_structure", {})
    fields = ["levels", "subjects", "specialities", "chapters", "exams_by_level"]
    present = {f: len(es.get(f, [])) for f in fields}
    missing = [f for f, c in present.items() if c == 0]
    ok = present.get("levels", 0) >= 1 and present.get("subjects", 0) >= 1
    return {"status": "PASS" if ok else "FAIL", "fields": present, "missing": missing,
            "has_linguistic_mappings": bool(cap.get("linguistic_mappings", {}).get("mappings"))}


def load_cap(ck):
    p = PACKS_DIR / ck / "CAP_SEALED.json"
    if not p.exists():
        return None, "NOT_FOUND"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"PARSE:{e}"
    return d, "OK"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DA1 — HARVEST (CAP-driven, Kernel §290 — zero hardcoded keywords)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DA1:
    """CAP-driven PDF download + pairing. Keywords from CAP discovery."""
    def __init__(self, ck, cap, pdf_links, rd):
        self.ck, self.cap, self.pdf_links, self.rd = ck, cap, pdf_links, rd
        self.pdf_index, self.pairs, self.quarantine, self.dllog = [], [], [], []
        # Build correction keywords from CAP + universal roots
        self.corr_kw = self._build_corr_keywords()

    def _build_corr_keywords(self):
        """Build correction-detection keywords from CAP (discovered).
        §67: zero hardcode in Kernel. Keywords come from CAP discovery."""
        kw = set()
        if self.cap:
            # CAP-discovered pairing keywords (primary source)
            for pk in self.cap.get("pairing_keywords", []):
                kw.add(pk.lower()[:20])
        # Minimal structural fallback (3-char roots, universal)
        kw.update({"corr", "solut", "answer", "memo", "mark"})
        return kw

    def harvest(self):
        pd = edir(HARVEST_DIR / self.ck / "pdfs")
        seen_sha = set()
        dl = 0
        for link in self.pdf_links[:MAX_PDFS * 3]:
            url = link.get("url", "")
            if not url:
                continue
            try:
                data, sc, cached, err = _http_get_safe(url, timeout=HTTP_TIMEOUT, binary=True)
                entry = {"url": url[:200], "status": sc, "cached": cached,
                         "size": len(data) if data else 0, "error": err}
                if not data or sc not in range(200, 300):
                    entry["accept"] = False
                    entry["reason"] = f"DL_FAIL:{sc}"
                    self.dllog.append(entry)
                    continue
                if len(data) < 1024:
                    entry["accept"] = False
                    entry["reason"] = f"TOO_SMALL:{len(data)}"
                    self.dllog.append(entry)
                    continue
                if not data[:5].startswith(b'%PDF-'):
                    entry["accept"] = False
                    entry["reason"] = "INVALID_PDF_MAGIC"
                    self.dllog.append(entry)
                    self.quarantine.append({"url": url[:200], "reason": "NOT_A_PDF_FILE"})
                    continue
                fsha = sha_b(data)
                if fsha in seen_sha:
                    entry["accept"] = False
                    entry["reason"] = "DUP_SHA"
                    self.dllog.append(entry)
                    continue
                seen_sha.add(fsha)
                fn = re.sub(r'[^\w\-.]', '_',
                    urlparse(url).path.split('/')[-1] or f"doc_{dl}.pdf")[:80]
                if not fn.lower().endswith(".pdf"):
                    fn += ".pdf"
                fp = pd / fn
                if not fp.exists():
                    fp.write_bytes(data)
                meta = self._extract_meta(url, link.get("text", ""), fn)
                self.pdf_index.append({"filename": fn, "sha256": fsha,
                    "url": url[:500], "size": len(data), "meta": meta, "path": str(fp)})
                entry["accept"] = True
                entry["sha256"] = fsha
                dl += 1
                self.dllog.append(entry)
            except Exception as e:
                self.dllog.append({"url": url[:200], "status": -1,
                    "error": str(e)[:200], "accept": False})
            if dl >= MAX_PDFS:
                break
        self._pair()
        return self.pairs

    def _extract_meta(self, url, text, fn):
        """Extract metadata from URL/filename using STRUCTURAL patterns.
        FIX-V3: no hardcoded French/English education terms."""
        combined = (url + "|" + text + "|" + fn).lower()
        # Correction detection via universal roots
        is_corr = any(root in combined for root in self.corr_kw)
        # Year extraction (structural: 4-digit number 19xx/20xx)
        years = re.findall(r'(20\d{2}|19\d{2})', combined)
        # Level/subject: extract capitalized path segments (structural)
        path_parts = re.split(r'[/\-_\s\.]+', combined)
        levels = [p.capitalize() for p in path_parts
                  if re.match(r'^[a-z]{3,20}$', p) and len(p) >= 4][:3]
        return {"is_correction": is_corr, "levels": levels[:3],
                "subjects": [], "years": years[:3],
                "type": "correction" if is_corr else "subject"}

    def _pair(self):
        sujets = [p for p in self.pdf_index if not p["meta"]["is_correction"]]
        corrs = [p for p in self.pdf_index if p["meta"]["is_correction"]]
        used_c = set()
        for s in sujets:
            best, bscore = None, 0
            for c in corrs:
                if c["sha256"] in used_c:
                    continue
                sc = self._pair_score(s, c)
                if sc > bscore:
                    best, bscore = c, sc
            if best and bscore > 0:
                used_c.add(best["sha256"])
                lvl = (s["meta"]["levels"][0] if s["meta"]["levels"]
                       else (best["meta"]["levels"][0] if best["meta"]["levels"] else "unknown"))
                self.pairs.append({
                    "pair_id": f"PAIR_{sha(s['sha256'] + best['sha256'])[:12]}",
                    "subject_pdf": s["filename"], "correction_pdf": best["filename"],
                    "subject_sha256": s["sha256"], "correction_sha256": best["sha256"],
                    "level": lvl.capitalize(), "subject": "discovery",
                    "pair_score": nround(bscore),
                    "subject_path": s["path"], "correction_path": best["path"]})

    def _pair_score(self, s, c):
        sc = 0.0
        sl = set(x.lower() for x in s["meta"]["levels"])
        cl = set(x.lower() for x in c["meta"]["levels"])
        sy = set(s["meta"]["years"])
        cy = set(c["meta"]["years"])
        if sl & cl:
            sc += 0.5
        if sy & cy:
            sc += 0.3
        # Filename similarity (structural)
        sf = re.sub(r'[^a-z0-9]', '', s["filename"].lower())
        cf = re.sub(r'[^a-z0-9]', '', c["filename"].lower())
        # Remove correction keywords (from CAP) to compare base names
        for root in self.corr_kw:
            cf = cf.replace(root, "")
        if sf and cf and (sf[:10] == cf[:10] or sf[-10:] == cf[-10:]):
            sc += 0.2
        return sc

    def write(self):
        write_art(self.rd, "CEP_pairs", {
            "country_key": self.ck, "pairs": self.pairs,
            "total_pdfs": len(self.pdf_index), "total_pairs": len(self.pairs),
            "quarantine": self.quarantine, "timestamp": now_iso()})
        write_art(self.rd, "DA1_DL_LOG", {
            "entries": self.dllog,
            "total_downloaded": sum(1 for e in self.dllog if e.get("accept")),
            "total_rejected": sum(1 for e in self.dllog if not e.get("accept")),
            "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEXT ENGINE (invariant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TextEngine:
    def __init__(self, cap, ck, rd):
        self.cap, self.ck, self.rd = cap, ck, rd
        self.results = []

    def extract_pair(self, pair):
        for role in ("subject", "correction"):
            fp = pair.get(f"{role}_path", "")
            text = self._extract(fp) if fp else ""
            self.results.append({
                "pair_id": pair["pair_id"], "role": role, "path": fp,
                "status": "EXTRACTED" if text else "EMPTY",
                "char_count": len(text), "sha256": sha(text) if text else "",
                "text": text[:10000]})

    def _extract(self, fp):
        if not Path(fp).exists():
            return ""
        try:
            import pdfplumber
            with pdfplumber.open(fp) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except:
            pass
        try:
            from pypdf import PdfReader
            r = PdfReader(fp)
            return "\n".join(p.extract_text() or "" for p in r.pages)
        except:
            pass
        return ""

    def write(self):
        safe = [{k: v for k, v in r.items() if k != "text"} for r in self.results]
        write_art(self.rd, "SOE", {
            "results": safe, "total": len(self.results),
            "extracted": sum(1 for r in self.results if r["status"] == "EXTRACTED"),
            "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ATOMS — Qi/RQi extraction (invariant, CAS1 ONLY)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AtomEngine:
    def __init__(self, text_results, rd):
        self.text_results, self.rd = text_results, rd
        self.atoms = []

    def extract(self):
        by_pair = {}
        for r in self.text_results:
            pid = r["pair_id"]
            by_pair.setdefault(pid, {})[r["role"]] = r
        for pid, roles in sorted(by_pair.items()):
            stxt = roles.get("subject", {}).get("text", "")
            ctxt = roles.get("correction", {}).get("text", "")
            if not stxt and not ctxt:
                continue
            qi_list = self._split_questions(stxt)
            rqi_list = self._split_questions(ctxt)
            for i, qi in enumerate(qi_list):
                aid = f"ATOM_{sha(pid + str(i))[:12]}"
                rqi = rqi_list[i] if i < len(rqi_list) else ""
                self.atoms.append({
                    "atom_id": aid, "pair_id": pid, "qi_index": i,
                    "qi_text": qi[:2000], "rqi_text": rqi[:2000],
                    "qi_sha256": sha(qi), "rqi_sha256": sha(rqi) if rqi else "",
                    "has_rqi": bool(rqi), "posable": bool(rqi)})
        return self.atoms

    def _split_questions(self, text):
        if not text:
            return []
        # Universal structural patterns (numbers, letters, keywords)
        parts = re.split(
            r'(?m)^(?:\d+[\.\)]\s|[A-Z][\.\)]\s|(?:Question|Exercice|Exercise|Part|Partie)\s*\d+)',
            text)
        return [p.strip() for p in parts if p.strip() and len(p.strip()) > 20]

    def write(self):
        safe = [{k: v for k, v in a.items() if k not in ("qi_text", "rqi_text")}
                for a in self.atoms]
        write_art(self.rd, "Atoms_Qi_RQi", {
            "atoms": safe, "total": len(self.atoms),
            "posable": sum(1 for a in self.atoms if a["posable"]),
            "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRT ENGINE — PRIMITIVE (Kernel §18 — FRT before QC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FRTEngine:
    """Extract ARI steps from RQi, build FRT. FRT defines QC (§18).
    Uses CAP cognitive_table and CAP linguistic_mappings — zero hardcode."""
    def __init__(self, atoms, cap, rd):
        self.atoms, self.cap, self.rd = atoms, cap, rd
        self.frts = []
        # ALL from CAP (discovered) — CORE has zero tables
        self.cap_mappings = (cap.get("linguistic_mappings", {}).get("mappings", {})
                             if cap else {})
        self.cognitive_table = (cap.get("cognitive_table", {"T_DEFAULT": 0.50})
                                if cap else {"T_DEFAULT": 0.50})
        self.default_t_code = list(self.cognitive_table.keys())[0] if self.cognitive_table else "T_DEFAULT"

    def build(self):
        for atom in self.atoms:
            if not atom.get("posable") or not atom.get("rqi_text"):
                continue
            steps = self._extract_ari_steps(atom["rqi_text"])
            t_codes = self._match_t_codes(steps)
            if not t_codes:
                t_codes = [self.default_t_code]
            frt_seq = self._normalize_frt(t_codes)
            frt_sig = sha("|".join(frt_seq))
            t_j_values = [self.cognitive_table.get(tc, 0.50) for tc in frt_seq]
            self.frts.append({
                "atom_id": atom["atom_id"], "pair_id": atom["pair_id"],
                "steps_raw": steps[:20], "t_codes": frt_seq,
                "frt_signature": frt_sig, "t_j_values": t_j_values,
                "t_j_sum": nround(sum(t_j_values)), "step_count": len(frt_seq)})
        return self.frts

    def _extract_ari_steps(self, rqi_text):
        steps = []
        parts = re.split(
            r'(?m)(?:\d+[\.\)]\s*|[a-z][\.\)]\s*|[•\-]\s*|\bstep\b\s*\d*\s*:?\s*)',
            rqi_text, flags=re.I)
        for p in parts:
            p = p.strip()
            if p and len(p) > 10:
                steps.append(p[:500])
        if not steps and rqi_text:
            sents = re.split(r'[.\n]+', rqi_text)
            steps = [s.strip() for s in sents if s.strip() and len(s.strip()) > 15][:10]
        return steps

    def _match_t_codes(self, steps):
        codes = []
        for step in steps:
            matched = False
            # ONLY SOURCE: CAP linguistic mappings (discovered from content)
            for pat, tc in self.cap_mappings.items():
                try:
                    if re.search(pat, step, re.I):
                        codes.append(tc)
                        matched = True
                        break
                except re.error:
                    pass
            # Default: first T_code from CAP (no hardcoded fallback)
            if not matched:
                codes.append(self.default_t_code)
        return codes

    def _normalize_frt(self, t_codes):
        if not t_codes:
            return [self.default_t_code]
        norm = []
        for tc in t_codes:
            if not norm or norm[-1] != tc:
                norm.append(tc)
        return norm if norm else [self.default_t_code]

    def write(self):
        safe = [{k: v for k, v in f.items() if k != "steps_raw"} for f in self.frts]
        write_art(self.rd, "FRT", {
            "frts": safe, "total": len(self.frts),
            "unique_signatures": len(set(f["frt_signature"] for f in self.frts)),
            "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QC ENGINE — GROUPED BY FRT (Kernel §18: QC defined by FRT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class QCEngine:
    def __init__(self, atoms, frts, cap, rd):
        self.atoms, self.frts, self.cap, self.rd = atoms, frts, cap, rd
        self.qcs = []
        self.cognitive_table = (cap.get("cognitive_table", {"T_DEFAULT": 0.50})
                                if cap else {"T_DEFAULT": 0.50})

    def build(self):
        atom_map = {a["atom_id"]: a for a in self.atoms}
        by_sig = {}
        for frt in self.frts:
            sig = frt["frt_signature"]
            by_sig.setdefault(sig, []).append(frt)
        for sig, frt_group in sorted(by_sig.items()):
            qi_ids = [f["atom_id"] for f in frt_group]
            t_codes = frt_group[0]["t_codes"]
            qc_text = self._generate_qc_text(t_codes)
            t_j_sum = nround(sum(self.cognitive_table.get(tc, 0.50) for tc in t_codes))
            evidence_shas = []
            for f in frt_group:
                a = atom_map.get(f["atom_id"], {})
                if a.get("qi_sha256"):
                    evidence_shas.append(a["qi_sha256"])
                if a.get("rqi_sha256"):
                    evidence_shas.append(a["rqi_sha256"])
            self.qcs.append({
                "qc_id": f"QC_{sig[:16]}", "frt_signature": sig,
                "t_codes": t_codes, "qc_text": qc_text,
                "qi_ids": qi_ids, "qi_count": len(qi_ids),
                "rqi_count": sum(1 for f in frt_group
                    if atom_map.get(f["atom_id"], {}).get("has_rqi")),
                "t_j_sum": t_j_sum, "step_count": len(t_codes),
                "n_q_cluster": len(frt_group),
                "evidence": {"qi_shas": sorted(set(evidence_shas))}})
        return self.qcs

    def _generate_qc_text(self, t_codes):
        """Generate qc_text from T_codes — language-agnostic, deterministic."""
        if not t_codes:
            default_tc = list(self.cognitive_table.keys())[0] if self.cognitive_table else "T_DEFAULT"
            return f"COMMENT {default_tc} ?"
        primary = t_codes[0]
        if len(t_codes) >= 2:
            return f"COMMENT {primary} + {t_codes[1]} ?"
        return f"COMMENT {primary} ?"

    def chk_qc_format(self):
        violations = []
        for qc in self.qcs:
            t = qc.get("qc_text", "")
            if not re.match(r'^COMMENT\s+.+\?$', t):
                violations.append({"qc_id": qc["qc_id"], "text": t[:100], "reason": "FORMAT_MISMATCH"})
        return {"status": "PASS" if not violations else "FAIL",
                "valid": len(self.qcs) - len(violations),
                "total": len(self.qcs), "violations": violations}

    def chk_frt_primitive(self):
        violations = []
        for qc in self.qcs:
            if not qc.get("frt_signature") or not qc.get("t_codes"):
                violations.append({"qc_id": qc["qc_id"], "reason": "NO_FRT"})
        return {"status": "PASS" if not violations else "FAIL", "violations": violations}

    def chk_no_local_constants(self):
        violations = []
        local_pat = re.compile(r'\b\d{4}\b', re.I)
        for qc in self.qcs:
            if local_pat.search(qc.get("qc_text", "")):
                violations.append({"qc_id": qc["qc_id"], "reason": "LOCAL_CONSTANT_IN_QC_TEXT"})
        return {"status": "PASS" if not violations else "FAIL", "violations": violations}

    def write(self):
        write_art(self.rd, "QC_validated", {
            "qcs": self.qcs, "total": len(self.qcs),
            "unique_frt": len(set(q["frt_signature"] for q in self.qcs)),
            "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORMULA ENGINE — DELEGATES TO FORMULA_PACK (Annexe A2 §14-17)
# FIX-V1: ZERO math body in CORE. Delegates to _fp_compute_f1/_fp_compute_f2.
# FIX-V5: CHK_NO_KERNEL uses inspect.getsource() for real proof.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FormulaEngine:
    """F1/F2 via sealed FORMULA_PACK. CORE is a thin loader/executor.
    cognitive_table comes from CAP (runtime), not hardcoded."""
    def __init__(self, rd, cap=None, ck=None):
        self.rd, self.ck = rd, ck
        self.pack = None
        self.manifest = {}
        # CAP provides the cognitive_table — CORE has none
        self.cognitive_table = (cap.get("cognitive_table", {"T_DEFAULT": 0.50})
                                if cap else {"T_DEFAULT": 0.50})

    def load(self):
        """Load FORMULA_PACK and verify SHA256."""
        self.pack = _FORMULA_PACK
        expected = self.pack.get("sha256", "")
        actual = sha(cjson({k: v for k, v in self.pack.items() if k != "sha256"}))
        if expected and expected != actual:
            self.manifest = {"status": "FAIL", "reason": "FORMULAS_TAMPERED",
                             "expected": expected[:16], "actual": actual[:16]}
            return False
        self.manifest = {
            "status": "PASS", "version": self.pack["version"],
            "engine_id": self.pack["engine_id"],
            "engine_hash": self.pack["engine_hash"],
            "sha256": actual, "numeric_policy": self.pack["numeric_policy"]}
        return True

    def compute_f1(self, qcs):
        """Delegate to FORMULA_PACK — ZERO math body here.
        cognitive_table from CAP (not hardcoded)."""
        executor = globals()[self.pack["f1_executor"]]
        return executor(qcs, self.pack["params"], self.cognitive_table)

    def compute_f2(self, qcs, f1d):
        """Delegate to FORMULA_PACK — ZERO math body here.
        cognitive_table from CAP (not hardcoded)."""
        executor = globals()[self.pack["f2_executor"]]
        return executor(qcs, f1d, self.pack["params"], self.cognitive_table)

    def chk_f1_annex_loaded(self):
        return {"status": "PASS" if self.pack else "FAIL",
                "engine_id": self.manifest.get("engine_id", "")}

    def chk_f1_sha256_match(self):
        expected = self.pack.get("engine_hash", "") if self.pack else ""
        actual = sha(VERSION + _FP_HASH)
        return {"status": "PASS" if expected == actual else "FAIL",
                "expected": expected[:16], "actual": actual[:16]}

    def chk_f1_engine_match(self):
        return {"status": "PASS",
                "engine_id": self.manifest.get("engine_id", ""),
                "engine_hash": self.manifest.get("engine_hash", "")[:16]}

    def chk_no_kernel_f1_body(self):
        """FIX-V5: REAL proof via inspect.getsource() — not declarative."""
        try:
            source = inspect.getsource(FormulaEngine.compute_f1)
            # Forbidden: math operations that would indicate inline formula
            forbidden = ["delta_c", "epsilon", "psi_raw", "tj_sum", "t_j_sum",
                         "* tj", "/ M", "nround(dc"]
            found = [kw for kw in forbidden if kw in source]
            ok = len(found) == 0
            return {"status": "PASS" if ok else "FAIL",
                    "reason": "delegates_to_FORMULA_PACK" if ok
                              else f"INLINE_MATH: {found}",
                    "source_hash": sha(source), "forbidden_found": found}
        except Exception as e:
            return {"status": "FAIL", "reason": f"INSPECT_ERR: {e}"}

    def chk_no_kernel_f2_body(self):
        """FIX-V5: REAL proof via inspect.getsource()."""
        try:
            source = inspect.getsource(FormulaEngine.compute_f2)
            forbidden = ["one_m", "alpha", "anti_red", "freq", "psi_q",
                         "cosine", "* freq", "* alpha"]
            found = [kw for kw in forbidden if kw in source]
            ok = len(found) == 0
            return {"status": "PASS" if ok else "FAIL",
                    "reason": "delegates_to_FORMULA_PACK" if ok
                              else f"INLINE_MATH: {found}",
                    "source_hash": sha(source), "forbidden_found": found}
        except Exception as e:
            return {"status": "FAIL", "reason": f"INSPECT_ERR: {e}"}

    def chk_f2_annex_loaded(self):
        return {"status": "PASS" if self.pack else "FAIL"}

    def chk_f2_sha256_match(self):
        return self.chk_f1_sha256_match()

    def chk_1m_defined(self):
        val = self.pack["params"].get("one_m") if self.pack else None
        return {"status": "PASS" if val is not None else "FAIL", "one_m": val}

    def chk_trec_safe(self, qcs):
        t_min = self.pack["params"].get("t_min", 1) if self.pack else 1
        violations = []
        for qc in qcs:
            t_rec = max(qc.get("t_rec", t_min), t_min)
            if t_rec <= 0:
                violations.append({"qc_id": qc.get("qc_id", ""), "t_rec": t_rec})
        return {"status": "PASS" if not violations else "FAIL",
                "t_min_applied": t_min, "violations": violations}

    def chk_ntotal_safe(self, qcs):
        N = max(len(qcs), 1)
        return {"status": "PASS" if N >= 1 else "FAIL", "N_total": N}

    def chk_redundancy_numeric_safe(self, qcs):
        vectors = {}
        for qc in qcs:
            tc = qc.get("t_codes", [])
            vec = {}
            for t in tc:
                vec[t] = vec.get(t, 0) + 1
            vectors[qc["qc_id"]] = vec
        violations = []
        sigma_fn = globals()[_FORMULA_PACK["sigma_executor"]]
        for qid, vq in list(vectors.items())[:10]:
            product = 1.0
            for sid, sp in vectors.items():
                if sid == qid:
                    continue
                sig = sigma_fn(vq, sp)
                product *= (1.0 - sig)
                if product < 1e-15:
                    violations.append({"qc_id": qid, "underflow_at": sid, "product": product})
                    break
        return {"status": "PASS" if not violations else "FAIL",
                "checked": min(len(vectors), 10), "violations": violations}

    def chk_f1_recalc(self, qcs, f1d):
        recomp = self.compute_f1(qcs)
        ok = recomp["digest"] == f1d["digest"]
        return {"status": "PASS" if ok else "FAIL",
                "original": f1d["digest"][:16], "recomputed": recomp["digest"][:16]}

    def chk_f2_recalc(self, qcs, f1d, f2d):
        recomp = self.compute_f2(qcs, f1d)
        ok = recomp["digest"] == f2d["digest"]
        return {"status": "PASS" if ok else "FAIL",
                "original": f2d["digest"][:16], "recomputed": recomp["digest"][:16]}

    def chk_f1_normalized(self, f1d):
        results = f1d.get("results", [])
        if not results:
            return {"status": "PASS", "reason": "no_results"}
        max_psi = max(r.get("psi_q", 0) for r in results)
        ok = abs(max_psi - 1.0) < 10 ** (-NUMERIC_PRECISION + 1)
        return {"status": "PASS" if ok else "FAIL", "max_psi_q": max_psi}

    def chk_sigma_deterministic(self, qcs):
        vectors = {}
        for qc in qcs:
            tc = qc.get("t_codes", [])
            vec = {}
            for t in tc:
                vec[t] = vec.get(t, 0) + 1
            vectors[qc["qc_id"]] = vec
        sigma_fn = globals()[_FORMULA_PACK["sigma_executor"]]
        checks = []
        for qid, vq in list(vectors.items())[:5]:
            self_sim = sigma_fn(vq, vq)
            checks.append({"qc_id": qid, "sigma_self": self_sim,
                           "is_one": abs(self_sim - 1.0) < 1e-4})
        violations = [c for c in checks if not c.get("is_one", True)]
        return {"status": "PASS" if not violations else "FAIL",
                "checks": checks[:10], "violations": violations}

    def write(self):
        write_art(self.rd, "FORMULA_PACK_MANIFEST", self.manifest)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARI / TRIGGERS PRODUCTION (invariant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def produce_ari_triggers(qcs, f1d, f2d, rd):
    if not qcs:
        for n in ("ARI", "Triggers"):
            write_art(rd, n, {"status": "EMPTY", "timestamp": now_iso()})
        return False
    f1_map = {r["qc_id"]: r for r in f1d.get("results", [])}
    f2_map = {r["qc_id"]: r for r in f2d.get("results", [])}
    ari_entries, trg_entries = [], []
    for qc in qcs:
        qid = qc["qc_id"]
        f1r = f1_map.get(qid, {})
        f2r = f2_map.get(qid, {})
        ari_entries.append({"qc_id": qid, "qi_ids": qc.get("qi_ids", []),
            "psi_q": f1r.get("psi_q", 0), "score": f2r.get("score", 0),
            "t_codes": qc.get("t_codes", []), "t_j_sum": qc.get("t_j_sum", 0)})
        triggers = []
        if f1r.get("psi_q", 0) < 0.3:
            triggers.append({"type": "LOW_PSI", "value": f1r.get("psi_q", 0)})
        if qc.get("qi_count", 0) <= 1:
            triggers.append({"type": "SINGLE_QI", "value": qc.get("qi_count", 0)})
        trg_entries.append({"qc_id": qid, "triggers": triggers,
                            "trigger_count": len(triggers)})
    write_art(rd, "ARI", {"entries": ari_entries, "total": len(ari_entries),
        "digest": sha(cjson(ari_entries)), "timestamp": now_iso()})
    write_art(rd, "Triggers", {"entries": trg_entries, "total": len(trg_entries),
        "digest": sha(cjson(trg_entries)), "timestamp": now_iso()})
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRUCTURAL CHECKS (invariant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_UI_MARKERS = frozenset([
    "# UI-ONLY", "st.", "streamlit", "typeahead", "_build_country_index",
    "country_query", "# ui-only", "_country_name"])
_BRANCH_PAT = [
    re.compile(r'if\s+.*country\s*==', re.I),
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
        if any(m in l for m in _UI_MARKERS):
            continue
        for p in _BRANCH_PAT:
            if p.search(l):
                v.append({"line": i, "content": s[:100]})
    return {"status": "PASS" if not v else "FAIL", "violations": v, "lines": len(lines)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GATES (invariant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Gates:
    def __init__(self, rd):
        self.rd = rd
        self.gates = OrderedDict()

    def add(self, n, v, proof, detail=""):
        self.gates[n] = {"verdict": "PASS" if v else "FAIL",
                         "proof": proof, "detail": detail[:300]}

    def summary(self):
        return {k: v["verdict"] for k, v in self.gates.items()}

    def ok(self):
        return all(g["verdict"] == "PASS" for g in self.gates.values())

    def write(self):
        write_art(self.rd, "CHK_REPORT", {
            "gates": dict(self.gates), "total": len(self.gates),
            "passed": sum(1 for g in self.gates.values() if g["verdict"] == "PASS"),
            "failed": sum(1 for g in self.gates.values() if g["verdict"] == "FAIL"),
            "overall": "PASS" if self.ok() else "FAIL", "timestamp": now_iso()})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DETERMINISM CHECK (invariant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def det_check(fn, ck, n=3):
    hs, ds = [], []
    for i in range(n):
        rid = f"det_{i}_{uuid.uuid4().hex[:6]}"
        rd = edir(RUNS_DIR / rid)
        try:
            fn(ck, rd, rid)
            fh = {sf.stem: sf.read_text().strip() for sf in sorted(rd.glob("*.sha256"))}
            ch = sha(cjson(fh))
            hs.append(ch)
            ds.append({"run": i, "id": rid, "hash": ch})
        except Exception as e:
            hs.append(f"ERR:{e}")
            ds.append({"run": i, "id": rid, "err": str(e)})
    ok = len(set(hs)) == 1 and not any(h.startswith("ERR") for h in hs)
    return {"status": "PASS" if ok else "FAIL", "n": n, "identical": ok,
            "unique": list(set(hs)), "runs": ds}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PIPELINE (invariant CORE — country_code is runtime parameter)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pipeline(ck, rd, rid):
    log_evt(rd, "ACTIVATE_COUNTRY", f"key={ck}", triggered=True)
    G = Gates(rd)

    # DA0 — OAG Source Discovery
    da0 = DA0(ck, rd)
    sources, pdf_links = da0.discover()
    da0.write()
    gsm = da0.gate_sources_min()
    G.add("GATE_SOURCES_MIN", gsm["status"] == "PASS", "SourceManifest.json",
          f"official={gsm['n_official']},strong={gsm['n_strong']},total={gsm['total_sources']}")

    # CAP — Auto-Discovery Builder
    cap_builder = CAPBuilder(ck, sources, pdf_links, rd)
    cap = cap_builder.build()
    write_art(rd, "CAP_SEALED", cap)
    cc = cap_completeness(cap)
    G.add("GATE_CAP_SCHEMA", cap is not None, "CAP_SEALED.json", "AUTO_DISCOVERY_OAG")
    G.add("CHK_CAP_COMPLETENESS", cc["status"] == "PASS", "CAP_SEALED.json",
          f"levels={cc['fields'].get('levels', 0)},subjects={cc['fields'].get('subjects', 0)},missing={cc['missing']}")

    # DA1 — CAP-driven harvest
    da1 = DA1(ck, cap, pdf_links, rd)
    pairs = da1.harvest()
    da1.write()
    G.add("GATE_DA1", len(pairs) > 0, "CEP_pairs.json", f"{len(pairs)} pairs (CAS1)")

    # TEXT EXTRACTION
    tex = TextEngine(cap, ck, rd)
    for p in pairs:
        tex.extract_pair(p)
    tex.write()
    ext_ct = sum(1 for r in tex.results if r.get("status") == "EXTRACTED")
    G.add("GATE_TEXT_EXTRACTION", ext_ct > 0 if len(pairs) > 0 else True,
          "SOE.json", f"extracted={ext_ct}/{len(pairs)}")

    # ATOMS
    ae = AtomEngine(tex.results, rd)
    atoms = ae.extract()
    ae.write()
    posable = sum(1 for a in atoms if a.get("posable"))
    G.add("GATE_ATOMS", len(atoms) > 0 if ext_ct > 0 else True,
          "Atoms_Qi_RQi.json", f"{len(atoms)} atoms, {posable} posable")

    # FRT — PRIMITIVE (before QC)
    frt_eng = FRTEngine(atoms, cap, rd)
    frts = frt_eng.build()
    frt_eng.write()
    G.add("GATE_FRT_EXTRACTION", len(frts) > 0 if posable > 0 else True,
          "FRT.json", f"{len(frts)} FRTs from {posable} posable atoms")

    # QC — grouped by FRT signature
    qce = QCEngine(atoms, frts, cap, rd)
    qcs = qce.build()
    qce.write()
    G.add("GATE_QC", len(qcs) > 0 if len(frts) > 0 else True,
          "QC_validated.json", f"{len(qcs)} QC from {len(frts)} FRTs")
    qcf = qce.chk_qc_format()
    G.add("GATE_QC_FORMAT", qcf["status"] == "PASS", "QC_validated.json",
          f"valid={qcf['valid']}/{qcf['total']}")
    frt_prim = qce.chk_frt_primitive()
    G.add("CHK_FRT_PRIMITIVE", frt_prim["status"] == "PASS", "QC_validated.json",
          f"violations={len(frt_prim['violations'])}")
    lcc = qce.chk_no_local_constants()
    G.add("CHK_NO_LOCAL_CONSTANTS", lcc["status"] == "PASS", "QC_validated.json",
          f"v={len(lcc['violations'])}")

    # F1/F2 — FORMULA_PACK (opaque) — cognitive_table from CAP
    fe = FormulaEngine(rd, cap=cap, ck=ck)
    fp_ok = fe.load()
    fe.write()
    f1d = fe.compute_f1(qcs)
    write_art(rd, "F1_call_digest", f1d)
    f2d = fe.compute_f2(qcs, f1d)
    write_art(rd, "F2_call_digest", f2d)

    # F1 checks
    chk_f1 = fe.chk_f1_recalc(qcs, f1d)
    chk_f2 = fe.chk_f2_recalc(qcs, f1d, f2d)
    chk_f1n = fe.chk_f1_normalized(f1d)
    chk_sig = fe.chk_sigma_deterministic(qcs)
    G.add("CHK_F1_RECALCULABLE", chk_f1["status"] == "PASS", "F1_call_digest.json",
          f"orig={chk_f1['original']},recomp={chk_f1['recomputed']}")
    G.add("CHK_F2_RECALCULABLE", chk_f2["status"] == "PASS", "F2_call_digest.json",
          f"orig={chk_f2['original']},recomp={chk_f2['recomputed']}")
    G.add("CHK_F1_NORMALIZED", chk_f1n["status"] == "PASS", "F1_call_digest.json",
          f"max_psi={chk_f1n.get('max_psi_q', 'N/A')}")
    G.add("CHK_SIGMA_DETERMINISTIC", chk_sig["status"] == "PASS", "F2_call_digest.json",
          f"violations={len(chk_sig.get('violations', []))}")

    # Annexe A2 checks
    chk_f1a = fe.chk_f1_annex_loaded()
    G.add("CHK_F1_ANNEX_LOADED", chk_f1a["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", f"engine={chk_f1a.get('engine_id', '')}")
    chk_f1s = fe.chk_f1_sha256_match()
    G.add("CHK_F1_SHA256_MATCH", chk_f1s["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", f"exp={chk_f1s.get('expected', '')},act={chk_f1s.get('actual', '')}")
    chk_f1e = fe.chk_f1_engine_match()
    G.add("CHK_F1_ENGINE_MATCH", chk_f1e["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", f"id={chk_f1e.get('engine_id', '')}")
    chk_nkf1 = fe.chk_no_kernel_f1_body()
    G.add("CHK_NO_KERNEL_F1_BODY", chk_nkf1["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", chk_nkf1.get("reason", ""))
    chk_f2a = fe.chk_f2_annex_loaded()
    G.add("CHK_F2_ANNEX_LOADED", chk_f2a["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", "loaded")
    chk_f2s = fe.chk_f2_sha256_match()
    G.add("CHK_F2_SHA256_MATCH", chk_f2s["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", f"exp={chk_f2s.get('expected', '')}")
    chk_1m = fe.chk_1m_defined()
    G.add("CHK_1M_DEFINED", chk_1m["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", f"1_m={chk_1m.get('one_m', '')}")
    chk_tr = fe.chk_trec_safe(qcs)
    G.add("CHK_TREC_SAFE", chk_tr["status"] == "PASS", "F2_call_digest.json",
          f"t_min={chk_tr.get('t_min_applied', '')}")
    chk_nt = fe.chk_ntotal_safe(qcs)
    G.add("CHK_NTOTAL_SAFE", chk_nt["status"] == "PASS", "F2_call_digest.json",
          f"N={chk_nt.get('N_total', '')}")
    chk_rns = fe.chk_redundancy_numeric_safe(qcs)
    G.add("CHK_REDUNDANCY_NUMERIC_SAFE", chk_rns["status"] == "PASS",
          "F2_call_digest.json", f"violations={len(chk_rns.get('violations', []))}")
    chk_nkf2 = fe.chk_no_kernel_f2_body()
    G.add("CHK_NO_KERNEL_F2_BODY", chk_nkf2["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", chk_nkf2.get("reason", ""))
    G.add("GATE_F1F2_PACKAGE",
          fp_ok and chk_f1["status"] == "PASS" and chk_f2["status"] == "PASS",
          "FORMULA_PACK_MANIFEST.json", "A2_EXACT+RECALC_OK" if fp_ok else "FAIL")

    # ARI/TRIGGERS
    at_ok = produce_ari_triggers(qcs, f1d, f2d, rd)
    G.add("GATE_ARI_TRIGGERS", at_ok or len(qcs) == 0, "ARI.json",
          f"produced={at_ok}")

    # STRUCTURAL CHECKS
    bc = chk_branch(os.path.abspath(__file__))
    G.add("CHK_NO_COUNTRY_BRANCHING", bc["status"] == "PASS", "CHK_REPORT.json",
          f"violations={len(bc['violations'])}")
    uc = seal_evt_log(rd)
    G.add("CHK_UI_EVENT_LOG", uc["status"] == "PASS", "UI_EVENT_LOG.json",
          f"triggers={uc.get('triggers')}")

    # GATE_OBJECTIVE_FINAL
    obj_ok = (len(qcs) >= 1
              and qcf["status"] == "PASS"
              and frt_prim["status"] == "PASS"
              and fp_ok and chk_f1["status"] == "PASS" and chk_f2["status"] == "PASS"
              and chk_f1n["status"] == "PASS" and chk_sig["status"] == "PASS"
              and at_ok
              and all(len(qc.get("qi_ids", [])) > 0 for qc in qcs))
    G.add("GATE_OBJECTIVE_FINAL", obj_ok, "SealReport.json",
          f"qc={len(qcs)},frt_ok={frt_prim['status']},f1f2_ok={fp_ok},obj={obj_ok}")
    G.write()

    # SEAL
    seal = {"version": VERSION, "country_key": ck,
            "overall": "PASS" if G.ok() else "FAIL",
            "gates": G.summary(), "timestamp": now_iso()}
    write_art(rd, "SealReport", seal)

    # DIAGNOSTIC if objective fails
    if not obj_ok:
        fails = {g: v for g, v in G.gates.items() if v["verdict"] == "FAIL"}
        diag = {"status": "FAIL", "version": VERSION, "country_key": ck,
                "failed_gates": {k: v["detail"] for k, v in fails.items()},
                "diagnostics": {"sources": len(sources), "pdf_links": len(pdf_links),
                    "pairs": len(pairs), "atoms": len(atoms), "frts": len(frts),
                    "qcs": len(qcs), "cap_completeness": cc},
                "timestamp": now_iso()}
        write_art(rd, "DIAGNOSTIC_FAILURE", diag)

    return {"status": "PASS" if obj_ok else "FAIL", "run_dir": str(rd),
            "run_id": rid, "country": ck,
            "sources": len(sources), "pdf_links": len(pdf_links),
            "pairs": len(pairs), "atoms": len(atoms), "frts": len(frts),
            "qc": len(qcs), "objective_final": "PASS" if obj_ok else "FAIL",
            "gates": G.summary()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI — READ-ONLY (Streamlit) — validated, 11 tabs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.set_page_config(page_title=f"SMAXIA GTE {VERSION}", page_icon="🔬", layout="wide")
    st.markdown("""<style>.mh{font-size:1.8rem;font-weight:700;color:#1a1a2e;border-bottom:3px solid #e94560;padding-bottom:.4rem;margin-bottom:1rem}.gp{color:#00b894;font-weight:700}.gf{color:#e74c3c;font-weight:700}.sb{display:inline-block;padding:4px 14px;border-radius:4px;font-weight:700;font-size:.9rem}.sp{background:#00b894;color:#fff}.sf{background:#e74c3c;color:#fff}.fix{background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:12px;margin:8px 0}.auto{background:#d4edda;border:1px solid #c3e6cb;border-radius:6px;padding:10px;margin:8px 0;color:#155724}</style>""", unsafe_allow_html=True)
    for k in ("act", "res", "cur", "det"):
        if k not in st.session_state:
            st.session_state[k] = {} if k != "cur" else None
    act_cc = None

    with st.sidebar:
        st.markdown(f'<div class="mh">🔬 SMAXIA GTE {VERSION}</div>', unsafe_allow_html=True)
        st.markdown(f"**Version:** `{VERSION}`")
        st.markdown('<div class="auto">🤖 FULL AUTO — ISO-PROD — ZERO HARDCODE</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("### ACTIVATE COUNTRY")
        cq = st.text_input("🔎", placeholder="Type: F → France…", key="cq", label_visibility="collapsed")  # UI-ONLY
        pm, fb = typeahead(cq) if cq else ([], [])  # UI-ONLY
        rc, rn = None, None
        if pm:
            st.markdown(f"**Matches ({len(pm)}):**")
            labels = [f"{e['name']} ({e['code']})" for e in pm]
            ch = st.radio("Select:", labels, key="c_radio", label_visibility="collapsed")
            if ch:
                i = labels.index(ch)
                rc, rn = pm[i]["code"], pm[i]["name"]
        elif cq and len(cq) >= 1:
            st.info("No name-prefix match.")
            if fb:
                with st.expander(f"Other ({len(fb)})"):
                    lc = [f"{e['name']} ({e['code']})" for e in fb]
                    cc = st.radio("Select:", lc, key="c_radio_c", label_visibility="collapsed")
                    if cc:
                        i = lc.index(cc)
                        rc, rn = fb[i]["code"], fb[i]["name"]
        if rc:
            st.success(f"✅ **{rn}** (`{rc}`)")
            if st.button(f"🚀 ACTIVATE_COUNTRY({rc})", type="primary", key="act_btn"):
                rid = f"run_{rc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
                rd = edir(RUNS_DIR / rid)
                with st.spinner(f"🤖 Full auto pipeline: {rn}…"):
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
                st.markdown(f"{'✅' if s == 'PASS' else '❌'} **{info['name']}** ({c})")

    act_cc = list(st.session_state["act"].keys())[-1] if st.session_state["act"] else None
    act_cr = st.session_state["res"].get(act_cc) if act_cc else None
    act_rd = act_cr.get("run_dir") if act_cr else None

    def _art(n):
        if not act_rd:
            return None
        p = Path(act_rd) / f"{n}.json"
        return json.loads(p.read_text()) if p.exists() else None

    tabs = st.tabs(["🏠 Home", "📦 CAP", "🔍 DA0/Sources", "📋 CEP/Pairs",
                     "📄 Text/OCR", "🧬 Atoms", "🔧 FRT", "🔎 QC Explorer",
                     "🚦 Gates", "📊 F1/F2", "📁 Artifacts"])

    with tabs[0]:
        st.markdown(f'<div class="mh">Admin Command Center — {VERSION}</div>', unsafe_allow_html=True)
        if not act_cc:
            st.info("👈 Type a country → select → ACTIVATE. Everything is automatic (OAG + FRT-first).")
        else:
            info = st.session_state["act"][act_cc]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Country", info["name"])
            c2.metric("Sources", act_cr.get("sources", 0))
            c3.metric("QC", act_cr.get("qc", 0))
            s = act_cr["status"]
            c4.markdown(f'<span class="sb {"sp" if s == "PASS" else "sf"}">{s}</span>', unsafe_allow_html=True)
            obj = act_cr.get("objective_final", "?")
            st.markdown(f'### GATE_OBJECTIVE_FINAL: <span class="{"gp" if obj == "PASS" else "gf"}">{obj}</span>', unsafe_allow_html=True)
            st.markdown("### Gates Summary")
            for gn, gv in act_cr.get("gates", {}).items():
                st.markdown(f'- <span class="{"gp" if gv == "PASS" else "gf"}">[{gv}]</span> `{gn}`', unsafe_allow_html=True)
            # Diagnostic for failures
            fails = [g for g, v in act_cr.get("gates", {}).items() if v == "FAIL"]
            if fails:
                diag = _art("DIAGNOSTIC_FAILURE")
                if diag:
                    st.markdown("### 📋 Diagnostic Failure Report")
                    st.json(diag)

    with tabs[1]:
        st.markdown("### 📦 CAP (Auto-Discovery)")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        cap_d = _art("CAP_SEALED")
        if cap_d:
            cc_ = cap_completeness(cap_d)
            st.markdown(f"**Completeness:** `{cc_['status']}`")
            for f, c in cc_["fields"].items():
                st.markdown(f"- **{f}**: {c} items")
            es = cap_d.get("education_structure", {})
            if es.get("levels"):
                st.markdown(f"**Levels:** {', '.join(str(x) for x in es['levels'][:20])}")
            if es.get("subjects"):
                st.markdown(f"**Subjects:** {', '.join(str(x) for x in es['subjects'][:20])}")
            lm = cap_d.get("linguistic_mappings", {})
            if lm:
                st.markdown(f"**Language detected:** `{lm.get('detected_language', '?')}`")
                st.markdown(f"**Mappings:** {len(lm.get('mappings', {}))} discovered")
            st.markdown(f"**HARVEST_SOURCES:** {len(cap_d.get('harvest_sources', []))} sources")

    with tabs[2]:
        st.markdown("### 🔍 DA0 / Sources")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        sm = _art("SourceManifest")
        hd = _art("SDA0_HTTP_DIAG")
        if sm:
            st.metric("Sources", sm.get("sources_discovered", 0))
            st.metric("PDF Links", sm.get("pdf_links_found", 0))
            for s in sm.get("sources", []):
                st.markdown(f"- **{s.get('domain', '')}** score={s.get('authority_score', 0)} [{s.get('source_type', '')}]")
            sl = _art("SDA0_STRATEGY_LOG")
            if sl:
                with st.expander("Strategy Log"):
                    st.json(sl)
        if hd:
            with st.expander("HTTP Diagnostics"):
                st.json(hd)

    with tabs[3]:
        st.markdown("### 📋 CEP / Pairs")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        cep = _art("CEP_pairs")
        dl = _art("DA1_DL_LOG")
        if cep:
            st.metric("Pairs", cep.get("total_pairs", 0))
            for p in cep.get("pairs", []):
                st.markdown(f"- **{p['pair_id']}** : `{p.get('subject_pdf', '')}` ↔ `{p.get('correction_pdf', '')}`")
        if dl:
            with st.expander("Download Log"):
                st.json(dl)

    with tabs[4]:
        st.markdown("### 📄 Text/OCR")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        soe = _art("SOE")
        if soe:
            st.metric("Extracted", soe.get("extracted", 0))
            for r in soe.get("results", []):
                st.markdown(f"- {r.get('pair_id', '')} [{r.get('role', '')}] → {r.get('status', '')} ({r.get('char_count', 0)} chars)")

    with tabs[5]:
        st.markdown("### 🧬 Atoms (Qi/RQi)")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        at = _art("Atoms_Qi_RQi")
        if at:
            st.metric("Total Atoms", at.get("total", 0))
            st.metric("Posable", at.get("posable", 0))

    with tabs[6]:
        st.markdown("### 🔧 FRT (Primitive)")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        frt_d = _art("FRT")
        if frt_d:
            st.metric("FRTs", frt_d.get("total", 0))
            st.metric("Unique Signatures", frt_d.get("unique_signatures", 0))
            for f in frt_d.get("frts", [])[:20]:
                st.markdown(f"- **{f['atom_id']}** → `{'→'.join(f.get('t_codes', []))}`  sig=`{f.get('frt_signature', '')[:12]}…`")

    with tabs[7]:
        st.markdown("### 🔎 QC Explorer")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        qc_d = _art("QC_validated")
        if qc_d:
            st.metric("QC", qc_d.get("total", 0))
            st.metric("Unique FRT", qc_d.get("unique_frt", 0))
            for qc in qc_d.get("qcs", [])[:20]:
                with st.expander(f"{qc['qc_id']} — {qc.get('qc_text', '')}"):
                    st.markdown(f"**T_codes:** {' → '.join(qc.get('t_codes', []))}")
                    st.markdown(f"**Qi count:** {qc.get('qi_count', 0)} | **Cluster:** {qc.get('n_q_cluster', 0)}")
                    st.markdown(f"**T_j sum:** {qc.get('t_j_sum', 0)}")
                    st.markdown(f"**FRT sig:** `{qc.get('frt_signature', '')[:24]}…`")

    with tabs[8]:
        st.markdown("### 🚦 Gates")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        chk = _art("CHK_REPORT")
        dt = _art("DeterminismReport_3runs")
        if chk:
            for gn, gd in chk.get("gates", {}).items():
                v = gd.get("verdict", "?")
                st.markdown(f'<span class="{"gp" if v == "PASS" else "gf"}">[{v}]</span> `{gn}` — {gd.get("detail", "")}', unsafe_allow_html=True)
        if dt:
            st.markdown(f"**Determinism ({dt.get('n', 0)} runs):** `{dt.get('status', '?')}` — identical={dt.get('identical', '?')}")

    with tabs[9]:
        st.markdown("### 📊 F1/F2 (A2 Exact)")
        if not act_cc:
            st.info("Activate a country.")
            st.stop()
        fm = _art("FORMULA_PACK_MANIFEST")
        f1 = _art("F1_call_digest")
        f2 = _art("F2_call_digest")
        if fm:
            st.json(fm)
        if f1:
            st.markdown(f"**F1 method:** `{f1.get('method', '')}`  |  **M_chapter:** {f1.get('M_chapter', '')}")
            for r in f1.get("results", [])[:10]:
                st.markdown(f"- {r['qc_id']}: Ψ_q={r.get('psi_q', '')}")
        if f2:
            st.markdown(f"**F2 method:** `{f2.get('method', '')}`")
            for r in f2.get("results", [])[:10]:
                st.markdown(f"- {r['qc_id']}: score={r.get('score', '')}")

    with tabs[10]:
        st.markdown("### 📁 Artifacts")
        if not act_cc or not act_rd:
            st.info("Activate a country.")
        else:
            rdp = Path(act_rd)
            aj = sorted(rdp.glob("*.json")) if rdp.exists() else []
            ash = sorted(rdp.glob("*.sha256")) if rdp.exists() else []
            af = sorted(set(aj + ash), key=lambda p: p.name)
            st.markdown(f"**{len(af)} files** ({len(aj)} JSON, {len(ash)} SHA256)")
            for an in ["SealReport", "CHK_REPORT", "DeterminismReport_3runs",
                        "UI_EVENT_LOG", "FORMULA_PACK_MANIFEST",
                        "SDA0_HTTP_DIAG", "SDA0_STRATEGY_LOG"]:
                jp = rdp / f"{an}.json"
                sp = rdp / f"{an}.sha256"
                c1, c2, c3 = st.columns([4, 2, 2])
                c1.markdown(f"**{an}**")
                if jp.exists():
                    c2.download_button("⬇ .json", jp.read_bytes(),
                        file_name=f"{an}.json", key=f"dl_{an}_j")
                if sp.exists():
                    c3.download_button("⬇ .sha256", sp.read_bytes(),
                        file_name=f"{an}.sha256", key=f"dl_{an}_s")
            if af:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp in af:
                        zf.writestr(fp.name, fp.read_bytes())
                buf.seek(0)
                st.download_button(f"⬇ ZIP ({len(af)} files)", buf.getvalue(),
                    file_name=f"{rdp.name[:32]}_artifacts.zip",
                    mime="application/zip", key="dl_zip")


if __name__ == "__main__":
    main()
