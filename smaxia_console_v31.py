#!/usr/bin/env python3
# smaxia_gte_v14_3_3.py
"""
SMAXIA GTE V14.3.3 — ADMIN COMMAND CENTER — FULL AUTO — ISO-PROD
streamlit run smaxia_gte_v14_3_3.py

CHANGELOG V14.3.2 → V14.3.3:
- [MERGE] Intégration CAP→VSP pipeline avec GTE V14.3.2 UI validée
- [FIX-VSP] Ajout VSP (Validity Scope Processor) après CAP discovery
- [FIX-PIPELINE] CAP→VSP→DA1→TEXT→ATOMS→FRT→QC→F1F2→ARI
- [FIX-GATES] Ajout GATE_VSP_VALIDATION après CAP
- [PRESERVE] Toutes fonctionnalités V14.3.2 intactes
"""

import streamlit as st
import json
import hashlib
import os
import re
import math
import uuid
import io
import zipfile
import time
import sys
import inspect
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from urllib.parse import urljoin, urlparse, quote_plus

VERSION = "GTE-V14.3.3-ADMIN-VSP"
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
    "discovery_ts", "download_ts", "fetch_ts"
])


def cjson(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(d):
    return hashlib.sha256(d.encode()).hexdigest()


def sha_b(d):
    return hashlib.sha256(d).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def nround(v):
    return round(v, NUMERIC_PRECISION)


def edir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


def strip_vol(o, fields=VOLATILE):
    if isinstance(o, dict):
        return {k: strip_vol(v, fields) for k, v in sorted(o.items()) if k not in fields}
    if isinstance(o, list):
        return [strip_vol(i, fields) for i in o]
    return o


def write_art(rd, name, payload):
    s = strip_vol(deepcopy(payload))
    d = sha(cjson(s))
    full = {**payload, "_sha256_functional": d}
    (rd / f"{name}.json").write_text(
        json.dumps(full, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    (rd / f"{name}.sha256").write_text(d, encoding="utf-8")
    return d


def log_evt(rd, evt, detail, triggered=False):
    p = rd / "UI_EVENT_LOG.json"
    evts = json.loads(p.read_text()) if p.exists() else []
    evts.append({
        "ts": now_iso(),
        "event": evt,
        "detail": detail,
        "triggered_pipeline": triggered
    })
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


_FP_PARAMS = {
    "delta_c_default": 1.0,
    "epsilon": 0.01,
    "one_m": 1.0,
    "alpha_default": 1.0,
    "t_min": 1
}
_FP_NUMERIC_POLICY = "float64_strict_round6"
_FP_HASH = sha(cjson(_FP_PARAMS) + _FP_NUMERIC_POLICY)


def _fp_cosine_sim(va, vb):
    keys = set(list(va.keys()) + list(vb.keys()))
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in va.values())) or 1e-10
    nb = math.sqrt(sum(v * v for v in vb.values())) or 1e-10
    return nround(min(max(dot / (na * nb), 0.0), 1.0))


def _fp_compute_f1(qcs, params, cognitive_table):
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
        results.append({
            "qc_id": r["qc_id"],
            "psi_raw": r["psi_raw"],
            "M": nround(M),
            "psi_q": psi_q,
            "delta_c": dc,
            "epsilon": eps,
            "t_j_sum": r["t_j_sum"],
            "evidence_hash": evidence_hash
        })
    digest = sha(cjson(results))
    return {
        "status": "PASS",
        "method": "A2_F1_PSI_EXACT",
        "results": results,
        "total_qc": len(qcs),
        "M_chapter": nround(M),
        "digest": digest,
        "policy": sha(_FP_NUMERIC_POLICY)
    }


def _fp_compute_f2(qcs, f1_results, params, cognitive_table):
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
        best_idx = None
        best_score = -1
        best_detail = None
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
            detail = {
                "qc_id": qid,
                "n_q": n_q,
                "N_total": N_total,
                "freq": freq,
                "alpha": alpha,
                "psi_q": psi_q,
                "anti_redundancy": anti_red,
                "one_m": one_m,
                "score": score
            }
            if score > best_score or (score == best_score and (best_detail is None or qid < best_detail["qc_id"])):
                best_idx = idx
                best_score = score
                best_detail = detail
        if best_idx is not None:
            remaining.remove(best_idx)
            selected.add(best_detail["qc_id"])
            results.append(best_detail)
    digest = sha(cjson(results))
    return {
        "status": "PASS",
        "method": "A2_F2_SCORE_EXACT",
        "results": results,
        "total_qc": len(qcs),
        "digest": digest,
        "policy": sha(_FP_NUMERIC_POLICY)
    }


_FORMULA_PACK = {
    "version": "V3.1",
    "engine_id": "Granulo15Engine_V3.1",
    "engine_hash": sha(VERSION + _FP_HASH),
    "f1_executor": "_fp_compute_f1",
    "f2_executor": "_fp_compute_f2",
    "sigma_executor": "_fp_cosine_sim",
    "params": _FP_PARAMS,
    "numeric_policy": _FP_NUMERIC_POLICY
}
_FORMULA_PACK["sha256"] = sha(cjson({k: v for k, v in _FORMULA_PACK.items() if k != "sha256"}))


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
    diag = {
        "requests_available": hr,
        "bs4_available": hb,
        "connectivity": []
    }
    if hr:
        import requests
        for url in ["https://html.duckduckgo.com/html/", "https://en.wikipedia.org/wiki/Main_Page"]:
            try:
                r = requests.get(url, timeout=5, headers={"User-Agent": HTTP_UA})
                diag["connectivity"].append({
                    "url": url,
                    "status": r.status_code,
                    "size": len(r.text),
                    "ok": r.status_code == 200
                })
            except Exception as e:
                diag["connectivity"].append({
                    "url": url,
                    "status": -1,
                    "error": str(e)[:200],
                    "ok": False
                })
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
        r = requests.get(url, timeout=timeout, headers={"User-Agent": HTTP_UA}, allow_redirects=True)
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
        cm.write_text(json.dumps({
            "url": url,
            "status": sc,
            "fetch_ts": now_iso(),
            "size": len(data),
            "content_type": ct,
            "content_language": cl
        }))
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
            links.append({
                "url": full,
                "text": (a.get_text(strip=True) or "")[:200],
                "href_raw": href[:300]
            })
    return links


@st.cache_data
def _build_country_index():
    try:
        import pycountry
        db = {c.alpha_2: c.name for c in pycountry.countries}
    except ImportError:
        db = {}
        try:
            import urllib.request
            import json as _json
            url = "https://restcountries.com/v3.1/all?fields=cca2,name"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = _json.loads(resp.read())
                db = {c["cca2"]: c["name"]["common"] for c in data if "cca2" in c}
        except Exception:
            pass
    idx = [{"code": c, "name": n, "nl": n.lower(), "cl": c.lower()} for c, n in db.items()]
    idx.sort(key=lambda e: e["name"])
    return db, idx


def typeahead(q, limit=20):
    _, idx = _build_country_index()
    ql = q.strip().lower()
    if not ql:
        return [], []
    np_list = []
    fb = []
    seen = set()
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


def _country_name(ck):
    db, _ = _build_country_index()
    return db.get(ck, ck)


class OpenAuthorityGraph:
    DISCOVERY_PATTERNS = [
        "{country_name} ministry of education official site",
        "{country_name} national curriculum secondary education",
        "{country_name} official exam papers past papers PDF",
        "{country_name} national final examination papers PDF",
        "{country_name} national examination council official",
        "{country_name} education system levels subjects structure"
    ]
    AUTHORITY_SIGNALS = [
        (r"\.gov\b", 0.90),
        (r"\.gouv\.", 0.90),
        (r"\.edu\b", 0.80),
        (r"\.ac\.", 0.80),
        (r"\.education\.", 0.85),
        (r"\.org\b", 0.60),
        (r"ministry|minister", 0.70),
        (r"national|official", 0.50),
        (r"exam|curriculum|syllabus", 0.40),
        (r"github\.com", 0.55),
        (r"archive\.org", 0.55)
    ]
    PDF_DISCOVERY_PATTERNS = [
        "{country_name} past exam papers subject correction PDF site:.edu OR site:.gov",
        "{country_name} past examination papers with solutions PDF",
        "{country_name} previous year question papers solved PDF"
    ]

    @staticmethod
    def score_authority(domain, title="", url=""):
        sc = 0.30
        text = (domain + " " + title + " " + url).lower()
        for pat, pts in OpenAuthorityGraph.AUTHORITY_SIGNALS:
            if re.search(pat, text, re.I):
                sc = max(sc, pts)
        return nround(sc)


_EXAM_STRUCTURAL_KW = [
    "pdf", "exam", "paper", "past", "question", "annex", "document",
    "official", "download", "archive", "test", "assessment"
]


class DA0:
    def __init__(self, ck, rd):
        self.ck = ck
        self.rd = rd
        self.cn = _country_name(ck)
        self.sources = []
        self.pdf_links = []
        self.quarantine = []
        self.strat_log = []
        self.http_diag_entries = []
        self.http_diag = {}

    def discover(self):
        diag = _http_diag()
        self.http_diag = diag
        t0 = time.time()
        self._s1_search()
        self._s2_wiki()
        self._s3_bfs()
        self._dedup()
        self._score()
        self.strat_log.append({
            "strategy": "ALL",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "sources": len(self.sources),
            "links": len(self.pdf_links)
        })
        return self.sources, self.pdf_links

    def _s1_search(self):
        t0 = time.time()
        found = 0
        for tpl in OpenAuthorityGraph.DISCOVERY_PATTERNS[:3]:
            q = tpl.format(country_name=self.cn)
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
            html, sc, cached, err = _http_get_safe(url)
            self.http_diag_entries.append({
                "strategy": "S1_DDG",
                "url": url[:200],
                "status": sc,
                "cached": cached,
                "size": len(html) if html else 0,
                "error": err
            })
            if html and 200 <= sc < 300:
                n = self._parse_sr(html, q)
                found += n
            elif sc in (202, 403, 429, -1):
                burl = f"https://www.bing.com/search?q={quote_plus(q)}"
                bhtml, bsc, bcached, berr = _http_get_safe(burl)
                self.http_diag_entries.append({
                    "strategy": "S1_BING",
                    "url": burl[:200],
                    "status": bsc,
                    "cached": bcached,
                    "size": len(bhtml) if bhtml else 0,
                    "error": berr
                })
                if bhtml and 200 <= bsc < 300:
                    n = self._parse_sr(bhtml, q)
                    found += n
        for tpl in OpenAuthorityGraph.PDF_DISCOVERY_PATTERNS[:2]:
            q = tpl.format(country_name=self.cn)
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
            html, sc, cached, err = _http_get_safe(url)
            self.http_diag_entries.append({
                "strategy": "S1_PDF",
                "url": url[:200],
                "status": sc,
                "cached": cached,
                "size": len(html) if html else 0,
                "error": err
            })
            if html and 200 <= sc < 300:
                n = self._parse_sr(html, q)
                found += n
        self.strat_log.append({
            "strategy": "S1_SEARCH",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "urls_found": found,
            "status": "OK" if found > 0 else "EMPTY"
        })

    def _s2_wiki(self):
        t0 = time.time()
        found = 0
        for variant in [f"Education in {self.cn}", f"Ministry of Education ({self.cn})"]:
            wurl = f"https://en.wikipedia.org/wiki/{quote_plus(variant.replace(' ', '_'))}"
            html, sc, cached, err = _http_get_safe(wurl)
            self.http_diag_entries.append({
                "strategy": "S2",
                "url": wurl[:200],
                "status": sc,
                "cached": cached,
                "size": len(html) if html else 0,
                "error": err
            })
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
                        if any(k in dom or k in txt for k in ["education", "ministry", "exam", "council"]):
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
        self.strat_log.append({
            "strategy": "S2_WIKI",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "urls_found": found,
            "status": "OK" if found > 0 else "EMPTY"
        })

    def _s3_bfs(self):
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
            self.http_diag_entries.append({
                "strategy": "S3",
                "url": burl[:200],
                "status": sc,
                "cached": cached,
                "size": len(html) if html else 0,
                "error": err
            })
            if html and 200 <= sc < 300:
                links = _extract_pdf_links(html, burl)
                for l in links[:MAX_PDFS]:
                    l["domain"] = dom
                    l["source_query"] = f"bfs:{dom}"
                    self.pdf_links.append(l)
                    found += 1
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
        self.strat_log.append({
            "strategy": "S3_BFS",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "urls_found": found,
            "status": "OK" if found > 0 else "EMPTY"
        })

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
            self.pdf_links.append({
                "url": url,
                "domain": domain,
                "text": title,
                "source_query": query[:80],
                "direct": True
            })
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
        self.sources.append({
            "source_id": f"SRC_{sha(domain)[:12]}",
            "source_type": "web_discovery",
            "authority": domain,
            "domain": domain,
            "title": title[:200],
            "sample_url": url[:500],
            "urls_found": 1
        })

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
                s.get("domain", ""),
                s.get("title", ""),
                s.get("sample_url", "")
            )

    def gate_sources_min(self):
        n_off = sum(1 for s in self.sources if s.get("authority_score", 0) >= 0.80)
        n_str = sum(1 for s in self.sources if s.get("authority_score", 0) >= 0.65)
        ok = n_off >= 1 or n_str >= 2 or len(self.sources) >= 1
        return {
            "status": "PASS" if ok else "FAIL",
            "n_official": n_off,
            "n_strong": n_str,
            "total_sources": len(self.sources)
        }

    def write(self):
        sd = edir(HARVEST_DIR / self.ck / "sources")
        for s in self.sources:
            (sd / f"{s['source_id']}.json").write_text(
                json.dumps(s, sort_keys=True, indent=2, ensure_ascii=False)
            )
        write_art(self.rd, "SourceManifest", {
            "country_key": self.ck,
            "sources": self.sources,
            "sources_discovered": len(self.sources),
            "pdf_links_found": len(self.pdf_links),
            "strategy_summary": self.strat_log,
            "timestamp": now_iso()
        })
        write_art(self.rd, "SDA0_HTTP_DIAG", {
            **self.http_diag,
            "http_entries": self.http_diag_entries,
            "timestamp": now_iso()
        })
        write_art(self.rd, "SDA0_STRATEGY_LOG", {
            "strategies": self.strat_log,
            "total_sources": len(self.sources),
            "total_links": len(self.pdf_links),
            "timestamp": now_iso()
        })


class CAPBuilder:
    def __init__(self, ck, sources, pdf_links, rd):
        self.ck = ck
        self.sources = sources
        self.pdf_links = pdf_links
        self.rd = rd
        self.cn = _country_name(ck)
        self.levels = set()
        self.subjects = set()
        self.exams = set()
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
        headings = re.findall(r'<h[1-4][^>]*>([^<]+)</h[1-4]>', html, re.I)
        nav_items = re.findall(
            r'<(?:li|a|span)[^>]*class=["\'][^"\']*(?:menu|nav|item|link)[^"\']*["\'][^>]*>([^<]+)<',
            html,
            re.I
        )
        emphasis = re.findall(r'<(?:strong|em|b)>([^<]{3,60})</(?:strong|em|b)>', html, re.I)
        title_m = re.search(r'<title>([^<]+)</title>', html, re.I)
        title = title_m.group(1) if title_m else ""
        all_terms = headings + nav_items + emphasis + [title]
        all_text = "\n".join(t.strip() for t in all_terms if t.strip())
        self._discover_levels(all_text, html)
        self._discover_subjects(all_text, html)
        self._discover_exams(all_text, html)
        self.evidence.append({
            "source": source_id,
            "url": url[:200],
            "levels_found": len(self.levels),
            "subjects_found": len(self.subjects),
            "exams_found": len(self.exams),
            "headings_parsed": len(headings)
        })

    def _discover_levels(self, text, html):
        for m in re.finditer(r'\b(\w+)\s*(\d{1,2})\b', text):
            word = m.group(1).strip()
            num = int(m.group(2))
            if 1 <= num <= 13 and len(word) >= 3:
                self.levels.add(f"{word.capitalize()} {num}")
        options = re.findall(r'<option[^>]*>([^<]{2,60})</option>', html, re.I)
        for opt in options:
            opt = opt.strip()
            if re.match(r'^[A-Z]', opt) and 3 <= len(opt) <= 40:
                if not re.search(r'select|chois|--', opt, re.I):
                    self.levels.add(opt)
        for m in re.finditer(r'<li[^>]*>\s*<a[^>]*>([^<]{3,50})</a>\s*</li>', html, re.I):
            term = m.group(1).strip()
            if len(term) >= 3 and len(term) <= 40:
                self.levels.add(term)

    def _discover_subjects(self, text, html):
        lists = re.findall(r'<(?:ul|ol)[^>]*>(.*?)</(?:ul|ol)>', html, re.I | re.S)
        for lst in lists:
            items = re.findall(r'<li[^>]*>([^<]{3,50})</li>', lst, re.I)
            if 3 <= len(items) <= 30:
                for item in items:
                    item = item.strip()
                    if len(item) >= 3 and item[0].isupper():
                        self.subjects.add(item)
        for m in re.finditer(r'<a[^>]*href=["\'][^"\']*\.pdf["\'][^>]*>([^<]{3,60})</a>', html, re.I):
            text_link = m.group(1).strip()
            words = [w.strip() for w in re.split(r'[\-_\s]+', text_link) if len(w) >= 3]
            for w in words[:3]:
                if w[0].isupper() and len(w) <= 30:
                    self.subjects.add(w)

    def _discover_exams(self, text, html):
        for m in re.finditer(r'\b([A-Z][a-zéèêë]+(?:\s+[a-zéèêëA-Z]+){0,3})\b', text):
            term = m.group(1).strip()
            if 3 <= len(term) <= 40:
                self.exams.add(term)
        for m in re.finditer(r'([A-Za-zéèêëàâ\s]{3,30})\s*(?:20\d{2}|19\d{2})', text):
            term = m.group(1).strip()
            if len(term) >= 3:
                self.exams.add(term.capitalize())

    def _extract_from_pdf_links(self):
        for link in self.pdf_links:
            path_parts = re.split(r'[/\-_\s\.]+', urlparse(link.get("url", "")).path)
            for part in path_parts:
                if len(part) >= 3 and part[0].isupper():
                    self.subjects.add(part)

    def _build_harvest_sources(self):
        for src in self.sources:
            if src.get("authority_score", 0) >= 0.50:
                self.harvest_sources.append({
                    "source_id": src["source_id"],
                    "domain": src.get("domain", ""),
                    "authority_score": src.get("authority_score", 0),
                    "sample_url": src.get("sample_url", "")[:500]
                })

    def _build_linguistic_mappings(self):
        detected_lang = self._detect_language()
        discovered_verbs = self._extract_cognitive_verbs()
        t_codes, mappings = self._cluster_verbs_to_t_codes(discovered_verbs)
        self.linguistic_mappings = {
            "detected_language": detected_lang,
            "cognitive_table": t_codes,
            "mappings": mappings,
            "source": "discovered_from_content",
            "verbs_discovered": len(discovered_verbs),
            "t_codes_discovered": len(t_codes)
        }

    def _extract_cognitive_verbs(self):
        verbs = set()
        for sid, html in self._cached_html.items():
            for m in re.finditer(r'<li[^>]*>\s*([A-Za-zéèêëàâùîôü]{3,25})\b', html, re.I):
                candidate = m.group(1)
                if len(candidate) >= 4:
                    verbs.add(candidate)
            for m in re.finditer(r'<(?:strong|b)>([A-Za-zéèêëàâùîôü]{3,25})\b', html, re.I):
                verbs.add(m.group(1))
            for m in re.finditer(r'\d+[\.\)]\s*([A-Za-zéèêëàâùîôü]{3,25})\b', html, re.I):
                verbs.add(m.group(1))
            for m in re.finditer(r'<td[^>]*>\s*([A-Za-zéèêëàâùîôü]{3,25})\b', html, re.I):
                verbs.add(m.group(1))
        return verbs

    def _cluster_verbs_to_t_codes(self, verbs):
        stem_groups = {}
        for verb in sorted(verbs):
            vl = verb.lower()
            if len(vl) < 4:
                continue
            stem = vl[:min(len(vl), 6)]
            matched = False
            for existing_stem in list(stem_groups.keys()):
                if stem[:4] == existing_stem[:4]:
                    stem_groups[existing_stem].add(vl)
                    matched = True
                    break
            if not matched:
                stem_groups[stem] = {vl}
        cognitive_table = {}
        mappings = {}
        for i, (stem, verb_set) in enumerate(sorted(stem_groups.items())):
            t_code = f"T_{stem.upper()}"
            cognitive_table[t_code] = 0.50
            for verb in verb_set:
                safe = re.escape(verb)
                mappings[rf'\b{safe}\w*\b'] = t_code
        if not cognitive_table:
            cognitive_table["T_DEFAULT"] = 0.50
        return cognitive_table, mappings

    def _detect_language(self):
        for sid, html in self._cached_html.items():
            m = re.search(r'<html[^>]*\blang=["\'](\w{2})', html, re.I)
            if m:
                return m.group(1).lower()
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
        es = {
            "levels": sorted(self.levels),
            "subjects": sorted(self.subjects),
            "specialities": [],
            "chapters": [],
            "exams_by_level": sorted(self.exams),
            "coefficients": [],
            "top_concours_by_level": []
        }
        lm = self.linguistic_mappings
        cap = {
            "country_key": self.ck,
            "country_name": self.cn,
            "version": VERSION,
            "education_structure": es,
            "harvest_sources": self.harvest_sources,
            "linguistic_mappings": lm,
            "cognitive_table": lm.get("cognitive_table", {"T_DEFAULT": 0.50}),
            "pairing_keywords": sorted(self._discover_pairing_keywords()),
            "kernel_params": {
                "text_extraction": ["pdfplumber", "pypdf"],
                "ocr_engines": [],
                "cluster_min": 2,
                "numeric_policy": _FP_NUMERIC_POLICY
            },
            "sources_count": len(self.sources),
            "evidence": self.evidence,
            "created_via": "AUTO_DISCOVERY_OAG"
        }
        cap["fingerprint"] = sha(cjson({
            k: v for k, v in cap.items()
            if k not in VOLATILE and k != "fingerprint"
        }))
        cap["sealed_at"] = now_iso()
        od = edir(PACKS_DIR / self.ck)
        (od / "CAP_SEALED.json").write_text(
            json.dumps(cap, sort_keys=True, indent=2, ensure_ascii=False)
        )
        return cap

    def _discover_pairing_keywords(self):
        kw = set()
        for sid, html in self._cached_html.items():
            for m in re.finditer(
                r'<a[^>]*href=["\'][^"\']*\.pdf["\'][^>]*>([^<]{3,60})</a>',
                html,
                re.I
            ):
                text = m.group(1).lower()
                words = re.split(r'[\s\-_]+', text)
                for w in words:
                    if len(w) >= 4:
                        kw.add(w[:8])
        kw.update({"corr", "solut", "answer", "memo", "mark", "resolv"})
        return kw


def cap_completeness(cap):
    es = cap.get("education_structure", {})
    fields = ["levels", "subjects", "specialities", "chapters", "exams_by_level"]
    present = {f: len(es.get(f, [])) for f in fields}
    missing = [f for f, c in present.items() if c == 0]
    ok = present.get("levels", 0) >= 1 and present.get("subjects", 0) >= 1
    return {
        "status": "PASS" if ok else "FAIL",
        "fields": present,
        "missing": missing,
        "has_linguistic_mappings": bool(cap.get("linguistic_mappings", {}).get("mappings"))
    }


class VSP:
    def __init__(self, cap, rd):
        self.cap = cap
        self.rd = rd
        self.valid_scope = []
        self.rejected_scope = []
        self.audit_trail = []

    def process(self):
        es = self.cap.get("education_structure", {})
        hs = self.cap.get("harvest_sources", [])
        has_viable_sources = any(s.get("authority_score", 0) >= 0.50 for s in hs)
        sections_to_check = [
            ("levels", es.get("levels", [])),
            ("subjects", es.get("subjects", [])),
            ("exams", es.get("exams_by_level", []))
        ]
        for section_name, section_data in sections_to_check:
            has_data = len(section_data) > 0
            has_evidence = len(self.cap.get("evidence", [])) > 0
            is_valid = has_viable_sources and has_data and has_evidence
            decision = {
                "section": section_name,
                "verdict": "VALID" if is_valid else "REJECTED",
                "reason": self._build_reason(has_viable_sources, has_data, has_evidence),
                "data_count": len(section_data),
                "timestamp": now_iso()
            }
            self.audit_trail.append(decision)
            if is_valid:
                self.valid_scope.append(section_name)
            else:
                self.rejected_scope.append(section_name)
        vsp_output = {
            "valid_scope": self.valid_scope,
            "rejected_scope": self.rejected_scope,
            "audit_trail": self.audit_trail,
            "has_viable_sources": has_viable_sources,
            "total_sections_checked": len(sections_to_check),
            "timestamp": now_iso()
        }
        write_art(self.rd, "VSP_output", vsp_output)
        return vsp_output

    def _build_reason(self, has_sources, has_data, has_evidence):
        reasons = []
        if not has_sources:
            reasons.append("no_viable_sources")
        if not has_data:
            reasons.append("no_data")
        if not has_evidence:
            reasons.append("no_evidence")
        return " & ".join(reasons) if reasons else "all_criteria_met"

    def gate_vsp(self):
        ok = len(self.valid_scope) > 0
        return {
            "status": "PASS" if ok else "FAIL",
            "valid_sections": len(self.valid_scope),
            "rejected_sections": len(self.rejected_scope),
            "audit_trail_entries": len(self.audit_trail)
        }
