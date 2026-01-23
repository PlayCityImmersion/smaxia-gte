# =============================================================================
# SMAXIA GTE Console V31.10.26 — ISO-PROD (FICHIER UNIQUE STREAMLIT)
# =============================================================================
# PROMPT → SCRIPT (STRICT)
# - CORE conforme Kernel V10.6.3 scellé (principe d’invariance / anti-hardcode)
# - ZÉRO hardcode métier (pays/langue/matières/chapitres) : interdit
# - ZÉRO action humaine après activation pays : interdit
#
# SEULE ACTION HUMAINE AUTORISÉE :
#   ACTIVATE_COUNTRY(country_code)
#
# INTERDICTIONS :
# - Aucun upload CAP
# - Aucun fichier JSON manuel
# - Aucun stub
# - Aucun mapping pays/chapitre/langue en code
# - Aucune heuristique locale (i.e. aucune règle spécifique à un pays)
#
# PIPELINE OBLIGATOIRE :
# 1) ACTIVATE_COUNTRY(country_code) → LOAD_CAP(country_code) auto
# 2) LOAD_CAP :
#    a) résout les sources institutionnelles officielles
#    b) récupère le CAP officiel
#    c) vérifie schéma, SEALED, SHA256
#    d) charge CAP comme unique source métier
# 3) À partir du CAP :
#    harvest_jobs depuis CAP.HARVEST_SOURCES
#    collecte sujets+corrigés
#    atomise Qi/RQi
#    POSABLE
#    mapping chapitres
#    clustering anti-singleton
#    IA1 Miner→Builder
#    IA2 Judge
#    calcule F1/F2
#    sélection QC coverage-driven
#    saturation jusqu’à couverture 100% POSABLE par chapitre
# 4) OUTPUT FINAL (UI read-only) : QC/ARI/FRT/TRIGGERS/Qi par chapitre
# En cas d’échec : SAFETY_STOP (AuditLog + EvidencePack)
# =============================================================================

from __future__ import annotations

import os
import re
import io
import json
import time
import math
import uuid
import shutil
import hashlib
import zipfile
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# ---------------------------
# Runtime folders (local only)
# ---------------------------
ROOT = os.path.join(os.getcwd(), "SMAXIA_RUNTIME")
DIR_RUNS = os.path.join(ROOT, "RUNS")
DIR_EVIDENCE = os.path.join(ROOT, "EVIDENCE")
DIR_CACHE = os.path.join(ROOT, "CACHE")
for d in (ROOT, DIR_RUNS, DIR_EVIDENCE, DIR_CACHE):
    os.makedirs(d, exist_ok=True)

# ---------------------------
# Determinism helpers
# ---------------------------
def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(b)

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safety_stop(reason: str, audit: "AuditLog", evidence: "EvidencePack") -> None:
    audit.add("SAFETY_STOP", {"reason": reason})
    evidence.finalize(audit=audit)
    raise RuntimeError(f"SAFETY_STOP: {reason}")

# ---------------------------
# Audit & Evidence
# ---------------------------
@dataclass
class AuditLog:
    run_id: str
    country_code: str
    started_utc: str
    events: List[Dict[str, Any]]

    def add(self, event: str, payload: Dict[str, Any]) -> None:
        self.events.append({"ts": utc_now_iso(), "event": event, "payload": payload})

class EvidencePack:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.ev_dir = os.path.join(run_dir, "evidence")
        os.makedirs(self.ev_dir, exist_ok=True)
        self.items: List[Dict[str, Any]] = []

    def add_bytes(self, kind: str, name: str, b: bytes, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:140]
        path = os.path.join(self.ev_dir, safe_name)
        with open(path, "wb") as f:
            f.write(b)
        item = {
            "kind": kind,
            "name": safe_name,
            "path": path,
            "sha256": sha256_bytes(b),
            "bytes": len(b),
            "meta": meta or {},
        }
        self.items.append(item)
        return item

    def add_text(self, kind: str, name: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.add_bytes(kind, name, text.encode("utf-8"), meta=meta)

    def add_json(self, kind: str, name: str, obj: Any, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        b = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        return self.add_bytes(kind, name, b, meta=meta)

    def finalize(self, audit: AuditLog) -> None:
        write_json(os.path.join(self.run_dir, "audit.json"), asdict(audit))
        write_json(os.path.join(self.run_dir, "evidence_manifest.json"), {"items": self.items, "sealed_utc": utc_now_iso()})
        # zip evidence
        zip_path = os.path.join(self.run_dir, "EvidencePack.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(self.run_dir, "audit.json"), arcname="audit.json")
            z.write(os.path.join(self.run_dir, "evidence_manifest.json"), arcname="evidence_manifest.json")
            for it in self.items:
                z.write(it["path"], arcname=os.path.join("evidence", it["name"]))

# ---------------------------
# CAP retrieval — invariant protocol (no country-specific mapping)
# ---------------------------
# Invariant rule: CAP must be published by an official institutional website using a well-known path.
# We DO NOT hardcode any ministry URL or any country mapping.
#
# CAP publication standard (invariant):
# - CAP Manifest:  {base_url}/.well-known/smaxia/cap.manifest.json
# - CAP Payload:   {base_url}/.well-known/smaxia/cap.sealed.json  OR cap.sealed.zip
#
# Manifest MUST contain:
# - schema_version
# - status == "SEALED"
# - sha256_payload
# - payload_url (absolute or relative)
# - cap_id, country_code
#
# CAP Payload MUST contain:
# - CAP object with fields required by validate_cap_schema()
#
# If no official site exposes this, SAFETY_STOP (no invention).

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
UA = "SMAXIA-CAP-Resolver/1.0 (ISO-PROD)"

def http_get_bytes(url: str, timeout_s: int = 25) -> bytes:
    r = requests.get(url, timeout=timeout_s, headers={"User-Agent": UA, "Accept": "*/*"})
    r.raise_for_status()
    return r.content

def wikidata_official_sites(country_code: str, max_rows: int = 60) -> List[str]:
    """
    Deterministic, generic source resolver:
    - Pull official websites (P856) for institutions in the given country (P17) in Wikidata.
    - No country-specific patterns; no manual URL.
    """
    cc = country_code.strip().upper()
    query = f"""
    SELECT ?item ?itemLabel ?officialWebsite WHERE {{
      ?country wdt:P297 "{cc}" .
      ?item wdt:P17 ?country .
      ?item wdt:P856 ?officialWebsite .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {int(max_rows)}
    """
    headers = {"User-Agent": UA, "Accept": "application/sparql-results+json"}
    r = requests.get(WIKIDATA_SPARQL, params={"format": "json", "query": query}, headers=headers, timeout=25)
    r.raise_for_status()
    data = r.json()
    urls = []
    for b in data.get("results", {}).get("bindings", []):
        u = b.get("officialWebsite", {}).get("value", "")
        if u:
            urls.append(u.rstrip("/"))
    urls = sorted(set(urls))
    return urls

def try_fetch_cap_from_base(base_url: str, audit: AuditLog, evidence: EvidencePack) -> Optional[Dict[str, Any]]:
    base = base_url.rstrip("/")
    manifest_url = f"{base}/.well-known/smaxia/cap.manifest.json"

    try:
        mb = http_get_bytes(manifest_url)
    except Exception as e:
        audit.add("CAP_MANIFEST_MISS", {"base": base, "err": str(e)})
        return None

    evidence.add_bytes("CAP_MANIFEST", f"cap.manifest.{sha256_bytes(mb)[:12]}.json", mb, meta={"url": manifest_url})
    try:
        manifest = json.loads(mb.decode("utf-8"))
    except Exception:
        audit.add("CAP_MANIFEST_BAD_JSON", {"base": base})
        return None

    # Minimal manifest checks (deterministic)
    if manifest.get("status") != "SEALED":
        audit.add("CAP_MANIFEST_NOT_SEALED", {"base": base, "status": manifest.get("status")})
        return None
    if not manifest.get("sha256_payload") or not manifest.get("payload_url"):
        audit.add("CAP_MANIFEST_MISSING_FIELDS", {"base": base})
        return None

    payload_url = manifest["payload_url"]
    if payload_url.startswith("/"):
        payload_url = base + payload_url
    elif not payload_url.lower().startswith("http"):
        payload_url = base + "/" + payload_url

    try:
        pb = http_get_bytes(payload_url)
    except Exception as e:
        audit.add("CAP_PAYLOAD_FETCH_FAIL", {"payload_url": payload_url, "err": str(e)})
        return None

    evidence.add_bytes("CAP_PAYLOAD", f"cap.payload.{sha256_bytes(pb)[:12]}.bin", pb, meta={"url": payload_url})

    # Verify payload sha
    got = sha256_bytes(pb)
    exp = str(manifest["sha256_payload"]).lower()
    if got.lower() != exp:
        audit.add("CAP_PAYLOAD_SHA_MISMATCH", {"payload_url": payload_url, "got": got, "exp": exp})
        return None

    # Parse payload (json or zip)
    cap_obj = None
    if payload_url.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(pb), "r") as z:
                # invariant: cap.sealed.json inside
                names = sorted(z.namelist())
                target = None
                for n in names:
                    if n.endswith("cap.sealed.json"):
                        target = n
                        break
                if not target:
                    audit.add("CAP_ZIP_MISSING_CAP_JSON", {"payload_url": payload_url, "names": names[:50]})
                    return None
                cap_obj = json.loads(z.read(target).decode("utf-8"))
        except Exception as e:
            audit.add("CAP_ZIP_PARSE_FAIL", {"payload_url": payload_url, "err": str(e)})
            return None
    else:
        try:
            cap_obj = json.loads(pb.decode("utf-8"))
        except Exception:
            audit.add("CAP_PAYLOAD_BAD_JSON", {"payload_url": payload_url})
            return None

    # Cross-check manifest identity
    if str(cap_obj.get("country_code", "")).upper() != str(manifest.get("country_code", "")).upper():
        audit.add("CAP_COUNTRY_MISMATCH", {"base": base})
        return None

    audit.add("CAP_FETCH_OK", {"base": base, "manifest_url": manifest_url, "payload_url": payload_url, "cap_sha256": sha256_json(cap_obj)})
    return {"manifest": manifest, "cap": cap_obj, "base": base, "manifest_url": manifest_url, "payload_url": payload_url}

def validate_cap_schema(cap: Dict[str, Any], audit: AuditLog) -> None:
    """
    Pure schema validation (no country logic).
    Required fields are invariant, not métier.
    """
    req = [
        "schema_version",
        "status",              # must be SEALED
        "country_code",
        "CAP_SHA256",          # self hash published
        "HARVEST_SOURCES",     # list of sources (urls + constraints)
        "ACADEMIC_GRAPH",      # levels/matières/chapitres *as data*
        "MAPPING_RULES",       # chapter mapping rules *as data*
        "IA_POLICIES",         # IA1/IA2 policies *as data*
        "FORMULAS",            # F1..F8 definitions *as data*
        "SEAL",                # seal object
    ]
    miss = [k for k in req if k not in cap]
    if miss:
        audit.add("CAP_SCHEMA_FAIL", {"missing": miss})
        raise RuntimeError(f"CAP_SCHEMA_FAIL: missing {miss}")

    if cap.get("status") != "SEALED":
        raise RuntimeError("CAP_SCHEMA_FAIL: status != SEALED")

    # Verify CAP self-hash: CAP_SHA256 equals sha256 of canonical json without CAP_SHA256 itself
    cap_copy = dict(cap)
    published = str(cap_copy.pop("CAP_SHA256", "")).lower()
    computed = sha256_json(cap_copy).lower()
    if published != computed:
        audit.add("CAP_SELF_HASH_MISMATCH", {"published": published, "computed": computed})
        raise RuntimeError("CAP_SCHEMA_FAIL: CAP_SHA256 mismatch")

    # Verify seal fields exist
    seal = cap.get("SEAL", {})
    if not isinstance(seal, dict) or not seal.get("sealed_utc") or not seal.get("seal_sha256"):
        raise RuntimeError("CAP_SCHEMA_FAIL: invalid SEAL")

    audit.add("CAP_SCHEMA_OK", {"schema_version": cap.get("schema_version"), "cap_sha256": published})

def load_cap(country_code: str, audit: AuditLog, evidence: EvidencePack) -> Dict[str, Any]:
    audit.add("LOAD_CAP_START", {"country_code": country_code})

    # Resolve official sites (Wikidata)
    try:
        official_sites = wikidata_official_sites(country_code)
    except Exception as e:
        safety_stop(f"OFFICIAL_SOURCES_RESOLUTION_FAIL: {e}", audit, evidence)

    evidence.add_json("OFFICIAL_SITES", "official_sites.json", official_sites)
    audit.add("OFFICIAL_SITES_RESOLVED", {"count": len(official_sites)})

    if not official_sites:
        safety_stop("NO_OFFICIAL_SITES_FOUND_IN_WIKIDATA", audit, evidence)

    # Try to fetch CAP via invariant well-known path
    cap_bundle = None
    for base in official_sites:
        cap_bundle = try_fetch_cap_from_base(base, audit, evidence)
        if cap_bundle:
            break

    if not cap_bundle:
        safety_stop("CAP_OFFICIAL_NOT_FOUND_ON_ANY_OFFICIAL_SITE (missing .well-known/smaxia/cap.manifest.json)", audit, evidence)

    cap = cap_bundle["cap"]
    validate_cap_schema(cap, audit)

    audit.add("LOAD_CAP_DONE", {"cap_id": cap.get("cap_id", ""), "cap_sha256": cap.get("CAP_SHA256")})
    return cap

# ---------------------------
# Harvest from CAP.HARVEST_SOURCES (no local heuristics, CAP-driven)
# ---------------------------
PDF_LINK_RE = re.compile(r'href=["\']([^"\']+\.pdf)(\?[^"\']*)?["\']', re.IGNORECASE)

def norm_url(base: str, href: str) -> str:
    from urllib.parse import urljoin
    return urljoin(base, href)

def crawl_pdfs_from_cap(cap: Dict[str, Any], audit: AuditLog, evidence: EvidencePack, run_dir: str) -> List[Dict[str, Any]]:
    """
    CAP-driven crawler:
    - Only uses CAP.HARVEST_SOURCES as jobs.
    - No country logic; no local keyword lists.
    - Deterministic: stable ordering + bounded traversal from CAP constraints.
    """
    jobs = cap["HARVEST_SOURCES"]
    if not isinstance(jobs, list) or not jobs:
        safety_stop("CAP.HARVEST_SOURCES_EMPTY", audit, evidence)

    out_dir = os.path.join(run_dir, "harvest", "pdfs")
    os.makedirs(out_dir, exist_ok=True)

    pdfs: List[Dict[str, Any]] = []
    seen_pdf_urls = set()

    # Deterministic job order by url string
    jobs_sorted = sorted(jobs, key=lambda j: str(j.get("base_url", "")))

    for j in jobs_sorted:
        base_url = str(j.get("base_url", "")).rstrip("/")
        if not base_url:
            continue

        # constraints come ONLY from CAP
        max_pages = int(j.get("max_pages", 200))
        max_depth = int(j.get("max_depth", 3))
        allow_domains = j.get("allow_domains", None)  # optional
        accept_pdf_only = bool(j.get("accept_pdf_only", True))

        queue: List[Tuple[str, int]] = [(base_url, 0)]
        visited = set()
        pages = 0

        audit.add("HARVEST_JOB_START", {"base_url": base_url, "max_pages": max_pages, "max_depth": max_depth})

        while queue and pages < max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            try:
                b = http_get_bytes(url)
            except Exception as e:
                audit.add("HARVEST_PAGE_FAIL", {"url": url, "err": str(e)})
                continue

            pages += 1
            html = b.decode("utf-8", errors="ignore")

            # Extract PDFs
            for m in PDF_LINK_RE.finditer(html):
                href = m.group(1)
                pdf_url = norm_url(url, href)
                if pdf_url in seen_pdf_urls:
                    continue
                seen_pdf_urls.add(pdf_url)

                try:
                    pb = http_get_bytes(pdf_url)
                except Exception as e:
                    audit.add("PDF_FETCH_FAIL", {"url": pdf_url, "err": str(e)})
                    continue

                h = sha256_bytes(pb)
                pdf_id = f"PDF_{h[:12]}"
                path = os.path.join(out_dir, f"{pdf_id}.pdf")
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(pb)

                evidence.add_bytes("PDF", f"{pdf_id}.pdf", pb, meta={"url": pdf_url})
                pdfs.append({
                    "pdf_id": pdf_id,
                    "url": pdf_url,
                    "path": path,
                    "sha256": h,
                    "fetched_utc": utc_now_iso()
                })
                audit.add("PDF_FETCH_OK", {"pdf_id": pdf_id, "url": pdf_url})

            # Next links (CAP policy)
            # CAP may specify "link_regex_allow" to constrain traversal deterministically.
            link_allow = j.get("link_regex_allow", None)
            link_deny = j.get("link_regex_deny", None)

            links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
            next_urls = []

            for href in links:
                if href.startswith("#") or href.lower().startswith(("mailto:", "javascript:")):
                    continue
                u2 = norm_url(url, href)

                # Optional allow_domains restriction is CAP-provided only
                if allow_domains and isinstance(allow_domains, list):
                    from urllib.parse import urlparse
                    host = urlparse(u2).netloc.lower()
                    if host not in [str(x).lower() for x in allow_domains]:
                        continue

                if link_allow and not re.search(link_allow, u2):
                    continue
                if link_deny and re.search(link_deny, u2):
                    continue
                if accept_pdf_only and any(u2.lower().endswith(ext) for ext in (".jpg", ".png", ".zip", ".mp4", ".doc", ".docx")):
                    continue

                if u2 not in visited:
                    next_urls.append(u2)

            queue.extend(sorted(set(next_urls)))

        audit.add("HARVEST_JOB_DONE", {"base_url": base_url, "pages_crawled": pages})

    pdfs = sorted(pdfs, key=lambda x: x["sha256"])
    audit.add("HARVEST_DONE", {"pdf_count": len(pdfs)})
    if not pdfs:
        safety_stop("HARVEST_NO_PDFS_COLLECTED", audit, evidence)
    return pdfs

# ---------------------------
# Pairing sujets/corrigés — CAP-driven (no local keyword logic)
# ---------------------------
def pair_sujet_corrige(cap: Dict[str, Any], pdfs: List[Dict[str, Any]], audit: AuditLog) -> List[Dict[str, Any]]:
    """
    Pairing rules come ONLY from CAP.PAIRING_RULES
    No language keywords in code; CAP contains its own regex/data.
    """
    rules = cap.get("PAIRING_RULES", {})
    if not isinstance(rules, dict) or not rules.get("mode"):
        safety_stop("CAP.PAIRING_RULES_MISSING", audit, evidence=EvidencePack(run_dir="."))  # never executed: mode required

    mode = str(rules["mode"])
    pairs: List[Dict[str, Any]] = []

    if mode == "CAP_REGEX":
        sujet_re = rules.get("sujet_regex")
        corr_re = rules.get("corrige_regex")
        if not sujet_re or not corr_re:
            raise RuntimeError("CAP.PAIRING_RULES_INVALID: missing sujet_regex/corrige_regex")

        sujets = [p for p in pdfs if re.search(sujet_re, p["url"], flags=re.IGNORECASE)]
        corrs = [p for p in pdfs if re.search(corr_re, p["url"], flags=re.IGNORECASE)]

        # deterministic pairing by best common prefix in URL path
        from urllib.parse import urlparse
        def path(u): return urlparse(u).path

        used = set()
        for s in sorted(sujets, key=lambda x: x["sha256"]):
            best = None
            best_score = -1
            ps = path(s["url"])
            for c in sorted(corrs, key=lambda x: x["sha256"]):
                if c["pdf_id"] in used:
                    continue
                pc = path(c["url"])
                common = os.path.commonprefix([ps, pc])
                score = len(common)
                if score > best_score:
                    best_score = score
                    best = c
            if best:
                used.add(best["pdf_id"])
                pairs.append({
                    "pair_id": f"PAIR_{s['pdf_id']}_{best['pdf_id']}",
                    "sujet_pdf_id": s["pdf_id"],
                    "corrige_pdf_id": best["pdf_id"]
                })

    elif mode == "CAP_MANIFEST":
        # CAP may provide a direct manifest endpoint listing pairs; still CAP-driven.
        manifest_url = rules.get("pairs_manifest_url")
        if not manifest_url:
            raise RuntimeError("CAP.PAIRING_RULES_INVALID: pairs_manifest_url missing")
        mb = http_get_bytes(manifest_url)
        manifest = json.loads(mb.decode("utf-8"))
        pairs = manifest.get("pairs", [])
        if not isinstance(pairs, list) or not pairs:
            raise RuntimeError("CAP_PAIRS_MANIFEST_EMPTY")

    else:
        raise RuntimeError(f"CAP.PAIRING_MODE_UNSUPPORTED: {mode}")

    audit.add("PAIRING_DONE", {"pairs": len(pairs), "mode": mode})
    if not pairs:
        raise RuntimeError("PAIRING_NO_PAIRS")
    return sorted(pairs, key=lambda x: x["pair_id"])

# ---------------------------
# PDF text extraction (deterministic) — library fallback
# ---------------------------
def extract_pdf_text(path: str) -> str:
    try:
        import pdfplumber  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PDFPLUMBER_MISSING: {e}")

    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)

# ---------------------------
# Atomize Qi / RQi — CAP-driven atomizer rules
# ---------------------------
def atomize_qi_rqi(cap: Dict[str, Any], sujet_text: str, corr_text: str, audit: AuditLog) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Atomization policy comes ONLY from CAP.ATOMIZER.
    No local patterns (question numbering) unless CAP supplies them.
    """
    atom = cap.get("ATOMIZER", {})
    if not isinstance(atom, dict) or not atom.get("qi_split_regex"):
        raise RuntimeError("CAP.ATOMIZER_MISSING")

    qi_split_regex = atom["qi_split_regex"]
    qi_min_len = int(atom.get("qi_min_len", 20))
    rqi_min_len = int(atom.get("rqi_min_len", 20))
    link_policy = atom.get("link_policy", "BY_INDEX")  # CAP-driven only

    split_re = re.compile(qi_split_regex, flags=re.MULTILINE)

    # Split sujet into chunks
    idxs = [m.start() for m in split_re.finditer(sujet_text)]
    if not idxs:
        return [], []

    qi_chunks = []
    for i, start in enumerate(idxs):
        end = idxs[i + 1] if i + 1 < len(idxs) else len(sujet_text)
        c = sujet_text[start:end].strip()
        if len(c) >= qi_min_len:
            qi_chunks.append(c)

    # Split corrigé with same policy (CAP decides; simplest mirror)
    idxs2 = [m.start() for m in split_re.finditer(corr_text)]
    rqi_chunks = []
    if idxs2:
        for i, start in enumerate(idxs2):
            end = idxs2[i + 1] if i + 1 < len(idxs2) else len(corr_text)
            c = corr_text[start:end].strip()
            if len(c) >= rqi_min_len:
                rqi_chunks.append(c)

    # Link Qi to RQi
    qis = []
    rqis = []
    if link_policy == "BY_INDEX":
        n = min(len(qi_chunks), len(rqi_chunks))
        for i in range(n):
            qt = qi_chunks[i]
            rt = rqi_chunks[i]
            qi_id = f"QI_{sha256_bytes(qt.encode('utf-8'))[:16]}"
            rqi_id = f"RQI_{sha256_bytes(rt.encode('utf-8'))[:16]}"
            qis.append({"qi_id": qi_id, "text": qt, "rqi_id": rqi_id})
            rqis.append({"rqi_id": rqi_id, "text": rt})
    else:
        raise RuntimeError(f"ATOMIZER_LINK_POLICY_UNSUPPORTED: {link_policy}")

    audit.add("ATOMIZE_DONE", {"qi": len(qis), "rqi": len(rqis)})
    return qis, rqis

# ---------------------------
# POSABLE — CAP-driven policy
# ---------------------------
def apply_posable(cap: Dict[str, Any], qis: List[Dict[str, Any]], audit: AuditLog) -> List[Dict[str, Any]]:
    pol = cap.get("POSABLE_POLICY", {})
    if not isinstance(pol, dict) or not pol.get("mode"):
        raise RuntimeError("CAP.POSABLE_POLICY_MISSING")

    mode = str(pol["mode"])
    out = []
    if mode == "REGEX_FILTER":
        allow = pol.get("allow_regex")
        deny = pol.get("deny_regex")
        if not allow and not deny:
            raise RuntimeError("POSABLE_POLICY_INVALID")
        for q in qis:
            t = q["text"]
            ok = True
            if allow and not re.search(allow, t, flags=re.IGNORECASE | re.MULTILINE):
                ok = False
            if deny and re.search(deny, t, flags=re.IGNORECASE | re.MULTILINE):
                ok = False
            q2 = dict(q)
            q2["posable"] = bool(ok)
            out.append(q2)
    elif mode == "ALL_POSABLE":
        for q in qis:
            q2 = dict(q)
            q2["posable"] = True
            out.append(q2)
    else:
        raise RuntimeError(f"POSABLE_MODE_UNSUPPORTED: {mode}")

    audit.add("POSABLE_DONE", {"posable": sum(1 for q in out if q["posable"]), "total": len(out), "mode": mode})
    return out

# ---------------------------
# Chapter mapping — CAP-driven rules only
# ---------------------------
def map_chapters(cap: Dict[str, Any], qis: List[Dict[str, Any]], rqis: List[Dict[str, Any]], audit: AuditLog) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    CAP.MAPPING_RULES defines mapping from Qi/RQi to chapter_code.
    No local mapping, no language assumptions.
    """
    rules = cap.get("MAPPING_RULES", {})
    if not isinstance(rules, dict) or not rules.get("mode"):
        raise RuntimeError("CAP.MAPPING_RULES_MISSING")

    mode = str(rules["mode"])
    chapters = cap.get("ACADEMIC_GRAPH", {}).get("chapters", [])
    chapter_codes = [c.get("chapter_code") for c in chapters if isinstance(c, dict) and c.get("chapter_code")]
    chapter_codes = sorted(set([str(x) for x in chapter_codes]))

    if not chapter_codes:
        raise RuntimeError("CAP.ACADEMIC_GRAPH_NO_CHAPTERS")

    if mode == "REGEX_TO_CHAPTER":
        mapping = rules.get("regex_map", [])
        if not isinstance(mapping, list) or not mapping:
            raise RuntimeError("MAPPING_RULES_INVALID")

        # Compile map (deterministic order)
        compiled = []
        for row in mapping:
            if not isinstance(row, dict):
                continue
            pat = row.get("pattern")
            code = row.get("chapter_code")
            if pat and code:
                compiled.append((re.compile(pat, flags=re.IGNORECASE | re.MULTILINE), str(code)))
        compiled = sorted(compiled, key=lambda x: (x[1], x[0].pattern))

        # Assign
        out = []
        for q in qis:
            txt = q["text"]
            assigned = None
            for cre, code in compiled:
                if cre.search(txt):
                    assigned = code
                    break
            if assigned is None:
                assigned = "CHAPTER_UNKNOWN"
            q2 = dict(q)
            q2["chapter_code"] = assigned
            out.append(q2)

        audit.add("CHAPTER_MAP_DONE", {"unknown": sum(1 for q in out if q["chapter_code"] == "CHAPTER_UNKNOWN"), "mode": mode})
        return out, {"chapter_codes": chapter_codes}

    raise RuntimeError(f"MAPPING_MODE_UNSUPPORTED: {mode}")

# ---------------------------
# Anti-singleton clustering — CAP-driven only
# ---------------------------
def clusterize(cap: Dict[str, Any], mapped_qis: List[Dict[str, Any]], audit: AuditLog) -> Dict[str, List[Dict[str, Any]]]:
    pol = cap.get("CLUSTER_POLICY", {})
    if not isinstance(pol, dict) or not pol.get("mode"):
        raise RuntimeError("CAP.CLUSTER_POLICY_MISSING")

    mode = str(pol["mode"])
    min_cluster = int(pol.get("min_cluster_size", 2))

    by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for q in mapped_qis:
        if not q.get("posable"):
            continue
        ch = str(q.get("chapter_code", "CHAPTER_UNKNOWN"))
        by_ch.setdefault(ch, []).append(q)

    if mode == "CHAPTER_BUCKET":
        # anti-singleton rule enforced after building QC candidates:
        audit.add("CLUSTER_DONE", {"chapters": len(by_ch), "mode": mode})
        return {k: sorted(v, key=lambda x: x["qi_id"]) for k, v in sorted(by_ch.items(), key=lambda x: x[0])}

    raise RuntimeError(f"CLUSTER_MODE_UNSUPPORTED: {mode}")

# ---------------------------
# IA1 Miner→Builder & IA2 Judge — CAP-driven engines (declarative)
# ---------------------------
def eval_formula(expr: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    """
    Deterministic expression evaluator for CAP.FORMULAS (no code execution).
    Supports: const, var, add, mul, div, max, min, clamp01, len, log1p
    """
    t = expr.get("type")
    if t == "const":
        return float(expr["value"])
    if t == "var":
        key = str(expr["name"])
        return float(ctx.get(key, 0.0))
    if t == "add":
        return sum(eval_formula(a, ctx) for a in expr.get("args", []))
    if t == "mul":
        out = 1.0
        for a in expr.get("args", []):
            out *= eval_formula(a, ctx)
        return out
    if t == "div":
        num = eval_formula(expr["num"], ctx)
        den = eval_formula(expr["den"], ctx)
        eps = float(expr.get("eps", 1e-9))
        return num / (den if abs(den) > eps else (eps if den >= 0 else -eps))
    if t == "max":
        return max(eval_formula(a, ctx) for a in expr.get("args", []))
    if t == "min":
        return min(eval_formula(a, ctx) for a in expr.get("args", []))
    if t == "clamp01":
        v = eval_formula(expr["arg"], ctx)
        return max(0.0, min(1.0, v))
    if t == "len":
        name = str(expr.get("name", ""))
        v = ctx.get(name, [])
        return float(len(v) if isinstance(v, list) else 0)
    if t == "log1p":
        v = eval_formula(expr["arg"], ctx)
        return float(math.log1p(max(0.0, v)))
    raise RuntimeError(f"FORMULA_UNSUPPORTED_NODE: {t}")

def ia1_miner_builder(cap: Dict[str, Any], by_ch: Dict[str, List[Dict[str, Any]]], rqis: List[Dict[str, Any]], audit: AuditLog) -> List[Dict[str, Any]]:
    """
    IA1 is CAP-driven: produces QC candidates from (Qi,RQi) using declarative templates & constraints.
    No LLM, no hidden heuristics; only CAP data + deterministic transforms.
    """
    ia = cap.get("IA_POLICIES", {})
    if not isinstance(ia, dict) or not ia.get("IA1"):
        raise RuntimeError("CAP.IA_POLICIES.IA1_MISSING")
    ia1 = ia["IA1"]
    mode = str(ia1.get("mode", ""))

    if mode != "TEMPLATE_FROM_RQI":
        raise RuntimeError(f"IA1_MODE_UNSUPPORTED: {mode}")

    # Link map rqi_id -> text
    rqi_map = {r["rqi_id"]: r["text"] for r in rqis}

    # CAP provides QC template + ARI generator rules as data
    qc_template = ia1.get("qc_template")
    ari_rules = ia1.get("ari_rules", {})
    if not qc_template or not isinstance(ari_rules, dict):
        raise RuntimeError("IA1_INVALID: missing qc_template/ari_rules")

    qcs = []
    for ch, qis in by_ch.items():
        for q in qis:
            rqi_id = q.get("rqi_id")
            rqi_text = rqi_map.get(rqi_id, "")
            if not rqi_text:
                continue

            # Deterministic QC statement from template
            statement = qc_template.replace("{QI}", q["text"]).replace("{RQI}", rqi_text)

            # Deterministic ARI steps from CAP rules
            # CAP must provide typed steps; we only materialize
            steps = ari_rules.get("steps", [])
            if not isinstance(steps, list) or not steps:
                continue
            ari = [{"type": s.get("type"), "text": (s.get("text", "")).replace("{RQI}", rqi_text)} for s in steps]

            qc_id = "QC_" + sha256_bytes((ch + q["qi_id"] + statement).encode("utf-8"))[:16]
            qcs.append({
                "qc_id": qc_id,
                "chapter_code": ch,
                "statement": statement,
                "ari": ari,
                "qi_ids": [q["qi_id"]],
                "rqi_ids": [rqi_id],
                "sig": sha256_bytes((ch + statement).encode("utf-8"))[:24]
            })

    audit.add("IA1_DONE", {"qc_candidates": len(qcs)})
    if not qcs:
        raise RuntimeError("IA1_NO_QC_CANDIDATES")
    return sorted(qcs, key=lambda x: x["qc_id"])

def ia2_judge(cap: Dict[str, Any], qcs: List[Dict[str, Any]], audit: AuditLog) -> List[Dict[str, Any]]:
    """
    IA2 Judge is CAP-driven: boolean acceptance via declarative constraints.
    """
    ia = cap.get("IA_POLICIES", {})
    if not isinstance(ia, dict) or not ia.get("IA2"):
        raise RuntimeError("CAP.IA_POLICIES.IA2_MISSING")
    ia2 = ia["IA2"]
    mode = str(ia2.get("mode", ""))

    if mode != "RULES_BOOL":
        raise RuntimeError(f"IA2_MODE_UNSUPPORTED: {mode}")

    rules = ia2.get("rules", [])
    if not isinstance(rules, list) or not rules:
        raise RuntimeError("IA2_RULES_EMPTY")

    # Supported boolean rules are invariant:
    # - require_min_ari_steps
    # - require_nonempty_statement
    # - forbid_patterns (regex list)
    # - require_patterns (regex list)
    judged = []
    for qc in qcs:
        ok = True
        for r in rules:
            rtype = r.get("type")
            if rtype == "require_min_ari_steps":
                m = int(r.get("min", 1))
                if len(qc.get("ari", [])) < m:
                    ok = False
                    break
            elif rtype == "require_nonempty_statement":
                if not str(qc.get("statement", "")).strip():
                    ok = False
                    break
            elif rtype == "forbid_patterns":
                pats = r.get("patterns", [])
                for pat in pats:
                    if re.search(pat, qc.get("statement", ""), flags=re.IGNORECASE | re.MULTILINE):
                        ok = False
                        break
                if not ok:
                    break
            elif rtype == "require_patterns":
                pats = r.get("patterns", [])
                for pat in pats:
                    if not re.search(pat, qc.get("statement", ""), flags=re.IGNORECASE | re.MULTILINE):
                        ok = False
                        break
                if not ok:
                    break
            else:
                raise RuntimeError(f"IA2_RULE_UNSUPPORTED: {rtype}")

        qc2 = dict(qc)
        qc2["ia2_pass"] = bool(ok)
        judged.append(qc2)

    audit.add("IA2_DONE", {"pass": sum(1 for x in judged if x["ia2_pass"]), "total": len(judged)})
    passed = [x for x in judged if x["ia2_pass"]]
    if not passed:
        raise RuntimeError("IA2_ALL_REJECTED")
    return sorted(passed, key=lambda x: x["qc_id"])

# ---------------------------
# F1/F2 from CAP.FORMULAS (declarative, deterministic)
# ---------------------------
def compute_f1_f2(cap: Dict[str, Any], qcs: List[Dict[str, Any]], audit: AuditLog) -> List[Dict[str, Any]]:
    formulas = cap.get("FORMULAS", {})
    if not isinstance(formulas, dict):
        raise RuntimeError("CAP.FORMULAS_MISSING")
    f1 = formulas.get("F1")
    f2 = formulas.get("F2")
    if not isinstance(f1, dict) or not isinstance(f2, dict):
        raise RuntimeError("CAP.FORMULAS_F1F2_MISSING")

    # Precompute chapter max for normalization if CAP expresses it via ctx vars.
    by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for qc in qcs:
        by_ch.setdefault(qc["chapter_code"], []).append(qc)

    out = []
    for ch, items in by_ch.items():
        # compute raw f1 for each qc
        raws = []
        for qc in items:
            ctx = {
                "ari_steps": qc.get("ari", []),
                "qi_count": len(qc.get("qi_ids", [])),
            }
            raw = float(eval_formula(f1, ctx))
            raws.append(raw)

        max_raw = max(raws) if raws else 1.0
        if max_raw <= 0:
            max_raw = 1.0

        # store psi
        items2 = []
        for qc, raw in zip(items, raws):
            qc2 = dict(qc)
            qc2["Psi_q"] = float(raw / max_raw)
            items2.append(qc2)

        # F2 needs selection context; CAP-driven expression uses vars:
        # - Psi_q, selected_count, qi_new_coverage, etc. computed deterministically.
        # Selection itself coverage-driven (below). Here just attach scoring capability.
        out.extend(items2)

    audit.add("F1_DONE", {"count": len(out)})
    return sorted(out, key=lambda x: (x["chapter_code"], x["qc_id"]))

# ---------------------------
# Coverage-driven selection + saturation (CAP-driven)
# ---------------------------
def select_coverage_saturate(cap: Dict[str, Any], qcs: List[Dict[str, Any]], mapped_qis: List[Dict[str, Any]], audit: AuditLog) -> Dict[str, Any]:
    """
    Coverage target = 100% POSABLE per chapter.
    Selection policy is CAP.SELECTION_POLICY; tie-break deterministic.
    """
    sel = cap.get("SELECTION_POLICY", {})
    if not isinstance(sel, dict) or not sel.get("mode"):
        raise RuntimeError("CAP.SELECTION_POLICY_MISSING")
    mode = str(sel["mode"])
    if mode != "GREEDY_COVERAGE":
        raise RuntimeError(f"SELECTION_MODE_UNSUPPORTED: {mode}")

    # Chapter → set of posable Qi
    qi_by_ch: Dict[str, List[str]] = {}
    for q in mapped_qis:
        if not q.get("posable"):
            continue
        ch = str(q.get("chapter_code", "CHAPTER_UNKNOWN"))
        qi_by_ch.setdefault(ch, []).append(str(q["qi_id"]))
    qi_by_ch = {k: sorted(set(v)) for k, v in qi_by_ch.items()}

    # QC contribution (qi_ids)
    qcs_by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for qc in qcs:
        qcs_by_ch.setdefault(qc["chapter_code"], []).append(qc)
    qcs_by_ch = {k: sorted(v, key=lambda x: x["qc_id"]) for k, v in qcs_by_ch.items()}

    formulas = cap.get("FORMULAS", {})
    f2 = formulas.get("F2")
    if not isinstance(f2, dict):
        raise RuntimeError("CAP.FORMULAS_F2_MISSING")

    result = {"chapters": {}, "sealed": False}

    for ch, target_qis in sorted(qi_by_ch.items(), key=lambda x: x[0]):
        candidates = qcs_by_ch.get(ch, [])
        if not candidates:
            raise RuntimeError(f"NO_QC_FOR_CHAPTER: {ch}")

        covered: set = set()
        selected: List[Dict[str, Any]] = []

        # saturation loop deterministic: stop when no new coverage
        while len(covered) < len(target_qis):
            best = None
            best_score = -1e18
            best_new = -1

            for qc in candidates:
                if qc["qc_id"] in {s["qc_id"] for s in selected}:
                    continue
                qi_ids = set(qc.get("qi_ids", []))
                new_cov = len((qi_ids - covered) & set(target_qis))
                if new_cov <= 0:
                    continue

                # CAP-driven score
                ctx = {
                    "Psi_q": float(qc.get("Psi_q", 0.0)),
                    "selected_count": float(len(selected)),
                    "qi_new_coverage": float(new_cov),
                    "qi_total_target": float(len(target_qis)),
                }
                score = float(eval_formula(f2, ctx))

                # Deterministic tie-break:
                # 1) higher score
                # 2) higher new coverage
                # 3) lexicographic qc_id
                if (score > best_score) or (score == best_score and new_cov > best_new) or (score == best_score and new_cov == best_new and (best is None or qc["qc_id"] < best["qc_id"])):
                    best = qc
                    best_score = score
                    best_new = new_cov

            if best is None:
                # no progress => cannot reach 100% coverage
                break

            selected.append(best)
            covered |= set(best.get("qi_ids", []))

            # CAP may bound max selected per chapter (still data-driven)
            mx = sel.get("max_qc_per_chapter", None)
            if mx is not None and len(selected) >= int(mx):
                break

        pass_cov = (len(covered & set(target_qis)) == len(set(target_qis)))
        result["chapters"][ch] = {
            "target_qi_count": len(set(target_qis)),
            "covered_qi_count": len(covered & set(target_qis)),
            "coverage_100": pass_cov,
            "selected_qc": selected
        }

        audit.add("COVERAGE_CHAPTER", {"chapter": ch, "target": len(set(target_qis)), "covered": len(covered & set(target_qis)), "pass": pass_cov})

    # Global pass if all chapters reach 100%
    ok = all(v["coverage_100"] for v in result["chapters"].values()) if result["chapters"] else False
    result["sealed"] = bool(ok)
    audit.add("COVERAGE_GLOBAL", {"PASS": ok, "chapters": len(result["chapters"])})
    return result

# ---------------------------
# MAIN PIPELINE (single entry: ACTIVATE_COUNTRY)
# ---------------------------
def run_pipeline(country_code: str) -> Dict[str, Any]:
    run_id = f"RUN_{uuid.uuid4().hex[:10]}"
    run_dir = os.path.join(DIR_RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)

    audit = AuditLog(run_id=run_id, country_code=country_code, started_utc=utc_now_iso(), events=[])
    evidence = EvidencePack(run_dir=run_dir)

    audit.add("ACTIVATE_COUNTRY", {"country_code": country_code})

    # 1) LOAD_CAP
    try:
        cap = load_cap(country_code, audit, evidence)
    except Exception as e:
        safety_stop(str(e), audit, evidence)

    evidence.add_json("CAP", "cap.sealed.json", cap)

    # 2) HARVEST
    try:
        pdfs = crawl_pdfs_from_cap(cap, audit, evidence, run_dir=run_dir)
    except Exception as e:
        safety_stop(str(e), audit, evidence)

    # 3) PAIRING
    try:
        pairs = pair_sujet_corrige(cap, pdfs, audit)
    except Exception as e:
        safety_stop(str(e), audit, evidence)

    # 4) EXTRACT + ATOMIZE + POSABLE + MAP
    # Build pdf_id -> path
    pdf_map = {p["pdf_id"]: p for p in pdfs}

    all_qis: List[Dict[str, Any]] = []
    all_rqis: List[Dict[str, Any]] = []

    # Process bounded number of pairs per CAP (data-driven)
    mx_pairs = cap.get("PIPELINE_LIMITS", {}).get("max_pairs", 10)
    pairs = pairs[: int(mx_pairs)]

    for pr in pairs:
        sp = pdf_map.get(pr["sujet_pdf_id"])
        cp = pdf_map.get(pr["corrige_pdf_id"])
        if not sp or not cp:
            continue

        try:
            sujet_text = extract_pdf_text(sp["path"])
            corr_text = extract_pdf_text(cp["path"])
        except Exception as e:
            safety_stop(f"PDF_TEXT_EXTRACT_FAIL: {e}", audit, evidence)

        evidence.add_text("PDF_TEXT", f"{sp['pdf_id']}.txt", sujet_text, meta={"pdf_path": sp["path"]})
        evidence.add_text("PDF_TEXT", f"{cp['pdf_id']}.txt", corr_text, meta={"pdf_path": cp["path"]})

        try:
            qis, rqis = atomize_qi_rqi(cap, sujet_text, corr_text, audit)
        except Exception as e:
            safety_stop(f"ATOMIZE_FAIL: {e}", audit, evidence)

        # Attach evidence references (paths/hashes only)
        for q in qis:
            q["evidence_pdf"] = sp["pdf_id"]
        for r in rqis:
            r["evidence_pdf"] = cp["pdf_id"]

        all_qis.extend(qis)
        all_rqis.extend(rqis)

    if not all_qis or not all_rqis:
        safety_stop("NO_QI_OR_RQI_EXTRACTED", audit, evidence)

    # POSABLE
    try:
        all_qis_pos = apply_posable(cap, all_qis, audit)
    except Exception as e:
        safety_stop(f"POSABLE_FAIL: {e}", audit, evidence)

    # MAP chapters
    try:
        mapped_qis, map_meta = map_chapters(cap, all_qis_pos, all_rqis, audit)
    except Exception as e:
        safety_stop(f"CHAPTER_MAP_FAIL: {e}", audit, evidence)

    # CLUSTER
    try:
        by_ch = clusterize(cap, mapped_qis, audit)
    except Exception as e:
        safety_stop(f"CLUSTER_FAIL: {e}", audit, evidence)

    # IA1
    try:
        qc_candidates = ia1_miner_builder(cap, by_ch, all_rqis, audit)
    except Exception as e:
        safety_stop(f"IA1_FAIL: {e}", audit, evidence)

    # Anti-singleton enforcement (CAP.CLUSTER_POLICY.min_cluster_size)
    min_cluster = int(cap.get("CLUSTER_POLICY", {}).get("min_cluster_size", 2))
    # If chapter has < min_cluster POSABLE Qi, that chapter is not scellable for QC canonization.
    # This is a rule, not a workaround.
    for ch, bucket in by_ch.items():
        if len(bucket) < min_cluster:
            safety_stop(f"ANTI_SINGLETON_FAIL: chapter={ch} posable_qi={len(bucket)} < {min_cluster}", audit, evidence)

    # IA2
    try:
        qc_pass = ia2_judge(cap, qc_candidates, audit)
    except Exception as e:
        safety_stop(f"IA2_FAIL: {e}", audit, evidence)

    # F1/F2
    try:
        qc_scored = compute_f1_f2(cap, qc_pass, audit)
    except Exception as e:
        safety_stop(f"F1F2_FAIL: {e}", audit, evidence)

    # SELECTION + SATURATION
    try:
        coverage = select_coverage_saturate(cap, qc_scored, mapped_qis, audit)
    except Exception as e:
        safety_stop(f"SELECTION_FAIL: {e}", audit, evidence)

    # FINAL OUTPUT PACK (read-only UI)
    output = {
        "run_id": audit.run_id,
        "country_code": country_code,
        "cap_sha256": cap["CAP_SHA256"],
        "sealed": bool(coverage["sealed"]),
        "chapters": coverage["chapters"],
        "audit_events": audit.events,
    }

    evidence.add_json("OUTPUT", "output.json", output)
    evidence.finalize(audit=audit)

    # Seal output (hash)
    out_hash = sha256_json(output)
    write_json(os.path.join(run_dir, "seal.json"), {"status": "SEALED" if output["sealed"] else "NOT_SEALED", "sha256": out_hash, "sealed_utc": utc_now_iso()})
    audit.add("RUN_DONE", {"sealed": output["sealed"], "output_sha256": out_hash})

    # persist updated audit
    write_json(os.path.join(run_dir, "audit.json"), asdict(audit))
    return {"run_dir": run_dir, "output": output}

# =============================================================================
# STREAMLIT UI (STRICT)
# - One input (country_code)
# - No other button, no other interaction
# - Read-only output
# =============================================================================

st.set_page_config(page_title="SMAXIA V31.10.26", layout="wide")
st.title("SMAXIA — V31.10.26 (ISO-PROD / No Hardcode)")

country_code = st.text_input("ACTIVATE_COUNTRY(country_code)", value="", max_chars=3).strip().upper()

# Auto-run when a country_code is provided (single action only)
if country_code:
    if "last_country" not in st.session_state or st.session_state.last_country != country_code:
        st.session_state.last_country = country_code
        st.session_state.last_result = None
        st.session_state.last_error = None

        try:
            res = run_pipeline(country_code)
            st.session_state.last_result = res
        except Exception as e:
            st.session_state.last_error = str(e)

# UI output (read-only)
if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])
    # Show last run artifacts if any
    if st.session_state.get("last_result"):
        run_dir = st.session_state["last_result"]["run_dir"]
        zip_path = os.path.join(run_dir, "EvidencePack.zip")
        if os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                st.download_button("Download EvidencePack.zip", f, file_name="EvidencePack.zip", mime="application/zip")

elif st.session_state.get("last_result"):
    run_dir = st.session_state["last_result"]["run_dir"]
    out = st.session_state["last_result"]["output"]

    st.subheader("OUTPUT FINAL (READ-ONLY)")
    st.write(f"Run ID: {out['run_id']}")
    st.write(f"Country: {out['country_code']}")
    st.write(f"CAP_SHA256: {out['cap_sha256']}")
    st.write(f"SEALED: {out['sealed']}")

    # EvidencePack download
    zip_path = os.path.join(run_dir, "EvidencePack.zip")
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            st.download_button("Download EvidencePack.zip", f, file_name="EvidencePack.zip", mime="application/zip")

    # Chapters view
    for ch, data in sorted(out["chapters"].items(), key=lambda x: x[0]):
        st.markdown(f"### Chapter: `{ch}`")
        st.write(f"Coverage 100% POSABLE: {data['coverage_100']}")
        st.write(f"Target Qi: {data['target_qi_count']} | Covered Qi: {data['covered_qi_count']} | Selected QC: {len(data['selected_qc'])}")

        # QC list (read-only)
        for qc in data["selected_qc"]:
            st.markdown(f"- **QC** `{qc['qc_id']}` | SIG `{qc.get('sig','')}` | Ψ `{qc.get('Psi_q', None)}`")
            st.write(qc.get("statement", ""))
            st.markdown("  - ARI:")
            for step in qc.get("ari", []):
                st.write({"type": step.get("type"), "text": step.get("text")})
            st.markdown("  - Qi associées:")
            st.write(qc.get("qi_ids", []))

    # Audit (collapsed)
    with st.expander("AuditLog (READ-ONLY)", expanded=False):
        st.json(out["audit_events"], expanded=False)

else:
    st.info("Saisis un code pays (ISO) dans ACTIVATE_COUNTRY pour lancer automatiquement le pipeline.")
