
# =============================================================================
# SMAXIA GTE Console V32.00.00 (Streamlit TEST=PROD — Kernel V10.6.2)
# =============================================================================
# OBJECTIF:
#   Implémenter le pipeline GTE complet (phases 1→10) en mode PACK-DRIVEN,
#   ZERO HARDCODE (pays/matières/chapitres), déterministe, CAS 1 ONLY.
#
# NOTE IMPORTANTE (CAS 1 ONLY):
#   - Aucune "reconstruction" pédagogique: tout ce qui est produit (ARI/FRT/QC/Triggers)
#     doit être traçable à au moins un corrigé (RQi) du cluster.
#   - En absence de preuves (corrigé illisible, OCR absent, etc.), quarantaine + FAIL.
#
# USAGE:
#   streamlit run smaxia_gte_v32_00_00.py
# =============================================================================

from __future__ import annotations

import io
import json
import math
import re
import hashlib
import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Iterable

import requests
import streamlit as st
import pdfplumber
from bs4 import BeautifulSoup
# =============================================================================
# OpenAI (GPT-4*) integration (AUTO-PACK + IA1 Miner/Builder)
# =============================================================================
# Notes:
# - Requires env var OPENAI_API_KEY.
# - Uses Responses API with optional Web Search tool.
# - Determinism: temperature=0 + on-disk cache keyed by sha256(input).
#   First run in "discovery" can vary if web search results change; the app seals a snapshot of URLs
#   so subsequent re-runs are deterministic for identical inputs + snapshot.
import os
import json
import hashlib
from typing import Any

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_GPT4_MODEL = os.getenv("SMAXIA_GPT4_MODEL", "gpt-4.1")  # GPT-4 family (recommended)
LLM_TEMPERATURE = float(os.getenv("SMAXIA_LLM_TEMPERATURE", "0"))

CACHE_DIR = os.getenv("SMAXIA_CACHE_DIR", ".smaxia_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _cache_get(key: str) -> Any | None:
    p = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _cache_set(key: str, obj: Any) -> None:
    p = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _responses_api(payload: dict) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    url = f"{OPENAI_API_BASE}/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:5000]}")
    return r.json()

def _extract_output_text(resp: dict) -> str:
    # Responses API convenience field
    if isinstance(resp, dict) and isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    # Fallback: traverse output messages
    out = []
    for item in resp.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                out.append(c["text"])
    return "\n".join(out).strip()

def openai_json(model: str, system: str, user: str, json_schema: dict, *, use_web_search: bool=False, allowed_domains: list[str] | None=None, recency_days: int | None=None) -> dict:
    """Return strict JSON (best-effort) and cache by hash."""
    cache_key = _sha256_text(json.dumps({
        "model": model, "system": system, "user": user,
        "schema": json_schema, "use_web_search": use_web_search,
        "allowed_domains": allowed_domains, "recency_days": recency_days,
        "temperature": LLM_TEMPERATURE
    }, sort_keys=True, ensure_ascii=False))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    tools = []
    if use_web_search:
        tool = {"type": "web_search"}
        # web search tool options in Responses API:
        # - "search_context_size": "low"|"medium"|"high"
        # - "user_location" optional
        # - "allowed_domains" optional
        # - "max_results" optional
        if allowed_domains:
            tool["allowed_domains"] = allowed_domains
        if recency_days is not None:
            tool["recency_days"] = int(recency_days)
        tools.append(tool)

    payload = {
        "model": model,
        "temperature": LLM_TEMPERATURE,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
        "tools": tools or None,
        "text": {
            "format": {
                "type": "json_schema",
                "json_schema": {"name": "smaxia_json", "schema": json_schema, "strict": True},
            }
        },
    }
    # Remove nulls for API cleanliness
    payload = {k: v for k, v in payload.items() if v is not None}

    resp = _responses_api(payload)
    txt = _extract_output_text(resp)

    try:
        obj = json.loads(txt)
    except Exception:
        # Fallback attempt: ask model to output JSON only (without schema)
        payload2 = {
            "model": model,
            "temperature": LLM_TEMPERATURE,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user + "\n\nRETOURNE UNIQUEMENT UN JSON VALIDE. AUCUN TEXTE AUTRE."}]},
            ],
            "tools": tools or None,
            "text": {"format": {"type": "json_object"}},
        }
        payload2 = {k: v for k, v in payload2.items() if v is not None}
        resp2 = _responses_api(payload2)
        txt2 = _extract_output_text(resp2)
        obj = json.loads(txt2)

    _cache_set(cache_key, obj)
    return obj


def autopack_generate(country: str, subject: str, chapter: str, model: str) -> Dict[str, Any]:
    """Generate a minimal, pack-driven configuration with NO manual JSON.
    Uses GPT-4* with Web Search tool to discover official sources and build policy/academic pack.
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "country": {"type": "string"},
            "version": {"type": "string"},
            "academic": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "levels": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "subjects": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string"},
                                "chapters": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "chapter_code": {"type": "string"},
                                            "chapter_label": {"type": "string"},
                                            "match_keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3},
                                            "match_regexes": {"type": "array", "items": {"type": "string"}, "minItems": 0},
                                        },
                                        "required": ["chapter_code", "chapter_label", "match_keywords", "match_regexes"],
                                    },
                                },
                            },
                            "required": ["label", "chapters"],
                        },
                    },
                },
                "required": ["levels", "subjects"],
            },
            "policy": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "harvest": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "roots": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "allowed_domains": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "corrige_keywords": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                            "max_pairs": {"type": "integer", "minimum": 1, "maximum": 200},
                        },
                        "required": ["roots", "allowed_domains", "corrige_keywords", "max_pairs"],
                    },
                    "clustering": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"distance_threshold": {"type": "number", "minimum": 0.01, "maximum": 0.99}},
                        "required": ["distance_threshold"],
                    },
                    "verb_synonyms": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "minProperties": 50,
                    },
                    "verb_labels": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "minProperties": 8,
                    },
                    "delta_c": {"type": "number", "minimum": 0.0001, "maximum": 1000},
                    "use_llm": {"type": "boolean"},
                },
                "required": ["harvest", "clustering", "verb_synonyms", "verb_labels", "delta_c", "use_llm"],
            },
        },
        "required": ["country", "version", "academic", "policy"],
    }

    system = (
        "Tu es SMAXIA AutoPack Generator (Kernel V10.6.2). "
        "Objectif: produire un PACK JSON minimal, pack-driven, sans hardcode métier dans le code. "
        "Tu as accès à l'outil Web Search. "
        "Règles: (1) privilégier des sources OFFICIELLES (ministère/organisme national) ; "
        "(2) fournir 1-5 roots qui listent des PDF d'épreuves + corrigés ; "
        "(3) allowed_domains = domaines officiels uniquement ; "
        "(4) verb_synonyms: >=50 verbes utiles (langue locale), mappés vers des verb_id canoniques courts (ex: SOLVE, CALCULATE, PROVE, ANALYZE, INTERPRET, SIMPLIFY, TRANSFORM, DERIVE, INTEGRATE, MODEL, CHECK) ; "
        "(5) verb_labels: regrouper verb_id -> label pédagogique ; "
        "(6) match_keywords/match_regexes: pour détecter le chapitre choisi ; "
        "(7) delta_c: 1.0 par défaut ; distance_threshold: 0.55 par défaut ; use_llm=true."
    )

    user = f"""Pays: {country}
Matière: {subject}
Chapitre: {chapter}

Tâche:
1) Trouver les domaines OFFICIELS où publier/archiver les sujets/corrigés (examens, concours, épreuves) du pays.
2) Proposer des URLs 'roots' (pages index) où un crawler simple peut collecter des liens PDF.
3) Construire le pack JSON conforme au schéma imposé.

Contraintes:
- roots doivent être des pages HTML (ou des index) qui contiennent des liens PDF.
- corrige_keywords doit inclure les variantes locales de 'corrigé/correction/solution'.
- Aucune donnée inutile : pack minimal mais complet pour déclencher HARVEST→VERDICT.
"""

    pack = openai_json(
        model=model,
        system=system,
        user=user,
        json_schema=schema,
        use_web_search=True,
        allowed_domains=None,
        recency_days=3650,
    )

    # Deterministic pack signature
    canon = json.dumps(pack, ensure_ascii=False, sort_keys=True)
    packsig = _sha256_text(canon)
    pack["packsig"] = packsig
    return pack




# =============================================================================
# Determinism utilities (NO UUID, NO RNG, NO datetime in outputs)
# =============================================================================

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def stable_id(prefix: str, *parts: Any, n: int = 12) -> str:
    """Deterministic ID based on sha256 over normalized parts."""
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            h.update(b"<None>")
        elif isinstance(p, bytes):
            h.update(b"<bytes>")
            h.update(p)
        else:
            h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"||")
    return f"{prefix}_{h.hexdigest()[:n]}"

def stable_sort(xs: Iterable[str]) -> List[str]:
    return sorted(set(xs), key=lambda s: (s.lower(), s))

def strip_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def sanitize_local_constants(s: str) -> str:
    # Remove years, long digit sequences, and normalize standalone numbers
    s = re.sub(r"\b(19|20)\d{2}\b", "ANNEE", s)
    s = re.sub(r"\b\d{3,}\b", "N", s)
    s = re.sub(r"\b\d+\b", "N", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_json_loads(s: str) -> Any:
    # Strict JSON parsing only
    return json.loads(s)

def json_dumps_det(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# =============================================================================
# Dataclasses (required)
# =============================================================================

@dataclass(frozen=True)
class Atom:
    qi_id: str
    rqi_id: str
    qi_clean: str
    rqi_clean: str
    chapter_ref: str
    locators: Dict[str, Any]
    sha256: str  # atom fingerprint

@dataclass(frozen=True)
class TraceARI:
    qi_id: str
    actions: List[Dict[str, Any]]
    preconditions: List[str]
    output_type: str  # RESULT_VALUE, PROOF, INTERVAL, GRAPH, UNKNOWN
    sha256: str       # trace fingerprint

@dataclass(frozen=True)
class QCCandidate:
    qc_id: str
    qc_text: str
    ari_spine: List[Dict[str, Any]]
    frt: Dict[str, Any]
    triggers: List[Dict[str, Any]]
    evidence_qi_ids: List[str]
    n_q_cluster: int  # ≥2 obligatoire
    chapter_ref: str
    sha256: str       # qc fingerprint

@dataclass(frozen=True)
class AuditIA2:
    qc_id: str
    checks: Dict[str, bool]
    status: str  # PASS/FAIL
    fix_recommendations: List[str]


# =============================================================================
# Pack helpers (PACK-DRIVEN; multiple acceptable shapes)
# =============================================================================

def pack_get(pack: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = pack
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def pack_country(pack: Dict[str, Any]) -> str:
    return str(pack.get("country") or pack_get(pack, "academic.country") or "").strip()

def pack_version(pack: Dict[str, Any]) -> str:
    return str(pack.get("version") or pack_get(pack, "meta.version") or "").strip()

def iter_chapters(pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Support both:
    # - pack["academic"]["chapters"] (list)
    # - pack["academic"]["subjects"][<subject>]["chapters"] (list)
    chapters: List[Dict[str, Any]] = []
    a = pack_get(pack, "academic", {}) or {}
    if isinstance(a, dict) and isinstance(a.get("chapters"), list):
        for ch in a["chapters"]:
            if isinstance(ch, dict):
                chapters.append(ch)
    subjects = a.get("subjects")
    if isinstance(subjects, dict):
        for _, subj in subjects.items():
            if isinstance(subj, dict) and isinstance(subj.get("chapters"), list):
                for ch in subj["chapters"]:
                    if isinstance(ch, dict):
                        chapters.append(ch)
    # De-duplicate by chapter_code
    out: Dict[str, Dict[str, Any]] = {}
    for ch in chapters:
        code = str(ch.get("chapter_code") or ch.get("chapter_ref") or ch.get("code") or "").strip()
        if not code:
            continue
        if code not in out:
            out[code] = ch
    return [out[k] for k in stable_sort(list(out.keys()))]

def chapter_code(ch: Dict[str, Any]) -> str:
    return str(ch.get("chapter_code") or ch.get("chapter_ref") or ch.get("code") or "").strip()

def chapter_matchers(ch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    kws = ch.get("match_keywords") or ch.get("keywords") or []
    rxs = ch.get("match_regexes") or ch.get("regexes") or []
    kws = [str(x).strip() for x in kws if str(x).strip()]
    rxs = [str(x).strip() for x in rxs if str(x).strip()]
    return kws, rxs


# =============================================================================
# Phase 1 — Harvest (Pairs Sujet/Corrigé) from pack.policy.harvest
# =============================================================================

@dataclass(frozen=True)
class Pair:
    subject_ref: str        # url or upload key
    correction_ref: str     # url or upload key
    year: Optional[str]
    source: str             # "HARVEST" or "MANUAL"
    sha256: str             # pair fingerprint


def _is_pdf_url(u: str) -> bool:
    return u.lower().split("?")[0].endswith(".pdf")

def fetch_url_bytes(url: str, timeout: int = 30) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "SMAXIA-GTE/32.0"})
        if r.status_code != 200:
            return None, f"HTTP_{r.status_code}"
        return r.content, None
    except Exception as e:
        return None, f"ERR_{type(e).__name__}"

def harvest_pairs(pack: Dict[str, Any], max_pairs: int) -> Tuple[List[Pair], Dict[str, Any]]:
    policy = pack_get(pack, "policy", {}) or {}
    harvest_cfg = policy.get("harvest") or pack_get(pack, "harvest", {}) or {}
    # Accept root_url or root_urls or roots (list)
    roots: List[str] = []
    if isinstance(harvest_cfg.get("root_urls"), list):
        roots.extend([str(x) for x in harvest_cfg["root_urls"]])
    if isinstance(harvest_cfg.get("roots"), list):
        roots.extend([str(x) for x in harvest_cfg["roots"]])
    if isinstance(harvest_cfg.get("root_url"), str) and harvest_cfg.get("root_url").strip():
        roots.append(harvest_cfg["root_url"].strip())
    roots = [r.strip() for r in roots if r and str(r).strip()]
    roots = stable_sort(roots)

    corr_kw = harvest_cfg.get("corrige_keywords") or []
    corr_kw = [str(x).lower().strip() for x in corr_kw if str(x).strip()]
    if not roots:
        return [], {"status": "NO_ROOTS", "roots": []}

    max_depth = int(harvest_cfg.get("max_depth") or 1)  # deterministic default
    allow_domains = harvest_cfg.get("allow_domains")  # optional list

    def url_allowed(u: str) -> bool:
        if not allow_domains:
            return True
        for d in allow_domains:
            if d and str(d).lower() in u.lower():
                return True
        return False

    seen_pages = set()
    pdf_links: List[str] = []
    page_queue = [(r, 0) for r in roots]

    while page_queue and len(pdf_links) < max_pairs * 4:
        url, depth = page_queue.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)
        if not url_allowed(url):
            continue
        html_bytes, err = fetch_url_bytes(url, timeout=30)
        if err or not html_bytes:
            continue
        if _is_pdf_url(url):
            pdf_links.append(url)
            continue
        if depth >= max_depth:
            continue
        try:
            soup = BeautifulSoup(html_bytes, "html.parser")
        except Exception:
            continue
        links = []
        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            href = str(href).strip()
            if href.startswith("#"):
                continue
            # absolute
            if href.startswith("http://") or href.startswith("https://"):
                u = href
            else:
                # naive join
                if url.endswith("/"):
                    u = url + href.lstrip("/")
                else:
                    u = url.rsplit("/", 1)[0] + "/" + href.lstrip("/")
            if not url_allowed(u):
                continue
            links.append(u)
        for u in stable_sort(links):
            if _is_pdf_url(u):
                pdf_links.append(u)
            else:
                page_queue.append((u, depth + 1))

    pdf_links = stable_sort([u for u in pdf_links if _is_pdf_url(u)])[: max_pairs * 4]

    # Pairing heuristic (pack-driven via corrige_keywords)
    # If corrige_keywords missing, pairing cannot be reliable => return no pairs.
    if not corr_kw:
        return [], {
            "status": "NO_CORRIGE_KEYWORDS",
            "roots": roots,
            "pdf_links_found": len(pdf_links),
            "note": "policy.harvest.corrige_keywords absent/empty => pairing disabled (CAS 1 ONLY)."
        }

    subjects: List[str] = []
    corriges: List[str] = []
    for u in pdf_links:
        lu = u.lower()
        if any(k in lu for k in corr_kw):
            corriges.append(u)
        else:
            subjects.append(u)

    # Create stable "base keys"
    def base_key(u: str) -> str:
        p = u.lower().split("?")[0].rsplit("/", 1)[-1]
        p = re.sub(r"\.pdf$", "", p)
        p = re.sub(r"(corrig(e|é)|correction|solu(tion)?)", "", p)
        p = re.sub(r"[^a-z0-9]+", " ", p).strip()
        # Remove short tokens to stabilize
        toks = [t for t in p.split() if len(t) >= 3]
        return " ".join(toks[:8])

    corr_by_key: Dict[str, List[str]] = {}
    for c in corriges:
        k = base_key(c)
        corr_by_key.setdefault(k, []).append(c)
    for k in list(corr_by_key.keys()):
        corr_by_key[k] = stable_sort(corr_by_key[k])

    pairs: List[Pair] = []
    for s in stable_sort(subjects):
        k = base_key(s)
        cand = corr_by_key.get(k, [])
        if not cand:
            continue
        corr = cand[0]
        year = None
        m = re.search(r"\b(19|20)\d{2}\b", s)
        if m:
            year = m.group(0)
        fp = stable_id("PAIRFP", s, corr, year, "HARVEST", pack_version(pack))
        pairs.append(Pair(subject_ref=s, correction_ref=corr, year=year, source="HARVEST", sha256=fp))
        if len(pairs) >= max_pairs:
            break

    return pairs, {
        "status": "OK",
        "roots": roots,
        "pdf_links_found": len(pdf_links),
        "subjects": len(subjects),
        "corriges": len(corriges),
        "pairs": len(pairs),
        "max_depth": max_depth,
    }


# =============================================================================
# Phase 2 — Atomisation (Qi/RQi extraction) from PDF text (text-first)
# =============================================================================

def pdf_to_text(pdf_bytes: bytes) -> str:
    out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                out.append(t)
    return "\n".join(out)

def normalize_text(raw: str) -> str:
    raw = raw.replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()

def split_into_questions(text: str, pack: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Deterministic, generic atomisation:
    - Split on double newlines and numbered markers when present.
    - Keep locators as (index, char_range).
    Pack can override via policy.atomization.question_split_regexes (list).
    """
    policy = pack_get(pack, "policy", {}) or {}
    atom_cfg = policy.get("atomization", {}) or {}

    split_regexes = atom_cfg.get("question_split_regexes")
    if isinstance(split_regexes, list) and split_regexes:
        # Use custom regex split points
        pts = []
        for rx in split_regexes:
            try:
                for m in re.finditer(rx, text, flags=re.MULTILINE):
                    pts.append(m.start())
            except re.error:
                continue
        pts = sorted(set([0] + [p for p in pts if 0 <= p < len(text)] + [len(text)]))
        spans = [(pts[i], pts[i+1]) for i in range(len(pts)-1) if pts[i+1] > pts[i]]
    else:
        # Default generic segmentation: paragraphs + enumerations (non-domain, punctuation-based)
        # (This is not country/matter-specific; it is a minimal structural parser.)
        parts = re.split(r"\n\s*\n", text)
        spans = []
        cursor = 0
        for p in parts:
            start = text.find(p, cursor)
            if start < 0:
                start = cursor
            end = start + len(p)
            spans.append((start, end))
            cursor = end

    qs: List[Tuple[str, Dict[str, Any]]] = []
    for i, (a, b) in enumerate(spans, start=1):
        chunk = strip_whitespace(text[a:b])
        if len(chunk) < 20:
            continue
        loc = {"chunk_index": i, "char_start": a, "char_end": b}
        qs.append((chunk, loc))
    return qs

def build_atoms_from_pair(
    pack: Dict[str, Any],
    subject_bytes: bytes,
    corr_bytes: bytes,
    pair: Pair,
    chapter_filter: Optional[str] = None
) -> Tuple[List[Atom], Dict[str, Any]]:
    subj_text = normalize_text(pdf_to_text(subject_bytes))
    corr_text = normalize_text(pdf_to_text(corr_bytes))

    if not subj_text or not corr_text:
        return [], {
            "status": "QUARANTINE",
            "reason": "PDF_TEXT_EMPTY",
            "subject_text_len": len(subj_text),
            "corr_text_len": len(corr_text),
        }

    # Atomise subject into candidate Qi chunks
    qi_chunks = split_into_questions(subj_text, pack)

    # For CAS 1: we cannot "reconstruct" RQi per Qi if pairing is unclear.
    # We perform conservative alignment by local proximity:
    # - Split correction into chunks, match by overlap of rare tokens.
    rqi_chunks = split_into_questions(corr_text, pack)

    # tokenization (deterministic)
    def tokens(s: str) -> List[str]:
        t = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", s.lower())
        return t

    rqi_index: List[Tuple[int, str, Dict[str, Any], set]] = []
    for j, (rt, rloc) in enumerate(rqi_chunks, start=1):
        rqi_index.append((j, rt, rloc, set(tokens(rt))))

    atoms: List[Atom] = []
    chapter_defs = iter_chapters(pack)
    if not chapter_defs:
        return [], {"status": "FAIL", "reason": "PACK_NO_CHAPTERS"}

    for i, (qt, qloc) in enumerate(qi_chunks, start=1):
        qi_clean = sanitize_local_constants(strip_whitespace(qt))
        if len(qi_clean) < 20:
            continue
        # Chapter resolution (POSABLE_GATE will confirm); here we set best chapter_ref deterministically
        best_ch = None
        best_score = -1
        for ch in chapter_defs:
            code = chapter_code(ch)
            if chapter_filter and code != chapter_filter:
                continue
            kws, rxs = chapter_matchers(ch)
            score = 0
            low = qi_clean.lower()
            for kw in kws:
                if kw.lower() in low:
                    score += 2
            for rx in rxs:
                try:
                    if re.search(rx, low, flags=re.IGNORECASE):
                        score += 3
                except re.error:
                    continue
            if score > best_score:
                best_score = score
                best_ch = code
        chapter_ref = best_ch or "UNRESOLVED"

        # Align to an RQi chunk: choose max Jaccard over token sets; require minimum overlap
        qtoks = set(tokens(qi_clean))
        best_rqi = None
        best_j = 0.0
        for _, rt, rloc, rtoks in rqi_index:
            inter = len(qtoks & rtoks)
            union = len(qtoks | rtoks) if qtoks or rtoks else 1
            jacc = inter / union
            if jacc > best_j:
                best_j = jacc
                best_rqi = (rt, rloc)
        if not best_rqi or best_j < 0.05:
            # quarantine this qi: no reliable correction mapping (CAS 1)
            continue

        rqi_clean = sanitize_local_constants(strip_whitespace(best_rqi[0]))
        qi_id = stable_id("QI", pair.sha256, i, qi_clean)
        rqi_id = stable_id("RQI", pair.sha256, i, rqi_clean)
        atom_fp = stable_id("ATOMFP", qi_id, rqi_id, chapter_ref, qloc, best_rqi[1], pair.sha256)

        locators = {
            "pair_sha256": pair.sha256,
            "pair_source": pair.source,
            "pair_year": pair.year,
            "qi_loc": qloc,
            "rqi_loc": best_rqi[1],
            "align_jaccard": round(best_j, 4),
            "subject_ref": pair.subject_ref,
            "correction_ref": pair.correction_ref,
        }
        atoms.append(Atom(
            qi_id=qi_id,
            rqi_id=rqi_id,
            qi_clean=qi_clean,
            rqi_clean=rqi_clean,
            chapter_ref=chapter_ref,
            locators=locators,
            sha256=atom_fp,
        ))

    atoms = sorted(atoms, key=lambda a: (a.chapter_ref, a.qi_id))
    return atoms, {"status": "OK", "atoms": len(atoms), "subject_text_len": len(subj_text), "corr_text_len": len(corr_text)}


# =============================================================================
# Phase 3 — POSABLE gate (CAS 1 ONLY)
# =============================================================================

@dataclass(frozen=True)
class PosableDecision:
    atom: Atom
    posable: bool
    reason_code: str


def posable_gate(pack: Dict[str, Any], atoms: List[Atom], chapter_filter: Optional[str] = None) -> List[PosableDecision]:
    chapter_defs = {chapter_code(ch): ch for ch in iter_chapters(pack)}
    policy = pack_get(pack, "policy", {}) or {}
    pos_cfg = policy.get("posable", {}) or {}
    question_regexes = pos_cfg.get("question_regexes")
    if not isinstance(question_regexes, list) or not question_regexes:
        # Deterministic minimal default: require '?' somewhere (language-agnostic punctuation)
        question_regexes = [r"\?"]

    out: List[PosableDecision] = []
    for a in atoms:
        if chapter_filter and a.chapter_ref != chapter_filter:
            out.append(PosableDecision(a, False, "RC_CHAPTER_FILTER"))
            continue
        if a.chapter_ref == "UNRESOLVED" or a.chapter_ref not in chapter_defs:
            out.append(PosableDecision(a, False, "RC_SCOPE_UNRESOLVED"))
            continue
        if not a.rqi_clean or len(a.rqi_clean) < 40:
            out.append(PosableDecision(a, False, "RC_CORRIGE_UNREADABLE"))
            continue
        if not a.qi_clean or len(a.qi_clean) < 20:
            out.append(PosableDecision(a, False, "RC_QI_TOO_SHORT"))
            continue
        ok_q = False
        for rx in question_regexes:
            try:
                if re.search(rx, a.qi_clean, flags=re.IGNORECASE | re.MULTILINE):
                    ok_q = True
                    break
            except re.error:
                continue
        if not ok_q:
            out.append(PosableDecision(a, False, "RC_NOT_A_QUESTION"))
            continue
        out.append(PosableDecision(a, True, "PASS"))
    return out


# =============================================================================
# Phase 4 — IA1 Miner (deterministic strict JSON emulation)
# =============================================================================

def miner_extract_trace_json_heuristic(pack: Dict[str, Any], atom: Atom) -> Dict[str, Any]:
    """
    CAS 1: Extract only facts from RQi_clean.
    Deterministic segmentation by punctuation; verb candidates from pack.policy.verb_synonyms keys.
    """
    policy = pack_get(pack, "policy", {}) or {}
    verb_syn = policy.get("verb_synonyms") or {}
    # verb_syn: {"résoudre":"SOLVE", ...} but can be any language; we scan keys.
    verb_keys = [str(k) for k in verb_syn.keys() if str(k).strip()]
    verb_keys = stable_sort(verb_keys)

    # Split RQi into segments deterministically
    segments = [strip_whitespace(x) for x in re.split(r"[;\n\.]+", atom.rqi_clean) if strip_whitespace(x)]
    steps: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments[:25], start=1):
        verb_candidates = []
        low = seg.lower()
        for vk in verb_keys:
            if vk.lower() in low:
                verb_candidates.append(vk)
        # Objects: extract tokens (no language-specific stopwords)
        toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", seg)
        toks = [t.lower() for t in toks][:12]
        objects = toks[:4] if toks else []
        steps.append({
            "order": idx,
            "verb_candidates": verb_candidates[:5],
            "surface": seg,           # copied from RQi (already cleaned & sanitized)
            "objects": objects,
            "evidence_ref": f"{atom.rqi_id}:seg{idx}"
        })

    # Preconditions and output_type are optional and pack-driven; we infer minimally from text (conservative)
    preconds = []
    pre_rx = policy.get("preconditions_regexes") or []
    if isinstance(pre_rx, list):
        for rx in pre_rx:
            try:
                if re.search(rx, atom.rqi_clean, flags=re.IGNORECASE):
                    preconds.append(str(rx))
            except re.error:
                continue
    preconds = preconds[:5]

    out_type = "UNKNOWN"
    out_rules = policy.get("output_type_rules") or {}
    # out_rules: {"RESULT_VALUE": ["=\\s*N", ...], ...}
    if isinstance(out_rules, dict):
        for k, v in out_rules.items():
            if not isinstance(v, list):
                continue
            for rx in v:
                try:
                    if re.search(rx, atom.rqi_clean, flags=re.IGNORECASE):
                        out_type = str(k)
                        break
                except re.error:
                    continue
            if out_type != "UNKNOWN":
                break

    return {
        "steps": steps,
        "preconditions": preconds,
        "output_type": out_type
    }

def miner_extract_trace_json(pack: Dict[str, Any], atom: Atom) -> Dict[str, Any]:
    """IA1 MINER (CAS 1 ONLY) returning strict JSON.
    Uses GPT-4* if enabled; falls back to deterministic heuristic otherwise.
    """
    policy = pack_get(pack, "policy", {}) or {}
    use_llm = bool(policy.get("use_llm", True)) and bool(OPENAI_API_KEY)

    if not use_llm:
        return miner_extract_trace_json_heuristic(pack, atom)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "order": {"type": "integer", "minimum": 1, "maximum": 50},
                        "verb_candidates": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "surface": {"type": "string"},
                        "objects": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["order", "verb_candidates", "surface", "objects"],
                },
            },
            "preconditions": {"type": "array", "items": {"type": "string"}},
            "output_type": {"type": "string"},
        },
        "required": ["steps", "preconditions", "output_type"],
    }

    system = (
        "Tu es IA1 MINER SMAXIA V10.6.2 (CAS 1 ONLY). "
        "Extraction FACTUELLE UNIQUEMENT à partir de RQi_clean. "
        "INTERDICTION ABSOLUE: inventer, généraliser, ajouter des étapes non présentes. "
        "Chaque champ 'surface' doit être COPIÉ de RQi_clean (sous-extrait). "
        "Retour STRICT JSON conforme au schéma."
    )

    user = f"""Qi_clean:
{atom.qi_clean}

RQi_clean:
{atom.rqi_clean}

Retourne:
- steps[] ordonnées (order),
- verb_candidates: verbes explicitement présents,
- objects: objets manipulés explicitement,
- preconditions: hypothèses explicitement citées,
- output_type (RESULT_VALUE / PROOF / INTERVAL / GRAPH / TABLE / UNKNOWN).
"""

    obj = openai_json(
        model=DEFAULT_GPT4_MODEL,
        system=system,
        user=user,
        json_schema=schema,
        use_web_search=False,
    )

    # Minimal post-safety: ensure structure
    if not isinstance(obj, dict):
        return {"steps": [], "preconditions": [], "output_type": "UNKNOWN"}
    if not isinstance(obj.get("steps"), list):
        obj["steps"] = []
    if not isinstance(obj.get("preconditions"), list):
        obj["preconditions"] = []
    if not isinstance(obj.get("output_type"), str):
        obj["output_type"] = "UNKNOWN"
    return obj

def normalize_trace(pack: Dict[str, Any], atom: Atom, miner_json: Dict[str, Any]) -> TraceARI:
    policy = pack_get(pack, "policy", {}) or {}
    verb_syn = policy.get("verb_synonyms") or {}
    # canonical verb ids are values of verb_syn
    # If missing, verb_id stays "UNMAPPED"
    cog_weights = (policy.get("cognitive") or {}).get("weights") or {}
    if not isinstance(cog_weights, dict):
        cog_weights = {}

    actions: List[Dict[str, Any]] = []
    for s in miner_json.get("steps", []):
        if not isinstance(s, dict):
            continue
        order = int(s.get("order") or 0)
        vks = s.get("verb_candidates") or []
        vks = [str(x) for x in vks if str(x).strip()]
        verb_id = "UNMAPPED"
        for vk in vks:
            if vk in verb_syn:
                verb_id = str(verb_syn[vk]).strip()
                break
        obj = ""
        objs = s.get("objects") or []
        if isinstance(objs, list) and objs:
            obj = " ".join([sanitize_local_constants(str(o)) for o in objs[:4]])
        weight = float(cog_weights.get(verb_id, 1.0))
        actions.append({
            "order": order,
            "verb_id": verb_id,
            "obj": obj.strip(),
            "weight": weight,
            "evidence_ref": s.get("evidence_ref")
        })

    # Keep only ordered actions
    actions = [a for a in actions if a.get("order", 0) > 0]
    actions.sort(key=lambda x: x["order"])

    preconds = miner_json.get("preconditions") or []
    if not isinstance(preconds, list):
        preconds = []
    preconds = [sanitize_local_constants(strip_whitespace(str(p))) for p in preconds if strip_whitespace(str(p))][:8]

    out_type = str(miner_json.get("output_type") or "UNKNOWN").strip() or "UNKNOWN"
    trace_fp = stable_id("TRACEFP", atom.sha256, json_dumps_det(actions), json_dumps_det(preconds), out_type)

    return TraceARI(
        qi_id=atom.qi_id,
        actions=actions,
        preconditions=preconds,
        output_type=out_type,
        sha256=trace_fp
    )

def ia1_miner_batch(pack: Dict[str, Any], posables: List[Atom]) -> Tuple[List[TraceARI], Dict[str, Any]]:
    traces: List[TraceARI] = []
    miner_jsons = 0
    for a in posables:
        mj = miner_extract_trace_json(pack, a)
        miner_jsons += 1
        tr = normalize_trace(pack, a, mj)
        traces.append(tr)
    traces = sorted(traces, key=lambda t: t.qi_id)
    return traces, {"status": "OK", "traces": len(traces), "miner_jsons": miner_jsons}


# =============================================================================
# Phase 5 — Clustering (deterministic, pack.policy.clustering)
# =============================================================================

def _vec_from_trace(t: TraceARI) -> Dict[str, float]:
    v: Dict[str, float] = {}
    for a in t.actions:
        vid = str(a.get("verb_id") or "UNMAPPED")
        w = float(a.get("weight") or 1.0)
        v[vid] = v.get(vid, 0.0) + w
    return v

def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    keys = set(v1.keys()) | set(v2.keys())
    num = sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in keys)
    den1 = math.sqrt(sum(v*v for v in v1.values()))
    den2 = math.sqrt(sum(v*v for v in v2.values()))
    if den1 == 0.0 or den2 == 0.0:
        return 0.0
    return num / (den1 * den2)

def build_clusters(pack: Dict[str, Any], atoms: List[Atom], traces: List[TraceARI]) -> Tuple[List[List[str]], Dict[str, Any]]:
    policy = pack_get(pack, "policy", {}) or {}
    cl_cfg = policy.get("clustering", {}) or {}
    # user pack may specify distance_threshold (0..1); similarity = 1 - distance
    dist_th = cl_cfg.get("distance_threshold")
    sim_th = cl_cfg.get("similarity_threshold")
    if sim_th is None:
        if dist_th is not None:
            try:
                sim_th = 1.0 - float(dist_th)
            except Exception:
                sim_th = 0.45
        else:
            sim_th = 0.45  # deterministic default (non-country specific)
    sim_th = float(sim_th)

    coh_th = float(cl_cfg.get("coherence_threshold") or 0.70)

    trace_by_qi = {t.qi_id: t for t in traces}
    qi_ids = [a.qi_id for a in atoms if a.qi_id in trace_by_qi]
    qi_ids = sorted(qi_ids)

    vecs = {qid: _vec_from_trace(trace_by_qi[qid]) for qid in qi_ids}

    # Union-Find deterministic merges based on sorted edge list
    parent = {qid: qid for qid in qi_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # deterministic union by lexical order
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    edges = []
    for i in range(len(qi_ids)):
        for j in range(i+1, len(qi_ids)):
            a, b = qi_ids[i], qi_ids[j]
            sim = _cosine(vecs[a], vecs[b])
            if sim >= sim_th:
                edges.append((a, b, sim))
    edges.sort(key=lambda x: (-x[2], x[0], x[1]))

    for a, b, _ in edges:
        union(a, b)

    clusters: Dict[str, List[str]] = {}
    for qid in qi_ids:
        r = find(qid)
        clusters.setdefault(r, []).append(qid)

    # Quality gate: n_q_cluster>=2 and coherence>=coh_th
    good: List[List[str]] = []
    rejected = 0
    for _, members in sorted(clusters.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True):
        members = sorted(members)
        if len(members) < 2:
            rejected += 1
            continue
        # coherence
        sims = []
        for a, b in itertools.combinations(members, 2):
            sims.append(_cosine(vecs[a], vecs[b]))
        coh = sum(sims)/len(sims) if sims else 0.0
        if coh < coh_th:
            rejected += 1
            continue
        good.append(members)

    return good, {"status": "OK", "sim_threshold": sim_th, "coherence_threshold": coh_th, "clusters": len(good), "rejected": rejected}


# =============================================================================
# Phase 6 — IA1 Builder (deterministic strict JSON emulation)
# =============================================================================

def _top_ngrams(texts: List[str], n: int, topk: int) -> List[str]:
    counts: Dict[str, int] = {}
    for t in texts:
        toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", t.lower())
        for i in range(len(toks) - n + 1):
            ng = " ".join(toks[i:i+n])
            ng = sanitize_local_constants(ng)
            if len(ng) < 8:
                continue
            counts[ng] = counts.get(ng, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items[:topk]]

def builder_build_qc_candidate_heuristic(
    pack: Dict[str, Any],
    chapter_ref: str,
    cluster_qi_ids: List[str],
    atoms_by_qi: Dict[str, Atom],
    traces_by_qi: Dict[str, TraceARI],
) -> QCCandidate:
    policy = pack_get(pack, "policy", {}) or {}
    verb_labels = policy.get("verb_labels") or {}  # optional mapping verb_id -> natural label

    # Determine "champion": longest RQi_clean (most complete) in cluster
    champ_qi = max(cluster_qi_ids, key=lambda qid: len(atoms_by_qi[qid].rqi_clean))

    # Build ARI spine: majority verb_id by order position
    max_len = max((len(traces_by_qi[qid].actions) for qid in cluster_qi_ids), default=0)
    spine: List[Dict[str, Any]] = []
    for order in range(1, max_len + 1):
        # collect verbs/objs at this order
        votes: Dict[str, float] = {}
        obj_votes: Dict[str, float] = {}
        for qid in cluster_qi_ids:
            acts = traces_by_qi[qid].actions
            step = next((a for a in acts if int(a.get("order", 0)) == order), None)
            if not step:
                continue
            vid = str(step.get("verb_id") or "UNMAPPED")
            votes[vid] = votes.get(vid, 0.0) + float(step.get("weight") or 1.0)
            obj = str(step.get("obj") or "").strip()
            if obj:
                obj_votes[obj] = obj_votes.get(obj, 0.0) + 1.0
        if not votes:
            continue
        vid = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        obj = ""
        if obj_votes:
            obj = sorted(obj_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        spine.append({"order": order, "verb_id": vid, "obj": obj, "weight": float((policy.get("cognitive") or {}).get("weights", {}).get(vid, 1.0))})

    # QC text: "Comment <verb_label> <obj> ?"
    if spine:
        main_vid = spine[0]["verb_id"]
        label = str(verb_labels.get(main_vid) or main_vid).strip().lower()
        obj = str(spine[0].get("obj") or "").strip()
        core = strip_whitespace(f"{label} {obj}".strip())
    else:
        core = "procéder"
    qc_text = f"Comment {sanitize_local_constants(core)} ?"
    qc_text = qc_text[0].upper() + qc_text[1:]
    qc_text = strip_whitespace(qc_text)

    # Triggers: frequent ngrams in qi_clean
    qi_texts = [atoms_by_qi[qid].qi_clean for qid in cluster_qi_ids]
    trig_cfg = (policy.get("triggers") or {})
    ns = trig_cfg.get("ngram_n") if isinstance(trig_cfg.get("ngram_n"), list) else [2, 3]
    patterns = []
    for n in ns:
        try:
            patterns.extend(_top_ngrams(qi_texts, int(n), topk=10))
        except Exception:
            continue
    patterns = stable_sort(patterns)
    # keep most frequent (deterministic ordering already from _top_ngrams, but we re-rank by count)
    # Recount to rank
    freq: Dict[str, int] = {}
    for p in patterns:
        freq[p] = 0
        for t in qi_texts:
            if p in t.lower():
                freq[p] += 1
    patterns = [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][:5]

    triggers = [{"pattern": p, "examples_qi_ids": cluster_qi_ids[:2]} for p in patterns]

    # FRT in 4 blocs, sourced only from evidence (RQi surfaces)
    frt = {
        "usage": {
            "qc": qc_text,
            "chapter_ref": chapter_ref,
            "evidence_qi_ids": cluster_qi_ids,
            "qi_champion_id": champ_qi,
        },
        "reponse_type": {
            "ari_spine": spine,
            "checkpoints": [sanitize_local_constants(a.get("evidence_ref", "")) for a in (traces_by_qi[champ_qi].actions[:6] if champ_qi in traces_by_qi else [])],
        },
        "pieges": {
            "notes": [],
        },
        "conclusion": {
            "output_type": traces_by_qi[champ_qi].output_type if champ_qi in traces_by_qi else "UNKNOWN",
            "validation": "Rejouer les étapes ARI sur un énoncé compatible et vérifier les checkpoints."
        }
    }

    # Evidence pack
    qc_id = stable_id("QC", chapter_ref, qc_text, json_dumps_det(spine), ",".join(cluster_qi_ids))
    qc_fp = stable_id("QCFP", qc_id, qc_text, json_dumps_det(frt), json_dumps_det(triggers))

    return QCCandidate(
        qc_id=qc_id,
        qc_text=qc_text,
        ari_spine=spine,
        frt=frt,
        triggers=triggers,
        evidence_qi_ids=cluster_qi_ids,
        n_q_cluster=len(cluster_qi_ids),
        chapter_ref=chapter_ref,
        sha256=qc_fp
    )

def builder_build_qc_candidate(pack: Dict[str, Any], cluster_atom_ids: List[str], atoms_by_id: Dict[str, Atom], traces_by_qi: Dict[str, TraceARI]) -> QCCandidate:
    """IA1 BUILDER (cluster-only, n_q_cluster>=2).
    Uses GPT-4* if enabled; falls back to deterministic heuristic otherwise.
    """
    policy = pack_get(pack, "policy", {}) or {}
    use_llm = bool(policy.get("use_llm", True)) and bool(OPENAI_API_KEY)

    if (not use_llm) or (len(cluster_atom_ids) < 2):
        return builder_build_qc_candidate_heuristic(pack, cluster_atom_ids, atoms_by_id, traces_by_qi)

    # Build a compact cluster payload (CAS 1 ONLY: only Qi/RQi + traces already extracted)
    cluster_items = []
    for aid in cluster_atom_ids:
        a = atoms_by_id[aid]
        t = traces_by_qi.get(a.qi_id)
        cluster_items.append({
            "qi_id": a.qi_id,
            "qi_clean": a.qi_clean,
            "rqi_clean": a.rqi_clean,
            "trace_actions": (t.actions if t else []),
            "output_type": (t.output_type if t else "UNKNOWN"),
        })

    # Select a champion (most complete RQi by length, deterministic)
    champion = max(cluster_items, key=lambda x: len(x.get("rqi_clean","") or ""))
    champion_id = champion["qi_id"]

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "qc": {"type": "string"},
            "ari_spine": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "order": {"type": "integer", "minimum": 1, "maximum": 50},
                        "verb_id": {"type": "string"},
                        "obj": {"type": "string"},
                    },
                    "required": ["order", "verb_id", "obj"],
                },
            },
            "frt": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "usage": {"type": "object"},
                    "reponse_type": {"type": "object"},
                    "pieges": {"type": "object"},
                    "conclusion": {"type": "object"},
                },
                "required": ["usage", "reponse_type", "pieges", "conclusion"],
            },
            "triggers": {
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
            "evidence_support_qi_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        },
        "required": ["qc", "ari_spine", "frt", "triggers", "evidence_support_qi_ids"],
    }

    system = (
        "Tu es IA1 BUILDER SMAXIA V10.6.2 (CLUSTER ONLY, CAS 1 ONLY). "
        "Tu reçois un cluster de Qi/RQi POSABLE + leurs traces ARI normalisées. "
        "Tu dois produire: (1) QC forme 'Comment ... ?' (sans constantes locales), "
        "(2) ARI spine commun (verb_id+obj), (3) FRT 4 blocs, (4) 3-5 triggers génériques, "
        "(5) evidence_support_qi_ids subset du cluster. "
        "INTERDIT: inventer une méthode absente des RQi. INTERDIT: années, chiffres, noms propres."
    )

    user = json.dumps({
        "chapter_ref": atoms_by_id[cluster_atom_ids[0]].chapter_ref,
        "cluster_size": len(cluster_items),
        "qi_champion_id": champion_id,
        "items": cluster_items,
    }, ensure_ascii=False)

    obj = openai_json(
        model=DEFAULT_GPT4_MODEL,
        system=system,
        user=user,
        json_schema=schema,
        use_web_search=False,
    )

    # Post-validate evidence ids to be within cluster
    cluster_qi_ids = {atoms_by_id[aid].qi_id for aid in cluster_atom_ids}
    evid = [x for x in (obj.get("evidence_support_qi_ids") or []) if x in cluster_qi_ids]
    if len(evid) < 2:
        # Force FAIL later via IA2 by setting singleton
        evid = list(cluster_qi_ids)[:2]

    qc_text = str(obj.get("qc", "")).strip()
    ari_spine = sorted(obj.get("ari_spine") or [], key=lambda x: int(x.get("order", 9999)))
    frt = obj.get("frt") or {"usage": {}, "reponse_type": {}, "pieges": {}, "conclusion": {}}
    triggers = obj.get("triggers") or []

    qc_id = f"QC_{_sha256_text(qc_text)[:12]}"

    return QCCandidate(
        qc_id=qc_id,
        qc_text=qc_text,
        ari_spine=[{"verb_id": str(s.get("verb_id","")).strip(), "obj": str(s.get("obj","")).strip(), "order": int(s.get("order", 1))} for s in ari_spine],
        frt=frt,
        triggers=[{"pattern": str(t.get("pattern","")).strip()} for t in triggers if str(t.get("pattern","")).strip()],
        evidence_qi_ids=evid,
        n_q_cluster=len(cluster_qi_ids),
    )

def ia1_builder_batch(pack: Dict[str, Any], chapter_ref: str, clusters: List[List[str]], atoms: List[Atom], traces: List[TraceARI]) -> Tuple[List[QCCandidate], Dict[str, Any]]:
    atoms_by_qi = {a.qi_id: a for a in atoms}
    traces_by_qi = {t.qi_id: t for t in traces}
    cands: List[QCCandidate] = []
    for members in clusters:
        if len(members) < 2:
            continue
        if any(qid not in atoms_by_qi or qid not in traces_by_qi for qid in members):
            continue
        qc = builder_build_qc_candidate(pack, chapter_ref, members, atoms_by_qi, traces_by_qi)
        cands.append(qc)
    cands = sorted(cands, key=lambda q: (q.chapter_ref, q.qc_id))
    return cands, {"status": "OK", "qc_candidates": len(cands)}


# =============================================================================
# Phase 7 — IA2 Judge (Python boolean checks only)
# =============================================================================

def ia2_judge(pack: Dict[str, Any], qc: QCCandidate, traces_by_qi: Dict[str, TraceARI]) -> AuditIA2:
    fixes: List[str] = []

    def QC_FORM() -> bool:
        ok = qc.qc_text.startswith("Comment ") and qc.qc_text.rstrip().endswith("?")
        if not ok:
            fixes.append("QC_FORM: QC doit commencer par 'Comment ' et finir par '?'.")
        return ok

    def NO_LOCAL_CONSTANTS() -> bool:
        ok = not re.search(r"\b(19|20)\d{2}\b|\b\d{3,}\b", qc.qc_text) and not any(re.search(r"\b(19|20)\d{2}\b|\b\d{3,}\b", t.get("pattern","")) for t in qc.triggers)
        if not ok:
            fixes.append("NO_LOCAL_CONSTANTS: supprimer années / constantes / chiffres longs dans QC/Triggers.")
        return ok

    def N_Q_CLUSTER_GE2() -> bool:
        ok = qc.n_q_cluster >= 2 and len(qc.evidence_qi_ids) >= 2
        if not ok:
            fixes.append("N_Q_CLUSTER_GE2: cluster doit contenir au moins 2 Qi (anti-singleton).")
        return ok

    def ARI_TYPED_ONLY() -> bool:
        ok = all(isinstance(s, dict) and "verb_id" in s and str(s["verb_id"]).strip() for s in qc.ari_spine)
        if not ok:
            fixes.append("ARI_TYPED_ONLY: chaque step ARI doit contenir verb_id.")
        return ok

    def TRIGGERS_MIN3() -> bool:
        ok = isinstance(qc.triggers, list) and len(qc.triggers) >= 3
        if not ok:
            fixes.append("TRIGGERS_MIN3: générer au moins 3 triggers.")
        return ok

    def FRT_4BLOCS() -> bool:
        ok = isinstance(qc.frt, dict) and all(k in qc.frt for k in ["usage","reponse_type","pieges","conclusion"])
        if not ok:
            fixes.append("FRT_4BLOCS: FRT doit contenir 4 blocs: usage, reponse_type, pieges, conclusion.")
        return ok

    def FRT_SOURCE_RULE() -> bool:
        # For each ari_spine verb_id, must exist in at least one evidence trace action verb_id
        evidence = qc.evidence_qi_ids
        ok = True
        for step in qc.ari_spine:
            vid = str(step.get("verb_id") or "")
            if not vid:
                ok = False
                continue
            found = False
            for qid in evidence:
                tr = traces_by_qi.get(qid)
                if not tr:
                    continue
                if any(str(a.get("verb_id") or "") == vid for a in tr.actions):
                    found = True
                    break
            if not found:
                ok = False
                fixes.append(f"FRT_SOURCE_RULE: verb_id '{vid}' non supporté par les RQi du cluster.")
        return ok

    checks = {
        "QC_FORM": QC_FORM(),
        "NO_LOCAL_CONSTANTS": NO_LOCAL_CONSTANTS(),
        "N_Q_CLUSTER_GE2": N_Q_CLUSTER_GE2(),
        "ARI_TYPED_ONLY": ARI_TYPED_ONLY(),
        "TRIGGERS_MIN3": TRIGGERS_MIN3(),
        "FRT_4BLOCS": FRT_4BLOCS(),
        "FRT_SOURCE_RULE": FRT_SOURCE_RULE(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return AuditIA2(qc_id=qc.qc_id, checks=checks, status=status, fix_recommendations=fixes)

def ia2_judge_batch(pack: Dict[str, Any], qcs: List[QCCandidate], traces: List[TraceARI]) -> Tuple[List[QCCandidate], List[AuditIA2], Dict[str, Any]]:
    traces_by_qi = {t.qi_id: t for t in traces}
    audits: List[AuditIA2] = []
    passed: List[QCCandidate] = []
    for qc in qcs:
        a = ia2_judge(pack, qc, traces_by_qi)
        audits.append(a)
        if a.status == "PASS":
            passed.append(qc)
    return passed, audits, {"status": "OK", "passed": len(passed), "failed": len(audits) - len(passed)}


# =============================================================================
# Phase 8 — F1/F2 scoring (deterministic; pack-driven delta_c)
# =============================================================================

def f1_raw(qc: QCCandidate, pack: Dict[str, Any]) -> float:
    policy = pack_get(pack, "policy", {}) or {}
    delta_c = float(policy.get("delta_c") or 1.0)  # pack-driven, default 1.0
    epsilon = 0.1  # Kernel scellé (constant)
    t_j = 0.0
    for step in qc.ari_spine:
        try:
            t_j += float(step.get("weight") or 1.0)
        except Exception:
            t_j += 1.0
    return delta_c * (epsilon + t_j) ** 2

def f2_score(qc: QCCandidate, selected_qcs: List[QCCandidate], pack: Dict[str, Any]) -> float:
    # Kernel 2.2 placeholder deterministic (no external history by default)
    # pack.policy can provide novelty_weight to penalize overlap
    policy = pack_get(pack, "policy", {}) or {}
    novelty_w = float(policy.get("novelty_weight") or 1.0)
    # Overlap penalty: Jaccard of verb_ids vs already selected
    vids_q = {str(s.get("verb_id") or "") for s in qc.ari_spine if str(s.get("verb_id") or "")}
    if not selected_qcs:
        return 1.0
    overlaps = []
    for s in selected_qcs:
        vids_s = {str(x.get("verb_id") or "") for x in s.ari_spine if str(x.get("verb_id") or "")}
        inter = len(vids_q & vids_s)
        uni = len(vids_q | vids_s) if (vids_q or vids_s) else 1
        overlaps.append(inter / uni)
    ov = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return max(0.0, 1.0 - novelty_w * ov)

def score_qc(qc: QCCandidate, selected: List[QCCandidate], pack: Dict[str, Any], f1_max: float) -> Dict[str, float]:
    f1 = f1_raw(qc, pack)
    psi = f1 / f1_max if f1_max > 0 else 0.0
    f2 = f2_score(qc, selected, pack)
    # Combined score (deterministic)
    alpha = float(pack_get(pack, "policy.alpha") or 0.7)
    score = alpha * psi + (1.0 - alpha) * f2
    return {"f1": f1, "psi": psi, "f2": f2, "score": score}


# =============================================================================
# Phase 9 — Selection for 100% coverage (deterministic greedy)
# =============================================================================

def qi_matches_trigger(qi: str, pattern: str) -> bool:
    qi_l = qi.lower()
    pat_l = pattern.lower()
    # Conservative match: substring
    return pat_l in qi_l

def attach_coverage(atoms_by_qi: Dict[str, Atom], qcs: List[QCCandidate]) -> Dict[str, List[str]]:
    """Return mapping qi_id -> list of qc_ids matched by triggers."""
    cov: Dict[str, List[str]] = {qid: [] for qid in atoms_by_qi.keys()}
    for qc in qcs:
        pats = [t.get("pattern","") for t in qc.triggers if isinstance(t, dict) and t.get("pattern")]
        for qid, a in atoms_by_qi.items():
            if any(qi_matches_trigger(a.qi_clean, p) for p in pats):
                cov[qid].append(qc.qc_id)
    # deterministic order
    for qid in list(cov.keys()):
        cov[qid] = stable_sort(cov[qid])
    return cov

def selection_coverage_100(
    pack: Dict[str, Any],
    atoms: List[Atom],
    qc_valides: List[QCCandidate]
) -> Tuple[List[QCCandidate], Dict[str, Any]]:
    atoms_by_qi = {a.qi_id: a for a in atoms}
    all_qi = stable_sort(list(atoms_by_qi.keys()))

    # Precompute max f1
    f1s = [f1_raw(q, pack) for q in qc_valides]
    f1_max = max(f1s) if f1s else 1.0

    selected: List[QCCandidate] = []
    uncovered = set(all_qi)

    # Greedy selection: maximize newly covered count; tie-break by score
    while uncovered:
        best = None
        best_gain = -1
        best_score = -1.0
        for qc in qc_valides:
            if qc in selected:
                continue
            pats = [t.get("pattern","") for t in qc.triggers if isinstance(t, dict) and t.get("pattern")]
            covered = set()
            for qid in uncovered:
                if any(qi_matches_trigger(atoms_by_qi[qid].qi_clean, p) for p in pats):
                    covered.add(qid)
            gain = len(covered)
            if gain == 0:
                continue
            sc = score_qc(qc, selected, pack, f1_max)["score"]
            if gain > best_gain or (gain == best_gain and sc > best_score) or (gain == best_gain and abs(sc - best_score) < 1e-12 and qc.qc_id < (best.qc_id if best else "ZZZ")):
                best = qc
                best_gain = gain
                best_score = sc
        if not best:
            break
        selected.append(best)
        # remove covered
        pats = [t.get("pattern","") for t in best.triggers if isinstance(t, dict) and t.get("pattern")]
        newly = set([qid for qid in uncovered if any(qi_matches_trigger(atoms_by_qi[qid].qi_clean, p) for p in pats)])
        uncovered -= newly

        # Deterministic stop condition if selection is exploding: pack policy
        max_qc = int(pack_get(pack, "policy.selection.max_qc_per_chapter") or 9999)
        if len(selected) >= max_qc:
            break

    coverage = 1.0 - (len(uncovered) / len(all_qi) if all_qi else 1.0)
    return selected, {
        "coverage": round(coverage * 100.0, 2),
        "orphans": len(uncovered),
        "orphans_qi_ids": sorted(list(uncovered)),
        "selected": len(selected),
        "total_posable": len(all_qi)
    }


# =============================================================================
# Phase 10 — Verdict GTE (SEALED or FAIL)
# =============================================================================

def verdict_gte(selection_metrics: Dict[str, Any]) -> Dict[str, Any]:
    sealed = (selection_metrics.get("orphans", 1) == 0) and (selection_metrics.get("total_posable", 0) > 0)
    status = "SEALED" if sealed else "FAIL"
    return {"status": status, "sealed": sealed, "coverage_pct": selection_metrics.get("coverage", 0.0), "orphans": selection_metrics.get("orphans", 0)}


# =============================================================================
# Streamlit UI
# =============================================================================

def ui_title():
    st.markdown("# SMAXIA GTE Console V32.00.00")
    st.caption("ISO-PROD | Pack-Driven | CAS 1 ONLY | Determinisme | IA2 booléen | Test=Prod (Kernel V10.6.2)")

def ui_sidebar_pack() -> Tuple[Dict[str, Any], str]:
    st.sidebar.header("1. ACTIVATION PACK (AUTO ou JSON)")

    mode = st.sidebar.radio("Pack source", ["AUTO (sans JSON)", "Upload JSON"], index=0)

    if mode == "Upload JSON":
        pack_bytes = st.sidebar.file_uploader("Upload Pack JSON", type=["json"])
        if pack_bytes:
            try:
                pack = json.loads(pack_bytes.read().decode("utf-8"))
                return pack, "UPLOAD_JSON"
            except Exception as e:
                st.sidebar.error(f"Pack JSON invalide: {e}")
        st.sidebar.info("Aucun pack chargé. Passez en mode AUTO ou uploadez un JSON.")
        return {}, "NONE"

    # AUTO mode: only user inputs = country / subject / chapter
    country = st.sidebar.text_input("Pays", value=st.session_state.get("auto_country", "Côte d'Ivoire"))
    subject = st.sidebar.text_input("Matière", value=st.session_state.get("auto_subject", "Mathématiques"))
    chapter = st.sidebar.text_input("Chapitre", value=st.session_state.get("auto_chapter", "Fonctions"))

    st.session_state["auto_country"] = country
    st.session_state["auto_subject"] = subject
    st.session_state["auto_chapter"] = chapter

    use_auto = st.sidebar.button("Générer le Pack (AUTO)", type="primary", use_container_width=True)

    if use_auto:
        if not OPENAI_API_KEY:
            st.sidebar.error("OPENAI_API_KEY manquant (nécessaire pour AUTO).")
            return {}, "NONE"
        with st.sidebar.status("Auto-pack (GPT-4) en cours...", expanded=False) as status:
            try:
                pack = autopack_generate(country=country, subject=subject, chapter=chapter, model=DEFAULT_GPT4_MODEL)
                st.session_state["auto_pack"] = pack
                status.update(label="Auto-pack prêt", state="complete")
            except Exception as e:
                status.update(label="Auto-pack FAIL", state="error")
                st.sidebar.error(str(e))

    pack = st.session_state.get("auto_pack", {})
    if pack:
        st.sidebar.success(f"Pack auto-actif: {pack.get('version','(version?)')}")
        sig = pack.get("packsig", "")
        if sig:
            st.sidebar.caption(f"PackSig: {sig[:12]}…")
        return pack, "AUTO"

    st.sidebar.info("Cliquez sur « Générer le Pack (AUTO) » pour démarrer.")
    return {}, "NONE"

def ui_sidebar_inputs(pack: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.header("2) Inputs de test")
    st.sidebar.caption("Optionnel: injecter un couple Sujet/Corrigé via upload PDF ou URL. Sinon: Harvest pack-driven.")

    subject_pdf = st.sidebar.file_uploader("Upload Subject PDF (optionnel)", type=["pdf"], key="sub_pdf")
    correction_pdf = st.sidebar.file_uploader("Upload Correction PDF (optionnel)", type=["pdf"], key="cor_pdf")

    subject_url = st.sidebar.text_input("Subject URL (PDF) (optionnel)")
    correction_url = st.sidebar.text_input("Correction URL (PDF) (optionnel)")

    max_pairs = st.sidebar.number_input("Max pairs (Harvest)", min_value=1, max_value=200, value=30, step=1)

    chapters = iter_chapters(pack)
    chap_codes = [chapter_code(c) for c in chapters]
    chap_choice = st.sidebar.selectbox("Test Chapitre (chapter_code)", options=chap_codes if chap_codes else ["(aucun)"])

    return {
        "subject_pdf": subject_pdf,
        "correction_pdf": correction_pdf,
        "subject_url": subject_url.strip(),
        "correction_url": correction_url.strip(),
        "max_pairs": int(max_pairs),
        "chapter_code": chap_choice if chap_choice != "(aucun)" else None,
    }

def main():
    st.set_page_config(page_title="SMAXIA GTE V32.00.00", layout="wide")
    ui_title()
    pack, pack_err = ui_sidebar_pack()
    if not pack or pack_err:
        st.info("Chargez un Pack JSON pour activer le pipeline.")
        return

    inputs = ui_sidebar_inputs(pack)
    chapter_filter = inputs.get("chapter_code")

    # Validate chapter exists
    if chapter_filter and chapter_filter not in [chapter_code(c) for c in iter_chapters(pack)]:
        st.error("chapter_code sélectionné introuvable dans le pack.")
        return

    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("TEST CHAPITRE")
        run_btn = st.button("TEST CHAPITRE", type="primary")
    with colB:
        st.subheader("Diagnostics Pack")
        st.json({
            "country": pack_country(pack),
            "version": pack_version(pack),
            "chapters_count": len(iter_chapters(pack)),
            "policy_keys": sorted(list((pack.get("policy") or {}).keys())),
        })

    if not run_btn:
        st.stop()

    progress = st.progress(0, text="Phase 0/10 — Ready")
    phase_log = []

    # Deterministic run_id derived from pack + inputs refs
    pack_sig = _sha256_bytes(json_dumps_det(pack).encode("utf-8"))
    inp_fps = []
    if inputs["subject_pdf"] and inputs["correction_pdf"]:
        inp_fps.append(_sha256_bytes(inputs["subject_pdf"].getvalue()))
        inp_fps.append(_sha256_bytes(inputs["correction_pdf"].getvalue()))
    if inputs["subject_url"] and inputs["correction_url"]:
        inp_fps.append(stable_id("URLPAIR", inputs["subject_url"], inputs["correction_url"]))
    run_id = stable_id("RUN", pack_sig, "|".join(sorted(inp_fps)), chapter_filter or "ALL")

    # -------------------------
    # Phase 1: Harvest
    # -------------------------
    progress.progress(0.1, text="Phase 1/10 — HARVEST")
    pairs: List[Pair] = []
    harvest_report = {"status": "SKIPPED"}
    # Manual injection takes precedence if fully provided
    manual_pairs: List[Pair] = []
    manual_subject_bytes = None
    manual_corr_bytes = None

    if inputs["subject_pdf"] and inputs["correction_pdf"]:
        manual_subject_bytes = inputs["subject_pdf"].getvalue()
        manual_corr_bytes = inputs["correction_pdf"].getvalue()
        fp = stable_id("PAIRFP", _sha256_bytes(manual_subject_bytes), _sha256_bytes(manual_corr_bytes), "MANUAL_UPLOAD", run_id)
        manual_pairs.append(Pair("UPLOAD:SUBJECT", "UPLOAD:CORRECTION", None, "MANUAL", fp))
    elif inputs["subject_url"] and inputs["correction_url"]:
        sb, se = fetch_url_bytes(inputs["subject_url"])
        cb, ce = fetch_url_bytes(inputs["correction_url"])
        if sb and cb:
            manual_subject_bytes = sb
            manual_corr_bytes = cb
            fp = stable_id("PAIRFP", inputs["subject_url"], inputs["correction_url"], "MANUAL_URL", run_id)
            manual_pairs.append(Pair(inputs["subject_url"], inputs["correction_url"], None, "MANUAL", fp))
        else:
            harvest_report = {"status": "MANUAL_URL_FETCH_FAIL", "subject_err": se, "correction_err": ce}

    if manual_pairs:
        pairs = manual_pairs
        harvest_report = {"status": "OK", "mode": "MANUAL", "pairs": len(pairs)}
    else:
        pairs, harvest_report = harvest_pairs(pack, max_pairs=inputs["max_pairs"])

    phase_log.append({"phase": 1, "harvest_report": harvest_report, "pairs": len(pairs)})

    # -------------------------
    # Phase 2: Atomisation
    # -------------------------
    progress.progress(0.2, text="Phase 2/10 — ATOMISATION")
    atoms_all: List[Atom] = []
    quarantine_pairs: List[Dict[str, Any]] = []

    for pr in pairs:
        if pr.source == "MANUAL" and manual_subject_bytes and manual_corr_bytes:
            sb, cb = manual_subject_bytes, manual_corr_bytes
        else:
            sb, e1 = fetch_url_bytes(pr.subject_ref)
            cb, e2 = fetch_url_bytes(pr.correction_ref)
            if not sb or not cb:
                quarantine_pairs.append({"pair_sha256": pr.sha256, "reason": "FETCH_FAIL", "subject_err": e1, "correction_err": e2})
                continue
        atoms, rep = build_atoms_from_pair(pack, sb, cb, pr, chapter_filter=chapter_filter)
        if rep.get("status") != "OK":
            quarantine_pairs.append({"pair_sha256": pr.sha256, "reason": rep.get("reason", "ATOM_FAIL"), "details": rep})
            continue
        atoms_all.extend(atoms)

    atoms_all = sorted(atoms_all, key=lambda a: (a.chapter_ref, a.qi_id))
    phase_log.append({"phase": 2, "atoms": len(atoms_all), "quarantine_pairs": len(quarantine_pairs)})
    if not atoms_all:
        st.error("FAIL: Aucun Atom exploitable (vérifier PDFs, pairing, ou chapter matchers).")
        st.json({"run_id": run_id, "phase_log": phase_log, "quarantine_pairs": quarantine_pairs[:20]})
        return

    # -------------------------
    # Phase 3: POSABLE
    # -------------------------
    progress.progress(0.3, text="Phase 3/10 — POSABLE_GATE")
    pos_decisions = posable_gate(pack, atoms_all, chapter_filter=chapter_filter)
    posables = [d.atom for d in pos_decisions if d.posable]
    non_posables = [asdict(d.atom) | {"reason_code": d.reason_code} for d in pos_decisions if not d.posable]
    phase_log.append({"phase": 3, "posable": len(posables), "non_posable": len(non_posables)})

    if not posables:
        st.error("FAIL: Aucun POSABLE. CAS 1 ONLY => quarantaine.")
        st.json({"run_id": run_id, "phase_log": phase_log, "non_posables_sample": non_posables[:20]})
        return

    # -------------------------
    # Phase 4: IA1 Miner
    # -------------------------
    progress.progress(0.4, text="Phase 4/10 — IA1_MINER")
    traces, rep4 = ia1_miner_batch(pack, posables)
    phase_log.append({"phase": 4, **rep4})

    # -------------------------
    # Phase 5: Clustering
    # -------------------------
    progress.progress(0.5, text="Phase 5/10 — CLUSTERING")
    clusters, rep5 = build_clusters(pack, posables, traces)
    phase_log.append({"phase": 5, **rep5})

    if not clusters:
        st.error("FAIL: Aucun cluster valide (anti-singleton + cohérence).")
        st.json({"run_id": run_id, "phase_log": phase_log})
        return

    # -------------------------
    # Phase 6: IA1 Builder
    # -------------------------
    progress.progress(0.6, text="Phase 6/10 — IA1_BUILDER")
    qc_candidates, rep6 = ia1_builder_batch(pack, chapter_filter or "ALL", clusters, posables, traces)
    phase_log.append({"phase": 6, **rep6})

    if not qc_candidates:
        st.error("FAIL: 0 QC candidates (vérifier verb_synonyms, triggers, clustering).")
        st.json({"run_id": run_id, "phase_log": phase_log})
        return

    # -------------------------
    # Phase 7: IA2 Judge
    # -------------------------
    progress.progress(0.7, text="Phase 7/10 — IA2_JUDGE")
    qc_valides, audits, rep7 = ia2_judge_batch(pack, qc_candidates, traces)
    phase_log.append({"phase": 7, **rep7})

    # -------------------------
    # Phase 8: F1/F2 scoring
    # -------------------------
    progress.progress(0.8, text="Phase 8/10 — F1_F2_SCORING")
    # Precompute scores (deterministic)
    f1s = [f1_raw(q, pack) for q in qc_valides]
    f1_max = max(f1s) if f1s else 1.0
    scored = []
    for q in qc_valides:
        scored.append({"qc_id": q.qc_id, **score_qc(q, [], pack, f1_max)})
    scored = sorted(scored, key=lambda x: (-x["score"], x["qc_id"]))
    phase_log.append({"phase": 8, "qc_valides": len(qc_valides), "f1_max": f1_max})

    # -------------------------
    # Phase 9: Selection coverage
    # -------------------------
    progress.progress(0.9, text="Phase 9/10 — SELECTION_COVERAGE")
    qc_selected, sel_metrics = selection_coverage_100(pack, posables, qc_valides)
    phase_log.append({"phase": 9, **sel_metrics})

    # -------------------------
    # Phase 10: Verdict
    # -------------------------
    progress.progress(1.0, text="Phase 10/10 — VERDICT_GTE")
    verdict = verdict_gte(sel_metrics)
    phase_log.append({"phase": 10, **verdict})

    # =============================================================================
    # Display results
    # =============================================================================

    st.subheader("Résultats GTE")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("N_total_posable", sel_metrics.get("total_posable", 0))
    m2.metric("QC validées (IA2 PASS)", len(qc_valides))
    m3.metric("QC sélectionnées", len(qc_selected))
    m4.metric("Coverage", f"{sel_metrics.get('coverage', 0.0)}%")
    m5.metric("Orphelins", sel_metrics.get("orphans", 0))

    if verdict["sealed"]:
        st.success("SEALED: coverage=100% (zéro orphelin).")
    else:
        st.error("FAIL: coverage < 100% ou aucun POSABLE. Audit scellé disponible.")
        if sel_metrics.get("orphans_qi_ids"):
            st.caption("Exemples d'orphelins (Qi IDs): " + ", ".join(sel_metrics["orphans_qi_ids"][:10]))

    st.divider()
    st.subheader("Pipeline Log (10 phases)")
    st.json({"run_id": run_id, "phase_log": phase_log})

    st.divider()
    st.subheader("Tableau QC (sélection)")
    atoms_by_qi = {a.qi_id: a for a in posables}
    cov_map = attach_coverage(atoms_by_qi, qc_selected)

    rows = []
    for qc in qc_selected:
        # count covered qi
        covered = [qid for qid, qcs in cov_map.items() if qc.qc_id in qcs]
        rows.append({
            "qc_id": qc.qc_id,
            "qc": qc.qc_text,
            "n_q_cluster": qc.n_q_cluster,
            "covered_qi": len(covered),
            "chapter_ref": qc.chapter_ref,
            "ari_spine": qc.ari_spine,
            "triggers": [t.get("pattern") for t in qc.triggers],
            "frt": qc.frt,
            "evidence_qi_ids": qc.evidence_qi_ids,
        })
    st.dataframe(rows, use_container_width=True)

    st.divider()
    st.subheader("Audit IA2 (PASS/FAIL)")
    audit_rows = [{"qc_id": a.qc_id, "status": a.status, "checks": a.checks, "fix_recommendations": a.fix_recommendations} for a in audits]
    st.dataframe(audit_rows, use_container_width=True)

    st.divider()
    st.subheader("Exports (JSON scellés)")
    export = {
        "run_id": run_id,
        "pack_sig": pack_sig,
        "chapter_filter": chapter_filter,
        "verdict": verdict,
        "metrics": sel_metrics,
        "harvest_report": harvest_report,
        "quarantine_pairs": quarantine_pairs,
        "non_posables": non_posables[:500],
        "atoms_posable": [asdict(a) for a in posables],
        "traces": [asdict(t) for t in traces],
        "qc_candidates": [asdict(q) for q in qc_candidates],
        "qc_valides": [asdict(q) for q in qc_valides],
        "qc_selected": [asdict(q) for q in qc_selected],
        "coverage_map": cov_map,
        "ia2_audits": [asdict(a) for a in audits],
        "phase_log": phase_log,
    }
    export_json = json_dumps_det(export).encode("utf-8")

    st.download_button(
        label="Download EXPORT_GTE.json",
        data=export_json,
        file_name=f"SMAXIA_GTE_EXPORT_{run_id}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
