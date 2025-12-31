# =============================================================================
# SMAXIA GTE Console V31.10.0 (ISO-PROD — UI ultra-essentiel + Harvest corrigés renforcé)
# =============================================================================
#
# OBJECTIF ISO-PROD (TEST) — CHAÎNE COMPLÈTE, UI LISIBLE, PREUVES EXPORTABLES :
# [1] Activation pays -> load_academic_pack("FR") (invisible)
# [2] Pack visible -> UI affiche Niveaux/Matières/Chapitres (pack-driven)
# [3] Sélection Niveau + 1-2 Matières
# [4] Harvest AUTO (pêche aveugle) depuis Pack.harvest_config.sources (progress visible)
# [5] Extraction Qi/RQi (structure-driven) -> intent_code (TEST harness)
# [6] Pipeline invariant : (ARI+FRT+TRG templates) -> QC
# [7] Mapping QC -> Chapitre via Pack.chapters.intent_allowlist (pack-driven)
# [8] VALIDATEURS B BLOQUANTS + chapter_report.json (preuve par chapitre)
# [9] Saturation : arrêt si new_proof_sig == 0 ET validateurs B PASS (et POSABLE>0, QC>0)
# [10] Explorateur QC->ARI->FRT->TRG->Qi + Exports
#
# RÈGLES KERNEL APPLIQUÉES :
# - Zéro hardcode métier/langue dans ARI/FRT/TRG (templates invariants)
# - Pas de hardcode chapitre/matière : tout vient du Pack (données)
# - Preuve par chapitre obligatoire (coverage 100% sur Qi POSABLES)
# - Orphelins POSABLES interdits (bloquant)
# - Anti-faux SEALED : interdiction de sceller si POSABLE=0 ou QC=0
#
# NOTE TEST HARNESS : intent_code dérivé de EXn-Qm (structure locale sujet).
# =============================================================================

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# =============================================================================
# ZONE A — PACK LOADER (SIMULATION ISO-PROD)
# =============================================================================

def load_academic_pack(country_code: str) -> Optional[Dict[str, Any]]:
    """
    ISO-PROD TEST: pack embarqué. En PROD: API/DB/Fichier.
    AUCUN hardcode métier dans le CORE: tout ce qui est spécifique doit vivre ici (données pack).
    """
    PACKS: Dict[str, Dict[str, Any]] = {
        "FR": {
            "pack_id": "CAP_FR_BAC_2024_V1",
            "country_code": "FR",
            "country_label": "France",
            "version": "1.0.0",
            "signature": {"hash": "sha256:TEST_ONLY", "signed_at": "TEST_ONLY"},
            "academic_structure": {
                "levels": [
                    {"id": "SECONDE", "label": "Seconde", "order": 1},
                    {"id": "PREMIERE", "label": "Première", "order": 2},
                    {"id": "TERMINALE", "label": "Terminale", "order": 3},
                    {"id": "L1", "label": "Licence 1", "order": 4},
                    {"id": "L2", "label": "Licence 2", "order": 5},
                    {"id": "L3", "label": "Licence 3", "order": 6},
                    {"id": "PREPA", "label": "Prépa (CPGE)", "order": 9},
                ],
                "subjects": [
                    {"id": "MATH", "label": "Mathématiques"},
                    {"id": "PHYS", "label": "Physique-Chimie"},
                    {"id": "SVT", "label": "SVT"},
                    {"id": "NSI", "label": "NSI"},
                ],
                "level_subjects": {
                    "TERMINALE": ["MATH", "PHYS", "SVT", "NSI"],
                    "PREMIERE": ["MATH", "PHYS", "SVT", "NSI"],
                    "SECONDE": ["MATH", "PHYS"],
                    "L1": ["MATH", "PHYS"],
                    "PREPA": ["MATH", "PHYS"],
                },
            },

            # CHAPTERS pack-driven (TEST) : mapping via intent_allowlist
            "chapters": [
                {
                    "chapter_code": "CH_ANALYSE",
                    "label": "Analyse - Fonctions",
                    "qc_target": 15,
                    "intent_allowlist": [
                        "STRUCT_EX1_Q1", "STRUCT_EX1_Q2", "STRUCT_EX1_Q3",
                        "STRUCT_EX1_Q4", "STRUCT_EX1_Q5",
                        "STRUCT_EX2_Q1", "STRUCT_EX2_Q2", "STRUCT_EX2_Q3",
                    ],
                },
                {
                    "chapter_code": "CH_PROBAS",
                    "label": "Probabilités - Variables aléatoires",
                    "qc_target": 15,
                    "intent_allowlist": [
                        "STRUCT_EX3_Q1", "STRUCT_EX3_Q2", "STRUCT_EX3_Q3",
                        "STRUCT_EX3_Q4", "STRUCT_EX3_Q5",
                    ],
                },
                {
                    "chapter_code": "CH_GEOMETRIE",
                    "label": "Géométrie - Nombres complexes",
                    "qc_target": 15,
                    "intent_allowlist": [
                        "STRUCT_EX4_Q1", "STRUCT_EX4_Q2", "STRUCT_EX4_Q3",
                        "STRUCT_EX4_Q4", "STRUCT_EX4_Q5",
                    ],
                },
                {
                    "chapter_code": "CH_SUITES",
                    "label": "Suites et Récurrence",
                    "qc_target": 15,
                    "intent_allowlist": [
                        "STRUCT_EX2_Q4", "STRUCT_EX2_Q5", "STRUCT_EX2_Q6",
                    ],
                },
            ],

            # Harvest config pack-driven : sources par (LEVEL|SUBJECT)
            "harvest_config": {
                "http": {
                    "user_agent": "SMAXIA-Harvester/1.0 (ISO-PROD)",
                    "timeout_sec": 25,
                    "max_pdf_mb": 25,
                    "max_links_scan": 900,

                    # V31.10.0: multi-pages + pairing fuzzy (configurable, data-driven)
                    "max_pages_scan": 8,
                    "pagination_link_regex": r"(debut_|\bdebut=|\bpage=|\bstart=|\boffset=)",
                    "fuzzy_min_ratio": 0.72,
                },
                "sources": {
                    "TERMINALE|MATH": [
                        {
                            "name": "APMEP",
                            "index_url": "https://www.apmep.fr/spip.php?rubrique387",
                            "pdf_regex": r"\.pdf($|\?)",
                            "corr_regex": r"(corrig[eé]|correction|corrige)",
                            # V31.10.0: anti-bruit (pack data) — adapte selon vos observations
                            "deny_regex": r"(bulletin|pv\d+|compte[- ]?rendu|rapport|statuts|ag|assemblee|proc[èe]s[- ]?verbal)",
                        }
                    ],
                    "TERMINALE|PHYS": [
                        {
                            "name": "Labolycee",
                            "index_url": "https://labolycee.org/",
                            "pdf_regex": r"\.pdf($|\?)",
                            "corr_regex": r"(corrig[eé]|correction|corrige)",
                            "deny_regex": r"(bulletin|pv\d+|rapport|compte[- ]?rendu)",
                        }
                    ],
                },
            },
        },

        "CI": {
            "pack_id": "CAP_CI_TEST_V1",
            "country_code": "CI",
            "country_label": "Côte d'Ivoire",
            "version": "1.0.0",
            "signature": {"hash": "sha256:TEST_ONLY", "signed_at": "TEST_ONLY"},
            "academic_structure": {
                "levels": [{"id": "TERMINALE", "label": "Terminale", "order": 1}],
                "subjects": [{"id": "MATH", "label": "Mathématiques"}],
                "level_subjects": {"TERMINALE": ["MATH"]},
            },
            "chapters": [
                {"chapter_code": "CH_ANALYSE", "label": "Analyse", "qc_target": 15, "intent_allowlist": []}
            ],
            "harvest_config": {"http": {}, "sources": {}},
        },
    }
    return PACKS.get(country_code)


# =============================================================================
# CORE HELPERS
# =============================================================================

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha8_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]

def sha10_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]

def safe_filename_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        p = urllib.parse.urlparse(url).path
        name = (p.split("/")[-1] or "").strip()
        return name or ""
    except Exception:
        return ""


# =============================================================================
# HTTP / CRAWL (multi-pages, pack-driven)
# =============================================================================

def http_get(url: str, ua: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return data, None
    except Exception as e:
        return None, f"http_get failed: {type(e).__name__}: {e}"

def extract_links_with_anchor(html: bytes, base_url: str, max_links_scan: int) -> List[Dict[str, str]]:
    """
    Retourne une liste de {url, text} depuis <a href="...">TEXT</a>
    (fall-back: href-only si parsing incomplet).
    """
    text = html.decode("utf-8", errors="ignore")
    items: List[Dict[str, str]] = []

    # 1) a href capture (best effort)
    # capture href + inner text (simple regex, suffisant pour test harness)
    for m in re.finditer(r'<a[^>]+href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>', text, flags=re.I | re.S):
        href = (m.group(1) or "").strip()
        inner = re.sub(r"<[^>]+>", " ", (m.group(2) or ""))
        inner = re.sub(r"\s+", " ", inner).strip()
        if not href:
            continue
        url = urllib.parse.urljoin(base_url, href)
        items.append({"url": url, "text": inner})

    # 2) fallback: href only
    if not items:
        hrefs = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', text, re.I)
        hrefs = hrefs[:max_links_scan]
        for h in hrefs:
            items.append({"url": urllib.parse.urljoin(base_url, h), "text": ""})

    # dedupe stable (par url)
    out: List[Dict[str, str]] = []
    seen = set()
    for it in items[:max_links_scan]:
        u = it.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(it)
    return out

def filter_pdf_candidates(
    link_items: List[Dict[str, str]],
    pdf_regex: str,
    deny_regex: Optional[str],
) -> List[Dict[str, str]]:
    rx_pdf = re.compile(pdf_regex or r"\.pdf($|\?)", re.I)
    rx_deny = re.compile(deny_regex, re.I) if deny_regex else None

    out = []
    for it in link_items:
        u = it.get("url", "")
        t = it.get("text", "")
        if not rx_pdf.search(u):
            continue
        if rx_deny and (rx_deny.search(u) or rx_deny.search(t)):
            continue
        out.append(it)
    return out

def extract_pagination_links(
    link_items: List[Dict[str, str]],
    base_url: str,
    pagination_link_regex: str,
) -> List[str]:
    rx = re.compile(pagination_link_regex or r"(page=|start=|offset=|debut_|\bdebut=)", re.I)
    base = urllib.parse.urlparse(base_url)
    base_netloc = base.netloc

    pages = []
    for it in link_items:
        u = it.get("url", "")
        if not u:
            continue
        pu = urllib.parse.urlparse(u)
        if pu.scheme not in ("http", "https"):
            continue
        if pu.netloc != base_netloc:
            continue
        if rx.search(u):
            pages.append(u)

    # dedupe stable
    return list(dict.fromkeys(pages))

def crawl_index_for_pdfs(
    index_url: str,
    ua: str,
    timeout: int,
    max_links_scan: int,
    max_pages_scan: int,
    pagination_link_regex: str,
    pdf_regex: str,
    deny_regex: Optional[str],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Crawl multi-pages: récupère PDFs sur index + pages pagination découvertes.
    """
    diag = {
        "index_url": index_url,
        "pages_visited": 0,
        "pages_discovered": 0,
        "pdf_candidates": 0,
        "errors": [],
    }

    visited = set()
    to_visit = [index_url]
    pdfs: List[Dict[str, str]] = []

    while to_visit and diag["pages_visited"] < int(max_pages_scan):
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html, err = http_get(url, ua, timeout)
        diag["pages_visited"] += 1
        if err or not html:
            diag["errors"].append(f"crawl_page_fail:{url}:{err}")
            continue

        links = extract_links_with_anchor(html, url, max_links_scan)
        # pdf candidates on this page
        pdfs_here = filter_pdf_candidates(links, pdf_regex, deny_regex)
        pdfs.extend(pdfs_here)

        # discover pagination pages
        pages = extract_pagination_links(links, index_url, pagination_link_regex)
        for p in pages:
            if p not in visited and p not in to_visit:
                to_visit.append(p)

    diag["pages_discovered"] = len(visited)
    # dedupe pdfs by url
    seen = set()
    pdfs_unique = []
    for it in pdfs:
        u = it.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        pdfs_unique.append(it)

    diag["pdf_candidates"] = len(pdfs_unique)
    return pdfs_unique, diag


# =============================================================================
# PAIRING sujet/corrigé (fuzzy + anchor text)
# =============================================================================

def normalize_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\.pdf.*$", "", s, flags=re.I)
    s = re.sub(r"(corrig[eé]|correction|corrige)", "", s, flags=re.I)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def split_subjects_and_corrections(
    pdf_items: List[Dict[str, str]],
    corr_regex: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rx = re.compile(corr_regex, re.I) if corr_regex else None
    corrs = []
    subs = []
    for it in pdf_items:
        u = it.get("url", "")
        t = it.get("text", "")
        is_corr = False
        if rx and (rx.search(u) or rx.search(t)):
            is_corr = True
        (corrs if is_corr else subs).append(it)
    return subs, corrs

def pair_subject_correction_fuzzy(
    pdf_items: List[Dict[str, str]],
    corr_regex: str,
    fuzzy_min_ratio: float,
) -> List[Tuple[Dict[str, str], Optional[Dict[str, str]]]]:
    subs, corrs = split_subjects_and_corrections(pdf_items, corr_regex)

    # index corrections by normalized key
    corr_keys: List[Tuple[str, Dict[str, str]]] = []
    for c in corrs:
        key = normalize_key(safe_filename_from_url(c.get("url")) + " " + (c.get("text") or ""))
        corr_keys.append((key, c))

    pairs: List[Tuple[Dict[str, str], Optional[Dict[str, str]]]] = []
    for s in subs:
        skey = normalize_key(safe_filename_from_url(s.get("url")) + " " + (s.get("text") or ""))
        best = None
        best_r = 0.0
        for ckey, c in corr_keys:
            r = similarity(skey, ckey)
            if r > best_r:
                best_r = r
                best = c
        if best and best_r >= float(fuzzy_min_ratio):
            pairs.append((s, best))
        else:
            pairs.append((s, None))

    # dedupe by subject url (stable)
    out = []
    seen = set()
    for subj, corr in pairs:
        su = subj.get("url")
        if not su or su in seen:
            continue
        seen.add(su)
        out.append((subj, corr))
    return out


# =============================================================================
# HARVEST (pack-driven) + MANIFEST
# =============================================================================

def harvest_subjects(
    pack: Dict[str, Any],
    level: str,
    subject: str,
    volume: int,
    progress_cb=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = pack.get("harvest_config", {}) or {}
    http_cfg = cfg.get("http", {}) or {}

    ua = http_cfg.get("user_agent", "Mozilla/5.0")
    timeout = int(http_cfg.get("timeout_sec", 25))
    max_pdf_mb = float(http_cfg.get("max_pdf_mb", 25))
    max_links_scan = int(http_cfg.get("max_links_scan", 600))

    max_pages_scan = int(http_cfg.get("max_pages_scan", 6))
    pagination_link_regex = str(http_cfg.get("pagination_link_regex", r"(page=|start=|offset=|debut_|\bdebut=)"))
    fuzzy_min_ratio = float(http_cfg.get("fuzzy_min_ratio", 0.70))

    key = f"{level}|{subject}"
    sources = (cfg.get("sources", {}) or {}).get(key, [])

    manifest = {
        "scope_key": key,
        "sources": [],
        "selected_pairs": 0,
        "download_ok": 0,
        "download_fail": 0,
        "pairs_with_correction": 0,
        "errors": [],
        "started_at": datetime.now().isoformat(),
        "ended_at": None,
    }

    if not sources:
        manifest["errors"].append(f"no_sources_for_scope:{key}")
        manifest["ended_at"] = datetime.now().isoformat()
        return [], manifest

    # 1) crawl + collect (multi-pages)
    all_pairs: List[Tuple[Dict[str, str], Optional[Dict[str, str]]]] = []
    for src in sources:
        src_entry = {
            "name": src.get("name"),
            "index_url": src.get("index_url"),
            "pdf_candidates": 0,
            "pages_visited": 0,
            "errors": [],
        }
        idx_url = src.get("index_url")
        if not idx_url:
            src_entry["errors"].append("missing_index_url")
            manifest["sources"].append(src_entry)
            continue

        pdf_items, diag = crawl_index_for_pdfs(
            index_url=idx_url,
            ua=ua,
            timeout=timeout,
            max_links_scan=max_links_scan,
            max_pages_scan=max_pages_scan,
            pagination_link_regex=pagination_link_regex,
            pdf_regex=src.get("pdf_regex", r"\.pdf($|\?)"),
            deny_regex=src.get("deny_regex"),
        )

        src_entry["pdf_candidates"] = int(diag.get("pdf_candidates", 0))
        src_entry["pages_visited"] = int(diag.get("pages_visited", 0))
        if diag.get("errors"):
            src_entry["errors"].extend(diag["errors"])

        pairs = pair_subject_correction_fuzzy(
            pdf_items=pdf_items,
            corr_regex=src.get("corr_regex", ""),
            fuzzy_min_ratio=fuzzy_min_ratio,
        )
        all_pairs.extend(pairs)
        manifest["sources"].append(src_entry)

    # dedupe by subject url
    seen = set()
    unique = []
    for subj, corr in all_pairs:
        su = subj.get("url")
        if not su or su in seen:
            continue
        seen.add(su)
        unique.append((subj, corr))

    if not unique:
        manifest["errors"].append("no_pdf_pairs_discovered")
        manifest["ended_at"] = datetime.now().isoformat()
        return [], manifest

    # 2) pêche aveugle : échantillonnage déterministe (seed fixe)
    rnd = random.Random(42)
    if len(unique) > int(volume):
        unique = rnd.sample(unique, int(volume))

    manifest["selected_pairs"] = len(unique)

    # 3) téléchargement avec progression + contrôle taille
    items: List[Dict[str, Any]] = []
    total = len(unique)

    for i, (subj, corr) in enumerate(unique, start=1):
        subj_url = subj.get("url")
        corr_url = corr.get("url") if corr else None

        if progress_cb:
            progress_cb(i, total, subj_url or "")

        sb, err = http_get(subj_url, ua, timeout) if subj_url else (None, "missing_subject_url")
        if err or not sb:
            manifest["download_fail"] += 1
            manifest["errors"].append(f"subject_download_fail:{subj_url}:{err}")
            continue

        if (len(sb) / (1024 * 1024)) > max_pdf_mb:
            manifest["download_fail"] += 1
            manifest["errors"].append(f"subject_too_large_mb:{subj_url}:{len(sb)/(1024*1024):.1f}")
            continue

        cb = None
        corr_dl_ok = False
        if corr_url:
            cb, errc = http_get(corr_url, ua, timeout)
            if errc or not cb:
                manifest["errors"].append(f"corr_download_fail:{corr_url}:{errc}")
                cb = None
            elif (len(cb) / (1024 * 1024)) > max_pdf_mb:
                manifest["errors"].append(f"corr_too_large_mb:{corr_url}:{len(cb)/(1024*1024):.1f}")
                cb = None
            else:
                corr_dl_ok = True

        item = {
            "pair_id": f"PAIR_{key}_{i:04d}_{sha8_bytes(sb)}",
            "scope_key": key,
            "label": safe_filename_from_url(subj_url) or f"subject_{i}.pdf",
            "subject_url": subj_url,
            "corr_url": corr_url,
            "subject_bytes": sb,
            "correction_bytes": cb,
            "corr_dl_ok": corr_dl_ok,
            "corr_label": safe_filename_from_url(corr_url) if corr_url else "",
        }
        items.append(item)
        manifest["download_ok"] += 1
        if cb:
            manifest["pairs_with_correction"] += 1

    manifest["ended_at"] = datetime.now().isoformat()
    return items, manifest


# =============================================================================
# PDF -> Qi/RQi EXTRACTION (structure-driven)
# =============================================================================

def read_pdf_text(pdf_bytes: bytes) -> str:
    if not pdfplumber:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            parts = []
            for p in pdf.pages:
                try:
                    parts.append((p.extract_text() or "").replace("\u00a0", " "))
                except Exception:
                    continue
            return "\n".join(parts)
    except Exception:
        return ""

def segment_questions(text: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    current_ex = "0"
    current_marker = None
    current_lines: List[str] = []

    def flush():
        nonlocal current_marker, current_lines
        if current_marker and current_lines:
            txt = "\n".join(current_lines).strip()
            if len(re.sub(r"\s+", " ", txt)) >= 30:
                segments.append({"marker": current_marker, "text": txt, "ex": current_ex})
        current_marker = None
        current_lines = []

    for line in (text or "").splitlines():
        s = (line or "").strip()
        if not s:
            continue

        up = s.upper().replace(" ", "")
        if up.startswith("EXERCICE"):
            flush()
            m = re.search(r"(\d+)", up)
            if m:
                current_ex = m.group(1)
            continue

        m = re.match(r"^\s*(\d{1,3})\s*[\)\.\-]\s*", s)
        if m:
            flush()
            current_marker = f"EX{current_ex}-Q{m.group(1)}"
            current_lines = [line]
            continue

        m = re.match(r"^\s*([a-h])\s*[\)\.\-]\s*", s, re.I)
        if m and current_marker:
            flush()
            base = re.match(r"(EX\d+-Q\d+)", current_marker)
            if base:
                current_marker = f"{base.group(1)}{m.group(1).lower()}"
            current_lines = [line]
            continue

        if current_marker:
            current_lines.append(line)

    flush()
    return segments

def marker_to_intent(marker: str) -> str:
    m = re.match(r"EX(\d+)-Q(\d+)", (marker or "").strip(), re.I)
    if not m:
        return "STRUCT_UNK"
    return f"STRUCT_EX{m.group(1)}_Q{m.group(2)}"

def extract_qi_rqi(pairs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    qi_list: List[Dict[str, Any]] = []
    stats = {
        "pairs": len(pairs),
        "qi_total": 0,
        "qi_posable": 0,
        "aligned_rqi": 0,
        "pdfplumber": bool(pdfplumber),
        "errors": [],
    }

    for pair in pairs:
        try:
            subj_text = read_pdf_text(pair["subject_bytes"])
            corr_text = read_pdf_text(pair.get("correction_bytes") or b"")

            subj_segs = segment_questions(subj_text)
            corr_segs = segment_questions(corr_text)

            # index correction by marker (STRICT)
            corr_idx = {s["marker"]: s["text"] for s in corr_segs if s.get("marker")}

            for seg in subj_segs:
                marker = seg["marker"]
                intent_code = marker_to_intent(marker)

                rqi = corr_idx.get(marker, "")
                has_rqi = bool(rqi.strip())

                content_h = sha10_str(f"{pair.get('pair_id','')}|{marker}|{sha10_str(seg['text'])}")
                qi_id = f"QI_{sha8_bytes((pair.get('pair_id','') + marker).encode('utf-8'))}_{content_h}"

                qi_list.append({
                    "qi_id": qi_id,
                    "pair_id": pair.get("pair_id"),
                    "scope_key": pair.get("scope_key"),
                    "marker": marker,
                    "intent_code": intent_code,
                    "statement": seg["text"],
                    "rqi": rqi,
                    "has_rqi": has_rqi,
                })
        except Exception as e:
            stats["errors"].append(f"extract_failed:{pair.get('pair_id')}:{type(e).__name__}:{e}")

    stats["qi_total"] = len(qi_list)
    stats["qi_posable"] = sum(1 for q in qi_list if q.get("has_rqi"))
    stats["aligned_rqi"] = stats["qi_posable"]
    return qi_list, stats


# =============================================================================
# PIPELINE GTE — INVARIANT (ARI/FRT/TRG templates) -> QC
# =============================================================================

def cluster_by_intent(qi_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for qi in qi_list:
        if qi.get("has_rqi"):
            clusters[qi.get("intent_code", "STRUCT_UNK")].append(qi)
    return dict(clusters)

def make_ari_template(intent_code: str, proof_sig: str) -> Dict[str, Any]:
    return {
        "ari_id": f"ARI_{intent_code}_{proof_sig[:10]}",
        "intent_code": intent_code,
        "template": "ARI_INVARIANT_V1",
        "steps": [
            {"step_id": "S1", "op": "OP_READ"},
            {"step_id": "S2", "op": "OP_PLAN"},
            {"step_id": "S3", "op": "OP_EXECUTE"},
            {"step_id": "S4", "op": "OP_CHECK"},
            {"step_id": "S5", "op": "OP_CONCLUDE"},
        ],
    }

def make_frt_template(intent_code: str, proof_sig: str) -> Dict[str, Any]:
    return {
        "frt_id": f"FRT_{intent_code}_{proof_sig[:10]}",
        "intent_code": intent_code,
        "template": "FRT_INVARIANT_V1",
        "sections": {
            "given": "GIVEN",
            "goal": "GOAL",
            "method": "METHOD",
            "checks": ["CHECKS"],
            "traps": ["TRAPS"],
            "final": "FINAL",
        },
    }

def make_triggers_template(intent_code: str, proof_sig: str) -> List[Dict[str, Any]]:
    return [{
        "trigger_id": f"TRG_{intent_code}_{proof_sig[:10]}",
        "type": "TRG_INTENT_MATCH",
        "pattern": intent_code,
        "confidence": 1.0,
    }]

def proof_signature(intent_code: str, qi_ids: List[str]) -> str:
    payload = intent_code + "|" + "|".join(sorted(qi_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def build_qc(intent_code: str, cluster: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(cluster) < 2:
        return None

    qi_ids = [q["qi_id"] for q in cluster if q.get("qi_id")]
    sig = proof_signature(intent_code, qi_ids)

    qc_id_stable = f"QC_{intent_code}_{sha10_str(intent_code)}"
    ari = make_ari_template(intent_code, sig)
    frt = make_frt_template(intent_code, sig)
    trgs = make_triggers_template(intent_code, sig)

    return {
        "qc_id": qc_id_stable,
        "intent_code": intent_code,
        "qc_proof_sig": sig,
        "formulation": "Comment procéder ?",
        "n_q_cluster": len(cluster),
        "qi_ids": qi_ids,
        "ari": ari,
        "frt": frt,
        "triggers": trgs,
        "created_at": datetime.now().isoformat(),
    }

def map_qc_to_chapter(intent_code: str, chapters: List[Dict[str, Any]]) -> Optional[str]:
    for ch in chapters:
        if intent_code in (ch.get("intent_allowlist") or []):
            return ch.get("chapter_code")
    return None

def run_pipeline_gte(qi_list: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    clusters = cluster_by_intent(qi_list)

    qc_list: List[Dict[str, Any]] = []
    singletons: List[Dict[str, Any]] = []

    for intent_code, cluster in sorted(clusters.items(), key=lambda x: x[0]):
        if len(cluster) < 2:
            singletons.append({"intent_code": intent_code, "count": len(cluster)})
            continue
        qc = build_qc(intent_code, cluster)
        if qc:
            qc["chapter_code"] = map_qc_to_chapter(intent_code, chapters)
            qc_list.append(qc)

    stats = {
        "total_qi": len(qi_list),
        "qi_posable": sum(1 for q in qi_list if q.get("has_rqi")),
        "clusters": len(clusters),
        "singletons": len(singletons),
        "qc_generated": len(qc_list),
    }
    return qc_list, stats


# =============================================================================
# PROOFS + VALIDATORS B (BLOQUANTS) — V31.10.0 anti-faux SEALED
# =============================================================================

def compute_orphans_posable(qi_list: List[Dict[str, Any]], qcs: List[Dict[str, Any]]) -> List[str]:
    posable_ids = {q["qi_id"] for q in qi_list if q.get("has_rqi") and q.get("qi_id")}
    covered = set()
    for qc in qcs:
        for qid in qc.get("qi_ids", []) or []:
            covered.add(qid)
    return sorted([qid for qid in posable_ids if qid not in covered])

def compute_active_chapters(chapters: List[Dict[str, Any]], qi_list: List[Dict[str, Any]]) -> List[str]:
    intents_seen = {q.get("intent_code") for q in qi_list if q.get("has_rqi")}
    active = []
    for ch in chapters:
        allow = set(ch.get("intent_allowlist") or [])
        if allow.intersection(intents_seen):
            active.append(ch.get("chapter_code"))
    return active

def build_chapter_report(chapters: List[Dict[str, Any]], qi_list: List[Dict[str, Any]], qcs: List[Dict[str, Any]]) -> Dict[str, Any]:
    active_codes = set(compute_active_chapters(chapters, qi_list))

    qc_by_chapter = defaultdict(list)
    for qc in qcs:
        qc_by_chapter[qc.get("chapter_code") or "ORPHAN"].append(qc)

    qi_by_intent = defaultdict(list)
    for qi in qi_list:
        if qi.get("has_rqi"):
            qi_by_intent[qi.get("intent_code", "STRUCT_UNK")].append(qi.get("qi_id"))

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "active_chapters": sorted(list(active_codes)),
        "chapters": {},
    }

    for ch in chapters:
        code = ch.get("chapter_code")
        if code not in active_codes:
            continue

        allow = ch.get("intent_allowlist") or []

        chap_qi = set()
        for intent in allow:
            for qid in (qi_by_intent.get(intent) or []):
                if qid:
                    chap_qi.add(qid)

        chap_cov = set()
        for qc in qc_by_chapter.get(code, []):
            for qid in (qc.get("qi_ids") or []):
                chap_cov.add(qid)

        orphans = sorted([qid for qid in chap_qi if qid not in chap_cov])
        qi_count = len(chap_qi)
        covered_count = qi_count - len(orphans)
        coverage_pct = 0.0 if qi_count == 0 else round((covered_count / qi_count) * 100.0, 2)
        qc_count = len(qc_by_chapter.get(code, []))

        report["chapters"][code] = {
            "label": ch.get("label"),
            "qc_target": ch.get("qc_target", 15),
            "qc_count": qc_count,
            "qi_posable_count": qi_count,
            "covered_qi_count": covered_count,
            "coverage_pct": coverage_pct,
            "orphans": orphans,
            "status": "PASS" if (qi_count > 0 and coverage_pct == 100.0 and qc_count > 0) else "FAIL",
        }

    return report

def validate_B_blocking(chapters: List[Dict[str, Any]], qi_list: List[Dict[str, Any]], qcs: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    qi_posable = sum(1 for q in qi_list if q.get("has_rqi"))
    qc_total = len(qcs)

    orphans_global = compute_orphans_posable(qi_list, qcs)
    chapter_report = build_chapter_report(chapters, qi_list, qcs)

    failures = []

    # B-00: anti-faux SEALED (pas de POSABLE => test invalide)
    if qi_posable == 0:
        failures.append({"code": "B-00_NO_POSABLE_QI", "msg": "Aucune Qi POSABLE: corrigés absents/non-alignés."})

    # B-00b: anti-faux SEALED (pas de QC => test invalide)
    if qc_total == 0:
        failures.append({"code": "B-00B_NO_QC", "msg": "Aucune QC générée: impossible de valider couverture."})

    # B1: orphans posables interdits
    if orphans_global:
        failures.append({"code": "B-01_ORPHANS_POSABLE", "count": len(orphans_global), "sample": orphans_global[:20]})

    # B2: preuve par chapitre actif
    for ch_code, data in (chapter_report.get("chapters") or {}).items():
        if data.get("qi_posable_count", 0) == 0:
            failures.append({"code": "B-02_NO_EVIDENCE_CHAPTER", "chapter": ch_code})
        if data.get("coverage_pct") != 100.0:
            failures.append({"code": "B-03_COVERAGE_NOT_100", "chapter": ch_code, "coverage_pct": data.get("coverage_pct")})
        if data.get("qc_count", 0) == 0:
            failures.append({"code": "B-04_NO_QC_CHAPTER", "chapter": ch_code})

    ok = (len(failures) == 0)
    details = {
        "ok": ok,
        "qi_posable": qi_posable,
        "qc_total": qc_total,
        "orphans_global": orphans_global,
        "chapter_report": chapter_report,
        "failures": failures,
    }
    return ok, details


# =============================================================================
# SATURATION — compare proof signatures (anti-faux SEALED)
# =============================================================================

def saturation_delta(prev_qcs: List[Dict[str, Any]], new_qcs: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    prev_sigs = {qc.get("qc_proof_sig") for qc in prev_qcs if qc.get("qc_proof_sig")}
    new_sigs = {qc.get("qc_proof_sig") for qc in new_qcs if qc.get("qc_proof_sig")}
    added = sorted(list(new_sigs - prev_sigs))
    return len(added), added


# =============================================================================
# STREAMLIT UI — ULTRA ESSENTIEL (ISO-PROD TEST)
# =============================================================================

def init_state():
    defaults = {
        "pack": None,
        "country": "FR",
        "level": None,
        "subjects": [],

        # Bibliothèque unique consommée par la chaîne (AUTO + MANUAL)
        "library_pairs": [],             # list[dict] with bytes
        "harvest_manifest": None,

        "qi_global": [],
        "qi_stats": None,
        "qcs_global": [],
        "gte_stats": None,

        "iterations": [],
        "sealed": False,
        "last_B": None,

        "logs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log(line: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {line}")

def sidebar_minimal(pack: Dict[str, Any]):
    st.sidebar.markdown("## ÉTAPE 1 — ACTIVATION PAYS")
    st.sidebar.caption("1 clic = load_academic_pack(country) (invisible). Pack visible = preuve.")
    st.session_state.country = st.sidebar.selectbox("Pays (TEST)", ["FR", "CI"], index=0 if st.session_state.country == "FR" else 1)

    if st.sidebar.button("🔓 ACTIVER", type="primary", use_container_width=True):
        p = load_academic_pack(st.session_state.country)
        if not p:
            st.sidebar.error("Pack introuvable.")
        else:
            st.session_state.pack = p
            st.session_state.level = None
            st.session_state.subjects = []
            st.session_state.library_pairs = []
            st.session_state.harvest_manifest = None
            st.session_state.qi_global = []
            st.session_state.qi_stats = None
            st.session_state.qcs_global = []
            st.session_state.gte_stats = None
            st.session_state.iterations = []
            st.session_state.sealed = False
            st.session_state.last_B = None
            st.session_state.logs = []
            log(f"PACK ACTIVÉ: {p.get('pack_id')}")

    st.sidebar.markdown("---")

    if st.session_state.pack:
        st.sidebar.success(f"Pack actif:\n\n{st.session_state.pack.get('pack_id')}")
        sig = st.session_state.pack.get("signature", {})
        st.sidebar.caption(f"Signature: {sig.get('hash')} / {sig.get('signed_at')}")

        acs = st.session_state.pack.get("academic_structure", {})
        levels = sorted(acs.get("levels", []), key=lambda x: x.get("order", 999))
        level_ids = [l["id"] for l in levels]
        level_labels = {l["id"]: l["label"] for l in levels}

        st.sidebar.markdown("## ÉTAPE 2 — SÉLECTION")
        st.session_state.level = st.sidebar.radio(
            "Niveau",
            level_ids,
            index=level_ids.index(st.session_state.level) if st.session_state.level in level_ids else 0,
            format_func=lambda x: level_labels.get(x, x),
        )

        subject_labels = {s["id"]: s["label"] for s in acs.get("subjects", [])}
        allowed = acs.get("level_subjects", {}).get(st.session_state.level, [])

        st.session_state.subjects = st.sidebar.multiselect(
            "Matières (1–2 recommandé pour test)",
            options=allowed,
            default=st.session_state.subjects if st.session_state.subjects else (allowed[:1] if allowed else []),
            format_func=lambda x: subject_labels.get(x, x),
        )

        st.sidebar.markdown("---")
        st.sidebar.caption("✅ Sélection:")
        st.sidebar.write(f"- Niveau: **{level_labels.get(st.session_state.level)}**")
        st.sidebar.write(f"- Matières: **{', '.join(st.session_state.subjects) if st.session_state.subjects else '—'}**")

        with st.sidebar.expander("Chapitres (pack-driven)", expanded=False):
            for ch in st.session_state.pack.get("chapters", []):
                st.write(f"- {ch.get('chapter_code')} — {ch.get('label')}")


def kpi_row():
    items = len(st.session_state.library_pairs)
    corr_ok = sum(1 for p in st.session_state.library_pairs if p.get("correction_bytes"))
    qi = len(st.session_state.qi_global)
    qi_pos = sum(1 for q in st.session_state.qi_global if q.get("has_rqi"))
    qc = len(st.session_state.qcs_global)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Items", items)
    c2.metric("Corrigés DL", corr_ok)
    c3.metric("Qi", qi)
    c4.metric("Qi POSABLE", qi_pos)
    c5.metric("QC", qc)
    c6.metric("SEALED", "YES" if st.session_state.sealed else "NO")


def main():
    st.set_page_config(page_title="SMAXIA GTE V31.10.0", layout="wide")
    init_state()

    st.title("🔐 SMAXIA GTE Console V31.10.0 — ISO-PROD TEST")
    st.caption("Flux: Activation → Pack visible → Sélection → Harvest → Extraction → RUN GTE → Saturation → Résultats")

    if st.session_state.pack is None:
        sidebar_minimal({"pack_id": ""})
    else:
        sidebar_minimal(st.session_state.pack)

    tabs = st.tabs(["📥 Import PDF", "⚡ Chaîne complète", "📊 Résultats / Exports"])

    # ---------------------------------------------------------------------
    # TAB 1 — Import PDF (bibliothèque visible)
    # ---------------------------------------------------------------------
    with tabs[0]:
        st.header("Import PDF (sujets + corrigés)")
        st.caption("Bibliothèque = Harvest AUTO + Upload manuel. La chaîne GTE consomme la bibliothèque.")
        kpi_row()

        colA, colB = st.columns([2, 1])

        with colA:
            st.subheader("Bibliothèque Harvest (visible)")
            if not st.session_state.library_pairs:
                st.info("Aucun item en bibliothèque. Lancez la chaîne complète (Harvest AUTO) ou faites un upload manuel.")
            else:
                rows = []
                for p in st.session_state.library_pairs[:300]:
                    rows.append({
                        "pair_id": p.get("pair_id"),
                        "scope": p.get("scope_key"),
                        "sujet": p.get("label"),
                        "corrigé": p.get("corr_label") if p.get("corr_label") else ("—" if not p.get("corr_url") else safe_filename_from_url(p.get("corr_url"))),
                        "corrigé?": "✅" if p.get("correction_bytes") else "❌",
                    })
                st.dataframe(rows, use_container_width=True, height=360)
                st.caption("Affichage limité à 300 lignes (export complet disponible).")

                with st.expander("Harvest manifest (diagnostic)", expanded=False):
                    st.json(st.session_state.harvest_manifest or {})

        with colB:
            st.subheader("Upload manuel (optionnel)")
            subj_files = st.file_uploader("PDF Sujet", type=["pdf"], accept_multiple_files=True, key="manual_subj")
            corr_files = st.file_uploader("PDF Correction (opt)", type=["pdf"], accept_multiple_files=True, key="manual_corr")

            if subj_files and st.button("➕ Ajouter à la bibliothèque", type="primary", use_container_width=True):
                corr_map = {}
                if corr_files:
                    for cf in corr_files:
                        corr_map[cf.name] = cf

                pairs = []
                for i, sf in enumerate(subj_files, start=1):
                    sb = sf.read()
                    sf.seek(0)
                    cb = None
                    corr_name = ""

                    if corr_files:
                        sbase = sf.name.lower()
                        best = None
                        best_r = 0.0
                        for cf in corr_files:
                            cname = cf.name.lower()
                            # fuzzy pairing local par nom
                            r = similarity(normalize_key(sbase), normalize_key(cname))
                            if r > best_r:
                                best_r = r
                                best = cf
                        if best and best_r >= 0.65:
                            cb = best.read()
                            best.seek(0)
                            corr_name = best.name

                    pairs.append({
                        "pair_id": f"MANUAL_{i:04d}_{sha8_bytes(sb)}",
                        "scope_key": "MANUAL",
                        "label": sf.name,
                        "subject_url": None,
                        "corr_url": None,
                        "subject_bytes": sb,
                        "correction_bytes": cb,
                        "corr_dl_ok": bool(cb),
                        "corr_label": corr_name,
                    })

                st.session_state.library_pairs.extend(pairs)
                st.session_state.harvest_manifest = {
                    "mode": "MANUAL",
                    "total_pairs": len(st.session_state.library_pairs),
                    "generated_at": datetime.now().isoformat(),
                }
                log(f"UPLOAD MANUEL: +{len(pairs)} items (bibliothèque={len(st.session_state.library_pairs)})")
                st.success(f"Ajouté: {len(pairs)} item(s).")

    # ---------------------------------------------------------------------
    # TAB 2 — Chaîne complète (Harvest -> Extraction -> RUN GTE -> Saturation)
    # ---------------------------------------------------------------------
    with tabs[1]:
        st.header("ÉTAPE 3 — LANCER CHAÎNE COMPLÈTE (Harvest → Extraction → RUN GTE → Saturation)")
        st.caption("ISO-PROD: preuves exportables, validateurs B bloquants, scellement seulement si new_proof_sig=0 ET B PASS (et POSABLE>0, QC>0).")

        if not st.session_state.pack:
            st.warning("Activez d'abord un pays.")
        elif not st.session_state.level or not st.session_state.subjects:
            st.warning("Sélectionnez Niveau + Matières.")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                base_volume = st.number_input("Volume initial (par matière)", min_value=5, value=30, step=5)
            with col2:
                max_iter = st.number_input("Max itérations", min_value=1, value=4, step=1)
            with col3:
                incr = st.number_input("Incrément volume", min_value=5, value=50, step=5)

            if st.button("⚡ LANCER CHAÎNE COMPLÈTE", type="primary", use_container_width=True):
                st.session_state.sealed = False
                st.session_state.iterations = []
                st.session_state.logs = []
                log("═══ DÉMARRAGE CHAÎNE V31.10.0 ═══")
                log(f"Pays: {st.session_state.country} | Niveau: {st.session_state.level} | Matières: {st.session_state.subjects}")

                current_volume = int(base_volume)
                prev_qcs = st.session_state.qcs_global[:] if st.session_state.qcs_global else []

                logbox = st.empty()
                prog = st.progress(0.0)

                def render_logs():
                    logbox.code("\n".join(st.session_state.logs[-120:]), language="text")

                for it in range(1, int(max_iter) + 1):
                    if st.session_state.sealed:
                        break

                    log(f"══ ITÉRATION {it} — Volume: {current_volume} sujets/matière ══")
                    render_logs()

                    # Harvest per subject
                    all_pairs = []
                    all_manifests = []
                    def pcb(cur, total, url):
                        prog.progress(cur / max(1, total), text=f"{cur}/{total} — {safe_filename_from_url(url)[:60]}")
                    for subj in st.session_state.subjects:
                        scope = f"{st.session_state.level}|{subj}"
                        log(f"[HARVEST] Scope: {scope} ...")
                        render_logs()
                        pairs, manifest = harvest_subjects(st.session_state.pack, st.session_state.level, subj, current_volume, pcb)
                        all_pairs.extend(pairs)
                        all_manifests.append(manifest)
                        log(f"[HARVEST] {scope}: {len(pairs)} items (corrigé DL si trouvé).")
                        render_logs()

                    if not all_pairs:
                        log("0 items récoltés. Arrêt.")
                        render_logs()
                        break

                    # Bibliothèque: merge
                    st.session_state.library_pairs.extend(all_pairs)
                    st.session_state.harvest_manifest = {
                        "country": st.session_state.country,
                        "level": st.session_state.level,
                        "subjects": st.session_state.subjects,
                        "manifests": all_manifests,
                        "total_pairs": len(st.session_state.library_pairs),
                        "generated_at": datetime.now().isoformat(),
                    }
                    log(f"[BIBLIOTHÈQUE] +{len(all_pairs)} items (total={len(st.session_state.library_pairs)}).")
                    render_logs()

                    # Extraction
                    log("[EXTRACTION] Dépessage Qi/RQi ...")
                    render_logs()
                    qi_new, qi_stats = extract_qi_rqi(all_pairs)

                    existing = {q["qi_id"] for q in st.session_state.qi_global}
                    new_only = [q for q in qi_new if q.get("qi_id") not in existing]
                    st.session_state.qi_global.extend(new_only)
                    st.session_state.qi_stats = qi_stats | {
                        "new_qi_added": len(new_only),
                        "qi_global_total": len(st.session_state.qi_global),
                    }
                    log(f"[EXTRACTION] +{len(new_only)} Qi (global={len(st.session_state.qi_global)}). POSABLE={sum(1 for q in st.session_state.qi_global if q.get('has_rqi'))}")
                    render_logs()

                    # RUN GTE
                    log("[GTE] Construction QC (ARI/FRT/TRG -> QC) ...")
                    render_logs()
                    chapters = st.session_state.pack.get("chapters", [])
                    qcs, gte_stats = run_pipeline_gte(st.session_state.qi_global, chapters)
                    st.session_state.qcs_global = qcs
                    st.session_state.gte_stats = gte_stats
                    log(f"[GTE] QC total={len(qcs)} | clusters={gte_stats.get('clusters')} | singletons={gte_stats.get('singletons')}")
                    render_logs()

                    # Validate B
                    okB, B = validate_B_blocking(chapters, st.session_state.qi_global, st.session_state.qcs_global)
                    st.session_state.last_B = B
                    if okB:
                        log("[B] PASS")
                    else:
                        log(f"[B] FAIL | {len(B.get('failures') or [])} violation(s)")
                    render_logs()

                    # Saturation delta
                    new_count, _ = saturation_delta(prev_qcs, qcs)
                    log(f"[SAT] new_proof_sig={new_count}")
                    render_logs()

                    st.session_state.iterations.append({
                        "iter": it,
                        "volume": current_volume,
                        "pairs": len(all_pairs),
                        "new_qi": len(new_only),
                        "qc_total": len(qcs),
                        "new_proof_sig": new_count,
                        "B_ok": okB,
                        "timestamp": datetime.now().isoformat(),
                    })

                    # scellage strict
                    qi_posable = B.get("qi_posable", 0)
                    qc_total = B.get("qc_total", 0)
                    if new_count == 0 and okB and qi_posable > 0 and qc_total > 0:
                        st.session_state.sealed = True
                        log("🎯 ═══ SATURATION ATTEINTE — SEALED (new_proof_sig=0 ET B PASS) ═══")
                        render_logs()
                        break
                    else:
                        if new_count == 0 and (qi_posable == 0 or qc_total == 0):
                            log("⛔ new_proof_sig=0 MAIS POSABLE=0 ou QC=0 => SEALED INTERDIT (anti-faux).")
                        elif new_count == 0 and not okB:
                            log("⛔ new_proof_sig=0 MAIS B FAIL => SEALED INTERDIT.")
                        else:
                            log("↩︎ Non scellé: augmenter volume et poursuivre.")
                        render_logs()

                    prev_qcs = qcs[:]
                    current_volume += int(incr)

                prog.empty()

            st.markdown("---")
            st.subheader("LOG EN TEMPS RÉEL")
            st.code("\n".join(st.session_state.logs[-150:]), language="text")

            st.markdown("---")
            st.subheader("Table itérations")
            if st.session_state.iterations:
                st.dataframe(st.session_state.iterations, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 3 — Résultats / Explorateur / Exports
    # ---------------------------------------------------------------------
    with tabs[2]:
        st.header("ÉTAPE 4 — RÉSULTATS / EXPLORATEUR / EXPORTS")
        kpi_row()

        if st.session_state.pack and st.session_state.qcs_global:
            chapters = st.session_state.pack.get("chapters", [])
            okB, B = validate_B_blocking(chapters, st.session_state.qi_global, st.session_state.qcs_global)
            st.session_state.last_B = B

            if okB:
                st.success("VALIDATEURS B : PASS (bloquants)")
            else:
                st.error("VALIDATEURS B : FAIL (bloquants)")
                with st.expander("Détails FAIL", expanded=True):
                    st.json(B.get("failures"))

            with st.expander("Chapter report (preuve)", expanded=False):
                st.json((B.get("chapter_report") or {}))

            st.markdown("---")
            st.subheader("Explorateur : QC → ARI → FRT → Triggers → Qi")
            by_ch = defaultdict(list)
            for qc in st.session_state.qcs_global:
                by_ch[qc.get("chapter_code") or "ORPHAN"].append(qc)

            if not by_ch:
                st.info("Aucune QC. Vérifiez corrigés (POSABLE) et extraction.")
            else:
                ch_label = {c.get("chapter_code"): c.get("label") for c in chapters}
                for ch_code in sorted(by_ch.keys()):
                    label = ch_label.get(ch_code, ch_code)
                    with st.expander(f"{ch_code} — {label} ({len(by_ch[ch_code])} QC)", expanded=False):
                        for qc in by_ch[ch_code][:30]:
                            st.markdown(f"**{qc.get('qc_id')}**")
                            st.caption(f"intent_code={qc.get('intent_code')} | proof_sig={str(qc.get('qc_proof_sig'))[:16]}…")
                            st.write(f"Formulation: {qc.get('formulation')}")
                            st.write("ARI"); st.json(qc.get("ari"))
                            st.write("FRT"); st.json(qc.get("frt"))
                            st.write("Triggers"); st.json(qc.get("triggers"))
                            st.write("Qi associées (sample)"); st.json((qc.get("qi_ids") or [])[:20])

        else:
            st.info("Exécutez la chaîne complète pour obtenir des QC.")

        st.markdown("---")
        st.subheader("Exports (preuves)")
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.download_button("qi_pack.json", canonical_json(st.session_state.qi_global), "qi_pack.json", use_container_width=True)
        with c2:
            st.download_button("qc_pack.json", canonical_json(st.session_state.qcs_global), "qc_pack.json", use_container_width=True)
        with c3:
            chapter_report = (st.session_state.last_B or {}).get("chapter_report") or {}
            st.download_button("chapter_report.json", canonical_json(chapter_report), "chapter_report.json", use_container_width=True)
        with c4:
            st.download_button("harvest_manifest.json", canonical_json(st.session_state.harvest_manifest or {}), "harvest_manifest.json", use_container_width=True)
        with c5:
            st.download_button("logs.txt", "\n".join(st.session_state.logs), "logs.txt", use_container_width=True)


if __name__ == "__main__":
    main()
