# =============================================================================
# SMAXIA GTE Console V31.10.0 (ISO-PROD — UIFLUX + Library visible + Kernel locks)
# =============================================================================
#
# FLUX VALIDÉ:
# [1] Activation pays -> load_academic_pack("FR") (invisible)
# [2] Pack visible -> UI affiche niveaux/matières/chapitres
# [3] Sélection -> Niveau + 1-2 matières
# [4] Harvest AUTO (pêche aveugle) -> bibliothèque visible (sujets + corrigés)
# [5] Extraction -> Qi/RQi
# [6] Pipeline -> ARI/FRT/TRG puis QC
# [7] Mapping QC -> chapitres (pack-driven)
# [8] Validateurs B BLOQUANTS (inclut B-00_MIN_POSABLE & B-00_MIN_QC)
# [9] Saturation -> SEALED seulement si new_proof_sig=0 ET B PASS ET QC>0 ET POSABLE>0
# [10] Résultats / Explorateur / Exports
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
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

BUILD_TAG = "V31.10.0-UIFLUX-LIBRARY-2025-12-31"


# =============================================================================
# PACK LOADER (SIMULATION ISO-PROD)
# =============================================================================

def load_academic_pack(country_code: str) -> Optional[Dict[str, Any]]:
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
            "harvest_config": {
                "http": {
                    "user_agent": "SMAXIA-Harvester/1.0 (ISO-PROD)",
                    "timeout_sec": 25,
                    "max_pdf_mb": 25,
                    "max_links_scan": 700,
                },
                "sources": {
                    "TERMINALE|MATH": [
                        {
                            "name": "APMEP",
                            "index_url": "https://www.apmep.fr/spip.php?rubrique387",
                            "pdf_regex": r"\.pdf($|\?)",
                            "corr_regex": r"corrig[eé]|correction|corrige",
                        }
                    ],
                    "TERMINALE|PHYS": [
                        {
                            "name": "Labolycee",
                            "index_url": "https://labolycee.org/",
                            "pdf_regex": r"\.pdf($|\?)",
                            "corr_regex": r"corrig[eé]|correction|corrige",
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

def now_iso() -> str:
    return datetime.now().isoformat()


# =============================================================================
# LIBRARY (bibliothèque visible) — sujets + corrigés (AUTO + MANUAL)
# =============================================================================

def library_init():
    if "library" not in st.session_state:
        st.session_state.library = []  # list[dict(pair_id, label, scope_key, subject_bytes, correction_bytes, subject_url, corr_url, added_at, source)]
    if "logs" not in st.session_state:
        st.session_state.logs = []

def log(line: str):
    st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")

def library_add_items(items: List[Dict[str, Any]], source: str):
    library_init()
    existing = {x.get("pair_id") for x in st.session_state.library if x.get("pair_id")}
    added = 0
    for it in items:
        pid = it.get("pair_id")
        if not pid or pid in existing:
            continue
        row = dict(it)
        row["added_at"] = now_iso()
        row["source"] = source
        st.session_state.library.append(row)
        existing.add(pid)
        added += 1
    return added

def library_rows(limit: int = 400) -> List[Dict[str, Any]]:
    library_init()
    rows = []
    for it in st.session_state.library[:limit]:
        sb = it.get("subject_bytes") or b""
        cb = it.get("correction_bytes") or b""
        rows.append({
            "pair_id": it.get("pair_id"),
            "scope": it.get("scope_key"),
            "source": it.get("source"),
            "sujet": it.get("label"),
            "corrigé?": "✅" if cb else "❌",
            "size_sujet_kb": int(len(sb) / 1024) if sb else 0,
            "size_corr_kb": int(len(cb) / 1024) if cb else 0,
            "added_at": (it.get("added_at") or "")[:19],
        })
    return rows

def library_counts() -> Tuple[int, int]:
    library_init()
    total = len(st.session_state.library)
    with_corr = sum(1 for it in st.session_state.library if it.get("correction_bytes"))
    return total, with_corr


# =============================================================================
# HARVEST (pack-driven, pêche aveugle) + MANIFEST
# =============================================================================

def http_get(url: str, ua: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return data, None
    except Exception as e:
        return None, f"http_get failed: {type(e).__name__}: {e}"

def extract_pdf_links(html: bytes, base_url: str, pdf_regex: str, max_links_scan: int) -> List[str]:
    text = html.decode("utf-8", errors="ignore")
    hrefs = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', text, re.I)
    hrefs = hrefs[:max_links_scan]
    rx = re.compile(pdf_regex or r"\.pdf($|\?)", re.I)
    pdfs = [urllib.parse.urljoin(base_url, h) for h in hrefs if rx.search(h)]
    return list(dict.fromkeys(pdfs))

def pair_subject_correction(pdfs: List[str], corr_regex: str) -> List[Tuple[str, Optional[str]]]:
    rx = re.compile(corr_regex, re.I) if corr_regex else None
    corrs = [p for p in pdfs if rx and rx.search(p)]
    subjs = [p for p in pdfs if (not rx) or (not rx.search(p))]

    def normalize(u: str) -> str:
        base = urllib.parse.urlparse(u).path.split("/")[-1]
        base = re.sub(r"\.pdf.*$", "", base, flags=re.I)
        if rx:
            base = rx.sub("", base)
        return re.sub(r"[^a-z0-9]", "", base.lower())

    corr_map = {normalize(c): c for c in corrs}
    return [(s, corr_map.get(normalize(s))) for s in subjs]

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
    max_links_scan = int(http_cfg.get("max_links_scan", 700))

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
        "started_at": now_iso(),
        "ended_at": None,
    }

    if not sources:
        manifest["errors"].append(f"no_sources_for_scope:{key}")
        manifest["ended_at"] = now_iso()
        return [], manifest

    all_pairs: List[Tuple[str, Optional[str]]] = []
    for src in sources:
        src_entry = {"name": src.get("name"), "index_url": src.get("index_url"), "pdfs_found": 0, "errors": []}
        idx_url = src.get("index_url")
        if not idx_url:
            src_entry["errors"].append("missing_index_url")
            manifest["sources"].append(src_entry)
            continue

        html, err = http_get(idx_url, ua, timeout)
        if err or not html:
            src_entry["errors"].append(err or "empty_html")
            manifest["sources"].append(src_entry)
            continue

        pdfs = extract_pdf_links(html, idx_url, src.get("pdf_regex", r"\.pdf($|\?)"), max_links_scan)
        src_entry["pdfs_found"] = len(pdfs)
        pairs = pair_subject_correction(pdfs, src.get("corr_regex", ""))
        all_pairs.extend(pairs)
        manifest["sources"].append(src_entry)

    seen = set()
    unique = [(s, c) for (s, c) in all_pairs if s and (s not in seen) and not seen.add(s)]
    if not unique:
        manifest["errors"].append("no_pdf_pairs_discovered")
        manifest["ended_at"] = now_iso()
        return [], manifest

    rnd = random.Random(42)
    if len(unique) > volume:
        unique = rnd.sample(unique, volume)
    manifest["selected_pairs"] = len(unique)

    items: List[Dict[str, Any]] = []
    total = len(unique)

    for i, (subj_url, corr_url) in enumerate(unique, start=1):
        if progress_cb:
            progress_cb(i, total, subj_url)

        sb, err = http_get(subj_url, ua, timeout)
        if err or not sb:
            manifest["download_fail"] += 1
            manifest["errors"].append(f"subject_download_fail:{subj_url}:{err}")
            continue

        if (len(sb) / (1024 * 1024)) > max_pdf_mb:
            manifest["download_fail"] += 1
            manifest["errors"].append(f"subject_too_large_mb:{subj_url}:{len(sb)/(1024*1024):.1f}")
            continue

        cb = None
        if corr_url:
            cb2, errc = http_get(corr_url, ua, timeout)
            if errc:
                manifest["errors"].append(f"corr_download_fail:{corr_url}:{errc}")
            else:
                if (len(cb2) / (1024 * 1024)) > max_pdf_mb:
                    manifest["errors"].append(f"corr_too_large_mb:{corr_url}:{len(cb2)/(1024*1024):.1f}")
                else:
                    cb = cb2

        item = {
            "pair_id": f"PAIR_{key}_{i:04d}_{sha8_bytes(sb)}",
            "scope_key": key,
            "label": urllib.parse.urlparse(subj_url).path.split("/")[-1] or f"subject_{i}.pdf",
            "subject_url": subj_url,
            "corr_url": corr_url,
            "subject_bytes": sb,
            "correction_bytes": cb,
        }
        items.append(item)
        manifest["download_ok"] += 1
        if cb:
            manifest["pairs_with_correction"] += 1

    manifest["ended_at"] = now_iso()
    return items, manifest


# =============================================================================
# PDF -> Qi/RQi EXTRACTION (structure-driven) — TEST HARNESS
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

def extract_qi_rqi_from_library(library_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    qi_list: List[Dict[str, Any]] = []
    stats = {
        "library_items": len(library_items),
        "qi_total": 0,
        "qi_posable": 0,
        "aligned_rqi": 0,
        "pdfplumber": bool(pdfplumber),
        "errors": [],
    }

    for pair in library_items:
        try:
            subj_text = read_pdf_text(pair.get("subject_bytes") or b"")
            corr_text = read_pdf_text(pair.get("correction_bytes") or b"")

            subj_segs = segment_questions(subj_text)
            corr_segs = segment_questions(corr_text)
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

def proof_signature(intent_code: str, qi_ids: List[str]) -> str:
    payload = intent_code + "|" + "|".join(sorted(qi_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        "created_at": now_iso(),
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
# VALIDATORS B (BLOQUANTS) + REPORTS
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
        "generated_at": now_iso(),
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

    # B-00: MIN POSABLE + MIN QC (anti-faux SEALED)
    if qi_posable <= 0:
        failures.append({"code": "B-00_MIN_POSABLE", "detail": "qi_posable==0 (aucune correction alignée)."})
    if qc_total <= 0:
        failures.append({"code": "B-00_MIN_QC", "detail": "qc_total==0 (aucune QC générée)."})

    # B-01: orphans posables interdits
    if orphans_global:
        failures.append({"code": "B-01_ORPHANS_POSABLE", "count": len(orphans_global), "sample": orphans_global[:20]})

    # B-02..B-04: preuve par chapitre actif
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
# SATURATION — proof signatures
# =============================================================================

def saturation_delta(prev_qcs: List[Dict[str, Any]], new_qcs: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    prev_sigs = {qc.get("qc_proof_sig") for qc in prev_qcs if qc.get("qc_proof_sig")}
    new_sigs = {qc.get("qc_proof_sig") for qc in new_qcs if qc.get("qc_proof_sig")}
    added = sorted(list(new_sigs - prev_sigs))
    return len(added), added


# =============================================================================
# STREAMLIT UI — 3 tabs (Import PDF / Chaîne complète / Résultats)
# =============================================================================

def init_state():
    defaults = {
        "pack": None,
        "country": "FR",
        "level": None,
        "subjects": [],
        "qi_global": [],
        "qi_stats": None,
        "qcs_global": [],
        "gte_stats": None,
        "sealed": False,
        "last_B": None,
        "harvest_manifest": None,
        "iterations": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    library_init()

def sidebar(pack: Optional[Dict[str, Any]]):
    st.sidebar.markdown("## ÉTAPE 1 — ACTIVATION PAYS")
    st.sidebar.caption("1 clic = load_academic_pack(country) (invisible). Pack visible = preuve.")

    st.session_state.country = st.sidebar.selectbox("Pays (TEST)", ["FR", "CI"], index=0 if st.session_state.country == "FR" else 1)
    if st.sidebar.button("🗝️ ACTIVER", type="primary", use_container_width=True):
        p = load_academic_pack(st.session_state.country)
        if not p:
            st.sidebar.error("Pack introuvable.")
        else:
            st.session_state.pack = p
            st.session_state.level = None
            st.session_state.subjects = []
            st.session_state.qi_global = []
            st.session_state.qcs_global = []
            st.session_state.qi_stats = None
            st.session_state.gte_stats = None
            st.session_state.sealed = False
            st.session_state.last_B = None
            st.session_state.harvest_manifest = None
            st.session_state.iterations = []
            st.session_state.logs = []
            st.sidebar.success("Pack activé.")

    if pack:
        st.sidebar.markdown("---")
        st.sidebar.success(f"Pack actif:\n\n**{pack.get('pack_id')}**")
        st.sidebar.caption(f"Signature: {pack.get('signature', {}).get('hash')} / {pack.get('signature', {}).get('signed_at')}")

        st.sidebar.markdown("### Niveaux")
        acs = pack.get("academic_structure", {})
        levels = sorted(acs.get("levels", []), key=lambda x: x.get("order", 999))
        for lv in levels:
            st.sidebar.caption(f"• {lv.get('label')} ({lv.get('id')})")

        st.sidebar.markdown("### Chapitres (pack-driven)")
        for ch in pack.get("chapters", []):
            st.sidebar.caption(f"• {ch.get('chapter_code')} — {ch.get('label')}")

        st.sidebar.markdown("---")
        st.sidebar.markdown("## ÉTAPE 2 — SÉLECTION")
        level_ids = [l["id"] for l in levels]
        level_labels = {l["id"]: l["label"] for l in levels}
        st.session_state.level = st.sidebar.radio(
            "Niveau",
            level_ids,
            index=level_ids.index(st.session_state.level) if st.session_state.level in level_ids else (level_ids.index("TERMINALE") if "TERMINALE" in level_ids else 0),
        )

        subj_labels = {s["id"]: s["label"] for s in acs.get("subjects", [])}
        allowed = acs.get("level_subjects", {}).get(st.session_state.level, [])
        st.session_state.subjects = st.sidebar.multiselect(
            "Matières (1–2 recommandé pour test)",
            options=allowed,
            default=[allowed[0]] if allowed else [],
            format_func=lambda x: subj_labels.get(x, x),
        )

        st.sidebar.markdown("---")
        st.sidebar.success("Sélection:")
        st.sidebar.write(f"• Niveau: **{level_labels.get(st.session_state.level, st.session_state.level)}**")
        st.sidebar.write(f"• Matières: **{', '.join(st.session_state.subjects) if st.session_state.subjects else '—'}**")

        st.sidebar.markdown("---")
        st.sidebar.markdown("## Bibliothèque (visible)")
        total, with_corr = library_counts()
        st.sidebar.write(f"Items: **{total}**")
        st.sidebar.write(f"Avec corrigé: **{with_corr}**")

def render_library_zone(title: str):
    st.subheader(title)
    total, with_corr = library_counts()
    st.caption(f"Bibliothèque = Harvest AUTO + Upload manuel. Total={total} | Avec corrigé={with_corr}")

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("🧹 Vider bibliothèque", use_container_width=True):
            st.session_state.library = []
            st.session_state.qi_global = []
            st.session_state.qcs_global = []
            st.session_state.qi_stats = None
            st.session_state.gte_stats = None
            st.session_state.sealed = False
            st.session_state.last_B = None
            st.session_state.iterations = []
            st.session_state.logs = []
            st.success("Bibliothèque vidée (et résultats reset).")
            return

    rows = library_rows()
    if not rows:
        st.info("Aucun item en bibliothèque. Lancez la chaîne complète (Harvest AUTO) ou faites un upload manuel.")
        return

    st.dataframe(rows, use_container_width=True, height=300)

    # Détails + download d’un item
    pair_ids = [r["pair_id"] for r in rows if r.get("pair_id")]
    choice = st.selectbox("Détails d’un sujet", options=["—"] + pair_ids, index=0)
    if choice and choice != "—":
        it = next((x for x in st.session_state.library if x.get("pair_id") == choice), None)
        if it:
            st.markdown("#### Détails item")
            st.write({
                "pair_id": it.get("pair_id"),
                "scope_key": it.get("scope_key"),
                "source": it.get("source"),
                "subject_url": it.get("subject_url"),
                "corr_url": it.get("corr_url"),
                "subject_size_kb": int(len(it.get("subject_bytes") or b"") / 1024),
                "corr_size_kb": int(len(it.get("correction_bytes") or b"") / 1024),
                "added_at": it.get("added_at"),
            })
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ Télécharger SUJET",
                    data=it.get("subject_bytes") or b"",
                    file_name=it.get("label") or "sujet.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with c2:
                if it.get("correction_bytes"):
                    st.download_button(
                        "⬇️ Télécharger CORRIGÉ",
                        data=it.get("correction_bytes") or b"",
                        file_name=("CORR_" + (it.get("label") or "corrige.pdf")),
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.warning("Aucun corrigé détecté pour cet item.")


def main():
    st.set_page_config(page_title=f"SMAXIA GTE {BUILD_TAG}", layout="wide")
    init_state()

    st.title(f"🔒 SMAXIA GTE Console {BUILD_TAG}")
    st.caption("ISO-PROD TEST — Activation → Pack visible → Sélection → Harvest → Extraction → RUN GTE → Saturation → Résultats")

    pack = st.session_state.pack
    sidebar(pack)

    tabs = st.tabs(["📥 Import PDF", "⚡ Chaîne complète", "📊 Résultats / Explorateur / Exports"])

    # -------------------------------------------------------------------------
    # TAB 1 — Import PDF (auto library visible + upload manual)
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.header("Import PDF (sujets + corrigés)")
        st.caption("Bibliothèque = Harvest AUTO + Upload manuel. La chaîne GTE consomme la bibliothèque.")

        colL, colR = st.columns([2, 1])
        with colL:
            render_library_zone("Bibliothèque Harvest (visible)")

        with colR:
            st.subheader("Upload manuel (optionnel)")
            st.caption("Fallback si Harvest AUTO indisponible. Ajout direct en bibliothèque.")
            subj_files = st.file_uploader("PDF Sujet", type=["pdf"], accept_multiple_files=True, key="manual_subj")
            corr_files = st.file_uploader("PDF Correction (opt)", type=["pdf"], accept_multiple_files=True, key="manual_corr")

            if st.button("➕ Ajouter à la bibliothèque", type="primary", use_container_width=True):
                if not subj_files:
                    st.error("Chargez au moins un PDF Sujet.")
                else:
                    # pairing best-effort par nom
                    corr_list = corr_files or []
                    pairs = []
                    for i, sf in enumerate(subj_files, start=1):
                        sb = sf.read()
                        sf.seek(0)

                        cb = None
                        if corr_list:
                            sname = sf.name.lower()
                            for cf in corr_list:
                                cname = cf.name.lower()
                                if ("corrig" in cname or "correc" in cname) and (
                                    re.sub(r"(corrig[eé]|correction|corrige)", "", cname)[:25] in re.sub(r"(enonce|énoncé)", "", sname)[:25]
                                    or re.sub(r"(enonce|énoncé)", "", sname)[:25] in re.sub(r"(corrig[eé]|correction|corrige)", "", cname)[:25]
                                ):
                                    cb = cf.read()
                                    cf.seek(0)
                                    break

                        pairs.append({
                            "pair_id": f"MANUAL_{i:04d}_{sha8_bytes(sb)}",
                            "scope_key": "MANUAL",
                            "label": sf.name,
                            "subject_url": None,
                            "corr_url": None,
                            "subject_bytes": sb,
                            "correction_bytes": cb,
                        })

                    added = library_add_items(pairs, source="MANUAL")
                    log(f"[BIBLIOTHÈQUE] +{added} items ajoutés (MANUAL).")
                    st.success(f"Ajouté à la bibliothèque : {added} item(s).")

    # -------------------------------------------------------------------------
    # TAB 2 — Chaîne complète (Harvest -> Extraction -> RUN GTE -> Saturation)
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.header("ÉTAPE 3 — LANCER CHAÎNE COMPLÈTE (Harvest → Extraction → RUN GTE → Saturation)")
        st.caption("Objectif ISO-PROD : preuves exportables, validateurs B bloquants, scellement seulement si new_proof_sig=0 ET B PASS (et POSABLE>0 et QC>0).")

        if not pack:
            st.warning("Activez d’abord un pays.")
        elif not st.session_state.level or not st.session_state.subjects:
            st.warning("Sélectionnez Niveau + Matières dans la sidebar.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                base_volume = st.number_input("Volume initial (par matière)", min_value=5, value=25, step=5)
            with c2:
                max_iter = st.number_input("Max itérations", min_value=1, value=4, step=1)
            with c3:
                incr = st.number_input("Incrément volume", min_value=5, value=50, step=5)

            btn = st.button("⚡ LANCER CHAÎNE COMPLÈTE", type="primary", use_container_width=True)

            log_box = st.empty()
            table_box = st.empty()

            def render_logs():
                log_box.code("\n".join(st.session_state.logs[-200:]), language="text")

            if btn:
                st.session_state.sealed = False
                st.session_state.iterations = []
                prev_qcs = st.session_state.qcs_global[:] if st.session_state.qcs_global else []
                current_volume = int(base_volume)

                log(f"═══ DÉMARRAGE CHAÎNE {BUILD_TAG} ═══")
                log(f"Pays: {st.session_state.country} | Niveau: {st.session_state.level} | Matières: {st.session_state.subjects}")
                render_logs()

                for it in range(1, int(max_iter) + 1):
                    if st.session_state.sealed:
                        break

                    log(f"══ ITÉRATION {it} — Volume: {current_volume} sujets/matière ══")
                    render_logs()

                    progress = st.progress(0.0, text="Harvest…")
                    scanned = 0
                    target = max(1, len(st.session_state.subjects) * current_volume)

                    def pcb(cur, total, url):
                        nonlocal scanned
                        scanned += 1
                        progress.progress(min(1.0, scanned / target), text=f"{scanned}/{target} — {url.split('/')[-1][:60]}")

                    all_items = []
                    all_manifests = []

                    for subj in st.session_state.subjects:
                        scope = f"{st.session_state.level}|{subj}"
                        log(f"[HARVEST] Scope: {scope} …")
                        items, manifest = harvest_subjects(pack, st.session_state.level, subj, current_volume, pcb)
                        all_items.extend(items)
                        all_manifests.append(manifest)
                        log(f"[HARVEST] {scope}: {len(items)} pairs téléchargées (corrigé si trouvé).")
                        render_logs()

                    progress.empty()

                    if not all_items:
                        log("⚠️ 0 pairs récoltées. Arrêt.")
                        render_logs()
                        break

                    added = library_add_items(all_items, source="AUTO")
                    log(f"[BIBLIOTHÈQUE] +{added} items (total={len(st.session_state.library)}).")
                    render_logs()

                    # Extraction depuis la BIBLIOTHÈQUE (et non un lot seulement)
                    log("[EXTRACTION] Dépessage Qi/RQi …")
                    qi, qi_stats = extract_qi_rqi_from_library(st.session_state.library)
                    st.session_state.qi_global = qi
                    st.session_state.qi_stats = qi_stats
                    log(f"[EXTRACTION] +{qi_stats.get('qi_total')} Qi (global={len(st.session_state.qi_global)}). POSABLE={qi_stats.get('qi_posable')}")
                    render_logs()

                    # RUN GTE
                    log("[GTE] Construction QC (ARI/FRT/TRG -> QC) …")
                    qcs, gte_stats = run_pipeline_gte(st.session_state.qi_global, pack.get("chapters", []))
                    st.session_state.qcs_global = qcs
                    st.session_state.gte_stats = gte_stats
                    log(f"[GTE] QC total={len(qcs)} | clusters={gte_stats.get('clusters')} | singletons={gte_stats.get('singletons')}")
                    render_logs()

                    okB, B = validate_B_blocking(pack.get("chapters", []), st.session_state.qi_global, st.session_state.qcs_global)
                    st.session_state.last_B = B
                    log(f"[B] {'PASS' if okB else 'FAIL'} | qi_posable={B.get('qi_posable')} | qc_total={B.get('qc_total')} | orphans_global={len(B.get('orphans_global') or [])}")
                    render_logs()

                    new_count, _ = saturation_delta(prev_qcs, qcs)

                    st.session_state.iterations.append({
                        "iter": it,
                        "volume": current_volume,
                        "pairs": len(all_items),
                        "library_total": len(st.session_state.library),
                        "qi_total": len(st.session_state.qi_global),
                        "qi_posable": B.get("qi_posable"),
                        "qc_total": len(st.session_state.qcs_global),
                        "new_proof_sig": new_count,
                        "B_ok": okB,
                        "timestamp": now_iso(),
                    })
                    table_box.dataframe(st.session_state.iterations, use_container_width=True)

                    # SEALED condition strict
                    can_seal = (new_count == 0 and okB and (B.get("qi_posable", 0) > 0) and (B.get("qc_total", 0) > 0))
                    if can_seal:
                        st.session_state.sealed = True
                        log("🎯 ═══ SATURATION ATTEINTE — SEALED (new_proof_sig=0 ET B PASS ET POSABLE>0 ET QC>0) ═══")
                        render_logs()
                        break
                    else:
                        if new_count == 0 and not okB:
                            log("⛔ new_proof_sig=0 MAIS B FAIL -> interdit de sceller.")
                        else:
                            log(f"↻ Non scellé : new_proof_sig={new_count} | B_ok={okB}")
                        render_logs()

                    prev_qcs = qcs[:]
                    current_volume += int(incr)

                if not st.session_state.sealed:
                    log("⚠️ Saturation non atteinte (ou preuves insuffisantes) dans la limite d'itérations.")
                    render_logs()

    # -------------------------------------------------------------------------
    # TAB 3 — Résultats / Explorateur / Exports
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.header("ÉTAPE 4 — RÉSULTATS / EXPLORATEUR / EXPORTS")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Qi global", len(st.session_state.qi_global))
        m2.metric("Qi POSABLE", sum(1 for q in st.session_state.qi_global if q.get("has_rqi")))
        m3.metric("QC", len(st.session_state.qcs_global))
        m4.metric("SEALED", "YES" if st.session_state.sealed else "NO")

        st.markdown("---")
        st.subheader("Explorateur : QC → ARI → FRT → Triggers → Qi")

        if not st.session_state.qcs_global:
            st.info("Exécutez la chaîne complète pour obtenir des QC (et vérifiez qu’il y a des corrigés pour obtenir des Qi POSABLE).")
        else:
            by_ch = defaultdict(list)
            for qc in st.session_state.qcs_global:
                by_ch[qc.get("chapter_code") or "ORPHAN"].append(qc)

            ch_label = {c.get("chapter_code"): c.get("label") for c in (pack.get("chapters", []) if pack else [])}

            for ch_code in sorted(by_ch.keys()):
                label = ch_label.get(ch_code, ch_code)
                with st.expander(f"{ch_code} — {label} ({len(by_ch[ch_code])} QC)", expanded=False):
                    for qc in by_ch[ch_code]:
                        st.markdown(f"### {qc.get('qc_id')}")
                        st.caption(f"intent_code={qc.get('intent_code')} | proof_sig={str(qc.get('qc_proof_sig'))[:16]}…")
                        st.write(f"**Formulation**: {qc.get('formulation')}")
                        st.write(f"**Cluster**: {qc.get('n_q_cluster')} Qi")
                        st.write("**ARI (template invariant)**")
                        st.json(qc.get("ari"))
                        st.write("**FRT (template invariant)**")
                        st.json(qc.get("frt"))
                        st.write("**Triggers**")
                        st.json(qc.get("triggers"))
                        st.write("**Qi associées (preuve)**")
                        st.json((qc.get("qi_ids") or [])[:30])

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
