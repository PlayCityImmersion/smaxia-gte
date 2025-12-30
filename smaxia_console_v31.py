# =============================================================================
# SMAXIA GTE Console V31.9.11 (ISO-PROD — Kernel-aligned) — UI FLUX STRICT
# =============================================================================
#
# UI FLUX (OBLIGATOIRE)
# [1] Activation pays (1 clic) -> load_academic_pack("FR") (invisible)
# [2] Pack visible -> Niveaux / Matières / Chapitres
# [3] Sélection -> Niveau + 1-2 matières
# [4] Harvest AUTO (pêche aveugle) -> barre progression visible
# [5] Pipeline -> Qi/RQi -> (ARI,FRT,Triggers) -> QC (ordre strict)
# [6] Mapping chapitre -> intent_code -> Pack.chapter_code (pack-driven)
# [7] Saturation -> arrêt si new_proof_sig=0 ET validateurs B PASS -> SEALED
# [8] Explorateur -> QC → ARI → FRT → Triggers → Qi
# [9] Exports -> preuves + logs
#
# KERNEL V10.6.1 (CAS 1 ONLY) : couverture 100% POSABLE, orphelins POSABLE bloquants,
# pas de reconstruction, déterminisme, preuves exportables. (Réf doc scellé)
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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# -------------------------
# BUILD TAG (preuve UI)
# -------------------------
BUILD_TAG = "V31.9.11-UIFLUX-2025-12-31"


# =============================================================================
# ZONE A — PACK LOADER (SIMULATION ISO-PROD)
# =============================================================================
def load_academic_pack(country_code: str) -> Optional[Dict[str, Any]]:
    # NOTE TEST: ce loader est un HARNESS ISO-PROD.
    # En PROD: API/DB/Prompt5 admin.
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
                    "max_links_scan": 1200,
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


# =============================================================================
# HARVEST (pack-driven) + pairing
# =============================================================================
def http_get(url: str, ua: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), None
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
    max_links_scan = int(http_cfg.get("max_links_scan", 600))

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
        manifest["ended_at"] = datetime.now().isoformat()
        return [], manifest

    rnd = random.Random(42)  # déterminisme ISO
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
            cb, errc = http_get(corr_url, ua, timeout)
            if errc:
                manifest["errors"].append(f"corr_download_fail:{corr_url}:{errc}")
                cb = None
            elif cb and (len(cb) / (1024 * 1024)) > max_pdf_mb:
                manifest["errors"].append(f"corr_too_large_mb:{corr_url}:{len(cb)/(1024*1024):.1f}")
                cb = None

        items.append({
            "pair_id": f"PAIR_{key}_{i:04d}_{sha8_bytes(sb)}",
            "scope_key": key,
            "label": urllib.parse.urlparse(subj_url).path.split("/")[-1] or f"subject_{i}.pdf",
            "subject_url": subj_url,
            "corr_url": corr_url,
            "subject_bytes": sb,
            "correction_bytes": cb,
        })
        manifest["download_ok"] += 1
        if cb:
            manifest["pairs_with_correction"] += 1

    manifest["ended_at"] = datetime.now().isoformat()
    return items, manifest


# =============================================================================
# PDF -> Qi/RQi EXTRACTION (structure-driven HARNESS)
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
    # HARNESS TEST (non PROD): EXn-Qm -> STRUCT_EXn_Qm
    m = re.match(r"EX(\d+)-Q(\d+)", (marker or "").strip(), re.I)
    if not m:
        return "STRUCT_UNK"
    return f"STRUCT_EX{m.group(1)}_Q{m.group(2)}"

def extract_qi_rqi(pairs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    qi_list: List[Dict[str, Any]] = []
    stats = {
        "pairs": len(pairs),
        "qi_total": 0,
        "qi_posable": 0,  # CAS 1 ONLY (test): posable = corrigé exploitable présent
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
# PIPELINE GTE — INVARIANT (ARI/FRT/TRG templates) -> QC (conteneur final)
# =============================================================================
def cluster_by_intent(qi_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for qi in qi_list:
        if qi.get("has_rqi"):  # CAS 1 ONLY: POSABLE
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
            "usage": "USAGE",
            "preconditions": "PRECONDITIONS",
            "ari": "ARI_TYPED",
            "checkpoints": ["CHECKPOINTS"],
            "traps": ["TRAPS"],
            "final": "FINAL",
            "proofs": ["EVIDENCE_REFS"],
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
    # Anti-singleton (normatif): n_q_cluster >= 2
    if len(cluster) < 2:
        return None

    qi_ids = [q["qi_id"] for q in cluster if q.get("qi_id")]
    sig = proof_signature(intent_code, qi_ids)

    ari = make_ari_template(intent_code, sig)
    frt = make_frt_template(intent_code, sig)
    trgs = make_triggers_template(intent_code, sig)

    return {
        "qc_id": f"QC_{intent_code}_{sha10_str(intent_code)}",
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
    # Pack-driven allowlist (Option A)
    for ch in chapters:
        if intent_code in (ch.get("intent_allowlist") or []):
            return ch.get("chapter_code")
    return None

def run_pipeline_gte(qi_list: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    clusters = cluster_by_intent(qi_list)

    qc_list: List[Dict[str, Any]] = []
    singletons: List[Dict[str, Any]] = []

    # ORDRE VALIDÉ: ARI/FRT/TRG générés AVANT QC (QC = conteneur final)
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
# PROOFS + VALIDATORS B (BLOQUANTS)
# =============================================================================
def compute_orphans_posable(qi_list: List[Dict[str, Any]], qcs: List[Dict[str, Any]]) -> List[str]:
    posable_ids = {q["qi_id"] for q in qi_list if q.get("has_rqi") and q.get("qi_id")}
    covered = set()
    for qc in qcs:
        for qid in (qc.get("qi_ids") or []):
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
    orphans_global = compute_orphans_posable(qi_list, qcs)
    chapter_report = build_chapter_report(chapters, qi_list, qcs)

    failures = []
    if orphans_global:
        failures.append({"code": "B-01_ORPHANS_POSABLE", "count": len(orphans_global), "sample": orphans_global[:20]})

    for ch_code, data in (chapter_report.get("chapters") or {}).items():
        if data.get("qi_posable_count", 0) == 0:
            failures.append({"code": "B-02_NO_EVIDENCE_CHAPTER", "chapter": ch_code})
        if data.get("coverage_pct") != 100.0:
            failures.append({"code": "B-03_COVERAGE_NOT_100", "chapter": ch_code, "coverage_pct": data.get("coverage_pct")})
        if data.get("qc_count", 0) == 0:
            failures.append({"code": "B-04_NO_QC_CHAPTER", "chapter": ch_code})

    ok = (len(failures) == 0)
    return ok, {
        "ok": ok,
        "orphans_global": orphans_global,
        "chapter_report": chapter_report,
        "failures": failures,
    }


# =============================================================================
# SATURATION — proof signatures delta (anti-faux SEALED)
# =============================================================================
def saturation_delta(prev_qcs: List[Dict[str, Any]], new_qcs: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    prev_sigs = {qc.get("qc_proof_sig") for qc in prev_qcs if qc.get("qc_proof_sig")}
    new_sigs = {qc.get("qc_proof_sig") for qc in new_qcs if qc.get("qc_proof_sig")}
    added = sorted(list(new_sigs - prev_sigs))
    return len(added), added


# =============================================================================
# STREAMLIT UI — FLUX STRICT
# =============================================================================
def init_state():
    defaults = {
        "country": "FR",
        "pack": None,
        "level": None,
        "subjects": [],
        "library_pairs": [],          # bibliothèque (auto+manuel)
        "harvest_manifest": None,
        "qi_global": [],
        "qi_stats": None,
        "qcs_global": [],
        "gte_stats": None,
        "last_B": None,
        "sealed": False,
        "iterations": [],
        "log_lines": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log(line: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_lines.append(f"[{ts}] {line}")

def reset_runtime_state(keep_pack: bool = True):
    pack = st.session_state.pack if keep_pack else None
    country = st.session_state.country
    st.session_state.level = None
    st.session_state.subjects = []
    st.session_state.library_pairs = []
    st.session_state.harvest_manifest = None
    st.session_state.qi_global = []
    st.session_state.qi_stats = None
    st.session_state.qcs_global = []
    st.session_state.gte_stats = None
    st.session_state.last_B = None
    st.session_state.sealed = False
    st.session_state.iterations = []
    st.session_state.log_lines = []
    st.session_state.pack = pack
    st.session_state.country = country

def sidebar_activation_and_pack():
    st.sidebar.markdown("## ÉTAPE 1 — ACTIVATION PAYS")
    st.sidebar.caption("1 clic = load_academic_pack(country) (invisible). Pack visible = preuve.")

    st.session_state.country = st.sidebar.selectbox("Pays (TEST)", ["FR", "CI"], index=0 if st.session_state.country == "FR" else 1)

    if st.sidebar.button("🔓 ACTIVER", type="primary", use_container_width=True):
        pack = load_academic_pack(st.session_state.country)
        if not pack:
            st.sidebar.error("Pack introuvable.")
        else:
            st.session_state.pack = pack
            reset_runtime_state(keep_pack=True)
            log(f"═══ ACTIVATION ═══ Pays: {pack.get('country_code')} | Pack: {pack.get('pack_id')} | Version: {pack.get('version')}")
            st.sidebar.success(f"Pack actif: {pack.get('pack_id')}")

    pack = st.session_state.pack
    if not pack:
        st.sidebar.info("Activez un pays.")
        return

    st.sidebar.markdown("---")
    st.sidebar.write(f"✅ **Pack actif:** {pack.get('pack_id')}")
    st.sidebar.caption(f"Signature: {pack.get('signature', {}).get('hash')} / {pack.get('signature', {}).get('signed_at')}")

    acs = pack.get("academic_structure", {}) or {}
    st.sidebar.markdown("### Niveaux")
    for lv in sorted(acs.get("levels", []), key=lambda x: x.get("order", 999)):
        st.sidebar.caption(f"• {lv.get('label')} ({lv.get('id')})")

    st.sidebar.markdown("### Matières (selon niveau)")
    if st.session_state.level:
        labels = {s["id"]: s["label"] for s in acs.get("subjects", [])}
        allowed = acs.get("level_subjects", {}).get(st.session_state.level, [])
        for s in allowed:
            st.sidebar.caption(f"• {labels.get(s, s)} ({s})")
    else:
        for s in acs.get("subjects", []):
            st.sidebar.caption(f"• {s.get('label')} ({s.get('id')})")

    st.sidebar.markdown("### Chapitres (pack-driven)")
    for ch in pack.get("chapters", [])[:18]:
        st.sidebar.caption(f"• {ch.get('chapter_code')} — {ch.get('label')}")
    if len(pack.get("chapters", [])) > 18:
        st.sidebar.caption("…")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ÉTAPE 2 — SÉLECTION")
    if not st.session_state.pack:
        st.sidebar.info("Activez un pays.")
        return

    levels = sorted(acs.get("levels", []), key=lambda x: x.get("order", 999))
    level_ids = [l["id"] for l in levels]
    level_labels = {l["id"]: l["label"] for l in levels}

    st.session_state.level = st.sidebar.radio(
        "Niveau",
        level_ids,
        horizontal=False,
        format_func=lambda x: level_labels.get(x, x),
        index=level_ids.index(st.session_state.level) if st.session_state.level in level_ids else 0,
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
    st.sidebar.write("✅ Sélection:")
    st.sidebar.write(f"- Niveau: **{level_labels.get(st.session_state.level, st.session_state.level)}**")
    st.sidebar.write(f"- Matières: **{', '.join(st.session_state.subjects) if st.session_state.subjects else '—'}**")


def ui_import_pdf():
    st.subheader("Import PDF (sujets + corrigés)")
    st.caption("Bibliothèque = Harvest AUTO + Upload manuel. La chaîne GTE consomme la bibliothèque.")

    colL, colR = st.columns([2, 1])

    with colL:
        st.markdown("### Bibliothèque Harvest (visible)")
        if not st.session_state.library_pairs:
            st.info("Aucun item en bibliothèque. Lancez la chaîne complète (Harvest AUTO) ou faites un upload manuel.")
        else:
            rows = []
            for p in st.session_state.library_pairs[:300]:
                rows.append({
                    "pair_id": p.get("pair_id"),
                    "scope": p.get("scope_key"),
                    "sujet": p.get("label"),
                    "corrigé?": "✅" if p.get("correction_bytes") else "❌",
                })
            st.dataframe(rows, use_container_width=True, height=340)
            st.caption("Affichage limité à 300 lignes (export complet disponible).")

    with colR:
        st.markdown("### Upload manuel (optionnel)")
        subj_files = st.file_uploader("PDF Sujet", type=["pdf"], accept_multiple_files=True, key="manual_subj")
        corr_files = st.file_uploader("PDF Correction (opt)", type=["pdf"], accept_multiple_files=True, key="manual_corr")

        if st.button("➕ Ajouter à la bibliothèque", use_container_width=True):
            if not subj_files:
                st.error("Ajoutez au moins un PDF Sujet.")
                return

            pairs = []
            corr_list = corr_files or []

            for i, sf in enumerate(subj_files, start=1):
                sb = sf.read()
                sf.seek(0)
                cb = None

                # pairing best-effort par nom
                sname = (sf.name or "").lower()
                for cf in corr_list:
                    cname = (cf.name or "").lower()
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

            before = len(st.session_state.library_pairs)
            st.session_state.library_pairs.extend(pairs)
            after = len(st.session_state.library_pairs)

            log(f"[MANUAL] +{after-before} pairs ajoutées à la bibliothèque.")
            st.success(f"Ajouté : {after-before} pair(s).")


def ui_chain_complete():
    st.subheader("ÉTAPE 3 — LANCER CHAÎNE COMPLÈTE (Harvest → Extraction → RUN GTE → Saturation)")
    st.caption("Objectif ISO-PROD : preuves exportables, validateurs B bloquants, scellement seulement si new_proof_sig=0 ET B PASS.")

    if not st.session_state.pack:
        st.warning("Activez d'abord un pays.")
        return
    if not st.session_state.level or not st.session_state.subjects:
        st.warning("Sélectionnez Niveau + au moins une matière.")
        return

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        base_volume = st.number_input("Volume initial (par matière)", min_value=5, value=20, step=5)
    with colB:
        max_iter = st.number_input("Max itérations", min_value=1, value=5, step=1)
    with colC:
        incr = st.number_input("Incrément volume", min_value=5, value=50, step=5)

    run_btn = st.button("⚡ LANCER CHAÎNE COMPLÈTE", type="primary", use_container_width=True)

    log_box = st.empty()
    prog = st.progress(0.0)
    iter_table = st.empty()

    def render_log():
        last = "\n".join(st.session_state.log_lines[-200:])
        log_box.code(last or "(log vide)", language="text")

    if run_btn:
        # reset runtime (mais conserve pack + sélection + bibliothèque existante)
        st.session_state.sealed = False
        st.session_state.iterations = []
        st.session_state.qi_global = []
        st.session_state.qcs_global = []
        st.session_state.qi_stats = None
        st.session_state.gte_stats = None
        st.session_state.last_B = None

        st.session_state.log_lines = []
        log(f"═══ DÉMARRAGE CHAÎNE {BUILD_TAG} ═══")
        log(f"Pays: {st.session_state.pack.get('country_code')} | Niveau: {st.session_state.level} | Matières: {st.session_state.subjects}")

        current_volume = int(base_volume)
        prev_qcs: List[Dict[str, Any]] = []

        chapters = st.session_state.pack.get("chapters", [])

        for it in range(1, int(max_iter) + 1):
            if st.session_state.sealed:
                break

            log(f"══ ITÉRATION {it} — Volume: {current_volume} sujets/matière ══")
            render_log()

            # Harvest per subject
            all_pairs = []
            all_manifests = []

            def progress_cb(cur, total, url):
                prog.progress(cur / max(1, total), text=f"Harvest {cur}/{total} — {url.split('/')[-1][:60]}")

            for subj in st.session_state.subjects:
                scope = f"{st.session_state.level}|{subj}"
                log(f"[HARVEST] Scope: {scope} …")
                render_log()

                pairs, manifest = harvest_subjects(st.session_state.pack, st.session_state.level, subj, current_volume, progress_cb)
                all_pairs.extend(pairs)
                all_manifests.append(manifest)

                log(f"[HARVEST] {scope}: {len(pairs)} pairs téléchargées (corrigé si trouvé).")
                render_log()

            prog.progress(1.0)

            if not all_pairs:
                log("⛔ 0 pairs récoltées. Arrêt.")
                render_log()
                break

            # Ajouter à bibliothèque visible
            before = len(st.session_state.library_pairs)
            st.session_state.library_pairs.extend(all_pairs)
            log(f"[BIBLIOTHÈQUE] +{len(st.session_state.library_pairs)-before} items (total={len(st.session_state.library_pairs)}).")
            render_log()

            # Extraction Qi/RQi (CAS 1 ONLY: POSABLE = has_rqi)
            log("[EXTRACTION] Dépessage Qi/RQi …")
            render_log()
            qi_new, qi_stats = extract_qi_rqi(all_pairs)

            # Merge by qi_id (déterministe)
            existing = {q["qi_id"] for q in st.session_state.qi_global}
            new_only = [q for q in qi_new if q.get("qi_id") not in existing]
            st.session_state.qi_global.extend(new_only)
            st.session_state.qi_stats = qi_stats | {"new_qi_added": len(new_only), "qi_global_total": len(st.session_state.qi_global)}

            log(f"[EXTRACTION] +{len(new_only)} Qi (global={len(st.session_state.qi_global)}). POSABLE={sum(1 for q in st.session_state.qi_global if q.get('has_rqi'))}")
            render_log()

            # Pipeline GTE (ARI/FRT/TRG -> QC) + mapping chapitre
            log("[GTE] Construction QC (ARI/FRT/TRG -> QC) …")
            render_log()
            qcs, gte_stats = run_pipeline_gte(st.session_state.qi_global, chapters)
            st.session_state.qcs_global = qcs
            st.session_state.gte_stats = gte_stats

            log(f"[GTE] QC total={len(qcs)} | clusters={gte_stats.get('clusters')} | singletons={gte_stats.get('singletons')}")
            render_log()

            # Validateurs B (bloquants)
            okB, B = validate_B_blocking(chapters, st.session_state.qi_global, st.session_state.qcs_global)
            st.session_state.last_B = B
            log(f"[B] {'PASS' if okB else 'FAIL'} | orphans_global={len(B.get('orphans_global') or [])}")
            render_log()

            # Saturation delta proof sig
            new_count, _added_sigs = saturation_delta(prev_qcs, qcs)

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

            iter_table.dataframe(st.session_state.iterations, use_container_width=True)

            if new_count == 0 and okB:
                st.session_state.sealed = True
                log("🎯 ═══ SATURATION ATTEINTE — SEALED (new_proof_sig=0 ET B PASS) ═══")
                render_log()
                break
            else:
                if new_count == 0 and not okB:
                    log("⛔ new_proof_sig=0 MAIS B FAIL -> SCELLEMENT INTERDIT")
                else:
                    log(f"Non scellé : new_proof_sig={new_count} | B_ok={okB}")
                render_log()

            prev_qcs = qcs[:]
            current_volume += int(incr)
            time.sleep(0.05)

        if not st.session_state.sealed:
            log("Fin chaîne: non scellé (limite itérations atteinte ou preuves insuffisantes).")
            render_log()


def ui_results_explorer_exports():
    st.subheader("ÉTAPE 4 — RÉSULTATS / EXPLORATEUR / EXPORTS")

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qi global", len(st.session_state.qi_global))
    c2.metric("Qi POSABLE", sum(1 for q in st.session_state.qi_global if q.get("has_rqi")))
    c3.metric("QC", len(st.session_state.qcs_global))
    c4.metric("SEALED", "YES" if st.session_state.sealed else "NO")

    # Validateurs B
    if st.session_state.pack and st.session_state.qcs_global:
        okB, B = validate_B_blocking(st.session_state.pack.get("chapters", []), st.session_state.qi_global, st.session_state.qcs_global)
        st.session_state.last_B = B

        st.markdown("### Validateurs B (BLOQUANTS)")
        if okB:
            st.success("PASS — aucun orphelin POSABLE, preuve 100% sur chapitres actifs.")
        else:
            st.error("FAIL — violation bloquante (orphelins POSABLE ou couverture chapitre ≠ 100%).")
            with st.expander("Détails FAIL", expanded=True):
                st.json(B.get("failures"))

        st.markdown("### Chapter Report (preuve)")
        st.json((B.get("chapter_report") or {}))

    st.markdown("---")
    st.markdown("### Explorateur : QC → ARI → FRT → Triggers → Qi")

    if not st.session_state.pack or not st.session_state.qcs_global:
        st.info("Exécutez la chaîne complète pour obtenir des QC.")
    else:
        chapters = st.session_state.pack.get("chapters", [])
        ch_label = {c.get("chapter_code"): c.get("label") for c in chapters}

        by_ch = defaultdict(list)
        for qc in st.session_state.qcs_global:
            by_ch[qc.get("chapter_code") or "ORPHAN"].append(qc)

        ch_options = sorted(by_ch.keys())
        selected_ch = st.selectbox(
            "Chapitre",
            options=ch_options,
            format_func=lambda x: f"{x} — {ch_label.get(x, x)} ({len(by_ch.get(x, []))} QC)",
        )

        qcs = by_ch.get(selected_ch, [])
        if not qcs:
            st.info("Aucune QC dans ce chapitre.")
        else:
            qc_ids = [qc.get("qc_id") for qc in qcs]
            selected_qc_id = st.selectbox("QC", options=qc_ids)

            qc_obj = next((q for q in qcs if q.get("qc_id") == selected_qc_id), None)
            if qc_obj:
                st.markdown(f"**Formulation**: {qc_obj.get('formulation')}")
                st.caption(f"intent_code={qc_obj.get('intent_code')} | proof_sig={str(qc_obj.get('qc_proof_sig'))[:16]}… | n_q_cluster={qc_obj.get('n_q_cluster')}")
                st.write("**ARI (template invariant)**")
                st.json(qc_obj.get("ari"))
                st.write("**FRT (template invariant)**")
                st.json(qc_obj.get("frt"))
                st.write("**Triggers**")
                st.json(qc_obj.get("triggers"))
                st.write("**Qi associées (preuve)**")
                st.json((qc_obj.get("qi_ids") or [])[:60])

    st.markdown("---")
    st.markdown("### Exports (preuves)")
    colE1, colE2, colE3, colE4, colE5 = st.columns([1, 1, 1, 1, 1])

    with colE1:
        st.download_button("qi_pack.json", canonical_json(st.session_state.qi_global), "qi_pack.json", use_container_width=True)
    with colE2:
        st.download_button("qc_pack.json", canonical_json(st.session_state.qcs_global), "qc_pack.json", use_container_width=True)
    with colE3:
        cr = (st.session_state.last_B or {}).get("chapter_report") or {}
        st.download_button("chapter_report.json", canonical_json(cr), "chapter_report.json", use_container_width=True)
    with colE4:
        st.download_button("harvest_manifest.json", canonical_json(st.session_state.harvest_manifest or {}), "harvest_manifest.json", use_container_width=True)
    with colE5:
        st.download_button("logs.txt", "\n".join(st.session_state.log_lines), "logs.txt", use_container_width=True)


def main():
    st.set_page_config(page_title=f"SMAXIA GTE {BUILD_TAG}", layout="wide")

    init_state()
    sidebar_activation_and_pack()

    st.title(f"🔒 SMAXIA GTE Console {BUILD_TAG}")
    st.caption("ISO-PROD TEST — Activation → Pack visible → Sélection → Harvest → Extraction → RUN GTE → Saturation → Résultats")

    tabs = st.tabs([
        "📥 Import PDF",
        "⚡ Chaîne complète",
        "📊 Résultats / Explorateur / Exports",
    ])

    with tabs[0]:
        ui_import_pdf()

    with tabs[1]:
        ui_chain_complete()

    with tabs[2]:
        ui_results_explorer_exports()

    # Manifest global (après actions)
    if st.session_state.pack:
        st.session_state.harvest_manifest = {
            "build_tag": BUILD_TAG,
            "country": st.session_state.country,
            "pack_id": st.session_state.pack.get("pack_id"),
            "level": st.session_state.level,
            "subjects": st.session_state.subjects,
            "library_total_pairs": len(st.session_state.library_pairs),
            "generated_at": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    main()
