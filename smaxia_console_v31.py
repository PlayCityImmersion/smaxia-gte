# =============================================================================
# SMAXIA GTE Console V31.9.11 (ISO-PROD — Kernel-aligned)
# =============================================================================
#
# OBJECTIF ISO-PROD (TEST) — CHAÎNE COMPLÈTE, UI LISIBLE, PREUVES EXPORTABLES :
# [1] Activation pays (ex: FR) -> load_academic_pack("FR") (simulation API/DB)
# [2] UI affiche Niveaux/Matières/Chapitres (pack visible)
# [3] Sélection Niveau + 1-2 Matières
# [4] Harvest AUTO (pêche aveugle) depuis Pack.harvest_config.sources (progress visible)
# [5] Extraction Qi/RQi (structure-driven) -> intent_code (TEST harness)
# [6] Pipeline GTE invariant : (ARI+FRT+TRG templates) -> QC (conteneur)
# [7] Mapping QC -> Chapitre via Pack.chapters.intent_allowlist (pack-driven)
# [8] VALIDATEURS B BLOQUANTS + chapter_report.json (preuve par chapitre)
# [9] Saturation : arrêt si new_proof_sig == 0 ET validateurs B PASS
# [10] Explorateur QC->ARI->FRT->TRG->Qi + Exports
#
# RÈGLES KERNEL (appliquées ici) :
# - Zéro keywords métier/langue dans ARI/FRT/TRG (templates invariants)
# - Pas de hardcode chapitre/matière : tout vient du Pack (données)
# - Preuve par chapitre obligatoire (coverage 100% sur Qi POSABLES)
# - Orphelins POSABLES interdits (bloquant)
# - Saturation basée sur "proof signatures" (anti-faux SEALED)
#
# NOTE : intent_code basé sur EXn-Qm est un HARNESS DE TEST (non PROD) :
# en PROD, intent_code doit provenir du "catalogue invariant" (pack/moteur).
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


# =============================================================================
# ZONE A — PACK LOADER (SIMULATION ISO-PROD)
# =============================================================================
# En PROD : load_academic_pack() doit lire A) fichier serveur, B) API Prompt5 admin, ou C) DB.
# Ici : embarqué pour tester toute la chaîne sans dépendances externes.
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
                    {"id": "L2", "label": "Licence 2", "order": 5},
                    {"id": "L3", "label": "Licence 3", "order": 6},
                    {"id": "M1", "label": "Master 1", "order": 7},
                    {"id": "M2", "label": "Master 2", "order": 8},
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

            # CHAPTERS pack-driven (exemple TEST) : mapping via intent_allowlist
            # IMPORTANT : en PROD, allowlist doit refléter le "catalogue invariant".
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
                    "max_links_scan": 600,
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
# HARVEST (pack-driven, pêche aveugle) + MANIFEST + ERRORS
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
    # dedupe stable
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

    # 1) collecte des liens (pêche aveugle)
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

    # dedupe by subject url
    seen = set()
    unique = [(s, c) for (s, c) in all_pairs if s and (s not in seen) and not seen.add(s)]
    if not unique:
        manifest["errors"].append("no_pdf_pairs_discovered")
        manifest["ended_at"] = datetime.now().isoformat()
        return [], manifest

    # 2) pêche aveugle : échantillonnage aléatoire déterministe (seed fixe ISO-PROD)
    rnd = random.Random(42)
    if len(unique) > volume:
        unique = rnd.sample(unique, volume)

    manifest["selected_pairs"] = len(unique)

    # 3) téléchargement avec progression + contrôle taille
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
    # HARNESS TEST : EXn-Qm -> STRUCT_EXn_Qm (stable uniquement au sein d'un sujet, pas inter-sujets)
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
            corr_idx = {s["marker"]: s["text"] for s in corr_segs if s.get("marker")}

            for seg in subj_segs:
                marker = seg["marker"]
                intent_code = marker_to_intent(marker)
                rqi = corr_idx.get(marker, "")
                has_rqi = bool(rqi.strip())

                # qi_id deterministic per (pair_id + marker + statement hash)
                content_h = sha10_str(f"{pair.get('pair_id','') રાખ}|{marker}|{sha10_str(seg['text'])}")
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
    # POSABLE = has_rqi True (en ISO-PROD test)
    for qi in qi_list:
        if qi.get("has_rqi"):
            clusters[qi.get("intent_code", "STRUCT_UNK")].append(qi)
    return dict(clusters)

def make_ari_template(intent_code: str, proof_sig: str) -> Dict[str, Any]:
    # ARI strictement invariant : aucune sémantique, aucune langue, aucun métier
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
        "pattern": intent_code,  # structure (test harness)
        "confidence": 1.0,
    }]

def proof_signature(intent_code: str, qi_ids: List[str]) -> str:
    # Signature de preuve = intent_code + set couvert (anti-faux SEALED)
    payload = intent_code + "|" + "|".join(sorted(qi_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def build_qc(intent_code: str, cluster: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Anti-singleton : cluster >= 2
    if len(cluster) < 2:
        return None

    qi_ids = [q["qi_id"] for q in cluster if q.get("qi_id")]
    sig = proof_signature(intent_code, qi_ids)

    qc_id_stable = f"QC_{intent_code}_{sha10_str(intent_code)}"  # stable by intent (OK)
    ari = make_ari_template(intent_code, sig)
    frt = make_frt_template(intent_code, sig)
    trgs = make_triggers_template(intent_code, sig)

    return {
        "qc_id": qc_id_stable,
        "intent_code": intent_code,
        "qc_proof_sig": sig,                 # CRITICAL: used for saturation + proofs
        "formulation": "Comment procéder ?", # format imposé
        "n_q_cluster": len(cluster),
        "qi_ids": qi_ids,                    # preuve de couverture
        "ari": ari,
        "frt": frt,
        "triggers": trgs,
        "created_at": datetime.now().isoformat(),
    }

def map_qc_to_chapter(intent_code: str, chapters: List[Dict[str, Any]]) -> Optional[str]:
    # Option A pack-driven : intent_allowlist only
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
# PROOFS + VALIDATORS B (BLOQUANTS)
# =============================================================================

def compute_orphans_posable(qi_list: List[Dict[str, Any]], qcs: List[Dict[str, Any]]) -> List[str]:
    posable_ids = {q["qi_id"] for q in qi_list if q.get("has_rqi") and q.get("qi_id")}
    covered = set()
    for qc in qcs:
        for qid in qc.get("qi_ids", []) or []:
            covered.add(qid)
    orphans = sorted([qid for qid in posable_ids if qid not in covered])
    return orphans

def compute_active_chapters(chapters: List[Dict[str, Any]], qi_list: List[Dict[str, Any]]) -> List[str]:
    intents_seen = {q.get("intent_code") for q in qi_list if q.get("has_rqi")}
    active = []
    for ch in chapters:
        allow = set(ch.get("intent_allowlist") or [])
        if allow.intersection(intents_seen):
            active.append(ch.get("chapter_code"))
    return active

def build_chapter_report(
    chapters: List[Dict[str, Any]],
    qi_list: List[Dict[str, Any]],
    qcs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Chapitres "actifs" = ceux qui ont au moins une Qi posable dans leur allowlist
    active_codes = set(compute_active_chapters(chapters, qi_list))

    # Index QC by chapter
    qc_by_chapter = defaultdict(list)
    for qc in qcs:
        qc_by_chapter[qc.get("chapter_code") or "ORPHAN"].append(qc)

    # Precompute qi ids by intent
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
        label = ch.get("label")
        allow = ch.get("intent_allowlist") or []
        if code not in active_codes:
            continue

        # Qi posables "du chapitre" = union des qi_ids des intents allowlist vus
        chap_qi = set()
        for intent in allow:
            for qid in (qi_by_intent.get(intent) or []):
                if qid:
                    chap_qi.add(qid)

        # Couverture = union des qi_ids des QC mappées au chapitre
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
            "label": label,
            "qc_target": ch.get("qc_target", 15),
            "qc_count": qc_count,
            "qi_posable_count": qi_count,
            "covered_qi_count": covered_count,
            "coverage_pct": coverage_pct,
            "orphans": orphans,
            "status": "PASS" if (qi_count > 0 and coverage_pct == 100.0 and qc_count > 0) else "FAIL",
        }

    return report

def validate_B_blocking(
    chapters: List[Dict[str, Any]],
    qi_list: List[Dict[str, Any]],
    qcs: List[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    orphans_global = compute_orphans_posable(qi_list, qcs)
    chapter_report = build_chapter_report(chapters, qi_list, qcs)

    failures = []

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
# STREAMLIT UI (simple, lisible, orientée preuves)
# =============================================================================

def init_state():
    defaults = {
        "pack": None,
        "country": "FR",
        "level": None,
        "subjects": [],
        "harvested_pairs": [],            # list of dicts with bytes
        "harvest_manifest": None,
        "qi_global": [],
        "qi_stats": None,
        "qcs_global": [],
        "gte_stats": None,
        "iterations": [],
        "sealed": False,
        "last_B": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def sidebar_pack_view(pack: Dict[str, Any], level: Optional[str], subjects: List[str]):
    st.sidebar.markdown("## Activation Pays (ISO-PROD)")
    st.sidebar.caption("Pack visible (preuve de chargement)")

    st.sidebar.write(f"**Pack ID:** {pack.get('pack_id')}")
    st.sidebar.write(f"**Pays:** {pack.get('country_code')} — {pack.get('country_label')}")
    st.sidebar.write(f"**Version:** {pack.get('version')}")
    sig = pack.get("signature", {})
    st.sidebar.caption(f"Signature: {sig.get('hash')} / {sig.get('signed_at')}")

    acs = pack.get("academic_structure", {})
    st.sidebar.markdown("---")
    st.sidebar.write("### Niveaux")
    for lv in sorted(acs.get("levels", []), key=lambda x: x.get("order", 999)):
        st.sidebar.caption(f"• {lv.get('label')} ({lv.get('id')})")

    st.sidebar.write("### Matières")
    sb_labels = {s["id"]: s["label"] for s in acs.get("subjects", [])}
    if level:
        allowed = acs.get("level_subjects", {}).get(level, [])
        for s in allowed:
            st.sidebar.caption(f"• {sb_labels.get(s, s)} ({s})")
    else:
        for s in acs.get("subjects", []):
            st.sidebar.caption(f"• {s.get('label')} ({s.get('id')})")

    st.sidebar.markdown("---")
    st.sidebar.write("### Chapitres (pack-driven)")
    for ch in pack.get("chapters", [])[:12]:
        st.sidebar.caption(f"• {ch.get('chapter_code')} — {ch.get('label')}")
    if len(pack.get("chapters", [])) > 12:
        st.sidebar.caption("…")

    st.sidebar.markdown("---")
    st.sidebar.write("### Contexte sélectionné")
    st.sidebar.write(f"Niveau: **{level or '—'}**")
    st.sidebar.write(f"Matières: **{', '.join(subjects) if subjects else '—'}**")

def main():
    st.set_page_config(page_title="SMAXIA GTE V31.9.11", layout="wide")
    init_state()

    st.title("SMAXIA GTE Console V31.9.11 — ISO-PROD (Kernel-aligned)")
    st.caption("Flux : Activation → Sélection → Harvest AUTO → Extraction → Pipeline → Preuves (chapitre) → Saturation → Exports")

    tabs = st.tabs(["1) Activation & Sélection", "2) Harvest (AUTO)", "3) Extraction + Pipeline", "4) Preuves & Explorateur", "5) Exports"])

    # ----------------------------
    # TAB 1 — Activation & Sélection
    # ----------------------------
    with tabs[0]:
        st.header("Étape 1 — Activation Pays + Sélection Niveau/Matières")

        colA, colB = st.columns([1, 2])

        with colA:
            st.session_state.country = st.selectbox("Pays", ["FR", "CI"], index=0 if st.session_state.country == "FR" else 1)
            if st.button("ACTIVER CE PAYS", type="primary", use_container_width=True):
                pack = load_academic_pack(st.session_state.country)
                if not pack:
                    st.error("Pack introuvable.")
                else:
                    st.session_state.pack = pack
                    st.session_state.level = None
                    st.session_state.subjects = []
                    st.session_state.harvested_pairs = []
                    st.session_state.harvest_manifest = None
                    st.session_state.qi_global = []
                    st.session_state.qi_stats = None
                    st.session_state.qcs_global = []
                    st.session_state.gte_stats = None
                    st.session_state.iterations = []
                    st.session_state.sealed = False
                    st.session_state.last_B = None
                    st.success("Pack activé et visible dans la sidebar.")

        with colB:
            pack = st.session_state.pack
            if not pack:
                st.info("Cliquez sur ACTIVER CE PAYS.")
            else:
                acs = pack.get("academic_structure", {})
                levels = sorted(acs.get("levels", []), key=lambda x: x.get("order", 999))
                level_ids = [l["id"] for l in levels]
                level_labels = {l["id"]: l["label"] for l in levels}

                st.subheader("Sélection (obligatoire)")
                st.session_state.level = st.radio(
                    "Niveau",
                    level_ids,
                    horizontal=True,
                    format_func=lambda x: level_labels.get(x, x),
                )

                subject_labels = {s["id"]: s["label"] for s in acs.get("subjects", [])}
                allowed = acs.get("level_subjects", {}).get(st.session_state.level, [])

                st.session_state.subjects = st.multiselect(
                    "Matières (cocher 1–2 pour le test)",
                    options=allowed,
                    default=allowed[:1] if allowed else [],
                    format_func=lambda x: subject_labels.get(x, x),
                )

                if st.session_state.subjects:
                    st.success(f"Sélection : {level_labels.get(st.session_state.level)} / " +
                               ", ".join(subject_labels.get(s, s) for s in st.session_state.subjects))
                else:
                    st.warning("Sélectionnez au moins une matière.")

    # Sidebar pack view
    if st.session_state.pack:
        sidebar_pack_view(st.session_state.pack, st.session_state.level, st.session_state.subjects)

    # ----------------------------
    # TAB 2 — Harvest AUTO
    # ----------------------------
    with tabs[1]:
        st.header("Étape 2 — Harvest AUTO (pêche aveugle)")

        if not st.session_state.pack:
            st.warning("Activez d'abord un pays.")
        elif not st.session_state.level or not st.session_state.subjects:
            st.warning("Sélectionnez Niveau + Matières.")
        else:
            pack = st.session_state.pack
            volume = st.number_input("Volume (nb de sujets à récupérer par matière)", min_value=5, value=20, step=5)
            st.caption("Le Harvest démarre depuis pack.harvest_config.sources (zéro hardcode). Progression visible pour validation ISO-PROD.")

            progress = st.progress(0.0)
            status = st.empty()

            def progress_cb(cur, total, url):
                progress.progress(cur / max(1, total), text=f"{cur}/{total} — {url.split('/')[-1][:60]}")

            if st.button("LANCER AUTO-HARVEST", type="primary", use_container_width=True):
                all_pairs = []
                all_manifests = []
                progress.progress(0.0)
                status.info("Harvest en cours…")

                for subj in st.session_state.subjects:
                    key = f"{st.session_state.level}|{subj}"
                    status.info(f"Harvest scope: {key}")
                    pairs, manifest = harvest_subjects(pack, st.session_state.level, subj, int(volume), progress_cb)
                    all_pairs.extend(pairs)
                    all_manifests.append(manifest)
                    time.sleep(0.1)

                progress.progress(1.0)
                status.empty()

                st.session_state.harvested_pairs = all_pairs
                st.session_state.harvest_manifest = {
                    "country": st.session_state.country,
                    "level": st.session_state.level,
                    "subjects": st.session_state.subjects,
                    "manifests": all_manifests,
                    "total_pairs": len(all_pairs),
                    "generated_at": datetime.now().isoformat(),
                }

                if not all_pairs:
                    st.error("Aucun sujet récupéré. Vérifiez pack.harvest_config.sources, ou utilisez l'upload manuel (onglet Exports).")
                else:
                    st.success(f"Harvest terminé : {len(all_pairs)} paires sujet/corrigé (corrigé présent si trouvé).")

            # Visualisation Import PDF (liste des sujets/corrigés)
            if st.session_state.harvested_pairs:
                st.subheader("Import PDF (auto) — Sujets & Corrigés chargés")
                rows = []
                for p in st.session_state.harvested_pairs[:200]:
                    rows.append({
                        "pair_id": p.get("pair_id"),
                        "scope": p.get("scope_key"),
                        "sujet": p.get("label"),
                        "corrigé?": "✅" if p.get("correction_bytes") else "❌",
                    })
                st.dataframe(rows, use_container_width=True, height=320)
                st.caption("Limite affichage : 200 lignes (export complet disponible).")

    # ----------------------------
    # TAB 3 — Extraction + Pipeline
    # ----------------------------
    with tabs[2]:
        st.header("Étape 3 — Extraction Qi/RQi puis Pipeline GTE")

        if not st.session_state.harvested_pairs:
            st.warning("Lancez d'abord le Harvest AUTO (onglet 2) ou chargez en manuel (onglet 5).")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("EXTRAIRE Qi/RQi", type="primary", use_container_width=True):
                    qi, qi_stats = extract_qi_rqi(st.session_state.harvested_pairs)

                    # Merge by qi_id
                    existing = {q["qi_id"] for q in st.session_state.qi_global}
                    new_only = [q for q in qi if q.get("qi_id") not in existing]
                    st.session_state.qi_global.extend(new_only)
                    st.session_state.qi_stats = qi_stats | {
                        "new_qi_added": len(new_only),
                        "qi_global_total": len(st.session_state.qi_global),
                    }
                    st.success(f"Extraction OK : +{len(new_only)} Qi (global={len(st.session_state.qi_global)}).")

            with col2:
                if st.button("RUN GTE (QC/ARI/FRT/TRG)", type="primary", use_container_width=True):
                    if not st.session_state.pack:
                        st.error("Pack manquant.")
                    else:
                        chapters = st.session_state.pack.get("chapters", [])
                        qcs, gte_stats = run_pipeline_gte(st.session_state.qi_global, chapters)
                        st.session_state.qcs_global = qcs
                        st.session_state.gte_stats = gte_stats

                        # Validateurs B (bloquants)
                        okB, B = validate_B_blocking(chapters, st.session_state.qi_global, st.session_state.qcs_global)
                        st.session_state.last_B = B

                        st.success(f"GTE OK : QC={len(qcs)} | Qi_posable={gte_stats.get('qi_posable')} | Singletons={gte_stats.get('singletons')}")
                        if okB:
                            st.success("VALIDATEURS B : PASS (bloquants)")
                        else:
                            st.error("VALIDATEURS B : FAIL (bloquants) — voir onglet 4 (Preuves).")

            # Résumé
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Qi global", len(st.session_state.qi_global))
            c2.metric("Qi POSABLE", sum(1 for q in st.session_state.qi_global if q.get("has_rqi")))
            c3.metric("QC", len(st.session_state.qcs_global))
            c4.metric("SEALED", "YES" if st.session_state.sealed else "NO")

            if st.session_state.qi_stats:
                with st.expander("Détails extraction", expanded=False):
                    st.json(st.session_state.qi_stats)
            if st.session_state.gte_stats:
                with st.expander("Détails pipeline GTE", expanded=False):
                    st.json(st.session_state.gte_stats)

    # ----------------------------
    # TAB 4 — Preuves & Explorateur + Saturation
    # ----------------------------
    with tabs[3]:
        st.header("Étape 4 — Preuves (chapitre) + Explorateur + Saturation")

        if not st.session_state.pack or not st.session_state.qcs_global:
            st.info("Exécutez Extraction + RUN GTE.")
        else:
            chapters = st.session_state.pack.get("chapters", [])
            # Preuves & validateurs B
            okB, B = validate_B_blocking(chapters, st.session_state.qi_global, st.session_state.qcs_global)
            st.session_state.last_B = B

            st.subheader("Validateurs B (BLOQUANTS)")
            if okB:
                st.success("PASS — aucun orphelin posable, preuve 100% sur chapitres actifs.")
            else:
                st.error("FAIL — au moins une violation bloquante.")
                with st.expander("Détails FAIL", expanded=True):
                    st.json(B.get("failures"))

            st.subheader("Chapter Report (preuve)")
            chapter_report = B.get("chapter_report") or {}
            st.json(chapter_report)

            # Saturation loop (contrôlée)
            st.markdown("---")
            st.subheader("Boucle Saturation (ISO-PROD)")
            st.caption("Arrêt si (new_proof_sig == 0) ET validateurs B PASS. Sinon : augmenter volume et recommencer Harvest+Extraction+RUN GTE.")

            colS1, colS2 = st.columns([1, 2])
            with colS1:
                max_iter = st.number_input("Max itérations", min_value=1, value=4, step=1)
                base_volume = st.number_input("Volume initial (par matière)", min_value=5, value=20, step=5)
                incr = st.number_input("Incrément volume", min_value=5, value=30, step=5)

            with colS2:
                st.info("La saturation est basée sur les signatures de preuve (qc_proof_sig), pas sur qc_id. Anti-faux SEALED.")
                if st.button("LANCER SATURATION (AUTO)", type="primary", use_container_width=True):
                    if not st.session_state.pack or not st.session_state.level or not st.session_state.subjects:
                        st.error("Activation + sélection requises.")
                    else:
                        st.session_state.sealed = False
                        st.session_state.iterations = []

                        current_volume = int(base_volume)
                        prev_qcs = st.session_state.qcs_global[:]  # snapshot initial

                        for it in range(1, int(max_iter) + 1):
                            if st.session_state.sealed:
                                break

                            st.write(f"### Itération {it} — volume={current_volume}")
                            prog = st.progress(0.0)
                            msg = st.empty()

                            def pcb(cur, total, url):
                                prog.progress(cur / max(1, total), text=f"{cur}/{total} — {url.split('/')[-1][:60]}")

                            # Harvest per subject
                            all_pairs = []
                            all_manifests = []
                            for subj in st.session_state.subjects:
                                msg.info(f"Harvest {st.session_state.level}|{subj} …")
                                pairs, manifest = harvest_subjects(st.session_state.pack, st.session_state.level, subj, current_volume, pcb)
                                all_pairs.extend(pairs)
                                all_manifests.append(manifest)

                            prog.empty()
                            msg.empty()

                            if not all_pairs:
                                st.warning("0 pairs récoltées. Arrêt.")
                                break

                            # Extraction merge
                            qi_new, qi_stats = extract_qi_rqi(all_pairs)
                            existing = {q["qi_id"] for q in st.session_state.qi_global}
                            new_only = [q for q in qi_new if q.get("qi_id") not in existing]
                            st.session_state.qi_global.extend(new_only)

                            # Pipeline
                            qcs, gte_stats = run_pipeline_gte(st.session_state.qi_global, st.session_state.pack.get("chapters", []))
                            st.session_state.qcs_global = qcs

                            # Validate B
                            okB2, B2 = validate_B_blocking(st.session_state.pack.get("chapters", []), st.session_state.qi_global, st.session_state.qcs_global)
                            st.session_state.last_B = B2

                            # Saturation delta on proof_sig
                            new_count, added_sigs = saturation_delta(prev_qcs, qcs)

                            st.session_state.iterations.append({
                                "iter": it,
                                "volume": current_volume,
                                "pairs": len(all_pairs),
                                "new_qi": len(new_only),
                                "qc_total": len(qcs),
                                "new_proof_sig": new_count,
                                "B_ok": okB2,
                                "timestamp": datetime.now().isoformat(),
                            })

                            st.dataframe(st.session_state.iterations, use_container_width=True)

                            if new_count == 0 and okB2:
                                st.session_state.sealed = True
                                st.success("SEALED — new_proof_sig=0 ET validateurs B PASS.")
                                break
                            else:
                                if new_count == 0 and not okB2:
                                    st.error("new_proof_sig=0 MAIS validateurs B FAIL -> interdit de sceller.")
                                else:
                                    st.warning(f"Non scellé : new_proof_sig={new_count} | B_ok={okB2}")

                            prev_qcs = qcs[:]  # update snapshot
                            current_volume += int(incr)

                        if not st.session_state.sealed:
                            st.warning("Saturation non atteinte (ou preuves insuffisantes) dans la limite d'itérations.")

            # Explorateur QC par chapitre
            st.markdown("---")
            st.subheader("Explorateur — QC par chapitre")
            by_ch = defaultdict(list)
            for qc in st.session_state.qcs_global:
                by_ch[qc.get("chapter_code") or "ORPHAN"].append(qc)

            # chapter label map
            ch_label = {c.get("chapter_code"): c.get("label") for c in st.session_state.pack.get("chapters", [])}

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
                        st.json((qc.get("qi_ids") or [])[:20])

    # ----------------------------
    # TAB 5 — Exports + Upload manuel fallback
    # ----------------------------
    with tabs[4]:
        st.header("Étape 5 — Exports (preuves) + Upload manuel fallback")

        # Exports
        st.subheader("Exports")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.download_button(
                "qi_pack.json",
                canonical_json(st.session_state.qi_global),
                "qi_pack.json",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "qc_pack.json",
                canonical_json(st.session_state.qcs_global),
                "qc_pack.json",
                use_container_width=True
            )
        with c3:
            chapter_report = (st.session_state.last_B or {}).get("chapter_report") or {}
            st.download_button(
                "chapter_report.json",
                canonical_json(chapter_report),
                "chapter_report.json",
                use_container_width=True
            )
        with c4:
            st.download_button(
                "harvest_manifest.json",
                canonical_json(st.session_state.harvest_manifest or {}),
                "harvest_manifest.json",
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("Upload manuel (fallback si Harvest AUTO indisponible)")
        st.caption("Permet de tester Extraction+GTE avec PDFs locaux, sans internet. N'affecte pas le core.")

        subj_files = st.file_uploader("PDF Sujets (énoncés)", type=["pdf"], accept_multiple_files=True, key="manual_subj")
        corr_files = st.file_uploader("PDF Corrigés", type=["pdf"], accept_multiple_files=True, key="manual_corr")

        if subj_files and st.button("CHARGER EN MANUEL -> pairs", type="secondary", use_container_width=True):
            # Pairing simple par nom (best effort)
            corr_map = {}
            if corr_files:
                for cf in corr_files:
                    corr_map[cf.name] = cf

            pairs = []
            for i, sf in enumerate(subj_files, start=1):
                sb = sf.read()
                sf.seek(0)
                cb = None

                # tentative : corrige <-> enonce
                if corr_files:
                    sbase = sf.name.lower()
                    for cf in corr_files:
                        cname = cf.name.lower()
                        if ("corrig" in cname or "correc" in cname) and (
                            re.sub(r"(corrig[eé]|correction|corrige)", "", cname)[:25] in re.sub(r"(enonce|énoncé)", "", sbase)[:25]
                            or re.sub(r"(enonce|énoncé)", "", sbase)[:25] in re.sub(r"(corrig[eé]|correction|corrige)", "", cname)[:25]
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

            st.session_state.harvested_pairs = pairs
            st.session_state.harvest_manifest = {
                "mode": "MANUAL",
                "total_pairs": len(pairs),
                "generated_at": datetime.now().isoformat(),
            }
            st.success(f"Chargé : {len(pairs)} paires (manuel). Retournez à l’onglet 3 pour Extraction+RUN GTE.")


if __name__ == "__main__":
    main()
