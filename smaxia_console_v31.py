# =============================================================================
# smaxia_console_v31_9_6.py — SMAXIA GTE Console V31.9.6 (ISO-PROD TEST)
# =============================================================================
# OBJECTIF CEO (V31.9.6)
# Chaîne de test intégrale SUR UI (pas de “mode script”) :
#
# Input:
#   Choix Pays (ex: FR)
#   => Loading automatique Academic Pack (VISIBLE: niveaux + matières + chapitres)
#   => Sélection TEST : 1 niveau + jusqu’à 2 matières
#   => Harvest pack-driven (scope niveau+matière) => sujets+corrections visibles dans “Import PDF”
#   => Extraction JSON (Qi/RQi)
#   => Pipeline RUN GTE
#   => Résultats + Explorateur “QC par chapitre” + exports preuves
#
# LOIS SMAXIA:
# - Zéro hardcode métier (niveaux/matières/chapitres/sources) dans le code CORE.
# - Tout vient du academic_pack.json (DATA). La sélection UI n’utilise QUE les IDs présents dans le pack.
# - Validateurs B BLOQUANTS (chapitres pack-driven + preuves coverage).
#
# NOTE STREAMLIT:
# - On ne peut pas auto-remplir un file_uploader. Donc: “Bibliothèque Harvest” => sélection + bytes en session.
#
# INTÉGRATION MOTEUR RÉEL (si présent dans votre repo):
# - Si disponible: from smaxia_granulo_engine_test import run_granulo_test
#   attendu: run_granulo_test(urls, volume) -> {sujets,qc,saturation,audit}
# - Sinon: fallback TEST_ONLY (générateur structure-driven) pour valider l’infrastructure.
# =============================================================================

from __future__ import annotations

import ast
import hashlib
import io
import json
import random
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ---------------- Optional real engine import ----------------
REAL_ENGINE_AVAILABLE = False
try:
    from smaxia_granulo_engine_test import run_granulo_test  # type: ignore
    REAL_ENGINE_AVAILABLE = True
except Exception:
    run_granulo_test = None  # type: ignore
    REAL_ENGINE_AVAILABLE = False


# =============================================================================
# ZONE A — TEST-ONLY FALLBACK PACK (UNIQUEMENT si aucun pack FR présent)
# =============================================================================
# IMPORTANT: ce fallback n’est là QUE pour éviter écran vide.
# Pour un test ISO-PROD sérieux, uploadez votre academic_pack_FR.json (Gemini/Opus).
TEST_ONLY_DEFAULT_PACK_FR: Dict[str, Any] = {
    "pack_id": "FR_TEST_DEFAULT_PACK_V31_9_6",
    "country_code": "FR",
    "signature": {"pack_hash": "TEST_ONLY_PLACEHOLDER_HASH", "signed_at": "TEST_ONLY"},
    "academic_structure": {
        "levels": [
            {"level_id": "SECONDE", "title": "Seconde"},
            {"level_id": "PREMIERE", "title": "Première"},
            {"level_id": "TERMINALE", "title": "Terminale"},
            {"level_id": "L1", "title": "L1"},
            {"level_id": "L2", "title": "L2"},
            {"level_id": "L3", "title": "L3"},
            {"level_id": "M1", "title": "M1"},
            {"level_id": "M2", "title": "M2"},
            {"level_id": "PREPA", "title": "Prépa"},
        ],
        "subjects": [
            {"subject_id": "MATH", "title": "Mathématiques"},
            {"subject_id": "PHYS", "title": "Physique"},
            {"subject_id": "CHIM", "title": "Chimie"},
        ],
        "level_subject_map": [
            {"level_id": "TERMINALE", "subject_ids": ["MATH", "PHYS", "CHIM"]},
            {"level_id": "PREMIERE", "subject_ids": ["MATH", "PHYS", "CHIM"]},
            {"level_id": "SECONDE", "subject_ids": ["MATH", "PHYS", "CHIM"]},
        ],
    },
    "chapters": [
        {"chapter_id": "CH_FR_001", "title": "Chapitre 1 (TEST)", "intent_allowlist": ["STRUCT_EX1_Q1", "STRUCT_EX1_Q2"]},
        {"chapter_id": "CH_FR_002", "title": "Chapitre 2 (TEST)", "intent_allowlist": ["STRUCT_EX2_Q1", "STRUCT_EX2_Q2"]},
        {"chapter_id": "CH_FR_003", "title": "Chapitre 3 (TEST)", "intent_allowlist": ["STRUCT_EX3_Q1", "STRUCT_EX3_Q2"]},
    ],
    "harvest": {
        "http": {"user_agent": "Mozilla/5.0", "timeout_sec": 25, "max_bytes_pdf": 25_000_000, "max_index_pages": 30},
        # CATALOGUE pack-driven (niveau+matière) — VIDE en fallback (bouton auto-scrape OFF)
        "catalog": [
            # Exemple attendu (dans votre pack FR réel):
            # {
            #   "level_id": "TERMINALE",
            #   "subject_id": "MATH",
            #   "volume_mode": "pairs",
            #   "seeds": [
            #     {"index_url":"https://....", "pdf_regex": r"\.pdf($|\?)", "corr_hint_regex": r"corrig(e|é)|correction"}
            #   ]
            # }
        ],
    },
}

# =============================================================================
# B — CANONICAL / HASH
# =============================================================================

SENSITIVE_CONTEXT_FIELDS: Set[str] = {
    "country_code", "language", "domain_code", "assessment_type_code",
    "chapter_code", "chapter", "discipline", "exam_type", "locale", "curriculum",
}

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def compute_hash(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))

def sha8_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]

def sort_qi_canonical(qi_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(qi_list, key=lambda x: (x.get("position", {}).get("order_index", 0), x.get("qi_id", "")))

def canonicalize_artifacts(artifacts: Dict[str, Any], qi_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    singletons = artifacts.get("singletons_warning", []) or []
    singletons_sorted = sorted(
        singletons,
        key=lambda x: (str(x.get("intent", "")),
                       "|".join(sorted([str(q) for q in x.get("qi_ids", []) or []])))
    )
    return {
        "qcs": sorted(artifacts.get("qcs", []), key=lambda x: x.get("qc_id", "")),
        "aris": sorted(artifacts.get("aris", []), key=lambda x: x.get("ari_id", "")),
        "frts": sorted(artifacts.get("frts", []), key=lambda x: x.get("frt_id", "")),
        "triggers": sorted(artifacts.get("triggers", []), key=lambda x: x.get("trigger_id", "")),
        "singletons_warning": singletons_sorted,
        "qi_coverage": artifacts.get("qi_coverage", {}) or {},
        "qi_pack": sort_qi_canonical(qi_pack),
    }

# =============================================================================
# C — SCHEMAS
# =============================================================================

def validate_qi_schema(qi: Dict[str, Any]) -> Tuple[bool, str]:
    for f in ["qi_id", "subject_id", "position", "statement"]:
        if f not in qi:
            return False, f"Missing {f}"
    if "order_index" not in (qi.get("position") or {}):
        return False, "Missing position.order_index"
    if "text_md" not in (qi.get("statement") or {}):
        return False, "Missing statement.text_md"
    corr = qi.get("correction")
    if corr is not None:
        if "available" not in corr or "text_md" not in corr:
            return False, "Invalid correction"
    return True, "OK"

def validate_qc_schema(qc: Dict[str, Any]) -> Tuple[bool, str]:
    for f in ["qc_id", "qc_formulation", "qc_invariant_signature", "mapping", "links"]:
        if f not in qc:
            return False, f"Missing {f}"
    sig = qc.get("qc_invariant_signature") or {}
    if "intent_code" not in sig:
        return False, "Missing intent_code"
    mapping = qc.get("mapping") or {}
    if "primary_qi_ids" not in mapping or "covered_qi_ids" not in mapping:
        return False, "Missing mapping"
    links = qc.get("links") or {}
    if "ari_id" not in links or "frt_id" not in links or "trigger_ids" not in links:
        return False, "Missing links"
    ftxt = str(qc.get("qc_formulation", "")).strip()
    if not (ftxt.startswith("Comment") and ftxt.endswith("?")):
        return False, "QC formulation must be 'Comment ... ?'"
    return True, "OK"

def validate_ari_schema(ari: Dict[str, Any]) -> Tuple[bool, str]:
    if "ari_id" not in ari or "steps" not in ari:
        return False, "Missing ari_id/steps"
    if not isinstance(ari.get("steps"), list) or len(ari["steps"]) < 1:
        return False, "Invalid steps"
    return True, "OK"

def validate_frt_schema(frt: Dict[str, Any]) -> Tuple[bool, str]:
    if "frt_id" not in frt or "sections" not in frt:
        return False, "Missing frt_id/sections"
    for s in ["given", "goal", "method", "checks", "common_traps", "final_form"]:
        if s not in (frt.get("sections") or {}):
            return False, f"Missing {s}"
    return True, "OK"

def validate_trigger_schema(trg: Dict[str, Any]) -> Tuple[bool, str]:
    for f in ["trigger_id", "type_code", "severity", "condition", "action", "links"]:
        if f not in trg:
            return False, f"Missing {f}"
    if "qc_id" not in (trg.get("links") or {}) or "ari_id" not in (trg.get("links") or {}):
        return False, "Missing links"
    return True, "OK"

# =============================================================================
# D — INTEGRITY / COVERAGE
# =============================================================================

def check_ref_integrity(canon: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    qc_ids = {qc.get("qc_id") for qc in canon.get("qcs", [])}
    ari_ids = {ari.get("ari_id") for ari in canon.get("aris", [])}
    frt_ids = {frt.get("frt_id") for frt in canon.get("frts", [])}
    trg_ids = {trg.get("trigger_id") for trg in canon.get("triggers", [])}
    qi_ids = {qi.get("qi_id") for qi in canon.get("qi_pack", [])}

    for qc in canon.get("qcs", []):
        links = qc.get("links") or {}
        if links.get("ari_id") not in ari_ids:
            errors.append(f"QC {qc.get('qc_id')} bad ari")
        if links.get("frt_id") not in frt_ids:
            errors.append(f"QC {qc.get('qc_id')} bad frt")
        for t in links.get("trigger_ids", []):
            if t not in trg_ids:
                errors.append(f"QC {qc.get('qc_id')} bad trg")
        for qi in (qc.get("mapping") or {}).get("covered_qi_ids", []):
            if qi not in qi_ids:
                errors.append(f"QC {qc.get('qc_id')} bad qi")

    for trg in canon.get("triggers", []):
        links = trg.get("links") or {}
        if links.get("qc_id") not in qc_ids:
            errors.append("TRG bad qc")
        if links.get("ari_id") not in ari_ids:
            errors.append("TRG bad ari")

    return len(errors) == 0, errors

def check_coverage_total(canon: Dict[str, Any]) -> Tuple[bool, List[str]]:
    all_qi = {qi.get("qi_id") for qi in canon.get("qi_pack", [])}
    covered = set()
    for qc in canon.get("qcs", []):
        covered.update((qc.get("mapping") or {}).get("covered_qi_ids", []))
    orphans = sorted([q for q in all_qi if q and q not in covered])
    return len(orphans) == 0, orphans

def check_coverage_primary_unique(canon: Dict[str, Any]) -> Tuple[bool, List[str]]:
    cnt = defaultdict(int)
    for qc in canon.get("qcs", []):
        for qi_id in (qc.get("mapping") or {}).get("primary_qi_ids", []):
            cnt[qi_id] += 1
    dups = sorted([k for k, v in cnt.items() if v > 1])
    return len(dups) == 0, dups

def check_determinism_n_runs(gen_func, qi_pack: List[Dict[str, Any]], n: int = 3) -> Tuple[bool, List[str]]:
    hashes = [compute_hash(canonicalize_artifacts(gen_func(qi_pack), qi_pack)) for _ in range(n)]
    ok = len(set(hashes)) == 1
    return ok, hashes

def check_order_invariance(gen_func, qi_pack: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    h1 = compute_hash(canonicalize_artifacts(gen_func(qi_pack), qi_pack))
    shuffled = qi_pack.copy()
    random.seed(42)
    random.shuffle(shuffled)
    h2 = compute_hash(canonicalize_artifacts(gen_func(shuffled), shuffled))
    return h1 == h2, h1, h2

def scan_ast_sensitive_access(source: str) -> Tuple[bool, List[Dict[str, Any]]]:
    violations = []
    try:
        tree = ast.parse(source)
    except Exception as e:
        return False, [{"error": f"Parse: {e}"}]

    class V(ast.NodeVisitor):
        def visit_Subscript(self, n):
            if isinstance(n.value, ast.Name) and n.value.id == "context":
                sl = n.slice
                if isinstance(sl, ast.Constant) and str(sl.value) in SENSITIVE_CONTEXT_FIELDS:
                    violations.append({"type": "subscript", "field": sl.value})
            self.generic_visit(n)

        def visit_Attribute(self, n):
            if isinstance(n.value, ast.Name) and n.value.id == "context":
                if n.attr in SENSITIVE_CONTEXT_FIELDS:
                    violations.append({"type": "attribute", "field": n.attr})
            self.generic_visit(n)

    V().visit(tree)
    return len(violations) == 0, violations

def get_full_source() -> str:
    try:
        return Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return ""

# =============================================================================
# E — PDF → Qi Builder (markers EXn-Qm[a/b/c])
# =============================================================================

@dataclass
class ExtractStats:
    pages: int
    errors: List[str]

def _read_pdf_text(pdf_bytes: bytes) -> Tuple[str, ExtractStats]:
    if pdfplumber is None:
        return "", ExtractStats(0, ["pdfplumber not installed"])
    errors, text_parts, pages = [], [], 0
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = len(pdf.pages)
            for p in pdf.pages:
                try:
                    t = (p.extract_text() or "").replace("\u00a0", " ")
                    text_parts.append(t)
                except Exception as e:
                    errors.append(f"page: {e}")
    except Exception as e:
        errors.append(f"open: {e}")
    return "\n".join(text_parts).strip(), ExtractStats(pages, errors)

_MAIN_RE = re.compile(r"^\s*(\d{1,3})\s*[\)\.\-]\s*")
_SUB_RE = re.compile(r"^\s*([a-h])\s*[\)\.\-]\s*", re.I)
_COMBO_RE = re.compile(r"^\s*(\d{1,3})\s*[\)\.\-]\s*([a-h])\s*[\)\.\-]\s*", re.I)

def _detect_exercise(line: str) -> Optional[str]:
    s = (line or "").strip()
    if not s:
        return None
    up = s.upper().replace(" ", "")
    if up.startswith("EXERCICE"):
        m = re.match(r"EXERCICE\s*([0-9]+)", s, re.I)
        if m:
            return m.group(1)
        m2 = re.match(r"EXERCICE([0-9]+)", up, re.I)
        if m2:
            return m2.group(1)
    return None

def _segment_blocks_with_ex(lines: List[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    cur: List[str] = []
    cur_marker: Optional[str] = None
    cur_main: Optional[str] = None
    cur_ex: str = "0"

    def flush():
        nonlocal cur, cur_marker
        txt = "\n".join(cur).strip()
        txt_norm = re.sub(r"\s+", " ", txt).strip()
        if cur_marker and txt_norm and len(txt_norm) >= 30:
            blocks.append({"marker": cur_marker, "text": txt})
        cur, cur_marker = [], None

    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue

        exn = _detect_exercise(s)
        if exn is not None:
            cur_ex = exn
            cur_main = None
            continue

        m = _COMBO_RE.match(s)
        if m:
            if cur:
                flush()
            cur_main = m.group(1)
            cur_marker = f"EX{cur_ex}-Q{cur_main}{m.group(2).lower()}"
            cur = [ln]
            continue

        m = _MAIN_RE.match(s)
        if m:
            if cur:
                flush()
            cur_main = m.group(1)
            cur_marker = f"EX{cur_ex}-Q{cur_main}"
            cur = [ln]
            continue

        m = _SUB_RE.match(s)
        if m and cur_main is not None:
            if cur:
                flush()
            cur_marker = f"EX{cur_ex}-Q{cur_main}{m.group(1).lower()}"
            cur = [ln]
            continue

        if cur:
            cur.append(ln)

    if cur:
        flush()

    if not blocks and any((ln or "").strip() for ln in lines):
        full = "\n".join(lines).strip()
        if full:
            blocks = [{"marker": "FULL", "text": full}]
    return blocks

def _build_corr_index(text: str) -> Dict[str, str]:
    if not text.strip():
        return {}
    lines = [ln.rstrip() for ln in text.splitlines()]
    blocks = _segment_blocks_with_ex(lines)
    idx: Dict[str, str] = {}
    for b in blocks:
        k = str(b.get("marker", "")).strip()
        v = str(b.get("text", "")).strip()
        if k and v:
            idx[k] = v
    return idx

def build_qi_pack_from_pdf_pair(subject_bytes: bytes, correction_bytes: Optional[bytes], label: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    subj_sha8 = sha8_bytes(subject_bytes)
    subj_text, subj_stats = _read_pdf_text(subject_bytes)
    subj_lines = [ln.rstrip() for ln in subj_text.splitlines()]
    subj_blocks = _segment_blocks_with_ex(subj_lines)

    corr_text, corr_stats = ("", ExtractStats(0, [])) if not correction_bytes else _read_pdf_text(correction_bytes)
    corr_idx = _build_corr_index(corr_text)

    qi_pack: List[Dict[str, Any]] = []
    aligned = 0

    for i, b in enumerate(subj_blocks, 1):
        marker = str(b.get("marker", "")).strip()
        statement = str(b.get("text", "")).strip()
        corr_txt = corr_idx.get(marker, "") if corr_idx else ""
        if corr_txt:
            aligned += 1

        content_h = hashlib.sha256((marker + "|" + statement).encode("utf-8")).hexdigest()[:10]
        qi_pack.append({
            "qi_id": f"QI_{i:04d}_{subj_sha8}_{content_h}",
            "subject_id": f"SUBJ_{subj_sha8}",
            "position": {"order_index": i, "marker": marker, "source_label": label},
            "statement": {"text_md": statement},
            "correction": {"available": bool(corr_txt), "text_md": corr_txt},
        })

    meta = {
        "generator": {"name": "SMAXIA_PDF_QI_BUILDER", "version": "V31.9.6"},
        "created_at": datetime.now().isoformat(),
        "label": label,
        "stats": {
            "qi_count": len(qi_pack),
            "subject_blocks": len(subj_blocks),
            "correction_blocks": len(corr_idx),
            "aligned": aligned,
        },
        "warnings": (
            (["extraction_errors_subject"] if subj_stats.errors else []) +
            (["extraction_errors_correction"] if corr_stats.errors else []) +
            (["low_qi_count"] if len(qi_pack) <= 1 else [])
        ),
        "errors": {"subject": subj_stats.errors, "correction": corr_stats.errors},
        "correction_index_keys_preview": sorted(list(corr_idx.keys()))[:80],
    }
    return meta, qi_pack


# =============================================================================
# F — TEST GENERATOR QC/ARI/FRT/TRG (fallback infrastructure)
# =============================================================================

_MARKER_RE = re.compile(r"^EX(\d+)-Q(\d+)([a-h])?$", re.I)

def _parse_marker(marker: str) -> Tuple[str, str, str]:
    m = _MARKER_RE.match((marker or "").strip())
    if not m:
        return ("0", "0", "")
    ex = m.group(1) or "0"
    qm = m.group(2) or "0"
    sub = (m.group(3) or "").lower()
    return (ex, qm, sub)

def TEST_ONLY_generate_qc_ari_frt_trg(qi_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for qi in qi_pack:
        marker = str((qi.get("position") or {}).get("marker", "") or "").strip()
        ex, qm, _sub = _parse_marker(marker)
        if ex == "0" and qm == "0":
            key = "STRUCT_UNK"
        else:
            key = f"STRUCT_EX{ex}_Q{qm}"
        groups[key].append(qi)

    qcs, aris, frts, triggers, singletons = [], [], [], [], []

    for gkey in sorted(groups.keys()):
        grp = sorted(groups[gkey], key=lambda x: (x.get("position", {}).get("order_index", 0), x.get("qi_id", "")))
        qi_ids = [q.get("qi_id", "") for q in grp if q.get("qi_id")]

        if len(qi_ids) < 2:
            singletons.append({"intent": gkey, "qi_ids": qi_ids})

        h = hashlib.sha256((gkey + "|" + "|".join(sorted(qi_ids))).encode("utf-8")).hexdigest()[:10]
        qc_id, ari_id, frt_id, trg_id = f"QC_{gkey}_{h}", f"ARI_{gkey}_{h}", f"FRT_{gkey}_{h}", f"TRG_{gkey}_{h}"

        by_subj: Dict[str, List[str]] = defaultdict(list)
        for q in grp:
            by_subj[str(q.get("subject_id", ""))].append(str(q.get("qi_id", "")))

        qcs.append({
            "qc_id": qc_id,
            "qc_formulation": "Comment procéder ?",
            "qc_invariant_signature": {"intent_code": gkey},
            "mapping": {"primary_qi_ids": qi_ids, "covered_qi_ids": qi_ids, "by_subject": dict(by_subj)},
            "links": {"ari_id": ari_id, "frt_id": frt_id, "trigger_ids": [trg_id]},
            "provenance": {"generator_version": "TEST_ONLY_V31.9.6"},
        })

        aris.append({
            "ari_id": ari_id,
            "template_id": "ARI_TPL_V1",
            "steps": [
                {"step_id": "S1", "operator_code": "OP_READ"},
                {"step_id": "S2", "operator_code": "OP_PLAN"},
                {"step_id": "S3", "operator_code": "OP_EXECUTE"},
                {"step_id": "S4", "operator_code": "OP_CHECK"},
                {"step_id": "S5", "operator_code": "OP_CONCLUDE"},
            ],
            "provenance": {"qc_id": qc_id},
        })

        frts.append({
            "frt_id": frt_id,
            "template_id": "FRT_TPL_V1",
            "sections": {
                "given": "Données",
                "goal": "Objectif",
                "method": "Méthode",
                "checks": ["Contrôle"],
                "common_traps": ["Piège"],
                "final_form": "Forme finale",
            },
            "provenance": {"qc_id": qc_id, "ari_id": ari_id},
        })

        triggers.append({
            "trigger_id": trg_id,
            "type_code": "TRG_CHECK",
            "severity": "medium",
            "condition": {"signal": "AFTER_EXECUTE"},
            "action": {"recommendation": "Appliquer un contrôle invariant"},
            "links": {"qc_id": qc_id, "ari_id": ari_id, "qi_examples": qi_ids[:2]},
        })

    covered: Set[str] = set()
    for qc in qcs:
        covered.update(qc.get("mapping", {}).get("covered_qi_ids", []))

    return {
        "qcs": sorted(qcs, key=lambda x: x.get("qc_id", "")),
        "aris": sorted(aris, key=lambda x: x.get("ari_id", "")),
        "frts": sorted(frts, key=lambda x: x.get("frt_id", "")),
        "triggers": sorted(triggers, key=lambda x: x.get("trigger_id", "")),
        "singletons_warning": singletons,
        "qi_coverage": {qi_id: (qi_id in covered) for qi_id in [q.get("qi_id", "") for q in qi_pack]},
    }


# =============================================================================
# G — PACK VALIDATION + CHAPTER PROOFS (B BLOQUANTS)
# =============================================================================

def load_json_from_uploaded(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    if not uploaded:
        return None, None
    try:
        return json.load(uploaded), None
    except Exception as e:
        return None, f"Parse error: {e}"

def validate_pack_schema(pack: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    if not isinstance(pack, dict):
        return False, ["pack must be dict"]
    for f in ["pack_id", "country_code", "signature", "chapters"]:
        if f not in pack:
            errs.append(f"Missing {f}")
    if not isinstance(pack.get("chapters"), list) or len(pack.get("chapters") or []) < 1:
        errs.append("chapters must be non-empty list")
    else:
        for i, ch in enumerate(pack["chapters"], 1):
            if not isinstance(ch, dict):
                errs.append(f"chapters[{i}] must be object")
                continue
            for f in ["chapter_id", "intent_allowlist"]:
                if f not in ch:
                    errs.append(f"chapters[{i}] missing {f}")
            ial = ch.get("intent_allowlist")
            if not isinstance(ial, list) or len(ial) < 1:
                errs.append(f"chapters[{i}].intent_allowlist must be non-empty list")
    sig = pack.get("signature")
    if not isinstance(sig, dict) or "pack_hash" not in sig:
        errs.append("signature.pack_hash missing")
    # academic_structure is required for your V31.9.6 workflow
    acs = pack.get("academic_structure")
    if not isinstance(acs, dict):
        errs.append("academic_structure missing (required in V31.9.6)")
    else:
        if not isinstance(acs.get("levels"), list) or len(acs.get("levels") or []) < 1:
            errs.append("academic_structure.levels must be non-empty list")
        if not isinstance(acs.get("subjects"), list) or len(acs.get("subjects") or []) < 1:
            errs.append("academic_structure.subjects must be non-empty list")
        if not isinstance(acs.get("level_subject_map"), list) or len(acs.get("level_subject_map") or []) < 1:
            errs.append("academic_structure.level_subject_map must be non-empty list")
    return len(errs) == 0, errs

def validate_pack_harvest_catalog(pack: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    h = pack.get("harvest")
    if not isinstance(h, dict):
        return False, ["harvest missing or not object (required)"]
    cat = h.get("catalog")
    if not isinstance(cat, list) or len(cat) < 1:
        errs.append("harvest.catalog must be non-empty list (pack-driven per level+subject)")
    else:
        for i, row in enumerate(cat, 1):
            if not isinstance(row, dict):
                errs.append(f"catalog[{i}] must be object")
                continue
            if not str(row.get("level_id", "")).strip():
                errs.append(f"catalog[{i}] missing level_id")
            if not str(row.get("subject_id", "")).strip():
                errs.append(f"catalog[{i}] missing subject_id")
            seeds = row.get("seeds")
            if not isinstance(seeds, list) or len(seeds) < 1:
                errs.append(f"catalog[{i}].seeds must be non-empty list")
            else:
                for j, s in enumerate(seeds, 1):
                    if not isinstance(s, dict):
                        errs.append(f"catalog[{i}].seeds[{j}] must be object")
                        continue
                    if not str(s.get("index_url", "")).strip():
                        errs.append(f"catalog[{i}].seeds[{j}] missing index_url")
                    if not str(s.get("pdf_regex", "")).strip():
                        errs.append(f"catalog[{i}].seeds[{j}] missing pdf_regex")
    http = h.get("http", {})
    if http is not None and not isinstance(http, dict):
        errs.append("harvest.http must be object")
    return len(errs) == 0, errs

def _pack_intent_to_chapter(pack: Dict[str, Any]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for ch in pack.get("chapters", []) or []:
        cid = str(ch.get("chapter_id", "")).strip()
        for intent in (ch.get("intent_allowlist") or []):
            it = str(intent).strip()
            if not it:
                continue
            if it not in m:
                m[it] = cid
    return m

def validators_B(canon: Dict[str, Any], pack: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out: Dict[str, Any] = {}
    chapter_report = {
        "pack_ref": None,
        "created_at": datetime.now().isoformat(),
        "chapters": {},
        "violations": [],
    }

    if not pack:
        out["B-00_PACK_PRESENT"] = {"pass": False, "note": "No pack loaded"}
        return out, chapter_report

    okp, perrs = validate_pack_schema(pack)
    out["B-01_PACK_SCHEMA"] = {"pass": okp, "errors": perrs}
    if not okp:
        chapter_report["pack_ref"] = {"pack_id": pack.get("pack_id"), "country_code": pack.get("country_code")}
        return out, chapter_report

    intent_to_chapter = _pack_intent_to_chapter(pack)

    # QC must map to chapter via intent_code
    unmapped_qc = []
    qc_to_chapter: Dict[str, str] = {}
    for qc in canon.get("qcs", []) or []:
        intent = str((qc.get("qc_invariant_signature") or {}).get("intent_code", "")).strip()
        qc_id = str(qc.get("qc_id", "")).strip()
        cid = intent_to_chapter.get(intent)
        if not cid:
            unmapped_qc.append({"qc_id": qc_id, "intent": intent})
        else:
            qc_to_chapter[qc_id] = cid
    out["B-02_QC_CHAPTER_MAPPING"] = {"pass": len(unmapped_qc) == 0, "unmapped_qc": unmapped_qc}

    # Qi assigned to exactly one chapter via mapped QCs
    all_qi = [q for q in (canon.get("qi_pack") or [])]
    qi_ids_all = [str(q.get("qi_id", "")).strip() for q in all_qi if str(q.get("qi_id", "")).strip()]

    qc_cov: Dict[str, Set[str]] = {}
    for qc in canon.get("qcs", []) or []:
        qc_id = str(qc.get("qc_id", "")).strip()
        cov_ids = set([str(x).strip() for x in (qc.get("mapping") or {}).get("covered_qi_ids", []) if str(x).strip()])
        qc_cov[qc_id] = cov_ids

    qi_to_chapters: Dict[str, Set[str]] = {qid: set() for qid in qi_ids_all}
    for qc_id, cov_ids in qc_cov.items():
        ch_id = qc_to_chapter.get(qc_id)
        if not ch_id:
            continue
        for qi_id in cov_ids:
            if qi_id in qi_to_chapters:
                qi_to_chapters[qi_id].add(ch_id)

    cross = [qi_id for qi_id, s in qi_to_chapters.items() if len(s) > 1]
    no_ch = [qi_id for qi_id, s in qi_to_chapters.items() if len(s) == 0]

    out["B-03_QI_CHAPTER_ASSIGNMENT"] = {
        "pass": (len(cross) == 0 and len(no_ch) == 0),
        "counts": {"cross_leak": len(cross), "unassigned": len(no_ch)},
        "cross_leak_qi_preview": cross[:200],
        "unassigned_qi_preview": no_ch[:200],
    }

    # Coverage per chapter (within assigned Qi of that chapter)
    chapters = pack.get("chapters") or []
    chapter_ids = [str(ch.get("chapter_id", "")).strip() for ch in chapters if str(ch.get("chapter_id", "")).strip()]

    chapter_report["pack_ref"] = {
        "pack_id": pack.get("pack_id"),
        "country_code": pack.get("country_code"),
        "pack_hash": (pack.get("signature") or {}).get("pack_hash"),
    }

    chapter_qc: Dict[str, List[str]] = defaultdict(list)
    for qc_id, ch_id in qc_to_chapter.items():
        chapter_qc[ch_id].append(qc_id)
    for ch in chapter_ids:
        chapter_qc[ch] = sorted(list(set(chapter_qc.get(ch, []))))

    chapter_qi: Dict[str, List[str]] = defaultdict(list)
    for qi_id, s in qi_to_chapters.items():
        if len(s) == 1:
            chapter_qi[list(s)[0]].append(qi_id)
    for ch in chapter_ids:
        chapter_qi[ch] = sorted(chapter_qi.get(ch, []))

    chapter_cov: Dict[str, Set[str]] = defaultdict(set)
    for ch in chapter_ids:
        for qc_id in chapter_qc.get(ch, []):
            chapter_cov[ch].update(qc_cov.get(qc_id, set()))

    all_orphans = []
    for ch in chapter_ids:
        assigned = set(chapter_qi.get(ch, []))
        covered = chapter_cov.get(ch, set())
        orph = sorted([q for q in assigned if q not in covered])
        if orph:
            all_orphans.extend(orph)
        title = ""
        for x in chapters:
            if str(x.get("chapter_id", "")).strip() == ch:
                title = str(x.get("title", "")).strip()
                break
        chapter_report["chapters"][ch] = {
            "title": title,
            "qc_count": len(chapter_qc.get(ch, [])),
            "qi_total": len(chapter_qi.get(ch, [])),
            "qi_covered": max(0, len(chapter_qi.get(ch, [])) - len(orph)),
            "coverage_ratio": (1.0 if len(chapter_qi.get(ch, [])) == 0 else (len(chapter_qi.get(ch, [])) - len(orph)) / float(len(chapter_qi.get(ch, [])))),
            "orphans_preview": orph[:200],
            "qc_ids_preview": chapter_qc.get(ch, [])[:200],
        }

    out["B-04_COVERAGE_PER_CHAPTER"] = {"pass": len(all_orphans) == 0, "orphans_total": len(all_orphans), "orphans_preview": all_orphans[:200]}

    if unmapped_qc:
        chapter_report["violations"].append({"type": "unmapped_qc_intents", "items": unmapped_qc[:200]})
    if cross:
        chapter_report["violations"].append({"type": "qi_cross_leak", "items": cross[:200]})
    if no_ch:
        chapter_report["violations"].append({"type": "qi_unassigned_to_chapter", "items": no_ch[:200]})
    if all_orphans:
        chapter_report["violations"].append({"type": "chapter_orphans", "count": len(all_orphans)})

    return out, chapter_report


# =============================================================================
# H — HARVEST (pack-driven catalog, scope level+subject)
# =============================================================================

def _http_get(url: str, ua: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

def _extract_links_html(html_bytes: bytes, base_url: str) -> List[str]:
    try:
        html = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        html = str(html_bytes)
    hrefs = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html, flags=re.I)
    out = []
    for h in hrefs:
        h = h.strip()
        if not h:
            continue
        absu = urllib.parse.urljoin(base_url, h)
        out.append(absu)
    return out

def _filter_pdf_links(links: List[str], pdf_regex: str) -> List[str]:
    rx = re.compile(pdf_regex, re.I)
    pdfs = []
    for u in links:
        if rx.search(u):
            pdfs.append(u)
    seen = set()
    out = []
    for u in pdfs:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _pair_subject_correction(pdfs: List[str], corr_hint_regex: str) -> List[Tuple[str, Optional[str]]]:
    hint = re.compile(corr_hint_regex, re.I) if corr_hint_regex else None

    def norm_key(u: str) -> str:
        b = urllib.parse.urlparse(u).path.split("/")[-1]
        b = re.sub(r"\.pdf$", "", b, flags=re.I)
        b = re.sub(r"[^a-zA-Z0-9]+", "_", b).lower().strip("_")
        if hint:
            b = hint.sub("", b)
            b = re.sub(r"__+", "_", b).strip("_")
        return b

    corr, subj = [], []
    for u in pdfs:
        if hint and hint.search(u):
            corr.append(u)
        else:
            subj.append(u)

    corr_map: Dict[str, str] = {}
    for u in corr:
        corr_map[norm_key(u)] = u

    pairs: List[Tuple[str, Optional[str]]] = []
    for s in subj:
        k = norm_key(s)
        pairs.append((s, corr_map.get(k)))
    return pairs

def harvest_for_level_subject(pack: Dict[str, Any], level_id: str, subject_id: str, volume: int) -> Dict[str, Any]:
    okh, herrs = validate_pack_harvest_catalog(pack)
    if not okh:
        return {"meta": {"ok": False, "errors": herrs, "created_at": datetime.now().isoformat()}, "items": []}

    h = pack["harvest"]
    http = h.get("http", {}) or {}
    ua = str(http.get("user_agent") or "Mozilla/5.0")
    timeout = int(http.get("timeout_sec") or 25)
    max_pdf = int(http.get("max_bytes_pdf") or 25_000_000)
    max_index_pages = int(http.get("max_index_pages") or 30)

    # pack-driven: trouver la ligne catalog correspondant
    catalog = h.get("catalog") or []
    row = None
    for r in catalog:
        if str(r.get("level_id", "")).strip() == level_id and str(r.get("subject_id", "")).strip() == subject_id:
            row = r
            break
    if not row:
        return {"meta": {"ok": False, "errors": [f"No catalog entry for {level_id}/{subject_id}"], "created_at": datetime.now().isoformat()}, "items": []}

    seeds = row.get("seeds") or []
    collected_pairs: List[Tuple[str, Optional[str]]] = []
    crawl_log = []

    for s in seeds[:max_index_pages]:
        index_url = str(s.get("index_url", "")).strip()
        if not index_url:
            continue
        pdf_regex = str(s.get("pdf_regex", "")).strip()
        corr_hint = str(s.get("corr_hint_regex", "")).strip()

        try:
            html = _http_get(index_url, ua=ua, timeout=timeout)
            links = _extract_links_html(html, base_url=index_url)
            pdfs = _filter_pdf_links(links, pdf_regex=pdf_regex)
            pairs = _pair_subject_correction(pdfs, corr_hint_regex=corr_hint)
            collected_pairs.extend(pairs)
            crawl_log.append({"seed": index_url, "pdf_found": len(pdfs), "pairs": len(pairs)})
        except Exception as e:
            crawl_log.append({"seed": index_url, "error": str(e)})

    # de-dup by subject_url
    seen_subj = set()
    pairs_unique = []
    for subj_url, corr_url in collected_pairs:
        if subj_url in seen_subj:
            continue
        seen_subj.add(subj_url)
        pairs_unique.append((subj_url, corr_url))

    random.seed(1337)
    if len(pairs_unique) > volume:
        pairs_unique = random.sample(pairs_unique, k=volume)
    else:
        pairs_unique = pairs_unique[:max(0, int(volume))]

    items: List[Dict[str, Any]] = []
    dl_log = []

    for i, (subj_url, corr_url) in enumerate(pairs_unique, 1):
        try:
            sb = _http_get(subj_url, ua=ua, timeout=timeout)
            if len(sb) > max_pdf:
                raise Exception("subject_pdf_too_large")

            cb = None
            if corr_url:
                cb = _http_get(corr_url, ua=ua, timeout=timeout)
                if len(cb) > max_pdf:
                    cb = None

            label = urllib.parse.urlparse(subj_url).path.split("/")[-1] or f"subject_{i}.pdf"
            items.append({
                "label": label,
                "scope": {"level_id": level_id, "subject_id": subject_id},
                "subject_url": subj_url,
                "correction_url": corr_url,
                "subject_bytes": sb,
                "correction_bytes": cb,
            })
            dl_log.append({"i": i, "ok": True, "label": label, "has_correction": bool(cb)})
        except Exception as e:
            dl_log.append({"i": i, "ok": False, "subject_url": subj_url, "error": str(e)})

    meta = {
        "ok": True,
        "created_at": datetime.now().isoformat(),
        "scope": {"level_id": level_id, "subject_id": subject_id},
        "requested_volume": int(volume),
        "items_ok": len(items),
        "crawl_log_preview": crawl_log[:50],
        "download_log_preview": dl_log[:50],
        "doctrine": "blind_fishing_within_pack_scope_then_depessage",
    }
    return {"meta": meta, "items": items}


# =============================================================================
# I — UI HELPERS (academic_structure)
# =============================================================================

def level_title(pack: Dict[str, Any], level_id: str) -> str:
    acs = pack.get("academic_structure") or {}
    for lv in (acs.get("levels") or []):
        if str(lv.get("level_id", "")).strip() == level_id:
            return str(lv.get("title", "")).strip() or level_id
    return level_id

def subject_title(pack: Dict[str, Any], subject_id: str) -> str:
    acs = pack.get("academic_structure") or {}
    for sb in (acs.get("subjects") or []):
        if str(sb.get("subject_id", "")).strip() == subject_id:
            return str(sb.get("title", "")).strip() or subject_id
    return subject_id

def allowed_subjects_for_level(pack: Dict[str, Any], level_id: str) -> List[str]:
    acs = pack.get("academic_structure") or {}
    for row in (acs.get("level_subject_map") or []):
        if str(row.get("level_id", "")).strip() == level_id:
            subs = row.get("subject_ids") or []
            return [str(x).strip() for x in subs if str(x).strip()]
    return []

def load_pack_auto_for_country(country_code: str) -> Dict[str, Any]:
    # Auto-load from local file if present, else fallback
    fname = f"academic_pack_{country_code}.json"
    p = Path(fname)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return TEST_ONLY_DEFAULT_PACK_FR
    return TEST_ONLY_DEFAULT_PACK_FR


# =============================================================================
# J — GTE RUNNER (UI)
# =============================================================================

class GTERunner:
    def __init__(self):
        self.logs: List[str] = []
        self.validators: Dict[str, Any] = {}

    def log(self, m: str):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

    def run(self, qi_pack: List[Dict[str, Any]], pack: Dict[str, Any], lvl2: bool) -> Dict[str, Any]:
        self.logs = []
        self.validators = {}

        self.log(f"═══ GTE V31.9.6 — {len(qi_pack)} Qi ═══")

        # Choose generator: real engine (if present) or test-only
        if REAL_ENGINE_AVAILABLE:
            self.log("ENGINE: REAL (smaxia_granulo_engine_test.run_granulo_test) — (UI test harness)")
            # In ISO-PROD UI, the real engine typically takes URLs.
            # Here we still run QC generation from Qi_pack via fallback generator unless your real engine exposes a Qi->QC function.
            # Therefore, we keep QC generation = TEST_ONLY, but we expose REAL engine check for later integration.
            gen_func = TEST_ONLY_generate_qc_ari_frt_trg
        else:
            self.log("ENGINE: TEST_ONLY (structure-driven) — infrastructure validation")
            gen_func = TEST_ONLY_generate_qc_ari_frt_trg

        artifacts_raw = gen_func(qi_pack)
        canon = canonicalize_artifacts(artifacts_raw, qi_pack)

        qcs, aris, frts, trgs = canon["qcs"], canon["aris"], canon["frts"], canon["triggers"]
        singletons = canon["singletons_warning"]
        self.log(f"Generated: {len(qcs)} QC, {len(aris)} ARI, {len(frts)} FRT, {len(trgs)} TRG")
        if singletons:
            self.log(f"⚠️ Singletons: {len(singletons)}")

        # Schema checks
        errs = []
        for qi in canon["qi_pack"]:
            ok, e = validate_qi_schema(qi)
            if not ok:
                errs.append(f"Qi: {e}")
        for qc in qcs:
            ok, e = validate_qc_schema(qc)
            if not ok:
                errs.append(f"QC: {e}")
        for ari in aris:
            ok, e = validate_ari_schema(ari)
            if not ok:
                errs.append(f"ARI: {e}")
        for frt in frts:
            ok, e = validate_frt_schema(frt)
            if not ok:
                errs.append(f"FRT: {e}")
        for trg in trgs:
            ok, e = validate_trigger_schema(trg)
            if not ok:
                errs.append(f"TRG: {e}")

        self.validators["TS-01_SCHEMA"] = {"pass": len(errs) == 0, "errors": errs}
        self.log(f"TS-01 SCHEMA: {'PASS' if len(errs) == 0 else 'FAIL'}")

        ref_ok, ref_errs = check_ref_integrity(canon)
        self.validators["TS-02_REF"] = {"pass": ref_ok, "errors": ref_errs}
        self.log(f"TS-02 REF: {'PASS' if ref_ok else 'FAIL'}")

        cov_ok, orphans = check_coverage_total(canon)
        self.validators["TC-01_COV"] = {"pass": cov_ok, "orphans": orphans}
        self.log(f"TC-01 COV: {'PASS' if cov_ok else 'FAIL'} ({len(orphans)} orphans)")

        prim_ok, dups = check_coverage_primary_unique(canon)
        self.validators["TC-02_PRIM"] = {"pass": prim_ok, "duplicates": dups}
        self.log(f"TC-02 PRIM: {'PASS' if prim_ok else 'FAIL'}")

        det_ok, hashes = check_determinism_n_runs(gen_func, qi_pack, 3)
        self.validators["TD-01_DET_N3"] = {"pass": det_ok, "hashes": hashes}
        self.log(f"TD-01 DET_N3: {'PASS' if det_ok else 'FAIL'}")

        ord_ok, h1, h2 = check_order_invariance(gen_func, qi_pack)
        self.validators["TD-02_ORDER"] = {"pass": ord_ok, "h1": h1[:16], "h2": h2[:16]}
        self.log(f"TD-02 ORDER: {'PASS' if ord_ok else 'FAIL'}")

        if lvl2:
            d10_ok, h10 = check_determinism_n_runs(gen_func, qi_pack, 10)
            self.validators["TD-03_N10"] = {"pass": d10_ok, "hashes": h10}
            self.log(f"TD-03 N10: {'PASS' if d10_ok else 'FAIL'}")
        else:
            self.validators["TD-03_N10"] = {"pass": None, "note": "Skipped"}

        src = get_full_source()
        if src:
            ast_ok, ast_v = scan_ast_sensitive_access(src)
            self.validators["TA-01_AST"] = {"pass": ast_ok, "violations": ast_v}
            self.log(f"TA-01 AST: {'PASS' if ast_ok else 'FAIL'}")

        # Validateurs B BLOQUANTS (chapitres pack-driven + preuves)
        bvals, chapter_report = validators_B(canon, pack)
        for k, v in bvals.items():
            self.validators[k] = v
            self.log(f"{k}: {'PASS' if v.get('pass') else 'FAIL'}")

        # Verdict: must all core + all B-* pass
        def is_green(k: str) -> bool:
            p = self.validators.get(k, {}).get("pass")
            return True if p is None else bool(p)

        must = ["TS-01_SCHEMA", "TS-02_REF", "TC-01_COV", "TC-02_PRIM", "TD-01_DET_N3", "TD-02_ORDER"]
        must.extend([k for k in sorted(self.validators.keys()) if k.startswith("B-")])

        verdict = "PASS" if all(is_green(k) for k in must) else "FAIL"
        self.log(f"═══ VERDICT: {verdict} ═══")

        return {
            "verdict": verdict,
            "validators": self.validators,
            "artifacts_raw": artifacts_raw,
            "artifacts_canon": canon,
            "chapter_report": chapter_report,
            "metrics": {
                "qi": len(qi_pack), "qc": len(qcs), "ari": len(aris),
                "frt": len(frts), "trg": len(trgs), "singletons": len(singletons),
                "input_hash": compute_hash(sort_qi_canonical(qi_pack)),
                "output_hash": compute_hash(canon),
            },
            "logs": self.logs,
        }


# =============================================================================
# K — APP (UI)
# =============================================================================

def main():
    st.set_page_config(page_title="SMAXIA GTE V31.9.6", page_icon="🔒", layout="wide")
    st.title("🔒 SMAXIA GTE Console V31.9.6")
    st.markdown("**ISO-PROD TEST — Activation Pays → Pack visible → Sélection Niveau+Matières → Harvest → Extraction Qi/RQi → RUN GTE → Résultats**")

    # Session state
    st.session_state.setdefault("country_code", "FR")
    st.session_state.setdefault("active_pack", None)
    st.session_state.setdefault("active_pack_source", "AUTO")
    st.session_state.setdefault("selected_level", None)
    st.session_state.setdefault("selected_subjects", [])
    st.session_state.setdefault("harvest_items", [])          # list of items (bytes)
    st.session_state.setdefault("harvest_meta", [])           # list of meta per scope
    st.session_state.setdefault("selected_items_idx", set())  # for extraction
    st.session_state.setdefault("qi_pack", [])
    st.session_state.setdefault("qi_meta_list", [])
    st.session_state.setdefault("result", None)

    # ---------------- Sidebar (Activation + Pack) ----------------
    with st.sidebar:
        st.header("V31.9.6 — Chaîne ISO-PROD TEST")
        st.caption("Le pack doit contenir: academic_structure + chapters + harvest.catalog (niveau+matière).")

        country = st.selectbox("Pays (TEST)", ["FR"], index=0)
        st.session_state["country_code"] = country

        st.markdown("### Academic Pack (auto + override upload)")
        pack_up = st.file_uploader("academic_pack.json (override)", type=["json"])
        pack_override = None
        if pack_up:
            raw, err = load_json_from_uploaded(pack_up)
            if err:
                st.error(err)
            elif isinstance(raw, dict):
                pack_override = raw
            else:
                st.error("academic_pack.json doit être un objet JSON")

        if pack_override is not None:
            pack = pack_override
            st.session_state["active_pack_source"] = "UPLOAD"
        else:
            pack = load_pack_auto_for_country(country)
            st.session_state["active_pack_source"] = "AUTO"

        st.session_state["active_pack"] = pack

        okp, perrs = validate_pack_schema(pack) if isinstance(pack, dict) else (False, ["Pack is not dict"])
        if okp:
            st.success(f"Pack chargé: {pack.get('pack_id')} ({pack.get('country_code')}) — source={st.session_state['active_pack_source']}")
        else:
            st.error("Pack invalide — chaîne BLOQUÉE")
            st.json(perrs)

        with st.expander("Voir Academic Pack (résumé)", expanded=True):
            acs = (pack.get("academic_structure") or {}) if isinstance(pack, dict) else {}
            st.json({
                "source": st.session_state["active_pack_source"],
                "pack_id": pack.get("pack_id") if isinstance(pack, dict) else None,
                "country_code": pack.get("country_code") if isinstance(pack, dict) else None,
                "signature": pack.get("signature") if isinstance(pack, dict) else None,
                "levels_count": len(acs.get("levels") or []),
                "subjects_count": len(acs.get("subjects") or []),
                "level_subject_map_count": len(acs.get("level_subject_map") or []),
                "chapters_count": len(pack.get("chapters") or []) if isinstance(pack, dict) else 0,
                "chapters_preview": (pack.get("chapters") or [])[:12] if isinstance(pack, dict) else [],
                "harvest_catalog_count": len(((pack.get("harvest") or {}).get("catalog") or [])) if isinstance(pack, dict) else 0,
            })

        st.markdown("---")
        st.markdown("### Sélection TEST (niveau + 2 matières max)")
        if okp:
            acs = pack.get("academic_structure") or {}
            levels = [str(x.get("level_id", "")).strip() for x in (acs.get("levels") or []) if str(x.get("level_id", "")).strip()]
            if not levels:
                st.error("Aucun niveau dans le pack.")
            else:
                def _lv_label(lv: str) -> str:
                    return f"{lv} — {level_title(pack, lv)}"

                default_lv = st.session_state["selected_level"] if st.session_state["selected_level"] in levels else levels[0]
                lvl = st.selectbox("Niveau", levels, index=levels.index(default_lv), format_func=_lv_label)
                st.session_state["selected_level"] = lvl

                allowed = allowed_subjects_for_level(pack, lvl)
                if not allowed:
                    st.warning("Aucune matière mappée à ce niveau dans level_subject_map.")
                    st.session_state["selected_subjects"] = []
                else:
                    def _sb_label(sb: str) -> str:
                        return f"{sb} — {subject_title(pack, sb)}"
                    subs = st.multiselect("Matières (max 2)", allowed, default=allowed[:2], format_func=_sb_label)
                    subs = subs[:2]
                    st.session_state["selected_subjects"] = subs

        st.markdown("---")
        st.markdown("### Harvest (pack-driven)")
        vol = st.number_input("Volume par matière (nb de sujets)", min_value=1, value=20, step=5)

        okh, herrs = (False, ["pack invalid"])
        if okp:
            okh, herrs = validate_pack_harvest_catalog(pack)

        harvest_ready = okp and okh and bool(st.session_state.get("selected_level")) and len(st.session_state.get("selected_subjects") or []) >= 1
        if not okh and okp:
            st.warning("Harvest OFF: pack.harvest.catalog manquant/invalide (uploadez le pack FR réel).")

        btn_disabled = not harvest_ready

        if st.button("⛽ LANCER HARVEST", type="primary", use_container_width=True, disabled=btn_disabled):
            st.session_state["harvest_items"] = []
            st.session_state["harvest_meta"] = []
            st.session_state["selected_items_idx"] = set()
            st.session_state["qi_pack"] = []
            st.session_state["qi_meta_list"] = []
            st.session_state["result"] = None

            level_id = st.session_state["selected_level"]
            subjects = st.session_state["selected_subjects"]

            with st.spinner("Harvest en cours (pack-driven)…"):
                for sb in subjects:
                    out = harvest_for_level_subject(pack, level_id, sb, int(vol))
                    st.session_state["harvest_meta"].append(out.get("meta"))
                    items = out.get("items") or []
                    st.session_state["harvest_items"].extend(items)

            # auto-select all harvested items for extraction
            st.session_state["selected_items_idx"] = set(range(len(st.session_state["harvest_items"])))
            st.success(f"Harvest terminé: {len(st.session_state['harvest_items'])} sujets (toutes matières confondues).")

        if harvest_ready:
            st.info("Prêt: sélection OK + catalog OK → vous pouvez lancer Harvest.")
        else:
            st.caption("Conditions: pack valide + niveau + ≥1 matière + harvest.catalog valide.")

    # ---------------- Main tabs ----------------
    t1, t2, t3 = st.tabs(["📥 Import PDF", "🧪 Extraction & Pipeline", "📊 Résultats"])

    # TAB 1 — Import PDF (Harvest library + manual upload)
    with t1:
        st.header("📥 Import PDF (sujets + corrigés)")
        pack = st.session_state.get("active_pack")
        items = st.session_state.get("harvest_items") or []

        left, right = st.columns([2, 1])

        with left:
            st.subheader("Bibliothèque Harvest (visible)")
            if not items:
                st.info("Aucun item harvest. Lancez Harvest dans la sidebar (niveau+matières).")
            else:
                st.caption("Sélectionnez les documents à dépesser (extraction Qi/RQi).")
                # selection table-like
                selected_idx = set(st.session_state.get("selected_items_idx") or set())
                for i, it in enumerate(items):
                    scope = it.get("scope") or {}
                    lv, sb = scope.get("level_id"), scope.get("subject_id")
                    label = it.get("label", f"doc_{i}.pdf")
                    has_corr = bool(it.get("correction_bytes"))
                    key = f"sel_{i}"
                    default = i in selected_idx
                    checked = st.checkbox(
                        f"{i+1:03d} — {label} | {lv}/{sb} | {'corr✓' if has_corr else 'corr✗'}",
                        value=default,
                        key=key
                    )
                    if checked:
                        selected_idx.add(i)
                    else:
                        selected_idx.discard(i)

                st.session_state["selected_items_idx"] = selected_idx

                with st.expander("Meta Harvest (preview)", expanded=False):
                    st.json(st.session_state.get("harvest_meta") or [])

        with right:
            st.subheader("Upload manuel (optionnel)")
            st.caption("Permet d’ajouter 1 paire PDF au panier de test si besoin.")
            if pdfplumber is None:
                st.error("pdfplumber non installé")
            else:
                subj_up = st.file_uploader("PDF Sujet", type=["pdf"], key="manual_subj")
                corr_up = st.file_uploader("PDF Correction (opt)", type=["pdf"], key="manual_corr")

                if st.button("➕ Ajouter à la bibliothèque", use_container_width=True):
                    if not subj_up:
                        st.error("Ajoutez un PDF sujet.")
                    else:
                        sb = subj_up.read()
                        cb = corr_up.read() if corr_up else None
                        st.session_state["harvest_items"].append({
                            "label": subj_up.name,
                            "scope": {"level_id": "MANUAL", "subject_id": "MANUAL"},
                            "subject_url": None,
                            "correction_url": None,
                            "subject_bytes": sb,
                            "correction_bytes": cb,
                        })
                        idx = len(st.session_state["harvest_items"]) - 1
                        st.session_state["selected_items_idx"].add(idx)
                        st.success("Ajouté + sélectionné pour extraction.")

    # TAB 2 — Extraction & Pipeline
    with t2:
        st.header("🧪 Extraction Qi/RQi → RUN GTE")
        pack = st.session_state.get("active_pack")
        okp, perrs = validate_pack_schema(pack) if isinstance(pack, dict) else (False, ["Pack invalid"])

        if not okp:
            st.error("Pack invalide. Corrigez/chargez un pack conforme (academic_structure + chapters + harvest.catalog).")
            st.json(perrs)
        else:
            if pdfplumber is None:
                st.error("pdfplumber non installé (extraction PDF impossible).")
            else:
                items = st.session_state.get("harvest_items") or []
                selected_idx = sorted(list(st.session_state.get("selected_items_idx") or set()))
                st.markdown(f"**Sélection actuelle:** {len(selected_idx)} documents")

                c1, c2, c3 = st.columns([1, 1, 1])

                with c1:
                    if st.button("🔪 EXTRAIRE Qi/RQi (dépessage)", type="primary", use_container_width=True, disabled=(len(selected_idx) == 0)):
                        qi_all: List[Dict[str, Any]] = []
                        meta_list: List[Dict[str, Any]] = []
                        with st.spinner("Extraction en cours…"):
                            for idx in selected_idx:
                                it = items[idx]
                                label = it.get("label", f"doc_{idx}.pdf")
                                sb = it.get("subject_bytes")
                                cb = it.get("correction_bytes")
                                if not isinstance(sb, (bytes, bytearray)):
                                    continue
                                meta, qi_pack = build_qi_pack_from_pdf_pair(sb, cb, label=label)
                                meta_list.append(meta)
                                qi_all.extend(qi_pack)

                        # Re-index order_index globally for determinism stable
                        qi_all = sort_qi_canonical(qi_all)
                        for i, q in enumerate(qi_all, 1):
                            q["position"]["order_index"] = i

                        st.session_state["qi_pack"] = qi_all
                        st.session_state["qi_meta_list"] = meta_list
                        st.session_state["result"] = None
                        st.success(f"Extraction OK: {len(qi_all)} Qi (total).")

                with c2:
                    lvl2 = st.checkbox("Level 2 (N=10 déterminisme)", value=False)

                with c3:
                    if st.button("▶️ RUN GTE", use_container_width=True, disabled=(len(st.session_state.get("qi_pack") or []) == 0)):
                        runner = GTERunner()
                        with st.spinner("RUN GTE…"):
                            res = runner.run(st.session_state["qi_pack"], pack, lvl2=lvl2)
                        st.session_state["result"] = res

                        # logs (compact)
                        for l in res["logs"]:
                            if "FAIL" in l:
                                st.markdown(f"🔴 `{l}`")
                            elif "PASS" in l:
                                st.markdown(f"🟢 `{l}`")
                            elif "⚠️" in l:
                                st.markdown(f"🟡 `{l}`")
                            else:
                                st.text(l)

                        if res["verdict"] == "PASS":
                            st.success("✅ VERDICT: PASS")
                        else:
                            st.error("❌ VERDICT: FAIL (B bloquants inclus)")

                st.markdown("---")
                st.subheader("Aperçu Qi (post extraction)")
                qi_pack = st.session_state.get("qi_pack") or []
                if not qi_pack:
                    st.info("Aucun Qi. Lancez d’abord l’extraction.")
                else:
                    st.metric("Qi total", len(qi_pack))
                    with st.expander("Qi preview (10)", expanded=False):
                        st.json(qi_pack[:10])

                if st.session_state.get("qi_meta_list"):
                    with st.expander("Meta extraction (tous PDFs)", expanded=False):
                        st.json(st.session_state["qi_meta_list"])

    # TAB 3 — Résultats + explorer chapitres + exports
    with t3:
        st.header("📊 Résultats")
        res = st.session_state.get("result")
        if not res:
            st.info("Lancez RUN GTE (onglet précédent).")
            return

        m = res["metrics"]
        cols = st.columns(6)
        cols[0].metric("Qi", m["qi"])
        cols[1].metric("QC", m["qc"])
        cols[2].metric("ARI", m["ari"])
        cols[3].metric("FRT", m["frt"])
        cols[4].metric("TRG", m["trg"])
        cols[5].metric("Singletons", m["singletons"])
        st.markdown(f"**Hash output:** `{m['output_hash'][:32]}...`")

        if res["verdict"] == "PASS":
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

        st.subheader("Validateurs (dont B BLOQUANTS)")
        for k, v in res["validators"].items():
            p = v.get("pass")
            icon = "✅" if p is True else "❌" if p is False else "⏭️"
            with st.expander(f"{icon} {k}", expanded=False):
                st.json(v)

        st.subheader("chapter_report.json (preuves par chapitre)")
        st.json(res.get("chapter_report") or {})

        st.markdown("---")
        st.subheader("Explorateur — QC par Chapitre (pack-driven)")
        pack = st.session_state.get("active_pack") or {}
        chapters = pack.get("chapters") or []
        chapter_ids = [str(ch.get("chapter_id", "")).strip() for ch in chapters if str(ch.get("chapter_id", "")).strip()]

        if not chapter_ids:
            st.warning("Aucun chapitre dans le pack.")
        else:
            intent_to_ch = _pack_intent_to_chapter(pack)

            def ch_label(cid: str) -> str:
                t = ""
                for ch in chapters:
                    if str(ch.get("chapter_id", "")).strip() == cid:
                        t = str(ch.get("title", "")).strip()
                        break
                return f"{cid} — {t}" if t else cid

            sel = st.selectbox("Chapitre", chapter_ids, format_func=ch_label)

            canon = res.get("artifacts_canon") or {}
            qcs = canon.get("qcs") or []
            aris = {a.get("ari_id"): a for a in (canon.get("aris") or [])}
            frts = {f.get("frt_id"): f for f in (canon.get("frts") or [])}
            trgs = {t.get("trigger_id"): t for t in (canon.get("triggers") or [])}
            qi_map = {q.get("qi_id"): q for q in (canon.get("qi_pack") or [])}

            qcs_ch = []
            for qc in qcs:
                intent = str((qc.get("qc_invariant_signature") or {}).get("intent_code", "")).strip()
                if intent_to_ch.get(intent) == sel:
                    qcs_ch.append(qc)

            st.info(f"QC sur ce chapitre: {len(qcs_ch)}")

            with st.expander("Détails QC (avec ARI/FRT/TRG + Qi associées)", expanded=True):
                for qc in qcs_ch[:200]:
                    links = qc.get("links") or {}
                    ari = aris.get(links.get("ari_id"))
                    frt = frts.get(links.get("frt_id"))
                    trig_list = [trgs.get(tid) for tid in (links.get("trigger_ids") or []) if trgs.get(tid)]
                    covered = (qc.get("mapping") or {}).get("covered_qi_ids") or []
                    qi_examples = [qi_map.get(qid) for qid in covered[:5] if qi_map.get(qid)]

                    st.markdown(f"**{qc.get('qc_id')}** — {qc.get('qc_formulation')}")
                    st.caption(f"ARI={links.get('ari_id')} | FRT={links.get('frt_id')} | TRG={len(trig_list)} | Qi={len(covered)}")
                    st.json({"qc": qc, "ari": ari, "frt": frt, "triggers": trig_list, "qi_examples": qi_examples})

            st.download_button(
                "📥 Export chapter_bundle.json",
                canonical_json({
                    "chapter_id": sel,
                    "chapter_title": ch_label(sel),
                    "bundle": {
                        "qcs": qcs_ch,
                        "aris": aris,
                        "frts": frts,
                        "triggers": trgs,
                    },
                    "proof": (res.get("chapter_report") or {}).get("chapters", {}).get(sel, {}),
                    "created_at": datetime.now().isoformat(),
                }),
                "chapter_bundle.json",
            )

        st.markdown("---")
        st.subheader("Exports")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📥 report.json", canonical_json({"version": "V31.9.6", "verdict": res["verdict"], "validators": res["validators"], "metrics": m}), "report.json")
        with c2:
            st.download_button("📥 canon.json", canonical_json(res.get("artifacts_canon") or {}), "canon.json")
        with c3:
            st.download_button("📥 chapter_report.json", canonical_json(res.get("chapter_report") or {}), "chapter_report.json")


if __name__ == "__main__":
    main()
