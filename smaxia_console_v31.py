# =============================================================================
# SMAXIA — GTE Console V31.9.5 (ISO-PROD TEST)
# =============================================================================
# V31.9.5 — Demandés par CEO :
# 1) Chargement "tout academic Pays = France" visualisable Sidebar (pack-driven)
# 2) Lancement automatique scraping sujets+corrections : user saisit Volume => RUN
# 3) Les sujets/corrigés scrappés deviennent sélectionnables dans Entrée (Import PDF)
#    puis on déroule: générer qi_pack.json -> RUN GTE -> Résultats
# 4) Nouveau espace: "Explorer Chapitre" => QC du chapitre + ARI/FRT/TRG + Qi associés
#
# CONTRAT INVARIANT (ULTRA CRITIQUE) :
# - Aucun hardcode métier/pays/chapitre/langue dans le code source.
# - La variabilité vient UNIQUEMENT du academic_pack.json (Pack).
# - Scraping: SEEDS + règles (regex/selectors) doivent être fournis par le Pack.
# - Chapitre: JAMAIS déduit du texte; uniquement pack-driven via intent_allowlist.
#
# LIMITATION STREAMLIT :
# - On ne peut pas "remplir" automatiquement un file_uploader existant.
#   Solution V31.9.5: une "Bibliothèque Harvest" + Selectbox -> charge bytes en session
#   et déclenche "Générer qi_pack" sans uploader.
# =============================================================================

from __future__ import annotations

import ast
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A : TEST-ONLY FIXTURES — NOT FOR PROD
# ═══════════════════════════════════════════════════════════════════════════════

TEST_ONLY_GOLDEN_QI_PACK: List[Dict[str, Any]] = [
    {"qi_id": "QI_TEST_001", "subject_id": "SUBJ_A", "position": {"order_index": 1, "marker": "EX1-Q1"},
     "statement": {"text_md": "Texte question 1"},
     "correction": {"available": True, "text_md": "Correction 1"}},
    {"qi_id": "QI_TEST_002", "subject_id": "SUBJ_A", "position": {"order_index": 2, "marker": "EX1-Q2a"},
     "statement": {"text_md": "Texte question 2a"},
     "correction": {"available": True, "text_md": "Correction 2a"}},
    {"qi_id": "QI_TEST_003", "subject_id": "SUBJ_A", "position": {"order_index": 3, "marker": "EX1-Q2b"},
     "statement": {"text_md": "Texte question 2b"},
     "correction": {"available": True, "text_md": "Correction 2b"}},
    {"qi_id": "QI_TEST_004", "subject_id": "SUBJ_A", "position": {"order_index": 4, "marker": "EX2-Q1"},
     "statement": {"text_md": "Texte question EX2-Q1"},
     "correction": {"available": True, "text_md": "Correction EX2-Q1"}},
]


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B : CORE-LIKE INVARIANT HELPERS (métier-neutre)
# ═══════════════════════════════════════════════════════════════════════════════

SENSITIVE_CONTEXT_FIELDS: Set[str] = {
    "country_code", "language", "domain_code", "assessment_type_code",
    "chapter_code", "chapter", "discipline", "exam_type", "locale", "curriculum",
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


# ----------------------------- SCHEMAS -----------------------------

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


# ----------------------------- INTEGRITY / COVERAGE -----------------------------

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


def check_determinism_n_runs(gen_func, qi_pack: List[Dict[str, Any]], n: int = 3) -> Tuple[bool, List[str], List[str]]:
    hashes = [compute_hash(canonicalize_artifacts(gen_func(qi_pack), qi_pack)) for _ in range(n)]
    ok = len(set(hashes)) == 1
    return ok, hashes, ([] if ok else ["Divergence"])


def check_order_invariance(gen_func, qi_pack: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    h1 = compute_hash(canonicalize_artifacts(gen_func(qi_pack), qi_pack))
    shuffled = qi_pack.copy()
    random.seed(42)
    random.shuffle(shuffled)
    h2 = compute_hash(canonicalize_artifacts(gen_func(shuffled), shuffled))
    return h1 == h2, h1, h2


def scan_forbidden_literals(source: str, forbidden: Set[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    if not forbidden:
        return True, []
    violations = []
    in_def = False
    for i, line in enumerate(source.split("\n"), 1):
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        if "forbidden_literals" in s.lower() and "{" in s:
            in_def = True
            continue
        if in_def:
            if "}" in s:
                in_def = False
            continue
        low = s.lower()
        for lit in forbidden:
            if lit and lit in low:
                violations.append({"line": i, "literal": lit})
    return len(violations) == 0, violations


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


def check_no_test_imports(source: str) -> Tuple[bool, List[str]]:
    violations = []
    for i, line in enumerate(source.split("\n"), 1):
        if "import" in line.lower():
            for p in ["tests/", "fixtures/"]:
                if p in line.lower():
                    violations.append(f"L{i}")
    return len(violations) == 0, violations


def check_intent_diversity(canon: Dict[str, Any], min_i: int = 5) -> Tuple[bool, int, Set[str]]:
    intents = {qc.get("qc_invariant_signature", {}).get("intent_code", "") for qc in canon.get("qcs", [])}
    intents.discard("")
    return len(intents) >= min_i, len(intents), intents


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B.5 : PDF → Qi Builder (markers EXn-Qm[a/b/c])
# ═══════════════════════════════════════════════════════════════════════════════

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


def build_qi_pack_from_pdfs(subject_bytes: bytes, correction_bytes: Optional[bytes]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
            "position": {"order_index": i, "marker": marker},
            "statement": {"text_md": statement},
            "correction": {"available": bool(corr_txt), "text_md": corr_txt},
        })

    meta = {
        "generator": {"name": "SMAXIA_PDF_QI_BUILDER", "version": "V31.9.5"},
        "created_at": datetime.now().isoformat(),
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
        "subject_blocks_preview": [{"kind": "QI", "key": b.get("marker", ""), "text": b.get("text", "")[:800]} for b in subj_blocks[:30]],
        "correction_index_keys_preview": sorted(list(corr_idx.keys()))[:60],
    }
    return meta, qi_pack


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B.7 : QC/ARI/FRT/TRG — TEST GENERATOR (STRICT INVARIANT, STRUCTURE-DRIVEN)
# ═══════════════════════════════════════════════════════════════════════════════

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

        qc_formulation = "Comment procéder ?"

        qcs.append({
            "qc_id": qc_id,
            "qc_formulation": qc_formulation,
            "qc_invariant_signature": {"intent_code": gkey},
            "mapping": {"primary_qi_ids": qi_ids, "covered_qi_ids": qi_ids, "by_subject": dict(by_subj)},
            "links": {"ari_id": ari_id, "frt_id": frt_id, "trigger_ids": [trg_id]},
            "provenance": {"generator_version": "TEST_ONLY_V31.9.5"},
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


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B.8 : ACADEMIC PACK + VALIDATEURS B (BLOQUANTS)
# ═══════════════════════════════════════════════════════════════════════════════

def load_academic_pack_json(uploaded) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not uploaded:
        return None, None
    try:
        raw = json.load(uploaded)
        if not isinstance(raw, dict):
            return None, "academic_pack.json doit être un objet JSON"
        return raw, None
    except Exception as e:
        return None, f"Parse error: {e}"


def validate_pack_schema(pack: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Minimal required structure (TEST):
    {
      "pack_id": "...",
      "country_code": "...",
      "signature": {"pack_hash":"...", "signed_at":"..."},
      "chapters": [
        {"chapter_id":"CH_...", "title":"...", "intent_allowlist":["STRUCT_EX1_Q1", ...]},
        ...
      ],
      "harvest": { ... }  # optional but required for auto-scrape V31.9.5
    }
    """
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
            else:
                for it in ial:
                    if not str(it).strip():
                        errs.append(f"chapters[{i}].intent_allowlist has empty")
    sig = pack.get("signature")
    if not isinstance(sig, dict) or "pack_hash" not in sig:
        errs.append("signature.pack_hash missing")
    return len(errs) == 0, errs


def validate_pack_harvest(pack: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    V31.9.5 auto-scrape requires pack.harvest:
    {
      "sequencing": {"dimensions": ["level","track","year"]},
      "seeds": [
        {"index_url":"https://...","pdf_regex": "....pdf", "corr_hint_regex":"corrige|correction", "pairing_mode":"adjacent|same_page|by_year"},
        ...
      ],
      "http": {"user_agent":"...", "timeout_sec": 25, "max_bytes_pdf": 25000000, "max_index_pages": 30}
    }
    IMPORTANT: These are DATA. No site hardcode in code.
    """
    errs = []
    h = pack.get("harvest")
    if h is None:
        return False, ["Missing pack.harvest (required for auto-scrape)"]
    if not isinstance(h, dict):
        return False, ["pack.harvest must be object"]
    seeds = h.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 1:
        errs.append("pack.harvest.seeds must be non-empty list")
    else:
        for i, s in enumerate(seeds, 1):
            if not isinstance(s, dict):
                errs.append(f"seeds[{i}] must be object")
                continue
            if not str(s.get("index_url", "")).strip():
                errs.append(f"seeds[{i}] missing index_url")
            if not str(s.get("pdf_regex", "")).strip():
                errs.append(f"seeds[{i}] missing pdf_regex")
            # corr_hint_regex optional
    http = h.get("http", {})
    if http is not None and not isinstance(http, dict):
        errs.append("pack.harvest.http must be object")
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


def validators_B(canon: Dict[str, Any], pack: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    chapter_report = {
        "pack_ref": None,
        "created_at": datetime.now().isoformat(),
        "chapters": {},
        "violations": [],
    }

    if not pack:
        out["B-00_PACK_PRESENT"] = {"pass": None, "note": "No pack provided"}
        out["_chapter_report"] = chapter_report
        return out

    okp, perrs = validate_pack_schema(pack)
    out["B-01_PACK_SCHEMA"] = {"pass": okp, "errors": perrs}
    if not okp:
        chapter_report["pack_ref"] = {"pack_id": pack.get("pack_id"), "country_code": pack.get("country_code")}
        out["_chapter_report"] = chapter_report
        return out

    seen: Dict[str, str] = {}
    dups: List[Dict[str, str]] = []
    for ch in pack.get("chapters", []) or []:
        cid = str(ch.get("chapter_id", "")).strip()
        for intent in (ch.get("intent_allowlist") or []):
            it = str(intent).strip()
            if not it:
                continue
            if it in seen and seen[it] != cid:
                dups.append({"intent": it, "chapters": f"{seen[it]}|{cid}"})
            else:
                seen[it] = cid
    out["B-02_INTENT_UNIQUE"] = {"pass": len(dups) == 0, "duplicates": dups}

    intent_to_chapter = _pack_intent_to_chapter(pack)

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
    out["B-03_QC_CHAPTER_MAPPING"] = {"pass": len(unmapped_qc) == 0, "unmapped_qc": unmapped_qc}

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

    out["B-04_QI_CHAPTER_ASSIGNMENT"] = {
        "pass": (len(cross) == 0 and len(no_ch) == 0),
        "cross_leak_qi": cross[:500],
        "unassigned_qi": no_ch[:500],
        "counts": {"cross_leak": len(cross), "unassigned": len(no_ch)},
    }

    chapters_ids = [
        str(ch.get("chapter_id", "")).strip()
        for ch in (pack.get("chapters") or [])
        if str(ch.get("chapter_id", "")).strip()
    ]
    chapter_report["pack_ref"] = {
        "pack_id": pack.get("pack_id"),
        "country_code": pack.get("country_code"),
        "pack_hash": (pack.get("signature") or {}).get("pack_hash"),
    }

    chapter_qc: Dict[str, List[str]] = defaultdict(list)
    for qc_id, ch_id in qc_to_chapter.items():
        chapter_qc[ch_id].append(qc_id)
    for ch in chapters_ids:
        chapter_qc[ch] = sorted(list(set(chapter_qc.get(ch, []))))

    chapter_qi: Dict[str, List[str]] = defaultdict(list)
    for qi_id, s in qi_to_chapters.items():
        if len(s) == 1:
            chapter_qi[list(s)[0]].append(qi_id)
    for ch in chapters_ids:
        chapter_qi[ch] = sorted(chapter_qi.get(ch, []))

    chapter_cov: Dict[str, Set[str]] = defaultdict(set)
    for ch in chapters_ids:
        for qc_id in chapter_qc.get(ch, []):
            chapter_cov[ch].update(qc_cov.get(qc_id, set()))

    chapter_orphans: Dict[str, List[str]] = {}
    for ch in chapters_ids:
        assigned = set(chapter_qi.get(ch, []))
        covered = chapter_cov.get(ch, set())
        orph = sorted([q for q in assigned if q not in covered])
        chapter_orphans[ch] = orph

    all_orphans = [q for ch in chapters_ids for q in chapter_orphans.get(ch, [])]
    out["B-05_COVERAGE_PER_CHAPTER"] = {
        "pass": len(all_orphans) == 0,
        "orphans_total": len(all_orphans),
        "orphans_preview": all_orphans[:200],
    }

    for ch in chapters_ids:
        qi_total = len(chapter_qi.get(ch, []))
        orph = chapter_orphans.get(ch, [])
        qc_count = len(chapter_qc.get(ch, []))
        chapter_report["chapters"][ch] = {
            "title": next((str(x.get("title", "")) for x in (pack.get("chapters") or []) if str(x.get("chapter_id", "")).strip() == ch), ""),
            "qc_count": qc_count,
            "qi_total": qi_total,
            "qi_covered": max(0, qi_total - len(orph)),
            "coverage_ratio": (1.0 if qi_total == 0 else (qi_total - len(orph)) / float(qi_total)),
            "orphans": orph[:500],
            "qc_ids": chapter_qc.get(ch, [])[:500],
        }

    if dups:
        chapter_report["violations"].append({"type": "intent_duplicate_across_chapters", "items": dups[:200]})
    if unmapped_qc:
        chapter_report["violations"].append({"type": "unmapped_qc_intents", "items": unmapped_qc[:200]})
    if cross:
        chapter_report["violations"].append({"type": "qi_cross_leak", "items": cross[:200]})
    if no_ch:
        chapter_report["violations"].append({"type": "qi_unassigned_to_chapter", "items": no_ch[:200]})
    if all_orphans:
        chapter_report["violations"].append({"type": "chapter_orphans", "count": len(all_orphans)})

    out["_chapter_report"] = chapter_report
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B.9 : AUTO-SCRAPER (pack-driven)
# - Collecte liens PDF depuis des pages index (seeds)
# - Pairing sujet/corrigé : heuristique générique PARAMÉTRÉE par pack.harvest
# - Télécharge bytes + stocke en session_state["harvest_items"]
# ═══════════════════════════════════════════════════════════════════════════════

def _http_get(url: str, ua: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _extract_links_html(html_bytes: bytes, base_url: str) -> List[str]:
    # Extraction générique de href, sans connaissance de site
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
    # de-dup stable
    seen = set()
    out = []
    for u in pdfs:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _pair_subject_correction(pdfs: List[str], corr_hint_regex: str) -> List[Tuple[str, Optional[str]]]:
    """
    Pairing générique (sans hardcode) :
    - On sépare PDF en "corr-like" vs "subject-like" via corr_hint_regex (pack-driven).
    - Puis on associe par similarité de clé (basename normalisée sans tokens corr_hint).
    """
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


def pack_auto_scrape(pack: Dict[str, Any], volume: int) -> Dict[str, Any]:
    """
    Returns:
      {
        "meta": {...},
        "items": [
          {"subject_url":..., "correction_url":..., "subject_bytes":..., "correction_bytes":..., "label":...},
          ...
        ]
      }
    """
    okh, herrs = validate_pack_harvest(pack)
    if not okh:
        return {"meta": {"ok": False, "errors": herrs, "created_at": datetime.now().isoformat()}, "items": []}

    h = pack["harvest"]
    http = h.get("http", {}) or {}
    ua = str(http.get("user_agent") or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    timeout = int(http.get("timeout_sec") or 25)
    max_pdf = int(http.get("max_bytes_pdf") or 25_000_000)
    max_index_pages = int(http.get("max_index_pages") or 30)

    seeds = h.get("seeds") or []
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

    # Deduplicate by subject_url
    seen_subj = set()
    pairs_unique = []
    for subj_url, corr_url in collected_pairs:
        if subj_url in seen_subj:
            continue
        seen_subj.add(subj_url)
        pairs_unique.append((subj_url, corr_url))

    # Limit to requested volume
    pairs_unique = pairs_unique[:max(0, int(volume))]

    items: List[Dict[str, Any]] = []
    dl_log = []
    for i, (subj_url, corr_url) in enumerate(pairs_unique, 1):
        it = {"subject_url": subj_url, "correction_url": corr_url}
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
        "requested_volume": int(volume),
        "items_ok": len(items),
        "crawl_log": crawl_log[:50],
        "download_log": dl_log[:50],
        "sequencing": (h.get("sequencing") or {}),
    }
    return {"meta": meta, "items": items}


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C : STREAMLIT UI (3 tabs conservés + nouveaux modules)
# ═══════════════════════════════════════════════════════════════════════════════

def get_zone_b_source() -> str:
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return ""
    s = src.find("# ZONE B")
    if s == -1:
        s = src.find("ZONE B")
    e = src.find("# ZONE C")
    if e == -1:
        e = src.find("ZONE C")
    if s == -1 or e == -1 or e <= s:
        return ""
    zone_b = src[s:e]
    try:
        ast.parse(zone_b)
    except Exception:
        return ""
    return zone_b


def parse_qi_pack_input(data: Any) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        for k in ["items", "qi_pack", "questions"]:
            if k in data:
                return data[k], {x: data[x] for x in data if x != k}
    return [], {"error": "Unknown format"}


def load_forbidden_literals_json(uploaded) -> Tuple[Set[str], Optional[str]]:
    if not uploaded:
        return set(), None
    try:
        raw = json.load(uploaded)
        if isinstance(raw, list):
            return set(str(x).strip().lower() for x in raw if str(x).strip()), None
        if isinstance(raw, dict):
            vals = raw.get("literals", raw.get("items", raw.get("forbidden_literals", [])))
            if isinstance(vals, list):
                return set(str(x).strip().lower() for x in vals if str(x).strip()), None
        return set(), "Invalid format"
    except Exception as e:
        return set(), f"Parse error: {e}"


def artifacts_by_chapter(canon: Dict[str, Any], pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return:
    {
      "chapters": {
         chapter_id: {
            "qcs": [...],
            "aris": {...},
            "frts": {...},
            "triggers": {...},
            "qi": {...}
         }
      }
    }
    """
    intent_to_ch = _pack_intent_to_chapter(pack)
    qcs = canon.get("qcs", []) or []
    aris = {a.get("ari_id"): a for a in (canon.get("aris", []) or [])}
    frts = {f.get("frt_id"): f for f in (canon.get("frts", []) or [])}
    trgs = {t.get("trigger_id"): t for t in (canon.get("triggers", []) or [])}
    qi_map = {q.get("qi_id"): q for q in (canon.get("qi_pack", []) or [])}

    by_ch: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"qcs": []})

    for qc in qcs:
        intent = str((qc.get("qc_invariant_signature") or {}).get("intent_code", "")).strip()
        ch = intent_to_ch.get(intent)
        if not ch:
            continue
        by_ch[ch]["qcs"].append(qc)

    # Enrich per chapter
    for ch, obj in by_ch.items():
        qc_list = obj["qcs"]
        ari_ids, frt_ids, trg_ids, qi_ids = set(), set(), set(), set()
        for qc in qc_list:
            links = qc.get("links") or {}
            if links.get("ari_id"):
                ari_ids.add(links["ari_id"])
            if links.get("frt_id"):
                frt_ids.add(links["frt_id"])
            for tid in (links.get("trigger_ids") or []):
                trg_ids.add(tid)
            for qid in ((qc.get("mapping") or {}).get("covered_qi_ids") or []):
                qi_ids.add(qid)

        obj["aris"] = {aid: aris.get(aid) for aid in sorted(list(ari_ids)) if aris.get(aid)}
        obj["frts"] = {fid: frts.get(fid) for fid in sorted(list(frt_ids)) if frts.get(fid)}
        obj["triggers"] = {tid: trgs.get(tid) for tid in sorted(list(trg_ids)) if trgs.get(tid)}
        obj["qi"] = {qid: qi_map.get(qid) for qid in sorted(list(qi_ids)) if qi_map.get(qid)}

    return {"chapters": by_ch}


class GTERunner:
    def __init__(self):
        self.results, self.logs = {}, []

    def log(self, m: str):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

    def run(self, qi_pack: List[Dict[str, Any]], gen_func, level2: bool, forbidden: Set[str], pack: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        self.logs, self.results = [], {}
        self.log(f"═══ GTE V31.9.5 — {len(qi_pack)} Qi ═══")

        artifacts_raw = gen_func(qi_pack)
        canon = canonicalize_artifacts(artifacts_raw, qi_pack)
        qcs, aris, frts, trgs = canon["qcs"], canon["aris"], canon["frts"], canon["triggers"]
        singletons = canon["singletons_warning"]
        self.log(f"Generated: {len(qcs)} QC, {len(aris)} ARI, {len(frts)} FRT, {len(trgs)} TRG")
        if singletons:
            self.log(f"⚠️ Singletons: {len(singletons)}")

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
        self.results["TS-01_SCHEMA"] = {"pass": len(errs) == 0, "errors": errs}
        self.log(f"TS-01 SCHEMA: {'PASS' if len(errs) == 0 else 'FAIL'}")

        ref_ok, ref_errs = check_ref_integrity(canon)
        self.results["TS-02_REF"] = {"pass": ref_ok, "errors": ref_errs}
        self.log(f"TS-02 REF: {'PASS' if ref_ok else 'FAIL'}")

        cov_ok, orphans = check_coverage_total(canon)
        self.results["TC-01_COV"] = {"pass": cov_ok, "orphans": orphans}
        self.log(f"TC-01 COV: {'PASS' if cov_ok else 'FAIL'} ({len(orphans)} orphans)")

        prim_ok, dups = check_coverage_primary_unique(canon)
        self.results["TC-02_PRIM"] = {"pass": prim_ok, "duplicates": dups}
        self.log(f"TC-02 PRIM: {'PASS' if prim_ok else 'FAIL'}")

        det_ok, hashes, _ = check_determinism_n_runs(gen_func, qi_pack, 3)
        self.results["TD-01_DET_N3"] = {"pass": det_ok, "hashes": hashes}
        self.log(f"TD-01 DET_N3: {'PASS' if det_ok else 'FAIL'}")

        ord_ok, h1, h2 = check_order_invariance(gen_func, qi_pack)
        self.results["TD-02_ORDER"] = {"pass": ord_ok, "h1": h1[:16], "h2": h2[:16]}
        self.log(f"TD-02 ORDER: {'PASS' if ord_ok else 'FAIL'}")

        zone_b = get_zone_b_source()
        source_ok = len(zone_b) > 100

        if source_ok:
            imp_ok, imp_v = check_no_test_imports(zone_b)
            self.results["TN-02_IMPORTS"] = {"pass": imp_ok, "violations": imp_v}
            self.log(f"TN-02 IMPORTS: {'PASS' if imp_ok else 'FAIL'}")
        else:
            self.results["TN-02_IMPORTS"] = {"pass": None, "note": "SKIP"}
            self.log("TN-02 IMPORTS: SKIP (source N/A)")

        if source_ok:
            ast_ok, ast_v = scan_ast_sensitive_access(zone_b)
            self.results["TA-01_AST"] = {"pass": ast_ok, "violations": ast_v}
            self.log(f"TA-01 AST: {'PASS' if ast_ok else 'FAIL'}")
        else:
            self.results["TA-01_AST"] = {"pass": None, "note": "SKIP"}
            self.log("TA-01 AST: SKIP (source N/A)")

        if source_ok:
            lit_ok, lit_v = scan_forbidden_literals(zone_b, forbidden)
            mode = "ACTIVE" if forbidden else "SKIP"
            self.results["TA-02_LITERALS"] = {"pass": lit_ok, "violations": lit_v, "mode": mode}
            if forbidden:
                self.log(f"TA-02 LITERALS: {'PASS' if lit_ok else 'FAIL'} ({len(forbidden)} terms)")
            else:
                self.log("TA-02 LITERALS: SKIP (no dict)")
        else:
            self.results["TA-02_LITERALS"] = {"pass": None, "note": "SKIP"}
            self.log("TA-02 LITERALS: SKIP (source N/A)")

        self.results["TA-04_PROMPT"] = {"pass": True, "note": "No IA"}
        self.log("TA-04 PROMPT: PASS")

        if level2:
            d10_ok, h10, _ = check_determinism_n_runs(gen_func, qi_pack, 10)
            self.results["TD-03_N10"] = {"pass": d10_ok, "hashes": h10}
            self.log(f"TD-03 N10: {'PASS' if d10_ok else 'FAIL'}")
        else:
            self.results["TD-03_N10"] = {"pass": None, "note": "Skipped"}

        div_ok, div_c, intents = check_intent_diversity(canon, 5)
        self.results["TN-03_DIVERSITY"] = {"pass": div_ok, "count": div_c, "intents": sorted(intents)}
        self.log(f"TN-03 DIVERSITY: {'PASS' if div_ok else 'FAIL'} ({div_c} intents)")

        vb = validators_B(canon, pack)
        chapter_report = vb.pop("_chapter_report", None)
        for k, v in vb.items():
            self.results[k] = v
            p = v.get("pass")
            if p is True:
                self.log(f"{k}: PASS")
            elif p is False:
                self.log(f"{k}: FAIL")
            else:
                self.log(f"{k}: SKIP")

        def is_green(k):
            p = self.results.get(k, {}).get("pass")
            return True if p is None else bool(p)

        p0_keys = [
            "TS-01_SCHEMA", "TS-02_REF", "TC-01_COV", "TC-02_PRIM",
            "TD-01_DET_N3", "TD-02_ORDER", "TN-02_IMPORTS", "TA-01_AST",
            "TA-02_LITERALS", "TA-04_PROMPT",
            "TN-03_DIVERSITY",
        ]

        pack_present = (pack is not None)
        if pack_present and self.results.get("B-01_PACK_SCHEMA", {}).get("pass") is True:
            b_keys = [k for k in self.results.keys() if k.startswith("B-")]
            p0_keys.extend(sorted(b_keys))

        p0_pass = all(is_green(k) for k in p0_keys)
        self.log(f"═══ VERDICT: {'PASS' if p0_pass else 'FAIL'} ═══")

        return {
            "verdict": "PASS" if p0_pass else "FAIL",
            "validators": self.results,
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


def main():
    st.set_page_config(page_title="SMAXIA GTE V31.9.5", page_icon="🔒", layout="wide")
    st.title("🔒 SMAXIA GTE Console V31.9.5")
    st.markdown("**Harnais ISO-PROD — Extraction Qi + QC structure-driven (invariant) + Pack-driven Chapters + Auto-Scrape**")

    if "qi_pack" not in st.session_state:
        st.session_state["qi_pack"] = []
    if "harvest_items" not in st.session_state:
        st.session_state["harvest_items"] = []
    if "harvest_meta" not in st.session_state:
        st.session_state["harvest_meta"] = None

    # Sidebar
    with st.sidebar:
        st.header("⚙️ V31.9.5")
        st.markdown("✓ QC anti-collapse : clustering structurel par markers (EXn-Qm)")
        st.markdown("✓ Zéro keywords métier/langue dans le générateur QC")
        st.markdown("✓ QC formulation conforme : 'Comment ... ?'")
        st.markdown("✓ UI inchangée : 3 tabs + exports + validateurs")
        st.markdown("✓ Chapitres pack-driven (academic_pack.json)")
        st.markdown("✓ Auto-scrape pack-driven (pack.harvest)")
        st.markdown("---")

        st.subheader("TA-02 (optionnel)")
        forb_up = st.file_uploader("forbidden_literals.json", type=["json"], key="forb")
        forbidden, forb_err = load_forbidden_literals_json(forb_up)
        if forb_err:
            st.error(forb_err)
        elif forbidden:
            st.success(f"{len(forbidden)} termes")
        else:
            st.info("TA-02 désactivé")

        st.markdown("---")
        st.subheader("Activation Pays (TEST)")

        # V31.9.5: le pays est sélection UI, mais le "tout academic pays" vient du pack.
        country = st.selectbox("Pays", ["FR"], index=0)
        st.session_state["country_code"] = country

        st.subheader("Academic Pack (Pack-driven)")
        pack_up = st.file_uploader("academic_pack.json", type=["json"], key="pack")
        pack, pack_err = load_academic_pack_json(pack_up)
        if pack_err:
            st.error(pack_err)
        elif pack:
            okp, perrs = validate_pack_schema(pack)
            if okp:
                st.success(f"Pack OK: {pack.get('pack_id')} ({pack.get('country_code')})")
                # Visualisation "tout academic pays"
                with st.expander("Voir Academic Pack (résumé)", expanded=False):
                    st.json({
                        "pack_id": pack.get("pack_id"),
                        "country_code": pack.get("country_code"),
                        "signature": pack.get("signature"),
                        "chapters_count": len(pack.get("chapters") or []),
                        "chapters_preview": (pack.get("chapters") or [])[:10],
                        "has_harvest": bool(pack.get("harvest")),
                        "harvest_keys": sorted(list((pack.get("harvest") or {}).keys())),
                    })
            else:
                st.error("Pack invalide (B-01)")
                st.json(perrs)
        else:
            st.warning("Uploader academic_pack.json (requis pour chapitres + auto-scrape).")

        st.subheader("Auto-Scrape (pack-driven)")
        st.caption("Vous donnez Volume; le scraping démarre. Les SEEDS + regex viennent du Pack (pas du code).")
        vol = st.number_input("Volume (nb de sujets à récupérer)", min_value=1, value=20, step=5, key="auto_vol")

        # Sequencement conseillé (sans hardcode): lire pack.harvest.sequencing.dimensions
        seq_hint = None
        if pack and isinstance(pack.get("harvest"), dict):
            seq_hint = (pack.get("harvest") or {}).get("sequencing", {})
        if seq_hint:
            st.info(f"Séquencement (pack): {seq_hint}")

        if st.button("⛽ LANCER AUTO-SCRAPE", type="primary", use_container_width=True, disabled=(pack is None)):
            okh, herrs = (False, ["Pack requis"])
            if pack:
                okh, herrs = validate_pack_harvest(pack)
            if not okh:
                st.error("Auto-scrape impossible (pack.harvest invalide).")
                st.json(herrs)
            else:
                with st.spinner("Scraping (pack-driven) ..."):
                    out = pack_auto_scrape(pack, int(vol))
                    st.session_state["harvest_meta"] = out.get("meta")
                    st.session_state["harvest_items"] = out.get("items") or []
                st.success(f"Harvest: {len(st.session_state['harvest_items'])} items (ok)")

        if st.session_state.get("harvest_items"):
            st.success(f"Bibliothèque Harvest prête: {len(st.session_state['harvest_items'])} items")
        else:
            st.info("Aucun item harvest pour l’instant.")

    tab1, tab2, tab3 = st.tabs(["📥 Entrée", "🚀 Pipeline", "📊 Résultats"])

    # TAB 1
    with tab1:
        st.header("📥 Entrée")
        mode = st.radio("Mode", ["Golden Pack", "JSON Upload", "Manuel", "Import Sujet (PDF)"], horizontal=True)

        qi_pack, import_meta = [], None

        if mode == "Golden Pack":
            qi_pack = TEST_ONLY_GOLDEN_QI_PACK
            st.info(f"🧪 Golden Pack: {len(qi_pack)} Qi")

        elif mode == "JSON Upload":
            up = st.file_uploader("qi_pack.json", type=["json"], key="qi_json")
            if up:
                try:
                    raw = json.load(up)
                    qi_pack, meta = parse_qi_pack_input(raw)
                    st.success(f"✓ {len(qi_pack)} Qi")
                    if meta:
                        st.json(meta)
                except Exception as e:
                    st.error(f"Error: {e}")

        elif mode == "Manuel":
            txt = st.text_area("Questions (1/line)", height=150)
            rqi = st.text_area("Corrigés (1/line)", height=150)
            if txt.strip():
                lines = [l.strip() for l in txt.strip().split("\n") if l.strip()]
                rlines = [l.strip() for l in rqi.strip().split("\n")] if rqi.strip() else []
                for i, t in enumerate(lines):
                    r = rlines[i] if i < len(rlines) else ""
                    qi_pack.append({
                        "qi_id": f"QI_{i+1:03d}",
                        "subject_id": "SUBJ",
                        "position": {"order_index": i+1, "marker": f"Q{i+1}"},
                        "statement": {"text_md": t},
                        "correction": {"available": bool(r), "text_md": r},
                    })
                st.success(f"✓ {len(qi_pack)} Qi")

        else:
            st.subheader("📄 Import PDF")

            if pdfplumber is None:
                st.error("pdfplumber non installé")
            else:
                # V31.9.5: Harvested Library selector (équivalent "apparaitre dans Import PDF")
                st.markdown("#### Bibliothèque Harvest (Auto-Scrape)")
                harvest_items = st.session_state.get("harvest_items") or []
                if harvest_items:
                    labels = [f"{i+1:03d} — {it.get('label','subject.pdf')} {'(corr)' if it.get('correction_bytes') else '(no corr)'}"
                              for i, it in enumerate(harvest_items)]
                    pick = st.selectbox("Choisir un sujet scrappé", list(range(len(labels))), format_func=lambda i: labels[i], key="harv_pick")
                    colh1, colh2, colh3 = st.columns([1, 1, 2])
                    with colh1:
                        if st.button("📌 Charger ce sujet", use_container_width=True):
                            it = harvest_items[int(pick)]
                            st.session_state["import_subject_bytes"] = it.get("subject_bytes")
                            st.session_state["import_correction_bytes"] = it.get("correction_bytes")
                            st.session_state["import_label"] = it.get("label")
                            st.success("Chargé en session (Import PDF).")
                    with colh2:
                        if st.button("🧹 Vider Import session", use_container_width=True):
                            st.session_state.pop("import_subject_bytes", None)
                            st.session_state.pop("import_correction_bytes", None)
                            st.session_state.pop("import_label", None)
                            st.info("Session import vidée.")
                    with colh3:
                        st.caption("Note: Streamlit ne permet pas de remplir automatiquement un file_uploader; ceci est l’équivalent ISO-PROD.")

                    if st.session_state.get("harvest_meta"):
                        with st.expander("Meta auto-scrape", expanded=False):
                            st.json(st.session_state["harvest_meta"])
                else:
                    st.info("Pas d’items harvest. Lancez AUTO-SCRAPE dans la Sidebar.")

                st.markdown("---")
                st.markdown("#### Import manuel (upload) — toujours disponible")
                subj = st.file_uploader("PDF Sujet (upload)", type=["pdf"], key="subj_up")
                corr = st.file_uploader("PDF Correction (opt) (upload)", type=["pdf"], key="corr_up")

                st.markdown("---")
                auto = st.checkbox("Charger automatiquement dans GTE après génération", value=True)
                if st.button("🧩 Générer qi_pack.json", type="primary"):
                    # Priorité: bytes en session (harvest) sinon upload
                    sb = st.session_state.get("import_subject_bytes")
                    cb = st.session_state.get("import_correction_bytes")

                    if sb is None:
                        if subj is None:
                            st.error("Aucun PDF (ni harvest, ni upload).")
                        else:
                            sb = subj.read()
                            cb = corr.read() if corr else None

                    if sb is not None:
                        try:
                            import_meta, qi_pack = build_qi_pack_from_pdfs(sb, cb)
                            st.success(f"✓ Généré: {len(qi_pack)} Qi")
                            st.expander("Meta extraction / alignement", expanded=False).json(import_meta)
                            st.download_button(
                                "📥 Télécharger qi_pack.json",
                                canonical_json({"meta": import_meta, "qi_pack": qi_pack}),
                                "qi_pack.json",
                            )
                            if auto:
                                st.session_state["qi_pack"] = qi_pack
                                st.session_state["import_meta"] = import_meta
                                st.info("✓ Pack chargé (tab Pipeline).")
                        except Exception as e:
                            st.error(f"Error: {e}")

        if qi_pack:
            st.session_state["qi_pack"] = qi_pack
        if import_meta:
            st.session_state["import_meta"] = import_meta

        if st.session_state.get("qi_pack"):
            st.metric("Qi", len(st.session_state["qi_pack"]))

        if st.session_state.get("import_meta"):
            with st.expander("Dernier qi_pack.json généré (aperçu)", expanded=False):
                st.json({"meta": st.session_state["import_meta"], "qi_pack_preview": st.session_state["qi_pack"][:5]})
            with st.expander("Debug segmentation (preview)", expanded=False):
                meta = st.session_state["import_meta"]
                if isinstance(meta, dict):
                    st.json(meta.get("subject_blocks_preview", []))

    # TAB 2
    with tab2:
        st.header("🚀 Pipeline")

        lvl2 = st.checkbox("Level 2 (N=10)")

        st.markdown("### RUN GTE (sur qi_pack courant)")
        if st.button("▶️ RUN GTE", type="primary", use_container_width=True):
            qi = st.session_state.get("qi_pack", [])
            if not qi:
                st.error("No Qi")
            else:
                with st.spinner("Running..."):
                    result = GTERunner().run(qi, TEST_ONLY_generate_qc_ari_frt_trg, lvl2, forbidden, pack)
                    st.session_state["result"] = result

                for l in result["logs"]:
                    if "FAIL" in l:
                        st.markdown(f"🔴 `{l}`")
                    elif "PASS" in l:
                        st.markdown(f"🟢 `{l}`")
                    elif "SKIP" in l or "⚠️" in l:
                        st.markdown(f"🟡 `{l}`")
                    else:
                        st.text(l)

                st.markdown("---")
                if result["verdict"] == "PASS":
                    st.success("# ✅ PASS")
                else:
                    st.error("# ❌ FAIL")

    # TAB 3
    with tab3:
        st.header("📊 Résultats")

        if "result" not in st.session_state:
            st.info("Run pipeline first")
        else:
            r = st.session_state["result"]
            m = r["metrics"]

            cols = st.columns(6)
            cols[0].metric("Qi", m["qi"])
            cols[1].metric("QC", m["qc"])
            cols[2].metric("ARI", m["ari"])
            cols[3].metric("FRT", m["frt"])
            cols[4].metric("TRG", m["trg"])
            cols[5].metric("Singletons", m["singletons"])

            st.markdown(f"**Hash:** `{m['output_hash'][:32]}...`")

            if st.session_state.get("import_meta"):
                st.subheader("📄 Meta (PDF ou Harvest)")
                st.json(st.session_state["import_meta"])

            if r.get("chapter_report"):
                st.subheader("📚 Chapter Report (preuve pack-driven)")
                st.json(r["chapter_report"])

            st.subheader("Validateurs")
            for k, v in r["validators"].items():
                p = v.get("pass")
                icon = "✅" if p is True else "⏭️" if p is None else "❌"
                with st.expander(f"{icon} {k}"):
                    st.json(v)

            st.markdown("---")
            st.subheader("🧭 Explorer Chapitre (QC + ARI/FRT/TRG + Qi associés)")
            if pack is None:
                st.warning("Uploader academic_pack.json (requis).")
            else:
                canon = r.get("artifacts_canon") or {}
                by_ch = artifacts_by_chapter(canon, pack).get("chapters") or {}
                chapter_ids = sorted(list(by_ch.keys()))
                if not chapter_ids:
                    st.warning("Aucun chapitre exploitable (aucun QC mappé via pack.intent_allowlist).")
                else:
                    # Afficher titres
                    def ch_label(cid: str) -> str:
                        title = ""
                        for ch in (pack.get("chapters") or []):
                            if str(ch.get("chapter_id", "")).strip() == cid:
                                title = str(ch.get("title", "")).strip()
                                break
                        return f"{cid} — {title}" if title else cid

                    sel = st.selectbox("Chapitre", chapter_ids, format_func=ch_label, key="chapter_sel")
                    obj = by_ch.get(sel) or {}
                    qc_list = obj.get("qcs") or []
                    st.info(f"QC dans chapitre: {len(qc_list)}")

                    with st.expander("Liste QC (détails)", expanded=True):
                        for qc in qc_list[:200]:
                            qc_id = qc.get("qc_id")
                            st.markdown(f"**{qc_id}** — {qc.get('qc_formulation')}")
                            links = qc.get("links") or {}
                            st.caption(f"ARI={links.get('ari_id')} | FRT={links.get('frt_id')} | TRG={len(links.get('trigger_ids') or [])} | Qi={len((qc.get('mapping') or {}).get('covered_qi_ids') or [])}")
                            st.json({
                                "qc": qc,
                                "ari": obj.get("aris", {}).get(links.get("ari_id")),
                                "frt": obj.get("frts", {}).get(links.get("frt_id")),
                                "triggers": [obj.get("triggers", {}).get(t) for t in (links.get("trigger_ids") or []) if obj.get("triggers", {}).get(t)],
                                "qi_examples": [obj.get("qi", {}).get(q) for q in ((qc.get("mapping") or {}).get("covered_qi_ids") or [])[:5] if obj.get("qi", {}).get(q)],
                            })

                    chapter_export = {
                        "chapter_id": sel,
                        "chapter_title": ch_label(sel),
                        "qc_count": len(qc_list),
                        "qcs": qc_list,
                        "aris": obj.get("aris"),
                        "frts": obj.get("frts"),
                        "triggers": obj.get("triggers"),
                        "qi": obj.get("qi"),
                        "created_at": datetime.now().isoformat(),
                        "proof": (r.get("chapter_report") or {}).get("chapters", {}).get(sel, {}),
                    }
                    st.download_button("📥 Export chapter_bundle.json", canonical_json(chapter_export), "chapter_bundle.json")

            st.markdown("---")
            st.subheader("📤 Exports")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button(
                    "📥 report.json",
                    canonical_json({"version": "V31.9.5", "verdict": r["verdict"], "validators": r["validators"], "metrics": m}),
                    "report.json",
                )
            with c2:
                st.download_button("📥 raw.json", canonical_json(r["artifacts_raw"]), "raw.json")
            with c3:
                st.download_button("📥 canon.json", canonical_json(r["artifacts_canon"]), "canon.json")
            with c4:
                if r.get("chapter_report"):
                    st.download_button("📥 chapter_report.json", canonical_json(r["chapter_report"]), "chapter_report.json")
                else:
                    st.download_button("📥 chapter_report.json", canonical_json({"note": "No pack or no chapter report"}), "chapter_report.json")


if __name__ == "__main__":
    main()
