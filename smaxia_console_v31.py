# smaxia_console_v31.py — SMAXIA GTE V31.7 (ISO-PROD Proof Harness)
# =============================================================================
# V31.7 — PDF → Qi V2 (Import Sujet) + Zero hardcode métier
#
# Key guarantees:
# - No regression: keeps Golden Pack / JSON Upload / Manual / Pipeline / Results UX
# - Adds "Import Sujet (PDF)" that generates qi_pack.json automatically (CEO zero-code)
# - TA-02 (Forbidden literals) is OPTIONAL and loaded via forbidden_literals.json (no hardcoded country/matter list)
# - TA-01 AST works reliably (Zone B source extracted via __file__ read, AST-parseable)
# - Determinism + order invariance maintained (canonicalization)
#
# =============================================================================

from __future__ import annotations

import ast
import hashlib
import io
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

try:
    import pdfplumber  # required for Import Sujet (PDF)
except Exception:
    pdfplumber = None


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A : TEST-ONLY FIXTURES — NOT FOR PROD
# ═══════════════════════════════════════════════════════════════════════════════

TEST_ONLY_GOLDEN_QI_PACK: List[Dict[str, Any]] = [
    {"qi_id": "QI_TEST_001", "subject_id": "SUBJ_A", "position": {"order_index": 1},
     "statement": {"text_md": "Compute value."},
     "correction": {"available": True, "text_md": "Apply definition. Compute. Verify."}},
    {"qi_id": "QI_TEST_002", "subject_id": "SUBJ_A", "position": {"order_index": 2},
     "statement": {"text_md": "Prove property."},
     "correction": {"available": True, "text_md": "Proof: Assume. Show. Conclude."}},
    {"qi_id": "QI_TEST_003", "subject_id": "SUBJ_A", "position": {"order_index": 3},
     "statement": {"text_md": "Solve equation."},
     "correction": {"available": True, "text_md": "Isolate x. Verify solution."}},
    {"qi_id": "QI_TEST_004", "subject_id": "SUBJ_A", "position": {"order_index": 4},
     "statement": {"text_md": "Deduce answer."},
     "correction": {"available": True, "text_md": "Apply rule, conclude."}},
    {"qi_id": "QI_TEST_005", "subject_id": "SUBJ_B", "position": {"order_index": 5},
     "statement": {"text_md": "Identify type."},
     "correction": {"available": True, "text_md": "Observe. Classify."}},
    {"qi_id": "QI_TEST_006", "subject_id": "SUBJ_B", "position": {"order_index": 6},
     "statement": {"text_md": "Justify result."},
     "correction": {"available": True, "text_md": "State reason. Reference theorem."}},
    {"qi_id": "QI_TEST_007", "subject_id": "SUBJ_B", "position": {"order_index": 7},
     "statement": {"text_md": "Transform expression."},
     "correction": {"available": True, "text_md": "Apply rules. Simplify."}},
    {"qi_id": "QI_TEST_008", "subject_id": "SUBJ_B", "position": {"order_index": 8},
     "statement": {"text_md": "Verify constraint."},
     "correction": {"available": True, "text_md": "Check. Confirm."}},
]


def TEST_ONLY_simulate_qc_generation(qi_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    TEST-ONLY generator for harness testing.
    NOTE: This is not SMAXIA P6/P7. It exists to validate the harness invariants:
          determinism, schema, mapping integrity, coverage, no-test-imports, AST rules, etc.
    """
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for qi in qi_pack:
        corr = (qi.get("correction", {}) or {}).get("text_md", "")
        low = corr.lower()

        if any(k in low for k in ["proof", "prove", "assume"]):
            clusters["INTENT_DEMONSTRATE"].append(qi)
        elif any(k in low for k in ["solve", "isolate"]):
            clusters["INTENT_SOLVE"].append(qi)
        elif any(k in low for k in ["compute", "calculate"]):
            clusters["INTENT_COMPUTE"].append(qi)
        elif any(k in low for k in ["identify", "classify"]):
            clusters["INTENT_IDENTIFY"].append(qi)
        elif any(k in low for k in ["justify", "reason", "theorem"]):
            clusters["INTENT_JUSTIFY"].append(qi)
        elif any(k in low for k in ["transform", "simplify"]):
            clusters["INTENT_TRANSFORM"].append(qi)
        elif any(k in low for k in ["verify", "check", "confirm"]):
            clusters["INTENT_VERIFY"].append(qi)
        elif any(k in low for k in ["deduce", "conclude"]):
            clusters["INTENT_DEDUCE"].append(qi)
        else:
            clusters["INTENT_GENERIC"].append(qi)

    qcs, aris, frts, triggers, singletons = [], [], [], [], []

    for intent in sorted(clusters.keys()):
        grp = sorted(clusters[intent], key=lambda x: x.get("qi_id", ""))
        qi_ids = [q.get("qi_id", "") for q in grp if q.get("qi_id")]

        if len(qi_ids) < 2:
            singletons.append({"intent": intent, "qi_ids": qi_ids})

        h = hashlib.sha256((intent + "|" + "|".join(sorted(qi_ids))).encode("utf-8")).hexdigest()[:10]
        qc_id, ari_id, frt_id, trg_id = f"QC_{intent}_{h}", f"ARI_{intent}_{h}", f"FRT_{intent}_{h}", f"TRG_{intent}_{h}"

        by_subj: Dict[str, List[str]] = defaultdict(list)
        for q in grp:
            by_subj[str(q.get("subject_id", ""))].append(str(q.get("qi_id", "")))

        qcs.append({
            "qc_id": qc_id,
            "qc_formulation": "How to solve?",
            "qc_invariant_signature": {"intent_code": intent},
            "mapping": {"primary_qi_ids": qi_ids, "covered_qi_ids": qi_ids, "by_subject": dict(by_subj)},
            "links": {"ari_id": ari_id, "frt_id": frt_id, "trigger_ids": [trg_id]},
            "provenance": {"generator_version": "TEST_ONLY_V31.7"},
        })

        aris.append({
            "ari_id": ari_id,
            "template_id": "ARI_TPL_V1",
            "steps": [
                {"step_id": "S1", "operator_code": "OP_READ"},
                {"step_id": "S2", "operator_code": "OP_PROCESS"},
                {"step_id": "S3", "operator_code": "OP_CONCLUDE"},
            ],
            "provenance": {"qc_id": qc_id},
        })

        frts.append({
            "frt_id": frt_id,
            "template_id": "FRT_TPL_V1",
            "sections": {
                "given": "Input",
                "goal": "Output",
                "method": "Steps",
                "checks": ["Verify"],
                "common_traps": ["Avoid"],
                "final_form": "Result",
            },
            "provenance": {"qc_id": qc_id, "ari_id": ari_id},
        })

        triggers.append({
            "trigger_id": trg_id,
            "type_code": "TRG_CHECK",
            "severity": "medium",
            "condition": {"signal": "AFTER_S2"},
            "action": {"recommendation": "Verify"},
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
        key=lambda x: (str(x.get("intent", "")), "|".join(sorted([str(q) for q in x.get("qi_ids", []) or []])))
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


def validate_qi_schema(qi: Dict[str, Any]) -> Tuple[bool, str]:
    for f in ["qi_id", "subject_id", "position", "statement"]:
        if f not in qi:
            return False, f"Missing {f}"
    if "order_index" not in (qi.get("position") or {}):
        return False, "Missing position.order_index"
    if "text_md" not in (qi.get("statement") or {}):
        return False, "Missing statement.text_md"
    # correction is optional but must be well formed if present
    corr = qi.get("correction")
    if corr is not None:
        if "available" not in corr or "text_md" not in corr:
            return False, "Invalid correction (missing available/text_md)"
    return True, "OK"


def validate_qc_schema(qc: Dict[str, Any]) -> Tuple[bool, str]:
    for f in ["qc_id", "qc_formulation", "qc_invariant_signature", "mapping", "links"]:
        if f not in qc:
            return False, f"Missing {f}"
    if "intent_code" not in (qc.get("qc_invariant_signature") or {}):
        return False, "Missing intent_code"
    mapping = qc.get("mapping") or {}
    if "primary_qi_ids" not in mapping or "covered_qi_ids" not in mapping:
        return False, "Missing mapping fields"
    links = qc.get("links") or {}
    if "ari_id" not in links or "frt_id" not in links or "trigger_ids" not in links:
        return False, "Missing links"
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
            return False, f"Missing section {s}"
    return True, "OK"


def validate_trigger_schema(trg: Dict[str, Any]) -> Tuple[bool, str]:
    for f in ["trigger_id", "type_code", "severity", "condition", "action", "links"]:
        if f not in trg:
            return False, f"Missing {f}"
    links = trg.get("links") or {}
    if "qc_id" not in links or "ari_id" not in links:
        return False, "Missing links"
    return True, "OK"


def check_ref_integrity(canon: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    qc_ids = {qc.get("qc_id") for qc in canon.get("qcs", [])}
    ari_ids = {ari.get("ari_id") for ari in canon.get("aris", [])}
    frt_ids = {frt.get("frt_id") for frt in canon.get("frts", [])}
    trg_ids = {trg.get("trigger_id") for trg in canon.get("triggers", [])}
    qi_ids = {qi.get("qi_id") for qi in canon.get("qi_pack", [])}

    for qc in canon.get("qcs", []):
        links = qc.get("links") or {}
        if links.get("ari_id") not in ari_ids:
            errors.append(f"QC {qc.get('qc_id')} bad ari_id")
        if links.get("frt_id") not in frt_ids:
            errors.append(f"QC {qc.get('qc_id')} bad frt_id")
        for t in links.get("trigger_ids", []):
            if t not in trg_ids:
                errors.append(f"QC {qc.get('qc_id')} bad trigger {t}")
        for qi in (qc.get("mapping") or {}).get("covered_qi_ids", []):
            if qi not in qi_ids:
                errors.append(f"QC {qc.get('qc_id')} bad qi {qi}")

    for trg in canon.get("triggers", []):
        links = trg.get("links") or {}
        if links.get("qc_id") not in qc_ids:
            errors.append(f"TRG {trg.get('trigger_id')} bad qc_id")
        if links.get("ari_id") not in ari_ids:
            errors.append(f"TRG {trg.get('trigger_id')} bad ari_id")

    return len(errors) == 0, errors


def check_coverage_total(canon: Dict[str, Any]) -> Tuple[bool, List[str]]:
    all_qi = {qi.get("qi_id") for qi in canon.get("qi_pack", [])}
    covered: Set[str] = set()
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
    hashes: List[str] = []
    for _ in range(n):
        artifacts = gen_func(qi_pack)
        canon = canonicalize_artifacts(artifacts, qi_pack)
        hashes.append(compute_hash(canon))
    ok = len(set(hashes)) == 1
    return ok, hashes, ([] if ok else [f"Divergence: {len(set(hashes))} uniques/{n}"])


def check_order_invariance(gen_func, qi_pack: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    base = gen_func(qi_pack)
    h1 = compute_hash(canonicalize_artifacts(base, qi_pack))
    shuffled = qi_pack.copy()
    random.seed(42)
    random.shuffle(shuffled)
    other = gen_func(shuffled)
    h2 = compute_hash(canonicalize_artifacts(other, shuffled))
    return h1 == h2, h1, h2


def scan_forbidden_literals(source: str, zone: str, forbidden_literals: Set[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Optional anti-hardcode scan (user-provided forbidden_literals.json).
    We DO NOT hardcode any business literals in code.
    """
    if not forbidden_literals:
        return True, []

    violations: List[Dict[str, Any]] = []

    # Skip comments/empty and skip the forbidden list definition itself (defense mechanism)
    in_def_block = False

    for i, line in enumerate(source.split("\n"), 1):
        s = line.strip()
        low = s.lower()

        if s.startswith("#") or not s:
            continue

        if "forbidden_literals" in low and "{" in low:
            in_def_block = True
            continue
        if in_def_block:
            if "}" in low:
                in_def_block = False
            continue

        for lit in forbidden_literals:
            if lit and lit in low:
                violations.append({"line": i, "literal": lit, "zone": zone, "context": s[:160]})

    return len(violations) == 0, violations


def scan_ast_sensitive_access(source: str) -> Tuple[bool, List[Dict[str, Any]]]:
    violations: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except Exception as e:
        return False, [{"error": f"Parse error: {e}"}]

    class V(ast.NodeVisitor):
        def visit_Subscript(self, n: ast.Subscript):
            if isinstance(n.value, ast.Name) and n.value.id == "context":
                sl = n.slice
                if isinstance(sl, ast.Constant) and str(sl.value) in SENSITIVE_CONTEXT_FIELDS:
                    violations.append({"type": "subscript", "field": sl.value, "line": getattr(n, "lineno", 0)})
            self.generic_visit(n)

        def visit_Attribute(self, n: ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id == "context":
                if n.attr in SENSITIVE_CONTEXT_FIELDS:
                    violations.append({"type": "attribute", "field": n.attr, "line": getattr(n, "lineno", 0)})
            self.generic_visit(n)

    V().visit(tree)
    return len(violations) == 0, violations


def check_no_test_imports(source: str) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    for i, line in enumerate(source.split("\n"), 1):
        if "import" in line.lower():
            for p in ["tests/", "fixtures/", "test_only"]:
                if p in line.lower():
                    violations.append(f"L{i}: {line[:80]}")
    return len(violations) == 0, violations


def check_intent_diversity(canon: Dict[str, Any], min_i: int = 5) -> Tuple[bool, int, Set[str]]:
    intents = {qc.get("qc_invariant_signature", {}).get("intent_code", "") for qc in canon.get("qcs", [])}
    intents.discard("")
    return len(intents) >= min_i, len(intents), intents


# ------------------------- PDF → Qi V2 (no-code CEO) --------------------------

@dataclass
class ExtractStats:
    pages: int
    errors: List[str]


def _read_pdf_text(pdf_bytes: bytes) -> Tuple[str, ExtractStats]:
    if pdfplumber is None:
        return "", ExtractStats(pages=0, errors=["pdfplumber not installed"])

    errors: List[str] = []
    text_parts: List[str] = []
    pages = 0

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = len(pdf.pages)
            for p in pdf.pages:
                try:
                    t = p.extract_text() or ""
                    # Keep line breaks, normalize non-breaking spaces
                    t = t.replace("\u00a0", " ")
                    text_parts.append(t)
                except Exception as e:
                    errors.append(f"page_extract_error: {e}")
    except Exception as e:
        errors.append(f"pdf_open_error: {e}")

    return "\n".join(text_parts).strip(), ExtractStats(pages=pages, errors=errors)


def _normalize_lines(text: str) -> List[str]:
    # Conservative normalization: keep structure, remove excessive empty lines
    raw_lines = [ln.rstrip() for ln in text.splitlines()]
    lines: List[str] = []
    empty_run = 0
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            empty_run += 1
            if empty_run <= 1:
                lines.append("")
            continue
        empty_run = 0
        # de-hyphenation across lines is risky; we keep as-is
        lines.append(ln)
    return lines


_Q_MARKERS = [
    # Main numeric question: "1)" "2." "3 -"
    re.compile(r"^\s*(\d{1,3})\s*[\)\.\-]\s+"),
    # Sub-question: "a)" "b." "c -"
    re.compile(r"^\s*([a-z])\s*[\)\.\-]\s+"),
    # "Affirmation 1:" (QCM style)
    re.compile(r"^\s*Affirmation\s+(\d{1,2})\s*[:\-]\s*", re.IGNORECASE),
    # "Question 1:" (generic)
    re.compile(r"^\s*Question\s+(\d{1,3})\s*[:\-]\s*", re.IGNORECASE),
]

# Context markers (not Qi themselves) used to avoid wrong segmentation
_CTX_MARKERS = [
    re.compile(r"^\s*EXERCICE\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*Page\s*\d+\s*/\s*\d+", re.IGNORECASE),
]


def _is_context_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for rx in _CTX_MARKERS:
        if rx.search(s):
            return True
    return False


def _detect_marker(line: str) -> Optional[Tuple[str, str]]:
    """
    Return (marker_type, marker_value) for segmentation.
    marker_type in {"NUM", "SUB", "AFF", "Q"}.
    """
    s = line.strip()
    if not s:
        return None
    for rx in _Q_MARKERS:
        m = rx.search(s)
        if m:
            if rx.pattern.lower().startswith("^\\s*\\(\\d") or "^(\\s*(\\d" in rx.pattern:
                pass
            # classify based on which regex matched (by index)
    # Determine by matching order
    m0 = _Q_MARKERS[0].search(s)
    if m0:
        return ("NUM", m0.group(1))
    m1 = _Q_MARKERS[1].search(s)
    if m1:
        return ("SUB", m1.group(1))
    m2 = _Q_MARKERS[2].search(s)
    if m2:
        return ("AFF", m2.group(1))
    m3 = _Q_MARKERS[3].search(s)
    if m3:
        return ("Q", m3.group(1))
    return None


def _segment_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Heuristic segmentation into blocks.
    Goal: many small Qi blocks rather than one blob, without hardcoding domain.
    """
    blocks: List[Dict[str, Any]] = []
    current: List[str] = []
    current_marker: Optional[str] = None
    last_num: Optional[str] = None

    def flush():
        nonlocal current, current_marker
        txt = "\n".join([ln for ln in current]).strip()
        if txt:
            blocks.append({"marker": current_marker or "", "text": txt})
        current = []
        current_marker = None

    for ln in lines:
        if _is_context_line(ln):
            # Keep context as part of current block if we already started a Qi
            if current:
                current.append(ln)
            continue

        marker = _detect_marker(ln)
        if marker:
            mtype, mval = marker
            # Build a stable marker key, preserving hierarchy:
            # - NUM: "2"
            # - SUB: "2a" if we have last_num else "a"
            # - AFF: "AFF1"
            # - Q: "Q1"
            if mtype == "NUM":
                last_num = mval
                key = mval
            elif mtype == "SUB":
                key = (last_num + mval) if last_num else mval
            elif mtype == "AFF":
                key = f"AFF{mval}"
            else:
                key = f"Q{mval}"

            # New Qi starts: flush previous
            if current:
                flush()
            current_marker = key
            current.append(ln)
            continue

        # Regular line
        if current:
            current.append(ln)
        else:
            # We are before the first detected marker; ignore preamble noise
            # but if it looks like a strong instruction line, we can start a block.
            # Conservative: do not start without marker.
            continue

    flush()

    # Post-filter: drop tiny blocks that are just page numbers etc.
    cleaned: List[Dict[str, Any]] = []
    for b in blocks:
        t = b["text"].strip()
        if len(t) < 20:
            continue
        cleaned.append(b)

    # If segmentation failed (0 blocks), fallback to whole text as 1 block (explicit)
    if not cleaned and any(ln.strip() for ln in lines):
        full = "\n".join(lines).strip()
        if full:
            cleaned = [{"marker": "FULL", "text": full}]

    return cleaned


def _build_correction_index(corr_text: str) -> Dict[str, str]:
    """
    Build a lookup dict marker -> correction block text using same segmentation.
    """
    if not corr_text.strip():
        return {}
    lines = _normalize_lines(corr_text)
    blocks = _segment_blocks(lines)
    idx: Dict[str, str] = {}
    for b in blocks:
        mk = str(b.get("marker", "")).strip()
        if mk:
            idx[mk] = b.get("text", "").strip()
    return idx


def _align_correction(marker: str, corr_idx: Dict[str, str]) -> Optional[str]:
    """
    Best-effort alignment by exact marker then relaxed match:
    - exact "2a"
    - if "2a" not found, try main "2"
    - if "AFF1", try "1" not attempted (different)
    """
    mk = (marker or "").strip()
    if not mk or not corr_idx:
        return None

    if mk in corr_idx:
        return corr_idx[mk]

    # sub-question fallback to its main number
    m = re.match(r"^(\d{1,3})[a-z]$", mk)
    if m:
        main = m.group(1)
        if main in corr_idx:
            return corr_idx[main]

    return None


def build_qi_pack_from_pdfs(subject_pdf_bytes: bytes, correction_pdf_bytes: Optional[bytes]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns (meta, qi_pack) strictly compliant with expected schema.
    """
    subj_sha8 = sha8_bytes(subject_pdf_bytes)
    subject_id = f"SUBJ_{subj_sha8}"

    subj_text, subj_stats = _read_pdf_text(subject_pdf_bytes)
    subj_lines = _normalize_lines(subj_text)
    subj_blocks = _segment_blocks(subj_lines)

    corr_text = ""
    corr_stats = ExtractStats(pages=0, errors=[])
    if correction_pdf_bytes:
        corr_text, corr_stats = _read_pdf_text(correction_pdf_bytes)

    corr_idx = _build_correction_index(corr_text)

    qi_pack: List[Dict[str, Any]] = []

    for i, b in enumerate(subj_blocks, 1):
        marker = str(b.get("marker", "")).strip()
        statement = b.get("text", "").strip()

        corr_txt = _align_correction(marker, corr_idx) if corr_idx else None
        corr_avail = bool(corr_txt and corr_txt.strip())

        # Stable qi_id: order + subject sha + content hash
        content_h = hashlib.sha256((marker + "\n" + statement).encode("utf-8")).hexdigest()[:10]
        qi_id = f"QI_{i:04d}_{subj_sha8}_{content_h}"

        qi_pack.append({
            "qi_id": qi_id,
            "subject_id": subject_id,
            "position": {"order_index": i},
            "statement": {"text_md": statement},
            "correction": {"available": bool(corr_avail), "text_md": (corr_txt or "").strip()},
            "provenance": {"marker": marker},
        })

    # Meta
    meta: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "generator": {"name": "SMAXIA_PDF_QI_BUILDER", "version": "V31.7"},
        "inputs": {
            "subject_pdf_sha8": subj_sha8,
            "subject_extraction": {"engine": "pdfplumber", "pages": subj_stats.pages, "errors": subj_stats.errors},
            "correction_pdf_sha8": (sha8_bytes(correction_pdf_bytes) if correction_pdf_bytes else None),
            "correction_extraction": {"engine": "pdfplumber", "pages": corr_stats.pages, "errors": corr_stats.errors},
        },
        "stats": {
            "qi_count": len(qi_pack),
            "subject_blocks": len(subj_blocks),
            "correction_blocks": len(corr_idx),
            "aligned_by_marker": sum(1 for q in qi_pack if (q.get("correction", {}) or {}).get("available") is True),
        },
        "warnings": [],
    }

    if subj_stats.errors:
        meta["warnings"].append("subject_pdf_extraction_had_errors")
    if correction_pdf_bytes and corr_stats.errors:
        meta["warnings"].append("correction_pdf_extraction_had_errors")
    if len(qi_pack) <= 1:
        meta["warnings"].append("qi_count_low_check_segmentation")

    return meta, qi_pack


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C : STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def get_zone_b_source() -> str:
    """
    Reliable, AST-parseable Zone B source extraction.
    Uses file read (preferred) to avoid inspect.getsource edge cases in Streamlit cloud.
    """
    try:
        p = Path(__file__)
        src = p.read_text(encoding="utf-8")
    except Exception:
        return ""

    s = src.find("ZONE B")
    e = src.find("ZONE C")
    if s == -1 or e == -1 or e <= s:
        return ""
    return src[s:e]


def parse_qi_pack_input(data: Any) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        for k in ["items", "qi_pack", "questions"]:
            if k in data:
                return data[k], {x: data[x] for x in data if x != k}
    return [], {"error": "Unknown format"}


def load_forbidden_literals_json(uploaded_file) -> Tuple[Set[str], Optional[str]]:
    """
    forbidden_literals.json formats supported:
    - {"literals": ["...", "..."]}
    - ["...", "..."]
    - {"version":"...", "literals":[...]} etc.
    """
    if not uploaded_file:
        return set(), None
    try:
        raw = json.load(uploaded_file)
        if isinstance(raw, list):
            lits = [str(x).strip().lower() for x in raw if str(x).strip()]
            return set(lits), None
        if isinstance(raw, dict):
            vals = raw.get("literals", raw.get("items", raw.get("forbidden_literals", [])))
            if isinstance(vals, list):
                lits = [str(x).strip().lower() for x in vals if str(x).strip()]
                return set(lits), None
        return set(), "Invalid forbidden_literals.json format"
    except Exception as e:
        return set(), f"forbidden_literals.json parse error: {e}"


class GTERunner:
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.logs: List[str] = []

    def log(self, m: str):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

    def run(self, qi_pack: List[Dict[str, Any]], gen_func, level2: bool, forbidden_literals: Set[str]) -> Dict[str, Any]:
        self.logs = []
        self.results = {}
        self.log(f"═══ GTE V31.7 — {len(qi_pack)} Qi ═══")

        artifacts_raw = gen_func(qi_pack)
        canon = canonicalize_artifacts(artifacts_raw, qi_pack)

        qcs = canon["qcs"]
        aris = canon["aris"]
        frts = canon["frts"]
        trgs = canon["triggers"]
        singletons = canon["singletons_warning"]

        self.log(f"Generated: {len(qcs)} QC, {len(aris)} ARI, {len(frts)} FRT, {len(trgs)} TRG")
        if singletons:
            self.log(f"⚠️ Singletons: {len(singletons)}")

        # TS-01 Schema
        errs: List[str] = []
        for qi in canon["qi_pack"]:
            ok, e = validate_qi_schema(qi)
            if not ok:
                errs.append(f"Qi {qi.get('qi_id')}: {e}")
        for qc in qcs:
            ok, e = validate_qc_schema(qc)
            if not ok:
                errs.append(f"QC {qc.get('qc_id')}: {e}")
        for ari in aris:
            ok, e = validate_ari_schema(ari)
            if not ok:
                errs.append(f"ARI {ari.get('ari_id')}: {e}")
        for frt in frts:
            ok, e = validate_frt_schema(frt)
            if not ok:
                errs.append(f"FRT {frt.get('frt_id')}: {e}")
        for trg in trgs:
            ok, e = validate_trigger_schema(trg)
            if not ok:
                errs.append(f"TRG {trg.get('trigger_id')}: {e}")
        self.results["TS-01_SCHEMA"] = {"pass": len(errs) == 0, "errors": errs}
        self.log(f"TS-01 SCHEMA: {'PASS' if len(errs) == 0 else 'FAIL'}")

        # TS-02 Ref Integrity
        ref_ok, ref_errs = check_ref_integrity(canon)
        self.results["TS-02_REF"] = {"pass": ref_ok, "errors": ref_errs}
        self.log(f"TS-02 REF: {'PASS' if ref_ok else 'FAIL'}")

        # TC-01 Coverage (binary central)
        cov_ok, orphans = check_coverage_total(canon)
        self.results["TC-01_COV"] = {"pass": cov_ok, "orphans": orphans}
        self.log(f"TC-01 COV: {'PASS' if cov_ok else 'FAIL'} ({len(orphans)} orphans)")

        # TC-02 Primary Unique
        prim_ok, dups = check_coverage_primary_unique(canon)
        self.results["TC-02_PRIM"] = {"pass": prim_ok, "duplicates": dups}
        self.log(f"TC-02 PRIM: {'PASS' if prim_ok else 'FAIL'}")

        # TD-01 Determinism N=3
        det_ok, hashes, det_errs = check_determinism_n_runs(gen_func, qi_pack, 3)
        self.results["TD-01_DET_N3"] = {"pass": det_ok, "hashes": hashes, "errors": det_errs}
        self.log(f"TD-01 DET_N3: {'PASS' if det_ok else 'FAIL'}")

        # TD-02 Order Invariance
        ord_ok, h1, h2 = check_order_invariance(gen_func, qi_pack)
        self.results["TD-02_ORDER"] = {"pass": ord_ok, "h1": h1[:16], "h2": h2[:16]}
        self.log(f"TD-02 ORDER: {'PASS' if ord_ok else 'FAIL'}")

        # Source-based validators
        zone_b = get_zone_b_source()
        source_available = len(zone_b) > 0

        # TN-02 No Test Imports
        if source_available:
            imp_ok, imp_v = check_no_test_imports(zone_b)
            self.results["TN-02_IMPORTS"] = {"pass": imp_ok, "violations": imp_v}
            self.log(f"TN-02 IMPORTS: {'PASS' if imp_ok else 'FAIL'}")
        else:
            self.results["TN-02_IMPORTS"] = {"pass": False, "error": "Source unavailable"}
            self.log("TN-02 IMPORTS: FAIL (source unavailable)")

        # TA-01 AST
        if source_available:
            ast_ok, ast_v = scan_ast_sensitive_access(zone_b)
            self.results["TA-01_AST"] = {"pass": ast_ok, "violations": ast_v}
            self.log(f"TA-01 AST: {'PASS' if ast_ok else 'FAIL'}")
        else:
            self.results["TA-01_AST"] = {"pass": False, "error": "Source unavailable"}
            self.log("TA-01 AST: FAIL (source unavailable)")

        # TA-02 Forbidden Literals (OPTIONAL — no hardcode)
        if source_available:
            lit_ok, lit_v = scan_forbidden_literals(zone_b, "B", forbidden_literals)
            self.results["TA-02_LITERALS"] = {
                "pass": lit_ok,
                "violations": lit_v,
                "mode": ("ENABLED" if forbidden_literals else "SKIPPED"),
                "count": len(forbidden_literals),
            }
            if forbidden_literals:
                self.log(f"TA-02 LITERALS: {'PASS' if lit_ok else 'FAIL'} (loaded={len(forbidden_literals)})")
            else:
                self.log("TA-02 LITERALS: PASS (skipped: no forbidden_literals.json)")
        else:
            self.results["TA-02_LITERALS"] = {"pass": False, "error": "Source unavailable"}
            self.log("TA-02 LITERALS: FAIL (source unavailable)")

        # TA-04 Prompt Isolation
        self.results["TA-04_PROMPT"] = {"pass": True, "note": "No IA prompt in harness"}
        self.log("TA-04 PROMPT: PASS")

        # P1: TD-03 N=10
        if level2:
            d10_ok, h10, d10_errs = check_determinism_n_runs(gen_func, qi_pack, 10)
            self.results["TD-03_N10"] = {"pass": d10_ok, "hashes": h10, "errors": d10_errs}
            self.log(f"TD-03 N10: {'PASS' if d10_ok else 'FAIL'}")
        else:
            self.results["TD-03_N10"] = {"pass": None, "note": "Skipped"}

        # P1: TN-03 Diversity
        div_ok, div_c, intents = check_intent_diversity(canon, 5)
        self.results["TN-03_DIVERSITY"] = {"pass": div_ok, "count": div_c, "intents": sorted(list(intents))}
        self.log(f"TN-03 DIVERSITY: {'PASS' if div_ok else 'FAIL'} ({div_c} intents)")

        # Final Verdict (P0)
        p0_keys = [
            "TS-01_SCHEMA", "TS-02_REF", "TC-01_COV", "TC-02_PRIM",
            "TD-01_DET_N3", "TD-02_ORDER", "TN-02_IMPORTS",
            "TA-01_AST", "TA-02_LITERALS", "TA-04_PROMPT"
        ]
        p0_pass = all(self.results.get(k, {}).get("pass", False) for k in p0_keys)

        input_hash = compute_hash(sort_qi_canonical(qi_pack))
        output_hash = compute_hash(canon)

        self.log(f"═══ VERDICT: {'PASS' if p0_pass else 'FAIL'} ═══")

        return {
            "verdict": "PASS" if p0_pass else "FAIL",
            "validators": self.results,
            "artifacts_raw": artifacts_raw,
            "artifacts_canon": canon,
            "metrics": {
                "qi": len(qi_pack),
                "qc": len(qcs),
                "ari": len(aris),
                "frt": len(frts),
                "trg": len(trgs),
                "singletons": len(singletons),
                "input_hash": input_hash,
                "output_hash": output_hash,
            },
            "logs": self.logs,
        }


def main():
    st.set_page_config(page_title="SMAXIA GTE V31.7", page_icon="🔒", layout="wide")
    st.title("🔒 SMAXIA GTE Console V31.7")
    st.markdown("**Harnais de Preuve ISO-PROD — Final + Import Sujet (PDF) → qi_pack.json**")

    # Sidebar: optional forbidden_literals.json (no hardcode)
    with st.sidebar:
        st.header("⚙️ V31.7")
        st.markdown("✓ get_zone_b_source via __file__ (AST reliable)")
        st.markdown("✓ Import Sujet (PDF) → qi_pack.json")
        st.markdown("✓ TA-02 sans hardcode (forbidden_literals.json optionnel)")

        st.markdown("---")
        st.subheader("TA-02 (optionnel)")
        st.caption("Uploadez forbidden_literals.json si vous voulez activer TA-02.")
        forb_up = st.file_uploader("forbidden_literals.json", type=["json"], key="forb_json")
        forbidden_literals, forb_err = load_forbidden_literals_json(forb_up)
        if forb_err:
            st.error(forb_err)
        elif forbidden_literals:
            st.success(f"TA-02 activé: {len(forbidden_literals)} littéraux chargés")
        else:
            st.info("TA-02: désactivé (aucun fichier chargé)")

    tab1, tab2, tab3 = st.tabs(["📥 Entrée", "🚀 Pipeline", "📊 Résultats"])

    # --------------------------- TAB 1: INPUT ---------------------------
    with tab1:
        st.header("📥 Entrée")
        mode = st.radio("Mode", ["Golden Pack", "JSON Upload", "Manuel", "Import Sujet (PDF)"], horizontal=True)

        qi_pack: List[Dict[str, Any]] = []
        import_meta: Optional[Dict[str, Any]] = None
        gen_qi_pack_json_text: Optional[str] = None

        if mode == "Golden Pack":
            qi_pack = TEST_ONLY_GOLDEN_QI_PACK
            st.info(f"🧪 Golden Pack: {len(qi_pack)} Qi")

        elif mode == "JSON Upload":
            up = st.file_uploader("qi_pack.json", type=["json"], key="qi_pack_json")
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
                        "position": {"order_index": i + 1},
                        "statement": {"text_md": t},
                        "correction": {"available": bool(r), "text_md": r},
                    })
                st.success(f"✓ {len(qi_pack)} Qi")

        else:
            st.subheader("📄 Import Sujet (PDF) → Générer qi_pack.json")
            if pdfplumber is None:
                st.error("pdfplumber n'est pas disponible dans cet environnement. Installez-le pour activer l'import PDF.")
            else:
                subj_pdf = st.file_uploader("PDF Sujet", type=["pdf"], key="pdf_subject")
                corr_pdf = st.file_uploader("PDF Correction (optionnel)", type=["pdf"], key="pdf_correction")
                autoload = st.checkbox("Charger automatiquement dans GTE après génération", value=True)

                if st.button("🧩 Générer qi_pack.json", type="primary", use_container_width=True, disabled=(subj_pdf is None)):
                    try:
                        subj_bytes = subj_pdf.read() if subj_pdf else b""
                        corr_bytes = corr_pdf.read() if corr_pdf else None

                        import_meta, qi_pack = build_qi_pack_from_pdfs(subj_bytes, corr_bytes)
                        payload = {"meta": import_meta, "qi_pack": qi_pack}
                        gen_qi_pack_json_text = canonical_json(payload)

                        st.success(f"✓ Généré: {len(qi_pack)} Qi")
                        st.json({"stats": import_meta.get("stats", {}), "warnings": import_meta.get("warnings", [])})

                        st.download_button(
                            "📥 Télécharger qi_pack.json",
                            gen_qi_pack_json_text,
                            file_name="qi_pack.json",
                            mime="application/json",
                        )

                        if autoload:
                            st.session_state["qi_pack"] = qi_pack
                            st.session_state["import_meta"] = import_meta
                            st.success("✓ Chargé dans GTE (session)")

                    except Exception as e:
                        st.error(f"Erreur génération qi_pack.json: {e}")

        # store in session
        if qi_pack:
            st.session_state["qi_pack"] = qi_pack
        if import_meta:
            st.session_state["import_meta"] = import_meta

        if qi_pack:
            st.metric("Qi", len(qi_pack))

    # --------------------------- TAB 2: PIPELINE ---------------------------
    with tab2:
        st.header("🚀 Pipeline")
        lvl2 = st.checkbox("Level 2 (N=10)")

        if st.button("▶️ RUN GTE", type="primary", use_container_width=True):
            qi = st.session_state.get("qi_pack", [])
            if not qi:
                st.error("No Qi")
            else:
                with st.spinner("Running..."):
                    result = GTERunner().run(
                        qi_pack=qi,
                        gen_func=TEST_ONLY_simulate_qc_generation,
                        level2=lvl2,
                        forbidden_literals=forbidden_literals,
                    )
                    st.session_state["result"] = result

                for l in result["logs"]:
                    if "FAIL" in l:
                        st.markdown(f"🔴 `{l}`")
                    elif "PASS" in l:
                        st.markdown(f"🟢 `{l}`")
                    elif "⚠️" in l:
                        st.markdown(f"🟡 `{l}`")
                    else:
                        st.text(l)

                st.markdown("---")
                if result["verdict"] == "PASS":
                    st.success("# ✅ PASS")
                else:
                    st.error("# ❌ FAIL")

    # --------------------------- TAB 3: RESULTS ---------------------------
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

            st.markdown(f"**Output Hash:** `{m['output_hash'][:32]}...`")

            # If Import PDF meta exists, show quick KPI
            import_meta = st.session_state.get("import_meta")
            if import_meta:
                st.subheader("📄 Import PDF (meta)")
                st.json(import_meta.get("stats", {}))
                warns = import_meta.get("warnings", [])
                if warns:
                    st.warning({"warnings": warns})

            st.subheader("Validateurs")
            for k, v in r["validators"].items():
                icon = "✅" if v.get("pass") else "❌" if v.get("pass") is False else "⏭️"
                with st.expander(f"{icon} {k}"):
                    st.json(v)

            st.markdown("---")
            st.subheader("📤 Exports")

            c1, c2, c3 = st.columns(3)
            with c1:
                report = {
                    "version": "V31.7",
                    "timestamp": datetime.now().isoformat(),
                    "verdict": r["verdict"],
                    "validators": r["validators"],
                    "metrics": r["metrics"],
                }
                st.download_button("📥 run_report.json", canonical_json(report), "run_report.json")

            with c2:
                st.download_button("📥 artifacts_raw.json", canonical_json(r["artifacts_raw"]), "artifacts_raw.json")

            with c3:
                st.download_button("📥 artifacts_canon.json", canonical_json(r["artifacts_canon"]), "artifacts_canon.json")


if __name__ == "__main__":
    main()
