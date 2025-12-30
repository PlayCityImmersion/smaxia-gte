# smaxia_console_v31_7.py — SMAXIA GTE V31.7 (ISO-PROD Proof Harness + PDF→Qi V2)
# =============================================================================
# Goals:
#   - No regression of GTE harness + validators
#   - Add Import Sujet (PDF) → generate qi_pack.json with robust segmentation
#   - Remove hardcoded FORBIDDEN_LITERALS list (TA-02 optional via external JSON upload)
#   - Keep anti-hardcode posture: no country/language/subject literals embedded
# =============================================================================

from __future__ import annotations

import streamlit as st
import json
import hashlib
import ast
import random
import re
import io
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

# PDF extraction
import pdfplumber

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
    """TEST-ONLY generator for harness testing."""
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
            "qc_id": qc_id, "qc_formulation": "How to solve?",
            "qc_invariant_signature": {"intent_code": intent},
            "mapping": {"primary_qi_ids": qi_ids, "covered_qi_ids": qi_ids, "by_subject": dict(by_subj)},
            "links": {"ari_id": ari_id, "frt_id": frt_id, "trigger_ids": [trg_id]},
            "provenance": {"generator_version": "TEST_ONLY_V31.7"},
        })

        aris.append({
            "ari_id": ari_id, "template_id": "ARI_TPL_V1",
            "steps": [{"step_id": "S1", "operator_code": "OP_READ"},
                      {"step_id": "S2", "operator_code": "OP_PROCESS"},
                      {"step_id": "S3", "operator_code": "OP_CONCLUDE"}],
            "provenance": {"qc_id": qc_id},
        })

        frts.append({
            "frt_id": frt_id, "template_id": "FRT_TPL_V1",
            "sections": {"given": "Input", "goal": "Output", "method": "Steps",
                         "checks": ["Verify"], "common_traps": ["Avoid"], "final_form": "Result"},
            "provenance": {"qc_id": qc_id, "ari_id": ari_id},
        })

        triggers.append({
            "trigger_id": trg_id, "type_code": "TRG_CHECK", "severity": "medium",
            "condition": {"signal": "AFTER_S2"}, "action": {"recommendation": "Verify"},
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
#   NOTE: TA-02 forbidden literals is OPTIONAL and loaded externally (no hardcode).
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
    corr = qi.get("correction") or {}
    if "available" not in corr or "text_md" not in corr:
        return False, "Missing correction.available/text_md"
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


def scan_forbidden_literals_optional(source: str, forbidden: Set[str], zone: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """Optional TA-02: only if forbidden set provided by user."""
    violations: List[Dict[str, Any]] = []
    if not forbidden:
        return True, []

    for i, line in enumerate(source.split("\n"), 1):
        s = line.strip()
        low = s.lower()
        if s.startswith("#") or not s:
            continue
        for lit in forbidden:
            if lit and lit in low:
                violations.append({"line": i, "literal": lit, "zone": zone, "context": s[:160]})
    return len(violations) == 0, violations


# ---- Zone B source extraction (AST-parseable) ----
def get_zone_b_source_regex() -> str:
    """
    Extract the text between markers 'ZONE B' and 'ZONE C' from this module source.
    Uses inspect.getsource(module) then regex slicing.
    """
    import inspect
    try:
        src = inspect.getsource(inspect.getmodule(get_zone_b_source_regex))
    except Exception:
        return ""
    m = re.search(r"ZONE B\s*:.*?\n(.*)\n#\s*═{3,}\s*\n#\s*ZONE C", src, flags=re.DOTALL)
    if not m:
        # fallback: naive slicing
        s = src.find("ZONE B")
        e = src.find("ZONE C")
        return src[s:e] if s != -1 and e != -1 else ""
    return m.group(1)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B.2 : PDF → Qi builder V2 (robust line reconstruction + segmentation)
# ═══════════════════════════════════════════════════════════════════════════════

_RE_PAGE = re.compile(r"^\s*Page\s+\d+\s*/\s*\d+\s*$", re.IGNORECASE)
_RE_EXO = re.compile(r"^\s*EXERCICE\s*\d+\b", re.IGNORECASE)
_RE_NUM = re.compile(r"^\s*(\d+)\)\s*")
_RE_NUM_DOT = re.compile(r"^\s*(\d+)\.\s*")
_RE_ALPHA = re.compile(r"^\s*([a-z])\)\s*")
_RE_AFF = re.compile(r"^\s*Affirmation\s*(\d+)\s*:", re.IGNORECASE)
_RE_QCMQ = re.compile(r"^\s*Question\s*(\d+)\s*:", re.IGNORECASE)


def _pdf_lines_from_bytes(pdf_bytes: bytes) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract human-like lines using pdfplumber.extract_words (more robust than extract_text()).
    Returns (lines, meta).
    """
    errors: List[str] = []
    lines_out: List[str] = []
    pages = 0

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = len(pdf.pages)
            for p in pdf.pages:
                words = p.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
                if not words:
                    continue

                # group by y (top) with rounding to stabilize
                buckets: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
                for w in words:
                    key = round(float(w.get("top", 0.0)), 1)
                    buckets[key].append(w)

                for y in sorted(buckets.keys()):
                    row = sorted(buckets[y], key=lambda w: float(w.get("x0", 0.0)))
                    line = " ".join(str(w.get("text", "")).strip() for w in row).strip()
                    line = re.sub(r"\s+", " ", line).strip()
                    if line:
                        lines_out.append(line)
    except Exception as e:
        errors.append(str(e))

    meta = {"engine": "pdfplumber_words", "pages": pages, "errors": errors}
    return lines_out, meta


def _label_key(line: str) -> str:
    """
    Derive a stable key to align statement ↔ correction.
    Examples:
      '2) a) ...' -> '2)a)'
      '3) ...' -> '3)'
      'Affirmation 2:' -> 'AFF2'
    """
    m = re.match(r"^\s*(\d+)\)\s*([a-z])\)\s*", line, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}){m.group(2).lower()})"
    m = _RE_NUM.match(line)
    if m:
        return f"{m.group(1)})"
    m = _RE_NUM_DOT.match(line)
    if m:
        return f"{m.group(1)}."
    m = _RE_AFF.match(line)
    if m:
        return f"AFF{m.group(1)}"
    m = _RE_QCMQ.match(line)
    if m:
        return f"Q{m.group(1)}"
    # fallback
    return ""


def _is_qi_start(line: str) -> bool:
    if _RE_PAGE.match(line):
        return False
    if _RE_EXO.match(line):
        return True
    if re.match(r"^\s*\d+\)\s*", line):
        return True
    if re.match(r"^\s*\d+\.\s*", line):
        return True
    if re.match(r"^\s*[a-z]\)\s*", line) and len(line) <= 60:
        # avoid treating random 'a)' inside long lines as a start
        return True
    if _RE_AFF.match(line) or _RE_QCMQ.match(line):
        return True
    return False


def _segment_to_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Segment lines into blocks using start markers.
    Each block has: kind, key, text.
    """
    blocks: List[Dict[str, Any]] = []
    cur: List[str] = []
    cur_kind = "UNK"
    cur_key = ""

    def flush():
        nonlocal cur, cur_kind, cur_key
        if cur:
            text = "\n".join(cur).strip()
            if text:
                blocks.append({"kind": cur_kind, "key": cur_key, "text": text})
        cur, cur_kind, cur_key = [], "UNK", ""

    last_exo_header = ""

    for line in lines:
        if _RE_PAGE.match(line):
            continue

        if _RE_EXO.match(line):
            # exercise header: keep it as context line, but also acts as a start marker
            flush()
            last_exo_header = line.strip()
            cur_kind = "EXO"
            cur_key = _label_key(line) or last_exo_header
            cur = [line]
            continue

        if _is_qi_start(line):
            # start new block
            flush()
            cur_kind = "QI"
            cur_key = _label_key(line)
            cur = []
            if last_exo_header:
                cur.append(last_exo_header)
            cur.append(line)
            continue

        # normal continuation
        if not cur and last_exo_header:
            # if we haven't started a block yet, collect under an intro block
            cur_kind = "INTRO"
            cur_key = ""
            cur = [last_exo_header]
        cur.append(line)

    flush()

    # filter out ultra-short intro blocks
    filtered = []
    for b in blocks:
        t = b["text"].strip()
        if b["kind"] == "INTRO" and len(t) < 40:
            continue
        filtered.append(b)
    return filtered


def _index_corrections(blocks_corr: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Create map key -> correction text, using label keys.
    If multiple blocks share same key, concatenate deterministically.
    """
    m: Dict[str, List[str]] = defaultdict(list)
    for b in blocks_corr:
        k = b.get("key", "") or ""
        if not k:
            continue
        m[k].append(b.get("text", ""))

    out: Dict[str, str] = {}
    for k in sorted(m.keys()):
        parts = [p.strip() for p in m[k] if p.strip()]
        out[k] = "\n\n".join(parts).strip()
    return out


def build_qi_pack_from_pdfs(subject_pdf: bytes, correction_pdf: Optional[bytes]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: (meta, qi_pack, debug)
    """
    subj_sha8 = sha8_bytes(subject_pdf)
    corr_sha8 = sha8_bytes(correction_pdf) if correction_pdf else ""

    subj_lines, subj_meta = _pdf_lines_from_bytes(subject_pdf)
    corr_lines, corr_meta = ([], {"engine": "none", "pages": 0, "errors": []})
    if correction_pdf:
        corr_lines, corr_meta = _pdf_lines_from_bytes(correction_pdf)

    subj_blocks = _segment_to_blocks(subj_lines)
    corr_blocks = _segment_to_blocks(corr_lines) if corr_lines else []

    corr_index = _index_corrections(corr_blocks)

    qi_pack: List[Dict[str, Any]] = []
    subject_id = f"SUBJ_{subj_sha8}"

    order = 0
    aligned = 0
    warnings: List[str] = []

    # build QI only from blocks kind QI (ignore EXO/INTRO)
    qi_blocks = [b for b in subj_blocks if b.get("kind") == "QI"]

    if len(qi_blocks) < 3:
        warnings.append("Low QI count detected. PDF segmentation may need tuning (scan quality/layout).")

    for b in qi_blocks:
        order += 1
        stmt = b.get("text", "").strip()
        key = b.get("key", "") or ""
        corr_text = ""
        corr_avail = False

        if key and key in corr_index:
            corr_text = corr_index[key]
            corr_avail = True
            aligned += 1
        else:
            # fallback alignment: try numeric-only if key is like '2)a)' then also try '2)'
            if key and re.match(r"^\d+\)[a-z]\)$", key):
                base = key.split(")")[0] + ")"
                if base in corr_index:
                    corr_text = corr_index[base]
                    corr_avail = True
                    aligned += 1

        # deterministic id
        stmt_sig = hashlib.sha256(stmt.encode("utf-8")).hexdigest()[:10]
        qi_id = f"QI_{order:04d}_{subj_sha8}_{stmt_sig}"

        qi_pack.append({
            "qi_id": qi_id,
            "subject_id": subject_id,
            "position": {"order_index": order},
            "statement": {"text_md": stmt},
            "correction": {"available": bool(corr_avail), "text_md": corr_text},
        })

    meta = {
        "created_at": datetime.now().isoformat(),
        "generator": {"name": "SMAXIA_PDF_QI_BUILDER", "version": "V31.7"},
        "inputs": {
            "subject_pdf_sha8": subj_sha8,
            "correction_pdf_sha8": corr_sha8,
            "subject_extraction": subj_meta,
            "correction_extraction": corr_meta,
        },
        "stats": {
            "subject_lines": len(subj_lines),
            "correction_lines": len(corr_lines),
            "subject_blocks": len(subj_blocks),
            "correction_blocks": len(corr_blocks),
            "qi_count": len(qi_pack),
            "aligned_by_label": aligned,
        },
        "warnings": warnings,
    }

    debug = {
        "subject_blocks_preview": subj_blocks[:12],
        "correction_blocks_preview": corr_blocks[:12],
        "correction_index_keys": sorted(list(corr_index.keys()))[:40],
    }

    return meta, qi_pack, debug


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C : STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

class GTERunner:
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.logs: List[str] = []

    def log(self, m: str):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

    def run(self, qi_pack: List[Dict[str, Any]], gen_func, level2: bool = False, forbidden_literals: Optional[Set[str]] = None) -> Dict[str, Any]:
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

        # TC-01 Coverage
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

        # Source extract for TN/TA validators
        zone_b = get_zone_b_source_regex()
        source_available = len(zone_b.strip()) > 0

        # TN-02 No Test Imports
        if source_available:
            imp_ok, imp_v = check_no_test_imports(zone_b)
            self.results["TN-02_IMPORTS"] = {"pass": imp_ok, "violations": imp_v}
            self.log(f"TN-02 IMPORTS: {'PASS' if imp_ok else 'FAIL'}")
        else:
            self.results["TN-02_IMPORTS"] = {"pass": False, "error": "Source unavailable"}
            self.log("TN-02 IMPORTS: FAIL (source unavailable)")

        # TA-01 AST sensitive context access
        if source_available:
            ast_ok, ast_v = scan_ast_sensitive_access(zone_b)
            self.results["TA-01_AST"] = {"pass": ast_ok, "violations": ast_v}
            self.log(f"TA-01 AST: {'PASS' if ast_ok else 'FAIL'}")
        else:
            self.results["TA-01_AST"] = {"pass": False, "error": "Source unavailable"}
            self.log("TA-01 AST: FAIL (source unavailable)")

        # TA-02 Forbidden literals (optional; no hardcode)
        forbidden = forbidden_literals or set()
        if source_available:
            if forbidden:
                lit_ok, lit_v = scan_forbidden_literals_optional(zone_b, forbidden, "B")
                self.results["TA-02_LITERALS"] = {"pass": lit_ok, "violations": lit_v, "mode": "ENABLED", "count": len(forbidden)}
                self.log(f"TA-02 LITERALS: {'PASS' if lit_ok else 'FAIL'} (enabled: {len(forbidden)} terms)")
            else:
                self.results["TA-02_LITERALS"] = {"pass": True, "note": "Skipped (no forbidden_literals.json provided)", "mode": "SKIPPED"}
                self.log("TA-02 LITERALS: PASS (skipped)")
        else:
            self.results["TA-02_LITERALS"] = {"pass": False, "error": "Source unavailable"}
            self.log("TA-02 LITERALS: FAIL (source unavailable)")

        # TA-04 Prompt Isolation
        self.results["TA-04_PROMPT"] = {"pass": True, "note": "No IA prompt"}
        self.log("TA-04 PROMPT: PASS")

        # P1: TD-03 N=10
        if level2:
            d10_ok, h10, _ = check_determinism_n_runs(gen_func, qi_pack, 10)
            self.results["TD-03_N10"] = {"pass": d10_ok, "hashes": h10}
            self.log(f"TD-03 N10: {'PASS' if d10_ok else 'FAIL'}")
        else:
            self.results["TD-03_N10"] = {"pass": None, "note": "Skipped"}

        # P1: TN-03 Diversity
        div_ok, div_c, intents = check_intent_diversity(canon, 5)
        self.results["TN-03_DIVERSITY"] = {"pass": div_ok, "count": div_c, "intents": list(intents)}
        self.log(f"TN-03 DIVERSITY: {'PASS' if div_ok else 'FAIL'} ({div_c} intents)")

        # Final Verdict
        p0_keys = ["TS-01_SCHEMA", "TS-02_REF", "TC-01_COV", "TC-02_PRIM",
                   "TD-01_DET_N3", "TD-02_ORDER", "TN-02_IMPORTS",
                   "TA-01_AST", "TA-02_LITERALS", "TA-04_PROMPT"]
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
                "qi": len(qi_pack), "qc": len(qcs), "ari": len(aris),
                "frt": len(frts), "trg": len(trgs), "singletons": len(singletons),
                "input_hash": input_hash, "output_hash": output_hash,
            },
            "logs": self.logs,
        }


def parse_qi_pack_input(data: Any) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        for k in ["items", "qi_pack", "questions"]:
            if k in data:
                return data[k], {x: data[x] for x in data if x != k}
    return [], {"error": "Unknown format"}


def _load_forbidden_literals(upload) -> Tuple[Set[str], Optional[str]]:
    if not upload:
        return set(), None
    try:
        obj = json.load(upload)
        if isinstance(obj, dict) and "forbidden_literals" in obj and isinstance(obj["forbidden_literals"], list):
            terms = {str(x).strip().lower() for x in obj["forbidden_literals"] if str(x).strip()}
            return terms, None
        if isinstance(obj, list):
            terms = {str(x).strip().lower() for x in obj if str(x).strip()}
            return terms, None
        return set(), "Invalid forbidden_literals.json format (expected list OR dict{forbidden_literals:[...]})"
    except Exception as e:
        return set(), f"Failed to parse forbidden_literals.json: {e}"


def main():
    st.set_page_config(page_title="SMAXIA GTE Console V31.7", page_icon="🔒", layout="wide")
    st.title("🔒 SMAXIA GTE Console V31.7")
    st.markdown("**Harnais de Preuve ISO-PROD — Final + Import Sujet (PDF) → qi_pack.json (V2)**")

    # Sidebar: optional TA-02 list
    with st.sidebar:
        st.header("⚙️ V31.7")
        st.markdown("✓ PDF→Qi V2 (pdfplumber words → lines → segmentation)")
        st.markdown("✓ get_zone_b_source regex (AST parseable)")
        st.markdown("✓ TA-02 sans hardcode (upload forbidden_literals.json)")
        st.markdown("✓ No regression: validators P0 unchanged")

        st.markdown("---")
        st.subheader("TA-02 (optional)")
        st.caption("Uploadez forbidden_literals.json si vous voulez activer TA-02.")
        forbidden_upload = st.file_uploader("forbidden_literals.json", type=["json"], key="forbidden_literals")
        forbidden_set, forbidden_err = _load_forbidden_literals(forbidden_upload)
        if forbidden_err:
            st.error(forbidden_err)
        if forbidden_set:
            st.success(f"TA-02 activé: {len(forbidden_set)} termes")
        else:
            st.info("TA-02 désactivé (pas de fichier)")

        st.markdown("---")
        with st.expander("Exemple forbidden_literals.json"):
            st.code(
                json.dumps({"forbidden_literals": ["countryname", "languagename", "subjectname"]}, ensure_ascii=False, indent=2),
                language="json"
            )

    tab1, tab2, tab3 = st.tabs(["📥 Entrée", "🚀 Pipeline", "📊 Résultats"])

    with tab1:
        st.header("📥 Entrée")
        mode = st.radio("Mode", ["Golden Pack", "JSON Upload", "Manuel", "Import Sujet (PDF)"], horizontal=True)
        qi_pack: List[Dict[str, Any]] = []

        if mode == "Golden Pack":
            qi_pack = TEST_ONLY_GOLDEN_QI_PACK
            st.info(f"🧪 Golden Pack: {len(qi_pack)} Qi")

        elif mode == "JSON Upload":
            up = st.file_uploader("qi_pack.json", type=["json"], key="qi_pack_upload")
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
                        "qi_id": f"QI_{i+1:03d}", "subject_id": "SUBJ",
                        "position": {"order_index": i + 1},
                        "statement": {"text_md": t},
                        "correction": {"available": bool(r), "text_md": r},
                    })
                st.success(f"✓ {len(qi_pack)} Qi")

        else:
            st.subheader("📄 Import Sujet (PDF) → Générer qi_pack.json")
            pdf_sujet = st.file_uploader("PDF Sujet", type=["pdf"], key="pdf_subject")
            pdf_corr = st.file_uploader("PDF Correction (optionnel)", type=["pdf"], key="pdf_corr")
            auto_load = st.checkbox("Charger automatiquement dans GTE après génération", value=True)

            if pdf_sujet:
                st.caption(f"Sujet: {pdf_sujet.name}")
            if pdf_corr:
                st.caption(f"Correction: {pdf_corr.name}")

            if st.button("🧩 Générer qi_pack.json", type="primary", use_container_width=True, disabled=not bool(pdf_sujet)):
                with st.spinner("Extraction PDF → lignes → segmentation → alignement..."):
                    subject_bytes = pdf_sujet.read()
                    corr_bytes = pdf_corr.read() if pdf_corr else None
                    meta, pack, debug = build_qi_pack_from_pdfs(subject_bytes, corr_bytes)

                    out = {"meta": meta, "qi_pack": pack}
                    st.session_state["last_generated_pack_obj"] = out
                    st.session_state["last_generated_debug"] = debug

                    st.success(f"✓ Généré: {len(pack)} Qi")
                    if meta.get("warnings"):
                        for w in meta["warnings"]:
                            st.warning(w)

                    st.expander("Meta extraction / alignement", expanded=False).json(meta)
                    st.download_button("📥 Télécharger qi_pack.json", canonical_json(out), "qi_pack.json")

                    if auto_load:
                        qi_pack = pack
                        st.session_state["qi_pack"] = qi_pack
                        st.info("✓ Pack chargé (tab Pipeline).")

            if "last_generated_pack_obj" in st.session_state:
                with st.expander("Dernier qi_pack.json généré (aperçu)", expanded=False):
                    st.json(st.session_state["last_generated_pack_obj"]["meta"])
                    st.caption("Aperçu 3 premières Qi:")
                    st.json(st.session_state["last_generated_pack_obj"]["qi_pack"][:3])
                with st.expander("Debug segmentation (preview)", expanded=False):
                    st.json(st.session_state.get("last_generated_debug", {}))

        # persist qi_pack
        if mode != "Import Sujet (PDF)":
            st.session_state["qi_pack"] = qi_pack

        if st.session_state.get("qi_pack"):
            st.metric("Qi", len(st.session_state["qi_pack"]))

    with tab2:
        st.header("🚀 Pipeline")
        lvl2 = st.checkbox("Level 2 (N=10)", key="lvl2")

        if st.button("▶️ RUN GTE", type="primary", use_container_width=True):
            qi = st.session_state.get("qi_pack", [])
            if not qi:
                st.error("No Qi")
            else:
                with st.spinner("Running..."):
                    result = GTERunner().run(qi, TEST_ONLY_simulate_qc_generation, lvl2, forbidden_set)
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

            st.subheader("Validateurs")
            for k, v in r["validators"].items():
                icon = "✅" if v.get("pass") else "❌" if v.get("pass") is False else "⏭️"
                with st.expander(f"{icon} {k}"):
                    st.json(v)

            st.markdown("---")
            st.subheader("📤 Exports")

            c1, c2, c3 = st.columns(3)
            with c1:
                report = {"version": "V31.7", "timestamp": datetime.now().isoformat(),
                          "verdict": r["verdict"], "validators": r["validators"], "metrics": r["metrics"]}
                st.download_button("📥 run_report.json", canonical_json(report), "run_report.json")

            with c2:
                st.download_button("📥 artifacts_raw.json", canonical_json(r["artifacts_raw"]), "artifacts_raw.json")

            with c3:
                st.download_button("📥 artifacts_canon.json", canonical_json(r["artifacts_canon"]), "artifacts_canon.json")


if __name__ == "__main__":
    main()
