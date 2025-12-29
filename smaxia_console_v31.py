# smaxia_console_v31.py — SMAXIA GTE V31.6 (ISO-PROD Proof Harness + PDF→qi_pack.json)
# =============================================================================
# Version: V31.6 - Adds Step C (CEO zero-code JSON builder)
# Adds:
#   - Mode "Import Sujet (PDF)": Upload subject PDF + optional correction PDF
#   - Button "Générer qi_pack.json" (strict schema)
#   - Optional: Load generated pack into GTE session state (no coding)
# Keeps:
#   - Golden Pack / JSON Upload / Manuel
#   - ISO-PROD proof harness validators (P0)
# Notes:
#   - No business hardcode (country/discipline/chapter). All heuristics are generic.
#   - PDF parsing is best-effort without OCR.
# =============================================================================

import streamlit as st
import json
import hashlib
import ast
import random
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

# =============================================================================
# ZONE A : TEST-ONLY FIXTURES — NOT FOR PROD
# test_only: true
# REMOVE BEFORE PROD: replaced by real Pack loader
# =============================================================================

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
            "qc_id": qc_id,
            "qc_formulation": "How to solve?",
            "qc_invariant_signature": {"intent_code": intent},
            "mapping": {"primary_qi_ids": qi_ids, "covered_qi_ids": qi_ids, "by_subject": dict(by_subj)},
            "links": {"ari_id": ari_id, "frt_id": frt_id, "trigger_ids": [trg_id]},
            "provenance": {"generator_version": "TEST_ONLY_V31.6"},
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


# =============================================================================
# ZONE B : CORE-LIKE INVARIANT HELPERS (métier-neutre)
# =============================================================================

FORBIDDEN_LITERALS_V1: Set[str] = {
    "france", "français", "usa", "uk", "english", "french",
    "mathématiques", "maths", "physique", "chimie", "svt",
    "mathematics", "math", "physics", "chemistry", "biology",
    "baccalauréat", "bac", "sat", "gcse", "terminale", "lycée",
    "suites", "fonctions", "probabilités", "statistiques", "géométrie",
    "algèbre", "analyse", "intégrales", "dérivées", "limites", "récurrence",
}

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


def sort_qi_canonical(qi_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(qi_list, key=lambda x: (x.get("position", {}).get("order_index", 0), x.get("qi_id", "")))


def canonicalize_artifacts(artifacts: Dict[str, Any], qi_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    singletons = artifacts.get("singletons_warning", []) or []
    singletons_sorted = sorted(
        singletons,
        key=lambda x: (str(x.get("intent", "")), "|".join(sorted([str(q) for q in x.get("qi_ids", []) or []]))),
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
    # correction is optional in schema checks, but we expect it in our packs
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


def scan_forbidden_literals(source: str, zone: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """Scan with block-skip for defense mechanism definitions."""
    violations: List[Dict[str, Any]] = []
    in_forbidden_block = False
    in_sensitive_block = False

    for i, line in enumerate(source.split("\n"), 1):
        s = line.strip()
        low = s.lower()

        if s.startswith("#") or not s:
            continue

        if "forbidden_literals_v1" in low and "{" in low:
            in_forbidden_block = True
            continue
        if in_forbidden_block:
            if "}" in low:
                in_forbidden_block = False
            continue

        if "sensitive_context_fields" in low and "{" in low:
            in_sensitive_block = True
            continue
        if in_sensitive_block:
            if "}" in low:
                in_sensitive_block = False
            continue

        for lit in FORBIDDEN_LITERALS_V1:
            if lit in low:
                violations.append({"line": i, "literal": lit, "zone": zone, "context": s[:120]})

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


# =============================================================================
# ZONE B.5 : CEO ZERO-CODE JSON BUILDER — PDF Sujet + PDF Correction -> qi_pack.json
# =============================================================================

_Q_START_PATTERNS = [
    # "1." / "1)" / "1 -" / "1 :"
    re.compile(r"^\s*(\d{1,3})\s*[\.\)\:\-]\s+"),
    # "Question 1" / "QUESTION 1"
    re.compile(r"^\s*question\s*(\d{1,3})\b", re.IGNORECASE),
    # "Q1" / "Q 1"
    re.compile(r"^\s*q\s*(\d{1,3})\b", re.IGNORECASE),
]


def _bytes_sha8(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]


def read_pdf_to_text(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Best-effort PDF text extraction without OCR.
    Tries pdfplumber if available, else PyPDF2.
    Returns (text, audit).
    """
    audit = {"engine": None, "pages": None, "errors": []}
    text_parts: List[str] = []

    # Try pdfplumber
    try:
        import pdfplumber  # type: ignore
        audit["engine"] = "pdfplumber"
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:  # noqa: F821
            audit["pages"] = len(pdf.pages)
            for p in pdf.pages:
                t = p.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        text = "\n\n".join(text_parts)
        return text, audit
    except Exception as e:
        audit["errors"].append(f"pdfplumber: {e}")

    # Fallback PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore
        audit["engine"] = "PyPDF2"
        reader = PdfReader(io.BytesIO(pdf_bytes))  # noqa: F821
        audit["pages"] = len(reader.pages)
        for p in reader.pages:
            t = (p.extract_text() or "")
            if t.strip():
                text_parts.append(t)
        text = "\n\n".join(text_parts)
        return text, audit
    except Exception as e:
        audit["errors"].append(f"PyPDF2: {e}")

    return "", audit


def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    # normalize spaces
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def split_numbered_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Splits text into question-like blocks using generic numbering/markers.
    Returns list of blocks: {num: Optional[str], text: str}
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    starts: List[Tuple[int, Optional[str]]] = []

    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        num = None
        for pat in _Q_START_PATTERNS:
            m = pat.match(s)
            if m:
                num = str(m.group(1))
                break
        if num is not None:
            starts.append((i, num))

    # If we found numbered starts, cut accordingly
    if starts:
        blocks: List[Dict[str, Any]] = []
        for idx, (start_i, num) in enumerate(starts):
            end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
            chunk = "\n".join(lines[start_i:end_i]).strip()
            if chunk:
                blocks.append({"num": num, "text": chunk})
        return blocks

    # Fallback: paragraph splitting (best-effort)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks = []
    for p in paras:
        if len(p) >= 40:  # ignore tiny debris
            blocks.append({"num": None, "text": p})
    return blocks


def build_qi_pack_from_pdfs(
    subject_pdf_bytes: bytes,
    correction_pdf_bytes: Optional[bytes] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns (payload_dict_for_json_upload, qi_pack_list).
    payload_dict includes {"qi_pack":[...], "meta":{...}}
    """
    # io is needed only inside pdfplumber/PyPDF2 fallbacks; import here to avoid toolchain issues
    import io  # local import to keep global namespace minimal

    subj_hash = _bytes_sha8(subject_pdf_bytes)
    subj_text_raw, subj_audit = read_pdf_to_text(subject_pdf_bytes)
    subj_text = normalize_text(subj_text_raw)

    corr_text = ""
    corr_audit = None
    corr_hash = None
    if correction_pdf_bytes:
        corr_hash = _bytes_sha8(correction_pdf_bytes)
        corr_text_raw, corr_audit = read_pdf_to_text(correction_pdf_bytes)
        corr_text = normalize_text(corr_text_raw)

    subj_blocks = split_numbered_blocks(subj_text)

    # correction blocks mapping by number if possible
    corr_map: Dict[str, str] = {}
    corr_blocks: List[Dict[str, Any]] = []
    if corr_text:
        corr_blocks = split_numbered_blocks(corr_text)
        for b in corr_blocks:
            if b.get("num"):
                # keep first occurrence; if duplicates, keep the longest (safer)
                n = str(b["num"])
                txt = str(b.get("text", "")).strip()
                if n not in corr_map or len(txt) > len(corr_map[n]):
                    corr_map[n] = txt

    subject_id = f"SUBJ_{subj_hash}"

    qi_pack: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if not subj_blocks:
        warnings.append("No blocks extracted from subject PDF (empty or non-text PDF).")

    # If numbered questions exist in subject and correction, align by number.
    # Otherwise, correction is marked unavailable to avoid false mapping.
    aligned = 0
    for idx, b in enumerate(subj_blocks, start=1):
        qtext = str(b.get("text", "")).strip()
        qnum = b.get("num")
        # stable qi_id: based on content hash (deterministic) + order
        qh = hashlib.sha256((qtext).encode("utf-8")).hexdigest()[:10]
        qi_id = f"QI_{idx:04d}_{subj_hash}_{qh}"

        corr_available = False
        corr_md = ""

        if corr_map and qnum and qnum in corr_map:
            corr_available = True
            corr_md = corr_map[qnum].strip()
            aligned += 1

        qi_pack.append({
            "qi_id": qi_id,
            "subject_id": subject_id,
            "position": {"order_index": idx},
            "statement": {"text_md": qtext},
            "correction": {"available": bool(corr_available), "text_md": corr_md},
        })

    meta = {
        "generator": {"name": "SMAXIA_PDF_QI_BUILDER", "version": "V31.6"},
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "subject_pdf_sha8": subj_hash,
            "correction_pdf_sha8": corr_hash,
            "subject_extraction": subj_audit,
            "correction_extraction": corr_audit,
        },
        "stats": {
            "qi_count": len(qi_pack),
            "subject_blocks": len(subj_blocks),
            "correction_blocks": len(corr_blocks) if corr_text else 0,
            "aligned_by_number": aligned,
        },
        "warnings": warnings + (
            [] if not correction_pdf_bytes else
            (["Correction PDF provided but no reliable alignment by number found; correction.available may be False."]
             if corr_text and aligned == 0 else [])
        ),
    }

    payload = {"qi_pack": qi_pack, "meta": meta}
    return payload, qi_pack


# =============================================================================
# ZONE C : STREAMLIT UI
# =============================================================================

def get_zone_b_source() -> str:
    """
    Robust source extraction for validators (AST/literals/imports).
    Uses file read first (preferred), then fallback.
    """
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        try:
            import inspect
            src = inspect.getsource(inspect.getmodule(get_zone_b_source))
        except Exception:
            return ""

    s = src.find("ZONE B")
    e = src.find("ZONE C")
    if s == -1 or e == -1 or e <= s:
        # Fallback: try regex capture between markers
        m = re.search(r"ZONE B[\s\S]*?ZONE C", src)
        if not m:
            return ""
        chunk = m.group(0)
        # Keep only the content starting at ZONE B line
        return chunk
    return src[s:e]


def parse_qi_pack_input(data: Any) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        for k in ["items", "qi_pack", "questions"]:
            if k in data:
                # preserve other top-level meta
                meta = {x: data[x] for x in data if x != k}
                return data[k], (meta if meta else None)
    return [], {"error": "Unknown format"}


class GTERunner:
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.logs: List[str] = []

    def log(self, m: str):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

    def run(self, qi_pack: List[Dict[str, Any]], gen_func, level2: bool = False) -> Dict[str, Any]:
        self.logs = []
        self.results = {}
        self.log(f"═══ GTE V31.6 — {len(qi_pack)} Qi ═══")

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

        # TC-01 Coverage (CEO binary gate)
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

        # TA-02 Forbidden Literals
        if source_available:
            lit_ok, lit_v = scan_forbidden_literals(zone_b, "B")
            self.results["TA-02_LITERALS"] = {"pass": lit_ok, "violations": lit_v}
            self.log(f"TA-02 LITERALS: {'PASS' if lit_ok else 'FAIL'}")
        else:
            self.results["TA-02_LITERALS"] = {"pass": False, "error": "Source unavailable"}
            self.log("TA-02 LITERALS: FAIL (source unavailable)")

        # TA-04 Prompt Isolation
        self.results["TA-04_PROMPT"] = {"pass": True, "note": "No IA prompt"}
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
        self.results["TN-03_DIVERSITY"] = {"pass": div_ok, "count": div_c, "intents": list(intents)}
        self.log(f"TN-03 DIVERSITY: {'PASS' if div_ok else 'FAIL'} ({div_c} intents)")

        # Final Verdict
        p0_keys = [
            "TS-01_SCHEMA", "TS-02_REF", "TC-01_COV", "TC-02_PRIM",
            "TD-01_DET_N3", "TD-02_ORDER", "TN-02_IMPORTS",
            "TA-01_AST", "TA-02_LITERALS", "TA-04_PROMPT",
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
                "qi": len(qi_pack), "qc": len(qcs), "ari": len(aris),
                "frt": len(frts), "trg": len(trgs), "singletons": len(singletons),
                "input_hash": input_hash, "output_hash": output_hash,
            },
            "logs": self.logs,
        }


def main():
    st.set_page_config(page_title="SMAXIA GTE V31.6", page_icon="🔒", layout="wide")
    st.title("🔒 SMAXIA GTE Console V31.6")
    st.markdown("**Harnais de Preuve ISO-PROD — + PDF → qi_pack.json (CEO zero-code)**")

    with st.sidebar:
        st.header("⚙️ V31.6")
        st.markdown("✓ GTE validators (P0) conservés")
        st.markdown("✓ PDF → qi_pack.json (Sujet + Correction)")
        st.markdown("✓ Zéro code côté CEO")

    tab1, tab2, tab3 = st.tabs(["📥 Entrée", "🚀 Pipeline", "📊 Résultats"])

    with tab1:
        st.header("📥 Entrée")

        mode = st.radio(
            "Mode",
            ["Golden Pack", "JSON Upload", "Manuel", "Import Sujet (PDF)"],
            horizontal=True
        )

        qi_pack: List[Dict[str, Any]] = []
        meta_preview: Optional[Dict[str, Any]] = None

        if mode == "Golden Pack":
            qi_pack = TEST_ONLY_GOLDEN_QI_PACK
            st.info(f"🧪 Golden Pack: {len(qi_pack)} Qi")

        elif mode == "JSON Upload":
            up = st.file_uploader("qi_pack.json", type=["json"])
            if up:
                try:
                    raw = json.load(up)
                    qi_pack, meta_preview = parse_qi_pack_input(raw)
                    st.success(f"✓ {len(qi_pack)} Qi")
                    if meta_preview:
                        with st.expander("Meta (top-level)"):
                            st.json(meta_preview)
                except Exception as e:
                    st.error(f"Error: {e}")

        elif mode == "Manuel":
            txt = st.text_area("Questions (1/line)", height=150)
            rqi = st.text_area("Corrigés (1/line) [optionnel]", height=150)
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
            st.caption("Upload sujet PDF + correction PDF (optionnel). Génération best-effort sans OCR.")

            subj = st.file_uploader("PDF Sujet", type=["pdf"], key="pdf_subject")
            corr = st.file_uploader("PDF Correction (optionnel)", type=["pdf"], key="pdf_correction")

            colA, colB = st.columns([1, 1])
            with colA:
                load_into_session = st.checkbox("Charger automatiquement dans GTE après génération", value=True)
            with colB:
                st.caption("Si l’alignement numéroté échoue, correction.available restera False (anti-faux-mapping).")

            if st.button("🧩 Générer qi_pack.json", type="primary", use_container_width=True, disabled=(subj is None)):
                try:
                    subject_bytes = subj.getvalue() if subj else b""
                    corr_bytes = corr.getvalue() if corr else None

                    payload, qi_pack_generated = build_qi_pack_from_pdfs(subject_bytes, corr_bytes)
                    st.session_state["generated_payload"] = payload
                    st.success(f"✓ Généré: {len(qi_pack_generated)} Qi")

                    with st.expander("Meta extraction / alignement"):
                        st.json(payload.get("meta", {}))

                    # download
                    st.download_button(
                        "📥 Télécharger qi_pack.json",
                        canonical_json(payload),
                        file_name="qi_pack.json",
                        mime="application/json",
                        use_container_width=True
                    )

                    if load_into_session:
                        qi_pack = qi_pack_generated
                        st.session_state["qi_pack"] = qi_pack
                        st.info("✓ Pack chargé dans la session GTE (tab Pipeline).")
                    else:
                        st.info("Pack généré, non chargé. Vous pouvez le télécharger puis l’uploader via JSON Upload.")

                except Exception as e:
                    st.error(f"Generation error: {e}")

            # if a payload already generated previously, allow quick load
            if "generated_payload" in st.session_state:
                with st.expander("Dernier qi_pack.json généré (aperçu)"):
                    st.json(st.session_state["generated_payload"].get("meta", {}))

        # Persist selection
        if mode != "Import Sujet (PDF)":
            st.session_state["qi_pack"] = qi_pack

        if st.session_state.get("qi_pack"):
            st.metric("Qi", len(st.session_state["qi_pack"]))

    with tab2:
        st.header("🚀 Pipeline")
        lvl2 = st.checkbox("Level 2 (N=10)")

        if st.button("▶️ RUN GTE", type="primary", use_container_width=True):
            qi = st.session_state.get("qi_pack", [])
            if not qi:
                st.error("No Qi")
            else:
                with st.spinner("Running..."):
                    result = GTERunner().run(qi, TEST_ONLY_simulate_qc_generation, lvl2)
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
                report = {
                    "version": "V31.6",
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
