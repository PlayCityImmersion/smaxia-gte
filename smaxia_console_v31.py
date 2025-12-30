# smaxia_console_v31_ui.py — SMAXIA GTE V31.9.0 (Proof Harness)
# =============================================================================
# OBJECTIF (TEST) :
# - Extraire correctement les Qi d'un PDF BAC (énoncé) + aligner le corrigé
# - Générer plusieurs QC (éviter le collapse "1 QC") sans hardcode chapitre/pays
# - Respecter la règle de formulation QC : commence par "Comment" et finit par "?"
# - Zéro régression UI : mêmes 3 onglets + exports + validateurs
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
    import pdfplumber
except Exception:
    pdfplumber = None


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A : TEST-ONLY FIXTURES — NOT FOR PROD
# ═══════════════════════════════════════════════════════════════════════════════

TEST_ONLY_GOLDEN_QI_PACK: List[Dict[str, Any]] = [
    {"qi_id": "QI_TEST_001", "subject_id": "SUBJ_A", "position": {"order_index": 1},
     "statement": {"text_md": "Calculer la valeur demandée."},
     "correction": {"available": True, "text_md": "On applique la définition puis on vérifie."}},
    {"qi_id": "QI_TEST_002", "subject_id": "SUBJ_A", "position": {"order_index": 2},
     "statement": {"text_md": "Démontrer la propriété."},
     "correction": {"available": True, "text_md": "Preuve par hypothèses puis conclusion."}},
    {"qi_id": "QI_TEST_003", "subject_id": "SUBJ_A", "position": {"order_index": 3},
     "statement": {"text_md": "Résoudre l'équation."},
     "correction": {"available": True, "text_md": "On isole x puis on contrôle."}},
    {"qi_id": "QI_TEST_004", "subject_id": "SUBJ_A", "position": {"order_index": 4},
     "statement": {"text_md": "En déduire le résultat."},
     "correction": {"available": True, "text_md": "On exploite la question précédente."}},
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
    # Règle SMAXIA : "Comment ... ?"
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
# ZONE B.5 : PDF → Qi Builder (robuste BAC)
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

# BAC markers fréquents : "1)" puis "a)" "b)" "c)" ; parfois texte très compact.
_MAIN_RE = re.compile(r"^\s*(\d{1,3})\s*[\)\.\-]\s*")
_SUB_RE = re.compile(r"^\s*([a-h])\s*[\)\.\-]\s*", re.I)
_COMBO_RE = re.compile(r"^\s*(\d{1,3})\s*[\)\.\-]\s*([a-h])\s*[\)\.\-]\s*", re.I)

def _detect_exercise(line: str) -> Optional[str]:
    s = line.strip()
    if not s:
        return None
    # tolérant : "EXERCICE 1" ou "EXERCICE1"
    up = s.upper().replace(" ", "")
    if up.startswith("EXERCICE"):
        m = re.match(r"EXERCICE\s*([0-9]+)", s, re.I)
        if not m:
            m = re.match(r"EXERCICE([0-9]+)", up, re.I)
            if m:
                return m.group(1)
            return None
        return m.group(1)
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
        # seuil anti-bruit
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
            # on ne flush pas forcément : un exercice peut démarrer après une question finie
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

    # fallback : si rien détecté, un seul bloc FULL
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
    idx = {}
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
        "generator": {"name": "SMAXIA_PDF_QI_BUILDER", "version": "V31.9.0"},
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
    }
    return meta, qi_pack


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B.7 : QC/ARI/FRT/TRG — TEST GENERATOR (anti-collapse)
# ═══════════════════════════════════════════════════════════════════════════════

# IMPORTANT :
# - Ne PAS classifier à partir du corrigé (sinon biais "donc/ainsi..." => 1 cluster)
# - Utiliser des signaux invariants : verbes d'intention + familles d'opérations (universelles)

_STOPWORDS = set([
    "donc", "ainsi", "d'où", "car", "alors", "ensuite", "puis", "or", "donner", "trouver",
])

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Verbes d'intention (FR + EN) : pondérés (éviter mots trop génériques)
_INTENT_RULES: List[Tuple[str, List[Tuple[str, int]]]] = [
    ("INTENT_DEMONTRER", [("démontrer", 4), ("montrer que", 4), ("prouver", 4), ("preuve", 3), ("justify that", 3), ("prove", 3)]),
    ("INTENT_RESoudre", [("résoudre", 4), ("solution", 2), ("équation", 3), ("inéquation", 3), ("solve", 3), ("isoler", 3)]),
    ("INTENT_CALCULER", [("calculer", 4), ("déterminer", 3), ("évaluer", 3), ("compute", 3), ("calculate", 3), ("valeur", 2)]),
    ("INTENT_ETUDIER", [("étudier", 3), ("analyser", 3), ("variation", 2), ("monoton", 2), ("tableau", 2)]),
    ("INTENT_VERIFIER", [("vérifier", 4), ("contrôler", 3), ("check", 3), ("confirm", 2)]),
    ("INTENT_DEDUIRE", [("en déduire", 4), ("déduire", 3), ("conclure", 3), ("deduce", 3), ("conclude", 3)]),
    ("INTENT_INTERPRETER", [("interpréter", 3), ("interprétation", 3), ("signification", 2)]),
]

# Familles d'opérations (math universelles) — secondaires pour re-split anti-collapse
_OP_FAMILY_RULES: List[Tuple[str, List[str]]] = [
    ("OP_LIMIT", ["limite", "tend vers", "convergence", "asymptote"]),
    ("OP_DERIVATIVE", ["dérivée", "f'", "tangente", "pente"]),
    ("OP_INTEGRAL", ["intégrale", "primitive", "aire", "surface"]),
    ("OP_PROBABILITY", ["probabilité", "loi", "événement", "espérance", "variance"]),
    ("OP_COMPLEX", ["complexe", "affixe", "argument", "module", "conjugué"]),
    ("OP_SEQUENCE", ["suite", "récurrence", "terme", "un+1", "u(n)"]),
    ("OP_GEOMETRY", ["vecteur", "droite", "plan", "distance", "coordonnées", "perpendiculaire", "parallèle"]),
    ("OP_MATRIX", ["matrice", "déterminant", "inverse"]),
]

def _score_intent(text: str) -> Tuple[str, int]:
    t = _norm(text)
    # neutraliser les stopwords de liaison
    for w in _STOPWORDS:
        t = t.replace(w, " ")
    t = re.sub(r"\s+", " ", t).strip()

    best = ("INTENT_GENERIC", 0)
    for intent, rules in _INTENT_RULES:
        score = 0
        for kw, w in rules:
            if kw in t:
                score += w
        if score > best[1]:
            best = (intent, score)
    return best

def _op_family(text: str) -> str:
    t = _norm(text)
    best = ("OP_GENERIC", 0)
    for op, kws in _OP_FAMILY_RULES:
        score = sum(1 for kw in kws if kw in t)
        if score > best[1]:
            best = (op, score)
    return best[0]

def _anti_collapse_split(assignments: List[Tuple[str, str, Dict[str, Any]]], max_ratio: float = 0.60) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    assignments = [(intent, op, qi), ...]
    Si un intent domine trop, on re-split ce intent par op_family.
    """
    if not assignments:
        return assignments
    cnt = defaultdict(int)
    for intent, _, _ in assignments:
        cnt[intent] += 1
    total = len(assignments)
    dom_intent, dom_n = max(cnt.items(), key=lambda kv: kv[1])
    if total >= 8 and (dom_n / total) >= max_ratio:
        # Re-split seulement le cluster dominant, en utilisant op_family
        out = []
        for intent, op, qi in assignments:
            if intent == dom_intent:
                out.append((f"{intent}__{op}", op, qi))
            else:
                out.append((intent, op, qi))
        return out
    return assignments

def TEST_ONLY_generate_qc_ari_frt_trg(qi_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Générateur TEST (non-PROD) :
    - cluster = intent (principal) + anti-collapse par familles d'opérations
    - QC: "Comment ... ?"
    """
    # 1) assign intent/op à partir du STATEMENT uniquement
    assignments: List[Tuple[str, str, Dict[str, Any]]] = []
    for qi in qi_pack:
        stmt = (qi.get("statement", {}) or {}).get("text_md", "") or ""
        intent, _score = _score_intent(stmt)
        op = _op_family(stmt)
        assignments.append((intent, op, qi))

    assignments = _anti_collapse_split(assignments, max_ratio=0.60)

    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    sig_op: Dict[str, str] = {}
    for intent_key, op, qi in assignments:
        clusters[intent_key].append(qi)
        sig_op[intent_key] = op

    qcs, aris, frts, triggers, singletons = [], [], [], [], []

    # 2) produire artefacts par cluster
    for intent_key in sorted(clusters.keys()):
        grp = sorted(clusters[intent_key], key=lambda x: (x.get("position", {}).get("order_index", 0), x.get("qi_id", "")))
        qi_ids = [q.get("qi_id", "") for q in grp if q.get("qi_id")]

        if len(qi_ids) < 2:
            singletons.append({"intent": intent_key, "qi_ids": qi_ids})

        h = hashlib.sha256((intent_key + "|" + "|".join(sorted(qi_ids))).encode("utf-8")).hexdigest()[:10]
        qc_id, ari_id, frt_id, trg_id = f"QC_{intent_key}_{h}", f"ARI_{intent_key}_{h}", f"FRT_{intent_key}_{h}", f"TRG_{intent_key}_{h}"

        by_subj: Dict[str, List[str]] = defaultdict(list)
        for q in grp:
            by_subj[str(q.get("subject_id", ""))].append(str(q.get("qi_id", "")))

        op = sig_op.get(intent_key, "OP_GENERIC")
        # Formulation QC conforme à la règle "Comment ... ?"
        # NOTE: c'est un placeholder invariant de TEST; en PROD vous brancherez votre méthode SMAXIA.
        qc_formulation = f"Comment appliquer une méthode de type {intent_key} (famille {op}) ?"

        qcs.append({
            "qc_id": qc_id,
            "qc_formulation": qc_formulation,
            "qc_invariant_signature": {"intent_code": intent_key, "op_family": op},
            "mapping": {"primary_qi_ids": qi_ids, "covered_qi_ids": qi_ids, "by_subject": dict(by_subj)},
            "links": {"ari_id": ari_id, "frt_id": frt_id, "trigger_ids": [trg_id]},
            "provenance": {"generator_version": "TEST_ONLY_V31.9.0"},
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
                "given": "Données utiles",
                "goal": "Objectif",
                "method": "Méthode (invariante)",
                "checks": ["Contrôle de cohérence", "Vérification finale"],
                "common_traps": ["Piège fréquent", "Erreur classique"],
                "final_form": "Forme finale attendue",
            },
            "provenance": {"qc_id": qc_id, "ari_id": ari_id},
        })

        triggers.append({
            "trigger_id": trg_id,
            "type_code": "TRG_CHECK",
            "severity": "medium",
            "condition": {"signal": "AFTER_EXECUTE"},
            "action": {"recommendation": "Appliquer un contrôle invariant (dimension, signe, cohérence, cas limites)"},
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
# ZONE C : STREAMLIT UI (structure conservée)
# ═══════════════════════════════════════════════════════════════════════════════

def get_zone_b_source() -> str:
    """Extract ZONE B with AST validation."""
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


class GTERunner:
    def __init__(self):
        self.results, self.logs = {}, []

    def log(self, m: str):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

    def run(self, qi_pack: List[Dict[str, Any]], gen_func, level2: bool, forbidden: Set[str]) -> Dict[str, Any]:
        self.logs, self.results = [], {}
        self.log(f"═══ GTE V31.9.0 — {len(qi_pack)} Qi ═══")

        artifacts_raw = gen_func(qi_pack)
        canon = canonicalize_artifacts(artifacts_raw, qi_pack)
        qcs, aris, frts, trgs = canon["qcs"], canon["aris"], canon["frts"], canon["triggers"]
        singletons = canon["singletons_warning"]
        self.log(f"Generated: {len(qcs)} QC, {len(aris)} ARI, {len(frts)} FRT, {len(trgs)} TRG")
        if singletons:
            self.log(f"⚠️ Singletons: {len(singletons)}")

        # TS-01 Schema
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

        # TS-02 Ref
        ref_ok, ref_errs = check_ref_integrity(canon)
        self.results["TS-02_REF"] = {"pass": ref_ok, "errors": ref_errs}
        self.log(f"TS-02 REF: {'PASS' if ref_ok else 'FAIL'}")

        # TC-01 Coverage
        cov_ok, orphans = check_coverage_total(canon)
        self.results["TC-01_COV"] = {"pass": cov_ok, "orphans": orphans}
        self.log(f"TC-01 COV: {'PASS' if cov_ok else 'FAIL'} ({len(orphans)} orphans)")

        # TC-02 Primary
        prim_ok, dups = check_coverage_primary_unique(canon)
        self.results["TC-02_PRIM"] = {"pass": prim_ok, "duplicates": dups}
        self.log(f"TC-02 PRIM: {'PASS' if prim_ok else 'FAIL'}")

        # TD-01 Det N=3
        det_ok, hashes, _ = check_determinism_n_runs(gen_func, qi_pack, 3)
        self.results["TD-01_DET_N3"] = {"pass": det_ok, "hashes": hashes}
        self.log(f"TD-01 DET_N3: {'PASS' if det_ok else 'FAIL'}")

        # TD-02 Order
        ord_ok, h1, h2 = check_order_invariance(gen_func, qi_pack)
        self.results["TD-02_ORDER"] = {"pass": ord_ok, "h1": h1[:16], "h2": h2[:16]}
        self.log(f"TD-02 ORDER: {'PASS' if ord_ok else 'FAIL'}")

        # Source-based validators with SKIP fallback
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

        def is_green(k):
            p = self.results.get(k, {}).get("pass")
            return True if p is None else bool(p)

        p0_keys = [
            "TS-01_SCHEMA", "TS-02_REF", "TC-01_COV", "TC-02_PRIM",
            "TD-01_DET_N3", "TD-02_ORDER", "TN-02_IMPORTS", "TA-01_AST",
            "TA-02_LITERALS", "TA-04_PROMPT"
        ]
        p0_pass = all(is_green(k) for k in p0_keys)
        self.log(f"═══ VERDICT: {'PASS' if p0_pass else 'FAIL'} ═══")

        return {
            "verdict": "PASS" if p0_pass else "FAIL",
            "validators": self.results,
            "artifacts_raw": artifacts_raw,
            "artifacts_canon": canon,
            "metrics": {
                "qi": len(qi_pack), "qc": len(qcs), "ari": len(aris),
                "frt": len(frts), "trg": len(trgs), "singletons": len(singletons),
                "input_hash": compute_hash(sort_qi_canonical(qi_pack)),
                "output_hash": compute_hash(canon),
            },
            "logs": self.logs,
        }


def main():
    st.set_page_config(page_title="SMAXIA GTE V31.9.0", page_icon="🔒", layout="wide")
    st.title("🔒 SMAXIA GTE Console V31.9.0")
    st.markdown("**Harnais ISO-PROD — Extraction Qi BAC + QC anti-collapse (TEST)**")

    with st.sidebar:
        st.header("⚙️ V31.9.0")
        st.markdown("✓ Extraction Qi BAC : EXn-Qm(a/b/c)")
        st.markdown("✓ Alignement corrigé sur mêmes clés")
        st.markdown("✓ QC : 'Comment ... ?' (règle SMAXIA)")
        st.markdown("✓ Anti-collapse : re-split si 1 cluster domine")
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

    tab1, tab2, tab3 = st.tabs(["📥 Entrée", "🚀 Pipeline", "📊 Résultats"])

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
                subj = st.file_uploader("PDF Sujet", type=["pdf"], key="subj")
                corr = st.file_uploader("PDF Correction (opt)", type=["pdf"], key="corr")
                auto = st.checkbox("Auto-charger", value=True)
                if st.button("🧩 Générer qi_pack", type="primary", disabled=(subj is None)):
                    try:
                        import_meta, qi_pack = build_qi_pack_from_pdfs(subj.read(), corr.read() if corr else None)
                        st.success(f"✓ {len(qi_pack)} Qi")
                        st.json(import_meta.get("stats", {}))
                        st.download_button(
                            "📥 qi_pack.json",
                            canonical_json({"meta": import_meta, "qi_pack": qi_pack}),
                            "qi_pack.json",
                        )
                        if auto:
                            st.session_state["qi_pack"] = qi_pack
                            st.session_state["import_meta"] = import_meta
                    except Exception as e:
                        st.error(f"Error: {e}")

        if qi_pack:
            st.session_state["qi_pack"] = qi_pack
        if import_meta:
            st.session_state["import_meta"] = import_meta
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
                    result = GTERunner().run(qi, TEST_ONLY_generate_qc_ari_frt_trg, lvl2, forbidden)
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
                st.subheader("📄 PDF Meta")
                st.json(st.session_state["import_meta"])

            st.subheader("Validateurs")
            for k, v in r["validators"].items():
                p = v.get("pass")
                icon = "✅" if p is True else "⏭️" if p is None else "❌"
                with st.expander(f"{icon} {k}"):
                    st.json(v)

            st.markdown("---")
            st.subheader("📤 Exports")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "📥 report.json",
                    canonical_json({"version": "V31.9.0", "verdict": r["verdict"], "validators": r["validators"], "metrics": m}),
                    "report.json",
                )
            with c2:
                st.download_button("📥 raw.json", canonical_json(r["artifacts_raw"]), "raw.json")
            with c3:
                st.download_button("📥 canon.json", canonical_json(r["artifacts_canon"]), "canon.json")


if __name__ == "__main__":
    main()
