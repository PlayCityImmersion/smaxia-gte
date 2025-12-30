# =============================================================================
# smaxia_console_v31_9_5.py — SMAXIA GTE Console V31.9.5 (ISO-PROD TEST)
# =============================================================================
# OBJECTIF CEO (V31.9.5) — SANS RÉGRESSION UI (3 tabs + exports + validateurs)
#
# 1) Choix PAYS=FR  => le Academic Pack France est chargé AUTOMATIQUEMENT
#    et VISUALISABLE dans la Sidebar (preuve que la section fonctionne).
#
# 2) Auto-Scrape (HARVEST) : l’utilisateur saisit VOLUME => bouton ON => Lancer.
#    Doctrine Harvest : "Pêche à l’aveugle" (aucun filtrage métier/chapter avant capture),
#    uniquement régulé par lois internes (invariance / non-interference / déterminisme).
#
# 3) Après Harvest : la liste des SUJETS + CORRIGÉS scrappés doit apparaître
#    dans la zone Import Sujet / Import PDF (sélectionnable).
#    NOTE STREAMLIT : on ne peut pas auto-remplir un file_uploader; donc on fournit
#    une "Bibliothèque Harvest" + Selectbox qui charge les bytes en session (équivalent ISO).
#
# 4) Pipeline existant : Générer qi_pack.json -> RUN GTE -> Résultats
#    + Explorateur Chapitre : si je choisis un chapitre => QC de ce chapitre
#    + ARI/FRT/TRG + Qi associées.
#
# LOIS SMAXIA (RAPPEL)
# - Zéro hardcode métier (pays/matière/chapitre/langue) dans le CORE.
# - La variabilité vient UNIQUEMENT du academic_pack.json (PACK DATA).
# - "Chapitre" est pack-driven : QC->chapitre via intent_allowlist du pack (JAMAIS par texte).
# - Validateurs B : BLOQUANTS (si FAIL => FAIL).
#
# IMPORTANT
# - Le générateur QC/ARI/FRT/TRG ci-dessous est TEST-ONLY (structure-driven).
#   Il sert à valider l’infrastructure + la preuve "pack-driven chapters + coverage".
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


# =============================================================================
# ZONE A — TEST-ONLY FIXTURES (autorisées en ISO-PROD TEST, interdites en PROD)
# =============================================================================

# (A1) Pack France par défaut (fallback) : uniquement pour garantir “PAYS=FR => pack visible”
# La règle PROD reste : pack fourni par l’activation pays (academic_pack.json).
TEST_ONLY_DEFAULT_PACK_FR: Dict[str, Any] = {
    "pack_id": "FR_TEST_DEFAULT_PACK",
    "country_code": "FR",
    "signature": {
        "pack_hash": "TEST_ONLY_PLACEHOLDER_HASH",
        "signed_at": "TEST_ONLY",
    },
    # Chapitres : intent_allowlist = codes d’intent invariants (ici structurels)
    "chapters": [
        {"chapter_id": "CH_FR_001", "title": "Chapitre 1 (TEST)", "intent_allowlist": ["STRUCT_EX1_Q1", "STRUCT_EX1_Q2"]},
        {"chapter_id": "CH_FR_002", "title": "Chapitre 2 (TEST)", "intent_allowlist": ["STRUCT_EX2_Q1", "STRUCT_EX2_Q2"]},
        {"chapter_id": "CH_FR_003", "title": "Chapitre 3 (TEST)", "intent_allowlist": ["STRUCT_EX3_Q1", "STRUCT_EX3_Q2"]},
    ],
    # Harvest : seeds + regex viennent du pack (DATA). Pour fallback test-only, on met un exemple vide.
    # Si vous uploadez votre pack FR réel (Gemini/Opus), il remplacera celui-ci.
    "harvest": {
        "sequencing": {"dimensions": ["blind_fishing"]},
        "http": {"user_agent": "Mozilla/5.0", "timeout_sec": 25, "max_bytes_pdf": 25_000_000, "max_index_pages": 50},
        "seeds": [
            # TEST-ONLY placeholder : si vous gardez ce pack, auto-scrape ne trouvera rien.
            # => Uploadez votre pack FR réel pour activer harvest (avec seeds/regex).
            # {"index_url":"https://example.org", "pdf_regex": r"\.pdf($|\?)", "corr_hint_regex": r"corrige|correction"}
        ],
    },
}

# (A2) Golden pack Qi test
TEST_ONLY_GOLDEN_QI_PACK: List[Dict[str, Any]] = [
    {"qi_id": "QI_TEST_001", "subject_id": "SUBJ_A", "position": {"order_index": 1, "marker": "EX1-Q1"},
     "statement": {"text_md": "Texte question 1"}, "correction": {"available": True, "text_md": "Correction 1"}},
    {"qi_id": "QI_TEST_002", "subject_id": "SUBJ_A", "position": {"order_index": 2, "marker": "EX1-Q2a"},
     "statement": {"text_md": "Texte question 2a"}, "correction": {"available": True, "text_md": "Correction 2a"}},
    {"qi_id": "QI_TEST_003", "subject_id": "SUBJ_A", "position": {"order_index": 3, "marker": "EX1-Q2b"},
     "statement": {"text_md": "Texte question 2b"}, "correction": {"available": True, "text_md": "Correction 2b"}},
    {"qi_id": "QI_TEST_004", "subject_id": "SUBJ_A", "position": {"order_index": 4, "marker": "EX2-Q1"},
     "statement": {"text_md": "Texte question EX2-Q1"}, "correction": {"available": True, "text_md": "Correction EX2-Q1"}},
]


# =============================================================================
# ZONE B — INVARIANT HELPERS (CORE-LIKE)
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


def scan_forbidden_literals(source: str, forbidden: Set[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    if not forbidden:
        return True, []
    violations = []
    for i, line in enumerate(source.split("\n"), 1):
        s = line.strip()
        if s.startswith("#") or not s:
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


def get_zone_b_source() -> str:
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return ""
    # Heuristique simple : on prend tout le fichier (permet TA-01/TA-02)
    try:
        ast.parse(src)
    except Exception:
        return ""
    return src


# =============================================================================
# ZONE B.5 — PDF → Qi Builder (markers EXn-Qm[a/b/c])
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


# =============================================================================
# ZONE B.7 — TEST GENERATOR QC/ARI/FRT/TRG (structure-driven, invariant)
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


# =============================================================================
# ZONE B.8 — ACADEMIC PACK + VALIDATEURS B (BLOQUANTS)
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
            else:
                for it in ial:
                    if not str(it).strip():
                        errs.append(f"chapters[{i}].intent_allowlist has empty")
    sig = pack.get("signature")
    if not isinstance(sig, dict) or "pack_hash" not in sig:
        errs.append("signature.pack_hash missing")
    return len(errs) == 0, errs


def validate_pack_harvest(pack: Dict[str, Any]) -> Tuple[bool, List[str]]:
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


def validators_B(canon: Dict[str, Any], pack: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
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

    # Intent uniqueness across chapters (anti-leak)
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

    # QC must map to a chapter (via intent_code)
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

    # Qi must be assigned to exactly 1 chapter via covered_qi_ids of mapped QCs (no cross leak)
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

    # Coverage per chapter (within assigned Qi of that chapter)
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

    all_orphans = []
    for ch in chapters_ids:
        assigned = set(chapter_qi.get(ch, []))
        covered = chapter_cov.get(ch, set())
        orph = sorted([q for q in assigned if q not in covered])
        if orph:
            all_orphans.extend(orph)
        chapter_report["chapters"][ch] = {
            "title": next((str(x.get("title", "")) for x in (pack.get("chapters") or []) if str(x.get("chapter_id", "")).strip() == ch), ""),
            "qc_count": len(chapter_qc.get(ch, [])),
            "qi_total": len(chapter_qi.get(ch, [])),
            "qi_covered": max(0, len(chapter_qi.get(ch, [])) - len(orph)),
            "coverage_ratio": (1.0 if len(chapter_qi.get(ch, [])) == 0 else (len(chapter_qi.get(ch, [])) - len(orph)) / float(len(chapter_qi.get(ch, [])))),
            "orphans": orph[:500],
            "qc_ids": chapter_qc.get(ch, [])[:500],
        }

    out["B-05_COVERAGE_PER_CHAPTER"] = {
        "pass": len(all_orphans) == 0,
        "orphans_total": len(all_orphans),
        "orphans_preview": all_orphans[:200],
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

    # B keys are BLOQUANTS => on laisse le runner trancher (FAIL si un B pass==False)
    return out, chapter_report


# =============================================================================
# ZONE B.9 — AUTO-SCRAPER (pack-driven, blind fishing)
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


def pack_auto_scrape_blind(pack: Dict[str, Any], volume: int) -> Dict[str, Any]:
    """
    Blind fishing:
      - Récupère les liens PDF depuis pack.harvest.seeds (index pages)
      - Aucun filtrage métier/chapitre/niveau AVANT capture
      - On échantillonne (stable) jusqu’à 'volume'
      - On tente pairing sujet/corrigé via corr_hint_regex (data-driven)
    """
    okh, herrs = validate_pack_harvest(pack)
    if not okh:
        return {"meta": {"ok": False, "errors": herrs, "created_at": datetime.now().isoformat()}, "items": []}

    h = pack["harvest"]
    http = h.get("http", {}) or {}
    ua = str(http.get("user_agent") or "Mozilla/5.0")
    timeout = int(http.get("timeout_sec") or 25)
    max_pdf = int(http.get("max_bytes_pdf") or 25_000_000)
    max_index_pages = int(http.get("max_index_pages") or 50)

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

    # de-dup by subject_url
    seen_subj = set()
    pairs_unique = []
    for subj_url, corr_url in collected_pairs:
        if subj_url in seen_subj:
            continue
        seen_subj.add(subj_url)
        pairs_unique.append((subj_url, corr_url))

    # Blind fishing sampling (stable deterministic sampling)
    # - deterministic: seed fixed + stable list order
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
        "doctrine": "blind_fishing_then_depessage",
    }
    return {"meta": meta, "items": items}


# =============================================================================
# ZONE C — STREAMLIT UI (3 tabs + modules V31.9.5)
# =============================================================================

def load_pack_auto_for_country(country_code: str) -> Dict[str, Any]:
    """
    Règle ISO-PROD TEST:
      - si un fichier local academic_pack_FR.json existe => le charger (pas de hardcode métier dans code)
      - sinon fallback TEST_ONLY_DEFAULT_PACK_FR
    """
    fname = f"academic_pack_{country_code}.json"
    p = Path(fname)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return TEST_ONLY_DEFAULT_PACK_FR
    return TEST_ONLY_DEFAULT_PACK_FR


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
        self.logs: List[str] = []
        self.validators: Dict[str, Any] = {}

    def log(self, m: str):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

    def run(self, qi_pack: List[Dict[str, Any]], gen_func, forbidden: Set[str], pack: Optional[Dict[str, Any]], lvl2: bool) -> Dict[str, Any]:
        self.logs = []
        self.validators = {}
        self.log(f"═══ GTE V31.9.5 — {len(qi_pack)} Qi ═══")

        artifacts_raw = gen_func(qi_pack)
        canon = canonicalize_artifacts(artifacts_raw, qi_pack)
        qcs, aris, frts, trgs = canon["qcs"], canon["aris"], canon["frts"], canon["triggers"]
        singletons = canon["singletons_warning"]

        self.log(f"Generated: {len(qcs)} QC, {len(aris)} ARI, {len(frts)} FRT, {len(trgs)} TRG")
        if singletons:
            self.log(f"⚠️ Singletons: {len(singletons)}")

        # Schemas
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

        # Ref integrity
        ref_ok, ref_errs = check_ref_integrity(canon)
        self.validators["TS-02_REF"] = {"pass": ref_ok, "errors": ref_errs}
        self.log(f"TS-02 REF: {'PASS' if ref_ok else 'FAIL'}")

        # Coverage total
        cov_ok, orphans = check_coverage_total(canon)
        self.validators["TC-01_COV"] = {"pass": cov_ok, "orphans": orphans}
        self.log(f"TC-01 COV: {'PASS' if cov_ok else 'FAIL'} ({len(orphans)} orphans)")

        # Primary unique
        prim_ok, dups = check_coverage_primary_unique(canon)
        self.validators["TC-02_PRIM"] = {"pass": prim_ok, "duplicates": dups}
        self.log(f"TC-02 PRIM: {'PASS' if prim_ok else 'FAIL'}")

        # Determinism
        det_ok, hashes = check_determinism_n_runs(gen_func, qi_pack, 3)
        self.validators["TD-01_DET_N3"] = {"pass": det_ok, "hashes": hashes}
        self.log(f"TD-01 DET_N3: {'PASS' if det_ok else 'FAIL'}")

        # Order invariance
        ord_ok, h1, h2 = check_order_invariance(gen_func, qi_pack)
        self.validators["TD-02_ORDER"] = {"pass": ord_ok, "h1": h1[:16], "h2": h2[:16]}
        self.log(f"TD-02 ORDER: {'PASS' if ord_ok else 'FAIL'}")

        # TA checks
        src = get_zone_b_source()
        if src:
            ast_ok, ast_v = scan_ast_sensitive_access(src)
            self.validators["TA-01_AST"] = {"pass": ast_ok, "violations": ast_v}
            self.log(f"TA-01 AST: {'PASS' if ast_ok else 'FAIL'}")

            lit_ok, lit_v = scan_forbidden_literals(src, forbidden)
            self.validators["TA-02_LITERALS"] = {"pass": lit_ok, "violations": lit_v, "mode": ("ACTIVE" if forbidden else "SKIP")}
            if forbidden:
                self.log(f"TA-02 LITERALS: {'PASS' if lit_ok else 'FAIL'} ({len(forbidden)} terms)")
            else:
                self.log("TA-02 LITERALS: SKIP (no dict)")
        else:
            self.validators["TA-01_AST"] = {"pass": None, "note": "SKIP"}
            self.validators["TA-02_LITERALS"] = {"pass": None, "note": "SKIP"}
            self.log("TA-01/02: SKIP (source N/A)")

        # Level2 (optional)
        if lvl2:
            d10_ok, h10 = check_determinism_n_runs(gen_func, qi_pack, 10)
            self.validators["TD-03_N10"] = {"pass": d10_ok, "hashes": h10}
            self.log(f"TD-03 N10: {'PASS' if d10_ok else 'FAIL'}")
        else:
            self.validators["TD-03_N10"] = {"pass": None, "note": "Skipped"}

        # Validateurs B BLOQUANTS
        bvals, chapter_report = validators_B(canon, pack)
        for k, v in bvals.items():
            self.validators[k] = v
            self.log(f"{k}: {'PASS' if v.get('pass') else 'FAIL'}")

        # Verdict = PASS seulement si tout ce qui est booléen est True (None ignoré),
        # et si TOUS les B-* sont True (BLOQUANTS).
        def is_green(k: str) -> bool:
            p = self.validators.get(k, {}).get("pass")
            return True if p is None else bool(p)

        must = ["TS-01_SCHEMA", "TS-02_REF", "TC-01_COV", "TC-02_PRIM", "TD-01_DET_N3", "TD-02_ORDER"]
        # Bloquants B
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
# APP
# =============================================================================

def main():
    st.set_page_config(page_title="SMAXIA GTE V31.9.5", page_icon="🔒", layout="wide")
    st.title("🔒 SMAXIA GTE Console V31.9.5")
    st.markdown("**Harnais ISO-PROD — Extraction Qi + QC structure-driven (invariant) + Pack-driven Chapters + Harvest Blind Fishing**")

    # Session defaults
    st.session_state.setdefault("qi_pack", [])
    st.session_state.setdefault("import_meta", None)
    st.session_state.setdefault("harvest_items", [])
    st.session_state.setdefault("harvest_meta", None)
    st.session_state.setdefault("import_subject_bytes", None)
    st.session_state.setdefault("import_correction_bytes", None)
    st.session_state.setdefault("import_label", None)
    st.session_state.setdefault("active_pack", None)
    st.session_state.setdefault("active_pack_source", "AUTO")
    st.session_state.setdefault("country_code", "FR")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("⚙️ V31.9.5")
        st.markdown("✓ QC anti-collapse : clustering structurel par markers (EXn-Qm)")
        st.markdown("✓ Zéro keywords métier/langue dans le générateur QC")
        st.markdown("✓ QC formulation conforme : 'Comment ... ?'")
        st.markdown("✓ UI inchangée : 3 tabs + exports + validateurs")
        st.markdown("✓ Chapitres pack-driven (academic_pack.json)")
        st.markdown("✓ Harvest : Blind Fishing (pack.harvest)")

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
        country = st.selectbox("Pays", ["FR"], index=0, key="country_select")
        st.session_state["country_code"] = country

        st.markdown("### Academic Pack (Pack-driven)")
        st.caption("Règle ISO-PROD TEST: PAYS=FR charge un pack par défaut si aucun upload.")
        pack_up = st.file_uploader("academic_pack.json (upload override)", type=["json"], key="pack_upl")

        pack_override = None
        if pack_up:
            raw, err = load_json_from_uploaded(pack_up)
            if err:
                st.error(err)
            else:
                if isinstance(raw, dict):
                    pack_override = raw
                else:
                    st.error("academic_pack.json doit être un objet JSON")

        # Auto load pack when country changes or on first run
        if pack_override is not None:
            pack = pack_override
            st.session_state["active_pack_source"] = "UPLOAD"
        else:
            # Auto-load from local file if present, else fallback fixture
            pack = load_pack_auto_for_country(country)
            st.session_state["active_pack_source"] = "AUTO"

        st.session_state["active_pack"] = pack

        okp, perrs = validate_pack_schema(pack) if isinstance(pack, dict) else (False, ["Pack is not dict"])
        if okp:
            st.success(f"Pack chargé: {pack.get('pack_id')} ({pack.get('country_code')}) — source={st.session_state['active_pack_source']}")
        else:
            st.error("Pack invalide (B-01)")
            st.json(perrs)

        with st.expander("Voir Academic Pack (résumé)", expanded=True):
            st.json({
                "source": st.session_state["active_pack_source"],
                "pack_id": pack.get("pack_id") if isinstance(pack, dict) else None,
                "country_code": pack.get("country_code") if isinstance(pack, dict) else None,
                "signature": pack.get("signature") if isinstance(pack, dict) else None,
                "chapters_count": len(pack.get("chapters") or []) if isinstance(pack, dict) else 0,
                "chapters_preview": (pack.get("chapters") or [])[:15] if isinstance(pack, dict) else [],
                "has_harvest": bool(pack.get("harvest")) if isinstance(pack, dict) else False,
                "harvest_seeds_count": len((pack.get("harvest") or {}).get("seeds") or []) if isinstance(pack, dict) else 0,
            })

        st.markdown("---")
        st.subheader("Auto-Scrape (Blind Fishing)")
        st.caption("Bouton ON si pack.harvest.seeds est non vide. (Upload pack FR réel pour activer.)")
        vol = st.number_input("Volume (nb de sujets à récupérer)", min_value=1, value=20, step=5, key="auto_vol")

        harvest_ready = False
        if isinstance(pack, dict):
            okh, _ = validate_pack_harvest(pack)
            harvest_ready = okh

        btn_disabled = not (okp and harvest_ready and int(vol) > 0)

        if st.button("⛽ LANCER AUTO-SCRAPE", type="primary", use_container_width=True, disabled=btn_disabled):
            with st.spinner("Harvest (blind fishing) en cours..."):
                out = pack_auto_scrape_blind(pack, int(vol))
                st.session_state["harvest_meta"] = out.get("meta")
                st.session_state["harvest_items"] = out.get("items") or []

                # Auto-charge le premier item harvest dans Import PDF (équivalent “apparaitre” + prêt à générer)
                if st.session_state["harvest_items"]:
                    first = st.session_state["harvest_items"][0]
                    st.session_state["import_subject_bytes"] = first.get("subject_bytes")
                    st.session_state["import_correction_bytes"] = first.get("correction_bytes")
                    st.session_state["import_label"] = first.get("label")

            st.success(f"Harvest OK: {len(st.session_state['harvest_items'])} items disponibles")

        if not harvest_ready:
            st.warning("Auto-scrape OFF: pack.harvest.seeds vide ou absent. Uploadez votre academic_pack FR réel.")
        elif btn_disabled:
            st.info("Auto-scrape prêt, mais conditions non remplies (pack valide + seeds + volume).")

        if st.session_state.get("harvest_items"):
            st.success(f"Bibliothèque Harvest: {len(st.session_state['harvest_items'])} items")
        else:
            st.info("Bibliothèque Harvest vide")

    # ---------------- Tabs ----------------
    tab1, tab2, tab3 = st.tabs(["📥 Entrée", "🚀 Pipeline", "📊 Résultats"])

    # TAB 1
    with tab1:
        st.header("📥 Entrée")
        mode = st.radio("Mode", ["Golden Pack", "JSON Upload", "Manuel", "Import Sujet (PDF)"], horizontal=True)

        qi_pack: List[Dict[str, Any]] = []
        import_meta = None
        pack = st.session_state.get("active_pack")

        if mode == "Golden Pack":
            qi_pack = TEST_ONLY_GOLDEN_QI_PACK
            st.info(f"🧪 Golden Pack: {len(qi_pack)} Qi")

        elif mode == "JSON Upload":
            up = st.file_uploader("qi_pack.json", type=["json"], key="qi_json")
            if up:
                raw, err = load_json_from_uploaded(up)
                if err:
                    st.error(err)
                else:
                    if isinstance(raw, dict) and "qi_pack" in raw:
                        qi_pack = raw["qi_pack"]
                        import_meta = raw.get("meta")
                    elif isinstance(raw, list):
                        qi_pack = raw
                    else:
                        st.error("Format attendu: liste Qi ou {meta, qi_pack}")
                    st.success(f"✓ {len(qi_pack)} Qi")

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
                st.markdown("#### Bibliothèque Harvest (Auto-Scrape)")
                harvest_items = st.session_state.get("harvest_items") or []
                if harvest_items:
                    labels = [
                        f"{i+1:03d} — {it.get('label','subject.pdf')} "
                        f"{'(corr)' if it.get('correction_bytes') else '(no corr)'}"
                        for i, it in enumerate(harvest_items)
                    ]
                    pick = st.selectbox("Choisir un sujet scrappé", list(range(len(labels))),
                                        format_func=lambda i: labels[i], key="harv_pick")

                    cA, cB, cC = st.columns([1, 1, 2])
                    with cA:
                        if st.button("📌 Charger ce sujet scrappé", use_container_width=True):
                            it = harvest_items[int(pick)]
                            st.session_state["import_subject_bytes"] = it.get("subject_bytes")
                            st.session_state["import_correction_bytes"] = it.get("correction_bytes")
                            st.session_state["import_label"] = it.get("label")
                            st.success("Chargé en session (Import PDF).")

                    with cB:
                        if st.button("🧹 Vider Import session", use_container_width=True):
                            st.session_state["import_subject_bytes"] = None
                            st.session_state["import_correction_bytes"] = None
                            st.session_state["import_label"] = None
                            st.info("Session import vidée.")

                    with cC:
                        st.caption("Streamlit ne peut pas auto-remplir un file_uploader : cette sélection est l’équivalent ISO-PROD.")

                    if st.session_state.get("harvest_meta"):
                        with st.expander("Meta auto-scrape", expanded=False):
                            st.json(st.session_state["harvest_meta"])
                else:
                    st.info("Pas d’items harvest. Lancez AUTO-SCRAPE dans la Sidebar (pack.harvest requis).")

                st.markdown("---")
                st.markdown("#### Import manuel (upload) — toujours disponible")
                subj = st.file_uploader("PDF Sujet (upload)", type=["pdf"], key="subj_up")
                corr = st.file_uploader("PDF Correction (opt) (upload)", type=["pdf"], key="corr_up")

                auto = st.checkbox("Charger automatiquement dans GTE après génération", value=True)

                if st.button("🧩 Générer qi_pack.json", type="primary"):
                    sb = st.session_state.get("import_subject_bytes")
                    cb = st.session_state.get("import_correction_bytes")

                    # Si pas de sélection harvest, on prend upload
                    if sb is None and subj is not None:
                        sb = subj.read()
                        cb = corr.read() if corr else None

                    if sb is None:
                        st.error("Aucun PDF sélectionné (ni harvest, ni upload).")
                    else:
                        try:
                            import_meta, qi_pack = build_qi_pack_from_pdfs(sb, cb)
                            st.success(f"✓ Généré: {len(qi_pack)} Qi")
                            st.expander("Meta extraction / alignement", expanded=False).json(import_meta)
                            st.download_button("📥 Télécharger qi_pack.json", canonical_json({"meta": import_meta, "qi_pack": qi_pack}), "qi_pack.json")
                            if auto:
                                st.session_state["qi_pack"] = qi_pack
                                st.session_state["import_meta"] = import_meta
                                st.info("✓ qi_pack chargé (tab Pipeline).")
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

    # TAB 2
    with tab2:
        st.header("🚀 Pipeline")
        lvl2 = st.checkbox("Level 2 (N=10)")
        st.markdown("### RUN GTE (sur qi_pack courant)")

        if st.button("▶️ RUN GTE", type="primary", use_container_width=True):
            qi = st.session_state.get("qi_pack", [])
            if not qi:
                st.error("Aucun qi_pack. Générez ou uploadez d’abord.")
            else:
                pack = st.session_state.get("active_pack")
                with st.spinner("Running GTE..."):
                    res = GTERunner().run(qi, TEST_ONLY_generate_qc_ari_frt_trg, forbidden, pack, lvl2)
                    st.session_state["result"] = res

                for l in res["logs"]:
                    if "FAIL" in l:
                        st.markdown(f"🔴 `{l}`")
                    elif "PASS" in l:
                        st.markdown(f"🟢 `{l}`")
                    elif "⚠️" in l:
                        st.markdown(f"🟡 `{l}`")
                    else:
                        st.text(l)

                st.markdown("---")
                if res["verdict"] == "PASS":
                    st.success("# ✅ PASS")
                else:
                    st.error("# ❌ FAIL (B validateurs bloquants inclus)")

    # TAB 3
    with tab3:
        st.header("📊 Résultats")

        if "result" not in st.session_state:
            st.info("Exécutez RUN GTE (tab Pipeline).")
            return

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

        st.subheader("📚 Chapter Report (preuves pack-driven)")
        st.json(r.get("chapter_report") or {"note": "No chapter report"})

        st.subheader("Validateurs")
        for k, v in r["validators"].items():
            p = v.get("pass")
            icon = "✅" if p is True else "❌" if p is False else "⏭️"
            with st.expander(f"{icon} {k}", expanded=False):
                st.json(v)

        st.markdown("---")
        st.subheader("🧭 Explorer Chapitre (QC + ARI/FRT/TRG + Qi associées)")
        pack = st.session_state.get("active_pack")
        if not isinstance(pack, dict) or not pack.get("chapters"):
            st.warning("Pack absent ou invalide.")
        else:
            canon = r.get("artifacts_canon") or {}
            by_ch = artifacts_by_chapter(canon, pack).get("chapters") or {}

            # Liste des chapitres (on affiche même ceux sans QC)
            chapters = pack.get("chapters") or []
            chapter_ids = [str(c.get("chapter_id", "")).strip() for c in chapters if str(c.get("chapter_id", "")).strip()]

            def ch_label(cid: str) -> str:
                title = ""
                for ch in chapters:
                    if str(ch.get("chapter_id", "")).strip() == cid:
                        title = str(ch.get("title", "")).strip()
                        break
                return f"{cid} — {title}" if title else cid

            sel = st.selectbox("Chapitre", chapter_ids, format_func=ch_label, key="chapter_sel")

            obj = by_ch.get(sel) or {"qcs": [], "aris": {}, "frts": {}, "triggers": {}, "qi": {}}
            qc_list = obj.get("qcs") or []
            st.info(f"QC mappées sur ce chapitre: {len(qc_list)}")

            with st.expander("Liste QC (détails)", expanded=True):
                for qc in qc_list[:200]:
                    qc_id = qc.get("qc_id")
                    st.markdown(f"**{qc_id}** — {qc.get('qc_formulation')}")
                    links = qc.get("links") or {}
                    st.caption(
                        f"ARI={links.get('ari_id')} | FRT={links.get('frt_id')} | "
                        f"TRG={len(links.get('trigger_ids') or [])} | "
                        f"Qi={len((qc.get('mapping') or {}).get('covered_qi_ids') or [])}"
                    )
                    st.json({
                        "qc": qc,
                        "ari": obj.get("aris", {}).get(links.get("ari_id")),
                        "frt": obj.get("frts", {}).get(links.get("frt_id")),
                        "triggers": [obj.get("triggers", {}).get(t) for t in (links.get("trigger_ids") or []) if obj.get("triggers", {}).get(t)],
                        "qi_examples": [obj.get("qi", {}).get(q) for q in ((qc.get("mapping") or {}).get("covered_qi_ids") or [])[:5] if obj.get("qi", {}).get(q)],
                    })

            st.download_button("📥 Export chapter_bundle.json",
                               canonical_json({
                                   "chapter_id": sel,
                                   "chapter_title": ch_label(sel),
                                   "qcs": qc_list,
                                   "aris": obj.get("aris"),
                                   "frts": obj.get("frts"),
                                   "triggers": obj.get("triggers"),
                                   "qi": obj.get("qi"),
                                   "proof": (r.get("chapter_report") or {}).get("chapters", {}).get(sel, {}),
                                   "created_at": datetime.now().isoformat(),
                               }),
                               "chapter_bundle.json")

        st.markdown("---")
        st.subheader("📤 Exports")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("📥 report.json",
                               canonical_json({"version": "V31.9.5", "verdict": r["verdict"], "validators": r["validators"], "metrics": m}),
                               "report.json")
        with c2:
            st.download_button("📥 raw.json", canonical_json(r["artifacts_raw"]), "raw.json")
        with c3:
            st.download_button("📥 canon.json", canonical_json(r["artifacts_canon"]), "canon.json")
        with c4:
            st.download_button("📥 chapter_report.json", canonical_json(r.get("chapter_report") or {}), "chapter_report.json")


if __name__ == "__main__":
    main()
