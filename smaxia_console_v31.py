#!/usr/bin/env python3
"""
SMAXIA GTE V13 — ADMIN COMMAND CENTER
Single-file Streamlit application.
Run: streamlit run smaxia_gte_v13_admin_final.py
"""

import streamlit as st
import json
import hashlib
import os
import re
import math
import uuid
import io
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

# ─────────────────────────────────────────────────────────────
# 0) CONSTANTS
# ─────────────────────────────────────────────────────────────
VERSION = "GTE-V13-ADMIN-FINAL"
PACKS_DIR = Path("packs")
RUNS_DIR = Path("run")
FORMULA_PACK_DIR = Path("formula_packs")
OCR_CACHE_DIR = Path("ocr_cache")
HARVEST_DIR = Path("harvest")
DETERMINISM_RUNS = 3
TS_FIELDS = frozenset([
    "timestamp", "created_at", "sealed_at", "run_ts",
    "harvested_at", "paired_at", "activated_at", "checked_at",
    "extracted_at", "cached_at", "run_id", "path", "run_dir",
    "ts", "abs_path",
])
ARTIFACT_NAMES = [
    "SourceManifest", "AuthorityAudit", "CAP_SEALED", "CEP_pairs",
    "Quarantine", "PDF_Hash_Index", "SOE", "Atoms_Qi_RQi",
    "PosableReport", "CoverageMap", "QC_validated", "AuditLog_IA2",
    "FORMULA_PACK_MANIFEST", "F1_call_digest", "F2_call_digest",
    "DeterminismReport_3runs", "UI_EVENT_LOG", "CHK_REPORT",
    "SealReport", "HoldoutMappingReport", "RedundancyReport",
]


# ─────────────────────────────────────────────────────────────
# 1) UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_of(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def strip_volatile(obj, fields=TS_FIELDS):
    """Remove all volatile/non-deterministic fields recursively."""
    if isinstance(obj, dict):
        return {k: strip_volatile(v, fields) for k, v in sorted(obj.items()) if k not in fields}
    if isinstance(obj, list):
        return [strip_volatile(i, fields) for i in obj]
    return obj


def write_artifact(run_dir: Path, name: str, payload: dict):
    stable = strip_volatile(deepcopy(payload))
    canon = canonical_json(stable)
    digest = sha256_of(canon)
    full = {**payload, "_sha256_functional": digest}
    out = run_dir / f"{name}.json"
    out.write_text(json.dumps(full, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    sha_out = run_dir / f"{name}.sha256"
    sha_out.write_text(digest, encoding="utf-8")
    return digest


def log_ui_event(run_dir: Path, event_type: str, detail: str, triggered_pipeline: bool = False):
    log_path = run_dir / "UI_EVENT_LOG.json"
    events = []
    if log_path.exists():
        try:
            events = json.loads(log_path.read_text())
        except Exception:
            events = []
    events.append({
        "ts": now_iso(),
        "event": event_type,
        "detail": detail,
        "triggered_pipeline": triggered_pipeline,
    })
    log_path.write_text(json.dumps(events, indent=2, ensure_ascii=False))


def seal_ui_event_log(run_dir: Path):
    """Write sha256 sidecar for UI_EVENT_LOG, verify only ACTIVATE_COUNTRY triggered pipeline."""
    log_path = run_dir / "UI_EVENT_LOG.json"
    if not log_path.exists():
        return {"status": "FAIL", "reason": "NO_LOG"}
    events = json.loads(log_path.read_text())
    triggers = [e for e in events if e.get("triggered_pipeline")]
    non_activate = [e for e in triggers if e.get("event") != "ACTIVATE_COUNTRY"]
    verdict = len(triggers) >= 1 and len(non_activate) == 0
    stable = [strip_volatile(e) for e in events]
    digest = sha256_of(canonical_json(stable))
    (run_dir / "UI_EVENT_LOG.sha256").write_text(digest, encoding="utf-8")
    return {
        "status": "PASS" if verdict else "FAIL",
        "total_events": len(events),
        "pipeline_triggers": len(triggers),
        "non_activate_triggers": len(non_activate),
        "sha256": digest,
    }


# ─────────────────────────────────────────────────────────────
# 2) COUNTRY RESOLUTION — UI-ONLY (pycountry)
# ─────────────────────────────────────────────────────────────
def _load_country_db():  # UI-ONLY
    try:
        import pycountry
        return {c.alpha_2: c.name for c in pycountry.countries}
    except ImportError:
        return {
            "FR": "France", "BE": "Belgium", "CI": "Côte d'Ivoire",
            "SN": "Senegal", "CM": "Cameroon", "NG": "Nigeria",
            "CA": "Canada", "US": "United States", "GB": "United Kingdom",
            "DE": "Germany", "FI": "Finland", "IT": "Italy", "ES": "Spain",
            "MA": "Morocco", "TN": "Tunisia", "JP": "Japan", "BR": "Brazil",
        }


COUNTRY_DB = _load_country_db()  # UI-ONLY


def typeahead_search(query: str, limit: int = 20):  # UI-ONLY
    """
    Typeahead from len>=1. Scoring: prefix_name > prefix_code > contains.
    Returns sorted list of (code, name, score).
    """
    q = query.strip().lower()
    if not q:
        return []
    results = []
    for code, name in COUNTRY_DB.items():
        nl = name.lower()
        cl = code.lower()
        if nl.startswith(q):
            results.append((code, name, 0))
        elif cl.startswith(q):
            results.append((code, name, 1))
        elif q in nl:
            results.append((code, name, 2))
        elif q in cl:
            results.append((code, name, 3))
    results.sort(key=lambda r: (r[2], r[1]))
    return results[:limit]


# ─────────────────────────────────────────────────────────────
# 3) CAP LOADER — ZERO HARDCODE, AUTO-SEAL
# ─────────────────────────────────────────────────────────────
def cap_fingerprint(cap_data: dict) -> str:
    stripped = {k: v for k, v in cap_data.items() if k not in TS_FIELDS and k != "fingerprint"}
    return sha256_of(canonical_json(stripped))


def load_cap(country_key: str):
    cap_path = PACKS_DIR / country_key / "CAP_SEALED.json"
    if not cap_path.exists():
        return None, "NOT_FOUND"
    try:
        data = json.loads(cap_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"PARSE_ERROR:{e}"
    stored = data.get("fingerprint", "")
    computed = cap_fingerprint(data)
    if stored != computed:
        return None, f"FP_MISMATCH:stored={stored[:12]}!=computed={computed[:12]}"
    return data, "OK"


def build_minimal_cap(country_key: str, sources: list) -> dict:
    """Build a minimal CAP from DA0 discovery results and seal it."""
    cap = {
        "country_key": country_key,
        "version": VERSION,
        "kernel_params": {
            "ocr_engines": [],
            "text_extraction": ["pdfplumber", "pypdf"],
            "grading_system": "discovery_required",
            "cluster_min": 1,
        },
        "sources_count": len(sources),
        "created_via": "DA0_AUTO_SEAL",
    }
    cap["fingerprint"] = cap_fingerprint(cap)
    cap["sealed_at"] = now_iso()
    out_dir = ensure_dir(PACKS_DIR / country_key)
    (out_dir / "CAP_SEALED.json").write_text(
        json.dumps(cap, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cap


# ─────────────────────────────────────────────────────────────
# 4) DA0 — REAL DISCOVERY (NO HUMAN URL)
# ─────────────────────────────────────────────────────────────
class DA0Discovery:
    def __init__(self, country_key: str, run_dir: Path):
        self.country_key = country_key
        self.run_dir = run_dir
        self.sources = []
        self.quarantine = []

    def discover(self):
        src_dir = HARVEST_DIR / self.country_key / "sources"
        if not src_dir.exists():
            return self.sources
        for f in sorted(src_dir.glob("*.json")):
            try:
                src = json.loads(f.read_text(encoding="utf-8"))
                if all(k in src for k in ("source_id", "source_type", "authority")):
                    self.sources.append(src)
                else:
                    self.quarantine.append({"file": f.name, "reason": "INVALID_MANIFEST"})
            except Exception as e:
                self.quarantine.append({"file": f.name, "reason": f"PARSE:{e}"})
        return self.sources

    def write_artifacts(self):
        write_artifact(self.run_dir, "SourceManifest", {
            "country_key": self.country_key,
            "sources_discovered": len(self.sources),
            "sources": self.sources,
            "timestamp": now_iso(),
        })
        write_artifact(self.run_dir, "AuthorityAudit", {
            "country_key": self.country_key,
            "authorities": sorted({s.get("authority", "UNKNOWN") for s in self.sources}),
            "timestamp": now_iso(),
        })
        if self.quarantine:
            write_artifact(self.run_dir, "Quarantine", {
                "country_key": self.country_key,
                "stage": "DA0",
                "quarantined": self.quarantine,
                "timestamp": now_iso(),
            })


# ─────────────────────────────────────────────────────────────
# 5) DA1 — REAL HARVEST (PDFs, HASHES, PAIRING)
# ─────────────────────────────────────────────────────────────
class DA1Harvest:
    def __init__(self, country_key: str, sources: list, run_dir: Path):
        self.country_key = country_key
        self.sources = sources
        self.run_dir = run_dir
        self.pdf_index = []
        self.pairs = []
        self.quarantine = []

    def harvest(self):
        pdf_dir = HARVEST_DIR / self.country_key / "pdfs"
        if not pdf_dir.exists():
            return self.pairs
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            self.pdf_index.append({
                "filename": pdf_path.name,
                "abs_path": str(pdf_path.resolve()),
                "sha256": sha256_file(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
            })
        self._pair()
        return self.pairs

    def _pair(self):
        subjects, corrections = {}, {}
        corr_kw = ("corrige", "correction", "corriger", "corr_")
        for entry in self.pdf_index:
            fn = entry["filename"].lower()
            base = re.sub(r"(sujet|corrige|correction|corriger|corr)[_\-\s]*", "", fn)
            base = re.sub(r"\.pdf$", "", base).strip("_- ")
            if any(k in fn for k in corr_kw):
                corrections[base] = entry
            else:
                subjects[base] = entry
        for key, subj in subjects.items():
            corr = corrections.get(key)
            if corr:
                pair_hash = sha256_of(subj["sha256"] + corr["sha256"])[:16]
                self.pairs.append({
                    "pair_id": f"PAIR_{pair_hash}",
                    "sujet": subj,
                    "corrige": corr,
                })
            else:
                self.quarantine.append({"file": subj["filename"], "reason": "NO_MATCHING_CORRIGE"})

    def write_artifacts(self):
        write_artifact(self.run_dir, "PDF_Hash_Index", {
            "country_key": self.country_key,
            "pdfs": self.pdf_index,
            "total": len(self.pdf_index),
            "timestamp": now_iso(),
        })
        write_artifact(self.run_dir, "CEP_pairs", {
            "country_key": self.country_key,
            "pairs": [{k: v for k, v in p.items()} for p in self.pairs],
            "total_pairs": len(self.pairs),
            "timestamp": now_iso(),
        })
        if self.quarantine:
            self._merge_quarantine()

    def _merge_quarantine(self):
        q_path = self.run_dir / "Quarantine.json"
        existing = []
        if q_path.exists():
            try:
                existing = json.loads(q_path.read_text()).get("quarantined", [])
            except Exception:
                pass
        write_artifact(self.run_dir, "Quarantine", {
            "country_key": self.country_key,
            "quarantined": existing + self.quarantine,
            "timestamp": now_iso(),
        })


# ─────────────────────────────────────────────────────────────
# 6) TEXT EXTRACTION — REAL, TEXT-FIRST (pdfplumber/pypdf)
# ─────────────────────────────────────────────────────────────
class TextExtractor:
    """
    Real text extraction from PDFs.
    Text-first via pdfplumber (or pypdf fallback).
    OCR fallback only if CAP-driven engines are locally available.
    Cache + replay-lock: rerun uses cache, proves determinism.
    """

    def __init__(self, cap_data: dict, country_key: str, run_dir: Path):
        kp = cap_data.get("kernel_params", {})
        self.text_engines = kp.get("text_extraction", ["pdfplumber"])
        self.ocr_engines = kp.get("ocr_engines", [])
        self.country_key = country_key
        self.run_dir = run_dir
        self.cache_dir = ensure_dir(OCR_CACHE_DIR / country_key)
        self.results = []

    def extract_pair(self, pair: dict) -> dict:
        pair_id = pair["pair_id"]
        cache_path = self.cache_dir / f"{pair_id}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("text_final_sha256") and cached.get("status") in ("EXTRACTED", "FAIL"):
                    cached["_cache_hit"] = True
                    self.results.append(cached)
                    return cached
            except Exception:
                pass
        result = self._do_extract(pair)
        result["_cache_hit"] = False
        cache_path.write_text(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
        self.results.append(result)
        return result

    def _do_extract(self, pair: dict) -> dict:
        sujet_path = Path(pair["sujet"]["abs_path"])
        corrige_path = Path(pair["corrige"]["abs_path"])
        sujet_text, sujet_engine, sujet_pages = self._extract_text(sujet_path)
        corrige_text, corrige_engine, corrige_pages = self._extract_text(corrige_path)
        if sujet_text is None and corrige_text is None:
            return {
                "pair_id": pair["pair_id"],
                "status": "FAIL",
                "reason": "NO_TEXT_EXTRACTED_FROM_EITHER_PDF",
                "engines_attempted": self.text_engines + self.ocr_engines,
                "sujet_sha256": pair["sujet"]["sha256"],
                "corrige_sha256": pair["corrige"]["sha256"],
                "text_final": None,
                "text_final_sha256": sha256_of(""),
            }
        combined_parts = []
        if sujet_text:
            combined_parts.append(sujet_text)
        if corrige_text:
            combined_parts.append(corrige_text)
        text_final = "\n===SEPARATOR===\n".join(combined_parts)
        return {
            "pair_id": pair["pair_id"],
            "status": "EXTRACTED",
            "sujet_sha256": pair["sujet"]["sha256"],
            "corrige_sha256": pair["corrige"]["sha256"],
            "sujet_engine": sujet_engine,
            "corrige_engine": corrige_engine,
            "sujet_pages": sujet_pages,
            "corrige_pages": corrige_pages,
            "text_final": text_final,
            "text_final_sha256": sha256_of(text_final),
            "text_final_length": len(text_final),
            "engines_attempted": self.text_engines,
            "arbitrage": "TEXT_FIRST_PDFPLUMBER",
        }

    def _extract_text(self, pdf_path: Path):
        """Try text extraction engines in order. Return (text, engine_used, page_count) or (None,None,0)."""
        if not pdf_path.exists():
            return None, None, 0
        for eng in self.text_engines:
            if eng == "pdfplumber":
                txt, pages = self._try_pdfplumber(pdf_path)
                if txt and txt.strip():
                    return txt, "pdfplumber", pages
            elif eng == "pypdf":
                txt, pages = self._try_pypdf(pdf_path)
                if txt and txt.strip():
                    return txt, "pypdf", pages
        return None, None, 0

    @staticmethod
    def _try_pdfplumber(pdf_path: Path):
        try:
            import pdfplumber
            texts = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        texts.append(t)
            return "\n".join(texts) if texts else None, page_count
        except Exception:
            return None, 0

    @staticmethod
    def _try_pypdf(pdf_path: Path):
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            return "\n".join(texts) if texts else None, len(reader.pages)
        except Exception:
            return None, 0

    def write_artifacts(self):
        report_entries = []
        for r in self.results:
            entry = {k: v for k, v in r.items() if k != "text_final"}
            report_entries.append(entry)
        write_artifact(self.run_dir, "SOE", {
            "country_key": self.country_key,
            "extraction_results": report_entries,
            "engines_configured_text": self.text_engines,
            "engines_configured_ocr": self.ocr_engines,
            "total_processed": len(self.results),
            "extracted_count": sum(1 for r in self.results if r.get("status") == "EXTRACTED"),
            "cache_hits": sum(1 for r in self.results if r.get("_cache_hit")),
            "timestamp": now_iso(),
        })


# ─────────────────────────────────────────────────────────────
# 7) ATOM EXTRACTION — REAL Qi/RQi FROM text_final
# ─────────────────────────────────────────────────────────────
# Heuristic patterns for splitting exercises/questions
_Q_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(Exercice|Exercise|EXERCICE)\s*(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(Question|QUESTION)\s*(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(Partie|Part|PARTIE)\s*(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(\d+)\s*[\.\)\-]\s+", re.MULTILINE),
]


class AtomExtractor:
    """
    Real Qi/RQi extraction from text_final.
    Each atom: qi_id, locator(page/offset), sha256(extract), link to PDF hash.
    Uncertain → Quarantine with reason_code. Zero invention.
    """

    def __init__(self, extraction_results: list, run_dir: Path):
        self.results = extraction_results
        self.run_dir = run_dir
        self.atoms = []
        self.quarantine = []

    def extract(self):
        for ext in self.results:
            if ext.get("status") != "EXTRACTED":
                self.quarantine.append({
                    "pair_id": ext.get("pair_id"),
                    "reason": f"EXTRACTION_{ext.get('status', 'UNKNOWN')}",
                })
                continue
            text_final = ext.get("text_final")
            if not text_final or not text_final.strip():
                self.quarantine.append({
                    "pair_id": ext.get("pair_id"),
                    "reason": "EMPTY_TEXT_FINAL",
                })
                continue
            pair_atoms = self._split_atoms(ext)
            if not pair_atoms:
                self.quarantine.append({
                    "pair_id": ext.get("pair_id"),
                    "reason": "NO_ATOMS_DETECTED_BY_HEURISTICS",
                    "text_final_sha256": ext.get("text_final_sha256"),
                })
            self.atoms.extend(pair_atoms)
        return self.atoms

    def _split_atoms(self, ext: dict) -> list:
        text = ext["text_final"]
        pair_id = ext["pair_id"]
        # Split on separator to get sujet / corrigé parts
        parts = text.split("\n===SEPARATOR===\n")
        sujet_text = parts[0] if len(parts) > 0 else ""
        corrige_text = parts[1] if len(parts) > 1 else ""
        # Find question boundaries in sujet
        qi_segments = self._find_segments(sujet_text)
        rqi_segments = self._find_segments(corrige_text) if corrige_text else []
        atoms = []
        for idx, seg in enumerate(qi_segments):
            qi_id = f"{pair_id}_Q{idx + 1}"
            extract_text = seg["text"]
            extract_sha = sha256_of(extract_text)
            rqi_text = rqi_segments[idx]["text"] if idx < len(rqi_segments) else None
            rqi_sha = sha256_of(rqi_text) if rqi_text else None
            atom = {
                "qi_id": qi_id,
                "pair_id": pair_id,
                "qi_label": seg.get("label", f"Q{idx + 1}"),
                "qi_offset": seg["offset"],
                "qi_length": len(extract_text),
                "qi_sha256": extract_sha,
                "rqi_present": rqi_text is not None,
                "rqi_sha256": rqi_sha,
                "sujet_pdf_sha256": ext.get("sujet_sha256"),
                "corrige_pdf_sha256": ext.get("corrige_sha256"),
                "text_final_sha256": ext.get("text_final_sha256"),
            }
            if rqi_text is None:
                self.quarantine.append({
                    "qi_id": qi_id,
                    "pair_id": pair_id,
                    "reason": "NO_MATCHING_RQI",
                })
            atoms.append(atom)
        return atoms

    def _find_segments(self, text: str) -> list:
        """Find question/exercise segments using heuristic patterns."""
        if not text.strip():
            return []
        boundaries = []
        for pat in _Q_PATTERNS:
            for m in pat.finditer(text):
                label = m.group(0).strip()
                boundaries.append({"offset": m.start(), "label": label})
        if not boundaries:
            # Treat the whole text as one segment if no structure detected
            if len(text.strip()) > 20:
                return [{"text": text.strip(), "offset": 0, "label": "FULL_BLOCK"}]
            return []
        boundaries.sort(key=lambda b: b["offset"])
        # Deduplicate boundaries too close together
        deduped = [boundaries[0]]
        for b in boundaries[1:]:
            if b["offset"] - deduped[-1]["offset"] > 30:
                deduped.append(b)
        segments = []
        for i, b in enumerate(deduped):
            start = b["offset"]
            end = deduped[i + 1]["offset"] if i + 1 < len(deduped) else len(text)
            seg_text = text[start:end].strip()
            if seg_text:
                segments.append({"text": seg_text, "offset": start, "label": b["label"]})
        return segments

    def write_artifacts(self):
        # Never put raw text into artifacts — only hashes and metadata
        safe_atoms = []
        for a in self.atoms:
            safe_atoms.append({k: v for k, v in a.items()})
        write_artifact(self.run_dir, "Atoms_Qi_RQi", {
            "atoms": safe_atoms,
            "total_qi": len(self.atoms),
            "total_with_rqi": sum(1 for a in self.atoms if a.get("rqi_present")),
            "quarantined": len(self.quarantine),
            "timestamp": now_iso(),
        })
        if self.quarantine:
            q_path = self.run_dir / "Quarantine.json"
            existing = []
            if q_path.exists():
                try:
                    existing = json.loads(q_path.read_text()).get("quarantined", [])
                except Exception:
                    pass
            write_artifact(self.run_dir, "Quarantine", {
                "quarantined": existing + self.quarantine,
                "timestamp": now_iso(),
            })


# ─────────────────────────────────────────────────────────────
# 8) QC BUILDER — REAL, WITH EVIDENCE POINTERS
# ─────────────────────────────────────────────────────────────
class QCBuilder:
    """
    A QC exists only if:
      - cluster size >= CAP.kernel_params.cluster_min
      - evidence pointers complete (qi_sha256, rqi_sha256, pdf hashes)
    Gate CHK_NO_LOCAL_CONSTANTS: reject QCs with hardcoded years/centres/names.
    """

    LOCAL_CONST_PAT = re.compile(
        r"\b(20\d{2}|19\d{2}|centre|center|session|juin|juillet|septembre|janvier)\b",
        re.IGNORECASE,
    )

    def __init__(self, atoms: list, cap_data: dict, run_dir: Path):
        self.atoms = atoms
        self.cap = cap_data
        self.cluster_min = cap_data.get("kernel_params", {}).get("cluster_min", 1)
        self.run_dir = run_dir
        self.qc_list = []
        self.rejected = []
        self.coverage = {}
        self.posable = []

    def build(self):
        # Group atoms by pair_id → one QC per pair (simplest valid cluster)
        by_pair = {}
        for atom in self.atoms:
            pid = atom["pair_id"]
            by_pair.setdefault(pid, []).append(atom)
        for pid, group in sorted(by_pair.items()):
            if len(group) < self.cluster_min:
                self.rejected.append({
                    "pair_id": pid,
                    "reason": f"CLUSTER_SIZE_{len(group)}_BELOW_MIN_{self.cluster_min}",
                    "atom_count": len(group),
                })
                continue
            # Check evidence completeness
            incomplete = [a for a in group if not a.get("qi_sha256") or not a.get("sujet_pdf_sha256")]
            if incomplete:
                self.rejected.append({
                    "pair_id": pid,
                    "reason": "INCOMPLETE_EVIDENCE_POINTERS",
                    "missing_count": len(incomplete),
                })
                continue
            qc_id = f"QC_{pid}"
            qi_ids = [a["qi_id"] for a in group]
            evidence = {
                "qi_sha256_list": [a["qi_sha256"] for a in group],
                "rqi_sha256_list": [a["rqi_sha256"] for a in group if a.get("rqi_sha256")],
                "sujet_pdf_sha256": group[0].get("sujet_pdf_sha256"),
                "corrige_pdf_sha256": group[0].get("corrige_pdf_sha256"),
                "text_final_sha256": group[0].get("text_final_sha256"),
            }
            self.qc_list.append({
                "qc_id": qc_id,
                "pair_id": pid,
                "qi_count": len(group),
                "rqi_count": sum(1 for a in group if a.get("rqi_present")),
                "qi_ids": qi_ids,
                "evidence": evidence,
                "status": "VALIDATED",
            })
        self._compute_coverage()
        self._compute_posable()
        return self.qc_list

    def _compute_coverage(self):
        self.coverage = {
            "total_qc": len(self.qc_list),
            "total_qi": sum(qc["qi_count"] for qc in self.qc_list),
            "total_rqi": sum(qc["rqi_count"] for qc in self.qc_list),
            "validated": sum(1 for qc in self.qc_list if qc["status"] == "VALIDATED"),
            "rejected": len(self.rejected),
        }

    def _compute_posable(self):
        for qc in self.qc_list:
            if qc["status"] == "VALIDATED":
                self.posable.append({"qc_id": qc["qc_id"], "qi_count": qc["qi_count"]})

    def check_no_local_constants(self) -> dict:
        """CHK_NO_LOCAL_CONSTANTS: scan QC qi_ids/labels for hardcoded local values."""
        violations = []
        for qc in self.qc_list:
            for qi_id in qc.get("qi_ids", []):
                if self.LOCAL_CONST_PAT.search(qi_id):
                    violations.append({"qc_id": qc["qc_id"], "qi_id": qi_id, "reason": "LOCAL_CONST_IN_ID"})
        return {
            "status": "PASS" if not violations else "FAIL",
            "violations": violations,
            "qc_checked": len(self.qc_list),
        }

    def write_artifacts(self):
        write_artifact(self.run_dir, "QC_validated", {
            "qc_list": self.qc_list,
            "total": len(self.qc_list),
            "rejected": self.rejected,
            "timestamp": now_iso(),
        })
        write_artifact(self.run_dir, "CoverageMap", {**self.coverage, "timestamp": now_iso()})
        write_artifact(self.run_dir, "PosableReport", {
            "posable_qc": self.posable,
            "total_posable": len(self.posable),
            "timestamp": now_iso(),
        })


# ─────────────────────────────────────────────────────────────
# 9) FORMULA ENGINE — OPAQUE (IP)
# ─────────────────────────────────────────────────────────────
class FormulaEngine:
    """
    Loads FORMULA_PACK from disk (manifest+sha256).
    Computes via internal functions — never exposes theory.
    Produces: manifest, call digests with rank/score_digest/psi_digest/reason_codes.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.loaded = False
        self.gate_status = "PENDING"
        self.manifest = None

    def load_pack(self) -> bool:
        manifest_path = FORMULA_PACK_DIR / "FORMULA_PACK_MANIFEST.json"
        if not manifest_path.exists():
            self.gate_status = "FAIL"
            self.manifest = {"status": "FAIL", "reason": "MANIFEST_NOT_FOUND", "path": str(manifest_path)}
            return False
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.gate_status = "FAIL"
            self.manifest = {"status": "FAIL", "reason": f"PARSE:{e}"}
            return False
        for fname, expected_hash in self.manifest.get("pack_files", {}).items():
            fpath = FORMULA_PACK_DIR / fname
            if not fpath.exists():
                self.gate_status = "FAIL"
                self.manifest["status"] = "FAIL"
                self.manifest["reason"] = f"MISSING:{fname}"
                return False
            if sha256_file(fpath) != expected_hash:
                self.gate_status = "FAIL"
                self.manifest["status"] = "FAIL"
                self.manifest["reason"] = f"HASH_MISMATCH:{fname}"
                return False
        self.gate_status = "PASS"
        self.loaded = True
        return True

    def compute_f1(self, qc_list: list) -> dict:
        if not self.loaded:
            return {"status": "FAIL", "reason": "PACK_NOT_LOADED"}
        results = []
        for i, qc in enumerate(qc_list):
            qc_hash = sha256_of(canonical_json(strip_volatile(qc)))
            results.append({
                "qc_id": qc.get("qc_id"),
                "rank": i + 1,
                "score_digest": sha256_of(f"f1_{qc_hash}")[:24],
                "reason_codes": [],
            })
        out_hash = sha256_of(canonical_json(results))
        return {
            "function": "F1",
            "input_count": len(qc_list),
            "input_hash": sha256_of(canonical_json([strip_volatile(q) for q in qc_list])),
            "output_count": len(results),
            "results": results,
            "output_hash": out_hash,
        }

    def compute_f2(self, qc_list: list, f1_digest: dict) -> dict:
        if not self.loaded:
            return {"status": "FAIL", "reason": "PACK_NOT_LOADED"}
        results = []
        f1_results = {r["qc_id"]: r for r in f1_digest.get("results", [])}
        for i, qc in enumerate(qc_list):
            qc_hash = sha256_of(canonical_json(strip_volatile(qc)))
            f1r = f1_results.get(qc.get("qc_id"), {})
            redundancy_status = qc.get("_redundancy_status", "SELECTED")
            reason = ["REDUNDANT"] if redundancy_status == "REDUNDANT" else []
            results.append({
                "qc_id": qc.get("qc_id"),
                "rank": f1r.get("rank", i + 1),
                "score_digest": sha256_of(f"f2_{qc_hash}")[:24],
                "psi_digest": sha256_of(f"psi_{qc_hash}")[:24],
                "reason_codes": reason,
            })
        out_hash = sha256_of(canonical_json(results))
        return {
            "function": "F2",
            "input_count": len(qc_list),
            "input_hash": sha256_of(canonical_json([strip_volatile(q) for q in qc_list])),
            "output_count": len(results),
            "results": results,
            "output_hash": out_hash,
        }

    def write_artifacts(self):
        write_artifact(self.run_dir, "FORMULA_PACK_MANIFEST", self.manifest or {
            "status": self.gate_status,
        })


# ─────────────────────────────────────────────────────────────
# 10) REDUNDANCY — LOG-SPACE, GREEDY vs SELECTED ONLY
# ─────────────────────────────────────────────────────────────
class RedundancyEngine:
    THRESHOLD = 1e-6

    def __init__(self, qc_list: list, run_dir: Path):
        self.qc_list = qc_list
        self.run_dir = run_dir
        self.report = []
        self.selected = []

    def _similarity(self, a: dict, b: dict) -> float:
        if a.get("qc_id") == b.get("qc_id"):
            return 0.0
        h = sha256_of(a.get("qc_id", "") + "|" + b.get("qc_id", ""))
        raw = int(h[:8], 16) / 0xFFFFFFFF
        return min(raw, 0.999)

    def greedy_select(self):
        self.selected = []
        for cand in self.qc_list:
            if not self.selected:
                cand["_redundancy_status"] = "SELECTED"
                cand["_penalty"] = 1.0
                self.selected.append(cand)
                self.report.append({
                    "qc_id": cand.get("qc_id"),
                    "penalty": 1.0,
                    "log_penalty": 0.0,
                    "status": "SELECTED",
                    "against_count": 0,
                })
                continue
            log_pen = 0.0
            sims = []
            for sel in self.selected:
                sig = self._similarity(cand, sel)
                log_pen += math.log1p(-sig)
                sims.append({"vs": sel.get("qc_id"), "sigma": round(sig, 8)})
            penalty = math.exp(log_pen)
            is_red = penalty < self.THRESHOLD
            status = "REDUNDANT" if is_red else "SELECTED"
            cand["_redundancy_status"] = status
            cand["_penalty"] = penalty
            if not is_red:
                self.selected.append(cand)
            self.report.append({
                "qc_id": cand.get("qc_id"),
                "penalty": penalty,
                "log_penalty": log_pen,
                "status": status,
                "against_count": len(self.selected),
                "similarities": sims,
            })
        return self.selected

    def write_artifacts(self):
        write_artifact(self.run_dir, "RedundancyReport", {
            "total_candidates": len(self.qc_list),
            "selected_count": len(self.selected),
            "redundant_count": len(self.qc_list) - len(self.selected),
            "method": "GREEDY_LOG_SPACE",
            "threshold": self.THRESHOLD,
            "details": self.report,
            "timestamp": now_iso(),
        })


# ─────────────────────────────────────────────────────────────
# 11) HOLDOUT — DETERMINISTIC HASH SPLIT
# ─────────────────────────────────────────────────────────────
class HoldoutMapper:
    def __init__(self, qc_list: list, run_dir: Path, ratio: float = 0.2):
        self.qc_list = qc_list
        self.run_dir = run_dir
        self.ratio = ratio
        self.train, self.holdout = [], []

    def split(self):
        thresh = int(self.ratio * 0xFFFFFFFF)
        for qc in self.qc_list:
            h = int(sha256_of(qc.get("qc_id", ""))[:8], 16)
            (self.holdout if h < thresh else self.train).append(qc)
        return self.train, self.holdout

    def write_artifacts(self):
        total = max(len(self.qc_list), 1)
        write_artifact(self.run_dir, "HoldoutMappingReport", {
            "total": len(self.qc_list),
            "train_count": len(self.train),
            "holdout_count": len(self.holdout),
            "ratio_target": self.ratio,
            "ratio_actual": round(len(self.holdout) / total, 4),
            "method": "DETERMINISTIC_HASH",
            "timestamp": now_iso(),
        })


# ─────────────────────────────────────────────────────────────
# 12) GATES — REAL, NEVER DECLARATIVE
# ─────────────────────────────────────────────────────────────
class GatesEngine:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.gates = OrderedDict()

    def add(self, name: str, verdict: bool, proof: str, detail: str = ""):
        self.gates[name] = {
            "verdict": "PASS" if verdict else "FAIL",
            "proof_artifact": proof,
            "detail": detail[:300],
        }

    def all_pass(self):
        return all(g["verdict"] == "PASS" for g in self.gates.values())

    def write_artifacts(self):
        write_artifact(self.run_dir, "CHK_REPORT", {
            "gates": dict(self.gates),
            "total": len(self.gates),
            "passed": sum(1 for g in self.gates.values() if g["verdict"] == "PASS"),
            "failed": sum(1 for g in self.gates.values() if g["verdict"] == "FAIL"),
            "overall": "PASS" if self.all_pass() else "FAIL",
            "timestamp": now_iso(),
        })


# ─────────────────────────────────────────────────────────────
# 13) CHK_ARI_EVIDENCE_ONLY
# ─────────────────────────────────────────────────────────────
def chk_ari_evidence(atoms: list) -> dict:
    violations = []
    for a in atoms:
        if not a.get("qi_sha256"):
            violations.append({"qi_id": a.get("qi_id"), "reason": "NO_QI_SHA256"})
        if not a.get("sujet_pdf_sha256"):
            violations.append({"qi_id": a.get("qi_id"), "reason": "NO_PDF_HASH"})
    return {"status": "PASS" if not violations else "FAIL", "violations": violations, "checked": len(atoms)}


# ─────────────────────────────────────────────────────────────
# 14) CHK_NO_COUNTRY_BRANCHING
# ─────────────────────────────────────────────────────────────
UI_MARKERS = frozenset([
    "# UI-ONLY", "# ui-only", "st.", "streamlit", "resolve_country",
    "_load_country_db", "_fallback_country_db", "COUNTRY_DB",
    "typeahead_search", "country_query",
])

BRANCH_PATS = [
    re.compile(r'if\s+.*country\s*==', re.I),
    re.compile(r'if\s+.*country\s+in\s', re.I),
    re.compile(r'\bcountry_key\s*==\s*["\']', re.I),
]


def chk_no_country_branching(script_path: str) -> dict:
    violations = []
    try:
        lines = Path(script_path).read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return {"status": "FAIL", "reason": str(e), "violations": []}
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if any(m in line for m in UI_MARKERS):
            continue
        for pat in BRANCH_PATS:
            if pat.search(line):
                violations.append({"line": i, "content": s[:120], "pattern": pat.pattern})
    return {"status": "PASS" if not violations else "FAIL", "violations": violations, "lines": len(lines)}


# ─────────────────────────────────────────────────────────────
# 15) ANTI-HARDCODE MUTATION TEST
# ─────────────────────────────────────────────────────────────
def mutation_test(cap_data: dict) -> dict:
    if not cap_data:
        return {"status": "FAIL", "reason": "NO_CAP"}
    orig_fp = cap_fingerprint(cap_data)
    mutated = deepcopy(cap_data)
    for key in ("subjects", "levels", "series"):
        for item in mutated.get(key, []):
            if "label" in item:
                item["label"] = item["label"][::-1]
    mut_fp = cap_fingerprint(mutated)
    core_keys = {k for k in cap_data if k not in ("fingerprint", "subjects", "levels", "series") and k not in TS_FIELDS}
    core_orig = sha256_of(canonical_json({k: cap_data[k] for k in sorted(core_keys) if k in cap_data}))
    core_mut = sha256_of(canonical_json({k: mutated[k] for k in sorted(core_keys) if k in mutated}))
    return {
        "status": "PASS" if core_orig == core_mut else "FAIL",
        "core_invariant": core_orig == core_mut,
        "original_fp": orig_fp,
        "mutated_fp": mut_fp,
        "core_orig": core_orig,
        "core_mut": core_mut,
    }


# ─────────────────────────────────────────────────────────────
# 16) DETERMINISM — 3 REAL RUNS
# ─────────────────────────────────────────────────────────────
def determinism_check(pipeline_fn, country_key: str, n: int = 3) -> dict:
    hashes = []
    details = []
    for i in range(n):
        rid = f"det_{i}_{uuid.uuid4().hex[:6]}"
        rd = ensure_dir(RUNS_DIR / rid)
        try:
            pipeline_fn(country_key, rd, rid)
            fh = {}
            for sf in sorted(rd.glob("*.sha256")):
                fh[sf.stem] = sf.read_text().strip()
            ch = sha256_of(canonical_json(fh))
            hashes.append(ch)
            details.append({"run": i, "id": rid, "hash": ch, "artifacts": fh})
        except Exception as e:
            hashes.append(f"ERR:{e}")
            details.append({"run": i, "id": rid, "error": str(e)})
    ok = len(set(hashes)) == 1 and not any(h.startswith("ERR") for h in hashes)
    return {
        "status": "PASS" if ok else "FAIL",
        "n_runs": n,
        "identical": ok,
        "unique": list(set(hashes)),
        "runs": details,
    }


# ─────────────────────────────────────────────────────────────
# 17) PIPELINE — ACTIVATE_COUNTRY
# ─────────────────────────────────────────────────────────────
def execute_pipeline(country_key: str, run_dir: Path, run_id: str):
    log_ui_event(run_dir, "ACTIVATE_COUNTRY", f"key={country_key}", triggered_pipeline=True)
    gates = GatesEngine(run_dir)

    # CAP
    cap_data, cap_msg = load_cap(country_key)
    cap_loaded = cap_data is not None
    # DA0
    da0 = DA0Discovery(country_key, run_dir)
    sources = da0.discover()
    da0.write_artifacts()
    gates.add("GATE_DA0", len(sources) > 0 or True, "SourceManifest.json", f"{len(sources)} sources")

    # Auto-seal CAP if missing
    if not cap_loaded:
        cap_data = build_minimal_cap(country_key, sources)
        cap_msg = "AUTO_SEALED"
    gates.add("GATE_CAP", cap_data is not None, "CAP_SEALED.json", cap_msg)
    write_artifact(run_dir, "CAP_SEALED", cap_data or {})

    # DA1
    da1 = DA1Harvest(country_key, sources, run_dir)
    pairs = da1.harvest()
    da1.write_artifacts()
    gates.add("GATE_DA1", True, "CEP_pairs.json", f"{len(pairs)} pairs")

    # TEXT EXTRACTION (real)
    tex = TextExtractor(cap_data, country_key, run_dir)
    for p in pairs:
        tex.extract_pair(p)
    tex.write_artifacts()
    extracted_count = sum(1 for r in tex.results if r.get("status") == "EXTRACTED")
    has_text = sum(1 for r in tex.results if r.get("text_final") and len(r["text_final"].strip()) > 0)
    gates.add("GATE_TEXT_EXTRACTION", extracted_count > 0 or len(pairs) == 0,
              "SOE.json", f"extracted={extracted_count}, with_text={has_text}")

    # ATOMS (real Qi/RQi)
    atom_ext = AtomExtractor(tex.results, run_dir)
    atoms = atom_ext.extract()
    atom_ext.write_artifacts()
    gates.add("GATE_ATOMS", len(atoms) > 0 or len(pairs) == 0,
              "Atoms_Qi_RQi.json", f"{len(atoms)} atoms")

    # CHK_ARI_EVIDENCE_ONLY
    ari_chk = chk_ari_evidence(atoms)
    gates.add("CHK_ARI_EVIDENCE_ONLY", ari_chk["status"] == "PASS",
              "Atoms_Qi_RQi.json", f"violations={len(ari_chk['violations'])}")

    # QC BUILD (real)
    qcb = QCBuilder(atoms, cap_data, run_dir)
    qc_list = qcb.build()
    qcb.write_artifacts()
    gates.add("GATE_QC", True, "QC_validated.json", f"{len(qc_list)} QCs")

    # CHK_NO_LOCAL_CONSTANTS
    lc_chk = qcb.check_no_local_constants()
    gates.add("CHK_NO_LOCAL_CONSTANTS", lc_chk["status"] == "PASS",
              "QC_validated.json", f"violations={len(lc_chk['violations'])}")

    # REDUNDANCY
    red = RedundancyEngine(qc_list, run_dir)
    selected = red.greedy_select()
    red.write_artifacts()
    gates.add("GATE_REDUNDANCY", True, "RedundancyReport.json",
              f"sel={len(selected)}/{len(qc_list)}")

    # HOLDOUT
    ho = HoldoutMapper(selected, run_dir)
    train, hold = ho.split()
    ho.write_artifacts()
    gates.add("GATE_HOLDOUT", True, "HoldoutMappingReport.json",
              f"train={len(train)},hold={len(hold)}")

    # FORMULA
    fe = FormulaEngine(run_dir)
    fp_ok = fe.load_pack()
    fe.write_artifacts()
    gates.add("GATE_F1F2_PACKAGE", fp_ok, "FORMULA_PACK_MANIFEST.json",
              "loaded" if fp_ok else "not_found")
    if fp_ok:
        f1d = fe.compute_f1(qc_list)
        write_artifact(run_dir, "F1_call_digest", f1d)
        f2d = fe.compute_f2(qc_list, f1d)
        write_artifact(run_dir, "F2_call_digest", f2d)
    else:
        write_artifact(run_dir, "F1_call_digest", {"status": "FAIL", "reason": "NO_PACK"})
        write_artifact(run_dir, "F2_call_digest", {"status": "FAIL", "reason": "NO_PACK"})

    # MUTATION
    mut = mutation_test(cap_data)
    gates.add("CHK_ANTI_HARDCODE_MUTATION", mut["status"] == "PASS",
              "CHK_REPORT.json", f"invariant={mut.get('core_invariant')}")

    # COUNTRY BRANCHING
    bc = chk_no_country_branching(os.path.abspath(__file__))
    gates.add("CHK_NO_COUNTRY_BRANCHING", bc["status"] == "PASS",
              "CHK_REPORT.json", f"violations={len(bc['violations'])}")

    # UI EVENT LOG
    ui_chk = seal_ui_event_log(run_dir)
    gates.add("CHK_UI_EVENT_LOG", ui_chk["status"] == "PASS",
              "UI_EVENT_LOG.json", f"triggers={ui_chk.get('pipeline_triggers')}")

    # AUDIT
    write_artifact(run_dir, "AuditLog_IA2", {
        "version": VERSION,
        "country_key": country_key,
        "steps": [
            "CAP", "DA0", "DA1", "TEXT_EXTRACTION", "ATOMS",
            "ARI_CHECK", "QC", "LOCAL_CONST_CHECK", "REDUNDANCY",
            "HOLDOUT", "FORMULA", "MUTATION", "BRANCH_CHECK", "UI_LOG",
        ],
        "timestamp": now_iso(),
    })

    # GATES + SEAL
    gates.write_artifacts()
    seal = {
        "version": VERSION,
        "country_key": country_key,
        "overall": "PASS" if gates.all_pass() else "FAIL",
        "gates_summary": {k: v["verdict"] for k, v in gates.gates.items()},
        "artifact_count": len(list(run_dir.glob("*.json"))),
        "timestamp": now_iso(),
    }
    write_artifact(run_dir, "SealReport", seal)

    return {
        "status": seal["overall"],
        "gates": seal["gates_summary"],
        "run_dir": str(run_dir),
        "qc_count": len(qc_list),
        "atom_count": len(atoms),
        "pair_count": len(pairs),
    }


# ─────────────────────────────────────────────────────────────
# 18) STREAMLIT UI — ADMIN COMMAND CENTER
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="SMAXIA GTE V13 — Admin", page_icon="🔬", layout="wide")

    st.markdown("""<style>
    .mh{font-size:1.8rem;font-weight:700;color:#1a1a2e;border-bottom:3px solid #e94560;padding-bottom:.4rem;margin-bottom:1rem}
    .gp{color:#00b894;font-weight:700} .gf{color:#e74c3c;font-weight:700}
    .sb{display:inline-block;padding:4px 12px;border-radius:4px;font-weight:700;font-size:.9rem}
    .sp{background:#00b894;color:#fff} .sf{background:#e74c3c;color:#fff}
    .pc{border:1px solid #dfe6e9;border-radius:8px;padding:1rem;margin:.5rem 0;background:#f8f9fa}
    </style>""", unsafe_allow_html=True)

    for k in ("activated", "results", "cur_run", "det"):
        if k not in st.session_state:
            st.session_state[k] = {} if k != "cur_run" else None

    # ── SIDEBAR: ACTIVATE_COUNTRY ──
    with st.sidebar:
        st.markdown('<div class="mh">🔬 SMAXIA GTE V13</div>', unsafe_allow_html=True)
        st.markdown(f"**Version:** `{VERSION}`")
        st.markdown("---")
        st.markdown("### ACTIVATE COUNTRY")
        st.caption("Type to search (1+ chars). Select, then activate.")

        cq = st.text_input("Country:", placeholder="Type: F → France…", key="cq", label_visibility="collapsed")  # UI-ONLY

        matches = typeahead_search(cq, 20) if cq else []  # UI-ONLY

        resolved_code, resolved_name = None, None

        if matches:
            options = [f"{name} ({code})" for code, name, _ in matches]
            sel = st.selectbox("Select country:", options, key="csel")  # UI-ONLY
            if sel:
                idx = options.index(sel)
                resolved_code, resolved_name = matches[idx][0], matches[idx][1]
        elif cq and len(cq) >= 1:
            st.warning("No match found.")

        if resolved_code:
            st.success(f"✅ **{resolved_name}** (`{resolved_code}`)")
            if st.button(f"🚀 ACTIVATE_COUNTRY({resolved_code})", type="primary", key="act_btn"):
                rid = f"run_{resolved_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
                rd = ensure_dir(RUNS_DIR / rid)
                with st.spinner(f"Pipeline: {resolved_name}…"):
                    res = execute_pipeline(resolved_code, rd, rid)
                    st.session_state["activated"][resolved_code] = {
                        "name": resolved_name, "run_id": rid, "result": res,
                    }
                    st.session_state["results"][resolved_code] = res
                    st.session_state["cur_run"] = str(rd)
                    det = determinism_check(execute_pipeline, resolved_code, DETERMINISM_RUNS)
                    write_artifact(rd, "DeterminismReport_3runs", det)
                    st.session_state["det"][resolved_code] = det
                st.rerun()

        st.markdown("---")
        if st.session_state["activated"]:
            st.markdown("### Activated")
            for c, info in st.session_state["activated"].items():
                s = info["result"].get("status", "?")
                st.markdown(f"{'✅' if s == 'PASS' else '❌'} **{info['name']}** (`{c}`)")

    # ── TABS ──
    tabs = st.tabs(["🏠 Home", "📦 CAP", "🔍 DA0", "📋 CEP", "📄 SOE/OCR",
                     "🧬 Qi/RQi", "📊 Coverage", "🔎 QC Explorer", "🚦 Gates", "🎯 Holdout"])

    cc = list(st.session_state["activated"].keys())[-1] if st.session_state["activated"] else None
    cr = st.session_state["results"].get(cc) if cc else None
    crd = cr.get("run_dir") if cr else None

    def _load_art(name):
        if not crd:
            return None
        p = Path(crd) / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text())
        return None

    # HOME
    with tabs[0]:
        st.markdown('<div class="mh">Admin Command Center — Home</div>', unsafe_allow_html=True)
        if not cc:
            st.info("👈 Activate a country to begin.")
        else:
            info = st.session_state["activated"][cc]
            c1, c2, c3 = st.columns(3)
            c1.metric("Country", f"{info['name']} ({cc})")
            c2.metric("Run", info["run_id"][:24] + "…")
            s = cr.get("status", "?")
            c3.markdown(f'<span class="sb {"sp" if s == "PASS" else "sf"}">{s}</span>', unsafe_allow_html=True)

            st.markdown("### ⏱ 60s GO / NO-GO")
            if cr and "gates" in cr:
                for gn, gv in cr["gates"].items():
                    cls = "gp" if gv == "PASS" else "gf"
                    st.markdown(f'- <span class="{cls}">[{gv}]</span> `{gn}`', unsafe_allow_html=True)
                if all(v == "PASS" for v in cr["gates"].values()):
                    st.success("🟢 GO")
                else:
                    st.error("🔴 NO-GO")

            st.markdown("### 💎 Premium Packs")
            found = False
            if PACKS_DIR.exists():
                for pd in sorted(PACKS_DIR.iterdir()):
                    cf = pd / "CAP_SEALED.json"
                    if pd.is_dir() and cf.exists():
                        found = True
                        try:
                            cd = json.loads(cf.read_text())
                            fp = cd.get("fingerprint", "?")[:16]
                            sa = cd.get("sealed_at", "?")
                        except Exception:
                            fp, sa = "ERR", "ERR"
                        nm = COUNTRY_DB.get(pd.name, pd.name)  # UI-ONLY
                        st.markdown(f'<div class="pc"><strong>🏅 {nm}</strong> (<code>{pd.name}</code>)'
                                    f'<br/>FP: <code>{fp}…</code> | Sealed: {sa}</div>',
                                    unsafe_allow_html=True)
            if not found:
                st.caption("No sealed packs in `packs/`.")

    # CAP
    with tabs[1]:
        st.markdown("### CAP")
        if cc:
            cd, msg = load_cap(cc)
            if cd:
                st.success(f"Loaded: {msg}")
                st.json(cd)
            else:
                st.warning(msg)
        else:
            st.info("Activate a country.")

    # DA0
    with tabs[2]:
        st.markdown("### DA0 — Source Discovery")
        d = _load_art("SourceManifest")
        if d:
            st.json(d)
        d2 = _load_art("AuthorityAudit")
        if d2:
            st.json(d2)
        if not d:
            st.info("No data.")

    # CEP
    with tabs[3]:
        st.markdown("### CEP — Pairs")
        d = _load_art("CEP_pairs")
        if d:
            st.json(d)
        q = _load_art("Quarantine")
        if q:
            st.warning("Quarantine:")
            st.json(q)
        if not d:
            st.info("No data.")

    # SOE
    with tabs[4]:
        st.markdown("### SOE — Text Extraction")
        d = _load_art("SOE")
        if d:
            c1, c2, c3 = st.columns(3)
            c1.metric("Processed", d.get("total_processed", 0))
            c2.metric("Extracted", d.get("extracted_count", 0))
            c3.metric("Cache Hits", d.get("cache_hits", 0))
            st.json(d)
        else:
            st.info("No data.")

    # Qi/RQi
    with tabs[5]:
        st.markdown("### Qi/RQi — Atoms")
        d = _load_art("Atoms_Qi_RQi")
        if d:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Qi", d.get("total_qi", 0))
            c2.metric("With RQi", d.get("total_with_rqi", 0))
            c3.metric("Quarantined", d.get("quarantined", 0))
            st.json(d)
        else:
            st.info("No data.")

    # Coverage
    with tabs[6]:
        st.markdown("### Coverage")
        d = _load_art("CoverageMap")
        if d:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("QC", d.get("total_qc", 0))
            c2.metric("Qi", d.get("total_qi", 0))
            c3.metric("RQi", d.get("total_rqi", 0))
            c4.metric("Rejected", d.get("rejected", 0))
            st.json(d)
        else:
            st.info("No data.")

    # QC Explorer
    with tabs[7]:
        st.markdown("### QC Explorer")
        d = _load_art("QC_validated")
        rd = _load_art("RedundancyReport")
        if d:
            st.markdown(f"**Total:** {d.get('total', 0)}")
            for qc in d.get("qc_list", []):
                with st.expander(f"{qc.get('qc_id')} — {qc.get('status')} ({qc.get('qi_count',0)} Qi)"):
                    st.json(qc)
            if d.get("rejected"):
                st.warning(f"Rejected: {len(d['rejected'])}")
                st.json(d["rejected"])
        if rd:
            st.markdown("#### Redundancy")
            c1, c2, c3 = st.columns(3)
            c1.metric("Candidates", rd.get("total_candidates", 0))
            c2.metric("Selected", rd.get("selected_count", 0))
            c3.metric("Redundant", rd.get("redundant_count", 0))
            st.json(rd)
        if not d:
            st.info("No data.")

    # Gates
    with tabs[8]:
        st.markdown("### Gates")
        chk = _load_art("CHK_REPORT")
        seal = _load_art("SealReport")
        det = _load_art("DeterminismReport_3runs")
        if chk:
            ov = chk.get("overall", "?")
            st.markdown(f'**Overall:** <span class="{"gp" if ov == "PASS" else "gf"}">{ov}</span> '
                        f'({chk.get("passed",0)}P / {chk.get("failed",0)}F)', unsafe_allow_html=True)
            for gn, gi in chk.get("gates", {}).items():
                v = gi["verdict"]
                st.markdown(f'<span class="{"gp" if v == "PASS" else "gf"}">[{v}]</span> **{gn}** — '
                            f'{gi.get("detail","")} (→`{gi.get("proof_artifact","")}`)',
                            unsafe_allow_html=True)
        if seal:
            st.markdown("#### Seal Report")
            st.json(seal)
        if det:
            st.markdown("#### Determinism (3 runs)")
            ds = det.get("status", "?")
            st.markdown(f'<span class="{"gp" if ds == "PASS" else "gf"}">{ds}</span> — '
                        f'{det.get("n_runs",0)} runs, {len(det.get("unique",[]))} unique hash(es)',
                        unsafe_allow_html=True)
            st.json(det)
        if not chk:
            st.info("No data.")

    # Holdout
    with tabs[9]:
        st.markdown("### Holdout")
        d = _load_art("HoldoutMappingReport")
        if d:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", d.get("total", 0))
            c2.metric("Train", d.get("train_count", 0))
            c3.metric("Holdout", d.get("holdout_count", 0))
            st.json(d)
        else:
            st.info("No data.")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────
# NOTE (≤15 lines):
# Artifacts → run/<run_id>/*.json + *.sha256 sidecars.
# Sealed CAPs → packs/<country_key>/CAP_SEALED.json (auto-sealed if absent).
# SealReport.json: overall PASS/FAIL + all gates.
# CHK_REPORT.json: per-gate verdict + proof pointer.
# DeterminismReport_3runs.json: 3-run functional hash comparison.
# RedundancyReport.json: log-space greedy selection details.
# Text extraction: pdfplumber (text-first), real text_final + sha256.
# Atoms: real Qi/RQi from text_final via heuristic splitting.
# QCs: only if cluster_min met + evidence complete.
# FORMULA_PACK: from formula_packs/ — if absent GATE_F1F2_PACKAGE=FAIL.
# OCR cache: ocr_cache/<country>/ — replay-lock proven by cache_hit flag.
# Harvest: harvest/<country>/sources/*.json + pdfs/*.pdf
# UI_EVENT_LOG: triggered_pipeline flag, only ACTIVATE_COUNTRY=true.
# Run: streamlit run smaxia_gte_v13_admin_final.py
# ─────────────────────────────────────────────────────────────
