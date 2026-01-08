# =============================================================================
# SMAXIA GTE Console V31.10.24 (ISO-PROD — TEST=PROD)
# =============================================================================
# OBJECTIF (NON NÉGOCIABLE)
# - 100% Kernel V10.6.2: AUCUNE ré-implémentation de F1–F8 / IA1 / IA2 / mapping intent_code→chapter_code.
# - ISO-PROD: CAP (P3) + Loading Sujet+Corrigé (P5) exécutés automatiquement via prompts scellés (verbatim).
# - UI non régressive vs V31.10.22: même logique d’usage (Activation pays -> Sélection -> RUN -> QC par chapitre -> Explorer -> Export -> Verdict).
# - Verdict binaire central: EXISTE-T-IL une Qi posable qui ne converge vers AUCUNE QC ?
#   OUI => FAIL ; NON => SEALED
#
# EXÉCUTION
#   streamlit run smaxia_gte_v31_10_24_iso_prod_ui.py
#
# DÉPENDANCES
#   pip install streamlit requests pandas beautifulsoup4 python-docx
#
# CONFIG LLM (obligatoire pour ISO-PROD)
#   SMAXIA_LLM_PROVIDER = OPENAI | GEMINI | CUSTOM
#   OPENAI:
#     OPENAI_API_KEY, OPENAI_MODEL (ex: gpt-4.1, gpt-4.1-mini, etc.)
#   GEMINI:
#     GOOGLE_API_KEY, GEMINI_MODEL (ex: gemini-1.5-pro, etc.)
#   CUSTOM:
#     SMAXIA_LLM_EXECUTOR = "module:function"  (callable(prompt_text:str, variables:dict) -> dict)
#
# PROMPTS SCELLÉS (AUTO-LOAD, ZÉRO ACTION MANUELLE)
# - P3: Academic Pack
# - P5: Harvester Sentinel (pairs Sujet/Corrigé)
# Le script tente automatiquement de les charger depuis plusieurs emplacements.
# =============================================================================

from __future__ import annotations

import json
import os
import re
import time
import hashlib
import importlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional (docx prompts)
try:
    from docx import Document  # python-docx
except Exception:
    Document = None


# =============================================================================
# UI / APP CONFIG
# =============================================================================

APP_TITLE = "SMAXIA GTE — Audit QC/ARI/FRT (ISO-PROD)"
APP_VERSION = "V31.10.24"

PIPELINE_PHASES = [
    "PHASE_01_ACTIVATION_CAP",
    "PHASE_02_CATALOG_UI",
    "PHASE_03_HARVEST_PAIRS",
    "PHASE_04_SNAPSHOT_URLS",
    "PHASE_05_KERNEL_RUN",
    "PHASE_06_QC_BY_CHAPTER",
    "PHASE_07_EXPLORER",
    "PHASE_08_AUDIT_ORPHELINS",
    "PHASE_09_EXPORT_PROOF",
    "PHASE_10_VERDICT",
]

USER_AGENT = "SMAXIA-GTE/ISO-PROD"
REQ_TIMEOUT = (6.0, 30.0)

RUN_DIR = Path("./_gte_runs")
RUN_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA CONTRACTS (ISO)
# =============================================================================

@dataclass
class PromptBundle:
    p3_text: str
    p5_text: str
    p3_source: str
    p5_source: str

@dataclass
class PairSnapshot:
    sujet_url: str
    corrige_url: str
    sujet_url_sha256: str
    corrige_url_sha256: str
    evidence: Dict[str, Any]

@dataclass
class ProofExport:
    timestamp: str
    app_version: str
    mode: str
    inputs: Dict[str, Any]
    prompts_sources: Dict[str, str]
    cap_sha256: str
    cap: Dict[str, Any]
    pairs: List[Dict[str, Any]]
    kernel_import_ok: bool
    kernel_module: str
    kernel_result_keys: List[str]
    verdict: str
    audit: Dict[str, Any]


# =============================================================================
# Deterministic utils
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def is_pdf_url(u: str) -> bool:
    return u.lower().split("?")[0].endswith(".pdf")

def url_domain(u: str) -> str:
    from urllib.parse import urlparse
    return urlparse(u).netloc.lower().split(":")[0]

def domain_allowed(u: str, allowed_domains: List[str]) -> bool:
    d = url_domain(u)
    allowed = [a.lower().strip() for a in allowed_domains if a and isinstance(a, str)]
    return any(d == a or d.endswith("." + a) for a in allowed)


# =============================================================================
# Prompt auto-loader (verbatim)
# =============================================================================

CANDIDATE_P3 = [
    "PROMPTS_SCELLES/P3_ACADEMIC_PACK.txt",
    "PROMPTS_SCELLES/P3_ACADEMIC_PACK.md",
    "PROMPTS_SCELLES/P3_ACADEMIC_PACK.docx",
    "PROMPT P3 — ACADEMIC PACK.docx",
    "P3_ACADEMIC_PACK.docx",
    "P3_ACADEMIC_PACK.md",
    "P3_ACADEMIC_PACK.txt",
]

CANDIDATE_P5 = [
    "PROMPTS_SCELLES/P5_HARVESTER.txt",
    "PROMPTS_SCELLES/P5_HARVESTER.md",
    "PROMPTS_SCELLES/P5_HARVESTER.docx",
    "P5_HARVESTER_V2.2_SENTINEL.md",
    "P5_HARVESTER.md",
    "P5_HARVESTER.txt",
]

def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def _read_docx_file(path: Path) -> str:
    if Document is None:
        raise RuntimeError("python-docx non disponible, impossible de lire le prompt .docx")
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").rstrip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()

def _find_file(candidates: List[str]) -> Tuple[Optional[Path], Optional[str]]:
    """
    Recherche dans:
      - dossier courant
      - dossier du script (si connu)
      - /mnt/data (utile en sandbox)
    """
    roots = []
    try:
        roots.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    roots.append(Path.cwd())
    roots.append(Path("/mnt/data"))

    for root in roots:
        for rel in candidates:
            p = (root / rel).resolve()
            if p.exists() and p.is_file():
                return p, str(p)
    return None, None

def load_prompts_scelles() -> PromptBundle:
    p3_path, p3_src = _find_file(CANDIDATE_P3)
    p5_path, p5_src = _find_file(CANDIDATE_P5)

    if not p3_path:
        raise FileNotFoundError(
            "PROMPT P3 introuvable. Cherché: " + ", ".join(CANDIDATE_P3)
        )
    if not p5_path:
        raise FileNotFoundError(
            "PROMPT P5 introuvable. Cherché: " + ", ".join(CANDIDATE_P5)
        )

    def read_any(p: Path) -> str:
        if p.suffix.lower() == ".docx":
            return _read_docx_file(p)
        return _read_text_file(p)

    p3_text = read_any(p3_path)
    p5_text = read_any(p5_path)

    if len(p3_text) < 500:
        raise ValueError("PROMPT P3 semble vide ou tronqué (taille < 500 chars).")
    if len(p5_text) < 500:
        raise ValueError("PROMPT P5 semble vide ou tronqué (taille < 500 chars).")

    return PromptBundle(
        p3_text=p3_text,
        p5_text=p5_text,
        p3_source=p3_src or str(p3_path),
        p5_source=p5_src or str(p5_path),
    )


# =============================================================================
# LLM Strict JSON Executor (ISO)
# =============================================================================

def _llm_provider() -> str:
    return (os.getenv("SMAXIA_LLM_PROVIDER") or "OPENAI").strip().upper()

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Variable d'environnement manquante: {name}")
    return v

def _coerce_json_object(text: str) -> Dict[str, Any]:
    """
    Extrait le premier objet JSON dans la réponse (strict ISO: pas de prose).
    Tolère fences ```json.
    """
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()

    # Attempt direct parse
    try:
        obj = json.loads(t)
        if not isinstance(obj, dict):
            raise ValueError("Réponse JSON non-dict")
        return obj
    except Exception:
        pass

    # Fallback: find {...}
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError("Aucun objet JSON détecté dans la réponse.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Réponse JSON non-dict")
    return obj

@st.cache_data(ttl=900)
def llm_json_call_strict(prompt_text: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    provider = _llm_provider()

    # Compose ISO message: prompt verbatim + variables appended as strict JSON context
    payload_context = stable_json({"variables": variables})
    iso_input = (
        prompt_text.rstrip()
        + "\n\n"
        + "=== CONTEXTE (JSON STRICT) ===\n"
        + payload_context
        + "\n\n"
        + "=== RÉPONSE ATTENDUE ===\n"
        + "Retourne UNIQUEMENT un objet JSON valide (sans prose, sans markdown)."
    )

    if provider == "CUSTOM":
        spec = _require_env("SMAXIA_LLM_EXECUTOR")  # module:function
        if ":" not in spec:
            raise RuntimeError("SMAXIA_LLM_EXECUTOR doit être 'module:function'")
        mod_name, fn_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name, None)
        if not callable(fn):
            raise RuntimeError("SMAXIA_LLM_EXECUTOR function introuvable ou non callable.")
        out = fn(iso_input, variables)
        if not isinstance(out, dict):
            raise RuntimeError("CUSTOM executor doit retourner un dict.")
        return out

    if provider == "OPENAI":
        api_key = _require_env("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "input": iso_input,
            "temperature": 0,
        }
        r = requests.post(url, headers=headers, json=body, timeout=REQ_TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")
        data = r.json()

        # Extract text from responses API
        # Typical: output[0].content[0].text
        text = ""
        try:
            out = data.get("output", [])
            if out and isinstance(out, list):
                content = out[0].get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text", "") or ""
        except Exception:
            text = ""

        if not text:
            # fallback: any string
            text = json.dumps(data)

        return _coerce_json_object(text)

    if provider == "GEMINI":
        api_key = _require_env("GOOGLE_API_KEY")
        model = os.getenv("GEMINI_MODEL") or "gemini-1.5-pro"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"parts": [{"text": iso_input}]}],
            "generationConfig": {"temperature": 0},
        }
        r = requests.post(url, json=body, timeout=REQ_TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:500]}")
        data = r.json()
        text = ""
        try:
            cand = data.get("candidates", [])
            if cand and isinstance(cand, list):
                parts = cand[0].get("content", {}).get("parts", [])
                if parts and isinstance(parts, list):
                    text = parts[0].get("text", "") or ""
        except Exception:
            text = ""

        if not text:
            text = json.dumps(data)

        return _coerce_json_object(text)

    raise RuntimeError(f"SMAXIA_LLM_PROVIDER non supporté: {provider}")


# =============================================================================
# CAP helpers (robust extraction, no hardcode métier)
# =============================================================================

def cap_hash(cap: Dict[str, Any]) -> str:
    return sha256_text(stable_json(cap))

def cap_get_allowed_domains(cap: Dict[str, Any]) -> List[str]:
    # Try common locations (P3 schema varies)
    candidates = [
        ("allowed_domains",),
        ("allowlist_domains",),
        ("policy", "domains"),
        ("harvest_policy", "domains"),
        ("policy", "harvest", "domains"),
        ("harvest_policy", "harvest", "domains"),
        ("academic_system", "allowed_domains"),
    ]
    for path in candidates:
        cur = cap
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, list) and cur:
            return [str(x).strip().lower() for x in cur if str(x).strip()]
    return []

def cap_extract_levels_subjects(cap: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extract display labels from P3 schema example:
      cap["levels"] list with labels.localized.fr
      cap["subjects"] list with labels.localized.fr
    """
    levels, subjects = [], []

    lv = cap.get("levels")
    if isinstance(lv, list):
        for item in lv:
            if not isinstance(item, dict):
                continue
            label = None
            labels = item.get("labels", {})
            if isinstance(labels, dict):
                loc = labels.get("localized", {})
                if isinstance(loc, dict):
                    label = loc.get("fr") or loc.get("en") or labels.get("default")
                else:
                    label = labels.get("default")
            label = label or item.get("level_invariant") or item.get("name")
            if label:
                levels.append(str(label))
    sb = cap.get("subjects")
    if isinstance(sb, list):
        for item in sb:
            if not isinstance(item, dict):
                continue
            label = None
            labels = item.get("labels", {})
            if isinstance(labels, dict):
                loc = labels.get("localized", {})
                if isinstance(loc, dict):
                    label = loc.get("fr") or loc.get("en") or labels.get("default")
                else:
                    label = labels.get("default")
            label = label or item.get("subject_invariant") or item.get("name")
            if label:
                subjects.append(str(label))

    # Fallback: if embedded in academic_system
    if not levels and isinstance(cap.get("academic_system"), dict):
        lv = cap["academic_system"].get("levels")
        if isinstance(lv, list):
            levels = [str(x.get("name") or x.get("level_invariant") or "").strip() for x in lv if isinstance(x, dict)]
            levels = [x for x in levels if x]

    # ensure deterministic unique
    levels = sorted(set([x for x in levels if x.strip()]))
    subjects = sorted(set([x for x in subjects if x.strip()]))

    return levels, subjects


# =============================================================================
# P5 pairs validation (strict)
# =============================================================================

def parse_pairs_from_p5_output(p5_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accept flexible keys but require at least a list of pairs with sujet_url/corrige_url.
    """
    if "pairs" in p5_out and isinstance(p5_out["pairs"], list):
        return p5_out["pairs"]
    if "items" in p5_out and isinstance(p5_out["items"], list):
        return p5_out["items"]
    if "results" in p5_out and isinstance(p5_out["results"], list):
        return p5_out["results"]
    # else: attempt find first list of dicts containing sujet_url
    for k, v in p5_out.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and ("sujet_url" in v[0] or "subject_url" in v[0]):
            return v
    raise ValueError("Sortie P5 invalide: liste de paires introuvable (pairs/items/results).")

def normalize_pair_dict(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    sujet = d.get("sujet_url") or d.get("subject_url") or d.get("sujet") or d.get("subject")
    corr = d.get("corrige_url") or d.get("correction_url") or d.get("corrige") or d.get("correction")
    evidence = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
    return (str(sujet).strip() if sujet else None, str(corr).strip() if corr else None, evidence)

def validate_pairs(
    raw_pairs: List[Dict[str, Any]],
    allowed_domains: List[str],
    max_pairs: int
) -> List[PairSnapshot]:
    out: List[PairSnapshot] = []
    for d in raw_pairs:
        if not isinstance(d, dict):
            continue
        sujet, corr, evidence = normalize_pair_dict(d)
        if not sujet or not corr:
            continue
        if not is_pdf_url(sujet) or not is_pdf_url(corr):
            continue
        if allowed_domains:
            if not domain_allowed(sujet, allowed_domains) or not domain_allowed(corr, allowed_domains):
                continue
        out.append(PairSnapshot(
            sujet_url=sujet,
            corrige_url=corr,
            sujet_url_sha256=sha256_text(sujet),
            corrige_url_sha256=sha256_text(corr),
            evidence=evidence,
        ))
        if len(out) >= max_pairs:
            break

    if not out:
        raise ValueError("Aucune paire valide après contrôle PDF + domaines allowlist CAP.")
    return out


# =============================================================================
# Kernel call (PROD)
# =============================================================================

def try_import_kernel() -> Tuple[bool, Optional[Any], str]:
    """
    Attempts common module names without hardcoding business content.
    """
    candidates = [
        "smaxia_granulo_engine_test",
        "smaxia_granulo_engine",
        "smaxia_granulo_engine_prod",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "run_granulo_test"):
                return True, mod, mod_name
        except Exception:
            continue
    return False, None, "Kernel module introuvable (run_granulo_test)."

def run_kernel(kernel_mod: Any, urls: List[str], volume: int) -> Dict[str, Any]:
    return kernel_mod.run_granulo_test(urls, volume)


# =============================================================================
# Verdict (strict proof from kernel audit)
# =============================================================================

def verdict_from_kernel_audit(kernel_result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    ISO: scellage seulement si preuve explicite 0 orphelin.
    """
    audit = kernel_result.get("audit", {})
    if not isinstance(audit, dict):
        return "FAIL", {"reason": "audit_absent_ou_non_dict"}

    # Explicit orphans list
    for k in ["orphans", "qi_orphans", "uncovered_qi"]:
        if k in audit:
            orphans = audit.get(k) or []
            if not isinstance(orphans, list):
                return "FAIL", {"reason": f"{k}_non_list"}
            return ("SEALED" if len(orphans) == 0 else "FAIL"), {"orphans_key": k, "orphans": orphans}

    # coverage_map + posables list proof
    if isinstance(audit.get("coverage_map"), dict):
        qi_ids = audit.get("posables_qi_ids") or audit.get("qi_ids")
        if isinstance(qi_ids, list):
            covered = set(audit["coverage_map"].keys())
            orphans = [q for q in qi_ids if q not in covered]
            return ("SEALED" if len(orphans) == 0 else "FAIL"), {"orphans_key": "coverage_map", "orphans": orphans}

    return "FAIL", {"reason": "pas_de_preuve_explicite_orphelins_dans_audit"}


# =============================================================================
# UI components (V31.10.22-like logic)
# =============================================================================

def init_state() -> None:
    defaults = {
        "cap": None,
        "cap_sha": None,
        "cap_loaded": False,
        "prompts": None,
        "pairs": None,
        "kernel_ok": False,
        "kernel_mod": None,
        "kernel_mod_name": None,
        "kernel_result": None,
        "verdict": None,
        "audit_detail": None,
        "run_started": False,
        "logs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log(msg: str) -> None:
    st.session_state.logs.append(f"[{now_iso()}] {msg}")

def sidebar_ui() -> None:
    st.sidebar.header("SMAXIA — ISO-PROD (V31.10.24)")
    st.sidebar.caption("Kernel V10.6.2 — QC/ARI/FRT/Triggers uniquement via Kernel.")

    st.session_state.country_code = st.sidebar.text_input("Pays (ISO)", value=st.session_state.get("country_code", "FR")).upper().strip()
    st.session_state.volume = st.sidebar.slider("Volume (saturation)", 1, 10, int(st.session_state.get("volume", 2)))

    st.sidebar.divider()

    # Activation
    if st.sidebar.button("1) Activer Pays (CAP)", type="primary"):
        st.session_state.run_started = True
        st.session_state.kernel_result = None
        st.session_state.verdict = None
        st.session_state.audit_detail = None
        try:
            prompts = load_prompts_scelles()
            st.session_state.prompts = prompts
            log(f"Prompts scellés chargés: P3={prompts.p3_source} | P5={prompts.p5_source}")

            # CAP via P3
            cap_out = llm_json_call_strict(prompts.p3_text, {"country_code_iso": st.session_state.country_code, "country_code": st.session_state.country_code})
            st.session_state.cap = cap_out
            st.session_state.cap_sha = cap_hash(cap_out)
            st.session_state.cap_loaded = True
            log(f"CAP chargé (sha256={st.session_state.cap_sha[:12]}...).")
        except Exception as e:
            st.session_state.cap_loaded = False
            log(f"ERREUR CAP: {repr(e)}")
            st.sidebar.error(f"Erreur CAP: {repr(e)}")

    st.sidebar.divider()

    # Selection UI from CAP (best effort)
    if st.session_state.cap_loaded and isinstance(st.session_state.cap, dict):
        levels, subjects = cap_extract_levels_subjects(st.session_state.cap)
        if not levels:
            levels = ["(CAP: niveaux indisponibles)"]
        if not subjects:
            subjects = ["(CAP: matières indisponibles)"]

        st.session_state.level = st.sidebar.selectbox("2) Niveau", options=levels, index=0)
        st.session_state.subject = st.sidebar.selectbox("3) Matière", options=subjects, index=0)

        # Chapitre: selon P5, le chapitre peut venir après extraction Qi.
        # Pour non-régression UI (V31.10.22): on garde un champ, mais il sert au filtrage/affichage.
        st.session_state.chapter_filter = st.sidebar.text_input("4) Filtre chapitre (optionnel)", value=st.session_state.get("chapter_filter", ""))

        st.session_state.max_pairs = st.sidebar.slider("Max paires Sujet/Corrigé", 1, 30, int(st.session_state.get("max_pairs", 6)))

        if st.sidebar.button("🚀 RUN TEST CHAPITRE COMPLET", type="primary"):
            st.session_state.run_started = True
            st.session_state.kernel_result = None
            st.session_state.verdict = None
            st.session_state.audit_detail = None
            st.rerun()
    else:
        st.sidebar.info("Activer d'abord le pays pour charger le CAP (P3).")

    st.sidebar.divider()
    with st.sidebar.expander("Logs", expanded=False):
        for line in st.session_state.logs[-200:]:
            st.write(line)


# =============================================================================
# Pipeline (10 phases)
# =============================================================================

def phase_01_activation_cap() -> None:
    with st.expander("PHASE 01 — Activation CAP (P3)", expanded=True):
        if not st.session_state.cap_loaded:
            st.warning("CAP non chargé. Utilisez 'Activer Pays (CAP)'.")
            return
        st.success("CAP chargé.")
        st.write(f"CAP sha256: `{st.session_state.cap_sha}`")
        st.json(st.session_state.cap)

def phase_02_catalog_ui() -> None:
    with st.expander("PHASE 02 — Catalogue (Niveau/Matière)", expanded=True):
        if not st.session_state.cap_loaded:
            st.warning("CAP requis.")
            return
        levels, subjects = cap_extract_levels_subjects(st.session_state.cap)
        st.metric("Niveaux détectés", len(levels))
        st.metric("Matières détectées", len(subjects))
        st.write("Sélections:", {"niveau": st.session_state.get("level"), "matière": st.session_state.get("subject"), "filtre_chapitre": st.session_state.get("chapter_filter", "")})

def phase_03_harvest_pairs() -> None:
    with st.expander("PHASE 03 — Harvest Pairs Sujet/Corrigé (P5)", expanded=True):
        if not st.session_state.cap_loaded:
            st.warning("CAP requis.")
            return
        if not st.session_state.prompts:
            st.error("Prompts scellés non chargés.")
            return

        cap = st.session_state.cap
        allowed_domains = cap_get_allowed_domains(cap)

        st.info("Exécution P5 (pairs Sujet/Corrigé) — JSON strict.")
        try:
            p5_out = llm_json_call_strict(
                st.session_state.prompts.p5_text,
                {
                    "cap": cap,
                    "country_code": st.session_state.country_code,
                    "level": st.session_state.get("level"),
                    "subject": st.session_state.get("subject"),
                    "chapter_filter": st.session_state.get("chapter_filter", ""),
                    "max_pairs": int(st.session_state.get("max_pairs", 6)),
                }
            )

            raw_pairs = parse_pairs_from_p5_output(p5_out)
            pairs = validate_pairs(raw_pairs, allowed_domains=allowed_domains, max_pairs=int(st.session_state.max_pairs))
            st.session_state.pairs = pairs
            log(f"Paires valides: {len(pairs)} (allow_domains={len(allowed_domains)}).")

            df = pd.DataFrame([asdict(p) for p in pairs])
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.session_state.pairs = None
            log(f"ERREUR P5: {repr(e)}")
            st.error(f"Erreur P5: {repr(e)}")

def phase_04_snapshot_urls() -> None:
    with st.expander("PHASE 04 — Snapshot URLs + Hashes", expanded=True):
        pairs: Optional[List[PairSnapshot]] = st.session_state.pairs
        if not pairs:
            st.warning("Aucune paire disponible (exécutez P5).")
            return
        urls = []
        for p in pairs:
            urls.append(p.sujet_url)
            urls.append(p.corrige_url)
        st.metric("URLs (Sujets+Corrigés)", len(urls))
        st.code("\n".join(urls[:200]), language="text")
        st.caption("Hashes URL (sha256):")
        st.code("\n".join([sha256_text(u) for u in urls[:200]]), language="text")

def phase_05_kernel_run(progress_bar: st.progress, step_i: int, total_steps: int) -> None:
    with st.expander("PHASE 05 — Kernel Run (V10.6.2)", expanded=True):
        pairs: Optional[List[PairSnapshot]] = st.session_state.pairs
        if not pairs:
            st.warning("Aucune paire disponible.")
            return

        ok, kernel_mod, mod_name = try_import_kernel()
        st.session_state.kernel_ok = ok
        st.session_state.kernel_mod = kernel_mod
        st.session_state.kernel_mod_name = mod_name

        if not ok or kernel_mod is None:
            log("Kernel introuvable.")
            st.error("Kernel introuvable: module exposant run_granulo_test manquant.")
            return

        urls = []
        for p in pairs:
            urls.append(p.sujet_url)
            urls.append(p.corrige_url)

        st.info(f"Appel kernel: {mod_name}.run_granulo_test(urls, volume={st.session_state.volume})")
        try:
            with st.spinner("Exécution kernel..."):
                result = run_kernel(kernel_mod, urls=urls, volume=int(st.session_state.volume))
            if not isinstance(result, dict):
                raise ValueError("Kernel result non-dict (invalide).")

            st.session_state.kernel_result = result
            log(f"Kernel OK. Keys={list(result.keys())}")
            st.success("Kernel exécuté.")
            st.write("Keys:", list(result.keys()))
            st.json(result.get("audit", {}))

        except Exception as e:
            st.session_state.kernel_result = None
            log(f"ERREUR Kernel: {repr(e)}")
            st.error(f"Erreur Kernel: {repr(e)}")

        progress_bar.progress(min(1.0, (step_i + 1) / total_steps))

def _extract_qc_items(kernel_result: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Returns (mode, items)
      mode = "DICT_BY_CHAPTER" | "LIST" | None
      items = list of qc dicts (each may include chapter info)
    """
    qc = kernel_result.get("qc")
    if qc is None:
        return None, []
    if isinstance(qc, dict):
        items = []
        for chap, qlist in qc.items():
            if isinstance(qlist, list):
                for it in qlist:
                    if isinstance(it, dict):
                        it2 = dict(it)
                        it2.setdefault("chapter_code", chap)
                        items.append(it2)
                    else:
                        items.append({"chapter_code": chap, "qc": it})
            else:
                items.append({"chapter_code": chap, "qc": qlist})
        return "DICT_BY_CHAPTER", items
    if isinstance(qc, list):
        items = [it for it in qc if isinstance(it, dict)]
        return "LIST", items
    return None, []

def phase_06_qc_by_chapter() -> None:
    with st.expander("PHASE 06 — QC par Chapitre", expanded=True):
        kr = st.session_state.kernel_result
        if not isinstance(kr, dict):
            st.warning("Kernel non exécuté.")
            return

        mode, items = _extract_qc_items(kr)
        if not items:
            st.warning("Aucune QC détectée dans kernel_result['qc'].")
            return

        # Filter by chapter if user wants
        filt = (st.session_state.get("chapter_filter") or "").strip().lower()
        if filt:
            def keep(it: Dict[str, Any]) -> bool:
                ch = str(it.get("chapter_code") or it.get("intent_code") or it.get("chapter_ref") or "").lower()
                txt = str(it.get("qc_text") or it.get("qc") or "").lower()
                return filt in ch or filt in txt
            items = [it for it in items if keep(it)]

        rows = []
        for it in items:
            qc_text = it.get("qc_text") or it.get("qc") or ""
            chapter_code = it.get("chapter_code") or it.get("intent_code") or it.get("chapter_ref") or ""
            evidence = it.get("evidence_qi_ids") or it.get("qi_ids") or []
            triggers = it.get("triggers") or []
            rows.append({
                "chapter_code": chapter_code,
                "qc_id": it.get("qc_id", ""),
                "qc_text": str(qc_text)[:180],
                "n_qi": len(evidence) if isinstance(evidence, list) else None,
                "n_triggers": len(triggers) if isinstance(triggers, list) else None,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.caption(f"QC items: {len(items)} (mode={mode})")

        st.session_state._qc_items_cache = items  # for explorer

def phase_07_explorer() -> None:
    with st.expander("PHASE 07 — Explorateur QC → ARI → FRT → Triggers → Qi", expanded=True):
        kr = st.session_state.kernel_result
        items = st.session_state.get("_qc_items_cache") or []
        if not isinstance(kr, dict) or not items:
            st.warning("Exécutez d'abord PHASE 06.")
            return

        # Build select labels
        labels = []
        for idx, it in enumerate(items):
            ch = it.get("chapter_code") or it.get("intent_code") or ""
            txt = (it.get("qc_text") or it.get("qc") or "")
            labels.append(f"[{idx}] {ch} — {str(txt)[:80]}")

        sel = st.selectbox("Sélectionner une QC", options=list(range(len(labels))), format_func=lambda i: labels[i])
        it = items[int(sel)]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("QC (brut)")
            st.json(it)
        with c2:
            st.subheader("Détails")
            st.write({
                "qc_id": it.get("qc_id"),
                "chapter_code/intent": it.get("chapter_code") or it.get("intent_code"),
            })
            st.write("Triggers:")
            st.json(it.get("triggers", []))
            st.write("ARI spine (si fourni par kernel):")
            st.json(it.get("ari_spine") or it.get("ari") or {})
            st.write("FRT (si fourni par kernel):")
            st.json(it.get("frt") or {})

def phase_08_audit_orphelins() -> None:
    with st.expander("PHASE 08 — Audit Orphelins (Verdict binaire)", expanded=True):
        kr = st.session_state.kernel_result
        if not isinstance(kr, dict):
            st.warning("Kernel non exécuté.")
            return
        verdict, detail = verdict_from_kernel_audit(kr)
        st.session_state.verdict = verdict
        st.session_state.audit_detail = detail
        st.metric("VERDICT", verdict)
        st.json(detail)

def phase_09_export_proof() -> None:
    with st.expander("PHASE 09 — Export Proof (JSON scellable)", expanded=True):
        if not st.session_state.cap_loaded or not st.session_state.pairs or not st.session_state.kernel_result:
            st.warning("CAP + Paires + Kernel requis pour exporter.")
            return

        cap = st.session_state.cap
        cap_sha = st.session_state.cap_sha
        pairs = st.session_state.pairs
        kr = st.session_state.kernel_result

        prompts = st.session_state.prompts
        prompts_sources = {
            "P3": prompts.p3_source if prompts else "",
            "P5": prompts.p5_source if prompts else "",
        }

        export = ProofExport(
            timestamp=now_iso(),
            app_version=APP_VERSION,
            mode="ISO-PROD",
            inputs={
                "country_code": st.session_state.country_code,
                "level": st.session_state.get("level"),
                "subject": st.session_state.get("subject"),
                "chapter_filter": st.session_state.get("chapter_filter", ""),
                "volume": st.session_state.volume,
                "max_pairs": st.session_state.max_pairs,
            },
            prompts_sources=prompts_sources,
            cap_sha256=cap_sha,
            cap=cap,
            pairs=[asdict(p) for p in pairs],
            kernel_import_ok=bool(st.session_state.kernel_ok),
            kernel_module=str(st.session_state.kernel_mod_name or ""),
            kernel_result_keys=list(kr.keys()) if isinstance(kr, dict) else [],
            verdict=str(st.session_state.verdict or "FAIL"),
            audit=st.session_state.audit_detail or {"reason": "audit_not_run"},
        )

        out = json.dumps(asdict(export), indent=2, ensure_ascii=False)
        st.download_button(
            "📥 EXPORT JSON PROOF",
            data=out,
            file_name=f"smaxia_gte_proof_{st.session_state.country_code}_{int(time.time())}.json",
            mime="application/json",
        )
        st.code(out[:4000], language="json")

def phase_10_verdict() -> None:
    with st.expander("PHASE 10 — Verdict", expanded=True):
        v = st.session_state.get("verdict")
        if not v:
            st.info("Lancer PHASE 08 pour obtenir le verdict.")
            return
        if v == "SEALED":
            st.success("SEALED — Preuve: 0 orphelin (selon audit kernel).")
        else:
            st.error("FAIL — Preuve insuffisante ou orphelins détectés.")
        st.json(st.session_state.get("audit_detail") or {})


# =============================================================================
# Main UI
# =============================================================================

def main_ui() -> None:
    st.set_page_config(page_title=f"{APP_TITLE} {APP_VERSION}", layout="wide")
    st.title(f"{APP_TITLE} — {APP_VERSION}")
    st.caption("ISO-PROD / Industrial Safety — Kernel V10.6.2 = source de vérité QC/ARI/FRT/Triggers.")

    init_state()
    sidebar_ui()

    # Global progress and status
    progress = st.progress(0.0)
    status = st.empty()

    # If user clicked RUN, execute pipeline sequentially (no background)
    if st.session_state.get("run_started") and st.session_state.cap_loaded and st.session_state.get("level") and st.session_state.get("subject"):
        total = len(PIPELINE_PHASES)
        phase_fns = [
            phase_01_activation_cap,
            phase_02_catalog_ui,
            phase_03_harvest_pairs,
            phase_04_snapshot_urls,
            None,  # kernel run needs progress handle
            phase_06_qc_by_chapter,
            phase_07_explorer,
            phase_08_audit_orphelins,
            phase_09_export_proof,
            phase_10_verdict,
        ]

        # PHASE 01-04
        for i in range(0, 4):
            status.write(f"Exécution: {PIPELINE_PHASES[i]}")
            phase_fns[i]()
            progress.progress((i + 1) / total)

        # PHASE 05 kernel
        status.write(f"Exécution: {PIPELINE_PHASES[4]}")
        phase_05_kernel_run(progress, 4, total)

        # PHASE 06-10
        for i in range(5, total):
            status.write(f"Exécution: {PIPELINE_PHASES[i]}")
            phase_fns[i]()
            progress.progress((i + 1) / total)

        status.write("Pipeline terminé.")
    else:
        st.info("1) Activer Pays (CAP) puis 2) choisir Niveau/Matière, puis 3) RUN TEST CHAPITRE COMPLET.")

    # Always show phases (non régressif: visibilité pipeline)
    st.subheader("Pipeline (phases)")
    phase_01_activation_cap()
    phase_02_catalog_ui()
    phase_03_harvest_pairs()
    phase_04_snapshot_urls()
    with st.expander("PHASE 05 — Kernel Run (V10.6.2)", expanded=False):
        st.write("Exécutez via RUN TEST CHAPITRE COMPLET (sidebar).")
    phase_06_qc_by_chapter()
    phase_07_explorer()
    phase_08_audit_orphelins()
    phase_09_export_proof()
    phase_10_verdict()


if __name__ == "__main__":
    main_ui()
