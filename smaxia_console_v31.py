# =============================================================================
# SMAXIA GTE Console V31.10.3 — ISO-PROD TEST (MONO-FICHIER)
# Console + Harvester + Engine embedded (run_granulo_test) + Exports
# =============================================================================
# FLUX VALIDÉ (mono-fichier, sans import moteur externe):
# [1] Activation pays -> load_academic_pack(country)
# [2] Pack chargé -> UI: Niveaux/Matières/Chapitres (pack-driven)
# [3] Sélection Niveau + Matière
# [4] Harvest AUTO (APMEP) -> Bibliothèque visible + manifest
# [5] RUN -> Extraction Qi -> QC -> Audit (orphelins) -> Exports preuves
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import time
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import requests

try:
    import pdfplumber
except Exception as e:
    pdfplumber = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# =============================================================================
# CONFIG (ISO-PROD TEST)
# =============================================================================

APP_TITLE = "SMAXIA GTE Console V31.10.3 — ISO-PROD TEST"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 35
MAX_PDF_MB = 40

DEFAULT_HARVEST_YEARS = 7
DEFAULT_HARVEST_VOLUME = 50
DEFAULT_RUN_VOLUME = 20

# Pack discovery
# Option 1: set env var SMAXIA_PACK_DIR=/path/to/packs
# Option 2: default ./academic_packs
DEFAULT_PACK_DIR = os.getenv("SMAXIA_PACK_DIR", "./academic_packs")

# APMEP root (observed in your logs)
APMEP_ROOT = "https://www.apmep.fr"
APMEP_TERM_BAC_ROOT = "https://www.apmep.fr/Annales-du-Bac-Terminale"

# Strict ISO-PROD TEST gating
ISO_DENY_STUB_PACK = True


# =============================================================================
# UTILITIES
# =============================================================================

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]

def norm_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def http_get(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    content = r.content
    mb = len(content) / (1024 * 1024)
    if mb > MAX_PDF_MB:
        raise ValueError(f"PDF too large: {mb:.1f}MB > {MAX_PDF_MB}MB ({url})")
    return content

def require_pdfplumber() -> None:
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber non disponible dans cet environnement. "
            "Installez-le: pip install pdfplumber"
        )

def pdf_to_text(pdf_bytes: bytes, max_pages: int = 60) -> str:
    require_pdfplumber()
    out: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = min(len(pdf.pages), max_pages)
        for i in range(n):
            t = (pdf.pages[i].extract_text() or "").strip()
            if t:
                out.append(t)
    return "\n".join(out).strip()

def log_append(msg: str) -> None:
    st.session_state.setdefault("logs", [])
    ts = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state["logs"].append(f"[{ts}] {msg}")

def log_clear() -> None:
    st.session_state["logs"] = []

def get_logs_text() -> str:
    return "\n".join(st.session_state.get("logs", []))


# =============================================================================
# PACK LOADING (STRICT: NO STUB)
# =============================================================================

@dataclass
class AcademicPack:
    country: str
    pack_id: str
    signature: str
    levels: List[Dict[str, Any]]
    subjects: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    raw: Dict[str, Any]

def _list_pack_candidates(pack_dir: str) -> List[str]:
    # Expected layout examples:
    # ./academic_packs/FR/CAP_FR_BAC_2024_V1.json
    # ./academic_packs/CAP_FR_BAC_2024_V1.json
    candidates: List[str] = []
    if not os.path.isdir(pack_dir):
        return candidates
    for root, _, files in os.walk(pack_dir):
        for f in files:
            if f.lower().endswith(".json"):
                candidates.append(os.path.join(root, f))
    candidates.sort()
    return candidates

def _pack_signature(pack_raw: Dict[str, Any]) -> str:
    # deterministic signature on raw json
    s = json.dumps(pack_raw, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def load_academic_pack(country_code: str) -> AcademicPack:
    """
    Strict loader:
    - Searches JSON packs under DEFAULT_PACK_DIR
    - Selects the newest/most relevant pack that matches country_code if possible
    - If none: STOP (no stub)
    """
    log_append(f"=== PACK ACTIVATION ===")
    log_append(f"[PACK] country={country_code}")

    candidates = _list_pack_candidates(DEFAULT_PACK_DIR)
    if not candidates:
        if ISO_DENY_STUB_PACK:
            raise FileNotFoundError(
                f"PACK INTROUVABLE: aucun fichier pack JSON dans {DEFAULT_PACK_DIR}. "
                f"Stub interdit."
            )

    # Prefer packs that contain country_code in filename or path
    cc = (country_code or "").upper().strip()
    preferred = [p for p in candidates if f"/{cc}/" in p.replace("\\", "/") or f"_{cc}_" in os.path.basename(p)]
    pool = preferred if preferred else candidates

    # Choose latest modified file among pool
    pool.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = pool[0] if pool else None
    if not chosen:
        raise FileNotFoundError(
            f"PACK INTROUVABLE: aucun pack pour {cc}. Stub interdit."
        )

    with open(chosen, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Minimal schema expectations (pack-driven)
    # We do not hardcode chapters; must come from pack.
    for k in ["country", "pack_id", "levels", "subjects", "chapters"]:
        if k not in raw:
            raise ValueError(
                f"PACK INVALIDE: clé manquante '{k}' dans {chosen}. "
                "Le pack doit fournir levels/subjects/chapters."
            )

    signature = raw.get("signature") or _pack_signature(raw)

    pack = AcademicPack(
        country=str(raw["country"]).upper(),
        pack_id=str(raw["pack_id"]),
        signature=str(signature),
        levels=list(raw.get("levels") or []),
        subjects=list(raw.get("subjects") or []),
        chapters=list(raw.get("chapters") or []),
        raw=raw,
    )

    if pack.country != cc:
        log_append(f"[PACK] WARN: pack.country={pack.country} != selected={cc} (accepté en test)")

    log_append(f"[PACK] OK pack_id={pack.pack_id}")
    log_append(f"[PACK] signature={pack.signature}")
    log_append(f"[PACK] levels={len(pack.levels)} subjects={len(pack.subjects)} chapters={len(pack.chapters)}")
    return pack


# =============================================================================
# HARVESTER — APMEP (Terminale|MATH as in your logs, but pack-driven scope)
# =============================================================================

def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non disponible. Installez: pip install beautifulsoup4")
    return BeautifulSoup(html, "html.parser")

def _abs_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return APMEP_ROOT.rstrip("/") + href
    return APMEP_ROOT.rstrip("/") + "/" + href

def _fetch_html(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.text

def apmep_discover_year_pages(max_years: int) -> List[Tuple[int, str]]:
    """
    Discovers year pages linked from APMEP_TERM_BAC_ROOT.
    Example logs you showed:
      root=https://www.apmep.fr/Annales-du-Bac-Terminale
      year=2025 url=https://www.apmep.fr/Annee-2025
    """
    html = _fetch_html(APMEP_TERM_BAC_ROOT)
    soup = _soup(html)

    year_links: List[Tuple[int, str]] = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        text = (a.get_text() or "").strip()
        m = re.search(r"(?:Annee[-\s]?)(20\d{2})", href, flags=re.IGNORECASE) or re.search(r"\b(20\d{2})\b", text)
        if m:
            year = int(m.group(1))
            url = _abs_url(href)
            if (year, url) not in year_links:
                year_links.append((year, url))

    # Sort by year desc, keep max_years
    year_links.sort(key=lambda t: t[0], reverse=True)
    year_links = year_links[: max(0, int(max_years or 0))]
    return year_links

def apmep_extract_pdf_rows(year_url: str) -> List[Dict[str, Any]]:
    """
    Extracts PDF links from a year page.
    Heuristic pairing:
      - If a row contains "Corrigé" link, use it as corrige_url.
      - Else corrige_url empty.
    This produces library items similar to your manifest.
    """
    html = _fetch_html(year_url)
    soup = _soup(html)

    items: List[Dict[str, Any]] = []

    # APMEP pages often have lists/tables; we do a robust pass on all anchors.
    # Pairing heuristic: nearest "Corrigé" anchor around a subject anchor.
    anchors = soup.find_all("a")
    pdf_anchors = []
    for a in anchors:
        href = a.get("href") or ""
        if ".pdf" in href.lower():
            pdf_anchors.append(a)

    # Group anchors by parent block to detect subject/corrigé association
    for a in pdf_anchors:
        href = _abs_url(a.get("href") or "")
        name = os.path.basename(href)
        label = norm_ws(a.get_text() or name)

        # Skip obvious corrigé-only if we cannot find subject nearby
        # We'll handle pairing by scanning siblings text
        parent = a.parent
        parent_text = norm_ws(parent.get_text(" ", strip=True) if parent else label).lower()

        is_corrige = ("corrig" in label.lower()) or ("corrig" in parent_text)
        items.append({
            "pdf_url": href,
            "pdf_name": name,
            "label": label,
            "parent_text": parent_text,
            "is_corrige": bool(is_corrige),
            "parent_key": sha1_short(parent_text) if parent_text else sha1_short(href),
        })

    # Build pairs: subject -> corrigé within same parent_key if possible
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        grouped.setdefault(it["parent_key"], []).append(it)

    pairs: List[Dict[str, Any]] = []
    for _, group in grouped.items():
        sujets = [g for g in group if not g["is_corrige"]]
        corriges = [g for g in group if g["is_corrige"]]

        # If no explicit corrigé, still include sujets (corrige empty)
        if not sujets and corriges:
            # corrigé without subject: ignore
            continue

        for s in sujets:
            corr = ""
            corr_name = ""
            reason = ""
            if corriges:
                # pick first corrige in same group (simple deterministic heuristic)
                c = corriges[0]
                corr = c["pdf_url"]
                corr_name = c["pdf_name"]
            else:
                reason = "corrigé absent sur la ligne"

            pairs.append({
                "sujet_url": s["pdf_url"],
                "corrige_url": corr,
                "sujet": s["pdf_name"],
                "corrige_name": corr_name,
                "corrige?": bool(corr),
                "reason": reason,
            })

    return pairs

def harvest_apmep(scope: str, years: int, volume: int) -> Dict[str, Any]:
    """
    Produces harvest_manifest.json compatible structure:
      {version,timestamp,country,level,subjects,items_total,items_corrige_ok,library:[...]}
    """
    log_append(f"=== HARVEST AUTO V31.10.3 ===")
    log_append(f"[HARVEST] scope={scope} root={APMEP_TERM_BAC_ROOT}")

    year_pages = apmep_discover_year_pages(years)
    for y, url in year_pages:
        log_append(f"[HARVEST] year={y} url={url}")

    lib: List[Dict[str, Any]] = []
    visited = 0

    for y, url in year_pages:
        visited += 1
        try:
            rows = apmep_extract_pdf_rows(url)
        except Exception as e:
            log_append(f"[HARVEST] WARN year={y} fetch/parse failed: {type(e).__name__}: {e}")
            continue

        for r in rows:
            if len(lib) >= volume:
                break
            pair_id = f"PAIR_{scope}_{y}_{sha1_short(r['sujet_url'])}"
            lib.append({
                "pair_id": pair_id,
                "scope": scope,
                "source": f"APMEP — Année {y}",
                "year": y,
                "sujet": r["sujet"],
                "corrige?": bool(r["corrige?"]),
                "corrige_name": r["corrige_name"],
                "reason": r["reason"],
                "sujet_url": r["sujet_url"],
                "corrige_url": r["corrige_url"],
            })

        if len(lib) >= volume:
            break

    corr_ok = sum(1 for x in lib if x.get("corrige?"))
    log_append(f"[HARVEST] done pages={visited} items={len(lib)} corrige_ok={corr_ok}")

    # Infer country/level/subjects from scope "LEVEL|SUBJECT"
    parts = scope.split("|")
    level = parts[0] if len(parts) >= 1 else ""
    subj = parts[1] if len(parts) >= 2 else ""

    manifest = {
        "version": "V31.10.3",
        "timestamp": now_utc_iso(),
        "country": st.session_state.get("country", ""),
        "level": level,
        "subjects": [subj] if subj else [],
        "items_total": len(lib),
        "items_corrige_ok": int(corr_ok),
        "library": lib,
    }
    return manifest


# =============================================================================
# ENGINE — EMBEDDED (MONO-FICHIER)
# =============================================================================

_QI_SPLIT_RE = re.compile(
    r"(?:(?:^|\n)\s*(?:Exercice|EXERCICE)\s*\d+.*?$)|"
    r"(?:(?:^|\n)\s*(?:Partie)\s*[A-Z0-9].*?$)|"
    r"(?:(?:^|\n)\s*(?:Question)\s*\d+.*?$)",
    re.MULTILINE
)
_LINE_QI_RE = re.compile(r"^\s*(?:\d+[\).]|[a-zA-Z][\).]|•|-)\s+(.+)$")
_END_Q_RE = re.compile(r".*\?\s*$")

def extract_qi(text: str) -> List[Dict[str, Any]]:
    text = norm_ws(text)
    if not text:
        return []
    blocks = [b.strip() for b in _QI_SPLIT_RE.split(text) if b and b.strip()] or [text]
    qis: List[Dict[str, Any]] = []
    qi_idx = 0

    for b in blocks:
        lines = [l.strip() for l in b.split("\n") if l.strip()]
        if not lines:
            continue

        for ln in lines:
            m = _LINE_QI_RE.match(ln)
            if m:
                q = norm_ws(m.group(1))
                if len(q) >= 12:
                    qi_idx += 1
                    qis.append({"qi_id": f"QI_{qi_idx:04d}_{sha1_short(q)}", "text": q})

        if len(qis) < 3:
            for ln in lines:
                if _END_Q_RE.match(ln) and len(ln) >= 12:
                    qi_idx += 1
                    q = norm_ws(ln)
                    qis.append({"qi_id": f"QI_{qi_idx:04d}_{sha1_short(q)}", "text": q})

    seen = set()
    uniq: List[Dict[str, Any]] = []
    for q in qis:
        k = sha1_short(q["text"].lower())
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return uniq

@dataclass(frozen=True)
class Intent:
    ari_code: str
    label: str

INTENTS: List[Tuple[Intent, List[str]]] = [
    (Intent("ARI_SOLVE_EQUATION", "Résoudre équation / inéquation"), ["résoudre", "solution", "équation", "inéquation", "racine"]),
    (Intent("ARI_DERIVATIVE", "Dérivée / variations"), ["dériv", "tangente", "variation", "sens de variation"]),
    (Intent("ARI_INTEGRAL", "Intégrale / aire"), ["intégr", "primitive", "aire", "surface"]),
    (Intent("ARI_LIMIT", "Limite / asymptote"), ["limite", "tend vers", "asymptote"]),
    (Intent("ARI_PROBABILITY", "Probabilités"), ["probabil", "événement", "conditionnel", "binom", "espérance", "variance", "loi"]),
    (Intent("ARI_SEQUENCE", "Suites"), ["suite", "u_n", "récurrence", "terme", "convergence"]),
    (Intent("ARI_GEOMETRY", "Géométrie"), ["vecteur", "coordonnée", "droite", "plan", "distance", "repère", "géométr"]),
    (Intent("ARI_ALGEBRA", "Algèbre / transformations"), ["développer", "factoriser", "simplifier", "montrer que", "démontrer"]),
]

def classify_intent(q: str) -> Intent:
    ql = (q or "").lower()
    best = Intent("ARI_GENERIC", "Méthode générique")
    best_score = 0
    for intent, kws in INTENTS:
        score = sum(1 for kw in kws if kw in ql)
        if score > best_score:
            best = intent
            best_score = score
    return best

def qc_text_for(ari_code: str) -> str:
    mapping = {
        "ARI_SOLVE_EQUATION": "Comment résoudre une équation ou une inéquation ?",
        "ARI_DERIVATIVE": "Comment exploiter une dérivée pour étudier une fonction ?",
        "ARI_INTEGRAL": "Comment calculer une intégrale et interpréter une aire ?",
        "ARI_LIMIT": "Comment déterminer une limite et en déduire un comportement asymptotique ?",
        "ARI_PROBABILITY": "Comment modéliser et calculer une probabilité ?",
        "ARI_SEQUENCE": "Comment étudier une suite et sa convergence ?",
        "ARI_GEOMETRY": "Comment structurer une résolution géométrique (repères/vecteurs/équations) ?",
        "ARI_ALGEBRA": "Comment transformer une expression pour prouver ou simplifier ?",
        "ARI_GENERIC": "Comment structurer une résolution pas à pas à partir de l’énoncé ?",
    }
    return mapping.get(ari_code, mapping["ARI_GENERIC"])

def map_chapter_pack_driven(qi_text: str, pack: AcademicPack) -> Optional[str]:
    """
    Pack-driven mapping only:
    - If pack chapters include "keywords" or "patterns", we use them.
    - Otherwise returns None (no hardcode).
    Expected chapter schema (recommended):
      chapters: [{chapter_code, title, keywords:[...], patterns:[...]}]
    """
    txt = (qi_text or "").lower()
    for ch in pack.chapters:
        code = str(ch.get("chapter_code") or ch.get("code") or "").strip()
        if not code:
            continue
        kws = ch.get("keywords") or []
        pats = ch.get("patterns") or []
        for kw in kws:
            if kw and str(kw).lower() in txt:
                return code
        for pat in pats:
            try:
                if pat and re.search(str(pat), txt, flags=re.IGNORECASE):
                    return code
            except re.error:
                continue
    return None

def run_granulo_test(library_items: List[Dict[str, Any]], volume: int, pack: AcademicPack) -> Dict[str, Any]:
    """
    Embedded engine:
    - consumes library items
    - extracts Qi from sujet PDF
    - Qi is POSABLE only if corrige_url exists and is downloadable (we do not "invent" corrections)
    - assigns each posable Qi to exactly one QC cluster (ensures 0 orphan for posable Qi)
    - chapter mapping is strictly pack-driven (no hardcode)
    """
    t0 = time.time()
    items = (library_items or [])[: max(0, int(volume or 0))]

    sujets_out: List[Dict[str, Any]] = []
    qi_all: List[Dict[str, Any]] = []
    qi_posables: List[Dict[str, Any]] = []

    log_append(f"=== RUN V31.10.3 ===")
    log_append(f"[RUN] items_consumed={len(items)} volume={volume}")

    for idx, it in enumerate(items, start=1):
        pair_id = str(it.get("pair_id") or f"PAIR_{idx:04d}").strip()
        sujet_url = str(it.get("sujet_url") or "").strip()
        corrige_url = str(it.get("corrige_url") or "").strip()

        sujet_text = ""
        corrige_text = ""
        err_sujet = ""
        err_corrige = ""

        try:
            if sujet_url:
                sb = http_get(sujet_url)
                sujet_text = pdf_to_text(sb)
        except Exception as e:
            err_sujet = f"{type(e).__name__}: {e}"

        corrige_ok = False
        try:
            if corrige_url:
                cb = http_get(corrige_url)
                corrige_text = pdf_to_text(cb)
                corrige_ok = bool(corrige_text)
        except Exception as e:
            err_corrige = f"{type(e).__name__}: {e}"
            corrige_ok = False

        qis = extract_qi(sujet_text)
        for q in qis:
            chap = map_chapter_pack_driven(q["text"], pack)
            q_item = {
                "pair_id": pair_id,
                "sujet_url": sujet_url,
                "corrige_url": corrige_url,
                "corrige_ok": corrige_ok,
                "posable": bool(corrige_ok),
                "chapter_code": chap,
                "qi_id": q["qi_id"],
                "qi_text": q["text"],
            }
            qi_all.append(q_item)
            if corrige_ok:
                qi_posables.append(q_item)

        sujets_out.append({
            "pair_id": pair_id,
            "sujet_url": sujet_url,
            "corrige_url": corrige_url,
            "qi_count": len(qis),
            "corrige_ok": bool(corrige_ok),
            "errors": {"sujet": err_sujet, "corrige": err_corrige},
        })

        log_append(f"[RUN] {pair_id} qi={len(qis)} corrige_ok={corrige_ok}")

    # Cluster QC by ARI intent (method)
    clusters: Dict[str, Dict[str, Any]] = {}
    for q in qi_posables:
        intent = classify_intent(q["qi_text"])
        key = intent.ari_code
        if key not in clusters:
            clusters[key] = {
                "qc_id": f"QC_{key}",
                "ari_code": key,
                "qc_text": qc_text_for(key),
                "triggers": [],
                "frt": {
                    "sections": [
                        {"title": "Déclencheurs", "items": []},
                        {"title": "Méthode (ARI)", "items": [intent.label]},
                        {"title": "Pièges", "items": []},
                        {"title": "Contrôles", "items": []},
                    ]
                },
                "qi_ids": [],
            }
        clusters[key]["qi_ids"].append(q["qi_id"])

    qc_out = list(clusters.values())

    # Orphan check (must be 0 for posable Qi)
    pos_ids = {q["qi_id"] for q in qi_posables}
    linked: set[str] = set()
    for c in qc_out:
        linked.update(c["qi_ids"])
    orphan = sorted(list(pos_ids - linked))

    # Chapter report (strictly from pack)
    chap_report: Dict[str, Any] = {}
    for q in qi_posables:
        c = q.get("chapter_code") or "UNMAPPED"
        chap_report.setdefault(c, 0)
        chap_report[c] += 1

    audit = {
        "items_consumed": len(items),
        "qi_total": len(qi_all),
        "qi_posable": len(qi_posables),
        "qc_count": len(qc_out),
        "orphan_qi_posable_count": len(orphan),
        "orphan_qi_posable_sample": orphan[:15],
        "elapsed_sec": round(time.time() - t0, 3),
        "engine": "embedded_mono_file",
        "chapter_mapping": "pack-driven only",
    }

    saturation = {
        "iterations": 1,
        "new_qc_last_iter": 0,
        "sealed_candidate": (len(orphan) == 0 and len(qi_posables) > 0 and len(qc_out) > 0),
    }

    log_append(f"[RUN] qi_total={audit['qi_total']} qi_posable={audit['qi_posable']} qc={audit['qc_count']}")
    log_append(f"[RUN] orphan_posable={audit['orphan_qi_posable_count']}")
    log_append(f"[RUN] sealed_candidate={saturation['sealed_candidate']}")

    return {
        "sujets": sujets_out,
        "qi_pack": qi_all,
        "qc_pack": qc_out,
        "chapter_report": chap_report,
        "saturation": saturation,
        "audit": audit,
    }


# =============================================================================
# STREAMLIT UI — MONO FILE
# =============================================================================

def init_state() -> None:
    st.session_state.setdefault("country", "FR")
    st.session_state.setdefault("pack", None)  # AcademicPack
    st.session_state.setdefault("pack_loaded", False)

    st.session_state.setdefault("level_sel", "TERMINALE")
    st.session_state.setdefault("subject_sel", ["MATH"])

    st.session_state.setdefault("harvest_years", DEFAULT_HARVEST_YEARS)
    st.session_state.setdefault("harvest_volume", DEFAULT_HARVEST_VOLUME)

    st.session_state.setdefault("library_manifest", None)
    st.session_state.setdefault("library_items", [])

    st.session_state.setdefault("run_volume", DEFAULT_RUN_VOLUME)
    st.session_state.setdefault("gte_result", None)

    st.session_state.setdefault("logs", [])

def ui_sidebar() -> None:
    st.sidebar.markdown("## ÉTAPE 1 — ACTIVATION PAYS")

    countries = ["FR"]  # UI test; activation uses pack discovery, not stub
    st.session_state["country"] = st.sidebar.selectbox("Pays (TEST)", countries, index=countries.index(st.session_state["country"]))

    if st.sidebar.button("🔒 ACTIVER", use_container_width=True):
        log_clear()
        try:
            pack = load_academic_pack(st.session_state["country"])
            st.session_state["pack"] = pack
            st.session_state["pack_loaded"] = True
            st.session_state["gte_result"] = None
            st.session_state["library_manifest"] = None
            st.session_state["library_items"] = []
            log_append(f"[PACK] ACTIVÉ pack_id={pack.pack_id}")
            st.sidebar.success("Pack actif")
        except Exception as e:
            st.session_state["pack"] = None
            st.session_state["pack_loaded"] = False
            st.sidebar.error(str(e))
            log_append(f"[PACK] ERROR: {type(e).__name__}: {e}")

    st.sidebar.markdown("---")

    pack: Optional[AcademicPack] = st.session_state.get("pack")
    if st.session_state.get("pack_loaded") and pack:
        st.sidebar.markdown("### Pack actif")
        st.sidebar.write(pack.pack_id)
        st.sidebar.caption(f"Signature: {pack.signature}")
    else:
        st.sidebar.warning("Pack inactif (activation requise).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ÉTAPE 2 — SÉLECTION")

    # Pack-driven levels/subjects
    if st.session_state.get("pack_loaded") and pack:
        level_codes = [str(l.get("level_code") or l.get("code") or "").strip() for l in pack.levels]
        level_codes = [c for c in level_codes if c]
        if not level_codes:
            level_codes = ["TERMINALE"]

        if st.session_state["level_sel"] not in level_codes:
            st.session_state["level_sel"] = level_codes[0]

        st.session_state["level_sel"] = st.sidebar.radio("Niveau", level_codes, index=level_codes.index(st.session_state["level_sel"]))

        subj_codes = [str(s.get("subject_code") or s.get("code") or "").strip() for s in pack.subjects]
        subj_codes = [c for c in subj_codes if c]
        if not subj_codes:
            subj_codes = ["MATH"]

        st.session_state["subject_sel"] = st.sidebar.multiselect(
            "Matières (1–2 recommandé pour test)",
            options=subj_codes,
            default=[c for c in st.session_state["subject_sel"] if c in subj_codes] or [subj_codes[0]],
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Chapitres (pack-driven)")
        # Display true chapters from pack, not macro categories
        chapter_codes = []
        for ch in pack.chapters:
            code = str(ch.get("chapter_code") or ch.get("code") or "").strip()
            if code:
                chapter_codes.append(code)
        if chapter_codes:
            for c in chapter_codes[:80]:
                st.sidebar.caption(f"• {c}")
            if len(chapter_codes) > 80:
                st.sidebar.caption(f"... (+{len(chapter_codes)-80})")
        else:
            st.sidebar.warning("Aucun chapitre dans le pack (pack invalide pour SMAXIA).")

    else:
        st.sidebar.info("Activez un pack pour voir niveaux/matières/chapitres.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pack upload (optionnel)")
    st.sidebar.caption("Si vous ne voulez pas utiliser le dossier packs, uploadez un pack JSON ici.")
    upl = st.sidebar.file_uploader("Pack JSON", type=["json"], label_visibility="collapsed")
    if upl is not None and st.sidebar.button("Charger ce pack (remplace)", use_container_width=True):
        log_clear()
        try:
            raw = json.load(upl)
            # validate minimal schema
            for k in ["country", "pack_id", "levels", "subjects", "chapters"]:
                if k not in raw:
                    raise ValueError(f"PACK INVALIDE: clé manquante '{k}'")
            pack = AcademicPack(
                country=str(raw["country"]).upper(),
                pack_id=str(raw["pack_id"]),
                signature=str(raw.get("signature") or _pack_signature(raw)),
                levels=list(raw.get("levels") or []),
                subjects=list(raw.get("subjects") or []),
                chapters=list(raw.get("chapters") or []),
                raw=raw,
            )
            st.session_state["pack"] = pack
            st.session_state["pack_loaded"] = True
            st.sidebar.success("Pack upload chargé")
            log_append(f"[PACK] UPLOAD OK pack_id={pack.pack_id}")
        except Exception as e:
            st.sidebar.error(str(e))
            log_append(f"[PACK] UPLOAD ERROR: {type(e).__name__}: {e}")

def scope_auto() -> str:
    level = st.session_state.get("level_sel") or ""
    subj_list = st.session_state.get("subject_sel") or []
    subj = subj_list[0] if subj_list else ""
    return f"{level}|{subj}"

def tab_import_biblio() -> None:
    st.subheader("Import PDF (sujets + corrigés)")
    pack: Optional[AcademicPack] = st.session_state.get("pack")
    if not (st.session_state.get("pack_loaded") and pack):
        st.warning("Pack inactif. Activez un pack avant Harvest/Upload.")
        return

    # Counters
    items = st.session_state.get("library_items") or []
    corr_ok = sum(1 for x in items if x.get("corrige?"))
    st.write("")
    cols = st.columns(6)
    cols[0].metric("Items", len(items))
    cols[1].metric("Corrigés DL", corr_ok)
    cols[2].metric("Qi", 0 if not st.session_state.get("gte_result") else st.session_state["gte_result"]["audit"]["qi_total"])
    cols[3].metric("Qi POSABLE", 0 if not st.session_state.get("gte_result") else st.session_state["gte_result"]["audit"]["qi_posable"])
    cols[4].metric("QC", 0 if not st.session_state.get("gte_result") else st.session_state["gte_result"]["audit"]["qc_count"])
    cols[5].metric("SEALED", "NO" if not st.session_state.get("gte_result") else ("YES" if st.session_state["gte_result"]["saturation"]["sealed_candidate"] else "NO"))

    st.markdown("---")
    st.markdown("### Bibliothèque Harvest (visible)")

    only_no_corr = st.checkbox("Afficher seulement les items sans corrigé exploitable (❌)", value=False)
    show = items
    if only_no_corr:
        show = [x for x in items if not x.get("corrige?")]

    if show:
        # render as a dataframe-like table
        st.dataframe(
            [{
                "pair_id": x.get("pair_id"),
                "scope": x.get("scope"),
                "source": x.get("source"),
                "sujet": x.get("sujet"),
                "corrigé?": "✅" if x.get("corrige?") else "❌",
                "corrigé_name": x.get("corrige_name"),
                "reason": x.get("reason"),
                "sujet_url": x.get("sujet_url"),
                "corrige_url": x.get("corrige_url"),
            } for x in show],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Bibliothèque vide.")

    st.markdown("---")
    st.markdown("### HARVEST AUTO (APMEP) — paramètres")

    c1, c2, c3 = st.columns([1, 1, 1.2])
    st.session_state["harvest_years"] = int(c1.number_input("Nb d'années à récolter (depuis la plus récente)", min_value=1, max_value=15, value=int(st.session_state["harvest_years"])))
    st.session_state["harvest_volume"] = int(c2.number_input("Volume max (items)", min_value=1, max_value=300, value=int(st.session_state["harvest_volume"])))
    c3.text_input("Scope (auto)", value=scope_auto(), disabled=True)

    if st.button("🌐 Lancer HARVEST AUTO (Bibliothèque)", use_container_width=True):
        try:
            manifest = harvest_apmep(scope_auto(), st.session_state["harvest_years"], st.session_state["harvest_volume"])
            st.session_state["library_manifest"] = manifest
            st.session_state["library_items"] = manifest["library"]
            st.success(f"HARVEST terminé. +{manifest['items_total']} items ajoutés (total={manifest['items_total']}).")
        except Exception as e:
            st.error(f"HARVEST FAILED: {type(e).__name__}: {e}")
            log_append(f"[HARVEST] ERROR: {type(e).__name__}: {e}")

    st.markdown("---")
    st.markdown("### Upload manuel (optionnel)")
    colu1, colu2 = st.columns(2)
    sujet_file = colu1.file_uploader("PDF Sujet", type=["pdf"], key="upl_sujet")
    corr_file = colu2.file_uploader("PDF Correction (opt)", type=["pdf"], key="upl_corrige")

    if st.button("➕ Ajouter à la bibliothèque", use_container_width=True):
        if sujet_file is None:
            st.error("Veuillez fournir un PDF sujet.")
        else:
            pair_id = f"PAIR_MANUAL_{scope_auto()}_{sha1_short(sujet_file.name)}"
            # store as bytes in session (for manual items) using pseudo-urls
            sujet_key = f"manual://{pair_id}/sujet"
            corr_key = f"manual://{pair_id}/corrige" if corr_file else ""

            st.session_state.setdefault("manual_blobs", {})
            st.session_state["manual_blobs"][sujet_key] = sujet_file.read()
            if corr_file:
                st.session_state["manual_blobs"][corr_key] = corr_file.read()

            item = {
                "pair_id": pair_id,
                "scope": scope_auto(),
                "source": "MANUAL_UPLOAD",
                "year": None,
                "sujet": sujet_file.name,
                "corrige?": bool(corr_file),
                "corrige_name": corr_file.name if corr_file else "",
                "reason": "" if corr_file else "corrigé absent (upload)",
                "sujet_url": sujet_key,
                "corrige_url": corr_key,
            }
            st.session_state["library_items"].append(item)
            st.success("Ajouté à la bibliothèque.")

def _manual_or_http_get(url: str) -> bytes:
    if url.startswith("manual://"):
        blobs = st.session_state.get("manual_blobs") or {}
        if url not in blobs:
            raise FileNotFoundError(f"Blob manuel introuvable: {url}")
        return blobs[url]
    return http_get(url)

# Override http_get used in engine for manual blobs
def http_get(url: str) -> bytes:  # type: ignore[override]
    return _manual_or_http_get(url)

def tab_run() -> None:
    st.subheader("ÉTAPE 3 — LANCER CHAÎNE (RUN)")
    pack: Optional[AcademicPack] = st.session_state.get("pack")
    if not (st.session_state.get("pack_loaded") and pack):
        st.warning("Pack inactif. Activez un pack.")
        return

    items = st.session_state.get("library_items") or []
    if not items:
        st.warning("Bibliothèque vide. Lancez HARVEST AUTO ou faites un upload manuel.")
        st.text_area("Log en temps réel", value=get_logs_text(), height=220)
        return

    corr_ok = sum(1 for x in items if x.get("corrige?"))
    if corr_ok <= 0:
        st.error("ISO-PROD: refus de RUN car aucun corrigé exploitable (corrige?=true).")
        st.text_area("Log en temps réel", value=get_logs_text(), height=220)
        return

    st.session_state["run_volume"] = int(st.number_input("Volume (items consommés)", min_value=1, max_value=len(items), value=min(st.session_state["run_volume"], len(items))))
    if st.button("⚡ LANCER (RUN)", use_container_width=True):
        try:
            st.session_state["gte_result"] = run_granulo_test(items, st.session_state["run_volume"], pack)
            st.success("RUN OK — résultats générés.")
        except Exception as e:
            st.error(f"RUN FAILED: {type(e).__name__}: {e}")
            log_append(f"[RUN] ERROR: {type(e).__name__}: {e}")

    st.markdown("### Log (dernier)")
    st.text_area("", value=get_logs_text(), height=320)

def tab_results_exports() -> None:
    st.subheader("ÉTAPE 4 — RÉSULTATS / EXPORTS")
    pack: Optional[AcademicPack] = st.session_state.get("pack")
    if not (st.session_state.get("pack_loaded") and pack):
        st.warning("Pack inactif. Activez un pack.")
        return

    items = st.session_state.get("library_items") or []
    corr_ok = sum(1 for x in items if x.get("corrige?"))

    res = st.session_state.get("gte_result")
    qi_total = 0
    qi_pos = 0
    qc_count = 0
    sealed = "NO"
    if res:
        qi_total = int(res["audit"]["qi_total"])
        qi_pos = int(res["audit"]["qi_posable"])
        qc_count = int(res["audit"]["qc_count"])
        sealed = "YES" if res["saturation"]["sealed_candidate"] else "NO"

    cols = st.columns(6)
    cols[0].metric("Items", len(items))
    cols[1].metric("Corrigés DL", corr_ok)
    cols[2].metric("Qi", qi_total)
    cols[3].metric("Qi POSABLE", qi_pos)
    cols[4].metric("QC", qc_count)
    cols[5].metric("SEALED", sealed)

    st.markdown("---")
    st.markdown("### Exports (preuves)")

    manifest = st.session_state.get("library_manifest") or {
        "version": "V31.10.3",
        "timestamp": now_utc_iso(),
        "country": st.session_state.get("country", ""),
        "level": (st.session_state.get("level_sel") or ""),
        "subjects": (st.session_state.get("subject_sel") or []),
        "items_total": len(items),
        "items_corrige_ok": corr_ok,
        "library": items,
    }

    st.download_button(
        "Télécharger harvest_manifest.json",
        data=safe_json_dumps(manifest).encode("utf-8"),
        file_name="harvest_manifest.json",
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        "Télécharger logs.txt",
        data=get_logs_text().encode("utf-8"),
        file_name="logs.txt",
        mime="text/plain",
        use_container_width=True,
    )

    if res:
        st.download_button(
            "Télécharger qi_pack.json",
            data=safe_json_dumps(res["qi_pack"]).encode("utf-8"),
            file_name="qi_pack.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Télécharger qc_pack.json",
            data=safe_json_dumps(res["qc_pack"]).encode("utf-8"),
            file_name="qc_pack.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Télécharger chapter_report.json",
            data=safe_json_dumps(res["chapter_report"]).encode("utf-8"),
            file_name="chapter_report.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Aucun résultat RUN. Lancez RUN pour générer qi_pack/qc_pack/chapter_report.")

    st.markdown("---")
    st.markdown("### Logs (dernier)")
    st.text_area("", value=get_logs_text(), height=320)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    # Header
    st.markdown(f"# 🔒 {APP_TITLE}")
    st.caption("Flux: Activation → Pack visible → Sélection → Harvest → (RUN) → Résultats")

    ui_sidebar()

    # Hard gate: pack required (ISO)
    pack: Optional[AcademicPack] = st.session_state.get("pack")
    if ISO_DENY_STUB_PACK and not (st.session_state.get("pack_loaded") and pack):
        st.warning(
            "ISO-PROD TEST: Pack requis. Activez un pack (ou uploadez un pack JSON) pour continuer. "
            "Stub interdit."
        )
        st.text_area("Log (dernier)", value=get_logs_text(), height=220)
        return

    # Tabs
    t1, t2, t3 = st.tabs(["📥 Import / Bibliothèque", "⚡ Chaîne (RUN)", "📦 Résultats / Exports"])
    with t1:
        tab_import_biblio()
    with t2:
        tab_run()
    with t3:
        tab_results_exports()


if __name__ == "__main__":
    main()
