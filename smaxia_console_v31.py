"""
smaxia_gte_v32_04_iso_fullauto.py
SMAXIA GTE — Full Auto Harness (NO manual file creation)

Objectif:
- Harvest réel (APMEP) => URLs PDF + hash URL
- Appel du kernel scellé V10.6.2 via run_granulo_test(urls, volume)
- UI: QC par "chapitre" (champ libre) + Audit binaire orphelins
- Export preuve (JSON) sans upload requis

Notes d'ingénierie:
- Streamlit cache_data pour I/O web (recommandé pour API calls). :contentReference[oaicite:1]{index=1}
- requests timeout en tuple (connect, read). :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import json
import time
import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# =========================
# CORE (non-métier)
# =========================
APP_TITLE = "SMAXIA GTE – ISO Full Auto (V32.04.00)"
APP_VERSION = "V32.04.00"
USER_AGENT = "SMAXIA-GTE/1.0"
REQ_TIMEOUT = (5.0, 20.0)  # connect, read :contentReference[oaicite:3]{index=3}
RUN_DIR = Path("./_gte_runs")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Seeds "réels" (test harness) — APMEP annales
# (Vous pouvez en ajouter sans hardcoder de logique métier: ce sont des URLs de seeds)
DEFAULT_SEEDS = [
    "https://www.apmep.fr/Annales-du-Bac-Terminale",  # page d'index annales :contentReference[oaicite:4]{index=4}
    "https://www.apmep.fr/Annee-2025",                # page année 2025 :contentReference[oaicite:5]{index=5}
]

PDF_EXT = ".pdf"

# =========================
# PREUVE / STRUCTURES
# =========================
@dataclass
class HarvestItem:
    seed: str
    url: str
    url_sha256: str

@dataclass
class ProofSnapshot:
    timestamp: str
    seeds: List[str]
    harvested_count: int
    harvested_urls: List[str]
    harvested_url_hashes: List[str]
    kernel_import_ok: bool
    verdict: str
    audit: Dict[str, Any]

# =========================
# UTILS déterministes
# =========================
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def domain(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0]

def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

# =========================
# FETCH + PARSE (cache)
# =========================
@st.cache_data(ttl=3600)
def fetch_html(url: str) -> Tuple[int, str]:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=REQ_TIMEOUT, allow_redirects=True)
    return r.status_code, r.text

def extract_pdf_links(seed_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absu = urljoin(seed_url, href)
        if absu.lower().endswith(PDF_EXT):
            links.append(absu)
    return sorted(set(links))

def harvest_from_seeds(seeds: List[str], max_pdfs: int, allow_domains: Optional[List[str]] = None) -> List[HarvestItem]:
    items: List[HarvestItem] = []
    for seed in seeds:
        try:
            status, html = fetch_html(seed)
            if status >= 400:
                continue
            pdfs = extract_pdf_links(seed, html)

            if allow_domains:
                pdfs = [u for u in pdfs if any(domain(u) == d or domain(u).endswith("." + d) for d in allow_domains)]

            for u in pdfs:
                items.append(HarvestItem(seed=seed, url=u, url_sha256=sha256_text(u)))
        except Exception:
            continue

    # Dédup + ordre déterministe
    dedup: Dict[str, HarvestItem] = {}
    for it in items:
        dedup[it.url] = it
    out = sorted(dedup.values(), key=lambda x: (x.seed, x.url))
    return out[:max_pdfs]

# =========================
# KERNEL CALL (V10.6.2)
# =========================
def try_import_kernel() -> Tuple[bool, Optional[Any], str]:
    try:
        # Votre module scellé (déjà cité dans votre contexte projet)
        import smaxia_granulo_engine_test as kernel
        if not hasattr(kernel, "run_granulo_test"):
            return False, None, "Module importé mais run_granulo_test introuvable"
        return True, kernel, "OK"
    except Exception as e:
        return False, None, repr(e)

def run_kernel(kernel_mod: Any, urls: List[str], volume: int) -> Dict[str, Any]:
    # On n'interprète pas la “science” du kernel ici: on l'appelle.
    return kernel_mod.run_granulo_test(urls, volume)

# =========================
# UI
# =========================
def sidebar_inputs() -> None:
    st.sidebar.header("Paramètres (FULL AUTO)")

    st.session_state.chapter = st.sidebar.text_input("Chapitre (libre, pour affichage)", value="(non imposé)")
    st.session_state.volume = st.sidebar.slider("Volume (itérations / saturation)", 1, 10, 2)

    st.session_state.max_pdfs = st.sidebar.slider("Max PDFs à harvest", 1, 50, 12)

    st.session_state.allow_domain_apmep_only = st.sidebar.checkbox("Limiter domaine à apmep.fr", value=True)
    st.session_state.extra_seeds = st.sidebar.text_area(
        "Seeds additionnels (1 URL/ligne) — optionnel",
        value="",
        help="Aucune création manuelle de fichiers. Juste des URLs si vous voulez élargir."
    ).strip()

    if st.sidebar.button("RUN GTE (AUTO)", type="primary"):
        st.session_state.run = True
        st.rerun()

def display_harvest(items: List[HarvestItem]) -> None:
    st.subheader("PHASE 1 — Harvest (réel)")
    st.metric("PDFs trouvés", len(items))
    if items:
        st.dataframe(pd.DataFrame([asdict(i) for i in items]), use_container_width=True)

def display_kernel_result(result: Dict[str, Any]) -> None:
    st.subheader("PHASE 2 — Kernel V10.6.2 (run_granulo_test)")

    # Affichage safe (sans supposer la structure exacte au-delà de ce que vous avez indiqué)
    sujets = result.get("sujets", None)
    qc = result.get("qc", None)
    saturation = result.get("saturation", None)
    audit = result.get("audit", {})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sujets", len(sujets) if isinstance(sujets, list) else (1 if sujets else 0))
    with c2:
        st.metric("QC", len(qc) if isinstance(qc, list) else (len(qc) if isinstance(qc, dict) else (1 if qc else 0)))
    with c3:
        st.metric("Saturation", str(saturation)[:40] if saturation is not None else "n/a")

    st.write("Audit (brut):")
    st.json(audit)

    st.subheader("PHASE 3 — QC par chapitre (affichage)")
    # Votre kernel est censé ranger par chapitre via intent_code->chapter_code (selon votre V31.9.10).
    # Ici, on fait un affichage robuste:
    # - si qc est une dict {chapter: [qc...]} on affiche par clé
    # - sinon, on affiche la liste brute
    if isinstance(qc, dict):
        for ch, qlist in qc.items():
            with st.expander(f"Chapitre: {ch} — QC={len(qlist) if isinstance(qlist, list) else 'n/a'}", expanded=False):
                st.json(qlist)
    else:
        with st.expander(f"Chapitre (UI): {st.session_state.chapter}", expanded=True):
            st.json(qc)

def audit_orphans_strict(result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Règle centrale (binaire) selon votre spécification:
    Existe-t-il une Qi posable qui ne converge vers AUCUNE QC ?
    - Si le kernel fournit déjà un audit orphelins => on le respecte.
    - Sinon, verdict = FAIL (pas de preuve).
    """
    audit = result.get("audit", {}) if isinstance(result, dict) else {}
    # Heuristiques d'acceptation: le kernel doit fournir un champ explicitement auditable.
    # Noms possibles: "orphans", "qi_orphans", "uncovered_qi", "coverage_map" etc.
    if isinstance(audit, dict):
        for k in ["orphans", "qi_orphans", "uncovered_qi"]:
            if k in audit:
                orphans = audit.get(k) or []
                return ("SEALED" if len(orphans) == 0 else "FAIL", {"orphans_key": k, "orphans": orphans})

        # coverage_map: {qi_id: qc_id}
        if "coverage_map" in audit and isinstance(audit["coverage_map"], dict):
            # si le kernel fournit aussi la liste des qi posables
            qi_list = audit.get("posables_qi_ids") or audit.get("qi_ids") or None
            if isinstance(qi_list, list):
                covered = set(audit["coverage_map"].keys())
                orphans = [qi for qi in qi_list if qi not in covered]
                return ("SEALED" if len(orphans) == 0 else "FAIL", {"orphans_key": "coverage_map", "orphans": orphans})

    # Pas de preuve explicite => FAIL
    return ("FAIL", {"reason": "Kernel audit ne fournit pas de preuve explicite d'absence d'orphelins."})

def export_proof(seeds: List[str], items: List[HarvestItem], kernel_ok: bool, verdict: str, audit: Dict[str, Any]) -> None:
    st.subheader("Export preuve (JSON)")
    proof = ProofSnapshot(
        timestamp=now_iso(),
        seeds=seeds,
        harvested_count=len(items),
        harvested_urls=[i.url for i in items],
        harvested_url_hashes=[i.url_sha256 for i in items],
        kernel_import_ok=kernel_ok,
        verdict=verdict,
        audit=audit,
    )
    st.download_button(
        "EXPORT JSON PREUVE",
        data=json.dumps(asdict(proof), indent=2, ensure_ascii=False),
        file_name=f"smaxia_gte_proof_{int(time.time())}.json",
        mime="application/json",
    )

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"{APP_TITLE} — {APP_VERSION}")

    sidebar_inputs()
    if not st.session_state.get("run"):
        st.info("Cliquez RUN GTE (AUTO) pour lancer le pipeline.")
        st.caption("Sources de test: annales APMEP (pages listant sujets/corrigés PDF).")
        return

    # Seeds
    seeds = list(DEFAULT_SEEDS)
    if st.session_state.extra_seeds:
        seeds += [l.strip() for l in st.session_state.extra_seeds.splitlines() if l.strip()]
    seeds = sorted(set(seeds))

    allow_domains = ["apmep.fr"] if st.session_state.allow_domain_apmep_only else None

    # Phase 1 — Harvest
    items = harvest_from_seeds(seeds=seeds, max_pdfs=st.session_state.max_pdfs, allow_domains=allow_domains)
    display_harvest(items)

    if not items:
        st.error("HARVEST FAIL: aucun PDF trouvé sur les seeds.")
        export_proof(seeds, items, kernel_ok=False, verdict="FAIL", audit={"reason": "no_pdfs"})
        return

    urls = [i.url for i in items]

    # Phase 2 — Kernel
    kernel_ok, kernel_mod, kernel_msg = try_import_kernel()
    st.subheader("Kernel import")
    st.write({"kernel_import_ok": kernel_ok, "message": kernel_msg})

    if not kernel_ok:
        st.error("Impossible d'appeler le kernel scellé: import KO. Verdict=FAIL.")
        export_proof(seeds, items, kernel_ok=False, verdict="FAIL", audit={"reason": "kernel_import_fail", "message": kernel_msg})
        return

    # Run kernel
    with st.spinner("Exécution kernel V10.6.2 (run_granulo_test)..."):
        result = run_kernel(kernel_mod, urls=urls, volume=st.session_state.volume)

    display_kernel_result(result)

    # Phase 3 — Audit binaire orphelins
    verdict, audit_detail = audit_orphans_strict(result)
    st.subheader("VERDICT (binaire central: orphelins)")
    st.metric("VERDICT", verdict)
    st.json(audit_detail)

    export_proof(seeds, items, kernel_ok=True, verdict=verdict, audit=audit_detail)

if __name__ == "__main__":
    main()
