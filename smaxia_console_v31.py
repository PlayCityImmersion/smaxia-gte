# =============================================================================
# SMAXIA GTE Console V31.10.3 — ISO-PROD TEST (UI ÉPURÉE + HARVEST APMEP "PAIR BY ROW")
# =============================================================================
# OBJECTIF ISO-PROD :
# - ZÉRO faux SEALED
# - REFUS de RUN si : bibliothèque vide OU aucun corrigé exploitable
# - PREUVES exportables (manifest + logs + outputs moteur si dispo)
#
# CORRECTIF MAJEUR :
# - HARVEST APMEP ne "ramasse" plus des PDFs au hasard.
# - Il parse les pages "Année XXXX" et extrait les paires SUJET PDF ↔ CORRIGÉ PDF par ligne.
#
# NOTE INVARIANTS :
# - Le CORE SMAXIA ne doit pas hardcoder métier.
# - Ici, APMEP est une SOURCE TEST_ONLY dans la console Streamlit de test (acceptable).
# =============================================================================

from __future__ import annotations

import json
import re
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import streamlit as st

# Dépendances réseau / parsing
import requests
from bs4 import BeautifulSoup

# Dépendance PDF (optionnelle pour certains affichages)
try:
    import pdfplumber  # noqa: F401
    PDF_OK = True
except Exception:
    PDF_OK = False

# =============================================================================
# VERSION
# =============================================================================
APP_VERSION = "V31.10.3"
APP_TITLE = f"SMAXIA GTE Console {APP_VERSION} — ISO-PROD TEST"

# =============================================================================
# CONSTANTES TEST_ONLY (SOURCE)
# =============================================================================
APMEP_ROOT = "https://www.apmep.fr/Annales-du-Bac-Terminale"  # page index
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 SMAXIA-GTE/31.10.3"
REQ_TIMEOUT = 25
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.35


# =============================================================================
# UTILITAIRES
# =============================================================================
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def safe_get(url: str, logs: List[str]) -> Optional[str]:
    """GET robuste (retries)"""
    headers = {"User-Agent": UA}
    for i in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            time.sleep(SLEEP_BETWEEN)
            return r.text
        except Exception as e:
            logs.append(f"[HARVEST] WARN fetch failed ({i}/{MAX_RETRIES}): {url} :: {e}")
            time.sleep(0.8 * i)
    return None


def normalize_filename(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_\.]+", "", s)
    return s


def is_pdf_href(href: str) -> bool:
    return href.lower().endswith(".pdf") or ".pdf?" in href.lower()


def pick_year_links_from_root(html: str) -> List[Tuple[int, str]]:
    """
    Extrait les liens vers pages "Année 2024", "Année 2023", etc depuis la page racine APMEP.
    """
    soup = BeautifulSoup(html, "html.parser")
    years: List[Tuple[int, str]] = []
    for a in soup.find_all("a"):
        txt = (a.get_text(" ", strip=True) or "").strip()
        href = a.get("href") or ""
        m = re.search(r"\bAnn[ée]e\s+(20\d{2})\b", txt, flags=re.IGNORECASE)
        if m and href:
            y = int(m.group(1))
            full = urljoin(APMEP_ROOT, href)
            years.append((y, full))
    years = sorted(list(set(years)), key=lambda t: t[0], reverse=True)
    return years


def find_primary_table(soup: BeautifulSoup) -> Optional[Any]:
    """
    Sur pages Année XXXX, la section "Épreuves d’enseignement de spécialité" contient un tableau.
    On prend le premier tableau substantiel.
    """
    tables = soup.find_all("table")
    if not tables:
        return None
    # Heuristique : table dont l'entête contient "sujet" et "corrig"
    for t in tables:
        head_txt = t.get_text(" ", strip=True).lower()
        if "sujet" in head_txt and ("corrig" in head_txt or "corrige" in head_txt):
            return t
    # fallback
    return tables[0]


def parse_year_page_pairs(year: int, url: str, scope: str, logs: List[str]) -> List[Dict[str, Any]]:
    """
    Parse une page Année XXXX et retourne une liste d'items:
    - sujet_url
    - corrige_url
    par ligne du tableau (par colonnes "sujet PDF"/"corrigé PDF").
    """
    html = safe_get(url, logs)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    table = find_primary_table(soup)
    if table is None:
        logs.append(f"[HARVEST] WARN no table found on year page: {url}")
        return []

    # Lire entêtes (th) pour mapper les colonnes
    header_cells = table.find_all("th")
    headers = [(c.get_text(" ", strip=True) or "").lower() for c in header_cells]
    # Si pas de th, on tentera par positions
    # But: identifier index "sujet pdf" et "corrigé pdf"
    sujet_col_idx = None
    corr_col_idx = None
    if headers:
        for idx, h in enumerate(headers):
            if "sujet" in h and "pdf" in h:
                sujet_col_idx = idx
            if ("corrig" in h or "corrige" in h) and "pdf" in h:
                corr_col_idx = idx

    items: List[Dict[str, Any]] = []
    rows = table.find_all("tr")
    for tr in rows:
        tds = tr.find_all(["td", "th"])
        if not tds or (headers and tds == header_cells):
            continue

        def extract_pdf_link(cell) -> Optional[str]:
            for a in cell.find_all("a"):
                href = a.get("href") or ""
                if href and is_pdf_href(href):
                    return urljoin(url, href)
            return None

        sujet_url = None
        corrige_url = None

        if sujet_col_idx is not None and sujet_col_idx < len(tds):
            sujet_url = extract_pdf_link(tds[sujet_col_idx])
        if corr_col_idx is not None and corr_col_idx < len(tds):
            corrige_url = extract_pdf_link(tds[corr_col_idx])

        # fallback si colonnes non identifiées : prendre les 2 premiers liens pdf rencontrés (sujet puis corrigé)
        if not sujet_url or (corrige_url is None):
            pdfs = []
            for cell in tds:
                for a in cell.find_all("a"):
                    href = a.get("href") or ""
                    if href and is_pdf_href(href):
                        pdfs.append(urljoin(url, href))
            if not sujet_url and pdfs:
                sujet_url = pdfs[0]
            if corrige_url is None and len(pdfs) >= 2:
                corrige_url = pdfs[1]

        if not sujet_url:
            continue

        sujet_name = normalize_filename(sujet_url.split("/")[-1].split("?")[0])
        corr_name = normalize_filename(corrige_url.split("/")[-1].split("?")[0]) if corrige_url else ""

        # Filtre anti-bruit : exclure certains pdfs "index/compilation/explication"
        bad_tokens = ["explanation", "cfc", "1995-2020", "1998-2020", "index"]
        if any(tok in sujet_name for tok in bad_tokens):
            continue

        pair_id = f"PAIR_{scope}_{year}_{sha256_text(sujet_url)[:8]}"
        item = {
            "pair_id": pair_id,
            "scope": scope,
            "source": f"APMEP — Année {year}",
            "year": year,
            "sujet": sujet_name,
            "corrige?": bool(corrige_url),
            "corrige_name": corr_name if corrige_url else "",
            "reason": "" if corrige_url else "corrigé absent sur la ligne",
            "sujet_url": sujet_url,
            "corrige_url": corrige_url or "",
        }
        items.append(item)

    return items


def load_academic_pack(country: str, logs: List[str]) -> Dict[str, Any]:
    """
    Charge un academic pack depuis disque si disponible.
    Sinon retourne un stub minimal (TEST_ONLY).
    """
    # Chemins possibles (adaptez si besoin)
    candidates = [
        f"./academic_packs/{country}.json",
        f"./packs/{country}.json",
        f"./{country}_academic_pack.json",
    ]
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                pack = json.load(f)
                logs.append(f"[PACK] loaded: {p}")
                return pack
        except Exception:
            continue

    logs.append("[PACK] WARN pack file not found. Using stub pack.")
    return {
        "country": country,
        "signature": "TEST_ONLY",
        "levels": ["TERMinale"],
        "subjects": ["MATH"],
        "chapters": [],  # vide => avertissement UI
    }


def extract_leaf_chapters(pack: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Retourne une liste de chapitres 'leaf' si présents dans pack.
    On attend typiquement: pack["chapters"] = [{chapter_code,title,...}, ...]
    """
    ch = pack.get("chapters")
    if isinstance(ch, list) and ch and isinstance(ch[0], dict):
        out = []
        for c in ch:
            code = str(c.get("chapter_code") or c.get("code") or "").strip()
            title = str(c.get("title") or c.get("name") or "").strip()
            if code:
                out.append({"code": code, "title": title})
        return out
    return []


# =============================================================================
# MOTEUR (OPTIONNEL)
# =============================================================================
def try_import_engine() -> Tuple[bool, str]:
    try:
        from smaxia_granulo_engine_test import run_granulo_test  # type: ignore  # noqa
        return True, "ok"
    except Exception as e:
        return False, str(e)


def run_engine(urls: List[Dict[str, str]], volume: int) -> Dict[str, Any]:
    """
    Appel au moteur GTE existant si disponible.
    """
    from smaxia_granulo_engine_test import run_granulo_test  # type: ignore
    return run_granulo_test(urls=urls, volume=volume)


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")

if "logs" not in st.session_state:
    st.session_state.logs = []
if "pack" not in st.session_state:
    st.session_state.pack = None
if "pack_active" not in st.session_state:
    st.session_state.pack_active = False
if "country" not in st.session_state:
    st.session_state.country = "FR"
if "level" not in st.session_state:
    st.session_state.level = "TERMINALE"
if "subjects" not in st.session_state:
    st.session_state.subjects = ["MATH"]
if "library" not in st.session_state:
    st.session_state.library = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# Header
st.title(f"🔐 {APP_TITLE}")
st.caption("Flux: Activation → Pack visible → Sélection → Harvest → (RUN) → Résultats")

# Sidebar (essentiel)
with st.sidebar:
    st.subheader("ÉTAPE 1 — ACTIVATION PAYS")
    country = st.selectbox("Pays (TEST)", ["FR"], index=0)
    st.session_state.country = country

    if st.button("🔓 ACTIVER", use_container_width=True):
        st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] === PACK ACTIVATION ===")
        st.session_state.pack = load_academic_pack(country, st.session_state.logs)
        st.session_state.pack_active = True

    st.divider()
    if st.session_state.pack_active:
        sig = st.session_state.pack.get("signature", "n/a")
        st.success("Pack actif", icon="✅")
        st.caption(f"Signature: {sig}")
    else:
        st.info("Activez un pays.")

    st.subheader("ÉTAPE 2 — SÉLECTION")
    level = st.radio("Niveau", ["Seconde", "Première", "Terminale", "Licence 1", "Prépa (CPGE)"], index=2)
    st.session_state.level = "TERMINALE" if "Terminale" in level else level.upper()

    subj = st.multiselect("Matières (1–2 recommandé pour test)", ["MATH"], default=["MATH"])
    st.session_state.subjects = subj if subj else ["MATH"]

    st.subheader("Chapitres (pack-driven)")
    if st.session_state.pack_active and st.session_state.pack:
        leaf = extract_leaf_chapters(st.session_state.pack)
        if leaf:
            # Afficher une liste courte (et scroll)
            for c in leaf[:40]:
                label = f"{c['code']}" + (f" — {c['title']}" if c["title"] else "")
                st.caption(label)
            if len(leaf) > 40:
                st.caption(f"... ({len(leaf)} chapitres)")
        else:
            st.warning(
                "Le pack ne fournit pas de chapitres 'leaf' exploitables. "
                "ANALYSE/PROBAS/… sont souvent des groupes, pas des chapitres ISO.",
                icon="⚠️",
            )
    else:
        st.caption("Pack non activé.")

# Tabs (UI épurée)
tab1, tab2, tab3 = st.tabs(["📥 Import / Bibliothèque", "⚡ Chaîne (RUN)", "📦 Résultats / Exports"])

# ---- Tab 1: Import / Bibliothèque ----
with tab1:
    st.header("Import PDF (sujets + corrigés)")

    # Metrics essentiels
    items = len(st.session_state.library)
    corr_ok = sum(1 for x in st.session_state.library if x.get("corrige?"))
    st.write("")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Items", items)
    c2.metric("Corrigés exploitables", corr_ok)
    c3.metric("Qi", st.session_state.last_results["sujets"]["qi_total"] if st.session_state.last_results else 0)
    c4.metric("Qi POSABLE", st.session_state.last_results["sujets"]["qi_posable"] if st.session_state.last_results else 0)
    c5.metric("QC", st.session_state.last_results["qc"]["qc_total"] if st.session_state.last_results else 0)
    c6.metric("SEALED", "YES" if (st.session_state.last_results and st.session_state.last_results.get("sealed")) else "NO")

    st.subheader("Bibliothèque Harvest (visible)")

    only_no_corr = st.checkbox("Afficher seulement les items sans corrigé exploitable (❌)", value=False)
    view_rows = st.session_state.library
    if only_no_corr:
        view_rows = [r for r in view_rows if not r.get("corrige?")]

    if view_rows:
        st.dataframe(
            view_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Aucun item en bibliothèque. Lancez HARVEST AUTO ou faites un upload manuel.")

    st.markdown("---")
    st.subheader("HARVEST AUTO (APMEP) — paramètres")

    if not st.session_state.pack_active:
        st.warning("Activez d’abord le pack (Étape 1).")
    else:
        with st.form("harvest_form", clear_on_submit=False):
            colA, colB, colC = st.columns([1.2, 1.2, 1.2])
            years_back = colA.number_input("Nb d'années à récolter (depuis la plus récente)", min_value=1, max_value=15, value=10, step=1)
            max_items = colB.number_input("Volume max (items)", min_value=5, max_value=200, value=50, step=5)
            scope = colC.text_input("Scope (auto)", value=f"{st.session_state.level}|{st.session_state.subjects[0]}", disabled=True)

            go = st.form_submit_button("🌐 Lancer HARVEST AUTO (Bibliothèque)")

        if go:
            st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] === HARVEST AUTO {APP_VERSION} ===")
            st.session_state.logs.append(f"[HARVEST] scope={scope} root={APMEP_ROOT}")

            root_html = safe_get(APMEP_ROOT, st.session_state.logs)
            if not root_html:
                st.error("Impossible de charger la page APMEP root.")
            else:
                year_links = pick_year_links_from_root(root_html)
                if not year_links:
                    st.error("Aucun lien 'Année XXXX' trouvé sur la page APMEP root.")
                else:
                    # limiter aux N dernières années
                    year_links = year_links[: int(years_back)]
                    built = 0
                    new_items: List[Dict[str, Any]] = []
                    for y, yurl in year_links:
                        if built >= int(max_items):
                            break
                        st.session_state.logs.append(f"[HARVEST] year={y} url={yurl}")
                        pairs = parse_year_page_pairs(y, yurl, scope, st.session_state.logs)
                        for it in pairs:
                            if built >= int(max_items):
                                break
                            # dédup par sujet_url
                            if any(x.get("sujet_url") == it.get("sujet_url") for x in st.session_state.library + new_items):
                                continue
                            new_items.append(it)
                            built += 1

                    st.session_state.library.extend(new_items)
                    st.success(f"HARVEST terminé. +{len(new_items)} items ajoutés (total={len(st.session_state.library)}).")

    st.markdown("---")
    st.subheader("Upload manuel (optionnel)")

    up1, up2 = st.columns(2)
    sujet_file = up1.file_uploader("PDF Sujet", type=["pdf"], accept_multiple_files=False)
    corr_file = up2.file_uploader("PDF Correction (opt)", type=["pdf"], accept_multiple_files=False)

    add_manual = st.button("➕ Ajouter à la bibliothèque", use_container_width=True)
    if add_manual:
        if not sujet_file:
            st.error("Veuillez sélectionner un PDF Sujet.")
        else:
            sujet_name = normalize_filename(sujet_file.name)
            corr_name = normalize_filename(corr_file.name) if corr_file else ""
            # On stocke en mémoire (bytes) via session_state (acceptable en test)
            sujet_key = sha256_text(sujet_name + str(len(sujet_file.getvalue())))[:10]
            corr_key = sha256_text(corr_name + str(len(corr_file.getvalue())))[:10] if corr_file else ""

            scope = f"{st.session_state.level}|{st.session_state.subjects[0]}"
            pair_id = f"PAIR_{scope}_MANUAL_{sha256_text(sujet_name)[:8]}"
            item = {
                "pair_id": pair_id,
                "scope": scope,
                "source": "MANUAL_UPLOAD",
                "year": "",
                "sujet": sujet_name,
                "corrige?": bool(corr_file),
                "corrige_name": corr_name,
                "reason": "" if corr_file else "corrigé non fourni",
                "sujet_url": f"file://manual/{sujet_key}/{sujet_name}",
                "corrige_url": f"file://manual/{corr_key}/{corr_name}" if corr_file else "",
                "_manual_sujet_bytes": sujet_file.getvalue(),
                "_manual_corr_bytes": corr_file.getvalue() if corr_file else b"",
            }
            st.session_state.library.append(item)
            st.success("Ajouté à la bibliothèque.")

# ---- Tab 2: Chaîne (RUN) ----
with tab2:
    st.header("ÉTAPE 3 — LANCER CHAÎNE (RUN)")

    st.caption("ISO-PROD : refus de RUN si bibliothèque vide OU aucun corrigé exploitable.")

    engine_ok, engine_msg = try_import_engine()
    if not engine_ok:
        st.warning(f"Moteur non importable (run_granulo_test). Erreur: {engine_msg}")

    volume = st.number_input("Volume (items consommés)", min_value=1, max_value=200, value=20, step=1)

    run_btn = st.button("⚡ LANCER (RUN)", use_container_width=True)

    if run_btn:
        st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] === RUN {APP_VERSION} ===")

        if not st.session_state.library:
            st.error("Bibliothèque vide. Lancez HARVEST AUTO ou upload manuel.")
        else:
            usable = [x for x in st.session_state.library if x.get("corrige?") and x.get("corrige_url")]
            if not usable:
                st.error("Aucun corrigé exploitable. ISO-PROD refuse le RUN.")
            elif not engine_ok:
                st.error("Moteur indisponible. Ajoutez smaxia_granulo_engine_test.py (run_granulo_test).")
            else:
                # Construire la liste urls pour moteur (sujet + corrigé)
                urls = []
                for it in usable[: int(volume)]:
                    urls.append({"sujet_url": it["sujet_url"], "corrige_url": it["corrige_url"], "pair_id": it["pair_id"], "scope": it["scope"]})

                st.session_state.logs.append(f"[RUN] usable_pairs={len(urls)}")
                try:
                    out = run_engine(urls=urls, volume=int(volume))

                    # Normaliser quelques champs attendus
                    # out = {sujets,qc,saturation,audit} selon votre spec existante
                    sealed = False
                    if isinstance(out, dict):
                        sat = out.get("saturation") or {}
                        audit = out.get("audit") or {}
                        # Gating strict minimal (ISO-PROD): B_PASS + preuve stable + QC>0 + POSABLE>0
                        b_pass = bool(audit.get("B_ok", False))
                        new_proof_sig = int(sat.get("new_proof_sig", 1)) if isinstance(sat, dict) else 1
                        qi_posable = int((out.get("sujets") or {}).get("qi_posable", 0))
                        qc_total = int((out.get("qc") or {}).get("qc_total", 0))
                        sealed = bool(b_pass and new_proof_sig == 0 and qi_posable > 0 and qc_total > 0)

                    out["_meta"] = {"version": APP_VERSION, "timestamp": now_iso()}
                    out["sealed"] = sealed
                    st.session_state.last_results = out

                    st.success("RUN terminé. Consultez Résultats / Exports.")
                except Exception as e:
                    st.session_state.logs.append(f"[RUN] ERROR: {e}")
                    st.error(f"Erreur moteur: {e}")

    st.subheader("Log (dernier)")
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs[-120:]), language="text")
    else:
        st.info("Aucun log.")

# ---- Tab 3: Résultats / Exports ----
with tab3:
    st.header("ÉTAPE 4 — RÉSULTATS / EXPORTS")

    out = st.session_state.last_results
    if not out:
        st.info("Aucun résultat. Lancez d’abord un RUN.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        sujets = out.get("sujets") or {}
        qc = out.get("qc") or {}
        audit = out.get("audit") or {}
        sat = out.get("saturation") or {}

        c1.metric("Qi total", int(sujets.get("qi_total", 0)))
        c2.metric("Qi POSABLE", int(sujets.get("qi_posable", 0)))
        c3.metric("QC total", int(qc.get("qc_total", 0)))
        c4.metric("B PASS", "YES" if audit.get("B_ok") else "NO")
        c5.metric("SEALED", "YES" if out.get("sealed") else "NO")

        st.subheader("Résumé (audit / saturation)")
        st.json(
            {
                "audit": audit,
                "saturation": sat,
                "_meta": out.get("_meta"),
                "sealed": out.get("sealed"),
            }
        )

    st.markdown("---")
    st.subheader("Exports (preuves)")

    # 1) harvest_manifest.json (toujours dispo)
    manifest = {
        "version": APP_VERSION,
        "timestamp": now_iso(),
        "country": st.session_state.country,
        "level": st.session_state.level,
        "subjects": st.session_state.subjects,
        "items_total": len(st.session_state.library),
        "items_corrige_ok": sum(1 for x in st.session_state.library if x.get("corrige?")),
        "library": [
            {k: v for k, v in it.items() if not k.startswith("_manual_")}
            for it in st.session_state.library
        ],
    }
    st.download_button(
        "harvest_manifest.json",
        data=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="harvest_manifest.json",
        mime="application/json",
        use_container_width=True,
    )

    # 2) logs.txt
    st.download_button(
        "logs.txt",
        data=("\n".join(st.session_state.logs)).encode("utf-8"),
        file_name="logs.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # 3) sorties moteur si dispo
    if st.session_state.last_results:
        st.download_button(
            "qi_pack.json",
            data=json.dumps(st.session_state.last_results.get("sujets") or {}, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="qi_pack.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "qc_pack.json",
            data=json.dumps(st.session_state.last_results.get("qc") or {}, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="qc_pack.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "chapter_report.json",
            data=json.dumps(st.session_state.last_results.get("chapter_report") or {}, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="chapter_report.json",
            mime="application/json",
            use_container_width=True,
        )
