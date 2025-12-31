# =============================================================================
# SMAXIA GTE Console V31.10.2 — ISO-PROD TEST (UI + Harvest fixes)
# =============================================================================
# Objectif ISO-PROD (TEST) : Chaîne complète + preuves exportables
# - Activation (pack)
# - Sélection Niveau + Matière
# - Bibliothèque visible (pairs Sujet/Corrigé) + Upload manuel
# - Harvest AUTO (crawl multi-pages) -> remplit Bibliothèque
# - RUN (placeholder / hook moteur externe si présent)
# - Exports: harvest_manifest.json + logs.txt (+ placeholders qi/qc si moteur absent)
#
# NOTES IMPORTANTES
# - Aucun hardcode "métier" dans Core : ceci est une console TEST (UI/Harvest).
# - Les URL sources appartiennent au pack/config TEST (peuvent être remplacées par DB/API en PROD).
# - Le bloc "Domaines (pack-driven)" correspond à des regroupements (pas des chapitres programme).
#   Les "chapitres réels" ne sont affichés que si le pack les expose explicitement.
# =============================================================================

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

APP_VERSION = "V31.10.2"
APP_TITLE = f"🔐 SMAXIA GTE Console {APP_VERSION} — ISO-PROD TEST"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
REQ_TIMEOUT = 20

# -----------------------------
# Session state helpers
# -----------------------------
def ss_init():
    st.session_state.setdefault("pack", None)
    st.session_state.setdefault("country", "FR")
    st.session_state.setdefault("level", None)
    st.session_state.setdefault("subjects", [])
    st.session_state.setdefault("library", [])  # list[HarvestItem]
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("last_run", None)  # results dict
    st.session_state.setdefault("sealed", False)

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    st.session_state["logs"].append(line)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name).strip()
    return name[:160] if len(name) > 160 else name

# -----------------------------
# Pack model (TEST)
# -----------------------------
def load_academic_pack(country: str) -> Dict[str, Any]:
    """
    TEST loader. En PROD : cette fonction doit appeler DB/API et charger le Country Academic Pack.
    Ici : configuration minimale + sources Harvest corrigées (APMEP).
    """
    if country != "FR":
        raise ValueError("TEST: seul FR est activé dans cette console.")

    # Sources Harvest (corrigées)
    # IMPORTANT : on évite /Annales-Terminale (404) ; on vise des pages existantes.
    sources_by_level = {
        "SECONDE": [
            # On laisse vide si non défini ; l'utilisateur peut upload manuel.
        ],
        "PREMIERE": [
            # La structure APMEP varie ; on laisse vide par défaut en TEST.
        ],
        "TERMINALE": [
            {
                "name": "APMEP — Terminale Générale",
                "start_url": "https://www.apmep.fr/Annales-Terminale-Generale",
                "max_pages_default": 12,
            },
            {
                "name": "APMEP — Terminale Technologique",
                "start_url": "https://www.apmep.fr/Annales-Terminale-Technologique",
                "max_pages_default": 8,
            },
        ],
    }

    pack = {
        "pack_id": "CAP_FR_BAC_2024_V1",
        "country": "FR",
        "signature": "sha256:TEST_ONLY/TEST_ONLY",
        "levels": [
            {"code": "SECONDE", "label": "Seconde"},
            {"code": "PREMIERE", "label": "Première"},
            {"code": "TERMINALE", "label": "Terminale"},
            {"code": "L1", "label": "Licence 1"},
            {"code": "CPGE", "label": "Prépa (CPGE)"},
        ],
        "subjects": [
            {"code": "MATH", "label": "Mathématiques"},
        ],

        # Regroupements (domaines), pas "chapitres programme"
        "chapters_intent_allowlist": [
            {"code": "CH_ANALYSE", "label": "Analyse"},
            {"code": "CH_PROBAS", "label": "Probabilités"},
            {"code": "CH_GEOMETRIE", "label": "Géométrie"},
            {"code": "CH_SUITES", "label": "Suites & récurrence"},
        ],

        # Si votre vrai academic_pack expose une liste de chapitres, vous pouvez la charger ici.
        # Exemple (facultatif) :
        # "chapters": [{"code":"CH_FONCTIONS","label":"Fonctions"}, ...]
        #
        "harvest_config": {
            "sources_by_level": sources_by_level,
        },
    }
    return pack

def get_real_chapters(pack: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Détecte une liste de "chapitres réels" si le pack les expose.
    On teste plusieurs clés probables sans casser l'UI.
    """
    # 1) direct
    for k in ["chapters", "chapter_catalog", "chapters_catalog", "chapitre_catalogue", "chapter_list"]:
        v = pack.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict) and ("code" in v[0] or "id" in v[0]):
            out = []
            for it in v:
                code = it.get("code") or it.get("id") or ""
                label = it.get("label") or it.get("name") or code
                if code:
                    out.append({"code": str(code), "label": str(label)})
            return out if out else None

    # 2) nested candidates
    nested_paths = [
        ("academic_pack", "chapters"),
        ("academic_pack", "chapter_catalog"),
        ("pack", "chapters"),
        ("data", "chapters"),
    ]
    for path in nested_paths:
        cur = pack
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, list) and cur:
            out = []
            for it in cur:
                if not isinstance(it, dict):
                    continue
                code = it.get("code") or it.get("id") or ""
                label = it.get("label") or it.get("name") or code
                if code:
                    out.append({"code": str(code), "label": str(label)})
            return out if out else None

    return None

# -----------------------------
# Harvest model
# -----------------------------
@dataclass
class HarvestItem:
    pair_id: str
    scope: str                      # ex: TERMINALE|MATH
    source_name: str                # ex: APMEP — Terminale Générale
    sujet_name: str
    sujet_url: str
    corrige_name: Optional[str] = None
    corrige_url: Optional[str] = None
    corrige_found: bool = False
    reason: str = ""                # pourquoi corrigé introuvable / pairing incertain
    # pour upload manuel (optionnel)
    sujet_bytes_sha256: Optional[str] = None
    corrige_bytes_sha256: Optional[str] = None

def normalize_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"%[0-9a-f]{2}", "", s)
    s = re.sub(r"corrig[eé]|correction|corrige|corrig|corr\.?", "", s)
    s = re.sub(r"[\W_]+", " ", s)
    s = re.sub(r"\b(pdf|doc|docx)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]

def is_pdf(href: str) -> bool:
    return href.lower().split("?")[0].endswith(".pdf")

def is_correction(href: str, text: str) -> bool:
    blob = (href + " " + text).lower()
    return any(k in blob for k in ["corrig", "correction", "corrige", "corrigé", "corrigée", "corr."])

def same_domain(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc == urlparse(b).netloc
    except Exception:
        return False

def fetch_html(url: str) -> str:
    headers = {"User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.text

def crawl_collect_pdfs(start_url: str, max_pages: int) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Crawl simple multi-pages :
    - Visite start_url + liens internes pertinents (même domaine)
    - Collecte liens PDF (href, anchor_text)
    Retour : (pdf_links, visited_pages)
    """
    visited: Set[str] = set()
    queue: List[str] = [start_url]
    pdfs: List[Tuple[str, str]] = []

    def should_enqueue(u: str) -> bool:
        if not same_domain(start_url, u):
            return False
        # Limite : on garde les pages APMEP "Annales" et pages proches
        p = urlparse(u).path.lower()
        return ("annales" in p) or ("epreuve" in p) or ("bac" in p)

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            html = fetch_html(url)
        except Exception as e:
            log(f"[HARVEST] WARN fetch failed: {url} :: {e}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            text = (a.get_text() or "").strip()
            if not href:
                continue
            abs_u = urljoin(url, href)
            # collect pdf
            if is_pdf(abs_u):
                pdfs.append((abs_u, text))
                continue
            # enqueue internal pages
            if abs_u.startswith("http") and should_enqueue(abs_u) and abs_u not in visited:
                queue.append(abs_u)

    # de-dup
    seen = set()
    uniq = []
    for u, t in pdfs:
        key = u.split("#")[0]
        if key in seen:
            continue
        seen.add(key)
        uniq.append((u, t))
    return uniq, sorted(list(visited))

def build_pairs_from_pdfs(scope: str, source_name: str, pdfs: List[Tuple[str, str]], limit_pairs: int) -> List[HarvestItem]:
    """
    Pairing heuristique :
    - Sépare sujets vs corrigés
    - Tente d’associer par "clé normalisée" du nom/url
    - Sinon : sujet sans corrigé (corrige_found=False)
    """
    sujets: List[Tuple[str, str]] = []
    corrs: List[Tuple[str, str]] = []

    for u, t in pdfs:
        name = os.path.basename(urlparse(u).path) or "document.pdf"
        if is_correction(u, t):
            corrs.append((u, name))
        else:
            sujets.append((u, name))

    corr_by_key: Dict[str, List[Tuple[str, str]]] = {}
    for u, name in corrs:
        k = normalize_key(name + " " + u)
        corr_by_key.setdefault(k, []).append((u, name))

    out: List[HarvestItem] = []
    i = 0
    for u, name in sujets:
        if i >= limit_pairs:
            break
        i += 1
        k = normalize_key(name + " " + u)

        corr_choice = None
        # match direct key
        if k in corr_by_key and corr_by_key[k]:
            corr_choice = corr_by_key[k].pop(0)
        else:
            # fallback : match partiel (clé incluse)
            best = None
            for ck, lst in corr_by_key.items():
                if not lst:
                    continue
                if k and (k in ck or ck in k):
                    best = (ck, lst[0])
                    break
            if best:
                ck, (cu, cn) = best
                corr_choice = (cu, cn)
                corr_by_key[ck] = corr_by_key[ck][1:]

        pair_id = f"PAIR_{scope}_{i:04d}_{sha256_bytes((u+name).encode('utf-8'))[:7]}"
        item = HarvestItem(
            pair_id=pair_id,
            scope=scope,
            source_name=source_name,
            sujet_name=name,
            sujet_url=u,
        )
        if corr_choice:
            cu, cn = corr_choice
            item.corrige_name = cn
            item.corrige_url = cu
            item.corrige_found = True
        else:
            item.corrige_found = False
            item.reason = "corrigé introuvable (heuristique)"
        out.append(item)

    return out

def harvest_auto(pack: Dict[str, Any], level_code: str, subject_code: str, max_pages: int, volume: int) -> List[HarvestItem]:
    """
    Exécute Harvest AUTO sur les sources du pack pour ce niveau.
    """
    scope = f"{level_code}|{subject_code}"
    cfg = (pack.get("harvest_config") or {}).get("sources_by_level") or {}
    sources = cfg.get(level_code) or []
    if not sources:
        log(f"[HARVEST] Aucune source pack définie pour {level_code}. Utilisez Upload manuel.")
        return []

    all_items: List[HarvestItem] = []

    for src in sources:
        src_name = src.get("name", "SOURCE")
        start_url = src.get("start_url", "")
        if not start_url:
            continue

        log(f"[HARVEST] Source: {src_name}")
        log(f"[HARVEST] start={start_url} | max_pages={max_pages}")

        try:
            pdfs, visited = crawl_collect_pdfs(start_url, max_pages=max_pages)
        except Exception as e:
            log(f"[HARVEST] ERROR crawl failed: {e}")
            continue

        log(f"[HARVEST] pages_visited={len(visited)} | pdf_links={len(pdfs)}")

        pairs = build_pairs_from_pdfs(scope=scope, source_name=src_name, pdfs=pdfs, limit_pairs=volume)
        log(f"[HARVEST] pairs_built={len(pairs)} (corrigé si trouvé)")

        all_items.extend(pairs)

    # de-dup pair_id
    seen = set()
    uniq = []
    for it in all_items:
        if it.pair_id in seen:
            continue
        seen.add(it.pair_id)
        uniq.append(it)

    return uniq

# -----------------------------
# Optional hook to external engine
# -----------------------------
def run_engine_placeholder(items: List[HarvestItem]) -> Dict[str, Any]:
    """
    Si un moteur externe est disponible, vous pouvez l'appeler ici.
    On reste neutre : pas de logique propriétaire exposée.
    """
    # Hook optionnel
    try:
        from smaxia_granulo_engine_test import run_granulo_test  # type: ignore
        # On passe une liste de tuples (sujet_url, corrige_url) pour test
        urls = [(it.sujet_url, it.corrige_url) for it in items]
        res = run_granulo_test(urls, volume=len(urls))  # signature supposée
        return {"mode": "external_engine", "result": res}
    except Exception as e:
        # Fallback minimal : statistiques uniquement
        corr_ok = sum(1 for it in items if it.corrige_found)
        return {
            "mode": "stats_only",
            "error": str(e),
            "stats": {
                "items": len(items),
                "corriges_dl": corr_ok,
                "qi": 0,
                "qi_posable": 0,
                "qc": 0,
            }
        }

# -----------------------------
# UI
# -----------------------------
def sidebar(pack: Optional[Dict[str, Any]]):
    st.sidebar.markdown("## ÉTAPE 1 — ACTIVATION\n**PAYS**")
    country = st.sidebar.selectbox("Pays (TEST)", ["FR"], index=0, key="country")

    if st.sidebar.button("🔐 ACTIVER", use_container_width=True):
        try:
            st.session_state["pack"] = load_academic_pack(country)
            st.session_state["library"] = []
            st.session_state["last_run"] = None
            st.session_state["sealed"] = False
            log(f"=== PACK ACTIVÉ {st.session_state['pack']['pack_id']} ===")
        except Exception as e:
            st.session_state["pack"] = None
            log(f"[ACTIVATION] ERROR: {e}")

    st.sidebar.markdown("---")
    if pack:
        st.sidebar.success("Pack actif")
        st.sidebar.write(pack.get("pack_id", ""))
        st.sidebar.caption(f"Signature: {pack.get('signature','')}")
    else:
        st.sidebar.info("Activez un pays.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ÉTAPE 2 — SÉLECTION")
    if not pack:
        st.sidebar.caption("Activez le pack d'abord.")
        return

    level_labels = [f"{x['label']} ({x['code']})" for x in pack.get("levels", [])]
    level_codes = [x["code"] for x in pack.get("levels", [])]
    if level_codes:
        chosen = st.sidebar.radio("Niveau", options=list(range(len(level_codes))), format_func=lambda i: pack["levels"][i]["label"])
        st.session_state["level"] = level_codes[chosen]
    else:
        st.session_state["level"] = None

    subj_opts = pack.get("subjects", [])
    subj_codes = [x["code"] for x in subj_opts]
    subj_labels = [x["label"] for x in subj_opts]
    if subj_codes:
        st.session_state["subjects"] = st.sidebar.multiselect(
            "Matières (1–2 recommandé pour test)",
            options=subj_codes,
            default=["MATH"],
            format_func=lambda c: next((x["label"] for x in subj_opts if x["code"] == c), c),
        )

    st.sidebar.markdown("---")
    # Domaines (pack-driven)
    st.sidebar.markdown("## Domaines (pack-driven)")
    for it in pack.get("chapters_intent_allowlist", []):
        st.sidebar.caption(f"• {it.get('code','')} — {it.get('label','')}")

    # Chapitres réels si présents
    real_chaps = get_real_chapters(pack)
    if real_chaps:
        st.sidebar.markdown("## Chapitres (réels — pack)")
        for it in real_chaps[:18]:
            st.sidebar.caption(f"• {it['code']} — {it['label']}")
        if len(real_chaps) > 18:
            st.sidebar.caption(f"(+{len(real_chaps)-18} autres…)")

def kpi_row(items: List[HarvestItem], last_run: Optional[Dict[str, Any]], sealed: bool):
    corr_ok = sum(1 for it in items if it.corrige_found)
    qi = qc = qi_posable = 0
    if last_run:
        if last_run.get("mode") == "external_engine":
            # si le moteur renvoie des stats (optionnel)
            pass
        else:
            stats = (last_run.get("stats") or {})
            qi = int(stats.get("qi", 0))
            qc = int(stats.get("qc", 0))
            qi_posable = int(stats.get("qi_posable", 0))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Items", len(items))
    c2.metric("Corrigés DL", corr_ok)
    c3.metric("Qi", qi)
    c4.metric("Qi POSABLE", qi_posable)
    c5.metric("QC", qc)
    c6.metric("SEALED", "YES" if sealed else "NO")

def library_table(items: List[HarvestItem]):
    if not items:
        st.info("Aucun item en bibliothèque. Lancez HARVEST AUTO ou faites un upload manuel.")
        return
    rows = []
    for it in items:
        rows.append({
            "pair_id": it.pair_id,
            "scope": it.scope,
            "source": it.source_name,
            "sujet": it.sujet_name,
            "corrigé?": "✅" if it.corrige_found else "❌",
            "corrigé_name": it.corrige_name or "",
            "reason": it.reason or "",
            "sujet_url": it.sujet_url,
            "corrige_url": it.corrige_url or "",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

def export_buttons(pack: Optional[Dict[str, Any]], items: List[HarvestItem]):
    st.subheader("Exports (preuves)")
    manifest = {
        "app_version": APP_VERSION,
        "timestamp": datetime.now().isoformat(),
        "pack_id": (pack or {}).get("pack_id") if pack else None,
        "items": [asdict(it) for it in items],
    }
    st.download_button(
        "harvest_manifest.json",
        data=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="harvest_manifest.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        "logs.txt",
        data=("\n".join(st.session_state["logs"])).encode("utf-8"),
        file_name="logs.txt",
        mime="text/plain",
        use_container_width=True,
    )

def main():
    ss_init()

    st.set_page_config(page_title=f"SMAXIA GTE {APP_VERSION}", layout="wide")
    st.title(APP_TITLE)
    st.caption("Flux: Activation → Pack visible → Sélection → Harvest → (RUN) → Résultats")

    pack = st.session_state.get("pack")
    sidebar(pack)

    pack = st.session_state.get("pack")
    level = st.session_state.get("level")
    subjects = st.session_state.get("subjects") or []
    items: List[HarvestItem] = st.session_state.get("library") or []
    last_run = st.session_state.get("last_run")
    sealed = bool(st.session_state.get("sealed"))

    tab1, tab2, tab3 = st.tabs(["📥 Import / Bibliothèque", "⚡ Chaîne complète", "📦 Résultats / Exports"])

    with tab1:
        st.header("Import PDF (sujets + corrigés)")
        kpi_row(items, last_run, sealed)

        st.subheader("Bibliothèque Harvest (visible)")
        only_missing_corr = st.checkbox("Afficher seulement les items sans corrigé exploitable (❌)", value=False)
        if only_missing_corr:
            library_table([it for it in items if not it.corrige_found])
        else:
            library_table(items)

        st.markdown("---")
        st.subheader("HARVEST AUTO (récolte) — paramètres")

        if not pack:
            st.warning("Activez le pack d'abord.")
        elif not level or not subjects:
            st.warning("Sélectionnez un Niveau + au moins une Matière.")
        else:
            c1, c2, c3 = st.columns([1, 1, 1])
            max_pages = c1.number_input("Max pages crawl (par source)", min_value=1, max_value=40, value=12, step=1)
            volume = c2.number_input("Volume (sujets par matière)", min_value=1, max_value=200, value=50, step=5)
            scope = f"{level}|{subjects[0]}" if subjects else ""
            c3.text_input("Scope (auto)", value=scope, disabled=True)

            if st.button("🌐 Lancer HARVEST AUTO (Bibliothèque)", use_container_width=True):
                st.session_state["sealed"] = False
                st.session_state["last_run"] = None
                new_items_all: List[HarvestItem] = []
                for sub in subjects:
                    log(f"=== HARVEST AUTO {APP_VERSION} ===")
                    log(f"[HARVEST] Scope={level}|{sub}")
                    new_items = harvest_auto(pack, level, sub, max_pages=int(max_pages), volume=int(volume))
                    new_items_all.extend(new_items)

                # merge without duplicates by sujet_url
                existing_urls = {it.sujet_url for it in items}
                merged = items[:]
                added = 0
                for it in new_items_all:
                    if it.sujet_url in existing_urls:
                        continue
                    merged.append(it)
                    existing_urls.add(it.sujet_url)
                    added += 1

                st.session_state["library"] = merged
                log(f"[BIBLIOTHÈQUE] +{added} items (total={len(merged)})")

                st.success(f"Harvest terminé : +{added} items ajoutés.")
                st.rerun()

        st.markdown("---")
        st.subheader("Upload manuel (optionnel)")
        colA, colB = st.columns(2)
        with colA:
            up_sujet = st.file_uploader("PDF Sujet", type=["pdf"], key="up_sujet")
        with colB:
            up_corr = st.file_uploader("PDF Correction (opt)", type=["pdf"], key="up_corr")

        if st.button("➕ Ajouter à la bibliothèque", use_container_width=True, disabled=not (pack and level and subjects and up_sujet)):
            sujet_bytes = up_sujet.read() if up_sujet else b""
            corr_bytes = up_corr.read() if up_corr else b""
            sujet_sha = sha256_bytes(sujet_bytes) if sujet_bytes else None
            corr_sha = sha256_bytes(corr_bytes) if corr_bytes else None

            pair_id = f"UPLOAD_{level}|{subjects[0]}_{(sujet_sha or 'na')[:10]}"
            it = HarvestItem(
                pair_id=pair_id,
                scope=f"{level}|{subjects[0]}",
                source_name="UPLOAD_MANUEL",
                sujet_name=safe_filename(up_sujet.name if up_sujet else "sujet.pdf"),
                sujet_url=f"upload://{sujet_sha}",
                corrige_name=safe_filename(up_corr.name) if up_corr else None,
                corrige_url=f"upload://{corr_sha}" if up_corr else None,
                corrige_found=bool(up_corr),
                reason="" if up_corr else "corrigé non fourni (upload)",
                sujet_bytes_sha256=sujet_sha,
                corrige_bytes_sha256=corr_sha,
            )
            st.session_state["library"] = (st.session_state["library"] or []) + [it]
            log(f"[UPLOAD] ajout {it.pair_id} corrigé={it.corrige_found}")
            st.success("Ajouté à la bibliothèque.")
            st.rerun()

    with tab2:
        st.header("ÉTAPE 3 — LANCER CHAÎNE COMPLÈTE")
        st.caption("ISO-PROD: on refuse de sceller si bibliothèque vide ou si aucun corrigé exploitable.")
        if not items:
            st.warning("Bibliothèque vide. Lancez HARVEST AUTO ou Upload manuel.")
        else:
            if st.button("⚡ LANCER (RUN)", use_container_width=True):
                log(f"=== DÉMARRAGE CHAÎNE {APP_VERSION} ===")
                log(f"Pays: {st.session_state.get('country')} | Niveau: {level} | Matières: {subjects}")
                corr_ok = sum(1 for it in items if it.corrige_found)
                if corr_ok == 0:
                    log("[RUN] STOP: aucun corrigé exploitable => ISO-PROD test non valide.")
                    st.session_state["last_run"] = {"mode": "stats_only", "stats": {"items": len(items), "corriges_dl": 0, "qi": 0, "qi_posable": 0, "qc": 0}}
                    st.session_state["sealed"] = False
                    st.error("Aucun corrigé exploitable : test ISO-PROD invalide (POSABLE=0 attendu).")
                    st.rerun()

                # Lancer moteur externe si dispo (sinon stats_only)
                res = run_engine_placeholder(items)
                st.session_state["last_run"] = res

                # Règle ISO-PROD (minimale ici) : scellé si corrigés présents + pas d'erreur
                if res.get("mode") == "external_engine":
                    st.session_state["sealed"] = True
                    log("[SEALED] OK (external_engine).")
                else:
                    # si moteur absent -> pas de seal
                    st.session_state["sealed"] = False
                    log("[SEALED] NO (moteur externe absent/erreur).")

                st.rerun()

        st.subheader("Log (dernier)")
        st.code("\n".join(st.session_state["logs"][-40:]) or "(vide)", language="text")

    with tab3:
        st.header("ÉTAPE 4 — RÉSULTATS / EXPORTS")
        kpi_row(items, last_run, sealed)

        if last_run and last_run.get("mode") == "stats_only" and last_run.get("error"):
            st.warning("Moteur externe indisponible (ou erreur d’import). Console en mode stats_only.")
            st.caption(f"Erreur: {last_run.get('error')}")

        export_buttons(pack, items)

        st.subheader("Logs (dernier)")
        st.code("\n".join(st.session_state["logs"][-40:]) or "(vide)", language="text")

if __name__ == "__main__":
    main()
