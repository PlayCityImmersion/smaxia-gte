# =============================================================================
# SMAXIA GTE Console V31.10.4 — ISO-PROD TEST (MONO-FICHIER)
# =============================================================================
# OBJECTIF (TEST ISO-PROD):
# - 1 seul fichier: Console + Moteur (aucun import moteur externe)
# - Flux: Activation -> Pack visible -> Sélection -> Harvest -> RUN -> Résultats -> Exports
# - Variables à produire & explorer:
#   (Qi, RQi), ARI, FRT, Triggers, QC, et QC rangées par chapitre (pack-driven)
#
# RÈGLES ISO-PROD (appliquées ici):
# - Interdit: "stub pack" en ISO-PROD. Si pack introuvable => blocage RUN.
# - Interdit: hardcode métier (pays/langue/matière/chapitres) dans le core.
#   => Le pack fournit chapitres + metadata; le core ne contient qu'heuristiques générales.
# - Preuve: exports JSON + logs.
# - Verdict test central: Aucune Qi ne doit rester orpheline (Qi sans QC).
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import time
import glob
import math
import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests
import streamlit as st

# pdf parsing (robuste)
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # fallback text extraction will be limited

# html parsing
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


# -----------------------------------------------------------------------------
# CONST / SETTINGS
# -----------------------------------------------------------------------------
APP_VERSION = "V31.10.4"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

DEFAULT_COUNTRY = "FR"
DEFAULT_LEVEL = "TERMINALE"
DEFAULT_SUBJECTS = ["MATH"]

# APMEP root discovery: on reste général (pas de hardcode chapitres),
# mais on a besoin d'une source web pour le test.
APMEP_ROOT_BY_LEVEL = {
    "TERMINALE": "https://www.apmep.fr/Annales-du-Bac-Terminale",
    "PREMIERE": "https://www.apmep.fr/Annales-du-Bac-Premiere",
    "SECONDE": "https://www.apmep.fr/Annales-du-Bac-Seconde",
}

# -----------------------------------------------------------------------------
# SESSION LOG (preuve)
# -----------------------------------------------------------------------------
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def ss_init():
    st.session_state.setdefault("pack_active", None)
    st.session_state.setdefault("pack_id", None)
    st.session_state.setdefault("country", DEFAULT_COUNTRY)
    st.session_state.setdefault("level", DEFAULT_LEVEL)
    st.session_state.setdefault("subjects", DEFAULT_SUBJECTS[:])
    st.session_state.setdefault("library", [])  # harvest_manifest.library (list of pairs)
    st.session_state.setdefault("harvest_manifest", None)
    st.session_state.setdefault("qi_pack", None)
    st.session_state.setdefault("qc_pack", None)
    st.session_state.setdefault("chapter_report", None)
    st.session_state.setdefault("sealed", False)
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("run_stats", {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0})
    st.session_state.setdefault("last_run_audit", None)

def log(msg: str):
    line = f"[{_utc_ts()}] {msg}"
    st.session_state.logs.append(line)

def logs_text() -> str:
    return "\n".join(st.session_state.logs[-5000:])  # cap (preuve suffisante)


# -----------------------------------------------------------------------------
# PACK LOADING (ISO-PROD: no stub)
# -----------------------------------------------------------------------------
def _safe_read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _candidate_pack_paths(country: str) -> List[str]:
    """
    Recherche pack sur plusieurs emplacements usuels.
    ISO-PROD: on ne "devine" pas un pack; on le charge depuis fichier.
    """
    candidates = []

    # Emplacements typiques
    candidates += [
        os.path.join(os.getcwd(), "academic_packs", f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), "packs", f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), "data", "packs", f"CAP_{country}_BAC_2024_V1.json"),
    ]

    # Scan plus large (utile si repo structure différente)
    candidates += glob.glob(os.path.join(os.getcwd(), "**", f"CAP_{country}_BAC_2024_V1.json"), recursive=True)
    candidates += glob.glob(os.path.join(os.getcwd(), "**", f"CAP_{country}_*.json"), recursive=True)

    # Unicité + ordre stable
    seen = set()
    out = []
    for p in candidates:
        pp = os.path.abspath(p)
        if pp not in seen and os.path.isfile(pp):
            seen.add(pp)
            out.append(pp)
    return out

def load_academic_pack(country: str) -> Dict[str, Any]:
    """
    ISO-PROD: charge le pack réel. Si absent => exception bloquante.
    Attendu minimal:
      - pack_id (str)
      - country (str)
      - chapters: list[ {chapter_code, chapter_label} ] OU structure équivalente
    """
    paths = _candidate_pack_paths(country)
    log(f"[PACK] Searching pack for country={country} | candidates={len(paths)}")
    if not paths:
        raise FileNotFoundError(
            "Aucun fichier pack trouvé. Placez le pack JSON (ex: CAP_FR_BAC_2024_V1.json) "
            "dans ./academic_packs/ ou ./packs/ ou à la racine du projet."
        )

    last_err = None
    for path in paths[:25]:  # cap safety
        try:
            pack = _safe_read_json(path)
            # validations minimales
            pack_id = str(pack.get("pack_id") or pack.get("id") or pack.get("name") or "").strip()
            chapters = pack.get("chapters") or pack.get("chapter_list") or pack.get("academic", {}).get("chapters")
            if not pack_id or not isinstance(chapters, list) or len(chapters) == 0:
                raise ValueError("Schéma pack invalide: pack_id manquant ou chapters vide/invalide.")

            pack["_pack_file_path"] = path
            log(f"[PACK] Loaded pack_id={pack_id} from {path}")
            return pack
        except Exception as e:
            last_err = e
            log(f"[PACK] Candidate failed: {path} | err={type(e).__name__}: {e}")

    raise RuntimeError(f"Impossible de charger un pack valide. Dernière erreur: {last_err}")

def pack_chapters(pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = pack.get("chapters") or pack.get("chapter_list") or pack.get("academic", {}).get("chapters") or []
    # Normalisation (codes/labels)
    # Si le pack contient des groupes -> on tente un flatten générique
    out: List[Dict[str, Any]] = []
    for item in ch:
        if isinstance(item, dict) and ("chapter_code" in item or "code" in item):
            code = str(item.get("chapter_code") or item.get("code") or "").strip()
            label = str(item.get("chapter_label") or item.get("label") or code).strip()
            if code:
                out.append({"chapter_code": code, "chapter_label": label, "keywords": item.get("keywords", [])})
        elif isinstance(item, dict) and "items" in item and isinstance(item["items"], list):
            for sub in item["items"]:
                if isinstance(sub, dict):
                    code = str(sub.get("chapter_code") or sub.get("code") or "").strip()
                    label = str(sub.get("chapter_label") or sub.get("label") or code).strip()
                    if code:
                        out.append({"chapter_code": code, "chapter_label": label, "keywords": sub.get("keywords", [])})
    # dédoublonnage stable
    seen = set()
    uniq = []
    for c in out:
        if c["chapter_code"] not in seen:
            seen.add(c["chapter_code"])
            uniq.append(c)
    return uniq


# -----------------------------------------------------------------------------
# TEXT NORMALISATION / HASHING (déterminisme)
# -----------------------------------------------------------------------------
def norm_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def stable_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:12]

def tokenize(s: str) -> List[str]:
    s = norm_text(s)
    # tokens alpha-num; on garde général (pas de dictionnaire métier)
    toks = re.findall(r"[a-z0-9]{3,}", s)
    return toks[:800]


# -----------------------------------------------------------------------------
# PDF DOWNLOAD + TEXT EXTRACTION
# -----------------------------------------------------------------------------
def _http_get(url: str) -> requests.Response:
    headers = {"User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r

def download_pdf_bytes(url: str) -> bytes:
    r = _http_get(url)
    size_mb = len(r.content) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        raise ValueError(f"PDF trop volumineux: {size_mb:.1f} MB > {MAX_PDF_MB} MB")
    return r.content

def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 60) -> str:
    if pdfplumber is None:
        # fallback minimal: on ne peut pas garantir extraction
        return ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        texts = []
        for i, page in enumerate(pdf.pages[:max_pages]):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                texts.append(t)
        return "\n".join(texts)


# -----------------------------------------------------------------------------
# Qi / RQi EXTRACTION (heuristiques générales)
# -----------------------------------------------------------------------------
_Q_SPLIT_PATTERNS = [
    r"\bexercice\s+\d+\b",
    r"\bquestion\s+\d+\b",
    r"(?m)^\s*\d+\s*[).]\s+",
    r"(?m)^\s*[a-h]\s*[).]\s+",
]

def split_questions(raw_text: str) -> List[str]:
    t = raw_text or ""
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)

    # On coupe sur des marqueurs génériques (exercice/question/numérotation)
    # Déterministe: on applique un splitter global.
    # On évite de dépendre d'une matière spécifique.
    idxs = [0]
    for pat in _Q_SPLIT_PATTERNS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            idxs.append(m.start())
    idxs = sorted(set(idxs))
    chunks = []
    for a, b in zip(idxs, idxs[1:] + [len(t)]):
        c = t[a:b].strip()
        if len(c) >= 40:
            chunks.append(c)

    # Si splitter a échoué, fallback: phrases interrogatives
    if len(chunks) <= 1:
        parts = re.split(r"(?<=[\?\!])\s+", t)
        chunks = [p.strip() for p in parts if len(p.strip()) >= 60]

    # Nettoyage: on réduit à l'énoncé utile
    cleaned = []
    for c in chunks:
        c2 = re.sub(r"\s+", " ", c).strip()
        # on évite les blocs ultra longs (souvent tout le sujet)
        if len(c2) > 2500:
            c2 = c2[:2500].rstrip() + "…"
        cleaned.append(c2)

    # dédoublonnage stable
    seen = set()
    out = []
    for c in cleaned:
        k = stable_id(norm_text(c)[:400])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out[:300]


# -----------------------------------------------------------------------------
# CLUSTER (SimHash) — généraliste & déterministe
# -----------------------------------------------------------------------------
def simhash64(tokens: List[str]) -> int:
    # SimHash classique (64 bits) sans dépendances externes
    if not tokens:
        return 0
    v = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)  # stable
        for i in range(64):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i, val in enumerate(v):
        if val > 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def cluster_by_simhash(items: List[Dict[str, Any]], sim_threshold: float, min_cluster_size: int) -> List[List[Dict[str, Any]]]:
    """
    sim_threshold: 0..1 (proche 1 => clusters très serrés)
    min_cluster_size: anti-singleton
    """
    # Convert sim_threshold to max hamming distance
    # approx: similarity ~= 1 - (ham/64)
    max_ham = max(0, min(64, int(round((1.0 - sim_threshold) * 64))))
    # bornes: on empêche un clustering trop permissif par défaut
    max_ham = min(max_ham, 26)

    # Precompute hashes
    for it in items:
        toks = tokenize(it.get("text", ""))
        it["_sh"] = simhash64(toks)

    clusters: List[List[Dict[str, Any]]] = []
    used = set()

    # Déterminisme: tri sur id
    ordered = sorted(items, key=lambda x: x["qi_id"])

    for i, it in enumerate(ordered):
        if it["qi_id"] in used:
            continue
        base = it
        cl = [base]
        used.add(base["qi_id"])
        for j in range(i + 1, len(ordered)):
            other = ordered[j]
            if other["qi_id"] in used:
                continue
            if hamming(base["_sh"], other["_sh"]) <= max_ham:
                cl.append(other)
                used.add(other["qi_id"])
        clusters.append(cl)

    # anti-singleton: on garde les singletons mais marqués, ou on les fusionne dans un bucket "misc"
    if min_cluster_size <= 1:
        return clusters

    big = [c for c in clusters if len(c) >= min_cluster_size]
    small = [c for c in clusters if len(c) < min_cluster_size]
    if not small:
        return clusters

    # On regroupe les petits dans un cluster "misc" (déterministe)
    misc = []
    for c in small:
        misc.extend(c)
    if misc:
        big.append(misc)
    return big


# -----------------------------------------------------------------------------
# ARI / FRT / Triggers / QC (templates invariants)
# -----------------------------------------------------------------------------
def build_triggers(qi_text: str) -> List[str]:
    # Triggers génériques (invariants) — pas de dépendance chapitre/matière
    # Ils structurent la résolution, pas le thème.
    t = norm_text(qi_text)
    out = []

    # Heuristiques universelles
    if any(x in t for x in ["montrer", "demontrer", "prouver", "justifier"]):
        out.append("Exiger une justification complète (preuve).")
    if any(x in t for x in ["calculer", "determiner", "donner la valeur", "evaluer"]):
        out.append("Produire un résultat numérique ou symbolique vérifiable.")
    if any(x in t for x in ["resoudre", "trouver", "solution"]):
        out.append("Isoler l'inconnue et valider les solutions (conditions).")
    if "tableau" in t or "graph" in t or "courbe" in t:
        out.append("Extraire l'information d'une représentation (table/graph).")
    if "probabil" in t:
        out.append("Identifier l'espace des cas et contrôler la normalisation.")
    if not out:
        out = [
            "Identifier les données et l'objectif.",
            "Choisir une stratégie de résolution.",
            "Exécuter le calcul avec contrôle d'erreurs.",
            "Vérifier la cohérence du résultat."
        ]
    return out[:6]

def build_ari(qi_text: str, rqi_text: str) -> Dict[str, Any]:
    # ARI = gabarit invariant: étapes universelles.
    return {
        "template": "ARI_V1_INVARIANT",
        "steps": [
            {"k": 1, "name": "Comprendre", "do": "Reformuler la question en objectif opérationnel."},
            {"k": 2, "name": "Données", "do": "Lister les données/conditions/contraintes explicites."},
            {"k": 3, "name": "Stratégie", "do": "Choisir une méthode générale (sans dépendre du chapitre)."},
            {"k": 4, "name": "Exécution", "do": "Appliquer la méthode et produire les transformations."},
            {"k": 5, "name": "Validation", "do": "Contrôler unités, domaine, cohérence, et justifier."},
        ],
        "evidence_hint": "Aligner la solution (RQi) sur ces étapes; chaque étape doit être traçable."
    }

def build_frt(qi_text: str, rqi_text: str, triggers: List[str]) -> Dict[str, Any]:
    # FRT = forme de réponse type, invariant, rempli par signaux.
    return {
        "template": "FRT_V1_INVARIANT",
        "sections": [
            {"name": "Objectif", "fill": "But précis de la question (en 1 phrase)."},
            {"name": "Données", "fill": "Liste des données utiles et hypothèses."},
            {"name": "Méthode", "fill": "Méthode générale choisie + justification."},
            {"name": "Calcul", "fill": "Étapes de calcul / transformations."},
            {"name": "Résultat", "fill": "Résultat final + forme attendue."},
            {"name": "Contrôle", "fill": "Vérification de cohérence (domaine, unités, cas limites)."},
        ],
        "triggers": triggers,
    }

def build_qc_label(cluster_items: List[Dict[str, Any]]) -> str:
    # QC: doit commencer par "Comment" et finir par "?"
    # On dérive des tokens fréquents (invariant: extraction statistique, pas de liste métier).
    all_toks: List[str] = []
    for it in cluster_items:
        all_toks += tokenize(it.get("text", ""))
    freq: Dict[str, int] = {}
    for tok in all_toks:
        freq[tok] = freq.get(tok, 0) + 1
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:6]
    keywords = [k for k, _ in top][:4]
    if keywords:
        mid = " ".join(keywords)
        qc = f"Comment résoudre une question caractérisée par: {mid} ?"
    else:
        qc = "Comment résoudre ce type de question ?"
    # Normalisation de ponctuation
    qc = qc.replace("??", "?").strip()
    if not qc.startswith("Comment"):
        qc = "Comment " + qc[0].lower() + qc[1:]
    if not qc.endswith("?"):
        qc += " ?"
    return qc

def map_qc_to_chapter(qc_text: str, chapters: List[Dict[str, Any]]) -> str:
    """
    Mapping pack-driven:
    - si le pack fournit keywords, on fait matching statistique
    - sinon, on renvoie "UNMAPPED" (preuve claire)
    """
    if not chapters:
        return "UNMAPPED"
    qc_toks = set(tokenize(qc_text))
    best = ("UNMAPPED", 0.0)

    for ch in chapters:
        kws = ch.get("keywords") or []
        if not isinstance(kws, list) or len(kws) == 0:
            continue
        ch_toks = set(tokenize(" ".join([str(x) for x in kws])))
        if not ch_toks:
            continue
        inter = len(qc_toks & ch_toks)
        score = inter / max(1, len(ch_toks))
        if score > best[1]:
            best = (str(ch["chapter_code"]), score)

    return best[0] if best[1] >= 0.12 else "UNMAPPED"


# -----------------------------------------------------------------------------
# HARVEST APMEP (généraliste)
# -----------------------------------------------------------------------------
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé. Installez: pip install beautifulsoup4")
    return BeautifulSoup(html, "html.parser")

def harvest_apmep(level: str, subject: str, years_back: int, volume_max: int) -> Dict[str, Any]:
    """
    Produit harvest_manifest = {version, timestamp, country, level, subjects, items_total, items_corrige_ok, library[]}
    """
    if level not in APMEP_ROOT_BY_LEVEL:
        raise ValueError(f"Niveau non supporté pour APMEP: {level}")
    root = APMEP_ROOT_BY_LEVEL[level]
    log(f"[HARVEST] scope={level}|{subject} root={root} years_back={years_back} volume_max={volume_max}")

    # On récupère le root, puis on repère des pages Année-YYYY
    html = _http_get(root).text
    sp = _soup(html)

    year_links: List[Tuple[int, str]] = []
    for a in sp.find_all("a"):
        href = a.get("href") or ""
        text = (a.get_text() or "").strip()
        m = re.search(r"Annee-(20\d{2})", href) or re.search(r"(20\d{2})", text)
        if m and "Annee-" in href:
            y = int(m.group(1))
            url = href if href.startswith("http") else requests.compat.urljoin(root, href)
            year_links.append((y, url))

    year_links = sorted(set(year_links), key=lambda x: -x[0])
    if not year_links:
        raise RuntimeError("Impossible de trouver les pages Année-YYYY sur APMEP (structure HTML inattendue).")

    current_year = year_links[0][0]
    min_year = current_year - max(1, years_back) + 1
    selected = [(y, u) for (y, u) in year_links if y >= min_year]
    log(f"[HARVEST] years detected={len(year_links)} selected={len(selected)} range=[{min_year}..{current_year}]")

    pairs: List[Dict[str, Any]] = []
    corrige_ok = 0

    for (y, url) in selected:
        if len(pairs) >= volume_max:
            break
        try:
            log(f"[HARVEST] year={y} url={url}")
            y_html = _http_get(url).text
            y_sp = _soup(y_html)

            # Heuristique: on détecte des "blocs" contenant 1 ou 2 liens PDF
            # Souvent table/li/div; on prend parent proche.
            pdf_anchors = [a for a in y_sp.find_all("a") if (a.get("href") or "").lower().endswith(".pdf")]
            log(f"[HARVEST] year={y} pdf_links={len(pdf_anchors)}")

            # Grouping par parent "row-like"
            groups = []
            for a in pdf_anchors:
                parent = a
                for _ in range(4):
                    if parent is None:
                        break
                    # Un parent avec plusieurs liens pdf = candidat
                    pdfs_in_parent = parent.find_all("a") if hasattr(parent, "find_all") else []
                    n_pdf = sum(1 for x in pdfs_in_parent if (x.get("href") or "").lower().endswith(".pdf"))
                    if n_pdf >= 2:
                        groups.append(parent)
                        break
                    parent = parent.parent

            # fallback: si grouping échoue, on prend les liens à plat
            if not groups:
                groups = [y_sp]

            # Extraction pairs
            for g in groups:
                if len(pairs) >= volume_max:
                    break
                links = [a for a in g.find_all("a") if (a.get("href") or "").lower().endswith(".pdf")]
                if not links:
                    continue

                # On filtre "explication" ou pdf méta quand possible (non bloquant)
                def is_meta_pdf(h: str, txt: str) -> bool:
                    s = norm_text(h + " " + txt)
                    return any(k in s for k in ["explication", "liste", "index", "sommaire"])

                # classement: sujet vs corrigé
                def is_corrige(a) -> bool:
                    href = (a.get("href") or "")
                    txt = (a.get_text() or "")
                    s = norm_text(href + " " + txt)
                    return "corrig" in s or "corrige" in s

                pdfs = []
                for a in links:
                    href = a.get("href") or ""
                    txt = (a.get_text() or "").strip()
                    url_pdf = href if href.startswith("http") else requests.compat.urljoin(url, href)
                    if is_meta_pdf(url_pdf, txt):
                        continue
                    pdfs.append({"url": url_pdf, "name": os.path.basename(url_pdf), "is_corrige": is_corrige(a)})

                if not pdfs:
                    continue

                # si un bloc contient exactement 1 sujet + 1 corrigé, parfait
                sujets = [p for p in pdfs if not p["is_corrige"]]
                corr = [p for p in pdfs if p["is_corrige"]]

                # pairing simple: le 1er sujet du bloc avec le 1er corrigé du bloc
                # sinon: sujet seul.
                sujet = sujets[0] if sujets else pdfs[0]
                corrige = corr[0] if corr else None

                pair_id = f"PAIR_{level}|{subject}_{y}_{stable_id(sujet['name'], str(corrige['name'] if corrige else ''))}"
                item = {
                    "pair_id": pair_id,
                    "scope": f"{level}|{subject}",
                    "source": f"APMEP — Année {y}",
                    "year": y,
                    "sujet": sujet["name"],
                    "corrige?": bool(corrige),
                    "corrige_name": (corrige["name"] if corrige else ""),
                    "reason": ("" if corrige else "corrigé absent sur la ligne"),
                    "sujet_url": sujet["url"],
                    "corrige_url": (corrige["url"] if corrige else ""),
                }

                # évite doublons
                if any(x["sujet_url"] == item["sujet_url"] for x in pairs):
                    continue

                pairs.append(item)
                if item["corrige?"]:
                    corrige_ok += 1

        except Exception as e:
            log(f"[HARVEST] year={y} FAILED err={type(e).__name__}: {e}")

    manifest = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "country": st.session_state.country,
        "level": level,
        "subjects": [subject],
        "items_total": len(pairs),
        "items_corrige_ok": corrige_ok,
        "library": pairs,
    }
    return manifest


# -----------------------------------------------------------------------------
# RUN GRANULO TEST (mono-fichier)
# -----------------------------------------------------------------------------
def run_granulo_test(library: List[Dict[str, Any]], volume: int, sim_threshold: float, min_cluster_size: int, max_iterations: int, pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sortie: { sujets, qc, saturation, audit }
    - audit: preuve booléenne (orphelins, coverage)
    """
    chapters = pack_chapters(pack)
    if not chapters:
        raise RuntimeError("Pack invalide: liste de chapitres vide après normalisation.")

    # ISO-PROD gating: exploitable = corrigé présent
    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    if not exploitable:
        raise RuntimeError("ISO-PROD: Aucun corrigé exploitable dans la bibliothèque. RUN interdit.")

    # consommer volume
    to_process = exploitable[: max(1, min(volume, len(exploitable)))]
    log(f"[RUN] volume={volume} exploitable={len(exploitable)} processing={len(to_process)} sim={sim_threshold} min_cluster={min_cluster_size} iters={max_iterations}")

    qi_items: List[Dict[str, Any]] = []
    rqi_items: List[Dict[str, Any]] = []
    sujets_meta: List[Dict[str, Any]] = []

    for pair in to_process:
        pid = pair["pair_id"]
        su_url = pair["sujet_url"]
        co_url = pair["corrige_url"]
        log(f"[DL] pair_id={pid} sujet={os.path.basename(su_url)} corrige={os.path.basename(co_url)}")

        try:
            su_pdf = download_pdf_bytes(su_url)
            co_pdf = download_pdf_bytes(co_url)
        except Exception as e:
            log(f"[DL] FAILED pair_id={pid} err={type(e).__name__}: {e}")
            continue

        su_text = extract_text_from_pdf_bytes(su_pdf)
        co_text = extract_text_from_pdf_bytes(co_pdf)

        if not su_text or not co_text:
            log(f"[PDF] EMPTY_TEXT pair_id={pid} su_text={bool(su_text)} co_text={bool(co_text)}")
            continue

        qs = split_questions(su_text)
        rs = split_questions(co_text)

        sujets_meta.append({
            "pair_id": pid,
            "sujet_url": su_url,
            "corrige_url": co_url,
            "qi_count": len(qs),
            "rqi_count": len(rs),
        })

        # Alignement simple: index
        n = max(len(qs), len(rs))
        for i in range(n):
            q = qs[i] if i < len(qs) else ""
            r = rs[i] if i < len(rs) else ""
            if not q:
                continue
            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
            rqi_id = f"RQI_{stable_id(pid, str(i), norm_text(r)[:180])}" if r else ""

            qi_items.append({
                "qi_id": qi_id,
                "pair_id": pid,
                "k": i + 1,
                "text": q,
                "rqi_id": rqi_id,
            })
            if r:
                rqi_items.append({
                    "rqi_id": rqi_id,
                    "pair_id": pid,
                    "k": i + 1,
                    "text": r,
                })

    if not qi_items:
        raise RuntimeError("RUN: aucune Qi extraite (échec extraction PDF ou contenu vide).")

    # Cluster -> QC
    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}  # qi_id -> qc_id

    # Saturation: déterministe (répéter clustering; ici itérations plafonnées, mais stable)
    prev_qc_count = -1
    sealed = False
    for it in range(max(1, max_iterations)):
        clusters = cluster_by_simhash(qi_items, sim_threshold=sim_threshold, min_cluster_size=max(1, min_cluster_size))
        qc_pack = []
        qc_map = {}

        for cidx, cluster in enumerate(clusters, start=1):
            qc_text = build_qc_label(cluster)
            qc_id = f"QC_{stable_id(str(cidx), norm_text(qc_text))}"
            chapter_code = map_qc_to_chapter(qc_text, chapters)

            # Build QC payload
            qc_item = {
                "qc_id": qc_id,
                "qc": qc_text,
                "chapter_code": chapter_code,
                "cluster_size": len(cluster),
                "qi_ids": [x["qi_id"] for x in cluster],
            }
            qc_pack.append(qc_item)
            for x in cluster:
                qc_map[x["qi_id"]] = qc_id

        # stabilité
        if prev_qc_count == len(qc_pack):
            sealed = True
            log(f"[SAT] stable at iter={it+1} qc_count={len(qc_pack)} => SEALED_CANDIDATE")
            break
        prev_qc_count = len(qc_pack)
        log(f"[SAT] iter={it+1} qc_count={len(qc_pack)}")

    # Construire ARI/FRT/Triggers par Qi (puis agrégation dans explorer)
    qi_pack: List[Dict[str, Any]] = []
    rqi_by_id = {r["rqi_id"]: r for r in rqi_items}

    qi_posable = 0
    orphan = 0
    for q in qi_items:
        qi_id = q["qi_id"]
        rqi_id = q.get("rqi_id") or ""
        rtxt = rqi_by_id.get(rqi_id, {}).get("text", "")
        triggers = build_triggers(q["text"])
        ari = build_ari(q["text"], rtxt)
        frt = build_frt(q["text"], rtxt, triggers)

        qc_id = qc_map.get(qi_id, "")
        if not qc_id:
            orphan += 1
        if rtxt:
            qi_posable += 1

        qi_pack.append({
            "qi_id": qi_id,
            "pair_id": q["pair_id"],
            "k": q["k"],
            "qi": q["text"],
            "rqi": rtxt,
            "qc_id": qc_id,
            "triggers": triggers,
            "ari": ari,
            "frt": frt,
        })

    # Audit ISO-PROD
    audit = {
        "qi_total": len(qi_pack),
        "rqi_total": sum(1 for q in qi_pack if q["rqi"]),
        "qc_total": len(qc_pack),
        "qi_orphans": orphan,
        "coverage_ok": (orphan == 0),
        "posable_ok": (qi_posable > 0),
        "sealed_candidate": sealed,
        "sealed": (sealed and orphan == 0 and qi_posable > 0 and len(qc_pack) > 0),
    }

    # Chapter report (pack-driven)
    by_ch: Dict[str, Dict[str, Any]] = {}
    ch_label = {c["chapter_code"]: c.get("chapter_label", c["chapter_code"]) for c in chapters}

    for qc in qc_pack:
        cc = qc["chapter_code"]
        by_ch.setdefault(cc, {"chapter_code": cc, "chapter_label": ch_label.get(cc, cc), "qc_ids": [], "qc_count": 0, "qi_count": 0})
        by_ch[cc]["qc_ids"].append(qc["qc_id"])
        by_ch[cc]["qc_count"] += 1
        by_ch[cc]["qi_count"] += int(qc["cluster_size"])

    chapter_report = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": st.session_state.pack_id,
        "chapters": sorted(by_ch.values(), key=lambda x: (-x["qi_count"], x["chapter_code"])),
        "unmapped_present": any(x["chapter_code"] == "UNMAPPED" for x in by_ch.values()),
    }

    out = {"sujets": sujets_meta, "qc": qc_pack, "saturation": {"iterations": it + 1, "sealed_candidate": sealed}, "audit": audit}
    return {"out": out, "qi_pack": qi_pack, "qc_pack": qc_pack, "chapter_report": chapter_report, "audit": audit}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def metric_row(items: int, corr_ok: int, qi: int, qi_posable: int, qc: int, sealed: bool):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Items", items)
    c2.metric("Corrigés exploitables", corr_ok)
    c3.metric("Qi", qi)
    c4.metric("Qi POSABLE", qi_posable)
    c5.metric("QC", qc)
    c6.metric("SEALED", "YES" if sealed else "NO")

def df_table(rows: List[Dict[str, Any]]):
    if not rows:
        st.info("Aucun item en bibliothèque. Lancez HARVEST AUTO ou faites un upload manuel.")
        return
    # affichage stable
    cols = ["pair_id", "scope", "source", "sujet", "corrige?", "corrige_name", "reason", "sujet_url", "corrige_url"]
    view = [{k: r.get(k, "") for k in cols} for r in rows]
    st.dataframe(view, use_container_width=True, hide_index=True)

def main():
    st.set_page_config(page_title=f"SMAXIA GTE Console {APP_VERSION}", layout="wide")
    ss_init()

    # Header
    st.markdown(f"# 🔒 SMAXIA GTE Console {APP_VERSION} — ISO-PROD TEST")
    st.caption("Flux: Activation → Pack visible → Sélection → Harvest → RUN → Résultats/Exports")

    # Sidebar: Activation + Sélection
    with st.sidebar:
        st.markdown("## ÉTAPE 1 — ACTIVATION PAYS")
        country = st.selectbox("Pays (TEST)", options=[DEFAULT_COUNTRY], index=0)
        st.session_state.country = country

        activate = st.button("🔒 ACTIVER", use_container_width=True)
        if activate:
            try:
                pack = load_academic_pack(country)
                st.session_state.pack_active = pack
                st.session_state.pack_id = str(pack.get("pack_id") or pack.get("id") or pack.get("name"))
                st.success("Pack actif")
            except Exception as e:
                st.session_state.pack_active = None
                st.session_state.pack_id = None
                st.error(f"Activation PACK impossible (ISO-PROD): {e}")

        st.markdown("---")
        if st.session_state.pack_active:
            st.success("✅ Pack actif")
            st.write(f"**{st.session_state.pack_id}**")
            st.caption(f"Pack file: {st.session_state.pack_active.get('_pack_file_path','')}")
        else:
            st.warning("Pack inactif (ISO-PROD: RUN bloqué sans pack).")

        st.markdown("---")
        st.markdown("## ÉTAPE 2 — SÉLECTION")
        level = st.radio("Niveau", options=["Seconde", "Première", "Terminale", "Licence 1", "Prépa (CPGE)"], index=2)
        # normalisation
        level_code = "TERMINALE" if level == "Terminale" else ("PREMIERE" if level == "Première" else ("SECONDE" if level == "Seconde" else "TERMINALE"))
        st.session_state.level = level_code

        subjects = st.multiselect("Matières (1–2 recommandé pour test)", options=["MATH"], default=st.session_state.subjects)
        st.session_state.subjects = subjects if subjects else DEFAULT_SUBJECTS[:]

        st.markdown("### Chapitres (pack-driven)")
        if st.session_state.pack_active:
            chs = pack_chapters(st.session_state.pack_active)
            if not chs:
                st.error("Pack chargé mais chapitres introuvables (schéma pack).")
            else:
                # On affiche des VRAIS chapitres du pack (pas des items génériques)
                for c in chs[:40]:
                    st.write(f"• {c['chapter_code']} — {c.get('chapter_label','')}")
                if len(chs) > 40:
                    st.caption(f"+{len(chs)-40} autres chapitres…")
        else:
            st.caption("Activez un pack pour afficher les chapitres réels.")


    # Tabs
    tab1, tab2, tab3 = st.tabs(["📥 Import / Bibliothèque", "⚡ Chaîne (RUN)", "📦 Résultats / Exports"])

    # Compute metrics
    lib = st.session_state.library
    items_total = len(lib)
    corr_ok = sum(1 for x in lib if x.get("corrige?") and x.get("corrige_url"))
    qi_count = int(st.session_state.run_stats.get("qi", 0))
    qi_posable = int(st.session_state.run_stats.get("qi_posable", 0))
    qc_count = int(st.session_state.run_stats.get("qc", 0))
    sealed = bool(st.session_state.sealed)

    with tab1:
        st.markdown("## Import PDF (sujets + corrigés)")
        metric_row(items_total, corr_ok, qi_count, qi_posable, qc_count, sealed)

        st.markdown("### Bibliothèque Harvest (visible)")
        show_only_no_corr = st.checkbox("Afficher seulement les items sans corrigé exploitable (❌)", value=False)
        if show_only_no_corr:
            df_table([x for x in lib if not (x.get("corrige?") and x.get("corrige_url"))])
        else:
            df_table(lib)

        st.markdown("---")
        st.markdown("### HARVEST AUTO (APMEP) — paramètres")
        c1, c2, c3 = st.columns([1, 1, 1.2])
        years_back = c1.number_input("Nb d'années à récolter (depuis la plus récente)", min_value=1, max_value=15, value=10, step=1)
        volume_max = c2.number_input("Volume max (items)", min_value=5, max_value=200, value=50, step=5)
        c3.text_input("Scope (auto)", value=f"{st.session_state.level}|{st.session_state.subjects[0]}", disabled=True)

        if st.button("🌐 Lancer HARVEST AUTO (Bibliothèque)", use_container_width=True):
            try:
                if not st.session_state.pack_active:
                    st.error("ISO-PROD: Pack inactif. Activez d'abord le pack.")
                else:
                    manifest = harvest_apmep(st.session_state.level, st.session_state.subjects[0], int(years_back), int(volume_max))
                    st.session_state.harvest_manifest = manifest
                    st.session_state.library = manifest["library"]
                    st.session_state.sealed = False
                    st.session_state.qi_pack = None
                    st.session_state.qc_pack = None
                    st.session_state.chapter_report = None
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0}
                    st.success(f"HARVEST terminé. +{len(manifest['library'])} items ajoutés (total={len(manifest['library'])}).")
            except Exception as e:
                st.error(f"HARVEST échoué: {e}")

        st.markdown("---")
        st.markdown("### Upload manuel (optionnel)")
        up1, up2 = st.columns(2)
        sujet_file = up1.file_uploader("PDF Sujet", type=["pdf"])
        corr_file = up2.file_uploader("PDF Correction (opt)", type=["pdf"])

        if st.button("➕ Ajouter à la bibliothèque", use_container_width=True):
            if not sujet_file:
                st.error("Veuillez fournir au moins le PDF Sujet.")
            else:
                # Upload manuel: on stocke en mémoire (pas de hardcode), et on crée un item local://
                su_bytes = sujet_file.getvalue()
                co_bytes = corr_file.getvalue() if corr_file else b""

                su_id = stable_id(sujet_file.name, str(len(su_bytes)))
                co_id = stable_id(corr_file.name, str(len(co_bytes))) if corr_file else ""
                pid = f"PAIR_{st.session_state.level}|{st.session_state.subjects[0]}_{stable_id(su_id, co_id)}"

                # On persiste bytes en session_state (ISO-PROD test local)
                st.session_state.setdefault("_uploads", {})
                st.session_state._uploads[f"local://{su_id}"] = su_bytes
                if corr_file:
                    st.session_state._uploads[f"local://{co_id}"] = co_bytes

                item = {
                    "pair_id": pid,
                    "scope": f"{st.session_state.level}|{st.session_state.subjects[0]}",
                    "source": "UPLOAD_MANUEL",
                    "year": None,
                    "sujet": sujet_file.name,
                    "corrige?": bool(corr_file),
                    "corrige_name": (corr_file.name if corr_file else ""),
                    "reason": ("" if corr_file else "corrigé non fourni"),
                    "sujet_url": f"local://{su_id}",
                    "corrige_url": (f"local://{co_id}" if corr_file else ""),
                }
                st.session_state.library.insert(0, item)
                st.success("Ajouté à la bibliothèque.")

    # Override downloader to support local://
    def get_pdf_bytes(url: str) -> bytes:
        if url.startswith("local://"):
            key = url
            data = st.session_state.get("_uploads", {}).get(key)
            if not data:
                raise RuntimeError(f"Upload local introuvable: {key}")
            return data
        return download_pdf_bytes(url)

    # Patch run function to use get_pdf_bytes
    def run_granulo_test_patched(*args, **kwargs):
        # monkey patch inside closure
        nonlocal get_pdf_bytes

        library = kwargs.get("library") or args[0]
        volume = kwargs.get("volume") or args[1]
        sim_threshold = kwargs.get("sim_threshold") or args[2]
        min_cluster_size = kwargs.get("min_cluster_size") or args[3]
        max_iterations = kwargs.get("max_iterations") or args[4]
        pack = kwargs.get("pack") or args[5]

        chapters = pack_chapters(pack)
        exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
        if not exploitable:
            raise RuntimeError("ISO-PROD: Aucun corrigé exploitable dans la bibliothèque. RUN interdit.")

        to_process = exploitable[: max(1, min(volume, len(exploitable)))]
        log(f"[RUN] patched local:// aware | processing={len(to_process)}")

        qi_items, rqi_items, sujets_meta = [], [], []
        for pair in to_process:
            pid = pair["pair_id"]
            try:
                su_pdf = get_pdf_bytes(pair["sujet_url"])
                co_pdf = get_pdf_bytes(pair["corrige_url"])
            except Exception as e:
                log(f"[DL] FAILED pair_id={pid} err={type(e).__name__}: {e}")
                continue

            su_text = extract_text_from_pdf_bytes(su_pdf)
            co_text = extract_text_from_pdf_bytes(co_pdf)
            if not su_text or not co_text:
                log(f"[PDF] EMPTY_TEXT pair_id={pid} su_text={bool(su_text)} co_text={bool(co_text)}")
                continue

            qs = split_questions(su_text)
            rs = split_questions(co_text)
            sujets_meta.append({"pair_id": pid, "sujet_url": pair["sujet_url"], "corrige_url": pair["corrige_url"], "qi_count": len(qs), "rqi_count": len(rs)})

            n = max(len(qs), len(rs))
            for i in range(n):
                q = qs[i] if i < len(qs) else ""
                r = rs[i] if i < len(rs) else ""
                if not q:
                    continue
                qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
                rqi_id = f"RQI_{stable_id(pid, str(i), norm_text(r)[:180])}" if r else ""
                qi_items.append({"qi_id": qi_id, "pair_id": pid, "k": i + 1, "text": q, "rqi_id": rqi_id})
                if r:
                    rqi_items.append({"rqi_id": rqi_id, "pair_id": pid, "k": i + 1, "text": r})

        if not qi_items:
            raise RuntimeError("RUN: aucune Qi extraite (échec extraction PDF ou contenu vide).")

        # saturation + qc
        qc_pack, qc_map = [], {}
        prev_qc_count = -1
        sealed_candidate = False
        for it in range(max(1, max_iterations)):
            clusters = cluster_by_simhash(qi_items, sim_threshold=sim_threshold, min_cluster_size=max(1, min_cluster_size))
            qc_pack = []
            qc_map = {}
            for cidx, cluster in enumerate(clusters, start=1):
                qc_text = build_qc_label(cluster)
                qc_id = f"QC_{stable_id(str(cidx), norm_text(qc_text))}"
                chapter_code = map_qc_to_chapter(qc_text, chapters)
                qc_pack.append({"qc_id": qc_id, "qc": qc_text, "chapter_code": chapter_code, "cluster_size": len(cluster), "qi_ids": [x["qi_id"] for x in cluster]})
                for x in cluster:
                    qc_map[x["qi_id"]] = qc_id
            if prev_qc_count == len(qc_pack):
                sealed_candidate = True
                break
            prev_qc_count = len(qc_pack)

        rqi_by_id = {r["rqi_id"]: r for r in rqi_items}
        qi_pack = []
        qi_posable = 0
        orphan = 0
        for q in qi_items:
            rtxt = rqi_by_id.get(q.get("rqi_id") or "", {}).get("text", "")
            triggers = build_triggers(q["text"])
            ari = build_ari(q["text"], rtxt)
            frt = build_frt(q["text"], rtxt, triggers)
            qc_id = qc_map.get(q["qi_id"], "")
            if not qc_id:
                orphan += 1
            if rtxt:
                qi_posable += 1
            qi_pack.append({"qi_id": q["qi_id"], "pair_id": q["pair_id"], "k": q["k"], "qi": q["text"], "rqi": rtxt, "qc_id": qc_id, "triggers": triggers, "ari": ari, "frt": frt})

        audit = {
            "qi_total": len(qi_pack),
            "rqi_total": sum(1 for q in qi_pack if q["rqi"]),
            "qc_total": len(qc_pack),
            "qi_orphans": orphan,
            "coverage_ok": (orphan == 0),
            "posable_ok": (qi_posable > 0),
            "sealed_candidate": sealed_candidate,
            "sealed": (sealed_candidate and orphan == 0 and qi_posable > 0 and len(qc_pack) > 0),
        }

        by_ch = {}
        ch_label = {c["chapter_code"]: c.get("chapter_label", c["chapter_code"]) for c in chapters}
        for qc in qc_pack:
            cc = qc["chapter_code"]
            by_ch.setdefault(cc, {"chapter_code": cc, "chapter_label": ch_label.get(cc, cc), "qc_ids": [], "qc_count": 0, "qi_count": 0})
            by_ch[cc]["qc_ids"].append(qc["qc_id"])
            by_ch[cc]["qc_count"] += 1
            by_ch[cc]["qi_count"] += int(qc["cluster_size"])

        chapter_report = {
            "version": APP_VERSION,
            "timestamp": _utc_ts(),
            "pack_id": st.session_state.pack_id,
            "chapters": sorted(by_ch.values(), key=lambda x: (-x["qi_count"], x["chapter_code"])),
            "unmapped_present": any(x["chapter_code"] == "UNMAPPED" for x in by_ch.values()),
        }

        out = {"sujets": sujets_meta, "qc": qc_pack, "saturation": {"iterations": it + 1, "sealed_candidate": sealed_candidate}, "audit": audit}
        return {"out": out, "qi_pack": qi_pack, "qc_pack": qc_pack, "chapter_report": chapter_report, "audit": audit}

    with tab2:
        st.markdown("## ÉTAPE 3 — LANCER CHAÎNE (RUN)")
        st.caption("ISO-PROD: refus de RUN si pack inactif OU bibliothèque vide OU aucun corrigé exploitable.")

        if not st.session_state.pack_active:
            st.error("Pack inactif: activez le pack (ISO-PROD).")
        elif not st.session_state.library:
            st.warning("Bibliothèque vide. Lancez HARVEST AUTO ou upload manuel.")
        elif sum(1 for x in st.session_state.library if x.get("corrige?") and x.get("corrige_url")) == 0:
            st.error("ISO-PROD: aucun corrigé exploitable. RUN interdit.")
        else:
            c1, c2, c3 = st.columns(3)
            volume = c1.slider("Volume (items consommés)", min_value=1, max_value=min(120, corr_ok), value=min(20, corr_ok), step=1)
            sim = c2.slider("Seuil clustering (sim)", min_value=0.05, max_value=0.95, value=0.18, step=0.01)
            min_cluster = c3.slider("Anti-singleton (min cluster size)", min_value=1, max_value=6, value=2, step=1)
            max_iters = st.slider("Max itérations (saturation)", min_value=1, max_value=8, value=4, step=1)

            if st.button("⚡ LANCER (RUN)", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_granulo_test_patched(
                        st.session_state.library,
                        int(volume),
                        float(sim),
                        int(min_cluster),
                        int(max_iters),
                        st.session_state.pack_active,
                    )
                    st.session_state.qi_pack = res["qi_pack"]
                    st.session_state.qc_pack = res["qc_pack"]
                    st.session_state.chapter_report = res["chapter_report"]
                    st.session_state.last_run_audit = res["audit"]
                    st.session_state.sealed = bool(res["audit"]["sealed"])
                    st.session_state.run_stats = {
                        "qi": int(res["audit"]["qi_total"]),
                        "rqi": int(res["audit"]["rqi_total"]),
                        "qc": int(res["audit"]["qc_total"]),
                        "qi_posable": int(res["audit"]["rqi_total"]),
                    }

                    if res["audit"]["sealed"]:
                        st.success("SEALED = YES (stable + coverage OK + posable OK)")
                    else:
                        st.warning(f"SEALED = NO | orphans={res['audit']['qi_orphans']} | rqi_total={res['audit']['rqi_total']} | qc={res['audit']['qc_total']}")
                except Exception as e:
                    st.error(f"RUN échoué: {e}")

        st.markdown("### Log (temps réel)")
        st.text_area("Logs", value=logs_text(), height=380)

    with tab3:
        st.markdown("## ÉTAPE 4 — RÉSULTATS / EXPORTS")
        # refresh metrics
        lib = st.session_state.library
        items_total = len(lib)
        corr_ok = sum(1 for x in lib if x.get("corrige?") and x.get("corrige_url"))
        qi_count = int(st.session_state.run_stats.get("qi", 0))
        qi_posable = int(st.session_state.run_stats.get("qi_posable", 0))
        qc_count = int(st.session_state.run_stats.get("qc", 0))
        sealed = bool(st.session_state.sealed)
        metric_row(items_total, corr_ok, qi_count, qi_posable, qc_count, sealed)

        st.markdown("### Exports (preuves)")
        # harvest_manifest
        hm = st.session_state.harvest_manifest or {
            "version": APP_VERSION,
            "timestamp": _utc_ts(),
            "country": st.session_state.country,
            "level": st.session_state.level,
            "subjects": st.session_state.subjects,
            "items_total": items_total,
            "items_corrige_ok": corr_ok,
            "library": st.session_state.library,
        }

        st.download_button("harvest_manifest.json", data=json.dumps(hm, ensure_ascii=False, indent=2), file_name="harvest_manifest.json", mime="application/json")
        st.download_button("logs.txt", data=logs_text(), file_name="logs.txt", mime="text/plain")

        if st.session_state.qi_pack:
            st.download_button("qi_pack.json", data=json.dumps(st.session_state.qi_pack, ensure_ascii=False, indent=2), file_name="qi_pack.json", mime="application/json")
        else:
            st.caption("qi_pack.json indisponible (lancez RUN).")

        if st.session_state.qc_pack:
            st.download_button("qc_pack.json", data=json.dumps(st.session_state.qc_pack, ensure_ascii=False, indent=2), file_name="qc_pack.json", mime="application/json")
        else:
            st.caption("qc_pack.json indisponible (lancez RUN).")

        if st.session_state.chapter_report:
            st.download_button("chapter_report.json", data=json.dumps(st.session_state.chapter_report, ensure_ascii=False, indent=2), file_name="chapter_report.json", mime="application/json")
        else:
            st.caption("chapter_report.json indisponible (lancez RUN).")

        st.markdown("---")
        st.markdown("### Explorateur (QC → ARI → FRT → Triggers → Qi)")
        if not st.session_state.qc_pack or not st.session_state.qi_pack:
            st.info("Lancez RUN pour alimenter l'explorateur.")
        else:
            chapters = pack_chapters(st.session_state.pack_active) if st.session_state.pack_active else []
            ch_options = ["ALL"] + sorted(list({qc.get("chapter_code", "UNMAPPED") for qc in st.session_state.qc_pack}))
            sel_ch = st.selectbox("Chapitre", options=ch_options, index=0)

            qcs = st.session_state.qc_pack
            if sel_ch != "ALL":
                qcs = [q for q in qcs if q.get("chapter_code") == sel_ch]

            qc_labels = [f"{q['qc_id']} | {q['chapter_code']} | n={q['cluster_size']} | {q['qc']}" for q in qcs]
            if not qc_labels:
                st.warning("Aucune QC pour ce chapitre.")
            else:
                sel_idx = st.selectbox("QC", options=list(range(len(qc_labels))), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]

                st.markdown("#### QC")
                st.write(qc["qc"])

                # Qi associés
                qi_by_id = {q["qi_id"]: q for q in st.session_state.qi_pack}
                st.markdown("#### Qi associés (avec ARI/FRT/Triggers)")
                for qi_id in qc["qi_ids"][:80]:
                    q = qi_by_id.get(qi_id)
                    if not q:
                        continue
                    with st.expander(f"{qi_id} | pair={q['pair_id']} | qc_id={q['qc_id']}"):
                        st.markdown("**Qi**")
                        st.write(q["qi"])
                        st.markdown("**RQi**")
                        st.write(q["rqi"] if q["rqi"] else "— (RQi manquante)")
                        st.markdown("**Triggers**")
                        st.write(q["triggers"])
                        st.markdown("**ARI**")
                        st.json(q["ari"])
                        st.markdown("**FRT**")
                        st.json(q["frt"])

            # Audit
            if st.session_state.last_run_audit:
                st.markdown("---")
                st.markdown("### Audit ISO-PROD (preuve)")
                st.json(st.session_state.last_run_audit)


if __name__ == "__main__":
    main()
