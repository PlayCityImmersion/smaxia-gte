# =============================================================================
# SMAXIA GTE Console V31.10.19 — ISO-PROD (FICHIER UNIQUE)
# =============================================================================
# FUSION FINALE:
# - V31.10.17: Corrections saturation, clustering stable
# - Manus V32: Architecture Pack-Driven, F1/F2 conformes
# - GPT Audit: Externalisation des données métier dans le Pack
#
# USAGE:
# 1. Copier ce fichier dans smaxia_console_v31.py
# 2. Lancer: streamlit run smaxia_console_v31.py
# 3. Sélectionner le pays → Cliquer "ACTIVER" → Pack auto-généré
# 4. HARVEST → RUN → Vérifier SEALED=YES
#
# ARCHITECTURE:
# - KERNEL: Code invariant (formules F1/F2, algorithmes)
# - PACK: Données métier (chapitres, patterns ARI, URLs harvest)
# - Pour TEST: Pack généré automatiquement depuis l'UI
# - Pour PROD: Pack uploadé depuis fichier JSON externe
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import hashlib
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests
import streamlit as st

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


# =============================================================================
# KERNEL CONSTANTS (INVARIANTS MATHÉMATIQUES)
# =============================================================================
APP_VERSION = "V31.10.19"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

# Constantes F1/F2 (Kernel V10.6.1)
F1_EPSILON = 1e-6
F2_ALPHA = 1.0
F2_TREC = 1.0


# =============================================================================
# PACK GENESIS — DONNÉES MÉTIER INTÉGRÉES (PHASE TEST)
# =============================================================================
# NOTE: En PROD, ces données seraient dans des fichiers JSON externes.
# Pour la phase TEST, elles sont intégrées pour permettre un fichier unique.

def _genesis_pack_fr() -> Dict[str, Any]:
    """Pack Genesis France - Math Terminale"""
    return {
        "pack_id": "CAP_FR_GENESIS_V1",
        "pack_version": "1.0.0",
        "country_code": "FR",
        "country_name": "France",
        "language": "fr",
        "status": "GENESIS_TEST",
        
        "harvest_sources": [{
            "source_id": "APMEP",
            "source_name": "APMEP - Annales du Bac",
            "base_url": "https://www.apmep.fr",
            "levels": {
                "TERMINALE": "/Annales-du-Bac-Terminale",
                "PREMIERE": "/Annales-du-Bac-Premiere"
            },
            "year_pattern": r"Annee-(20\d{2})",
            "pdf_patterns": {
                "sujet": ["spe", "sujet", "enonce"],
                "corrige": ["corrig", "correction", "solution"]
            },
            "meta_exclude": ["index", "sommaire", "liste", "annexe", "grille", "formulaire"],
            "geographic_zones": ["metro", "metropole", "amerique", "nord", "sud", "asie", "polynesie", "etranger", "antilles", "liban"]
        }],
        
        "text_processing": {
            "common_words": [
                "le", "la", "les", "de", "du", "des", "un", "une", "et", "ou", "en", "que", "qui",
                "est", "sont", "par", "pour", "sur", "dans", "avec", "ce", "se", "ne", "pas",
                "au", "aux", "son", "sa", "ses", "tout", "tous", "on", "nous", "vous",
                "soit", "alors", "ainsi", "comme", "bien", "peut", "fait", "plus", "moins",
                "fonction", "nombre", "valeur", "point", "droite", "plan", "equation",
                "exercice", "partie", "question", "montrer", "demontrer", "calculer"
            ],
            "glued_patterns": [
                {"pattern": r"(\d)([a-zA-Z])", "replacement": r"\1 \2"},
                {"pattern": r"([a-zA-Z])(\d)", "replacement": r"\1 \2"},
                {"pattern": r"([a-z])([A-Z])", "replacement": r"\1 \2"}
            ],
            "intent_verbs": [
                "montrer", "demontrer", "prouver", "justifier", "determiner", "calculer",
                "resoudre", "etudier", "donner", "exprimer", "simplifier", "trouver",
                "verifier", "deduire", "conclure", "etablir", "tracer", "representer"
            ],
            "trigger_keywords": [
                "limite", "derivee", "integrale", "probabilite", "suite", "vecteur",
                "complexe", "equation", "fonction", "recurrence", "matrice"
            ]
        },
        
        "ari_config": {
            "op_patterns": [
                {"pattern": r"\b(probabilit|proba|loi\s+binomiale|loi\s+normale|esperance|variance)\b", "op": "OP_PROBABILITY"},
                {"pattern": r"\b(deriv|f'\s*\(|tangente)\b", "op": "OP_DERIVE"},
                {"pattern": r"\b(integr|primitive|aire\s+sous)\b", "op": "OP_INTEGRATE"},
                {"pattern": r"\b(limit|tend\s+vers|infini)\b", "op": "OP_LIMIT"},
                {"pattern": r"\b(recurr|induction|heredite|initialisation)\b", "op": "OP_INDUCTION"},
                {"pattern": r"\b(complex|imaginaire|module|argument|affixe)\b", "op": "OP_COMPLEX"},
                {"pattern": r"\b(vecteur|scalaire|orthogonal|colineaire)\b", "op": "OP_VECTOR_GEOM"},
                {"pattern": r"\b(cos|sin|tan|trigo|radian)\b", "op": "OP_TRIGO"},
                {"pattern": r"\b(ln|log|exp|exponentiel)\b", "op": "OP_LOGEXP"},
                {"pattern": r"\b(equat|resou|racine|solution|discriminant)\b", "op": "OP_SOLVE_EQUATION"},
                {"pattern": r"\b(tableau\s+de\s+variation|signe|croissant|decroissant)\b", "op": "OP_VARIATION_TABLE"},
                {"pattern": r"\b(demontr|prouv|justifi|montr)\b", "op": "OP_PROVE"}
            ],
            "primary_ops_order": [
                "OP_PROBABILITY", "OP_DERIVE", "OP_INTEGRATE", "OP_LIMIT",
                "OP_INDUCTION", "OP_COMPLEX", "OP_VECTOR_GEOM", "OP_TRIGO",
                "OP_LOGEXP", "OP_SOLVE_EQUATION", "OP_VARIATION_TABLE", "OP_PROVE", "OP_STANDARD"
            ],
            "op_labels": {
                "OP_PROBABILITY": "Calculer une probabilité",
                "OP_DERIVE": "Dériver une fonction",
                "OP_INTEGRATE": "Intégrer une fonction",
                "OP_LIMIT": "Calculer une limite",
                "OP_INDUCTION": "Démontrer par récurrence",
                "OP_COMPLEX": "Manipuler des nombres complexes",
                "OP_VECTOR_GEOM": "Résoudre en géométrie vectorielle",
                "OP_TRIGO": "Appliquer la trigonométrie",
                "OP_LOGEXP": "Utiliser logarithme/exponentielle",
                "OP_SOLVE_EQUATION": "Résoudre une équation",
                "OP_VARIATION_TABLE": "Établir un tableau de variation",
                "OP_PROVE": "Démontrer/prouver",
                "OP_STANDARD": "Résoudre un exercice standard"
            }
        },
        
        "chapter_taxonomy": {
            "MATH": {
                "TERMINALE": [
                    {"code": "CH_ANALYSE", "label": "Analyse", "keywords": ["limite", "derivee", "continuite", "asymptote", "variation", "tangente", "convexite"], "delta_c": 1.0},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "keywords": ["probabilite", "loi", "binomiale", "normale", "esperance", "variance", "aleatoire"], "delta_c": 1.0},
                    {"code": "CH_SUITES", "label": "Suites", "keywords": ["suite", "recurrence", "arithmetique", "geometrique", "convergence", "terme", "rang"], "delta_c": 1.0},
                    {"code": "CH_INTEGRATION", "label": "Intégration", "keywords": ["integrale", "primitive", "aire", "integration", "parties"], "delta_c": 1.0},
                    {"code": "CH_GEOMETRIE", "label": "Géométrie espace", "keywords": ["vecteur", "plan", "droite", "espace", "orthogonal", "colineaire"], "delta_c": 1.0},
                    {"code": "CH_COMPLEXES", "label": "Nombres complexes", "keywords": ["complexe", "imaginaire", "module", "argument", "affixe"], "delta_c": 1.0},
                    {"code": "CH_LOGEXP", "label": "Log & Exp", "keywords": ["exponentielle", "logarithme", "ln", "exp"], "delta_c": 1.0},
                    {"code": "CH_TRIGO", "label": "Trigonométrie", "keywords": ["cosinus", "sinus", "tangente", "radian", "cercle"], "delta_c": 1.0}
                ],
                "PREMIERE": [
                    {"code": "CH_ANALYSE", "label": "Analyse", "keywords": ["derivee", "variation", "fonction", "tangente"], "delta_c": 1.0},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "keywords": ["probabilite", "evenement", "aleatoire"], "delta_c": 1.0},
                    {"code": "CH_SUITES", "label": "Suites", "keywords": ["suite", "arithmetique", "geometrique"], "delta_c": 1.0}
                ]
            }
        },
        
        "f1_f2_params": {"epsilon": F1_EPSILON, "alpha": F2_ALPHA, "t_rec_default": F2_TREC}
    }


def _genesis_pack_ci() -> Dict[str, Any]:
    """Pack Genesis Côte d'Ivoire - Math Terminale"""
    pack = _genesis_pack_fr()
    pack["pack_id"] = "CAP_CI_GENESIS_V1"
    pack["country_code"] = "CI"
    pack["country_name"] = "Côte d'Ivoire"
    # Même source APMEP pour les tests (à adapter pour PROD)
    return pack


def _genesis_pack_sn() -> Dict[str, Any]:
    """Pack Genesis Sénégal - Math Terminale"""
    pack = _genesis_pack_fr()
    pack["pack_id"] = "CAP_SN_GENESIS_V1"
    pack["country_code"] = "SN"
    pack["country_name"] = "Sénégal"
    return pack


# Registre des Packs Genesis disponibles
GENESIS_PACKS = {
    "FR": ("France", _genesis_pack_fr),
    "CI": ("Côte d'Ivoire", _genesis_pack_ci),
    "SN": ("Sénégal", _genesis_pack_sn),
}


def generate_pack(country_code: str) -> Dict[str, Any]:
    """Génère un Pack pour le pays sélectionné."""
    if country_code not in GENESIS_PACKS:
        raise ValueError(f"Pays non supporté: {country_code}")
    
    _, generator = GENESIS_PACKS[country_code]
    pack = generator()
    pack["created_at"] = _utc_ts()
    pack["_source"] = "GENESIS_AUTO"
    pack["_pack_sig_sha256"] = hashlib.sha256(
        json.dumps(pack, sort_keys=True).encode()
    ).hexdigest()
    return pack


# =============================================================================
# SESSION & LOGS
# =============================================================================
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ss_init():
    """Initialise l'état de session Streamlit."""
    st.session_state.setdefault("pack_active", None)
    st.session_state.setdefault("pack_id", None)
    st.session_state.setdefault("pack_sig_sha256", None)
    st.session_state.setdefault("country", None)
    st.session_state.setdefault("level", None)
    st.session_state.setdefault("subject", None)
    st.session_state.setdefault("library", [])
    st.session_state.setdefault("harvest_manifest", None)
    st.session_state.setdefault("qi_pack", None)
    st.session_state.setdefault("qc_pack", None)
    st.session_state.setdefault("selection_report", None)
    st.session_state.setdefault("sealed", False)
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("run_stats", {
        "qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0, "orphans": 0, "sanity_ok": False
    })
    st.session_state.setdefault("last_run_audit", None)
    st.session_state.setdefault("_http_pdf_cache", {})


def log(msg: str):
    st.session_state.logs.append(f"[{_utc_ts()}] {msg}")


def logs_text() -> str:
    return "\n".join(st.session_state.logs[-8000:])


# =============================================================================
# TEXT UTILS (KERNEL INVARIANT)
# =============================================================================
def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:12]


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", norm_text(s))[:2000]


def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# =============================================================================
# PACK ACCESSORS
# =============================================================================
def pack_chapters(pack: Dict[str, Any], subject: str, level: str) -> List[Dict[str, Any]]:
    taxonomy = pack.get("chapter_taxonomy", {})
    return taxonomy.get(subject, {}).get(level, [])


def pack_harvest_source(pack: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sources = pack.get("harvest_sources", [])
    return sources[0] if sources else None


def pack_text_processing(pack: Dict[str, Any]) -> Dict[str, Any]:
    return pack.get("text_processing", {})


def pack_ari_config(pack: Dict[str, Any]) -> Dict[str, Any]:
    return pack.get("ari_config", {})


def pack_f1f2_params(pack: Dict[str, Any]) -> Dict[str, float]:
    params = pack.get("f1_f2_params", {})
    return {
        "epsilon": float(params.get("epsilon", F1_EPSILON)),
        "alpha": float(params.get("alpha", F2_ALPHA)),
        "t_rec": float(params.get("t_rec_default", F2_TREC)),
    }


# =============================================================================
# PDF TEXT EXTRACTION (PACK-DRIVEN)
# =============================================================================
def _fix_missing_spaces(text: str, text_proc: Dict[str, Any]) -> str:
    if not text:
        return ""
    
    for item in text_proc.get("glued_patterns", []):
        try:
            text = re.sub(item["pattern"], item["replacement"], text)
        except re.error:
            pass
    
    common_words = set(text_proc.get("common_words", []))
    if not common_words:
        return re.sub(r"\s+", " ", text).strip()
    
    def split_glued(match):
        word = match.group(0)
        if len(word) < 8:
            return word
        result, i = [], 0
        while i < len(word):
            found = False
            for length in range(min(12, len(word) - i), 2, -1):
                candidate = word[i:i+length].lower()
                if candidate in common_words:
                    if result:
                        result.append(" ")
                    result.append(word[i:i+length])
                    i += length
                    found = True
                    break
            if not found:
                result.append(word[i])
                i += 1
        reconstructed = "".join(result)
        return reconstructed if " " in reconstructed else word
    
    text = re.sub(r"[a-zA-ZéèêëàâäùûüôöîïçÉÈÊËÀÂÄÙÛÜÔÖÎÏÇ]{15,}", split_glued, text)
    return re.sub(r"\s+", " ", text).strip()


def _dehyphenate(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_pages_pdfplumber(pdf_bytes: bytes, max_pages: int = 120) -> List[str]:
    if not pdfplumber:
        return []
    pages = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:max_pages]:
                try:
                    pages.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
                except:
                    pages.append("")
    except:
        pass
    return pages


def _extract_pages_pypdf(pdf_bytes: bytes, max_pages: int = 120) -> List[str]:
    if not PdfReader:
        return []
    pages = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages[:max_pages]:
            try:
                pages.append(page.extract_text() or "")
            except:
                pages.append("")
    except:
        pass
    return pages


def clean_pdf_text(pages: List[str], text_proc: Dict[str, Any]) -> str:
    if not pages:
        return ""
    
    page_lines = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = _dehyphenate(p)
        p = _fix_missing_spaces(p, text_proc)
        page_lines.append([ln.strip() for ln in p.split("\n") if ln.strip()])
    
    # Détection headers/footers répétés
    top_counts, bot_counts = {}, {}
    for lines in page_lines:
        if lines:
            top_counts[lines[0]] = top_counts.get(lines[0], 0) + 1
        if len(lines) > 1:
            bot_counts[lines[-1]] = bot_counts.get(lines[-1], 0) + 1
    
    threshold = max(2, len(page_lines) // 3)
    skip_top = {k for k, v in top_counts.items() if v >= threshold}
    skip_bot = {k for k, v in bot_counts.items() if v >= threshold}
    
    out_lines = []
    for lines in page_lines:
        for i, ln in enumerate(lines):
            if i == 0 and ln in skip_top:
                continue
            if i == len(lines) - 1 and ln in skip_bot:
                continue
            if re.fullmatch(r"\d{1,3}", ln.strip()):
                continue
            out_lines.append(ln)
        out_lines.append("")
    
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


def extract_text_from_pdf(pdf_bytes: bytes, text_proc: Dict[str, Any]) -> str:
    pages = _extract_pages_pdfplumber(pdf_bytes)
    if not pages:
        pages = _extract_pages_pypdf(pdf_bytes)
    return clean_pdf_text(pages, text_proc)


# =============================================================================
# HTTP & PDF FETCH
# =============================================================================
def _http_get(url: str) -> requests.Response:
    res = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    res.raise_for_status()
    return res


def fetch_pdf_bytes(url: str) -> bytes:
    cache = st.session_state.get("_http_pdf_cache", {})
    if url in cache:
        return cache[url]
    
    res = _http_get(url)
    pdf_bytes = res.content
    
    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError(f"PDF trop volumineux")
    
    cache[url] = pdf_bytes
    st.session_state["_http_pdf_cache"] = cache
    return pdf_bytes


# =============================================================================
# HARVEST (PACK-DRIVEN)
# =============================================================================
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé")
    return BeautifulSoup(html, "html.parser")


def _abs_url(base: str, href: str) -> str:
    return href if href.startswith("http") else requests.compat.urljoin(base, href)


def harvest_from_pack(pack: Dict[str, Any], level: str, subject: str,
                      years_back: int, volume_max: int) -> Dict[str, Any]:
    """Harvest piloté par le Pack."""
    source = pack_harvest_source(pack)
    if not source:
        raise ValueError("Aucune source harvest dans le Pack")
    
    base_url = source.get("base_url", "")
    level_path = source.get("levels", {}).get(level)
    if not level_path:
        raise ValueError(f"Niveau '{level}' non configuré")
    
    root = f"{base_url}{level_path}"
    log(f"[HARVEST] root={root}")
    
    year_pattern = source.get("year_pattern", r"(20\d{2})")
    meta_exclude = source.get("meta_exclude", [])
    corrige_patterns = source.get("pdf_patterns", {}).get("corrige", [])
    geo_zones = source.get("geographic_zones", [])
    
    html = _http_get(root).text
    sp = _soup(html)
    
    year_links = []
    for a in sp.find_all("a"):
        href = a.get("href") or ""
        m = re.search(year_pattern, href)
        if m:
            year_links.append((int(m.group(1)), _abs_url(root, href)))
    
    year_links = sorted(set(year_links), key=lambda x: -x[0])
    if not year_links:
        raise RuntimeError("Aucune page Année trouvée")
    
    current_year = year_links[0][0]
    min_year = current_year - max(1, years_back) + 1
    selected = [(y, u) for (y, u) in year_links if y >= min_year]
    log(f"[HARVEST] années={len(selected)}")
    
    pairs = []
    corrige_ok = 0
    
    for (y, url) in selected:
        if len(pairs) >= volume_max:
            break
        try:
            y_html = _http_get(url).text
            y_sp = _soup(y_html)
            pdf_links = []
            
            for a in y_sp.find_all("a"):
                href = (a.get("href") or "").strip()
                if not href.lower().endswith(".pdf"):
                    continue
                absu = _abs_url(url, href)
                label = (a.get_text() or "").strip()
                
                s = norm_text(absu + " " + label)
                if any(w in s for w in meta_exclude):
                    continue
                
                is_corrige = any(p in s for p in corrige_patterns)
                pdf_links.append({"url": absu, "name": os.path.basename(absu), "is_corrige": is_corrige})
            
            sujets = [p for p in pdf_links if not p["is_corrige"]]
            corriges = [p for p in pdf_links if p["is_corrige"]]
            used_corr = set()
            
            for suj in sujets:
                if len(pairs) >= volume_max:
                    break
                
                best = (None, 0.0)
                s_name = norm_text(suj["name"])
                s_geo = next((z for z in geo_zones if z in s_name), "")
                
                for corr in corriges:
                    if corr["url"] in used_corr:
                        continue
                    
                    c_name = norm_text(corr["name"])
                    c_geo = next((z for z in geo_zones if z in c_name), "")
                    
                    score = 0.0
                    if s_geo and c_geo:
                        if s_geo == c_geo:
                            score += 0.5
                        else:
                            continue
                    
                    s_tok, c_tok = set(tokenize(suj["name"])), set(tokenize(corr["name"]))
                    if s_tok and c_tok:
                        score += 0.3 * len(s_tok & c_tok) / len(s_tok | c_tok)
                    
                    if score > best[1]:
                        best = (corr, score)
                
                corrige = best[0] if best[1] >= 0.3 else None
                if corrige:
                    used_corr.add(corrige["url"])
                
                item = {
                    "pair_id": f"PAIR_{level}_{subject}_{y}_{stable_id(suj['name'])}",
                    "year": y,
                    "sujet": suj["name"],
                    "corrige?": bool(corrige),
                    "corrige_name": corrige["name"] if corrige else "",
                    "match_score": round(best[1], 2) if corrige else 0.0,
                    "sujet_url": suj["url"],
                    "corrige_url": corrige["url"] if corrige else "",
                }
                
                if not any(x["sujet_url"] == item["sujet_url"] for x in pairs):
                    pairs.append(item)
                    if item["corrige?"]:
                        corrige_ok += 1
            
            log(f"[HARVEST] year={y} pairs={len([p for p in pairs if p['year']==y])}")
        except Exception as e:
            log(f"[HARVEST] year={y} ERREUR: {e}")
    
    return {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": pack.get("pack_id"),
        "items_total": len(pairs),
        "items_corrige_ok": corrige_ok,
        "library": pairs,
    }


# =============================================================================
# Qi/RQi EXTRACTION
# =============================================================================
_EXERCICE_RE = re.compile(r"(?i)(?:^|\n)\s*(?:EXERCICE|Ex\.?)\s*(\d+|[IVX]+)", re.MULTILINE)
_PARTIE_RE = re.compile(r"(?i)(?:^|\n)\s*(?:PARTIE|Part)\s*([A-Z]|\d+)", re.MULTILINE)
_QUESTION_RE = re.compile(r"(?m)^\s*(\d{1,2}|[a-h])\s*[\)\.\-:]\s+")
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√]")


def _build_intent_re(intent_verbs: List[str]) -> Optional[re.Pattern]:
    if not intent_verbs:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(v) for v in intent_verbs) + r")\b", re.IGNORECASE)


def _chunk_candidates(text: str) -> List[str]:
    t = (text or "").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []
    
    cut_points = {0}
    for m in _EXERCICE_RE.finditer(t):
        cut_points.add(m.start())
    for m in _PARTIE_RE.finditer(t):
        cut_points.add(m.start())
    for m in _QUESTION_RE.finditer(t):
        cut_points.add(m.start())
    for m in re.finditer(r"\n\n+", t):
        cut_points.add(m.end())
    
    idxs = sorted(cut_points)
    chunks = []
    for a, b in zip(idxs, idxs[1:] + [len(t)]):
        c = re.sub(r"\s+", " ", t[a:b]).strip()
        if c and len(c) > 20:
            chunks.append(c)
    return chunks


def _looks_like_question(s: str, intent_re) -> bool:
    if not s or len(s) < 25:
        return False
    has_intent = bool(intent_re and intent_re.search(s)) or ("?" in s)
    has_math = bool(_MATH_SYMBOL_RE.search(s)) or bool(re.search(r"\b\d+\b", s))
    return has_intent or (has_math and len(s) > 40)


def split_questions(text: str, text_proc: Dict[str, Any], max_items: int = 200) -> Tuple[List[str], Dict]:
    intent_re = _build_intent_re(text_proc.get("intent_verbs", []))
    raw_chunks = _chunk_candidates(text)
    
    keep, seen = [], set()
    for c in raw_chunks:
        key = stable_id(norm_text(c)[:400])
        if key in seen:
            continue
        seen.add(key)
        if len(c) > 2500:
            c = c[:2500]
        if _looks_like_question(c, intent_re):
            keep.append(c)
        if len(keep) >= max_items:
            break
    
    return keep, {"raw": len(raw_chunks), "kept": len(keep)}


def align_qi_rqi(questions: List[str], responses: List[str]) -> List[Optional[int]]:
    link = [None] * len(questions)
    if not questions or not responses:
        return link
    
    q_tokens = [tokenize(q) for q in questions]
    r_tokens = [tokenize(r) for r in responses]
    used_r = set()
    
    for i, q_tok in enumerate(q_tokens):
        best_j, best_s = None, 0.0
        for j, r_tok in enumerate(r_tokens):
            if j in used_r:
                continue
            s = jaccard(q_tok, r_tok)
            if s > best_s:
                best_s, best_j = s, j
        
        if best_j is not None and best_s >= 0.08:
            link[i] = best_j
            used_r.add(best_j)
    
    return link


# =============================================================================
# ARI EXTRACTION (PACK-DRIVEN)
# =============================================================================
def extract_op_trace(question: str, response: str, ari_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    op_patterns = ari_config.get("op_patterns", [])
    if not op_patterns:
        return [{"op": "OP_STANDARD", "confidence": 0.5}]
    
    combined = norm_text(f"{question} {response}")
    ops, seen = [], set()
    
    for item in op_patterns:
        pattern = item.get("pattern", "")
        op_code = item.get("op", "")
        if not pattern or not op_code:
            continue
        try:
            if re.search(pattern, combined, re.IGNORECASE) and op_code not in seen:
                ops.append({"op": op_code, "confidence": 0.8})
                seen.add(op_code)
        except re.error:
            pass
    
    return ops if ops else [{"op": "OP_STANDARD", "confidence": 0.5}]


def get_primary_op(op_trace: List[Dict[str, Any]], ari_config: Dict[str, Any]) -> str:
    primary_order = ari_config.get("primary_ops_order", [])
    ops_in_trace = {x["op"] for x in op_trace}
    
    for op in primary_order:
        if op in ops_in_trace:
            return op
    return op_trace[0]["op"] if op_trace else "OP_STANDARD"


def build_triggers(question: str, op_trace: List[Dict[str, Any]], text_proc: Dict[str, Any]) -> List[str]:
    triggers = [f"ARI:{op['op']}" for op in op_trace]
    
    q_norm = norm_text(question)
    for kw in text_proc.get("trigger_keywords", []):
        if kw in q_norm:
            triggers.append(f"KW:{kw.upper()}")
    
    return list(set(triggers))[:15]


def map_qi_to_chapter(question: str, chapters: List[Dict[str, Any]]) -> str:
    if not chapters:
        return "UNMAPPED"
    
    q_norm = norm_text(question)
    best_ch, best_score = "UNMAPPED", 0
    
    for ch in chapters:
        score = sum(1 for kw in ch.get("keywords", []) if kw in q_norm)
        if score > best_score:
            best_score, best_ch = score, ch.get("code", "UNMAPPED")
    
    return best_ch


# =============================================================================
# QC GENERATION
# =============================================================================
def qc_method_label(primary_op: str, ari_config: Dict[str, Any]) -> str:
    labels = ari_config.get("op_labels", {})
    return labels.get(primary_op, primary_op.replace("OP_", "").title())


def build_qc_from_qi(qi_pack: List[Dict[str, Any]], chapters: List[Dict[str, Any]],
                     ari_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    # Bucketing par (chapter, primary_op)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for qi in qi_pack:
        key = (qi.get("chapter_code", "UNMAPPED"), qi.get("primary_op", "OP_STANDARD"))
        buckets.setdefault(key, []).append(qi)
    
    log(f"[QC] Buckets: {len(buckets)}")
    
    # Compter buckets par chapitre
    ch_counts: Dict[str, int] = {}
    for (cc, _) in buckets:
        ch_counts[cc] = ch_counts.get(cc, 0) + 1
    
    qc_pack, qc_map = [], {}
    
    for (cc, primary_op), items in sorted(buckets.items()):
        posable = [x for x in items if x.get("has_rqi")]
        cluster_size = len(items)
        posable_count = len(posable)
        
        # Anti-singleton: créer QC si cluster>=2 OU unique dans chapitre OU posable>0
        is_singleton = cluster_size < 2
        is_only = ch_counts.get(cc, 0) <= 1
        
        if is_singleton and not is_only and posable_count == 0:
            continue
        
        qc_id = f"QC_{stable_id(cc, primary_op)}"
        qc_text = qc_method_label(primary_op, ari_config)
        
        if posable_count >= 2:
            qc_state = "POSABLE"
        elif posable_count == 1:
            qc_state = "POSABLE_WEAK"
        else:
            qc_state = "UNPOSABLE"
        
        # Triggers agrégés
        trig_counts: Dict[str, int] = {}
        for it in items:
            for t in it.get("triggers", []):
                trig_counts[t] = trig_counts.get(t, 0) + 1
        
        all_ops = set()
        for it in items:
            all_ops.update(it.get("ari_ops", []))
        
        qc_pack.append({
            "qc_id": qc_id,
            "qc": qc_text,
            "chapter_code": cc,
            "primary_op": primary_op,
            "all_ops": list(all_ops),
            "cluster_size": cluster_size,
            "posable_in_cluster": posable_count,
            "qc_state": qc_state,
            "qi_ids": [x["qi_id"] for x in items],
            "triggers_hint": [k for k, _ in sorted(trig_counts.items(), key=lambda x: -x[1])][:10],
        })
        
        for it in items:
            qc_map[it["qi_id"]] = qc_id
    
    log(f"[QC] Créées: {len(qc_pack)}")
    return qc_pack, qc_map


# =============================================================================
# F1/F2 FORMULAS
# =============================================================================
def compute_trigger_weights(triggers: List[str]) -> Dict[str, float]:
    uniq = list(dict.fromkeys(triggers or []))
    n = len(uniq)
    return {t: 1.0/n for t in uniq} if n else {}


def f1_raw(delta_c: float, epsilon: float, Tj: Dict[str, float], m_q: int) -> float:
    s = sum(v for _, v in sorted(Tj.items())[:m_q]) if Tj else 0.0
    return delta_c * (epsilon + s) ** 2


def f1_normalize_in_chapter(qc_list: List[Dict[str, Any]]) -> None:
    if not qc_list:
        return
    raws = [q.get("psi_raw", 0.0) for q in qc_list]
    mx = max(raws) if raws else 0.0
    for q in qc_list:
        q["Psi_q"] = q.get("psi_raw", 0.0) / mx if mx > 0 else 0.0


def sigma_similarity(qc_a: Dict, qc_b: Dict) -> float:
    a_ops, b_ops = set(qc_a.get("all_ops", [])), set(qc_b.get("all_ops", []))
    if a_ops and b_ops:
        return len(a_ops & b_ops) / len(a_ops | b_ops)
    return 0.0


def f2_score(qc: Dict, selected: List[Dict], n_hist: int, N_total: int, alpha: float, t_rec: float) -> float:
    Psi_q = qc.get("Psi_q", 0.0)
    base = (n_hist / max(1, N_total)) * (1.0 + alpha / max(1e-9, t_rec)) * Psi_q
    
    prod = 1.0
    for p in selected:
        prod *= (1.0 - sigma_similarity(qc, p)) * 100.0
    
    return base * prod


def progressive_select(qc_list: List[Dict], N_total: int, f1f2_params: Dict, top_k: int = 12) -> List[Dict]:
    selected = []
    remaining = [q for q in qc_list if q.get("qc_state") in ("POSABLE", "POSABLE_WEAK")]
    
    alpha, t_rec = f1f2_params.get("alpha", F2_ALPHA), f1f2_params.get("t_rec", F2_TREC)
    
    for _ in range(min(top_k, len(remaining))):
        best = None
        for qc in remaining:
            n_hist = qc.get("cluster_size", 1)
            sc = f2_score(qc, selected, n_hist, N_total, alpha, t_rec)
            qc["_f2_score"] = sc
            if best is None or sc > best["_f2_score"]:
                best = qc
        
        if best is None:
            break
        selected.append(best)
        remaining = [x for x in remaining if x["qc_id"] != best["qc_id"]]
    
    return selected


# =============================================================================
# SANITY GATE
# =============================================================================
def sanity_eval(qi_items: List[Dict]) -> Tuple[bool, Dict]:
    seen, dups = set(), 0
    for q in qi_items:
        k = stable_id(norm_text(q.get("qi", ""))[:300])
        if k in seen:
            dups += 1
        else:
            seen.add(k)
    
    dup_ratio = dups / max(1, len(qi_items))
    ok = dup_ratio <= 0.35
    return ok, {"dup_ratio": round(dup_ratio * 100, 2)}


# =============================================================================
# RUN PIPELINE
# =============================================================================
def run_phase(library: List[Dict], volume: int, pack: Dict) -> Dict[str, Any]:
    text_proc = pack_text_processing(pack)
    ari_config = pack_ari_config(pack)
    
    subject = st.session_state.get("subject", "MATH")
    level = st.session_state.get("level", "TERMINALE")
    chapters = pack_chapters(pack, subject, level)
    
    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    to_process = exploitable[:min(volume, len(exploitable))]
    log(f"[RUN] Processing {len(to_process)} pairs")
    
    qi_items = []
    
    for pair in to_process:
        pid = pair["pair_id"]
        try:
            su_pdf = fetch_pdf_bytes(pair["sujet_url"])
            co_pdf = fetch_pdf_bytes(pair["corrige_url"])
        except Exception as e:
            log(f"[DL] ERREUR {pid}: {e}")
            continue
        
        su_text = extract_text_from_pdf(su_pdf, text_proc)
        co_text = extract_text_from_pdf(co_pdf, text_proc)
        
        if not su_text or not co_text:
            continue
        
        qs, _ = split_questions(su_text, text_proc)
        rs, _ = split_questions(co_text, text_proc)
        link = align_qi_rqi(qs, rs)
        
        log(f"[ALIGN] {pid}: {len(qs)} Qi, {len(rs)} RQi, matched={sum(1 for x in link if x is not None)}")
        
        for i, q in enumerate(qs):
            if not q.strip():
                continue
            j = link[i] if i < len(link) else None
            r = rs[j] if j is not None and j < len(rs) else ""
            
            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
            
            op_trace = extract_op_trace(q, r, ari_config)
            ari_ops = [x["op"] for x in op_trace]
            primary_op = get_primary_op(op_trace, ari_config)
            triggers = build_triggers(q, op_trace, text_proc)
            chapter_code = map_qi_to_chapter(q, chapters)
            
            qi_items.append({
                "qi_id": qi_id,
                "pair_id": pid,
                "qi": q,
                "rqi": r,
                "has_rqi": bool(r),
                "chapter_code": chapter_code,
                "primary_op": primary_op,
                "ari_ops": ari_ops,
                "triggers": triggers,
            })
    
    # Sanity
    sanity_ok, sanity_audit = sanity_eval(qi_items)
    
    # Build QC
    qc_pack, qc_map = build_qc_from_qi(qi_items, chapters, ari_config)
    
    # F1
    f1f2_params = pack_f1f2_params(pack)
    for qc in qc_pack:
        ch_info = next((c for c in chapters if c.get("code") == qc["chapter_code"]), {})
        delta_c = ch_info.get("delta_c", 1.0)
        
        all_triggers = []
        for qi_id in qc.get("qi_ids", []):
            qi = next((q for q in qi_items if q["qi_id"] == qi_id), None)
            if qi:
                all_triggers.extend(qi.get("triggers", []))
        
        Tj = compute_trigger_weights(all_triggers)
        qc["psi_raw"] = f1_raw(delta_c, f1f2_params["epsilon"], Tj, len(Tj))
    
    # Normalize F1 by chapter
    by_ch: Dict[str, List] = {}
    for qc in qc_pack:
        by_ch.setdefault(qc["chapter_code"], []).append(qc)
    for qcs in by_ch.values():
        f1_normalize_in_chapter(qcs)
    
    # Mark orphans
    orphans = 0
    for qi in qi_items:
        if qi["qi_id"] not in qc_map:
            qi["is_orphan"] = True
            orphans += 1
        else:
            qi["is_orphan"] = False
            qi["qc_id"] = qc_map[qi["qi_id"]]
    
    # Stats
    qi_posable = sum(1 for q in qi_items if q.get("has_rqi") and not q.get("is_orphan"))
    qc_posable = sum(1 for qc in qc_pack if qc["qc_state"] in ("POSABLE", "POSABLE_WEAK"))
    
    return {
        "qi_pack": qi_items,
        "qc_pack": qc_pack,
        "qc_map": qc_map,
        "audit": {
            "qi_total": len(qi_items),
            "rqi_total": sum(1 for q in qi_items if q.get("has_rqi")),
            "qi_posable": qi_posable,
            "qi_orphans": orphans,
            "qc_total": len(qc_pack),
            "qc_posable": qc_posable,
            "sanity_ok": sanity_ok,
            "sanity_audit": sanity_audit,
        }
    }


def run_saturation_test(library: List[Dict], phase_a: int, phase_b: int, pack: Dict) -> Dict[str, Any]:
    log(f"[SATURATION] Phase A={phase_a}, Phase B={phase_b}")
    
    a_out = run_phase(library, phase_a, pack)
    b_out = run_phase(library, phase_b, pack)
    
    setA = {qc["qc_id"] for qc in a_out["qc_pack"]}
    setB = {qc["qc_id"] for qc in b_out["qc_pack"]}
    setB_minus_setA = sorted(setB - setA)
    
    saturation_ok = len(setB_minus_setA) == 0
    
    # F2 Selection
    f1f2_params = pack_f1f2_params(pack)
    N_total = b_out["audit"]["qi_posable"]
    
    subject = st.session_state.get("subject", "MATH")
    level = st.session_state.get("level", "TERMINALE")
    chapters = pack_chapters(pack, subject, level)
    
    selection = []
    by_ch: Dict[str, List] = {}
    for qc in b_out["qc_pack"]:
        by_ch.setdefault(qc["chapter_code"], []).append(qc)
    
    for cc, qcs in by_ch.items():
        selected = progressive_select(qcs, N_total, f1f2_params)
        ch_info = next((c for c in chapters if c.get("code") == cc), {"label": cc})
        selection.append({
            "chapter_code": cc,
            "chapter_label": ch_info.get("label", cc),
            "selected_qc": [{"qc_id": x["qc_id"], "qc": x["qc"], "Psi_q": x.get("Psi_q"), "f2_score": x.get("_f2_score")} for x in selected]
        })
    
    # SEALED
    orphans = b_out["audit"]["qi_orphans"]
    posable = b_out["audit"]["qi_posable"]
    sanity_ok = b_out["audit"]["sanity_ok"]
    qc_total = b_out["audit"]["qc_total"]
    qc_posable = b_out["audit"]["qc_posable"]
    
    sealed = saturation_ok and sanity_ok and orphans == 0 and posable > 0 and qc_total > 0 and qc_posable > 0
    
    return {
        "phaseA": a_out,
        "phaseB": b_out,
        "selection_report": {"chapters": selection},
        "audit": {
            "phaseA": a_out["audit"],
            "phaseB": b_out["audit"],
            "saturation_ok": saturation_ok,
            "setA_size": len(setA),
            "setB_size": len(setB),
            "setB_minus_setA_size": len(setB_minus_setA),
            "setB_minus_setA": setB_minus_setA[:20],
            "sealed": sealed,
        }
    }


# =============================================================================
# UI STREAMLIT
# =============================================================================
def metric_row(items, corr_ok, qi, qi_posable, qc, sealed):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pairs", items)
    c2.metric("Corrigés", corr_ok)
    c3.metric("Qi", qi)
    c4.metric("Qi POSABLE", qi_posable)
    c5.metric("QC", qc)
    c6.metric("SEALED", "YES" if sealed else "NO")


def main():
    st.set_page_config(page_title=f"SMAXIA GTE {APP_VERSION}", layout="wide")
    ss_init()
    
    st.markdown(f"# SMAXIA GTE Console {APP_VERSION}")
    st.caption("ISO-PROD | Pack-Driven | F1/F2 Conformes")
    
    # === SIDEBAR ===
    with st.sidebar:
        st.markdown("## 1. ACTIVATION PACK")
        
        # Sélection du pays
        country_options = list(GENESIS_PACKS.keys())
        country_labels = [f"{code} - {GENESIS_PACKS[code][0]}" for code in country_options]
        
        selected_idx = st.selectbox(
            "Sélectionner le pays",
            range(len(country_options)),
            format_func=lambda i: country_labels[i]
        )
        selected_country = country_options[selected_idx]
        
        # Upload Pack externe (optionnel)
        st.markdown("---")
        st.caption("Ou uploader un Pack externe (optionnel)")
        uploaded = st.file_uploader("Pack JSON", type=["json"])
        
        # Bouton ACTIVER
        if st.button("🚀 ACTIVER", use_container_width=True, type="primary"):
            try:
                if uploaded:
                    # Pack uploadé
                    pack = json.loads(uploaded.read().decode("utf-8"))
                    pack["_source"] = "UPLOAD"
                    pack["_pack_sig_sha256"] = hashlib.sha256(json.dumps(pack, sort_keys=True).encode()).hexdigest()
                    log(f"[PACK] Chargé depuis upload: {pack.get('pack_id')}")
                else:
                    # Pack Genesis auto-généré
                    pack = generate_pack(selected_country)
                    log(f"[PACK] Généré pour {selected_country}")
                
                st.session_state.pack_active = pack
                st.session_state.pack_id = pack.get("pack_id", "")
                st.session_state.pack_sig_sha256 = pack.get("_pack_sig_sha256", "")
                st.session_state.country = pack.get("country_code", "")
                
                # Dériver niveau et matière
                taxonomy = pack.get("chapter_taxonomy", {})
                subjects = list(taxonomy.keys())
                if subjects:
                    st.session_state.subject = subjects[0]
                    levels = list(taxonomy[subjects[0]].keys())
                    if levels:
                        st.session_state.level = levels[0]
                
                # Reset
                st.session_state.library = []
                st.session_state.qi_pack = None
                st.session_state.qc_pack = None
                st.session_state.sealed = False
                st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0, "orphans": 0, "sanity_ok": False}
                
                st.success(f"✅ Pack activé: {pack.get('pack_id')}")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur: {e}")
        
        # Afficher état Pack
        if st.session_state.pack_active:
            pack = st.session_state.pack_active
            st.success(f"✅ {st.session_state.pack_id}")
            st.caption(f"Pays: {pack.get('country_name', pack.get('country_code'))}")
            st.caption(f"Source: {pack.get('_source', 'N/A')}")
            
            st.markdown("---")
            st.markdown("## 2. SÉLECTION")
            
            taxonomy = pack.get("chapter_taxonomy", {})
            subjects = list(taxonomy.keys())
            
            if subjects:
                subject = st.selectbox("Matière", subjects, index=subjects.index(st.session_state.subject) if st.session_state.subject in subjects else 0)
                st.session_state.subject = subject
                
                levels = list(taxonomy.get(subject, {}).keys())
                if levels:
                    level = st.selectbox("Niveau", levels, index=levels.index(st.session_state.level) if st.session_state.level in levels else 0)
                    st.session_state.level = level
            
            # Chapitres
            chapters = pack_chapters(pack, st.session_state.subject, st.session_state.level)
            if chapters:
                st.markdown("### Chapitres")
                for ch in chapters[:6]:
                    st.caption(f"• {ch.get('label', ch.get('code'))}")
    
    # === MAIN ===
    if not st.session_state.pack_active:
        st.warning("⚠️ **Sélectionnez un pays et cliquez sur ACTIVER** dans la barre latérale.")
        return
    
    pack = st.session_state.pack_active
    
    tab1, tab2, tab3 = st.tabs(["Import", "RUN", "Exports"])
    
    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?"))
    
    with tab1:
        st.markdown("## Import")
        metric_row(len(lib), corr_ok, st.session_state.run_stats.get("qi", 0),
                   st.session_state.run_stats.get("qi_posable", 0),
                   st.session_state.run_stats.get("qc", 0), st.session_state.sealed)
        
        if lib:
            st.dataframe([{k: r.get(k, "") for k in ["pair_id", "year", "sujet", "corrige?", "match_score"]} for r in lib[:50]],
                         use_container_width=True, hide_index=True)
        
        st.markdown("### HARVEST")
        source = pack_harvest_source(pack)
        if source:
            st.info(f"Source: {source.get('source_name', source.get('source_id'))}")
            c1, c2 = st.columns(2)
            years_back = c1.number_input("Années", 1, 15, 5)
            volume_max = c2.number_input("Volume max", 5, 100, 30)
            
            if st.button("HARVEST", use_container_width=True):
                try:
                    manifest = harvest_from_pack(pack, st.session_state.level, st.session_state.subject, int(years_back), int(volume_max))
                    st.session_state.harvest_manifest = manifest
                    st.session_state.library = manifest["library"]
                    st.session_state.sealed = False
                    st.session_state.qi_pack = None
                    st.session_state.qc_pack = None
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0, "orphans": 0, "sanity_ok": False}
                    st.success(f"✅ HARVEST: {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                except Exception as e:
                    st.error(f"❌ HARVEST: {e}")
        else:
            st.error("Aucune source harvest configurée")
    
    with tab2:
        st.markdown("## RUN")
        lib2 = st.session_state.library
        corr_ok2 = sum(1 for x in lib2 if x.get("corrige?"))
        
        if not lib2:
            st.warning("Bibliothèque vide - faites un HARVEST")
        elif corr_ok2 <= 0:
            st.error("Aucun corrigé")
        else:
            max_vol = max(1, min(100, corr_ok2))
            c1, c2 = st.columns(2)
            phase_a = c1.slider("Phase A", 1, max_vol, min(5, max_vol))
            phase_b = c2.slider("Phase B", 1, max_vol, min(10, max_vol))
            
            if st.button("🚀 LANCER TEST SATURATION", use_container_width=True, type="primary"):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_saturation_test(lib2, phase_a, phase_b, pack)
                    
                    st.session_state.qi_pack = res["phaseB"]["qi_pack"]
                    st.session_state.qc_pack = res["phaseB"]["qc_pack"]
                    st.session_state.selection_report = res["selection_report"]
                    st.session_state.last_run_audit = res["audit"]
                    st.session_state.sealed = res["audit"]["sealed"]
                    
                    st.session_state.run_stats = {
                        "qi": res["audit"]["phaseB"]["qi_total"],
                        "rqi": res["audit"]["phaseB"]["rqi_total"],
                        "qc": res["audit"]["phaseB"]["qc_total"],
                        "qi_posable": res["audit"]["phaseB"]["qi_posable"],
                        "orphans": res["audit"]["phaseB"]["qi_orphans"],
                        "sanity_ok": res["audit"]["phaseB"]["sanity_ok"],
                    }
                    
                    status = "✅ SEALED=YES" if res["audit"]["sealed"] else "⚠️ SEALED=NO"
                    st.info(f"{status} | sat={res['audit']['saturation_ok']} | orphans={res['audit']['phaseB']['qi_orphans']} | QC={res['audit']['phaseB']['qc_total']}")
                    
                    if not res["audit"]["saturation_ok"]:
                        st.warning(f"Nouvelles QC: {res['audit']['setB_minus_setA_size']}")
                        st.json(res["audit"]["setB_minus_setA"][:10])
                
                except Exception as e:
                    st.error(f"❌ RUN: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        
        st.markdown("### Logs")
        st.text_area("", logs_text(), height=250)
    
    with tab3:
        st.markdown("## Exports")
        metric_row(len(st.session_state.library), sum(1 for x in st.session_state.library if x.get("corrige?")),
                   st.session_state.run_stats.get("qi", 0), st.session_state.run_stats.get("qi_posable", 0),
                   st.session_state.run_stats.get("qc", 0), st.session_state.sealed)
        
        st.download_button("logs.txt", logs_text(), "logs.txt")
        
        if st.session_state.qi_pack:
            st.download_button("qi_pack.json", json.dumps(st.session_state.qi_pack, ensure_ascii=False, indent=2), "qi_pack.json")
        if st.session_state.qc_pack:
            st.download_button("qc_pack.json", json.dumps(st.session_state.qc_pack, ensure_ascii=False, indent=2), "qc_pack.json")
        if st.session_state.selection_report:
            st.download_button("selection_report.json", json.dumps(st.session_state.selection_report, ensure_ascii=False, indent=2), "selection_report.json")
        if st.session_state.last_run_audit:
            st.download_button("audit.json", json.dumps(st.session_state.last_run_audit, ensure_ascii=False, indent=2), "audit.json")
        
        st.markdown("---")
        st.markdown("## Explorateur QC → Qi")
        
        if st.session_state.qc_pack and st.session_state.qi_pack:
            ch_opts = ["ALL"] + sorted({qc["chapter_code"] for qc in st.session_state.qc_pack})
            sel_ch = st.selectbox("Chapitre", ch_opts)
            
            qcs = st.session_state.qc_pack if sel_ch == "ALL" else [q for q in st.session_state.qc_pack if q["chapter_code"] == sel_ch]
            
            if qcs:
                qcs = sorted(qcs, key=lambda x: (-x.get("cluster_size", 0), x.get("qc_id", "")))
                qc_labels = [f"{q['qc_id']} | {q['chapter_code']} | {q.get('primary_op')} | n={q['cluster_size']} | Ψ={round(q.get('Psi_q', 0), 3)}" for q in qcs]
                sel_idx = st.selectbox("QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]
                
                st.markdown(f"### {qc['qc']}")
                st.write(f"**State:** {qc.get('qc_state')} | **F2:** {qc.get('_f2_score', 'N/A')}")
                
                qi_by_id = {q["qi_id"]: q for q in st.session_state.qi_pack}
                for qi_id in qc.get("qi_ids", [])[:15]:
                    q = qi_by_id.get(qi_id)
                    if q:
                        with st.expander(f"{qi_id} | RQi={'✅' if q.get('has_rqi') else '❌'}"):
                            st.text(q["qi"][:500])
                            if q.get("rqi"):
                                st.markdown("**RQi:**")
                                st.text(q["rqi"][:500])
        
        st.markdown("### Audit")
        if st.session_state.last_run_audit:
            st.json(st.session_state.last_run_audit)


if __name__ == "__main__":
    main()
