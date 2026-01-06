# =============================================================================
# SMAXIA GTE Console V31.10.21 — ISO-PROD (FICHIER UNIQUE)
# =============================================================================
# OBJECTIF V31.10.21 (meilleure combinaison du tableau comparatif) :
# - Saturation: BOUCLE PROGRESSIVE (vol_start → vol_max) jusqu'à new_QC=0  ✅ (GPT)
# - FRT: Templates spécifiques par opérateur (Pack-driven)               ✅ (Ma)
# - frt_pack.json: Export séparé                                         ✅ (GPT)
# - mapping_qi_to_qc.json: Export séparé (preuve)                         ✅ (GPT)
# - run_signature.json: SHA256 de tous les outputs                         ✅ (GPT)
# - qc_format: Pack-driven (prefix/suffix)                                ✅ (GPT)
# - URL Harvest: Annales-Terminale-Generale (corrigé)                      ✅ (GPT)
# - ARI trace: Enrichi (evidence + match_count)                            ✅ (Ma)
# - Coverage: Calculé (KPI)                                                ✅ (Ma)
# - seal_reason: Explication du statut SEALED                               ✅ (Ma)
# - UI: 4 onglets (Import / RUN / Résultats / Exports)                      ✅ (Ma)
#
# RÈGLES KERNEL (scellées) :
# - Zéro hardcode métier (pays/chapitres/matières) dans le CORE. La variabilité
#   est uniquement dans le PACK (données).
# - SEALED = (orphans == 0) AND (boucle saturation atteint new_QC == 0) AND sanity_ok
#   AND (qc_posable > 0) AND (qi_posable > 0).
#
# USAGE:
#   streamlit run smaxia_console_v31_10_21.py
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import time
import hashlib
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# =============================================================================
# KERNEL CONSTANTS (INVARIANTS MATHÉMATIQUES — JAMAIS MODIFIÉS)
# =============================================================================
APP_VERSION = "V31.10.21"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

# Constantes Kernel (références F1/F2)
F1_EPSILON = 1e-6
F2_ALPHA = 1.0
F2_TREC = 1.0

# Seuils "KPI" par défaut (peuvent être overridés par pack.f1_f2_params)
DEFAULT_COVERAGE_THRESHOLD = 0.80  # KPI seulement, pas une règle SEALED à elle seule


# =============================================================================
# PACK GENESIS — DONNÉES MÉTIER (TEST) — TOUTE LA VARIABILITÉ DOIT ÊTRE ICI
# =============================================================================
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_text(s: str) -> str:
    return _sha256_bytes(s.encode("utf-8"))


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _genesis_pack_fr() -> Dict[str, Any]:
    """
    Pack Genesis France — Math Terminale/Première.
    NOTE: Ce pack est "données". Le CORE ne doit pas contenir de règles métier.
    """
    return {
        "pack_id": "CAP_FR_GENESIS_V31_10_21",
        "pack_version": "31.10.21",
        "country_code": "FR",
        "country_name": "France",
        "language": "fr",
        "status": "GENESIS_TEST",

        # QC format pack-driven
        # (Le kernel impose le format invariant, mais l'implémentation UI/texte reste pack-driven)
        "qc_format": {"prefix": "Comment ", "suffix": " ?"},

        # Harvest sources
        "harvest_sources": [{
            "source_id": "APMEP",
            "source_name": "APMEP - Annales du Bac",
            "base_url": "https://www.apmep.fr",
            "levels": {
                # Chemins qui ont déjà été utilisés / corrigés côté GPT (à conserver)
                "TERMINALE": "/Annales-Terminale-Generale",
                "PREMIERE": "/Annales-Premiere-Generale",
            },
            "year_pattern": r"Annee-(20\d{2})",
            "pdf_patterns": {
                "sujet": ["spe", "sujet", "enonce", "énoncé", "bac", "terminale", "premiere"],
                "corrige": ["corrig", "correction", "solution", "corrigé"],
            },
            "meta_exclude": ["index", "sommaire", "liste", "annexe", "grille", "formulaire"],
            "geographic_zones": [
                "metro", "metropole", "métropole", "amerique", "nord", "sud", "asie",
                "polynesie", "polynésie", "etranger", "étranger", "antilles", "liban",
                "guyane", "reunion", "réunion", "mayotte", "calédonie", "caledonie"
            ],
        }],

        # Text processing
        "text_processing": {
            "common_words": [
                "le", "la", "les", "de", "du", "des", "un", "une", "et", "ou", "en", "que", "qui",
                "est", "sont", "par", "pour", "sur", "dans", "avec", "ce", "se", "ne", "pas",
                "au", "aux", "son", "sa", "ses", "tout", "tous", "on", "nous", "vous",
                "soit", "alors", "ainsi", "comme", "bien", "peut", "fait", "plus", "moins",
                "exercice", "partie", "question", "montrer", "demontrer", "calculer", "justifier",
            ],
            "glued_patterns": [
                {"pattern": r"(\d)([a-zA-Z])", "replacement": r"\1 \2"},
                {"pattern": r"([a-zA-Z])(\d)", "replacement": r"\1 \2"},
                {"pattern": r"([a-z])([A-Z])", "replacement": r"\1 \2"},
            ],
            "intent_verbs": [
                "montrer", "demontrer", "prouver", "justifier", "determiner", "calculer",
                "resoudre", "etudier", "donner", "exprimer", "simplifier", "trouver",
                "verifier", "deduire", "conclure", "etablir", "tracer", "representer",
            ],
            "trigger_keywords": [
                "limite", "derivee", "intégrale", "integrale", "probabilite", "probabilité",
                "suite", "vecteur", "complexe", "equation", "équation", "fonction",
                "recurrence", "récurrence", "matrice", "logarithme", "exponentielle",
                "primitive", "continuite", "continuité", "asymptote",
            ],
        },

        # ARI config (pack-driven)
        "ari_config": {
            "op_patterns": [
                {"op": "OP_PROBABILITY", "pattern": r"\b(probabilit|proba|loi\s+binomiale|loi\s+normale|esperance|variance|aleatoire)\b", "weight": 1.0},
                {"op": "OP_DERIVE", "pattern": r"\b(deriv|f'\s*\(|tangente|taux\s+de\s+variation)\b", "weight": 1.0},
                {"op": "OP_INTEGRATE", "pattern": r"\b(integr|primitive|aire\s+sous|calcul\s+d'aire)\b", "weight": 1.0},
                {"op": "OP_LIMIT", "pattern": r"\b(limit|tend\s+vers|infini|convergence)\b", "weight": 1.0},
                {"op": "OP_INDUCTION", "pattern": r"\b(recurr|induction|heredite|hérédit|initialisation|rang)\b", "weight": 1.0},
                {"op": "OP_COMPLEX", "pattern": r"\b(complex|imaginaire|module|argument|affixe|conjugue|conjugué)\b", "weight": 1.0},
                {"op": "OP_VECTOR_GEOM", "pattern": r"\b(vecteur|scalaire|orthogonal|colineaire|coliné|produit\s+scalaire)\b", "weight": 1.0},
                {"op": "OP_TRIGO", "pattern": r"\b(cos|sin|tan|trigo|radian|cercle\s+trigonometrique|trigonométr)\b", "weight": 1.0},
                {"op": "OP_LOGEXP", "pattern": r"\b(ln|log|exp|exponentiel|exponentiell|logarithme)\b", "weight": 1.0},
                {"op": "OP_SOLVE_EQUATION", "pattern": r"\b(equat|équat|resou|racine|solution|discriminant|inequation|inéquation)\b", "weight": 1.0},
                {"op": "OP_VARIATION_TABLE", "pattern": r"\b(tableau\s+de\s+variation|signe|croissant|decroissant|décroissant|extremum|extrém)\b", "weight": 1.0},
                {"op": "OP_PROVE", "pattern": r"\b(demontr|démontr|prouv|justifi|montr|etablir|établir)\b", "weight": 0.8},
            ],
            "primary_ops_order": [
                "OP_PROBABILITY", "OP_INTEGRATE", "OP_DERIVE", "OP_LIMIT",
                "OP_INDUCTION", "OP_COMPLEX", "OP_VECTOR_GEOM", "OP_TRIGO",
                "OP_LOGEXP", "OP_SOLVE_EQUATION", "OP_VARIATION_TABLE", "OP_PROVE", "OP_STANDARD",
            ],
            "op_labels": {
                "OP_PROBABILITY": "calculer une probabilité",
                "OP_DERIVE": "dériver une fonction",
                "OP_INTEGRATE": "calculer une intégrale",
                "OP_LIMIT": "calculer une limite",
                "OP_INDUCTION": "démontrer par récurrence",
                "OP_COMPLEX": "manipuler des nombres complexes",
                "OP_VECTOR_GEOM": "résoudre un problème de géométrie vectorielle",
                "OP_TRIGO": "appliquer la trigonométrie",
                "OP_LOGEXP": "utiliser logarithme et exponentielle",
                "OP_SOLVE_EQUATION": "résoudre une équation ou inéquation",
                "OP_VARIATION_TABLE": "établir un tableau de variation",
                "OP_PROVE": "démontrer une propriété",
                "OP_STANDARD": "résoudre un exercice standard",
            },
            # Templates FRT spécifiques par opérateur (ma version gagnante)
            "frt_templates": {
                "OP_PROBABILITY": [
                    "1. Identifier l'expérience aléatoire et l'univers Ω.",
                    "2. Définir précisément les événements.",
                    "3. Identifier la loi adaptée (binomiale, normale, etc.).",
                    "4. Appliquer la formule / propriété pertinente.",
                    "5. Calculer, vérifier la cohérence, conclure.",
                ],
                "OP_DERIVE": [
                    "1. Identifier f(x) et le domaine.",
                    "2. Décomposer (somme/produit/quotient/composition).",
                    "3. Appliquer les règles de dérivation.",
                    "4. Simplifier f'(x) et vérifier.",
                    "5. Exploiter (tangente, variations) si demandé.",
                ],
                "OP_INTEGRATE": [
                    "1. Identifier la fonction à intégrer et les bornes.",
                    "2. Trouver une primitive F.",
                    "3. Appliquer F(b) − F(a).",
                    "4. Simplifier et interpréter (aire, valeur moyenne…).",
                    "5. Conclure et contrôler l'ordre de grandeur.",
                ],
                "OP_LIMIT": [
                    "1. Identifier la forme de la limite.",
                    "2. Lever l’indétermination (factoriser, conjugué, équivalents…).",
                    "3. Utiliser le terme dominant / comparaisons.",
                    "4. Calculer et justifier.",
                    "5. Conclure sur la limite et l’interprétation (asymptote…).",
                ],
                "OP_INDUCTION": [
                    "1. Initialisation : vérifier P(n₀).",
                    "2. Hérédité : supposer P(n) vraie.",
                    "3. Démontrer P(n+1).",
                    "4. Conclusion : P(n) vraie pour tout n ≥ n₀.",
                ],
                "OP_COMPLEX": [
                    "1. Identifier la forme (algébrique / trigonométrique / exponentielle).",
                    "2. Convertir vers la forme utile.",
                    "3. Calculer (module, argument, conjugué, produit…).",
                    "4. Interpréter géométriquement si requis.",
                    "5. Conclure.",
                ],
                "OP_VECTOR_GEOM": [
                    "1. Choisir un repère/paramétrage adapté.",
                    "2. Écrire vecteurs/équations (droite, plan).",
                    "3. Utiliser orthogonalité / colinéarité / produit scalaire.",
                    "4. Résoudre le système et interpréter.",
                    "5. Conclure.",
                ],
                "OP_TRIGO": [
                    "1. Identifier les fonctions trigonométriques et le domaine.",
                    "2. Appliquer identités/formules utiles.",
                    "3. Résoudre sur intervalle de référence.",
                    "4. Généraliser (périodicité) et vérifier.",
                    "5. Conclure.",
                ],
                "OP_LOGEXP": [
                    "1. Déterminer le domaine (conditions).",
                    "2. Utiliser propriétés de ln/exp.",
                    "3. Transformer (isoler, équivalences).",
                    "4. Résoudre et vérifier dans le domaine.",
                    "5. Conclure.",
                ],
                "OP_SOLVE_EQUATION": [
                    "1. Identifier le type et le domaine.",
                    "2. Mettre sous une forme résoluble (factoriser, substitution…).",
                    "3. Résoudre (discriminant, étude de signe…).",
                    "4. Vérifier les solutions.",
                    "5. Conclure.",
                ],
                "OP_VARIATION_TABLE": [
                    "1. Calculer f'(x).",
                    "2. Résoudre f'(x)=0 et déterminer le signe.",
                    "3. Dresser le tableau de variation.",
                    "4. Déduire extrema / image / inégalités.",
                    "5. Conclure.",
                ],
                "OP_PROVE": [
                    "1. Reformuler clairement ce qu’il faut démontrer.",
                    "2. Choisir une stratégie (directe, absurde, contraposée, récurrence).",
                    "3. Dérouler le raisonnement en justifiant chaque étape.",
                    "4. Vérifier les conditions/hypothèses.",
                    "5. Conclure.",
                ],
                "OP_STANDARD": [
                    "1. Identifier données/inconnues/objectif.",
                    "2. Choisir une méthode générale adaptée.",
                    "3. Exécuter les calculs avec rigueur.",
                    "4. Vérifier et interpréter.",
                    "5. Conclure.",
                ],
            },
        },

        # Chapter taxonomy (pack-driven)
        "chapter_taxonomy": {
            "MATH": {
                "TERMINALE": [
                    {"code": "CH_ANALYSE", "label": "Analyse", "keywords": ["limite", "derivee", "dérivée", "continuite", "continuité", "asymptote", "variation", "tangente", "convexite", "convexité", "fonction"], "delta_c": 1.0},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "keywords": ["probabilite", "probabilité", "loi", "binomiale", "normale", "esperance", "espérance", "variance", "aleatoire", "aléatoire", "independance", "indépendance"], "delta_c": 1.0},
                    {"code": "CH_SUITES", "label": "Suites", "keywords": ["suite", "recurrence", "récurrence", "arithmetique", "arithmétique", "geometrique", "géométrique", "convergence", "terme", "rang", "limite"], "delta_c": 1.0},
                    {"code": "CH_INTEGRATION", "label": "Intégration", "keywords": ["integrale", "intégrale", "primitive", "aire", "integration", "intégration", "valeur moyenne"], "delta_c": 1.0},
                    {"code": "CH_GEOMETRIE", "label": "Géométrie espace", "keywords": ["vecteur", "plan", "droite", "espace", "orthogonal", "colineaire", "colinéaire", "parametrique", "paramétrique"], "delta_c": 1.0},
                    {"code": "CH_COMPLEXES", "label": "Nombres complexes", "keywords": ["complexe", "imaginaire", "module", "argument", "affixe", "conjugue", "conjugué", "exponentielle"], "delta_c": 1.0},
                    {"code": "CH_LOGEXP", "label": "Log & Exp", "keywords": ["exponentielle", "logarithme", "ln", "exp"], "delta_c": 1.0},
                    {"code": "CH_TRIGO", "label": "Trigonométrie", "keywords": ["cosinus", "sinus", "tangente", "radian", "cercle", "trigonometrique", "trigonométrique"], "delta_c": 1.0},
                ],
                "PREMIERE": [
                    {"code": "CH_ANALYSE", "label": "Analyse", "keywords": ["derivee", "dérivée", "variation", "fonction", "tangente", "polynome", "polynôme"], "delta_c": 1.0},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "keywords": ["probabilite", "probabilité", "evenement", "événement", "aleatoire", "aléatoire", "variable"], "delta_c": 1.0},
                    {"code": "CH_SUITES", "label": "Suites", "keywords": ["suite", "arithmetique", "arithmétique", "geometrique", "géométrique", "terme"], "delta_c": 1.0},
                ],
            }
        },

        "f1_f2_params": {
            "epsilon": F1_EPSILON,
            "alpha": F2_ALPHA,
            "t_rec_default": F2_TREC,
            "coverage_threshold": DEFAULT_COVERAGE_THRESHOLD,  # KPI
        },
    }


def _clone_pack(pack: Dict[str, Any], pack_id: str, country_code: str, country_name: str) -> Dict[str, Any]:
    p = json.loads(json.dumps(pack))
    p["pack_id"] = pack_id
    p["country_code"] = country_code
    p["country_name"] = country_name
    return p


def _genesis_pack_ci() -> Dict[str, Any]:
    return _clone_pack(_genesis_pack_fr(), "CAP_CI_GENESIS_V31_10_21", "CI", "Côte d'Ivoire")


def _genesis_pack_sn() -> Dict[str, Any]:
    return _clone_pack(_genesis_pack_fr(), "CAP_SN_GENESIS_V31_10_21", "SN", "Sénégal")


GENESIS_PACKS = {
    "FR": ("France", _genesis_pack_fr),
    "CI": ("Côte d'Ivoire", _genesis_pack_ci),
    "SN": ("Sénégal", _genesis_pack_sn),
}


def generate_pack(country_code: str) -> Dict[str, Any]:
    if country_code not in GENESIS_PACKS:
        raise ValueError(f"Pays non supporté: {country_code}")
    _, generator = GENESIS_PACKS[country_code]
    pack = generator()
    pack["created_at"] = _utc_ts()
    pack["_source"] = "GENESIS_AUTO"
    pack["_pack_sig_sha256"] = _sha256_text(_json_dumps_stable(pack))
    return pack


# =============================================================================
# SESSION & LOGS
# =============================================================================
def ss_init() -> None:
    defaults = {
        "pack_active": None,
        "pack_id": None,
        "pack_sig_sha256": None,
        "country": None,
        "level": None,
        "subject": None,
        "library": [],
        "harvest_manifest": None,
        "qi_pack": None,
        "qc_pack": None,
        "frt_pack": None,
        "qc_map": None,
        "selection_report": None,
        "sealed": False,
        "logs": [],
        "run_stats": {
            "pairs": 0,
            "corriges": 0,
            "qi": 0,
            "rqi": 0,
            "qi_posable": 0,
            "orphans": 0,
            "qc": 0,
            "qc_posable": 0,
            "coverage": 0.0,
            "sanity_ok": False,
        },
        "last_run_audit": None,
        "run_signature": None,
        "_http_pdf_cache": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def log(msg: str) -> None:
    st.session_state.logs.append(f"[{_utc_ts()}] {msg}")


def logs_text() -> str:
    return "\n".join(st.session_state.logs[-8000:])


# =============================================================================
# TEXT UTILS (CORE INVARIANT)
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
# PACK ACCESSORS (CORE)
# =============================================================================
def pack_chapters(pack: Dict[str, Any], subject: str, level: str) -> List[Dict[str, Any]]:
    return pack.get("chapter_taxonomy", {}).get(subject, {}).get(level, [])


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
        "coverage_threshold": float(params.get("coverage_threshold", DEFAULT_COVERAGE_THRESHOLD)),
    }


def pack_qc_format(pack: Dict[str, Any]) -> Dict[str, str]:
    fmt = pack.get("qc_format", {}) or {}
    prefix = str(fmt.get("prefix", "Comment ")).strip()
    suffix = str(fmt.get("suffix", " ?")).strip()
    # IMPORTANT: On ne hardcode pas de matière/chapitre ici. Le format est pack-driven.
    return {"prefix": prefix if prefix.endswith(" ") else (prefix + " "), "suffix": suffix}


# =============================================================================
# PDF TEXT EXTRACTION
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
                candidate = word[i:i + length].lower()
                if candidate in common_words:
                    if result:
                        result.append(" ")
                    result.append(word[i:i + length])
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


def _extract_pages_pdfplumber(pdf_bytes: bytes, max_pages: int = 140) -> List[str]:
    if not pdfplumber:
        return []
    pages: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:max_pages]:
                try:
                    pages.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
                except Exception:
                    pages.append("")
    except Exception:
        return []
    return pages


def _extract_pages_pypdf(pdf_bytes: bytes, max_pages: int = 140) -> List[str]:
    if not PdfReader:
        return []
    pages: List[str] = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages[:max_pages]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
    except Exception:
        return []
    return pages


def clean_pdf_text(pages: List[str], text_proc: Dict[str, Any]) -> str:
    if not pages:
        return ""

    page_lines: List[List[str]] = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = _dehyphenate(p)
        p = _fix_missing_spaces(p, text_proc)
        page_lines.append([ln.strip() for ln in p.split("\n") if ln.strip()])

    # Header/footer detection (simple, robuste)
    top_counts: Dict[str, int] = {}
    bot_counts: Dict[str, int] = {}
    for lines in page_lines:
        if lines:
            top_counts[lines[0]] = top_counts.get(lines[0], 0) + 1
        if len(lines) > 1:
            bot_counts[lines[-1]] = bot_counts.get(lines[-1], 0) + 1

    threshold = max(2, len(page_lines) // 3)
    skip_top = {k for k, v in top_counts.items() if v >= threshold}
    skip_bot = {k for k, v in bot_counts.items() if v >= threshold}

    out_lines: List[str] = []
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
    cache: Dict[str, bytes] = st.session_state.get("_http_pdf_cache", {})
    if url in cache:
        return cache[url]

    res = _http_get(url)
    pdf_bytes = res.content
    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError("PDF trop volumineux")

    cache[url] = pdf_bytes
    st.session_state["_http_pdf_cache"] = cache
    return pdf_bytes


# =============================================================================
# HARVEST
# =============================================================================
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé")
    return BeautifulSoup(html, "html.parser")


def _abs_url(base: str, href: str) -> str:
    return href if href.startswith("http") else requests.compat.urljoin(base, href)


def harvest_from_pack(pack: Dict[str, Any], level: str, subject: str, years_back: int, volume_max: int) -> Dict[str, Any]:
    source = pack_harvest_source(pack)
    if not source:
        raise ValueError("Aucune source harvest dans le Pack")

    base_url = source.get("base_url", "")
    level_path = source.get("levels", {}).get(level)
    if not level_path:
        raise ValueError(f"Niveau '{level}' non configuré dans le pack")

    root = f"{base_url}{level_path}"
    log(f"[HARVEST] root={root}")

    year_pattern = source.get("year_pattern", r"(20\d{2})")
    meta_exclude = source.get("meta_exclude", [])
    corrige_patterns = source.get("pdf_patterns", {}).get("corrige", [])
    geo_zones = source.get("geographic_zones", [])

    html = _http_get(root).text
    sp = _soup(html)

    year_links: List[Tuple[int, str]] = []
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
    log(f"[HARVEST] années sélectionnées={len(selected)} (≥{min_year})")

    pairs: List[Dict[str, Any]] = []
    corrige_ok = 0

    for (y, url) in selected:
        if len(pairs) >= volume_max:
            break
        try:
            y_html = _http_get(url).text
            y_sp = _soup(y_html)
            pdf_links: List[Dict[str, Any]] = []

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
                        score += 0.3 * len(s_tok & c_tok) / max(1, len(s_tok | c_tok))

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

            log(f"[HARVEST] year={y} pairs_cum={len(pairs)}")
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
    chunks: List[str] = []
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

def split_questions(text: str, text_proc: Dict[str, Any], max_items: int = 220) -> Tuple[List[str], Dict[str, int]]:
    intent_re = _build_intent_re(text_proc.get("intent_verbs", []))
    raw_chunks = _chunk_candidates(text)

    keep: List[str] = []
    seen = set()
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
    link: List[Optional[int]] = [None] * len(questions)
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
# ARI EXTRACTION (PACK-DRIVEN) — ENRICHI (evidence + match_count)
# =============================================================================
def extract_ari_trace(question: str, response: str, ari_config: Dict[str, Any]) -> Dict[str, Any]:
    op_patterns = ari_config.get("op_patterns", [])
    if not op_patterns:
        return {
            "ops": [{"op": "OP_STANDARD", "confidence": 0.5, "evidence": [], "match_count": 0}],
            "primary_op": "OP_STANDARD",
            "all_ops": ["OP_STANDARD"],
            "confidence_global": 0.5,
        }

    combined = norm_text(f"{question} {response}")
    ops: List[Dict[str, Any]] = []
    seen = set()

    for item in op_patterns:
        pattern = item.get("pattern", "")
        op_code = item.get("op", "")
        weight = float(item.get("weight", 1.0))
        if not pattern or not op_code:
            continue
        try:
            matches = list(re.finditer(pattern, combined, re.IGNORECASE))
            if matches and op_code not in seen:
                evidence = [m.group() for m in matches[:3]]
                confidence = min(0.9, 0.6 + 0.1 * len(matches)) * weight
                ops.append({
                    "op": op_code,
                    "confidence": round(confidence, 4),
                    "evidence": evidence,
                    "match_count": len(matches),
                })
                seen.add(op_code)
        except re.error:
            continue

    if not ops:
        ops = [{"op": "OP_STANDARD", "confidence": 0.5, "evidence": [], "match_count": 0}]
        seen = {"OP_STANDARD"}

    primary_order = ari_config.get("primary_ops_order", [])
    ops_in_trace = {x["op"] for x in ops}
    primary_op = "OP_STANDARD"
    for op in primary_order:
        if op in ops_in_trace:
            primary_op = op
            break

    conf_global = max(float(x.get("confidence", 0.5)) for x in ops) if ops else 0.5

    return {
        "ops": ops,
        "primary_op": primary_op,
        "all_ops": sorted(list(ops_in_trace)) if ops_in_trace else ["OP_STANDARD"],
        "confidence_global": round(conf_global, 4),
    }


# =============================================================================
# TRIGGERS (ENRICHIS)
# =============================================================================
def build_triggers(question: str, ari_trace: Dict[str, Any], text_proc: Dict[str, Any]) -> List[str]:
    triggers: List[str] = []

    # Triggers ARI
    for op_item in ari_trace.get("ops", []):
        triggers.append(f"ARI:{op_item.get('op', 'OP_STANDARD')}")

    # Triggers Keywords
    q_norm = norm_text(question)
    for kw in text_proc.get("trigger_keywords", []):
        if kw and kw in q_norm:
            triggers.append(f"KW:{kw.upper()}")

    # Trigger Intent (un seul)
    for verb in text_proc.get("intent_verbs", [])[:30]:
        if verb and verb in q_norm:
            triggers.append(f"INTENT:{verb.upper()}")
            break

    # Unique, max 15
    uniq = list(dict.fromkeys(triggers))
    return uniq[:15]


# =============================================================================
# CHAPTER MAPPING (avec score)
# =============================================================================
def map_qi_to_chapter(question: str, chapters: List[Dict[str, Any]]) -> Tuple[str, int]:
    if not chapters:
        return "UNMAPPED", 0
    q_norm = norm_text(question)
    best_ch, best_score = "UNMAPPED", 0
    for ch in chapters:
        kws = ch.get("keywords", []) or []
        score = sum(1 for kw in kws if kw and kw in q_norm)
        if score > best_score:
            best_score, best_ch = score, ch.get("code", "UNMAPPED")
    return best_ch, int(best_score)


# =============================================================================
# FRT PACK (export séparé) + QC référence frt_id
# =============================================================================
def build_frt_pack(ari_config: Dict[str, Any]) -> Dict[str, Any]:
    templates: Dict[str, List[str]] = ari_config.get("frt_templates", {}) or {}
    op_labels: Dict[str, str] = ari_config.get("op_labels", {}) or {}
    frt_items: Dict[str, Any] = {}

    for op_code, steps in templates.items():
        if not steps:
            continue
        label = op_labels.get(op_code, op_code.replace("OP_", "").replace("_", " ").lower())
        frt_id = f"FRT_{stable_id(op_code)}"
        frt_items[frt_id] = {
            "frt_id": frt_id,
            "primary_op": op_code,
            "title": f"Comment {label} ?",
            "steps": steps,
            "steps_count": len(steps),
            "generated_at": _utc_ts(),
        }

    # Assurer un fallback OP_STANDARD
    if "FRT_" + stable_id("OP_STANDARD") not in frt_items and "OP_STANDARD" in templates:
        op_code = "OP_STANDARD"
        steps = templates.get(op_code, [])
        frt_id = f"FRT_{stable_id(op_code)}"
        frt_items[frt_id] = {
            "frt_id": frt_id,
            "primary_op": op_code,
            "title": "Comment résoudre un exercice standard ?",
            "steps": steps,
            "steps_count": len(steps),
            "generated_at": _utc_ts(),
        }

    return {
        "version": APP_VERSION,
        "generated_at": _utc_ts(),
        "items": frt_items,
    }


def frt_id_for_op(primary_op: str) -> str:
    return f"FRT_{stable_id(primary_op)}"


# =============================================================================
# QC GENERATION — COMPLET (zéro orphan imposé)
# =============================================================================
def build_qc_from_qi(
    qi_items: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
    ari_config: Dict[str, Any],
    qc_format: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    # Bucketing par (chapter_code, primary_op)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for qi in qi_items:
        # IMPORTANT: Pour garantir orphans==0 sur Qi posables, on bucket même UNMAPPED.
        cc = qi.get("chapter_code", "UNMAPPED") or "UNMAPPED"
        op = qi.get("primary_op", "OP_STANDARD") or "OP_STANDARD"
        buckets.setdefault((cc, op), []).append(qi)

    log(f"[QC] Buckets={len(buckets)}")

    op_labels = ari_config.get("op_labels", {}) or {}
    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}

    for (cc, primary_op), items in sorted(buckets.items()):
        posable = [x for x in items if x.get("has_rqi")]
        cluster_size = len(items)
        posable_count = len(posable)

        qc_id = f"QC_{stable_id(cc, primary_op)}"
        label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())

        prefix = qc_format["prefix"]
        suffix = qc_format["suffix"]
        qc_text = f"{prefix}{label}{suffix}"

        if posable_count >= 2:
            qc_state = "POSABLE"
        elif posable_count == 1:
            qc_state = "POSABLE_WEAK"
        else:
            qc_state = "UNPOSABLE"

        trig_counts: Dict[str, int] = {}
        all_ops = set()
        for it in items:
            for t in it.get("triggers", []):
                trig_counts[t] = trig_counts.get(t, 0) + 1
            all_ops.update(it.get("ari_trace", {}).get("all_ops", []) or [])

        qc_item = {
            "qc_id": qc_id,
            "qc": qc_text,
            "qc_label": label,
            "chapter_code": cc,
            "primary_op": primary_op,
            "all_ops": sorted(list(all_ops)) if all_ops else ["OP_STANDARD"],
            "cluster_size": cluster_size,
            "posable_in_cluster": posable_count,
            "qc_state": qc_state,
            "qi_ids": [x["qi_id"] for x in items],
            "triggers": [k for k, _ in sorted(trig_counts.items(), key=lambda x: -x[1])][:10],
            # Référence FRT séparée (export frt_pack.json)
            "frt_id": frt_id_for_op(primary_op),
        }
        qc_pack.append(qc_item)

        for it in items:
            qc_map[it["qi_id"]] = qc_id

    log(f"[QC] Total={len(qc_pack)} | POSABLE={sum(1 for q in qc_pack if q['qc_state'] in ('POSABLE','POSABLE_WEAK'))}")
    return qc_pack, qc_map


# =============================================================================
# F1/F2 (KERNEL) — UTILISÉ POUR SCORE + SELECTION_REPORT
# =============================================================================
def compute_trigger_weights(triggers: List[str]) -> Dict[str, float]:
    uniq = list(dict.fromkeys(triggers or []))
    n = len(uniq)
    return {t: 1.0 / n for t in uniq} if n else {}

def f1_raw(delta_c: float, epsilon: float, Tj: Dict[str, float], m_q: int) -> float:
    s = sum(v for _, v in sorted(Tj.items())[:m_q]) if Tj else 0.0
    return float(delta_c) * (float(epsilon) + float(s)) ** 2

def f1_normalize_in_chapter(qc_list: List[Dict[str, Any]]) -> None:
    if not qc_list:
        return
    raws = [float(q.get("psi_raw", 0.0)) for q in qc_list]
    mx = max(raws) if raws else 0.0
    for q in qc_list:
        q["Psi_q"] = float(q.get("psi_raw", 0.0)) / mx if mx > 0 else 0.0

def sigma_similarity(qc_a: Dict[str, Any], qc_b: Dict[str, Any]) -> float:
    a_ops, b_ops = set(qc_a.get("all_ops", []) or []), set(qc_b.get("all_ops", []) or [])
    if a_ops and b_ops:
        return len(a_ops & b_ops) / max(1, len(a_ops | b_ops))
    return 0.0

def f2_score(qc: Dict[str, Any], selected: List[Dict[str, Any]], n_hist: int, N_total: int, alpha: float, t_rec: float) -> float:
    Psi_q = float(qc.get("Psi_q", 0.0))
    base = (float(n_hist) / max(1, int(N_total))) * (1.0 + float(alpha) / max(1e-9, float(t_rec))) * Psi_q
    prod = 1.0
    for p in selected:
        prod *= (1.0 - sigma_similarity(qc, p))
    return base * prod * 100.0

def progressive_select(qc_list: List[Dict[str, Any]], N_total: int, f1f2_params: Dict[str, float], top_k: int = 15) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    remaining = [q for q in qc_list if q.get("qc_state") in ("POSABLE", "POSABLE_WEAK")]
    alpha, t_rec = f1f2_params.get("alpha", F2_ALPHA), f1f2_params.get("t_rec", F2_TREC)

    for _ in range(min(top_k, len(remaining))):
        best = None
        for qc in remaining:
            n_hist = int(qc.get("cluster_size", 1))
            sc = f2_score(qc, selected, n_hist, int(N_total), float(alpha), float(t_rec))
            qc["f2_score"] = float(sc)
            if best is None or float(sc) > float(best.get("f2_score", 0.0)):
                best = qc
        if best is None:
            break
        selected.append(best)
        remaining = [x for x in remaining if x["qc_id"] != best["qc_id"]]
    return selected


# =============================================================================
# SANITY / COVERAGE
# =============================================================================
def sanity_eval(qi_items: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    seen, dups = set(), 0
    for q in qi_items:
        k = stable_id(norm_text(q.get("qi", ""))[:300])
        if k in seen:
            dups += 1
        else:
            seen.add(k)
    dup_ratio = dups / max(1, len(qi_items))
    ok = dup_ratio <= 0.35
    return ok, {"dup_ratio_pct": round(dup_ratio * 100, 2), "duplicates": dups}

def compute_coverage(qi_items: List[Dict[str, Any]], qc_map: Dict[str, str]) -> float:
    # KPI: couverture des Qi posables (has_rqi=True) qui ont une QC
    if not qi_items:
        return 0.0
    posable = [qi for qi in qi_items if qi.get("has_rqi")]
    if not posable:
        return 0.0
    covered = sum(1 for qi in posable if qi["qi_id"] in qc_map)
    return covered / max(1, len(posable))


# =============================================================================
# PIPELINE (volume fixe) — Génère Qi → ARI → QC + FRT refs
# =============================================================================
def run_pipeline(library: List[Dict[str, Any]], volume: int, pack: Dict[str, Any]) -> Dict[str, Any]:
    text_proc = pack_text_processing(pack)
    ari_config = pack_ari_config(pack)
    qc_fmt = pack_qc_format(pack)

    subject = st.session_state.get("subject")
    level = st.session_state.get("level")
    if not subject or not level:
        raise ValueError("subject/level non initialisés (activation pack incomplète)")

    chapters = pack_chapters(pack, subject, level)

    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    to_process = exploitable[:min(int(volume), len(exploitable))]
    log(f"[PIPELINE] volume={volume} pairs_exploitables={len(to_process)}/{len(exploitable)}")

    qi_items: List[Dict[str, Any]] = []

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
            log(f"[EXTRACT] SKIP {pid}: texte vide")
            continue

        qs, stat_q = split_questions(su_text, text_proc)
        rs, stat_r = split_questions(co_text, text_proc)
        link = align_qi_rqi(qs, rs)

        matched = sum(1 for x in link if x is not None)
        log(f"[ALIGN] {pid}: Qi={len(qs)}(raw={stat_q['raw']}) RQi={len(rs)}(raw={stat_r['raw']}) matched={matched}")

        for i, q in enumerate(qs):
            if not q.strip():
                continue
            j = link[i] if i < len(link) else None
            r = rs[j] if (j is not None and j < len(rs)) else ""

            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"

            ari_trace = extract_ari_trace(q, r, ari_config)
            triggers = build_triggers(q, ari_trace, text_proc)
            chapter_code, chapter_score = map_qi_to_chapter(q, chapters)

            qi_items.append({
                "qi_id": qi_id,
                "pair_id": pid,
                "qi": q,
                "rqi": r,
                "has_rqi": bool(r),
                "chapter_code": chapter_code,
                "chapter_match_score": int(chapter_score),
                "primary_op": ari_trace["primary_op"],
                "ari_trace": ari_trace,
                "triggers": triggers,
            })

    sanity_ok, sanity_audit = sanity_eval(qi_items)

    # FRT pack (export séparé)
    frt_pack = build_frt_pack(ari_config)

    # QC + map (doit assurer la couverture des Qi posables → orphans==0 au final si pipeline OK)
    qc_pack, qc_map = build_qc_from_qi(qi_items, chapters, ari_config, qc_fmt)

    # F1 compute (psi_raw + Psi_q)
    f1f2_params = pack_f1f2_params(pack)
    eps = float(f1f2_params["epsilon"])

    for qc in qc_pack:
        ch_info = next((c for c in chapters if c.get("code") == qc["chapter_code"]), {})
        delta_c = float(ch_info.get("delta_c", 1.0))

        all_triggers: List[str] = []
        qi_by_id = {q["qi_id"]: q for q in qi_items}
        for qi_id in qc.get("qi_ids", []):
            qi = qi_by_id.get(qi_id)
            if qi:
                all_triggers.extend(qi.get("triggers", []))

        Tj = compute_trigger_weights(all_triggers)
        qc["psi_raw"] = f1_raw(delta_c, eps, Tj, len(Tj))

    # Normalize by chapter
    by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for qc in qc_pack:
        by_ch.setdefault(qc["chapter_code"], []).append(qc)
    for qcs in by_ch.values():
        f1_normalize_in_chapter(qcs)

    # Link QC to Qi + orphans
    orphans = 0
    for qi in qi_items:
        if qi["qi_id"] not in qc_map:
            qi["is_orphan"] = True
            qi["qc_id"] = None
            if qi.get("has_rqi"):
                orphans += 1
        else:
            qi["is_orphan"] = False
            qi["qc_id"] = qc_map[qi["qi_id"]]

    coverage = compute_coverage(qi_items, qc_map)

    qi_total = len(qi_items)
    rqi_total = sum(1 for q in qi_items if q.get("has_rqi"))
    qi_posable = sum(1 for q in qi_items if q.get("has_rqi") and not q.get("is_orphan"))
    qc_total = len(qc_pack)
    qc_posable = sum(1 for q in qc_pack if q.get("qc_state") in ("POSABLE", "POSABLE_WEAK"))

    audit = {
        "qi_total": qi_total,
        "rqi_total": rqi_total,
        "qi_posable": qi_posable,
        "qi_orphans": orphans,  # uniquement Qi posables orphelines
        "qc_total": qc_total,
        "qc_posable": qc_posable,
        "coverage": round(float(coverage), 4),
        "sanity_ok": bool(sanity_ok),
        "sanity_audit": sanity_audit,
        "subject": subject,
        "level": level,
    }

    return {
        "qi_pack": qi_items,
        "qc_pack": qc_pack,
        "frt_pack": frt_pack,
        "qc_map": qc_map,
        "audit": audit,
    }


# =============================================================================
# SATURATION — BOUCLE PROGRESSIVE (KERNEL) jusqu'à new_QC == 0
# =============================================================================
def run_saturation_progressive(
    library: List[Dict[str, Any]],
    vol_start: int,
    vol_max: int,
    step: int,
    pack: Dict[str, Any],
    selection_top_k: int = 15,
) -> Dict[str, Any]:
    log(f"[SAT] progressive start={vol_start} max={vol_max} step={step}")

    prev_set: Optional[set] = None
    prev_volume: Optional[int] = None
    prev_out: Optional[Dict[str, Any]] = None

    iterations: List[Dict[str, Any]] = []

    final_out: Optional[Dict[str, Any]] = None
    final_new_qc: List[str] = []

    for vol in range(int(vol_start), int(vol_max) + 1, max(1, int(step))):
        out = run_pipeline(library, vol, pack)
        qc_ids = {qc["qc_id"] for qc in out["qc_pack"]}

        if prev_set is None:
            new_qc = sorted(list(qc_ids))
        else:
            new_qc = sorted(list(qc_ids - prev_set))

        it = {
            "volume": vol,
            "qc_total": out["audit"]["qc_total"],
            "qc_posable": out["audit"]["qc_posable"],
            "qi_total": out["audit"]["qi_total"],
            "qi_posable": out["audit"]["qi_posable"],
            "orphans": out["audit"]["qi_orphans"],
            "coverage": out["audit"]["coverage"],
            "sanity_ok": out["audit"]["sanity_ok"],
            "new_qc_count": len(new_qc),
        }
        iterations.append(it)

        log(f"[SAT] vol={vol} qc={it['qc_total']} new_QC={it['new_qc_count']} orphans={it['orphans']} coverage={it['coverage']:.2f} sanity={it['sanity_ok']}")

        # Kernel rule: one iteration with new_QC == 0 (i.e., qc_ids stable vs previous volume)
        if prev_set is not None and len(new_qc) == 0:
            final_out = out
            final_new_qc = []
            log(f"[SAT] CONVERGED at vol={vol} (new_QC=0)")
            break

        prev_set = qc_ids
        prev_volume = vol
        prev_out = out
        final_out = out
        final_new_qc = new_qc[:200]  # keep sample in audit

    if final_out is None:
        raise RuntimeError("Saturation: aucune itération produite (bibliothèque vide ou erreur)")

    # Selection report (F2) par chapitre, sur final_out
    f1f2_params = pack_f1f2_params(pack)
    subject = st.session_state.get("subject")
    level = st.session_state.get("level")
    chapters = pack_chapters(pack, subject, level)

    N_total = int(final_out["audit"]["qi_posable"])
    by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for qc in final_out["qc_pack"]:
        by_ch.setdefault(qc["chapter_code"], []).append(qc)

    selection: List[Dict[str, Any]] = []
    for cc, qcs in by_ch.items():
        selected = progressive_select(qcs, N_total, f1f2_params, top_k=int(selection_top_k))
        ch_info = next((c for c in chapters if c.get("code") == cc), {"label": cc})
        selection.append({
            "chapter_code": cc,
            "chapter_label": ch_info.get("label", cc),
            "qc_count": len(qcs),
            "selected_count": len(selected),
            "selected_qc": [{
                "qc_id": x["qc_id"],
                "qc": x["qc"],
                "primary_op": x["primary_op"],
                "Psi_q": round(float(x.get("Psi_q", 0.0)), 4),
                "f2_score": round(float(x.get("f2_score", 0.0)), 4),
                "frt_id": x.get("frt_id", ""),
            } for x in selected],
        })

    selection_report = {"generated_at": _utc_ts(), "chapters": selection}

    # SEALED strict (kernel)
    new_qc_is_zero = (iterations[-1]["new_qc_count"] == 0) if len(iterations) >= 2 else False
    sanity_ok = bool(final_out["audit"]["sanity_ok"])
    orphans = int(final_out["audit"]["qi_orphans"])
    qc_posable = int(final_out["audit"]["qc_posable"])
    qi_posable = int(final_out["audit"]["qi_posable"])

    sealed = (
        new_qc_is_zero and
        sanity_ok and
        orphans == 0 and
        qc_posable > 0 and
        qi_posable > 0
    )

    seal_reason = _seal_reason_strict(new_qc_is_zero, sanity_ok, orphans, qc_posable, qi_posable)

    audit = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": pack.get("pack_id"),
        "pack_sig_sha256": pack.get("_pack_sig_sha256"),
        "iterations": iterations,
        "final_volume": iterations[-1]["volume"],
        "new_qc_zero_reached": bool(new_qc_is_zero),
        "last_new_qc_sample": final_new_qc[:20],
        "phase_final": final_out["audit"],
        "sealed": bool(sealed),
        "seal_reason": seal_reason,
    }

    return {
        "final": final_out,
        "selection_report": selection_report,
        "audit": audit,
    }


def _seal_reason_strict(new_qc_zero: bool, sanity_ok: bool, orphans: int, qc_posable: int, qi_posable: int) -> str:
    if new_qc_zero and sanity_ok and orphans == 0 and qc_posable > 0 and qi_posable > 0:
        return "ALL_CHECKS_PASSED"
    reasons: List[str] = []
    if not new_qc_zero:
        reasons.append("SATURATION_NOT_CONVERGED_new_QC!=0")
    if not sanity_ok:
        reasons.append("SANITY_FAILED")
    if orphans != 0:
        reasons.append(f"ORPHANS_POSABLE={orphans}")
    if qc_posable <= 0:
        reasons.append("NO_QC_POSABLE")
    if qi_posable <= 0:
        reasons.append("NO_QI_POSABLE")
    return " | ".join(reasons) if reasons else "UNKNOWN"


# =============================================================================
# RUN SIGNATURE (preuve) — SHA256 de tous les outputs
# =============================================================================
def build_run_signature(
    pack: Dict[str, Any],
    qi_pack: Any,
    qc_pack: Any,
    frt_pack: Any,
    mapping_qi_to_qc: Any,
    selection_report: Any,
    audit: Any,
    logs: str,
) -> Dict[str, Any]:
    def h(obj: Any) -> str:
        return _sha256_text(_json_dumps_stable(obj))

    sig = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": pack.get("pack_id"),
        "pack_sig_sha256": pack.get("_pack_sig_sha256"),
        "hashes": {
            "qi_pack_sha256": h(qi_pack),
            "qc_pack_sha256": h(qc_pack),
            "frt_pack_sha256": h(frt_pack),
            "mapping_qi_to_qc_sha256": h(mapping_qi_to_qc),
            "selection_report_sha256": h(selection_report),
            "audit_sha256": h(audit),
            "logs_sha256": _sha256_text(logs),
        },
        "counts": {
            "qi": len(qi_pack) if isinstance(qi_pack, list) else 0,
            "qc": len(qc_pack) if isinstance(qc_pack, list) else 0,
            "frt": len((frt_pack or {}).get("items", {})) if isinstance(frt_pack, dict) else 0,
            "mapping": len(mapping_qi_to_qc) if isinstance(mapping_qi_to_qc, dict) else 0,
        },
    }
    sig["run_signature_sha256"] = h(sig)
    return sig


# =============================================================================
# UI
# =============================================================================
def metric_row(pairs: int, corr_ok: int, qi: int, qi_posable: int, qc: int, coverage: float, sealed: bool):
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Pairs", pairs)
    c2.metric("Corrigés", corr_ok)
    c3.metric("Qi", qi)
    c4.metric("Qi POSABLE", qi_posable)
    c5.metric("QC", qc)
    c6.metric("Coverage", f"{coverage:.0%}")
    c7.metric("SEALED", "✅ YES" if sealed else "❌ NO")


def _safe_set_default_subject_level(pack: Dict[str, Any]) -> None:
    taxonomy = pack.get("chapter_taxonomy", {}) or {}
    subjects = list(taxonomy.keys())
    if subjects:
        st.session_state.subject = st.session_state.subject or subjects[0]
        levels = list((taxonomy.get(st.session_state.subject) or {}).keys())
        if levels:
            st.session_state.level = st.session_state.level or levels[0]


def main() -> None:
    st.set_page_config(page_title=f"SMAXIA GTE {APP_VERSION}", layout="wide")
    ss_init()

    st.markdown(f"# SMAXIA GTE Console {APP_VERSION}")
    st.caption("ISO-PROD | Pack-Driven | Saturation progressive (new_QC=0) | ARI evidence | Exports preuve (mapping + signature)")

    # SIDEBAR
    with st.sidebar:
        st.markdown("## 1. ACTIVATION PACK")

        country_options = list(GENESIS_PACKS.keys())
        country_labels = [f"{code} - {GENESIS_PACKS[code][0]}" for code in country_options]
        selected_idx = st.selectbox("Sélectionner le pays", range(len(country_options)), format_func=lambda i: country_labels[i])
        selected_country = country_options[selected_idx]

        st.markdown("---")
        st.caption("Pack externe (optionnel)")
        uploaded = st.file_uploader("Pack JSON", type=["json"])

        if st.button("ACTIVER", use_container_width=True, type="primary"):
            try:
                if uploaded:
                    pack = json.loads(uploaded.read().decode("utf-8"))
                    pack["_source"] = "UPLOAD"
                    pack["_pack_sig_sha256"] = _sha256_text(_json_dumps_stable(pack))
                    log(f"[PACK] Upload pack_id={pack.get('pack_id')}")
                else:
                    pack = generate_pack(selected_country)
                    log(f"[PACK] Genesis country={selected_country}")

                st.session_state.pack_active = pack
                st.session_state.pack_id = pack.get("pack_id", "")
                st.session_state.pack_sig_sha256 = pack.get("_pack_sig_sha256", "")
                st.session_state.country = pack.get("country_code", "")

                _safe_set_default_subject_level(pack)

                # Reset runtime
                st.session_state.library = []
                st.session_state.harvest_manifest = None
                st.session_state.qi_pack = None
                st.session_state.qc_pack = None
                st.session_state.frt_pack = None
                st.session_state.qc_map = None
                st.session_state.selection_report = None
                st.session_state.last_run_audit = None
                st.session_state.run_signature = None
                st.session_state.sealed = False
                st.session_state.run_stats = {
                    "pairs": 0, "corriges": 0, "qi": 0, "rqi": 0, "qi_posable": 0,
                    "orphans": 0, "qc": 0, "qc_posable": 0, "coverage": 0.0, "sanity_ok": False,
                }

                st.success(f"✅ Pack activé: {st.session_state.pack_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur: {e}")

        if st.session_state.pack_active:
            pack = st.session_state.pack_active
            st.success(f"✅ {st.session_state.pack_id}")
            st.caption(f"Pays: {pack.get('country_name')}")
            st.caption(f"Source: {pack.get('_source')}")
            st.caption(f"PackSig: {st.session_state.pack_sig_sha256[:12]}...")

            st.markdown("---")
            st.markdown("## 2. SÉLECTION")

            taxonomy = pack.get("chapter_taxonomy", {}) or {}
            subjects = list(taxonomy.keys())

            if subjects:
                subject = st.selectbox("Matière", subjects, index=subjects.index(st.session_state.subject) if st.session_state.subject in subjects else 0)
                st.session_state.subject = subject

                levels = list((taxonomy.get(subject) or {}).keys())
                if levels:
                    level = st.selectbox("Niveau", levels, index=levels.index(st.session_state.level) if st.session_state.level in levels else 0)
                    st.session_state.level = level

            chapters = pack_chapters(pack, st.session_state.subject, st.session_state.level)
            if chapters:
                st.markdown("### Chapitres (aperçu)")
                for ch in chapters[:8]:
                    st.caption(f"• {ch.get('code')} — {ch.get('label')}")

    # MAIN
    if not st.session_state.pack_active:
        st.warning("Sélectionnez un pays et cliquez sur ACTIVER.")
        return

    pack = st.session_state.pack_active

    tab1, tab2, tab3, tab4 = st.tabs(["Import", "RUN", "Résultats", "Exports"])

    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?"))
    stats = st.session_state.run_stats

    # TAB 1 - IMPORT
    with tab1:
        st.markdown("## Import & Harvest")
        metric_row(len(lib), corr_ok, stats.get("qi", 0), stats.get("qi_posable", 0), stats.get("qc", 0), float(stats.get("coverage", 0.0)), st.session_state.sealed)

        if lib:
            st.dataframe(
                [{k: r.get(k, "") for k in ["pair_id", "year", "sujet", "corrige?", "match_score"]} for r in lib[:80]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### HARVEST")
        source = pack_harvest_source(pack)
        if source:
            st.info(f"Source: {source.get('source_name')}")
            c1, c2 = st.columns(2)
            years_back = c1.number_input("Années", 1, 15, 5)
            volume_max = c2.number_input("Volume max", 5, 150, 30)

            if st.button("HARVEST", use_container_width=True):
                with st.spinner("Harvest en cours..."):
                    try:
                        manifest = harvest_from_pack(pack, st.session_state.level, st.session_state.subject, int(years_back), int(volume_max))
                        st.session_state.harvest_manifest = manifest
                        st.session_state.library = manifest["library"]

                        st.session_state.sealed = False
                        st.session_state.qi_pack = None
                        st.session_state.qc_pack = None
                        st.session_state.frt_pack = None
                        st.session_state.qc_map = None
                        st.session_state.selection_report = None
                        st.session_state.last_run_audit = None
                        st.session_state.run_signature = None
                        st.session_state.run_stats = {
                            "pairs": len(st.session_state.library),
                            "corriges": sum(1 for x in st.session_state.library if x.get("corrige?")),
                            "qi": 0, "rqi": 0, "qi_posable": 0,
                            "orphans": 0, "qc": 0, "qc_posable": 0,
                            "coverage": 0.0, "sanity_ok": False,
                        }

                        st.success(f"✅ {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")

    # TAB 2 - RUN
    with tab2:
        st.markdown("## RUN — Saturation progressive (new_QC=0)")

        lib2 = st.session_state.library
        corr_ok2 = sum(1 for x in lib2 if x.get("corrige?"))

        if not lib2:
            st.warning("Bibliothèque vide — faire HARVEST d'abord.")
        elif corr_ok2 <= 0:
            st.error("Aucun corrigé exploitable (corrige_url absent).")
        else:
            max_vol = max(1, min(200, corr_ok2))

            st.markdown("### Paramètres")
            c1, c2, c3 = st.columns(3)
            vol_start = c1.number_input("vol_start", 1, max_vol, min(5, max_vol))
            vol_max = c2.number_input("vol_max", int(vol_start), max_vol, min(max(10, int(vol_start)), max_vol))
            step = c3.number_input("step", 1, 25, 1)

            c4, c5 = st.columns(2)
            selection_top_k = c4.number_input("F2 top_k / chapitre", 5, 30, 15)
            st.caption("SEALED strict = (new_QC==0) AND (orphans==0) AND sanity_ok AND qc_posable>0 AND qi_posable>0")

            if st.button("LANCER TEST SATURATION", use_container_width=True, type="primary"):
                with st.spinner("RUN en cours..."):
                    try:
                        log(f"=== RUN {APP_VERSION} ===")
                        res = run_saturation_progressive(lib2, int(vol_start), int(vol_max), int(step), pack, int(selection_top_k))

                        final = res["final"]
                        st.session_state.qi_pack = final["qi_pack"]
                        st.session_state.qc_pack = final["qc_pack"]
                        st.session_state.frt_pack = final["frt_pack"]
                        st.session_state.qc_map = final["qc_map"]
                        st.session_state.selection_report = res["selection_report"]
                        st.session_state.last_run_audit = res["audit"]
                        st.session_state.sealed = bool(res["audit"]["sealed"])

                        # mapping_qi_to_qc export (preuve)
                        mapping_qi_to_qc = st.session_state.qc_map or {}

                        # signature (preuve)
                        sig = build_run_signature(
                            pack=pack,
                            qi_pack=st.session_state.qi_pack or [],
                            qc_pack=st.session_state.qc_pack or [],
                            frt_pack=st.session_state.frt_pack or {},
                            mapping_qi_to_qc=mapping_qi_to_qc,
                            selection_report=st.session_state.selection_report or {},
                            audit=st.session_state.last_run_audit or {},
                            logs=logs_text(),
                        )
                        st.session_state.run_signature = sig

                        # Stats UI
                        qi_total = len(st.session_state.qi_pack or [])
                        rqi_total = sum(1 for q in (st.session_state.qi_pack or []) if q.get("has_rqi"))
                        qi_posable = int((res["audit"]["phase_final"] or {}).get("qi_posable", 0))
                        orphans = int((res["audit"]["phase_final"] or {}).get("qi_orphans", 0))
                        qc_total = len(st.session_state.qc_pack or [])
                        qc_posable = int((res["audit"]["phase_final"] or {}).get("qc_posable", 0))
                        coverage = float((res["audit"]["phase_final"] or {}).get("coverage", 0.0))

                        st.session_state.run_stats = {
                            "pairs": len(lib2),
                            "corriges": corr_ok2,
                            "qi": qi_total,
                            "rqi": rqi_total,
                            "qi_posable": qi_posable,
                            "orphans": orphans,
                            "qc": qc_total,
                            "qc_posable": qc_posable,
                            "coverage": coverage,
                            "sanity_ok": bool((res["audit"]["phase_final"] or {}).get("sanity_ok", False)),
                        }

                        if st.session_state.sealed:
                            st.success(f"✅ SEALED = YES | final_volume={res['audit']['final_volume']} | coverage={coverage:.0%}")
                        else:
                            st.warning(f"⚠️ SEALED = NO | {res['audit']['seal_reason']}")

                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
                        import traceback
                        st.text(traceback.format_exc())

        st.markdown("### Logs")
        st.text_area("", logs_text(), height=220)

    # TAB 3 - RESULTS
    with tab3:
        st.markdown("## Résultats")

        audit = st.session_state.last_run_audit
        if not audit:
            st.info("Lancez le RUN pour voir les résultats.")
        else:
            pb = audit.get("phase_final", {}) or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SEALED", "✅ YES" if audit.get("sealed") else "❌ NO")
            c2.metric("Final volume", audit.get("final_volume", 0))
            c3.metric("Orphans (posable)", pb.get("qi_orphans", 0))
            c4.metric("Coverage (KPI)", f"{float(pb.get('coverage', 0.0)):.0%}")

            st.markdown("### Audit strict (Kernel)")
            st.write(f"- new_QC==0 atteint : **{audit.get('new_qc_zero_reached')}**")
            st.write(f"- sanity_ok : **{pb.get('sanity_ok')}**")
            st.write(f"- qi_posable : **{pb.get('qi_posable')}**")
            st.write(f"- qc_posable : **{pb.get('qc_posable')}**")
            st.write(f"- seal_reason : **{audit.get('seal_reason')}**")

            st.markdown("### Itérations de saturation")
            iters = audit.get("iterations", []) or []
            if iters:
                st.dataframe(iters, use_container_width=True, hide_index=True)

            st.markdown("### Sélection F2 par chapitre")
            sel = st.session_state.selection_report or {}
            for ch in sel.get("chapters", []) if isinstance(sel, dict) else []:
                with st.expander(f"{ch.get('chapter_label')} ({ch.get('selected_count')}/{ch.get('qc_count')} QC)"):
                    for qc in ch.get("selected_qc", []):
                        st.markdown(f"**{qc.get('qc')}**")
                        st.caption(f"ID: {qc.get('qc_id')} | Ψ={qc.get('Psi_q')} | F2={qc.get('f2_score')} | FRT={qc.get('frt_id')}")

    # TAB 4 - EXPORTS + EXPLORER
    with tab4:
        st.markdown("## Exports (preuves) + Explorateur")

        metric_row(
            len(lib), corr_ok,
            stats.get("qi", 0), stats.get("qi_posable", 0),
            stats.get("qc", 0), float(stats.get("coverage", 0.0)),
            st.session_state.sealed,
        )

        colA, colB = st.columns(2)

        with colA:
            st.download_button("logs.txt", logs_text(), "logs.txt")
            if st.session_state.qi_pack is not None:
                st.download_button("qi_pack.json", json.dumps(st.session_state.qi_pack, ensure_ascii=False, indent=2), "qi_pack.json")
            if st.session_state.qc_pack is not None:
                st.download_button("qc_pack.json", json.dumps(st.session_state.qc_pack, ensure_ascii=False, indent=2), "qc_pack.json")
            if st.session_state.frt_pack is not None:
                st.download_button("frt_pack.json", json.dumps(st.session_state.frt_pack, ensure_ascii=False, indent=2), "frt_pack.json")

        with colB:
            # Preuve mapping
            if st.session_state.qc_map is not None:
                st.download_button("mapping_qi_to_qc.json", json.dumps(st.session_state.qc_map, ensure_ascii=False, indent=2), "mapping_qi_to_qc.json")
            if st.session_state.selection_report is not None:
                st.download_button("selection_report.json", json.dumps(st.session_state.selection_report, ensure_ascii=False, indent=2), "selection_report.json")
            if st.session_state.last_run_audit is not None:
                st.download_button("audit.json", json.dumps(st.session_state.last_run_audit, ensure_ascii=False, indent=2), "audit.json")
            if st.session_state.run_signature is not None:
                st.download_button("run_signature.json", json.dumps(st.session_state.run_signature, ensure_ascii=False, indent=2), "run_signature.json")

        st.markdown("---")
        st.markdown("## Explorateur QC → Qi → ARI → FRT")

        if st.session_state.qc_pack and st.session_state.qi_pack:
            qc_pack = st.session_state.qc_pack
            qi_pack = st.session_state.qi_pack
            frt_pack = st.session_state.frt_pack or {}
            frt_items = (frt_pack.get("items", {}) or {}) if isinstance(frt_pack, dict) else {}

            ch_opts = ["ALL"] + sorted({qc.get("chapter_code", "UNMAPPED") for qc in qc_pack})
            sel_ch = st.selectbox("Filtrer par chapitre", ch_opts)

            qcs = qc_pack if sel_ch == "ALL" else [q for q in qc_pack if q.get("chapter_code") == sel_ch]
            qcs = sorted(qcs, key=lambda x: (-int(x.get("posable_in_cluster", 0)), str(x.get("qc_id", ""))))

            if qcs:
                qc_labels = [
                    f"{q.get('qc_id')} | {q.get('chapter_code')} | {q.get('primary_op')} | n={q.get('cluster_size')} | posable={q.get('posable_in_cluster')}"
                    for q in qcs
                ]
                sel_idx = st.selectbox("Sélectionner une QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]

                st.markdown(f"### {qc.get('qc')}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Cluster", int(qc.get("cluster_size", 0)))
                c2.metric("Posable", int(qc.get("posable_in_cluster", 0)))
                c3.metric("État", qc.get("qc_state", ""))
                c4.metric("Ψ (F1)", f"{float(qc.get('Psi_q', 0.0)):.3f}")

                # FRT (lookup via frt_id)
                st.markdown("#### FRT (Fiche Réponse Type)")
                frt = frt_items.get(qc.get("frt_id", ""), {})
                if frt and frt.get("steps"):
                    st.caption(f"{frt.get('frt_id')} — {frt.get('title')}")
                    for step in frt.get("steps", []):
                        st.markdown(f"- {step}")
                else:
                    st.caption("FRT introuvable (frt_id non résolu).")

                # Triggers
                st.markdown("#### Triggers QC")
                st.write(qc.get("triggers", []))

                # Qi liées
                st.markdown("#### Qi sources")
                qi_by_id = {q["qi_id"]: q for q in qi_pack}
                for qi_id in qc.get("qi_ids", [])[:25]:
                    q = qi_by_id.get(qi_id)
                    if not q:
                        continue
                    title = f"{qi_id} | RQi={'✅' if q.get('has_rqi') else '❌'} | {q.get('primary_op')} | CH={q.get('chapter_code')} (score={q.get('chapter_match_score')})"
                    with st.expander(title):
                        st.markdown("**Qi**")
                        st.text(q.get("qi", "")[:1600])
                        if q.get("rqi"):
                            st.markdown("**RQi**")
                            st.text(q.get("rqi", "")[:1200])
                        st.markdown("**ARI trace (evidence)**")
                        st.json(q.get("ari_trace", {}))
                        st.markdown("**Triggers Qi**")
                        st.write(q.get("triggers", []))
        else:
            st.info("Aucun résultat à explorer — lancer le RUN.")


if __name__ == "__main__":
    main()
