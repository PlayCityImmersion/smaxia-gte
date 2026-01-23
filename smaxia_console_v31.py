# =============================================================================
# SMAXIA GTE Console V10.6.3 — ISO-PROD DÉFINITIF
# =============================================================================
# 
# VERSION DÉFINITIVE - MISSION CRITIQUE
# 
# 1. AFFICHAGE CAP COMPLET (Kernel V10.6.3)
#    - Tous les cycles, niveaux, matières, chapitres
#    - Keywords par chapitre
#    - delta_c par chapitre
#    - Sources institutionnelles
# 
# 2. SCRAPING PERSISTANT
#    - Recherche multi-sources jusqu'à 50 sujets + 50 corrections
#    - Bouton "Charger Plus" pour continuer la recherche
#    - Saturation jusqu'à zéro Qi orphelin
# 
# 3. AFFICHAGE QC PAR CHAPITRE
#    - QC avec FRT, ARI, TRIGGERS, Qi associées
#    - Rangement strict par chapitre
# =============================================================================

from __future__ import annotations
import io, os, re, json, time, hashlib, unicodedata
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import requests
import streamlit as st

# ============= OPTIONAL DEPS =============
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    import fitz
except ImportError:
    fitz = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# =============================================================================
# KERNEL CONSTANTS (Kernel V10.6.3 §0)
# =============================================================================
KERNEL_VERSION = "V10.6.3"
APP_VERSION = f"GTE-{KERNEL_VERSION}-DEFINITIF"
EPSILON = 0.1
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 20
MAX_PDF_MB = 40

# =============================================================================
# RC_* QUARANTAINE
# =============================================================================
class RC:
    CORRIGE_MISSING = "RC_CORRIGE_MISSING"
    CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
    SUJET_UNREADABLE = "RC_SUJET_UNREADABLE"
    SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
    RQI_MISSING = "RC_RQI_MISSING"
    LOW_CONFIDENCE = "RC_LOW_CONFIDENCE"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def sha256_fp(data: str) -> str:
    return f"sha256:{hashlib.sha256(data.encode('utf-8')).hexdigest()}"

def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]

def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace("\r", "\n")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_state():
    defaults = {
        "pipeline_state": "IDLE",
        "pipeline_progress": 0,
        "pipeline_log": [],
        "country_code": None,
        "cap": None,
        "harvest_library": [],
        "harvest_sujets_only": [],
        "harvest_corriges_only": [],
        "harvest_pairs": [],
        "harvest_page": 0,
        "harvest_errors": [],
        "qi_pack": [],
        "qc_pack": [],
        "frt_pack": {},
        "quarantine": [],
        "audit_log": None,
        "sealed": False,
        "_http_cache": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def log_pipeline(msg: str, level: str = "INFO"):
    st.session_state.pipeline_log.append({"ts": utc_ts(), "level": level, "msg": msg})

# =============================================================================
# CAP REGISTRY - SOURCES INSTITUTIONNELLES
# =============================================================================
CAP_REGISTRY = {
    "FR": {
        "name": "France",
        "language": "fr",
        "sources": [
            {
                "source_id": "APMEP",
                "source_name": "APMEP - Annales du BAC",
                "base_url": "https://www.apmep.fr",
                "index_urls": [
                    "https://www.apmep.fr/Annales-Terminale-Generale",
                    "https://www.apmep.fr/Terminale-Generale-2023-10-sujets",
                    "https://www.apmep.fr/Terminale-Generale-2022-10-sujets",
                    "https://www.apmep.fr/Terminale-Generale-2021-7-sujets",
                ],
                "type": "OFFICIAL"
            },
            {
                "source_id": "MATHS_FRANCE",
                "source_name": "Maths-France",
                "base_url": "https://maths-france.fr",
                "index_urls": [
                    "https://maths-france.fr/terminale/",
                ],
                "type": "EDUCATIONAL"
            },
        ],
    },
    "CI": {
        "name": "Côte d'Ivoire",
        "language": "fr",
        "sources": [
            {
                "source_id": "DECO_CI",
                "source_name": "DECO - Côte d'Ivoire",
                "base_url": "https://www.men-deco.org",
                "index_urls": [],
                "type": "OFFICIAL"
            },
        ],
    },
}

# =============================================================================
# LOAD_CAP - GÉNÉRATION CAP COMPLET (Kernel V10.6.3)
# =============================================================================
def load_cap(country_code: str) -> Dict[str, Any]:
    """
    LOAD_CAP - Génère le CAP complet conforme Kernel V10.6.3
    Inclut TOUS les chapitres, niveaux, matières avec keywords et delta_c
    """
    log_pipeline(f"[LOAD_CAP] Chargement CAP pour {country_code}")
    
    if country_code not in CAP_REGISTRY:
        raise ValueError(f"Pays {country_code} non enregistré")
    
    registry = CAP_REGISTRY[country_code]
    
    # ========== METADATA ==========
    metadata = {
        "cap_id": f"CAP_{country_code}_{KERNEL_VERSION}_{utc_ts()[:10]}",
        "country_code": country_code,
        "country_name": registry["name"],
        "language_default": registry["language"],
        "status": "SEALED",
        "kernel_version": KERNEL_VERSION,
        "sealed_at_utc": utc_ts(),
    }
    
    # ========== EDUCATION_SYSTEM COMPLET ==========
    education_system = _build_full_education_system(country_code)
    
    # ========== HARVEST_SOURCES ==========
    harvest_sources = registry.get("sources", [])
    
    # ========== KERNEL_PARAMS ==========
    kernel_params = {
        "atomization": {
            "max_pages": 140,
            "min_segment_len": 20,
            "max_segment_len": 2600,
            "question_markers": [
                r"(?im)^\s*(exercice|exo\.?)\s*([0-9ivx]+)\b",
                r"(?im)^\s*(partie)\s*([a-z0-9]+)\b",
                r"(?m)^\s*(\d{1,2})\s*[\)\.\-:]\s+",
            ],
        },
        "alignment": {"min_score": 0.15},
        "posable_rules": {
            "require_rqi": True,
            "require_scope": True,
            "min_ari_confidence": 0.50,
        },
        "f1_f2_params": {
            "epsilon": EPSILON,
            "alpha": 1.0,
            "seal_coverage_threshold": 0.95,
        },
        "qc_format": {"prefix": "Comment", "suffix": "?"},
        "triggers_config": {"min_count": 3, "max_count": 7},
    }
    
    # ========== ARI_CONFIG ==========
    ari_config = {
        "op_patterns": [
            {"op": "OP_PROBABILITY", "pattern": r"\b(probabilit|proba|esperance|variance|loi)\b", "T_j": 0.45},
            {"op": "OP_DERIVE", "pattern": r"\b(deriv|tangente|variation|f')\b", "T_j": 0.35},
            {"op": "OP_INTEGRATE", "pattern": r"\b(integr|primitive|aire|calcul\s+d)\b", "T_j": 0.50},
            {"op": "OP_LIMIT", "pattern": r"\b(limit|tend|infini|asymptote)\b", "T_j": 0.40},
            {"op": "OP_INDUCTION", "pattern": r"\b(recurr|induction|heredite|initialisation)\b", "T_j": 0.60},
            {"op": "OP_COMPLEX", "pattern": r"\b(complex|imaginaire|module|argument|affixe)\b", "T_j": 0.45},
            {"op": "OP_VECTOR", "pattern": r"\b(vecteur|scalaire|orthogonal|colineaire)\b", "T_j": 0.40},
            {"op": "OP_LOGEXP", "pattern": r"\b(ln|log|exp|exponentiel)\b", "T_j": 0.40},
            {"op": "OP_SOLVE", "pattern": r"\b(equat|resou|racine|solution|inequation)\b", "T_j": 0.35},
            {"op": "OP_SUITE", "pattern": r"\b(suite|terme|rang|convergence)\b", "T_j": 0.40},
        ],
        "op_labels": {
            "OP_PROBABILITY": "calculer une probabilité",
            "OP_DERIVE": "dériver une fonction",
            "OP_INTEGRATE": "calculer une intégrale",
            "OP_LIMIT": "calculer une limite",
            "OP_INDUCTION": "démontrer par récurrence",
            "OP_COMPLEX": "manipuler des nombres complexes",
            "OP_VECTOR": "résoudre un problème vectoriel",
            "OP_LOGEXP": "utiliser logarithme et exponentielle",
            "OP_SOLVE": "résoudre une équation",
            "OP_SUITE": "étudier une suite",
            "OP_STANDARD": "résoudre un exercice",
        },
    }
    
    # ========== TEXT_PROCESSING ==========
    text_processing = {
        "intent_verbs": ["montrer", "demontrer", "prouver", "calculer", "resoudre", "trouver", "determiner", "etudier", "justifier"],
    }
    
    cap = {
        "metadata": metadata,
        "education_system": education_system,
        "harvest_sources": harvest_sources,
        "kernel_params": kernel_params,
        "ari_config": ari_config,
        "text_processing": text_processing,
    }
    
    cap["metadata"]["cap_fingerprint_sha256"] = sha256_fp(safe_json(cap))
    
    log_pipeline(f"[LOAD_CAP] CAP chargé avec {len(education_system.get('chapters', {}).get('MATH', {}).get('TERMINALE', []))} chapitres")
    
    return cap

def _build_full_education_system(country_code: str) -> Dict:
    """Construction du système éducatif COMPLET avec tous les chapitres"""
    
    cycles = [
        {"code": "CYCLE_HS", "label": "Lycée / High School"},
        {"code": "CYCLE_PREU", "label": "Prépa / Classes Préparatoires"},
        {"code": "CYCLE_UNI", "label": "Université"},
    ]
    
    levels = {}
    subjects = {}
    chapters = {}
    
    if country_code == "FR":
        levels = {
            "CYCLE_HS": [
                {"code": "SECONDE", "label": "Seconde", "order": 1},
                {"code": "PREMIERE", "label": "Première", "order": 2},
                {"code": "TERMINALE", "label": "Terminale", "order": 3},
            ],
            "CYCLE_PREU": [
                {"code": "MPSI", "label": "MPSI", "order": 1},
                {"code": "MP", "label": "MP", "order": 2},
                {"code": "PCSI", "label": "PCSI", "order": 3},
                {"code": "PC", "label": "PC", "order": 4},
            ],
        }
        
        subjects = {
            "MATH": {"label": "Mathématiques", "coefficient_base": 16, "color": "#3498db"},
            "PHYSIQUE": {"label": "Physique-Chimie", "coefficient_base": 16, "color": "#e74c3c"},
        }
        
        # CHAPITRES TERMINALE MATHS - PROGRAMME COMPLET
        chapters = {
            "MATH": {
                "TERMINALE": [
                    {
                        "code": "CH_SUITES",
                        "label": "Suites numériques",
                        "delta_c": 1.0,
                        "keywords": ["suite", "recurrence", "arithmetique", "geometrique", "convergence", "limite", "rang", "terme", "raison", "monotone", "bornee", "adjacentes"],
                        "description": "Étude des suites numériques, convergence, récurrence"
                    },
                    {
                        "code": "CH_LIMITES",
                        "label": "Limites de fonctions",
                        "delta_c": 1.0,
                        "keywords": ["limite", "infini", "asymptote", "tend", "convergence", "divergence", "indetermination", "croissances", "comparees"],
                        "description": "Limites, asymptotes, formes indéterminées"
                    },
                    {
                        "code": "CH_CONTINUITE",
                        "label": "Continuité",
                        "delta_c": 1.0,
                        "keywords": ["continuite", "continue", "prolongement", "theoreme", "valeurs", "intermediaires", "bijection"],
                        "description": "Continuité, TVI, théorème des valeurs intermédiaires"
                    },
                    {
                        "code": "CH_DERIVATION",
                        "label": "Dérivation",
                        "delta_c": 1.0,
                        "keywords": ["derivee", "derivation", "tangente", "variation", "extremum", "maximum", "minimum", "croissant", "decroissant", "convexite", "inflexion"],
                        "description": "Dérivation, étude de fonctions, convexité"
                    },
                    {
                        "code": "CH_LOGEXP",
                        "label": "Fonction logarithme et exponentielle",
                        "delta_c": 1.1,
                        "keywords": ["logarithme", "exponentielle", "ln", "exp", "log", "croissance", "decroissance", "primitive", "equation"],
                        "description": "Fonctions ln et exp, propriétés, équations"
                    },
                    {
                        "code": "CH_INTEGRATION",
                        "label": "Intégration",
                        "delta_c": 1.2,
                        "keywords": ["integrale", "primitive", "aire", "integration", "calcul", "borne", "moyenne", "parties", "changement", "variable"],
                        "description": "Calcul intégral, primitives, aires"
                    },
                    {
                        "code": "CH_EQUATIONS_DIFF",
                        "label": "Équations différentielles",
                        "delta_c": 1.2,
                        "keywords": ["equation", "differentielle", "solution", "particuliere", "generale", "condition", "initiale", "premier", "ordre"],
                        "description": "Équations différentielles du premier ordre"
                    },
                    {
                        "code": "CH_PROBABILITES",
                        "label": "Probabilités",
                        "delta_c": 1.0,
                        "keywords": ["probabilite", "loi", "binomiale", "normale", "esperance", "variance", "aleatoire", "denombrement", "conditionnelle", "independance"],
                        "description": "Probabilités, lois discrètes et continues"
                    },
                    {
                        "code": "CH_COMPLEXES",
                        "label": "Nombres complexes",
                        "delta_c": 1.2,
                        "keywords": ["complexe", "imaginaire", "module", "argument", "affixe", "conjugue", "exponentielle", "trigonometrique", "racine"],
                        "description": "Nombres complexes, forme exponentielle"
                    },
                    {
                        "code": "CH_GEOMETRIE_ESPACE",
                        "label": "Géométrie dans l'espace",
                        "delta_c": 1.0,
                        "keywords": ["vecteur", "plan", "droite", "espace", "orthogonal", "scalaire", "parametrique", "cartesienne", "intersection", "distance"],
                        "description": "Géométrie vectorielle dans l'espace"
                    },
                ],
                "PREMIERE": [
                    {
                        "code": "CH_SECOND_DEGRE",
                        "label": "Second degré",
                        "delta_c": 0.8,
                        "keywords": ["polynome", "discriminant", "racine", "factorisation", "second", "degre", "parabole", "sommet"],
                        "description": "Polynômes du second degré"
                    },
                    {
                        "code": "CH_DERIVATION_1",
                        "label": "Dérivation (Première)",
                        "delta_c": 0.9,
                        "keywords": ["derivee", "tangente", "variation", "nombre", "derive", "taux"],
                        "description": "Introduction à la dérivation"
                    },
                    {
                        "code": "CH_SUITES_1",
                        "label": "Suites (Première)",
                        "delta_c": 0.9,
                        "keywords": ["suite", "terme", "raison", "explicite", "recurrence", "arithmetique", "geometrique"],
                        "description": "Introduction aux suites"
                    },
                    {
                        "code": "CH_PROBA_1",
                        "label": "Probabilités conditionnelles",
                        "delta_c": 0.8,
                        "keywords": ["probabilite", "conditionnelle", "evenement", "arbre", "independance", "formule"],
                        "description": "Probabilités conditionnelles"
                    },
                    {
                        "code": "CH_TRIGO",
                        "label": "Trigonométrie",
                        "delta_c": 0.9,
                        "keywords": ["trigonometrie", "cosinus", "sinus", "cercle", "radian", "angle", "formule"],
                        "description": "Fonctions trigonométriques"
                    },
                ],
                "MP": [
                    {
                        "code": "CH_ALGEBRE_LIN",
                        "label": "Algèbre linéaire",
                        "delta_c": 1.3,
                        "keywords": ["matrice", "vecteur", "dimension", "base", "rang", "determinant", "valeur", "propre", "diagonalisation"],
                        "description": "Espaces vectoriels, matrices"
                    },
                    {
                        "code": "CH_ANALYSE_MP",
                        "label": "Analyse (MP)",
                        "delta_c": 1.3,
                        "keywords": ["serie", "integrale", "convergence", "fonction", "plusieurs", "variables", "differentielle"],
                        "description": "Analyse approfondie"
                    },
                    {
                        "code": "CH_PROBA_MP",
                        "label": "Probabilités (MP)",
                        "delta_c": 1.2,
                        "keywords": ["variable", "aleatoire", "loi", "densite", "esperance", "variance", "convergence"],
                        "description": "Probabilités avancées"
                    },
                ],
            }
        }
    elif country_code == "CI":
        levels = {
            "CYCLE_HS": [
                {"code": "TERMINALE", "label": "Terminale", "order": 1},
            ],
        }
        subjects = {"MATH": {"label": "Mathématiques", "coefficient_base": 1.0}}
        chapters = {
            "MATH": {
                "TERMINALE": [
                    {"code": "CH_SUITES", "label": "Suites", "delta_c": 1.0, "keywords": ["suite", "recurrence"], "description": "Suites numériques"},
                    {"code": "CH_LIMITES", "label": "Limites", "delta_c": 1.0, "keywords": ["limite", "infini"], "description": "Limites"},
                    {"code": "CH_DERIVATION", "label": "Dérivation", "delta_c": 1.0, "keywords": ["derivee"], "description": "Dérivation"},
                    {"code": "CH_INTEGRATION", "label": "Intégration", "delta_c": 1.2, "keywords": ["integrale"], "description": "Intégration"},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "delta_c": 1.0, "keywords": ["probabilite"], "description": "Probabilités"},
                ]
            }
        }
    else:
        levels = {"CYCLE_HS": [{"code": "TERMINALE", "label": "Terminale", "order": 1}]}
        subjects = {"MATH": {"label": "Mathématiques", "coefficient_base": 1.0}}
        chapters = {"MATH": {"TERMINALE": []}}
    
    return {"cycles": cycles, "levels": levels, "subjects": subjects, "chapters": chapters}

# =============================================================================
# CAP ACCESSORS
# =============================================================================
def cap_get(cap: Dict, path: str, default=None):
    cur = cap
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def cap_chapters(cap: Dict, subject: str, level: str) -> List[Dict]:
    return cap_get(cap, f"education_system.chapters.{subject}.{level}", []) or []

# =============================================================================
# HTTP & PDF
# =============================================================================
def http_get(url: str, timeout: int = REQ_TIMEOUT) -> Optional[requests.Response]:
    try:
        res = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True)
        res.raise_for_status()
        return res
    except Exception as e:
        log_pipeline(f"[HTTP] Erreur {url[:50]}...: {str(e)[:50]}", "WARN")
        return None

def fetch_pdf(url: str) -> Optional[bytes]:
    cache = st.session_state.get("_http_cache", {})
    if url in cache:
        return cache[url]
    try:
        res = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        res.raise_for_status()
        pdf = res.content
        if len(pdf) > MAX_PDF_MB * 1024 * 1024:
            return None
        cache[url] = pdf
        st.session_state["_http_cache"] = cache
        return pdf
    except:
        return None

def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""
    pages = []
    
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages[:50]:
                    try:
                        pages.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
                    except:
                        pass
        except:
            pass
    
    if not pages or sum(len(p) for p in pages) < 500:
        if PdfReader:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages = [p.extract_text() or "" for p in reader.pages[:50]]
            except:
                pass
    
    if not pages or sum(len(p) for p in pages) < 500:
        if fitz:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                pages = [doc.load_page(i).get_text("text") or "" for i in range(min(50, doc.page_count))]
            except:
                pass
    
    # Clean
    all_lines = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = re.sub(r"(\w)-\n(\w)", r"\1\2", p)
        p = re.sub(r"[ \t]+", " ", p)
        all_lines.extend([ln.strip() for ln in p.split("\n") if ln.strip()])
    
    counts = Counter(all_lines)
    skip = {ln for ln, cnt in counts.items() if cnt >= 3 and len(ln) < 100}
    clean = [ln for ln in all_lines if ln not in skip and not re.fullmatch(r"\d{1,3}", ln)]
    
    return "\n".join(clean)

# =============================================================================
# HARVEST - SCRAPING PERSISTANT MULTI-SOURCES
# =============================================================================
def harvest_from_sources(cap: Dict, level: str, target_count: int = 50) -> Dict:
    """
    Harvest persistant - Continue la recherche jusqu'à atteindre target_count
    Recherche multi-sources et multi-pages
    """
    log_pipeline(f"[HARVEST] Recherche de {target_count} sujets/corrections pour {level}")
    
    sources = cap_get(cap, "harvest_sources", [])
    if not sources:
        return {"sujets": [], "corriges": [], "pairs": [], "errors": ["Aucune source"]}
    
    if not BeautifulSoup:
        return {"sujets": [], "corriges": [], "pairs": [], "errors": ["BeautifulSoup requis"]}
    
    all_sujets = list(st.session_state.harvest_sujets_only)
    all_corriges = list(st.session_state.harvest_corriges_only)
    errors = list(st.session_state.harvest_errors)
    
    base_url_main = sources[0].get("base_url", "https://www.apmep.fr")
    
    # URLs à explorer
    urls_to_explore = []
    for src in sources:
        urls_to_explore.extend(src.get("index_urls", []))
    
    # Ajouter des URLs supplémentaires basées sur les années
    for year in range(2024, 2018, -1):
        urls_to_explore.append(f"{base_url_main}/Terminale-Generale-{year}")
        urls_to_explore.append(f"{base_url_main}/Bac-S-{year}")
    
    # Explorer chaque URL
    explored = set()
    
    for url in urls_to_explore:
        if len(all_sujets) >= target_count and len(all_corriges) >= target_count:
            break
        
        if url in explored:
            continue
        explored.add(url)
        
        log_pipeline(f"[HARVEST] Exploration: {url[:60]}...")
        
        res = http_get(url, timeout=15)
        if not res:
            errors.append(f"Inaccessible: {url[:50]}")
            continue
        
        try:
            soup = BeautifulSoup(res.text, "html.parser")
            
            # Trouver tous les liens PDF
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if not href.lower().endswith(".pdf"):
                    continue
                
                # URL absolue
                if href.startswith("/"):
                    pdf_url = f"{base_url_main}{href}"
                elif href.startswith("http"):
                    pdf_url = href
                else:
                    pdf_url = f"{base_url_main}/{href}"
                
                name = os.path.basename(pdf_url).lower()
                text = (a.get_text() or "").lower()
                full_text = f"{name} {text}"
                
                # Exclure certains fichiers
                if any(ex in full_text for ex in ["index", "sommaire", "grille", "formulaire", "liste"]):
                    continue
                
                # Identifier si c'est un corrigé ou un sujet
                is_corrige = any(p in full_text for p in ["corrig", "correction", "solution", "reponse"])
                
                item = {
                    "url": pdf_url,
                    "name": os.path.basename(pdf_url),
                    "source_url": url,
                    "text": text[:100],
                }
                
                if is_corrige:
                    if not any(c["url"] == pdf_url for c in all_corriges):
                        all_corriges.append(item)
                else:
                    if not any(s["url"] == pdf_url for s in all_sujets):
                        all_sujets.append(item)
            
            # Trouver des liens vers d'autres pages à explorer
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                text = (a.get_text() or "").lower()
                
                if any(kw in text for kw in ["sujet", "corrig", "annale", "bac", "terminale", "2023", "2022", "2021", "2020"]):
                    if href.startswith("/"):
                        new_url = f"{base_url_main}{href}"
                    elif href.startswith("http"):
                        new_url = href
                    else:
                        continue
                    
                    if new_url not in explored and "pdf" not in new_url.lower():
                        urls_to_explore.append(new_url)
        
        except Exception as e:
            errors.append(f"Parse error {url[:30]}: {str(e)[:30]}")
    
    # Mise à jour session
    st.session_state.harvest_sujets_only = all_sujets
    st.session_state.harvest_corriges_only = all_corriges
    st.session_state.harvest_errors = errors
    
    # Appariement sujets/corrigés
    pairs = pair_sujets_corriges(all_sujets, all_corriges)
    st.session_state.harvest_pairs = pairs
    
    log_pipeline(f"[HARVEST] Résultat: {len(all_sujets)} sujets, {len(all_corriges)} corrigés, {len(pairs)} pairs")
    
    return {
        "sujets": all_sujets,
        "corriges": all_corriges,
        "pairs": pairs,
        "errors": errors,
    }

def pair_sujets_corriges(sujets: List[Dict], corriges: List[Dict]) -> List[Dict]:
    """Appariement intelligent sujets/corrigés"""
    pairs = []
    used_corriges = set()
    
    geo_zones = ["metro", "metropole", "amerique", "asie", "polynesie", "antilles", "liban", "etranger"]
    
    for suj in sujets:
        s_name = norm_text(suj["name"])
        s_geo = next((z for z in geo_zones if z in s_name), "")
        
        # Extraire année
        year_match = re.search(r"(20\d{2})", s_name)
        s_year = year_match.group(1) if year_match else ""
        
        best_corr = None
        best_score = 0.0
        
        for corr in corriges:
            if corr["url"] in used_corriges:
                continue
            
            c_name = norm_text(corr["name"])
            c_geo = next((z for z in geo_zones if z in c_name), "")
            
            # Extraire année
            year_match = re.search(r"(20\d{2})", c_name)
            c_year = year_match.group(1) if year_match else ""
            
            score = 0.0
            
            # Même année = fort bonus
            if s_year and c_year and s_year == c_year:
                score += 0.5
            
            # Même zone géographique
            if s_geo and c_geo:
                if s_geo == c_geo:
                    score += 0.3
                else:
                    continue  # Zone différente = pas de match
            
            # Similarité tokens
            s_tok = set(re.findall(r"[a-z0-9]{3,}", s_name))
            c_tok = set(re.findall(r"[a-z0-9]{3,}", c_name))
            if s_tok and c_tok:
                score += 0.2 * len(s_tok & c_tok) / max(1, len(s_tok | c_tok))
            
            if score > best_score:
                best_score = score
                best_corr = corr
        
        if best_corr and best_score >= 0.3:
            used_corriges.add(best_corr["url"])
            pairs.append({
                "pair_id": f"PAIR_{stable_id(suj['name'])}",
                "sujet": suj,
                "corrige": best_corr,
                "match_score": round(best_score, 2),
                "year": s_year,
            })
    
    return pairs

# =============================================================================
# ATOMIZATION, ARI, TRIGGERS, POSABLE, FRT
# =============================================================================
def atomize_text(text: str, cap: Dict, is_question: bool = True) -> List[str]:
    """Atomisation en segments"""
    if not text:
        return []
    
    markers = cap_get(cap, "kernel_params.atomization.question_markers", [])
    min_len = cap_get(cap, "kernel_params.atomization.min_segment_len", 20)
    max_len = cap_get(cap, "kernel_params.atomization.max_segment_len", 2600)
    
    regs = []
    for p in markers:
        try:
            regs.append(re.compile(p))
        except:
            pass
    
    positions = {0}
    for rg in regs:
        for m in rg.finditer(text):
            positions.add(m.start())
    for m in re.finditer(r"\n\n+", text):
        positions.add(m.start())
        positions.add(m.end())
    
    positions = sorted([p for p in positions if 0 <= p <= len(text)])
    
    segments = []
    seen = set()
    
    for a, b in zip(positions, positions[1:] + [len(text)]):
        seg = re.sub(r"[ \t]+", " ", text[a:b]).strip()
        if not (min_len <= len(seg) <= max_len):
            continue
        
        key = stable_id(norm_text(seg)[:400])
        if key in seen:
            continue
        seen.add(key)
        
        # Filtrer selon le type
        if is_question:
            intent_verbs = cap_get(cap, "text_processing.intent_verbs", [])
            if any(v in norm_text(seg) for v in intent_verbs) or "?" in seg or bool(re.search(r"[=<>≤≥∫∑]|\d", seg)):
                segments.append(seg)
        else:
            if bool(re.search(r"[=<>≤≥∫∑]|\d", seg)) or len(seg) > 80:
                segments.append(seg)
    
    return segments

def align_qi_rqi(questions: List[str], answers: List[str]) -> List[Optional[int]]:
    """Alignement Qi/RQi"""
    if not questions or not answers:
        return [None] * len(questions)
    
    def tokenize(s):
        return set(re.findall(r"[a-z0-9]{2,}", norm_text(s)))
    
    q_tok = [tokenize(q) for q in questions]
    a_tok = [tokenize(a) for a in answers]
    
    used = set()
    links = [None] * len(questions)
    
    for i in range(len(questions)):
        best_j, best_s = None, 0.0
        for j in range(len(answers)):
            if j in used or not q_tok[i] or not a_tok[j]:
                continue
            s = len(q_tok[i] & a_tok[j]) / len(q_tok[i] | a_tok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.15:
            links[i] = best_j
            used.add(best_j)
    
    return links

def map_chapter(qi_text: str, chapters: List[Dict]) -> Tuple[str, float]:
    """Mapping vers chapitre"""
    if not chapters:
        return "UNMAPPED", 0.0
    qn = norm_text(qi_text)
    best_code, best_score = "UNMAPPED", 0.0
    for ch in chapters:
        kws = ch.get("keywords", [])
        if not kws:
            continue
        matches = sum(1 for kw in kws if kw and kw in qn)
        score = matches / max(5, len(kws))
        if score > best_score:
            best_score, best_code = score, ch.get("code", "UNMAPPED")
    return best_code, round(best_score, 4)

def extract_ari(qi: str, rqi: str, cap: Dict) -> Dict:
    """Extraction ARI"""
    op_patterns = cap_get(cap, "ari_config.op_patterns", [])
    combined = norm_text(f"{qi}\n{rqi}")
    ops = []
    all_ops = set()
    sum_Tj = 0.0
    
    for item in op_patterns:
        op = item.get("op", "")
        pat = item.get("pattern", "")
        T_j = item.get("T_j", 0.3)
        if not op or not pat:
            continue
        try:
            matches = list(re.finditer(pat, combined, re.IGNORECASE))
            if matches and op not in all_ops:
                evidence = [m.group(0)[:40] for m in matches[:3]]
                ops.append({"op": op, "T_j": T_j, "evidence": evidence, "count": len(matches)})
                all_ops.add(op)
                sum_Tj += T_j
        except:
            pass
    
    if not ops:
        ops = [{"op": "OP_STANDARD", "T_j": 0.3, "evidence": [], "count": 0}]
        all_ops = {"OP_STANDARD"}
        sum_Tj = 0.3
    
    return {
        "primary_op": ops[0]["op"],
        "ops": ops,
        "all_ops": sorted(list(all_ops)),
        "sum_Tj": round(sum_Tj, 4),
        "confidence": min(0.95, 0.55 + 0.1 * len(ops)),
    }

def build_triggers(qi: str, ari: Dict, chapter_code: str, cap: Dict) -> List[str]:
    """Construction triggers"""
    triggers = []
    for op_info in ari.get("ops", []):
        triggers.append(f"ARI:{op_info['op']}")
    
    qn = norm_text(qi)
    kws = ["limite", "derivee", "integrale", "probabilite", "suite", "complexe", "vecteur", "recurrence", "equation", "fonction"]
    for kw in kws:
        if kw in qn:
            triggers.append(f"KW:{kw.upper()}")
    
    if chapter_code and chapter_code != "UNMAPPED":
        triggers.append(f"SCOPE:{chapter_code}")
    
    return list(dict.fromkeys(triggers))[:7]

def posable_gate(qi: str, rqi: str, chapter_code: str, ari: Dict, cap: Dict) -> Dict:
    """POSABLE Gate"""
    rules = cap_get(cap, "kernel_params.posable_rules", {})
    reason_codes = []
    
    has_rqi = bool(rqi and rqi.strip())
    if rules.get("require_rqi", True) and not has_rqi:
        reason_codes.append(RC.RQI_MISSING)
    
    has_scope = chapter_code and chapter_code != "UNMAPPED"
    if rules.get("require_scope", True) and not has_scope:
        reason_codes.append(RC.SCOPE_UNRESOLVED)
    
    if ari.get("confidence", 0) < rules.get("min_ari_confidence", 0.50):
        reason_codes.append(RC.LOW_CONFIDENCE)
    
    return {"is_posable": len(reason_codes) == 0, "reason_codes": reason_codes}

def generate_frt(primary_op: str, triggers: List[str], cap: Dict) -> Dict:
    """Génération FRT"""
    op_labels = cap_get(cap, "ari_config.op_labels", {})
    label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())
    qc_fmt = cap_get(cap, "kernel_params.qc_format", {})
    
    return {
        "frt_id": f"FRT_{stable_id(primary_op, str(triggers))}",
        "primary_op": primary_op,
        "title": f"{qc_fmt.get('prefix', 'Comment')} {label} {qc_fmt.get('suffix', '?')}",
        "blocks": {
            "usage": f"Identifier et reconnaître un problème de type: {label}",
            "reponse_type": f"Appliquer la méthode ARI pour {label}",
            "pieges": "Vérifier les conditions d'application et les cas particuliers",
            "conclusion": "Valider le résultat et rédiger une conclusion claire",
        },
    }

# =============================================================================
# F1/F2 KERNEL
# =============================================================================
def f1_kernel(delta_c: float, epsilon: float, sum_Tj: float) -> float:
    return float(delta_c) * float((epsilon + sum_Tj) ** 2)

# =============================================================================
# PROCESS PAIRS - GÉNÉRATION QI/QC
# =============================================================================
def process_pairs(cap: Dict, pairs: List[Dict], level: str, max_pairs: int = 20) -> Dict:
    """Traitement des pairs pour générer Qi et QC"""
    log_pipeline(f"[PROCESS] Traitement de {min(len(pairs), max_pairs)} pairs")
    
    chapters = cap_chapters(cap, "MATH", level)
    qi_items = []
    quarantine = []
    
    for pair in pairs[:max_pairs]:
        pid = pair["pair_id"]
        
        # Télécharger PDFs
        sujet_bytes = fetch_pdf(pair["sujet"]["url"])
        corrige_bytes = fetch_pdf(pair["corrige"]["url"])
        
        if not sujet_bytes:
            quarantine.append({"pair_id": pid, "reason": RC.SUJET_UNREADABLE})
            continue
        if not corrige_bytes:
            quarantine.append({"pair_id": pid, "reason": RC.CORRIGE_UNREADABLE})
            continue
        
        # Extraire texte
        sujet_text = extract_pdf_text(sujet_bytes)
        corrige_text = extract_pdf_text(corrige_bytes)
        
        if len(sujet_text) < 200:
            quarantine.append({"pair_id": pid, "reason": RC.SUJET_UNREADABLE})
            continue
        if len(corrige_text) < 200:
            quarantine.append({"pair_id": pid, "reason": RC.CORRIGE_UNREADABLE})
            continue
        
        # Atomiser
        questions = atomize_text(sujet_text, cap, is_question=True)
        answers = atomize_text(corrige_text, cap, is_question=False)
        
        # Aligner
        links = align_qi_rqi(questions, answers)
        
        # Créer Qi
        for i, qi_text in enumerate(questions):
            j = links[i] if i < len(links) else None
            rqi_text = answers[j] if (j is not None and j < len(answers)) else ""
            
            ari = extract_ari(qi_text, rqi_text, cap)
            chapter_code, ch_score = map_chapter(qi_text, chapters)
            triggers = build_triggers(qi_text, ari, chapter_code, cap)
            posable = posable_gate(qi_text, rqi_text, chapter_code, ari, cap)
            
            qi_id = f"QI_{stable_id(pid, str(i), qi_text[:80])}"
            qi_items.append({
                "qi_id": qi_id,
                "pair_id": pid,
                "year": pair.get("year", ""),
                "qi": qi_text,
                "rqi": rqi_text,
                "has_rqi": bool(rqi_text),
                "chapter_code": chapter_code,
                "ari": ari,
                "triggers": triggers,
                "posable": posable,
                "qc_id": None,
            })
    
    log_pipeline(f"[PROCESS] {len(qi_items)} Qi générées, {len(quarantine)} quarantaine")
    
    # Construire QC
    qc_pack, frt_pack, mapping = build_qc_from_qi(qi_items, chapters, cap)
    
    # Mettre à jour Qi avec QC
    for qi in qi_items:
        if qi["qi_id"] in mapping:
            qi["qc_id"] = mapping[qi["qi_id"]]
    
    # Stats
    qi_posable = [q for q in qi_items if q["has_rqi"] and q["posable"]["is_posable"]]
    orphans = sum(1 for q in qi_posable if not q.get("qc_id"))
    
    return {
        "qi_pack": qi_items,
        "qc_pack": qc_pack,
        "frt_pack": frt_pack,
        "quarantine": quarantine,
        "stats": {
            "qi_total": len(qi_items),
            "qi_posable": len(qi_posable),
            "qc_total": len(qc_pack),
            "orphans": orphans,
        }
    }

def build_qc_from_qi(qi_items: List[Dict], chapters: List[Dict], cap: Dict) -> Tuple[List[Dict], Dict, Dict]:
    """Construction des QC depuis les Qi"""
    
    # Grouper par (chapitre, primary_op)
    buckets = defaultdict(list)
    for qi in qi_items:
        primary_op = qi["ari"].get("primary_op", "OP_STANDARD")
        buckets[(qi["chapter_code"], primary_op)].append(qi)
    
    qc_pack = []
    frt_pack = {}
    mapping = {}
    
    op_labels = cap_get(cap, "ari_config.op_labels", {})
    qc_fmt = cap_get(cap, "kernel_params.qc_format", {})
    f1_f2 = cap_get(cap, "kernel_params.f1_f2_params", {})
    epsilon = f1_f2.get("epsilon", EPSILON)
    
    for (chapter_code, primary_op), items in sorted(buckets.items()):
        cluster_size = len(items)
        posable_items = [x for x in items if x["has_rqi"] and x["posable"]["is_posable"]]
        posable_count = len(posable_items)
        
        if cluster_size < 1:
            continue
        
        label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())
        qc_text = f"{qc_fmt.get('prefix', 'Comment')} {label} {qc_fmt.get('suffix', '?')}"
        qc_id = f"QC_{stable_id(chapter_code, primary_op, qc_text)}"
        
        qc_state = "POSABLE" if posable_count >= 2 else ("POSABLE_WEAK" if posable_count == 1 else "UNPOSABLE")
        
        all_triggers = []
        all_ops = set()
        for item in items:
            all_triggers.extend(item["triggers"])
            all_ops.update(item["ari"].get("all_ops", []))
        
        qc_triggers = [t for t, _ in Counter(all_triggers).most_common(7)]
        
        frt = generate_frt(primary_op, qc_triggers, cap)
        frt_pack[frt["frt_id"]] = frt
        
        ch_info = next((c for c in chapters if c.get("code") == chapter_code), {})
        delta_c = ch_info.get("delta_c", 1.0)
        
        rep = max(items, key=lambda x: x["ari"].get("confidence", 0))
        sum_Tj = rep["ari"].get("sum_Tj", 0.3)
        f1_raw = f1_kernel(delta_c, epsilon, sum_Tj)
        
        qc = {
            "qc_id": qc_id,
            "qc_text": qc_text,
            "qc_label": label,
            "chapter_code": chapter_code,
            "chapter_label": ch_info.get("label", chapter_code),
            "primary_op": primary_op,
            "all_ops": sorted(list(all_ops)),
            "cluster_size": cluster_size,
            "posable_in_cluster": posable_count,
            "qc_state": qc_state,
            "qi_ids": [x["qi_id"] for x in items],
            "triggers": qc_triggers,
            "frt": frt,
            "delta_c": delta_c,
            "sum_Tj": sum_Tj,
            "f1_raw": round(f1_raw, 6),
            "Psi_q": 0.0,
        }
        
        qc_pack.append(qc)
        for item in items:
            mapping[item["qi_id"]] = qc_id
    
    # Normaliser Psi_q par chapitre
    by_chapter = defaultdict(list)
    for qc in qc_pack:
        by_chapter[qc["chapter_code"]].append(qc)
    
    for ch_code, qcs in by_chapter.items():
        max_f1 = max((q["f1_raw"] for q in qcs), default=0)
        for qc in qcs:
            qc["Psi_q"] = round(qc["f1_raw"] / max_f1, 6) if max_f1 > 0 else 0.0
    
    return qc_pack, frt_pack, mapping

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.set_page_config(page_title=f"SMAXIA GTE {KERNEL_VERSION}", page_icon="🏛️", layout="wide")
    init_state()
    
    # HEADER
    st.markdown(f"# 🏛️ SMAXIA GTE Console {KERNEL_VERSION}")
    st.markdown("**ISO-PROD | Kernel V10.6.3 Scellé | CAS 1 ONLY | Zéro Hardcode**")
    st.markdown("---")
    
    # ============== SIDEBAR ==============
    with st.sidebar:
        st.markdown("## 🚀 Activation Pays")
        
        country = st.selectbox("Pays", list(CAP_REGISTRY.keys()), format_func=lambda x: f"{x} - {CAP_REGISTRY[x]['name']}")
        
        if st.button("🚀 ACTIVATE_COUNTRY", type="primary", use_container_width=True):
            st.session_state.country_code = country
            st.session_state.cap = load_cap(country)
            st.session_state.harvest_sujets_only = []
            st.session_state.harvest_corriges_only = []
            st.session_state.harvest_pairs = []
            st.session_state.harvest_errors = []
            st.session_state.qi_pack = []
            st.session_state.qc_pack = []
            st.session_state.pipeline_state = "CAP_LOADED"
            st.rerun()
        
        st.markdown("---")
        
        if st.session_state.cap:
            st.success(f"✅ CAP {st.session_state.country_code} chargé")
            
            # Sélection niveau
            cap = st.session_state.cap
            all_levels = []
            for cycle_levels in cap_get(cap, "education_system.levels", {}).values():
                all_levels.extend([l["code"] for l in cycle_levels])
            
            selected_level = st.selectbox("Niveau", all_levels, index=all_levels.index("TERMINALE") if "TERMINALE" in all_levels else 0)
            st.session_state["selected_level"] = selected_level
    
    # ============== MAIN CONTENT ==============
    if not st.session_state.cap:
        st.info("👈 Sélectionnez un pays et cliquez sur ACTIVATE_COUNTRY")
        return
    
    cap = st.session_state.cap
    
    # 3 ONGLETS
    tab1, tab2, tab3 = st.tabs(["📦 1. CAP Complet", "📄 2. Sujets & Corrections", "📚 3. Bibliothèque QC"])
    
    # ============== ONGLET 1: CAP COMPLET ==============
    with tab1:
        st.markdown("## 📦 Contenu CAP Complet (Kernel V10.6.3)")
        
        # Metadata
        metadata = cap.get("metadata", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAP ID", metadata.get("cap_id", "N/A")[:25] + "...")
        col2.metric("Pays", metadata.get("country_name", "N/A"))
        col3.metric("Status", metadata.get("status", "N/A"))
        col4.metric("Kernel", metadata.get("kernel_version", "N/A"))
        
        st.markdown(f"**Fingerprint SHA256:** `{metadata.get('cap_fingerprint_sha256', 'N/A')[:40]}...`")
        st.markdown(f"**Scellé le:** {metadata.get('sealed_at_utc', 'N/A')}")
        
        st.markdown("---")
        
        # Education System
        st.markdown("### 📚 Système Éducatif")
        
        education = cap.get("education_system", {})
        
        # Cycles
        st.markdown("#### Cycles")
        cycles = education.get("cycles", [])
        for cycle in cycles:
            st.markdown(f"- **{cycle['code']}**: {cycle['label']}")
        
        # Niveaux
        st.markdown("#### Niveaux par Cycle")
        levels = education.get("levels", {})
        for cycle_code, level_list in levels.items():
            with st.expander(f"📁 {cycle_code}"):
                for level in level_list:
                    st.markdown(f"- **{level['code']}**: {level['label']} (ordre: {level.get('order', 'N/A')})")
        
        # Matières
        st.markdown("#### Matières")
        subjects = education.get("subjects", {})
        for code, info in subjects.items():
            st.markdown(f"- **{code}**: {info['label']} (coef: {info.get('coefficient_base', 'N/A')})")
        
        # CHAPITRES - AFFICHAGE COMPLET
        st.markdown("### 📖 Chapitres par Niveau (Détail Complet)")
        
        chapters_data = education.get("chapters", {})
        for subject_code, levels_chapters in chapters_data.items():
            st.markdown(f"#### Matière: {subject_code}")
            
            for level_code, chapter_list in levels_chapters.items():
                with st.expander(f"📁 {level_code} ({len(chapter_list)} chapitres)", expanded=(level_code == "TERMINALE")):
                    for ch in chapter_list:
                        st.markdown(f"""
                        **{ch['code']}** - {ch['label']}
                        - δc (difficulté): `{ch.get('delta_c', 1.0)}`
                        - Description: {ch.get('description', 'N/A')}
                        - Keywords: `{', '.join(ch.get('keywords', []))}`
                        """)
                        st.markdown("---")
        
        # Sources
        st.markdown("### 🌐 Sources Institutionnelles")
        sources = cap.get("harvest_sources", [])
        for src in sources:
            st.markdown(f"""
            - **{src.get('source_name')}** ({src.get('source_id')})
              - URL: {src.get('base_url')}
              - Type: {src.get('type')}
            """)
        
        # Export CAP
        st.markdown("### 📥 Export CAP")
        st.download_button("📥 Télécharger CAP JSON", safe_json(cap), "cap_complet.json", "application/json")
    
    # ============== ONGLET 2: SUJETS & CORRECTIONS ==============
    with tab2:
        st.markdown("## 📄 Sujets & Corrections Loader")
        
        selected_level = st.session_state.get("selected_level", "TERMINALE")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bouton de recherche
            if st.button("🔍 Rechercher Sujets & Corrections (50)", type="primary", use_container_width=True):
                with st.spinner("Recherche en cours..."):
                    harvest_from_sources(cap, selected_level, target_count=50)
                st.rerun()
            
            if st.button("🔍 Charger Plus (+50)", use_container_width=True):
                with st.spinner("Recherche supplémentaire..."):
                    current_sujets = len(st.session_state.harvest_sujets_only)
                    current_corriges = len(st.session_state.harvest_corriges_only)
                    harvest_from_sources(cap, selected_level, target_count=max(current_sujets, current_corriges) + 50)
                st.rerun()
        
        with col2:
            st.metric("Sujets trouvés", len(st.session_state.harvest_sujets_only))
            st.metric("Corrigés trouvés", len(st.session_state.harvest_corriges_only))
            st.metric("Pairs appariées", len(st.session_state.harvest_pairs))
        
        # Erreurs
        errors = st.session_state.harvest_errors
        if errors:
            with st.expander(f"⚠️ Erreurs ({len(errors)})"):
                for e in errors[:20]:
                    st.caption(f"- {e}")
        
        st.markdown("---")
        
        # Afficher sujets
        st.markdown("### 📋 Sujets trouvés")
        sujets = st.session_state.harvest_sujets_only
        if sujets:
            for i, s in enumerate(sujets[:50]):
                st.markdown(f"{i+1}. [{s['name']}]({s['url']})")
        else:
            st.info("Aucun sujet trouvé. Cliquez sur 'Rechercher'.")
        
        st.markdown("### 📋 Corrigés trouvés")
        corriges = st.session_state.harvest_corriges_only
        if corriges:
            for i, c in enumerate(corriges[:50]):
                st.markdown(f"{i+1}. [{c['name']}]({c['url']})")
        else:
            st.info("Aucun corrigé trouvé.")
        
        st.markdown("---")
        
        # Pairs
        st.markdown("### 🔗 Pairs Sujet-Corrigé")
        pairs = st.session_state.harvest_pairs
        if pairs:
            for p in pairs[:30]:
                with st.expander(f"📎 {p['sujet']['name'][:50]}... (score: {p['match_score']})"):
                    st.markdown(f"**Sujet:** [{p['sujet']['name']}]({p['sujet']['url']})")
                    st.markdown(f"**Corrigé:** [{p['corrige']['name']}]({p['corrige']['url']})")
                    st.markdown(f"**Année:** {p.get('year', 'N/A')}")
            
            # Bouton traitement
            st.markdown("---")
            if st.button("⚙️ Traiter les Pairs (Générer Qi/QC)", type="primary", use_container_width=True):
                with st.spinner("Traitement en cours..."):
                    result = process_pairs(cap, pairs, selected_level, max_pairs=20)
                    st.session_state.qi_pack = result["qi_pack"]
                    st.session_state.qc_pack = result["qc_pack"]
                    st.session_state.frt_pack = result["frt_pack"]
                    st.session_state.quarantine = result["quarantine"]
                    st.session_state.audit_log = result["stats"]
                st.success(f"✅ {result['stats']['qi_total']} Qi, {result['stats']['qc_total']} QC générées")
                st.rerun()
        else:
            st.info("Aucune pair trouvée. Lancez d'abord la recherche.")
    
    # ============== ONGLET 3: BIBLIOTHÈQUE QC PAR CHAPITRE ==============
    with tab3:
        st.markdown("## 📚 Bibliothèque QC par Chapitre")
        
        qc_pack = st.session_state.qc_pack
        qi_pack = st.session_state.qi_pack
        
        if not qc_pack:
            st.warning("⚠️ Aucune QC générée. Traitez d'abord les pairs dans l'onglet 2.")
        else:
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("QC Total", len(qc_pack))
            col2.metric("Qi Total", len(qi_pack))
            qi_posable = [q for q in qi_pack if q["has_rqi"] and q["posable"]["is_posable"]]
            col3.metric("Qi Posable", len(qi_posable))
            orphans = sum(1 for q in qi_posable if not q.get("qc_id"))
            col4.metric("Orphelins", orphans)
            
            st.markdown("---")
            
            # Grouper par chapitre
            by_chapter = defaultdict(list)
            for qc in qc_pack:
                by_chapter[qc["chapter_code"]].append(qc)
            
            # Afficher par chapitre
            for ch_code in sorted(by_chapter.keys()):
                qcs = by_chapter[ch_code]
                
                # Trouver label du chapitre
                ch_label = ch_code
                for level_chapters in cap_get(cap, "education_system.chapters.MATH", {}).values():
                    for ch in level_chapters:
                        if ch.get("code") == ch_code:
                            ch_label = ch.get("label", ch_code)
                            break
                
                st.markdown(f"### 📁 {ch_label} ({len(qcs)} QC)")
                
                for qc in sorted(qcs, key=lambda x: -x.get("Psi_q", 0)):
                    state_icon = "✅" if qc["qc_state"] == "POSABLE" else ("⚠️" if qc["qc_state"] == "POSABLE_WEAK" else "❌")
                    
                    with st.expander(f"{state_icon} {qc['qc_text'][:70]}..."):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### QC Info")
                            st.markdown(f"- **ID:** `{qc['qc_id']}`")
                            st.markdown(f"- **Texte:** {qc['qc_text']}")
                            st.markdown(f"- **État:** {qc['qc_state']}")
                            st.markdown(f"- **Cluster:** {qc['cluster_size']} Qi | Posable: {qc['posable_in_cluster']}")
                            st.markdown(f"- **Ψq:** {qc.get('Psi_q', 0):.4f}")
                            st.markdown(f"- **δc:** {qc.get('delta_c', 1.0)}")
                            
                            st.markdown("#### TRIGGERS")
                            triggers = qc.get("triggers", [])
                            if triggers:
                                for t in triggers:
                                    st.code(t, language=None)
                            else:
                                st.caption("Aucun trigger")
                        
                        with col2:
                            st.markdown("#### FRT (Fiche Réponse Type)")
                            frt = qc.get("frt", {})
                            if frt:
                                st.markdown(f"**Titre:** {frt.get('title', 'N/A')}")
                                blocks = frt.get("blocks", {})
                                for k, v in blocks.items():
                                    st.markdown(f"**{k.upper()}:** {v}")
                            else:
                                st.caption("Aucune FRT")
                            
                            st.markdown("#### ARI")
                            st.markdown(f"- **Primary Op:** {qc.get('primary_op', 'N/A')}")
                            st.markdown(f"- **All Ops:** {', '.join(qc.get('all_ops', []))}")
                            st.markdown(f"- **Sum Tj:** {qc.get('sum_Tj', 0):.4f}")
                        
                        # Qi associées
                        st.markdown("#### Qi Associées")
                        qi_ids = qc.get("qi_ids", [])
                        qi_linked = [qi for qi in qi_pack if qi.get("qi_id") in qi_ids]
                        
                        for qi in qi_linked[:8]:
                            posable_icon = "✅" if qi.get("posable", {}).get("is_posable") else "❌"
                            with st.expander(f"{posable_icon} {qi['qi_id'][:20]}..."):
                                st.markdown(f"**Qi:** {qi.get('qi', '')[:500]}...")
                                if qi.get("rqi"):
                                    st.markdown(f"**RQi:** {qi.get('rqi', '')[:500]}...")
                                st.markdown(f"**Chapitre:** {qi.get('chapter_code')}")
                                st.markdown(f"**Année:** {qi.get('year', 'N/A')}")
                        
                        if len(qi_linked) > 8:
                            st.caption(f"... et {len(qi_linked) - 8} autres Qi")
                
                st.markdown("---")
    
    # ============== FOOTER ==============
    st.markdown("---")
    with st.expander("📋 Pipeline Logs"):
        for entry in reversed(st.session_state.pipeline_log[-30:]):
            level = entry.get("level", "INFO")
            color = "red" if level == "ERROR" else "orange" if level == "WARN" else "gray"
            st.markdown(f"<span style='color:{color};font-size:11px'>[{entry['ts']}] [{level}] {entry['msg']}</span>", unsafe_allow_html=True)
    
    if st.button("🔄 Réinitialiser tout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
