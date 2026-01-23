# =============================================================================
# SMAXIA GTE Console V10.6.3 — ISO-PROD (KERNEL V10.6.3 STRICT)
# =============================================================================
# VERSION AMÉLIORÉE - Gestion robuste des erreurs + 3 onglets obligatoires
#
# DOCTRINE KERNEL V10.6.3:
# - SEULE ACTION HUMAINE: ACTIVATE_COUNTRY(country_code)
# - ZÉRO HARDCODE MÉTIER
# - Pipeline 100% automatique après activation
# - UI LECTURE SEULE après exécution
#
# 3 ONGLETS OBLIGATOIRES:
# 1) CAP Loader + Sources
# 2) Sujets & Corrections Loader + Sources  
# 3) Bibliothèque QC par Chapitre (QC → FRT → ARI → Triggers → Qi)
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
# KERNEL CONSTANTS (INVARIANTS UNIVERSELS - Kernel V10.6.3 §0)
# =============================================================================
KERNEL_VERSION = "V10.6.3"
APP_VERSION = f"GTE-{KERNEL_VERSION}-ISO-PROD"
FINGERPRINT_ALGORITHM = "SHA256"
EPSILON = 0.1

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 30
MAX_PDF_MB = 40

# =============================================================================
# RC_* QUARANTAINE (Kernel §12)
# =============================================================================
class RC:
    CORRIGE_MISSING = "RC_CORRIGE_MISSING"
    CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
    SUJET_UNREADABLE = "RC_SUJET_UNREADABLE"
    SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
    RQI_MISSING = "RC_RQI_MISSING"
    LOW_CONFIDENCE = "RC_LOW_CONFIDENCE"
    SINGLETON = "RC_SINGLETON_IRREDUCTIBLE"

# =============================================================================
# PIPELINE STATE
# =============================================================================
class PipelineState:
    IDLE = "IDLE"
    ACTIVATING = "ACTIVATING"
    LOADING_CAP = "LOADING_CAP"
    HARVESTING = "HARVESTING"
    EXTRACTING = "EXTRACTING"
    ATOMIZING = "ATOMIZING"
    POSABLE_GATE = "POSABLE_GATE"
    CLUSTERING = "CLUSTERING"
    IA1_PROCESSING = "IA1_PROCESSING"
    IA2_JUDGE = "IA2_JUDGE"
    F1_F2_SCORING = "F1_F2_SCORING"
    SELECTION = "SELECTION"
    COVERAGE_CHECK = "COVERAGE_CHECK"
    SEALED = "SEALED"
    SAFETY_STOP = "SAFETY_STOP"
    ERROR = "ERROR"
    COMPLETED_WITH_DATA = "COMPLETED_WITH_DATA"

# =============================================================================
# UTILITY FUNCTIONS (Kernel invariants)
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
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_state():
    defaults = {
        "pipeline_state": PipelineState.IDLE,
        "pipeline_progress": 0,
        "pipeline_log": [],
        "pipeline_errors": [],
        "country_code": None,
        "cap": None,
        "cap_status": None,
        "harvest_manifest": None,
        "harvest_status": None,
        "qi_pack": [],
        "qc_pack": [],
        "frt_pack": {},
        "coverage_map": [],
        "evidence_pack": None,
        "audit_log": None,
        "quarantine": [],
        "sealed": False,
        "seal_reason": [],
        "_http_cache": {},
        "_run_ts": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def log_pipeline(msg: str, level: str = "INFO"):
    entry = {"ts": utc_ts(), "level": level, "msg": msg}
    st.session_state.pipeline_log.append(entry)
    if level == "ERROR":
        st.session_state.pipeline_errors.append(msg)

def set_state(state: str, progress: int = None):
    st.session_state.pipeline_state = state
    if progress is not None:
        st.session_state.pipeline_progress = progress

# =============================================================================
# CAP REGISTRY (Sources institutionnelles - SEULE donnée pays dans Kernel)
# =============================================================================
CAP_REGISTRY = {
    "FR": {
        "name": "France",
        "language": "fr",
        "institutional_sources": [
            {
                "source_id": "APMEP",
                "source_name": "APMEP - Annales du BAC",
                "base_url": "https://www.apmep.fr",
                "endpoints": {
                    "TERMINALE": "/Annales-Terminale-Generale",
                    "PREMIERE": "/Annales-du-Bac-Premiere",
                },
                "type": "OFFICIAL_ARCHIVE"
            }
        ],
    },
    "CI": {
        "name": "Côte d'Ivoire",
        "language": "fr",
        "institutional_sources": [
            {
                "source_id": "DECO_CI",
                "source_name": "Direction des Examens et Concours",
                "base_url": "https://www.men-deco.org",
                "endpoints": {},
                "type": "OFFICIAL_ARCHIVE"
            }
        ],
    },
}

# =============================================================================
# LOAD_CAP (Kernel §0.4)
# =============================================================================
def load_cap(country_code: str) -> Dict[str, Any]:
    """LOAD_CAP - Charge le CAP officiel"""
    log_pipeline(f"[LOAD_CAP] Chargement CAP pour {country_code}")
    set_state(PipelineState.LOADING_CAP, 5)
    
    if country_code not in CAP_REGISTRY:
        raise ValueError(f"Pays {country_code} non enregistré")
    
    registry = CAP_REGISTRY[country_code]
    
    # A) METADATA
    metadata = {
        "cap_id": f"CAP_{country_code}_{KERNEL_VERSION}_{utc_ts()[:10]}",
        "country_code": country_code,
        "country_name": registry["name"],
        "language_default": registry["language"],
        "status": "SEALED",
        "cap_fingerprint_sha256": "",
        "sealed_at_utc": utc_ts(),
        "kernel_version": KERNEL_VERSION,
    }
    
    # B) EDUCATION_SYSTEM
    education_system = _build_education_system(country_code, registry)
    
    # C) HARVEST_SOURCES
    harvest_sources = _build_harvest_sources(registry)
    
    # D) KERNEL_PARAMS
    kernel_params = _build_kernel_params()
    
    # E) ARI_CONFIG
    ari_config = _build_ari_config()
    
    # F) TEXT_PROCESSING
    text_processing = _build_text_processing(registry["language"])
    
    cap = {
        "metadata": metadata,
        "education_system": education_system,
        "harvest_sources": harvest_sources,
        "kernel_params": kernel_params,
        "ari_config": ari_config,
        "text_processing": text_processing,
    }
    
    cap["metadata"]["cap_fingerprint_sha256"] = sha256_fp(safe_json(cap))
    
    log_pipeline(f"[LOAD_CAP] CAP chargé: {cap['metadata']['cap_id']}")
    
    st.session_state.cap_status = {
        "status": "LOADED",
        "cap_id": cap["metadata"]["cap_id"],
        "fingerprint": cap["metadata"]["cap_fingerprint_sha256"][:24] + "...",
        "timestamp": utc_ts(),
        "country": registry["name"],
        "sources_count": len(harvest_sources),
    }
    
    return cap

def _build_education_system(country_code: str, registry: Dict) -> Dict:
    cycles = [
        {"code": "CYCLE_HS", "label": "Lycée"},
        {"code": "CYCLE_PREU", "label": "Prépa"},
    ]
    
    levels = {}
    subjects = {}
    chapters = {}
    
    if country_code == "FR":
        levels = {
            "CYCLE_HS": [
                {"code": "TERMINALE", "label": "Terminale", "order": 1},
                {"code": "PREMIERE", "label": "Première", "order": 2},
            ],
            "CYCLE_PREU": [
                {"code": "MP", "label": "MP", "order": 1},
            ],
        }
        subjects = {"MATH": {"label": "Mathématiques", "coefficient_base": 1.0}}
        chapters = {
            "MATH": {
                "TERMINALE": [
                    {"code": "CH_SUITES", "label": "Suites numériques", "delta_c": 1.0,
                     "keywords": ["suite", "recurrence", "arithmetique", "geometrique", "convergence", "limite", "rang"]},
                    {"code": "CH_LIMITES", "label": "Limites de fonctions", "delta_c": 1.0,
                     "keywords": ["limite", "infini", "asymptote", "tend", "convergence"]},
                    {"code": "CH_DERIVATION", "label": "Dérivation", "delta_c": 1.0,
                     "keywords": ["derivee", "derivation", "tangente", "variation", "extremum"]},
                    {"code": "CH_CONTINUITE", "label": "Continuité", "delta_c": 1.0,
                     "keywords": ["continuite", "continue", "prolongement", "theoreme"]},
                    {"code": "CH_LOGEXP", "label": "Logarithme et Exponentielle", "delta_c": 1.1,
                     "keywords": ["logarithme", "exponentielle", "ln", "exp", "log"]},
                    {"code": "CH_INTEGRATION", "label": "Intégration", "delta_c": 1.2,
                     "keywords": ["integrale", "primitive", "aire", "integration"]},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "delta_c": 1.0,
                     "keywords": ["probabilite", "loi", "binomiale", "normale", "esperance", "variance"]},
                    {"code": "CH_COMPLEXES", "label": "Nombres complexes", "delta_c": 1.2,
                     "keywords": ["complexe", "imaginaire", "module", "argument", "affixe"]},
                    {"code": "CH_GEOMETRIE", "label": "Géométrie espace", "delta_c": 1.0,
                     "keywords": ["vecteur", "plan", "droite", "espace", "orthogonal"]},
                ],
                "MP": [
                    {"code": "CH_ALGEBRE_LIN", "label": "Algèbre linéaire", "delta_c": 1.3,
                     "keywords": ["matrice", "vecteur", "dimension", "base", "rang", "determinant"]},
                    {"code": "CH_ANALYSE_MP", "label": "Analyse", "delta_c": 1.3,
                     "keywords": ["serie", "integrale", "convergence", "fonction"]},
                ],
            }
        }
    elif country_code == "CI":
        levels = {"CYCLE_HS": [{"code": "TERMINALE", "label": "Terminale", "order": 1}]}
        subjects = {"MATH": {"label": "Mathématiques", "coefficient_base": 1.0}}
        chapters = {
            "MATH": {
                "TERMINALE": [
                    {"code": "CH_SUITES", "label": "Suites", "delta_c": 1.0, "keywords": ["suite", "recurrence"]},
                    {"code": "CH_LIMITES", "label": "Limites", "delta_c": 1.0, "keywords": ["limite", "infini"]},
                    {"code": "CH_DERIVATION", "label": "Dérivation", "delta_c": 1.0, "keywords": ["derivee"]},
                    {"code": "CH_INTEGRATION", "label": "Intégration", "delta_c": 1.2, "keywords": ["integrale"]},
                    {"code": "CH_PROBABILITES", "label": "Probabilités", "delta_c": 1.0, "keywords": ["probabilite"]},
                ]
            }
        }
    else:
        levels = {"CYCLE_HS": [{"code": "TERMINALE", "label": "Terminale", "order": 1}]}
        subjects = {"MATH": {"label": "Mathématiques", "coefficient_base": 1.0}}
        chapters = {"MATH": {"TERMINALE": []}}
    
    return {"cycles": cycles, "levels": levels, "subjects": subjects, "chapters": chapters}

def _build_harvest_sources(registry: Dict) -> List[Dict]:
    sources = []
    for src in registry.get("institutional_sources", []):
        sources.append({
            "source_id": src["source_id"],
            "source_name": src["source_name"],
            "base_url": src["base_url"],
            "endpoints": src.get("endpoints", {}),
            "type": src.get("type", "ARCHIVE"),
            "scraping_rules": {
                "year_pattern": r"(20\d{2})",
                "corrige_patterns": ["corrig", "correction", "solution"],
                "exclude_patterns": ["index", "sommaire", "liste", "grille"],
            },
            "pairing_rules": {
                "geographic_zones": ["metro", "metropole", "amerique", "asie", "polynesie", "antilles", "liban"],
                "min_match_score": 0.25,
            },
        })
    return sources

def _build_kernel_params() -> Dict:
    return {
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
        "alignment": {"min_score": 0.18},
        "posable_rules": {
            "require_rqi": True,
            "require_scope": True,
            "min_ari_confidence": 0.55,
        },
        "f1_f2_params": {
            "epsilon": EPSILON,
            "alpha": 1.0,
            "seal_coverage_threshold": 0.95,
        },
        "qc_format": {"prefix": "Comment", "suffix": "?"},
        "triggers_config": {"min_count": 3, "max_count": 7},
        "clustering": {"anti_singleton_min": 2},
    }

def _build_ari_config() -> Dict:
    return {
        "op_patterns": [
            {"op": "OP_PROBABILITY", "pattern": r"\b(probabilit|proba|esperance|variance)\b", "T_j": 0.45},
            {"op": "OP_DERIVE", "pattern": r"\b(deriv|tangente|variation)\b", "T_j": 0.35},
            {"op": "OP_INTEGRATE", "pattern": r"\b(integr|primitive|aire)\b", "T_j": 0.50},
            {"op": "OP_LIMIT", "pattern": r"\b(limit|tend|infini)\b", "T_j": 0.40},
            {"op": "OP_INDUCTION", "pattern": r"\b(recurr|induction|heredite)\b", "T_j": 0.60},
            {"op": "OP_COMPLEX", "pattern": r"\b(complex|imaginaire|module)\b", "T_j": 0.45},
            {"op": "OP_VECTOR", "pattern": r"\b(vecteur|scalaire|orthogonal)\b", "T_j": 0.40},
            {"op": "OP_LOGEXP", "pattern": r"\b(ln|log|exp)\b", "T_j": 0.40},
            {"op": "OP_SOLVE", "pattern": r"\b(equat|resou|racine)\b", "T_j": 0.35},
        ],
        "op_labels": {
            "OP_PROBABILITY": "calculer une probabilité",
            "OP_DERIVE": "dériver une fonction",
            "OP_INTEGRATE": "calculer une intégrale",
            "OP_LIMIT": "calculer une limite",
            "OP_INDUCTION": "démontrer par récurrence",
            "OP_COMPLEX": "manipuler des complexes",
            "OP_VECTOR": "résoudre un problème vectoriel",
            "OP_LOGEXP": "utiliser log et exp",
            "OP_SOLVE": "résoudre une équation",
            "OP_STANDARD": "résoudre un exercice",
        },
    }

def _build_text_processing(language: str) -> Dict:
    if language == "fr":
        return {
            "intent_verbs": ["montrer", "demontrer", "prouver", "calculer", "resoudre", "trouver", "determiner"],
        }
    return {"intent_verbs": []}

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

def cap_levels(cap: Dict, cycle: str = "CYCLE_HS") -> List[Dict]:
    return cap_get(cap, f"education_system.levels.{cycle}", []) or []

# =============================================================================
# HTTP & PDF
# =============================================================================
def http_get(url: str, timeout: int = REQ_TIMEOUT) -> requests.Response:
    res = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    res.raise_for_status()
    return res

def fetch_pdf(url: str) -> bytes:
    cache = st.session_state.get("_http_cache", {})
    if url in cache:
        return cache[url]
    res = http_get(url)
    pdf = res.content
    if len(pdf) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError("PDF trop volumineux")
    cache[url] = pdf
    st.session_state["_http_cache"] = cache
    return pdf

def extract_pdf_text(pdf_bytes: bytes, cap: Dict) -> Tuple[str, Dict]:
    max_pages = cap_get(cap, "kernel_params.atomization.max_pages", 140)
    t0 = time.time()
    pages = []
    method = "none"
    
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages[:max_pages]:
                    try:
                        pages.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
                    except:
                        pages.append("")
            method = "pdfplumber"
        except:
            pass
    
    if not pages or sum(len(p) for p in pages) < 500:
        if PdfReader:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages2 = [p.extract_text() or "" for p in reader.pages[:max_pages]]
                if sum(len(p) for p in pages2) > sum(len(p) for p in pages):
                    pages, method = pages2, "pypdf"
            except:
                pass
    
    if not pages or sum(len(p) for p in pages) < 700:
        if fitz:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                pages3 = [doc.load_page(i).get_text("text") or "" for i in range(min(max_pages, doc.page_count))]
                if sum(len(p) for p in pages3) > sum(len(p) for p in pages):
                    pages, method = pages3, "pymupdf"
            except:
                pass
    
    text = _clean_text(pages)
    return text, {"method": method, "pages": len(pages), "chars": len(text), "seconds": round(time.time() - t0, 3)}

def _clean_text(pages: List[str]) -> str:
    if not pages:
        return ""
    all_lines = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = re.sub(r"(\w)-\n(\w)", r"\1\2", p)
        p = re.sub(r"[ \t]+", " ", p)
        all_lines.extend([ln.strip() for ln in p.split("\n") if ln.strip()])
        all_lines.append("")
    counts = Counter(all_lines)
    skip = {ln for ln, cnt in counts.items() if cnt >= max(2, len(pages) // 3) and len(ln) < 100}
    clean = [ln for ln in all_lines if ln not in skip and not re.fullmatch(r"\d{1,3}", ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(clean)).strip()

# =============================================================================
# HARVEST
# =============================================================================
def harvest_auto(cap: Dict, level: str, volume_max: int = 30) -> Dict:
    set_state(PipelineState.HARVESTING, 15)
    log_pipeline(f"[HARVEST] Démarrage pour {level}")
    
    sources = cap_get(cap, "harvest_sources", [])
    if not sources:
        log_pipeline("[HARVEST] Aucune source configurée", "WARN")
        return {"items_total": 0, "items_with_corrige": 0, "library": [], "errors": ["No sources"]}
    
    src = sources[0]
    base_url = src.get("base_url", "")
    endpoints = src.get("endpoints", {})
    scraping = src.get("scraping_rules", {})
    pairing = src.get("pairing_rules", {})
    
    endpoint = endpoints.get(level)
    if not endpoint:
        log_pipeline(f"[HARVEST] Pas d'endpoint pour {level}", "WARN")
        return {"items_total": 0, "items_with_corrige": 0, "library": [], "errors": [f"No endpoint for {level}"]}
    
    root_url = f"{base_url}{endpoint}"
    log_pipeline(f"[HARVEST] URL: {root_url}")
    
    if not BeautifulSoup:
        log_pipeline("[HARVEST] BeautifulSoup non disponible", "ERROR")
        return {"items_total": 0, "items_with_corrige": 0, "library": [], "errors": ["BeautifulSoup missing"]}
    
    pairs = []
    errors = []
    
    try:
        html = http_get(root_url, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        
        year_pattern = scraping.get("year_pattern", r"(20\d{2})")
        year_links = []
        for a in soup.find_all("a"):
            href = a.get("href", "")
            m = re.search(year_pattern, href)
            if m:
                try:
                    year = int(m.group(1))
                    url = href if href.startswith("http") else f"{base_url}{href}"
                    year_links.append((year, url))
                except:
                    pass
        
        year_links = sorted(set(year_links), key=lambda x: -x[0])[:5]
        log_pipeline(f"[HARVEST] {len(year_links)} années trouvées")
        
        corrige_patterns = scraping.get("corrige_patterns", ["corrig"])
        exclude_patterns = scraping.get("exclude_patterns", [])
        geo_zones = pairing.get("geographic_zones", [])
        min_match = pairing.get("min_match_score", 0.25)
        
        for year, year_url in year_links:
            if len(pairs) >= volume_max:
                break
            try:
                y_html = http_get(year_url, timeout=15).text
                y_soup = BeautifulSoup(y_html, "html.parser")
                
                pdf_links = []
                for a in y_soup.find_all("a"):
                    href = (a.get("href") or "").strip()
                    if not href.lower().endswith(".pdf"):
                        continue
                    abs_url = href if href.startswith("http") else f"{base_url}{href}"
                    name = os.path.basename(abs_url)
                    s = norm_text(f"{abs_url} {name}")
                    
                    if any(ex in s for ex in exclude_patterns):
                        continue
                    
                    is_corrige = any(p in s for p in corrige_patterns)
                    pdf_links.append({"url": abs_url, "name": name, "is_corrige": is_corrige})
                
                sujets = [p for p in pdf_links if not p["is_corrige"]]
                corriges = [p for p in pdf_links if p["is_corrige"]]
                used = set()
                
                for suj in sujets:
                    if len(pairs) >= volume_max:
                        break
                    
                    best_corr, best_score = None, 0.0
                    s_name = norm_text(suj["name"])
                    s_geo = next((z for z in geo_zones if z in s_name), "")
                    
                    for corr in corriges:
                        if corr["url"] in used:
                            continue
                        c_name = norm_text(corr["name"])
                        c_geo = next((z for z in geo_zones if z in c_name), "")
                        
                        score = 0.0
                        if s_geo and c_geo:
                            if s_geo == c_geo:
                                score += 0.6
                            else:
                                continue
                        
                        s_tok = set(re.findall(r"[a-z0-9]{2,}", s_name))
                        c_tok = set(re.findall(r"[a-z0-9]{2,}", c_name))
                        if s_tok and c_tok:
                            score += 0.4 * len(s_tok & c_tok) / max(1, len(s_tok | c_tok))
                        
                        if score > best_score:
                            best_score, best_corr = score, corr
                    
                    corrige = best_corr if best_score >= min_match else None
                    if corrige:
                        used.add(corrige["url"])
                    
                    pair = {
                        "pair_id": f"PAIR_{level}_{year}_{stable_id(suj['name'])}",
                        "year": year,
                        "sujet_name": suj["name"],
                        "sujet_url": suj["url"],
                        "has_corrige": bool(corrige),
                        "corrige_name": corrige["name"] if corrige else "",
                        "corrige_url": corrige["url"] if corrige else "",
                        "match_score": round(best_score, 2),
                    }
                    
                    if not any(p["sujet_url"] == pair["sujet_url"] for p in pairs):
                        pairs.append(pair)
                
                log_pipeline(f"[HARVEST] Année {year}: {len(pairs)} pairs")
            except Exception as e:
                errors.append(f"Year {year}: {str(e)[:50]}")
                log_pipeline(f"[HARVEST] Erreur année {year}: {e}", "WARN")
    
    except Exception as e:
        errors.append(f"Root: {str(e)[:100]}")
        log_pipeline(f"[HARVEST] Erreur accès source: {e}", "ERROR")
    
    manifest = {
        "version": APP_VERSION,
        "timestamp": utc_ts(),
        "source_id": src.get("source_id"),
        "source_name": src.get("source_name"),
        "source_url": root_url,
        "level": level,
        "items_total": len(pairs),
        "items_with_corrige": sum(1 for p in pairs if p["has_corrige"]),
        "library": pairs,
        "errors": errors,
    }
    
    st.session_state.harvest_status = {
        "status": "COMPLETED" if pairs else "NO_DATA",
        "items_total": manifest["items_total"],
        "items_with_corrige": manifest["items_with_corrige"],
        "source": src.get("source_name"),
        "errors": errors,
    }
    
    log_pipeline(f"[HARVEST] Terminé: {manifest['items_total']} pairs, {manifest['items_with_corrige']} avec corrigé")
    return manifest

# =============================================================================
# ATOMIZATION, ARI, TRIGGERS, POSABLE, FRT
# =============================================================================
def atomize_qi(text: str, cap: Dict) -> List[str]:
    markers = cap_get(cap, "kernel_params.atomization.question_markers", [])
    min_len = cap_get(cap, "kernel_params.atomization.min_segment_len", 20)
    max_len = cap_get(cap, "kernel_params.atomization.max_segment_len", 2600)
    intent_verbs = cap_get(cap, "text_processing.intent_verbs", [])
    
    if not text:
        return []
    
    regs = [re.compile(p) for p in markers if p]
    positions = {0}
    for rg in regs:
        for m in rg.finditer(text):
            positions.add(m.start())
    for m in re.finditer(r"\n\n+", text):
        positions.add(m.start())
        positions.add(m.end())
    
    positions = sorted([p for p in positions if 0 <= p <= len(text)])
    
    kept = []
    seen = set()
    for a, b in zip(positions, positions[1:] + [len(text)]):
        seg = re.sub(r"[ \t]+", " ", text[a:b]).strip()
        if not (min_len <= len(seg) <= max_len):
            continue
        key = stable_id(norm_text(seg)[:400])
        if key in seen:
            continue
        seen.add(key)
        s = norm_text(seg)
        if any(v in s for v in intent_verbs) or "?" in seg or bool(re.search(r"[=<>≤≥∫∑]|\d", seg)):
            kept.append(seg)
    
    return kept

def atomize_rqi(text: str, cap: Dict) -> List[str]:
    min_len = cap_get(cap, "kernel_params.atomization.min_segment_len", 20)
    max_len = cap_get(cap, "kernel_params.atomization.max_segment_len", 2600)
    
    if not text:
        return []
    
    regs = [
        re.compile(r"(?m)^\s*(\d{1,2}|[a-h])\s*[\)\.\-:]\s+"),
        re.compile(r"(?im)^\s*(exercice|exo\.?)\s*([0-9ivx]+)\b"),
    ]
    
    positions = {0}
    for rg in regs:
        for m in rg.finditer(text):
            positions.add(m.start())
    for m in re.finditer(r"\n\n+", text):
        positions.add(m.start())
        positions.add(m.end())
    
    positions = sorted([p for p in positions if 0 <= p <= len(text)])
    
    kept = []
    seen = set()
    for a, b in zip(positions, positions[1:] + [len(text)]):
        seg = re.sub(r"[ \t]+", " ", text[a:b]).strip()
        if not (min_len <= len(seg) <= max_len):
            continue
        key = stable_id(norm_text(seg)[:400])
        if key in seen:
            continue
        seen.add(key)
        if bool(re.search(r"[=<>≤≥∫∑]|\d", seg)) or len(seg) > 100:
            kept.append(seg)
    
    return kept

def align_qi_rqi(questions: List[str], answers: List[str], cap: Dict) -> List[Optional[int]]:
    min_score = cap_get(cap, "kernel_params.alignment.min_score", 0.18)
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
            if j in used:
                continue
            if not q_tok[i] or not a_tok[j]:
                continue
            s = len(q_tok[i] & a_tok[j]) / len(q_tok[i] | a_tok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= min_score:
            links[i] = best_j
            used.add(best_j)
    
    return links

def map_chapter(qi_text: str, chapters: List[Dict]) -> Tuple[str, float]:
    if not chapters:
        return "UNMAPPED", 0.0
    qn = norm_text(qi_text)
    best_code, best_score = "UNMAPPED", 0.0
    for ch in chapters:
        kws = ch.get("keywords", [])
        if not kws:
            continue
        matches = sum(1 for kw in kws if kw and kw in qn)
        score = matches / max(6, len(kws))
        if score > best_score:
            best_score, best_code = score, ch.get("code", "UNMAPPED")
    return best_code, round(best_score, 4)

def extract_ari(qi: str, rqi: str, cap: Dict) -> Dict:
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
        except:
            matches = []
        if matches and op not in all_ops:
            evidence = [m.group(0)[:40] for m in matches[:3]]
            ops.append({"op": op, "T_j": T_j, "evidence": evidence, "count": len(matches)})
            all_ops.add(op)
            sum_Tj += T_j
    
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
    triggers = []
    for op_info in ari.get("ops", []):
        triggers.append(f"ARI:{op_info['op']}")
    
    qn = norm_text(qi)
    kws = ["limite", "derivee", "integrale", "probabilite", "suite", "complexe", "vecteur", "recurrence"]
    for kw in kws:
        if kw in qn:
            triggers.append(f"KW:{kw.upper()}")
    
    intent_verbs = cap_get(cap, "text_processing.intent_verbs", [])
    for v in intent_verbs:
        if v in qn:
            triggers.append(f"INTENT:{v.upper()}")
            break
    
    if chapter_code and chapter_code != "UNMAPPED":
        triggers.append(f"SCOPE:{chapter_code}")
    
    return list(dict.fromkeys(triggers))[:7]

def posable_gate(qi: str, rqi: str, chapter_code: str, ari: Dict, cap: Dict) -> Dict:
    rules = cap_get(cap, "kernel_params.posable_rules", {})
    reason_codes = []
    
    has_rqi = bool(rqi and rqi.strip())
    if rules.get("require_rqi", True) and not has_rqi:
        reason_codes.append(RC.RQI_MISSING)
    
    has_scope = chapter_code and chapter_code != "UNMAPPED"
    if rules.get("require_scope", True) and not has_scope:
        reason_codes.append(RC.SCOPE_UNRESOLVED)
    
    if ari.get("confidence", 0) < rules.get("min_ari_confidence", 0.55):
        reason_codes.append(RC.LOW_CONFIDENCE)
    
    return {"is_posable": len(reason_codes) == 0, "reason_codes": reason_codes, "confidence": ari.get("confidence", 0)}

def generate_frt(primary_op: str, triggers: List[str], cap: Dict) -> Dict:
    op_labels = cap_get(cap, "ari_config.op_labels", {})
    label = op_labels.get(primary_op, primary_op.replace("OP_", "").replace("_", " ").lower())
    qc_fmt = cap_get(cap, "kernel_params.qc_format", {})
    
    return {
        "frt_id": f"FRT_{stable_id(primary_op, str(triggers))}",
        "primary_op": primary_op,
        "title": f"{qc_fmt.get('prefix', 'Comment')} {label} {qc_fmt.get('suffix', '?')}",
        "blocks": {
            "usage": f"Reconnaître un problème de type: {label}",
            "reponse_type": "Appliquer la méthode ARI",
            "pieges": "Vérifier les conditions",
            "conclusion": "Conclure clairement",
        },
        "generated_at": utc_ts(),
    }

# =============================================================================
# F1/F2 KERNEL
# =============================================================================
def f1_kernel(delta_c: float, epsilon: float, sum_Tj: float) -> float:
    return float(delta_c) * float((epsilon + sum_Tj) ** 2)

def ia2_judge(qc: Dict, cap: Dict) -> Tuple[bool, List[Dict]]:
    checks = []
    qc_fmt = cap_get(cap, "kernel_params.qc_format", {})
    trig_cfg = cap_get(cap, "kernel_params.triggers_config", {})
    
    posable_ok = qc.get("posable_in_cluster", 0) >= 1
    checks.append({"check_id": "CHK_POSABLE_VALID", "status": "PASS" if posable_ok else "FAIL"})
    
    prefix = qc_fmt.get("prefix", "Comment")
    suffix = qc_fmt.get("suffix", "?")
    form_ok = qc.get("qc_text", "").startswith(prefix) and qc.get("qc_text", "").endswith(suffix)
    checks.append({"check_id": "CHK_QC_FORM", "status": "PASS" if form_ok else "FAIL"})
    
    trig_ok = trig_cfg.get("min_count", 3) <= len(qc.get("triggers", [])) <= trig_cfg.get("max_count", 7)
    checks.append({"check_id": "CHK_TRIGGERS_QUALITY", "status": "PASS" if trig_ok else "FAIL"})
    
    frt = qc.get("frt", {})
    blocks = frt.get("blocks", {})
    frt_ok = all(k in blocks for k in ["usage", "reponse_type", "pieges", "conclusion"])
    checks.append({"check_id": "CHK_FRT_TEMPLATE_OK", "status": "PASS" if frt_ok else "FAIL"})
    
    singleton_ok = qc.get("cluster_size", 0) >= 2 or qc.get("posable_in_cluster", 0) == 0
    checks.append({"check_id": "CHK_ANTI_SINGLETON", "status": "PASS" if singleton_ok else "FAIL"})
    
    return all(c["status"] == "PASS" for c in checks), checks

# =============================================================================
# FULL PIPELINE
# =============================================================================
def run_full_pipeline(cap: Dict) -> Dict:
    st.session_state["_run_ts"] = utc_ts()
    
    levels = cap_levels(cap)
    if not levels:
        raise ValueError("Aucun niveau dans CAP")
    level = levels[0]["code"]  # TERMINALE
    
    # HARVEST
    manifest = harvest_auto(cap, level, volume_max=30)
    st.session_state.harvest_manifest = manifest
    
    if manifest["items_with_corrige"] == 0:
        log_pipeline("[PIPELINE] Aucun corrigé trouvé", "WARN")
        set_state(PipelineState.SAFETY_STOP, 15)
        st.session_state.seal_reason = ["NO_CORRIGE_FOUND"]
        return {"error": "No corrigés", "qi_pack": [], "qc_pack": [], "sealed": False}
    
    # EXTRACTION + PROCESSING
    set_state(PipelineState.EXTRACTING, 25)
    library = [p for p in manifest["library"] if p.get("has_corrige")]
    chapters = cap_chapters(cap, "MATH", level)
    
    quarantine = []
    qi_items = []
    
    for pair in library[:15]:
        pid = pair["pair_id"]
        try:
            sujet_bytes = fetch_pdf(pair["sujet_url"])
            corrige_bytes = fetch_pdf(pair["corrige_url"])
            
            sujet_text, _ = extract_pdf_text(sujet_bytes, cap)
            corrige_text, _ = extract_pdf_text(corrige_bytes, cap)
            
            if len(sujet_text) < 200 or len(corrige_text) < 200:
                quarantine.append({"pair_id": pid, "reason": RC.SUJET_UNREADABLE if len(sujet_text) < 200 else RC.CORRIGE_UNREADABLE})
                continue
            
            set_state(PipelineState.ATOMIZING, 35)
            questions = atomize_qi(sujet_text, cap)
            answers = atomize_rqi(corrige_text, cap)
            
            links = align_qi_rqi(questions, answers, cap)
            
            for i, qi_text in enumerate(questions):
                j = links[i] if i < len(links) else None
                rqi_text = answers[j] if (j is not None and j < len(answers)) else ""
                
                ari = extract_ari(qi_text, rqi_text, cap)
                chapter_code, ch_score = map_chapter(qi_text, chapters)
                triggers = build_triggers(qi_text, ari, chapter_code, cap)
                
                set_state(PipelineState.POSABLE_GATE, 45)
                posable = posable_gate(qi_text, rqi_text, chapter_code, ari, cap)
                
                qi_id = f"QI_{stable_id(pid, str(i), qi_text[:100])}"
                qi_items.append({
                    "qi_id": qi_id,
                    "pair_id": pid,
                    "year": pair.get("year", 0),
                    "qi": qi_text,
                    "rqi": rqi_text,
                    "has_rqi": bool(rqi_text),
                    "chapter_code": chapter_code,
                    "ari": ari,
                    "triggers": triggers,
                    "posable": posable,
                    "qc_id": None,
                })
        except Exception as e:
            log_pipeline(f"[PIPELINE] Erreur {pid}: {e}", "WARN")
            quarantine.append({"pair_id": pid, "reason": str(e)[:50]})
    
    log_pipeline(f"[PIPELINE] {len(qi_items)} Qi, {len(quarantine)} quarantaine")
    
    # CLUSTERING
    set_state(PipelineState.CLUSTERING, 55)
    buckets = defaultdict(list)
    for qi in qi_items:
        primary_op = qi["ari"].get("primary_op", "OP_STANDARD")
        buckets[(qi["chapter_code"], primary_op)].append(qi)
    
    # BUILD QC
    set_state(PipelineState.IA1_PROCESSING, 65)
    qc_pack = []
    mapping = {}
    frt_pack = {}
    
    op_labels = cap_get(cap, "ari_config.op_labels", {})
    qc_fmt = cap_get(cap, "kernel_params.qc_format", {})
    f1_f2 = cap_get(cap, "kernel_params.f1_f2_params", {})
    epsilon = f1_f2.get("epsilon", EPSILON)
    
    for (chapter_code, primary_op), items in sorted(buckets.items()):
        cluster_size = len(items)
        posable_items = [x for x in items if x["has_rqi"] and x["posable"]["is_posable"]]
        posable_count = len(posable_items)
        
        if cluster_size < 2 and posable_count == 0:
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
        
        rep = max(items, key=lambda x: x["ari"].get("confidence", 0))
        frt = generate_frt(primary_op, qc_triggers, cap)
        frt_pack[frt["frt_id"]] = frt
        
        ch_info = next((c for c in chapters if c.get("code") == chapter_code), {})
        delta_c = ch_info.get("delta_c", 1.0)
        sum_Tj = rep["ari"].get("sum_Tj", 0.3)
        f1_raw = f1_kernel(delta_c, epsilon, sum_Tj)
        
        years = [x["year"] for x in items if x.get("year")]
        t_rec = max(1.0, (2026 - max(years)) * 12) if years else 12.0
        
        qc = {
            "qc_id": qc_id,
            "qc_text": qc_text,
            "qc_label": label,
            "chapter_code": chapter_code,
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
            "t_rec": t_rec,
            "ia2_status": "PENDING",
            "ia2_checks": [],
        }
        
        qc_pack.append(qc)
        for item in items:
            mapping[item["qi_id"]] = qc_id
    
    # NORMALIZE Psi_q
    set_state(PipelineState.F1_F2_SCORING, 75)
    by_chapter = defaultdict(list)
    for qc in qc_pack:
        by_chapter[qc["chapter_code"]].append(qc)
    
    for ch_code, qcs in by_chapter.items():
        max_f1 = max((q["f1_raw"] for q in qcs), default=0)
        for qc in qcs:
            qc["Psi_q"] = round(qc["f1_raw"] / max_f1, 6) if max_f1 > 0 else 0.0
    
    # IA2 JUDGE
    set_state(PipelineState.IA2_JUDGE, 80)
    for qc in qc_pack:
        ia2_pass, ia2_checks = ia2_judge(qc, cap)
        qc["ia2_status"] = "PASS" if ia2_pass else "FAIL"
        qc["ia2_checks"] = ia2_checks
    
    # UPDATE QI
    for qi in qi_items:
        if qi["qi_id"] in mapping:
            qi["qc_id"] = mapping[qi["qi_id"]]
    
    # COVERAGE
    set_state(PipelineState.COVERAGE_CHECK, 90)
    qi_posable = [q for q in qi_items if q["has_rqi"] and q["posable"]["is_posable"]]
    N_total = len(qi_posable)
    coverage = round(sum(1 for q in qi_posable if q.get("qc_id")) / max(1, N_total), 4)
    orphans = sum(1 for q in qi_posable if not q.get("qc_id"))
    
    ia2_pass_count = sum(1 for qc in qc_pack if qc["ia2_status"] == "PASS")
    ia2_ok = ia2_pass_count == len(qc_pack) if qc_pack else True
    
    seal_reason = []
    if coverage < f1_f2.get("seal_coverage_threshold", 0.95):
        seal_reason.append(f"COVERAGE_LOW_{coverage:.0%}")
    if orphans > 0:
        seal_reason.append(f"ORPHANS_{orphans}")
    if not ia2_ok:
        seal_reason.append(f"IA2_FAIL_{len(qc_pack) - ia2_pass_count}")
    
    sealed = len(seal_reason) == 0
    
    if sealed:
        set_state(PipelineState.SEALED, 100)
    else:
        set_state(PipelineState.COMPLETED_WITH_DATA, 100)
    
    log_pipeline(f"[PIPELINE] Terminé: Coverage={coverage:.0%} | QC={len(qc_pack)} | Sealed={sealed}")
    
    audit = {
        "pairs_processed": len(library),
        "qi_total": len(qi_items),
        "qi_posable": N_total,
        "orphans": orphans,
        "qc_total": len(qc_pack),
        "qc_ia2_pass": ia2_pass_count,
        "coverage": coverage,
        "sealed": sealed,
        "seal_reason": seal_reason,
    }
    
    evidence_pack = {
        "version": APP_VERSION,
        "kernel_version": KERNEL_VERSION,
        "timestamp": st.session_state["_run_ts"],
        "cap_id": cap_get(cap, "metadata.cap_id"),
        "audit": audit,
    }
    
    st.session_state.qi_pack = qi_items
    st.session_state.qc_pack = qc_pack
    st.session_state.frt_pack = frt_pack
    st.session_state.quarantine = quarantine
    st.session_state.evidence_pack = evidence_pack
    st.session_state.audit_log = audit
    st.session_state.sealed = sealed
    st.session_state.seal_reason = seal_reason
    
    return {"qi_pack": qi_items, "qc_pack": qc_pack, "audit": audit, "sealed": sealed}

# =============================================================================
# ACTIVATE_COUNTRY
# =============================================================================
def activate_country(country_code: str):
    log_pipeline(f"[ACTIVATE] ACTIVATE_COUNTRY({country_code})")
    set_state(PipelineState.ACTIVATING, 0)
    
    st.session_state.country_code = country_code
    st.session_state.pipeline_errors = []
    
    try:
        cap = load_cap(country_code)
        st.session_state.cap = cap
        result = run_full_pipeline(cap)
        return result
    except Exception as e:
        log_pipeline(f"[ACTIVATE] ERREUR: {e}", "ERROR")
        set_state(PipelineState.ERROR)
        raise

# =============================================================================
# STREAMLIT UI - 3 ONGLETS OBLIGATOIRES
# =============================================================================
def main():
    st.set_page_config(page_title=f"SMAXIA GTE {KERNEL_VERSION}", page_icon="🏛️", layout="wide")
    init_state()
    
    # HEADER
    st.markdown(f"# 🏛️ SMAXIA GTE Console {KERNEL_VERSION}")
    st.markdown("**ISO-PROD | Kernel V10.6.3 Scellé | CAS 1 ONLY | Zéro Hardcode**")
    st.markdown("---")
    
    # ============== ACTIVATION (si IDLE) ==============
    if st.session_state.pipeline_state == PipelineState.IDLE:
        st.markdown("## 🚀 ACTIVATION PAYS")
        st.markdown("**Seule action humaine autorisée** (Kernel V10.6.3 §0.2.11)")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            country = st.selectbox("Pays", list(CAP_REGISTRY.keys()), format_func=lambda x: f"{x} - {CAP_REGISTRY[x]['name']}")
            if st.button("🚀 ACTIVATE_COUNTRY", type="primary", use_container_width=True):
                with st.spinner("Pipeline automatique en cours..."):
                    try:
                        activate_country(country)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur: {e}")
        with col2:
            st.info("**Pipeline automatique:** LOAD_CAP → Harvest → Extraction → Atomisation → POSABLE → Clustering → IA1/IA2 → F1/F2 → Seal")
        return
    
    # ============== STATUS BAR ==============
    state = st.session_state.pipeline_state
    progress = st.session_state.pipeline_progress
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("État", state)
    col2.metric("Progression", f"{progress}%")
    col3.metric("Pays", st.session_state.country_code or "N/A")
    col4.metric("SEALED", "✅ OUI" if st.session_state.sealed else "❌ NON")
    
    st.progress(progress / 100)
    
    if st.session_state.pipeline_errors:
        for err in st.session_state.pipeline_errors[-3:]:
            st.error(f"⚠️ {err}")
    
    st.markdown("---")
    
    # ============== 3 ONGLETS OBLIGATOIRES ==============
    tab1, tab2, tab3 = st.tabs(["📦 1. CAP Loader", "📄 2. Sujets & Corrections", "📚 3. Bibliothèque QC"])
    
    # ============== ONGLET 1: CAP LOADER ==============
    with tab1:
        st.markdown("## 📦 CAP Loader + Sources")
        
        cap_status = st.session_state.cap_status
        if cap_status:
            col1, col2, col3 = st.columns(3)
            col1.metric("Status", cap_status.get("status", "N/A"))
            col2.metric("CAP ID", cap_status.get("cap_id", "N/A")[:20] + "...")
            col3.metric("Sources", cap_status.get("sources_count", 0))
            
            st.markdown("### Détails CAP")
            st.json({
                "cap_id": cap_status.get("cap_id"),
                "country": cap_status.get("country"),
                "fingerprint": cap_status.get("fingerprint"),
                "timestamp": cap_status.get("timestamp"),
            })
            
            st.markdown("### Sources Institutionnelles")
            cap = st.session_state.cap
            if cap:
                sources = cap_get(cap, "harvest_sources", [])
                for src in sources:
                    st.markdown(f"- **{src.get('source_name')}** ({src.get('source_id')})")
                    st.caption(f"  URL: {src.get('base_url')}")
                    st.caption(f"  Type: {src.get('type')}")
        else:
            st.warning("⚠️ CAP non chargé - Activez un pays")
    
    # ============== ONGLET 2: SUJETS & CORRECTIONS ==============
    with tab2:
        st.markdown("## 📄 Sujets & Corrections Loader")
        
        harvest_status = st.session_state.harvest_status
        harvest_manifest = st.session_state.harvest_manifest
        
        if harvest_status:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Status", harvest_status.get("status", "N/A"))
            col2.metric("Total", harvest_status.get("items_total", 0))
            col3.metric("Avec Corrigé", harvest_status.get("items_with_corrige", 0))
            col4.metric("Source", harvest_status.get("source", "N/A")[:15])
            
            if harvest_status.get("errors"):
                st.warning(f"⚠️ Erreurs: {len(harvest_status['errors'])}")
                with st.expander("Voir erreurs"):
                    for e in harvest_status["errors"][:10]:
                        st.caption(f"- {e}")
            
            st.markdown("### Bibliothèque Sujets")
            if harvest_manifest and harvest_manifest.get("library"):
                library = harvest_manifest["library"]
                
                # Table
                table_data = []
                for p in library[:50]:
                    table_data.append({
                        "Année": p.get("year", ""),
                        "Sujet": p.get("sujet_name", "")[:40],
                        "Corrigé": "✅" if p.get("has_corrige") else "❌",
                        "Score": p.get("match_score", 0),
                    })
                
                st.dataframe(table_data, use_container_width=True)
                
                # Download si existe
                if library:
                    st.download_button(
                        "📥 Télécharger JSON",
                        data=safe_json(library),
                        file_name="harvest_library.json",
                        mime="application/json"
                    )
            else:
                st.info("Aucun sujet trouvé")
        else:
            st.warning("⚠️ Harvest non exécuté")
        
        # Quarantine
        quarantine = st.session_state.quarantine
        if quarantine:
            st.markdown("### Quarantaine")
            st.warning(f"{len(quarantine)} pairs en quarantaine")
            with st.expander("Voir quarantaine"):
                for q in quarantine[:20]:
                    st.caption(f"- {q.get('pair_id', 'N/A')}: {q.get('reason', 'Unknown')}")
    
    # ============== ONGLET 3: BIBLIOTHÈQUE QC PAR CHAPITRE ==============
    with tab3:
        st.markdown("## 📚 Bibliothèque QC par Chapitre")
        
        qc_pack = st.session_state.qc_pack
        qi_pack = st.session_state.qi_pack
        
        if not qc_pack:
            st.warning("⚠️ Aucune QC générée - Vérifiez le harvest")
        else:
            # SIDEBAR FILTRES (dans cet onglet)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("### Filtres")
                
                # Niveaux disponibles
                levels = list(set(qc.get("chapter_code", "").split("_")[0] if "_" in qc.get("chapter_code", "") else "TERMINALE" for qc in qc_pack))
                # Chapitres
                chapters = sorted(set(qc.get("chapter_code", "") for qc in qc_pack if qc.get("chapter_code")))
                
                selected_chapter = st.selectbox("Chapitre", ["TOUS"] + chapters)
                
                # Stats
                st.markdown("### Stats")
                st.metric("QC Total", len(qc_pack))
                st.metric("Qi Total", len(qi_pack))
                ia2_pass = sum(1 for qc in qc_pack if qc.get("ia2_status") == "PASS")
                st.metric("IA2 PASS", f"{ia2_pass}/{len(qc_pack)}")
            
            with col2:
                # Filtrer QC
                filtered_qc = qc_pack
                if selected_chapter != "TOUS":
                    filtered_qc = [qc for qc in qc_pack if qc.get("chapter_code") == selected_chapter]
                
                # Grouper par chapitre
                by_chapter = defaultdict(list)
                for qc in filtered_qc:
                    by_chapter[qc.get("chapter_code", "UNKNOWN")].append(qc)
                
                # Afficher par chapitre
                for ch_code in sorted(by_chapter.keys()):
                    qcs = by_chapter[ch_code]
                    
                    # Chercher label du chapitre
                    cap = st.session_state.cap
                    ch_label = ch_code
                    if cap:
                        for level_chapters in cap_get(cap, "education_system.chapters.MATH", {}).values():
                            for ch in level_chapters:
                                if ch.get("code") == ch_code:
                                    ch_label = ch.get("label", ch_code)
                                    break
                    
                    st.markdown(f"### 📁 {ch_label} ({len(qcs)} QC)")
                    
                    for qc in sorted(qcs, key=lambda x: -x.get("Psi_q", 0)):
                        ia2_icon = "✅" if qc.get("ia2_status") == "PASS" else "❌"
                        
                        with st.expander(f"{ia2_icon} {qc.get('qc_text', 'N/A')[:60]}..."):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**QC Info**")
                                st.markdown(f"- **ID:** `{qc.get('qc_id', 'N/A')}`")
                                st.markdown(f"- **Texte:** {qc.get('qc_text', 'N/A')}")
                                st.markdown(f"- **État:** {qc.get('qc_state', 'N/A')}")
                                st.markdown(f"- **IA2:** {qc.get('ia2_status', 'N/A')}")
                                st.markdown(f"- **Cluster:** {qc.get('cluster_size', 0)} Qi | Posable: {qc.get('posable_in_cluster', 0)}")
                                st.markdown(f"- **Ψq:** {qc.get('Psi_q', 0):.4f}")
                                
                                st.markdown("**TRIGGERS**")
                                triggers = qc.get("triggers", [])
                                if triggers:
                                    for t in triggers:
                                        st.code(t, language=None)
                                else:
                                    st.caption("Aucun trigger")
                            
                            with col_b:
                                st.markdown("**FRT (Fiche Réponse Type)**")
                                frt = qc.get("frt", {})
                                if frt:
                                    st.markdown(f"- **Titre:** {frt.get('title', 'N/A')}")
                                    blocks = frt.get("blocks", {})
                                    if blocks:
                                        for k, v in blocks.items():
                                            st.markdown(f"- **{k}:** {v}")
                                else:
                                    st.caption("Aucune FRT")
                                
                                st.markdown("**ARI**")
                                st.markdown(f"- **Primary Op:** {qc.get('primary_op', 'N/A')}")
                                st.markdown(f"- **All Ops:** {', '.join(qc.get('all_ops', []))}")
                                st.markdown(f"- **Sum Tj:** {qc.get('sum_Tj', 0):.4f}")
                            
                            # Qi associées
                            st.markdown("**Qi Associées**")
                            qi_ids = qc.get("qi_ids", [])
                            if qi_ids and qi_pack:
                                qi_linked = [qi for qi in qi_pack if qi.get("qi_id") in qi_ids]
                                for qi in qi_linked[:5]:
                                    posable_icon = "✅" if qi.get("posable", {}).get("is_posable") else "❌"
                                    st.markdown(f"- {posable_icon} `{qi.get('qi_id', 'N/A')[:15]}...`")
                                    st.caption(f"  Qi: {qi.get('qi', '')[:100]}...")
                                    if qi.get("rqi"):
                                        st.caption(f"  RQi: {qi.get('rqi', '')[:100]}...")
                                if len(qi_linked) > 5:
                                    st.caption(f"  ... et {len(qi_linked) - 5} autres")
                            else:
                                st.caption("Aucune Qi associée")
                            
                            # IA2 Checks
                            with st.expander("IA2 Checks"):
                                ia2_checks = qc.get("ia2_checks", [])
                                if ia2_checks:
                                    for check in ia2_checks:
                                        icon = "✅" if check.get("status") == "PASS" else "❌"
                                        st.markdown(f"{icon} {check.get('check_id', 'N/A')}: {check.get('status', 'N/A')}")
                                else:
                                    st.caption("Aucun check")
    
    # ============== FOOTER ==============
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.expander("📋 Pipeline Logs"):
            logs = st.session_state.pipeline_log[-30:]
            for entry in reversed(logs):
                level = entry.get("level", "INFO")
                color = "red" if level == "ERROR" else "orange" if level == "WARN" else "gray"
                st.markdown(f"<span style='color:{color};font-size:12px'>[{entry['ts']}] [{level}] {entry['msg']}</span>", unsafe_allow_html=True)
    
    with col2:
        with st.expander("📊 Audit"):
            audit = st.session_state.audit_log
            if audit:
                st.json(audit)
            else:
                st.caption("Aucun audit disponible")
    
    if st.button("🔄 Réinitialiser", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# =============================================================================
# ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    main()
