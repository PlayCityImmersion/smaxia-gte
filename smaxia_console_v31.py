# =============================================================================
# SMAXIA GTE Console V31.10.18 — ISO-PROD (ZÉRO HARDCODE RÉEL)
# =============================================================================
# FUSION:
# - V31.10.17: Corrections saturation, clustering stable par primary_op
# - Manus V32: Architecture Pack-Driven, F1/F2 conformes
# - GPT Audit: Externalisation complète des hardcodes
#
# PRINCIPE ZÉRO HARDCODE:
# - AUCUNE donnée métier (chapitres, keywords, patterns) dans le code
# - TOUT est piloté par le Country Academic Pack (CAP)
# - Le Pack est OBLIGATOIRE (upload JSON ou fichier externe)
# - Mode Genesis = Pack JSON minimal généré, PAS de données hardcodées
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import glob
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
# KERNEL CONSTANTS (INVARIANTS TECHNIQUES - PAS DE DONNÉES MÉTIER)
# =============================================================================
APP_VERSION = "V31.10.18"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

# Constantes mathématiques F1/F2 (invariants selon Kernel V10.6.1)
F1_EPSILON = 1e-6
F2_ALPHA = 1.0
F2_TREC = 1.0


# =============================================================================
# SESSION & LOGS
# =============================================================================
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ss_init():
    """Initialise l'état de session Streamlit - AUCUNE valeur métier par défaut."""
    st.session_state.setdefault("pack_active", None)
    st.session_state.setdefault("pack_id", None)
    st.session_state.setdefault("pack_sig_sha256", None)
    st.session_state.setdefault("country", None)  # Pas de défaut FR
    st.session_state.setdefault("level", None)     # Pas de défaut TERMINALE
    st.session_state.setdefault("subject", None)   # Pas de défaut MATH
    st.session_state.setdefault("library", [])
    st.session_state.setdefault("harvest_manifest", None)
    st.session_state.setdefault("qi_pack", None)
    st.session_state.setdefault("qc_pack", None)
    st.session_state.setdefault("chapter_report", None)
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
# TEXT UTILS (KERNEL INVARIANT - AUCUNE DONNÉE MÉTIER)
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


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# =============================================================================
# COUNTRY ACADEMIC PACK (CAP) - TOUT VIENT DU PACK
# =============================================================================
def validate_pack_structure(pack: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valide la structure minimale d'un Pack."""
    errors = []
    
    # Champs obligatoires
    required = ["pack_id", "country_code", "harvest_sources", "chapter_taxonomy"]
    for field in required:
        if field not in pack:
            errors.append(f"Champ obligatoire manquant: {field}")
    
    # Validation harvest_sources
    sources = pack.get("harvest_sources", [])
    if not sources:
        errors.append("harvest_sources vide ou manquant")
    else:
        for i, src in enumerate(sources):
            if "base_url" not in src:
                errors.append(f"harvest_sources[{i}]: base_url manquant")
            if "levels" not in src:
                errors.append(f"harvest_sources[{i}]: levels manquant")
    
    # Validation chapter_taxonomy
    taxonomy = pack.get("chapter_taxonomy", {})
    if not taxonomy:
        errors.append("chapter_taxonomy vide")
    
    # Validation text_processing (optionnel mais recommandé)
    text_proc = pack.get("text_processing", {})
    if not text_proc.get("common_words"):
        log("[PACK] Warning: text_processing.common_words absent")
    if not text_proc.get("intent_verbs"):
        log("[PACK] Warning: text_processing.intent_verbs absent")
    
    # Validation ari_config (obligatoire pour ARI)
    if "ari_config" not in pack:
        errors.append("ari_config manquant (requis pour extraction ARI)")
    else:
        ari = pack["ari_config"]
        if not ari.get("op_patterns"):
            errors.append("ari_config.op_patterns manquant")
        if not ari.get("primary_ops_order"):
            errors.append("ari_config.primary_ops_order manquant")
    
    return len(errors) == 0, errors


def load_pack_from_file(path: str) -> Dict[str, Any]:
    """Charge un Pack depuis un fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    pack["_source"] = "FILE"
    pack["_file_path"] = os.path.abspath(path)
    pack["_pack_sig_sha256"] = sha256_bytes(json.dumps(pack, sort_keys=True).encode())
    return pack


def load_pack_from_upload(uploaded_bytes: bytes) -> Dict[str, Any]:
    """Charge un Pack depuis un upload."""
    pack = json.loads(uploaded_bytes.decode("utf-8"))
    pack["_source"] = "UPLOAD"
    pack["_pack_sig_sha256"] = sha256_bytes(json.dumps(pack, sort_keys=True).encode())
    return pack


def find_pack_files(search_paths: List[str] = None) -> List[str]:
    """Recherche des fichiers Pack JSON."""
    if search_paths is None:
        search_paths = [
            os.getcwd(),
            os.path.join(os.getcwd(), "packs"),
            os.path.join(os.getcwd(), "academic_packs"),
            "/tmp",
        ]
    
    found = []
    for base in search_paths:
        if os.path.isdir(base):
            for f in glob.glob(os.path.join(base, "CAP_*.json")):
                if os.path.isfile(f):
                    found.append(f)
    return sorted(set(found))


def activate_pack(pack: Dict[str, Any]) -> bool:
    """Active un Pack après validation."""
    valid, errors = validate_pack_structure(pack)
    
    if not valid:
        for e in errors:
            log(f"[PACK] ERREUR: {e}")
        return False
    
    st.session_state.pack_active = pack
    st.session_state.pack_id = pack.get("pack_id", "UNKNOWN")
    st.session_state.pack_sig_sha256 = pack.get("_pack_sig_sha256", "")
    st.session_state.country = pack.get("country_code", "")
    
    # Extraire les niveaux et matières disponibles
    taxonomy = pack.get("chapter_taxonomy", {})
    subjects = list(taxonomy.keys())
    if subjects:
        st.session_state.subject = subjects[0]
        levels = list(taxonomy[subjects[0]].keys())
        if levels:
            st.session_state.level = levels[0]
    
    log(f"[PACK] Activé: {st.session_state.pack_id} | Source: {pack.get('_source')}")
    return True


# =============================================================================
# PACK ACCESSORS (TOUT VIENT DU PACK)
# =============================================================================
def pack_get_chapters(pack: Dict[str, Any], subject: str, level: str) -> List[Dict[str, Any]]:
    """Récupère les chapitres depuis le Pack."""
    taxonomy = pack.get("chapter_taxonomy", {})
    return taxonomy.get(subject, {}).get(level, [])


def pack_get_harvest_source(pack: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Récupère la première source de harvest."""
    sources = pack.get("harvest_sources", [])
    return sources[0] if sources else None


def pack_get_text_processing(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Récupère la config de traitement texte."""
    return pack.get("text_processing", {})


def pack_get_ari_config(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Récupère la config ARI (patterns, ordre des ops)."""
    return pack.get("ari_config", {})


def pack_get_f1f2_params(pack: Dict[str, Any]) -> Dict[str, float]:
    """Récupère les paramètres F1/F2 (avec fallback sur constantes Kernel)."""
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
    """Réparation des espaces manquants, pilotée par le Pack."""
    if not text:
        return ""
    
    # Appliquer les patterns depuis le Pack
    for item in text_proc.get("glued_patterns", []):
        try:
            pattern = item.get("pattern", "")
            replacement = item.get("replacement", "")
            if pattern:
                text = re.sub(pattern, replacement, text)
        except re.error:
            pass
    
    # Heuristique avec dictionnaire de mots du Pack
    common_words = set(text_proc.get("common_words", []))
    if not common_words:
        return re.sub(r"\s+", " ", text).strip()
    
    def split_glued_word(match):
        word = match.group(0)
        if len(word) < 8:
            return word
        
        result = []
        i = 0
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
        if " " in reconstructed and len(reconstructed) > len(word):
            return reconstructed
        return word
    
    text = re.sub(r"[a-zA-ZéèêëàâäùûüôöîïçÉÈÊËÀÂÄÙÛÜÔÖÎÏÇ]{15,}", split_glued_word, text)
    return re.sub(r"\s+", " ", text).strip()


def _dehyphenate(text: str) -> str:
    """Répare les mots coupés en fin de ligne."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
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
                    text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    pages.append(text)
                except Exception:
                    pages.append("")
    except Exception:
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
                text = page.extract_text() or ""
                pages.append(text)
            except Exception:
                pages.append("")
    except Exception:
        pass
    return pages


def clean_pdf_text(pages: List[str], text_proc: Dict[str, Any]) -> str:
    """Nettoie le texte PDF avec config du Pack."""
    if not pages:
        return ""
    
    page_lines = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = _dehyphenate(p)
        p = _fix_missing_spaces(p, text_proc)
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        page_lines.append(lines)
    
    # Détection headers/footers répétés
    top_counts, bot_counts = {}, {}
    n_pages = len(page_lines)
    
    for lines in page_lines:
        if lines:
            top_counts[lines[0]] = top_counts.get(lines[0], 0) + 1
        if len(lines) > 1:
            bot_counts[lines[-1]] = bot_counts.get(lines[-1], 0) + 1
    
    threshold = max(2, n_pages // 3)
    skip_top = {k for k, v in top_counts.items() if v >= threshold}
    skip_bot = {k for k, v in bot_counts.items() if v >= threshold}
    
    out_lines = []
    for lines in page_lines:
        for i, ln in enumerate(lines):
            if i == 0 and ln in skip_top:
                continue
            if i == len(lines) - 1 and ln in skip_bot:
                continue
            # Numéros de page (pattern universel)
            if re.fullmatch(r"\d{1,3}", ln.strip()):
                continue
            out_lines.append(ln)
        out_lines.append("")
    
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


def extract_text_from_pdf(pdf_bytes: bytes, text_proc: Dict[str, Any]) -> str:
    """Extrait le texte d'un PDF."""
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
    """Télécharge un PDF avec cache."""
    cache = st.session_state.get("_http_pdf_cache", {})
    if url in cache:
        return cache[url]
    
    res = _http_get(url)
    pdf_bytes = res.content
    
    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError(f"PDF trop volumineux: {len(pdf_bytes) / (1024*1024):.1f} MB")
    
    cache[url] = pdf_bytes
    st.session_state["_http_pdf_cache"] = cache
    return pdf_bytes


# =============================================================================
# HARVEST (100% PACK-DRIVEN)
# =============================================================================
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé")
    return BeautifulSoup(html, "html.parser")


def _abs_url(base: str, href: str) -> str:
    return href if href.startswith("http") else requests.compat.urljoin(base, href)


def harvest_from_pack(pack: Dict[str, Any], level: str, subject: str, 
                      years_back: int, volume_max: int) -> Dict[str, Any]:
    """
    Harvest 100% piloté par le Pack.
    Tous les patterns, URLs, zones géo viennent du Pack.
    """
    source = pack_get_harvest_source(pack)
    if not source:
        raise ValueError("Aucune source de harvest dans le Pack")
    
    base_url = source.get("base_url", "")
    level_path = source.get("levels", {}).get(level)
    if not level_path:
        raise ValueError(f"Niveau '{level}' non configuré dans le Pack")
    
    root = f"{base_url}{level_path}"
    log(f"[HARVEST] root={root}")
    
    # Tous les patterns viennent du Pack
    year_pattern = source.get("year_pattern", r"(20\d{2})")
    meta_exclude = source.get("meta_exclude", [])
    corrige_patterns = source.get("pdf_patterns", {}).get("corrige", [])
    geo_zones = source.get("geographic_zones", [])
    
    html = _http_get(root).text
    sp = _soup(html)
    
    # Trouver les liens années
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
    log(f"[HARVEST] années={len(selected)} (min={min_year})")
    
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
                
                # Exclure meta PDFs selon le Pack
                s = norm_text(absu + " " + label)
                if any(w in s for w in meta_exclude):
                    continue
                
                # Détecter corrigé selon patterns du Pack
                is_corrige = any(p in s for p in corrige_patterns)
                
                pdf_links.append({
                    "url": absu,
                    "name": os.path.basename(absu),
                    "label": label,
                    "is_corrige": is_corrige,
                })
            
            if not pdf_links:
                continue
            
            sujets = [p for p in pdf_links if not p["is_corrige"]]
            corriges = [p for p in pdf_links if p["is_corrige"]]
            used_corr = set()
            
            for suj in sujets:
                if len(pairs) >= volume_max:
                    break
                
                # Matching sujet-corrigé
                best = (None, 0.0)
                for corr in corriges:
                    if corr["url"] in used_corr:
                        continue
                    
                    # Score basé sur zones géo du Pack
                    s_name = norm_text(suj["name"])
                    c_name = norm_text(corr["name"])
                    
                    score = 0.0
                    s_geo = next((z for z in geo_zones if z in s_name), "")
                    c_geo = next((z for z in geo_zones if z in c_name), "")
                    
                    if s_geo and c_geo:
                        if s_geo == c_geo:
                            score += 0.5
                        else:
                            continue  # Zones différentes = pas de match
                    
                    # Jaccard sur noms
                    s_tok = set(tokenize(suj["name"]))
                    c_tok = set(tokenize(corr["name"]))
                    if s_tok and c_tok:
                        score += 0.3 * len(s_tok & c_tok) / len(s_tok | c_tok)
                    
                    if score > best[1]:
                        best = (corr, score)
                
                corrige = best[0] if best[1] >= 0.3 else None
                if corrige:
                    used_corr.add(corrige["url"])
                
                pair_id = f"PAIR_{level}_{subject}_{y}_{stable_id(suj['name'])}"
                item = {
                    "pair_id": pair_id,
                    "scope": f"{level}|{subject}",
                    "source": f"{source.get('source_id', 'SRC')} {y}",
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
            
            log(f"[HARVEST] year={y} pdfs={len(pdf_links)} pairs={len([p for p in pairs if p['year']==y])}")
        
        except Exception as e:
            log(f"[HARVEST] year={y} ERREUR: {e}")
    
    return {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": pack.get("pack_id"),
        "country": pack.get("country_code"),
        "level": level,
        "subject": subject,
        "items_total": len(pairs),
        "items_corrige_ok": corrige_ok,
        "library": pairs,
    }


# =============================================================================
# Qi/RQi EXTRACTION (PACK-DRIVEN)
# =============================================================================
# Patterns de segmentation (universels, pas spécifiques à une langue)
_EXERCICE_RE = re.compile(r"(?i)(?:^|\n)\s*(?:EXERCICE|Exercise|Ex\.?)\s*(\d+|[IVX]+)", re.MULTILINE)
_PARTIE_RE = re.compile(r"(?i)(?:^|\n)\s*(?:PARTIE|Part)\s*([A-Z]|\d+)", re.MULTILINE)
_QUESTION_RE = re.compile(r"(?m)^\s*(\d{1,2}|[a-h]|[ivx]{1,4})\s*[\)\.\-:]\s+")
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦±×÷]")


def _build_intent_re(intent_verbs: List[str]) -> Optional[re.Pattern]:
    """Construit la regex d'intention depuis le Pack."""
    if not intent_verbs:
        return None
    pattern = r"\b(" + "|".join(re.escape(v) for v in intent_verbs) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def _chunk_candidates(text: str) -> List[str]:
    """Segmentation en chunks candidats."""
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
        c = t[a:b].strip()
        c = re.sub(r"\s+", " ", c).strip()
        if c and len(c) > 20:
            chunks.append(c)
    
    return chunks


def _looks_like_question(s: str, intent_re: Optional[re.Pattern]) -> bool:
    """Détermine si un chunk ressemble à une question."""
    if not s or len(s.strip()) < 25:
        return False
    t = s.strip()
    
    has_intent = bool(intent_re and intent_re.search(t)) or ("?" in t)
    has_math = bool(_MATH_SYMBOL_RE.search(t)) or bool(re.search(r"\b\d+\b", t))
    
    return bool(has_intent or (has_math and len(t) > 40))


def split_questions(text: str, text_proc: Dict[str, Any], max_items: int = 200) -> Tuple[List[str], Dict[str, Any]]:
    """Segmente le texte en questions."""
    intent_verbs = text_proc.get("intent_verbs", [])
    intent_re = _build_intent_re(intent_verbs)
    
    raw_chunks = _chunk_candidates(text)
    
    keep = []
    reject = []
    seen = set()
    
    for c in raw_chunks:
        key = stable_id(norm_text(c)[:400])
        if key in seen:
            continue
        seen.add(key)
        
        if len(c) > 2500:
            c = c[:2500] + "…"
        
        if _looks_like_question(c, intent_re):
            keep.append(c)
        else:
            reject.append(c)
        
        if len(keep) >= max_items:
            break
    
    audit = {
        "raw_chunks": len(raw_chunks),
        "kept": len(keep),
        "rejected": len(reject),
        "reject_ratio": round((len(reject) / max(1, len(raw_chunks))) * 100, 2),
    }
    return keep, audit


# =============================================================================
# ALIGN Qi↔RQi (KERNEL)
# =============================================================================
def align_qi_rqi(questions: List[str], responses: List[str]) -> Tuple[List[Optional[int]], List[Dict[str, Any]]]:
    """Aligne Qi avec RQi par similarité Jaccard."""
    link: List[Optional[int]] = [None] * len(questions)
    audits: List[Dict[str, Any]] = []
    
    if not questions or not responses:
        return link, audits
    
    q_tokens = [tokenize(q) for q in questions]
    r_tokens = [tokenize(r) for r in responses]
    used_r = set()
    
    # Pass 1: fenêtre locale
    for i, q_tok in enumerate(q_tokens):
        best_j, best_s = None, 0.0
        lo, hi = max(0, i - 5), min(len(responses), i + 6)
        
        for j in range(lo, hi):
            if j in used_r:
                continue
            s = jaccard(q_tok, r_tokens[j])
            if s > best_s:
                best_s, best_j = s, j
        
        if best_j is not None and best_s >= 0.06:
            link[i] = best_j
            used_r.add(best_j)
            audits.append({"qi_k": i+1, "mode": "LOCAL", "score": round(best_s, 3), "rqi_k": best_j+1})
        else:
            audits.append({"qi_k": i+1, "mode": "PENDING", "score": round(best_s, 3), "rqi_k": None})
    
    # Pass 2: global
    for i in range(len(questions)):
        if link[i] is not None:
            continue
        
        best_j, best_s = None, 0.0
        for j in range(len(responses)):
            if j in used_r:
                continue
            s = jaccard(q_tokens[i], r_tokens[j])
            if s > best_s:
                best_s, best_j = s, j
        
        if best_j is not None and best_s >= 0.08:
            link[i] = best_j
            used_r.add(best_j)
            audits[i] = {"qi_k": i+1, "mode": "GLOBAL", "score": round(best_s, 3), "rqi_k": best_j+1}
        else:
            audits[i] = {"qi_k": i+1, "mode": "NONE", "score": round(best_s, 3), "rqi_k": None}
    
    return link, audits


# =============================================================================
# ARI EXTRACTION (100% PACK-DRIVEN)
# =============================================================================
def extract_op_trace(question: str, response: str, ari_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extrait la trace ARI depuis le Pack.
    Les patterns viennent ENTIÈREMENT du Pack.
    """
    op_patterns = ari_config.get("op_patterns", [])
    if not op_patterns:
        return [{"op": "OP_STANDARD", "confidence": 0.5}]
    
    combined = norm_text(f"{question} {response}")
    ops = []
    seen = set()
    
    for item in op_patterns:
        pattern = item.get("pattern", "")
        op_code = item.get("op", "")
        if not pattern or not op_code:
            continue
        
        try:
            if re.search(pattern, combined, re.IGNORECASE):
                if op_code not in seen:
                    ops.append({"op": op_code, "confidence": 0.8})
                    seen.add(op_code)
        except re.error:
            pass
    
    if not ops:
        ops.append({"op": "OP_STANDARD", "confidence": 0.5})
    
    return ops


def get_primary_op(op_trace: List[Dict[str, Any]], ari_config: Dict[str, Any]) -> str:
    """
    Détermine l'opérateur principal selon l'ordre du Pack.
    """
    primary_order = ari_config.get("primary_ops_order", [])
    if not primary_order:
        return op_trace[0]["op"] if op_trace else "OP_STANDARD"
    
    ops_in_trace = {x["op"] for x in op_trace}
    for op in primary_order:
        if op in ops_in_trace:
            return op
    
    return op_trace[0]["op"] if op_trace else "OP_STANDARD"


def normalize_ari_steps(op_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise les étapes ARI."""
    return [{"op": x["op"], "step": i + 1} for i, x in enumerate(op_trace)]


# =============================================================================
# TRIGGERS (PACK-DRIVEN)
# =============================================================================
def build_triggers(question: str, op_trace: List[Dict[str, Any]], text_proc: Dict[str, Any]) -> List[str]:
    """
    Construit les triggers - keywords viennent du Pack.
    """
    triggers = []
    
    # Triggers basés sur les opérateurs ARI
    for op_item in op_trace:
        triggers.append(f"ARI:{op_item['op']}")
    
    # Triggers basés sur les keywords du Pack
    trigger_keywords = text_proc.get("trigger_keywords", [])
    if trigger_keywords:
        q_norm = norm_text(question)
        for kw in trigger_keywords:
            if kw in q_norm:
                triggers.append(f"KW:{kw.upper()}")
    
    return list(set(triggers))[:15]


# =============================================================================
# CHAPTER MAPPING (PACK-DRIVEN)
# =============================================================================
def map_qi_to_chapter(question: str, chapters: List[Dict[str, Any]]) -> str:
    """Mappe une Qi à un chapitre selon les keywords du Pack."""
    if not chapters:
        return "UNMAPPED"
    
    q_norm = norm_text(question)
    best_chapter = "UNMAPPED"
    best_score = 0
    
    for ch in chapters:
        keywords = ch.get("keywords", [])
        score = sum(1 for kw in keywords if kw in q_norm)
        if score > best_score:
            best_score = score
            best_chapter = ch.get("code", "UNMAPPED")
    
    return best_chapter if best_score > 0 else "UNMAPPED"


# =============================================================================
# QC GENERATION (KERNEL - STABLE CLUSTERING)
# =============================================================================
def qc_method_label(primary_op: str, secondary_ops: List[str], ari_config: Dict[str, Any]) -> str:
    """Génère le libellé QC depuis le Pack."""
    op_labels = ari_config.get("op_labels", {})
    
    main_label = op_labels.get(primary_op, primary_op.replace("OP_", "").title())
    
    if secondary_ops:
        sec_labels = [op_labels.get(op, op.replace("OP_", "")) for op in secondary_ops[:2]]
        return f"{main_label} avec {', '.join(sec_labels)}"
    
    return main_label


def sigma_similarity(qc_a: Dict[str, Any], qc_b: Dict[str, Any]) -> float:
    """Similarité entre deux QC (Jaccard sur ops)."""
    a_ops = set(qc_a.get("all_ops", []))
    b_ops = set(qc_b.get("all_ops", []))
    if a_ops and b_ops:
        return len(a_ops & b_ops) / len(a_ops | b_ops)
    return 0.0


def build_qc_from_qi(qi_pack: List[Dict[str, Any]], chapters: List[Dict[str, Any]], 
                     ari_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Construit les QC - clustering stable par (chapter, primary_op).
    """
    # Bucketing
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    
    for qi in qi_pack:
        cc = qi.get("chapter_code", "UNMAPPED")
        primary_op = qi.get("primary_op", "OP_STANDARD")
        key = (cc, primary_op)
        buckets.setdefault(key, []).append(qi)
    
    log(f"[QC] Buckets: {len(buckets)}")
    
    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}
    
    # Compter buckets par chapitre
    chapter_bucket_counts: Dict[str, int] = {}
    for (cc, _) in buckets.keys():
        chapter_bucket_counts[cc] = chapter_bucket_counts.get(cc, 0) + 1
    
    for (cc, primary_op), items in sorted(buckets.items()):
        posable = [x for x in items if x.get("has_rqi")]
        cluster_size = len(items)
        posable_count = len(posable)
        
        # Créer QC si: cluster >= 2 OU unique dans chapitre OU >= 1 posable
        is_singleton = cluster_size < 2
        is_only_in_chapter = chapter_bucket_counts.get(cc, 0) <= 1
        
        if is_singleton and not is_only_in_chapter and posable_count == 0:
            continue
        
        # Collecter ops secondaires
        all_ops_set = set()
        for it in items:
            for op in it.get("ari_ops", []):
                all_ops_set.add(op)
        secondary_ops = [op for op in all_ops_set if op != primary_op]
        
        qc_text = qc_method_label(primary_op, secondary_ops, ari_config)
        qc_id = f"QC_{stable_id(cc, primary_op)}"
        
        # État QC
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
        trig_sorted = [k for (k, _) in sorted(trig_counts.items(), key=lambda x: (-x[1], x[0]))][:10]
        
        qc_obj = {
            "qc_id": qc_id,
            "qc": qc_text,
            "chapter_code": cc,
            "primary_op": primary_op,
            "all_ops": list(all_ops_set),
            "cluster_size": cluster_size,
            "posable_in_cluster": posable_count,
            "qc_state": qc_state,
            "qi_ids": [x["qi_id"] for x in items],
            "triggers_hint": trig_sorted,
        }
        qc_pack.append(qc_obj)
        
        for it in items:
            qc_map[it["qi_id"]] = qc_id
    
    log(f"[QC] Créées: {len(qc_pack)} | Qi mappées: {len(qc_map)}")
    
    return qc_pack, qc_map


# =============================================================================
# F1/F2 FORMULAS (KERNEL CONFORMANT)
# =============================================================================
def compute_trigger_weights(triggers: List[str]) -> Dict[str, float]:
    """Calcule les poids des triggers."""
    uniq = list(dict.fromkeys(triggers or []))
    n = len(uniq)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in uniq}


def f1_raw(delta_c: float, epsilon: float, Tj: Dict[str, float], m_q: int) -> float:
    """
    Formule F1: ψ_q = δ_c × (ε + Σ w_i)²
    """
    if not Tj:
        s = 0.0
    else:
        items = sorted(Tj.items(), key=lambda x: x[0])
        top = items[:max(0, int(m_q))]
        s = sum(v for _, v in top)
    return float(delta_c) * float((epsilon + s) ** 2)


def f1_normalize_in_chapter(qc_list: List[Dict[str, Any]]) -> None:
    """Normalise F1 par chapitre: Ψ_q = ψ_q / max(ψ_q)"""
    if not qc_list:
        return
    raws = [float(q.get("psi_raw", 0.0) or 0.0) for q in qc_list]
    mx = max(raws) if raws else 0.0
    for q in qc_list:
        psi = float(q.get("psi_raw", 0.0) or 0.0)
        Psi = psi / mx if mx > 0 else 0.0
        q["Psi_q"] = float(Psi)


def f2_score(qc: Dict[str, Any], selected: List[Dict[str, Any]], 
             n_q_historical: int, N_total: int, alpha: float, t_rec: float) -> Tuple[float, Dict[str, Any]]:
    """
    Formule F2: Score(q) = (n_q,hist/N_total) × (1 + α/t_rec) × Ψ_q × Π((1-σ)×100)
    """
    Psi_q = float(qc.get("Psi_q", 0.0) or 0.0)
    base = (float(n_q_historical) / float(max(1, N_total))) * (1.0 + float(alpha) / float(max(1e-9, t_rec))) * Psi_q
    
    prod = 1.0
    sigmas = []
    for p in selected:
        s = sigma_similarity(qc, p)
        sigmas.append(s)
        prod *= (1.0 - s) * 100.0
    
    score = base * prod
    audit = {"n_q_historical": n_q_historical, "N_total": N_total, "Psi_q": Psi_q, "sigmas": sigmas[:10]}
    return float(score), audit


def progressive_select(qc_list: List[Dict[str, Any]], N_total: int, 
                       f1f2_params: Dict[str, float], top_k: int = 12) -> List[Dict[str, Any]]:
    """Sélection progressive F2."""
    selected: List[Dict[str, Any]] = []
    remaining = [q for q in qc_list if q.get("qc_state") in ("POSABLE", "POSABLE_WEAK")]
    remaining = sorted(remaining, key=lambda x: x.get("qc_id", ""))
    
    alpha = f1f2_params.get("alpha", F2_ALPHA)
    t_rec = f1f2_params.get("t_rec", F2_TREC)
    
    for _ in range(min(top_k, len(remaining))):
        best = None
        for qc in remaining:
            n_q_hist = int(qc.get("n_q_historical", qc.get("cluster_size", 1)) or 0)
            sc, audit = f2_score(qc, selected, n_q_hist, N_total, alpha, t_rec)
            qc["_f2_score"] = sc
            qc["_f2_audit"] = audit
            if best is None or sc > best["_f2_score"]:
                best = qc
        
        if best is None:
            break
        selected.append(best)
        remaining = [x for x in remaining if x.get("qc_id") != best.get("qc_id")]
    
    return selected


# =============================================================================
# SANITY GATE
# =============================================================================
def sanity_eval(doc_audits: List[Dict[str, Any]], qi_items: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """Évalue la qualité des données."""
    total_raw = sum(int(d.get("raw_chunks", 0) or 0) for d in doc_audits)
    total_kept = sum(int(d.get("kept", 0) or 0) for d in doc_audits)
    total_rej = sum(int(d.get("rejected", 0) or 0) for d in doc_audits)
    
    # Détection duplications
    seen = set()
    dups = 0
    for q in qi_items:
        k = stable_id(norm_text(q.get("qi", ""))[:300])
        if k in seen:
            dups += 1
        else:
            seen.add(k)
    
    dup_ratio = dups / max(1, len(qi_items))
    rej_ratio = total_rej / max(1, total_raw) if total_raw else 0.0
    
    ok = True
    reasons = []
    
    if rej_ratio > 0.85 and total_raw > 100:
        ok = False
        reasons.append(f"reject_ratio_extreme({round(rej_ratio*100,2)}%)")
    if dup_ratio > 0.35:
        ok = False
        reasons.append(f"dup_ratio_high({round(dup_ratio*100,2)}%)")
    
    audit = {
        "total_raw_chunks": total_raw,
        "total_kept": total_kept,
        "total_rejected": total_rej,
        "reject_ratio": round(rej_ratio * 100, 2),
        "dup_ratio": round(dup_ratio * 100, 2),
        "reasons": reasons,
    }
    return ok, audit


# =============================================================================
# RUN PIPELINE
# =============================================================================
def run_phase(library: List[Dict[str, Any]], volume_pairs: int, pack: Dict[str, Any]) -> Dict[str, Any]:
    """Exécute une phase du pipeline."""
    text_proc = pack_get_text_processing(pack)
    ari_config = pack_get_ari_config(pack)
    
    subject = st.session_state.get("subject", "")
    level = st.session_state.get("level", "")
    chapters = pack_get_chapters(pack, subject, level)
    
    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    to_process = exploitable[:max(1, min(int(volume_pairs), len(exploitable)))]
    log(f"[RUN] Processing {len(to_process)} pairs")
    
    qi_items: List[Dict[str, Any]] = []
    doc_audits: List[Dict[str, Any]] = []
    
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
            log(f"[PDF] VIDE {pid}")
            continue
        
        qs, qs_audit = split_questions(su_text, text_proc, max_items=150)
        rs, rs_audit = split_questions(co_text, text_proc, max_items=200)
        
        doc_audits.append({"pair_id": pid, "role": "SUJET", **qs_audit})
        doc_audits.append({"pair_id": pid, "role": "CORRIGE", **rs_audit})
        
        link, _ = align_qi_rqi(qs, rs)
        matched = sum(1 for x in link if x is not None)
        log(f"[ALIGN] {pid}: {len(qs)} Qi, {len(rs)} RQi, matched={matched}")
        
        for i, q in enumerate(qs):
            if not q.strip():
                continue
            j = link[i] if i < len(link) else None
            r = rs[j] if (j is not None and 0 <= j < len(rs)) else ""
            
            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
            
            op_trace = extract_op_trace(q, r, ari_config)
            ari_norm = normalize_ari_steps(op_trace)
            ari_ops = [x["op"] for x in op_trace]
            primary_op = get_primary_op(op_trace, ari_config)
            
            triggers = build_triggers(q, op_trace, text_proc)
            chapter_code = map_qi_to_chapter(q, chapters)
            
            qi_items.append({
                "qi_id": qi_id,
                "pair_id": pid,
                "k": i + 1,
                "qi": q,
                "rqi": r,
                "has_rqi": bool(r),
                "chapter_code": chapter_code,
                "primary_op": primary_op,
                "ari_ops": ari_ops,
                "ari": {"steps": ari_norm},
                "triggers": triggers,
            })
    
    # Sanity check
    sanity_ok, sanity_audit = sanity_eval(doc_audits, qi_items)
    
    # Build QC
    qc_pack, qc_map = build_qc_from_qi(qi_items, chapters, ari_config)
    
    # Calculer F1
    f1f2_params = pack_get_f1f2_params(pack)
    epsilon = f1f2_params["epsilon"]
    
    for qc in qc_pack:
        ch_code = qc.get("chapter_code", "UNMAPPED")
        ch_info = next((c for c in chapters if c.get("code") == ch_code), {})
        delta_c = float(ch_info.get("delta_c", 1.0))
        
        all_triggers = []
        for qi_id in qc.get("qi_ids", []):
            qi = next((q for q in qi_items if q["qi_id"] == qi_id), None)
            if qi:
                all_triggers.extend(qi.get("triggers", []))
        
        Tj = compute_trigger_weights(all_triggers)
        m_q = len(Tj)
        psi_raw = f1_raw(delta_c, epsilon, Tj, m_q)
        qc["psi_raw"] = psi_raw
        qc["n_q_historical"] = qc.get("cluster_size", 1)
    
    # Normaliser F1 par chapitre
    by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    for qc in qc_pack:
        cc = qc.get("chapter_code", "UNMAPPED")
        by_chapter.setdefault(cc, []).append(qc)
    
    for cc, qcs in by_chapter.items():
        f1_normalize_in_chapter(qcs)
    
    # Marquer orphelins
    mapped_qi_ids = set(qc_map.keys())
    orphans = 0
    for qi in qi_items:
        if qi["qi_id"] not in mapped_qi_ids:
            qi["is_orphan"] = True
            orphans += 1
        else:
            qi["is_orphan"] = False
            qi["qc_id"] = qc_map.get(qi["qi_id"])
    
    # Stats
    qi_total = len(qi_items)
    rqi_total = sum(1 for q in qi_items if q.get("has_rqi"))
    qi_posable = sum(1 for q in qi_items if q.get("has_rqi") and not q.get("is_orphan"))
    qc_total = len(qc_pack)
    qc_posable = sum(1 for qc in qc_pack if qc.get("qc_state") in ("POSABLE", "POSABLE_WEAK"))
    qc_unposable = sum(1 for qc in qc_pack if qc.get("qc_state") == "UNPOSABLE")
    
    audit = {
        "qi_total": qi_total,
        "rqi_total": rqi_total,
        "qi_posable": qi_posable,
        "qi_orphans": orphans,
        "qc_total": qc_total,
        "qc_posable": qc_posable,
        "qc_unposable": qc_unposable,
        "sanity_ok": sanity_ok,
        "sanity_audit": sanity_audit,
    }
    
    return {
        "qi_pack": qi_items,
        "qc_pack": qc_pack,
        "qc_map": qc_map,
        "audit": audit,
    }


def run_saturation_test(library: List[Dict[str, Any]], phase_a: int, phase_b: int, 
                        pack: Dict[str, Any]) -> Dict[str, Any]:
    """Test de saturation ISO-PROD."""
    log(f"[SATURATION] Phase A: {phase_a}, Phase B: {phase_b}")
    
    a_out = run_phase(library, phase_a, pack)
    b_out = run_phase(library, phase_b, pack)
    
    setA = {qc["qc_id"] for qc in a_out["qc_pack"]}
    setB = {qc["qc_id"] for qc in b_out["qc_pack"]}
    setB_minus_setA = sorted(setB - setA)
    
    saturation_ok = len(setB_minus_setA) == 0
    
    # Sélection F2
    f1f2_params = pack_get_f1f2_params(pack)
    N_total = b_out["audit"]["qi_posable"]
    
    subject = st.session_state.get("subject", "")
    level = st.session_state.get("level", "")
    chapters = pack_get_chapters(pack, subject, level)
    
    selection = []
    by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    for qc in b_out["qc_pack"]:
        by_chapter.setdefault(qc["chapter_code"], []).append(qc)
    
    for cc, qcs in by_chapter.items():
        selected = progressive_select(qcs, N_total, f1f2_params, top_k=12)
        ch_info = next((c for c in chapters if c.get("code") == cc), {"label": cc})
        selection.append({
            "chapter_code": cc,
            "chapter_label": ch_info.get("label", cc),
            "selected_qc": [
                {
                    "qc_id": x.get("qc_id"),
                    "qc": x.get("qc"),
                    "Psi_q": x.get("Psi_q"),
                    "f2_score": x.get("_f2_score"),
                    "primary_op": x.get("primary_op"),
                }
                for x in selected
            ],
        })
    
    selection_report = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "phaseA_pairs": phase_a,
        "phaseB_pairs": phase_b,
        "N_total": N_total,
        "chapters": selection,
    }
    
    # SEALED gate
    orphans = b_out["audit"]["qi_orphans"]
    posable = b_out["audit"]["qi_posable"]
    sanity_ok = b_out["audit"]["sanity_ok"]
    qc_total = b_out["audit"]["qc_total"]
    qc_posable = b_out["audit"]["qc_posable"]
    
    sealed = bool(saturation_ok and sanity_ok and orphans == 0 and posable > 0 and qc_total > 0 and qc_posable > 0)
    
    audit = {
        "phaseA": a_out["audit"],
        "phaseB": b_out["audit"],
        "saturation_ok": saturation_ok,
        "setA_size": len(setA),
        "setB_size": len(setB),
        "setB_minus_setA_size": len(setB_minus_setA),
        "setB_minus_setA": setB_minus_setA[:20],
        "sealed": sealed,
    }
    
    return {
        "phaseA": a_out,
        "phaseB": b_out,
        "setA": setA,
        "setB": setB,
        "selection_report": selection_report,
        "audit": audit,
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
    st.caption("ISO-PROD | Zéro Hardcode | Pack-Driven | F1/F2 Conformes")
    
    # === SIDEBAR ===
    with st.sidebar:
        st.markdown("## 1. ACTIVATION PACK")
        st.info("⚠️ Un Pack JSON est OBLIGATOIRE. Aucune donnée métier n'est dans le code.")
        
        # Upload Pack
        up = st.file_uploader("Charger Pack JSON", type=["json"])
        
        # Ou rechercher fichiers existants
        pack_files = find_pack_files()
        if pack_files:
            st.markdown("**Ou sélectionner un fichier existant:**")
            selected_file = st.selectbox("Packs disponibles", [""] + pack_files)
        else:
            selected_file = ""
        
        if st.button("ACTIVER PACK", use_container_width=True):
            pack = None
            if up:
                try:
                    pack = load_pack_from_upload(up.read())
                    log("[PACK] Chargé depuis upload")
                except Exception as e:
                    st.error(f"Erreur upload: {e}")
            elif selected_file:
                try:
                    pack = load_pack_from_file(selected_file)
                    log(f"[PACK] Chargé depuis {selected_file}")
                except Exception as e:
                    st.error(f"Erreur fichier: {e}")
            else:
                st.error("Veuillez uploader un Pack JSON ou sélectionner un fichier")
            
            if pack:
                if activate_pack(pack):
                    st.success(f"✅ Pack activé: {st.session_state.pack_id}")
                else:
                    st.error("❌ Pack invalide - voir logs")
        
        # Afficher état Pack
        if st.session_state.pack_active:
            st.success(f"✅ {st.session_state.pack_id}")
            st.caption(f"Pays: {st.session_state.country}")
            st.caption(f"SHA256: {st.session_state.pack_sig_sha256[:16]}...")
            
            st.markdown("---")
            st.markdown("## 2. SÉLECTION")
            
            pack = st.session_state.pack_active
            taxonomy = pack.get("chapter_taxonomy", {})
            
            subjects = list(taxonomy.keys())
            if subjects:
                subject = st.selectbox("Matière", subjects, index=subjects.index(st.session_state.subject) if st.session_state.subject in subjects else 0)
                st.session_state.subject = subject
                
                levels = list(taxonomy.get(subject, {}).keys())
                if levels:
                    level = st.selectbox("Niveau", levels, index=levels.index(st.session_state.level) if st.session_state.level in levels else 0)
                    st.session_state.level = level
            
            # Afficher chapitres
            chapters = pack_get_chapters(pack, st.session_state.subject, st.session_state.level)
            if chapters:
                st.markdown("### Chapitres")
                for ch in chapters[:8]:
                    st.write(f"• {ch.get('code', 'N/A')}")
    
    # === MAIN TABS ===
    tab1, tab2, tab3 = st.tabs(["Import", "RUN", "Exports"])
    
    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?"))
    
    with tab1:
        st.markdown("## Import")
        metric_row(
            len(lib), corr_ok,
            st.session_state.run_stats.get("qi", 0),
            st.session_state.run_stats.get("qi_posable", 0),
            st.session_state.run_stats.get("qc", 0),
            st.session_state.sealed,
        )
        
        if lib:
            cols = ["pair_id", "year", "sujet", "corrige?", "match_score"]
            st.dataframe([{k: r.get(k, "") for k in cols} for r in lib[:50]], use_container_width=True, hide_index=True)
        
        st.markdown("### HARVEST")
        if not st.session_state.pack_active:
            st.warning("⚠️ Activez un Pack d'abord")
        else:
            pack = st.session_state.pack_active
            source = pack_get_harvest_source(pack)
            
            if source:
                st.info(f"Source: {source.get('source_name', source.get('source_id', 'N/A'))}")
                c1, c2 = st.columns(2)
                years_back = c1.number_input("Années", 1, 15, 5)
                volume_max = c2.number_input("Volume max", 5, 100, 30)
                
                if st.button("HARVEST", use_container_width=True):
                    try:
                        manifest = harvest_from_pack(
                            pack, 
                            st.session_state.level, 
                            st.session_state.subject,
                            int(years_back), 
                            int(volume_max)
                        )
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
                st.error("Aucune source de harvest dans le Pack")
    
    with tab2:
        st.markdown("## RUN")
        lib2 = st.session_state.library
        corr_ok2 = sum(1 for x in lib2 if x.get("corrige?"))
        
        if not st.session_state.pack_active:
            st.error("Pack inactif")
        elif not lib2:
            st.warning("Bibliothèque vide - faites un HARVEST d'abord")
        elif corr_ok2 <= 0:
            st.error("Aucun corrigé exploitable")
        else:
            pack = st.session_state.pack_active
            max_vol = max(1, min(100, corr_ok2))
            c1, c2 = st.columns(2)
            phase_a = c1.slider("Phase A", 1, max_vol, min(5, max_vol))
            phase_b = c2.slider("Phase B", 1, max_vol, min(10, max_vol))
            
            if st.button("LANCER TEST SATURATION", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_saturation_test(lib2, phase_a, phase_b, pack)
                    
                    st.session_state.qi_pack = res["phaseB"]["qi_pack"]
                    st.session_state.qc_pack = res["phaseB"]["qc_pack"]
                    st.session_state.selection_report = res["selection_report"]
                    st.session_state.last_run_audit = res["audit"]
                    
                    sealed = res["audit"]["sealed"]
                    st.session_state.sealed = sealed
                    
                    st.session_state.run_stats = {
                        "qi": res["audit"]["phaseB"]["qi_total"],
                        "rqi": res["audit"]["phaseB"]["rqi_total"],
                        "qc": res["audit"]["phaseB"]["qc_total"],
                        "qi_posable": res["audit"]["phaseB"]["qi_posable"],
                        "orphans": res["audit"]["phaseB"]["qi_orphans"],
                        "sanity_ok": res["audit"]["phaseB"]["sanity_ok"],
                    }
                    
                    status = "✅ SEALED=YES" if sealed else "⚠️ SEALED=NO"
                    st.info(f"{status} | sat={res['audit']['saturation_ok']} | orphans={res['audit']['phaseB']['qi_orphans']} | posable={res['audit']['phaseB']['qi_posable']} | QC={res['audit']['phaseB']['qc_total']}")
                    
                    if not res["audit"]["saturation_ok"]:
                        st.warning(f"Nouvelles QC: {res['audit']['setB_minus_setA_size']}")
                        st.json(res["audit"]["setB_minus_setA"][:10])
                
                except Exception as e:
                    st.error(f"❌ RUN: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        
        st.markdown("### Logs")
        st.text_area("", logs_text(), height=300)
    
    with tab3:
        st.markdown("## Exports")
        metric_row(
            len(st.session_state.library),
            sum(1 for x in st.session_state.library if x.get("corrige?")),
            st.session_state.run_stats.get("qi", 0),
            st.session_state.run_stats.get("qi_posable", 0),
            st.session_state.run_stats.get("qc", 0),
            st.session_state.sealed,
        )
        
        hm = st.session_state.harvest_manifest or {"version": APP_VERSION}
        st.download_button("harvest_manifest.json", json.dumps(hm, ensure_ascii=False, indent=2), "harvest_manifest.json")
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
                qc_labels = [
                    f"{q['qc_id']} | {q['chapter_code']} | {q.get('primary_op','')} | n={q['cluster_size']} | Ψ={round(float(q.get('Psi_q',0)),3)}"
                    for q in qcs
                ]
                sel_idx = st.selectbox("QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]
                
                st.markdown(f"### {qc['qc']}")
                st.write(f"primary_op: {qc.get('primary_op')}")
                st.write(f"qc_state: {qc.get('qc_state')}")
                st.write(f"Psi_q (F1): {qc.get('Psi_q', 0):.4f}")
                
                qi_by_id = {q["qi_id"]: q for q in st.session_state.qi_pack}
                
                for qi_id in qc.get("qi_ids", [])[:15]:
                    q = qi_by_id.get(qi_id)
                    if not q:
                        continue
                    with st.expander(f"{qi_id} | RQi={'✅' if q.get('has_rqi') else '❌'}"):
                        st.markdown("**Qi**")
                        st.text(q["qi"][:500])
                        st.markdown("**RQi**")
                        st.text(q["rqi"][:500] if q.get("rqi") else "—")
        
        st.markdown("### Audit")
        if st.session_state.last_run_audit:
            st.json(st.session_state.last_run_audit)


if __name__ == "__main__":
    main()
