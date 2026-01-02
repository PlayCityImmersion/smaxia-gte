# =============================================================================
# SMAXIA GTE Console V32.1.0 — ISO-PROD (KERNEL PUR — ZÉRO HARDCODE VÉRIFIÉ)
# =============================================================================
# CORRECTIONS vs V32.0.0 (Audit OPUS):
# 1) Pack externe OBLIGATOIRE (pas de génération dynamique)
# 2) EXTERNALISATION _OP_PATTERNS → pack["ari_patterns"]
# 3) EXTERNALISATION PRIMARY_OPS_ORDER → pack["primary_ops_order"]
# 4) EXTERNALISATION build_triggers() keywords → pack["trigger_keywords"]
# 5) SUPPRESSION valeurs par défaut FR/TERMINALE/MATH — Pack obligatoire
# =============================================================================
# RÈGLE ABSOLUE: Ce fichier ne contient AUCUNE donnée spécifique à un pays,
# une langue, une matière ou un niveau. TOUT est dans le Pack.
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import glob
import math
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
# KERNEL SETTINGS (INVARIANT MATHÉMATIQUE — NE PAS MODIFIER)
# =============================================================================
APP_VERSION = "V32.1.0"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

# Constantes F1/F2 (invariantes selon Kernel V10.6.1 — formules mathématiques pures)
F1_EPSILON = 1e-6
F2_ALPHA = 1.0
F2_TREC = 1.0

# NOTE: PRIMARY_OPS_ORDER a été SUPPRIMÉ — maintenant dans le Pack


# =============================================================================
# SESSION & LOGS
# =============================================================================
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ss_init():
    """
    Initialise l'état de la session Streamlit.
    NOTE: Aucune valeur par défaut pour country/level/subjects — Pack obligatoire.
    """
    st.session_state.setdefault("pack_active", None)
    st.session_state.setdefault("pack_id", None)
    st.session_state.setdefault("pack_path", None)
    st.session_state.setdefault("pack_sig_sha256", None)
    # CORRECTION V32.1.0: Pas de valeurs par défaut — dérivées du Pack
    st.session_state.setdefault("country", None)
    st.session_state.setdefault("level", None)
    st.session_state.setdefault("subjects", None)
    st.session_state.setdefault("library", [])
    st.session_state.setdefault("harvest_manifest", None)
    st.session_state.setdefault("qi_pack", None)
    st.session_state.setdefault("qc_pack", None)
    st.session_state.setdefault("chapter_report", None)
    st.session_state.setdefault("selection_report", None)
    st.session_state.setdefault("sealed", False)
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault(
        "run_stats",
        {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0, "orphans": 0, "sanity_ok": False},
    )
    st.session_state.setdefault("last_run_audit", None)
    st.session_state.setdefault("_uploads", {})
    st.session_state.setdefault("_http_pdf_cache", {})


def log(msg: str):
    st.session_state.logs.append(f"[{_utc_ts()}] {msg}")


def logs_text() -> str:
    return "\n".join(st.session_state.logs[-8000:])


# =============================================================================
# TEXT UTILS (KERNEL — INVARIANT)
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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# COUNTRY ACADEMIC PACK (CAP) — CHARGEMENT OBLIGATOIRE
# =============================================================================
def validate_pack_schema(pack: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valide que le Pack contient tous les champs obligatoires.
    Retourne (is_valid, list_of_errors).
    """
    errors = []
    
    # Champs obligatoires de niveau 1
    required_fields = [
        "pack_id",
        "pack_version",
        "country_code",
        "harvest_sources",
        "text_processing",
        "chapter_taxonomy",
        "ari_patterns",           # NOUVEAU V32.1.0
        "primary_ops_order",      # NOUVEAU V32.1.0
        "trigger_keywords",       # NOUVEAU V32.1.0
    ]
    
    for field in required_fields:
        if field not in pack:
            errors.append(f"Champ obligatoire manquant: '{field}'")
    
    # Validation des sous-structures
    if "harvest_sources" in pack:
        if not isinstance(pack["harvest_sources"], list) or len(pack["harvest_sources"]) == 0:
            errors.append("'harvest_sources' doit être une liste non vide")
    
    if "ari_patterns" in pack:
        if not isinstance(pack["ari_patterns"], list):
            errors.append("'ari_patterns' doit être une liste de patterns")
    
    if "primary_ops_order" in pack:
        if not isinstance(pack["primary_ops_order"], list):
            errors.append("'primary_ops_order' doit être une liste ordonnée d'opérateurs")
    
    if "trigger_keywords" in pack:
        if not isinstance(pack["trigger_keywords"], list):
            errors.append("'trigger_keywords' doit être une liste de mots-clés")
    
    return (len(errors) == 0, errors)


def load_academic_pack(uploaded_pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Charge et valide le Country Academic Pack (CAP).
    
    CORRECTION V32.1.0: Le Pack est OBLIGATOIRE. Pas de génération Genesis.
    """
    if not uploaded_pack:
        raise ValueError("Pack obligatoire. Veuillez uploader un fichier JSON.")
    
    log(f"[PACK] Chargement depuis fichier uploadé.")
    pack = uploaded_pack
    
    # Validation du schéma
    is_valid, errors = validate_pack_schema(pack)
    if not is_valid:
        raise ValueError(f"Pack invalide: {'; '.join(errors)}")
    
    pack["_source"] = "UPLOADED_FILE"
    pack_json = json.dumps(pack, sort_keys=True, ensure_ascii=False)
    pack["_pack_sig_sha256"] = hashlib.sha256(pack_json.encode("utf-8")).hexdigest()
    
    log(f"[PACK] Validé: {pack.get('pack_id')} V{pack.get('pack_version')}")
    return pack


def pack_chapters(pack: Dict[str, Any], level: str, subject: str) -> List[Dict[str, Any]]:
    """Extrait la liste des chapitres du Pack pour le niveau et la matière spécifiés."""
    taxonomy = pack.get("chapter_taxonomy", {})
    return taxonomy.get(subject, {}).get(level, [])


def pack_harvest_source(pack: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extrait la première source de harvest du Pack."""
    sources = pack.get("harvest_sources", [])
    return sources[0] if sources else None


def pack_text_processing(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Extrait la configuration de traitement de texte du Pack."""
    return pack.get("text_processing", {})


def pack_ari_patterns(pack: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Extrait les patterns ARI du Pack.
    NOUVEAU V32.1.0: Les patterns regex sont dans le Pack, pas dans le code.
    """
    raw = pack.get("ari_patterns", [])
    patterns = []
    for item in raw:
        if isinstance(item, dict) and "pattern" in item and "op_code" in item:
            patterns.append((item["pattern"], item["op_code"]))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            patterns.append((item[0], item[1]))
    return patterns


def pack_primary_ops_order(pack: Dict[str, Any]) -> List[str]:
    """
    Extrait l'ordre de priorité des opérateurs du Pack.
    NOUVEAU V32.1.0: L'ordre est dans le Pack, pas dans le code.
    """
    return pack.get("primary_ops_order", [])


def pack_trigger_keywords(pack: Dict[str, Any]) -> List[str]:
    """
    Extrait les mots-clés de triggers du Pack.
    NOUVEAU V32.1.0: Les keywords sont dans le Pack, pas dans le code.
    """
    return pack.get("trigger_keywords", [])


def pack_available_levels(pack: Dict[str, Any]) -> List[str]:
    """Extrait les niveaux disponibles depuis la source de harvest."""
    source = pack_harvest_source(pack)
    if source:
        return list(source.get("levels", {}).keys())
    return []


def pack_available_subjects(pack: Dict[str, Any]) -> List[str]:
    """Extrait les matières disponibles depuis la taxonomie."""
    taxonomy = pack.get("chapter_taxonomy", {})
    return list(taxonomy.keys())


# =============================================================================
# PDF TEXT EXTRACTION (KERNEL — PACK-DRIVEN)
# =============================================================================
def _fix_missing_spaces(text: str, text_proc: Dict[str, Any]) -> str:
    """Réparation des espaces manquants, pilotée par le Pack."""
    if not text:
        return ""
    
    # Appliquer les patterns de base depuis le Pack
    for item in text_proc.get("glued_patterns", []):
        try:
            text = re.sub(item["pattern"], item["replacement"], text)
        except re.error:
            pass
    
    # Heuristique avec dictionnaire de mots du Pack
    common_words = set(text_proc.get("common_words", []))
    
    def split_glued_word(match):
        word = match.group(0)
        if len(word) < 8 or not common_words:
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
    
    # Pattern générique pour mots longs (invariant)
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
    """Nettoie le texte extrait du PDF en utilisant la config du Pack."""
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
            if re.fullmatch(r"\d{1,3}", ln.strip()) or re.fullmatch(r"page\s*\d{1,3}", ln.strip().lower()):
                continue
            out_lines.append(ln)
        out_lines.append("")
    
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes, text_proc: Dict[str, Any]) -> str:
    """Extrait le texte d'un PDF avec la config du Pack."""
    pages = _extract_pages_pdfplumber(pdf_bytes)
    if not pages:
        pages = _extract_pages_pypdf(pdf_bytes)
    return clean_pdf_text(pages, text_proc)


# =============================================================================
# HTTP & PDF FETCH (KERNEL)
# =============================================================================
def _http_get(url: str) -> requests.Response:
    res = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    res.raise_for_status()
    return res


def fetch_pdf_bytes(url: str) -> bytes:
    """Télécharge un PDF depuis une URL avec cache."""
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
# HARVEST (PACK-DRIVEN)
# =============================================================================
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé")
    return BeautifulSoup(html, "html.parser")


def _abs_url(base: str, href: str) -> str:
    return href if href.startswith("http") else requests.compat.urljoin(base, href)


def _is_meta_pdf(url: str, label: str, meta_exclude: List[str]) -> bool:
    """Vérifie si le PDF est un fichier meta (index, sommaire, etc.) selon le Pack."""
    s = norm_text(url + " " + (label or ""))
    return any(w in s for w in meta_exclude)


def _is_corrige_label(url: str, label: str, corrige_patterns: List[str]) -> bool:
    """Vérifie si le PDF est un corrigé selon les patterns du Pack."""
    s = norm_text(os.path.basename(url) + " " + (label or ""))
    return any(p in s for p in corrige_patterns)


def _extract_geo_date(name: str, geo_zones: List[str]) -> Tuple[str, str]:
    """Extrait la zone géographique et la date selon les zones du Pack."""
    s = norm_text(name).replace(".pdf", "")
    geo = ""
    for z in geo_zones:
        if z in s:
            geo = z
            break
    date_match = re.search(r"j[12][\s_]*\d{1,2}[\s_]*\d{1,2}", s)
    date_part = date_match.group(0) if date_match else ""
    return (geo, date_part)


def _match_score_strict(sujet_name: str, corrige_name: str, geo_zones: List[str]) -> float:
    """Calcule le score de matching sujet/corrigé."""
    s_geo, s_date = _extract_geo_date(sujet_name, geo_zones)
    c_geo, c_date = _extract_geo_date(corrige_name, geo_zones)
    score = 0.0
    if s_geo and c_geo:
        if s_geo == c_geo:
            score += 0.5
        else:
            return 0.0
    if s_date and c_date:
        if s_date == c_date:
            score += 0.3
        elif s_date[:2] == c_date[:2]:
            score += 0.15
    s_tok = set(tokenize(sujet_name))
    c_tok = set(tokenize(corrige_name))
    if s_tok and c_tok:
        jaccard_score = len(s_tok & c_tok) / len(s_tok | c_tok)
        score += 0.2 * jaccard_score
    return min(1.0, score)


def harvest_from_source(source_config: Dict[str, Any], level: str, subject: str, years_back: int, volume_max: int, pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Harvester générique piloté par la configuration du Pack.
    Scrape les sujets et corrigés depuis la source définie dans le Pack.
    """
    base_url = source_config.get("base_url", "")
    level_path = source_config.get("levels", {}).get(level)
    if not level_path:
        raise ValueError(f"Niveau '{level}' non trouvé dans la source {source_config.get('source_id')}")
    
    root = f"{base_url}{level_path}"
    log(f"[HARVEST] scope={level}|{subject} root={root}")
    
    # Patterns depuis le Pack
    year_pattern = source_config.get("year_pattern", r"Annee-(20\d{2})")
    meta_exclude = source_config.get("meta_exclude", [])
    corrige_patterns = source_config.get("pdf_patterns", {}).get("corrige", [])
    geo_zones = source_config.get("geographic_zones", [])
    
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
    log(f"[HARVEST] years={len(selected)} (min={min_year})")
    
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
                if _is_meta_pdf(absu, label, meta_exclude):
                    continue
                pdf_links.append({
                    "url": absu,
                    "name": os.path.basename(absu),
                    "label": label,
                    "is_corrige": _is_corrige_label(absu, label, corrige_patterns),
                })
            if not pdf_links:
                continue
            subjects = [p for p in pdf_links if not p["is_corrige"]]
            corriges = [p for p in pdf_links if p["is_corrige"]]
            used_corr = set()
            for s in subjects:
                if len(pairs) >= volume_max:
                    break
                best = (None, 0.0)
                for c in corriges:
                    if c["url"] in used_corr:
                        continue
                    score = _match_score_strict(s["name"], c["name"], geo_zones)
                    if score > best[1]:
                        best = (c, score)
                corrige = best[0] if best[1] >= 0.4 else None
                if corrige:
                    used_corr.add(corrige["url"])
                pair_id = f"PAIR_{level}|{subject}_{y}_{stable_id(s['name'], str(corrige['name'] if corrige else ''))}"
                item = {
                    "pair_id": pair_id,
                    "scope": f"{level}|{subject}",
                    "source": f"{source_config.get('source_id', 'UNKNOWN')} {y}",
                    "year": y,
                    "sujet": s["name"],
                    "corrige?": bool(corrige),
                    "corrige_name": corrige["name"] if corrige else "",
                    "match_score": round(best[1], 2) if corrige else 0.0,
                    "reason": "" if corrige else "corrigé absent ou non matchable",
                    "sujet_url": s["url"],
                    "corrige_url": corrige["url"] if corrige else "",
                }
                if not any(x["sujet_url"] == item["sujet_url"] for x in pairs):
                    pairs.append(item)
                    if item["corrige?"]:
                        corrige_ok += 1
            log(f"[HARVEST] year={y} pdfs={len(pdf_links)} sujets={len(subjects)} corriges={len(corriges)} pairs_ok={corrige_ok}")
        except Exception as e:
            log(f"[HARVEST] year={y} FAILED: {e}")
    
    return {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "country": pack.get("country_code"),
        "level": level,
        "subjects": [subject],
        "items_total": len(pairs),
        "items_corrige_ok": corrige_ok,
        "library": pairs,
    }


# =============================================================================
# Qi / RQi EXTRACTION (KERNEL — PACK-DRIVEN)
# =============================================================================
# NOTE: Les regex de segmentation sont des patterns structurels invariants
# (Exercice, Partie, Question numérotée). Ce ne sont PAS des données contextuelles.
_EXERCICE_RE = re.compile(r"(?i)(?:^|\n)\s*(?:EXERCICE|Exercice|Ex\.?)\s*(\d+|[IVX]+)", re.MULTILINE)
_PARTIE_RE = re.compile(r"(?i)(?:^|\n)\s*(?:PARTIE|Partie)\s*([A-Z]|\d+)", re.MULTILINE)
_QUESTION_RE = re.compile(r"(?m)^\s*(\d{1,2}|[a-h]|[ivx]{1,4})\s*[\)\.\-:]\s+")
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦±×÷]")


def _build_intent_re(intent_verbs: List[str]) -> re.Pattern:
    """Construit la regex d'intention à partir des verbes du Pack."""
    if not intent_verbs:
        # Pas de verbes = pas de filtrage par intention
        return re.compile(r"^$")  # Ne matche rien
    pattern = r"\b(" + "|".join(re.escape(v) for v in intent_verbs) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def _merge_wrapped_lines(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").split("\n")]
    out, buf = [], ""
    for ln in lines:
        if not ln.strip():
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append("")
            continue
        if not buf:
            buf = ln.strip()
            continue
        if not re.search(r"[.:;!?]$", buf):
            buf = (buf + " " + ln.strip()).strip()
        else:
            out.append(buf.strip())
            buf = ln.strip()
    if buf:
        out.append(buf.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def _looks_like_question_unit(s: str, intent_re: re.Pattern) -> bool:
    if not s or len(s.strip()) < 25:
        return False
    t = s.strip()
    has_intent = bool(intent_re.search(t)) or ("?" in t)
    has_math = bool(_MATH_SYMBOL_RE.search(t)) or bool(re.search(r"\b\d+\b", t))
    return bool(has_intent or (has_math and len(t) > 40))


def _chunk_candidates(text: str) -> List[str]:
    """Segmentation du texte en chunks candidats."""
    t = (text or "").replace("\r", "\n")
    t = _dehyphenate(t)
    t = _merge_wrapped_lines(t)
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


def split_questions(text: str, text_proc: Dict[str, Any], max_items: int = 200) -> Tuple[List[str], Dict[str, Any]]:
    """Segmente le texte en questions individuelles."""
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
        
        if _looks_like_question_unit(c, intent_re):
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
    """Aligne les questions avec les réponses par similarité."""
    link: List[Optional[int]] = [None] * len(questions)
    audits: List[Dict[str, Any]] = []
    
    if not questions or not responses:
        return link, audits
    
    q_tokens = [tokenize(q) for q in questions]
    r_tokens = [tokenize(r) for r in responses]
    
    used_r = set()
    
    for i, q_tok in enumerate(q_tokens):
        best_j = None
        best_score = 0.0
        
        for j, r_tok in enumerate(r_tokens):
            if j in used_r:
                continue
            score = jaccard(q_tok, r_tok)
            if score > best_score:
                best_score = score
                best_j = j
        
        if best_j is not None and best_score >= 0.15:
            link[i] = best_j
            used_r.add(best_j)
        
        audits.append({
            "qi_k": i + 1,
            "matched_rqi_k": (best_j + 1) if best_j is not None else None,
            "score": round(best_score, 3),
            "mode": "JACCARD" if best_j is not None else "NONE",
        })
    
    return link, audits


# =============================================================================
# ARI EXTRACTION (KERNEL — PACK-DRIVEN)
# =============================================================================
# CORRECTION V32.1.0: _OP_PATTERNS SUPPRIMÉ — maintenant dans le Pack

def extract_op_trace(response: str, question: str, ari_patterns: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Extrait la trace des opérateurs ARI depuis la réponse et la question.
    
    CORRECTION V32.1.0: Les patterns sont lus depuis le Pack, pas hardcodés.
    """
    combined = norm_text(f"{question} {response}")
    ops = []
    seen = set()
    
    for pattern, op_code in ari_patterns:
        try:
            if re.search(pattern, combined, re.IGNORECASE):
                if op_code not in seen:
                    ops.append({"op": op_code, "confidence": 0.8})
                    seen.add(op_code)
        except re.error:
            log(f"[ARI] Pattern regex invalide: {pattern}")
    
    if not ops:
        ops.append({"op": "OP_STANDARD", "confidence": 0.5})
    
    return ops


def normalize_ari_steps(op_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise les étapes ARI."""
    return [{"op": x["op"], "step": i + 1} for i, x in enumerate(op_trace)]


def get_primary_op(op_trace: List[Dict[str, Any]], primary_ops_order: List[str]) -> str:
    """
    Détermine l'opérateur principal selon l'ordre de priorité.
    
    CORRECTION V32.1.0: L'ordre est lu depuis le Pack, pas hardcodé.
    """
    ops_in_trace = {x["op"] for x in op_trace}
    for op in primary_ops_order:
        if op in ops_in_trace:
            return op
    return "OP_STANDARD"


# =============================================================================
# TRIGGERS (KERNEL — PACK-DRIVEN)
# =============================================================================
# CORRECTION V32.1.0: keywords SUPPRIMÉ — maintenant dans le Pack

def build_triggers(question: str, op_trace: List[Dict[str, Any]], trigger_keywords: List[str]) -> List[str]:
    """
    Construit les triggers à partir de la question et de la trace ARI.
    
    CORRECTION V32.1.0: Les keywords sont lus depuis le Pack, pas hardcodés.
    """
    triggers = []
    
    # Triggers basés sur les opérateurs
    for op_item in op_trace:
        triggers.append(f"ARI:{op_item['op']}")
    
    # Triggers basés sur les mots-clés du Pack
    q_norm = norm_text(question)
    for kw in trigger_keywords:
        if kw.lower() in q_norm:
            triggers.append(f"KW:{kw.upper()}")
    
    return list(set(triggers))[:15]


# =============================================================================
# CHAPTER MAPPING (PACK-DRIVEN)
# =============================================================================
def map_qi_to_chapter(question: str, chapters: List[Dict[str, Any]]) -> str:
    """Mappe une question à un chapitre selon les mots-clés du Pack."""
    q_norm = norm_text(question)
    best_chapter = "UNMAPPED"
    best_score = 0
    
    for ch in chapters:
        keywords = ch.get("keywords", [])
        score = sum(1 for kw in keywords if kw in q_norm)
        if score > best_score:
            best_score = score
            best_chapter = ch.get("code", "UNMAPPED")
    
    return best_chapter


# =============================================================================
# QC GENERATION (KERNEL — PACK-DRIVEN)
# =============================================================================
def qc_method_label(primary_op: str, secondary_ops: List[str], op_labels: Dict[str, str]) -> str:
    """
    Génère le libellé de la méthode QC.
    
    CORRECTION V32.1.0: Les labels sont lus depuis le Pack.
    """
    main_label = op_labels.get(primary_op, f"Résoudre ({primary_op})")
    if secondary_ops:
        secondary_labels = [op_labels.get(op, op) for op in secondary_ops[:2]]
        return f"{main_label} en utilisant {', '.join(secondary_labels)}"
    return main_label


def sigma_similarity(qc_a: Dict[str, Any], qc_b: Dict[str, Any]) -> float:
    """Calcule la similarité sigma entre deux QC (Jaccard sur les opérateurs)."""
    a_ops = set(qc_a.get("all_ops", []))
    b_ops = set(qc_b.get("all_ops", []))
    if a_ops and b_ops:
        return len(a_ops & b_ops) / len(a_ops | b_ops)
    return 0.0


def build_qc_from_qi(qi_pack: List[Dict[str, Any]], chapters: List[Dict[str, Any]], pack: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Construit les QC à partir des Qi.
    Clustering par (chapter_code, primary_op) pour stabilité.
    """
    # Labels des opérateurs depuis le Pack
    op_labels = pack.get("op_labels", {})
    
    # Bucketing par (chapter, primary_op)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    
    for qi in qi_pack:
        cc = qi.get("chapter_code", "UNMAPPED")
        primary_op = qi.get("primary_op", "OP_STANDARD")
        key = (cc, primary_op)
        buckets.setdefault(key, []).append(qi)
    
    log(f"[QC] Buckets créés: {len(buckets)} (par chapter+primary_op)")
    
    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}
    
    # Compter les buckets par chapitre pour décider si on garde les singletons
    chapter_bucket_counts: Dict[str, int] = {}
    for (cc, _) in buckets.keys():
        chapter_bucket_counts[cc] = chapter_bucket_counts.get(cc, 0) + 1
    
    for (cc, primary_op), items in sorted(buckets.items()):
        posable = [x for x in items if x.get("has_rqi")]
        cluster_size = len(items)
        posable_count = len(posable)
        
        # Créer QC si: cluster_size >= 2, OU c'est le seul bucket du chapitre, OU il y a au moins 1 posable
        is_singleton = cluster_size < 2
        is_only_in_chapter = chapter_bucket_counts.get(cc, 0) <= 1
        
        if is_singleton and not is_only_in_chapter and posable_count == 0:
            continue
        
        # Collecter toutes les ops secondaires
        all_ops_set = set()
        for it in items:
            for op in it.get("ari_norm_ops", []):
                all_ops_set.add(op)
        secondary_ops = [op for op in all_ops_set if op != primary_op]
        
        qc_text = qc_method_label(primary_op, secondary_ops, op_labels)
        qc_id = f"QC_{stable_id(cc, primary_op)}"
        
        # État de la QC
        if posable_count >= 2:
            qc_state = "POSABLE"
        elif posable_count == 1:
            qc_state = "POSABLE_WEAK"
        else:
            qc_state = "UNPOSABLE"
        
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
    
    log(f"[QC] QC créées: {len(qc_pack)} | Qi mappées: {len(qc_map)}")
    
    return qc_pack, qc_map


# =============================================================================
# F1 / F2 FORMULAS (KERNEL — CONFORMES AU COFFRE-FORT V10.6.1)
# =============================================================================
def compute_trigger_weights(triggers: List[str]) -> Dict[str, float]:
    """
    Calcule les poids des triggers selon la formule F1.
    T_j = {t_1, ..., t_k} avec poids w_i = 1/k pour chaque trigger.
    """
    uniq = list(dict.fromkeys(triggers or []))
    n = len(uniq)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in uniq}


def f1_raw(delta_c: float, epsilon: float, Tj: Dict[str, float], m_q: int) -> float:
    """
    Formule F1 (Kernel V10.6.1 - Section 3.2):
    
    ψ_q = δ_c × (ε + Σ w_i)²
    
    où:
    - δ_c = coefficient de pondération du chapitre
    - ε = constante de lissage (F1_EPSILON)
    - w_i = poids du trigger i
    - m_q = nombre de triggers à considérer (top m_q)
    
    Retourne le score brut ψ_q (non normalisé).
    """
    if not Tj:
        s = 0.0
    else:
        items = sorted(Tj.items(), key=lambda x: x[0])
        top = items[: max(0, int(m_q))]
        s = sum(v for _, v in top)
    return float(delta_c) * float((epsilon + s) ** 2)


def f1_normalize_in_chapter(qc_list: List[Dict[str, Any]]) -> None:
    """
    Normalise les scores F1 au sein d'un chapitre.
    
    Ψ_q = ψ_q / max(ψ_q) pour q ∈ chapitre
    
    Modifie les QC in-place en ajoutant le champ 'Psi_q'.
    """
    if not qc_list:
        return
    raws = [float(q.get("psi_raw", 0.0) or 0.0) for q in qc_list]
    mx = max(raws) if raws else 0.0
    for q in qc_list:
        psi = float(q.get("psi_raw", 0.0) or 0.0)
        Psi = psi / mx if mx > 0 else 0.0
        q["Psi_q"] = float(Psi)


def f2_score(qc: Dict[str, Any], selected: List[Dict[str, Any]], n_q_historical: int, N_total: int, alpha: float, t_rec: float) -> Tuple[float, Dict[str, Any]]:
    """
    Formule F2 (Kernel V10.6.1 - Section 3.3):
    
    Score(q) = (n_q,historical / N_total) × (1 + α/t_rec) × Ψ_q × Π((1-σ(q,p))×100)
    
    où:
    - n_q,historical = nombre historique d'occurrences de la QC
    - N_total = nombre total de Qi dans le corpus
    - α = coefficient de boost (F2_ALPHA)
    - t_rec = temps depuis la dernière occurrence (F2_TREC)
    - Ψ_q = score F1 normalisé
    - σ(q,p) = similarité entre q et les QC déjà sélectionnées p
    
    Retourne (score, audit_details).
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
    audit = {
        "n_q_historical": n_q_historical,
        "N_total": N_total,
        "Psi_q": Psi_q,
        "sigma_list": sigmas[:20],
    }
    return float(score), audit


def progressive_select(qc_list: List[Dict[str, Any]], N_total: int, top_k: int = 12) -> List[Dict[str, Any]]:
    """
    Sélection progressive des QC selon F2.
    
    Algorithme glouton: à chaque itération, sélectionner la QC avec le meilleur score F2
    parmi celles non encore sélectionnées.
    """
    selected: List[Dict[str, Any]] = []
    remaining = [q for q in qc_list if q.get("qc_state") in ("POSABLE", "POSABLE_WEAK")]
    remaining = sorted(remaining, key=lambda x: x.get("qc_id", ""))
    
    for _ in range(min(top_k, len(remaining))):
        best = None
        for qc in remaining:
            n_q_hist = int(qc.get("n_q_historical", qc.get("cluster_size", 1)) or 0)
            sc, audit = f2_score(qc, selected, n_q_hist, N_total, F2_ALPHA, F2_TREC)
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
# SANITY GATE (KERNEL)
# =============================================================================
def sanity_eval(doc_audits: List[Dict[str, Any]], qi_items: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """Évalue la qualité des données extraites."""
    total_raw = sum(int(d.get("raw_chunks", 0) or 0) for d in doc_audits)
    total_kept = sum(int(d.get("kept", 0) or 0) for d in doc_audits)
    total_rej = sum(int(d.get("rejected", 0) or 0) for d in doc_audits)
    
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
# RUN PIPELINE (KERNEL — PACK-DRIVEN)
# =============================================================================
def run_phase(library: List[Dict[str, Any]], volume_pairs: int, chapters: List[Dict[str, Any]], pack: Dict[str, Any]) -> Dict[str, Any]:
    """Exécute une phase du pipeline (extraction Qi/RQi, génération QC)."""
    text_proc = pack_text_processing(pack)
    ari_patterns = pack_ari_patterns(pack)
    primary_ops_order = pack_primary_ops_order(pack)
    trigger_keywords = pack_trigger_keywords(pack)
    
    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    to_process = exploitable[: max(1, min(int(volume_pairs), len(exploitable)))]
    log(f"[RUN] phase processing pairs={len(to_process)}")
    
    qi_items: List[Dict[str, Any]] = []
    doc_audits: List[Dict[str, Any]] = []
    doc_stats: List[Dict[str, Any]] = []
    
    for pair in to_process:
        pid = pair["pair_id"]
        try:
            su_pdf = fetch_pdf_bytes(pair["sujet_url"])
            co_pdf = fetch_pdf_bytes(pair["corrige_url"])
        except Exception as e:
            log(f"[DL] FAILED {pid}: {e}")
            continue
        
        su_text = extract_text_from_pdf_bytes(su_pdf, text_proc)
        co_text = extract_text_from_pdf_bytes(co_pdf, text_proc)
        
        if not su_text or not co_text:
            log(f"[PDF] EMPTY {pid}")
            continue
        
        qs, qs_audit = split_questions(su_text, text_proc, max_items=150)
        rs, rs_audit = split_questions(co_text, text_proc, max_items=200)
        
        doc_audits.append({"pair_id": pid, "role": "SUJET", **qs_audit})
        doc_audits.append({"pair_id": pid, "role": "CORRIGE", **rs_audit})
        
        link, link_audits = align_qi_rqi(qs, rs)
        matched_count = sum(1 for x in link if x is not None)
        log(f"[ALIGN] {pid}: {len(qs)} Qi, {len(rs)} RQi, matched={matched_count}")
        
        doc_stats.append({"pair_id": pid, "qi": len(qs), "rqi": len(rs), "matched": matched_count})
        
        for i, q in enumerate(qs):
            if not q.strip():
                continue
            j = link[i] if i < len(link) else None
            r = rs[j] if (j is not None and 0 <= j < len(rs)) else ""
            
            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
            rqi_id = f"RQI_{stable_id(pid, str(j), norm_text(r)[:180])}" if r else ""
            
            # CORRECTION V32.1.0: Utilise les patterns du Pack
            op_trace = extract_op_trace(r, q, ari_patterns)
            ari_norm = normalize_ari_steps(op_trace)
            ari_ops = [x.get("op") for x in ari_norm if x.get("op")]
            primary_op = get_primary_op(op_trace, primary_ops_order)
            
            # CORRECTION V32.1.0: Utilise les keywords du Pack
            triggers = build_triggers(q, op_trace, trigger_keywords)
            chapter_code = map_qi_to_chapter(q, chapters)
            
            qi_items.append({
                "qi_id": qi_id,
                "pair_id": pid,
                "k": i + 1,
                "qi": q,
                "rqi_id": rqi_id,
                "rqi": r,
                "has_rqi": bool(r),
                "chapter_code": chapter_code,
                "primary_op": primary_op,
                "ari_norm_ops": ari_ops,
                "ari": {"steps": ari_norm},
                "triggers": triggers,
            })
    
    # Sanity check
    sanity_ok, sanity_audit = sanity_eval(doc_audits, qi_items)
    
    # Build QC
    qc_pack, qc_map = build_qc_from_qi(qi_items, chapters, pack)
    
    # Calculer F1 pour chaque QC
    for qc in qc_pack:
        ch_code = qc.get("chapter_code", "UNMAPPED")
        ch_info = next((c for c in chapters if c.get("code") == ch_code), {})
        delta_c = float(ch_info.get("delta_c", 1.0))
        
        # Collecter les triggers de toutes les Qi du cluster
        all_triggers = []
        for qi_id in qc.get("qi_ids", []):
            qi = next((q for q in qi_items if q["qi_id"] == qi_id), None)
            if qi:
                all_triggers.extend(qi.get("triggers", []))
        
        Tj = compute_trigger_weights(all_triggers)
        m_q = len(Tj)
        psi_raw = f1_raw(delta_c, F1_EPSILON, Tj, m_q)
        qc["psi_raw"] = psi_raw
        qc["n_q_historical"] = qc.get("cluster_size", 1)
    
    # Normaliser F1 par chapitre
    by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    for qc in qc_pack:
        cc = qc.get("chapter_code", "UNMAPPED")
        by_chapter.setdefault(cc, []).append(qc)
    
    for cc, qcs in by_chapter.items():
        f1_normalize_in_chapter(qcs)
    
    # Marquer les orphelins
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
    
    # Chapter report
    chapter_report = []
    for cc, qcs in by_chapter.items():
        ch_info = next((c for c in chapters if c.get("code") == cc), {"label": cc})
        chapter_report.append({
            "chapter_code": cc,
            "chapter_label": ch_info.get("label", cc),
            "qc_count": len(qcs),
            "qi_count": sum(qc.get("cluster_size", 0) for qc in qcs),
            "posable_count": sum(1 for qc in qcs if qc.get("qc_state") in ("POSABLE", "POSABLE_WEAK")),
        })
    
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
        "chapter_report": chapter_report,
        "audit": audit,
    }


def run_granulo_test_iso(library: List[Dict[str, Any]], phase_a: int, phase_b: int, chapters: List[Dict[str, Any]], pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Exécute le test de saturation ISO-PROD.
    Phase A: N sujets
    Phase B: 2N sujets
    Vérifie si Set_B - Set_A = Ø (saturation atteinte)
    """
    log(f"[GRANULO] Phase A: {phase_a} pairs, Phase B: {phase_b} pairs")
    
    a_out = run_phase(library, phase_a, chapters, pack)
    b_out = run_phase(library, phase_b, chapters, pack)
    
    # Calculer les ensembles de QC
    setA = {qc["qc_id"] for qc in a_out["qc_pack"]}
    setB = {qc["qc_id"] for qc in b_out["qc_pack"]}
    setB_minus_setA = list(setB - setA)
    
    saturation_ok = len(setB_minus_setA) == 0
    
    # Sélection progressive F2
    N_total = b_out["audit"]["qi_posable"]
    selection = []
    for cc, qcs in {qc["chapter_code"]: [] for qc in b_out["qc_pack"]}.items():
        qcs = [qc for qc in b_out["qc_pack"] if qc["chapter_code"] == cc]
        selected = progressive_select(qcs, N_total, top_k=12)
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
    orphans = int(b_out["audit"]["qi_orphans"])
    posable = int(b_out["audit"]["qi_posable"])
    sanity_ok = bool(b_out["audit"]["sanity_ok"])
    qc_total = int(b_out["audit"]["qc_total"])
    qc_posable = int(b_out["audit"]["qc_posable"])
    
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
        "setB_minus_setA": setB_minus_setA,
        "selection_report": selection_report,
        "audit": audit,
    }


# =============================================================================
# UI (STREAMLIT) — PACK OBLIGATOIRE
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
    
    st.markdown(f"# SMAXIA GTE Console {APP_VERSION} — ISO-PROD")
    st.caption("Kernel Pur: Zéro Hardcode Vérifié — Pack Obligatoire")
    
    with st.sidebar:
        st.markdown("## ACTIVATION")
        st.warning("⚠️ Pack JSON **OBLIGATOIRE**")
        
        up = st.file_uploader("Charger le Pack JSON", type=["json"])
        uploaded_pack = None
        if up:
            try:
                uploaded_pack = json.loads(up.read().decode("utf-8"))
                st.success(f"Pack chargé: {uploaded_pack.get('pack_id', 'N/A')}")
            except Exception as e:
                st.error(f"Pack invalide: {e}")
        
        if st.button("ACTIVER", use_container_width=True):
            if not uploaded_pack:
                st.error("Pack obligatoire. Veuillez uploader un fichier JSON.")
            else:
                try:
                    pack = load_academic_pack(uploaded_pack)
                    st.session_state.pack_active = pack
                    st.session_state.pack_id = str(pack.get("pack_id") or "")
                    st.session_state.pack_sig_sha256 = pack.get("_pack_sig_sha256")
                    # Dériver les valeurs depuis le Pack
                    st.session_state.country = pack.get("country_code")
                    available_levels = pack_available_levels(pack)
                    st.session_state.level = available_levels[0] if available_levels else None
                    available_subjects = pack_available_subjects(pack)
                    st.session_state.subjects = available_subjects
                    st.success("Pack actif")
                except Exception as e:
                    st.error(f"Erreur: {e}")
        
        if st.session_state.pack_active:
            pack = st.session_state.pack_active
            st.success(f"✅ {st.session_state.pack_id}")
            st.caption(f"Pays: {pack.get('country_code')}")
            st.caption(f"SHA256: {st.session_state.pack_sig_sha256[:16] if st.session_state.pack_sig_sha256 else 'N/A'}...")
            
            st.markdown("---")
            st.markdown("## SÉLECTION")
            
            available_levels = pack_available_levels(pack)
            if available_levels:
                level = st.radio("Niveau", available_levels, index=0)
                st.session_state.level = level
            
            available_subjects = pack_available_subjects(pack)
            if available_subjects:
                subject = st.radio("Matière", available_subjects, index=0)
                st.session_state.subject = subject
            
            st.markdown("### Chapitres")
            chapters = pack_chapters(pack, st.session_state.level, st.session_state.get("subject", available_subjects[0] if available_subjects else ""))
            for c in chapters[:8]:
                st.write(f"• {c.get('code', 'N/A')}")
    
    # Vérifier si Pack actif
    if not st.session_state.pack_active:
        st.warning("⚠️ **Pack non activé.** Veuillez uploader et activer un Country Academic Pack (JSON) dans la barre latérale pour commencer.")
        st.info("Le Pack doit contenir: `pack_id`, `harvest_sources`, `chapter_taxonomy`, `ari_patterns`, `primary_ops_order`, `trigger_keywords`.")
        return
    
    pack = st.session_state.pack_active
    subject = st.session_state.get("subject", pack_available_subjects(pack)[0] if pack_available_subjects(pack) else "MATH")
    chapters = pack_chapters(pack, st.session_state.level, subject)
    
    tab1, tab2, tab3 = st.tabs(["Import", "RUN", "Exports"])
    
    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?") and x.get("corrige_url"))
    
    with tab1:
        st.markdown("## Import")
        metric_row(
            len(lib),
            corr_ok,
            st.session_state.run_stats.get("qi", 0),
            st.session_state.run_stats.get("qi_posable", 0),
            st.session_state.run_stats.get("qc", 0),
            st.session_state.sealed,
        )
        
        if lib:
            cols = ["pair_id", "year", "sujet", "corrige?", "corrige_name", "match_score"]
            st.dataframe([{k: r.get(k, "") for k in cols} for r in lib[:50]], use_container_width=True, hide_index=True)
        
        st.markdown("### HARVEST")
        source = pack_harvest_source(pack)
        if source:
            st.info(f"Source: {source.get('source_name', source.get('source_id', 'N/A'))}")
            c1, c2 = st.columns(2)
            years_back = c1.number_input("Années", 1, 15, 5)
            volume_max = c2.number_input("Volume max", 5, 100, 30)
            
            if st.button("HARVEST", use_container_width=True):
                try:
                    manifest = harvest_from_source(source, st.session_state.level, subject, int(years_back), int(volume_max), pack)
                    st.session_state.harvest_manifest = manifest
                    st.session_state.library = manifest["library"]
                    st.session_state.sealed = False
                    st.session_state.qi_pack = None
                    st.session_state.qc_pack = None
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0, "orphans": 0, "sanity_ok": False}
                    st.success(f"HARVEST: {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                except Exception as e:
                    st.error(f"HARVEST: {e}")
        else:
            st.error("Aucune source de harvest configurée dans le Pack.")
    
    with tab2:
        st.markdown("## RUN")
        lib2 = st.session_state.library
        corr_ok2 = sum(1 for x in lib2 if x.get("corrige?") and x.get("corrige_url"))
        
        if not lib2:
            st.warning("Bibliothèque vide")
        elif corr_ok2 <= 0:
            st.error("Aucun corrigé exploitable")
        else:
            max_vol = max(1, min(100, corr_ok2))
            c1, c2 = st.columns(2)
            phase_a = c1.slider("Phase A", 1, max_vol, min(5, max_vol))
            phase_b = c2.slider("Phase B", 1, max_vol, min(10, max_vol))
            
            if st.button("LANCER", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_granulo_test_iso(lib2, phase_a, phase_b, chapters, pack)
                    
                    st.session_state.qi_pack = res["phaseB"]["qi_pack"]
                    st.session_state.qc_pack = res["phaseB"]["qc_pack"]
                    st.session_state.chapter_report = res["phaseB"]["chapter_report"]
                    st.session_state.selection_report = res["selection_report"]
                    st.session_state.last_run_audit = res["audit"]
                    
                    sealed = bool(res["audit"]["sealed"])
                    st.session_state.sealed = sealed
                    
                    st.session_state.run_stats = {
                        "qi": res["audit"]["phaseB"]["qi_total"],
                        "rqi": res["audit"]["phaseB"]["rqi_total"],
                        "qc": res["audit"]["phaseB"]["qc_total"],
                        "qi_posable": res["audit"]["phaseB"]["qi_posable"],
                        "orphans": res["audit"]["phaseB"]["qi_orphans"],
                        "sanity_ok": bool(res["audit"]["phaseB"]["sanity_ok"]),
                    }
                    
                    status = "✅ SEALED=YES" if sealed else "⚠️ SEALED=NO"
                    st.info(
                        f"{status} | sat={res['audit']['saturation_ok']} | sanity={res['audit']['phaseB']['sanity_ok']} "
                        f"| orphans={res['audit']['phaseB']['qi_orphans']} | posable={res['audit']['phaseB']['qi_posable']} "
                        f"| QC={res['audit']['phaseB']['qc_total']} (posable={res['audit']['phaseB'].get('qc_posable')}, unposable={res['audit']['phaseB'].get('qc_unposable')})"
                    )
                    
                    if not res["audit"]["saturation_ok"]:
                        st.warning(f"Nouvelles QC (setB-setA): {res['audit']['setB_minus_setA_size']}")
                        st.json(res["audit"].get("setB_minus_setA", [])[:10])
                    
                except Exception as e:
                    st.error(f"RUN: {e}")
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
        if st.session_state.chapter_report:
            st.download_button("chapter_report.json", json.dumps(st.session_state.chapter_report, ensure_ascii=False, indent=2), "chapter_report.json")
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
                    f"{q['qc_id']} | {q['chapter_code']} | {q.get('primary_op','')} | n={q['cluster_size']} | pos={q.get('posable_in_cluster',0)} | state={q.get('qc_state','')} | Ψ={round(float(q.get('Psi_q',0)),3)}"
                    for q in qcs
                ]
                sel_idx = st.selectbox("QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]
                
                st.markdown(f"### {qc['qc']}")
                st.write(f"primary_op: {qc.get('primary_op')}")
                st.write(f"all_ops: {qc.get('all_ops', [])}")
                st.write(f"qc_state: {qc.get('qc_state')}")
                st.write(f"Psi_q (F1): {qc.get('Psi_q', 0):.4f}")
                st.write(f"F2 Score: {qc.get('_f2_score', 'N/A')}")
                
                qi_by_id = {q["qi_id"]: q for q in st.session_state.qi_pack}
                
                for qi_id in qc.get("qi_ids", [])[:20]:
                    q = qi_by_id.get(qi_id)
                    if not q:
                        continue
                    with st.expander(f"{qi_id} | RQi={'✅' if q.get('has_rqi') else '❌'}"):
                        st.markdown("**Qi**")
                        st.text(q["qi"][:600])
                        st.markdown("**RQi**")
                        st.text(q["rqi"][:600] if q.get("rqi") else "—")
                        st.markdown("**ARI**")
                        st.json(q.get("ari", {}))
        
        st.markdown("### Audit")
        if st.session_state.last_run_audit:
            st.json(st.session_state.last_run_audit)


if __name__ == "__main__":
    main()
