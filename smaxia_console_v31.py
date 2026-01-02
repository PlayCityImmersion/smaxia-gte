# =============================================================================
# SMAXIA GTE Console V31.10.16 — ISO-PROD (CORRECTIONS DEFINITIVES + ANTI-FAKE)
# =============================================================================
# CORRECTIONS vs V31.10.15:
# 1) Alignement Qi↔RQi: SUPPRESSION du fallback séquentiel (anti “fake POSABLE”)
# 2) Zéro orphelin SANS TRICHE: création de QC même si posable_in_cluster=0
#    → QC marquées qc_state="UNPOSABLE" (mappage Qi→QC garanti)
# 3) Audit d’alignement: stockage mode + score pour chaque Qi (preuve)
# 4) UI inchangée (tabs/boutons/exports identiques) — NO REGRESSION INTENTIONNELLE
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


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
APP_VERSION = "V31.10.16"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

DEFAULT_COUNTRY = "FR"
DEFAULT_LEVEL = "TERMINALE"
DEFAULT_SUBJECTS = ["MATH"]

APMEP_ROOT_BY_LEVEL = {
    "TERMINALE": "https://www.apmep.fr/Annales-du-Bac-Terminale",
    "PREMIERE": "https://www.apmep.fr/Annales-du-Bac-Premiere",
    "SECONDE": "https://www.apmep.fr/Annales-du-Bac-Seconde",
}

# F1/F2 constants
F1_EPSILON = 1e-6
F2_ALPHA = 1.0
F2_TREC = 1.0


# -----------------------------------------------------------------------------
# SESSION
# -----------------------------------------------------------------------------
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ss_init():
    st.session_state.setdefault("pack_active", None)
    st.session_state.setdefault("pack_id", None)
    st.session_state.setdefault("pack_path", None)
    st.session_state.setdefault("pack_sig_sha256", None)
    st.session_state.setdefault("country", DEFAULT_COUNTRY)
    st.session_state.setdefault("level", DEFAULT_LEVEL)
    st.session_state.setdefault("subjects", DEFAULT_SUBJECTS[:])
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


# -----------------------------------------------------------------------------
# TEXT UTILS
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# PDF TEXT EXTRACTION — ROBUSTE
# -----------------------------------------------------------------------------
def _fix_missing_spaces(text: str) -> str:
    """
    Répare le texte PDF où les espaces sont manquants.
    Détecte les patterns comme "motmot" → "mot mot"
    """
    if not text:
        return ""

    # Pattern: minuscule suivie de majuscule sans espace
    text = re.sub(
        r"([a-zéèêëàâäùûüôöîïç])([A-ZÉÈÊËÀÂÄÙÛÜÔÖÎÏÇ])", r"\1 \2", text
    )

    # Pattern: chiffre collé à lettre
    text = re.sub(
        r"(\d)([a-zA-ZéèêëàâäùûüôöîïçÉÈÊËÀÂÄÙÛÜÔÖÎÏÇ])", r"\1 \2", text
    )
    text = re.sub(
        r"([a-zA-ZéèêëàâäùûüôöîïçÉÈÊËÀÂÄÙÛÜÔÖÎÏÇ])(\d)", r"\1 \2", text
    )

    # Pattern: ponctuation collée au mot suivant
    text = re.sub(
        r"([.!?;:,])([A-Za-zéèêëàâäùûüôöîïçÉÈÊËÀÂÄÙÛÜÔÖÎÏÇ])", r"\1 \2", text
    )

    # Pattern: mots français collés (heuristique basique)
    def split_long_words(match):
        word = match.group(0)
        if len(word) > 25:
            word = re.sub(
                r"(que|qui|dont|ou|et|de|la|le|les|un|une|des|en|à|par|pour|sur|avec|dans|est|sont|soit|donc|car|mais|alors|ainsi|comme)",
                r" \1 ",
                word,
            )
            word = re.sub(r"\s+", " ", word).strip()
        return word

    text = re.sub(r"[a-zéèêëàâäùûüôöîïç]{25,}", split_long_words, text)

    return re.sub(r"\s+", " ", text).strip()


def _dehyphenate(text: str) -> str:
    """Répare les mots coupés en fin de ligne."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _looks_like_page_number(line: str) -> bool:
    s = norm_text(line)
    return bool(re.fullmatch(r"\d{1,3}", s) or re.fullmatch(r"page\s*\d{1,3}", s))


def _extract_pages_pdfplumber(pdf_bytes: bytes, max_pages: int = 120) -> List[str]:
    """Extraction via pdfplumber avec reconstruction du texte."""
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
    """Extraction via pypdf comme fallback."""
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


def clean_pdf_text(pages: List[str]) -> str:
    """Nettoie et reconstruit le texte extrait des pages PDF."""
    if not pages:
        return ""

    page_lines = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = _dehyphenate(p)
        p = _fix_missing_spaces(p)
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        page_lines.append(lines)

    # Détection headers/footers répétés
    top_counts, bot_counts = {}, {}
    n_pages = len(page_lines)

    for lines in page_lines:
        for ln in lines[:3]:
            key = norm_text(ln)[:100]
            if key and len(key) > 5:
                top_counts[key] = top_counts.get(key, 0) + 1
        for ln in lines[-3:]:
            key = norm_text(ln)[:100]
            if key and len(key) > 5:
                bot_counts[key] = bot_counts.get(key, 0) + 1

    threshold = max(2, int(0.4 * n_pages))
    header_keys = {k for k, c in top_counts.items() if c >= threshold}
    footer_keys = {k for k, c in bot_counts.items() if c >= threshold}

    cleaned_pages = []
    for lines in page_lines:
        keep = []
        for ln in lines:
            k = norm_text(ln)[:100]
            if k in header_keys or k in footer_keys:
                continue
            if _looks_like_page_number(ln):
                continue
            if len(ln.strip()) < 3:
                continue
            keep.append(ln)
        cleaned_pages.append("\n".join(keep))

    text = "\n\n".join([p for p in cleaned_pages if p.strip()])
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extraction robuste: essaie pdfplumber, puis pypdf, avec reconstruction.
    """
    pages = _extract_pages_pdfplumber(pdf_bytes)
    text = clean_pdf_text(pages)

    if text:
        space_ratio = text.count(" ") / max(1, len(text))
        if space_ratio >= 0.08:
            return text
        log(f"[PDF] pdfplumber space_ratio={space_ratio:.2%} (faible), essai pypdf")

    pages2 = _extract_pages_pypdf(pdf_bytes)
    text2 = clean_pdf_text(pages2)

    if text2:
        space_ratio2 = text2.count(" ") / max(1, len(text2))
        if space_ratio2 > (text.count(" ") / max(1, len(text)) if text else 0):
            log(f"[PDF] pypdf meilleur space_ratio={space_ratio2:.2%}")
            return text2

    return text if text else text2


# -----------------------------------------------------------------------------
# HTTP + PDF CACHE
# -----------------------------------------------------------------------------
def _http_get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r


def download_pdf_bytes(url: str) -> bytes:
    r = _http_get(url)
    if len(r.content) / (1024 * 1024) > MAX_PDF_MB:
        raise ValueError("PDF trop volumineux")
    return r.content


def fetch_pdf_bytes(url: str) -> bytes:
    if url.startswith("local://"):
        data = st.session_state.get("_uploads", {}).get(url)
        if not data:
            raise RuntimeError(f"Upload local introuvable: {url}")
        return data
    cache = st.session_state.get("_http_pdf_cache", {})
    if url in cache:
        return cache[url]
    b = download_pdf_bytes(url)
    cache[url] = b
    st.session_state._http_pdf_cache = cache
    return b


# -----------------------------------------------------------------------------
# PACK GENESIS
# -----------------------------------------------------------------------------
def _pack_genesis_fr_math(n_chapters: int) -> Dict[str, Any]:
    themes = [
        ("CH_001_ANALYSE", ["limite", "continuite", "derivee", "derivation", "variation", "convexite", "integrale", "primitive", "fonction"]),
        ("CH_002_PROBABILITES", ["probabilite", "loi", "binomiale", "normale", "esperance", "variance", "variable", "aleatoire", "evenement"]),
        ("CH_003_SUITES", ["suite", "recurrence", "convergence", "arithmetique", "geometrique", "terme"]),
        ("CH_004_COMPLEXES", ["complexe", "affixe", "module", "argument", "imaginaire", "conjugue"]),
        ("CH_005_GEOMETRIE", ["vecteur", "plan", "espace", "droite", "orthogonal", "scalaire", "repere", "coordonnees"]),
        ("CH_006_ALGORITHMES", ["algo", "algorithme", "python", "boucle", "programme", "instruction"]),
        ("CH_007_EQUATIONS", ["equation", "systeme", "inequation", "discriminant", "racine", "solution", "resoudre"]),
        ("CH_008_FONCTIONS", ["fonction", "courbe", "image", "antecedent", "composition", "graphe"]),
        ("CH_009_TRIGO", ["sinus", "cosinus", "tangente", "trigonometrie", "radian", "angle", "cos", "sin"]),
        ("CH_010_LOGEXP", ["logarithme", "ln", "exp", "exponentielle", "log"]),
        ("CH_011_VECTMATR", ["matrice", "determinant", "vecteur", "base", "dimension"]),
        ("CH_012_STAT", ["statistique", "moyenne", "ecart", "mediane", "quartile", "frequence"]),
    ]
    themes = themes[: max(1, min(n_chapters, len(themes)))]

    return {
        "pack_id": "CAP_FR_GENESIS_V1",
        "country_code": "FR",
        "country_label": "France",
        "version": "1.0.0",
        "_source": "GENERATED_FILE",
        "chapters": [
            {"chapter_code": code, "chapter_label": code.replace("_", " "), "keywords": kws, "delta_c": 1.0}
            for (code, kws) in themes
        ],
    }


def _candidate_pack_paths(country: str) -> List[str]:
    candidates = [
        os.path.join(os.getcwd(), "academic_packs", f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), "packs", f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), f"CAP_{country}_BAC_2024_V1.json"),
    ]
    candidates += glob.glob(os.path.join(os.getcwd(), "**", f"CAP_{country}_*.json"), recursive=True)
    out, seen = [], set()
    for p in candidates:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
    return out


def _safe_read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_tmp_pack(pack: Dict[str, Any]) -> str:
    raw = json.dumps(pack, ensure_ascii=False, indent=2).encode("utf-8")
    sid = stable_id(sha256_bytes(raw), _utc_ts())
    path = f"/tmp/smaxia_pack_{pack.get('pack_id','PACK')}_{sid}.json"
    with open(path, "wb") as f:
        f.write(raw)
    return path


def load_academic_pack(country: str, allow_genesis: bool, genesis_n_chapters: int, uploaded_pack: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if uploaded_pack:
        tmp_path = _write_tmp_pack(uploaded_pack)
        sig = sha256_file(tmp_path)
        uploaded_pack["_source"] = "UPLOADED_FILE"
        uploaded_pack["_pack_file_path"] = tmp_path
        uploaded_pack["_pack_sig_sha256"] = sig
        log(f"[PACK] Loaded UPLOADED_FILE pack_id={uploaded_pack.get('pack_id')} sha256={sig}")
        return uploaded_pack

    paths = _candidate_pack_paths(country)
    log(f"[PACK] Searching for country={country} | candidates={len(paths)}")

    for path in paths[:30]:
        try:
            pack = _safe_read_json(path)
            pack_id = str(pack.get("pack_id") or pack.get("id") or "").strip()
            chapters = pack.get("chapters") or []
            if pack_id and isinstance(chapters, list) and len(chapters) > 0:
                pack["_pack_file_path"] = os.path.abspath(path)
                pack["_source"] = "FILE"
                pack["_pack_sig_sha256"] = sha256_file(pack["_pack_file_path"])
                log(f"[PACK] Loaded FILE: {pack_id} sha256={pack['_pack_sig_sha256']}")
                return pack
        except Exception as e:
            log(f"[PACK] Failed: {path} | {e}")

    if allow_genesis:
        pack = _pack_genesis_fr_math(int(genesis_n_chapters))
        tmp_path = _write_tmp_pack(pack)
        sig = sha256_file(tmp_path)
        pack["_pack_file_path"] = tmp_path
        pack["_pack_sig_sha256"] = sig
        log(f"[PACK] Activated GENERATED_FILE (Genesis) path={tmp_path} sha256={sig}")
        return pack

    raise RuntimeError(f"Aucun pack FILE trouvé pour {country} et Genesis interdit")


def pack_chapters(pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = pack.get("chapters") or []
    out = []
    for item in ch:
        if isinstance(item, dict):
            code = str(item.get("chapter_code") or item.get("code") or "").strip()
            label = str(item.get("chapter_label") or item.get("label") or code).strip()
            if code:
                out.append({
                    "chapter_code": code,
                    "chapter_label": label,
                    "keywords": item.get("keywords", []),
                    "delta_c": float(item.get("delta_c", 1.0) or 1.0),
                })
    return out


# -----------------------------------------------------------------------------
# HARVEST APMEP — STRICT MATCHING
# -----------------------------------------------------------------------------
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé")
    return BeautifulSoup(html, "html.parser")


def _abs_url(base: str, href: str) -> str:
    return href if href.startswith("http") else requests.compat.urljoin(base, href)


def _is_meta_pdf(url: str, label: str) -> bool:
    s = norm_text(url + " " + (label or ""))
    return any(w in s for w in ["index", "sommaire", "liste", "annexe", "grille", "formulaire"])


def _is_corrige_label(url: str, label: str) -> bool:
    s = norm_text(os.path.basename(url) + " " + (label or ""))
    return "corrig" in s or "correction" in s or "solution" in s


def _extract_geo_date(name: str) -> Tuple[str, str]:
    s = norm_text(name).replace(".pdf", "")

    zones = [
        "metro", "metropole", "amerique", "nord", "sud", "asie", "polynesie",
        "etranger", "centre", "antilles", "guyane", "caledonie", "nouvelle", "liban",
    ]

    geo = ""
    for z in zones:
        if z in s:
            geo = z
            break

    date_match = re.search(r"j[12][\s_]*\d{1,2}[\s_]*\d{1,2}", s)
    date_part = date_match.group(0) if date_match else ""

    return (geo, date_part)


def _match_score_strict(sujet_name: str, corrige_name: str) -> float:
    s_geo, s_date = _extract_geo_date(sujet_name)
    c_geo, c_date = _extract_geo_date(corrige_name)

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


def harvest_apmep(level: str, subject: str, years_back: int, volume_max: int) -> Dict[str, Any]:
    if level not in APMEP_ROOT_BY_LEVEL:
        raise ValueError(f"Niveau non supporté: {level}")

    root = APMEP_ROOT_BY_LEVEL[level]
    log(f"[HARVEST] scope={level}|{subject} root={root}")

    html = _http_get(root).text
    sp = _soup(html)

    year_links = []
    for a in sp.find_all("a"):
        href = a.get("href") or ""
        m = re.search(r"Annee-(20\d{2})", href)
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
                if _is_meta_pdf(absu, label):
                    continue
                pdf_links.append({
                    "url": absu,
                    "name": os.path.basename(absu),
                    "label": label,
                    "is_corrige": _is_corrige_label(absu, label),
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
                    score = _match_score_strict(s["name"], c["name"])
                    if score > best[1]:
                        best = (c, score)

                corrige = best[0] if best[1] >= 0.4 else None
                if corrige:
                    used_corr.add(corrige["url"])

                pair_id = f"PAIR_{level}|{subject}_{y}_{stable_id(s['name'], str(corrige['name'] if corrige else ''))}"
                item = {
                    "pair_id": pair_id,
                    "scope": f"{level}|{subject}",
                    "source": f"APMEP {y}",
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
        "country": st.session_state.country,
        "level": level,
        "subjects": [subject],
        "items_total": len(pairs),
        "items_corrige_ok": corrige_ok,
        "library": pairs,
    }


# -----------------------------------------------------------------------------
# Qi / RQi EXTRACTION — AMÉLORÉE
# -----------------------------------------------------------------------------
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦±×÷]")
_Q_MARKER_RE = re.compile(r"(?m)^\s*(\d{1,3}|[a-h]|[ivxIVX]{1,8})\s*[\)\.\-:]\s+")

_INTENT_RE = re.compile(
    r"\b("
    r"montrer|demontrer|prouver|justifier|determiner|calculer|resoudre|etudier|"
    r"donner|exprimer|simplifier|trouver|verifier|en\s*deduire|conclure|"
    r"etablir|calculer|tracer|representer|construire|completer|"
    r"show|prove|justify|determine|compute|solve|study|find|verify|deduce"
    r")\b",
    flags=re.IGNORECASE,
)


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


def _looks_like_question_unit(s: str) -> bool:
    if not s or len(s.strip()) < 30:
        return False
    t = s.strip()

    has_intent = bool(_INTENT_RE.search(t)) or ("?" in t)
    has_math = bool(_MATH_SYMBOL_RE.search(t)) or bool(re.search(r"\b\d+\b", t))

    return bool(has_intent or (has_math and len(t) > 50))


def _chunk_candidates(text: str) -> List[str]:
    t = (text or "").replace("\r", "\n")
    t = _dehyphenate(t)
    t = _merge_wrapped_lines(t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []

    starts = {0}
    for m in _Q_MARKER_RE.finditer(t):
        starts.add(m.start())

    idxs = sorted(starts)
    chunks = []
    for a, b in zip(idxs, idxs[1:] + [len(t)]):
        c = t[a:b].strip()
        c = re.sub(r"\s+", " ", c).strip()
        if c:
            chunks.append(c)

    if len(chunks) <= 3:
        paras = re.split(r"\n\n+", t)
        for p in paras:
            p = re.sub(r"\s+", " ", p).strip()
            if p and p not in chunks:
                chunks.append(p)

    return chunks


def split_questions(text: str, max_items: int = 200) -> Tuple[List[str], Dict[str, Any]]:
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

        if _looks_like_question_unit(c):
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


# -----------------------------------------------------------------------------
# ALIGN Qi↔RQi — ANTI-FAKE (V31.10.16)
# -----------------------------------------------------------------------------
def align_qi_rqi(qs: List[str], rs: List[str]) -> Tuple[List[Optional[int]], List[Dict[str, Any]]]:
    """
    Alignement Qi↔RQi amélioré avec seuils adaptatifs.
    IMPORTANT V31.10.16: AUCUN fallback séquentiel (anti fake-posables).
    Retour:
      - out: index RQi ou None
      - audits: liste d'audit par Qi (mode/score/fenêtre)
    """
    if not qs or not rs:
        return [None for _ in qs], [{"mode": "NO_RS", "score": 0.0, "j": None, "window": ""} for _ in qs]

    qtok = [tokenize(q) for q in qs]
    rtok = [tokenize(r) for r in rs]
    out: List[Optional[int]] = [None] * len(qs)
    audits: List[Dict[str, Any]] = [{"mode": "INIT", "score": 0.0, "j": None, "window": ""} for _ in qs]
    used_r = set()

    # Pass 1: fenêtre locale, seuil bas
    for i in range(len(qs)):
        best_j, best_s = None, 0.0
        lo, hi = max(0, i - 5), min(len(rs), i + 6)
        for j in range(lo, hi):
            if j in used_r:
                continue
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.06:
            out[i] = best_j
            used_r.add(best_j)
            audits[i] = {"mode": "LOCAL", "score": float(best_s), "j": int(best_j), "window": f"{lo}:{hi}"}

    # Pass 2: global pour non-matchés
    for i in range(len(qs)):
        if out[i] is not None:
            continue
        best_j, best_s = None, 0.0
        for j in range(len(rs)):
            if j in used_r:
                continue
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.08:
            out[i] = best_j
            used_r.add(best_j)
            audits[i] = {"mode": "GLOBAL", "score": float(best_s), "j": int(best_j), "window": "ALL"}
        else:
            audits[i] = {"mode": "NONE", "score": float(best_s), "j": (int(best_j) if best_j is not None else None), "window": "ALL"}

    return out, audits


# -----------------------------------------------------------------------------
# IA1 Proxy (déterministe) — trace d'opérations depuis RQi
# -----------------------------------------------------------------------------
_OP_RULES: List[Tuple[str, str, str]] = [
    (r"\bderiv", "OP_DERIVE", "Calculer une dérivée / variation"),
    (r"\blimit", "OP_LIMIT", "Calculer / encadrer une limite"),
    (r"\bintegr|primitive", "OP_INTEGRATE", "Calculer une intégrale / primitive"),
    (r"\brecur|recurrence|induction", "OP_INDUCTION", "Raisonnement par récurrence"),
    (r"\bdiscriminant|\bdelta\b", "OP_DISCRIMINANT", "Calculer un discriminant / racines"),
    (r"\bequation|inequation|systeme", "OP_SOLVE_EQUATION", "Résoudre équation / système"),
    (r"\bproba|probabil", "OP_PROBABILITY", "Calcul de probabilité"),
    (r"\b(binomial|normale|loi)\b", "OP_DISTRIBUTION", "Manipuler une loi de probabilité"),
    (r"\besperance|variance|ecart", "OP_MOMENTS", "Calculer espérance / variance"),
    (r"\bcomplexe|affixe|module|argument", "OP_COMPLEX", "Manipuler des nombres complexes"),
    (r"\bvecteur|orthogon|scalaire", "OP_VECTOR_GEOM", "Géométrie vectorielle / orthogonalité"),
    (r"\btrigo|sinus|cosinus|tangente", "OP_TRIGO", "Manipuler identités trigonométriques"),
    (r"\blog|ln|exp|exponent", "OP_LOGEXP", "Manipuler log/exp"),
    (r"\btableau\b.*\bvariat|\bvariat\b.*\btableau", "OP_VARIATION_TABLE", "Construire tableau de variations"),
    (r"\bencadr|major|minim|borne", "OP_BOUND", "Encadrer / majorer / minorer"),
    (r"\bmontrer|demontrer|prouver|justifier", "OP_PROVE", "Justifier / démontrer"),
]


def extract_op_trace(rqi_text: str, qi_text: str) -> List[Dict[str, Any]]:
    t = norm_text(rqi_text or "") + " " + norm_text(qi_text or "")
    steps = []
    used = set()
    for (pat, op, desc) in _OP_RULES:
        if re.search(pat, t, flags=re.IGNORECASE):
            if op not in used:
                used.add(op)
                steps.append({"op": op, "desc": desc})

    if not steps:
        steps = [{"op": "OP_STANDARD", "desc": "Méthode standard"}]

    return steps[:12]


def normalize_ari_steps(op_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for stp in op_trace or []:
        op = stp.get("op") or ""
        if op and op not in seen:
            seen.add(op)
            out.append(stp)
    return out


# -----------------------------------------------------------------------------
# TRIGGERS
# -----------------------------------------------------------------------------
def build_triggers(qi_text: str, op_trace: List[Dict[str, Any]]) -> List[str]:
    t = qi_text or ""
    out = []

    if "?" in t:
        out.append("TRG_QMARK")
    if _MATH_SYMBOL_RE.search(t):
        out.append("TRG_MATHSYM")
    if re.search(r"\b\d+\b", t):
        out.append("TRG_NUMBER")

    for stp in op_trace or []:
        op = stp.get("op")
        if op:
            out.append("TRG_" + op)

    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res[:12] if res else ["TRG_GENERIC"]


# -----------------------------------------------------------------------------
# CHAPTER MAPPING
# -----------------------------------------------------------------------------
def map_qi_to_chapter(qi_text: str, chapters: List[Dict[str, Any]]) -> str:
    if not chapters:
        return "UNMAPPED"
    qi_toks = set(tokenize(qi_text))
    best = ("UNMAPPED", 0.0)
    for ch in chapters:
        kws = ch.get("keywords") or []
        if not kws:
            continue
        ch_toks = set(tokenize(" ".join([str(x) for x in kws])))
        if not ch_toks:
            continue
        score = len(qi_toks & ch_toks) / max(1, len(ch_toks))
        if score > best[1]:
            best = (str(ch["chapter_code"]), score)
    return best[0] if best[1] >= 0.05 else "UNMAPPED"


# -----------------------------------------------------------------------------
# QC BUILDING — ZÉRO ORPHELIN SANS TRICHE (V31.10.16)
# -----------------------------------------------------------------------------
def qc_method_label(op_codes: List[str]) -> str:
    if not op_codes:
        return "Comment résoudre une question par méthode standard ?"
    core = " → ".join(op_codes[:3])
    return f"Comment résoudre en appliquant: {core} ?"


def sigma_similarity(qc_a: Dict[str, Any], qc_b: Dict[str, Any]) -> float:
    a_ops = qc_a.get("op_codes", [])
    b_ops = qc_b.get("op_codes", [])
    if a_ops and b_ops:
        s_ops = jaccard(list(a_ops), list(b_ops))
    else:
        s_ops = 0.0
    s_txt = jaccard(tokenize(qc_a.get("qc", "")), tokenize(qc_b.get("qc", "")))
    return max(s_ops, 0.7 * s_txt)


def build_qc_from_qi(
    qi_pack: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
    min_posable_global: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Build QC avec seuil adaptatif:
      - min_cluster_posable = 1 si peu de posable global
    V31.10.16:
      - NE JAMAIS laisser un bucket sans QC (zéro orphelin)
      - si posable_in_cluster=0 => qc_state="UNPOSABLE" (QC non posable mais mappage garanti)
    """
    total_posable = sum(1 for q in qi_pack if q.get("has_rqi"))
    min_cluster_posable = 1 if total_posable < 10 else 2
    log(f"[QC] total_posable={total_posable} → min_cluster_posable={min_cluster_posable}")

    buckets: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = {}
    for qi in qi_pack:
        cc = qi.get("chapter_code", "UNMAPPED")
        op_codes = tuple(qi.get("ari_norm_ops", []))
        buckets.setdefault((cc, op_codes), []).append(qi)

    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}

    for idx, ((cc, op_codes), items) in enumerate(sorted(buckets.items()), start=1):
        posable = [x for x in items if x.get("has_rqi")]

        # On produit toujours une QC; on marque l'état selon posable.
        qc_state = "POSABLE" if len(posable) >= 1 else "UNPOSABLE"

        # Seuil adaptatif (pour classification “POSABLE” robuste), mais production QC toujours.
        # Si posable>0 mais < min_cluster_posable: QC reste POSABLE mais faible.
        # Si posable==0: QC UNPOSABLE.
        qc_text = qc_method_label(list(op_codes))
        qc_id = f"QC_{stable_id(cc, qc_text, str(op_codes))}"

        trig_counts: Dict[str, int] = {}
        for it in items:
            for t in it.get("triggers", []):
                trig_counts[t] = trig_counts.get(t, 0) + 1
        trig_sorted = [k for (k, _) in sorted(trig_counts.items(), key=lambda x: (-x[1], x[0]))][:10]

        qc_obj = {
            "qc_id": qc_id,
            "qc": qc_text,
            "chapter_code": cc,
            "cluster_size": len(items),
            "posable_in_cluster": len(posable),
            "qc_state": qc_state,  # NEW
            "qi_ids": [x["qi_id"] for x in items],
            "op_codes": list(op_codes),
            "triggers_hint": trig_sorted,
            "cluster_gate": {
                "min_cluster_posable": int(min_cluster_posable),
                "posable_ok": bool(len(posable) >= min_cluster_posable),
            },
        }
        qc_pack.append(qc_obj)

        # Mappage garanti Qi→QC
        for it in items:
            qc_map[it["qi_id"]] = qc_id

    return qc_pack, qc_map


# -----------------------------------------------------------------------------
# F1 / F2
# -----------------------------------------------------------------------------
def compute_trigger_weights(triggers: List[str]) -> Dict[str, float]:
    uniq = list(dict.fromkeys(triggers or []))
    n = len(uniq)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in uniq}


def f1_raw(delta_c: float, epsilon: float, Tj: Dict[str, float], m_q: int) -> float:
    if not Tj:
        s = 0.0
    else:
        items = sorted(Tj.items(), key=lambda x: x[0])
        top = items[: max(0, int(m_q))]
        s = sum(v for _, v in top)
    return float(delta_c) * float((epsilon + s) ** 2)


def f1_normalize_in_chapter(qc_list: List[Dict[str, Any]]) -> None:
    if not qc_list:
        return
    raws = [float(q.get("psi_raw", 0.0) or 0.0) for q in qc_list]
    mx = max(raws) if raws else 0.0
    for q in qc_list:
        psi = float(q.get("psi_raw", 0.0) or 0.0)
        Psi = psi / mx if mx > 0 else 0.0
        q["Psi_q"] = float(Psi)


def f2_score(qc: Dict[str, Any], selected: List[Dict[str, Any]], n_q_historical: int, N_total: int, alpha: float, t_rec: float) -> Tuple[float, Dict[str, Any]]:
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
    selected: List[Dict[str, Any]] = []
    remaining = sorted(qc_list, key=lambda x: x.get("qc_id", ""))

    for _ in range(min(top_k, len(remaining))):
        best = None
        for qc in remaining:
            # si UNPOSABLE => n_q_hist = 0 (score nul), donc naturellement non sélectionnée
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


# -----------------------------------------------------------------------------
# SANITY GATE
# -----------------------------------------------------------------------------
def sanity_eval(doc_audits: List[Dict[str, Any]], qi_items: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    total_raw = sum(int(d.get("raw_chunks", 0) or 0) for d in doc_audits)
    total_kept = sum(int(d.get("kept", 0) or 0) for d in doc_audits)
    total_rej = sum(int(d.get("rejected", 0) or 0) for d in doc_audits)

    seen = set()
    dups = 0
    for q in qi_items:
        k = stable_id(norm_text(q.get("text", ""))[:300])
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


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
def run_phase(library: List[Dict[str, Any]], volume_pairs: int, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    to_process = exploitable[: max(1, min(int(volume_pairs), len(exploitable)))]
    log(f"[RUN] phase processing pairs={len(to_process)}")

    qi_items: List[Dict[str, Any]] = []
    rqi_items: List[Dict[str, Any]] = []
    doc_audits: List[Dict[str, Any]] = []
    doc_stats: List[Dict[str, Any]] = []

    align_audits_all: List[Dict[str, Any]] = []

    for pair in to_process:
        pid = pair["pair_id"]
        try:
            su_pdf = fetch_pdf_bytes(pair["sujet_url"])
            co_pdf = fetch_pdf_bytes(pair["corrige_url"])
        except Exception as e:
            log(f"[DL] FAILED {pid}: {e}")
            continue

        su_text = extract_text_from_pdf_bytes(su_pdf)
        co_text = extract_text_from_pdf_bytes(co_pdf)

        if not su_text or not co_text:
            log(f"[PDF] EMPTY {pid}")
            continue

        qs, qs_audit = split_questions(su_text, max_items=150)
        rs, rs_audit = split_questions(co_text, max_items=200)

        doc_audits.append({"pair_id": pid, "role": "SUJET", **qs_audit})
        doc_audits.append({"pair_id": pid, "role": "CORRIGE", **rs_audit})

        link, link_audits = align_qi_rqi(qs, rs)
        matched_count = sum(1 for x in link if x is not None)
        log(f"[ALIGN] {pid}: {len(qs)} Qi, {len(rs)} RQi, matched={matched_count}")

        doc_stats.append({"pair_id": pid, "qi": len(qs), "rqi": len(rs), "matched": matched_count})
        for i, ad in enumerate(link_audits[:500]):
            align_audits_all.append({"pair_id": pid, "qi_k": i + 1, **ad})

        for i, q in enumerate(qs):
            if not q.strip():
                continue
            j = link[i] if i < len(link) else None
            r = rs[j] if (j is not None and 0 <= j < len(rs)) else ""

            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
            rqi_id = f"RQI_{stable_id(pid, str(j), norm_text(r)[:180])}" if r else ""

            op_trace = extract_op_trace(r, q)
            ari_norm = normalize_ari_steps(op_trace)
            ari_ops = [x.get("op") for x in ari_norm if x.get("op")]

            triggers = build_triggers(q, op_trace)
            chapter_code = map_qi_to_chapter(q, chapters)

            qi_items.append({
                "qi_id": qi_id,
                "pair_id": pid,
                "k": i + 1,
                "text": q,
                "rqi_id": rqi_id,
                "chapter_code": chapter_code,
                "has_rqi": bool(r),
                "align_audit": link_audits[i] if i < len(link_audits) else {"mode": "NA", "score": 0.0, "j": None, "window": ""},
                "op_trace": op_trace,
                "ari_norm": ari_norm,
                "ari_norm_ops": ari_ops,
                "triggers": triggers,
            })
            if r:
                rqi_items.append({"rqi_id": rqi_id, "pair_id": pid, "k": i + 1, "text": r})

    sanity_ok, sanity_audit = sanity_eval(doc_audits, qi_items)

    return {
        "pairs_used": len(to_process),
        "qi_items": qi_items,
        "rqi_items": rqi_items,
        "doc_stats": doc_stats[:200],
        "doc_audits": doc_audits[:300],
        "align_audits": align_audits_all[:2000],
        "sanity_ok": sanity_ok,
        "sanity_audit": sanity_audit,
    }


def build_outputs_for_phase(phase_res: Dict[str, Any], chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    qi_items = phase_res["qi_items"]
    rqi_by_id = {r["rqi_id"]: r for r in phase_res["rqi_items"]}

    total_posable = sum(1 for q in qi_items if q.get("has_rqi"))
    qc_pack, qc_map = build_qc_from_qi(qi_items, chapters, min_posable_global=total_posable)

    qi_pack = []
    qi_posable = 0
    orphan_ids = []

    for q in qi_items:
        rqi_id = q.get("rqi_id") or ""
        rtxt = rqi_by_id.get(rqi_id, {}).get("text", "")

        qc_id = qc_map.get(q["qi_id"], "")
        # V31.10.16: qc_map est garanti; si jamais absent => orphelin réel (bug)
        is_orphan = not qc_id

        if q.get("has_rqi"):
            qi_posable += 1
        if is_orphan:
            orphan_ids.append(q["qi_id"])

        ari_obj = {
            "template": "ARI_SMAXIA_V3",
            "op_trace": q.get("op_trace", []),
            "normalized_steps": q.get("ari_norm", []),
            "m_q": len(q.get("ari_norm", [])),
            "evidence": {
                "has_rqi": bool(rtxt),
                "rqi_chars": len(rtxt),
                "qi_chars": len(q.get("text", "")),
                "align": q.get("align_audit", {}),
            },
        }

        frt_obj = {
            "template": "FRT_SMAXIA_V3",
            "source": "RQi" if rtxt else "MISSING_RQi",
            "sections": ["OBJ", "DAT", "MTH", "EXE", "RES", "VAL"],
        }

        qi_pack.append({
            "qi_id": q["qi_id"],
            "pair_id": q["pair_id"],
            "k": q["k"],
            "chapter_code": q.get("chapter_code", "UNMAPPED"),
            "qi": q["text"],
            "rqi": rtxt,
            "has_rqi": bool(rtxt),
            "qc_id": qc_id,
            "is_orphan": is_orphan,
            "triggers": q.get("triggers", []),
            "ari": ari_obj,
            "frt": frt_obj,
        })

    # F1 computation
    delta_by_ch = {c["chapter_code"]: float(c.get("delta_c", 1.0) or 1.0) for c in chapters}
    qi_by_id = {x["qi_id"]: x for x in qi_pack}
    qc_by_ch: Dict[str, List[Dict[str, Any]]] = {}

    for qc in qc_pack:
        cc = qc["chapter_code"]
        qc_by_ch.setdefault(cc, []).append(qc)

        m_list = []
        for qi_id in qc.get("qi_ids", []):
            qi = qi_by_id.get(qi_id)
            if qi and qi.get("has_rqi"):
                m_list.append(int(qi.get("ari", {}).get("m_q", 0) or 0))
        m_q = int(sorted(m_list)[len(m_list)//2]) if m_list else 1

        Tj = compute_trigger_weights(qc.get("triggers_hint", []))
        delta_c = float(delta_by_ch.get(cc, 1.0))
        psi = f1_raw(delta_c=delta_c, epsilon=F1_EPSILON, Tj=Tj, m_q=m_q)

        qc["m_q"] = m_q
        qc["Tj"] = Tj
        qc["delta_c"] = delta_c
        qc["psi_raw"] = psi
        qc["f1_audit"] = {"delta_c": delta_c, "epsilon": F1_EPSILON, "m_q": m_q, "psi_raw": psi}

        # IMPORTANT: UNPOSABLE => n_q_historical = 0 (score F2 nul)
        qc["n_q_historical"] = int(qc.get("posable_in_cluster", 0) or 0)

    for cc, qcs in qc_by_ch.items():
        f1_normalize_in_chapter(qcs)

    # Chapter report
    ch_label = {c["chapter_code"]: c.get("chapter_label", c["chapter_code"]) for c in chapters}
    qc_by_id = {q["qc_id"]: q for q in qc_pack}

    qi_by_chapter: Dict[str, List[str]] = {}
    qi_covered_by_chapter: Dict[str, List[str]] = {}
    qc_count_by_chapter: Dict[str, int] = {}

    for qc in qc_pack:
        qc_count_by_chapter[qc["chapter_code"]] = qc_count_by_chapter.get(qc["chapter_code"], 0) + 1

    for qi in qi_pack:
        qc_id = qi.get("qc_id", "")
        cc = qc_by_id[qc_id]["chapter_code"] if qc_id and qc_id in qc_by_id else qi.get("chapter_code", "UNMAPPED")
        qi_by_chapter.setdefault(cc, []).append(qi["qi_id"])
        if qi.get("has_rqi") and not qi.get("is_orphan"):
            qi_covered_by_chapter.setdefault(cc, []).append(qi["qi_id"])

    chapter_entries = []
    for cc in sorted(set(qi_by_chapter.keys()) | set(qc_count_by_chapter.keys())):
        qi_total = len(qi_by_chapter.get(cc, []))
        qi_cov = len(qi_covered_by_chapter.get(cc, []))
        qc_count = int(qc_count_by_chapter.get(cc, 0))
        cov_pct = round((qi_cov / qi_total) * 100, 2) if qi_total else 0.0
        chapter_entries.append({
            "chapter_code": cc,
            "chapter_label": ch_label.get(cc, cc),
            "qi_total": qi_total,
            "covered_qi_count": qi_cov,
            "coverage_pct": cov_pct,
            "qc_count": qc_count,
        })

    chapter_report = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": st.session_state.pack_id,
        "chapters": sorted(chapter_entries, key=lambda x: (-x["qi_total"], x["chapter_code"])),
    }

    audit = {
        "qi_total": len(qi_pack),
        "qi_posable": qi_posable,
        "rqi_total": sum(1 for q in qi_pack if q.get("rqi")),
        "qc_total": len(qc_pack),
        "qc_posable": sum(1 for q in qc_pack if q.get("qc_state") == "POSABLE"),
        "qc_unposable": sum(1 for q in qc_pack if q.get("qc_state") == "UNPOSABLE"),
        "qi_orphans": len(orphan_ids),
        "orphan_ids_sample": orphan_ids[:25],
        "sanity_ok": bool(phase_res.get("sanity_ok")),
        "sanity_audit": phase_res.get("sanity_audit"),
        "pairs_used": phase_res.get("pairs_used"),
        "doc_stats": phase_res.get("doc_stats"),
        "doc_audits_sample": phase_res.get("doc_audits"),
        "align_audits_sample": phase_res.get("align_audits"),
    }

    return {
        "qi_pack": qi_pack,
        "qc_pack": qc_pack,
        "chapter_report": chapter_report,
        "audit": audit,
    }


def run_granulo_test_iso(library: List[Dict[str, Any]], phase_a: int, phase_b: int, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Phase A
    a_res = run_phase(library, phase_a, chapters)
    a_out = build_outputs_for_phase(a_res, chapters)
    setA = sorted({q["qc_id"] for q in a_out["qc_pack"]})

    # Phase B
    b_res = run_phase(library, phase_b, chapters)
    b_out = build_outputs_for_phase(b_res, chapters)
    setB = sorted({q["qc_id"] for q in b_out["qc_pack"]})

    setB_minus_setA = sorted(list(set(setB) - set(setA)))
    saturation_ok = (len(setB_minus_setA) == 0)

    # Selection report
    qc_by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for qc in b_out["qc_pack"]:
        qc_by_ch.setdefault(qc["chapter_code"], []).append(qc)

    selection = []
    N_total = int(b_out["audit"]["qi_posable"] or b_out["audit"]["qi_total"] or 1)

    for cc, qcs in sorted(qc_by_ch.items()):
        selected = progressive_select(qcs, N_total=N_total, top_k=12)
        selection.append({
            "chapter_code": cc,
            "selected_qc_ids": [x["qc_id"] for x in selected],
            "top_items": [
                {
                    "qc_id": x["qc_id"],
                    "score": x.get("_f2_score"),
                    "Psi_q": x.get("Psi_q"),
                    "cluster_size": x.get("cluster_size"),
                    "posable_in_cluster": x.get("posable_in_cluster"),
                    "qc_state": x.get("qc_state"),
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

    # V31.10.16: orphans==0 est désormais atteignable sans triche (QC UNPOSABLE)
    # SEALED exige toujours des POSABLE réels + QC posables > 0
    sealed = bool(saturation_ok and sanity_ok and orphans == 0 and posable > 0 and qc_total > 0 and qc_posable > 0)

    audit = {
        "phaseA": a_out["audit"],
        "phaseB": b_out["audit"],
        "saturation_ok": saturation_ok,
        "setA_size": len(setA),
        "setB_size": len(setB),
        "setB_minus_setA_size": len(setB_minus_setA),
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


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
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
    st.caption("Corrections V31.10.16: anti-fake POSABLE, QC UNPOSABLE (zéro orphelin), audit alignement")

    with st.sidebar:
        st.markdown("## ACTIVATION")
        country = st.selectbox("Pays", [DEFAULT_COUNTRY])
        st.session_state.country = country

        up = st.file_uploader("Pack JSON (optionnel)", type=["json"])
        uploaded_pack = None
        if up:
            try:
                uploaded_pack = json.loads(up.read().decode("utf-8"))
            except Exception as e:
                st.error(f"Pack invalide: {e}")

        allow_genesis = st.checkbox("Autoriser Pack Genesis (TEST)", value=True)
        genesis_n = st.number_input("Genesis chapitres", 4, 12, 12)

        if st.button("ACTIVER", use_container_width=True):
            try:
                pack = load_academic_pack(country, allow_genesis, int(genesis_n), uploaded_pack)
                st.session_state.pack_active = pack
                st.session_state.pack_id = str(pack.get("pack_id") or "")
                st.session_state.pack_path = pack.get("_pack_file_path")
                st.session_state.pack_sig_sha256 = pack.get("_pack_sig_sha256")
                st.success("Pack actif")
            except Exception as e:
                st.error(f"Erreur: {e}")

        if st.session_state.pack_active:
            st.success(f"✅ {st.session_state.pack_id}")
            st.caption(f"Source: {st.session_state.pack_active.get('_source')}")
            st.caption(f"SHA256: {st.session_state.pack_sig_sha256[:16]}...")

        st.markdown("---")
        st.markdown("## SÉLECTION")
        level = st.radio("Niveau", ["Terminale", "Première", "Seconde"], index=0)
        st.session_state.level = {"Terminale": "TERMINALE", "Première": "PREMIERE", "Seconde": "SECONDE"}[level]

        if st.session_state.pack_active:
            st.markdown("### Chapitres")
            for c in pack_chapters(st.session_state.pack_active)[:8]:
                st.write(f"• {c['chapter_code']}")

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

        st.markdown("### HARVEST APMEP")
        c1, c2 = st.columns(2)
        years_back = c1.number_input("Années", 1, 15, 5)
        volume_max = c2.number_input("Volume max", 5, 100, 30)

        if st.button("HARVEST", use_container_width=True):
            if not st.session_state.pack_active:
                st.error("Pack inactif")
            else:
                try:
                    manifest = harvest_apmep(st.session_state.level, "MATH", int(years_back), int(volume_max))
                    st.session_state.harvest_manifest = manifest
                    st.session_state.library = manifest["library"]
                    st.session_state.sealed = False
                    st.session_state.qi_pack = None
                    st.session_state.qc_pack = None
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0, "orphans": 0, "sanity_ok": False}
                    st.success(f"HARVEST: {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                except Exception as e:
                    st.error(f"HARVEST: {e}")

    with tab2:
        st.markdown("## RUN")
        lib2 = st.session_state.library
        corr_ok2 = sum(1 for x in lib2 if x.get("corrige?") and x.get("corrige_url"))

        if not st.session_state.pack_active:
            st.error("Pack inactif")
        elif not lib2:
            st.warning("Bibliothèque vide")
        elif corr_ok2 <= 0:
            st.error("Aucun corrigé exploitable")
        else:
            chapters = pack_chapters(st.session_state.pack_active)
            max_vol = max(1, min(100, corr_ok2))
            c1, c2 = st.columns(2)
            phase_a = c1.slider("Phase A", 1, max_vol, min(5, max_vol))
            phase_b = c2.slider("Phase B", 1, max_vol, min(10, max_vol))

            if st.button("LANCER", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_granulo_test_iso(lib2, phase_a, phase_b, chapters)

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
                    f"{q['qc_id']} | {q['chapter_code']} | n={q['cluster_size']} | pos={q.get('posable_in_cluster',0)} | state={q.get('qc_state','')} | Ψ={round(float(q.get('Psi_q',0)),3)}"
                    for q in qcs
                ]
                sel_idx = st.selectbox("QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]

                st.markdown(f"### {qc['qc']}")
                st.write(f"qc_state: {qc.get('qc_state')}")
                st.write(f"op_codes: {qc.get('op_codes', [])}")

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
