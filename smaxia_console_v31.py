# =============================================================================
# SMAXIA GTE Console V31.10.14 — ISO-PROD TEST (MONO-FICHIER)
# =============================================================================
# OBJECTIF CEO:
# - UI + Moteur mono-fichier
# - Pack FILE signé (Genesis -> /tmp) ; aucun EMBEDDED preloaded
# - Harvest réel (APMEP)
# - Extraction Qi/RQi (anti sur-segmentation) + sanity qualitative (non arbitraire)
# - IA1 Proxy déterministe: ARI op-trace dérivée de RQi (pas de template vide)
# - QC: "méthode" (ce que la question fait) et non "signature surface"
# - Mapping chapitre data-driven (pack keywords) via vote des Qi
# - F1/F2 (formules scellées) + auditability
# - Saturation Set_A / Set_B: SEALED si (SetB - SetA == ∅) + sanity_ok + orphans=0 + posable>0
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
APP_VERSION = "V31.10.14"
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

# F1/F2 constants (deterministic; may be parameterized later)
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
    return re.findall(r"[a-z0-9]{3,}", norm_text(s))[:2000]


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


def _dehyphenate(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _looks_like_page_number(line: str) -> bool:
    s = norm_text(line)
    return bool(re.fullmatch(r"\d{1,3}", s) or re.fullmatch(r"page\s*\d{1,3}", s))


# -----------------------------------------------------------------------------
# PACK GENESIS (FILE signé) — Aucun EMBEDDED pack en mémoire
# -----------------------------------------------------------------------------
def _pack_genesis_fr_math(n_chapters: int) -> Dict[str, Any]:
    """
    Pack Genesis TEST: Chapitres 'réels' FR (nomenclature) + keywords,
    MAIS le CORE reste data-driven: mapping via keywords pack.
    delta_c laissé à 1.0 (peut être ajusté plus tard par data).
    """
    themes = [
        ("CH_001_ANALYSE", ["limite", "continuite", "derivee", "variation", "convexite", "integrale", "primitive"]),
        ("CH_002_PROBABILITES", ["probabilite", "loi", "binomiale", "normale", "esperance", "variance", "variable aleatoire"]),
        ("CH_003_SUITES", ["suite", "recurrence", "convergence", "arithmetique", "geometrique"]),
        ("CH_004_COMPLEXES", ["complexe", "affixe", "module", "argument", "imaginaire"]),
        ("CH_005_GEOMETRIE", ["vecteur", "plan", "espace", "droite", "orthogonal", "produit scalaire", "repere"]),
        ("CH_006_ALGORITHMES", ["algo", "algorithme", "python", "boucle", "fonction", "programme"]),
        ("CH_007_EQUATIONS", ["equation", "systeme", "inequation", "discriminant", "racine", "solution"]),
        ("CH_008_FONCTIONS", ["fonction", "courbe", "image", "antecedent", "composition"]),
        ("CH_009_TRIGO", ["sinus", "cosinus", "tangente", "trigonomet", "radian", "angle"]),
        ("CH_010_LOGEXP", ["logarithme", "ln", "exp", "exponentielle"]),
        ("CH_011_VECTMATR", ["matrice", "determinant", "vecteur", "base", "coordonnees"]),
        ("CH_012_STAT", ["statistique", "moyenne", "ecart", "mediane", "quartile", "nuage"]),
    ]
    themes = themes[: max(1, min(n_chapters, len(themes)))]

    return {
        "pack_id": "CAP_FR_GENESIS_V1",
        "country_code": "FR",
        "country_label": "France",
        "version": "1.0.0",
        "_source": "GENERATED_FILE",
        "chapters": [
            {
                "chapter_code": code,
                "chapter_label": code.replace("_", " "),
                "keywords": kws,
                "delta_c": 1.0,
            }
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
    """
    ISO-PROD rule (test mode):
    - In PROD: pack must come from FILE.
    - In Streamlit Cloud TEST: allow Genesis, but it MUST still be a FILE on disk with signature.
    """
    # 1) Uploaded pack has priority (still FILE logically)
    if uploaded_pack:
        tmp_path = _write_tmp_pack(uploaded_pack)
        sig = sha256_file(tmp_path)
        uploaded_pack["_source"] = "UPLOADED_FILE"
        uploaded_pack["_pack_file_path"] = tmp_path
        uploaded_pack["_pack_sig_sha256"] = sig
        log(f"[PACK] Loaded UPLOADED_FILE pack_id={uploaded_pack.get('pack_id')} path={tmp_path} sha256={sig}")
        return uploaded_pack

    # 2) Try real files in repo
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

    # 3) Genesis fallback (ONLY if allowed) — still written to /tmp and signed
    if allow_genesis:
        pack = _pack_genesis_fr_math(int(genesis_n_chapters))
        tmp_path = _write_tmp_pack(pack)
        sig = sha256_file(tmp_path)
        pack["_pack_file_path"] = tmp_path
        pack["_pack_sig_sha256"] = sig
        log(f"[PACK] Activated GENERATED_FILE (Genesis) path={tmp_path} sha256={sig} pack_id={pack.get('pack_id')}")
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
                out.append(
                    {
                        "chapter_code": code,
                        "chapter_label": label,
                        "keywords": item.get("keywords", []),
                        "delta_c": float(item.get("delta_c", 1.0) or 1.0),
                    }
                )
    return out


# -----------------------------------------------------------------------------
# HTTP + PDF
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


def _extract_pages_text(pdf_bytes: bytes, max_pages: int = 120) -> List[str]:
    pages = []
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages[:max_pages]:
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pages.append("")
            if any(pages):
                return pages
        except Exception:
            pass

    if PdfReader:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages[:max_pages]:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
        except Exception:
            pass
    return pages


def clean_pdf_text(pages: List[str]) -> str:
    if not pages:
        return ""

    page_lines = []
    for p in pages:
        p = (p or "").replace("\r", "\n")
        p = _dehyphenate(p)
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        page_lines.append(lines)

    # detect repeated headers/footers
    top_counts, bot_counts = {}, {}
    n_pages = len(page_lines)

    for lines in page_lines:
        for ln in lines[:3]:
            key = norm_text(ln)[:120]
            if key:
                top_counts[key] = top_counts.get(key, 0) + 1
        for ln in lines[-3:]:
            key = norm_text(ln)[:120]
            if key:
                bot_counts[key] = bot_counts.get(key, 0) + 1

    header_keys = {k for k, c in top_counts.items() if c >= max(3, int(0.6 * n_pages))}
    footer_keys = {k for k, c in bot_counts.items() if c >= max(3, int(0.6 * n_pages))}

    cleaned_pages = []
    for lines in page_lines:
        keep = []
        for ln in lines:
            k = norm_text(ln)[:120]
            if k in header_keys or k in footer_keys:
                continue
            if _looks_like_page_number(ln):
                continue
            keep.append(ln)
        cleaned_pages.append("\n".join(keep))

    text = "\n\n".join([p for p in cleaned_pages if p.strip()])
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    pages = _extract_pages_text(pdf_bytes)
    return clean_pdf_text(pages)


# -----------------------------------------------------------------------------
# HARVEST APMEP (volume réel)
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


def _filekey(name: str) -> str:
    s = norm_text(name)
    s = re.sub(r"\.pdf$", "", s)
    s = re.sub(r"\b(corrig(e|é|es)?|correction|solutions?)\b", "", s)
    return re.sub(r"[_\-]+", " ", s).strip()


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
                pdf_links.append(
                    {
                        "url": absu,
                        "name": os.path.basename(absu),
                        "label": label,
                        "is_corrige": _is_corrige_label(absu, label),
                    }
                )

            if not pdf_links:
                continue

            subjects = [p for p in pdf_links if not p["is_corrige"]]
            corriges = [p for p in pdf_links if p["is_corrige"]]

            corr_index = [(p, _filekey(p["name"]), tokenize(_filekey(p["name"]))) for p in corriges]
            used_corr = set()

            for s in subjects:
                if len(pairs) >= volume_max:
                    break

                skey = _filekey(s["name"])
                stok = tokenize(skey)

                best = (None, 0.0)
                for (c, _ckey, ctok) in corr_index:
                    if c["url"] in used_corr:
                        continue
                    score = jaccard(stok, ctok)
                    if score > best[1]:
                        best = (c, score)

                corrige = best[0] if best[1] >= 0.15 else None
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
                    "reason": "" if corrige else "corrigé absent",
                    "sujet_url": s["url"],
                    "corrige_url": corrige["url"] if corrige else "",
                }

                if not any(x["sujet_url"] == item["sujet_url"] for x in pairs):
                    pairs.append(item)
                    if item["corrige?"]:
                        corrige_ok += 1

            log(f"[HARVEST] year={y} pdfs={len(pdf_links)} sujets={len(subjects)} pairs={len(pairs)}")

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
# Qi / RQi extraction — orientation "définition Qi", pas "seuil quantité"
# -----------------------------------------------------------------------------
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦±×÷]")
_Q_MARKER_RE = re.compile(r"(?m)^\s*(\d{1,3}|[a-h]|[ivxIVX]{1,8})\s*[\)\.\-:]\s+")

# "Intent" verbs (FR + minimal EN); used to validate Qi as question-like unit.
_INTENT_RE = re.compile(
    r"\b("
    r"montrer|demontrer|prouver|justifier|determiner|calculer|resoudre|etudier|"
    r"donner|exprimer|simplifier|trouver|verifier|en\s+deduire|conclure|"
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
        if re.search(r"[.:;!?]$", buf):
            out.append(buf.strip())
            buf = ln.strip()
        else:
            buf = (buf + " " + ln.strip()).strip()
    if buf:
        out.append(buf.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def _looks_like_question_unit(s: str) -> bool:
    """Definition-oriented Qi validation: intent + constraint."""
    if not s or len(s.strip()) < 40:
        return False
    t = s.strip()

    has_intent = bool(_INTENT_RE.search(t)) or ("?" in t)
    has_constraint = bool(_MATH_SYMBOL_RE.search(t)) or bool(re.search(r"\b\d+\b", t)) or bool(re.search(r"\b(x|y|z|n)\b", norm_text(t)))

    # must have at least intent AND some constraint-like content
    return bool(has_intent and has_constraint)


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

    # fallback: split by question marks if numbering absent
    if len(chunks) <= 2:
        parts = re.split(r"(?<=\?)\s+", re.sub(r"\s+", " ", t))
        chunks = [p.strip() for p in parts if p.strip()]

    return chunks


def split_questions(text: str, max_items: int = 250) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns (items, audit)
    audit includes ratios for sanity gate.
    """
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


def align_qi_rqi(qs: List[str], rs: List[str]) -> List[Optional[int]]:
    if not qs or not rs:
        return [None for _ in qs]

    qtok = [tokenize(q) for q in qs]
    rtok = [tokenize(r) for r in rs]
    out = [None] * len(qs)

    # local window first
    for i in range(len(qs)):
        best_j, best_s = None, 0.0
        lo, hi = max(0, i - 6), min(len(rs), i + 7)
        for j in range(lo, hi):
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.10:
            out[i] = best_j

    # global for unmatched
    for i in range(len(qs)):
        if out[i] is not None:
            continue
        best_j, best_s = None, 0.0
        for j in range(len(rs)):
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.14:
            out[i] = best_j

    return out


# -----------------------------------------------------------------------------
# IA1 Proxy (déterministe) — trace d'opérations depuis RQi (pas un LLM)
# -----------------------------------------------------------------------------
# Keyword -> operator code (operators are cognitive, not chapter labels)
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
    (r"\btableau\b.*\bvariat|\bvariat\b.*\btableau", "OP_VARIATION_TABLE", "Construire un tableau de variations"),
    (r"\bencadr|major|minim|borne", "OP_BOUND", "Encadrer / majorer / minorer"),
    (r"\bmontrer|demontrer|prouver|justifier", "OP_PROVE", "Justifier / démontrer"),
]


def extract_op_trace(rqi_text: str, qi_text: str) -> List[Dict[str, Any]]:
    t = norm_text(rqi_text or "")
    steps = []
    used = set()
    for (pat, op, desc) in _OP_RULES:
        if re.search(pat, t, flags=re.IGNORECASE):
            if op not in used:
                used.add(op)
                steps.append({"op": op, "desc": desc})

    # Always include minimal cognitive scaffold if something exists
    if not steps and (rqi_text or "").strip():
        steps = [{"op": "OP_STANDARD", "desc": "Appliquer une méthode standard (trace faible)"}]

    # deterministically cap length
    return steps[:12]


def normalize_ari_steps(op_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    IA2-normalization proxy: deduplicate and keep order.
    This list defines m(q) for F1.
    """
    out = []
    seen = set()
    for stp in op_trace or []:
        op = stp.get("op") or ""
        if op and op not in seen:
            seen.add(op)
            out.append(stp)
    return out


# -----------------------------------------------------------------------------
# TRIGGERS (enrichis, toujours déterministes)
# -----------------------------------------------------------------------------
def build_triggers(qi_text: str, op_trace: List[Dict[str, Any]]) -> List[str]:
    t = qi_text or ""
    out = []

    # surface triggers (kept but not sufficient alone)
    if "?" in t:
        out.append("TRG_QMARK")
    if _MATH_SYMBOL_RE.search(t):
        out.append("TRG_MATHSYM")
    if re.search(r"\b\d+\b", t):
        out.append("TRG_NUMBER")

    # operator-based triggers (stronger)
    for stp in op_trace or []:
        op = stp.get("op")
        if op:
            out.append("TRG_" + op)

    # dedupe + cap
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res[:12] if res else ["TRG_GENERIC"]


# -----------------------------------------------------------------------------
# CHAPTER MAPPING (data-driven) — map Qi to chapter by keywords overlap
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
    return best[0] if best[1] >= 0.08 else "UNMAPPED"


def majority_vote(values: List[str]) -> str:
    if not values:
        return "UNMAPPED"
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    # tie-break by lexical order for determinism
    best = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best


# -----------------------------------------------------------------------------
# QC BUILDING — "méthode" = op_trace majoritaire, pas features surface
# -----------------------------------------------------------------------------
def qc_method_label(op_codes: List[str]) -> str:
    if not op_codes:
        return "Comment résoudre une question par une méthode standard ?"
    # keep first 3 ops for readability
    core = " → ".join(op_codes[:3])
    return f"Comment résoudre une question en appliquant: {core} ?"


def sigma_similarity(qc_a: Dict[str, Any], qc_b: Dict[str, Any]) -> float:
    """
    σ(q,p): similarity in [0,1]. Deterministic.
    Uses op_codes + tokens.
    """
    a_ops = qc_a.get("op_codes", [])
    b_ops = qc_b.get("op_codes", [])
    if a_ops and b_ops:
        s_ops = jaccard(list(a_ops), list(b_ops))
    else:
        s_ops = 0.0
    s_txt = jaccard(tokenize(qc_a.get("qc", "")), tokenize(qc_b.get("qc", "")))
    return max(s_ops, 0.7 * s_txt)


# -----------------------------------------------------------------------------
# F1 / F2 (formules scellées) — implémentation + audit
# -----------------------------------------------------------------------------
def compute_trigger_weights(triggers: List[str]) -> Dict[str, float]:
    """
    Deterministic T_j in [0,1]: weight by type and multiplicity not used (dedup triggers).
    We keep simple: each trigger contributes 1.0, then normalized by count.
    """
    uniq = []
    seen = set()
    for t in triggers or []:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    n = len(uniq)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in uniq}


def f1_raw(delta_c: float, epsilon: float, Tj: Dict[str, float], m_q: int) -> float:
    """
    ψ_raw(q) = δ_c * ( ε + Σ_{j=1}^{m(q)} T_j )^2
    Here m(q) = number of normalized ARI steps.
    We interpret Σ T_j as sum of top-m_q trigger weights (deterministic ordering).
    """
    if not Tj:
        s = 0.0
    else:
        # deterministic order
        items = sorted(Tj.items(), key=lambda x: x[0])
        top = items[: max(0, int(m_q))]
        s = sum(v for _, v in top)

    return float(delta_c) * float((epsilon + s) ** 2)


def f1_normalize_in_chapter(qc_list: List[Dict[str, Any]]) -> None:
    """
    Compute Ψ_q = ψ_raw(q) / max ψ_raw(p) within chapter.
    Mutates qc_list in-place with fields: psi_raw, Psi_q, f1_audit.
    """
    if not qc_list:
        return
    raws = [float(q.get("psi_raw", 0.0) or 0.0) for q in qc_list]
    mx = max(raws) if raws else 0.0
    for q in qc_list:
        psi = float(q.get("psi_raw", 0.0) or 0.0)
        Psi = psi / mx if mx > 0 else 0.0
        # domain: (0,1] expected if mx>0 and psi>0
        q["Psi_q"] = float(Psi)


def f2_score(qc: Dict[str, Any], selected: List[Dict[str, Any]], n_q_historical: int, N_total: int, alpha: float, t_rec: float) -> Tuple[float, Dict[str, Any]]:
    """
    Score(q) = (n_q_historical / N_total) * (1 + alpha / t_rec) * Psi_q * Π_{p∈S} ((1 - σ(q,p)) * 100)
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
        "alpha": alpha,
        "t_rec": t_rec,
        "Psi_q": Psi_q,
        "sigma_list": sigmas[:50],
        "prod_term": prod,
        "base_term": base,
    }
    return float(score), audit


def progressive_select(qc_list: List[Dict[str, Any]], N_total: int, top_k: int = 12) -> List[Dict[str, Any]]:
    """
    Progressive selection per chapter using F2 score.
    Deterministic tie-break by qc_id.
    """
    selected: List[Dict[str, Any]] = []
    remaining = sorted(qc_list, key=lambda x: x.get("qc_id", ""))

    for _ in range(min(top_k, len(remaining))):
        best = None
        for qc in remaining:
            n_q_hist = int(qc.get("n_q_historical", qc.get("cluster_size", 1)) or 1)
            sc, audit = f2_score(qc, selected, n_q_hist, N_total, F2_ALPHA, F2_TREC)
            qc["_f2_score"] = sc
            qc["_f2_audit"] = audit
            if best is None:
                best = qc
            else:
                # compare score then qc_id for determinism
                if (sc > best["_f2_score"]) or (sc == best["_f2_score"] and qc.get("qc_id","") < best.get("qc_id","")):
                    best = qc

        if best is None:
            break
        selected.append(best)
        remaining = [x for x in remaining if x.get("qc_id") != best.get("qc_id")]

    return selected


# -----------------------------------------------------------------------------
# SANITY GATE — qualitative (non arbitraire) + audit
# -----------------------------------------------------------------------------
def sanity_eval(doc_audits: List[Dict[str, Any]], qi_items: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """
    - No fixed 'Qi>60' rule.
    - Fail only if evidence of non-Qi fragments is strong.
    """
    total_raw = sum(int(d.get("raw_chunks", 0) or 0) for d in doc_audits)
    total_kept = sum(int(d.get("kept", 0) or 0) for d in doc_audits)
    total_rej = sum(int(d.get("rejected", 0) or 0) for d in doc_audits)

    # duplicate ratio (by normalized text hash)
    seen = set()
    dups = 0
    for q in qi_items:
        k = stable_id(norm_text(q.get("text",""))[:300])
        if k in seen:
            dups += 1
        else:
            seen.add(k)

    dup_ratio = dups / max(1, len(qi_items))

    # fail if:
    # - majority of chunks rejected (means segmentation produced non-questions)
    # - or duplicates explode
    # - or extreme explosion (safety for broken PDF extraction)
    rej_ratio = total_rej / max(1, total_raw) if total_raw else 0.0

    extreme_explosion = total_raw > 5000 and total_kept < 200  # indicates OCR/extraction disaster

    ok = True
    reasons = []
    if rej_ratio > 0.55 and total_raw > 200:
        ok = False
        reasons.append(f"reject_ratio_high({round(rej_ratio*100,2)}%)")
    if dup_ratio > 0.20 and len(qi_items) > 200:
        ok = False
        reasons.append(f"dup_ratio_high({round(dup_ratio*100,2)}%)")
    if extreme_explosion:
        ok = False
        reasons.append("extreme_explosion_pdf_extraction")

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
# RUN (Phase A / Phase B) — builds QC with F1/F2 + saturation sets
# -----------------------------------------------------------------------------
def build_qc_from_qi(qi_pack: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Build QC candidates by grouping Qi by (chapter_code, op_signature).
    This creates method-like QC (op-trace), not feature signatures.
    """
    buckets: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = {}
    for qi in qi_pack:
        cc = qi.get("chapter_code", "UNMAPPED")
        op_codes = tuple(qi.get("ari_norm_ops", []))
        buckets.setdefault((cc, op_codes), []).append(qi)

    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}

    for idx, ((cc, op_codes), items) in enumerate(sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])) , start=1):
        # anti-singleton normative: at least 2 POSABLE Qi to canonize a QC
        posable = [x for x in items if x.get("has_rqi")]
        if len(posable) < 2:
            continue

        qc_text = qc_method_label(list(op_codes))
        qc_id = f"QC_{stable_id(cc, qc_text)}"

        # triggers hint: most common triggers across cluster
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
            "qi_ids": [x["qi_id"] for x in items],
            "op_codes": list(op_codes),
            "triggers_hint": trig_sorted,
        }
        qc_pack.append(qc_obj)
        for it in items:
            qc_map[it["qi_id"]] = qc_id

    return qc_pack, qc_map


def run_phase(library: List[Dict[str, Any]], volume_pairs: int, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    to_process = exploitable[: max(1, min(int(volume_pairs), len(exploitable)))]
    log(f"[RUN] phase processing pairs={len(to_process)}")

    qi_items: List[Dict[str, Any]] = []
    rqi_items: List[Dict[str, Any]] = []
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

        su_text = extract_text_from_pdf_bytes(su_pdf)
        co_text = extract_text_from_pdf_bytes(co_pdf)

        if not su_text or not co_text:
            log(f"[PDF] EMPTY {pid}")
            continue

        qs, qs_audit = split_questions(su_text, max_items=260)
        rs, rs_audit = split_questions(co_text, max_items=320)  # allow more in correction

        doc_audits.append({"pair_id": pid, "role": "SUJET", **qs_audit})
        doc_audits.append({"pair_id": pid, "role": "CORRIGE", **rs_audit})

        link = align_qi_rqi(qs, rs)
        doc_stats.append({"pair_id": pid, "qi": len(qs), "rqi": len(rs)})

        for i, q in enumerate(qs):
            if not q.strip():
                continue
            j = link[i] if i < len(link) else None
            r = rs[j] if (j is not None and 0 <= j < len(rs)) else ""

            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
            rqi_id = f"RQI_{stable_id(pid, str(j), norm_text(r)[:180])}" if r else ""

            # IA1 Proxy: op-trace from RQi
            op_trace = extract_op_trace(r, q)
            ari_norm = normalize_ari_steps(op_trace)
            ari_ops = [x.get("op") for x in ari_norm if x.get("op")]

            triggers = build_triggers(q, op_trace)
            chapter_code = map_qi_to_chapter(q, chapters)

            qi_items.append(
                {
                    "qi_id": qi_id,
                    "pair_id": pid,
                    "k": i + 1,
                    "text": q,
                    "rqi_id": rqi_id,
                    "chapter_code": chapter_code,
                    "has_rqi": bool(r),
                    "op_trace": op_trace,
                    "ari_norm": ari_norm,
                    "ari_norm_ops": ari_ops,
                    "triggers": triggers,
                }
            )
            if r:
                rqi_items.append({"rqi_id": rqi_id, "pair_id": pid, "k": i + 1, "text": r})

    sanity_ok, sanity_audit = sanity_eval(doc_audits, qi_items)

    return {
        "pairs_used": len(to_process),
        "qi_items": qi_items,
        "rqi_items": rqi_items,
        "doc_stats": doc_stats[:200],
        "doc_audits": doc_audits[:300],
        "sanity_ok": sanity_ok,
        "sanity_audit": sanity_audit,
    }


def build_outputs_for_phase(phase_res: Dict[str, Any], chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    qi_items = phase_res["qi_items"]
    rqi_by_id = {r["rqi_id"]: r for r in phase_res["rqi_items"]}

    # Orphans are defined at QC mapping step
    qc_pack, qc_map = build_qc_from_qi(qi_items, chapters)

    # Enforce mapping completeness: each Qi must converge to a QC (axiome test)
    qi_pack = []
    qi_posable = 0
    orphan_ids = []

    for q in qi_items:
        rqi_id = q.get("rqi_id") or ""
        rtxt = rqi_by_id.get(rqi_id, {}).get("text", "")
        qc_id = qc_map.get(q["qi_id"], "")
        is_orphan = not qc_id

        if q.get("has_rqi"):
            qi_posable += 1
        if is_orphan:
            orphan_ids.append(q["qi_id"])

        # ARI object: include op_trace + normalized steps
        ari_obj = {
            "template": "ARI_SMAXIA_V3_DETERMINISTIC",
            "op_trace": q.get("op_trace", []),
            "normalized_steps": q.get("ari_norm", []),
            "m_q": len(q.get("ari_norm", [])),
            "evidence": {"has_rqi": bool(rtxt), "rqi_chars": len(rtxt), "qi_chars": len(q.get("text",""))},
        }

        # FRT object: evidence-driven scaffold (no hallucinated content)
        frt_obj = {
            "template": "FRT_SMAXIA_V3_EVIDENCE_DRIVEN",
            "rule": "FRT_SOURCE_RULE",
            "source": "RQi_clean" if rtxt else "MISSING_RQi",
            "sections": [
                {"id": "OBJ", "name": "OBJECTIF", "evidence": "from Qi intent"},
                {"id": "DAT", "name": "DONNÉES/CONTRAINTES", "evidence": "from Qi + RQi"},
                {"id": "MTH", "name": "MÉTHODE", "evidence": "from ARI normalized steps"},
                {"id": "EXE", "name": "EXÉCUTION", "evidence": "from RQi (trace)"},
                {"id": "RES", "name": "RÉSULTAT", "evidence": "from RQi"},
                {"id": "VAL", "name": "VALIDATION", "evidence": "from RQi checks"},
            ],
            "anchors": {
                "qi_excerpt": (q.get("text","")[:300] + "…") if len(q.get("text","")) > 300 else q.get("text",""),
                "rqi_excerpt": (rtxt[:400] + "…") if len(rtxt) > 400 else rtxt,
            },
        }

        qi_pack.append(
            {
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
            }
        )

    # Compute F1 within each chapter on QC pack
    # Need delta_c from pack chapter
    delta_by_ch = {c["chapter_code"]: float(c.get("delta_c", 1.0) or 1.0) for c in chapters}

    # We define for each QC:
    # - m(q): median m_q of Qi in cluster (normalized steps length)
    # - Tj: from triggers_hint (deterministic weights)
    # - psi_raw, then Psi_q normalized intra-chapter
    qi_by_id = {x["qi_id"]: x for x in qi_pack}
    qc_by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for qc in qc_pack:
        cc = qc["chapter_code"]
        qc_by_ch.setdefault(cc, []).append(qc)

        # gather m_q from member Qi
        m_list = []
        for qi_id in qc.get("qi_ids", []):
            qi = qi_by_id.get(qi_id)
            if qi and qi.get("has_rqi"):
                m_list.append(int(qi.get("ari", {}).get("m_q", 0) or 0))
        m_q = int(sorted(m_list)[len(m_list)//2]) if m_list else 1

        # triggers weights
        Tj = compute_trigger_weights(qc.get("triggers_hint", []))
        delta_c = float(delta_by_ch.get(cc, 1.0))
        psi = f1_raw(delta_c=delta_c, epsilon=F1_EPSILON, Tj=Tj, m_q=m_q)

        qc["m_q"] = m_q
        qc["Tj"] = Tj
        qc["delta_c"] = delta_c
        qc["psi_raw"] = psi
        qc["f1_audit"] = {
            "delta_c": delta_c,
            "epsilon": F1_EPSILON,
            "m_q": m_q,
            "Tj": Tj,
            "psi_raw": psi,
        }

        # n_q_historical (for test): use posable_in_cluster
        qc["n_q_historical"] = int(qc.get("posable_in_cluster", qc.get("cluster_size", 1)) or 1)

    # Normalize Psi_q per chapter
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
        cc = qc_by_id[qc_id]["chapter_code"] if qc_id and qc_id in qc_by_id else qi.get("chapter_code","UNMAPPED")
        qi_by_chapter.setdefault(cc, []).append(qi["qi_id"])
        if qi.get("has_rqi") and not qi.get("is_orphan"):
            qi_covered_by_chapter.setdefault(cc, []).append(qi["qi_id"])

    chapter_entries = []
    for cc in sorted(set(qi_by_chapter.keys()) | set(qc_count_by_chapter.keys())):
        qi_total = len(qi_by_chapter.get(cc, []))
        qi_cov = len(qi_covered_by_chapter.get(cc, []))
        qc_count = int(qc_count_by_chapter.get(cc, 0))
        cov_pct = round((qi_cov / qi_total) * 100, 2) if qi_total else 0.0
        chapter_entries.append(
            {
                "chapter_code": cc,
                "chapter_label": ch_label.get(cc, cc),
                "qi_total": qi_total,
                "covered_qi_count": qi_cov,
                "coverage_pct": cov_pct,
                "qc_count": qc_count,
            }
        )

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
        "qi_orphans": len(orphan_ids),
        "orphan_ids_sample": orphan_ids[:25],
        "sanity_ok": bool(phase_res.get("sanity_ok")),
        "sanity_audit": phase_res.get("sanity_audit"),
        "pairs_used": phase_res.get("pairs_used"),
        "doc_stats": phase_res.get("doc_stats"),
        "doc_audits_sample": phase_res.get("doc_audits"),
    }

    return {
        "qi_pack": qi_pack,
        "qc_pack": qc_pack,
        "chapter_report": chapter_report,
        "audit": audit,
    }


def run_granulo_test_iso(library: List[Dict[str, Any]], phase_a: int, phase_b: int, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Two phases: A then B.
    SEALED only if SetB - SetA == empty AND IA2-like gates pass.
    """
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

    # Progressive selection report (per chapter) using Phase B QC
    qc_by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for qc in b_out["qc_pack"]:
        qc_by_ch.setdefault(qc["chapter_code"], []).append(qc)

    selection = []
    N_total = int(b_out["audit"]["qi_posable"] or b_out["audit"]["qi_total"] or 1)

    for cc, qcs in sorted(qc_by_ch.items(), key=lambda x: x[0]):
        selected = progressive_select(qcs, N_total=N_total, top_k=12)
        selection.append(
            {
                "chapter_code": cc,
                "selected_qc_ids": [x["qc_id"] for x in selected],
                "top_items": [
                    {
                        "qc_id": x["qc_id"],
                        "score": x.get("_f2_score"),
                        "Psi_q": x.get("Psi_q"),
                        "cluster_size": x.get("cluster_size"),
                        "posable_in_cluster": x.get("posable_in_cluster"),
                        "op_codes": x.get("op_codes"),
                        "f2_audit": x.get("_f2_audit"),
                    }
                    for x in selected
                ],
            }
        )

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

    sealed = bool(saturation_ok and sanity_ok and orphans == 0 and posable > 0 and qc_total > 0)

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


def df_table(rows):
    if not rows:
        st.info("Bibliothèque vide")
        return
    cols = ["pair_id", "year", "sujet", "corrige?", "corrige_name", "sujet_url"]
    st.dataframe([{k: r.get(k, "") for k in cols} for r in rows], use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title=f"SMAXIA GTE {APP_VERSION}", layout="wide")
    ss_init()

    st.markdown(f"# SMAXIA GTE Console {APP_VERSION} — ISO-PROD TEST")
    st.caption("UI + Moteur mono-fichier. Pack FILE signé. Preuve F1/F2 + saturation Set_B-Set_A.")

    # Sidebar: Activation + Selection scope
    with st.sidebar:
        st.markdown("## ÉTAPE 1 — ACTIVATION (Pack FILE signé)")
        country = st.selectbox("Pays", [DEFAULT_COUNTRY], index=0)
        st.session_state.country = country

        st.markdown("### Upload Pack JSON (optionnel)")
        up = st.file_uploader("Pack JSON", type=["json"])
        uploaded_pack = None
        if up is not None:
            try:
                uploaded_pack = json.loads(up.read().decode("utf-8"))
            except Exception as e:
                st.error(f"Pack invalide: {e}")
                uploaded_pack = None

        st.markdown("### Autoriser Pack Genesis (TEST)")
        allow_genesis = st.checkbox("Autoriser Pack Genesis (TEST)", value=True)
        genesis_n = st.number_input("Genesis: nombre de chapitres", 4, 12, 12)

        if st.button("ACTIVER", use_container_width=True):
            try:
                pack = load_academic_pack(country, allow_genesis=allow_genesis, genesis_n_chapters=int(genesis_n), uploaded_pack=uploaded_pack)
                st.session_state.pack_active = pack
                st.session_state.pack_id = str(pack.get("pack_id") or "")
                st.session_state.pack_path = pack.get("_pack_file_path")
                st.session_state.pack_sig_sha256 = pack.get("_pack_sig_sha256")
                st.success("Pack actif")
            except Exception as e:
                st.session_state.pack_active = None
                st.error(f"Erreur: {e}")

        if st.session_state.pack_active:
            st.success(f"Pack: {st.session_state.pack_id}")
            st.caption(f"Source: {st.session_state.pack_active.get('_source', 'FILE')}")
            st.caption(f"Signature SHA256: {st.session_state.pack_sig_sha256}")
            st.caption(f"Path: {st.session_state.pack_path}")
        else:
            st.warning("Pack inactif")

        st.markdown("---")
        st.markdown("## ÉTAPE 2 — SÉLECTION (scope)")
        level = st.radio("Niveau", ["Seconde", "Première", "Terminale"], index=2)
        st.session_state.level = {"Seconde": "SECONDE", "Première": "PREMIERE", "Terminale": "TERMINALE"}[level]

        st.markdown("Matière (test)")
        _ = st.selectbox("Matière", ["MATH"], index=0)

        st.markdown("### Chapitres (pack-driven)")
        if st.session_state.pack_active:
            chs = pack_chapters(st.session_state.pack_active)
            st.caption(f"{len(chs)} chapitres")
            for c in chs:
                st.write(f"• {c['chapter_code']}")

    tab1, tab2, tab3 = st.tabs(["Import", "RUN", "Exports"])

    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?") and x.get("corrige_url"))

    # --- IMPORT ---
    with tab1:
        st.markdown("## Import / Bibliothèque")
        metric_row(
            len(lib),
            corr_ok,
            st.session_state.run_stats.get("qi", 0),
            st.session_state.run_stats.get("qi_posable", 0),
            st.session_state.run_stats.get("qc", 0),
            st.session_state.sealed,
        )

        st.markdown("### Bibliothèque")
        df_table(lib)

        st.markdown("---")
        st.markdown("## HARVEST AUTO (APMEP)")
        c1, c2 = st.columns(2)
        years_back = c1.number_input("Années", 1, 15, 10)
        volume_max = c2.number_input("Volume max (pairs)", 5, 200, 50)

        if st.button("HARVEST", use_container_width=True):
            if not st.session_state.pack_active:
                st.error("Activez le pack d'abord")
            else:
                try:
                    manifest = harvest_apmep(st.session_state.level, "MATH", int(years_back), int(volume_max))
                    st.session_state.harvest_manifest = manifest
                    st.session_state.library = manifest["library"]

                    # reset run state
                    st.session_state.sealed = False
                    st.session_state.qi_pack = None
                    st.session_state.qc_pack = None
                    st.session_state.chapter_report = None
                    st.session_state.selection_report = None
                    st.session_state.last_run_audit = None
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0, "orphans": 0, "sanity_ok": False}

                    st.success(f"HARVEST OK: {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                except Exception as e:
                    st.error(f"HARVEST échoué: {e}")

        st.markdown("---")
        st.markdown("## Upload manuel (Sujets/Corrigés)")
        cL, cR = st.columns(2)
        up_s = cL.file_uploader("PDF Sujet", type=["pdf"], key="up_s")
        up_c = cR.file_uploader("PDF Corrigé", type=["pdf"], key="up_c")
        if st.button("Ajouter à la bibliothèque"):
            if not up_s:
                st.error("PDF Sujet manquant")
            else:
                sid = stable_id(up_s.name, _utc_ts())
                su_url = f"local://sujet/{sid}"
                st.session_state._uploads[su_url] = up_s.read()

                corr_ok_flag = False
                co_url = ""
                co_name = ""
                if up_c:
                    co_url = f"local://corrige/{sid}"
                    st.session_state._uploads[co_url] = up_c.read()
                    corr_ok_flag = True
                    co_name = up_c.name

                row = {
                    "pair_id": f"PAIR_LOCAL_{sid}",
                    "scope": f"{st.session_state.level}|MATH",
                    "source": "UPLOAD",
                    "year": 0,
                    "sujet": up_s.name,
                    "corrige?": corr_ok_flag,
                    "corrige_name": co_name,
                    "reason": "" if corr_ok_flag else "corrigé absent",
                    "sujet_url": su_url,
                    "corrige_url": co_url,
                }
                st.session_state.library = [row] + st.session_state.library
                st.success("Ajouté")

    # --- RUN ---
    with tab2:
        st.markdown("## RUN (preuve F1/F2 + saturation)")
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

            max_vol = max(1, min(200, corr_ok2))
            c1, c2 = st.columns(2)
            phase_a = c1.slider("Phase A (pairs)", 1, max_vol, min(10, max_vol))
            phase_b = c2.slider("Phase B (pairs)", 1, max_vol, min(20, max_vol))

            if st.button("LANCER", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_granulo_test_iso(lib2, phase_a, phase_b, chapters)

                    # Keep phaseB as main packs for explorer/exports
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

                    st.info(
                        f"SEALED = {'YES' if sealed else 'NO'} | "
                        f"saturation_ok={res['audit']['saturation_ok']} | "
                        f"sanity_ok={res['audit']['phaseB']['sanity_ok']} | "
                        f"orphans={res['audit']['phaseB']['qi_orphans']} | "
                        f"posable={res['audit']['phaseB']['qi_posable']}"
                    )

                    if not res["audit"]["phaseB"]["sanity_ok"]:
                        st.error("SANITY FAIL détecté (qualitatif)")

                except Exception as e:
                    st.error(f"RUN échoué: {e}")

        st.markdown("### Logs")
        st.text_area("", logs_text(), height=320)

    # --- EXPORTS + EXPLORER ---
    with tab3:
        st.markdown("## Exports")
        metric_row(
            len(st.session_state.library),
            sum(1 for x in st.session_state.library if x.get("corrige?") and x.get("corrige_url")),
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

        # saturation sets if exist
        if st.session_state.last_run_audit:
            # rebuild from audit structure? (kept in logs for UI; not stored sets here)
            pass

        st.markdown("---")
        st.markdown("## Explorateur QC → Qi → (RQi, Triggers, F1, ARI, FRT)")

        if st.session_state.qc_pack and st.session_state.qi_pack:
            ch_options = ["ALL"] + sorted({qc["chapter_code"] for qc in st.session_state.qc_pack})
            sel_ch = st.selectbox("Chapitre", ch_options)

            qcs = st.session_state.qc_pack if sel_ch == "ALL" else [q for q in st.session_state.qc_pack if q["chapter_code"] == sel_ch]

            if qcs:
                # sort by cluster_size desc
                qcs = sorted(qcs, key=lambda x: (-int(x.get("cluster_size",0)), x.get("qc_id","")))
                qc_labels = [
                    f"{q['qc_id']} | {q['chapter_code']} | n={q['cluster_size']} | Ψ={round(float(q.get('Psi_q',0.0)),3)}"
                    for q in qcs
                ]
                sel_idx = st.selectbox("QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]

                st.markdown(f"### QC: {qc['qc']}")
                st.caption(f"qc_id={qc['qc_id']} | chapter={qc['chapter_code']} | cluster_size={qc['cluster_size']}")
                st.write(f"op_codes={qc.get('op_codes', [])}")
                st.write(f"triggers_hint={qc.get('triggers_hint', [])}")

                st.markdown("#### F1 (audit)")
                st.json(qc.get("f1_audit", {}))

                qi_by_id = {q["qi_id"]: q for q in st.session_state.qi_pack}

                st.markdown("#### Qi associées (max 40)")
                for qi_id in qc.get("qi_ids", [])[:40]:
                    q = qi_by_id.get(qi_id)
                    if not q:
                        continue

                    with st.expander(f"{qi_id} | has_rqi={q.get('has_rqi')} | orphan={q.get('is_orphan')}"):
                        st.markdown("**Qi (énoncé)**")
                        st.write(q["qi"][:1200])

                        st.markdown("**RQi (correction)**")
                        st.write(q["rqi"][:1200] if q.get("rqi") else "— Pas de RQi alignée —")

                        st.markdown("**Triggers**")
                        st.write(q.get("triggers", []))

                        st.markdown("**ARI**")
                        st.json(q.get("ari", {}))

                        st.markdown("**FRT**")
                        st.json(q.get("frt", {}))

        st.markdown("---")
        st.markdown("## Audit (dernier RUN)")
        if st.session_state.last_run_audit:
            st.json(st.session_state.last_run_audit)


if __name__ == "__main__":
    main()
