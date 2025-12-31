# =============================================================================
# SMAXIA GTE Console V31.10.13 — ISO-PROD TEST (MONO-FICHIER)
# =============================================================================
# OBJECTIF CEO:
# - UI + Moteur dans un seul fichier (copier/coller => smaxia_console_v31.py)
# - Pack FILE signé (upload) OU Pack Genesis (TEST) => généré sur disque /tmp
# - Harvest réel APMEP + extraction PDF + Qi/RQi + ARI/FRT/Triggers + QC
# - Mapping QC->Chapitre (cluster-driven, pas QC-text driven)
# - Preuve de saturation: Set_A (Phase A) vs Set_B (Phase B), SEALED si Set_B-Set_A=∅
# - Zéro régression UI: Import / RUN / Exports + Explorateur QC -> Qi -> (RQi, Triggers, F1, ARI, FRT)
#
# NOTE:
# - Pas d’IA payante: "IA1 Proxy" = heuristiques déterministes sur RQi pour ARI "réel".
# - F1/F2 propriétaires: ici "audit shell" + métriques; remplacer par F1/F2 exactes dès que fournies.
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import glob
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


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
APP_VERSION = "V31.10.13"
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


# -----------------------------------------------------------------------------
# SESSION
# -----------------------------------------------------------------------------
def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ss_init():
    st.session_state.setdefault("pack_active", None)
    st.session_state.setdefault("pack_id", None)
    st.session_state.setdefault("pack_source", None)
    st.session_state.setdefault("pack_path", None)
    st.session_state.setdefault("pack_sha256", None)

    st.session_state.setdefault("country", DEFAULT_COUNTRY)
    st.session_state.setdefault("level", DEFAULT_LEVEL)
    st.session_state.setdefault("subjects", DEFAULT_SUBJECTS[:])

    st.session_state.setdefault("library", [])
    st.session_state.setdefault("harvest_manifest", None)

    st.session_state.setdefault("qi_pack", None)
    st.session_state.setdefault("qc_pack", None)
    st.session_state.setdefault("chapter_report", None)
    st.session_state.setdefault("last_run_audit", None)

    st.session_state.setdefault("sealed", False)
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("run_stats", {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0})

    st.session_state.setdefault("_uploads", {})          # local://pdf cache
    st.session_state.setdefault("_http_pdf_cache", {})   # url -> bytes
    st.session_state.setdefault("_pack_upload_bytes", None)

    # RUN phase A/B tracking
    st.session_state.setdefault("setA_qc_ids", [])
    st.session_state.setdefault("setB_qc_ids", [])
    st.session_state.setdefault("set_diff", [])
    st.session_state.setdefault("saturation_ok", False)


def log(msg: str):
    st.session_state.logs.append(f"[{_utc_ts()}] {msg}")


def logs_text() -> str:
    return "\n".join(st.session_state.logs[-9000:])


# -----------------------------------------------------------------------------
# UTIL: HASH / NORMALIZE
# -----------------------------------------------------------------------------
def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:12]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b or b"").hexdigest()


# -----------------------------------------------------------------------------
# PACK: FILE SIGNÉ OU GENESIS (TEST) -> FILE /tmp
# -----------------------------------------------------------------------------
def _safe_read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_pack_paths(country: str) -> List[str]:
    candidates = [
        os.path.join(os.getcwd(), "academic_packs", f"CAP_{country}_*.json"),
        os.path.join(os.getcwd(), "packs", f"CAP_{country}_*.json"),
        os.path.join(os.getcwd(), f"CAP_{country}_*.json"),
    ]
    out = []
    for pat in candidates:
        out.extend(glob.glob(pat))
    out2, seen = [], set()
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            out2.append(ap)
    return out2


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
                })
    return out


def _write_pack_to_tmp(pack: Dict[str, Any]) -> Tuple[str, str]:
    """
    Ecrit un pack JSON sur /tmp, calcule sha256 et renvoie (path, sha256).
    """
    pack_id = str(pack.get("pack_id") or "CAP_UNKNOWN").strip()
    suffix = stable_id(pack_id, _utc_ts(), str(time.time()))
    path = f"/tmp/smaxia_pack_{pack_id}_{suffix}.json"
    b = json.dumps(pack, ensure_ascii=False, indent=2).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b)
    return path, sha256_bytes(b)


def _pack_genesis_fr(genesis_n_chapters: int = 12) -> Dict[str, Any]:
    """
    Pack Genesis TEST (France) avec chapitres réels (noms + keywords)
    Important: pack généré à la demande (pas EMBEDDED en mémoire par défaut).
    """
    base = [
        ("ANALYSE", "Analyse (Fonctions, limites, dérivation, intégration)",
         ["fonction", "limite", "derivee", "dérivée", "continuite", "continuité", "integrale", "intégrale", "primitive", "variation", "tableau"]),
        ("PROBABILITES", "Probabilités (lois, espérance, variance)",
         ["probabilite", "probabilité", "loi", "variable", "aleatoire", "aléatoire", "esperance", "espérance", "variance", "binomiale", "normale"]),
        ("SUITES", "Suites et récurrence",
         ["suite", "recurrence", "récurrence", "convergence", "arithmetique", "arithmétique", "geometrique", "géométrique"]),
        ("COMPLEXES", "Nombres complexes",
         ["complexe", "affixe", "module", "argument", "imaginaire", "reel", "réel"]),
        ("GEOMETRIE", "Géométrie (plan, espace, vecteurs)",
         ["vecteur", "plan", "espace", "droite", "orthogonal", "scalaire", "repere", "repère"]),
        ("ALGORITHMES", "Algorithmique / raisonnement / logique",
         ["algorithme", "programme", "boucle", "condition", "preuve", "raisonnement"]),
        ("EQUATIONS", "Équations / inéquations",
         ["equation", "équation", "inequation", "inégalité", "discriminant", "racine", "solutions"]),
        ("FONCTIONS", "Étude de fonctions",
         ["courbe", "variations", "extremum", "maximum", "minimum", "derivee", "dérivée"]),
        ("TRIGO", "Trigonométrie",
         ["sinus", "cosinus", "tangente", "angle", "radian", "trigonometrie", "trigonométrie"]),
        ("LOGEXP", "Logarithme / exponentielle",
         ["logarithme", "ln", "exp", "exponentielle", "exponentielle", "croissance", "decroissance"]),
        ("VECTMATR", "Vecteurs / matrices (si présents)",
         ["matrice", "determinant", "déterminant", "vecteur", "coordonnees", "coordonnées"]),
        ("STAT", "Statistiques / estimation (si présents)",
         ["statistique", "moyenne", "ecart", "écart", "estimation", "intervalle", "confiance"]),
    ]

    # Ajuste à genesis_n_chapters
    base = base[:max(4, min(genesis_n_chapters, len(base)))]

    chapters = []
    for i, (code, label, kws) in enumerate(base, start=1):
        chapters.append({
            "chapter_code": f"CH_{i:03d}_{code}",
            "chapter_label": label,
            "keywords": kws
        })

    pack = {
        "pack_id": "CAP_FR_GENESIS_V1",
        "country_code": "FR",
        "country_label": "France",
        "version": "1.0.0",
        "_source": "GENERATED_FILE",
        "chapters": chapters,
    }
    return pack


def load_academic_pack(country: str, allow_genesis: bool, genesis_n_chapters: int,
                      uploaded_pack_bytes: Optional[bytes]) -> Dict[str, Any]:
    """
    Règle ISO-PROD:
    - Pas d'embedded pack "préchargé".
    - On consomme un PACK FILE (upload) OU un PACK FILE généré (Genesis) sur /tmp.
    """
    # 1) Upload JSON => écrire sur /tmp, signer, charger depuis file
    if uploaded_pack_bytes:
        try:
            b = uploaded_pack_bytes
            obj = json.loads(b.decode("utf-8"))
            # normalisation minimale
            if not (obj.get("pack_id") and isinstance(obj.get("chapters"), list) and len(obj["chapters"]) > 0):
                raise RuntimeError("Pack upload invalide: pack_id/chapters manquants")
            path, sig = _write_pack_to_tmp(obj)
            log(f"[PACK] Activated UPLOADED_FILE path={path} sha256={sig} pack_id={obj.get('pack_id')}")
            obj["_pack_file_path"] = path
            obj["_source"] = "UPLOADED_FILE"
            st.session_state.pack_path = path
            st.session_state.pack_sha256 = sig
            return obj
        except Exception as e:
            raise RuntimeError(f"Pack upload invalide: {e}")

    # 2) Si Genesis autorisé (TEST), générer pack -> /tmp
    if allow_genesis and country == "FR":
        pack = _pack_genesis_fr(genesis_n_chapters)
        path, sig = _write_pack_to_tmp(pack)
        log(f"[PACK] Activated GENERATED_FILE (Genesis) path={path} sha256={sig} pack_id={pack.get('pack_id')}")
        pack["_pack_file_path"] = path
        pack["_source"] = "GENERATED_FILE"
        st.session_state.pack_path = path
        st.session_state.pack_sha256 = sig
        return pack

    # 3) Sinon: chercher un pack présent dans le repo
    paths = _candidate_pack_paths(country)
    log(f"[PACK] Searching for country={country} | candidates={len(paths)}")
    for path in paths[:40]:
        try:
            pack = _safe_read_json(path)
            pack_id = str(pack.get("pack_id") or pack.get("id") or "").strip()
            chapters = pack.get("chapters") or []
            if pack_id and isinstance(chapters, list) and len(chapters) > 0:
                b = json.dumps(pack, ensure_ascii=False, indent=2).encode("utf-8")
                sig = sha256_bytes(b)
                log(f"[PACK] Loaded FILE path={path} sha256={sig} pack_id={pack_id}")
                pack["_pack_file_path"] = path
                pack["_source"] = "FILE"
                st.session_state.pack_path = path
                st.session_state.pack_sha256 = sig
                return pack
        except Exception as e:
            log(f"[PACK] Failed: {path} | {e}")

    raise RuntimeError(f"Aucun pack trouvé pour {country}. Upload requis ou activer Genesis (TEST).")


# -----------------------------------------------------------------------------
# TEXT UTILS / TOKENS / SIMILARITY
# -----------------------------------------------------------------------------
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_MATH_SYMBOL_RE = re.compile(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦±×÷]")

def tokenize(s: str) -> List[str]:
    s = norm_text(s)
    toks = re.findall(r"[a-z]{3,}|\d+(?:[.,]\d+)?", s)
    return toks[:1600]

def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _num_set(s: str) -> set:
    return set(_NUM_RE.findall((s or "").replace(",", ".")))

def score_q_r(q: str, r: str) -> float:
    tq, tr = tokenize(q), tokenize(r)
    j = jaccard(tq, tr)
    nq, nr = _num_set(q), _num_set(r)
    num_overlap = (len(nq & nr) / max(1, len(nq))) if nq else 0.0
    return 0.70 * j + 0.30 * num_overlap

def _has_math_or_constraint(s: str) -> bool:
    if "?" in (s or ""):
        return True
    if _MATH_SYMBOL_RE.search(s or ""):
        return True
    if re.search(r"\b\d+\b", s or "") and re.search(r"[=<>≤≥]", s or ""):
        return True
    # verbes de consigne courants
    if re.search(r"\b(montrer|demontrer|démontrer|calculer|determiner|déterminer|resoudre|résoudre|etudier|étudier|justifier)\b", norm_text(s)):
        return True
    return False


# -----------------------------------------------------------------------------
# PDF CLEANING
# -----------------------------------------------------------------------------
def _dehyphenate(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def _looks_like_page_number(line: str) -> bool:
    s = norm_text(line)
    return bool(re.fullmatch(r"\d{1,3}", s) or re.fullmatch(r"page\s*\d{1,3}", s))

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

def _extract_pages_text(pdf_bytes: bytes, max_pages: int = 90) -> List[str]:
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
# SEGMENTATION: SUJET vs CORRIGÉ (différenciée)
# -----------------------------------------------------------------------------
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

_Q_START_RE = re.compile(r"(?m)^\s*(\d{1,2}|[a-h]|[ivxIVX]{1,6})\s*[\)\.\-:]\s+")
_INLINE_QNUM_RE = re.compile(r"(?<!\w)(\(?\s*(\d{1,2}|[a-h]|[ivxIVX]{1,6})\s*\)?\s*[)\.\-:])\s+")

def _resplit_inline_numbering(chunk: str) -> List[str]:
    s = (chunk or "").strip()
    if len(s) < 40:
        return [s] if s else []
    matches = list(_INLINE_QNUM_RE.finditer(s))
    if len(matches) <= 1:
        return [s]
    idxs = [m.start() for m in matches] + [len(s)]
    out = []
    for a, b in zip(idxs, idxs[1:]):
        part = s[a:b].strip()
        if part:
            out.append(part)
    return out

def split_questions_sujet(raw_text: str) -> List[str]:
    t = (raw_text or "").replace("\r", "\n")
    t = _dehyphenate(t)
    t = _merge_wrapped_lines(t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []

    starts = {0}
    for m in _Q_START_RE.finditer(t):
        starts.add(m.start())

    idxs = sorted(starts)
    chunks = []
    for a, b in zip(idxs, idxs[1:] + [len(t)]):
        c = t[a:b].strip()
        # SUJET: filtre "vraie consigne"
        if len(c) >= 50 and _has_math_or_constraint(c):
            chunks.append(re.sub(r"\s+", " ", c).strip())

    # Fallback: split par '?'
    if len(chunks) <= 2:
        parts = re.split(r"(?<=\?)\s+", re.sub(r"\s+", " ", t))
        chunks = [p.strip() for p in parts if len(p.strip()) >= 60 and _has_math_or_constraint(p)]

    out, seen = [], set()
    for c in chunks:
        for p in _resplit_inline_numbering(c):
            p = p.strip()
            if len(p) < 40:
                continue
            k = stable_id(norm_text(p)[:400])
            if k in seen:
                continue
            seen.add(k)
            out.append(p[:2000] + ("…" if len(p) > 2000 else ""))

    return out[:250]

def split_solutions(raw_text: str) -> List[str]:
    t = (raw_text or "").replace("\r", "\n")
    t = _dehyphenate(t)
    t = _merge_wrapped_lines(t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []

    parts = re.split(r"(?mi)(?=^\s*(exercice|partie|question)\b)|(?=^\s*(\d{1,2}|[a-h]|[ivx]{1,6})\s*[\)\.\-:]\s+)", t)

    buf = []
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        p = re.sub(r"\s+", " ", p).strip()
        # CORRIGE: plus permissif
        if len(p) >= 35:
            buf.append(p)

    out = []
    for c in buf:
        out.extend([x.strip() for x in _resplit_inline_numbering(c) if x.strip()])

    final, seen = [], set()
    for c in out:
        k = stable_id(norm_text(c)[:300])
        if k in seen:
            continue
        seen.add(k)
        final.append(c[:2500] + ("…" if len(c) > 2500 else ""))

    return final[:450]


def sanity_qi_segments(qs: List[str]) -> Tuple[bool, Dict[str, Any]]:
    if not qs:
        return False, {"reason": "empty"}

    lens = sorted([len(x) for x in qs if x.strip()])
    med = lens[len(lens)//2] if lens else 0

    good = 0
    for q in qs:
        if len(q) >= 40 and _has_math_or_constraint(q):
            good += 1
    good_ratio = good / max(1, len(qs))

    ok = (good_ratio >= 0.55 and med >= 60)
    return ok, {"good_ratio": round(good_ratio, 3), "median_len": med, "n": len(qs)}


def align_qi_rqi(qs: List[str], rs: List[str]) -> List[Optional[int]]:
    if not qs or not rs:
        return [None for _ in qs]

    out = [None] * len(qs)

    # fenêtre locale
    for i in range(len(qs)):
        best_j, best_s = None, 0.0
        lo, hi = max(0, i - 10), min(len(rs), i + 11)
        for j in range(lo, hi):
            s = score_q_r(qs[i], rs[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.10:
            out[i] = best_j

    # global fallback
    for i in range(len(qs)):
        if out[i] is not None:
            continue
        best_j, best_s = None, 0.0
        for j in range(len(rs)):
            s = score_q_r(qs[i], rs[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= 0.13:
            out[i] = best_j

    return out


# -----------------------------------------------------------------------------
# CLUSTERING: signature + F2_split (anti-fusion)
# -----------------------------------------------------------------------------
def _feature_signature(text: str) -> Tuple[str, ...]:
    s = text or ""
    feats = []
    if "?" in s:
        feats.append("F_QMARK")
    if _MATH_SYMBOL_RE.search(s):
        feats.append("F_MATHSYM")
    if re.search(r"\b\d+[,\.]\d+\b", s):
        feats.append("F_DECIMAL")
    if re.search(r"\b\d+\b", s):
        feats.append("F_NUMBER")
    if re.search(r"[∑∏∫]", s):
        feats.append("F_AGGREGATE")
    if re.search(r"[≤≥<>]", s):
        feats.append("F_RELATION")
    if re.search(r"\b(derivee|dérivée|integrale|intégrale|limite|suite|recurrence|récurrence|probabilite|probabilité|complexe)\b", norm_text(s)):
        feats.append("F_KEYWORD")
    if not feats:
        feats.append("F_GENERIC")
    return tuple(sorted(set(feats)))

def cluster_by_signature(items: List[Dict[str, Any]], min_cluster_size: int) -> List[List[Dict[str, Any]]]:
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for it in items:
        sig = _feature_signature(it.get("text", ""))
        it["_sig"] = sig
        buckets.setdefault(sig, []).append(it)

    clusters, small = [], []
    for sig, bucket in buckets.items():
        if len(bucket) >= max(1, min_cluster_size):
            clusters.append(bucket)
        else:
            small.extend(bucket)

    if small:
        clusters.append(small)

    clusters = [sorted(c, key=lambda x: x["qi_id"]) for c in clusters]
    clusters.sort(key=lambda c: (-len(c), c[0]["qi_id"]))
    return clusters

def f2_split_clusters(clusters: List[List[Dict[str, Any]]], max_cluster_soft: int = 250) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Anti-fusion déterministe:
    - si un cluster est énorme, on le découpe par sous-signature enrichie (présence de certains tokens pivots).
    Ce n'est pas "F2 propriétaire", c'est un mécanisme de test pour éviter des clusters massifs inutilisables.
    """
    new_clusters = []
    splits = 0

    pivot_patterns = [
        ("PIV_DERIVE", re.compile(r"\b(dérivée|derivee|variation|tableau)\b", re.I)),
        ("PIV_LIM", re.compile(r"\b(limite|tend vers)\b", re.I)),
        ("PIV_INT", re.compile(r"\b(intégrale|integrale|primitive|aire)\b", re.I)),
        ("PIV_PROBA", re.compile(r"\b(probabilit|binom|loi|variable aléatoire|esperance|variance)\b", re.I)),
        ("PIV_SUITES", re.compile(r"\b(suite|récurrence|recurrence|convergence)\b", re.I)),
        ("PIV_COMP", re.compile(r"\b(complexe|affixe|argument|module)\b", re.I)),
    ]

    for c in clusters:
        if len(c) <= max_cluster_soft:
            new_clusters.append(c)
            continue

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for it in c:
            txt = it.get("text", "")
            key = "PIV_OTHER"
            for name, pat in pivot_patterns:
                if pat.search(txt or ""):
                    key = name
                    break
            buckets.setdefault(key, []).append(it)

        if len(buckets) <= 1:
            new_clusters.append(c)
        else:
            for _, b in buckets.items():
                new_clusters.append(sorted(b, key=lambda x: x["qi_id"]))
            splits += 1

    audit = {"splits": splits, "clusters_in": len(clusters), "clusters_out": len(new_clusters), "max_cluster_soft": max_cluster_soft}
    new_clusters.sort(key=lambda c: (-len(c), c[0]["qi_id"]))
    return new_clusters, audit


# -----------------------------------------------------------------------------
# TRIGGERS / IA1 PROXY (ARI réel déterministe) / FRT / F1-AUDIT SHELL
# -----------------------------------------------------------------------------
def build_triggers(qi_text: str) -> List[str]:
    t = qi_text or ""
    out = []
    if "?" in t:
        out.append("TRG_QMARK")
    if _MATH_SYMBOL_RE.search(t):
        out.append("TRG_MATHSYM")
    if re.search(r"\b\d+\b", t):
        out.append("TRG_NUMBER")
    if re.search(r"[∑∏∫]", t):
        out.append("TRG_AGGREGATE")
    if re.search(r"[≤≥<>]", t):
        out.append("TRG_RELATION")
    if not out:
        out = ["TRG_GENERIC"]
    return out[:8]


class SmaxiaExpertAgent:
    """
    IA1 Proxy (sans LLM): extraction d'opérateurs cognitifs à partir de la RQi.
    Résultat: ARI "réel" (liste d'opérations) auditables.
    """
    OPS = [
        ("OP_DERIVATE", re.compile(r"\b(dérivée|derivee)\b", re.I), "Calcul de dérivée / étude de variations."),
        ("OP_LIMIT", re.compile(r"\b(limite|tend vers)\b", re.I), "Calcul de limite / comportement asymptotique."),
        ("OP_TABLEAU", re.compile(r"\b(tableau|variat)\b", re.I), "Construction tableau (variations / signes)."),
        ("OP_INTEGRAL", re.compile(r"\b(intégrale|integrale|primitive|aire)\b", re.I), "Calcul d'intégrale / primitive / aire."),
        ("OP_DISCRIMINANT", re.compile(r"\b(discriminant|delta)\b", re.I), "Calcul discriminant / racines."),
        ("OP_SOLVE_EQ", re.compile(r"\b(résoudre|resoudre|solutions?)\b", re.I), "Résolution d'équation / système."),
        ("OP_INDUCTION", re.compile(r"\b(récurrence|recurrence|hérédité|initialisation)\b", re.I), "Preuve par récurrence."),
        ("OP_PROBA", re.compile(r"\b(probabilit|binom|loi|variable aléatoire|esperance|variance)\b", re.I), "Calcul probabilités / lois / espérance."),
        ("OP_COMPLEX", re.compile(r"\b(complexe|affixe|argument|module)\b", re.I), "Manipulation nombres complexes."),
        ("OP_GEO", re.compile(r"\b(vecteur|orthogon|scalaire|plan|espace)\b", re.I), "Géométrie / vecteurs / produit scalaire."),
    ]

    @staticmethod
    def extract_real_ari(rqi_text: str, qi_text: str) -> Dict[str, Any]:
        t = (rqi_text or "").strip()
        if not t:
            # si pas de RQi, ARI minimal: lecture + plan + exécution + check
            return {
                "template": "ARI_SMAXIA_PROXY_V1",
                "steps": [
                    {"id": "S1", "op": "OP_READ", "desc": "Extraire données et contraintes de l'énoncé."},
                    {"id": "S2", "op": "OP_PLAN", "desc": "Choisir une stratégie de résolution."},
                    {"id": "S3", "op": "OP_EXECUTE", "desc": "Appliquer les transformations."},
                    {"id": "S4", "op": "OP_CHECK", "desc": "Vérifier cohérence / unités / domaine."},
                    {"id": "S5", "op": "OP_CONCLUDE", "desc": "Formuler la réponse finale."},
                ],
                "evidence": {"has_rqi": False, "qi_chars": len(qi_text or "")},
                "complexity": 5
            }

        steps = []
        for op, pat, desc in SmaxiaExpertAgent.OPS:
            if pat.search(t):
                steps.append({"id": f"S{len(steps)+1}", "op": op, "desc": desc})

        if not steps:
            steps = [{"id": "S1", "op": "OP_STANDARD", "desc": "Application d'un théorème / méthode standard."}]

        return {
            "template": "ARI_SMAXIA_PROXY_V1",
            "steps": steps,
            "evidence": {"has_rqi": True, "ops_found": [s["op"] for s in steps], "rqi_chars": len(t)},
            "complexity": len(steps)
        }


def build_frt(qi_text: str, rqi_text: str, triggers: List[str], ari: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "template": "FRT_V2_INVARIANT",
        "sections": [
            {"section_id": "SEC_OBJ", "name": "OBJECTIF", "desc": "But extrait de l'énoncé"},
            {"section_id": "SEC_DAT", "name": "DONNÉES", "desc": "Valeurs, contraintes, hypothèses"},
            {"section_id": "SEC_MTH", "name": "MÉTHODE", "desc": "Stratégie sélectionnée"},
            {"section_id": "SEC_EXE", "name": "EXÉCUTION", "desc": "Transformations pas à pas"},
            {"section_id": "SEC_RES", "name": "RÉSULTAT", "desc": "Réponse finale"},
            {"section_id": "SEC_VAL", "name": "VALIDATION", "desc": "Vérification cohérence"},
        ],
        "triggers_applied": triggers,
        "ari_ops": [s.get("op") for s in (ari.get("steps") or [])][:12],
        "has_rqi": bool(rqi_text),
    }


def compute_f1_audit(qi_text: str, triggers: List[str], ari: Dict[str, Any]) -> Dict[str, Any]:
    """
    F1 audit shell (remplacer par F1 exacte dès que disponible).
    But: auditability + déterminisme (mêmes inputs => mêmes sorties).
    """
    trig_map = {}
    for t in triggers or []:
        trig_map[t] = round((len(t) % 17) / 17.0 + (len(qi_text or "") % 23) / 100.0, 6)

    psi_raw = round(sum(trig_map.values()), 6)
    psi_q = round(max(trig_map.values()) if trig_map else 0.0, 6)

    return {
        "delta_C": len(set(triggers or [])),
        "m_q": len(triggers or []),
        "triggers_Tj": trig_map,
        "psi_raw": psi_raw,
        "Psi_q": psi_q,
        "ari_complexity": ari.get("complexity"),
    }


def build_qc_label(signature: Tuple[str, ...], triggers: List[str]) -> str:
    sig_txt = " ".join(signature[:6]) if signature else "F_GENERIC"
    t_txt = " ".join((triggers or [])[:4])
    # règle QC: commence par Comment, finit par ?
    return f"Comment résoudre une question caractérisée par: {sig_txt} T::{t_txt} ?"


def map_cluster_to_chapter(cluster_qi_texts: List[str], chapters: List[Dict[str, Any]]) -> str:
    if not chapters:
        return "UNMAPPED"
    blob = " ".join(cluster_qi_texts[:80])
    qi_toks = set(tokenize(blob))

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

            corr_index = [(p, _filekey(p["name"]), tokenize(_filekey(p["name"]))) for p in corriges]
            used_corr = set()

            for s in subjects:
                if len(pairs) >= volume_max:
                    break

                skey = _filekey(s["name"])
                stok = tokenize(skey)

                best = (None, 0.0)
                for (c, ckey, ctok) in corr_index:
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
# RUN CORE: extraction + clusters + QC + packs + audit
# -----------------------------------------------------------------------------
def run_granulo_once(library: List[Dict[str, Any]], volume_pairs: int, min_cluster_size: int,
                    pack: Dict[str, Any], max_cluster_soft: int) -> Dict[str, Any]:

    chapters = pack_chapters(pack)
    if not chapters:
        raise RuntimeError("Pack invalide: chapitres vides")

    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    if not exploitable:
        raise RuntimeError("Aucun corrigé exploitable")

    to_process = exploitable[:max(1, min(volume_pairs, len(exploitable)))]
    log(f"[RUN] processing={len(to_process)} pairs")

    qi_items, rqi_items = [], []
    doc_stats, suspicious_docs = [], []

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

        qs = split_questions_sujet(su_text)
        rs = split_solutions(co_text)

        ok, info = sanity_qi_segments(qs)
        if not ok:
            suspicious_docs.append({"pair_id": pid, "qi": len(qs), "rqi": len(rs), "note": f"Segmentation instable: {info}"})

        link = align_qi_rqi(qs, rs)
        doc_stats.append({"pair_id": pid, "qi": len(qs), "rqi": len(rs)})

        for i, q in enumerate(qs):
            if not (q or "").strip():
                continue
            j = link[i] if i < len(link) else None
            r = rs[j] if (j is not None and 0 <= j < len(rs)) else ""

            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
            rqi_id = f"RQI_{stable_id(pid, str(j), norm_text(r)[:180])}" if r else ""

            qi_items.append({"qi_id": qi_id, "pair_id": pid, "k": i + 1, "text": q, "rqi_id": rqi_id})
            if r:
                rqi_items.append({"rqi_id": rqi_id, "pair_id": pid, "k": i + 1, "text": r})

    if not qi_items:
        raise RuntimeError("Aucune Qi extraite")

    # Clustering initial
    clusters = cluster_by_signature(qi_items, min_cluster_size)
    # F2 split anti-fusion
    clusters2, f2_audit = f2_split_clusters(clusters, max_cluster_soft=max_cluster_soft)

    qc_pack, qc_map = [], {}

    # QC build
    for cidx, cluster in enumerate(clusters2, start=1):
        sig = cluster[0].get("_sig", ("F_GENERIC",))
        # triggers "dominants" du cluster (simple: union limité)
        trig_union = []
        for it in cluster[:60]:
            trig_union.extend(build_triggers(it.get("text", "")))
        trig_union = list(dict.fromkeys(trig_union))[:6]

        qc_text = build_qc_label(sig, trig_union)
        qc_id = f"QC_{stable_id(str(cidx), norm_text(qc_text))}"
        chapter_code = map_cluster_to_chapter([x.get("text", "") for x in cluster], chapters)

        qc_pack.append({
            "qc_id": qc_id,
            "qc": qc_text,
            "chapter_code": chapter_code,
            "cluster_size": len(cluster),
            "signature": list(sig),
            "triggers_hint": trig_union,
            "qi_ids": [x["qi_id"] for x in cluster],
        })
        for x in cluster:
            qc_map[x["qi_id"]] = qc_id

    # Build Qi pack (avec ARI proxy + F1 audit shell + FRT)
    rqi_by_id = {r["rqi_id"]: r for r in rqi_items}
    qi_pack = []
    qi_posable, orphan_count, orphan_ids = 0, 0, []

    for q in qi_items:
        rqi_id = q.get("rqi_id") or ""
        rtxt = rqi_by_id.get(rqi_id, {}).get("text", "")

        triggers = build_triggers(q["text"])
        ari = SmaxiaExpertAgent.extract_real_ari(rtxt, q["text"])
        f1 = compute_f1_audit(q["text"], triggers, ari)
        frt = build_frt(q["text"], rtxt, triggers, ari)

        qc_id = qc_map.get(q["qi_id"], "")
        is_orphan = not qc_id

        if is_orphan:
            orphan_count += 1
            orphan_ids.append(q["qi_id"])
        if rtxt:
            qi_posable += 1

        qi_pack.append({
            "qi_id": q["qi_id"],
            "pair_id": q["pair_id"],
            "k": q["k"],
            "qi": q["text"],
            "rqi": rtxt,
            "has_rqi": bool(rtxt),
            "qc_id": qc_id,
            "is_orphan": is_orphan,
            "triggers": triggers,
            "f1_audit": f1,
            "ari": ari,
            "frt": frt,
        })

    # Chapter report
    ch_label = {c["chapter_code"]: c.get("chapter_label", c["chapter_code"]) for c in chapters}
    qc_by_chapter = {}
    qc_by_id = {}
    for qc in qc_pack:
        qc_by_chapter.setdefault(qc["chapter_code"], []).append(qc)
        qc_by_id[qc["qc_id"]] = qc

    qi_by_chapter = {}
    qi_covered_by_chapter = {}

    for qi in qi_pack:
        qc_id = qi.get("qc_id", "")
        cc = qc_by_id[qc_id]["chapter_code"] if qc_id and qc_id in qc_by_id else "UNMAPPED"
        qi_by_chapter.setdefault(cc, []).append(qi["qi_id"])
        if qi.get("has_rqi") and not qi.get("is_orphan"):
            qi_covered_by_chapter.setdefault(cc, []).append(qi["qi_id"])

    chapter_entries = []
    for cc in sorted(set(qc_by_chapter.keys()) | set(qi_by_chapter.keys())):
        qi_total = len(qi_by_chapter.get(cc, []))
        qi_covered = len(qi_covered_by_chapter.get(cc, []))
        qc_count = len(qc_by_chapter.get(cc, []))
        cov_pct = round((qi_covered / qi_total) * 100, 2) if qi_total > 0 else 0.0

        chapter_entries.append({
            "chapter_code": cc,
            "chapter_label": ch_label.get(cc, cc),
            "qi_total": qi_total,
            "covered_qi_count": qi_covered,
            "coverage_pct": cov_pct,
            "qc_count": qc_count,
        })

    chapter_report = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": st.session_state.pack_id,
        "chapters": sorted(chapter_entries, key=lambda x: -x["qi_total"]),
    }

    # SANITY global: basé sur la segmentation instable (pas sur quantité brute)
    sanity_ok = (len(suspicious_docs) == 0)

    audit = {
        "qi_total": len(qi_pack),
        "qi_posable": qi_posable,
        "rqi_total": sum(1 for q in qi_pack if q.get("rqi")),
        "qc_total": len(qc_pack),
        "qi_orphans": orphan_count,
        "sanity_ok": sanity_ok,
        "suspicious_docs": suspicious_docs,
        "doc_stats": doc_stats[:120],
        "f2_audit": f2_audit,
        "unmapped_qc": sum(1 for q in qc_pack if q.get("chapter_code") == "UNMAPPED"),
    }

    return {
        "qi_pack": qi_pack,
        "qc_pack": qc_pack,
        "chapter_report": chapter_report,
        "audit": audit,
        "qc_ids": [q["qc_id"] for q in qc_pack],
    }


def run_granulo_test_AB(library: List[Dict[str, Any]], phaseA_pairs: int, phaseB_pairs: int,
                        min_cluster_size: int, pack: Dict[str, Any], max_cluster_soft: int) -> Dict[str, Any]:
    """
    Preuve de saturation:
    - Run A => Set_A QC IDs
    - Run B => Set_B QC IDs
    - saturation_ok si Set_B - Set_A est vide
    """
    # Phase A
    log(f"[SAT] Phase A pairs={phaseA_pairs}")
    resA = run_granulo_once(library, phaseA_pairs, min_cluster_size, pack, max_cluster_soft)
    setA = set(resA["qc_ids"])

    # Phase B
    log(f"[SAT] Phase B pairs={phaseB_pairs}")
    resB = run_granulo_once(library, phaseB_pairs, min_cluster_size, pack, max_cluster_soft)
    setB = set(resB["qc_ids"])

    diff = sorted(list(setB - setA))
    saturation_ok = (len(diff) == 0)

    # On expose le résultat de la phase B comme "current run"
    audit = resB["audit"]
    audit["saturation_ok"] = saturation_ok
    audit["setA_qc"] = len(setA)
    audit["setB_qc"] = len(setB)
    audit["setB_minus_setA"] = len(diff)
    audit["set_diff_sample"] = diff[:50]

    # SEALED: règles binaires
    sealed = bool(
        saturation_ok and
        audit.get("sanity_ok") and
        audit.get("qi_orphans", 1) == 0 and
        audit.get("qi_posable", 0) > 0 and
        audit.get("qc_total", 0) > 0
    )

    return {
        "qi_pack": resB["qi_pack"],
        "qc_pack": resB["qc_pack"],
        "chapter_report": resB["chapter_report"],
        "audit": audit,
        "sealed": sealed,
        "setA_qc_ids": sorted(list(setA)),
        "setB_qc_ids": sorted(list(setB)),
        "set_diff": diff,
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
    st.caption("UI + Moteur en mono-fichier. Pack FILE signé. Preuve F1/F2 (audit shell) + saturation Set_B-Set_A.")

    # SIDEBAR
    with st.sidebar:
        st.markdown("## ÉTAPE 1 — ACTIVATION (Pack FILE signé)")

        country = st.selectbox("Pays", [DEFAULT_COUNTRY], index=0)
        st.session_state.country = country

        st.markdown("### Upload Pack JSON (optionnel)")
        up = st.file_uploader("Pack JSON", type=["json"])
        if up is not None:
            st.session_state._pack_upload_bytes = up.getvalue()
        else:
            st.session_state._pack_upload_bytes = None

        st.markdown("### Autoriser Pack Genesis (TEST)")
        allow_genesis = st.checkbox("Autoriser Pack Genesis (TEST)", value=True)
        genesis_n = st.number_input("Genesis: nombre de chapitres", 4, 12, 12)

        if st.button("ACTIVER", use_container_width=True):
            try:
                pack = load_academic_pack(
                    country=country,
                    allow_genesis=allow_genesis,
                    genesis_n_chapters=int(genesis_n),
                    uploaded_pack_bytes=st.session_state._pack_upload_bytes
                )
                st.session_state.pack_active = pack
                st.session_state.pack_id = str(pack.get("pack_id") or "")
                st.session_state.pack_source = str(pack.get("_source") or "")
                st.success("Pack actif")
            except Exception as e:
                st.session_state.pack_active = None
                st.session_state.pack_id = None
                st.session_state.pack_source = None
                st.session_state.pack_path = None
                st.session_state.pack_sha256 = None
                st.error(f"Erreur: {e}")

        if st.session_state.pack_active:
            st.success(f"Pack: {st.session_state.pack_id}")
            st.caption(f"Source: {st.session_state.pack_source}")
            st.caption(f"Signature SHA256: {st.session_state.pack_sha256}")
            st.caption(f"Path: {st.session_state.pack_path}")
        else:
            st.warning("Pack inactif")

        st.markdown("---")
        st.markdown("## ÉTAPE 2 — SÉLECTION (scope)")
        level = st.radio("Niveau", ["Seconde", "Première", "Terminale"], index=2)
        st.session_state.level = {"Seconde": "SECONDE", "Première": "PREMIERE", "Terminale": "TERMINALE"}[level]

        st.selectbox("Matière (test)", ["MATH"], index=0)

        st.markdown("### Chapitres (pack-driven)")
        if st.session_state.pack_active:
            chs = pack_chapters(st.session_state.pack_active)
            st.caption(f"{len(chs)} chapitres")
            for c in chs:
                st.write(f"• {c['chapter_code']}")
        else:
            st.caption("—")

    tab1, tab2, tab3 = st.tabs(["Import", "RUN", "Exports"])

    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?") and x.get("corrige_url"))

    # IMPORT
    with tab1:
        st.markdown("## Import / Bibliothèque")
        metric_row(len(lib), corr_ok, st.session_state.run_stats.get("qi", 0),
                   st.session_state.run_stats.get("qi_posable", 0),
                   st.session_state.run_stats.get("qc", 0), st.session_state.sealed)

        st.markdown("### Bibliothèque")
        df_table(lib)

        st.markdown("---")
        st.markdown("### HARVEST AUTO (APMEP)")
        c1, c2 = st.columns(2)
        years_back = c1.number_input("Années", 1, 15, 10)
        volume_max = c2.number_input("Volume max (pairs)", 5, 200, 30)

        if st.button("HARVEST", use_container_width=True):
            if not st.session_state.pack_active:
                st.error("Activez le pack d'abord")
            else:
                try:
                    manifest = harvest_apmep(st.session_state.level, "MATH", int(years_back), int(volume_max))
                    st.session_state.harvest_manifest = manifest
                    st.session_state.library = manifest["library"]
                    st.session_state.sealed = False
                    st.session_state.qi_pack = None
                    st.session_state.qc_pack = None
                    st.session_state.chapter_report = None
                    st.session_state.last_run_audit = None
                    st.session_state.setA_qc_ids = []
                    st.session_state.setB_qc_ids = []
                    st.session_state.set_diff = []
                    st.session_state.saturation_ok = False
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0}
                    st.success(f"HARVEST OK: {manifest['items_total']} pairs (corrigés={manifest['items_corrige_ok']})")
                except Exception as e:
                    st.error(f"HARVEST échoué: {e}")

        st.markdown("---")
        st.markdown("### Upload manuel (Sujets/Corrigés)")
        c1, c2 = st.columns(2)
        up_sujet = c1.file_uploader("PDF Sujet", type=["pdf"], key="up_sujet_pdf")
        up_corr = c2.file_uploader("PDF Corrigé", type=["pdf"], key="up_corr_pdf")

        if st.button("Ajouter à la bibliothèque", use_container_width=True):
            if up_sujet is None or up_corr is None:
                st.error("Veuillez uploader Sujet + Corrigé")
            else:
                su_b = up_sujet.getvalue()
                co_b = up_corr.getvalue()
                su_url = f"local://SUJET_{stable_id(up_sujet.name, str(time.time()))}.pdf"
                co_url = f"local://CORR_{stable_id(up_corr.name, str(time.time()))}.pdf"
                st.session_state._uploads[su_url] = su_b
                st.session_state._uploads[co_url] = co_b

                pair_id = f"PAIR_LOCAL_{stable_id(up_sujet.name, up_corr.name, str(time.time()))}"
                item = {
                    "pair_id": pair_id,
                    "scope": f"{st.session_state.level}|MATH",
                    "source": "UPLOAD",
                    "year": "",
                    "sujet": up_sujet.name,
                    "corrige?": True,
                    "corrige_name": up_corr.name,
                    "reason": "",
                    "sujet_url": su_url,
                    "corrige_url": co_url,
                }
                st.session_state.library = [item] + st.session_state.library
                st.success("Ajouté")

    # RUN
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
            max_vol = max(1, min(200, corr_ok2))
            c1, c2 = st.columns(2)
            phaseA = c1.slider("Phase A (pairs)", 1, max_vol, min(10, max_vol))
            phaseB = c2.slider("Phase B (pairs)", 1, max_vol, min(20, max_vol))

            c3, c4 = st.columns(2)
            min_cluster = c3.slider("Min cluster", 1, 12, 2)
            max_cluster_soft = c4.slider("Max cluster (soft)", 80, 600, 250)

            if st.button("LANCER", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_granulo_test_AB(
                        library=lib2,
                        phaseA_pairs=int(phaseA),
                        phaseB_pairs=int(phaseB),
                        min_cluster_size=int(min_cluster),
                        pack=st.session_state.pack_active,
                        max_cluster_soft=int(max_cluster_soft),
                    )

                    st.session_state.qi_pack = res["qi_pack"]
                    st.session_state.qc_pack = res["qc_pack"]
                    st.session_state.chapter_report = res["chapter_report"]
                    st.session_state.last_run_audit = res["audit"]

                    st.session_state.setA_qc_ids = res["setA_qc_ids"]
                    st.session_state.setB_qc_ids = res["setB_qc_ids"]
                    st.session_state.set_diff = res["set_diff"]
                    st.session_state.saturation_ok = bool(res["audit"].get("saturation_ok"))

                    st.session_state.sealed = bool(res["sealed"])

                    st.session_state.run_stats = {
                        "qi": res["audit"]["qi_total"],
                        "rqi": res["audit"]["rqi_total"],
                        "qc": res["audit"]["qc_total"],
                        "qi_posable": res["audit"]["qi_posable"],
                    }

                    if st.session_state.sealed:
                        st.success("SEALED = YES")
                    else:
                        st.warning(
                            f"SEALED = NO | saturation_ok={res['audit'].get('saturation_ok')} | "
                            f"sanity_ok={res['audit'].get('sanity_ok')} | orphans={res['audit'].get('qi_orphans')} | "
                            f"posable={res['audit'].get('qi_posable')}"
                        )
                        if res["audit"].get("suspicious_docs"):
                            st.error("SANITY FAIL détecté (segmentation instable)")

                except Exception as e:
                    st.error(f"RUN échoué: {e}")

        st.markdown("### Logs")
        st.text_area("", logs_text(), height=320)

    # EXPORTS + EXPLORATEUR
    with tab3:
        st.markdown("## Exports")
        metric_row(len(st.session_state.library),
                   sum(1 for x in st.session_state.library if x.get("corrige?") and x.get("corrige_url")),
                   st.session_state.run_stats.get("qi", 0),
                   st.session_state.run_stats.get("qi_posable", 0),
                   st.session_state.run_stats.get("qc", 0),
                   st.session_state.sealed)

        hm = st.session_state.harvest_manifest or {"version": APP_VERSION}
        st.download_button("harvest_manifest.json", json.dumps(hm, ensure_ascii=False, indent=2), "harvest_manifest.json")
        st.download_button("logs.txt", logs_text(), "logs.txt")

        if st.session_state.qi_pack:
            st.download_button("qi_pack.json", json.dumps(st.session_state.qi_pack, ensure_ascii=False, indent=2), "qi_pack.json")
        if st.session_state.qc_pack:
            st.download_button("qc_pack.json", json.dumps(st.session_state.qc_pack, ensure_ascii=False, indent=2), "qc_pack.json")
        if st.session_state.chapter_report:
            st.download_button("chapter_report.json", json.dumps(st.session_state.chapter_report, ensure_ascii=False, indent=2), "chapter_report.json")

        # Saturation exports
        if st.session_state.setA_qc_ids:
            st.download_button("setA_qc_ids.json", json.dumps(st.session_state.setA_qc_ids, ensure_ascii=False, indent=2), "setA_qc_ids.json")
        if st.session_state.setB_qc_ids:
            st.download_button("setB_qc_ids.json", json.dumps(st.session_state.setB_qc_ids, ensure_ascii=False, indent=2), "setB_qc_ids.json")
        if st.session_state.set_diff is not None:
            st.download_button("setB_minus_setA.json", json.dumps(st.session_state.set_diff, ensure_ascii=False, indent=2), "setB_minus_setA.json")

        st.markdown("---")
        st.markdown("## Explorateur QC → Qi → (RQi, Triggers, F1, ARI, FRT)")

        if st.session_state.qc_pack and st.session_state.qi_pack:
            ch_options = ["ALL"] + sorted({qc["chapter_code"] for qc in st.session_state.qc_pack})
            sel_ch = st.selectbox("Chapitre", ch_options)

            qcs = st.session_state.qc_pack if sel_ch == "ALL" else [q for q in st.session_state.qc_pack if q["chapter_code"] == sel_ch]

            if qcs:
                qc_labels = [
                    f"{q['qc_id']} | {q['chapter_code']} | n={q['cluster_size']} | {','.join(q.get('signature', []))}"
                    for q in qcs
                ]
                sel_idx = st.selectbox("QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]

                st.markdown(f"### QC: {qc['qc']}")
                st.caption(f"qc_id={qc['qc_id']} | chapter={qc['chapter_code']} | cluster_size={qc['cluster_size']}")
                st.caption(f"signature={qc.get('signature')} | triggers_hint={qc.get('triggers_hint')}")

                qi_by_id = {q["qi_id"]: q for q in st.session_state.qi_pack}

                st.markdown("### Qi associées")
                for qi_id in qc["qi_ids"][:40]:
                    q = qi_by_id.get(qi_id)
                    if not q:
                        continue

                    with st.expander(f"{qi_id} | has_rqi={q.get('has_rqi')} | orphan={q.get('is_orphan')}"):
                        st.markdown("**Qi (énoncé)**")
                        st.write((q.get("qi") or "")[:2200])

                        st.markdown("**RQi (correction)**")
                        st.write((q.get("rqi") or "")[:2200] if q.get("rqi") else "— Pas de RQi alignée —")

                        st.markdown("**Triggers**")
                        st.write(q.get("triggers") or [])

                        st.markdown("**F1 (audit)**")
                        st.json(q.get("f1_audit") or {})

                        st.markdown("**ARI (proxy déterministe)**")
                        st.json(q.get("ari") or {})

                        st.markdown("**FRT**")
                        st.json(q.get("frt") or {})

        if st.session_state.last_run_audit:
            st.markdown("---")
            st.markdown("## Audit")
            st.json(st.session_state.last_run_audit)


if __name__ == "__main__":
    main()
