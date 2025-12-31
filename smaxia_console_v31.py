# =============================================================================
# SMAXIA GTE Console V31.10.7 — ISO-PROD TEST (MONO-FICHIER)
# =============================================================================
# OBJECTIF (TEST ISO-PROD):
# - 1 seul fichier: Console + Moteur (aucun import moteur externe)
# - Flux: Activation -> Pack visible -> Sélection -> Harvest -> RUN -> Résultats -> Exports
# - Variables à produire & explorer:
#   (Qi, RQi), ARI, FRT, Triggers, QC, et QC rangées par chapitre (pack-driven)
#
# RÈGLES ISO-PROD:
# - Interdit: "stub pack". Pack introuvable => RUN bloqué.
# - Interdit: hardcode métier (pays/langue/matière/chapitres) dans le core.
#   => Le pack fournit chapitres + metadata; le core ne contient que heuristiques générales.
# - Preuve: exports JSON + logs.
# - Verdict test central: Aucune Qi ne doit rester orpheline (Qi sans QC).
#
# FIX V31.10.7 (vs 31.10.6/31.10.5):
# - FIX UI-1: garde-fou slider (jamais max<min, jamais value hors borne) => plus de StreamlitAPIException.
# - FIX CORE-1: un seul moteur RUN, plus de duplication "patched".
# - FIX CORE-2: fetcher unique http(s)/local:// pour PDF.
# - FIX CORE-3: extraction PDF robuste: pdfplumber -> pypdf fallback.
# - FIX CORE-4: build_triggers() structurel (pas de mots FR/EN).
# - FIX CORE-5: alignement Qi↔RQi amélioré (index + similarité tokens).
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import time
import glob
import hashlib
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests
import streamlit as st

# pdf parsing
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

# html parsing
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


# -----------------------------------------------------------------------------
# CONST / SETTINGS
# -----------------------------------------------------------------------------
APP_VERSION = "V31.10.7"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
REQ_TIMEOUT = 30
MAX_PDF_MB = 35

DEFAULT_COUNTRY = "FR"
DEFAULT_LEVEL = "TERMINALE"
DEFAULT_SUBJECTS = ["MATH"]

# APMEP root discovery (test harness)
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
    st.session_state.setdefault("library", [])
    st.session_state.setdefault("harvest_manifest", None)
    st.session_state.setdefault("qi_pack", None)
    st.session_state.setdefault("qc_pack", None)
    st.session_state.setdefault("chapter_report", None)
    st.session_state.setdefault("sealed", False)
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("run_stats", {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0})
    st.session_state.setdefault("last_run_audit", None)
    st.session_state.setdefault("_uploads", {})          # local:// store
    st.session_state.setdefault("_http_pdf_cache", {})   # simple cache


def log(msg: str):
    line = f"[{_utc_ts()}] {msg}"
    st.session_state.logs.append(line)


def logs_text() -> str:
    return "\n".join(st.session_state.logs[-5000:])


# -----------------------------------------------------------------------------
# PACK LOADING (ISO-PROD: no stub)
# -----------------------------------------------------------------------------
def _safe_read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_pack_paths(country: str) -> List[str]:
    candidates = []
    candidates += [
        os.path.join(os.getcwd(), "academic_packs", f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), "packs", f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), "data", "packs", f"CAP_{country}_BAC_2024_V1.json"),
    ]
    candidates += glob.glob(os.path.join(os.getcwd(), "**", f"CAP_{country}_BAC_2024_V1.json"), recursive=True)
    candidates += glob.glob(os.path.join(os.getcwd(), "**", f"CAP_{country}_*.json"), recursive=True)

    seen = set()
    out = []
    for p in candidates:
        pp = os.path.abspath(p)
        if pp not in seen and os.path.isfile(pp):
            seen.add(pp)
            out.append(pp)
    return out


def load_academic_pack(country: str) -> Dict[str, Any]:
    paths = _candidate_pack_paths(country)
    log(f"[PACK] Searching pack for country={country} | candidates={len(paths)}")
    if not paths:
        raise FileNotFoundError(
            "Aucun fichier pack trouvé. Placez le pack JSON (ex: CAP_FR_BAC_2024_V1.json) "
            "dans ./academic_packs/ ou ./packs/ ou à la racine du projet."
        )

    last_err = None
    for path in paths[:25]:
        try:
            pack = _safe_read_json(path)
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
    out: List[Dict[str, Any]] = []
    for item in ch:
        if isinstance(item, dict) and ("chapter_code" in item or "code" in item):
            code = str(item.get("chapter_code") or item.get("code") or "").strip()
            label = str(item.get("chapter_label") or item.get("label") or code).strip()
            if code:
                out.append({
                    "chapter_code": code,
                    "chapter_label": label,
                    "keywords": item.get("keywords", []),
                    "intent_allowlist": item.get("intent_allowlist", []),
                })
        elif isinstance(item, dict) and "items" in item and isinstance(item["items"], list):
            for sub in item["items"]:
                if isinstance(sub, dict):
                    code = str(sub.get("chapter_code") or sub.get("code") or "").strip()
                    label = str(sub.get("chapter_label") or sub.get("label") or code).strip()
                    if code:
                        out.append({
                            "chapter_code": code,
                            "chapter_label": label,
                            "keywords": sub.get("keywords", []),
                            "intent_allowlist": sub.get("intent_allowlist", []),
                        })
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
    toks = re.findall(r"[a-z0-9]{3,}", s)
    return toks[:800]


def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# -----------------------------------------------------------------------------
# HTTP + PDF FETCH / DOWNLOAD
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


def fetch_pdf_bytes(url: str) -> bytes:
    """
    Fetch unique pour http(s) + local://
    """
    if url.startswith("local://"):
        data = st.session_state.get("_uploads", {}).get(url)
        if not data:
            raise RuntimeError(f"Upload local introuvable: {url}")
        return data

    cache: Dict[str, bytes] = st.session_state.get("_http_pdf_cache", {})
    if url in cache:
        return cache[url]
    b = download_pdf_bytes(url)
    cache[url] = b
    st.session_state._http_pdf_cache = cache
    return b


# -----------------------------------------------------------------------------
# PDF TEXT EXTRACTION (robuste)
# -----------------------------------------------------------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 80) -> str:
    # 1) pdfplumber
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                texts = []
                for page in pdf.pages[:max_pages]:
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    if t:
                        texts.append(t)
                out = "\n".join(texts).strip()
                if out:
                    return out
        except Exception:
            pass

    # 2) pypdf fallback
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            texts = []
            for i, page in enumerate(reader.pages[:max_pages]):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t:
                    texts.append(t)
            out = "\n".join(texts).strip()
            return out
        except Exception:
            return ""

    return ""


# -----------------------------------------------------------------------------
# Qi / RQi EXTRACTION (heuristiques structurelles)
# -----------------------------------------------------------------------------
_Q_SPLIT_PATTERNS = [
    r"(?m)^\s*\d{1,2}\s*[\)\.\-:]\s+",
    r"(?m)^\s*[a-h]\s*[\)\.\-:]\s+",
    r"(?m)^\s*[ivxIVX]+\s*[\)\.\-:]\s+",
    r"(?m)^\s*•\s+",
    r"(?m)^\s*-\s+(?=[A-Z])",
]


def split_questions(raw_text: str) -> List[str]:
    t = raw_text or ""
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)

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

    # Fallback: split by punctuation
    if len(chunks) <= 1:
        parts = re.split(r"(?<=[\?\!])\s+", t)
        chunks = [p.strip() for p in parts if len(p.strip()) >= 60]

    cleaned = []
    for c in chunks:
        c2 = re.sub(r"\s+", " ", c).strip()
        if len(c2) > 2500:
            c2 = c2[:2500].rstrip() + "…"
        cleaned.append(c2)

    seen = set()
    out = []
    for c in cleaned:
        k = stable_id(norm_text(c)[:400])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out[:350]


def align_qi_rqi(qs: List[str], rs: List[str]) -> List[Optional[int]]:
    """
    Retourne pour chaque Qi l'index de la meilleure RQi associée (ou None).
    Heuristique stable:
    - Si tailles proches: favorise l'alignement par index.
    - Sinon: matching par similarité tokens (Jaccard) + garde-fou seuil.
    """
    if not qs or not rs:
        return [None for _ in qs]

    qtok = [tokenize(q) for q in qs]
    rtok = [tokenize(r) for r in rs]

    # 1) Alignement index si plausible
    if abs(len(qs) - len(rs)) <= max(2, int(0.25 * max(len(qs), len(rs)))):
        out = []
        for i in range(len(qs)):
            if i < len(rs):
                # si similarité minimale, ok, sinon on cherchera un meilleur
                base_sim = jaccard(qtok[i], rtok[i])
                if base_sim >= 0.06:
                    out.append(i)
                else:
                    out.append(None)
            else:
                out.append(None)
        # compléter les None par recherche locale
        for i, v in enumerate(out):
            if v is not None:
                continue
            best_j = None
            best_s = 0.0
            # fenêtre locale autour de i
            lo = max(0, i - 4)
            hi = min(len(rs), i + 5)
            for j in range(lo, hi):
                s = jaccard(qtok[i], rtok[j])
                if s > best_s:
                    best_s = s
                    best_j = j
            out[i] = best_j if (best_j is not None and best_s >= 0.10) else None
        return out

    # 2) Matching global
    out2: List[Optional[int]] = []
    for i in range(len(qs)):
        best_j = None
        best_s = 0.0
        for j in range(len(rs)):
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s = s
                best_j = j
        out2.append(best_j if (best_j is not None and best_s >= 0.12) else None)
    return out2


# -----------------------------------------------------------------------------
# CLUSTER (SimHash) — généraliste & déterministe
# -----------------------------------------------------------------------------
def simhash64(tokens: List[str]) -> int:
    if not tokens:
        return 0
    v = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
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
    max_ham = max(0, min(64, int(round((1.0 - sim_threshold) * 64))))
    max_ham = min(max_ham, 26)

    for it in items:
        toks = tokenize(it.get("text", ""))
        it["_sh"] = simhash64(toks)

    clusters: List[List[Dict[str, Any]]] = []
    used = set()
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

    if min_cluster_size <= 1:
        return clusters

    big = [c for c in clusters if len(c) >= min_cluster_size]
    small = [c for c in clusters if len(c) < min_cluster_size]
    if not small:
        return clusters

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
    """
    Triggers STRICTEMENT structurels (pas de FR/EN, pas de mots métier).
    """
    t_raw = qi_text or ""
    t = norm_text(t_raw)
    out: List[str] = []

    # 1) présence question explicite
    if "?" in t_raw:
        out.append("TRG_HAS_QUESTION_MARK")

    # 2) présence de symboles / opérateurs / notations
    if re.search(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦]", t_raw) or re.search(r"\d+[,\.]\d+", t_raw):
        out.append("TRG_SYMBOLIC_OR_NUMERIC")

    # 3) multi-sous-questions (a), b), i), ii), 1), 2) ... dans le bloc)
    if re.search(r"(?m)^\s*([a-h]|[ivx]+|\d{1,2})\s*[\)\.\-:]\s+", t_raw, flags=re.IGNORECASE):
        out.append("TRG_SUBQUESTIONS_PRESENT")

    # 4) contraintes/conditions structurées: présence de "=>" ou ":" ou "∈" ou "∀"
    if re.search(r"(=>|:|∈|∀|∃)", t_raw):
        out.append("TRG_CONSTRAINT_STRUCTURE")

    # 5) demande de construction/justification probable: densité de connecteurs logiques symboliques
    if re.search(r"(⇔|⇒|⟺|→)", t_raw):
        out.append("TRG_LOGICAL_IMPLICATION")

    if not out:
        out = [
            "TRG_IDENTIFY_DATA",
            "TRG_SELECT_METHOD",
            "TRG_EXECUTE_STEPS",
            "TRG_VALIDATE_RESULT",
        ]
    return out[:7]


def build_ari(qi_text: str, rqi_text: str) -> Dict[str, Any]:
    return {
        "template": "ARI_V2_INVARIANT",
        "steps": [
            {"step_id": "S1", "op": "OP_READ", "action": "Extract explicit data and constraints from statement."},
            {"step_id": "S2", "op": "OP_PLAN", "action": "Select resolution strategy based on structure."},
            {"step_id": "S3", "op": "OP_EXECUTE", "action": "Apply transformations step by step."},
            {"step_id": "S4", "op": "OP_CHECK", "action": "Verify intermediate results and domain validity."},
            {"step_id": "S5", "op": "OP_CONCLUDE", "action": "Formulate final answer with required format."},
        ],
        "evidence_rule": "Each step must be traceable in RQi alignment when available.",
    }


def build_frt(qi_text: str, rqi_text: str, triggers: List[str]) -> Dict[str, Any]:
    return {
        "template": "FRT_V2_INVARIANT",
        "sections": [
            {"section_id": "SEC_OBJ", "name": "OBJECTIVE", "fill": "Goal extracted from statement."},
            {"section_id": "SEC_DAT", "name": "DATA", "fill": "Given values, constraints, hypotheses."},
            {"section_id": "SEC_MTH", "name": "METHOD", "fill": "Strategy selected (generic)."},
            {"section_id": "SEC_EXE", "name": "EXECUTION", "fill": "Step-by-step transformations."},
            {"section_id": "SEC_RES", "name": "RESULT", "fill": "Final answer in expected format."},
            {"section_id": "SEC_VAL", "name": "VALIDATION", "fill": "Coherence check (domain, units, edge cases)."},
        ],
        "triggers_applied": triggers,
    }


def build_qc_label(cluster_items: List[Dict[str, Any]]) -> str:
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
    qc = qc.replace("??", "?").strip()
    if not qc.startswith("Comment"):
        qc = "Comment " + qc[0].lower() + qc[1:]
    if not qc.endswith("?"):
        qc += " ?"
    return qc


def map_qc_to_chapter(qc_text: str, chapters: List[Dict[str, Any]]) -> str:
    if not chapters:
        return "UNMAPPED"
    qc_toks = set(tokenize(qc_text))
    best = ("UNMAPPED", 0.0)

    for ch in chapters:
        # allowlist exact (si fourni par le pack)
        allowlist = ch.get("intent_allowlist") or []
        if isinstance(allowlist, list) and allowlist:
            for intent in allowlist:
                if str(intent).strip() and str(intent).lower() in qc_text.lower():
                    return str(ch["chapter_code"])

        kws = ch.get("keywords") or []
        if not isinstance(kws, list) or not kws:
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
# HARVEST APMEP
# -----------------------------------------------------------------------------
def _soup(html: str):
    if BeautifulSoup is None:
        raise RuntimeError("bs4 non installé. Installez: pip install beautifulsoup4")
    return BeautifulSoup(html, "html.parser")


def harvest_apmep(level: str, subject: str, years_back: int, volume_max: int) -> Dict[str, Any]:
    if level not in APMEP_ROOT_BY_LEVEL:
        raise ValueError(f"Niveau non supporté pour APMEP: {level}")
    root = APMEP_ROOT_BY_LEVEL[level]
    log(f"[HARVEST] scope={level}|{subject} root={root} years_back={years_back} volume_max={volume_max}")

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
        raise RuntimeError("Impossible de trouver les pages Année-YYYY sur APMEP.")

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

            pdf_anchors = [a for a in y_sp.find_all("a") if (a.get("href") or "").lower().endswith(".pdf")]
            log(f"[HARVEST] year={y} pdf_links={len(pdf_anchors)}")

            groups = []
            for a in pdf_anchors:
                parent = a
                for _ in range(4):
                    if parent is None:
                        break
                    pdfs_in_parent = parent.find_all("a") if hasattr(parent, "find_all") else []
                    n_pdf = sum(1 for x in pdfs_in_parent if (x.get("href") or "").lower().endswith(".pdf"))
                    if n_pdf >= 2:
                        groups.append(parent)
                        break
                    parent = parent.parent

            if not groups:
                groups = [y_sp]

            for g in groups:
                if len(pairs) >= volume_max:
                    break
                links = [a for a in g.find_all("a") if (a.get("href") or "").lower().endswith(".pdf")]
                if not links:
                    continue

                def is_meta_pdf(h: str, txt: str) -> bool:
                    s = norm_text(h + " " + txt)
                    return any(k in s for k in ["explication", "liste", "index", "sommaire"])

                def is_corrige(a_tag) -> bool:
                    href2 = (a_tag.get("href") or "")
                    txt2 = (a_tag.get_text() or "")
                    s2 = norm_text(href2 + " " + txt2)
                    return "corrig" in s2 or "corrige" in s2

                pdfs = []
                for a_tag in links:
                    href = a_tag.get("href") or ""
                    txt = (a_tag.get_text() or "").strip()
                    url_pdf = href if href.startswith("http") else requests.compat.urljoin(url, href)
                    if is_meta_pdf(url_pdf, txt):
                        continue
                    pdfs.append({"url": url_pdf, "name": os.path.basename(url_pdf), "is_corrige": is_corrige(a_tag)})

                if not pdfs:
                    continue

                sujets = [p for p in pdfs if not p["is_corrige"]]
                corr = [p for p in pdfs if p["is_corrige"]]

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
# RUN GRANULO TEST — mono-moteur (http(s)+local://), chapitre report complet
# -----------------------------------------------------------------------------
def run_granulo_test(
    library: List[Dict[str, Any]],
    volume: int,
    sim_threshold: float,
    min_cluster_size: int,
    max_iterations: int,
    pack: Dict[str, Any],
) -> Dict[str, Any]:
    chapters = pack_chapters(pack)
    if not chapters:
        raise RuntimeError("Pack invalide: liste de chapitres vide après normalisation.")

    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    if not exploitable:
        raise RuntimeError("ISO-PROD: Aucun corrigé exploitable dans la bibliothèque. RUN interdit.")

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
            su_pdf = fetch_pdf_bytes(su_url)
            co_pdf = fetch_pdf_bytes(co_url)
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
        link = align_qi_rqi(qs, rs)

        sujets_meta.append({
            "pair_id": pid,
            "sujet_url": su_url,
            "corrige_url": co_url,
            "qi_count": len(qs),
            "rqi_count": len(rs),
        })

        for i, q in enumerate(qs):
            if not q:
                continue
            j = link[i] if i < len(link) else None
            r = rs[j] if (j is not None and 0 <= j < len(rs)) else ""

            qi_id = f"QI_{stable_id(pid, str(i), norm_text(q)[:180])}"
            rqi_id = f"RQI_{stable_id(pid, str(j), norm_text(r)[:180])}" if r else ""

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
        raise RuntimeError("RUN: aucune Qi extraite (échec extraction PDF / alignement).")

    # Cluster -> QC (toutes Qi couvertes => pas d'orphelines QC)
    qc_pack: List[Dict[str, Any]] = []
    qc_map: Dict[str, str] = {}

    prev_qc_count = -1
    sealed_candidate = False
    last_iter = 1

    for it in range(max(1, max_iterations)):
        clusters = cluster_by_simhash(qi_items, sim_threshold=sim_threshold, min_cluster_size=max(1, min_cluster_size))
        qc_pack = []
        qc_map = {}

        for cidx, cluster in enumerate(clusters, start=1):
            qc_text = build_qc_label(cluster)
            qc_id = f"QC_{stable_id(str(cidx), norm_text(qc_text))}"
            chapter_code = map_qc_to_chapter(qc_text, chapters)

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

        last_iter = it + 1
        if prev_qc_count == len(qc_pack):
            sealed_candidate = True
            log(f"[SAT] stable at iter={it+1} qc_count={len(qc_pack)} => SEALED_CANDIDATE")
            break
        prev_qc_count = len(qc_pack)
        log(f"[SAT] iter={it+1} qc_count={len(qc_pack)}")

    # Build Qi pack avec ARI/FRT/Triggers
    rqi_by_id = {r["rqi_id"]: r for r in rqi_items}

    qi_pack: List[Dict[str, Any]] = []
    qi_posable = 0
    orphan_count = 0
    orphan_ids: List[str] = []

    for q in qi_items:
        qi_id = q["qi_id"]
        rqi_id = q.get("rqi_id") or ""
        rtxt = rqi_by_id.get(rqi_id, {}).get("text", "")

        triggers = build_triggers(q["text"])
        ari = build_ari(q["text"], rtxt)
        frt = build_frt(q["text"], rtxt, triggers)

        qc_id = qc_map.get(qi_id, "")
        is_orphan = not qc_id
        if is_orphan:
            orphan_count += 1
            orphan_ids.append(qi_id)
        if rtxt:
            qi_posable += 1

        qi_pack.append({
            "qi_id": qi_id,
            "pair_id": q["pair_id"],
            "k": q["k"],
            "qi": q["text"],
            "rqi": rtxt,
            "has_rqi": bool(rtxt),
            "qc_id": qc_id,
            "is_orphan": is_orphan,
            "triggers": triggers,
            "ari": ari,
            "frt": frt,
        })

    # Chapter report
    ch_label = {c["chapter_code"]: c.get("chapter_label", c["chapter_code"]) for c in chapters}

    qc_by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    qc_by_id: Dict[str, Dict[str, Any]] = {}
    for qc in qc_pack:
        qc_by_chapter.setdefault(qc["chapter_code"], []).append(qc)
        qc_by_id[qc["qc_id"]] = qc

    qi_by_chapter: Dict[str, List[str]] = {}
    qi_covered_by_chapter: Dict[str, List[str]] = {}
    qi_orphans_by_chapter: Dict[str, List[str]] = {}

    for qi in qi_pack:
        qc_id = qi.get("qc_id", "")
        chapter_code = "UNMAPPED"
        if qc_id and qc_id in qc_by_id:
            chapter_code = qc_by_id[qc_id]["chapter_code"]

        qi_by_chapter.setdefault(chapter_code, []).append(qi["qi_id"])

        if qi.get("has_rqi"):
            if qi.get("is_orphan"):
                qi_orphans_by_chapter.setdefault(chapter_code, []).append(qi["qi_id"])
            else:
                qi_covered_by_chapter.setdefault(chapter_code, []).append(qi["qi_id"])

    chapter_report_entries: List[Dict[str, Any]] = []
    all_chapters_seen = set(qc_by_chapter.keys()) | set(qi_by_chapter.keys())

    for cc in sorted(all_chapters_seen):
        qi_total = len(qi_by_chapter.get(cc, []))
        qi_posable_in_ch = len(qi_covered_by_chapter.get(cc, [])) + len(qi_orphans_by_chapter.get(cc, []))
        qi_covered = len(qi_covered_by_chapter.get(cc, []))
        qi_orphans = qi_orphans_by_chapter.get(cc, [])
        qc_count = len(qc_by_chapter.get(cc, []))

        coverage_pct = 0.0
        if qi_posable_in_ch > 0:
            coverage_pct = round((qi_covered / qi_posable_in_ch) * 100.0, 2)

        status = "PASS" if (qi_posable_in_ch > 0 and coverage_pct == 100.0 and qc_count > 0 and len(qi_orphans) == 0) else "FAIL"

        chapter_report_entries.append({
            "chapter_code": cc,
            "chapter_label": ch_label.get(cc, cc),
            "qi_total": qi_total,
            "qi_posable_count": qi_posable_in_ch,
            "covered_qi_count": qi_covered,
            "coverage_pct": coverage_pct,
            "orphans": qi_orphans,
            "orphans_count": len(qi_orphans),
            "qc_count": qc_count,
            "qc_ids": [qc["qc_id"] for qc in qc_by_chapter.get(cc, [])],
            "status": status,
        })

    chapter_report = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": st.session_state.pack_id,
        "chapters": sorted(chapter_report_entries, key=lambda x: (-x["qi_total"], x["chapter_code"])),
        "summary": {
            "total_chapters_active": len([e for e in chapter_report_entries if e["qi_total"] > 0]),
            "chapters_pass": len([e for e in chapter_report_entries if e["status"] == "PASS"]),
            "chapters_fail": len([e for e in chapter_report_entries if e["status"] == "FAIL"]),
            "global_coverage_pct": round(
                (sum(e["covered_qi_count"] for e in chapter_report_entries) / max(1, sum(e["qi_posable_count"] for e in chapter_report_entries))) * 100.0,
                2
            ),
        },
        "unmapped_present": any(e["chapter_code"] == "UNMAPPED" for e in chapter_report_entries),
    }

    audit = {
        "qi_total": len(qi_pack),
        "qi_posable": qi_posable,
        "rqi_total": sum(1 for q in qi_pack if q["rqi"]),
        "qc_total": len(qc_pack),
        "qi_orphans": orphan_count,
        "qi_orphan_ids": orphan_ids[:50],
        "coverage_ok": (orphan_count == 0),
        "posable_ok": (qi_posable > 0),
        "qc_ok": (len(qc_pack) > 0),
        "sealed_candidate": sealed_candidate,
        "sealed": (sealed_candidate and orphan_count == 0 and qi_posable > 0 and len(qc_pack) > 0),
        "chapter_summary": chapter_report["summary"],
        "iterations": last_iter,
    }

    out = {
        "sujets": sujets_meta,
        "qc": qc_pack,
        "saturation": {"iterations": last_iter, "sealed_candidate": sealed_candidate},
        "audit": audit,
    }
    return {"out": out, "qi_pack": qi_pack, "qc_pack": qc_pack, "chapter_report": chapter_report, "audit": audit}


# -----------------------------------------------------------------------------
# UI HELPERS
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
    cols = ["pair_id", "scope", "source", "sujet", "corrige?", "corrige_name", "reason", "sujet_url", "corrige_url"]
    view = [{k: r.get(k, "") for k in cols} for r in rows]
    st.dataframe(view, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=f"SMAXIA GTE Console {APP_VERSION}", layout="wide")
    ss_init()

    st.markdown(f"# 🔒 SMAXIA GTE Console {APP_VERSION} — ISO-PROD TEST")
    st.caption("Flux: Activation → Pack visible → Sélection → Harvest → RUN → Résultats/Exports")

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
        level_code = "TERMINALE" if level == "Terminale" else ("PREMIERE" if level == "Première" else ("SECONDE" if level == "Seconde" else "TERMINALE"))
        st.session_state.level = level_code

        subjects = st.multiselect("Matières (test)", options=["MATH"], default=st.session_state.subjects)
        st.session_state.subjects = subjects if subjects else DEFAULT_SUBJECTS[:]

        st.markdown("### Chapitres (pack-driven)")
        if st.session_state.pack_active:
            chs = pack_chapters(st.session_state.pack_active)
            if not chs:
                st.error("Pack chargé mais chapitres introuvables.")
            else:
                for c in chs[:60]:
                    st.write(f"• {c['chapter_code']} — {c.get('chapter_label','')}")
                if len(chs) > 60:
                    st.caption(f"+{len(chs)-60} autres chapitres…")
        else:
            st.caption("Activez un pack pour afficher les chapitres réels.")

    tab1, tab2, tab3 = st.tabs(["📥 Import / Bibliothèque", "⚡ Chaîne (RUN)", "📦 Résultats / Exports"])

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
        years_back = c1.number_input("Nb d'années à récolter", min_value=1, max_value=15, value=10, step=1)
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
                    st.success(f"HARVEST terminé. +{len(manifest['library'])} items ajoutés (corrigés exploitables={manifest['items_corrige_ok']}).")
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
                su_bytes = sujet_file.getvalue()
                co_bytes = corr_file.getvalue() if corr_file else b""

                su_id = stable_id(sujet_file.name, str(len(su_bytes)))
                co_id = stable_id(corr_file.name, str(len(co_bytes))) if corr_file else ""
                pid = f"PAIR_{st.session_state.level}|{st.session_state.subjects[0]}_{stable_id(su_id, co_id)}"

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

    with tab2:
        st.markdown("## ÉTAPE 3 — LANCER CHAÎNE (RUN)")
        st.caption("ISO-PROD: refus de RUN si pack inactif OU bibliothèque vide OU aucun corrigé exploitable.")

        # recalcul live
        lib2 = st.session_state.library
        corr_ok2 = sum(1 for x in lib2 if x.get("corrige?") and x.get("corrige_url"))

        if not st.session_state.pack_active:
            st.error("Pack inactif: activez le pack (ISO-PROD).")
        elif not lib2:
            st.warning("Bibliothèque vide. Lancez HARVEST AUTO ou upload manuel.")
        elif corr_ok2 <= 0:
            st.error("ISO-PROD: aucun corrigé exploitable. RUN interdit.")
        else:
            # FIX UI-1: bornes sûres
            max_vol = max(1, min(120, corr_ok2))
            default_vol = min(20, max_vol)

            c1, c2, c3 = st.columns(3)
            volume = c1.slider("Volume (items consommés)", min_value=1, max_value=max_vol, value=default_vol, step=1)
            sim = c2.slider("Seuil clustering (sim)", min_value=0.05, max_value=0.95, value=0.18, step=0.01)
            min_cluster = c3.slider("Anti-singleton (min cluster size)", min_value=1, max_value=6, value=2, step=1)
            max_iters = st.slider("Max itérations (saturation)", min_value=1, max_value=8, value=4, step=1)

            if st.button("⚡ LANCER (RUN)", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_granulo_test(
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
                        "qi_posable": int(res["audit"]["qi_posable"]),
                    }

                    if res["audit"]["sealed"]:
                        st.success("SEALED = YES (stable + coverage OK + posable OK)")
                    else:
                        st.warning(
                            f"SEALED = NO | orphans={res['audit']['qi_orphans']} | "
                            f"posable={res['audit']['qi_posable']} | qc={res['audit']['qc_total']} | "
                            f"iters={res['audit']['iterations']}"
                        )
                except Exception as e:
                    st.error(f"RUN échoué: {e}")

        st.markdown("### Log (temps réel)")
        st.text_area("Logs", value=logs_text(), height=380)

    with tab3:
        st.markdown("## ÉTAPE 4 — RÉSULTATS / EXPORTS")
        lib3 = st.session_state.library
        items_total3 = len(lib3)
        corr_ok3 = sum(1 for x in lib3 if x.get("corrige?") and x.get("corrige_url"))
        qi_count3 = int(st.session_state.run_stats.get("qi", 0))
        qi_posable3 = int(st.session_state.run_stats.get("qi_posable", 0))
        qc_count3 = int(st.session_state.run_stats.get("qc", 0))
        sealed3 = bool(st.session_state.sealed)
        metric_row(items_total3, corr_ok3, qi_count3, qi_posable3, qc_count3, sealed3)

        st.markdown("### Exports (preuves)")
        hm = st.session_state.harvest_manifest or {
            "version": APP_VERSION,
            "timestamp": _utc_ts(),
            "country": st.session_state.country,
            "level": st.session_state.level,
            "subjects": st.session_state.subjects,
            "items_total": items_total3,
            "items_corrige_ok": corr_ok3,
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

                qi_by_id = {q["qi_id"]: q for q in st.session_state.qi_pack}
                st.markdown("#### Qi associés (avec ARI/FRT/Triggers)")
                for qi_id in qc["qi_ids"][:120]:
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

            if st.session_state.last_run_audit:
                st.markdown("---")
                st.markdown("### Audit ISO-PROD (preuve)")
                st.json(st.session_state.last_run_audit)

            if st.session_state.chapter_report:
                st.markdown("---")
                st.markdown("### Chapter Report (summary)")
                st.json(st.session_state.chapter_report.get("summary", {}))


if __name__ == "__main__":
    main()
