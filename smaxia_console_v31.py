# =============================================================================
# SMAXIA GTE Console V31.10.8 — ISO-PROD TEST (MONO-FICHIER)
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
# V31.10.8 = V31.10.7 GPT + Pack embarqué TEST (zéro régression activation)
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
# CONST / SETTINGS
# -----------------------------------------------------------------------------
APP_VERSION = "V31.10.8"
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
    st.session_state.setdefault("_uploads", {})
    st.session_state.setdefault("_http_pdf_cache", {})


def log(msg: str):
    st.session_state.logs.append(f"[{_utc_ts()}] {msg}")


def logs_text() -> str:
    return "\n".join(st.session_state.logs[-5000:])


# -----------------------------------------------------------------------------
# PACK EMBARQUÉ TEST
# -----------------------------------------------------------------------------
def _get_embedded_pack(country: str) -> Optional[Dict[str, Any]]:
    if country == "FR":
        return {
            "pack_id": "CAP_FR_BAC_2024_V1",
            "country_code": "FR",
            "country_label": "France",
            "version": "1.0.0",
            "_source": "EMBEDDED_TEST",
            "chapters": [
                {"chapter_code": "CH_ANALYSE", "chapter_label": "Analyse - Fonctions", "keywords": ["fonction", "derivee", "limite", "continuite", "integrale", "primitive"], "intent_allowlist": []},
                {"chapter_code": "CH_PROBAS", "chapter_label": "Probabilités - Variables aléatoires", "keywords": ["probabilite", "variable", "aleatoire", "esperance", "variance", "loi", "binomiale", "normale"], "intent_allowlist": []},
                {"chapter_code": "CH_GEOMETRIE", "chapter_label": "Géométrie - Nombres complexes", "keywords": ["complexe", "argument", "module", "affixe", "geometrie", "vecteur", "plan"], "intent_allowlist": []},
                {"chapter_code": "CH_SUITES", "chapter_label": "Suites et Récurrence", "keywords": ["suite", "recurrence", "convergence", "arithmetique", "geometrique"], "intent_allowlist": []},
                {"chapter_code": "CH_LOGARITHME", "chapter_label": "Logarithme et Exponentielle", "keywords": ["logarithme", "exponentielle", "ln", "exp", "croissance"], "intent_allowlist": []},
                {"chapter_code": "CH_TRIGONOMETRIE", "chapter_label": "Trigonométrie", "keywords": ["cosinus", "sinus", "tangente", "trigonometrie", "angle", "radian"], "intent_allowlist": []},
            ],
        }
    return None


# -----------------------------------------------------------------------------
# PACK LOADING
# -----------------------------------------------------------------------------
def _safe_read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_pack_paths(country: str) -> List[str]:
    candidates = [
        os.path.join(os.getcwd(), "academic_packs", f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), "packs", f"CAP_{country}_BAC_2024_V1.json"),
        os.path.join(os.getcwd(), f"CAP_{country}_BAC_2024_V1.json"),
    ]
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

    for path in paths[:25]:
        try:
            pack = _safe_read_json(path)
            pack_id = str(pack.get("pack_id") or pack.get("id") or pack.get("name") or "").strip()
            chapters = pack.get("chapters") or pack.get("chapter_list") or pack.get("academic", {}).get("chapters")
            if not pack_id or not isinstance(chapters, list) or len(chapters) == 0:
                continue
            pack["_pack_file_path"] = path
            pack["_source"] = "FILE"
            log(f"[PACK] Loaded from FILE: pack_id={pack_id}")
            return pack
        except Exception as e:
            log(f"[PACK] Candidate failed: {path} | {e}")

    embedded = _get_embedded_pack(country)
    if embedded:
        log(f"[PACK] Loaded EMBEDDED TEST pack for country={country}")
        return embedded

    raise RuntimeError(f"Aucun pack trouvé pour country={country}")


def pack_chapters(pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = pack.get("chapters") or pack.get("chapter_list") or pack.get("academic", {}).get("chapters") or []
    out = []
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
    seen = set()
    uniq = []
    for c in out:
        if c["chapter_code"] not in seen:
            seen.add(c["chapter_code"])
            uniq.append(c)
    return uniq


# -----------------------------------------------------------------------------
# TEXT UTILS
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
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:12]


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", norm_text(s))[:800]


def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


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
        raise ValueError(f"PDF trop volumineux")
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


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 80) -> str:
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                texts = []
                for page in pdf.pages[:max_pages]:
                    try:
                        t = page.extract_text() or ""
                    except:
                        t = ""
                    if t:
                        texts.append(t)
                out = "\n".join(texts).strip()
                if out:
                    return out
        except:
            pass

    if PdfReader:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            texts = []
            for page in reader.pages[:max_pages]:
                try:
                    t = page.extract_text() or ""
                except:
                    t = ""
                if t:
                    texts.append(t)
            return "\n".join(texts).strip()
        except:
            return ""
    return ""


# -----------------------------------------------------------------------------
# Qi / RQi EXTRACTION
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

    if len(chunks) <= 1:
        parts = re.split(r"(?<=[\?\!])\s+", t)
        chunks = [p.strip() for p in parts if len(p.strip()) >= 60]

    cleaned = []
    for c in chunks:
        c2 = re.sub(r"\s+", " ", c).strip()
        if len(c2) > 2500:
            c2 = c2[:2500] + "…"
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
    if not qs or not rs:
        return [None for _ in qs]

    qtok = [tokenize(q) for q in qs]
    rtok = [tokenize(r) for r in rs]

    if abs(len(qs) - len(rs)) <= max(2, int(0.25 * max(len(qs), len(rs)))):
        out = []
        for i in range(len(qs)):
            if i < len(rs) and jaccard(qtok[i], rtok[i]) >= 0.06:
                out.append(i)
            else:
                out.append(None)
        for i, v in enumerate(out):
            if v is not None:
                continue
            best_j, best_s = None, 0.0
            lo, hi = max(0, i - 4), min(len(rs), i + 5)
            for j in range(lo, hi):
                s = jaccard(qtok[i], rtok[j])
                if s > best_s:
                    best_s, best_j = s, j
            out[i] = best_j if best_j is not None and best_s >= 0.10 else None
        return out

    out2 = []
    for i in range(len(qs)):
        best_j, best_s = None, 0.0
        for j in range(len(rs)):
            s = jaccard(qtok[i], rtok[j])
            if s > best_s:
                best_s, best_j = s, j
        out2.append(best_j if best_j is not None and best_s >= 0.12 else None)
    return out2


# -----------------------------------------------------------------------------
# CLUSTER
# -----------------------------------------------------------------------------
def simhash64(tokens: List[str]) -> int:
    if not tokens:
        return 0
    v = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i, val in enumerate(v):
        if val > 0:
            out |= (1 << i)
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def cluster_by_simhash(items: List[Dict[str, Any]], sim_threshold: float, min_cluster_size: int) -> List[List[Dict[str, Any]]]:
    max_ham = min(26, max(0, int(round((1.0 - sim_threshold) * 64))))

    for it in items:
        it["_sh"] = simhash64(tokenize(it.get("text", "")))

    ordered = sorted(items, key=lambda x: x["qi_id"])
    used = set()
    clusters = []

    for i, it in enumerate(ordered):
        if it["qi_id"] in used:
            continue
        cl = [it]
        used.add(it["qi_id"])
        for j in range(i + 1, len(ordered)):
            other = ordered[j]
            if other["qi_id"] in used:
                continue
            if hamming(it["_sh"], other["_sh"]) <= max_ham:
                cl.append(other)
                used.add(other["qi_id"])
        clusters.append(cl)

    if min_cluster_size <= 1:
        return clusters

    big = [c for c in clusters if len(c) >= min_cluster_size]
    small = [c for c in clusters if len(c) < min_cluster_size]
    if small:
        misc = []
        for c in small:
            misc.extend(c)
        if misc:
            big.append(misc)
    return big


# -----------------------------------------------------------------------------
# ARI / FRT / Triggers (STRUCTURELS - zéro mot FR/EN)
# -----------------------------------------------------------------------------
def build_triggers(qi_text: str) -> List[str]:
    out = []
    if "?" in qi_text:
        out.append("TRG_HAS_QUESTION_MARK")
    if re.search(r"[=<>≤≥∈∀∃∑∏∫√≈≠→↦]", qi_text) or re.search(r"\d+[,\.]\d+", qi_text):
        out.append("TRG_SYMBOLIC_OR_NUMERIC")
    if re.search(r"(?m)^\s*([a-h]|[ivx]+|\d{1,2})\s*[\)\.\-:]\s+", qi_text, flags=re.IGNORECASE):
        out.append("TRG_SUBQUESTIONS_PRESENT")
    if re.search(r"(=>|:|∈|∀|∃)", qi_text):
        out.append("TRG_CONSTRAINT_STRUCTURE")
    if re.search(r"(⇔|⇒|⟺|→)", qi_text):
        out.append("TRG_LOGICAL_IMPLICATION")
    if not out:
        out = ["TRG_IDENTIFY_DATA", "TRG_SELECT_METHOD", "TRG_EXECUTE_STEPS", "TRG_VALIDATE_RESULT"]
    return out[:7]


def build_ari(qi_text: str, rqi_text: str) -> Dict[str, Any]:
    return {
        "template": "ARI_V2_INVARIANT",
        "steps": [
            {"step_id": "S1", "op": "OP_READ", "action": "Extract data and constraints."},
            {"step_id": "S2", "op": "OP_PLAN", "action": "Select resolution strategy."},
            {"step_id": "S3", "op": "OP_EXECUTE", "action": "Apply transformations."},
            {"step_id": "S4", "op": "OP_CHECK", "action": "Verify results."},
            {"step_id": "S5", "op": "OP_CONCLUDE", "action": "Formulate answer."},
        ],
    }


def build_frt(qi_text: str, rqi_text: str, triggers: List[str]) -> Dict[str, Any]:
    return {
        "template": "FRT_V2_INVARIANT",
        "sections": [
            {"section_id": "SEC_OBJ", "name": "OBJECTIVE"},
            {"section_id": "SEC_DAT", "name": "DATA"},
            {"section_id": "SEC_MTH", "name": "METHOD"},
            {"section_id": "SEC_EXE", "name": "EXECUTION"},
            {"section_id": "SEC_RES", "name": "RESULT"},
            {"section_id": "SEC_VAL", "name": "VALIDATION"},
        ],
        "triggers_applied": triggers,
    }


def build_qc_label(cluster_items: List[Dict[str, Any]]) -> str:
    all_toks = []
    for it in cluster_items:
        all_toks += tokenize(it.get("text", ""))
    freq = {}
    for tok in all_toks:
        freq[tok] = freq.get(tok, 0) + 1
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:6]
    keywords = [k for k, _ in top][:4]
    if keywords:
        qc = f"Comment résoudre une question caractérisée par: {' '.join(keywords)} ?"
    else:
        qc = "Comment résoudre ce type de question ?"
    return qc


def map_qc_to_chapter(qc_text: str, chapters: List[Dict[str, Any]]) -> str:
    if not chapters:
        return "UNMAPPED"
    qc_toks = set(tokenize(qc_text))
    best = ("UNMAPPED", 0.0)
    for ch in chapters:
        kws = ch.get("keywords") or []
        if not kws:
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
        raise RuntimeError("bs4 non installé")
    return BeautifulSoup(html, "html.parser")


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
        if m and "Annee-" in href:
            y = int(m.group(1))
            url = href if href.startswith("http") else requests.compat.urljoin(root, href)
            year_links.append((y, url))

    year_links = sorted(set(year_links), key=lambda x: -x[0])
    if not year_links:
        raise RuntimeError("Aucune page Année trouvée sur APMEP")

    current_year = year_links[0][0]
    min_year = current_year - max(1, years_back) + 1
    selected = [(y, u) for (y, u) in year_links if y >= min_year]
    log(f"[HARVEST] years={len(selected)}")

    pairs = []
    corrige_ok = 0

    for (y, url) in selected:
        if len(pairs) >= volume_max:
            break
        try:
            y_html = _http_get(url).text
            y_sp = _soup(y_html)
            pdf_anchors = [a for a in y_sp.find_all("a") if (a.get("href") or "").lower().endswith(".pdf")]

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

            def is_meta_pdf(h, txt):
                s = norm_text(h + " " + txt)
                return any(k in s for k in ["explication", "liste", "index", "sommaire"])

            def is_corrige(a_tag):
                s = norm_text((a_tag.get("href") or "") + " " + (a_tag.get_text() or ""))
                return "corrig" in s

            for g in groups:
                if len(pairs) >= volume_max:
                    break
                links = [a for a in g.find_all("a") if (a.get("href") or "").lower().endswith(".pdf")]
                if not links:
                    continue

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
                    "source": f"APMEP {y}",
                    "year": y,
                    "sujet": sujet["name"],
                    "corrige?": bool(corrige),
                    "corrige_name": corrige["name"] if corrige else "",
                    "reason": "" if corrige else "corrigé absent",
                    "sujet_url": sujet["url"],
                    "corrige_url": corrige["url"] if corrige else "",
                }

                if any(x["sujet_url"] == item["sujet_url"] for x in pairs):
                    continue

                pairs.append(item)
                if item["corrige?"]:
                    corrige_ok += 1

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
# RUN GRANULO TEST
# -----------------------------------------------------------------------------
def run_granulo_test(library, volume, sim_threshold, min_cluster_size, max_iterations, pack):
    chapters = pack_chapters(pack)
    if not chapters:
        raise RuntimeError("Pack invalide: chapitres vides")

    exploitable = [it for it in library if it.get("corrige?") and it.get("corrige_url")]
    if not exploitable:
        raise RuntimeError("ISO-PROD: Aucun corrigé exploitable")

    to_process = exploitable[:max(1, min(volume, len(exploitable)))]
    log(f"[RUN] processing={len(to_process)}")

    qi_items, rqi_items, sujets_meta = [], [], []

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

        qs = split_questions(su_text)
        rs = split_questions(co_text)
        link = align_qi_rqi(qs, rs)

        sujets_meta.append({"pair_id": pid, "qi_count": len(qs), "rqi_count": len(rs)})

        for i, q in enumerate(qs):
            if not q:
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

    qc_pack, qc_map = [], {}
    prev_qc_count = -1
    sealed_candidate = False
    last_iter = 1

    for it in range(max(1, max_iterations)):
        clusters = cluster_by_simhash(qi_items, sim_threshold, max(1, min_cluster_size))
        qc_pack, qc_map = [], {}

        for cidx, cluster in enumerate(clusters, start=1):
            qc_text = build_qc_label(cluster)
            qc_id = f"QC_{stable_id(str(cidx), norm_text(qc_text))}"
            chapter_code = map_qc_to_chapter(qc_text, chapters)

            qc_pack.append({
                "qc_id": qc_id,
                "qc": qc_text,
                "chapter_code": chapter_code,
                "cluster_size": len(cluster),
                "qi_ids": [x["qi_id"] for x in cluster],
            })
            for x in cluster:
                qc_map[x["qi_id"]] = qc_id

        last_iter = it + 1
        if prev_qc_count == len(qc_pack):
            sealed_candidate = True
            log(f"[SAT] stable iter={it+1}")
            break
        prev_qc_count = len(qc_pack)

    rqi_by_id = {r["rqi_id"]: r for r in rqi_items}
    qi_pack = []
    qi_posable, orphan_count, orphan_ids = 0, 0, []

    for q in qi_items:
        rqi_id = q.get("rqi_id") or ""
        rtxt = rqi_by_id.get(rqi_id, {}).get("text", "")
        triggers = build_triggers(q["text"])
        ari = build_ari(q["text"], rtxt)
        frt = build_frt(q["text"], rtxt, triggers)
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
    qi_orphans_by_chapter = {}

    for qi in qi_pack:
        qc_id = qi.get("qc_id", "")
        cc = qc_by_id[qc_id]["chapter_code"] if qc_id and qc_id in qc_by_id else "UNMAPPED"
        qi_by_chapter.setdefault(cc, []).append(qi["qi_id"])
        if qi.get("has_rqi"):
            if qi.get("is_orphan"):
                qi_orphans_by_chapter.setdefault(cc, []).append(qi["qi_id"])
            else:
                qi_covered_by_chapter.setdefault(cc, []).append(qi["qi_id"])

    chapter_entries = []
    all_cc = sorted(set(qc_by_chapter.keys()) | set(qi_by_chapter.keys()))

    for cc in all_cc:
        qi_total = len(qi_by_chapter.get(cc, []))
        qi_posable_ch = len(qi_covered_by_chapter.get(cc, [])) + len(qi_orphans_by_chapter.get(cc, []))
        qi_covered = len(qi_covered_by_chapter.get(cc, []))
        qi_orphans = qi_orphans_by_chapter.get(cc, [])
        qc_count = len(qc_by_chapter.get(cc, []))
        cov_pct = round((qi_covered / qi_posable_ch) * 100, 2) if qi_posable_ch > 0 else 0.0
        status = "PASS" if (qi_posable_ch > 0 and cov_pct == 100.0 and qc_count > 0 and not qi_orphans) else "FAIL"

        chapter_entries.append({
            "chapter_code": cc,
            "chapter_label": ch_label.get(cc, cc),
            "qi_total": qi_total,
            "qi_posable_count": qi_posable_ch,
            "covered_qi_count": qi_covered,
            "coverage_pct": cov_pct,
            "orphans": qi_orphans,
            "orphans_count": len(qi_orphans),
            "qc_count": qc_count,
            "status": status,
        })

    chapter_report = {
        "version": APP_VERSION,
        "timestamp": _utc_ts(),
        "pack_id": st.session_state.pack_id,
        "chapters": sorted(chapter_entries, key=lambda x: (-x["qi_total"], x["chapter_code"])),
        "summary": {
            "total_chapters_active": len([e for e in chapter_entries if e["qi_total"] > 0]),
            "chapters_pass": len([e for e in chapter_entries if e["status"] == "PASS"]),
            "chapters_fail": len([e for e in chapter_entries if e["status"] == "FAIL"]),
            "global_coverage_pct": round(
                (sum(e["covered_qi_count"] for e in chapter_entries) / max(1, sum(e["qi_posable_count"] for e in chapter_entries))) * 100, 2
            ),
        },
    }

    audit = {
        "qi_total": len(qi_pack),
        "qi_posable": qi_posable,
        "rqi_total": sum(1 for q in qi_pack if q["rqi"]),
        "qc_total": len(qc_pack),
        "qi_orphans": orphan_count,
        "coverage_ok": orphan_count == 0,
        "posable_ok": qi_posable > 0,
        "qc_ok": len(qc_pack) > 0,
        "sealed_candidate": sealed_candidate,
        "sealed": sealed_candidate and orphan_count == 0 and qi_posable > 0 and len(qc_pack) > 0,
        "iterations": last_iter,
    }

    return {"qi_pack": qi_pack, "qc_pack": qc_pack, "chapter_report": chapter_report, "audit": audit}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def metric_row(items, corr_ok, qi, qi_posable, qc, sealed):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Items", items)
    c2.metric("Corrigés", corr_ok)
    c3.metric("Qi", qi)
    c4.metric("Qi POSABLE", qi_posable)
    c5.metric("QC", qc)
    c6.metric("SEALED", "YES" if sealed else "NO")


def df_table(rows):
    if not rows:
        st.info("Bibliothèque vide. Lancez HARVEST AUTO ou upload manuel.")
        return
    cols = ["pair_id", "scope", "source", "sujet", "corrige?", "corrige_name", "sujet_url", "corrige_url"]
    view = [{k: r.get(k, "") for k in cols} for r in rows]
    st.dataframe(view, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title=f"SMAXIA GTE Console {APP_VERSION}", layout="wide")
    ss_init()

    st.markdown(f"# 🔒 SMAXIA GTE Console {APP_VERSION} — ISO-PROD TEST")

    with st.sidebar:
        st.markdown("## ÉTAPE 1 — ACTIVATION")
        country = st.selectbox("Pays", options=[DEFAULT_COUNTRY], index=0)
        st.session_state.country = country

        if st.button("🔒 ACTIVER", use_container_width=True):
            try:
                pack = load_academic_pack(country)
                st.session_state.pack_active = pack
                st.session_state.pack_id = str(pack.get("pack_id") or pack.get("id") or "")
                st.success("Pack actif")
            except Exception as e:
                st.session_state.pack_active = None
                st.error(f"Activation impossible: {e}")

        st.markdown("---")
        if st.session_state.pack_active:
            st.success(f"✅ Pack: {st.session_state.pack_id}")
            src = st.session_state.pack_active.get("_source", "FILE")
            st.caption(f"Source: {src}")
        else:
            st.warning("Pack inactif")

        st.markdown("---")
        st.markdown("## ÉTAPE 2 — SÉLECTION")
        level = st.radio("Niveau", ["Seconde", "Première", "Terminale"], index=2)
        level_code = {"Seconde": "SECONDE", "Première": "PREMIERE", "Terminale": "TERMINALE"}.get(level, "TERMINALE")
        st.session_state.level = level_code

        st.markdown("### Chapitres")
        if st.session_state.pack_active:
            for c in pack_chapters(st.session_state.pack_active)[:20]:
                st.write(f"• {c['chapter_code']} — {c['chapter_label']}")

    tab1, tab2, tab3 = st.tabs(["📥 Import", "⚡ RUN", "📦 Exports"])

    lib = st.session_state.library
    corr_ok = sum(1 for x in lib if x.get("corrige?") and x.get("corrige_url"))
    qi_count = st.session_state.run_stats.get("qi", 0)
    qi_posable = st.session_state.run_stats.get("qi_posable", 0)
    qc_count = st.session_state.run_stats.get("qc", 0)
    sealed = st.session_state.sealed

    with tab1:
        st.markdown("## Import PDF")
        metric_row(len(lib), corr_ok, qi_count, qi_posable, qc_count, sealed)

        st.markdown("### Bibliothèque")
        df_table(lib)

        st.markdown("---")
        st.markdown("### HARVEST AUTO")
        c1, c2 = st.columns(2)
        years_back = c1.number_input("Années", 1, 15, 10)
        volume_max = c2.number_input("Volume max", 5, 200, 50)

        if st.button("🌐 HARVEST", use_container_width=True):
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
                    st.session_state.run_stats = {"qi": 0, "rqi": 0, "qc": 0, "qi_posable": 0}
                    st.success(f"HARVEST: +{len(manifest['library'])} items (corrigés={manifest['items_corrige_ok']})")
                except Exception as e:
                    st.error(f"HARVEST échoué: {e}")

        st.markdown("---")
        st.markdown("### Upload manuel")
        up1, up2 = st.columns(2)
        sujet_file = up1.file_uploader("PDF Sujet", type=["pdf"])
        corr_file = up2.file_uploader("PDF Corrigé", type=["pdf"])

        if st.button("➕ Ajouter"):
            if sujet_file:
                su_id = stable_id(sujet_file.name, str(len(sujet_file.getvalue())))
                co_id = stable_id(corr_file.name, str(len(corr_file.getvalue()))) if corr_file else ""

                st.session_state._uploads[f"local://{su_id}"] = sujet_file.getvalue()
                if corr_file:
                    st.session_state._uploads[f"local://{co_id}"] = corr_file.getvalue()

                item = {
                    "pair_id": f"PAIR_{st.session_state.level}|MATH_{stable_id(su_id, co_id)}",
                    "scope": f"{st.session_state.level}|MATH",
                    "source": "UPLOAD",
                    "sujet": sujet_file.name,
                    "corrige?": bool(corr_file),
                    "corrige_name": corr_file.name if corr_file else "",
                    "sujet_url": f"local://{su_id}",
                    "corrige_url": f"local://{co_id}" if corr_file else "",
                }
                st.session_state.library.insert(0, item)
                st.success("Ajouté")

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
            max_vol = max(1, min(120, corr_ok2))
            c1, c2, c3 = st.columns(3)
            volume = c1.slider("Volume", 1, max_vol, min(20, max_vol))
            sim = c2.slider("Seuil clustering", 0.05, 0.95, 0.18)
            min_cluster = c3.slider("Anti-singleton", 1, 6, 2)
            max_iters = st.slider("Max itérations", 1, 8, 4)

            if st.button("⚡ LANCER", use_container_width=True):
                try:
                    log(f"=== RUN {APP_VERSION} ===")
                    res = run_granulo_test(lib2, volume, sim, min_cluster, max_iters, st.session_state.pack_active)
                    st.session_state.qi_pack = res["qi_pack"]
                    st.session_state.qc_pack = res["qc_pack"]
                    st.session_state.chapter_report = res["chapter_report"]
                    st.session_state.last_run_audit = res["audit"]
                    st.session_state.sealed = res["audit"]["sealed"]
                    st.session_state.run_stats = {
                        "qi": res["audit"]["qi_total"],
                        "rqi": res["audit"]["rqi_total"],
                        "qc": res["audit"]["qc_total"],
                        "qi_posable": res["audit"]["qi_posable"],
                    }
                    if res["audit"]["sealed"]:
                        st.success("SEALED = YES")
                    else:
                        st.warning(f"SEALED = NO | orphans={res['audit']['qi_orphans']} | posable={res['audit']['qi_posable']}")
                except Exception as e:
                    st.error(f"RUN échoué: {e}")

        st.markdown("### Logs")
        st.text_area("", logs_text(), height=300)

    with tab3:
        st.markdown("## Exports")
        lib3 = st.session_state.library
        corr_ok3 = sum(1 for x in lib3 if x.get("corrige?") and x.get("corrige_url"))
        metric_row(len(lib3), corr_ok3, st.session_state.run_stats.get("qi", 0), st.session_state.run_stats.get("qi_posable", 0), st.session_state.run_stats.get("qc", 0), st.session_state.sealed)

        hm = st.session_state.harvest_manifest or {"version": APP_VERSION, "library": lib3}
        st.download_button("harvest_manifest.json", json.dumps(hm, ensure_ascii=False, indent=2), "harvest_manifest.json")
        st.download_button("logs.txt", logs_text(), "logs.txt")

        if st.session_state.qi_pack:
            st.download_button("qi_pack.json", json.dumps(st.session_state.qi_pack, ensure_ascii=False, indent=2), "qi_pack.json")
        if st.session_state.qc_pack:
            st.download_button("qc_pack.json", json.dumps(st.session_state.qc_pack, ensure_ascii=False, indent=2), "qc_pack.json")
        if st.session_state.chapter_report:
            st.download_button("chapter_report.json", json.dumps(st.session_state.chapter_report, ensure_ascii=False, indent=2), "chapter_report.json")

        st.markdown("---")
        st.markdown("### Explorateur")
        if st.session_state.qc_pack and st.session_state.qi_pack:
            ch_options = ["ALL"] + sorted({qc["chapter_code"] for qc in st.session_state.qc_pack})
            sel_ch = st.selectbox("Chapitre", ch_options)

            qcs = st.session_state.qc_pack if sel_ch == "ALL" else [q for q in st.session_state.qc_pack if q["chapter_code"] == sel_ch]

            if qcs:
                qc_labels = [f"{q['qc_id']} | {q['chapter_code']} | n={q['cluster_size']}" for q in qcs]
                sel_idx = st.selectbox("QC", range(len(qc_labels)), format_func=lambda i: qc_labels[i])
                qc = qcs[sel_idx]

                st.markdown(f"**QC:** {qc['qc']}")

                qi_by_id = {q["qi_id"]: q for q in st.session_state.qi_pack}
                for qi_id in qc["qi_ids"][:50]:
                    q = qi_by_id.get(qi_id)
                    if q:
                        with st.expander(f"{qi_id}"):
                            st.write(f"**Qi:** {q['qi'][:500]}")
                            st.write(f"**RQi:** {q['rqi'][:500] if q['rqi'] else '—'}")
                            st.write(f"**Triggers:** {q['triggers']}")

        if st.session_state.last_run_audit:
            st.markdown("### Audit")
            st.json(st.session_state.last_run_audit)


if __name__ == "__main__":
    main()
