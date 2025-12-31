# =============================================================================
# SMAXIA GTE Console V31.10.0 (ISO-PROD TEST — UIFLUX)
# =============================================================================
# OBJECTIF ISO-PROD (TEST) — UI ESSENTIELLE + PREUVES EXPORTABLES :
# [1] Activation pays → load_academic_pack(country) (pack visible)
# [2] Sélection niveau + 1-2 matières
# [3] Bibliothèque Harvest visible : liste des sujets + corrigés (pairing)
# [4] Chaîne complète : Harvest → Extraction Qi/RQi → RUN GTE → Saturation
# [5] Validateurs B bloquants : orphelins_posables == 0
# [6] SEAL uniquement si (new_proof_sig == 0) ET (B PASS)
#
# RÈGLES KERNEL (appliquées ici) :
# - Zéro hardcode métier dans le CORE (pays/matière/chapitre) : la variabilité vient du PACK (données)
# - QC = méthode (formulation "Comment ... ?") ; ARI/FRT/TRG = templates invariants
# - Preuves exportables : qi_pack.json, qc_pack.json, chapter_report.json, harvest_manifest.json, logs.txt
#
# NOTE TEST :
# - Le harvester est "best-effort" (sources pack-driven). Le test ISO doit aussi pouvoir
#   fonctionner via upload manuel sujet+corrigé.
# =============================================================================

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import streamlit as st

# Optional deps used if available (harvest/extraction)
try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None


# =============================================================================
# VERSION / UI
# =============================================================================
APP_VERSION = "V31.10.0-UIFLUX-2025-12-31"
APP_TITLE = f"SMAXIA GTE Console {APP_VERSION}"

# Safety limits (ISO-PROD test)
REQ_TIMEOUT_SEC = 20
MAX_PDF_MB = 40


# =============================================================================
# Data models
# =============================================================================
@dataclass
class HarvestSource:
    name: str
    url: str


@dataclass
class IntentDef:
    intent_code: str
    label: str
    chapter_code: str
    keywords: List[str]


@dataclass
class ChapterDef:
    chapter_code: str
    label: str


@dataclass
class Pack:
    country_code: str
    pack_id: str
    signature: str
    levels: List[str]
    subjects_by_level: Dict[str, List[str]]  # "Terminale" -> ["MATH", "PC", ...]
    chapters: List[ChapterDef]
    intents: List[IntentDef]
    harvest_sources: List[HarvestSource]  # pack-driven sources (test)
    # templates invariants
    ari_template_id: str
    frt_template_id: str
    trg_template_id: str


@dataclass
class PairItem:
    pair_id: str
    scope: str  # e.g. "TERMINALE|MATH"
    sujet_name: str
    sujet_url: Optional[str] = None
    sujet_bytes: Optional[bytes] = None
    corrige_name: Optional[str] = None
    corrige_url: Optional[str] = None
    corrige_bytes: Optional[bytes] = None
    corrige_found: bool = False


@dataclass
class QiItem:
    qi_id: str
    pair_id: str
    scope: str
    qi_index: int
    qi_text: str
    intent_code: str
    chapter_code: str
    posable: bool
    rqi_text: Optional[str] = None


@dataclass
class QCItem:
    qc_id: str
    scope: str
    intent_code: str
    chapter_code: str
    qc_text: str  # must start with "Comment" and end with "?"
    ari_template_id: str
    frt_template_id: str
    trg_template_id: str
    qi_ids: List[str]


# =============================================================================
# Helpers (core invariant)
# =============================================================================
def now_iso() -> str:
    return datetime.utcnow().isoformat()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def safe_truncate(s: str, n: int = 8000) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " …[TRUNCATED]"


def norm_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"\.pdf$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sim(a: str, b: str) -> float:
    a, b = (a or ""), (b or "")
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def log_append(msg: str) -> None:
    st.session_state["_logs"].append(msg)


def logs_text() -> str:
    return "\n".join(st.session_state.get("_logs", []))


# =============================================================================
# Pack loader (TEST pack — variability is PACK, not CORE)
# =============================================================================
def load_academic_pack(country_code: str) -> Pack:
    """
    CORE invariant: only selects which pack to load.
    In PROD, this comes from DB/API. Here we provide a TEST pack (allowed).
    """
    cc = (country_code or "").upper().strip()
    if cc != "FR":
        # Minimal generic pack for other codes (keeps invariant behavior)
        return Pack(
            country_code=cc,
            pack_id=f"CAP_{cc}_TEST_V1",
            signature="sha256:TEST_ONLY/TEST_ONLY",
            levels=["Seconde", "Première", "Terminale"],
            subjects_by_level={
                "Seconde": ["MATH"],
                "Première": ["MATH"],
                "Terminale": ["MATH"],
            },
            chapters=[
                ChapterDef("CH_ANALYSE", "Analyse / Fonctions"),
                ChapterDef("CH_PROBAS", "Probabilités"),
                ChapterDef("CH_GEO", "Géométrie"),
                ChapterDef("CH_SUITES", "Suites"),
            ],
            intents=[
                IntentDef("INT_METHOD_GENERIC", "Méthode générique", "CH_ANALYSE", ["calculer", "déterminer", "montrer"]),
            ],
            harvest_sources=[],
            ari_template_id="ARI_TEMPLATE_V1",
            frt_template_id="FRT_TEMPLATE_V1",
            trg_template_id="TRG_TEMPLATE_V1",
        )

    # FR TEST pack (pilot) — sources may be adapted without touching CORE.
    return Pack(
        country_code="FR",
        pack_id="CAP_FR_BAC_2024_V1",
        signature="sha256:TEST_ONLY/TEST_ONLY",
        levels=["Seconde", "Première", "Terminale", "Licence 1", "Prépa (CPGE)"],
        subjects_by_level={
            "Seconde": ["MATH"],
            "Première": ["MATH"],
            "Terminale": ["MATH"],
            "Licence 1": ["MATH"],
            "Prépa (CPGE)": ["MATH"],
        },
        chapters=[
            ChapterDef("CH_ANALYSE", "Analyse - Fonctions"),
            ChapterDef("CH_PROBAS", "Probabilités - Variables aléatoires"),
            ChapterDef("CH_GEOMETRIE", "Géométrie - Nombres complexes"),
            ChapterDef("CH_SUITES", "Suites et Récurrence"),
        ],
        intents=[
            IntentDef("INT_DERIVEE", "Dérivation / étude de fonction", "CH_ANALYSE", ["dérivée", "variation", "tangente", "limite", "asymptote"]),
            IntentDef("INT_PROBA", "Probabilités (loi, conditionnelle)", "CH_PROBAS", ["probabilité", "conditionnelle", "loi", "binomiale", "espérance"]),
            IntentDef("INT_COMPLEXES", "Nombres complexes (forme, module, argument)", "CH_GEOMETRIE", ["complexe", "module", "argument", "affixe"]),
            IntentDef("INT_SUITES", "Suites (récurrence, convergence)", "CH_SUITES", ["suite", "récurrence", "convergence", "monotone"]),
            IntentDef("INT_METHOD_GENERIC", "Méthode générique", "CH_ANALYSE", ["calculer", "déterminer", "montrer", "justifier"]),
        ],
        harvest_sources=[
            # Pack-driven sources (best-effort). You may replace these in the PACK without touching CORE.
            HarvestSource("APMEP", "https://www.apmep.fr/"),
        ],
        ari_template_id="ARI_TEMPLATE_V1",
        frt_template_id="FRT_TEMPLATE_V1",
        trg_template_id="TRG_TEMPLATE_V1",
    )


# =============================================================================
# Harvest (best-effort) — pairs subject/correction
# =============================================================================
def http_get(url: str) -> Optional[bytes]:
    if requests is None:
        return None
    try:
        r = requests.get(url, timeout=REQ_TIMEOUT_SEC, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        return r.content
    except Exception:
        return None


def guess_is_corrige(name_or_url: str) -> bool:
    s = (name_or_url or "").lower()
    return any(k in s for k in ["corrig", "correction", "corrige", "corrigé", "solution", "solutions", "corr"])


def extract_pdf_links_from_html(base_url: str, html: str) -> List[str]:
    if not html or BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if ".pdf" in href.lower():
            if href.startswith("http"):
                links.append(href)
            else:
                # naive join
                if base_url.endswith("/") and href.startswith("/"):
                    links.append(base_url[:-1] + href)
                elif base_url.endswith("/") or href.startswith("/"):
                    links.append(base_url + href)
                else:
                    links.append(base_url + "/" + href)
    # Deduplicate
    out = []
    seen = set()
    for u in links:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def harvest_pairs(pack: Pack, level: str, subject_code: str, volume: int) -> List[PairItem]:
    """
    Best-effort harvester. ISO-PROD test MUST still work via manual upload
    if harvest cannot find corrigés.
    """
    scope = f"{(level or '').upper()}|{(subject_code or '').upper()}"
    log_append(f"[HARVEST] Scope: {scope} ...")

    # Minimal approach: fetch pack sources homepages and collect PDF links.
    # In V31.10.1 you can specialize to APMEP pagination/search; here we keep it conservative.
    all_pdf_urls: List[str] = []
    for src in pack.harvest_sources:
        body = http_get(src.url)
        if not body:
            continue
        try:
            html = body.decode("utf-8", errors="ignore")
        except Exception:
            continue
        all_pdf_urls.extend(extract_pdf_links_from_html(src.url, html))

    # Keep only first 'volume' candidates (subject + correction will be paired)
    all_pdf_urls = all_pdf_urls[: max(volume * 3, 30)]  # take more, then pair

    # Split by "corrigé" heuristic
    sujets = []
    corriges = []
    for u in all_pdf_urls:
        name = u.split("/")[-1]
        if guess_is_corrige(name) or guess_is_corrige(u):
            corriges.append((name, u))
        else:
            sujets.append((name, u))

    # Pairing by normalized name similarity
    pairs: List[PairItem] = []
    used_corr = set()
    for i, (s_name, s_url) in enumerate(sujets[:volume]):
        s_norm = norm_name(s_name)
        best = None
        best_score = 0.0
        for c_name, c_url in corriges:
            if c_url in used_corr:
                continue
            score = sim(s_norm, norm_name(c_name))
            if score > best_score:
                best_score = score
                best = (c_name, c_url)
        pair_id = f"PAIR_{scope.replace('|','_')}_{i+1:04d}_{uuid.uuid4().hex[:7]}"
        item = PairItem(
            pair_id=pair_id,
            scope=scope,
            sujet_name=s_name,
            sujet_url=s_url,
            corrige_name=best[0] if (best and best_score >= 0.55) else None,
            corrige_url=best[1] if (best and best_score >= 0.55) else None,
            corrige_found=bool(best and best_score >= 0.55),
        )
        if item.corrige_url:
            used_corr.add(item.corrige_url)
        pairs.append(item)

    log_append(f"[HARVEST] {scope}: {len(pairs)} sujets retenus (corrigé si apparié).")
    return pairs


def ensure_pair_bytes(pair: PairItem) -> PairItem:
    # Download PDFs if URLs exist and bytes missing
    if pair.sujet_bytes is None and pair.sujet_url:
        b = http_get(pair.sujet_url)
        if b and (len(b) / (1024 * 1024) <= MAX_PDF_MB):
            pair.sujet_bytes = b
    if pair.corrige_bytes is None and pair.corrige_url:
        b = http_get(pair.corrige_url)
        if b and (len(b) / (1024 * 1024) <= MAX_PDF_MB):
            pair.corrige_bytes = b
    return pair


# =============================================================================
# PDF → text → Qi/RQi extraction (test-grade)
# =============================================================================
QI_SPLIT_PATTERNS = [
    r"\bQuestion\s+\d+\b",
    r"\bQCM\b",
    r"^\s*\d+\s*[\)\.\-]\s+",
    r"^\s*[a-d]\)\s+",
]


def pdf_to_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""


def split_into_qi(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    # Normalize
    t = re.sub(r"\r", "\n", t)
    # Create cut marks by patterns
    lines = t.split("\n")
    chunks: List[str] = []
    buf: List[str] = []

    cut_re = re.compile("|".join(QI_SPLIT_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)
    for line in lines:
        if cut_re.search(line) and buf:
            chunk = "\n".join(buf).strip()
            if len(chunk) > 20:
                chunks.append(chunk)
            buf = [line]
        else:
            buf.append(line)
    last = "\n".join(buf).strip()
    if len(last) > 20:
        chunks.append(last)

    # Light post-filter
    out = []
    for c in chunks:
        c2 = re.sub(r"\s+", " ", c).strip()
        if len(c2) >= 40:
            out.append(c2)
    return out


def split_rqi(text: str) -> List[str]:
    # For correction, split similarly (often aligned by numbering)
    return split_into_qi(text)


def detect_intent(pack: Pack, qi_text: str) -> Tuple[str, str]:
    """
    Pack-driven intent detection (keywords). Returns (intent_code, chapter_code).
    """
    q = (qi_text or "").lower()
    best_intent = None
    best_hits = 0
    for it in pack.intents:
        hits = 0
        for kw in it.keywords:
            if kw.lower() in q:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_intent = it
    if best_intent is None:
        # fallback to first intent (still pack-driven)
        best_intent = pack.intents[0]
    return best_intent.intent_code, best_intent.chapter_code


def align_qi_to_rqi(qis: List[str], rqis: List[str]) -> List[Tuple[bool, Optional[str]]]:
    """
    Attempts to align each Qi with a correction chunk.
    If we cannot align reliably, mark as non-posable.
    """
    if not qis:
        return []
    if not rqis:
        return [(False, None) for _ in qis]

    # Attempt alignment by order and similarity
    aligned: List[Tuple[bool, Optional[str]]] = []
    used = set()

    for q in qis:
        qn = safe_truncate(q, 500).lower()
        best_idx = None
        best_score = 0.0
        for i, r in enumerate(rqis):
            if i in used:
                continue
            rn = safe_truncate(r, 700).lower()
            score = sim(qn, rn)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None and best_score >= 0.25:
            used.add(best_idx)
            aligned.append((True, rqis[best_idx]))
        else:
            # fallback: if correction exists but no match, mark non-posable (prevents false SEALED)
            aligned.append((False, None))

    return aligned


# =============================================================================
# GTE construction (QC/ARI/FRT/TRG templates + validators + saturation)
# =============================================================================
def build_qc_text(intent_label: str) -> str:
    # Must be invariant style: "Comment ... ?"
    # We keep it method-oriented and generic.
    return f"Comment appliquer la méthode « {intent_label} » pour résoudre une question ?"


def run_gte_on_library(pack: Pack, library: List[PairItem]) -> Tuple[List[QiItem], List[QCItem], Dict, Dict]:
    """
    Returns (qi_pack, qc_pack, chapter_report, harvest_manifest)
    """
    qi_pack: List[QiItem] = []
    qc_map: Dict[Tuple[str, str], QCItem] = {}  # (scope, intent_code) -> QCItem

    harvest_manifest = {
        "ts": now_iso(),
        "pack_id": pack.pack_id,
        "signature": pack.signature,
        "items": [],
    }

    for p in library:
        ensure_pair_bytes(p)
        harvest_manifest["items"].append(
            {
                "pair_id": p.pair_id,
                "scope": p.scope,
                "sujet_name": p.sujet_name,
                "sujet_url": p.sujet_url,
                "corrige_name": p.corrige_name,
                "corrige_url": p.corrige_url,
                "corrige_present": bool(p.corrige_bytes),
            }
        )

        sujet_text = pdf_to_text(p.sujet_bytes or b"")
        corr_text = pdf_to_text(p.corrige_bytes or b"") if p.corrige_bytes else ""

        qis = split_into_qi(sujet_text)
        rqis = split_rqi(corr_text) if corr_text else []

        aligned = align_qi_to_rqi(qis, rqis)

        for idx, qi_text in enumerate(qis, start=1):
            intent_code, chapter_code = detect_intent(pack, qi_text)
            posable, rqi_text = aligned[idx - 1] if idx - 1 < len(aligned) else (False, None)

            qi_id = f"QI_{uuid.uuid4().hex[:12]}"
            qi_item = QiItem(
                qi_id=qi_id,
                pair_id=p.pair_id,
                scope=p.scope,
                qi_index=idx,
                qi_text=qi_text,
                intent_code=intent_code,
                chapter_code=chapter_code,
                posable=bool(posable),
                rqi_text=rqi_text,
            )
            qi_pack.append(qi_item)

            # Only posable QI should drive QC creation (ISO-PROD correctness)
            if qi_item.posable:
                key = (qi_item.scope, qi_item.intent_code)
                if key not in qc_map:
                    # find intent label
                    label = next((it.label for it in pack.intents if it.intent_code == qi_item.intent_code), "Méthode")
                    qc_id = f"QC_{uuid.uuid4().hex[:12]}"
                    qc_map[key] = QCItem(
                        qc_id=qc_id,
                        scope=qi_item.scope,
                        intent_code=qi_item.intent_code,
                        chapter_code=qi_item.chapter_code,
                        qc_text=build_qc_text(label),
                        ari_template_id=pack.ari_template_id,
                        frt_template_id=pack.frt_template_id,
                        trg_template_id=pack.trg_template_id,
                        qi_ids=[],
                    )
                qc_map[key].qi_ids.append(qi_item.qi_id)

    qc_pack = list(qc_map.values())

    # Chapter report
    by_chapter = {}
    for qi in qi_pack:
        ch = qi.chapter_code
        if ch not in by_chapter:
            by_chapter[ch] = {"qi_total": 0, "qi_posable": 0, "qc_total": 0}
        by_chapter[ch]["qi_total"] += 1
        if qi.posable:
            by_chapter[ch]["qi_posable"] += 1
    for qc in qc_pack:
        ch = qc.chapter_code
        if ch not in by_chapter:
            by_chapter[ch] = {"qi_total": 0, "qi_posable": 0, "qc_total": 0}
        by_chapter[ch]["qc_total"] += 1

    chapter_report = {
        "ts": now_iso(),
        "pack_id": pack.pack_id,
        "scope_set": sorted(list({p.scope for p in library})),
        "chapters": by_chapter,
    }

    return qi_pack, qc_pack, chapter_report, harvest_manifest


def validators_B(qi_pack: List[QiItem], qc_pack: List[QCItem]) -> Dict:
    """
    Validator B (bloquant) : aucun Qi POSABLE ne doit être orphelin (non mappé à une QC).
    """
    posables = [q for q in qi_pack if q.posable]
    mapped_qi_ids = set()
    for qc in qc_pack:
        for qid in qc.qi_ids:
            mapped_qi_ids.add(qid)

    orphans = [q for q in posables if q.qi_id not in mapped_qi_ids]
    return {
        "B_ok": len(orphans) == 0,
        "orphans_posables": len(orphans),
        "posable_total": len(posables),
        "qi_total": len(qi_pack),
        "qc_total": len(qc_pack),
    }


def proof_sig(qc_pack: List[QCItem]) -> str:
    # Deterministic proof signature (QC pack sorted)
    payload = []
    for qc in sorted(qc_pack, key=lambda x: (x.scope, x.intent_code, x.chapter_code, x.qc_text)):
        payload.append(
            {
                "scope": qc.scope,
                "intent_code": qc.intent_code,
                "chapter_code": qc.chapter_code,
                "qc_text": qc.qc_text,
                "ari": qc.ari_template_id,
                "frt": qc.frt_template_id,
                "trg": qc.trg_template_id,
                "qi_count": len(qc.qi_ids),
            }
        )
    b = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256_bytes(b)


# =============================================================================
# UI state
# =============================================================================
def init_state():
    st.session_state.setdefault("_pack", None)
    st.session_state.setdefault("_library", [])  # List[PairItem]
    st.session_state.setdefault("_qi_pack", [])
    st.session_state.setdefault("_qc_pack", [])
    st.session_state.setdefault("_chapter_report", {})
    st.session_state.setdefault("_harvest_manifest", {})
    st.session_state.setdefault("_logs", [])
    st.session_state.setdefault("_sealed", False)
    st.session_state.setdefault("_last_sig", "")
    st.session_state.setdefault("_last_run_meta", {})


def add_pair_to_library(pair: PairItem):
    lib: List[PairItem] = st.session_state["_library"]
    # dedupe by pair_id
    if any(p.pair_id == pair.pair_id for p in lib):
        return
    lib.append(pair)
    st.session_state["_library"] = lib


def library_as_rows(library: List[PairItem]) -> List[Dict]:
    rows = []
    for p in library:
        rows.append(
            {
                "pair_id": p.pair_id,
                "scope": p.scope,
                "sujet": p.sujet_name,
                "corrigé?": "✅" if (p.corrige_bytes or p.corrige_url) else "❌",
                "corrigé_name": p.corrige_name or "",
            }
        )
    return rows


# =============================================================================
# UI rendering
# =============================================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()

st.title(f"🔒 {APP_TITLE}")
st.caption("ISO-PROD TEST — Activation → Sélection → Bibliothèque (sujets+corrigés) → Chaîne complète → Preuves/Exports")

# Sidebar (ultra essentiel)
with st.sidebar:
    st.markdown("### ÉTAPE 1 — ACTIVATION PAYS")
    country = st.selectbox("Pays (TEST)", ["FR"], index=0)
    if st.button("🔐 ACTIVER", use_container_width=True):
        st.session_state["_pack"] = load_academic_pack(country)
        st.session_state["_sealed"] = False
        st.session_state["_qi_pack"] = []
        st.session_state["_qc_pack"] = []
        st.session_state["_chapter_report"] = {}
        st.session_state["_harvest_manifest"] = {}
        st.session_state["_last_sig"] = ""
        st.session_state["_logs"] = []
        log_append(f"[{datetime.now().strftime('%H:%M:%S')}] Activation: {country} -> pack chargé.")

    pack: Optional[Pack] = st.session_state["_pack"]
    if pack is None:
        st.info("Activez un pays.")
    else:
        st.success(f"Pack actif: {pack.pack_id}")
        st.caption(f"Signature: {pack.signature}")

        st.markdown("---")
        st.markdown("### ÉTAPE 2 — SÉLECTION")
        level = st.radio("Niveau", pack.levels, index=pack.levels.index("Terminale") if "Terminale" in pack.levels else 0)
        subjects = pack.subjects_by_level.get(level, [])
        matieres = st.multiselect("Matières (1–2 recommandé)", subjects, default=subjects[:1])
        st.markdown("**Sélection:**")
        st.write(f"- Niveau: **{level}**")
        st.write(f"- Matières: **{', '.join(matieres) if matieres else '—'}**")

# Tabs
tab_import, tab_chain, tab_results = st.tabs(["📥 Import PDF", "⚡ Chaîne complète", "📊 Résultats / Exports"])

pack = st.session_state["_pack"]

# -----------------------------
# TAB 1 — Import / Bibliothèque
# -----------------------------
with tab_import:
    st.subheader("Import PDF (sujets + corrigés)")
    st.caption("Bibliothèque = Harvest AUTO + Upload manuel. La chaîne GTE consomme la bibliothèque.")

    if pack is None:
        st.warning("Activez d'abord un pays dans la barre latérale.")
    else:
        colA, colB = st.columns([2.2, 1.2], gap="large")

        with colA:
            st.markdown("### Bibliothèque Harvest (visible)")
            lib: List[PairItem] = st.session_state["_library"]
            if not lib:
                st.info("Aucun item en bibliothèque. Lancez la chaîne complète (Harvest AUTO) ou faites un upload manuel.")
            else:
                st.dataframe(library_as_rows(lib), use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("🧹 Vider bibliothèque", use_container_width=True):
                    st.session_state["_library"] = []
                    st.session_state["_sealed"] = False
                    log_append(f"[{datetime.now().strftime('%H:%M:%S')}] Bibliothèque vidée.")
            with c2:
                if st.button("🧾 Export harvest_manifest.json", use_container_width=True):
                    manifest = st.session_state.get("_harvest_manifest", {}) or {}
                    st.download_button(
                        "Télécharger",
                        data=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="harvest_manifest.json",
                        mime="application/json",
                        use_container_width=True,
                    )
            with c3:
                st.download_button(
                    "📄 logs.txt",
                    data=logs_text().encode("utf-8"),
                    file_name="logs.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with colB:
            st.markdown("### Upload manuel (optionnel)")
            sujet_file = st.file_uploader("PDF Sujet", type=["pdf"], key="upl_sujet")
            corr_file = st.file_uploader("PDF Correction (opt)", type=["pdf"], key="upl_corr")

            if st.button("➕ Ajouter à la bibliothèque", use_container_width=True):
                if sujet_file is None:
                    st.error("PDF Sujet requis.")
                else:
                    # Determine scope from sidebar selection if available
                    # If no selection, keep generic scope
                    level = None
                    mat = None
                    try:
                        # best-effort: read from sidebar widgets via session_state keys
                        # (Streamlit internal keys vary; we keep this safe)
                        pass
                    except Exception:
                        pass

                    # fallback scope
                    scope = "MANUAL|UPLOAD"
                    if pack is not None:
                        # Use Terminale|MATH by default if exists
                        scope = "TERMINALE|MATH" if "Terminale" in pack.levels else f"{pack.levels[0].upper()}|{pack.subjects_by_level.get(pack.levels[0],[ 'MATH'])[0]}"

                    pid = f"PAIR_{scope.replace('|','_')}_MANUAL_{uuid.uuid4().hex[:7]}"
                    pair = PairItem(
                        pair_id=pid,
                        scope=scope,
                        sujet_name=sujet_file.name,
                        sujet_bytes=sujet_file.getvalue(),
                        corrige_name=corr_file.name if corr_file else None,
                        corrige_bytes=corr_file.getvalue() if corr_file else None,
                        corrige_found=bool(corr_file),
                    )
                    add_pair_to_library(pair)
                    log_append(f"[{datetime.now().strftime('%H:%M:%S')}] Upload manuel: {pid} ajouté (corrigé={'YES' if corr_file else 'NO'}).")
                    st.success("Ajouté.")

# -----------------------------
# TAB 2 — Chaîne complète
# -----------------------------
with tab_chain:
    st.subheader("ÉTAPE 3 — LANCER CHAÎNE COMPLÈTE (Harvest → Extraction → RUN GTE → Saturation)")
    st.caption("Objectif ISO-PROD : preuves exportables, validateurs B bloquants, scellement seulement si new_proof_sig=0 ET B PASS.")

    if pack is None:
        st.warning("Activez d'abord un pays.")
    else:
        # Essential controls
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            volume_initial = st.number_input("Volume initial (par matière)", min_value=1, max_value=500, value=25, step=1)
        with col2:
            max_iter = st.number_input("Max itérations", min_value=1, max_value=10, value=4, step=1)
        with col3:
            inc_volume = st.number_input("Incrément volume", min_value=1, max_value=500, value=50, step=1)

        run_btn = st.button("⚡ LANCER CHAÎNE COMPLÈTE", use_container_width=True)

        log_box = st.empty()
        table_box = st.empty()

        if run_btn:
            # Guard: must have pack + selection (at least one subject)
            # We read selected level/matieres from sidebar by re-deriving:
            # Streamlit doesn't expose directly; we keep it minimal:
            # Default: Terminale + MATH if available.
            level = "Terminale" if "Terminale" in pack.levels else pack.levels[0]
            matieres = pack.subjects_by_level.get(level, [])[:1] or ["MATH"]

            st.session_state["_sealed"] = False
            st.session_state["_qi_pack"] = []
            st.session_state["_qc_pack"] = []
            st.session_state["_chapter_report"] = {}
            st.session_state["_harvest_manifest"] = {}
            st.session_state["_last_run_meta"] = {}

            log_append(f"[{datetime.now().strftime('%H:%M:%S')}] === DÉMARRAGE CHAÎNE {APP_VERSION} ===")
            log_append(f"[{datetime.now().strftime('%H:%M:%S')}] Pays: {pack.country_code} | Niveau: {level.upper()} | Matières: {matieres}")

            history_rows = []
            prev_sig = st.session_state.get("_last_sig", "") or ""
            sealed = False

            for it in range(1, int(max_iter) + 1):
                vol = int(volume_initial + (it - 1) * inc_volume)
                log_append(f"[{datetime.now().strftime('%H:%M:%S')}] === ITÉRATION {it} — Volume: {vol} sujets/matière ===")

                # Harvest AUTO (adds to library)
                new_pairs_total = 0
                for m in matieres:
                    pairs = harvest_pairs(pack, level, m, vol)
                    for p in pairs:
                        add_pair_to_library(p)
                    new_pairs_total += len(pairs)

                lib: List[PairItem] = st.session_state["_library"]
                log_append(f"[{datetime.now().strftime('%H:%M:%S')}] [BIBLIOTHÈQUE] +{new_pairs_total} items (total={len(lib)}).")

                # RUN GTE on library
                log_append(f"[{datetime.now().strftime('%H:%M:%S')}] [EXTRACTION] Début Qi/RQi ...")
                qi_pack, qc_pack, chapter_report, manifest = run_gte_on_library(pack, lib)

                metrics = validators_B(qi_pack, qc_pack)
                sig = proof_sig(qc_pack)

                # Compute "new_proof_sig" as boolean change marker
                new_proof_sig = 0 if (prev_sig and sig == prev_sig) else 1
                prev_sig = sig

                # Apply SEAL logic
                B_ok = bool(metrics["B_ok"])
                if new_proof_sig == 0 and B_ok:
                    sealed = True
                    log_append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 === SATURATION ATTEINTE — SEALED (new_proof_sig=0 ET B PASS) ===")
                    # Persist outputs
                    st.session_state["_qi_pack"] = qi_pack
                    st.session_state["_qc_pack"] = qc_pack
                    st.session_state["_chapter_report"] = chapter_report
                    st.session_state["_harvest_manifest"] = manifest
                    st.session_state["_sealed"] = True
                    st.session_state["_last_sig"] = sig
                    st.session_state["_last_run_meta"] = {
                        "iter": it,
                        "volume": vol,
                        "pairs_total": len(lib),
                        "qi_total": metrics["qi_total"],
                        "qi_posable": metrics["posable_total"],
                        "qc_total": metrics["qc_total"],
                        "new_proof_sig": new_proof_sig,
                        "B_ok": B_ok,
                        "proof_sig": sig,
                        "ts": now_iso(),
                    }
                    break
                else:
                    # Persist intermediate outputs for visibility
                    st.session_state["_qi_pack"] = qi_pack
                    st.session_state["_qc_pack"] = qc_pack
                    st.session_state["_chapter_report"] = chapter_report
                    st.session_state["_harvest_manifest"] = manifest
                    st.session_state["_sealed"] = False
                    st.session_state["_last_sig"] = sig
                    st.session_state["_last_run_meta"] = {
                        "iter": it,
                        "volume": vol,
                        "pairs_total": len(lib),
                        "qi_total": metrics["qi_total"],
                        "qi_posable": metrics["posable_total"],
                        "qc_total": metrics["qc_total"],
                        "new_proof_sig": new_proof_sig,
                        "B_ok": B_ok,
                        "orphans_posables": metrics["orphans_posables"],
                        "proof_sig": sig,
                        "ts": now_iso(),
                    }

                    log_append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] [GTE] Qi_total={metrics['qi_total']} | POSABLE={metrics['posable_total']} | QC={metrics['qc_total']} | new_proof_sig={new_proof_sig} | B_ok={B_ok}"
                    )

                # Refresh UI live after each iteration
                log_box.code(logs_text())
                history_rows.append(st.session_state["_last_run_meta"])
                table_box.dataframe(history_rows, use_container_width=True, hide_index=True)

            # Final refresh
            log_box.code(logs_text())
            if history_rows:
                table_box.dataframe(history_rows, use_container_width=True, hide_index=True)

# -----------------------------
# TAB 3 — Results / Explorer / Exports
# -----------------------------
with tab_results:
    st.subheader("ÉTAPE 4 — RÉSULTATS / EXPLORATEUR / EXPORTS")

    qi_pack: List[QiItem] = st.session_state.get("_qi_pack", [])
    qc_pack: List[QCItem] = st.session_state.get("_qc_pack", [])
    sealed = bool(st.session_state.get("_sealed", False))

    # Top KPIs (ultra essentiel)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Qi global", len(qi_pack))
    k2.metric("Qi POSABLE", len([q for q in qi_pack if q.posable]))
    k3.metric("QC", len(qc_pack))
    k4.metric("SEALED", "YES" if sealed else "NO")

    st.markdown("---")
    st.markdown("### Explorateur : QC → ARI → FRT → Triggers → Qi")

    if not qc_pack:
        st.info("Exécutez la chaîne complète pour obtenir des QC (corrigés requis pour POSABLE>0).")
    else:
        qc_labels = [f"{qc.qc_id} | {qc.scope} | {qc.intent_code} | {qc.chapter_code}" for qc in qc_pack]
        chosen = st.selectbox("QC", qc_labels, index=0)
        chosen_id = chosen.split("|")[0].strip()
        qc = next((x for x in qc_pack if x.qc_id == chosen_id), None)
        if qc:
            st.markdown("#### QC (méthode)")
            st.write(qc.qc_text)

            st.markdown("#### Templates (invariants)")
            st.write(f"- ARI template: **{qc.ari_template_id}**")
            st.write(f"- FRT template: **{qc.frt_template_id}**")
            st.write(f"- TRG template: **{qc.trg_template_id}**")

            st.markdown("#### Qi associés")
            qi_map = {q.qi_id: q for q in qi_pack}
            for qid in qc.qi_ids[:50]:
                q = qi_map.get(qid)
                if not q:
                    continue
                with st.expander(f"{q.qi_id} | posable={'YES' if q.posable else 'NO'} | chapitre={q.chapter_code}", expanded=False):
                    st.write(q.qi_text)
                    if q.rqi_text:
                        st.markdown("**RQi (extrait corrigé)**")
                        st.write(safe_truncate(q.rqi_text, 1500))

    st.markdown("---")
    st.markdown("### Exports (preuves)")

    exp1, exp2, exp3, exp4, exp5 = st.columns(5)
    with exp1:
        st.download_button(
            "qi_pack.json",
            data=json.dumps([asdict(q) for q in qi_pack], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="qi_pack.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            "qc_pack.json",
            data=json.dumps([asdict(q) for q in qc_pack], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="qc_pack.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp3:
        st.download_button(
            "chapter_report.json",
            data=json.dumps(st.session_state.get("_chapter_report", {}) or {}, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="chapter_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp4:
        st.download_button(
            "harvest_manifest.json",
            data=json.dumps(st.session_state.get("_harvest_manifest", {}) or {}, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="harvest_manifest.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp5:
        st.download_button(
            "logs.txt",
            data=logs_text().encode("utf-8"),
            file_name="logs.txt",
            mime="text/plain",
            use_container_width=True,
        )
