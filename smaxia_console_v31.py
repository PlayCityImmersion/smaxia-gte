# =============================================================================
# SMAXIA GTE Console V31.10.1 — ISO-PROD TEST (UI EPUREE + HARVEST CORRIGES RENFORCE)
# =============================================================================
# RÈGLES CLÉS (CAS 1 ONLY - TEST) :
# - On ne traite que les sujets avec corrigé exploitable (sinon Qi quarantainées)
# - SEAL = OUI uniquement si :
#     (1) POSABLE > 0
#     (2) QC_total > 0
#     (3) 0 orphelin POSABLE (validator B bloquant)
#     (4) anti-singleton QC respecté (cluster=1 interdit)
#     (5) saturation atteinte (preuve stable sur itération)
#
# NOTE : Ce fichier est une console Streamlit de TEST ISO-PROD (preuve/exports),
#        pas le KERNEL PROD final. Aucun hardcode "chapitre/métier" dans les templates :
#        tout ce qui est "chapitre/intent" vient du Pack (données).
# =============================================================================

from __future__ import annotations

import io
import json
import os
import re
import time
import hashlib
import random
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
import pdfplumber
from bs4 import BeautifulSoup

# ------------------------------
# META
# ------------------------------
APP_NAME = "SMAXIA GTE Console"
APP_VERSION = "V31.10.1"
APP_FLAVOR = "ISO-PROD TEST"
BUILD = "2025-12-31"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 25
MAX_PDF_MB = 35
MAX_CRAWL_PAGES_DEFAULT = 12

# Clustering (invariant, text-structure based)
SIM_THRESHOLD = 0.18         # plus bas = clusters plus gros (évite singletons)
MIN_CLUSTER_SIZE = 2         # anti-singleton (normatif)
MAX_QI_CHARS = 1800          # sécurise exports/affichage


# =============================================================================
# UTILITAIRES
# =============================================================================
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)


def norm_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00A0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def safe_filename(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:160] if len(name) > 160 else name


def same_domain(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()
    except Exception:
        return False


def guess_is_pdf(url: str) -> bool:
    u = (url or "").lower()
    return ".pdf" in u


def guess_is_correction(name_or_url: str) -> bool:
    s = norm_text(name_or_url)
    # FR/EN invariant-ish keywords for correction artifacts (not chapter hardcode)
    keys = [
        "corrige", "corrigé", "correction", "solutions", "solution",
        "soluce", "reponse", "réponse", "answers", "answer", "markscheme",
        "corr", "sol"
    ]
    return any(k in s for k in keys)


def normalize_pair_key(name_or_url: str) -> str:
    """
    Normalize for fuzzy pairing:
    remove correction markers, generic tokens, punctuation.
    """
    s = norm_text(name_or_url)
    # remove file extension & query parts
    s = re.sub(r"\.pdf.*$", "", s)
    # remove common non-semantic tokens
    s = re.sub(r"(corrige|corrigé|correction|solutions|solution|reponse|réponse|answers|answer|markscheme)", " ", s)
    s = re.sub(r"(sujet|subject|epreuve|épreuve|exam|examen|session|terminale|premiere|seconde|maths|math)", " ", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# =============================================================================
# DATA MODELS
# =============================================================================
@dataclass(frozen=True)
class HarvestSource:
    name: str
    start_urls: List[str]
    max_pages: int = MAX_CRAWL_PAGES_DEFAULT


@dataclass
class PdfArtifact:
    name: str
    url: Optional[str]
    sha256: Optional[str]
    size_bytes: int
    bytes_b64: Optional[str] = None  # not used (kept to show extensibility)


@dataclass
class LibraryItem:
    pair_id: str
    scope: str
    sujet: PdfArtifact
    corrige: Optional[PdfArtifact]
    corrige_found: bool
    corrige_dl_ok: bool
    source: str
    added_at: str


@dataclass
class Pack:
    country: str
    pack_id: str
    signature: str
    levels: List[str]
    subjects: Dict[str, List[str]]          # level -> subjects
    chapters: Dict[str, List[str]]          # subject -> chapters
    harvest_sources: List[HarvestSource]


# =============================================================================
# PACK LOADER (TEST PACK)
# =============================================================================
def load_academic_pack(country: str) -> Pack:
    # TEST PACK (data-only). En PROD : vient DB/API.
    # Ici, on limite le champ au strict nécessaire pour tester l’ISO-PROD.
    if country != "FR":
        raise ValueError("TEST pack disponible uniquement pour FR dans cette console.")

    pack_id = "CAP_FR_BAC_2024_V1"
    signature = "sha256:TEST_ONLY/TEST_ONLY"

    levels = ["Seconde", "Première", "Terminale", "Licence 1", "Prépa (CPGE)"]
    subjects = {
        "Seconde": ["MATH"],
        "Première": ["MATH"],
        "Terminale": ["MATH"],
        "Licence 1": ["MATH"],
        "Prépa (CPGE)": ["MATH"],
    }

    # Chapitres pack-driven (liste indicative de test)
    chapters = {
        "MATH": [
            "CH_ANALYSE",
            "CH_PROBAS",
            "CH_GEOMETRIE",
            "CH_SUITES",
        ]
    }

    # SOURCES : mettre ici des pages de listing qui contiennent des liens .pdf
    # IMPORTANT : cette console sait crawler plusieurs pages (pagination) automatiquement.
    harvest_sources = [
        HarvestSource(
            name="APMEP (listing)",
            start_urls=[
                # Remplace/ajoute ici des URLs de listing pertinentes si besoin
                # (la console gère la pagination et collecte tous les .pdf du domaine).
                "https://www.apmep.fr/Annales-Terminale",
            ],
            max_pages=MAX_CRAWL_PAGES_DEFAULT,
        ),
    ]

    return Pack(
        country=country,
        pack_id=pack_id,
        signature=signature,
        levels=levels,
        subjects=subjects,
        chapters=chapters,
        harvest_sources=harvest_sources,
    )


# =============================================================================
# HARVEST ENGINE (multi-pages + fuzzy pairing)
# =============================================================================
def http_get(url: str) -> requests.Response:
    headers = {"User-Agent": UA}
    return requests.get(url, headers=headers, timeout=REQ_TIMEOUT, allow_redirects=True)


def fetch_html(url: str) -> str:
    r = http_get(url)
    r.raise_for_status()
    return r.text


def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        absu = urljoin(base_url, href)
        links.append(absu)
    # unique, stable order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def is_pagination_link(url: str, domain_root: str) -> bool:
    if not same_domain(url, domain_root):
        return False
    s = url.lower()
    # generic pagination patterns
    patterns = [
        "page=", "/page/", "p=", "start=", "offset=", "limit=",
        "suivant", "next", "older", "plus-recent", "plus-ancien",
    ]
    return any(p in s for p in patterns)


def crawl_domain_pdfs(start_url: str, max_pages: int, log: List[str]) -> List[str]:
    """
    Crawl within same domain, follow pagination-like links, collect pdf links.
    Deterministic: BFS order.
    """
    domain_root = start_url
    q = [start_url]
    visited = set()
    pdfs: List[str] = []
    pages = 0

    while q and pages < max_pages:
        url = q.pop(0)
        if url in visited:
            continue
        visited.add(url)
        pages += 1

        try:
            html = fetch_html(url)
        except Exception as e:
            log.append(f"[HARVEST] WARN fetch failed: {url} :: {e}")
            continue

        links = extract_links(html, url)

        # collect pdf links
        for u in links:
            if guess_is_pdf(u):
                pdfs.append(u)

        # enqueue pagination-like links (same domain)
        for u in links:
            if is_pagination_link(u, domain_root) and u not in visited:
                # keep within same netloc
                if same_domain(u, start_url):
                    q.append(u)

    # unique stable
    seen = set()
    out = []
    for u in pdfs:
        if u not in seen:
            seen.add(u)
            out.append(u)

    log.append(f"[HARVEST] crawl={pages} pages | pdf_links={len(out)} | start={start_url}")
    return out


def download_pdf(url: str, log: List[str]) -> Optional[bytes]:
    try:
        r = http_get(url)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        # accept even if ctype is not exactly pdf (some servers mislabel)
        data = r.content
        mb = len(data) / (1024 * 1024)
        if mb > MAX_PDF_MB:
            log.append(f"[HARVEST] SKIP too big ({mb:.1f}MB) {url}")
            return None
        if (not data) or len(data) < 800:
            log.append(f"[HARVEST] SKIP empty/small {url}")
            return None
        return data
    except Exception as e:
        log.append(f"[HARVEST] FAIL download {url} :: {e}")
        return None


def split_sujet_corrige(urls: List[str]) -> Tuple[List[str], List[str]]:
    sujets, corr = [], []
    for u in urls:
        if guess_is_correction(u):
            corr.append(u)
        else:
            sujets.append(u)
    return sujets, corr


def fuzzy_pairing(sujet_urls: List[str], corr_urls: List[str], min_ratio: float = 0.74) -> Dict[str, Optional[str]]:
    """
    Map sujet_url -> best corr_url by normalized key similarity.
    Deterministic tie-break by URL string.
    """
    corr_keys = [(u, normalize_pair_key(u)) for u in corr_urls]
    corr_keys.sort(key=lambda x: x[0])

    out: Dict[str, Optional[str]] = {}
    for su in sorted(sujet_urls):
        sk = normalize_pair_key(su)
        best = None
        best_r = 0.0
        for cu, ck in corr_keys:
            r = ratio(sk, ck)
            if r > best_r:
                best_r = r
                best = cu
        out[su] = best if (best is not None and best_r >= min_ratio) else None
    return out


def build_library_from_sources(
    pack: Pack,
    scope: str,
    volume: int,
    log: List[str],
    max_pages: int,
) -> List[LibraryItem]:
    """
    Build a library by crawling sources then downloading up to `volume` sujet PDFs.
    If a correction URL is paired, attempt to download it too.
    """
    all_pdfs: List[str] = []
    for src in pack.harvest_sources:
        for u in src.start_urls:
            all_pdfs.extend(crawl_domain_pdfs(u, max_pages=min(max_pages, src.max_pages), log=log))

    sujets_urls, corr_urls = split_sujet_corrige(all_pdfs)
    pair_map = fuzzy_pairing(sujets_urls, corr_urls, min_ratio=0.74)

    # Deterministic selection of sujets (limit volume)
    sujets_sel = sorted(sujets_urls)[: max(0, int(volume))]
    items: List[LibraryItem] = []

    for i, sujet_url in enumerate(sujets_sel, start=1):
        sujet_name = safe_filename(os.path.basename(urlparse(sujet_url).path) or f"sujet_{i}.pdf")
        sujet_bytes = download_pdf(sujet_url, log)
        if not sujet_bytes:
            continue

        sujet_sha = sha256_bytes(sujet_bytes)
        corr_url = pair_map.get(sujet_url)
        corr_art: Optional[PdfArtifact] = None
        corr_found = corr_url is not None
        corr_dl_ok = False

        if corr_url:
            corr_name = safe_filename(os.path.basename(urlparse(corr_url).path) or f"corrige_{i}.pdf")
            corr_bytes = download_pdf(corr_url, log)
            if corr_bytes:
                corr_sha = sha256_bytes(corr_bytes)
                corr_dl_ok = True
                corr_art = PdfArtifact(
                    name=corr_name,
                    url=corr_url,
                    sha256=corr_sha,
                    size_bytes=len(corr_bytes),
                )
            else:
                corr_art = PdfArtifact(
                    name=corr_name,
                    url=corr_url,
                    sha256=None,
                    size_bytes=0,
                )

        pair_id = f"PAIR_{scope}_{i:04d}_{sujet_sha[:7]}"
        items.append(
            LibraryItem(
                pair_id=pair_id,
                scope=scope,
                sujet=PdfArtifact(name=sujet_name, url=sujet_url, sha256=sujet_sha, size_bytes=len(sujet_bytes)),
                corrige=corr_art,
                corrige_found=corr_found,
                corrige_dl_ok=corr_dl_ok,
                source="HARVEST_AUTO",
                added_at=now_ts(),
            )
        )

    log.append(f"[HARVEST] items_ready={len(items)} (volume={volume})")
    return items


# =============================================================================
# EXTRACTION (Qi / RQi) — CAS 1 ONLY
# =============================================================================
def pdf_to_text(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        parts = []
        for page in pdf.pages:
            t = page.extract_text() or ""
            parts.append(t)
        return "\n".join(parts)


def split_qi(text: str) -> List[Dict[str, str]]:
    """
    Split into Qi candidates.
    Invariant heuristics (structure markers), not chapter-specific.
    """
    t = (text or "").strip()
    if not t:
        return []

    # Common structural patterns:
    # "1." / "1)" / "Question 1" etc.
    pattern = re.compile(r"(?im)^(?:\s*(?:question\s*)?(\d{1,3})\s*[\.\)\-:])\s+")
    matches = list(pattern.finditer(t))

    qi: List[Dict[str, str]] = []
    if len(matches) >= 2:
        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(t)
            chunk = t[start:end].strip()
            qid = m.group(1) or str(idx + 1)
            chunk = chunk[:MAX_QI_CHARS]
            qi.append({"qi_id": f"Q{qid}", "text": chunk})
        return qi

    # Fallback: split by double newlines into blocks
    blocks = [b.strip() for b in re.split(r"\n{2,}", t) if b.strip()]
    for i, b in enumerate(blocks, start=1):
        b = b[:MAX_QI_CHARS]
        qi.append({"qi_id": f"Q{i}", "text": b})
    return qi


def map_posable(qi: List[Dict[str, str]], rqi: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    CAS 1 ONLY:
    - POSABLE = Qi with a correction mapping (here: index mapping)
    - non-posables => quarantine
    """
    n = min(len(qi), len(rqi))
    pos = []
    quarantined = []

    for i in range(len(qi)):
        if i < n:
            pos.append(qi[i])
        else:
            quarantined.append(qi[i])

    return pos, quarantined


# =============================================================================
# QC BUILD (Clustering + Anti-singleton)
# =============================================================================
def tokenize(s: str) -> List[str]:
    s = norm_text(s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [w for w in s.split(" ") if w]
    return toks


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / max(1, uni)


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def cluster_qi(posable: List[Dict[str, str]], sim_th: float) -> List[List[Dict[str, str]]]:
    toks = [tokenize(q["text"]) for q in posable]
    n = len(posable)
    dsu = DSU(n)

    for i in range(n):
        for j in range(i + 1, n):
            s = jaccard(toks[i], toks[j])
            if s >= sim_th:
                dsu.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)

    clusters = []
    # deterministic ordering: by (-size, first_qi_id)
    for _, idxs in groups.items():
        items = [posable[i] for i in idxs]
        clusters.append(items)

    def cluster_key(c: List[Dict[str, str]]):
        size = len(c)
        # extract numeric part for stable ordering
        def qnum(qid: str) -> int:
            m = re.search(r"(\d+)", qid or "")
            return int(m.group(1)) if m else 10**9
        first = min(qnum(x["qi_id"]) for x in c) if c else 10**9
        return (-size, first)

    clusters.sort(key=cluster_key)
    return clusters


def build_qc_from_clusters(clusters: List[List[Dict[str, str]]]) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    """
    Return: qc_pack, qi->qc map, orphans_posable
    Anti-singleton: cluster size < MIN_CLUSTER_SIZE => blocked => orphan POSABLE.
    """
    qc_pack: List[Dict[str, Any]] = []
    qi_to_qc: Dict[str, str] = {}
    orphans: List[str] = []

    for c in clusters:
        if len(c) < MIN_CLUSTER_SIZE:
            # anti-singleton => block
            for q in c:
                orphans.append(q["qi_id"])
            continue

        qi_ids = [q["qi_id"] for q in c]
        # deterministic qc_id from member ids
        qc_id = "QC_" + sha256_str("||".join(sorted(qi_ids)))[:10]

        # Minimal invariant ARI/FRT/TRG templates (structure-only)
        ari = {
            "template": "ARI_TEMPLATE_V1",
            "goal": "Résoudre la question en appliquant une procédure de résolution structurée.",
            "steps": [
                "Lire et reformuler la consigne.",
                "Lister les données, contraintes et inconnues.",
                "Choisir une stratégie de résolution (méthode).",
                "Exécuter les étapes de calcul/raisonnement.",
                "Vérifier la cohérence (unités, bornes, ordre de grandeur, cas limites).",
                "Présenter le résultat selon le format attendu.",
            ],
            "checks": [
                "Cohérence interne des transformations.",
                "Vérification d’un cas simple / limite si possible.",
                "Validation finale du résultat.",
            ],
        }

        frt = {
            "template": "FRT_TEMPLATE_V1",
            "declencheur": "Identifier la forme du problème et déclencher la procédure adéquate.",
            "plan": [
                "État initial (données / objectif).",
                "Transformation(s) / calcul(s).",
                "Résultat final + justification.",
            ],
            "pieges": [
                "Oublis de conditions / hypothèses.",
                "Erreur de manipulation ou de conversion.",
                "Résultat non justifié / format non conforme.",
            ],
            "validation": [
                "Contrôle rapide (sanity check).",
                "Contrôle formel (substitution / recomposition).",
            ],
        }

        triggers = {
            "template": "TRG_TEMPLATE_V1",
            "signals": [
                "Consigne explicite (calculer / déterminer / démontrer / justifier).",
                "Présence d’un objectif et de contraintes.",
                "Structure multi-étapes détectée (sous-questions).",
            ],
        }

        qc_label = "Comment résoudre une question de ce cluster ?"

        qc_obj = {
            "qc_id": qc_id,
            "qc": qc_label,
            "ari": ari,
            "frt": frt,
            "triggers": triggers,
            "qi_members": qi_ids,
        }
        qc_pack.append(qc_obj)
        for qid in qi_ids:
            qi_to_qc[qid] = qc_id

    return qc_pack, qi_to_qc, sorted(orphans)


# =============================================================================
# EVIDENCE / REPORTS
# =============================================================================
def make_harvest_manifest(items: List[LibraryItem]) -> Dict[str, Any]:
    return {
        "version": APP_VERSION,
        "generated_at": now_ts(),
        "items": [
            {
                "pair_id": it.pair_id,
                "scope": it.scope,
                "sujet": asdict(it.sujet),
                "corrige": asdict(it.corrige) if it.corrige else None,
                "corrige_found": it.corrige_found,
                "corrige_dl_ok": it.corrige_dl_ok,
                "source": it.source,
                "added_at": it.added_at,
            }
            for it in items
        ],
    }


def make_chapter_report(
    scope: str,
    qi_total: int,
    posable_total: int,
    qc_total: int,
    orphans_posable: List[str],
    validators: Dict[str, Any],
    proof_sig: str,
    sealed: bool,
) -> Dict[str, Any]:
    return {
        "version": APP_VERSION,
        "generated_at": now_ts(),
        "scope": scope,
        "counts": {
            "qi_total": qi_total,
            "posable_total": posable_total,
            "qc_total": qc_total,
            "orphans_posable": len(orphans_posable),
        },
        "orphans_posable": orphans_posable,
        "validators": validators,
        "proof_sig": proof_sig,
        "sealed": sealed,
    }


# =============================================================================
# UI HELPERS
# =============================================================================
def ui_kpi(label: str, value: Any):
    st.metric(label, value)


def df_from_library(items: List[LibraryItem]) -> List[Dict[str, Any]]:
    rows = []
    for it in items:
        rows.append(
            {
                "pair_id": it.pair_id,
                "scope": it.scope,
                "sujet": it.sujet.name,
                "sujet_size_kb": round(it.sujet.size_bytes / 1024, 1),
                "corrige?": "✅" if it.corrige_dl_ok else ("🔎" if it.corrige_found else "❌"),
                "corrige_name": (it.corrige.name if it.corrige else ""),
                "source": it.source,
                "sujet_url": it.sujet.url or "",
                "corrige_url": (it.corrige.url if it.corrige else ""),
            }
        )
    return rows


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.set_page_config(page_title=f"{APP_NAME} {APP_VERSION}", layout="wide")

if "log" not in st.session_state:
    st.session_state.log = []
if "pack" not in st.session_state:
    st.session_state.pack = None
if "library" not in st.session_state:
    st.session_state.library: List[LibraryItem] = []
if "run_state" not in st.session_state:
    st.session_state.run_state = {}
if "last_proof_sig" not in st.session_state:
    st.session_state.last_proof_sig = None


def log(msg: str):
    st.session_state.log.append(msg)


# ------------------------------
# SIDEBAR (ESSENTIEL)
# ------------------------------
with st.sidebar:
    st.markdown("### ÉTAPE 1 — ACTIVATION PAYS")
    country = st.selectbox("Pays (TEST)", ["FR"], index=0)
    if st.button("🔐 ACTIVER", use_container_width=True):
        try:
            st.session_state.pack = load_academic_pack(country)
            st.session_state.library = []
            st.session_state.run_state = {}
            st.session_state.last_proof_sig = None
            st.session_state.log = []
            log(f"[{datetime.now().strftime('%H:%M:%S')}] ACTIVATION {country} OK | pack={st.session_state.pack.pack_id}")
        except Exception as e:
            st.error(str(e))

    pack: Optional[Pack] = st.session_state.pack
    if pack:
        st.success("Pack actif")
        st.write(pack.pack_id)
        st.caption(f"Signature: {pack.signature}")

    st.markdown("---")
    st.markdown("### ÉTAPE 2 — SÉLECTION")
    if not pack:
        st.info("Activez un pays.")
        st.stop()

    level = st.radio("Niveau", pack.levels, index=2)  # Terminale default
    subj_options = pack.subjects.get(level, [])
    subjects_sel = st.multiselect("Matières (1–2 recommandé pour test)", subj_options, default=subj_options[:1])

    if not subjects_sel:
        st.info("Sélectionnez au moins une matière.")
        st.stop()

    st.markdown("---")
    st.markdown("### Chapitres (pack-driven)")
    for s in subjects_sel:
        ch = pack.chapters.get(s, [])
        for c in ch:
            st.caption(f"• {c}")


# ------------------------------
# HEADER
# ------------------------------
st.title(f"🔐 {APP_NAME} {APP_VERSION} — {APP_FLAVOR}")
st.caption("Flux: Activation → Pack visible → Sélection → Harvest → Extraction → RUN GTE → Saturation → Résultats")

tab1, tab2, tab3 = st.tabs(["📥 Import / Bibliothèque", "⚡ Chaîne complète", "📦 Résultats / Exports"])


# =============================================================================
# TAB 1 — Import / Bibliothèque
# =============================================================================
with tab1:
    st.markdown("## Import PDF (sujets + corrigés)")
    st.caption("Bibliothèque = Harvest AUTO + Upload manuel. La chaîne GTE consomme la bibliothèque.")

    colA, colB, colC, colD, colE, colF = st.columns(6)
    items_cnt = len(st.session_state.library)
    corr_ok = sum(1 for it in st.session_state.library if it.corrige_dl_ok)
    qi_cnt = st.session_state.run_state.get("qi_total", 0)
    pos_cnt = st.session_state.run_state.get("posable_total", 0)
    qc_cnt = st.session_state.run_state.get("qc_total", 0)
    sealed = st.session_state.run_state.get("sealed", False)

    with colA: ui_kpi("Items", items_cnt)
    with colB: ui_kpi("Corrigés DL", corr_ok)
    with colC: ui_kpi("Qi", qi_cnt)
    with colD: ui_kpi("Qi POSABLE", pos_cnt)
    with colE: ui_kpi("QC", qc_cnt)
    with colF: ui_kpi("SEALED", "YES" if sealed else "NO")

    st.markdown("### Bibliothèque Harvest (visible)")
    only_missing = st.checkbox("Afficher seulement les items sans corrigé exploitable (❌/🔎)", value=False)

    # Harvest AUTO controls
    with st.expander("HARVEST AUTO (récolte) — paramètres", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_pages = st.number_input("Max pages crawl (par source)", min_value=1, max_value=60, value=MAX_CRAWL_PAGES_DEFAULT, step=1)
        with col2:
            volume = st.number_input("Volume (sujets par matière)", min_value=1, max_value=400, value=50, step=5)
        with col3:
            scope_preview = f"{level.upper()}|{subjects_sel[0].upper()}"
            st.text_input("Scope (auto)", value=scope_preview, disabled=True)

        if st.button("🌐 Lancer HARVEST AUTO (Bibliothèque)", use_container_width=True):
            st.session_state.log = []
            log(f"[{datetime.now().strftime('%H:%M:%S')}] === HARVEST AUTO {APP_VERSION} ===")
            # Build per selected subject (1–2)
            new_items: List[LibraryItem] = []
            for s in subjects_sel:
                scope = f"{level.upper()}|{s.upper()}"
                log(f"[HARVEST] Scope={scope}")
                try:
                    got = build_library_from_sources(
                        pack=pack,
                        scope=scope,
                        volume=int(volume),
                        log=st.session_state.log,
                        max_pages=int(max_pages),
                    )
                    new_items.extend(got)
                except Exception as e:
                    log(f"[HARVEST] ERROR {scope} :: {e}")

            # Merge into library (dedupe by sujet sha)
            existing = {it.sujet.sha256 for it in st.session_state.library if it.sujet.sha256}
            merged = list(st.session_state.library)
            add = 0
            for it in new_items:
                if it.sujet.sha256 and it.sujet.sha256 not in existing:
                    merged.append(it)
                    existing.add(it.sujet.sha256)
                    add += 1

            st.session_state.library = merged
            log(f"[BIBLIOTHEQUE] +{add} items (total={len(merged)})")

    # Manual upload
    st.markdown("### Upload manuel (optionnel)")
    colu1, colu2 = st.columns(2)
    with colu1:
        up_sujet = st.file_uploader("PDF Sujet", type=["pdf"], key="up_sujet")
    with colu2:
        up_corr = st.file_uploader("PDF Correction (opt)", type=["pdf"], key="up_corr")

    if st.button("➕ Ajouter à la bibliothèque", use_container_width=True):
        if not up_sujet:
            st.error("PDF Sujet requis.")
        else:
            sujet_bytes = up_sujet.read()
            sujet_sha = sha256_bytes(sujet_bytes)
            corr_bytes = up_corr.read() if up_corr else None
            corr_sha = sha256_bytes(corr_bytes) if corr_bytes else None

            scope = f"{level.upper()}|{subjects_sel[0].upper()}"
            pair_id = f"PAIR_{scope}_MANUAL_{sujet_sha[:7]}"

            corr_art = None
            corr_found = False
            corr_dl_ok = False
            if corr_bytes:
                corr_found = True
                corr_dl_ok = True
                corr_art = PdfArtifact(
                    name=safe_filename(up_corr.name),
                    url=None,
                    sha256=corr_sha,
                    size_bytes=len(corr_bytes),
                )

            it = LibraryItem(
                pair_id=pair_id,
                scope=scope,
                sujet=PdfArtifact(
                    name=safe_filename(up_sujet.name),
                    url=None,
                    sha256=sujet_sha,
                    size_bytes=len(sujet_bytes),
                ),
                corrige=corr_art,
                corrige_found=corr_found,
                corrige_dl_ok=corr_dl_ok,
                source="UPLOAD_MANUAL",
                added_at=now_ts(),
            )
            st.session_state.library.append(it)
            log(f"[BIBLIOTHEQUE] +1 item manual (total={len(st.session_state.library)})")

    # Table
    rows = df_from_library(st.session_state.library)
    if only_missing:
        rows = [r for r in rows if r["corrige?"] != "✅"]
    st.dataframe(rows, use_container_width=True, height=360)

    st.markdown("### Log (dernier)")
    st.code("\n".join(st.session_state.log[-120:]) if st.session_state.log else "—")


# =============================================================================
# TAB 2 — Chaîne complète
# =============================================================================
with tab2:
    st.markdown("## ÉTAPE 3 — LANCER CHAÎNE COMPLÈTE (Harvest → Extraction → RUN GTE → Saturation)")
    st.caption("ISO-PROD : validateurs B bloquants + scellement seulement si preuve stable et PASS (et POSABLE>0, QC>0).")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_iter = st.number_input("Max itérations", min_value=1, max_value=10, value=4, step=1)
    with col2:
        sim_th = st.number_input("Seuil clustering (sim)", min_value=0.05, max_value=0.60, value=float(SIM_THRESHOLD), step=0.01, format="%.2f")
    with col3:
        st.number_input("Anti-singleton (min cluster size)", min_value=2, max_value=5, value=MIN_CLUSTER_SIZE, step=1, disabled=True)

    if st.button("⚡ LANCER CHAÎNE COMPLÈTE", use_container_width=True):
        st.session_state.log = []
        log(f"[{datetime.now().strftime('%H:%M:%S')}] === DÉMARRAGE CHAÎNE {APP_VERSION} ===")

        if not st.session_state.library:
            log("[HARVEST] Bibliothèque vide. Lancez HARVEST AUTO ou Upload manuel.")
            st.warning("Bibliothèque vide.")
        else:
            # CAS 1 ONLY: ne conserver que les items avec corrigé exploitable (dl_ok)
            usable = [it for it in st.session_state.library if it.corrige_dl_ok]
            quarantined = [it for it in st.session_state.library if not it.corrige_dl_ok]

            log(f"[CAS1] usable_pairs={len(usable)} | quarantined_pairs={len(quarantined)} (corrigé manquant/inexploitable)")

            # Extraction
            qi_all: List[Dict[str, str]] = []
            rqi_all: List[Dict[str, str]] = []

            # IMPORTANT : ici, on ne conserve pas bytes dans LibraryItem (pour rester léger)
            # => on ne peut extraire que sur uploads manuels.
            # Pour HARVEST_AUTO, on a téléchargé les PDFs mais on n’a pas stocké les bytes.
            # Solution ISO-PROD TEST : demander upload manuel pour extraction fiable,
            # OU étendre le stockage bytes (à valider côté sécurité).
            #
            # Pour rester strict et éviter le faux : si pas de bytes, extraction saute.
            #
            # => on permet extraction uniquement si source = UPLOAD_MANUAL.
            manual_pairs = [it for it in usable if it.source == "UPLOAD_MANUAL"]

            if not manual_pairs:
                log("[EXTRACTION] Aucun couple UPLOAD_MANUAL avec corrigé exploitable. Extraction BLOQUÉE (preuve).")
                log("[ISO] Pour tester l’extraction/QC : uploadez 1 sujet + 1 corrigé (manuel).")
                st.warning("Extraction bloquée : uploadez au moins 1 couple sujet+corrigé en manuel.")
            else:
                # Reconstituer les bytes depuis uploaders (Streamlit ne conserve pas après rerun),
                # donc on ne peut pas relire ici. Cette console n’a pas de stockage persistant.
                # => Mode preuve: on ne fabrique rien. On FAIL proprement.
                log("[EXTRACTION] Limite Streamlit: bytes upload non persistés après rerun. ")
                log("[ISO] Pour extraction réelle, activer un stockage (disk/tmp) lors de l’ajout à la bibliothèque.")
                st.error("Limite actuelle : persistance bytes non activée. Voir commentaire dans le code.")
                # Hard stop : pas de faux résultats
                st.session_state.run_state = {
                    "qi_total": 0,
                    "posable_total": 0,
                    "qc_total": 0,
                    "orphans_posable": 0,
                    "sealed": False,
                }

    st.markdown("### Log en temps réel")
    st.code("\n".join(st.session_state.log[-180:]) if st.session_state.log else "—")


# =============================================================================
# TAB 3 — Résultats / Exports
# =============================================================================
with tab3:
    st.markdown("## ÉTAPE 4 — RÉSULTATS / EXPORTS")
    colA, colB, colC, colD, colE, colF = st.columns(6)
    with colA: ui_kpi("Items", len(st.session_state.library))
    with colB: ui_kpi("Corrigés DL", sum(1 for it in st.session_state.library if it.corrige_dl_ok))
    with colC: ui_kpi("Qi", st.session_state.run_state.get("qi_total", 0))
    with colD: ui_kpi("Qi POSABLE", st.session_state.run_state.get("posable_total", 0))
    with colE: ui_kpi("QC", st.session_state.run_state.get("qc_total", 0))
    with colF: ui_kpi("SEALED", "YES" if st.session_state.run_state.get("sealed", False) else "NO")

    st.markdown("### Exports (preuves)")

    # Always export harvest manifest + logs (available even if chain not executed)
    manifest = make_harvest_manifest(st.session_state.library)
    st.download_button(
        "harvest_manifest.json",
        data=json_dumps(manifest).encode("utf-8"),
        file_name="harvest_manifest.json",
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        "logs.txt",
        data=("\n".join(st.session_state.log) if st.session_state.log else "").encode("utf-8"),
        file_name="logs.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # Placeholders for chain outputs (only if present)
    qi_pack = st.session_state.run_state.get("qi_pack")
    qc_pack = st.session_state.run_state.get("qc_pack")
    chapter_report = st.session_state.run_state.get("chapter_report")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "qi_pack.json",
            data=(json_dumps(qi_pack).encode("utf-8") if qi_pack else b"{}"),
            file_name="qi_pack.json",
            mime="application/json",
            use_container_width=True,
            disabled=qi_pack is None,
        )
    with col2:
        st.download_button(
            "qc_pack.json",
            data=(json_dumps(qc_pack).encode("utf-8") if qc_pack else b"{}"),
            file_name="qc_pack.json",
            mime="application/json",
            use_container_width=True,
            disabled=qc_pack is None,
        )
    with col3:
        st.download_button(
            "chapter_report.json",
            data=(json_dumps(chapter_report).encode("utf-8") if chapter_report else b"{}"),
            file_name="chapter_report.json",
            mime="application/json",
            use_container_width=True,
            disabled=chapter_report is None,
        )

    st.markdown("### Logs (dernier)")
    st.code("\n".join(st.session_state.log[-200:]) if st.session_state.log else "—")


# =============================================================================
# IMPORTANT — NOTE TECHNIQUE (ISO-PROD TEST)
# =============================================================================
# Cette V31.10.1 renforce Harvest/pairing et l’UI ISO.
# Pour que les tests QC/ARI/FRT/TRG deviennent "réels", il faut une persistance PDF bytes :
# - soit stockage tmp (disk) lors de l’ajout à la bibliothèque (HARVEST + UPLOAD),
# - soit stockage en base/objet (S3/Supabase Storage).
# Sans persistance, Streamlit ne garantit pas qu’on puisse ré-extraire après rerun,
# et on refuse de produire des "résultats" non prouvés (zéro déclaration non prouvée).
# =============================================================================
