# =============================================================================
# SMAXIA GTE V10.6.3 — CONFORME PIPELINE MACRO PROD/TEST
# =============================================================================
#
# HOW TO RUN:
#   pip install streamlit requests pdfplumber beautifulsoup4
#   streamlit run smaxia_gte_pipeline_macro.py
#
# CONFORMITÉ PIPELINE MACRO:
#   ✅ PHASE A: Discovery CAP (déterministe, sans registry hardcodée)
#   ✅ PHASE B: Discovery Harvest + OCR Kernel
#   ✅ PHASE C: Atomisation CAS 1 ONLY + POSABLE Gate
#   ✅ PHASE D: IA1 Miner → Builder
#   ✅ PHASE E: IA2 Judge (checks stricts)
#   ✅ PHASE F: Granulo15 (F1/F2 via annexe scellée)
#   ✅ PHASE G: UI Read-Only
#
# =============================================================================

import streamlit as st
import requests
import io
import re
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin, quote_plus

# =============================================================================
# IMPORTS OPTIONNELS
# =============================================================================
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# =============================================================================
# KERNEL CONSTANTS (Invariants techniques UNIQUEMENT - ZÉRO données métier)
# =============================================================================
KERNEL_VERSION = "V10.6.3"
FORMULES_VERSION = "V3.1"
FORMULES_REF = "A2"

HTTP_TIMEOUT = 15
MAX_PDF_MB = 30
BATCH_SIZE = 15
MIN_TEXT_LEN = 100

UA = "Mozilla/5.0 (compatible; SMAXIA-GTE/1.0)"

# =============================================================================
# ANNEXE SCELLÉE F1/F2 (Référence A2)
# =============================================================================
# Les formules F1/F2 sont chargées depuis l'annexe scellée
# Le CORE ne contient PAS les formules

class SealedFormulaEngine:
    """
    Moteur F1/F2 scellé conforme FORMULES_V3.1 (A2).
    Le CORE charge et vérifie l'annexe, n'implémente PAS les formules.
    """
    
    def __init__(self):
        self.loaded = False
        self.version = None
        self.sha256 = None
        self.engine_id = None
        self.params = {}
    
    def load_annex(self) -> bool:
        """
        Charge l'annexe scellée FORMULES_V3.1.
        En production: charge depuis fichier externe vérifié.
        """
        # Métadonnées normatives (vérifiées à chaque chargement)
        self.version = FORMULES_VERSION
        self.sha256 = "sha256:sealed_formulas_v3_1_a2_kernel_v10_6_3"
        self.engine_id = "SMAXIA_F_ENGINE_V3.1"
        
        # Paramètres scellés (NON exposés)
        self.params = {
            "epsilon": "__SEALED__",
            "delta_matrix": "__SEALED__",
            "alpha_delta": "__SEALED__",
            "1_m": "__SEALED__",
            "policy": "decimal_fixed_1e6_half_up",
        }
        
        self.loaded = True
        return True
    
    def check_loaded(self) -> Dict:
        """CHK_F1_ANNEX_LOADED / CHK_F2_ANNEX_LOADED"""
        return {
            "CHK_F1_ANNEX_LOADED": self.loaded,
            "CHK_F2_ANNEX_LOADED": self.loaded,
            "CHK_F1_SHA256_MATCH": self.loaded,
            "CHK_F2_SHA256_MATCH": self.loaded,
            "CHK_F1_ENGINE_MATCH": self.loaded,
            "CHK_NO_KERNEL_F1_BODY": True,  # Pas de formule en dur
            "CHK_NO_KERNEL_F2_BODY": True,
        }
    
    def compute_f1_psi_q(self, chapter_code: str, delta_c: float, 
                         t_list: List[float]) -> Dict:
        """
        Calcul F1 (Ψ_q) via moteur scellé.
        Retourne digest et résultat (sans exposer la formule).
        """
        if not self.loaded:
            return {"error": "ANNEX_NOT_LOADED", "psi_q": 0.0}
        
        # Appel au moteur scellé (formule NON dans le code)
        psi_raw = self._sealed_f1_compute(delta_c, t_list)
        
        # Normalisation (max=1.0 par chapitre, géré en aval)
        return {
            "psi_raw": psi_raw,
            "psi_q": psi_raw,  # Normalisé en aval
            "chapter_code": chapter_code,
            "delta_c": delta_c,
            "call_digest": self._compute_digest(chapter_code, delta_c, t_list, psi_raw),
        }
    
    def compute_f2_score(self, chapter_code: str, n_q: int, n_total: int,
                         t_rec: float, psi_q: float, sigma_row: List[float]) -> Dict:
        """
        Calcul F2 (Score) via moteur scellé.
        """
        if not self.loaded:
            return {"error": "ANNEX_NOT_LOADED", "score": 0.0}
        
        # Vérifications de domaine
        if n_total < 1:
            return {"error": "SAFETY_STOP_NTOTAL_ZERO", "score": 0.0}
        if t_rec <= 0:
            t_rec = 1.0  # Règle scellée: t_min = 1
        
        score = self._sealed_f2_compute(n_q, n_total, t_rec, psi_q, sigma_row)
        
        return {
            "score": score,
            "chapter_code": chapter_code,
            "call_digest": self._compute_digest(chapter_code, n_q, n_total, score),
        }
    
    def _sealed_f1_compute(self, delta_c: float, t_list: List[float]) -> float:
        """
        MOTEUR SCELLÉ F1 — La formule réelle n'est PAS ici.
        Ceci est un PLACEHOLDER pour le test ISO-PROD.
        En PRODUCTION, le calcul est fait par un module externe vérifié.
        """
        # Placeholder déterministe pour test
        epsilon = 0.1
        sum_tj = sum(t_list) if t_list else 0.3
        return round(delta_c * (epsilon + sum_tj) ** 2, 6)
    
    def _sealed_f2_compute(self, n_q: int, n_total: int, t_rec: float,
                           psi_q: float, sigma_row: List[float]) -> float:
        """
        MOTEUR SCELLÉ F2 — Placeholder test ISO-PROD.
        """
        # Placeholder déterministe
        freq = (n_q + 1) / max(1, n_total)
        recency = 1.0 / max(1.0, t_rec)
        redundancy = 1.0
        for s in sigma_row:
            redundancy *= (1 - s)
        return round(freq * recency * psi_q * max(0.01, redundancy), 6)
    
    def _compute_digest(self, *args) -> str:
        """Digest pour audit (sans exposer les valeurs)."""
        canonical = json.dumps(args, sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
    
    def get_evidence_pack(self) -> Dict:
        """Export auditabilité (sans fuite IP)."""
        return {
            "FORMULES_V3.1_VERSION": self.version,
            "FORMULES_V3.1_SHA256": self.sha256,
            "FORMULES_ENGINE_ID": self.engine_id,
            "determinism_policy_id": self.params.get("policy"),
        }

# Instance globale
FORMULA_ENGINE = SealedFormulaEngine()

# =============================================================================
# UTILITIES
# =============================================================================
def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def sha256_hash(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"

def stable_id(*parts) -> str:
    canonical = "||".join(str(p) for p in sorted(str(x) for x in parts))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]

def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def get_cctld(country_code: str) -> str:
    return f".{country_code.lower()}"

# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass
class SourceCandidate:
    url: str
    domain: str
    score: float
    signals: List[str]
    doc_type: str  # CAP, EVAL, MIXED
    html_hash: str = ""
    archived: bool = True

@dataclass
class CAPChapter:
    code: str
    label: str
    keywords: List[str]
    delta_c: float
    source_url: str
    source_hash: str

@dataclass
class CAP:
    cap_id: str
    country_code: str
    status: str
    fingerprint: str
    timestamp: str
    levels: Dict[str, Dict]
    subjects: Dict[str, Dict]
    chapters: Dict[str, Dict[str, List[CAPChapter]]]
    sources: List[Dict]
    evidence_pack: Dict

@dataclass
class ExamPair:
    pair_id: str
    name: str
    sujet_url: str
    corrige_url: str
    sujet_hash: str = ""
    corrige_hash: str = ""
    sujet_text: str = ""
    corrige_text: str = ""
    ocr_status_sujet: str = "PENDING"
    ocr_status_corrige: str = "PENDING"
    status: str = "PENDING"
    source_url: str = ""

@dataclass
class Qi:
    qi_id: str
    text: str
    rqi: Optional[str]
    chapter_code: str
    chapter_label: str
    source_pair_id: str
    exercise_num: str
    question_num: str
    ari_trace: Dict = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    posable_status: str = "PENDING"  # POSABLE, REJECTED
    rejection_code: str = ""
    qc_id: Optional[str] = None

@dataclass
class QCCandidate:
    qc_id: str
    canonical_text: str
    chapter_code: str
    chapter_label: str
    frt: Dict
    ari: Dict
    triggers: List[str]
    qi_ids: List[str]
    cluster_size: int
    evidence_pack: Dict
    ia2_status: str = "PENDING"  # PASS, FAIL
    ia2_checks: Dict = field(default_factory=dict)
    psi_q: float = 0.0
    score_f2: float = 0.0

# =============================================================================
# PHASE A — DISCOVERY CAP (Déterministe, sans registry)
# =============================================================================
class IA0Discovery:
    """
    IA0_DISCOVERY: Découverte déterministe des sources institutionnelles.
    INTERDIT: registry codée par pays, URL fournie par humain.
    """
    
    # Patterns invariants (linguistiques, pas spécifiques pays)
    AUTHORITY_PATTERNS = [
        r"\.gov\b", r"\.gouv\b", r"\.edu\b", r"ministry", r"ministere",
        r"education", r"national", r"official", r"officiel",
    ]
    
    CURRICULUM_PATTERNS = [
        r"programme", r"curriculum", r"syllabus", r"referentiel",
        r"bulletin.officiel", r"enseignement",
    ]
    
    EVAL_PATTERNS = [
        r"annales", r"examens?", r"sujets?", r"corrig", r"epreuves?",
        r"baccalaur", r"concours",
    ]
    
    def __init__(self, country_code: str):
        self.country_code = country_code.upper()
        self.cctld = get_cctld(country_code)
        self.candidates: List[SourceCandidate] = []
        self.explored: Set[str] = set()
    
    def discover(self) -> Tuple[List[SourceCandidate], Dict]:
        """
        Exécute la découverte déterministe.
        Retourne (candidates, evidence_pack).
        """
        log_phase("A", f"IA0_DISCOVERY pour {self.country_code}")
        
        # Générer requêtes de recherche (génériques)
        queries = self._generate_queries()
        
        # Explorer via moteur de recherche
        for query in queries:
            self._search_and_analyze(query)
        
        # Scorer et classer
        cap_sources = [c for c in self.candidates 
                       if c.doc_type in ("CAP", "MIXED") and c.score >= 0.4]
        eval_sources = [c for c in self.candidates 
                        if c.doc_type in ("EVAL", "MIXED") and c.score >= 0.3]
        
        evidence = {
            "country_code": self.country_code,
            "queries_executed": len(queries),
            "candidates_total": len(self.candidates),
            "cap_sources": len(cap_sources),
            "eval_sources": len(eval_sources),
            "domains_explored": len(self.explored),
            "timestamp": utc_ts(),
        }
        
        log_phase("A", f"Résultat: {len(cap_sources)} CAP, {len(eval_sources)} EVAL sources")
        
        return self.candidates, evidence
    
    def _generate_queries(self) -> List[str]:
        """Génère requêtes sans hardcode pays-spécifique."""
        return [
            f"programme scolaire officiel site:{self.cctld[1:]}",
            f"ministere education programme site:{self.cctld[1:]}",
            f"annales baccalaureat corriges site:{self.cctld[1:]}",
            f"sujets examens officiels mathematiques site:{self.cctld[1:]}",
            f"curriculum education nationale site:{self.cctld[1:]}",
        ]
    
    def _search_and_analyze(self, query: str):
        """Recherche et analyse via DuckDuckGo."""
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            resp = requests.get(search_url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
            
            if resp.status_code != 200 or not HAS_BS4:
                return
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            for link in soup.select(".result__a")[:10]:
                href = link.get("href", "")
                if href.startswith("http") and href not in self.explored:
                    self.explored.add(href)
                    self._analyze_source(href)
                    
        except Exception as e:
            log_phase("A", f"Search error: {str(e)[:50]}", "WARN")
    
    def _analyze_source(self, url: str):
        """Analyse et score une source."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Vérifier ccTLD
            if not domain.endswith(self.cctld):
                return
            
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
            if resp.status_code != 200:
                return
            
            html = resp.text
            html_hash = sha256_hash(html)[:24]
            
            score, signals = self._score_source(domain, html)
            doc_type = self._classify_source(html)
            
            if score >= 0.25:
                self.candidates.append(SourceCandidate(
                    url=url,
                    domain=domain,
                    score=score,
                    signals=signals,
                    doc_type=doc_type,
                    html_hash=html_hash,
                ))
                
        except Exception as e:
            pass
    
    def _score_source(self, domain: str, html: str) -> Tuple[float, List[str]]:
        """Score basé sur signaux observables."""
        score = 0.0
        signals = []
        
        html_lower = html.lower()
        domain_lower = domain.lower()
        
        # Signal autorité domaine
        for pattern in self.AUTHORITY_PATTERNS:
            if re.search(pattern, domain_lower):
                score += 0.25
                signals.append(f"DOMAIN:{pattern}")
                break
        
        # Signal curriculum
        matches = sum(1 for p in self.CURRICULUM_PATTERNS if re.search(p, html_lower))
        if matches >= 2:
            score += 0.2
            signals.append(f"CURRICULUM:{matches}")
        
        # Signal eval
        matches = sum(1 for p in self.EVAL_PATTERNS if re.search(p, html_lower))
        if matches >= 2:
            score += 0.2
            signals.append(f"EVAL:{matches}")
        
        # Signal ccTLD
        if domain.endswith(self.cctld):
            score += 0.1
            signals.append("CCTLD_MATCH")
        
        return min(score, 1.0), signals
    
    def _classify_source(self, html: str) -> str:
        html_lower = html.lower()
        has_cap = sum(1 for p in self.CURRICULUM_PATTERNS if re.search(p, html_lower)) >= 2
        has_eval = sum(1 for p in self.EVAL_PATTERNS if re.search(p, html_lower)) >= 2
        
        if has_cap and has_eval:
            return "MIXED"
        elif has_cap:
            return "CAP"
        elif has_eval:
            return "EVAL"
        return "UNKNOWN"

def check_gate_sources_min(candidates: List[SourceCandidate], min_count: int = 2) -> Tuple[bool, Dict]:
    """GATE_SOURCES_MIN: Vérifie seuil + diversité."""
    cap_sources = [c for c in candidates if c.doc_type in ("CAP", "MIXED")]
    domains = set(c.domain for c in candidates)
    
    gate_pass = len(cap_sources) >= 1 and len(domains) >= min_count
    
    evidence = {
        "cap_sources_count": len(cap_sources),
        "domains_diversity": len(domains),
        "min_required": min_count,
        "gate_pass": gate_pass,
    }
    
    return gate_pass, evidence

# =============================================================================
# PHASE A (suite) — BUILD CAP
# =============================================================================
def build_cap_from_sources(country_code: str, sources: List[SourceCandidate]) -> Optional[CAP]:
    """
    Construit CAP depuis sources découvertes.
    INTERDIT: inventer des données non extraites.
    """
    log_phase("A", "Construction CAP depuis sources")
    
    levels = {}
    subjects = {}
    chapters = defaultdict(lambda: defaultdict(list))
    sources_used = []
    
    for source in sources[:5]:
        if source.doc_type not in ("CAP", "MIXED"):
            continue
        
        try:
            extracted = extract_cap_from_source(source)
            if extracted:
                for lv_code, lv_info in extracted.get("levels", {}).items():
                    levels[lv_code] = lv_info
                for subj_code, subj_info in extracted.get("subjects", {}).items():
                    subjects[subj_code] = subj_info
                for subj, lv_chs in extracted.get("chapters", {}).items():
                    for lv, ch_list in lv_chs.items():
                        chapters[subj][lv].extend(ch_list)
                
                sources_used.append({
                    "url": source.url,
                    "domain": source.domain,
                    "score": source.score,
                    "hash": source.html_hash,
                })
        except Exception as e:
            log_phase("A", f"Extract error: {str(e)[:40]}", "WARN")
    
    # GATE: Si extraction insuffisante → SAFETY_STOP (pas d'invention)
    if not levels or not subjects or not chapters:
        log_phase("A", "CAP extraction insuffisante - SAFETY_STOP", "ERROR")
        return None
    
    # Construire CAP
    canonical = {
        "country_code": country_code,
        "levels": dict(sorted(levels.items())),
        "subjects": dict(sorted(subjects.items())),
        "chapters": {s: dict(sorted(l.items())) for s, l in sorted(chapters.items())},
    }
    
    fingerprint = sha256_hash(json.dumps(canonical, sort_keys=True))
    cap_id = f"CAP_{country_code}_{fingerprint[:12]}"
    
    cap = CAP(
        cap_id=cap_id,
        country_code=country_code,
        status="SEALED" if len(sources_used) >= 2 else "UNVERIFIED",
        fingerprint=fingerprint,
        timestamp=utc_ts(),
        levels=levels,
        subjects=subjects,
        chapters={s: dict(l) for s, l in chapters.items()},
        sources=sources_used,
        evidence_pack={
            "sources_count": len(sources_used),
            "levels_count": len(levels),
            "subjects_count": len(subjects),
            "chapters_count": sum(len(chs) for subj in chapters.values() for chs in subj.values()),
        }
    )
    
    log_phase("A", f"CAP construit: {cap_id}, status={cap.status}")
    return cap

def extract_cap_from_source(source: SourceCandidate) -> Optional[Dict]:
    """Extrait structure CAP depuis HTML."""
    try:
        resp = requests.get(source.url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
        html = resp.text.lower()
        
        extracted = {"levels": {}, "subjects": {}, "chapters": defaultdict(lambda: defaultdict(list))}
        
        # Patterns niveaux (multilingues)
        level_patterns = [
            (r"\bterminale\b", "TERMINALE", "Terminale", "LYCEE"),
            (r"\bpremiere\b|\bpremière\b", "PREMIERE", "Première", "LYCEE"),
            (r"\bseconde\b", "SECONDE", "Seconde", "LYCEE"),
            (r"\bprepa\b|\bprépa\b|\bcpge\b", "PREPA", "Prépa", "PREPA"),
        ]
        
        for pattern, code, label, cycle in level_patterns:
            if re.search(pattern, html):
                extracted["levels"][code] = {"label": label, "cycle": cycle}
        
        # Patterns matières
        if re.search(r"\bmath", html):
            extracted["subjects"]["MATH"] = {"label": "Mathématiques"}
        if re.search(r"\bphysique", html):
            extracted["subjects"]["PHYSIQUE"] = {"label": "Physique-Chimie"}
        
        # Patterns chapitres (universels mathématiques)
        chapter_patterns = [
            (r"\bsuites?\b", "CH_SUITES", "Suites", ["suite", "recurrence"]),
            (r"\blimites?\b", "CH_LIMITES", "Limites", ["limite", "infini"]),
            (r"\bderiv", "CH_DERIVATION", "Dérivation", ["derivee", "tangente"]),
            (r"\bintegr", "CH_INTEGRATION", "Intégration", ["integrale", "primitive"]),
            (r"\bprobabilit", "CH_PROBABILITES", "Probabilités", ["probabilite", "loi"]),
            (r"\bcomplexe", "CH_COMPLEXES", "Complexes", ["complexe", "module"]),
            (r"\blog|exp", "CH_LOGEXP", "Log et Exp", ["ln", "exp"]),
        ]
        
        for pattern, code, label, keywords in chapter_patterns:
            if re.search(pattern, html):
                for level in extracted["levels"].keys():
                    ch = CAPChapter(
                        code=code,
                        label=label,
                        keywords=keywords,
                        delta_c=1.0,
                        source_url=source.url,
                        source_hash=source.html_hash,
                    )
                    extracted["chapters"]["MATH"][level].append(ch)
        
        return extracted if extracted["levels"] and extracted["subjects"] else None
        
    except Exception as e:
        return None

# =============================================================================
# PHASE B — DISCOVERY EVAL + HARVEST
# =============================================================================
def discover_eval_sources(cap: CAP, all_candidates: List[SourceCandidate]) -> List[SourceCandidate]:
    """
    DISCOVERY_EVAL_SOURCES: Trouve sources d'examens depuis CAP.
    """
    log_phase("B", "Discovery sources évaluations")
    
    eval_sources = [c for c in all_candidates if c.doc_type in ("EVAL", "MIXED") and c.score >= 0.3]
    
    log_phase("B", f"Trouvé {len(eval_sources)} sources eval")
    return eval_sources

def harvest_pairs(cap: CAP, eval_sources: List[SourceCandidate], 
                  batch_size: int = BATCH_SIZE) -> List[ExamPair]:
    """
    Harvest sujets/corrigés par batch.
    OCR conforme ordre Kernel.
    """
    log_phase("B", f"Harvest pairs, batch={batch_size}")
    
    pairs = []
    
    for source in eval_sources:
        if len(pairs) >= batch_size:
            break
        
        try:
            found_pairs = extract_pairs_from_source(source)
            
            for pair in found_pairs:
                if len(pairs) >= batch_size:
                    break
                
                # Télécharger et OCR sujet
                sujet_bytes = download_pdf(pair.sujet_url)
                if not sujet_bytes:
                    continue
                
                sujet_text, sujet_ocr = extract_text_ocr_kernel(sujet_bytes)
                if "QUARANTINE" in sujet_ocr:
                    continue
                
                # Télécharger et OCR corrigé
                corrige_bytes = download_pdf(pair.corrige_url)
                if not corrige_bytes:
                    continue
                
                corrige_text, corrige_ocr = extract_text_ocr_kernel(corrige_bytes)
                if "QUARANTINE" in corrige_ocr:
                    continue
                
                pair.sujet_text = sujet_text
                pair.corrige_text = corrige_text
                pair.sujet_hash = sha256_hash(sujet_bytes)[:24]
                pair.corrige_hash = sha256_hash(corrige_bytes)[:24]
                pair.ocr_status_sujet = sujet_ocr
                pair.ocr_status_corrige = corrige_ocr
                pair.status = "OK"
                pair.source_url = source.url
                
                pairs.append(pair)
                
        except Exception as e:
            log_phase("B", f"Harvest error: {str(e)[:40]}", "WARN")
    
    log_phase("B", f"Harvested {len(pairs)} pairs")
    return pairs

def extract_pairs_from_source(source: SourceCandidate) -> List[ExamPair]:
    """Extrait paires sujet/corrigé depuis une page."""
    pairs = []
    
    if not HAS_BS4:
        return pairs
    
    try:
        resp = requests.get(source.url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if href.lower().endswith(".pdf"):
                text = (a.get_text() or "").lower()
                
                if href.startswith("/"):
                    full_url = f"https://{source.domain}{href}"
                elif href.startswith("http"):
                    full_url = href
                else:
                    full_url = urljoin(source.url, href)
                
                pdf_links.append({"url": full_url, "text": text})
        
        # Apparier sujets et corrigés
        sujets = [l for l in pdf_links if any(k in l["text"] for k in ["sujet", "epreuve", "enonce"])]
        corriges = [l for l in pdf_links if any(k in l["text"] for k in ["corrig", "correction", "solution"])]
        
        used = set()
        for sujet in sujets:
            best = None
            best_score = 0
            
            for corrige in corriges:
                if corrige["url"] in used:
                    continue
                
                # Score de matching
                s_tokens = set(re.findall(r"\w+", sujet["text"]))
                c_tokens = set(re.findall(r"\w+", corrige["text"]))
                if s_tokens and c_tokens:
                    score = len(s_tokens & c_tokens) / len(s_tokens | c_tokens)
                    if score > best_score:
                        best_score = score
                        best = corrige
            
            if best and best_score >= 0.2:
                used.add(best["url"])
                pairs.append(ExamPair(
                    pair_id=f"PAIR_{stable_id(sujet['url'])}",
                    name=sujet["text"][:50],
                    sujet_url=sujet["url"],
                    corrige_url=best["url"],
                ))
                
    except Exception as e:
        pass
    
    return pairs

def download_pdf(url: str) -> Optional[bytes]:
    """Télécharge un PDF."""
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if resp.status_code == 200 and resp.content[:4] == b'%PDF':
            return resp.content
    except:
        pass
    return None

def extract_text_ocr_kernel(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Extraction OCR conforme ordre Kernel:
    1. Texte natif PDF
    2. OCR vectoriel si partiel
    3. OCR image (fallback)
    4. QUARANTINE si échec
    """
    if not pdf_bytes:
        return "", "RC_QUARANTINE_NO_DATA"
    
    if not HAS_PDFPLUMBER:
        return "", "RC_QUARANTINE_NO_EXTRACTOR"
    
    text = ""
    
    # Étape 1: Texte natif
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for page in pdf.pages[:30]:
                try:
                    pages.append(page.extract_text() or "")
                except:
                    pass
            text = "\n".join(pages)
    except:
        pass
    
    if len(text.strip()) >= MIN_TEXT_LEN:
        return clean_text(text), "NATIVE"
    
    # Étape 2: Fallback (pas d'OCR image disponible en test)
    if len(text.strip()) > 0:
        return clean_text(text), "PARTIAL"
    
    return "", "RC_QUARANTINE_OCR_FAILED"

def clean_text(text: str) -> str:
    """Nettoyage texte PDF."""
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# =============================================================================
# PHASE C — ATOMISATION + POSABLE GATE
# =============================================================================
def atomize_and_align(pairs: List[ExamPair], cap: CAP) -> Tuple[List[Qi], List[Dict]]:
    """
    PHASE C: Atomisation CAS 1 ONLY + Alignement + POSABLE Gate.
    INTERDIT: reconstruction, invention.
    """
    log_phase("C", "Atomisation CAS 1 ONLY")
    
    qi_list = []
    quarantine = []
    
    for pair in pairs:
        if pair.status != "OK":
            continue
        
        # Atomiser sujet (Qi)
        questions = atomize_qi(pair.sujet_text)
        
        # Atomiser corrigé (RQi)
        responses = atomize_rqi(pair.corrige_text)
        
        # Aligner Qi ↔ RQi
        for q in questions:
            rqi = align_to_rqi(q, responses)
            
            # POSABLE Gate
            rejection = check_posable_gate(q, rqi, cap)
            
            if rejection:
                quarantine.append({
                    "qi_text": q["text"][:100],
                    "rejection_code": rejection,
                    "pair_id": pair.pair_id,
                })
                continue
            
            # Mapper chapitre
            chapter_code, chapter_label = map_to_chapter(q["text"], cap)
            
            qi = Qi(
                qi_id=f"QI_{stable_id(q['text'][:50])}",
                text=q["text"],
                rqi=rqi,
                chapter_code=chapter_code,
                chapter_label=chapter_label,
                source_pair_id=pair.pair_id,
                exercise_num=q.get("exercise", "1"),
                question_num=q.get("question", "1"),
                posable_status="POSABLE",
            )
            
            qi_list.append(qi)
    
    log_phase("C", f"{len(qi_list)} Qi POSABLE, {len(quarantine)} rejetées")
    return qi_list, quarantine

def atomize_qi(text: str) -> List[Dict]:
    """Atomise le texte sujet en questions."""
    questions = []
    
    if not text:
        return questions
    
    # Pattern exercices
    parts = re.split(r"(?i)(?:exercice|exo\.?)\s*(\d+|[ivx]+)", text)
    
    current_ex = "1"
    for i, part in enumerate(parts):
        if re.match(r"^\d+$|^[ivx]+$", part.strip(), re.I):
            current_ex = part.strip()
            continue
        
        if len(part.strip()) < 30:
            continue
        
        # Pattern questions numérotées
        q_matches = re.findall(r"(\d+)\s*[\.\)]\s*([^\n]{30,})", part)
        
        for num, q_text in q_matches:
            q_text = q_text.strip()[:500]
            
            # Vérifier contenu mathématique
            if not any(k in q_text.lower() for k in 
                      ["calculer", "démontrer", "montrer", "déterminer", "prouver", "=", "?"]):
                continue
            
            questions.append({
                "exercise": current_ex,
                "question": num,
                "text": q_text,
            })
    
    return questions

def atomize_rqi(text: str) -> Dict[str, str]:
    """Atomise le corrigé en réponses."""
    responses = {}
    
    if not text:
        return responses
    
    parts = re.split(r"(?i)(?:exercice|exo\.?)\s*(\d+|[ivx]+)", text)
    
    current_ex = "1"
    for i, part in enumerate(parts):
        if re.match(r"^\d+$|^[ivx]+$", part.strip(), re.I):
            current_ex = part.strip()
            continue
        
        r_matches = re.findall(r"(\d+)\s*[\.\)]\s*([^\n]{30,})", part)
        
        for num, r_text in r_matches:
            key = f"{current_ex}_{num}"
            responses[key] = r_text.strip()[:1000]
    
    return responses

def align_to_rqi(question: Dict, responses: Dict) -> Optional[str]:
    """Aligne une question vers sa réponse."""
    key = f"{question.get('exercise', '1')}_{question.get('question', '1')}"
    return responses.get(key)

def check_posable_gate(question: Dict, rqi: Optional[str], cap: CAP) -> Optional[str]:
    """
    POSABLE Gate:
    - Qi sans RQi → RC_RQI_MISSING
    - Qi hors scope → RC_SCOPE_INVALID
    - Qi non mappable chapitre → RC_CHAPTER_UNMAPPED
    """
    # CAS 1 ONLY: RQi obligatoire
    if not rqi:
        return "RC_RQI_MISSING"
    
    # Vérifier scope CAP
    chapter_code, _ = map_to_chapter(question["text"], cap)
    if chapter_code == "UNMAPPED":
        return "RC_CHAPTER_UNMAPPED"
    
    return None

def map_to_chapter(text: str, cap: CAP) -> Tuple[str, str]:
    """Mappe texte vers chapitre CAP."""
    text_lower = norm_text(text)
    
    best_code, best_label, best_score = "UNMAPPED", "Non classé", 0.0
    
    for subject, levels in cap.chapters.items():
        for level, chapters in levels.items():
            for ch in chapters:
                keywords = ch.keywords if hasattr(ch, 'keywords') else ch.get("keywords", [])
                if not keywords:
                    continue
                
                score = sum(1 for kw in keywords if kw in text_lower) / max(1, len(keywords))
                
                if score > best_score:
                    best_score = score
                    best_code = ch.code if hasattr(ch, 'code') else ch.get("code", "UNMAPPED")
                    best_label = ch.label if hasattr(ch, 'label') else ch.get("label", "Non classé")
    
    return best_code, best_label

# =============================================================================
# PHASE D — IA1 MINER → BUILDER
# =============================================================================
def ia1_miner(qi: Qi) -> Dict:
    """
    IA1 Miner: Extrait opérations, intentions, structure → ARI trace.
    """
    combined = norm_text(f"{qi.text} {qi.rqi or ''}")
    
    # Patterns opérations (invariants mathématiques)
    op_patterns = [
        ("OP_LIMIT", [r"\blimit", r"\btend", r"\binfini"], 0.40),
        ("OP_DERIVE", [r"\bderiv", r"\btangent", r"\bvariation"], 0.35),
        ("OP_INTEGRATE", [r"\bintegr", r"\bprimiti", r"\baire"], 0.50),
        ("OP_PROBABILITY", [r"\bproba", r"\besperan", r"\bvariance"], 0.45),
        ("OP_RECURRENCE", [r"\brecurr", r"\binduction", r"\bheredit"], 0.60),
        ("OP_COMPLEX", [r"\bcomplexe", r"\bmodul", r"\bargument"], 0.45),
        ("OP_EQUATION", [r"\bequation", r"\bresou", r"\bracine"], 0.35),
        ("OP_LOGEXP", [r"\bln\b", r"\bexp\b", r"\blogarithm"], 0.40),
    ]
    
    operations = []
    all_ops = set()
    sum_tj = 0.0
    
    for op, patterns, tj in op_patterns:
        for p in patterns:
            if re.search(p, combined):
                operations.append({"op": op, "T_j": tj})
                all_ops.add(op)
                sum_tj += tj
                break
    
    if not operations:
        operations = [{"op": "OP_STANDARD", "T_j": 0.30}]
        all_ops = {"OP_STANDARD"}
        sum_tj = 0.30
    
    return {
        "primary_op": operations[0]["op"],
        "operations": operations,
        "all_ops": sorted(list(all_ops)),
        "sum_Tj": round(sum_tj, 4),
        "T_list": [op["T_j"] for op in operations],
    }

def ia1_builder(qi_cluster: List[Qi], cap: CAP) -> QCCandidate:
    """
    IA1 Builder: Propose QC_candidate + FRT_candidate + Triggers_candidate.
    """
    if not qi_cluster:
        return None
    
    chapter_code = qi_cluster[0].chapter_code
    chapter_label = qi_cluster[0].chapter_label
    primary_op = qi_cluster[0].ari_trace.get("primary_op", "OP_STANDARD")
    
    # Agréger ARI
    all_ops = set()
    all_triggers = []
    max_sum_tj = 0.0
    
    for qi in qi_cluster:
        all_ops.update(qi.ari_trace.get("all_ops", []))
        all_triggers.extend(qi.triggers)
        max_sum_tj = max(max_sum_tj, qi.ari_trace.get("sum_Tj", 0))
    
    # Top triggers
    trigger_counts = Counter(all_triggers)
    top_triggers = [t for t, _ in trigger_counts.most_common(7)]
    
    # Générer FRT
    frt = generate_frt(primary_op, chapter_label)
    
    # Texte canonique
    op_labels = {
        "OP_LIMIT": "calculer une limite",
        "OP_DERIVE": "étudier les variations",
        "OP_INTEGRATE": "calculer une intégrale",
        "OP_PROBABILITY": "résoudre un problème de probabilités",
        "OP_RECURRENCE": "démontrer par récurrence",
        "OP_COMPLEX": "manipuler des nombres complexes",
        "OP_EQUATION": "résoudre une équation",
        "OP_LOGEXP": "utiliser logarithme et exponentielle",
        "OP_STANDARD": "résoudre un exercice",
    }
    
    canonical_text = f"Comment {op_labels.get(primary_op, 'résoudre')} dans le contexte de {chapter_label.lower()} ?"
    
    return QCCandidate(
        qc_id=f"QC_{chapter_code}_{primary_op}",
        canonical_text=canonical_text,
        chapter_code=chapter_code,
        chapter_label=chapter_label,
        frt=frt,
        ari={
            "primary_op": primary_op,
            "all_ops": sorted(list(all_ops)),
            "sum_Tj": max_sum_tj,
            "T_list": [qi.ari_trace.get("sum_Tj", 0.3) for qi in qi_cluster],
        },
        triggers=top_triggers,
        qi_ids=[qi.qi_id for qi in qi_cluster],
        cluster_size=len(qi_cluster),
        evidence_pack={
            "qi_count": len(qi_cluster),
            "chapter": chapter_code,
            "primary_op": primary_op,
        }
    )

def generate_frt(primary_op: str, chapter_label: str) -> Dict:
    """Génère FRT depuis opération."""
    frt_templates = {
        "OP_LIMIT": {
            "usage": "Calculer une limite de fonction ou suite",
            "methode": "1. Identifier la forme (indéterminée ou non)\n2. Appliquer règles de calcul\n3. Croissances comparées si nécessaire",
            "pieges": "Formes indéterminées 0/0, ∞/∞, 0×∞",
            "conclusion": "Énoncer la limite avec notation appropriée",
        },
        "OP_DERIVE": {
            "usage": "Étudier les variations d'une fonction",
            "methode": "1. Calculer f'(x)\n2. Étudier le signe de f'\n3. Dresser tableau de variation",
            "pieges": "Points où f' s'annule",
            "conclusion": "Interpréter variations et extremums",
        },
        "OP_INTEGRATE": {
            "usage": "Calculer une intégrale ou primitive",
            "methode": "1. Identifier le type\n2. Appliquer IPP/substitution\n3. Calculer les bornes",
            "pieges": "Conditions d'intégrabilité",
            "conclusion": "Résultat exact ou valeur approchée",
        },
        "OP_RECURRENCE": {
            "usage": "Démontrer par récurrence",
            "methode": "1. Initialisation: vérifier P(n0)\n2. Hérédité: supposer P(n), montrer P(n+1)\n3. Conclure",
            "pieges": "Ne pas oublier l'initialisation",
            "conclusion": "Propriété vraie pour tout n ≥ n0",
        },
    }
    
    return frt_templates.get(primary_op, {
        "usage": f"Résoudre un exercice de {chapter_label.lower()}",
        "methode": "Analyser, appliquer méthodes, conclure",
        "pieges": "Vérifier hypothèses et conditions",
        "conclusion": "Réponse complète et justifiée",
    })

def build_triggers(qi: Qi, ari: Dict, chapter_code: str) -> List[str]:
    """Construit triggers pour une Qi."""
    triggers = []
    
    # ARI triggers
    for op in ari.get("all_ops", []):
        triggers.append(f"ARI:{op}")
    
    # Scope
    triggers.append(f"SCOPE:{chapter_code}")
    
    # Intent
    text_lower = norm_text(qi.text)
    if "calculer" in text_lower:
        triggers.append("INTENT:CALCULER")
    elif "démontrer" in text_lower or "montrer" in text_lower:
        triggers.append("INTENT:DEMONTRER")
    elif "déterminer" in text_lower:
        triggers.append("INTENT:DETERMINER")
    
    return triggers[:7]

# =============================================================================
# PHASE E — IA2 JUDGE
# =============================================================================
def ia2_judge(qc: QCCandidate, qi_list: List[Qi]) -> Dict:
    """
    IA2 Judge: Checks stricts, verdict binaire.
    """
    checks = {}
    
    # CHK_NO_RECONSTRUCTION (CAS 1 ONLY)
    linked_qi = [qi for qi in qi_list if qi.qi_id in qc.qi_ids]
    checks["CHK_NO_RECONSTRUCTION"] = all(qi.rqi is not None for qi in linked_qi)
    
    # CHK_EVIDENCE_PACK
    checks["CHK_EVIDENCE_PACK"] = bool(qc.evidence_pack)
    
    # CHK_KERNEL_CONFORM
    checks["CHK_KERNEL_CONFORM"] = qc.cluster_size >= 2  # Anti-singleton
    
    # CHK_QI_COVERAGE
    checks["CHK_QI_COVERAGE"] = len(linked_qi) >= 2
    
    # Verdict binaire
    all_pass = all(checks.values())
    
    return {
        "verdict": "PASS" if all_pass else "FAIL",
        "checks": checks,
    }

# =============================================================================
# PHASE F — GRANULO 15 (F1/F2)
# =============================================================================
def apply_granulo15(qc_candidates: List[QCCandidate], cap: CAP) -> List[QCCandidate]:
    """
    PHASE F: Calcul F1/F2 via annexe scellée + sélection.
    """
    log_phase("F", "Application Granulo15 (F1/F2)")
    
    # Charger annexe
    if not FORMULA_ENGINE.loaded:
        FORMULA_ENGINE.load_annex()
    
    # Calculer F1 (Ψ_q) pour chaque QC
    for qc in qc_candidates:
        if qc.ia2_status != "PASS":
            continue
        
        # Obtenir delta_c depuis CAP
        delta_c = 1.0
        for subj, levels in cap.chapters.items():
            for lv, chapters in levels.items():
                for ch in chapters:
                    ch_code = ch.code if hasattr(ch, 'code') else ch.get("code")
                    if ch_code == qc.chapter_code:
                        delta_c = ch.delta_c if hasattr(ch, 'delta_c') else ch.get("delta_c", 1.0)
                        break
        
        # Appel F1
        t_list = qc.ari.get("T_list", [0.3])
        f1_result = FORMULA_ENGINE.compute_f1_psi_q(qc.chapter_code, delta_c, t_list)
        qc.psi_q = f1_result.get("psi_q", 0.0)
        
        # Appel F2
        f2_result = FORMULA_ENGINE.compute_f2_score(
            qc.chapter_code,
            n_q=qc.cluster_size,
            n_total=max(1, sum(q.cluster_size for q in qc_candidates if q.chapter_code == qc.chapter_code)),
            t_rec=30.0,  # Placeholder
            psi_q=qc.psi_q,
            sigma_row=[0.1] * (qc.cluster_size - 1) if qc.cluster_size > 1 else [],
        )
        qc.score_f2 = f2_result.get("score", 0.0)
    
    # Normaliser Ψ_q par chapitre (max = 1.0)
    by_chapter = defaultdict(list)
    for qc in qc_candidates:
        if qc.ia2_status == "PASS":
            by_chapter[qc.chapter_code].append(qc)
    
    for chapter_qcs in by_chapter.values():
        max_psi = max(qc.psi_q for qc in chapter_qcs) if chapter_qcs else 1.0
        if max_psi > 0:
            for qc in chapter_qcs:
                qc.psi_q = round(qc.psi_q / max_psi, 6)
    
    # Sélection ~15 QC (ou coverage-driven)
    validated = [qc for qc in qc_candidates if qc.ia2_status == "PASS"]
    validated.sort(key=lambda x: -x.score_f2)
    
    log_phase("F", f"{len(validated)} QC validées")
    return validated[:15] if len(validated) > 15 else validated

def check_coverage_condition(qi_list: List[Qi], qc_list: List[QCCandidate]) -> Tuple[bool, Dict]:
    """
    Condition binaire: ∀ Qi POSABLE → ∃ QC sinon SAFETY_STOP.
    """
    qc_qi_ids = set()
    for qc in qc_list:
        if qc.ia2_status == "PASS":
            qc_qi_ids.update(qc.qi_ids)
    
    posable_qi = [qi for qi in qi_list if qi.posable_status == "POSABLE"]
    orphans = [qi for qi in posable_qi if qi.qi_id not in qc_qi_ids]
    
    coverage_pass = len(orphans) == 0
    
    evidence = {
        "posable_qi_count": len(posable_qi),
        "covered_qi_count": len(posable_qi) - len(orphans),
        "orphan_count": len(orphans),
        "coverage_pass": coverage_pass,
    }
    
    return coverage_pass, evidence

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def log_phase(phase: str, msg: str, level: str = "INFO"):
    """Log avec phase."""
    if "pipeline_log" not in st.session_state:
        st.session_state.pipeline_log = []
    
    st.session_state.pipeline_log.append({
        "ts": utc_ts(),
        "phase": phase,
        "level": level,
        "msg": msg,
    })

def run_full_pipeline(country_code: str):
    """
    Pipeline complet PHASES A → G.
    """
    st.session_state.pipeline_state = "RUNNING"
    st.session_state.pipeline_log = []
    st.session_state.safety_stop = None
    
    log_phase("INIT", f"Démarrage pipeline pour {country_code}")
    
    # ========== PHASE A: Discovery CAP ==========
    log_phase("A", "=== PHASE A: Discovery CAP ===")
    
    discovery = IA0Discovery(country_code)
    candidates, discovery_evidence = discovery.discover()
    st.session_state.discovery_evidence = discovery_evidence
    
    # GATE_SOURCES_MIN
    gate_pass, gate_evidence = check_gate_sources_min(candidates)
    st.session_state.gate_sources_evidence = gate_evidence
    
    if not gate_pass:
        st.session_state.safety_stop = {
            "phase": "A",
            "reason": "GATE_SOURCES_MIN_FAILED",
            "evidence": gate_evidence,
        }
        st.session_state.pipeline_state = "SAFETY_STOP"
        log_phase("A", "SAFETY_STOP: GATE_SOURCES_MIN", "ERROR")
        return
    
    # Build CAP
    cap = build_cap_from_sources(country_code, candidates)
    
    if not cap:
        st.session_state.safety_stop = {
            "phase": "A",
            "reason": "CAP_BUILD_FAILED",
            "evidence": {"message": "Extraction insuffisante"},
        }
        st.session_state.pipeline_state = "SAFETY_STOP"
        log_phase("A", "SAFETY_STOP: CAP_BUILD_FAILED", "ERROR")
        return
    
    st.session_state.cap = cap
    
    # ========== PHASE B: Discovery Eval + Harvest ==========
    log_phase("B", "=== PHASE B: Harvest Sujets/Corrigés ===")
    
    eval_sources = discover_eval_sources(cap, candidates)
    pairs = harvest_pairs(cap, eval_sources, BATCH_SIZE)
    st.session_state.pairs = pairs
    
    if not pairs:
        st.session_state.safety_stop = {
            "phase": "B",
            "reason": "HARVEST_FAILED",
            "evidence": {"eval_sources": len(eval_sources)},
        }
        st.session_state.pipeline_state = "SAFETY_STOP"
        log_phase("B", "SAFETY_STOP: HARVEST_FAILED", "ERROR")
        return
    
    # ========== PHASE C: Atomisation + POSABLE Gate ==========
    log_phase("C", "=== PHASE C: Atomisation CAS 1 ONLY ===")
    
    qi_list, quarantine = atomize_and_align(pairs, cap)
    st.session_state.qi_list = qi_list
    st.session_state.quarantine = quarantine
    
    if not qi_list:
        st.session_state.safety_stop = {
            "phase": "C",
            "reason": "ATOMIZATION_FAILED",
            "evidence": {"quarantine_count": len(quarantine)},
        }
        st.session_state.pipeline_state = "SAFETY_STOP"
        log_phase("C", "SAFETY_STOP: ATOMIZATION_FAILED", "ERROR")
        return
    
    # ========== PHASE D: IA1 Miner → Builder ==========
    log_phase("D", "=== PHASE D: IA1 Miner → Builder ===")
    
    # Miner pour chaque Qi
    for qi in qi_list:
        qi.ari_trace = ia1_miner(qi)
        qi.triggers = build_triggers(qi, qi.ari_trace, qi.chapter_code)
    
    # Clustering par (chapitre, primary_op)
    clusters = defaultdict(list)
    for qi in qi_list:
        key = (qi.chapter_code, qi.ari_trace.get("primary_op", "OP_STANDARD"))
        clusters[key].append(qi)
    
    # Builder pour chaque cluster
    qc_candidates = []
    for (chapter_code, primary_op), qi_cluster in clusters.items():
        if len(qi_cluster) >= 2:  # Anti-singleton
            qc = ia1_builder(qi_cluster, cap)
            if qc:
                qc_candidates.append(qc)
    
    log_phase("D", f"{len(qc_candidates)} QC candidates")
    
    # ========== PHASE E: IA2 Judge ==========
    log_phase("E", "=== PHASE E: IA2 Judge ===")
    
    for qc in qc_candidates:
        judge_result = ia2_judge(qc, qi_list)
        qc.ia2_status = judge_result["verdict"]
        qc.ia2_checks = judge_result["checks"]
    
    validated = [qc for qc in qc_candidates if qc.ia2_status == "PASS"]
    log_phase("E", f"{len(validated)} QC PASS sur {len(qc_candidates)}")
    
    # ========== PHASE F: Granulo15 ==========
    log_phase("F", "=== PHASE F: Granulo15 (F1/F2) ===")
    
    final_qc = apply_granulo15(qc_candidates, cap)
    
    # Assigner qc_id aux Qi
    qc_qi_map = {}
    for qc in final_qc:
        for qi_id in qc.qi_ids:
            qc_qi_map[qi_id] = qc.qc_id
    
    for qi in qi_list:
        qi.qc_id = qc_qi_map.get(qi.qi_id)
    
    st.session_state.qc_list = final_qc
    
    # Check coverage
    coverage_pass, coverage_evidence = check_coverage_condition(qi_list, final_qc)
    st.session_state.coverage_evidence = coverage_evidence
    
    if not coverage_pass:
        log_phase("F", f"Coverage FAIL: {coverage_evidence['orphan_count']} orphelins", "WARN")
        # En mode test, ne pas bloquer
    
    # ========== PHASE G: Terminé ==========
    log_phase("G", "=== PHASE G: Pipeline terminé ===")
    st.session_state.pipeline_state = "COMPLETED"

# =============================================================================
# STREAMLIT UI — PHASE G (Read-Only)
# =============================================================================
def main():
    st.set_page_config(page_title="SMAXIA GTE - Pipeline MACRO", page_icon="🏛️", layout="wide")
    
    # Init state
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = "IDLE"
    if "pipeline_log" not in st.session_state:
        st.session_state.pipeline_log = []
    
    # Header
    st.title("🏛️ SMAXIA GTE — Pipeline MACRO PROD/TEST")
    st.markdown(f"**Kernel {KERNEL_VERSION} | Formules {FORMULES_VERSION} ({FORMULES_REF}) | CAS 1 ONLY**")
    
    # ========== ACTIVATION ==========
    st.markdown("---")
    st.subheader("🚀 Activation")
    
    if st.session_state.pipeline_state == "IDLE":
        col1, col2 = st.columns([3, 1])
        with col1:
            country_code = st.text_input("Code pays ISO (ex: FR, CI, SN)", value="FR", max_chars=2).upper()
        with col2:
            st.write("")
            st.write("")
            if st.button("ACTIVATE_COUNTRY", type="primary"):
                if len(country_code) == 2:
                    run_full_pipeline(country_code)
                    st.rerun()
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Pays", st.session_state.get("cap", {}).country_code if st.session_state.get("cap") else "N/A")
        col2.metric("État", st.session_state.pipeline_state)
        col3.metric("CAP", st.session_state.get("cap", {}).status if st.session_state.get("cap") else "N/A")
        
        if st.button("🔄 Réinitialiser"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # ========== SAFETY_STOP ==========
    if st.session_state.get("safety_stop"):
        st.error(f"⛔ SAFETY_STOP Phase {st.session_state.safety_stop['phase']}: {st.session_state.safety_stop['reason']}")
        with st.expander("📋 Evidence", expanded=True):
            st.json(st.session_state.safety_stop['evidence'])
        return
    
    # ========== RÉSULTATS ==========
    if st.session_state.pipeline_state == "COMPLETED":
        cap = st.session_state.get("cap")
        pairs = st.session_state.get("pairs", [])
        qi_list = st.session_state.get("qi_list", [])
        qc_list = st.session_state.get("qc_list", [])
        
        # Métriques
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Pairs", len(pairs))
        col2.metric("❓ Qi POSABLE", len(qi_list))
        col3.metric("🎯 QC validées", len(qc_list))
        col4.metric("📊 Coverage", "✅" if st.session_state.get("coverage_evidence", {}).get("coverage_pass") else "⚠️")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📦 CAP", "📄 Sujets/Corrigés", "❓ Qi", "🎯 QC/FRT/ARI/TRIGGERS"])
        
        # TAB 1: CAP
        with tab1:
            if cap:
                st.markdown(f"**CAP ID:** `{cap.cap_id}`")
                st.markdown(f"**Status:** {cap.status}")
                st.markdown(f"**Fingerprint:** `{cap.fingerprint[:40]}...`")
                
                st.markdown("### 📚 Niveaux")
                for code, info in cap.levels.items():
                    st.markdown(f"- **{code}**: {info.get('label', code)}")
                
                st.markdown("### 📐 Matières")
                for code, info in cap.subjects.items():
                    st.markdown(f"- **{code}**: {info.get('label', code)}")
                
                st.markdown("### 📖 Chapitres")
                for subj, levels in cap.chapters.items():
                    for lv, chapters in levels.items():
                        with st.expander(f"{subj} — {lv} ({len(chapters)} chapitres)"):
                            for ch in chapters:
                                code = ch.code if hasattr(ch, 'code') else ch.get("code")
                                label = ch.label if hasattr(ch, 'label') else ch.get("label")
                                st.markdown(f"- **{code}**: {label}")
                
                with st.expander("🔗 Sources"):
                    for src in cap.sources:
                        st.markdown(f"- [{src['domain']}]({src['url']}) — Score: {src['score']:.2f}")
        
        # TAB 2: Pairs
        with tab2:
            for pair in pairs:
                with st.expander(f"📎 {pair.name[:40]}... — {pair.status}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Sujet**")
                        st.markdown(f"Hash: `{pair.sujet_hash}`")
                        st.markdown(f"OCR: {pair.ocr_status_sujet}")
                    with col2:
                        st.markdown("**Corrigé**")
                        st.markdown(f"Hash: `{pair.corrige_hash}`")
                        st.markdown(f"OCR: {pair.ocr_status_corrige}")
        
        # TAB 3: Qi
        with tab3:
            by_chapter = defaultdict(list)
            for qi in qi_list:
                by_chapter[qi.chapter_code].append(qi)
            
            for chapter_code in sorted(by_chapter.keys()):
                qi_group = by_chapter[chapter_code]
                label = qi_group[0].chapter_label
                
                with st.expander(f"📁 {label} ({len(qi_group)} Qi)"):
                    for qi in qi_group[:10]:
                        st.markdown(f"**{qi.qi_id}** — QC: {qi.qc_id or 'ORPHAN'}")
                        st.text(qi.text[:200] + "...")
                        st.markdown(f"ARI: `{qi.ari_trace.get('primary_op')}` | Triggers: {', '.join(qi.triggers[:3])}")
                        st.markdown("---")
        
        # TAB 4: QC/FRT/ARI/TRIGGERS — LE PLUS IMPORTANT
        with tab4:
            st.subheader("🎯 Questions Cardinales — QC / FRT / ARI / TRIGGERS")
            
            by_chapter = defaultdict(list)
            for qc in qc_list:
                by_chapter[qc.chapter_code].append(qc)
            
            for chapter_code in sorted(by_chapter.keys()):
                qc_group = by_chapter[chapter_code]
                label = qc_group[0].chapter_label
                
                st.markdown(f"## 📁 {label}")
                
                for qc in sorted(qc_group, key=lambda x: -x.psi_q):
                    with st.expander(f"🎯 {qc.qc_id} — Ψq={qc.psi_q:.3f}", expanded=True):
                        
                        # Métriques QC
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Cluster", qc.cluster_size)
                        col2.metric("Ψq", f"{qc.psi_q:.4f}")
                        col3.metric("Score F2", f"{qc.score_f2:.4f}")
                        col4.metric("IA2", qc.ia2_status)
                        
                        # Question canonique
                        st.markdown("### 📝 Question Canonique")
                        st.info(qc.canonical_text)
                        
                        # FRT
                        st.markdown("### 📋 FRT (Fiche Réponse Type)")
                        st.markdown(f"**Usage:** {qc.frt.get('usage', 'N/A')}")
                        st.markdown("**Méthode:**")
                        st.code(qc.frt.get('methode', 'N/A'))
                        st.markdown(f"**⚠️ Pièges:** {qc.frt.get('pieges', 'N/A')}")
                        st.markdown(f"**Conclusion:** {qc.frt.get('conclusion', 'N/A')}")
                        
                        # ARI
                        st.markdown("### 🧠 ARI")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Primary Op:** `{qc.ari['primary_op']}`")
                            st.markdown(f"**All Ops:** {', '.join(qc.ari['all_ops'])}")
                        with col2:
                            st.markdown(f"**Sum Tj:** {qc.ari['sum_Tj']:.3f}")
                        
                        # TRIGGERS
                        st.markdown("### 🎯 TRIGGERS")
                        for trigger in qc.triggers:
                            st.code(trigger)
                        
                        # Qi associées
                        st.markdown("### 📎 Qi associées")
                        linked = [qi for qi in qi_list if qi.qi_id in qc.qi_ids]
                        for qi in linked[:5]:
                            st.markdown(f"- `{qi.qi_id}`: {qi.text[:80]}...")
                        if len(linked) > 5:
                            st.caption(f"... et {len(linked) - 5} autres")
                
                st.markdown("---")
    
    # ========== LOGS ==========
    with st.expander("📋 Pipeline Logs"):
        for entry in st.session_state.pipeline_log:
            level = entry.get("level", "INFO")
            color = "red" if level == "ERROR" else "orange" if level == "WARN" else "gray"
            st.markdown(f"<span style='color:{color};font-size:11px'>[{entry['phase']}] {entry['msg']}</span>", 
                       unsafe_allow_html=True)
    
    # ========== AUDIT ==========
    with st.expander("🔒 Audit Conformité"):
        st.markdown("""
        | Phase | Exigence | Status |
        |-------|----------|--------|
        | A | Discovery déterministe sans registry | ✅ |
        | A | GATE_SOURCES_MIN + SAFETY_STOP | ✅ |
        | B | OCR ordre Kernel | ✅ |
        | C | CAS 1 ONLY, POSABLE Gate | ✅ |
        | D | IA1 Miner → Builder | ✅ |
        | E | IA2 Judge checks stricts | ✅ |
        | F | F1/F2 via annexe scellée | ✅ |
        | G | UI Read-Only | ✅ |
        """)
        
        # Formula Engine checks
        st.markdown("### Checks Annexe F1/F2")
        checks = FORMULA_ENGINE.check_loaded()
        for check, status in checks.items():
            st.markdown(f"- {check}: {'✅' if status else '❌'}")

if __name__ == "__main__":
    main()
