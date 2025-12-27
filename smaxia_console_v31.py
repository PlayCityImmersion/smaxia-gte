# smaxia_console_v31.py
# =============================================================================
# SMAXIA - Console V31 (Saturation Proof) — STANDALONE
# =============================================================================
# Fichier unique contenant:
# - Moteur Granulo V10.4 (Kernel scellé, GPT 5.2 corrigé)
# - Adapter V31 ↔ V10.4
# - Console UI V31
# =============================================================================

import streamlit as st
import pandas as pd
import hashlib
import json
import re
import io
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
from enum import Enum
from collections import defaultdict
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# =============================================================================
# 1) KERNEL CONSTANTS (INVARIANTS)
# =============================================================================

class KernelConstants:
    KERNEL_VERSION = "V10.4"
    KERNEL_DATE = "2025-12-27"
    KERNEL_STATUS = "SEALED"
    FINGERPRINT_ALGORITHM = "SHA256"

    # F1 / F2 constants used by engine (kept here because sealed in kernel)
    EPSILON = 0.1
    ALPHA_DEFAULT = 5.0
    T_REC_MIN = 0.01

    # Coverage / selection
    SIGMA_QUASI_DOUBLON = 0.95

    # Safety stop (singletons irreductibles)
    ORPHAN_TOLERANCE_THRESHOLD = 0.02
    ORPHAN_TOLERANCE_ABSOLUTE = 2
    SCORE_MIN_VIABLE = 1.0
    LOW_HISTORY_N_Q_MAX = 1
    LOW_HISTORY_T_REC_MIN = 5  # years


class ReasonCode(Enum):
    RC_CORRIGE_MISSING = "RC_CORRIGE_MISSING"
    RC_CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
    RC_CORRIGE_MISMATCH = "RC_CORRIGE_MISMATCH"

    RC_SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
    RC_SCOPE_CONFLICT = "RC_SCOPE_CONFLICT"
    RC_SCOPE_OUTSIDE_PACK = "RC_SCOPE_OUTSIDE_PACK"

    RC_NOT_A_QUESTION = "RC_NOT_A_QUESTION"
    RC_DEPENDENCY_MISSING_CONTEXT = "RC_DEPENDENCY_MISSING_CONTEXT"
    RC_NON_DETERMINISTIC_STATEMENT = "RC_NON_DETERMINISTIC_STATEMENT"

    RC_DUPLICATE_ATOM = "RC_DUPLICATE_ATOM"
    RC_EXTRACTION_CORRUPTED = "RC_EXTRACTION_CORRUPTED"
    RC_LANGUAGE_UNSUPPORTED_BY_PACK = "RC_LANGUAGE_UNSUPPORTED_BY_PACK"

    RC_CORRECTION_LOOP_EXCEEDED = "RC_CORRECTION_LOOP_EXCEEDED"
    RC_SINGLETON_IRREDUCTIBLE = "RC_SINGLETON_IRREDUCTIBLE"

    ATT_PRECOND_FAIL = "ATT_PRECOND_FAIL"
    ATT_TRIGGER_MISS = "ATT_TRIGGER_MISS"
    ATT_SIGNATURE_MISMATCH = "ATT_SIGNATURE_MISMATCH"
    ATT_NEEDS_EXTRA_STEP = "ATT_NEEDS_EXTRA_STEP"
    ATT_OUTPUT_TYPE_MISMATCH = "ATT_OUTPUT_TYPE_MISMATCH"


# =============================================================================
# 2) KERNEL TABLE — COGNITIVE VERBS (INVARIANT)
# =============================================================================

COGNITIVE_VERBS_TABLE: Dict[str, float] = {
    # L1
    "identifier": 0.15, "repérer": 0.15, "nommer": 0.15, "définir": 0.15,
    "reconnaître": 0.12, "lister": 0.12,

    # L2
    "analyser": 0.20, "observer": 0.20, "synthétiser": 0.22, "interpréter": 0.20,
    "expliquer": 0.18, "décrire": 0.18,

    # L3
    "contextualiser": 0.25, "simplifier": 0.25, "factoriser": 0.28,
    "appliquer": 0.28, "utiliser": 0.25,

    # L4
    "calculer": 0.32, "exprimer": 0.30, "formuler": 0.30,
    "comparer": 0.32, "distinguer": 0.30,

    # L5
    "résoudre": 0.40, "déterminer": 0.40, "dériver": 0.38, "intégrer": 0.38,
    "construire": 0.35, "établir": 0.38,

    # L6
    "argumenter": 0.45, "justifier": 0.45, "vérifier": 0.42,

    # L7
    "démontrer": 0.52, "prouver": 0.52, "montrer": 0.50,

    # L8
    "récurrence": 0.58,

    # conclusions (kept as verbs but low weight)
    "conclure": 0.20, "déduire": 0.35,
}

PROOF_VERBS = {"démontrer", "prouver", "montrer"}
VALUE_VERBS = {"calculer", "déterminer"}
TABLE_VERBS = {"étudier", "dresser"}


def _norm_words(text: str) -> List[str]:
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]+", (text or "").lower())


def get_verb_weight(verb: str) -> Tuple[float, bool]:
    v = (verb or "").lower().strip()
    if v in COGNITIVE_VERBS_TABLE:
        return (COGNITIVE_VERBS_TABLE[v], True)
    for canon, w in COGNITIVE_VERBS_TABLE.items():
        if canon in v or v in canon:
            return (w, True)
    return (0.0, False)


# =============================================================================
# 3) PACK CONFIG (LOCAL ONLY)
# =============================================================================

@dataclass
class PackConfig:
    pack_id: str
    pack_version: str
    country: str
    level: str
    subject: str

    chapters: List[str]
    chapter_keywords: Dict[str, Set[str]]
    stopwords: Set[str]
    delta_c_by_chapter: Dict[str, float]
    atomization_patterns: Dict[str, str] = field(default_factory=dict)

    # Optional: precondition matchers (local policies)
    precondition_matchers: Dict[str, List[str]] = field(default_factory=dict)


def create_pack_france_terminale_maths() -> PackConfig:
    return PackConfig(
        pack_id="FR-TERM-MATHS-2025",
        pack_version="1.0.0",
        country="France",
        level="Terminale",
        subject="MATHS",
        chapters=[
            "SUITES NUMÉRIQUES",
            "FONCTIONS",
            "INTÉGRALES",
            "PROBABILITÉS",
            "GÉOMÉTRIE DANS L'ESPACE",
            "NOMBRES COMPLEXES",
            "MATRICES",
            "ARITHMÉTIQUE",
        ],
        chapter_keywords={
            "SUITES NUMÉRIQUES": {
                "suite", "suites", "arithmétique", "géométrique", "raison",
                "u_n", "un", "vn", "récurrence", "convergence", "monotone",
                "bornée", "terme", "somme", "adjacentes",
            },
            "FONCTIONS": {
                "fonction", "dérivée", "primitive", "variation", "croissante",
                "décroissante", "extremum", "tangente", "asymptote", "limite",
                "continuité", "exponentielle", "logarithme",
            },
            "INTÉGRALES": {"intégrale", "primitive", "aire", "intégration"},
            "PROBABILITÉS": {
                "probabilité", "événement", "loi", "espérance", "variance",
                "écart-type", "binomiale", "normale", "conditionnelle",
                "indépendance", "arbre", "aléatoire",
            },
            "GÉOMÉTRIE DANS L'ESPACE": {
                "plan", "droite", "vecteur", "orthogonal", "parallèle",
                "intersection", "équation", "paramétrique", "distance",
                "repère", "normale",
            },
            "NOMBRES COMPLEXES": {
                "complexe", "affixe", "module", "argument", "conjugué",
                "trigonométrique", "exponentielle", "racine", "équation",
            },
            "MATRICES": {"matrice", "inverse", "produit", "déterminant", "système"},
            "ARITHMÉTIQUE": {"divisibilité", "pgcd", "ppcm", "premier", "congruence", "modulo"},
        },
        stopwords={
            "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "a",
            "en", "pour", "que", "qui", "est", "sont", "dans", "sur", "avec",
            "ce", "cette", "ces", "son", "sa", "ses", "au", "aux", "par",
            "on", "tout", "tous", "toute", "toutes", "être", "avoir",
        },
        delta_c_by_chapter={
            "SUITES NUMÉRIQUES": 1.0,
            "FONCTIONS": 1.0,
            "INTÉGRALES": 1.1,
            "PROBABILITÉS": 0.9,
            "GÉOMÉTRIE DANS L'ESPACE": 1.0,
            "NOMBRES COMPLEXES": 1.0,
            "MATRICES": 1.2,
            "ARITHMÉTIQUE": 1.1,
        },
        atomization_patterns={
            "exercice": r"(?:Exercice|EXERCICE)\s*(\d+|[IVX]+)",
            "partie": r"(?:Partie|PARTIE)\s*([A-Z]|\d+)",
            "question": r"^(\d+)[.)]\s*",
            "sous_question": r"^([a-z])[.)]\s*",
        },
        precondition_matchers={},
    )


# =============================================================================
# 4) DATA STRUCTURES (KERNEL SCHEMAS)
# =============================================================================

@dataclass(frozen=True)
class Locator:
    page: int
    exercice: Optional[str]
    partie: Optional[str]
    question: Optional[str]
    sous_question: Optional[str]
    line_start: int
    line_end: int

    def to_ref(self) -> str:
        parts: List[str] = []
        if self.exercice:
            parts.append(f"Ex{self.exercice}")
        if self.partie:
            parts.append(f"P{self.partie}")
        if self.question:
            parts.append(f"Q{self.question}")
        if self.sous_question:
            parts.append(f"{self.sous_question}")
        return "-".join(parts) if parts else "UNKNOWN"


@dataclass
class EvidencePack:
    source_id: str
    source_fingerprint: str
    extracted_at: str
    locator: Locator
    raw_segment: str
    derivation_log: List[str]


@dataclass
class Atom:
    atom_id: str
    subject_id: str
    correction_id: Optional[str]

    qi_id: str
    qi_raw: str
    qi_clean: str
    qi_evidence: EvidencePack

    rqi_id: Optional[str]
    rqi_raw: Optional[str]
    rqi_clean: Optional[str]
    rqi_evidence: Optional[EvidencePack]

    year_ref: int
    pairing_confidence: float
    pairing_method: str

    verbe_action: Optional[str]
    chapter_detected: Optional[str]
    scope_confidence: float


@dataclass
class PosableDecision:
    qi_id: str
    posable: bool
    has_exploitable_correction: bool
    scope_resolved: bool
    is_evaluable: bool
    chapter_ref: Optional[str]
    scope_trace: str
    reason_codes: List[ReasonCode]
    evidence_refs: List[str]


@dataclass(frozen=True)
class Signature:
    action_spine: Tuple[str, ...]
    preconditions_set: Tuple[str, ...]
    output_type: str
    checkpoints_core: Tuple[str, ...]

    def to_hash(self) -> str:
        payload = {
            "A": list(self.action_spine),
            "P": list(self.preconditions_set),
            "O": self.output_type,
            "X": list(self.checkpoints_core),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def similarity(self, other: "Signature") -> float:
        s1 = set(self.action_spine)
        s2 = set(other.action_spine)
        if not s1 or not s2:
            return 0.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0.0


@dataclass
class FRT:
    triggers: List[str]
    preconditions: List[str]
    exemples_enonce: List[str]

    ari_steps: List[Dict[str, Any]]
    erreurs_frequentes: List[str]
    confusions_conceptuelles: List[str]
    oublis_fatals: List[str]

    phrase_conclusion: str
    format_reponse: str
    elements_obligatoires: List[str]

    source: str
    evidence_refs: List[str]


@dataclass
class QCCandidate:
    qc_id: str
    chapter_ref: str
    title: str

    psi_raw: float
    psi_normalized: float
    sum_tj: float

    score: float
    n_q: int
    n_total: int
    t_rec: float
    freq_ratio: float
    recency_boost: float

    signature: Signature
    frt: FRT
    triggers: List[str]
    ari: List[str]

    source_qi_ids: List[str]
    source_rqi_ids: List[str]

    covered_qi_ids: Set[str] = field(default_factory=set)


@dataclass
class AttachResult:
    qi_id: str
    qc_id: str
    attached: bool
    reason_code: Optional[ReasonCode]
    evidence_refs: List[str]

    preconditions_ok: bool
    triggers_match: bool
    ari_compatible: bool
    output_type_ok: bool


@dataclass
class CoverageMap:
    chapter_ref: str
    qi_to_qc: Dict[str, str]
    orphans: Set[str]
    attach_results: List[AttachResult]
    n_total_target: int
    n_covered: int
    coverage_ratio: float


# =============================================================================
# 5) SCOPE CLASSIFIER (PACK POLICY)
# =============================================================================

class ScopeClassifier:
    def __init__(self, pack: PackConfig):
        self.pack = pack
        self.keyword_to_chapter: Dict[str, List[str]] = defaultdict(list)
        for ch, kws in pack.chapter_keywords.items():
            for kw in kws:
                self.keyword_to_chapter[kw.lower()].append(ch)

    def classify(self, qi_text: str, rqi_text: Optional[str]) -> Tuple[Optional[str], float, str]:
        combined = (qi_text or "") + " " + (rqi_text or "")
        tokens = set(_norm_words(combined)) - self.pack.stopwords

        scores: Dict[str, int] = {ch: 0 for ch in self.pack.chapters}
        matched: Dict[str, List[str]] = {ch: [] for ch in self.pack.chapters}

        for tok in tokens:
            if tok in self.keyword_to_chapter:
                for ch in self.keyword_to_chapter[tok]:
                    scores[ch] += 1
                    matched[ch].append(tok)

        best_ch, best_sc = max(scores.items(), key=lambda x: x[1])
        if best_sc == 0:
            return (None, 0.0, "NO_KEYWORD_MATCH")

        total = sum(scores.values()) or 1
        conf = best_sc / total
        trace = f"MATCHED:{best_ch}(kw={sorted(set(matched[best_ch]))[:5]})"
        return (best_ch, conf, trace)


# =============================================================================
# 6) ATOMIZER (STRUCTURED)
# =============================================================================

class BACAtomizer:
    def __init__(self, pack: PackConfig):
        self.pack = pack
        self.exercice_re = re.compile(pack.atomization_patterns.get("exercice", r"(?:Exercice|EXERCICE)\s*(\d+|[IVX]+)"), re.IGNORECASE)
        self.partie_re = re.compile(pack.atomization_patterns.get("partie", r"(?:Partie|PARTIE)\s*([A-Z]|\d+)"), re.IGNORECASE)
        self.question_re = re.compile(pack.atomization_patterns.get("question", r"^(\d+)[.)]\s*"), re.MULTILINE)
        self.sous_question_re = re.compile(pack.atomization_patterns.get("sous_question", r"^([a-z])[.)]\s*"), re.MULTILINE)
        self.verbes_action = list(COGNITIVE_VERBS_TABLE.keys())  # ordre stable (déterminisme)

    def atomize(self, text: str) -> List[Tuple[str, str, Locator, Optional[str]]]:
        lines = (text or "").split("\n")
        results: List[Tuple[str, str, Locator, Optional[str]]] = []

        current_ex = None
        current_part = None
        current_q = None
        current_sq = None

        buf: List[str] = []
        buf_start = 0

        def finalize(end_line: int) -> None:
            nonlocal buf, current_ex, current_part, current_q, current_sq, buf_start
            if not buf or not current_q:
                return
            qi_raw = " ".join(buf).strip()
            qi_clean = re.sub(r"\s+", " ", qi_raw).strip()
            if len(qi_clean) < 15:
                buf = []
                return

            verbe = None
            low = qi_clean.lower()
            for v in self.verbes_action:
                if re.search(rf"\b{re.escape(v)}\b", low):
                    verbe = v
                    break

            loc = Locator(
                page=1,
                exercice=current_ex,
                partie=current_part,
                question=current_q,
                sous_question=current_sq,
                line_start=buf_start,
                line_end=end_line,
            )
            results.append((qi_raw, qi_clean, loc, verbe))
            buf = []

        for i, line in enumerate(lines):
            s = (line or "").strip()
            if not s:
                continue

            ex_m = self.exercice_re.search(s)
            if ex_m:
                finalize(i - 1)
                current_ex = ex_m.group(1)
                current_part = None
                current_q = None
                current_sq = None
                buf = []
                continue

            part_m = self.partie_re.search(s)
            if part_m:
                finalize(i - 1)
                current_part = part_m.group(1)
                current_q = None
                current_sq = None
                buf = []
                continue

            q_m = self.question_re.match(s)
            if q_m:
                finalize(i - 1)
                current_q = q_m.group(1)
                current_sq = None
                rest = s[q_m.end():].strip()
                buf = [rest] if rest else []
                buf_start = i
                continue

            sq_m = self.sous_question_re.match(s)
            if sq_m and current_q:
                finalize(i - 1)
                current_sq = sq_m.group(1)
                rest = s[sq_m.end():].strip()
                buf = [rest] if rest else []
                buf_start = i
                continue

            if current_q:
                buf.append(s)

        finalize(len(lines) - 1)
        return results


# =============================================================================
# 7) PAIRING ENGINE (Qi <-> RQi)
# =============================================================================

class PairingEngine:
    @staticmethod
    def _key(loc: Locator) -> str:
        return f"{loc.exercice}|{loc.partie}|{loc.question}|{loc.sous_question}"

    def pair(
        self,
        qi_items: List[Tuple[str, str, Locator, Optional[str]]],
        rqi_items: List[Tuple[str, str, Locator, Optional[str]]],
        subject_id: str,
        correction_id: Optional[str],
        fingerprint_subject: str,
        fingerprint_correction: Optional[str],
        extracted_at: str,
        year_ref: int,
    ) -> List[Atom]:
        rqi_by_key: Dict[str, Tuple[str, str, Locator, Optional[str]]] = {}
        for item in rqi_items:
            rqi_by_key[self._key(item[2])] = item

        atoms: List[Atom] = []
        for idx, (qi_raw, qi_clean, qi_loc, qi_verbe) in enumerate(qi_items, start=1):
            key = self._key(qi_loc)
            rqi_match = rqi_by_key.get(key)

            pairing_conf = 0.0
            pairing_method = "NONE"

            if rqi_match:
                pairing_conf = 0.95
                pairing_method = "EXACT_LOCATOR"

            atom = Atom(
                atom_id=f"ATOM-{idx:05d}",
                subject_id=subject_id,
                correction_id=correction_id if rqi_match else None,

                qi_id=f"QI-{qi_loc.to_ref()}-{idx:05d}",
                qi_raw=qi_raw,
                qi_clean=qi_clean,
                qi_evidence=EvidencePack(
                    source_id=subject_id,
                    source_fingerprint=fingerprint_subject,
                    extracted_at=extracted_at,
                    locator=qi_loc,
                    raw_segment=qi_raw[:300],
                    derivation_log=["EXTRACTED", "SANITIZED", "NORMALIZED"],
                ),

                rqi_id=(f"RQI-{qi_loc.to_ref()}-{idx:05d}" if rqi_match else None),
                rqi_raw=(rqi_match[0] if rqi_match else None),
                rqi_clean=(rqi_match[1] if rqi_match else None),
                rqi_evidence=(
                    EvidencePack(
                        source_id=correction_id or "NO_CORRECTION",
                        source_fingerprint=fingerprint_correction or "NO_FINGERPRINT",
                        extracted_at=extracted_at,
                        locator=rqi_match[2],
                        raw_segment=(rqi_match[0] or "")[:300],
                        derivation_log=["EXTRACTED", "SANITIZED", "NORMALIZED", "PAIRED:EXACT_LOCATOR"],
                    )
                    if rqi_match else None
                ),

                year_ref=year_ref,
                pairing_confidence=pairing_conf,
                pairing_method=pairing_method,
                verbe_action=qi_verbe,
                chapter_detected=None,
                scope_confidence=0.0,
            )
            atoms.append(atom)

        return atoms


# =============================================================================
# 8) POSABLE GATE (SEALED DEFINITION)
# =============================================================================

class PosableGate:
    def __init__(self, classifier: ScopeClassifier):
        self.classifier = classifier

    def evaluate(self, atom: Atom) -> PosableDecision:
        reasons: List[ReasonCode] = []
        evidence_refs: List[str] = [atom.qi_id] + ([atom.rqi_id] if atom.rqi_id else [])

        # POSABLE_CORRIGE
        has_corr = False
        if atom.rqi_clean and len(atom.rqi_clean) >= 30 and atom.pairing_confidence >= 0.80:
            has_corr = True
        else:
            if atom.rqi_clean is None:
                reasons.append(ReasonCode.RC_CORRIGE_MISSING)
            elif len(atom.rqi_clean or "") < 30:
                reasons.append(ReasonCode.RC_CORRIGE_UNREADABLE)
            else:
                reasons.append(ReasonCode.RC_CORRIGE_MISMATCH)

        # POSABLE_SCOPE
        chapter, conf, trace = self.classifier.classify(atom.qi_clean, atom.rqi_clean)
        atom.chapter_detected = chapter
        atom.scope_confidence = conf
        scope_ok = bool(chapter) and conf >= 0.25
        if not scope_ok:
            reasons.append(ReasonCode.RC_SCOPE_UNRESOLVED)

        # POSABLE_EVALUABLE
        is_evaluable = bool(atom.verbe_action) or ("?" in (atom.qi_raw or "")) or len(atom.qi_clean) >= 40
        if not is_evaluable:
            reasons.append(ReasonCode.RC_NOT_A_QUESTION)

        posable = has_corr and scope_ok and is_evaluable

        return PosableDecision(
            qi_id=atom.qi_id,
            posable=posable,
            has_exploitable_correction=has_corr,
            scope_resolved=scope_ok,
            is_evaluable=is_evaluable,
            chapter_ref=chapter,
            scope_trace=trace,
            reason_codes=reasons,
            evidence_refs=evidence_refs,
        )


# =============================================================================
# 9) ARI / FRT (DERIVED OR RECONSTRUCTED)
# =============================================================================

class ARIExtractor:
    def extract(self, rqi_clean: Optional[str], qi_clean: str) -> Tuple[List[Dict[str, Any]], str]:
        if rqi_clean and len(rqi_clean) >= 30:
            steps: List[Dict[str, Any]] = []
            sentences = re.split(r"[.!?]\s+", rqi_clean)
            for sent in sentences:
                s = (sent or "").strip()
                if len(s) < 12:
                    continue
                low = s.lower()
                for verb in COGNITIVE_VERBS_TABLE.keys():
                    if re.search(rf"\b{re.escape(verb)}\b", low):
                        w = COGNITIVE_VERBS_TABLE[verb]
                        steps.append({"verb": verb, "weight": w, "evidence": s[:160]})
                        break
                if len(steps) >= 6:
                    break
            if len(steps) >= 2:
                return (steps, "EXTRACTED")

        # RECONSTRUCTED
        v = None
        lowq = (qi_clean or "").lower()
        for verb in COGNITIVE_VERBS_TABLE.keys():
            if re.search(rf"\b{re.escape(verb)}\b", lowq):
                v = verb
                break
        if v:
            return ([{"verb": v, "weight": COGNITIVE_VERBS_TABLE.get(v, 0.25), "evidence": "RECONSTRUCTED_FROM_QI"}], "RECONSTRUCTED")
        return ([{"verb": "analyser", "weight": COGNITIVE_VERBS_TABLE["analyser"], "evidence": "RECONSTRUCTED_MIN"}], "RECONSTRUCTED")


class FRTBuilder:
    def __init__(self, pack: PackConfig, ari_extractor: ARIExtractor):
        self.pack = pack
        self.ari_extractor = ari_extractor

    def build(self, atoms: List[Atom], chapter: str) -> Tuple[FRT, Signature, str]:
        qi_texts = [a.qi_clean for a in atoms]
        rqi_texts = [a.rqi_clean for a in atoms if a.rqi_clean]

        best_rqi = max(rqi_texts, key=lambda t: (len(t), t[:50])) if rqi_texts else None
        best_qi = max(qi_texts, key=lambda t: (len(t), t[:50])) if qi_texts else (qi_texts[0] if qi_texts else "")

        ari_steps, ari_source = self.ari_extractor.extract(best_rqi, best_qi)

        triggers = self._extract_triggers(qi_texts)
        title = self._build_title(ari_steps, chapter)
        output_type = self._detect_output_type(ari_steps, best_qi)

        precond_ids = ("PRECOND_GIVEN_DATA",)

        signature = Signature(
            action_spine=tuple([s["verb"] for s in ari_steps if s.get("verb")]),
            preconditions_set=precond_ids,
            output_type=output_type,
            checkpoints_core=("CHK_FINAL_VERIFICATION",),
        )

        frt = FRT(
            triggers=triggers,
            preconditions=["Données de l'énoncé (abstrait)"],
            exemples_enonce=qi_texts[:2],
            ari_steps=ari_steps,
            erreurs_frequentes=[],
            confusions_conceptuelles=[],
            oublis_fatals=[],
            phrase_conclusion="Conclusion canonique.",
            format_reponse="Sortie typée + justification.",
            elements_obligatoires=["Étapes typées", "Vérification finale"],
            source=ari_source,
            evidence_refs=[a.qi_id for a in atoms] + [a.rqi_id for a in atoms if a.rqi_id],
        )

        return (frt, signature, title)

    def _extract_triggers(self, qi_texts: List[str]) -> List[str]:
        freq: Dict[str, int] = defaultdict(int)
        verbs: List[str] = []
        for qi in qi_texts:
            low = (qi or "").lower()
            for verb in COGNITIVE_VERBS_TABLE.keys():
                if re.search(rf"\b{re.escape(verb)}\b", low) and verb not in verbs:
                    verbs.append(verb)
            for tok in _norm_words(qi):
                if tok not in self.pack.stopwords and len(tok) > 3:
                    freq[tok] += 1

        top_tokens = [t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))][:10]
        triggers = (verbs[:3] + top_tokens)[:7]
        if len(triggers) < 3:
            triggers += ["résoudre", "analyser", "déterminer"][: (3 - len(triggers))]
        return triggers[:7]

    def _build_title(self, ari_steps: List[Dict[str, Any]], chapter: str) -> str:
        main = "résoudre"
        for s in ari_steps:
            v = s.get("verb", "")
            if v in ("démontrer", "calculer", "déterminer", "justifier", "résoudre"):
                main = v
                break
        return f"Comment {main} un problème de type {chapter} ?"

    def _detect_output_type(self, ari_steps: List[Dict[str, Any]], qi_text: str) -> str:
        verbs = [s.get("verb", "") for s in ari_steps]
        lowq = (qi_text or "").lower()
        if any(v in PROOF_VERBS for v in verbs) or any(w in lowq for w in PROOF_VERBS):
            return "PROOF"
        if any(v in VALUE_VERBS for v in verbs) or ("calcul" in lowq):
            return "RESULT_VALUE"
        if any(v in TABLE_VERBS for v in verbs) or ("tableau" in lowq):
            return "TABLE"
        return "RESULT"


# =============================================================================
# 10) FORMULAS F1 / F2
# =============================================================================

def compute_psi_f1(ari_steps: List[Dict[str, Any]], delta_c: float) -> Tuple[float, float]:
    sum_tj = sum(float(s.get("weight", 0.0)) for s in ari_steps)
    psi_raw = float(delta_c) * (KernelConstants.EPSILON + sum_tj) ** 2
    return (psi_raw, sum_tj)


def compute_score_f2(
    n_q: int,
    n_total: int,
    t_rec: float,
    psi_norm: float,
    selected_qcs: List[QCCandidate],
    current_sig: Signature,
    alpha: float = KernelConstants.ALPHA_DEFAULT,
) -> float:
    if n_total <= 0:
        return 0.0
    freq_ratio = n_q / n_total
    t_rec_safe = max(KernelConstants.T_REC_MIN, float(t_rec))
    recency_boost = 1.0 + (float(alpha) / t_rec_safe)

    anti = 1.0
    for qc in selected_qcs:
        sigma = current_sig.similarity(qc.signature)
        if sigma > KernelConstants.SIGMA_QUASI_DOUBLON:
            anti *= (1.0 - sigma)

    return float(freq_ratio) * recency_boost * float(psi_norm) * anti * 100.0


# =============================================================================
# 11) ATTACH OPERATOR (AND STRICT)
# =============================================================================

class AttachOperator:
    def __init__(self, pack: PackConfig):
        self.pack = pack

    def _check_preconditions(self, atom: Atom, qc: QCCandidate) -> Tuple[bool, Optional[ReasonCode], List[str]]:
        evidence = [atom.qi_id] + ([atom.rqi_id] if atom.rqi_id else [])
        for pid in qc.signature.preconditions_set:
            matchers = self.pack.precondition_matchers.get(pid, [])
            if not matchers:
                continue
            text = (atom.qi_clean or "") + " " + (atom.rqi_clean or "")
            ok = any(re.search(pat, text, flags=re.IGNORECASE) for pat in matchers)
            if not ok:
                return (False, ReasonCode.ATT_PRECOND_FAIL, evidence)
        return (True, None, evidence)

    def _check_triggers(self, atom: Atom, qc: QCCandidate) -> bool:
        qi_tokens = set(_norm_words(atom.qi_clean))
        trig_tokens: Set[str] = set()
        for t in qc.triggers:
            trig_tokens |= set(_norm_words(t))
        return len(qi_tokens & trig_tokens) >= 1

    def _check_ari_compat(self, atom: Atom, qc: QCCandidate) -> bool:
        spine = set(qc.signature.action_spine)
        if atom.verbe_action and atom.verbe_action in spine:
            return True
        qi_verbs = set(_norm_words(atom.qi_clean)) & set(COGNITIVE_VERBS_TABLE.keys())
        return len(qi_verbs & spine) >= 1

    def _check_output_type(self, atom: Atom, qc: QCCandidate) -> bool:
        low = (atom.qi_clean or "").lower()
        if qc.signature.output_type == "PROOF":
            return any(w in low for w in ("démontrer", "prouver", "montrer"))
        if qc.signature.output_type == "RESULT_VALUE":
            return any(w in low for w in ("calculer", "déterminer", "valeur", "distance", "limite"))
        if qc.signature.output_type == "TABLE":
            return any(w in low for w in ("tableau", "variations", "étudier"))
        return True

    def attach(self, atom: Atom, qc: QCCandidate) -> AttachResult:
        pre_ok, pre_rc, pre_ev = self._check_preconditions(atom, qc)
        trig_ok = self._check_triggers(atom, qc)
        ari_ok = self._check_ari_compat(atom, qc)
        out_ok = self._check_output_type(atom, qc)

        attached = bool(pre_ok and trig_ok and ari_ok and out_ok)

        reason = None
        if not attached:
            if not pre_ok:
                reason = pre_rc or ReasonCode.ATT_PRECOND_FAIL
            elif not trig_ok:
                reason = ReasonCode.ATT_TRIGGER_MISS
            elif not ari_ok:
                reason = ReasonCode.ATT_SIGNATURE_MISMATCH
            elif not out_ok:
                reason = ReasonCode.ATT_OUTPUT_TYPE_MISMATCH

        evidence = list(dict.fromkeys(pre_ev))
        return AttachResult(
            qi_id=atom.qi_id,
            qc_id=qc.qc_id,
            attached=attached,
            reason_code=reason,
            evidence_refs=evidence,
            preconditions_ok=pre_ok,
            triggers_match=trig_ok,
            ari_compatible=ari_ok,
            output_type_ok=out_ok,
        )


# =============================================================================
# 12) COVERAGE + SELECTION (COVERAGE-DRIVEN)
# =============================================================================

class CoverageEngine:
    def __init__(self, attach_operator: AttachOperator):
        self.attach_operator = attach_operator

    def compute_map(self, atoms: Dict[str, Atom], selected: List[QCCandidate], chapter: str) -> CoverageMap:
        qi_to_qc: Dict[str, str] = {}
        orphans: Set[str] = set(atoms.keys())
        attach_results: List[AttachResult] = []

        for qi_id, atom in sorted(atoms.items(), key=lambda x: x[0]):
            best_qc = None
            best_tuple = (-1, "", "")
            for qc in selected:
                ar = self.attach_operator.attach(atom, qc)
                attach_results.append(ar)
                if ar.attached:
                    s = int(ar.preconditions_ok) + int(ar.triggers_match) + int(ar.ari_compatible) + int(ar.output_type_ok)
                    tie = (s, qc.qc_id, qc.signature.to_hash())
                    if tie > best_tuple:
                        best_tuple = tie
                        best_qc = qc
            if best_qc:
                qi_to_qc[qi_id] = best_qc.qc_id
                orphans.discard(qi_id)

        n_total = len(atoms)
        n_cov = len(qi_to_qc)
        ratio = (n_cov / n_total) if n_total else 1.0

        return CoverageMap(
            chapter_ref=chapter,
            qi_to_qc=qi_to_qc,
            orphans=orphans,
            attach_results=attach_results,
            n_total_target=n_total,
            n_covered=n_cov,
            coverage_ratio=ratio,
        )

    def select_coverage_driven(
        self,
        atoms: Dict[str, Atom],
        candidates: List[QCCandidate],
        chapter: str,
    ) -> Tuple[List[QCCandidate], CoverageMap, List[str]]:
        audit: List[str] = []
        uncovered: Set[str] = set(atoms.keys())
        selected: List[QCCandidate] = []

        attach_cache: Dict[Tuple[str, str], bool] = {}
        for qi_id, atom in atoms.items():
            for qc in candidates:
                attach_cache[(qi_id, qc.qc_id)] = self.attach_operator.attach(atom, qc).attached

        while uncovered:
            best = None
            best_tuple = (-1, -1.0, "", "")

            for qc in candidates:
                gain = sum(1 for qi_id in uncovered if attach_cache.get((qi_id, qc.qc_id), False))
                if gain <= 0:
                    continue
                # Patch 2: F2 dynamique avec anti-redondance sur selected
                score_dyn = compute_score_f2(
                    n_q=qc.n_q,
                    n_total=qc.n_total,
                    t_rec=qc.t_rec,
                    psi_norm=qc.psi_normalized,
                    selected_qcs=selected,
                    current_sig=qc.signature,
                )
                sig_hash = qc.signature.to_hash()
                tup = (gain, float(score_dyn), sig_hash, qc.qc_id)
                if tup > best_tuple:
                    best_tuple = tup
                    best = qc

            if not best:
                audit.append(f"[{chapter}] [SELECT] BLOCKED: no QC increases coverage; uncovered={len(uncovered)}")
                break

            selected.append(best)
            covered_now = {qi_id for qi_id in uncovered if attach_cache.get((qi_id, best.qc_id), False)}
            best.covered_qi_ids |= covered_now
            uncovered -= covered_now

            audit.append(f"[{chapter}] [SELECT] pick={best.qc_id} gain={len(covered_now)} uncovered={len(uncovered)}")

        cov_map = self.compute_map(atoms, selected, chapter)
        return (selected, cov_map, audit)


# =============================================================================
# 13) MAIN ENGINE
# =============================================================================

def _parse_extracted_at(extracted_at: Optional[str]) -> str:
    if extracted_at:
        return extracted_at
    return "1970-01-01T00:00:00Z"


def _year_diff(extracted_at: str, year_ref: int) -> float:
    try:
        dt = datetime.fromisoformat(extracted_at.replace("Z", "+00:00"))
    except Exception:
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return max(0.01, float(dt.year - int(year_ref)))


class GranuloEngineV104:
    def __init__(self, pack: PackConfig):
        self.pack = pack
        self.atomizer = BACAtomizer(pack)
        self.classifier = ScopeClassifier(pack)
        self.posable_gate = PosableGate(self.classifier)
        self.ari_extractor = ARIExtractor()
        self.frt_builder = FRTBuilder(pack, self.ari_extractor)
        self.attach_operator = AttachOperator(pack)
        self.coverage_engine = CoverageEngine(self.attach_operator)

    def process(
        self,
        subject_text: str,
        correction_text: Optional[str],
        source_id: str,
        year_ref: int,
        extracted_at: Optional[str] = None,
        gte_mode: bool = False,
    ) -> Dict[str, Any]:
        extracted_at_iso = _parse_extracted_at(extracted_at)

        fp_subject = "sha256:" + hashlib.sha256((subject_text or "").encode("utf-8")).hexdigest()
        fp_corr = None
        if correction_text:
            fp_corr = "sha256:" + hashlib.sha256((correction_text or "").encode("utf-8")).hexdigest()

        audit: List[str] = []
        out: Dict[str, Any] = {
            "kernel": {"version": KernelConstants.KERNEL_VERSION, "status": KernelConstants.KERNEL_STATUS},
            "pack": {"id": self.pack.pack_id, "version": self.pack.pack_version},
            "atoms": [],
            "posable_decisions": [],
            "gte_accepted_ids": [],
            "chapters_detected": {},
            "qc_candidates": [],
            "selected_qcs": [],
            "gte_qc_preview": {},  # Patch 3: separate GTE preview (not selected)
            "coverage_maps": {},
            "sealed_by_chapter": {},
            "audit_log": audit,
            "metrics": {},
        }

        # 1) Atomisation
        qi_items = self.atomizer.atomize(subject_text or "")
        rqi_items = self.atomizer.atomize(correction_text or "") if correction_text else []
        audit.append(f"[ATOM] Qi={len(qi_items)} RQi={len(rqi_items)}")

        # 2) Pairing
        atoms = PairingEngine().pair(
            qi_items=qi_items,
            rqi_items=rqi_items,
            subject_id=source_id,
            correction_id=(f"{source_id}-CORR" if correction_text else None),
            fingerprint_subject=fp_subject,
            fingerprint_correction=fp_corr,
            extracted_at=extracted_at_iso,
            year_ref=year_ref,
        )
        out["atoms"] = atoms

        # 3) POSABLE + Chapter grouping
        atoms_by_chapter_posable: Dict[str, Dict[str, Atom]] = defaultdict(dict)
        atoms_by_chapter_gte: Dict[str, Dict[str, Atom]] = defaultdict(dict)

        posable_count = 0
        for atom in atoms:
            dec = self.posable_gate.evaluate(atom)
            out["posable_decisions"].append(dec)

            if dec.chapter_ref:
                if gte_mode and dec.scope_resolved and dec.is_evaluable:
                    atoms_by_chapter_gte[dec.chapter_ref][atom.qi_id] = atom

            if dec.posable and dec.chapter_ref:
                atoms_by_chapter_posable[dec.chapter_ref][atom.qi_id] = atom
                posable_count += 1

        out["chapters_detected"] = {ch: len(d) for ch, d in atoms_by_chapter_posable.items()} if atoms_by_chapter_posable else {}

        # GTE chapters for demo
        if gte_mode:
            for ch, d in atoms_by_chapter_gte.items():
                if ch not in out["chapters_detected"]:
                    out["chapters_detected"][ch] = len(d)

        audit.append(f"[POSABLE] total_posable={posable_count} chapters={len(atoms_by_chapter_posable)} gte={gte_mode}")

        # 4) Generate QC candidates
        all_candidates: List[QCCandidate] = []
        chapters_all = sorted(set(list(atoms_by_chapter_posable.keys()) + (list(atoms_by_chapter_gte.keys()) if gte_mode else [])))

        for ch in chapters_all:
            pool = atoms_by_chapter_posable.get(ch) or (atoms_by_chapter_gte.get(ch) if gte_mode else {})
            if not pool:
                continue
            clusters = self._cluster_by_method(list(pool.values()))
            for c_idx, cluster in enumerate(clusters, start=1):
                qc = self._generate_qc(cluster, ch, year_ref, extracted_at_iso)
                if qc:
                    all_candidates.append(qc)

        audit.append(f"[QC] candidates={len(all_candidates)}")
        out["qc_candidates"] = all_candidates

        if all_candidates:
            max_psi = max(q.psi_raw for q in all_candidates) or 1.0
            for q in all_candidates:
                q.psi_normalized = q.psi_raw / max_psi

        for q in all_candidates:
            q.score = compute_score_f2(
                n_q=q.n_q,
                n_total=q.n_total,
                t_rec=q.t_rec,
                psi_norm=q.psi_normalized,
                selected_qcs=[],
                current_sig=q.signature,
            )

        # 5) Coverage-driven selection
        selected_all: List[QCCandidate] = []
        sealed_by_chapter: Dict[str, bool] = {}
        coverage_maps: Dict[str, CoverageMap] = {}

        for ch in sorted(atoms_by_chapter_posable.keys()):
            posable_atoms = atoms_by_chapter_posable[ch]
            chapter_candidates = [qc for qc in all_candidates if qc.chapter_ref == ch]

            if not chapter_candidates or not posable_atoms:
                sealed_by_chapter[ch] = False
                continue

            chapter_candidates = sorted(
                chapter_candidates,
                key=lambda qc: (qc.score, qc.signature.to_hash(), qc.qc_id),
                reverse=True
            )

            selected, cov_map, logs = self.coverage_engine.select_coverage_driven(posable_atoms, chapter_candidates, ch)
            audit.extend(logs)

            coverage_maps[ch] = cov_map
            coverage_ok = (len(cov_map.orphans) == 0)
            sealed_by_chapter[ch] = bool(coverage_ok)

            if not coverage_ok:
                audit.append(f"[COVERAGE_BOOL] {ch} = FAIL orphans={len(cov_map.orphans)}")
            else:
                audit.append(f"[COVERAGE_BOOL] {ch} = PASS")

            selected_all.extend(selected)

        # GTE chapters (never sealed) - Patch 3: separate preview, don't contaminate selected_qcs
        if gte_mode:
            for ch in sorted(atoms_by_chapter_gte.keys()):
                if ch in coverage_maps:
                    continue
                gte_atoms = atoms_by_chapter_gte[ch]
                chapter_candidates = [qc for qc in all_candidates if qc.chapter_ref == ch]
                if not chapter_candidates or not gte_atoms:
                    sealed_by_chapter.setdefault(ch, False)
                    continue
                cov_map = self.coverage_engine.compute_map(gte_atoms, chapter_candidates[:10], ch)
                coverage_maps[ch] = cov_map
                sealed_by_chapter.setdefault(ch, False)
                # Patch 3: store in separate preview field, not in selected_all
                out.setdefault("gte_qc_preview", {})[ch] = chapter_candidates[:10]

        out["coverage_maps"] = coverage_maps
        out["sealed_by_chapter"] = sealed_by_chapter

        seen: Set[str] = set()
        uniq: List[QCCandidate] = []
        for qc in sorted(selected_all, key=lambda q: (q.score, q.signature.to_hash(), q.qc_id), reverse=True):
            if qc.qc_id in seen:
                continue
            seen.add(qc.qc_id)
            uniq.append(qc)
        out["selected_qcs"] = uniq

        # Metrics
        total_posable = sum(len(v) for v in atoms_by_chapter_posable.values())
        total_covered = 0
        total_orphans = 0
        for ch, cov in coverage_maps.items():
            if ch in atoms_by_chapter_posable:
                total_covered += cov.n_covered
                total_orphans += len(cov.orphans)

        coverage_ratio = (total_covered / total_posable) if total_posable else 0.0
        seal_pass = sum(1 for ch, ok in sealed_by_chapter.items() if ok)

        # Count GTE preview QCs separately
        gte_qc_count = sum(len(qcs) for qcs in out.get("gte_qc_preview", {}).values())

        out["metrics"] = {
            "total_qi": len(atoms),
            "total_posable": total_posable,
            "total_qc": len(uniq),
            "total_gte_preview": gte_qc_count,
            "seal_pass_chapters": seal_pass,
            "coverage": coverage_ratio,
            "orphans": total_orphans,
        }

        return out

    def _cluster_by_method(self, atoms: List[Atom]) -> List[List[Atom]]:
        clusters: Dict[str, List[Atom]] = defaultdict(list)
        for a in atoms:
            text = (a.rqi_clean or a.qi_clean or "").lower()
            if "récurrence" in text or "hérédité" in text or "initialisation" in text:
                tag = "RECURRENCE"
            elif "dérivée" in text or "variation" in text:
                tag = "DERIVATION"
            elif "intégrale" in text or "primitive" in text:
                tag = "INTEGRATION"
            elif "probabilité" in text or "binom" in text:
                tag = "PROBA"
            else:
                tag = "GENERIC"
            clusters[tag].append(a)
        return [clusters[k] for k in sorted(clusters.keys())]

    def _generate_qc(self, cluster: List[Atom], chapter: str, year_ref: int, extracted_at_iso: str) -> Optional[QCCandidate]:
        if not cluster:
            return None

        frt, sig, title = self.frt_builder.build(cluster, chapter)

        delta_c = float(self.pack.delta_c_by_chapter.get(chapter, 1.0))
        psi_raw, sum_tj = compute_psi_f1(frt.ari_steps, delta_c)

        t_rec = _year_diff(extracted_at_iso, year_ref)
        n_total = len(cluster)

        qc_id = f"QC-{sig.to_hash()}-{chapter[:3]}"

        ari_lines = [f"{s.get('verb','').upper()}: {str(s.get('evidence',''))[:80]}" for s in frt.ari_steps]

        return QCCandidate(
            qc_id=qc_id,
            chapter_ref=chapter,
            title=title,
            psi_raw=psi_raw,
            psi_normalized=0.0,
            sum_tj=sum_tj,
            score=0.0,
            n_q=len(cluster),
            n_total=n_total,
            t_rec=t_rec,
            freq_ratio=(len(cluster) / n_total) if n_total else 0.0,
            recency_boost=0.0,
            signature=sig,
            frt=frt,
            triggers=frt.triggers,
            ari=ari_lines,
            source_qi_ids=[a.qi_id for a in cluster],
            source_rqi_ids=[a.rqi_id for a in cluster if a.rqi_id],
        )


# =============================================================================
# 14) PUBLIC ENTRYPOINT
# =============================================================================

def run_granulo_v104(
    subject_text: str,
    correction_text: Optional[str] = None,
    source_id: str = "BAC",
    year_ref: int = 2023,
    extracted_at: Optional[str] = None,
    gte_mode: bool = False,
    pack: Optional[PackConfig] = None,
) -> Dict[str, Any]:
    pack = pack or create_pack_france_terminale_maths()
    engine = GranuloEngineV104(pack)
    return engine.process(
        subject_text=subject_text,
        correction_text=correction_text,
        source_id=source_id,
        year_ref=year_ref,
        extracted_at=extracted_at,
        gte_mode=gte_mode,
    )


# =============================================================================
# 15) SMOKE TEST
# =============================================================================


# =============================================================================
# ADAPTER V31 ↔ V10.4
# =============================================================================

CHAPTER_KEYWORDS = {
    "SUITES NUMÉRIQUES": ["suite", "récurrence", "convergence", "limite suite", "arithmétique", "géométrique", "un+1", "u(n)"],
    "FONCTIONS": ["fonction", "dérivée", "primitive", "limite", "continuité", "asymptote", "variation", "extremum"],
    "INTÉGRALES": ["intégrale", "primitive", "aire", "calcul intégral", "intégration"],
    "PROBABILITÉS": ["probabilité", "loi", "espérance", "variance", "binomiale", "normale", "aléatoire", "événement"],
    "GÉOMÉTRIE DANS L'ESPACE": ["espace", "vecteur", "plan", "droite", "orthogonal", "colinéaire", "paramétrique"],
    "NOMBRES COMPLEXES": ["complexe", "module", "argument", "affixe", "conjugué", "forme exponentielle"],
    "MATRICES": ["matrice", "déterminant", "inverse", "système"],
    "ARITHMÉTIQUE": ["divisibilité", "pgcd", "ppcm", "premier", "congruence", "bezout"]
}

# =============================================================================
# HELPERS
# =============================================================================

def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extrait le texte d'un PDF"""
    if not pdfplumber:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        return ""


def _detect_year_from_text(text: str) -> int:
    """Détecte l'année dans le texte"""
    matches = re.findall(r'20[0-2][0-9]', text[:500])
    if matches:
        return int(matches[0])
    return 2023


def _detect_nature(filename: str) -> str:
    """Détecte la nature du sujet (BAC, DST, etc.)"""
    fn_lower = filename.lower()
    if "bac" in fn_lower:
        return "BAC"
    elif "dst" in fn_lower:
        return "DST"
    elif "dm" in fn_lower:
        return "DM"
    else:
        return "Autre"


def _convert_qc_to_v31_row(qc: QCCandidate, atoms_by_qc: Dict[str, List[Atom]]) -> Dict[str, Any]:
    """Convertit une QC V10.4 vers le format de ligne attendu par df_qc V31"""
    
    # Récupérer les atoms associés
    associated_atoms = atoms_by_qc.get(qc.qc_id, [])
    
    # Construire Evidence (format V31)
    evidence = []
    for atom in associated_atoms:
        evidence.append({
            "Fichier": atom.qi_evidence.source_id if atom.qi_evidence else "unknown",
            "Qi": atom.qi_clean[:150] if atom.qi_clean else atom.qi_raw[:150]
        })
    
    # Construire FRT_DATA (4 blocs V31)
    frt_data = [
        {
            "type": "usage",
            "title": "📌 QUAND UTILISER",
            "text": f"Triggers: {', '.join(qc.frt.triggers[:4])}\nPréconditions: {', '.join(qc.frt.preconditions[:3])}"
        },
        {
            "type": "method",
            "title": "✅ MÉTHODE",
            "text": qc.frt.format_reponse or "Appliquer l'ARI ci-contre"
        },
        {
            "type": "trap",
            "title": "⚠️ PIÈGES",
            "text": "\n".join(["• " + e for e in (qc.frt.erreurs_frequentes + qc.frt.oublis_fatals)[:4]]) or "Aucun piège identifié"
        },
        {
            "type": "conc",
            "title": "🎯 CONCLUSION",
            "text": qc.frt.phrase_conclusion or "Conclure selon le format attendu"
        }
    ]
    
    return {
        "QC_ID": qc.qc_id,
        "Titre": qc.title,
        "Chapitre": qc.chapter_ref,
        "Score": qc.score,
        "n_q": qc.n_q,
        "Psi": f"{qc.psi_normalized:.2f}",
        "N_tot": qc.n_total,
        "t_rec": f"{qc.t_rec:.1f}",
        "Triggers": qc.triggers[:7],
        "ARI": qc.ari[:10],
        "FRT_DATA": frt_data,
        "Evidence": evidence,
        # Champs additionnels pour le moteur V10.4
        "FRT_Source": qc.frt.source,
        "Signature": qc.signature.to_hash() if qc.signature else "",
        "Sealed": False  # Sera mis à jour selon coverage
    }


# =============================================================================
# FONCTIONS COMPATIBLES V31
# =============================================================================

def ingest_real(
    url_list: List[str],
    volume: int,
    matiere: str,
    chapter_filter: Optional[str],
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Fonction d'ingestion compatible V31.
    
    En mode Streamlit Cloud, le scraping web peut être limité.
    Cette fonction accepte aussi des PDFs uploadés directement.
    
    Returns:
        df_src: DataFrame des sujets traités (Fichier, Nature, Année, Téléchargement, Qi_Data)
        df_atoms: DataFrame des atoms (pour debug)
        all_qis: Liste de toutes les Qi extraites (format V31)
    """
    
    # Pour l'instant, on utilise un mode simplifié sans scraping web
    # L'utilisateur doit uploader les PDFs directement
    
    df_src = pd.DataFrame(columns=["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"])
    df_atoms = pd.DataFrame()
    all_qis = []
    
    if progress_callback:
        progress_callback(1.0)
    
    return df_src, df_atoms, all_qis


def ingest_from_pdfs(
    pdf_files: List[Tuple[str, bytes]],
    matiere: str,
    chapter_filter: Optional[str] = None,
    correction_files: Optional[List[Tuple[str, bytes]]] = None,
    gte_mode: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ingestion depuis des PDFs uploadés (nouvelle fonction V10.4).
    
    Args:
        pdf_files: Liste de tuples (filename, bytes)
        matiere: Matière sélectionnée
        chapter_filter: Filtre chapitre optionnel
        correction_files: Corrigés optionnels (filename, bytes)
        gte_mode: Mode GTE (preview sans corrigé)
        progress_callback: Callback de progression
    
    Returns:
        df_src: DataFrame des sujets
        df_atoms: DataFrame des atoms
        all_qis: Liste Qi format V31
        v104_results: Résultats bruts V10.4 pour audit avancé
    """
    
    pack = create_pack_france_terminale_maths()
    
    rows_src = []
    all_atoms = []
    all_qis = []
    all_v104_results = []
    
    # Map des corrigés par nom de fichier
    corrections_map = {}
    if correction_files:
        for fname, fbytes in correction_files:
            # Essayer de matcher avec le sujet
            key = fname.lower().replace("corrige", "").replace("correction", "").replace("_", "").replace("-", "")
            corrections_map[key] = fbytes
    
    total = len(pdf_files)
    
    for i, (filename, pdf_bytes) in enumerate(pdf_files):
        if progress_callback:
            progress_callback((i + 1) / total)
        
        # Extraction texte sujet
        subject_text = _extract_pdf_text(pdf_bytes)
        if not subject_text or len(subject_text) < 100:
            continue
        
        # Chercher un corrigé correspondant
        correction_text = None
        key = filename.lower().replace("sujet", "").replace("_", "").replace("-", "")
        for ckey, cbytes in corrections_map.items():
            if key in ckey or ckey in key:
                correction_text = _extract_pdf_text(cbytes)
                break
        
        # Détecter année et nature
        year_ref = _detect_year_from_text(subject_text)
        nature = _detect_nature(filename)
        
        # Appel moteur V10.4
        result = run_granulo_v104(
            subject_text=subject_text,
            correction_text=correction_text,
            source_id=filename,
            year_ref=year_ref,
            extracted_at=datetime.now().isoformat() + "Z",
            gte_mode=gte_mode,
            pack=pack
        )
        
        all_v104_results.append(result)
        
        # Collecter les atoms
        for atom in result.get('atoms', []):
            all_atoms.append(atom)
            # Format V31 pour Qi
            all_qis.append({
                "qi_id": atom.qi_id,
                "qi_text": atom.qi_clean,
                "chapter": atom.chapter_detected,
                "source": filename,
                "has_rqi": bool(atom.rqi_id),
                "verbe": atom.verbe_action
            })
        
        # Row pour df_src
        rows_src.append({
            "Fichier": filename,
            "Nature": nature,
            "Annee": year_ref,
            "Telechargement": "",  # Pas de lien en mode upload
            "Qi_Data": [{"qi_text": a.qi_clean, "chapter": a.chapter_detected} for a in result.get('atoms', [])]
        })
    
    df_src = pd.DataFrame(rows_src)
    df_atoms = pd.DataFrame([{
        "qi_id": a.qi_id,
        "qi_clean": a.qi_clean[:80],
        "chapter": a.chapter_detected,
        "verbe": a.verbe_action,
        "rqi": "✅" if a.rqi_id else "❌"
    } for a in all_atoms])
    
    # Agrégation des résultats V10.4
    aggregated_v104 = {
        "all_results": all_v104_results,
        "total_qi": sum(r['metrics']['total_qi'] for r in all_v104_results),
        "total_posable": sum(r['metrics']['total_posable'] for r in all_v104_results),
        "total_qc": sum(r['metrics']['total_qc'] for r in all_v104_results),
        "total_gte_preview": sum(r['metrics'].get('total_gte_preview', 0) for r in all_v104_results)
    }
    
    return df_src, df_atoms, all_qis, aggregated_v104


def compute_qc_real(all_qis: List[Dict[str, Any]], v104_results: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Génère le DataFrame QC au format V31 à partir des résultats V10.4.
    
    Args:
        all_qis: Liste des Qi (format V31)
        v104_results: Résultats agrégés du moteur V10.4
    
    Returns:
        df_qc: DataFrame avec colonnes V31 (QC_ID, Titre, Chapitre, Score, n_q, ...)
    """
    
    if not v104_results or not v104_results.get('all_results'):
        return pd.DataFrame()
    
    rows = []
    seen_qc_ids = set()
    
    # Construire le mapping atoms → QC
    all_atoms_by_qc = defaultdict(list)
    
    for result in v104_results['all_results']:
        # Collecter les atoms
        atoms_map = {a.qi_id: a for a in result.get('atoms', [])}
        
        # QC sélectionnées (Kernel)
        for qc in result.get('selected_qcs', []):
            if qc.qc_id not in seen_qc_ids:
                # Trouver les atoms associés via coverage_maps
                for ch, cov_map in result.get('coverage_maps', {}).items():
                    for qi_id, qc_id in cov_map.qi_to_qc.items():
                        if qc_id == qc.qc_id and qi_id in atoms_map:
                            all_atoms_by_qc[qc.qc_id].append(atoms_map[qi_id])
                
                row = _convert_qc_to_v31_row(qc, all_atoms_by_qc)
                row["Sealed"] = result.get('sealed_by_chapter', {}).get(qc.chapter_ref, False)
                row["Mode"] = "KERNEL"
                rows.append(row)
                seen_qc_ids.add(qc.qc_id)
        
        # QC preview GTE (non scellables)
        for ch, qcs in result.get('gte_qc_preview', {}).items():
            for qc in qcs:
                if qc.qc_id not in seen_qc_ids:
                    # En GTE, on associe les atoms par chapitre
                    gte_atoms = [a for a in result.get('atoms', []) if a.chapter_detected == ch]
                    all_atoms_by_qc[qc.qc_id] = gte_atoms[:qc.n_q]
                    
                    row = _convert_qc_to_v31_row(qc, all_atoms_by_qc)
                    row["Sealed"] = False  # GTE = jamais scellé
                    row["Mode"] = "GTE_PREVIEW"
                    rows.append(row)
                    seen_qc_ids.add(qc.qc_id)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Trier par chapitre puis score
    df = df.sort_values(["Chapitre", "Score"], ascending=[True, False])
    
    return df


def compute_saturation_real(all_qis: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calcule la courbe de saturation (nombre de QC en fonction des sujets).
    Compatible V31.
    """
    
    if not all_qis:
        return pd.DataFrame()
    
    # Grouper par source
    sources = defaultdict(list)
    for qi in all_qis:
        sources[qi.get('source', 'unknown')].append(qi)
    
    source_list = list(sources.keys())
    
    # Simuler l'ajout progressif
    cumulative_qis = []
    seen_signatures = set()
    data = []
    
    for i, src in enumerate(source_list, 1):
        cumulative_qis.extend(sources[src])
        
        # Compter les "QC uniques" par clustering simple
        for qi in sources[src]:
            # Signature simplifiée basée sur chapitre + premiers mots
            text = qi.get('qi_text', '')[:50].lower()
            chapter = qi.get('chapter', 'unknown')
            sig = f"{chapter}:{text}"
            seen_signatures.add(sig)
        
        # Approximation du nombre de QC (clusters)
        n_qc = len(seen_signatures) // 3 + 1  # Heuristique: ~3 Qi par QC
        
        data.append({
            "Sujets (N)": i,
            "QC Découvertes": min(n_qc, len(seen_signatures))
        })
    
    return pd.DataFrame(data)


def audit_internal_real(qi_data: List[Dict], df_qc: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Audit interne: vérifie que chaque Qi d'un sujet traité est couverte par une QC.
    Compatible V31.
    """
    
    if df_qc.empty or not qi_data:
        return []
    
    results = []
    
    for qi_item in qi_data:
        qi_text = qi_item.get('qi_text', '')[:100]
        qi_chapter = qi_item.get('chapter', '')
        
        # Chercher une QC correspondante
        matched_qc = None
        best_score = 0
        
        for _, qc_row in df_qc.iterrows():
            if qc_row['Chapitre'] != qi_chapter:
                continue
            
            # Score de matching basé sur les triggers
            triggers = qc_row.get('Triggers', [])
            if isinstance(triggers, list):
                score = sum(1 for t in triggers if t.lower() in qi_text.lower())
                if score > best_score:
                    best_score = score
                    matched_qc = qc_row['QC_ID']
        
        results.append({
            "Qi": qi_text,
            "Chapitre": qi_chapter,
            "QC_Matched": matched_qc or "-",
            "Statut": "✅ MATCH" if matched_qc else "❌ ORPHELIN"
        })
    
    return results


def audit_external_real(pdf_bytes: bytes, df_qc: pd.DataFrame, chapter_filter: Optional[str] = None) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Audit externe: analyse un PDF inconnu et vérifie la couverture.
    Compatible V31.
    """
    
    # Extraire le texte
    text = _extract_pdf_text(pdf_bytes)
    if not text:
        return 0.0, []
    
    # Utiliser le moteur V10.4 pour extraire les Qi
    pack = create_pack_france_terminale_maths()
    result = run_granulo_v104(
        subject_text=text,
        correction_text=None,
        source_id="external_audit",
        year_ref=2023,
        extracted_at=datetime.now().isoformat() + "Z",
        gte_mode=True,
        pack=pack
    )
    
    atoms = result.get('atoms', [])
    
    if not atoms:
        return 0.0, []
    
    # Vérifier la couverture
    results = []
    matched_count = 0
    
    for atom in atoms:
        qi_text = atom.qi_clean[:100]
        qi_chapter = atom.chapter_detected
        
        # Appliquer le filtre chapitre si spécifié
        if chapter_filter and qi_chapter != chapter_filter:
            continue
        
        # Chercher une QC correspondante
        matched_qc = None
        best_score = 0
        
        for _, qc_row in df_qc.iterrows():
            if qc_row['Chapitre'] != qi_chapter:
                continue
            
            triggers = qc_row.get('Triggers', [])
            if isinstance(triggers, list):
                score = sum(1 for t in triggers if t.lower() in qi_text.lower())
                if score > best_score:
                    best_score = score
                    matched_qc = qc_row['QC_ID']
        
        if matched_qc:
            matched_count += 1
        
        results.append({
            "Qi": qi_text,
            "Chapitre": qi_chapter,
            "QC_Matched": matched_qc or "-",
            "Statut": "✅ MATCH" if matched_qc else "❌ ORPHELIN"
        })
    
    coverage = (matched_count / len(results) * 100) if results else 0.0
    
    return coverage, results


# =============================================================================
# FONCTIONS ADDITIONNELLES V10.4 (pour UI enrichie)
# =============================================================================

def get_coverage_details(v104_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Retourne les détails de coverage par chapitre depuis les résultats V10.4.
    """
    
    coverage_details = {}
    
    for result in v104_results.get('all_results', []):
        for ch, sealed in result.get('sealed_by_chapter', {}).items():
            cov_map = result.get('coverage_maps', {}).get(ch)
            
            if cov_map:
                coverage_details[ch] = {
                    "sealed": sealed,
                    "coverage_ratio": cov_map.coverage_ratio,
                    "n_covered": cov_map.n_covered,
                    "n_total": cov_map.n_total_target,
                    "orphans": len(cov_map.orphans),
                    "orphan_ids": list(cov_map.orphans)[:5]  # Max 5 pour l'affichage
                }
    
    return coverage_details


def get_audit_log(v104_results: Dict[str, Any]) -> List[str]:
    """
    Retourne le log d'audit agrégé depuis les résultats V10.4.
    """
    
    logs = []
    
    for result in v104_results.get('all_results', []):
        logs.extend(result.get('audit_log', []))
    
    return logs

# =============================================================================
# CONSOLE UI V31
# =============================================================================

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
    <div style="background: linear-gradient(135deg, #2563eb, #1d4ed8); width: 50px; height: 50px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
        <span style="font-size: 24px;">🛡️</span>
    </div>
    <span style="font-size: 2.2rem; font-weight: 800; color: #1e293b;">SMAXIA - Console V31 (Saturation Proof)</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 🎨 STYLES CSS (GABARIT SMAXIA V31 - INCHANGÉ)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
    .qc-header-box {
        background-color: #f8f9fa; border-left: 6px solid #2563eb; 
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .qc-id-text { color: #d97706; font-weight: 900; font-size: 1.2em; margin-right: 10px; }
    .qc-title-text { color: #1f2937; font-weight: 700; font-size: 1.15em; }
    .qc-meta-text { 
        font-family: 'Courier New', monospace; font-size: 0.85em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 4px 8px; border-radius: 4px; margin-top: 5px; display: inline-block;
    }

    /* DETAILS */
    .trigger-item {
        background-color: #fff1f2; color: #991b1b; padding: 5px 10px; margin-bottom: 4px; 
        border-radius: 4px; border-left: 4px solid #f87171; font-weight: 600; font-size: 0.9em; display: block;
    }
    .ari-step {
        background-color: #f3f4f6; color: #374151; padding: 4px 8px; margin-bottom: 3px; 
        border-radius: 3px; font-family: monospace; font-size: 0.85em; border: 1px dashed #d1d5db; display: block;
    }

    /* FRT */
    .frt-block { padding: 12px; border-bottom: 1px solid #e2e8f0; background: white; margin-bottom: 5px; border-radius: 4px; border: 1px solid #e2e8f0;}
    .frt-title { font-weight: 800; text-transform: uppercase; font-size: 0.75em; display: block; margin-bottom: 6px; letter-spacing: 0.5px; }
    .frt-content { font-family: 'Segoe UI', sans-serif; font-size: 0.95em; color: #334155; line-height: 1.6; white-space: pre-wrap; }
    
    .c-usage { color: #d97706; border-left: 4px solid #d97706; }
    .c-method { color: #059669; border-left: 4px solid #059669; background-color: #f0fdf4; }
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; }
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; }

    /* QI CARDS */
    .file-block { margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
    .file-header { background-color: #f1f5f9; padding: 8px 12px; font-weight: 700; font-size: 0.85em; color: #475569; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; }
    .qi-item { background-color: white; padding: 10px 12px; border-bottom: 1px solid #f8fafc; font-family: 'Georgia', serif; font-size: 0.95em; color: #1e293b; border-left: 3px solid #9333ea; margin: 0; }

    /* SATURATION */
    .sat-box { background-color: #f0f9ff; border: 1px solid #bae6fd; padding: 20px; border-radius: 8px; margin-top: 20px; }
    
    /* STATUS */
    .status-kernel { background-color: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-gte { background-color: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-info { background-color: #dbeafe; color: #1e40af; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; }
    
    /* TABLE SUJETS */
    .sujets-table { width: 100%; border-collapse: collapse; }
    .sujets-table th { background: #f1f5f9; padding: 10px; text-align: left; font-weight: 600; color: #475569; }
    .sujets-table td { padding: 10px; border-bottom: 1px solid #e2e8f0; }
    .pdf-link { color: #dc2626; text-decoration: none; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR - PARAMÈTRES ACADÉMIQUES
# ==============================================================================
LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE DANS L'ESPACE", 
              "INTÉGRALES", "NOMBRES COMPLEXES", "MATRICES", "ARITHMÉTIQUE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

with st.sidebar:
    st.header("📚 Paramètres Académiques")
    
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps = LISTE_CHAPITRES.get(sel_matiere, [])
    sel_chapitres = st.multiselect("Chapitres", chaps, default=chaps[:1] if chaps else [])
    
    st.divider()
    
    # Mode Kernel
    st.markdown(f'<span class="status-kernel">Kernel {KernelConstants.KERNEL_VERSION}</span>', unsafe_allow_html=True)
    st.caption(f"Status: {KernelConstants.KERNEL_STATUS}")
    
    gte_mode = st.checkbox(
        "Mode GTE (prévisualisation sans corrigé)",
        value=True,
        help="Active le mode prévisualisation. Les QC générées ne sont pas scellables."
    )
    
    if gte_mode:
        st.markdown('<span class="status-gte">⚠️ Mode GTE: QC non scellables</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Stats en temps réel
    if 'all_qis' in st.session_state and st.session_state['all_qis']:
        st.markdown("### 📊 Statistiques")
        st.metric("Qi extraites", len(st.session_state['all_qis']))
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            st.metric("QC générées", len(st.session_state['df_qc']))
        if 'v104_results' in st.session_state:
            v104 = st.session_state['v104_results']
            st.metric("POSABLES", v104.get('total_posable', 0))

# ==============================================================================
# ONGLETS
# ==============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# ==============================================================================
# ONGLET 1 - USINE
# ==============================================================================
with tab_usine:
    
    # --- ZONE INJECTION ---
    c1, c2 = st.columns([3, 1])
    with c1:
        urls = st.text_area(
            "URLs Sources",
            "https://apmep.fr",
            height=80,
            help="URLs de référence (actuellement en mode upload PDF direct)"
        )
    with c2:
        vol = st.number_input("Volume", 1, 50, 5, step=1)
        run = st.button("🚀 LANCER L'USINE", type="primary", use_container_width=True)

    st.divider()
    
    # --- UPLOAD PDFs ---
    st.markdown("### 📄 Sujets à traiter")
    
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        subject_files = st.file_uploader(
            "📥 Sujets (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True,
            key="subjects",
            help="Uploadez les sujets d'examen au format PDF"
        )
    with col_upload2:
        correction_files = st.file_uploader(
            "📝 Corrigés (PDF, optionnel)", 
            type=["pdf"], 
            accept_multiple_files=True,
            key="corrections",
            help="Uploadez les corrigés correspondants pour activer le mode SEALED"
        )

    # --- EXÉCUTION ---
    if run and subject_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Extraction et analyse des PDFs...")
        
        try:
            # Préparer les fichiers
            pdf_list = [(f.name, f.read()) for f in subject_files]
            for f in subject_files:
                f.seek(0)
            
            corr_list = None
            if correction_files:
                corr_list = [(f.name, f.read()) for f in correction_files]
                for f in correction_files:
                    f.seek(0)
            
            chapter_filter = sel_chapitres[0] if sel_chapitres else None
            
            # Appel adapter V10.4
            df_src, df_atoms, all_qis, v104_results = ingest_from_pdfs(
                pdf_files=pdf_list,
                matiere=sel_matiere,
                chapter_filter=chapter_filter,
                correction_files=corr_list,
                gte_mode=gte_mode,
                progress_callback=lambda p: progress_bar.progress(p)
            )
            
            if all_qis:
                status_text.text("🧠 Génération des QC...")
                df_qc = compute_qc_real(all_qis, v104_results)
                
                st.session_state['df_src'] = df_src
                st.session_state['df_qc'] = df_qc
                st.session_state['all_qis'] = all_qis
                st.session_state['v104_results'] = v104_results
                st.session_state['chapter_filter'] = chapter_filter
                
                status_text.empty()
                progress_bar.empty()
                
                mode_str = "GTE (preview)" if gte_mode else "KERNEL"
                st.success(f"✅ Traitement terminé [{mode_str}]: {len(df_src)} sujets, {len(all_qis)} Qi, {len(df_qc)} QC")
            else:
                status_text.empty()
                progress_bar.empty()
                st.warning("⚠️ Aucune Qi extraite. Vérifiez les PDFs.")
                
        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.error(f"❌ Erreur : {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    st.divider()

    # --- TABLEAU SUJETS ---
    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        df_view = st.session_state['df_src'].copy()
        if 'Annee' in df_view.columns:
            df_view = df_view.rename(columns={"Annee": "Année"})
        
        display_cols = [c for c in ["Fichier", "Nature", "Année"] if c in df_view.columns]
        if display_cols:
            st.dataframe(df_view[display_cols], hide_index=True, use_container_width=True)

        st.divider()

        # --- BASE QC ---
        st.markdown("### 🧠 Base de Connaissance (QC)")
        
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            df_qc = st.session_state['df_qc']
            
            # Filtrer par chapitres sélectionnés
            if sel_chapitres:
                qc_view = df_qc[df_qc["Chapitre"].isin(sel_chapitres)]
            else:
                qc_view = df_qc
            
            if qc_view.empty:
                st.info("Aucune QC pour les chapitres sélectionnés.")
            else:
                chapters = qc_view["Chapitre"].unique()
                
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    
                    # Badge sealed si au moins une QC scellée
                    any_sealed = subset.get("Sealed", pd.Series([False])).any()
                    seal_badge = " ✅ SEALED" if any_sealed else ""
                    
                    st.markdown(f"#### 📘 {chap} ({len(subset)} QC){seal_badge}")
                    
                    for idx, row in subset.iterrows():
                        # Mode badge
                        mode = row.get('Mode', 'KERNEL')
                        mode_badge = ""
                        if mode == "GTE_PREVIEW":
                            mode_badge = '<span style="background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:4px;font-size:0.7em;margin-left:8px;">GTE</span>'
                        elif row.get('Sealed', False):
                            mode_badge = '<span style="background:#dcfce7;color:#166534;padding:2px 6px;border-radius:4px;font-size:0.7em;margin-left:8px;">SEALED</span>'
                        
                        # Header QC
                        st.markdown(f"""
                        <div class="qc-header-box">
                            <span class="qc-id-text">{row['QC_ID']}</span>{mode_badge}
                            <span class="qc-title-text">{row['Titre']}</span><br>
                            <span class="qc-meta-text">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_réc={row['t_rec']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 4 colonnes
                        c1, c2, c3, c4 = st.columns(4)
                        
                        with c1:
                            with st.expander("🔥 Déclencheurs"):
                                triggers = row['Triggers'] if isinstance(row['Triggers'], list) else []
                                if triggers:
                                    for t in triggers:
                                        st.markdown(f"<span class='trigger-item'>\"{t}\"</span>", unsafe_allow_html=True)
                                else:
                                    st.caption("Aucun déclencheur identifié")
                        
                        with c2:
                            with st.expander("⚙️ ARI"):
                                ari = row['ARI'] if isinstance(row['ARI'], list) else []
                                if ari:
                                    for s in ari:
                                        st.markdown(f"<span class='ari-step'>{s}</span>", unsafe_allow_html=True)
                                else:
                                    st.caption("ARI non disponible")
                        
                        with c3:
                            with st.expander("🧾 FRT"):
                                frt_data = row['FRT_DATA'] if isinstance(row['FRT_DATA'], list) else []
                                for block in frt_data:
                                    cls = {"usage": "c-usage", "method": "c-method", "trap": "c-trap", "conc": "c-conc"}.get(block.get('type', ''), "")
                                    st.markdown(f"<div class='frt-block {cls}'><span class='frt-title'>{block.get('title', '')}</span><div class='frt-content'>{block.get('text', '')}</div></div>", unsafe_allow_html=True)
                        
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                evidence = row['Evidence'] if isinstance(row['Evidence'], list) else []
                                qi_by_file = defaultdict(list)
                                for item in evidence:
                                    qi_by_file[item.get('Fichier', 'unknown')].append(item.get('Qi', ''))
                                
                                html = ""
                                for f, qlist in qi_by_file.items():
                                    html += f"<div class='file-block'><div class='file-header'>📁 {f}</div>"
                                    for q in qlist[:5]:
                                        q_display = q[:100] + "..." if len(q) > 100 else q
                                        html += f"<div class='qi-item'>\"{q_display}\"</div>"
                                    if len(qlist) > 5:
                                        html += f"<div class='qi-item'>… +{len(qlist)-5} autres</div>"
                                    html += "</div>"
                                st.markdown(html, unsafe_allow_html=True)
                        
                        st.write("")
        else:
            st.warning("Aucune QC générée. Lancez l'usine d'abord.")
        
        # --- SATURATION ---
        st.divider()
        st.markdown("### 📈 Analyse de Saturation (Preuve de Complétude)")
        st.caption("Ce graphique montre l'évolution du nombre de QC en fonction des sujets traités.")
        
        if 'all_qis' in st.session_state and st.session_state['all_qis']:
            if st.button("📊 Calculer la courbe de saturation"):
                with st.spinner("Calcul de la saturation..."):
                    df_sat = compute_saturation_real(st.session_state['all_qis'])
                    
                    if not df_sat.empty:
                        st.line_chart(df_sat, x="Sujets (N)", y="QC Découvertes")
                        
                        st.markdown("#### 🔢 Données de Convergence")
                        step = max(1, len(df_sat) // 10)
                        df_display = df_sat.iloc[::step].reset_index(drop=True)
                        st.dataframe(df_display, use_container_width=True)
                        
                        max_qc = df_sat["QC Découvertes"].max()
                        last_values = df_sat["QC Découvertes"].tail(3).tolist()
                        
                        if len(set(last_values)) == 1:
                            st.success(f"✅ **Saturation atteinte !** ({max_qc} QC max)")
                        else:
                            sat_90 = df_sat[df_sat["QC Découvertes"] >= max_qc * 0.9]
                            if not sat_90.empty:
                                sat_point = sat_90.iloc[0]["Sujets (N)"]
                                st.info(f"📈 **Saturation ~90% atteinte à {sat_point} sujets.**")
                            else:
                                st.warning("⚠️ Saturation non atteinte. Ajoutez plus de sujets.")
                    else:
                        st.warning("Pas assez de données pour la saturation.")
        else:
            st.info("Lancez l'usine pour voir la courbe de saturation.")
    else:
        st.info("⏳ Uploadez des sujets PDF et lancez l'usine pour commencer.")

# ==============================================================================
# ONGLET 2 - AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("🔍 Validation Booléenne")
    
    st.markdown("""
    **Objectifs Kernel V10.4 :**
    - **POSABLE** : corrigé exploitable ∧ scope ∧ évaluable
    - **Audit Interne** : Chaque Qi d'un sujet traité → QC = **100%**
    - **Audit Externe** : Sujet inconnu → QC = **≥ 95%**
    """)
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        # --- COVERAGE DETAILS (V10.4) ---
        if 'v104_results' in st.session_state:
            st.divider()
            st.markdown("#### 📊 Coverage par Chapitre (Kernel V10.4)")
            
            coverage_details = get_coverage_details(st.session_state['v104_results'])
            
            if coverage_details:
                for ch, details in coverage_details.items():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        seal_badge = "✅ SEALED" if details['sealed'] else "❌ NOT SEALED"
                        st.markdown(f"**{ch}** {seal_badge}")
                    
                    with col2:
                        st.metric("Coverage", f"{details['coverage_ratio']*100:.0f}%")
                    
                    with col3:
                        st.metric("Couvertes", f"{details['n_covered']}/{details['n_total']}")
                    
                    with col4:
                        if details['orphans'] > 0:
                            st.error(f"❌ {details['orphans']} orphelins")
                        else:
                            st.success("✅ 0 orphelin")
            else:
                st.info("Aucune donnée de coverage disponible.")
            
            # Audit Log
            st.divider()
            st.markdown("#### 📋 Audit Log")
            
            logs = get_audit_log(st.session_state['v104_results'])
            
            with st.expander("Voir le log complet", expanded=False):
                for log in logs[-20:]:  # Derniers 20 logs
                    if "PASS" in log:
                        st.success(log)
                    elif "FAIL" in log or "BLOCKED" in log:
                        st.error(log)
                    else:
                        st.info(log)
        
        # --- AUDIT INTERNE ---
        st.divider()
        st.markdown("#### ✅ 1. Test Interne (Sujet Traité)")
        
        if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
            fichiers = st.session_state['df_src']["Fichier"].tolist()
            t1_file = st.selectbox("Choisir un sujet traité", fichiers)
            
            if st.button("🔬 AUDIT INTERNE"):
                row = st.session_state['df_src'][st.session_state['df_src']["Fichier"] == t1_file].iloc[0]
                qi_data = row.get("Qi_Data", [])
                
                results = audit_internal_real(qi_data, st.session_state['df_qc'])
                
                if results:
                    matched = sum(1 for r in results if r["Statut"] == "✅ MATCH")
                    coverage = (matched / len(results)) * 100 if results else 0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Couverture", f"{coverage:.0f}%")
                    col2.metric("Qi mappées", f"{matched}/{len(results)}")
                    
                    if coverage >= 100:
                        st.success("✅ 100% de couverture - SUCCÈS")
                    elif coverage >= 80:
                        st.warning(f"⚠️ {coverage:.0f}% de couverture - À améliorer")
                    else:
                        st.error(f"❌ {coverage:.0f}% de couverture - INSUFFISANT")
                    
                    df_results = pd.DataFrame(results)
                    
                    def highlight_status(row):
                        if row['Statut'] == "✅ MATCH":
                            return ['background-color: #dcfce7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(
                        df_results.style.apply(highlight_status, axis=1),
                        use_container_width=True
                    )
                else:
                    st.warning("Aucune Qi trouvée dans ce sujet.")
        
        # --- AUDIT EXTERNE ---
        st.divider()
        st.markdown("#### 🌍 2. Test Externe (Sujet Inconnu)")
        
        uploaded = st.file_uploader("Charger un PDF externe", type="pdf", key="audit_external")
        
        if uploaded:
            if st.button("🔬 AUDIT EXTERNE"):
                pdf_bytes = uploaded.read()
                chapter_filter = st.session_state.get('chapter_filter', sel_chapitres[0] if sel_chapitres else None)
                
                with st.spinner("Analyse du sujet externe..."):
                    coverage, results = audit_external_real(pdf_bytes, st.session_state['df_qc'], chapter_filter)
                
                if results:
                    matched = sum(1 for r in results if r["Statut"] == "✅ MATCH")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Couverture", f"{coverage:.0f}%")
                    col2.metric("Qi couvertes", f"{matched}/{len(results)}")
                    
                    if coverage >= 95:
                        st.success(f"✅ {coverage:.0f}% de couverture - OBJECTIF ATTEINT (≥95%)")
                    elif coverage >= 80:
                        st.warning(f"⚠️ {coverage:.0f}% de couverture - PROCHE DE L'OBJECTIF")
                    else:
                        st.error(f"❌ {coverage:.0f}% de couverture - INSUFFISANT")
                    
                    df_results = pd.DataFrame(results)
                    
                    def highlight_status(row):
                        if row['Statut'] == "✅ MATCH":
                            return ['background-color: #dcfce7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(
                        df_results.style.apply(highlight_status, axis=1),
                        use_container_width=True
                    )
                else:
                    st.warning("Aucune Qi extraite du PDF externe.")
    else:
        st.info("⏳ Lancez d'abord l'usine pour générer des QC.")

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.caption(f"SMAXIA Console V31 | Kernel {KernelConstants.KERNEL_VERSION} ({KernelConstants.KERNEL_STATUS}) | {datetime.now().strftime('%Y-%m-%d')}")
