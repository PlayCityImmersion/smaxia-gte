# smaxia_granulo_engine_v104.py
# =============================================================================
# SMAXIA KERNEL V10.4 — GRANULO ENGINE (GPT-5.2 corrected)
# =============================================================================
# Conformité scellée :
# - POSABLE = corrigé exploitable ∧ scope ∧ évaluable (définition scellée)
# - Attach = AND strict (préconditions ∧ triggers ∧ ARI spine ∧ output type)
# - Sélection coverage-driven : Uncovered, gain marginal, tie-break deterministe
# - Determinism : inputs identiques => outputs identiques (hash(SIG) tie-break)
# - EvidencePack : sources + fingerprints + timestamps + locators + traces
# - Zéro déclaration non prouvée : dérivé de corrigé OU marqué RECONSTRUCTED
#
# Remarque UI :
# - GTE mode : expose un espace "GTE_ACCEPTED" (scope ∧ évaluable) pour démo,
#   mais ne renomme pas cela POSABLE et ne scelle jamais sans corrigé.
# =============================================================================

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

try:
    import pdfplumber  # optional
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

if __name__ == "__main__":
    test_subject = """
    Exercice 1
    1. Démontrer par récurrence que pour tout entier n >= 0, u_n >= 0.
    2. Calculer la limite de la suite (u_n).
    Exercice 2
    1. Déterminer une représentation paramétrique de la droite D.
    2. Calculer la distance du point A au plan P.
    """

    test_correction = """
    Exercice 1
    1. Initialisation: Pour n=0, u_0 = 1 >= 0. OK.
       Hérédité: Supposons u_n >= 0 pour un n.
       Alors u_{n+1} = u_n + 1 >= 0 + 1 >= 0.
       Conclusion: Par récurrence, u_n >= 0 pour tout n.
    2. On calcule u_{n+1} - u_n = 1 > 0.
       Donc la suite est croissante.
       De plus lim u_n = +∞.
    Exercice 2
    1. On détermine le vecteur directeur de D.
       Donc D: x=1+2t, y=2, z=3-t.
    2. On calcule la distance : d(A,P) = 4/sqrt(5).
    """

    res = run_granulo_v104(
        subject_text=test_subject,
        correction_text=test_correction,
        source_id="TEST",
        year_ref=2023,
        extracted_at="2025-12-27T00:00:00Z",
        gte_mode=False,
    )

    print("==== AUDIT LOG ====")
    for l in res["audit_log"]:
        print(l)

    print("\n==== SEALED BY CHAPTER ====")
    print(res["sealed_by_chapter"])

    print("\n==== METRICS ====")
    print(res["metrics"])

    print("\n==== SELECTED QC (head) ====")
    for qc in res["selected_qcs"][:5]:
        print(qc.qc_id, qc.chapter_ref, qc.title, "score=", round(qc.score, 2))
