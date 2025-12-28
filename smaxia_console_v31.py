# smaxia_console_v31.py
# =============================================================================
# SMAXIA - Console V31 (Saturation Proof) — STANDALONE REFACTORED
# =============================================================================
# Version corrigée avec:
# - Extraction PDF layout-aware (plus de mots collés)
# - Atomizer BAC robuste (vraies questions)
# - Pairing multi-niveaux (exact, sequential, fuzzy)
# - Triggers/ARI/FRT de qualité
# - POSABLES > 0 avec corrigés
# =============================================================================

import streamlit as st
import pandas as pd
import hashlib
import json
import re
import io
import time
import difflib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
from enum import Enum
from collections import defaultdict, Counter
from datetime import datetime
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

# =============================================================================
# 1) KERNEL CONSTANTS (INVARIANTS)
# =============================================================================

class KernelConstants:
    KERNEL_VERSION = "V10.4"
    KERNEL_DATE = "2025-12-27"
    KERNEL_STATUS = "SEALED"
    FINGERPRINT_ALGORITHM = "SHA256"
    
    EPSILON = 0.1
    ALPHA_DEFAULT = 5.0
    T_REC_MIN = 0.01
    DELTA_C_DEFAULT = 1.0
    CLUSTER_COHERENCE_THRESHOLD = 0.70
    PAIRING_CONFIDENCE_MIN = 0.50
    SCOPE_CONFIDENCE_MIN = 0.20
    
    MAX_IA1_IA2_ITERATIONS = 3
    ORPHAN_TOLERANCE_THRESHOLD = 0.02
    ORPHAN_TOLERANCE_ABSOLUTE = 2
    SCORE_MIN_VIABLE = 1.0

class ReasonCode(Enum):
    RC_CORRIGE_MISSING = "RC_CORRIGE_MISSING"
    RC_CORRIGE_UNREADABLE = "RC_CORRIGE_UNREADABLE"
    RC_CORRIGE_MISMATCH = "RC_CORRIGE_MISMATCH"
    RC_SCOPE_UNRESOLVED = "RC_SCOPE_UNRESOLVED"
    RC_SCOPE_CONFLICT = "RC_SCOPE_CONFLICT"
    RC_SCOPE_OUTSIDE_PACK = "RC_SCOPE_OUTSIDE_PACK"
    RC_NOT_A_QUESTION = "RC_NOT_A_QUESTION"
    RC_DEPENDENCY_MISSING = "RC_DEPENDENCY_MISSING"
    RC_NON_DETERMINISTIC = "RC_NON_DETERMINISTIC"
    RC_DUPLICATE_ATOM = "RC_DUPLICATE_ATOM"
    RC_EXTRACTION_CORRUPTED = "RC_EXTRACTION_CORRUPTED"
    RC_LANGUAGE_UNSUPPORTED = "RC_LANGUAGE_UNSUPPORTED"
    RC_RESTRICTED_CONTENT = "RC_RESTRICTED_CONTENT"
    RC_HORS_SCOPE = "RC_HORS_SCOPE"
    RC_TEXT_EMPTY = "RC_TEXT_EMPTY"
    ATT_PRECOND_FAIL = "ATT_PRECOND_FAIL"
    ATT_TRIGGER_MISS = "ATT_TRIGGER_MISS"
    ATT_SIGNATURE_MISMATCH = "ATT_SIGNATURE_MISMATCH"
    ATT_NEEDS_EXTRA_STEP = "ATT_NEEDS_EXTRA_STEP"
    ATT_OUTPUT_TYPE_MISMATCH = "ATT_OUTPUT_TYPE_MISMATCH"

# =============================================================================
# 2) PACK CONFIGURATION (FRANCE TERMINALE MATHS)
# =============================================================================

CHAPTER_KEYWORDS = {
    "SUITES NUMÉRIQUES": {
        "keywords": ["suite", "récurrence", "convergence", "terme", "arithmétique", 
                     "géométrique", "limite de suite", "u_n", "v_n", "raison"],
        "weight": 1.0,
        "stopwords": ["suivant", "suivante"]
    },
    "FONCTIONS": {
        "keywords": ["fonction", "dérivée", "dériver", "primitive", "limite", 
                     "continuité", "asymptote", "variation", "extremum", "tangente",
                     "croissante", "décroissante", "maximum", "minimum", "f(x)", "f'(x)"],
        "weight": 1.0,
        "stopwords": []
    },
    "INTÉGRALES": {
        "keywords": ["intégrale", "primitive", "aire", "calcul intégral", 
                     "intégration", "∫", "bornes", "valeur moyenne"],
        "weight": 1.2,
        "stopwords": []
    },
    "PROBABILITÉS": {
        "keywords": ["probabilité", "loi", "espérance", "variance", "binomiale", 
                     "normale", "aléatoire", "événement", "indépendant", "épreuve",
                     "bernoulli", "écart-type", "p(", "conditionnelle"],
        "weight": 1.0,
        "stopwords": []
    },
    "GÉOMÉTRIE DANS L'ESPACE": {
        "keywords": ["espace", "vecteur", "plan", "droite", "orthogonal", 
                     "colinéaire", "paramétrique", "normale", "intersection",
                     "coordonnées", "repère", "parallèle", "perpendiculaire"],
        "weight": 1.0,
        "stopwords": []
    },
    "NOMBRES COMPLEXES": {
        "keywords": ["complexe", "module", "argument", "affixe", "conjugué", 
                     "exponentielle", "trigonométrique", "partie réelle", 
                     "partie imaginaire", "i²", "z", "z̄"],
        "weight": 1.2,
        "stopwords": []
    },
    "MATRICES": {
        "keywords": ["matrice", "déterminant", "inverse", "système", 
                     "vecteur propre", "valeur propre", "diagonale"],
        "weight": 1.0,
        "stopwords": []
    },
    "ARITHMÉTIQUE": {
        "keywords": ["divisibilité", "pgcd", "ppcm", "premier", "congruence", 
                     "bézout", "euclide", "modulo", "diviseur"],
        "weight": 1.0,
        "stopwords": []
    }
}

# Verbes cognitifs pour triggers et évaluation
COGNITIVE_VERBS = {
    "démontrer": 1.0, "prouver": 1.0, "justifier": 0.9,
    "calculer": 0.8, "déterminer": 0.8, "trouver": 0.7,
    "résoudre": 0.9, "vérifier": 0.7, "montrer": 0.9,
    "établir": 0.9, "exprimer": 0.7, "simplifier": 0.6,
    "factoriser": 0.6, "développer": 0.6, "déduire": 0.8,
    "conjecturer": 0.7, "interpréter": 0.7, "représenter": 0.6,
    "tracer": 0.5, "construire": 0.6, "étudier": 0.8,
    "analyser": 0.8, "comparer": 0.7, "identifier": 0.6,
    "énoncer": 0.5, "rappeler": 0.4, "donner": 0.5
}

# Marqueurs de raisonnement pour extraction ARI
REASONING_MARKERS = [
    "on a", "donc", "ainsi", "d'où", "or", "par conséquent",
    "il vient", "ce qui donne", "on en déduit", "cela implique",
    "en effet", "puisque", "car", "comme", "sachant que",
    "il suffit de", "on obtient", "finalement", "en conclusion"
]

# =============================================================================
# 3) PDF EXTRACTION - LAYOUT AWARE (CORRIGE MOTS COLLÉS)
# =============================================================================

def _extract_pdf_text_layout(pdf_bytes: bytes) -> Tuple[str, List[Dict]]:
    """
    Extraction PDF layout-aware pour éviter les mots collés.
    Utilise extract_words() et reconstruit le texte par lignes.
    
    Returns:
        text: Texte complet extrait
        pages_info: Info par page (numéro, lignes, texte)
    """
    if not pdfplumber:
        return "", []
    
    pages_info = []
    all_lines = []
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extraire les mots avec leurs coordonnées
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=True,
                    use_text_flow=True
                )
                
                if not words:
                    # Fallback sur extract_text si extract_words échoue
                    text = page.extract_text() or ""
                    if text:
                        lines = text.split('\n')
                        for line in lines:
                            all_lines.append((page_num, line))
                        pages_info.append({
                            "page": page_num,
                            "lines": lines,
                            "text": text
                        })
                    continue
                
                # Regrouper les mots par ligne (basé sur coordonnée top)
                lines_dict = defaultdict(list)
                for word in words:
                    # Arrondir top pour grouper les mots sur la même ligne
                    line_key = round(word['top'] / 5) * 5
                    lines_dict[line_key].append(word)
                
                # Trier et reconstruire les lignes
                page_lines = []
                for line_key in sorted(lines_dict.keys()):
                    line_words = sorted(lines_dict[line_key], key=lambda w: w['x0'])
                    
                    # Reconstruire la ligne avec espaces appropriés
                    line_text = ""
                    prev_x1 = None
                    for word in line_words:
                        if prev_x1 is not None:
                            # Ajouter espace si gap > seuil
                            gap = word['x0'] - prev_x1
                            if gap > 3:
                                line_text += " "
                        line_text += word['text']
                        prev_x1 = word['x1']
                    
                    if line_text.strip():
                        page_lines.append(line_text.strip())
                        all_lines.append((page_num, line_text.strip()))
                
                pages_info.append({
                    "page": page_num,
                    "lines": page_lines,
                    "text": "\n".join(page_lines)
                })
    
    except Exception as e:
        return f"[EXTRACTION_ERROR: {str(e)}]", []
    
    # Assembler le texte complet
    full_text = "\n".join(line for _, line in all_lines)
    
    return full_text, pages_info


def _sanitize_text(text: str) -> str:
    """
    Nettoie le texte extrait:
    - Répare césures de fin de ligne
    - Normalise espaces
    - Supprime artefacts
    - Corrige mots collés courants
    """
    if not text:
        return ""
    
    # Réparer césures (mot- \n suite → motsuite)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    # Normaliser retours ligne multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normaliser espaces multiples
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Supprimer numéros de page isolés
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    
    # Supprimer lignes avec seulement des tirets ou underscores
    text = re.sub(r'\n[-_]{3,}\n', '\n', text)
    
    # Réparer certains patterns de mots collés courants
    # Pattern: minuscule suivie de majuscule sans espace
    text = re.sub(r'([a-zéèêëàâäùûüôöîïç])([A-ZÉÈÊËÀÂÄÙÛÜÔÖÎÏÇ])', r'\1 \2', text)
    
    # Pattern: chiffre collé à lettre (sauf cas valides comme x1, n2)
    text = re.sub(r'(\d)([A-Za-zéèêëàâäùûüôöîïç]{3,})', r'\1 \2', text)
    
    # Nettoyer espaces en début/fin de ligne
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def _remove_headers_footers(text: str, pages_info: List[Dict]) -> str:
    """
    Supprime les en-têtes et pieds de page répétitifs.
    """
    if not pages_info or len(pages_info) < 2:
        return text
    
    # Collecter les premières et dernières lignes de chaque page
    first_lines = []
    last_lines = []
    
    for p in pages_info:
        lines = p.get('lines', [])
        if lines:
            first_lines.append(lines[0] if lines else "")
            last_lines.append(lines[-1] if lines else "")
    
    # Trouver lignes répétitives (apparaissent dans > 50% des pages)
    threshold = len(pages_info) * 0.5
    
    lines_to_remove = set()
    
    for line in first_lines:
        if first_lines.count(line) > threshold and len(line) < 100:
            lines_to_remove.add(line)
    
    for line in last_lines:
        if last_lines.count(line) > threshold and len(line) < 100:
            lines_to_remove.add(line)
    
    # Supprimer les lignes identifiées
    if lines_to_remove:
        result_lines = []
        for line in text.split('\n'):
            if line.strip() not in lines_to_remove:
                result_lines.append(line)
        return '\n'.join(result_lines)
    
    return text


def extract_pdf_clean(pdf_bytes: bytes) -> Tuple[str, List[Dict], bool]:
    """
    Extraction PDF complète avec nettoyage.
    
    Returns:
        text: Texte nettoyé
        pages_info: Info par page
        is_valid: True si extraction réussie avec contenu
    """
    text, pages_info = _extract_pdf_text_layout(pdf_bytes)
    
    if not text or len(text) < 100:
        return text, pages_info, False
    
    # Nettoyage
    text = _sanitize_text(text)
    text = _remove_headers_footers(text, pages_info)
    
    # Vérifier si c'est du contenu maths (pas un PDF administratif)
    is_maths = _is_maths_content(text)
    
    return text, pages_info, is_maths


def _is_maths_content(text: str) -> bool:
    """
    Vérifie si le texte contient du contenu mathématique.
    """
    text_lower = text.lower()
    
    # Mots-clés maths obligatoires
    maths_indicators = [
        "exercice", "question", "démontrer", "calculer", "fonction",
        "équation", "nombre", "suite", "limite", "dérivée", "intégrale",
        "probabilité", "vecteur", "matrice", "complexe", "géométrie",
        "théorème", "propriété", "formule", "variable", "expression"
    ]
    
    # Patterns maths
    maths_patterns = [
        r'\d+\s*[+\-×÷=<>≤≥]\s*\d+',  # Opérations
        r'[xyz]\s*[+\-=]',  # Variables
        r'f\s*\([^)]+\)',  # Fonctions
        r'\d+\s*°',  # Angles
        r'√',  # Racine
        r'∫|∑|∏|lim',  # Symboles maths
    ]
    
    # Compter les indicateurs
    indicator_count = sum(1 for kw in maths_indicators if kw in text_lower)
    pattern_count = sum(1 for p in maths_patterns if re.search(p, text))
    
    # Seuil: au moins 3 indicateurs OU 2 patterns
    return indicator_count >= 3 or pattern_count >= 2

# =============================================================================
# 4) ATOMIZER BAC - EXTRACTION DES VRAIES QUESTIONS
# =============================================================================

@dataclass
class QuestionLocator:
    """Localisation précise d'une question dans le document."""
    page: int
    exercice: Optional[str] = None
    partie: Optional[str] = None
    question: Optional[str] = None
    sous_question: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    
    def to_key(self) -> str:
        """Génère une clé unique pour appariement."""
        parts = []
        if self.exercice:
            parts.append(f"Ex{self.exercice}")
        if self.partie:
            parts.append(f"P{self.partie}")
        if self.question:
            parts.append(f"Q{self.question}")
        if self.sous_question:
            parts.append(f"q{self.sous_question}")
        return "-".join(parts) if parts else f"page{self.page}"


@dataclass
class AtomizedQuestion:
    """Une question atomisée extraite du document."""
    qi_id: str
    text: str
    locator: QuestionLocator
    source_file: str
    is_evaluable: bool = True
    detected_verbs: List[str] = field(default_factory=list)
    chapter_hint: Optional[str] = None


class BACAtomizer:
    """
    Atomiseur pour sujets de BAC.
    Extrait les vraies questions avec localisation précise.
    """
    
    # Patterns pour détecter les structures
    EXERCICE_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:EXERCICE|Exercice|Ex\.?)\s*([1-9IVX]+)(?:\s*[:\-\.]|\s*\(|(?=\s))',
        re.IGNORECASE | re.MULTILINE
    )
    
    PARTIE_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:PARTIE|Partie|Part\.?)\s*([A-Z1-9IVX]+)(?:\s*[:\-\.]|\s*\(|(?=\s))',
        re.IGNORECASE | re.MULTILINE
    )
    
    QUESTION_PATTERNS = [
        # "1." ou "1)" ou "1 -" ou "1°"
        re.compile(r'(?:^|\n)\s*(\d+)\s*[.\)\-°]\s*(.+?)(?=(?:\n\s*\d+\s*[.\)\-°]|\n\s*[a-z]\s*[.\)]|\Z))', re.DOTALL),
        # "Question 1" ou "Q1"
        re.compile(r'(?:^|\n)\s*(?:Question|Q)\s*(\d+)\s*[:\-\.]?\s*(.+?)(?=(?:\n\s*(?:Question|Q)\s*\d+|\Z))', re.IGNORECASE | re.DOTALL),
    ]
    
    SOUS_QUESTION_PATTERN = re.compile(
        r'(?:^|\n)\s*([a-z])\s*[.\)]\s*(.+?)(?=(?:\n\s*[a-z]\s*[.\)]|\n\s*\d+\s*[.\)]|\Z))',
        re.DOTALL
    )
    
    def __init__(self, source_file: str):
        self.source_file = source_file
        self.questions: List[AtomizedQuestion] = []
        self.current_exercice = None
        self.current_partie = None
        self.current_page = 1
    
    def atomize(self, text: str, pages_info: List[Dict]) -> List[AtomizedQuestion]:
        """
        Extrait toutes les questions du texte.
        """
        self.questions = []
        
        # Construire une map ligne -> page
        line_to_page = self._build_line_page_map(text, pages_info)
        
        # Chercher les exercices
        exercices = list(self.EXERCICE_PATTERN.finditer(text))
        
        if exercices:
            # Traiter chaque exercice
            for i, match in enumerate(exercices):
                self.current_exercice = match.group(1)
                start = match.end()
                end = exercices[i + 1].start() if i + 1 < len(exercices) else len(text)
                exercice_text = text[start:end]
                
                self._extract_from_block(exercice_text, start, line_to_page)
        else:
            # Pas de structure exercice claire, chercher directement les questions
            self._extract_from_block(text, 0, line_to_page)
        
        # Filtrer les questions non évaluables
        self.questions = [q for q in self.questions if q.is_evaluable and len(q.text) > 20]
        
        return self.questions
    
    def _build_line_page_map(self, text: str, pages_info: List[Dict]) -> Dict[int, int]:
        """Construit une map global_line_index -> numéro de page."""
        line_to_page = {}
        global_line = 0
        
        for page_info in pages_info:
            page_num = page_info.get('page', 1)
            lines = page_info.get('lines', []) or []
            
            for _ in lines:
                line_to_page[global_line] = page_num
                global_line += 1
        
        return line_to_page
    
    def _get_page_for_position(self, pos: int, text: str, line_to_page: Dict) -> int:
        """Détermine le numéro de page pour une position donnée."""
        # Compter les lignes jusqu'à cette position
        line_num = text[:pos].count('\n')
        
        # Chercher la page exacte ou la plus proche
        if line_num in line_to_page:
            return line_to_page[line_num]
        
        # Chercher la page la plus proche (inférieure)
        closest_page = 1
        for ln in sorted(line_to_page.keys()):
            if ln <= line_num:
                closest_page = line_to_page[ln]
            else:
                break
        
        return closest_page
    
    def _extract_from_block(self, block_text: str, offset: int, line_to_page: Dict):
        """Extrait les questions d'un bloc (exercice ou texte entier)."""
        
        # Chercher les parties
        parties = list(self.PARTIE_PATTERN.finditer(block_text))
        
        if parties:
            for i, match in enumerate(parties):
                self.current_partie = match.group(1)
                start = match.end()
                end = parties[i + 1].start() if i + 1 < len(parties) else len(block_text)
                partie_text = block_text[start:end]
                
                self._extract_questions(partie_text, offset + start, line_to_page)
        else:
            self._extract_questions(block_text, offset, line_to_page)
    
    def _extract_questions(self, text: str, offset: int, line_to_page: Dict):
        """Extrait les questions numérotées."""
        
        # Essayer les différents patterns
        questions_found = []
        
        for pattern in self.QUESTION_PATTERNS:
            for match in pattern.finditer(text):
                q_num = match.group(1)
                q_text = match.group(2).strip()
                
                if len(q_text) > 15:  # Filtrer les questions trop courtes
                    questions_found.append({
                        'num': q_num,
                        'text': q_text,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Dédupliquer par position
        seen_positions = set()
        unique_questions = []
        for q in sorted(questions_found, key=lambda x: x['start']):
            if q['start'] not in seen_positions:
                unique_questions.append(q)
                seen_positions.add(q['start'])
        
        # Créer les AtomizedQuestion
        for q in unique_questions:
            page = self._get_page_for_position(offset + q['start'], text, line_to_page)
            
            locator = QuestionLocator(
                page=page,
                exercice=self.current_exercice,
                partie=self.current_partie,
                question=q['num'],
                line_start=offset + q['start'],
                line_end=offset + q['end']
            )
            
            # Extraire les sous-questions
            sous_questions = list(self.SOUS_QUESTION_PATTERN.finditer(q['text']))
            
            if sous_questions:
                for sq in sous_questions:
                    sq_locator = QuestionLocator(
                        page=page,
                        exercice=self.current_exercice,
                        partie=self.current_partie,
                        question=q['num'],
                        sous_question=sq.group(1),
                        line_start=locator.line_start,
                        line_end=locator.line_end
                    )
                    
                    sq_text = sq.group(2).strip()
                    if len(sq_text) > 15:
                        self._add_question(sq_text, sq_locator)
            else:
                self._add_question(q['text'], locator)
    
    def _add_question(self, text: str, locator: QuestionLocator):
        """Ajoute une question après validation."""
        
        # Nettoyer le texte
        text = self._clean_question_text(text)
        
        if len(text) < 20:
            return
        
        # Vérifier si c'est évaluable
        is_evaluable, verbs = self._check_evaluable(text)
        
        # Détecter le chapitre
        chapter_hint = self._detect_chapter(text)
        
        # Générer ID unique
        qi_id = f"QI-{hashlib.sha256((self.source_file + locator.to_key() + text[:50]).encode()).hexdigest()[:12]}"
        
        self.questions.append(AtomizedQuestion(
            qi_id=qi_id,
            text=text,
            locator=locator,
            source_file=self.source_file,
            is_evaluable=is_evaluable,
            detected_verbs=verbs,
            chapter_hint=chapter_hint
        ))
    
    def _clean_question_text(self, text: str) -> str:
        """Nettoie le texte d'une question."""
        # Supprimer les numéros de question en début
        text = re.sub(r'^[\d\s]*[.\)]\s*', '', text)
        text = re.sub(r'^[a-z]\s*[.\)]\s*', '', text, flags=re.IGNORECASE)
        
        # Normaliser espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _check_evaluable(self, text: str) -> Tuple[bool, List[str]]:
        """Vérifie si la question est évaluable."""
        text_lower = text.lower()
        
        found_verbs = []
        for verb in COGNITIVE_VERBS:
            if verb in text_lower:
                found_verbs.append(verb)
        
        # Question évaluable si: verbe cognitif OU "?" OU longueur suffisante
        is_evaluable = len(found_verbs) > 0 or "?" in text or len(text) > 50
        
        return is_evaluable, found_verbs
    
    def _detect_chapter(self, text: str) -> Optional[str]:
        """Détecte le chapitre probable de la question."""
        text_lower = text.lower()
        
        scores = {}
        for chapter, config in CHAPTER_KEYWORDS.items():
            score = 0
            for kw in config['keywords']:
                if kw.lower() in text_lower:
                    score += config['weight']
            
            # Pénaliser les stopwords
            for sw in config.get('stopwords', []):
                if sw.lower() in text_lower:
                    score -= 0.5
            
            if score > 0:
                scores[chapter] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return None

# =============================================================================
# 5) PAIRING MULTI-NIVEAUX (SUJET ↔ CORRIGÉ)
# =============================================================================

@dataclass
class PairingResult:
    """Résultat d'un appariement Qi ↔ RQi."""
    qi_id: str
    rqi_id: Optional[str]
    rqi_text: Optional[str]
    method: str  # "exact", "sequential", "fuzzy", "none"
    confidence: float
    locator_match: bool


class MultiLevelPairer:
    """
    Appariement multi-niveaux entre questions et corrections.
    Niveau 1: Clé exacte (exercice/partie/question)
    Niveau 2: Séquentiel par exercice
    Niveau 3: Fuzzy matching textuel
    """
    
    def __init__(self):
        self.results: List[PairingResult] = []
        self.stats = {
            "total": 0,
            "exact": 0,
            "sequential": 0,
            "fuzzy": 0,
            "none": 0
        }
    
    def pair(self, 
             questions: List[AtomizedQuestion], 
             corrections: List[AtomizedQuestion]) -> List[PairingResult]:
        """
        Apparie les questions avec les corrections.
        """
        self.results = []
        self.stats = {"total": len(questions), "exact": 0, "sequential": 0, "fuzzy": 0, "none": 0}
        
        # Index des corrections par clé de locator
        corr_by_key = {}
        for c in corrections:
            key = c.locator.to_key()
            if key not in corr_by_key:
                corr_by_key[key] = []
            corr_by_key[key].append(c)
        
        # Index des corrections par exercice (pour matching séquentiel)
        corr_by_exercice = defaultdict(list)
        for c in corrections:
            ex_key = c.locator.exercice or "default"
            corr_by_exercice[ex_key].append(c)
        
        used_corrections = set()
        
        for q in questions:
            result = self._pair_single(q, corr_by_key, corr_by_exercice, corrections, used_corrections)
            self.results.append(result)
            
            if result.rqi_id:
                used_corrections.add(result.rqi_id)
        
        return self.results
    
    def _pair_single(self, 
                     q: AtomizedQuestion,
                     corr_by_key: Dict,
                     corr_by_exercice: Dict,
                     all_corrections: List[AtomizedQuestion],
                     used: Set[str]) -> PairingResult:
        """Apparie une seule question."""
        
        # NIVEAU 1: Clé exacte
        key = q.locator.to_key()
        if key in corr_by_key:
            for c in corr_by_key[key]:
                if c.qi_id not in used:
                    self.stats["exact"] += 1
                    return PairingResult(
                        qi_id=q.qi_id,
                        rqi_id=c.qi_id,
                        rqi_text=c.text,
                        method="exact",
                        confidence=0.95,
                        locator_match=True
                    )
        
        # NIVEAU 2: Séquentiel par exercice
        ex_key = q.locator.exercice or "default"
        if ex_key in corr_by_exercice:
            # Chercher la correction avec le même numéro de question
            for c in corr_by_exercice[ex_key]:
                if c.qi_id not in used and c.locator.question == q.locator.question:
                    self.stats["sequential"] += 1
                    return PairingResult(
                        qi_id=q.qi_id,
                        rqi_id=c.qi_id,
                        rqi_text=c.text,
                        method="sequential",
                        confidence=0.80,
                        locator_match=False
                    )
        
        # NIVEAU 3: Fuzzy matching
        best_match = None
        best_ratio = 0.0
        
        for c in all_corrections:
            if c.qi_id in used:
                continue
            
            # Calculer similarité
            ratio = difflib.SequenceMatcher(None, q.text[:200], c.text[:200]).ratio()
            
            if ratio > best_ratio and ratio > 0.4:
                best_ratio = ratio
                best_match = c
        
        if best_match and best_ratio > 0.5:
            self.stats["fuzzy"] += 1
            return PairingResult(
                qi_id=q.qi_id,
                rqi_id=best_match.qi_id,
                rqi_text=best_match.text,
                method="fuzzy",
                confidence=best_ratio,
                locator_match=False
            )
        
        # Aucun match
        self.stats["none"] += 1
        return PairingResult(
            qi_id=q.qi_id,
            rqi_id=None,
            rqi_text=None,
            method="none",
            confidence=0.0,
            locator_match=False
        )
    
    def get_pairing_rate(self) -> float:
        """Retourne le taux d'appariement."""
        if self.stats["total"] == 0:
            return 0.0
        paired = self.stats["exact"] + self.stats["sequential"] + self.stats["fuzzy"]
        return paired / self.stats["total"]


# =============================================================================
# 6) TRIGGERS / ARI / FRT - QUALITÉ
# =============================================================================

@dataclass
class FRTBlock:
    """Un bloc de la Fiche Réponse Type."""
    type: str  # usage, method, trap, conc
    title: str
    text: str


@dataclass
class QualityFRT:
    """FRT de qualité extraite des corrigés."""
    triggers: List[str]
    preconditions: List[str]
    ari_steps: List[str]
    erreurs_frequentes: List[str]
    oublis_fatals: List[str]
    phrase_conclusion: str
    format_reponse: str
    source: str  # "CORRIGE" ou "RECONSTRUCTED"
    
    def to_blocks(self) -> List[Dict]:
        """Convertit en format V31."""
        return [
            {
                "type": "usage",
                "title": "📌 QUAND UTILISER",
                "text": f"Triggers: {', '.join(self.triggers[:5])}\nPréconditions: {', '.join(self.preconditions[:3])}"
            },
            {
                "type": "method",
                "title": "✅ MÉTHODE",
                "text": self.format_reponse or "\n".join(self.ari_steps[:5])
            },
            {
                "type": "trap",
                "title": "⚠️ PIÈGES",
                "text": "\n".join(["• " + e for e in (self.erreurs_frequentes + self.oublis_fatals)[:4]]) or "Vérifier les conditions d'application"
            },
            {
                "type": "conc",
                "title": "🎯 CONCLUSION",
                "text": self.phrase_conclusion or "Conclure en répondant précisément à la question posée"
            }
        ]


class ARIExtractor:
    """
    Extracteur d'ARI (Algorithme de Résolution Invariant) depuis les corrigés.
    """
    
    def extract_from_correction(self, correction_text: str) -> List[str]:
        """
        Extrait les étapes de résolution depuis un corrigé.
        """
        if not correction_text:
            return []
        
        steps = []
        
        # Chercher les marqueurs de raisonnement
        text_lower = correction_text.lower()
        
        # Découper par marqueurs
        for marker in REASONING_MARKERS:
            pattern = re.compile(
                rf'(?:^|\.\s*){marker}\s*[,:]?\s*([^.]+\.)',
                re.IGNORECASE | re.MULTILINE
            )
            for match in pattern.finditer(correction_text):
                step = match.group(1).strip()
                if 10 < len(step) < 200:
                    # Identifier le verbe principal
                    verb = self._identify_verb(step)
                    if verb:
                        steps.append(f"{verb.upper()}: {step}")
                    else:
                        steps.append(step)
        
        # Si pas assez de marqueurs, découper par phrases
        if len(steps) < 3:
            sentences = re.split(r'[.!?]\s+', correction_text)
            for sent in sentences[:10]:
                sent = sent.strip()
                if 20 < len(sent) < 200:
                    verb = self._identify_verb(sent)
                    if verb:
                        steps.append(f"{verb.upper()}: {sent}")
        
        # Dédupliquer et limiter
        seen = set()
        unique_steps = []
        for s in steps:
            s_clean = s[:50].lower()
            if s_clean not in seen:
                seen.add(s_clean)
                unique_steps.append(s)
        
        return unique_steps[:10]
    
    def _identify_verb(self, text: str) -> Optional[str]:
        """Identifie le verbe cognitif principal."""
        text_lower = text.lower()
        
        for verb in COGNITIVE_VERBS:
            if verb in text_lower:
                return verb
        
        return None


class TriggerExtractor:
    """
    Extracteur de triggers de qualité.
    """
    
    def extract_triggers(self, question_text: str, correction_text: Optional[str] = None) -> List[str]:
        """
        Extrait des triggers propres (pas de tokens bruités).
        """
        triggers = []
        
        # 1) Verbes cognitifs de la question
        text_lower = question_text.lower()
        for verb, weight in COGNITIVE_VERBS.items():
            if verb in text_lower and weight >= 0.6:
                triggers.append(verb)
        
        # 2) Termes mathématiques pertinents
        maths_terms = self._extract_maths_terms(question_text)
        triggers.extend(maths_terms)
        
        # 3) Si corrigé disponible, enrichir
        if correction_text:
            corr_terms = self._extract_maths_terms(correction_text)
            for t in corr_terms:
                if t not in triggers:
                    triggers.append(t)
        
        # Filtrer et nettoyer
        triggers = self._clean_triggers(triggers)
        
        return triggers[:7]
    
    def _extract_maths_terms(self, text: str) -> List[str]:
        """Extrait les termes mathématiques pertinents."""
        terms = []
        text_lower = text.lower()
        
        # Termes des chapitres
        for chapter, config in CHAPTER_KEYWORDS.items():
            for kw in config['keywords']:
                if kw.lower() in text_lower:
                    terms.append(kw)
        
        return terms
    
    def _clean_triggers(self, triggers: List[str]) -> List[str]:
        """Nettoie les triggers."""
        cleaned = []
        seen = set()
        
        for t in triggers:
            t = t.strip().lower()
            
            # Filtrer
            if len(t) < 3:
                continue
            if t.isdigit():
                continue
            if not re.match(r'^[a-zéèêëàâäùûüôöîïç\'\-\s]+$', t):
                continue
            if t in seen:
                continue
            
            # Pas de mots collés (contient plus de 20 chars sans espace)
            if len(t) > 20 and ' ' not in t:
                continue
            
            seen.add(t)
            cleaned.append(t)
        
        return cleaned


class FRTBuilder:
    """
    Constructeur de FRT de qualité.
    """
    
    def __init__(self):
        self.ari_extractor = ARIExtractor()
        self.trigger_extractor = TriggerExtractor()
    
    def build_frt(self, 
                  questions: List[AtomizedQuestion],
                  pairings: List[PairingResult],
                  chapter: str) -> QualityFRT:
        """
        Construit une FRT de qualité pour un cluster de questions.
        """
        
        # Collecter les textes
        question_texts = [q.text for q in questions]
        correction_texts = [p.rqi_text for p in pairings if p.rqi_text]
        
        has_corrections = len(correction_texts) > 0
        
        # Extraire triggers
        all_triggers = []
        for q in questions:
            q_triggers = self.trigger_extractor.extract_triggers(q.text)
            all_triggers.extend(q_triggers)
        
        if correction_texts:
            for ct in correction_texts[:3]:
                c_triggers = self.trigger_extractor.extract_triggers("", ct)
                all_triggers.extend(c_triggers)
        
        # Compter et trier les triggers
        trigger_counts = Counter(all_triggers)
        top_triggers = [t for t, _ in trigger_counts.most_common(7)]
        
        # Extraire ARI
        ari_steps = []
        if correction_texts:
            for ct in correction_texts[:2]:
                steps = self.ari_extractor.extract_from_correction(ct)
                ari_steps.extend(steps)
        
        if not ari_steps:
            # Fallback: générer des étapes minimales basées sur les verbes
            verbs = set()
            for q in questions:
                verbs.update(q.detected_verbs)
            
            for verb in list(verbs)[:3]:
                ari_steps.append(f"{verb.upper()}: Appliquer la méthode appropriée")
        
        # Préconditions basées sur le chapitre
        preconditions = self._get_chapter_preconditions(chapter)
        
        # Erreurs fréquentes
        erreurs = self._get_common_errors(chapter)
        
        # Phrase de conclusion
        conclusion = self._get_conclusion_template(chapter)
        
        # Format réponse
        format_reponse = self._get_format_template(chapter, questions)
        
        return QualityFRT(
            triggers=top_triggers,
            preconditions=preconditions,
            ari_steps=ari_steps[:8],
            erreurs_frequentes=erreurs[:3],
            oublis_fatals=["Vérifier les conditions initiales", "Justifier chaque étape"],
            phrase_conclusion=conclusion,
            format_reponse=format_reponse,
            source="CORRIGE" if has_corrections else "RECONSTRUCTED"
        )
    
    def _get_chapter_preconditions(self, chapter: str) -> List[str]:
        """Retourne les préconditions typiques du chapitre."""
        preconds = {
            "SUITES NUMÉRIQUES": ["Suite définie", "Terme initial donné", "Relation de récurrence"],
            "FONCTIONS": ["Fonction définie sur un intervalle", "Dérivabilité requise"],
            "INTÉGRALES": ["Fonction continue", "Bornes d'intégration définies"],
            "PROBABILITÉS": ["Événements définis", "Loi de probabilité connue"],
            "GÉOMÉTRIE DANS L'ESPACE": ["Repère orthonormé", "Points/vecteurs donnés"],
            "NOMBRES COMPLEXES": ["Forme algébrique/trigonométrique", "Module et argument"],
            "MATRICES": ["Matrice carrée", "Dimensions compatibles"],
            "ARITHMÉTIQUE": ["Entiers naturels", "Propriétés de divisibilité"]
        }
        return preconds.get(chapter, ["Données de l'énoncé (abstrait)"])
    
    def _get_common_errors(self, chapter: str) -> List[str]:
        """Retourne les erreurs fréquentes du chapitre."""
        errors = {
            "SUITES NUMÉRIQUES": ["Oubli de l'initialisation dans la récurrence", "Confusion sens de variation/limite"],
            "FONCTIONS": ["Erreur de signe dans la dérivée", "Oubli du domaine de définition"],
            "INTÉGRALES": ["Inversion des bornes", "Oubli de la primitive"],
            "PROBABILITÉS": ["Confusion P(A∩B) et P(A)×P(B)", "Oubli de vérifier l'indépendance"],
            "GÉOMÉTRIE DANS L'ESPACE": ["Erreur de calcul vectoriel", "Confusion parallèle/perpendiculaire"],
            "NOMBRES COMPLEXES": ["Erreur sur le module", "Confusion argument/angle"],
            "MATRICES": ["Non-vérification de l'inversibilité", "Erreur d'ordre de multiplication"],
            "ARITHMÉTIQUE": ["Oubli des cas particuliers", "Erreur dans l'algorithme d'Euclide"]
        }
        return errors.get(chapter, ["Vérifier les calculs", "Justifier les étapes"])
    
    def _get_conclusion_template(self, chapter: str) -> str:
        """Retourne le template de conclusion pour le chapitre."""
        templates = {
            "SUITES NUMÉRIQUES": "La suite converge vers... / La suite est...",
            "FONCTIONS": "La fonction admet... / Sur l'intervalle...",
            "INTÉGRALES": "L'aire/valeur de l'intégrale est...",
            "PROBABILITÉS": "La probabilité est... / L'espérance vaut...",
            "GÉOMÉTRIE DANS L'ESPACE": "Le point/plan/droite a pour...",
            "NOMBRES COMPLEXES": "L'affixe/module/argument est...",
            "MATRICES": "La matrice inverse/valeur propre est...",
            "ARITHMÉTIQUE": "PGCD = ... / L'entier vérifie..."
        }
        return templates.get(chapter, "Conclure en répondant à la question")
    
    def _get_format_template(self, chapter: str, questions: List[AtomizedQuestion]) -> str:
        """Génère le template de format de réponse."""
        # Détecter le type de sortie attendu
        all_text = " ".join(q.text for q in questions).lower()
        
        if "démontrer" in all_text or "prouver" in all_text:
            return "Rédiger une démonstration structurée avec hypothèses → raisonnement → conclusion"
        elif "calculer" in all_text:
            return "Présenter le calcul étape par étape et encadrer le résultat"
        elif "déterminer" in all_text:
            return "Identifier la méthode, appliquer, et conclure avec le résultat"
        elif "tracer" in all_text or "représenter" in all_text:
            return "Construire le graphique avec les éléments caractéristiques"
        else:
            return "Répondre de manière structurée avec justifications"


# =============================================================================
# 7) CORE ENGINE - QC GENERATION & COVERAGE
# =============================================================================

@dataclass
class Signature:
    """Signature invariante d'une QC."""
    action_spine: List[str]
    preconditions: List[str]
    output_type: str
    checkpoints: List[str]
    
    def to_hash(self) -> str:
        data = {
            "A": sorted(self.action_spine),
            "P": sorted(self.preconditions),
            "O": self.output_type,
            "X": sorted(self.checkpoints)
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class QCCandidate:
    """Question Clé candidate."""
    qc_id: str
    title: str
    chapter_ref: str
    triggers: List[str]
    ari: List[str]
    frt: QualityFRT
    signature: Signature
    n_q: int
    n_total: int
    t_rec: float
    psi_normalized: float
    score: float
    source_qi_ids: List[str]
    has_correction: bool


@dataclass 
class CoverageResult:
    """Résultat de coverage pour un chapitre."""
    chapter: str
    total_posable: int
    covered: int
    orphans: List[str]
    coverage_ratio: float
    is_sealed: bool


class GranuloEngine:
    """
    Moteur Granulo V10.5 - Génération de QC avec qualité.
    """
    
    def __init__(self, gte_mode: bool = True):
        self.gte_mode = gte_mode
        self.frt_builder = FRTBuilder()
        self.pairer = MultiLevelPairer()
        
        self.questions: List[AtomizedQuestion] = []
        self.corrections: List[AtomizedQuestion] = []
        self.pairings: List[PairingResult] = []
        self.qc_candidates: List[QCCandidate] = []
        self.coverage_results: Dict[str, CoverageResult] = {}
        
        self.audit_log: List[str] = []
        self.stats = {
            "total_qi": 0,
            "total_posable": 0,
            "total_qc": 0,
            "pairing_rate": 0.0
        }
    
    def process(self,
                subject_texts: List[Tuple[str, str, List[Dict]]],  # (filename, text, pages_info)
                correction_texts: List[Tuple[str, str, List[Dict]]]) -> Dict[str, Any]:
        """
        Traite les sujets et corrigés.
        
        Args:
            subject_texts: Liste de (filename, text, pages_info) pour les sujets
            correction_texts: Liste de (filename, text, pages_info) pour les corrigés
        
        Returns:
            Résultats complets du traitement
        """
        
        self._log("=== DÉMARRAGE TRAITEMENT GRANULO V10.5 ===")
        
        # 1) Atomisation des sujets
        self._log(f"Atomisation de {len(subject_texts)} sujets...")
        for filename, text, pages_info in subject_texts:
            atomizer = BACAtomizer(filename)
            questions = atomizer.atomize(text, pages_info)
            self.questions.extend(questions)
            self._log(f"  → {filename}: {len(questions)} Qi extraites")
        
        self.stats["total_qi"] = len(self.questions)
        self._log(f"Total Qi: {self.stats['total_qi']}")
        
        # 2) Atomisation des corrigés
        if correction_texts:
            self._log(f"Atomisation de {len(correction_texts)} corrigés...")
            for filename, text, pages_info in correction_texts:
                atomizer = BACAtomizer(filename)
                corrections = atomizer.atomize(text, pages_info)
                self.corrections.extend(corrections)
                self._log(f"  → {filename}: {len(corrections)} RQi extraites")
        
        # 3) Appariement
        if self.corrections:
            self._log("Appariement Qi ↔ RQi...")
            self.pairings = self.pairer.pair(self.questions, self.corrections)
            self.stats["pairing_rate"] = self.pairer.get_pairing_rate()
            self._log(f"Taux d'appariement: {self.stats['pairing_rate']*100:.1f}%")
            self._log(f"  - Exact: {self.pairer.stats['exact']}")
            self._log(f"  - Sequential: {self.pairer.stats['sequential']}")
            self._log(f"  - Fuzzy: {self.pairer.stats['fuzzy']}")
            self._log(f"  - None: {self.pairer.stats['none']}")
        else:
            # Mode GTE - pas d'appariement
            self.pairings = [
                PairingResult(q.qi_id, None, None, "none", 0.0, False)
                for q in self.questions
            ]
        
        # 4) Classification par chapitre
        self._log("Classification par chapitre...")
        questions_by_chapter = self._classify_by_chapter()
        
        # 5) Calcul POSABLES
        self._log("Calcul des POSABLES...")
        posables_by_chapter = self._compute_posables(questions_by_chapter)
        self.stats["total_posable"] = sum(len(qs) for qs in posables_by_chapter.values())
        self._log(f"Total POSABLES: {self.stats['total_posable']}")
        
        # 6) Génération des QC
        self._log("Génération des QC...")
        self._generate_qcs(posables_by_chapter)
        self.stats["total_qc"] = len(self.qc_candidates)
        self._log(f"Total QC générées: {self.stats['total_qc']}")
        
        # 7) Calcul coverage
        self._log("Calcul de la coverage...")
        self._compute_coverage(posables_by_chapter)
        
        return self._build_results()
    
    def _log(self, message: str):
        """Ajoute un message au log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.audit_log.append(f"[{timestamp}] {message}")
    
    def _classify_by_chapter(self) -> Dict[str, List[AtomizedQuestion]]:
        """Classifie les questions par chapitre."""
        by_chapter = defaultdict(list)
        unclassified = 0
        
        for q in self.questions:
            chapter = q.chapter_hint
            
            if not chapter:
                # Essayer de détecter
                chapter = self._detect_chapter_advanced(q.text)
            
            if chapter:
                by_chapter[chapter].append(q)
            else:
                by_chapter["NON_CLASSIFIÉ"].append(q)
                unclassified += 1
        
        self._log(f"Classification: {len(by_chapter)} chapitres, {unclassified} non classifiés")
        
        for ch, qs in by_chapter.items():
            self._log(f"  → {ch}: {len(qs)} Qi")
        
        return dict(by_chapter)
    
    def _detect_chapter_advanced(self, text: str) -> Optional[str]:
        """Détection avancée du chapitre."""
        text_lower = text.lower()
        
        scores = {}
        for chapter, config in CHAPTER_KEYWORDS.items():
            score = 0
            
            # Mots-clés avec poids
            for kw in config['keywords']:
                if kw.lower() in text_lower:
                    score += config['weight']
                    # Bonus si mot-clé apparaît plusieurs fois
                    count = text_lower.count(kw.lower())
                    if count > 1:
                        score += 0.2 * (count - 1)
            
            # Pénaliser stopwords
            for sw in config.get('stopwords', []):
                if sw.lower() in text_lower:
                    score -= 0.3
            
            if score > 0:
                scores[chapter] = score
        
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] >= KernelConstants.SCOPE_CONFIDENCE_MIN:
                return best
        
        return None
    
    def _compute_posables(self, questions_by_chapter: Dict[str, List[AtomizedQuestion]]) -> Dict[str, List[AtomizedQuestion]]:
        """
        Calcule les questions POSABLES par chapitre.
        
        POSABLE = corrigé_exploitable ∧ scope_rattachable ∧ évaluable
        """
        posables = {}
        
        # Créer un index des pairings
        pairing_by_qi = {p.qi_id: p for p in self.pairings}
        
        for chapter, questions in questions_by_chapter.items():
            if chapter == "NON_CLASSIFIÉ":
                continue
            
            chapter_posables = []
            
            for q in questions:
                pairing = pairing_by_qi.get(q.qi_id)
                
                # En mode GTE, on accepte sans corrigé
                if self.gte_mode:
                    if q.is_evaluable:
                        chapter_posables.append(q)
                else:
                    # Mode Kernel: besoin de corrigé
                    if pairing and pairing.rqi_text and pairing.confidence >= KernelConstants.PAIRING_CONFIDENCE_MIN:
                        if q.is_evaluable:
                            chapter_posables.append(q)
            
            if chapter_posables:
                posables[chapter] = chapter_posables
        
        return posables
    
    def _generate_qcs(self, posables_by_chapter: Dict[str, List[AtomizedQuestion]]):
        """Génère les QC pour chaque chapitre."""
        
        pairing_by_qi = {p.qi_id: p for p in self.pairings}
        
        for chapter, questions in posables_by_chapter.items():
            if not questions:
                continue
            
            # Clustering simple: regrouper par verbes similaires
            clusters = self._cluster_questions(questions)
            
            for cluster_questions in clusters:
                if not cluster_questions:
                    continue
                
                # Collecter les pairings du cluster
                cluster_pairings = [
                    pairing_by_qi.get(q.qi_id)
                    for q in cluster_questions
                    if q.qi_id in pairing_by_qi
                ]
                cluster_pairings = [p for p in cluster_pairings if p]
                
                # Construire la FRT
                frt = self.frt_builder.build_frt(cluster_questions, cluster_pairings, chapter)
                
                # Construire la signature
                action_spine = list(set(v for q in cluster_questions for v in q.detected_verbs))[:5]
                signature = Signature(
                    action_spine=action_spine or ["résoudre"],
                    preconditions=frt.preconditions,
                    output_type=self._detect_output_type(cluster_questions),
                    checkpoints=frt.ari_steps[:3]
                )
                
                # Générer l'ID
                sig_hash = signature.to_hash()
                qc_id = f"QC-{sig_hash}-{chapter[:3].upper()}"
                
                # Calculer les métriques
                n_q = len(cluster_questions)
                n_total = len(questions)
                t_rec = 6.0  # Valeur par défaut (années)
                
                # Score F2 simplifié
                psi = n_q / max(n_total, 1)
                score = (n_q * KernelConstants.ALPHA_DEFAULT + psi * 100) / (t_rec + KernelConstants.EPSILON)
                
                # Titre
                main_verb = action_spine[0] if action_spine else "résoudre"
                title = f"Comment {main_verb} un problème de type {chapter} ?"
                
                # Vérifier si a des corrections
                has_correction = any(p.rqi_text for p in cluster_pairings)
                
                qc = QCCandidate(
                    qc_id=qc_id,
                    title=title,
                    chapter_ref=chapter,
                    triggers=frt.triggers,
                    ari=frt.ari_steps,
                    frt=frt,
                    signature=signature,
                    n_q=n_q,
                    n_total=n_total,
                    t_rec=t_rec,
                    psi_normalized=psi,
                    score=score,
                    source_qi_ids=[q.qi_id for q in cluster_questions],
                    has_correction=has_correction
                )
                
                self.qc_candidates.append(qc)
    
    def _cluster_questions(self, questions: List[AtomizedQuestion]) -> List[List[AtomizedQuestion]]:
        """
        Clustering simple des questions par similarité de verbes/contenu.
        """
        if len(questions) <= 3:
            return [questions]
        
        # Regrouper par verbe principal
        by_verb = defaultdict(list)
        for q in questions:
            main_verb = q.detected_verbs[0] if q.detected_verbs else "autre"
            by_verb[main_verb].append(q)
        
        clusters = list(by_verb.values())
        
        # Fusionner les petits clusters
        result = []
        small_cluster = []
        
        for cluster in clusters:
            if len(cluster) >= 2:
                result.append(cluster)
            else:
                small_cluster.extend(cluster)
        
        if small_cluster:
            result.append(small_cluster)
        
        return result
    
    def _detect_output_type(self, questions: List[AtomizedQuestion]) -> str:
        """Détecte le type de sortie attendu."""
        all_text = " ".join(q.text for q in questions).lower()
        
        if "démontrer" in all_text or "prouver" in all_text:
            return "PROOF"
        elif "calculer" in all_text:
            return "VALUE"
        elif "tracer" in all_text:
            return "GRAPH"
        elif "tableau" in all_text:
            return "TABLE"
        else:
            return "RESULT"
    
    def _compute_coverage(self, posables_by_chapter: Dict[str, List[AtomizedQuestion]]):
        """Calcule la coverage par chapitre."""
        
        for chapter, posables in posables_by_chapter.items():
            if not posables:
                continue
            
            posable_ids = {q.qi_id for q in posables}
            
            # Trouver les QC du chapitre
            chapter_qcs = [qc for qc in self.qc_candidates if qc.chapter_ref == chapter]
            
            # IDs couverts par les QC
            covered_ids = set()
            for qc in chapter_qcs:
                covered_ids.update(qc.source_qi_ids)
            
            # Calculer orphelins
            orphans = posable_ids - covered_ids
            
            coverage_ratio = len(covered_ids) / len(posable_ids) if posable_ids else 0
            
            # Scellé si 0 orphelin et mode non-GTE
            is_sealed = len(orphans) == 0 and not self.gte_mode
            
            self.coverage_results[chapter] = CoverageResult(
                chapter=chapter,
                total_posable=len(posables),
                covered=len(covered_ids),
                orphans=list(orphans),
                coverage_ratio=coverage_ratio,
                is_sealed=is_sealed
            )
            
            self._log(f"Coverage {chapter}: {coverage_ratio*100:.1f}% ({len(covered_ids)}/{len(posables)}), orphelins: {len(orphans)}")
    
    def _build_results(self) -> Dict[str, Any]:
        """Construit le dictionnaire de résultats."""
        return {
            "questions": self.questions,
            "corrections": self.corrections,
            "pairings": self.pairings,
            "qc_candidates": self.qc_candidates,
            "coverage_results": self.coverage_results,
            "stats": self.stats,
            "audit_log": self.audit_log,
            "pairing_stats": self.pairer.stats if self.corrections else None
        }


# =============================================================================
# 8) WEB SCRAPING & INGESTION
# =============================================================================

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
REQ_TIMEOUT = 20
MAX_PAGES_SCAN = 100
MAX_PDF_MB = 30
SLEEP_BETWEEN = 0.2

def _safe_get(url: str) -> Optional[Any]:
    if not requests:
        return None
    try:
        return requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT, allow_redirects=True)
    except:
        return None

def _is_pdf_link(url: str) -> bool:
    u = (url or "").lower()
    return ".pdf" in u

def _filename_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
        fn = path.split("/")[-1] or "document.pdf"
        if not fn.lower().endswith(".pdf"):
            fn += ".pdf"
        return fn
    except:
        return "document.pdf"

def _download_pdf(url: str) -> Optional[bytes]:
    r = _safe_get(url)
    if not r or r.status_code != 200:
        return None
    content = r.content or b""
    if len(content) > MAX_PDF_MB * 1024 * 1024:
        return None
    if b"%PDF" not in content[:1024]:
        return None
    return content

def crawl_pdf_links(seed_urls: List[str], target_count: int, progress_cb=None) -> List[str]:
    """Crawl des URLs pour trouver des liens PDF."""
    if not requests or not BeautifulSoup:
        return []
    
    seed_urls = [u.strip() for u in seed_urls if u.strip()]
    seed_urls = [u if u.startswith("http") else "https://" + u for u in seed_urls]
    
    visited = set()
    pdfs = []
    queue = list(seed_urls)
    pages_scanned = 0
    
    while queue and pages_scanned < MAX_PAGES_SCAN and len(pdfs) < target_count * 3:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        pages_scanned += 1
        
        if progress_cb:
            progress_cb(0.05 + 0.25 * min(1.0, pages_scanned / MAX_PAGES_SCAN))
        
        r = _safe_get(url)
        if not r or r.status_code != 200:
            continue
        
        ctype = (r.headers.get("content-type") or "").lower()
        if "pdf" in ctype:
            if url not in pdfs:
                pdfs.append(url)
            continue
        
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "").strip()
                if not href:
                    continue
                
                full_url = urljoin(url, href)
                
                if _is_pdf_link(full_url):
                    if full_url not in pdfs:
                        pdfs.append(full_url)
                elif full_url not in visited:
                    # Rester sur le même domaine
                    try:
                        if urlparse(full_url).netloc == urlparse(url).netloc:
                            queue.append(full_url)
                    except:
                        pass
        except:
            pass
        
        time.sleep(SLEEP_BETWEEN)
    
    return pdfs


def ingest_from_web(url_list: List[str], volume: int, matiere: str, 
                    chapter_filter: Optional[str], gte_mode: bool,
                    progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Ingestion depuis le web: crawl + download + traitement.
    """
    
    empty_result = {
        "df_src": pd.DataFrame(),
        "questions": [],
        "qc_candidates": [],
        "coverage_results": {},
        "stats": {"total_qi": 0, "total_posable": 0, "total_qc": 0},
        "audit_log": ["Aucun PDF trouvé"]
    }
    
    if progress_cb:
        progress_cb(0.02)
    
    # 1) Crawler
    pdf_links = crawl_pdf_links(url_list, volume, progress_cb)
    
    if not pdf_links:
        return empty_result
    
    # 2) Séparer sujets/corrigés
    corr_words = ("corrig", "correction", "solution")
    subjects = []
    corrections = []
    
    for url in pdf_links:
        low = url.lower()
        if any(w in low for w in corr_words):
            corrections.append(url)
        else:
            subjects.append(url)
    
    subjects = subjects[:volume]
    
    if progress_cb:
        progress_cb(0.35)
    
    # 3) Télécharger
    downloaded_subjects = []
    downloaded_corrections = []
    
    with ThreadPoolExecutor(max_workers=6) as ex:
        def dl(u):
            return (u, _download_pdf(u))
        
        futs = {ex.submit(dl, u): u for u in subjects}
        done = 0
        for f in as_completed(futs):
            done += 1
            u, b = f.result()
            if b:
                downloaded_subjects.append((_filename_from_url(u), b))
            if progress_cb:
                progress_cb(0.35 + 0.30 * done / max(1, len(subjects)))
    
    # Télécharger quelques corrigés
    corr_pool = corrections[:min(10, volume * 2)]
    if corr_pool:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(dl, u): u for u in corr_pool}
            done = 0
            for f in as_completed(futs):
                done += 1
                u, b = f.result()
                if b:
                    downloaded_corrections.append((_filename_from_url(u), b))
                if progress_cb:
                    progress_cb(0.65 + 0.15 * done / max(1, len(corr_pool)))
    
    if not downloaded_subjects:
        return empty_result
    
    # 4) Traiter avec le moteur
    return process_pdfs(downloaded_subjects, downloaded_corrections, matiere, chapter_filter, gte_mode, progress_cb)


def process_pdfs(subject_files: List[Tuple[str, bytes]], 
                 correction_files: Optional[List[Tuple[str, bytes]]],
                 matiere: str,
                 chapter_filter: Optional[str],
                 gte_mode: bool,
                 progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Traite les PDFs avec le moteur Granulo.
    """
    
    if progress_cb:
        progress_cb(0.82)
    
    # Extraire texte des sujets
    subject_texts = []
    rows_src = []
    
    for filename, pdf_bytes in subject_files:
        text, pages_info, is_valid = extract_pdf_clean(pdf_bytes)
        
        if not is_valid or len(text) < 200:
            continue
        
        subject_texts.append((filename, text, pages_info))
        
        # Détecter année
        year = 2023
        year_match = re.search(r'20[0-2][0-9]', text[:500])
        if not year_match:
            year_match = re.search(r'20[0-2][0-9]', filename)
        if year_match:
            year = int(year_match.group())
        
        # Fix 7: Nature - meilleure détection BAC
        text_lower_preview = text[:3000].lower()
        filename_lower = filename.lower()
        is_bac = (
            "bac" in filename_lower or
            "baccalauréat" in text_lower_preview or
            "baccalaureat" in text_lower_preview or
            ("exercice" in text_lower_preview and re.search(r'\b20[0-2][0-9]\b', filename))
        )
        nature = "BAC" if is_bac else "Autre"
        
        rows_src.append({
            "Fichier": filename,
            "Nature": nature,
            "Annee": year,
            "Telechargement": ""
        })
    
    # Extraire texte des corrigés
    correction_texts = []
    if correction_files:
        for filename, pdf_bytes in correction_files:
            text, pages_info, is_valid = extract_pdf_clean(pdf_bytes)
            if is_valid and len(text) > 100:
                correction_texts.append((filename, text, pages_info))
    
    if progress_cb:
        progress_cb(0.88)
    
    # Traiter avec le moteur
    engine = GranuloEngine(gte_mode=gte_mode)
    results = engine.process(subject_texts, correction_texts)
    
    if progress_cb:
        progress_cb(1.0)
    
    # Construire df_src
    df_src = pd.DataFrame(rows_src)
    
    # Ajouter Qi_Data pour audit
    qi_by_file = defaultdict(list)
    for q in results["questions"]:
        qi_by_file[q.source_file].append({
            "qi_text": q.text[:100],
            "chapter": q.chapter_hint or "N/A"
        })
    
    if not df_src.empty:
        df_src["Qi_Data"] = df_src["Fichier"].apply(lambda f: qi_by_file.get(f, []))
    
    # Fix 5: Ajouter gte_mode aux résultats pour convert_to_v31_df
    return {
        "df_src": df_src,
        "questions": results["questions"],
        "corrections": results["corrections"],
        "pairings": results["pairings"],
        "qc_candidates": results["qc_candidates"],
        "coverage_results": results["coverage_results"],
        "stats": results["stats"],
        "audit_log": results["audit_log"],
        "pairing_stats": results.get("pairing_stats"),
        "gte_mode": gte_mode
    }


# =============================================================================
# 9) ADAPTER - CONVERSION VERS FORMAT V31 UI
# =============================================================================

def convert_to_v31_df(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Convertit les résultats du moteur vers le DataFrame V31.
    """
    if not results.get("qc_candidates"):
        return pd.DataFrame()
    
    rows = []
    
    # Créer un index des questions
    questions_by_id = {q.qi_id: q for q in results.get("questions", [])}
    
    # Fix 5: Respecter le mode GTE global
    is_gte_mode = results.get("gte_mode", True)
    
    for qc in results["qc_candidates"]:
        # Construire Evidence (Qi associées)
        evidence = []
        for qi_id in qc.source_qi_ids[:10]:
            q = questions_by_id.get(qi_id)
            if q:
                evidence.append({
                    "Fichier": q.source_file,
                    "Qi": q.text[:150]
                })
        
        # FRT Data
        frt_data = qc.frt.to_blocks()
        
        # Coverage info
        cov = results.get("coverage_results", {}).get(qc.chapter_ref)
        is_sealed = cov.is_sealed if cov else False
        
        # Fix 5: Mode doit respecter gte_mode UI
        if is_gte_mode:
            mode = "GTE_PREVIEW"
        elif qc.has_correction:
            mode = "KERNEL"
        else:
            mode = "KERNEL_NO_CORR"
        
        rows.append({
            "QC_ID": qc.qc_id,
            "Titre": qc.title,
            "Chapitre": qc.chapter_ref,
            "Score": qc.score,
            "n_q": qc.n_q,
            "Psi": f"{qc.psi_normalized:.2f}",
            "N_tot": qc.n_total,
            "t_rec": f"{qc.t_rec:.1f}",
            "Triggers": qc.triggers,
            "ARI": qc.ari,
            "FRT_DATA": frt_data,
            "Evidence": evidence,
            "FRT_Source": qc.frt.source,
            "Signature": qc.signature.to_hash(),
            "Sealed": is_sealed,
            "Mode": mode
        })
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values(["Chapitre", "Score"], ascending=[True, False])
    
    return df


def get_coverage_details_v31(results: Dict[str, Any]) -> Dict[str, Dict]:
    """Extrait les détails de coverage pour l'UI."""
    details = {}
    
    for chapter, cov in results.get("coverage_results", {}).items():
        details[chapter] = {
            "sealed": cov.is_sealed,
            "coverage_ratio": cov.coverage_ratio,
            "n_covered": cov.covered,
            "n_total": cov.total_posable,
            "orphans": len(cov.orphans),
            "orphan_ids": cov.orphans[:5]
        }
    
    return details


def get_pairing_examples(results: Dict[str, Any], n: int = 5) -> List[Dict]:
    """Retourne des exemples d'appariement pour l'audit."""
    examples = []
    
    questions_by_id = {q.qi_id: q for q in results.get("questions", [])}
    
    for pairing in results.get("pairings", [])[:n*2]:
        q = questions_by_id.get(pairing.qi_id)
        if not q:
            continue
        
        examples.append({
            "qi_text": q.text[:100],
            "rqi_text": (pairing.rqi_text or "")[:100],
            "method": pairing.method,
            "confidence": f"{pairing.confidence:.2f}",
            "locator": q.locator.to_key()
        })
        
        if len(examples) >= n:
            break
    
    return examples


def get_qi_examples(results: Dict[str, Any], n: int = 5) -> List[Dict]:
    """Retourne des exemples de Qi pour l'audit."""
    examples = []
    
    for q in results.get("questions", [])[:n]:
        examples.append({
            "qi_id": q.qi_id,
            "text": q.text[:120],
            "page": q.locator.page,
            "exercice": q.locator.exercice or "-",
            "question": q.locator.question or "-",
            "chapter": q.chapter_hint or "N/A",
            "verbs": ", ".join(q.detected_verbs[:3])
        })
    
    return examples

# =============================================================================
# 10) STREAMLIT UI - CONSOLE V31 (ZÉRO RÉGRESSION)
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
# STYLES CSS V31 (INCHANGÉ)
# ==============================================================================
st.markdown("""
<style>
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
    .trigger-item {
        background-color: #fff1f2; color: #991b1b; padding: 5px 10px; margin-bottom: 4px; 
        border-radius: 4px; border-left: 4px solid #f87171; font-weight: 600; font-size: 0.9em; display: block;
    }
    .ari-step {
        background-color: #f3f4f6; color: #374151; padding: 4px 8px; margin-bottom: 3px; 
        border-radius: 3px; font-family: monospace; font-size: 0.85em; border: 1px dashed #d1d5db; display: block;
    }
    .frt-block { padding: 12px; border-bottom: 1px solid #e2e8f0; background: white; margin-bottom: 5px; border-radius: 4px; border: 1px solid #e2e8f0;}
    .frt-title { font-weight: 800; text-transform: uppercase; font-size: 0.75em; display: block; margin-bottom: 6px; letter-spacing: 0.5px; }
    .frt-content { font-family: 'Segoe UI', sans-serif; font-size: 0.95em; color: #334155; line-height: 1.6; white-space: pre-wrap; }
    .c-usage { color: #d97706; border-left: 4px solid #d97706; }
    .c-method { color: #059669; border-left: 4px solid #059669; background-color: #f0fdf4; }
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; }
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; }
    .file-block { margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
    .file-header { background-color: #f1f5f9; padding: 8px 12px; font-weight: 700; font-size: 0.85em; color: #475569; border-bottom: 1px solid #e2e8f0; }
    .qi-item { background-color: white; padding: 10px 12px; border-bottom: 1px solid #f8fafc; font-family: 'Georgia', serif; font-size: 0.95em; color: #1e293b; border-left: 3px solid #9333ea; margin: 0; }
    .status-kernel { background-color: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-gte { background-color: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR - PARAMÈTRES ACADÉMIQUES
# ==============================================================================
LISTE_CHAPITRES_UI = {
    "MATHS": list(CHAPTER_KEYWORDS.keys()),
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

with st.sidebar:
    st.header("📚 Paramètres Académiques")
    
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps = LISTE_CHAPITRES_UI.get(sel_matiere, [])
    sel_chapitres = st.multiselect("Chapitres", chaps, default=[])
    
    st.divider()
    
    st.markdown(f'<span class="status-kernel">Kernel {KernelConstants.KERNEL_VERSION}</span>', unsafe_allow_html=True)
    st.caption(f"Status: {KernelConstants.KERNEL_STATUS}")
    
    gte_mode = st.checkbox(
        "Mode GTE (preview sans corrigé)",
        value=True,
        help="Active le mode prévisualisation. Les QC générées ne sont pas scellables."
    )
    
    if gte_mode:
        st.markdown('<span class="status-gte">⚠️ Mode GTE: QC non scellables</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Stats en temps réel
    if 'results' in st.session_state and st.session_state['results']:
        stats = st.session_state['results'].get('stats', {})
        st.markdown("### 📊 Statistiques")
        st.metric("Qi extraites", stats.get('total_qi', 0))
        st.metric("QC générées", stats.get('total_qc', 0))
        st.metric("POSABLES", stats.get('total_posable', 0))
        if stats.get('pairing_rate'):
            st.metric("Taux appariement", f"{stats['pairing_rate']*100:.0f}%")

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
            "https://www.apmep.fr/Terminale-S-702-sujets-702",
            height=80,
            help="URLs de sites contenant des sujets BAC (une par ligne)"
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
            key="subjects"
        )
    with col_upload2:
        correction_files = st.file_uploader(
            "📝 Corrigés (PDF, optionnel)", 
            type=["pdf"], 
            accept_multiple_files=True,
            key="corrections"
        )

    # --- EXÉCUTION ---
    # Validation des inputs
    url_list_check = [u.strip() for u in (urls or "").splitlines() if u.strip()]
    can_run = bool(url_list_check) or bool(subject_files)
    
    if run:
        if not can_run:
            st.error("❌ Aucune URL fournie et aucun PDF uploadé.")
            st.stop()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Fix 6: Passer la liste complète de chapitres
            chapter_filter = sel_chapitres if sel_chapitres else None
            
            if subject_files:
                # Mode upload
                status_text.text("🔍 Extraction des PDFs uploadés...")
                
                pdf_list = [(f.name, f.read()) for f in subject_files]
                for f in subject_files:
                    f.seek(0)
                
                corr_list = None
                if correction_files:
                    corr_list = [(f.name, f.read()) for f in correction_files]
                    for f in correction_files:
                        f.seek(0)
                
                results = process_pdfs(
                    pdf_list, corr_list, sel_matiere, chapter_filter, gte_mode,
                    lambda p: progress_bar.progress(p)
                )
            else:
                # Mode web scraping
                url_list = [u.strip() for u in (urls or "").splitlines() if u.strip()]
                if not url_list:
                    status_text.empty()
                    progress_bar.empty()
                    st.error("❌ Aucune URL fournie et aucun PDF uploadé.")
                    st.stop()
                
                status_text.text("🌐 Scraping et téléchargement des PDFs...")
                
                results = ingest_from_web(
                    url_list, int(vol), sel_matiere, chapter_filter, gte_mode,
                    lambda p: progress_bar.progress(p)
                )
            
            # Convertir en format V31
            df_qc = convert_to_v31_df(results)
            
            # Sauvegarder en session
            st.session_state['df_src'] = results.get('df_src', pd.DataFrame())
            st.session_state['df_qc'] = df_qc
            st.session_state['results'] = results
            
            status_text.empty()
            progress_bar.empty()
            
            stats = results.get('stats', {})
            mode_str = "GTE (preview)" if gte_mode else "KERNEL"
            n_src = len(results.get('df_src', []))
            st.success(f"✅ Traitement terminé [{mode_str}]: {n_src} sujets, {stats.get('total_qi', 0)} Qi, {stats.get('total_qc', 0)} QC")
                
        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.error("❌ Erreur interne pendant l'exécution.")
            with st.expander("Journal technique (debug)", expanded=False):
                import traceback
                st.code(traceback.format_exc())
            st.stop()

    st.divider()

    # --- TABLEAU SUJETS ---
    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        df_src = st.session_state['df_src']
        st.markdown(f"### 📥 Sujets Traités ({len(df_src)})")
        
        display_cols = [c for c in ["Fichier", "Nature", "Annee"] if c in df_src.columns]
        if display_cols:
            df_view = df_src[display_cols].copy()
            if "Annee" in df_view.columns:
                df_view = df_view.rename(columns={"Annee": "Année"})
            st.dataframe(df_view, hide_index=True, use_container_width=True)

        st.divider()

        # --- BASE QC ---
        st.markdown("### 🧠 Base de Connaissance (QC)")
        
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            df_qc = st.session_state['df_qc']
            
            # Filtrer par chapitres
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
                    
                    st.markdown(f"#### 📘 {chap} ({len(subset)} QC)")
                    
                    for idx, row in subset.iterrows():
                        mode = row.get('Mode', 'GTE_PREVIEW')
                        mode_badge = ""
                        if mode == "GTE_PREVIEW":
                            mode_badge = '<span style="background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:4px;font-size:0.7em;margin-left:8px;">GTE</span>'
                        elif row.get('Sealed', False):
                            mode_badge = '<span style="background:#dcfce7;color:#166534;padding:2px 6px;border-radius:4px;font-size:0.7em;margin-left:8px;">SEALED</span>'
                        
                        st.markdown(f"""
                        <div class="qc-header-box">
                            <span class="qc-id-text">{row['QC_ID']}</span>{mode_badge}
                            <span class="qc-title-text">{row['Titre']}</span><br>
                            <span class="qc-meta-text">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_réc={row['t_rec']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        
                        with c1:
                            with st.expander("🔥 Déclencheurs"):
                                triggers = row.get('Triggers', [])
                                if isinstance(triggers, list) and triggers:
                                    for t in triggers:
                                        st.markdown(f"<span class='trigger-item'>\"{t}\"</span>", unsafe_allow_html=True)
                                else:
                                    st.caption("Aucun déclencheur")
                        
                        with c2:
                            with st.expander("⚙️ ARI"):
                                ari = row.get('ARI', [])
                                if isinstance(ari, list) and ari:
                                    for s in ari:
                                        st.markdown(f"<span class='ari-step'>{s}</span>", unsafe_allow_html=True)
                                else:
                                    st.caption("ARI non disponible")
                        
                        with c3:
                            with st.expander("🧾 FRT"):
                                frt_data = row.get('FRT_DATA', [])
                                if isinstance(frt_data, list):
                                    for block in frt_data:
                                        cls = {"usage": "c-usage", "method": "c-method", "trap": "c-trap", "conc": "c-conc"}.get(block.get('type', ''), "")
                                        st.markdown(f"<div class='frt-block {cls}'><span class='frt-title'>{block.get('title', '')}</span><div class='frt-content'>{block.get('text', '')}</div></div>", unsafe_allow_html=True)
                        
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                evidence = row.get('Evidence', [])
                                if isinstance(evidence, list) and evidence:
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
                                else:
                                    st.caption("Aucune Qi associée")
                        
                        st.write("")
        else:
            st.warning("Aucune QC générée. Lancez l'usine.")
    else:
        st.info("⏳ Uploadez des sujets PDF ou entrez une URL et lancez l'usine.")


# ==============================================================================
# ONGLET 2 - AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("🔍 Validation & Diagnostic")
    
    st.markdown("""
    **Objectifs Kernel V10.5:**
    - **POSABLE** = corrigé exploitable ∧ scope rattachable ∧ évaluable
    - **Coverage** = 100% Qi POSABLES couvertes (zéro orphelin)
    """)
    
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        stats = results.get('stats', {})
        
        # --- MÉTRIQUES GLOBALES ---
        st.divider()
        st.markdown("#### 📊 Métriques d'extraction")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Qi totales", stats.get('total_qi', 0))
        col2.metric("POSABLES", stats.get('total_posable', 0))
        col3.metric("QC générées", stats.get('total_qc', 0))
        col4.metric("Taux appariement", f"{stats.get('pairing_rate', 0)*100:.0f}%")
        
        # --- EXEMPLES QI ---
        st.divider()
        st.markdown("#### 📝 Exemples de Qi extraites")
        
        qi_examples = get_qi_examples(results, 5)
        if qi_examples:
            df_qi = pd.DataFrame(qi_examples)
            st.dataframe(df_qi, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune Qi extraite")
        
        # --- APPARIEMENT ---
        if results.get('pairing_stats'):
            st.divider()
            st.markdown("#### 🔗 Détails appariement Qi ↔ RQi")
            
            ps = results['pairing_stats']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Exact (clé)", ps.get('exact', 0))
            col2.metric("Séquentiel", ps.get('sequential', 0))
            col3.metric("Fuzzy", ps.get('fuzzy', 0))
            col4.metric("Non appariés", ps.get('none', 0))
            
            st.markdown("**Exemples d'appariement:**")
            pair_examples = get_pairing_examples(results, 5)
            if pair_examples:
                df_pair = pd.DataFrame(pair_examples)
                st.dataframe(df_pair, use_container_width=True, hide_index=True)
        
        # --- COVERAGE PAR CHAPITRE ---
        st.divider()
        st.markdown("#### 📊 Coverage par Chapitre")
        
        coverage = get_coverage_details_v31(results)
        
        if coverage:
            for ch, details in coverage.items():
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
            st.info("Aucune donnée de coverage")
        
        # --- LOG AUDIT ---
        st.divider()
        st.markdown("#### 📋 Log d'exécution")
        
        with st.expander("Voir le log complet", expanded=False):
            for log in results.get('audit_log', [])[-30:]:
                if "PASS" in log or "✅" in log:
                    st.success(log)
                elif "FAIL" in log or "❌" in log or "ERREUR" in log.upper():
                    st.error(log)
                else:
                    st.info(log)
        
        # --- AUDIT INTERNE ---
        st.divider()
        st.markdown("#### ✅ Test Interne (Sujet Traité)")
        
        df_src = st.session_state.get('df_src', pd.DataFrame())
        if not df_src.empty and 'Fichier' in df_src.columns:
            fichiers = df_src["Fichier"].tolist()
            t1_file = st.selectbox("Choisir un sujet traité", fichiers)
            
            if st.button("🔬 AUDIT INTERNE"):
                # Trouver les Qi du fichier
                file_qis = [q for q in results.get('questions', []) if q.source_file == t1_file]
                df_qc = st.session_state.get('df_qc', pd.DataFrame())
                
                if file_qis and not df_qc.empty:
                    audit_results = []
                    matched = 0
                    
                    for q in file_qis:
                        # Chercher si couverte par une QC
                        is_covered = False
                        matched_qc = "-"
                        
                        for _, qc_row in df_qc.iterrows():
                            evidence = qc_row.get('Evidence', [])
                            if isinstance(evidence, list):
                                for ev in evidence:
                                    if q.text[:50] in ev.get('Qi', ''):
                                        is_covered = True
                                        matched_qc = qc_row['QC_ID']
                                        break
                            if is_covered:
                                break
                        
                        if is_covered:
                            matched += 1
                        
                        audit_results.append({
                            "Qi": q.text[:80],
                            "Chapitre": q.chapter_hint or "N/A",
                            "QC_Matched": matched_qc,
                            "Statut": "✅ MATCH" if is_covered else "❌ ORPHELIN"
                        })
                    
                    coverage_pct = (matched / len(file_qis)) * 100 if file_qis else 0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Couverture", f"{coverage_pct:.0f}%")
                    col2.metric("Qi mappées", f"{matched}/{len(file_qis)}")
                    
                    if coverage_pct >= 100:
                        st.success("✅ 100% de couverture - SUCCÈS")
                    elif coverage_pct >= 80:
                        st.warning(f"⚠️ {coverage_pct:.0f}% de couverture")
                    else:
                        st.error(f"❌ {coverage_pct:.0f}% de couverture")
                    
                    df_audit = pd.DataFrame(audit_results)
                    st.dataframe(df_audit, use_container_width=True, hide_index=True)
                else:
                    st.warning("Aucune Qi pour ce fichier")
        
        # --- AUDIT EXTERNE ---
        st.divider()
        st.markdown("#### 🌍 Test Externe (Sujet Inconnu)")
        
        uploaded = st.file_uploader("Charger un PDF externe", type="pdf", key="audit_external")
        
        if uploaded:
            if st.button("🔬 AUDIT EXTERNE"):
                pdf_bytes = uploaded.read()
                
                with st.spinner("Analyse..."):
                    text, pages_info, is_valid = extract_pdf_clean(pdf_bytes)
                    
                    if not is_valid:
                        st.warning("PDF non reconnu comme contenu maths")
                    else:
                        atomizer = BACAtomizer(uploaded.name)
                        ext_questions = atomizer.atomize(text, pages_info)
                        
                        df_qc = st.session_state.get('df_qc', pd.DataFrame())
                        
                        if ext_questions and not df_qc.empty:
                            matched = 0
                            audit_results = []
                            
                            for q in ext_questions:
                                is_covered = False
                                matched_qc = "-"
                                
                                # Matching par triggers
                                for _, qc_row in df_qc.iterrows():
                                    triggers = qc_row.get('Triggers', [])
                                    if isinstance(triggers, list):
                                        score = sum(1 for t in triggers if t.lower() in q.text.lower())
                                        if score >= 2:
                                            is_covered = True
                                            matched_qc = qc_row['QC_ID']
                                            break
                                
                                if is_covered:
                                    matched += 1
                                
                                audit_results.append({
                                    "Qi": q.text[:80],
                                    "Chapitre": q.chapter_hint or "N/A",
                                    "QC_Matched": matched_qc,
                                    "Statut": "✅ MATCH" if is_covered else "❌ ORPHELIN"
                                })
                            
                            coverage_pct = (matched / len(ext_questions)) * 100 if ext_questions else 0
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Couverture", f"{coverage_pct:.0f}%")
                            col2.metric("Qi mappées", f"{matched}/{len(ext_questions)}")
                            
                            if coverage_pct >= 95:
                                st.success(f"✅ {coverage_pct:.0f}% - OBJECTIF ATTEINT")
                            elif coverage_pct >= 80:
                                st.warning(f"⚠️ {coverage_pct:.0f}%")
                            else:
                                st.error(f"❌ {coverage_pct:.0f}%")
                            
                            df_audit = pd.DataFrame(audit_results)
                            st.dataframe(df_audit, use_container_width=True, hide_index=True)
                        else:
                            st.warning("Aucune Qi extraite du PDF externe")
    else:
        st.info("⏳ Lancez d'abord l'usine pour générer des données.")

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.caption(f"SMAXIA Console V31 | Kernel {KernelConstants.KERNEL_VERSION} ({KernelConstants.KERNEL_STATUS}) | {datetime.now().strftime('%Y-%m-%d')}")
