# smaxia_granulo_console_adapter.py
# =============================================================================
# ADAPTER V31 ↔ MOTEUR GRANULO V10.4
# =============================================================================
# Expose les mêmes fonctions que l'ancien moteur (ingest_real, compute_qc_real, etc.)
# mais utilise le moteur V10.4 conforme Kernel (GPT 5.2 corrigé)
# ZÉRO RÉGRESSION UI — L'UI V31 reste identique
# =============================================================================

import pandas as pd
import hashlib
import io
import re
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import defaultdict
from datetime import datetime

# Import du moteur V10.4
from smaxia_granulo_engine_v104 import (
    run_granulo_v104,
    create_pack_france_terminale_maths,
    KernelConstants,
    QCCandidate,
    Atom
)

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
# CONSTANTES (compatibilité V31)
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
