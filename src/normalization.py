"""
Data-driven text normalization module for Amazon ML Challenge 2026.
Normalizes business names, addresses, and country codes without loss of critical identity signals.
"""

import re
import unicodedata
import pandas as pd
from typing import List, Set

# Compiled regular expressions
RE_UNICODE = re.compile(r"[^\x00-\x7F]+")
RE_PUNCT = re.compile(r"[\.,;:!\?\"'’`\(\)\[\]\{\}<>\\/\|\-_+=~*^&#@$%]")
RE_WHITESPACE = re.compile(r"\s+")

# Legal suffix tokens to strip or normalize
LEGAL_SUFFIX_PATTERNS = [
    (r"\b(private\s+limited|pvt\s*\.?\s*ltd\.?|pvt\s+ltd)\b", " pvt ltd "),
    (r"\b(proprietary\s+limited|pty\s*\.?\s*ltd\.?)\b", " pty ltd "),
    (r"\b(pte\s*\.?\s*ltd\.?)\b", " pte ltd "),
    (r"\b(limited\s+liability\s+company|llc)\b", " llc "),
    (r"\b(limited\s+liability\s+partnership|llp)\b", " llp "),
    (r"\b(incorporated|inc\.?)\b", " inc "),
    (r"\b(corporation|corp\.?)\b", " corp "),
    (r"\b(company|co\.?)\b", " co "),
    (r"\b(limited|ltd\.?)\b", " ltd "),
    (r"\b(gmbh\s*&\s*co\s*\.?\s*kg|gmbh)\b", " gmbh "),
    (r"\b(kabushiki\s+kaisha|k\s*\.?\s*k\.?)\b", " kk "),
    (r"\b(godo\s+kaisha|g\s*\.?\s*k\.?)\b", " gk "),
    (r"\b(besloten\s+vennootschap|b\s*\.?\s*v\.?)\b", " bv "),
    (r"\b(naamloze\s+vennootschap|n\s*\.?\s*v\.?)\b", " nv "),
    (r"\b(societe\s+anonyme|s\s*\.?\s*a\.?)\b", " sa "),
    (r"\b(sarl|sas)\b", " sa "),
    (r"\b(ag|ug)\b", " ag "),
]

# Business term expansions / standardizations
BUSINESS_ABBREVIATIONS = {
    "tech": "technologies",
    "techno": "technologies",
    "technology": "technologies",
    "soln": "solutions",
    "solns": "solutions",
    "solution": "solutions",
    "intl": "international",
    "sys": "systems",
    "system": "systems",
    "ent": "enterprises",
    "enterprise": "enterprises",
    "mfg": "manufacturing",
    "mfg.": "manufacturing",
    "svcs": "services",
    "svc": "services",
    "service": "services",
    "grp": "group",
    "mgmt": "management",
    "comm": "communications",
    "dist": "distribution",
    "corp": "corporation",
    "assoc": "associates",
    "eng": "engineering",
    "engg": "engineering",
    "dev": "development",
    "pharma": "pharmaceuticals",
    "pharm": "pharmaceuticals",
    "chem": "chemical",
    "hldg": "holdings",
    "hldgs": "holdings"
}

# Address standardizations
ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bld": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "fl": "floor",
    "ste": "suite",
    "apt": "apartment",
    "pkwy": "parkway",
    "pl": "place",
    "sq": "square",
    "ctr": "center",
    "cntr": "center",
    "bldg": "building",
    "no": "number",
    "dept": "department",
    "ind": "industrial",
    "dist": "district",
    "sec": "sector"
}

STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "by", "with", "&"
}

def clean_unicode(text: str) -> str:
    """Normalize unicode characters to ascii approximation."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", str(text))
    return norm.encode("ascii", "ignore").decode("ascii")

def normalize_country(country: str) -> str:
    """Normalize country code to uppercase ISO representation."""
    if not country:
        return ""
    c = clean_unicode(country).strip().upper()
    c = RE_PUNCT.sub("", c)
    return c

def normalize_business_name(name: str, strip_legal_suffixes: bool = False) -> str:
    """
    Standardize business name: lowercase, unicode normalization, abbreviation expansion,
    whitespace cleaning, and optional legal suffix stripping.
    """
    if not name:
        return ""
    text = clean_unicode(name).lower()
    
    # Standardize legal suffixes
    for pat, rep in LEGAL_SUFFIX_PATTERNS:
        text = re.sub(pat, rep, text)
        
    # Replace punctuation with spaces
    text = RE_PUNCT.sub(" ", text)
    
    # Expand abbreviations
    tokens = text.split()
    expanded = []
    legal_tokens = {"ltd", "limited", "inc", "corp", "llc", "llp", "gmbh", "pvt", "pty", "pte", "co", "bv", "sa", "ag"}
    
    for t in tokens:
        if strip_legal_suffixes and t in legal_tokens:
            continue
        exp = BUSINESS_ABBREVIATIONS.get(t, t)
        expanded.append(exp)
        
    return " ".join(expanded).strip()

def normalize_address(address: str) -> str:
    """
    Standardize address text: lowercase, unicode clean, standard road abbreviations,
    and whitespace formatting.
    """
    if not address:
        return ""
    text = clean_unicode(address).lower()
    text = RE_PUNCT.sub(" ", text)
    
    tokens = text.split()
    normalized = []
    for t in tokens:
        norm_t = ADDRESS_ABBREVIATIONS.get(t, t)
        normalized.append(norm_t)
        
    return " ".join(normalized).strip()

def tokenize(text: str, remove_stops: bool = True) -> List[str]:
    """Tokenize string into list of clean alphanumeric tokens."""
    if not text:
        return []
    tokens = text.split()
    if remove_stops:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a business record DataFrame without modifying raw fields.
    """
    df = df.copy()
    
    df["business_name_norm"] = df["business_name"].apply(lambda x: normalize_business_name(x, strip_legal_suffixes=False))
    df["business_name_core"] = df["business_name"].apply(lambda x: normalize_business_name(x, strip_legal_suffixes=True))
    df["business_name_tokens"] = df["business_name_core"].apply(tokenize)
    
    df["business_address_norm"] = df["business_address"].apply(normalize_address)
    df["business_address_tokens"] = df["business_address_norm"].apply(tokenize)
    
    df["country_norm"] = df["country"].apply(normalize_country)
    
    return df
