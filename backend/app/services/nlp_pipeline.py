"""
NLP Pipeline Service — Milestone 1
Adapted from Colab notebook: Uses arXiv dataset (20k papers bundled as JSON),
exact preprocessing pipeline (clean → tokenize → stop-words → lemmatize),
corpus-level TF-IDF, and extractive summarization.
No LLMs or pre-trained models used.
"""

from __future__ import annotations

import json
import string
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Global state — loaded once at startup
# ──────────────────────────────────────────────────────────────────────────
_df: Optional[pd.DataFrame] = None
_tfidf_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix = None
_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()
_is_loaded = False


# ──────────────────────────────────────────────────────────────────────────
# Step 1: Text Cleaning (from Colab)
# ──────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase + remove punctuation."""
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
    else:
        text = ''
    return text


# ──────────────────────────────────────────────────────────────────────────
# Step 2: Tokenization (from Colab)
# ──────────────────────────────────────────────────────────────────────────

def tokenize_text(text: str) -> List[str]:
    """Split cleaned text into words."""
    if isinstance(text, str):
        return text.split()
    return []


# ──────────────────────────────────────────────────────────────────────────
# Step 3: Stop-word Removal (from Colab)
# ──────────────────────────────────────────────────────────────────────────

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stop words from token list."""
    if isinstance(tokens, list):
        return [word for word in tokens if word not in _stop_words]
    return []


# ──────────────────────────────────────────────────────────────────────────
# Step 4: Lemmatization (from Colab)
# ──────────────────────────────────────────────────────────────────────────

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize each token to its base form."""
    if isinstance(tokens, list):
        return [_lemmatizer.lemmatize(word) for word in tokens]
    return []


# ──────────────────────────────────────────────────────────────────────────
# Step 5: Join tokens for TF-IDF (from Colab)
# ──────────────────────────────────────────────────────────────────────────

def join_tokens(tokens: List[str]) -> str:
    """Join token list back to a single string for TF-IDF."""
    if isinstance(tokens, list):
        return ' '.join(tokens)
    return ''


# ──────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline (same as Colab)
# ──────────────────────────────────────────────────────────────────────────

def full_preprocess(text: str) -> str:
    """Clean → tokenize → remove stop-words → lemmatize → join."""
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    no_stop = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(no_stop)
    return join_tokens(lemmatized)


# ──────────────────────────────────────────────────────────────────────────
# Dataset Loading — from bundled JSON (runs once at startup)
# ──────────────────────────────────────────────────────────────────────────

# Path to the bundled arXiv data file
_DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "arxiv_5k.json"


def load_dataset():
    """
    Load 20k arXiv papers from bundled JSON file,
    preprocess all abstracts, and build corpus-level TF-IDF matrix.
    Mirrors the exact Colab pipeline — no Kaggle credentials needed.
    """
    global _df, _tfidf_vectorizer, _tfidf_matrix, _is_loaded

    if _is_loaded:
        return

    logger.info(f"Loading arXiv dataset from {_DATA_FILE}...")
    with open(_DATA_FILE, "r") as f:
        data = json.load(f)

    _df = pd.DataFrame(data)
    logger.info(f"Loaded {len(_df)} arXiv papers.")

    # ── Preprocessing pipeline (exact Colab steps) ──
    # Step 1: Clean abstract
    _df['cleaned_abstract'] = _df['abstract'].apply(clean_text)

    # Step 2: Tokenize
    _df['tokenized_abstract'] = _df['cleaned_abstract'].apply(
        lambda x: x.split() if isinstance(x, str) else []
    )

    # Step 3: Remove stop words
    _df['abstract_no_stopwords'] = _df['tokenized_abstract'].apply(remove_stopwords)

    # Step 4: Lemmatize
    _df['lemmatized_abstract'] = _df['abstract_no_stopwords'].apply(lemmatize_tokens)

    # Step 5: Join tokens for TF-IDF
    _df['abstract_for_tfidf'] = _df['lemmatized_abstract'].apply(join_tokens)

    # ── TF-IDF on full corpus ──
    logger.info("Building TF-IDF matrix on full corpus...")
    _tfidf_vectorizer = TfidfVectorizer()
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(_df['abstract_for_tfidf'])

    logger.info(f"TF-IDF matrix shape: {_tfidf_matrix.shape}")
    _is_loaded = True


# ──────────────────────────────────────────────────────────────────────────
# Extractive Summarization (exact Colab method)
# ──────────────────────────────────────────────────────────────────────────

def extractive_summarize(
    abstract_text: str,
    abstract_index: int,
    num_sentences: int = 3,
) -> str:
    """
    Score each sentence by summing TF-IDF values of its words
    from the corpus-level TF-IDF matrix. Pick top N sentences.
    Exactly matches the Colab implementation.
    """
    if not isinstance(abstract_text, str) or pd.isna(abstract_text):
        return ""

    sentences = sent_tokenize(abstract_text)
    sentence_scores = []

    for sentence in sentences:
        cleaned_sentence = clean_text(sentence)
        tokens = cleaned_sentence.split()
        no_stopwords_tokens = remove_stopwords(tokens)
        lemmatized_toks = lemmatize_tokens(no_stopwords_tokens)

        sentence_score = 0
        for word in lemmatized_toks:
            if word in _tfidf_vectorizer.vocabulary_:
                word_idx = _tfidf_vectorizer.vocabulary_[word]
                sentence_score += _tfidf_matrix[abstract_index, word_idx]

        sentence_scores.append((sentence_score, sentence))

    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    summary_sentences = [s for score, s in sentence_scores[:num_sentences]]

    return ' '.join(summary_sentences)


# ──────────────────────────────────────────────────────────────────────────
# Key Terms Extraction from matched documents
# ──────────────────────────────────────────────────────────────────────────

def extract_key_terms(doc_indices: List[int], top_n: int = 30) -> List[Dict[str, Any]]:
    """Get top terms by mean TF-IDF score across matched documents."""
    if not doc_indices:
        return []

    sub_matrix = _tfidf_matrix[doc_indices]
    mean_scores = np.asarray(sub_matrix.mean(axis=0)).flatten()
    feature_names = _tfidf_vectorizer.get_feature_names_out()

    top_idx = mean_scores.argsort()[::-1][:top_n]
    return [
        {"term": feature_names[i], "score": round(float(mean_scores[i]), 4)}
        for i in top_idx
        if mean_scores[i] > 0
    ]


# ──────────────────────────────────────────────────────────────────────────
# Topic Clustering via LDA on matched documents
# ──────────────────────────────────────────────────────────────────────────

_TOPIC_LABELS = [
    "Neural & Deep Learning",
    "Data & Feature Engineering",
    "Text & Language Processing",
    "Evaluation & Methods",
    "Applications & Systems",
    "Statistical Modeling",
    "Optimization & Training",
    "Knowledge & Reasoning",
    "Vision & Multimodal",
    "Architecture & Design",
]


def cluster_topics(
    doc_indices: List[int],
    n_topics: int = 5,
    top_words: int = 8,
) -> List[Dict[str, Any]]:
    """Run LDA on the subset of matched documents."""
    if not doc_indices:
        return []

    texts = _df.iloc[doc_indices]['abstract_for_tfidf'].tolist()
    texts = [t for t in texts if t.strip()]
    if not texts:
        return []

    n_topics = min(n_topics, max(1, len(texts)))

    vec = CountVectorizer(max_features=500)
    try:
        dtm = vec.fit_transform(texts)
    except ValueError:
        return []

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=15,
        learning_method="online",
        random_state=42,
    )
    lda.fit(dtm)
    feature_names = vec.get_feature_names_out()

    doc_topics = lda.transform(dtm)
    topic_doc_counts = np.bincount(doc_topics.argmax(axis=1), minlength=n_topics)

    clusters = []
    for idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[::-1][:top_words]
        keywords = [feature_names[i] for i in top_idx]
        label = _TOPIC_LABELS[idx % len(_TOPIC_LABELS)]
        clusters.append({
            "label": label,
            "keywords": keywords,
            "doc_count": int(topic_doc_counts[idx]),
        })
    return clusters


# ──────────────────────────────────────────────────────────────────────────
# Search arXiv papers by keywords
# ──────────────────────────────────────────────────────────────────────────

def search_papers(keywords: List[str], max_results: int = 200) -> List[int]:
    """
    Find papers whose title or abstract contains ANY of the keywords.
    Returns list of DataFrame integer indices.
    """
    if not keywords:
        return []

    query = '|'.join([kw.lower().strip() for kw in keywords if kw.strip()])
    if not query:
        return []

    mask = (
        _df['abstract'].str.lower().str.contains(query, na=False) |
        _df['title'].str.lower().str.contains(query, na=False)
    )
    indices = _df[mask].index.tolist()

    # Limit results for performance
    return indices[:max_results]


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline(
    keywords: List[str],
    num_topics: int = 5,
    summary_sentences: int = 3,
) -> Dict[str, Any]:
    """
    Full Milestone 1 NLP pipeline:
    1. Search arXiv corpus for papers matching keywords
    2. Extract key terms via corpus-level TF-IDF
    3. Cluster topics via LDA
    4. Generate extractive summaries
    """
    t0 = time.perf_counter()

    # Ensure dataset is loaded
    load_dataset()

    # Search for matching papers
    matched_indices = search_papers(keywords)
    if not matched_indices:
        return {
            "terms": [],
            "clusters": [],
            "summary": [],
            "matched_papers": [],
            "meta": {
                "doc_count": 0,
                "total_corpus": len(_df),
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "word_count": 0,
                "sent_count": 0,
                "message": "No papers found matching your keywords. Try broader terms.",
            },
        }

    # Key terms from matched docs
    terms = extract_key_terms(matched_indices, top_n=30)

    # Topic clusters from matched docs
    clusters = cluster_topics(matched_indices, n_topics=num_topics)

    # Extractive summaries for top matched docs (limit to 20 for speed)
    summary_indices = matched_indices[:20]
    summaries = []
    for idx in summary_indices:
        row = _df.iloc[idx]
        s = extractive_summarize(row['abstract'], idx, num_sentences=summary_sentences)
        if s.strip():
            summaries.append(s)

    # Sample of matched papers (for display)
    sample_indices = matched_indices[:10]
    matched_papers = []
    for idx in sample_indices:
        row = _df.iloc[idx]
        matched_papers.append({
            "title": str(row.get('title', '')).strip(),
            "authors": str(row.get('authors', '')).strip()[:200],
            "categories": str(row.get('categories', '')).strip(),
            "abstract_preview": str(row.get('abstract', ''))[:300].strip() + "...",
        })

    elapsed = time.perf_counter() - t0

    # Word/sentence counts across matched abstracts
    combined = " ".join(_df.iloc[matched_indices[:50]]['abstract'].tolist())
    word_count = len(combined.split())
    sent_count = len(sent_tokenize(combined))

    return {
        "terms": terms,
        "clusters": clusters,
        "summary": summaries,
        "matched_papers": matched_papers,
        "meta": {
            "doc_count": len(matched_indices),
            "total_corpus": len(_df),
            "elapsed_s": round(elapsed, 3),
            "word_count": word_count,
            "sent_count": sent_count,
        },
    }
