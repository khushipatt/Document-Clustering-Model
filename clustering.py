"""
clustering.py — Core ML logic for Document Clustering
Supports: K-Means, Hierarchical (Agglomerative), LDA Topic Modeling
"""

import re
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = list(stopwords.words('english'))
EXTRA_STOPS = ['said', 'also', 'would', 'could', 'one', 'two', 'three',
               'get', 'got', 'like', 'just', 'know', 'think', 'make', 'way']
ALL_STOPS = STOP_WORDS + EXTRA_STOPS


# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean and normalize a single document."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\S+@\S+', ' ', text)          # remove emails
    text = re.sub(r'http\S+|www\.\S+', ' ', text) # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)      # keep only letters
    text = re.sub(r'\s+', ' ', text)               # collapse whitespace
    return text.lower().strip()


def clean_corpus(docs: list) -> list:
    return [clean_text(d) for d in docs]


# ─── Vectorization ─────────────────────────────────────────────────────────────

def vectorize(docs: list, max_features: int = 3000):
    """TF-IDF vectorize a list of documents."""
    cleaned = clean_corpus(docs)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=ALL_STOPS,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(cleaned)
    return X, vectorizer, cleaned


# ─── Clustering Algorithms ────────────────────────────────────────────────────

def kmeans_cluster(X, n_clusters: int = 5):
    """Run K-Means clustering."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=15, max_iter=300)
    labels = model.fit_predict(X)
    return labels, model


def hierarchical_cluster(X, n_clusters: int = 5, linkage_type: str = 'ward'):
    """Run Agglomerative (Hierarchical) clustering."""
    # Agglomerative needs dense input
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    # Reduce dims first for performance on large datasets
    if X_dense.shape[1] > 200:
        pca = PCA(n_components=100, random_state=42)
        X_dense = pca.fit_transform(X_dense)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    labels = model.fit_predict(X_dense)
    return labels, model


def lda_cluster(X, n_topics: int = 5):
    """Run LDA Topic Modeling and assign dominant topic as cluster label."""
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    lda.fit(X)
    doc_topics = lda.transform(X)
    labels = np.argmax(doc_topics, axis=1)
    return labels, lda


# ─── Top Words per Cluster ────────────────────────────────────────────────────

def get_top_words(model, vectorizer, n_words: int = 12, method: str = 'kmeans') -> dict:
    """Extract top keywords for each cluster/topic."""
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    if method == 'kmeans':
        centers = model.cluster_centers_
        for i, center in enumerate(centers):
            top_idx = center.argsort()[-n_words:][::-1]
            top_words[i] = [feature_names[j] for j in top_idx]
    elif method == 'hierarchical':
        # For hierarchical, compute mean TF-IDF per cluster from labels
        # model doesn't have cluster_centers_, so we return empty here
        # (caller should pass X and labels separately — see get_top_words_from_labels)
        top_words = {}
    elif method == 'lda':
        for i, topic in enumerate(model.components_):
            top_idx = topic.argsort()[-n_words:][::-1]
            top_words[i] = [feature_names[j] for j in top_idx]
    return top_words


def get_top_words_from_labels(X, labels: np.ndarray, vectorizer, n_words: int = 12) -> dict:
    """Compute top words per cluster by averaging TF-IDF scores (for Hierarchical)."""
    feature_names = vectorizer.get_feature_names_out()
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    top_words = {}
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        mean_vec = X_dense[mask].mean(axis=0)
        top_idx = mean_vec.argsort()[-n_words:][::-1]
        top_words[cluster_id] = [feature_names[j] for j in top_idx]
    return top_words


# ─── Evaluation ──────────────────────────────────────────────────────────────

def compute_silhouette(X, labels: np.ndarray) -> float:
    """Compute silhouette score (higher is better, range -1 to 1)."""
    try:
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        if X_dense.shape[1] > 300:
            pca = PCA(n_components=100, random_state=42)
            X_dense = pca.fit_transform(X_dense)
        n_unique = len(np.unique(labels))
        if n_unique < 2:
            return 0.0
        return round(silhouette_score(X_dense, labels, sample_size=min(500, len(labels))), 4)
    except Exception:
        return 0.0


def elbow_method(X, max_k: int = 10):
    """Compute inertia for K=2..max_k (for elbow chart)."""
    inertias = []
    silhouettes = []
    ks = list(range(2, max_k + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(compute_silhouette(X, km.labels_))
    return ks, inertias, silhouettes


# ─── Dimension Reduction ─────────────────────────────────────────────────────

def reduce_to_2d(X, method: str = 'pca') -> np.ndarray:
    """Reduce high-dim TF-IDF to 2D for plotting."""
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    # Always pre-reduce with PCA first for speed
    n_components = min(50, X_dense.shape[1], X_dense.shape[0] - 1)
    if X_dense.shape[1] > 50:
        pca_pre = PCA(n_components=n_components, random_state=42)
        X_dense = pca_pre.fit_transform(X_dense)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_dense)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_from_csv(df: pd.DataFrame, text_col: str) -> list:
    """Extract document list from a DataFrame column."""
    docs = df[text_col].dropna().astype(str).tolist()
    return docs


def load_from_txt(content: str) -> list:
    """Split a plain text file into documents by double newline or line."""
    parts = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 20]
    if len(parts) < 5:
        # Fallback: split by single newlines
        parts = [p.strip() for p in content.split('\n') if len(p.strip()) > 20]
    return parts
