# app/reach.py
from typing import List, Tuple
import asyncio
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import torch

from .models import Article, ReachCluster

_model: SentenceTransformer | None = None

# ───────────────────────────────────────────
async def get_or_load_similarity_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # load once; CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=device
        )
    return _model

# ───────────────────────────────────────────
async def cluster_with_labels(
    articles: List[Article],
    distance_threshold: float = 0.6,
) -> Tuple[List[ReachCluster], list[int]]:
    """
    Returns (clusters, label_for_each_article)
    """
    if not articles:
        return [], []

    model  = await get_or_load_similarity_model()
    titles = [a.title for a in articles]

    # run blocking encode() off the event-loop
    embeds = await asyncio.to_thread(
        model.encode,
        titles,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    clustering = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        metric="cosine",
        linkage="average",
    ).fit(embeds)

    # group articles per label
    clusters_dict: dict[int, List[str]] = {}
    for lbl, title in zip(clustering.labels_, titles):
        clusters_dict.setdefault(lbl, []).append(title)

    clusters: List[ReachCluster] = []
    for cluster_titles in clusters_dict.values():
        clusters.append(
            ReachCluster(
                centroid_title=max(cluster_titles, key=len),
                size=len(cluster_titles),
                titles=cluster_titles,
            )
        )

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters, clustering.labels_.tolist()
