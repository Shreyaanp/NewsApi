# app/reach.py
from typing import List, Tuple
import asyncio
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
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
    use_description: bool = True,
) -> Tuple[List[ReachCluster], list[int]]:
    """Cluster articles by semantic similarity.

    Returns a tuple of (clusters, label_for_each_article).
    """
    if not articles:
        return [], []

    model  = await get_or_load_similarity_model()
    # Include description text for richer clustering if available
    titles = [a.title for a in articles]
    if use_description:
        texts = [f"{a.title} {a.description}".strip() for a in articles]
    else:
        texts = titles

    # run blocking encode() off the event-loop
    embeds = await asyncio.to_thread(
        model.encode,
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    clustering = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        metric="cosine",
        linkage="average",
    ).fit(embeds)

    # group article indices per label
    clusters_dict: dict[int, List[int]] = {}
    for idx, lbl in enumerate(clustering.labels_):
        clusters_dict.setdefault(lbl, []).append(idx)

    clusters: List[ReachCluster] = []
    for indices in clusters_dict.values():
        cluster_embeds = embeds[indices]
        # average pairwise similarity; 1.0 for single-item clusters
        if len(indices) > 1:
            sim_matrix = cosine_similarity(cluster_embeds)
            avg_sim = float((np.sum(sim_matrix) - len(indices)) / (len(indices) * (len(indices) - 1)))
            centroid_idx = indices[int(np.argmax(sim_matrix.mean(axis=1)))]
        else:
            avg_sim = 1.0
            centroid_idx = indices[0]

        cluster_titles = [titles[i] for i in indices]
        clusters.append(
            ReachCluster(
                centroid_title=titles[centroid_idx],
                size=len(indices),
                titles=cluster_titles,
                similarity_score=avg_sim,
            )
        )

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters, clustering.labels_.tolist()
