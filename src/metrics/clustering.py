"""Cluster emergence and domain coherence metrics.

Detects emergence of semantic clusters over time using:
1. k-NN graph community detection (Louvain algorithm)
2. Predefined legal domain coherence tracking
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import networkx as nx
from community import community_louvain
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predefined legal concept groups for coherence tracking
# ---------------------------------------------------------------------------

LEGAL_DOMAINS: dict[str, list[str]] = {
    "competition": [
        "competition", "cartel", "abuse", "dominant", "merger",
        "undertaking", "antitrust", "collusion", "fine", "penalty",
    ],
    "free_movement": [
        "movement", "goods", "persons", "services", "capital",
        "establishment", "worker", "restriction", "barrier", "customs",
    ],
    "data_protection": [
        "data", "protection", "privacy", "processing", "controller",
        "processor", "consent", "personal", "transfer", "surveillance",
    ],
    "state_aid": [
        "aid", "state", "subsidy", "advantage", "selectivity",
        "recovery", "notification", "exemption", "compatible", "distortion",
    ],
    "fundamental_rights": [
        "rights", "fundamental", "dignity", "liberty", "equality",
        "discrimination", "charter", "convention", "freedom", "asylum",
    ],
    "environment": [
        "environment", "environmental", "emission", "waste", "pollution",
        "habitat", "conservation", "sustainability", "climate", "energy",
    ],
    "intellectual_property": [
        "trademark", "patent", "copyright", "intellectual", "property",
        "infringement", "design", "mark", "registration", "licence",
    ],
    "consumer_protection": [
        "consumer", "contract", "unfair", "trader", "distance",
        "warranty", "product", "liability", "safety", "advertising",
    ],
    "preliminary_reference": [
        "reference", "preliminary", "interpretation", "validity",
        "national", "court", "question", "referred", "jurisdiction",
    ],
    "proportionality": [
        "proportionality", "proportionate", "necessary", "appropriate",
        "suitable", "excessive", "legitimate", "objective", "justified",
        "balance",
    ],
}


def build_knn_graph(
    kv: KeyedVectors,
    words: Sequence[str],
    k: int = 10,
) -> nx.Graph:
    """Build a weighted k-NN graph from embeddings.

    Nodes are words, edges connect each word to its k nearest neighbors
    in the vocabulary subset. Edge weight = cosine similarity.
    """
    word_set = set(words) & set(kv.key_to_index.keys())
    G = nx.Graph()
    G.add_nodes_from(word_set)

    for word in word_set:
        neighbors = kv.most_similar(word, topn=k * 2)
        added = 0
        for neighbor, sim in neighbors:
            if neighbor in word_set and added < k:
                G.add_edge(word, neighbor, weight=max(sim, 0.0))
                added += 1

    return G


def detect_communities(G: nx.Graph) -> dict[str, int]:
    """Run Louvain community detection on a k-NN graph.

    Returns dict mapping word -> community_id.
    """
    if len(G) == 0:
        return {}
    partition = community_louvain.best_partition(G, weight="weight")
    return partition


def track_community_evolution(
    partitions: dict[str, dict[str, int]],
) -> list[dict]:
    """Track how communities evolve across time slices.

    Args:
        partitions: Dict mapping time_label -> {word -> community_id}.

    Returns:
        List of event dicts describing births, deaths, merges, splits.
    """
    labels = sorted(partitions.keys())
    events = []

    for i in range(1, len(labels)):
        prev_label = labels[i - 1]
        curr_label = labels[i]
        prev = partitions[prev_label]
        curr = partitions[curr_label]

        # Invert: community_id -> set of words
        prev_communities: dict[int, set[str]] = {}
        for word, cid in prev.items():
            prev_communities.setdefault(cid, set()).add(word)

        curr_communities: dict[int, set[str]] = {}
        for word, cid in curr.items():
            curr_communities.setdefault(cid, set()).add(word)

        # Match communities by maximum Jaccard overlap
        for curr_id, curr_words in curr_communities.items():
            best_overlap = 0.0
            best_prev_id = None
            for prev_id, prev_words in prev_communities.items():
                inter = len(curr_words & prev_words)
                union = len(curr_words | prev_words)
                jaccard = inter / union if union > 0 else 0.0
                if jaccard > best_overlap:
                    best_overlap = jaccard
                    best_prev_id = prev_id

            if best_overlap < 0.1:
                events.append({
                    "type": "birth",
                    "time": curr_label,
                    "community_id": curr_id,
                    "size": len(curr_words),
                    "sample_words": sorted(curr_words)[:10],
                })
            elif best_overlap < 0.5 and best_prev_id is not None:
                events.append({
                    "type": "major_change",
                    "time": curr_label,
                    "community_id": curr_id,
                    "prev_community_id": best_prev_id,
                    "overlap": best_overlap,
                    "gained": sorted(curr_words - prev_communities[best_prev_id])[:10],
                    "lost": sorted(prev_communities[best_prev_id] - curr_words)[:10],
                })

    return events


def domain_coherence(
    kv: KeyedVectors,
    domain_words: list[str],
) -> float:
    """Compute intra-group mean cosine similarity for a set of domain words.

    coherence(C) = mean_{i,j in C, i<j} cos(v_i, v_j)
    Higher = the concept group is tightly clustered. Lower = fragmented.
    """
    present = [w for w in domain_words if w in kv]
    if len(present) < 2:
        return float("nan")

    vecs = np.array([kv[w] for w in present])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs_normed = vecs / norms

    sim_matrix = vecs_normed @ vecs_normed.T
    n = len(present)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def compute_all_domain_coherence(
    kv: KeyedVectors,
    domains: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Compute coherence for all predefined legal domains.

    Returns dict mapping domain_name -> coherence_score.
    """
    domains = domains or LEGAL_DOMAINS
    return {
        name: domain_coherence(kv, words)
        for name, words in domains.items()
    }
