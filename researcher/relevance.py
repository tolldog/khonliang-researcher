"""Embedding-based relevance scoring for pre-distillation filtering.

Uses Ollama's embed API with a lightweight embedding model to compute
cosine similarity between paper content and project descriptions.
Papers below the relevance threshold are skipped, saving expensive
LLM distillation calls.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_THRESHOLD = 0.6
# How much of the paper to embed (title + first N chars of content)
CONTENT_PREFIX_LEN = 1500


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RelevanceScorer:
    """Scores papers against project descriptions using embedding similarity.

    Supports adaptive learning via a persistent blackboard: positive signals
    (promoted/read papers) boost future scores for similar content, negative
    signals (skipped/archived) dampen them.
    """

    # Signal weights for score blending
    SIGNAL_BOOST = 0.08    # max boost from positive signals
    SIGNAL_DAMPEN = -0.05  # max penalty from negative signals
    SIGNAL_THRESHOLD = 0.75  # min similarity to count a signal match

    def __init__(
        self,
        projects: Dict[str, Dict[str, Any]],
        ollama_url: str = "http://localhost:11434",
        model: str = DEFAULT_EMBED_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        blackboard=None,
    ):
        self.projects = projects
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.threshold = threshold
        self._project_embeddings: Dict[str, List[float]] = {}
        self._ready = False
        self._board = blackboard  # Optional: persistent blackboard for signal learning

    async def initialize(self):
        """Embed all project descriptions. Call once at startup."""
        if self._ready:
            return

        for name, cfg in self.projects.items():
            desc = cfg.get("description", "")
            if not desc:
                continue
            embedding = await self._embed(desc)
            if embedding:
                self._project_embeddings[name] = embedding

        if self._project_embeddings:
            self._ready = True
            logger.info(
                "Relevance scorer ready: %d projects embedded with %s",
                len(self._project_embeddings),
                self.model,
            )
        else:
            logger.warning("Relevance scorer: no project embeddings generated")

    async def _embed(self, text: str) -> Optional[List[float]]:
        """Get embedding vector from Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/embed",
                    json={"model": self.model, "input": text},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Embed API returned %d", resp.status)
                        return None
                    data = await resp.json()
                    embeddings = data.get("embeddings", [])
                    return embeddings[0] if embeddings else None
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None

    async def record_signal(
        self, title: str, content: str, signal: str, project: str = ""
    ):
        """Record a relevance signal for adaptive learning.

        Args:
            title: Paper title
            content: Paper content (will be truncated for embedding)
            signal: 'positive' (promoted/read) or 'negative' (skipped/archived)
            project: Optional project context for the signal
        """
        if not self._board:
            return

        text = f"{title}\n\n{content[:CONTENT_PREFIX_LEN]}"
        embedding = await self._embed(text)
        if not embedding:
            return

        import time
        key = f"{signal}_{int(time.time())}_{hash(title) % 10000}"
        self._board.post(
            agent_id="relevance_scorer",
            section="relevance_signals",
            key=key,
            content={"title": title, "signal": signal, "project": project},
            embedding=embedding,
            ttl=86400 * 90,  # 90 day retention
        )
        logger.info("Recorded %s signal for: %s", signal, title[:60])

    def _compute_signal_adjustment(self, paper_embedding: List[float]) -> float:
        """Compute score adjustment from learned signals.

        Finds similar past signals on the blackboard and returns a blended
        boost (positive signals) or dampen (negative signals) value.
        """
        if not self._board:
            return 0.0

        try:
            matches = self._board.search_similar(
                embedding=paper_embedding,
                threshold=self.SIGNAL_THRESHOLD,
                limit=10,
                section="relevance_signals",
            )
        except Exception:
            return 0.0

        if not matches:
            return 0.0

        positive_weight = 0.0
        negative_weight = 0.0
        for entry, sim in matches:
            signal = entry.content.get("signal", "") if isinstance(entry.content, dict) else ""
            if signal == "positive":
                positive_weight += sim
            elif signal == "negative":
                negative_weight += sim

        # Normalize and cap
        if positive_weight + negative_weight == 0:
            return 0.0

        # Net signal: positive signals boost, negative dampen
        net = (positive_weight * self.SIGNAL_BOOST + negative_weight * self.SIGNAL_DAMPEN)
        # Cap the total adjustment
        return max(self.SIGNAL_DAMPEN, min(self.SIGNAL_BOOST, net / max(len(matches), 1)))

    async def score(self, title: str, content: str) -> Dict[str, float]:
        """Score paper against all projects. Returns {project: similarity}.

        Blends base embedding similarity with learned signal adjustments
        from the blackboard (if available).
        """
        if not self._ready:
            await self.initialize()
        if not self._ready:
            return {}

        # Embed paper: title + content prefix
        text = f"{title}\n\n{content[:CONTENT_PREFIX_LEN]}"
        paper_embedding = await self._embed(text)
        if not paper_embedding:
            return {}

        # Base scores from project description similarity
        scores = {}
        for name, proj_emb in self._project_embeddings.items():
            scores[name] = cosine_similarity(paper_embedding, proj_emb)

        # Apply learned signal adjustment
        adjustment = self._compute_signal_adjustment(paper_embedding)
        if adjustment != 0.0:
            for name in scores:
                scores[name] = max(0.0, min(1.0, scores[name] + adjustment))
            logger.debug("Signal adjustment: %.3f for %s", adjustment, title[:40])

        return scores

    async def is_relevant(self, title: str, content: str) -> Tuple[bool, Dict[str, float]]:
        """Check if paper is relevant to any project.

        Returns (is_relevant, scores_dict).
        A paper is relevant if its max project score >= threshold.
        """
        scores = await self.score(title, content)
        if not scores:
            # Can't determine relevance — assume relevant to avoid false negatives
            return True, {}

        max_score = max(scores.values())
        return max_score >= self.threshold, scores
