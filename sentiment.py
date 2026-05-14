"""
Transformer-based sentiment analysis with aspect-level granularity.
Supports batch inference, confidence thresholding, and ABSA.
"""

from __future__ import annotations
import torch
import logging
from dataclasses import dataclass
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    text: str
    label: str
    score: float
    is_confident: bool
    aspects: Optional[dict] = None


class SentimentAnalyzer:
    """
    Multi-class sentiment classifier using fine-tuned RoBERTa.
    Supports batch inference and aspect-based sentiment analysis (ABSA).
    """

    LABEL_MAP = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    }

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        confidence_threshold: float = 0.7,
        batch_size: int = 32,
        device: Optional[int] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        logger.info(f"Loading sentiment model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            return_all_scores=True,
        )
        logger.info("Sentiment model loaded successfully")

    def _map_label(self, label: str) -> str:
        return self.LABEL_MAP.get(label, label.lower())

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Run sentiment analysis on a batch of texts."""
        if not texts:
            return []

        # Handle empty/whitespace texts
        cleaned = [t.strip() if t and t.strip() else "[empty]" for t in texts]
        all_scores = self.pipe(cleaned, truncation=True, max_length=512)

        results = []
        for text, scores in zip(texts, all_scores):
            best = max(scores, key=lambda x: x["score"])
            label = self._map_label(best["label"])
            score = round(best["score"], 4)
            results.append(
                SentimentResult(
                    text=text,
                    label=label,
                    score=score,
                    is_confident=score >= self.confidence_threshold,
                )
            )
        return results

    def analyze_with_aspects(
        self, text: str, aspects: list[str]
    ) -> SentimentResult:
        """
        Aspect-Based Sentiment Analysis (ABSA).
        Runs the model with aspect-conditioned input to score sentiment
        for each named entity/topic independently.
        """
        overall = self.analyze_batch([text])[0]

        aspect_scores = {}
        if aspects:
            aspect_texts = [f"{aspect}: {text}" for aspect in aspects]
            aspect_results = self.analyze_batch(aspect_texts)
            aspect_scores = {
                aspect: {"label": r.label, "score": r.score}
                for aspect, r in zip(aspects, aspect_results)
            }

        overall.aspects = aspect_scores
        return overall

    def get_sentiment_distribution(
        self, texts: list[str]
    ) -> dict[str, float]:
        """Return percentage distribution of sentiment labels in a batch."""
        results = self.analyze_batch(texts)
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for r in results:
            if r.label in counts:
                counts[r.label] += 1
        n = len(results) or 1
        return {k: round(v / n * 100, 2) for k, v in counts.items()}


class StreamingSentimentProcessor:
    """
    Kafka consumer wrapper for real-time sentiment processing.
    Buffers messages and processes them in configurable batches.
    """

    def __init__(
        self,
        analyzer: SentimentAnalyzer,
        kafka_config: dict,
        buffer_size: int = 50,
        flush_interval_sec: float = 1.0,
    ):
        self.analyzer = analyzer
        self.kafka_config = kafka_config
        self.buffer_size = buffer_size
        self.flush_interval_sec = flush_interval_sec
        self._buffer: list[dict] = []

    def process_message(self, message: dict) -> SentimentResult:
        """Process a single Kafka message."""
        text = message.get("text", "")
        entities = message.get("entities", [])
        return self.analyzer.analyze_with_aspects(text, entities)

    def flush_buffer(self) -> list[SentimentResult]:
        """Process all buffered messages and clear the buffer."""
        if not self._buffer:
            return []
        texts = [m.get("text", "") for m in self._buffer]
        results = self.analyzer.analyze_batch(texts)
        self._buffer.clear()
        return results
