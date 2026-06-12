from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from hd_tilag.features.sentiment import lexicon_sentiment, normalize_sentiment_label


@dataclass
class SentimentPrediction:
    text: str
    label: str


class BaseFDSAdapter:
    """Interface for the paper's Financial Domain Sentiment model."""

    def predict(self, texts: Iterable[str]) -> list[SentimentPrediction]:
        raise NotImplementedError

    def annotate_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        text_column: str = "text",
    ) -> None:
        frame = pd.read_csv(input_csv)
        predictions = self.predict(frame[text_column].fillna("").astype(str).tolist())
        frame["sentiment"] = [item.label for item in predictions]
        frame.to_csv(output_csv, index=False)


class LexiconFDSAdapter(BaseFDSAdapter):
    """Lightweight fallback when no DeepSeek/LoRA checkpoint is available."""

    def predict(self, texts: Iterable[str]) -> list[SentimentPrediction]:
        return [
            SentimentPrediction(text=str(text), label=normalize_sentiment_label(lexicon_sentiment(str(text))))
            for text in texts
        ]


class TransformersFDSAdapter(BaseFDSAdapter):
    """Adapter for an externally trained FDS-compatible Hugging Face classifier."""

    def __init__(self, model_name_or_path: str, device: int | str = -1) -> None:
        try:
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TransformersFDSAdapter requires `pip install -e .[sentiment]`."
            ) from exc
        self.pipe = pipeline("text-classification", model=model_name_or_path, device=device)

    def predict(self, texts: Iterable[str]) -> list[SentimentPrediction]:
        text_list = [str(text) for text in texts]
        outputs = self.pipe(text_list, truncation=True)
        return [
            SentimentPrediction(text=text, label=normalize_sentiment_label(output["label"]))
            for text, output in zip(text_list, outputs)
        ]
