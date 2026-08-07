"""Sentence-BERT wrapper for paraphrase detection.

Training fine-tunes a SentenceTransformer with CosineSimilarityLoss on
labeled pairs; inference scores pairs by embedding cosine similarity and
thresholds it into a paraphrase / non-paraphrase decision.

Quick CPU sanity check (small, safe locally):  python -m src.model
Actual fine-tuning belongs on the Kaggle GPU via src/train.py.
"""

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# MiniLM-sized cross-encoder: same backbone capacity as DEFAULT_MODEL, so
# bi-vs-cross comparisons isolate the architecture, and it fits in limited RAM.
DEFAULT_CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ParaphraseModel:
    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, device: str | None = None):
        self.model = SentenceTransformer(model_name_or_path, device=device)

    @property
    def device(self) -> str:
        return str(self.model.device)

    def train(
        self,
        pairs,
        epochs: int = 1,
        batch_size: int = 32,
        warmup_ratio: float = 0.1,
        output_path: str | None = None,
    ) -> None:
        """Fine-tune on (sentence1, sentence2, label) triples. GPU strongly recommended."""
        examples = [InputExample(texts=[s1, s2], label=float(lbl)) for s1, s2, lbl in pairs]
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss = losses.CosineSimilarityLoss(self.model)
        warmup_steps = int(len(loader) * epochs * warmup_ratio)
        self.model.fit(
            train_objectives=[(loader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
        )

    def similarity(self, pairs, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Cosine similarity in [-1, 1] for each (s1, s2) pair."""
        s1 = [p[0] for p in pairs]
        s2 = [p[1] for p in pairs]
        e1 = self.model.encode(
            s1, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=show_progress
        )
        e2 = self.model.encode(
            s2, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=show_progress
        )
        return torch.nn.functional.cosine_similarity(e1, e2).cpu().numpy()

    def predict(self, pairs, threshold: float = 0.5, batch_size: int = 64) -> np.ndarray:
        """0/1 paraphrase predictions at the given similarity threshold."""
        return (self.similarity(pairs, batch_size=batch_size) >= threshold).astype(int)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "ParaphraseModel":
        return cls(model_name_or_path=path, device=device)


class CrossEncoderParaphraseModel:
    """Cross-encoder variant: both sentences go through the transformer
    JOINTLY, so attention can align tokens across the pair — the mechanism
    the bi-encoder lacks. Same interface as ParaphraseModel; similarity()
    returns the model's paraphrase score in [0, 1] instead of cosine sim.

    Much slower at inference: one full transformer forward per PAIR, and
    embeddings cannot be precomputed or indexed.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_CROSS_MODEL,
        device: str | None = None,
        max_length: int = 256,
    ):
        self.model = CrossEncoder(
            model_name_or_path, num_labels=1, device=device, max_length=max_length
        )

    @property
    def device(self) -> str:
        try:
            return str(self.model.device)
        except AttributeError:
            return str(next(self.model.model.parameters()).device)

    def train(
        self,
        pairs,
        epochs: int = 1,
        batch_size: int = 32,
        warmup_ratio: float = 0.1,
        output_path: str | None = None,  # accepted for interface parity; saved via save()
    ) -> None:
        """Fine-tune on (sentence1, sentence2, label) triples with BCE loss."""
        examples = [InputExample(texts=[s1, s2], label=float(lbl)) for s1, s2, lbl in pairs]
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        warmup_steps = int(len(loader) * epochs * warmup_ratio)
        self.model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=warmup_steps)

    def similarity(self, pairs, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Paraphrase score in [0, 1] for each (s1, s2) pair (sigmoid output)."""
        inputs = [[p[0], p[1]] for p in pairs]
        scores = self.model.predict(
            inputs, batch_size=batch_size, show_progress_bar=show_progress
        )
        return np.asarray(scores, dtype=float)

    def predict(self, pairs, threshold: float = 0.5, batch_size: int = 64) -> np.ndarray:
        return (self.similarity(pairs, batch_size=batch_size) >= threshold).astype(int)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "CrossEncoderParaphraseModel":
        return cls(model_name_or_path=path, device=device)


if __name__ == "__main__":
    m = ParaphraseModel()
    print(f"Loaded {DEFAULT_MODEL} on {m.device}")
    demo = [
        ("A man is eating food.", "A man is eating a meal.", 1),
        ("A man is eating food.", "The stock market crashed today.", 0),
    ]
    sims = m.similarity(demo)
    for (s1, s2, lbl), sim in zip(demo, sims):
        print(f"  sim={sim:.3f} (label={lbl}): {s1!r} / {s2!r}")
