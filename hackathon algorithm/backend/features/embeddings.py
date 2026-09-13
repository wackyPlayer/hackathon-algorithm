"""Optional self-supervised speech embeddings (WavLM / wav2vec2 via transformers).

Frozen mid-layer hidden states, mean+std pooled over the caller's speech, give a strong
speaker-/engine-agnostic representation that a small logistic-regression head can be trained on
with a few hundred calls (training/train.py --embeddings). Requires: pip install torch transformers.
The model expects 16 kHz audio, so the 8 kHz caller channel is upsampled.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..audio import resample
from ..vad import Vad

log = logging.getLogger("detector.embeddings")


class Embedder:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", layer: int = 6, device: str | None = None):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        t0 = time.time()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device).eval()
        self.layer = layer
        self.model_name = model_name
        log.info("loaded %s on %s in %.1fs", model_name, self.device, time.time() - t0)

    def speech_audio(self, x8k: np.ndarray, vad: Vad, max_seconds: float) -> np.ndarray:
        parts, budget = [], int(max_seconds * 8000)
        for s, e in vad.turns:
            a, b = int(s * 8000), int(e * 8000)
            parts.append(x8k[a:b])
            budget -= b - a
            if budget <= 0:
                break
        if not parts:
            return x8k[: int(max_seconds * 8000)]
        return np.concatenate(parts)[: int(max_seconds * 8000)]

    def embed(self, x8k: np.ndarray, vad: Vad, max_seconds: float = 20.0) -> np.ndarray:
        audio = resample(self.speech_audio(x8k, vad, max_seconds), 8000, 16000)
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(inputs["input_values"].to(self.device))
        h = out.hidden_states[self.layer][0].cpu().numpy()   # (T, hidden)
        return np.concatenate([h.mean(axis=0), h.std(axis=0)]).astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size) * 2
