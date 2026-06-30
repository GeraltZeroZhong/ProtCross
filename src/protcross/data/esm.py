"""ESM-C feature extraction."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch

from .structure import MAX_ESM_RESIDUES


class ESMFeatureExtractor:
    """Load a local ESM-C 600M checkpoint and emit residue embeddings."""

    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ESM-C weights not found: {model_path}")

        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein
        except ImportError as exc:
            raise ImportError("The 'esm' package is required for ESM-C feature extraction.") from exc

        self._protein_cls = ESMProtein
        self.tokenizer = self._build_tokenizer()

        model_args = {"d_model": 1152, "n_layers": 36, "n_heads": 18}
        try:
            self.model = ESMC(tokenizer=self.tokenizer, **model_args)
        except Exception as exc:
            raise RuntimeError("Failed to initialize the ESM-C 600M model structure.") from exc

        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[7:]
            if new_key.startswith("model."):
                new_key = new_key[6:]
            cleaned_state_dict[new_key] = value

        load_result = self.model.load_state_dict(cleaned_state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            warnings.warn(
                "ESM-C weights were loaded with missing or unexpected keys. "
                "Verify that the checkpoint matches ESM-C 600M.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _build_tokenizer():
        try:
            from esm.tokenization import EsmSequenceTokenizer
        except ImportError:
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        try:
            return EsmSequenceTokenizer()
        except Exception:
            return EsmSequenceTokenizer(model_name="esmc_600m")

    @torch.no_grad()
    def extract_residue_embeddings(self, sequence: str) -> torch.Tensor:
        sequence = sequence[:MAX_ESM_RESIDUES]
        protein = self._protein_cls(sequence=sequence)
        tokenized = self.model.encode(protein)
        input_ids = tokenized.sequence
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)

        try:
            output = self.model(input_ids, attention_mask=attention_mask)
        except TypeError:
            output = self.model(input_ids)

        return output.embeddings[0, 1:-1, :].cpu()
