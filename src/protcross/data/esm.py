"""ESM-C feature extraction."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from numbers import Integral
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

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
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

    @torch.no_grad()
    def extract_residue_embeddings_many(
        self,
        sequences: Sequence[str],
        *,
        max_batch_size: int = 8,
        max_padded_tokens: int = 2048,
    ) -> list[torch.Tensor]:
        """Embed sequences in bounded, padding-aware microbatches.

        ESM-C masks its pad token internally.  A one-sequence microbatch uses
        the established single-sequence path so batch size one remains exactly
        backward compatible.  On an accelerator OOM, a microbatch is split
        recursively without changing sequence order or chain boundaries.
        """
        normalized = [str(sequence)[:MAX_ESM_RESIDUES] for sequence in sequences]
        if not normalized:
            return []
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, Integral)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer.")
        if (
            isinstance(max_padded_tokens, bool)
            or not isinstance(max_padded_tokens, Integral)
            or max_padded_tokens < 3
        ):
            raise ValueError("max_padded_tokens must be an integer of at least 3.")

        outputs: list[torch.Tensor] = []
        start = 0
        while start < len(normalized):
            stop = start
            longest_tokens = 0
            while stop < len(normalized) and stop - start < max_batch_size:
                candidate_tokens = len(normalized[stop]) + 2
                if candidate_tokens > max_padded_tokens:
                    raise ValueError(
                        "A sequence requires "
                        f"{candidate_tokens} padded tokens, exceeding max_padded_tokens="
                        f"{max_padded_tokens}."
                    )
                next_longest = max(longest_tokens, candidate_tokens)
                next_size = stop - start + 1
                if stop > start and next_longest * next_size > max_padded_tokens:
                    break
                longest_tokens = next_longest
                stop += 1
            outputs.extend(self._extract_microbatch(normalized[start:stop]))
            start = stop
        return outputs

    def _extract_microbatch(self, sequences: list[str]) -> list[torch.Tensor]:
        if len(sequences) == 1:
            return [self.extract_residue_embeddings(sequences[0])]

        token_rows = []
        for sequence in sequences:
            protein = self._protein_cls(sequence=sequence)
            tokens = self.model.encode(protein).sequence
            token_rows.append(tokens if tokens.dim() == 1 else tokens.squeeze(0))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            token_rows,
            batch_first=True,
            padding_value=int(self.tokenizer.pad_token_id),
        ).to(self.device)
        try:
            output = self.model(input_ids)
        except RuntimeError as exc:
            if len(sequences) <= 1 or "out of memory" not in str(exc).lower():
                raise
            del input_ids
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if str(self.device).startswith("mps") and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            middle = len(sequences) // 2
            return self._extract_microbatch(sequences[:middle]) + self._extract_microbatch(sequences[middle:])
        return [
            output.embeddings[index, 1 : len(sequence) + 1, :].cpu()
            for index, sequence in enumerate(sequences)
        ]
