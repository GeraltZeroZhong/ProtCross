"""Preprocessing pipeline for raw PDB/mmCIF structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from . import MAX_ESM_RESIDUES, PCAReducer, StructureParser, truncate_parsed_structure
from .esm import ESMFeatureExtractor


@dataclass
class PreprocessConfig:
    data_dir: Path
    output_dir: Path
    model_name: Path
    pca_model_path: Path = Path("pca_esmc_128.pkl")
    fit_pca: bool = False
    pca_dim: int = 128
    is_af2: bool = False
    sample_ratio: float = 0.1
    device: str | None = None
    max_len: int = MAX_ESM_RESIDUES


def preprocess_directory(config: PreprocessConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = _find_structure_files(config.data_dir)
    if not raw_files:
        raise FileNotFoundError(f"No PDB/mmCIF files found in {config.data_dir}")

    esm_extractor = ESMFeatureExtractor(model_path=config.model_name, device=config.device)
    structure_parser = StructureParser()
    pca_reducer = PCAReducer(n_components=config.pca_dim)

    if config.fit_pca:
        _fit_pca(config, raw_files, structure_parser, esm_extractor, pca_reducer)
    else:
        if not config.pca_model_path.exists():
            raise FileNotFoundError(f"PCA model not found: {config.pca_model_path}")
        pca_reducer.load(config.pca_model_path)
        print(f"Loaded PCA model from {config.pca_model_path}")

    return _process_files(config, raw_files, structure_parser, esm_extractor, pca_reducer)


def _find_structure_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.pdb")) + sorted(data_dir.glob("*.cif")) + sorted(data_dir.glob("*.mmcif"))


def _fit_pca(
    config: PreprocessConfig,
    raw_files: list[Path],
    structure_parser: StructureParser,
    esm_extractor: ESMFeatureExtractor,
    pca_reducer: PCAReducer,
) -> None:
    print(f"Fitting PCA (target dim: {config.pca_dim})...")
    if 0 < config.sample_ratio < 1:
        ratio_count = max(1, int(len(raw_files) * config.sample_ratio))
    else:
        ratio_count = int(config.sample_ratio)
    sample_count = min(len(raw_files), 500, max(1, ratio_count))
    sample_files = np.random.choice(raw_files, sample_count, replace=False)
    embeddings = []
    for file_path in tqdm(sample_files, desc="PCA sampling"):
        parsed = structure_parser.parse_file_with_labels(file_path)
        if not parsed:
            continue
        parsed = truncate_parsed_structure(parsed, config.max_len)
        try:
            embeddings.append(esm_extractor.extract_residue_embeddings(parsed["sequence"]))
        except Exception as exc:
            print(f"PCA fit skipped {file_path}: {exc}")

    if not embeddings:
        raise RuntimeError("No valid embeddings were extracted for PCA fitting.")
    pca_reducer.fit(embeddings)
    pca_reducer.save(config.pca_model_path)
    print(f"PCA model saved to {config.pca_model_path}")


def _process_files(
    config: PreprocessConfig,
    raw_files: list[Path],
    structure_parser: StructureParser,
    esm_extractor: ESMFeatureExtractor,
    pca_reducer: PCAReducer,
) -> int:
    print(f"Processing {len(raw_files)} files (is_af2={config.is_af2})...")
    success_count = 0
    for file_path in tqdm(raw_files, desc="Processing"):
        output_path = config.output_dir / f"{file_path.stem}.pt"
        if output_path.exists():
            output_path.unlink()

        try:
            parsed = structure_parser.parse_file_with_labels(file_path)
            if not parsed:
                continue
            parsed = truncate_parsed_structure(parsed, config.max_len)

            raw_embeddings = esm_extractor.extract_residue_embeddings(parsed["sequence"])
            reduced_embeddings = pca_reducer.transform(raw_embeddings)

            if config.is_af2:
                plddt = torch.from_numpy(parsed["plddts"]) / 100.0
            else:
                plddt = torch.ones(len(parsed["coords"]))

            torch.save(
                {
                    "pos": torch.from_numpy(parsed["coords"]),
                    "x": reduced_embeddings,
                    "plddt": plddt.unsqueeze(1),
                    "y": torch.from_numpy(parsed["labels"]).float(),
                    "residue_ids": parsed["residue_ids"],
                },
                output_path,
            )
            success_count += 1
        except Exception as exc:
            print(f"Failed {file_path}: {exc}")

    print(f"Done. Processed {success_count}/{len(raw_files)} files.")
    return success_count
