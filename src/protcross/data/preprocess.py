"""Preprocessing pipeline for raw PDB/mmCIF structures."""

from __future__ import annotations

import json
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
    seed: int = 42
    fail_on_error: bool = True


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
    rng = np.random.default_rng(config.seed)
    sample_files = rng.choice(raw_files, sample_count, replace=False)
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
    _write_pca_manifest(config, sample_files)
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
    failures = []
    skipped = []
    for file_path in tqdm(raw_files, desc="Processing"):
        output_path = config.output_dir / f"{file_path.stem}.pt"

        try:
            parsed = structure_parser.parse_file_with_labels(file_path)
            if not parsed:
                skipped.append((file_path, "no standard amino-acid residues with CA atoms"))
                continue
            parsed = truncate_parsed_structure(parsed, config.max_len)

            raw_embeddings = esm_extractor.extract_residue_embeddings(parsed["sequence"])
            reduced_embeddings = pca_reducer.transform(raw_embeddings)

            if config.is_af2:
                plddt = torch.from_numpy(parsed["plddts"]) / 100.0
            else:
                plddt = torch.ones(len(parsed["coords"]))

            payload = {
                "pos": torch.from_numpy(parsed["coords"]),
                "x": reduced_embeddings,
                "plddt": plddt.unsqueeze(1),
                "y": torch.from_numpy(parsed["labels"]).float(),
                "residue_ids": parsed["residue_ids"],
                "sequence": parsed["sequence"],
                "truncated": bool(parsed.get("truncated", False)),
                "original_length": int(parsed.get("original_length", len(parsed["sequence"]))),
                "source_path": str(file_path),
                "source_mtime_ns": file_path.stat().st_mtime_ns,
            }
            tmp_path = output_path.with_suffix(output_path.suffix + ".part")
            torch.save(payload, tmp_path)
            tmp_path.replace(output_path)
            success_count += 1
        except Exception as exc:
            output_path.with_suffix(output_path.suffix + ".part").unlink(missing_ok=True)
            failures.append((file_path, str(exc)))
            print(f"Failed {file_path}: {exc}")

    print(f"Done. Processed {success_count}/{len(raw_files)} files.")
    if skipped:
        print(f"Skipped {len(skipped)} files without usable protein residues.")
    if failures:
        print(f"Failed {len(failures)} files during preprocessing.")
    if config.fail_on_error and (failures or success_count == 0):
        first = failures[0] if failures else skipped[0]
        raise RuntimeError(f"Preprocessing did not complete cleanly; first problem: {first[0]}: {first[1]}")
    return success_count


def _write_pca_manifest(config: PreprocessConfig, sample_files) -> None:
    manifest = {
        "schema_version": "protcross-pca-v1",
        "pca_dim": config.pca_dim,
        "seed": config.seed,
        "sample_ratio": config.sample_ratio,
        "sample_count": len(sample_files),
        "sample_files": [str(path) for path in sample_files],
        "max_len": config.max_len,
    }
    manifest_path = Path(str(config.pca_model_path) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
