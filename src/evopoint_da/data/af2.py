"""AlphaFold structure download workflow."""

from __future__ import annotations

import concurrent.futures
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class AF2DownloadConfig:
    raw_pdb_dir: Path = Path("data/raw_pdb")
    output_dir: Path = Path("data/raw_af2")
    mapping_file: Path = Path("pdb_uniprot_mapping.json")
    max_workers: int = 8
    uniprot_candidates: int = 3
    timeout_seconds: int = 30


class AF2Downloader:
    def __init__(self, config: AF2DownloadConfig) -> None:
        self.config = config
        self.print_lock = threading.Lock()
        self.mapping_lock = threading.Lock()
        self.mapping: dict[str, str] = {}

    def run(self) -> dict[str, str]:
        if not self.config.raw_pdb_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.config.raw_pdb_dir}")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        pdb_files = sorted(self.config.raw_pdb_dir.glob("*.pdb"))
        if not pdb_files:
            print(f"No .pdb files found in {self.config.raw_pdb_dir}.")
            return {}

        print(f"Found {len(pdb_files)} PDB files. Starting AlphaFold downloads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            list(executor.map(self.process_pdb_file, pdb_files))

        self.save_mapping()
        return self.mapping

    def process_pdb_file(self, pdb_file: Path) -> None:
        pdb_id = pdb_file.stem.upper()
        uniprot_ids = self.fetch_uniprot_ids(pdb_id)
        if not uniprot_ids:
            self.safe_print(f"[skip] No UniProt accession found for PDB {pdb_id}.")
            return

        for accession in uniprot_ids:
            if self.download_structure(accession):
                with self.mapping_lock:
                    self.mapping[pdb_id] = accession
                return

        self.safe_print(f"[skip] All AlphaFold candidates failed for PDB {pdb_id}: {uniprot_ids}")

    def fetch_uniprot_ids(self, pdb_id: str) -> list[str]:
        url = (
            "https://rest.uniprot.org/uniprotkb/search"
            f"?query=xref:pdb-{pdb_id}&fields=accession&size={self.config.uniprot_candidates}"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [item["primaryAccession"] for item in data.get("results", [])]
        except Exception as exc:
            self.safe_print(f"[warn] UniProt lookup failed for PDB {pdb_id}: {exc}")
            return []

    def download_structure(self, uniprot_id: str) -> bool:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
        output_path = self.config.output_dir / f"AF-{uniprot_id}.pdb"

        if output_path.exists():
            self.safe_print(f"[ok] {output_path.name} already exists.")
            return True

        try:
            response = requests.get(url, timeout=self.config.timeout_seconds)
            if response.status_code == 200:
                output_path.write_bytes(response.content)
                self.safe_print(f"[ok] Downloaded {output_path}")
                return True
            if response.status_code != 404:
                self.safe_print(f"[warn] AlphaFold download failed for {uniprot_id}: HTTP {response.status_code}")
            return False
        except Exception as exc:
            self.safe_print(f"[warn] AlphaFold download failed for {uniprot_id}: {exc}")
            return False

    def save_mapping(self) -> None:
        print(f"Writing PDB-to-UniProt mapping to {self.config.mapping_file}...")
        self.config.mapping_file.write_text(json.dumps(self.mapping, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved {len(self.mapping)} mapping records.")

    def safe_print(self, message: str) -> None:
        with self.print_lock:
            print(message)


def download_af2_structures(config: AF2DownloadConfig) -> dict[str, str]:
    return AF2Downloader(config).run()

