"""AlphaFold structure download workflow."""

from __future__ import annotations

import concurrent.futures
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import requests


PDB_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{4}$")


@dataclass
class AF2DownloadConfig:
    raw_pdb_dir: Path = Path("data/raw_pdb")
    output_dir: Path = Path("data/raw_af2")
    mapping_file: Path = Path("artifacts/pdb_uniprot_mapping.json")
    pdb_id_file: Path | None = None
    initial_mapping_file: Path | None = None
    max_workers: int = 8
    uniprot_candidates: int = 3
    timeout_seconds: int = 30


class AF2Downloader:
    def __init__(self, config: AF2DownloadConfig) -> None:
        self.config = config
        self.print_lock = threading.Lock()
        self.mapping_lock = threading.Lock()
        self.download_locks_lock = threading.Lock()
        self.download_locks: dict[Path, threading.Lock] = {}
        existing_mapping = self.load_mapping_file(config.mapping_file) if config.mapping_file.exists() else {}
        initial_mapping = self.load_mapping_file(config.initial_mapping_file) if config.initial_mapping_file else {}
        self.preloaded_mapping: dict[str, str] = {**initial_mapping, **existing_mapping}
        self.mapping: dict[str, str] = dict(existing_mapping)

    def run(self) -> dict[str, str]:
        pdb_ids = self.collect_pdb_ids()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if not pdb_ids:
            source = self.config.pdb_id_file or self.config.raw_pdb_dir
            print(f"No PDB IDs found in {source}.")
            return {}

        print(f"Found {len(pdb_ids)} PDB IDs. Starting AlphaFold downloads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            list(executor.map(self.process_pdb_id, pdb_ids))

        self.save_mapping()
        return self.mapping

    def collect_pdb_ids(self) -> list[str]:
        if self.config.pdb_id_file is not None:
            return self.load_pdb_id_file(self.config.pdb_id_file)

        if not self.config.raw_pdb_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.config.raw_pdb_dir}")

        pdb_ids = []
        seen = set()
        for pattern in ("*.pdb", "*.cif", "*.mmcif"):
            for structure_file in sorted(self.config.raw_pdb_dir.glob(pattern)):
                pdb_id = structure_file.stem[:4].upper()
                if PDB_ID_PATTERN.match(pdb_id) and pdb_id not in seen:
                    seen.add(pdb_id)
                    pdb_ids.append(pdb_id)
        return pdb_ids

    def load_pdb_id_file(self, pdb_id_file: Path) -> list[str]:
        if not pdb_id_file.exists():
            raise FileNotFoundError(f"PDB ID file not found: {pdb_id_file}")

        pdb_ids: list[str] = []
        seen: set[str] = set()
        for line_number, line in enumerate(pdb_id_file.read_text(encoding="utf-8").splitlines(), start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if not PDB_ID_PATTERN.match(value):
                raise ValueError(
                    f"Invalid PDB ID on line {line_number} of {pdb_id_file}: {value!r}. "
                    "Use one 4-character PDB ID per line."
                )
            pdb_id = value.upper()
            if pdb_id not in seen:
                seen.add(pdb_id)
                pdb_ids.append(pdb_id)
        return pdb_ids

    def process_pdb_file(self, pdb_file: Path) -> None:
        self.process_pdb_id(pdb_file.stem.upper())

    def process_pdb_id(self, pdb_id: str) -> None:
        preloaded_accession = self.preloaded_mapping.get(pdb_id)
        uniprot_ids = [preloaded_accession] if preloaded_accession else self.fetch_uniprot_ids(pdb_id)
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
        download_lock = self.get_download_lock(output_path)

        with download_lock:
            if output_path.exists():
                if output_path.stat().st_size > 0:
                    self.safe_print(f"[ok] {output_path.name} already exists.")
                    return True
                output_path.unlink()

            try:
                response = requests.get(url, timeout=self.config.timeout_seconds)
                if response.status_code == 200:
                    if not self._looks_like_pdb(response.content):
                        self.safe_print(f"[warn] AlphaFold response for {uniprot_id} did not look like a PDB file.")
                        return False
                    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
                    tmp_path.write_bytes(response.content)
                    tmp_path.replace(output_path)
                    self.safe_print(f"[ok] Downloaded {output_path}")
                    return True
                if response.status_code != 404:
                    self.safe_print(f"[warn] AlphaFold download failed for {uniprot_id}: HTTP {response.status_code}")
                return False
            except Exception as exc:
                output_path.with_suffix(output_path.suffix + ".part").unlink(missing_ok=True)
                self.safe_print(f"[warn] AlphaFold download failed for {uniprot_id}: {exc}")
                return False

    def get_download_lock(self, output_path: Path) -> threading.Lock:
        with self.download_locks_lock:
            return self.download_locks.setdefault(output_path, threading.Lock())

    def save_mapping(self) -> None:
        print(f"Writing PDB-to-UniProt mapping to {self.config.mapping_file}...")
        self.config.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.mapping_file.write_text(json.dumps(self.mapping, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved {len(self.mapping)} mapping records.")

    def load_mapping_file(self, mapping_file: Path) -> dict[str, str]:
        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

        data = json.loads(mapping_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Mapping file must contain a JSON object: {mapping_file}")

        mapping: dict[str, str] = {}
        for pdb_id, accession in data.items():
            normalized_pdb_id = str(pdb_id).strip().upper()
            normalized_accession = str(accession).strip()
            if PDB_ID_PATTERN.match(normalized_pdb_id) and normalized_accession:
                mapping[normalized_pdb_id] = normalized_accession
        return mapping

    def safe_print(self, message: str) -> None:
        with self.print_lock:
            print(message)

    @staticmethod
    def _looks_like_pdb(content: bytes) -> bool:
        if not content:
            return False
        prefix = content[:2048].decode("utf-8", errors="ignore")
        return any(token in prefix for token in ("HEADER", "ATOM", "MODEL", "TITLE"))


def download_af2_structures(config: AF2DownloadConfig) -> dict[str, str]:
    return AF2Downloader(config).run()
