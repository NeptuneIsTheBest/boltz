"""Resumable, evidence-aware human TRPV1 screening with Boltz-2.

Run with the repository's Python environment. Network access is required only
for assets and preparation; prediction and reporting use local inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import gemmi
import numpy as np
import requests
import yaml
from bs4 import BeautifulSoup
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolAlign
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "runs" / "trpv1"
START, END = 430, 710
CATALOG_URL = "https://tcmsp-e.com/browse.php?qc=ingredients"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
PARAMETERS = {
    "model": "boltz2", "precision": "bf16-mixed", "low_memory": True,
    "recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1,
    "sampling_steps_affinity": 200, "diffusion_samples_affinity": 5,
    "max_parallel_samples": 1, "num_workers": 0, "preprocessing_threads": 1,
    "no_kernels": True, "seed": 42, "template_threshold_angstrom": 2.0,
}
ANCHORS = [
    {"id": "SB366791", "name": "SB-366791", "query": "SB-366791", "role": "antagonist_control", "source": "https://pubmed.ncbi.nlm.nih.gov/37117175/"},
    {"id": "AMG9810", "name": "AMG9810", "query": "AMG-9810", "role": "antagonist_control", "source": "https://pubmed.ncbi.nlm.nih.gov/15615864/"},
    {"id": "SAF312", "name": "SAF312", "query": "SAF312", "role": "similarity_reference", "source": "https://pubmed.ncbi.nlm.nih.gov/39107321/"},
    {"id": "CAPSAICIN", "name": "capsaicin", "query": "capsaicin", "role": "agonist_control", "source": "https://pubmed.ncbi.nlm.nih.gov/15685214/"},
]


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{now()}] {message}", flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    temporary.replace(path)


def fetch(url, path, *, expected_sha=None):
    """Download atomically, rejecting error HTML and validating reusable files."""
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".source.json")
    if path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("url") == url and meta.get("size") == path.stat().st_size and digest(path) == meta.get("sha256"):
            return path
        raise ValueError(f"Cached file failed provenance/integrity validation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() and path.suffix in {".tar", ".ckpt"} else 0
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", "") and path.suffix != ".html":
                    raise ValueError(f"Expected data but received HTML: {url}")
                append = offset and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Incorrect Content-Range in resumed download")
                received = 0
                with partial.open("ab" if append else "wb") as output:
                    last = time.monotonic()
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last >= 30:
                            log(f"Downloading {path.name}: {output.tell() / 1024**2:.0f} MiB")
                            last = time.monotonic()
                length = response.headers.get("Content-Length")
                if length and not response.headers.get("Content-Encoding") and received != int(length):
                    raise ValueError(f"Incomplete download: {path.name}")
            sha = digest(partial)
            if expected_sha and sha != expected_sha:
                raise ValueError(f"SHA-256 mismatch: {path.name}")
            partial.replace(path)
            write_json(meta_path, {"url": url, "fetched_at": now(), "size": path.stat().st_size, "sha256": sha})
            return path
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Download failed")


def prepare_assets(root):
    cache = root / "cache" / "boltz"
    listing = requests.get("https://huggingface.co/api/models/boltz-community/boltz-2/tree/main", timeout=30)
    listing.raise_for_status()
    hashes = {item["path"]: item.get("lfs", {}).get("oid") for item in listing.json()}
    for name in ["mols.tar", "boltz2_conf.ckpt", "boltz2_aff.ckpt"]:
        log(f"Preparing model asset {name}")
        fetch(f"https://huggingface.co/boltz-community/boltz-2/resolve/main/{name}", cache / name, expected_sha=hashes.get(name))
    marker = cache / "mols.ready.json"
    if not marker.exists():
        with tarfile.open(cache / "mols.tar") as archive:
            for member in archive.getmembers():
                target = (cache / member.name).resolve()
                if not target.is_relative_to(cache.resolve()) or member.issym() or member.islnk():
                    raise ValueError("Unsafe CCD archive member")
            archive.extractall(cache, filter="data")
        if not (cache / "mols" / "ALA.pkl").exists():
            raise ValueError("CCD archive is missing ALA.pkl")
        write_json(marker, {"archive_sha256": digest(cache / "mols.tar"), "completed_at": now()})


def standardize(smiles, *, expected_inchikey=None):
    original = Chem.MolFromSmiles(smiles)
    if original is None:
        raise ValueError("invalid_smiles")
    if expected_inchikey and Chem.MolToInchiKey(original) != expected_inchikey:
        raise ValueError("source_identity_mismatch")
    cleaned = rdMolStandardize.Cleanup(original)
    fragments = Chem.GetMolFrags(cleaned, asMols=True)
    organic = [mol for mol in fragments if any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())]
    if not organic:
        raise ValueError("no_organic_component")
    if len(organic) != 1:
        raise ValueError("multiple_organic_components")
    mol = Chem.RemoveHs(organic[0])
    if mol.GetNumAtoms() > 56:
        raise ValueError("above_56_model_atoms")
    if mol.GetNumAtoms() < 3:
        raise ValueError("too_small")
    if any(atom.GetAtomicNum() not in {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53} for atom in mol.GetAtoms()):
        raise ValueError("unsupported_element")
    if any(str(info.specified) == "Unspecified" for info in Chem.FindPotentialStereo(mol)):
        raise ValueError("unspecified_stereochemistry")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    return {
        "smiles": canonical, "original_smiles": smiles,
        "inchikey": Chem.MolToInchiKey(mol), "atom_count": mol.GetNumAtoms(),
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "ACYCLIC",
        "mw": Descriptors.MolWt(mol), "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol), "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol), "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
    }


def prepare_reference(root):
    reference = root / "reference"
    sequence_path = fetch("https://rest.uniprot.org/uniprotkb/Q8NER1.json", reference / "Q8NER1.json")
    uniprot = read_json(sequence_path)
    assert uniprot["primaryAccession"] == "Q8NER1" and uniprot["organism"]["taxonId"] == 9606
    full_sequence = uniprot["sequence"]["value"]
    assert len(full_sequence) == 839
    sequence = full_sequence[START - 1:END]
    source = fetch("https://files.rcsb.org/download/8GFA.cif", reference / "8GFA.cif")
    structure = gemmi.read_structure(str(source))
    model = structure[0]
    cropped = gemmi.Structure()
    cropped.name = "human_TRPV1_430_710_CD"
    cropped.add_model(gemmi.Model("1"))
    mapping = []
    for original_chain, new_chain in [("C", "A"), ("D", "B")]:
        chain = gemmi.Chain(new_chain)
        present = set()
        for residue in model[original_chain]:
            position = residue.seqid.num
            if START <= position <= END and residue.het_flag == "A":
                letter = gemmi.find_tabulated_residue(residue.name).one_letter_code
                if full_sequence[position - 1] != letter:
                    raise ValueError(f"Reference/UniProt mismatch at {original_chain}:{position}")
                copy = residue.clone()
                for index in reversed(range(len(copy))):
                    if copy[index].element.is_hydrogen:
                        del copy[index]
                copy.seqid = gemmi.SeqId(position - START + 1, " ")
                copy.label_seq = position - START + 1
                copy.subchain = new_chain
                chain.add_residue(copy)
                present.add(position)
        cropped[0].add_chain(chain)
        mapping += [{"model_chain": new_chain, "model_residue": i - START + 1,
                     "pdb_chain": original_chain, "uniprot_residue": i,
                     "amino_acid": full_sequence[i - 1], "template_resolved": i in present}
                    for i in range(START, END + 1)]
    cropped.write_pdb(str(reference / "receptor.pdb"))
    from Bio.Data.IUPACData import protein_letters_1to3
    cropped.setup_entities()
    for entity in cropped.entities:
        entity.full_sequence = [protein_letters_1to3[letter].upper() for letter in sequence]
        entity.polymer_type = gemmi.PolymerType.PeptideL
    cropped.make_mmcif_document().write_file(str(reference / "receptor.cif"))
    write_csv(reference / "residue_mapping.csv", mapping)
    reference_ligand = next(residue for residue in model["C"] if residue.name == "ZEI")
    ligand = [{"name": atom.name, "element": atom.element.name, "xyz": list(atom.pos)} for atom in reference_ligand if not atom.element.is_hydrogen]
    ligand_xyz = np.array([atom["xyz"] for atom in ligand])
    pocket = []
    for chain in model:
        for residue in chain:
            atoms = [list(atom.pos) for atom in residue if not atom.element.is_hydrogen]
            if residue.het_flag != "A" or not atoms:
                continue
            distance = float(cdist(np.array(atoms), ligand_xyz).min())
            if distance <= 8:
                if chain.name not in {"C", "D"} or not START <= residue.seqid.num <= END:
                    raise ValueError("The receptor crop loses an 8 Å pocket residue")
                pocket.append({"pdb_chain": chain.name, "uniprot_residue": residue.seqid.num,
                               "model_chain": {"C": "A", "D": "B"}[chain.name],
                               "model_residue": residue.seqid.num - START + 1,
                               "distance_angstrom": distance})
    write_json(reference / "target.json", {"uniprot": "Q8NER1", "organism_id": 9606,
               "sequence": sequence, "start": START, "end": END,
               "template_sha256": digest(reference / "receptor.cif"), "pocket": pocket,
               "reference_ligand": ligand, "parameters": PARAMETERS})
    anchors = []
    for anchor in ANCHORS:
        path = reference / f"{anchor['id']}_pubchem.json"
        url = f"{PUBCHEM}/compound/name/{quote(anchor['query'])}/property/SMILES,InChIKey/JSON"
        try:
            fetch(url, path)
            props = read_json(path)["PropertyTable"]["Properties"]
            if len(props) != 1:
                raise ValueError(f"Ambiguous reference ligand identity: {anchor['name']}")
            props = props[0]
            molecule = standardize(props.get("SMILES") or props["IsomericSMILES"], expected_inchikey=props["InChIKey"])
            identity = {"pubchem_cid": str(props["CID"])}
        except requests.RequestException as error:
            log(f"PubChem unavailable for {anchor['name']}; checking an exact ChEMBL synonym match: {error}")
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_synonyms__molecule_synonym__iexact={quote(anchor['query'])}&limit=100"
            path = reference / f"{anchor['id']}_chembl_exact.json"
            fetch(url, path)
            normalize = lambda value: re.sub(r"[^a-z0-9]", "", (value or "").lower())
            exact = []
            for candidate in read_json(path)["molecules"]:
                names = [candidate.get("pref_name")] + [s.get("molecule_synonym") for s in candidate.get("molecule_synonyms", [])]
                if normalize(anchor["query"]) in {normalize(name) for name in names}:
                    exact.append(candidate)
            if len(exact) != 1:
                raise ValueError(f"No unambiguous exact ChEMBL identity for {anchor['name']}") from error
            props = exact[0]["molecule_structures"]
            molecule = standardize(props["canonical_smiles"], expected_inchikey=props["standard_inchi_key"])
            identity = {"chembl_id": exact[0]["molecule_chembl_id"]}
        anchors.append({**anchor, **molecule, **identity, "structure_source": url,
                        "evidence": "human_direct_inhibition" if anchor["role"] != "agonist_control" else "human_agonist",
                        "evidence_sources": [anchor["source"]], "fetched_at": now()})
    write_json(reference / "anchors.json", anchors)
    log("Human receptor crop and reference ligand identities prepared")


def prepare_msa(root):
    from boltz.main import compute_msa

    target = read_json(root / "reference" / "target.json")
    directory = root / "cache" / "msa"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trpv1.csv"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        if row["sequence"] != target["sequence"]:
            raise ValueError("Cached MSA query is not the intended human receptor crop")
        return
    compute_msa(data={"trpv1": target["sequence"]}, target_id="trpv1",
                msa_dir=directory, msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy")
    if not path.exists():
        raise RuntimeError("MSA server did not produce an alignment")
    write_json(directory / "provenance.json", {"server": "https://api.colabfold.com",
               "query_sha256": hashlib.sha256(target["sequence"].encode()).hexdigest(),
               "msa_sha256": digest(path), "created_at": now()})
    prepare_msa(root)


def default_evidence():
    return [
        {"names": ["capsaicin", "capsaicine"], "category": "human_agonist", "species": "Homo sapiens", "source": "https://pubmed.ncbi.nlm.nih.gov/15685214/", "exclude": True},
        {"names": ["piperine"], "category": "human_agonist_desensitization", "species": "Homo sapiens", "source": "https://pubmed.ncbi.nlm.nih.gov/15685214/", "exclude": True},
        {"names": ["6-gingerol", "[6]-gingerol", "gingerol", "6-shogaol", "[6]-shogaol", "shogaol", "zingerone"], "category": "other_species_agonist", "species": "mouse TRPV1; verify assay before human extrapolation", "source": "https://pubmed.ncbi.nlm.nih.gov/31207668/", "exclude": True},
        {"names": ["6-paradol", "[6]-paradol"], "category": "agonist_species_review_required", "species": "see primary study", "source": "https://pubmed.ncbi.nlm.nih.gov/19594761/", "exclude": True},
    ]


def evidence_for(record, evidence):
    names = {re.sub(r"[^a-z0-9]", "", str(name).lower()) for name in [record.get("name", "")] + record.get("synonyms", [])}
    matches = []
    for entry in evidence:
        entry_names = {re.sub(r"[^a-z0-9]", "", name.lower()) for name in entry.get("names", [])}
        by_id = record.get("id") in entry.get("tcmsp_ids", []) or record.get("pubchem_cid") in entry.get("pubchem_cids", [])
        if by_id or names.intersection(entry_names):
            matches.append(entry)
    return {
        "evidence": ";".join(sorted({m["category"] for m in matches})) or "binding_prediction_only_unreviewed",
        "evidence_sources": sorted({m["source"] for m in matches}),
        "exclude_function": any(m.get("exclude", False) for m in matches),
        "literature_reviewed": bool(matches),
    }


def load_evidence(root):
    path = root / "evidence.json"
    if not path.exists():
        write_json(path, default_evidence())
    return read_json(path)


def parse_catalog(html):
    """Read the site's explicit JSON data arrays, never evaluate JavaScript."""
    records = {}
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        source = script.get_text()
        for match in re.finditer(r"\bdata\s*:\s*\[", source):
            rows, _ = json.JSONDecoder().raw_decode(source[match.end() - 1:])
            for row in rows:
                if isinstance(row, dict) and re.fullmatch(r"MOL\d{6}", row.get("MOL_ID", "")):
                    records[row["MOL_ID"]] = row
    if not records:
        raise ValueError("No compound table found; the TCMSP page format may have changed")
    return [records[key] for key in sorted(records)]


def prepare_library(root, limit=None, retry_failures=False):
    from tcmsp import TCMSPClient, Herb
    from tcmsp.errors import StructureUnavailableError

    library = root / "library"
    catalog_path = fetch(CATALOG_URL, library / "catalog.html")
    catalog = parse_catalog(catalog_path.read_text(encoding="utf-8"))
    write_json(library / "catalog.json", {"source_url": CATALOG_URL, "count": len(catalog), "rows": catalog})
    evidence = load_evidence(root)
    completed = 0
    consecutive_failures = 0
    with TCMSPClient(cache_path=root / "cache" / "tcmsp.sqlite3", timeout=30, max_retries=2, request_interval=1) as client:
        for item in catalog:
            identity = item["MOL_ID"]
            output = library / "records" / f"{identity}.json"
            if output.exists() and not (retry_failures and read_json(output)["status"] == "fetch_failed"):
                continue
            if limit is not None and completed >= limit:
                break
            record = {"id": identity, "name": item["molecule_name"], "catalog_properties": item,
                      "catalog_source": CATALOG_URL, "updated_at": now()}
            try:
                detail = client.get_compound(identity)
                compound = detail.entity
                raw = library / "details" / f"{identity}.json"
                write_json(raw, detail.to_dict())
                record.update({"name": compound.name, "synonyms": list(compound.synonyms),
                               "pubchem_cid": compound.pubchem_cid, "tcmsp_inchikey": compound.inchikey,
                               "source_url": detail.source.url, "fetched_at": detail.source.fetched_at.isoformat(),
                               "herbs": [{"name": h.name, "chinese_name": h.chinese_name} for h in detail.related if isinstance(h, Herb)],
                               "ob": compound.ob, "dl": compound.dl})
                record.update(evidence_for(record, evidence))
                try:
                    result = client.download_compound_structure(compound, library / "structures", format="smiles")
                    original_smiles = Path(result.path).read_text(encoding="utf-8").split()[0]
                except Exception:
                    # A source InChIKey, including its stereochemical layer, is
                    # mandatory for this independently downloaded MOL2 fallback.
                    if not compound.inchikey:
                        raise
                    result = client.download_compound_structure(compound, library / "structures", format="mol2")
                    mol = Chem.MolFromMol2File(str(result.path), removeHs=False)
                    if mol is None:
                        raise ValueError("invalid_tcmsp_mol2")
                    Chem.AssignStereochemistryFrom3D(mol)
                    original_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
                    record["structure_fallback"] = "TCMSP_MOL2_verified_against_full_source_InChIKey"
                record.update(standardize(original_smiles, expected_inchikey=compound.inchikey or None))
                record.update({"structure_source": result.source.url, "structure_path": str(result.path), "status": "valid"})
                if record["exclude_function"]:
                    record.update(status="excluded", exclusion_reason="known_agonism_or_desensitization")
            except (ValueError, StructureUnavailableError) as error:
                record.update(status="excluded", exclusion_reason=str(error))
            except Exception as error:
                record.update(status="fetch_failed", error_type=type(error).__name__, error=str(error))
            write_json(output, record)
            completed += 1
            consecutive_failures = consecutive_failures + 1 if record["status"] == "fetch_failed" else 0
            if consecutive_failures >= 3:
                summarize_library(root)
                raise RuntimeError("Three consecutive source failures; collection paused without treating failures as missing molecules")
            if completed % 10 == 0:
                log(f"Collected {completed} new compounds; latest {identity}: {record['status']}")
            if completed % 50 == 0:
                summarize_library(root)
    summarize_library(root)


def summarize_library(root):
    rows = [read_json(path) for path in sorted((root / "library" / "records").glob("*.json"))]
    counts = Counter(row["status"] for row in rows)
    expected = read_json(root / "library" / "catalog.json")["count"]
    summary = {"expected": expected, "processed": len(rows), "pending": expected - len(rows),
               "counts": dict(counts), "enumeration_complete": len(rows) == expected,
               "updated_at": now()}
    write_json(root / "library" / "summary.json", summary)
    write_csv(root / "library" / "all_compounds.csv", rows)
    write_csv(root / "library" / "excluded.csv", [row for row in rows if row["status"] != "valid"])
    log(f"Library status: {summary}")
    return rows


def choose_candidates(records, anchors, n_similar=50, n_diverse=50, scaffold_cap=3):
    """Deterministic selection; exact stereoisomers retain separate identities."""
    unique = {}
    for record in sorted(records, key=lambda r: r["id"]):
        if record["status"] == "valid" and not record.get("exclude_function", False):
            unique.setdefault(record["inchikey"], record)
    pool = list(unique.values())
    anchor_fps = [FINGERPRINT.GetFingerprint(Chem.MolFromSmiles(anchor["smiles"])) for anchor in anchors if anchor["role"] != "agonist_control"]
    fps = {record["id"]: FINGERPRINT.GetFingerprint(Chem.MolFromSmiles(record["smiles"])) for record in pool}
    pool = [{**record, "similarity": max(DataStructs.TanimotoSimilarity(fps[record["id"]], fp) for fp in anchor_fps)} for record in pool]
    pool.sort(key=lambda record: (-record["similarity"], record["id"]))
    selected, scaffold_counts = [], Counter()
    for record in pool:
        if len(selected) >= n_similar:
            break
        if scaffold_counts[record["scaffold"]] < scaffold_cap:
            selected.append({**record, "selection": "antagonist_similarity"})
            scaffold_counts[record["scaffold"]] += 1
    selected_ids = {record["id"] for record in selected}
    remaining = sorted([record for record in pool if record["id"] not in selected_ids], key=lambda record: record["id"])
    rng = np.random.default_rng(42)
    rng.shuffle(remaining)
    reference_fps = [fps[record["id"]] for record in selected] or anchor_fps
    distances = {record["id"]: min(1 - DataStructs.TanimotoSimilarity(fps[record["id"]], fp) for fp in reference_fps) for record in remaining}
    for _ in range(n_diverse):
        eligible = [record for record in remaining if scaffold_counts[record["scaffold"]] < scaffold_cap]
        if not eligible:
            break
        record = max(eligible, key=lambda record: distances[record["id"]])
        selected.append({**record, "selection": "structural_diversity"})
        scaffold_counts[record["scaffold"]] += 1
        remaining.remove(record)
        for other in remaining:
            distances[other["id"]] = min(distances[other["id"]], 1 - DataStructs.TanimotoSimilarity(fps[record["id"]], fps[other["id"]]))
    return selected, pool


def choose_decoys(pool, selected, anchors, count=10):
    """Property matched, low-similarity backgrounds, never asserted inactive."""
    excluded = {row["inchikey"] for row in selected + anchors}
    eligible = [row for row in pool if row["inchikey"] not in excluded and row["similarity"] <= 0.2]
    positives = [a for a in anchors if a["role"] == "antagonist_control"]
    keys = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds"]
    scale = np.array([50, 1, 25, 1, 2, 2])
    output = []
    for index in range(count):
        anchor = positives[index % len(positives)]
        def distance(row):
            return float(np.linalg.norm((np.array([row[k] for k in keys]) - np.array([anchor[k] for k in keys])) / scale))
        if not eligible:
            break
        record = min(eligible, key=lambda row: (distance(row), row["id"]))
        output.append({**record, "role": "computational_decoy", "matched_control": anchor["id"],
                       "property_distance": distance(record), "activity_label": "unknown_not_experimental_inactive"})
        eligible.remove(record)
    return output


def select_library(root):
    summary = read_json(root / "library" / "summary.json")
    if not summary["enumeration_complete"]:
        raise ValueError("Library collection is incomplete; refusing to label a partial catalog as the full selection pool")
    records = summarize_library(root)
    evidence = load_evidence(root)
    for record in records:
        record.update(evidence_for(record, evidence))
    anchors = read_json(root / "reference" / "anchors.json")
    candidates, pool = choose_candidates(records, anchors)
    decoys = choose_decoys(pool, candidates, anchors)
    manifest = {"created_at": now(), "catalog_count": summary["expected"], "library_counts": summary["counts"],
                "evidence_sha256": digest(root / "evidence.json"), "parameters": PARAMETERS,
                "candidates": candidates, "decoys": decoys,
                "controls": [anchor for anchor in anchors if anchor["role"] != "similarity_reference"]}
    path = root / "selection.json"
    if path.exists() and (root / "jobs").exists():
        old = read_json(path)
        if old["candidates"] != candidates or old["decoys"] != decoys:
            raise ValueError("Selection changed after jobs were created; use a new --root to preserve run provenance")
    write_json(path, manifest)
    write_csv(root / "selected_candidates.csv", candidates)
    write_csv(root / "background_decoys.csv", decoys)
    log(f"Selected {len(candidates)} candidates and {len(decoys)} background decoys")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("assets")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--stage", choices=["reference", "msa", "library", "select", "all"], default="all")
    prepare.add_argument("--limit", type=int, default=None, help="Limit NEW library records for a resumable collection pass")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "assets":
        prepare_assets(root)
    elif args.command == "prepare":
        if args.stage in {"reference", "all"}:
            prepare_reference(root)
        if args.stage in {"msa", "all"}:
            prepare_msa(root)
        if args.stage in {"library", "all"}:
            prepare_library(root, args.limit)
        if args.stage in {"select", "all"}:
            select_library(root)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
