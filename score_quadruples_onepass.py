#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set, Optional

import numpy as np
from scipy import sparse


# -----------------------------
# Vocab loader
# -----------------------------
@dataclass(frozen=True)
class VocabEntry:
    token: str
    token_id: int
    count: int


def load_vocab_tsv(vocab_path: str | Path) -> Dict[str, VocabEntry]:
    vocab_path = Path(vocab_path)
    token2entry: Dict[str, VocabEntry] = {}
    with vocab_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad vocab.tsv line {line_no}: {line!r}")
            w, sid, sc = parts
            token2entry[w] = VocabEntry(token=w, token_id=int(sid), count=int(sc))
    return token2entry


# -----------------------------
# Cooc streaming row extraction
# -----------------------------
REC_DTYPE = np.dtype([("i", "<u4"), ("j", "<u4"), ("c", "<u8")])  # 16 bytes


def list_bucket_files(cooc_dir: str | Path) -> List[Path]:
    cooc_dir = Path(cooc_dir)
    files = sorted(cooc_dir.glob("b*.bin"))
    if not files:
        raise FileNotFoundError(f"No bucket files found in {cooc_dir}")
    return files


def extract_rows_one_pass(
    cooc_dir: str | Path,
    vocab_size: int,
    target_rows: Set[int],
    chunk_records: int = 2_000_000,
) -> Dict[int, sparse.csr_matrix]:
    """
    Single pass over all bucket files, collecting all (i,j,count) where i in target_rows.
    Returns row_id -> 1xV CSR row (float32 counts), duplicates summed.
    """
    cols: Dict[int, List[np.ndarray]] = {r: [] for r in target_rows}
    vals: Dict[int, List[np.ndarray]] = {r: [] for r in target_rows}

    # For fast membership filter, turn targets into a sorted numpy array once
    # We'll use np.isin per chunk.
    target_arr = np.array(sorted(target_rows), dtype=np.uint32)

    for bf in list_bucket_files(cooc_dir):
        with bf.open("rb") as f:
            while True:
                arr = np.fromfile(f, dtype=REC_DTYPE, count=chunk_records)
                if arr.size == 0:
                    break

                i_arr = arr["i"]
                mask = np.isin(i_arr, target_arr, assume_unique=False)
                if not mask.any():
                    continue

                sub = arr[mask]
                sub_i = sub["i"]
                sub_j = sub["j"].astype(np.int32, copy=False)
                sub_c = sub["c"].astype(np.float32, copy=False)

                # group per row id
                # (targets set is not huge; loop is fine)
                for r in target_rows:
                    m = (sub_i == r)
                    if m.any():
                        cols[r].append(sub_j[m].copy())
                        vals[r].append(sub_c[m].copy())

    out: Dict[int, sparse.csr_matrix] = {}
    for r in target_rows:
        if not cols[r]:
            out[r] = sparse.csr_matrix((1, vocab_size), dtype=np.float32)
            continue
        j = np.concatenate(cols[r])
        c = np.concatenate(vals[r])
        row_idx = np.zeros_like(j, dtype=np.int32)
        coo = sparse.coo_matrix((c, (row_idx, j)), shape=(1, vocab_size), dtype=np.float32)
        out[r] = coo.tocsr()  # sums duplicates
    return out


# -----------------------------
# Math helpers
# -----------------------------
def scale_row_by_count(row: sparse.csr_matrix, count: int) -> sparse.csr_matrix:
    if count <= 0:
        return row
    return row * (1.0 / float(count))


def sparse_cosine(u: sparse.csr_matrix, v: sparse.csr_matrix) -> float:
    dot = float(u.multiply(v).sum())
    nu = float(np.sqrt(u.multiply(u).sum()))
    nv = float(np.sqrt(v.multiply(v).sum()))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


# -----------------------------
# CSV scoring
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Score analogy quadruples CSV using one-pass cooc scan.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="OUTDIR containing vocab.tsv and cooc/")
    ap.add_argument("--in_csv", type=str, required=True,
                    help="Input CSV created by make_bats_quadruples.py")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Output CSV with cosine filled where possible")
    ap.add_argument("--vocab_size", type=int, default=300_000,
                    help="Vocabulary size (must match cooc/vocab)")
    ap.add_argument("--chunk_records", type=int, default=2_000_000,
                    help="Records to read per chunk per bucket file")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    vocab_path = out_dir / "vocab.tsv"
    cooc_dir = out_dir / "cooc"

    token2entry = load_vocab_tsv(vocab_path)

    # Read input rows first; collect targets
    rows: List[dict] = []
    needed_ids: Set[int] = set()

    with Path(args.in_csv).open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required_cols = {"a", "b", "c", "d"}
        missing_cols = required_cols - set(r.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Input CSV missing columns: {sorted(missing_cols)}")

        for row in r:
            a, b, c, d = row["a"], row["b"], row["c"], row["d"]
            missing_words = [w for w in (a, b, c, d) if w not in token2entry]
            if not missing_words:
                ea, eb, ec, ed = token2entry[a], token2entry[b], token2entry[c], token2entry[d]
                needed_ids.update([ea.token_id, eb.token_id, ec.token_id, ed.token_id])
                row["_status"] = "OK"
                row["_missing"] = ""
            else:
                row["_status"] = "OOV"
                row["_missing"] = ",".join(missing_words)
            rows.append(row)

    print(f"Loaded {len(rows)} quadruples. Need {len(needed_ids)} unique rows from cooc.")

    # One-pass extract all needed rows
    cooc_rows: Dict[int, sparse.csr_matrix] = {}
    if needed_ids:
        cooc_rows = extract_rows_one_pass(
            cooc_dir=cooc_dir,
            vocab_size=args.vocab_size,
            target_rows=needed_ids,
            chunk_records=args.chunk_records,
        )

    # Score and write output
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Preserve original columns + ensure cosine/status/missing exist
    base_fieldnames = list(rows[0].keys()) if rows else []
    # remove internal keys if present
    base_fieldnames = [k for k in base_fieldnames if not k.startswith("_")]
    # enforce result columns
    for col in ["cosine", "status", "missing"]:
        if col not in base_fieldnames:
            base_fieldnames.append(col)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fieldnames)
        w.writeheader()

        for row in rows:
            a, b, c, d = row["a"], row["b"], row["c"], row["d"]

            if row["_status"] != "OK":
                # Leave cosine blank for OOV as requested
                row["cosine"] = ""
                row["status"] = "OOV"
                row["missing"] = row["_missing"]
                w.writerow({k: row.get(k, "") for k in base_fieldnames})
                continue

            ea, eb, ec, ed = token2entry[a], token2entry[b], token2entry[c], token2entry[d]

            ra = scale_row_by_count(cooc_rows[ea.token_id], ea.count)
            rb = scale_row_by_count(cooc_rows[eb.token_id], eb.count)
            rc = scale_row_by_count(cooc_rows[ec.token_id], ec.count)
            rd = scale_row_by_count(cooc_rows[ed.token_id], ed.count)

            v1 = ra - rb
            v2 = rc - rd

            cos = sparse_cosine(v1, v2)

            row["cosine"] = f"{cos:.6f}"
            row["status"] = "OK"
            row["missing"] = ""
            w.writerow({k: row.get(k, "") for k in base_fieldnames})

    print(f"Wrote scored CSV to {out_csv}")


if __name__ == "__main__":
    main()
