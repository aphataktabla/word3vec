#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


COOC_REC_DTYPE = np.dtype([("i", "<u4"), ("j", "<u4"), ("c", "<u8")])


def load_vocab_size(vocab_path: Path) -> int:
    max_id = -1
    with vocab_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad vocab.tsv line {line_no}: {line!r}")
            wid = int(parts[1])
            if wid > max_id:
                max_id = wid

    if max_id < 0:
        raise ValueError(f"No vocabulary entries found in {vocab_path}")
    return max_id + 1


def list_bucket_files(cooc_dir: Path) -> List[Path]:
    files = sorted(cooc_dir.glob("b*.bin"))
    if not files:
        raise FileNotFoundError(f"No co-occurrence bucket files found in {cooc_dir}")
    return files


def load_cooc_from_buckets(
    cooc_dir: Path,
    vocab_size: int,
    chunk_records: int,
) -> sparse.csr_matrix:
    rows = []
    cols = []
    vals = []

    for bf in list_bucket_files(cooc_dir):
        with bf.open("rb") as f:
            while True:
                arr = np.fromfile(f, dtype=COOC_REC_DTYPE, count=chunk_records)
                if arr.size == 0:
                    break
                rows.append(arr["i"].astype(np.int64, copy=False))
                cols.append(arr["j"].astype(np.int64, copy=False))
                vals.append(arr["c"].astype(np.float32, copy=False))

    if not rows:
        return sparse.csr_matrix((vocab_size, vocab_size), dtype=np.float32)

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    data = np.concatenate(vals)
    return sparse.coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(vocab_size, vocab_size),
        dtype=np.float32,
    ).tocsr()


def load_cooc_matrix(
    out_dir: Path,
    chunk_records: int,
) -> sparse.csr_matrix:
    npz_path = out_dir / "cooc_matrix.npz"
    if npz_path.exists():
        return sparse.load_npz(npz_path).tocsr()

    vocab_path = out_dir / "vocab.tsv"
    cooc_dir = out_dir / "cooc"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}")
    if not cooc_dir.exists():
        raise FileNotFoundError(f"Missing {cooc_dir}")

    vocab_size = load_vocab_size(vocab_path)
    return load_cooc_from_buckets(
        cooc_dir=cooc_dir,
        vocab_size=vocab_size,
        chunk_records=chunk_records,
    )


def compute_left_singular_vectors(
    mat: sparse.csr_matrix,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = mat.shape
    if n_rows != n_cols:
        raise ValueError(f"Co-occurrence matrix must be square, got shape {mat.shape}")
    if rank <= 0 or rank >= min(n_rows, n_cols):
        raise ValueError(
            f"--rank must be in [1, {min(n_rows, n_cols) - 1}] for matrix shape {mat.shape}"
        )

    u, s, _vt = svds(mat.astype(np.float32), k=rank)
    order = np.argsort(s)[::-1]
    return (
        u[:, order].astype(np.float32, copy=False),
        s[order].astype(np.float32, copy=False),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute truncated SVD of the co-occurrence matrix and write raw "
            "left singular vectors with one row per vocab id."
        )
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory containing vocab.tsv and cooc/",
    )
    ap.add_argument(
        "--rank",
        type=int,
        default=300,
        help="Number of left singular vectors to keep",
    )
    ap.add_argument(
        "--chunk_records",
        type=int,
        default=2_000_000,
        help="Records to read per chunk when loading bucketed co-occurrence data",
    )
    ap.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save the sparse co-occurrence matrix as cooc_matrix.npz",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    mat = load_cooc_matrix(
        out_dir=out_dir,
        chunk_records=args.chunk_records,
    )
    u, singular_values = compute_left_singular_vectors(
        mat=mat,
        rank=args.rank,
    )

    emb_path = out_dir / "cooc_svd_u.npy"
    sv_path = out_dir / "cooc_svd_singular_values.npy"

    np.save(emb_path, u)
    np.save(sv_path, singular_values)

    if args.save_npz:
        npz_path = out_dir / "cooc_matrix.npz"
        sparse.save_npz(npz_path, mat)

    print(f"Loaded matrix shape: {mat.shape}")
    print(f"Wrote embeddings matrix to {emb_path}")
    print(f"Wrote singular values to {sv_path}")
    if args.save_npz:
        print(f"Saved sparse matrix to {out_dir / 'cooc_matrix.npz'}")


if __name__ == "__main__":
    main()
