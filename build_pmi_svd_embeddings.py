#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


PMI_REC_DTYPE = np.dtype([("i", "<u4"), ("j", "<u4"), ("pmi", "<f4")])


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


def list_bucket_files(pmi_dir: Path) -> List[Path]:
    files = sorted(pmi_dir.glob("b*.bin"))
    if not files:
        raise FileNotFoundError(f"No PMI bucket files found in {pmi_dir}")
    return files


def load_pmi_from_buckets(
    pmi_dir: Path,
    vocab_size: int,
    chunk_records: int,
) -> sparse.csr_matrix:
    rows = []
    cols = []
    vals = []

    for bf in list_bucket_files(pmi_dir):
        with bf.open("rb") as f:
            while True:
                arr = np.fromfile(f, dtype=PMI_REC_DTYPE, count=chunk_records)
                if arr.size == 0:
                    break
                rows.append(arr["i"].astype(np.int64, copy=False))
                cols.append(arr["j"].astype(np.int64, copy=False))
                vals.append(arr["pmi"].astype(np.float32, copy=False))

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


def load_pmi_matrix(
    out_dir: Path,
    use_ppmi: bool,
    chunk_records: int,
) -> sparse.csr_matrix:
    npz_path = out_dir / ("ppmi_matrix.npz" if use_ppmi else "pmi_matrix.npz")
    if npz_path.exists():
        return sparse.load_npz(npz_path).tocsr()

    vocab_path = out_dir / "vocab.tsv"
    vocab_size = load_vocab_size(vocab_path)
    pmi_dir = out_dir / ("ppmi" if use_ppmi else "pmi")
    if not pmi_dir.exists():
        raise FileNotFoundError(
            f"Missing {pmi_dir} and {npz_path}. Build the PMI matrix first."
        )
    return load_pmi_from_buckets(
        pmi_dir=pmi_dir,
        vocab_size=vocab_size,
        chunk_records=chunk_records,
    )


def compute_embeddings(
    mat: sparse.csr_matrix,
    rank: int,
    scale_by_singular_values: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = mat.shape
    if n_rows != n_cols:
        raise ValueError(f"PMI matrix must be square, got shape {mat.shape}")
    if rank <= 0 or rank >= min(n_rows, n_cols):
        raise ValueError(
            f"--rank must be in [1, {min(n_rows, n_cols) - 1}] for matrix shape {mat.shape}"
        )

    u, s, _vt = svds(mat.astype(np.float32), k=rank)

    order = np.argsort(s)[::-1]
    u = u[:, order]
    s = s[order]

    if scale_by_singular_values:
        embeddings = u * np.sqrt(s)
    else:
        embeddings = u

    return embeddings.astype(np.float32, copy=False), s.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute truncated SVD of the PMI matrix and write word embeddings "
            "with one row per vocab id."
        )
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory containing vocab.tsv and PMI outputs",
    )
    ap.add_argument(
        "--rank",
        type=int,
        default=300,
        help="Number of singular vectors to keep",
    )
    ap.add_argument(
        "--ppmi",
        action="store_true",
        help="Use PPMI inputs instead of PMI inputs",
    )
    ap.add_argument(
        "--chunk_records",
        type=int,
        default=2_000_000,
        help="Records to read per chunk when loading bucketed PMI",
    )
    ap.add_argument(
        "--raw_u",
        action="store_true",
        help="Write raw left singular vectors U instead of U * sqrt(S)",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    mat = load_pmi_matrix(
        out_dir=out_dir,
        use_ppmi=args.ppmi,
        chunk_records=args.chunk_records,
    )

    embeddings, singular_values = compute_embeddings(
        mat=mat,
        rank=args.rank,
        scale_by_singular_values=not args.raw_u,
    )

    stem = "ppmi" if args.ppmi else "pmi"
    suffix = "u" if args.raw_u else "embeddings"
    emb_path = out_dir / f"{stem}_svd_{suffix}.npy"
    sv_path = out_dir / f"{stem}_svd_singular_values.npy"

    np.save(emb_path, embeddings)
    np.save(sv_path, singular_values)

    print(f"Loaded matrix shape: {mat.shape}")
    print(f"Wrote embeddings matrix to {emb_path}")
    print(f"Wrote singular values to {sv_path}")


if __name__ == "__main__":
    main()
