from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Iterable, Set, Optional

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
    """
    vocab.tsv written by your C++:
      word \t id \t count\n
    """
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
# Analogy parsing
# -----------------------------
_ANALOGY_RE = re.compile(r"^\s*([^:]+)\s*:\s*([^:]+)\s*::\s*([^:]+)\s*:\s*([^:]+)\s*$")


def parse_analogy(s: str) -> Tuple[str, str, str, str]:
    m = _ANALOGY_RE.match(s)
    if not m:
        raise ValueError(f"Bad analogy format: {s!r} (expected a:b::c:d)")
    return tuple(m.group(i).strip() for i in range(1, 5))  # type: ignore


# -----------------------------
# Cooc row extraction (streaming)
# -----------------------------
REC_DTYPE = np.dtype([("i", "<u4"), ("j", "<u4"), ("c", "<u8")])  # 16 bytes


def list_bucket_files(cooc_dir: str | Path) -> List[Path]:
    cooc_dir = Path(cooc_dir)
    files = sorted(cooc_dir.glob("b*.bin"))
    if not files:
        raise FileNotFoundError(f"No bucket files found in {cooc_dir}")
    return files


def extract_rows_from_buckets(
    cooc_dir: str | Path,
    vocab_size: int,
    target_rows: Iterable[int],
    chunk_records: int = 2_000_000,
) -> Dict[int, sparse.csr_matrix]:
    """
    One pass over ALL bucket files, extracting only rows in target_rows.

    Returns: dict row_id -> 1xV CSR row vector with float32 counts.
    """
    targets: Set[int] = set(int(x) for x in target_rows)
    cols: Dict[int, List[np.ndarray]] = {r: [] for r in targets}
    vals: Dict[int, List[np.ndarray]] = {r: [] for r in targets}

    bucket_files = list_bucket_files(cooc_dir)

    for bf in bucket_files:
        with bf.open("rb") as f:
            while True:
                arr = np.fromfile(f, dtype=REC_DTYPE, count=chunk_records)
                if arr.size == 0:
                    break

                i_arr = arr["i"]

                mask = np.zeros(arr.size, dtype=bool)
                for r in targets:
                    mask |= (i_arr == r)

                if not mask.any():
                    continue

                sub = arr[mask]
                sub_i = sub["i"]
                sub_j = sub["j"].astype(np.int32, copy=False)
                sub_c = sub["c"].astype(np.float32, copy=False)

                for r in targets:
                    m = (sub_i == r)
                    if m.any():
                        cols[r].append(sub_j[m].copy())
                        vals[r].append(sub_c[m].copy())

    out: Dict[int, sparse.csr_matrix] = {}
    for r in targets:
        if len(cols[r]) == 0:
            out[r] = sparse.csr_matrix((1, vocab_size), dtype=np.float32)
            continue

        j = np.concatenate(cols[r])
        c = np.concatenate(vals[r])

        row_idx = np.zeros_like(j, dtype=np.int32)
        coo = sparse.coo_matrix((c, (row_idx, j)), shape=(1, vocab_size), dtype=np.float32)
        out[r] = coo.tocsr()

    return out


# -----------------------------
# Scaling + cosine similarity
# -----------------------------
def scale_row_by_count(row: sparse.csr_matrix, count: int) -> sparse.csr_matrix:
    # Confirmed: scale by word count => divide by frequency
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
# Analogy cosine
# -----------------------------
def analogy_cosine(
    analogy: str,
    token2entry: Dict[str, VocabEntry],
    cooc_dir: str | Path,
    vocab_size: int,
) -> Tuple[float, Dict[str, str]]:
    a, b, c, d = parse_analogy(analogy)

    missing = [w for w in (a, b, c, d) if w not in token2entry]
    if missing:
        return 0.0, {"status": "OOV", "missing": ", ".join(missing)}

    ea, eb, ec, ed = token2entry[a], token2entry[b], token2entry[c], token2entry[d]

    rows = extract_rows_from_buckets(
        cooc_dir=cooc_dir,
        vocab_size=vocab_size,
        target_rows=[ea.token_id, eb.token_id, ec.token_id, ed.token_id],
    )

    ra = scale_row_by_count(rows[ea.token_id], ea.count)
    rb = scale_row_by_count(rows[eb.token_id], eb.count)
    rc = scale_row_by_count(rows[ec.token_id], ec.count)
    rd = scale_row_by_count(rows[ed.token_id], ed.count)

    v1 = ra - rb
    v2 = rc - rd

    cos = sparse_cosine(v1, v2)

    dbg = {
        "status": "OK",
        "a": a, "b": b, "c": c, "d": d,
        "a_id": str(ea.token_id), "b_id": str(eb.token_id),
        "c_id": str(ec.token_id), "d_id": str(ed.token_id),
    }
    return cos, dbg


# -----------------------------
# CLI / interactive input
# -----------------------------
def iter_analogies_from_file(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield s


def main():
    p = argparse.ArgumentParser(description="Analogy cosine similarity using cooc rows.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Your OUTDIR that contains vocab.tsv and cooc/ (e.g. /mnt/data/out/cc100_en_60M_v300k_w5)")
    p.add_argument("--vocab_size", type=int, default=300_000,
                   help="Vocabulary size V (default: 300000). Must match cooc/vocab.")
    p.add_argument("--analogy", type=str, default=None,
                   help='Single analogy like "fall:rise::under:over"')
    p.add_argument("--analogies_file", type=str, default=None,
                   help="Text file with one analogy per line (blank lines and #comments ignored)")
    p.add_argument("--interactive", action="store_true",
                   help="Interactive mode: type analogies line-by-line; blank line exits")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    vocab_tsv = out_dir / "vocab.tsv"
    cooc_dir = out_dir / "cooc"

    token2entry = load_vocab_tsv(vocab_tsv)

    analogies: List[str] = []
    if args.analogy:
        analogies.append(args.analogy)
    if args.analogies_file:
        analogies.extend(list(iter_analogies_from_file(Path(args.analogies_file))))

    if args.interactive or (not analogies):
        print('Enter analogies like "fall:rise::under:over" (blank line to quit):')
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            analogies.append(line)

    for s in analogies:
        try:
            cos, dbg = analogy_cosine(
                analogy=s,
                token2entry=token2entry,
                cooc_dir=cooc_dir,
                vocab_size=args.vocab_size,
            )
            if dbg.get("status") == "OOV":
                print(f"{s}\tcosine=0.0\tOOV: {dbg.get('missing')}")
            else:
                print(f"{s}\tcosine={cos:.6f}\tids=({dbg['a_id']},{dbg['b_id']},{dbg['c_id']},{dbg['d_id']})")
        except Exception as e:
            print(f"{s}\tERROR: {e}")


if __name__ == "__main__":
    main()
