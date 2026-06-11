#!/usr/bin/env python3
'''
Extract per-residue qFitLatent embeddings: the single representation s_i after
the final IPA block — i.e. exactly what the chi head reads. One [d_single]
vector per residue, SE(3)-invariant, derived from backbone geometry + sequence.

Like inference, only N/CA/C and the residue identities are used, so a
backbone-only PDB is a valid input.

Usage (from the repo root):
    python scripts/extract_embeddings.py --pdb structures/5v92.pdb
    python scripts/extract_embeddings.py --pdb backbone.pdb --out emb.npz \
        --ckpt checkpoints/vmm_ar_k8/latest.pt

Output (.npz):
    emb      [N, d_single] float32   per-residue embedding (post-IPA s_i)
    chain    [N] str                 chain id
    resseq   [N] int                 residue number
    resname  [N] str                 3-letter residue name
    aa_token [N] int                 amino-acid token (0..20)
'''
import argparse, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import load_model, DEFAULT_CKPT
from qfit_latent.data.data import ground_truth, parse_pdb, IDX_TO_AA3


@torch.no_grad()
def extract_embeddings(model, sample: dict, device: str) -> torch.Tensor:
    '''Run the encoder only; return the post-IPA single rep s [N, d_single].'''
    s = model.aa_embed(sample["aa_tokens"].to(device))
    z = model.pair_init(s, sample["t"].to(device))
    for block in model.ipa_blocks:
        s, z = block(s, z, sample["R"].to(device), sample["t"].to(device))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb",    type=Path, required=True,
                    help="Input PDB (backbone-only is sufficient)")
    ap.add_argument("--ckpt",   type=Path, default=DEFAULT_CKPT,
                    help=f"Model checkpoint (default: {DEFAULT_CKPT})")
    ap.add_argument("--out",    type=Path, default=None,
                    help="Output .npz path (default: <pdb_stem>_embeddings.npz)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Torch device (default: cuda if available else cpu)")
    args = ap.parse_args()

    model  = load_model(args.ckpt, args.device)
    sample = ground_truth(args.pdb, max_len=None)
    if sample is None:
        sys.exit(f"failed to parse {args.pdb}")

    emb  = extract_embeddings(model, sample, args.device).cpu().numpy()
    keys = sorted(parse_pdb(args.pdb))                       # same order as ground_truth
    records = parse_pdb(args.pdb)
    aa   = sample["aa_tokens"].numpy()

    out = args.out or args.pdb.parent / f"{args.pdb.stem}_embeddings.npz"
    np.savez(
        out,
        emb=emb.astype(np.float32),
        chain=np.array([k[0] for k in keys]),
        resseq=np.array([k[1] for k in keys], dtype=np.int64),
        resname=np.array([records[k]["resname"] for k in keys]),
        aa_token=aa,
    )
    print(f"wrote → {out}  ({emb.shape[0]} residues × {emb.shape[1]} dims)")


if __name__ == "__main__":
    main()
