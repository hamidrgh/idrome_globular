"""
Generate fusion-construct FASTA files for the IDR x GFP x topology study.

Inputs
------
    GFP_sequence.fasta             -- the 3 GFP variants (+15, +5, -15)
    subset/IDRome_subset_cdhit50.csv -- the 521 IDR rows (post CD-HIT)

Outputs (in ./constructs/)
--------------------------
    all_constructs.fasta                       3126 sequences, single file
    manifest.csv                               one row per construct
    by_variant/GFP{p15,p5,m15}_{Ntail,Ctail}.fasta   6 split files
    individual/<construct_id>.fasta            3126 per-construct FASTAs

Construct definition
--------------------
    Ntail  -> IDR is N-terminus    sequence = IDR + GFP
    Ctail  -> IDR is C-terminus    sequence = GFP + IDR
    LINKER -> currently '' (direct fusion); change LINKER to e.g. 'GSGSGS'
             to insert a flexible spacer.

Headers
-------
    >{construct_id} | idr={seq_name} | gfp={label} | topology={Ntail|Ctail}
        | N_idr={n} | N_construct={N} | ncpr_idr={x:.3f} | fcr_idr={x:.3f}
        | kappa_idr={x:.3f} | nu_idr_free={x:.3f} | Z_gfp={+/-d}
        | linker_len={k}
"""

from __future__ import annotations

import os
import shutil

import pandas as pd

GFP_FASTA   = "GFP_sequence.fasta"
IDR_CSV     = "subset/IDRome_subset_cdhit50.csv"
OUT_DIR     = "constructs"
LINKER      = ""                  # set to "GSGSGS" etc. for a flexible spacer

# friendly label  ->  filesystem-safe tag, written net charge
GFP_LABEL_TO_TAG = {
    "GFP +15": ("GFPp15", +15),
    "GFP +5":  ("GFPp5",   +5),
    "GFP -15": ("GFPm15", -15),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def parse_fasta(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    name: str | None = None
    seq: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(seq)
                name = line.lstrip(">").strip()
                seq = []
            else:
                seq.append(line)
    if name is not None:
        out[name] = "".join(seq)
    return out


def write_fasta_block(fh, header: str, seq: str, width: int = 60) -> None:
    fh.write(f">{header}\n")
    for i in range(0, len(seq), width):
        fh.write(seq[i : i + width] + "\n")


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
print("loading inputs ...")
gfps_raw = parse_fasta(GFP_FASTA)
gfps: dict[str, tuple[str, int, str]] = {}      # label -> (tag, Z, sequence)
for label, (tag, z) in GFP_LABEL_TO_TAG.items():
    if label not in gfps_raw:
        raise SystemExit(f"GFP label '{label}' not found in {GFP_FASTA}; "
                         f"available: {list(gfps_raw)}")
    gfps[label] = (tag, z, gfps_raw[label])
    print(f"  {label!r:12}  tag={tag:6}  Z={z:+3d}  len={len(gfps_raw[label])}")

idr_df = pd.read_csv(IDR_CSV)
needed = ["seq_name", "fasta", "N", "ncpr", "fcr", "kappa", "nu"]
missing = [c for c in needed if c not in idr_df.columns]
if missing:
    raise SystemExit(f"missing IDR columns: {missing}")
print(f"  IDRs loaded: {len(idr_df)} from {IDR_CSV}")


# ---------------------------------------------------------------------------
# prepare output directories
# ---------------------------------------------------------------------------
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)
os.makedirs(os.path.join(OUT_DIR, "by_variant"))
os.makedirs(os.path.join(OUT_DIR, "individual"))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
topologies = ["Ntail", "Ctail"]
manifest_rows: list[dict] = []

# 6 split-file handles, opened up front
split_files: dict[tuple[str, str], object] = {}
for label, (tag, _z, _seq) in gfps.items():
    for topo in topologies:
        split_files[(tag, topo)] = open(
            os.path.join(OUT_DIR, "by_variant", f"{tag}_{topo}.fasta"), "w"
        )

master_fh = open(os.path.join(OUT_DIR, "all_constructs.fasta"), "w")

n_written = 0
for _, row in idr_df.iterrows():
    seq_name = row["seq_name"]
    idr_seq = row["fasta"]
    n_idr = int(row["N"])

    for gfp_label, (gfp_tag, gfp_z, gfp_seq) in gfps.items():
        for topo in topologies:
            if topo == "Ntail":                  # IDR - linker - GFP
                full_seq = idr_seq + LINKER + gfp_seq
            else:                                # Ctail: GFP - linker - IDR
                full_seq = gfp_seq + LINKER + idr_seq

            construct_id = f"{seq_name}__{gfp_tag}__{topo}"
            header = (
                f"{construct_id} | idr={seq_name} | gfp={gfp_label.replace(' ', '')}"
                f" | topology={topo} | N_idr={n_idr} | N_construct={len(full_seq)}"
                f" | ncpr_idr={row['ncpr']:.3f} | fcr_idr={row['fcr']:.3f}"
                f" | kappa_idr={row['kappa']:.3f} | nu_idr_free={row['nu']:.3f}"
                f" | Z_gfp={gfp_z:+d}"
            )
            if LINKER:
                header += f" | linker={LINKER}"

            write_fasta_block(master_fh, header, full_seq)
            write_fasta_block(split_files[(gfp_tag, topo)], header, full_seq)
            with open(os.path.join(OUT_DIR, "individual",
                                   f"{construct_id}.fasta"), "w") as fh:
                write_fasta_block(fh, header, full_seq)

            manifest_rows.append({
                "construct_id": construct_id,
                "idr_seq_name": seq_name,
                "gfp_label":    gfp_label.replace(" ", ""),
                "gfp_tag":      gfp_tag,
                "Z_gfp":        gfp_z,
                "topology":     topo,
                "N_idr":        n_idr,
                "N_construct":  len(full_seq),
                "ncpr_idr":     row["ncpr"],
                "fcr_idr":      row["fcr"],
                "kappa_idr":    row["kappa"],
                "nu_idr_free":  row["nu"],
                "linker":       LINKER,
                "linker_len":   len(LINKER),
                "fasta_path":   f"individual/{construct_id}.fasta",
            })
            n_written += 1

master_fh.close()
for fh in split_files.values():
    fh.close()

manifest = pd.DataFrame(manifest_rows)
manifest_path = os.path.join(OUT_DIR, "manifest.csv")
manifest.to_csv(manifest_path, index=False)

print(f"\nwrote {n_written} constructs")
print(f"  {OUT_DIR}/all_constructs.fasta")
print(f"  {OUT_DIR}/by_variant/  (6 files)")
print(f"  {OUT_DIR}/individual/  ({n_written} files)")
print(f"  {manifest_path}")
