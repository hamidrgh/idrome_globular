"""
Stratified, diversity-aware sampling of a ~500-sequence representative subset
from the IDRome database.

Design
------
Each IDR is binned on a 4-D grid:

    1. Length N           : log-spaced  ( <50, 50-100, 100-200, 200-400, >=400 )
    2. NCPR (signed)      : 5 bins separating polyampholytes / polyelectrolytes
                            ( <-.15, [-.15,-.05), [-.05,.05], (.05,.15], >.15 )
    3. Das-Pappu FCR band : 3 bands ( weak <=.25, Janus .25-.35, strong >.35 )
    4. Kappa (patterning) : 3 bands ( low <.15, mid .15-.27, high >=.27 )

For every non-empty cell we

    (a) reduce sequence redundancy with a greedy k-mer-Jaccard clustering
        (~50% identity proxy, CD-HIT style); cells are already length- and
        composition-constrained so the proxy is well behaved.
    (b) run k-medoids (KMeans + nearest real point) on standardised
        biophysical features [fcr, kappa, scd, nu, shd, mean_lambda] to keep
        k diverse representatives per cell.

The per-cell quota k is set adaptively so the total subset size hits TARGET_N.

Outputs (in ./subset/)
----------------------
    IDRome_subset.csv          : selected rows, full original columns + bin ids
    IDRome_subset.fasta        : FASTA of the selected IDRs
    IDRome_subset_coverage.png : coverage diagnostics
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
INPUT_CSV   = "IDRome_DB.csv"
OUT_DIR     = "subset"
TARGET_N    = 500
SEED        = 0

N_EDGES     = [30, 50, 100, 200, 400, 10_000]
NCPR_EDGES  = [-1.0, -0.15, -0.05, 0.05, 0.15, 1.0]
FCR_EDGES   = [0.0, 0.25, 0.35, 1.01]          # Das-Pappu weak / Janus / strong
KAPPA_EDGES = [0.0, 0.15, 0.27, 1.01]          # low / mid / high patterning

KMER_K            = 3
JACCARD_THRESH    = 0.50       # ~50% identity proxy for short IDR fragments
FEATURES_FOR_KMED = ["fcr", "kappa", "scd", "nu", "shd", "mean_lambda"]

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def kmer_set(seq: str, k: int = KMER_K) -> set[str]:
    seq = seq.upper()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else {seq}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def greedy_redundancy_reduce(df_cell: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """CD-HIT-style greedy clustering on k-mer Jaccard.

    Sequences are sorted longest-first; each candidate is kept iff its Jaccard
    similarity to every already-kept sequence is < thresh.
    """
    order = df_cell["N"].sort_values(ascending=False).index
    kept_idx: list[int] = []
    kept_kmers: list[set[str]] = []
    for idx in order:
        ks = kmer_set(df_cell.at[idx, "fasta"])
        if all(jaccard(ks, ref) < thresh for ref in kept_kmers):
            kept_idx.append(idx)
            kept_kmers.append(ks)
    return df_cell.loc[kept_idx]


def kmedoid_pick(df_cell: pd.DataFrame, k: int, features: list[str]) -> pd.DataFrame:
    """KMeans on standardised features + nearest-real-point selection."""
    if len(df_cell) <= k:
        return df_cell
    X = df_cell[features].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
    picks: list[int] = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(Xs[members] - km.cluster_centers_[c], axis=1)
        picks.append(members[int(np.argmin(d))])
    return df_cell.iloc[sorted(set(picks))]


# ---------------------------------------------------------------------------
# load + bin
# ---------------------------------------------------------------------------
print(f"loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df):,} IDRs loaded")

df["bin_N"]     = pd.cut(df["N"],     N_EDGES,     right=False, labels=False)
df["bin_ncpr"]  = pd.cut(df["ncpr"],  NCPR_EDGES,  right=False, labels=False)
df["bin_fcr"]   = pd.cut(df["fcr"],   FCR_EDGES,   right=False, labels=False)
df["bin_kappa"] = pd.cut(df["kappa"], KAPPA_EDGES, right=False, labels=False)

bin_cols = ["bin_N", "bin_ncpr", "bin_fcr", "bin_kappa"]
df = df.dropna(subset=bin_cols).copy()
for c in bin_cols:
    df[c] = df[c].astype(int)

cell_sizes = df.groupby(bin_cols).size()
n_cells_nonempty = (cell_sizes > 0).sum()
print(f"  non-empty cells: {n_cells_nonempty} / {5*5*3*3}")
print(f"  median cell size: {int(cell_sizes.median())}, "
      f"max: {int(cell_sizes.max())}, "
      f"singletons: {(cell_sizes == 1).sum()}")


# ---------------------------------------------------------------------------
# per-cell: redundancy reduction first, then determine adaptive quota
# (sqrt allocation: dense cells get a few more reps, sparse cells get 1)
# ---------------------------------------------------------------------------
print("\nstep 1: per-cell redundancy reduction (~50%% k-mer Jaccard) ...")
deduped_by_cell: dict[tuple, pd.DataFrame] = {}
for keys, sub in df.groupby(bin_cols, sort=True):
    deduped_by_cell[keys] = (
        greedy_redundancy_reduce(sub, JACCARD_THRESH) if len(sub) > 1 else sub
    )

sizes_after_dedup = {k: len(v) for k, v in deduped_by_cell.items()}
sqrt_sum = sum(np.sqrt(s) for s in sizes_after_dedup.values())
scale    = TARGET_N / sqrt_sum
quota = {
    k: max(1, min(int(round(scale * np.sqrt(s))), s))
    for k, s in sizes_after_dedup.items()
}
print(f"  quota: min={min(quota.values())} max={max(quota.values())} "
      f"sum={sum(quota.values())}")

print("\nstep 2: k-medoid pick per cell ...")
picked_parts: list[pd.DataFrame] = []
cell_report: list[dict] = []
for keys, sub_dedup in deduped_by_cell.items():
    k = quota[keys]
    picks = kmedoid_pick(sub_dedup, k, FEATURES_FOR_KMED)
    picked_parts.append(picks)
    cell_report.append({
        "bin_N": keys[0], "bin_ncpr": keys[1],
        "bin_fcr": keys[2], "bin_kappa": keys[3],
        "n_raw": int(cell_sizes.get(keys, 0)),
        "n_after_dedup": len(sub_dedup),
        "quota": k, "n_picked": len(picks),
    })

subset = pd.concat(picked_parts).reset_index(drop=True)
print(f"  picked {len(subset)} sequences across {len(cell_report)} non-empty cells")


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
csv_out   = os.path.join(OUT_DIR, "IDRome_subset.csv")
fasta_out = os.path.join(OUT_DIR, "IDRome_subset.fasta")
report_out = os.path.join(OUT_DIR, "IDRome_subset_cell_report.csv")

subset.to_csv(csv_out, index=False)
pd.DataFrame(cell_report).to_csv(report_out, index=False)

with open(fasta_out, "w") as fh:
    for _, row in subset.iterrows():
        fh.write(f">{row['seq_name']} N={row['N']} ncpr={row['ncpr']:.3f} "
                 f"fcr={row['fcr']:.3f} kappa={row['kappa']:.3f} nu={row['nu']:.3f}\n")
        fh.write(row["fasta"] + "\n")

print(f"\nwrote {csv_out}")
print(f"wrote {fasta_out}")
print(f"wrote {report_out}")


# ---------------------------------------------------------------------------
# diagnostic plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. log-N vs ncpr, background = full DB, foreground = subset
ax = axes[0, 0]
ax.scatter(df["N"], df["ncpr"], s=2, c="lightgrey", alpha=0.4, label=f"IDRome ({len(df):,})")
ax.scatter(subset["N"], subset["ncpr"], s=14, c="C3", alpha=0.85, label=f"subset ({len(subset)})")
ax.set_xscale("log"); ax.set_xlabel("length N"); ax.set_ylabel("NCPR")
ax.set_title("length x NCPR coverage"); ax.legend(loc="lower right", fontsize=8)

# 2. Das-Pappu plane (FCR vs NCPR)
ax = axes[0, 1]
ax.scatter(df["ncpr"], df["fcr"], s=2, c="lightgrey", alpha=0.4)
ax.scatter(subset["ncpr"], subset["fcr"], s=14, c="C3", alpha=0.85)
# Das-Pappu region boundaries
ax.axhline(0.25, ls="--", c="k", lw=0.6); ax.axhline(0.35, ls="--", c="k", lw=0.6)
ax.set_xlabel("NCPR"); ax.set_ylabel("FCR")
ax.set_title("Das-Pappu plane (FCR vs NCPR)")

# 3. kappa distribution
ax = axes[0, 2]
ax.hist(df["kappa"],     bins=40, density=True, alpha=0.35, label="IDRome", color="grey")
ax.hist(subset["kappa"], bins=40, density=True, alpha=0.7,  label="subset", color="C3")
for e in KAPPA_EDGES[1:-1]:
    ax.axvline(e, ls="--", c="k", lw=0.6)
ax.set_xlabel("kappa"); ax.set_ylabel("density"); ax.set_title("patterning"); ax.legend()

# 4. nu distribution
ax = axes[1, 0]
ax.hist(df["nu"],     bins=40, density=True, alpha=0.35, color="grey",  label="IDRome")
ax.hist(subset["nu"], bins=40, density=True, alpha=0.7,  color="C3",    label="subset")
ax.set_xlabel("scaling exponent nu"); ax.set_ylabel("density")
ax.set_title("conformational compactness"); ax.legend()

# 5. length histogram (log)
ax = axes[1, 1]
bins_log = np.logspace(np.log10(30), np.log10(df["N"].max()), 40)
ax.hist(df["N"],     bins=bins_log, density=True, alpha=0.35, color="grey", label="IDRome")
ax.hist(subset["N"], bins=bins_log, density=True, alpha=0.7,  color="C3",   label="subset")
ax.set_xscale("log"); ax.set_xlabel("length N"); ax.set_ylabel("density")
ax.set_title("length"); ax.legend()

# 6. IDR class composition
ax = axes[1, 2]
classes = ["is_idp", "is_btw_folded", "is_nterm", "is_cterm"]
db_frac     = [df[c].mean()     for c in classes]
subset_frac = [subset[c].mean() for c in classes]
x = np.arange(len(classes)); w = 0.38
ax.bar(x - w/2, db_frac,     w, color="grey", label="IDRome")
ax.bar(x + w/2, subset_frac, w, color="C3",   label="subset")
ax.set_xticks(x); ax.set_xticklabels([c.replace("is_", "") for c in classes], rotation=20)
ax.set_ylabel("fraction"); ax.set_title("IDR class composition"); ax.legend()

fig.suptitle(
    f"IDRome stratified subset",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plot_out = os.path.join(OUT_DIR, "IDRome_subset_coverage.png")
fig.savefig(plot_out, dpi=140)
print(f"wrote {plot_out}")
