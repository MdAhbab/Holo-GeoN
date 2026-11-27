"""
scripts/figures
===============
Every paper figure and table is produced by code in this package, from released
artifacts only (Holo-StabBench parquet / JSON), so each number in the paper traces
to a file in the Zenodo release and rebuilds without a GPU.

    python -m scripts.make_figures --release <Holo-StabBench dir> --results <results dir>

Modules
-------
style         palette, rcParams, small drawing helpers, saving + provenance
loaders       artifact discovery / readers with fail-fast messages
audit_figs    F2 fold-leakage figure, per-split leakage table, T1 inventory
result_figs   F3-F9 and T2-T5 from scorer / ablation / calibration / epistasis outputs
"""
