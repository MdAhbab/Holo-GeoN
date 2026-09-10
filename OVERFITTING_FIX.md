# OVERFITTING_FIX.md

Diagnosis and plan for the ΔΔG overfitting in `runs/main` (epoch 15 checkpoint,
commit `dbd1546`, dirty). Nothing here has been trained; every number below is
either read from an existing artifact or computed from the split tables and the
model definition in this working tree. Where I ran a calculation myself I say so
and give the command shape, so each one is reproducible.

**The headline, before the detail.** The train/test gap is real but it is not the
reason the model scores 0.758. Validation RMSE is flat from epoch 1 (0.6582) to
epoch 25 (0.6505) while training RMSE falls 0.6347 → 0.2590. The memorisation
that opens the gap is *orthogonal* to held-out performance: removing it recovers
the same held-out number, faster, with an honest σ. What limits the model to
0.758 is a separate and harder problem — 322 training proteins in 152 identity
clusters is not enough to learn a transferable site→effect map, and no
regulariser creates training proteins. Treat the gap as a calibration defect and
the 0.758 as a data/representation ceiling. They need different fixes and only
one of them is cheap.

---

## 1. Root cause

### 1.1 The lookup-key hypothesis: confirmed in substance, refined in three ways

The proposed mechanism is right. I confirm it and then sharpen it, because two
of the refinements change what the fix should be.

**Refinement 1 — the key is `(protein, site)`, not the row triple.** The context
vector `z` is a function of the parent protein and the site alone. Look at the
call path: `HoloGeoN.context` ([src/model.py:127](src/model.py#L127)) runs the
backbone and the readout, and neither receives `wt_idx` or `mt_idx`. The
substituted residue enters only at
[src/heads.py:88](src/heads.py#L88), through a 32-dimensional shared amino-acid
embedding and one shared MLP. So the memorisable channel has 17,071 distinct
values (the training sites), not 305,995. The substitution conditioning is a
global, low-capacity, genuinely transferable function — it is the part of the
architecture that is working, and it should not be the target of any fix.

**Refinement 2 — what is memorised is a per-site *profile*, not a per-site
scalar.** A per-site mean predictor reaches train r = 0.7910 (I reproduced this:
0.7910). The model reaches train r = 0.9538. Since the only per-site quantity
available to it is `z`, the surplus from 0.79 to 0.95 must come from `z`
encoding enough structure for the shared head to reconstruct the *ranking of the
19 mutants at that site*. Measured on the training split, within-site variance is
37.4% of the total, and the model is capturing most of it. So `z` is behaving as
a ~256-dimensional learned record per site, not a memorised mean.

**Refinement 3 — the "memorising is cheaper than generalising" comparison needs
restating.** The brief compares a train-split oracle (0.7910) against a test
score (0.7582). Those are not comparable. The like-for-like number is the
per-site oracle evaluated *on the test split*, which I measured at **0.7705**
(in-sample: it is handed the test labels). The conclusion survives — a predictor
that knew only site identity would beat the model — but the margin is 0.012, not
0.033.

### 1.2 Why the lookup is free, mechanistically

The precondition that makes this work is in the data path, not the loss. The
encoder is frozen and cached, and there is no augmentation anywhere in
[src/data.py](src/data.py) (I grepped for noise/jitter/augment/mask: the only
hits are padding masks and the split RNG). `MutationDataset.__getitem__`
([src/data.py:630](src/data.py#L630)) calls `store.get(row["parent_seq"])` and
builds the graph deterministically. Therefore **row *i* produces a bit-identical
`esm` tensor at every epoch of every run**.

That single property is the whole mechanism. The model is not fitting a function
on a distribution of inputs; it is fitting a table with 17,071 fixed, distinct,
never-perturbed keys, using 19.9M backbone parameters — about 1,170 parameters
per distinct key. The 1280-dimensional frozen ESM vectors at those sites are
already mutually separable, so the network does not have to *learn* to
distinguish sites; it only has to route them, and a near-arbitrary map suffices.

### 1.3 A second contributing cause the brief does not mention: the ClinVar task

This is the one place I would add to the diagnosis rather than refine it.

`RUN_STATE.md` §4 records that a gene-level positive-rate oracle reaches 0.952
val AUROC on ClinVar and that 99.8% of the AUROC pair mass is cross-gene. So the
*optimal* solution to the ClinVar objective is largely "encode which gene this
is". That task supplies **41.6%** of the training steps
(`task_sampling_probabilities`: stability 0.584, pathogenicity 0.416) at an
essentially equal loss weight (final weights: stability 0.1274, pathogenicity
0.1225 — a 1.04x difference). ClinVar train AUROC is 0.9998.

So roughly two-fifths of the gradient reaching the shared backbone comes from an
objective whose easiest descent direction is *protein-identity encoding* — which
is precisely the lookup channel that hurts the ΔΔG task. This is a plausible
active driver of the representation, it is cheap to test (intervention B2), and
nothing in the artifacts currently separates it.

### 1.4 What the numbers say the model is actually doing

I decomposed the ΔΔG variance three ways, on each split
(`groupby(parent_seq)` and `groupby([parent_seq, site])` on
`data/splits_megascale.parquet`):

| component | train | val | test |
|---|---|---|---|
| between protein | 0.108 | 0.092 | 0.079 |
| between site, within protein | 0.517 | 0.522 | 0.514 |
| within site (substitution identity + noise) | 0.374 | 0.385 | 0.406 |

The task is overwhelmingly "which site is this?" (51%) plus "which substitution
is this?" (37-41%). Only 8-11% is protein-level. That is why a model with a
per-site lookup does so well in-sample, and it is also why the ceiling analysis
in §4 is structured around a site term and a substitution term.

---

## 2. Why weight decay failed

Not because it was too small or wired wrongly. The parameter groups at
[scripts/train.py:889-905](scripts/train.py#L889-L905) are correct: decay on
`p.ndim >= 2` only, 0.0 on norms/biases and on the Kendall & Gal log-variances.
And 1e-2 is not decorative — at lr 3e-4 the per-step decoupled shrink is
`lr*wd = 3e-6`, which compounds to `exp(-3e-6 * 204,625) ≈ 0.54` over the run for
a parameter no gradient opposes. It genuinely halves unopposed weights. It still
changed nothing.

Three mechanisms, in increasing order of importance.

**(a) L2 penalises magnitude; the lookup does not need magnitude.** Separating
17,071 points that are *already separated in the frozen input space* requires no
large weights — a random projection into 256 dimensions already does it, and
shrinking every weight matrix by 46% shrinks the whole representation
proportionally, leaving the routing intact. Weight decay constrains the scale
and smoothness of the learned map. Memorisation here is a property of the
map's *injectivity on a fixed finite input set*, which is scale-invariant. The
regulariser and the pathology are close to orthogonal.

**(b) No parameter-space regulariser can break a lookup over a fixed input
set.** Following from §1.2: because the cached embedding for row *i* is
byte-identical every epoch, there is no input distribution over which a
smoothness penalty could bite. Weight decay can make the table cheaper to
represent; it cannot make the keys collide.

**(c) The decisive one — the heteroscedastic NLL makes the memorisation
gradient grow without bound, while the decay force stays constant.** From
[src/losses.py:47](src/losses.py#L47), the Gaussian NLL's gradient with respect
to the mean is

```
d/dmu  0.5 * (y - mu)^2 * exp(-logvar)   =   (mu - y) / sigma^2
```

so the pull toward fitting a training residual is amplified by `1/sigma^2`.
Mean predicted σ falls monotonically across the run — 0.5115 (ep1) → 0.2581
(ep15) → 0.1999 (ep25) — so the effective gradient on the mean channel is
**6.5x larger at epoch 25 than at epoch 1**, purely from σ collapse. Weight
decay contributes a fixed `lr*wd*p`. A constant restoring force cannot hold
against a driving force that grows quadratically as the model becomes confident.

Two corroborations of (c) from the recorded history. First, the σ/error ratio is
the quantity that moves monotonically and never saturates: 1.29 → 3.25 over 25
epochs, while val Pearson is flat after epoch 6. Second, the 100x weight-decay
change left that ratio *identical* at matched epochs (ep5 1.79x vs 1.82x; ep6
1.92x vs 1.91x; ep8 2.15x vs 2.06x). A 100x change in a regulariser that moved
nothing is itself strong evidence that the regulariser was never the binding
constraint.

There is also a fourth, smaller factor worth recording: the Kendall & Gal
weighting cannot compensate, because it is arithmetically uniform. With
`GLOBAL_LOSS_FLOOR = -4.0811` ([src/losses.py:116](src/losses.py#L116)) and task
losses in the 0.03-0.59 range, the stationary weight `0.5/(L - floor)` varies
only over `0.5/4.11` to `0.5/4.67`. The measured final weights span
0.1117-0.1274, a **1.14x** range. The multi-task objective is a fixed uniform
weight wearing a learned-weighting costume.

---

## 3. Ranked interventions

Cost basis: `runs/main/metrics.json` records `wall_clock_hours: 6.80` for 25
epochs = **16.3 min/epoch**. A 12-epoch run is 3.3h; an 8-epoch run is 2.2h.
GPU is shared, so these are wall-clock upper bounds.

The column that matters most is the last one. Several of these reduce the gap
and will not move test performance, and I have marked them so.

| # | intervention | Δ gap | Δ test Pearson | cost | improves test? |
|---|---|---|---|---|---|
| A1 | stop earlier + let the LR anneal | 2.05 → ~1.6 | −0.005 to +0.008 | **saves 3.5h** | **no — gap only** |
| A2 | run the existing ablation grid | none | none (diagnostic) | 19.6h | no — measurement |
| B1 | Gaussian noise on cached embeddings | 2.05 → 1.4-1.7 | −0.005 to **+0.020** | 10h (3 runs) | **plausibly yes** |
| B2 | drop / downweight the ClinVar task | 2.05 → ~1.8 | −0.010 to **+0.020** | 3.3h | **plausibly yes** |
| B3 | mask `z_site` in the readout | 2.05 → ~1.5 | −0.010 to +0.010 | 6.6h | **probably not — gap only** |
| C1 | shrink the cross-attention fusion | 2.05 → ~1.9 | −0.005 to +0.005 | 3.3h + edit | **no — capacity hygiene** |
| C2 | `hidden`/`out_dim` 256 → 128 | ~1.95 | ±0.005 | 3.3h | **no — and see the caveat** |
| D1 | k-fold over the 171 non-test clusters | n/a | +0.000 to +0.010 | 16.5h | marginally |
| D2 | unfreeze the encoder | worse | unpredictable, likely worse | 3h rebuild + 10-30x/epoch | **no — do not** |

### A1. Stop earlier, and let the learning rate actually anneal — do this first

**Change.** [configs/run.yaml:104](configs/run.yaml#L104) `epochs: 33` → `12`;
[configs/run.yaml:118](configs/run.yaml#L118) `early_stop_patience: 10` → `4`.

**Why it is first.** It costs negative GPU hours and it halves the price of every
experiment below.

Two defects are fixed at once. The obvious one: val Pearson saturates at epoch 6
(0.7532) and the best epoch's 0.7616 is +0.0084 over nine further epochs, well
inside the ±0.035 cluster CI. Those nine epochs buy nothing measurable and take
the gap ratio from 1.56 to 2.05 and σ/error from 1.91x to 2.49x.

The subtler one is the schedule. `total_steps = steps_per_epoch * cfg.epochs`
([scripts/train.py:941](scripts/train.py#L941)) assumes all 33 epochs, so with
early stopping the cosine never reaches its tail. I evaluated `lr_at`
([scripts/train.py:960](scripts/train.py#L960)) at the relevant steps:
`steps_per_epoch = 8185`, `total_steps = 270,105`, `warmup = 13,505`.

| run | best epoch | LR at that epoch | fraction of peak |
|---|---|---|---|
| `runs/baseline_wd1e-4` | 8 | 2.71e-4 | **90%** |
| `runs/main` | 15 | 1.85e-4 | **62%** |
| `runs/main` (last epoch run) | 25 | 4.57e-5 | 15% |

(The brief's "~90% of peak" is exact — for the `baseline_wd1e-4` run's best
epoch. For the shipped checkpoint it is 62%.) Both selected checkpoints were
taken mid-schedule at a high learning rate, which is where a model is noisiest.
Setting `epochs: 12` makes the anneal complete inside the budget actually run.

**Predicted.** Gap ratio 2.05 → ~1.6; σ/error 2.49x → ~1.9x; test Pearson within
±0.008 of 0.758. **This reduces the gap without improving test performance.**
The gain is 3.5h per run and an honest checkpoint, not accuracy.

**Risk.** Very low. Worst case the completed anneal is worth a little and
`epochs: 12` slightly under-trains; `epochs: 16` is the conservative variant.

### A2. Run the ablation grid that already exists — the decisive measurement

**Change.** None to source. `python run.py ablations --seeds 42 43 44 --epochs 8
--arms full no_mech_stream global_pool_readout no_geo_stream`.

This is not a fix; it is the experiment that answers "is the graph earning its
complexity?", and it has been implemented and never run
([scripts/ablations.py:76](scripts/ablations.py#L76) defines five arms with
paired-across-seeds testing and Holm correction).

**My prediction, stated in advance so it can be wrong.** Every arm will land
within ±0.010 Pearson of `full`, and none will survive Holm correction at three
seeds. The basis: the matched-capacity MLP is within 0.0045 Pearson of the full
model and *beats* it on RMSE (0.6415 vs 0.6482) and R² (0.5550 vs 0.5457). That
MLP differs from the full model in the graph, the readout, the SSM, the fusion
and the mechanistic channels *simultaneously*. If the joint contribution of all
five is ≈0, no individual component can be large unless two are large and
cancelling. If any arm shows >0.02, my §1 analysis is wrong and should be
revisited.

One thing this grid *can* now measure honestly: `encoder_cache_measured` in
`metrics.json` reports `cds_active: 64/64`, `cds_coverage: 1.0`, so E_mech is
genuinely populated for this run and `no_mech_stream` is a meaningful arm rather
than a plumbing artifact.

**Cost.** 4 arms × 3 seeds × 2.2h = 26h; drop to 3 arms for 19.6h. Note
`no_ssm` comes nearly free as a capacity experiment (see C3).

### B1. Gaussian noise on the cached embeddings — the only fix aimed at §1.2

**Change.** New config key `embed_noise_std: 0.0`, applied *training-split only*
in `MutationDataset.__getitem__` ([src/data.py:630](src/data.py#L630)) or in
`build_sample_graph` ([src/data.py:522](src/data.py#L522)):

```python
if self.train and self.noise_std > 0:
    g["esm"] = g["esm"] + self.noise_std * torch.randn_like(g["esm"])
```

`MutationDataset` needs a `train: bool` flag; `build_split_loaders`
([scripts/train.py:198](scripts/train.py#L198)) already knows which split it is
constructing.

**Why this one and not more dropout.** It is the only proposal that attacks the
precondition rather than the symptom: it makes row *i*'s input differ between
epochs, so the fixed-key table stops being a table. Everything else on this list
constrains the map while leaving the keys pristine.

**Scale, measured rather than guessed.** I opened two cache records: per-dimension
standard deviation *across residues* is 0.196 and 0.165, and the global
element-wise std is ~0.27. So `0.10` is roughly half the across-residue
variation — substantial but not destructive. Sweep `0.05 / 0.10 / 0.20`.

**Antisymmetry is safe.** The noise is drawn once per sample, before the
backbone, so `z` is a single draw shared by both `s()` calls in
[src/heads.py:114](src/heads.py#L114). The difference `s(z,e_mt) - s(z,e_wt)`
remains exact.

**Predicted.** Gap 2.05 → 1.4-1.7. Test Pearson −0.005 to +0.020; I would bet on
+0.005 to +0.010 at σ=0.10. This is the intervention with real upside, because
it is the only one that raises the *effective* number of distinct training
inputs. **Genuinely uncertain** — if the site's own embedding is the dominant
signal, noise costs more than it buys.

**Cost.** 3 runs × 3.3h = 10h. **Risk.** Moderate; needs the sweep, and σ=0.20
may well be harmful.

### B2. Train stability-only, or downweight ClinVar

**Change.** No source edit needed: point `--pathogenicity-splits` at a
nonexistent path, and `build_split_loaders`
([scripts/train.py:198](scripts/train.py#L198)) returns empty, dropping the task.
A softer variant is `task_sampling_temperature: 0.5` → `0.2`
([configs/run.yaml:133](configs/run.yaml#L133)), which cuts ClinVar's share of
steps.

**Rationale.** §1.3. Two-fifths of the backbone's gradient currently comes from
an objective best solved by encoding protein identity.

**Predicted.** Gap 2.05 → ~1.8. Test Pearson −0.010 to +0.020. **This is the
cheapest test of a hypothesis nothing else on the list tests**, and the outcome
is informative in both directions: if ΔΔG test performance *rises* when ClinVar
is removed, the multi-task design is costing the primary task and that is a
significant finding for the write-up. If it does not move, §1.3 is wrong and the
lookup is driven by the ΔΔG objective alone.

**Cost.** 1 run, 3.3h at `epochs: 12`. **Risk.** Low, but note it forfeits the
ClinVar head for that run, so do not use that checkpoint for anything else.

### B3. Mask the site's own embedding in the readout

**Change.** In `SiteAnchoredReadout.forward`
([src/readout.py:119](src/readout.py#L119)), with probability *p* during
training, replace `z_site` with a learned mask vector before `fuse`.

This removes the most direct site-identity channel and forces the neighbourhood
and global pools to carry the site effect. Antisymmetry is preserved for the
same reason as B1 — `z` is computed once per row.

**Predicted.** Gap 2.05 → ~1.5, the largest gap reduction on the list. Test
Pearson −0.010 to +0.010. **I expect this to reduce the gap without improving
test performance**, because `z_site` is genuinely informative: it carries the
residue identity and its local ESM context, which is real signal, not just an
index. Included because it cleanly separates "how much of `z` is index" from
"how much is content" — if test performance is unchanged at p=0.3, most of
`z_site` was serving as an index.

**Cost.** 2 runs (p = 0.1, 0.3) × 3.3h = 6.6h. **Risk.** Moderate.

### C1. Shrink the cross-attention fusion — 6.57M parameters buying a rank-32 correction

I instantiated the model and counted parameters per submodule. The distribution
is not what the module names suggest:

| submodule | parameters | share |
|---|---|---|
| `backbone.fusion` (cross-attention) | 6,572,800 | 31.9% |
| `backbone.ssm` | 6,563,840 | 31.8% |
| `backbone.geo_stream` | 3,174,912 | 15.4% |
| `backbone.mech_stream` | 3,174,912 | 15.4% |
| `backbone.residual_proj` | 327,680 | 1.6% |
| `readout` | 329,730 | 1.6% |
| all six heads + AA embedding | 334,090 | 1.6% |
| **total** | **20,607,788** | |

**The finding.** In `CrossAttentionFusion`
([src/sequence_mixers.py:65](src/sequence_mixers.py#L65)), keys and values come
from `mech_proj(mech)`, an 8 → 1280 linear map of the 8 mechanistic channels.
The value vectors therefore lie in an ≤8-dimensional affine subspace; attention
output is a convex combination of them, so it stays in that subspace; with 4
heads each contributing ≤8 dimensions, and `out_proj` being linear, the
per-residue correction is confined to a **32-dimensional** subspace of R¹²⁸⁰.

I verified this numerically rather than trusting the argument. In float64, over
random inputs: `mech_kv` rank 8, `V = v_proj(mech_kv)` rank 8, `attn_out` rank
**exactly 32** with top-36 singular energy 1.000000. (In float32 the same
measurement reports full rank with top-20 energy 0.798 — that is round-off
spreading energy across directions, not expressiveness. Worth knowing, because
the naive float32 check refutes a true claim.)

So 6.56M parameters — 31.9% of the model — produce at most a rank-32 per-residue
correction, whose magnitude at initialisation is 8.8% of the ESM signal norm.
The `mech_proj` that feeds the whole module is 11,520 parameters. The q/k
projections (4.9M) do buy expressiveness in *which* mixture each residue
receives, but the space that mixture lives in is 32-dimensional regardless.

**Change.** Replace `nn.MultiheadAttention` with a low-rank gated projection
(e.g. 1280 → 64 → 1280 modulated by mech-derived gates), removing ~6.3M
parameters without reducing the expressible output space.

**Predicted.** Gap 2.05 → ~1.9; test Pearson ±0.005. **This is capacity hygiene,
not a performance fix** — but it removes 31% of the parameters available for
memorisation at essentially no representational cost, which makes every other
experiment cleaner.

**Cost.** An architecture edit plus 3.3h. **Risk.** Low on expressiveness
grounds; it is still a source change, so it invalidates checkpoint
compatibility. Note `no_mech_stream` in A2 covers a related question more
cheaply — run A2 first.

### C2. Halving `hidden` / `out_dim` — the obvious knob barely moves

**Change.** [configs/run.yaml:75-76](configs/run.yaml#L75-L76) `hidden: 256` →
`128`, `out_dim: 256` → `128`.

**The caveat, which is the reason this entry exists.** I measured it: this takes
the total from 20,607,788 to **16,421,996** — a 20% cut, not the ~4x someone
would expect from halving a width. The reason is the table above: 63.7% of the
parameters sit in the SSM and the fusion, both of which operate on the
1280-dimensional ESM axis and *do not scale with `hidden` at all*. Recording this
so nobody spends a run on it expecting a large change.

**The knob that does work** is `use_ssm`
([configs/run.yaml:80](configs/run.yaml#L80)): `true` → `false` takes the model
to **14,043,948**, a 32% cut, and it is already an ablation arm, so A2 measures
it for free.

**Predicted.** Gap ~1.95; test ±0.005. **Gap only.** **Cost.** 3.3h.

### D1. k-fold over the non-test clusters

`cv_folds: 5` exists at [configs/run.yaml:58](configs/run.yaml#L58) and **has no
consumer**. I grepped `src/`, `scripts/`, `tests/` and `run.py`: the only hits
are the default in [src/config.py:66](src/config.py#L66) and a test fixture in
`tests/test_pipeline.py`. This must be built, not enabled.

The gain is modest and worth stating honestly: folding the 53 val proteins into
training gives 375 proteins instead of 322, a 16% increase, and it is the only
cheap increase in training proteins available (see §4). The larger benefit is
statistical — five folds give a variance estimate rather than a point estimate,
which §5 argues this project badly needs.

**Predicted.** +0.000 to +0.010 test Pearson for the final model. **Cost.**
5 × 3.3h = 16.5h plus implementation. **Risk.** Low, but it changes what "val"
means, so the early-stopping path needs care.

### D2. Unfreezing the encoder — the explicit cost

Stating this plainly because the brief asks for it. Unfreezing ESM-2 650M:

- **Invalidates the cache.** 25 GB, 15,733 records at
  `C:/Users/User/AppData/Local/holo-geon/cache/esm2_650M`, ~3h to rebuild — and
  the rebuild is pointless, because a trainable encoder cannot be cached at all.
- **Destroys the training economics.** Every batch would need an ESM-2 650M
  forward *and backward* pass instead of a cache read. The 16.3 min epoch
  becomes hours. A 12-epoch run stops being a 3.3h experiment.
- **Would very likely make overfitting worse.** Adding 650M trainable parameters
  to a problem with 152 independent training clusters moves in exactly the wrong
  direction on §1.

If it is attempted anyway, the only defensible form is unfreezing the last 2-4
transformer blocks at a 10-100x lower learning rate than the head, with the cache
abandoned. I do not recommend it, and I would run everything in tiers A and B
first.

---

## 4. The honest ceiling

The useful question is not "how far is train from test" but "what is the best a
predictor of this shape could do on 37 unseen proteins". I measured the ladder
rather than asserting it. All numbers are on the test split.

| predictor | test Pearson | what it is given |
|---|---|---|
| measurement-noise limit | ~0.999 | σ median 0.051 vs target σ 0.962 |
| **additive ceiling: site effect + substitution effect** | **0.871** (0.86 adjusted) | fitted on test labels, 2,220 free parameters |
| per-site oracle, estimated from 18 siblings, + train substitution prior | 0.795 | test labels of the *other* mutants at each site |
| **perfect site knowledge alone** | **0.771** | fitted per-site means on test labels |
| **full model** | **0.758** | nothing |
| matched-capacity MLP | 0.754 | nothing |
| per-site oracle from 18 siblings, alone | 0.738 | test labels of the other mutants |
| substitution matrix from train | 0.555 | train labels only |
| ESM-2 zero-shot | 0.400 | nothing |

Method: the additive fit alternates site-effect and substitution-effect means to
convergence (20 sweeps) on the test split; in-sample R² is 0.7583 with 1,840 site
plus 380 substitution parameters on 33,226 rows, so adjusting for the fitted
degrees of freedom gives R²_adj ≈ 0.741, i.e. **r ≈ 0.86**. The substitution-prior
row reproduces `baselines.json`'s `substitution_frequency` exactly (0.5546 /
RMSE 0.8005), which is a useful check that my pipeline matches theirs.

**Three conclusions follow, and the first is the non-obvious one.**

**The ceiling is not 0.76.** There is roughly **0.10 Pearson of headroom** inside
a purely additive site-plus-substitution model, and the model does not reach it.
It is easy to read "matched MLP at 0.7537" as proof that 0.75 is the limit of
frozen embeddings. The additive analysis says otherwise: a predictor that knew
each test site's true effect and a global substitution matrix would score ~0.86.
The model and the MLP both fall well short of that, which means the binding
constraint is the accuracy of the *predicted site effect on unseen proteins*.

**Measurement noise is not the limiter.** Median per-row σ is 0.051 kcal/mol
against a test target spread of 0.962, so noise alone caps r at ~0.999. The
caveat from
[scripts/build_megascale_table.py:78-84](scripts/build_megascale_table.py#L78-L84)
applies — that σ is the authors' ML-fit confidence interval, not replicate
reproducibility, so true noise is larger — but it would have to be about ten
times larger to bind at 0.86.

**Beyond ~0.86 requires modelling site×substitution interaction**, which is 24.2%
of the test variance (the additive fit's residual). Nothing in the current
architecture is aimed at that specifically; the head is fully interacting through
an MLP but is fed a `z` that, per §1, is being used as an index.

### What would actually raise the number

*More independent proteins — and this is harder than it sounds.* MegaScale is
**exhausted**: 322 + 53 + 37 = 412 domains, all of them in use. There is no
held-back portion. The realistic options, ranked by honesty:

- k-fold to recover the 53 val proteins (D1): +16% proteins, worth ~+0.005.
- FireProtDB's 176 non-MegaScale proteins are already committed as the external
  transfer set. Merging them into training would destroy the only independent
  measurement this project has (r = 0.4518), and I would not do it.
- ProTherm / ThermoMutDB add proteins but are heterogeneous in assay and
  condition, and are largely the same classical mutations FireProtDB draws from.
- ProteinGym-scale DMS data is far larger but measures fitness, not ΔΔG, so it
  is a different target requiring a different head.

The blunt version: **within ΔΔG there is no large untapped protein source**, and
that is the single most important constraint on this project. 322 proteins is
the budget.

*A better site representation.* Since the limiting quantity is "predict this
site's effect on an unseen protein", and 51% of the variance is between-site, the
highest-value representational change is anything that improves site
characterisation — true structural features (the repo already has 373 true
contact maps under `data/true_contacts/`, currently used only for the
`contact_sanity` check at precision@L 0.729), relative solvent accessibility,
or explicit burial. This is speculative, and I would only pursue it after A2
reports whether the *predicted*-contact graph does anything at all.

*Realistic expectation.* With everything in tiers A-C: **0.76-0.79**. Reaching
0.82+ would need substantially more proteins or a materially better site
representation. I would not promise 0.86 to anyone.

### Is the graph earning its complexity?

Treated as open, per the brief. My assessment: **on this benchmark, no — and the
benchmark cannot tell you whether it would elsewhere.**

Against the graph: the matched-capacity MLP is within 0.0045 Pearson and better
on both error metrics, and it falls inside the full model's cluster CI. The
graph, readout, SSM, fusion and mechanistic channels together contribute ≈0.

The nuance that keeps it open: every MegaScale protein is 32-72 residues
(`window_coverage.fraction_protein_shorter_than_window: 1.0`, median length 56),
so the window always spans the entire protein and a mean-pooled window plus the
site embedding already sees everything the graph sees. On domains this short a
2-hop contact neighbourhood and a global pool are nearly the same object. The
architecture's premise — that spatial structure beats sequence proximity — is
barely testable on 56-residue domains. ClinVar proteins have median length 878,
but that task is degenerate for the separate reason in §1.3.

So the fair statement for a write-up: *measured on MegaScale, the graph
contributes approximately nothing over a parameter-matched MLP, and MegaScale's
short domains are a weak test of the claim.* A2 is what turns this from an
inference into a per-component measurement.

---

## 5. Validation protocol, with thresholds fixed in advance

### 5.1 The independent-unit problem is worse than 37

The test split has 37 proteins but only **19 MMseqs clusters** (`group_used`).
Worse, they are extremely unbalanced — I measured the cluster sizes:

```
proteins per cluster:  8, 7, 4, 2, 2, 1×14
rows per cluster:      5915, 5525, 4778, 2334, 1422, then 14 clusters of 760-1346
```

The three largest clusters hold 19 of 37 proteins and **49% of all test rows**.
The effective number of independent units is nearer 10-12 than 37.

`scripts/overfit_report.py` resamples `parent_seq`
([scripts/overfit_report.py:155](scripts/overfit_report.py#L155)), i.e. 37 units,
not the 19 clusters. I measured the design effect using a proxy predictor (the
sibling-estimated site oracle, test r = 0.7950) with 1,000 bootstrap replicates:

| resampling unit | bootstrap SD | 95% CI half-width |
|---|---|---|
| rows | 0.0026 | 0.0049 |
| proteins (what the tool does) | 0.0089 | 0.0169 |
| clusters (what it should do) | 0.0097 | 0.0191 |

So the row-level interval is **3.4x too narrow**, and the existing protein-level
bootstrap is *close to right* — cluster resampling widens it by only ~10%. Credit
where due: the protein-level choice was the important one and it was made
correctly. **Fix:** change `groups = frame["parent_seq"].to_numpy()` to
`frame["group_used"].to_numpy()`. For the model itself, whose per-protein errors
are more heterogeneous than the proxy's, expect the ±0.035 protein CI to become
roughly ±0.040.

### 5.2 Run-to-run noise exceeds almost every effect on the list

Round-1 review measured two identical commands differing by **0.030 test
Pearson**, which is why seeding was added. That is larger than the predicted
effect of every intervention in §3 except possibly B1. **A single-run comparison
cannot resolve anything proposed here.** This is the most important line in the
protocol.

Therefore: **≥3 seeds, paired, for every claim.** `scripts/ablations.py` already
implements paired-across-seeds testing with Holm correction over a declared
metric family and refuses to test below `MIN_SEEDS_FOR_TEST`; reuse that
machinery rather than eyeballing point estimates.

### 5.3 The pre-registered decision rule

Declare the family before running: **{test Pearson, test RMSE, gap ratio,
σ/error ratio}**. Four metrics, Holm-corrected, nothing else compared.

Primary metric: **test Pearson**, paired across ≥3 shared seeds, with a cluster
bootstrap on `group_used`.

| verdict | rule |
|---|---|
| **adopt** | paired mean ΔPearson ≥ **+0.010** and the paired 95% interval excludes 0 |
| **no effect** | \|ΔPearson\| < 0.010, or the interval includes 0 |
| **harmful** | paired mean ΔPearson ≤ **−0.010** with the interval excluding 0 |

A "no effect" result is not a failure and does not mean the change is worthless:
adopt it anyway if it reduces cost (A1), reduces capacity at no cost (C1), or
improves calibration. Say which reason applies.

Secondary thresholds, on the gap itself:

- **gap ratio** (val RMSE / train RMSE, already logged as
  `generalisation_gap_rmse`): currently 2.05 at the selected epoch. Target
  **≤ 1.3**.
- **σ/error ratio** (`rmse_over_predicted_sigma`): currently 2.49 val / 2.50
  test. Target **≤ 1.3**. This is the metric that matters for the product, not
  just the diagnosis: `runs/main/calibration.json` records
  `well_calibrated: false`, ECE 0.117, and 50% intervals covering only 24.3% —
  the interface needs a post-hoc variance scale of 2.385x to draw an honest
  band. A fix that brings this near 1.0 is worth shipping *even at ΔPearson = 0*,
  and A1 alone should get most of the way.

### 5.4 The one diagnostic to run before anything else

Cheap, decisive, and not currently produced by any script: **decompose the test
error into between-site and within-site components.** Save per-row test
predictions (one inference pass, ~33k rows) and compute:

1. Pearson between the model's per-site mean prediction and the true per-site
   mean, over the 1,840 test sites. *This is the model's site-effect accuracy —
   the quantity §4 identifies as binding.*
2. Pearson of the within-site residuals, model vs truth. *This is its
   substitution-effect accuracy given the site.*

The additive ceiling of 0.86 says quantity 1 should be the weak one. If instead
quantity 1 is high and quantity 2 is low, my §4 analysis is wrong and the effort
should go to the head rather than to site representation. **Nothing else on this
list changes direction based on a single measurement this cheap, so run it
first.**

### 5.5 Always report the external number alongside the test number

Test Pearson 0.7582 and FireProtDB 0.4518 disagree by 0.31, with RMSE nearly
tripling (0.648 → 1.664). Quoting the test figure alone overstates transfer, as
`RUN_STATE.md` §4 already says. Any evaluation of a fix should report both; a
change that improves the test number while the external number falls has made
the model more MegaScale-specific, not better.

---

## 6. What not to do

**Do not raise weight decay further.** 100x moved nothing and matched the σ/error
trajectory at every epoch. §2 explains why the binding constraint is not weight
norm. A third value on this axis is a wasted 3.3h.

**Do not add dropout inside the ΔΔG or three-state heads.** `_DETERMINISTIC =
0.0` at [src/heads.py:43](src/heads.py#L43) is load-bearing, not an oversight.
An independent mask per `score()` call would make `s(z, e_wt)` differ between the
forward and reverse directions, so exact antisymmetry would hold only in eval
mode — trading a structural guarantee for a regulariser that §1 says would not
work anyway. The comment there is correct and should be left alone. Dropout in
the readout is fine, for the reason the brief gives: `z` is computed once and
cancels in the difference.

**Do not add reverse-mutation augmentation.** With this head it is a
mathematical no-op: `ddG(mt→wt) ≡ −ddG(wt→mt)` by construction, so a swapped pair
with a negated label yields an identical loss and identical gradients. The
docstring at the top of [scripts/train.py](scripts/train.py) already says this
and is right. It would cost epoch time and regularise nothing.

**Do not optimise the gap as an objective.** The gap can be driven to ~1.0 by
crippling the model, and that would not raise test performance by a single point.
The evidence is in the history: val RMSE 0.6582 at epoch 1 and 0.6505 at epoch 25
while train RMSE fell 0.6347 → 0.2590. The memorisation is orthogonal to
held-out performance. Report the gap as a calibration defect, and judge fixes on
test Pearson and σ/error.

**Do not use row-level bootstrap intervals.** Measured 3.4x too narrow (§5.1).
Every difference discussed in this document would look significant.

**Do not compare single runs.** Run-to-run noise is 0.030 Pearson (§5.2), larger
than nearly every effect proposed. This is the single most common way the
remaining work could produce a confident wrong conclusion.

**Do not merge FireProtDB into training.** It is the only independent transfer
measurement, and its 0.4518 is the number that keeps the test figure honest.

**Do not increase model capacity or train longer.** Both are the wrong direction
on every diagnostic here: 63.7% of the parameters already sit in two modules
whose contribution is unmeasured, one of which (§C1) provably emits a rank-32
correction, and val performance has been flat since epoch 6.

**Do not read A2's likely "no significant effect" as vindication.** If the
ablation grid returns null deltas at three seeds, that is evidence the components
contribute little, *or* that three seeds cannot resolve them —
`scripts/ablations.py` is careful about this distinction in its own output
(`alpha=0.05; null means too few seeds to test, not 'no effect'`) and any
write-up must be too.

**Do not describe the ClinVar AUROC as within-protein variant ranking.** A
gene-level positive-rate oracle reaches 0.952 val AUROC and 99.8% of the pair
mass is cross-gene (`RUN_STATE.md` §4). The model has saturated a gene-level
task. This is not an overfitting fix, but it is the claim most likely to be
overstated in a write-up that follows this one.

---

## Recommended order

1. **§5.4 diagnostic** — one inference pass. Decides whether §4's framing holds.
2. **A1** — config only, saves 3.5h per run, halves the cost of everything below.
3. **B2** — 3.3h, tests the ClinVar-contamination hypothesis from §1.3.
4. **B1** — 10h, the only intervention with real upside on test performance.
5. **A2** — 19.6h, settles "is the graph earning its complexity" per component.
6. **C1 / C3** — capacity hygiene once A2 says what is safe to remove.
7. **D1** — only if the 16% protein increase is worth 16.5h.

Steps 1-3 are 7h of GPU time and will resolve most of the open questions. Do not
start D2.
