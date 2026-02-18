# TIDE: Tree Inference for Dryland Ecosystems

Maps annual tree cover states across US CONUS drylands (1986–2024) using a 6-state Zero-Inflated Beta Hidden Markov Model. Woody encroachment is pushing open rangelands toward higher tree cover from below; drought-driven dieback in pinyon-juniper woodlands and fire-driven contraction in montane forests are reducing cover from above. TIDE produces spatially explicit, temporally coherent classifications of both directions of this structural shift at 30m resolution.

**[SCIENCE.md](SCIENCE.md)** — model formulation, emission distributions, covariate design, ecological rationale
**[ENGINEERING.md](ENGINEERING.md)** — architecture, I/O pipeline, GPU performance, API reference

---

## Model

| | |
|---|---|
| States (K=6) | Bare · Trace · Sparse · Open · Woodland · Forest |
| Emissions | Zero-Inflated Beta per state |
| Transitions | Static or covariate-driven (fire severity, SPEI-12, elevation, CHILI, aridity, seed distance, focal mean cover) |
| Years (T=39) | 1986–2024 |
| Domain | ~4.9B pixels, CONUS drylands wildlands (NLCD-masked) |
| Resolution | 30m (RAP tree cover) |

### State definitions

| State | Cover range | Ecological description |
|-------|------------|------------------------|
| 0 Bare | ~0% | Rangeland baseline, no woody cover |
| 1 Trace | 0–4% | First tree establishment |
| 2 Sparse | 4–10% | Active early encroachment |
| 3 Open | 10–25% | Advancing encroachment / post-fire partial recovery |
| 4 Woodland | 25–50% | Open woodland (pinyon-juniper, oak) |
| 5 Forest | 50%+ | Closed canopy; dieback and fire source |

---

## Outputs

```
output/
  states/
    states_1986.tif ... states_2024.tif   # 1-band uint8 per year (~2 GB each)
  posteriors/
    posteriors_1986.tif ... posteriors_2024.tif  # 6-band uint8 per year (~12 GB each)
  trace_onset_year.tif   # int16 — first year state ≥ 1
  sparse_year.tif        # int16 — first year state ≥ 2
  open_year.tif          # int16 — first year state ≥ 3
  woodland_year.tif      # int16 — first year state ≥ 4
  forest_year.tif        # int16 — first year state ≥ 5
  max_state.tif          # uint8 — maximum state ever reached
```

Posterior bands: `band k+1 = P(state=k) × 200`, stored as uint8.
NoData: 255 (uint8), -9999 (int16).

---

## Data requirements

| Input | Source | Notes |
|-------|--------|-------|
| Tree cover | RAP v3 (USDA) | Annual, 30m, uint8 (0–100%) |
| Fire severity | MTBS (USDA/USGS) | Annual, ordinal 0–4 |
| Drought (SPEI-12) | gridMET | Annual |
| Elevation | NED 30m | Static |
| CHILI insolation | CSP-ERGo | Static |
| Aridity index | UNEP/TerraClimate | Static |
| Analysis mask | NLCD | Excludes urban, water, cropland |

All inputs mosaicked to CONUS extent and stored in `data/`.
See `scripts/mosaic_gee_tiles.py` for GEE export → mosaic workflow.

---

## Install

```bash
conda activate zibhmm          # Python 3.12, JAX + CUDA 12
pip install -e .
pip install -e ".[monitoring]" # optional: GPU utilization in logs
```

---

## Usage

```bash
# Full run: EM parameter learning + inference
tide run --data-dir data/

# Reuse existing parameters (skip EM)
tide run --data-dir data/ --skip-em --params-file params.npz

# Covariate-driven dynamic transitions
tide run --data-dir data/ --dynamic-transitions

# Key options
#   --n-states 6            HMM states (default 6)
#   --chunk-size 512        Tile dimension in pixels (default 512)
#   --em-sample 500000      Pixels sampled for EM
#   --no-posteriors         Skip posterior output (faster, smaller)
#   --dask --workers N      Multi-GPU via Dask
```

### Pre-chunked mode (6x faster inference)

For production runs, a 3-phase pipeline pre-converts GeoTIFFs to `.npy` files, reducing per-chunk I/O from ~200ms to ~30ms:

```bash
# Phase 0: Pre-chunk inputs (one-time, ~14 min with 6 threads)
tide prechunk --data-dir data --chunks-dir data/chunks --threads 6

# Phase 1: Inference (~10 min vs ~63 min)
tide run --data-dir data --skip-em --params-file params.npz \
    --prechunked --chunks-dir data/chunks

# Phase 2: Reassemble into GeoTIFFs
tide assemble --chunks-dir data/chunks --output-dir output --data-dir data
```

Add `--dynamic-transitions` to both `prechunk` and `run` for covariate-driven transitions.

---

## Performance

**Pre-chunked mode** (recommended for production):

| Phase | Time | Notes |
|-------|------|-------|
| Pre-chunk | ~14 min | One-time, 6 threads, ~30 chunks/s |
| Inference | ~10 min (est.) | .npy read 2ms + GPU 6ms + .npy write 25ms |
| Assemble | ~30 min (est.) | One-time GeoTIFF writes |

**Single-pass mode** (GeoTIFF I/O, chunk=512, stacked treeCover):

| Stage | Cost per chunk |
|-------|---------------|
| I/O (read + decompress) | ~50 ms |
| GPU inference (Viterbi + ZIB) | ~6 ms |
| Write (deflate compress) | ~151 ms |

**CONUS**: 25,130 chunks, 4.9B valid pixels. Pre-chunked mode is ~6x faster for inference.

---

## Tests

```bash
pytest tests/   # 62 tests (~17 min including JAX compilation)
```
