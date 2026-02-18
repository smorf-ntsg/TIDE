# TIDE Scientific Methodology: Zero-Inflated Beta HMM for Drylands Tree Cover Classification

## Summary

This document describes a Hidden Markov Model (HMM) framework for classifying tree cover dynamics in US CONUS drylands from 1986-2024 using annual remote sensing observations. The model uses a **Zero-Inflated Beta (ZIB) emission distribution** to handle the unique statistical properties of fractional cover data, learns parameters via **Baum-Welch expectation-maximization**, and optionally incorporates **covariate-driven dynamic transitions** to model bidirectional tree cover change including encroachment, dieback, and fire-driven contraction.

**Key advances over traditional approaches**:
- Six ecological states capturing a gradient from open rangeland to closed-canopy forest
- Bidirectional dynamics: encroachment, drought/insect dieback, and fire-driven contraction
- Probabilistic emissions appropriate for proportion data (0-100% cover)
- Data-driven parameter learning rather than fixed assumptions
- Integration of fire, drought, terrain, and dispersal covariates
- Rigorous uncertainty quantification via posterior probabilities

---

## 1. Ecological Context and Objectives

### 1.1 Problem Statement

Tree cover dynamics in CONUS drylands involve three interacting bidirectional processes that reshape wildland vegetation structure:

1. **Woody encroachment into rangelands**: Trees and shrubs expand into grasslands and sagebrush steppe, reducing habitat quality for sagebrush-obligate species and altering ecosystem function.

2. **Woodland and forest dieback**: Drought, heat stress, and insect outbreaks drive canopy loss in established woodlands and forests, particularly during prolonged dry periods.

3. **Fire-driven contraction and type conversion**: Wildfire removes tree cover and, in some cases, triggers permanent type conversion where burned forests fail to regenerate and transition to shrubland or grassland.

These three processes are not independent. Encroachment is reshaping the lower end of the structural gradient — absolute tree cover in US rangelands has increased approximately 50% over the past three decades, with more than 25% of all rangeland experiencing expansion. At the same time, drought-driven dieback in pinyon-juniper woodlands and fire-driven contraction in montane forests are reducing cover from above; annual burned area in the western US has roughly tripled since the mid-1980s. The interplay of these opposing forces is shifting the biome-scale tree cover distribution toward intermediate woodland structure. Capturing both directions of this shift — and distinguishing genuine ecological transitions from measurement noise — is the central motivation for this framework.

The analysis mask covers all wildlands (excluding urban, water, and cropland), encompassing the full range of arid and semi-arid ecosystems where these processes operate. Quantifying the spatial patterns and temporal dynamics of these bidirectional changes requires:

1. **Separating signal from noise**: Remote sensing products have measurement error (~6-7% RMSE for tree cover). Year-to-year fluctuations may reflect sensor artifacts rather than real ecological change.

2. **Characterizing gradual transitions**: Tree cover change is not a binary event but a continuous process involving establishment, growth, dieback, and potential recovery or type conversion.

3. **Identifying ecological drivers**: Fire disturbance, drought stress, topography, and seed availability all influence both encroachment and contraction patterns.

### 1.2 Objectives

This modeling framework aims to:
- Classify each pixel's tree cover state across the 39-year time series
- Estimate the posterior probability of each state given the observations
- Identify transition years when pixels cross ecologically meaningful thresholds
- Quantify the influence of environmental covariates on state transitions
- Provide scientifically defensible estimates with associated uncertainty

---

## 2. Data Sources

### 2.1 Primary Observation: Tree Cover Time Series

**Source**: Rangeland Analysis Platform (RAP), Tree Cover Product
**Temporal extent**: 1986-2024 (39 annual observations)
**Spatial resolution**: 30m
**Units**: Percent tree cover (0-100%), stored as uint8
**NoData**: 255
**Measurement uncertainty**: ~6.7% RMSE (published product specification)

The RAP tree cover product provides the observed fraction of each pixel covered by tree canopy, derived from Landsat imagery and field-calibrated rangeland monitoring plots. Observations are continuous proportions subject to:
- Random measurement error from sensor noise
- Systematic biases in optical phenology (e.g., deciduous vs. coniferous)
- Missing data due to cloud contamination or processing failures

### 2.2 Environmental Covariates

Seven environmental covariates are optionally integrated to model transition dynamics:

| Covariate | Source | Type | Ecological Interpretation |
|-----------|--------|------|---------------------------|
| **Fire severity** | MTBS | Ordinal (0-4) | Disturbance removes existing trees, creates establishment opportunities |
| **Drought index** | SPEI-12 (gridMET) | Continuous | Cumulative water stress; captures multi-year drought impacts on deep-rooted trees |
| **Elevation** | NED DEM | Continuous | Proxy for temperature, moisture availability |
| **Solar insolation** | CSP-ERGo CHILI | Continuous | Topographic water balance, establishment microsite |
| **Aridity** | UNEP Aridity Index (P/PET) | Continuous | Longitudinal precipitation gradient; lower = more arid |
| **Seed distance** | Derived from RAP tree cover | Continuous | Distance to nearest seed source (10% threshold) |
| **Focal mean cover** | Derived from RAP tree cover | Continuous | Local density of potential seed sources (500m radius) |

**SPEI-12 rationale**: The 12-month standardized precipitation-evapotranspiration index integrates moisture anomalies over a full year, better capturing the cumulative drought stress that drives tree mortality and limits seedling establishment compared to shorter timescales (e.g., SPEI-6). Trees with deep root systems respond to multi-season moisture deficits rather than single-season dry spells.

**Aridity Index rationale**: The UNEP Aridity Index (P/PET from TerraClimate/gridMET long-term mean) captures the longitudinal precipitation gradient across CONUS drylands that is orthogonal to the elevation/CHILI topographic effects. This static covariate encodes baseline water availability, distinguishing semi-arid rangelands (~0.2-0.5 P/PET) where encroachment is favored from hyper-arid deserts where tree establishment is precluded.

**Temporal alignment**: Covariates measured in year *t* are used to model the transition from *t* to *t+1*. This reflects the ecological assumption that conditions during year *t* determine change entering year *t+1*.

### 2.3 Analysis Extent

**Mask**: US CONUS drylands wildlands (NLCD-based exclusion of urban, water, and cropland)
**Grid**: 118,742 × 96,888 pixels (EPSG:4326, ~30m native resolution)
**Valid pixels**: ~4.9 billion (42.6% of bounding box); ~191 billion pixel-years

The mask restricts analysis to arid and semi-arid wildlands (excluding urban, water, and cropland) where bidirectional tree cover dynamics are ecologically relevant.

---

## 3. Model Framework: Zero-Inflated Beta Hidden Markov Model

### 3.1 Hidden Markov Model Structure

An HMM assumes that each pixel belongs to one of *K* hidden states at each time step *t*. The state is not directly observed; instead, we observe tree cover values that are probabilistically emitted by the current state. The model consists of:

1. **Initial state distribution** π: Probability of starting in each state at *t* = 1986
2. **Transition matrix** A: Probability of transitioning between states year-to-year
3. **Emission distributions** B: Probability of observing a given tree cover value from each state

**Markov assumption**: The state at time *t* depends only on the state at *t-1*, not on the full history. This simplifies computation while capturing temporal autocorrelation.

**Why HMMs for tree cover?**
- **Temporal smoothing**: Measurement noise is averaged over the time series rather than affecting individual years
- **State persistence**: Transitions are penalized, preventing implausible year-to-year fluctuations
- **Principled inference**: Viterbi decoding finds the globally optimal state sequence given all observations

### 3.2 Six-State Bidirectional Ecological Framework

We model tree cover as a six-state system representing a gradient from bare ground to closed-canopy forest, designed to capture both encroachment and contraction dynamics:

| State | Ecological Label | Approx Range | Ecological Role |
|-------|-----------------|-------------|-----------------|
| 0 | **Bare** | ~0% | Rangeland baseline -- grassland/sagebrush, no trees |
| 1 | **Trace** | 0-4% | First establishment, isolated individuals |
| 2 | **Sparse** | 4-10% | Active early encroachment |
| 3 | **Open** | 10-25% | Advancing encroachment / post-fire partial recovery |
| 4 | **Woodland** | 25-50% | Open woodland (P-J, oak). Type-conversion endpoint? |
| 5 | **Forest** | 50%+ | Closed canopy. Dieback/fire source |

**Rationale for six states**:
- **Fine-grained low end**: Three states (Bare, Trace, Sparse) resolve the critical early stages of woody encroachment where management intervention is most cost-effective.
- **Woodland/Forest split**: Separating States 4 and 5 enables tracking of dieback (Forest to Woodland) and fire-driven type conversion (Forest to lower states that never recover).
- **Bidirectional dynamics**: The framework captures both upward transitions (encroachment) and downward transitions (dieback, fire-driven contraction) without imposing a directional bias.
- **Statistical discriminability**: With 6.7% RMSE, six states provide sufficient separation while avoiding over-fitting.

**Approximate ranges are not hard boundaries**: The emission distributions are probabilistic and overlapping. A pixel in State 2 can emit 8% cover or 2% cover, just with different probabilities.

### 3.3 Zero-Inflated Beta Emission Distribution

Traditional HMMs use Gaussian emissions, but tree cover is a **proportion** (bounded 0-1) with two distinctive properties:

1. **Zero-inflation**: True bare ground (0% cover) is qualitatively different from very low cover (1-2%). Many pixels have exact zero cover, creating a point mass at 0.

2. **Beta-distributed non-zeros**: For cover > 0, the distribution is skewed and bounded, which the Beta distribution naturally captures.

The **Zero-Inflated Beta (ZIB)** distribution models tree cover *y* in state *k* as:

```
y | k ~ ZIB(π_k, μ_k, φ_k)

where:
  - With probability π_k: y = 0 (point mass at zero)
  - With probability (1 - π_k): y ~ Beta(α_k, β_k)
    where α_k = μ_k × φ_k
          β_k = (1 - μ_k) × φ_k
```

**Parameters per state**:
- **π_k** ∈ [0, 1]: Zero-inflation probability (probability of exact zero)
- **μ_k** ∈ (0, 1): Mean of the Beta component (expected cover when not zero)
- **φ_k** > 0: Precision of the Beta component (inverse variance)

**Ecological interpretation**:
- **State 0 (Bare)**: High π_0 (~80%), low μ_0 (~1%), high φ_0 → mostly zeros with rare low-cover pixels
- **State 5 (Forest)**: Low π_5 (~0%), high μ_5 (~65%), low φ_5 → continuous cover centered on closed-canopy values

**Why ZIB is superior to Gaussian**:
- Respects the [0, 1] bounds (Gaussian can produce negative or >100% predictions)
- Models the zero-inflation explicitly (Gaussian treats zero as just another value)
- Captures the skewness of proportion data (Gaussian is symmetric)
- Better statistical efficiency (tighter posteriors, more confident inferences)

**Implementation**: Tree cover is converted from uint8 (0-100) to float32 (0.0-1.0) before computing log-probabilities. The ZIB log-probability is:

```python
log P(y | k) = log(π_k)                                    if y == 0
             = log(1 - π_k) + Beta_logpdf(y; α_k, β_k)    if y > 0
```

### 3.4 Monotonicity Constraints

To ensure states are ordered along the tree cover gradient, we enforce:

**μ_0 < μ_1 < μ_2 < μ_3 < μ_4 < μ_5**

Monotonicity is enforced **post-hoc at each M-step**: after L-BFGS optimization of μ and φ independently per state, states are re-ordered by ascending μ and all emission parameters (μ, φ, π) are permuted accordingly. The same permutation is applied to the transition matrix rows/columns and initial state distribution to maintain consistency. This:
- Aligns with ecological interpretation (State 5 must have higher cover than State 0)
- Prevents label-switching during parameter learning
- Allows each state's Beta parameters to be optimized independently without monotonicity constraints in the inner L-BFGS loop

---

## 4. Parameter Learning: Baum-Welch Expectation-Maximization

### 4.1 Learning Strategy

This framework **learns parameters from the data** using the Baum-Welch EM algorithm. This provides:

1. **Data-driven parameterization**: Emission distributions reflect the actual tree cover distribution in the drylands
2. **Adaptation to measurement properties**: Model automatically accounts for the specific noise characteristics of the Rangeland Analysis Platform (RAP) product
3. **Reproducibility**: Parameters are algorithmically derived rather than hand-tuned

### 4.2 Training Sample Selection

**Full-dataset EM is computationally prohibitive** (~4.9 billion pixels × 39 years). Instead:

1. **Stratified random sampling**: Sample up to 2 million pixels uniformly from the valid mask (configurable via `--em-sample`; actual counts depend on valid pixel availability per chunk — see Section 4.4)
2. **Temporal completeness**: Exclude pixels with >5 missing observations
3. **Spatial coverage**: Ensure samples span the full study area

**Assumptions**:
- Sampled pixels are representative of the full population
- Spatial dependence is weak enough that random sampling is valid
- ~500K–2M pixels × 39 years provides sufficient statistical power

### 4.3 EM Algorithm

**Initialization**:
- **π_k**: Linearly spaced from 0.9 (State 0) to 0.01 (State 5)
- **μ_k**: Linearly spaced from 0.01 to 0.70
- **φ_k**: Set to 10.0 for all states
- **log_init**: Uniform over states: log(1/K)
- **log_trans**: Diagonal-dominant, 90% self-persistence

**E-step**: Compute expected state occupancies (γ) and state-pair transitions (ξ) using forward-backward algorithm

**M-step**: Update parameters to maximize expected log-likelihood:
- **Initial probabilities**: π = γ_{t=0}
- **Transitions**: A_{ij} = Σ_t ξ_{t,ij} / Σ_t γ_{t,i}
- **Emissions**:
  - π_k: Closed-form (fraction of observations that are zero)
  - (μ_k, φ_k): L-BFGS optimization of weighted Beta log-likelihood per state independently; monotonicity enforced by post-hoc sort (see Section 3.4)

**Convergence**: Stop when relative log-likelihood improvement < 1×10⁻⁴ or 100 iterations reached

**Typical convergence**: 15-20 iterations, ~15 minutes on GPU (500K sample pixels, H100 NVL)

### 4.4 Learned Parameters

K=6 parameters learned via Baum-Welch EM on a stratified random sample of 495,914 pixels (2026-02-16). Convergence at iteration 15 (rel_change = 8.56×10⁻⁵ < tol = 1×10⁻⁴); 902s on NVIDIA H100 NVL.

| State | Label | μ (Beta mean) | φ (precision) | π (zero-inflation) |
|-------|-------|---------------|---------------|-------------------|
| 0 | Bare | 0.010 | 10,104 | 0.768 |
| 1 | Trace | 0.025 | 232 | 0.022 |
| 2 | Sparse | 0.068 | 58 | 0.019 |
| 3 | Open | 0.189 | 35 | 0.001 |
| 4 | Woodland | 0.385 | 24 | ~0 |
| 5 | Forest | 0.724 | 4.6 | ~0 |

**Interpretation**:
- **Monotonic means**: 1.0% → 2.5% → 6.8% → 18.9% → 38.5% → 72.4% (strictly increasing ✓)
- **Decreasing zero-inflation**: 76.8% → 2.2% → 1.9% → 0.1% → ~0 → ~0. Zero-inflation is concentrated in the lowest states, capturing the large fraction of dryland pixels with no detectable tree cover.
- **Decreasing precision**: φ falls from 10,104 (Bare — tight near-zero distribution) to 4.6 (Forest — broad, variable canopy). High φ reflects certainty about near-zero values; low φ reflects real variability in closed-canopy cover.
- **State 0 (Bare)**: μ = 1.0% with very high precision and 77% zero-mass — most dryland pixels have truly zero tree cover in most years.
- **State 5 (Forest)**: μ = 72%, confirming clean separation from Woodland (39%) as anticipated.

Parameters saved to `output_v2_k6_smoke2/params.npz`.

---

## 5. Covariate-Driven Dynamic Transitions (Optional)

### 5.1 Rationale

The default model assumes **stationary transitions**: the probability of transitioning from State *i* to State *j* is constant across time and space. This is ecologically unrealistic—fire in year *t* should increase the probability of transitioning back to bare ground.

**Dynamic transitions** allow transition probabilities to vary based on environmental covariates:

```
A_t(x, y) = function(fire_t, drought_t, elevation, ...)
```

where *A_t(x, y)* is the transition matrix for pixel *(x, y)* at time *t*.

### 5.2 Multinomial Logistic Regression

For each source state *i*, we model the log-odds of transitioning to state *j* as a linear function of covariates:

```
logit(A_ij) = β_i0 + β_i1 × fire + β_i2 × drought + ... + β_i6 × focal_mean
```

**Softmax normalization** ensures rows sum to 1:
```
A_ij = exp(logit_ij) / Σ_k exp(logit_ik)
```

**Feature vector** (D = 8):
1. Intercept (always 1)
2. Fire severity (ordinal 0-4)
3. SPEI-12 drought index (continuous)
4. Elevation (continuous, standardized)
5. CHILI insolation (continuous, standardized)
6. Aridity Index P/PET (continuous, standardized)
7. Seed distance (continuous, log-transformed)
8. Focal mean cover (continuous)

**Result**: Each pixel has a unique *(T-1) × K × K* transition tensor reflecting local environmental conditions each year.

### 5.3 Weight Estimation (Phase A½)

Dynamic weights are estimated **after EM convergence** using a separate iterative procedure:

1. **Sample covariates**: Re-sample the 2M pixels used for EM, but now also read covariate data
2. **Freeze emissions**: Keep ZIB parameters fixed
3. **Iterate**:
   - **E-step**: Run forward-backward with current dynamic transitions → compute ξ (expected transitions)
   - **M-step**: For each source state *i*, run L-BFGS to maximize:
     ```
     Σ_t,j ξ_{t,ij} × log(softmax(β_i × cov_t)_j)
     ```
4. **Converge**: Stop when log-likelihood improvement < 0.001 or 20 iterations

**Regularization**: L2 penalty on weights (λ = 0.01) to prevent overfitting

**Typical convergence**: 10-15 iterations, ~3-5 minutes on GPU

### 5.4 Ecological Hypotheses

Expected sign of coefficients for bidirectional dynamics:

| Covariate | Transition Direction | Expected Effect | Ecological Mechanism |
|-----------|---------------------|-----------------|----------------------|
| Fire severity | High states downward (e.g., Forest/Woodland -> Bare/Trace) | Positive | Fire removes canopy, can trigger type conversion if regeneration fails |
| Fire severity | Low states upward | Weakly negative | Post-fire conditions may temporarily inhibit establishment |
| Drought | Upward transitions (encroachment) | Negative | Water stress limits seedling survival and growth |
| Drought | Downward transitions from Forest/Woodland | Positive | Prolonged drought drives canopy dieback and mortality |
| Elevation | Low -> High states | Depends on region | Complex moisture/temperature gradient |
| Aridity (P/PET) | Low -> High states | Positive | Higher moisture availability favors establishment and persistence |
| Aridity (P/PET) | High states downward | Negative | Less arid sites buffer against drought-driven dieback |
| Seed distance | Low -> High states | Negative | Farther from seed source = less likely to establish |
| Focal mean | Low -> High states | Positive | High local density = more seed rain |

**Key bidirectional hypotheses**:
- **Fire-driven contraction**: High-severity fire in States 4-5 should strongly increase probability of transitioning to States 0-2. If post-fire recovery is absent over multiple years, this represents type conversion.
- **Drought-driven dieback**: Sustained negative SPEI should increase downward transitions from Forest (State 5) and Woodland (State 4), reflecting drought mortality.
- **Encroachment continues in absence of disturbance**: Without fire or drought, upward transitions should dominate, reflecting the baseline encroachment trend.

**Note**: Actual coefficients are data-driven and may reveal unexpected patterns.

---

## 6. Inference: Viterbi Decoding and Posterior Probabilities

### 6.1 Viterbi Algorithm

The **Viterbi algorithm** finds the most likely state sequence given the observations:

```
s* = argmax_s P(s_1, ..., s_T | y_1, ..., y_T)
```

**Forward pass** (dynamic programming):
```python
δ_t(k) = max_{s_{t-1}} [ δ_{t-1}(s_{t-1}) × A_{s_{t-1},k} × B_k(y_t) ]
```

**Backward pass** (backtracking):
```python
s*_t = argmax_k δ_t(k)  # at t=T
s*_t = pointer[t+1, s*_{t+1}]  # for t < T
```

**Output**: A single state sequence *(s_1, ..., s_T)* for each pixel, representing the globally optimal classification.

### 6.2 Forward-Backward Algorithm

The **forward-backward algorithm** computes posterior probabilities:

```
γ_t(k) = P(s_t = k | y_1, ..., y_T)
```

This quantifies uncertainty: γ_t(k) = 0.95 means 95% confident the pixel was in state *k* at time *t*.

**Forward recursion**:
```python
α_t(k) = [ Σ_{s_{t-1}} α_{t-1}(s_{t-1}) × A_{s_{t-1},k} ] × B_k(y_t)
```

**Backward recursion**:
```python
β_t(k) = Σ_{s_{t+1}} A_{k,s_{t+1}} × B_{s_{t+1}}(y_{t+1}) × β_{t+1}(s_{t+1})
```

**Posterior**:
```python
γ_t(k) = (α_t(k) × β_t(k)) / Σ_j (α_t(j) × β_t(j))
```

**Output**: A *T × K* posterior matrix for each pixel, providing full probabilistic classification.

### 6.3 Implementation: JAX Vectorization

Both algorithms are implemented in log-space using **JAX** to:
- Prevent numerical underflow (probabilities become tiny after 39 multiplications)
- Enable GPU acceleration (process 500,000 pixels in a single batch)
- Automatic differentiation (for EM gradient computation)

**Key optimization**: Use `jax.lax.scan` to loop over time steps while `jax.vmap` vectorizes over pixels:
```python
viterbi_batch = jax.vmap(viterbi_single)  # vectorize over N pixels
states = viterbi_batch(log_obs, log_init, log_trans)  # shape: (N, T)
```

This achieves ~100,000 pixels/second throughput on modern GPUs.

---

## 7. Output Products and Ecological Interpretation

### 7.1 Primary Outputs

All outputs are GeoTIFF rasters matching the input grid (118,742 × 96,888, EPSG:4326):

| Product | Dimensions | dtype | NoData | Description |
|---------|-----------|-------|--------|-------------|
| `states.tif` | (39 bands) | uint8 | 255 | Viterbi-decoded states (0-5) for each year |
| `posterior_k{0-5}.tif` | (39 bands each) | uint8 | 255 | P(state=k) x 200, scaled to [0, 200] |
| `max_state.tif` | (1 band) | uint8 | 255 | Mode state across all years |
| `trace_onset_year.tif` | (1 band) | int16 | 0 | Year first classified as State >= 1 (first tree establishment) |
| `sparse_year.tif` | (1 band) | int16 | 0 | Year first classified as State >= 2 (encroachment underway) |
| `open_year.tif` | (1 band) | int16 | 0 | Year first classified as State >= 3 (open canopy established) |
| `woodland_year.tif` | (1 band) | int16 | 0 | Year first classified as State >= 4 (woodland density) |
| `forest_year.tif` | (1 band) | int16 | 0 | Year first classified as State >= 5 (forest canopy closure) |

**Posterior encoding**: Posteriors are stored as uint8 (0-200) to save space. Divide by 200 to recover probabilities. Value of 100 = 50% probability.

### 7.2 Ecological Interpretation of States

**State 0 (Bare)**: Rangeland baseline -- grassland or sagebrush steppe with no established trees. Background condition for intact rangeland ecosystems.

**State 1 (Trace)**: First establishment. Isolated individual trees or small clusters present but not altering ecosystem function. Window of opportunity for low-cost intervention.

**State 2 (Sparse)**: Active early encroachment. Tree density increasing, scattered individuals coalescing. Intervention still feasible but requires more effort.

**State 3 (Open)**: Open canopy established. Advancing encroachment front, or partial post-fire recovery. Tree density beginning to alter understory composition and ecosystem function.

**State 4 (Woodland)**: Open woodland (e.g., pinyon-juniper, oak). Canopy 25-50%. Potential endpoint of type conversion where burned forests regenerate only to woodland density.

**State 5 (Forest)**: Closed canopy (50%+). Tree-dominated, high biomass. Source state for dieback (drought/insects driving transitions to State 4) and fire-driven contraction (high-severity fire driving transitions to States 0-2).

### 7.3 Transition Year Products

**Transition years** mark when pixels first cross ecologically meaningful thresholds:

- **Trace onset** (State >= 1): Earliest evidence of tree establishment
- **Sparse** (State >= 2): Encroachment underway, scattered trees coalescing
- **Open** (State >= 3): Open canopy established, ecosystem function altering
- **Woodland** (State >= 4): Open woodland density achieved
- **Forest** (State >= 5): Closed-canopy forest established

**Use cases**:
- Identify areas with recent rapid encroachment (short interval between trace onset and woodland)
- Map spatial patterns of encroachment timing (e.g., post-fire pulses, elevation gradients)
- Target management intervention to pixels approaching but not yet crossed critical thresholds
- Track where forest canopy closure has occurred and assess vulnerability to dieback or fire

**Pixels that never transition** are assigned year = 0 (NoData).

### 7.4 Uncertainty Quantification

**Posterior probabilities** provide pixel-level uncertainty:

- **High confidence**: max(γ_t) > 0.90 → clear state assignment
- **Moderate uncertainty**: max(γ_t) = 0.60-0.90 → state assignment likely but not certain
- **High uncertainty**: max(γ_t) < 0.60 → ambiguous, possibly transitional

**Spatiotemporal patterns in uncertainty**:
- Expect higher uncertainty during transition periods (e.g., year before/after crossing threshold)
- Expect lower uncertainty in stable states (State 0 for many years, or State 5 for many years)
- Pixels with frequent missing observations will have higher uncertainty

**Management implications**: High-uncertainty pixels may require field validation before intervention planning.

---

## 8. Key Assumptions and Limitations

### 8.1 Model Assumptions

1. **Markov property**: State at time *t* depends only on *t-1*, not on full history
   - **Justification**: Tree cover in year *t* is primarily determined by cover in *t-1* plus annual growth/mortality. Long-term memory (e.g., fire 10 years ago) is encoded in the current state.
   - **Limitation**: Ignores legacy effects beyond 1 year (e.g., multi-year drought recovery)

2. **State persistence**: Trees do not disappear without disturbance (high diagonal in transition matrix)
   - **Justification**: Established trees persist unless killed by fire, drought, or management
   - **Limitation**: Model may under-represent rapid state reversals (e.g., post-fire recovery)

3. **Spatial independence**: Each pixel is modeled independently
   - **Justification**: Simplifies computation, allows embarrassingly parallel processing
   - **Limitation**: Ignores spatial autocorrelation (neighbors tend to have similar states), which could be leveraged for additional smoothing

4. **Stationary emissions**: ZIB parameters (π, μ, φ) are constant over time
   - **Justification**: RAP product methodology is consistent across years
   - **Limitation**: Sensor changes, algorithm updates may introduce temporal non-stationarity

5. **Sample representativeness**: 2M sampled pixels for EM are representative of the full 4.9B
   - **Justification**: Random stratified sampling across the full extent
   - **Limitation**: Rare ecotypes or edge-of-range dynamics may be under-represented

### 8.2 Data Limitations

1. **Tree cover measurement error**: ~6.7% RMSE, with potential bias in low-cover pixels
   - **Impact**: ZIB emissions explicitly model this uncertainty, but systematic biases (e.g., underestimation of sparse trees) could shift state boundaries

2. **Covariate alignment**: MTBS fire data and SPEI have different native resolutions and projections
   - **Impact**: Reprojection to common grid introduces interpolation error, particularly for categorical fire severity

3. **Temporal lag structure**: We assume covariates in year *t* affect transition *t→t+1*, but effects may have multi-year lags (e.g., post-fire recovery takes 5-10 years)
   - **Impact**: Dynamic transitions capture immediate effects but may miss lagged responses

### 8.3 Computational Constraints

1. **Batch processing**: Full 4.9B-pixel inference is split into spatial chunks
   - **Impact**: Pixels at chunk boundaries are processed independently (no spatial context sharing)

2. **GPU memory**: Batch size limited to 500,000 pixels x 39 years x 6 states x float32 = ~456 MB per batch
   - **Impact**: Larger batches would improve GPU utilization but risk out-of-memory errors

### 8.4 Ecological Generalizability

1. **Drylands wildlands scope**: Model is parameterized for all CONUS drylands wildlands (grasslands, sagebrush steppe, pinyon-juniper woodland, dry forest)
   - **Limitation**: State boundaries (e.g., 50% = forest) are calibrated for dryland systems and may not apply to mesic forests where 50% cover is relatively open

2. **Tree-centric**: Model focuses on tree cover, ignoring sagebrush cover or other vegetation components
   - **Limitation**: Cannot directly assess sagebrush condition or co-dominant shrubs

3. **Bidirectional but symmetric structure**: The transition matrix allows both upward (encroachment) and downward (dieback/fire) transitions, but the model does not impose explicit constraints on transition direction
   - **Justification**: Allows the data to determine the balance between encroachment and contraction in each region
   - **Limitation**: In regions where fire is rare, the model may still learn a forward bias; in fire-prone regions, downward transitions may dominate. Interpretation requires regional context

---

## 9. Validation and Quality Control

### 9.1 Internal Validation

1. **EM convergence diagnostics**:
   - Log-likelihood must increase monotonically
   - Learned parameters must satisfy monotonicity constraints
   - Transition matrix rows must sum to 1.0

2. **State discriminability**:
   - Emission distributions should have limited overlap
   - Posteriors should show clear maxima (not flat across states)

3. **Temporal consistency**:
   - State sequences should exhibit persistence (few rapid oscillations)
   - Transition years should align with known disturbances (e.g., MTBS fire polygons)

### 9.2 External Validation (Proposed)

1. **Field plot comparison**: Where available, compare HMM states to field-measured tree density/cover
2. **Fire history alignment**: Pixels with MTBS high-severity fire should show state reversals
3. **Expert review**: Ecologists familiar with dryland systems assess whether spatial patterns and temporal dynamics are plausible

### 9.3 Sensitivity Analysis (Future Work)

1. **Number of states**: Re-run with K = 3, 4, 6 to assess sensitivity to state granularity
2. **Prior strength**: Test impact of stronger vs. weaker prior on initial state distribution
3. **Emission distribution**: Compare ZIB to Gaussian, Beta (no zero-inflation), or other distributions
4. **Sample size**: Test whether 1M or 5M sampled pixels for EM produces different parameter estimates. Note: the validated K=6 params were fit on ~496K pixels (below the 2M default); confirm whether this was intentional and whether larger samples meaningfully shift mu estimates.
5. **Ecoregion-stratified sampling**: Current EM sampling is spatially random, which may under-represent rare ecotypes (e.g., riparian woodland, high-elevation forest). Proposed: stratify the 2M-pixel sample by EPA Level II ecoregion, drawing proportional or equal samples per region, and compare resulting mu/phi values to the global fit. If regional mu values differ substantially, consider fitting separate parameter sets per ecoregion rather than a single global model.


---

## 11. Future Directions

### 11.1 Methodological Extensions

1. **Spatially-explicit HMM**: Incorporate spatial autocorrelation (e.g., via Markov Random Field priors)
2. **Semi-Markov models**: Allow state durations to be explicitly modeled (e.g., minimum 5 years in State 4 before reversion)
3. **Changepoint detection**: Formally test for structural breaks in transition probabilities (e.g., fire regime shifts)
4. **Multi-sensor fusion**: Integrate Landsat, Sentinel, NAIP to improve temporal resolution and reduce missing data

### 11.2 Ecological Applications

1. **Management scenario modeling**: Simulate intervention impacts (e.g., "What if we treat all State 2 pixels?")
2. **Climate change projections**: Use dynamic transitions with future climate covariates to forecast encroachment under warming/drying
3. **Fire regime interactions**: Couple HMM with fire spread models to assess feedbacks between tree cover and fire risk
4. **Sagebrush recovery modeling**: Extend to multi-species model that jointly tracks tree and sagebrush cover

### 11.3 Computational Optimizations

1. **Multi-GPU parallelization**: Distribute chunks across multiple GPUs for faster inference
2. **JIT compilation warmup**: Pre-compile JAX functions to reduce startup latency
3. **Gradient checkpointing**: Trade compute for memory to enable larger batch sizes
4. **COG pyramids**: Add overview layers to output GeoTIFFs for faster visualization

---

## 12. Conclusion

This Zero-Inflated Beta HMM framework provides a statistically rigorous, ecologically interpretable, and computationally scalable approach to classifying bidirectional tree cover dynamics in CONUS drylands. By learning parameters from data, explicitly modeling measurement uncertainty, and optionally integrating environmental drivers, the model produces defensible classifications with quantified uncertainty -- critical for informing management decisions across dryland wildlands.

The six-state gradient captures the continuous nature of tree cover change better than binary classifications, resolving both the early stages of woody encroachment (Bare through Sparse) and the distinction between open woodland and closed-canopy forest that is essential for tracking dieback and fire-driven type conversion. The ZIB emission distribution respects the statistical properties of proportion data. Dynamic transitions extend the framework to attribution, enabling hypothesis testing about fire, drought, and dispersal effects on both encroachment and contraction patterns.

With 62 tests and a modular, well-documented codebase, the model is ready for production deployment and provides a foundation for future methodological and ecological extensions.

---

**Document version**: 2.1
**Last updated**: 2026-02-16
**Authors**: SLM
