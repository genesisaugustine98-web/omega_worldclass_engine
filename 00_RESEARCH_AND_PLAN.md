# OMEGA-WORLDCLASS-ENGINE — Phase 0 Research and Adversarial Plan

> **Purpose:** educational and academic research only. This project is not financial advice, a recommendation, or a promise of profitability.
>
> **Epistemic rule:** “zero chance of wrong data” is not attainable. The engineering target is instead **detectable, reproducible, provenance-preserving failure**: every observation should have a source, retrieval time, schema, quality report, point-in-time eligibility rule, and content hash. Ambiguous or failed data must be quarantined rather than silently repaired.

## Executive decision

OMEGA should not begin as a 500-feature, four-model trading machine. That would maximize the surface area for leakage, false discovery, licensing failures, and unverifiable stories. It should begin as a **market-state research system** whose first artifact is a trustworthy, point-in-time panel and whose first benchmark is deliberately simple.

The scientific unit is:

\[
P(\text{phenomenon}_{t:t+H} \mid \text{information observable by } t)
\]

not “the next candle will be green.” A phenomenon is an operationally defined future path property—such as trend ignition, failed breakout, volatility expansion, liquidity sweep/reversal, compression, or mean reversion. Predictions must be timestamped, uncertainty-aware, and evaluated out of sample by regime, instrument, source, cost assumption, and calendar period.

No model output establishes why traders acted. “Why” requires a hierarchy of evidence: institutional rules and contemporaneous records; identified causal designs or event studies; market microstructure evidence; and only then model attribution. SHAP describes model reliance, not economic causation.

---

## A. Deconstructing a 30-minute move

A 0.3% FX move in 30 minutes is large for major pairs and ordinary for some emerging-market pairs or crisis windows. Its meaning depends on volatility, pair, session, quote convention, and event context. The following 30 factors are ranked in rough causal tiers—not as a universal ordering. Causes interact, and many are latent.

### Tier 1 — Direct repricing impulses

1. **Central-bank policy surprise:** the difference between the announced decision/guidance and the market-implied path. The surprise, not the headline level, reprices discount-rate differentials.
2. **Macro release surprise:** point-in-time actual minus survey expectation, normalized by historical surprise dispersion. Payrolls, CPI, GDP, PMIs, and wage data matter differently by regime.
3. **Unscheduled official communication:** interventions, emergency facilities, resignations, capital controls, or geopolitical statements can abruptly alter expected policy and convertibility risk.
4. **Order-flow imbalance:** aggressive buy versus sell flow consumes available liquidity and moves the marginal price. Evans and Lyons link order flow to exchange-rate determination.
5. **Dealer inventory constraints:** dealers shade quotes to manage unwanted inventory, especially when balance-sheet capacity is scarce.
6. **Cross-market rate shock:** a move in sovereign yields, OIS, or rate futures changes expected carry and relative valuation.
7. **Risk-premium shock:** abrupt changes in required compensation for funding, liquidity, volatility, or crash risk can overwhelm expected-rate effects.
8. **Actual or suspected FX intervention:** official purchases/sales and signaling can alter both flow and beliefs.

### Tier 2 — Amplifiers and nonlinear market mechanics

9. **Thin order-book depth:** the same flow produces a larger price response when displayed and latent liquidity withdraw.
10. **Stop-loss cascade:** crossing clustered technical or option-related levels converts conditional orders into marketable flow.
11. **Option dealer hedging:** changes in delta and gamma exposure can force procyclical or countercyclical spot hedging; sign depends on dealer positioning, which is usually only partially observed.
12. **Volatility-control deleveraging:** systematic strategies reduce exposure after volatility rises, potentially reinforcing cross-asset moves.
13. **CTA/trend threshold crossing:** breakout or moving-average rules can synchronize directional flows, but their exact positions are estimated, not known.
14. **Margin and collateral calls:** adverse moves force liquidation across assets and currencies.
15. **Funding-market stress:** scarcity of dollars or another funding currency can move spot, forwards, and cross-currency basis together.
16. **Prime-broker or internal risk-limit tightening:** reduced leverage and credit lines can trigger position reduction without a new public-information shock.
17. **Liquidity-provider withdrawal:** adverse-selection risk around news causes wider spreads and reduced depth.
18. **Execution-algorithm synchronization:** VWAP/TWAP schedules, benchmark fixes, and common signals can concentrate otherwise dispersed demand.

### Tier 3 — Structured demand and institutional timing

19. **Benchmark fixing flows:** hedgers and asset managers transact around WM/R and other fix windows, creating time-localized imbalances.
20. **Month/quarter/year-end rebalancing:** international portfolios hedge or rebalance after relative asset moves.
21. **Corporate hedging:** importers, exporters, and treasury desks transact around cash-flow, issuance, dividend, and acquisition needs.
22. **Reserve-manager activity:** official diversification can create persistent flow, though attribution is difficult in real time.
23. **Debt issuance and repatriation:** cross-border issuance, coupons, dividends, and M&A generate mechanical currency demand.
24. **Session transitions:** London open, New York overlap, fixings, and Asia handoffs alter participant composition and depth.

### Tier 4 — State variables that condition impact

25. **Pre-event positioning:** crowded carry, consensus shorts, or options skew determine how violently new information is absorbed.
26. **Recent realized volatility:** high-volatility states alter leverage, risk limits, spreads, and the significance of a 0.3% move.
27. **Technical reference levels:** prior highs/lows, round numbers, and volume concentrations matter insofar as participants condition orders on them; they are coordination devices, not physical laws.
28. **Cross-asset correlation state:** equity, commodity, credit, and rates shocks transmit differently during inflation, growth, and crisis regimes.
29. **Information uncertainty/disagreement:** greater disagreement can increase volume, adverse selection, and jump risk.
30. **Data/venue artifacts:** stale quotes, bad ticks, broker rollovers, duplicated bars, timestamp errors, and indicative-versus-tradable prices can imitate a move and must be ruled out before interpretation.

### What can be measured honestly

Retail OHLCV does not reveal “smart money.” Tick volume is often venue-specific; spot FX is decentralized; consolidated depth is unavailable. We can observe proxies: returns, ranges, spreads where available, quote frequency, futures volume/commitment data, rate changes, event surprises, session/fix flags, volatility, and cross-asset moves. Language in reports must say **“consistent with”** rather than “caused by” unless an identification design supports the latter.

Useful research foundations include Evans & Lyons on order flow; Andersen et al. on high-frequency FX responses to macro news; BIS Triennial Surveys and market-structure publications; Menkhoff et al. on currency momentum; Lustig, Roussanov & Verdelhan on common currency risk factors; and Moskowitz, Ooi & Pedersen on time-series momentum. Public AQR research can motivate trend and risk-premium hypotheses, but vendor papers must not be treated as peer-reviewed causal proof. JPMorgan and Two Sigma publications may inform hypotheses only when a stable, legally accessible primary source can be archived; we will not fabricate or paraphrase inaccessible proprietary research.

---

## B. Market historian: six operational phenomena

The historical examples below generate hypotheses. They do not prove that one candle pattern had the same cause across decades.

### 1. Trend ignition

**Definition:** volatility-adjusted directional displacement followed by continuation over horizon H, with limited adverse excursion and evidence of rising participation or cross-market confirmation.

- **1998:** the Asian/Russian crises and LTCM unwind produced funding stress, de-risking, and sharp yen dynamics. Leveraged positions were reduced because capital and liquidity constraints changed, not because banks “defended a 200 EMA.” Participants benefited by reducing insolvency and funding risk; fast trend followers could benefit from persistent forced flow.
- **2008:** deleveraging, dollar funding demand, policy surprises, and balance-sheet impairment created persistent directional moves. The economic motive was survival, collateral, and funding—not merely technical breakout trading.
- **2020:** the pandemic shock first produced a dollar-liquidity scramble, then extraordinary policy intervention. The Fed’s swap lines and facilities changed the state. Trends could reverse when the policy reaction function changed.

**Changed:** execution became more electronic, algorithmic, fragmented, and fast; policy communication expanded; yet leverage and funding constraints remained central.

### 2. Liquidity sweep and reversal

**Definition:** price crosses a prior extreme by a volatility-scaled threshold, exhibits transient spread/range expansion, then closes back inside and reverses over H.

- **1998:** sparse liquidity and forced unwinds could overshoot fundamental repricing. Traders providing liquidity after exhaustion benefited if funding survived.
- **2008:** apparent sweeps could be genuine information jumps or liquidation. Naive mean reversion was dangerous because balance-sheet constraints persisted.
- **2020:** discontinuous headlines and electronic stop execution generated fast overshoots, while policy announcements could abruptly reverse them.

**Changed:** stop discovery and execution are faster, but decentralized spot data makes “the” swept level venue-dependent.

### 3. Volatility expansion / jump state

**Definition:** future realized variance or range exceeds a rolling conditional quantile, optionally with jump-test confirmation.

- **1998:** contagion and opaque leverage increased uncertainty and reduced liquidity.
- **2008:** solvency and counterparty uncertainty created repeated jumps; protection buyers and option holders benefited, while short-volatility participants faced convex losses.
- **2020:** epidemiological and policy news created exogenous jumps, followed by policy suppression or relocation of volatility.

**Changed:** options markets and systematic volatility strategies are larger, but event risk and liquidity withdrawal remain mechanisms.

### 4. Compression / liquidity build-up

**Definition:** declining realized volatility and range, low directional efficiency, and stable spreads before a transition; compression alone does not determine breakout direction.

- **1998:** pre-crisis calm could reflect hidden leverage and policy credibility until a constraint broke.
- **2008:** calm windows often reflected temporary intervention or waiting for institutional decisions.
- **2020:** markets compressed between scheduled decisions and pandemic updates, then repriced on new information.

**Benefit:** market makers earn spread during stable periods; option sellers earn premium if realized volatility stays below implied. Both bear gap risk.

### 5. Failed breakout / trapped positioning

**Definition:** a breakout beyond a reference level fails to maintain acceptance and reverses sufficiently to impose losses on late entrants.

- **1998:** policy defense, intervention, or exhaustion could invalidate extrapolation.
- **2008:** rescue announcements repeatedly changed expectations; some breakouts failed, while others resumed when policy credibility proved insufficient.
- **2020:** headline-driven moves reversed as details, implementation, and cross-market reactions were digested.

**Changed:** social/news propagation is faster, but the mechanism—belief revision plus one-sided positioning—persists.

### 6. Mean-reversion / inventory normalization

**Definition:** a standardized deviation from a state-dependent anchor contracts over H without first entering a structural break state.

- **1998:** dealer inventory normalization could support short-horizon reversal, but crisis contagion invalidated stable anchors.
- **2008:** ordinary mean reversion often failed because the equilibrium itself moved and funding constraints forced persistent flow.
- **2020:** policy backstops sometimes restored anchors rapidly; pandemic uncertainty sometimes destroyed them.

**Benefit:** liquidity providers are compensated for absorbing temporary imbalance. The risk is mistaking permanent information for temporary pressure.

---

## C. Flip the problem: state first, response second

1. Construct features from information with `available_at <= prediction_time`.
2. Produce multi-label probabilities for the six future-path phenomena. Labels may overlap; forcing a single class can be false precision.
3. Calibrate probabilities by rolling, untouched validation periods. Report Brier score, log loss, calibration error, precision-recall, and regime-conditional coverage—not only F1.
4. Apply split-conformal or adaptive conformal methods only under stated exchangeability/drift caveats. Conformal methods target empirical coverage; they do not guarantee correct probabilities under arbitrary regime change.
5. Define a response policy separately. It can abstain. It maps probability, uncertainty, spread, regime, and risk budget to a hypothetical position.
6. Backtest the response with delayed execution, bid/ask costs, spread widening, slippage scenarios, financing, unavailable-bar handling, and source-specific prices.
7. Attribute three different objects separately: **prediction evidence**, **policy decision**, and **realized P&L**. Never convert a SHAP value into a historical causal statement.

The primary research question is whether state probabilities generalize. Trading performance is secondary and should not rescue an uncalibrated classifier through threshold mining.

---

## D. Twenty failure modes: detection and mitigation plan

| # | Failure mode | Detection | Mitigation/code plan |
|---|---|---|---|
| 1 | Future leakage | Feature-availability audit; deliberately shift labels; inspect suspiciously high scores | Point-in-time joins; `available_at`; embargo and purge; pipeline tests |
| 2 | Revised macro data | Compare latest values with vintage releases | Prefer ALFRED/vintage data; store release and revision timestamps |
| 3 | Timestamp/DST errors | UTC monotonicity, duplicate and DST transition tests | Store UTC internally; explicit source timezone and calendar version |
| 4 | Weekend/holiday confusion | Session-calendar expected-bar report | Do **not** demand bars while FX is closed; distinguish expected closures from missing data |
| 5 | Bad ticks/outliers | Cross-source disagreement, return/spread robust z-scores | Quarantine; never silently winsorize raw data; retain repair ledger |
| 6 | Duplicate/overlapping bars | Key uniqueness and OHLC reconciliation | Deterministic deduplication with provenance and hard conflict failure |
| 7 | Broker/venue differences | Pairwise source basis and event-window comparisons | Train/evaluate by source; avoid pretending indicative mid is executable |
| 8 | Symbol/quote inversion | Economic identity and sign tests | Canonical pair metadata; tested inversion function; no string guessing |
| 9 | Corporate/vendor backfill changes | Hash drift on identical retrieval partitions | Immutable raw snapshots, manifests, SHA-256, versioned transformations |
| 10 | Survivorship/selection bias | Audit universe construction by date | Predeclare pairs and availability; include delisted/pegged periods where relevant |
| 11 | Label leakage/overlap | Trace each label’s maximum timestamp; overlap diagnostics | Pure future windows; multi-label formulation; gap between feature and label windows |
| 12 | Data snooping | Count trials and researcher degrees of freedom | Hypothesis registry; nested walk-forward; untouched final test; deflated Sharpe/PBO |
| 13 | Overfitting | Train-validation gap, seed instability, permutation tests | Simple baselines; regularization; feature ablation; repeated temporal splits |
| 14 | Regime shift | Rolling calibration/PSI; change-point tests | Recency weighting, abstention, adaptive conformal, explicit retraining policy |
| 15 | Spread/slippage fantasy | Compare assumed costs with available quote data and stress grids | Bid/ask execution; latency; 1x/2x/3x cost scenarios; no same-close fills |
| 16 | News trading impossibility | Event timestamp precision and latency audit | Use embargo windows; model realistic publication-to-decision delay; abstain if uncertain |
| 17 | Missing-not-at-random data | Correlate missingness with volatility/session/vendor outages | Missingness features; quarantine severe outages; sensitivity analyses |
| 18 | Class imbalance | Per-label prevalence and PR curves by period | Weighted losses, thresholding on validation only, event-based metrics |
| 19 | Multiple testing in attribution | False-discovery diagnostics across slices | Predeclared slices; hierarchical summaries; confidence intervals; minimum sample sizes |
| 20 | Operational/reproducibility failure | Clean-runtime smoke test and manifest comparison | Seed registry, pinned environment, checkpoints, atomic writes, resumable stages |

Additional red-team tests: randomized labels must destroy performance; feature timestamps shifted forward must trigger the leakage guard; cost increases must degrade P&L monotonically in controlled fixtures; and deleting a cached partition must trigger a deterministic provenance failure rather than an automatic unlogged substitution.

---

## E. Seven optional SOTA modules

1. **[OPTIONAL] LLM news causality assistant.** Retrieval-augmented classification over legally obtained, timestamped documents. It proposes event mechanisms and evidence links; it never declares causality autonomously. FinGPT may provide design ideas, but data licensing and temporal contamination are the primary constraints.
2. **[OPTIONAL] Quantum-walk features.** Treat as an experimental representation benchmark, not a “quantum edge.” Require comparison against matched classical spectral/random-walk features and reject unless it improves untouched temporal tests after compute adjustment.
3. **[OPTIONAL] Adaptive conformal prediction.** Prediction sets or intervals with rolling coverage monitoring, regime slices, and abstention. Report when exchangeability assumptions are implausible.
4. **[OPTIONAL] Constrained RL for position sizing.** Only after a stable supervised state model and simulator exist. Compare against fixed-risk and convex optimization baselines; constrain turnover, drawdown, leverage, and action changes.
5. **[OPTIONAL] Causal graphs.** Encode hypotheses linking policy surprises, rates, funding stress, order-flow proxies, liquidity, and phenomena. Use DAGs to state assumptions and identify estimands; do-calculus is not credible without defensible interventions and measurements.
6. **[OPTIONAL] Anomaly narrative assistant.** An LLM summarizes unusual windows from structured evidence and retrieved contemporaneous sources. It must cite inputs and distinguish observations from hypotheses.
7. **[OPTIONAL] Cross-asset temporal graph network.** Nodes for currencies, sovereign curves, equity indices, commodities, volatility, and funding measures; edges learned with temporal restrictions or imposed from economics. Benchmark against simple lagged cross-asset features.

### Model realism notes

- **TFT** is useful when covariates, horizons, and interpretability justify its complexity, but it is not automatically superior to boosted trees on engineered 30-minute features.
- **TabPFN** is optional and constrained by dataset size, license/version, memory, and suitability for large temporal panels. It should be tested on sampled or meta-learning tasks, not enabled by marketing date.
- **XGBoost/LightGBM plus calibration** is the likely V1 champion because it is fast, inspectable, and strong on tabular data.
- “2025 libraries” should not be blindly pinned. We pin only versions verified in a clean Colab runtime and record Python/CUDA compatibility.

---

## F. Subtractions required to ship a scientifically useful V1

### Cut from one-hour V1

- Do not download 35 years from multiple vendors in the first run.
- Do not scrape Forex Factory; access, licensing, robots policy, timestamp quality, and historical completeness must be reviewed first.
- Do not claim Dukascopy starts in 1995 for every pair or HistData reliably fills 1990–1995. Coverage must be discovered per symbol and source.
- Do not build 500 features. Start with 25–50 predeclared, tested features.
- Do not train TFT, TabPFN, RL, GNN, quantum, or LLM modules.
- Do not produce a PDF before metrics and provenance are trustworthy.
- Do not use Kelly sizing; estimation error makes unconstrained Kelly hazardous. V1 uses no capital recommendation and a simple capped hypothetical risk rule.
- Do not send data or secrets to Telegram by default.

### V1 learning slice

1. One or two liquid pairs, a clearly documented source, and a manageable period.
2. Immutable raw cache plus normalized 30-minute Parquet partitions.
3. Seven validation families: schema, temporal integrity, market calendar/gaps, OHLC consistency/outliers, spread/source quality, point-in-time alignment/leakage, and hash/provenance.
4. Six config-driven phenomenon labels with unit tests on synthetic paths.
5. Naive prevalence baseline, logistic regression, and gradient-boosted trees.
6. Purged walk-forward evaluation, probability calibration, uncertainty/abstention, and regime slices.
7. A no-trade state report first; optional hypothetical backtest second.
8. Checkpoint after completed stages and at a configurable time interval; checkpointing every ten minutes is not guaranteed during a blocking library call.

### Roadmap to 35 years

“1990–2025” is not a homogeneous dataset. The euro did not trade as EUR/USD before 1999; venue structure, decimalization, liquidity, and data collection changed. A 35-year study should use a coverage matrix and potentially predecessor currencies or restricted universes, with explicit non-comparability flags. Daily macro history may span decades while reliable tick/bid-ask history may not. Storage estimates and Colab Drive quotas must be measured before promising free-tier feasibility.

---

## Proposed architecture and scientific contracts

The requested module tree is directionally sound, with these upgrades:

- Add `data/catalog.py`, `data/manifests.py`, and `data/point_in_time.py`.
- Add `evaluation/splits.py`, `evaluation/calibration.py`, and `evaluation/red_team.py`.
- Separate `labels/` from `phenomena/`: labels are measurable future-path definitions; phenomena modules are state estimators and research hypotheses.
- Add `tests/` with synthetic fixtures before download code.
- Add `docs/data_cards/`, `docs/model_cards/`, and `experiments/registry.parquet`.
- Use configuration validation (Pydantic or OmegaConf schema) rather than an ungoverned 100-flag YAML file.
- Use content-addressed immutable raw storage and atomic writes. “Re-download = fail” is too rigid: re-download is allowed only as a new immutable retrieval version, never as silent overwrite.

Every report must include dataset hash, transform version, split boundaries, availability policy, model version, seed, threshold source, cost model, and known limitations.

---

## Evidence and citation policy

Initial primary/public references to archive in a later bibliography lockfile:

1. Andersen, Bollerslev, Diebold & Vega (2003), *Micro Effects of Macro Announcements: Real-Time Price Discovery in Foreign Exchange*, American Economic Review.
2. Evans & Lyons (2002), *Order Flow and Exchange Rate Dynamics*, Journal of Political Economy.
3. Menkhoff, Sarno, Schmeling & Schrimpf (2012), *Currency Momentum Strategies*, Journal of Financial Economics.
4. Lustig, Roussanov & Verdelhan (2011), *Common Risk Factors in Currency Markets*, Review of Financial Studies.
5. Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, Journal of Financial Economics.
6. Bailey et al. (2017), *The Probability of Backtest Overfitting*, Journal of Computational Finance.
7. López de Prado (2018), *Advances in Financial Machine Learning*—useful engineering methods, not unquestioned authority.
8. BIS Triennial Central Bank Survey and BIS FX market-structure publications, using archived edition/date.
9. Federal Reserve ALFRED documentation for real-time macro vintages.
10. Lim et al. (2021), *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, International Journal of Forecasting.
11. Angelopoulos & Bates (2023), *Conformal Prediction: A Gentle Introduction*, Foundations and Trends in Machine Learning.
12. Public AQR papers on trend following and style premia, archived by exact title/version where used.

Before a claim enters code documentation, we will record the exact source, version/date, quotation or proposition supported, and limitations. No invented citations, proprietary-paper claims, or retrofitted historical narratives.

---

## Phase 0 acceptance gates

Phase 1 should start only after the user chooses the primary objective and accepts these constraints:

- We can minimize and expose data risk, not guarantee perfect data.
- We will not equate model explanation with trader intent or causality.
- We will verify coverage and legal access before hard-coding vendors.
- We will build a tested narrow V1 before scaling to 35 years, 500 features, and deep models.
- Optional LLM and quantum modules default to **OFF**.
- Academic rigor outranks a visually impressive but invalid backtest.

## QUESTION FOR USER

**Do we prioritize CAUSAL ACCURACY or PREDICTIVE POWER for V1?**

- **Causal accuracy:** slower, fewer claims, explicit DAGs/event studies/point-in-time macro surprises, stronger emphasis on mechanisms and identification.
- **Predictive power:** faster benchmark competition under strict temporal validation, with explanations explicitly labeled associational rather than causal.
- **Recommended:** a predictive V1 with causal hygiene—build a calibrated state benchmark first, while preserving point-in-time evidence and refusing causal language that the design cannot support.

---

**CONVERSATION_HOOK**

- TODO(user): Choose `V1_PRIORITY = causal_accuracy | predictive_power | predictive_with_causal_hygiene`.
- TODO(research): Confirm initial currency pairs, legally permitted data sources, and whether the user already has vendor credentials/data.
- TODO(optional): Evaluate a Causal Transformer only after a reproducible non-neural benchmark exists.
- TODO(optional): Quantum-walk experiments remain off until a falsifiable benchmark and compute budget are approved.
