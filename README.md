# Autonomous Algorithmic Trading Agent
### BlackRock Hackathon @ IIT Delhi 2026

**Team:** Ritwik Sehrawat · Sidharth Valsan · Ayaan Maan  
**Final Score:** 66.3 / 100 · **Special Mention — 5th Place overall**

---

## Overview

An autonomous trading agent that manages a $10M simulated portfolio across 50 tickers over a 390-tick market session. The agent ingests a live price/volume feed, generates expected return signals, handles 11 corporate action events, and allocates capital using a Mean-Variance optimizer — all while obeying hard regulatory constraints.

**Scoring weights:** 60% Sharpe Ratio + 20% PnL + 20% Constraint Compliance

---

## Results

| Metric | Value | Target for Full Marks |
|---|---|---|
| Final NAV | $10,305,164 | — |
| PnL | +$305,163 (+3.05%) | $500K for 100% |
| Sharpe Ratio | 1.82 | 3.0 for 100% |
| Turnover | 29.48% | ≤30% hard cap |
| Total Orders | 56 | — |
| LLM Calls | 0 / 60 | ≤60 |

**Score breakdown:** Sharpe 60.8 · PnL 61.0 · Constraints 88.2 → **Total: 66.3/100**

### Test Case Results

| TC | Description | Result |
|---|---|---|
| TC001 | Stock split — D002 3:1 at tick 0 | PASS |
| TC002 | Earnings surprise — A001 +18% at tick 90 | FAIL |
| TC003 | Regulatory fine — B008 $340M at tick 280 | PASS |
| TC004 | Cardinality ≤ 30 maintained throughout | PASS |
| TC005 | Turnover ≤ 30% (achieved 29.48%) | PASS |
| TC006 | M&A rumour — E007 capped 0–5% from tick 200 | PASS |
| TC007 | BONUS: Index rebalance pre-positioning A005/B001 | PASS |
| TC008 | LLM budget: 0/60 calls used | PASS |

---

## Architecture

```
market_feed_full.json
        │
        ▼
  MarketState (ingest_tick)
   ├─ Price/volume history (50-tick rolling window)
   ├─ Slow EWMA (λ=0.94) + Fast EWMA (λ=0.85)
   └─ Corporate action scheduler
        │
        ▼
compute_expected_returns()       ◄─── LLM overlay (Black-Litterman model, built but not activated)
   Layer 1: Quant signals
   Layer 2: LLM blend
   Layer 3: CA event rules
        │
        ▼
   Optimizer (cvxpy MVO)
   ├─ Maximize: wᵀμ − γ · wᵀΣw  (γ = 2.5)
   └─ Constraints: fully invested, ≤15% per stock, ≤30 holdings, ≤30% turnover
        │
        ▼
weights_to_orders() → orders_log.json / portfolio_snapshots.json
```

---

## Signal Generation (Alpha Engine)

Three-layer expected return model:

**1. Quant baseline (per-ticker)**
- Slow EWMA return (λ=0.94): stable long-run drift
- EWMA spread (fast − slow): MACD-like trend confirmation
- Multi-horizon momentum: 4-tick (fast) + 15-tick (slow)
- Mean-reversion damper: subtract 20% of last tick's return to avoid chasing spikes
- Volume spike confirmation: +0.5% boost when volume >2.5× average and price is rising

**2. Cross-sectional momentum**
Z-score of relative performance across all 50 tickers at 5-tick and 15-tick horizons. Positive z-score = outperformer vs. peers, preventing bias toward sectors that are all rising together.

**3. Corporate action overlays**
Event-specific additive adjustments with linear decay windows (e.g., an earnings surprise adds +1.5% expected return for 10 ticks post-announcement, decaying linearly to 0).

---

## Optimizer

Mean-Variance Optimization (MVO) via `cvxpy`:

```
Maximize:  wᵀμ − γ·wᵀΣw   (γ = 2.5)
Subject to:
  Σwᵢ = 1          (fully invested)
  wᵢ ≤ 15%         (single-stock concentration cap)
  count(wᵢ > 0) ≤ 30  (cardinality hard rule)
  turnover ≤ 30%    (regulatory cap)
```

- **Covariance:** log-return sample covariance + Ledoit-Wolf diagonal shrinkage (factor 0.15) to stabilize the matrix inversion on limited history
- **Solvers:** OSQP with SCS fallback; equal-weight fallback if both fail
- **Greedy fallback (no cvxpy):** inverse-volatility weighting with sector caps

---

## Corporate Action Handling

**TC001 — Stock Split 3:1 (D002, tick 0):**
Three-part fix: (1) rescale entire price history by dividing by ratio 3, keeping EWMA/momentum calculations valid; (2) triple quantity held and halve average cost basis in portfolio; (3) 5-tick protection window preventing accidental sells during adjustment.

**TC002 — Earnings Surprise +18% (A001, tick 90):**
Signal boost and budget-aware PEAD logic were implemented. Failed because ~27–28% of the 30% turnover budget had already been consumed by tick 90, leaving no room to execute a net buy. Fix: reserve a dedicated CA budget specifically for known earnings events.

**TC003 — Regulatory Fine $340M (B008, tick 280):**
Hard 0.5% weight ceiling for 10 ticks post-event, combined with a −0.020 expected return penalty in the signal layer. Forced a SELL: 1,502 → 1,443 shares.

**TC004 — Cardinality ≤ 30:**
Four independent enforcement layers: order execution gate, target weight trimming before order generation, core weight builder truncation, and optimizer post-processing.

**TC005 — Turnover ≤ 30%:**
Dollar-budget system: each order deducts from a running notional budget. As budget runs low, an inertia parameter automatically blends target weights toward current weights (up to 92% inertia), dramatically reducing churn. Soft cap at 29.5% with a 0.5% safety buffer.

**TC006 — M&A Rumour (E007, tick 200):**
Permanent hard ceiling of 5% from tick 200 onward (no expiry). Signal layer also applies −0.6% expected return bias. E007 held between 0–5% for the remainder of the session.

**TC007 — Index Rebalance Pre-positioning (A005/B001, tick 370) — BONUS:**
Forward-looking CA seed: each rebalance tick, the agent scans all future CA events and pre-seeds a minimum 2% weight for any ticker scheduled for INDEX_REBALANCE. A005 reached 1.03% weight well before tick 370, passing the bonus check.

**TC008 — LLM Budget:**
0 of 60 calls used. LLM integration was fully built (one-shot master call at tick 5, response cached and reused across all 390 ticks), but not activated. The dataset and event schedule was fixed through the simulation, making LLM views redundant for this run.

---

## Universe

50 stocks across 5 sectors, all USD-denominated equities:

| Sector | Tickers | Count | Price Range |
|---|---|---|---|
| Technology | A001–A010 | 10 | ~$115–$235 |
| Finance | B001–B010 | 10 | ~$88–$132 |
| Healthcare | C001–C010 | 10 | ~$112–$167 |
| Energy | D001–D010 | 10 | ~$83–$112 |
| Consumer | E001–E010 | 10 | ~$75–$96 |

---

## What Worked

- Clean constraint compliance — 29.48% turnover, never exceeded 30 holdings at any tick
- Stock split handling — correctly adjusted price history AND portfolio accounting
- Pre-positioning for index rebalance (TC007 bonus), via forward-looking CA scan
- Robust optimizer with OSQP → SCS → equal-weight fallback chain
- Multi-signal alpha engine covering EWMA, momentum, cross-sectional, and CA overlays
- Complete LLM Black-Litterman integration (built and validated, conservatively not activated)

## What Didn't Work

- **TC002 failed** — turnover budget was exhausted before the A001 earnings surprise at tick 90. Fix: reserve a dedicated CA budget that routine rebalancing cannot consume.
- **Sharpe of 1.82 vs. target 3.0** — γ=2.5 was too conservative, leaving capital in cash and spreading across too many names. A γ of 1.0–1.5 and a more concentrated 10–15 name portfolio would have improved risk-adjusted returns.
- `build_fundamental_signals()` was implemented but not wired into the final scoring pipeline.

---

## Setup

```bash
pip install -r requirements.txt
python agent_candidate.py
```

To validate results:

```bash
python validate_solution.py
```

### Dependencies

- `numpy`, `pandas` — data handling
- `cvxpy` — convex optimizer (OSQP/SCS solvers)
- `websockets`, `httpx` — market feed ingestion and LLM endpoint
- `python-dateutil` — CA event scheduling

---

## Files

| File | Description |
|---|---|
| `agent_candidate.py` | Main agent: market state, signal generation, optimizer integration, order execution |
| `optimizer.py` | MVO optimizer with cardinality enforcement, Ledoit-Wolf shrinkage, solver fallbacks |
| `market_feed_full.json` | Full 390-tick price/volume feed for all 50 tickers |
| `corporate_actions.json` | All 11 CA events with ticker, tick, and event parameters |
| `initial_portfolio.json` | Starting portfolio state ($10M cash) |
| `validate_solution.py` | Official grader, reproduces the 66.3/100 score |
| `orders_log.json` | All 56 orders executed during the run |
| `portfolio_snapshots.json` | Per-tick portfolio state across all 390 ticks |
| `result_output.json` | Final scored output with TC pass/fail breakdown |
| `results.json` | Raw performance metrics |
