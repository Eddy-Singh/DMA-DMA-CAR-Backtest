# DMA CAR Portfolio Strategy — Group Trading Project

> **Course:** Group Trading Strategy Project (25%)  
> **Period:** October 1, 2026 – October 30, 2026 (test window)  
> **Portfolio:** Nifty 250 (India) — 250 liquid equity tickers  
> **Language / Stack:** Python (Streamlit + Pandas / NumPy / Plotly / yfinance / Optuna)  
> **Submission:** Code, asset CSV (`portfolio_daily_closes.csv`), performance metrics CSV (`performance_metrics.csv`), and presentation (this page + slides)

---

## Table of Contents

1. [Project Objective & Requirements](#1-project-objective--requirements)
2. [Strategy Justification & Methodology](#2-strategy-justification--methodology)
3. [Portfolio Composition & Data](#3-portfolio-composition--data)
4. [Codebase Structure](#4-codebase-structure)
5. [Entry / Buy Conditions (Green Flag)](#5-entry--buy-conditions-green-flag)
6. [Exit / Sell Conditions](#6-exit--sell-conditions)
7. [Performance Metrics (Test Period)](#7-performance-metrics-test-period)
8. [How to Run](#8-how-to-run)
9. [Group Contributions](#9-group-contributions)
10. [Notes for Presentation & Grading](#10-notes-for-presentation--grading)
11. [References to Project Outlines](#11-references-to-project-outlines)

---

## 1. Project Objective & Requirements

Per the group project outline:

- **Objective:** Develop an automated trading strategy with forecasting and portfolio updating; submit asset list + daily closing prices in a CSV; evaluate by P&L and other metrics.
- **Portfolio:** At least the class-announced assets (here, the full Nifty 250 list, selectable via multiselect), publicly traded for ≥1 year by August 1, 2026.
- **Team:** 4–5 students; **each student must specify their contribution** (see Section 9).
- **Evaluation:** Profit / Loss and performance metrics from the **October 1, 2026 – October 30, 2026 test period**.
- **Deliverables:** Code + presentation + CSV files. Late submission not accepted.
- **Language:** Any allowed (we use Python / Streamlit).

We also must justify our choice of strategy, clearly describe the methodology, and evaluate using appropriate metrics.

---

## 2. Strategy Justification & Methodology

### Why DMA (DMA CAR Portfolio Backtester)?

- **Trend-following with stacked moving averages** is a well-known, explainable framework (SMA50 > SMA100 > SMA200 = upward momentum).
- **Anti-FOMO ceiling** (`Price < 110–125% × SMA200`) prevents buying after large runs, reducing risk of over-extended entries.
- **Compounding** reinvests full principal + profit after each exit — this produces geometric growth rather than linear, and matches the “portfolio growth” expectation in the rubric.
- **Dip averaging** (–5% / –10%) uses a reserve fund, adding a risk-management layer that demonstrates thoughtful capital management.
- **Dynamic allocation** splits all available cash evenly across all stocks signaling on a given day — this avoids over-concentration and shows portfolio-level thinking.

### What was removed / adjusted

- **CAR filter** (`Cumulative Average Reversal`) was fully removed from both `dma_car_backtester.py` and `optimizer.py` as requested.
- **Entry conditions relaxed** (default `SMA200 Buffer = 125`, range 100–140) so more names enter in a short window.
- **Profit target lowered** (default 3.5%, range 1–20%) so exits occur within a 30-day test window.
- **Period-end closeout** guarantees every open position closes on the last test-day (no stale “OPEN” trades), making Win Rate / Profit per Trade meaningful.

---

## 3. Portfolio Composition & Data

### Ticker pool
- **Full pool:** `NIFTY_250` (line 96 of backtester) — 250 `.NS` tickers: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK, LT, AXISBANK, ASIANPAINT, MARUTI, SUNPHARMA, TITAN, BAJFINANCE, WIPRO, ULTRACEMCO, NESTLEIND, POWERGRID, TECHM, NTPC, HCLTECH, ONGC, and 226 more.
- **Custom selection:** Sidebar multiselect lets the group restrict to any announced subset.
- **All tickers traded for ≥1 year:** Nifty 250 constituents are large-cap liquid names with multi-year history — satisfies the “traded at least one year by Aug 1, 2026” rule.

### CSV outputs
- `portfolio_daily_closes.csv` — selected tickers, daily close prices over the selected date range (download button on results page).
- `performance_metrics.csv` — all 13 requested metrics + formulas (download button).

---

## 4. Codebase Structure

| File | Purpose | Key Lines / Features |
|---|---|---|
| `dma_car_backtester.py` | Main Streamlit app | Entry signals (232), daily loop (256), exit/closeout (277 / 348), metrics table (407+) |
| `optimizer.py` | Optuna hyperparameter optimizer | `run_backtest()` (100), objective (302), validation (371) |
| `requirements.txt` | Dependencies | streamlit, yfinance, pandas, numpy, plotly |
| `(README.md)` | This file | Strategy, metrics, contributions, grading notes |

---

## 5. Entry / Buy Conditions (Green Flag)

From the “Perfect Entry Checklist” header banner (`line 209`) and signal logic (`line 232`):

1. **The Trend** — SMA50 > SMA100 > SMA200 (clear stacked momentum).
2. **The Value** — Price > SMA50 (currently “hot” relative to recent average).
3. **The Ceiling** — Price ≤ 125% × SMA200 (relaxed from 110%; avoids buying extreme over-extensions while allowing stronger names to enter — updated in sidebar slider at `line 192`).
4. **The Floor** — Price > SMA200 (never buys falling knives).

When **all four** are true on the same day for a ticker, it becomes eligible. Cash is split evenly: `alloc = cash_for_entries / len(eligible_tickers)` (`line 325`). No stock is entered twice (already-active tickers excluded, `line 319`).

---

## 6. Exit / Sell Conditions

### Target exit (`line 277`)
```
if (current_value - total_cost) / total_cost >= profit_pct / 100:
    close, add cash + profit to pool, outcome = WIN, exit = "Target"
```
Default `profit_pct = 3.5%` (was 6.28%). Lower target = faster exits = more trades in a 13-day window.

### Period-end closeout (`line 348`)
Every open position is forcibly closed at `last_date` (end of Oct 30 / selected window):
- If positive → `Outcome = WIN`, `Exit = "Period End"`
- If negative → `Outcome = LOSS`, `Exit = "Period End"`

This eliminates the old “OPEN” category that inflated metrics incorrectly over a 30-day window.

### Dip-averaging (reserve fund)
- At –5% from entry: buy more if reserve > 0 (`line 294`)
- At –10% from entry: buy more if reserve > 0 (`line 303`)
- Both track `avg5_done` / `avg10_done` to prevent double-averaging.

---

## 7. Performance Metrics (Test Period)

### How computed (all from `dma_car_backtester.py:407–478`)

| Metric | Formula / Method | Source Line |
|---|---|---|
| **Annual Return** | `(final / seed)^(252 / days) − 1` | 411 |
| **Sharpe Ratio** | `annual_return / annual_volatility` (rf = 0) | 415 |
| **Max Drawdown** | `min((equity − peak) / peak)` over window | 420 |
| **Win Rate** | `wins / total_closed_trades × 100` | 377 |
| **Annual Volatility** | `std(daily_return) × √252` | 413 |
| **Drawdown Patterns** | Count of separate `equity < peak` episodes | 422 |
| **Number of Trades** | `len(df_trades)` | 424 |
| **Average Holding Period** | `mean(exit_date − entry_date)` | 374 |
| **Profit per Trade** | `mean(PnL ₹)` | 425 |
| **Additional:** Return / Drawdown | `annual_return / max_drawdown` | 429 |
| **Additional:** Best Trade | `max(PnL ₹)` | 426 |
| **Additional:** Worst Trade | `min(PnL ₹)` | 427 |
| **Additional:** Trades per Week | `n_trades / (days / 5)` | 428 |

### Sample / interpretation note (important for grading)
With only ~13 trading days (Oct 13 → Oct 30), expect **small trade counts**. The metric table includes a warning (`line 479`): “Only N trades — Win Rate and Profit per Trade are statistically weak below ~20 trades.” For the presentation, we recommend also referencing the long-horizon stability run from the optimizer (walk-forward in-sample / out-of-sample) to show the strategy is not curve-fitted.

### CSV outputs
- Download button on results page writes `performance_metrics.csv` with Metric / Value / Calculation columns.
- `portfolio_daily_closes.csv` writes all selected tickers’ close prices over the window.

---

## 8. How to Run

```bash
# Install
pip install streamlit yfinance pandas numpy plotly optuna

# Main backtester (window: Oct 1 – Oct 30, 2026)
streamlit run dma_car_backtester.py

# Optimizer (find best profit / SMA parameters)
streamlit run optimizer.py
```

**In the sidebar:**
- Set Start Date = `2026-10-01`, End Date = `2026-10-30`
- Profit Target = `3.5%` (default; adjustable)
- SMA200 Buffer = `125%` (default; adjustable 100–140)
- Select stocks (blank = all Nifty 250), set tranches / dip options, press **RUN PORTFOLIO BACKTEST**

---

## 9. Group Contributions

> Per the outline: “Each student MUST specify his or her own contribution.”

| Student | Contribution |
|---|---|
| **Student A** | Strategy design & entry logic (stacked-SMA + anti-FOMO ceiling); wrote the green-flag checklist and justification text |
| **Student B** | Code build — backtester (`dma_car_backtester.py`) loop, trade logging, period-end closeout, metrics table, CSV export |
| **Student C** | Optimization — `optimizer.py` (Optuna + walk-forward), parameter search ranges, in-sample vs out-of-sample validation charts |
| **Student D** | Data & presentation — asset selection (Nifty 250), price CSV generation, metric interpretation, slide formatting |
| **Student E** (optional / if 5) | Testing & verification — synthetic-data metric validation, sample-size analysis, grading-readiness checks |

All contributions are documented in this README and in the code comments (each major block has a header: `# ── Data Fetching ──`, `# ── Sidebar ──`, etc.).

---

## 10. Notes for Presentation & Grading

- **Methodology:** Clearly described (stacked SMA entry, compound exit, reserve fund dips, dynamic split allocation).
- **Justification:** Trend-following with anti-FOMO is defensible for a 30-day window; explained in the “Perfect Entry Checklist” banner.
- **P&L & metrics:** All 9 required metrics present (+ 4 additional). CSV files submitted.
- **Sample size honesty:** We explicitly note that with ~13 trading days, some metrics (Win Rate, Profit/Trade, Drawdown Patterns) rely on only 2–5 trades. We mitigate by showing the optimizer’s long-horizon stability reference.
- **CAR filter:** Fully removed (per instruction); not used in any metric.
- **Assets:** All traded >1 year; CSV submitted.
- **No late submission:** All files complete as of submission date.

---

## 11. References to Project Outlines

### Trading Strategy Outline (from pasted content)
- ✅ Portfolio created (Nifty 250, selectable subset)
- ✅ Assets publicly traded >1 year (Nifty 250 large-cap)
- ✅ Automatic update: backtester + optimizer both refresh from yfinance
- ✅ Forecasting included: optimizer predicts best parameter set via walk-forward
- ✅ Justification: Section 2 of this README + header banner in app
- ✅ Metrics evaluated: Section 7 + CSV outputs
- ✅ CSV submitted: `portfolio_daily_closes.csv`
- ✅ Presentation: README + slides (this file + metrics table)
- ✅ Team: 4–5 students with contributions specified (Section 9)

### Performance Metric Outline (from pasted content)
- ✅ Annual Return — computed + in table
- ✅ Sharpe Ratio — computed + in table
- ✅ Max Drawdown — computed + in table
- ✅ Win Rate — computed + in table
- ✅ Annual Volatility — computed + in table
- ✅ Drawdown Patterns — computed (episode count) + in table
- ✅ Number of Trades — computed + in table
- ✅ Average Holding Period — computed + in table
- ✅ Profit per Trade — computed + in table
- ✅ Additional Metrics — Return/Drawdown, Best/Worst Trade, Trades/Week (with formulas)
- ✅ Table format — metric / value / calculation, readable, comparable
- ✅ CSV — `performance_metrics.csv`

---

*Document written for the group project submission. All code, metrics, and explanations reflect the current state of the repository (post-CAR-removal, post-relaxation, with period-end closeout and full metric table).*
