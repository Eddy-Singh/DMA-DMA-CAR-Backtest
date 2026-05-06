"""
DMA CAR Strategy — Hyperparameter Optimizer
============================================
Uses Optuna (Bayesian optimisation) + walk-forward validation to find the
best combination of:
    • Profit target %
    • SMA200 buffer %
    • SMA fast / mid / slow periods

Objective : Calmar Ratio  (annualised return ÷ max drawdown)
            — rewards profit while penalising large drawdowns.

Walk-forward split
    IN-SAMPLE  : first 80 % of the date range  → optimised
    OUT-OF-SAMPLE : last 20 %                  → blind validation
    A strategy that looks good on both is trustworthy.

Run:
    pip install streamlit yfinance pandas numpy plotly optuna
    streamlit run optimizer.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import optuna
from datetime import datetime, timedelta
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DMA CAR Optimizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root { --bg:#0d1117; --surface:#161b22; --border:#21262d; --accent:#00d4aa; --text:#e6edf3; --muted:#7d8590; }
html,body,[data-testid="stAppViewContainer"]{ background:var(--bg)!important; color:var(--text)!important; font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"]{ background:var(--surface)!important; border-right:1px solid var(--border); }
h1,h2,h3{ font-family:'Space Mono',monospace!important; }
.card{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px 24px; text-align:center; }
.card:hover{ border-color:var(--accent); }
.val{ font-size:1.5rem; font-weight:700; font-family:'Space Mono',monospace; }
.lbl{ font-size:.72rem; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; margin-top:4px; }
.green{color:#00d4aa;} .red{color:#ff6b6b;} .yellow{color:#ffd166;}
div[data-testid="stButton"]>button{ background:var(--accent)!important; color:#0d1117!important; font-weight:700!important; border:none!important; border-radius:8px!important; padding:10px 28px!important; font-family:'Space Mono',monospace!important; width:100%; }
div[data-testid="stButton"]>button:hover{ background:#00b894!important; }
.warn-box{ background:#ffd16615; border:1px solid #ffd16655; border-radius:10px; padding:14px 18px; margin-bottom:18px; font-size:.88rem; }
.header-banner{ background:linear-gradient(135deg,#00d4aa15 0%,#0d111700 60%); border:1px solid #00d4aa33; border-radius:16px; padding:28px 32px; margin-bottom:28px; }
</style>
""", unsafe_allow_html=True)

# ── Ticker pool (same as backtester) ─────────────────────────────────────────
NIFTY_250 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "TECHM.NS", "NTPC.NS", "HCLTECH.NS", "ONGC.NS",
    "COALINDIA.NS", "TATAMOTORS.NS", "BAJAJFINSV.NS", "ADANIPORTS.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "CIPLA.NS", "JSWSTEEL.NS", "TATACONSUM.NS", "EICHERMOT.NS",
    "BRITANNIA.NS", "HINDALCO.NS", "BPCL.NS", "GRASIM.NS", "APOLLOHOSP.NS",
    "SBILIFE.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "ADANIENT.NS", "VEDL.NS",
    "TATAPOWER.NS", "DLF.NS", "PIDILITIND.NS", "HAVELLS.NS", "SIEMENS.NS",
    "ZOMATO.NS", "DIXON.NS", "VOLTAS.NS", "TITAN.NS", "PERSISTENT.NS",
    "LTIM.NS", "COFORGE.NS", "MPHASIS.NS", "HDFCAMC.NS", "CHOLAFIN.NS",
    "TATASTEEL.NS", "JINDALSTEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS",
]
NIFTY_250 = sorted(list(set([t for t in NIFTY_250 if "." in t])))

# ── Data fetch (cached) ───────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(tickers, start_date, end_date):
    start_fetch = pd.to_datetime(start_date) - pd.Timedelta(days=400)
    df = yf.download(tickers, start=start_fetch, end=end_date,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df_close = df["Close"] if "Close" in df.columns.levels[0] else df.xs(
            "Close", axis=1, level=0)
    else:
        df_close = df[["Close"]]
    df_close.index = df_close.index.tz_localize(None)
    return df_close.ffill()

# ── Pure backtest engine (no Streamlit — called by Optuna) ────────────────────


def run_backtest(
    df_close: pd.DataFrame,
    date_range,
    profit_pct: float,
    sma_buffer: float,        # e.g. 110  means 110 %
    sma_fast: int,            # e.g. 50
    sma_mid: int,             # e.g. 100
    sma_slow: int,            # e.g. 200
    use_car: bool,
    starting_capital: float = 100_000.0,
) -> dict:
    """
    Runs the full DMA-CAR backtest on the given date slice and returns
    a dict of performance metrics.
    """
    df = df_close.copy()

    sma_f = df.rolling(sma_fast).mean()
    sma_m = df.rolling(sma_mid).mean()
    sma_s = df.rolling(sma_slow).mean()

    signals = (
        (df < (sma_buffer / 100.0) * sma_s) &
        (df > sma_f) &
        (sma_f > sma_m) &
        (sma_m > sma_s)
    )

    if use_car:
        car = _compute_car(df)
        signals = signals & car

    valid = df[df.index >= pd.to_datetime(date_range[0])].index
    if len(valid) < 20:
        return {"calmar": -999, "net_pct": -999, "max_dd": 0, "win_rate": 0, "trades": 0}

    cash = starting_capital
    active = {}
    equity = []
    trades_log = []
    profit_mult = 1.0 + profit_pct / 100.0

    for date in valid:
        prices_today = df.loc[date]
        signals_today = signals.loc[date]

        # Exits
        to_close = []
        for t, pos in list(active.items()):
            p = prices_today.get(t, np.nan)
            if pd.isna(p):
                continue
            if p >= pos["target"]:
                pnl = pos["shares"] * p - pos["cost"]
                cash += pos["shares"] * p
                trades_log.append(pnl / pos["cost"])
                to_close.append(t)
        for t in to_close:
            del active[t]

        # Entries
        if cash > 500:
            eligible = [
                t for t in df.columns
                if signals_today.get(t, False)
                and t not in active
                and not pd.isna(prices_today.get(t, np.nan))
            ]
            if eligible:
                alloc = cash / len(eligible)
                for t in eligible:
                    p = prices_today[t]
                    active[t] = {
                        "shares": alloc / p,
                        "cost":   alloc,
                        "target": p * profit_mult,
                    }
                cash = 0.0

        open_val = sum(
            pos["shares"] * prices_today.get(t, np.nan)
            for t, pos in active.items()
            if not pd.isna(prices_today.get(t, np.nan))
        )
        equity.append(cash + open_val)

    if not equity or equity[0] == 0:
        return {"calmar": -999, "net_pct": -999, "max_dd": 0, "win_rate": 0, "trades": 0}

    eq = np.array(equity, dtype=float)
    net_pct = (eq[-1] - eq[0]) / eq[0] * 100.0

    # Max drawdown
    roll_max = np.maximum.accumulate(eq)
    dd = (eq - roll_max) / roll_max
    max_dd = float(abs(dd.min())) if dd.min() < 0 else 1e-6

    # Annualised return
    n_years = len(valid) / 252.0
    ann_ret = ((eq[-1] / eq[0]) ** (1.0 / max(n_years, 0.1)) - 1.0) * 100.0

    calmar = ann_ret / (max_dd * 100.0) if max_dd > 0 else ann_ret

    win_rate = (
        (np.array(trades_log) > 0).mean() * 100.0
        if trades_log else 0.0
    )

    return {
        "calmar":   calmar,
        "net_pct":  net_pct,
        "ann_ret":  ann_ret,
        "max_dd":   max_dd * 100.0,
        "win_rate": win_rate,
        "trades":   len(trades_log),
        "equity":   eq.tolist(),
        "dates":    [str(d.date()) for d in valid],
    }


def _compute_car(df_close: pd.DataFrame, window: int = 252, lookback: int = 10) -> pd.DataFrame:
    result = pd.DataFrame(False, index=df_close.index,
                          columns=df_close.columns)
    for col in df_close.columns:
        s = df_close[col].dropna()
        n = len(s)
        if n < window + lookback:
            continue
        vals = s.values.astype(np.float64)
        prefix = np.concatenate([[0.0], np.cumsum(vals)])
        cum_avg = np.full(n, np.nan)
        for i in range(window - 1, n):
            hp = (i - window + 1) + int(np.argmax(vals[i - window + 1: i + 1]))
            cum_avg[i] = (prefix[i + 1] - prefix[hp]) / (i - hp + 1)
        car = np.zeros(n, dtype=bool)
        start_i = window + lookback - 2
        for i in range(start_i, n):
            p10 = vals[i - lookback + 1: i + 1]
            c10 = cum_avg[i - lookback + 1: i + 1]
            if np.any(np.isnan(c10)):
                continue
            if np.all(p10 > c10):
                car[i] = True
        result.loc[s.index, col] = car
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Optimizer Config")
    st.markdown("---")

    default_end = datetime.today()
    default_start = default_end - timedelta(days=5 * 365)

    start_date = st.date_input("History Start", value=default_start)
    end_date = st.date_input("History End",   value=default_end)

    st.markdown("### 🔀 Walk-Forward Split")
    split_pct = st.slider(
        "In-sample %  (rest = out-of-sample)",
        50, 85, 80, 5,
        help="80 means the first 80% of dates are used to optimise; "
             "the last 20% is a blind test you never optimise against."
    )

    st.markdown("### 🎛️ Parameter Search Space")
    st.caption("Set the min/max range Optuna will search.")

    profit_min, profit_max = st.slider(
        "Profit Target % range",  1.0, 30.0, (3.0, 15.0), 0.5)
    buffer_min, buffer_max = st.slider(
        "SMA200 Buffer % range", 100, 130, (100, 120), 1)
    fast_min,   fast_max = st.slider(
        "Fast SMA period range",   20,  80, (40,  60), 1)
    mid_min,    mid_max = st.slider(
        "Mid SMA period range",    60, 150, (80, 120), 1)
    slow_min,   slow_max = st.slider(
        "Slow SMA period range",  150, 300, (180, 220), 1)

    use_car = st.checkbox("Include CAR Filter", value=True)

    st.markdown("### ⚡ Trials")
    n_trials = st.slider("Optuna trials", 30, 300, 100, 10,
                         help="More trials = better optimisation but slower. "
                              "100 is a good balance.")

    all_tickers = [t.replace(".NS", "") for t in NIFTY_250]
    chosen = st.multiselect("Stocks (blank = default pool)",
                            all_tickers, placeholder="Default pool")
    ticker_pool = [f"{t}.NS" for t in chosen] if chosen else NIFTY_250

    run_btn = st.button("▶ RUN OPTIMISATION")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1 style="margin:0;font-size:1.8rem;letter-spacing:.04em">🔬 DMA CAR Parameter Optimizer</h1>
    <p style="margin:6px 0 0;color:#7d8590;font-size:.9rem">
        Bayesian Optimisation (Optuna) · Walk-Forward Validation · Calmar Ratio Objective
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warn-box">
⚠️ <strong>Overfitting Warning</strong> — optimising on the same data you evaluate on produces 
parameters that look perfect historically but fail live. This tool splits your data:
<strong>in-sample</strong> (optimised) vs <strong>out-of-sample</strong> (blind test). 
Only trust parameters that perform well on <em>both</em>.
</div>
""", unsafe_allow_html=True)

if run_btn:

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    with st.spinner("Downloading market data…"):
        df_close = fetch_data(ticker_pool, start_date, end_date)

    all_dates = df_close[df_close.index >= pd.to_datetime(start_date)].index
    if len(all_dates) < 100:
        st.error("Not enough trading days in the selected range.")
        st.stop()

    split_idx = int(len(all_dates) * split_pct / 100)
    in_dates = (all_dates[0],  all_dates[split_idx - 1])
    out_dates = (all_dates[split_idx], all_dates[-1])

    st.info(
        f"📅 In-sample: **{in_dates[0].date()}  →  {in_dates[1].date()}**  "
        f"({split_idx} days)   |   "
        f"Out-of-sample: **{out_dates[0].date()}  →  {out_dates[1].date()}**  "
        f"({len(all_dates) - split_idx} days)"
    )

    # ── 2. Optuna objective ───────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        profit_pct = trial.suggest_float(
            "profit_pct", profit_min, profit_max, step=0.1)
        sma_buffer = trial.suggest_int("sma_buffer",   buffer_min, buffer_max)
        sma_fast = trial.suggest_int("sma_fast",     fast_min,   fast_max)
        sma_mid = trial.suggest_int("sma_mid",      mid_min,    mid_max)
        sma_slow = trial.suggest_int("sma_slow",     slow_min,   slow_max)

        # Constraint: fast < mid < slow
        if not (sma_fast < sma_mid < sma_slow):
            return -999.0

        result = run_backtest(
            df_close, in_dates,
            profit_pct, sma_buffer, sma_fast, sma_mid, sma_slow,
            use_car=use_car,
        )
        return result["calmar"]

    # ── 3. Run optimisation with live progress ────────────────────────────────
    progress_bar = st.progress(0, text="Starting Optuna…")
    status_text = st.empty()
    best_so_far = st.empty()
    history_vals = []

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    for i in range(n_trials):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        trial_val = study.trials[-1].value
        history_vals.append(trial_val if trial_val is not None else -999)

        pct = (i + 1) / n_trials
        progress_bar.progress(pct, text=f"Trial {i+1}/{n_trials}")

        best = study.best_trial
        best_so_far.markdown(
            f"**Best so far** — Calmar: `{best.value:.3f}` | "
            f"profit_pct: `{best.params['profit_pct']:.1f}%` | "
            f"buffer: `{best.params['sma_buffer']}%` | "
            f"SMAs: `{best.params['sma_fast']}/{best.params['sma_mid']}/{best.params['sma_slow']}`"
        )

    progress_bar.empty()
    status_text.empty()
    best_so_far.empty()

    # ── 4. Best parameters ────────────────────────────────────────────────────
    bp = study.best_params
    st.success("✅ Optimisation complete!")

    st.markdown("### 🏆 Best Parameters Found")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in zip(
        [c1, c2, c3, c4, c5],
        ["Profit Target", "SMA200 Buffer", "Fast SMA", "Mid SMA", "Slow SMA"],
        [f"{bp['profit_pct']:.1f}%", f"{bp['sma_buffer']}%",
         str(bp['sma_fast']), str(bp['sma_mid']), str(bp['sma_slow'])]
    ):
        col.markdown(
            f'<div class="card"><div class="val green">{val}</div><div class="lbl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 5. In-sample vs Out-of-sample comparison ──────────────────────────────
    st.markdown("### 📊 In-Sample vs Out-of-Sample Validation")

    res_in = run_backtest(df_close, in_dates,  **{k: bp[k] for k in [
                          "profit_pct", "sma_buffer", "sma_fast", "sma_mid", "sma_slow"]}, use_car=use_car)
    res_out = run_backtest(df_close, out_dates, **{k: bp[k] for k in [
                           "profit_pct", "sma_buffer", "sma_fast", "sma_mid", "sma_slow"]}, use_car=use_car)

    def _colour(v, good=0):
        return "green" if v > good else "red"

    metrics = [
        ("Net Return %",    f"{res_in['net_pct']:+.1f}%",
         f"{res_out['net_pct']:+.1f}%",   res_in['net_pct'],  res_out['net_pct']),
        ("Ann. Return %",   f"{res_in['ann_ret']:+.1f}%",
         f"{res_out['ann_ret']:+.1f}%",   res_in['ann_ret'],  res_out['ann_ret']),
        ("Max Drawdown %",  f"{res_in['max_dd']:.1f}%",
         f"{res_out['max_dd']:.1f}%",     -res_in['max_dd'],  -res_out['max_dd']),
        ("Calmar Ratio",    f"{res_in['calmar']:.2f}",
         f"{res_out['calmar']:.2f}",       res_in['calmar'],   res_out['calmar']),
        ("Win Rate %",      f"{res_in['win_rate']:.1f}%",
         f"{res_out['win_rate']:.1f}%",   res_in['win_rate'], res_out['win_rate']),
        ("Closed Trades",   str(res_in['trades']),           str(
            res_out['trades']),           1,                  1),
    ]

    header_cols = st.columns([2, 1, 1])
    header_cols[0].markdown("**Metric**")
    header_cols[1].markdown("**In-Sample**")
    header_cols[2].markdown("**Out-of-Sample**")
    st.markdown("---")
    for label, v_in, v_out, raw_in, raw_out in metrics:
        rc = st.columns([2, 1, 1])
        rc[0].write(label)
        rc[1].markdown(
            f'<span class="{_colour(raw_in)}">{v_in}</span>', unsafe_allow_html=True)
        rc[2].markdown(
            f'<span class="{_colour(raw_out)}">{v_out}</span>', unsafe_allow_html=True)

    # ── 6. Equity curves side by side ─────────────────────────────────────────
    st.markdown("### 📈 Equity Curves")
    eq_c1, eq_c2 = st.columns(2)

    for col, res, title, colour in [
        (eq_c1, res_in,  "In-Sample",       "#00d4aa"),
        (eq_c2, res_out, "Out-of-Sample ✅", "#ffd166"),
    ]:
        with col:
            if res.get("equity"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=res["dates"], y=res["equity"],
                    fill="tozeroy", line=dict(color=colour, width=2), name=title
                ))
                fig.add_hline(y=res["equity"][0],
                              line_dash="dash", line_color="#7d8590")
                fig.update_layout(
                    title=title,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6edf3", height=320,
                    xaxis=dict(gridcolor="#21262d"),
                    yaxis=dict(gridcolor="#21262d",
                               title="Portfolio Value (₹)")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No equity data.")

    # ── 7. Optimisation history ───────────────────────────────────────────────
    st.markdown("### 🔁 Optimisation History")
    hist_c1, hist_c2 = st.columns(2)

    with hist_c1:
        best_running = [max(history_vals[: i + 1])
                        for i in range(len(history_vals))]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            y=history_vals, mode="markers",
            marker=dict(color="#7d8590", size=4), name="Trial value"
        ))
        fig_hist.add_trace(go.Scatter(
            y=best_running, mode="lines",
            line=dict(color="#00d4aa", width=2), name="Best so far"
        ))
        fig_hist.update_layout(
            title="Calmar Ratio per Trial",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", height=320,
            xaxis=dict(gridcolor="#21262d", title="Trial #"),
            yaxis=dict(gridcolor="#21262d", title="Calmar Ratio"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with hist_c2:
        # Parameter importance
        try:
            importances = optuna.importance.get_param_importances(study)
            fig_imp = px.bar(
                x=list(importances.values()),
                y=list(importances.keys()),
                orientation="h",
                title="Parameter Importance",
                color=list(importances.values()),
                color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3", height=320,
                xaxis=dict(gridcolor="#21262d", title="Relative Importance"),
                yaxis_title="", coloraxis_showscale=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.info("Run more trials to compute parameter importance.")

    # ── 8. Top 15 trials table ────────────────────────────────────────────────
    st.markdown("### 📋 Top 15 Trials")
    trials_df = pd.DataFrame([
        {
            "Calmar":     round(t.value, 3),
            "Profit %":   t.params.get("profit_pct"),
            "Buffer %":   t.params.get("sma_buffer"),
            "Fast SMA":   t.params.get("sma_fast"),
            "Mid SMA":    t.params.get("sma_mid"),
            "Slow SMA":   t.params.get("sma_slow"),
        }
        for t in study.trials if t.value is not None and t.value > -999
    ]).sort_values("Calmar", ascending=False).head(15).reset_index(drop=True)

    st.dataframe(trials_df.style.background_gradient(
        subset=["Calmar"], cmap="YlGn"), use_container_width=True)

    # ── 9. Paste-ready config ─────────────────────────────────────────────────
    st.markdown("### 📋 Plug These Into the Backtester")
    st.code(
        f"Profit Target  : {bp['profit_pct']:.1f} %\n"
        f"SMA200 Buffer  : {bp['sma_buffer']} %\n"
        f"Fast SMA period: {bp['sma_fast']}\n"
        f"Mid  SMA period: {bp['sma_mid']}\n"
        f"Slow SMA period: {bp['sma_slow']}\n"
        f"CAR Filter     : {'ON' if use_car else 'OFF'}",
        language="text"
    )

    st.markdown("""
<div class="warn-box">
🔑 <strong>How to interpret results:</strong><br>
• <strong>Out-of-sample Calmar &gt; 0.5</strong> → strategy has genuine edge, not just curve-fit<br>
• <strong>Out-of-sample close to in-sample</strong> → parameters generalise well<br>
• <strong>Out-of-sample much worse</strong> → overfitting; try widening search ranges or fewer parameters<br>
• <strong>Parameter Importance chart</strong> → focus on what actually matters; ignore low-importance params
</div>
""", unsafe_allow_html=True)
