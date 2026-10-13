"""
DMA Strategy Backtester — Nifty 250 (Portfolio Compounding Edition)
===================================================================
Strategy Rules:
  ENTRY  : 110% × SMA200 > Price > SMA50 > SMA100 > SMA200
  EXIT   : profit target % (slider, default 3.5%) or end of period
  CAPITAL: Tranche 1 & 2 logic. Available cash is always split evenly across
           ALL stocks signaling on that specific day. Profits are compounded.

Run:
    pip install streamlit yfinance pandas numpy plotly
    streamlit run dma_car_backtester.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DMA CAR Portfolio Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #21262d;
    --accent:   #00d4aa;
    --text:     #e6edf3;
    --muted:    #7d8590;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color .2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-val  { font-size: 1.6rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.metric-lbl  { font-size: .75rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; margin-top: 4px; }
.green  { color: #00d4aa; }
.red    { color: #ff6b6b; }
.yellow { color: #ffd166; }

div[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    font-family: 'Space Mono', monospace !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover { background: #00b894 !important; }

.header-banner {
    background: linear-gradient(135deg, #00d4aa15 0%, #0d111700 60%);
    border: 1px solid #00d4aa33;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
}
</style>
""", unsafe_allow_html=True)

# ── Nifty 250 tickers ────────────────────────────────────────────────────────
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
    "MOTHERSON.NS", "DABUR.NS", "MCDOWELL-N.NS", "GODREJCP.NS", "MARICO.NS",
    "BERGEPAINT.NS", "TORNTPHARM.NS", "AMBUJACEM.NS", "ICICIGI.NS", "BANDHANBNK.NS",
    "BIOCON.NS", "MUTHOOTFIN.NS", "LUPIN.NS", "PFC.NS", "RECLTD.NS",
    "SBICARD.NS", "IRCTC.NS", "BALKRISIND.NS", "INDIGO.NS", "NAUKRI.NS",
    "GAIL.NS", "HDFCLIFE.NS", "CHOLAFIN.NS", "PAGEIND.NS", "COFORGE.NS",
    "MPHASIS.NS", "PERSISTENT.NS", "LTIM.NS", "OFSS.NS", "HDFCAMC.NS",
    "CONCOR.NS", "COLPAL.NS", "WHIRLPOOL.NS", "JUBLFOOD.NS", "ALKEM.NS",
    "IPCALAB.NS", "LICI.NS", "FEDERALBNK.NS", "AUBANK.NS", "IDFCFIRSTB.NS",
    "CANBK.NS", "PNB.NS", "BANKBARODA.NS", "NMDC.NS", "SAIL.NS",
    "NATIONALUM.NS", "NBCC.NS", "HUDCO.NS", "RVNL.NS", "IRFC.NS",
    "RAILTEL.NS", "NHPC.NS", "SJVN.NS", "BHEL.NS", "HAL.NS",
    "BEL.NS", "BEML.NS", "GRSE.NS", "COCHINSHIP.NS", "MAZAGON.NS",
    "ASTRAL.NS", "POLYCAB.NS", "KEI.NS", "RRKABEL.NS", "KPIL.NS",
    "KALPATPOWR.NS", "GPIL.NS", "JKCEMENT.NS", "RAMCOCEM.NS", "HEIDELBERG.NS",
    "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "DELHIVERY.NS", "POLICYBZR.NS",
    "TATASTEEL.NS", "JINDALSTEL.NS", "MOIL.NS", "RATNAMANI.NS", "WELSPUNIND.NS",
    "SUPREMEIND.NS", "FINOLEX.NS", "TDPOWERSYS.NS", "ABB.NS", "CUMMINSIND.NS",
    "THERMAX.NS", "BHARAT FORGE.NS", "KAJARIACER.NS", "CERA.NS", "ORIENTCEM.NS",
    "SUNTV.NS", "ZEEL.NS", "PVR.NS", "INOXLEISUR.NS", "BATAINDIA.NS",
    "RELAXO.NS", "VBL.NS", "RADICO.NS", "UNITDSPR.NS", "ATUL.NS",
    "DEEPAKNTR.NS", "NAVINFLUOR.NS", "SRF.NS", "PIIND.NS", "COROMANDEL.NS",
    "UPL.NS", "CHAMBLFERT.NS", "GNFC.NS", "GSFC.NS", "TATACOMM.NS",
    "DIXON.NS", "AMBER.NS", "VOLTAS.NS", "BLUESTAR.NS", "CROMPTON.NS",
    "ORIENTELEC.NS", "MINDA.NS", "EXIDEIND.NS", "AMARAJABAT.NS", "BOSCHLTD.NS",
    "SUPRAJIT.NS", "CRAFTSMAN.NS", "SCHAEFFLER.NS", "TIMKEN.NS",
    "GRINDWELL.NS", "CARBORUNIV.NS", "CRISIL.NS", "ICRA.NS", "CARE.NS",
    "CDSL.NS", "BSE.NS", "MCX.NS", "IEX.NS", "MSEI.NS",
    "MAXHEALTH.NS", "FORTIS.NS", "NARAYANA.NS", "ASTER.NS", "KIMS.NS",
    "METROPOLIS.NS", "LALPATHLAB.NS", "THYROCARE.NS", "KRSNAA.NS", "VIJAYA.NS",
    "AIAENG.NS", "ELGIEQUIP.NS", "GREAVESCOT.NS", "KIRLOSENG.NS", "IGARASHI.NS",
    "SHOPERSTOP.NS", "VMART.NS", "TRENT.NS", "METRO.NS", "BATA.NS",
    "BLUEDART.NS", "GICRE.NS", "NIACL.NS", "STARHEALTH.NS", "GODIGIT.NS",
    "MANAPPURAM.NS", "LICHSGFIN.NS", "CANFINHOME.NS", "AAVAS.NS", "HOMEFIRST.NS",
    "APTUS.NS", "REPCO.NS", "MAHINDRA&MAHINDRA.NS", "ESCORTS.NS", "SONACOMS.NS",
    "TIINDIA.NS", "ENDURANCE.NS", "VARROC.NS", "GABRIEL.NS",
    "GMRINFRA.NS", "IRB.NS", "KNRCON.NS", "ASHOKA.NS", "HG INFRA.NS",
    "PRESTIGE.NS", "GODREJPROP.NS", "PHOENIXLTD.NS", "BRIGADE.NS", "SOBHA.NS",
    "OBEROIRLTY.NS", "MAHLIFE.NS", "LODHA.NS", "SUNTECKREAL.NS", "ANANTRAJ.NS",
    "ZYDUSLIFE.NS", "GLENMARK.NS", "GRANULES.NS", "LAURUSLABS.NS", "SUVEN.NS",
    "DIVI.NS", "NATCOPHARM.NS", "AJANTPHARM.NS", "JBCHEPHARM.NS", "ERIS.NS",
]
NIFTY_250 = sorted(list(set([t for t in NIFTY_250 if "." in t])))[:250]

# ── Data Fetching ─────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_market_data(tickers, start_date, end_date):
    start_fetch = pd.to_datetime(start_date) - pd.Timedelta(days=300)
    df = yf.download(tickers, start=start_fetch, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.levels[0]:
            df_close = df['Close']
        else:
            df_close = df.xs('Close', axis=1, level=0)
    else:
        df_close = df[['Close']]
    df_close.index = df_close.index.tz_localize(None)
    return df_close.ffill()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Strategy Config")
    st.markdown("---")

    default_end = datetime.today()
    default_start = default_end - timedelta(days=2 * 365)

    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date",   value=default_end)

    st.markdown("### 💰 Capital Allocation")
    tranche_1 = st.number_input("Tranche 1 (Day 1) ₹", value=25000, step=5000)
    tranche_2 = st.number_input("Tranche 2 (Day 20) ₹", value=25000, step=5000)
    reserve_cap = st.number_input("Reserve for Dips ₹", value=50000, step=5000)
    st.caption(
        f"**Total Starting Capital: ₹{tranche_1 + tranche_2 + reserve_cap:,.0f}**")

    st.markdown("### 📉 Dip Buying (Averaging Down)")
    use_avg5 = st.checkbox("Buy at –5% dip", value=True)
    use_avg10 = st.checkbox("Buy at –10% dip", value=True)

    st.markdown("### Entry & Exit")
    profit_pct = st.slider("Profit Target (%)", 1.0, 20.0, 3.5, 0.01)
    sma_buffer = st.slider("SMA200 Buffer (%)", 100, 140, 125, 1)

    all_tickers = [t.replace(".NS", "") for t in NIFTY_250]
    chosen = st.multiselect("Select stocks (blank = All Nifty 250)",
                            all_tickers, placeholder="All Nifty 250")
    ticker_pool = [f"{t}.NS" for t in chosen] if chosen else NIFTY_250

    run_btn = st.button("▶ RUN PORTFOLIO BACKTEST")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-banner">
    <h1 style="margin:0;font-size:2rem;letter-spacing:.04em">📈 DMA CAR Portfolio Backtester</h1>
    <p style="margin:10px 0; color:#e6edf3; font-size:1.1rem; font-weight:500;">
        A trend-following system designed to capture momentum in India's top 250 companies.
    </p>
    <div style="background: rgba(0,212,170,0.1); border-left: 4px solid #00d4aa; padding: 15px; margin: 20px 0;">
        <h4 style="margin-top:0; color:#00d4aa; font-family:'Space Mono';">The "Perfect Entry" Checklist</h4>
        <p style="font-size:0.9rem; color:#7d8590; line-height:1.6;">
            For the system to buy a stock, it must pass <b>four strict health checks</b> simultaneously:
            <br>✅ <b>The Trend:</b> The 50-day average must be above the 100-day, which must be above the 200-day (Clear upward momentum).
            <br>✅ <b>The Value:</b> The current price must be <i>higher</i> than the 50-day average (The stock is currently "hot").
            <br>✅ <b>The Ceiling:</b> The price must stay within ~25% of its 200-day average (Relaxed anti-FOMO rule — allows stronger momentum names to enter).
            <br>✅ <b>The Floor:</b> The price must stay above the 200-day average (Ensures we don't buy "falling knives").
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

if run_btn:
    status_box = st.info(
        "Fetching batch data for all tickers... This might take ~10 seconds.")

    # 1. Fetch & Compute Signals
    df_close = fetch_market_data(ticker_pool, start_date, end_date)

    sma50 = df_close.rolling(50).mean()
    sma100 = df_close.rolling(100).mean()
    sma200 = df_close.rolling(200).mean()

    signals = (df_close < (sma_buffer / 100.0) * sma200) & \
              (df_close > sma50) & \
              (sma50 > sma100) & \
              (sma100 > sma200)

    # 2. Setup Portfolio Timeline
    valid_dates = df_close[df_close.index >= pd.to_datetime(start_date)].index
    if len(valid_dates) == 0:
        st.error("No valid trading days found in the selected date range.")
        st.stop()

    status_box.info("Simulating chronological portfolio compounding...")

    first_date = valid_dates[0]
    tranche_2_date = valid_dates[20] if len(valid_dates) > 20 else None

    cash_for_entries = 0.0
    cash_reserve = float(reserve_cap)
    active_positions = {}
    trades = []
    portfolio_history = []
    trade_counter = 0

    # 3. Daily Loop
    for date in valid_dates:
        # Deploy Tranches exactly on Day 1 and Day 20
        if date == first_date:
            cash_for_entries += tranche_1
        if tranche_2_date and date == tranche_2_date:
            cash_for_entries += tranche_2

        prices_today = df_close.loc[date]
        signals_today = signals.loc[date]

        # A. Check Exits and Dips for Active Positions
        tickers_to_close = []
        for ticker, pos in list(active_positions.items()):
            p = prices_today[ticker]
            if pd.isna(p):
                continue

            current_val = pos['shares'] * p
            pnl_pct = (current_val - pos['total_cost']) / pos['total_cost']

            # Exit Check (Hits Profit Target)
            if pnl_pct >= (profit_pct / 100.0):
                cash_for_entries += current_val  # Compounding: Reinvest principal + profit
                trades.append({
                    "Trade #": pos['trade_id'], "Ticker": ticker.replace(".NS", ""),
                    "Entry Date": pos['entry_date'].date(), "Exit Date": date.date(),
                    "Entry Price": pos['initial_price'], "Exit Price": p,
                    "Invested ₹": pos['total_cost'], "Exit Value ₹": current_val,
                    "PnL ₹": current_val - pos['total_cost'], "PnL %": pnl_pct * 100,
                    "Hold Days": (date - pos['entry_date']).days, "Outcome": "WIN",
                    "Exit": "Target", "Lots": 1 + pos['avg_downs']
                })
                tickers_to_close.append(ticker)
                continue

            # Dip Check (Averaging down using Reserve)
            drop = (p - pos['initial_price']) / pos['initial_price']

            if use_avg5 and not pos['avg5_done'] and drop <= -0.05 and cash_reserve > 0:
                amount = min(pos['initial_cost'], cash_reserve)
                if amount > 0:
                    pos['shares'] += amount / p
                    pos['total_cost'] += amount
                    cash_reserve -= amount
                    pos['avg5_done'] = True
                    pos['avg_downs'] += 1

            if use_avg10 and not pos['avg10_done'] and drop <= -0.10 and cash_reserve > 0:
                amount = min(pos['initial_cost'], cash_reserve)
                if amount > 0:
                    pos['shares'] += amount / p
                    pos['total_cost'] += amount
                    cash_reserve -= amount
                    pos['avg10_done'] = True
                    pos['avg_downs'] += 1

        for t in tickers_to_close:
            del active_positions[t]

        # B. Check New Entries (Dynamic Split Allocation)
        if cash_for_entries > 500:  # Min buffer required to bother investing
            # Find how many stocks are flashing signals today
            eligible_tickers = [t for t in ticker_pool
                                if signals_today.get(t, False)
                                and t not in active_positions
                                and not pd.isna(prices_today.get(t, np.nan))]

            if eligible_tickers:
                # Divide the ENTIRE cash pool evenly among the triggering stocks
                alloc_per_ticker = cash_for_entries / len(eligible_tickers)

                for t in eligible_tickers:
                    p = prices_today[t]
                    trade_counter += 1
                    active_positions[t] = {
                        'trade_id': trade_counter, 'entry_date': date,
                        'initial_price': p, 'initial_cost': alloc_per_ticker,
                        'total_cost': alloc_per_ticker, 'shares': alloc_per_ticker / p,
                        'avg5_done': False, 'avg10_done': False, 'avg_downs': 0
                    }
                # Empty the cash pool since it has been fully distributed
                cash_for_entries = 0.0

        # C. Record Daily Equity
        open_val = sum(pos['shares'] * prices_today[t]
                       for t, pos in active_positions.items() if not pd.isna(prices_today[t]))
        total_portfolio_value = cash_for_entries + cash_reserve + open_val
        portfolio_history.append({
            "Date": date, "Total Value": total_portfolio_value,
            "Cash Ready": cash_for_entries, "Reserve": cash_reserve, "Open Positions": open_val
        })

    # Wrap up open trades at end of simulation (period-end closeout)
    last_date = valid_dates[-1]
    for ticker, pos in active_positions.items():
        p = df_close.loc[last_date, ticker]
        current_val = pos['shares'] * p
        pnl_pct = (current_val - pos['total_cost']) / pos['total_cost']
        trades.append({
            "Trade #": pos['trade_id'], "Ticker": ticker.replace(".NS", ""),
            "Entry Date": pos['entry_date'].date(), "Exit Date": last_date.date(),
            "Entry Price": pos['initial_price'], "Exit Price": p,
            "Invested ₹": pos['total_cost'], "Exit Value ₹": current_val,
            "PnL ₹": current_val - pos['total_cost'], "PnL %": pnl_pct * 100,
            "Hold Days": (last_date - pos['entry_date']).days,
            "Outcome": "WIN" if pnl_pct >= 0 else "LOSS",
            "Exit": "Period End", "Lots": 1 + pos['avg_downs']
        })

    status_box.empty()

    # ── Display Results ──────────────────────────────────────────────
    df_trades = pd.DataFrame(trades).round(2)
    df_curve = pd.DataFrame(portfolio_history)

    # Metrics Calculations
    if not df_trades.empty:
        resolved_trades = df_trades[df_trades["Outcome"].isin(["WIN", "LOSS"])]
        avg_hold_days = resolved_trades["Hold Days"].mean(
        ) if not resolved_trades.empty else 0

        wins = len(df_trades[df_trades["Outcome"] == "WIN"])
        # All positions are now closed (target hit or period end),
        # so win rate is simply wins over total trades.
        win_rate = (wins / len(df_trades) *
                    100) if len(df_trades) > 0 else 0.0
    else:
        win_rate = 0.0
        avg_hold_days = 0.0
        resolved_trades = pd.DataFrame()

    st.markdown("### 📈 Portfolio Results")

    initial_total_cap = tranche_1 + tranche_2 + reserve_cap
    final_cap = df_curve['Total Value'].iloc[-1]
    net_profit = final_cap - initial_total_cap

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(
        f'<div class="metric-card"><div class="metric-val">₹{initial_total_cap:,.0f}</div><div class="metric-lbl">Total Seeded</div></div>', unsafe_allow_html=True)
    m2.markdown(
        f'<div class="metric-card"><div class="metric-val {"green" if final_cap > initial_total_cap else "red"}">₹{final_cap:,.0f}</div><div class="metric-lbl">Final Value</div></div>', unsafe_allow_html=True)
    m3.markdown(
        f'<div class="metric-card"><div class="metric-val {"green" if net_profit > 0 else "red"}">₹{net_profit:+,.0f}</div><div class="metric-lbl">Net Profit</div></div>', unsafe_allow_html=True)
    m4.markdown(
        f'<div class="metric-card"><div class="metric-val {"green" if win_rate >= 50 else "yellow"}">{win_rate:.1f}%</div><div class="metric-lbl">Win Rate</div></div>', unsafe_allow_html=True)
    m5.markdown(
        f'<div class="metric-card"><div class="metric-val {"green" if avg_hold_days < 30 else "yellow"}">{avg_hold_days:.1f}d</div><div class="metric-lbl">Avg Hold Time</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Project Performance Metrics (test period) ────────────────────
    total_return = final_cap / initial_total_cap - 1.0
    daily_ret = df_curve["Total Value"].pct_change().dropna()
    n_years = len(df_curve) / 252.0
    annual_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0 \
        if n_years > 0 else 0.0
    annual_vol = float(daily_ret.std() * np.sqrt(252)) \
        if len(daily_ret) > 1 else 0.0
    sharpe_ratio = (annual_return / annual_vol) if annual_vol > 0 else 0.0

    equity_vals = df_curve["Total Value"].to_numpy()
    running_peak = np.maximum.accumulate(equity_vals)
    dd_series = (equity_vals - running_peak) / running_peak
    max_drawdown = float(abs(dd_series.min())) if dd_series.min() < 0 else 0.0
    in_dd = dd_series < 0
    dd_episodes = int((in_dd & ~np.concatenate(([False], in_dd[:-1]))).sum())

    n_trades = len(df_trades)
    profit_per_trade = float(df_trades["PnL ₹"].mean()) if n_trades else 0.0
    best_trade = float(df_trades["PnL ₹"].max()) if n_trades else 0.0
    worst_trade = float(df_trades["PnL ₹"].min()) if n_trades else 0.0
    trades_per_week = n_trades / (len(df_curve) / 5.0) if len(df_curve) else 0.0
    ret_to_dd = (annual_return / max_drawdown) if max_drawdown > 0 else 0.0

    metrics_df = pd.DataFrame([
        ("Annual Return",          f"{annual_return*100:+.2f}%",
         "(final value / seed)^(252/days) − 1"),
        ("Sharpe Ratio",           f"{sharpe_ratio:.2f}",
         "annual return ÷ annual volatility (risk-free = 0)"),
        ("Max Drawdown",           f"{max_drawdown*100:.2f}%",
         "largest peak-to-trough % drop in equity"),
        ("Win Rate",               f"{win_rate:.1f}%",
         "profitable trades ÷ total trades"),
        ("Annual Volatility",      f"{annual_vol*100:.2f}%",
         "std(daily returns) × √252"),
        ("Drawdown Patterns",      str(dd_episodes),
         "count of separate peak→trough episodes"),
        ("Number of Trades",       str(n_trades),
         "positions closed in the test window"),
        ("Average Holding Period", f"{avg_hold_days:.1f} days",
         "mean(exit date − entry date)"),
        ("Profit per Trade",       f"₹{profit_per_trade:+,.0f}",
         "mean P&L ₹ across all trades"),
        ("Return / Drawdown",      f"{ret_to_dd:.2f}",
         "annual return ÷ max drawdown"),
        ("Best Trade",             f"₹{best_trade:+,.0f}",
         "max single-trade P&L"),
        ("Worst Trade",            f"₹{worst_trade:+,.0f}",
         "min single-trade P&L"),
        ("Trades per Week",        f"{trades_per_week:.1f}",
         "trades ÷ weeks in window"),
    ], columns=["Metric", "Value", "Calculation"])

    st.markdown("### 📊 Performance Metrics — Test Period")
    st.dataframe(metrics_df, use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "📥 Download Metrics CSV",
        metrics_df.to_csv(index=False),
        file_name="performance_metrics.csv",
        mime="text/csv",
    )
    price_cols = [c for c in ticker_pool if c in df_close.columns]
    price_csv = df_close.loc[valid_dates, price_cols]
    col_dl2.download_button(
        "📥 Download Asset Prices CSV",
        price_csv.to_csv(),
        file_name="portfolio_daily_closes.csv",
        mime="text/csv",
    )

    if n_trades < 20:
        st.caption(
            f"⚠️ Small-sample note: only **{n_trades} trades** in this window. "
            "Win Rate and Profit per Trade are statistically weak below ~20 trades — "
            "cross-check with a long-horizon run before drawing conclusions."
        )

    # ── Charts Row ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])

    with c1:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=df_curve["Date"], y=df_curve["Total Value"], fill="tozeroy", line=dict(
            color="#00d4aa", width=2), name="Total Value"))
        fig_eq.add_hline(y=initial_total_cap, line_dash="dash", line_color="#ffd166",
                         annotation_text="Break Even / Seed Capital", annotation_font_color="#ffd166")
        fig_eq.update_layout(
            title="True Portfolio Equity Curve (Compounded)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=400,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d", title="Value (₹)")
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    with c2:
        # Top 10 Fastest Clearing Stocks
        if not resolved_trades.empty:
            hold_stats = resolved_trades.groupby("Ticker").agg(
                Trades=("Hold Days", "count"),
                Avg_Hold=("Hold Days", "mean")
            ).reset_index().sort_values("Avg_Hold", ascending=True).head(10)

            fig_fast = px.bar(
                hold_stats, x="Avg_Hold", y="Ticker", orientation='h', text="Avg_Hold",
                title="Top 10 Fastest Clearing (Avg Hold)",
                color="Avg_Hold", color_continuous_scale="Viridis_r"
            )
            fig_fast.update_traces(
                texttemplate="%{text:.1f}d", textposition="outside")
            fig_fast.update_layout(
                yaxis={'categoryorder': 'total descending'},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                xaxis=dict(gridcolor="#21262d", title="Avg Days"), yaxis_title="",
                coloraxis_showscale=False, height=400, margin=dict(l=10, r=30, t=40, b=10)
            )
            st.plotly_chart(fig_fast, use_container_width=True)
        else:
            st.info("Not enough closed trades to show fastest clearing stocks.")

    # ── Trade Log ─────────────────────────────────────────────────────────────
    st.markdown("### 📋 Trade Log")
    if df_trades.empty or "Outcome" not in df_trades.columns:
        st.info("No trades were generated. The entry conditions may be too strict for the selected date range and ticker pool.")
    else:
        styled_trades = df_trades.style.map(
            lambda x: "color: #00d4aa" if x == "WIN" else ("color: #ffd166" if x == "OPEN" else "color: #ff6b6b"), subset=["Outcome"]
        ).map(
            lambda x: "color: #00d4aa" if isinstance(x, (int, float)) and x > 0 else ("color: #ff6b6b" if isinstance(x, (int, float)) and x < 0 else ""), subset=["PnL %", "PnL ₹"]
        ).format({"Invested ₹": "₹{:,.2f}", "Exit Value ₹": "₹{:,.2f}", "PnL ₹": "₹{:+,.2f}", "PnL %": "{:+.2f}%", "Entry Price": "₹{:,.2f}", "Exit Price": "₹{:,.2f}"})

        st.dataframe(styled_trades, use_container_width=True, height=400)

# ── Glossary / Educational Section ──────────────────────────────────────────
st.markdown("---")
with st.expander("📖 New to Trading? Understanding the Abbreviations & Strategy"):
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### The Entry Logic Explained
        The formula `110% × SMA200 > Price > SMA50 > SMA100 > SMA200` means:
        - **Price > SMA50:** The stock is performing better than its recent 2-month average.
        - **SMA50 > SMA100 > SMA200:** This is a "stacked" trend. It shows the stock is gaining strength across short, medium, and long-term horizons.
        - **110% × SMA200 > Price:** We only buy if the price is within 10% of its long-term average. This is the **"Anti-FOMO"** rule—it stops us from "chasing" a stock that has already gone up too much.
        """)

    with col_b:
        st.markdown("""
        ### Key Terms
        - **SMA (Simple Moving Average):** The average price over a set period (50, 100, or 200 days). 
        - **Tranches:** Breaking your total investment into parts (Day 1 and Day 20) to reduce the risk of bad timing.
        - **Averaging Down:** Using the **Reserve Fund** to buy more if the price drops -5% or -10% after our first entry.
        - **Compounding:** Every time we hit our **profit target**, the original money PLUS the profit is put back into the "Cash Ready" pool for the next trade.
        """)
