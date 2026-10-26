import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import config, data, risk, rl_env, ui

ui.header("RL Trading Agent",
          "A PPO agent (Stable-Baselines3) learns when to be invested or in cash. It's trained on one period "
          "and evaluated on a later one it has never seen.")

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

if PPO is None or rl_env.TradingEnv is None:
    st.info(
        "This page needs the optional reinforcement-learning packages (about 1 GB with PyTorch):\n\n"
        "```bash\npip install -r requirements-rl.txt\n```",
        icon=":material/download:",
    )
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="rl_symbol")
period = c2.selectbox("History", ["5Y", "10Y"], index=0)
split_pct = c3.slider("Training share", 50, 85, 70, 5) / 100
c1, c2, c3 = st.columns(3)
timesteps = c1.select_slider("Training steps", [5_000, 10_000, 25_000, 50_000, 100_000], value=25_000)
cost_bps = c2.number_input("Cost per trade (bps)", 0.0, 50.0, 5.0)
seed = c3.number_input("Random seed", 0, 9999, 42)

df = data.get_history(symbol, period=period)
if len(df) < 400:
    st.error("Need more history.")
    st.stop()
split_date = df.index[int(len(df) * split_pct)]
train_df, test_df = df.loc[:split_date], df.loc[split_date - pd.Timedelta(days=120):]  # warm-up for indicators
st.caption(f"Training: {train_df.index[0]:%b %Y} – {split_date:%b %Y} · Testing: {split_date:%b %Y} – "
           f"{df.index[-1]:%b %Y}")

model_path = config.MODELS_DIR / f"ppo_{data.clean_symbol(symbol).replace('^', '')}.zip"


def evaluate(model, frame):
    env = rl_env.TradingEnv(frame, cost_bps=cost_bps)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    hist = env.history_frame()
    return hist.loc[hist.index >= split_date] if frame is test_df else hist


c1, c2 = st.columns(2)
train_clicked = c1.button("Train agent", type="primary", icon=":material/play_arrow:")
load_clicked = c2.button("Load saved agent", disabled=not model_path.exists(), icon=":material/folder_open:")

if train_clicked:
    env = rl_env.TradingEnv(train_df, cost_bps=cost_bps)
    model = PPO("MlpPolicy", env, seed=int(seed), verbose=0, n_steps=512, batch_size=64, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.01)
    bar = st.progress(0.0, "Training…")
    chunk = max(timesteps // 10, 1000)
    done_steps = 0
    while done_steps < timesteps:
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done_steps += chunk
        bar.progress(min(done_steps / timesteps, 1.0), f"Training… {min(done_steps, timesteps):,}/{timesteps:,} steps")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    st.session_state["rl_model_path"] = str(model_path)
    bar.empty()
    st.toast("Agent trained and saved", icon=":material/check:")
elif load_clicked:
    st.session_state["rl_model_path"] = str(model_path)

if "rl_model_path" not in st.session_state:
    st.info("Train an agent to see how it trades on the unseen test period.")
    st.stop()

model = PPO.load(st.session_state["rl_model_path"])
hist = evaluate(model, test_df)
prices = hist["Price"]
agent_eq = hist["Value"] / hist["Value"].iloc[0]
bh_eq = prices / prices.iloc[0]
agent_r, bh_r = agent_eq.pct_change().dropna(), bh_eq.pct_change().dropna()

m = st.columns(4)
m[0].metric("Agent return (test)", ui.pct(agent_eq.iloc[-1] - 1, sign=True), f"B&H {bh_eq.iloc[-1] - 1:+.1%}",
            delta_color="off")
m[1].metric("Agent Sharpe", f"{risk.sharpe(agent_r):.2f}", f"B&H {risk.sharpe(bh_r):.2f}", delta_color="off")
m[2].metric("Agent max drawdown", ui.pct(risk.max_drawdown(agent_eq)), f"B&H {risk.max_drawdown(bh_eq):.1%}",
            delta_color="off")
m[3].metric("Trades / time invested", f"{int((hist['Action'].isin(['BUY', 'SELL'])).sum())} / "
                                       f"{hist['Position'].mean():.0%}")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.04)
fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Price", line=dict(color="#CBD5E1")), 1, 1)
buys, sells = hist[hist["Action"] == "BUY"], hist[hist["Action"] == "SELL"]
fig.add_trace(go.Scatter(x=buys.index, y=buys["Price"], mode="markers", name="Buy",
                         marker=dict(symbol="triangle-up", size=10, color=ui.UP)), 1, 1)
fig.add_trace(go.Scatter(x=sells.index, y=sells["Price"], mode="markers", name="Sell",
                         marker=dict(symbol="triangle-down", size=10, color=ui.DOWN)), 1, 1)
fig.add_trace(go.Scatter(x=agent_eq.index, y=agent_eq * 100, name="Agent", line=dict(color=ui.ACCENT, width=2.5)), 2, 1)
fig.add_trace(go.Scatter(x=bh_eq.index, y=bh_eq * 100, name="Buy & hold", line=dict(color="#F59E0B", dash="dot")), 2, 1)
fig.update_yaxes(title_text="Growth of 100", row=2, col=1)
ui.chart(fig, 600, f"Out-of-sample test on {symbol}")

with st.expander("Trade log"):
    log = hist[hist["Action"].isin(["BUY", "SELL"])]
    st.dataframe(log, width="stretch")
    ui.to_csv_download(log, f"rl_trades_{symbol}.csv")

st.caption("RL results vary a lot between random seeds. Train with several seeds before concluding anything. "
           "Beating buy-and-hold on one test period is weak evidence.")
ui.data_status()
