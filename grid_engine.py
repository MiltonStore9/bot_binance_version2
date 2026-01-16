import time
import requests
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import plotly.graph_objects as go

# =========================================================
# CONFIG
# =========================================================
@dataclass
class GridConfig:
    symbol: str = "BTCUSDT"

    # Binance Spot fee model (default): fee charged in the asset you RECEIVE
    # - BUY: fee in BASE (BTC)
    # - SELL: fee in QUOTE (USDT)
    fee_rate: float = 0.001

    quote_start: float = 1000.0       # total USDT capital
    order_quote: float = 25.0         # target USDT per grid action (per level trigger)

    grids: list = None                # ONLY real grids (no midline)
    start_price: float = None         # trading bot start price (reference)

    allow_multi_fills_per_bar: bool = True
    slippage: float = 0.0             # 0.0 = off
    show_timezone: str = "America/Lima"

    # Inventory initial mode
    init_mode: str = "balanced"       # "balanced" | "all_usdt" | "custom"
    base_start_custom: float = 0.0
    quote_start_custom: float = 1000.0


# =========================================================
# BINANCE: fetch 1m klines (UTC index)
# =========================================================
BINANCE_SPOT = "https://api.binance.com"


def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines_1m(symbol: str, start_utc: datetime, end_utc: datetime,
                    limit: int = 1000, pause_s: float = 0.08) -> pd.DataFrame:
    start_ms = _ms(start_utc)
    end_ms = _ms(end_utc)

    rows = []
    cur = start_ms

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit
        }
        r = requests.get(BINANCE_SPOT + "/api/v3/klines", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        rows.extend(data)

        last_open_time = data[-1][0]
        next_cur = last_open_time + 60_000
        if next_cur <= cur:
            break
        cur = next_cur

        if len(data) < limit:
            break

        time.sleep(pause_s)

    if not rows:
        raise RuntimeError("No se recibieron klines en ese rango. Revisa symbol/fechas.")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


# =========================================================
# Helpers: grid classification + midline support
# =========================================================
def classify_levels(grids_sorted, start_price):
    sells = [g for g in grids_sorted if g > start_price]
    buys = [g for g in grids_sorted if g < start_price]
    return sells, buys


def compute_midline(grids_sorted, start_price):
    sells, buys = classify_levels(grids_sorted, start_price)
    if not sells or not buys:
        return None, None, None
    lowest_sell = min(sells)
    highest_buy = max(buys)
    midline = (lowest_sell + highest_buy) / 2.0
    return midline, lowest_sell, highest_buy


def init_inventory(cfg: GridConfig, grids_sorted):
    if cfg.start_price is None:
        raise ValueError("cfg.start_price es obligatorio.")
    total = float(cfg.quote_start)

    if cfg.init_mode == "all_usdt":
        return total, 0.0

    if cfg.init_mode == "custom":
        return float(cfg.quote_start_custom), float(cfg.base_start_custom)

    sells, buys = classify_levels(grids_sorted, cfg.start_price)
    n_sells, n_buys = len(sells), len(buys)

    if n_sells == 0 and n_buys == 0:
        usdt_for_buys = total * 0.5
    else:
        usdt_for_buys = total * (n_buys / max(1, (n_buys + n_sells)))

    usdt_for_sells = total - usdt_for_buys
    btc_for_sells = usdt_for_sells / float(cfg.start_price)
    return usdt_for_buys, btc_for_sells


# =========================================================
# Cross detection (wicks)
# =========================================================
def crossed_down_hilo(prev_close: float, lo: float, level: float) -> bool:
    return (prev_close > level) and (lo <= level)


def crossed_up_hilo(prev_close: float, hi: float, level: float) -> bool:
    return (prev_close < level) and (hi >= level)


# =========================================================
# Execution price helper (optional slippage)
# =========================================================
def exec_price(level: float, side: str, slippage: float) -> float:
    if slippage <= 0:
        return level
    return level * (1 + slippage) if side == "BUY" else level * (1 - slippage)


# =========================================================
# SIMULATOR (Binance-like fees, wicks, midline support)
# =========================================================
def simulate_grid_hilo_binance_fees(df: pd.DataFrame, cfg: GridConfig):
    if cfg.grids is None or len(cfg.grids) < 2:
        raise ValueError("cfg.grids debe ser una lista de >=2 niveles")
    if cfg.start_price is None:
        raise ValueError("cfg.start_price es obligatorio")
    if df is None or len(df) < 3:
        raise ValueError("df muy pequeño")

    grids_real = sorted([float(x) for x in cfg.grids])
    start_price = float(cfg.start_price)

    midline, lowest_sell, highest_buy = compute_midline(grids_real, start_price)

    ladder = grids_real.copy()
    if midline is not None and all(abs(midline - g) > 1e-9 for g in ladder):
        ladder.append(midline)
    ladder = sorted(ladder)
    idx = {g: i for i, g in enumerate(ladder)}

    quote, base = init_inventory(cfg, grids_real)
    quote0, base0 = quote, base

    long_open = {}
    short_open = {}

    events = []
    pairs = []

    prev_close = float(df["close"].iloc[0])

    for k in range(1, len(df)):
        row = df.iloc[k]
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        ts = df.index[k]  # UTC timestamp

        down_crossed = [g for g in ladder if crossed_down_hilo(prev_close, lo, g)]
        up_crossed = [g for g in ladder if crossed_up_hilo(prev_close, hi, g)]

        if not cfg.allow_multi_fills_per_bar:
            if down_crossed:
                down_crossed = [min(down_crossed)]
            if up_crossed:
                up_crossed = [max(up_crossed)]

        # 1) CLOSES first

        # CLOSE LONG when price crosses UP into its target level
        for g in up_crossed:
            to_close = [buy_level for buy_level, info in long_open.items() if info["target_level"] == g]
            for buy_level in to_close:
                info = long_open[buy_level]
                qty = info["qty_base_net"]

                if base + 1e-12 < qty:
                    continue

                px = exec_price(g, "SELL", cfg.slippage)

                proceeds_quote = qty * px
                fee_quote = proceeds_quote * cfg.fee_rate
                net_quote = proceeds_quote - fee_quote

                base -= qty
                quote += net_quote

                pnl = net_quote - info["spent_quote"]

                pairs.append({
                    "type": "BUY->SELL",
                    "open_time": info["time_open"],
                    "close_time": ts,
                    "open_price": info["buy_price"],
                    "close_price": px,
                    "open_grid": info["buy_grid"],
                    "close_grid": g,
                    "qty_base": qty,
                    "spent_quote": info["spent_quote"],
                    "fee_open_base": info["fee_open_base"],
                    "fee_close_quote": fee_quote,
                    "pnl_quote": pnl
                })

                events.append({
                    "time": ts, "side": "SELL", "price": px, "grid": g,
                    "qty_base": qty,
                    "fee_quote": fee_quote,
                    "fee_base": 0.0,
                    "quote_after": quote, "base_after": base,
                    "kind": "LONG_CLOSE",
                    "pnl_pair": pnl
                })

                del long_open[buy_level]

        # CLOSE "SHORT cycle" (SELL->BUY) when price crosses DOWN into its target level
        for g in down_crossed:
            to_close = [sell_level for sell_level, info in short_open.items() if info["target_level"] == g]
            for sell_level in to_close:
                info = short_open[sell_level]
                qty = info["qty_base_sold"]

                px = exec_price(g, "BUY", cfg.slippage)

                cost_quote = qty * px
                if quote + 1e-12 < cost_quote:
                    continue

                quote -= cost_quote

                qty_gross = qty
                fee_base = qty_gross * cfg.fee_rate
                qty_net = qty_gross - fee_base
                base += qty_net

                pnl = info["net_quote_open"] - cost_quote

                pairs.append({
                    "type": "SELL->BUY",
                    "open_time": info["time_open"],
                    "close_time": ts,
                    "open_price": info["sell_price"],
                    "close_price": px,
                    "open_grid": info["sell_grid"],
                    "close_grid": g,
                    "qty_base": qty,
                    "received_quote_net_open": info["net_quote_open"],
                    "fee_open_quote": info["fee_open_quote"],
                    "fee_close_base": fee_base,
                    "pnl_quote": pnl
                })

                events.append({
                    "time": ts, "side": "BUY", "price": px, "grid": g,
                    "qty_base": qty_net,
                    "fee_quote": 0.0,
                    "fee_base": fee_base,
                    "quote_after": quote, "base_after": base,
                    "kind": "SHORT_CLOSE",
                    "pnl_pair": pnl
                })

                del short_open[sell_level]

        # 2) OPENS (only real grids)

        # OPEN LONG (BUY) on down-crossed real grids below start
        for g in down_crossed:
            if g not in grids_real:
                continue
            if g >= start_price:
                continue
            if g in long_open:
                continue
            gi = idx.get(g, None)
            if gi is None or gi >= len(ladder) - 1:
                continue
            target = ladder[gi + 1]

            px = exec_price(g, "BUY", cfg.slippage)

            spent_quote = float(cfg.order_quote)
            if quote + 1e-12 < spent_quote:
                continue

            quote -= spent_quote
            qty_gross = spent_quote / px
            fee_base = qty_gross * cfg.fee_rate
            qty_net = qty_gross - fee_base
            base += qty_net

            long_open[g] = {
                "buy_price": px,
                "buy_grid": g,
                "qty_base_net": qty_net,
                "spent_quote": spent_quote,
                "fee_open_base": fee_base,
                "time_open": ts,
                "target_level": target
            }

            events.append({
                "time": ts, "side": "BUY", "price": px, "grid": g,
                "qty_base": qty_net,
                "fee_quote": 0.0,
                "fee_base": fee_base,
                "quote_after": quote, "base_after": base,
                "kind": "LONG_OPEN"
            })

        # OPEN "SHORT cycle" (SELL) on up-crossed real grids above start
        for g in up_crossed:
            if g not in grids_real:
                continue
            if g <= start_price:
                continue
            if g in short_open:
                continue
            gi = idx.get(g, None)
            if gi is None or gi <= 0:
                continue
            target = ladder[gi - 1]

            px = exec_price(g, "SELL", cfg.slippage)

            qty = float(cfg.order_quote) / px
            if base + 1e-12 < qty:
                continue

            proceeds_quote = qty * px
            fee_quote = proceeds_quote * cfg.fee_rate
            net_quote = proceeds_quote - fee_quote

            base -= qty
            quote += net_quote

            short_open[g] = {
                "sell_price": px,
                "sell_grid": g,
                "qty_base_sold": qty,
                "fee_open_quote": fee_quote,
                "net_quote_open": net_quote,
                "time_open": ts,
                "target_level": target
            }

            events.append({
                "time": ts, "side": "SELL", "price": px, "grid": g,
                "qty_base": qty,
                "fee_quote": fee_quote,
                "fee_base": 0.0,
                "quote_after": quote, "base_after": base,
                "kind": "SHORT_OPEN"
            })

        prev_close = cl

    events_df = pd.DataFrame(events)
    pairs_df = pd.DataFrame(pairs)

    # Summary + mark-to-market
    last_price = float(df["close"].iloc[-1])
    final_value = quote + base * last_price
    initial_value = quote0 + base0 * start_price

    unrealized = 0.0

    for info in long_open.values():
        qty = info["qty_base_net"]
        proceeds = qty * last_price
        fee_q = proceeds * cfg.fee_rate
        net_q = proceeds - fee_q
        unrealized += (net_q - info["spent_quote"])

    for info in short_open.values():
        qty = info["qty_base_sold"]
        cost = qty * last_price
        unrealized += (info["net_quote_open"] - cost)

    realized = float(pairs_df["pnl_quote"].sum()) if len(pairs_df) else 0.0
    total_pnl = realized + unrealized

    summary = {
        "start_price": start_price,
        "midline": midline,
        "highest_buy": highest_buy,
        "lowest_sell": lowest_sell,
        "quote_end": quote,
        "base_end": base,
        "last_price": last_price,
        "final_value": final_value,
        "initial_value": initial_value,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": total_pnl,
        "completed_pairs": int(len(pairs_df)),
        "open_longs": int(len(long_open)),
        "open_shorts": int(len(short_open)),
        "grids_real": grids_real,
        "ladder": ladder,
        "buy_levels": [g for g in grids_real if g < start_price],
        "sell_levels": [g for g in grids_real if g > start_price],
    }

    return summary, events_df, pairs_df


# =========================================================
# PLOTLY (dark, responsive)
# =========================================================
def build_plotly_grid_trades(
    df_utc: pd.DataFrame,
    grids_real: list,
    events_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    cfg: GridConfig,
    midline=None,
    title="Grid Emulator (1m) - Binance Fees",
):
    tz_local = ZoneInfo(cfg.show_timezone)

    df = df_utc.copy()
    x = df.index.tz_convert(tz_local)
    y = df["close"].astype(float).values

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name="Price",
        line=dict(width=1.2, color="#E6E6E6"),
        hovertemplate="Time=%{x}<br>Price=%{y:,.2f}<extra></extra>",
    ))

    # Grid lines (real)
    for g in sorted([float(v) for v in grids_real]):
        fig.add_hline(
            y=g,
            line_width=1,
            line_color="red",
            opacity=0.55,
        )

    # Start line
    fig.add_hline(
        y=float(cfg.start_price),
        line_width=1.4,
        line_color="#2F6BFF",
        opacity=0.9,
    )

    # Midline
    if midline is not None:
        fig.add_hline(
            y=float(midline),
            line_width=1.1,
            line_dash="dash",
            line_color="#9A9A9A",
            opacity=0.9,
        )

    # Markers BUY/SELL
    if events_df is not None and len(events_df):
        ev = events_df.copy()
        ev["time_local"] = pd.to_datetime(ev["time"], utc=True).dt.tz_convert(tz_local)

        buys = ev[ev["side"] == "BUY"]
        sells = ev[ev["side"] == "SELL"]

        if len(buys):
            fig.add_trace(go.Scatter(
                x=buys["time_local"],
                y=buys["price"].astype(float),
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=10, color="red"),
                hovertemplate="BUY<br>Time=%{x}<br>Price=%{y:,.2f}<extra></extra>",
            ))

        if len(sells):
            fig.add_trace(go.Scatter(
                x=sells["time_local"],
                y=sells["price"].astype(float),
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=10, color="#00C853"),
                hovertemplate="SELL<br>Time=%{x}<br>Price=%{y:,.2f}<extra></extra>",
            ))

    # Dotted lines for completed pairs
    if pairs_df is not None and len(pairs_df):
        pr = pairs_df.copy()
        pr["open_time_local"] = pd.to_datetime(pr["open_time"], utc=True).dt.tz_convert(tz_local)
        pr["close_time_local"] = pd.to_datetime(pr["close_time"], utc=True).dt.tz_convert(tz_local)

        for _, r in pr.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["open_time_local"], r["close_time_local"]],
                y=[float(r["open_price"]), float(r["close_price"])],
                mode="lines",
                showlegend=False,
                line=dict(color="#9A9A9A", width=1, dash="dot"),
                opacity=0.7,
                hoverinfo="skip",
            ))

    fig.update_layout(
        template="plotly_dark",
        title=title,
        margin=dict(l=20, r=20, t=50, b=30),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        xaxis=dict(title=f"Time ({cfg.show_timezone})"),
        yaxis=dict(title="Price"),
    )

    return fig
 