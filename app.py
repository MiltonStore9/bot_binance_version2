from flask import Flask, render_template
import json
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from plotly.utils import PlotlyJSONEncoder

from grid_engine import (
    GridConfig,
    fetch_klines_1m,
    simulate_grid_hilo_binance_fees,
    build_plotly_grid_trades,   # <- NUEVO (Plotly)
)

app = Flask(__name__)

REFRESH_SECONDS = 15


def fmt(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)


@app.route("/")
def index():
    tz_pe = ZoneInfo("America/Lima")

    # ---- rango ----
    start_local = datetime(2026, 1, 11, 23, 54, tzinfo=tz_pe)
    end_local = datetime.now(tz_pe)

    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    end_utc = end_local.astimezone(ZoneInfo("UTC"))

    # ---- TU CONFIG ----
    my_grids = [91508, 92908]  # ONLY real grids
    trading_bot_start_price = 92188.95
    capital_invertido = 106.95

    cfg = GridConfig(
        symbol="ETHUSDT",
        fee_rate=0.001,
        quote_start=capital_invertido,
        order_quote=capital_invertido / len(my_grids),  # per action
        grids=my_grids,
        start_price=trading_bot_start_price,
        allow_multi_fills_per_bar=True,
        slippage=0.0,
        show_timezone="America/Lima",
        init_mode="balanced",
    )

    # ---- RUN ----
    df = fetch_klines_1m(cfg.symbol, start_utc, end_utc)
    summary, events_df, pairs_df = simulate_grid_hilo_binance_fees(df, cfg)

    fig = build_plotly_grid_trades(
        df_utc=df,
        grids_real=summary["grids_real"],
        events_df=events_df,
        pairs_df=pairs_df,
        cfg=cfg,
        midline=summary["midline"],
        title=f"{cfg.symbol} Grid Emulator (1m) - Lima (HIGH/LOW wicks) - Binance Fees",
    )

    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    # ---- tablas (últimas filas) ----
    tz_local = ZoneInfo(cfg.show_timezone)

    ev_rows = []
    if len(events_df):
        ev = events_df.copy()
        ev["time_local"] = (
            pd.to_datetime(ev["time"], utc=True)
            .dt.tz_convert(tz_local)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        cols = ["time_local", "side", "kind", "price", "grid", "qty_base", "fee_quote", "fee_base", "quote_after", "base_after"]
        if "pnl_pair" in ev.columns:
            cols.insert(6, "pnl_pair")
        ev = ev[cols].tail(80)
        for _, r in ev.iterrows():
            ev_rows.append({c: r.get(c, "") for c in cols})

    pr_rows = []
    if len(pairs_df):
        pr = pairs_df.copy()
        pr["open_time_local"] = (
            pd.to_datetime(pr["open_time"], utc=True)
            .dt.tz_convert(tz_local)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        pr["close_time_local"] = (
            pd.to_datetime(pr["close_time"], utc=True)
            .dt.tz_convert(tz_local)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        cols = ["type", "open_time_local", "close_time_local", "open_price", "close_price", "open_grid", "close_grid", "qty_base", "pnl_quote"]
        pr = pr[cols].tail(80)
        for _, r in pr.iterrows():
            pr_rows.append({c: r.get(c, "") for c in cols})

    # ---- summary fmt ----
    summary_fmt = dict(summary)
    summary_fmt["initial_value_s"] = fmt(summary.get("initial_value"), 2)
    summary_fmt["final_value_s"] = fmt(summary.get("final_value"), 2)
    summary_fmt["realized_pnl_s"] = fmt(summary.get("realized_pnl"), 4)
    summary_fmt["unrealized_pnl_s"] = fmt(summary.get("unrealized_pnl"), 4)
    summary_fmt["total_pnl_s"] = fmt(summary.get("total_pnl"), 4)
    summary_fmt["base_end_s"] = fmt(summary.get("base_end"), 8)
    summary_fmt["quote_end_s"] = fmt(summary.get("quote_end"), 2)
    summary_fmt["last_price_s"] = fmt(summary.get("last_price"), 2)

    return render_template(
        "index.html",
        refresh=REFRESH_SECONDS,
        now=end_local.strftime("%Y-%m-%d %H:%M:%S"),
        fig_json=fig_json,
        summary=summary_fmt,
        cfg=cfg,
        events=ev_rows,
        pairs=pr_rows,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
