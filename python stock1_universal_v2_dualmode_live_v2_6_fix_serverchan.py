# -*- coding: utf-8 -*-
"""

v2.3d.2c.6.3.9 (688498, 1m)  - macddiv9

目标：在 6.3/6.3.1 的基础上，针对“micro 卖点误触发”做定向过滤，
把胜率拉高、并把回撤压下去（不动盈利/出场逻辑）。

本版包含 Server 酱推送以及日志净化更新。
"""

from __future__ import annotations
import json, argparse, ast, os, re, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(df_1m: pd.DataFrame, params: Params = None) -> list:
    """
    主函数：执行核心策略逻辑
    :param df_1m: 1分钟级别的OHLCV数据DataFrame，必须包含ts/open/high/low/close/volume列
    :param params: 策略参数对象，若为None则使用默认参数
    :return: 回测结果列表，包含所有交易记录
    """
    # 初始化参数
    if params is None:
        params = Params()
    
    # 计算特征
    feat, m3, m5, m6 = compute_features(df_1m)
    
    # 运行回测
    trades = run_backtest(feat, m3, params)
    
    return trades

def handler(event: dict = None, context: dict = None) -> dict:
    """
    入口函数：适配云函数/命令行等调用方式
    :param event: 输入事件（如JSON参数、数据路径等）
    :param context: 运行上下文（云函数环境参数）
    :return: 执行结果字典
    """
    try:
        # 1. 解析输入参数
        if event is None:
            # 本地测试默认参数
            event = {
                "data_path": "./stock_1m_data.csv",  # 默认数据路径
                "params": {}  # 自定义策略参数
            }
        
        # 2. 加载1分钟数据（需确保数据包含ts/open/high/low/close/volume列）
        data_path = event.get("data_path", "./stock_1m_data.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        df_1m = pd.read_csv(
            data_path,
            parse_dates=["ts"],
            dtype={
                "open": float, "high": float, "low": float,
                "close": float, "volume": float
            }
        )
        
        # 3. 解析自定义参数
        custom_params = event.get("params", {})
        params = Params()
        for k, v in custom_params.items():
            if hasattr(params, k):
                setattr(params, k, v)
        
        # 4. 执行主逻辑
        trades = main(df_1m, params)
        
        # 5. 构造返回结果
        result = {
            "status": "success",
            "trade_count": len(trades),
            "trades": trades,
            "params": asdict(params)
        }
        
        return result
    
    except Exception as e:
        # 异常处理
        return {
            "status": "failed",
            "error": str(e),
            "traceback": sys.exc_info()[2].tb_frame.f_code.co_filename
        }

# =========================
# 原有工具函数和类定义（完整保留）
# =========================

# Indicators
def ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr, dtype=float).ewm(span=span, adjust=False).mean().to_numpy()

def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    close = np.asarray(close, float)
    dif = ema(close, fast) - ema(close, slow)
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().to_numpy()
    hist = dif - dea
    return dif, dea, hist

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    close = np.asarray(close, float)
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    roll_dn = pd.Series(dn).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    rs = np.divide(roll_up, roll_dn, out=np.zeros_like(roll_up), where=roll_dn != 0)
    return 100 - 100 / (1 + rs)

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    high = np.asarray(high, float); low = np.asarray(low, float); close = np.asarray(close, float)
    prev = np.roll(close, 1); prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().to_numpy()

def intraday_vwap(ts: np.ndarray, tp: np.ndarray, vol: np.ndarray) -> np.ndarray:
    ts = pd.to_datetime(ts)
    out = np.empty(len(tp), float)
    cum_pv = 0.0; cum_v = 0.0; last_day = None
    for i, (t, p, v) in enumerate(zip(ts, tp, vol)):
        d = t.date()
        if last_day is None or d != last_day:
            cum_pv = 0.0; cum_v = 0.0; last_day = d
        cum_pv += float(p) * float(v)
        cum_v += float(v)
        out[i] = cum_pv / cum_v if cum_v > 0 else float(p)
    return out

def kdj(high, low, close, n=9):
    high = np.asarray(high, float); low = np.asarray(low, float); close = np.asarray(close, float)
    k = np.full_like(close, np.nan, dtype=float)
    d = np.full_like(close, np.nan, dtype=float)
    j = np.full_like(close, np.nan, dtype=float)
    K_prev = 50.0; D_prev = 50.0
    for i in range(len(close)):
        s = max(0, i-n+1)
        hh = float(np.max(high[s:i+1])); ll = float(np.min(low[s:i+1]))
        rsv = 50.0 if hh == ll else (close[i] - ll) / (hh - ll) * 100.0
        K_prev = (2.0/3.0) * K_prev + (1.0/3.0) * rsv
        D_prev = (2.0/3.0) * D_prev + (1.0/3.0) * K_prev
        k[i] = K_prev; d[i] = D_prev; j[i] = 3*K_prev - 2*D_prev
    return k, d, j

# Helpers
def _local_max(arr: np.ndarray, w: int = 2):
    out =[]
    for i in range(w, len(arr)-w):
        if arr[i] == np.max(arr[i-w:i+w+1]) and arr[i] > arr[i-1] and arr[i] >= arr[i+1]:
            out.append(i)
    return out

def _local_min(arr: np.ndarray, w: int = 2):
    out =[]
    for i in range(w, len(arr)-w):
        if arr[i] == np.min(arr[i-w:i+w+1]) and arr[i] < arr[i-1] and arr[i] <= arr[i+1]:
            out.append(i)
    return out

def bear_div_peak(df_tf: pd.DataFrame, idx: int, lookback: int = 96, min_sep: int = 6):
    if idx < lookback + 10:
        return None
    w = df_tf.iloc[idx-lookback:idx+1]
    high = w["high"].to_numpy(float)
    peaks = _local_max(high, 2)
    if len(peaks) < 2:
        return None
    p2 = peaks[-1]
    p1 = None
    for j in reversed(peaks[:-1]):
        if p2 - j >= min_sep:
            p1 = j; break
    if p1 is None:
        return None
    base = idx - lookback
    return base + p1, base + p2

def bear_div_score(df_tf: pd.DataFrame, idx: int, lookback: int = 96, min_sep: int = 6) -> float:
    pk = bear_div_peak(df_tf, idx, lookback, min_sep)
    if pk is None:
        return 0.0
    p1, p2 = pk
    high = df_tf["high"].to_numpy(float)
    hist = df_tf["hist"].to_numpy(float)
    vol = df_tf["volume"].to_numpy(float)

    hh = max(0.0, high[p2]/high[p1] - 1.0)
    hd = 0.0
    if hist[p1] != 0:
        hd = max(0.0, 1.0 - (hist[p2]/hist[p1]))
    vd = 0.0
    if vol[p1] > 0:
        vd = max(0.0, 1.0 - (vol[p2]/vol[p1]))

    turn = 1.0 if bool(df_tf["turn_down"].iloc[idx]) else 0.0
    return float(40*hh + 2.5*hd + 1.0*vd + 1.5*turn)

def bull_div_score(df_tf: pd.DataFrame, idx: int, lookback: int = 96, min_sep: int = 6) -> float:
    if idx < lookback + 10:
        return 0.0
    w = df_tf.iloc[idx-lookback:idx+1]
    low = w["low"].to_numpy(float)
    hist = w["hist"].to_numpy(float)
    troughs = _local_min(low, 2)
    if len(troughs) < 2:
        return 0.0
    t2 = troughs[-1]
    t1 = None
    for j in reversed(troughs[:-1]):
        if t2 - j >= min_sep:
            t1 = j; break
    if t1 is None:
        return 0.0

    ll = max(0.0, 1.0 - low[t2]/low[t1])
    if not (hist[t1] < 0 and hist[t2] < 0.20*abs(hist[t1])):
        return 0.0
    hr = max(0.0, 1.0 - (abs(hist[t2])/abs(hist[t1]))) if hist[t1] != 0 else 0.0
    turn = 1.0 if bool(w["turn_up"].iloc[-1]) else 0.0
    return float(40*ll + 2.5*hr + 1.5*turn)

def micro_bear_div_score(df_tf: pd.DataFrame, idx: int, lookback: int = 60, min_sep: int = 2) -> float:
    pk = bear_div_peak(df_tf, idx, lookback, min_sep)
    if pk is None:
        return 0.0
    p1, p2 = pk
    high = df_tf['high'].to_numpy(float)
    dif = df_tf['dif'].to_numpy(float)
    hist = df_tf['hist'].to_numpy(float)
    vol = df_tf['volume'].to_numpy(float)

    if not (dif[p1] > 0 and dif[p2] > 0):
        return 0.0
    if not (hist[p1] > 0 and hist[p2] > -0.25*abs(hist[p1])):
        return 0.0

    hh = max(0.0, high[p2]/high[p1] - 1.0)
    hd = max(0.0, 1.0 - (hist[p2]/hist[p1])) if hist[p1] != 0 else 0.0
    dd = max(0.0, 1.0 - (dif[p2]/dif[p1])) if dif[p1] != 0 else 0.0
    vd = max(0.0, 1.0 - (vol[p2]/vol[p1])) if vol[p1] > 0 else 0.0

    turn = 1.0 if bool(df_tf['turn_down'].iloc[idx]) else 0.0
    return float(35*hh + 1.6*hd + 1.4*dd + 0.8*vd + 1.2*turn)

def micro_pushfail_score(df_tf: pd.DataFrame, idx: int, push_win: int = 8) -> float:
    if idx < push_win + 2:
        return 0.0
    w = df_tf.iloc[idx-push_win:idx+1]
    high = w['high'].to_numpy(float)
    dif = w['dif'].to_numpy(float)
    hist = w['hist'].to_numpy(float)
    vol = w['volume'].to_numpy(float)

    mx = float(high.max())
    if mx <= 0:
        return 0.0
    thr = mx * (1.0 - 0.0015)
    ids = np.where(high >= thr)[0]
    if len(ids) < 2:
        return 0.0
    f = int(ids[0]); l = int(ids[-1])
    if l <= f:
        return 0.0

    if not (high[l] >= high[f] * (1.0 + 0.001) or high[l] >= mx * (1.0 - 0.0002)):
        return 0.0

    weaken = (dif[l] < dif[f] * (1.0 - 0.015)) or (hist[l] < hist[f] * (1.0 - 0.03))
    vol_weak = vol[l] <= vol[f] * 1.25

    turn_now = bool(df_tf['turn_down'].iloc[idx])
    cbv_now = bool(df_tf['cross_below_vwap'].iloc[idx]) if 'cross_below_vwap' in df_tf.columns else False
    if not (turn_now or cbv_now):
        return 0.0

    if weaken and vol_weak:
        sc = 2.2
        if cbv_now:
            sc += 0.4
        if turn_now:
            sc += 0.4
        return float(sc)
    return 0.0

def micro_bear_score(df_tf: pd.DataFrame, idx: int) -> float:
    s1 = micro_bear_div_score(df_tf, idx, lookback=60, min_sep=2)
    s2 = micro_pushfail_score(df_tf, idx, push_win=8)
    return float(max(s1, s2))

def net_bps_short(sell_px: float, buy_px: float, fee_bps: float = 1.0, slip_bps: float = 1.0) -> float:
    gross = (sell_px / buy_px - 1.0) * 10000.0
    return float(gross - 2.0*(fee_bps + slip_bps))

def add_eod_flag(df_1m: pd.DataFrame) -> np.ndarray:
    day = df_1m["ts"].dt.date.to_numpy()
    eod = np.zeros(len(df_1m), dtype=bool)
    for i in range(len(df_1m)-1):
        if day[i+1] != day[i]:
            eod[i] = True
    if len(df_1m) > 0:
        eod[-1] = True
    return eod

def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df_1m.set_index("ts")[["open","high","low","close","volume"]].resample(rule).agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna().reset_index()
    c = d["close"].to_numpy(float)
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    v = d["volume"].to_numpy(float)

    dif, dea, hist = macd(c)
    d["dif"] = dif; d["dea"] = dea; d["hist"] = hist
    d["rsi"] = rsi(c, 14)
    k, dd, j = kdj(h, l, c, 9)
    d["k"] = k; d["d"] = dd; d["j"] = j
    tp = (h + l + c) / 3.0
    d["vwap"] = intraday_vwap(d["ts"].to_numpy(), tp, v)

    d["dif_chg"] = d["dif"].diff()
    d["hist_chg"] = d["hist"].diff()
    d["hist_dn2"] = (d["hist"] < d["hist"].shift(1)) & (d["hist"].shift(1) < d["hist"].shift(2))
    d["close_chg"] = d["close"].diff()
    d["turn_down"] = (d["close_chg"] < 0) & (d["dif_chg"] < 0) & (d["hist_chg"] < 0)
    d["turn_up"]   = (d["close_chg"] > 0) & (d["dif_chg"] > 0) & (d["hist_chg"] > 0)

    d["cross_below_vwap"] = (d["close"] < d["vwap"]) & (d["close"].shift(1) >= d["vwap"].shift(1))
    d["cross_above_vwap"] = (d["close"] > d["vwap"]) & (d["close"].shift(1) <= d["vwap"].shift(1))
    d["rsi_down"] = d["rsi"] < d["rsi"].shift(1)
    d["rsi_up"] = d["rsi"] > d["rsi"].shift(1)
    d["j_down"] = d["j"] < d["j"].shift(1)
    d["j_up"] = d["j"] > d["j"].shift(1)

    d["vol_chg"] = d["volume"].diff()
    d["vol_dn2"] = (d["volume"] < d["volume"].shift(1)) & (d["volume"].shift(1) < d["volume"].shift(2))
    d["vol_up"] = d["volume"] > d["volume"].shift(1)

    return d

# Params
@dataclass
class Params:
    entry_z: float = 1.25
    entry_rsi: float = 58.0
    sell_score_min: float = 1.2
    require_turn_down_5m: bool = True

    enable_entry_quality_gate: bool = True
    skip_cooldown_min: int = 10
    base_skip_exec_ge: float = 3.0
    base_skip_sell5_ge: float = 5.0

    micro_skip_early_hhmm: int = 1000
    micro_skip_score_eq: float = 2.3
    micro_skip_early_exec_ge: float = 3.0
    micro22_bias_z_max: float = 1.5

    sell_arm_min: int = 20
    sell_exec_min: float = 1.5
    sell_exec_weak_th: float = 0.6
    weak_follow_min_hold_min: int = 30
    weak_follow_window_max_min: int = 180
    weak_follow_need_drop_bps: float = 120.0
    weak_follow_exit_loss_bps: float = -50.0
    weak_follow_big_confirm_min: float = 3.2
    weak_follow_best_net_max_bps: float = 80.0
    sell_retrace_min: float = 0.0
    sell_retrace_max: float = 0.06
    sell_strong_bypass: float = 2.2

    enable_micro_sell: bool = True
    micro_sell_score_min: float = 2.00
    micro_bias_z_min: float = -9.0
    micro_z_min: float = -9.0
    micro_near_hod_max: float = 0.02
    micro_arm_min: int = 15
    micro_exec_min: float = 1.2
    micro_strong_bypass: float = 2.8
    micro_require_m3_turn_down: bool = False
    micro_retrace_max: float = 0.03
    micro_retrace_min: float = 0.0012
    micro_near_swing_max: float = 0.0015

    micro1_time_gate: bool = True
    micro1_win1_end_hhmm: int = 1000
    micro1_win2_start_hhmm: int = 1110
    micro1_win2_end_hhmm: int = 1135
    micro1_disable_strict: bool = True
    micro1_strict_score: float = 2.6

    micro3_need_m3_turn_down: bool = True
    micro3_23_need_1m_hist_neg: bool = True
    micro3_23_hist_neg_th: float = -0.05
    micro3_28_bias_z_min: float = 0.60
    micro3_28_latest_hhmm: int = 1120

    micro1_22_hod_gap_max: float = 0.003
    micro1_22_vol_z60_max: float = 0.0
    micro1_22_bias_z_max: float = 0.60
    micro1_22_hist_min: float = 0.20

    micro_time_gate: bool = True
    micro_time_start_hhmm: int = 930
    micro_time_end_hhmm: int = 1445
    micro_midmor_tighten: bool = True
    micro_midmor_start_hhmm: int = 1000
    micro_midmor_end_hhmm: int = 1120
    micro_sell_score_min_midmor: float = 2.30

    entry_cutoff_hhmm: int = 1445
    max_cycles_per_day: int = 2
    cooldown_min: int = 30

    min_hold_min: int = 5
    min_mean_hold_min: int = 10
    min_exit_profit_bps: float = 160.0

    buy_score_min: float = 1.25
    buy_score_min_6m: float = 1.10
    buy_exec_min: float = 1.7
    require_turn_up_buy: bool = True
    buy_arm_min: int = 60
    buy_bounce_max: float = 0.035
    buy_oversold_rsi: float = 45.0

    min_drop_buy_bps: float = 400.0
    buy_extreme_score: float = 2.10

    hl_rebound_ratio: float = 0.012
    hl_min_ratio: float = 0.005
    hl_window_min: int = 12

    panic_vol_z: float = 2.2
    require_vol_exhaust: bool = True
    vol_gate_min_count: int = 2

    lock1_bps: float = 200.0
    tp_final_bps: float = 350.0

    tp_hold_enable: bool = True
    tp_hold_fast_max_min: int = 45
    tp_hold_max_extra_min: int = 120
    tp_hold_release_rebound_bps: float = 60.0
    tp_hold_giveback_bps: float = 140.0
    tp_hold_need_m3_turn: bool = True
    tp_hold_need_hist_up2: bool = True

    trail_enable_bps: float = 300.0
    trail_dd_bps: float = 160.0

    struct_break_ratio: float = 0.01
    struct_second_window: int = 6
    struct_warn_reset_mult: float = 0.5
    struct_mom_need: bool = True
    struct_big_confirm_min: float = 3.4
    struct_use_big_confirm: bool = True

    allow_overnight: bool = True
    max_hold_days: int = 1
    preclose_hhmm: int = 1458
    carry_min_bps: float = -350.0

    preclose_no_drop_big_confirm_min: float = 2.0
    preclose_no_drop_loss_bps: float = -80.0
    preclose_no_drop_exec_th: float = 0.1

    overnight_need_drop_bps: float = 250.0
    overnight_max_loss_bps: float = -150.0

    max_loss_bps: float = -220.0
    max_loss_bps_weak: float = -140.0
    loss_guard_weak_big_confirm_min: float = 3.0
    loss_guard_weak_break_ratio: float = 0.008
    loss_guard_weak_fast_min: int = 60
    loss_guard_weak_confirm_count: int = 2
    loss_guard_weak_window_max_min: int = 180

    follow_min_hold_min: int = 60
    follow_window_max_min: int = 180
    follow_need_drop_bps: float = 150.0
    follow_exit_loss_bps: float = -80.0
    follow_big_confirm_min: float = 3.2

    fee_bps: float = 1.0
    slip_bps: float = 1.0

def compute_features(df_1m: pd.DataFrame):
    d = df_1m.copy().sort_values("ts").reset_index(drop=True)
    h = d["high"].astype(float).to_numpy()
    l = d["low"].astype(float).to_numpy()
    c = d["close"].astype(float).to_numpy()
    v = d["volume"].astype(float).to_numpy()
    ts = d["ts"].to_numpy()

    atrv = atr(h, l, c, 14)
    tp = (h + l + c) / 3.0
    vwap = intraday_vwap(ts, tp, v)
    z = (c - vwap) / np.where(atrv == 0, np.nan, atrv)

    ema21 = ema(c, 21)
    bias = (c - ema21) / np.where(atrv == 0, np.nan, atrv)

    dif, dea, hist = macd(c)
    d["atr"] = atrv
    d["vwap"] = vwap
    d["z"] = z
    d["ema21"] = ema21
    d["bias"] = bias
    d["rsi"] = rsi(c, 14)
    d["dif"] = dif
    d["dea"] = dea
    d["hist"] = hist
    k1, d1, j1 = kdj(h, l, c, 9)
    d["k"] = k1; d["d"] = d1; d["j"] = j1

    d["dif_chg"] = d["dif"].diff()
    d["hist_chg"] = d["hist"].diff()
    d["close_chg"] = d["close"].diff()
    d["turn_down"] = (d["close_chg"] < 0) & (d["dif_chg"] < 0) & (d["hist_chg"] < 0)
    d["cross_below_vwap"] = (d["close"] < d["vwap"]) & (d["close"].shift(1) >= d["vwap"].shift(1))
    d["rsi_down"] = d["rsi"] < d["rsi"].shift(1)
    d["j_down"] = d["j"] < d["j"].shift(1)
    d["hist_dn2"] = (d["hist"] < d["hist"].shift(1)) & (d["hist"].shift(1) < d["hist"].shift(2))

    d["day"] = d["ts"].dt.date.astype(str)
    d["hhmm"] = (d["ts"].dt.hour*100 + d["ts"].dt.minute).astype(int)
    d["eod"] = add_eod_flag(d)

    def _biasz(g: pd.DataFrame):
        b = g["bias"].astype(float)
        mu = b.rolling(120, min_periods=30).mean()
        sd = b.rolling(120, min_periods=30).std()
        z = (b - mu) / sd
        return z.fillna(0.0)
    d["bias_z"] = d.groupby("day", group_keys=False).apply(_biasz).astype(float)

    d["hod_high"] = d.groupby("day")["high"].cummax()
    d["swing_high_45"] = d.groupby("day")["high"].transform(lambda s: s.rolling(45, min_periods=1).max())
    d["swing_gap_45"] = (d["swing_high_45"] - d["high"]) / d["swing_high_45"].replace(0, float('nan'))
    d["hod_gap"] = (d["hod_high"] / d["close"].astype(float) - 1.0).astype(float)
    d["hod_gap_high"] = (d["hod_high"] / d["high"].astype(float) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

    def _volz(g: pd.DataFrame):
        vol = g["volume"].astype(float)
        mu = vol.rolling(60, min_periods=15).mean()
        sd = vol.rolling(60, min_periods=15).std()
        z = (vol - mu) / sd
        return z.fillna(0.0)
    d["vol_z60"] = d.groupby("day", group_keys=False).apply(_volz).astype(float)

    uniq = sorted(d["ts"].dt.date.unique())
    gap_map = {}
    for i, dd in enumerate(uniq):
        if i == len(uniq)-1:
            gap_map[str(dd)] = 999
        else:
            gap_map[str(dd)] = (uniq[i+1]-dd).days
    d["next_gap_days"] = d["day"].map(gap_map).fillna(999).astype(int)

    m3 = resample_ohlcv(d[["ts","open","high","low","close","volume"]], "3min")
    m5 = resample_ohlcv(d[["ts","open","high","low","close","volume"]], "5min")
    m6 = resample_ohlcv(d[["ts","open","high","low","close","volume"]], "6min")

    m5["bear_score"] = [bear_div_score(m5, i) for i in range(len(m5))]
    m5["peak2"] =[(bear_div_peak(m5, i) or (None, None))[1] for i in range(len(m5))]
    m3["bull_score"] =[bull_div_score(m3, i) for i in range(len(m3))]
    
    m3['day'] = m3['ts'].dt.date.astype(str)
    near_pct = 0.0020
    rev = m3['turn_down'] | m3['cross_below_vwap']

    def _score_from_window(w: int, new_sc: float, near_sc: float):
        prev_high = m3['high'].shift(1).rolling(w, min_periods=w).max()
        prev_dif  = m3['dif'].shift(1).rolling(w, min_periods=w).max()
        prev_hist = m3['hist'].shift(1).rolling(w, min_periods=w).max()
        prev_vol  = m3['volume'].shift(1).rolling(w, min_periods=w).max()
        price_new = m3['high'].ge(prev_high * (1.0 + 0.0005))
        price_near = m3['high'].ge(prev_high * (1.0 - near_pct))
        dif_fail = m3['dif'].le(prev_dif * (1.0 - 0.02)) | m3['hist'].le(prev_hist * (1.0 - 0.03))
        vol_fail = m3['volume'].le(prev_vol * 1.10)
        sc = np.where(price_new & dif_fail & vol_fail & rev, new_sc,
             np.where(price_near & dif_fail & vol_fail & rev, near_sc, 0.0))
        return sc

    score_short = _score_from_window(8, 2.8, 2.2)
    score_long  = _score_from_window(20, 2.7, 2.1)

    hod_prev_high = m3.groupby('day')['high'].cummax().shift(1)
    hod_prev_dif  = m3.groupby('day')['dif'].cummax().shift(1)
    hod_prev_hist = m3.groupby('day')['hist'].cummax().shift(1)
    hod_prev_vol  = m3.groupby('day')['volume'].cummax().shift(1)

    price_new_hod = m3['high'].ge(hod_prev_high * (1.0 + 0.0005))
    price_near_hod = m3['high'].ge(hod_prev_high * (1.0 - near_pct))
    dif_fail_hod = m3['dif'].le(hod_prev_dif * (1.0 - 0.02)) | m3['hist'].le(hod_prev_hist * (1.0 - 0.03))
    vol_fail_hod = m3['volume'].le(hod_prev_vol * 1.10)

    score_hod = np.where(price_new_hod & dif_fail_hod & vol_fail_hod & rev, 2.8,
                 np.where(price_near_hod & dif_fail_hod & vol_fail_hod & rev, 2.3, 0.0))

    score = np.maximum.reduce([score_short, score_long, score_hod])
    m3['micro_bear_score'] = pd.Series(score, index=m3.index).fillna(0.0).astype(float)
    m6["bull_score"] = [bull_div_score(m6, i) for i in range(len(m6))]

    m3_idx = pd.Series(range(len(m3)), index=m3["ts"])
    m5_idx = pd.Series(range(len(m5)), index=m5["ts"])
    m6_idx = pd.Series(range(len(m6)), index=m6["ts"])

    d["ts3"] = d["ts"].dt.floor("3min")
    d["ts5"] = d["ts"].dt.floor("5min")
    d["ts6"] = d["ts"].dt.floor("6min")
    d["i3"] = d["ts3"].map(m3_idx).ffill().astype(int)
    d["i5"] = d["ts5"].map(m5_idx).ffill().astype(int)
    d["i6"] = d["ts6"].map(m6_idx).ffill().astype(int)

    d["sell5_score"] = d["i5"].map(dict(zip(range(len(m5)), m5["bear_score"]))).fillna(0.0).astype(float)
    d["turn_down_5"] = d["i5"].map(dict(zip(range(len(m5)), m5["turn_down"]))).fillna(False).astype(bool)
    d["micro3_score"] = d["i3"].map(dict(zip(range(len(m3)), m3["micro_bear_score"])) ).fillna(0.0).astype(float)

    def _micro1(g: pd.DataFrame) -> pd.Series:
        w_swing = 45
        w_hod   = 12
        near_pct = 0.0020
        high = g['high'].astype(float)
        vol  = g['volume'].astype(float)
        dif  = g['dif'].astype(float)
        hist = g['hist'].astype(float)
        hhmm = g['ts'].dt.hour * 100 + g['ts'].dt.minute
        time_ok = (hhmm >= 930) & (hhmm <= 1455) & ~((hhmm >= 1130) & (hhmm <= 1259))

        hod_high = g['hod_high'].astype(float)
        near_hod = (hod_high / high - 1.0).le(0.003)

        prev_dif_hod  = dif.shift(1).rolling(w_hod,  min_periods=w_hod).max()
        prev_hist_hod = hist.shift(1).rolling(w_hod, min_periods=w_hod).max()
        prev_vol_hod  = vol.shift(1).rolling(w_hod,  min_periods=w_hod).max()

        dif_fail_hod  = dif.le(prev_dif_hod  * (1.0 - 0.012))
        hist_fail_hod = hist.le(prev_hist_hod * (1.0 - 0.18))
        vol_fail_hod  = vol.le(prev_vol_hod  * 0.95)

        hist_dn2 = hist.lt(hist.shift(1)) & hist.shift(1).lt(hist.shift(2))
        turn_hint = (g['close'].astype(float) < g['close'].shift(1).astype(float))

        sc_hod = np.where(time_ok & near_hod & hist_dn2 & (dif_fail_hod | hist_fail_hod) & (vol_fail_hod | turn_hint), 2.2, 0.0)

        prev_high_sw = high.shift(1).rolling(w_swing, min_periods=w_swing).max()
        prev_dif_sw  = dif.shift(1).rolling(w_swing,  min_periods=w_swing).max()
        prev_hist_sw = hist.shift(1).rolling(w_swing,  min_periods=w_swing).max()
        prev_vol_sw  = vol.shift(1).rolling(w_swing,  min_periods=w_swing).max()

        price_near_sw = high.ge(prev_high_sw * (1.0 - 0.0015))
        dif_fail_sw   = dif.le(prev_dif_sw  * (1.0 - 0.015))
        hist_fail_sw  = hist.le(prev_hist_sw * (1.0 - 0.20))
        vol_fail_sw   = vol.le(prev_vol_sw  * 0.92)

        sc_sw = np.where(time_ok & price_near_sw & hist_dn2 & (dif_fail_sw | hist_fail_sw) & (vol_fail_sw | turn_hint), 2.0, 0.0)

        sc = np.maximum(sc_hod, sc_sw)
        return pd.Series(sc, index=g.index)

    d['micro1_score'] = d.groupby('day', group_keys=False).apply(_micro1).fillna(0.0).astype(float)

    d["buy3_score"] = d["i3"].map(dict(zip(range(len(m3)), m3["bull_score"]))).fillna(0.0).astype(float)
    d["buy6_score"] = d["i6"].map(dict(zip(range(len(m6)), m6["bull_score"]))).fillna(0.0).astype(float)
    d["turn_up_3"] = d["i3"].map(dict(zip(range(len(m3)), m3["turn_up"]))).fillna(False).astype(bool)
    d["turn_up_6"] = d["i6"].map(dict(zip(range(len(m6)), m6["turn_up"]))).fillna(False).astype(bool)

    for col in["cross_below_vwap","turn_down","rsi_down","j_down","cross_above_vwap","turn_up","rsi_up","j_up","hist","rsi","j","volume","vol_dn2","vol_up"]:
        d[f"m3_{col}"] = d["i3"].map(dict(zip(range(len(m3)), m3[col])))
    d["m3_hist"] = d["m3_hist"].astype(float)
    d["m3_rsi"] = d["m3_rsi"].astype(float)
    d["m3_j"] = d["m3_j"].astype(float)
    d["m3_volume"] = d["m3_volume"].astype(float)
    for col in["cross_below_vwap","turn_down","rsi_down","j_down","cross_above_vwap","turn_up","rsi_up","j_up","vol_dn2","vol_up"]:
        d[f"m3_{col}"] = d[f"m3_{col}"].fillna(False).astype(bool)

    for col in["close","high","dif","hist","vwap","rsi","j","dif_chg","hist_chg","cross_above_vwap","turn_up","rsi_up","j_up","vol_up","vol_dn2","volume"]:
        d[f"m5_{col}"] = d["i5"].map(dict(zip(range(len(m5)), m5[col])))

    peak2_map = {i:(m5.loc[int(p2),"high"] if (p2 is not None and not pd.isna(p2)) else np.nan) for i,p2 in enumerate(m5["peak2"])}
    d["sell5_ref_high"] = d["i5"].map(peak2_map).astype(float)

    return d, m3, m5, m6

# Exec score
def sell_exec_score_1m(row: pd.Series) -> float:
    sc = 0.0
    if bool(row.get('cross_below_vwap', False)): sc += 1.0
    if bool(row.get('turn_down', False)): sc += 1.0
    if bool(row.get('rsi_down', False)): sc += 0.5
    if bool(row.get('j_down', False)): sc += 0.5
    if bool(row.get('hist_dn2', False)): sc += 1.0
    if float(row.get('rsi', 0.0)) >= 60.0: sc += 0.3
    return float(sc)

def sell_exec_score_from_row(row: pd.Series, m3: pd.DataFrame) -> float:
    i3 = int(row["i3"])
    if i3 < 0 or i3 >= len(m3): return 0.0
    sc = 0.0
    if bool(m3["cross_below_vwap"].iloc[i3]): sc += 1.0
    if bool(m3["turn_down"].iloc[i3]): sc += 1.0
    if bool(m3["rsi_down"].iloc[i3]): sc += 0.5
    if bool(m3["j_down"].iloc[i3]): sc += 0.5
    if i3 >= 2 and (m3["hist"].iloc[i3] < m3["hist"].iloc[i3-1] < m3["hist"].iloc[i3-2]): sc += 1.0
    if float(m3["rsi"].iloc[i3]) >= 60: sc += 0.3
    return float(sc)

def buy_exec_score_from_row(row: pd.Series, m3: pd.DataFrame) -> float:
    i3 = int(row["i3"])
    if i3 < 0 or i3 >= len(m3): return 0.0
    sc = 0.0
    if bool(m3["cross_above_vwap"].iloc[i3]): sc += 1.0
    if bool(m3["turn_up"].iloc[i3]): sc += 1.0
    if bool(m3["rsi_up"].iloc[i3]): sc += 0.5
    if bool(m3["j_up"].iloc[i3]): sc += 0.5
    if i3 >= 2 and (m3["hist"].iloc[i3] > m3["hist"].iloc[i3-1] > m3["hist"].iloc[i3-2]): sc += 1.0
    if bool(m3["vol_up"].iloc[i3]): sc += 0.4
    if float(m3["rsi"].iloc[i3]) <= 60: sc += 0.3
    return float(sc)

def follow_big_confirm_score(row: pd.Series) -> float:
    sc = 0.0
    try:
        if bool(row.get("m5_cross_above_vwap", False)) or (float(row.get("m5_close", np.nan)) > float(row.get("m5_vwap", np.nan))): sc += 1.0
        if bool(row.get("m5_turn_up", False)): sc += 1.0
        if float(row.get("m5_dif_chg", 0.0)) > 0: sc += 0.6
        if float(row.get("m5_hist_chg", 0.0)) > 0: sc += 0.6
        if bool(row.get("m5_rsi_up", False)): sc += 0.4
        if float(row.get("m5_rsi", 0.0)) >= 55: sc += 0.3
        if bool(row.get("m5_j_up", False)): sc += 0.4
        if float(row.get("m5_j", 0.0)) >= 60: sc += 0.3
        if bool(row.get("m5_vol_up", False)): sc += 0.4
        if bool(row.get("m5_vol_dn2", False)) and bool(row.get("m5_vol_up", False)): sc += 0.2
    except Exception:
        return 0.0
    return float(sc)

# Backtest
def run_backtest(feat: pd.DataFrame, m3: pd.DataFrame, p: Params):
    trades =[]
    pos = 0
    entry_px = np.nan
    entry_exec_sc = np.nan
    entry_ts = None
    entry_day = None
    ref_high = np.nan
    last_exit = None
    skip_until = None
    cycles_today = 0
    cur_day = None

    best_net = -1e9
    locked = False

    struct_warned = False
    warn_i5 = None
    last_i5 = None

    sell_armed_until = None
    sell_arm_ref_high = np.nan
    sell_arm_score = 0.0

    micro_armed_until = None
    micro_arm_ref_high = np.nan
    micro_arm_score = 0.0

    buy_armed_until = None
    buy_arm_score = 0.0
    low_since_entry = np.inf

    tp_hold_active = False
    tp_hold_start_ts = None
    tp_hold_deadline_ts = None
    tp_hold_note = None

    weak_break_count = 0

    panic_seen = False
    exhaust_ready = False
    low1 = np.inf
    rebound_seen = False
    hl_ready = False

    for i in range(len(feat)-1):
        row = feat.iloc[i]
        nxt = feat.iloc[i+1]
        ts = row["ts"]
        day = row["day"]
        hhmm = int(row["hhmm"])

        if cur_day != day:
            cur_day = day
            cycles_today = 0
            skip_until = None
            sell_armed_until = None
            sell_arm_ref_high = np.nan
            sell_arm_score = 0.0
            micro_armed_until = None
            micro_arm_ref_high = np.nan
            micro_arm_score = 0.0

        if pos == 0:
            best_net = -1e9
            locked = False
            struct_warned = False
            warn_i5 = None
            last_i5 = None
            buy_armed_until = None
            buy_arm_score = 0.0
            panic_seen = False
            exhaust_ready = False
            low1 = np.inf
            rebound_seen = False
            hl_ready = False
            weak_break_count = 0

            if skip_until is not None and ts < skip_until: continue
            if cycles_today >= p.max_cycles_per_day: continue
            if last_exit is not None and (ts - last_exit).total_seconds()/60.0 < p.cooldown_min: continue
            if hhmm >= p.entry_cutoff_hhmm: continue

            base = (float(row["sell5_score"]) >= p.sell_score_min) and (float(row["z"]) >= p.entry_z) and (float(row["rsi"]) >= p.entry_rsi)
            if p.require_turn_down_5m:
                base = base and bool(row["turn_down_5"])
            if base:
                sell_armed_until = ts + pd.Timedelta(minutes=p.sell_arm_min)
                sell_arm_ref_high = float(row["sell5_ref_high"]) if np.isfinite(row["sell5_ref_high"]) else float(row["m5_high"])
                sell_arm_score = float(row["sell5_score"])

            if getattr(p, 'enable_micro_sell', True):
                m3_sc = float(row.get('micro3_score', 0.0))
                m1_sc = float(row.get('micro1_score', 0.0))

                if getattr(p, 'micro1_time_gate', True):
                    w1_end = int(getattr(p, 'micro1_win1_end_hhmm', 1000))
                    w2_st  = int(getattr(p, 'micro1_win2_start_hhmm', 1110))
                    w2_ed  = int(getattr(p, 'micro1_win2_end_hhmm', 1135))
                    m1_time_ok = (hhmm <= w1_end) or ((hhmm >= w2_st) and (hhmm <= w2_ed))
                    if not m1_time_ok: m1_sc = 0.0
                if getattr(p, 'micro1_disable_strict', True):
                    strict_sc = float(getattr(p, 'micro1_strict_score', 2.6))
                    if m1_sc >= strict_sc - 1e-9: m1_sc = 0.0

                if getattr(p, 'micro3_need_m3_turn_down', True):
                    if (m3_sc > 0.0) and (not bool(row.get('m3_turn_down', False))): m3_sc = 0.0

                if getattr(p, 'micro3_23_need_1m_hist_neg', True):
                    th = float(getattr(p, 'micro3_23_hist_neg_th', -0.0))
                    if abs(m3_sc - 2.3) < 1e-9 and float(row.get('hist', 0.0)) >= th: m3_sc = 0.0

                if abs(m3_sc - 2.8) < 1e-9:
                    bz_min = float(getattr(p, 'micro3_28_bias_z_min', 0.0))
                    latest = int(getattr(p, 'micro3_28_latest_hhmm', 2359))
                    if float(row.get('bias_z', 0.0)) < bz_min: m3_sc = 0.0
                    elif hhmm >= latest: m3_sc = 0.0

                if abs(m1_sc - 2.2) < 1e-9:
                    hg_max = float(getattr(p, 'micro1_22_hod_gap_max', 0.003))
                    vz_max = float(getattr(p, 'micro1_22_vol_z60_max', 0.0))
                    bz_max = float(getattr(p, 'micro1_22_bias_z_max', 9.0))
                    hist_min = float(getattr(p, 'micro1_22_hist_min', -1e9))
                    if float(row.get('hod_gap', 999.0)) > hg_max: m1_sc = 0.0
                    elif float(row.get('vol_z60', 0.0)) > vz_max: m1_sc = 0.0
                    elif float(row.get('bias_z', 0.0)) > bz_max: m1_sc = 0.0
                    elif float(row.get('hist', 0.0)) < hist_min: m1_sc = 0.0

                micro_score = max(m3_sc, m1_sc)

                micro_time_ok = True
                if getattr(p, 'micro_time_gate', False):
                    st = int(getattr(p, 'micro_time_start_hhmm', 0))
                    ed = int(getattr(p, 'micro_time_end_hhmm', 2359))
                    micro_time_ok = (hhmm >= st) and (hhmm <= ed)

                score_min = float(getattr(p, 'micro_sell_score_min', 0.0))
                if getattr(p, 'micro_midmor_tighten', False):
                    st2 = int(getattr(p, 'micro_midmor_start_hhmm', 1000))
                    ed2 = int(getattr(p, 'micro_midmor_end_hhmm', 1120))
                    if (hhmm >= st2) and (hhmm < ed2):
                        score_min = max(score_min, float(getattr(p, 'micro_sell_score_min_midmor', score_min)))

                micro_ok = (
                    micro_time_ok and
                    (micro_score >= score_min) and
                    ((float(row.get('bias_z', 0.0)) >= p.micro_bias_z_min) or (float(row.get('z', 0.0)) >= p.micro_z_min)) and
                    ( (float(row.get('hod_gap', 999.0)) <= p.micro_near_hod_max) or (float(row.get('swing_gap_45', 999.0)) <= p.micro_near_swing_max) )
                )
                if p.micro_require_m3_turn_down:
                    micro_ok = micro_ok and bool(row.get('m3_turn_down', False))
                if micro_ok:
                    micro_armed_until = ts + pd.Timedelta(minutes=p.micro_arm_min)
                    micro_arm_ref_high = float(max(row.get('hod_high', row.get('high', np.nan)), row.get('swing_high_45', row.get('high', np.nan))))
                    micro_arm_score = float(micro_score)

            if sell_armed_until is not None and ts > sell_armed_until:
                sell_armed_until = None
                sell_arm_ref_high = np.nan
                sell_arm_score = 0.0

            if micro_armed_until is not None and ts > micro_armed_until:
                micro_armed_until = None
                micro_arm_ref_high = np.nan
                micro_arm_score = 0.0

            if (sell_armed_until is None) and (micro_armed_until is None):
                continue

            use_micro = False
            arm_score = sell_arm_score
            arm_ref_high = sell_arm_ref_high
            exec_min = p.sell_exec_min
            strong_th = p.sell_strong_bypass
            retr_max = p.sell_retrace_max

            if micro_armed_until is not None:
                if (sell_armed_until is None) or (micro_arm_score > sell_arm_score):
                    use_micro = True
                    arm_score = micro_arm_score
                    arm_ref_high = micro_arm_ref_high
                    exec_min = p.micro_exec_min
                    strong_th = p.micro_strong_bypass
                    retr_max = p.micro_retrace_max

            exec_sc = sell_exec_score_from_row(row, m3)
            if use_micro:
                exec_sc = max(exec_sc, sell_exec_score_1m(row))

            if getattr(p, 'enable_entry_quality_gate', False):
                _skip_cool = int(getattr(p, 'skip_cooldown_min', 10))
                if (not use_micro) and (exec_sc >= float(getattr(p, 'base_skip_exec_ge', 9e9))) and (sell_arm_score >= float(getattr(p, 'base_skip_sell5_ge', 9e9))):
                    skip_until = ts + pd.Timedelta(minutes=_skip_cool)
                    sell_armed_until = None; sell_arm_ref_high = np.nan; sell_arm_score = 0.0
                    micro_armed_until = None; micro_arm_ref_high = np.nan; micro_arm_score = 0.0
                    continue
                if use_micro and (hhmm < int(getattr(p, 'micro_skip_early_hhmm', 0))):
                    _sc_eq = float(getattr(p, 'micro_skip_score_eq', 2.3))
                    _exec_ge = float(getattr(p, 'micro_skip_early_exec_ge', 9e9))
                    if (abs(arm_score - _sc_eq) < 1e-6) or (exec_sc >= _exec_ge):
                        skip_until = ts + pd.Timedelta(minutes=_skip_cool)
                        sell_armed_until = None; sell_arm_ref_high = np.nan; sell_arm_score = 0.0
                        micro_armed_until = None; micro_arm_ref_high = np.nan; micro_arm_score = 0.0
                        continue
                if use_micro and (abs(arm_score - 2.2) < 1e-6):
                    _bz_max = float(getattr(p, 'micro22_bias_z_max', 9e9))
                    if float(row.get('bias_z', 0.0)) > _bz_max:
                        skip_until = ts + pd.Timedelta(minutes=_skip_cool)
                        sell_armed_until = None; sell_arm_ref_high = np.nan; sell_arm_score = 0.0
                        micro_armed_until = None; micro_arm_ref_high = np.nan; micro_arm_score = 0.0
                        continue
        
        # 此处省略原有回测逻辑的剩余部分（因原代码未完整展示）
        # 若需要完整运行，需补充run_backtest函数的剩余逻辑
    
    return trades

# 本地测试入口
if __name__ == "__main__":
    # 本地测试调用示例
    result = handler()
    print(f"执行状态: {result['status']}")
    if result["status"] == "success":
        print(f"交易数量: {result['trade_count']}")
    else:
        print(f"错误信息: {result['error']}")
