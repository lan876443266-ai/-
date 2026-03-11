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

# =========================
# Indicators
# =========================

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

# =========================
# Helpers
# =========================

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

# =========================
# Params
# =========================

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
        prev_hist_sw = hist.shift(1).rolling(w_swing, min_periods=w_swing).max()
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

# =========================
# Exec score
# =========================

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

# =========================
# Backtest
# =========================

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

            strong_bypass = arm_score >= strong_th
            if (exec_sc < exec_min) and (not strong_bypass):
                continue

            if np.isfinite(arm_ref_high) and arm_ref_high > 0:
                retr = (arm_ref_high - float(row["close"])) / arm_ref_high
                retr_min = (p.micro_retrace_min if use_micro else p.sell_retrace_min)
                if retr < retr_min or retr > retr_max:
                    continue

            pos = -1
            entry_px = float(nxt["open"])
            entry_exec_sc = float(exec_sc)
            entry_ts = nxt["ts"]
            entry_day = entry_ts.date()
            ref_high = float(sell_arm_ref_high) if np.isfinite(sell_arm_ref_high) else float(row["m5_high"])
            best_net = -1e9
            locked = False
            struct_warned = False
            warn_i5 = None
            last_i5 = None
            low_since_entry = np.inf
            weak_break_count = 0

            panic_seen = False
            exhaust_ready = False
            low1 = np.inf
            rebound_seen = False
            hl_ready = False

            trades.append({
                "ts": entry_ts, "action": "SELL_SHORT", "price": entry_px, "ref_high": ref_high,
                "note": ("micro" if use_micro else "base"),
                "sell5_score": float(sell_arm_score) if (not use_micro) else float('nan'),
                "micro3_score": float(micro_arm_score) if use_micro else float('nan'),
                "bias_z": float(row.get("bias_z", 0.0)),
                "sell_exec_score": float(exec_sc)
            })

            sell_armed_until = None
            sell_arm_ref_high = np.nan
            sell_arm_score = 0.0
            micro_armed_until = None
            micro_arm_ref_high = np.nan
            micro_arm_score = 0.0

        else:
            low_since_entry = min(low_since_entry, float(row["low"]))
            if np.isfinite(entry_exec_sc) and (entry_exec_sc < p.sell_exec_weak_th) and np.isfinite(ref_high) and ref_high > 0:
                thr = ref_high * (1.0 + p.loss_guard_weak_break_ratio)
                if float(row["close"]) >= thr:
                    weak_break_count += 1
                else:
                    weak_break_count = 0
            exec_px = float(nxt["open"])
            net_now = net_bps_short(entry_px, exec_px, p.fee_bps, p.slip_bps)
            hold_min = (ts - entry_ts).total_seconds()/60.0

            if net_now > best_net:
                best_net = net_now
            if (not locked) and best_net >= p.lock1_bps:
                locked = True

            if (not panic_seen) and (float(row["vol_z60"]) >= p.panic_vol_z) and (float(row["close"]) < float(row["open"])):
                panic_seen = True
            if panic_seen and (not exhaust_ready):
                if bool(row.get("m3_vol_dn2", False)) and (float(row["m3_hist"]) > float(feat.iloc[max(i-1,0)]["m3_hist"])):
                    exhaust_ready = True

            if float(row["low"]) < low1:
                low1 = float(row["low"])
                rebound_seen = False
                hl_ready = False
            if (not rebound_seen) and np.isfinite(low1) and low1 > 0:
                if float(row["close"]) >= low1 * (1.0 + p.hl_rebound_ratio):
                    rebound_seen = True

            if rebound_seen and (not hl_ready):
                w = p.hl_window_min
                if i >= w:
                    win_low = float(feat.iloc[i-w:i+1]["low"].min())
                    if abs(float(row["low"]) - win_low) < 1e-12 and float(row["low"]) >= low1 * (1.0 + p.hl_min_ratio):
                        hl_ready = True

            is_end_5 = (ts.minute % 5 == 4)
            cur_i5 = int(row["i5"])
            if last_i5 is None: last_i5 = cur_i5
            struct_stop = False
            if is_end_5 and cur_i5 != last_i5:
                last_i5 = cur_i5
                cond_break = float(row["m5_close"]) >= ref_high * (1.0 + p.struct_break_ratio)
                if getattr(p, "struct_use_big_confirm", False):
                    sc5 = follow_big_confirm_score(row)
                    cond_break = cond_break and (sc5 >= getattr(p, "struct_big_confirm_min", 3.4))
                if p.struct_mom_need:
                    mom_ok = (float(row.get("m5_hist_chg", 0.0)) > 0) and (float(row.get("m5_dif_chg", 0.0)) > 0)
                    cond_break = cond_break and mom_ok

                if not struct_warned:
                    if cond_break:
                        struct_warned = True
                        warn_i5 = cur_i5
                else:
                    if warn_i5 is None or cur_i5 > int(warn_i5) + p.struct_second_window:
                        struct_warned = False
                        warn_i5 = None
                        if cond_break:
                            struct_warned = True
                            warn_i5 = cur_i5
                    else:
                        if cond_break:
                            struct_stop = True
                        else:
                            if float(row["m5_close"]) <= ref_high * (1.0 + p.struct_break_ratio * p.struct_warn_reset_mult):
                                struct_warned = False
                                warn_i5 = None

            tp_final_raw = net_now >= p.tp_final_bps
            tp_final = tp_final_raw
            tp_note = "tp_final"

            h0 = float(row["hist"])
            h1 = float(feat.iloc[max(i-1,0)]["hist"])
            h2 = float(feat.iloc[max(i-2,0)]["hist"])
            hist_up2 = (h0 > h1) and (h1 > h2)

            if p.tp_hold_enable and tp_final_raw and (not tp_hold_active) and (hold_min <= float(p.tp_hold_fast_max_min)):
                dif_now = float(row["dif"])
                dea_now = float(row["dea"])
                if (dif_now < dea_now) and (h0 < 0.0):
                    tp_hold_active = True
                    tp_hold_start_ts = ts
                    tp_hold_deadline_ts = ts + pd.Timedelta(minutes=int(p.tp_hold_max_extra_min))
                    tp_hold_note = None
                    tp_final = False

            if tp_hold_active:
                rebound_bps = (float(row["close"]) / low_since_entry - 1.0) * 10000.0 if np.isfinite(low_since_entry) and low_since_entry > 0 else 0.0
                m3_turn = bool(row.get("m3_turn_up", False)) or bool(row.get("m3_rsi_up", False)) or bool(row.get("m3_j_up", False))

                need_turn_ok = (not p.tp_hold_need_m3_turn) or m3_turn
                need_hist_ok = (not p.tp_hold_need_hist_up2) or hist_up2
                release = (rebound_bps >= float(p.tp_hold_release_rebound_bps)) and need_turn_ok and need_hist_ok

                giveback = (best_net - net_now) >= float(p.tp_hold_giveback_bps)
                timeout = (tp_hold_deadline_ts is not None) and (ts >= tp_hold_deadline_ts)

                if release:
                    tp_final = True; tp_note = "tp_hold_release"; tp_hold_active = False
                elif giveback:
                    tp_final = True; tp_note = "tp_hold_giveback"; tp_hold_active = False
                elif timeout:
                    tp_final = True; tp_note = "tp_hold_timeout"; tp_hold_active = False
                else:
                    tp_final = False

            trail = locked and (best_net >= p.trail_enable_bps) and ((best_net - net_now) >= p.trail_dd_bps)

            buy_score = max(float(row["buy3_score"]), float(row["buy6_score"]))
            turn_up = bool(row["turn_up_3"]) or bool(row["turn_up_6"])
            oversold = float(row["m3_rsi"]) <= p.buy_oversold_rsi
            score_ok = (float(row["buy6_score"]) >= p.buy_score_min_6m) or (buy_score >= p.buy_score_min)

            if score_ok and (turn_up or (not p.require_turn_up_buy)) and oversold:
                buy_armed_until = ts + pd.Timedelta(minutes=p.buy_arm_min)
                buy_arm_score = buy_score
            if buy_armed_until is not None and ts > buy_armed_until:
                buy_armed_until = None
                buy_arm_score = 0.0

            mean_exit = False
            buy_exec_sc = 0.0
            if buy_armed_until is not None:
                buy_exec_sc = buy_exec_score_from_row(row, m3)
                bounce = (float(row["close"]) - low_since_entry) / low_since_entry if low_since_entry > 0 else 0.0
                hist_improve = float(row["hist"]) > float(feat.iloc[max(i-1, 0)]["hist"])

                drop_bps = (entry_px / low_since_entry - 1.0) * 10000.0 if np.isfinite(low_since_entry) and low_since_entry > 0 else 0.0
                drop_ok = drop_bps >= p.min_drop_buy_bps
                extreme_ok = buy_arm_score >= p.buy_extreme_score

                hl_ok = hl_ready
                vwap_reclaim = bool(row.get("m3_cross_above_vwap", False))
                vol_ok = (not p.require_vol_exhaust) or ((int(panic_seen) + int(exhaust_ready) + int(vwap_reclaim)) >= p.vol_gate_min_count)

                gate_ok = (drop_ok and hl_ok and vol_ok) or (extreme_ok and hl_ok and vol_ok)

                mean_exit = (
                    gate_ok and
                    (hold_min >= p.min_mean_hold_min) and
                    (net_now >= p.min_exit_profit_bps) and
                    hist_improve and
                    (buy_exec_sc >= p.buy_exec_min) and
                    (bounce <= p.buy_bounce_max)
                )

            if hold_min < p.min_hold_min and (not struct_stop):
                continue

            weak_follow_fail = False
            if np.isfinite(entry_exec_sc) and (entry_exec_sc < p.sell_exec_weak_th):
                if (hold_min >= p.weak_follow_min_hold_min) and (hold_min <= p.weak_follow_window_max_min):
                    favorable_drop_bps = (entry_px / low_since_entry - 1.0) * 10000.0 if np.isfinite(low_since_entry) and low_since_entry > 0 else 0.0
                    if (favorable_drop_bps < p.weak_follow_need_drop_bps) and (best_net < p.weak_follow_best_net_max_bps) and (net_now <= p.weak_follow_exit_loss_bps):
                        sc5 = follow_big_confirm_score(row)
                        if sc5 >= p.weak_follow_big_confirm_min:
                            weak_follow_fail = True

            follow_fail = False
            if (hold_min >= p.follow_min_hold_min) and (hold_min <= p.follow_window_max_min):
                favorable_drop_bps = (entry_px / low_since_entry - 1.0) * 10000.0 if np.isfinite(low_since_entry) and low_since_entry > 0 else 0.0
                if (favorable_drop_bps < p.follow_need_drop_bps) and (net_now <= p.follow_exit_loss_bps):
                    sc5 = follow_big_confirm_score(row)
                    if sc5 >= p.follow_big_confirm_min:
                        follow_fail = True

            loss_guard = False
            loss_guard_weak = False
            above_vwap = (float(row['close']) > float(row['vwap'])) or bool(row.get('m3_cross_above_vwap', False))
            mom_up = bool(row.get('m3_turn_up', False)) or bool(row.get('m3_rsi_up', False)) or bool(row.get('m3_j_up', False))
            if net_now <= p.max_loss_bps:
                if above_vwap and mom_up:
                    loss_guard = True
                elif np.isfinite(entry_exec_sc) and (entry_exec_sc < p.sell_exec_weak_th):
                    loss_guard_weak = True
            elif np.isfinite(entry_exec_sc) and (entry_exec_sc < p.sell_exec_weak_th) and (net_now <= p.max_loss_bps_weak) and (hold_min <= float(p.loss_guard_weak_window_max_min)):
                sc5_lg = follow_big_confirm_score(row)
                persist_ok = False
                if hold_min <= float(p.loss_guard_weak_fast_min):
                    persist_ok = weak_break_count >= 1
                else:
                    persist_ok = weak_break_count >= int(p.loss_guard_weak_confirm_count)
                if persist_ok and ((above_vwap and mom_up) or (sc5_lg >= p.loss_guard_weak_big_confirm_min)):
                    loss_guard_weak = True

            force = False
            force_note = None
            day_gap = int(row["next_gap_days"])
            day_diff = (ts.date() - entry_day).days if entry_day else 0
            hhmm = int(row["hhmm"])
            if hhmm >= p.preclose_hhmm:
                if day_gap > 1:
                    force = True; force_note = "preclose_gap"
                elif not p.allow_overnight:
                    force = True; force_note = "preclose_intraday"
                elif day_diff >= p.max_hold_days:
                    force = True; force_note = "preclose_maxday"
                else:
                    if day_diff == 0:
                        favorable_drop_bps = (entry_px / low_since_entry - 1.0) * 10000.0 if np.isfinite(low_since_entry) and low_since_entry > 0 else 0.0
                        sc5_pc = follow_big_confirm_score(row)
                        if (
                            (favorable_drop_bps < p.overnight_need_drop_bps) and
                            (net_now <= p.preclose_no_drop_loss_bps) and
                            (sc5_pc >= p.preclose_no_drop_big_confirm_min) and
                            (np.isfinite(entry_exec_sc) and entry_exec_sc <= p.preclose_no_drop_exec_th)
                        ):
                            force = True; force_note = "preclose_no_drop"
                        elif (net_now <= p.overnight_max_loss_bps) and (best_net < p.lock1_bps*0.5) and (favorable_drop_bps < p.overnight_need_drop_bps):
                            force = True; force_note = "preclose_no_drop"
                    if (not force) and (net_now < p.carry_min_bps):
                        force = True; force_note = "preclose_carry_fail"

            if struct_stop or tp_final or trail or mean_exit or weak_follow_fail or follow_fail or loss_guard or loss_guard_weak or force or (i == len(feat)-2):
                if struct_stop: note = "struct_stop"
                elif tp_final: note = tp_note if "tp_note" in locals() else "tp_final"
                elif trail: note = "trail"
                elif mean_exit: note = "mean"
                elif weak_follow_fail: note = "weak_follow_fail"
                elif follow_fail: note = "follow_fail"
                elif loss_guard: note = "loss_guard"
                elif loss_guard_weak: note = "loss_guard_weak"
                elif force: note = force_note or "force"
                else: note = "eof"

                exit_ts = nxt["ts"]
                exit_px = exec_px
                trades.append({
                    "ts": exit_ts, "action": "COVER", "price": exit_px, "ref_high": np.nan,
                    "note": note, "net_bps": float(net_bps_short(entry_px, exit_px, p.fee_bps, p.slip_bps)),
                    "hold_min": float((exit_ts - entry_ts).total_seconds()/60.0),
                    "best_net_bps": float(best_net),
                    "buy_score": float(buy_arm_score), "buy_exec_score": float(buy_exec_sc)
                })

                pos = 0
                entry_exec_sc = np.nan
                tp_hold_active = False
                tp_hold_start_ts = None
                tp_hold_deadline_ts = None
                tp_hold_note = None
                weak_break_count = 0
                last_exit = exit_ts
                cycles_today += 1

    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0 and 'ts' in trades_df.columns:
        trades_df = trades_df.sort_values('ts').reset_index(drop=True)
    else:
        trades_df = pd.DataFrame(columns=['ts','side','price','qty','reason','tag'])

    pairs =[]
    stack = None
    for _, r in trades_df.iterrows():
        if r["action"] == "SELL_SHORT":
            stack = r
        elif r["action"] == "COVER" and stack is not None:
            pairs.append({
                "sell_ts": stack["ts"], "sell_price": float(stack["price"]),
                "buy_ts": r["ts"], "buy_price": float(r["price"]),
                "net_bps": float(r.get("net_bps", np.nan)),
                "hold_min": float(r.get("hold_min", np.nan)),
                "exit_note": r.get("note", ""),
                "sell5_score": float(stack.get("sell5_score", np.nan)),
                "sell_exec_score": float(stack.get("sell_exec_score", np.nan)),
                "buy_score": float(r.get("buy_score", np.nan)),
                "buy_exec_score": float(r.get("buy_exec_score", np.nan)),
            })
            stack = None
    pairs_df = pd.DataFrame(pairs).sort_values("sell_ts").reset_index(drop=True)

    eq = [1.0]
    for n in pairs_df["net_bps"].fillna(0.0).to_list():
        eq.append(eq[-1]*(1+n/10000.0))
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq/peak - 1.0)) if len(eq) else 0.0

    summary = {
        "params": asdict(p),
        "period": {"start": str(feat["ts"].min()), "end": str(feat["ts"].max())},
        "cycles": int(len(pairs_df)),
        "win_rate": float((pairs_df["net_bps"] > 0).mean()) if len(pairs_df) else None,
        "total_return": float(eq[-1]-1.0) if len(eq) else 0.0,
        "max_drawdown": mdd,
        "avg_net_bps": float(pairs_df["net_bps"].mean()) if len(pairs_df) else None,
        "median_net_bps": float(pairs_df["net_bps"].median()) if len(pairs_df) else None,
        "median_hold_min": float(pairs_df["hold_min"].median()) if len(pairs_df) else None,
        "exit_counts": pairs_df["exit_note"].value_counts().to_dict() if len(pairs_df) else {}
    }

    return trades_df, pairs_df, summary

# =========================
# Plot utils & Diagnostics
# =========================

def plot_equity(pairs_df: pd.DataFrame, out_png: str, title: str):
    eq = [1.0]
    for n in pairs_df["net_bps"].fillna(0.0).to_list(): eq.append(eq[-1]*(1+n/10000.0))
    eq = np.array(eq)
    plt.figure(figsize=(10,4))
    plt.plot(eq)
    plt.title(title)
    plt.xlabel("Cycle #")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def make_diagnostics(price_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df.empty: return pd.DataFrame()
    d = price_df.copy()
    d["day"] = d["ts"].dt.date.astype(str)
    day_hl = d.groupby("day").agg(day_high=("high","max"), day_low=("low","min")).reset_index()
    day_hl_map = {r.day:(float(r.day_high), float(r.day_low)) for r in day_hl.itertuples(index=False)}

    rows =[]
    for r in pairs_df.itertuples(index=False):
        sell_ts = pd.Timestamp(r.sell_ts)
        buy_ts  = pd.Timestamp(r.buy_ts)
        sell_px = float(r.sell_price)
        buy_px  = float(r.buy_price)
        day = str(sell_ts.date())
        day_high, day_low = day_hl_map.get(day, (float('nan'), float('nan')))

        w = d[(d["ts"] >= sell_ts) & (d["ts"] <= buy_ts)]
        if w.empty: min_low = float('nan'); max_high = float('nan')
        else: min_low = float(w["low"].min()); max_high = float(w["high"].max())

        sell_vs_day_high_bp = (day_high/sell_px - 1.0)*10000.0 if (sell_px>0 and day_high==day_high) else float('nan')
        buy_vs_post_low_bp  = (buy_px/min_low - 1.0)*10000.0 if (min_low>0 and buy_px>0 and min_low==min_low) else float('nan')
        adverse_move_bp     = (max_high/sell_px - 1.0)*10000.0 if (sell_px>0 and max_high==max_high) else float('nan')
        favorable_drop_bp   = (sell_px/min_low - 1.0)*10000.0 if (sell_px>0 and min_low>0 and min_low==min_low) else float('nan')

        rows.append({
            "sell_ts": sell_ts, "buy_ts": buy_ts, "sell_price": sell_px, "buy_price": buy_px,
            "net_bps": float(r.net_bps), "hold_min": float(r.hold_min), "exit_note": str(r.exit_note),
            "sell_day_high": day_high, "sell_day_low": day_low, "min_low_after_sell": min_low, "max_high_after_sell": max_high,
            "sell_vs_day_high_bp": sell_vs_day_high_bp, "buy_vs_post_low_bp": buy_vs_post_low_bp,
            "adverse_move_bp": adverse_move_bp, "favorable_drop_bp": favorable_drop_bp,
            "sell5_score": float(getattr(r, 'sell5_score', float('nan'))),
            "sell_exec_score": float(getattr(r, 'sell_exec_score', float('nan'))),
        })
    return pd.DataFrame(rows)

# =========================
# Universal v2 (Dual-Mode)
# =========================

def cci_20(df_1m: pd.DataFrame) -> pd.Series:
    tp = (df_1m["high"].astype(float) + df_1m["low"].astype(float) + df_1m["close"].astype(float)) / 3.0
    sma = tp.rolling(20, min_periods=20).mean()
    mad = (tp - sma).abs().rolling(20, min_periods=20).mean()
    return (tp - sma) / (0.015 * mad)

def add_extra_indicators(df_1m: pd.DataFrame) -> pd.DataFrame:
    d = df_1m[["ts","open","high","low","close","volume"]].copy().sort_values("ts").reset_index(drop=True)
    d["cci20"] = cci_20(d)
    close = d["close"].astype(float)
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    d["bb_mid"] = mid; d["bb_up"] = mid + 2*std; d["bb_dn"] = mid - 2*std
    d["bb_pct"] = (close - d["bb_dn"]) / (d["bb_up"] - d["bb_dn"])
    d["bb_bw"] = (d["bb_up"] - d["bb_dn"]) / mid
    d["sma20"] = close.rolling(20, min_periods=20).mean()
    d["sma60"] = close.rolling(60, min_periods=60).mean()
    d["sma20_slope"] = (d["sma20"] - d["sma20"].shift(10)) / d["sma20"].shift(10)
    return d[["ts","cci20","bb_mid","bb_up","bb_dn","bb_pct","bb_bw","sma20","sma60","sma20_slope"]]

def net_bps_long(buy_px: float, sell_px: float, fee_bps: float = 1.0, slip_bps: float = 1.0) -> float:
    gross = (sell_px / buy_px - 1.0) * 10000.0
    return float(gross - 2.0*(fee_bps + slip_bps))

def compute_trend_regime(feat_1m: pd.DataFrame, extra: pd.DataFrame) -> pd.Series:
    df = feat_1m[["ts","m5_dif","m5_hist"]].merge(extra[["ts","sma20","sma60","sma20_slope"]], on="ts", how="left")
    flag = (df["m5_dif"] > 0) & (df["m5_hist"] > 0) & (df["sma20"] > df["sma60"]) & (df["sma20_slope"] > 0)
    return pd.Series(flag.fillna(False).to_numpy(), index=df["ts"])

def build_long_trades_m5(
    feat_1m: pd.DataFrame, extra: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
    *, fee_bps: float = 1.0, slip_bps: float = 1.0, rsi_min: float = 50.0, tp_bps: float = 200.0,
    sl_bps: float = -100.0, trail_start: float = 120.0, trail_gap: float = 70.0, dip_memory: int = 30,
) -> pd.DataFrame:
    df = feat_1m[["ts","open","high","low","close","vwap","day","hhmm","eod","m5_dif","m5_hist","m5_rsi","m5_turn_up","m5_cross_above_vwap"]].merge(extra[["ts","bb_pct","sma20","sma60","sma20_slope"]], on="ts", how="left")
    df = df[(df["ts"]>=start_ts) & (df["ts"]<=end_ts)].reset_index(drop=True)
    trades =[]
    pos = 0; entry_px = 0.0; entry_ts = None; best_bps = -1e9; dip_timer = 0
    for i in range(len(df)-1):
        row = df.iloc[i]; nxt = df.iloc[i+1]; hhmm = int(row["hhmm"])
        if float(row["close"]) < float(row["vwap"]): dip_timer = dip_memory
        else: dip_timer = max(0, dip_timer-1)

        if pos == 1:
            cur_bps = net_bps_long(entry_px, float(row["close"]), fee_bps, slip_bps)
            best_bps = max(best_bps, cur_bps)
            exit_note = None
            if cur_bps >= tp_bps: exit_note = "tp"
            elif cur_bps <= sl_bps: exit_note = "sl"
            elif best_bps >= trail_start and cur_bps <= best_bps - trail_gap: exit_note = "trail"
            if exit_note is None and bool(row["eod"]): exit_note = "eod"
            if exit_note is not None:
                exit_px = float(nxt["open"]); exit_ts = pd.Timestamp(nxt["ts"])
                nbps = net_bps_long(entry_px, exit_px, fee_bps, slip_bps)
                trades.append({"entry_ts": entry_ts, "exit_ts": exit_ts, "dir": "LONG", "entry_px": entry_px, "exit_px": exit_px, "net_bps": float(nbps), "entry_note": "m5_pull", "exit_note": exit_note})
                pos = 0; entry_px = 0.0; entry_ts = None; best_bps = -1e9
            continue

        if pos == 0:
            if pd.isna(row.get("m5_hist")) or pd.isna(row.get("m5_dif")) or pd.isna(row.get("m5_rsi")): continue
            if pd.isna(row.get("sma60")): continue
            if float(row["m5_hist"]) <= 0 or float(row["m5_dif"]) <= 0: continue
            if float(row["sma20"]) <= float(row["sma60"]) or float(row["sma20_slope"]) <= 0: continue
            if dip_timer == 0: continue
            if not (bool(row.get("m5_cross_above_vwap", False)) and bool(row.get("m5_turn_up", False))): continue
            if float(row["m5_rsi"]) < rsi_min: continue
            if float(row.get("bb_pct", 0.0)) > 0.92: continue
            if hhmm >= 1450 or (1125 <= hhmm <= 1130): continue
            entry_px = float(nxt["open"]); entry_ts = pd.Timestamp(nxt["ts"])
            pos = 1; best_bps = -1e9

    if pos == 1 and entry_ts is not None:
        last = df.iloc[-1]
        exit_px = float(last["close"]); exit_ts = pd.Timestamp(last["ts"])
        nbps = net_bps_long(entry_px, exit_px, fee_bps, slip_bps)
        trades.append({"entry_ts": entry_ts, "exit_ts": exit_ts, "dir": "LONG", "entry_px": entry_px, "exit_px": exit_px, "net_bps": float(nbps), "entry_note": "m5_pull", "exit_note": "force"})
    return pd.DataFrame(trades)

def build_short_trades_dayf4(df_1m: pd.DataFrame, p: Params):
    feat, m3, m5, m6 = compute_features(df_1m)
    trades_df, pairs_df, summary = run_backtest(feat, m3, p)
    entry_map = trades_df[trades_df["action"]=="SELL_SHORT"][["ts","note","micro3_score"]].rename(columns={"ts":"sell_ts","note":"entry_note","micro3_score":"entry_micro3_score"})
    pairs_en = pairs_df.merge(entry_map, on="sell_ts", how="left")
    m5_hist_map = dict(zip(feat["ts"], feat.get("m5_hist", pd.Series([np.nan]*len(feat)))))
    pairs_en["m5_hist"] = pairs_en["sell_ts"].map(m5_hist_map).astype(float)

    short_trades = pd.DataFrame({
        "entry_ts": pairs_en["sell_ts"], "exit_ts": pairs_en["buy_ts"], "dir": "SHORT", "entry_px": pairs_en["sell_price"],
        "exit_px": pairs_en["buy_price"], "net_bps": pairs_en["net_bps"], "entry_note": pairs_en["entry_note"],
        "exit_note": pairs_en["exit_note"], "sell5_score": pairs_en["sell5_score"], "sell_exec_score": pairs_en["sell_exec_score"],
        "entry_micro3_score": pairs_en["entry_micro3_score"], "m5_hist": pairs_en["m5_hist"],
    })
    return short_trades, feat, trades_df, pairs_df, summary

def simulate_no_overlap(trades_df: pd.DataFrame):
    if trades_df.empty: return trades_df, {"cycles":0,"win_rate":0.0,"total_return":0.0,"max_drawdown":0.0}
    d = trades_df.sort_values("entry_ts").reset_index(drop=True)
    picked =[]; t_cur = pd.Timestamp.min
    for r in d.itertuples(index=False):
        if pd.Timestamp(r.entry_ts) >= t_cur:
            picked.append(r._asdict()); t_cur = pd.Timestamp(r.exit_ts)
    picked_df = pd.DataFrame(picked).sort_values("entry_ts").reset_index(drop=True)
    equity = 1.0; peak = 1.0; maxdd = 0.0
    for nbps in picked_df["net_bps"].astype(float):
        equity *= (1.0 + nbps/10000.0); peak = max(peak, equity); maxdd = min(maxdd, equity/peak - 1.0)
    cycles = int(len(picked_df))
    return picked_df, {"cycles": cycles, "win_rate": float((picked_df["net_bps"] > 0).mean()) if cycles > 0 else 0.0, "total_return": float(equity - 1.0), "max_drawdown": float(maxdd)}

def build_trades_log_from_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df.empty: return pd.DataFrame(columns=["ts","action","price","note","net_bps","hold_min","dir","entry_note","exit_note"])
    rows =[]
    for r in pairs_df.itertuples(index=False):
        entry_ts = pd.Timestamp(r.entry_ts); exit_ts  = pd.Timestamp(r.exit_ts)
        entry_px = float(r.entry_px); exit_px  = float(r.exit_px); nbps     = float(r.net_bps)
        hold_min = float((exit_ts - entry_ts).total_seconds()/60.0); d = str(r.dir)
        en_note = str(getattr(r, "entry_note", "")) if getattr(r, "entry_note", None) is not None else ""
        ex_note  = str(getattr(r, "exit_note", "")) if getattr(r, "exit_note", None) is not None else ""
        if d == "SHORT":
            rows.append({"ts": entry_ts, "action":"SELL_SHORT", "price": entry_px, "note": en_note, "net_bps": np.nan, "hold_min": np.nan, "dir": d, "entry_note": en_note, "exit_note": ex_note})
            rows.append({"ts": exit_ts,  "action":"COVER",      "price": exit_px,  "note": ex_note,  "net_bps": nbps, "hold_min": hold_min, "dir": d, "entry_note": en_note, "exit_note": ex_note})
        else:
            rows.append({"ts": entry_ts, "action":"BUY",        "price": entry_px, "note": en_note, "net_bps": np.nan, "hold_min": np.nan, "dir": d, "entry_note": en_note, "exit_note": ex_note})
            rows.append({"ts": exit_ts,  "action":"SELL",       "price": exit_px,  "note": ex_note,  "net_bps": nbps, "hold_min": hold_min, "dir": d, "entry_note": en_note, "exit_note": ex_note})
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

def run_backtest_universal_v2(df_1m: pd.DataFrame, *, p_short: Params, long_rsi_min: float = 50.0, long_tp_bps: float = 200.0, long_sl_bps: float = -100.0, long_trail_start: float = 120.0, long_trail_gap: float = 70.0, micro_trend_m5hist_min: float = 4.0, base_trend_sell5_max: float = 3.0):
    extra = add_extra_indicators(df_1m)
    short_trades, feat, dayf4_trades, dayf4_pairs, dayf4_summary = build_short_trades_dayf4(df_1m, p_short)
    trend_flag = compute_trend_regime(feat, extra)
    short_trades = short_trades.merge(trend_flag.rename("trend_regime"), left_on="entry_ts", right_index=True, how="left")
    short_trades["trend_regime"] = short_trades["trend_regime"].fillna(False).astype(bool)

    long_trades = build_long_trades_m5(feat, extra, df_1m["ts"].min(), df_1m["ts"].max(), fee_bps=p_short.fee_bps, slip_bps=p_short.slip_bps, rsi_min=long_rsi_min, tp_bps=long_tp_bps, sl_bps=long_sl_bps, trail_start=long_trail_start, trail_gap=long_trail_gap)
    if not long_trades.empty:
        long_trades = long_trades.merge(trend_flag.rename("trend_regime"), left_on="entry_ts", right_index=True, how="left")
        long_trades["trend_regime"] = long_trades["trend_regime"].fillna(False).astype(bool)
        long_trades = long_trades[long_trades["trend_regime"]==True].copy()

    allow_micro_trend = (short_trades["trend_regime"]==True) & (short_trades["entry_note"]=="micro") & (short_trades["m5_hist"]>=micro_trend_m5hist_min)
    allow_base_trend  = (short_trades["trend_regime"]==True) & (short_trades["entry_note"]=="base")  & (short_trades["sell5_score"]<=base_trend_sell5_max)
    short_keep = (short_trades["trend_regime"]!=True) | allow_micro_trend | allow_base_trend
    short_kept = short_trades[short_keep].copy()

    all_candidates = pd.concat([short_kept, long_trades], ignore_index=True) if not long_trades.empty else short_kept
    picked, summ = simulate_no_overlap(all_candidates)

    summ.update({
        "short_cycles": int((picked["dir"]=="SHORT").sum()) if not picked.empty else 0,
        "long_cycles": int((picked["dir"]=="LONG").sum()) if not picked.empty else 0,
        "trend_drop_short_cycles": int(((short_trades["trend_regime"]==True) & (~allow_micro_trend) & (~allow_base_trend)).sum()),
        "trend_allow_micro_cycles": int(allow_micro_trend.sum()),
        "trend_allow_base_cycles": int(allow_base_trend.sum()),
        "dayf4_cycles": int(dayf4_summary.get("cycles", 0)),
        "dayf4_win_rate": float(dayf4_summary.get("win_rate", 0.0)),
        "dayf4_total_return": float(dayf4_summary.get("total_return", 0.0)),
        "dayf4_max_drawdown": float(dayf4_summary.get("max_drawdown", 0.0)),
        "params": {"mode": "universal_v2", "micro_trend_m5hist_min": micro_trend_m5hist_min, "base_trend_sell5_max": base_trend_sell5_max, "long_rsi_min": long_rsi_min, "long_tp_bps": long_tp_bps, "long_sl_bps": long_sl_bps, "long_trail_start": long_trail_start, "long_trail_gap": long_trail_gap}
    })
    return build_trades_log_from_pairs(picked), picked, summ, dayf4_trades, dayf4_pairs, dayf4_summary

# =========================
# V2.6 live helpers
# =========================

def send_serverchan(title: str, desp: str):
    sendkey = "SCT313010T7pd5e5aV7Rq0RpX2Ja5TA0BD"
    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    data = {"title": title, "desp": desp}
    try:
        req = Request(url, data=urlencode(data).encode('utf-8'))
        with urlopen(req, timeout=5.0) as resp:
            pass # ignore
    except Exception as e:
        print(f"[WARN] Server酱推送失败: {e}", file=sys.stderr)

def _infer_symbol_live(df: pd.DataFrame | None = None, symbol_arg: str | None = None, csv_path: str | None = None) -> str:
    if symbol_arg: return str(symbol_arg).strip().lower().replace('sh', '').replace('sz', '')
    if df is not None and 'code' in df.columns and df['code'].notna().any():
        try: return str(df['code'].dropna().iloc[0]).strip().lower().replace('sh', '').replace('sz', '')
        except Exception: pass
    if csv_path:
        m = re.search(r'([0-9]{6})', os.path.basename(str(csv_path)))
        if m: return m.group(1)
    return ''

def _normalize_cn_symbol(symbol: str) -> str:
    s = str(symbol).strip().lower()
    if not s: raise ValueError('empty symbol')
    if s.startswith(('sh', 'sz', 'bj')) and len(s) >= 8: return s
    code = re.sub(r'[^0-9]', '', s)
    if len(code) != 6: raise ValueError(f'bad symbol: {symbol}')
    if code.startswith(('5', '6', '9')): return 'sh' + code
    return 'sz' + code

def _parse_sina_jsonish(text: str):
    s = (text or '').strip()
    if not s: return[]
    for candidate in (s, s.replace('null', 'None')):
        try: return json.loads(candidate)
        except Exception: pass
        try: return ast.literal_eval(candidate)
        except Exception: pass
    raise ValueError(f'cannot parse sina payload: {s[:160]}')

def fetch_sina_kline(symbol: str, scale: int = 1, datalen: int = 1023, timeout: float = 10.0) -> pd.DataFrame:
    sym = _normalize_cn_symbol(symbol)
    q = urlencode({'symbol': sym, 'scale': int(scale), 'ma': 'no', 'datalen': int(datalen)})
    urls =[
        f'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?{q}',
        f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?{q}',
        f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?{q}',
    ]
    last_err = None
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36', 'Referer': 'https://finance.sina.com.cn/', 'Accept': 'application/json,text/plain,*/*'}
    for url in urls:
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode('utf-8', errors='ignore').strip()
            if not raw or raw in ('null', 'None'): continue
            raw2 = raw
            if raw2 and (not raw2.startswith('[') and not raw2.startswith('{')) and '(' in raw2 and raw2.endswith(')'):
                raw2 = raw2[raw2.find('(')+1:-1]
            arr = _parse_sina_jsonish(raw2)
            if not isinstance(arr, list) or len(arr) == 0: continue
            rows =[]
            for x in arr:
                if not isinstance(x, dict): continue
                ts = pd.to_datetime(x.get('day'), errors='coerce')
                if pd.isna(ts): continue
                rows.append({'ts': ts, 'open': float(x.get('open', 'nan')), 'high': float(x.get('high', 'nan')), 'low': float(x.get('low', 'nan')), 'close': float(x.get('close', 'nan')), 'volume': float(x.get('volume', 'nan'))})
            df = pd.DataFrame(rows)
            if df.empty: continue
            df = df.dropna(subset=['ts', 'open', 'high', 'low', 'close', 'volume']).drop_duplicates(subset=['ts'], keep='last').sort_values('ts').reset_index(drop=True)
            if len(df) == 0: continue
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f'fetch sina kline failed for {sym}: {last_err}')

def fetch_sina_quote(symbol: str, timeout: float = 10.0) -> dict:
    sym = _normalize_cn_symbol(symbol)
    url = f'https://hq.sinajs.cn/list={sym}'
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'}
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode('gbk', errors='ignore').strip()
    m = re.match(r'var\s+hq_str_(\w+)="(.*)";?$', raw)
    if not m: return {'symbol': sym, 'raw': raw}
    parts = m.group(2).split(',')
    out = {'symbol': sym, 'name': parts[0] if parts else sym}
    try:
        out.update({'open': float(parts[1]), 'prev_close': float(parts[2]), 'now': float(parts[3]), 'high': float(parts[4]), 'low': float(parts[5]), 'date': parts[30], 'time': parts[31]})
    except Exception: pass
    return out

def _clean_close_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.mask(s <= 0)
    return s.ffill().bfill()

def build_short_records_dayf4_live(df_1m: pd.DataFrame, p: Params):
    feat, m3, m5, m6 = compute_features(df_1m)
    trades_df, pairs_df, summary = run_backtest(feat, m3, p)
    trades_df = trades_df.sort_values('ts').reset_index(drop=True)
    last_bar_ts = pd.Timestamp(df_1m['ts'].max())
    m5_hist_map = dict(zip(feat['ts'], feat.get('m5_hist', pd.Series([np.nan] * len(feat)))))

    rows =[]
    open_entry = None
    for _, r in trades_df.iterrows():
        action = str(r.get('action', ''))
        if action == 'SELL_SHORT':
            open_entry = {'entry_ts': pd.Timestamp(r['ts']), 'entry_px': float(r['price']), 'entry_note': str(r.get('note', '')), 'sell5_score': float(r.get('sell5_score', np.nan)), 'sell_exec_score': float(r.get('sell_exec_score', np.nan)), 'entry_micro3_score': float(r.get('micro3_score', np.nan))}
        elif action == 'COVER' and open_entry is not None:
            exit_ts = pd.Timestamp(r['ts']); exit_note = str(r.get('note', ''))
            if exit_note == 'eof' and exit_ts == last_bar_ts: continue
            rec = dict(open_entry)
            rec.update({'exit_ts': exit_ts, 'exit_px': float(r['price']), 'exit_note': exit_note, 'net_bps': float(r.get('net_bps', np.nan)), 'hold_min': float(r.get('hold_min', np.nan)), 'is_open': False})
            rows.append(rec); open_entry = None

    if open_entry is not None:
        rec = dict(open_entry)
        rec.update({'exit_ts': pd.NaT, 'exit_px': np.nan, 'exit_note': '', 'net_bps': np.nan, 'hold_min': np.nan, 'is_open': True})
        rows.append(rec)

    short_df = pd.DataFrame(rows)
    if short_df.empty: short_df = pd.DataFrame(columns=['entry_ts','exit_ts','entry_px','exit_px','entry_note','exit_note','sell5_score','sell_exec_score','entry_micro3_score','m5_hist','net_bps','hold_min','is_open'])
    else: short_df['m5_hist'] = short_df['entry_ts'].map(m5_hist_map).astype(float); short_df = short_df.sort_values('entry_ts').reset_index(drop=True)
    return short_df, feat, trades_df, pairs_df, summary

def default_v26_live_config(symbol: str) -> dict:
    s = str(symbol).strip().lower().replace('sh', '').replace('sz', '')
    if s == '300308':
        return {'total_shares': 800, 'core_shares': 500, 'skip_1000_1030': True, 'skip_1030_1300': True, 'entry_cutoff_hhmm': 1430, 'base_sell5_min': None, 'micro_before_hhmm': 1000}
    return {'total_shares': 800, 'core_shares': 500, 'skip_1000_1030': True, 'skip_1030_1300': True, 'entry_cutoff_hhmm': 1430, 'base_sell5_min': 3.0, 'micro_before_hhmm': None}

def apply_v26_live_filters(short_df: pd.DataFrame, *, skip_1000_1030: bool = True, skip_1030_1300: bool = True, entry_cutoff_hhmm: int = 1430, base_sell5_min: float | None = None, micro_before_hhmm: int | None = None) -> pd.DataFrame:
    if short_df.empty: return short_df.copy()
    s = short_df.copy()
    hhmm = s['entry_ts'].dt.hour * 100 + s['entry_ts'].dt.minute
    keep = pd.Series(True, index=s.index)
    if skip_1000_1030: keep &= ~((hhmm >= 1000) & (hhmm < 1030))
    if skip_1030_1300: keep &= ~((hhmm >= 1030) & (hhmm < 1300))
    if entry_cutoff_hhmm is not None: keep &= (hhmm < int(entry_cutoff_hhmm))
    if base_sell5_min is not None: keep &= ~((s['entry_note'] == 'base') & (pd.to_numeric(s['sell5_score'], errors='coerce').fillna(-999) < float(base_sell5_min)))
    if micro_before_hhmm is not None: keep &= ~((s['entry_note'] == 'micro') & (hhmm >= int(micro_before_hhmm)))
    return s.loc[keep].copy().sort_values('entry_ts').reset_index(drop=True)

def _stable_float_token(value, ndigits: int = 4) -> str:
    try:
        v = float(value)
        if not np.isfinite(v): return 'nan'
        return f"{v:.{int(ndigits)}f}"
    except Exception: return 'nan'

def _stable_pair_id_from_record(rec: dict) -> str:
    entry_ts = pd.Timestamp(rec.get('entry_ts'))
    return f"{entry_ts.strftime('%Y%m%d%H%M%S')}|{str(rec.get('entry_note', ''))}|{_stable_float_token(rec.get('entry_px'), 4)}"

def _stable_event_id(ts, action: str, qty: int, pair_id: str, price=None) -> str:
    return f"{pd.Timestamp(ts).isoformat()}|{str(action)}|{int(qty)}|{_stable_float_token(price, 4)}|{str(pair_id)}"

def _load_seen_event_ids(state: dict) -> tuple[list[str], set[str]]:
    seen_list: list[str] = []; seen_set: set[str] = set()
    for raw in state.get('seen_event_ids',[]):
        eid = str(raw)
        if (not eid) or (eid in seen_set): continue
        seen_list.append(eid); seen_set.add(eid)
    return seen_list, seen_set

def _remember_seen_event(eid: str, seen_list: list[str], seen_set: set[str], limit: int = 2000):
    eid = str(eid)
    if (not eid) or (eid in seen_set): return
    seen_list.append(eid); seen_set.add(eid)
    overflow = len(seen_list) - int(limit)
    if overflow > 0:
        del seen_list[:overflow]; seen_set.clear(); seen_set.update(seen_list)

def build_v26_overlay_events(short_df: pd.DataFrame, *, total_shares: int = 800, core_shares: int = 500, fee_bps: float = 1.0, slip_bps: float = 1.0, lot_size: int = 100) -> tuple[pd.DataFrame, dict]:
    total_shares = int(total_shares); core_shares = max(0, min(int(core_shares), total_shares))
    trade_max = int(total_shares - core_shares); trade_hold = trade_max; cash = 0.0
    per_side_cost = float(fee_bps + slip_bps) / 10000.0; rows =[]

    if short_df.empty: return pd.DataFrame(columns=['ts','action','price','qty','cash_after','shares_after','note','event_id','entry_note','exit_note','pair_id']), {'trade_hold': trade_hold, 'trade_max': trade_max, 'cash': cash, 'core_shares': core_shares, 'total_shares': total_shares, 'open_pair_id': ''}

    s = short_df.sort_values('entry_ts').reset_index(drop=True)
    for rec in s.to_dict('records'):
        pair_id = _stable_pair_id_from_record(rec)
        sell_qty = int(trade_hold)
        if sell_qty > 0:
            entry_px = float(rec['entry_px'])
            cash += float(sell_qty * entry_px * (1.0 - per_side_cost)); trade_hold -= sell_qty
            ts = pd.Timestamp(rec['entry_ts'])
            rows.append({'ts': ts, 'action': 'SELL', 'price': entry_px, 'qty': sell_qty, 'cash_after': float(cash), 'shares_after': int(core_shares + trade_hold), 'note': str(rec.get('entry_note', '')), 'entry_note': str(rec.get('entry_note', '')), 'exit_note': str(rec.get('exit_note', '')), 'pair_id': pair_id, 'event_id': _stable_event_id(ts, 'SELL', sell_qty, pair_id, entry_px)})

        if pd.notna(rec.get('exit_ts')):
            exit_px = float(rec['exit_px'])
            per_share_buy = exit_px * (1.0 + per_side_cost)
            max_buy = int(cash // per_share_buy)
            if lot_size > 1: max_buy = (max_buy // int(lot_size)) * int(lot_size)
            buy_qty = int(min(int(trade_max - trade_hold), max_buy))
            if buy_qty > 0:
                cash -= float(buy_qty * per_share_buy); trade_hold += buy_qty
                ts = pd.Timestamp(rec['exit_ts'])
                rows.append({'ts': ts, 'action': 'BUY', 'price': exit_px, 'qty': buy_qty, 'cash_after': float(cash), 'shares_after': int(core_shares + trade_hold), 'note': str(rec.get('exit_note', '')), 'entry_note': str(rec.get('entry_note', '')), 'exit_note': str(rec.get('exit_note', '')), 'pair_id': pair_id, 'event_id': _stable_event_id(ts, 'BUY', buy_qty, pair_id, exit_px)})

    open_pair_id = ''
    if not s.empty and bool(s.iloc[-1].get('is_open', False)): open_pair_id = _stable_pair_id_from_record(s.iloc[-1].to_dict())
    ev = pd.DataFrame(rows).sort_values(['ts', 'action']).reset_index(drop=True) if rows else pd.DataFrame(columns=['ts','action','price','qty','cash_after','shares_after','note','event_id','entry_note','exit_note','pair_id'])
    return ev, {'trade_hold': int(trade_hold), 'trade_max': int(trade_max), 'cash': float(cash), 'core_shares': int(core_shares), 'total_shares': int(total_shares), 'open_pair_id': open_pair_id}

def _market_open_cn(now: datetime | None = None) -> bool:
    tz = ZoneInfo('Asia/Shanghai')
    now = now or datetime.now(tz)
    if now.tzinfo is None: now = now.replace(tzinfo=tz)
    if now.weekday() >= 5: return False
    hhmm = now.hour * 100 + now.minute
    return (930 <= hhmm < 1131) or (1300 <= hhmm < 1501)

def _load_json(path: str, default):
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return default

def _save_json(path: str, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)

def _append_alert_rows(path: str, rows: list[dict]):
    if not rows: return
    df = pd.DataFrame(rows)
    if os.path.exists(path) and os.path.getsize(path) > 0: df.to_csv(path, mode='a', header=False, index=False)
    else: df.to_csv(path, index=False)

def _fmt_action_row(symbol: str, row: dict, status: dict, latest_bar_ts: pd.Timestamp, latest_close: float) -> str:
    action = row['action']
    side = '卖出' if action == 'SELL' else '买入'
    shares_after = int(row.get('shares_after', status.get('current_total_shares', 0)))
    cash_after = float(row.get('cash_after', status.get('cash', 0.0)))
    return (
        f"[{action}] {symbol} | {side}{int(row['qty'])}股 | 参考价={float(row['price']):.3f} | note={row.get('note','')} | "
        f"信号bar={pd.Timestamp(row['ts']).strftime('%Y-%m-%d %H:%M:%S')} | "
        f"当前扫描bar={pd.Timestamp(latest_bar_ts).strftime('%Y-%m-%d %H:%M:%S')} (close {float(latest_close):.3f}) | "
        f"事件后持股={shares_after}股 现金={cash_after:.2f}"
    )

def build_live_v26_snapshot(df_1m: pd.DataFrame, symbol: str, config: dict) -> dict:
    p = Params()
    short_df, feat, raw_trades, raw_pairs, raw_summary = build_short_records_dayf4_live(df_1m, p)
    filt = apply_v26_live_filters(short_df, skip_1000_1030=bool(config.get('skip_1000_1030', True)), skip_1030_1300=bool(config.get('skip_1030_1300', True)), entry_cutoff_hhmm=int(config.get('entry_cutoff_hhmm', 1430)), base_sell5_min=config.get('base_sell5_min', None), micro_before_hhmm=config.get('micro_before_hhmm', None))
    events_df, pos = build_v26_overlay_events(filt, total_shares=int(config.get('total_shares', 800)), core_shares=int(config.get('core_shares', 500)), fee_bps=p.fee_bps, slip_bps=p.slip_bps, lot_size=100)
    latest_bar_ts = pd.Timestamp(df_1m['ts'].max())
    latest_close = float(_clean_close_series(df_1m['close']).iloc[-1])
    return {
        'symbol': symbol, 'config': config, 'latest_bar_ts': latest_bar_ts, 'latest_close': latest_close,
        'short_records': short_df, 'filtered_records': filt, 'events': events_df,
        'latest_event': events_df.iloc[-1].to_dict() if not events_df.empty else None,
        'status': {
            'current_total_shares': int(pos['core_shares'] + pos['trade_hold']), 'core_shares': int(pos['core_shares']),
            'trade_hold_shares': int(pos['trade_hold']), 'trade_target_shares': int(pos['trade_max']), 'cash': float(pos['cash']),
            'holding_trade': bool(pos['trade_hold'] > 0), 'waiting_action': 'SELL' if (pos['trade_hold'] > 0) else 'BUY',
            'open_pair_id': pos.get('open_pair_id', ''),
            'open_signal': filt.iloc[-1].to_dict() if (not filt.empty and bool(filt.iloc[-1].get('is_open', False))) else None,
        },
        'raw_summary': raw_summary,
    }

def run_live_v26(args):
    symbol = _infer_symbol_live(None, args.symbol, args.csv)
    if not symbol: raise SystemExit('--mode live_v26 requires --symbol')
    symbol = _normalize_cn_symbol(symbol)
    symbol6 = symbol[-6:]

    cfg = default_v26_live_config(symbol6)
    if int(args.live_total_shares) > 0: cfg['total_shares'] = int(args.live_total_shares)
    if int(args.live_core_shares) >= 0: cfg['core_shares'] = int(args.live_core_shares)
    if args.live_skip_1000_1030: cfg['skip_1000_1030'] = True
    if args.live_no_skip_1000_1030: cfg['skip_1000_1030'] = False
    if args.live_skip_1030_1300: cfg['skip_1030_1300'] = True
    if args.live_no_skip_1030_1300: cfg['skip_1030_1300'] = False
    if int(args.live_entry_cutoff_hhmm) > 0: cfg['entry_cutoff_hhmm'] = int(args.live_entry_cutoff_hhmm)
    if not np.isnan(args.live_base_sell5_min): cfg['base_sell5_min'] = float(args.live_base_sell5_min)
    if int(args.live_micro_before_hhmm) > 0: cfg['micro_before_hhmm'] = int(args.live_micro_before_hhmm)
    if args.live_no_micro_cutoff: cfg['micro_before_hhmm'] = None

    out_prefix = args.out_prefix or f'v26_live_{symbol6}'
    state_file = args.state_file or f'{out_prefix}_state.json'
    alerts_csv = args.alerts_csv or f'{out_prefix}_alerts.csv'
    latest_json = args.latest_json or f'{out_prefix}_latest.json'

    state = _load_json(state_file, {'initialized': False, 'seen_event_ids':[]})
    seen_list, seen = _load_seen_event_ids(state)
    seen_limit = 2000
    tz = ZoneInfo('Asia/Shanghai')

    while True:
        now = datetime.now(tz)
        if (not args.disable_market_hours_gate) and (not _market_open_cn(now)):
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {symbol6} 非交易时段，等待下一轮。")
            if args.once: return
            time.sleep(max(int(args.poll_sec), 5))
            continue

        try:
            if args.csv:
                raw = pd.read_csv(args.csv, parse_dates=['ts'])
                df = raw[[c for c in ['ts','code','open','high','low','close','volume'] if c in raw.columns]].dropna().sort_values('ts').reset_index(drop=True)
            else:
                df = fetch_sina_kline(symbol6, scale=1, datalen=int(args.datalen), timeout=float(args.timeout_sec))
                df['code'] = symbol6
            
            snap = build_live_v26_snapshot(df, symbol6, cfg)
            status = snap['status']
            quote = {}
            if args.fetch_quote:
                try: quote = fetch_sina_quote(symbol6, timeout=float(args.timeout_sec))
                except Exception: pass

            events_df = snap['events']
            latest_bar_ts = pd.Timestamp(snap['latest_bar_ts'])
            latest_close = float(snap['latest_close'])
            new_rows =[]
            
            prev_latest_bar_ts = pd.NaT
            try:
                last_latest_bar_ts_raw = state.get('last_latest_bar_ts', '')
                if last_latest_bar_ts_raw: prev_latest_bar_ts = pd.Timestamp(last_latest_bar_ts_raw)
            except Exception: pass

            if not state.get('initialized', False):
                # 首次启动：把当前已有的事件全扫一遍，加入 seen。只有在力挺且刚好是最新的那一根时才推送
                for _, ev in events_df.iterrows():
                    eid = str(ev['event_id'])
                    row = ev.to_dict()
                    event_ts = pd.Timestamp(row['ts'])

                    if args.force_alert_latest and (event_ts == latest_bar_ts) and (eid not in seen):
                        msg = _fmt_action_row(symbol6, row, status, latest_bar_ts, latest_close)
                        print(msg)
                        send_serverchan(f"[{row['action']}] {symbol6} 策略触发", msg)
                        new_rows.append({
                            'symbol': symbol6, 'alert_ts': datetime.now(tz).isoformat(), 'signal_ts': event_ts.isoformat(),
                            'action': row['action'], 'price': float(row['price']), 'qty': int(row['qty']), 'note': row.get('note',''),
                            'latest_bar_ts': latest_bar_ts.isoformat(), 'latest_close': latest_close,
                            'shares_after': int(row['shares_after']), 'cash_after': float(row['cash_after']), 'event_id': eid,
                        })
                    _remember_seen_event(eid, seen_list, seen, limit=seen_limit)
                state['initialized'] = True
            else:
                # 正常轮询扫描
                for _, ev in events_df.iterrows():
                    eid = str(ev['event_id'])
                    if eid in seen: continue
                    
                    row = ev.to_dict()
                    event_ts = pd.Timestamp(row['ts'])
                    
                    should_alert = bool(getattr(args, 'alert_all_unseen', False))
                    if not should_alert:
                        # 只对“晚于上次最新 bar 的新事件”或“当前最新生成的这一根 bar 的事件”进行通知，阻止所有老日期的幽灵信号
                        if pd.notna(prev_latest_bar_ts):
                            should_alert = (event_ts > prev_latest_bar_ts)
                        else:
                            should_alert = (event_ts >= latest_bar_ts)
                            
                    if should_alert:
                        msg = _fmt_action_row(symbol6, row, status, latest_bar_ts, latest_close)
                        print(msg)
                        send_serverchan(f"[{row['action']}] {symbol6} 策略触发", msg)
                        new_rows.append({
                            'symbol': symbol6, 'alert_ts': datetime.now(tz).isoformat(), 'signal_ts': event_ts.isoformat(),
                            'action': row['action'], 'price': float(row['price']), 'qty': int(row['qty']), 'note': row.get('note',''),
                            'latest_bar_ts': latest_bar_ts.isoformat(), 'latest_close': latest_close,
                            'shares_after': int(row['shares_after']), 'cash_after': float(row['cash_after']), 'event_id': eid,
                        })
                    # 无论是否发通知，都把它记下避免重复算
                    _remember_seen_event(eid, seen_list, seen, limit=seen_limit)

            status_payload = {
                'symbol': symbol6, 'asof': datetime.now(tz).isoformat(), 'latest_bar_ts': latest_bar_ts.isoformat(),
                'latest_close': latest_close, 'quote': quote, 'status': status, 'config': cfg,
                'latest_event': snap['latest_event'], 'filtered_signal_count': int(len(snap['filtered_records'])),
                'raw_signal_count': int(len(snap['short_records'])),
            }
            _save_json(latest_json, status_payload)
            _append_alert_rows(alerts_csv, new_rows)

            state['seen_event_ids'] = seen_list[-seen_limit:]
            state['last_latest_bar_ts'] = latest_bar_ts.isoformat()
            state['symbol'] = symbol6
            _save_json(state_file, state)

            if args.print_status_each_loop or args.once:
                cur = status['current_total_shares']
                trade = status['trade_hold_shares']
                quote_now = quote.get('now', latest_close)
                print(
                    f"[STATUS] {symbol6} 最新bar={latest_bar_ts.strftime('%Y-%m-%d %H:%M:%S')} close={latest_close:.3f} now={float(quote_now):.3f} | "
                    f"总持股={cur} 核心仓={status['core_shares']} 交易仓={trade}/{status['trade_target_shares']} | "
                    f"下一动作={status['waiting_action']} | 原始信号={len(snap['short_records'])} 过滤后={len(snap['filtered_records'])}"
                )

        except Exception as e:
            print(f'[ERROR] {symbol6} live loop failed: {e}', file=sys.stderr)

        if args.once: return
        time.sleep(max(int(args.poll_sec), 5))

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', help='1m merged csv for backtest, or optional offline source for live mode')
    ap.add_argument('--symbol', default='', help='6-digit A-share symbol, e.g. 688498 / sh688498 / 300308')
    ap.add_argument('--start', default='2025-09-01')
    ap.add_argument('--end', default='2026-03-04')
    ap.add_argument('--mode', choices=['dayf4','universal_v2','live_v26'], default='live_v26')
    ap.add_argument('--out_prefix', default='universal_v2_out')
    ap.add_argument('--no_plots', action='store_true')

    ap.add_argument('--long_rsi_min', type=float, default=50.0)
    ap.add_argument('--long_tp_bps', type=float, default=200.0)
    ap.add_argument('--long_sl_bps', type=float, default=-100.0)
    ap.add_argument('--long_trail_start', type=float, default=120.0)
    ap.add_argument('--long_trail_gap', type=float, default=70.0)
    ap.add_argument('--micro_trend_m5hist_min', type=float, default=4.0)
    ap.add_argument('--base_trend_sell5_max', type=float, default=3.0)

    ap.add_argument('--poll_sec', type=int, default=20)
    ap.add_argument('--datalen', type=int, default=1023)
    ap.add_argument('--timeout_sec', type=float, default=10.0)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--state_file', default='')
    ap.add_argument('--alerts_csv', default='')
    ap.add_argument('--latest_json', default='')
    ap.add_argument('--fetch_quote', action='store_true')
    ap.add_argument('--force_alert_latest', action='store_true')
    ap.add_argument('--print_status_each_loop', action='store_true')
    ap.add_argument('--disable_market_hours_gate', action='store_true')
    ap.add_argument('--alert_all_unseen', action='store_true')

    ap.add_argument('--live_total_shares', type=int, default=-1)
    ap.add_argument('--live_core_shares', type=int, default=-1)
    ap.add_argument('--live_skip_1000_1030', action='store_true')
    ap.add_argument('--live_no_skip_1000_1030', action='store_true')
    ap.add_argument('--live_skip_1030_1300', action='store_true')
    ap.add_argument('--live_no_skip_1030_1300', action='store_true')
    ap.add_argument('--live_entry_cutoff_hhmm', type=int, default=-1)
    ap.add_argument('--live_base_sell5_min', type=float, default=np.nan)
    ap.add_argument('--live_micro_before_hhmm', type=int, default=-1)
    ap.add_argument('--live_no_micro_cutoff', action='store_true')

    args = ap.parse_args()

    if args.mode == 'live_v26':
        run_live_v26(args)
        return

    if not args.csv: raise SystemExit('--csv is required for backtest')
    df = pd.read_csv(args.csv, parse_dates=['ts'])
    df = df[['ts','open','high','low','close','volume']].dropna().sort_values('ts').reset_index(drop=True)
    def _to_ts(s: str, is_end: bool) -> pd.Timestamp:
        s = str(s).strip()
        if len(s) == 10: s = s + (' 23:59:59' if is_end else ' 00:00:00')
        return pd.Timestamp(s)
    df = df[(df['ts'] >= _to_ts(args.start, False)) & (df['ts'] <= _to_ts(args.end, True))].copy().reset_index(drop=True)

    out_trades = f'{args.out_prefix}_trades.csv'; out_pairs  = f'{args.out_prefix}_pairs.csv'
    out_sum    = f'{args.out_prefix}_summary.json'; out_eqpng  = f'{args.out_prefix}_equity.png'
    out_diag   = f'{args.out_prefix}_diagnostics.csv'

    if args.mode == 'dayf4':
        p = Params()
        feat, m3, m5, m6 = compute_features(df)
        trades_df, pairs_df, summary = run_backtest(feat, m3, p)
        trades_df.to_csv(out_trades, index=False); pairs_df.to_csv(out_pairs, index=False)
        diag_df = make_diagnostics(df, pairs_df)
        if not diag_df.empty: diag_df.to_csv(out_diag, index=False)
        with open(out_sum, 'w', encoding='utf-8') as f: json.dump(summary, f, ensure_ascii=False, indent=2)
        if not args.no_plots: plot_equity(pairs_df, out_eqpng, f'Equity Curve {args.out_prefix}')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    p_short = Params()
    trades_log, picked_pairs, summary, dayf4_trades, dayf4_pairs, dayf4_summary = run_backtest_universal_v2(
        df, p_short=p_short, long_rsi_min=args.long_rsi_min, long_tp_bps=args.long_tp_bps, long_sl_bps=args.long_sl_bps, long_trail_start=args.long_trail_start, long_trail_gap=args.long_trail_gap, micro_trend_m5hist_min=args.micro_trend_m5hist_min, base_trend_sell5_max=args.base_trend_sell5_max,
    )
    trades_log.to_csv(out_trades, index=False); picked_pairs.to_csv(out_pairs, index=False)
    with open(out_sum, 'w', encoding='utf-8') as f: json.dump(summary, f, ensure_ascii=False, indent=2)
    if not args.no_plots: plot_equity(picked_pairs.rename(columns={'net_bps':'net_bps'}), out_eqpng, f'Equity Curve {args.out_prefix}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()