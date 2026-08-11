import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastdtw import fastdtw
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
REMOTE_TRACKER_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit/main/docs/ai_recommend_tracker.json"

# 공통 헤더 정의
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def calculate_dtw_similarity(seq1, seq2):
    try:
        s1 = np.asarray(seq1, dtype=np.float64).reshape(-1)
        s2 = np.asarray(seq2, dtype=np.float64).reshape(-1)
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0
        s1, s2 = s1[-min_len:], s2[-min_len:]
        distance, _ = fastdtw(s1, s2, dist=lambda x, y: abs(x - y))
        avg_dist = distance / min_len
        return round(float(np.exp(-1.5 * avg_dist) * 100.0), 1)
    except Exception:
        return 0.0

def calculate_max_dtw(seq1, golden_patterns):
    if not golden_patterns:
        return 0.0
    max_sim = 0.0
    for pattern in golden_patterns:
        sim = calculate_dtw_similarity(seq1, pattern)
        if sim > max_sim:
            max_sim = sim
    return max_sim

def fetch_5m_candles(market, count=120):
    url = f"https://api.upbit.com/v1/candles/minutes/5?market={market}&count={count}"
    try:
        # 업비트 Open API 초당 제한 준수를 위한 미세 지연
        time.sleep(0.08)
        res = requests.get(url, headers=HEADERS, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def calculate_atr(df, period=14):
    try:
        high, low, close = df['high_price'], df['low_price'], df['trade_price'].shift(1)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr if not np.isnan(atr) else (df['trade_price'].iloc[-1] * 0.015)
    except Exception:
        return df['trade_price'].iloc[-1] * 0.015

def fetch_remote_recommendations():
    try:
        res = requests.get(f"{REMOTE_TRACKER_URL}?t={int(time.time())}", headers=HEADERS, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data and isinstance(data, list):
                latest = data[-1]
                return [c.get("symbol") for c in latest.get("recommended_coins", []) if c.get("symbol")]
    except Exception:
        pass
    return []

def analyze_single_coin(market, k_name, golden_price_patterns, golden_vol_patterns, weights, recommended_symbols):
    ticker = market.replace("KRW-", "")
    candles = fetch_5m_candles(market, count=120)
    if len(candles) < 60:
        return None

    df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)
    current_price = df.iloc[-1]["trade_price"]
    
    if 'prev_closing_price' in df.columns:
        prev_close = df.iloc[-1]['prev_closing_price']
    elif 'trade_price' in df.columns and len(df) > 1:
        prev_close = df.iloc[-2]['trade_price']
    else:
        prev_close = df.iloc[-1].get('opening_price', df.iloc[-1]['trade_price'])

    change_rate = ((current_price - prev_close) / (prev_close + 1e-8)) * 100.0 if prev_close > 0 else 0.0

    df_2h = df.iloc[-24:].copy().reset_index(drop=True)
    prices, volumes = df_2h["trade_price"].values, df_2h["candle_acc_trade_volume"].values
    
    p_range = (prices.max() - prices.min()) or 1.0
    v_range = (volumes.max() - volumes.min()) or 1.0

    norm_prices = (prices - prices.min()) / p_range
    norm_volumes = (volumes - volumes.min()) / v_range

    price_sim = calculate_max_dtw(norm_prices, golden_price_patterns)
    vol_sim = calculate_max_dtw(norm_volumes, golden_vol_patterns)
    combined_pattern_sim = round(price_sim * 0.7 + vol_sim * 0.3, 1)

    recent_vol = df.iloc[-1]["candle_acc_trade_volume"]
    avg_prev_vol = df.iloc[-21:-1]["candle_acc_trade_volume"].mean()
    vol_cliff_score = min(100.0, max(0.0, (1.0 - (recent_vol / (avg_prev_vol + 1e-8))) * 100.0)) if avg_prev_vol > 0 else 0.0

    df["ma5"] = df["trade_price"].rolling(5).mean()
    df["ma20"] = df["trade_price"].rolling(20).mean()
    df["ma60"] = df["trade_price"].rolling(60).mean()
    last = df.iloc[-1]
    ma_score = 100.0 if last["ma5"] > last["ma20"] > last["ma60"] else (60.0 if last["ma5"] > last["ma20"] else 20.0)

    base_score = (
        combined_pattern_sim * weights.get("w_pattern", 0.20) +
        vol_cliff_score * weights.get("w_vol_cliff", 0.25) +
        ma_score * weights.get("w_ma_alignment", 0.25) +
        min(100.0, max(0.0, change_rate * 3.33)) * weights.get("w_daily_momentum", 0.10) +
        (current_price / df["high_price"].max() * 100) * weights.get("w_breakout", 0.05)
    )

    if ticker in recommended_symbols:
        base_score += 15.0

    atr = calculate_atr(df)
    tp1 = current_price + (atr * 2.0)
    sl = current_price - (atr * 1.5)

    return {
        "market": market, "ticker": ticker, "name": k_name,
        "current_price": current_price, "change_rate": round(change_rate, 2),
        "score": round(min(100.0, base_score), 2), "pattern_similarity": combined_pattern_sim,
        "tp1": round(tp1, 2), "sl": round(sl, 2), "is_repo1_recommended": ticker in recommended_symbols
    }

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    weights = load_json(WEIGHTS_FILE, {
        "w_pattern": 0.20, "w_vol_cliff": 0.25, "w_ma_alignment": 0.25,
        "w_vol_surge": 0.15, "w_daily_momentum": 0.10, "w_breakout": 0.05
    })

    pattern_data = load_json(PATTERN_FILE, {})
    golden_price_patterns = pattern_data.get("golden_patterns", [])
    golden_vol_patterns = pattern_data.get("golden_volume_patterns", [])

    recommended_symbols = fetch_remote_recommendations()

    # 업비트 전체 KRW 마켓 코인 조회
    all_krw = []
    try:
        res = requests.get("https://api.upbit.com/v1/market/all", headers=HEADERS, timeout=5)
        if res.status_code == 200 and isinstance(res.json(), list):
            all_krw = [m for m in res.json() if isinstance(m, dict) and m.get("market", "").startswith("KRW-")]
    except Exception as e:
        print(f"⚠️ 업비트 마켓 목록 조회 실패: {e}")

    if not all_krw:
        print("⚠️ KRW 마켓 코인을 불러오지 못했습니다.")
        return

    analyzed_results = []
    # API 요청 제한을 고려해 max_workers를 3으로 안정화
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                analyze_single_coin, item["market"], item["korean_name"],
                golden_price_patterns, golden_vol_patterns, weights, recommended_symbols
            ) for item in all_krw
        ]
        for f in as_completed(futures):
            r = f.result()
            if r: 
                analyzed_results.append(r)

    if analyzed_results:
        analyzed_results.sort(key=lambda x: x["score"], reverse=True)
        save_json(HISTORY_FILE, analyzed_results[:20])
        print(f"✅ 전체 {len(all_krw)}개 중 {len(analyzed_results)}개 코인 분석 완료 (1위: {analyzed_results[0]['ticker']} - {analyzed_results[0]['score']}점)")
    else:
        print("⚠️ 분석된 결과가 없습니다.")

if __name__ == "__main__":
    main()
