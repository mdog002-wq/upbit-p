import asyncio
import json
import requests
import time
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastdtw import fastdtw

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history_db.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
PATTERN_FILE = os.path.join(DATA_DIR, "golden_pattern.json")
REMOTE_TRACKER_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit/main/docs/ai_recommend_tracker.json"

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
        res = requests.get(url, timeout=5)
        if res.status_code == 200: 
            return res.json()
    except Exception: 
        pass
    return []

# ==========================================
# [신규] 과거 급등 데이터 기반 골든 패턴 자동 학습 함수
# ==========================================
def extract_and_update_golden_patterns(all_krw_markets):
    print("🔍 과거 급등 종목 패턴 수집 및 학습 시작...")
    collected_price_patterns = []
    collected_vol_patterns = []

    for item in all_krw_markets[:30]:  # 주요 30개 마켓 샘플링 학습
        candles = fetch_5m_candles(item["market"], count=120)
        if len(candles) < 72: 
            continue

        df = pd.DataFrame(candles).sort_values("timestamp").reset_index(drop=True)

        # 5분 봉 기준 최근 6시간 동안 단기 +4% 이상 상승이 발생한 구간의 '직전 36봉(3시간)' 추출
        for i in range(36, len(df) - 1):
            prev_close = df.iloc[i-1]["trade_price"]
            curr_high = df.iloc[i]["high_price"]
            surge_pct = ((curr_high - prev_close) / prev_close) * 100.0

            if surge_pct >= 4.0:  # 상승 파동 포착
                pattern_df = df.iloc[i-36:i]
                p = pattern_df["trade_price"].values
                v = pattern_df["candle_acc_trade_volume"].values

                p_norm = (p - p.min()) / ((p.max() - p.min()) or 1.0)
                v_norm = (v - v.min()) / ((v.max() - v.min()) or 1.0)

                collected_price_patterns.append(p_norm.tolist())
                collected_vol_patterns.append(v_norm.tolist())

    # 추출된 성공 패턴 중 대표 패턴 3가지 그룹화 (기본값 보장)
    if len(collected_price_patterns) >= 3:
        # 간단한 평균 기반 대표 패턴 3종 분할 생성
        step = len(collected_price_patterns) // 3
        golden_price_patterns = [
            np.mean(collected_price_patterns[:step], axis=0).tolist(),
            np.mean(collected_price_patterns[step:step*2], axis=0).tolist(),
            np.mean(collected_price_patterns[step*2:], axis=0).tolist()
        ]
        golden_vol_patterns = [
            np.mean(collected_vol_patterns[:step], axis=0).tolist(),
            np.mean(collected_vol_patterns[step:step*2], axis=0).tolist(),
            np.mean(collected_vol_patterns[step*2:], axis=0).tolist()
        ]
    else:
        # 데이터 부족 시 기본 상승/눌림목 표준 패턴으로 보장
        x = np.linspace(0, 1, 36)
        golden_price_patterns = [
            (x**2).tolist(),                    # 1. 완만한 가속 상승형
            (0.5 + 0.5 * np.sin(x * np.pi)).tolist(), # 2. N자형 눌림목 반등형
            (x).tolist()                        # 3. 우상향 직선형
        ]
        golden_vol_patterns = [
            (x**3).tolist(),
            (1.0 - x*0.5).tolist(),
            (x).tolist()
        ]

    pattern_payload = {
        "updated_at": time.time(),
        "golden_patterns": golden_price_patterns,
        "golden_volume_patterns": golden_vol_patterns
    }
    save_json(PATTERN_FILE, pattern_payload)
    print(f"✅ 골든 패턴 라이브러리 저장 완료 (패턴 수: {len(golden_price_patterns)}개)")
    return golden_price_patterns, golden_vol_patterns

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
        res = requests.get(f"{REMOTE_TRACKER_URL}?t={int(time.time())}", timeout=5)
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

    # 3시간(36개 캔들) 프레임으로 확대 분석
    df_frame = df.iloc[-36:].copy().reset_index(drop=True)
    prices, volumes = df_frame["trade_price"].values, df_frame["candle_acc_trade_volume"].values
    
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
        combined_pattern_sim * weights.get("w_pattern", 0.25) +
        vol_cliff_score * weights.get("w_vol_cliff", 0.20) +
        ma_score * weights.get("w_ma_alignment", 0.20) +
        min(100.0, max(0.0, change_rate * 3.33)) * weights.get("w_daily_momentum", 0.15) +
        (current_price / df["high_price"].max() * 100) * weights.get("w_breakout", 0.20)
    )

    if ticker in recommended_symbols:
        base_score += 5.0

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
        "w_pattern": 0.25, "w_vol_cliff": 0.20, "w_ma_alignment": 0.20,
        "w_vol_surge": 0.15, "w_daily_momentum": 0.15, "w_breakout": 0.05
    })

    # 1. 업비트 전체 KRW 마켓 코인 조회
    res = requests.get("https://api.upbit.com/v1/market/all")
    all_krw = [m for m in res.json() if m["market"].startswith("KRW-")]

    # 2. [신규 추가] 실시간 과거 급등 차트를 학습하여 golden_pattern.json 업데이트
    golden_price_patterns, golden_vol_patterns = extract_and_update_golden_patterns(all_krw)

    recommended_symbols = fetch_remote_recommendations()

    # 3. 추출된 골든 패턴을 기반으로 전체 종목 스코어링 진행
    analyzed_results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
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
        print(f"🎯 학습 및 스코어링 완료 (1위: {analyzed_results[0]['ticker']} - {analyzed_results[0]['score']}점)")
    else:
        print("⚠️ 분석된 결과가 없습니다.")

if __name__ == "__main__":
    main()
