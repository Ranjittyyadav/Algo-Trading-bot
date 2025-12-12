"""
FINAL TRADING BOT - With Optimized Parameters
Ready to use with your Oanda account
"""


import time
import logging
import requests
import pandas as pd

# ✅ MATPLOTLIB - Set backend FIRST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ✅ PYTORCH - All required imports
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ✅ Other imports
from pathlib import Path
from dataclasses import dataclass, field
from threading import Thread
from typing import Dict
import json
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

import requests

BACKEND_URL = "http://0.0.0.0:5001"



@dataclass
class OandaConfig:
    api_key: str
    account_id: str
    base_url: str = "https://api-fxpractice.oanda.com"
    instruments: list = field(default_factory=lambda: ["BTC_USD", "XAU_USD"])
    timeframes: list = field(default_factory=lambda: ["M5", "M15"])
    lookback_candles: int = 100
    units: int = 1
    poll_sec: int = 30
    ema_fast: int = 9
    ema_slow: int = 15
    atr_period: int = 14
    use_atr: bool = True
    # ✅ OPTIMIZED PATTERN PARAMETERS
    max_lower_shadow_factor: float = 0.4   # Strict (was 0.65)
    min_upper_shadow_ratio: float = 2.5    # Strict (was 1.5)
    # ✅ OPTIMIZED SL/TP PARAMETERS
    buy_sl_mult: float = 1.0
    buy_tp_mult: float = 1.5
    sell_sl_mult: float = 1.5
    sell_tp_mult: float = 3.0


CFG = OandaConfig(
    api_key="5c821a3a1a23d3b8a18ffb0b8d10d852-887ec40cabdda2551683deb7e6d329a4",
    account_id="101-011-36217286-002",
)

HEADERS = {
    "Authorization": f"Bearer {CFG.api_key}",
    "Content-Type": "application/json"
}

INSTRUMENT_PRECISION = {
    "BTC_USD": 2,
    "XAU_USD": 3,
}

active_positions: Dict[str, Dict] = {}

def send_signal(signal_type, message, instrument, timeframe):
    requests.post('http://0.0.0.0:5000/api/signal', json={
        "type": signal_type,  # 'bullish', 'bearish', 'candle_complete'
        "message": message,
        "instrument": instrument,
        "timeframe": timeframe
    })


def get_position_key(instrument: str, timeframe: str) -> str:
    """Generate unique key for position tracking"""
    return f"{instrument}_{timeframe}"


def has_active_position(instrument: str, timeframe: str) -> bool:
    """Check if there's an active position"""
    key = get_position_key(instrument, timeframe)
    return key in active_positions


def add_active_position(instrument: str, timeframe: str, trade_type: str, entry_price: float):
    """Record active position"""
    key = get_position_key(instrument, timeframe)
    active_positions[key] = {
        "type": trade_type,
        "entry": entry_price,
        "timestamp": time.time()
    }
    logging.info(f"✅ Position added: {key} | Type: {trade_type} | Entry: {entry_price}")


def remove_active_position(instrument: str, timeframe: str):
    """Remove position when closed"""
    key = get_position_key(instrument, timeframe)
    if key in active_positions:
        del active_positions[key]
        logging.info(f"✅ Position removed: {key}")


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(df) < period + 1:
        return 0.0
    
    df = df.copy()
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift(1))
    df['lc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    
    atr = df['tr'].rolling(window=period).mean().iloc[-1]
    return atr if atr > 0 else 1.0


def calculate_dynamic_sl_tp(entry: float, atr: float, precision: int, trade_type: str = "BUY") -> tuple:
    """
    ✅ OPTIMIZED: Calculate SL/TP with best parameters
    
    BUY:  SL = entry - (1.0 * ATR), TP = entry + (1.5 * ATR)
    SELL: SL = entry + (1.5 * ATR), TP = entry - (3.0 * ATR)
    """
    if trade_type == "BUY":
        sl_distance = round(CFG.buy_sl_mult * atr, precision)
        tp_distance = round(CFG.buy_tp_mult * atr, precision)
        sl_price = round(entry - sl_distance, precision)
        tp_price = round(entry + tp_distance, precision)
    else:  # SELL
        sl_distance = round(CFG.sell_sl_mult * atr, precision)
        tp_distance = round(CFG.sell_tp_mult * atr, precision)
        sl_price = round(entry + sl_distance, precision)
        tp_price = round(entry - tp_distance, precision)
    
    return sl_price, tp_price, sl_distance, tp_distance


def fetch_ohlc(instrument: str, granularity: str, count: int) -> pd.DataFrame:
    """Fetch OHLC data from Oanda with retry logic"""
    url = f"{CFG.base_url}/v3/instruments/{instrument}/candles"
    params = {"count": count, "granularity": granularity, "price": "M"}
    
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=20
            )
            response.raise_for_status()
            break
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout fetching {instrument} {granularity}, attempt {attempt+1}/3")
            if attempt == 2:
                raise
            time.sleep(2)

    candles = response.json().get("candles", [])
    if not candles:
        return pd.DataFrame()
    
    data = [{
        "time": candle["time"],
        "open": float(candle["mid"]["o"]),
        "high": float(candle["mid"]["h"]),
        "low": float(candle["mid"]["l"]),
        "close": float(candle["mid"]["c"]),
        "volume": candle["volume"],
        "complete": candle["complete"]
    } for candle in candles]
    
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%S.%f000Z", errors='coerce')
    df.set_index("time", inplace=True)
    return df


def compute_emas(df: pd.DataFrame, fast_period: int, slow_period: int) -> pd.DataFrame:
    """Calculate EMA indicators"""
    df = df.copy()
    df["EMA_fast"] = df["close"].ewm(span=fast_period, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=slow_period, adjust=False).mean()
    df["EMA_lower"] = df[["EMA_fast", "EMA_slow"]].min(axis=1)
    df["EMA_upper"] = df[["EMA_fast", "EMA_slow"]].max(axis=1)
    return df


# ✅ OPTIMIZED PATTERN DETECTION
def is_bearish_hammer(row) -> bool:
    """
    ✅ OPTIMIZED: Very Strict bearish hammer detection
    Parameters from pattern optimizer
    """
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    if c >= o:
        return False
    body = o - c
    if body < 0.0001:
        return False
    upper_shadow = h - o
    lower_shadow = c - l
    
    # ✅ STRICT PARAMETERS
    if lower_shadow > CFG.max_lower_shadow_factor * body:  # 0.4
        return False
    if upper_shadow < CFG.min_upper_shadow_ratio * body:  # 2.5
        return False
    return True


def is_bullish_hammer(row) -> bool:
    """
    ✅ OPTIMIZED: Very Strict bullish hammer detection
    Parameters from pattern optimizer
    """
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    if c <= o:
        return False
    body = c - o
    if body < 0.0001:
        return False
    upper_shadow = h - c
    lower_shadow = o - l
    
    # ✅ STRICT PARAMETERS
    if upper_shadow > CFG.max_lower_shadow_factor * body:  # 0.4
        return False
    if lower_shadow < CFG.min_upper_shadow_ratio * body:  # 2.5
        return False
    return True


def is_uptrend(row) -> bool:
    """Check if price is in uptrend (close > EMA_lower)"""
    return row["close"] > row["EMA_lower"]


def is_downtrend(row) -> bool:
    """Check if price is in downtrend (close < EMA_upper)"""
    return row["close"] < row["EMA_upper"]


def place_market_buy_with_sl_tp(instrument: str, units: int, sl_price: float, tp_price: float, 
                                entry: float, atr: float, timeframe: str):
    """Place a BUY market order with SL and TP"""
    precision = INSTRUMENT_PRECISION.get(instrument, 2)
    
    # Round for precision
    sl_price = round(sl_price, precision)
    tp_price = round(tp_price, precision)
    
    url = f"{CFG.base_url}/v3/accounts/{CFG.account_id}/orders"
    order = {
        "order": {
            "units": str(units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{sl_price:.{precision}f}"},
            "takeProfitOnFill": {"price": f"{tp_price:.{precision}f}"}
        }
    }
    
    response = requests.post(url, headers=HEADERS, json=order, timeout=10)
    if response.status_code == 201:
        logging.info(f"✅ BUY order placed for {instrument}: Entry={entry:.{precision}f}, SL={sl_price:.{precision}f}, TP={tp_price:.{precision}f}, ATR={atr:.{precision}f}")
        add_active_position(instrument, timeframe, "BUY", entry)
                # ✅ SEND TO DASHBOARD

        return True
    else:
        logging.error(f"❌ Failed to place buy order for {instrument}: {response.status_code} {response.text}")
        return False


def place_market_sell_with_sl_tp(instrument: str, units: int, sl_price: float, tp_price: float, 
                                 entry: float, atr: float, timeframe: str):
    """Place a SELL market order with SL and TP"""
    precision = INSTRUMENT_PRECISION.get(instrument, 2)
    
    # Round for precision
    sl_price = round(sl_price, precision)
    tp_price = round(tp_price, precision)
    
    url = f"{CFG.base_url}/v3/accounts/{CFG.account_id}/orders"
    order = {
        "order": {
            "units": str(-units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{sl_price:.{precision}f}"},
            "takeProfitOnFill": {"price": f"{tp_price:.{precision}f}"}
        }
    }
    
    response = requests.post(url, headers=HEADERS, json=order, timeout=10)
    if response.status_code == 201:
        logging.info(f"✅ SELL order placed for {instrument}: Entry={entry:.{precision}f}, SL={sl_price:.{precision}f}, TP={tp_price:.{precision}f}, ATR={atr:.{precision}f}")
        add_active_position(instrument, timeframe, "SELL", entry)
                # ✅ SEND TO DASHBOARD

        return True
    else:
        logging.error(f"❌ Failed to place sell order for {instrument}: {response.status_code} {response.text}")
        return False

         

def run_for_instrument_and_timeframe(instrument: str, timeframe: str):
    """Main trading loop for one instrument+timeframe combination"""
    logging.info(f"🚀 Starting {instrument} bot for {timeframe}")
    last_time = None
    precision = INSTRUMENT_PRECISION.get(instrument, 2)
    
    while True:
        try:
            # Fetch OHLC data
            df = fetch_ohlc(instrument, timeframe, CFG.lookback_candles)
            if df.empty or len(df) < CFG.ema_slow + 1:
                logging.warning(f"⚠️ Not enough data for {instrument} {timeframe}")
                time.sleep(CFG.poll_sec)
                continue

            # Filter only completed candles
            df = df[df["complete"] == True].copy()
            if df.empty:
                logging.info(f"⏳ No completed candle yet for {instrument} {timeframe}")
                time.sleep(CFG.poll_sec)
                continue

            # Calculate indicators
            df = compute_emas(df, CFG.ema_fast, CFG.ema_slow)
            last = df.iloc[-1]
            time_stamp = df.index[-1]

            # Skip if same candle
            if time_stamp == last_time:
                time.sleep(CFG.poll_sec)
                continue
            last_time = time_stamp

            # Calculate ATR
            atr = calculate_atr(df, CFG.atr_period) if CFG.use_atr else 1.0

            logging.info(f"📊 {instrument} {timeframe} @ {time_stamp} | "
                         f"O:{last['open']:.2f} H:{last['high']:.2f} L:{last['low']:.2f} C:{last['close']:.2f} | "
                         f"ATR:{atr:.2f}")
            
            send_signal(
                "candle_complete",
                f"⏳ {timeframe} Candle Completed | {instrument} | O:{last['open']:.2f} H:{last['high']:.2f} L:{last['low']:.2f} C:{last['close']:.2f}",
                instrument,
                timeframe
            )

            # Skip if position exists
            if has_active_position(instrument, timeframe):
                logging.info(f"⏸️ Already in position for {instrument} {timeframe}")
                time.sleep(CFG.poll_sec)
                continue

                        # ✅ NEW: Send position updates for live P&L
            current_price = last["close"]
            key = get_position_key(instrument, timeframe)
            if key in active_positions:
                pos = active_positions[key]
                entry = pos["entry"]
                if pos["type"] == "BUY":
                    pnl = current_price - entry
                else:  # SELL
                    pnl = entry - current_price
                send_position_update(instrument, timeframe, current_price, pnl)


            # Save chart for model prediction
            chart_path = f"charts/{instrument}_{timeframe}_{int(time.time())}.png"
            save_candlestick_chart(df.tail(40), chart_path)  # Last 40 candles for chart

            # Get model prediction
            model_result = predict_hammer(chart_path)

            is_model_hammer = model_result["is_hammer"] and model_result["confidence"] >= 0.7

            bearish_confirmed = (
                is_model_hammer
                and is_bearish_hammer(last)
                and is_downtrend(last)
            )

            bullish_confirmed = (
                is_model_hammer
                and is_bullish_hammer(last)
                and is_uptrend(last)
            )

            if bearish_confirmed:
                entry = last["close"]
                send_signal(
                    "bearish",
                    f"🔴 Bearish Hammer (MODEL+LOGIC) | {instrument} {timeframe} | Entry: ${entry:.2f}",
                    instrument,
                    timeframe,
                )
                sl_price, tp_price, sl_dist, tp_dist = calculate_dynamic_sl_tp(entry, atr, precision, "SELL")
                send_new_trade("SELL", instrument, timeframe, entry, sl_price, tp_price)
                
                place_market_sell_with_sl_tp(
                    instrument, CFG.units,
                    sl_price=sl_price, tp_price=tp_price,
                    entry=entry, atr=atr, timeframe=timeframe,
                )

            elif bullish_confirmed:
                entry = last["close"]
                send_signal(
                    "bullish",
                    f"🟢 Bullish Hammer (MODEL+LOGIC) | {instrument} {timeframe} | Entry: ${entry:.2f}",
                    instrument,
                    timeframe,
                )
                sl_price, tp_price, sl_dist, tp_dist = calculate_dynamic_sl_tp(entry, atr, precision, "BUY")
                send_new_trade("BUY", instrument, timeframe, entry, sl_price, tp_price)
                
                place_market_buy_with_sl_tp(
                    instrument, CFG.units,
                    sl_price=sl_price, tp_price=tp_price,
                    entry=entry, atr=atr, timeframe=timeframe,
                )



            time.sleep(CFG.poll_sec)

        except Exception as e:
            logging.exception(f"❌ Error in {instrument} {timeframe}: {e}")
            time.sleep(CFG.poll_sec)

def run_all():
    """Start bot for all instruments and timeframes"""
    threads = []
    for instrument in CFG.instruments:
        for timeframe in CFG.timeframes:
            t = Thread(target=run_for_instrument_and_timeframe, args=(instrument, timeframe), daemon=True)
            threads.append(t)
            t.start()
    
    for t in threads:
        t.join()


def load_historical_trades():
    """Load closed trades from Oanda and send to dashboard"""
    try:
        logger.info("🔄 Starting to load historical trades from Oanda...")
        
        url = f"{CFG.base_url}/v3/accounts/{CFG.account_id}/trades"
        params = {"state": "CLOSED", "count": 50}
        
        logger.info(f"📡 Requesting Oanda API: {url}")
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
        logger.info(f"📨 Oanda response status: {resp.status_code}")
        
        if resp.status_code != 200:
            logger.warning(f"⚠️ Failed to fetch trades from Oanda: {resp.status_code}")
            logger.warning(f"⚠️ Response: {resp.text[:300]}")
            return
        
        resp_json = resp.json()

        trades_data = resp_json.get("trades", [])
        logger.info(f"📚 Found {len(trades_data)} closed trades in Oanda")
        
        if len(trades_data) == 0:
            logger.warning("⚠️ No closed trades found in Oanda account")
            return
        
        historical_trades = []
        winning_count = 0
        losing_count = 0
        total_pnl = 0
        skipped_count = 0
        
        for idx, trade in enumerate(trades_data):
            try:
                # ✅ CORRECT FIELD NAMES for Oanda
                close_time = trade.get("closeTime")
                
                if not close_time:
                    logger.debug(f"⚠️ Trade {idx} has no closeTime, skipping")
                    skipped_count += 1
                    continue
                
                # ✅ CORRECT: Use 'price' for entry, 'averageClosePrice' for exit
                entry_price = float(trade.get("price", 0))
                exit_price = float(trade.get("averageClosePrice", 0))
                pnl = float(trade.get("realizedPL", 0))
                
                # Extract timestamp (handle both formats)
                try:
                    if "T" in close_time:
                        # Extract just the date and time part before the dot
                        timestamp_parts = close_time.split(".")[0]  # Remove milliseconds
                        timestamp = timestamp_parts.replace("T", " ")  # Replace T with space
                    else:
                        timestamp = close_time[:19] if len(close_time) > 19 else close_time  # First 19 characters
                except:
                    timestamp = "N/A"
                
                # Track stats
                if pnl > 0:
                    winning_count += 1
                else:
                    losing_count += 1
                total_pnl += pnl
                
                trade_obj = {
                    "type": "BUY" if float(trade.get("initialUnits", 0)) > 0 else "SELL",
                    "instrument": trade.get("instrument"),
                    "timeframe": "M5",
                    "entry": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "status": "CLOSED",
                    "timestamp": timestamp,
                }
                historical_trades.append(trade_obj)
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing trade {idx}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"📊 Trade processing summary:")
        logger.info(f"   - Total found: {len(trades_data)}")
        logger.info(f"   - Processed: {len(historical_trades)}")
        logger.info(f"   - Skipped: {skipped_count}")
        
        if not historical_trades:
            logger.warning("⚠️ No valid historical trades to send")
            if trades_data:
                logger.warning(f"⚠️ First trade structure: {trades_data}")
            return
        
        logger.info(f"✅ Processed {len(historical_trades)} trades:")
        logger.info(f"   - Winning: {winning_count}")
        logger.info(f"   - Losing: {losing_count}")
        logger.info(f"   - Total P&L: ${total_pnl:.2f}")

        summary_url = f"{CFG.base_url}/v3/accounts/{CFG.account_id}/summary"
        logger.info(f"📡 Requesting Oanda account summary: {summary_url}")
        summary_resp = requests.get(summary_url, headers=HEADERS, timeout=15)
        logger.info(f"📨 Oanda summary response status: {summary_resp.status_code}")

        current_balance = 0.0
        unrealized_pl = 0.0
        realized_pl = 0.0
        if summary_resp.status_code != 200:
            logger.warning(f"⚠️ Failed to fetch account summary: {summary_resp.status_code}")
            logger.warning(f"⚠️ Summary response: {summary_resp.text[:300]}")
        else:
            account = summary_resp.json().get("account", {})
            # Strings → float
            current_balance = float(account.get("balance", 0.0))
            unrealized_pl = float(account.get("unrealizedPL", 0.0))    # open-trade P&L
            realized_pl = float(account.get("pl", 10.0))  # realized P&L
            logger.info(
                f"💰 Account summary: balance={current_balance}, "
                f"unrealizedPL={unrealized_pl}, realizedPL={realized_pl}"
            )
        
        # Send to backend
        logger.info(f"📨 Sending {len(historical_trades)} trades to backend...")
        r2 = requests.post(
            f"{BACKEND_URL}/api/load-history",
            json={
                "trades": historical_trades,
                "account_balance": current_balance,   # <- new
                "floating_pnl": realized_pl,              # <- open positions P&L
                "realized_pl": realized_pl                  # <- closed trades P&L
            },
            timeout=10,
        )


        logger.info(f"📨 Backend response status: {r2.status_code}")
        logger.info(f"📨 Backend response: {r2.text[:200]}")
        
        if r2.status_code == 200:
            logger.info("✅ Historical trades successfully sent to dashboard!")
        else:
            logger.error(f"❌ Backend failed to process trades: {r2.text}")
    
    except Exception as e:
        logger.exception(f"❌ Error in load_historical_trades: {e}")


def send_signal(signal_type, message, instrument, timeframe):
    """Send trading signal to dashboard"""
    try:
        requests.post(
            f"{BACKEND_URL}/api/signal",
            json={
                "type": signal_type,
                "message": message,
                "instrument": instrument,
                "timeframe": timeframe
            },
            timeout=2
        )
        logger.info(f"📊 Signal sent: {signal_type} | {instrument} | {timeframe}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to send signal to backend: {e}")


def send_new_trade(trade_type, instrument, timeframe, entry, sl, tp):
    """Send new trade to dashboard"""
    try:
        requests.post(
            f"{BACKEND_URL}/api/trade",
            json={
                "type": trade_type,
                "instrument": instrument,
                "timeframe": timeframe,
                "entry": entry,
                "sl": sl,
                "tp": tp
            },
            timeout=2
        )
        logger.info(f"✅ Trade sent to dashboard: {trade_type} {instrument}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to send trade to backend: {e}")


def send_trade_close(instrument, timeframe, exit_price, pnl):
    """Send trade close event to dashboard"""
    try:
        requests.post(
            f"{BACKEND_URL}/api/trade-close",
            json={
                "instrument": instrument,
                "timeframe": timeframe,
                "exit_price": exit_price,
                "pnl": pnl
            },
            timeout=2
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to send trade close to backend: {e}")


# def save_candlestick_chart(df: pd.DataFrame, filename: str):
#     """Save candlestick chart as PNG"""
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Plot candles
#     for i, row in df.iterrows():
#         color = 'green' if row['close'] >= row['open'] else 'red'
#         ax.plot([i, i], [row['low'], row['high']], color=color)
#         ax.add_patch(plt.Rectangle((i-0.3, row['open']), 0.6, row['close']-row['open'], 
#                                 facecolor=color, edgecolor=color))
    
#     # Formatting
#     ax.set_xlim(-0.5, len(df)-0.5)
#     ax.set_ylim(df['low'].min()*0.99, df['high'].max()*1.01)
#     ax.set_title(f"Candlestick Chart")
#     ax.grid(True, alpha=0.3)
    
#     # Save
#     Path("charts").mkdir(exist_ok=True)
#     plt.savefig(filename, dpi=100, bbox_inches='tight')
#     plt.close()
def save_candlestick_chart(df: pd.DataFrame, filename: str) -> bool:
    try:
        df = df.reset_index(drop=True).copy()

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)  # smaller square
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        for i, row in df.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1.5)

            rect = mpatches.Rectangle(
                xy=(i - 0.3, min(row['open'], row['close'])),
                width=0.6,
                height=abs(row['close'] - row['open']),
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
            )
            ax.add_patch(rect)

        ax.set_xlim(-0.5, len(df) - 0.5)
        lows = df['low'].min()
        highs = df['high'].max()
        margin = (highs - lows) * 0.05 if highs > lows else 1
        ax.set_ylim(lows - margin, highs + margin)

        ax.axis('off')  # no axes, pure chart

        Path("charts").mkdir(exist_ok=True)
        plt.tight_layout(pad=0.1)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        logger.error(f"❌ Chart save error: {e}")
        return False


def predict_hammer(image_path: str) -> dict:
    """Predict hammer pattern from chart image"""
    try:
        checkpoint = torch.load("models/cv_hammer_multi.pth", map_location='cpu')
        
        if not checkpoint or not isinstance(checkpoint, dict):
            logger.error("❌ Invalid checkpoint format")
            return {"class": "none", "confidence": 0.0, "is_hammer": False}
        
        class_names = checkpoint.get('train_classes', ['bearish', 'bullish', 'none'])
        num_classes = len(class_names)
        
        # Create model
        model = models.resnet18(weights=None)
        num_f = model.fc.in_features
        model.fc = nn.Linear(num_f, num_classes)
        
        # Load weights
        if 'model_state_dict' in checkpoint and checkpoint['model_state_dict']:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.error("❌ No model_state_dict in checkpoint")
            return {"class": "none", "confidence": 0.0, "is_hammer": False}
        
        model.eval()
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load image
        if not Path(image_path).exists():
            logger.warning(f"⚠️ Image not found: {image_path}")
            return {"class": "none", "confidence": 0.0, "is_hammer": False}
        
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        
        # Predict - FIXED
        with torch.no_grad():
            outputs = model(img_t)  # Shape: (1, 3)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)  # Shape: (3,)
            
            # ✅ FIXED: Convert tensor scalars to Python values
            confidence, predicted = torch.max(probs, 0)
            confidence_value = confidence.item()  # ← Convert tensor to float
            predicted_idx = predicted.item()      # ← Convert tensor to int
        
        class_name = class_names[predicted_idx]

        logger.info(f"✅ Model prediction: {class_name} ({confidence_value:.1%})")


        
        return {
            "class": class_name,
            "confidence": confidence_value,
            "is_hammer": class_name != "none"
        }
        
    except FileNotFoundError as e:
        logger.error(f"❌ Model file error: {e}")
        return {"class": "none", "confidence": 0.0, "is_hammer": False}
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {"class": "none", "confidence": 0.0, "is_hammer": False}


def send_position_update(instrument, timeframe, current_price, pnl):
    try:
        requests.post(
            f"{BACKEND_URL}/api/position-update",
            json={
                "instrument": instrument,
                "timeframe": timeframe,
                "current_price": current_price,
                "pnl": pnl,
            },
            timeout=2,
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to send position update to backend: {e}")



if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🚀 TRADING BOT STARTING UP...")
    logger.info("=" * 80)
    
    # Load history FIRST, before print statements
    load_historical_trades()
    
    logger.info("\n" + "="*80)
    logger.info("🤖 TRADING BOT - OPTIMIZED PARAMETERS")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"   Max Lower Shadow: {CFG.max_lower_shadow_factor}")
    logger.info(f"   Min Upper Shadow: {CFG.min_upper_shadow_ratio}")
    logger.info("="*80 + "\n")
    
    run_all()
