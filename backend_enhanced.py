"""
FIXED backend_enhanced.py - Clean version without duplicates
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import time
import requests
import logging
from threading import Thread
from datetime import datetime
from collections import deque
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Create SocketIO with threading mode (no eventlet needed)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Bot state
bot_state = {
    "status": "LIVE",
    "account_balance": 10000,
    "floating_pnl": 0,
    "open_positions": 0,
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "win_rate": 0,
    "profit_factor": 0,
    "active_positions": [],
    "recent_trades": deque(maxlen=100),
    "trading_signals": deque(maxlen=50),
    "config": {}
}

def deque_to_list(obj):
    """Convert deque objects to lists for JSON serialization"""
    if isinstance(obj, deque):
        return list(obj)
    raise TypeError


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/signal', methods=['POST'])
def receive_signal():
    """Receive real-time trading signals from bot"""
    try:
        data = request.get_json()
        signal_type = data.get('type')
        message = data.get('message')
        instrument = data.get('instrument')
        timeframe = data.get('timeframe')
        
        signal_event = {
            "type": signal_type,
            "message": message,
            "instrument": instrument,
            "timeframe": timeframe,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "timestamp_ms": time.time()
        }
        
        bot_state["trading_signals"].append(signal_event)
        logger.info(f"📊 Signal received: {signal_type} | {instrument} | {timeframe}")
        
        # Emit to all connected clients
        socketio.emit('trading_signal', signal_event)
        socketio.emit('bot_update', {
            "status": bot_state["status"],
            "account_balance": bot_state["account_balance"],
            "floating_pnl": bot_state["floating_pnl"],
            "open_positions": bot_state["open_positions"],
            "total_trades": bot_state["total_trades"],
            "winning_trades": bot_state["winning_trades"],
            "losing_trades": bot_state["losing_trades"],
            "win_rate": bot_state["win_rate"],
            "active_positions": bot_state["active_positions"],
            "recent_trades": list(bot_state["recent_trades"]),
            "trading_signals": list(bot_state["trading_signals"]),
        })

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"❌ Error processing signal: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/trade', methods=['POST'])
def receive_trade():
    """Receive new trade execution from bot"""
    try:
        data = request.get_json()
        trade = {
            "id": f"{data.get('instrument')}_{data.get('timeframe')}_{int(time.time())}",
            "type": data.get('type'),
            "instrument": data.get('instrument'),
            "timeframe": data.get('timeframe'),
            "entry": data.get('entry'),
            "sl": data.get('sl'),
            "tp": data.get('tp'),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "pnl": 0,
            "status": "OPEN"
        }
        
        bot_state["active_positions"].append(trade)
        bot_state["open_positions"] = len(bot_state["active_positions"])
        bot_state["total_trades"] += 1
        bot_state["recent_trades"].append(trade)
        
        logger.info(f"✅ Trade added: {trade['type']} {trade['instrument']} @ {trade['entry']}")
        
        socketio.emit('bot_update', {
            "status": bot_state["status"],
            "account_balance": bot_state["account_balance"],
            "floating_pnl": bot_state["floating_pnl"],
            "open_positions": bot_state["open_positions"],
            "total_trades": bot_state["total_trades"],
            "winning_trades": bot_state["winning_trades"],
            "losing_trades": bot_state["losing_trades"],
            "win_rate": bot_state["win_rate"],
            "active_positions": bot_state["active_positions"],
            "recent_trades": list(bot_state["recent_trades"]),
            "trading_signals": list(bot_state["trading_signals"]),
        })

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"❌ Error adding trade: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/position-update', methods=['POST'])
def update_position():
    """Update P&L for open position"""
    try:
        data = request.get_json()
        instrument = data.get('instrument')
        timeframe = data.get('timeframe')
        pnl = data.get('pnl')
        current_price = data.get('current_price')
        
        for pos in bot_state["active_positions"]:
            if pos["instrument"] == instrument and pos["timeframe"] == timeframe:
                pos["pnl"] = pnl
                pos["current_price"] = current_price
                break
        
        bot_state["floating_pnl"] = sum(p.get("pnl", 0) for p in bot_state["active_positions"])
        
        socketio.emit('bot_update', {
            "status": bot_state["status"],
            "account_balance": bot_state["account_balance"],
            "floating_pnl": bot_state["floating_pnl"],
            "open_positions": bot_state["open_positions"],
            "total_trades": bot_state["total_trades"],
            "winning_trades": bot_state["winning_trades"],
            "losing_trades": bot_state["losing_trades"],
            "win_rate": bot_state["win_rate"],
            "active_positions": bot_state["active_positions"],
            "recent_trades": list(bot_state["recent_trades"]),
            "trading_signals": list(bot_state["trading_signals"]),
        })

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"❌ Error updating position: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/trade-close', methods=['POST'])
def close_trade():
    """Close a trade and calculate P&L"""
    try:
        data = request.get_json()
        instrument = data.get('instrument')
        timeframe = data.get('timeframe')
        exit_price = data.get('exit_price')
        pnl = data.get('pnl')
        
        for i, pos in enumerate(bot_state["active_positions"]):
            if pos["instrument"] == instrument and pos["timeframe"] == timeframe:
                pos["exit_price"] = exit_price
                pos["pnl"] = pnl
                pos["status"] = "CLOSED"
                pos["close_time"] = datetime.now().strftime("%H:%M:%S")
                
                if pnl > 0:
                    bot_state["winning_trades"] += 1
                else:
                    bot_state["losing_trades"] += 1
                
                bot_state["recent_trades"].append(pos)
                bot_state["active_positions"].pop(i)
                break
        
        bot_state["open_positions"] = len(bot_state["active_positions"])
        bot_state["floating_pnl"] = sum(p.get("pnl", 0) for p in bot_state["active_positions"])
        
        total = bot_state["winning_trades"] + bot_state["losing_trades"]
        if total > 0:
            bot_state["win_rate"] = round((bot_state["winning_trades"] / total) * 100, 1)
        
        logger.info(f"🏁 Trade closed: {pnl} | Win rate: {bot_state['win_rate']}%")
        
        socketio.emit('bot_update', {
            "status": bot_state["status"],
            "account_balance": bot_state["account_balance"],
            "floating_pnl": bot_state["floating_pnl"],
            "open_positions": bot_state["open_positions"],
            "total_trades": bot_state["total_trades"],
            "winning_trades": bot_state["winning_trades"],
            "losing_trades": bot_state["losing_trades"],
            "win_rate": bot_state["win_rate"],
            "active_positions": bot_state["active_positions"],
            "recent_trades": list(bot_state["recent_trades"]),
            "trading_signals": list(bot_state["trading_signals"]),
        })

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"❌ Error closing trade: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/load-history', methods=['POST'])
def load_history():
    """Load historical trades from Oanda"""
    try:
        data = request.get_json()
        historical_trades = data.get('trades', [])
        
        # Get account balance, floating P&L, and realized P&L from the data if provided
        account_balance = data.get('account_balance')
        floating_pnl = data.get('floating_pnl', 0)
        realized_pl = data.get('realized_pl', 0)

        bot_state["recent_trades"].clear()
        bot_state["active_positions"] = []
        bot_state["total_trades"] = 0
        bot_state["winning_trades"] = 0
        bot_state["losing_trades"] = 0

        total_closed_pnl = 0.0

        for trade in historical_trades:
            bot_state["recent_trades"].append(trade)
            bot_state["total_trades"] += 1

            if trade.get("status") == "CLOSED":
                pnl = float(trade.get("pnl", 0) or 0)
                total_closed_pnl += pnl
                if pnl > 0:
                    bot_state["winning_trades"] += 1
                else:
                    bot_state["losing_trades"] += 1
            elif trade.get("status") == "OPEN":
                bot_state["active_positions"].append(trade)

        bot_state["open_positions"] = len(bot_state["active_positions"])

        total = bot_state["winning_trades"] + bot_state["losing_trades"]
        if total > 0:
            bot_state["win_rate"] = round((bot_state["winning_trades"] / total) * 100, 1)

        # Use account balance from Oanda if provided, otherwise derive from history
        if account_balance is not None:
            bot_state["account_balance"] = account_balance
            bot_state["floating_pnl"] = floating_pnl if floating_pnl is not None else 0
        else:
            # Fallback to deriving balance from history
            starting_balance = 10000
            bot_state["account_balance"] = starting_balance + total_closed_pnl
            bot_state["floating_pnl"] = sum(
                float(p.get("pnl", 0) or 0) for p in bot_state["active_positions"]
            )

        logger.info(f"Loaded {len(historical_trades)} trades. "
                    f"Account Balance={bot_state['account_balance']}, "
                    f"Floating P&L={bot_state['floating_pnl']}, "
                    f"Realized P&L received={realized_pl}")

        socketio.emit('bot_update', {
            "status": bot_state["status"],
            "account_balance": bot_state["account_balance"],
            "floating_pnl": bot_state["floating_pnl"],
            "open_positions": bot_state["open_positions"],
            "total_trades": bot_state["total_trades"],
            "winning_trades": bot_state["winning_trades"],
            "losing_trades": bot_state["losing_trades"],
            "win_rate": bot_state["win_rate"],
            "active_positions": bot_state["active_positions"],
            "recent_trades": list(bot_state["recent_trades"]),
            "trading_signals": list(bot_state["trading_signals"]),
        })

        return jsonify({
            "status": "success",
            "trades_loaded": len(historical_trades),
            "total_trades": bot_state["total_trades"],
            "win_rate": bot_state["win_rate"],
            "account_balance": bot_state["account_balance"],
            "floating_pnl": bot_state["floating_pnl"],
        }), 200
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/')
def index():
    """Serve dashboard"""
    return render_template('dashboard_enhanced.html')


@app.route('/api/status')
def get_status():
    # bot_state["account_balance"] = 12345
    # bot_state["floating_pnl"] = 99.99
    return jsonify({
        "status": "success",
        "data": {
            "status": bot_state["status"],
            "total_trades": bot_state["total_trades"],
            "winning_trades": bot_state["winning_trades"],
            "losing_trades": bot_state["losing_trades"],
            "win_rate": bot_state["win_rate"],
            "open_positions": bot_state["open_positions"],
            "floating_pnl": bot_state["floating_pnl"],
            "account_balance": bot_state["account_balance"],
            "recent_trades": list(bot_state["recent_trades"]),
            "trading_signals": list(bot_state["trading_signals"])
        }
    }), 200


@app.route('/charts/<path:filename>')
def serve_chart(filename):
    """Serve chart images from charts folder"""
    charts_dir = Path('charts')
    if not charts_dir.exists():
        charts_dir.mkdir(parents=True, exist_ok=True)
    return send_from_directory('charts', filename)


@app.route('/api/chart-update', methods=['POST'])
def chart_update():
    """Receive chart update notification from bot"""
    try:
        data = request.get_json()
        instrument = data.get('instrument')
        timeframe = data.get('timeframe')
        chart_path = data.get('chart_path')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.info(f"📊 Chart update: {instrument} {timeframe} - {chart_path}")
        
        socketio.emit('bot_update', {
            "chart_update": {
                "instrument": instrument,
                "timeframe": timeframe,
                "chart_path": chart_path.split('\\')[-1] if '\\' in chart_path else chart_path,
                "timestamp": timestamp
            }
        }, broadcast=True)
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"❌ Error processing chart update: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# SOCKETIO EVENTS
# ============================================================================

connected_clients = 0


@socketio.on('connect')
def handle_connect():
    global connected_clients
    connected_clients += 1
    logger.info(f"✅ Dashboard connected! (Total: {connected_clients})")

    socketio.emit('initial_data', {
        "status": bot_state["status"],
        "account_balance": bot_state["account_balance"],
        "floating_pnl": bot_state["floating_pnl"],
        "open_positions": bot_state["open_positions"],
        "total_trades": bot_state["total_trades"],
        "winning_trades": bot_state["winning_trades"],
        "losing_trades": bot_state["losing_trades"],
        "win_rate": bot_state["win_rate"],
        "active_positions": bot_state["active_positions"],
        "recent_trades": list(bot_state["recent_trades"]),
        "trading_signals": list(bot_state["trading_signals"]),
    })


@socketio.on('disconnect')
def handle_disconnect():
    global connected_clients
    connected_clients -= 1
    logger.info(f"❌ Dashboard disconnected! (Total: {connected_clients})")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("🚀 Backend starting on http://0.0.0.0:5001...")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)         
