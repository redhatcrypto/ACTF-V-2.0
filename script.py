import numpy as np
import ccxt
import pandas as pd
import json
import logging
import time
import os
from datetime import datetime
import sys
import io
import csv
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving and loading the model

load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Initialize exchange (Binance Futures Testnet)
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
testnet_url = os.getenv("TESTNET_URL")

if not api_key or not api_secret:
    logging.error("API_KEY or API_SECRET not found in .env file. Please set them and try again.")
    sys.exit(1)

if not testnet_url:
    logging.warning("TESTNET_URL not found in .env file. Using default Binance Futures Testnet URL.")
    testnet_url = "https://testnet.binancefuture.com"

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'urls': {'api': testnet_url}
})
exchange.set_sandbox_mode(True)

# Function to get server time
def get_server_time():
    try:
        server_time = exchange.fetch_time()
        return server_time
    except Exception as e:
        logging.error(f"Failed to fetch server time: {e}")
        return None

# Safe API call with retry mechanism
def safe_api_call(api_call, *args, **kwargs):
    for attempt in range(3):
        try:
            return api_call(*args, **kwargs)
        except ccxt.NetworkError as e:
            logging.error(f"Network error: {e}. Retrying...")
            time.sleep(1)
        except ccxt.ExchangeError as e:
            if e.args[0]['code'] == -1021:
                logging.error("Timestamp error. Adjusting time...")
                time.sleep(1)
            else:
                raise
    raise Exception("API call failed after retries.")

# Load state from trade_state.json
def load_state():
    try:
        with open("trade_state.json", "r") as f:
            content = f.read().strip()
            if not content:
                logging.warning("trade_state.json is empty, returning default state")
                return get_default_state()

            state = json.loads(content)
            default_state = get_default_state()
            for key, value in default_state.items():
                if key not in state:
                    state[key] = value
            
            return state

    except FileNotFoundError:
        logging.info("trade_state.json not found, returning default state")
        return get_default_state()

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse trade_state.json: {e}, returning default state")
        return get_default_state()

    except Exception as e:
        logging.error(f"Unexpected error while loading trade_state.json: {e}, returning default state")
        return get_default_state()

# Helper function to define default state
def get_default_state():
    return {
        "in_position": False,
        "entry_price": 0,
        "stop_loss": 0,
        "take_profit": 0,
        "quantity": 0,
        "partial_quantity": 0,
        "side": None,
        "cumulative_fees": 0.0,
        "current_leverage": config["leverage"],
        "last_price": 0,
        "positions": {},
        "skipped_trades": 0,
        "last_price_history": [],
        "dynamic_volume_ratio": config["min_volume_ratio"],
        "dynamic_rsi_buy": config["rsi_buy"],
        "dynamic_rsi_sell": config["rsi_sell"],
        "dynamic_stoch_rsi_buy": config["stoch_rsi_buy"],
        "dynamic_stoch_rsi_sell": config["stoch_rsi_sell"],
        "trade_cooldown": 0,
        "last_trade_time": 0,
        "trade_skipped": False  # New: Track if a trade was skipped
    }

# Save state to trade_state.json
def save_state(state):
    try:
        with open("trade_state.json", "w") as f:
            json.dump(state, f, indent=4)
        logging.info("Saved state to trade_state.json")
    except Exception as e:
        logging.error(f"Failed to save state to trade_state.json: {e}")

# Fetch OHLCV data
def fetch_data(timeframe, limit=None):
    try:
        limit = limit or config["limit"]
        data = safe_api_call(exchange.fetch_ohlcv, config["symbol"], timeframe, limit=limit)
        if not data:
            logging.error(f"fetch_data: No data returned for {timeframe}")
            return None
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        logging.error(f"fetch_data error: {e}")
        return None

# Calculate technical indicators
def calculate_indicators(df):
    if df is None or df.empty:
        logging.error("calculate_indicators: DataFrame is None or empty")
        return None
    try:
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        rsi = df["rsi"]
        rsi_min = rsi.rolling(window=14).min()
        rsi_max = rsi.rolling(window=14).max()
        df["stoch_rsi"] = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df["bb_ma"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_ma"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_ma"] - (df["bb_std"] * 2)

        # ATR
        df["tr"] = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["atr"] = df["tr"].rolling(window=config["atr_period"]).mean()

        return df
    except Exception as e:
        logging.error(f"calculate_indicators error: {e}")
        return None

# Check if volume is rising
def is_volume_rising(df):
    if len(df) < 2:
        return False
    return df["volume"].iloc[-1] > df["volume"].iloc[-2]

# Fetch current ticker price with fallback
def fetch_ticker_with_fallback(state):
    try:
        ticker = safe_api_call(exchange.fetch_ticker, config["symbol"])
        price = float(ticker["last"])
        state["last_price"] = price
        return price
    except Exception as e:
        logging.error(f"fetch_ticker_with_fallback error: {e}")
        return state.get("last_price", None)

# Check open positions
def check_positions():
    try:
        positions = safe_api_call(exchange.fetch_positions, [config["symbol"]])
        position_dict = {}
        for pos in positions:
            symbol = pos["symbol"]
            contracts = float(pos["contracts"]) if pos["contracts"] else 0
            if contracts > 0:
                position_dict[symbol] = pos
        logging.info(f"Positions fetched: {position_dict}")
        return position_dict
    except Exception as e:
        logging.error(f"check_positions error: {e}")
        return {}

# Adjust leverage dynamically
def adjust_leverage(state):
    try:
        df = fetch_data(config["timeframe"], limit=50)
        if df is None:
            raise ValueError("Failed to fetch data")
        df = calculate_indicators(df)
        if df is None:
            raise ValueError("Failed to calculate indicators")
        
        # Calculate volatility using ATR
        latest_atr = float(df.iloc[-1]["atr"])
        avg_price = float(df["close"].mean())
        volatility = latest_atr / avg_price  # Normalized volatility
        
        # Adjust leverage based on volatility
        current_leverage = state["current_leverage"]
        if volatility > config["leverage_adjustment_threshold"]:
            new_leverage = max(config["min_leverage"], current_leverage - 1)
        else:
            new_leverage = min(config["max_leverage"], current_leverage + 1)
        
        state["current_leverage"] = new_leverage
        logging.info(f"Adjusted leverage to {new_leverage} based on volatility {volatility:.4f}")
        return state
    except Exception as e:
        logging.error(f"adjust_leverage: {e}")
        return state

# Adjust SL/TP dynamically
def adjust_sl_tp(state, current_price, df):
    if not state["in_position"]:
        return state
    
    entry_price = state["entry_price"]
    side = state["side"]
    
    # Calculate ATR for dynamic SL/TP adjustment
    atr = float(df.iloc[-1]["atr"]) if df is not None and "atr" in df else 0
    if atr == 0:
        logging.warning("ATR calculation failed, using config stop-loss")
        stop_loss = entry_price * (1 - config["stop_loss"]) if side == "buy" else entry_price * (1 + config["stop_loss"])
        take_profit = entry_price * (1 + config["take_profit"]) if side == "buy" else entry_price * (1 - config["take_profit"])
    else:
        # Dynamic SL/TP adjustment based on market conditions
        if side == "buy":
            new_sl = max(
                state["stop_loss"],
                entry_price - (atr * config["atr_multiplier"] * 0.8)  # Tighten SL
            )
            new_tp = entry_price * (1 + config["take_profit"] * 1.2)  # Extend TP
        else:  # sell (short)
            new_sl = min(
                state["stop_loss"],
                entry_price + (atr * config["atr_multiplier"] * 0.8)  # Tighten SL
            )
            new_tp = entry_price * (1 - config["take_profit"] * 1.2)  # Extend TP
        
        stop_loss = new_sl
        take_profit = new_tp

    # Update state if changes are significant
    if abs(stop_loss - state["stop_loss"]) / entry_price > 0.001:  # 0.1% threshold
        state["stop_loss"] = stop_loss
        logging.info(f"Adjusted stop loss to {stop_loss} for {side} position")
    
    if abs(take_profit - state["take_profit"]) / entry_price > 0.001:
        state["take_profit"] = take_profit
        logging.info(f"Adjusted take profit to {take_profit} for {side} position")

    return state

# Calculate market volatility using ATR
def calculate_volatility(df):
    if df is None or "atr" not in df or df.empty:
        return 0.0
    latest_atr = float(df.iloc[-1]["atr"])
    avg_price = float(df["close"].mean())
    return latest_atr / avg_price  # Normalized volatility

# Adjust volume threshold dynamically
def adjust_volume_threshold(state, df):
    base_volume_ratio = config["min_volume_ratio"]
    dynamic_volume_ratio = df["volume"].rolling(window=20).mean().iloc[-1] * 1.2  # Example adjustment
    state["dynamic_volume_ratio"] = dynamic_volume_ratio
    logging.info(f"Adjusted volume ratio to {state['dynamic_volume_ratio']} based on recent activity")
    return state

# Adjust RSI/Stoch RSI thresholds based on volatility
def adjust_indicator_thresholds(state, df):
    volatility = calculate_volatility(df)
    if volatility > config["volatility_threshold"]:
        state["dynamic_rsi_buy"] = max(config["rsi_buy"] - 5, 20)
        state["dynamic_rsi_sell"] = min(config["rsi_sell"] + 5, 80)
        state["dynamic_stoch_rsi_buy"] = max(config["stoch_rsi_buy"] - 5, 10)
        state["dynamic_stoch_rsi_sell"] = min(config["stoch_rsi_sell"] + 5, 90)
    else:
        state["dynamic_rsi_buy"] = config["rsi_buy"]
        state["dynamic_rsi_sell"] = config["rsi_sell"]
        state["dynamic_stoch_rsi_buy"] = config["stoch_rsi_buy"]
        state["dynamic_stoch_rsi_sell"] = config["stoch_rsi_sell"]
    logging.info(f"Adjusted RSI: Buy={state['dynamic_rsi_buy']}, Sell={state['dynamic_rsi_sell']}; "
                 f"Stoch RSI: Buy={state['dynamic_stoch_rsi_buy']}, Sell={state['dynamic_stoch_rsi_sell']}")
    return state

# Monitor skipped trades and loosen conditions if needed
def monitor_skipped_trades(state, current_price, df):
    window = config["skipped_trade_window"]
    
    # Update price history
    state["last_price_history"].append(current_price)
    if len(state["last_price_history"]) > window:
        state["last_price_history"].pop(0)
    
    if state["trade_skipped"]:
        state["skipped_trades"] += 1
    else:
        state["skipped_trades"] = 0
    
    if len(state["last_price_history"]) == window:
        price_change = abs(state["last_price_history"][-1] - state["last_price_history"][0]) / state["last_price_history"][0]
        if state["skipped_trades"] >= config["max_skipped_trades"] and price_change >= config["price_move_threshold"]:
            state["dynamic_rsi_buy"] = max(state["dynamic_rsi_buy"] - 2, 15)
            state["dynamic_rsi_sell"] = min(state["dynamic_rsi_sell"] + 2, 85)
            state["dynamic_stoch_rsi_buy"] = max(state["dynamic_stoch_rsi_buy"] - 2, 5)
            state["dynamic_stoch_rsi_sell"] = min(state["dynamic_stoch_rsi_sell"] + 2, 95)
            state["skipped_trades"] = 0
            logging.info(f"Loosened conditions due to {state['skipped_trades']} skipped trades and {price_change:.2%} price move")
    
    return state

# Log skipped trades
import csv  # Ensure to import the csv module at the top of your file

# Log skipped trades in CSV format
def log_skipped_trade(reason, current_price, df):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "current_price": current_price,
        "volume": df["volume"].iloc[-1],
        "rsi": df["rsi"].iloc[-1],
        "stoch_rsi": df["stoch_rsi"].iloc[-1]
    }
    
    # Append to CSV file
    file_exists = os.path.isfile("skipped_trades_log.csv")
    with open("skipped_trades_log.csv", mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()  # Write header only if file does not exist
        writer.writerow(log_entry)

# Analyze skipped trades with machine learning
def analyze_skipped_trades():
    try:
        # Load skipped trades data
        skipped_trades_data = pd.read_json("skipped_trades_log.json", lines=True)
        
        # Feature engineering
        skipped_trades_data['price_change'] = skipped_trades_data['current_price'].diff().shift(-1)  # Price change after the trade was skipped
        features = skipped_trades_data[['volume', 'rsi', 'stoch_rsi', 'price_change']]
        labels = skipped_trades_data['reason'].apply(lambda x: 1 if x == "Expected profit below threshold" else 0)  # Example label

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train a Random Forest Classifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"Model accuracy: {accuracy:.2f}")

        # Save the model
        joblib.dump(model, "skipped_trades_model.pkl")
        logging.info("Saved machine learning model for skipped trades analysis.")

    except Exception as e:
        logging.error(f"Error analyzing skipped trades: {e}")

# Dynamic risk adjustment based on market conditions
def adjust_risk(state, df, balance):
    volatility = calculate_volatility(df)
    risk_factor = config["risk_adjustment_factor"]
    
    if volatility > config["volatility_threshold"]:
        dynamic_position_size = max(config["position_size"] * (1 - risk_factor), 0.1)
        dynamic_stop_loss = config["stop_loss"] * (1 + risk_factor)
        dynamic_take_profit = config["take_profit"] * (1 + risk_factor)
    else:
        dynamic_position_size = min(config["position_size"] * (1 + risk_factor), 0.9)
        dynamic_stop_loss = config["stop_loss"] * (1 - risk_factor / 2)
        dynamic_take_profit = config["take_profit"] * (1 - risk_factor / 2)
    
    logging.info(f"Risk adjusted: Position size={dynamic_position_size}, SL={dynamic_stop_loss}, TP={dynamic_take_profit}")
    return dynamic_position_size, dynamic_stop_loss, dynamic_take_profit

# Update trade_stats.json
def update_trade_stats(state):
    trade_stats = {
        "timestamp": datetime.now().isoformat(),
        "in_position": state["in_position"],
        "entry_price": state["entry_price"],
        "current_price": state["last_price"],
        "stop_loss": state["stop_loss"],
        "take_profit": state["take_profit"],
        "quantity": state["quantity"],
        "side": state["side"],
        "cumulative_fees": state["cumulative_fees"],
        "current_leverage": state["current_leverage"]
    }
    
    try:
        with open("trade_stats.json", "w") as f:
            json.dump(trade_stats, f, indent=4)
        logging.info("Updated trade_stats.json")
    except Exception as e:
        logging.error(f"Failed to update trade_stats.json: {e}")

# Place order
def place_order(side, quantity, dry_run=False):
    try:
        symbol_info = exchange.market(config["symbol"])
        quantity = exchange.amount_to_precision(config["symbol"], quantity)
        if dry_run:
            logging.info(f"Dry run: Placing {side} order for {quantity}")
            return {"price": fetch_ticker_with_fallback(load_state())}, None, None
        order = safe_api_call(exchange.create_market_order, config["symbol"], side, quantity)
        logging.info(f"Placed {side} order: {order}")
        return order, None, None
    except Exception as e:
        logging.error(f"Order placement failed: {e}")
        return None, None, None

# Backtest (placeholder with starting balance)
def backtest(days=30):
    logging.info(f"Starting backtest with initial balance: {config['starting_balance']} USDT")
    return config["starting_balance"] * 1.1, []  # Simulate 10% profit

# Main bot logic
def run_bot(dry_run=False):
    state = load_state()
    in_position = state["in_position"]
    entry_price = state["entry_price"]
    stop_loss = state["stop_loss"]
    take_profit = state["take_profit"]
    quantity = state["quantity"]
    partial_quantity = state["partial_quantity"]
    side = state["side"]
    cumulative_fees = state["cumulative_fees"]
    current_leverage = state["current_leverage"]
    last_review_time = time.time()
    review_interval = 3600  # Review every hour

    while True:
        try:
            # Fetch server time
            server_time = get_server_time()
            if server_time:
                logging.info(f"Server time: {server_time}")

            # Cooldown mechanism
            if state["trade_cooldown"] > 0:
                state["trade_cooldown"] -= 1
                time.sleep(60)
                continue

            # Manual trading mode check
            if config.get("manual_mode", False):
                logging.info("📌 Manual trading mode is enabled. Skipping automatic trades.")
                manual_side = input("Enter trade side (buy/sell): ").strip().lower()
                manual_quantity = float(input("Enter quantity: "))
                if manual_side in ["buy", "sell"] and manual_quantity > 0:
                    place_order(manual_side, manual_quantity, dry_run=dry_run)
                else:
                    logging.warning("❌ Invalid trade input. No order placed.")
                time.sleep(60)
                continue

            # Fetch balance
            account = safe_api_call(exchange.fetch_balance, {'type': 'future'})
            if account is None or 'total' not in account or 'USDT' not in account['total']:
                logging.error("run_bot: Invalid balance data")
                time.sleep(60)
                continue
            balance = float(account['total']['USDT'])
            logging.info(f"Current balance: {balance} USDT")
            if balance < config["min_balance"]:
                logging.error(f"Insufficient balance ({balance} USDT). Stopping bot.")
                break

            # Check cumulative fees
            if cumulative_fees > config["max_cumulative_fees"]:
                logging.warning(f"Cumulative fees ({cumulative_fees} USDT) exceed threshold.")
                break

            # Check existing positions
            positions = check_positions()
            state["positions"] = positions
            expected_symbol = exchange.market(config["symbol"])["id"]

            # Update state based on open positions
            if positions and expected_symbol in positions:
                pos = positions[expected_symbol]
                if pos["contracts"] > 0:
                    in_position = True
                    entry_price = float(pos["entryPrice"])
                    quantity = float(pos["contracts"])
                    side = "buy" if pos["side"] == "long" else "sell"
                    if state["stop_loss"] == 0 or state["take_profit"] == 0:
                        df = fetch_data(config["timeframe"], limit=config["atr_period"])
                        if df is not None:
                            df = calculate_indicators(df)
                            if df is not None:
                                atr = float(df.iloc[-1]["atr"])
                                stop_loss = entry_price - (atr * config["atr_multiplier"]) if side == "buy" else entry_price + (atr * config["atr_multiplier"])
                                take_profit = entry_price * (1 + config["take_profit"]) if side == "buy" else entry_price * (1 - config["take_profit"])
                            else:
                                stop_loss = entry_price * (1 - config["stop_loss"]) if side == "buy" else entry_price * (1 + config["stop_loss"])
                                take_profit = entry_price * (1 + config["take_profit"]) if side == "buy" else entry_price * (1 - config["take_profit"])
                    else:
                        stop_loss = state["stop_loss"]
                        take_profit = state["take_profit"]
                    state.update({
                        "in_position": True,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "quantity": quantity,
                        "partial_quantity": quantity / 2,
                        "side": side,
                        "cumulative_fees": cumulative_fees,
                        "current_leverage": current_leverage
                    })
                    save_state(state)
                    update_trade_stats(state)
                    logging.info(f"Synced state with position: quantity={quantity}, entry_price={entry_price}")

            elif not positions and in_position:
                in_position = False
                state.update({
                    "in_position": False,
                    "entry_price": 0,
                    "quantity": 0,
                    "side": None,
                    "stop_loss": 0,
                    "take_profit": 0,
                    "partial_quantity": 0
                })
                save_state(state)
                update_trade_stats(state)
                logging.info("Position closed externally, reset state")

            if not in_position:
                # Fetch and calculate data
                df = fetch_data(config["timeframe"])
                if df is None:
                    logging.error("run_bot: Failed to fetch data for trading")
                    time.sleep(60)
                    continue
                df_trend = fetch_data(config["trend_timeframe"])
                if df_trend is None:
                    logging.error("run_bot: Failed to fetch trend data")
                    time.sleep(60)
                    continue
                df = calculate_indicators(df)
                if df is None or df.empty:
                    logging.error("run_bot: Failed to calculate indicators or DataFrame is empty")
                    time.sleep(60)
                    continue
                df_trend = calculate_indicators(df_trend)
                if df_trend is None or df_trend.empty:
                    logging.error("run_bot: Failed to calculate trend indicators or DataFrame is empty")
                    time.sleep(60)
                    continue
                latest = df.iloc[-1]
                latest_trend = df_trend.iloc[-1]
                if latest is None or latest_trend is None:
                    logging.error("run_bot: Latest data row is None")
                    time.sleep(60)
                    continue
                current_price = fetch_ticker_with_fallback(state)
                if current_price is None:
                    logging.error("run_bot: No valid price available")
                    time.sleep(60)
                    continue
                avg_volume = df["volume"].rolling(window=20).mean().iloc[-1] if len(df) > 20 else df["volume"].mean()

                # Apply dynamic adjustments
                state = adjust_volume_threshold(state, df)
                state = adjust_indicator_thresholds(state, df)
                state = monitor_skipped_trades(state, current_price, df)
                position_size, stop_loss_pct, take_profit_pct = adjust_risk(state, df, balance)

                # Log current indicators continuously
                logging.info(f"Current indicators: RSI={latest['rsi']}, Stoch RSI={latest['stoch_rsi']}, "
                             f"MACD={latest['macd']}, MACD Signal={latest['macd_signal']}, Close={latest['close']}, "
                             f"BB Lower={latest['bb_lower']}, BB Upper={latest['bb_upper']}, Volume={latest['volume']}, "
                             f"Avg Volume={avg_volume}")

                # Adjust leverage
                state = adjust_leverage(state)
                current_leverage = state["current_leverage"]
                save_state(state)

                # Calculate quantity and fee
                quantity = min((position_size * balance) / current_price, balance * 0.01 / current_price)
                fee = (quantity * current_price / current_leverage) * 0.0004
                if fee / (quantity * current_price / current_leverage) > config["max_fee_ratio"]:
                    quantity *= (config["max_fee_ratio"] / (fee / (quantity * current_price / current_leverage)))
                    fee = (quantity * current_price / current_leverage) * 0.0004
                expected_profit = (current_price * (1 + take_profit_pct) - current_price) * quantity / current_leverage - 2 * fee
                if expected_profit / (quantity * current_price / current_leverage) < config["min_profit_ratio"]:
                    logging.info(f"Trade skipped: Expected profit ({expected_profit}) below threshold")
                    state["trade_skipped"] = True
                    log_skipped_trade("Expected profit below threshold", current_price, df)
                    save_state(state)
                    time.sleep(60)
                    continue
                if latest["volume"] < avg_volume * state["dynamic_volume_ratio"] and not is_volume_rising(df):
                    logging.info(f"Trade skipped: Volume ({latest['volume']}) below threshold ({avg_volume * state['dynamic_volume_ratio']}) and not rising")
                    state["trade_skipped"] = True
                    log_skipped_trade("Volume below threshold and not rising", current_price, df)
                    save_state(state)
                    time.sleep(60)
                    continue
                state["trade_skipped"] = False

                # Buy signal
                buy_conditions = {
                    "rsi": latest["rsi"] < state["dynamic_rsi_buy"] if pd.notna(latest["rsi"]) else False,
                    "stoch_rsi": latest["stoch_rsi"] < state["dynamic_stoch_rsi_buy"] if pd.notna(latest["stoch_rsi"]) else False,
                    "macd": latest["macd"] > -10 if pd.notna(latest["macd"]) else False,
                    "trend": (latest_trend["close"] > latest_trend["bb_upper"]) * config["trend_confirmation_weight"]  # Adjusted weight
                }
                if all(buy_conditions.values()):
                    order, sl_order, tp_order = place_order("buy", quantity, dry_run=dry_run)
                    if order:
                        entry_price = order["price"] if dry_run else float(order["average"])
                        if entry_price is None:
                            logging.error("run_bot: No valid price after buy order")
                            time.sleep(60)
                            continue
                        df_atr = calculate_indicators(fetch_data(config["timeframe"], limit=config["atr_period"]))
                        if df_atr is None:
                            stop_loss = entry_price * (1 - stop_loss_pct)
                            take_profit = entry_price * (1 + take_profit_pct)
                        else:
                            stop_loss = entry_price - (float(df_atr.iloc[-1]["atr"]) * config["atr_multiplier"])
                            take_profit = entry_price * (1 + take_profit_pct)
                        side = "buy"
                        in_position = True
                        partial_quantity = quantity / 2
                        fee = (quantity * entry_price / current_leverage) * 0.0004
                        cumulative_fees += fee
                        state.update({
                            "in_position": True,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "quantity": quantity,
                            "partial_quantity": partial_quantity,
                            "side": side,
                            "cumulative_fees": cumulative_fees,
                            "current_leverage": current_leverage,
                            "last_trade_time": time.time(),
                            "trade_cooldown": config.get("trade_cooldown", 0)
                        })
                        save_state(state)
                        update_trade_stats(state)
                else:
                    logging.info(f"Buy conditions not met: {buy_conditions}")
                    state["trade_skipped"] = True
                    log_skipped_trade("Buy conditions not met", current_price, df)
                    save_state(state)

                # Sell signal (shorting)
                sell_conditions = {
                    "rsi": latest["rsi"] > state["dynamic_rsi_sell"] if pd.notna(latest["rsi"]) else False,
                    "stoch_rsi": latest["stoch_rsi"] > state["dynamic_stoch_rsi_sell"] if pd.notna(latest["stoch_rsi"]) else False,
                    "macd": latest["macd"] < 10 if pd.notna(latest["macd"]) else False,
                    "trend": (latest_trend["close"] < latest_trend["bb_lower"]) * config["trend_confirmation_weight"]  # Adjusted weight
                }
                if all(sell_conditions.values()):
                    order, sl_order, tp_order = place_order("sell", quantity, dry_run=dry_run)
                    if order:
                        entry_price = order["price"] if dry_run else float(order["average"])
                        if entry_price is None:
                            logging.error("run_bot: No valid price after sell order")
                            time.sleep(60)
                            continue
                        df_atr = calculate_indicators(fetch_data(config["timeframe"], limit=config["atr_period"]))
                        if df_atr is None:
                            stop_loss = entry_price * (1 + stop_loss_pct)
                            take_profit = entry_price * (1 - take_profit_pct)
                        else:
                            stop_loss = entry_price + (float(df_atr.iloc[-1]["atr"]) * config["atr_multiplier"])
                            take_profit = entry_price * (1 - take_profit_pct)
                        side = "sell"
                        in_position = True
                        partial_quantity = quantity / 2
                        fee = (quantity * entry_price / current_leverage) * 0.0004
                        cumulative_fees += fee
                        state.update({
                            "in_position": True,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "quantity": quantity,
                            "partial_quantity": partial_quantity,
                            "side": side,
                            "cumulative_fees": cumulative_fees,
                            "current_leverage": current_leverage,
                            "last_trade_time": time.time(),
                            "trade_cooldown": config.get("trade_cooldown", 0)
                        })
                        save_state(state)
                        update_trade_stats(state)
                else:
                    logging.info(f"Sell conditions not met: {sell_conditions}")
                    state["trade_skipped"] = True
                    log_skipped_trade("Sell conditions not met", current_price, df)
                    save_state(state)

            elif in_position:
                # Check position status and adjust SL/TP
                current_price = fetch_ticker_with_fallback(state)
                if current_price is None:
                    logging.error("run_bot: No valid price available during position check")
                    time.sleep(60)
                    continue
                profit_loss = (current_price - entry_price) / entry_price if side == "buy" else (entry_price - current_price) / entry_price

                # Dynamic SL/TP adjustment
                df = fetch_data(config["timeframe"], limit=config["atr_period"])
                if df is not None:
                    df = calculate_indicators(df)
                    if df is not None:
                        atr = float(df.iloc[-1]["atr"]) if "atr" in df else 0
                        if atr == 0:
                            logging.warning("ATR calculation failed, using config stop-loss")
                            stop_loss = entry_price * (1 - config["stop_loss"]) if side == "buy" else entry_price * (1 + config["stop_loss"])
                            take_profit = entry_price * (1 + config["take_profit"]) if side == "buy" else entry_price * (1 - config["take_profit"])
                        else:
                            if side == "buy":
                                new_sl = max(state["stop_loss"], entry_price - (atr * config["atr_multiplier"] * 0.8))
                                new_tp = entry_price * (1 + config["take_profit"] * 1.1)
                            else:  # sell (short)
                                new_sl = min(state["stop_loss"], entry_price + (atr * config["atr_multiplier"] * 0.8))
                                new_tp = entry_price * (1 - config["take_profit"] * 1.1)
                            if abs(new_sl - state["stop_loss"]) / entry_price > 0.001:
                                stop_loss = new_sl
                                state["stop_loss"] = stop_loss
                                logging.info(f"Adjusted stop loss to {stop_loss} for {side} position")
                            if abs(new_tp - state["take_profit"]) / entry_price > 0.001:
                                take_profit = new_tp
                                state["take_profit"] = take_profit
                                logging.info(f"Adjusted take profit to {take_profit} for {side} position")
                        save_state(state)
                        update_trade_stats(state)

                # Trailing stop logic
                if profit_loss >= config["trailing_stop_trigger"] and quantity > partial_quantity:
                    new_stop = current_price * (1 - config["trailing_stop_pct"]) if side == "buy" else current_price * (1 + config["trailing_stop_pct"])
                    if (side == "buy" and new_stop > stop_loss) or (side == "sell" and new_stop < stop_loss):
                        stop_loss = new_stop
                        state["stop_loss"] = stop_loss
                        save_state(state)
                        update_trade_stats(state)
                        logging.info(f"Trailing stop updated to {stop_loss}")

                # Reversal fallback for dynamic profit locking
                if side == "buy" and current_price >= entry_price * (1 + config["take_profit"] / 2):
                    # Secure partial profits
                    partial_exit_quantity = quantity / 2
                    order = safe_api_call(exchange.create_market_order, config["symbol"], "sell", partial_exit_quantity)
                    if order:
                        logging.info(f"Partial profit taken at {current_price} for buy position.")
                        quantity -= partial_exit_quantity  # Reduce the remaining quantity
                        state["quantity"] = quantity
                        if quantity == 0:
                            in_position = False
                            state["in_position"] = False
                            save_state(state)
                            update_trade_stats(state)
                elif side == "sell" and current_price <= entry_price * (1 - config["take_profit"] / 2):
                    # Secure partial profits
                    partial_exit_quantity = quantity / 2
                    order = safe_api_call(exchange.create_market_order, config["symbol"], "buy", partial_exit_quantity)
                    if order:
                        logging.info(f"Partial profit taken at {current_price} for sell position.")
                        quantity -= partial_exit_quantity  # Reduce the remaining quantity
                        state["quantity"] = quantity
                        if quantity == 0:
                            in_position = False
                            state["in_position"] = False
                            save_state(state)
                            update_trade_stats(state)

                # Stop-loss exit
                if (side == "buy" and current_price <= stop_loss) or (side == "sell" and current_price >= stop_loss):
                    exit_side = "sell" if side == "buy" else "buy"
                    order = safe_api_call(exchange.create_market_order, config["symbol"], exit_side, quantity)
                    if order is None:
                        logging.error("run_bot: create_market_order returned None")
                        time.sleep(60)
                        continue
                    exit_price = float(order["average"])
                    fee = (quantity * exit_price / current_leverage) * 0.0004
                    cumulative_fees += fee
                    logging.info(f"Exited {side} at {exit_price} due to stop-loss, P/L: {profit_loss * current_leverage:.2%}")
                    in_position = False
                    state.update({
                        "in_position": False,
                        "entry_price": 0,
                        "quantity": 0,
                        "side": None,
                        "stop_loss": 0,
                        "take_profit": 0,
                        "partial_quantity": 0,
                        "cumulative_fees": cumulative_fees,
                        "current_leverage": current_leverage
                    })
                    save_state(state)
                    update_trade_stats(state)

            # Analyze skipped trades periodically
            if time.time() - last_review_time > review_interval:
                analyze_skipped_trades()
                last_review_time = time.time()

            time.sleep(60)
        except ccxt.RateLimitExceeded as e:
            logging.error(f"Rate limit hit: {e}. Retrying with backoff...")
            time.sleep(30)
        except Exception as e:
            logging.error(f"Bot error: {e}, Traceback: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    final_balance, trades = backtest(days=30)
    print(f"Backtest result: Final balance = {final_balance} USDT, Total trades = {len(trades)}")
    run_bot(dry_run=False)  # Start in dry-run mode for safety