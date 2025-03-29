# ACTF-V.2.0
 
# New Features 

<!-- 
Analysis of Proposed Improvements
Reversal Fallback (Dynamic Profit Locking):
Continuous Indicator Logging:
Dynamic Stop-Loss & Take-Profit (Volatility-Based Adjustments):
Multi-Timeframe Analysis:
Better Risk Management (Dynamic Leverage & Capital Allocation):
Partial Profit-Taking:
Trade Cooldown Mechanism:
Scalping Strategy Implementation:
Configurable Trading Balance & Position Sizing:
Trend Confirmation with Moving Averages (EMA/SMA):
Position Hedging (Multiple Positions Simultaneously):
g rather than full hedging. 
-->


# Previous Features 
<!-- 

Bot Features and Capabilities Analysis

Configuration Management

Loads configuration settings from a JSON file (config.json).
Configurable parameters include trading symbol, timeframes, leverage settings, risk management parameters, and technical indicator thresholds.
Exchange Integration

Utilizes the CCXT library to connect to Binance Futures Testnet.
Supports API key and secret for authentication.
Can operate in sandbox mode for testing without real funds.
State Management

Loads and saves trading state from/to a JSON file (trade_state.json).
Maintains information about current positions, entry prices, stop-loss, take-profit, and cumulative fees.
Data Fetching

Fetches OHLCV (Open, High, Low, Close, Volume) data for technical analysis.
Retrieves current ticker prices with fallback mechanisms to ensure reliability.
Technical Indicator Calculations

Calculates various technical indicators including:
RSI (Relative Strength Index)
Stochastic RSI
MACD (Moving Average Convergence Divergence)
Bollinger Bands
ATR (Average True Range)
Dynamic Risk Management

Adjusts position size, stop-loss, and take-profit levels based on market volatility.
Implements a risk adjustment factor to modify trading parameters dynamically.
Leverage Adjustment

Dynamically adjusts leverage based on market volatility to manage risk effectively.
Trade Execution

Places market orders for buying and selling based on predefined conditions.
Supports both long and short positions.
Trade Conditions and Signals

Implements buy and sell conditions based on technical indicators.
Uses dynamic thresholds for RSI and Stochastic RSI based on market volatility.
Volume and Fee Management

Monitors trading volume and adjusts trading conditions based on average volume.
Ensures that trading fees do not exceed a specified ratio of the trade size.
Trailing Stop Mechanism

Implements a trailing stop feature to lock in profits as the market moves favorably.
Manual Trading Mode

Allows for manual trading input, enabling users to place trades directly through the console.
Error Handling and Logging

Comprehensive logging of actions, errors, and state changes for debugging and monitoring.
Handles exceptions gracefully, including rate limits and API errors.
Backtesting Capability

Includes a placeholder for backtesting functionality to simulate trading performance over a specified period.
Monitoring Skipped Trades

Tracks consecutive skipped trades and adjusts trading conditions to improve trade opportunities.
Dynamic Adjustment of Trading Conditions

Adjusts trading parameters based on market conditions, including volume thresholds and indicator thresholds.
Trade Statistics Management

Updates and saves trade statistics to a JSON file (trade_stats.json) for performance tracking.
Continuous Operation

Runs in a loop, continuously monitoring market conditions and executing trades based on the defined strategy.

 -->