*24 Hours Data Fetch Script*

*Overview*

The 24_hours.py script fetches D-1 (previous day’s) trade data from a MySQL database. It retrieves trade details for the last 24 hours, enriches the data with asset class classification, and combines it with account, customer, and country details. The final dataset is saved as a CSV file (trades_with_asset_classes.csv).

- How It Works
1. Database Connection
 - The script connects to a MySQL database using SQLAlchemy.
 - The connection parameters (host, port, user, password, database) are defined in db_config.
2. Fetching Trades Data (24-Hour Window)
 - The script queries the trades table to get trades opened or closed in the last 24 hours (D-1 period).
 - This is done by filtering timestamps between start_time (2025-02-24 00:00:00) and end_time (2025-02-24 23:59:59).
 - It also calculates FinalLot based on login patterns.
3. Classifying Trades by Asset Class
 - The script categorizes each trade into Crypto, Commodities, Indices, or Forex based on predefined lists.
 - If a trade’s symbol doesn’t match known categories, it is marked as "Unknown".
4. Fetching Additional Data
 - Accounts Data: Retrieves account details (e.g., login, equity, starting_balance, breachedby, etc.).
 - Customers Data: Fetches associated customer details (email, country_id).
 - Country Names: Fetches country names using country_id.
5. Data Merging and Cleaning
 - The script merges trades with account, customer, and country data.
U - nnecessary columns (account_id, updated_at, deleted_at, etc.) are dropped.
6. Saving the Output
 - The cleaned and enriched data is saved as a CSV file (trades_with_asset_classes.csv).
7. Performance Logging
 - The script records the execution time and prints how long it took to fetch and process the data.
8. Database Connection Closure
 - The script ensures the database connection is closed properly after execution.

Trade Analysis
- Overview
This Python script processes trading data from a CSV file, performs various analyses, and generates a detailed Excel report containing multiple summaries related to trading activity. It categorizes trades by asset class, calculates holding times, PnL (Profit and Loss), session activity, high-frequency trading patterns, and identifies top traders.

- How It Works
1. Load and Preprocess Data
- The script prompts the user to select a CSV file containing trade data.
- Extracts the date from the filename to use in the output file.
- Converts open_time and close_time to proper datetime format.
- Calculates holding time (duration of each trade in seconds).
2. Categorize Trades by Asset Class
- The script assigns trades to one of four categories: Forex, Crypto, Commodities, or Indices based on the symbol.
3. Calculate Maximum Total Lots for Each Login
- Identifies the maximum active lot size at any given time for each trader by considering buy/sell logic.
- Uses a custom function to track ongoing trades and dynamically compute lot exposure.
4. Create Summaries by Asset Class
- Generates dataframes for Commodities, Indices, Forex, and Crypto based on predefined lot thresholds.
- Stores the following details:
   - Max Total Lots
   - Equity
   - Package Type
   - Breachedby status
   - PnL (Profit and Loss)
5. Session Analysis
- Assigns trades to specific trading sessions based on their opening time:
- Market-Open Session (00:15 - 02:44)
- Prime Asia Session (02:45 - 08:59)
- London Opening Session (10:00 - 10:04)
- NY Session (16:36 - 21:00)
And more...
- Determines which sessions have the highest profit or trade volume.
6. High-Frequency Trading (HFT) Summary
- Identifies traders executing trades within 9 seconds or 3 seconds.
- Calculates:
   - Total trades per login
   - Percentage of trades executed within 3 seconds
   - Equity and account type
7. Top Performing Traders (Real & Demo Accounts)
- Finds the most profitable traders in real and demo accounts based on:
- Highest PnL
- Highest trade count
- Total lots used
- Identifies top-performing countries based on trader profits.
8. Arbitrage Traders Detection
- Detects traders with extremely short holding times (trades closed within 3 minutes).
- Computes average, min, and max holding time per trader.
9. Settlement and Late Night Trading Summary
- Detects traders actively trading during settlement hours (e.g., 00:00-01:00 for Forex).
- Filters out LL Hour (Low Liquidity) trading activity (trades placed between 22:00 - 24:00).
10. Overall Symbol Summary
- Provides a symbol-wise breakdown of:
- Total profit
- Number of trades
- Total volume
- Positive and negative profit breakdown
11. Session-wise PnL Summary
- Computes total PnL, average lot size, and number of trades per session.
- Shows the most active trading session per trader.


