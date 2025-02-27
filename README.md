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
