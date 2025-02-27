import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from datetime import datetime

# Function to select and load the CSV file
def select_csv_file():
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    file_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path



# Select and load the CSV file
file_path = select_csv_file()
if not file_path:
    print("No file selected. Exiting...")
    exit()

trades_df = pd.read_csv(file_path)

# Extract date part from the file name for naming the output file
input_file_name = file_path.split('/')[-1]
file_date = input_file_name.split('_')[1].split('.')[0]  # Assumes format like 'trades_df_filtered_2024-08-22.csv'

# Ensure that 'open_time' and 'close_time' are in datetime format
trades_df['open_time'] = pd.to_datetime(trades_df['open_time'], unit='s')
trades_df['close_time'] = pd.to_datetime(trades_df['close_time'], unit='s', errors='coerce')
trades_df['close_time'] = trades_df['close_time'].fillna(pd.Timestamp('1970-01-01 00:00:00'))

# Calculate holding time in seconds
trades_df['holding_time'] = (trades_df['close_time'] - trades_df['open_time']).dt.total_seconds()


# Function to calculate the max total lots considering buy and sell for each login
def calculate_max_total_lots(trades):
    trades = sorted(trades, key=lambda x: x[0])  # Sort trades by open_time
    max_total_lots = 0
    active_trades = []

    for trade in trades:
        open_time, close_time, lot_size, symbol, type_str = trade

        # Remove expired trades (those that have already closed)
        active_trades = [t for t in active_trades if t[1] >= open_time]

        # Add the new trade to active trades
        active_trades.append(trade)

        # Calculate net lot size for each symbol considering buy/sell
        symbol_lots = {}
        for t in active_trades:
            sym = t[3]
            lot = t[2] if t[4].lower() == 'buy' else -t[2]  # Positive for 'Buy', negative for 'Sell'
            symbol_lots[sym] = symbol_lots.get(sym, 0) + lot

        # Calculate total active lots as the sum of absolute net positions for each symbol
        current_lots = sum(abs(net_lot) for net_lot in symbol_lots.values())

        # Update max_total_lots if current total lots exceed previous max
        if current_lots > max_total_lots:
            max_total_lots = current_lots

    return max_total_lots


# Function to create category DataFrame based on asset class and specific symbols
def create_category_df(trade_df, threshold, symbols):
    data = {
        'Login': [], 'Max Total Lots': [], 'Equity': [], 'Package Type': [], 'Breachedby': [], 'PnL': []
    }
    # Filter trades based on the provided symbols
    trade_df = trade_df[trade_df['symbol'].isin(symbols)]
    for login, trades in trade_df.groupby('login'):
        # Include type_str and symbol in the trade data to apply buy/sell logic
        trade_data = list(
            trades[['open_time', 'close_time', 'FinalLot', 'symbol', 'type_str']].itertuples(index=False, name=None))
        max_total_lots = calculate_max_total_lots(trade_data)

        if max_total_lots >= threshold:
            equity = trades['equity'].iloc[0]
            package_type = trades['type_account'].iloc[0]
            breachedby = trades['breachedby'].iloc[0]
            starting_balance = trades['starting_balance'].iloc[0]  # Ensure starting_balance is included
            pnl = equity - starting_balance  # Calculate PnL directly
            data['Login'].append(login)
            data['Max Total Lots'].append(max_total_lots)
            data['Equity'].append(equity)
            data['Package Type'].append(package_type)
            data['Breachedby'].append(breachedby)
            data['PnL'].append(pnl)  # Append PnL directly
    return pd.DataFrame(data)


# Define the symbols to filter for each asset class
symbols_mapping = {
    'Commodities': ['XAUUSD'],
    'Indices': ["NDX100", "US30", "GER30", "FRA40", "EUSTX50", "US2000", "UK100", "SWI20"],
    'Forex': trades_df['symbol'].unique(),  # All Forex symbols
    'Crypto': ['BTCUSD']
}

# Define thresholds for different asset classes
thresholds = {
    'Commodities': 18,
    'Indices': 7,
    'Forex': 70,
    'Crypto': 3
}

# Create DataFrames for each asset class with the specified symbols
commodities_df = create_category_df(trades_df[trades_df['asset_class'] == 'Commodities'], thresholds['Commodities'], symbols_mapping['Commodities'])
indices_df = create_category_df(trades_df[trades_df['asset_class'] == 'Indices'], thresholds['Indices'], symbols_mapping['Indices'])
forex_df = create_category_df(trades_df[trades_df['asset_class'] == 'Forex'], thresholds['Forex'], symbols_mapping['Forex'])
crypto_df = create_category_df(trades_df[trades_df['asset_class'] == 'Crypto'], thresholds['Crypto'], symbols_mapping['Crypto'])

# Combine DataFrames into one
max_rows = max(len(commodities_df), len(indices_df), len(forex_df), len(crypto_df))

commodities_df = commodities_df.reindex(range(max_rows)).reset_index(drop=True)
indices_df = indices_df.reindex(range(max_rows)).reset_index(drop=True)
forex_df = forex_df.reindex(range(max_rows)).reset_index(drop=True)
crypto_df = crypto_df.reindex(range(max_rows)).reset_index(drop=True)

combined_df = pd.concat([commodities_df, indices_df, forex_df, crypto_df], axis=1)

# Rename columns to include PnL
combined_df.columns = [
    'Commodities_Login', 'Commodities_Max Total Lots', 'Commodities_Equity', 'Commodities_Package Type',
    'Commodities_Breachedby', 'Commodities_PnL',
    'Indices_Login', 'Indices_Max Total Lots', 'Indices_Equity', 'Indices_Package Type', 'Indices_Breachedby',
    'Indices_PnL',
    'Forex_Login', 'Forex_Max Total Lots', 'Forex_Equity', 'Forex_Package Type', 'Forex_Breachedby',
    'Forex_PnL',
    'Crypto_Login', 'Crypto_Max Total Lots', 'Crypto_Equity', 'Crypto_Package Type', 'Crypto_Breachedby',
    'Crypto_PnL'
]




# Vectorized session calculation
time_obj = pd.to_datetime(trades_df['open_time_str']).dt.time

conditions = [
    (time_obj >= pd.to_datetime('00:15').time()) & (time_obj <= pd.to_datetime('02:44').time()),
    (time_obj >= pd.to_datetime('02:45').time()) & (time_obj <= pd.to_datetime('08:59').time()),
    (time_obj >= pd.to_datetime('09:00').time()) & (time_obj <= pd.to_datetime('09:59').time()),
    (time_obj >= pd.to_datetime('10:00').time()) & (time_obj <= pd.to_datetime('10:04').time()),
    (time_obj >= pd.to_datetime('10:05').time()) & (time_obj <= pd.to_datetime('13:59').time()),
    (time_obj >= pd.to_datetime('14:00').time()) & (time_obj <= pd.to_datetime('14:59').time()),
    (time_obj >= pd.to_datetime('15:00').time()) & (time_obj <= pd.to_datetime('15:04').time()),
    (time_obj >= pd.to_datetime('15:05').time()) & (time_obj <= pd.to_datetime('16:29').time()),
    (time_obj >= pd.to_datetime('16:30').time()) & (time_obj <= pd.to_datetime('16:35').time()),
    (time_obj >= pd.to_datetime('16:36').time()) & (time_obj <= pd.to_datetime('21:00').time()),
    (time_obj >= pd.to_datetime('21:01').time()) & (time_obj <= pd.to_datetime('22:59').time()),
    (time_obj >= pd.to_datetime('23:00').time()) & (time_obj <= pd.to_datetime('23:59').time()),
]

choices = [
    'Market-Open Session', 'Prime Asia Session', 'Pre London Session',
    'London Opening Session', 'London Session', 'Pre-NY Session',
    'NY-Open Session', 'Pre-NYSE Session', 'NYSE-Open Session',
    'NY Session', 'Late Trading Hours Session', 'Market-Closing Session'
]

trades_df['session'] = pd.Series(np.select(conditions, choices, default='Unknown Session'))

# Generate Pairwise Summary for Real Accounts
real_accounts = trades_df[trades_df['type_account'].str.contains("Real", case=False, na=False)]

pairwise_summary_real = real_accounts.groupby('symbol').agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    unique_logins=('login', 'nunique')
).reset_index()

# Group the trades by symbol and session to calculate profit and lots
max_profit_session = real_accounts.groupby(['symbol', 'session']).agg(
    session_profit=('profit', 'sum'),
    session_lot=('FinalLot', 'sum')
).reset_index()

# Get the session with the maximum profit for each symbol
max_profit_session = max_profit_session.loc[max_profit_session.groupby('symbol')['session_profit'].idxmax()]
max_profit_session = max_profit_session.rename(columns={
    'session': 'max_profit_session',
    'session_profit': 'max_profit_session_profit',
    'session_lot': 'max_profit_session_lot'
})

# Get the session with the maximum lot size for each symbol
max_lot_session = real_accounts.groupby(['symbol', 'session']).agg(
    session_lot=('FinalLot', 'sum')
).reset_index()

# Get the session with the maximum lot size for each symbol
max_lot_session = max_lot_session.loc[max_lot_session.groupby('symbol')['session_lot'].idxmax()]
max_lot_session = max_lot_session.rename(columns={
    'session': 'max_lot_session',
    'session_lot': 'max_lot_session_lot'
})

# Merge the max profit session and max lot session data with the main summary
pairwise_summary_real = pairwise_summary_real.merge(
    max_profit_session[['symbol', 'max_profit_session', 'max_profit_session_profit', 'max_profit_session_lot']],
    on='symbol', how='left'
)

pairwise_summary_real = pairwise_summary_real.merge(
    max_lot_session[['symbol', 'max_lot_session', 'max_lot_session_lot']],
    on='symbol', how='left'
)

# Rename columns to reflect the final lot usage
pairwise_summary_real = pairwise_summary_real.rename(columns={'max_lot_session_lot': 'FinalLot'})

# Generate Pairwise Summary for Demo Accounts
demo_accounts = trades_df[~trades_df['type_account'].str.contains("Real", case=False, na=False)]

pairwise_summary_demo = demo_accounts.groupby('symbol').agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    unique_logins=('login', 'nunique')
).reset_index()

# Do the same for Pairwise Summary Demo
max_profit_session_demo = demo_accounts.groupby(['symbol', 'session']).agg(
    session_profit=('profit', 'sum'),
    session_lot=('FinalLot', 'sum')
).reset_index()

# Get the session with the maximum profit for each symbol
max_profit_session_demo = max_profit_session_demo.loc[max_profit_session_demo.groupby('symbol')['session_profit'].idxmax()]
max_profit_session_demo = max_profit_session_demo.rename(columns={
    'session': 'max_profit_session',
    'session_profit': 'max_profit_session_profit',
    'session_lot': 'max_profit_session_lot'
})

# Get the session with the maximum lot size for each symbol
max_lot_session_demo = demo_accounts.groupby(['symbol', 'session']).agg(
    session_lot=('FinalLot', 'sum')
).reset_index()

# Get the session with the maximum lot size for each symbol
max_lot_session_demo = max_lot_session_demo.loc[max_lot_session_demo.groupby('symbol')['session_lot'].idxmax()]
max_lot_session_demo = max_lot_session_demo.rename(columns={
    'session': 'max_lot_session',
    'session_lot': 'max_lot_session_lot'
})

# Merge the max profit session and max lot session data with the main summary
pairwise_summary_demo = pairwise_summary_demo.merge(
    max_profit_session_demo[['symbol', 'max_profit_session', 'max_profit_session_profit', 'max_profit_session_lot']],
    on='symbol', how='left'
)

pairwise_summary_demo = pairwise_summary_demo.merge(
    max_lot_session_demo[['symbol', 'max_lot_session', 'max_lot_session_lot']],
    on='symbol', how='left'
)

# Rename columns to reflect the final lot usage
pairwise_summary_demo = pairwise_summary_demo.rename(columns={'max_lot_session_lot': 'FinalLot'})

# High-Frequency Trading (HFT) Summary
valid_trades = trades_df[trades_df['close_time'] > trades_df['open_time']]

trade_counts = {}
trade_counts_3s = {}
total_trade_counts = {}
equity_dict = {}
type_dict = {}

for index, row in valid_trades.iterrows():
    login = row['login']
    holding_time = row['holding_time']

    if holding_time <= 9:
        trade_counts[login] = trade_counts.get(login, 0) + 1

    if holding_time <= 3:
        trade_counts_3s[login] = trade_counts_3s.get(login, 0) + 1

    total_trade_counts[login] = total_trade_counts.get(login, 0) + 1

    if login not in equity_dict:
        equity_dict[login] = row['equity']
    if login not in type_dict:
        type_dict[login] = row['type_account']

hft_summary = pd.DataFrame({
    'Login': list(trade_counts.keys()),
    'Trade Count (<=9s)': [trade_counts[login] for login in trade_counts],
    'Trade Count (<=3s)': [trade_counts_3s.get(login, 0) for login in trade_counts],
    'Total Trades of the Day': [total_trade_counts[login] for login in trade_counts],
    'Equity': [equity_dict[login] for login in trade_counts],
    'Type': [type_dict[login] for login in trade_counts]
})

hft_summary['Trade Count % (<=3s)'] = hft_summary['Trade Count (<=3s)'] / hft_summary['Total Trades of the Day']

hft_summary = hft_summary.sort_values(by='Trade Count (<=9s)', ascending=False).head(20)

# High PNL Traders Real
high_pnl_traders_real = real_accounts.groupby('login').agg(
    profit=('profit', 'sum'),
    country_name=('country_name', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    highest_profit_trade=('profit', 'max'),
    highest_profit_pair=('symbol', lambda x: x[real_accounts.loc[x.index, 'profit'].idxmax()] if not x.empty else None),
    highest_used_lot=('FinalLot', 'max'),
    total_used_lot=('FinalLot', 'sum'),
    trade_count=('id', 'count'),
    email=('email', 'first')
).reset_index().sort_values(by='profit', ascending=False).head(10)

# Login Trade Lot Real
login_trade_lot_summary_real = real_accounts.groupby('login').agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    country_name=('country_name', 'first'),
    email=('email', 'first'),
    account_type=('type_account', 'first'),
    created_at=('created_at', 'first')
).reset_index()

# Login Trade Lot P2
p2_accounts = trades_df[(trades_df['type_account'].str.contains("P2|Stellar 1-Step Demo", na=False, case=False)) &
                        trades_df['type_account'].str.contains("100K|200K", na=False, case=False)]
login_trade_lot_summary_p2 = p2_accounts.groupby('login').agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    country_name=('country_name', 'first'),
    email=('email', 'first'),
    account_type=('type_account', 'first'),
    created_at=('created_at', 'first')
).reset_index()

# Most Profitable Countries Real
most_profitable_countries_real = real_accounts.groupby('country_name').agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    trader_count=('login', 'nunique')
).reset_index().sort_values(by='total_profit', ascending=False).head(10)

# Filter the real accounts to include only those from the top 10 countries
top_countries = most_profitable_countries_real['country_name'].tolist()
filtered_real_accounts = real_accounts[real_accounts['country_name'].isin(top_countries)]

# Most Profitable Logins Real
most_profitable_logins_real = filtered_real_accounts.groupby(['country_name', 'login']).agg(
    profit=('profit', 'sum'),
    account_type=('type_account', 'first'),
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    email=('email', 'first')
).reset_index().sort_values(by=['country_name', 'profit'], ascending=[True, False]).groupby('country_name').head(3)

# Most Profitable Countries Demo
most_profitable_countries_demo = demo_accounts.groupby('country_name').agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    trader_count=('login', 'nunique')
).reset_index().sort_values(by='total_profit', ascending=False).head(10)

# Filter the demo accounts to include only those from the top 10 countries
filtered_demo_accounts = demo_accounts[demo_accounts['country_name'].isin(top_countries)]

# Most Profitable Logins Demo
most_profitable_logins_demo = filtered_demo_accounts.groupby(['country_name', 'login']).agg(
    profit=('profit', 'sum'),
    account_type=('type_account', 'first'),
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum')
).reset_index().sort_values(by=['country_name', 'profit'], ascending=[True, False]).groupby('country_name').head(3)

# Top Logins Real: Which login is making the highest profit in each symbol for real accounts
top_logins_real = real_accounts.groupby(['symbol', 'login']).agg(
    profit=('profit', 'sum'),
    account_type=('type_account', 'first'),
    current_pnl=('FinalLot', 'sum'),
    country_name=('country_name', 'first')
).reset_index().sort_values(by=['symbol', 'profit'], ascending=[True, False]).groupby('symbol').head(3)

# Top Logins Demo: Which login is making the highest profit in each symbol for demo accounts
top_logins_demo = demo_accounts.groupby(['symbol', 'login']).agg(
    profit=('profit', 'sum'),
    account_type=('type_account', 'first'),
    current_pnl=('FinalLot', 'sum'),
    country_name=('country_name', 'first')
).reset_index().sort_values(by=['symbol', 'profit'], ascending=[True, False]).groupby('symbol').head(3)

# Top Lots Per Trade Real: Which login used the highest lot per trade in each symbol for real accounts
top_lots_per_trade_real = real_accounts.groupby(['symbol', 'login']).agg(
    max_lot_per_trade=('FinalLot', 'max'),
    account_type=('type_account', 'first'),
    current_pnl=('profit', 'sum'),
    country_name=('country_name', 'first')
).reset_index().sort_values(by=['symbol', 'max_lot_per_trade'], ascending=[True, False]).groupby('symbol').head(3)

# Top Lots Per Trade Demo: Which login used the highest lot per trade in each symbol for demo accounts
top_lots_per_trade_demo = demo_accounts.groupby(['symbol', 'login']).agg(
    max_lot_per_trade=('FinalLot', 'max'),
    account_type=('type_account', 'first'),
    current_pnl=('profit', 'sum'),
    country_name=('country_name', 'first')
).reset_index().sort_values(by=['symbol', 'max_lot_per_trade'], ascending=[True, False]).groupby('symbol').head(3)

# Swap Summary: Display the swap details for each login and symbol
swap_summary = trades_df.groupby(['login', 'symbol'])['swap'].sum().reset_index()

# Open Hour Summary Real: Summarize trading activity by open hour for real accounts
open_hour_summary_real = real_accounts.groupby(real_accounts['open_time'].dt.hour).agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    unique_logins=('login', 'nunique'),
    highest_lot_country=('country_name', lambda x: x[real_accounts.loc[x.index, 'FinalLot'].idxmax()]),
    highest_lot_country_lot=('FinalLot', 'max'),
    highest_profit_country=('country_name', lambda x: x[real_accounts.loc[x.index, 'profit'].idxmax()]),
    highest_profit_country_profit=('profit', 'max')
).reset_index().rename(columns={'open_time': 'open_hour'})

# Open Hour Summary Demo: Summarize trading activity by open hour for demo accounts
open_hour_summary_demo = demo_accounts.groupby(demo_accounts['open_time'].dt.hour).agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    unique_logins=('login', 'nunique'),
    highest_lot_country=('country_name', lambda x: x[demo_accounts.loc[x.index, 'FinalLot'].idxmax()]),
    highest_lot_country_lot=('FinalLot', 'max'),
    highest_profit_country=('country_name', lambda x: x[demo_accounts.loc[x.index, 'profit'].idxmax()]),
    highest_profit_country_profit=('profit', 'max')
).reset_index().rename(columns={'open_time': 'open_hour'})

# Close Hour Summary Real: Summarize trading activity by close hour for real accounts
close_hour_summary_real = real_accounts.groupby(real_accounts['close_time'].dt.hour).agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    unique_logins=('login', 'nunique'),
    highest_lot_country=('country_name', lambda x: x[real_accounts.loc[x.index, 'FinalLot'].idxmax()]),
    highest_lot_country_lot=('FinalLot', 'max'),
    highest_profit_country=('country_name', lambda x: x[real_accounts.loc[x.index, 'profit'].idxmax()]),
    highest_profit_country_profit=('profit', 'max')
).reset_index().rename(columns={'close_time': 'close_hour'})

# Close Hour Summary Demo: Summarize trading activity by close hour for demo accounts
close_hour_summary_demo = demo_accounts.groupby(demo_accounts['close_time'].dt.hour).agg(
    trade_count=('id', 'count'),
    total_lots=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    unique_logins=('login', 'nunique'),
    highest_lot_country=('country_name', lambda x: x[demo_accounts.loc[x.index, 'FinalLot'].idxmax()]),
    highest_lot_country_lot=('FinalLot', 'max'),
    highest_profit_country=('country_name', lambda x: x[demo_accounts.loc[x.index, 'profit'].idxmax()]),
    highest_profit_country_profit=('profit', 'max')
).reset_index().rename(columns={'close_time': 'close_hour'})

# Symbol Summary: Provides a detailed summary of each symbol
symbol_summary = trades_df.groupby('symbol').agg(
    trade_count=('id', 'count'),
    total_profit=('profit', 'sum'),
    trader_count=('login', 'nunique'),
    positive_trades_sum=('profit', lambda x: x[x > 0].sum()),
    negative_trades_sum=('profit', lambda x: x[x < 0].sum()),
    final_lot=('FinalLot', 'sum')
).reset_index()

# Calculate additional metrics for Symbol Summary
symbol_summary['final_lot_per_trade'] = symbol_summary['final_lot'] / symbol_summary['trade_count']
symbol_summary['final_lot_per_trader'] = symbol_summary['final_lot'] / symbol_summary['trader_count']

def is_within_time_range(start_time, close_time, start_hour, end_hour):
    return (start_hour <= start_time.hour < end_hour) and (start_hour <= close_time.hour < end_hour)

# Function to count logins for specific instruments and time ranges
def count_logins(df, start_hour, end_hour):
    login_dict = {}
    for _, row in df.iterrows():
        login = row['login']
        start_time = row['open_time']
        close_time = row['close_time']
        instrument = row['symbol']
        if is_within_time_range(start_time, close_time, start_hour, end_hour):
            if login not in login_dict:
                login_dict[login] = 1
            else:
                login_dict[login] += 1
    return login_dict

# Count logins for Settlement (Forex)
settlement_forex_df = trades_df[(trades_df['asset_class'] == 'Forex') &
                                (~trades_df['close_time'].isin([pd.Timestamp('1970-01-01 00:00:00'), pd.Timestamp('1970-01-01 03:00:00')]))]
settlement_forex_dict = count_logins(settlement_forex_df, 0, 1)
settlement_forex_df = pd.DataFrame(list(settlement_forex_dict.items()), columns=['Login', 'Count'])
settlement_forex_df = settlement_forex_df[settlement_forex_df['Count'] >= 5]

# Count logins for Settlement (Indices)
settlement_indices_df = trades_df[(trades_df['asset_class'] == 'Indices') &
                                  (~trades_df['close_time'].isin([pd.Timestamp('1970-01-01 00:00:00'), pd.Timestamp('1970-01-01 03:00:00')]))]
settlement_indices_dict = count_logins(settlement_indices_df, 1, 2)
settlement_indices_df = pd.DataFrame(list(settlement_indices_dict.items()), columns=['Login', 'Count'])
settlement_indices_df = settlement_indices_df[settlement_indices_df['Count'] >= 5]

# Define the specific pairs to be included in the LL hour summary
ll_hour_pairs = [
    "AUDJPY", "AUDNZD", "AUDSGD", "CADCHF", "CADJPY", "CADSGD",
    "CHFJPY", "EURCHF", "EURGBP", "EURHKD", "EURHUF", "EURJPY",
    "EURNOK", "EURNZD", "EURSGD", "EURTRY", "GBPAUD", "GBPCAD",
    "GBPCHF", "GBPJPY", "GBPNZD", "GBPSGD", "MXNJPY", "NOKJPY",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDSGD", "SGDJPY", "ZARJPY",
    "AUDCHF", "CADEUR"
]

# Filter trades to include only those from the specific pairs during LL hour
ll_hour_df = trades_df[
    trades_df['symbol'].isin(ll_hour_pairs) &
    trades_df['asset_class'].isin(['Forex', 'Indices', 'Commodities', 'Crypto']) &
    (~trades_df['close_time'].isin([pd.Timestamp('1970-01-01 00:00:00'), pd.Timestamp('1970-01-01 03:00:00')]))
]

# Count logins for LL Hour with the specified pairs
ll_hour_dict = count_logins(ll_hour_df, 22, 24)
ll_hour_df = pd.DataFrame(list(ll_hour_dict.items()), columns=['Login', 'Count'])
ll_hour_df = ll_hour_df[ll_hour_df['Count'] >= 5]

# Update only for the arbitrage traders part to calculate in minutes
arbitrage_traders_summary = real_accounts.groupby(['login', 'email', 'type_account']).agg(
    average_holding_time=('holding_time', lambda x: x.mean() / 60),  # Convert to minutes
    max_holding_time=('holding_time', lambda x: x.max() / 60),  # Convert to minutes
    min_holding_time=('holding_time', lambda x: x.min() / 60),  # Convert to minutes
    trade_count=('id', 'count'),
    total_used_lot=('FinalLot', 'sum'),
    total_profit=('profit', 'sum'),
    country=('country_name', lambda x: x.mode().iloc[0] if not x.mode().empty else None)
).reset_index()

# Filter out traders with missing holding time data (trades not closed)
arbitrage_traders_summary = arbitrage_traders_summary.dropna(subset=['average_holding_time', 'max_holding_time', 'min_holding_time'])

# Calculate percentage of trades within 3 minutes (3 * 60 = 180 seconds)
trade_within_3_min = real_accounts[real_accounts['holding_time'] <= 180].groupby('login').size()
arbitrage_traders_summary['trades_within_3_min'] = arbitrage_traders_summary['login'].map(trade_within_3_min).fillna(0)
arbitrage_traders_summary['trades_within_3_min_pct'] = (arbitrage_traders_summary['trades_within_3_min'] / arbitrage_traders_summary['trade_count']) * 100

# Rearrange columns to bring the last two columns after trade_count
arbitrage_traders_summary = arbitrage_traders_summary[
    ['login', 'email', 'type_account', 'average_holding_time', 'max_holding_time', 'min_holding_time', 'trade_count',
     'trades_within_3_min', 'trades_within_3_min_pct', 'total_used_lot', 'total_profit', 'country']
]

# Filter the trades to include only real accounts
real_accounts = trades_df[trades_df['type_account'].str.contains("Real", case=False, na=False)]

# Calculate the cumulative PnL for the day's trades by login
today_pnl_by_login = real_accounts.groupby('login')['profit'].sum().reset_index()
today_pnl_by_login.columns = ['login', 'today_pnl']

# Calculate overall PnL for each login using the provided starting_balance
def calculate_overall_pnl(group):
    last_equity = group['equity'].iloc[-1]
    starting_balance = group['starting_balance'].iloc[0]
    overall_pnl = last_equity - starting_balance
    return overall_pnl

# Apply the function to calculate overall PnL
overall_pnl_by_login = real_accounts.groupby('login', as_index=False).apply(calculate_overall_pnl, include_groups=False).reset_index(drop=True)
overall_pnl_by_login.columns = ['login', 'overall_pnl']

# Merge today's PnL and overall PnL into one DataFrame
pnl_summary = pd.merge(today_pnl_by_login, overall_pnl_by_login, on='login')

# Filter to only include traders with both positive today's PnL and positive overall PnL
positive_pnl_traders = pnl_summary[(pnl_summary['today_pnl'] > 0) & (pnl_summary['overall_pnl'] > 0)]

# Add trader name and starting balance for output
trader_info = real_accounts[['login', 'starting_balance']].drop_duplicates()
positive_pnl_traders = pd.merge(positive_pnl_traders, trader_info, on='login')

# Make sure real_accounts is not a slice of another DataFrame
real_accounts = real_accounts.copy()


# Aggregate the data by symbol to calculate total PnL and the number of trades for each symbol
# Filter out breached accounts
real_accounts = trades_df[
    (trades_df['type_account'].str.contains("Real", case=False, na=False)) &
    (trades_df['breachedby'].isna())
]
real_accounts.loc[:, 'positive_profit'] = real_accounts['profit'].apply(lambda x: x if x > 0 else 0)

# Recalculate symbol PnL summary using the filtered real_accounts
symbol_pnl_summary = real_accounts.groupby('symbol').agg(
    total_pnl=('profit', 'sum'),
    positive_profit_total=('positive_profit', 'sum'),  # Ensure positive profit is calculated correctly
    trade_count=('id', 'count')
).reset_index()

# Find the top 5 symbols based on PnL
top_5_symbols_by_pnl = symbol_pnl_summary.sort_values(by='total_pnl', ascending=False).head(5)

# Find the top 5 symbols based on the number of trades
top_5_symbols_by_trade_count = symbol_pnl_summary.sort_values(by='trade_count', ascending=False).head(5)

# Filter the real_accounts DataFrame to only include these traders
filtered_accounts = real_accounts[real_accounts['login'].isin(positive_pnl_traders['login'])]

 #Define the session time ranges by pairing the conditions with the choices
session_time_ranges = [
    ('Market-Open Session', '00:15 - 02:44'),
    ('Prime Asia Session', '02:45 - 08:59'),
    ('Pre London Session', '09:00 - 09:59'),
    ('London Opening Session', '10:00 - 10:04'),
    ('London Session', '10:05 - 13:59'),
    ('Pre-NY Session', '14:00 - 14:59'),
    ('NY-Open Session', '15:00 - 15:04'),
    ('Pre-NYSE Session', '15:05 - 16:29'),
    ('NYSE-Open Session', '16:30 - 16:35'),
    ('NY Session', '16:36 - 21:00'),
    ('Late Trading Hours Session', '21:01 - 22:59'),
    ('Market-Closing Session', '23:00 - 23:59')
]

# Convert the list to a DataFrame for easy merging
session_time_df = pd.DataFrame(session_time_ranges, columns=['session', 'session_time_range'])


# Check if there are any missing or invalid open_time values
missing_open_time_trades = real_accounts[real_accounts['open_time'].isna()]
if not missing_open_time_trades.empty:
    print(f"Trades with missing or invalid open_time:\n{missing_open_time_trades[['login', 'symbol', 'open_time', 'FinalLot', 'profit']]}")

# Check for outliers in open_time (outside expected market hours)
outlier_trades = real_accounts[(real_accounts['open_time'].dt.time < pd.to_datetime('00:15').time()) |
                               (real_accounts['open_time'].dt.time > pd.to_datetime('23:59').time())]
if not outlier_trades.empty:
    print(f"Trades with outlier open_time values (outside market hours):\n{outlier_trades[['login', 'symbol', 'open_time', 'FinalLot', 'profit']]}")

# Assuming the 'session' column is already created in your real_accounts DataFrame
# Group by session and calculate the required metrics
session_summary = real_accounts.groupby('session').agg(
    lot_used=('FinalLot', 'sum'),
    number_of_trades=('id', 'count'),  # Assuming 'id' is unique for each trade
    total_pnl=('profit', 'sum'),
    average_pnl=('profit', 'mean'),
    average_lot=('FinalLot', 'mean')
).reset_index()

# Optional: Remove "Unknown Session" from the summary if needed
session_summary = session_summary[session_summary['session'] != 'Unknown Session']

# Merge the session time ranges into the session summary
session_summary = pd.merge(session_summary, session_time_df, on='session', how='left')

# Reorder columns to match the order shown in the image
session_summary = session_summary[['session', 'session_time_range', 'lot_used', 'number_of_trades', 'total_pnl', 'average_pnl', 'average_lot']]


# Create the output file name based on the date in the file
output_excel_file = f"trades_analysis_{file_date}.xlsx"

# Save the summaries to the new Excel file
with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
    pairwise_summary_real.to_excel(writer, sheet_name='Pairwise Summary Real', index=False)
    pairwise_summary_demo.to_excel(writer, sheet_name='Pairwise Summary Demo', index=False)
    hft_summary.to_excel(writer, sheet_name='HFT Summary', index=False)
    high_pnl_traders_real.to_excel(writer, sheet_name='High PNL Traders Real', index=False)
    login_trade_lot_summary_real.to_excel(writer, sheet_name='Login Trade Lot Real', index=False)
    login_trade_lot_summary_p2.to_excel(writer, sheet_name='Login Trade Lot P2', index=False)
    combined_df.to_excel(writer, sheet_name='Combined Asset Classes', index=False)
    most_profitable_countries_real.to_excel(writer, sheet_name='Most Profitable Countries Real', index=False)
    most_profitable_countries_demo.to_excel(writer, sheet_name='Most Profitable Countries Demo', index=False)
    most_profitable_logins_real.to_excel(writer, sheet_name='Most Profitable Logins Real', index=False)
    most_profitable_logins_demo.to_excel(writer, sheet_name='Most Profitable Logins Demo', index=False)
    top_logins_real.to_excel(writer, sheet_name='Top Logins Real', index=False)
    top_logins_demo.to_excel(writer, sheet_name='Top Logins Demo', index=False)
    top_lots_per_trade_real.to_excel(writer, sheet_name='Top Lots Per Trade Real', index=False)
    top_lots_per_trade_demo.to_excel(writer, sheet_name='Top Lots Per Trade Demo', index=False)
    swap_summary.to_excel(writer, sheet_name='Swap Summary', index=False)
    open_hour_summary_real.to_excel(writer, sheet_name='Open Hour Summary Real', index=False)
    open_hour_summary_demo.to_excel(writer, sheet_name='Open Hour Summary Demo', index=False)
    close_hour_summary_real.to_excel(writer, sheet_name='Close Hour Summary Real', index=False)
    close_hour_summary_demo.to_excel(writer, sheet_name='Close Hour Summary Demo', index=False)
    symbol_summary.to_excel(writer, sheet_name='Symbol Summary', index=False)
    settlement_forex_df.to_excel(writer, sheet_name='Settlement (Forex)', index=False)
    settlement_indices_df.to_excel(writer, sheet_name='Settlement (Indices)', index=False)
    ll_hour_df.to_excel(writer, sheet_name='LL Hour', index=False)
    arbitrage_traders_summary.to_excel(writer, sheet_name='Arbitrage Traders Summary', index=False)
    positive_pnl_traders.to_excel(writer, sheet_name='Positive PnL Traders', index=False)
    top_5_symbols_by_pnl.to_excel(writer, sheet_name='Top 5 Symbols by PnL', index=False)
    top_5_symbols_by_trade_count.to_excel(writer, sheet_name='Top 5 Symbols by Trades', index=False)
    session_summary.to_excel(writer, sheet_name='Session Summary', index=False)


print(f"Session-wise analysis saved to {output_excel_file}")