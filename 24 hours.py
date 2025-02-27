import time
import pandas as pd
from sqlalchemy import create_engine

# Database connection details
db_config = {
    'user': ****
    'password': ****
    'host': ****
    'database': ****
    'port': ****
}

# Create the connection string
connection_string = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

# Create the SQLAlchemy engine
engine = create_engine(connection_string)

# Input times (in "YYYY-MM-DD HH:MM:SS" format)
start_time = "2025-02-24 00:00:00"
end_time = "2025-02-24 23:59:59"

# Asset class lists
crypto_list = ["ADAUSD", "BCHUSD", "BTCUSD", "DOGUSD", "ETHUSD", "LNKUSD", "LTCUSD", "XLMUSD", "XMRUSD", "XRPUSD"]
commodities_list = ["UKOUSD", "USOUSD", "XAUUSD", "XAGUSD", "XPTUSD"]
indices_list = ["AUS200", "HK50", "EUSTX50", "FRA40", "GER30", "NTH25", "SWI20", "AUDUSD", "SPX500", "UK100", "US30", "JP225", "US2000", "NDX100"]
forex_list = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDSGD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD",
              "EURCAD", "EURCHF", "EURGBP", "EURHKD",
              "EURHUF", "EURJPY", "EURNOK", "EURNZD", "EURSGD", "EURTRY", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF",
              "GBPJPY", "GBPNZD", "GBPSGD", "GBPUSD",
              "MXNJPY", "NOKJPY", "NZDCAD", "NZDCHF", "NZDJPY", "NZDSGD", "NZDUSD", "SGDJPY", "USDCAD", "USDCHF",
              "USDCNH", "USDDKK", "USDHKD", "USDHUF",
              "USDJPY", "USDMXN", "USDNOK", "USDPLN", "USDSEK", "USDSGD", "USDTRY", "USDZAR", "ZARJPY", "CADEUR"]

# Mapping symbols to asset classes
def get_asset_class(symbol):
    if symbol in crypto_list:
        return 'Crypto'
    elif symbol in commodities_list:
        return 'Commodities'
    elif symbol in indices_list:
        return 'Indices'
    elif symbol in forex_list:
        return 'Forex'
    else:
        return 'Unknown'

# Start measuring time
script_start_time = time.time()

try:
    print("Fetching trades data...")
    trades_query = f"""
    SELECT id, open_time, close_time, symbol, open_price, close_price, login, volume, close_time_str, 
           commission, digits, open_time_str, profit, reason, sl, swap, ticket, tp, type_str, created_at,
           CASE 
               WHEN login LIKE '70%' OR login LIKE '30%' THEN lots
               ELSE volume / 100
           END AS FinalLot
    FROM trades
    WHERE (open_time BETWEEN UNIX_TIMESTAMP('{start_time}') AND UNIX_TIMESTAMP('{end_time}'))
       OR (close_time BETWEEN UNIX_TIMESTAMP('{start_time}') AND UNIX_TIMESTAMP('{end_time}'));
    """
    trades_df = pd.read_sql(trades_query, engine)
    print(f"Fetched {len(trades_df)} trades.")

    if trades_df.empty:
        print("No trades found within the specified time range.")
    else:
        # Add asset class information to trades_df
        trades_df['asset_class'] = trades_df['symbol'].apply(get_asset_class)

        print("Fetching accounts data...")
        login_ids = tuple(int(x) for x in trades_df['login'].unique())
        accounts_query = f"""
        SELECT id AS account_id, login, type AS type_account, equity, breachedby, customer_id, starting_balance FROM accounts 
        WHERE login IN {login_ids};
        """
        accounts_df = pd.read_sql(accounts_query, engine)
        print(f"Fetched {len(accounts_df)} accounts.")

        combined_df = pd.merge(trades_df, accounts_df, on='login', suffixes=('_trade', '_account'))

        print("Fetching customers data...")
        customer_ids = tuple(int(x) for x in combined_df['customer_id'].unique())
        customers_query = f"""
        SELECT id AS customer_id, email, country_id FROM customers 
        WHERE id IN {customer_ids};
        """
        customers_df = pd.read_sql(customers_query, engine)
        print(f"Fetched {len(customers_df)} customers.")

        combined_df = pd.merge(combined_df, customers_df, on='customer_id', suffixes=('', '_customer'))

        print("Fetching country names...")
        country_ids = tuple(int(x) for x in customers_df['country_id'].unique())
        countries_query = f"""
        SELECT id AS country_id, name AS country_name FROM countries 
        WHERE id IN {country_ids};
        """
        countries_df = pd.read_sql(countries_query, engine)
        print(f"Fetched {len(countries_df)} countries.")

        final_df = pd.merge(combined_df, countries_df, on='country_id', suffixes=('', '_country'))

        # Add starting_balance as the last column
        final_df['starting_balance'] = final_df['starting_balance']

        columns_to_drop = ['account_id', 'updated_at', 'deleted_at', 'customer_id', 'country_id']
        final_df = final_df.drop(columns=[col for col in columns_to_drop if col in final_df.columns])

        csv_file_name = "trades_with_asset_classes.csv"
        final_df.to_csv(csv_file_name, index=False)
        print(f"Data has been written to {csv_file_name}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Ensure the connection is closed
    engine.dispose()
    print("Database connection closed.")

# End measuring time
script_end_time = time.time()
print(f"Time taken to run the script: {script_end_time - script_start_time} seconds")
