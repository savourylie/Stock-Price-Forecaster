from __future__ import division
import pandas as pd

class DataLoader:
    """
    The DataLoader object is initialized with data source, and date range of interest and includes attributes
    of stock names and minimal processed data stored in the stock_dict_original.

    Data loaded by the DataLoader can then be further processed in an ETL object.
    """
    # Load SPY data
    dfSPY = pd.read_csv('stock_data/SPY.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'], na_values = ['nan'])
    # Load FSI data
    dfFSI = pd.read_csv('STLFSI.csv', index_col='DATE', parse_dates=True, na_values = ['nan'])

    def __init__(self, target_stocks_csv, start_date_str, end_date_str):
        # Declare properties
        self.df_stock_names = None
        self.df_base = None
        self.stock_dict_original = {} # stock symbol and stock data pair

        # Set up base dataframe using SPY and the target date range
        self._set_up_base_df(start_date_str, end_date_str)

        # Get list of stock names available
        self._load_stock_names(target_stocks_csv)

    def _load_stock_names(self, target_stocks_csv):
        """This function loads the csv file and store it into self.df_stock_names
        """
        self.df_stock_names = pd.read_csv(target_stocks_csv, header=None, usecols = [1])
        self.df_stock_names.columns = ['Symbol']

    def _set_up_base_df(self, start_date_str, end_date_str):
        date_range = pd.date_range(start_date_str, end_date_str)
        self.df_base = pd.DataFrame(index=date_range)
        self.dfSPY = self.dfSPY.rename(columns={'Adj Close': 'SPY'})

        self.df_base = self.df_base.join(self.dfSPY)
        self.df_base = self.df_base.dropna()

        self.df_base = self.df_base.join(self.dfFSI)
        self.df_base['STLFSI'].fillna(method='ffill', inplace=True)

    def load_stock_data(self):
        for symbol in self.df_stock_names.loc[:, 'Symbol']:
            df_temp = pd.read_csv('stock_data/' + symbol + '.csv', index_col="Date", parse_dates=True, usecols = ['Date', 'Volume', 'Adj Close'], na_values=['nan'])
            df_temp = df_temp.rename(columns={'Volume': symbol + '_Vol', 'Adj Close': symbol})

            # Forward/Back fill missing values
            df_temp.fillna(method='ffill', inplace=True)
            df_temp.fillna(method='bfill', inplace=True)

            df1 = self.df_base.join(df_temp, how='left')
            self.stock_dict_original[symbol] = df1



