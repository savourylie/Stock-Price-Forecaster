from __future__ import division
import numpy as np
import pandas as pd
from os import listdir, getcwd, path
from IPython.display import display
from datetime import datetime, timedelta


class DataCollector:
	def __init__(self):
		self.stock_dict_original = {} # stock symbol and stock data pair

		# self.dfSPY500_2009 = pd.read_csv('sp500_2009.csv', header=None, usecols = [1])
		self.dfSPY500_2009 = pd.read_csv('dev.csv', header=None, usecols = [1])
		display(self.dfSPY500_2009)
  		self.dfSPY500_2009.columns = ['Symbol'] # assign 'Symbol' to the column of stock symbols of the dataframe

  		dfSPY = pd.read_csv('stock_data/SPY.csv', index_col=0) # for getting the trade days
  		for symbol in self.dfSPY500_2009.loc[:, 'Symbol']:
 			df1 = pd.DataFrame(index=dfSPY.index)
  			df2 = pd.read_csv('stock_data/' + symbol + '.csv', index_col="Date", parse_dates=True, na_values=['nan'])
  			df1 = df1.join(df2)
  			df1 = df1.dropna()

  			self.stock_dict_original[symbol] = df1

  		self.stock_dict = {}

  		self.dataset_original_columns = ['Symbol', 'Date', \
  		'open_1d', 'high_1d', 'low_1d', 'volume_1d', 'adjclose_1d', \
  		'open_3d_mean', 'high_3d_mean', 'low_3d_mean', 'volume_3d_mean', 'adjclose_3d_mean', \
		'open_3d_max', 'high_3d_max', 'low_3d_max', 'volume_3d_max', 'adjclose_3d_max', \
		'open_3d_min', 'high_3d_min', 'low_3d_min', 'volume_3d_min', 'adjclose_3d_min', \
		'open_3d_std', 'high_3d_std', 'low_3d_std', 'volume_3d_std', 'adjclose_3d_std', \
		'open_5d_mean', 'high_5d_mean', 'low_5d_mean', 'volume_5d_mean', 'adjclose_5d_mean', \
		'open_5d_max', 'high_5d_max', 'low_5d_max', 'adjclose_5d_max', 'volume_5d_max', \
		'open_5d_min', 'high_5d_min', 'low_5d_min', 'adjclose_5d_min', 'volume_5d_min', \
		'open_5d_std', 'high_5d_std', 'low_5d_std', 'adjclose_5d_std', 'volume_5d_std', \
		'open_10d_mean', 'high_10d_mean', 'low_10d_mean', 'volume_10d_mean', 'adjclose_10d_mean', \
		'open_10d_max', 'high_10d_max', 'low_10d_max', 'volume_10d_max', 'adjclose_10d_max', \
		'open_10d_min', 'high_10d_min', 'low_10d_min', 'volume_10d_min', 'adjclose_10d_min', \
		'open_10d_std', 'high_10d_std', 'low_10d_std', 'volume_10d_std', 'adjclose_10d_std', \
		'open_1m_mean', 'high_1m_mean', 'low_1m_mean', 'volume_1m_mean', 'adjclose_1m_mean', \
		'open_1m_max', 'high_1m_max', 'low_1m_max', 'volume_1m_max', 'adjclose_1m_max', \
		'open_1m_min', 'high_1m_min', 'low_1m_min', 'volume_1m_min', 'adjclose_1m_min', \
		'open_1m_std', 'high_1m_std', 'low_1m_std', 'volume_1m_std', 'adjclose_1m_std', \
		'open_3m_mean', 'high_3m_mean', 'low_3m_mean', 'volume_3m_mean', 'adjclose_3m_mean', \
		'open_3m_max', 'high_3m_max', 'low_3m_max', 'volume_3m_max', 'adjclose_3m_max', \
		'open_3m_min', 'high_3m_min', 'low_3m_min', 'volume_3m_min', 'adjclose_3m_min', \
		'open_3m_std', 'high_3m_std', 'low_3m_std', 'volume_3m_std', 'adjclose_3m_std', \
		'open_6m_mean', 'high_6m_mean', 'low_6m_mean', 'volume_6m_mean', 'adjclose_6m_mean', \
		'open_6m_max', 'high_6m_max', 'low_6m_max', 'volume_6m_max', 'adjclose_6m_max', \
		'open_6m_min', 'high_6m_min', 'low_6m_min', 'volume_6m_min', 'adjclose_6m_min', \
		'open_6m_std', 'high_6m_std', 'low_6m_std', 'volume_6m_std', 'adjclose_6m_std', \
		'open_1y_mean', 'high_1y_mean', 'low_1y_mean', 'volume_1y_mean', 'adjclose_1y_mean', \
		'open_1y_max', 'high_1y_max', 'low_1y_max', 'volume_1y_max', 'adjclose_1y_max', \
		'open_1y_min', 'high_1y_min', 'low_1y_min', 'volume_1y_min', 'adjclose_1y_min', \
		'open_1y_std', 'high_1y_std', 'low_1y_std', 'volume_1y_std', 'adjclose_1y_std', \
		'Adj Close', 'RiseOrFall'
		]

  		self.dataset_original = pd.DataFrame(index=dfSPY['2010-01-01':].index)

  		df = pd.DataFrame(columns=self.dataset_original_columns)
  		symbol = 'IBM'
		date = datetime(2010, 1, 4) # Dataset start date
		dfTemp = self.stock_dict_original[symbol][date:].copy()

		for index, row in dfTemp.iterrows():

			df.ix[index, 'Symbol'] = symbol
			df.ix[index, 'Date'] = index

			# 1-Day Features
			df.ix[index, ['open_1d', 'high_1d', 'low_1d', 'volume_1d', 'adjclose_1d']] = self._df_indexer(symbol, 'adjusted', index, 1).mean().tolist()

			# 3-Day Features
			df.ix[index, ['open_3d_mean', 'high_3d_mean', 'low_3d_mean', 'volume_3d_mean', 'adjclose_3d_mean']] = self._df_indexer(symbol, 'adjusted', index, 3).mean().tolist()
			df.ix[index, ['open_3d_max', 'high_3d_max', 'low_3d_max', 'volume_3d_max', 'adjclose_3d_max']] = self._df_indexer(symbol, 'adjusted', index, 3).max().tolist()
			df.ix[index, ['open_3d_min', 'high_3d_min', 'low_3d_min', 'volume_3d_min', 'adjclose_3d_min']] = self._df_indexer(symbol, 'adjusted', index, 3).min().tolist()
			df.ix[index, ['open_3d_std', 'high_3d_std', 'low_3d_std', 'volume_3d_std', 'adjclose_3d_std']] = self._df_indexer(symbol, 'adjusted', index, 3).std().tolist()

			# 5-Day Features
			df.ix[index, ['open_5d_mean', 'high_5d_mean', 'low_5d_mean', 'volume_5d_mean', 'adjclose_5d_mean']] = self._df_indexer(symbol, 'adjusted', index, 5).mean().tolist()
			df.ix[index, ['open_5d_max', 'high_5d_max', 'low_5d_max', 'volume_5d_max', 'adjclose_5d_max']] = self._df_indexer(symbol, 'adjusted', index, 5).max().tolist()
			df.ix[index, ['open_5d_min', 'high_5d_min', 'low_5d_min', 'volume_5d_min', 'adjclose_5d_min']] = self._df_indexer(symbol, 'adjusted', index, 5).min().tolist()
			df.ix[index, ['open_5d_std', 'high_5d_std', 'low_5d_std', 'volume_5d_std', 'adjclose_5d_std']] = self._df_indexer(symbol, 'adjusted', index, 5).std().tolist()

			# 10-Day Features
			df.ix[index, ['open_10d_mean', 'high_10d_mean', 'low_10d_mean', 'volume_10d_mean', 'adjclose_10d_mean']] = self._df_indexer(symbol, 'adjusted', index, 10).mean().tolist()
			df.ix[index, ['open_10d_max', 'high_10d_max', 'low_10d_max', 'volume_10d_max', 'adjclose_10d_max']] = self._df_indexer(symbol, 'adjusted', index, 10).max().tolist()
			df.ix[index, ['open_10d_min', 'high_10d_min', 'low_10d_min', 'volume_10d_min', 'adjclose_10d_min']] = self._df_indexer(symbol, 'adjusted', index, 10).min().tolist()
			df.ix[index, ['open_10d_std', 'high_10d_std', 'low_10d_std', 'volume_10d_std', 'adjclose_10d_std']] = self._df_indexer(symbol, 'adjusted', index, 10).std().tolist()

			# 1-Month Features
			df.ix[index, ['open_1m_mean', 'high_1m_mean', 'low_1m_mean', 'volume_1m_mean', 'adjclose_1m_mean']] = self._df_indexer(symbol, 'adjusted', index, 21).mean().tolist()
			df.ix[index, ['open_1m_max', 'high_1m_max', 'low_1m_max', 'volume_1m_max', 'adjclose_1m_max']] = self._df_indexer(symbol, 'adjusted', index, 21).max().tolist()
			df.ix[index, ['open_1m_min', 'high_1m_min', 'low_1m_min', 'volume_1m_min', 'adjclose_1m_min']] = self._df_indexer(symbol, 'adjusted', index, 21).min().tolist()
			df.ix[index, ['open_1m_std', 'high_1m_std', 'low_1m_std', 'volume_1m_std', 'adjclose_1m_std']] = self._df_indexer(symbol, 'adjusted', index, 21).std().tolist()

			# 3-Month Features
			df.ix[index, ['open_3m_mean', 'high_3m_mean', 'low_3m_mean', 'volume_3m_mean', 'adjclose_3m_mean']] = self._df_indexer(symbol, 'adjusted', index, 63).mean().tolist()
			df.ix[index, ['open_3m_max', 'high_3m_max', 'low_3m_max', 'volume_3m_max', 'adjclose_3m_max']] = self._df_indexer(symbol, 'adjusted', index, 63).max().tolist()
			df.ix[index, ['open_3m_min', 'high_3m_min', 'low_3m_min', 'volume_3m_min', 'adjclose_3m_min']] = self._df_indexer(symbol, 'adjusted', index, 63).min().tolist()
			df.ix[index, ['open_3m_std', 'high_3m_std', 'low_3m_std', 'volume_3m_std', 'adjclose_3m_std']] = self._df_indexer(symbol, 'adjusted', index, 63).std().tolist()

			# 6-Month Features
			df.ix[index, ['open_6m_mean', 'high_6m_mean', 'low_6m_mean', 'volume_6m_mean', 'adjclose_6m_mean']] = self._df_indexer(symbol, 'adjusted', index, 126).mean().tolist()
			df.ix[index, ['open_6m_max', 'high_6m_max', 'low_6m_max', 'volume_6m_max', 'adjclose_6m_max']] = self._df_indexer(symbol, 'adjusted', index, 126).max().tolist()
			df.ix[index, ['open_6m_min', 'high_6m_min', 'low_6m_min', 'volume_6m_min', 'adjclose_6m_min']] = self._df_indexer(symbol, 'adjusted', index, 126).min().tolist()
			df.ix[index, ['open_6m_std', 'high_6m_std', 'low_6m_std', 'volume_6m_std', 'adjclose_6m_std']] = self._df_indexer(symbol, 'adjusted', index, 126).std().tolist()

			# 1-Year Features
			df.ix[index, ['open_1y_mean', 'high_1y_mean', 'low_1y_mean', 'volume_1y_mean', 'adjclose_1y_mean']] = self._df_indexer(symbol, 'adjusted', index, 252).mean().tolist()
			df.ix[index, ['open_1y_max', 'high_1y_max', 'low_1y_max', 'volume_1y_max', 'adjclose_1y_max']] = self._df_indexer(symbol, 'adjusted', index, 252).max().tolist()
			df.ix[index, ['open_1y_min', 'high_1y_min', 'low_1y_min', 'volume_1y_min', 'adjclose_1y_min']] = self._df_indexer(symbol, 'adjusted', index, 252).min().tolist()
			df.ix[index, ['open_1y_std', 'high_1y_std', 'low_1y_std', 'volume_1y_std', 'adjclose_1y_std']] = self._df_indexer(symbol, 'adjusted', index, 252).std().tolist()

			df.ix[index, 'Adj Close'] = dfTemp.ix[index, 'Adj Close']

			if df.ix[index, 'Adj Close'] - self._df_indexer(symbol, 'adjusted', index, 1).ix[0, 'Adj Close'] >= 0:
			    df.ix[index, 'RiseOrFall'] = 1
			else:
			    df.ix[index, 'RiseOrFall'] = 0

			print("IBM")
			print("Date {}".format(index) + " clear!")

		df.to_csv("IBM.csv")

  	def _df_indexer(self, symbol, data_source, current_date, date_span):
  		"""
  		Given current_date, slice df according to the date_span.
  		For example, for dataframe df, if current_date == '2010-01-01' and date_span == 252,
  		then df_indexer(symbol, data, current_date, date_span) will be a dataframe containing all the data
  		in the past 252 available days (missing dates don't count).
  		"""
  		if data_source == 'original':
  			data = self.stock_dict_original[symbol]

  		elif data_source == 'adjusted':
  			data = self._stock_price_adjuster(symbol)

  		else:
			raise ValueError("Hmmm...something's fishy about the data source...")

  		index = 1
		counter = 0
		date_list = []

		while counter < date_span:
		    try:
		        data.loc[current_date - timedelta(days=index)]
		    except KeyError:
		        # print("KeyError occuring...")
		        index += 1
		    else:
		        # print("All looking good!")
		        date_list.append(current_date - timedelta(days=index))
		        index += 1
		        counter += 1

		return data.loc[date_list]

	def _stock_price_adjuster(self, symbol):
		dfTemp = self.stock_dict_original[symbol].copy()

		dfTemp['Adj Open'] = (dfTemp['Adj Close'] / dfTemp['Close']) * dfTemp['Open']
		dfTemp['Adj High'] = (dfTemp['Adj Close'] / dfTemp['Close']) * dfTemp['High']
		dfTemp['Adj Low'] = (dfTemp['Adj Close'] / dfTemp['Close']) * dfTemp['Low']
		dfTemp['Adj Volume'] = (dfTemp['Adj Close'] / dfTemp['Close']) * dfTemp['Volume']

		dfTemp.drop(['Open', 'High', 'Low', 'Volume', 'Close'], axis=1, inplace=True)

		return dfTemp



def test_run():
	print('===== Test Run =====')
	dc = DataCollector()

	dfSPY = pd.read_csv('stock_data/SPY.csv', index_col=0)
	df1 = pd.DataFrame(index=dfSPY.index)
	df2 = pd.read_csv('stock_data/' + 'GOOGL' + '.csv', index_col="Date", parse_dates=True, na_values=['nan'])
	df1 = df1.join(df2)
	df1 = df1.dropna()

	GOOGL = df1

	# display(GOOGL)
	print(dc._df_indexer(GOOGL, datetime(2010, 1, 1), 10))

if __name__ == '__main__':
	test_run()

