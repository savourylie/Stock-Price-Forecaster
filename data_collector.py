from __future__ import division
import numpy as np
import pandas as pd
from os import listdir, getcwd, path
from IPython.display import display
from datetime import datetime, timedelta


class DataCollector:
	def __init__(self):
		dfSPY = pd.read_csv('stock_data/SPY.csv', index_col=0)

		self.stock_dict_original = {}
		# self.dfSPY500_2009 = pd.read_csv('sp500_2009.csv', header=None, usecols = [1])
		self.dfSPY500_2009 = pd.read_csv('dev.csv', header=None, usecols = [1])
		display(self.dfSPY500_2009)
  		self.dfSPY500_2009.columns = ['Symbol']
  		self.dataset_original = pd.DataFrame(index=dfSPY['2010-01-01':].index)

  		for symbol in self.dfSPY500_2009.loc[:, 'Symbol']:
 			df1 = pd.DataFrame(index=dfSPY.index)
  			df2 = pd.read_csv('stock_data/' + symbol + '.csv', index_col="Date", parse_dates=True, na_values=['nan'])
  			df1 = df1.join(df2)
  			df1 = df1.dropna()

  			self.stock_dict_original[symbol] = df1

  	def _df_indexer(self, df, current_date, date_span):
  		"""
  		Given current_date, slice df according to the date_span.
  		For example, for dataframe df, if current_date == '2010-01-01' and date_span == 252,
  		then df_indexer(df, current_date, date_span) will be a dataframe containing all the data
  		in the past 252 available days (missing dates don't count).
  		"""
  		index = 1
		counter = 0
		date_list = []

		while counter < date_span:
		    try:
		        df.loc[current_date - timedelta(days=index)]
		    except KeyError:
		        print("KeyError occuring...")
		        index += 1
		    else:
		        print("All looking good!")
		        date_list.append(current_date - timedelta(days=index))
		        index += 1
		        counter += 1

		# print(date_list)
		# print(len(date_list))
		return df.loc[date_list]

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
	print(dc.__df_indexer(GOOGL, datetime(2010, 1, 1), 10))

if __name__ == '__main__':
	test_run()

