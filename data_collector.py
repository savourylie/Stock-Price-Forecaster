from __future__ import division
import numpy as np
import pandas as pd
from os import listdir, getcwd, path
from IPython.display import display


class DataCollector:
	def __init__(self):
		dfSPY = pd.read_csv('stock_data/SPY.csv', index_col=0)

		self.stock_dict_original = {}
		self.dfSPY500_2009 = pd.read_csv('sp500_2009.csv', header=None, usecols = [1])
  		self.dfSPY500_2009.columns = ['Symbol']
  		self.dataset_original = pd.DataFrame(index=dfSPY['2010-01-01':].index)

  		for symbol in self.dfSPY500_2009.loc[:, 'Symbol']:
 			df1 = pd.DataFrame(index=dfSPY.index)
  			df2 = pd.read_csv('stock_data/' + symbol + '.csv', index_col="Date", parse_dates=True, na_values=['nan'])
  			df1 = df1.join(df2)
  			df1 = df1.dropna()

  			self.stock_dict_original[symbol] = df1


def test_run():
	print('This is a test.')

if __name__ == '__main__':
	test_run()

