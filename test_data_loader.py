from data_loader import DataLoader
from datetime import datetime
import pytest

class TestDataLoader:
	@classmethod
	def setup_class(cls):
		print("Setting up CLASS {0}".format(cls.__name__))
		cls.d = DataLoader('dev.csv', '2009-01-01', '2016-06-30')

	def test__load_stock_names(self):
		assert self.d.df_stock_names.ix[0, 'Symbol'] == 'A'
		assert self.d.df_stock_names.ix[len(self.d.df_stock_names) - 1, 'Symbol'] == 'YHOO'
		assert self.d.df_stock_names.ix[3, 'Symbol'] == 'AKAM'

	def test_stock_dict_values(self):
		self.d.load_stock_data()
		assert self.d.stock_dict_original['GOOGL'].ix[datetime(2009, 1, 2), 'SPY'] - 79.60265 < 0.001
		assert self.d.stock_dict_original['IBM'].ix[datetime(2009, 5, 21), 'SPY'] - 76.938255 < 0.001
		assert self.d.stock_dict_original['JEC'].ix[datetime(2011, 5, 10), 'JEC'] - 48.610001 < 0.001

class TestETL:
	@classmethod
	def setup_class(cls):
		print("Setting up CLASS {0}".format(cls.__name__))
		cls.d = DataLoader('dev.csv', '2009-01-01', '2016-06-30')
		cls.e = ETL()



