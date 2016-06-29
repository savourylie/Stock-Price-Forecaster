from data_collector import DataCollector
from datetime import datetime
import pytest

class TestDataCollecor:
	@classmethod
	def setup_class(cls):
		print("Setting up CLASS {0}".format(cls.__name__))
		cls.d = DataCollector()

	def test_stock_list(self):
		assert self.d.dfSPY500_2009.loc[0, 'Symbol'] == 'A'
		assert self.d.dfSPY500_2009.loc[1, 'Symbol'] == 'AA'
		assert self.d.dfSPY500_2009.loc[10, 'Symbol'] == 'JEC'

	def test_stock_dict_original_value(self):
		assert self.d.stock_dict_original['GOOG'].get_value(datetime(2009, 1, 2), 'Adj Close') - 160 < 1
		assert self.d.stock_dict_original['A'].get_value(datetime(2009, 2, 17), 'Adj Close') - 12 < 1
		assert self.d.stock_dict_original['SCG'].get_value(datetime(2010, 4, 8), 'Open') - 39 < 1
		assert self.d.stock_dict_original['AKAM'].get_value(datetime(2011, 5, 2), 'High') - 35 < 1
		assert self.d.stock_dict_original['BF-B'].get_value(datetime(2012, 8, 20), 'Low') - 61 < 1
		assert self.d.stock_dict_original['CCL'].get_value(datetime(2013, 3, 13), 'Close') - 35 < 1
		assert self.d.stock_dict_original['DOW'].get_value(datetime(2014, 2, 24), 'Volume') - 8379400 < 1
		assert self.d.stock_dict_original['YHOO'].get_value(datetime(2015, 9, 28), 'Adj Close') - 27 < 1
		assert self.d.stock_dict_original['AAPL'].get_value(datetime(2016, 6, 6), 'Adj Close') - 26404100 < 1

	def test_dataset_original_length(self):
		assert len(self.d.dataset_original) == 1633

	def test_df_indexer(self):
		try:
			self.d._df_indexer(self.d.stock_dict_original['GOOG'], datetime(2009, 2, 2), 1).get_value(datetime(2009, 2, 2), 'Adj Close')
		except KeyError:
			pass
		else:
			pytest.fail("There's no error? Something's clearly wrong...")

		print(self.d._df_indexer(self.d.stock_dict_original['GOOG'], datetime(2013, 1, 21), 50))
		assert self.d._df_indexer(self.d.stock_dict_original['GOOG'], datetime(2009, 1, 21), 1).get_value(datetime(2009, 1, 20), 'Adj Close') - 141 < 1

	# def test_match_SPY_dates(self):
	# 	pass






