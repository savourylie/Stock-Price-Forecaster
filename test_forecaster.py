from __future__ import division
from data_loader import DataLoader
from ETL import ETL
from datetime import datetime
import pytest
import numpy as np

class TestDataLoader:
	@classmethod
	def setup_class(cls):
		print("Setting up CLASS {0}".format(cls.__name__))
		cls.d = DataLoader('dev.csv', '2009-01-01', '2016-06-30')

	def test__load_stock_names(self):
		assert self.d.df_stock_names.ix[0, 'Symbol'] == 'A'
		assert self.d.df_stock_names.ix[13, 'Symbol'] == 'YHOO'
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
		cls.d.load_stock_data()
		cls.e = ETL(cls.d, 'GOOGL')


	def test_avg_runup(self):
		e1 = ETL(self.d, 'YHOO')
		assert np.abs(e1.df_temp.ix[datetime(2016, 5, 19), 'YHOO_Avg_Runup'] - (-0.022897)) < 0.0001

		e2 = ETL(self.d, 'MSFT')
		assert np.abs(e2.df_temp.ix[datetime(2016, 6, 8), 'MSFT_Avg_Runup'] - 0.030256) < 0.0001

	def test_daily_return(self):
		e1 = ETL(self.d, 'A')
		assert np.abs(e1.df_temp.ix[datetime(2009, 2, 13), 'A_return'] - -0.008588) < 0.0001

		e2 = ETL(self.d, 'AA')
		assert np.abs(e2.df_temp.ix[datetime(2016, 6, 28), 'AA_return'] - 0.025275) < 0.0001

	def test_stock_mean63d(self):
		e1 = ETL(self.d, 'AAPL')
		assert np.abs(e1.df_temp.ix[datetime(2016, 5, 25), 'AAPL_Mean63d'] - 0.000676)  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2016, 6, 1), 'AAPL_Std63d'] - 0.014677)  < 0.0001

		e2 = ETL(self.d, 'AKAM')
		assert np.abs(e2.df_temp.ix[datetime(2014, 7, 25), 'AKAM_Mean63d'] - 0.002280) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2013, 1, 7), 'AKAM_Std63d'] - 0.022000)  < 0.0001

	def test_cov63d(self):
		e1 = ETL(self.d, 'BF-B')
		assert np.abs(e1.df_temp.ix[datetime(2013, 1, 24), 'BF-B_Cov63d'] - 0.000048)  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2013, 2, 5), 'BF-B_Cov63d'] - 0.000041)  < 0.0001

		e2 = ETL(self.d, 'CCL')
		assert np.abs(e2.df_temp.ix[datetime(2014, 7, 25), 'CCL_Cov63d'] - 0.000027) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2014, 8, 25), 'CCL_Cov63d'] - 0.000026) < 0.0001

	def test_beta63d(self):
		e1 = ETL(self.d, 'DOW')
		assert np.abs(e1.df_temp.ix[datetime(2013, 1, 9), 'DOW_Beta'] - 0.343213)  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2013, 1, 16), 'DOW_Beta'] - 0.332140)  < 0.0001

		e2 = ETL(self.d, 'GOOG')
		assert np.abs(e2.df_temp.ix[datetime(2013, 1, 22), 'GOOG_Beta'] - 0.280741) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2014, 9, 2), 'GOOG_Beta'] - 0.367537) < 0.0001

	def test_ema(self):
		e1 = ETL(self.d, 'GOOGL')
		assert np.abs(e1.df_temp.ix[datetime(2010, 8, 26), 'GOOGL_EMA'] - 247.943286)  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2012, 11, 12), 'GOOGL_EMA'] - 338.467381)  < 0.0001

		e2 = ETL(self.d, 'IBM')
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 4), 'IBM_EMA'] - 110.986086) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2010, 9, 7), 'IBM_EMA'] - 111.070117) < 0.0001

	def test_mma(self):
		e1 = ETL(self.d, 'JEC')
		assert np.abs(e1.df_temp.ix[datetime(2010, 8, 26), 'JEC_MMA'] - 40.437522)  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2012, 11, 12), 'JEC_MMA'] - 40.214954)  < 0.0001

		e2 = ETL(self.d, 'MSFT')
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 2), 'MSFT_MMA'] - 22.728518) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 11), 'MSFT_MMA'] - 22.654056) < 0.0001

	def test_sma(self):
		e1 = ETL(self.d, 'SCG')
		assert np.abs(e1.df_temp.ix[datetime(2010, 8, 26), 'SCG_SMA'] - 29.149889)  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2012, 11, 12), 'SCG_SMA'] - 41.288754)  < 0.0001

		e2 = ETL(self.d, 'YHOO')
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 2), 'YHOO_SMA'] - 15.814554) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 11), 'YHOO_SMA'] - 15.653366) < 0.0001

	def test_sma_momentum(self):
		e1 = ETL(self.d, 'GWW')
		assert np.abs(e1.df_temp.ix[datetime(2010, 8, 26), 'GWW_SMA_Momentum'] - (-4.818233))  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2012, 11, 12), 'GWW_SMA_Momentum'] - 12.634321)  < 0.0001

		e2 = ETL(self.d, 'HAL')
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 2), 'HAL_SMA_Momentum'] - (-0.067255)) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 11), 'HAL_SMA_Momentum'] - (-2.495234)) < 0.0001

	def test_vol_momentum(self):
		e1 = ETL(self.d, 'HAR')
		assert np.abs(e1.df_temp.ix[datetime(2010, 9, 10), 'HAR_Vol_Momentum'] - (-61882700))  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2010, 9, 13), 'HAR_Vol_Momentum'] - (-18180000))  < 0.0001

		e2 = ETL(self.d, 'HAS')
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 2), 'HAS_Vol_Momentum'] - (-78002300.0)) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 11), 'HAS_Vol_Momentum'] - (-63306800.0)) < 0.0001

	def test_vol_momentum_r1(self):
		e1 = ETL(self.d, 'HBAN')
		assert np.abs(e1.df_temp.ix[datetime(2010, 9, 10), 'HBAN_p_real1'] - 0)  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2010, 9, 13), 'HBAN_p_real1'] - 1)  < 0.0001

		e2 = ETL(self.d, 'HCP')
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 2), 'HCP_p_real1'] - 1) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2010, 8, 11), 'HCP_p_real1'] - 1) < 0.0001

	def test_vol_momentum_r2(self):
		e1 = ETL(self.d, 'HD')
		assert np.abs(e1.df_temp.ix[datetime(2011, 1, 10), 'HD_p_real2'] - 0)  < 0.0001
		assert np.abs(e1.df_temp.ix[datetime(2011, 1, 13), 'HD_p_real2'] - 0)  < 0.0001

		e2 = ETL(self.d, 'HES')
		assert np.abs(e2.df_temp.ix[datetime(2012, 11, 13), 'HES_p_real2'] - 0) < 0.0001
		assert np.abs(e2.df_temp.ix[datetime(2012, 11, 15), 'HES_p_real2'] - 1) < 0.0001

	def test_SR63d(self):
		assert np.abs(self.e.df_temp.ix[datetime(2011, 1, 26), 'GOOGL_SR63d'] - 0.001164)  < 0.0001
		assert np.abs(self.e.df_temp.ix[datetime(2011, 2, 14), 'GOOGL_SR63d'] - 0.059467)  < 0.0001

	def test_STLFSI(self):
		assert np.abs(self.e.df_temp.ix[datetime(2011, 1, 14), 'STLFSI'] - (-0.811))  < 0.0001
		assert np.abs(self.e.df_temp.ix[datetime(2012, 11, 30), 'STLFSI'] - (-1.411))  < 0.0001
