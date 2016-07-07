from __future__ import division
from data_loader import DataLoader
from forecaster import Forecaster
from datetime import datetime
import pytest
import numpy as np

class TestForecaster:
	@classmethod
	def setup_class(cls):
		print("Setting up CLASS {0}".format(cls.__name__))
		cls.d = DataLoader('dev.csv', '2009-01-01', '2016-06-30')
		cls.d.load_stock_data()

		cls.f = Forecaster(cls.d, 'GOOGL')

	def test_x_days(self):
		self.f.set_x_days(13)
		assert self.f.x_days == 13

	def test_make_label(self):
		x_days = 15
		self.f.set_x_days(x_days)
		assert np.abs(self.f.df_main.ix[datetime(2009, 1, 15), 'GOOGL' + str(x_days) + 'd'] - 185.825823) < 0.0001

		x_days = 20
		self.f.set_x_days(x_days)
		assert np.abs(self.f.df_main.ix[datetime(2009, 1, 15), 'GOOGL' + str(x_days) + 'd'] - 179.019015) < 0.0001
