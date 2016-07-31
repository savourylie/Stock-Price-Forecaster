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
		assert np.abs(self.f.df_main.ix[datetime(2010, 1, 4), 'GOOGL' + str(x_days) + 'd'] - 271.481479) < 0.0001

		x_days = 20
		self.f.set_x_days(x_days)
		assert np.abs(self.f.df_main.ix[datetime(2010, 1, 15), 'GOOGL' + str(x_days) + 'd'] - 270.920932) < 0.0001

	def test_drop_features(self):
		x_days = 20
		self.f.set_x_days(x_days)
		print(self.f.df_main.columns)
		# All SPYs gone!
		try:
			self.f.df_main['SPY']
		except KeyError:
			print("SPYs gone!")
		else:
			pytest.fail("SPY detected!")

	def test_splitting_features_and_label(self):
		x_days = 11
		self.f.set_x_days(x_days)

		try:
			self.f.data_X
		except AttributeError:
			pytest.fail("Where is my feature data?")
		else:
			assert self.f.data_X is not None

		try:
			self.f.data_X[self.f.label_name]
		except KeyError:
			print("Well done! Label gone!")
		else:
			pytest.fail("Ahhhhh!! Label is still here! Take it awayyyyy!!")

		try:
			self.f.data_y
		except AttributeError:
			pytest.fail("Where is my label?")
		else:
			assert self.f.data_y is not None

	def test_split_training_test_sets(self):
		x_days = 11
		self.f.set_x_days(x_days)

		try:
			self.f.test_X
		except AttributeError:
			pytest.fail("Where is my test data!!!!")

		try:
			self.f.test_y
		except AttributeError:
			pytest.fail("Where is my test data!!!!")

		try:
			self.f.train_X
		except AttributeError:
			pytest.fail("Where is my training data!!!!")

		try:
			self.f.train_y
		except AttributeError:
			pytest.fail("Where is my training data!!!!")

	def test_regressor(self):
		x_days = 5
		self.f.set_x_days(x_days)





