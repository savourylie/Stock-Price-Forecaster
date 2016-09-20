from __future__ import division
from etl_spy import ETL_SPY
from datetime import datetime
import pytest
import numpy as np

class TestETLSPY:
	@classmethod
	def setup_class(cls):
		print("Setting up CLASS {0}".format(cls.__name__))
		cls.e = ETL_SPY('allSPY.csv')

	def test_dfmain(self):
		assert self.e.dfMain is None

	def test_load_data(self):
		self.e.load_data()

