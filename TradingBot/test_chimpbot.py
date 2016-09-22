from chimpbot import ChimpBot

import numpy as np
import pandas as pd
from numbers import Number

# Test failing due to the newly added ETL process and change of gamma.

class TestChimp:
	def setup_method(self, method):
		print("Setting up METHOD {0}".format(method.__name__))
		self.dfEnv = pd.read_csv('data_train.csv', index_col=0, parse_dates=True, na_values = ['nan'])
		self.chimp = ChimpBot(self.dfEnv)