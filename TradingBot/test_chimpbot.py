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

	def test_dfEnv(self):
		print("Testing dfEnv...")
		print(self.chimp.env.head())
		print(type(self.chimp.env.index[0]))
		assert self.chimp.env.ix[0, '-1d_Vol1'] == 3
		assert self.chimp.env.ix[4, 'Trade Price'] - 29.407375 < 0.0001

	def test_init_cash(self):
		assert np.abs(self.chimp.cash - 1000) < 0.0001

	def test_init_share(self):
		assert np.abs(self.chimp.share - 0) < 0.0001

	def test_init_pv(self):
		assert np.abs(self.chimp.pv - 0) < 0.0001

	def test_init_pv_history_list(self):
		assert type(self.chimp.pv_history_list) == type([])
		assert len(self.chimp.pv_history_list) == 0

	def test_buy(self):
		self.chimp.buy(21)

		assert np.abs(self.chimp.share - 47) < 0.0001
		assert np.abs(self.chimp.pv - 987) < 0.0001
		assert np.abs(self.chimp.cash - 13) < 0.0001

	def test_buy_no_money(self):
		self.chimp.buy(21.1811)
		self.chimp.buy(19.8142)

		assert np.abs(self.chimp.share - 47) < 0.0001
		assert np.abs(self.chimp.pv - 931.2674) < 0.0001
		assert np.abs(self.chimp.cash - 4.4882) < 0.0001

	def test_buy_less_money(self):
		self.chimp.buy(78.3456)

		assert np.abs(self.chimp.share - 12) < 0.0001
		assert np.abs(self.chimp.pv - 940.1472) < 0.0001
		assert np.abs(self.chimp.cash - 59.8527) < 0.0001

		self.chimp.buy(19.8142)

		assert np.abs(self.chimp.share - 15) < 0.0001
		assert np.abs(self.chimp.pv - 297.2129) < 0.0001
		assert np.abs(self.chimp.cash - 0.4101) < 0.0001

	def test_sell_profit(self):
		self.chimp.buy(52)
		self.chimp.sell(59)

		assert np.abs(self.chimp.share - 0) < 0.0001
		assert np.abs(self.chimp.pv - 0) < 0.0001
		assert np.abs(self.chimp.cash - 1133) < 0.0001

	def test_sell_loss(self):
		self.chimp.buy(101.2230)
		self.chimp.sell(29.1458)

		assert np.abs(self.chimp.share - 0) < 0.0001
		assert np.abs(self.chimp.pv - 0) < 0.0001
		assert np.abs(self.chimp.cash - 351.3052) < 0.0001

	def test_sell_no_share(self):
		self.chimp.sell(29.1458)

		assert np.abs(self.chimp.share - 0) < 0.0001
		assert np.abs(self.chimp.pv - 0) < 0.0001
		assert np.abs(self.chimp.cash - 1000) < 0.0001

	def test_sell_even(self):
		self.chimp.buy(298.1234)
		self.chimp.sell(298.1234)

		assert np.abs(self.chimp.share - 0) < 0.0001
		assert np.abs(self.chimp.pv - 0) < 0.0001
		assert np.abs(self.chimp.cash - 1000) < 0.0001

	def test_hold(self):
		self.chimp.buy(123.4567)
		self.chimp.hold(288.1020)

		assert np.abs(self.chimp.cash - 12.3464) < 0.0001
		assert np.abs(self.chimp.share - 8) < 0.0001
		assert np.abs(self.chimp.pv - 2304.816) < 0.0001

		self.chimp.sell(111.2848)
		self.chimp.hold(211.1020)

		assert np.abs(self.chimp.cash - 902.6248) < 0.0001
		assert np.abs(self.chimp.share - 0) < 0.0001
		assert np.abs(self.chimp.pv - 0) < 0.0001

	def test_make_decision(self):
		self.chimp.buy(234.5678)
		pv = self.chimp.make_decision(291.0221)
		# print(pv)
		assert np.abs(pv - 4 * 291.0221) < 0.0001 or np.abs(pv - 0) < 0.0001


	def test_reset(self):
		self.chimp.buy(245.1902)
		self.chimp.reset()

		assert np.abs(self.chimp.cash - 1000) < 0.0001
		assert np.abs(self.chimp.share - 0) < 0.0001
		assert np.abs(self.chimp.pv - 0) < 0.0001