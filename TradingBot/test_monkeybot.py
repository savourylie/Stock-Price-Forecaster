from monkeybot import MonkeyBot

import numpy as np
import pandas as pd
from numbers import Number

# Test failing due to the newly added ETL process and change of gamma.

class TestMonkey:
	def setup_method(self, method):
		print("Setting up METHOD {0}".format(method.__name__))
		self.dfEnv = pd.read_csv('data_train.csv', index_col=0, parse_dates=True, na_values = ['nan'])
		self.monkey = MonkeyBot(self.dfEnv)

	def test_dfEnv(self):
		# q_dict is (next_way_point, tl, oa_oc, oa_lt, oa_rt, act): [q_value, t]
		print("Testing dfEnv...")
		print(self.monkey.env.head())
		print(type(self.monkey.env.index[0]))
		assert self.monkey.env.ix[0, '-1d_Vol1'] == 3
		assert self.monkey.env.ix[4, 'Trade Price'] - 29.407375 < 0.0001

	def test_init_cash(self):
		assert np.abs(self.monkey.cash - 1000) < 0.0001

	def test_init_share(self):
		assert np.abs(self.monkey.share - 0) < 0.0001

	def test_init_pv(self):
		assert np.abs(self.monkey.pv - 0) < 0.0001

	def test_init_pv_history_list(self):
		assert type(self.monkey.pv_history_list) == type([])
		assert len(self.monkey.pv_history_list) == 0

	def test_buy(self):
		self.monkey.buy(21)

		assert np.abs(self.monkey.share - 47) < 0.0001
		assert np.abs(self.monkey.pv - 987) < 0.0001
		assert np.abs(self.monkey.cash - 13) < 0.0001

	def test_buy_no_money(self):
		self.monkey.buy(21.1811)
		self.monkey.buy(19.8142)

		assert np.abs(self.monkey.share - 47) < 0.0001
		assert np.abs(self.monkey.pv - 931.2674) < 0.0001
		assert np.abs(self.monkey.cash - 4.4882) < 0.0001

	def test_buy_less_money(self):
		self.monkey.buy(78.3456)

		assert np.abs(self.monkey.share - 12) < 0.0001
		assert np.abs(self.monkey.pv - 940.1472) < 0.0001
		assert np.abs(self.monkey.cash - 59.8527) < 0.0001

		self.monkey.buy(19.8142)

		assert np.abs(self.monkey.share - 15) < 0.0001
		assert np.abs(self.monkey.pv - 297.2129) < 0.0001
		assert np.abs(self.monkey.cash - 0.4101) < 0.0001

	def test_sell_profit(self):
		self.monkey.buy(52)
		self.monkey.sell(59)

		assert np.abs(self.monkey.share - 0) < 0.0001
		assert np.abs(self.monkey.pv - 0) < 0.0001
		assert np.abs(self.monkey.cash - 1133) < 0.0001

	def test_sell_loss(self):
		self.monkey.buy(101.2230)
		self.monkey.sell(29.1458)

		assert np.abs(self.monkey.share - 0) < 0.0001
		assert np.abs(self.monkey.pv - 0) < 0.0001
		assert np.abs(self.monkey.cash - 351.3052) < 0.0001

	def test_sell_no_share(self):
		self.monkey.sell(29.1458)

		assert np.abs(self.monkey.share - 0) < 0.0001
		assert np.abs(self.monkey.pv - 0) < 0.0001
		assert np.abs(self.monkey.cash - 1000) < 0.0001

	def test_sell_even(self):
		self.monkey.buy(298.1234)
		self.monkey.sell(298.1234)

		assert np.abs(self.monkey.share - 0) < 0.0001
		assert np.abs(self.monkey.pv - 0) < 0.0001
		assert np.abs(self.monkey.cash - 1000) < 0.0001

	def test_hold(self):
		self.monkey.buy(123.4567)
		self.monkey.hold(288.1020)

		assert np.abs(self.monkey.cash - 12.3464) < 0.0001
		assert np.abs(self.monkey.share - 8) < 0.0001
		assert np.abs(self.monkey.pv - 2304.816) < 0.0001

		self.monkey.sell(111.2848)
		self.monkey.hold(211.1020)

		assert np.abs(self.monkey.cash - 902.6248) < 0.0001
		assert np.abs(self.monkey.share - 0) < 0.0001
		assert np.abs(self.monkey.pv - 0) < 0.0001

	def test_make_decision(self):
		self.monkey.buy(234.5678)
		assert np.abs(self.monkey.make_decision(291.0221) - 4 * 291.0221) < 0.0001 or np.abs(self.monkey.make_decision(291.0221) - 0) < 0.0001


	def test_reset(self):
		self.monkey.buy(245.1902)
		self.monkey.reset()

		assert np.abs(self.monkey.cash - 1000) < 0.0001
		assert np.abs(self.monkey.share - 0) < 0.0001
		assert np.abs(self.monkey.pv - 0) < 0.0001




