from chimpbot import ChimpBot

import numpy as np
import pandas as pd
from numbers import Number
import random
import types
from datetime import datetime, timedelta

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
		now_row = list(self.chimp.now_row)
		now_row.pop() # disregard the Trade Price
		now_row.append(self.chimp.yes_share())
		now_row.append('Buy')
		self.chimp.q_dict[tuple(now_row)] = [123, 1]

		now_row.pop()
		now_row.append('Hold')
		self.chimp.q_dict[tuple(now_row)] = [223, 9]

		now_row.pop()
		now_row.append('Sell')
		self.chimp.q_dict[tuple(now_row)] = [323, 4]

		assert self.chimp.make_decision(self.chimp.now_row) == 'Sell'

		# Next row
		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()

		now_row = list(self.chimp.now_row)
		now_row.pop() # disregard the Trade Price
		now_row.append(self.chimp.yes_share())
		now_row.append('Buy')
		self.chimp.q_dict[tuple(now_row)] = [13, 1]

		now_row.pop()
		now_row.append('Hold')
		self.chimp.q_dict[tuple(now_row)] = [31, 9]

		now_row.pop()
		now_row.append('Sell')
		self.chimp.q_dict[tuple(now_row)] = [23, 4]

		assert self.chimp.make_decision(self.chimp.now_row) == 'Hold'

	def test_reset(self):
		self.chimp.buy(245.1902)
		self.chimp.reset()

		assert np.abs(self.chimp.cash - 1000) < 0.0001
		assert np.abs(self.chimp.share - 0) < 0.0001
		assert np.abs(self.chimp.pv - 0) < 0.0001

	def test_iter_env_type(self):
		assert isinstance(self.chimp.iter_env, types.GeneratorType)


	def test_iter_env_index(self):
		assert self.chimp.now_env_index == datetime(1993, 6, 23)
		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()
		assert self.chimp.now_env_index == datetime(1993, 6, 24)
		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()
		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()
		assert self.chimp.now_env_index == datetime(1993, 6, 28)

	def test_iter_env_row(self):
		assert self.chimp.now_row[0] == 3
		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()
		assert self.chimp.now_row[13] == -1
		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()
		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()
		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()
		assert self.chimp.now_row[2] == 5
		assert np.abs(self.chimp.now_row[-1] -  29.407375) < 0.0001

	def test_q_df_columns_type(self):
		assert type(self.chimp.q_df_columns) == type([])

	def test_q_df_columns_value(self):
		assert self.chimp.q_df_columns[-1] == 'Q Value'
		assert self.chimp.q_df_columns[-6] == '10d_lowerwick'
		assert self.chimp.q_df_columns[11] == '-4d_Vol2'

	def test_q_dict(self):
		assert self.chimp.q_dict[(self.chimp.now_row[0], \
			    self.chimp.now_row[1], \
			    self.chimp.now_row[2], \
			    self.chimp.now_row[3], \
			    self.chimp.now_row[4], \
			    self.chimp.now_row[5], \
			    self.chimp.now_row[6], \
			    self.chimp.now_row[7], \
			    self.chimp.now_row[8], \
			    self.chimp.now_row[9], \
			    self.chimp.now_row[10], \
			    self.chimp.now_row[11], \
			    self.chimp.now_row[12], \
			    self.chimp.now_row[13], \
			    self.chimp.now_row[14], \
			    self.chimp.now_row[15], \
			    self.chimp.now_row[16], \
			    self.chimp.now_row[17], \
			    self.chimp.now_row[18], \
			    self.chimp.now_row[19], \
			    self.chimp.now_row[20], \
			    self.chimp.now_row[21], \
			    self.chimp.now_row[22], \
			    self.chimp.now_row[23], \
			    self.chimp.now_row[24], \
			    self.chimp.now_row[25], \
			    self.chimp.now_row[26], \
			    self.chimp.now_row[27], \
			    self.chimp.now_row[28], \
			    self.chimp.now_row[29], \
			    self.chimp.now_row[30], \
			    self.chimp.now_row[31], \
			    self.chimp.now_row[32], \
			    self.chimp.now_row[33], \
			    self.chimp.now_row[34], \
			    self.chimp.now_row[35], \
			    self.chimp.now_row[36], \
			    self.chimp.yes_share(), 'Buy')] == [0, 0]

		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()

		assert self.chimp.q_dict[(self.chimp.now_row[0], \
			    self.chimp.now_row[1], \
			    self.chimp.now_row[2], \
			    self.chimp.now_row[3], \
			    self.chimp.now_row[4], \
			    self.chimp.now_row[5], \
			    self.chimp.now_row[6], \
			    self.chimp.now_row[7], \
			    self.chimp.now_row[8], \
			    self.chimp.now_row[9], \
			    self.chimp.now_row[10], \
			    self.chimp.now_row[11], \
			    self.chimp.now_row[12], \
			    self.chimp.now_row[13], \
			    self.chimp.now_row[14], \
			    self.chimp.now_row[15], \
			    self.chimp.now_row[16], \
			    self.chimp.now_row[17], \
			    self.chimp.now_row[18], \
			    self.chimp.now_row[19], \
			    self.chimp.now_row[20], \
			    self.chimp.now_row[21], \
			    self.chimp.now_row[22], \
			    self.chimp.now_row[23], \
			    self.chimp.now_row[24], \
			    self.chimp.now_row[25], \
			    self.chimp.now_row[26], \
			    self.chimp.now_row[27], \
			    self.chimp.now_row[28], \
			    self.chimp.now_row[29], \
			    self.chimp.now_row[30], \
			    self.chimp.now_row[31], \
			    self.chimp.now_row[32], \
			    self.chimp.now_row[33], \
			    self.chimp.now_row[34], \
			    self.chimp.now_row[35], \
			    self.chimp.now_row[36], \
			    self.chimp.yes_share(), 'Hold')] == [0, 0]

	def test_q_df_type(self):
		assert type(self.chimp.q_df) == type(pd.DataFrame())

	def test_q_df_columns(self):
		# print(self.chimp.q_df)
		assert list(self.chimp.q_df.columns) == self.chimp.q_df_columns

	def test_make_q_df(self):
		self.chimp.q_dict[(self.chimp.now_row[0], \
			    self.chimp.now_row[1], \
			    self.chimp.now_row[2], \
			    self.chimp.now_row[3], \
			    self.chimp.now_row[4], \
			    self.chimp.now_row[5], \
			    self.chimp.now_row[6], \
			    self.chimp.now_row[7], \
			    self.chimp.now_row[8], \
			    self.chimp.now_row[9], \
			    self.chimp.now_row[10], \
			    self.chimp.now_row[11], \
			    self.chimp.now_row[12], \
			    self.chimp.now_row[13], \
			    self.chimp.now_row[14], \
			    self.chimp.now_row[15], \
			    self.chimp.now_row[16], \
			    self.chimp.now_row[17], \
			    self.chimp.now_row[18], \
			    self.chimp.now_row[19], \
			    self.chimp.now_row[20], \
			    self.chimp.now_row[21], \
			    self.chimp.now_row[22], \
			    self.chimp.now_row[23], \
			    self.chimp.now_row[24], \
			    self.chimp.now_row[25], \
			    self.chimp.now_row[26], \
			    self.chimp.now_row[27], \
			    self.chimp.now_row[28], \
			    self.chimp.now_row[29], \
			    self.chimp.now_row[30], \
			    self.chimp.now_row[31], \
			    self.chimp.now_row[32], \
			    self.chimp.now_row[33], \
			    self.chimp.now_row[34], \
			    self.chimp.now_row[35], \
			    self.chimp.now_row[36], \
			    self.chimp.yes_share(), 'Buy')] = [102.2881, 8]

		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()

		self.chimp.q_dict[(self.chimp.now_row[0], \
			    self.chimp.now_row[1], \
			    self.chimp.now_row[2], \
			    self.chimp.now_row[3], \
			    self.chimp.now_row[4], \
			    self.chimp.now_row[5], \
			    self.chimp.now_row[6], \
			    self.chimp.now_row[7], \
			    self.chimp.now_row[8], \
			    self.chimp.now_row[9], \
			    self.chimp.now_row[10], \
			    self.chimp.now_row[11], \
			    self.chimp.now_row[12], \
			    self.chimp.now_row[13], \
			    self.chimp.now_row[14], \
			    self.chimp.now_row[15], \
			    self.chimp.now_row[16], \
			    self.chimp.now_row[17], \
			    self.chimp.now_row[18], \
			    self.chimp.now_row[19], \
			    self.chimp.now_row[20], \
			    self.chimp.now_row[21], \
			    self.chimp.now_row[22], \
			    self.chimp.now_row[23], \
			    self.chimp.now_row[24], \
			    self.chimp.now_row[25], \
			    self.chimp.now_row[26], \
			    self.chimp.now_row[27], \
			    self.chimp.now_row[28], \
			    self.chimp.now_row[29], \
			    self.chimp.now_row[30], \
			    self.chimp.now_row[31], \
			    self.chimp.now_row[32], \
			    self.chimp.now_row[33], \
			    self.chimp.now_row[34], \
			    self.chimp.now_row[35], \
			    self.chimp.now_row[36], \
			    self.chimp.yes_share(), 'Hold')] = [122.2289, 8]

		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()

		self.chimp.q_dict[(self.chimp.now_row[0], \
			    self.chimp.now_row[1], \
			    self.chimp.now_row[2], \
			    self.chimp.now_row[3], \
			    self.chimp.now_row[4], \
			    self.chimp.now_row[5], \
			    self.chimp.now_row[6], \
			    self.chimp.now_row[7], \
			    self.chimp.now_row[8], \
			    self.chimp.now_row[9], \
			    self.chimp.now_row[10], \
			    self.chimp.now_row[11], \
			    self.chimp.now_row[12], \
			    self.chimp.now_row[13], \
			    self.chimp.now_row[14], \
			    self.chimp.now_row[15], \
			    self.chimp.now_row[16], \
			    self.chimp.now_row[17], \
			    self.chimp.now_row[18], \
			    self.chimp.now_row[19], \
			    self.chimp.now_row[20], \
			    self.chimp.now_row[21], \
			    self.chimp.now_row[22], \
			    self.chimp.now_row[23], \
			    self.chimp.now_row[24], \
			    self.chimp.now_row[25], \
			    self.chimp.now_row[26], \
			    self.chimp.now_row[27], \
			    self.chimp.now_row[28], \
			    self.chimp.now_row[29], \
			    self.chimp.now_row[30], \
			    self.chimp.now_row[31], \
			    self.chimp.now_row[32], \
			    self.chimp.now_row[33], \
			    self.chimp.now_row[34], \
			    self.chimp.now_row[35], \
			    self.chimp.now_row[36], \
			    self.chimp.yes_share(), 'Sell')] = [12.2289, 8]

		# print(self.chimp.q_dict)
		self.chimp.make_q_df()
		# print(self.chimp.q_df)

		if self.chimp.q_df.ix[0, -2] == 0:
			assert np.abs(self.chimp.q_df.ix[0, -1] -  122.2289) < 0.0001
			assert self.chimp.q_df.ix[0, -4] == 2

		elif self.chimp.q_df.ix[0, -2] == 1:
			assert np.abs(self.chimp.q_df.ix[0, -1] -  102.2881) < 0.0001
		else:
			assert np.abs(self.chimp.q_df.ix[0, -1] -  12.2289) < 0.0001

	def test_split_q_df(self):
		self.chimp.q_dict[(self.chimp.now_row[0], \
		    self.chimp.now_row[1], \
		    self.chimp.now_row[2], \
		    self.chimp.now_row[3], \
		    self.chimp.now_row[4], \
		    self.chimp.now_row[5], \
		    self.chimp.now_row[6], \
		    self.chimp.now_row[7], \
		    self.chimp.now_row[8], \
		    self.chimp.now_row[9], \
		    self.chimp.now_row[10], \
		    self.chimp.now_row[11], \
		    self.chimp.now_row[12], \
		    self.chimp.now_row[13], \
		    self.chimp.now_row[14], \
		    self.chimp.now_row[15], \
		    self.chimp.now_row[16], \
		    self.chimp.now_row[17], \
		    self.chimp.now_row[18], \
		    self.chimp.now_row[19], \
		    self.chimp.now_row[20], \
		    self.chimp.now_row[21], \
		    self.chimp.now_row[22], \
		    self.chimp.now_row[23], \
		    self.chimp.now_row[24], \
		    self.chimp.now_row[25], \
		    self.chimp.now_row[26], \
		    self.chimp.now_row[27], \
		    self.chimp.now_row[28], \
		    self.chimp.now_row[29], \
		    self.chimp.now_row[30], \
		    self.chimp.now_row[31], \
		    self.chimp.now_row[32], \
		    self.chimp.now_row[33], \
		    self.chimp.now_row[34], \
		    self.chimp.now_row[35], \
		    self.chimp.now_row[36], \
		    self.chimp.yes_share(), 'Buy')] = [102.2881, 8]

		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()

		self.chimp.q_dict[(self.chimp.now_row[0], \
		    self.chimp.now_row[1], \
		    self.chimp.now_row[2], \
		    self.chimp.now_row[3], \
		    self.chimp.now_row[4], \
		    self.chimp.now_row[5], \
		    self.chimp.now_row[6], \
		    self.chimp.now_row[7], \
		    self.chimp.now_row[8], \
		    self.chimp.now_row[9], \
		    self.chimp.now_row[10], \
		    self.chimp.now_row[11], \
		    self.chimp.now_row[12], \
		    self.chimp.now_row[13], \
		    self.chimp.now_row[14], \
		    self.chimp.now_row[15], \
		    self.chimp.now_row[16], \
		    self.chimp.now_row[17], \
		    self.chimp.now_row[18], \
		    self.chimp.now_row[19], \
		    self.chimp.now_row[20], \
		    self.chimp.now_row[21], \
		    self.chimp.now_row[22], \
		    self.chimp.now_row[23], \
		    self.chimp.now_row[24], \
		    self.chimp.now_row[25], \
		    self.chimp.now_row[26], \
		    self.chimp.now_row[27], \
		    self.chimp.now_row[28], \
		    self.chimp.now_row[29], \
		    self.chimp.now_row[30], \
		    self.chimp.now_row[31], \
		    self.chimp.now_row[32], \
		    self.chimp.now_row[33], \
		    self.chimp.now_row[34], \
		    self.chimp.now_row[35], \
		    self.chimp.now_row[36], \
		    self.chimp.yes_share(), 'Hold')] = [111.2289, 8]

		self.chimp.now_env_index, self.chimp.now_row = self.chimp.iter_env.next()

		self.chimp.q_dict[(self.chimp.now_row[0], \
			    self.chimp.now_row[1], \
			    self.chimp.now_row[2], \
			    self.chimp.now_row[3], \
			    self.chimp.now_row[4], \
			    self.chimp.now_row[5], \
			    self.chimp.now_row[6], \
			    self.chimp.now_row[7], \
			    self.chimp.now_row[8], \
			    self.chimp.now_row[9], \
			    self.chimp.now_row[10], \
			    self.chimp.now_row[11], \
			    self.chimp.now_row[12], \
			    self.chimp.now_row[13], \
			    self.chimp.now_row[14], \
			    self.chimp.now_row[15], \
			    self.chimp.now_row[16], \
			    self.chimp.now_row[17], \
			    self.chimp.now_row[18], \
			    self.chimp.now_row[19], \
			    self.chimp.now_row[20], \
			    self.chimp.now_row[21], \
			    self.chimp.now_row[22], \
			    self.chimp.now_row[23], \
			    self.chimp.now_row[24], \
			    self.chimp.now_row[25], \
			    self.chimp.now_row[26], \
			    self.chimp.now_row[27], \
			    self.chimp.now_row[28], \
			    self.chimp.now_row[29], \
			    self.chimp.now_row[30], \
			    self.chimp.now_row[31], \
			    self.chimp.now_row[32], \
			    self.chimp.now_row[33], \
			    self.chimp.now_row[34], \
			    self.chimp.now_row[35], \
			    self.chimp.now_row[36], \
			    self.chimp.yes_share(), 'Sell')] = [12.2289, 8]

		self.chimp.make_q_df()
		# print(self.chimp.q_df)

		self.chimp.split_q_df()
		# print(self.chimp.q_df_train)
		# print(self.chimp.q_df_test)

		if self.chimp.q_df_train.ix[0, -1] == 0:
			assert np.abs(self.chimp.q_df_train.ix[0, -3] -  2) < 0.0001
			assert self.chimp.q_df_train.ix[0, -4] == 0

		elif self.chimp.q_df.ix[0, -2] == 1:
			assert np.abs(self.chimp.q_df.ix[0, -3] -  2) < 0.0001
			assert np.abs(self.chimp.q_df.ix[0, -7] -  1) < 0.0001
		else:
			assert np.abs(self.chimp.q_df.ix[0, -2] -  3) < 0.0001
			assert np.abs(self.chimp.q_df.ix[0, -7] -  1) < 0.0001

	def test_train_on_q_df(self):
		for i in range(500):
			# Generate q table randomly as q_dict
			mu = random.randint(10, 20)
			sigma = random.uniform(3, 50)

			vols = [random.randint(1, 5) for rand in xrange(13)]
			spreads = [random.choice([1, 2, 3, 4, 5, -1, -2, -3, -4, -5]) for r in xrange(8)]
			up_lw_wicks = [random.randint(0, 5) for r in xrange(16)]
			yes_share = random.choice([0, 1])
			action = random.choice(['Buy', 'Sell', 'Hold'])

			row_data = vols + spreads + up_lw_wicks
			row_data.append(yes_share)
			row_data.append(action)

			self.chimp.q_dict[tuple(row_data)] = [np.random.normal(mu, sigma, 1)[0], random.randint(0, 10)]

		self.chimp.update_q_df()

		# print(self.chimp.q_df_train)
		# print(self.chimp.q_df_test)

		self.chimp.train_on_q_df()

		# vols_test = [random.randint(1, 5) for rand in xrange(13)]
		# spreads_test = [random.choice([1, 2, 3, 4, 5, -1, -2, -3, -4, -5]) for r in xrange(8)]
		# up_lw_wicks_test = [random.randint(0, 5) for r in xrange(16)]
		# action_test = random.choice([0, 1, 2])

		# row_data_test = vols_test + spreads_test + up_lw_wicks_test
		# row_data_test.append(action_test)
		# row_data_test = [row_data_test]

		# row_data_pred = self.chimp.q_reg.predict(row_data_test)
		# print(row_data_pred)

	def test_from_state_action_predict_q(self):
		# Generate q table randomly as q_dict
		for i in range(500):
			mu = random.randint(10, 20)
			sigma = random.uniform(3, 50)

			vols = [random.randint(1, 5) for rand in xrange(13)]
			spreads = [random.choice([1, 2, 3, 4, 5, -1, -2, -3, -4, -5]) for r in xrange(8)]
			up_lw_wicks = [random.randint(0, 5) for r in xrange(16)]
			yes_share = random.choice([0, 1])
			action = random.choice(['Buy', 'Sell', 'Hold'])

			row_data = vols + spreads + up_lw_wicks
			row_data.append(yes_share)
			row_data.append(action)

			self.chimp.q_dict[tuple(row_data)] = [np.random.normal(mu, sigma, 1)[0], random.randint(0, 10)]

		# Generate q_df from q_dict
		self.chimp.update_q_df()

		# Train on the q_df and get model
		self.chimp.train_on_q_df()

		# Generate the real time test data
		vols_test = [random.randint(1, 5) for rand in xrange(13)]
		spreads_test = [random.choice([1, 2, 3, 4, 5, -1, -2, -3, -4, -5]) for r in xrange(8)]
		up_lw_wicks_test = [random.randint(0, 5) for r in xrange(16)]
		yes_share_test = random.choice([0, 1])
		action_test = random.choice([0, 1, 2]) # 'Hold' -> 0, 'Buy' -> 1, 'Sell' -> 2

		row_data_test = vols_test + spreads_test + up_lw_wicks_test
		row_data_test.append(yes_share)
		row_data_test.append(action_test)
		# row_data_test = [row_data_test]

		# row_data_pred = self.chimp.q_reg.predict(row_data_test)
		# print(row_data_pred)

		# print(self.chimp.from_state_action_predict_q(row_data_test))

	def test_max_q_output_type(self):
		self.chimp.q_dict[(self.chimp.now_row[0], \
		    self.chimp.now_row[1], \
		    self.chimp.now_row[2], \
		    self.chimp.now_row[3], \
		    self.chimp.now_row[4], \
		    self.chimp.now_row[5], \
		    self.chimp.now_row[6], \
		    self.chimp.now_row[7], \
		    self.chimp.now_row[8], \
		    self.chimp.now_row[9], \
		    self.chimp.now_row[10], \
		    self.chimp.now_row[11], \
		    self.chimp.now_row[12], \
		    self.chimp.now_row[13], \
		    self.chimp.now_row[14], \
		    self.chimp.now_row[15], \
		    self.chimp.now_row[16], \
		    self.chimp.now_row[17], \
		    self.chimp.now_row[18], \
		    self.chimp.now_row[19], \
		    self.chimp.now_row[20], \
		    self.chimp.now_row[21], \
		    self.chimp.now_row[22], \
		    self.chimp.now_row[23], \
		    self.chimp.now_row[24], \
		    self.chimp.now_row[25], \
		    self.chimp.now_row[26], \
		    self.chimp.now_row[27], \
		    self.chimp.now_row[28], \
		    self.chimp.now_row[29], \
		    self.chimp.now_row[30], \
		    self.chimp.now_row[31], \
		    self.chimp.now_row[32], \
		    self.chimp.now_row[33], \
		    self.chimp.now_row[34], \
		    self.chimp.now_row[35], \
		    self.chimp.now_row[36], \
		    1, 'Hold')] = [102.1234, 8]

		self.chimp.q_dict[(self.chimp.now_row[0], \
		    self.chimp.now_row[1], \
		    self.chimp.now_row[2], \
		    self.chimp.now_row[3], \
		    self.chimp.now_row[4], \
		    self.chimp.now_row[5], \
		    self.chimp.now_row[6], \
		    self.chimp.now_row[7], \
		    self.chimp.now_row[8], \
		    self.chimp.now_row[9], \
		    self.chimp.now_row[10], \
		    self.chimp.now_row[11], \
		    self.chimp.now_row[12], \
		    self.chimp.now_row[13], \
		    self.chimp.now_row[14], \
		    self.chimp.now_row[15], \
		    self.chimp.now_row[16], \
		    self.chimp.now_row[17], \
		    self.chimp.now_row[18], \
		    self.chimp.now_row[19], \
		    self.chimp.now_row[20], \
		    self.chimp.now_row[21], \
		    self.chimp.now_row[22], \
		    self.chimp.now_row[23], \
		    self.chimp.now_row[24], \
		    self.chimp.now_row[25], \
		    self.chimp.now_row[26], \
		    self.chimp.now_row[27], \
		    self.chimp.now_row[28], \
		    self.chimp.now_row[29], \
		    self.chimp.now_row[30], \
		    self.chimp.now_row[31], \
		    self.chimp.now_row[32], \
		    self.chimp.now_row[33], \
		    self.chimp.now_row[34], \
		    self.chimp.now_row[35], \
		    self.chimp.now_row[36], \
		    1, 'Buy')] = [12.2881, 8]

		self.chimp.q_dict[(self.chimp.now_row[0], \
		    self.chimp.now_row[1], \
		    self.chimp.now_row[2], \
		    self.chimp.now_row[3], \
		    self.chimp.now_row[4], \
		    self.chimp.now_row[5], \
		    self.chimp.now_row[6], \
		    self.chimp.now_row[7], \
		    self.chimp.now_row[8], \
		    self.chimp.now_row[9], \
		    self.chimp.now_row[10], \
		    self.chimp.now_row[11], \
		    self.chimp.now_row[12], \
		    self.chimp.now_row[13], \
		    self.chimp.now_row[14], \
		    self.chimp.now_row[15], \
		    self.chimp.now_row[16], \
		    self.chimp.now_row[17], \
		    self.chimp.now_row[18], \
		    self.chimp.now_row[19], \
		    self.chimp.now_row[20], \
		    self.chimp.now_row[21], \
		    self.chimp.now_row[22], \
		    self.chimp.now_row[23], \
		    self.chimp.now_row[24], \
		    self.chimp.now_row[25], \
		    self.chimp.now_row[26], \
		    self.chimp.now_row[27], \
		    self.chimp.now_row[28], \
		    self.chimp.now_row[29], \
		    self.chimp.now_row[30], \
		    self.chimp.now_row[31], \
		    self.chimp.now_row[32], \
		    self.chimp.now_row[33], \
		    self.chimp.now_row[34], \
		    self.chimp.now_row[35], \
		    self.chimp.now_row[36], \
		    0, 'Sell')] = [102.2881, 8]

		assert isinstance(self.chimp.max_q(self.chimp.now_row)[1], Number)
		assert isinstance(self.chimp.max_q(self.chimp.now_row)[2], Number)

	def test_max_q_output_value(self): # return (action, q_value, t)
		self.chimp.q_dict[(self.chimp.now_row[0], \
		    self.chimp.now_row[1], \
		    self.chimp.now_row[2], \
		    self.chimp.now_row[3], \
		    self.chimp.now_row[4], \
		    self.chimp.now_row[5], \
		    self.chimp.now_row[6], \
		    self.chimp.now_row[7], \
		    self.chimp.now_row[8], \
		    self.chimp.now_row[9], \
		    self.chimp.now_row[10], \
		    self.chimp.now_row[11], \
		    self.chimp.now_row[12], \
		    self.chimp.now_row[13], \
		    self.chimp.now_row[14], \
		    self.chimp.now_row[15], \
		    self.chimp.now_row[16], \
		    self.chimp.now_row[17], \
		    self.chimp.now_row[18], \
		    self.chimp.now_row[19], \
		    self.chimp.now_row[20], \
		    self.chimp.now_row[21], \
		    self.chimp.now_row[22], \
		    self.chimp.now_row[23], \
		    self.chimp.now_row[24], \
		    self.chimp.now_row[25], \
		    self.chimp.now_row[26], \
		    self.chimp.now_row[27], \
		    self.chimp.now_row[28], \
		    self.chimp.now_row[29], \
		    self.chimp.now_row[30], \
		    self.chimp.now_row[31], \
		    self.chimp.now_row[32], \
		    self.chimp.now_row[33], \
		    self.chimp.now_row[34], \
		    self.chimp.now_row[35], \
		    self.chimp.now_row[36], \
		    1, 'Hold')] = [102.1234, 8]

		self.chimp.q_dict[(self.chimp.now_row[0], \
		    self.chimp.now_row[1], \
		    self.chimp.now_row[2], \
		    self.chimp.now_row[3], \
		    self.chimp.now_row[4], \
		    self.chimp.now_row[5], \
		    self.chimp.now_row[6], \
		    self.chimp.now_row[7], \
		    self.chimp.now_row[8], \
		    self.chimp.now_row[9], \
		    self.chimp.now_row[10], \
		    self.chimp.now_row[11], \
		    self.chimp.now_row[12], \
		    self.chimp.now_row[13], \
		    self.chimp.now_row[14], \
		    self.chimp.now_row[15], \
		    self.chimp.now_row[16], \
		    self.chimp.now_row[17], \
		    self.chimp.now_row[18], \
		    self.chimp.now_row[19], \
		    self.chimp.now_row[20], \
		    self.chimp.now_row[21], \
		    self.chimp.now_row[22], \
		    self.chimp.now_row[23], \
		    self.chimp.now_row[24], \
		    self.chimp.now_row[25], \
		    self.chimp.now_row[26], \
		    self.chimp.now_row[27], \
		    self.chimp.now_row[28], \
		    self.chimp.now_row[29], \
		    self.chimp.now_row[30], \
		    self.chimp.now_row[31], \
		    self.chimp.now_row[32], \
		    self.chimp.now_row[33], \
		    self.chimp.now_row[34], \
		    self.chimp.now_row[35], \
		    self.chimp.now_row[36], \
		    1, 'Buy')] = [12.2881, 8]

		self.chimp.q_dict[(self.chimp.now_row[0], \
		    self.chimp.now_row[1], \
		    self.chimp.now_row[2], \
		    self.chimp.now_row[3], \
		    self.chimp.now_row[4], \
		    self.chimp.now_row[5], \
		    self.chimp.now_row[6], \
		    self.chimp.now_row[7], \
		    self.chimp.now_row[8], \
		    self.chimp.now_row[9], \
		    self.chimp.now_row[10], \
		    self.chimp.now_row[11], \
		    self.chimp.now_row[12], \
		    self.chimp.now_row[13], \
		    self.chimp.now_row[14], \
		    self.chimp.now_row[15], \
		    self.chimp.now_row[16], \
		    self.chimp.now_row[17], \
		    self.chimp.now_row[18], \
		    self.chimp.now_row[19], \
		    self.chimp.now_row[20], \
		    self.chimp.now_row[21], \
		    self.chimp.now_row[22], \
		    self.chimp.now_row[23], \
		    self.chimp.now_row[24], \
		    self.chimp.now_row[25], \
		    self.chimp.now_row[26], \
		    self.chimp.now_row[27], \
		    self.chimp.now_row[28], \
		    self.chimp.now_row[29], \
		    self.chimp.now_row[30], \
		    self.chimp.now_row[31], \
		    self.chimp.now_row[32], \
		    self.chimp.now_row[33], \
		    self.chimp.now_row[34], \
		    self.chimp.now_row[35], \
		    self.chimp.now_row[36], \
		    0, 'Sell')] = [102.2881, 8]

		assert self.chimp.max_q(self.chimp.now_row)[0] == 'Sell'
		assert np.abs(self.chimp.max_q(self.chimp.now_row)[1] - 102.2881) < 0.0001

		self.chimp.q_dict[(1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    1, \
		    2, \
		    3, \
		    4, \
		    1, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    6, \
		    1, \
		    1, \
		    1, \
		    0, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    1, \
		    0, 'Hold')] = [28, 8]

		self.chimp.q_dict[(1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    1, \
		    2, \
		    3, \
		    4, \
		    1, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    6, \
		    1, \
		    1, \
		    1, \
		    0, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    1, \
		    0, 'Buy')] = [11.2881, 8]

		self.chimp.q_dict[(1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    1, \
		    2, \
		    3, \
		    4, \
		    1, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    6, \
		    1, \
		    1, \
		    1, \
		    0, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    1, \
		    0, 'Sell')] = [10.2881, 8]

		assert self.chimp.max_q((1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    1, \
		    2, \
		    3, \
		    4, \
		    1, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    6, \
		    1, \
		    1, \
		    1, \
		    0, \
		    1, \
		    2, \
		    3, \
		    4, \
		    5, \
		    1, \
		    0))[0] == 'Hold'







