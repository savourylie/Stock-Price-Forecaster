from __future__ import division

import numpy as np

class ETL:
	def __init__(self, dataloader, symbol):
		self.symbols = ['SPY']
		self.symbols.append(symbol)

		self.current_stock_name = symbol
		self.df_temp = dataloader.stock_dict_original[symbol]

		self._add_avg_runup()
		self._add_daily_return()
		self._add_stock_mean_std63d()
		self._add_stock_cov63d_beta()
		self._add_ema()
		self._add_mma()
		self._add_sma()
		self._add_sma_momentum()
		self._add_vol_momentum()
		self._add_momentum_r1()
		self._add_momentum_r2()
		self._add_SR63d()

	def _add_avg_runup(self):
		"""Make Average Run-up columns (252 days)
		"""

		self.df_temp[self.current_stock_name + '_Avg_Runup'] = ((self.df_temp - self.df_temp.shift(252)) / 252)[self.current_stock_name]

	def _add_daily_return(self):
		"""Make Daily Return Columns
		"""
		for symbol in self.symbols:
			self.df_temp[symbol + '_return'] = (self.df_temp / self.df_temp.shift(1) - 1)[symbol]

	def _add_stock_mean_std63d(self):
		for symbol in self.symbols:
			self.df_temp[symbol + '_Mean63d'] = self.df_temp[symbol + '_return'].rolling(window=63, center=False).mean()
			self.df_temp[symbol + '_Std63d'] = self.df_temp[symbol + '_return'].rolling(window=63, center=False).std()


	def _add_stock_cov63d_beta(self):
		cov_dict = {}

		for symbol in self.symbols:
			cov_dict[symbol] = []
			for i in self.df_temp.index:
				(u,) = self.df_temp.index.get_indexer_for([i])
				if u - 62 >= 0:
					cov_dict[symbol].append(self.df_temp['SPY_return'].iloc[(u - 62):u+1].cov(self.df_temp[symbol + '_return'].iloc[(u - 62):u+1]))
				else:
					cov_dict[symbol].append(np.nan)
		self.df_temp[symbol + '_Cov63d'] = cov_dict[symbol]
		self.df_temp[symbol + '_Beta'] = self.df_temp[symbol + '_Cov63d'] / self.df_temp[symbol + '_Std63d']**2

	def _add_ema(self):
		EMA_dict = {}
		alpha = 2 / (100 + 1)

		for symbol in self.symbols:
			EMA_dict[symbol] = []
			EMA_dict[symbol].append(self.df_temp[symbol].iloc[0])

			for i in self.df_temp.index[1:]:
				(u,) = self.df_temp.index.get_indexer_for([i])
				EMA_dict[symbol].append(EMA_dict[symbol][u - 1] + alpha * (self.df_temp[symbol].iloc[u] - EMA_dict[symbol][u - 1]))

			self.df_temp[symbol + '_EMA'] = EMA_dict[symbol]

	def _add_mma(self):
		MMA_dict = {}
		alpha = 1 / 100

		for symbol in self.symbols:
			MMA_dict[symbol] = []
			MMA_dict[symbol].append(self.df_temp[symbol].iloc[0])

			for i in self.df_temp.index[1:]:
				(u,) = self.df_temp.index.get_indexer_for([i])
				MMA_dict[symbol].append(MMA_dict[symbol][u - 1] + alpha * (self.df_temp[symbol].iloc[u] - MMA_dict[symbol][u - 1]))

			self.df_temp[symbol + '_MMA'] = MMA_dict[symbol]

	def _add_sma(self):
		for symbol in self.symbols:
			self.df_temp[symbol + '_SMA'] = self.df_temp[symbol].rolling(window=101, center=False).mean()

	def _add_sma_momentum(self):
		for symbol in self.symbols:
			self.df_temp[symbol + '_SMA_Momentum'] = (self.df_temp - self.df_temp.shift(1))[symbol + '_SMA']*(100 + 1)

	def _add_vol_momentum(self):
		self.df_temp[self.current_stock_name + '_Vol_Momentum'] = (self.df_temp - self.df_temp.shift(1))[self.current_stock_name + '_Vol']*(100 + 1)

	def _add_momentum_r1(self):
		self.df_temp[self.current_stock_name + '_p_real1'] = np.nan
		self.df_temp.loc[self.df_temp[self.current_stock_name + '_Vol_Momentum'] >= 0, self.current_stock_name + '_p_real1'] = 1
		self.df_temp.loc[self.df_temp[self.current_stock_name + '_Vol_Momentum'] < 0, self.current_stock_name + '_p_real1'] = 0

	def _add_momentum_r2(self):
		self.df_temp[self.current_stock_name + '_p_real2'] = np.nan
		self.df_temp.loc[self.df_temp[self.current_stock_name + '_Vol'] >= (self.df_temp[self.current_stock_name + '_Vol'].mean() + self.df_temp[self.current_stock_name + '_Vol'].std()), self.current_stock_name + '_p_real2'] = 1
		self.df_temp.loc[self.df_temp[self.current_stock_name + '_Vol'] < (self.df_temp[self.current_stock_name + '_Vol'].mean() + self.df_temp[self.current_stock_name + '_Vol'].std()), self.current_stock_name + '_p_real2'] = 0

	def _add_momentum_r3(self): # To be implemented
		pass

	def _add_SR63d(self):
		for symbol in self.symbols:
			self.df_temp[symbol + '_SR63d'] = self.df_temp[symbol + '_return'].rolling(window=63, center=False).mean() / self.df_temp[symbol + '_Std63d']


