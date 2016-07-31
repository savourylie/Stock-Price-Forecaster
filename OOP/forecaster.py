from __future__ import division

# Core
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from data_loader import DataLoader
from ETL import ETL

# Data Visualization
# %matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display

# Supervised Learning
from sklearn import grid_search
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Unsupervised Learning / PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Misc
from os import getcwd
import operator
from math import floor

# Page Configuration
pd.set_option('display.max_columns', 200)

class Forecaster:
	def __init__(self, dataloader, symbol):
		self.label_ready = False
		self.label_name = None
		self.data_ready = False
		self.data_X = None
		self.data_y = None

		self.symbol = symbol
		self.e = ETL(dataloader, symbol)
		self.x_days = None
		self.df_main = self.e.df_temp

	def set_x_days(self, x):
		self.x_days = x
		self._make_label()

	def _make_label(self):
		if self.label_ready:
			self.df_main.drop([self.label_name], axis=1, inplace=True)
			self._ready_data()
			self.pca()
			self.start_regressor()

		else:
			self.label_ready = True
			self._ready_data()
			self.pca()
			self.start_regressor()

	def _drop_features(self):
		# Drop all SPYs
		for column in self.df_main:
			if 'SPY' in column:
				self.df_main.drop([column], axis=1, inplace=True)

		### Drop NaN rows
		self.df_main.dropna(inplace=True)
		print(self.df_main.shape)

	def _splitting_features_label(self):
		# Separate features set and label
		self.data_X = self.df_main.drop([self.label_name], axis=1)
		self.data_y = self.df_main[self.label_name]

	def _normalization(self):
		self.data_X = (self.data_X - self.data_X.mean()) / self.data_X.std()

	def _ready_data(self):
		# Set label name
		self.label_name = self.symbol + str(self.x_days) + 'd'

		# Generate label data
		self.df_main[self.label_name] = self.df_main.shift(-self.x_days)[self.symbol]

		# Drop Features
		self._drop_features()

		# Separate features set and label
		self._splitting_features_label()

		# Normalize data_X
		self._normalization()

		# Split datasets into training/test
		self._split_training_test_sets()

	def _split_training_test_sets(self, train=0.6):
		self.train_X = self.data_X.iloc[:int(floor(self.data_X.shape[0] * train))]
		self.train_y = self.data_y.iloc[:int(floor(self.data_y.shape[0] * train))]
		self.test_X = self.data_X.iloc[int(floor(self.data_X.shape[0] * train)):]
		self.test_y = self.data_y.iloc[int(floor(self.data_y.shape[0] * train)):]

	def pca(self, n_components = 30):
		pass

	def start_regressor(self):
		lr = LinearRegression()
		parameters = {}
		reg = grid_search.GridSearchCV(lr, parameters, cv=9, scoring='mean_squared_error')

		reg.fit(self.train_X, self.train_y)
		self.pred_y = reg.predict(self.test_X)

		print(reg.score(self.test_X, self.test_y))
		print(r2_score(self.test_y, self.pred_y))

		return reg


