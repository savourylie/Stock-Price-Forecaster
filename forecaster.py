from __future__ import division

# Core
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

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
	def __init__(self, symbol, dataloader):
		self.e = ETL(symbol, dataloader)
		self.x_days = None

	def set_x_days(self, x):
		self.x_days = x



