from __future__ import division

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

# Set global random seeds
np.random.seed(0)
random.seed(0)

class ETL_SPY(object):
	def __init__(self, filename):
		# Load SPY dataset
		self.raw_data = pd.read_csv(filename, index_col='Date', parse_dates=True, na_values = ['nan'])
		self.dfMain = None

	def load_data(self):
		# Delete Index name
		del self.raw_data.index.name

		# Set date range
		start_date = '1993-01-29'
		end_date = '2016-09-06'
		dates = pd.date_range(start_date, end_date)

		self.dfMain = pd.DataFrame(index=dates)
		self.dfMain = self.dfMain.join(self.raw_data)
		self.dfMain.dropna(inplace=True)
		print(self.dfMain)

