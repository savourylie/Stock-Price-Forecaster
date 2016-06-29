import numpy as np
import pandas as pd
import os


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def get_stock_names(data_folder):
    result_list = []

    for name in find_csv_filenames(os.getcwd() + data_folder):
        result_list.append(os.path.splitext(name)[0])

    return result_list

def get_data(data_folder, dates):
	df = pd.DataFrame(index=dates)

	stock_names = get_stock_names(data_folder)

	for name in stock_names:
		df_temp = pd.read_csv(name + ".csv", index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
		df_temp = df_temp.rename(columns={'Adj Close': name})
		df = df.join(df_temp)

		if name == 'SPY': # drop days SPY didn't trade
			df = df.dropna(subset=["SPY"])

	return df

