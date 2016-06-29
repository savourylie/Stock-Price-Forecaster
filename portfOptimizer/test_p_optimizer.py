import os
from p_optimizer import find_csv_filenames, get_stock_names, get_data


def test_find_csv_filenames():
	assert find_csv_filenames(os.getcwd() + "/stocksYahoo")[0] == "AAPL.csv"
	assert find_csv_filenames(os.getcwd() + "/stocksYahoo")[1] == "GOOG.csv"
	assert find_csv_filenames(os.getcwd() + "/stocksYahoo")[2] == "IBM.csv"
	assert find_csv_filenames(os.getcwd() + "/stocksYahoo")[3] == "SPY.csv"


def test_get_stock_names():
	assert get_stock_names("/stocksYahoo")[0] == "AAPL"
	assert get_stock_names("/stocksYahoo")[1] == "GOOG"
	assert get_stock_names("/stocksYahoo")[2] == "IBM"
	assert get_stock_names("/stocksYahoo")[3] == "SPY"


# def test_get_data():
# 	dates = pd.date_range('2009-01-01', '2015-12-31') # From start of 2009 to end of 2015
# 	assert get_data("/stocksYahoo", dates) ===