from chimpbot import ChimpBot
import pandas as pd

def main():
	# Initiating data and the chimp
	dfEnv = pd.read_csv('data_train.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	chimp = ChimpBot(dfEnv)

	print(chimp.env)






if __name__ == '__main__':
	main()