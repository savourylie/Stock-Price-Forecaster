from chimpbot import ChimpBot
import pandas as pd
from collections import defaultdict
from copy import deepcopy

def main():
	# Initiating data and the chimp
	# dfEnv = pd.read_csv('data_train_char.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	dfEnv = pd.read_csv('data_train.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	dfTest = pd.read_csv('data_cv.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	# dfEnv.ix[:, :-1] = dfEnv.ix[:, :-1].astype('int')
	chimp = ChimpBot(dfEnv)

	print(chimp.env)
	# q_dict_length = []

	for i in range(500):
		for j in range(len(chimp.env)):
			print("{0}-{1}".format(i + 1, j + 1))
			chimp.update()
		# q_dict_length.append(len(chimp.q_dict))
		chimp.reset()
	# print(chimp.pv_history_list)
	# print(q_dict_length)

	q_df = deepcopy(chimp.q_df)
	q_dict = deepcopy(chimp.q_dict)
	q_reg = deepcopy(chimp.q_reg)

	chimp_test = ChimpBot(dfTest)

	for i in range(1000):
		chimp_test.q_df = deepcopy(q_df)
		chimp_test.q_dict = deepcopy(q_dict)
		chimp_test.q_reg = deepcopy(q_reg)
		chimp_test.epsilon = 0.05

		for j in range(len(chimp_test.env)):
			print("Iter-Row: {0}-{1}".format(i, j))
			chimp_test.update()
		chimp_test.reset()

	print(chimp.pv_history_list)
	print(chimp_test.pv_history_list)


	# money_list = []
	# share_list = []
	# pv_list = []

	# for i in range(len(chimp.env)):
	# 	print("Day {0}".format(i + 1)
	# 	chimp.update()
	# 	money_list.append[chimp.cash]
	# 	share_list.append[chimp.share]
	# 	pv_list.append[chimp.pv]


	# Convert Q-Table to Dataframe
	# result_dict = defaultdict(list)
	# for index, row in chimp.q_dict.iteritems():
	#     for i in range(len(chimp.q_dict.keys()[0])):
	#         column_name = 'col' + str(i + 1)
	#         result_dict[column_name].append(index[i])
	#     result_dict['Q'].append(chimp.q_dict[index][0])

	# q_df = pd.DataFrame(result_dict)
	# q_df.to_csv('q_df.csv')

	# print(chimp.track_key1)
	# print(chimp.track_key2)
	# print("Track Random Decision: {}".format(chimp.track_random_decision))
	# print(chimp.q_dict.keys())


if __name__ == '__main__':
	main()