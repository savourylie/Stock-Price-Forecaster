from chimpbot import ChimpBot
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import pickle

def main():
	# Initiating data and the chimp
	dfEnv = pd.read_csv('data_train.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	dfTest = pd.read_csv('data_test.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	chimp_train = ChimpBot(dfEnv)
	chimp_test = ChimpBot(dfTest)


	for i in range(3500):
		# Train the Chimp on train_data
		for j in range(len(chimp_train.env)):
			print("Train Round {0}-{1}".format(i + 1, j + 1))
			chimp_train.update()
		chimp_train.reset()

		# Train the Chimp on test_data
		for k in range(len(chimp_test.env)):
			print("Test Round {0}-{1}".format(i + 1, k + 1))
			chimp_test.update()
		chimp_test.reset()

	with open('chimp_train_3200_pv_history.pickle', 'wb') as f1:
		pickle.dump(chimp_train.pv_history_list, f1, pickle.HIGHEST_PROTOCOL)
	print(chimp_train.pv_history_list)

	with open('chimp_test_3200_pv_history.pickle', 'wb') as f2:
		pickle.dump(chimp_test.pv_history_list, f2, pickle.HIGHEST_PROTOCOL)
	print(chimp_test.pv_history_list)

	# Convert Q-Table to Dataframe from trained chimp (train)
	result_dict_train = defaultdict(list)
	for index, row in chimp_train.q_dict.iteritems():
	    for i in range(len(chimp_train.q_dict.keys()[0])):
	        column_name = 'col' + str(i + 1)
	        result_dict_train[column_name].append(index[i])
	    result_dict_train['Q'].append(chimp_train.q_dict[index][0])

	q_df = pd.DataFrame(result_dict_train)
	q_df.to_csv('q_df_3200_train.csv')

	# Convert Q-Table to Dataframe from trained chimp (train)
	result_dict_test = defaultdict(list)
	for index, row in chimp_test.q_dict.iteritems():
	    for i in range(len(chimp_test.q_dict.keys()[0])):
	        column_name = 'col' + str(i + 1)
	        result_dict_test[column_name].append(index[i])
	    result_dict_test['Q'].append(chimp_test.q_dict[index][0])

	q_df = pd.DataFrame(result_dict_test)
	q_df.to_csv('q_df_3200_test.csv')

	# Save the chimp train properties
	# Save q_df
	with open('3200_train_q_df.pickle', 'wb') as f:
		pickle.dump(chimp_train.q_df, f, pickle.HIGHEST_PROTOCOL)
	# Save q_dict
	with open('3200_train_q_dict.pickle', 'wb') as f:
		pickle.dump(chimp_train.q_dict, f, pickle.HIGHEST_PROTOCOL)
	# Save q_reg
	with open('3200_train_q_reg.pickle', 'wb') as f:
		pickle.dump(chimp_train.q_reg, f, pickle.HIGHEST_PROTOCOL)

	# Test the Chimp!
	# q_df = deepcopy(chimp.q_df)
	# q_dict = deepcopy(chimp.q_dict)
	# q_reg = deepcopy(chimp.q_reg)

	# chimp_test = ChimpBot(dfTest)

	# for i in range(1000):
	# 	chimp_test.q_df = deepcopy(q_df)
	# 	chimp_test.q_dict = deepcopy(q_dict)
	# 	chimp_test.q_reg = deepcopy(q_reg)
	# 	chimp_test.epsilon = 0.05

	# 	for j in range(len(chimp_test.env)):
	# 		print("Iter-Row: {0}-{1}".format(i, j))
	# 		chimp_test.update()
	# 	chimp_test.reset()

	# print(chimp.pv_history_list)
	# print(chimp_test.pv_history_list)


if __name__ == '__main__':
	main()