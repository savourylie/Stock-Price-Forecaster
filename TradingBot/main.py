from chimpbot import ChimpBot
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import pickle

def main():
	# Initiating data and the chimp
	dfFull = pd.read_csv('data_full.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	dfEnv = pd.read_csv('data_train.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	dfTest = pd.read_csv('data_test.csv', index_col=0, parse_dates=True, na_values = ['nan'])

	chimp_full = ChimpBot(dfFull)
	chimp_train = ChimpBot(dfEnv)
	chimp_test = ChimpBot(dfTest)

	for i in range(12000):
		# Train the Chimp on full_data
		for l in range(len(chimp_full.env)):
			print("Full Round {0}-{1}".format(i + 1, l + 1))
			chimp_full.update()
		chimp_full.reset()

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

	with open('chimp_full_11500_pv_history.pickle', 'wb') as f1:
		pickle.dump(chimp_full.pv_history_list, f1, pickle.HIGHEST_PROTOCOL)
	print(chimp_full.pv_history_list)

	with open('chimp_train_11500_pv_history.pickle', 'wb') as f1:
		pickle.dump(chimp_train.pv_history_list, f1, pickle.HIGHEST_PROTOCOL)
	print(chimp_train.pv_history_list)

	with open('chimp_test_11500_pv_history.pickle', 'wb') as f2:
		pickle.dump(chimp_test.pv_history_list, f2, pickle.HIGHEST_PROTOCOL)
	print(chimp_test.pv_history_list)

	# Convert Q-Table to Dataframe from trained chimp (train)
	result_dict_full = defaultdict(list)
	for index, row in chimp_full.q_dict.iteritems():
	    for i in range(len(chimp_full.q_dict.keys()[0])):
	        column_name = 'col' + str(i + 1)
	        result_dict_full[column_name].append(index[i])
	    result_dict_full['Q'].append(chimp_full.q_dict[index][0])

	q_df = pd.DataFrame(result_dict_full)
	q_df.to_csv('q_df_11500_full.csv')

	# Convert Q-Table to Dataframe from trained chimp (train)
	result_dict_train = defaultdict(list)
	for index, row in chimp_train.q_dict.iteritems():
	    for i in range(len(chimp_train.q_dict.keys()[0])):
	        column_name = 'col' + str(i + 1)
	        result_dict_train[column_name].append(index[i])
	    result_dict_train['Q'].append(chimp_train.q_dict[index][0])

	q_df = pd.DataFrame(result_dict_train)
	q_df.to_csv('q_df_11500_train.csv')

	# Convert Q-Table to Dataframe from trained chimp (train)
	result_dict_test = defaultdict(list)
	for index, row in chimp_test.q_dict.iteritems():
	    for i in range(len(chimp_test.q_dict.keys()[0])):
	        column_name = 'col' + str(i + 1)
	        result_dict_test[column_name].append(index[i])
	    result_dict_test['Q'].append(chimp_test.q_dict[index][0])

	q_df = pd.DataFrame(result_dict_test)
	q_df.to_csv('q_df_11500_test.csv')

	# Save the chimp train properties
	# Save q_df
	# with open('3200_train_q_df.pickle', 'wb') as f:
	# 	pickle.dump(chimp_train.q_df, f, pickle.HIGHEST_PROTOCOL)
	# # Save q_dict
	# with open('3200_train_q_dict.pickle', 'wb') as f:
	# 	pickle.dump(chimp_train.q_dict, f, pickle.HIGHEST_PROTOCOL)
	# # Save q_reg
	# with open('3200_train_q_reg.pickle', 'wb') as f:
	# 	pickle.dump(chimp_train.q_reg, f, pickle.HIGHEST_PROTOCOL)

	try:
		print(chimp_train.q_dict)
	except AttributeError:
		print("No q_dict? No big deal I guess...?")

	# Test the Chimp!
	q_df = deepcopy(chimp_train.q_df)
	q_dict = deepcopy(chimp_train.q_dict)
	q_reg = deepcopy(chimp_train.q_reg)

	chimp_real_test = ChimpBot(dfTest)

	for i in range(1000): # For statistic significance
		chimp_real_test.q_df = deepcopy(q_df)
		chimp_real_test.q_dict = deepcopy(q_dict)
		chimp_real_test.q_reg = deepcopy(q_reg)
		chimp_real_test.epsilon = 0.01

		for j in range(len(chimp_real_test.env)):
			print("Iter-Row: {0}-{1}".format(i, j))
			chimp_real_test.update()
		chimp_real_test.reset()

	with open('chimp_real_test_11500_pv_history.pickle', 'wb') as f:
		pickle.dump(chimp_real_test.pv_history_list, f, pickle.HIGHEST_PROTOCOL)
	print(chimp_real_test.pv_history_list)


if __name__ == '__main__':
	main()