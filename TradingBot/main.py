from chimpbot import ChimpBot
import pandas as pd
from collections import defaultdict

def main():
	# Initiating data and the chimp
	# dfEnv = pd.read_csv('data_train_char.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	dfEnv = pd.read_csv('data_train.csv', index_col=0, parse_dates=True, na_values = ['nan'])
	# dfEnv.ix[:, :-1] = dfEnv.ix[:, :-1].astype('int')
	chimp = ChimpBot(dfEnv)

	print(chimp.env)
	q_dict_length = []

	for i in range(500):
		for j in range(len(chimp.env)):
			print("{0}-{1}".format(i + 1, j + 1))
			chimp.update()
		q_dict_length.append(len(chimp.q_dict))
		chimp.reset()
	print(chimp.pv_history_list)
	print(q_dict_length)

	# Convert Q-Table to Dataframe
	result_dict = defaultdict(list)
	for index, row in chimp.q_dict.iteritems():
	    for i in range(len(chimp.q_dict.keys()[0])):
	        column_name = 'col' + str(i + 1)
	        result_dict[column_name].append(index[i])
	    result_dict['Q'].append(chimp.q_dict[index][0])

	q_df = pd.DataFrame(result_dict)
	q_df.to_csv('q_df.csv')

	# print(chimp.track_key1)
	# print(chimp.track_key2)
	# print("Track Random Decision: {}".format(chimp.track_random_decision))
	# print(chimp.q_dict.keys())


if __name__ == '__main__':
	main()