from chimpbot import ChimpBot
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import pickle
import time

def main_simulate():
    # Initiating data and the chimp
    dfFull = pd.read_csv('data_full.csv', index_col=0, parse_dates=True, na_values = ['nan'])
    train_size = 8095
    date_range = dfFull.index[train_size:] # Using one-year data to predict one day
    print(date_range)
    day_count = 0
    pv_history_list = []

    cash = 1000
    share = 0
    pv = 0
    now_yes_share = 0

    for date in date_range:
        start_time = time.time()
        day_count += 1
        print("Day {}".format(day_count))

        dfTest = dfFull.ix[date].to_frame().T
        (u,) = dfFull.index.get_indexer_for([date])

        if u - train_size < 0:
            raise ValueError("Not enough training data!")

        # dfTrain = dfFull.iloc[(u - train_size):u]
        dfTrain = dfFull.iloc[:u]

        chimp_train = ChimpBot(dfTrain)

        for i in range(3500):
            for l in range(len(chimp_train.env)):
                # print("Train Round {0}-{1}".format(i + 1, l + 1))
                chimp_train.update()
            chimp_train.reset()

        # Test the Chimp!
        q_df = deepcopy(chimp_train.q_df)
        q_dict = deepcopy(chimp_train.q_dict)
        q_reg = deepcopy(chimp_train.q_reg)

        try:
            _ = chimp_test
        except NameError:
            print("First time running...")
        else:
            cash = chimp_test.cash
            share = chimp_test.share
            pv = chimp_test.pv
            now_yes_share = chimp_test.now_yes_share

        chimp_test = ChimpBot(dfTest, cash=cash, share=share, pv=pv, now_yes_share=now_yes_share)

        chimp_test.q_df = deepcopy(q_df)
        chimp_test.q_dict = deepcopy(q_dict)
        chimp_test.q_reg = deepcopy(q_reg)
        chimp_test.epsilon = 0

        # Pass the cheatsheet to the next chimp
        try:
            chimp_test.prev_states = prev_states
            chimp_test.now_action = now_action
            chimp_test.prev_action = prev_action
            chimp_test.prev_yes_share = prev_yes_share
            chimp_test.reward = reward
            chimp_test.prev_cash = prev_cash
            chimp_test.prev_share = prev_share
            chimp_test.prev_pv = prev_pv
            chimp_test.prev_env_index = prev_env_index

        except UnboundLocalError:
            print("No cheatsheet to pass over yet...no worries!")
        else:
            print("Prev States: {}".format(prev_states))
            print("Prev Action: {}".format(prev_action))

        chimp_test.update()

        # Create cheatsheet for the next chimp
        prev_states = chimp_test.prev_states
        now_action = chimp_test.now_action
        prev_action = chimp_test.prev_action
        prev_yes_share = chimp_test.prev_yes_share
        prev_env_index = chimp_test.prev_env_index
        reward = chimp_test.reward
        prev_cash = chimp_test.prev_cash
        prev_share = chimp_test.prev_share
        prev_pv = chimp_test.prev_pv

        pv_history_list.append(chimp_test.cash + chimp_test.pv)

        print("Training + Prediction for Day {0} took {1} seconds.".format(day_count, time.time() - start_time))
        print("Now States: {}".format((chimp_test.now_env_index, chimp_test.now_row.to_frame().T)))
        print("Now Action: {}".format(chimp_test.now_action))
        print "ChimpTest.update(): Action: {0} at Price: {1}, Cash: {2}, Num_Share: {3}, Cash + PV = {4}, Reward = {5}".format(now_action, chimp_test.now_row[-1], chimp_test.cash, chimp_test.share, chimp_test.cash + chimp_test.pv, chimp_test.reward)  # [debug]
        print('Portfolio + Cash: {}'.format(chimp_test.cash + chimp_test.pv))
        print("================================")

        if day_count % 10 == 0:
            print(pv_history_list)
    print(pv_history_list)

    # with open('chimp_full_11500_pv_history.pickle', 'wb') as f1:
    #   pickle.dump(chimp_full.pv_history_list, f1, pickle.HIGHEST_PROTOCOL)
    # print(chimp_full.pv_history_list)

    # with open('chimp_train_11500_pv_history.pickle', 'wb') as f1:
    #   pickle.dump(chimp_train.pv_history_list, f1, pickle.HIGHEST_PROTOCOL)
    # print(chimp_train.pv_history_list)

    # with open('chimp_test_11500_pv_history.pickle', 'wb') as f2:
    #   pickle.dump(chimp_test.pv_history_list, f2, pickle.HIGHEST_PROTOCOL)
    # print(chimp_test.pv_history_list)

    # Convert Q-Table to Dataframe from trained chimp (full)
    # print(chimp_full.q_dict)
    # result_dict_full = defaultdict(list)
    # for index, row in chimp_full.q_dict.iteritems():
    #     for i in range(len(chimp_full.q_dict.keys()[0])):
    #         column_name = 'col' + str(i + 1)
    #         result_dict_full[column_name].append(index[i])
    #     result_dict_full['Q'].append(chimp_full.q_dict[index][0])

    # q_df = pd.DataFrame(result_dict_full)
    # q_df.to_csv('q_df_100_full.csv')

    # # Convert Q-Table to Dataframe from trained chimp (train)
    # result_dict_train = defaultdict(list)
    # for index, row in chimp_train.q_dict.iteritems():
    #     for i in range(len(chimp_train.q_dict.keys()[0])):
    #         column_name = 'col' + str(i + 1)
    #         result_dict_train[column_name].append(index[i])
    #     result_dict_train['Q'].append(chimp_train.q_dict[index][0])

    # q_df = pd.DataFrame(result_dict_train)
    # q_df.to_csv('q_df_11500_train.csv')

    # # Convert Q-Table to Dataframe from trained chimp (test)
    # result_dict_test = defaultdict(list)
    # for index, row in chimp_test.q_dict.iteritems():
    #     for i in range(len(chimp_test.q_dict.keys()[0])):
    #         column_name = 'col' + str(i + 1)
    #         result_dict_test[column_name].append(index[i])
    #     result_dict_test['Q'].append(chimp_test.q_dict[index][0])

    # q_df = pd.DataFrame(result_dict_test)
    # q_df.to_csv('q_df_11500_test.csv')

    # Save the chimp train properties <--- Not working
    # Save q_df
    # with open('3200_train_q_df.pickle', 'wb') as f:
    #   pickle.dump(chimp_train.q_df, f, pickle.HIGHEST_PROTOCOL)
    # # Save q_dict
    # with open('3200_train_q_dict.pickle', 'wb') as f:
    #   pickle.dump(chimp_train.q_dict, f, pickle.HIGHEST_PROTOCOL)
    # # Save q_reg
    # with open('3200_train_q_reg.pickle', 'wb') as f:
    #   pickle.dump(chimp_train.q_reg, f, pickle.HIGHEST_PROTOCOL)

    # try:
    #   print(chimp_train.q_dict)
    # except AttributeError:
    #   print("No q_dict? No big deal I guess...?")

    # # Test the Chimp!
    # q_df = deepcopy(chimp_train.q_df)
    # q_dict = deepcopy(chimp_train.q_dict)
    # q_reg = deepcopy(chimp_train.q_reg)

    # chimp_real_test = ChimpBot(dfTest)

    # for i in range(1000): # For statistic significance
    #   chimp_real_test.q_df = deepcopy(q_df)
    #   chimp_real_test.q_dict = deepcopy(q_dict)
    #   chimp_real_test.q_reg = deepcopy(q_reg)
    #   chimp_real_test.epsilon = 0.01

    #   for j in range(len(chimp_real_test.env)):
    #       print("Iter-Row: {0}-{1}".format(i, j))
    #       chimp_real_test.update()
    #   chimp_real_test.reset()

    # with open('chimp_real_test_11500_pv_history.pickle', 'wb') as f:
    #   pickle.dump(chimp_real_test.pv_history_list, f, pickle.HIGHEST_PROTOCOL)
    # print(chimp_real_test.pv_history_list)

def generate_Q_table():
    dfFull = pd.read_csv('data_full.csv', index_col=0, parse_dates=True, na_values = ['nan'])
    num_iter = 3500
    day_count = 0
    pv_history_list = []

    chimp = ChimpBot(dfFull)

    for i in range(num_iter):
        print("Iter {}".format(i + 1))
        for l in range(len(chimp.env)):
            chimp.update()
        print("Total Asset {}: ".format(chimp.cash + chimp.pv))
        pv_history_list.append(chimp.cash + chimp.pv)
        chimp.reset()

    print(pv_history_list)

    # Convert Q-Table to Dataframe from trained chimp (full)
    result_dict = defaultdict(list)
    for index, row in chimp.q_dict_analysis.iteritems():
        for i in range(len(chimp.q_dict_analysis.keys()[0])):
            column_name = 'col' + str(i + 1)
            result_dict[column_name].append(index[i])
        result_dict['Q'].append(chimp.q_dict_analysis[index][0])
        result_dict['Date'].append(chimp.q_dict_analysis[index][1])

    q_df = pd.DataFrame(result_dict)
    q_df.to_csv('q_df_' + str(num_iter) + '_full.csv')


if __name__ == '__main__':
    main_simulate()
    generate_Q_table()
