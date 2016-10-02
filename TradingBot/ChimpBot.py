from __future__ import division

from monkeybot import MonkeyBot

from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import grid_search

class ChimpBot(MonkeyBot):
    """An agent that learns to drive in the smartcab world."""
    valid_actions = ['Buy', 'Sell', 'Hold']
    num_trial = 500
    trial_counter = 0 # For getting the trial number
    total_net_reward = 0 # For counting total reward


    random_rounds = 150 # Number of rounds where the bot chooses to go monkey

    trial_meta_info = {} # For monitoring what happens in each trial

    epsilon = 1
    gamma = 0.8
    random_reward = [0]

    random_counter = 0
    policy_counter = 0

    track_key1 = {'Sell': 0, 'Buy': 0, 'Hold': 0}
    track_key2 = {'Sell': 0, 'Buy': 0, 'Hold': 0}

    track_random_decision = {'Sell': 0, 'Buy': 0, 'Hold': 0}

    reset_counter = 0

    def __init__(self, dfEnv):
        super(ChimpBot, self).__init__(dfEnv)
        # sets self.cash = 1000
        # sets self.share = 0
        # sets self.pv = 0
        # sets self.pv_history_list = []
        # sets self.env = dfEnv
        # implements buy(self, stock_price)
        # implements sell(self, stock_price)
        # implements hold(self)

        self.iter_env = self.env.iterrows()
        self.now_env_index, self.now_row = self.iter_env.next()

        self.now_yes_share = 0
        self.now_action = ''
        # self.now_q = 0

        self.prev_cash = self.cash
        self.prev_share = self.share
        self.prev_pv = self.pv

        self.q_df_columns = list(self.env.columns)
        self.q_df_columns.pop()
        self.q_df_columns.extend(['Yes Share', 'Action', 'Q Value'])
        self.q_df = pd.DataFrame(columns=self.q_df_columns)
        self.q_dict = defaultdict(lambda: (0, 0)) # element of q_dict is (state, act): [q_value, t]

        self.negative_reward = 0
        self.n_reward_hisotry = []
        self.net_reward = 0

        self.reset_counter = 0

        # Smartcab use only
        # self.penalty = False
        # self.num_step = 0 # Number of steps for each trial; get reset each time a new trial begins

    def make_q_df(self):
        result_dict = defaultdict(list)

        for index, row in self.q_dict.iteritems():
            for i in range(len(self.q_dict.keys()[0])):
                column_name = 'col' + str(i + 1)
                result_dict[column_name].append(index[i])
            result_dict['Q'].append(self.q_dict[index][0])

        self.q_df = pd.DataFrame(result_dict)
        q_df_column_list = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24', 'col25', 'col26', 'col27', 'col28', 'col29', 'col30', 'col31', 'col32', 'col33', 'col34', 'col35', 'col36', 'col37', 'col38', 'col39', 'Q']
        self.q_df = self.q_df[q_df_column_list]

        def transfer_action(x):
            if x == 'Buy':
                return 1
            elif x == 'Sell':
                return 2
            elif x == 'Hold':
                return 0
            else:
                raise ValueError("Wrong action!")

        def str_float_int(x):
            return int(float(x))

        arr_int = np.vectorize(str_float_int)

        self.q_df['col39'] = self.q_df['col39'].apply(transfer_action)
        print(self.q_df)
        # self.q_df.to_csv('temp_q_df.csv')
        # self.q_df = pd.read_csv('temp_q_df.csv', index_col=0, parse_dates=True, na_values = ['nan'])
        # self.q_df.dropna(inplace=True)
        self.q_df.ix[:, :-1] = self.q_df.ix[:, :-1].apply(arr_int)

    def split_q_df(self):
        self.q_df_X = self.q_df.ix[:, :-1]
        self.q_df_y = self.q_df.ix[:, -1]
        # self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.q_df_X, self.q_df_y, test_size=0.1, random_state=0)

    def train_on_q_df(self):
        reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=20), random_state=0)
        ada_parameters = {'n_estimators':[50]}
        reg_gs = grid_search.GridSearchCV(reg, ada_parameters, cv=9)

        # self.q_reg = KNeighborsRegressor(n_neighbors=5)
        self.q_reg = reg_gs

        print(self.q_df_X)
        self.q_reg = self.q_reg.fit(self.q_df_X, self.q_df_y)

    def update_q_model(self):
        print("Updating Q model...")
        start_time = time.time()
        self.make_q_df()
        self.split_q_df()
        self.train_on_q_df()
        print("Update took {} seconds".format(time.time() - start_time))

    def from_state_action_predict_q(self, state_action):
        state_action = [state_action]

        pred_q = self.q_reg.predict(state_action)

        return pred_q

    def yes_share(self):
        # Represent chimp asset in state_action
        if self.share > 0:
            return 1
        else:
            return 0

    def max_q(self, now_row):
        def transfer_action(x):
            if x == 'Buy':
                return 1
            elif x == 'Sell':
                return 2
            elif x == 'Hold':
                return 0
            else:
                raise ValueError("Wrong action!")

        def str_float_int(x):
            return int(float(x))

        now_row2 = list(now_row)
        now_row2.append(self.now_yes_share)
        max_q = ''
        q_compare_dict = {}

        if len(now_row2) > 38:
            raise ValueError("Got ya bastard! @ MaxQ")

        # Populate the q_dict
        for act in set(self.valid_actions):
            now_row2.append(act)
            now_row_key = tuple(now_row2)

            _ = self.q_dict[now_row_key]

            # # K-Q Algorithm
            # if np.random.choice(2, p = [0.9, 0.1]) == 1 and len(self.q_dict) > 30000:
            # if _[1] == 0 and np.random.choice(2, p = [0.7, 0.3]) == 1 and len(self.q_dict) > 30000:
            if _[1] == 0 and len(self.q_dict) > 20000:
                print("Dreaming mode...")
                start_time = time.time()
                # self.update_q_model()

                single_X = np.array(now_row_key)
                # print(single_X)
                arr_int = np.vectorize(str_float_int)
                single_X[-1] = transfer_action(single_X[-1])
                single_X = arr_int(single_X)
                single_X = single_X.reshape(1, -1)
                pred_q = self.q_reg.predict(single_X)
                dreamed_q = (1 - (1 / (self.q_dict[now_row_key][1] + 1))) * self.q_dict[now_row_key][0] + (1 / (self.q_dict[now_row_key][1] + 1)) * pred_q[0]
                self.q_dict[now_row_key] = (dreamed_q, self.q_dict[now_row_key][1] + 1)
                print("Q-dreamed: {0} for Act: {1}, taking {2} seconds.".format(self.q_dict[now_row_key], act, time.time() - start_time))


            print(act, self.q_dict[now_row_key])

            q_compare_dict[now_row_key] = self.q_dict[now_row_key]
            now_row2.pop()

        try:
            max(q_compare_dict.iteritems(), key=lambda x:x[1])
        except ValueError:
            print("Wrong Q Value in Q Compare Dict!")
        else:
            key, qAndT = max(q_compare_dict.iteritems(), key=lambda x:x[1])
            print("Action: {0}, with Q-value: {1}".format(key[-1], qAndT[0]))
            return key[-1], qAndT[0], qAndT[1]

    def q_update(self):
        print("Data Index: {}".format(self.now_env_index))
        now_states = list(self.now_row)
        # now_states = list(now_states)
        now_states.pop() # disregard the Trade Price

        prev_states = list(self.prev_states)

        if len(prev_states) > 37:
            raise ValueError("Got ya bastard! @ Q_Update...something wrong with the self.prev_states!!!")

        prev_states.append(self.prev_yes_share)
        prev_states.append(self.prev_action)
        prev_states_key = tuple(prev_states)

        if len(prev_states_key) > 39:
            raise ValueError("Got ya bastard! @ Q_Update")

        q_temp = self.q_dict[prev_states_key]

        q_temp0 = (1 - (1 / (q_temp[1] + 1))) * q_temp[0] + (1 / (q_temp[1] + 1)) * (self.prev_reward + self.gamma * self.max_q(now_states)[1])

        if prev_states_key[:-1] == ('Low', 'Low', 'Average', 'Average', 'Low', 'Average', 'Average', 'Average', 'Low', 'Low', 'Low', 'Low', 'Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'N-Very Low', 'Low', 'Average', 'N-Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'High', 'Yes'):
            self.track_key1[prev_states_key[-1]] += 1
        elif prev_states_key[:-1] == ('Low', 'Low', 'Average', 'Average', 'Low', 'Average', 'Average', 'Average', 'Low', 'Low', 'Low', 'Low', 'Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'N-Very Low', 'Low', 'Average', 'N-Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'High', 'No'):
            self.track_key2[prev_states_key[-1]] += 1
        # elif prev_states_key[:-1] == ('Very High', 'Very High', 'Very High', 'Very High', 'Very High', 'Very High', 'Average', 'High', 'Average', 'Average', 'Average', 'Low', 'Average', 'Very Low', 'Low', 'N-Very Low', 'N-Very Low', 'N-Very Low', 'N-Very Low', 'Very Low', 'Very Low', 'Average', 'Very Low', 'Low', 'Low', 'Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Low', 'Very Low', 'Low', 'Very Low', 'Average', 'No'):
        #     self.track_key2[prev_states_key[-1]] += 1

        self.q_dict[prev_states_key] = (q_temp0, q_temp[1] + 1)
        # print("Now Action: {}".format())
        print(prev_states_key)
        return (self.q_dict[prev_states_key])

    def policy(self, now_row):
        return self.max_q(now_row)[0]

    def reset(self):
        # Portfolio change over iterations
        self.pv_history_list.append(self.pv + self.cash)

        self.iter_env = self.env.iterrows()
        self.now_env_index, self.now_row = self.iter_env.next()

        self.cash = 1000
        self.share = 0
        self.pv = 0

        self.prev_cash = self.cash
        self.prev_share = self.share
        self.prev_pv = self.pv

        if self.epsilon - 1/self.random_rounds > 0.05: # Epislon threshold: 0.05
            self.random_counter += 1
            self.epsilon = self.epsilon - 1/self.random_rounds
        else:
            self.epsilon = 0.05 # Epislon threshold: 0.1
            self.policy_counter += 1

        self.net_reward = 0

        if self.reset_counter % 100 == 0:
            self.update_q_model()

        self.reset_counter += 1
        # self.num_step = 0 # Recalculate the steps for the new trial
        # self.penalty = False
        # self.fail = False

    def make_decision(self, now_row):
        return self.policy(now_row)

    def update(self):
        # Update state
        now_states = list(self.now_row)

        if len(now_states) > 38:
            print(now_states)
            raise ValueError("Got ya bastard! @ Q_Update...something wrong with the self.now_row!!!")

        # now_states = list(now_states)
        # print(type(self.now_row))
        now_states.pop() # disregard the Trade Price

        if len(now_states) > 37:
            print(now_states)
            raise ValueError("Got ya bastard! @ Q_Update...something wrong with now_states after pop!!!")

        # Exploitation-exploration decisioning
        random.seed(datetime.now())
        self.decision = np.random.choice(2, p = [self.epsilon, 1 - self.epsilon]) # decide to go random or with the policy
        # self.decision = 0 # Force random mode

        print("Random decision: {0}, Epislon: {1}".format(self.decision, self.epsilon))
        # print("What the FUCK?!")
        if self.decision == 0: # if zero, go random
            random.seed(datetime.now())
            action = random.choice(self.valid_actions)
            if tuple(now_states) == ('Low', 'Low', 'Average', 'Average', 'Low', 'Average', 'Average', 'Average', 'Low', 'Low', 'Low', 'Low', 'Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'N-Very Low', 'Low', 'Average', 'N-Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'Very Low', 'High'):
                self.track_random_decision[action] += 1
        else: # else go with the policy
            print("now_states: {}".format(now_states))
            action = self.make_decision(now_states)

        if len(now_states) > 37:
            print(now_states)
            raise ValueError("Got ya bastard! @ Q_Update...something wrong with now_states after make_decision!!!")

        self.now_yes_share = self.yes_share()

        print("Now Action: {}".format(action))
        # Execute action and get reward
        if action == 'Buy':
            # print(self.now_row)
            self.buy(self.now_row[-1])
        elif action == 'Sell':
            # print(self.now_row)
            self.sell(self.now_row[-1])
        elif action == 'Hold':
            # print(self.now_row)
            self.hold(self.now_row[-1])
        else:
            raise ValueError("Wrong action man!")

        try:
            self.prev_states
        except AttributeError:
            print("Running the first time...no prevs exist.")
        else:
            self.q_update()

        reward = ((self.cash - self.prev_cash) + (self.pv - self.prev_pv)) / (self.prev_cash + self.prev_pv)

        self.prev_states = now_states

        if len(now_states) > 37:
            raise ValueError("Got ya bastard! @ Q_Update...something wrong with the now_states!!!")

        self.now_action = action
        self.prev_action = action
        self.prev_yes_share = self.now_yes_share
        self.prev_reward = reward
        self.prev_cash = self.cash
        self.prev_share = self.share
        self.prev_pv = self.pv

        # if len(self.q_dict) > 20000:
        #     self.update_q_model()

        try:
            self.now_env_index, self.now_row = self.iter_env.next()
        except StopIteration:
            print("End of data.")
        else:
            pass

        self.net_reward += reward

        # if reward < 0:
        #     self.penalty = True

        self.total_net_reward += reward

        print "ChimpBot.update(): Action: {0} at Price: {1}, Cash: {2}, Num_Share: {3}, Cash + PV = {4}, Reward = {5}".format(action, self.now_row[-1], self.cash, self.share, self.cash + self.pv, reward)  # [debug]
        print('Portfolio + Cash: {}'.format(self.cash + self.pv))
        print("================================")

