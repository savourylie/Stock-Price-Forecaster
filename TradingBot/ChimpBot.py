from __future__ import division

from monkeybot import MonkeyBot

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
from sklearn.ensemble import RandomForestRegressor

class ChimpBot(MonkeyBot):
    """An agent that learns to drive in the smartcab world."""
    valid_actions = ['Buy', 'Sell', 'Hold']
    num_trial = 500
    trial_counter = 0 # For getting the trial number
    total_net_reward = 0 # For counting total reward


    random_rounds = 100 # Number of rounds where the bot chooses to go monkey

    trial_meta_info = {} # For monitoring what happens in each trial

    epsilon = 1
    gamma = 0.8
    random_reward = [0]

    random_counter = 0
    policy_counter = 0

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

        self.q_df_columns = list(self.env.columns)
        self.q_df_columns.pop()
        self.q_df_columns.append('Q Value')
        self.q_df = pd.DataFrame(columns=self.q_df_columns)
        self.q_dict = defaultdict(lambda: [0, 0]) # element of q_dict is (state, act): [q_value, t]

        self.net_reward = 0

        # Smartcab use only
        # self.penalty = False
        # self.num_step = 0 # Number of steps for each trial; get reset each time a new trial begins

    def update_q_df(self):
        self.q_df = pd.DataFrame(columns=self.q_df_columns)

        for key, qAndT in q_dict.iteritems():
            key_list = list(key)
            key_list.append(qAndT[0])
            dfTemp = pd.DataFrame(key_list)
            self.q_df = self.q_df.append(dfTemp, ignore_index=True)

        self.split_q_df()

    def split_q_df(self):
        self.q_df_train = self.q_df.ix[:, :-1]
        self.q_df_test = self.q_df.ix[:, -1]

    def train_on_q_df(self):
        self.q_reg = RandomForestRegressor(n_estimators=64)
        self.q_reg = self.q_reg.fit(self.q_df_train, self.q_df_test)


    def max_q(self):
        # start = time.time()
        max_q = ''
        q_compare_dict = {}

        # Populate the q_dict
        for act in set(self.valid_actions):
            _ = self.q_dict[(self.now_row[0], \
                self.now_row[1], \
                self.now_row[2], \
                self.now_row[3], \
                self.now_row[4], \
                self.now_row[5], \
                self.now_row[6], \
                self.now_row[7], \
                self.now_row[8], \
                self.now_row[9], \
                self.now_row[10], \
                self.now_row[11], \
                self.now_row[12], \
                self.now_row[13], \
                self.now_row[14], \
                self.now_row[15], \
                self.now_row[16], \
                self.now_row[17], \
                self.now_row[18], \
                self.now_row[19], \
                self.now_row[20], \
                self.now_row[21], \
                self.now_row[22], \
                self.now_row[23], \
                self.now_row[24], \
                self.now_row[25], \
                self.now_row[26], \
                self.now_row[27], \
                self.now_row[28], \
                self.now_row[29], \
                self.now_row[30], \
                self.now_row[31], \
                self.now_row[32], \
                self.now_row[33], \
                self.now_row[34], \
                self.now_row[35], \
                self.now_row[36], \
                act)]

            if np.random.choice(2, p = [0.5, 1 - 0.5]) == 0 and len(self.q_dict) > 5000:
                test_X = list(self.now_row[:36])
                test_X.append(act)
                pred_q = self.q_reg.predict(test_X.reshape(1, -1))

                self.q_dict[(self.now_row[0], \
                self.now_row[1], \
                self.now_row[2], \
                self.now_row[3], \
                self.now_row[4], \
                self.now_row[5], \
                self.now_row[6], \
                self.now_row[7], \
                self.now_row[8], \
                self.now_row[9], \
                self.now_row[10], \
                self.now_row[11], \
                self.now_row[12], \
                self.now_row[13], \
                self.now_row[14], \
                self.now_row[15], \
                self.now_row[16], \
                self.now_row[17], \
                self.now_row[18], \
                self.now_row[19], \
                self.now_row[20], \
                self.now_row[21], \
                self.now_row[22], \
                self.now_row[23], \
                self.now_row[24], \
                self.now_row[25], \
                self.now_row[26], \
                self.now_row[27], \
                self.now_row[28], \
                self.now_row[29], \
                self.now_row[30], \
                self.now_row[31], \
                self.now_row[32], \
                self.now_row[33], \
                self.now_row[34], \
                self.now_row[35], \
                self.now_row[36], \
                act)] \
                = (pred_q, self.q_dict[(self.now_row[0], \
                self.now_row[1], \
                self.now_row[2], \
                self.now_row[3], \
                self.now_row[4], \
                self.now_row[5], \
                self.now_row[6], \
                self.now_row[7], \
                self.now_row[8], \
                self.now_row[9], \
                self.now_row[10], \
                self.now_row[11], \
                self.now_row[12], \
                self.now_row[13], \
                self.now_row[14], \
                self.now_row[15], \
                self.now_row[16], \
                self.now_row[17], \
                self.now_row[18], \
                self.now_row[19], \
                self.now_row[20], \
                self.now_row[21], \
                self.now_row[22], \
                self.now_row[23], \
                self.now_row[24], \
                self.now_row[25], \
                self.now_row[26], \
                self.now_row[27], \
                self.now_row[28], \
                self.now_row[29], \
                self.now_row[30], \
                self.now_row[31], \
                self.now_row[32], \
                self.now_row[33], \
                self.now_row[34], \
                self.now_row[35], \
                self.now_row[36], \
                act)][1] + 1)

            q_compare_dict[(self.now_row[0], \
            self.now_row[1], \
            self.now_row[2], \
            self.now_row[3], \
            self.now_row[4], \
            self.now_row[5], \
            self.now_row[6], \
            self.now_row[7], \
            self.now_row[8], \
            self.now_row[9], \
            self.now_row[10], \
            self.now_row[11], \
            self.now_row[12], \
            self.now_row[13], \
            self.now_row[14], \
            self.now_row[15], \
            self.now_row[16], \
            self.now_row[17], \
            self.now_row[18], \
            self.now_row[19], \
            self.now_row[20], \
            self.now_row[21], \
            self.now_row[22], \
            self.now_row[23], \
            self.now_row[24], \
            self.now_row[25], \
            self.now_row[26], \
            self.now_row[27], \
            self.now_row[28], \
            self.now_row[29], \
            self.now_row[30], \
            self.now_row[31], \
            self.now_row[32], \
            self.now_row[33], \
            self.now_row[34], \
            self.now_row[35], \
            self.now_row[36], \
            act)] \
            = self.q_dict[(self.now_row[0], \
            self.now_row[1], \
            self.now_row[2], \
            self.now_row[3], \
            self.now_row[4], \
            self.now_row[5], \
            self.now_row[6], \
            self.now_row[7], \
            self.now_row[8], \
            self.now_row[9], \
            self.now_row[10], \
            self.now_row[11], \
            self.now_row[12], \
            self.now_row[13], \
            self.now_row[14], \
            self.now_row[15], \
            self.now_row[16], \
            self.now_row[17], \
            self.now_row[18], \
            self.now_row[19], \
            self.now_row[20], \
            self.now_row[21], \
            self.now_row[22], \
            self.now_row[23], \
            self.now_row[24], \
            self.now_row[25], \
            self.now_row[26], \
            self.now_row[27], \
            self.now_row[28], \
            self.now_row[29], \
            self.now_row[30], \
            self.now_row[31], \
            self.now_row[32], \
            self.now_row[33], \
            self.now_row[34], \
            self.now_row[35], \
            self.now_row[36], \
            act)]

        try:
            max(q_compare_dict.iteritems(), key=lambda x:x[1])
        except ValueError:
            print("Wrong Q Value in Q Compare Dict!")
        else:
            key, qAndT = max(q_compare_dict.iteritems(), key=lambda x:x[1])
            return key[-1], qAndT[0], qAndT[1]

    def q_update(self):
        q_temp = self.q_dict[(self.prev_states[0], \
            self.prev_states[1], \
            self.prev_states[2], \
            self.prev_states[3], \
            self.prev_states[4], \
            self.prev_states[5], \
            self.prev_states[6], \
            self.prev_states[7], \
            self.prev_states[8], \
            self.prev_states[9], \
            self.prev_states[10], \
            self.prev_states[11], \
            self.prev_states[12], \
            self.prev_states[13], \
            self.prev_states[14], \
            self.prev_states[15], \
            self.prev_states[16], \
            self.prev_states[17], \
            self.prev_states[18], \
            self.prev_states[19], \
            self.prev_states[20], \
            self.prev_states[21], \
            self.prev_states[22], \
            self.prev_states[23], \
            self.prev_states[24], \
            self.prev_states[25], \
            self.prev_states[26], \
            self.prev_states[27], \
            self.prev_states[28], \
            self.prev_states[29], \
            self.prev_states[30], \
            self.prev_states[31], \
            self.prev_states[32], \
            self.prev_states[33], \
            self.prev_states[34], \
            self.prev_states[35], \
            self.prev_states[36], \
            self.prev_states[37], \
            self.prev_action)]

        q_temp0 = (1 - (1 / (q_temp[1] + 1))) * q_temp[0] + (1 / (q_temp[1] + 1)) * (self.prev_reward + self.gamma * self.max_q()[1])

        self.q_dict[(self.prev_states[0], \
            self.prev_states[1], \
            self.prev_states[2], \
            self.prev_states[3], \
            self.prev_states[4], \
            self.prev_states[5], \
            self.prev_states[6], \
            self.prev_states[7], \
            self.prev_states[8], \
            self.prev_states[9], \
            self.prev_states[10], \
            self.prev_states[11], \
            self.prev_states[12], \
            self.prev_states[13], \
            self.prev_states[14], \
            self.prev_states[15], \
            self.prev_states[16], \
            self.prev_states[17], \
            self.prev_states[18], \
            self.prev_states[19], \
            self.prev_states[20], \
            self.prev_states[21], \
            self.prev_states[22], \
            self.prev_states[23], \
            self.prev_states[24], \
            self.prev_states[25], \
            self.prev_states[26], \
            self.prev_states[27], \
            self.prev_states[28], \
            self.prev_states[29], \
            self.prev_states[30], \
            self.prev_states[31], \
            self.prev_states[32], \
            self.prev_states[33], \
            self.prev_states[34], \
            self.prev_states[35], \
            self.prev_states[36], \
            self.prev_states[37], \
            self.prev_action)] \
            = (q_temp0, q_temp[1] + 1)

        return ([(self.prev_states[0], \
            self.prev_states[1], \
            self.prev_states[2], \
            self.prev_states[3], \
            self.prev_states[4], \
            self.prev_states[5], \
            self.prev_states[6], \
            self.prev_states[7], \
            self.prev_states[8], \
            self.prev_states[9], \
            self.prev_states[10], \
            self.prev_states[11], \
            self.prev_states[12], \
            self.prev_states[13], \
            self.prev_states[14], \
            self.prev_states[15], \
            self.prev_states[16], \
            self.prev_states[17], \
            self.prev_states[18], \
            self.prev_states[19], \
            self.prev_states[20], \
            self.prev_states[21], \
            self.prev_states[22], \
            self.prev_states[23], \
            self.prev_states[24], \
            self.prev_states[25], \
            self.prev_states[26], \
            self.prev_states[27], \
            self.prev_states[28], \
            self.prev_states[29], \
            self.prev_states[30], \
            self.prev_states[31], \
            self.prev_states[32], \
            self.prev_states[33], \
            self.prev_states[34], \
            self.prev_states[35], \
            self.prev_states[36], \
            self.prev_states[37], \
            self.prev_action)])

    def policy(self):
        return self.max_q()[0]

    def reset(self):
        self.cash = 1000
        self.share = 0
        self.pv = 0

        if self.epsilon - 1/self.random_rounds > 0.2: # Epislon threshold: 0.2
            self.random_counter += 1
            self.epsilon = self.epsilon - 1/self.random_rounds
        else:
            self.epsilon = 0.2 # Epislon threshold: 0.2
            self.policy_counter += 1

        self.net_reward = 0
        self.num_step = 0 # Recalculate the steps for the new trial
        self.penalty = False
        self.fail = False

    def update(self, t):
        # Update state
        now_states = self.now_row

        # Exploitation-exploration decisioning
        self.decision = np.random.choice(2, p = [self.epsilon, 1 - self.epsilon]) # decide to go random or with the policy
        # self.decision = 0 # Force random mode

        # print("random decision: {}".format(self.decision))
        if self.decision == 0: # if zero, go random
            action = random.choice(self.valid_actions)
        else: # else go with the policy
            action = self.policy()


        # Execute action and get reward
        if action == 'Buy':
            self.buy(now_states[-1])
        elif action == 'Sell':
            self.sell(now_states[-1])
        else:
            self.hold(now_states[-1])

        try:
            self.prev_states
        except AttributeError:
            print("Running the first time...no prevs exist.")
        else:
            reward = (self.cash - self.prev_cash) + (self.pv - self.prev)
            self.q_update()

        self.prev_states = now_states
        self.prev_action = action
        self.prev_reward = reward

        try:
            self.now_env_index, self.now_row = self.iter_env.next()
        except StopIteration:
            print("End of data. Start again.")
        else:
            pass

        self.net_reward += reward

        # if reward < 0:
        #     self.penalty = True

        self.total_net_reward += reward

        print "ChimpBot.update(): Action: {0} at Price: {1}, Cash + PV = {2}, Reward = {3}".format(action, now_states[-1], self.cash + self.pv, reward)  # [debug]


