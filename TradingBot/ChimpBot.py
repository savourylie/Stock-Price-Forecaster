class ChimpBot(MonkeyBot):
	"""An agent that learns to drive in the smartcab world."""
	valid_actions = ['Buy', 'Sell', 'Hold']

    num_trial = 500

    total_net_reward = 0 # For counting total reward
    update_counter = 0 # For counting total steps
    trial_counter = 0 # For getting the trial number
    random_rounds = 100

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

	    self.q_dict = defaultdict(lambda: [0, 0]) # element of q_dict is (state, act): [q_value, t]
        self.net_reward = 0
        self.num_step = 0 # Number of steps for each trial; get reset each time a new trial begins
        self.state = ()

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

    def q_update(self, now_states):
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

    def reset(self, destination=None):
        self.planner.route_to(destination)

        if self.epsilon - 1/self.random_rounds > 0:
            self.random_counter += 1
            self.epsilon = self.epsilon - 1/self.random_rounds
        else:
            self.epsilon = 0
            self.policy_counter += 1

        self.net_reward = 0
        self.num_step = 0 # Recalculate the steps for the new trial
        self.penalty = False
        self.fail = False


