from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
import numpy as np

"""
Code by https://github.com/philtabor
at https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/dqn_keras.py
Modified by Gonzalo Miranda.
"""

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                          dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    
    def store_transition(self, state, action, reward, state_, done):
        '''
        Stores experiences from the agent.
        '''
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        '''
            Returns a batch of random sampled experiences from the memory buffer.
        '''
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                     input_shape=(*input_dims,), data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                     data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                     data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=RMSprop(lr=lr), loss='mean_squared_error')

    return model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace,
                 input_dims, eps_dec=1e-5,  eps_min=0.01,
                 mem_size=1000000, save_name='dqn_model',
                 load_name='target_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace = replace
        self.save_file = save_name
        self.load_file = load_name
        self.learn_step = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.main_model = build_dqn(alpha, n_actions, input_dims, 512)
        self.target_model = build_dqn(alpha, n_actions, input_dims, 512)

    def replace_target_network(self):
        '''
            Copies the weights from main model to target model.
        '''
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.target_model.set_weights(self.main_model.get_weights())

    def store_transition(self, state, action, reward, new_state, done):
        '''
            Calls the ReplayBuffer store_transition
        '''
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        '''
            Take a random action based on epsilon value, otherwise
            makes a greedy action.
        '''
        if np.random.random() < self.epsilon:
            # Random action
            action = np.random.choice(self.action_space)
        else:
            # Greedy action
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.main_model.predict(state)
            action = np.argmax(actions)

        return action
    
    def choose_action_test(self, observation):
        '''
            Take a random action based on epsilon value, otherwise
            makes a greedy action. For test eps = 0.05
        '''
        if np.random.random() < 0.05:
            # Random action
            action = np.random.choice(self.action_space)
            value = 0
        else:
            # Greedy action
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.main_model.predict(state)
            # print(actions.shape)
            # print(actions)
            action = np.argmax(actions)
            # print(f"Max index: {action}")
            value = actions[0][action]
            # print("Action value: ", value)

        return action, value

    def learn(self):
        '''
            Main function for agent's learning.
        '''
        # Only learn when having stored experiences.
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            self.replace_target_network()

            #action_values = np.array(self.action_space, dtype=np.int8)
            #action_indices = np.dot(action, action_values)

            q_next = self.target_model.predict(new_state)
            q_eval = self.main_model.predict(new_state)
            q_pred = self.main_model.predict(state)

            # Calcular las mejores acciones dado el batch de estados con
            # el modelo main
            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # Evaluar las mejores acciones del modelo main en el modelo target
            q_target[batch_index, action] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*(1- done)

            self.main_model.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

            self.learn_step += 1

    def save_models(self, n):
        save_file = "models/" + self.save_file + n + ".h5"
        self.main_model.save(save_file)
        print('... saving models ...')

    def load_models(self):
        self.main_model = load_model(self.load_file)
        self.target_model.set_weights(self.main_model.get_weights())
        print('... loading models ...')