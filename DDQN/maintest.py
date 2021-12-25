import time
import gym
import numpy as np
import concurrent.futures
import os
import sys

import pylab
import random
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


# Get ./src/ folder & add it to path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# import your drivers here
from pkg.drivers import DisparityExtender

# choose your drivers here (1-4)
drivers = [DisparityExtender()]

# choose your racetrack here (SILVERSTONE, SILVERSTONE_OBS)
RACETRACK = 'SILVERSTONE'


class DoubleDQNAgent:


    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = inputs
        self.action_size = outputs

        # these is hyper parameters for the Double DQN
        self.discount_factor = discountFactor
        self.learning_rate = learningRate
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = learnStart
        # create replay memory using deque
        self.memory = deque(maxlen=memorySize)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()


        if self.load_model:
            self.model.load_weights("/home/sun/path/Network Weights/DDQN_Main.h5")



    def build_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(12, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1,len(state)))
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


class GymRunner(object):

    def __init__(self, racetrack, drivers):
        self.racetrack = racetrack
        self.drivers = drivers

    def run(self):
        # load map
        env = gym.make('f110_gym:f110-v0',
                       map="{}/maps/{}".format(current_dir, RACETRACK),
                       map_ext=".png", num_agents=len(drivers))

        # specify starting positions of each agent
        poses = np.array([[0. + (i * 0.75), 0. - (i*1.5), np.radians(60)] for i in range(len(drivers))])
        #print(poses) #[[0.         0.         1.04719755]]

        env.render()


        laptime = 0.0
        start = time.time()
        EPISODES = 1500
        xlist = [-5.31687531203] * EPISODES
        ylist = [-4.09487880244] * EPISODES
        x_data = [0] * EPISODES
        y_data = [0] * EPISODES
        env = env()
        # get size of state and action from environment
        # state_size = env.observation_space.shape[0]
        # action_size = env.action_space.n
        network_inputs = 1080
        network_outputs = 120
        memorySize = 1000000
        discountFactor = 0.99
        learningRate = 0.00025
        learnStart = 64
        agent = DoubleDQNAgent(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        scores, episodes = [], []
        for e in range(EPISODES):
            # print e
            done = False
            score = 0
            episode_step = 0
            obs, step_reward, done, info = env.reset(poses=poses)




            #state  = env.reset()
        #print(done)#false
            while not done:

                action = agent.get_action(obs)
                next_state, reward, done, info= env.step(action, episode_step)
                reward = reward if not done or score == 499 else -100


                agent.append_sample(state, action, reward, next_state, done)
                # every time step do the training
                agent.train_model()
                score += reward
                state = next_state
                if done:
                    # every episode update the target model to be same with model
                    agent.update_target_model()

                    # every episode, plot the play time
                    score = score if score == 500 else score + 100
                    scores.append(score)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("/home/sun/path/Network Weights/DDQN_Main.png")
                    print("episode:", e, "  score:", score, "  memory length:",
                          len(agent.memory), "  epsilon:", agent.epsilon)

                    # if the mean of scores of last 10 episode is bigger than 490
                    # stop training
                    if e > 1500:
                        sys.exit()

            # save the model
            if e % 100 == 0:
                agent.model.save_weights("/home/sun/path/Network Weights/DDQN_Main.h5")


            '''
            actions = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, driver in enumerate(drivers):
                    #print(drivers) [<pkg.drivers.DisparityExtender object at 0x7fc8a7ad5cc0>]
                    # scans =1080. (obs['scans'][i])
                   # print(enumerate(drivers))

                    futures.append(executor.submit(driver.process_lidar, obs['scans'][i]))
            for future in futures:
                speed, steer = future.result()
                actions.append([steer, speed])
            actions = np.array(actions)
            #print(actions)
            obs, step_reward, done, info = env.step(actions)
            '''
            laptime += step_reward
            env.render(mode='human')

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    runner = GymRunner(RACETRACK, drivers)
    runner.run()
    DoubleDQNAgent=DoubleDQNAgent()
