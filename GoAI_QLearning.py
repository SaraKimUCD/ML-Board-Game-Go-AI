# https://github.com/aigagror/Go-AI
# https://blog.tanka.la/2018/10/19/build-your-first-ai-game-bot-using-openai-gym-keras-tensorflow-in-python/
import argparse
import gym
import time
import numpy as np
import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Input, RepeatVector
from sklearn.model_selection import train_test_split


goal_steps = 5
score_requirement = 1
game_plays = 20

# Arguments
parser = argparse.ArgumentParser(description='Go Environment')
parser.add_argument('--boardsize', type=int, default=4)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi, reward_method='real')
go_env.reset()

print(go_env.action_space)
#print(go_env.observation_space)

def generate_training_data():
    training_data_x = []
    training_data_y = []
    successful_scores = []
    
    # Play through game_plays amount of games
    for game in range(game_plays):
        print("game #:",game)
        score = 0
        game_memory = []
        previous_state = []
        # Try to win game within goal_steps
        #for step in range(goal_steps):
        done = False
        while not done:
            action = go_env.uniform_random_action()
            state, reward, done, info = go_env.step(action)
            #go_env.render("terminal")
            
            #print("action:",action)
            #print("previous_state:")
            #print(state[3])

            if len(previous_state) > 0:
                game_memory.append([previous_state, action])
            previous_state = state[3]
            if go_env.game_ended():
                score = reward
                go_env.render("terminal")
                break

        if score == score_requirement:
            print("BLACK wins")
            output = []
            successful_scores.append(score)
            train_x = []
            train_y = []
            for data in game_memory:
                train_x.append(data[0])
                train_y.append(data[1])
            training_data_x.append(train_x)
            training_data_y.append(train_y)
            # for data in game_memory:
                # training_data.append(data[1])
        go_env.reset()
        #break
    return training_data_x, training_data_y

##First channel = black pieces
##Second channel = white pieces
##Third channel = whose turn (black is 0 white is 1)
##Fourth channel: invalid moves marked as 1
##Fifth channel: check if previous action was pass
##Sixth channel: game over indicator
def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(64, activation="relu"))
    model.add(LSTM(64,input_shape=(input_size,1)))
    #model.add(Dense(64,activation="relu"))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    # model.save_weights('rnn.h5')
    # model.load_weights('rnn.h5')
    return model

training_data_x, training_data_y = generate_training_data()
print(len(training_data_x))

# x = np.array([i for i in training_data])
# y = np.array([i for i in training_data])
# y = y[1:]
# y = np.append(y, x[-1])

#model = build_model(input_size=len(x), output_size = 1)
# model.fit(tf.expand_dims(x, axis=-1), y, epochs=5)

#batch_size = 2
#number_of_batches = int(len(train)/batch_size)
epoch = 1
for (train_x, train_y) in zip(training_data_x, training_data_y):
    print(train_y)
    ##x = np.array([i for i in train_x])
    x = np.array([i for i in train_y])
    y = np.array([i for i in train_y])
    y = y[1:]
    y = np.append(y, y[-1])
    
    model = build_model(input_size=len(x), output_size = 1)
    ## model.fit(
        ## tf.expand_dims(x, axis=-1), y, epochs=epoch
    ## )
    model.fit(
        x, y, epochs=epoch
    )
    
scores = []
choices = []
for each_game in range(10):
    score = 0
    prev_obs = []
    previous_action = 0
    #for step_index in range(goal_steps):
    done = False
    while not done:
        go_env.render("terminal")
        if len(prev_obs)==0:
            print("random")
            action = go_env.uniform_random_action()
            print(action)
            previous_action = action
        else:
            print("else")
            prev_len = len(prev_obs)
            #p = previous_action[None, :]
            previous_action = np.array(previous_action)
            p = tf.expand_dims(previous_action, axis=0)
            prediction = model.predict(p)
            print(prediction)
            previous_action = action

        choices.append(action)
        state, reward, done, info = go_env.step(action)
        prev_obs = state[3]
        score = reward
    go_env.reset()
    scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))

go_env.close()
