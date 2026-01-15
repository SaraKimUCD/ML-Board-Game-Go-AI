# https://github.com/aigagror/Go-AI
# https://blog.tanka.la/2018/10/19/build-your-first-ai-game-bot-using-openai-gym-keras-tensorflow-in-python/
import argparse
import gym
import time
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense

goal_steps = 100
score_requirement = 10
game_plays = 1000

# Arguments
parser = argparse.ArgumentParser(description='Go Environment')
parser.add_argument('--boardsize', type=int, default=5)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi, reward_method='real')
go_env.reset()

print(go_env.action_space)
#print(go_env.observation_space)

# for i in range(50):
#     action = go_env.uniform_random_action()
#     state, reward, done, info = go_env.step(action)
#     go_env.render("terminal")
#     print(reward)
#     time.sleep(1)
# go_env.close()

def generate_training_data():
    training_data = []
    successful_scores = []
    
    for game in range(game_plays):
        score = 0
        game_memory = []
        previous_state = []
        for step in range(goal_steps):
            done = False
            while not done:
                action = go_env.uniform_random_action()
                state, reward, done, info = go_env.step(action)
                #go_env.render("terminal")
    
                if len(previous_state) > 0:
                    game_memory.append([previous_state, action])
                previous_state = state
                score += reward
                if go_env.game_ended():
                    go_env.render("terminal")
                    break
                
            if score >= score_requirement:
                output = []
                successful_scores.append(score)
                for data in game_memory:
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]
                    training_data.append([data[0],output])
            go_env.reset()
        print("successful_scores:")
        print(successful_scores)
        return training_data

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(64, input_dim =input_size, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

training_data = generate_training_data()
X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
X = np.asarray(X).astype(np.float32)
y = np.array([i[1] for i in training_data])
y = np.asarray(X).astype(np.float32)
model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    
model.fit(X, y, epochs=1)


scores = []
choices = []
for each_game in range(10):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        done = False
        while not done:
            go_env.render("terminal")
            if len(prev_obs)==0:
                print("random")
                action = go_env.uniform_random_action()
                print(action)
            else:
                print("else")
                prev_len = len(prev_obs)
                prediction = model.predict(prev_obs.reshape(-1, prev_len))[0]
                print(prediction)
                action = np.argmax(prediction)
                print(action)

            choices.append(action)
            print("next step")
            state, reward, done, info = go_env.step(action)
            prev_obs = state
            score+=reward
    go_env.reset()
    scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))


go_env.close()
