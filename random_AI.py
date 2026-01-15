# https://github.com/aigagror/Go-AI
import argparse
import gym
import time

# Arguments
parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--boardsize', type=int, default=9)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi, reward_method='real')
go_env.reset()

# Game loop
done = False
while not done:
    #go_env.render(mode = "human")
    action = go_env.render(mode="human")
    #action = go_env.uniform_random_action()
    state, reward, done, info = go_env.step(action)

    if go_env.game_ended():
        break
    action = go_env.uniform_random_action()
    state, reward, done, info = go_env.step(action)
go_env.render(mode="human")
