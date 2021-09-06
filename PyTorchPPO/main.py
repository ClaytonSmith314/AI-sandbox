import gym
import numpy as np
from PyTorchPPO.agent import Agent
from PyTorchPPO.conv_agent import Agent as ConvAgent
#from utils import plot_learning_curve
#import time

def train(env, agent, n_games=300, N=20, avg_score_limit = 170):

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    i = 0
    while i<n_games and avg_score<avg_score_limit:
        observation = env.reset()
        done = False
        score = 0
        n_steps = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'episode {i}:\t\tscore {score},\t\tavg score {round(avg_score)},\t\t'
              f'time_steps {n_steps},\t\tlearning_steps {learn_iters}')
        i += 1

def observe(env, agent, n_games=10):

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        env.render()
        done = False
        score = 0
        n_steps = 0
        end_game_timeout = 1
        while end_game_timeout > 0:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            n_steps += 1
            score += reward
            #agent.remember(observation, action, prob, val, reward, done)
            observation = observation_
            if done:
                end_game_timeout -= 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

        print(f'Observation episode {i}:\t\tscore {score},\t\tavg score {round(avg_score)}')

if __name__ == '__main__':
    env_ = gym.make('Assault-ram-v0') # Envs: CartPole-v0, Acrobot-v1, MountainCar-v0
    agent_ = Agent(input_dims=env_.observation_space.shape,
                  n_actions=env_.action_space.n,
                  batch_size=5,
                  alpha=0.0003,
                  n_epochs=4)

    observe(env_, agent_, n_games=1)
    train(env_, agent_, n_games=500, N=20)
    observe(env_, agent_, n_games=10)

    env_.close()











