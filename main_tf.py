from DDPG import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
rng = np.random.default_rng(3)
EPISODES = 1000

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')

    agent = Agent(alpha=0.0001, beta=0.001, input_dims=[3], tau=0.001, env=env, batch_size=64,
                  layer1_size=400, layer2_size=300, n_actions=1)

    score_history = []
    for i in range(EPISODES):
        obs = env.reset()[0]
        done = False
        truncated = False
        score = 0

        while not done and not truncated:
            action = agent.choose_action(obs)
            new_state, reward, done, truncated, info = env.step(action)
            agent.remember(obs, action, reward, new_state, done)
            agent.learn()
            score += reward
            obs = new_state
        
        score_history.append(score)
            
        print('episode : {} score : {:.2f} average score {:.2f}'.format(i, score, np.mean(score_history[-100:])))

    filename = 'pendulum.png'
    figure_file = 'plots/' + filename

    plt.plot(list(range(EPISODES)), score_history)
    plt.xticks(list(range(0, 1000, 100)))
    plt.savefig(figure_file)
