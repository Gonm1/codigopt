import numpy as np
import pandas
from ddqn import Agent
from utils import make_env
from tqdm import tqdm
from timeit import default_timer as timer

"""
Code by https://github.com/philtabor
at https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/main_keras_dqn_pong.py
Modified by Gonzalo Miranda.
"""

if __name__ == '__main__':

    #env = make_env('PongNoFrameskip-v4')
    env = make_env('BreakoutNoFrameskip-v4')
    #env = make_env('SpaceInvadersNoFrameskip-v4')

    test_rewards, test_qvalue, test_times = [], [], []
    scores, eps_history = [], []
    num_games = 10_000
    number_of_tests = 30 # Numero de pruebas a realizar
    n_steps, n_test = 0, 1 # Contador de steps y pruebas individuales
    n_test_instance = 1 # Contador de instancias de prueba
    test_every_frames = 520_000 # Realizar pruebas cada n frames
    load_checkpoint = False # Cargar modelo (?)
    render = False
    

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025,
                  input_dims=env.observation_space.shape, n_actions=env.action_space.n, mem_size=200_000,
                  eps_min=0.1, batch_size=32, replace=10_000, eps_dec=1e-5, 
                  save_name='dqn_model', load_name='dqn_model_5000it.h5')

    if load_checkpoint:
        agent.epsilon = 0.1
        agent.load_models()

    last_ep = 0
    for episode in tqdm(range(num_games)):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            if not render:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                agent.learn()
            else:
                env.render()
            observation = observation_

            # time for tests
            if n_steps % test_every_frames == 0:
                break

        if not render:
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode: ', episode,'score: ', score,
                    ' average score %.3f' % avg_score,
                    'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

            eps_history.append(agent.epsilon)

        # test time
        if n_steps % test_every_frames == 0 or episode == num_games - 1:

            # Save the model at test time
            agent.save_models(f"{episode + 1}")
            
            for test in range(number_of_tests):
                t_reward, t_value = 0, 0
                timeout = 300 + timer()
                done = False
                observation = env.reset()
                while not done and timer() < timeout:
                    action, value = agent.choose_action_test(observation)
                    observation_, reward, done, info = env.step(action)
                    t_reward += reward
                    t_value += value
                    
                test_times.append(n_test)
                test_rewards.append(t_reward)
                test_qvalue.append(t_value)
                n_test +=1

            reward_avg = np.mean(test_rewards)
            value_avg = np.mean(test_qvalue)

            # Save data

            df = pandas.DataFrame(data={"Test_episode": test_times, "rewards": test_rewards, "values": test_qvalue})
            df.to_csv(f"./csv/ddqnTestData_PerEpisode{n_test - 1}.csv", sep=',',index=False)

            df = pandas.DataFrame(data={"Test_instance": [n_test_instance], "Score_avg": [reward_avg], "Value_avg": [value_avg], "N_steps": [n_steps]})
            df.to_csv(f"./csv/ddqnTestData_{n_test_instance}.csv", sep=',',index=False)
            n_test_instance += 1

            x = [i+1 for i in range(last_ep, episode + 1)]
            last_ep = episode + 1

            df = pandas.DataFrame(data={"Episode": x, "Score": scores, "Epsilon": eps_history})
            df.to_csv(f"./csv/ddqnTrainingData-{episode+1}.csv", sep=',',index=False)
            
            # Clear lists
            scores.clear()
            eps_history.clear()
            test_times.clear()
            test_rewards.clear()
            test_qvalue.clear()
            x.clear()
