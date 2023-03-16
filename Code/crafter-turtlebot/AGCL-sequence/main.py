import numpy as np
from enviroments import get_envs
import os
from agents import ActorCriticPolicy
from args import Args
import random
import time
from collections import namedtuple

from DQN import DQN


def check_training_done_callback(reward_array, done_array, is_final_env):
    done_cond = 0
    reward_cond = 0

    if is_final_env == 1:
        return 0
        # if len(done_array) > 30:
        #     if np.mean(done_array[-10:]) > 0.80 and np.mean(done_array[-40:]) > 0.80:
        #         if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
        #             done_cond = 1

        #     if done_cond:
        #         if np.mean(reward_array[-40:]) > 800:
        #             reward_cond = 1

        #     if done_cond and reward_cond:
        #         return 1
        #     else:
        #         return 0
        # else:
        #     return 0
    else:
        if len(done_array) > 30:
            if np.mean(done_array[-10:]) > 0.5 and np.mean(done_array[-40:]) > 0.5:
                if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
                    done_cond = 1

            if done_cond:
                if np.mean(reward_array[-40:]) > -100:
                    reward_cond = 1

            if done_cond and reward_cond:
                return 1
            else:
                return 0
        else:
            return 0



def train(args, env, agent, index_env, is_final_env):  # fill in more args if it's needed

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])    
    env.reset()
    episode = 0
    time_step = 0
    reward_sum = 0
    done_arr = []
    curr_task_completion_array = []
    reward_arr = []
    avg_reward = []
    timestep_arr = []
    episode_arr = []
    global_timestep = 0
    global_timestep_to_return = 0
    check = 0
    while True:
        # env.render()
        # time.sleep(.1)
        obs = env.get_observation()
        # print("obs: ", obs)
        a = agent.select_action(obs)

        # print("action", a)

        new_obs, reward, done, info = env.step(a)

        transition = Transition(obs, a, reward, new_obs)
        agent.store_transition(transition)

        agent.set_rewards(reward)
        reward_sum += reward

        time_step += 1
        global_timestep += 1

        if time_step > args.time_limit or done:

            # finish agent
            if done:
                done_arr.append(1)
                curr_task_completion_array.append(1)
            elif time_step > args.time_limit:
                done_arr.append(0)
                curr_task_completion_array.append(0)

            reward_arr.append(reward_sum)
            avg_reward.append(np.mean(reward_arr[-40:]))
            timestep_arr.append(time_step)

            done = 1
            # agent.finish_episode()

            print("\n\nfinished episode = " + str(episode) + " with " + str(reward_sum) + "\n")

            episode += 1
            time_step = 0

            env.reset()
            reward_sum = 0

            env_flag = 0

            env_flag = check_training_done_callback(reward_arr, done_arr,is_final_env)

            if env_flag == 0:
                agent.update()

            if env_flag == 1 and is_final_env == 1:
                check += 1
                env_flag = 0
                if check == 1:
                    global_timestep_to_return = global_timestep

            # quit after some number of episodes
            if episode > 15000 or env_flag == 1:
                agent.save_model(0, 0, index_env)
                episode_arr.append(episode)
                if is_final_env ==0:
                    global_timestep_to_return = global_timestep
                break

    return reward_arr, avg_reward, timestep_arr, episode_arr, index_env, global_timestep_to_return


def main(args):
    random.seed(args.seed)
    envs = get_envs()
    results = {'reward':[], 'avg_reward':[],'timesteps':[],'episodes_per_task':[]}
    # agent = ActorCriticPolicy(args.num_actions,
    #                           args.input_size,
    #                           args.hidden_size,
    #                           args.learning_rate,
    #                           args.gamma,
    #                           args.decay_rate,
    #                           args.epsilon)

    agent = DQN(args.input_size,
                args.num_actions
                )

    is_final_env = 0
    source_task_timesteps = 0
    for index_env, env in enumerate(envs):
        agent.reset()
        if index_env > 0:
            agent.load_model(0, 0, index_env-1)
            # agent.reinit()

        if index_env == len(envs) - 1:
            is_final_env = 1

        result = train(args, env, agent, index_env, is_final_env)
        results['reward'].extend(result[0])
        results['avg_reward'].extend(result[1])
        results['timesteps'].extend(result[2])
        results['episodes_per_task'].extend(result[3])
        print('results:', results)
        if is_final_env == 0:
            source_task_timesteps = result[5]

    log_dir = 'logs_' + str(args.seed) 
    os.makedirs(log_dir, exist_ok=True)
    path_to_save_total_reward = log_dir + os.sep + "randomseed_" + str(args.seed) + "_reward_.npz"
    np.savez_compressed(path_to_save_total_reward, curriculum_reward = np.asarray(results["reward"]))

    path_to_save_avg_reward = log_dir + os.sep + "randomseed_" + str(args.seed) + "_avg_reward_.npz"
    np.savez_compressed(path_to_save_avg_reward, curriculum_avg_reward = np.asarray(results["avg_reward"]))

    path_to_save_timesteps = log_dir + os.sep + "randomseed_" + str(args.seed) + "_timesteps_.npz"
    np.savez_compressed(path_to_save_timesteps, curriculum_reward = np.asarray(results['timesteps']))

    path_to_save_episodes = log_dir + os.sep + "randomseed_" + str(args.seed) + "_episodes_.npz"
    np.savez_compressed(path_to_save_episodes, curriculum_reward = np.asarray(results["episodes_per_task"]))
    print("global timestep: ", result[5] )
    print("source timestep: ", source_task_timesteps)


if __name__ == '__main__':
    opt = Args()
    opt = opt.update_args()
    main(opt)
