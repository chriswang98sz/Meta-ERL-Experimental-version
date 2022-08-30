from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np

import torch

import torch.multiprocessing as mp
from torch.autograd import Variable

from es_network import ESContinuous, ESDiscrete
import time


def gradient_update(args, synced_model, returns, random_seeds, neg_list,
                    num_eps, num_frames, unperturbed_results, env):
    def fitness_shaping():
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = 0
        for r in returns:
            num = max(0, math.log(lamb + 1, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2))
            denom += num
            shaped_returns.append(num)
        shaped_returns = np.array(shaped_returns)
        shaped_returns = list(shaped_returns / denom - 1 / lamb )
        return shaped_returns

    batch_size = len(returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping()
    print('Episode num: %d\n'
          'Average reward: %f\n'
          'Standard Deviation: %f\n'
          'Max reward: %f\n'
          'Sigma: %f\n'
          'Meta Sigma: %f\n'
          'Total num frames seen: %d\n'
          'Unperturbed reward: %f' %
          (num_eps, np.mean(returns), np.std(returns), max(returns),
           args.sigma, args.metasigma, num_frames,
           unperturbed_results))
    for i in range(args.n):
        np.random.seed(random_seeds[i])
        multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]

        for k, v in synced_model.es_params():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(args.lr / (args.n * args.sigma) *
                                  (reward * multiplier * eps)).float()
    return synced_model



def do_rollouts(args, model, random_seeds, return_queue, env, is_negative):
    state = env.reset()
    state = torch.from_numpy(state)
    this_model_return = 0
    this_model_num_frames = 0
    for step in range(args.max_episode_length):
        state = state.float()
        state = state.view(1, env.observation_space.shape[0])
        with torch.no_grad():
            state = Variable(state)
        prob, entropy = model(state)
        action = prob.max(1)[1].data.numpy()
        next_state, reward, done, _ = env.step(action[0])
        state = next_state
        this_model_return += reward
        this_model_num_frames += 1
        if done:
            break
        state = torch.from_numpy(state)
    return_queue.put((random_seeds, this_model_return, this_model_num_frames, is_negative))



def perturb_model(sigma, model, random_seed, env):
    new_model = ESDiscrete(env)
    new_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v)in new_model.es_params():
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(sigma * eps).float()
    return [new_model]

def train_loop_es_discrete(args, synced_model, env):
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return notflat_results

    if args.save == 'True':
        synced_model.save(args.save_path)
    if args.load == 'True':
        synced_model.load(args.save_path + "/" + args.load_name)
    print("============================================================================================")
    print("Training Discrete Env...")
    print("Initial Sigma:{},".format(args.sigma))
    print("Learning Rate of Network:{},".format(args.lr))
    print("Batch Size of Network:{},".format(args.n))
    print("Total Interations:{},".format(args.T))
    print("============================================================================================")
    np.random.seed()
    num_eps = 0
    total_num_frames = 0
    start_time = time.time()
    for gradient_updates in range(args.T):
        processes = []
        return_queue = mp.Queue()
        all_seeds, all_models = [], []
        for i in range(args.n):
            random_seed = np.random.randint(2 ** 30)
            new_model = perturb_model(args.sigma, synced_model, random_seed, env)
            all_seeds.append(random_seed)
            all_models += new_model
        assert len(all_seeds) == len(all_models)
        is_negative = False
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            p = mp.Process(target=do_rollouts, args=(args, perturbed_model, seed, return_queue, env, is_negative))
            p.start()
            processes.append(p)
        assert len(all_seeds) == 0
        p = mp.Process(target=do_rollouts, args=(args, synced_model, 'dummy_seed', return_queue, env, 'dummy_neg'))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, results, num_frames, neg_list = [flatten(raw_results, index)
                                                                      for index in [0, 1, 2, 3]]
        _ = unperturbed_index = seeds.index('dummy_seed')
        seeds.pop(unperturbed_index)
        unperturbed_results = results.pop(unperturbed_index)
        _ = num_frames.pop(unperturbed_index)
        _ = neg_list.pop(unperturbed_index)
        total_num_frames += sum(num_frames)
        num_eps += len(results)
        synced_model = gradient_update(args, synced_model, results,seeds,
                                       neg_list, num_eps, total_num_frames,
                                       unperturbed_results, env)
        print('Time: %.1f\n' % (time.time() - start_time))
