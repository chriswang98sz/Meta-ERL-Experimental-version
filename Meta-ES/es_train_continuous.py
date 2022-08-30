from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np

import torch

import torch.multiprocessing as mp
from torch.autograd import Variable

from es_network import ESContinuous, ESDiscrete
import time


def gradient_update(args, synced_model, returns, returns_with_entropy, random_seeds, neg_list,
                    num_eps, num_frames, unperturbed_results, env):
    def fitness_shaping():
        sorted_returns_backwards = sorted(returns_with_entropy)[::-1]
        lamb = len(returns_with_entropy)
        mu = lamb // 2
        shaped_returns = []
        denom = 0
        flag = 0
        for r in returns_with_entropy:
            if sorted_returns_backwards.index(r) >= mu or flag >= mu:
                shaped_returns.append(0)
            else:
                num = math.log(mu + 0.5) - math.log(sorted_returns_backwards.index(r) + 1)
                flag += 1
                denom += num
                shaped_returns.append(num)
        shaped_returns = np.array(shaped_returns)
        shaped_returns = list(shaped_returns / denom)
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
          'Total num frames seen: %d\n'
          'Unperturbed reward: %f' %
          (num_eps, np.mean(returns), np.std(returns), max(returns),
           args.sigma, num_frames,
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

def gradient_update_sigma(args, returns, random_seeds, neg_list):
    def fitness_shaping():
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = 0
        for r in returns:
            num = math.log(lamb + 0.5) - math.log(sorted_returns_backwards.index(r) + 1)
            denom += num
            shaped_returns.append(num)
        shaped_returns = np.array(shaped_returns)
        shaped_returns = list(shaped_returns / denom)
        return shaped_returns
    batch_size = len(returns)
    assert batch_size == args.m
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping()
    sigma = args.sigma
    for i in range(args.m):
        np.random.seed(random_seeds[i])
        multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]
        eps = np.random.normal(0, 1)
        sigma += args.lr_meta/(args.m* args.metasigma)*(reward*multiplier*eps)
    return sigma

def do_rollouts(args, model, random_seeds, return_queue, env, is_negative):
    state = env.reset()
    state = torch.from_numpy(state)
    this_model_return = 0
    this_model_return_with_entropy = 0
    this_model_num_frames = 0
    for step in range(args.max_episode_length):
        state = state.float()
        dist = model.forward(state)
        action = dist.sample()
        if type(action)==torch.Tensor:
            action=action.data.numpy()[0]
        entropy = sum(dist.entropy().data.numpy()[0])*args.sigma
        next_state, reward, done, _ = env.step(action)
        if type(reward)==torch.Tensor:
            reward=reward.data.numpy()[0]
        state = next_state
        this_model_return += reward
        this_model_return_with_entropy += reward
        this_model_return_with_entropy += entropy
        this_model_num_frames += 1
        if done:
            break
        state = torch.from_numpy(state)
    return_queue.put((random_seeds, this_model_return, this_model_return_with_entropy,
                      this_model_num_frames, is_negative))
def do_rollouts_unperturbed(args, model,env):
    state = env.reset()
    state = torch.from_numpy(state)
    this_model_return = 0
    for step in range(args.max_episode_length):
        if args.render=='True':
            try:
                env.render()
            except:
                pass
        state = state.float()
        dist = model.forward(state)
        action = dist.sample()
        if type(action) == torch.Tensor:
            action = action.data.numpy()[0]
        next_state, reward, done, _ = env.step(action)
        if type(reward) == torch.Tensor:
            reward = reward.data.numpy()[0]
        state = next_state
        this_model_return += reward
        if done:
            break
        state = torch.from_numpy(state)
    return this_model_return

def perturb_model(sigma, model, random_seed, env):
    positive_model = ESContinuous(env)
    negative_model = ESContinuous(env)
    positive_model.load_state_dict(model.state_dict())
    negative_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (positive_k, positive_v), (negative_k, negative_v) in zip(positive_model.es_params(),
                                                                  negative_model.es_params()):
        eps = np.random.normal(0, 1, positive_v.size())
        positive_v += torch.from_numpy(sigma * eps).float()
        negative_v += torch.from_numpy(sigma * -eps).float()
    return [positive_model, negative_model]

def perturb_model_single(sigma, model, random_seed, env):
    new_model = ESDiscrete(env)
    new_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v)in new_model.es_params():
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(sigma * eps).float()
    return [new_model]

def train_loop_es_continuous(args,synced_model,env):
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return notflat_results
    if args.save=='True':
        synced_model.save(args.save_path)
    if args.load=='True':
        synced_model.load(args.save_path + "/" + args.load_name)
    print("============================================================================================")
    print("Training Continuous Env...")
    if args.use_meta:
        print("Initial Sigma:{},\nMeta Sigma:{},\nTemperature Factor:{},".format(args.sigma,
                                                                                 args.metasigma,
                                                                                 args.alpha))
        print("Learning Rate of Network:{},\nLearning Rate of Sigma:{},".format(args.lr,
                                                                                                  args.lr_meta))
        print("Batch Size of Network:{},\nBatch Size of Sigma:{},".format(args.n, args.m))
        print("Total Interations:{},\nUpdate Frequency of Sigma:{},".format(args.T, args.t))
    else:
        print("Initial Sigma:{},\nTemperature Factor:{},".format(args.sigma, args.alpha))
        print("Learning Rate of Network:{},\n,".format(args.lr))
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
        for i in range(int(args.n / 2)):
            random_seed = np.random.randint(2 ** 30)
            two_models = perturb_model(args.sigma, synced_model, random_seed, env)
            all_seeds.append(random_seed)
            all_seeds.append(random_seed)
            all_models += two_models
        assert len(all_seeds) == len(all_models)
        is_negative = True
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            p = mp.Process(target=do_rollouts, args=(args, perturbed_model, seed, return_queue, env, is_negative))
            p.start()
            processes.append(p)
            is_negative = not is_negative
        assert len(all_seeds) == 0

        raw_results = [return_queue.get() for p in processes]
        seeds, results, results_with_entropy, num_frames, neg_list = [flatten(raw_results, index)
                                                                      for index in [0, 1, 2, 3, 4]]
        unperturbed_results = do_rollouts_unperturbed(args, synced_model, env)
        total_num_frames += sum(num_frames)
        num_eps += len(results)
        synced_model = gradient_update(args, synced_model, results, results_with_entropy, seeds,
                                       neg_list, num_eps, total_num_frames,
                                       unperturbed_results, env)
        if args.use_meta == 'True' and gradient_updates % args.t == 0:
            processes = []
            return_queue = mp.Queue()
            all_seeds, all_models = [], []
            for j in range(int(args.m / 2)):
                random_seed_sigma = np.random.randint(2 ** 30)
                random_seed_positive_net = np.random.randint(2 ** 30)
                random_seed_negative_net = np.random.randint(2 ** 30)
                np.random.seed(random_seed_sigma)
                eps_meta = np.random.normal(0, 1)
                new_sigma_positive = eps_meta * args.metasigma + args.sigma
                positive_model = perturb_model_single(new_sigma_positive, synced_model, random_seed_positive_net, env)
                all_seeds.append(random_seed_sigma)
                all_models += positive_model
                new_sigma_negative = -eps_meta * args.metasigma + args.sigma
                negative_model = perturb_model_single(new_sigma_negative, synced_model, random_seed_negative_net, env)
                all_seeds.append(random_seed_sigma)
                all_models += negative_model
            assert len(all_seeds) == len(all_models)
            is_negative = True
            flag = 0
            while all_models:
                perturbed_model = all_models.pop()
                seed = all_seeds.pop()
                p = mp.Process(target=do_rollouts, args=(args, perturbed_model, seed, return_queue, env, is_negative))
                p.start()
                processes.append(p)
                flag += 1
                if flag == 2:
                    flag = 0
                    is_negative = not is_negative
            assert len(all_seeds) == 0
            for p in processes:
                p.join()
            raw_results = [return_queue.get() for p in processes]
            seeds, results, results_with_entropy,num_frames, neg_list = [flatten(raw_results, index) for index in [0, 1, 2, 3,4]]
            args.sigma = gradient_update_sigma(args, results, seeds, neg_list)
        print('Time: %.1f\n' % (time.time() - start_time))