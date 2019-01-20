from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from tensorboardX import SummaryWriter

import numpy as np
import pickle


def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_render = True
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    use_cuda = False
    use_gae = default_config.getboolean('UseGAE')
    #use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    sticky_action = False
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae
    )

    print('Loading Pre-trained model....')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
        agent.rnd.target.load_state_dict(torch.load(target_path))
    else:
        agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    print('End load...')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    steps = 0
    rall = 0
    rd = False
    intrinsic_reward_list = []
    while not rd:
        steps += 1
        actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            rall += r
            next_states = s.reshape([1, 4, 84, 84])
            next_obs = s[3, :, :].reshape([1, 1, 84, 84])

        # total reward = int reward + ext Reward
        intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
        intrinsic_reward_list.append(intrinsic_reward)
        states = next_states[:, :, :, :]

        if rd:
            intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
                intrinsic_reward_list)
            with open('int_reward', 'wb') as f:
                pickle.dump(intrinsic_reward_list, f)
            steps = 0
            rall = 0


if __name__ == '__main__':
    main()
