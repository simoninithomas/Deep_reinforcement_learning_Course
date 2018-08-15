import tensorflow as tf
import numpy as np
import gym
import math
import os

import model
import architecture as policies
import sonic_env as env

# SubprocVecEnv creates a vector of n environments to run them simultaneously.
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def main():
    config = tf.ConfigProto()

    # Avoid warning message errors
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Allowing GPU memory growth
    config.gpu_options.allow_growth = True


    with tf.Session(config=config):

    	# 0: {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act1'},
        # 1: {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        # 2: {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act2'},
        # 3: {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act3'},
        # 4: {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act1'},
        # 5: {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        # 6: {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MetropolisZone.Act3'},
        # 7: {'game': 'SonicTheHedgehog2-Genesis', 'state': 'HillTopZone.Act2'},
        # 8: {'game': 'SonicTheHedgehog2-Genesis', 'state': 'CasinoNightZone.Act2'},
        # 9: {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LavaReefZone.Act1'},
        # 10: {'game': 'SonicAndKnuckles3-Genesis', 'state': 'FlyingBatteryZone.Act2'},
        # 11: {'game': 'SonicAndKnuckles3-Genesis', 'state': 'HydrocityZone.Act1'},
        # 12: {'game': 'SonicAndKnuckles3-Genesis', 'state': 'AngelIslandZone.Act2'}

        model.play(policy=policies.PPOPolicy, 
            env= DummyVecEnv([env.make_train_1]),
            update = 120)

if __name__ == '__main__':
    main()










