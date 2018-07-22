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
        
        model.play(policy=policies.A2CPolicy, 
            env= DummyVecEnv([env.make_train_3]))

if __name__ == '__main__':
    main()










