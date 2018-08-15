import numpy as np
import tensorflow as tf

# This function selects the probability distribution over actions
from baselines.common.distributions import make_pdtype

# Convolution layer
def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(strides, strides),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.orthogonal_initializer(gain=gain))


# Fully connected layer
def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_fn,
                           kernel_initializer=tf.orthogonal_initializer(gain))


"""
This object creates the PPO Network architecture
"""
class PPOPolicy(object):
    def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse = False):
        # This will use to initialize our kernels
        gain = np.sqrt(2)

        # Based on the action space, will select what probability distribution type
        # we will use to distribute action in our stochastic policy (in our case DiagGaussianPdType
        # aka Diagonal Gaussian, 3D normal distribution
        self.pdtype = make_pdtype(action_space)

        height, weight, channel = ob_space.shape
        ob_shape = (height, weight, channel)

        # Create the input placeholder
        inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")

        # Normalize the images
        scaled_images = tf.cast(inputs_, tf.float32) / 255.

        """
        Build the model
        3 CNN for spatial dependencies
        Temporal dependencies is handle by stacking frames
        (Something funny nobody use LSTM in OpenAI Retro contest)
        1 common FC
        1 FC for policy
        1 FC for value
        """
        with tf.variable_scope("model", reuse = reuse):
            conv1 = conv_layer(scaled_images, 32, 8, 4, gain)
            conv2 = conv_layer(conv1, 64, 4, 2, gain)
            conv3 = conv_layer(conv2, 64, 3, 1, gain)
            flatten1 = tf.layers.flatten(conv3)
            fc_common = fc_layer(flatten1, 512, gain=gain)

            # This build a fc connected layer that returns a probability distribution
            # over actions (self.pd) and our pi logits (self.pi).
            self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)

            # Calculate the v(s)
            vf = fc_layer(fc_common, 1, activation_fn=None)[:, 0]

        self.initial_state = None

        # Take an action in the action distribution (remember we are in a situation
        # of stochastic policy so we don't always take the action with the highest probability
        # for instance if we have 2 actions 0.7 and 0.3 we have 30% chance to take the second)
        a0 = self.pd.sample()

        # Calculate the neg log of our probability
        neglogp0 = self.pd.neglogp(a0)

        # Function use to take a step returns action to take and V(s)
        def step(state_in, *_args, **_kwargs):

            # return a0, vf, neglogp0
            return sess.run([a0, vf, neglogp0], {inputs_: state_in})

        # Function that calculates only the V(s)
        def value(state_in, *_args, **_kwargs):
            return sess.run(vf, {inputs_: state_in})

        # Function that output only the action to take
        def select_action(state_in, *_args, **_kwargs):
            return sess.run(a0, {inputs_: state_in})

        self.inputs_ = inputs_
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action



