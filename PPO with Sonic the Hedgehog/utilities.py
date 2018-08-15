import tensorflow as tf
# Get the variables
def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


# Make directory
def make_path(f):
    # exist_ok: if the folder already exist makes no exception error
    return os.makedirs(f, exist_ok=True)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]
