import tensorflow as tf


def get_initializer(name, m=0.0, s=1.0):
    if name == 'identity':
        initializer = tf.identity_initializer(s)
    elif name == 'truncated':
        initializer = tf.truncated_normal_initializer(m, s)
    elif name == 'uniform':
        initializer = tf.random_uniform_initializer(-s, s)
    elif name == 'normal':
        initializer = tf.random_normal_initializer(m, s)
    elif name == 'orthogonal':
        initializer = tf.orthogonal_initializer(s)
    elif name == 'glorot' or name == 'xavier':
        initializer = tf.glorot_uniform_initializer()
    elif name == 'variance':
        initializer = tf.variance_scaling_initializer(s)
    else:
        initializer = None
    return initializer


def get_opt(name, learning_rate):
    if name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif name == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
        optimizer = None

    return optimizer


def get_loss(name, data, model):
    if name == 'mse':
        loss = tf.losses.mean_squared_error(data.vec, model.output)
    elif name == 'abs':
        loss = tf.losses.absolute_difference(data.vec, model.output)
    elif name == 'l2':
        loss = tf.losses.compute_weighted_loss(tf.norm(data.vec - model.output, axis=1))
    elif name == 'cos':
        loss = tf.losses.cosine_distance(tf.nn.l2_normalize(data.vec, 1),
                                         tf.nn.l2_normalize(model.output, 1), 1)
    elif name == 'huber':
        loss = tf.losses.huber_loss(data.vec, model.output)
    elif name == 'mpse':
        loss = tf.losses.mean_pairwise_squared_error(data.vec, model.output)
    else:
        loss = 0
    return loss
