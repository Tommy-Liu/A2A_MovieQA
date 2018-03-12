import tensorflow as tf


def extract_axis_1(data, ind):
    batch_range = tf.cast(tf.range(tf.shape(data)[0]), dtype=tf.int64)
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def get_initializer(name, m=0.0, s=1.0):
    if name == 'truncated':
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


def get_lr(name, lr, global_step, decay_steps, decay_rate=0.5, staircase=False):
    if name == 'cos':
        learning_rate = tf.train.cosine_decay(lr, global_step, decay_steps)
    elif name == 'exp':
        learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase)
    elif name == 'inv':
        learning_rate = tf.train.inverse_time_decay(lr, global_step, decay_steps, decay_rate, staircase)
    elif name == 'inv_sqrt':
        learning_rate = tf.train.inverse_time_decay(lr, tf.sqrt(tf.cast(global_step, tf.float32)),
                                                    1.0, decay_rate, staircase)
    elif name == 'linear_cos':
        learning_rate = tf.train.linear_cosine_decay(lr, global_step, decay_rate)
    elif name == 'natural_exp':
        learning_rate = tf.train.natural_exp_decay(lr, global_step, decay_steps, decay_rate, staircase)
    elif name == 'noisy_linear_cos':
        learning_rate = tf.train.noisy_linear_cosine_decay(lr, global_step, decay_steps)
    elif name == 'poly':
        learning_rate = tf.train.polynomial_decay(lr, global_step, decay_steps)
    else:
        learning_rate = lr
    return learning_rate


def get_opt(name, learning_rate):
    if name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.98, epsilon=1e-9)
    elif name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif name == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif name == 'adag':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
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
