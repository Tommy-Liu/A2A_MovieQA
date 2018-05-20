import tensorflow as tf
from tensorflow.contrib.opt import LazyAdamOptimizer, AddSignOptimizer, PowerSignOptimizer
from tensorflow.contrib.opt.python.training.sign_decay \
    import get_cosine_decay_fn, get_linear_decay_fn, get_restart_decay_fn


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
        learning_rate = tf.train.linear_cosine_decay(lr, global_step, decay_steps, beta=0.01)
    elif name == 'natural_exp':
        learning_rate = tf.train.natural_exp_decay(lr, global_step, decay_steps, decay_rate, staircase)
    elif name == 'noisy_linear_cos':
        learning_rate = tf.train.noisy_linear_cosine_decay(lr, global_step, decay_steps)
    elif name == 'poly':
        learning_rate = tf.train.polynomial_decay(lr, global_step, decay_steps)
    else:
        learning_rate = lr
    return learning_rate


def get_opt(name, learning_rate, decay_steps=None):
    if name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.98, epsilon=1e-9)
    elif name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif name == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif name == 'lazyadam':
        optimizer = LazyAdamOptimizer(learning_rate)
    elif name == 'powersign':
        optimizer = PowerSignOptimizer(learning_rate)
    elif name == 'powersign-ld':
        optimizer = PowerSignOptimizer(learning_rate, sign_decay_fn=get_linear_decay_fn(decay_steps))
    elif name == 'powersign-cd':
        optimizer = PowerSignOptimizer(learning_rate, sign_decay_fn=get_cosine_decay_fn(decay_steps))
    elif name == 'powersign-rd':
        optimizer = PowerSignOptimizer(learning_rate, sign_decay_fn=get_restart_decay_fn(decay_steps))
    elif name == 'addsign':
        optimizer = AddSignOptimizer(learning_rate)
    elif name == 'addsign-ld':
        optimizer = AddSignOptimizer(learning_rate, sign_decay_fn=get_linear_decay_fn(decay_steps))
    elif name == 'addsign-cd':
        optimizer = AddSignOptimizer(learning_rate, sign_decay_fn=get_cosine_decay_fn(decay_steps))
    elif name == 'addsign-rd':
        optimizer = AddSignOptimizer(learning_rate, sign_decay_fn=get_restart_decay_fn(decay_steps))
    else:
        optimizer = None

    return optimizer


def get_loss(name, labels, outputs):
    if name == 'mse':
        loss = tf.losses.mean_squared_error(labels, outputs)
    elif name == 'abs':
        loss = tf.losses.absolute_difference(labels, outputs)
    elif name == 'l2':
        loss = tf.losses.compute_weighted_loss(tf.norm(labels - outputs, axis=1))
    elif name == 'cos':
        loss = tf.losses.cosine_distance(tf.nn.l2_normalize(labels, 1),
                                         tf.nn.l2_normalize(outputs, 1), 1)
    elif name == 'hinge':
        loss = tf.losses.hinge_loss(labels, outputs)
    elif name == 'huber':
        loss = tf.losses.huber_loss(labels, outputs)
    elif name == 'mpse':
        loss = tf.losses.mean_pairwise_squared_error(labels, outputs)
    elif name == 'sparse_softmax':
        loss = tf.losses.sparse_softmax_cross_entropy(labels, outputs)
    elif name == 'softmax':
        labels = tf.one_hot(labels, 5, axis=-1)
        loss = tf.losses.softmax_cross_entropy(labels, outputs)
    elif name == 'sigmoid':
        labels = tf.one_hot(labels, 5, axis=-1)
        loss = tf.losses.sigmoid_cross_entropy(labels, outputs)
    else:
        loss = 0
    return loss


def get_acc(labels, outputs, name='accuracy'):
    accuracy, accuracy_update = tf.metrics.accuracy(labels, outputs, name=name)
    accuracy_initializer = tf.variables_initializer(tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope=name))
    return accuracy, accuracy_update, accuracy_initializer
