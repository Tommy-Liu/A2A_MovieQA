import tensorflow as tf

import utils.model_utils as mu
from model.basic_model import BasicModel
from raw_input import Input


class SetupModel(BasicModel):
    def __init__(self, attn, checkpoint):
        super(SetupModel, self).__init__()
        self.data = Input()
        self.output = None
        self.saver = None
        self.best_saver = None
        self.train_gv_summaries_op = None
        self.val_init_op_list = None
        self.train_init_op_list = None
        self.train_op_list = None
        self.train_summaries_op = None
        self.val_op_list = None
        self.attn = attn
        self.checkpoint = checkpoint

    def eval_setup(self):
        val_answer = tf.argmax(self.output, axis=1)
        val_accuracy, val_accuracy_update, val_accuracy_initializer \
            = mu.get_acc(self.data.gt, tf.argmax(self.output, axis=1), name='val_accuracy')

        val_summaries_op = tf.summary.scalar('val_accuracy', val_accuracy)
        self.val_init_op_list = [self.data.initializer, val_accuracy_initializer]
        self.val_op_list = [val_accuracy, val_accuracy_update, val_summaries_op]

    def train_setup(self):
        main_loss = mu.get_loss(self._hp['loss'], self.data.gt, self.output)
        regu_loss = tf.losses.get_regularization_loss()

        loss = main_loss + regu_loss

        train_answer = tf.argmax(self.output, axis=1)
        train_accuracy, train_accuracy_update, train_accuracy_initializer \
            = mu.get_acc(self.data.gt, train_answer, name='train_accuracy')

        global_step = tf.train.get_or_create_global_step()

        decay_step = int(self._hp['decay_epoch'] * len(self.data))
        learning_rate = mu.get_lr(self._hp['decay_type'], self._hp['learning_rate'], global_step,
                                  decay_step, self._hp['decay_rate'])

        optimizer = mu.get_opt(self._hp['opt'], learning_rate, decay_step)

        grads_and_vars = optimizer.compute_gradients(loss)
        # grads_and_vars = [(tf.clip_by_norm(grad, 0.01, axes=[0]), var) if grad is not None else (grad, var)
        #                   for grad, var in grads_and_vars ]
        gradients, variables = list(zip(*grads_and_vars))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(optimizer.apply_gradients(grads_and_vars, global_step),
                                train_accuracy_update)

        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables())

        # Summary
        train_gv_summaries = []
        for idx, grad in enumerate(gradients):
            if grad is not None:
                train_gv_summaries.append(tf.summary.histogram('gradients/' + variables[idx].name, grad))
                train_gv_summaries.append(tf.summary.histogram(variables[idx].name, variables[idx]))

        train_summaries = [
            tf.summary.scalar('train_loss', loss),
            tf.summary.scalar('train_accuracy', train_accuracy),
            tf.summary.scalar('learning_rate', learning_rate)
        ]
        self.train_summaries_op = tf.summary.merge(train_summaries)
        self.train_gv_summaries_op = tf.summary.merge(train_gv_summaries + train_summaries)

        self.train_init_op_list = [self.data.initializer, train_accuracy_initializer]

        self.train_op_list = [train_op, loss, train_accuracy, global_step]

        # if attn:
        #     self.train_op_list += [self.train_model.sq, self.train_model.sa,
        #                            self.train_data.gt, self.train_answer]
        #     self.val_op_list += [self.val_model.sq, self.val_model.sa,
        #                          self.val_data.gt, self.val_answer]
