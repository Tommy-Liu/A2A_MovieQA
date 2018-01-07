import sys


class EmbeddingTrainManager(object):
    # TODO(tommy8054): 1. Grid search 2. Training setup
    perturbation = {
        'learning_rate': lambda a, i: a * (10 ** i),
        'init_mean': lambda a, i: a + (0.1 * i),
        'init_scale': lambda a, i: a + (0.025 * i)
    }

    def __init__(self, arguments, args_parser):
        # Initialize all constant parameters,
        # including paths, hyper-parameters, model name...etc.
        self.args = arguments
        self.args_dict = self.args.__dict__
        self.val_dict = {'baseline': 0, 'max_length': 1, 'lock': True}
        self.args_parser = args_parser

        if len(sys.argv) == 1 or self.args.mode == 'inspect':
            self.search_param()
        else:
            self.retrieve_param(self.param_file)

        if self.args.reset:
            self.remove()
        du.exist_make_dirs(self.log_dir)
        du.exist_make_dirs(self.checkpoint_dir)
        if args.checkpoint_file:
            self.checkpoint_file = args.checkpoint_file
        else:
            self.checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        # Lock the current experiment.
        self.lock_exp()
        try:
            # tensorflow pipeline
            tf.reset_default_graph()
            self.data = EmbeddingData(self.args.batch_size)
            with tf.variable_scope('model'):
                self.model = self.get_model()
            self.loss = get_loss(self.args.loss, self.data, self.model) + \
                        get_loss(self.args.sec_loss, self.data, self.model) + \
                        tf.reduce_mean(tf.abs(tf.norm(self.model.output, axis=1) - 1))

            self.baseline = get_loss('mse', self.data, self.model)

            self.global_step = tf.train.get_or_create_global_step()
            self.local_step = tf.get_variable('local_step', dtype=tf.int32, initializer=0, trainable=False)

            self.lr_placeholder = tf.placeholder(tf.float32, name='lr_placeholder')
            self.step_placeholder = tf.placeholder(tf.int64, name='step_placeholder')
            self.dr_placeholder = tf.placeholder(tf.float32, name='dr_placeholder')
            self.learning_rate = tf.train.exponential_decay(self.lr_placeholder,
                                                            self.local_step,
                                                            self.args.decay_epoch * self.step_placeholder,
                                                            self.dr_placeholder,
                                                            staircase=True)
            with tf.variable_scope('optimizer'):
                self.optimizer = get_opt(self.args.optimizer, self.learning_rate)

            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            gradients, variables = list(zip(*grads_and_vars))
            for var in variables:
                print(var.name, var.shape)

            if not args.model == 'myconv':
                capped_grads_and_vars = [(tf.clip_by_norm(gv[0], args.clip_norm), gv[1]) for gv in grads_and_vars]
            else:
                capped_grads_and_vars = grads_and_vars

            self.train_op = tf.group(self.optimizer.apply_gradients(capped_grads_and_vars, self.global_step),
                                     tf.assign(self.local_step, self.local_step + 1))
            self.saver = tf.train.Saver(tf.global_variables(), )

            # Summary
            for idx, var in enumerate(variables):
                tf.summary.histogram('gradient/' + var.name, gradients[idx])
                tf.summary.histogram(var.name, var)

            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('baseline', self.baseline),
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.train_summaries_op = tf.summary.merge_all()

            self.init_op = tf.variables_initializer(tf.global_variables() + tf.local_variables())
            self.init_feed_dict = {self.model.init_mean: self.args.init_mean,
                                   self.model.init_stddev: self.args.init_scale}

            pp.pprint(tf.global_variables())
        finally:
            self.unlock_exp()

    def lock_exp(self):
        self.inject_param(val={'lock': True})

    def unlock_exp(self):
        self.inject_param(val={'lock': False})

    @property
    def exp_name(self):
        args_dict = vars(self.args)
        # Filter the different value.
        pairs = []
        for k in sorted(args_dict.keys()):
            default = self.args_parser.get_default(k)
            if default is not None \
                    and args_dict[k] != default \
                    and not isinstance(args_dict[k], bool):
                if isinstance(args_dict[k], numbers.Number):
                    pairs.append((k, '%.2E' % args_dict[k]))
                else:
                    pairs.append((k, str(args_dict[k])))
        # Compose the experiment name.
        if pairs:
            return '$'.join(['.'.join(pair) for pair in pairs])
        else:
            return '.'.join(['learning_rate', '%.2E' % self.args.learning_rate])

    @property
    def log_dir(self):
        return join(config.log_dir, 'embedding_log', self.exp_name)

    @property
    def checkpoint_dir(self):
        return join(config.checkpoint_dir, 'embedding_checkpoint', self.exp_name)

    @property
    def checkpoint_name(self):
        return join(self.checkpoint_dir, 'embedding')

    @property
    def param_file(self):
        return join(self.log_dir, '%s.json' % self.exp_name)

    @property
    def train_feed_dict(self):
        return {
            self.lr_placeholder: self.args.learning_rate,
            self.dr_placeholder: self.args.decay_rate,
            self.step_placeholder: self.args.decay_epoch * self.data.total_step
        }

    def search_param(self):
        exp_paths = glob(join(config.log_dir, 'embedding_log', '**', '*.json'), recursive=True)
        exps = []
        for p in exp_paths:
            d = du.jload(p)
            exp = {k: d['args'][k] for k in self.target}
            exp.update(d['val'])
            exps.append(exp)

        for idx, exp in enumerate(exps):
            print('%d.' % (idx + 1))
            pp.pprint(exp)

        choice = int(input('Choose one (0 or nothing=random perturbation):') or 0)
        while choice and exps[choice - 1]['lock']:
            choice = int(input('Your choice is locked. Please try the other one:') or 0)

        if choice:
            pp.pprint(exps[choice - 1])
            for k in self.target:
                self.args_dict[k] = exps[choice - 1][k]
            for k in self.val_dict.keys():
                self.val_dict[k] = exps[choice - 1][k]
        else:
            grid = set(tuple(self.diff[k](self.args_dict[k], exp[k])
                             for k in self.target)
                       for exp in exps)
            base = set(list(itertools.product(range(-1, 2), repeat=len(self.target))))
            chosen = base.difference(grid)
            print(len(grid), len(base), len(chosen))
            for idx, p in enumerate(list(iter(chosen))[0]):
                param = self.target[idx]
                self.args_dict[param] = self.perturbation[param](self.args_dict[param], p)
                print(param + ':', self.args_dict[param])

        self.args.reset = bool(input('Reset? (any input=True):'))

    def inject_param(self, arg=None, val=None):
        if arg:
            for k in arg:
                if k in self.args_dict:
                    self.args_dict[k] = arg[k]
        if val:
            for k in val:
                if k in self.val_dict:
                    self.val_dict[k] = val[k]
        d = {'args': self.args_dict, 'val': self.val_dict}
        print('1*')
        pp.pprint(d)
        print('2*')
        du.jdump(d, self.param_file)
        print('3*')

    def retrieve_param(self, file_name=None):
        if exists(file_name):
            d = du.jload(file_name)
            self.args_dict, self.val_dict = d['args'], d['val']

    def remove(self):
        if exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        if exists(self.log_dir):
            shutil.rmtree(self.log_dir)

    def train(self):
        config_ = tf.ConfigProto(allow_soft_placement=True, )
        config_.gpu_options.allow_growth = True
        with tf.Session(config=config_) as sess, tf.summary.FileWriter(self.log_dir) as sw:
            # Initialize all variables
            sess.run(self.init_op, feed_dict=self.init_feed_dict)
            if self.checkpoint_file:
                print(self.checkpoint_file)
                self.saver.restore(sess, self.checkpoint_file)

            max_length = self.val_dict['max_length']
            # Minimize loss until each of length commits.
            self.lock_exp()
            try:
                while max_length < 13:
                    delta, prev_loss, step = 1, 0, 1
                    try:
                        if not args.debug:
                            # Give instances of current length.
                            sess.run(self.data.iterator.initializer, feed_dict={
                                self.data.file_names_placeholder: self.data.get_records(
                                    list(range(1, max_length + 1)))})
                            # Minimize loss of current length until loss unimproved.
                            while abs(delta) > 1E-8:
                                _, bl, step, summary, loss = sess.run(
                                    [self.train_op, self.baseline, self.global_step, self.train_summaries_op,
                                     self.loss],
                                    feed_dict=self.train_feed_dict)
                                if step % 10 == 0:
                                    sw.add_summary(summary, tf.train.global_step(sess, self.global_step))
                                    print('|-- {:<15}: {:>30}'.format('total_step', self.data.total_step))
                                    print('\n'.join(['|-- {:<15}: {:>30.2E}'
                                                    .format(k, self.args_dict[k]) for k in self.target]))
                                if step % 100 == 0:
                                    self.saver.save(sess, self.checkpoint_name,
                                                    tf.train.global_step(sess, self.global_step))
                                print('[{:>2}/{:>2}] step: {:>6} delta: {:>+10.2E} base: {:>+10.2E} loss: {:>+10.2E}'
                                      .format(max_length, self.args.max_length, step, delta, bl, loss))
                                delta = bl - prev_loss
                                prev_loss = bl
                            # Increment the current length if loss is lower than threshold,
                            # or reset to 1 to search for other possibility.
                            pp.pprint(sess.run([self.data.vec, self.model.output]))
                            print('Length %d loss minimization done.' % max_length)
                        else:
                            bl = random.random() * 1E-3

                        if bl > 1E-1:
                            sess.run(self.init_op, feed_dict=self.init_feed_dict)
                            max_length = 1
                            print('Length %d fail. Reset all.' % max_length)
                        else:
                            print('1-')
                            sess.run(self.local_step.initializer)
                            print('2-')
                            #  Numpy type object is not JSON serializable. Since that, apply float to bl.
                            self.inject_param(val={'max_length': max_length, 'baseline': float(bl)})
                            print('3-')
                            max_length += 1
                            print('4-')
                            print('Length %d commits.' % max_length)
                            print('5-')
                    except KeyboardInterrupt:
                        self.saver.save(sess, self.checkpoint_name, tf.train.global_step(sess, self.global_step))
                        break
            finally:
                self.unlock_exp()

    def get_model(self):
        if self.args.model == 'myconv':
            model = MyConvModel(self.data, char_dim=self.args.char_dim, conv_channel=self.args.conv_channel)
        elif self.args.model == 'myrnn':
            model = MyModel(self.data, char_dim=self.args.char_dim,
                            hidden_dim=self.args.hidden_dim, num_layers=self.args.nlayers)
        elif self.args.model == 'mimick':
            model = EmbeddingModel(self.data, char_dim=self.args.char_dim, hidden_dim=self.args.hidden_dim)
        elif self.args.model == 'matrice':
            model = MatricesModel(self.data)
        else:
            model = None

        return model
