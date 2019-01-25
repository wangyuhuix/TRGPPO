# this file does not need to handle Dim

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fcntl
import pickle
import time

import pathos.multiprocessing as multiprocessing

import tensorflow as tf

import baselines.TRPPO.KL2Clip_reduce_v3.prepare_data_train as prepare_data
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import baselines.common.tf_util as U
# path_root = '/media/d/e/et/baselines'
from baselines.common.distributions import DiagGaussianPd

import os
import pandas as pd
from baselines.common import tools

TabularActionPrecision = 5

# path_root = '../../KL2Clip'
path_root = os.path.abspath('./KL2Clip')

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs, momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


from baselines.common import tools_process

path_root_tabular = f'{path_root}/tabular'
tools.mkdir(path_root_tabular)
path_root_tabular += f'/precision_{TabularActionPrecision}'
tools.mkdir(path_root_tabular)
path_root_tabluar_locker = f'{path_root_tabular}/locker'
tools.mkdir(path_root_tabluar_locker)


class KL2Clip_tabular(object):
    def __init__(self, createtablur_initialwithpresol=True):
        self.deltas_dict = {}
        self.createtablur_initialwithpresol = createtablur_initialwithpresol
        ...

    def get_tabular(self, delta):
        save_path = f'{path_root_tabular}/{delta:.16f}'
        if delta in self.deltas_dict:
            pass
        # TODO: file lock
        elif os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            self.deltas_dict[delta] = tools.load_vars(save_path)
        else:
            with tools_process.FileLocker(f'{path_root_tabluar_locker}/{delta:.16f}'):
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    self.deltas_dict[delta] = tools.load_vars(save_path)
                else:
                    self.deltas_dict[delta] = self.create_tabular(delta)
                    tools.save_vars(save_path, self.deltas_dict[delta])
        return self.deltas_dict[delta]

    def create_tabular(self, delta):
        time0 = time.time()
        print(f'start to generate tabular of delta: {delta} Taburlar action precision{10 ** (-TabularActionPrecision)}')
        nn = KL2Clip_NN()
        actions = np.arange(0, 5, 10 ** (-TabularActionPrecision))
        ratio_min, ratio_max, min_mu_logsigma_fsolve, max_mu_logsigma_fsolve = \
            nn(action=actions, delta=delta, initialwithpresol=self.createtablur_initialwithpresol)
        # 注意这里一定要指定actions为float32 不然默认float64 后面用Dataframe索引时会有bug
        tabular = {np.float64(actions[i].__round__(TabularActionPrecision)): (ratio_min[i], ratio_max[i]) for i in
                   range(actions.shape[0])}
        df = pd.DataFrame(data=tabular, dtype=np.float32)
        time0 = time.time() - time0
        print(f'Successfully generate tabular, with time {time0}s')
        return df

    def __call__(self, action, delta):
        df = self.get_tabular(delta=delta)
        if action.ndim == 2:
            action = np.squeeze(action, 0)
        assert action.ndim == 1 and np.all(action >= 0)
        action = np.clip(action, 0, 4.99999).astype(np.float64)
        action = np.round(action, TabularActionPrecision)
        # print('df.head:\n', df.head())
        # print('action.round(TabularActionPrecision):\n', action)

        # TODO: check df columns
        if len(str(df.columns[0])) > 7 or len(str(df.columns[1])) > 7:
            df.columns = np.round(df.columns.values.astype(np.float64), TabularActionPrecision)

        ratio_min, ratio_max = np.split(df.loc[:, action].values, 2, axis=0)
        # ratio_min, ratio_max = np.split(df.reindex(columns=action).values, 2, axis=0)
        return np.squeeze(ratio_min, 0), np.squeeze(ratio_max, 0), None, None  # TODO: DEBUG!!!!!!


class KL2Clip_NN(object):
    def __init__(self):
        # self.dir = f'normal1D(delta x 100version)'#TOD tmp
        self.dir = f'normal1D'  # TOD tmp
        self.path_data = f'{path_root}/data/train'

        # my_feature_columns = [tf.feature_column.numeric_column(key='input', shape=[action_space * 2])]
        my_feature_columns = [tf.feature_column.numeric_column(key='input', shape=[2])]
        weight_columns = [tf.feature_column.numeric_column(key='weight', shape=())]

        self.session_config = tf.ConfigProto()
        self.session_config.gpu_options.allow_growth = True

        self.path_model = f'{path_root}/model/{self.dir}'
        folder_contents = os.listdir(self.path_model)
        must_in_folder = ['checkpoint', 'model.ckpt-408582.data-00000-of-00001', 'model.ckpt-408582.index']
        for i in must_in_folder:
            if not any(i in _ for _ in folder_contents):
                print('tensorflow estimator models missing !!!')
                exit()

        self.pool = multiprocessing.Pool(4)
        self.regressor = tf.estimator.Estimator(
            model_fn=my_model,
            model_dir=self.path_model,
            config=tf.estimator.RunConfig(
                session_config=self.session_config,
                keep_checkpoint_max=None
            ),
            params={
                'feature_columns': my_feature_columns,
                'weight_columns': weight_columns,
                'hidden_units': [1024, 512, 256],
                'label_dimension': 2 * 1,
                'action_space': 1
            })

        def get_ratios(batch_size):

            action = tf.placeholder(shape=[batch_size, 1], dtype=tf.float32)
            mu_logsigma_min = tf.placeholder(shape=[batch_size, 2], dtype=tf.float32)
            mu_logsigma_max = tf.placeholder(shape=[batch_size, 2], dtype=tf.float32)

            x0 = tf.zeros(shape=[batch_size, 2])
            distNormal = DiagGaussianPd(x0)
            dist_min, dist_max = DiagGaussianPd(mu_logsigma_min), DiagGaussianPd(mu_logsigma_max)
            ratio_min = dist_min.p(action) / distNormal.p(action)
            ratio_max = dist_max.p(action) / distNormal.p(action)
            fn_ratio = U.function([action, mu_logsigma_min, mu_logsigma_max], (ratio_min, ratio_max))
            return fn_ratio

        self._get_ratios = get_ratios

        if not tf.get_default_session():
            from baselines.TRPPO.KL2Clip_reduce.preparedata_KL2Clip_opt_tf import KL2Clip
            self.sess = KL2Clip.create_session()
            self.sess.__enter__()  # TODO: tmp
        else:
            self.sess = None

    def __del__(self):
        if self.sess is not None:
            self.sess.__exit__(None, None, None)

    def train(self, batch_size=200, epoch=10):
        # neural network (input: 1D-action, delta*D^{-1})   (output: mu, logsigma)
        # --- load data
        train_x, train_y, train_weight, eval_x, eval_y, eval_weight = prepare_data.load_data_normal(self.path_data,
                                                                                                    USE_MULTIPROCESSING=False)

        # train_x, train_y, eval_x, eval_y = prepare_data.load_data(self.path_data)
        # train_weight = np.ones( train_x.shape[0] )
        # eval_weight = np.ones(eval_x.shape[0])
        self.eval_x, self.eval_y = eval_x, eval_y
        num_train_samples = train_x.shape[0]
        train_x, eval_x = dict(input=train_x, weight=train_weight), dict(input=eval_x, weight=eval_weight)
        print(f'train_size:{num_train_samples}, batch_size:{batch_size}')
        input_fn_train = lambda: prepare_data.train_input_fn(train_x, train_y, batch_size, epoch)
        input_fn_eval = lambda: prepare_data.eval_input_fn(eval_x, eval_y, batch_size)
        self.input_fn_eval = input_fn_eval
        # return
        # --- config training parameters
        epoch_eval = 50
        save_summary_steps = 1000
        save_checkpoints_steps = int(
            round((epoch_eval * num_train_samples * 1.0 / batch_size) / save_summary_steps, 0)) * save_summary_steps
        print(f'save_checkpoints_steps:{save_checkpoints_steps}')
        self.regressor._config = self.regressor._config.replace(
            save_summary_steps=save_summary_steps,
            save_checkpoints_steps=save_checkpoints_steps,
            log_step_count_steps=save_checkpoints_steps,
        )

        # --- train
        checkpointslistener = MyCheckpointSaverListener(fn_before_save=self.eval)
        print('Start Train!')
        self.regressor.train(input_fn=input_fn_train, saving_listeners=[checkpointslistener])
        self.predict_test()

    def eval(self, checkpoint=None, checkpoint_path=None):
        assert not (checkpoint is not None and checkpoint_path is not None)
        if checkpoint is not None:
            checkpoint_path = f'{self.path_model}/model.ckpt-{checkpoint}'
        eval_result = self.regressor.evaluate(self.input_fn_eval, checkpoint_path=checkpoint_path)
        print(f"Test set average loss: {eval_result['loss']}, \nCheckpoint: {checkpoint}")

    @property
    def checkpoint_best(self):
        if hasattr(self, '_checkpoint_best'):
            return self._checkpoint_best
        files = tools.get_files(path_rel=f'{self.path_model}/eval', sort=False)
        eval_results = []
        for f in files:
            for i in tf.train.summary_iterator(f):
                for v in i.summary.value:
                    if v.tag == 'loss':
                        eval_results.append([i.step, v.simple_value])
        eval_results = np.array(eval_results)
        ind_min = np.argmin(eval_results[:, 1])
        step_min = int(eval_results[ind_min, 0])
        value_min = eval_results[ind_min, 1]
        print(f'step_best: {step_min}, value_best:{value_min}. (step_latest: {self.regressor.latest_checkpoint()})')
        self._checkpoint_best = step_min

        # import matplotlib.pyplot as plt
        # eval_results = eval_results[3:]
        # plt.plot( eval_results[:,0], eval_results[:,1] )
        # plt.show()
        return self._checkpoint_best

    @property
    def checkpoint_path_best(self):
        if not hasattr(self, '_checkpoint_path_best'):
            self._checkpoint_path_best = f'{self.path_model}/model.ckpt-{self.checkpoint_best}'
        return self._checkpoint_path_best

    def predict_test(self, checkpoint=None, checkpoint_path=None):
        assert not (checkpoint is not None and checkpoint_path is not None)
        # if not hasattr(self, "eval_x"):
        actions, deltas = np.split(self.eval_x, indices_or_sections=2, axis=1)
        ratio_min, ratio_max, *_ = self(np.squeeze(actions), deltas)
        print('predict ratio_min:', ratio_min)
        print('predict ratio_max:', ratio_max)

    def __call__(self, action, delta, USE_MULTIPROCESSING=False, initialwithpresol=False):
        '''

        :param action:
        :param delta:
        :return:
        '''
        if not isinstance(delta, float):
            delta = delta[0]
        assert action.ndim == 1

        batch_size = action.shape[0]
        # action[action <= 1e-3] = 1e-3  # TODO: debug!!!!!
        if initialwithpresol:
            print('Optimize with ordered action')
            action_t = action[-1:]
            # action_t = action
            predict_x = dict(input=np.stack((action_t, delta * np.ones_like(action_t)), axis=1),
                             weight=np.ones_like(action_t))
        else:
            predict_x = dict(input=np.stack((action, delta * np.ones_like(action)), axis=1),
                             weight=np.ones_like(action))

        # print('predict_x:\n', predict_x)
        predict_input_func = lambda: prepare_data.eval_input_fn(predict_x, None, batch_size)
        results = list(self.regressor.predict(predict_input_func))
        mu_tf_estimator = np.array([r['mu_tf_estimator'] for r in results])

        min_mu_tf_estimator, max_mu_tf_estimator = np.split(mu_tf_estimator, indices_or_sections=2, axis=1)
        calculate_mu = prepare_data.get_calculate_mu_func(ROBUST=False)
        action_clip = np.clip(action, -5., 5.)  # 神经网络 只能处理 action in [-5, 5] 区间

        # print('action_clip :', action_clip)
        # print('min_mu_tf_estimator :', min_mu_tf_estimator)
        if USE_MULTIPROCESSING:
            min_mu_fsolve = self.pool.map(calculate_mu,
                                          zip(min_mu_tf_estimator, action_clip, delta * np.ones_like(action_clip)))
            max_mu_fsolve = self.pool.map(calculate_mu,
                                          zip(max_mu_tf_estimator, action_clip, delta * np.ones_like(action_clip)))
        else:
            if initialwithpresol:
                '''
                    用之前优化的解去作为新的解的初始解
                    由于这个优化问题当a=0的时候得到的初始mu都一样（logsigma不一样，这是因为问题在a=0和a!=0的时候要解的不是一个式子）
                    所以下面是用倒序的方法计算
                '''
                print('Initialize with Previous Solutions!')

                sol_ini = min_mu_tf_estimator[-1]
                min_mu_logsigma_fsolve = []
                cnt_all = action_clip.shape[0]
                for ind, args in enumerate(reversed(list(zip(action_clip, delta * np.ones_like(action_clip),
                                                             -1 * np.ones_like(action_clip))))):
                    sol = calculate_mu([sol_ini] + list(args))
                    sol_ini = sol[0]  # sol=(mu,logsigma)
                    min_mu_logsigma_fsolve.append(sol)
                    tools.print_refresh(f'min:{ind}/{cnt_all}')
                min_mu_logsigma_fsolve.reverse()

                print('')
                sol_ini = max_mu_tf_estimator[-1]
                max_mu_logsigma_fsolve = []
                for ind, args in enumerate(reversed(list(zip(action_clip, delta * np.ones_like(action_clip),
                                                             0 * np.ones_like(action_clip))))):
                    sol = calculate_mu([sol_ini] + list(args))
                    sol_ini = sol[0]
                    max_mu_logsigma_fsolve.append(sol)
                    tools.print_refresh(f'min:{ind}/{cnt_all}')
                max_mu_logsigma_fsolve.reverse()
                print('')
            else:
                min_mu_logsigma_fsolve = list(
                    map(calculate_mu, zip(min_mu_tf_estimator, action_clip, delta * np.ones_like(action_clip),
                                          -1 * np.ones_like(action_clip))))
                max_mu_logsigma_fsolve = list(
                    map(calculate_mu, zip(max_mu_tf_estimator, action_clip, delta * np.ones_like(action_clip),
                                          0 * np.ones_like(action_clip))))

        min_mu_logsigma_fsolve = np.asarray(min_mu_logsigma_fsolve)
        max_mu_logsigma_fsolve = np.asarray(max_mu_logsigma_fsolve)

        # print(min_mu_logsigma_fsolve)  # TODO: logsigma有问题！

        if np.asarray(min_mu_logsigma_fsolve).ndim == 3:
            min_mu_logsigma_fsolve = min_mu_logsigma_fsolve.squeeze()
            max_mu_logsigma_fsolve = max_mu_logsigma_fsolve.squeeze()
        if np.asarray(min_mu_logsigma_fsolve).ndim == 1:
            min_mu_logsigma_fsolve = np.expand_dims(min_mu_logsigma_fsolve, 0)
            max_mu_logsigma_fsolve = np.expand_dims(max_mu_logsigma_fsolve, 0)

        fn_ratio = self._get_ratios(batch_size)
        ratio_min, ratio_max = fn_ratio(np.expand_dims(action, 1), min_mu_logsigma_fsolve, max_mu_logsigma_fsolve)

        # if self.sess is not None:
        #     self.sess.__enter__()
        # ratio_min, ratio_max = fn_ratio(action, mu_logsigma_min, mu_logsigma_max)
        # if self.sess is not None:
        #     self.sess.__exit__()
        return ratio_min, ratio_max, min_mu_logsigma_fsolve, max_mu_logsigma_fsolve


def my_model(features, labels, mode, params):
    input_ = tf.feature_column.input_layer(features, params['feature_columns'])  # shape (N, 2)
    # delta times 100 to normalize input, make sure action & delta have same magnitude
    a, d = tf.split(input_, 2, axis=1)
    # input_ = tf.concat((a, tf.log(d) + 5.), axis=1)  # TOO tmp
    # input_ = tf.concat((a, tf.log(d)), axis=1)  # TOO tmp
    input_ = tf.concat((a, d * 100), axis=1)
    action_space = params['action_space']
    net = input_
    # is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_training = True  # TODO: tmp
    # net = batch_norm_relu(net, is_training=is_training)
    for units in params['hidden_units']:
        # net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        # net = batch_norm_relu(net, is_training=is_training)  # TODO: tmp

    predictions = tf.layers.dense(net, params['label_dimension'], activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        actions, deltas = tf.split(input_, num_or_size_splits=2, axis=1)
        predict = {
            'action': actions,
            'delta': deltas,
            'mu_tf_estimator': predictions
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predict)

    # weight = tf.feature_column.input_layer(features, params['weight_columns'])
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions, weights=weight)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions, )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer()
    # optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=1e-2)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class MyCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, fn_before_save, *args, **kwargs):
        self.fn_before_save = fn_before_save
        super(MyCheckpointSaverListener).__init__(*args, **kwargs)

    def after_save(self, session, global_step_value):
        # print('Done writing checkpoint.')
        pass
        self.fn_before_save()


if __name__ == '__main__':
    kl2clip = KL2Clip_tabular(createtablur_initialwithpresol=False)
    kl2clip_ordered = KL2Clip_tabular(createtablur_initialwithpresol=True)
    tf.logging.set_verbosity(tf.logging.INFO)

    # a, b = tools.load_vars('../t/a.pkl')
    #
    # exit()
    a = np.arange(0, 4, 0.1)
    for D in [1]:
        tools.reset_time()
        sol1 = kl2clip_ordered(a, delta=0.1 / D)
        tools.print_time()

        tools.reset_time()
        sol2 = kl2clip(a, delta=0.1 / D)
        print(sol1[0] == sol2[0])
        print(sol1[1] == sol2[1])
        tools.print_time()
        # tools.save_vars('../t/a.pkl',a,b)
    # nn.train(batch_size=2048, epoch=1000)
    # nn.train(batch_size=2048, epoch=10)
    # nn.predict_test()
