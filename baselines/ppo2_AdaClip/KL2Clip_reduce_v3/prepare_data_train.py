import time

import numpy as np

# from baselines.ppo2.prepare_data.tools import load_vars
# from baselines.ppo2.prepare_data import tools



from toolsm import tools
from toolsm.tools import load_vars, save_vars
import tensorflow as tf
import pathos.multiprocessing as multiprocessing
import os
from warnings import warn

def get_calculate_mu_func(ROBUST=False):
    def calculate_mu(args):
        import scipy
        from scipy.optimize import fsolve
        from scipy.special import lambertw
        mu_estimate, a, delta, flag = args

        def func(m):

            sigma = (m - a) * (m ** 2. - a * m - 1.) / a
            result__ = -scipy.log(sigma) + sigma + m ** 2 - 1 - 2 * delta
            # TODO: It will incur an error, and obtain complex64 or complex128 types
            # TypeError: Cannot cast array data from dtype('complex128') to dtype('float64') according to the rule 'safe'
            # if  result__.dtype == np.complex or result__.dtype == np.complex64:
            #     raise Exception('complex value')
                # result__ = result__.astype( np.float64 )
                # print(result__, float(result__), result__.dtype)
                # return None
            return result__

        def f_mu_to_logsigma(m):
            return 0.5 * np.log((m - a) * (m ** 2 - a * m - 1) / a)

        def get_real(m):
            if isinstance(m, complex):
                return m.real
            return m

        if a == 0:
            # return 0, 0.5 * np.log(fsolve(lambda x: -scipy.log(x) + x - 1 - 2 * delta, 0.))
            # return 0, 0.5 * np.log(-delta * lambertw((2 - delta) / (delta * np.e)) / (delta - 2))
            # return 0, 0.5 * np.log(-lambertw(-np.e ** (-2 * delta - 1), k=int(flag)))
            mu_sol = 0
            logsigma_sol = get_real( 0.5 * np.log(-lambertw(-np.e ** (-2 * delta - 1), k=int(flag))) )
        else:
            if ROBUST:
                try:
                    mu_sol = fsolve(func, mu_estimate, full_output=False)
                    logsigma_sol = f_mu_to_logsigma(mu_sol)
                except Exception as e:
                    warn(f'KL2clip calculate_mu error, return NAN! action: {a},  mu_estimate: {mu_estimate},  delta: {delta}')
                    mu_sol = np.NAN
                    logsigma_sol = np.NAN
            else:
                try:
                    mu_sol = fsolve(func, mu_estimate, full_output=False)
                    # print('----End')
                    logsigma_sol = f_mu_to_logsigma(mu_sol)
                except Exception as e:
                    # warn(f'KL2clip calculate_mu error, return zero! action: {a},  mu_estimate: {mu_estimate},  delta: {delta}')
                    mu_sol = 0
                    logsigma_sol = get_real(0.5 * np.log(-lambertw(-np.e ** (-2 * delta - 1), k=int(flag))))

        return mu_sol, logsigma_sol
    return calculate_mu


def load_data_normal(path_data, USE_MULTIPROCESSING=True):
    path_save = f'{path_data}/train_preprocessed_reduce_v3'

    if os.path.exists(f'{path_save}/data'):
        print(f'load data from {path_save}/data')
        vs = load_vars(f'{path_save}/data')
        return vs
    tools.mkdir(f'{path_data}/train_preprocessed')
    files = tools.get_files(path_rel=path_data, only_sub=False, sort=False, suffix='.pkl')

    actions, deltas, max_mu_logsigma, min_mu_logsigma = [], [], [], []
    for ind, f in enumerate(files[:1]):
        a_s_batch, _, _, ress_tf = load_vars(f)
        actions.append(a_s_batch)
        deltas.append(np.ones_like(a_s_batch) * ress_tf.delta)
        min_mu_logsigma.append(ress_tf.x.min)
        max_mu_logsigma.append(ress_tf.x.max)
    actions = np.concatenate(actions, axis=0)
    deltas = np.concatenate(deltas, axis=0)
    min_mu_logsigma = np.concatenate(min_mu_logsigma, axis=0)
    max_mu_logsigma = np.concatenate(max_mu_logsigma, axis=0)

    min_mu_tfopt, _ = np.split(min_mu_logsigma, indices_or_sections=2, axis=-1)
    max_mu_tfopt, _ = np.split(max_mu_logsigma, indices_or_sections=2, axis=-1)

    time0 = time.time()
    calculate_mu = get_calculate_mu_func(True)
    # TODO: 以下为mu_logsigma_fsolve
    if USE_MULTIPROCESSING:
        p = multiprocessing.Pool(4)
        min_mu_fsolve = p.map(calculate_mu, zip(min_mu_tfopt, actions, deltas))
        max_mu_fsolve = p.map(calculate_mu, zip(max_mu_tfopt, actions, deltas))
    else:
        min_mu_fsolve = list(map(calculate_mu, zip(min_mu_tfopt, actions, deltas)))
        max_mu_fsolve = list(map(calculate_mu, zip(max_mu_tfopt, actions, deltas)))

    min_mu_fsolve = [_[0] for _ in min_mu_fsolve]
    max_mu_fsolve = [_[0] for _ in max_mu_fsolve]
    # f_mu_to_logsigma = lambda m, a: (m - a) * (m ** 2 - a * u - 1) / a
    time1 = time.time()
    print(time1 - time0)
    mu_tf_opt = np.concatenate((min_mu_tfopt, max_mu_tfopt), axis=1)
    mu_fsolve = np.stack(
        (np.concatenate(min_mu_fsolve, axis=0).squeeze(),
         np.concatenate(max_mu_fsolve, axis=0).squeeze())
        , axis=1)
    print(mu_tf_opt - mu_fsolve)
    # exit()

    inds_shuffle = np.random.permutation(actions.shape[0])
    all_ = np.concatenate((actions, deltas, mu_fsolve), axis=1)[inds_shuffle]
    all_ = all_[~np.isnan(all_).any(axis=1)]
    inputs_all, outputs_all = np.split(all_, indices_or_sections=2,
                                       axis=1)  # (actions, deltas) (lambda_min_true, lambda_max_true)
    weights = np.ones(shape=(inputs_all.shape[0],))

    print(outputs_all.shape)

    ind_split = -3000
    train_x, train_y, train_weight = \
        inputs_all[:ind_split], outputs_all[:ind_split], weights[:ind_split]
    eval_x, eval_y, eval_weight = \
        inputs_all[ind_split:], outputs_all[ind_split:], weights[ind_split:]
    save_vars(f'{path_save}/data', train_x, train_y, train_weight, eval_x, eval_y, eval_weight)
    return train_x, train_y, train_weight, eval_x, eval_y, eval_weight,


def load_data(path_data, action_space, force_reload=False):
    path_data_processed = path_data + ', processed'
    file_data_processed = path_data_processed + '/data'
    if not force_reload and os.path.exists(file_data_processed):
        print(f'load data from {file_data_processed}')
        vs = load_vars(file_data_processed)
        return vs

    print(f'load data from {path_data}')
    tools.mkdir(path_data_processed)
    files = tools.get_files(path_rel=path_data, sort=True)
    # inputs_final, outputs_final = np.zeros((0, 2)), np.zeros((0, 4))
    inputs_final, outputs_final = np.zeros((0, 2 * action_space)), np.zeros((0, 4 * action_space))
    counts = np.zeros((len(files)), dtype=np.int)
    for ind, f in enumerate(files):
        mu0s_ats_batch, logsigma0s_batch, ress = load_vars(f)
        inputs = np.concatenate((mu0s_ats_batch, logsigma0s_batch), axis=-1)

        max_values = np.array([res['max'].x for res in ress])
        min_values = np.array([res['min'].x for res in ress])
        outputs = np.concatenate((max_values, min_values), axis=-1)

        inputs_final = np.concatenate((inputs_final, inputs))  # shape:(None, 2)
        outputs_final = np.concatenate((outputs_final, outputs))  # shape:(None, 4)
        counts[ind] = mu0s_ats_batch.shape[0]
    weights = []
    cnt_normalize = counts.mean()
    for cnt in counts:
        weight = cnt_normalize * 1. / cnt * np.ones(cnt)
        weights.append(weight)
    weights = np.concatenate(weights, axis=0)

    # final = np.concatenate((inputs_final, outputs_final), axis=-1)

    # --- delete nan and inf
    # final = final[~np.isnan(final).any(axis=1)]
    # final = final[~np.isinf(final).any(axis=1)]
    inds_reserve = np.logical_and(~np.isnan(outputs_final).any(axis=1), ~np.isinf(outputs_final).any(axis=1))
    inputs_final = inputs_final[inds_reserve]
    outputs_final = outputs_final[inds_reserve]
    weights = weights[inds_reserve]

    # --- shuffle
    # np.random.shuffle(final)
    N = inputs_final.shape[0]
    inds_shuffle = np.random.permutation(N)
    inputs_final = inputs_final[inds_shuffle]
    outputs_final = outputs_final[inds_shuffle]
    weights = weights[inds_shuffle]

    # inputs_final, outputs_final = np.split(final, indices_or_sections=[2], axis=-1)

    ind_split = -500
    train_x, train_y, train_weight = \
        inputs_final[:ind_split], outputs_final[:ind_split], weights[:ind_split]
    eval_x, eval_y, eval_weight = \
        inputs_final[ind_split:], outputs_final[ind_split:], weights[ind_split:]
    save_vars(file_data_processed, train_x, train_y, train_weight, eval_x, eval_y, eval_weight)
    return train_x, train_y, train_weight, eval_x, eval_y, eval_weight


def train_input_fn(features, labels, batch_size, epoch):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(labels.shape[0]).repeat(epoch).batch(batch_size)

    # Return the dataset.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction
    将高维的均值方差reshape为一维"""
    features = features['input']
    if features.shape[1] > 2:
        mu0s, logsigma0s = tf.split(features, num_or_size_splits=2, axis=-1)
        mu0s = tf.reshape(mu0s, (-1, 1))
        logsigma0s = tf.reshape(logsigma0s, (-1, 1))
        features = tf.concat((mu0s, logsigma0s), axis=1)

    assert features.shape[1] == 2

    features = {'input': features}
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    # path_root = '/media/d/e/et/baselines'
    # path_root += '/KL2Clip/data/'
    # path_data = f'{path_root}/D:1, delta:0.01'
    # load_data(path_data, force_reload=True)
    path_root = '/home/hugo/Desktop/wxm/KL2Clip'
    path_data = f'{path_root}/data/train'
    temp = load_data_normal(path_data)
    ...
