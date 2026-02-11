# adapted from Sia open artifacts
#  - removed dynamic batch size related code

from Suika.external.Sia.utils import PerfParams, _wrap_PerfParams

# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import autograd
import numpy as np
import scipy.optimize


class GoodputFunction(object):

    def __init__(self, perf_params):
        self._perf_params = _wrap_PerfParams(perf_params) # perf_params may be a dict. so PerfParams(*perf_params) may fail. use _wrap instead

    # accum_steps = num_micro_batch_size - 1
    def __call__(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        return self.evaluate(num_nodes, num_replicas, atomic_bsz, accum_steps)
    
    def __repr__(self):
        rep = "GoodputFunction("
        rep += ('_perf_params: ' + str(self._perf_params))
        rep += ')'
        return rep

    # note: atomic_bsz - local batch size per micro-batch
    def evaluate(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)        
        return self.throughput(num_nodes, num_replicas, atomic_bsz,
                               accum_steps) * self.efficiency(batch_size)

    def throughput(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        accum_time = _predict_accum_time(self._perf_params, atomic_bsz)
        network_time = _predict_network_time(self._perf_params,
                                             num_nodes, num_replicas)
        optim_time = np.exp(_predict_log_optim_time(self._perf_params,
                                                    accum_time, network_time))
        total_time = accum_steps * accum_time + optim_time
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
        return batch_size / total_time

    def efficiency(self, batch_size):
        return 1.0
    
def fit_perf_params(num_nodes, num_replicas, atomic_bsz,
                    accum_step_time, optim_step_time):
    # Fit the performance model given accum time and optim time measurements
    # for different configurations of num_nodes, num_replicas, and atomic_bsz.

    # HACK: We want to use the original numpy module for calls from the
    # SpeedupFunction for performance reasons, but also need those functions to
    # use autograd.numpy when we want to differentiate them. We patch the
    # global np reference only for the code invoked rom this function.
    global np  # Replace numpy from autograd.
    orig_np = np
    np = autograd.numpy

    num_nodes = np.array(num_nodes)
    num_replicas = np.array(num_replicas)
    local_bsz = np.array(atomic_bsz)
    accum_step_time = np.array(accum_step_time)
    optim_step_time = np.array(optim_step_time)

    # Set initial params to reasonable values.
    params = [1e-1, 1e-2] * 3 + [1.0 + 1e-3]
    # Set lower/upper bounds for each parameter. Add a small slack to lower
    # bounds to avoid numerical instability issues.
    lower = [1e-8, 1e-8] * 3 + [1.0]
    upper = [np.inf, np.inf] * 3 + [10.0]
    if len(np.unique(atomic_bsz)) == 1:
        # Fix alpha_c if only observed a single atomic batch size.
        # This makes the speedup model optimistic with respect to
        # scaling up the batchsize. This will assign equal weight
        # to the constant and multplicative factors for accum time
        # if there is only a single datapoint (which is by far the
        # most likely case for this scenario)
        params[0] = upper[0] = lower[0] = np.mean(accum_step_time) / 2
    if not np.any(num_nodes > 1):
        # Fix alpha_n and beta_n if no multi-node observations.
        params[2] = upper[2] = lower[2]
        params[3] = upper[3] = lower[3]
    if not np.any(np.logical_and(num_nodes == 1, num_replicas > 1)):
        # Fix alpha_r and beta_r if no single-node/multi-replica observations.
        params[4] = upper[4] = lower[4]
        params[5] = upper[5] = lower[5]
    if not np.any(num_replicas > 2):
        # Fix beta_n and beta_r if no replicas > 2.
        params[3] = upper[3] = lower[3]
        params[5] = upper[5] = lower[5]
    bounds = scipy.optimize.Bounds(lower, upper, keep_feasible=True)
    args = (num_nodes, num_replicas, atomic_bsz,
            accum_step_time, optim_step_time)
    # FIXME: need to handle optimization failures and propagate to the Trainer.
    grad_fn = autograd.grad(_obj_fn)
    result = scipy.optimize.minimize(_obj_fn, params, args=args,
                                     jac=grad_fn, bounds=bounds)
    params = result.x
    if not any(num_nodes > 1):
        # Enforce prior: alpha_n and beta_n are at least alpha_r and beta_r.
        params[2] = max(params[2], params[4] * 1.1)
        params[3] = max(params[3], params[5] * 1.1)
    np = orig_np  # Restore original numpy.
    return PerfParams(*params)


def _rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())

# accum_step_time: comp time per microbatch
# optim_step_time: comp + commu time of last microbatch

def _obj_fn(params, num_nodes, num_replicas, atomic_bsz,
            accum_step_time, optim_step_time):
    params = PerfParams(*params)
    pred_accum = _predict_accum_time(params, atomic_bsz)
    pred_network = _predict_network_time(params, num_nodes, num_replicas)
    pred_log_optim = _predict_log_optim_time(params, pred_accum, pred_network)
    # RMSLError of accum step time predictions.
    err1 = _rmse(np.log(pred_accum), np.log(accum_step_time))
    # RMSLError of optim step time predictions.
    err2 = _rmse(pred_log_optim, np.log(optim_step_time))
    # L2 regularization towards a smaller gamma, because it's easier to
    # optimize the alpha and beta parameters when gamma is smaller.
    reg1 = 1e-3 * (params.gamma - 1) ** 2
    # Penalize retrogression terms to prefer a more optimistic model.
    reg2 = 1e-2 * ((params.beta_n / params.alpha_n) ** 2 +
                   (params.beta_r / params.alpha_r) ** 2)
    return err1 + err2 + reg1 + reg2



def _predict_accum_time(params, atomic_bsz):
    params = PerfParams(*params)
    # Forward/backward passes should scale linearly with the batch size.
    return params.alpha_c + params.beta_c * atomic_bsz


def _predict_log_optim_time(params, accum_time, network_time):
    gamma = PerfParams(*params).gamma
    return np.log(accum_time ** gamma + network_time ** gamma) / gamma


def _predict_network_time(params, num_nodes, num_replicas):
    params = PerfParams(*params)
    # Select the most significant link between replicas, currently either
    # inter-node (nodes > 1) or intra-node (replicas > 1). Note that if
    # replicas == 1 then neither of these two conditions are matched.
    conds = [num_nodes > 1, num_replicas > 1]
    # Bandwidth is bottlenecked by the most significant link, alpha models
    # the overhead of transferring data across that link.
    bottleneck = np.select(conds, [params.alpha_n, params.alpha_r], 1e-8)
    # Assuming ring all-reduce, communication happens in a number of rounds
    # equal to the number of replicas. beta models the performance
    # retrogression from increasing the number of replicas beyond 2.
    retrogress = np.select(conds, [params.beta_n, params.beta_r], 1e-8)
    retrogress = retrogress * np.maximum(num_replicas - 2, 1e-8)
    return (bottleneck + retrogress)
