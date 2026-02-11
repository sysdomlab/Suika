import cvxpy as cp
import numpy as np
import time

from Suika.external.Sia.utils import get_valid_configs, sia_logger
from Suika.external.Sia.hyperparam import DETAILED_LOGGING

"""
  The Sia artifact slightly differs from the algorithm in the paper. 
  
  It adds several minor improvements: 
    anothor regularization term for change-allocation,
    cross-cluster penalty,
    timesharing version scheduling
  
  Below is the implement reflecting default Sia setting in the artifact:
  # - change-allocation regularization: enabled (lambda_a=0.01)
  # - cross-cluster penalty: disabled (homogeneous cluster, no effect anyway)
  # - timesharing scheduling: disabled
  
  One may set lambda_a to 0 to faithfully use the original algorithm described in paper.
  
  This MILP_solve has been tested against Sia's artifact as reference. 
  Install cvxpy and GLPK (cvxopt perhaps) to run this code.
"""

def MILP_solve(
  goodput_matrix: np.ndarray,
  prev_allocation: np.ndarray,
  total_num_gpus: int,
  p_fairness: float,
  lambda_a: float,
  lambda_n: float,  
) -> np.ndarray:
    """
    Solve the MILP problem to find the optimal allocation of jobs to configurations.
    
    Args:
        goodput_matrix (np.ndarray): The goodput matrix with shape (n_jobs, n_configs). All adjustments to goodput are supposed to be done already.
        p_fairness (float): The fairness parameter.
        lambda_a (float): Regularization factor for change-allocation.
        lambda_n (float): Regularization factor for no-allocation.
        
    Returns:
        allocation: (np.ndarray) A binary matrix with shape (n_jobs, n_configs) indicating the allocation of jobs to configurations.
    """
    assert p_fairness != 0, "p_fairness should not be zero."
    
    # logging
    sia_logger.info(f"MILP_solve(): start with \n> goodput_matrix={goodput_matrix.shape}, total_num_gpus={total_num_gpus}, p_fairness={p_fairness}, lambda_a={lambda_a}, lambda_n={lambda_n}")
    if DETAILED_LOGGING:
      sia_logger.debug(f"MILP_solve(): goodput_matrix=\n{goodput_matrix}")
      sia_logger.debug(f"MILP_solve(): prev_allocation=\n{prev_allocation}")
        
    
    # prepare
    n_jobs, n_configs = goodput_matrix.shape
    job_weights = np.ones((1, n_jobs))
    
    # ensure correct sign even if user is careless
    opt_lambda_no_alloc = -abs(lambda_n) if p_fairness > 0 else abs(lambda_n)
    opt_lambda_change_alloc = -abs(lambda_a) if p_fairness > 0 else abs(lambda_a)
    
    # variables
    allocation = cp.Variable((n_jobs, n_configs), integer=True)
    
    # objective & constrants
    goal = cp.Maximize if p_fairness > 0 else cp.Minimize
    objective = cp.sum(job_weights @ (cp.multiply(goodput_matrix, allocation)))
    
    constraints = []
    
    # for i in range(n_jobs):
    #   # each job can only be allocated to one configuration
    #   constraints.append(cp.sum(allocation[i, :]) <= 1)
      
    #   # no-alloc penalty
    #   objective += opt_lambda_no_alloc * (1 - cp.sum(allocation[i, :]))
      
    #   # change-alloc penalty
    #   # if sum(prev_allocation[i, :]) > 0:
        
    #   #   # alloc_change = |allocation[i, :] - prev_allocation[i, :]|
    #   #   alloc_change = cp.Variable(shape=(n_configs,))
    #   #   objective += opt_lambda_change_alloc * cp.sum(alloc_change)
        
    #   #   constraints.append(alloc_change >= (prev_allocation[i, :] - allocation[i, :]))
    #   #   constraints.append(alloc_change >= (allocation[i, :] - prev_allocation[i, :]))
    #   #   constraints.append(alloc_change <= 1)

    #   # More efficient formulation for change penalty      
    #   if np.sum(prev_allocation[i, :]) > 0:
    #     # no_change is 1 if allocation is the same as prev_allocation, 0 otherwise.
    #     no_change = prev_allocation[i, :] @ allocation[i, :].T
    #     # penalty is applied when there is a change (1 - no_change)
    #     objective += opt_lambda_change_alloc * (1 - no_change)

    # Vectorized constraints and penalties
    # 1. Each job can only be allocated to one configuration
    constraints.append(cp.sum(allocation, axis=1) <= 1)
    
    # 2. No-allocation penalty
    # cp.sum(allocation, axis=1) is a vector of shape (n_jobs,) where each element is 0 or 1.
    # (1 - cp.sum(allocation, axis=1)) is 1 for jobs with no allocation.
    no_alloc_penalty = opt_lambda_no_alloc * cp.sum(1 - cp.sum(allocation, axis=1))
    objective += no_alloc_penalty
    
    # 3. Change-allocation penalty (for jobs that had a previous allocation)
    # Identify jobs that had a previous allocation
    prev_alloc_mask = np.sum(prev_allocation, axis=1) > 0
    
    if np.any(prev_alloc_mask):
        # no_change is 1 if allocation is the same as prev_allocation, 0 otherwise.
        # We use cp.sum(cp.multiply(...)) for element-wise product and sum, equivalent to a dot product for each job.
        no_change = cp.sum(cp.multiply(allocation[prev_alloc_mask, :], prev_allocation[prev_alloc_mask, :]), axis=1)
        
        # Penalty is applied when there is a change (1 - no_change)
        change_alloc_penalty = opt_lambda_change_alloc * cp.sum(1 - no_change)
        objective += change_alloc_penalty


    # boolean
    constraints.append(allocation >= 0)
    constraints.append(allocation <= 1)
    
    # gpu count
    valid_configs = np.array(get_valid_configs(total_num_gpus), dtype=np.int32)
    constraints.append(cp.sum(allocation @ valid_configs.T) <= total_num_gpus)  # each job can only be allocated to one config

    objective = goal(objective)
    
    # problem
    problem = cp.Problem(objective, constraints)
    
    st_time = time.time()
    problem.solve(solver=cp.GLPK_MI, glpk={'msg_lev': 'GLP_MSG_OFF'}, verbose=False)
    ed_time = time.time()
    
    # result
    sia_logger.info(f"MILP_solve(): result status={problem.status}, value={problem.value}, time={ed_time - st_time:.2f}sec")
    
    if problem.status != "optimal":
      sia_logger.warning(f"MILP_solve(): solver cannot find optimal")
    
    allocation = allocation.value
    
    if allocation is None:
      sia_logger.error(f"MILP_solve(): solver failed to find a solution. problem=\n{problem}")
      raise RuntimeError("MILP_solve(): solver failed to find a solution.")
      return np.zeros_like(prev_allocation, dtype=int)
    
    allocation = np.round(allocation).astype(int)
    
    if DETAILED_LOGGING:
      sia_logger.debug(f"MILP_solve(): allocation=\n{allocation}\n{visualize_allocation(allocation, total_num_gpus)}")
      
    return allocation


def visualize_allocation(allocation: np.ndarray, total_num_gpus: int) -> str:
  output = ["--- allocation summary ---"]
  total_gpus_used = 0
  valid_configs = get_valid_configs(total_num_gpus)
  for i, job_alloc in enumerate(allocation):
    if np.sum(job_alloc) > 0:
      alloc_idx = np.argmax(job_alloc)
      gpus = valid_configs[alloc_idx]
      total_gpus_used += gpus
      output.append(f"job {i}: alloc {gpus:2} gpus")
    else:
      output.append(f"job {i}: no alloc")
    
  output.append(f"total gpus used: {total_gpus_used}/{int(valid_configs[-1])}")
  output.append("--------------------------")
  return "\n".join(output)
  