# This is the task template for all model using suika_train

# copy this file to tasks/task.py and modify it accordingly

from numpy import broadcast
import torch
from torch.nn import Module
import torch.distributed as dist
import os
import sys
import logging
sys.path.append("../")

torch.set_default_dtype(torch.float16)
torch.set_default_device("cuda:0")

from Suika.utils import broadcast_objects, get_logger
from Suika.engine import PPEngine
from Suika.cluster.plan_config import PlanConfig
from Suika.training import suika_train, create_kwargs_template
from Suika.models.utils import ModelConfig, ModelStrategy, get_model_provider_from_spec
from Suika.migration import migration_load, migration_save
from Suika.models.utils import compute_model_parameter_size

def get_dummy_dataloader(lbsz, mc: ModelConfig, profiler=None):
  while True:
    if profiler is not None:
      profiler.start("batch_gen")
    result = torch.randn(lbsz, mc.n_positions, mc.n_embd)
    if profiler is not None:
      profiler.stop("batch_gen")
    yield result

def get_suika_train_task(callable):
  return (callable, create_kwargs_template(callable))

def prepare_context(plan_config: PlanConfig, model_spec: str):
  
  rank = os.getenv('RANK', '-1')
  job_name = os.getenv('JOB_NAME', 'suika_train')
  master_port = int(os.getenv('MASTER_PORT', '-1'))
  logger = get_logger(f"{job_name}/worker_[{rank}]_{master_port % 2}")
  
  model_logger = get_logger(f"{job_name}/worker_model_[{rank}]_{master_port % 2}")

  logger.info("prepare_context begin")
  logger.info("before init dist")
  device = torch.device("cuda:0")
  dist.init_process_group(backend="nccl", init_method="env://")
  logger.info("after init dist")
  

  pc = plan_config
  mc = ModelStrategy.from_spec(model_spec).model_config
  model_provider = get_model_provider_from_spec(model_spec, add_embedding_and_lm_head=False, init_on_meta=True)
  model = pc.to_PPEngine(model_provider=model_provider, device="cuda:0", init_on_meta=True, logger = model_logger)

  if "llama" in model_spec.lower():
    model.set_activation_mem_np_scale(8.0)
  else:
    model.set_activation_mem_np_scale(5.0)
  
  feature_loader = get_dummy_dataloader(model.lbsz, mc)
  label_loader = get_dummy_dataloader(model.lbsz, mc, model.prof)

  logger.info("prepare_context end")

  return {}, {
    'model': model, 
    'feature_loader': feature_loader,
    'label_loader': label_loader,
    'logger': logger
  }
  
def train_load(model: Module, IPC_queues, checkpoint_resume, logger):
  
  logger.info("train_load begin")

  def optimizer_provider():
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
  
  def loss_fn(output, target):
    loss = torch.nn.functional.mse_loss(output, target)
    return loss
  
  optimizer, other_context = migration_load(IPC_queues=IPC_queues, model=model, optimizer=optimizer_provider, log_func=logger.info, checkpoint_resume=checkpoint_resume)
  
  if other_context is not None:
    cur_iter = other_context
  else:
    cur_iter = 0
    
  logger.info("train_load end")

  return {'cur_iter': cur_iter}, {'optimizer': optimizer, 'loss_fn': loss_fn}

def train_save(model: Module, IPC_queues, checkpoint_resume, cur_iter: int, optimizer, logger):
  
  logger.info("train_save begin")

  migration_save(IPC_queues=IPC_queues, model=model, optimizer=optimizer, other_context=cur_iter, log_func=logger.info, checkpoint_resume=checkpoint_resume)
  
  dist.barrier()
  dist.destroy_process_group()
  
  logger.info("train_save end")

  return {}, {}

def train_step(model: PPEngine, optimizer, feature_loader, label_loader, loss_fn, cur_iter: int, n_iter: int, logger: logging.Logger, plan_config: PlanConfig, model_spec: str, ff_iter: int = None):
  if ff_iter is not None and ff_iter > cur_iter:
    logger.warning(f"fast forward from {cur_iter} to {ff_iter}")
    cur_iter = ff_iter

  cur_iter = broadcast_objects(cur_iter) # broadcast cur_iter from rank 0, ensure fast-forward consistency
  
  model.train()
  
  if cur_iter >= n_iter:
    file = f"./{model_spec}_profile_metis.csv"
    if dist.get_rank() == 0:
      logger.critical(f"CREATING FILE!!! {file}")
      with open(file, "a+") as f:
        tp = len(plan_config.acc_mesh[0][0])
        gbs = plan_config.global_batch_size()
        compute_time = model.prof.report(f"PPE-forward")
        optimizer_time = model.prof.report("optimizer")
        batch_gen_time = model.prof.report("batch_gen")
        max_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
        param = compute_model_parameter_size(model, True) * tp
        # pp dp tp, accum_step, optim_step, iter time
        f.write(f"({tp}, {gbs}), {compute_time:.10f}, {optimizer_time:.10f}, {batch_gen_time:.10f}, {max_memory + param / 1024 ** 2:.4f}, {param}")
        f.write("\n")
    else:
      logger.critical("NOT ME!")
    raise StopIteration

  logger.info(f"train_step it {cur_iter}/{n_iter}")

  model.forward(label_loader, labels=feature_loader, loss_fn=loss_fn)
  with model.profile_optimizer():
    optimizer.step()
    optimizer.zero_grad()
  
  profile = model.profile_result_to_scheduler()
    
  return  {'profile': profile, 'cur_iter': cur_iter + 1}, {}

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Suika Train Task")
  parser.add_argument('model_spec', type=str, help="Model specification string (e.g., 'GPT-350M')")
  parser.add_argument('n_iter', type=int, help="Number of iterations to train")
  
  args = parser.parse_args()
  
  import os
  rank = os.getenv('RANK', '-1')
  job_name = os.getenv('JOB_NAME', 'suika_train')
  logger_guard = get_logger(f"{job_name}/guard_[{rank}]")

  suika_train(
    prepare_context_task=get_suika_train_task(prepare_context),
    train_load_task=get_suika_train_task(train_load),
    train_save_task=get_suika_train_task(train_save),
    train_step_task=get_suika_train_task(train_step),
    shared_kwargs={
      "model_spec": args.model_spec,
      "n_iter": args.n_iter,
    },
    logger=logger_guard
  )

if __name__ == "__main__":
  main()