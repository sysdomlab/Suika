# Defines TraceEntry, Trace class

from tracemalloc import stop
from typing import List, Dict, Optional
from collections import UserList
import time
import logging
import os
import threading

from Suika.cluster.scheduler import Scheduler, SimulatorScheduler
from Suika.cluster.job import Job
from Suika.models.utils import ModelStrategy

import random

SJF_NOISE = 0.3

def job_cmd_lookup(model_spec, n_iter):
  model_type = model_spec.split('-')[0]
  # e.g. ./GPT.sh GPT-350M 1000
  # return f"./{model_type}.sh {model_spec} {n_iter}"
  return f"./task.sh {model_spec} {n_iter}"

class TraceEntry():
  job: Job
  submit_time: float
  n_iter: int
  
  def __init__(self, job: Job, submit_time: float, n_iter: int, iter_time: float = None):
    self.job = job
    self.submit_time = float(submit_time) 
    self.n_iter = int(n_iter)
    
    job.other_context["n_iter"] = n_iter
    if iter_time is not None:
      job.other_context["iter_time"] = iter_time

  """
    Get TraceEntry from a csv line in format:
      (model_spec, submit_time, n_iter)
      
    The job's other_context will be populated with model_strategy, n_iter, and checkpoint_resume
  """
  @staticmethod
  def from_csv(csv_line, checkpoint_dir = None) -> "TraceEntry":
    # csv_line: model_spec, submit_time, n_iter
    assert len(csv_line) == 3, f"csv_line must be in format: (model_spec, submit_time, n_iter), got {csv_line}"
    
    model_spec, submit_time, n_iter = csv_line
    submit_time = float(submit_time)
    n_iter = int(n_iter)
    
    assert submit_time >= 0, f"submit time must be non-negative, got {submit_time}"
    assert n_iter >= 0, f"n_iter must be non-negative, got {n_iter}"
    
    try:
      ms = ModelStrategy.from_spec(model_spec)
    except KeyError as e:
      raise KeyError(f"TraceEntry.from_csv: Unkown model_spec {model_spec}")
    assert ms.iter_time is not None, f"{model_spec} missing iter time"
    iter_time = ms.iter_time
  
    pc = ms.get_plan_config()
    job_name = f"{model_spec}-it-{n_iter}@{(int(submit_time))}"
    checkpoint_resume = os.path.join(checkpoint_dir, job_name) if checkpoint_dir is not None else None

    random.seed(hash(job_name))
    noise = random.uniform(1 - SJF_NOISE, 1 + SJF_NOISE)

    other_context = {
      "model_strategy": ms,
      "n_iter": n_iter,
      "iter_time": iter_time * noise,
      "checkpoint_resume": checkpoint_resume
    }
    
    job = Job(
      job_name=job_name,
      plan_config=pc,
      cmd=job_cmd_lookup(model_spec=model_spec, n_iter=n_iter),
      cwd="./tasks",
      other_context=other_context
    )
    
    return TraceEntry(job, submit_time, n_iter, iter_time=iter_time)
  
  def __str__(self) -> str:
    return f"TraceEntry(job={self.job.job_name}, submit_time={self.submit_time}, n_iter={self.n_iter})"
  
  def __repr__(self) -> str:
    return self.__str__()
  
  def __hash__(self):
    return hash(self.job.job_name)

  def __eq__(self, other):
    if not isinstance(other, TraceEntry):
      return False
    return self.job.job_name == other.job.job_name
      
class Trace(UserList):
  
  def __init__(self, entries: List[TraceEntry] = None, logger: Optional[logging.Logger] = None):
    if entries is None:
      entries = []
    super().__init__(entries)
    self.logger = logger
  
  @staticmethod
  def from_csv(csv_file, checkpoint_dir = None) -> "Trace":
    entries = []
    try:
      with open(csv_file, 'r') as f:
        import csv
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
          if not line:
            continue
          if idx == 0 and line[0] == "model_spec": # skip header
            continue
          parsed_row = TraceEntry.from_csv(line, checkpoint_dir=checkpoint_dir)
          entries.append(parsed_row)            
    except FileNotFoundError:
      raise FileNotFoundError(f"Trace.from_csv: File {csv_file} not found")
    
    return Trace(entries)

  """
    run trace with a physical scheduler
  """  
  def run(self, scheduler: Scheduler) -> Dict[Job, Dict]:
    self.data.sort(key=lambda x: x.submit_time)
    
    try:
      self.log(f"Start to run trace with {len(self)} entries...", level=logging.INFO)
      begin_wallclock_time = time.time()

      def print_active_job():
        while scheduler.is_running:
          time.sleep(30)
          self.log(f">> active_jobs: {self._get_active_jobs_info(scheduler)}", level=logging.INFO)
          scheduler.log(f">> active_jobs: {self._get_active_jobs_info(scheduler)}", level=logging.DEBUG)

      print_thread = threading.Thread(target=print_active_job)
      print_thread.start()

      for entry in self:
        scheduler.register_submit_time(entry.submit_time)

      for entry in self:
        while scheduler._record_time() < entry.submit_time:
          time.sleep(1)
        
        scheduler.submit(entry.job)

      while scheduler.active_jobs:
        time.sleep(1)
    
      end_wallclock_time = time.time()
      self.log(f"Trace completed in {scheduler._record_time():.2f}(theoretical) / {end_wallclock_time - begin_wallclock_time:.2f}(wallclock)  sec.", level=logging.INFO)
      time.sleep(1)
      scheduler.stop(wait_for_guard=True)
      
    except Exception as e:
      self.log(f"Exception when running trace: {e}", level=logging.CRITICAL)
      scheduler.stop(wait_for_guard=False)
      time.sleep(1)
      raise e
        
    return scheduler.get_job_statistics()

  def run_simu(self, scheduler: SimulatorScheduler) -> Dict[Job, Dict]:
    self.data.sort(key=lambda x: x.submit_time)
    
    for entry in self.data:
      scheduler.submit(entry.job, entry.submit_time)
    
    start_time = time.time()
    scheduler.simulate()
    end_time = time.time()
    self.log(f"Simulation completed in {scheduler._record_time():.1f}(theoretical)/{end_time-start_time:.1f}(wallclock) sec.", level=logging.INFO)
    
    return scheduler.get_job_statistics()
  
  def _get_active_jobs_info(self, scheduler: Scheduler):
    active_jobs = [scheduler.jobs[jid] for jid in scheduler.active_jobs]
    active_jobs_str = []
    for job in active_jobs:
      n_acc = scheduler._get_job_n_acc(job)
      iter_time = scheduler._get_job_iter_time(job)
      age = scheduler._get_job_age(job)
      up_time = scheduler._get_job_up_time(job)
      active_jobs_str.append(f"{job}[{n_acc}/{iter_time:.2f}]~({up_time:.0f}/{age:.0f})")
    return ", ".join(active_jobs_str)

  """
    trace statistics
  """
    
  def job_size_CDF(self) -> List[float]:
    
    def get_job_size(job: Job) -> int:
      return job.usr_plan_config.world_size()
    
    job_sizes = [get_job_size(entry.job) for entry in self.data]
    
    max_job_size = max(job_sizes) if job_sizes else 0
    job_size_count = [0] * (max_job_size + 1)
    for size in job_sizes:
      job_size_count[size] += 1
    
    job_size_prefix_sum = [sum(job_size_count[:i+1]) for i in range(len(job_size_count))]
      
    return [prefix_sum / len(job_sizes) for prefix_sum in job_size_prefix_sum]
  
  def plot_job_size_CDF(self):
    import matplotlib.pyplot as plt
    
    cdf = self.job_size_CDF()
    cdf.append(1.0)
    x = list(range(len(cdf)))
    y_ticks = [0.1 * i for i in range(11)]
    
    plt.figure(figsize=(10, 6))
    plt.step(x, cdf, where='post', linewidth=2.5)
    plt.title('Job Size CDF')
    plt.xlabel('Job Size (Num of GPUs)')
    plt.ylabel('Cumulative Probability')
    plt.ylim(0, 1)
    plt.yticks(y_ticks)
    plt.xlim(0, len(x) - 1)
    plt.xticks(x)
    plt.grid(linestyle='--')
    plt.show()
  
  def avg_submit_interval(self) -> float:
    if not self.data:
      return 0.0
    
    self.data.sort(key=lambda x: x.submit_time)
    total_interval = 0.0
    for i in range(1, len(self.data)):
      total_interval += self.data[i].submit_time - self.data[i-1].submit_time
    
    return total_interval / (len(self.data) - 1) if len(self.data) > 1 else 0.0
  
  def avg_n_iter(self) -> float:
    if not self.data:
      return 0.0
    
    total_n_iter = sum(entry.n_iter for entry in self.data)
    return total_n_iter / len(self.data) if self.data else 0.0
  
  def std_n_iter(self) -> float:
    if not self.data:
      return 0.0
    
    avg_n_iter = self.avg_n_iter()
    variance = sum((entry.n_iter - avg_n_iter) ** 2 for entry in self.data) / len(self.data)
    return variance ** 0.5
  
  def log(self, msg: str, level = logging.DEBUG):
    if self.logger:
      self.logger.log(level, msg)
    else:
      print(msg, flush = True)
