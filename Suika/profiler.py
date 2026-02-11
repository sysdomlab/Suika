import torch
import time
import tracemalloc
from typing import Any, Dict, List, Callable, Optional, Union, Iterable
import copy


class BaseProfiler():
  
  name_dict: Dict[str, bool]
  pattern: str
  
  def __init__(self, pattern: str = "{}"):
    self.pattern = ""
    self.name_dict = {}
    self.set_pattern(pattern)
  
  def set_pattern(self, pattern: str):
    assert pattern.count("{}") == 1, "pattern should have exactly one {}"
    self.pattern = pattern
    
  def _get_name(self, name):
    patterned = self.pattern.format(name)
    self.name_dict[patterned] = True
    return patterned
  
  def start(self, name, *args, **kwargs):
    name = self._get_name(name)
    return self._start_impl(name, *args, **kwargs)
    
  def stop(self, name, *args, **kwargs):
    name = self._get_name(name)
    return self._stop_impl(name, *args, **kwargs)
    
  def report(self, name, *args, **kwargs):
    name = self._get_name(name)
    return self._report_impl(name, *args, **kwargs)
  
  def full_report(self, *args, **kwargs) -> Dict[str, Any]:
    reports = {}
    for name in self.name_dict.keys():
      reports[name] = self._report_impl(name, *args, **kwargs)
    return reports
    
  def _start_impl(self, name: str):
    # the name in impl has been patterned
    pass
  
  def _stop_impl(self, name: str):
    pass
  
  def _report_impl(self, name: str):
    pass
  
  
  
"""
Time profiler
"""
class Timer(BaseProfiler):
  begin: Dict[str, Union[float, List[torch.cuda.Event]]]
  end: Dict[str, Union[float, List[torch.cuda.Event]]]
  durations: Dict[str, List[float]]
  
  strategy: str

  use_cuda_event: bool
  
  attach_to_last: Dict[str, List[bool]]
    
  """
  Time profiler
    strategy: 
      "accumulate"
      "average"
      "max"
      "last"
      "average_skip_first"
  """
  strategy_choice: List[str] = ["accumulate", "average", "max", "last", "average_skip_first"]
  
  
  
  def __init__(self, pattern: str = "{}", strategy: str = "accumulate", use_cuda_event: bool = True):
    super().__init__(pattern)
    
    self.begin = {}
    self.end = {}
    self.durations = {}
    
    assert strategy in Timer.strategy_choice, \
      f"strategy should be one of {Timer.strategy_choice}; got {strategy}"
      
    self.strategy = strategy
    self.use_cuda_event = use_cuda_event
    
    self.attach_to_last = {}
  
  
  def _start_impl(self, name: str, barrier_group: Optional[torch.distributed.ProcessGroup] = None):
    
    if barrier_group is not None:
      torch.distributed.barrier(group=barrier_group)
    
    if self.use_cuda_event:
      if not name in self.begin:
        self.begin[name] = []
      self.begin[name].append(torch.cuda.Event(enable_timing=True))
      self.begin[name][-1].record()
    else:
      self.begin[name] = time.perf_counter()
      
  def _stop_impl(self, name: str, barrier_group: Optional[torch.distributed.ProcessGroup] = None, attach_to_last = False):
    
    if barrier_group is not None:
      torch.distributed.barrier(group=barrier_group)
    
    if self.use_cuda_event:
      if not name in self.end:
        self.end[name] = []
      self.end[name].append(torch.cuda.Event(enable_timing=True))
      self.end[name][-1].record()
      
      if not name in self.attach_to_last:
        self.attach_to_last[name] = []
      self.attach_to_last[name].append(attach_to_last)
      
    else:
      self.end[name] = time.perf_counter()

      if self._valid_record(name, type="stop"):
        elapsed = self.end[name] - self.begin[name] if not self.use_cuda_event else \
          self.begin[name].elapsed_time(end_event = self.end[name]) / 1000.0 # convert to seconds
        
        if not name in self.durations:
          self.durations[name] = []
        self.durations[name].append(elapsed)
        
        del self.begin[name]
        del self.end[name]
              


  def _report_impl(self, name: str, strategy: Optional[str] = None):
    if strategy is None:
      strategy = self.strategy
    
    assert strategy in Timer.strategy_choice, \
      f"strategy should be one of {Timer.strategy_choice}; got {strategy}"
      
    
    
    if self.use_cuda_event:
      # lazy stop to delay synchronize
      torch.cuda.synchronize()
      
      if self._valid_record(name, type="stop"):
        assert len(self.begin[name]) == len(self.end[name]), \
          f"invalid record {name} with use_cuda_event, start / stop do not match"
        
        raw_durations = [
          b.elapsed_time(e) / 1000 for b, e in zip(self.begin[name], self.end[name])
        ]

        for i in range(len(raw_durations) - 1, -1, -1):
          if self.attach_to_last[name][i]:
            assert i != 0, f"invalid use of attach_to_last with record {name}"
            raw_durations[i - 1] += raw_durations[i]
            raw_durations.pop(i)
        
        if not name in self.durations:
          self.durations[name] = []
        self.durations[name].extend(raw_durations)
          
        self.begin[name] = []
        self.end[name] = []
        self.attach_to_last[name] = []
    
    if self._valid_record(name, type="report"): 
      if strategy == "accumulate":
        return sum(self.durations[name])
      elif strategy == "average":
        return sum(self.durations[name]) / len(self.durations[name])
      elif strategy == "max":
        return max(self.durations[name])
      elif strategy == "last":
        return self.durations[name][-1]
      else:
        return sum(self.durations[name][1:]) / (len(self.durations[name]) - 1) if len(self.durations[name]) > 1 else None
    else:
      return None
    
  def _valid_record(self, name, type: str) -> bool:
    
    if type == "stop":
      if (name in self.begin and \
        name in self.end):
        return True
      else:
        # assert False, f"Timer: invalid record for {name}, start / stop do not match"
        return False
    else:
      if name in self.durations:
        return True
      else:
        return False








"""
Two context managers to track CUDA memory allocation changes and Saved Tensors in forward.
borrowed from https://www.determined.ai/blog/act-mem-2
"""

class AllocatedMemContext:
    def __init__(self) -> None:
        # Ensure CUDA libraries are loaded:
        torch.cuda.current_blas_handle()

        self.before: dict[str, int] = {}
        self.after: dict[str, int] = {}
        self.delta: dict[str, int] = {}

    def _get_mem_dict(self) -> dict[str, int]:
        # Only need `allocated_bytes.all`-prefixed keys here
        key_prefix = "allocated_bytes.all."
        return {
            k.replace(key_prefix, ""): v
            for k, v in torch.cuda.memory_stats().items()
            if key_prefix in k
        }

    def __enter__(self) -> "AllocatedMemContext":
        self.before = self._get_mem_dict()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.after = self._get_mem_dict()
        self.delta = {k: v - self.before[k] for k, v in self.after.items()}
    
class SavedTensorContext:
    def __init__(
        self,
        ignored_tensors: Optional[Iterable[torch.Tensor]] = None,
    ) -> None:
        self._ignored_data_ptrs = (
            set()
            if ignored_tensors is None
            else {t.untyped_storage().data_ptr() for t in ignored_tensors}
        )

        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            data_ptr = saved_tensor.untyped_storage().data_ptr()
            if data_ptr not in self._ignored_data_ptrs:
                self.saved_tensor_dict[saved_tensor] = data_ptr
            return saved_tensor

        def unpack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            return saved_tensor

        self._saved_tensors_hook = torch.autograd.graph.saved_tensors_hooks(
            pack_hook, unpack_hook
        )

    def __enter__(self) -> "SavedTensorContext":
        self._saved_tensors_hook.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._saved_tensors_hook.__exit__(*args, **kwargs)

    @property
    def saved_tensor_mem(self) -> int:
        """
        The memory in bytes of all saved tensors, accounting for views into the same storage.
        """
        accounted_for = self._ignored_data_ptrs.copy()
        total_bytes = 0
        for t in self.saved_tensor_dict:
            data_ptr = t.untyped_storage().data_ptr()
            if data_ptr not in accounted_for:
                total_bytes += t.untyped_storage().nbytes()
                accounted_for.add(data_ptr)
        return total_bytes


"""
Memory profiler
"""
class Memorier(BaseProfiler):

  memory_usage: Dict[str, Dict[str, float]]  
  alloc_mem_ctx: Dict[str, AllocatedMemContext]
  device: torch.device
  
  # remark: CPU memory usage measurement is only for test purpose. do not use in production
  def __init__(self, pattern: str = "{}", device: torch.device = "cuda"):
    super().__init__(pattern)
    
    self.memory_usage = {}
    self.alloc_mem_ctx = {}
    self.device = device
    self.set_pattern(pattern)
  
  def _start_impl(self, name): 
    if str(self.device) == "cpu":
      tracemalloc.start()
    else:
      self.alloc_mem_ctx[name] = AllocatedMemContext()
      self.alloc_mem_ctx[name].__enter__()
  def _stop_impl(self, name):
    if str(self.device) == "cpu":
      current, peak = tracemalloc.get_traced_memory()
      self.memory_usage[name] = {"current": current / 1024 / 1024, "peak": peak / 1024 / 1024}
      tracemalloc.stop()
    else:
      assert name in self.alloc_mem_ctx, f"Memory profiler: {name} not started"
      self.alloc_mem_ctx[name].__exit__()
      result = {k: v / 1024 / 1024 for k, v in self.alloc_mem_ctx[name].delta.items()}
      result["change"] = result["current"]
      result["current"] = self.alloc_mem_ctx[name].after["current"] / 1024 / 1024
      result["peak"] = self.alloc_mem_ctx[name].after["peak"] / 1024 / 1024
      self.memory_usage[name] = result

  """
  valid fields for CUDA memory are:
    allocated, change, current, freed, peak
    
    allocated/change/freed are difference between "allocated/current/freed" at stop and start respectively
    current / peak is measured at stop
  """  
  def _report_impl(self, name, field: str = None):
    if name in self.memory_usage and isinstance(self.memory_usage[name], dict):
      if field is not None:
        return self.memory_usage[name][field]
      else:
        return self.memory_usage[name]
    return None
    
class Profiler(BaseProfiler):
  timer: Timer
  memorier: Memorier
      
  def __init__(
              self, 
              pattern: str = "{}", 
              strategy: str = "average_skip_first", 
              device: torch.device = "cuda",
              use_cuda_event: bool = None,
              ):
    super().__init__(pattern)
    
    if use_cuda_event is None:
      use_cuda_event = torch.device(device).type == "cuda"
      
    self.timer = Timer(pattern=pattern, strategy=strategy, use_cuda_event=use_cuda_event)
    self.memorier = Memorier(pattern=pattern, device=device)

  def _start_impl(self, name, require_time = True, require_memory =  False, enable_profiling = True, barrier_group: Optional[torch.distributed.ProcessGroup] = None, attach_to_last = False):
    if not enable_profiling:
      return
    if require_memory:
      self.memorier._start_impl(name)
    # attach_to_last does not matter in start
    if require_time: # start timer after memorier, since memorier measure takes too long
      self.timer._start_impl(name, barrier_group=barrier_group)
    
  def _stop_impl(self, name, require_time = True, require_memory = False, enable_profiling = True, barrier_group: Optional[torch.distributed.ProcessGroup] = None, attach_to_last = False):
    if not enable_profiling:
      return
    # stop timer first, since memorier measure takes time
    if require_time:
      self.timer._stop_impl(name, barrier_group=barrier_group, attach_to_last=attach_to_last)
    if require_memory:
      self.memorier._stop_impl(name)

  def _report_impl(self, name, require_time = True, require_memory = False, enable_profiling = True, strategy: Optional[str] = None):

    if enable_profiling:
      if require_time:
        timer_report = self.timer._report_impl(name, strategy=strategy)
      if require_memory:
        memory_report = self.memorier._report_impl(name)
    else:
      timer_report, memory_report = None, None
      
    if require_time and require_memory:
      return (timer_report, memory_report)
    elif require_time:
      return timer_report
    elif require_memory:
      return memory_report
    else:
      return None
    
  def count(self, name: str) -> int:
    """
    Count the number of times a name has been called
    """
    self.report(name) # force lazy 'append record to duration' in report of timer to be performed

    name = self._get_name(name)
    if self.timer._valid_record(name, type = "report"):
      # return len(self.timer.durations[name]) - (self.timer.strategy == "average_skip_first")
      # the reason we abandon -1 for avergage_skip_first is that it causes severe round issue.
      return len(self.timer.durations[name])
    else:
      return 0
    
  def full_report_str(self, require_time = True, require_memory = False) -> str:
    """
    Report all the metrics in a human readable format
    """
    reports = list(self.full_report(require_time=require_time, require_memory=require_memory).items())
    reports.sort(key=lambda kv: kv[0])
    report_str = ""
    
    for name, report in reports:
      if require_time and require_memory:
        time_report = report[0]
        memory_report = report[1]
      elif require_time:
        time_report = report
        memory_report = None
      elif require_memory:
        time_report = None
        memory_report = report
      else:
        continue
      
      time_str = f"{time_report:.5f} s " if time_report is not None else " "
      memory_str = "(" + ", ".join([f"{k}: {v:.1f}" for k, v in memory_report.items()]) + ") MB" if memory_report is not None else " "
      
      report_str += f"{name} {time_str}{memory_str}\n"
        
    return report_str
    
class DummyProfiler(BaseProfiler):
  """
  Dummy profiler that do nothing for either start or stop (which can repalce Optional[Profiler] for code simplicity)
  """
  def __init__(self, *args, **kwargs):
    super().__init__()
  
  def _start_impl(self, name, *args, **kwargs): pass
  def _stop_impl(self, name, *args, **kwargs): pass
  def _report_impl(self, name, *args, **kwargs): 
    # "Dummy profiler should not be called to report"
    return None
