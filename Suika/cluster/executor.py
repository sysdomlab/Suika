"""
Instance to launch jobs that runs on each node. An alternative to direct SSH access launch.

The Executor does not monitor the job execution, it only sends the command received from scheduler to the node it is running on.
"""

from typing import Optional
from Suika.cluster.commu import SuikaSocketClient, Executor2SchedulerMsg, Scheduler2ExecutorMsg

import threading as th
import time
import logging
import subprocess

from Suika.utils import get_logger

class Executor:
  
  client: SuikaSocketClient
  start_timeout: int

  guard_thread: th.Thread
  
  logger: Optional[logging.Logger]
  
  
  def __init__ (self, node_id: int, endpoint: str, logger: Optional[logging.Logger] = None, start_timeout: Optional[int] = None):
    
    self.client = SuikaSocketClient(
      client_id=node_id,
      endpoint=endpoint,
    )
    self.start_timeout = start_timeout if start_timeout is not None else 60

    self.logger = logger
    
    self.guard_thread = th.Thread(target=self.guard)
    self.guard_thread.start()
    

  def alive(self):
    return self.client.running
  
  def guard(self):
    exec_sleep_interval = 0.1
    heart_beat_interval = 30 # should be smaller than timeout on scheduler side
    last_heart_beat = time.time()
    
    self.log(f"connecting to {self.client.host}:{self.client.port}")
    connected = self.client.start(timeout=self.start_timeout)
    
    if not connected:
      self.log(f"failed to connect to {self.client.host}:{self.client.port}", level=logging.ERROR)
      return
    else:
      self.log(f"connected")
    
    while self.alive():
      msg = self.client.recv()
      if msg is None:
        pass
      else:
        msg_type, value = msg
        if msg_type == Scheduler2ExecutorMsg.EXEC:     
          # rank is the rank of the process inside a job     
          rank, exec_cmd = value

          self.log(f"recv <exec>({rank})\n" \
                   f"     {exec_cmd}", level=logging.DEBUG)
          self.exec(rank, exec_cmd)
        elif msg_type == Scheduler2ExecutorMsg.EXIT:
          self.log("recv <exit>", level=logging.DEBUG)
          
          self.client.stop()
          break

      if time.time() - last_heart_beat > heart_beat_interval:
        # self.client.send(
        #   (Executor2SchedulerMsg.HEARTBEAT, None)
        # )
        self.log(f"heartbeat", level=logging.DEBUG)
        last_heart_beat = time.time()

      time.sleep(exec_sleep_interval)
    self.log("exiting")
      
  def _exec_monitor(self, rank: int, exec_cmd: str):
    """
    Internal method to monitor a single command execution in a thread.
    Sends feedback to the scheduler.
    """
    timeout = 0.5 # seconds
    process = None
    try:
      self.log(f"exec({rank}): issued")
      process = subprocess.Popen(exec_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      
      try:
        stdout, stderr = process.communicate(timeout=timeout) 
        return_code = process.returncode
        
        if return_code == 0:
          self.log(f"exec({rank}): succeeded (competed within timeout)")
          self.client.send((Executor2SchedulerMsg.SUCCEEDED, rank))
        else:
          self.log(
            f"exec({rank}): failed with code {return_code}\nstdout:\n{stdout.decode(errors='ignore')}\nstderr:\n{stderr.decode(errors='ignore')}"
            , level=logging.ERROR)
          self.client.send((Executor2SchedulerMsg.FAILED, rank))
          
      except subprocess.TimeoutExpired:
        self.log(f"exec({rank}): succeeded (timeout expired)")
        self.client.send((Executor2SchedulerMsg.SUCCEEDED, rank))

    except Exception as e:
        self.log(f"exec({rank}): failed with exception: {e}", level=logging.ERROR)
        if process and process.poll() is None:
          try:
            process.terminate()
            process.wait(timeout=0.5)
          except Exception as term_e:
            self.log(f"exec({rank}): error during termination: {term_e}", level=logging.ERROR)
        self.client.send((Executor2SchedulerMsg.FAILED, rank))

  def exec(self, rank: int, exec_cmd: str):
    """
    Execute a command on the node non-blockingly.
    Starts a monitor thread to handle feedback.
    
    Args: rank: the rank of process to launch, for Job to identify feedback
          exec_cmd: the command to execute
    """
    rank = int(rank)
    monitor_thread = th.Thread(target=self._exec_monitor, args=(rank, exec_cmd))
    monitor_thread.daemon = True
    monitor_thread.start()
    
  def join(self):
    self.guard_thread.join()
    
  def log(self, msg: str, level: int = logging.INFO):
    if self.logger is not None:
      msg = f"[executor]({self.client.client_id}): {msg}"
      self.logger.log(level, msg)


"""
@EXPORT
name: run_executor
usage: invoke as a main function of a script
argv: 
  --node-id <int> 
  --endpoint <str> (host:port format)
  --config <str> 
  (optional) --log <file> (default to stdout)
  (optional) --start-timeout <int> (default to 60 seconds)

  locally launch an executor with one of the three methods:
  
  1. specify node_id and endpoint
  2. specify only config file, deduce both node_id and endpoint 
    (hostname must exactly match node name for node_id deduction)
  3. specify node_id and config_file, deduce endpoint
  
  note that --config does not have a default to ~/.suika.json
  
  hostname will first lookup up env var HOSTNAME, if not exists, fall back to real hostname.
  
example usage
  python -m Suika.cluster.executor --node-id 0 --config ./.suika.json
"""
def run_executor():
  import argparse
  import os
  
  parser = argparse.ArgumentParser(description="Run executor.")
  parser.add_argument("--node-id", "--node_id", type=int, default=None, help="Node ID")
  parser.add_argument("--endpoint", type=str, default=None, help="Scheduler Endpoint")
  parser.add_argument("--config", type=str, default=None, help="Scheduler Config")
  parser.add_argument("--log", type=str, default=None, help="Log file name")
  parser.add_argument("--start-timeout", "--start_timeout", type=int, default=None, help="Start timeout in seconds to connect to scheduler")
  
  args = parser.parse_args()
  
  node_id, endpoint = args.node_id, args.endpoint
  config = args.config
  log = args.log
  start_timeout = args.start_timeout

  # Load the config file
  if config is not None:
    import json
    with open(config, 'r') as f:
      config = json.load(f)
      
  assert not (node_id is not None and endpoint is not None and config is not None), "Cannot specify all three of node_id, endpoint and config file"
  
  # Deduce node_id
  if node_id is None:
    assert config is not None, "Must specify node_id or config file for node_id deduction"
    import socket
    hostname = os.environ.get("HOSTNAME", socket.gethostname())
    for node_name in config["node"]:
      if node_name == hostname:
        node_id = config["node"][node_name]["node_id"]
        break
    assert node_id is not None, f"Cannot find node_id for {hostname} in config file"
  
  # Deduce endpoint
  if endpoint is None:
    assert config is not None, "Must specify endpoint or config file for endpoint deduction"
    from Suika.cluster.scheduler import default_scheduler_port_begin
    
    endpoint = f"{config['scheduler_addr']}:{config.get('scheduler_port_begin', default_scheduler_port_begin)}"
  
  logger = get_logger(f"executor_[{node_id}]" if not args.log else args.log)

  # Launch the executor
  executor = Executor(node_id=node_id,
           endpoint=endpoint,
           logger=logger,
           start_timeout=start_timeout 
          )
  
  executor.join()

if __name__ == "__main__":
  run_executor()
  