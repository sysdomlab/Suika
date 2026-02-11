import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import socket
import pickle
import queue
import time
from logging import Logger
from enum import Enum


"""
Message Set
"""

"""
A sets of communication type identifier

A standard msg should be a tuple of 2 elements:
  - 1. type of message
  - 2. any addtional information
"""
class Worker2JobMsg(Enum):
  COMPLETED = 0
  FAILED = 1
  HEARTBEAT = 2
  PROFILE = 3
  EXITED = 4 # ack to Job2WorkerMsg.EXIT
  RECONFIG_START = 5 # for precise reconfig overhead measuring
  RECONFIG_COMPLETED = 6
  SHRANK = 7
  FF = 8 # fast forwarding info (now(wall clock time), cur_iter, iter_time) tuple
  
class Job2WorkerMsg(Enum):
  EXIT = 0
  RECONFIG = 1
  ACK = 2
  COMPLETED = 3 # in case that a reconfig happens when train ends, use this msg to notify new worker to quit.
  CHECKPOINT_RESUME = 4 # checkpoint-resume style reconfiguration (for baselines)
  FF = 5 # instruct worker to fast foward to (target_cur_iter)


class Executor2SchedulerMsg(Enum):
  """
  Message types from executor to scheduler.
  """
  SUCCEEDED = 0
  FAILED = 1
  HEARTBEAT = 2
  
class Scheduler2ExecutorMsg(Enum):
  """
  Message types from scheduler to executor.
  """
  EXEC = 0 # (EXEC, (exec_id, exec_cmd))
  EXIT = 1


"""
IPC
"""

"""
 A simple utility to coordinate training process switches
"""
wait_sleep_interval = 0.05
class InterProcessFlag:
  
  def __init__(self, manager):
    self.flags = manager.dict()  # Shared dictionary between processes
    
  def reset_all(self, exclude: List[str] = []):
    if not isinstance(exclude, list):
      exclude = [exclude]
      
    for flag in self.flags.keys():
      if flag not in exclude:
        self.flags[flag] = False
    
  def reset(self, flag: str):
    if flag in self.flags:
      self.flags[flag] = False
    
  def set(self, flag: str):
    self.flags[flag] = True
    
  def check(self, flag: str) -> bool:
    return self.flags.get(flag, False)
    
  def wait(self, flag: str, timeout: Optional[int] = None, exit_flag: Optional[Union[str, List[str]]] = None) -> bool:
    start_time = time.time()
    if exit_flag is None:
      exit_flag = []
    if isinstance(exit_flag, str):
      exit_flag = [exit_flag]

    while True:
      if any([self.check(exit_flag_) for exit_flag_ in exit_flag]):
        return False
      if self.check(flag):
        return True
      
      if timeout is not None and time.time() - start_time > timeout:
        return False
      time.sleep(wait_sleep_interval)


"""
Socket
Send/Recv any pickleable object

These two classes are primarily Copilot generated, but are well-tested.
"""

class SuikaSocketServer:

  port: int
  connections: Dict[int, socket.socket]  # rank to connection
  background: List[threading.Thread]
  running: bool
  recv_queues: Dict[int, queue.Queue]
  logger: Optional[Logger]
  
  def __init__(self, endpoint: str, logger: Optional[Logger] = None):
    """Initialize server with endpoint string 'host:port'"""
    host, port_str = endpoint.split(':')
    self.host = host
    self.port = int(port_str)
    self.connections = {}
    self.background = []
    self.running = False
    self.recv_queues = {}
    self.socket = None
    self.logger = logger
    self.lock = threading.Lock()  # Add lock for thread safety
  
  def log(self, msg: str, level=logging.DEBUG) -> None:
    """Log message using logger if available"""
    if self.logger:
      msg = f"[SocketServer]: {msg}"
      self.logger.log(level, msg)        
        
  def wait_for(self, dst: int, timeout: float = None) -> bool:
    """Wait for a connection from a specific client"""
    start_time = time.time()
    while timeout is None or time.time() - start_time < timeout:
      if dst in self.connections:
        return True
      time.sleep(wait_sleep_interval)  # Prevent CPU spin
    return False
  
  def send(self, dst: int, obj: Any) -> bool:
    """Send pickled object to client with specified rank"""
    if dst not in self.connections:
      return False
    
    try:
      data = pickle.dumps(obj)
      header = len(data).to_bytes(4, byteorder='big')
      
      # Get a reference to the connection to avoid race conditions
      with self.lock:  # Add a lock for thread safety
        if dst not in self.connections:  # Double-check after acquiring lock
          return False
        client_socket = self.connections[dst]
      
      # Send data (outside of lock to avoid blocking)
      client_socket.sendall(header + data)
      return True
    except Exception as e:
      self.log(f"Error sending to {dst}: {e}", logging.CRITICAL)
      self._close_connection(dst)
      return False
  
  def broadcast(self, obj: Any) -> Dict[int, bool]:
    """Send object to all connected clients"""
    results = {}
    for rank in list(self.connections.keys()):
      results[rank] = self.send(rank, obj)
    return results
  
  def recv(self, rank: int, wait: bool = False, timeout: float = None) -> Any:
    """Receive object from specific client"""
    if rank not in self.recv_queues:
      return None
    
    try:
      if wait:
        return self.recv_queues[rank].get(block=True, timeout=timeout)
      else:
        try:
          return self.recv_queues[rank].get_nowait()
        except queue.Empty:
          return None
    except queue.Empty:
      return None
  
  def recv_any(self, wait: bool = False, timeout: float = None) -> Tuple[Optional[int], Any]:
    """Receive object from any client"""
    if wait:
      start_time = time.time()
      while timeout is None or time.time() - start_time < timeout:
        for rank, q in list(self.recv_queues.items()):
          try:
            return rank, q.get_nowait()
          except queue.Empty:
            continue
        time.sleep(wait_sleep_interval)  # Prevent CPU spin
      return None, None
    else:
      for rank, q in list(self.recv_queues.items()):
        try:
          return rank, q.get_nowait()
        except queue.Empty:
          continue
      return None, None
  
  def start(self):
    """Start server in background thread (non-blocking)"""
    with self.lock:
      if self.running:
        self.log("SocketServer already running", level=logging.WARNING)
        return
      self.running = True
      
      self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      try:
        self.socket.bind((self.host, self.port))
      except OSError as e:
        self.log(f"Binding to {self.host}:{self.port} failed: {e}", level=logging.ERROR)
        raise e

      self.socket.listen(5)
      
    # Non-blocking: start listener thread
    thread = threading.Thread(target=self._listen)
    thread.daemon = True
    thread.start()
    self.background.append(thread)
    
    self.log(f"Server started on {self.host}:{self.port}")
  
  def close_connection(self, rank):
    if rank not in self.connections:
      return
    with self.lock:
      self._close_connection(rank)

  def stop(self):
    """Stop server and close all connections"""
    
    # Use a lock to safely close all connections
    with self.lock:
      self.running = False
      # Close all connections
      for rank in list(self.connections.keys()):
        self._close_connection(rank)
      
      # Close server socket
      if self.socket:
        try:
          self.socket.close()
        except Exception as e:
          self.log(f"Error closing server socket: {e}", logging.CRITICAL)
        self.socket = None
    
    # Wait for background threads to finish
    for thread in self.background:
      thread.join(timeout=0.5)
    
    self.log("Server stopped")
  
  def _listen(self):
    """Background thread that accepts connections"""
    self.socket.settimeout(0.5)  # Allow checking running flag
    
    while self.running:
      try:
        client_socket, addr = self.socket.accept()
        
        # Start client handler thread
        handler = threading.Thread(target=self._handle_client, 
                                  args=(client_socket, addr))
        handler.daemon = True
        handler.start()
        self.background.append(handler)
      except socket.timeout:
        continue
      except Exception as e:
        if self.running:
          self.log(f"Accept error: {e}", logging.CRITICAL)
  
  def _handle_client(self, client_socket, addr):
    """Handle new client connection"""
    client_socket.settimeout(5.0)  # Initial message timeout
    rank = None  # Store rank for error reporting
    
    try:
      # First message should contain client rank
      header = client_socket.recv(4)
      if len(header) != 4:
        client_socket.close()
        return
      
      msg_len = int.from_bytes(header, byteorder='big')
      data = client_socket.recv(msg_len)
      if len(data) != msg_len:
        client_socket.close()
        return
      
      rank = pickle.loads(data)
      if not isinstance(rank, int):
        client_socket.close()
        return
      
      self.log(f"Client {rank} connected from {addr}")
      
      # Set up connection with thread safety
      with self.lock:
        # Close existing connection with same rank if it exists
        if rank in self.connections:
          self._close_connection(rank)
        
        # Store new connection
        self.connections[rank] = client_socket
        self.recv_queues[rank] = queue.Queue()
      
      # Send acknowledgment
      ack = pickle.dumps("connected")
      ack_header = len(ack).to_bytes(4, byteorder='big')
      client_socket.sendall(ack_header + ack)
      
      # Set timeout for polling
      client_socket.settimeout(0.5)
      
      # Receive loop
      while self.running:
        try:
          # Check if socket is still valid
          if client_socket.fileno() == -1:
            self.log(f"Socket for client {rank} is no longer valid")
            break
            
          # Read message header
          header = client_socket.recv(4)
          if not header or len(header) != 4:
            break
          
          # Read message body
          msg_len = int.from_bytes(header, byteorder='big')
          data = b''
          remaining = msg_len
          
          while remaining > 0:
            chunk = client_socket.recv(min(4096, remaining))
            if not chunk:
              break
            data += chunk
            remaining -= len(chunk)
          
          if len(data) != msg_len:
            break
          
          # Process message
          obj = pickle.loads(data)
          self.recv_queues[rank].put(obj)
        except socket.timeout:
          # This is expected, continue the loop
          continue
        except ConnectionError as e:
          # Connection errors are expected when client disconnects
          self.log(f"Connection closed for client {rank}: {e}")
          break
        except OSError as e:
          # Handle bad file descriptor and other OS errors
          if e.errno == 9:  # Bad file descriptor
            self.log(f"Socket for client {rank} was closed")
          else:
            self.log("error", f"OS error receiving from client {rank}: {e}")
          break
        except Exception as e:
          self.log(f"Error receiving from client {rank}: {e}", logging.CRITICAL)
          break
      
      # Connection closed - use thread safety
      with self.lock:
        if rank in self.connections:
          self._close_connection(rank)
    except Exception as e:
      self.log(f"Client handler error for rank {rank}: {e}", logging.CRITICAL)
      try:
        client_socket.close()
      except:
        pass  # Ignore errors during close
  
  def _close_connection(self, rank):
    """Close client connection and clean up resources"""
    if rank in self.connections:
      try:
        # Check if socket is still valid before closing
        sock = self.connections[rank]
        if sock.fileno() != -1:  # Only close if socket is still valid
          sock.close()
      except Exception as e:
        self.log(f"Error closing connection for rank {rank}: {e}", logging.WARNING)
      
      # Remove from our collections regardless of close errors
      self.connections.pop(rank, None)
      self.recv_queues.pop(rank, None)
      self.log(f"Connection with client {rank} closed")


class SuikaSocketClient:

  client_id: int
  port: int
  background: List[threading.Thread]
  connection: Optional[socket.socket]
  running: bool
  recv_queue: queue.Queue
  logger: Optional[Logger]
  
  def __init__(self, client_id: int, endpoint: str, logger: Optional[Logger] = None):
    """Initialize client with ID and endpoint string 'host:port'"""
    host, port_str = endpoint.split(':')
    self.client_id = int(client_id)
    self.host = host
    self.port = int(port_str)
    self.connection = None
    self.background = []
    self.running = False
    self.recv_queue = queue.Queue()
    self.logger = logger
    self.lock = threading.Lock()  # Add lock for thread safety
  
  def log(self, msg: str, level: int = logging.DEBUG) -> None:
    """Log message using logger if available"""
    if self.logger:
      msg = f"[SocketClient-{self.client_id}]: {msg}"
      self.logger.log(level, msg)
  
  def start(self, timeout: float = 10.0) -> bool:
    """Connect to server and start listening thread, with retry"""
    start_time = time.time()
    while time.time() - start_time < timeout:
      try:
        # Connect to server
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((self.host, self.port))
        
        # Send client ID
        data = pickle.dumps(self.client_id)
        header = len(data).to_bytes(4, byteorder='big')
        self.connection.sendall(header + data)
        
        # Wait for acknowledgment
        ack_header = self.connection.recv(4)
        if not ack_header or len(ack_header) != 4:
          raise ConnectionError("Did not receive acknowledgment header")
        ack_len = int.from_bytes(ack_header, byteorder='big')
        ack_data = self.connection.recv(ack_len)
        if not ack_data or len(ack_data) != ack_len:
          raise ConnectionError("Did not receive full acknowledgment")
        response = pickle.loads(ack_data)
        
        if response != "connected":
          self.log(f"Connection failed: {response}", level=logging.CRITICAL)
          self.connection.close()
          self.connection = None
          # Continue retrying
        else:
          # Start receiver thread
          self.running = True
          thread = threading.Thread(target=self._listen)
          thread.daemon = True
          thread.start()
          self.background.append(thread)
          
          self.log(f"Connected to server as client {self.client_id}")
          return True # Connection successful
          
      except (ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError, socket.gaierror, ConnectionError, EOFError) as e:
        self.log(f"Connection attempt failed: {e}. Retrying...", level=logging.WARNING)
        if self.connection:
          try:
            self.connection.close()
          except Exception:
            pass # Ignore errors during close on failed attempt
          self.connection = None
      except Exception as e:
        self.log(f"Unexpected error during connection attempt: {e}", level=logging.CRITICAL)
        if self.connection:
          try:
            self.connection.close()
          except Exception:
            pass # Ignore errors during close on failed attempt
          self.connection = None

      # Wait before retrying
      time.sleep(0.1)

    # Timeout reached
    self.log(f"Connection timed out after {timeout} seconds.", level=logging.CRITICAL)
    return False
  
  def stop(self):
    """Disconnect from server"""
    self.running = False
    
    # Close connection with thread safety
    with self.lock:
      if self.connection:
        try:
          self.connection.close()
        except Exception as e:
          self.log(f"Error closing connection: {e}", level=logging.WARNING)
        self.connection = None
    
    # Wait for threads to finish
    for thread in self.background:
      thread.join(timeout=0.5)
    
    self.log(f"Client {self.client_id} stopped")
  
  def send(self, obj: Any) -> bool:
    """Send object to server"""
    with self.lock:  # Thread safety
      if not self.connection or not self.running:
        return False
      
      # Get socket reference inside lock
      connection = self.connection
    
    # Send data outside lock
    try:
      data = pickle.dumps(obj)
      header = len(data).to_bytes(4, byteorder='big')
      connection.sendall(header + data)
      return True
    except Exception as e:
      self.log(f"Send error: {e}", level=logging.CRITICAL)
      self.running = False
      return False

  def recv(self, wait: bool = False, timeout: float = None) -> Any:
    """Receive object from server"""
    if not self.running:
      return None
    
    try:
      if wait:
        return self.recv_queue.get(block=True, timeout=timeout)
      else:
        try:
          return self.recv_queue.get_nowait()
        except queue.Empty:
          return None
    except queue.Empty:
      return None
  
  def _listen(self):
    """Background thread that receives messages"""
    with self.lock:  # Thread safety
      if not self.connection:
        return
      self.connection.settimeout(0.5)  # Allow checking running flag
    
    while self.running:
      try:
        # Get socket reference safely
        with self.lock:
          if not self.connection:
            break
          connection = self.connection
            
        # Check if socket is still valid
        if connection.fileno() == -1:
          self.log("Socket is no longer valid")
          break
            
        # Read message header
        header = connection.recv(4)
        if not header or len(header) != 4:
          break
        
        # Read message body
        msg_len = int.from_bytes(header, byteorder='big')
        data = b''
        remaining = msg_len
        
        while remaining > 0:
          chunk = connection.recv(min(4096, remaining))
          if not chunk:
            break
          data += chunk
          remaining -= len(chunk)
        
        if len(data) != msg_len:
          break
        
        # Process message
        obj = pickle.loads(data)
        self.recv_queue.put(obj)
      except socket.timeout:
        # This is expected for the polling timeout
        continue
      except ConnectionError as e:
        # Connection errors are expected when server disconnects
        if self.running:
          self.log(f"Connection closed: {e}")
        break
      except OSError as e:
        # Handle bad file descriptor and other OS errors
        if e.errno == 9:  # Bad file descriptor
          self.log("Socket was closed")
        else:
          self.log(f"OS error: {e}", level=logging.CRITICAL)
        break
      except Exception as e:
        if self.running:
          self.log(f"Receive error: {e}", level=logging.CRITICAL)
        break
    
    # Connection lost - mark client as no longer running
    self.running = False
    if self.running:
      self.log(f"Connection lost")