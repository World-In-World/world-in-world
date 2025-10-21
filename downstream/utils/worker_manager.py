#!/usr/bin/env python3

"""
manager.py -- A manager that:
  - Spawns local workers (os.pipe + subprocess).
  - Listens on a TCP socket for multiple clients.
  - For each client, creates a dedicated (input_queue, output_queue).
  - A dedicated sender_thread for each client dispatches tasks to the least busy worker.
  - A single global receiver_thread reads results from all workers and routes them
    to the correct client's output_queue.
  - Each client handler grabs results from its own output_queue and sends them back.
"""

import sys
import os
import socket
import select
import threading
import queue
import pickle
import struct
import subprocess
import ctypes
import signal
import time
import argparse
import errno
import random

from collections import defaultdict
import os.path as osp
import numpy as np
import torch
import fcntl
import types, importlib
################################################################################
# Utilities for reading/writing pickled data with a 4-byte length header
################################################################################
def _alias(dst: str, src: str) -> None:
    try:
        mod = sys.modules.get(src) or importlib.import_module(src)
        sys.modules[dst] = mod
    except Exception:
        pass

def ensure_numpy_pickle_compat() -> None:
    """
    Make NumPy 1.x and 2.x pickle payloads mutually importable by aliasing
    a few internal modules commonly referenced by ndarray reductions.
    Safe to call multiple times; cheap if already set.
    """
    # A) If running on NumPy 1.x, provide numpy._core.* aliases.
    try:
        import numpy._core  # noqa: F401
    except ModuleNotFoundError:
        if hasattr(np, "core"):
            pkg = types.ModuleType("numpy._core")
            pkg.__path__ = []               # mark as package
            sys.modules.setdefault("numpy._core", pkg)
            for name in ("numeric", "numerictypes", "multiarray",
                         "_multiarray_umath", "overrides"):
                _alias(f"numpy._core.{name}", f"numpy.core.{name}")

    # B) If running on NumPy 2.x (no public numpy.core), provide back-compat.
    try:
        import numpy.core  # noqa: F401
    except ModuleNotFoundError:
        try:
            import numpy._core  # noqa: F401
            pkg = types.ModuleType("numpy.core")
            pkg.__path__ = []
            sys.modules.setdefault("numpy.core", pkg)
            for name in ("numeric", "numerictypes", "multiarray",
                         "_multiarray_umath", "overrides"):
                _alias(f"numpy.core.{name}", f"numpy._core.{name}")
        except ModuleNotFoundError:
            # Very old / unusual build; nothing else to do.
            pass
# ensure_numpy_pickle_compat()

def ensure_numpy_core_numeric_compat():
    # NumPy 1.x has numpy.core.numeric; NumPy 2.x moved stuff under numpy._core
    # or vice versa depending on direction. We alias whichever is missing.
    try:
        import numpy._core.numeric  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    # Create numpy._core and numpy._core.numeric aliases that point to real modules.
    if hasattr(np, "core") and hasattr(np.core, "numeric"):
        core_mod = types.ModuleType("numpy._core")
        core_mod.numeric = np.core.numeric
        sys.modules.setdefault("numpy._core", core_mod)
        sys.modules["numpy._core.numeric"] = np.core.numeric


def check_img(var, expected_dtype=[np.uint8]):
    if hasattr(var, 'dtype'):
        assert var.dtype in expected_dtype, \
            f"Expected dtype {expected_dtype}, got {var.dtype} for variable {var}"
    else:
        assert isinstance(var, (*expected_dtype,)), \
            f"Expected type {expected_dtype}, got {type(var)} for variable {var}"

def check_inputdict(input_dict, server_type="world_model"):
    """
    Check if the input_dict contains the expected keys and types.
    """
    if server_type == "world_model":
        necessary_keys = ["b_action", "save_dirs"]
    elif server_type == "sam2":
        necessary_keys = ["bbox_coords", "save_dirs", "pred_frames"]
    elif server_type == "gd_sam2":
        necessary_keys = ["save_dirs"]
    else:
        raise ValueError(f"Unknown server_type: {server_type}. Expected 'world_model' or 'sam2'.")

    assert isinstance(input_dict, dict)
    missing = [k for k in necessary_keys if k not in input_dict]
    if missing:
        raise KeyError(f"Missing required keys: {missing}. Required: {necessary_keys}")

    for k, v in input_dict.items():
        if k in ["b_image", "pred_frames"]:
            check_img(v)
        elif k in ["b_action"]:
            check_img(v, [np.int64, list])
        elif k in ["save_dirs"]:
            assert isinstance(v, list) and all(isinstance(d, str) for d in v), \
                f"save_dirs should be list[str], got {type(v)} with value {v}"
        elif k in ["return_objects"]:
            assert isinstance(v, list) and all(isinstance(d, bool) for d in v), \
                f"return_objects should be list[bool], got {type(v)} with value {v}"



def check_outputdict(output_dict):
    # 1. optional keys:
    pred_frames = output_dict.get("pred_frames", None)
    assert pred_frames is None or (isinstance(pred_frames, np.ndarray) and pred_frames.dtype == np.uint8)
    assert "video_tensors" not in output_dict

    # 2. necessary keys:
    save_dirs = output_dict["save_dirs"]
    assert isinstance(save_dirs, list), \
        f"save_dirs should be list[str], got {type(save_dirs)} with value {save_dirs}"


CHUNK     = 512 * 1024                         # each os.read() ≤ 512 KiB
def _set_nonblocking(fd: int) -> None:
    """Set O_NONBLOCK on an integer file descriptor."""
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if not flags & os.O_NONBLOCK:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

def _read_fully(fd: int, n: int, watchdog_secs: float) -> bytes:
    """Read exactly *n* bytes from *fd* within *watchdog_secs*."""
    buf   = bytearray()
    start = time.time()

    while len(buf) < n:
        left = n - len(buf)
        # Wait for readability, respecting the remaining watchdog budget.
        budget = watchdog_secs - (time.time() - start)
        if budget <= 0:
            raise TimeoutError(f"read_fully timed-out after {watchdog_secs}s")
        r, _, _ = select.select([fd], [], [], budget)
        if not r:                       # select expired: watchdog fires
            raise TimeoutError(f"Worker stalled (> {watchdog_secs}s)")
        chunk = os.read(fd, min(CHUNK, left))
        if not chunk:                   # EOF mid-message
            raise EOFError(f"Expected {n} bytes, got {len(buf)} before EOF")
        buf.extend(chunk)
    return bytes(buf)

def _read_exactly(f, n):
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError(f"Expected {n} bytes, got {len(buf)} before EOF.")
        buf += chunk
    return buf

def read_pickled_data_non_blocking(fd: int, watchdog_secs=300.0):
    """
    Read one length-prefixed Pickle frame from *fd* in non-blocking mode.
    Header = 4-byte big-endian length.
    """
    _set_nonblocking(fd)

    header = _read_fully(fd, 4, watchdog_secs)
    data_len = int.from_bytes(header, 'big')
    if data_len > 10 ** 14:
        raise ValueError(f"Data length {data_len} is too large.")

    payload = _read_fully(fd, data_len, watchdog_secs)
    try:
        return pickle.loads(payload)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e) or "numpy.core" in str(e):
            ensure_numpy_core_numeric_compat()
            return pickle.loads(payload)
        raise

def read_pickled_data(fileobj):
    """
    Blocking version for workers reading tasks from stdin.
    Keeps the original length-prefixed format.
    """
    header = _read_exactly(fileobj, 4)          # blocks until 4 bytes
    data_len = int.from_bytes(header, 'big')
    if data_len > 10 ** 14:
        raise ValueError(f"Data length {data_len} is too large.")
    payload = _read_exactly(fileobj, data_len)  # blocks until full frame
    try:
        return pickle.loads(payload)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e) or "numpy.core" in str(e):
            ensure_numpy_core_numeric_compat()
            return pickle.loads(payload)
        raise


def write_pickled_data(fileobj, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = len(data).to_bytes(4, byteorder='big')
    fileobj.write(header)
    fileobj.write(data)
    fileobj.flush()


def convert_to_python_variable(var):
    """
    Convert a variable to a Python variable.
    if the var is a np.array or tensor object, convert it to a list.
    """
    if isinstance(var, (np.ndarray, torch.Tensor)):
        return var.tolist()
    elif isinstance(var, dict):
        return {k: convert_to_python_variable(v) for k, v in var.items()}
    elif isinstance(var, list):
        return [convert_to_python_variable(v) for v in var]
    else:
        return var


def read_framed(sock):
    """Reads a 4-byte length from sock, then that many bytes, unpickles."""
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise EOFError("Socket closed while reading header.")
        header += chunk
        sock.setblocking(True)

    data_len = struct.unpack(">I", header)[0]
    data = b""
    while len(data) < data_len:
        chunk = sock.recv(data_len - len(data))
        if not chunk:
            raise EOFError("Socket closed mid-message.")
        data += chunk
    try:
        return pickle.loads(data)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e) or "numpy.core" in str(e):
            ensure_numpy_core_numeric_compat()
            return pickle.loads(data)
        raise

def write_framed(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack(">I", len(data))
    while True:
        sock.setblocking(True)
        try:
            sock.sendall(header + data)
        except socket.error as e:
            # if e.errno == errno.EAGAIN:
            #     time.sleep(0.02)  # wait a bit, then try again
            #     continue
            # else:
            raise e
        break


################################################################################
# Worker creation
################################################################################

def set_death_signal():
    """(Optional) on Linux: if manager dies, worker is killed."""
    try:
        libc = ctypes.CDLL("libc.so.6")
    except Exception as e:
        raise ValueError(f"Failed to load libc: {e}")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)

def create_workers(worker_cmds, num_workers):
    """
    Spawns 'num_workers' local worker processes with distinct pipes.
    worker_cmds can be a single list (if all workers share same cmd),
    or a dict{wid: list} if each worker has distinct commands.

    Returns: a list of worker dicts, each containing:
      - id: numeric ID
      - proc: the subprocess.Popen object
      - r_fd: read pipe FD
      - w_file: a file object for writing to that worker's stdin
      - pending: how many tasks are inflight
      - lock: a Lock to protect 'pending'
    """
    workers = []
    for wid in range(num_workers):
        if isinstance(worker_cmds, dict):
            cmd = worker_cmds[wid]
        else:
            cmd = worker_cmds[:]

        r_fd, w_fd = os.pipe()
        cmd.append(str(w_fd))  # pass FD as last arg
        proc = subprocess.Popen(
            " ".join(cmd),
            stdin=subprocess.PIPE,
            stdout=None,
            bufsize=4096,
            pass_fds=[w_fd],
            shell=True,
            # preexec_fn=set_death_signal,  # optional
        )
        time.sleep(2)
        w_file = os.fdopen(proc.stdin.fileno(), 'wb', buffering=2048*1024)
        workers.append({
            'id': wid,
            'proc': proc,
            'r_fd': r_fd,
            'w_file': w_file,
            'pending': 0,
            'lock': threading.Lock()
        })
        print(f"[Manager] Spawned worker {wid} with PID={proc.pid}")
    return workers


################################################################################
# A single global "receiver_thread" for all workers
################################################################################
def receiver_thread(client_out_queues, workers, worker_type):
    """
    1) We use select() on all workers' r_fd.
    2) For each message from a worker, read (client_id, task_id, result).
    3) Decrement that worker's 'pending', then do:
         manager.client_out_queues[client_id].put((task_id, result))
    This means each client has an output_queue to store its results.
    """
    fd_to_worker = {w['r_fd']: w for w in workers}
    fds = list(fd_to_worker.keys())

    while fds:
        readable, _, _ = select.select(fds, [], [], 1)  # set timeout=1, if needs receiver_thread to do other things
        for fd in readable:
            w = fd_to_worker[fd]
            try:
                client_id, task_id, result = read_pickled_data_non_blocking(fd)
            except (EOFError, TimeoutError) as e:
                print(f"[{worker_type}_Manager] Worker {w['id']} "
                      f"(PID={w['proc'].pid}) closed r_fd with error: {e}")
                fds.remove(fd)
                os.close(fd)
                continue
            except Exception as e:
                print(f"[{worker_type}_Manager] Worker {w['id']} read error: {e}")
                fds.remove(fd)
                os.close(fd)
                continue

            # bookkeeping as before
            with w['lock']:
                w['pending'] = max(0, w['pending'] - 1)

            # Route result to the correct client's output queue
            out_q = client_out_queues[client_id]
            out_q.put((task_id, result))
            print(f"[{worker_type}_Manager] out_q put (client_id={client_id}, "
                  f"task_id={task_id}) from worker {w['id']} (PID={w['proc'].pid}).")

def receiver_for_worker(stdin_fd: int,
             task_queue: "queue.Queue[tuple[int,int,dict]]",
             stop_evt: threading.Event,
             max_pending: int = 200) -> None:
    """
    Continuously pulls framed Pickle messages from stdin (non-blocking)
    and pushes them into *task_queue*.  If pending tasks exceed *max_pending*
    we raise an error, write it to manager, and exit the worker.
    """
    while not stop_evt.is_set():
        try:
            item = read_pickled_data_non_blocking(stdin_fd)
        except TimeoutError:
            print("[worker] Waiting for input from stdin, retrying...")
            continue  # No data yet – try again
        except EOFError:
            break     # Manager closed pipe – graceful shutdown
        except Exception as e:
            # Unrecoverable read error – surface to main thread
            task_queue.put(("ERROR", -1, {"read_error": str(e)}))
            break

        if item == "DONE":
            stop_evt.set()
            break

        task_queue.put(item)         # (client_id, task_id, payload)
        pending = task_queue.qsize()

        if pending > max_pending:
            # Queue overflow → terminate itself
            print(f"[worker] Pending tasks exceeded {max_pending}, exiting.")
            os.kill(os.getpid(), signal.SIGTERM)


################################################################################
# The ClientHandler (one per TCP client)
################################################################################

class Batcher:
    """
    A simple dispatcher that sends tasks to workers with the minimal batch size.
    """
    def __init__(self, batch_size, worker_type, addr, client_id):
        self.worker_type = worker_type
        self.bs = batch_size
        self.addr = addr
        self.client_id = client_id
        self.batch_id = 0
        self.init_task_id = 0
        self.batchID_to_taskID = {}  #the remaining unreceived tasks in each batch
        self.taskID_to_batchID = {}

        self.batchID_to_results = {}   # the results of each batch
        self.batchID_to_output = {}  # the final composed output of each batch
        self.prev_time = time.time()

    def split_batch(self, tasks, init_task_id):
        """Split a batch of tasks into individual tasks to enable better parallelism."""
        assert isinstance(tasks, dict), f"Tasks must be a dict, but got {type(tasks)}"
        data_len = len(next(iter(tasks.values())))

        assert self.init_task_id == init_task_id, f"Task ID mismatch: {self.init_task_id} vs {init_task_id}"

        # Create sub-batches of size 'self.batch_size' and enqueue them.
        composed_batchs = []
        for start in range(0, data_len, self.bs):
            batch = {k: v[start:start + self.bs] for k, v in tasks.items()}
            composed_batchs.append((init_task_id, batch))

            self.batchID_to_taskID.setdefault(self.batch_id, []).append(init_task_id)
            assert init_task_id not in self.taskID_to_batchID, f"Task ID {init_task_id} already exists."
            self.taskID_to_batchID[init_task_id] = self.batch_id
            init_task_id += 1
        self.batch_id += 1
        self.init_task_id = init_task_id
        print(f"[{self.worker_type}_Manager] Split batch (len={data_len}) into {len(composed_batchs)} sub-batches.")
        assert data_len > 0, f"Data length must be > 0, but got {data_len}"
        return composed_batchs, init_task_id

    def _recompose_batch(self, batch_id):
        """Recompose the results of a batch of tasks, if all tasks in this batch are done."""
        #sort the results by task_id, default is ascending
        results = sorted(self.batchID_to_results[batch_id], key=lambda x: x[0])
        # reorganize the results into a output_dict
        output_dict = {}
        for t_id, item in results:
            for k, v in item.items():
                output_dict.setdefault(k, []).extend(v)
        del self.batchID_to_results[batch_id]       # means this batch is done
        return output_dict

    def get(self):
        """Return final output of a batch, if any."""
        min_processing_batch_id = min(list(self.batchID_to_taskID.keys()), default=float('inf'))
        min_finished_batch_id = min(list(self.batchID_to_output.keys()), default=float('inf'))
        if min_processing_batch_id > min_finished_batch_id:
            output_dict = self.batchID_to_output.pop(min_finished_batch_id)
            return output_dict
        else:
            return None

    def monitor_by_time(self, timeout=600):
        """Monitor the time interval of "success get()" by time (300s), if exceed then print the debug message."""
        curr_time = time.time()
        if (curr_time - self.prev_time) > timeout:
            print(f"[{self.worker_type}_Manager] No output of Client {self.addr} (client_id={self.client_id}) for {timeout} seconds: \n"
                  f"self.batchID_to_taskID={self.batchID_to_taskID}, \n"
                  f"self.taskID_to_batchID={self.taskID_to_batchID}, \n"
                  f"self.batchID_to_results={str(self.batchID_to_results)[:200]}, \n"
                  f"self.batchID_to_output={str(self.batchID_to_output)[:200]}")
            self.prev_time = curr_time

    def get_unfinished_task_ids(self):
        """Return the task_ids of unfinished tasks."""
        return set(self.taskID_to_batchID.keys())

    def put_new_result(self, task_id, result):
        """Put a new result into the dispatcher."""
        batch_id = self.taskID_to_batchID.pop(task_id)
        self.batchID_to_taskID[batch_id].remove(task_id)
        self.batchID_to_results.setdefault(batch_id, []).append((task_id, result))

        if len(self.batchID_to_taskID[batch_id]) == 0:
            output_dict = self._recompose_batch(batch_id)
            self.batchID_to_output[batch_id] = output_dict
            self.batchID_to_taskID.pop(batch_id)


class ClientHandler(threading.Thread):
    """
    Handles a single client connection:
    - Reads tasks from socket immediately dispatching them to workers.
    - Keeps track of dispatched task_ids and collects results from its output_queue.
    """
    def __init__(self, conn, addr, client_id, manager, batch_size, worker_type):
        super().__init__(daemon=True)
        self.worker_type = worker_type
        self.conn = conn
        self.addr = addr
        self.client_id = client_id
        self.manager = manager
        self.output_queue = queue.Queue()
        self.manager.client_out_queues[client_id] = self.output_queue
        if batch_size > 0:
            self.batcher = Batcher(batch_size, worker_type, addr, client_id)
        self.next_task_id = 0  # task id generator for this client
        self.unfinished_tasks = set()
        self.running = True

    def run(self):
        print(f"[{self.worker_type}_Manager] Client {self.addr} connected with client_id={self.client_id}")
        self.conn.setblocking(False)
        try:
            while self.running:
                self._read_client_requests()
                self._send_ready_results()
                time.sleep(0.05)  # short delay to avoid busy loop
        except Exception as e:
            print(f"[{self.worker_type}_Manager] Client {self.addr} error: {e}")
        finally:
            self.conn.close()
            print(f"[{self.worker_type}_Manager] Client {self.addr} connection closed.")

    def _dispatch_task(self, task_id, payload):
        """Pick worker with fewest pending tasks and send (client_id, task_id, payload)."""
        # # 1. Find the minimum pending task count among workers.
        # min_pending = min(worker['pending'] for worker in self.manager.workers)
        # candidate_workers = [w for w in self.manager.workers if w['pending'] == min_pending]
        # # Randomly select one worker from the candidates.
        # best_worker = random.choice(candidate_workers)
        #2. Find the worker with the least pending tasks.
        best_worker = min(self.manager.workers, key=lambda w: w['pending'])

        with best_worker['lock']:
            best_worker['pending'] += 1
            write_pickled_data(best_worker['w_file'], (self.client_id, task_id, payload))
        print(f"[{self.worker_type}_Manager] Dispatched (client={self.client_id}, "
              f"task={task_id}) to worker {best_worker['id']} (PID={best_worker['proc'].pid}).")
        time.sleep(0.06)  # short delay to avoid busy loop

    def _read_client_requests(self):
        """Non-blocking read from self.conn and immediately dispatch tasks."""
        while True:
            try:
                msg = read_framed(self.conn)
                self.conn.setblocking(False)
                print("[worker] Finished Read msg from connection")
            except socket.error as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break  # no more data
                else:
                    self.running = False
                    break
            except EOFError:
                self.running = False
                break

            if msg == "DONE":
                print(f"[{self.worker_type}_Manager] {self.addr} => DONE. Closing client.")
                self.running = False
                break

            if hasattr(self, 'batcher'):
                composed_batchs, self.next_task_id = self.batcher.split_batch(
                    msg, self.next_task_id
                )
                self.unfinished_tasks = self.batcher.get_unfinished_task_ids()
                # dispatch tasks:
                for task_id, task in composed_batchs:
                    self._dispatch_task(task_id, task)
            else:
                self._dispatch_task(self.next_task_id, msg)
                self.unfinished_tasks.add(self.next_task_id)
                self.next_task_id += 1

    def _send_ready_results(self):
        """Send completed results to the socket."""
        while True:
            try:
                task_id, result = self.output_queue.get_nowait()
            except queue.Empty:
                self.batcher.monitor_by_time()
                break
            if task_id in self.unfinished_tasks:
                self.unfinished_tasks.remove(task_id)
                try:
                    if hasattr(self, 'batcher'):
                        self.batcher.put_new_result(task_id, result)
                        output = self.batcher.get()
                        if output is not None:
                            write_framed(self.conn, output)
                    else:
                        write_framed(self.conn, result)
                    self.conn.setblocking(False)
                except socket.error as e:
                    print(f"[{self.worker_type}_Manager] Error sending result to {self.addr}: {e}")
                    # print("to be transfered Result is:\n", result)
                    self.running = False
                    raise ValueError("Error encountered in _send_ready_results()")
            else:
                RuntimeError(f"Task {task_id} not found in unfinished_tasks.")

################################################################################
# The Manager
################################################################################
class ManagerState:
    def __init__(self, workers):
        self.workers = workers
        # A global dictionary: client_id -> that client's output_queue
        self.client_out_queues = {}


def run_manager_server(host, port, manager, batch_size, worker_type):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(100)
        print(f"[{worker_type}_Manager] Listening on {host}:{port}")

        next_client_id = 0
        while True:
            conn, addr = srv.accept()
            ch = ClientHandler(conn, addr, next_client_id, manager, batch_size, worker_type)
            next_client_id += 1
            ch.start()


# ======== Worker's Main Function (a wrapper)========
def worker_main(pipe_fd: int, task_fn) -> None:
    """Process loop executed by one worker thread or process.
    Parameters
    ----------
    pipe_fd : int
        File descriptor opened by the parent (write-end of the pipe).
    task_fn : Callable[[dict], dict]
        Pure function that turns an input payload into an output payload.
    """
    import sys
    pipe_file = os.fdopen(pipe_fd, "wb", buffering=2048 * 1024)
    stdin_fd  = sys.stdin.fileno()

    task_q: queue.Queue[tuple[int, int, dict]] = queue.Queue()
    stop_evt = threading.Event()

    threading.Thread(
        target=receiver_for_worker,
        args=(stdin_fd, task_q, stop_evt),
        daemon=True,
    ).start()

    while not stop_evt.is_set():
        try:
            client_id, task_id, payload = task_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if client_id == "ERROR" or payload == "DONE":
            print(f"[worker] Error received from receiver thread: {task_id}, ERROR: {payload}")
            break

        print(f"[worker] Processing task (client={client_id}, "
              f"task={task_id}).  Remaining queue size: {task_q.qsize()}")

        result = task_fn(payload)
        check_outputdict(result)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

        write_pickled_data(pipe_file, (client_id, task_id, result))

    pipe_file.close()
    stop_evt.set()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="06.15_use_igenex")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The *minimal* batch size for each worker, when enable auto-batching/task split, if batch_size==-1, no auto-batching.")
    parser.add_argument("--worker_type", type=str, default="sam2",
                        help="Type of worker to spawn: e.g. 'igenex' or 'sam2'.")
    # Capture extra flags not explicitly defined above.
    args, unknown = parser.parse_known_args()
    extra_dict = parse_extra_cli(unknown)

    # Base forwarded args always sent to workers; extras override if same key.
    forward_args = {"log_dir": args.log_dir, "exp_id": args.exp_id, **extra_dict}

    log_path = osp.join(args.log_dir, args.exp_id, "manager", f"{args.worker_type}_manager.log")
    setup_logger(log_path)
    print(f"[Manager] All Args:\n {args}")
    if unknown:
        print(f"[Manager] Extra forwarded CLI tokens: {unknown}")
        print(f"[Manager] Parsed extra args dict: {extra_dict}")
    print(f"[Manager] manager pid is {os.getpid()}")

    # 1) Spawn local workers
    if args.worker_type == "igenex":
        worker_cmds = get_genex_workers_cmd(
            num_workers=args.num_workers,
            add_args=forward_args,
        )
    if args.worker_type == "igenex_manip":
        worker_cmds = get_genex_workers_manip_cmd(
            num_workers=args.num_workers,
            add_args=forward_args,
        )
    elif args.worker_type == "sam2":
        worker_cmds = get_sam2_workers_cmd(
            num_workers=args.num_workers,
            add_args=forward_args,
        )
    elif args.worker_type == "gd_sam2":
        worker_cmds = get_gd_sam2_workers_cmd(
            num_workers=args.num_workers,
            add_args=forward_args,
        )
    elif args.worker_type in ["se3ds", "pathdreamer", "nwm", "hunyuan", "ltx", "wan21", "wan22", "cosmos", "svd", "gen4tur"] + \
                             ["FTcosmos", "FTltx", "FTwan21", "FTwan22", "FTwan22-14B"]:
        worker_cmds = get_worldmodel_workers_cmd(
            num_workers=args.num_workers,
            add_args=forward_args,
            model_type=f"{args.worker_type}_worker",
        )
    workers = create_workers(worker_cmds, args.num_workers)

    # 2) Construct manager state
    manager = ManagerState(workers)

    # 3) Start the global receiver_thread
    r_thread = threading.Thread(
        target=receiver_thread,
        args=(manager.client_out_queues, manager.workers, args.worker_type),
        daemon=True,
    )
    r_thread.start()

    # 4) Start the manager server loop
    run_manager_server(args.host, args.port, manager, args.batch_size, args.worker_type)


if __name__ == "__main__":
    from utils.parser_additions import parse_extra_cli
    from utils.logger import setup_logger
    from downstream.utils.workers_cfg import (
        get_genex_workers_cmd,
        get_sam2_workers_cmd,
        get_gd_sam2_workers_cmd,
        get_worldmodel_workers_cmd,
        get_genex_workers_manip_cmd,
    )
    main()
