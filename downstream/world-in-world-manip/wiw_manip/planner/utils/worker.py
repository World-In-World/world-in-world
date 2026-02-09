import subprocess
import sys
import queue
import threading
import pickle
import time
import select
import ctypes
import signal
import os
import math
from collections import defaultdict

##############################
# Helper: Sender thread
##############################
def write_pickled_data(pipe_file, order, output):
    data = pickle.dumps((order, output), protocol=pickle.HIGHEST_PROTOCOL)
    # Write a 4-byte big-endian length header before the data
    pipe_file.write(len(data).to_bytes(4, byteorder='big'))
    pipe_file.write(data)
    pipe_file.flush()

def sender_thread(task_queue, workers_info):
    """
    Send tasks concurrently to workers in round-robin fashion.
    Each task is a tuple: (order, task_data).
    After sending all tasks, send a "DONE" sentinel to each worker.
    """
    worker_index = 0
    order = 0
    num_workers = len(workers_info)

    while True:
        item = task_queue.get()  # block until a new item is available
        if item == "DONE":
            # We have determined that the queue is finished.
            # Send "DONE" to ALL workers so they can exit.
            for (proc, r_fd, w_fd) in workers_info:
                pickle.dump("DONE", proc.stdin)
                proc.stdin.flush()
            break

        # Otherwise, assign this item to exactly one worker (round-robin):
        (proc, r_fd, w_fd) = workers_info[worker_index]
        pickle.dump((order, item), proc.stdin)
        proc.stdin.flush()

        worker_index = (worker_index + 1) % num_workers
        order += 1
        time.sleep(1)  # optional

    # Once we're done, close all worker pipes
    for i, (proc, r_fd, w_fd) in enumerate(workers_info):
        proc.stdin.close()
        print(f"[BridgingThread] Closed Worker{i}'s stdin.")

    print("[BridgingThread] Finished sending tasks and sent DONE to all workers.")


##############################
# Helper: Single output manager thread
##############################
def output_manager(workers_info, results_queue):
    """
    Monitors all worker pipes concurrently using select,
    reads each pickled (order, output) result,
    and releases them into results_queue in correct order.
    """
    # Build a mapping from read-fd -> (proc, fileobj)
    fd_to_file = {}
    fds = []
    for (proc, r_fd, w_fd) in workers_info:
        f = os.fdopen(r_fd, 'rb', buffering=2048*1024)
        fd_to_file[r_fd] = (proc, f)
        fds.append(r_fd)

    pending_results = {}
    next_expected = 0

    # We'll keep looping until no fds remain open
    while fds:
        readable, _, _ = select.select(fds, [], [], 1.0)
        for fd in readable:
            (proc, fileobj) = fd_to_file[fd]

            try:
                result_tuple = read_pickled_data(fileobj)
            except (EOFError, ValueError) as e:
                # Worker closed its pipe; remove from the set
                fds.remove(fd)
                fileobj.close()
                print(f"[OutputManager] Worker {proc.pid} has error: {e}")
                print(f"[OutputManager] Worker {proc.pid} has closed/corrupted pipe.")
                continue

            # Expecting (order, result)
            order, result = result_tuple
            pending_results[order] = result

            # Release results in correct order
            while next_expected in pending_results:
                results_queue.put(pending_results[next_expected])
                del pending_results[next_expected]
                next_expected += 1

    print("[OutputManager] Finished collecting outputs.")

def read_exactly(fileobj, n):
    """Read exactly n bytes from fileobj, buffering as needed."""
    buf = b""
    while len(buf) < n:
        chunk = fileobj.read(n - len(buf))
        if not chunk:
            raise EOFError(f"Expected {n} bytes, got only {len(buf)} bytes before EOF")
        buf += chunk
    return buf

def read_pickled_data(fileobj):
    # Read the 4-byte header that contains the length
    header = read_exactly(fileobj, 4)
    data_len = int.from_bytes(header, byteorder='big')
    # Sanity check: if data_len is unreasonably large, skip until we find a valid header
    if data_len > 10**13:  # for example, 1MB threshold
        # Optionally, clear the buffer or raise an error
        raise ValueError(f"Data length {data_len} is too high, indicating header corruption")
    data = read_exactly(fileobj, data_len)
    result_tuple = pickle.loads(data)
    return result_tuple


##############################
# Worker Creation and Cleanup
##############################
def set_death_signal():
    """
    Set the parent-death signal of the calling process to SIGTERM.
    When the parent dies, the child will be sent SIGTERM.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
    except Exception as e:
        # If libc can't be loaded, simply return (this will not set the death signal)
        raise ValueError(f"Failed to load libc: {e}")
        return
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def create_workers(worker_args, num_workers=2):
    """
    Creates worker processes and returns a list of tuples:
      [(proc, r_fd, w_fd), (proc, r_fd, w_fd), ...]
    Each worker is launched with extra command-line arguments (if provided)
    passed via the worker_args parameter.
    """
    workers_info = []
    for wid in range(num_workers):
        if isinstance(worker_args, dict):
            assert len(worker_args) == num_workers, "Dictionary must have one entry per worker"
            worker_arg = worker_args[wid]
        else:
            worker_arg = worker_args.copy()
    
        # Create a dedicated pipe
        r_fd, w_fd = os.pipe()
        # Base argument list: interpreter, worker script, and the FD
        # worker_args[0],      #e.g., "/data/jieneng/software/anaconda/envs/sam2/bin/python", 
        # worker_args[1],      #e.g., "/home/jchen293/igenex_code/downstream/sam2_model.py",

        worker_arg.append(str(w_fd))  # Pass the write-end FD as a string in the end

        proc = subprocess.Popen(
            worker_arg,
            stdin=subprocess.PIPE,
            stdout=None,         # Normal stdout is not captured
            stderr=None,         # Or capture in separate pipe if you want
            bufsize=0,
            pass_fds=[w_fd],     # Do not close w_fd in the child
            # preexec_fn=set_death_signal,   # Optional
        )
        workers_info.append((proc, r_fd, w_fd))
        print(f"[Main] Init subprocess WorkerID <{wid}> with PID: <{proc.pid}>")

    return workers_info


def close_all(all_processes):
    for t in all_processes:
        if isinstance(t, threading.Thread):
            t.join()
        elif isinstance(t, subprocess.Popen):
            t.terminate()   # Request termination (if the process handles SIGTERM)
            t.wait()        # Wait until it has terminated
        else:
            raise ValueError(f"Unknown thread type: {t}")
    print("[Main] All threads have exited.")


##############################
# Main Runner Function
##############################
class BatchedQueue:
    """
    A simple queue mechanism supporting batched input/output.
    - Items are dictionaries, where each value is a list (e.g., 'b_image': [...], 'b_action': [...]).
    - The first dimension of each list is treated as the 'batch dimension'.
    - Allows splitting large input dicts into smaller batches via 'put', and then merging
      them back into larger chunks (exact 'data_len' items) via 'get'.
    Attributes:
        maxsize (int) : Maximum size for both input_queue and output_queue.
        num_workers (int) : Number of workers used for splitting batches if batch_size is not provided.
        batch_size (int) : Fixed batch size to split inputs. If None, it's auto-derived from data length.
        input_queue (queue.Queue)  : Stores input batches.
        output_queue (queue.Queue) : Stores output batches.
        cache_get (dict) : Temporary storage for leftover items after partial retrieval from a batch.
    """
    def __init__(self, maxsize, num_workers, batch_size=None):
        self.maxsize = maxsize
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.input_queue = queue.Queue(maxsize=maxsize)
        self.output_queue = queue.Queue(maxsize=maxsize)
        self.cache_get = {}  # leftover items from partially consumed batches

    def put(self, input_dict):
        """
        Splits 'input_dict' into smaller batches and puts them into input_queue.

        Assumes all lists in 'input_dict' have the same length and represent the batch dimension.
        """
        # Determine how many items (N) are in the first key's list.
        data_len = len(next(iter(input_dict.values())))
        if self.batch_size is None:
            # If no batch_size was provided, compute an upper bound based on num_workers.
            self.batch_size = math.ceil(data_len / self.num_workers)

        # Create sub-batches of size 'self.batch_size' and enqueue them.
        for start in range(0, data_len, self.batch_size):
            batch = {k: v[start:start + self.batch_size] for k, v in input_dict.items()}
            self.input_queue.put(batch)

    def get(self, data_len):
        """
        Retrieves exactly 'data_len' items (per key) from output_queue by merging batches.
        If 'data_len' == 'all_available', returns ALL items from output_queue (concatenated).

        If the last retrieved batch is larger than needed, leftover items go into self.cache_get
        for future retrieval.

        Returns:
            A dict with exactly 'data_len' items per key, except for the 'all_available' case,
            where all queued items are returned.
        """
        # 1) 'all_available': simply drain the output_queue
        if data_len == 'all_available':
            all_data = defaultdict(list)
            while not self.output_queue.empty():
                batch = self.output_queue.get()
                for k, v in batch.items():
                    all_data[k].extend(v)
            return dict(all_data)

        # 2) Start building result from cached items (if any)
        result = {}
        current_len = 0
        if self.cache_get:
            # Copy from cache
            result = {k: list(v) for k, v in self.cache_get.items()}
            current_len = len(next(iter(result.values()))) if result else 0
            # If cache alone already exceeds data_len, split it
            if current_len > data_len:
                for k, v in result.items():
                    self.cache_get[k] = v[data_len:]
                    result[k] = v[:data_len]
                return result
            # Otherwise, we've used all cached items
            self.cache_get.clear()

        # 3) Consume from output_queue until we reach 'data_len'
        while current_len < data_len:
            batch = self.output_queue.get()  # blocks until a batch is available
            batch_len = len(next(iter(batch.values())))
            needed = data_len - current_len
            if batch_len > needed:
                # Only take what we need; leftover goes back to cache
                for k, v in batch.items():
                    result.setdefault(k, []).extend(v[:needed])
                    self.cache_get.setdefault(k, []).extend(v[needed:])
                current_len += needed
            else:
                # Consume entire batch
                for k, v in batch.items():
                    result.setdefault(k, []).extend(v)
                current_len += batch_len

        # Should have exactly data_len items now
        assert current_len == data_len, f"Expected {data_len}, got {current_len}."
        return result

    def put_and_get(self, input_dict):
        """
        Convenience method that puts 'input_dict' into input_queue in batches,
        then retrieves exactly the same number of items from output_queue.
        """
        self.put(input_dict)
        data_len = len(next(iter(input_dict.values())))
        return self.get(data_len)

    def qsize(self):
        """
        Returns a tuple (input_qsize, output_qsize) for quick inspection.
        """
        return self.input_queue.qsize(), self.output_queue.qsize()


def init_workers(worker_args, num_workers=2, bs=None):
    queue = BatchedQueue(maxsize=20, num_workers=num_workers, batch_size=bs)

    # 1) Create the worker processes (with dedicated pipes)
    workers_info = create_workers(worker_args, num_workers)

    # 2) Start the sender thread
    sender = threading.Thread(
        target=sender_thread,
        args=(queue.input_queue, workers_info),
        daemon=True
    )
    sender.start()

    # 3) Start the output manager thread
    output_thread = threading.Thread(
        target=output_manager,
        args=(workers_info, queue.output_queue),
        daemon=True
    )
    output_thread.start()

    # Return references so we can join or terminate later
    all_processes = [sender, output_thread] + [info[0] for info in workers_info]
    return all_processes, queue



##############################
# example usage:
##############################
if __name__ == "__main__":
    # Create a task queue and a results queue
    task_queue = queue.Queue()
    results_queue = queue.Queue()

    # Initialize workers and threads
    all_processes = init_workers(task_queue, results_queue, num_workers=2)

    # Add tasks to the task queue
    for i in range(10):
        task_queue.put(f"Task {i}")

    # Signal that we're done adding tasks
    task_queue.put("DONE")

    # Wait for all tasks to complete
    while not results_queue.empty():
        result = results_queue.get()
        print(f"Received result: {result}")

    # Clean up all processes
    close_all(all_processes)
