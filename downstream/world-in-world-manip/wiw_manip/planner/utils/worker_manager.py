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
import pickle
import struct
import time
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