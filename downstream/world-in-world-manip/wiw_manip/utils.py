import pandas as pd
from typing import List
import pickle
import struct
import socket
from PIL import Image
from torchvision.transforms import ToTensor
import os

class State:
    """
    Class to store the state trajectory of an agent.
    It wraps a pandas DataFrame for state rows (e.g. position, rotation, etc.),
    provides lists for actions/answers, and tracks the best recognized answer.
    """

    def __init__(self, obs=None):
        self.origin_obs = ToTensor()(Image.open(obs).convert("RGB"))
        self.after_obs_list = []    # current
        self.action_list = [] # current
        self.reason_list = []
        self.save_dir = None
    
    def set_save_dir(self, save_dir: str):
        self.save_dir = save_dir
    
    def add_action(self, action, reason):
        self.action_list.append(action)
        self.reason_list.append(reason)
        
    def read_obs(self):
        if self.save_dir is not None:
            obs_files = sorted([
                f for f in os.listdir(self.save_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.after_obs_list = [
                ToTensor()(Image.open(os.path.join(self.save_dir, f)).convert("RGB"))
                for f in obs_files
            ]
        else:
            self.after_obs_list = []

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
    return pickle.loads(data)

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