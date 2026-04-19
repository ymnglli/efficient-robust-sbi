import time
from datetime import datetime as dt
import numpy as np
import os

class Timer:
    def __init__(self, name, root="."):
        self.name = name
        self.root = root
        self.laps = np.array([])
    
    def start(self):
        print(f"Starting timer for {self.name}")
        self.start_time_ns = time.time_ns()
    
    def lap(self, verbose=True):
        if (self.laps.shape[0] > 0):
            lap_time_ns = time.time_ns() - (self.laps.sum() + self.start_time_ns)
        else:
            lap_time_ns = time.time_ns() - self.start_time_ns
        laps = [t for t in self.laps]
        laps.append(lap_time_ns)
        self.laps = np.array(laps)
        if (verbose):
            log = f"[{self.name}] Lap time: {lap_time_ns / 1e9:.3f}s"
            print(log)
    
    def stop(self, save=True, verbose=True):
        total_time_s = (time.time_ns() - self.start_time_ns) / 1e9
        mean_time_s = np.mean(np.array(self.laps)) / 1e9
        log = [f"--- Timing: {self.name} ---\n",
              f"Elapsed time: {total_time_s:.3f}s\n",
              f"Timer ran for {len(self.laps)} laps\n",
              f"Average time per epoch: {mean_time_s:.3f}s\n"]

        if (verbose):
            print("".join(log))
        if (save):
            if not os.path.exists(self.root):
                os.makedirs(self.root)
            now = dt.now().strftime("%Y%m%d_%H.%M.%S")
            file = f"{now}.log"
            filepath = os.path.join(self.root, file)
            with(open(filepath, "w")) as f:
                f.writelines(log)
                f.writelines([f"{lap / 1e9:.4f}s, " for lap in self.laps])

