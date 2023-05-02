import psutil
import os
import signal

# Get the process IDs of the running Python processes
python_processes = [p for p in psutil.process_iter() if 'python' in p.name()]

# Filter processes based on command line
script_name = 'analysis/compute_dist.py'
script_processes = [
    p for p in python_processes if script_name in ' '.join(p.cmdline())]

# Send a signal to terminate each process
for proc in script_processes:
    os.kill(proc.pid, signal.SIGTERM)
