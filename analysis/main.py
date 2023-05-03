import subprocess
import itertools
import time
import os
import glob
from utils import DataStreamEnum

eval_experiences = list(range(0, 10))
splits = ["test", "train"]
scripts = ["collect_activations.py", "compute_dist.py"]
alphas = [0, 0.5, 1]

# eval_experiences = list(range(2))
# splits = ["train"]
# scripts = ["collect_activations.py"]
# alphas = [0.5]

python_files_with_args = [
    (script, ['--experience', str(exp), '--split', split] +
     (['--alpha', str(alpha)] if script == 'compute_dist.py' else []))
    for exp in eval_experiences
    for split in splits
    for script in scripts
    for alpha in (alphas if script == 'compute_dist.py' else [None])
]

# for item in python_files_with_args:
#     print(item)

for script, args in python_files_with_args:

    # Remove activations
    if (script == "collect_activations.py"):
        path = DataStreamEnum[args[3]].value
        pattern = "*act_on*"
        print(path)
        for file_path in glob.glob(os.path.join(path, pattern)):
            os.remove(file_path)

    print(f"Executing {script} with arguments: {args}...")
    command = ["python", f"analysis/{script}"] + args
    subprocess.run(command, check=True)
    print(f"{script} finished\n")

    # Run killer after compute dist
    if (script == "compute_dist.py"):
        command = ["python", f"analysis/kill.py"]
        subprocess.run(command, check=True)

    # Sleep
    time.sleep(10)
print("All files executed.")
