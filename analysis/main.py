import subprocess
import itertools
import time

eval_experiences = list(range(0, 10))
splits = ["test", "train"]
scripts = ["collect_activations.py", "compute_dist.py"]
alphas = [0, 0.5, 1]

# eval_experiences = list(range(1))
# splits = ["train"]
# scripts = ["compute_dist.py"]
# alphas = [0.5]

python_files_with_args = [
    (script, ['--experience', str(exp), '--split', split] +
     (['--alpha', str(alpha)] if script == 'compute_dist.py' else []))
    for script in scripts
    for exp in eval_experiences
    for split in splits
    for alpha in (alphas if script == 'compute_dist.py' else [None])
]

# print(python_files_with_args)

for script, args in python_files_with_args:
    print(f"Executing {script} with arguments: {args}...")
    command = ["python", f"analysis/{script}"] + args
    subprocess.run(command, check=True)
    print(f"{script} finished\n")

    # Run killer
    command = ["python", f"analysis/kill.py"]
    subprocess.run(command, check=True)

    # Sleep
    time.sleep(10)
print("All files executed.")
