from glob import glob
import subprocess
from tqdm import tqdm

obj_object_paths = glob("object_data/*.obj")
stl_object_paths = glob("object_data/*.stl")
paths = obj_object_paths + stl_object_paths
progress = tqdm(paths, total=len(paths), leave=False, position=0)
for path in progress:
    command = f"blender --background --python blender_scripts/dataset_rendering.py -- --object_path '{path}' --num_renders 32 --output_dir 'temp_data' --engine CYCLES"
    result = subprocess.run(
            ["bash", "-c", command],
            timeout=3000,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(result.stderr.decode())