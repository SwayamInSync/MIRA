# Workflow
### Loading and processing input data
```bash
git clone https://github.com/allenai/objaverse-xl.git && cd objaverse-xl/scripts/rendering
pip install objaverse --upgrade
```
### Training Dataset rendering
```bash
apt-get install -y xvfb
```
```bash
wget https://ftp.nluug.nl/pub/graphics/blender//release/Blender4.0/blender-4.0.2-linux-x64.tar.xz && tar -xf blender-4.0.2-linux-x64.tar.xz && rm blender-4.0.2-linux-x64.tar.xz
```

```bash
apt-get install libxrender1
apt-get install libxi6 libgconf-2-4
sudo apt-get install libxkbcommon-x11-0
```
For Linux
```bash
DISPLAY=:0.0 && xvfb-run --auto-servernum blender --background --python blender_scripts/dataset_rendering.py -- --object_path 'temp_data/impeller.obj' --num_renders 5 --output_dir 'temp_data/render' --engine CYCLES
```
For Mac
```bash
blender --background --python blender_scripts/dataset_rendering.py -- --object_path 'temp_data/impeller.obj' --num_renders 5 --output_dir 'temp_data/render' --engine CYCLES
```
 
# Server setup
```bash
apt-get update && apt-get install -y libgl1-mesa-glx
```
Getting IP of server
```bash
hostname -i
```
DDP running script
```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_ddp.py
```
