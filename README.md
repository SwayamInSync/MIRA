# MIRA - Multimodal Image Reconstruction with Attention

**MIRA** is a multimodal transformer (Encoder-Decoder) based architecture for Text or Image to 3D reconstruction focussing on generating the 3D representation just using single 2D image of object within seconds. Text pipeline utilizes the stable diffusion methods to generate image from prompt and passing to model after necessary preprocessing.

The architecture uses a pre-trained **DINO-V2** as the image encoder and a **custom triplane decoder**. The decoder learns to project image features on triplane via cross-attention and model the relations among the spatially-structured triplane tokens via self-attention, camera features are modulated within the decoder.

It is highly efficient and adaptable, capable of handling a wide range of multi-view image datasets. It’s trained by minimizing the difference between the rendered images and ground truth images at novel views, without the need for excessive 3D-aware regularization or delicate hyper-parameter tuning.

> Due to limited resources, I wasn't able to perform a robust training so attached samples are from the limited trained checkpoint (which is useless for public release)

| Image                                                        | Prompt                               | 3D generation                                                |
| ------------------------------------------------------------ | ------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/SwayamInSync/MIRA/main/assets/water_can.png" width=128 height="128"> | None                                 | <img src="https://github.com/SwayamInSync/MIRA/raw/main/assets/water_can.mov"> |
| <img src="https://github.com/SwayamInSync/MIRA/blob/main/assets/girl.png?raw=true" width=128 height="128"> | None                                 | <img src="https://github.com/SwayamInSync/MIRA/raw/main/assets/girl.mov"> |
| <img src="https://github.com/SwayamInSync/MIRA/blob/main/assets/cat.png?raw=true" width=128 height="128"> | A photograph of cat sitting on table | <img src="https://github.com/SwayamInSync/MIRA/raw/main/assets/cat.mov"> |
| <img src="https://github.com/SwayamInSync/MIRA/blob/main/assets/lamp.png?raw=true" width=128 height=128> | None                                 | <img src="https://github.com/SwayamInSync/MIRA/raw/main/assets/lamp.mov"> |
| <img src="https://github.com/SwayamInSync/MIRA/blob/main/assets/whale.png?raw=true" width=128 height=128> | None                                 | <img src="https://github.com/SwayamInSync/MIRA/raw/main/assets/whale.mov"> |



## Setup

- Clone the repository

Only For **dataset preprocessing/rendering (Linux)**

- ```bash
  apt-get update -y
  apt-get install -y xvfb
  apt-get install libxrender1
  apt-get install libxi6 libgconf-2-4
  apt-get install libxkbcommon-x11-0
  apt-get install -y libgl1-mesa-glx
  
  echo "Installing Blender-4.0.2..."
  wget https://ftp.nluug.nl/pub/graphics/blender//release/Blender4.0/blender-4.0.2-linux-x64.tar.xz && tar -xf blender-4.0.2-linux-x64.tar.xz && rm blender-4.0.2-linux-x64.tar.xz
  ```

Install the python requirements

- ```bash
  pip install -r requirements.txt
  ```

## Dataset preparation

- Run the `load_input_data.py` as `python load_input_data.py`. Update the dataset directory in `config.json`

  > For computational contraint, it is recommended to download the object data and render them individually as

  - Linux

    - ```bash
      DISPLAY=:0.0 && xvfb-run --auto-servernum blender --background --python blender_scripts/dataset_rendering.py -- --object_path 'path to 3D object' --num_renders 32 --output_dir 'path to dataset_dir' --engine CYCLES
      ```

  - Others

    - ```bash
      blender --background --python blender_scripts/dataset_rendering.py -- --object_path 'path to 3D object' --num_renders 32 --output_dir 'path to dataset_dir' --engine CYCLES
      ```

## Training

- Get the hostname as `hostname -i`

- Run the following command

  - ```bash
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_ddp.py
    ```

    Replace the `$MASTER_ADDR` by the hostname of main system

## Inference

Run the `test.py` script as

```bash
python test.py --checkpoint_path=<path to model checkpoint> --config_path=<path to config.json file> --mode=<text/image> --input=<prompt/image_path> --output_path=<path to output directory> --export_video --export_mesh
```

This will save the rendered video and mesh as `.ply` format inside the specified output directory.

## References

- Papers
  - [LRM](https://arxiv.org/abs/2311.04400), [Efficient Geometry-aware 3D Generative Adversarial Networks](https://arxiv.org/abs/2112.07945), [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517)
- Open-Source repos
  - [OpenLRM](https://github.com/3DTopia/OpenLRM), [TensorRF](https://github.com/apchenstu/TensoRF), [Eg3D](https://github.com/NVlabs/eg3d)
