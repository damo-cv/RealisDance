# Environment Setup

To prepare the condition videos, please install the required `environments` and download the `required checkpoints` for [DWPose](https://github.com/IDEA-Research/DWPose), [HaMeR](https://github.com/geopavlakos/hamer) and [SMPLerX(The checkpoints are prepared below)](https://github.com/caizhongang/SMPLer-X) by following the instructions provided in their respective GitHub repositories.
# **NOTICE**
👏👏👏 We have prepared all the models required for SMPLer-X and made them available on [Google Drive](https://drive.google.com/file/d/1hKiLcJ-NdqQ3APrC6wpdm6xlalJ2Lr4O/view?usp=sharing) now. You can download these models and place them in the correct paths as specified in the [SMPLerX](https://github.com/caizhongang/SMPLer-X).
Ensure that each environment and checkpoints are properly set up before running the following code.
# Inference Steps

After setting up the required environments, you can perform inference on the original video by following these steps:

1. **DWPose Inference:**
    ```bash
    source activate {YOUR_DWPOSE_ENV}
    cd prepare_pose/DWPose/ControlNet-v1-1-nightly
    python inference_video.py --video_path {YOUR_VIDEO_PATH} --output_path {OUTPUT_PATH}
    ```

2. **HaMeR Inference:**
    ```bash
    cd -
    source activate {YOUR_HaMeR_ENV}
    cd prepare_pose/hamer-main
    python inference_video.py --video_path {YOUR_VIDEO_PATH} --output_path {OUTPUT_PATH}
    ```
3. **SMPLerX Inference:**
    ```bash
    cd -
    source activate {YOUR_SMPLerX_ENV}
    cd prepare_pose/smplerX/main
    python inference_video.py --video_path {YOUR_VIDEO_PATH} --output_path {OUTPUT_PATH} --pretrained_model smpler_x_h32
    ```
    If you encounter the following errors:
    ```bash
    ImportError: ('Unable to load OpenGL library', 'OSMesa: cannot open shared object file: No such file or directory', 'OSMesa', None)
    ```
    or
    ```bash
    ImportError: cannot import name 'OSMesaCreateContextAttribs' from 'OpenGL.osmesa' (/root/miniconda/envs/smplerx/lib/python3.8/site-packages/OpenGL/osmesa/__init__.py)
    ```
    Try running the following commands to resolve the issue:
    ```bash
    apt-get install -y python-opengl libosmesa6
    apt-get install libosmesa6-dev
    pip install --upgrade pyopengl==3.1.4
    ```
    You may see a version-related warning when upgrading pyopengl. You can safely ignore this warning.


By following these steps, you will be able to generate the condition video and proceed to create a human dance video.