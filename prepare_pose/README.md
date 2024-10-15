# Environment Setup

To prepare the condition videos, please install the required environments and download the needed checkpoints for [DWPose](https://github.com/IDEA-Research/DWPose), [HaMeR](https://github.com/geopavlakos/hamer) and [SMPLerX](https://github.com/caizhongang/SMPLer-X) by following the instructions provided in their respective GitHub repositories.

Please ensure that each environment and checkpoints are properly set up before running the following code.
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
    source activate {YOUR_HaMeR_ENV}
    cd prepare_pose/hamer-main
    python inference_video.py --video_path {YOUR_VIDEO_PATH} --output_path {OUTPUT_PATH}
    ```
3. **SMPLerX Inference:**
    ```bash
    source activate {YOUR_SMPLerX_ENV}
    cd prepare_pose/smplerX/main
    python inference_video.py --video_path {YOUR_VIDEO_PATH} --output_path {OUTPUT_PATH}
    ``` 


By following these steps, you will be able to generate the condition video and proceed to create a human dance video.

