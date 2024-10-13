# Installation
## 📙 We support two independent environments. You just need to install corresponding environment.
## For MMPose
🌵🌵🌵 This environment helps you to train and test our DWPose. You can ignore the following installation for ControlNet.

🌵 You can refer [MMPose Installation](https://mmpose.readthedocs.io/en/latest/installation.html) or
```
# Set MMPose environment
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## For ControlNet
🌵🌵🌵 This environment helps you to apply DWPose to ControlNet. You can ignore the above installation for mmpose.

🌵 First, make sure to run ControlNet successfully.
```
# Set ControlNet environment
conda env create -f environment.yaml
conda activate control-v11
```
🌵 Second, install tools to apply DWPose to ControlNet. If it's hard to install onnxruntime, you can refer branch [opencv_onnx](https://github.com/IDEA-Research/DWPose/tree/opencv_onnx), which runs the onnx model with opencv.
```
# Set ControlNet environment
pip install onnxruntime
# if gpu is available
pip install onnxruntime-gpu
```

First, you need to download our Pose model dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and Det model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)), then put them into ControlNet-v1-1-nightly/annotator/ckpts. Then you can use DWPose to generate the images you like.
