from annotator.dwpose import DWposeDetector
import math
import argparse
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default='', help = 'input path')
parser.add_argument('--output_path', type=str, default='out_demo', help = 'output path')
args = parser.parse_args()


if __name__ == "__main__":
    video_path = args.video_path
    assert os.path.exists(video_path)
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    pose = DWposeDetector()
    cap = cv2.VideoCapture(video_path)
    results = []
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret: 
            break
        results.append(pose(frame))

    with open(os.path.join(output_path, 'dwpose_video.pkl'), 'wb') as f:
        pickle.dump(results, f)

    cap.release()