from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full

# Import the DWposeDetector
from DWPose.ControlNet.annotator.dwpose import DWposeDetector

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HaMeR video processing script')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video')
    parser.add_argument('--output_path', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    args = parser.parse_args()

    # Download and load checkpoints
    # download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Keypoint detector
    pose = DWposeDetector()

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.output_path, exist_ok=True)

    # Open the video file
    video_path = args.video_path
    if not os.path.exists(video_path):
        print('Video path does not exist: {}'.format(video_path))
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file: {}'.format(video_path))
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the output video writer
    out_video_path = os.path.join(args.output_path, 'hamer_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or 'MJPG' depending on your needs
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame of the video
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc='Processing video')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_cv2 = frame.copy()
        vitposes_out = pose(img_cv2)
        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes[-42:-21]
            right_hand_keyp = vitposes[-21:]
            # Rejecting not confident detections

            keyp = left_hand_keyp
            if (keyp != -1).sum() > 6:
                valid_indices = np.where(keyp != -1)
                valid_keyp_x = keyp[valid_indices[0], 0]
                valid_keyp_y = keyp[valid_indices[0], 1]
                bbox = [valid_keyp_x.min(), valid_keyp_y.min(), valid_keyp_x.max(), valid_keyp_y.max()]
                bboxes.append(bbox)
                is_right.append(0)

            keyp = right_hand_keyp
            if (keyp != -1).sum() > 6:
                valid_indices = np.where(keyp != -1)
                valid_keyp_x = keyp[valid_indices[0], 0]
                valid_keyp_y = keyp[valid_indices[0], 1]
                bbox = [valid_keyp_x.min(), valid_keyp_y.min(), valid_keyp_x.max(), valid_keyp_y.max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            # If no hands are detected, write a black image
            black_frame = np.zeros_like(frame)
            out_video.write(black_frame)
            frame_idx += 1
            pbar.update(1)
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        img_size = None

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Collect all verts and cams to list
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right_hand = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right_hand - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_hand)


        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view_diff = renderer.render_rgba_multiple_diff(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
            processed_frame = (cam_view_diff * 255).astype(np.uint8)[:, :, ::-1]

            # Write the processed frame to the output video
            out_video.write(processed_frame)
            frame_idx += 1
            pbar.update(1)

    # Release resources
    cap.release()
    out_video.release()
    pbar.close()
    print('Processing complete. Output saved to:', out_video_path)

if __name__ == '__main__':
    main()
