import argparse
import glob
import numpy as np
import os
import torch

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image
from src.pipelines.rd_dit_pipeline import RealisDanceDiTPipeline
from transformers import CLIPVisionModel

import decord

decord.bridge.set_bridge("torch")


def is_image(path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return path.lower().endswith(valid_extensions)


def load_image(path, div=127.5, add=-1):
    # read image
    image = torch.from_numpy(np.array(Image.open(path).convert("RGB")))
    # [0, 255] -> [-1, 1]
    image = image / div + add
    # H W C -> B C F H W
    image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return image


def load_video(
    path,
    fps=16,
    start_index=0,
    num_frames=81,
    div=127.5,
    add=-1,
):
    video_reader = decord.VideoReader(path)
    video_length = len(video_reader)
    ori_fps = video_reader.get_avg_fps()
    normed_video_length = max(round(video_length / ori_fps * fps), num_frames)
    batch_index_all = np.linspace(0, video_length - 1, normed_video_length).round().astype(int).tolist()
    batch_index = batch_index_all[start_index:start_index + num_frames]
    video = video_reader.get_batch(batch_index).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    del video_reader
    video = video / div + add
    return video  # 1 C T H W


def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default=None, help='path to reference image')
    parser.add_argument('--smpl', type=str, default=None, help='Path to smpl video')
    parser.add_argument('--hamer', type=str, default=None, help='Path to hamer video')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for video')
    parser.add_argument('--root', type=str, default=None, help='Root path for batch inference')
    parser.add_argument('--save-dir', type=str, default="./output", help='Path to output folder')
    parser.add_argument('--max-res', type=int, default=768 * 768, help='Resolution of the generated video')
    parser.add_argument('--num-frames', type=int, default=81, help='Number of the generated video frames')
    args = parser.parse_args()

    # assign args
    ref_path = args.ref
    smpl_path = args.smpl
    hamer_path = args.hamer
    prompt = args.prompt
    root = args.root
    save_dir = args.save_dir
    max_res = args.max_res
    num_frames = args.num_frames
    os.makedirs(save_dir, exist_ok=True)

    # check args
    if root is None and (ref_path is None or smpl_path is None or hamer_path is None):
        raise ValueError("`root` and `ref` / `smpl` / `hamer` cannot be None at the same time.")
    elif root is not None and (ref_path is not None or smpl_path is not None or hamer_path is not None):
        print("WARNING: Will not use `ref` / `smpl` / `hamer` when `root` is not None.")

    # load model
    model_id = "theFoxofSky/RealisDance-DiT"
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = RealisDanceDiTPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # inference
    if root is not None:
        for ref_path in glob.glob(os.path.join(root, "ref", "*")):
            if not is_image(ref_path):
                continue
            vid = os.path.splitext(os.path.basename(ref_path))[0]
            smpl_path = os.path.join(root, "smpl", f"{vid}.mp4")
            hamer_path = os.path.join(root, "hamer", f"{vid}.mp4")
            prompt_path = os.path.join(root, "prompt", f"{vid}.txt")
            output_path = os.path.join(save_dir, f"{vid}.mp4")

            # prepare inputs
            ref_image = load_image(ref_path)
            smpl = load_video(smpl_path, num_frames=num_frames)
            hamer = load_video(hamer_path, num_frames=num_frames)
            prompt = ""
            with open(prompt_path, 'r', encoding='utf-8') as file:
                for l in file.readlines():
                    prompt += l.strip()

            output = pipe(image=ref_image, smpl=smpl, hamer=hamer, prompt=prompt, max_resolution=max_res).frames[0]
            export_to_video(output, output_path, fps=16)
    else:
        vid = os.path.splitext(os.path.basename(ref_path))[0]
        pose_id = os.path.splitext(os.path.basename(smpl_path))[0]
        ref_image = load_image(ref_path)
        smpl = load_video(smpl_path, num_frames=num_frames)
        hamer = load_video(hamer_path, num_frames=num_frames)
        output_path = os.path.join(save_dir, f"{vid}_{pose_id}.mp4")
        output = pipe(image=ref_image, smpl=smpl, hamer=hamer, prompt=prompt, max_resolution=max_res).frames[0]
        export_to_video(output, output_path, fps=16)


if __name__ == "__main__":
    main()
