# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from SphereAR.model import create_model, get_model_args


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    # delete the sample folder to save space
    os.system(f"rm -r {sample_dir}")
    return npz_path


def main(args):
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    # create and load gpt model
    precision = {"none": torch.float32, "bf16": torch.bfloat16}[args.mixed_precision]
    model = create_model(args, device)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "ema" in checkpoint and not args.no_ema:
        print("use ema weight")
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    else:
        raise Exception("please check model weight")

    model.load_state_dict(model_weight, strict=True)
    model.eval()
    del checkpoint

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = (
        os.path.basename(args.ckpt).replace(".pth", "").replace(".pt", "")
    )
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-"
        f"steps-{args.sample_steps}-cfg-{args.cfg_scale}-seed-{args.seed}"
    )
    if not args.no_ema:
        folder_name += "-ema"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if os.path.isfile(sample_folder_dir + ".npz"):
        if rank == 0:
            print(f"Found {sample_folder_dir}.npz, skipping sampling.")
        dist.barrier()
        dist.destroy_process_group()
        return 1
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(
        math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size
    )
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert (
        total_samples % dist.get_world_size() == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    total = 0
    start_time = time.time()
    for i in tqdm(range(iterations), desc="Sampling"):
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        with torch.amp.autocast("cuda", dtype=precision):
            samples = model.sample(
                c_indices,
                sample_steps=args.sample_steps,
                cfg_scale=args.cfg_scale,
            )

        samples = (
            torch.clamp(127.5 * samples + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        print(
            f"Rank {rank} has sampled {total} images so far, cost {time.time() - start_time:.2f} seconds"
        )

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0 and args.to_npz:
        print(f"Total time taken for sampling: {time.time() - start_time:.2f} seconds")
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_model_args()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--cfg-scale", type=float, default=4.6)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--mixed-precision", type=str, default="bf16", choices=["none", "bf16"]
    )
    parser.add_argument("--to-npz", action="store_true")
    args = parser.parse_args()
    main(args)
