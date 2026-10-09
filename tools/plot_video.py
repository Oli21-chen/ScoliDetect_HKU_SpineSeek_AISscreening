import os
import numpy as np
from PIL import Image
import torch

# -----------------------------------------
# 1. Your video array / tensor goes here
# -----------------------------------------
# Examples:
# video: np.ndarray of shape (T, H, W, 3) or (T, 3, H, W)
# video: torch.Tensor with similar shape

# video = ...  # TODO: set this to your video
video_frames = video_clip
# If it's a PyTorch tensor, convert to NumPy first:
if isinstance(video_frames, torch.Tensor):
    video_np = video_frames.detach().cpu().numpy()
else:
    video_np = video_frames  # already numpy

# -----------------------------------------
# 2. Output directory (where to save frames)
# -----------------------------------------
output_dir = r"C:\Users\Administrator\project\tools"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------
# 3. Save frames one by one
# -----------------------------------------
for idx, frame in enumerate(video_frames):
    # Handle (T, 3, H, W) -> (H, W, 3)
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)

    # Ensure uint8 [0, 255]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 1) * 255 if frame.max() <= 1.0 else np.clip(frame, 0, 255)
        frame = frame.astype(np.uint8)

    img = Image.fromarray(frame)
    filename = os.path.join(output_dir, f"frame_{idx:04d}.png")
    img.save(filename)

print("Saved", len(video_frames), "frames to", output_dir)