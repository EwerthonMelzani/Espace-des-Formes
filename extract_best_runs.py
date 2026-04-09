import os
import shutil
from PIL import Image

BEST_RUNS = {
    "circles": {"2D": 2, "3D": 2},
    "moons": {"2D": 0, "3D": 0},
    "blobs": {"2D": 1, "3D": 1},
    "gaussian_quantiles": {"2D": 0, "3D": 0}
}

def extract(gif_path, out_dir, prefix="frame_"):
    if not os.path.exists(gif_path):
        print(f"Skipping {gif_path}, not found.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    with Image.open(gif_path) as im:
        for i in range(im.n_frames):
            im.seek(i)
            # convert to RGB to avoid transparency issues
            rgb_im = im.convert('RGB')
            file_path = os.path.join(out_dir, f"{prefix}{i}.png")
            rgb_im.save(file_path)
    print(f"Extracted {im.n_frames} frames to {out_dir}")

def generate_static_before_after(out_dir):
    frames = sorted([f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".png")],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))
    if not frames:
        return
    
    first_frame = os.path.join(out_dir, frames[0])
    last_frame = os.path.join(out_dir, frames[-1])
    
    # Copy to standardized names so LaTeX can find them reliably without knowing total frames
    shutil.copy2(first_frame, os.path.join(out_dir, "before.png"))
    shutil.copy2(last_frame, os.path.join(out_dir, "after.png"))
    print(f"Generated before.png and after.png for {out_dir}")


def main():
    base_dir = "animacoes"
    out_base = os.path.join(base_dir, "frames")
    
    os.makedirs(out_base, exist_ok=True)

    for ds, dim_seeds in BEST_RUNS.items():
        for dim, seed in dim_seeds.items():
            # Example path: animacoes/2D/circles_seed2_2D.gif
            gif_name = f"{ds}_seed{seed}_{dim}.gif"
            gif_path = os.path.join(base_dir, dim, gif_name)
            
            # Output dir: animacoes/frames/circles_2D_frames
            out_dir = os.path.join(out_base, f"{ds}_{dim}_frames")
            
            extract(gif_path, out_dir)
            generate_static_before_after(out_dir)

if __name__ == "__main__":
    main()
