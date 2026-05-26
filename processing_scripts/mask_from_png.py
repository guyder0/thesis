import cv2
from pathlib import Path
from tqdm import tqdm

def main(
    input_dir: Path,
    factor: int = 1,
):
    image_files = list(input_dir.glob("*.png"))
    parent_dir = input_dir.parent

    mask_out = parent_dir / ("masks" if factor == 1 else f"masks_{factor}")
    mask_out.mkdir(exist_ok=True)
    
    if factor > 1:
        img_out = parent_dir / f"images_{factor}"
        img_out.mkdir(exist_ok=True)

    for img_path in tqdm(image_files, desc=f"Processing factor {factor}"):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]
        new_size = (w // factor, h // factor)
        
        if factor > 1:
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(str(mask_out / img_path.name), mask)

        if factor > 1:
            img_rgb = img[:, :, :3] if img.shape[2] == 4 else img
            cv2.imwrite(str(img_out / img_path.name), img_rgb)

if __name__ == "__main__":
    import tyro
    tyro.cli(main)