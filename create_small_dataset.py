import os
import random
from PIL import Image, ImageDraw

# dataset structure we will create
splits = {
    "train": {"good": 10, "defect": 10},
    "val": {"good": 10, "defect": 10},
    "test": {"good": 10, "defect": 10},
}

img_size = 256

# create folders
for split in splits:
    for cls in splits[split]:
        os.makedirs(f"dataset/{split}/{cls}", exist_ok=True)


def make_tile(defect=False):
    """Generate a synthetic tile image (with or without cracks)."""
    img = Image.new("RGB", (img_size, img_size), (220, 220, 220))
    d = ImageDraw.Draw(img)

    # draw border
    d.rectangle([4, 4, img_size - 5, img_size - 5], outline=(200, 200, 200), width=2)

    # add cracks for defect images
    if defect:
        for _ in range(random.randint(1, 2)):
            x1, y1 = random.randint(10, 245), random.randint(10, 245)
            x2, y2 = random.randint(10, 245), random.randint(10, 245)
            d.line([x1, y1, x2, y2], fill=(50, 50, 50), width=random.randint(2, 4))

    return img


# generate images
for split in splits:
    for cls, count in splits[split].items():
        for i in range(count):
            img = make_tile(defect=(cls == "defect"))
            img.save(f"dataset/{split}/{cls}/{cls}_{i}.png")

print("\n🎉 Dataset created successfully!")
print("Check the 'dataset' folder.")
