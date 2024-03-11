import os
from PIL import Image
import random
import matplotlib.pyplot as plt

#------------------------------------------------------------
# Text-based EDA:

image_dir = 'human_mask_data/images'
mask_dir = 'human_mask_data/masks'

total_images = 0
total_image_size = 0

num_square_images = 0
num_non_square_images = 0

min_image_width = 1000000
max_image_width = 0
min_image_height = 1000000
max_image_height = 0

num_mismatched_sizes = 0

for img_name in os.listdir(image_dir):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        total_images += 1
        image_path = os.path.join(image_dir, img_name)
        image = Image.open(image_path)
        image_width, image_height = image.size
        image_size = os.path.getsize(image_path)
        total_image_size += image_size
        mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + '.png')
        if Image.open(image_path).size != Image.open(mask_path).size:
            num_mismatched_sizes += 1

        if image_width == image_height:
            num_square_images += 1
        else:
            num_non_square_images += 1
        min_image_width = min(min_image_width, image_width)
        max_image_width = max(max_image_width, image_width)
        min_image_height = min(min_image_height, image_height)
        max_image_height = max(max_image_height, image_height)


print(f"Total images: {total_images}")
print(f"Total image size (bytes): {total_image_size}")
print(f"Number of square images: {num_square_images}")
print(f"Number of non-square images: {num_non_square_images}")
print(f"Minimum image width: {min_image_width}")
print(f"Maximum image width: {max_image_width}")
print(f"Minimum image height: {min_image_height}")
print(f"Maximum image height: {max_image_height}")
print(f"Number of images with masks of different sizes: {num_mismatched_sizes}")

#------------------------------------------------------------
# now the visual EDA:

image_files = os.listdir(image_dir)

# 3 images and their masks used
selected_images = random.sample(image_files, 3)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))

for i, image_name in enumerate(selected_images):
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)
    axes[0,i].imshow(image)
    axes[0,i].axis('off')
    axes[0,i].set_title('Provided Image')

    mask_name = os.path.splitext(image_name)[0] + '.png'
    mask_path = os.path.join(mask_dir, mask_name)
    mask = Image.open(mask_path)
    axes[1,i].imshow(mask, cmap='gray', alpha=0.5)
    axes[1,i].set_title('Provided Mask')
    axes[1,i].axis('off')

plt.tight_layout()
plt.savefig('my_eda.png')
        