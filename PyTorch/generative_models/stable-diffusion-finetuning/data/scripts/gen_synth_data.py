###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
from PIL import Image, ImageDraw
import numpy as np
import os
import random

def create_image_with_yellow_square_face(idx=0,path='./',dirname='yellow_rect',min_size=400,max_size=2000,min_prop=0.25,max_prop=0.9):
    # Generate random image size (let's say between 100 and 1000)
    X,Y = np.random.randint((min_size,min_size),(max_size,max_size))

    # Create a white background image
    img = Image.new('RGB', (X, Y), (255, 255, 255))
    # Generate random mask size (let's say between 10 and image size)
    mask_x = random.randint(int(X*min_prop), int(X*max_prop))
    mask_y = random.randint(int(Y*min_prop), int(Y*max_prop))
    # Generate random top-left point for the mask (x and y coordinates)
    mask_start_x = random.randint(0, X - mask_x)
    mask_start_y = random.randint(0, Y - mask_y)

    # Create an ImageDraw object
    draw = ImageDraw.Draw(img)
    # Draw the mask first
    mask = Image.new('L', (X, Y), 1)
    draw_mask=ImageDraw.Draw(mask)
    draw_mask.rectangle([mask_start_x, mask_start_y, mask_start_x+mask_x, mask_start_y+mask_y], fill=255)
    data_path=os.path.join(path,dirname)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    mask.save(os.path.join(data_path,f'{idx}.mask.png'))

    # Draw a yellow square (mask) on the image
    draw.rectangle([mask_start_x, mask_start_y, mask_start_x+mask_x, mask_start_y+mask_y], fill=(255,255,0))

    # Draw two eyes on the square
    eye_size = min(mask_x // 10,mask_y//10)
    eye_y = mask_start_y + mask_y // 4
    for i in range(2):
        eye_x = mask_start_x + (i+1) * mask_x // 3 - eye_size // 2
        draw.ellipse([eye_x, eye_y, eye_x+eye_size, eye_y+eye_size], fill=(0,0,0))

    # Draw a smile on the square
    smile_size = mask_x // 2
    smile_y = mask_start_y + 3 * mask_y // 5
    smile_height = mask_y // 5
    smile_start_x = mask_start_x + mask_x // 4
    draw.arc([smile_start_x, smile_y, smile_start_x+smile_size, smile_y+smile_height], start=0, end=180, fill=(0,0,0))

    # Save the image to a file
    img.save(os.path.join(data_path,f'image_{idx}.png'))

    # Return the mask coordinates
    return mask_start_x, mask_start_y

if __name__ == '__main__':
    base_path = "./"
    rel_dir_path = "yellow_rect"

    # Generate dataset of 10 images with corresponding masks
    for i in range(10):
        create_image_with_yellow_square_face(i, path=base_path, dirname=rel_dir_path)
    print("The synthetic dataset is generated at ", os.path.abspath(os.path.join(base_path, rel_dir_path)))
