import os
import numpy as np
from PIL import Image
from PIL import ImageChops


def merge_images(pic_dir, out_path):
    """
    Merges all images in the specified directory using the lighter blending mode and saves the result to the specified
    output path.

    Args:
        pic_dir (str): The path to the directory containing the input images.
        out_path (str): The path to save the output image.

    Returns:
        PIL.Image: The merged image as a PIL Image object.
    """
    files = os.listdir(pic_dir)
    final_image = Image.open(os.path.join(pic_dir, files[0]))
    for i in range(1, len(files)):
        current_image = Image.open(os.path.join(pic_dir, files[i]))
        final_image = ImageChops.lighter(final_image, current_image)
    final_image.save(out_path, 'JPEG')

    return final_image


if __name__ == '__main__':
    pic_dir = '/workspaces/MeteorDection/test/merge_tests'
    out_path = '/workspaces/MeteorDection/test/merged.jpg'
    merge_images(pic_dir, out_path)
