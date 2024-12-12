import os
from tqdm import tqdm
from rect_fisheye import rectify_fisheye
from merge_pics import merge_images
from area_selection import AreaSelector

def rectify_images(image_folder, rect_folder):
    # Rectify each image in the image folder and save it to the output folder
    if not os.path.exists(rect_folder):
        os.makedirs(rect_folder)

    pbar = tqdm(os.listdir(image_folder), desc='Rectifying fisheye images...') 
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(image_folder, filename)
            output_path = os.path.join(rect_folder, filename)
            rectify_fisheye(input_path, output_path)
            pbar.update(1)

def merge_rectified_images(rect_folder, merge_path):
    # Merge the rectified images
    merge_images(rect_folder, merge_path)


def select_roi(image_path, coords_file):
    # Create the AreaSelector object and run it
    selector = AreaSelector(image_path, coords_file)
    selector.run()

def main():
    image_folder = 'test/merge_tests'
    rect_folder = 'test/rectify_test'
    merge_path = 'test/merged_rectified.jpg'

    # rectify_images(image_folder, rect_folder)
    # merge_rectified_images(rect_folder, merge_path)
    select_roi(merge_path, 'outputs/coords.csv')

if __name__ == '__main__':
    main()