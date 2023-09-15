from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import ImageDraw
import csv

class ImageSelector:
    def __init__(self, img_dir, coords_file, output_path) -> None:
        self.img_dir = img_dir
        self.coords_file = coords_file
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        else:
            os.system(f'rm -rf {self.output_path}/*')


    def run(self):
        coords = pd.read_csv(self.coords_file, comment='#')

        for i, row in tqdm(coords.iterrows(), desc='Processing ROIs...'):
            label, x1, y1, x2, y2 = row
            if not os.path.exists(self.output_path + f'/{label}'):
                os.makedirs(self.output_path + f'/{label}')
            self.plot_roi_image(label, x1, y1, x2, y2)
            self.search_for_meteor(label, x1, y1, x2, y2)


    def plot_roi_image(self, label, x1, y1, x2, y2):
        with open(self.coords_file, 'r') as f:
            reader = csv.reader(f)
            first_row = next(reader)
            filename_line = first_row[0]
            filename_prefix = '# Image file name: '
            if filename_line.startswith(filename_prefix):
                filename = filename_line[len(filename_prefix):]

                img = Image.open(filename)
                # draw a rectangle on img
                drawer = ImageDraw.Draw(img)
                drawer.rectangle(((x1, y1), (x2, y2)), outline='red', width=5)
                img.save(self.output_path + f'/{label}/roi_{os.path.basename(filename)}')

    def search_for_meteor(self, label, x1, y1, x2, y2):
        for filename in os.listdir(self.img_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = Image.open(self.img_dir + '/' + filename)
                img = img.crop((x1, y1, x2, y2))
                img = np.array(img)
                self.detect_lines(img, self.output_path+ f'/{label}', os.path.splitext(filename)[0])

    def detect_lines(self, img, output_path, image_name):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        low_threshold = 100
        high_threshold = 200
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 2  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
        
         # Draw the lines on the image
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()
        axes[0].imshow(gray, cmap='gray')
        axes[1].imshow(blur_gray, cmap='gray')
        axes[2].imshow(edges, cmap='gray')

        if lines is not None:  
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.rectangle(line_image,(x1,y1),(x2,y2),(255,0,0),2)
            lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
            axes[3].imshow(lines_edges, cmap='gray')
            fig.suptitle(f'Detected {len(lines)} lines')
        else:
            fig.suptitle(f'No lines detected')

        fig.tight_layout()
        fig.savefig(output_path + f'/{image_name}_line.png')

        if lines is None:
            return None
        else:
            return lines

    def calculate_mean_pixel_value(self, image_path, x1, y1, x2, y2):
        # Load the image
        image = Image.open(image_path)

        # Crop the image to the specified rectangle
        cropped_image = image.crop((x1, y1, x2, y2))

        # Calculate the mean pixel value
        mean_pixel_value = sum(map(sum, cropped_image.getdata())) / len(cropped_image.getdata())

        return mean_pixel_value

    def calculate_mean_brightness(self, image_path, x1, y1, x2, y2):
        # Load the image
        image = Image.open(image_path)

        # Crop the image to the rectangle
        rect_image = image.crop((x1, y1, x2, y2))
        plt.imshow(rect_image)
        plt.show()

        # Convert the image to grayscale
        gray_image = rect_image.convert('L')

        # Calculate the mean brightness of the grayscale image
        mean_brightness = sum(gray_image.getdata()) / len(gray_image.getdata())

        return mean_brightness


if __name__ == '__main__':
    rect_dir = '/workspaces/MeteorDection/test/rectify_test'
    coords_file = '/workspaces/MeteorDection/outputs/coords.csv'
    output_path = '/workspaces/MeteorDection/test/line_detection'

    # roi, mean_pixel_values = dummy(rect_dir, coords_file)
    # plt.plot(mean_pixel_values, '-o')
    # plt.title(f'Mean pixel value of {roi}')
    # plt.show()

    # img = Image.open('test/rectify_test/DSC03845.jpg')
    # plt.imshow(img)
    # plt.show()

    # coords = pd.read_csv(coords_file, comment='#')

    # label, x1, y1, x2, y2 = coords.iloc[0]
    # img = Image.open('test/merged_rectified.jpg')
    # img = img.crop((x1, y1, x2, y2))
    # img = np.array(img)
    # # plt.imshow(img)
    # # plt.show()
    # detect_lines(img, output_path=f'{label}')
    c = ImageSelector(rect_dir, coords_file, output_path)
    c.run()
    
