# pylint: disable=no-member
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import ImageDraw
import csv
import astroalign as aa
from config import METEOR_DETECTION, HOUGH_LINE_PARAMS, LINE_DETECTION_OUTPUT

class ImageSelector:
    def __init__(self, img_dir, background_img, coords_file, output_path) -> None:
        self.img_dir = img_dir
        self.coords_file = coords_file
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        else:
            os.system(f'rm -rf {self.output_path}/*')
        self.background_img = background_img


    def run(self):
        coords = pd.read_csv(self.coords_file, comment='#')
        pbar = tqdm(coords.iterrows(), desc='Processing ROIs...')
        for i, row in coords.iterrows():
            label, x1, y1, x2, y2 = row
            if not os.path.exists(self.output_path + f'/{label}'):
                os.makedirs(self.output_path + f'/{label}')
            self.plot_roi_image(label, x1, y1, x2, y2)
            self.search_for_meteor(label, x1, y1, x2, y2)
            pbar.update(1)

    def generate_background_image(self, background_img):
        background_image = cv2.imread(background_img)
        # background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.output_path + '/background.jpg', background_image)
        return background_image

    def plot_roi_image(self, label, x1, y1, x2, y2):
        """Plot the region of interest on the image and save it to the output directory."""
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
        """In the region of interest, search for meteors in each image."""
        m,b = self.fit_line(0, 0, x2-x1, y2-y1)
        background_img = self.generate_background_image(self.background_img)
        print(f'Target line: {m}, {b}')
        for filename in os.listdir(self.img_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                if self.img_dir + '/' + filename == self.background_img:
                    continue
                img = cv2.imread(self.img_dir + '/' + filename)
                denoised_img = self.remove_noise_from_light_image(img, background_img, (x1, y1, x2, y2))
                self.detect_lines(denoised_img, m, b, self.output_path+ f'/{label}', os.path.splitext(filename)[0])

    def detect_lines(self, img, m, b, output_path, image_name):
        gray = img

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold, apertureSize=3)

        # 使用配置参数优化Hough变换
        rho = HOUGH_LINE_PARAMS['rho']
        theta = HOUGH_LINE_PARAMS['theta']
        threshold = HOUGH_LINE_PARAMS['threshold']
        min_line_length = HOUGH_LINE_PARAMS['min_line_length']
        max_line_gap = HOUGH_LINE_PARAMS['max_line_gap']
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
        
         # Draw the lines on the image
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Original Image (denoised)')
        axes[1].imshow(blur_gray, cmap='gray')
        axes[1].set_title('Blurred Image')
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title('Canny Edges')
        
        line_count = 0
        valid_lines = []
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    # 验证是否为流星
                    if self.validate_meteor_line(x1, y1, x2, y2, m, b):
                        print(f'Line detected: {m_line}, {b_line}')
                        cv2.rectangle(line_image,(x1,y1),(x2,y2),(255,0,0),2)
                        line_count += 1
                        valid_lines.append((x1, y1, x2, y2))
            lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
            axes[3].imshow(lines_edges, cmap='gray')
            axes[3].set_title(f'Detected Lines ({line_count})')
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

    def fit_line(self, x1, y1, x2, y2):
        # calculate slope and interception of the points (x1,y1) and (x2,y2)
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    def validate_meteor_line(self, x1, y1, x2, y2, expected_m, expected_b):
        """
        验证检测到的直线是否符合流星特征
        """
        # 1. 斜率验证
        m_line, b_line = self.fit_line(x1, y1, x2, y2)
        if abs(m_line - expected_m) >= METEOR_DETECTION['slope_tolerance']:
            return False

        # 2. 长度验证（流星通常较长）
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length < METEOR_DETECTION['min_line_length']:
            return False

        # 3. 长宽比验证
        line_width = 2  # 假设线条宽度为2像素
        aspect_ratio = line_length / line_width
        if aspect_ratio < METEOR_DETECTION['min_aspect_ratio']:
            return False

        return True

    def remove_noise_from_light_image(self, light_image, background_image, coords):
        x1, y1, x2, y2 = coords
        width = light_image.shape[0]
        height = light_image.shape[1]

        if width != background_image.shape[0] or height != background_image.shape[1]:
            raise Exception('Image sizes are not equal')
        
        target_img = background_image
        source_img = light_image

        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        target_crop = target_gray[y1:y2, x1:x2]
        source_crop = source_gray[y1:y2, x1:x2]

        registered_image, footprint = aa.register(target_crop, source_crop)

        new_image = np.subtract(source_crop, registered_image)
        black_image = np.zeros([new_image.shape[0], new_image.shape[1]], dtype=np.uint8)
        return np.uint8(np.maximum(new_image, black_image))
    

if __name__ == '__main__':
    rect_dir = '/workspaces/MeteorDection/test/rectify_test'
    coords_file = '/workspaces/MeteorDection/outputs/coords.csv'
    output_path = '/workspaces/MeteorDection/test/line_detection'
    background_image = '/workspaces/MeteorDection/test/rectify_test/DSC03826.jpg'

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
    c = ImageSelector(rect_dir, background_image, coords_file, output_path)
    c.run()
    
