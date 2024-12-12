# pylint: skip-file
import cv2
import tensorflow as tf
import numpy as np
import rawpy

from object_detection.utils import label_map_util


tf.keras.backend.clear_session()
class_id = None
threshold = 0.01
detect_fn = tf.saved_model.load('data/trt_fp16_dir/saved_model')
print(type(detect_fn))
label_map = label_map_util.load_labelmap('data/labelmap.txt')
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_from_image(image):
    im_height, im_width, _ = image.shape
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    input_tensor = np.expand_dims(image, 0)
    print(input_tensor.shape)
    detections = detect_fn(input_tensor)

    bboxes = detections['detection_boxes'][0].numpy()
    bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
    bscores = detections['detection_scores'][0].numpy()
    det_boxes = extract_bboxes(bboxes, bclasses, bscores, im_width, im_height)

    return det_boxes


def extract_bboxes(bboxes, bclasses, bscores, im_width, im_height):
    boxes = []
    for idx, bbox in enumerate(bboxes):
        if class_id is None or bclasses[idx] in class_id:
            if bscores[idx] >= threshold:
                y_min = int(bbox[0] * im_height)
                x_min = int(bbox[1] * im_width)
                y_max = int(bbox[2] * im_height)
                x_max = int(bbox[3] * im_width)
                class_label = category_index[int(bclasses[idx])]['name']
                boxes.append([x_min, y_min, x_max, y_max,
                             class_label, float(bscores[idx])])
    return boxes


def display_detections(image, boxes_list, det_time=None):
    if not boxes_list:
        return image  # input list is empty
    img = image.copy()
    for idx in range(len(boxes_list)):
        x_min = boxes_list[idx][0]
        y_min = boxes_list[idx][1]
        x_max = boxes_list[idx][2]
        y_max = boxes_list[idx][3]
        cls = str(boxes_list[idx][4])
        score = str(np.round(boxes_list[idx][-1], 2))

        text = cls + ": " + score
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.rectangle(img, (x_min, y_min - 20),
                      (x_min, y_min), (255, 255, 255), -1)
        if y_min > 10:
            cv2.putText(img, text, (x_min + 5, y_min - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(img, text, (x_min + 5, y_max + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if det_time != None:
        fps = round(1000. / det_time, 1)
        fps_txt = str(fps) + " FPS"
        cv2.putText(img, fps_txt, (25, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return img

# image = io.imread('test/DSC_5064.NEF')
image = rawpy.imread('test/DSC_5064.NEF').postprocess(use_camera_wb=True)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# display image
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

det_boxes = detect_from_image(image)
if det_boxes[0][5] > 0.1:
    img = display_detections(image, det_boxes[:5])