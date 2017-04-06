# Import the required modules
# coding:utf-8
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage import color, transform
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from nms import nms
from config import *
import matplotlib.pyplot as plt
import os


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image",
                        default='/media/gongxijun/work/dl/redlantern/VOCdevkit/redlantern/tmp/JPEGImages')
    parser.add_argument('-d', '--downscale', help="Downscale ratio", default=4,
                        type=int)
    parser.add_argument('-v', '--visualize', default=False, help="Visualize the sliding window",
                        action="store_true")
    args = vars(parser.parse_args())

    list = ['/home/gongxijun/data/222',
            '/media/gongxijun/work/dl/redlantern/object-detector/11',
            '/media/gongxijun/work/data/灯笼',
            '/media/gongxijun/work/data/VOCdevkit (2)/VOC2007/JPEGImages',
            '/media/gongxijun/work/test-srf-0316/test-srf -0316',
            '/home/gongxijun/data/dataset/neg',
            '/home/gongxijun/data/dataset/train/pos',
            '/media/gongxijun/work/dl/redlantern/VOCdevkit/redlantern/JPEGImages',
            '/home/gongxijun/data/橘子',
            '/home/gongxijun/data/轮子',
            '/home/gongxijun/data/苹果',
            '/home/gongxijun/data/山楂',
            '/home/gongxijun/data/西瓜',
            '/home/gongxijun/data/object-detector/11',
            '/home/gongxijun/data/灯笼',
            '/home/gongxijun/data/test3',
            '/home/gongxijun/文档'
            ]
    # Read the image
    root_path = list[-1]
    _cnt = 0;
    _num = 0
    for image_path in os.listdir(root_path):
        img = cv2.imread(os.path.join(root_path, image_path))
        img = transform.resize(img, (400, 400))
        imt = color.rgb2gray(img)
        min_wdw_sz = (90, 128)
        step_size = (10, 5)
        downscale = args['downscale']
        visualize_det = args['visualize']

        # Load the classifier
        clf = joblib.load(model_path)

        # List to store the detections
        detections = []
        # The current scale of the image
        scale = 0
        # Downscale the image and iterate
        for im_scaled in pyramid_gaussian(imt, downscale=downscale):
            # This list contains detections at the current scale
            cd = []
            print scale
            im = imt
            # If the width or height of the scaled image is less than
            # the width or height of the window, then end the iterations.
            if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue
                # Calculate the HOG features
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
                # pred = clf.predict([fd])
                prod = clf.predict_proba([fd])[0][1]
                prod = round(prod, 3);
                if prod >= 0.9:
                    print  "Detection:: Location -> ({}, {})".format(x, y)
                    print "Scale ->  {} | Confidence Score {} \n".format(scale, prod)
                    detections.append((x, y, prod,
                                       int(min_wdw_sz[0] * (downscale ** scale)),
                                       int(min_wdw_sz[1] * (downscale ** scale))))
                    cd.append(detections[-1])
                # If visualize is set to true, display the working
                # of the sliding window
                if visualize_det:
                    clone = img
                    for x1, y1, _, _, _ in cd:
                        # Draw the detections at this scale
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                        im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                                  im_window.shape[0]), (255, 255, 255), thickness=2)

                    cv2.imshow("Sliding Window in Progress", clone)
                    cv2.waitKey(30)
            # Move the the next scale
            scale += 1

        # # Display the results before performing NMS
        # clone = img
        #
        # for (x_tl, y_tl, _, w, h) in detections:
        #     # Draw the detections
        #     cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)
        # cv2.imshow("Sliding Window in Progress", im)
        # cv2.waitKey(80)
        # cv2.imshow("Raw Detections before NMS", im)
        # cv2.waitKey()
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        # Perform Non Maxima Suppression
        print "-----------------------------------------------------------"
        detections = nms(detections, threshold)
        print "-----------------------------------------------------------"
        clone = img
        # Display the results after performing NMS
        detections_count = 0
        for (x_tl, y_tl, _prod, w, h) in detections:
            # Draw the detections
            # save_neg_path = '/home/gongxijun/data/neg_seg'
            # cv2.imwrite(
            #     os.path.join(save_neg_path, image_path.split('.')[0] + '_' + str(detections_count + 2) + '.jpg'),
            #     img[y_tl:y_tl + h,x_tl:x_tl+w] * 255)
            detections_count += 1
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)
            cv2.putText(clone, str(_prod), (x_tl, y_tl + 12), font, 0.5, (255, 0, 0), 1)
        cv2.imshow("Final Detections after applying NMS", clone)
        file_path = '/home/gongxijun/data/lantern-detector/demo/' + str(_num + 1) + '.jpg'
        clone *= 255
        cv2.waitKey(30)
        cv2.imwrite(file_path, clone)
        cv2.waitKey(30)
        _num += 1;
        if len(detections) > 0:
            _cnt += 1;
        print '@@@@@@@@@@@@@@@@@@@@', _num, _cnt;
