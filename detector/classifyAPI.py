# Import the required modules
# coding:utf-8
import os
import cv2
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage import color, transform
from skimage.feature import hog
from sklearn.externals import joblib
import argparse as ap
from nms import nms
from config import *
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import threading as _thread
# from mythread import job, thread_pool
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
import time
import math

# List to store the detections
mutex = _thread.Lock()  # 互斥锁
detections = []


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


def get_cpu_count():
    '''
     get number of cpu
    :return:
    '''
    return cpu_count();


def get_similarIamgeRect(*args):
    """
    获取相似图片区域.
    :param img:
    :param lefttop_x:
    :param lefttop_y:
    :return:
    """
    kargs = args[0]
    img = kargs[0];
    lefttop_x = kargs[1];
    lefttop_y = kargs[2];
    min_wdw_sz = kargs[3];
    downscale = kargs[4]
    scale = kargs[5];

    # Calculate the HOG features
    right_y = lefttop_y + min_wdw_sz[1];
    right_x = lefttop_x + min_wdw_sz[0];
    fd = hog(img[lefttop_y:right_y, lefttop_x:right_x],
             orientations,
             pixels_per_cell,
             cells_per_block,
             visualize,
             normalize)
    prod = clf.predict_proba([fd])[0][1]
    prod = round(prod, 3);
    # print '------------------'
    if prod >= 0.85:  ##提取相似度大于0.5的图像区域
        # mutex.acquire();
        detections.append((int(lefttop_x * (downscale ** scale)), int(lefttop_y * (downscale ** scale)), prod,
                           int(min_wdw_sz[0] * (downscale ** scale)),
                           int(min_wdw_sz[1] * (downscale ** scale))));
        # print  _thread.current_thread(), "Detection:: Location -> ({}, {})".format(lefttop_x, lefttop_y)
        # print _thread.current_thread(), "Scale ->  {} | Confidence Score {} \n".format(scale, prod)
        # mutex.release();


def sliding_window_pos(image, min_wdw_sz, step_size):
    """
    get win of left point
    :param image:
    :param step_size:
    :return:
    """
    for y in xrange(0, image.shape[0], step_size[1]):
        if y + min_wdw_sz[1] > image.shape[0]:
            break;
        for x in xrange(0, image.shape[1], step_size[0]):
            if x + min_wdw_sz[0] > image.shape[1]:
                break;
            yield (x, y)


# max_area = 480000;
min_wdw_sz = (90, 128)
step_size = (13, 17)
downscale = 1.5
# detections = []
# Load the classifier
clf = joblib.load(model_path)
job_manager = ThreadPoolExecutor(max_workers=get_cpu_count()>>1)  # 创建一个最大可容纳2个task的线程池


def classityImage(img=None):
    _start = time.time()
    # img = cv2.imread('/home/gongxijun/data/object-detector/11/6442798_215617200193_2.jpg')
    global detections
    detections = []
    # print os.path.join('/home/gongxijun/data/object-detector/object-detector/webs/image', filename)
    # img = cv2.imread(os.path.join('/home/gongxijun/data/object-detector/object-detector/webs/image', filename));
    # print img
    scale_size = (max(img.shape[0], img.shape[1]) / 400.)  # 最大允许1000*1000的图像
    # # scale_size = 1 if scale_size < 1 else scale_size
    # if scale_size < 0.2:
    #     scale_size = 0.5;
    img = transform.rescale(img, 1.0 / scale_size)
    imt = color.rgb2gray(img)
    # job_manager = thread_pool.JobTaskManager(job_num=30, thread_num=cpu_count() << 1)
    # The current scale of the image
    scale = 0
    # Downscale the image and iterate
    # create thread and make thread num equal cpu kernel
    _step_size = step_size
    futures = []
    for im_scaled in pyramid_gaussian(imt, downscale=downscale):
        # This list contains detections at the current scale
        # im = imt
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        _step_size = (_step_size[0] / downscale, _step_size[1] / downscale)
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for y in xrange(0, im_scaled.shape[0], step_size[1]):
            if y + min_wdw_sz[1] > im_scaled.shape[0]:
                break;
            for x in xrange(0, im_scaled.shape[1], step_size[0]):
                if x + min_wdw_sz[0] > im_scaled.shape[1]:
                    break;
                ##开启多线程.
                futures.append(job_manager.submit(get_similarIamgeRect,
                                                  (im_scaled, x, y, min_wdw_sz, downscale, scale)));

        # Move the the next scale
        scale += 1
    wait(futures)
    # print(wait(futures))
    # wait(futures)
    # job_manager._join_all();
    # del job_manager
    # job_manager.join()
    # while (job_manager.getthreadStatus()):
    #    time.sleep(0.1)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    # Perform Non Maxima Suppression
    print "-----------------------------------------------------------"
    _num_status = True;
    while _num_status and len(detections) > 0:
        detections, _num_status = nms(detections, threshold)
    print "-----------------------------------------------------------"
    print "-----------------------------------------------------------"
    clone = img
    # Display the results after performing NMS
    for (x_tl, y_tl, _prod, w, h) in detections:
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)
        cv2.putText(clone, str(_prod), (x_tl, y_tl + 12), font, 0.5, (255, 0, 0), 1)
    _end = time.time()
    print   _end - _start
    return clone[:, :, ::-1] * 255


if __name__ == '__main__':
    classityImage()
