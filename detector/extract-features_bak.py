# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from skimage import color
from skimage import transform
from skimage import io
from sklearn.externals import joblib
# To read file names
import argparse as ap
import glob
import os
from config import *
import cv2

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive images",
                        default='/home/gongxijun/data/dataset/train/pos')
    parser.add_argument('-n', "--negpath", help="Path to negative images",
                        default='/home/gongxijun/data/dataset/neg')  # '/home/gongxijun/data/dataset/neg'
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
                        default="HOG")
    args = vars(parser.parse_args())

    pos_im_path = args["pospath"]
    neg_im_path = args["negpath"]

    des_type = args["descriptor"]  # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)
    step_size = (100, 90)  # (128, 90)
    min_wdw_sz = (128, 90)
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        print im_path
        img = io.imread(im_path)
        im = color.rgb2gray(img)
        im = transform.resize(im, (128, 90));  # (128,90)
        im *= 255;
        #cv2.imshow('ss',im)
        #cv2.waitKey()
        #io.imshow(im)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        fd_name = 'pos' + os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print "Positive features saved in {}".format(pos_feat_ph)
    print "Calculating the descriptors for the negative samples and saving them"
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        image = io.imread(im_path)
        image = color.rgb2gray(image)
        print im_path
        # img = cv2.imread(im_path)
        # img = transform.resize(img, (128,90))
        # im = color.rgb2gray(img)
        # if des_type == "HOG":
        #     fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        # fd_name = 'neg' + os.path.split(im_path)[1].split(".")[0] + ".feat"
        # fd_path = os.path.join(neg_feat_ph, fd_name)
        # joblib.dump(fd, fd_path)
        # print im_path
        for y in xrange(0, image.shape[0], step_size[0]):
            for x in xrange(0, image.shape[1], step_size[1]):
                im = image[y:y + min_wdw_sz[0], x:x + min_wdw_sz[1]]
                if im.shape[0] != min_wdw_sz[0] or im.shape[1] != min_wdw_sz[1]:
                    continue
                # cv2.imshow('ss', img)
                # cv2.waitKey()
                if des_type == "HOG":
                    fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
                fd_name = 'neg' + os.path.split(im_path)[1].split(".")[0] + '_' + str(y) + '_' + str(x) + ".feat"
                fd_path = os.path.join(neg_feat_ph, fd_name)
                joblib.dump(fd, fd_path)
        print "Negative features saved in {}".format(neg_feat_ph)
    print "Completed calculating features from training images"
