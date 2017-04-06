from  PIL import Image
from numpy import array

from django.shortcuts import render

import sys

sys.path.append("..")
# Create your views here.
from django.shortcuts import render
#from picture.models import Image
import classifyAPI
import cv2
import os


# Create your views here.
def uploadImg(request):
    if request.method == 'POST':
        # new_img = Image(
        #     _input_path=request.FILES.get('img')
        # )
        # new_img.save()
        photo = request.FILES['img']
        img = Image.open(photo)
        img = classifyAPI.classityImage(array(img))
        cv2.imwrite('/home/gongxijun/data/object-detector/object-detector/webs/image/upload_image/demo.jpg', img)
    # new_img.save()
    return render(request, '/home/gongxijun/data/object-detector/object-detector/webs/picture/templates/uploadimg.html')


def showImg(request):
    imgs = Image.objects.all()
    content = {
        'imgs': imgs,
    }
    return render(request, '/home/gongxijun/data/object-detector/object-detector/webs/picture/templates/showimg.html',
                  content)
