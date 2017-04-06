from __future__ import unicode_literals

from django.db import models


# Create your models here.
class Image(models.Model):
    _input_path = models.FileField(upload_to='/home/gongxijun/data/upload_image');
    _output_path = models.FileField(upload_to='/home/gongxijun/data/result_image');
