红灯笼检测：　　
　１.　使用的是hog+svm传统的机器学习物体检测.　　
　2.　模块描述:  
     ── data
│   ├── config
│   │   └── config.cfg
│   └── models  --模型  
│       ├── svm.model  
│       ├── svm.model_01.npy  
│       ├── svm.model_02.npy  
│       ├── svm.model_03.npy  
│       ├── svm.model_04.npy  
│       ├── svm.model_05.npy  
│       ├── svm.model_06.npy  
│       ├── svm.model_07.npy  
│       ├── svm.model_08.npy  
│       ├── svm.model_09.npy  
│       ├── svm.model_10.npy  
│       ├── svm.model_11.npy  
│       ├── svm.model_12.npy  
│       └── svm.model_13.npy  
├── demo  
│   ├── 14.jpg  
│   ├── 15.jpg  
│   ├── 17.jpg  
│   ├── 19.jpg  
│   ├── 1.jpg  
│   ├── 20.jpg  
│   ├── 22.jpg  
│   ├── 24.jpg  
│   ├── 25.jpg  
│   ├── 27.jpg  
│   ├── 30.jpg  
│   ├── 32.jpg   
│   ├── 33.jpg  
│   └── 7.jpg  
├── detector   
│   ├── classifyAPI.py　　web调用接口  
│   ├── classifyAPI.pyc  
│   ├── config.py 配置读取  
│   ├── config.pyc   
│   ├── extract-features_bak.py 特征打包已废弃  
│   ├── extract-features.py 特征提取,进行svm模型训练前的准备  
│   ├── __init__.pyc   
│   ├── main-classifier_bak.py　分类预测废弃   
│   ├── main-classifier.py 分类预测  
│   ├── mythread 自定义的线程池  
│   │   ├── __init__.py   
│   │   ├── __init__.pyc   
│   │   ├── job.py   
│   │   ├── job.pyc   
│   │   ├── thread_pool.py   
│   │   └── thread_pool.pyc   
│   ├── nms_.py    
│   ├── nms.py 去重框,两种算法实现但是目的相同  
│   ├── nms_.pyc   
│   ├── nms.pyc   
│   ├── test-classifier.py 分类预测已近废弃  
│   ├── train-classifier_bak.py模型训练已经废弃   
│   ├── train-classifier.py 模型训练  
│   └── webs web框  
│       ├── db.sqlite3  
│       ├── image   
│       │   └── upload_image   
│       │       └── demo.jpg   
│       ├── manage.py   
│       ├── picture   
│       │   ├── admin.py   
│       │   ├── admin.pyc   
│       │   ├── apps.py   
│       │   ├── __init__.py   
│       │   ├── __init__.pyc   
│       │   ├── job.pyc    
│       │   ├── migrations   
│       │   │   ├── 0001_initial.py   
│       │   │   ├── 0001_initial.pyc   
│       │   │   ├── 0002_auto_20170327_1123.py   
│       │   │   ├── 0002_auto_20170327_1123.pyc  
│       │   │   ├── __init__.py  
│       │   │   └── __init__.pyc   
│       │   ├── models.py   
│       │   ├── models.pyc  
│       │   ├── templates    
│       │   │   ├── showimg.html  
│       │   │   └── uploadimg.html  
│       │   ├── tests.py   
│       │   ├── thread_pool.pyc   
│       │   ├── views.py     
│       │   └── views.pyc   
│       └── webs   
│           ├── demo.py   
│           ├── __init__.py   
│           ├── __init__.pyc   
│           ├── settings.py    
│           ├── settings.pyc  
│           ├── urls.py  
│           ├── urls.pyc  
│           ├── wsgi.py  
│           └── wsgi.pyc  
├── ReadMe.md　　
└── tool　　
    ├── __init__.py
    ├── produceImage.py 通过旋转,变形制造更多的训练图像出来.  
使用简单介绍:  
   1. 首先自定义好pos和neg，也就是正样本和负样本训练集合.需要注意的是pos中的图片要尽可能的去除无关背景.同时pos和neg都需要灰度化,hog只能对2d
   的图像进行轮廓绘画.
   2. 使用extract-features.py提取特征,然后在使用train-classifier.py训练模型，值得注意的一点,svm模型中有一个参数C,这里做一个简单的说明
   C越大,划分的越细致,注意别出现过拟合,C越小,划分也就越粗,容易出现欠拟合.最后可以使用main-文件进行预测测试了.  
   3. 运行web:  
      python manage.py migrate  
      python manage.py runserver 127.0.0.1:8080  
 效果图：  
    ![image](https://github.com/gongxijun/lantern-detection/blob/master/demo/14.jpg)　　　
    ![image](https://github.com/gongxijun/lantern-detection/blob/master/demo/1.jpg)　　
    ![image](https://github.com/gongxijun/lantern-detection/blob/master/demo/7.jpg)　　　
    ![image](https://github.com/gongxijun/lantern-detection/blob/master/demo/15.jpg)　　　
    ![image](https://github.com/gongxijun/lantern-detection/blob/master/demo/17.jpg)　　
    ![image](https://github.com/gongxijun/lantern-detection/blob/master/demo/19.jpg)　　
    ![image](https://github.com/gongxijun/lantern-detection/blob/master/demo/20.jpg)　　
    
