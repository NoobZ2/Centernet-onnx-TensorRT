Centernet Tensorrt加速
环境：
    opencv-python 4.5.1
    onnx 1.6.0
    numpy 1.19.5
    matplotlib 3.3.4
    onnxruntime 1.7.0
    protobuf 3.15.8
    pycocotools 2.0.2
    torch 1.8.1
    torchvision 0.9
    pycuda 2020.1
1.获取trt文件
      
	修改torch_onnx_tensorrt文件，修改main函数，将原本的PTH文件修改为所需要转换的PTH文件即可。
需要注意在转换TRT文件时，嵌入式设备的TensorRT版本要和执行转换的设备的TensorRT版本一致。
2.CenterNet推理加速

KeyPoints推理加速：  
	执行keypointDemo.py文件 
	如若需要对自定义关键点模型进行加速，需要修改keypointDemo.py文件中 bbox_decode函数以及show_results函数，修改方法已经添加注释。
	修改demo/utils/debugger.py 文件 修改方法参考：https://blog.csdn.net/xianquji1676/article/details/114000032
目标检测推理加速：
        执行demo.py



