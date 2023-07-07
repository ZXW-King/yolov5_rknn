import cv2
import numpy as np

from rknn.api import RKNN
import os

def test():
    rknn1 = RKNN(verbose=True)

if __name__ == '__main__':

    platform = 'rk3566'
    Width = 640
    Height = 416
    MODEL_PATH = '/media/xin/work1/npu/alike_3566/yolov5_rknn/rknn_convert_tools/convert_models/alinet-n.onnx'

    NEED_BUILD_MODEL = True
    # NEED_BUILD_MODEL = False

    # Create RKNN object
    rknn = RKNN()

    OUT_DIR = "rknn_models"
    RKNN_MODEL_PATH = 'alinet-n_255.rknn'
    if NEED_BUILD_MODEL:
        DATASET = '/media/xin/work1/npu/alike_3566/yolov5_rknn/rknn_convert_tools/single_dataset.txt'
        rknn.config(mean_values=[0,0,0], std_values=[255.0,255.0,255.0], target_platform="rk3566")
        # Load model
        print('--> Loading model')
        ret = rknn.load_onnx(MODEL_PATH)
        if ret != 0:
            print('load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset=DATASET)
        if ret != 0:
            print('build model failed.')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
        ret = rknn.export_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Export rknn model failed.')
            exit(ret)
        print('done')
        print('--> Load RKNN model: {}'.format(RKNN_MODEL_PATH))
        ret = rknn.load_rknn(RKNN_MODEL_PATH)
        print('rknn的输出', ret)
        print("done")
    else:
        ret = rknn.load_rknn(RKNN_MODEL_PATH)

    rknn.release()
