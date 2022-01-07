# ./venv/Scripts/python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 18:59
# @Author  : Stephen(Wenwen Zhu)
# @FileName: predict.py
# @Software: PyCharm
# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch


def yolov4_train(classes_path, anchors_path, model_path, train_annotation_path, val_annotation_path,
                 Init_Epoch=0, Freeze_Epoch=50, Freeze_batch_size=8, Freeze_lr=1e-3, UnFreeze_Epoch=100,
                 Unfreeze_batch_size=4, Unfreeze_lr=1e-4, num_workers=0, input_shape=[416, 416], Freeze_Train=True,
                 Cuda=True, mosaic=False, Cosine_lr=False,
                 label_smoothing=0):
    """
    :param classes_path: 对应自己的数据集类别的路径
    :param anchors_path: 代表先验框对应的txt文件，一般不修改
    :param model_path: 预训练模型路径
    :param train_annotation_path: 获得图片的路径
    :param val_annotation_path: 获得图片的标签
    :param Init_Epoch: 冻结阶段初始 epoch
    :param Freeze_Epoch: 冻结阶段结束 epoch
    :param Freeze_batch_size: 冻结阶段 batch_size
    :param Freeze_lr: 冻结阶段学习率
    :param UnFreeze_Epoch: 解冻结阶段 epoch
    :param Unfreeze_batch_size: 解冻结阶段 batch_size
    :param Unfreeze_lr: 解冻结阶段学习率
    :param num_workers: 设置是否使用多线程读取数据，默认为0
    :param input_shape: 输入的shape大小，一定要是32的倍数
    :param Freeze_Train: 是否进行冻结训练，默认先冻结主干训练后解冻训练
    :param Cuda: 是否使用gpu训练
    :param mosaic: 是否使用mosaic马赛克数据增强 True or False，所以默认为False
    :param Cosine_lr: 是否使用弦退火学习率 True or False
    :param label_smoothing: 标签平滑处理，降低过拟合
    :return: 训练生成的模型保存在 logs 文件夹下面
    """
    # anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # 冻结阶段训练参数 | 此时模型的主干被冻结了，特征提取网络不发生改变 | 占用的显存较小，仅对网络进行微调
    Init_Epoch = Init_Epoch
    Freeze_Epoch = Freeze_Epoch
    Freeze_batch_size = Freeze_batch_size
    Freeze_lr = Freeze_lr
    # 解冻阶段训练参数 | 此时模型的主干不被冻结了，特征提取网络会发生改变 | 占用的显存较大，网络所有的参数都会发生改变
    UnFreeze_Epoch = UnFreeze_Epoch
    Unfreeze_batch_size = Unfreeze_batch_size
    Unfreeze_lr = Unfreeze_lr

    # 获取classes和anchor
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)

    if model_path != '':
        # ------------------------------------------------------#
        # 权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory("logs/")

    # 读取数据集对应的txt
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ------------------------------------#
        # 冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ------------------------------------#
        #  冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()


def yolov4_predict(input_origin_path, input_save_path, mode, model_path, classes_path, anchors_path, input_shape, cuda):
    """
    :param dir_origin_path: 指定了用于检测的图片的文件夹路径
    :param dir_save_path: 指定了检测完图片的保存路径
    :param mode: 预测模式（分为单张和文件夹）
    :param model_path: 模型所在路径
    :param classes_path: 预测的类别枚举
    :param anchors_path: 代表先验框对应的txt文件，一般不修改
    :param input_shape: 输入图片的大小，必须为32的倍数
    :param cuda: 是否使用gpu
    """
    yolo = YOLO(model_path, classes_path, anchors_path, input_shape, cuda)
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = mode
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    # -------------------------------------------------------------------------#
    test_interval = 100
    # -------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = input_origin_path
    dir_save_path = input_save_path

    if mode == "single":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        import os
        img = dir_origin_path
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
        else:
            r_image = yolo.detect_image(image)[0]
            r_image.save(os.path.join(dir_save_path, dir_origin_path.split('\\')[-1]))
            r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)[0]
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")


if __name__ == '__main__':
    classes = 'input_data/classes.txt'
    anchors = 'input_data/yolo_anchors.txt'
    model = 'input_data/yolo4_weights.pth'
    train_annotation = '2007_train.txt'
    val_annotation = '2007_val.txt'

    input_origin_path = r"F:\zhuwenwen\DeepCV\test\test_yolov4\test\1"
    input_save_path = r"F:\zhuwenwen\DeepCV\test\test_yolov4\test\2"

    model_path = "logs/ep053-loss1.864-val_loss2.282.pth"
    classes_path = 'input_data/classes.txt'
    anchors_path = 'input_data/yolo_anchors.txt'
    input_shape = [608, 608]
    cuda = True

    yolov4_predict(input_origin_path, input_save_path, "dir", model_path, classes_path, anchors_path, input_shape, cuda)
    # yolov4_train(classes_path=classes, anchors_path=anchors, model_path=model, train_annotation_path=train_annotation,
    #        val_annotation_path=val_annotation)
