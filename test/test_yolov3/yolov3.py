# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import numpy as np
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

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
'''


def yolov3_train(classes_path, anchors_path, model_path, train_annotation_path, val_annotation_path,
                 Init_Epoch=0, Freeze_Epoch=50, Freeze_batch_size=8, Freeze_lr=1e-3, UnFreeze_Epoch=100,
                 Unfreeze_lr=1e-4, Unfreeze_batch_size=4, num_workers=0, input_shape=[416, 416], Freeze_Train=True, Cuda=True):
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
    :return: 训练生成的模型保存在 logs 文件夹下面
    """
    #   是否使用Cuda
    Cuda = Cuda

    #   训练前一定要修改classes_path，使其对应自己的数据集
    classes_path = classes_path

    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    anchors_path = anchors_path
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = model_path
    # ------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    # ------------------------------------------------------#
    input_shape = input_shape

    # ----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    # ----------------------------------------------------#
    # ----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    # ----------------------------------------------------#
    Init_Epoch = Init_Epoch
    Freeze_Epoch = Freeze_Epoch
    Freeze_batch_size = Freeze_batch_size
    Freeze_lr = Freeze_lr
    # ----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    # ----------------------------------------------------#
    UnFreeze_Epoch = UnFreeze_Epoch
    Unfreeze_batch_size = Unfreeze_batch_size
    Unfreeze_lr = Unfreeze_lr
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = Freeze_Train
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0  
    # ------------------------------------------------------#
    num_workers = num_workers
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = train_annotation_path
    val_annotation_path = val_annotation_path

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
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

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    loss_history = LossHistory("logs/")

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
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

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

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

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

def yolov3_predict(input_origin_path, input_save_path, mode, model_path, classes_path, anchors_path, input_shape, cuda):
    yolo = YOLO(model_path, classes_path, anchors_path, input_shape, cuda)
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'single'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
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
            r_image = yolo.detect_image(image)
            r_image.save(os.path.join(dir_save_path, dir_origin_path.split('\\')[-1]))
            r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
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

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
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
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
# 单元测试
if __name__ == "__main__":
    classes = 'input_data/classes.txt'
    anchors = 'input_data/yolo_anchors.txt'
    model = 'input_data/yolo3_weights.pth'
    train_annotation = '2007_train.txt'
    val_annotation = '2007_val.txt'

    input_origin_path = r"F:\zhuwenwen\DeepCV\test\test_yolov4\test\1\000009.jpg"
    input_save_path = r"F:\zhuwenwen\DeepCV\test\test_yolov4\test\2"

    model_path = "logs/ep007-loss5.946-val_loss5.383.pth"
    classes_path = 'input_data/classes.txt'
    anchors_path = 'input_data/yolo_anchors.txt'
    input_shape = [416, 416]
    cuda = True

    # yolov3_train(classes_path=classes, anchors_path=anchors, model_path=model, train_annotation_path=train_annotation,
    #        val_annotation_path=val_annotation)

    yolov3_predict(input_origin_path, input_save_path, "single", model_path, classes_path, anchors_path, input_shape, cuda)