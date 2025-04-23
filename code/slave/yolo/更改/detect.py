# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import time
import glob
import paramiko

import torch

# 当前文件路径
rootpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(rootpath)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

# -----------------
from UploadClient import upload_image
from scp import SCPClient

import yaml

# 指定YAML文件路径
yaml_file_path = '../dev.yml'
# 打开文件并读取YAML内容
with open(yaml_file_path, 'r') as yaml_file:
    # 使用yaml.safe_load()方法加载YAML内容
    yaml_data = yaml.safe_load(yaml_file)


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    # screenshot = source.lower().startswith('screen')
    # if is_url and is_file:
    #     source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # if webcam:
    #     view_img = check_imshow(warn=True)
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    #     bs = len(dataset)
    # elif screenshot:
    #     dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # else:
    a = True
    while a:
        # a = False
        if len(glob.glob(source)) == 0:
            continue
        # print(glob.glob(source))
        path_list = glob.glob(source)
        time.sleep(0.1)
        modified_list = [s[:-5] + ("1" if s[-5] == "0" else "0") + s[-4:] for s in path_list]

        print(len(path_list), ' 等待检测的图像...')
        dataset = LoadImages(path_list, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # vid_path, vid_writer = [None] * bs, [None] * bs # 视频

        # Run inference
        begin = time.time()
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # 检测结果
        det_results = []
        for path, im, im0s, vid_cap, s in dataset:
            pos_num = path.split('\\')[-1].split('pos')[-1][0]  # 获取图片位置编号
            img2paint_path = f'/home/nvidia/demo/slave/output/Camera/pos' + str(pos_num) + '/gc0.png'   # 需要绘制检测框的图片路径
            img2paint = cv2.imread(img2paint_path)
            
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Define the path for the CSV file
            csv_path = save_dir / 'predictions.csv'

            # Create or append to the CSV file
            def write_to_csv(image_name, prediction, confidence):
                data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not csv_path.is_file():
                        writer.writeheader()
                    writer.writerow(data)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                # if webcam:  # batch_size >= 1
                #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
                #     s += f'{i}: '
                # else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name[:-5]) + ("1" if str(p.name[-5]) == "0" else "0") + p.name[
                                                                                                     -4:]  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem[0:-2]) + ("_1" if p.stem[-1] == "0" else "_0") + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                print(f"保存路径 => {save_dir}")
                print(f"原文件名称 => {p.stem}")
                s += '%gx%g ' % im.shape[2:]  # print string

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(img2paint, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        print(f"检测结果 {n} {names[int(c)]}{'s' * (n > 1)}")
                        det_results.append(f"{n} {names[int(c)]}{'s' * (n > 1)}")

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f'{names[c]}'
                        confidence = float(conf)
                        confidence_str = f'{confidence:.2f}'

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    # else:  # 'video' or 'stream'
                    #     if vid_path[i] != save_path:  # new video
                    #         vid_path[i] = save_path
                    #         if isinstance(vid_writer[i], cv2.VideoWriter):
                    #             vid_writer[i].release()  # release previous video writer
                    #         if vid_cap:  # video
                    #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    #         else:  # stream
                    #             fps, w, h = 30, im0.shape[1], im0.shape[0]
                    #         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    #         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    #     vid_writer[i].write(im0)

            # Print time (inference-only)

            print(f"检测耗时 => {dt[1].dt * 1E3:.1f}ms")
            print(f"结果文件路径与名称 => {save_path}")
            # 打开文件
            # with open(save_path+'your_file.txt', 'r') as file:
            #     # 读取一行
            #     line = file.readline()

            #     # 如果文件不为空
            #     if line:
            #         # 获取第一个字符数组
            #         first_char_array = list(line[0])

            #         # 输出结果
            #         print("First character array:", first_char_array)
            #     else:
            #         print("File is empty")

            if len(det):
                print(f"检测结果 => {s} {'' if len(det) else '(no detections), '}")
                print(f"det_results => {det_results}")
                datas = {
                    'device_id': 'ORIN-1'
                    , 'defect_type': det_results
                    , 'detection_time': str(time.time())
                }
                request_net_start_time = time.time()
                # 只有有检测结果后才上传文件
                upload_image(save_path, datas,
                             yaml_data['server'] + "/upload")
                request_net_end_time = time.time()
                print('网络请求耗时: ', request_net_end_time - request_net_start_time)
            else:
                print("没有检测到缺陷结果")
            # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        # Print result
        end = time.time()
        print('耗时: ', end - begin)
        # 每张图像的速度
        t = tuple(x.t / seen * 1E3 for x in dt)
        LOGGER.info(f'计算速度 => %.1fms 预处理 => %.1fms 推理 => %.1fms 计算边框 => {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"{len(list(save_dir.glob('labels/*.txt')))} {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"结果图片保存到了 {colorstr('bold', save_dir)}")
            LOGGER.info(f"结果标签保存到了 {s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        # print(modified_list)
        for idx in range(len(path_list)):
            os.rename(path_list[idx], modified_list[idx])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/exp111811best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=yaml_data['yolo_detect_path'] + '/*_0.png',
                        help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default=ROOT / '../CodeRepository/paint-surface-defect-detection-gdzy/orin/slave/output/Absolute_pha/*_0.png', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[2048, 2432],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU 可用")
    else:
        print("没有 GPU 可用, 使用 CPU")

    opt = parse_opt()
    main(opt)
