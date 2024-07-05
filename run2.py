from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker
from utils_ds.saveToDB import save_img

import argparse
import os
import time
import numpy as np
import warnings
import cv2
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import sys

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))

cudnn.benchmark = True

class VideoTracker(object):
    def __init__(self, args):
        print('Initialize DeepSORT & YOLO-V5')
        self.args = args
        self.scale = args.scale
        self.margin_ratio = args.margin_ratio
        self.frame_interval = args.frame_interval

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()

        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)
        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        self.face_detector = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Load known faces and their embeddings
        self.known_embeddings, self.known_names = self.load_known_faces('facedata')
        print('Done..')
        if self.device == 'cpu':
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

    def load_known_faces(self, data_dir):
        dataset = datasets.ImageFolder(data_dir)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=lambda x: x[0], num_workers=4)

        aligned = []
        names = []
        for x, y in loader:
            x_aligned, prob = self.face_detector(x, return_prob=True)
            if x_aligned is not None:
                print('Face detected with probability: {:8f}'.format(prob))
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])

        if aligned:
            aligned = torch.stack(aligned).to(self.device)
            embeddings = self.resnet(aligned).detach().cpu()
            return embeddings, names
        else:
            return None, None

    def recognize_face(self, img_cropped):
        img_embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device)).detach().cpu()
        dists = [(e - img_embedding).norm().item() for e in self.known_embeddings]
        min_dist_idx = torch.argmin(torch.tensor(dists))
        return self.known_names[min_dist_idx], dists[min_dist_idx]

    def __enter__(self):
        if self.args.cam != -1:
            print('Camera ...')
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            assert os.path.isfile(self.args.input_path), "Path error"
            self.vdo.open(self.args.input_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()
            print('Done. Load video file ', self.args.input_path)

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            self.save_video_path = os.path.join(self.args.save_path, "results.mp4")
            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                          self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
            print('Done. Create output file ', self.save_video_path)

        if self.args.save_txt:
            os.makedirs(self.args.save_txt, exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.vdo.release()
        self.writer.release()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_out = None
        while self.vdo.grab():
            t0 = time.time()
            _, img0 = self.vdo.retrieve()

            if idx_frame % self.args.frame_interval == 0:
                outputs, yt, st = self.image_track(img0)
                last_out = outputs
                yolo_time.append(yt)
                sort_time.append(st)
                print('Frame %d Done. Det-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))
            else:
                outputs = last_out

            t1 = time.time()
            avg_fps.append(t1 - t0)

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                img0 = self.draw_and_recognize_faces(img0, bbox_xyxy, identities)

            if self.args.display:
                cv2.imshow("test", img0)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break

            if self.args.save_path:
                self.writer.write(img0)

            if self.args.save_txt:
                with open(self.args.save_txt + str(idx_frame).zfill(4) + '.txt', 'a') as f:
                    for i in range(len(outputs)):
                        x1, y1, x2, y2, idx = outputs[i]
                        f.write('{}\t{}\t{}\t{}\t{}\n'.format(x1, y1, x2, y2, idx))

            idx_frame += 1

        print('Avg Det time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),
                                                                     sum(sort_time) / len(sort_time)))
        t_end = time.time()
        print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

    def image_track(self, im0):
        h, w, _ = im0.shape
        img = cv2.resize(im0, (w // self.scale, h // self.scale))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t1 = time.time()
        with torch.no_grad():
            boxes, confs = self.face_detector.detect(img)
        t2 = time.time()

        if boxes is not None and len(boxes):
            boxes = boxes * self.scale
            bbox_xywh = xyxy2xywh(boxes)
            bbox_xywh[:, 2:] = bbox_xywh[:, 2:] * (1 + self.margin_ratio)
            outputs = self.deepsort.update(bbox_xywh, confs, im0)
        else:
            outputs = torch.zeros((0, 5))

        t3 = time.time()
        return outputs, t2 - t1, t3 - t2

    def draw_and_recognize_faces(self, img, bbox_xyxy, identities):
        for box, identity in zip(bbox_xyxy, identities):
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            img_cropped, prob = self.face_detector(face_pil, return_prob=True)

            if img_cropped is not None:
                name, distance = self.recognize_face(img_cropped)
                label = f'{name} ({distance:.2f})'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            img = draw_boxes(img, [box], [identity])
        return img


def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='video.mp4', help='source')
    parser.add_argument('--save_path', type=str, default='output/', help='output folder')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_txt', default='output/predict/', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--display", action="store_true", default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="0")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--margin_ratio", type=int, default=0.2)
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")

    args = parser.parse_args()
    print(args)

    with VideoTracker(args) as vdo_trk:
        vdo_trk.run()
