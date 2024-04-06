
import numpy as np
import torch
import cv2
import glob
import argparse
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
import json

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        alpha = 1.5
        beta = -70
        ori_img = cv2.convertScaleAbs(ori_img, alpha=alpha, beta=beta)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.heads.get_lanes(data)
        return data

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        return data

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def reproject(X, K, R, t):
    return K @ (R @ X + t)

K = np.array([
    [1.5947, 0, 0.6553], 
    [0, 1.6077, 0.4144], 
    [0, 0, 0.0010]]) * 1e3

def project_lane(lane, K, img):
    lane_points = lane.points * np.array([img.shape[1], img.shape[0]])
    homogenous = np.hstack([lane_points, np.ones((lane_points.shape[0], 1))])
    K_inv = np.linalg.inv(K)
    lane_points = (K_inv @ homogenous.T).T

    lambda_ = 1.5 / lane_points[:, 1]
    lane_points *= lambda_[:, None]

    # Sanity check
    # reprojected_points = reproject(lane_points.T, K, np.eye(3), np.zeros(3)[:, None])
    # reprojected_points /= reprojected_points[2]
    # sanity_check = reprojected_points[:2].T
    return lane_points.tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    detect = Detect(cfg)

    for i in range(13):
        video_num = i + 1
        video_path = glob.glob(f'P3Data/Sequences/scene{video_num}/Undist/*front*.mp4')[0]
        video = read_video(video_path)
        skip_seconds = 0
        max_frames = None
        frame_interval = 6

        frames = []
        for _ in range(int(skip_seconds * 36)):
            next(video)
        frame_num = 0
        for frame in tqdm(video):
            frame_num += 1
            if frame_num % frame_interval - 1 != 0:
                continue
            if max_frames is not None and frame_num > (max_frames + skip_seconds * 36):
                break
            data = detect.run(frame)
            lanes = data['lanes']
            lanes_proj = [project_lane(lane, K, frame) for lane in lanes]
            frames.append({
                'frame': frame_num,
                'lanes': lanes_proj
            })
            # Image output for debugging and paper
            # cv2.imwrite('frame.jpg', frame)
            # imshow_lanes(data['ori_img'], [lane.to_array(cfg) for lane in lanes], out_file="./output.jpg")
            # break
        
        with open(f'scene{video_num}_lanes.json', 'w') as f:
            json.dump(frames, f)