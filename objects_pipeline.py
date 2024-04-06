import glob
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch
import json
import matplotlib.pyplot as plt

#zoe model needs timm==0.6.5

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

def main():
    model = YOLO('yolov9e-seg.pt')
    model_zoe_k = torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_k.to(DEVICE)

    K_inv = np.linalg.inv(K)


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
            yolo = model(frame)[0] #, persist=True, tracker="botsort.yaml")[0]
            if len(yolo.boxes) == 0:
                frames.append({
                    'frame': frame_num,
                    'objects': []
                })
                continue
            torch_frame = torch.Tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)/255.0
            depth = zoe.infer(torch_frame)
            # extract useful information from network outputs
            depth = depth.reshape(depth.shape[2], depth.shape[3]).detach().cpu().numpy()
            names = yolo.names
            boxes = yolo.boxes.xyxy.detach().cpu().numpy()
            classes = yolo.boxes.cls.detach().cpu().numpy()
            masks = yolo.masks.data.detach().cpu().numpy()
            
            # save images for debugging and paper
            # cv2.imwrite('frame.jpg', frame)
            # plt.imshow(depth)
            # plt.savefig('depth.jpg')
            # yolo.save('YOLOv9.jpg')
            # plt.imshow(depth[::2, ::2]*masks[0])
            # plt.savefig('depth.jpg')
            
            # calculate 3D locations of objects
            centers_img = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, np.ones(boxes.shape[0])])
            center_rays = (K_inv @ centers_img).T
            center_rays /= np.linalg.norm(center_rays, axis=1)[:, None]
            medians = np.array([
                np.median(depth[::2, ::2][masks[i].nonzero()])
                for i in range(masks.shape[0])
            ])[:, None]

            centers_world = medians * center_rays
            classes = [names[int(i)] for i in yolo.boxes.cls]
            frames.append({
                'frame': frame_num,
                'objects': [
                    {
                        'type': classes[i],
                        'location': centers_world[i].tolist(),
                        'rotation': [0, 0, 0],
                        'scaling': 1
                    }
                    for i in range(len(centers_world))
                ]
            })
        
        with open(f'scene{video_num}_assets.json', 'w') as f:
            json.dump(frames, f)


if __name__ == '__main__':
    main()