import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json

#zoe model needs timm==0.6.5

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

def reproject(X, K, R, t):
    return K @ (R @ X + t)

K = np.array([
    [1.5947, 0, 0.6553], 
    [0, 1.6077, 0.4144], 
    [0, 0, 0.0010]]) * 1e3

def main():
    model = YOLO('yolov9e.pt')
    # model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True) # Could be faster?
    model_zoe_k = torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_k.to(DEVICE)

    K_inv = np.linalg.inv(K)

    frames = []
    video = read_video('P3Data/Sequences/scene5/Undist/2023-02-14_11-56-56-front_undistort.mp4')
    # skip first 5 seconds
    for _ in range(int(9 * 36)):
        next(video)
    for _ in range(1):
        frame = next(video)
        cv2.imwrite('frame.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        torch_frame = torch.Tensor(frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)/255.0
        depth = zoe.infer(torch_frame)
        depth = depth.reshape(depth.shape[2], depth.shape[3]).detach().cpu().numpy()
        yolo = model.track(frame, persist=True, tracker="botsort.yaml")[0]
        names = yolo.names
        boxes = yolo.boxes.xyxy.cpu().numpy()
        centers_img = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, np.ones(boxes.shape[0])])
        center_rays = (K_inv @ centers_img).T
        center_rays /= np.linalg.norm(center_rays, axis=1)[:, None]
        centers_world = depth[centers_img[1].astype(int), centers_img[0].astype(int)][:, None] * center_rays
        classes = [names[int(i)] for i in yolo.boxes.cls]
        frames.append({
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
    
    with open('output.json', 'w') as f:
        json.dump(frames, f, indent=4)


if __name__ == '__main__':
    main()