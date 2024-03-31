import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

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

def train_traffic_light():
    model_traffic_light_yolov8 = YOLO('yolov8n.pt') # Small model for traffic lights
    traffic_light_dataset_yaml = os.path.normpath(os.path.join(BASE_PATH, '../P3Data/Datasets/Red-Green-Yellow.v1i.yolov8/data.yaml'))
    # print(os.path.exists(traffic_light_dataset_yaml))

    traffic_light_model = model_traffic_light_yolov8.train(data=traffic_light_dataset_yaml, epochs=5)
    traffic_light_model.save(os.path.abspath(os.path.join(BASE_PATH, '../traffic_light_model2.pt')))
    return traffic_light_model

# def cut_traffic_light_boxes(boxes: list[list], frames, tl_model: YOLO):
#     # Boxes is a list of bounding boxes around traffic lights for each frame
#     # Frames is a list of frames
#     # Cut out the traffic light boxes and reclassify them

#     for i, frame in enumerate(frames):

#         bbox_images_before = []
#         bbox_images = []

#         for j, box in enumerate(boxes[i]):
#             print("Box", box)
#             # cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#             # cv2.waitKey()

#             box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
#             # inflate bounding box by 10 pixels
#             box = (box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10)
#             # Draw bounding box around traffic light
#             tl_frame = frame[box[1]:box[3], box[0]:box[2]]
#             cv2.imshow('Traffic light', cv2.cvtColor(tl_frame, cv2.COLOR_RGB2BGR))
#             cv2.waitKey()

#             tl_result = tl_model(tl_frame)[0] # Run traffic light model on cropped traffic light frame
#             # print(tl_result)
#             # Print class names of traffic light
#             classes = [tl_result.names[int(i)] for i in tl_result.boxes.cls]
#             print("Classes: ", classes)

#             # Show bounding box result from tl_model on the traffic light frame
#             for box in tl_result.boxes.xyxy.cpu().numpy():
#                 box = box.astype(int)
#                 bbox_image = tl_frame[box[1]:box[3], box[0]:box[2]]
#                 bbox_images.append(bbox_image)

#         if len(bbox_images) == 0:
#             continue
#         result_image = np.concatenate(bbox_images, axis=1)

#         cv2.imshow('Traffic light', cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR))
#         cv2.waitKey()

def cut_traffic_light_boxes(boxes: list[list], frames, tl_model: YOLO):
    # Boxes is a list of bounding boxes around traffic lights for each frame
    # Frames is a list of frames
    # Cut out the traffic light boxes and reclassify them

    print("Frames length", len(frames))
    for i, frame in enumerate(frames):
        bbox_images = []
        bbox_images_noedit = []

        for j, box in enumerate(boxes[i]):
            box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            # Inflate bounding box by k pixels
            k = 10
            box = (max(0, box[0] - k), max(0, box[1] - k), min(frame.shape[1], box[2] + k), min(frame.shape[0], box[3] + k))
            # Cut out the traffic light
            tl_frame = frame[box[1]:box[3], box[0]:box[2]]
            bbox_images_noedit.append(tl_frame)
            # Upsample frame to double size 
            tl_frame = cv2.resize(tl_frame, (2 * tl_frame.shape[1], 2 * tl_frame.shape[0]))
            
            # # Increase contrast of the image
            alpha = 1.5
            beta = -70
            tl_frame = cv2.convertScaleAbs(tl_frame, alpha=alpha, beta=beta)
            # # Decrease green
            # factor = 0.5
            # tl_frame[:,:,1] = np.clip(tl_frame[:,:,1]*factor, 0, 255)
            # hsv = cv2.cvtColor(tl_frame, cv2.COLOR_BGR2HSV)

            # # Increase the saturation by a factor, making sure to not exceed the maximum value of 255
            # factor = 1.5
            # hsv[:,:,1] = np.clip(hsv[:,:,1]*factor, 0, 255)

            # # Convert the image back to BGR color space
            # tl_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            hsv = cv2.cvtColor(tl_frame, cv2.COLOR_BGR2HSV)

            # Define range for yellow color
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # Create a mask for yellow color
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Reduce the saturation of the yellow hues by a factor
            factor = 0.2
            hsv[:,:,1] = hsv[:,:,1] * (1 - mask_yellow/255 * (1 - factor))

            # Convert the image back to BGR color space
            tl_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            bbox_images.append(tl_frame)
            

        # Concatenate and show all traffic lights in the frame
        if bbox_images and len(bbox_images) > 1 and False:
            max_height = max(image.shape[0] for image in bbox_images)
            bbox_images = [cv2.resize(image, (image.shape[1], max_height)) for image in bbox_images]

            # Concatenate and show all traffic lights in the frame
            result_image = np.concatenate(bbox_images, axis=1)
            cv2.imshow('Traffic lights', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey() == ord('q'):
                cv2.destroyAllWindows()
                return
            cv2.destroyAllWindows()

        # Run the model on each traffic light and show the results
        
        for i, bbox_image in enumerate(bbox_images):
            # cv2.imshow('Light light', cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            tl_result = tl_model(bbox_image)[0]

            # Concatonate images
            if len(tl_result.boxes) > 0:
                max_height = max(image.shape[0] for image in bbox_images)
                bbox_images = [cv2.resize(image, (image.shape[1], max_height)) for image in bbox_images]
                result_image = np.concatenate(bbox_images, axis=1)
                cv2.imshow('Traffic lights', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey()
                cv2.destroyAllWindows()


            # for box in tl_result.boxes.xyxy.cpu().numpy():
            #     box = box.astype(int)
            #     bbox_image = bbox_image[box[1]:box[3], box[0]:box[2]]
            #     cv2.destroyAllWindows()
            #     cv2.imshow('Traffic light', cv2.cvtColor(bbox_images[i], cv2.COLOR_RGB2BGR))
            #     cv2.waitKey()
            #     cv2.destroyAllWindows()


def main():
    
    zoe_bool, yolov9_bool, traffic_light_bool = False, True, True
    
    model = YOLO('yolov9e.pt') if yolov9_bool else None
    # model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True) # Could be faster?
    model_zoe_k = torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True) if zoe_bool else None

    traffic_light_model = YOLO(os.path.abspath(os.path.join(BASE_PATH, '../traffic_light_model.pt')))  if traffic_light_bool else None

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_k.to(DEVICE) if zoe_bool else None

    K_inv = np.linalg.inv(K)

    frames = []
    # video = read_video('P3Data/Sequences/scene5/Undist/2023-02-14_11-56-56-front_undistort.mp4')

    #Trafic light video 
    video = read_video('P3Data/Sequences/scene3/Undist/2023-02-14_11-49-54-front_undistort.mp4')
    # video = read_video('P3Data/Sequences/scene6/Undist/2023-03-03_15-31-56-front_undistort.mp4')

    # skip first 5 seconds plus 40 seconds
    for _ in range(int(9 * 36) + int(46 * 36)):
        next(video)
    # for _ in range(1):

    while (frame := next(video, None)) is not None:
        print('Processing frame')

        if zoe_bool:
            # frame = next(video)
            # cv2.imwrite('frame.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            torch_frame = torch.Tensor(frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)/255.0
            depth = zoe.infer(torch_frame)
            depth = depth.reshape(depth.shape[2], depth.shape[3]).detach().cpu().numpy()
        
        if yolov9_bool:
            # yolo = model.track(frame, persist=True, tracker="botsort.yaml")[0]
            yolo = model(frame)[0]
            names = yolo.names
            boxes = yolo.boxes.xyxy.cpu().numpy()

        if traffic_light_bool and False:
            traffic_light_yolo_result = traffic_light_model(frame)[0]
            traffic_light_names = traffic_light_yolo_result.names
            traffic_light_boxes = traffic_light_yolo_result.boxes.xyxy.cpu().numpy()

        if zoe_bool and yolov9_bool:
            centers_img = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, np.ones(boxes.shape[0])])
            center_rays = (K_inv @ centers_img).T
            center_rays /= np.linalg.norm(center_rays, axis=1)[:, None]
            centers_world = depth[centers_img[1].astype(int), centers_img[0].astype(int)][:, None] * center_rays
            classes = [names[int(i)] for i in yolo.boxes.cls]

        tl_boxes = []
        tl_frames = []
        tl_boxes_this_frame = []
        # Get traffic light boxes from yolov9e result so it can be cropped and reclassified
        print([names[int(i)] for i in range(len(boxes))])
        for i, box in enumerate(boxes):
            if yolo.boxes.cls[i] == 9:
                tl_boxes_this_frame.append(box)
        tl_frames.append(frame)
        tl_boxes_this_frame = np.array(tl_boxes_this_frame)
        tl_boxes.append(tl_boxes_this_frame)
        print("Len boxes", len(tl_boxes_this_frame), "---", tl_boxes_this_frame, '---', len(tl_frames))

        cut_traffic_light_boxes(tl_boxes, tl_frames, traffic_light_model)

        continue

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
    # train_traffic_light()
    main()