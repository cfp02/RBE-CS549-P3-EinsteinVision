import os
import sys
import cv2



def get_camera_video(scene_num, camera, P3Data_path):
    # ../P3Data/Sequences/scene{x}/Undist/YYY-MM-DD_HH-MM-SS-{back/front/left/right}_undistort.mp4 

    if camera not in ["front", "back", "left", "right"]:
        raise ValueError("Camera must be one of 'front', 'back', 'left', 'right'")
    if scene_num not in range(1, 14):
        raise ValueError("Scene number must be in range 1-13")

    scene_path = os.path.join(P3Data_path, "Sequences", "scene" + str(scene_num), "Undist")
    files = os.listdir(scene_path)
    for file in files:
        # Pick out the front camera video
        if file.endswith("_undistort.mp4"):
            if camera in file:
                vid_path = os.path.join(scene_path, file)
                return os.path.normpath(vid_path) 
    return None


def main():
    
    BASE_PATH = os.path.dirname(__file__)
    P3_Data_path = os.path.join(BASE_PATH, "..\P3Data")

    video_path = get_camera_video(2, "front", P3_Data_path)
    print(video_path)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit(1)

    frames: list[tuple] = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append((frame, frame_num))
            frame_num += 1
        else:
            break

    print(f"Number of frames: {len(frames)}")



if __name__ == "__main__":
    main()