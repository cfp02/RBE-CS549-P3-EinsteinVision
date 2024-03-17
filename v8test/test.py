from ultralytics import YOLO


model = YOLO('yolov8m.pt')


results = model.track(source='P3Data/Sequences/scene3/Undist/2023-02-14_11-49-54-front_undistort.mp4', show=True, tracker="botsort.yaml")