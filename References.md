# Einstein Vision: Phase 1
Blake Bruell and Cole Parks

## Models and Frameworks Used
Our team is using the following models and frameworks to build our project:
- Ultralytrics: Framework for running Yolo models
- YoloV9: Object detection model
- ZoeDepth: Metrix depth estimation based on MiDaS
- ClrNet: Lane detection model
- 3D Bounding Box: Model for estimating 3D bounding boxes of vehicles
- YoloV8: Object detection model, fine-tuned to detect traffic light color
- YoloPv2: Panoptic segmentation model
- Blender: 3D rendering software

## How We Are Using These Models
### Ultralytics
The Ultralytics framework made it easy to download pretrained checkpoints of both YoloV8, and YoloV9. We are using YoloV9 for our object detection model. The framework also provides a simple API for running the model on images and videos, and adds a layer to track objects across frames.

### YoloV9
YoloV9 is a state-of-the-art object detection model that is capable of detecting objects in real-time. We are using this model to detect objects in our video feed, which is done by passing a model-id (`yolov9e.pt`) to the Ultralytics framework.

### ZoeDepth
ZoeDepth is a depth estimation model that is based on MiDaS. We are using this model to estimate the depth of objects in our video feed. ZoeDepth was used by loading the model off of torch hub with a pretrained checkpoint, via
```py
import torch
model_zoe_k = torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True)
```
To get the 3d position of an object, we simply take the ray direction as given by the object bounding box center, normalize it, and multiply it by the depth value.

### ClrNet
ClrNet is a lane detection model that is capable of detecting lanes in real-time. We are using this model to detect lanes in our video feed. We take the lane outputs and assume that they lie on the ground plane, and recover the 3d position using a basic inverse projection.

### 3D Bounding Box
The 3D Bounding Box model is used to estimate the 3D bounding boxes of vehicles in our video feed. This model allows us to get a pose and location estimate for the vehicles in the video feed. We use the yaw angle of the rotation, and the 3D position of the object to render the cars in blender. The bounding boxes proved to be not that accurate overall, but the yaw angle was useful for rendering the cars in blender.

### Fine Tuned YoloV8
We fine-tuned YoloV8 to detect the color of traffic lights. We started with a pretrained YoloV8 model and fine-tuned it the cinTA_v2 dataset. We then used this model to detect the color of traffic lights in our video feed. Results were not very accurate, but we were able to get some detections.

### YoloPv2
YoloPv2 is a panoptic segmentation model that is capable of segmenting objects in real-time. We are using this model to segment objects in our video feed. We are using the model to detect the drivable area, which will be used to later find arrows on the road.

### Blender
Our pipeline for blender includes outputting all detection data to a json file, which is then read by a blender script. The script reads the json file and renders the objects in 3D space. We then render out frames from the blender feed.