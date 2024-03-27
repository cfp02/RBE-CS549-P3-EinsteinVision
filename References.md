# Einstein Vision: Phase 1
Blake Bruell and Cole Parks

## Models and Frameworks Used
Our team is using the following models and frameworks to build our project:
- Ultralytrics: Framework for running Yolo models
- YoloV9: Object detection model
- ZoeDepth: Metrix depth estimation based on MiDaS
- ClrNet: Lane detection model

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
ClrNet is a lane detection model that is capable of detecting lanes in real-time. We are using this model to detect lanes in our video feed. We did not get this model working in time for this phase, but we plan to use it in the future, or potentially a more recent model.