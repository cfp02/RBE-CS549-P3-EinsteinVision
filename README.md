# RBE/CS 549 Project 3: Einstein Vision
## Environment
The recommended way of working with this repo is using Python venv.
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `venv` is not installed run:
```sh
pip install virtualenv
```
## Downloading the data
Download the P3Data folder into the root directory of this project, and make sure to unzip it. Be sure not to overwrite the existing files.

## Objects Pipeline:
The code for the objects pipeline is in the `Code/objects_pipeline.py` file. The code will run inference on all scenes, and automatically download all needed checkpoints. The code will output the results in the directory the code is run from, in the form of a JSON file for each scene.

```sh
python Code/objects_pipeline.py
```
Outputs:
- `scene1_assets.json`
- `scene2_assets.json`
- ...

## Lane Detection Pipeline:
The code for the lane detection pipeline is in the `Code/lane_detection_pipeline.py` file. This script is intended to be run from the root directory of the [CLRNet repo](https://github.com/Turoad/CLRNet/tree/7269e9d1c1c650343b6c7febb8e764be538b1aed) with all the dependencies for CLRNet setup as stated in the repo. A checkpoint must also be downloaded, in our case the [LLAMAS DLA-34 checkpoint](https://github.com/Turoad/CLRNet/releases/download/models/llamas_r18.pth.zip), and placed in the root of the CLRNet repo. Finally, the config script `Code/clr_dla34_llamas.py` and the detection script `Code/detect.py` from this project must be placed in the root of the CLRNet repo. The code will output the results in the directory the code is run from, in the form of a JSON file for each scene.

```sh
python detect.py clr_dla34_llamas.py --load_from llamas_dla34.pth
```
Outputs:
- `scene1_lanes.json`
- `scene2_lanes.json`
- ...

## Car Pose Detection Pipeline:
The code for the car pose detection is from the 3D-BoundingBox repository (https://github.com/skhadem/3D-BoundingBox). The network needs to be trained on the KITTI dataset, which needs to be downloaded, and there were no pre-trained weights available. The modified main Test script to generate output to be implemented in the repository is in 'Code/3D-BoundingBox.py,' and outputs a JSON file for each video passed in. 

```sh

## Render Pipeline:
First, the output files from the previous steps must be placed in the directory `JSONData/scene<scene_number>`, where `<scene_number>` is the number of the scene. Make sure to also create the `Output` directory in the root of this project.

To run the render pipeline, open `Code/script.blend` file with Blender. Set the scene to render in the main function of the script, and then run the script. Output is shown in the Blender console. The code will output the images in the `Output/scene<scene_number>/scene<scene_number>run<run_number>` directory.

Finally, run the following command to generate the video from the frames. The command is using scene 3, run 1, and 6 fps as example arguments:
```sh
python Code/video_stitcher.py 3 1 6
```
The final mp4 video will be saved as `Output/scene<scene_number>/scene<scene_number>run<run_number>.mp4`.