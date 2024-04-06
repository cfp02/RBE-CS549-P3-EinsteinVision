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
Download the P3Data folder into the root of the repo. Make sure not to overwrite the existing files.

## Objects Pipeline:
The code for the objects pipeline is in the `objects_pipeline.py` file. The code will run inference on all scenes, and automatically download all needed checkpoints. The code will output the results in the directory the code is run from, in the form of a JSON file for each scene.

```sh
python objects_pipeline.py
```
Outputs:
- `scene1_assets.json`
- `scene2_assets.json`
- ...

## Lane Detection Pipeline:
The code for the lane detection pipeline is in the `lane_detection_pipeline.py` file. This script is intended to be run from the root directory of the [CLRNet repo](https://github.com/Turoad/CLRNet/tree/7269e9d1c1c650343b6c7febb8e764be538b1aed). A checkpoint must also be downloaded, in our case the [LLAMAS DLA-34 checkpoint](https://github.com/Turoad/CLRNet/releases/download/models/llamas_r18.pth.zip), and placed in the root of the repo. Finally, the config script `clr_dla34_llamas.py` in this repo must be placed in the root of the CLRNet repo. The code will output the results in the directory the code is run from, in the form of a JSON file for each scene.

```sh
python clr_dla34_llamas.py --load_from llamas_dla34.pth
```
Outputs:
- `scene1_lanes.json`
- `scene2_lanes.json`
- ...

## Render Pipeline:
First, the output files from the previous steps must be placed in the directory `JSONData/scene<scene_number>`, where `<scene_number>` is the number of the scene. Make sure to also create the `Output` directory in the root of the repo.

To run the render pipeline, open `Blender/script.blend` file with Blender. Set the scene to render in the main function, and then run the script. Output is shown in the Blender console. The code will output the images in the `Output/scene<scene_number>/scene<scene_number>run<run_number>` directory.

Finally, run the following command to generate the video from the frames. The command is using scene 3, run 1, and 6 fps as example arguments:
```sh
python Blender/video_stitcher.py 3 1 6
```
The final mp4 video will be saved as `Output/scene<scene_number>/scene<scene_number>run<run_number>.mp4`.