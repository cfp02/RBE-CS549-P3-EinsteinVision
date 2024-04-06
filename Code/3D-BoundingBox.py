"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo
from ultralytics import YOLO

import os
import time

import numpy as np
import cv2
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")

parser.add_argument("--einstein", action="store_true")

parser.add_argument("--video-path", default=None,
                    help="Path to video file to run Einstein on")

parser.add_argument("--use-every", default=1, help="Use every nth frame from the video")

parser.add_argument("--poses-out", default="poses.txt",
                    help="Output file for poses. Default is poses.txt")



def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location, orient


def get_frames_from_video(video_path, use_every=1):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % use_every == 0:
            yield i, frame
        i += 1
    cap.release()


def main():
    
    EINSTEIN = False
    yolov9 = True
    show_video = True

    FLAGS = parser.parse_args()

    output_file = FLAGS.poses_out

    if FLAGS.einstein:
        EINSTEIN = True
        FLAGS.video = True
        video_path = FLAGS.video_path if FLAGS.video_path else None
        if video_path is None:
            print("Error: --einstein flag passed without --video-path")
            exit()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    if yolov9:
        yolo = YOLO("yolov9e.pt")
    else:
        yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
        yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video and not EINSTEIN:
        if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "eval/video/2011_09_26/image_2/"
            cal_dir = "eval/video/2011_09_26/"

 
    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    if EINSTEIN:
        
        FLAGS.use_every = int(FLAGS.use_every)

        if video_path is None:
            print("Error: --einstein flag passed without --video-path")
        video = get_frames_from_video(video_path, use_every = FLAGS.use_every)

        frames_for_json = []

        while(f := next(video, None)):

            start_time = time.time()

            idx, frame = f
            # skip the first x frames
            if idx < 170:
                continue
            print("Frame: ", idx)
            truth_img = frame
            img = np.copy(truth_img)
            yolo_img = np.copy(truth_img)

            if yolov9:
                output = yolo(yolo_img)[0]
                classes = np.array([output.names[int(i)] for i in output.boxes.cls])
                boxes = np.array(output.boxes.xyxy.cpu())

                Detection = namedtuple('Detection', ['detected_class', 'box_2d'])

                detections = []
                for i, box in enumerate(boxes):
                    box = box.astype(int)
                    this_box = [(box[0], box[1]), (box[2], box[3])]
                    px = 15 
                    # Make box px pixels bigger, unless it's too close to the edge
                    if this_box[0][0] > px:
                        this_box[0] = (this_box[0][0] - px, this_box[0][1])
                    if this_box[0][1] > px:
                        this_box[0] = (this_box[0][0], this_box[0][1] - px)
                    if this_box[1][0] < img.shape[1] - px:
                        this_box[1] = (this_box[1][0] + px, this_box[1][1])
                    if this_box[1][1] < img.shape[0] - px:
                        this_box[1] = (this_box[1][0], this_box[1][1] + px)

                    detections.append(Detection(classes[i], this_box))

            else:
                detections = yolo.detect(yolo_img)              


            outputs: list[tuple] = []

            for i, detection in enumerate(detections):


                if not averages.recognized_class(detection.detected_class):
                    continue

                # this is throwing when the 2d bbox is invalid
                # TODO: better check
                try:
                    detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
                except Exception as e:
                    print(e)
                    continue

                theta_ray = detectedObject.theta_ray
                input_img = detectedObject.img
                proj_matrix = detectedObject.proj_matrix
                box_2d = detection.box_2d
                detected_class = detection.detected_class
                # print("Detected class: ", detected_class)
                input_tensor = torch.zeros([1,3,224,224]).cuda()
                input_tensor[0,:,:,:] = input_img

                [orient, conf, dim] = model(input_tensor)
                orient = orient.cpu().data.numpy()[0, :, :]
                conf = conf.cpu().data.numpy()[0, :]
                dim = dim.cpu().data.numpy()[0, :]

                dim += averages.get_item(detected_class)

                argmax = np.argmax(conf)
                orient = orient[argmax, :]
                cos = orient[0]
                sin = orient[1]
                alpha = np.arctan2(sin, cos)
                alpha += angle_bins[argmax]
                alpha -= np.pi

                if FLAGS.show_yolo:
                    location, orientation = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
                else:
                    location, orientation = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

                if not FLAGS.hide_debug:
                    # Truncate to 3 decimal places
                    print('Estimated pose: ', [round(loc,3) for loc in location], round(orientation,4), "\t\t", detected_class)

                type = detected_class
                rotation = (0,0, orientation)
                scaling = 1

                outputs.append({
                    "type": type,
                    "location": location,
                    "rotation": rotation,
                    "scaling": scaling
                })


            if FLAGS.show_yolo:
                numpy_vertical = np.concatenate((truth_img, img), axis=0)
                cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
            else:
                if (not EINSTEIN) or show_video:
                    cv2.imshow('3D detections', img)


            
            frames_for_json.append({
                "frame": idx,
                "objects": outputs
            })

            if not FLAGS.hide_debug:
                print("\n")
                print('Got %s poses in %.3f seconds'%(i, time.time() - start_time))
                print('-------------')

            

            if FLAGS.video:
                if (not EINSTEIN) or show_video:
                    cv2.waitKey(1)

            if idx == 190:
                # Save this frame
                cv2.imwrite("output.jpg", img)
                # break        
            else:
                if cv2.waitKey(0) != 32: # space bar
                    exit()

        import json
        with open(output_file, 'w') as f:
            json.dump(frames_for_json, f, indent=4)

    # ORIGINAL CODE
    else:

        try:
            ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
        except:
            print("\nError: no images in %s"%img_path)
            exit()

        for img_id in ids:

            start_time = time.time()

            img_file = img_path + img_id + ".png"

            # P for each frame
            # calib_file = calib_path + id + ".txt"

            truth_img = cv2.imread(img_file)
            img = np.copy(truth_img)
            yolo_img = np.copy(truth_img)

            detections = yolo.detect(yolo_img)

            for detection in detections:

                if not averages.recognized_class(detection.detected_class):
                    continue

                # this is throwing when the 2d bbox is invalid
                # TODO: better check
                try:
                    detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
                except:
                    continue

                theta_ray = detectedObject.theta_ray
                input_img = detectedObject.img
                proj_matrix = detectedObject.proj_matrix
                box_2d = detection.box_2d
                detected_class = detection.detected_class
                # print("Detected class: ", detected_class)

                input_tensor = torch.zeros([1,3,224,224]).cuda()
                input_tensor[0,:,:,:] = input_img

                [orient, conf, dim] = model(input_tensor)
                orient = orient.cpu().data.numpy()[0, :, :]
                conf = conf.cpu().data.numpy()[0, :]
                dim = dim.cpu().data.numpy()[0, :]

                dim += averages.get_item(detected_class)

                argmax = np.argmax(conf)
                orient = orient[argmax, :]
                cos = orient[0]
                sin = orient[1]
                alpha = np.arctan2(sin, cos)
                alpha += angle_bins[argmax]
                alpha -= np.pi

                if FLAGS.show_yolo:
                    location, _ = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
                else:
                    location, _ = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

                if not FLAGS.hide_debug:
                    print('Estimated pose: %s'%location)

            if FLAGS.show_yolo:
                numpy_vertical = np.concatenate((truth_img, img), axis=0)
                cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
            else:
                cv2.imshow('3D detections', img)

            if not FLAGS.hide_debug:
                print("\n")
                print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
                print('-------------')

            if FLAGS.video:
                cv2.waitKey(1)
            else:
                if cv2.waitKey(0) != 32: # space bar
                    exit()

if __name__ == '__main__':
    # Run with --einstein to be able to pass a video file 
    main()
