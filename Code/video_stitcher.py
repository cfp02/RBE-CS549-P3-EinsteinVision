import sys
import os
import cv2

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

## This python file should take a directory where there are only png images and stitch them into a video

def stitch_images_to_video(image_folder, output_folder, video_name, fps):

    # Images saved do not have leading zeroes, so we need to sort them by the number in the filename. Files are in the format P2SceneXFrameY.png

    img_list = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(img_list, key=lambda x: int(x.split('Frame')[1].split('.')[0]))


    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(output_folder,video_name), fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    print("Saved to ", os.path.join(output_folder, video_name))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    if len(sys.argv) != 4:

        scene, run = 2, 1

        print("Usage: python video_stitcher.py <image_folder> <video_name> <fps>")
        image_folder = os.path.join(BASE_PATH, '../', 'Output', 'scene' + str(scene), 'scene' + str(scene) + 'run' + str(run))
        output_folder = os.path.abspath(os.path.join(BASE_PATH, '../', 'Output', 'scene' + str(scene)))
        video_name = 'scene' + str(scene) + 'run' + str(run) + '.mp4'
        fps = 6
    else:
        image_folder = sys.argv[1]
        video_name = sys.argv[2]
        fps = int(sys.argv[3])

    stitch_images_to_video(image_folder, output_folder, video_name, fps)