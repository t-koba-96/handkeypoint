import sys
sys.path.insert(0, 'python')
import cv2
import os
import model
import util
from hand import Hand
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np
from mmdet.apis import init_detector, inference_detector
from argparse import ArgumentParser
import time

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument("mode", type=str, help="handpose or openpose")
    argparser.add_argument("video_name", type=str, help="specify the video_name (e.g. test.mp4) or webcam")
    argparser.add_argument("--config_file", type=str, help="specify the config_file", default="./src/configs/ttfnet.py")
    argparser.add_argument("--weight_file", type=str, help="specify the checkpoint_file", default="./work_dirs/ttfnet_pretrained/latest.pth")
    
    #openpose weight
    argparser.add_argument("--body_weight_path", type=str, help="specify the weight_file", default="./model/body_pose_model.pth")
    argparser.add_argument("--hand_weight_path", type=str, help="specify the weight_file", default="./model/hand_pose_model.pth")
   
    return argparser.parse_args()


def main(webcamera = False):
    args = get_option()
    video_name = args.video_name
    config_file = args.config_file
    weight_file = args.weight_file

    #bounding box model
    model = init_detector(config_file, weight_file, device="cuda:0")
    hand_estimation = Hand('model/hand_pose_model.pth')
    body_estimation = Body('model/body_pose_model.pth')

    if video_name == "webcam":
        video = cv2.VideoCapture(0)

    else:
        video = cv2.VideoCapture(os.path.join('demo','input',video_name))
    
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc("m","p","4","v")
    new_video = cv2.VideoWriter(os.path.join('demo','output',video_name), fourcc, fps, (width, height))


    i=0
    start = time.perf_counter()
    while True:
        ret, frame = video.read()

        if ret:

            canvas = copy.deepcopy(frame)

            if args.mode == "openpose":
                candidate, subset = body_estimation(frame)
                canvas = util.draw_bodypose(canvas, candidate, subset)
                # detect hand
                hands_list = util.handDetect(candidate, subset, frame)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # if is_left:
                        # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
                        # plt.show()
                    peaks = hand_estimation(frame[y:y+w, x:x+w, :])
                    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                    # else:
                    #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
                    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
                    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                    #     print(peaks)
                    all_hand_peaks.append(peaks)

            elif args.mode == "handpose":
                bounding_box = inference_detector(model, frame)
                all_hand_peaks = []
                for xmin, ymin, xmax, ymax, prob in bounding_box[0]:
                    if prob < 0.5:
                        continue
                
                    fixed_xmin = int(xmin) - 50
                    if fixed_xmin <=0:
                        fixed_xmin = 1
                    fixed_xmax = int(xmax) + 50
                    if fixed_xmax >= width:
                        fixed_xmax = width - 1
                    fixed_ymin = int(ymin) - 50
                    if fixed_ymin <=0:
                        fixed_ymin = 1
                    fixed_ymax = int(ymax) + 50
                    if fixed_ymax >= height:
                        fixed_ymax = height - 1

                    cv2.rectangle(canvas, (fixed_xmin, fixed_ymin), (fixed_xmax, fixed_ymax), (0, 0, 255), 2)
                    peaks = hand_estimation(frame[fixed_ymin:fixed_ymax, fixed_xmin:fixed_xmax, :])
                    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+fixed_xmin)
                    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+fixed_ymin)
                    all_hand_peaks.append(peaks)

            if all_hand_peaks:
                 canvas = util.draw_handpose(canvas, all_hand_peaks)
                 canvas = cv2.resize(canvas,(width, height))
            
            new_video.write(canvas)

            i = i + 1
            print(i)

        else:
            break

    end = time.perf_counter()
    print (end-start)

    new_video.release()
    return


if __name__ == "__main__":
    main()