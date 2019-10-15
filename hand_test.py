import sys
sys.path.insert(0, 'python')
import cv2
import model
import util
from hand import Hand
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np
from mmdet.apis import init_detector, inference_detector


#bounding box model
model = init_detector("./src/configs/ttfnet.py", "./work_dirs/ttfnet_pretrained/latest.pth", device="cuda:0")

hand_estimation = Hand('model/hand_pose_model.pth')

test_image = 'demo/input/a.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order
canvas = copy.deepcopy(oriImg)

bounding_box = inference_detector(model, oriImg)


all_hand_peaks = []
for xmin, ymin, xmax, ymax, prob in bounding_box[0]:
        if prob < 0.4:
            continue
        
        fixed_xmin = int(xmin) - 50
        fixed_xmax = int(xmax) + 50
        fixed_ymin = int(ymin) - 50
        fixed_ymax = int(ymax) + 50

        cv2.rectangle(canvas, (fixed_xmin, fixed_ymin), (fixed_xmax, fixed_ymax), (0, 0, 255), 2)
        peaks = hand_estimation(oriImg[fixed_ymin:fixed_ymax, fixed_xmin:fixed_xmax, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+fixed_xmin)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+fixed_ymin)

        all_hand_peaks.append(peaks)

canvas = util.draw_handpose(canvas, all_hand_peaks)


cv2.imwrite("demo/output/result.jpg",canvas)
#plt.imshow(canvas[:, :, [2, 1, 0]])
#plt.axis('off')
#plt.show()
