from argparse import ArgumentParser
import cv2
from mmdet.apis import init_detector, inference_detector


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument("--config_file", type=str, help="specify the config_file", default="./src/configs/ttfnet.py")
    argparser.add_argument("--checkpoint_file", type=str, help="specify the checkpoint_file", default="./work_dirs/ttfnet_pretrained/latest.pth")
    argparser.add_argument("--video", type=str, help="path to the video", default="./demo/input/test.mp4")
    argparser.add_argument("--result", type=str, help="path to the result", default="./demo/output/result.mp4") 
    return argparser.parse_args()


def main():
    args = get_option()
    
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    
    # initialize the model
    model = init_detector(config_file, checkpoint_file, device="cuda:0")
    
    video = cv2.VideoCapture(args.video)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    fourcc = cv2.VideoWriter_fourcc("m","p","4","v")
    new_video = cv2.VideoWriter(args.result, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()

        if ret:
            # inference the image
            result = inference_detector(model, frame)
            
            # The model returns the list of candidates for a bounding box
            for xmin, ymin, xmax, ymax, prob in result[0]:
                if prob < 0.5:
                    continue
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            new_video.write(frame)
        else:
            break

    new_video.release()
    return


if __name__ == "__main__":
    main()

