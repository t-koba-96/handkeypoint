import cv2
from mmdet.apis import init_detector, inference_detector


def main():    
    config_file = "./src/configs/ttfnet.py"
    checkpoint_file = "./work_dirs/ttfnet_pretrained/latest.pth"
    model = init_detector(config_file, checkpoint_file, device="cuda:0")
    
    image = cv2.imread("./src/images/image.jpg")
    result = inference_detector(model, image)
            
    # The model returns the list of candidates for a bounding box
    for xmin, ymin, xmax, ymax, prob in result[0]:
        if prob < 0.5:
            continue
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)

    cv2.imwrite("./src/images/result.jpg", image)
    return


if __name__ == "__main__":
    main()

