import glob
import numpy as np
import pickle
import scipy.io


def annotate():
    path = "./data/EgoHands/_LABELLED_SAMPLES/"
    videos = glob.glob(path + "*")
    
    width = 1280
    height = 720

    annotations = []
    for video in videos:
        images = sorted(glob.glob(video + "/*.jpg"))
        polygons = scipy.io.loadmat(video + "/polygons.mat")
        polygons = polygons["polygons"][0]
        
        assert len(images) == len(polygons)
        
        for i in range(len(images)):
            filename = images[i]
            
            bboxes = []
            labels = []

            polygon = polygons[i]
            for j in range(4):
                if len(polygon[j]) == 0 or polygon[j].shape == (1, 0):
                    continue
                bbox = np.hstack([np.min(polygon[j], axis=0), np.max(polygon[j], axis=0)])
                bboxes.append(bbox)
                labels.append(1)

            bboxes = np.array(bboxes)
            labels = np.array(labels)
                                     
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0, ))

            annotation = {
                    "filename": filename,
                    "width": width, 
                    "height": height,
                    "ann": {
                        "bboxes": bboxes.astype(np.float32),
                        "labels": labels.astype(np.int64)
                    }
            }
            annotations.append(annotation)
    
    with open("./data/EgoHands/annotations/annotations.pkl","wb") as f:
        pickle.dump(annotations, f)

    return


def main():
    annotate()
    
    return


if __name__ == "__main__":
    main()
