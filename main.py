import cv2
import argparse
import os
import torch
from ultralytics import YOLO
import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")

    parser.add_argument("--mode",
                        choices=["cam","video"],
                        default="cam",
                        help="Choose to open webcam or to analize a video"
                        )
    
    parser.add_argument("--webcam-resolution",
                        default=(1280,720),
                        nargs=2,
                        type=int
                        )
    parser.add_argument("--path",
                        default=None,
                        type=str,
                        help="Path of the video you want to open"
                        )
    parser.add_argument("--dev",
                        choices = ["cpu","cuda"],
                        default="cpu",
                        type=str,
                        help="choose if you want to run it with cpu or gpu"
                        )
    args = parser.parse_args()
    return args

def det(model, frame, dev, classes=[0]):

    result = model(frame, classes=classes,device=dev)[0]
    detection = sv.Detections.from_ultralytics(result)
    return detection


def webcam(args):

    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
    )

    while True:
        ret, frame = cap.read()

        detection = det(model,frame,args.dev)

        frame = box_annotator.annotate(scene=frame,
                                       detections=detection,
                                       )
        cv2.imshow("yolov8",frame)

        if(cv2.waitKey(30) == 27):
            break
    return



def video_exists(video_path):
    if not os.path.exists(video_path):
        print(f"Errore: il file '{video_path}' non esiste!")
        return False
    return True




def video(args):

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
    )
    video_path = "people.mp4"
    if not video_exists(video_path):
        return -1
    vid = cv2.VideoCapture(video_path)

    while(True):
        success, frame =vid.read()

        detection = det(model,frame,args.dev)

        frame = box_annotator.annotate(scene=frame,
                                       detections=detection,
                                       )
        cv2.imshow("Outptut",frame)
        
        if(cv2.waitKey(30) == 27):
            break




def main():

    args = parse_arguments()
    dev = args.dev
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA non disponibile, uso CPU")
        dev = "cpu"
    if args.mode == "cam":
        webcam(args)
    else:
        video(args)
        return



if __name__ == "__main__":
    main()