# extract landmarks from video
import argparse
import cv2
import torch
import numpy as np
from utils import landmark_98_to_68, visualize_alignment
from awing_arch import FAN

from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Extract landmarks from video")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file",
        default="/mnt/hwdata/cv/users/zhangjinyang/datasets/videos/cloth_gen/biggift/000000004251/video_resized_1024.mp4",
    )
    parser.add_argument(
        "--output", type=str, help="Path to output file", default="result/output.mp4"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name",
        default="awing_fan",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--to68",
        action="store_false",
        help="Convert landmarks from 98 to 68 points",
    )
    parser.add_argument(
        "--save_csv_path",
        type=str,
        help="Path to save the output image",
        default="result/landmarks.csv",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
        default="/mnt/hwdata/cv/users/zhangjinyang/models/facexlib/alignment_WFLW_4HG.pth",
    )
    #
    return parser.parse_args()


def process_video(args):
    print(f"Processing video {args.video} and saving to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model = FAN(num_modules=4, num_landmarks=98, device=args.device)
    model.load_state_dict(torch.load(args.model_path)["state_dict"], strict=True)
    model.eval()
    model = model.to(args.device)
    align_net = model

    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
    )
    face_helper.clean_all()

    det_net = init_detection_model("retinaface_resnet50", half=False)

    cap = cv2.VideoCapture(args.video)
    cap_writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    os.makedirs(os.path.dirname(args.save_csv_path), exist_ok=True)
    csv_writer = open(args.save_csv_path, "w")
    # write header
    csv_writer.write(
        "frame, face_id, timestamp, confidence, success, x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33, x_34, x_35, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44, x_45, x_46, x_47, x_48, x_49, x_50, x_51, x_52, x_53, x_54, x_55, x_56, x_57, x_58, x_59, x_60, x_61, x_62, x_63, x_64, x_65, x_66, x_67, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20, y_21, y_22, y_23, y_24, y_25, y_26, y_27, y_28, y_29, y_30, y_31, y_32, y_33, y_34, y_35, y_36, y_37, y_38, y_39, y_40, y_41, y_42, y_43, y_44, y_45, y_46, y_47, y_48, y_49, y_50, y_51, y_52, y_53, y_54, y_55, y_56, y_57, y_58, y_59, y_60, y_61, y_62, y_63, y_64, y_65, y_66, y_67, AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r, AU01_c, AU02_c, AU04_c, AU05_c, AU06_c, AU07_c, AU09_c, AU10_c, AU12_c, AU14_c, AU15_c, AU17_c, AU20_c, AU23_c, AU25_c, AU26_c, AU28_c, AU45_c\n"
    )
    bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        bar.update(1)
        image_save = img.copy()
        # process frame
        bboxes = det_net.detect_faces(img, 0.8)
        bboxes = bboxes[0]

        bound = 40
        bboxes[0] -= bound
        bboxes[1] -= bound
        bboxes[2] += bound
        bboxes[3] += bound
        img_box = img[
            int(bboxes[1]) : int(bboxes[3]), int(bboxes[0]) : int(bboxes[2]), :
        ]

        landmarks = align_net.get_landmarks(img_box)
        if args.to68:
            landmarks = landmark_98_to_68(landmarks)
        # back landmarks to original image
        landmarks[:, 0] += bboxes[0]
        landmarks[:, 1] += bboxes[1]
        img_vis = visualize_alignment(image_save, [landmarks])
        cap_writer.write(img_vis)

        # write landmarks to csv
        csv_writer.write(f"{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, 0, 0, 0, 1,")
        for lm in landmarks:
            csv_writer.write(f"{lm[0]:.2f},")
        for lm in landmarks:
            csv_writer.write(f"{lm[1]:.2f},")
        csv_writer.write("\n")
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > 100:
            break

        # break
    cap.release()
    cap_writer.release()
    csv_writer.close()


if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    process_video(args)
