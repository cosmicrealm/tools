import cv2
import mediapipe as mp
from ultralytics import YOLO

# python -m scripts.script_mediapipe_face
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
yolo_face = YOLO("./pre_trained_model/yolov8m-face.pt")

# 遍历每个检测到的面部
landmark_points_68 = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]


def landmarkdetect(img_path):
    img = cv2.imread(img_path)
    # 转换为RGB图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = yolo_face(img_rgb)
    boxes = results[0].boxes.xywh.cpu()
    if len(boxes) > 0:
        # get x,y,w,h
        max_area = 0
        max_index = 0
        for index, box in enumerate(boxes):  # only get the first face
            _, _, w, h = box
            area = w * h
            if area > max_area:
                max_area = area
                max_index = index
        x, y, w, h = boxes[max_index]
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        scale = 0.15
        y1 = max(0, int(y1 * (1 - scale)))
        y2 = min(img.shape[0], int(y2 * (1 + scale)))
        x1 = max(0, int(x1 * (1 - scale)))
        x2 = min(img.shape[1], int(x2 * (1 + scale)))
        img_face = img_rgb[y1:y2, x1:x2]
    else:
        y1 = 0
        x1 = 0
        img_face = img_rgb
    # 使用MediaPipe进行面部landmarks检测
    results = face_mesh.process(img_face)
    # convert landmark to landmark68
    if results.multi_face_landmarks is not None:
        multi_face_landmarks_68 = []
        for face_landmarks in results.multi_face_landmarks:
            face_landmarks_68 = [face_landmarks.landmark[i] for i in landmark_points_68]
            multi_face_landmarks_68.append(face_landmarks_68)

        for face_landmarks in multi_face_landmarks_68:
            for id, lm in enumerate(face_landmarks):
                # 获取landmark的x和y坐标（注意这是归一化坐标）
                h, w, c = img_face.shape
                x, y = int(lm.x * w), int(lm.y * h)
                x += x1
                y += y1
                # 在图像上标记landmark
                cv2.circle(img, (x, y), 1, (0, 255, 0), 1)

    # for face_landmarks in results.multi_face_landmarks:
    #     for id, lm in enumerate(face_landmarks.landmark):
    #         # 获取landmark的x和y坐标（注意这是归一化坐标）
    #         h, w, c = img.shape
    #         x, y = int(lm.x * w), int(lm.y * h)
    #         # 在图像上标记landmark
    #         cv2.circle(img, (x, y), 3, (0, 255, 0), 3)
    return img


# img_path = "/mnt/hwdata/cv/users/zhangjinyang/datasets/exp/dinet_bigdata/split_video_25fps_frame/000008/0000000002.jpg"
# img_path = "./asserts/person_test.png"
img_path = "./asserts/nothing.png"
img_res = landmarkdetect(img_path)
cv2.imwrite("face_landmarks_68_1.jpg", img_res)
print("done")
