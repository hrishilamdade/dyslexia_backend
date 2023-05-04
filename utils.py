from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp 
import pandas as pd
import time

columns = ["LX", "LY", "RX", "RY"]
finList = []

FREQ = 100
CYCLETIME = 1/FREQ

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def get_eyes_points():
    cap = cv.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        t0 = time.perf_counter()
        time_counter = t0
        while True:
            now = time.perf_counter()
            elapsed_time = now - t0
            target_time = time_counter + CYCLETIME
            if elapsed_time < target_time:
                time.sleep(target_time - elapsed_time)

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)
                eyePositions = np.append(center_left, center_right)
                finList.append(eyePositions)
            cv.imshow('img', frame)
            time_counter += CYCLETIME
            key = cv.waitKey(1)
            if key == ord('q'):
                finDf = pd.DataFrame(data = finList, columns = columns)
                dummy_df = pd.DataFrame(columns = columns)
                for i in range(len(finDf),2000):
                    finDf.loc[i] = [0,0,0,0]

                for i in range(len(dummy_df),2000):
                    dummy_df.loc[i] = [0,0,0,0]
                # print(finDf)
                cap.release()
                cv.destroyAllWindows()
                return finDf,dummy_df

def test_with_txt(path):
    fileData = pd.read_csv(path, sep = "\t", decimal = ",")
    for i in range(len(fileData),2000):
        fileData.loc[i] = [0,0,0,0,0]
    fileData["LX"] = fileData["LX"].astype("float")
    fileData["LY"] = fileData["LY"].astype("float")
    fileData["RX"] = fileData["RX"].astype("float")
    fileData["RY"] = fileData["RY"].astype("float")
    fileData = fileData.drop(["T"], axis = 1)

    dummy_df = pd.DataFrame(columns = columns)

    for i in range(len(dummy_df),2000):
        dummy_df.loc[i] = [0,0,0,0]

    return fileData,dummy_df
