import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 角度を求める関数
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

cap = cv2.VideoCapture(0)

squat_count = 0
jump_count = 0
stage_squat = "up"
stage_jump = "ground"

initial_ankle_height = None
jump_threshold_ratio = 0.12  # 立位より12%以上上昇したらジャンプと判定

def reset_counts():
    global squat_count, jump_count
    squat_count = 0
    jump_count = 0

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            # 各主要ポイント（姿勢チェック含む）
            required_points = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP
            ]

            all_visible = all(lm[p].visibility > 0.6 for p in required_points)

            # 全身が映っていない場合
            if not all_visible:
                cv2.putText(frame, "全身が映っていません", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("Workout Counter", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            # 座標取り出し
            L_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                     lm[mp_pose.PoseLandmark.LEFT_HIP].y * h]
            R_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                     lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h]

            L_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                      lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h]
            R_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                      lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h]

            L_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w,
                       lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h]
            R_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w,
                       lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h]

            # 角度計算（左・右）
            left_knee_angle = calculate_angle(L_hip, L_knee, L_ankle)
            right_knee_angle = calculate_angle(R_hip, R_knee, R_ankle)

            # ---- スクワット判定（両脚必須） ----
            if left_knee_angle < 120 and right_knee_angle < 120:
                stage_squat = "down"

            if stage_squat == "down" and left_knee_angle > 160 and right_knee_angle > 160:
                squat_count += 1
                stage_squat = "up"

            # ---- ジャンプ判定 ----
            avg_ankle_y = (L_ankle[1] + R_ankle[1]) / 2

            if initial_ankle_height is None:
                initial_ankle_height = avg_ankle_y

            jump_up_border = initial_ankle_height * (1 - jump_threshold_ratio)

            # 上方向へ移動
            if avg_ankle_y < jump_up_border and stage_jump == "ground":
                stage_jump = "air"

            # 着地
            if avg_ankle_y >= initial_ankle_height * 0.98 and stage_jump == "air":
                jump_count += 1
                stage_jump = "ground"

            # 骨格ライン（体の線）描画
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # テキスト表示
            cv2.putText(frame, f"Squats : {squat_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Jumps  : {jump_count}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, "Press R to Reset Counts", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        cv2.imshow("Workout Counter", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('r'):
            reset_counts()

cap.release()
cv2.destroyAllWindows()
