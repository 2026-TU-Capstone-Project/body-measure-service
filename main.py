import cv2
import mediapipe as mp
import math
import json

# ==========================================
# [설정] 사용자 정보 및 보정 상수
# ==========================================
USER_HEIGHT_CM = 179.0  # 사용자 실제 키
IMAGE_FILE = 'test.jpg' # 분석할 파일명

# ⭐ 보정 계수 (Calibration Factor)
# 코~발목 길이는 실제 키의 약 88~89%로 가정
HEIGHT_CORRECTION_RATIO = 0.89 

# MediaPipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# 거리 계산 함수 (Point 객체 or Landmark 객체 모두 처리 가능하도록 수정)
def calculate_distance(p1, p2):
    # p1, p2가 landmark 객체일 경우 x, y 속성 사용, 딕셔너리나 객체일 경우 처리
    x1 = p1.x if hasattr(p1, 'x') else p1['x']
    y1 = p1.y if hasattr(p1, 'y') else p1['y']
    x2 = p2.x if hasattr(p2, 'x') else p2['x']
    y2 = p2.y if hasattr(p2, 'y') else p2['y']
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# AI 모델 시작
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    image = cv2.imread(IMAGE_FILE)
    if image is None:
        print(f"Error: '{IMAGE_FILE}' 파일을 찾을 수 없습니다.")
        exit()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # -------------------------------------------------
        # 1. 주요 관절 좌표 가져오기
        # -------------------------------------------------
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        
        # 어깨
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # 팔꿈치
        l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # 손목
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # 골반 (Hip)
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 무릎
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # 발목
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # -------------------------------------------------
        # 2. 픽셀 -> cm 변환 비율 계산 (보정 로직)
        # -------------------------------------------------
        # 발목 중앙 계산
        mid_ankle_x = (l_ankle.x + r_ankle.x) / 2
        mid_ankle_y = (l_ankle.y + r_ankle.y) / 2
        mid_ankle_point = type('Point', (), {'x': mid_ankle_x, 'y': mid_ankle_y})

        # 코~발목 길이 측정
        body_pixel_len = calculate_distance(nose, mid_ankle_point)
        
        # 실제 반영 길이 (키의 89%)
        real_height_covered_cm = USER_HEIGHT_CM * HEIGHT_CORRECTION_RATIO
        cm_per_pixel = real_height_covered_cm / body_pixel_len

        # -------------------------------------------------
        # 3. 상세 치수 측정 Logic
        # -------------------------------------------------
        
        # A. 어깨 너비 (Shoulder Width)
        shoulder_px = calculate_distance(l_shoulder, r_shoulder)
        shoulder_cm = shoulder_px * cm_per_pixel

        # B. 상체 길이 (Torso Length): 어깨 중앙 ~ 골반 중앙
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        mid_shoulder = type('Point', (), {'x': mid_shoulder_x, 'y': mid_shoulder_y})

        mid_hip_x = (l_hip.x + r_hip.x) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2
        mid_hip = type('Point', (), {'x': mid_hip_x, 'y': mid_hip_y})

        torso_px = calculate_distance(mid_shoulder, mid_hip)
        torso_cm = torso_px * cm_per_pixel

        # C. 팔 길이 (Arm Length): (어깨-팔꿈치) + (팔꿈치-손목)
        # 왼쪽, 오른쪽 평균값 사용
        l_arm_px = calculate_distance(l_shoulder, l_elbow) + calculate_distance(l_elbow, l_wrist)
        r_arm_px = calculate_distance(r_shoulder, r_elbow) + calculate_distance(r_elbow, r_wrist)
        avg_arm_px = (l_arm_px + r_arm_px) / 2
        arm_cm = avg_arm_px * cm_per_pixel

        # D. 다리 길이 (Leg Length): (골반-무릎) + (무릎-발목)
        # 왼쪽, 오른쪽 평균값 사용 (골반부터 발목뼈까지. 인심(Inseam)과는 다름)
        l_leg_px = calculate_distance(l_hip, l_knee) + calculate_distance(l_knee, l_ankle)
        r_leg_px = calculate_distance(r_hip, r_knee) + calculate_distance(r_knee, r_ankle)
        avg_leg_px = (l_leg_px + r_leg_px) / 2
        leg_cm = avg_leg_px * cm_per_pixel


        # -------------------------------------------------
        # 4. JSON 출력
        # -------------------------------------------------
        result_data = {
            "status": "success",
            "user_info": {
                "height_ref_cm": USER_HEIGHT_CM
            },
            "measurements": {
                "shoulder_width_cm": round(shoulder_cm, 1),
                "upper_body_length_cm": round(torso_cm, 1), # 상체
                "arm_length_cm": round(arm_cm, 1),          # 팔
                "leg_length_cm": round(leg_cm, 1)           # 다리
            }
        }

        print("\n" + "="*30)
        print("📢 [Detailed Measurements] JSON Output:")
        print(json.dumps(result_data, indent=4, ensure_ascii=False))
        print("="*30 + "\n")

        # -------------------------------------------------
        # 5. 시각화 (Visualization)
        # -------------------------------------------------
        annotated_image = image.copy()
        
        # 뼈대 그리기 (얼굴 제외)
        body_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # 상체
            (11, 23), (12, 24), (23, 24),                     # 몸통
            (23, 25), (24, 26), (25, 27), (26, 28),           # 다리
            (27, 29), (28, 30), (29, 31), (30, 32)            # 발
        ]
        
        h, w, _ = annotated_image.shape

        # 점 찍기
        for idx, lm in enumerate(landmarks):
            if idx < 11: continue
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)

        # 선 긋기
        for start_idx, end_idx in body_connections:
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)

        # 텍스트 출력 (왼쪽 상단)
        # 가독성을 위해 배경 박스 추가 함수
        def draw_text(img, text, pos):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            color = (0, 255, 0) # Green
            cv2.putText(img, text, pos, font, scale, (0,0,0), thickness+2) # 검은 테두리
            cv2.putText(img, text, pos, font, scale, color, thickness)

        draw_text(annotated_image, f"Height Ref: {USER_HEIGHT_CM}cm", (20, 40))
        draw_text(annotated_image, f"Shoulder: {round(shoulder_cm, 1)}cm", (20, 70))
        draw_text(annotated_image, f"Torso: {round(torso_cm, 1)}cm", (20, 100))
        draw_text(annotated_image, f"Arm: {round(arm_cm, 1)}cm", (20, 130))
        draw_text(annotated_image, f"Leg: {round(leg_cm, 1)}cm", (20, 160))

        cv2.imshow('MyFit Analysis (Full Body)', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("❌ 사람을 찾지 못했습니다.")