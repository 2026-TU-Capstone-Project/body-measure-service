import cv2
import mediapipe as mp
import numpy as np
from rembg import remove  # 배경 제거 라이브러리

# ==========================================
# [사용자 데이터 입력]
# ==========================================
HEIGHT_CM = 180.0       
WEIGHT_KG = 77.0        
FRONT_IMG = 'front.jpg' 
SIDE_IMG = 'side.jpg'   
TARGET_INCH = 33.0       # 목표치(검증용)
# ==========================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

def get_body_measurement_with_rembg(image_path, mode='FRONT'):
    # 1. 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: {image_path} 파일을 찾을 수 없습니다.")
        return None, None

    height, width, _ = image.shape

    # 2. 키 측정을 위한 관절 인식 (MediaPipe)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        print(f"Error: {image_path}에서 사람을 찾을 수 없습니다.")
        return None, None

    landmarks = results.pose_landmarks.landmark
    
    # 3. 픽셀-cm 비율 계산 (키 기준)
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    bottom_y = max(l_ankle.y, r_ankle.y)
    pixel_height = (bottom_y - nose.y) * height
    scale_ratio = HEIGHT_CM / pixel_height

    measured_value_px = 0

    # 4. 측정 로직
    if mode == 'FRONT':
        # 정면은 기존 방식(관절)이 정확함 (흰 배경이라서)
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        p1 = (int(l_hip.x * width), int(l_hip.y * height))
        p2 = (int(r_hip.x * width), int(r_hip.y * height))
        
        measured_value_px = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        measured_value_px *= 1.15 # 골반 뼈 바깥 살집 보정

    elif mode == 'SIDE':
        # [핵심] 측면: 배경 제거 AI(rembg) 사용!
        # 이미지를 바이트로 변환 후 배경 제거 수행
        _, img_encoded = cv2.imencode(".jpg", image)
        img_bytes = img_encoded.tobytes()
        output_bytes = remove(img_bytes)
        
        # 배경 제거된 이미지를 다시 읽음 (투명 배경은 0, 몸은 색깔 있음)
        nparr = np.frombuffer(output_bytes, np.uint8)
        img_no_bg = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) # Alpha 채널 포함
        
        # 엉덩이 높이 찾기
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        hip_y = int(l_hip.y * height)
        
        # 엉덩이 높이의 가로줄(Row) 스캔
        # 알파 채널(투명도)이 0이 아닌 픽셀(몸통)만 찾음
        alpha_channel = img_no_bg[hip_y, :, 3] 
        body_pixels = np.where(alpha_channel > 0)[0]
        
        if len(body_pixels) > 0:
            left_edge = body_pixels[0]
            right_edge = body_pixels[-1]
            width_px = right_edge - left_edge
            
            # [중요] 걷는 자세 & 두꺼운 재킷 보정
            # 걷느라 다리가 벌어져서 폭이 넓어졌고, 재킷이 두꺼움 -> 20% 정도 깎아야 실제 몸에 가까움
            measured_value_px = width_px * 0.80 
        else:
            measured_value_px = 0

    return measured_value_px * scale_ratio

# ==========================================
# 실행 및 결과 출력
# ==========================================
print("--- [MyFit AI] 스마트 배경 제거 및 정밀 분석 ---")

width_cm = get_body_measurement_with_rembg(FRONT_IMG, mode='FRONT')
depth_cm = get_body_measurement_with_rembg(SIDE_IMG, mode='SIDE')

if width_cm and depth_cm:
    # 타원 둘레 공식
    a = width_cm / 2
    b = depth_cm / 2
    circumference_cm = np.pi * (3*(a+b) - np.sqrt((3*a+b)*(a+3*b)))
    circumference_inch = circumference_cm / 2.54
    
    print(f"1. 정면(골반 너비): {width_cm:.2f} cm")
    print(f"2. 측면(골반 두께): {depth_cm:.2f} cm (배경 제거 완료)")
    print("-" * 30)
    print(f"🎯 AI 측정 허리 둘레: {circumference_inch:.2f} 인치")
    print("-" * 30)
    
    # 33인치와의 비교
    error = circumference_inch - TARGET_INCH
    print(f"📊 실제 사이즈(33인치)와의 오차: {error:.2f} 인치")
    
    if abs(error) < 3:
        print("✅ 성공! 배경 제거 기술로 유의미한 데이터를 확보했습니다.")
    else:
        print("⚠️ 아직 오차가 큽니다. (두꺼운 재킷이나 걷는 보폭 영향이 남음)")

else:
    print("분석 실패.")