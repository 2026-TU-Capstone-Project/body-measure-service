#!/usr/bin/env python3
"""
Body Analysis Service

입력: 전신사진(정면), 키(cm), 몸무게(kg), 허리둘레(cm)
출력:
  - 상의: 총장, 어깨, 가슴
  - 하의: 허리, 총장, 허벅지

사용법:
  python service.py <이미지경로> --height 165 --weight 60 --waist 70

  또는 Python 모듈로:
    from service import BodyAnalysisService
    svc = BodyAnalysisService()
    result = svc.predict("photo.jpg", height_cm=165, weight_kg=60, waist_cm=70)
"""

import os
import sys
import argparse
import numpy as np
import cv2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

# 이 스크립트는 body_part_measurement/ 디렉터리 기준으로 실행됩니다.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model.model import get_model

# ──────────────────────────────────────────────
# 모델 설정 (학습 시 사용한 config와 동일해야 함)
# ──────────────────────────────────────────────
_MODEL_CONFIG = {
    "input_shape": [256, 256, 3],
    "batch_size": 1,
    "type_backbone": "blazepose",
    "is_with_seg": False,
    "type_attention": "regression",   # [height, bmi] 2개 입력
    "num_category_bmi": 10,
    "num_category_height": 10,
}

_DEFAULT_WEIGHTS = os.path.join(_HERE, "blazepose_attention_0_3.2034787193590604.h5")

# ──────────────────────────────────────────────
# 31개 예측값 → 의류 치수 매핑
#
# NIA21 신체 치수 데이터셋 컬럼 분석 결과:
#   index  0 : 목뒤높이   (바닥~목뒤점, cm)   → 122~156 범위
#   index  3 : 허리높이   (바닥~허리, cm)     → 90~112  범위  ← 하의 총장
#   index  4 : 엉덩이높이 (바닥~엉덩이, cm)   → 64~83   범위
#   index  5 : 어깨너비   (cm)               → 38~50   범위  ← 상의 어깨
#   index  8 : 가슴둘레   (cm)               → 80~106  범위  ← 상의 가슴
#   index 12 : 허벅지둘레 (cm)               → 50~61   범위  ← 하의 허벅지
#
#  상의 총장 = 목뒤높이 - 엉덩이높이  (목뒤점 ~ 엉덩이선 길이)
#  하의 허리 = 사용자 직접 입력값
# ──────────────────────────────────────────────
_IDX_CERVICAL_HEIGHT = 0   # 목뒤높이
_IDX_WAIST_HEIGHT    = 3   # 허리높이  (= 하의 총장)
_IDX_HIP_HEIGHT      = 4   # 엉덩이높이
_IDX_SHOULDER_WIDTH  = 5   # 어깨너비  (= 상의 어깨)
_IDX_CHEST_GIRTH     = 8   # 가슴둘레  (= 상의 가슴)
_IDX_THIGH_GIRTH     = 12  # 허벅지둘레(= 하의 허벅지)


class BodyAnalysisService:
    """전신 사진 + 체형 정보로 상/하의 치수를 추정하는 서비스."""

    def __init__(self, model_path: str = _DEFAULT_WEIGHTS):
        self.model = get_model(_MODEL_CONFIG)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 가중치 파일을 찾을 수 없습니다: {model_path}\n"
                "blazepose_attention_0_3.2034787193590604.h5 파일이 같은 폴더에 있어야 합니다."
            )
        self.model.load_weights(model_path)
        print(f"[BodyAnalysis] 모델 로드 완료: {model_path}")

    # ------------------------------------------------------------------
    # 전처리
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_image(image_path: str) -> np.ndarray:
        """이미지를 256×256으로 리사이즈하고 배치 차원을 추가합니다."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        h, w = _MODEL_CONFIG["input_shape"][0], _MODEL_CONFIG["input_shape"][1]
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32)
        return np.expand_dims(img, axis=0)  # (1, 256, 256, 3)

    @staticmethod
    def _build_attention(height_cm: float, weight_kg: float) -> np.ndarray:
        """BMI를 계산하고 Attention 입력 벡터를 만듭니다."""
        bmi = weight_kg / (height_cm / 100.0) ** 2
        return np.array([[height_cm, bmi]], dtype=np.float32)  # (1, 2)

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------
    def predict(
        self,
        image_path: str,
        height_cm: float,
        weight_kg: float,
        waist_cm: float,
    ) -> dict:
        """
        Args:
            image_path : 전신 정면 사진 경로
            height_cm  : 키 (cm)
            weight_kg  : 몸무게 (kg)
            waist_cm   : 허리둘레 (cm) — 사용자 직접 입력

        Returns:
            {
              "top":    {"총장": float, "어깨": float, "가슴": float},
              "bottom": {"허리": float, "총장": float, "허벅지": float},
              "raw_predictions": list[float]  # 31개 원본 예측값 (참고용)
            }
        """
        # 1. 입력 검증
        if not (50 <= height_cm <= 250):
            raise ValueError(f"키 값이 범위를 벗어났습니다: {height_cm}cm (50~250 허용)")
        if not (20 <= weight_kg <= 300):
            raise ValueError(f"몸무게 값이 범위를 벗어났습니다: {weight_kg}kg (20~300 허용)")
        if not (40 <= waist_cm <= 200):
            raise ValueError(f"허리둘레 값이 범위를 벗어났습니다: {waist_cm}cm (40~200 허용)")

        # 2. 전처리
        img_batch       = self._preprocess_image(image_path)
        attention_batch = self._build_attention(height_cm, weight_kg)

        # 3. 모델 추론
        preds = self.model.predict([img_batch, attention_batch], verbose=0)
        preds = preds[0]  # (31,)

        # 4. 치수 계산
        cervical_height = float(preds[_IDX_CERVICAL_HEIGHT])  # 목뒤높이
        waist_height    = float(preds[_IDX_WAIST_HEIGHT])     # 허리높이 (= 하의 총장)
        hip_height      = float(preds[_IDX_HIP_HEIGHT])       # 엉덩이높이
        shoulder_width  = float(preds[_IDX_SHOULDER_WIDTH])   # 어깨너비
        chest_girth     = float(preds[_IDX_CHEST_GIRTH])      # 가슴둘레
        thigh_girth     = float(preds[_IDX_THIGH_GIRTH])      # 허벅지둘레

        top_length    = cervical_height - hip_height  # 상의 총장 (목뒤~엉덩이선)
        bottom_length = waist_height                  # 하의 총장 (허리~바닥)

        return {
            "top": {
                "총장":  round(top_length,    1),
                "어깨":  round(shoulder_width, 1),
                "가슴":  round(chest_girth,    1),
            },
            "bottom": {
                "허리":   round(waist_cm,      1),  # 사용자 입력값 그대로 사용
                "총장":   round(bottom_length, 1),
                "허벅지": round(thigh_girth,   1),
            },
            "raw_predictions": [round(float(v), 2) for v in preds],
        }


# ──────────────────────────────────────────────
# CLI 인터페이스
# ──────────────────────────────────────────────
def _print_result(result: dict) -> None:
    print()
    print("=" * 35)
    print("    신체 분석 결과 (Body Analysis)")
    print("=" * 35)
    print()
    print("[상의 (Top)]")
    print(f"  총장   : {result['top']['총장']:>6.1f} cm")
    print(f"  어깨   : {result['top']['어깨']:>6.1f} cm")
    print(f"  가슴   : {result['top']['가슴']:>6.1f} cm")
    print()
    print("[하의 (Bottom)]")
    print(f"  허리   : {result['bottom']['허리']:>6.1f} cm")
    print(f"  총장   : {result['bottom']['총장']:>6.1f} cm")
    print(f"  허벅지 : {result['bottom']['허벅지']:>6.1f} cm")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="전신 사진과 기본 체형 정보로 의류 치수를 추정합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python service.py photo.jpg --height 165 --weight 60 --waist 72
  python service.py photo.jpg --height 175 --weight 75 --waist 82 --model custom.h5
        """,
    )
    parser.add_argument("image_path",         help="전신 정면 사진 경로 (.jpg / .png)")
    parser.add_argument("--height", type=float, required=True, metavar="CM",   help="키 (cm)")
    parser.add_argument("--weight", type=float, required=True, metavar="KG",   help="몸무게 (kg)")
    parser.add_argument("--waist",  type=float, required=True, metavar="CM",   help="허리둘레 (cm)")
    parser.add_argument("--model",  type=str,   default=_DEFAULT_WEIGHTS,      help="모델 가중치 파일 경로")
    parser.add_argument("--raw",    action="store_true",                        help="31개 원본 예측값도 출력")

    args = parser.parse_args()

    service = BodyAnalysisService(model_path=args.model)
    result  = service.predict(
        image_path=args.image_path,
        height_cm=args.height,
        weight_kg=args.weight,
        waist_cm=args.waist,
    )

    _print_result(result)

    if args.raw:
        print("[원본 예측값 (31개)]")
        for i, v in enumerate(result["raw_predictions"]):
            print(f"  [{i:2d}] {v:.2f} cm")
        print()


if __name__ == "__main__":
    main()
