#!/usr/bin/env python3
"""
Dual-View Body Analysis Service  (정면 + 측면)

입력: 정면 사진, 측면 사진, 키(cm), 몸무게(kg), 허리둘레(cm)
출력:
  - 상의: 총장, 어깨, 가슴
  - 하의: 허리, 총장, 허벅지

사용법:
  python service_dual.py <정면사진> --side <측면사진> --height 165 --weight 60 --waist 72

  또는 Python 모듈로:
    from service_dual import DualBodyAnalysisService
    svc = DualBodyAnalysisService()
    result = svc.predict("front.jpg", "side.jpg", height_cm=165, weight_kg=60, waist_cm=72)

NOTE:
  현재 DualBlazePose는 학습 전 상태입니다.
  기존 단일뷰 pretrained 가중치를 부분 로드(by_name, skip_mismatch)하여
  BlazePose 인코더/회귀 헤드 부분만 초기화됩니다.
  side_encoder와 확장된 attention MLP(19차원)는 랜덤 초기화 상태이므로
  측면 정보의 효과는 dual-view 데이터로 재학습 후 반영됩니다.
"""

import os
import sys
import argparse
import numpy as np
import cv2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model.model import get_model

# ──────────────────────────────────────────────
# 모델 설정
# ──────────────────────────────────────────────
_DUAL_MODEL_CONFIG = {
    "input_shape":        [256, 256, 3],
    "batch_size":         1,
    "type_backbone":      "dual_blazepose",
    "is_with_seg":        False,
}

_DEFAULT_WEIGHTS = os.path.join(
    _HERE, "blazepose_attention_0_3.2034787193590604.h5"
)

# ──────────────────────────────────────────────
# 31개 예측값 → 의류 치수 매핑 (service.py와 동일)
# ──────────────────────────────────────────────
_IDX_CERVICAL_HEIGHT = 0    # 목뒤높이
_IDX_WAIST_HEIGHT    = 3    # 허리높이  (= 하의 총장)
_IDX_HIP_HEIGHT      = 4    # 엉덩이높이
_IDX_SHOULDER_WIDTH  = 5    # 어깨너비  (= 상의 어깨)
_IDX_CHEST_GIRTH     = 8    # 가슴둘레  (= 상의 가슴)
_IDX_THIGH_GIRTH     = 12   # 허벅지둘레(= 하의 허벅지)


class DualBodyAnalysisService:
    """정면 + 측면 사진으로 상/하의 치수를 추정하는 서비스."""

    def __init__(self, model_path: str = _DEFAULT_WEIGHTS):
        self.model = get_model(_DUAL_MODEL_CONFIG)

        if not os.path.exists(model_path):
            print(
                f"[DualBodyAnalysis] 경고: 가중치 파일 없음 ({model_path})\n"
                "  → 랜덤 초기화로 실행됩니다. 정확한 결과를 위해 dual-view 학습이 필요합니다."
            )
        else:
            # BlazePose 인코더 / 회귀 헤드 부분만 로드
            # side_encoder, 확장된 attention MLP는 skip_mismatch로 건너뜀
            self.model.load_weights(model_path, by_name=True, skip_mismatch=True)
            print(
                f"[DualBodyAnalysis] 단일뷰 가중치 부분 로드 완료: {model_path}\n"
                "  → BlazePose 인코더 초기화 완료 / side_encoder는 랜덤 초기화"
            )

    # ------------------------------------------------------------------
    # 전처리
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_image(image_path: str) -> np.ndarray:
        """이미지를 256×256으로 리사이즈하고 배치 차원을 추가합니다."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        img = cv2.resize(img, (256, 256))
        return np.expand_dims(img.astype(np.float32), axis=0)  # (1, 256, 256, 3)

    @staticmethod
    def _build_scalars(height_cm: float, weight_kg: float, waist_cm: float) -> np.ndarray:
        """[키, BMI, 허리둘레] 3차원 scalar 벡터를 생성합니다."""
        bmi = weight_kg / (height_cm / 100.0) ** 2
        return np.array([[height_cm, bmi, waist_cm]], dtype=np.float32)  # (1, 3)

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------
    def predict(
        self,
        front_path: str,
        side_path:  str,
        height_cm:  float,
        weight_kg:  float,
        waist_cm:   float,
    ) -> dict:
        """
        Args:
            front_path : 정면 전신 사진 경로
            side_path  : 측면 전신 사진 경로
            height_cm  : 키 (cm)
            weight_kg  : 몸무게 (kg)
            waist_cm   : 허리둘레 (cm)

        Returns:
            {
              "top":    {"총장": float, "어깨": float, "가슴": float},
              "bottom": {"허리": float, "총장": float, "허벅지": float},
              "raw_predictions": list[float]
            }
        """
        # 1. 입력 검증
        if not (50 <= height_cm <= 250):
            raise ValueError(f"키 값이 범위를 벗어났습니다: {height_cm}cm")
        if not (20 <= weight_kg <= 300):
            raise ValueError(f"몸무게 값이 범위를 벗어났습니다: {weight_kg}kg")
        if not (40 <= waist_cm <= 200):
            raise ValueError(f"허리둘레 값이 범위를 벗어났습니다: {waist_cm}cm")

        # 2. 전처리
        front_batch  = self._preprocess_image(front_path)
        side_batch   = self._preprocess_image(side_path)
        scalar_batch = self._build_scalars(height_cm, weight_kg, waist_cm)

        # 3. 추론 (입력 순서: front, side, scalars)
        preds = self.model.predict(
            [front_batch, side_batch, scalar_batch], verbose=0
        )
        preds = preds[0]  # (31,)

        # 4. 치수 계산
        cervical_height = float(preds[_IDX_CERVICAL_HEIGHT])
        waist_height    = float(preds[_IDX_WAIST_HEIGHT])
        hip_height      = float(preds[_IDX_HIP_HEIGHT])
        shoulder_width  = float(preds[_IDX_SHOULDER_WIDTH])
        chest_girth     = float(preds[_IDX_CHEST_GIRTH])
        thigh_girth     = float(preds[_IDX_THIGH_GIRTH])

        top_length    = cervical_height - hip_height
        bottom_length = waist_height

        return {
            "top": {
                "총장": round(top_length,    1),
                "어깨": round(shoulder_width, 1),
                "가슴": round(chest_girth,    1),
            },
            "bottom": {
                "허리":   round(waist_cm,      1),
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
    print("=" * 38)
    print("  신체 분석 결과 (Dual-View Analysis)")
    print("=" * 38)
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
        description="정면 + 측면 사진으로 의류 치수를 추정합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python service_dual.py front.jpg --side side.jpg --height 165 --weight 60 --waist 72
  python service_dual.py front.jpg --side side.jpg --height 175 --weight 75 --waist 82 --raw
        """,
    )
    parser.add_argument("front_path",           help="정면 전신 사진 경로 (.jpg / .png)")
    parser.add_argument("--side",   required=True, metavar="PATH", help="측면 전신 사진 경로")
    parser.add_argument("--height", required=True, type=float, metavar="CM",  help="키 (cm)")
    parser.add_argument("--weight", required=True, type=float, metavar="KG",  help="몸무게 (kg)")
    parser.add_argument("--waist",  required=True, type=float, metavar="CM",  help="허리둘레 (cm)")
    parser.add_argument("--model",  default=_DEFAULT_WEIGHTS,                 help="가중치 파일 경로")
    parser.add_argument("--raw",    action="store_true",                       help="31개 원본 예측값 출력")

    args = parser.parse_args()

    service = DualBodyAnalysisService(model_path=args.model)
    result  = service.predict(
        front_path=args.front_path,
        side_path=args.side,
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
