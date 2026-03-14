import tensorflow as tf
from tensorflow.keras.models import Model

from .blazepose import BlazePose
from .blazepose_layers import BlazeBlock
from .side_encoder import get_side_encoder
from .measurement_attention_mlp import get_measurement_attention_mlp

# 측면 특징 벡터 차원
_NUM_SIDE_FEATURES = 16
# Attention 입력 = [키, BMI, 허리둘레] + side_features
_ATTENTION_INPUT_DIM = 3 + _NUM_SIDE_FEATURES  # 19


class DualBlazePose(BlazePose):
    """
    정면 + 측면 이미지를 입력으로 받는 이중 스트림 BlazePose.

    입력:
        front_image  : (batch, 256, 256, 3) - 정면 사진
        side_image   : (batch, 256, 256, 3) - 측면 사진
        scalars      : (batch, 3)           - [키(cm), BMI, 허리둘레(cm)]

    구조:
        측면 이미지 → side_encoder → 16-dim vector
        scalars + side_features → attention MLP → [32x32, 16x16, 8x8] attention maps
        정면 이미지 → BlazePose 인코더 (부모 레이어 재사용) → multi-scale features
        front features + attention maps → Regression Head → 31개 치수

    학습 데이터 준비 전까지 추론 시:
        model.load_weights(pretrained_path, by_name=True, skip_mismatch=True)
        → BlazePose 인코더/회귀 레이어는 기존 가중치 로드,
          side_encoder / attention MLP 확장 부분은 랜덤 초기화
    """

    def __init__(self, batch_size, input_shape, num_keypoints=31, num_seg_channels=10):
        # 부모 __init__에 attention_model을 넘겨야 conv14a/conv15 채널이
        # 289/290으로 설정됨 (attention 없으면 288/288로 생성됨)
        _dummy_attn = get_measurement_attention_mlp(
            batch_size=batch_size, num_input_features=2
        )
        super().__init__(
            batch_size=batch_size,
            input_shape=input_shape,
            num_keypoints=num_keypoints,
            num_seg_channels=num_seg_channels,
            attention_model=_dummy_attn,
        )

        # 측면 인코더 (별도 학습 가중치)
        self.side_encoder = get_side_encoder(
            batch_size=batch_size,
            input_shape=(input_shape[0], input_shape[1], 3),
            num_output_features=_NUM_SIDE_FEATURES,
        )

        # Attention MLP: 19차원 입력 (기존 2차원에서 확장)
        self.dual_attention_mlp = get_measurement_attention_mlp(
            batch_size=batch_size,
            num_input_features=_ATTENTION_INPUT_DIM,
        )

    # ------------------------------------------------------------------
    # build_model 오버라이드
    # ------------------------------------------------------------------
    def build_model(self, model_type="REGRESSION"):
        """
        Args:
            model_type: "REGRESSION" | "SEGMENTATION" | "REGRESSION_AND_SEGMENTATION"

        Returns:
            tf.keras.Model with inputs=[front_image, side_image, scalars]
        """
        # ── 입력 정의 ─────────────────────────────────────────────────
        input_front = tf.keras.layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 3),
            batch_size=self.batch_size,
            name="front_image",
        )
        input_side = tf.keras.layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 3),
            batch_size=self.batch_size,
            name="side_image",
        )
        input_scalars = tf.keras.layers.Input(
            shape=(3,),          # [키(cm), BMI, 허리둘레(cm)]
            batch_size=self.batch_size,
            name="scalars",
        )

        # ── 측면 인코딩 + Attention 맵 생성 ───────────────────────────
        side_features = self.side_encoder(input_side)                     # (B, 16)
        combined = tf.keras.layers.Concatenate(name="attn_input")(
            [input_scalars, side_features]
        )                                                                   # (B, 19)
        # attention_model을 레이어처럼 functional 호출
        attn_maps = self.dual_attention_mlp(combined)
        # attn_maps[0]: (B, 32, 32, 1)
        # attn_maps[1]: (B, 16, 16, 1)
        # attn_maps[2]: (B,  8,  8, 1)

        # ── 정면 인코더 (부모 클래스 레이어 재사용) ────────────────────
        # Block 1-3
        x = self.conv1(input_front)                     # 128x128x24
        x = x + self.conv2_1(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = x + self.conv2_2(x)
        y0 = tf.keras.layers.Activation("relu")(x)     # 128x128x24

        # Encoder
        y1 = self.conv3(y0)    # 64x64x48
        y2 = self.conv4(y1)    # 32x32x96
        y3 = self.conv5(y2)    # 16x16x192
        y4 = self.conv6(y3)    # 8x8x288

        # FPN decoder (heatmap branch)
        x = self.conv7a(y4) + self.conv7b(y3)
        x = self.conv8a(x) + self.conv8b(y2)
        x = self.conv9a(x) + self.conv9b(y1)
        y = self.conv10a(x) + self.conv10b(y0)
        y = self.conv11(y)
        segs = y                                        # 128x128x{num_seg_channels}

        # REGRESSION_AND_SEGMENTATION 모드: 세그멘테이션 브랜치에서 그래디언트 차단
        if model_type == "REGRESSION_AND_SEGMENTATION":
            x  = tf.stop_gradient(x)
            y2 = tf.stop_gradient(y2)
            y3 = tf.stop_gradient(y3)
            y4 = tf.stop_gradient(y4)

        # ── Regression 헤드 + Attention 주입 ─────────────────────────
        x = self.conv12a(x) + self.conv12b(y2)          # 32x32x96
        x = tf.keras.layers.Concatenate(axis=3)([x, attn_maps[0]])   # 32x32x97

        x = self.conv13a(x) + self.conv13b(y3)          # 16x16x192
        x = tf.keras.layers.Concatenate(axis=3)([x, attn_maps[1]])   # 16x16x193

        x = self.conv14a(x) + self.conv14b(y4)          # 8x8x289
        x = tf.keras.layers.Concatenate(axis=3)([x, attn_maps[2]])   # 8x8x290

        x = self.conv15(x)                              # 2x2x290
        joints = self.conv16(x)                         # 1x1x31
        output = tf.keras.layers.Reshape((self.num_keypoints,))(joints)  # (B, 31)

        # ── 모델 반환 ─────────────────────────────────────────────────
        inputs = [input_front, input_side, input_scalars]

        if model_type == "REGRESSION_AND_SEGMENTATION":
            return Model(inputs=inputs, outputs=[output, segs], name="dual_blazepose")
        elif model_type == "SEGMENTATION":
            return Model(inputs=inputs, outputs=segs, name="dual_blazepose")
        else:  # REGRESSION
            return Model(inputs=inputs, outputs=output, name="dual_blazepose")
