import tensorflow as tf
from model.blazepose import BlazePose
from model.mobilenet_v3 import MobileNetV3
from model.measurement_attention_mlp import get_measurement_attention_mlp

def get_model(config):
    input_shape = config["input_shape"]
    batch_size  = config["batch_size"]
    type_backbone = config["type_backbone"]
    is_with_seg   = config.get("is_with_seg", False)
    type_attention     = config.get("type_attention", "none")
    num_category_bmi    = config.get("num_category_bmi", 10)
    num_category_height = config.get("num_category_height", 10)

    input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size)

    if type_attention == "categorical":
        attention_mlp = get_measurement_attention_mlp(
            batch_size=batch_size,
            shape_categorical_data=num_category_bmi + num_category_height,
        )
    elif type_attention == "regression":
        attention_mlp = get_measurement_attention_mlp(
            batch_size=batch_size, num_input_features=2
        )

    if type_backbone == "blazepose":
        if type_attention != "none":
            blazepose_model = BlazePose(
                batch_size=batch_size, input_shape=input_shape,
                num_keypoints=31, num_seg_channels=10,
                attention_model=attention_mlp,
            )
        else:
            blazepose_model = BlazePose(
                batch_size=batch_size, input_shape=input_shape,
                num_keypoints=31, num_seg_channels=10,
            )
        model_type = "REGRESSION_AND_SEGMENTATION" if is_with_seg else "REGRESSION"
        model = blazepose_model.build_model(model_type=model_type)

    elif type_backbone == "mbnv3":
        num_seg_channels = 10 if is_with_seg else 0
        if type_attention != "none":
            model = MobileNetV3(
                input_layer=input_layer, type="small",
                attention_model=attention_mlp,
                num_seg_channels=num_seg_channels, num_keypoints=31,
            )
        else:
            model = MobileNetV3(
                input_layer=input_layer, type="small",
                num_seg_channels=num_seg_channels, num_keypoints=31,
            )

    return model

