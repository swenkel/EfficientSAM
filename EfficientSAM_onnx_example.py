# Onnx export code is from
# [labelme annotation tool](https://github.com/labelmeai/efficient-sam).
# Huge thanks to Kentaro Wada.
import os
import time

import numpy as np
import onnxruntime
from PIL import Image


def predict_onnx(input_image: np.ndarray,
                 input_points: np.ndarray,
                 input_labels: np.ndarray,
                 sample_image_np: np.ndarray,
                 model_name: str):
    if 1:
        inference_session = onnxruntime.InferenceSession(
            f"weights/efficient_sam_{model_name}.onnx"
        )

        predicted_logits, predicted_iou, predicted_lowres_logits = \
            inference_session.run(
                output_names=None,
                input_feed={
                    "batched_images": input_image,
                    "batched_point_coords": input_points,
                    "batched_point_labels": input_labels,
                },
            )
    else:
        inference_session = onnxruntime.InferenceSession(
            f"weights/efficient_sam_{model_name}_encoder.onnx"
        )
        t_start = time.time()
        image_embeddings = inference_session.run(
            output_names=None,
            input_feed={
                "batched_images": input_image,
            },
        )
        print("encoder time", time.time() - t_start)

        inference_session = onnxruntime.InferenceSession(
            f"weights/efficient_sam_{model_name}_decoder.onnx"
        )
        t_start = time.time()

        predicted_logits, predicted_iou, predicted_lowres_logits = \
            inference_session.run(
                output_names=None,
                input_feed={
                    "image_embeddings": image_embeddings,
                    "batched_point_coords": input_points,
                    "batched_point_labels": input_labels,
                    "orig_im_size": np.array(
                        input_image.shape[2:],
                        dtype=np.int64
                    )
                },
            )
        print("decoder time", time.time() - t_start)
    mask = predicted_logits[0, 0, 0, :, :] >= 0
    masked_image_np = \
        sample_image_np.copy().astype(np.uint8) * mask[:, :, None]
    Image.fromarray(masked_image_np).save(
        f"figs/examples/dogs_{model_name}_onnx_mask.png"
    )


def main():
    image = np.array(Image.open("figs/examples/dogs.jpg"))
    sample_image_np = image.copy()

    input_image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    # batch_size, num_queries, num_points, 2
    input_points = np.array([[[[580, 350], [650, 350]]]], dtype=np.float32)
    # batch_size, num_queries, num_points
    input_labels = np.array([[[1, 1]]], dtype=np.float32)

    model_weights = [fn for fn in os.listdir("weights/") if "onnx" in fn]
    model_names = ["vitt", "vits"]
    model_names_available = []
    for weights_fn in model_weights:
        for model in model_names:
            if model in weights_fn and model not in model_names_available:
                model_names_available.append(model)

    for model_name in model_names_available:
        predict_onnx(input_image,
                     input_points,
                     input_labels,
                     sample_image_np,
                     model_name)


if __name__ == "__main__":
    main()
