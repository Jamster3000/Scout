"""
Use this to convert pytorch clip models to ONNX format
"""

import argparse
import os
import torch
import onnxruntime
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL = "openai/clip-vit-base-patch16"
OUTPUT_DIR = "models"

class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, pixel_values):
        features = self.model.get_image_features(pixel_values=pixel_values)
        return features / features.norm(dim=-1, keepdim=True)

class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, input_ids, attention_mask):
        features = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return features / features.norm(dim=-1, keepdim=True)


def export(model_name: str, encoder: str = "both"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {model_name} ...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    if encoder in ("image", "both"):
        image_path = os.path.join(OUTPUT_DIR, "clip_image_encoder.onnx")
        print(f"Exporting image encoder → {image_path}")
        image_encoder = ImageEncoder(model)
        dummy_pixels = torch.zeros(1, 3, 224, 224)
        torch.onnx.export(
            image_encoder, (dummy_pixels,), image_path,
            input_names=["pixel_values"], output_names=["image_embeddings"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_embeddings": {0: "batch"}},
            opset_version=18,
        )
        del image_encoder

    if encoder in ("text", "both"):
        text_path = os.path.join(OUTPUT_DIR, "clip_text_encoder.onnx")
        print(f"Exporting text encoder → {text_path}")
        text_encoder = TextEncoder(model)
        dummy_ids  = torch.zeros(1, 77, dtype=torch.long)
        dummy_mask = torch.ones(1, 77, dtype=torch.long)
        torch.onnx.export(
            text_encoder, (dummy_ids, dummy_mask), text_path,
            input_names=["input_ids", "attention_mask"], output_names=["text_embeddings"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "text_embeddings": {0: "batch"}},
            opset_version=18,
        )
        del text_encoder

    print("\nVerifying exports with onnxruntime ...")
    if encoder in ("image", "both"):
        img_session = onnxruntime.InferenceSession(os.path.join(OUTPUT_DIR, "clip_image_encoder.onnx"))
        img_out = img_session.run(None, {"pixel_values": torch.zeros(1, 3, 224, 224).numpy()})
        print(f"  Image embedding shape : {img_out[0].shape}")

    if encoder in ("text", "both"):
        text_session = onnxruntime.InferenceSession(os.path.join(OUTPUT_DIR, "clip_text_encoder.onnx"))
        text_out = text_session.run(None, {"input_ids": torch.zeros(1, 77, dtype=torch.long).numpy(), "attention_mask": torch.ones(1, 77, dtype=torch.long).numpy()})
        print(f"  Text  embedding shape : {text_out[0].shape}")

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--encoder", choices=["image", "text", "both"], default="both")
    args = parser.parse_args()
    export(args.model, args.encoder)

