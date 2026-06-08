"""
Exports CLIP models to ONNX format for use in Scout.

Models exported:
  - CLIP ViT-B/16        (openai/clip-vit-base-patch16)
  - MetaCLIP ViT-B/16    (facebook/metaclip-b16-fullcc2.5b)
  - OpenCLIP ViT-B/16    (laion/CLIP-ViT-B-16-laion2B-s34B-b88K)
"""

import os
import torch
import numpy as np
import onnxruntime
from transformers import CLIPModel

OUTPUT_DIR = "models"

MODELS = [
    {
        "id":    "clip-vit-base-patch16",
        "hf_id": "openai/clip-vit-base-patch16",
        "dir":   "CLIP ViT-B-16",
    },
    {
        "id":    "metaclip-b16",
        "hf_id": "facebook/metaclip-b16-fullcc2.5b",
        "dir":   "MetaCLIP ViT-B-16",
    },
]



class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        features = self.model.get_image_features(pixel_values=pixel_values)
        return features / features.norm(dim=-1, keepdim=True)


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        features = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return features / features.norm(dim=-1, keepdim=True)



def export_model(hf_id: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Loading {hf_id} ...")
    model = CLIPModel.from_pretrained(hf_id)
    model.eval()

    # Image encoder
    image_path = os.path.join(out_dir, "image_encoder.onnx")
    print(f"  Exporting image encoder → {image_path}")
    image_enc = ImageEncoder(model)
    dummy_pixels = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        image_enc,
        (dummy_pixels,),
        image_path,
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "pixel_values":     {0: "batch"},
            "image_embeddings": {0: "batch"},
        },
        opset_version=18,
    )
    del image_enc

    # Text encoder
    text_path = os.path.join(out_dir, "text_encoder.onnx")
    print(f"  Exporting text encoder  → {text_path}")
    text_enc = TextEncoder(model)
    dummy_ids  = torch.zeros(1, 77, dtype=torch.long)
    dummy_mask = torch.ones(1, 77, dtype=torch.long)
    torch.onnx.export(
        text_enc,
        (dummy_ids, dummy_mask),
        text_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "input_ids":        {0: "batch"},
            "attention_mask":   {0: "batch"},
            "text_embeddings":  {0: "batch"},
        },
        opset_version=18,
    )
    del text_enc
    del model

    return image_path, text_path


def verify(image_path: str, text_path: str):
    print(f"  Verifying ...")

    img_session = onnxruntime.InferenceSession(image_path)
    img_out = img_session.run(None, {
        "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)
    })
    print(f"    Image embedding shape: {img_out[0].shape}")

    text_session = onnxruntime.InferenceSession(text_path)
    text_out = text_session.run(None, {
        "input_ids":      np.zeros((1, 77), dtype=np.int64),
        "attention_mask": np.ones((1, 77),  dtype=np.int64),
    })
    print(f"    Text  embedding shape: {text_out[0].shape}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Exporting {len(MODELS)} models to '{OUTPUT_DIR}/'")
    print("=" * 60)

    for entry in MODELS:
        print(f"\n[{entry['id']}]")
        out_dir = os.path.join(OUTPUT_DIR, entry["dir"])
        image_path, text_path = export_model(entry["hf_id"], out_dir)
        verify(image_path, text_path)
        print(f"  Done → {out_dir}/")

    print("\n" + "=" * 60)
    print("All models exported successfully.")