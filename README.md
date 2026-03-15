# Scout - Model Conversion

One-time script to export the CLIP model's to ONNX format to be compatable and usable in the Rust backend (see rust branch)

## Output
It will generate a new `models/` folder which it converts the 16-vit-clip model into a image_encoder and text_encoder

## Setup
Used with Python 3.13

`pip install torch transformers onnx onnxruntime onnxscript`

## Usage

`python convert_model.py`

Can also be more specific

```bash
python convert_model.py --encoder image
python convert_model.py --encoder text
python convert_model.py --model openai/clip-vit-base-patch32
```

Running the image encoder and text encoder is useful if hitting memory issues (I ran with 16GB and still hit memory issues)