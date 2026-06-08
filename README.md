# Scout - Model Conversion

This is a one time developer script that I used to convert the models to ONNX format.

The current builds of CLIP models on huggingface are used with pytorch. Given the use of rust, I wrote this python script to convert those models to ONNX which is can then be used with rust (through the `ort` crate)

## Output
It will create a `models` folder which downloads the ViT-B/16 and metaclip models from huggingface and converts them into `image_encoder` and `text_encoder` in their respective folders inside `models`

## Setup
Used with Python 3.13

`pip install torch transformers onnx onnxruntime onnxscript`

## Usage

`python convert_model.py`

# Mobileclip
This model is an apple model, I didn't add this converstion into the script because it had a different process

https://huggingface.co/memojo/mobileclip-s2-onnx/tree/main

The mobile clip image and text encoding models can be found here.
