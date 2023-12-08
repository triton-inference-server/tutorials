cd /mount
pip3 install -r requirements_model_export.txt
huggingface-cli login --token $HF_TOKEN
python export.py
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16
mkdir -p model_repository/vae/1
mkdir -p model_repository/text_encoder/1
mv vae.plan model_repository/vae/1/model.plan
mv encoder.onnx model_repository/text_encoder/1/model.onnx

