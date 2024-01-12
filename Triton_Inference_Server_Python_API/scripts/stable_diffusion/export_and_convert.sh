cd /mount
pip3 install -r requirements_model_export.txt
huggingface-cli login --token $HF_TOKEN
python export.py
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16
mkdir -p models_stable_diffusion/vae/1
mkdir -p models_stable_diffusion/text_encoder/1
mv vae.plan models_stable_diffusion/vae/1/model.plan
mv encoder.onnx models_stable_diffusion/text_encoder/1/model.onnx

