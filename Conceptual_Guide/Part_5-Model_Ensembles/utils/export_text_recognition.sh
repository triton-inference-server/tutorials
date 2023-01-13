## Execute from Part_5-Model_Ensembles Directory
if [ ! -d "./model_repository" ]; then
    echo "Execute from the 'Part_5-Model_Ensembles' directory"
    exit 1
fi

## Download Text Detection Model
mkdir -p downloads
wget -P downloads https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth

## Convert to ONNX
python utils/export_text_recognition.py
