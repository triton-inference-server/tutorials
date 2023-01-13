## Execute from Part_5-Model_Ensembles Directory
if [ ! -d "./model_repository" ]; then
    echo "Execute from the 'Part_5-Model_Ensembles' directory"
    exit 1
fi

## Download Text Detection Model
mkdir -p downloads
wget -P downloads https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz
tar -xvf downloads/frozen_east_text_detection.tar.gz -C downloads

## Convert to ONNX
pip install -U tf2onnx
mkdir -p model_repository/text_detection/1
python -m tf2onnx.convert \
    --input downloads/frozen_east_text_detection.pb \
    --inputs "input_images:0" \
    --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" \
    --output model_repository/text_detection/1/model.onnx
