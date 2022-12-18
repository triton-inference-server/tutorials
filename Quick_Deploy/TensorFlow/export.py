import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

# Load model0
model = ResNet50(weights='imagenet')
model.save('resnet50_saved_model') 
