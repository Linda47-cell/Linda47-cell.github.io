import tensorflow as tf
from tensorflow.keras.models import load_model

# 加载训练好的模型
model = load_model('trained_vgg_best.h5')

from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # 使用与训练时相同的尺寸
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    img_array = img_array / 255.0  # 标准化到 [0, 1] 之间
    return img_array

# 预处理你的测试图片
img_path = 'data/class_10_train/n01882714/images/n01882714_8WXJ.062.024_page_1.JPEG'
processed_img = preprocess_image(img_path)

predictions = model.predict(processed_img)

# 输出预测结果
print("Predictions:", predictions)

# 如果你有类别映射，可以将预测值转换为类别
predicted_class = np.argmax(predictions, axis=1)
print("Predicted class:", predicted_class)
