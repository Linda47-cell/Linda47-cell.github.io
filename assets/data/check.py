import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np

# 读取JSON文件
with open('model.json', 'r') as json_file:
    model_json = json.load(json_file)

# 从JSON数据构建模型
model = model_from_json(json.dumps(model_json['modelTopology']['model_config']))

# 加载权重
weights_manifest = model_json['weightsManifest'][0]
weights_path = weights_manifest['paths'][0]

# 读取二进制权重文件
with open(weights_path, 'rb') as weight_file:
    weights_data = weight_file.read()

# 将权重数据拆分并设置到模型中
offset = 0
for weight in weights_manifest['weights']:
    weight_name = weight['name']
    weight_shape = weight['shape']
    weight_dtype = weight['dtype']

    # 计算权重数据的字节大小
    weight_size = np.prod(weight_shape)
    weight_bytes = np.dtype(weight_dtype).itemsize * weight_size

    # 从二进制数据中提取权重值
    weight_values = np.frombuffer(weights_data, dtype=weight_dtype, count=weight_size, offset=offset)
    weight_values = weight_values.reshape(weight_shape)
    offset += weight_bytes

    # 设置权重到模型中
    layer_name, weight_type = weight_name.split('/')
    layer = model.get_layer(layer_name)
    if weight_type == 'kernel':
        layer.set_weights([weight_values, layer.get_weights()[1]])
    elif weight_type == 'bias':
        layer.set_weights([layer.get_weights()[0], weight_values])

# 输出模型结构
model.summary()

# 测试模型是否能正常运行
input_shape = model_json['modelTopology']['model_config']['config']['layers'][0]['config']['batch_input_shape']
dummy_input = np.zeros([1] + input_shape[1:])  # 创建一个全零的输入数据

try:
    output = model.predict(dummy_input)
    print("模型运行正常，输出形状:", output.shape)
except Exception as e:
    print("模型运行时出错:", e)

