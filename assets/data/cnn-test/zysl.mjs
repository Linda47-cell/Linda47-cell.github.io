import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import fs from 'fs';
import Npyjs from 'npyjs';

async function loadNpyFile(filePath) {
    const npyjs = new Npyjs();
    const buffer = fs.readFileSync(filePath);  // 读取文件为二进制缓冲区
    const data = npyjs.parse(buffer.buffer);  // 使用 Npyjs 的 parse 方法解析缓冲区
    return tf.tensor(data.data, data.shape, data.dtype);
}

async function runModel() {
    const modelPath = path.resolve('/Users/liyujia/haha/public/assets/data/cnn-test', 'model.json');

    // 加载 TensorFlow.js 模型
    const model = await tf.loadLayersModel(`file://${modelPath}`);

    // 加载 .npy 文件并转换为 TensorFlow.js 的张量
    const inputNpyPath = path.resolve('/Users/liyujia/haha/public/assets/data/cnn-test', 'input_data.npy');
    const inputTensor = await loadNpyFile(inputNpyPath);

    // 使用模型进行预测
    const tfjsOutput = model.predict(inputTensor);

    // 打印输出
    tfjsOutput.print();

    // 打印模型结构
    model.summary();
}

// 调用异步函数
runModel();
