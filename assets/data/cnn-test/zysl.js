import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import Npyjs from 'npyjs';
import { fileURLToPath, pathToFileURL } from 'url';

async function loadNpyFile(filePath) {
    const npyjs = new Npyjs();
    const fileUrl = pathToFileURL(filePath);  // 将路径转换为 file:// URL
    const data = await npyjs.load(fileUrl.href);  // 使用 href 以字符串形式获取 URL
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

