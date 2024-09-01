const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp'); // 用于图像处理

async function preprocessImage(imagePath) {
    // 读取图像并使用 sharp 库进行预处理
    const imageBuffer = fs.readFileSync(imagePath);
    
    const resizedImageBuffer = await sharp(imageBuffer)
        .resize(256, 256) // 调整图像大小为 256x256
        .toFormat('jpeg')
        .toBuffer();

    // 将图像缓冲区转换为 TensorFlow.js 张量
    let imgTensor = tf.node.decodeImage(resizedImageBuffer, 3); // 3 通道（RGB）
    imgTensor = imgTensor.expandDims(0); // 添加批次维度
    imgTensor = imgTensor.toFloat().div(tf.scalar(255.0)); // 标准化到 [0, 1] 之间

    return imgTensor;
}

async function runModel() {
    // 加载 TensorFlow.js 模型
    const modelPath = path.resolve(__dirname, 'model.json');
    const model = await tf.loadLayersModel(`file://${modelPath}`);

    // 预处理图像
    const imagePath = path.resolve(__dirname, '/Users/liyujia/haha/tiny-vgg/data/class_10_train/n03662601/images/n03662601_8WXJ.236.682C_page_1.JPEG');
    const processedImg = await preprocessImage(imagePath);

    // 使用模型进行预测
    const predictions = model.predict(processedImg);

    // 输出预测结果
    predictions.print();

    // 获取预测的类别
    const predictedClass = predictions.argMax(-1).dataSync()[0];
    console.log("Predicted class:", predictedClass);
}

// 调用主函数
runModel();

