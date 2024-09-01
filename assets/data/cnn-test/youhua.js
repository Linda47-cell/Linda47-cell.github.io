const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function preprocessImage(imagePath) {
    const imageBuffer = fs.readFileSync(imagePath);
    
    // 解码图像并调整大小
    let imgTensor = tf.node.decodeImage(imageBuffer, 3); // 3 通道（RGB）
    imgTensor = tf.image.resizeBilinear(imgTensor, [256, 256]); // 调整大小
    imgTensor = imgTensor.expandDims(0); // 添加批次维度
    imgTensor = imgTensor.toFloat().div(tf.scalar(255.0)); // 标准化到 [0, 1] 之间

    return imgTensor;
}

async function runModel() {
    const modelPath = 'model.json';
    const model = await tf.loadLayersModel(`file://${modelPath}`);

    const imagePath = '/Users/liyujia/haha/tiny-vgg/data/class_10_train/n03662601/images/n03662601_8WXJ.236.682C_page_1.JPEG';
    const processedImg = await preprocessImage(imagePath);

    const predictions = model.predict(processedImg);
    predictions.print();

    const predictedClass = predictions.argMax(-1).dataSync()[0];
    console.log(`Predicted class: ${predictedClass}`);
}

runModel();
