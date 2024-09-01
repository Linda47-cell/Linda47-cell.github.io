const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function loadAndTestModel(modelPath, imagePath) {
    const model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('Model loaded successfully');

    // 读取和解码图像
    const imageBuffer = fs.readFileSync(imagePath);
    let imageTensor = tf.node.decodeImage(imageBuffer);

    // 调整图像大小以符合模型的输入要求
    imageTensor = tf.image.resizeBilinear(imageTensor, [256, 256]);

    // 确保图像像素值归一化到 [0, 1]
    imageTensor = imageTensor.toFloat().div(tf.scalar(255.0));
    
    // 添加批次维度
    imageTensor = imageTensor.expandDims(0);

    console.log('Model input shape:', model.inputs[0].shape);

    // 使用模型进行预测
    const predictions = model.predict(imageTensor);

    // 打印预测的 logits 结果
    predictions.print();

    // 应用 softmax 以获得概率分布
    const softmaxPredictions = tf.softmax(predictions);
    softmaxPredictions.print();

    // 获取预测的类别
    const predictedClass = softmaxPredictions.argMax(-1).dataSync()[0];
    console.log(`Predicted class: ${predictedClass}`);
}

const modelPath = 'model.json';
const imagePath = '/Users/liyujia/haha/tiny-vgg/data/class_10_train/n03662601/images/n03662601_8XJ.052.033_page_1.JPEG';
loadAndTestModel(modelPath, imagePath);
