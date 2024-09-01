const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

async function extractModelDetails() {
    // 加载模型
    const model = await tf.loadLayersModel('file://model.json');

    // 创建一个数组来存储每一层的信息
    const modelDetails = [];

    // 遍历每一层，提取信息
    model.layers.forEach(layer => {
        const layerInfo = {
            name: layer.name,
            input_shape: layer.inputShape ? layer.inputShape.slice(1) : null, // 忽略 batch 大小
            output_shape: layer.outputShape ? layer.outputShape.slice(1) : null, // 忽略 batch 大小
            num_neurons: layer.units || layer.filters || null, // 对于 Dense 或 Conv2D 层提取单元或滤波器数量
            weights: []
        };

        // 提取权重和偏置
        const layerWeights = layer.getWeights();
        layerWeights.forEach((weightTensor, index) => {
            const weightValues = weightTensor.arraySync();
            const biasOrWeight = {
                bias: index === 1 ? weightValues : undefined,
                weights: index === 0 ? weightValues : undefined
            };
            layerInfo.weights.push(biasOrWeight);
        });

        // 添加到模型详情数组中
        modelDetails.push(layerInfo);
    });

    // 将结果写入到 JSON 文件
    fs.writeFileSync('model_details.json', JSON.stringify(modelDetails, null, 2));
}

extractModelDetails();
