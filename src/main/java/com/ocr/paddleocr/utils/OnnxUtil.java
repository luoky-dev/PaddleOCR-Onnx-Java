package com.ocr.paddleocr.utils;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.TensorInfo;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;

public class OnnxUtil {

    /**
     * 判断模型是否是动态图像尺寸输入
     * @param session ONNX Runtime 会话
     * @return true 是动态输入
     * @throws OrtException 异常信息
     */
    public static Boolean isDynamicImageInput(OrtSession session) throws OrtException {
        return isDynamicHeightInput(session) && isDynamicWithInput(session);
    }

    /**
     * 判断模型是否是图像动态高度尺寸输入
     * @param session ONNX Runtime 会话
     * @return true 是动态高度输入
     * @throws OrtException 异常信息
     */
    public static Boolean isDynamicHeightInput(OrtSession session) throws OrtException {
        long[] inputShape = getModelInputShape(session);
        return inputShape[2] == -1;
    }

    /**
     * 判断模型是否是图像动态宽度尺寸输入
     * @param session ONNX Runtime 会话
     * @return true 是动态宽度输入
     * @throws OrtException 异常信息
     */
    public static Boolean isDynamicWithInput(OrtSession session) throws OrtException {
        long[] inputShape = getModelInputShape(session);
        return inputShape[3] == -1;
    }

    /**
     * 从 ONNX Runtime 会话中提取模型的输入张量形状
     * Paddle模型的输入形状为[Batch,Channel,Height,Width]
     * @param session ONNX Runtime 会话
     * @return long[] 输入张量形状
     * @throws OrtException 异常信息
     */
    public static long[] getModelInputShape(OrtSession session) throws OrtException {
        // 获取模型输入信息
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        // 验证输入不为空
        if (inputInfo == null || inputInfo.isEmpty()) {
            throw new OrtException("Model has no input info");
        }
        // 查找名为 "x" 的输入, PaddleOCR 标准模型使用 "x" 作为输入名称
        NodeInfo nodeInfo = inputInfo.get("x");
        // 降级处理: 取第一个输入, 如果找不到 "x"，可能是其他框架导出的模型
        if (nodeInfo == null) {
            Map.Entry<String, NodeInfo> first = inputInfo.entrySet().iterator().next();
            nodeInfo = first.getValue();
        }
        // 验证类型为 TensorInfo, 如果不是 TensorInfo 格式不兼容
        if (!(nodeInfo.getInfo() instanceof TensorInfo)) {
            throw new OrtException("Input info is not TensorInfo");
        }
        // 提取形状数组
        long[] shape = ((TensorInfo) nodeInfo.getInfo()).getShape();
        // 验证形状有效性, PaddleOCR 模型要求输入是 4 维张量 (N, C, H, W)
        if (shape == null || shape.length != 4) {
            throw new OrtException("Input shape is invalid");
        }
        return shape;
    }

    /**
     * 创建单张输入 Tensor
     */
    public static OnnxTensor createInputTensor(Mat image, OrtEnvironment env) throws OrtException {
        int height = image.rows();
        int width = image.cols();
        int channels = image.channels();

        float[] chwData = new float[channels * height * width];
        float[] data = new float[channels * height * width];
        image.get(0, 0, data);

        // 转换为CHW格式
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int chwIndex = (c * height + h) * width + w;
                    int hwcIndex = (h * width + w) * channels + c;
                    chwData[chwIndex] = data[hwcIndex];
                }
            }
        }

        long[] shape = {1, channels, height, width};
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(chwData), shape);
    }

    /**
     * 创建批量输入 Tensor
     */
    public static OnnxTensor createBatchInputTensor(List<float[]> chwList,
                                                    OrtEnvironment env,
                                                    int channels,
                                                    int height,
                                                    int width) throws OrtException {
        int batch = chwList.size();
        float[] data = new float[batch * channels * height * width];
        int one = channels * height * width;
        for (int i = 0; i < batch; i++) {
            System.arraycopy(chwList.get(i), 0, data, i * one, one);
        }
        long[] shape = {batch, channels, height, width};
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
    }

    /**
     * 解析检测模型输出
     */
    public static float[][] parseDetOutput(Result output) throws OrtException {
        OnnxValue out = output.get(0);
        try {
            float[][][][] v4 = (float[][][][]) out.getValue();
            return v4[0][0];
        } catch (ClassCastException e1) {
            try {
                float[][][] v3 = (float[][][]) out.getValue();
                return v3[0];
            } catch (ClassCastException e2) {
                throw new OrtException("Unsupported det output shape");
            }
        }
    }

    /**
     * 解析分类模型输出（角度分类）
     * 模型输出形状：[batch, classes] 或 [batch, 1, classes]
     */
    public static float[][] parseClsOutput(Result output) throws OrtException {
        OnnxValue v = output.get(0);
        Object value = v.getValue();
        if (value instanceof float[][]) {
            return (float[][]) value;
        }
        if (value instanceof float[][][]) {
            float[][][] v3 = (float[][][]) value;
            float[][] result = new float[v3.length][];
            for (int i = 0; i < v3.length; i++) {
                result[i] = v3[i][0];
            }
            return result;
        }
        throw new OrtException("Unsupported cls output shape");
    }

    /**
     * 解析 rec 输出为 [batch_size, time_steps, num_classes]
     */
    public static float[][][] parseRecOutput(Result output) throws OrtException {
        OnnxValue v = output.get(0);
        Object value = v.getValue();
        if (value instanceof float[][][]) {
            return (float[][][]) value;
        }
        if (value instanceof float[][]) {
            float[][] v2 = (float[][]) value;
            return new float[][][]{v2};
        }
        if (value instanceof float[][][][]) {
            float[][][][] v4 = (float[][][][]) value;
            return v4[0];
        }
        throw new OrtException("Unsupported rec output shape");
    }
}
