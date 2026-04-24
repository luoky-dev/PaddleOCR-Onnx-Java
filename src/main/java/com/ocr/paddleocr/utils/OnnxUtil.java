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
     * 判断模型输入是否为动态 shape
     * true: dynamic input; false: fixed input.
     */
    public static Boolean isDynamicInput(OrtSession session) {
        try {
            long[] shape = getInputShape(session);
            if (shape.length != 4) {
                return true;
            }
            return shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0;
        } catch (Exception e) {
            return true;
        }
    }

    /**
     * 返回模型输入 shape
     * 动态维度一般为 <= 0（如 -1）。
     */
    public static long[] getInputShape(OrtSession session) throws OrtException {
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        if (inputInfo == null || inputInfo.isEmpty()) {
            throw new OrtException("Model has no input info");
        }

        NodeInfo nodeInfo = inputInfo.get("x");
        if (nodeInfo == null) {
            Map.Entry<String, NodeInfo> first = inputInfo.entrySet().iterator().next();
            nodeInfo = first.getValue();
        }

        if (!(nodeInfo.getInfo() instanceof TensorInfo)) {
            throw new OrtException("Input info is not TensorInfo");
        }

        long[] shape = ((TensorInfo) nodeInfo.getInfo()).getShape();
        if (shape == null || shape.length != 4) {
            throw new OrtException("Input shape is invalid");
        }
        return shape;
    }

    /**
     * 创建单张输入 Tensor。
     */
    public static OnnxTensor createInputTensor(Mat image, OrtEnvironment env) throws OrtException {
        int height = image.rows();
        int width = image.cols();
        int channels = image.channels();

        float[] chwData = new float[channels * height * width];
        float[] data = new float[channels * height * width];
        image.get(0, 0, data);

        // 转换为CHW格式（已经是RGB顺序）
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
     * 创建批量输入 Tensor。
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
