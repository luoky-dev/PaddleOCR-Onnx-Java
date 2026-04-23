package com.ocr.paddleocr.utils;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.ocr.paddleocr.config.OCRConfig;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.List;

public class OnnxUtil {

    /**
     * 创建单张输入Tensor
     * 注意：输入Mat必须是CHW格式，RGB通道顺序
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
     * 创建批量输入Tensor
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
     * 创建批量输入Tensor
     */
    public static OnnxTensor createBatchInputTensor(List<Mat> images, OrtEnvironment env) throws OrtException {
        if (images.isEmpty()) {
            throw new IllegalArgumentException("Image list is empty");
        }

        int batchSize = images.size();
        int channels = images.get(0).channels();
        int height = images.get(0).rows();
        int width = images.get(0).cols();

        // 验证所有图像尺寸一致
        for (Mat img : images) {
            if (img.rows() != height || img.cols() != width) {
                throw new IllegalArgumentException("All images must have same dimensions");
            }
            if (img.channels() != channels) {
                throw new IllegalArgumentException("All images must have same number of channels");
            }
        }

        float[] batchData = new float[batchSize * channels * height * width];

        for (int b = 0; b < batchSize; b++) {
            Mat image = images.get(b);
            float[] imageData = new float[channels * height * width];
            image.get(0, 0, imageData);

            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int chwIndex = (c * height + h) * width + w;
                        int hwcIndex = (h * width + w) * channels + c;
                        int batchIndex = b * channels * height * width + chwIndex;
                        batchData[batchIndex] = imageData[hwcIndex];
                    }
                }
            }
        }

        long[] shape = {batchSize, channels, height, width};
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(batchData), shape);
    }

    public static OrtSession getSession(String modelPath, OrtEnvironment env, OCRConfig config) throws OrtException {
        SessionOptions sessionOptions = new SessionOptions();
        // GPU配置
        if (config.isUseGpu()) {
            sessionOptions.addCUDA(config.getGpuId());
        }
        // CPU线程数配置
        sessionOptions.setIntraOpNumThreads(config.getNumThreads());
        // 设置执行模式
        sessionOptions.setExecutionMode(SessionOptions.ExecutionMode.SEQUENTIAL);
        return env.createSession(modelPath, sessionOptions);
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
     * 解析 rec 输出为 [batch, time, classes]。
     */
    private float[][][] parseRecOutput(Result output) throws OrtException {
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
