package com.ocr.paddleocr.utils;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.TensorInfo;
import org.opencv.core.Size;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;

public class OnnxUtil {

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
     * 从 ONNX Runtime 会话中提取模型的输出张量形状
     * @param session ONNX Runtime 会话
     * @return long[] 输出张量形状
     * @throws OrtException 异常信息
     */
    public static long[] getModelOutputShape(OrtSession session) throws OrtException {
        // 获取模型输出信息
        Map<String, NodeInfo> outputInfo = session.getOutputInfo();
        // 验证输出不为空
        if (outputInfo == null || outputInfo.isEmpty()) {
            throw new OrtException("Model has no output info");
        }
        // 获取第一个输出（大多数模型只有一个输出）
        Map.Entry<String, NodeInfo> first = outputInfo.entrySet().iterator().next();
        NodeInfo nodeInfo = first.getValue();
        // 验证类型为 TensorInfo
        if (!(nodeInfo.getInfo() instanceof TensorInfo)) {
            throw new OrtException("Output info is not TensorInfo");
        }
        // 提取形状数组
        long[] shape = ((TensorInfo) nodeInfo.getInfo()).getShape();
        // 验证形状有效性
        if (shape == null || shape.length == 0) {
            throw new OrtException("Output shape is invalid");
        }
        return shape;
    }

    /**
     * 创建ONNX模型的批量输入张量
     *
     * @param chwList  CHW格式的图像数据列表 (每个元素是一张图的像素数据)
     * @param env      ONNX运行时环境
     * @param modelInputSize 模型期望的输入尺寸 (高度x宽度)
     * @return ONNX张量对象, 可直接用于模型推理
     * @throws OrtException ONNX运行时异常
     */
    public static OnnxTensor createBatchInputTensor(List<float[]> chwList,
                                                    OrtEnvironment env,
                                                    Size modelInputSize) throws OrtException {
        // 获取批次信息
        int batch = chwList.size();
        // 定义图像维度
        int channels = 3;
        int height = (int) modelInputSize.height;
        int width = (int) modelInputSize.width;
        // 计算总数据量
        float[] data = new float[batch * channels * height * width];
        // 计算单张图像数据量
        int one = channels * height * width;
        // 将每张图像的数据依次复制到总数组中
        for (int i = 0; i < batch; i++) {
            System.arraycopy(chwList.get(i), 0, data, i * one, one);
        }
        // 定义ONNX张量的维度形状
        long[] shape = {batch, channels, height, width};
        // 创建ONNX张量
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
    }

    /**
     * 解析ONNX模型的3维输出张量为Java 3维数组
     * @param output ONNX模型运行结果对象, 包含一个或多个输出张量
     * @return 3维浮点数数组
     * @throws OrtException 当输出张量不是3维类型时抛出异常
     */
    public static float[][][] parseOnnxValue3D(Result output) throws OrtException {
        // 1. 获取第一个输出张量
        OnnxValue out = output.get(0);
        // 2. 获取张量的实际Java对象
        Object value = out.getValue();
        // 3. 类型检查和转换
        if (value instanceof float[][][][]) {
            float[][][][] value4D = (float[][][][]) value;
            return value4D[0];
        } else if (value instanceof float[][][]) {
            return (float[][][]) value;
        } else {
            // 类型不匹配，抛出异常
            throw new OrtException("Unsupported output shape");
        }
    }

    /**
     * 解析ONNX模型的2维输出张量为Java 2维数组
     * @param output ONNX模型运行结果对象, 包含一个或多个输出张量
     * @return 2维浮点数数组
     * @throws OrtException 当输出张量不是2维类型时抛出异常
     */
    public static float[][] parseOnnxValue2D(Result output) throws OrtException {
        // 1. 获取第一个输出张量
        OnnxValue out = output.get(0);
        // 2. 获取张量的实际Java对象
        Object value = out.getValue();
        // 3. 类型检查和转换
        if (value instanceof float[][][][]) {
            float[][][][] value4D = (float[][][][]) value;
            return value4D[0][0];
        } else if (value instanceof float[][][]) {
            float[][][] value3D = (float[][][]) value;
            return value3D[0];
        } else if (value instanceof float[][]) {
            return (float[][]) value;
        } else {
            // 类型不匹配，抛出异常
            throw new OrtException("Unsupported output shape");
        }
    }
}
