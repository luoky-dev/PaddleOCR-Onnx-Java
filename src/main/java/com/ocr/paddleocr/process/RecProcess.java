package com.ocr.paddleocr.process;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.Result;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.ModelProcessContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.MatPipeline;
import com.ocr.paddleocr.utils.ModelUtil;
import com.ocr.paddleocr.utils.OnnxUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * 文本识别处理器
 */
@Slf4j
public class RecProcess implements AutoCloseable {

    // ==================== 模型配置常量 ====================

    /** 模型输入高度 */
    private static final int MODEL_INPUT_HEIGHT = 48;

    /** 模型输入宽度 */
    private static final int MODEL_INPUT_WIDTH = 320;

    /** CTC空白索引（blank token） */
    private static final int BLANK_INDEX = 0;

    // ==================== 成员变量 ====================

    /** ONNX模型会话 */
    private OrtSession session;

    /** ONNX运行时环境 */
    private final OrtEnvironment env;

    /** OCR配置 */
    private final OCRConfig config;

    /** 字符字典（索引→字符映射） */
    private final List<String> dict;

    /** 忽略的token集合（blank和可选空格） */
    private final Set<Integer> ignoredTokens;

    /**
     * 构造函数
     *
     * @param config OCR配置
     * @throws OrtException ONNX异常
     */
    public RecProcess(OCRConfig config) throws OrtException {
        this.config = config;
        this.env = OrtEnvironment.getEnvironment();
        this.dict = readDictionary(config.getDictPath());
        this.ignoredTokens = new HashSet<>();

        initIgnoredTokens();
        loadModel();
        log.info("文本识别处理器初始化完成，模型路径: {}, 字典大小: {}, 批量大小: {}",
                config.getRecModelPath(), dict.size(), config.getRecBatchSize());
    }

    /**
     * 初始化忽略的token
     */
    private void initIgnoredTokens() {
        ignoredTokens.clear();
        ignoredTokens.add(BLANK_INDEX);

        boolean useSpaceChar = !"ch".equals(config.getLang()) &&
                !"japan".equals(config.getLang()) &&
                !"korean".equals(config.getLang());

        if (!useSpaceChar) {
            int spaceIdx = getSpaceIndex();
            if (spaceIdx > 0) {
                ignoredTokens.add(spaceIdx);
                log.info("不使用空格字符，空格将被忽略");
            }
        } else {
            log.info("使用空格字符，空格将被保留");
        }
    }

    /**
     * 获取空格在字典中的索引
     */
    private int getSpaceIndex() {
        for (int i = 0; i < dict.size(); i++) {
            if (" ".equals(dict.get(i))) {
                return i + 1; // 索引从1开始（0是blank）
            }
        }
        return -1;
    }

    /**
     * 读取字典文件
     */
    private List<String> readDictionary(String dictPath) {
        List<String> dictionary = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(dictPath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    dictionary.add(line);
                }
            }
            log.info("字典文件加载成功: {}, 共 {} 个字符", dictPath, dictionary.size());
        } catch (Exception e) {
            log.error("读取字典文件失败: {}", dictPath, e);
            throw new RuntimeException("读取字典文件失败: " + dictPath, e);
        }

        return dictionary;
    }

    /**
     * 加载识别模型
     */
    private void loadModel() throws OrtException {
        OrtSession.SessionOptions sessionOptions = ModelUtil.getSessionOptions(config);
        session = env.createSession(config.getRecModelPath(), sessionOptions);
        log.info("识别模型加载成功: {}", config.getRecModelPath());
    }

    /**
     * 对检测结果进行文本识别
     * 直接使用ClsProcess已经旋转矫正过的图像
     *
     * @param context 模型处理上下文
     */
    public void recognize(ModelProcessContext context) {
        context.setSuccess(false);
        long startTime = System.currentTimeMillis();

        try {
            List<TextBox> boxes = context.getBoxes();

            if (boxes == null || boxes.isEmpty()) {
                context.setError("检测结果为空，无法进行文本识别");
                context.setRecProcessTime(System.currentTimeMillis() - startTime);
                return;
            }

            // 1. 准备识别图像（优先使用ClsProcess已旋转的图像）
            List<Mat> recognizeImages = new ArrayList<>();
            for (TextBox box : boxes) {
                Mat image;
                if (box.getRotMat() != null && !box.getRotMat().empty()) {
                    image = box.getRotMat();
                } else {
                    // 如果没有旋转图像，使用原始图像
                    image = box.getRawMat();
                }
                recognizeImages.add(image);
            }
            // 2. 批量识别
            List<TextBox> recResults = recognizeBatch(recognizeImages);

            // 3. 设置识别结果
            context.setSuccess(true);
            context.setBoxes(recResults);
            context.setRecProcessTime(System.currentTimeMillis() - startTime);

            log.info("文本识别完成，成功识别 {} 个文本框，耗时: {}ms", recResults.size(), context.getRecProcessTime());

        } catch (Exception e) {
            context.setSuccess(false);
            context.setError(e.getMessage());
            log.error("文本识别失败", e);
        }
    }

    /**
     * 批量识别
     */
    private List<TextBox> recognizeBatch(List<Mat> images) throws OrtException {
        // 1. 批量预处理
        List<Mat> processedImages = new ArrayList<>();
        for (Mat image : images) {
            processedImages.add(preprocess(image));
        }

        // 2. 创建批量输入Tensor
        OnnxTensor inputTensor = OnnxUtil.createBatchInputTensor(processedImages, env);

        // 3. 批量推理
        Map<String, OnnxTensor> inputs = Collections.singletonMap("x", inputTensor);
        Result output = session.run(inputs);

        // 4. 解析输出
        float[][][] results = parseOutput(output);

        // 5. CTC解码
        List<TextBox> boxes = new ArrayList<>();
        for (float[][] result : results) {
            TextBox box = ctcDecode(result);
            boxes.add(box);
        }
        // 6. 释放资源
        inputTensor.close();
        output.close();
        for (Mat processed : processedImages) {
            processed.release();
        }
        return boxes;
    }

    /**
     * 图像预处理（保持BGR通道，与官方一致）
     */
    private Mat preprocess(Mat image) {
        Mat result = MatPipeline.fromMat(image)
                .toRGB()
                .resize(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
//                .normalize()
                .convertTo(CvType.CV_32FC3, 1.0 / 255.0, 0)  // 直接使用带参数的方法
                .get();

        // 调试：打印类型
        log.info("识别预处理后图像类型: {}", result.type());
        log.info("CV_32FC3 = {}, 实际类型 = {}", CvType.CV_32FC3, result.type());

        return result;
    }

    /**
     * 从ONNX输出值中提取数据（单张）
     * 支持多种输出格式：
     * - float[time_steps][num_classes]
     * - float[1][time_steps][num_classes]
     * - float[time_steps * num_classes] (1D)
     *
     * @param output ONNX输出值
     * @return 2D输出数据 [time_steps, num_classes]
     * @throws OrtException 解析异常
     */
    private float[][][] parseOutput(Result output) throws OrtException {
        OnnxValue outputValue = output.get(0);
        try {
            // 格式1: 直接是3D数组 [batch, time_steps, num_classes]
            return (float[][][]) outputValue.getValue();
        } catch (ClassCastException e1) {
            try {
                // 格式2: 2D数组，需要按batch拆分（单个样本的情况）
                float[][] outputData2D = (float[][]) outputValue.getValue();
                // 将2D包装成3D（batch=1）
                return new float[][][]{outputData2D};
            } catch (ClassCastException e2) {
                throw new OrtException("无法解析批量识别模型输出格式: 期望3D数组或2D数组");
            }
        }
    }

    /**
     * CTC解码算法（完全遵循PaddleOCR官方实现）
     *
     * @param outputData 模型输出 [time_steps, num_classes]
     * @return 识别结果
     */
    private TextBox ctcDecode(float[][] outputData) {
        int seqLen = outputData.length;
        if (seqLen == 0) {
            return TextBox.builder().text("").recConfidence(0.0f).build();
        }

        int numClasses = outputData[0].length;

        // 1. 获取每个时间步的最大概率索引
        int[] indices = new int[seqLen];
        float[] probs = new float[seqLen];

        for (int t = 0; t < seqLen; t++) {
            int maxIdx = 0;
            float maxProb = outputData[t][0];
            for (int c = 1; c < numClasses; c++) {
                if (outputData[t][c] > maxProb) {
                    maxProb = outputData[t][c];
                    maxIdx = c;
                }
            }
            indices[t] = maxIdx;
            probs[t] = maxProb;
        }

        // 2. CTC解码：去除重复和blank
        StringBuilder text = new StringBuilder();
        float totalConfidence = 0;
        int validCount = 0;

        int lastIdx = BLANK_INDEX;
        for (int t = 0; t < seqLen; t++) {
            int currentIdx = indices[t];

            // 跳过blank
            if (currentIdx == BLANK_INDEX) {
                lastIdx = BLANK_INDEX;
                continue;
            }

            // 跳过连续重复
            if (currentIdx == lastIdx) {
                continue;
            }

            // 跳过需要忽略的token（如空格）
            if (ignoredTokens.contains(currentIdx)) {
                lastIdx = currentIdx;
                continue;
            }

            // 转换为字符
            int dictIdx = currentIdx - 1;
            if (dictIdx >= 0 && dictIdx < dict.size()) {
                String character = dict.get(dictIdx);
                text.append(character);
                totalConfidence += probs[t];
                validCount++;
            }

            lastIdx = currentIdx;
        }

        String resultText = text.toString();
        float confidence = validCount > 0 ? totalConfidence / validCount : 0.0f;

        // 去除首尾空格（但保留中间空格）
        resultText = resultText.trim();

        return TextBox.builder().text(resultText).recConfidence(confidence).build();
    }

    /**
     * 关闭模型会话，释放资源
     */
    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
            log.info("识别模型会话已关闭");
        }
        // 注意：不要关闭env，因为可能被多个processor共享
        ignoredTokens.clear();
    }

}