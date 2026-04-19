package com.ocr.paddleocr.process;

import ai.onnxruntime.*;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.ModelProcessContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.MatPipeline;
import com.ocr.paddleocr.utils.OnnxUtil;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.*;

/**
 * 文本方向分类处理器
 * 对应官方 Classifier
 * 支持批量处理优化
 *
 * @author PaddleOCR Team
 */
@Slf4j
public class ClsProcess implements AutoCloseable {

    /** 模型输入高度 */
    private static final int MODEL_INPUT_HEIGHT = 48;

    /** 模型输入宽度 */
    private static final int MODEL_INPUT_WIDTH = 192;

    /** 方向标签映射：索引0→0°, 1→180°, 2→90°, 3→270° */
    private static final int[] ANGLES = {0, 180, 90, 270};

    /** 需要旋转的角度阈值（置信度低于此值时不旋转） */
    private static final float ROTATE_CONFIDENCE_THRESHOLD = 0.6f;

    /** ONNX模型会话 */
    private OrtSession session;

    /** ONNX运行时环境 */
    private final OrtEnvironment env;

    /** OCR配置 */
    private final OCRConfig config;

    /**
     * 构造函数
     *
     * @param config OCR配置
     * @throws OrtException ONNX异常
     */
    public ClsProcess(OCRConfig config) throws OrtException {
        this.config = config;
        this.env = OrtEnvironment.getEnvironment();
        loadModel();
        log.info("方向分类处理器初始化完成，模型路径: {}", config.getClsModelPath());
    }

    /**
     * 加载方向分类模型
     */
    private void loadModel() throws OrtException {
        session = OnnxUtil.getSession(config.getClsModelPath(), env, config);
        log.info("方向分类模型加载成功: {}", config.getClsModelPath());
    }

    /**
     * 对检测结果进行方向分类
     */
    public void classify(ModelProcessContext context) {
        context.setSuccess(false);
        long startTime = System.currentTimeMillis();
        try {
            Mat prepMat = context.getDetPrepMat();
            List<TextBox> boxes = context.getBoxes();

            if (boxes == null || boxes.isEmpty()) {
                context.setSuccess(true);
                context.setClsProcessTime(System.currentTimeMillis() - startTime);
                return;
            }

            // 1. 收集所有需要分类的文本框图像
            List<Mat> cropImages = new ArrayList<>();
            List<TextBox> validBoxes = new ArrayList<>();

            for (TextBox box : boxes) {
                Mat cropped;
                try {
                    cropped = cropTextBox(prepMat, box.getBox());
                    if (cropped != null && !cropped.empty()) {
                        cropImages.add(cropped);
                        validBoxes.add(box);
                    } else {
                        // 设置默认值
                        box.setAngle(0);
                        box.setClsConfidence(0.0f);
                    }
                } catch (Exception e) {
                    log.debug("裁剪文本框失败: {}", e.getMessage());
                    box.setAngle(0);
                    box.setClsConfidence(0.0f);
                }
            }

            if (cropImages.isEmpty()) {
                context.setSuccess(true);
                context.setClsProcessTime(System.currentTimeMillis() - startTime);
                return;
            }

            // 2. 批量分类
            List<ClassificationResult> batchResults = classifyBatch(cropImages);

            // 3. 处理分类结果并执行旋转
            int rotCount = 0;
            for (int i = 0; i < batchResults.size() && i < validBoxes.size(); i++) {
                TextBox box = validBoxes.get(i);
                ClassificationResult result = batchResults.get(i);

                box.setAngle(result.angle);
                box.setClsConfidence(result.confidence);

                // 判断是否需要旋转并执行
                if (needRotate(box)) {
                    Mat originalCrop = cropImages.get(i);
                    rotateTextBox(box, originalCrop);
                    rotCount++;
                }
            }

            // 4. 释放临时资源
            for (Mat crop : cropImages) {
                if (crop != null && !crop.empty()) {
                    crop.release();
                }
            }

            context.setClsRotBox(rotCount);
            context.setSuccess(true);
            context.setClsProcessTime(System.currentTimeMillis() - startTime);
            log.info("方向分类完成，处理 {} 个文本框，旋转 {} 个，耗时: {}ms",
                    boxes.size(), rotCount, context.getClsProcessTime());

        } catch (Exception e) {
            context.setSuccess(false);
            context.setError(e.getMessage());
            log.error("方向分类失败", e);
        }
    }

    /**
     * 批量分类
     *
     * @param images 待分类的图像列表
     * @return 分类结果列表
     * @throws OrtException ONNX异常
     */
    private List<ClassificationResult> classifyBatch(List<Mat> images) throws OrtException {
        if (images == null || images.isEmpty()) {
            return new ArrayList<>();
        }

        // 1. 批量预处理
        List<Mat> processedImages = batchPreprocess(images);

        // 2. 创建批量输入Tensor
        OnnxTensor inputTensor = OnnxUtil.createBatchInputTensor(processedImages, env);

        // 3. 批量推理
        Map<String, OnnxTensor> inputs = Collections.singletonMap("x", inputTensor);
        OrtSession.Result output = session.run(inputs);

        // 4. 解析输出
        float[][][] batchOutput = parseBatchOutput(output);

        // 5. 解析分类结果
        List<ClassificationResult> results = new ArrayList<>();
        for (float[][] outputData : batchOutput) {
            ClassificationResult result = parseClassificationOutput(outputData);
            results.add(result);
        }

        // 6. 释放资源
        inputTensor.close();
        output.close();
        for (Mat processed : processedImages) {
            if (processed != null && !processed.empty()) {
                processed.release();
            }
        }

        return results;
    }



    /**
     * 批量预处理
     *
     * @param images 原始图像列表
     * @return 预处理后的图像列表
     */
    private List<Mat> batchPreprocess(List<Mat> images) {
        List<Mat> processed = new ArrayList<>();
        for (Mat image : images) {
            processed.add(preprocess(image));
        }
        return processed;
    }

    /**
     * 预处理：转RGB + 缩放到固定尺寸 + 官方归一化
     * 官方分类模型使用ImageNet均值标准差
     */
    private Mat preprocess(Mat image) {
        return MatPipeline.fromMat(image)
                .toRGB()  // BGR转RGB
                .resize(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
                .normalize()  // 使用ImageNet均值标准差
                .get();
    }

    /**
     * 解析批量输出（3D数组格式）
     *
     * @param output ONNX模型输出
     * @return 3D输出数据 [batch, num_classes]
     * @throws OrtException ONNX异常
     */
    private float[][][] parseBatchOutput(OrtSession.Result output) throws OrtException {
        OnnxValue outputValue = output.get(0);

        try {
            // 格式1: 直接是3D数组 [batch, 1, num_classes] 或 [batch, num_classes]
            Object value = outputValue.getValue();

            if (value instanceof float[][][]) {
                float[][][] value3D = (float[][][]) value;
                // 如果是 [batch, 1, num_classes] 格式，压缩中间维度
                if (value3D.length > 0 && value3D[0].length == 1) {
                    float[][][] result = new float[value3D.length][1][value3D[0][0].length];
                    for (int i = 0; i < value3D.length; i++) {
                        result[i][0] = value3D[i][0];
                    }
                    return result;
                }
                return value3D;
            }
            else if (value instanceof float[][]) {
                // 格式2: 2D数组 [batch, num_classes]
                float[][] value2D = (float[][]) value;
                float[][][] result = new float[value2D.length][1][value2D[0].length];
                for (int i = 0; i < value2D.length; i++) {
                    result[i][0] = value2D[i];
                }
                return result;
            }

            throw new OrtException("不支持的输出格式: " + value.getClass().getName());

        } catch (Exception e) {
            throw new OrtException("无法解析批量分类模型输出格式: " + e.getMessage());
        }
    }

    /**
     * 解析分类输出，获取角度和置信度
     *
     * @param outputData 模型输出 [1, num_classes] 或 [num_classes]
     * @return 分类结果
     */
    private ClassificationResult parseClassificationOutput(float[][] outputData) {
        if (outputData == null || outputData.length == 0) {
            return new ClassificationResult(0, 0.0f);
        }

        // 获取预测结果（取第一行）
        float[] probabilities = outputData[0];

        // 获取最大概率的索引
        int predClass = argmax(probabilities);
        float confidence = probabilities[predClass];
        int angle = ANGLES[predClass];

        log.debug("方向分类完成，角度: {}, 置信度: {}%",
                angle, String.format("%.2f", confidence * 100));

        return new ClassificationResult(angle, confidence);
    }

    /**
     * 获取最大值的索引
     */
    private int argmax(float[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int maxIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * 判断是否需要旋转文本框
     *
     * @param detBox 文本框对象
     * @return true表示需要旋转
     */
    private boolean needRotate(TextBox detBox) {
        int angle = detBox.getAngle();
        float confidence = detBox.getClsConfidence();

        // 只有180度需要旋转（PaddleOCR主要修正180度颠倒）
        // 且置信度需要达到阈值
        if (angle == 180 && confidence >= ROTATE_CONFIDENCE_THRESHOLD) {
            log.debug("检测到{}度旋转，将进行旋转处理", angle);
            return true;
        }

        // 90度和270度可选旋转（根据配置决定）
        if (config.isUseAngleCls() && (angle == 90 || angle == 270)
                && confidence >= ROTATE_CONFIDENCE_THRESHOLD) {
            log.debug("检测到{}度旋转，将进行旋转处理", angle);
            return true;
        }

        return false;
    }

    /**
     * 旋转文本框图像并更新坐标
     *
     * @param detBox 文本框对象
     * @param originalImage 原始裁剪图像（用于旋转）
     */
    private void rotateTextBox(TextBox detBox, Mat originalImage) {
        detBox.setRawMat(originalImage);
        try {
            int angle = detBox.getAngle();
            Mat rotatedMat = null;

            // 根据角度进行旋转
            if (angle == 180) {
                rotatedMat = rotateImage180(originalImage);
                detBox.setRotAngle(180);
            } else if (angle == 90) {
                rotatedMat = rotateImage90(originalImage, true);
                detBox.setRotAngle(90);
            } else if (angle == 270) {
                rotatedMat = rotateImage90(originalImage, false);
                detBox.setRotAngle(270);
            }

            if (rotatedMat != null) {
                // 重要：将旋转后的图像设置到TextBox中，供RecProcess使用
                detBox.setRotMat(rotatedMat);
                detBox.setRotate(true);

                log.debug("文本框已旋转 {} 度，旋转后图像尺寸: {}x{}",
                        angle, rotatedMat.width(), rotatedMat.height());
            }

        } catch (Exception e) {
            log.error("旋转文本框失败", e);
            detBox.setRotate(false);
        }
    }

    /**
     * 180度旋转图像
     *
     * @param src 源图像
     * @return 旋转后的图像
     */
    private Mat rotateImage180(Mat src) {
        Mat dst = new Mat();
        Core.flip(src, dst, -1); // -1表示同时翻转x和y轴
        return dst;
    }

    /**
     * 90度旋转图像
     *
     * @param src 源图像
     * @param clockwise true:顺时针90度, false:逆时针90度
     * @return 旋转后的图像
     */
    private Mat rotateImage90(Mat src, boolean clockwise) {
        Mat dst = new Mat();
        if (clockwise) {
            Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE);
        } else {
            Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE);
        }
        return dst;
    }

    /**
     * 从图像中裁剪文本框区域
     *
     * @param image 原始图像
     * @param box 文本框四点坐标
     * @return 裁剪后的图像
     */
    public static Mat cropTextBox(Mat image, List<Point> box) {
        if (image == null || image.empty() || box == null || box.size() < 4) {
            return null;
        }
        try {
            // 使用透视变换获得校正后的文本图像
            return OpenCVUtil.perspectiveTransformCrop(image, box);
        } catch (Exception e) {
            // 如果透视变换失败，回退到矩形裁剪
            return OpenCVUtil.rectangleCrop(image, box);
        }
    }

    /**
     * 关闭模型会话，释放资源
     */
    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
            log.info("方向分类模型会话已关闭");
        }
        // 注意：不要关闭env，因为可能被多个processor共享
    }

    /**
     * 分类结果内部类
     */
    private static class ClassificationResult {
        final int angle;
        final float confidence;

        ClassificationResult(int angle, float confidence) {
            this.angle = angle;
            this.confidence = confidence;
        }
    }
}