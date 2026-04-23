package com.ocr.paddleocr.process;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.ModelProcessContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.ModelUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * ClsProcessUpdate:
 * 独立的文本方向分类实现，按 PaddleOCR 官方 cls 思路组织。
 * 说明：该类暂时不接入主流程，仅用于对照和验证。
 */
@Slf4j
public class ClsProcessUpdate implements AutoCloseable {

    // ===== Static parameters (temporarily hardcoded) =====
    public static final int CLS_IMAGE_H = 48;
    public static final int CLS_IMAGE_W = 192;
    public static final int CLS_BATCH_SIZE = 16;
    public static final float CLS_THRESH = 0.9f;

    /**
     * 当输出类别缺少显式标签时，按常见顺序映射角度。
     * 2类场景通常为 [0, 180]；4类场景常见为 [0, 180, 90, 270]。
     */
    private static final int[] FALLBACK_ANGLE_MAP = {0, 180, 90, 270};

    private final OCRConfig config;
    private final OrtEnvironment env;
    private OrtSession session;

    public ClsProcessUpdate(OCRConfig config) throws OrtException {
        this.config = config;
        this.env = OrtEnvironment.getEnvironment();
        loadModel();
    }

    private void loadModel() throws OrtException {
        session = env.createSession(config.getClsModelPath(), ModelUtil.getSessionOptions(config));
    }

    /**
     * 对 context.boxes 做方向分类，并在高置信度时写入旋转后的 rotMat。
     */
    public void classify(ModelProcessContext context) {
        long start = System.currentTimeMillis();
        try {
            List<TextBox> boxes = context.getBoxes();
            if (boxes == null || boxes.isEmpty()) {
                context.setSuccess(true);
                context.setClsProcessTime(System.currentTimeMillis() - start);
                return;
            }

            int rotatedCount = 0;
            for (int begin = 0; begin < boxes.size(); begin += CLS_BATCH_SIZE) {
                int end = Math.min(begin + CLS_BATCH_SIZE, boxes.size());
                List<TextBox> batch = boxes.subList(begin, end);

                List<float[]> chwList = new ArrayList<>();
                for (TextBox box : batch) {
                    Mat src = box.getRawMat();
                    if (src == null || src.empty()) {
                        chwList.add(new float[3 * CLS_IMAGE_H * CLS_IMAGE_W]);
                        continue;
                    }
                    chwList.add(resizeNormImgForCls(src, CLS_IMAGE_H, CLS_IMAGE_W));
                }

                OnnxTensor input = createBatchTensor(chwList, env, 3, CLS_IMAGE_H, CLS_IMAGE_W);
                OrtSession.Result output = session.run(Collections.singletonMap("x", input));
                float[][] logits = parseClsOutput(output);

                for (int i = 0; i < batch.size(); i++) {
                    TextBox box = batch.get(i);
                    int[] decoded = decodeCls(logits[i]);
                    int angle = decoded[0];
                    float score = Float.intBitsToFloat(decoded[1]);

                    box.setAngle(angle);
                    box.setClsConfidence(score);
                    box.setClsAngle(angle);

                    if (needRotate(angle, score)) {
                        applyRotation(box, angle);
                        if (box.isRotate()) {
                            rotatedCount++;
                        }
                    } else {
                        box.setRotate(false);
                    }
                }

                output.close();
                input.close();
            }

            context.setBoxes(boxes);
            context.setClsRotBox(rotatedCount);
            context.setSuccess(true);
            context.setClsProcessTime(System.currentTimeMillis() - start);
        } catch (Exception e) {
            context.setSuccess(false);
            context.setError(e.getMessage());
            log.error("cls update failed", e);
        }
    }



    /**
     * 官方 cls 预处理核心：
     * 1) 按比例缩放到固定高
     * 2) 宽度不超过 cls_image_w，并在右侧零填充
     * 3) 归一化到 [-1, 1]
     * 4) 输出 CHW
     */
    private float[] resizeNormImgForCls(Mat src, int imgH, int imgW) {
        int srcH = src.rows();
        int srcW = src.cols();
        float ratio = srcH > 0 ? (float) srcW / (float) srcH : 1.0f;
        int resizedW = Math.min(imgW, Math.max(1, Math.round(imgH * ratio)));

        // 官方思路：固定高，宽按比例缩放并右侧 padding。
        Mat resized = new Mat();
        Imgproc.resize(src, resized, new Size(resizedW, imgH));
        Mat floatMat = new Mat();
        resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);
        resized.release();

        float[] hwc = new float[imgH * resizedW * 3];
        floatMat.get(0, 0, hwc);
        floatMat.release();

        // 右侧padding默认0，对应归一化后为-1（与官方零填充语义一致）。
        float[] chw = new float[3 * imgH * imgW];
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < imgH; h++) {
                for (int w = 0; w < imgW; w++) {
                    int chwIdx = (c * imgH + h) * imgW + w;
                    if (w < resizedW) {
                        int hwcIdx = (h * resizedW + w) * 3 + c;
                        chw[chwIdx] = (hwc[hwcIdx] - 0.5f) / 0.5f;
                    } else {
                        chw[chwIdx] = -1.0f;
                    }
                }
            }
        }
        return chw;
    }

    private OnnxTensor createBatchTensor(List<float[]> chwList, OrtEnvironment env,
                                         int channels, int height, int width) throws OrtException {
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
     * 解析 cls 输出为 [batch, num_classes]。
     */
    private float[][] parseClsOutput(OrtSession.Result output) throws OrtException {
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
     * 返回 [angle, Float.floatToIntBits(score)]，避免创建小对象。
     */
    private int[] decodeCls(float[] probs) {
        int bestIdx = 0;
        float best = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > best) {
                best = probs[i];
                bestIdx = i;
            }
        }

        int angle;
        if (probs.length == 2) {
            angle = bestIdx == 1 ? 180 : 0;
        } else if (bestIdx < FALLBACK_ANGLE_MAP.length) {
            angle = FALLBACK_ANGLE_MAP[bestIdx];
        } else {
            angle = 0;
        }
        return new int[]{angle, Float.floatToIntBits(best)};
    }

    private boolean needRotate(int angle, float score) {
        if (score < CLS_THRESH) {
            return false;
        }
        if (angle == 180) {
            return true;
        }
        return config.isUseAngleCls() && (angle == 90 || angle == 270);
    }

    private void applyRotation(TextBox box, int angle) {
        Mat src = box.getRawMat();
        if (src == null || src.empty()) {
            box.setRotate(false);
            return;
        }
        Mat dst = new Mat();
        if (angle == 180) {
            Core.rotate(src, dst, Core.ROTATE_180);
        } else if (angle == 90) {
            Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE);
        } else if (angle == 270) {
            Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE);
        } else {
            box.setRotate(false);
            dst.release();
            return;
        }
        box.setRotMat(dst);
        box.setRotate(true);
        box.setRotAngle(angle);
    }
    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
