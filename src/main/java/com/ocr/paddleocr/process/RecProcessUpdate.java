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
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * RecProcessUpdate:
 * 独立的文本识别实现，按 PaddleOCR 官方 rec + CTC 思路组织。
 * 说明：该类暂时不接入主流程，仅用于对照和验证。
 */
@Slf4j
public class RecProcessUpdate implements AutoCloseable {

    // ===== Static parameters (temporarily hardcoded) =====
    public static final int REC_IMAGE_H = 48;
    public static final int REC_IMAGE_W = 320;
    public static final int REC_BATCH_SIZE = 16;
    public static final int BLANK_INDEX = 0;

    private final OCRConfig config;
    private final OrtEnvironment env;
    private OrtSession session;
    private final List<String> dict;
    private final Set<Integer> ignoredTokens;

    public RecProcessUpdate(OCRConfig config) throws OrtException {
        this.config = config;
        this.env = OrtEnvironment.getEnvironment();
        this.dict = readDictionary(config.getDictPath());
        this.ignoredTokens = initIgnoredTokens(config.getLang(), dict);
        loadModel();
    }

    private void loadModel() throws OrtException {
        session = env.createSession(config.getRecModelPath(), ModelUtil.getSessionOptions(config));
    }

    /**
     * 对 context.boxes 执行文本识别。
     * 输入图像优先使用 cls 阶段的 rotMat；若无则回退 rawMat。
     */
    public void recognize(ModelProcessContext context) {
        long start = System.currentTimeMillis();
        try {
            List<TextBox> boxes = context.getBoxes();
            if (boxes == null || boxes.isEmpty()) {
                context.setSuccess(true);
                context.setRecProcessTime(System.currentTimeMillis() - start);
                return;
            }

            for (int begin = 0; begin < boxes.size(); begin += REC_BATCH_SIZE) {
                int end = Math.min(begin + REC_BATCH_SIZE, boxes.size());
                List<TextBox> batch = boxes.subList(begin, end);

                List<float[]> chwList = new ArrayList<>();
                for (TextBox box : batch) {
                    Mat src = (box.getRotMat() != null && !box.getRotMat().empty())
                            ? box.getRotMat() : box.getRawMat();
                    if (src == null || src.empty()) {
                        chwList.add(new float[3 * REC_IMAGE_H * REC_IMAGE_W]);
                        continue;
                    }
                    chwList.add(resizeNormImgForRec(src, REC_IMAGE_H, REC_IMAGE_W));
                }

                OnnxTensor input = createBatchTensor(chwList, env, 3, REC_IMAGE_H, REC_IMAGE_W);
                OrtSession.Result output = session.run(Collections.singletonMap("x", input));
                float[][][] probs = parseRecOutput(output);

                for (int i = 0; i < batch.size(); i++) {
                    DecodedText decoded = ctcDecode(probs[i], dict, ignoredTokens);
                    TextBox box = batch.get(i);
                    box.setText(decoded.text);
                    box.setRecConfidence(decoded.confidence);
                }

                output.close();
                input.close();
            }

            context.setBoxes(boxes);
            context.setSuccess(true);
            context.setRecProcessTime(System.currentTimeMillis() - start);
        } catch (Exception e) {
            context.setSuccess(false);
            context.setError(e.getMessage());
            log.error("rec update failed", e);
        }
    }

    /**
     * 便捷入口：直接输入图像路径和检测框，执行 rec 并输出可视化结果。
     * 检测框坐标应在原图坐标系内，每个框四点顺序可任意。
     */
    public ModelProcessContext recognizeFromImage(String imagePath, List<List<Point>> boxes, String outputDir) {
        Mat image = Imgcodecs.imread(imagePath);
        if (image == null || image.empty()) {
            throw new IllegalArgumentException("Cannot read image: " + imagePath);
        }
        if (boxes == null || boxes.isEmpty()) {
            throw new IllegalArgumentException("boxes is empty");
        }

        ModelProcessContext context = new ModelProcessContext();
        context.setRawMat(image);
        context.setOriginalWidth(image.cols());
        context.setOriginalHeight(image.rows());

        List<TextBox> textBoxes = new ArrayList<>();
        for (List<Point> boxPts : boxes) {
            if (boxPts == null || boxPts.size() < 4) {
                continue;
            }
            TextBox tb = new TextBox();
            tb.setBoxPoint(boxPts);
            Mat crop = cropByBoundingRect(image, boxPts);
            tb.setRawMat(crop);
            textBoxes.add(tb);
        }
        context.setBoxes(textBoxes);

        recognize(context);
        if (context.isSuccess()) {
            saveRecVisualizations(context, outputDir);
        }
        return context;
    }

    /**
     * 官方 rec 预处理核心：
     * 1) 固定高、按比例缩放宽
     * 2) 宽不够时右侧padding
     * 3) 归一化到 [-1, 1]
     * 4) 输出 CHW
     */
    private float[] resizeNormImgForRec(Mat src, int imgH, int imgW) {
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

        // 右侧padding默认0，对应归一化后为-1。
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
     * 解析 rec 输出为 [batch, time, classes]。
     */
    private float[][][] parseRecOutput(OrtSession.Result output) throws OrtException {
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

    /**
     * CTC greedy decode：
     * - 每个时间步取最大概率类别
     * - 去重（连续重复）
     * - 去 blank / ignored token
     * - 映射字典得到最终文本与平均置信度
     */
    private DecodedText ctcDecode(float[][] timeSteps, List<String> dict, Set<Integer> ignoredTokens) {
        if (timeSteps == null || timeSteps.length == 0) {
            return new DecodedText("", 0.0f);
        }

        StringBuilder sb = new StringBuilder();
        float confSum = 0.0f;
        int confCount = 0;
        int prev = BLANK_INDEX;

        // 官方思路：按时间步做 greedy 取最大，再做 CTC 去重与去 blank。
        for (float[] step : timeSteps) {
            int bestIdx = 0;
            float bestProb = step[0];
            for (int i = 1; i < step.length; i++) {
                if (step[i] > bestProb) {
                    bestProb = step[i];
                    bestIdx = i;
                }
            }

            if (bestIdx == prev) {
                continue;
            }
            prev = bestIdx;

            if (ignoredTokens.contains(bestIdx)) {
                continue;
            }

            int dictIdx = bestIdx - 1; // 0 is blank
            if (dictIdx >= 0 && dictIdx < dict.size()) {
                sb.append(dict.get(dictIdx));
                confSum += bestProb;
                confCount++;
            }
        }

        String text = sb.toString().trim();
        float conf = confCount > 0 ? (confSum / confCount) : 0.0f;
        return new DecodedText(text, conf);
    }

    private List<String> readDictionary(String dictPath) {
        List<String> dictionary = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(dictPath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    dictionary.add(line);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to read dictionary: " + dictPath, e);
        }
        return dictionary;
    }

    /**
     * 官方常见策略：
     * - 始终忽略 blank(0)
     * - 部分语言配置下忽略空格 token（若字典中存在）
     */
    private Set<Integer> initIgnoredTokens(String lang, List<String> dictionary) {
        Set<Integer> set = new HashSet<>();
        set.add(BLANK_INDEX);

        boolean useSpaceChar = !"ch".equals(lang) && !"chi".equals(lang)
                && !"japan".equals(lang) && !"korean".equals(lang);
        if (!useSpaceChar) {
            int spacePos = -1;
            for (int i = 0; i < dictionary.size(); i++) {
                if (" ".equals(dictionary.get(i))) {
                    spacePos = i + 1; // +1 because blank index occupies 0
                    break;
                }
            }
            if (spacePos > 0) {
                set.add(spacePos);
            }
        }
        return set;
    }

    private Mat cropByBoundingRect(Mat image, List<Point> points) {
        double minX = points.stream().mapToDouble(p -> p.x).min().orElse(0);
        double minY = points.stream().mapToDouble(p -> p.y).min().orElse(0);
        double maxX = points.stream().mapToDouble(p -> p.x).max().orElse(0);
        double maxY = points.stream().mapToDouble(p -> p.y).max().orElse(0);
        int x1 = Math.max(0, (int) Math.floor(minX));
        int y1 = Math.max(0, (int) Math.floor(minY));
        int x2 = Math.min(image.cols() - 1, (int) Math.ceil(maxX));
        int y2 = Math.min(image.rows() - 1, (int) Math.ceil(maxY));
        int w = Math.max(1, x2 - x1 + 1);
        int h = Math.max(1, y2 - y1 + 1);
        return new Mat(image, new org.opencv.core.Rect(x1, y1, w, h)).clone();
    }

    private void saveRecVisualizations(ModelProcessContext context, String outputDir) {
        if (outputDir == null || outputDir.trim().isEmpty()) {
            return;
        }
        File dir = new File(outputDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // 1) 保存输入图
        Imgcodecs.imwrite(new File(dir, "rec_input.jpg").getAbsolutePath(), context.getRawMat());

        // 2) 保存识别结果可视化
        Mat vis = context.getRawMat().clone();
        Scalar boxColor = new Scalar(255, 0, 0);
        Scalar textColor = new Scalar(0, 255, 0);
        int idx = 0;
        for (TextBox tb : context.getBoxes()) {
            if (tb.getBoxPoint() == null || tb.getBoxPoint().size() < 4) {
                continue;
            }
            org.opencv.core.MatOfPoint poly = new org.opencv.core.MatOfPoint();
            poly.fromList(tb.getBoxPoint());
            Imgproc.polylines(vis, Collections.singletonList(poly), true, boxColor, 2);
            poly.release();

            Point p = tb.getBoxPoint().get(0);
            String txt = tb.getText() == null ? "" : tb.getText();
            String label = String.format("#%d %s (%.2f)", idx, txt, tb.getRecConfidence());
            Imgproc.putText(vis, label, new Point(p.x, Math.max(0, p.y - 4)),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1);

            if (tb.getRawMat() != null && !tb.getRawMat().empty()) {
                Imgcodecs.imwrite(new File(dir, String.format("rec_crop_%03d.jpg", idx)).getAbsolutePath(), tb.getRawMat());
            }
            idx++;
        }
        Imgcodecs.imwrite(new File(dir, "rec_boxes.jpg").getAbsolutePath(), vis);
        vis.release();
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
        ignoredTokens.clear();
    }

    private static class DecodedText {
        final String text;
        final float confidence;

        DecodedText(String text, float confidence) {
            this.text = text;
            this.confidence = confidence;
        }
    }
}
