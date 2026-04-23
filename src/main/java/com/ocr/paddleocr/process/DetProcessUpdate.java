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
import com.ocr.paddleocr.utils.OnnxUtil;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Independent DET implementation aligned with PaddleOCR DB post-process flow.
 * This class is intentionally not wired into the current OCR pipeline.
 */
@Slf4j
public class DetProcessUpdate implements AutoCloseable {

    // ===== Static parameters (temporarily hardcoded) =====
    public static final int DET_MAX_SIDE_LEN = 960;
    public static final int RESIZE_ALIGN = 32;

    public static final float DB_THRESH = 0.3f;
    public static final float DB_BOX_THRESH = 0.6f;
    public static final float DB_UNCLIP_RATIO = 2.0f;
    public static final int DB_MIN_SIZE = 3;
    public static final int DB_MAX_CANDIDATES = 1000;
    public static final boolean DB_USE_DILATION = false;
    public static final int DB_DILATE_KERNEL_SIZE = 2;
    public static final boolean DB_RETURN_POLYGON = false;

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final OCRConfig config;
    private final OrtEnvironment env;
    private OrtSession session;

    public DetProcessUpdate(OCRConfig config) throws OrtException {
        this.config = config;
        this.env = OrtEnvironment.getEnvironment();
        loadModel();
    }

    private void loadModel() throws OrtException {
        session = env.createSession(config.getDetModelPath(), ModelUtil.getSessionOptions(config));
    }

    public void detect(ModelProcessContext context) {
        long start = System.currentTimeMillis();
        try {
            if (context.getRawMat() == null || context.getRawMat().empty()) {
                throw new IllegalArgumentException("rawMat is empty");
            }

            if (context.getOriginalWidth() <= 0 || context.getOriginalHeight() <= 0) {
                context.setOriginalWidth(context.getRawMat().cols());
                context.setOriginalHeight(context.getRawMat().rows());
            }

            preprocess(context);
            OnnxTensor input = OnnxUtil.createInputTensor(context.getDetPrepMat(), env);
            OrtSession.Result output = session.run(Collections.singletonMap("x", input));

            float[][] probMap = parseOutput(output);
            List<List<Point>> boxes = dbPostProcess(probMap, context);

            context.setBoxes(boxes.stream().map(b -> {
                TextBox tb = new TextBox();
                tb.setBoxPoint(b);
                return tb;
            }).collect(Collectors.toList()));

            context.setSuccess(true);
            context.setDetProcessTime(System.currentTimeMillis() - start);

            output.close();
            input.close();
        } catch (Exception e) {
            context.setSuccess(false);
            context.setError(e.getMessage());
            log.error("det update failed", e);
        }
    }

    /**
     * 便捷入口：直接输入图像路径执行 DET，并保存可视化结果。
     * 用于快速观察该官方流程实现的预期检测效果。
     */
    public ModelProcessContext detectFromImage(String imagePath, String outputDir) {
        Mat image = Imgcodecs.imread(imagePath);
        if (image == null || image.empty()) {
            throw new IllegalArgumentException("Cannot read image: " + imagePath);
        }

        ModelProcessContext context = new ModelProcessContext();
        context.setRawMat(image);
        context.setOriginalWidth(image.cols());
        context.setOriginalHeight(image.rows());
        detect(context);

        if (context.isSuccess() && context.getBoxes() != null) {
            saveDetVisualizations(context, outputDir);
        }
        return context;
    }

    private void preprocess(ModelProcessContext context) {
        Mat raw = context.getRawMat();
        int srcW = raw.cols();
        int srcH = raw.rows();

        int maxSide = Math.max(srcW, srcH);
        float scale = maxSide > DET_MAX_SIDE_LEN ? (float) DET_MAX_SIDE_LEN / maxSide : 1.0f;
        int dstW = Math.max((int) (srcW * scale), RESIZE_ALIGN);
        int dstH = Math.max((int) (srcH * scale), RESIZE_ALIGN);
        dstW = Math.max((dstW / RESIZE_ALIGN) * RESIZE_ALIGN, RESIZE_ALIGN);
        dstH = Math.max((dstH / RESIZE_ALIGN) * RESIZE_ALIGN, RESIZE_ALIGN);

        // 官方思路：长边限制 + 32对齐，保证下采样后的特征图尺寸合法。
        Mat resized = new Mat();
        Imgproc.resize(raw, resized, new Size(dstW, dstH));

        Mat rgb = new Mat();
        // 官方 det 输入通常是 RGB 顺序。
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        resized.release();

        Mat normalized = normalizeRgbToFloat(rgb);
        rgb.release();

        context.setDetPrepMat(normalized);
        context.setDetPrepWidth(dstW);
        context.setDetPrepHeight(dstH);
        context.setScale(scale);
    }

    private Mat normalizeRgbToFloat(Mat rgb) {
        Mat floatMat = new Mat();
        rgb.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);

        List<Mat> channels = new ArrayList<>();
        Core.split(floatMat, channels);
        for (int i = 0; i < 3; i++) {
            Core.subtract(channels.get(i), new Scalar(MEAN[i]), channels.get(i));
            Core.divide(channels.get(i), new Scalar(STD[i]), channels.get(i));
        }
        Core.merge(channels, floatMat);
        for (Mat ch : channels) {
            ch.release();
        }
        return floatMat;
    }

    private float[][] parseOutput(OrtSession.Result output) throws OrtException {
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

    private List<List<Point>> dbPostProcess(float[][] probMap, ModelProcessContext context) {
        int h = probMap.length;
        int w = probMap[0].length;

        Mat prob = new Mat(h, w, CvType.CV_32FC1);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                prob.put(y, x, probMap[y][x]);
            }
        }

        // 官方思路：先 pred > thresh 二值化，再做可选膨胀。
        Mat bitmap = new Mat();
        Imgproc.threshold(prob, bitmap, DB_THRESH, 255, Imgproc.THRESH_BINARY);
        bitmap.convertTo(bitmap, CvType.CV_8UC1);

        if (DB_USE_DILATION) {
            Mat kernel = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT, new Size(DB_DILATE_KERNEL_SIZE, DB_DILATE_KERNEL_SIZE));
            Imgproc.dilate(bitmap, bitmap, kernel);
            kernel.release();
        }

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bitmap, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        bitmap.release();
        prob.release();

        contours.sort((a, b) -> Double.compare(Imgproc.contourArea(b), Imgproc.contourArea(a)));
        if (contours.size() > DB_MAX_CANDIDATES) {
            contours = new ArrayList<>(contours.subList(0, DB_MAX_CANDIDATES));
        }

        List<List<Point>> boxes = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area <= 1.0) {
                continue;
            }

            // 官方思路：按轮廓区域在概率图上的均值做 box score 过滤。
            double score = DBPostProcess.getScore(contour, probMap);
            if (score < DB_BOX_THRESH) {
                continue;
            }

            Point[] contourPoints = contour.toArray();
            MatOfPoint2f contour2f = new MatOfPoint2f(contourPoints);
            double perimeter = Imgproc.arcLength(contour2f, true);
            if (perimeter < 1e-6) {
                contour2f.release();
                continue;
            }

            // 官方 unclip 距离公式：area * unclip_ratio / perimeter。
            double distance = area * DB_UNCLIP_RATIO / perimeter;
            List<Point> expanded = DBPostProcess.unclipPolygon(contourPoints, distance);
            contour2f.release();
            if (expanded.size() < 4) {
                continue;
            }

            MatOfPoint2f expanded2f = new MatOfPoint2f(expanded.toArray(new Point[0]));
            double epsilon = 0.002 * Imgproc.arcLength(expanded2f, true);
            List<Point> approx = DBPostProcess.approxPolyDP(expanded, epsilon, true);

            List<Point> box;
            MatOfPoint2f box2f;
            if (DB_RETURN_POLYGON) {
                box = approx;
                box2f = new MatOfPoint2f(box.toArray(new Point[0]));
            } else {
                if (approx.size() < 4) {
                    expanded2f.release();
                    continue;
                }
                MatOfPoint2f approx2f = new MatOfPoint2f(approx.toArray(new Point[0]));
                RotatedRect rr = Imgproc.minAreaRect(approx2f);
                approx2f.release();
                Point[] vertices = new Point[4];
                rr.points(vertices);
                box = OpenCVUtil.orderPoints(Arrays.asList(vertices));
                box2f = new MatOfPoint2f(box.toArray(new Point[0]));
            }

            RotatedRect sizeRect = Imgproc.minAreaRect(box2f);
            box2f.release();
            expanded2f.release();
            if (Math.min(sizeRect.size.width, sizeRect.size.height) < DB_MIN_SIZE) {
                continue;
            }

            boxes.add(box);
        }

        return restoreAndClip(boxes, context.getScale(), context.getOriginalWidth(), context.getOriginalHeight());
    }

    private List<List<Point>> restoreAndClip(List<List<Point>> boxes, float scale, int srcW, int srcH) {
        float safeScale = scale <= 0 ? 1.0f : scale;
        List<List<Point>> restored = new ArrayList<>();
        for (List<Point> box : boxes) {
            List<Point> one = new ArrayList<>(box.size());
            for (Point p : box) {
                double x = p.x / safeScale;
                double y = p.y / safeScale;
                one.add(new Point(clamp(x, 0, srcW - 1), clamp(y, 0, srcH - 1)));
            }
            restored.add(one);
        }
        return restored;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(v, hi));
    }

    private void saveDetVisualizations(ModelProcessContext context, String outputDir) {
        if (outputDir == null || outputDir.trim().isEmpty()) {
            return;
        }
        File dir = new File(outputDir);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Failed to create output directory: " + dir.getAbsolutePath());
        }

        // 1) 保存输入图像
        Imgcodecs.imwrite(new File(dir, "det_input.jpg").getAbsolutePath(), context.getRawMat());

        // 2) 保存检测框可视化
        Mat vis = context.getRawMat().clone();
        Scalar color = new Scalar(0, 255, 0);
        for (TextBox tb : context.getBoxes()) {
            if (tb.getBoxPoint() == null || tb.getBoxPoint().size() < 4) {
                continue;
            }
            MatOfPoint poly = new MatOfPoint();
            poly.fromList(tb.getBoxPoint());
            Imgproc.polylines(vis, Collections.singletonList(poly), true, color, 2);
            poly.release();
        }
        Imgcodecs.imwrite(new File(dir, "det_boxes.jpg").getAbsolutePath(), vis);
        vis.release();
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
