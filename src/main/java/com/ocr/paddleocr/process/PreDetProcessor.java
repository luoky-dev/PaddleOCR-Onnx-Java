package com.ocr.paddleocr.process;

import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.function.ToDoubleFunction;
import com.ocr.paddleocr.domain.TextBox;

/**
 * 预检测处理器：缩边 + 大图切块。
 * 仅新增独立类，不修改原有流程。
 */
@Slf4j
@Getter
public class PreDetProcessor {

    private final OCRConfig ocrConfig;
    private final ModelConfig modelConfig;

    // 可调参数（先固定为成员变量，避免改动其他配置类）
    private final int whiteThreshold = 245;
    private final double nonWhiteRatioThreshold = 0.005d;
    private final int minBorderPaddingPx = 12;
    private final int maxTileSideLen = 1600;
    private final double tileOverlapRatio = 0.15d;

    // 预处理结果（成员变量存储，供外部读取）
    private Mat preDetMat;
    private Rect preDetRoiInRaw;
    private List<TileRegion> tileRegions = Collections.emptyList();
    private List<Mat> tileMats = Collections.emptyList();

    public PreDetProcessor(ModelManager modelManager) {
        this.ocrConfig = modelManager.getOcrConfig();
        this.modelConfig = modelManager.getModelConfig();
    }

    public void preprocess(OCRContext context) {
        long start = System.currentTimeMillis();
        Mat raw = context.getRawMat();
        if (raw == null || raw.empty()) {
            reset();
            log.warn("PreDet skipped because input image is empty");
            return;
        }

        Rect roi = detectContentRoi(raw);
        Mat cropped = new Mat(raw, roi);

        List<TileRegion> regions = buildTileRegions(cropped.cols(), cropped.rows());
        List<Mat> tiles = new ArrayList<>(regions.size());
        for (TileRegion region : regions) {
            Mat tile = new Mat(cropped, region.getLocalRect());
            tiles.add(tile);
        }

        releasePreviousResults();
        this.preDetMat = cropped.clone();
        this.preDetRoiInRaw = roi;
        this.tileRegions = regions;
        this.tileMats = tiles;

        OpenCVUtil.releaseMat(cropped);
        log.info("PreDet done, raw={}x{}, roi={}x{}@({},{}), tiles={}, elapsed={} ms",
                raw.cols(), raw.rows(),
                roi.width, roi.height, roi.x, roi.y,
                tileRegions.size(),
                System.currentTimeMillis() - start);
    }

    public void reset() {
        releasePreviousResults();
        this.preDetMat = null;
        this.preDetRoiInRaw = null;
        this.tileRegions = Collections.emptyList();
        this.tileMats = Collections.emptyList();
    }

    private void releasePreviousResults() {
        OpenCVUtil.releaseMat(this.preDetMat);
        for (Mat tile : this.tileMats) {
            OpenCVUtil.releaseMat(tile);
        }
    }

    /**
     * 自动检测图像主体区域（用于缩边）。
     */
    private Rect detectContentRoi(Mat raw) {
        int width = raw.cols();
        int height = raw.rows();
        if (width <= 0 || height <= 0) {
            return new Rect(0, 0, 0, 0);
        }

        Mat gray = new Mat();
        Imgproc.cvtColor(raw, gray, Imgproc.COLOR_BGR2GRAY);

        Mat nonWhiteMask = new Mat();
        Imgproc.threshold(gray, nonWhiteMask, whiteThreshold, 255, Imgproc.THRESH_BINARY_INV);

        int nonZero = Core.countNonZero(nonWhiteMask);
        double ratio = (double) nonZero / Math.max(1, width * height);
        if (ratio < nonWhiteRatioThreshold) {
            OpenCVUtil.releaseMat(gray);
            OpenCVUtil.releaseMat(nonWhiteMask);
            return new Rect(0, 0, width, height);
        }

        MatOfPoint nz = new MatOfPoint();
        Core.findNonZero(nonWhiteMask, nz);
        Rect content = Imgproc.boundingRect(nz);

        int pad = Math.max(minBorderPaddingPx, modelConfig.getResizeAlign() / 2);
        Rect padded = expandRect(content, pad, width, height);

        OpenCVUtil.releaseMat(gray);
        OpenCVUtil.releaseMat(nonWhiteMask);
        OpenCVUtil.releaseMat(nz);
        return padded;
    }

    private Rect expandRect(Rect rect, int pad, int maxW, int maxH) {
        int x1 = Math.max(0, rect.x - pad);
        int y1 = Math.max(0, rect.y - pad);
        int x2 = Math.min(maxW, rect.x + rect.width + pad);
        int y2 = Math.min(maxH, rect.y + rect.height + pad);
        return new Rect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
    }

    /**
     * 根据尺寸切块，保留重叠区域减少边缘漏检。
     */
    private List<TileRegion> buildTileRegions(int width, int height) {
        int targetSide = Math.max(modelConfig.getDetMaxSideLen(), maxTileSideLen);
        if (width <= targetSide && height <= targetSide) {
            return Collections.singletonList(new TileRegion(
                    new Rect(0, 0, width, height),
                    new Rect(
                            preDetRoiInRaw == null ? 0 : preDetRoiInRaw.x,
                            preDetRoiInRaw == null ? 0 : preDetRoiInRaw.y,
                            width,
                            height
                    )));
        }

        int tileW = Math.min(targetSide, width);
        int tileH = Math.min(targetSide, height);
        int stepW = Math.max(1, (int) Math.round(tileW * (1.0d - tileOverlapRatio)));
        int stepH = Math.max(1, (int) Math.round(tileH * (1.0d - tileOverlapRatio)));

        List<TileRegion> regions = new ArrayList<>();
        for (int top = 0; top < height; top += stepH) {
            int h = Math.min(tileH, height - top);
            int y = (h < tileH) ? Math.max(0, height - tileH) : top;
            h = Math.min(tileH, height - y);

            for (int left = 0; left < width; left += stepW) {
                int w = Math.min(tileW, width - left);
                int x = (w < tileW) ? Math.max(0, width - tileW) : left;
                w = Math.min(tileW, width - x);

                Rect local = new Rect(x, y, w, h);
                Rect global = new Rect(
                        (preDetRoiInRaw == null ? 0 : preDetRoiInRaw.x) + x,
                        (preDetRoiInRaw == null ? 0 : preDetRoiInRaw.y) + y,
                        w,
                        h
                );
                if (!containsLocalRect(regions, local)) {
                    regions.add(new TileRegion(local, global));
                }
            }
        }
        return regions;
    }

    private boolean containsLocalRect(List<TileRegion> regions, Rect localRect) {
        for (TileRegion region : regions) {
            Rect r = region.getLocalRect();
            if (r.x == localRect.x && r.y == localRect.y && r.width == localRect.width && r.height == localRect.height) {
                return true;
            }
        }
        return false;
    }

    @Getter
    public static final class TileRegion {
        private final Rect localRect;
        private final Rect globalRect;

        public TileRegion(Rect localRect, Rect globalRect) {
            this.localRect = localRect;
            this.globalRect = globalRect;
        }
    }

    /**
     * 矩形 IoU NMS（默认按框面积降序作为分数）。
     */
    public static List<TextBox> nmsByRectIoU(List<TextBox> boxes, double iouThreshold) {
        return nmsByRectIoU(boxes, iouThreshold, PreDetProcessor::defaultScore);
    }

    /**
     * 矩形 IoU NMS（可自定义分数函数，如 det score/rec confidence）。
     */
    public static List<TextBox> nmsByRectIoU(List<TextBox> boxes,
                                             double iouThreshold,
                                             ToDoubleFunction<TextBox> scoreFunction) {
        if (boxes == null || boxes.isEmpty()) {
            return Collections.emptyList();
        }
        List<ScoredBox> sorted = new ArrayList<>(boxes.size());
        for (TextBox box : boxes) {
            sorted.add(new ScoredBox(box, scoreFunction.applyAsDouble(box)));
        }
        sorted.sort(Comparator.comparingDouble(ScoredBox::score).reversed());

        List<TextBox> kept = new ArrayList<>();
        boolean[] removed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (removed[i]) {
                continue;
            }
            TextBox current = sorted.get(i).box();
            kept.add(current);
            Rect a = toBoundingRect(current);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (removed[j]) {
                    continue;
                }
                Rect b = toBoundingRect(sorted.get(j).box());
                if (calcRectIoU(a, b) > iouThreshold) {
                    removed[j] = true;
                }
            }
        }
        return kept;
    }

    /**
     * 多边形近似 IoU NMS：先用外接矩形快速过滤，再用 mask 近似交并比。
     */
    public static List<TextBox> nmsByPolygonIoUApprox(List<TextBox> boxes, double iouThreshold) {
        return nmsByPolygonIoUApprox(boxes, iouThreshold, PreDetProcessor::defaultScore);
    }

    public static List<TextBox> nmsByPolygonIoUApprox(List<TextBox> boxes,
                                                      double iouThreshold,
                                                      ToDoubleFunction<TextBox> scoreFunction) {
        if (boxes == null || boxes.isEmpty()) {
            return Collections.emptyList();
        }
        List<ScoredBox> sorted = new ArrayList<>(boxes.size());
        for (TextBox box : boxes) {
            sorted.add(new ScoredBox(box, scoreFunction.applyAsDouble(box)));
        }
        sorted.sort(Comparator.comparingDouble(ScoredBox::score).reversed());

        List<TextBox> kept = new ArrayList<>();
        boolean[] removed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (removed[i]) {
                continue;
            }
            TextBox current = sorted.get(i).box();
            kept.add(current);
            Rect aRect = toBoundingRect(current);

            for (int j = i + 1; j < sorted.size(); j++) {
                if (removed[j]) {
                    continue;
                }
                TextBox candidate = sorted.get(j).box();
                Rect bRect = toBoundingRect(candidate);
                double fastIou = calcRectIoU(aRect, bRect);
                if (fastIou <= 0) {
                    continue;
                }
                // 先用矩形 IoU 做轻量过滤，降低 mask 运算开销
                if (fastIou < Math.min(0.1d, iouThreshold * 0.5d)) {
                    continue;
                }
                double polyIou = calcPolygonIoUApprox(current, candidate);
                if (polyIou > iouThreshold) {
                    removed[j] = true;
                }
            }
        }
        return kept;
    }

    private static double defaultScore(TextBox box) {
        Rect r = toBoundingRect(box);
        return Math.max(1.0d, (double) r.width * (double) r.height);
    }

    private static Rect toBoundingRect(TextBox box) {
        List<Point> points = box == null ? null : box.getRestorePoints();
        if (points == null || points.size() < 3) {
            return new Rect(0, 0, 1, 1);
        }
        MatOfPoint mat = new MatOfPoint();
        mat.fromList(points);
        Rect rect = Imgproc.boundingRect(mat);
        OpenCVUtil.releaseMat(mat);
        return rect.width > 0 && rect.height > 0 ? rect : new Rect(0, 0, 1, 1);
    }

    private static double calcRectIoU(Rect a, Rect b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);
        int iw = Math.max(0, x2 - x1);
        int ih = Math.max(0, y2 - y1);
        double inter = (double) iw * ih;
        double union = (double) a.width * a.height + (double) b.width * b.height - inter;
        return union <= 0 ? 0.0d : inter / union;
    }

    private static double calcPolygonIoUApprox(TextBox a, TextBox b) {
        List<Point> pa = a.getRestorePoints();
        List<Point> pb = b.getRestorePoints();
        if (pa == null || pb == null || pa.size() < 3 || pb.size() < 3) {
            return 0.0d;
        }

        Rect ra = toBoundingRect(a);
        Rect rb = toBoundingRect(b);
        int minX = Math.min(ra.x, rb.x);
        int minY = Math.min(ra.y, rb.y);
        int maxX = Math.max(ra.x + ra.width, rb.x + rb.width);
        int maxY = Math.max(ra.y + ra.height, rb.y + rb.height);
        int w = Math.max(1, maxX - minX);
        int h = Math.max(1, maxY - minY);

        Mat maskA = Mat.zeros(h, w, CvType.CV_8UC1);
        Mat maskB = Mat.zeros(h, w, CvType.CV_8UC1);

        MatOfPoint ma = new MatOfPoint();
        ma.fromList(shiftPoints(pa, minX, minY));
        MatOfPoint mb = new MatOfPoint();
        mb.fromList(shiftPoints(pb, minX, minY));

        Imgproc.fillPoly(maskA, Collections.singletonList(ma), new Scalar(255));
        Imgproc.fillPoly(maskB, Collections.singletonList(mb), new Scalar(255));

        Mat inter = new Mat();
        Mat uni = new Mat();
        Core.bitwise_and(maskA, maskB, inter);
        Core.bitwise_or(maskA, maskB, uni);
        double interArea = Core.countNonZero(inter);
        double unionArea = Core.countNonZero(uni);

        OpenCVUtil.releaseMat(maskA);
        OpenCVUtil.releaseMat(maskB);
        OpenCVUtil.releaseMat(ma);
        OpenCVUtil.releaseMat(mb);
        OpenCVUtil.releaseMat(inter);
        OpenCVUtil.releaseMat(uni);

        return unionArea <= 0 ? 0.0d : interArea / unionArea;
    }

    private static List<Point> shiftPoints(List<Point> src, int minX, int minY) {
        List<Point> shifted = new ArrayList<>(src.size());
        for (Point p : src) {
            shifted.add(new Point(p.x - minX, p.y - minY));
        }
        return shifted;
    }

    private static final class ScoredBox {
        private final TextBox box;
        private final double score;

        private ScoredBox(TextBox box, double score) {
            this.box = box;
            this.score = score;
        }

        private TextBox box() {
            return box;
        }

        private double score() {
            return score;
        }
    }
}
