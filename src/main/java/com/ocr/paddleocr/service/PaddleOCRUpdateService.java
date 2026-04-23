package com.ocr.paddleocr.service;

import com.google.gson.Gson;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.ModelProcessContext;
import com.ocr.paddleocr.domain.OCRPrediction;
import com.ocr.paddleocr.domain.OCRResult;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.process.ClsProcessUpdate;
import com.ocr.paddleocr.process.DetProcessUpdate;
import com.ocr.paddleocr.process.RecProcessUpdate;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
public class PaddleOCRUpdateService implements Closeable {
    static {
        // Ensure OpenCV native library is loaded before calling Imgcodecs.imread.
        nu.pattern.OpenCV.loadLocally();
    }

    private static final String DET_MODEL_PATH = "src/main/java/resources/models/chi/det_model.onnx";
    private static final String CLS_MODEL_PATH = "src/main/java/resources/models/chi/cls_model.onnx";
    private static final String REC_MODEL_PATH = "src/main/java/resources/models/chi/rec_model.onnx";
    private static final String DICT_PATH = "src/main/java/resources/models/chi/ppocr_keys_v1.txt";

    private final OCRConfig runtimeConfig;
    private final Gson gson;
    private DetProcessUpdate detProcess;
    private ClsProcessUpdate clsProcess;
    private RecProcessUpdate recProcess;
    private volatile boolean initialized = false;
    private volatile boolean closed = false;

    public PaddleOCRUpdateService() throws Exception {
        this.runtimeConfig = buildRuntimeConfig();
        this.gson = new Gson();
        initialize();
    }

    private OCRConfig buildRuntimeConfig() {
        return OCRConfig.builder()
                .detModelPath(DET_MODEL_PATH)
                .clsModelPath(CLS_MODEL_PATH)
                .recModelPath(REC_MODEL_PATH)
                .dictPath(DICT_PATH)
                .useAngleCls(true)
                .build();
    }

    private void initialize() throws Exception {
        if (initialized) {
            return;
        }
        detProcess = new DetProcessUpdate(runtimeConfig);
        clsProcess = new ClsProcessUpdate(runtimeConfig);
        recProcess = new RecProcessUpdate(runtimeConfig);
        initialized = true;
    }


    public String ocr(String imagePath) {
        return gson.toJson(ocrToObject(imagePath));
    }


    public OCRResult ocrToObject(String imagePath) {
        checkState();
        long start = System.currentTimeMillis();

        OCRResult.OCRResultBuilder builder = OCRResult.builder()
                .imagePath(imagePath)
                .success(false);

        Mat image = null;
        ModelProcessContext context = new ModelProcessContext();
        try {
            image = Imgcodecs.imread(imagePath);
            if (image == null || image.empty()) {
                return builder.error("Cannot read image: " + imagePath)
                        .processingTime(System.currentTimeMillis() - start)
                        .build();
            }

            context.setRawMat(image);
            context.setOriginalWidth(image.cols());
            context.setOriginalHeight(image.rows());

            builder.imageWidth(image.cols()).imageHeight(image.rows());

            // 1) DET
            detProcess.detect(context);
            if (!context.isSuccess() || context.getBoxes() == null || context.getBoxes().isEmpty()) {
                return builder.error(context.getError() == null ? "No text boxes detected" : context.getError())
                        .processingTime(System.currentTimeMillis() - start)
                        .build();
            }


            cropTextBoxes(context);


            clsProcess.classify(context);
            if (!context.isSuccess()) {
                log.warn("ClsProcessUpdate failed, continue rec with raw crops: {}", context.getError());
                context.setSuccess(true);
            }


            recProcess.recognize(context);
            if (!context.isSuccess()) {
                return builder.error(context.getError() == null ? "Recognition failed" : context.getError())
                        .processingTime(System.currentTimeMillis() - start)
                        .build();
            }


            List<OCRPrediction> predictions = convertToPredictions(context.getBoxes());
            return builder.success(true)
                    .predictions(predictions)
                    .allText(buildAllText(predictions))
                    .processingTime(System.currentTimeMillis() - start)
                    .build();
        } catch (Exception e) {
            log.error("PaddleOCRUpdateService failed", e);
            return builder.error(e.getMessage())
                    .processingTime(System.currentTimeMillis() - start)
                    .build();
        } finally {
            releaseTextBoxMats(context.getBoxes());
            if (image != null && !image.empty()) {
                image.release();
            }
            if (context.getDetPrepMat() != null && !context.getDetPrepMat().empty()) {
                context.getDetPrepMat().release();
            }
        }
    }


    private static void cropTextBoxes(ModelProcessContext context) {
        if (context.getRawMat() == null || context.getRawMat().empty() || context.getBoxes() == null) {
            return;
        }
        Mat raw = context.getRawMat();
        for (TextBox box : context.getBoxes()) {
            if (box.getBoxPoint() == null || box.getBoxPoint().size() < 4) {
                continue;
            }
            try {
                Mat crop = OpenCVUtil.perspectiveTransformCrop(raw, box.getBoxPoint());
                if (!crop.empty()) {
                    box.setRawMat(crop);
                } else {
                    box.setRawMat(new Mat());
                }
            } catch (Exception e) {
                box.setRawMat(new Mat());
            }
        }
    }

    private List<OCRPrediction> convertToPredictions(List<TextBox> boxes) {
        if (boxes == null) {
            return new ArrayList<>();
        }
        return boxes.stream()
                .filter(b -> b.getText() != null && !b.getText().isEmpty())
                .map(b -> OCRPrediction.builder()
                        .box(b.getBoxPoint())
                        .text(b.getText())
                        .confidence(b.getRecConfidence())
                        .build())
                .collect(Collectors.toList());
    }

    private String buildAllText(List<OCRPrediction> predictions) {
        if (predictions == null || predictions.isEmpty()) {
            return "";
        }
        return predictions.stream()
                .map(OCRPrediction::getText)
                .collect(Collectors.joining("\n"));
    }

    private void releaseTextBoxMats(List<TextBox> boxes) {
        if (boxes == null) {
            return;
        }
        for (TextBox box : boxes) {
            if (box.getRawMat() != null && !box.getRawMat().empty()) {
                box.getRawMat().release();
            }
            if (box.getRotMat() != null && !box.getRotMat().empty()) {
                box.getRotMat().release();
            }
        }
    }

    private void checkState() {
        if (closed) {
            throw new IllegalStateException("PaddleOCRUpdateService is closed");
        }
        if (!initialized) {
            throw new IllegalStateException("PaddleOCRUpdateService is not initialized");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        try {
            if (detProcess != null) {
                detProcess.close();
            }
            if (clsProcess != null) {
                clsProcess.close();
            }
            if (recProcess != null) {
                recProcess.close();
            }
        } catch (Exception e) {
            log.error("Failed to close PaddleOCRUpdateService", e);
        }
        closed = true;
    }
}



