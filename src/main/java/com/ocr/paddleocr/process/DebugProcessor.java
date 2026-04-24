package com.ocr.paddleocr.process;

import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.utils.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

public class DebugProcessor {

    public static void printBoxes(OCRContext context, String debugPath){
        List<List<Point>> detectBoxes = new ArrayList<>();
        context.getDetResultBoxes().forEach(box -> detectBoxes.add(box.getRestorePoints()));
        Mat detectImage = ImageUtil.drawBoxes(context.getRawMat(),detectBoxes);
        String fileName = debugPath + "/detectBoxes.jpg";
        ImageUtil.saveImage(detectImage, fileName);
    }
}
