package com.ocr.paddleocr.domain;

import lombok.Data;
import org.opencv.core.Point;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.RoundingMode;

@Data
public class Word implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 文本框四个顶点
     */
    private Point[] box;

    /**
     * 识别的文本内容
     */
    private String text;

    /**
     * 识别置信度 (0-1)
     */
    private float confidence;

    // ========== 构造函数 ==========

    public Word() {
    }

    private Word(Builder builder) {
        this.box = builder.box;
        this.text = builder.text;
        this.confidence = builder.confidence;
    }

    // ========== Setter ==========

    /**
     * 设置文本框顶点
     * 自动保留2位小数
     */
    public void setBox(Point[] box) {
        if (box == null) {
            this.box = null;
            return;
        }

        this.box = new Point[box.length];
        for (int i = 0; i < box.length; i++) {
            if (box[i] != null) {
                BigDecimal x = BigDecimal.valueOf(box[i].x)
                        .setScale(2, RoundingMode.HALF_UP);
                BigDecimal y = BigDecimal.valueOf(box[i].y)
                        .setScale(2, RoundingMode.HALF_UP);
                this.box[i] = new Point(x.doubleValue(), y.doubleValue());
            } else {
                this.box[i] = null;
            }
        }
    }

    /**
     * 设置置信度
     * 自动保留4位小数
     */
    public void setConfidence(float confidence) {
        BigDecimal bd = BigDecimal.valueOf(confidence)
                .setScale(4, RoundingMode.HALF_UP);
        this.confidence = bd.floatValue();
    }

    // ========== Builder ==========

    public static class Builder {
        private Point[] box;
        private String text;
        private float confidence;

        /**
         * 设置box (自动保留2位小数) 
         */
        public Builder box(Point[] box) {
            if (box != null) {
                this.box = new Point[box.length];
                for (int i = 0; i < box.length; i++) {
                    if (box[i] != null) {
                        // 使用BigDecimal保留2位小数, 与setter保持一致
                        BigDecimal x = BigDecimal.valueOf(box[i].x)
                                .setScale(2, RoundingMode.HALF_UP);
                        BigDecimal y = BigDecimal.valueOf(box[i].y)
                                .setScale(2, RoundingMode.HALF_UP);
                        this.box[i] = new Point(x.doubleValue(), y.doubleValue());
                    }
                }
            }
            return this;
        }

        /**
         * 设置文本
         */
        public Builder text(String text) {
            this.text = text;
            return this;
        }

        /**
         * 设置置信度 (自动保留4位小数) 
         */
        public Builder confidence(float confidence) {
            // 使用BigDecimal保留4位小数, 与setter保持一致
            BigDecimal bd = BigDecimal.valueOf(confidence)
                    .setScale(4, RoundingMode.HALF_UP);
            this.confidence = bd.floatValue();
            return this;
        }

        /**
         * 构建Word对象
         */
        public Word build() {
            return new Word(this);
        }
    }

    /**
     * 获取Builder实例
     */
    public static Builder builder() {
        return new Builder();
    }
}