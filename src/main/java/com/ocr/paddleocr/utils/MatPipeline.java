package com.ocr.paddleocr.utils;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * OpenCV Mat 处理管道 - DSL 风格
 * <p>
 * 所有中间资源会自动释放，最终 Mat 的所有权转移给调用者。
 * </p>
 *
 * <h3>使用示例：</h3>
 * <pre>{@code
 * // 方式1：获取最终 Mat，手动释放
 * Mat mat = MatPipeline.fromImage("input.jpg")
 *     .resize(640, 480)
 *     .toGray()
 *     .get();
 * try {
 *     // 使用 mat...
 * } finally {
 *     if (mat != null && !mat.empty()) {
 *         mat.release();
 *     }
 * }
 *
 * // 方式2：使用 use() 方法，自动释放所有资源
 * MatPipeline.fromImage("input.jpg")
 *     .resize(640, 480)
 *     .toGray()
 *     .use(mat -> Imgcodecs.imwrite("output.jpg", mat));
 *
 * // 方式3：使用 map() 方法，自动释放并返回结果
 * long nonZero = MatPipeline.fromImage("input.jpg")
 *     .toGray()
 *     .binary(127)
 *     .map(mat -> Core.countNonZero(mat));
 *
 * // 方式4：获取所有 Mat（调试用）
 * List<Mat> allMats = MatPipeline.fromImage("input.jpg")
 *     .resize(640, 480)
 *     .toGray()
 *     .getAll();
 * try {
 *     // 查看中间结果...
 * } finally {
 *     for (Mat m : allMats) {
 *         if (m != null && !m.empty()) {
 *             m.release();
 *         }
 *     }
 * }
 *
 * // 方式5：手动释放管道（不推荐，建议使用上述自动释放方式）
 * MatPipeline pipeline = MatPipeline.fromImage("input.jpg");
 * try {
 *     pipeline.resize(640, 480).toGray();
 *     Mat result = pipeline.get();
 *     // 使用 result...
 * } finally {
 *     pipeline.release();
 * }
 * }</pre>
 *
 * @author Luoky-dev
 * @version 1.0
 */
public class MatPipeline {

    static {
        // 加载OpenCV库
        nu.pattern.OpenCV.loadLocally();
    }

    private Mat current;
    private final List<Mat> resources = new ArrayList<>();

    private MatPipeline() {}

    // ==================== 静态工厂方法 ====================

    /**
     * 创建空的 MatPipeline 实例
     *
     * @return MatPipeline 实例
     */
    public static MatPipeline create() {
        return new MatPipeline();
    }

    /**
     * 从已有的 Mat 创建管道
     * <p>注意：管道会接管该 Mat 的所有权，调用者不应再手动释放</p>
     *
     * @param mat 已有的 Mat
     * @return MatPipeline 实例
     */
    public static MatPipeline fromMat(Mat mat) {
        return create().load(mat);
    }

    /**
     * 从概率图数组创建管道
     *
     * @param probMap 概率图二维数组
     * @return MatPipeline 实例
     */
    public static MatPipeline fromMap(float[][] probMap) {
        return create().loadMap(probMap);
    }

    /**
     * 从图像文件创建管道
     *
     * @param path 图像文件路径
     * @return MatPipeline 实例
     * @throws IllegalArgumentException 如果路径为空或图像无法读取
     */
    public static MatPipeline fromImage(String path) {
        if (path == null || path.trim().isEmpty()) {
            throw new IllegalArgumentException("Path cannot be empty");
        }
        Mat image = Imgcodecs.imread(path);
        if (image.empty()) {
            throw new IllegalArgumentException("Unable to read image");
        }
        return fromMat(image);
    }

    // ==================== 静态工厂方法（深拷贝版本） ====================

    /**
     * 从已有的 Mat 创建管道（深拷贝）
     * <p>传入的 Mat 不会被释放，调用者继续拥有所有权</p>
     *
     * @param mat 已有的 Mat
     * @return MatPipeline 实例
     */
    public static MatPipeline fromMatCopy(Mat mat) {
        Mat copy = new Mat();
        mat.copyTo(copy);
        return create().load(copy);
    }

    /**
     * 从概率图数组创建管道（深拷贝）
     * <p>创建独立的 Mat，不影响原始数组</p>
     *
     * @param probMap 概率图二维数组
     * @return MatPipeline 实例
     */
    public static MatPipeline fromMapCopy(float[][] probMap) {
        if (probMap == null || probMap.length == 0 || probMap[0].length == 0) {
            return create().load(new Mat());
        }

        int height = probMap.length;
        int width = probMap[0].length;
        Mat mat = new Mat(height, width, CvType.CV_32FC1);

        for (int i = 0; i < height; i++) {
            mat.put(i, 0, probMap[i]); // 一次写入整行
        }

        return create().load(mat);
    }

    // ==================== 加载方法 ====================

    /**
     * 加载 Mat 到管道
     *
     * @param mat 要加载的 Mat
     * @return this
     */
    private MatPipeline load(Mat mat) {
        this.current = mat;
        this.resources.add(mat);
        return this;
    }

    /**
     * 从概率图数组加载到管道
     *
     * @param probMap 概率图二维数组
     * @return this
     */
    private MatPipeline loadMap(float[][] probMap) {
        if (probMap == null || probMap.length == 0 || probMap[0].length == 0) {
            this.current = new Mat();
            resources.add(current);
            return this;
        }

        int height = probMap.length;
        int width = probMap[0].length;
        Mat mat = new Mat(height, width, CvType.CV_32FC1);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                mat.put(i, j, probMap[i][j]);
            }
        }

        this.current = mat;
        resources.add(mat);
        return this;
    }

    // ==================== 中间操作方法 ====================

    /**
     * 应用自定义操作
     *
     * @param operation 对当前 Mat 的操作函数
     * @return this
     */
    public MatPipeline apply(Function<Mat, Mat> operation) {
        if (empty()) return this;
        Mat result = operation.apply(current);
        if (result != null && result != current) {
            registerNewMat(result);
        }
        return this;
    }

    /**
     * 窥视当前Mat（用于调试）
     *
     * @param action 对当前 Mat 的操作
     * @return this
     */
    public MatPipeline peek(Consumer<Mat> action) {
        if (!empty()) {
            action.accept(current);
        }
        return this;
    }

    // ==================== 基础图像变换 ====================

    /**
     * 调整图像大小
     *
     * @param width  目标宽度
     * @param height 目标高度
     * @return this
     */
    public MatPipeline resize(int width, int height) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.resize(mat, result, new Size(width, height));
            return result;
        });
    }

    /**
     * 按比例调整图像大小
     *
     * @param fx x 轴缩放比例
     * @param fy y 轴缩放比例
     * @return this
     */
    public MatPipeline resize(double fx, double fy) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.resize(mat, result, new Size(0, 0), fx, fy, Imgproc.INTER_LINEAR);
            return result;
        });
    }

    /**
     * 转换为 RGB 色彩空间
     *
     * @return this
     */
    public MatPipeline cvtColor(int code) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.cvtColor(mat, result, code);
            return result;
        });
    }

    /**
     * 转换为灰度图
     *
     * @return this
     */
    public MatPipeline toGray() {
        return cvtColor(Imgproc.COLOR_BGR2GRAY);
    }

    /**
     * 转换为RGB
     */
    public MatPipeline toRGB() {
        return cvtColor(Imgproc.COLOR_BGR2RGB);
    }

    /**
     * 转换为BGR
     */
    public MatPipeline toBGR() {
        return cvtColor(Imgproc.COLOR_RGB2BGR);
    }

    /**
     * 类型转换
     */
    public MatPipeline convertTo(int rtype, double alpha, double beta) {
        return apply(mat -> {
            Mat result = new Mat();
            mat.convertTo(result, rtype, alpha, beta);
            return result;
        });
    }

    /**
     * 归一化到 [0, 1] 范围
     *
     * @return this
     */
    public MatPipeline normalize() {
        return convertTo(CvType.CV_32FC3, 1.0 / 255.0, 0);
    }

    /**
     * 自定义归一化
     *
     * @param alpha    归一化后的最小值
     * @param beta     归一化后的最大值
     * @param normType 归一化类型
     * @return this
     */
    public MatPipeline normalize(double alpha, double beta, int normType) {
        return apply(mat -> {
            Mat result = new Mat();
            Core.normalize(mat, result, alpha, beta, normType);
            return result;
        });
    }

    /**
     * 添加边框
     *
     * @param top        上边框宽度
     * @param bottom     下边框宽度
     * @param left       左边框宽度
     * @param right      右边框宽度
     * @param borderType 边框类型
     * @param value      边框填充值
     * @return this
     */
    public MatPipeline copyMakeBorder(int top, int bottom, int left, int right, int borderType, Scalar value) {
        return apply(mat -> {
            Mat result = new Mat();
            Core.copyMakeBorder(mat, result, top, bottom, left, right, borderType, value);
            return result;
        });
    }

    /**
     * 添加黑色边框
     *
     * @param top    上边框宽度
     * @param bottom 下边框宽度
     * @param left   左边框宽度
     * @param right  右边框宽度
     * @return this
     */
    public MatPipeline padding(int top, int bottom, int left, int right) {
        return copyMakeBorder(top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
    }

    /**
     * 旋转图像
     *
     * @param rotateCode 旋转代码（Core.ROTATE_90_CLOCKWISE 等）
     * @return this
     */
    public MatPipeline rotate(int rotateCode) {
        return apply(mat -> {
            Mat result = new Mat();
            Core.rotate(mat, result, rotateCode);
            return result;
        });
    }

    /**
     * 翻转图像
     *
     * @param flipCode 翻转代码（0: 垂直翻转，1: 水平翻转，-1: 同时翻转）
     * @return this
     */
    public MatPipeline flip(int flipCode) {
        return apply(mat -> {
            Mat result = new Mat();
            Core.flip(mat, result, flipCode);
            return result;
        });
    }

    /**
     * 高斯模糊
     *
     * @param ksize  内核大小
     * @param sigmaX X 方向标准差
     * @return this
     */
    public MatPipeline gaussianBlur(Size ksize, double sigmaX) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.GaussianBlur(mat, result, ksize, sigmaX);
            return result;
        });
    }

    /**
     * 中值模糊
     *
     * @param ksize 内核大小（必须是奇数）
     * @return this
     */
    public MatPipeline medianBlur(int ksize) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.medianBlur(mat, result, ksize);
            return result;
        });
    }

    /**
     * 直方图均衡化（仅适用于灰度图）
     *
     * @return this
     */
    public MatPipeline equalizeHist() {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.equalizeHist(mat, result);
            return result;
        });
    }

    // ==================== 阈值操作 ====================

    /**
     * 阈值分割
     *
     * @param thresh 阈值
     * @param maxval 最大值
     * @param type   阈值类型
     * @return this
     */
    public MatPipeline threshold(double thresh, double maxval, int type) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.threshold(mat, result, thresh, maxval, type);
            return result;
        });
    }

    /**
     * 二值化
     *
     * @param thresh 阈值
     * @return this
     */
    public MatPipeline binary(double thresh) {
        return threshold(thresh, 255, Imgproc.THRESH_BINARY);
    }

    /**
     * 反二值化
     *
     * @param thresh 阈值
     * @return this
     */
    public MatPipeline binaryInv(double thresh) {
        return threshold(thresh, 255, Imgproc.THRESH_BINARY_INV);
    }

    /**
     * 自适应阈值分割
     *
     * @param maxval         最大值
     * @param adaptiveMethod 自适应方法
     * @param thresholdType  阈值类型
     * @param blockSize      块大小（奇数）
     * @param C              常数
     * @return this
     */
    public MatPipeline adaptiveThreshold(int maxval, int adaptiveMethod, int thresholdType, int blockSize, double C) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.adaptiveThreshold(mat, result, maxval, adaptiveMethod, thresholdType, blockSize, C);
            return result;
        });
    }

    // ==================== 形态学操作 ====================

    /**
     * 创建形态学操作的结构元素
     * @param shape 结构元素的形状
     * @param ksize 核的大小（宽度, 高度）
     * @return this
     */
    public MatPipeline structuringElement(int shape, Size ksize) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.getStructuringElement(shape, ksize);
            return result;
        });
    }

    /**
     * 膨胀操作
     *
     * @param kernel 结构元素
     * @return this
     */
    public MatPipeline dilate(Mat kernel) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.dilate(mat, result, kernel);
            return result;
        });
    }

    /**
     * 膨胀操作（使用矩形结构元素）
     *
     * @param kernelSize 结构元素大小
     * @return this
     */
    public MatPipeline dilate(Size kernelSize) {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernelSize);
        try {
            return dilate(kernel);
        } finally {
            kernel.release();
        }
    }

    /**
     * 腐蚀操作
     *
     * @param kernel 结构元素
     * @return this
     */
    public MatPipeline erode(Mat kernel) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.erode(mat, result, kernel);
            return result;
        });
    }

    /**
     * 腐蚀操作（使用矩形结构元素）
     *
     * @param kernelSize 结构元素大小
     * @return this
     */
    public MatPipeline erode(Size kernelSize) {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernelSize);
        try {
            return erode(kernel);
        } finally {
            kernel.release();
        }
    }

    /**
     * 形态学变换
     * @param op 参数
     * @param kernel 结构元素
     * @return this
     */
    public MatPipeline morphologyEx(int op, Mat kernel) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.morphologyEx(mat, result, op, kernel);
            return result;
        });
    }

    /**
     * 开运算
     *
     * @param kernelSize 结构元素大小
     * @return this
     */
    public MatPipeline open(Size kernelSize) {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernelSize);
        try {
            return morphologyEx(Imgproc.MORPH_OPEN, kernel);
        } finally {
            kernel.release();
        }
    }

    /**
     * 闭运算
     *
     * @param kernelSize 结构元素大小
     * @return this
     */
    public MatPipeline close(Size kernelSize) {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernelSize);
        try {
            return morphologyEx(Imgproc.MORPH_CLOSE, kernel);
        } finally {
            kernel.release();
        }
    }

    // ==================== 边缘检测 ====================

    /**
     * Canny边缘检测
     *
     * @param threshold1 低阈值
     * @param threshold2 高阈值
     * @return this
     */
    public MatPipeline canny(double threshold1, double threshold2) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.Canny(mat, result, threshold1, threshold2);
            return result;
        });
    }

    /**
     * Sobel边缘检测
     *
     * @param dx    x 方向导数阶数
     * @param dy    y 方向导数阶数
     * @param ksize Sobel 内核大小
     * @return this
     */
    public MatPipeline sobel(int dx, int dy, int ksize) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.Sobel(mat, result, CvType.CV_32F, dx, dy, ksize);
            return result;
        });
    }

    /**
     * Laplacian边缘检测
     *
     * @param ksize 内核大小
     * @return this
     */
    public MatPipeline laplacian(int ksize) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.Laplacian(mat, result, CvType.CV_32F, ksize);
            return result;
        });
    }

    // ==================== 几何变换 ====================

    /**
     * 透视变换
     *
     * @param transform 透视变换矩阵
     * @param dsize     目标图像尺寸
     * @return this
     */
    public MatPipeline warpPerspective(Mat transform, Size dsize) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.warpPerspective(mat, result, transform, dsize);
            return result;
        });
    }

    /**
     * 仿射变换
     *
     * @param transform 仿射变换矩阵
     * @param dsize     目标图像尺寸
     * @return this
     */
    public MatPipeline warpAffine(Mat transform, Size dsize) {
        return apply(mat -> {
            Mat result = new Mat();
            Imgproc.warpAffine(mat, result, transform, dsize);
            return result;
        });
    }

    /**
     * 旋转变换（角度版）
     *
     * @param angle 旋转角度
     * @param center 旋转中心
     * @param scale 缩放比例
     * @return this
     */
    public MatPipeline warpRotate(double angle, Point center, double scale) {
        return apply(mat -> {
            Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, scale);
            Mat result = new Mat();
            Imgproc.warpAffine(mat, result, rotMat, mat.size());
            rotMat.release();
            return result;
        });
    }

    // ==================== 类型转换方法 ====================

    /**
     * 转换为 8位单通道（灰度/二值图像）
     * 适用于：轮廓查找、形态学操作、二值化
     */
    public MatPipeline toCV8UC1() {
        return convertTo(CvType.CV_8UC1);
    }

    /**
     * 转换为 8位3通道（BGR彩色图像）
     * 适用于：显示、保存、彩色图像处理
     */
    public MatPipeline toCV8UC3() {
        return convertTo(CvType.CV_8UC3);
    }

    /**
     * 转换为 8位4通道（BGRA图像）
     * 适用于：带透明通道的图像
     */
    public MatPipeline toCV8UC4() {
        return convertTo(CvType.CV_8UC4);
    }

    /**
     * 转换为 32位浮点单通道（概率图/深度图）
     * 适用于：模型输出、概率计算
     */
    public MatPipeline toCV32FC1() {
        return convertTo(CvType.CV_32FC1);
    }

    /**
     * 转换为 32位浮点3通道（归一化彩色图像）
     * 适用于：深度学习模型输入
     */
    public MatPipeline toCV32FC3() {
        return convertTo(CvType.CV_32FC3);
    }

    /**
     * 转换为 32位浮点4通道
     * 适用于：带透明通道的归一化图像
     */
    public MatPipeline toCV32FC4() {
        return convertTo(CvType.CV_32FC4);
    }

    /**
     * 转换为 64位浮点单通道
     * 适用于：高精度计算
     */
    public MatPipeline toCV64FC1() {
        return convertTo(CvType.CV_64FC1);
    }

    /**
     * 转换为 16位有符号单通道
     * 适用于：梯度计算
     */
    public MatPipeline toCV16SC1() {
        return convertTo(CvType.CV_16SC1);
    }

    /**
     * 转换为 16位无符号单通道
     * 适用于：深度图像
     */
    public MatPipeline toCV16UC1() {
        return convertTo(CvType.CV_16UC1);
    }

    /**
     * 转换为指定类型（使用 OpenCV 类型常量）
     * @param targetType 目标类型，如 CvType.CV_8UC1, CvType.CV_32FC3 等
     * @return this
     *
     * <p>使用示例：
     * <pre>{@code
     * // 转换为 8位单通道（二值图像）
     * pipeline.convertTo(CvType.CV_8UC1)
     *
     * // 转换为 32位浮点3通道（归一化图像）
     * pipeline.convertTo(CvType.CV_32FC3)
     * }</pre>
     */
    public MatPipeline convertTo(int targetType) {
        return apply(mat -> {
            if (mat.type() == targetType) {
                Mat result = new Mat();
                mat.copyTo(result);
                return result;
            }
            Mat result = new Mat();
            mat.convertTo(result, targetType);
            return result;
        });
    }

    /**
     * 转换为指定深度和通道数
     * @param depth 深度 (CvType.CV_8U, CV_16U, CV_32F, CV_64F)
     * @param channels 通道数 (1, 2, 3, 4)
     * @return this
     *
     * <p>使用示例：
     * <pre>{@code
     * // 转换为 8位3通道
     * pipeline.convertTo(CvType.CV_8U, 3)
     *
     * // 转换为 32位1通道
     * pipeline.convertTo(CvType.CV_32F, 1)
     * }</pre>
     */
    public MatPipeline convertTo(int depth, int channels) {
        int targetType = CvType.makeType(depth, channels);
        return convertTo(targetType);
    }

    // ==================== 私有辅助方法 ====================

    private boolean empty() {
        return current == null || current.empty();
    }

    private void registerNewMat(Mat mat) {
        if (mat != null && mat != current) {
            resources.add(mat);
            current = mat;
        }
    }

    // ==================== 终端操作方法 ====================

    /**
     * 获取最终处理的 Mat 并自动释放所有中间资源
     * <p>调用此方法后，管道不再可用</p>
     * <p><b>重要：返回的 Mat 需要调用者负责释放</b></p>
     *
     * <pre>{@code
     * Mat mat = pipeline.get();
     * try {
     *     // 使用 mat...
     * } finally {
     *     if (mat != null && !mat.empty()) {
     *         mat.release();
     *     }
     * }
     * }</pre>
     *
     * @return 最终处理的 Mat
     * @throws IllegalStateException 如果管道中没有可用的 Mat
     */
    public Mat get() {
        if (current == null) {
            throw new IllegalStateException("No Mat available in pipeline");
        }

        Mat result = current;

        for (Mat mat : resources) {
            if (mat != null && mat != result && !mat.empty()) {
                mat.release();
            }
        }

        resources.clear();
        current = null;

        return result;
    }

    /**
     * 获取所有 Mat（包括中间结果和最终结果）
     * <p>调用此方法后，管道不再可用</p>
     * <p><b>重要：返回的 List 中的所有 Mat 需要调用者负责释放</b></p>
     *
     * <pre>{@code
     * List<Mat> allMats = pipeline.getAll();
     * try {
     *     // 查看中间结果...
     *     Mat finalMat = allMats.get(allMats.size() - 1);
     * } finally {
     *     for (Mat m : allMats) {
     *         if (m != null && !m.empty()) {
     *             m.release();
     *         }
     *     }
     * }
     * }</pre>
     *
     * @return 所有 Mat 的列表（按处理顺序，最后一个为最终结果）
     */
    public List<Mat> getAll() {
        if (resources.isEmpty()) {
            return new ArrayList<>();
        }

        List<Mat> allMats = new ArrayList<>(resources);

        // 清空管道引用，所有权转移给调用者
        resources.clear();
        current = null;

        return allMats;
    }

    /**
     * 获取最终 Mat 并使用，自动释放所有资源（包括最终 Mat）
     * <p>这是最推荐的使用方式，无需手动释放内存</p>
     *
     * <pre>{@code
     * pipeline.use(mat -> {
     *     Imgcodecs.imwrite("output.jpg", mat);
     * });
     * }</pre>
     *
     * @param consumer 处理 Mat 的消费者
     * @throws IllegalStateException 如果管道中没有可用的 Mat
     */
    public void use(Consumer<Mat> consumer) {
        Mat mat = get();
        try {
            consumer.accept(mat);
        } finally {
            if (mat != null && !mat.empty()) {
                mat.release();
            }
        }
    }

    /**
     * 获取最终 Mat 并映射为其他值，自动释放所有资源（包括最终 Mat）
     * <p>适用于需要从图像中计算某个结果的场景</p>
     *
     * <pre>{@code
     * // 计算非零像素数量
     * long nonZero = pipeline.map(mat -> Core.countNonZero(mat));
     *
     * // 计算图像面积
     * double area = pipeline.map(mat -> mat.total() * mat.channels());
     *
     * // 提取特征值
     * double[] mean = pipeline.map(mat -> Core.mean(mat).val);
     *
     * // 自定义对象返回
     * DetectionResult result = pipeline.map(mat -> {
     *     return new DetectionResult(mat.width(), mat.height());
     * });
     * }</pre>
     *
     * @param mapper 映射函数，输入 Mat，输出任意类型的结果
     * @return 映射结果
     * @throws IllegalStateException 如果管道中没有可用的 Mat
     */
    public <R> R map(Function<Mat, R> mapper) {
        Mat mat = get();
        try {
            return mapper.apply(mat);
        } finally {
            if (mat != null && !mat.empty()) {
                mat.release();
            }
        }
    }
}