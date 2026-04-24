package com.ocr.paddleocr.process;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Data
@NoArgsConstructor
public class ModelManager implements AutoCloseable {
    private static volatile ModelManager instance;
    private OrtEnvironment env;
    private OrtSession detSession;
    private OrtSession recSession;
    private OrtSession clsSession;
    private OCRConfig ocrConfig;
    private ModelConfig modelConfig;
    /**
     * 初始化标志
     */
    private volatile boolean initialized = false;
    /**
     * OpenCV加载标志
     */
    private static final AtomicBoolean openCVLoaded = new AtomicBoolean(false);

    /**
     * 获取单例实例
     */
    public static ModelManager getInstance() {
        if (instance == null) {
            synchronized (ModelManager.class) {
                if (instance == null) {
                    instance = new ModelManager();
                }
            }
        }
        return instance;
    }

    /**
     * 初始化模型管理器
     */
    public synchronized void init(OCRConfig ocrConfig) throws Exception {
        if (initialized) {
            log.warn("模型管理器已初始化, 跳过重复加载");
            return;
        }

        long startTime = System.currentTimeMillis();
        log.info("开始初始化模型管理器");

        this.ocrConfig = ocrConfig;
        this.modelConfig = new ModelConfig();

        // 加载OpenCV
        loadOpenCV();

        // 加载ONNX模型
        loadONNXModels();

        long loadTimeMs = System.currentTimeMillis() - startTime;
        this.initialized = true;

        log.info("模型管理器初始化完成, 耗时: {}ms", loadTimeMs);
    }

    /**
     * 加载OpenCV库
     */
    private void loadOpenCV() {
        if (openCVLoaded.get()) {
            return;
        }
        try {
            nu.pattern.OpenCV.loadLocally();
            openCVLoaded.set(true);
            log.info("OpenCV加载成功");
        } catch (Exception e) {
            log.error("OpenCV加载失败", e);
            throw new RuntimeException("OpenCV加载失败", e);
        }
    }

    /**
     * 加载ONNX模型
     */
    private void loadONNXModels() throws OrtException {
        log.info("开始加载ONNX模型");

        // 创建ONNX环境
        this.env = OrtEnvironment.getEnvironment();

        // 配置会话选项
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        sessionOptions.setInterOpNumThreads(ocrConfig.getNumThreads());
        sessionOptions.setIntraOpNumThreads(ocrConfig.getNumThreads());

        // 加载检测模型
        if (ocrConfig.getDetModelPath() != null) {
            log.info("加载检测模型: {}", ocrConfig.getDetModelPath());
            this.detSession = env.createSession(ocrConfig.getDetModelPath(), sessionOptions);
            log.info("检测模型加载完成");
        }

        // 加载识别模型
        if (ocrConfig.getRecModelPath() != null) {
            log.info("加载识别模型: {}", ocrConfig.getRecModelPath());
            this.recSession = env.createSession(ocrConfig.getRecModelPath(), sessionOptions);
            log.info("识别模型加载完成");
        }

        // 加载分类模型
        if (ocrConfig.isUseCls() && ocrConfig.getClsModelPath() != null) {
            log.info("加载分类模型: {}", ocrConfig.getClsModelPath());
            this.clsSession = env.createSession(ocrConfig.getClsModelPath(), sessionOptions);
            log.info("分类模型加载完成");
        }
    }

    /**
     * 获取检测模型会话
     */
    public OrtSession getDetSession() {
        checkInitialized();
        return detSession;
    }

    /**
     * 获取识别模型会话
     */
    public OrtSession getRecSession() {
        checkInitialized();
        return recSession;
    }

    /**
     * 获取分类模型会话
     */
    public OrtSession getClsSession() {
        checkInitialized();
        return clsSession;
    }

    /**
     * 获取ONNX环境
     */
    public OrtEnvironment getEnv() {
        checkInitialized();
        return env;
    }

    private void checkInitialized() {
        if (!initialized) {
            throw new IllegalStateException("模型管理器未初始化, 请先调用 init() 方法");
        }
    }

    /**
     * 释放资源
     */
    @Override
    public void close() {
        log.info("开始释放模型资源");

        try {
            if (detSession != null) {
                detSession.close();
                log.debug("检测模型已释放");
            }
            if (recSession != null) {
                recSession.close();
                log.debug("识别模型已释放");
            }
            if (clsSession != null) {
                clsSession.close();
                log.debug("分类模型已释放");
            }
            if (env != null) {
                env.close();
                log.debug("ONNX环境已释放");
            }
        } catch (OrtException e) {
            log.error("释放模型资源失败", e);
        }

        initialized = false;
        log.info("模型资源释放完成");
    }

}
