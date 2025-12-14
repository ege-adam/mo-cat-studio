import { useControls, button, folder } from 'leva';
import { useEffect } from 'react';

// Storage key for settings
const STORAGE_KEY = 'mocap_settings';

// Load settings from sessionStorage
const loadSettings = () => {
    try {
        const stored = sessionStorage.getItem(STORAGE_KEY);
        if (stored) {
            return JSON.parse(stored);
        }
    } catch (e) {
        console.error('Failed to load settings:', e);
    }
    return null;
};

// Settings type
interface MocapSettings {
    startFrame: number;
    skipFrames: number;
    distanceLimit: number;
    groundHeight: number;
    bboxThreshold: number;
    resizeFactor: number;
    useGpu: boolean;
    useFp16: boolean;
    renderMode: string;
    maxAge: number;
    ghostingThreshold: number;
    ghost3dThreshold: number;
    histMatchThreshold: number;
    disableGhostSuppression: boolean;
    // Face recognition settings
    enableFaceRecognition: boolean;
    embedRetryInterval: number;
    embedMaxRetries: number;
    embedQualityThreshold: number;
    galleryMaxAge: number;
    faceMatchThreshold: number;
    maxEmbeddingsPerPerson: number;
    embedSimilarityThreshold: number;
}

// Save settings to sessionStorage
const saveSettings = (settings: Partial<MocapSettings>) => {
    try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (e) {
        console.error('Failed to save settings:', e);
    }
};

// Default values
const defaults = {
    startFrame: 0,
    skipFrames: 1,
    distanceLimit: 20,
    groundHeight: 0,
    bboxThreshold: 0.5,
    resizeFactor: 1.0,
    useGpu: true,
    useFp16: false,
    renderMode: 'points',
    maxAge: 30,
    ghostingThreshold: 0.25,
    ghost3dThreshold: 0.5,
    histMatchThreshold: 0.3,
    disableGhostSuppression: false,
    // Face recognition defaults
    enableFaceRecognition: true,
    embedRetryInterval: 10,
    embedMaxRetries: 3,
    embedQualityThreshold: 0.4,
    galleryMaxAge: 300,
    faceMatchThreshold: 0.5,
    maxEmbeddingsPerPerson: 10,
    embedSimilarityThreshold: 0.9,
};

export const useMocapControls = (
    isConnected: boolean, 
    videoPath: string | null, 
    sendMessage: (msg: Record<string, unknown>) => void,
    onStart?: () => void
) => {
    // Load stored values or use defaults
    const stored = loadSettings();
    const initial = stored ? { ...defaults, ...stored } : defaults;

    const controls = useControls({
        startFrame: { value: initial.startFrame, min: 0, step: 1 },
        skipFrames: { value: initial.skipFrames, min: 1, step: 1 },
        distanceLimit: { value: initial.distanceLimit, min: 1, max: 100, step: 1, label: 'Dist Limit (m)' },
        groundHeight: { value: initial.groundHeight, min: -5, max: 5, step: 0.1, label: 'Ground Height' },
        bboxThreshold: { 
            value: initial.bboxThreshold, 
            min: 0.1, 
            max: 0.9, 
            step: 0.05, 
            label: 'Detection Confidence',
            hint: 'Lower: More detections (may include false positives) | Higher: Only confident detections (may miss partially visible people)'
        },
        resizeFactor: { value: initial.resizeFactor, min: 0.1, max: 1.0, step: 0.1, label: 'Resolution' },
        useGpu: { value: initial.useGpu, label: 'Use GPU' },
        useFp16: { value: initial.useFp16, label: 'Use FP16 (Fast)' },
        renderMode: { options: { 'Points': 'points', 'Mesh': 'mesh' }, value: initial.renderMode, label: 'Render Mode' },
        
        // Tracking & Ghost Detection Settings
        trackingSettings: folder({
            maxAge: { 
                value: initial.maxAge, 
                min: 1, 
                max: 300, 
                step: 1, 
                label: 'Max Track Age',
                hint: 'Frames before deleting lost track. Lower: Quickly delete disappeared people (faster ID recycling) | Higher: Keep tracks longer (better occlusion handling)'
            },
            ghostingThreshold: { 
                value: initial.ghostingThreshold, 
                min: 0.1, 
                max: 1.0, 
                step: 0.05, 
                label: '2D Ghost Distance',
                hint: 'Max movement as fraction of body size. Lower (0.25): Stricter - prevents ghost tracks | Higher: More lenient - allows faster movement'
            },
            ghost3dThreshold: { 
                value: initial.ghost3dThreshold, 
                min: 0.1, 
                max: 2.0, 
                step: 0.1, 
                label: '3D Ghost Distance (m)',
                hint: 'Max 3D world distance for merging tracks. Lower: Only merge nearby tracks (safer) | Higher: Merge tracks farther apart (may incorrectly merge different people)'
            },
            histMatchThreshold: { 
                value: initial.histMatchThreshold, 
                min: 0.1, 
                max: 1.0, 
                step: 0.05, 
                label: 'Appearance Match',
                hint: 'Appearance similarity for ghost merging (0=identical, 1=different). Lower (0.3): Must look very similar | Higher: Can merge different-looking people'
            },
            disableGhostSuppression: { 
                value: initial.disableGhostSuppression, 
                label: 'Disable Ghost Suppression',
                hint: 'Turn off ghost detection (duplicate person removal). Use for debugging or densely crowded scenes.'
            },
        }, { collapsed: true }),
        
        // Face Recognition & Re-ID Settings
        faceRecognition: folder({
            enableFaceRecognition: { 
                value: initial.enableFaceRecognition, 
                label: 'Enable Face Re-ID',
                hint: 'Use face recognition to re-identify people who left and returned. Requires InsightFace.'
            },
            embedRetryInterval: { 
                value: initial.embedRetryInterval, 
                min: 1, 
                max: 30, 
                step: 1, 
                label: 'Retry Interval',
                hint: 'Frames between face detection retries. Lower (5): Retry quickly (more computation) | Higher (20): Retry slowly (saves computation)'
            },
            embedMaxRetries: { 
                value: initial.embedMaxRetries, 
                min: 1, 
                max: 500, 
                step: 1, 
                label: 'Max Retries',
                hint: 'Maximum face detection attempts. Lower (2): Give up quickly | Higher (5): Keep trying longer'
            },
            embedQualityThreshold: { 
                value: initial.embedQualityThreshold, 
                min: 0.1, 
                max: 1.0, 
                step: 0.05, 
                label: 'Quality Threshold',
                hint: 'Min face quality to accept (0-1). Lower (0.3): Accept blurry/small faces (more re-IDs, less reliable) | Higher (0.6): Only clear faces (fewer re-IDs, more accurate)'
            },
            galleryMaxAge: { 
                value: initial.galleryMaxAge, 
                min: 30, 
                max: 3000, 
                step: 30, 
                label: 'Gallery Memory',
                hint: 'Frames to keep lost track embeddings. Lower (150 = 5s): Shorter memory | Higher (600 = 20s): Can re-identify people who left longer'
            },
            faceMatchThreshold: { 
                value: initial.faceMatchThreshold, 
                min: 0.3, 
                max: 0.9, 
                step: 0.05, 
                label: 'Match Threshold',
                hint: 'Face similarity for matching (0=different, 1=identical). Lower (0.4): More re-IDs (may merge different people) | Higher (0.7): Fewer re-IDs (more accurate)'
            },
            maxEmbeddingsPerPerson: { 
                value: initial.maxEmbeddingsPerPerson, 
                min: 1, 
                max: 20, 
                step: 1, 
                label: 'Max Face Variants',
                hint: 'Maximum face variants per person. More variants = better re-ID but slower processing.'
            },
            embedSimilarityThreshold: { 
                value: initial.embedSimilarityThreshold, 
                min: 0.7, 
                max: 0.99, 
                step: 0.01, 
                label: 'Face Variant Similarity',
                hint: 'Only add face variants less similar than this threshold (0=identical, 1=completely different). Lower = more variants.'
            },
        }, { collapsed: true }),
        
        process: button((get) => {
            if (!isConnected) {
                alert("Backend not connected");
                return;
            }
            if (videoPath) {
                if (onStart) onStart();
                sendMessage({
                    type: 'start',
                    video_path: videoPath,
                    start_frame: get('startFrame'),
                    skip_frames: get('skipFrames'),
                    distanceLimit: get('distanceLimit'),
                    ghosting_threshold: get('trackingSettings.ghostingThreshold'),
                    ghost_3d_threshold: get('trackingSettings.ghost3dThreshold'),
                    hist_match_threshold: get('trackingSettings.histMatchThreshold'),
                    use_gpu: get('useGpu'),
                    max_age: get('trackingSettings.maxAge'),
                    bbox_threshold: get('bboxThreshold'),
                    resize_factor: get('resizeFactor'),
                    use_fp16: get('useFp16'),
                    disable_ghost_suppression: get('trackingSettings.disableGhostSuppression'),
                    // Face recognition parameters
                    enable_face_recognition: get('faceRecognition.enableFaceRecognition'),
                    embed_retry_interval: get('faceRecognition.embedRetryInterval'),
                    embed_max_retries: get('faceRecognition.embedMaxRetries'),
                    embed_quality_threshold: get('faceRecognition.embedQualityThreshold'),
                    gallery_max_age: get('faceRecognition.galleryMaxAge'),
                    face_match_threshold: get('faceRecognition.faceMatchThreshold'),
                    maxEmbeddingsPerPerson: get('faceRecognition.maxEmbeddingsPerPerson'),
                    embedSimilarityThreshold: get('faceRecognition.embedSimilarityThreshold'),
                });
            } else {
                alert("Please upload a video first");
            }
        }, { disabled: !isConnected }),
        stop: button(() => {
            sendMessage({ type: 'stop' });
        }, { disabled: !isConnected })
    }, [isConnected, videoPath, sendMessage, onStart]);

    // Save settings whenever they change
    // Extract raw values (Leva returns tuples for some controls)
    const getValue = (val: unknown): unknown => {
        if (Array.isArray(val) && val.length === 2 && typeof val[1] === 'function') {
            return val[0]; // Extract value from [value, setter] tuple
        }
        return val;
    };

    useEffect(() => {
        const settings = {
            startFrame: getValue(controls.startFrame) as number,
            skipFrames: getValue(controls.skipFrames) as number,
            distanceLimit: getValue(controls.distanceLimit) as number,
            groundHeight: getValue(controls.groundHeight) as number,
            bboxThreshold: getValue(controls.bboxThreshold) as number,
            resizeFactor: getValue(controls.resizeFactor) as number,
            useGpu: getValue(controls.useGpu) as boolean,
            useFp16: getValue(controls.useFp16) as boolean,
            renderMode: getValue(controls.renderMode) as string,
            // Tracking settings
            maxAge: getValue(controls.maxAge) as number,
            ghostingThreshold: getValue(controls.ghostingThreshold) as number,
            ghost3dThreshold: getValue(controls.ghost3dThreshold) as number,
            histMatchThreshold: getValue(controls.histMatchThreshold) as number,
            disableGhostSuppression: getValue(controls.disableGhostSuppression) as boolean,
            // Face recognition settings
            enableFaceRecognition: getValue(controls.enableFaceRecognition) as boolean,
            embedRetryInterval: getValue(controls.embedRetryInterval) as number,
            embedMaxRetries: getValue(controls.embedMaxRetries) as number,
            embedQualityThreshold: getValue(controls.embedQualityThreshold) as number,
            galleryMaxAge: getValue(controls.galleryMaxAge) as number,
            faceMatchThreshold: getValue(controls.faceMatchThreshold) as number,
            maxEmbeddingsPerPerson: getValue(controls.maxEmbeddingsPerPerson) as number,
            embedSimilarityThreshold: getValue(controls.embedSimilarityThreshold) as number,
        };
        saveSettings(settings);
    }, [
        controls.startFrame,
        controls.skipFrames, 
        controls.distanceLimit,
        controls.groundHeight,
        controls.bboxThreshold,
        controls.resizeFactor,
        controls.useGpu,
        controls.useFp16,
        controls.renderMode,
        controls.maxAge,
        controls.ghostingThreshold,
        controls.ghost3dThreshold,
        controls.histMatchThreshold,
        controls.disableGhostSuppression,
        controls.enableFaceRecognition,
        controls.embedRetryInterval,
        controls.embedMaxRetries,
        controls.embedQualityThreshold,
        controls.galleryMaxAge,
        controls.faceMatchThreshold,
        controls.maxEmbeddingsPerPerson,
        controls.embedSimilarityThreshold
    ]);

    return controls;
};
