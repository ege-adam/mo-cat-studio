import { useState, useRef, useEffect, useCallback } from 'react';
import { mocapDB } from '../utils/indexedDB';

export enum PlaybackMode {
  LIVE = 'live',
  REPLAY = 'replay',
  PAUSED = 'paused'
}

interface PersonData {
  vertices: number[][];
  cam_t?: number[];
  joints?: number[][];
}

interface CachedFrame {
  frame_idx: number;
  persons: Record<string, PersonData>;
  timestamp?: number;
}

interface PlaybackState {
  mode: PlaybackMode;
  isPlaying: boolean;
  currentFrame: number;
  totalFrames: number;
  playbackSpeed: number;
}

const STORAGE_KEY_STATE = 'mocap_playback_state';
const MAX_CACHE_SIZE = 500; // Limit cache to 500 frames to prevent memory issues

export const usePlaybackManager = (fps: number = 30) => {
  const frameCache = useRef<Map<number, CachedFrame>>(new Map());

  const [mode, setMode] = useState<PlaybackMode>(PlaybackMode.LIVE);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [totalFrames, setTotalFrames] = useState(0);

  const playbackTimerRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const [storageInfo, setStorageInfo] = useState<{ usage: number; quota: number; percentage: number }>({
    usage: 0,
    quota: 0,
    percentage: 0
  });

  // Initialize IndexedDB and request persistent storage
  useEffect(() => {
    const initDB = async () => {
      try {
        // Initialize IndexedDB
        await mocapDB.init();
        console.log('IndexedDB initialized');

        // Request persistent storage permission
        const isPersisted = await mocapDB.requestPersistentStorage();
        console.log('Persistent storage:', isPersisted ? 'granted' : 'not granted');

        // Load frame indices from IndexedDB to rebuild in-memory index
        const indices = await mocapDB.getAllFrameIndices();
        if (indices.length > 0) {
          console.log(`Found ${indices.length} cached frames in IndexedDB`);
          // Update totalFrames based on max cached frame
          const maxFrame = Math.max(...indices);
          setTotalFrames(maxFrame + 1);
        }

        // Update storage info
        const info = await mocapDB.estimateStorageUsage();
        setStorageInfo(info);
        console.log(`Storage usage: ${(info.usage / 1024 / 1024).toFixed(2)} MB / ${(info.quota / 1024 / 1024).toFixed(2)} MB (${info.percentage.toFixed(1)}%)`);
      } catch (err) {
        console.error('Failed to initialize IndexedDB:', err);
      }
    };

    initDB();
  }, []);

  // Load state from sessionStorage on mount (lightweight state only)
  useEffect(() => {
    try {
      const stateData = sessionStorage.getItem(STORAGE_KEY_STATE);
      if (stateData) {
        const state: PlaybackState = JSON.parse(stateData);
        setMode(state.mode);
        setCurrentFrame(state.currentFrame);
        setTotalFrames(state.totalFrames);
        setPlaybackSpeed(state.playbackSpeed);
      }
    } catch (err) {
      console.error('Failed to load state from sessionStorage:', err);
    }
  }, []);

  // Save state to sessionStorage (lightweight state only, no frame data)
  useEffect(() => {
    const state: PlaybackState = {
      mode,
      isPlaying: false, // Don't persist playing state
      currentFrame,
      totalFrames,
      playbackSpeed
    };
    try {
      sessionStorage.setItem(STORAGE_KEY_STATE, JSON.stringify(state));
    } catch (err) {
      // Ignore quota errors for state
    }
  }, [mode, currentFrame, totalFrames, playbackSpeed]);

  // Cache management with IndexedDB + in-memory LRU cache
  const addFrame = useCallback((frame_idx: number, persons: Record<string, PersonData>) => {
    const frame: CachedFrame = {
      frame_idx,
      persons,
      timestamp: Date.now()
    };

    // Add to in-memory cache with LRU eviction
    if (frameCache.current.size >= MAX_CACHE_SIZE) {
      const oldestKey = frameCache.current.keys().next().value;
      if (oldestKey !== undefined) {
        frameCache.current.delete(oldestKey);
      }
    }
    frameCache.current.set(frame_idx, frame);

    // Persist to IndexedDB asynchronously (non-blocking)
    mocapDB.addFrame(frame).catch(err => {
      console.error('Failed to persist frame to IndexedDB:', err);
    });

    // Update totalFrames if this is a new max
    if (frame_idx >= totalFrames) {
      setTotalFrames(frame_idx + 1);
    }

    // Update storage info periodically (every 50 frames)
    if (frame_idx % 50 === 0) {
      mocapDB.estimateStorageUsage().then(info => {
        setStorageInfo(info);
      }).catch(err => {
        console.error('Failed to estimate storage usage:', err);
      });
    }
  }, [totalFrames]);

  const getFrame = useCallback(async (frame_idx: number): Promise<CachedFrame | undefined> => {
    // Try in-memory cache first
    const cached = frameCache.current.get(frame_idx);
    if (cached) {
      return cached;
    }

    // Fall back to IndexedDB
    try {
      const frame = await mocapDB.getFrame(frame_idx);
      if (frame) {
        // Add to in-memory cache for faster access next time
        if (frameCache.current.size >= MAX_CACHE_SIZE) {
          const oldestKey = frameCache.current.keys().next().value;
          if (oldestKey !== undefined) {
            frameCache.current.delete(oldestKey);
          }
        }
        frameCache.current.set(frame_idx, frame);
      }
      return frame;
    } catch (err) {
      console.error('Failed to get frame from IndexedDB:', err);
      return undefined;
    }
  }, []);

  const hasFrame = useCallback((frame_idx: number): boolean => {
    return frameCache.current.has(frame_idx);
  }, []);

  const getCachedFrameCount = useCallback((): number => {
    return frameCache.current.size;
  }, []);

  const getCachedFrames = useCallback(async (): Promise<number[]> => {
    try {
      // Get all frame indices from IndexedDB (source of truth)
      const indices = await mocapDB.getAllFrameIndices();
      return indices.sort((a, b) => a - b);
    } catch (err) {
      console.error('Failed to get cached frames from IndexedDB:', err);
      // Fall back to in-memory cache
      return Array.from(frameCache.current.keys()).sort((a, b) => a - b);
    }
  }, []);

  const clearCache = useCallback(async () => {
    frameCache.current.clear();
    try {
      await mocapDB.clearAll();
      console.log('Cache cleared (memory + IndexedDB)');
      setStorageInfo({ usage: 0, quota: 0, percentage: 0 });
    } catch (err) {
      console.error('Failed to clear IndexedDB:', err);
    }
  }, []);

  // Playback controls
  const play = useCallback(() => {
    setIsPlaying(true);
    // Switch to REPLAY mode when playing
    if (mode !== PlaybackMode.REPLAY) {
      setMode(PlaybackMode.REPLAY);
    }
  }, [mode]);

  const pause = useCallback(() => {
    setIsPlaying(false);
    if (playbackTimerRef.current !== null) {
      cancelAnimationFrame(playbackTimerRef.current);
      playbackTimerRef.current = null;
    }
  }, []);

  const seekToFrame = useCallback((frame_idx: number) => {
    const clampedFrame = Math.max(0, Math.min(frame_idx, totalFrames - 1));
    setCurrentFrame(clampedFrame);
  }, [totalFrames]);

  const stepForward = useCallback(() => {
    seekToFrame(currentFrame + 1);
  }, [currentFrame, seekToFrame]);

  const stepBackward = useCallback(() => {
    seekToFrame(currentFrame - 1);
  }, [currentFrame, seekToFrame]);

  const jumpFrames = useCallback((delta: number) => {
    seekToFrame(currentFrame + delta);
  }, [currentFrame, seekToFrame]);

  const goToStart = useCallback(() => {
    const cachedFrames = getCachedFrames();
    if (cachedFrames.length > 0) {
      seekToFrame(cachedFrames[0]);
    } else {
      seekToFrame(0);
    }
  }, [getCachedFrames, seekToFrame]);

  const goToEnd = useCallback(() => {
    const cachedFrames = getCachedFrames();
    if (cachedFrames.length > 0) {
      seekToFrame(cachedFrames[cachedFrames.length - 1]);
    } else {
      seekToFrame(totalFrames - 1);
    }
  }, [getCachedFrames, totalFrames, seekToFrame]);

  // Auto-playback loop for replay mode
  useEffect(() => {
    if (!isPlaying || mode !== PlaybackMode.REPLAY) {
      if (playbackTimerRef.current !== null) {
        cancelAnimationFrame(playbackTimerRef.current);
        playbackTimerRef.current = null;
      }
      return;
    }

    const frameInterval = (1000 / fps) / playbackSpeed;

    const playbackLoop = (timestamp: number) => {
      if (!lastFrameTimeRef.current) {
        lastFrameTimeRef.current = timestamp;
      }

      const elapsed = timestamp - lastFrameTimeRef.current;

      if (elapsed >= frameInterval) {
        setCurrentFrame(prev => {
          const nextFrame = prev + 1;

          // Loop back to start if we reach the end
          if (nextFrame >= totalFrames) {
            return 0;
          }

          return nextFrame;
        });

        lastFrameTimeRef.current = timestamp;
      }

      playbackTimerRef.current = requestAnimationFrame(playbackLoop);
    };

    playbackTimerRef.current = requestAnimationFrame(playbackLoop);

    return () => {
      if (playbackTimerRef.current !== null) {
        cancelAnimationFrame(playbackTimerRef.current);
      }
      lastFrameTimeRef.current = 0;
    };
  }, [isPlaying, mode, playbackSpeed, fps, totalFrames]);

  // Mode transitions
  const switchToLive = useCallback(() => {
    pause();
    setMode(PlaybackMode.LIVE);
  }, [pause]);

  const exitLive = useCallback(() => {
    pause();
    setMode(PlaybackMode.PAUSED);
  }, [pause]);

  const switchToReplay = useCallback(() => {
    setMode(PlaybackMode.REPLAY);
    // Start from the beginning or current position
  }, []);

  const togglePlayPause = useCallback(() => {
    if (isPlaying) {
      pause();
      setMode(PlaybackMode.PAUSED);
    } else {
      play();
      if (mode === PlaybackMode.PAUSED || mode === PlaybackMode.LIVE) {
        setMode(PlaybackMode.REPLAY);
      }
    }
  }, [isPlaying, mode, play, pause]);

  return {
    // State
    mode,
    isPlaying,
    currentFrame,
    totalFrames,
    playbackSpeed,
    storageInfo,

    // State setters
    setMode,
    setIsPlaying,
    setCurrentFrame,
    setTotalFrames,
    setPlaybackSpeed,

    // Cache management
    addFrame,
    getFrame,
    hasFrame,
    getCachedFrameCount,
    getCachedFrames,
    clearCache,

    // Playback controls
    play,
    pause,
    togglePlayPause,
    seekToFrame,
    stepForward,
    stepBackward,
    jumpFrames,
    goToStart,
    goToEnd,

    // Mode transitions
    switchToLive,
    exitLive,
    switchToReplay
  };
};
