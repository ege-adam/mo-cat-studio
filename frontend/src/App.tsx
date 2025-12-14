import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Leva } from 'leva';
import { Viewer3D } from './components/Viewer3D';
import { Timeline } from './components/Timeline';
import { PlaybackControls } from './components/PlaybackControls';
import { useMocapWebSocket } from './hooks/useMocapWebSocket';
import { useMocapControls } from './hooks/useMocapControls';
import { usePlaybackManager, PlaybackMode } from './hooks/usePlaybackManager';
import { useKeyboardShortcuts, KeyboardShortcutsHelp } from './hooks/useKeyboardShortcuts';
import './App.css';

const API_URL = 'http://localhost:8001';

function App() {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const [cachedFrameIndices, setCachedFrameIndices] = useState<number[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Initialize playback manager
  const playback = usePlaybackManager(30); // 30 fps

  // WebSocket hook with playback integration
  const {
    status,
    currentFrame: liveCurrentFrame,
    currentImage,
    persons: livePersons,
    faces,
    isConnected,
    trackRanges,
    setTrackRanges,
    videoPath,
    setVideoPath,
    sendMessage
  } = useMocapWebSocket({
    onFrameReceived: (frameIdx, persons) => {
      // Cache frames as they arrive from live processing
      if (playback.mode === PlaybackMode.LIVE) {
        playback.addFrame(frameIdx, persons);
      }
    },
    playbackMode: playback.mode
  });

  // State for cached frame persons
  const [cachedPersons, setCachedPersons] = useState<Record<string, any>>({});

  // Load frame from cache when in replay/paused mode
  useEffect(() => {
    if (playback.mode !== PlaybackMode.LIVE) {
      playback.getFrame(playback.currentFrame).then(frame => {
        if (frame) {
          setCachedPersons(frame.persons);
        }
      });
    }
  }, [playback.currentFrame, playback.mode, playback]);

  // Update cached frame indices periodically
  useEffect(() => {
    const updateCachedFrames = async () => {
      const indices = await playback.getCachedFrames();
      setCachedFrameIndices(indices);
    };

    // Update immediately
    updateCachedFrames();

    // Update every 2 seconds
    const interval = setInterval(updateCachedFrames, 2000);
    return () => clearInterval(interval);
  }, [playback]);

  // Determine what to display based on mode
  const currentFrame = playback.mode === PlaybackMode.LIVE ? liveCurrentFrame : playback.currentFrame;
  const persons = playback.mode === PlaybackMode.LIVE ? livePersons : cachedPersons;

  // Use videoPath from WebSocket hook (recovered from backend state)
  // or from sessionStorage as fallback
  const videoPathRef = useRef<string | null>(null);
  
  // Sync videoPath from WebSocket hook to ref
  useEffect(() => {
    if (videoPath) {
      videoPathRef.current = videoPath;
      setUploadStatus(`Video: ${videoPath.split('/').pop()}`);
    } else {
      // Try to recover from sessionStorage
      const storedPath = sessionStorage.getItem('mocap_video_path');
      if (storedPath) {
        videoPathRef.current = storedPath;
        setVideoPath(storedPath);
        setUploadStatus(`Video: ${storedPath.split('/').pop()}`);
      }
    }
  }, [videoPath, setVideoPath]);

  const controls = useMocapControls(
    isConnected,
    videoPathRef.current,
    sendMessage,
    () => {
      setTrackRanges({}); // onStart: Clear timeline
      playback.clearCache(); // Clear cache on new video
      playback.switchToLive(); // Switch to live mode
    }
  );

  // Handle playback mode transitions
  useEffect(() => {
    if (playback.mode === PlaybackMode.LIVE) {
      // Resume backend processing
      sendMessage({ type: 'resume' });
    } else {
      // Pause backend processing when in replay/paused mode
      sendMessage({ type: 'pause' });
    }
  }, [playback.mode, sendMessage]);

  // Keyboard shortcuts
  useKeyboardShortcuts({
    mode: playback.mode,
    isPlaying: playback.isPlaying,
    onTogglePlayPause: playback.togglePlayPause,
    onStepForward: playback.stepForward,
    onStepBackward: playback.stepBackward,
    onJumpFrames: playback.jumpFrames,
    onGoToStart: playback.goToStart,
    onGoToEnd: playback.goToEnd,
    onSwitchToLive: playback.switchToLive,
    enabled: true
  });

  // Toggle keyboard help with '?'
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === '?' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        setShowKeyboardHelp(prev => !prev);
      }
    };
    window.addEventListener('keypress', handleKeyPress);
    return () => window.removeEventListener('keypress', handleKeyPress);
  }, []);
  
  // Extract raw values from Leva (may return tuples [value, setter])
  const getValue = <T,>(val: T | [T, unknown]): T => {
    if (Array.isArray(val) && val.length === 2 && typeof val[1] === 'function') {
      return val[0] as T;
    }
    return val as T;
  };
  
  const distanceLimit = getValue<number>(controls.distanceLimit);
  const groundHeight = getValue<number>(controls.groundHeight);
  const renderMode = getValue<string>(controls.renderMode);

  const handleExport = async (trackId: string, segmentIdx?: number) => {
      try {
          // Build URL - use segment endpoint if segmentIdx is provided
          const url = segmentIdx !== undefined
              ? `${API_URL}/export/${trackId}/segment/${segmentIdx}`
              : `${API_URL}/export/${trackId}`;
          
          const response = await fetch(url);
          if (!response.ok) throw new Error('Export failed');
          
          const blob = await response.blob();
          const downloadUrl = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = downloadUrl;
          // Include segment index in filename if applicable
          const filename = segmentIdx !== undefined
              ? `track_${trackId}_seg${segmentIdx}.glb`
              : `track_${trackId}.glb`;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(downloadUrl);
          document.body.removeChild(a);
      } catch (e) {
          alert(`Export failed: ${e}`);
      }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const selectedFile = event.target.files[0];
      await handleUpload(selectedFile);
    }
  };

  const handleUpload = async (fileToUpload: File) => {
    setUploading(true);
    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      videoPathRef.current = response.data.path;
      setVideoPath(response.data.path);
      // Save to sessionStorage for persistence
      sessionStorage.setItem('mocap_video_path', response.data.path);
      setUploadStatus(`Uploaded: ${response.data.filename}`);
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus('Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleSeek = (frameIdx: number) => {
    // Auto-exit LIVE mode when user seeks
    if (playback.mode === PlaybackMode.LIVE) {
      playback.setMode(PlaybackMode.PAUSED);
    }
    playback.seekToFrame(frameIdx);
  };

  return (
    <div className="app-container">
      {/* Floating Sidebar */}
      <div className="sidebar">
        <div className="logo-header">
          <img src="/logo.png" alt="Mo=Cat Studio" className="logo" />
          <div>
            <h1>Mo=Cat Studio</h1>
            <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem', margin: 0 }}>
              AI-Powered Motion Capture
            </p>
          </div>
        </div>

        <div className="control-group">
          <h3>📹 Upload Video</h3>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            disabled={uploading || !isConnected}
          />
          {uploading && <p style={{ color: 'var(--candy-yellow)', margin: '8px 0 0 0', fontSize: '0.9rem' }}>⏳ Uploading...</p>}
          {uploadStatus && <p style={{ color: 'var(--candy-mint)', margin: '8px 0 0 0', fontSize: '0.9rem' }}>✓ {uploadStatus}</p>}
        </div>

        <div className="control-group">
          <h3>📊 Status</h3>
          <div className="status-box">
            <div>
              <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`} />
              <span style={{ fontWeight: 600 }}>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div style={{ color: 'var(--text-secondary)' }}>{status}</div>
            {currentFrame > 0 && (
              <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                <span className="accent-cyan">Frame:</span> <span style={{ fontWeight: 600 }}>{currentFrame}</span>
                {' | '}
                <span className="accent-mint">Tracks:</span> <span style={{ fontWeight: 600 }}>{Object.keys(persons).length}</span>
              </div>
            )}
          </div>
        </div>

        <div className="control-group" style={{ flex: 1 }}>
          <h3>🎮 Controls</h3>
          <div className="leva-container">
            <Leva fill flat titleBar={false} />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <Viewer3D
              persons={persons}
              distanceLimit={distanceLimit}
              groundHeight={groundHeight}
              currentFrame={currentFrame}
              currentImage={currentImage}
              faces={faces}
              renderMode={renderMode}
              videoRef={videoRef}
              videoSrc={videoPath ? `${API_URL}${videoPath}` : null}
              fps={30}
          />
        </div>

        {/* Playback Controls */}
        <PlaybackControls
          mode={playback.mode}
          isPlaying={playback.isPlaying}
          currentFrame={playback.currentFrame}
          totalFrames={playback.totalFrames}
          playbackSpeed={playback.playbackSpeed}
          cachedFrameCount={playback.getCachedFrameCount()}
          onPlay={playback.play}
          onPause={playback.pause}
          onStepBackward={playback.stepBackward}
          onStepForward={playback.stepForward}
          onGoToStart={playback.goToStart}
          onGoToEnd={playback.goToEnd}
          onSwitchToLive={playback.switchToLive}
          onExitLive={playback.exitLive}
          onSetSpeed={playback.setPlaybackSpeed}
        />

        {/* Keyboard Shortcuts Help (Toggle with ?) */}
        {showKeyboardHelp && <KeyboardShortcutsHelp />}

        {/* Timeline */}
        <Timeline
            tracks={trackRanges}
            currentFrame={currentFrame}
            totalFrames={Math.max(playback.totalFrames, currentFrame + 100)}
            onExport={handleExport}
            onSeek={handleSeek}
            cachedFrames={cachedFrameIndices}
            isSeekable={true}
        />
      </div>
    </div>
  );
}

export default App;
