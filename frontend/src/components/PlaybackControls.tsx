import React from 'react';
import { PlaybackMode } from '../hooks/usePlaybackManager';

interface PlaybackControlsProps {
    mode: PlaybackMode;
    isPlaying: boolean;
    currentFrame: number;
    totalFrames: number;
    playbackSpeed: number;
    cachedFrameCount: number;
    onPlay: () => void;
    onPause: () => void;
    onStepBackward: () => void;
    onStepForward: () => void;
    onGoToStart: () => void;
    onGoToEnd: () => void;
    onSwitchToLive: () => void;
    onExitLive: () => void;
    onSetSpeed: (speed: number) => void;
}

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
    mode,
    isPlaying,
    currentFrame,
    totalFrames,
    playbackSpeed,
    cachedFrameCount,
    onPlay,
    onPause,
    onStepBackward,
    onStepForward,
    onGoToStart,
    onGoToEnd,
    onSwitchToLive,
    onExitLive,
    onSetSpeed
}) => {
    const getModeColor = () => {
        switch (mode) {
            case PlaybackMode.LIVE:
                return 'var(--candy-mint)';
            case PlaybackMode.REPLAY:
                return 'var(--candy-cyan)';
            case PlaybackMode.PAUSED:
                return 'var(--candy-yellow)';
            default:
                return 'var(--text-dim)';
        }
    };

    const getModeIcon = () => {
        switch (mode) {
            case PlaybackMode.LIVE:
                return '🔴';
            case PlaybackMode.REPLAY:
                return '▶️';
            case PlaybackMode.PAUSED:
                return '⏸️';
            default:
                return '⏹️';
        }
    };

    const speedOptions = [0.25, 0.5, 1, 1.5, 2];

    return (
        <div className="playback-controls">
            {/* Mode Indicator */}
            <div className="mode-indicator" style={{ borderColor: getModeColor() }}>
                <span className="mode-icon">{getModeIcon()}</span>
                <span className="mode-label" style={{ color: getModeColor() }}>
                    {mode.toUpperCase()}
                </span>
            </div>

            {/* Transport Controls */}
            <div className="transport-controls">
                <button
                    onClick={onGoToStart}
                    className="transport-btn"
                    title="Go to Start (Home)"
                    disabled={mode === PlaybackMode.LIVE}
                >
                    ⏮
                </button>

                <button
                    onClick={onStepBackward}
                    className="transport-btn"
                    title="Step Backward (←)"
                    disabled={mode === PlaybackMode.LIVE}
                >
                    ⏪
                </button>

                <button
                    onClick={isPlaying ? onPause : onPlay}
                    className="transport-btn play-pause"
                    title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
                    disabled={mode === PlaybackMode.LIVE}
                >
                    {isPlaying ? '⏸' : '▶'}
                </button>

                <button
                    onClick={onStepForward}
                    className="transport-btn"
                    title="Step Forward (→)"
                    disabled={mode === PlaybackMode.LIVE}
                >
                    ⏩
                </button>

                <button
                    onClick={onGoToEnd}
                    className="transport-btn"
                    title="Go to End (End)"
                    disabled={mode === PlaybackMode.LIVE}
                >
                    ⏭
                </button>
            </div>

            {/* Frame Counter */}
            <div className="frame-counter">
                <span className="current-frame">{currentFrame}</span>
                <span className="frame-separator">/</span>
                <span className="total-frames">{totalFrames}</span>
            </div>

            {/* Playback Speed */}
            <div className="playback-speed">
                <label>Speed:</label>
                <select
                    value={playbackSpeed}
                    onChange={(e) => onSetSpeed(parseFloat(e.target.value))}
                    disabled={mode === PlaybackMode.LIVE}
                >
                    {speedOptions.map(speed => (
                        <option key={speed} value={speed}>
                            {speed}x
                        </option>
                    ))}
                </select>
            </div>

            {/* Cache Info */}
            <div className="cache-info">
                <span className="cache-icon">💾</span>
                <span className="cache-count">{cachedFrameCount} frames cached</span>
            </div>

            {/* Live Mode Button */}
            <button
                onClick={mode === PlaybackMode.LIVE ? onExitLive : onSwitchToLive}
                className="live-mode-btn"
                title={mode === PlaybackMode.LIVE ? "Exit Live Mode (L)" : "Switch to Live Mode (L)"}
            >
                {mode === PlaybackMode.LIVE ? '⏸ Exit Live' : '🔴 Go Live'}
            </button>
        </div>
    );
};
