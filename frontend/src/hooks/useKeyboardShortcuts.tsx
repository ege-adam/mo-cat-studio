import React, { useEffect } from 'react';
import { PlaybackMode } from './usePlaybackManager';

interface UseKeyboardShortcutsProps {
    mode: PlaybackMode;
    isPlaying: boolean;
    onTogglePlayPause: () => void;
    onStepForward: () => void;
    onStepBackward: () => void;
    onJumpFrames: (delta: number) => void;
    onGoToStart: () => void;
    onGoToEnd: () => void;
    onSwitchToLive: () => void;
    enabled?: boolean;
}

export const useKeyboardShortcuts = ({
    mode,
    isPlaying,
    onTogglePlayPause,
    onStepForward,
    onStepBackward,
    onJumpFrames,
    onGoToStart,
    onGoToEnd,
    onSwitchToLive,
    enabled = true
}: UseKeyboardShortcutsProps) => {
    useEffect(() => {
        if (!enabled) return;

        const handleKeyDown = (e: KeyboardEvent) => {
            // Ignore if user is typing in an input field
            const target = e.target as HTMLElement;
            if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
                return;
            }

            // Only allow shortcuts when not in live mode (except 'L' to switch to live)
            if (mode === PlaybackMode.LIVE && e.key.toLowerCase() !== 'l') {
                return;
            }

            switch (e.key) {
                case ' ': // Space - Play/Pause
                    e.preventDefault();
                    onTogglePlayPause();
                    break;

                case 'ArrowRight': // Right arrow - Step forward
                    e.preventDefault();
                    if (e.shiftKey) {
                        onJumpFrames(10); // Shift + Right = Jump 10 frames forward
                    } else {
                        onStepForward();
                    }
                    break;

                case 'ArrowLeft': // Left arrow - Step backward
                    e.preventDefault();
                    if (e.shiftKey) {
                        onJumpFrames(-10); // Shift + Left = Jump 10 frames backward
                    } else {
                        onStepBackward();
                    }
                    break;

                case 'Home': // Home - Go to start
                    e.preventDefault();
                    onGoToStart();
                    break;

                case 'End': // End - Go to end
                    e.preventDefault();
                    onGoToEnd();
                    break;

                case 'l':
                case 'L': // L - Switch to live mode
                    e.preventDefault();
                    onSwitchToLive();
                    break;

                case 'k':
                case 'K': // K - Play/Pause (alternative, like YouTube)
                    e.preventDefault();
                    onTogglePlayPause();
                    break;

                case 'j':
                case 'J': // J - Jump backward 10 frames (like YouTube)
                    e.preventDefault();
                    onJumpFrames(-10);
                    break;

                case '>': // > - Step forward (with shift)
                case '.': // . - Step forward
                    e.preventDefault();
                    onStepForward();
                    break;

                case '<': // < - Step backward (with shift)
                case ',': // , - Step backward
                    e.preventDefault();
                    onStepBackward();
                    break;

                default:
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [
        mode,
        isPlaying,
        onTogglePlayPause,
        onStepForward,
        onStepBackward,
        onJumpFrames,
        onGoToStart,
        onGoToEnd,
        onSwitchToLive,
        enabled
    ]);
};

// Helper component to display keyboard shortcuts
export const KeyboardShortcutsHelp: React.FC = () => {
    return (
        <div style={{
            position: 'fixed',
            bottom: '220px',
            right: '20px',
            background: 'rgba(20, 20, 40, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '16px',
            fontSize: '12px',
            color: 'var(--text-secondary)',
            maxWidth: '250px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
            zIndex: 100
        }}>
            <h4 style={{
                margin: '0 0 12px 0',
                color: 'var(--candy-cyan)',
                fontSize: '14px',
                fontWeight: 600
            }}>
                ⌨️ Keyboard Shortcuts
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div><kbd>Space</kbd> or <kbd>K</kbd> - Play/Pause</div>
                <div><kbd>→</kbd> - Next frame</div>
                <div><kbd>←</kbd> - Previous frame</div>
                <div><kbd>Shift</kbd> + <kbd>→</kbd> - Jump 10 frames</div>
                <div><kbd>Shift</kbd> + <kbd>←</kbd> - Back 10 frames</div>
                <div><kbd>J</kbd> - Jump backward 10 frames</div>
                <div><kbd>,</kbd> or <kbd>&lt;</kbd> - Previous frame</div>
                <div><kbd>.</kbd> or <kbd>&gt;</kbd> - Next frame</div>
                <div><kbd>Home</kbd> - Go to start</div>
                <div><kbd>End</kbd> - Go to end</div>
                <div><kbd>L</kbd> - Switch to live mode</div>
            </div>
        </div>
    );
};
