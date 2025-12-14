import React, { useState } from 'react';
import ReactDOM from 'react-dom';

interface TrackSegment {
    start: number;
    end: number;
    history_start?: number;
    history_end?: number;
}

interface TrackRange {
    id: string;
    segments: TrackSegment[];
    color: string;
    face_crops?: Array<{image: string, quality: number, frame: number}>;
}

interface TimelineProps {
    tracks: Record<string, TrackRange>;
    currentFrame: number;
    totalFrames: number;
    onExport: (trackId: string, segmentIdx?: number) => void;
    onSeek?: (frameIdx: number) => void;
    cachedFrames?: number[];
    isSeekable?: boolean;
}

// Pixels per frame for timeline scale
const PIXELS_PER_FRAME = 4;
const MIN_TIMELINE_WIDTH = 800;

export const Timeline: React.FC<TimelineProps> = ({
    tracks,
    currentFrame,
    totalFrames,
    onExport,
    onSeek,
    cachedFrames = [],
    isSeekable = false
}) => {
    const [contextMenu, setContextMenu] = useState<{
        x: number;
        y: number;
        trackId: string;
        segmentIdx?: number;
    } | null>(null);

    const [hoverInfo, setHoverInfo] = useState<{
        trackId: string;
        x: number;
        y: number;
        faceCrops: Array<{image: string, quality: number, frame: number}>;
    } | null>(null);

    const [isDragging, setIsDragging] = useState(false);
    const timelineRef = React.useRef<HTMLDivElement>(null);

    const handleContextMenu = (e: React.MouseEvent, trackId: string, segmentIdx?: number) => {
        e.preventDefault();
        e.stopPropagation();

        // Adjust position if close to edges
        let x = e.clientX;
        let y = e.clientY;

        // Simple boundary check (assuming menu width ~200px, height ~300px)
        if (x + 200 > window.innerWidth) x = window.innerWidth - 210;
        if (y + 300 > window.innerHeight) y = window.innerHeight - 310;

        setContextMenu({ x, y, trackId, segmentIdx });
    };

    const handleCloseContextMenu = () => {
        setContextMenu(null);
    };

    // Convert pixel position to frame index
    const pixelToFrame = (pixelX: number): number => {
        if (!timelineRef.current) return 0;
        const rect = timelineRef.current.getBoundingClientRect();
        const relativeX = pixelX - rect.left + timelineRef.current.scrollLeft;
        const frameIdx = Math.floor(relativeX / PIXELS_PER_FRAME);
        return Math.max(0, Math.min(frameIdx, maxFrame - 1));
    };

    // Handle timeline click to seek
    const handleTimelineClick = (e: React.MouseEvent) => {
        // Close context menu when clicking anywhere on timeline
        handleCloseContextMenu();

        if (!isSeekable || !onSeek || isDragging) return;

        // Don't seek if clicking on a track segment
        const target = e.target as HTMLElement;
        if (target.closest('.track-segment')) return;

        const frameIdx = pixelToFrame(e.clientX);
        onSeek(frameIdx);
    };

    // Handle drag to scrub
    const handleMouseDown = (e: React.MouseEvent) => {
        if (!isSeekable || !onSeek) return;

        // Don't start dragging if clicking on a track segment
        const target = e.target as HTMLElement;
        if (target.closest('.track-segment')) return;

        setIsDragging(true);
        const frameIdx = pixelToFrame(e.clientX);
        onSeek(frameIdx);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDragging || !isSeekable || !onSeek) return;
        const frameIdx = pixelToFrame(e.clientX);
        onSeek(frameIdx);
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    // Add global mouse up listener when dragging
    React.useEffect(() => {
        if (isDragging) {
            const handleGlobalMouseUp = () => setIsDragging(false);
            window.addEventListener('mouseup', handleGlobalMouseUp);
            return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
        }
    }, [isDragging]);

    // Calculate max frame from all segments
    const maxFrame = Math.max(
        totalFrames,
        100,
        ...Object.values(tracks).flatMap(t => t.segments?.map(s => s.end) || [])
    );

    // Calculate timeline width in pixels
    const timelineWidth = Math.max(MIN_TIMELINE_WIDTH, maxFrame * PIXELS_PER_FRAME);

    // Frame markers for timeline ruler
    const frameMarkers = [];
    const markerInterval = Math.ceil(maxFrame / 20) * 5; // Round to nearest 5
    for (let i = 0; i <= maxFrame; i += markerInterval || 10) {
        frameMarkers.push(i);
    }

    return (
        <div
            ref={timelineRef}
            style={{
                width: '100%',
                height: '200px',
                background: 'rgba(15, 15, 30, 0.85)',
                backdropFilter: 'blur(20px)',
                borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                position: 'relative',
                overflowX: 'auto',
                overflowY: 'auto',
                boxShadow: '0 -4px 24px rgba(0, 0, 0, 0.3)',
                cursor: isSeekable ? 'pointer' : 'default'
            }}
            onClick={handleTimelineClick}
            onMouseDown={handleMouseDown}
            onMouseMove={(e) => {
                // Don't scrub if hovering over track segment
                const target = e.target as HTMLElement;
                if (target.closest('.track-segment')) return;
                handleMouseMove(e);
            }}
            onMouseUp={handleMouseUp}
        >
            {/* Timeline Content - scrollable area */}
            <div style={{
                width: `${timelineWidth}px`,
                minWidth: '100%',
                padding: '12px 0',
                position: 'relative',
                pointerEvents: isSeekable ? 'auto' : 'auto'
            }}>
                {/* Frame Ruler */}
                <div style={{
                    height: '30px',
                    borderBottom: '1px solid rgba(110, 195, 245, 0.2)',
                    position: 'relative',
                    marginBottom: '8px',
                    paddingLeft: '10px'
                }}>
                    {frameMarkers.map(frame => (
                        <div key={frame} style={{
                            position: 'absolute',
                            left: `${(frame / maxFrame) * (timelineWidth - 20) + 10}px`,
                            top: 0,
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center'
                        }}>
                            <span style={{
                                fontSize: '11px',
                                color: 'var(--candy-cyan)',
                                transform: 'translateX(-50%)',
                                fontWeight: 600,
                                fontFamily: '"Fira Code", monospace'
                            }}>{frame}</span>
                            <div style={{
                                width: '2px',
                                height: '10px',
                                background: 'linear-gradient(to bottom, var(--candy-cyan), transparent)',
                                marginTop: '4px'
                            }} />
                        </div>
                    ))}
                </div>

                {/* Cached Frames Indicator */}
                {cachedFrames.length > 0 && (
                    <div style={{
                        position: 'absolute',
                        top: '35px',
                        left: '10px',
                        right: 0,
                        height: '3px',
                        pointerEvents: 'none',
                        zIndex: 5
                    }}>
                        {cachedFrames.map((frameIdx, i) => {
                            // Group consecutive frames into ranges for performance
                            if (i > 0 && cachedFrames[i - 1] === frameIdx - 1) {
                                return null; // Skip, will be part of previous range
                            }

                            let rangeEnd = frameIdx;
                            while (i + 1 < cachedFrames.length && cachedFrames[i + 1] === rangeEnd + 1) {
                                rangeEnd = cachedFrames[i + 1];
                                i++;
                            }

                            const startX = (frameIdx / maxFrame) * (timelineWidth - 20);
                            const width = ((rangeEnd - frameIdx + 1) / maxFrame) * (timelineWidth - 20);

                            return (
                                <div
                                    key={frameIdx}
                                    style={{
                                        position: 'absolute',
                                        left: `${startX}px`,
                                        width: `${Math.max(width, 2)}px`,
                                        height: '100%',
                                        background: 'rgba(110, 239, 198, 0.4)',
                                        borderRadius: '1px'
                                    }}
                                />
                            );
                        })}
                    </div>
                )}

                {/* Time Cursor */}
                <div style={{
                    position: 'absolute',
                    left: `${(currentFrame / maxFrame) * (timelineWidth - 20) + 10}px`,
                    top: 0,
                    bottom: 0,
                    width: '3px',
                    background: 'linear-gradient(to bottom, var(--candy-pink), var(--candy-purple))',
                    zIndex: 10,
                    pointerEvents: 'none',
                    boxShadow: '0 0 12px rgba(255, 107, 157, 0.6)',
                    borderRadius: '2px'
                }} />

                {/* Cursor indicator at top */}
                <div style={{
                    position: 'absolute',
                    left: `${(currentFrame / maxFrame) * (timelineWidth - 20) + 10}px`,
                    top: '-6px',
                    width: '0',
                    height: '0',
                    borderLeft: '6px solid transparent',
                    borderRight: '6px solid transparent',
                    borderTop: '8px solid var(--candy-pink)',
                    transform: 'translateX(-6px)',
                    zIndex: 11,
                    pointerEvents: 'none',
                    filter: 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))'
                }} />

                {/* Tracks */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', padding: '0 10px' }}>
                    {Object.values(tracks).map((track) => (
                        <div
                            key={track.id}
                            style={{
                                position: 'relative',
                                height: '28px',
                                background: 'rgba(255, 255, 255, 0.03)',
                                backdropFilter: 'blur(10px)',
                                borderRadius: '8px',
                                width: `${timelineWidth - 20}px`,
                                border: '1px solid rgba(255, 255, 255, 0.05)',
                                transition: 'all 0.2s'
                            }}
                        >
                            {/* Track Label */}
                            <span style={{
                                position: 'absolute',
                                left: '10px',
                                top: '50%',
                                transform: 'translateY(-50%)',
                                fontSize: '12px',
                                color: 'var(--text-secondary)',
                                zIndex: 5,
                                pointerEvents: 'none',
                                fontWeight: 600,
                                textShadow: '0 1px 2px rgba(0, 0, 0, 0.5)'
                            }}>
                                Person {track.id}
                            </span>

                            {/* Render each segment */}
                            {(track.segments || []).map((segment, segIdx) => {
                                const left = (segment.start / maxFrame) * 100;
                                const width = ((segment.end - segment.start) / maxFrame) * 100;

                                return (
                                    <div
                                        key={segIdx}
                                        className="track-segment"
                                        onClick={(e) => e.stopPropagation()}
                                        onMouseDown={(e) => e.stopPropagation()}
                                        onContextMenu={(e) => handleContextMenu(e, track.id, segIdx)}
                                        style={{
                                            position: 'absolute',
                                            left: `${left}%`,
                                            width: `${Math.max(width, 0.3)}%`,
                                            height: '100%',
                                            background: `linear-gradient(135deg, ${track.color}, ${track.color}dd)`,
                                            borderRadius: '8px',
                                            cursor: 'pointer',
                                            opacity: 0.85,
                                            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                                            border: '1px solid rgba(255, 255, 255, 0.2)',
                                            boxShadow: `0 2px 8px ${track.color}40`,
                                        }}
                                        onMouseEnter={(e) => {
                                            e.stopPropagation();
                                            e.currentTarget.style.opacity = '1';
                                            e.currentTarget.style.transform = 'translateY(-2px)';
                                            e.currentTarget.style.boxShadow = `0 4px 16px ${track.color}60`;

                                            // Calculate position
                                            let x = e.clientX;
                                            let y = e.clientY;

                                            // Check vertical boundary (tooltip height approx 350px)
                                            const isNearBottom = window.innerHeight - y < 380;
                                            if (isNearBottom) {
                                                y -= 360;
                                            } else {
                                                y += 15;
                                            }

                                            // Check horizontal boundary (tooltip width approx 400px)
                                            if (x + 420 > window.innerWidth) {
                                                x = window.innerWidth - 430;
                                            } else {
                                                x += 10;
                                            }

                                            setHoverInfo({
                                                trackId: track.id,
                                                x,
                                                y,
                                                faceCrops: track.face_crops || []
                                            });
                                        }}
                                        onMouseLeave={(e) => {
                                            e.stopPropagation();
                                            e.currentTarget.style.opacity = '0.85';
                                            e.currentTarget.style.transform = 'translateY(0)';
                                            e.currentTarget.style.boxShadow = `0 2px 8px ${track.color}40`;
                                            setHoverInfo(null);
                                        }}
                                        title={`Segment ${segIdx + 1}: Frames ${segment.start}-${segment.end}`}
                                    />
                                );
                            })}
                        </div>
                    ))}

                    {/* Empty state */}
                    {Object.keys(tracks).length === 0 && (
                        <div style={{
                            color: 'var(--text-dim)',
                            textAlign: 'center',
                            padding: '30px',
                            fontSize: '14px',
                            background: 'rgba(255, 255, 255, 0.02)',
                            borderRadius: '12px',
                            border: '1px dashed rgba(255, 255, 255, 0.1)'
                        }}>
                            No tracks yet. Start processing to see timeline.
                        </div>
                    )}
                </div>
            </div>

            {/* Context Menu (Portaled to document.body) */}
            {contextMenu && ReactDOM.createPortal(
                <div style={{
                    position: 'fixed',
                    top: contextMenu.y,
                    left: contextMenu.x,
                    background: 'rgba(20, 20, 40, 0.95)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '12px',
                    padding: '8px',
                    zIndex: 2000,
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
                    minWidth: '200px'
                }}>
                    {/* Header showing what we're exporting */}
                    <div style={{
                        padding: '8px 12px',
                        color: 'var(--text-dim)',
                        fontSize: '11px',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                        marginBottom: '6px',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                    }}>
                        {contextMenu.segmentIdx !== undefined
                            ? `Track ${contextMenu.trackId} - Seg ${contextMenu.segmentIdx + 1}`
                            : `Track ${contextMenu.trackId} - All`
                        }
                    </div>

                    {/* Segment export options (if clicked on a segment) */}
                    {contextMenu.segmentIdx !== undefined && (
                        <>
                            <div
                                style={{
                                    padding: '10px 14px',
                                    cursor: 'pointer',
                                    color: 'white',
                                    fontSize: '14px',
                                    borderRadius: '6px',
                                    transition: 'all 0.2s',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px'
                                }}
                                onClick={() => {
                                    onExport(contextMenu.trackId, contextMenu.segmentIdx);
                                    handleCloseContextMenu();
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.background = 'rgba(110, 195, 245, 0.2)';
                                    e.currentTarget.style.transform = 'translateX(2px)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.background = 'transparent';
                                    e.currentTarget.style.transform = 'translateX(0)';
                                }}
                            >
                                <span style={{ color: 'var(--candy-cyan)' }}>📦</span>
                                Export Segment GLB
                            </div>
                        </>
                    )}

                    {/* Full track export options */}
                    <div
                        style={{
                            padding: '10px 14px',
                            cursor: 'pointer',
                            color: 'white',
                            fontSize: '14px',
                            borderRadius: '6px',
                            transition: 'all 0.2s',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                        }}
                        onClick={() => {
                            onExport(contextMenu.trackId);
                            handleCloseContextMenu();
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = 'rgba(201, 110, 245, 0.2)';
                            e.currentTarget.style.transform = 'translateX(2px)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'transparent';
                            e.currentTarget.style.transform = 'translateX(0)';
                        }}
                    >
                        <span style={{ color: 'var(--candy-purple)' }}>📦</span>
                        Export Full Track GLB
                    </div>
                    </div>,
                document.body
            )}

            {/* Hover Tooltip with Face Grid (Portaled to document.body) */}
            {hoverInfo && hoverInfo.faceCrops.length > 0 && ReactDOM.createPortal(
                <div style={{
                    position: 'fixed',
                    top: hoverInfo.y,
                    left: hoverInfo.x,
                    background: 'rgba(20, 20, 40, 0.95)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '16px',
                    padding: '16px',
                    zIndex: 1000,
                    boxShadow: '0 12px 48px rgba(0, 0, 0, 0.6)',
                    maxWidth: '420px',
                    pointerEvents: 'none'
                }}>
                    <div style={{
                        color: '#fff',
                        marginBottom: '12px',
                        fontSize: '13px',
                        fontWeight: 600,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                        paddingBottom: '8px'
                    }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ color: 'var(--candy-mint)' }}>👤</span>
                            Person {hoverInfo.trackId}
                        </span>
                        <span style={{
                            background: 'linear-gradient(135deg, var(--candy-purple), var(--candy-pink))',
                            padding: '3px 8px',
                            borderRadius: '6px',
                            fontSize: '10px',
                            fontWeight: 700
                        }}>
                            {hoverInfo.faceCrops.length} Face{hoverInfo.faceCrops.length !== 1 ? 's' : ''}
                        </span>
                    </div>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(75px, 1fr))',
                        gap: '10px',
                        maxHeight: '320px',
                        overflowY: 'auto'
                    }}>
                        {hoverInfo.faceCrops.map((crop, idx) => (
                            <div key={idx} style={{
                                textAlign: 'center',
                                background: 'rgba(0, 0, 0, 0.3)',
                                padding: '6px',
                                borderRadius: '8px',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                transition: 'all 0.2s'
                            }}>
                                <img
                                    src={`data:image/jpeg;base64,${crop.image}`}
                                    style={{
                                        width: '100%',
                                        aspectRatio: '1',
                                        objectFit: 'cover',
                                        borderRadius: '6px',
                                        border: '2px solid rgba(255, 255, 255, 0.15)',
                                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
                                    }}
                                    alt={`Face ${idx + 1}`}
                                />
                                <div style={{
                                    fontSize: '11px',
                                    color: crop.quality > 0.7 ? 'var(--candy-mint)' : crop.quality > 0.4 ? 'var(--candy-yellow)' : 'var(--candy-coral)',
                                    marginTop: '6px',
                                    fontWeight: 700,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '4px'
                                }}>
                                    <span>{crop.quality > 0.7 ? '✓' : crop.quality > 0.4 ? '~' : '!'}</span>
                                    {(crop.quality * 100).toFixed(0)}%
                                </div>
                                <div style={{
                                    fontSize: '9px',
                                    color: 'var(--text-dim)',
                                    marginTop: '2px'
                                }}>
                                    Frame #{crop.frame}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>,
                document.body
            )}
        </div>
    );
};
