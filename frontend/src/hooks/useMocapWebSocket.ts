import { useState, useEffect, useRef, useCallback } from 'react';
import { PlaybackMode } from './usePlaybackManager';

const WS_URL = 'ws://localhost:8001/ws';
const API_URL = 'http://localhost:8001';

interface TrackSegment {
    start: number;
    end: number;
}

interface TrackRange {
    id: string;
    segments: TrackSegment[];
    color: string;
    lastSeenFrame: number; // Track when we last saw this person
    face_crops?: Array<{image: string, quality: number, frame: number}>;
}

interface PersonData {
    vertices: number[][];
    cam_t?: number[];
    joints?: number[][];
    face_crops?: Array<{image: string, quality: number, frame: number}>;
}

interface StateResponse {
    active: boolean;
    initialized: boolean;
    video_path: string | null;
    current_frame: number | null;
    persons: Record<string, PersonData>;
    image_base64: string | null;
    faces: number[];
    track_ranges: Record<string, { 
        id: string; 
        segments: TrackSegment[];
        color: string;
    }>;
}

// Gap threshold: if person is missing for more than this many frames, start new segment
const GAP_THRESHOLD = 5;

// Candy color palette matching backend and Viewer3D
const CANDY_COLORS = [
    '#FF6B9D', // Pink
    '#C96EF5', // Purple
    '#6EC3F5', // Blue
    '#5EEEFF', // Cyan
    '#6EFFC6', // Mint
    '#FFE66D', // Yellow
    '#FFAA6D', // Orange
    '#FF8B7B', // Coral
];

// Generate client ID once outside component
const generateClientId = () => Math.random().toString(36).substring(7);

interface UseMocapWebSocketProps {
    onFrameReceived?: (frameIdx: number, persons: Record<string, PersonData>) => void;
    playbackMode?: PlaybackMode;
}

export const useMocapWebSocket = (props?: UseMocapWebSocketProps) => {
    const { onFrameReceived, playbackMode = PlaybackMode.LIVE } = props || {};

    const [status, setStatus] = useState<string>('Idle');
    const [persons, setPersons] = useState<Record<string, PersonData>>({});
    const [currentFrame, setCurrentFrame] = useState<number>(0);
    const [currentImage, setCurrentImage] = useState<string | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [faces, setFaces] = useState<number[]>([]);
    const [trackRanges, setTrackRanges] = useState<Record<string, TrackRange>>({});
    const [videoPath, setVideoPath] = useState<string | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const clientId = useRef<string | null>(null);
    
    // Initialize client ID only once
    if (clientId.current === null) {
        clientId.current = generateClientId();
    }

    // Fetch state from backend for recovery after page refresh
    const fetchState = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/state`);
            const state: StateResponse = await response.json();
            
            if (state.initialized && state.current_frame !== null) {
                console.log('Recovering state from backend:', state);
                
                if (state.persons && Object.keys(state.persons).length > 0) {
                    setPersons(state.persons);
                }
                if (state.current_frame !== null) {
                    setCurrentFrame(state.current_frame);
                }
                if (state.image_base64) {
                    setCurrentImage(state.image_base64);
                }
                if (state.faces && state.faces.length > 0) {
                    setFaces(state.faces);
                }
                if (state.track_ranges && Object.keys(state.track_ranges).length > 0) {
                    // Convert backend track_ranges to our format with lastSeenFrame
                    const converted: Record<string, TrackRange> = {};
                    for (const [id, track] of Object.entries(state.track_ranges)) {
                        const lastSegment = track.segments[track.segments.length - 1];
                        converted[id] = {
                            ...track,
                            lastSeenFrame: lastSegment?.end || 0
                        };
                    }
                    setTrackRanges(converted);
                }
                if (state.video_path) {
                    setVideoPath(state.video_path);
                    sessionStorage.setItem('mocap_video_path', state.video_path);
                }
                
                setStatus(state.active ? 'Processing...' : 'Ready (Recovered)');
            }
        } catch (error) {
            console.error('Failed to fetch state:', error);
        }
    }, []);

    useEffect(() => {
        let timeoutId: ReturnType<typeof setTimeout>;

        const connect = () => {
            const ws = new WebSocket(`${WS_URL}/${clientId.current}`);
            
            ws.onopen = () => {
                console.log('Connected to WebSocket');
                setIsConnected(true);
                fetchState();
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'status') {
                    setStatus(message.message);
                    if (message.faces) {
                        setFaces(message.faces);
                    }
                } else if (message.type === 'frame') {
                    const frameIdx = message.data.frame_idx;
                    const framePersons = message.data.persons;

                    // Only update state if in live mode
                    if (playbackMode === PlaybackMode.LIVE) {
                        setPersons(framePersons);
                        setCurrentFrame(frameIdx);
                        if (message.data.image_base64) {
                            setCurrentImage(message.data.image_base64);
                        }
                        setStatus(`Processing Frame ${frameIdx}`);
                    }

                    // Notify callback for caching (works in all modes)
                    if (onFrameReceived) {
                        onFrameReceived(frameIdx, framePersons);
                    }
                    
                    // Update track ranges with segment support
                    setTrackRanges(prev => {
                        const next = { ...prev };
                        const activeIds = new Set(Object.keys(message.data.persons));
                        
                        // Update tracks for persons in this frame
                        activeIds.forEach(id => {
                            if (!next[id]) {
                                // New track - create with first segment
                                const colorIndex = parseInt(id) % CANDY_COLORS.length;
                                next[id] = {
                                    id,
                                    segments: [{ start: frameIdx, end: frameIdx }],
                                    color: CANDY_COLORS[colorIndex],
                                    lastSeenFrame: frameIdx,
                                    face_crops: message.data.persons[id].face_crops || []
                                };
                            } else {
                                const track = next[id];
                                const gap = frameIdx - track.lastSeenFrame;
                                
                                if (gap > GAP_THRESHOLD) {
                                    // Person reappeared after gap - start new segment
                                    track.segments.push({ start: frameIdx, end: frameIdx });
                                } else {
                                    // Extend current segment
                                    const lastSegment = track.segments[track.segments.length - 1];
                                    if (lastSegment) {
                                        lastSegment.end = frameIdx;
                                    }
                                }
                                track.lastSeenFrame = frameIdx;
                                if (message.data.persons[id].face_crops) {
                                    track.face_crops = message.data.persons[id].face_crops;
                                }
                            }
                        });
                        
                        return next;
                    });

                    // Handle merge events
                    if (message.data.merge_events && message.data.merge_events.length > 0) {
                        setTrackRanges(prev => {
                            const next = { ...prev };
                            message.data.merge_events.forEach((evt: any) => {
                                const oldId = String(evt.old_id);
                                const newId = String(evt.new_id);
                                
                                if (next[oldId] && next[newId]) {
                                    console.log(`Merging track ${oldId} into ${newId}`);
                                    
                                    // Merge segments
                                    next[newId].segments = [
                                        ...next[newId].segments,
                                        ...next[oldId].segments
                                    ].sort((a, b) => a.start - b.start);
                                    
                                    // Merge face crops (avoid duplicates based on frame)
                                    const existingFrames = new Set(
                                        (next[newId].face_crops || []).map(c => c.frame)
                                    );
                                    
                                    if (next[oldId].face_crops) {
                                        const newCrops = next[oldId].face_crops!.filter(
                                            c => !existingFrames.has(c.frame)
                                        );
                                        next[newId].face_crops = [
                                            ...(next[newId].face_crops || []),
                                            ...newCrops
                                        ];
                                    }
                                    
                                    // Remove old track
                                    delete next[oldId];
                                }
                            });
                            return next;
                        });
                    }
                } else if (message.type === 'complete') {
                    setStatus('Processing Complete');
                } else if (message.type === 'error') {
                    setStatus(`Error: ${message.message}`);
                } else if (message.type === 'paused') {
                    setStatus('Processing Paused');
                } else if (message.type === 'resumed') {
                    setStatus('Processing Resumed');
                }
            };

            ws.onclose = () => {
                console.log('Disconnected from WebSocket');
                setIsConnected(false);
                timeoutId = setTimeout(connect, 2000);
            };

            wsRef.current = ws;
        };

        connect();

        return () => {
            if (wsRef.current) {
                wsRef.current.onclose = null;
                wsRef.current.close();
            }
            clearTimeout(timeoutId);
        };
    }, [fetchState]);

    const sendMessage = useCallback((message: Record<string, unknown>) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
        } else {
            console.error("WebSocket not connected");
        }
    }, []);

    return {
        status,
        setStatus,
        persons,
        currentFrame,
        currentImage,
        isConnected,
        faces,
        trackRanges,
        setTrackRanges,
        videoPath,
        setVideoPath,
        sendMessage
    };
};
