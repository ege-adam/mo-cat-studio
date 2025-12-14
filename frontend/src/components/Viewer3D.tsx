import React, { useMemo, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Environment, ContactShadows, SpotLight, Sky } from '@react-three/drei';
import * as THREE from 'three';

interface PersonData {
    vertices: number[][]; // [N, 3]
    cam_t?: number[];
    joints?: number[][];
}

interface Viewer3DProps {
    persons: Record<string, PersonData>;
    distanceLimit: number;
    groundHeight: number;
    currentFrame: number;
    currentImage: string | null;
    faces: number[];
    renderMode: string;
    videoRef?: React.RefObject<HTMLVideoElement>;
    videoSrc?: string | null;
    fps?: number;
}

// Candy color palette for persons
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

const PersonMesh: React.FC<{
    data: PersonData;
    color: string;
    groundHeight: number;
    faces: number[];
    renderMode: string;
}> = ({ data, color, groundHeight, faces, renderMode }) => {
    const geometryRef = useRef<THREE.BufferGeometry>(null);
    const meshRef = useRef<THREE.Mesh>(null);

    // Update geometry with new vertices
    React.useEffect(() => {
        if (geometryRef.current && data.vertices) {
            const rawVertices = data.vertices.flat();
            const vertices = new Float32Array(rawVertices.length);

            // Fix coordinate system: backend applies (x, -y, -z) transform
            // Undo the flip to get upright characters: (x, y, z)
            for (let i = 0; i < rawVertices.length; i += 3) {
                vertices[i] = rawVertices[i];          // X unchanged
                vertices[i + 1] = -rawVertices[i + 1]; // Y: negate to flip upright
                vertices[i + 2] = -rawVertices[i + 2]; // Z: negate to undo backend flip
            }

            // Update or create position attribute
            if (!geometryRef.current.attributes.position || geometryRef.current.attributes.position.count !== vertices.length / 3) {
                geometryRef.current.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            } else {
                const posAttr = geometryRef.current.attributes.position as THREE.BufferAttribute;
                posAttr.set(vertices);
                posAttr.needsUpdate = true;
            }

            // If mesh mode and faces available, set index and compute normals
            if (renderMode === 'mesh' && faces.length > 0) {
                if (!geometryRef.current.index) {
                    geometryRef.current.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));
                }
                geometryRef.current.computeVertexNormals();
            } else {
                // If switching from mesh to points, clear index
                if (geometryRef.current.index) {
                    geometryRef.current.setIndex(null);
                }
            }
        }
    }, [data.vertices, faces, renderMode]);

    // Center of mass for positioning
    const position = useMemo(() => {
        if (data.cam_t) {
            return new THREE.Vector3(data.cam_t[0], -data.cam_t[1] + groundHeight, -data.cam_t[2]);
        }
        return new THREE.Vector3(0, groundHeight, 0);
    }, [data.cam_t, groundHeight]);

    return (
        <group position={position}>
            {renderMode === 'mesh' && faces.length > 0 ? (
                <mesh ref={meshRef} castShadow receiveShadow>
                    <bufferGeometry ref={geometryRef} />
                    <meshStandardMaterial
                        color={color}
                        roughness={0.4}
                        metalness={0.2}
                        side={THREE.DoubleSide}
                        emissive={color}
                        emissiveIntensity={0.1}
                    />
                </mesh>
            ) : (
                <points>
                    <bufferGeometry ref={geometryRef} />
                    <pointsMaterial
                        size={0.025}
                        color={color}
                        sizeAttenuation
                        transparent
                        opacity={0.9}
                    />
                </points>
            )}
        </group>
    );
};

// Distance ring component with candy colors
const DistanceRing: React.FC<{ radius: number }> = ({ radius }) => {
    return (
        <group>
            {/* Main ring */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
                <ringGeometry args={[radius - 0.05, radius + 0.05, 64]} />
                <meshBasicMaterial
                    color="#FF6B9D"
                    opacity={0.3}
                    transparent
                    side={THREE.DoubleSide}
                />
            </mesh>
            {/* Glowing inner ring */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
                <ringGeometry args={[radius - 0.1, radius + 0.1, 64]} />
                <meshBasicMaterial
                    color="#C96EF5"
                    opacity={0.15}
                    transparent
                    side={THREE.DoubleSide}
                />
            </mesh>
        </group>
    );
};

export const Viewer3D: React.FC<Viewer3DProps> = ({
    persons,
    distanceLimit,
    groundHeight,
    currentFrame,
    currentImage,
    faces,
    renderMode,
    videoRef,
    videoSrc,
    fps = 30
}) => {
    // Sync video with current frame
    React.useEffect(() => {
        if (videoRef?.current && videoRef.current.readyState >= 2) {
            const targetTime = currentFrame / fps;
            // Only seek if difference is significant (> 1 frame)
            if (Math.abs(videoRef.current.currentTime - targetTime) > (1 / fps)) {
                videoRef.current.currentTime = targetTime;
            }
        }
    }, [currentFrame, fps, videoRef]);
    return (
        <div style={{
            width: '100%',
            height: '100%',
            background: 'linear-gradient(180deg, #0F0F1E 0%, #1A1A3E 100%)',
            position: 'relative'
        }}>
            {/* Video Background - positioned behind 3D canvas */}
            {videoSrc && (
                <video
                    ref={videoRef}
                    src={videoSrc}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        zIndex: 0,
                        opacity: 0.3,
                        pointerEvents: 'none'
                    }}
                    muted
                    playsInline
                />
            )}

            <Canvas
                camera={{ position: [0, 2.5, 6], fov: 50 }}
                shadows
                gl={{
                    antialias: true,
                    toneMapping: THREE.ACESFilmicToneMapping,
                    toneMappingExposure: 1.2,
                    alpha: true
                }}
                style={{
                    position: 'relative',
                    zIndex: 1
                }}
            >
                {/* Lighting Setup */}
                <ambientLight intensity={0.4} />

                {/* Key Light - Warm */}
                <directionalLight
                    position={[5, 8, 5]}
                    intensity={1.5}
                    castShadow
                    shadow-mapSize-width={2048}
                    shadow-mapSize-height={2048}
                    shadow-camera-far={50}
                    shadow-camera-left={-10}
                    shadow-camera-right={10}
                    shadow-camera-top={10}
                    shadow-camera-bottom={-10}
                    color="#FFE6F0"
                />

                {/* Fill Light - Cool */}
                <directionalLight
                    position={[-5, 5, -5]}
                    intensity={0.8}
                    color="#E6F0FF"
                />

                {/* Rim Light - Cyan accent */}
                <SpotLight
                    position={[0, 5, -8]}
                    angle={0.6}
                    penumbra={0.5}
                    intensity={0.5}
                    color="#5EEEFF"
                    castShadow
                />

                {/* Environment */}
                <Sky
                    distance={450000}
                    sunPosition={[5, 1, 8]}
                    inclination={0.6}
                    azimuth={0.25}
                />
                <Environment preset="night" />

                {/* Ground Plane */}
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
                    <planeGeometry args={[50, 50]} />
                    <meshStandardMaterial
                        color="#1A1A3E"
                        roughness={0.8}
                        metalness={0.1}
                        envMapIntensity={0.5}
                    />
                </mesh>

                {/* Ground with Contact Shadows */}
                <ContactShadows
                    opacity={0.5}
                    scale={30}
                    blur={2}
                    far={10}
                    position={[0, 0.01, 0]}
                    color="#0F0F1E"
                />

                {/* Grid with candy accent */}
                <Grid
                    infiniteGrid
                    sectionColor="#6EC3F5"
                    cellColor="#C96EF5"
                    fadeDistance={40}
                    position={[0, 0.02, 0]}
                    cellSize={1}
                    sectionSize={5}
                    args={[30, 30]}
                />

                {/* Distance Limit Visual Guide */}
                <DistanceRing radius={distanceLimit} />

                {/* Camera Controls */}
                <OrbitControls
                    makeDefault
                    enableDamping
                    dampingFactor={0.05}
                    minDistance={2}
                    maxDistance={20}
                    maxPolarAngle={Math.PI / 2 - 0.1}
                />

                {/* Render Persons with Distance Filtering */}
                {Object.entries(persons).map(([id, person]) => {
                    // Calculate distance from origin
                    let dist = 0;
                    if (person.cam_t) {
                        dist = Math.sqrt(
                            person.cam_t[0]**2 +
                            person.cam_t[1]**2 +
                            person.cam_t[2]**2
                        );
                    }

                    if (dist > distanceLimit) return null;

                    // Get candy color for this person
                    const colorIndex = parseInt(id) % CANDY_COLORS.length;
                    const candyColor = CANDY_COLORS[colorIndex];

                    return (
                        <PersonMesh
                            key={id}
                            data={person}
                            color={candyColor}
                            groundHeight={groundHeight}
                            faces={faces}
                            renderMode={renderMode}
                        />
                    );
                })}
            </Canvas>

            {/* Frame Counter Overlay - Glassmorphism */}
            <div style={{
                position: 'absolute',
                top: '24px',
                left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(20, 20, 40, 0.6)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                padding: '12px 28px',
                borderRadius: '20px',
                fontFamily: '"Fira Code", monospace',
                fontSize: '1.1rem',
                fontWeight: 600,
                pointerEvents: 'none',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                color: '#FFFFFF',
                letterSpacing: '0.05em'
            }}>
                <span style={{ color: 'var(--candy-cyan)' }}>Frame:</span> {currentFrame}
            </div>

            {/* Stats Overlay */}
            <div style={{
                position: 'absolute',
                top: '24px',
                right: '24px',
                background: 'rgba(20, 20, 40, 0.6)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                padding: '14px 20px',
                borderRadius: '16px',
                fontSize: '0.9rem',
                pointerEvents: 'none',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                minWidth: '140px'
            }}>
                <div style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: 'var(--candy-mint)' }}>●</span>
                    <span style={{ color: 'var(--text-secondary)' }}>Persons:</span>
                    <span style={{ fontWeight: 600, marginLeft: 'auto' }}>{Object.keys(persons).length}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: 'var(--candy-purple)' }}>●</span>
                    <span style={{ color: 'var(--text-secondary)' }}>Mode:</span>
                    <span style={{ fontWeight: 600, marginLeft: 'auto', textTransform: 'capitalize' }}>{renderMode}</span>
                </div>
            </div>

            {/* Frame Image Overlay - Glassmorphism */}
            {currentImage ? (
                <div style={{
                    position: 'absolute',
                    bottom: '24px',
                    right: '24px',
                    width: '340px',
                    background: 'rgba(20, 20, 40, 0.6)',
                    backdropFilter: 'blur(20px)',
                    WebkitBackdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '16px',
                    overflow: 'hidden',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                    zIndex: 100
                }}>
                    <div style={{
                        padding: '10px 14px',
                        background: 'rgba(0, 0, 0, 0.3)',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                        fontSize: '0.85rem',
                        fontWeight: 600,
                        color: 'var(--text-secondary)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}>
                        <span style={{ color: 'var(--candy-yellow)' }}>📷</span>
                        Source Frame
                    </div>
                    <img
                        src={`data:image/jpeg;base64,${currentImage}`}
                        alt="Current Frame"
                        style={{
                            width: '100%',
                            display: 'block',
                            borderBottomLeftRadius: '16px',
                            borderBottomRightRadius: '16px'
                        }}
                    />
                </div>
            ) : null}
        </div>
    );
};
