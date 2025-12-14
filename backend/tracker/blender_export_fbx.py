"""
Blender script to export SAM3D Body skeleton animation to FBX file.

SIMPLIFIED VERSION:
- Uses raw model output data directly
- Only converts Y-up to Z-up for Blender
- No complex coordinate transforms

The model outputs:
- pred_joint_coords: 127 joint positions (Y-up, but Y and Z are negated in mhr_head)
- pred_global_rots: 127 global rotation matrices (Y-up, NOT transformed)

For Blender (Z-up):
- Positions: (x, y, z) -> (x, z, -y) to swap Y/Z axes
- Rotations: Apply same axis swap transform

Usage: blender --background --python blender_export_fbx.py -- <input_obj> <output_fbx> [skeleton_json]
"""

import bpy
import sys
import os
import json
import numpy as np
from mathutils import Vector, Matrix, Quaternion

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_export_fbx.py -- <input_obj> <output_fbx> [skeleton_json]")
    sys.exit(1)

input_obj = argv[0]
output_fbx = argv[1]
skeleton_json = argv[2] if len(argv) > 2 else None

print(f"[SAM3D FBX Export] Input OBJ: {input_obj}")
print(f"[SAM3D FBX Export] Output FBX: {output_fbx}")
print(f"[SAM3D FBX Export] Skeleton JSON: {skeleton_json if skeleton_json else 'None'}")

# Load skeleton data from JSON
skeleton_data = {}
joints = None
num_joints = 0
joint_parents_list = None

if skeleton_json and os.path.exists(skeleton_json):
    try:
        with open(skeleton_json, 'r') as f:
            skeleton_data = json.load(f)

        joint_positions = skeleton_data.get('joint_positions', [])
        num_joints = skeleton_data.get('num_joints', len(joint_positions))
        joint_parents_list = skeleton_data.get('joint_parents')

        if joint_positions:
            joints = np.array(joint_positions, dtype=np.float32)
            print(f"[SAM3D FBX Export] Loaded skeleton with {num_joints} joints")
    except Exception as e:
        print(f"[SAM3D FBX Export] Warning: Failed to load skeleton JSON: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Coordinate System Conversion
# ============================================================================
# IMPORTANT: The model's mhr_head.py applies to positions:
#   jcoords[..., [1, 2]] *= -1  i.e., (x, y, z) -> (x, -y, -z)
# But rotations are NOT transformed!
#
# So positions come as (x, -y, -z) and rotations are in original Y-up space.
# 
# Strategy: 
# 1. First UNDO the position transform: (x, -y, -z) -> (x, y, z) by multiplying Y,Z by -1
# 2. Then convert both positions and rotations from Y-up to Z-up

def undo_position_flip(pos):
    """Undo the Y,Z negation that mhr_head applies to positions."""
    return [pos[0], -pos[1], -pos[2]]

def y_up_to_z_up_pos(pos):
    """Convert position from Y-up to Z-up (Blender)."""
    # First undo the flip, then convert
    p = undo_position_flip(pos)
    return Vector((p[0], -p[2], p[1]))

def y_up_to_z_up_rot(rot_mat):
    """
    Convert rotation matrix from Y-up to Z-up.
    The permutation matrix for (x, y, z) -> (x, z, -y) is:
    P = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    R_new = P @ R @ P.T
    """
    P = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=np.float32)
    
    rot_np = np.array(rot_mat, dtype=np.float32).reshape(3, 3)
    rot_new = P @ rot_np @ P.T
    
    return Matrix(rot_new.tolist())

# Clean default scene
def clean_bpy():
    """Remove all default Blender objects"""
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)

clean_bpy()

# Create collection
collection = bpy.data.collections.new('SAM3D_Export')
bpy.context.scene.collection.children.link(collection)

# Import OBJ mesh (optional)
mesh_obj = None
if input_obj != "NONE":
    print("[SAM3D FBX Export] Importing OBJ mesh...")
    try:
        bpy.ops.wm.obj_import(filepath=input_obj)
        imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        if imported_objects:
            mesh_obj = imported_objects[0]
            mesh_obj.name = 'SAM3D_Character'
            if mesh_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(mesh_obj)
            collection.objects.link(mesh_obj)
            print(f"[SAM3D FBX Export] Imported mesh: {len(mesh_obj.data.vertices)} vertices")
    except Exception as e:
        print(f"[SAM3D FBX Export] Failed to import OBJ: {e}")
else:
    print("[SAM3D FBX Export] Skipping OBJ import")

# Create armature from skeleton
armature_obj = None
if joints is not None and num_joints > 0:
    print(f"[SAM3D FBX Export] Creating armature with {num_joints} joints...")
    try:
        # Convert joint positions to Blender space
        joints_blender = np.array([y_up_to_z_up_pos(j) for j in joints])
        
        # Create armature
        bpy.ops.object.armature_add(enter_editmode=True)
        armature = bpy.data.armatures.get('Armature')
        armature.name = 'SAM3D_Skeleton'
        armature_obj = bpy.context.active_object
        armature_obj.name = 'SAM3D_Skeleton'

        if armature_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(armature_obj)
        collection.objects.link(armature_obj)

        edit_bones = armature.edit_bones
        extrude_size = 0.02

        # Remove default bone
        default_bone = edit_bones.get('Bone')
        if default_bone:
            edit_bones.remove(default_bone)

        # Build children map
        children_map = {i: [] for i in range(num_joints)}
        if joint_parents_list:
            for i, p in enumerate(joint_parents_list):
                if p >= 0 and p < num_joints:
                    children_map[p].append(i)

        # Find root joint
        root_idx = 0
        if joint_parents_list:
            for i, p in enumerate(joint_parents_list):
                if p == -1:
                    root_idx = i
                    break
        print(f"[SAM3D FBX Export] Root joint: {root_idx}")

        # Create bones
        bones_dict = {}
        for i in range(num_joints):
            bone_name = f'Joint_{i:03d}'
            bone = edit_bones.new(bone_name)
            bone.head = Vector(joints_blender[i])
            bone.tail = Vector(joints_blender[i]) + Vector((0, 0, extrude_size))
            bones_dict[bone_name] = bone

        # Set parent relationships
        if joint_parents_list and len(joint_parents_list) == num_joints:
            for i in range(num_joints):
                parent_idx = joint_parents_list[i]
                if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                    bone_name = f'Joint_{i:03d}'
                    parent_bone_name = f'Joint_{parent_idx:03d}'
                    bones_dict[bone_name].parent = bones_dict[parent_bone_name]
                    bones_dict[bone_name].use_connect = False

        # Adjust bone tails to point toward children
        for i in range(num_joints):
            bone_name = f'Joint_{i:03d}'
            bone = bones_dict[bone_name]
            child_indices = children_map[i]
            
            if child_indices:
                avg_child = Vector((0, 0, 0))
                for ci in child_indices:
                    avg_child += Vector(joints_blender[ci])
                avg_child /= len(child_indices)
                
                if (avg_child - bone.head).length > 0.001:
                    bone.tail = avg_child
            else:
                if bone.parent:
                    direction = bone.head - bone.parent.head
                    if direction.length > 0.001:
                        bone.tail = bone.head + direction.normalized() * extrude_size

        print(f"[SAM3D FBX Export] Created {len(bones_dict)} bones")

        # Switch to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        armature_obj.location = Vector((0, 0, 0))

    except Exception as e:
        print(f"[SAM3D FBX Export] Armature creation failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Animation
# ============================================================================
rot_history = skeleton_data.get('rot_history')
root_pos_history = skeleton_data.get('root_pos_history')
fps = skeleton_data.get('fps', 30.0)

if rot_history and root_pos_history and armature_obj:
    num_frames = len(rot_history)
    print(f"[SAM3D FBX Export] Processing {num_frames} frames of animation...")
    
    scene = bpy.context.scene
    scene.render.fps = int(fps)
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    
    # Switch to Pose Mode
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    
    pose_bones = {bone.name: bone for bone in armature_obj.pose.bones}
    
    # Get rest pose positions in Blender space
    rest_positions_blender = [y_up_to_z_up_pos(p) for p in skeleton_data['joint_positions']]
    
    for frame_idx, (frame_rots, root_pos) in enumerate(zip(rot_history, root_pos_history)):
        scene.frame_set(frame_idx)
        
        # Convert all rotations to Blender space
        blender_rots = [y_up_to_z_up_rot(rot_mat) for rot_mat in frame_rots]
        
        # Apply rotations to bones
        for j in range(num_joints):
            bone_name = f'Joint_{j:03d}'
            if bone_name not in pose_bones:
                continue
            
            pose_bone = pose_bones[bone_name]
            R_global = blender_rots[j]
            
            # Compute local rotation: R_local = R_parent^-1 @ R_global
            if joint_parents_list and joint_parents_list[j] >= 0:
                p = joint_parents_list[j]
                R_parent = blender_rots[p]
                R_local = R_parent.inverted() @ R_global
            else:
                R_local = R_global
            
            # Set rotation
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = R_local.to_quaternion()
            pose_bone.keyframe_insert(data_path="rotation_quaternion", index=-1)
            
            # Set location for root bone only
            if j == root_idx:
                root_pos_blender = y_up_to_z_up_pos(root_pos)
                rest_pos = rest_positions_blender[root_idx]
                pose_bone.location = root_pos_blender - rest_pos
                pose_bone.keyframe_insert(data_path="location", index=-1)
    
    print(f"[SAM3D FBX Export] Animation baked for {num_frames} frames")

# Export to FBX
print("[SAM3D FBX Export] Exporting to FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    for obj in collection.objects:
        obj.select_set(True)

    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        use_selection=True,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
        axis_forward='-Z',
        axis_up='Y',
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
    )
    print(f"[SAM3D FBX Export] Successfully saved to: {output_fbx}")

except Exception as e:
    print(f"[SAM3D FBX Export] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[SAM3D FBX Export] Done!")
