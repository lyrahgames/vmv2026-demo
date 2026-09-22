//! glTF/FBX scene loading and animation-pose preparation.
//!
//! The renderer keeps the richer source representation on the CPU, but the
//! playback path uploads immutable skinned vertices and only a compact matrix
//! palette. Materials, textures, and lights are intentionally ignored.

use crate::{
  common::*,
  mesh::{Mesh, SkinTransform, SkinnedVertex, Vertex},
};
use gltf::{Gltf, animation};
use oxideav_fbx::{FbxDecoder, binary::FbxProperty};
use oxideav_mesh3d::{self, Mesh3DDecoder};
use std::collections::HashMap;
use std::path::Path;

/// Metadata exposed to Lua and JavaScript for one glTF animation.
#[derive(Clone, Debug)]
pub struct AnimationInfo {
  /// The glTF name, or a stable generated name when the file omitted one.
  pub name:     String,
  /// Duration in seconds, calculated from all channel input samplers.
  pub duration: f32,
  /// Every channel target in the animation, including channels not used by
  /// the current scene's visible mesh.
  pub channels: Vec<AnimationChannelInfo>,
}

/// Metadata for one animation channel.
#[derive(Clone, Debug)]
pub struct AnimationChannelInfo {
  /// Index of the node targeted by the channel.
  pub node:          usize,
  /// glTF property name (`translation`, `rotation`, `scale`, or `weights`).
  pub property:      String,
  /// Sampler interpolation mode used by this channel.
  pub interpolation: String,
}

/// A complete parsed glTF scene retained by the viewer for playback.
#[derive(Clone)]
pub struct AnimatedScene {
  nodes:      Vec<SceneNode>,
  primitives: Vec<ScenePrimitive>,
  animations: Vec<AnimationClip>,
}

/// Immutable GPU geometry plus the pose palette layout for one animated
/// scene.  The geometry is uploaded once; only `transforms` changes during
/// playback.
#[derive(Clone)]
pub struct AnimatedGpuMesh {
  pub vertices:        Vec<SkinnedVertex>,
  pub indices:         Vec<u32>,
  pub transforms:      Vec<SkinTransform>,
  /// Morph deltas are static geometry; their animated weights are uploaded
  /// beside the transform palette.
  pub morph_positions: Vec<[f32; 4]>,
  pub morph_weights:   Vec<f32>,
}

impl AnimatedScene {
  /// Loads a native `.gltf`, `.glb`, or `.fbx` file.
  ///
  /// FBX is decoded directly from the file bytes. glTF keeps its existing
  /// path-aware loader so external `.bin` buffers remain relative to the
  /// `.gltf` file, while images are ignored because this viewer does not use
  /// materials or textures.
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_file(path: &str) -> Result<Self> {
    let path_ref = Path::new(path);
    let bytes = std::fs::read(path_ref).with_context(|| format!("reading scene {path}"))?;
    if is_fbx_bytes(&bytes) {
      return Self::from_fbx_bytes(&bytes);
    }
    let asset = Gltf::open(path_ref).with_context(|| format!("reading glTF {path}"))?;
    let base = path_ref.parent();
    let buffers = load_buffers(&asset.document, base, asset.blob)?;
    Self::from_document(asset.document, buffers)
  }

  /// Loads browser-provided glTF/GLB or FBX bytes.
  ///
  /// The JavaScript adapter embeds external glTF buffers before calling this
  /// method. FBX is self-contained in the supplied file, so it crosses the
  /// WASM boundary as-is and needs no companion-resource handling.
  pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
    if is_fbx_bytes(bytes) {
      return Self::from_fbx_bytes(bytes);
    }
    let asset = Gltf::from_slice(bytes).context("parsing glTF bytes")?;
    let buffers =
      load_buffers(&asset.document, None, asset.blob).context("loading glTF buffer data")?;
    Self::from_document(asset.document, buffers)
  }

  /// Converts the format-neutral FBX scene into the small internal scene
  /// representation used by the renderer. Keeping this conversion here is
  /// important: FBX and glTF then use exactly the same CPU animation sampler
  /// and exactly the same skinning/morphing path during rendering.
  fn from_fbx_bytes(bytes: &[u8]) -> Result<Self> {
    let mut decoder = FbxDecoder::new();
    let mut source = decoder
      .decode(bytes)
      .map_err(|error| anyhow!(error.to_string()))
      .context("parsing FBX scene")?;
    // The decoder handles the usual FBX connection orientation.  Some
    // exporters, including the supplied Mixamo kick asset, write the
    // Model↔Cluster edge backwards.  Repairing only that missing typed
    // skin data here keeps the rest of the application format-neutral.
    let fbx_geometry_ids = decoder
      .last_document
      .as_ref()
      .map(fbx_geometry_ids)
      .unwrap_or_default();
    let fbx_corner_indices = decoder
      .last_document
      .as_ref()
      .map(fbx_geometry_corner_indices)
      .unwrap_or_default();
    if let Some(document) = decoder.last_document.as_ref() {
      repair_reversed_fbx_skin_bindings(&mut source, document);
    }

    let mut parents = vec![None; source.nodes.len()];
    for (parent, node) in source.nodes.iter().enumerate() {
      for child in &node.children {
        let child = child.0 as usize;
        if child < parents.len() {
          parents[child] = Some(parent);
        }
      }
    }

    let nodes = source
      .nodes
      .iter()
      .enumerate()
      .map(|(node_index, node)| {
        let (translation, rotation, scale) = fbx_transform(node.transform);
        let weights = if node.weights.is_empty() {
          node
            .mesh
            .and_then(|mesh| source.meshes.get(mesh.0 as usize))
            .map(|mesh| mesh.weights.clone())
            .unwrap_or_default()
        } else {
          node.weights.clone()
        };
        SceneNode {
          parent: parents[node_index],
          translation: Vec3::from_array(translation),
          rotation: Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
          scale: Vec3::from_array(scale),
          weights,
        }
      })
      .collect::<Vec<_>>();

    let mut primitives = Vec::new();
    for (node_index, node) in source.nodes.iter().enumerate() {
      let Some(mesh_id) = node.mesh else { continue };
      let Some(mesh) = source.meshes.get(mesh_id.0 as usize) else {
        continue;
      };
      for primitive in &mesh.primitives {
        if primitive.positions.is_empty() {
          continue;
        }
        let triangles = primitive.triangle_indices();
        if triangles.is_empty() {
          bail!("FBX mesh on node {node_index} has no triangle surface")
        }

        // FBX's decoder exposes render corners because it must be able
        // to represent per-corner attributes.  The viewer deliberately
        // does not use that representation: positions are collapsed
        // back to the file's shared PolygonVertexIndex space, while
        // authored normals and skin weights are combined onto the
        // shared vertices.  A normal seam therefore becomes one
        // averaged vertex normal instead of another duplicated vertex.
        let corner_indices = fbx_geometry_ids
          .get(mesh_id.0 as usize)
          .and_then(|geometry_id| fbx_corner_indices.get(geometry_id));
        let collapsed = corner_indices
          .map(|corner_indices| collapse_fbx_primitive(primitive, &triangles, corner_indices));
        let (positions, normals, indices, morph_positions, joints, weights) = collapsed
          .unwrap_or_else(|| {
            (
              primitive
                .positions
                .iter()
                .copied()
                .map(Vec3::from_array)
                .collect(),
              primitive
                .normals
                .as_ref()
                .map(|values| values.iter().copied().map(Vec3::from_array).collect()),
              triangles
                .iter()
                .flat_map(|triangle| triangle.iter().copied())
                .collect(),
              primitive
                .targets
                .iter()
                .map(|target| {
                  target
                    .position
                    .as_ref()
                    .map(|values| values.iter().copied().map(Vec3::from_array).collect())
                    .unwrap_or_else(|| vec![Vec3::ZERO; primitive.positions.len()])
                })
                .collect(),
              primitive.joints.clone().unwrap_or_default(),
              primitive.weights.clone().unwrap_or_default(),
            )
          });
        let skin = node.skin.and_then(|skin_id| {
          let skin = source.skins.get(skin_id.0 as usize)?;
          let skeleton = source.skeletons.get(skin.skeleton.0 as usize)?;
          Some(SceneSkin {
            joints:       skeleton
              .joints
              .iter()
              .map(|joint| joint.0 as usize)
              .collect(),
            inverse_bind: skeleton
              .inverse_bind_matrices
              .iter()
              .map(|matrix| Mat4::from_cols_array_2d(matrix))
              .collect(),
          })
        });
        primitives.push(ScenePrimitive {
          node: node_index,
          positions,
          normals,
          indices,
          joints,
          weights,
          morph_positions,
          skin,
        });
      }
    }
    if primitives.is_empty() {
      bail!("FBX scene contains no triangle meshes")
    }

    let animations = source
      .animations
      .iter()
      .enumerate()
      .map(|(index, animation)| fbx_animation(animation, index, nodes.len()))
      .collect::<Result<Vec<_>>>()?;

    Ok(Self {
      nodes,
      primitives,
      animations,
    })
  }

  /// Returns immutable animation metadata without exposing parser objects.
  pub fn animations(&self) -> Vec<AnimationInfo> {
    self.animations.iter().map(AnimationClip::info).collect()
  }

  /// Produces the base frame or the requested animation frame.
  pub fn sample(&self, animation: Option<usize>, time: f32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    self.sample_into_buffers(animation, time, &mut vertices, Some(&mut indices));
    Mesh::from_parts(vertices, indices).expect("validated scene produced an empty frame")
  }

  /// Samples one frame for initial camera bounds and compatibility callers.
  ///
  /// Playback keeps the topology fixed, so rebuilding the index vector and
  /// allocating a second temporary `Mesh` for every frame is unnecessary.
  /// The viewer's live playback path uses [`Self::update_gpu_pose`] instead;
  /// this method remains useful for one-time CPU-side inspection and tests.
  pub fn sample_into(&self, animation: Option<usize>, time: f32, mesh: &mut Mesh) {
    mesh.vertices.clear();
    self.sample_into_buffers(animation, time, &mut mesh.vertices, None);
    // Keep the same missing-normal behavior as `sample`, without paying
    // for a temporary Mesh allocation on every animated frame.
    Mesh::fill_missing_normals(&mut mesh.vertices, &mesh.indices);
  }

  /// Builds the immutable GPU geometry and its rest-pose transform palette.
  ///
  /// The mesh-local attributes are never rewritten during playback.  The
  /// returned palette is laid out as one node transform followed by that
  /// primitive's joint transforms, repeated for each primitive.  Vertices
  /// store the relevant palette indices, allowing one draw call for the
  /// complete scene without a CPU vertex-skinning pass.
  pub fn gpu_mesh(&self) -> AnimatedGpuMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut transforms = Vec::new();
    let mut morph_positions = Vec::new();
    let mut morph_weights = Vec::new();
    self.build_gpu_mesh(
      &mut vertices,
      &mut indices,
      &mut transforms,
      &mut morph_positions,
      &mut morph_weights,
    );
    AnimatedGpuMesh {
      vertices,
      indices,
      transforms,
      morph_positions,
      morph_weights,
    }
  }

  /// Recomputes only the small transform palette for one animation frame.
  /// Position and normal attributes stay on the GPU and are evaluated by
  /// the vertex shader using these matrices and the per-vertex weights.
  pub fn update_gpu_pose(
    &self,
    animation: Option<usize>,
    time: f32,
    transforms: &mut Vec<SkinTransform>,
    morph_weights: &mut Vec<f32>,
  ) {
    let (locals, weights) = self.sampled_locals(animation, time);
    let mut world = vec![None; self.nodes.len()];
    transforms.clear();
    morph_weights.clear();
    for primitive in &self.primitives {
      let node_world = world_matrix(primitive.node, &self.nodes, &locals, &mut world);
      transforms.push(gpu_transform(node_world));
      morph_weights.extend(weights.get(primitive.node).into_iter().flat_map(|weights| {
        weights
          .iter()
          .copied()
          .take(primitive.morph_positions.len())
      }));
      if let Some(node_weights) = weights.get(primitive.node) {
        let missing = primitive
          .morph_positions
          .len()
          .saturating_sub(node_weights.len());
        morph_weights.extend(std::iter::repeat_n(0.0, missing));
      } else {
        morph_weights.extend(std::iter::repeat_n(0.0, primitive.morph_positions.len()));
      }
      if let Some(skin) = &primitive.skin {
        for (joint, &joint_node) in skin.joints.iter().enumerate() {
          let inverse_bind = skin
            .inverse_bind
            .get(joint)
            .copied()
            .unwrap_or(Mat4::IDENTITY);
          let matrix = world_matrix(joint_node, &self.nodes, &locals, &mut world) * inverse_bind;
          transforms.push(gpu_transform(matrix));
        }
      }
    }
    // Keep the optional morph-weight upload non-empty for WebGPU even
    // when the current scene has no morph targets at all.
    if morph_weights.is_empty() {
      morph_weights.push(0.0);
    }
  }

  /// Creates static GPU vertices and records the same palette layout used by
  /// `update_gpu_pose`. Morph deltas are static storage data, while their
  /// weights are part of the small per-frame pose upload.
  fn build_gpu_mesh(
    &self,
    vertices: &mut Vec<SkinnedVertex>,
    indices: &mut Vec<u32>,
    transforms: &mut Vec<SkinTransform>,
    morph_positions: &mut Vec<[f32; 4]>,
    morph_weights: &mut Vec<f32>,
  ) {
    let (locals, weights) = self.sampled_locals(None, 0.0);
    let mut world = vec![None; self.nodes.len()];
    for primitive in &self.primitives {
      let base_transform = transforms.len() as u32;
      let node_world = world_matrix(primitive.node, &self.nodes, &locals, &mut world);
      transforms.push(gpu_transform(node_world));
      let joint_base = transforms.len() as u32;
      if let Some(skin) = &primitive.skin {
        for (joint, &joint_node) in skin.joints.iter().enumerate() {
          let inverse_bind = skin
            .inverse_bind
            .get(joint)
            .copied()
            .unwrap_or(Mat4::IDENTITY);
          let matrix = world_matrix(joint_node, &self.nodes, &locals, &mut world) * inverse_bind;
          transforms.push(gpu_transform(matrix));
        }
      }
      let morph_weight_base = morph_weights.len() as u32;
      morph_weights.extend(weights.get(primitive.node).into_iter().flat_map(|weights| {
        weights
          .iter()
          .copied()
          .take(primitive.morph_positions.len())
      }));
      if let Some(node_weights) = weights.get(primitive.node) {
        let missing = primitive
          .morph_positions
          .len()
          .saturating_sub(node_weights.len());
        morph_weights.extend(std::iter::repeat_n(0.0, missing));
      } else {
        morph_weights.extend(std::iter::repeat_n(0.0, primitive.morph_positions.len()));
      }
      let morph_count = primitive.morph_positions.len() as u32;
      let morph_base = morph_positions.len() as u32;
      // Interleave target deltas per vertex, making each shader lookup
      // contiguous and avoiding a second vertex-index attribute.
      for vertex_index in 0..primitive.positions.len() {
        for target in &primitive.morph_positions {
          morph_positions.push(
            target
              .get(vertex_index)
              .copied()
              .unwrap_or(Vec3::ZERO)
              .extend(0.0)
              .to_array(),
          );
        }
      }
      let base_vertex = vertices.len() as u32;
      // GPU vertices keep normals in local space. Reuse the same
      // winding-aware fallback as the static mesh path when a source
      // primitive omitted normals; authored normals remain untouched.
      let mut local_vertices: Vec<_> = primitive
        .positions
        .iter()
        .enumerate()
        .map(|(index, &position)| Vertex {
          position: position.to_array(),
          normal:   primitive
            .normals
            .as_ref()
            .and_then(|normals| normals.get(index))
            .copied()
            .unwrap_or(Vec3::ZERO)
            .to_array(),
        })
        .collect();
      Mesh::fill_missing_normals(&mut local_vertices, &primitive.indices);
      for (index, &position) in primitive.positions.iter().enumerate() {
        let joints = primitive.joints.get(index).copied().unwrap_or([0; 4]);
        let weights = primitive.weights.get(index).copied().unwrap_or([0.0; 4]);
        vertices.push(SkinnedVertex {
          position: position.to_array(),
          normal: local_vertices[index].normal,
          base_transform,
          joints: joints.map(|joint| joint_base + joint as u32),
          weights,
          morph_base: morph_base + index as u32 * morph_count,
          morph_count,
          morph_weight_base,
        });
      }
      indices.extend(primitive.indices.iter().map(|index| base_vertex + *index));
    }
  }

  /// Applies animation channels and returns local node matrices. The second
  /// return value is retained for the CPU sampler's morph-weight handling.
  fn sampled_locals(&self, animation: Option<usize>, time: f32) -> (Vec<Mat4>, Vec<Vec<f32>>) {
    let mut translations: Vec<Vec3> = self.nodes.iter().map(|n| n.translation).collect();
    let mut rotations: Vec<Quat> = self.nodes.iter().map(|n| n.rotation).collect();
    let mut scales: Vec<Vec3> = self.nodes.iter().map(|n| n.scale).collect();
    let mut weights: Vec<Vec<f32>> = self.nodes.iter().map(|n| n.weights.clone()).collect();
    if let Some(clip) = animation.and_then(|index| self.animations.get(index)) {
      let time = wrap_time(time, clip.info.duration);
      for channel in &clip.channels {
        match &channel.data {
          ChannelData::Translation(keys) => {
            if let Some(value) = sample_vec3(keys, channel.interpolation, time) {
              translations[channel.node] = value;
            }
          }
          ChannelData::Rotation(keys) => {
            if let Some(value) = sample_quat(keys, channel.interpolation, time) {
              rotations[channel.node] = value;
            }
          }
          ChannelData::Scale(keys) => {
            if let Some(value) = sample_vec3(keys, channel.interpolation, time) {
              scales[channel.node] = value;
            }
          }
          ChannelData::Weights(keys) => {
            if let Some(value) = sample_weights(keys, channel.interpolation, time) {
              weights[channel.node] = value;
            }
          }
        }
      }
    }
    let locals = self
      .nodes
      .iter()
      .enumerate()
      .map(|(index, _)| {
        Mat4::from_scale_rotation_translation(scales[index], rotations[index], translations[index])
      })
      .collect();
    (locals, weights)
  }

  /// Shared frame sampler used by both one-time mesh creation and the
  /// allocation-free playback path. `indices` is populated only when a new
  /// mesh is being constructed; an existing animated mesh already owns the
  /// immutable topology buffer.
  fn sample_into_buffers(
    &self,
    animation: Option<usize>,
    time: f32,
    vertices: &mut Vec<Vertex>,
    mut indices: Option<&mut Vec<u32>>,
  ) {
    let (locals, weights) = self.sampled_locals(animation, time);
    // Compute animated local transforms first. The recursive world-matrix
    // helper below then composes each node with its animated parents.
    let mut world = vec![None; self.nodes.len()];

    for primitive in &self.primitives {
      let node_world = world_matrix(primitive.node, &self.nodes, &locals, &mut world);
      // These matrices depend only on the current animated pose, not on
      // an individual vertex. Compute them once per joint per frame;
      // inverting a 4x4 matrix inside the vertex loop was the dominant
      // cost of animated FBX playback.
      let node_normal_matrix = node_world.inverse().transpose();
      let joint_matrices = primitive.skin.as_ref().map(|skin| {
        skin
          .joints
          .iter()
          .enumerate()
          .map(|(joint, &joint_node)| {
            let inverse_bind = skin.inverse_bind.get(joint).copied()?;
            let joint_world = world_matrix(joint_node, &self.nodes, &locals, &mut world);
            let skin_matrix = joint_world * inverse_bind;
            Some((skin_matrix, skin_matrix.inverse().transpose()))
          })
          .collect::<Vec<_>>()
      });
      let base = vertices.len() as u32;
      let node_weights = weights
        .get(primitive.node)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

      for vertex_index in 0..primitive.positions.len() {
        let mut position = primitive.positions[vertex_index];
        for (target_index, target) in primitive.morph_positions.iter().enumerate() {
          if let Some(delta) = target.get(vertex_index) {
            position += *delta * node_weights.get(target_index).copied().unwrap_or(0.0);
          }
        }

        let position = if primitive.skin.is_some() {
          let joints = primitive
            .joints
            .get(vertex_index)
            .copied()
            .unwrap_or([0; 4]);
          let influences = primitive
            .weights
            .get(vertex_index)
            .copied()
            .unwrap_or([0.0; 4]);
          let mut skinned = Vec3::ZERO;
          let mut total_weight = 0.0;
          for influence in 0..4 {
            let joint = joints[influence] as usize;
            if let Some(Some((skin_matrix, _))) = joint_matrices
              .as_ref()
              .and_then(|matrices| matrices.get(joint))
            {
              let weight = influences[influence];
              skinned += skin_matrix.transform_point3(position) * weight;
              total_weight += weight;
            }
          }
          if total_weight > f32::EPSILON {
            skinned
          } else {
            node_world.transform_point3(position)
          }
        } else {
          node_world.transform_point3(position)
        };
        let normal = primitive
          .normals
          .as_ref()
          .and_then(|normals| normals.get(vertex_index).copied());
        let normal = if let Some(normal) = normal {
          if primitive.skin.is_some() {
            let joints = primitive
              .joints
              .get(vertex_index)
              .copied()
              .unwrap_or([0; 4]);
            let influences = primitive
              .weights
              .get(vertex_index)
              .copied()
              .unwrap_or([0.0; 4]);
            let mut skinned = Vec3::ZERO;
            let mut total_weight = 0.0;
            for influence in 0..4 {
              let joint = joints[influence] as usize;
              if let Some(Some((_, normal_matrix))) = joint_matrices
                .as_ref()
                .and_then(|matrices| matrices.get(joint))
              {
                let weight = influences[influence];
                skinned += normal_matrix.transform_vector3(normal) * weight;
                total_weight += weight;
              }
            }
            if total_weight > f32::EPSILON && skinned.length_squared() > f32::EPSILON {
              skinned.normalize()
            } else {
              // A source normal still has meaning on an
              // unweighted vertex. Keep it instead of turning
              // it into a generated normal just because the
              // skin has no influence for this corner.
              node_normal_matrix
                .transform_vector3(normal)
                .normalize_or_zero()
            }
          } else {
            node_normal_matrix
              .transform_vector3(normal)
              .normalize_or_zero()
          }
        } else {
          Vec3::ZERO
        };
        vertices.push(Vertex {
          position: position.to_array(),
          // Authored normals are retained and transformed into world
          // space. A zero value means the source omitted normals and
          // asks Mesh::from_parts for its documented fallback.
          normal:   normal.to_array(),
        });
      }
      if let Some(indices) = indices.as_deref_mut() {
        indices.extend(primitive.indices.iter().map(|index| base + *index));
      }
    }
  }

  /// Returns the initial static frame used while the GPU viewer is created.
  pub fn initial_mesh(&self) -> Mesh {
    self.sample(None, 0.0)
  }
}

/// FBX files may store either decomposed TRS or a matrix. The renderer's
/// animation representation is TRS-based, so matrices are decomposed once at
/// load time rather than on every rendered frame.
fn fbx_transform(transform: oxideav_mesh3d::Transform) -> ([f32; 3], [f32; 4], [f32; 3]) {
  match transform {
    oxideav_mesh3d::Transform::Trs {
      translation,
      rotation,
      scale,
    } => (translation, rotation, scale),
    oxideav_mesh3d::Transform::Matrix(matrix) => {
      fbx_transform(oxideav_mesh3d::Transform::from_matrix(matrix))
    }
  }
}

/// Converts one format-neutral FBX animation into the internal sampler format
/// used by both glTF and FBX. The FBX crate has already merged its X/Y/Z
/// curves and converted Euler rotation curves to quaternions; this function
/// only adapts the shared typed values and exposes channel metadata.
fn fbx_animation(
  animation: &oxideav_mesh3d::Animation,
  index: usize,
  node_count: usize,
) -> Result<AnimationClip> {
  let mut channels = Vec::new();
  let mut info_channels = Vec::new();
  let mut duration = 0.0_f32;

  for channel in &animation.channels {
    let node = channel.target.node.0 as usize;
    if node >= node_count {
      bail!("FBX animation channel targets missing node {node}")
    }
    let interpolation = fbx_interpolation(channel.sampler.interpolation);
    duration = duration.max(
      channel
        .sampler
        .keyframes
        .iter()
        .copied()
        .fold(0.0, f32::max),
    );
    let data = match (channel.target.property, &channel.sampler.values) {
      (
        oxideav_mesh3d::AnimationProperty::Translation,
        oxideav_mesh3d::AnimationValues::Vec3(values),
      ) => ChannelData::Translation(make_vec3_keys(
        channel.sampler.keyframes.clone(),
        values.clone(),
        interpolation,
      )),
      (
        oxideav_mesh3d::AnimationProperty::Rotation,
        oxideav_mesh3d::AnimationValues::Quat(values),
      ) => ChannelData::Rotation(make_quat_keys(
        channel.sampler.keyframes.clone(),
        values.clone(),
        interpolation,
      )),
      (oxideav_mesh3d::AnimationProperty::Scale, oxideav_mesh3d::AnimationValues::Vec3(values)) => {
        ChannelData::Scale(make_vec3_keys(
          channel.sampler.keyframes.clone(),
          values.clone(),
          interpolation,
        ))
      }
      (
        oxideav_mesh3d::AnimationProperty::MorphWeights,
        oxideav_mesh3d::AnimationValues::Scalar(values),
      ) => ChannelData::Weights(make_fbx_weight_keys(
        &channel.sampler.keyframes,
        values,
        interpolation,
      )),
      (property, values) => bail!(
        "FBX animation property {:?} has incompatible values {:?}",
        property,
        values
      ),
    };
    info_channels.push(AnimationChannelInfo {
      node,
      property: fbx_property_name(channel.target.property).to_string(),
      interpolation: interpolation_name(interpolation).to_string(),
    });
    channels.push(AnimationChannel {
      node,
      interpolation,
      data,
    });
  }

  Ok(AnimationClip {
    info: AnimationInfo {
      name: animation
        .name
        .clone()
        .unwrap_or_else(|| format!("Animation {index}")),
      duration,
      channels: info_channels,
    },
    channels,
  })
}

fn fbx_interpolation(interpolation: oxideav_mesh3d::Interpolation) -> Interpolation {
  match interpolation {
    oxideav_mesh3d::Interpolation::Linear => Interpolation::Linear,
    oxideav_mesh3d::Interpolation::Step => Interpolation::Step,
    oxideav_mesh3d::Interpolation::CubicSpline => Interpolation::CubicSpline,
  }
}

fn fbx_property_name(property: oxideav_mesh3d::AnimationProperty) -> &'static str {
  match property {
    oxideav_mesh3d::AnimationProperty::Translation => "translation",
    oxideav_mesh3d::AnimationProperty::Rotation => "rotation",
    oxideav_mesh3d::AnimationProperty::Scale => "scale",
    oxideav_mesh3d::AnimationProperty::MorphWeights => "weights",
  }
}

/// FBX morph values are a flattened scalar stream. Convert that stream to the
/// internal per-keyframe vectors, including the in/value/out layout used by
/// cubic-spline samplers if a decoder ever supplies one.
fn make_fbx_weight_keys(
  times: &[f32],
  values: &[f32],
  interpolation: Interpolation,
) -> Vec<Key<Vec<f32>>> {
  let key_count = times.len();
  if key_count == 0 {
    return Vec::new();
  }
  let stride = match interpolation {
    Interpolation::CubicSpline => values.len() / (key_count * 3).max(1),
    _ => values.len() / key_count,
  };
  times
    .iter()
    .enumerate()
    .map(|(index, time)| {
      let value_index = match interpolation {
        Interpolation::CubicSpline => index * 3 + 1,
        _ => index,
      };
      let read = |frame: usize| {
        let start = frame * stride;
        values
          .get(start..start.saturating_add(stride))
          .unwrap_or(&[])
          .to_vec()
      };
      Key {
        time:        *time,
        value:       read(value_index),
        in_tangent:  (interpolation == Interpolation::CubicSpline).then(|| read(index * 3)),
        out_tangent: (interpolation == Interpolation::CubicSpline).then(|| read(index * 3 + 2)),
      }
    })
    .collect()
}

/// FBX has two encodings, both recognized by the decoder: the binary Kaydara
/// header and the ASCII `; FBX` banner. Detecting them before trying glTF gives
/// native and browser callers one format-independent `from_bytes` entry point.
fn is_fbx_bytes(bytes: &[u8]) -> bool {
  let first_non_whitespace = bytes
    .iter()
    .position(|byte| !byte.is_ascii_whitespace())
    .unwrap_or(bytes.len());
  oxideav_fbx::decoder::is_binary_fbx(bytes) || bytes[first_non_whitespace..].starts_with(b"; FBX")
}

/// Returns FBX geometry object IDs in the same order used by the mesh3d
/// decoder.  This lets the format adapter associate a decoded primitive with
/// its original PolygonVertexIndex stream without retaining FBX IDs in the
/// renderer's runtime structures.
fn fbx_geometry_ids(document: &oxideav_fbx::binary::FbxDocument) -> Vec<i64> {
  document
    .root
    .child("Objects")
    .map(|objects| {
      objects
        .children
        .iter()
        .filter(|object| {
          object.name == "Geometry"
            && object.properties.get(2).and_then(FbxProperty::as_str) == Some("Mesh")
        })
        .filter_map(|object| object.properties.first().and_then(FbxProperty::as_i64))
        .collect()
    })
    .unwrap_or_default()
}

/// Reads the triangulated shared-vertex mapping used by the FBX decoder.
/// `corner_indices[n]` is the source PolygonVertexIndex represented by the
/// decoder's nth render corner.
fn fbx_geometry_corner_indices(
  document: &oxideav_fbx::binary::FbxDocument,
) -> HashMap<i64, Vec<u32>> {
  let Some(objects) = document.root.child("Objects") else {
    return HashMap::new();
  };
  let mut result = HashMap::new();
  for geometry in objects.children.iter().filter(|object| {
    object.name == "Geometry"
      && object.properties.get(2).and_then(FbxProperty::as_str) == Some("Mesh")
  }) {
    let Some(id) = geometry.properties.first().and_then(FbxProperty::as_i64) else {
      continue;
    };
    let Some(values) = geometry
      .child("PolygonVertexIndex")
      .and_then(|node| node.properties.first())
      .and_then(fbx_i32_array)
    else {
      continue;
    };
    let mut corners = Vec::new();
    let mut polygon = Vec::new();
    for value in values {
      polygon.push(if *value < 0 {
        (-*value - 1) as u32
      } else {
        *value as u32
      });
      if *value < 0 {
        // The decoder triangulates an n-gon as a fan. Repeating the
        // exact rule keeps all attribute and skin streams aligned.
        for index in 1..polygon.len().saturating_sub(1) {
          corners.extend([polygon[0], polygon[index], polygon[index + 1]]);
        }
        polygon.clear();
      }
    }
    result.insert(id, corners);
  }
  result
}

/// Collapses decoder render corners into the FBX file's shared vertex space.
///
/// FBX stores normals and weights in mapping/reference streams that may be
/// per-corner, but that is not a reason to duplicate positions in this viewer:
/// there is no UV/material system here.  Conflicting authored normals are
/// averaged on the shared vertex, and repeated skin influences are combined
/// before retaining the strongest four.
fn collapse_fbx_primitive(
  primitive: &oxideav_mesh3d::Primitive,
  triangles: &[[u32; 3]],
  corner_indices: &[u32],
) -> (
  Vec<Vec3>,
  Option<Vec<Vec3>>,
  Vec<u32>,
  Vec<Vec<Vec3>>,
  Vec<[u16; 4]>,
  Vec<[f32; 4]>,
) {
  let shared_count = corner_indices
    .iter()
    .copied()
    .max()
    .map(|index| index as usize + 1)
    .unwrap_or(primitive.positions.len());
  let mut positions = vec![Vec3::ZERO; shared_count];
  let mut position_seen = vec![false; shared_count];
  for (corner, &shared) in corner_indices.iter().enumerate() {
    let shared = shared as usize;
    if let Some(position) = primitive.positions.get(corner) {
      if !position_seen[shared] {
        positions[shared] = Vec3::from_array(*position);
        position_seen[shared] = true;
      }
    }
  }
  let indices = triangles
    .iter()
    .flat_map(|triangle| {
      triangle.iter().map(|corner| {
        corner_indices
          .get(*corner as usize)
          .copied()
          .unwrap_or(*corner)
      })
    })
    .collect();

  let normals = primitive.normals.as_ref().map(|source| {
    let mut sums = vec![Vec3::ZERO; shared_count];
    let mut counts = vec![0u32; shared_count];
    for (corner, &shared) in corner_indices.iter().enumerate() {
      if let Some(normal) = source.get(corner) {
        sums[shared as usize] += Vec3::from_array(*normal);
        counts[shared as usize] += 1;
      }
    }
    sums
      .into_iter()
      .zip(counts)
      .map(|(sum, count)| {
        if count == 0 {
          Vec3::ZERO
        } else {
          (sum / count as f32).normalize_or_zero()
        }
      })
      .collect()
  });

  let morph_positions = primitive
    .targets
    .iter()
    .map(|target| {
      let mut sums = vec![Vec3::ZERO; shared_count];
      let mut counts = vec![0u32; shared_count];
      if let Some(source) = &target.position {
        for (corner, &shared) in corner_indices.iter().enumerate() {
          if let Some(value) = source.get(corner) {
            sums[shared as usize] += Vec3::from_array(*value);
            counts[shared as usize] += 1;
          }
        }
      }
      sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| {
          if count == 0 {
            Vec3::ZERO
          } else {
            sum / count as f32
          }
        })
        .collect()
    })
    .collect();

  let (joints, weights) = collapse_fbx_influences(primitive, corner_indices, shared_count);
  (
    positions,
    normals,
    indices,
    morph_positions,
    joints,
    weights,
  )
}

fn collapse_fbx_influences(
  primitive: &oxideav_mesh3d::Primitive,
  corner_indices: &[u32],
  shared_count: usize,
) -> (Vec<[u16; 4]>, Vec<[f32; 4]>) {
  let (Some(source_joints), Some(source_weights)) = (&primitive.joints, &primitive.weights) else {
    return (Vec::new(), Vec::new());
  };
  let mut accumulated = vec![HashMap::<u16, f32>::new(); shared_count];
  for (corner, &shared) in corner_indices.iter().enumerate() {
    let Some(joints) = source_joints.get(corner) else {
      continue;
    };
    let Some(weights) = source_weights.get(corner) else {
      continue;
    };
    for slot in 0..4 {
      *accumulated[shared as usize]
        .entry(joints[slot])
        .or_default() += weights[slot];
    }
  }
  let mut joints = vec![[0; 4]; shared_count];
  let mut weights = vec![[0.0; 4]; shared_count];
  for (index, influences) in accumulated.iter_mut().enumerate() {
    let mut values: Vec<_> = influences.drain().collect();
    values.sort_by(|left, right| right.1.total_cmp(&left.1));
    for (slot, &(joint, weight)) in values.iter().take(4).enumerate() {
      joints[index][slot] = joint;
      weights[index][slot] = weight;
    }
    let total: f32 = weights[index].iter().sum();
    if total > f32::EPSILON {
      for weight in &mut weights[index] {
        *weight /= total;
      }
    }
  }
  (joints, weights)
}

/// Repairs the one FBX skin layout that the current decoder cannot connect.
///
/// FBX's `Connections` table is nominally directed as `child -> parent`, but
/// a number of exporters emit the Model/Cluster relation as `Model -> Cluster`
/// instead of `Cluster -> Model`.  The decoder still exposes the complete
/// document through `last_document`, so we can safely recover the missing
/// skeleton from the already parsed public scene without reparsing geometry.
/// This function is deliberately a no-op when the decoder already produced a
/// skin, which keeps standard FBX files on the maintained decoder path.
fn repair_reversed_fbx_skin_bindings(
  scene: &mut oxideav_mesh3d::Scene3D,
  document: &oxideav_fbx::binary::FbxDocument,
) {
  if !scene.skins.is_empty() {
    return;
  }
  let Some(objects) = document.root.child("Objects") else {
    return;
  };

  let mut model_ids = Vec::new();
  let mut geometry_ids = Vec::new();
  let mut skins = HashMap::new();
  let mut clusters = HashMap::new();
  for object in &objects.children {
    let Some(id) = object.properties.first().and_then(FbxProperty::as_i64) else {
      continue;
    };
    match object.name.as_str() {
      "Model" => model_ids.push(id),
      "Geometry" if object.properties.get(2).and_then(FbxProperty::as_str) == Some("Mesh") => {
        geometry_ids.push(id);
      }
      "Deformer" => match object.properties.get(2).and_then(FbxProperty::as_str) {
        Some("Skin") => {
          skins.insert(id, object);
        }
        Some("Cluster") => {
          clusters.insert(id, object);
        }
        _ => {}
      },
      _ => {}
    }
  }
  if skins.is_empty() || clusters.is_empty() || model_ids.len() != scene.nodes.len() {
    return;
  }

  // Scene3D preserves the FBX Objects/Model order.  Keeping this mapping
  // local also avoids making FBX object IDs leak into the renderer model.
  let model_nodes: HashMap<_, _> = model_ids
    .into_iter()
    .zip((0..scene.nodes.len()).map(|index| oxideav_mesh3d::NodeId(index as u32)))
    .collect();
  let geometry_meshes: HashMap<_, _> = geometry_ids
    .into_iter()
    .zip((0..scene.meshes.len()).map(|index| oxideav_mesh3d::MeshId(index as u32)))
    .collect();

  // The reversed-connection exporter also stores bind matrices in a
  // convention that does not match the renderer's column-vector matrices.
  // Build rest-pose world transforms from the decoded node hierarchy once,
  // then derive inverse binds in precisely the convention used below by
  // AnimatedScene::sample. This guarantees that frame zero is the original
  // mesh, which is the essential invariant for a recognizable animation.
  let mut parents = vec![None; scene.nodes.len()];
  for (parent, node) in scene.nodes.iter().enumerate() {
    for child in &node.children {
      if (child.0 as usize) < parents.len() {
        parents[child.0 as usize] = Some(parent);
      }
    }
  }
  let locals = scene
    .nodes
    .iter()
    .map(|node| {
      let (translation, rotation, scale) = fbx_transform(node.transform);
      Mat4::from_scale_rotation_translation(
        Vec3::from_array(scale),
        Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
        Vec3::from_array(translation),
      )
    })
    .collect::<Vec<_>>();
  let mut world = vec![None; scene.nodes.len()];

  let mut skin_geometry = HashMap::new();
  let mut skin_clusters: HashMap<i64, Vec<i64>> = HashMap::new();
  let mut cluster_bone = HashMap::new();
  let mut geometry_corners = HashMap::new();
  let Some(connections) = document.root.child("Connections") else {
    return;
  };
  for connection in connections.children_named("C") {
    if connection.properties.first().and_then(FbxProperty::as_str) != Some("OO") {
      continue;
    }
    let Some(child) = connection.properties.get(1).and_then(FbxProperty::as_i64) else {
      continue;
    };
    let Some(parent) = connection.properties.get(2).and_then(FbxProperty::as_i64) else {
      continue;
    };
    if skins.contains_key(&child) && geometry_meshes.contains_key(&parent) {
      skin_geometry.insert(child, parent);
    } else if clusters.contains_key(&child) && skins.contains_key(&parent) {
      skin_clusters.entry(parent).or_default().push(child);
    } else if clusters.contains_key(&child) && model_nodes.contains_key(&parent) {
      // Standard direction: Cluster -> Model.
      cluster_bone.insert(child, parent);
    } else if model_nodes.contains_key(&child) && clusters.contains_key(&parent) {
      // Reversed direction emitted by the affected FBX exporter.
      cluster_bone.insert(parent, child);
    }
  }

  // PolygonVertexIndex is shared-vertex space, while the decoder's render
  // primitive is a per-corner buffer.  Preserve that relationship when
  // expanding each cluster's weights to render vertices.
  for geometry_id in skin_geometry.values().copied() {
    let Some(geometry) = objects.children.iter().find(|object| {
      object.name == "Geometry"
        && object.properties.first().and_then(FbxProperty::as_i64) == Some(geometry_id)
    }) else {
      continue;
    };
    let Some(values) = geometry
      .child("PolygonVertexIndex")
      .and_then(|node| node.properties.first())
      .and_then(fbx_i32_array)
    else {
      continue;
    };
    let mut corners = Vec::new();
    let mut polygon = Vec::new();
    for value in values {
      polygon.push(if *value < 0 {
        (-*value - 1) as u32
      } else {
        *value as u32
      });
      if *value < 0 {
        // oxideav-fbx fans polygons exactly this way when it creates
        // the render primitive, so the recovered weights stay aligned
        // with its per-corner positions and normals.
        for index in 1..polygon.len().saturating_sub(1) {
          corners.extend([polygon[0], polygon[index], polygon[index + 1]]);
        }
        polygon.clear();
      }
    }
    geometry_corners.insert(geometry_id, corners);
  }

  for (skin_id, geometry_id) in skin_geometry {
    let Some(&mesh_id) = geometry_meshes.get(&geometry_id) else {
      continue;
    };
    let Some(cluster_ids) = skin_clusters.get(&skin_id) else {
      continue;
    };
    let Some(corner_indices) = geometry_corners.get(&geometry_id) else {
      continue;
    };
    let mesh_node = scene
      .nodes
      .iter()
      .position(|node| node.mesh == Some(mesh_id));
    let mesh_world = mesh_node.map(|node| source_world_matrix(node, &parents, &locals, &mut world));
    let Some(mesh) = scene.meshes.get_mut(mesh_id.0 as usize) else {
      continue;
    };
    let Some(primitive) = mesh.primitives.first_mut() else {
      continue;
    };
    if primitive.positions.len() != corner_indices.len() {
      continue;
    }

    // Build the shared-index lookup once. The previous implementation
    // scanned every render corner for every cluster weight, turning FBX
    // startup into O(clusters × corners). This map makes import linear in
    // the size of the geometry and the authored weight arrays.
    let mut shared_to_corners: HashMap<u32, Vec<usize>> = HashMap::new();
    for (corner, &shared_index) in corner_indices.iter().enumerate() {
      shared_to_corners
        .entry(shared_index)
        .or_default()
        .push(corner);
    }

    let mut skeleton = oxideav_mesh3d::Skeleton::new();
    let mut corner_weights = vec![Vec::<(u16, f32)>::new(); corner_indices.len()];
    for (joint_index, cluster_id) in cluster_ids.iter().enumerate() {
      let Some(cluster) = clusters.get(cluster_id) else {
        continue;
      };
      let Some(&bone_id) = cluster_bone.get(cluster_id) else {
        continue;
      };
      let Some(&bone_node) = model_nodes.get(&bone_id) else {
        continue;
      };
      skeleton.joints.push(bone_node);
      let inverse_bind = mesh_world
        .map(|mesh_world| {
          let joint_world =
            source_world_matrix(bone_node.0 as usize, &parents, &locals, &mut world);
          (joint_world.inverse() * mesh_world).to_cols_array_2d()
        })
        .unwrap_or_else(|| {
          let transform = fbx_matrix(cluster, "Transform").unwrap_or(fbx_identity_matrix());
          let transform_link =
            fbx_matrix(cluster, "TransformLink").unwrap_or(fbx_identity_matrix());
          fbx_matrix_mul(fbx_inverse_affine(&transform_link), transform)
        });
      skeleton.inverse_bind_matrices.push(inverse_bind);

      let indices = cluster
        .child("Indexes")
        .and_then(|node| node.properties.first())
        .and_then(fbx_i32_array)
        .unwrap_or(&[]);
      let weights = cluster
        .child("Weights")
        .and_then(|node| node.properties.first())
        .and_then(fbx_f64_array)
        .unwrap_or(&[]);
      for (shared_index, weight) in indices.iter().zip(weights.iter()) {
        if *shared_index < 0 || *weight == 0.0 || joint_index > u16::MAX as usize {
          continue;
        }
        if let Some(corners) = shared_to_corners.get(&(*shared_index as u32)) {
          for &corner in corners {
            corner_weights[corner].push((joint_index as u16, *weight as f32));
          }
        }
      }
    }
    if skeleton.joints.is_empty() {
      continue;
    }

    let mut joints = vec![[0u16; 4]; corner_weights.len()];
    let mut weights = vec![[0.0f32; 4]; corner_weights.len()];
    for (corner, influences) in corner_weights.iter_mut().enumerate() {
      influences.sort_by(|left, right| right.1.total_cmp(&left.1));
      for (slot, &(joint, weight)) in influences.iter().take(4).enumerate() {
        joints[corner][slot] = joint;
        weights[corner][slot] = weight;
      }
      let total: f32 = weights[corner].iter().sum();
      if total > f32::EPSILON {
        for weight in &mut weights[corner] {
          *weight /= total;
        }
      }
    }
    primitive.joints = Some(joints);
    primitive.weights = Some(weights);
    let skeleton_id = scene.add_skeleton(skeleton);
    let skin_id = scene.add_skin(oxideav_mesh3d::Skin::new(skeleton_id));
    if let Some(node) = scene
      .nodes
      .iter_mut()
      .find(|node| node.mesh == Some(mesh_id))
    {
      node.skin = Some(skin_id);
    }
  }
}

fn fbx_i32_array(property: &FbxProperty) -> Option<&[i32]> {
  match property {
    FbxProperty::I32Array(values) => Some(values.as_slice()),
    _ => None,
  }
}

fn fbx_f64_array(property: &FbxProperty) -> Option<&[f64]> {
  match property {
    FbxProperty::F64Array(values) => Some(values.as_slice()),
    _ => None,
  }
}

fn fbx_matrix(node: &oxideav_fbx::binary::FbxNode, name: &str) -> Option<[[f32; 4]; 4]> {
  let values = node
    .child(name)
    .and_then(|child| child.properties.first())
    .and_then(fbx_f64_array)?;
  (values.len() == 16).then(|| {
    let mut matrix = [[0.0; 4]; 4];
    for row in 0..4 {
      for column in 0..4 {
        matrix[row][column] = values[row * 4 + column] as f32;
      }
    }
    matrix
  })
}

fn fbx_identity_matrix() -> [[f32; 4]; 4] {
  [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ]
}

fn fbx_matrix_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
  let mut result = [[0.0; 4]; 4];
  for row in 0..4 {
    for column in 0..4 {
      result[row][column] = (0..4).map(|index| a[row][index] * b[index][column]).sum();
    }
  }
  result
}

/// Computes one decoded FBX node's rest-pose world matrix while repairing a
/// skin. The cache makes the hierarchy traversal linear in the node count even
/// when dozens of clusters refer to the same armature.
fn source_world_matrix(
  index: usize,
  parents: &[Option<usize>],
  locals: &[Mat4],
  cache: &mut [Option<Mat4>],
) -> Mat4 {
  if let Some(matrix) = cache[index] {
    return matrix;
  }
  let matrix = parents[index]
    .map(|parent| source_world_matrix(parent, parents, locals, cache) * locals[index])
    .unwrap_or(locals[index]);
  cache[index] = Some(matrix);
  matrix
}

fn fbx_inverse_affine(matrix: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
  // This is the same affine inverse convention used by oxideav-fbx for
  // TransformLink.  Singular bind matrices degrade to identity rather than
  // making an otherwise viewable FBX fail during import.
  let a = matrix[0][0];
  let b = matrix[0][1];
  let c = matrix[0][2];
  let d = matrix[1][0];
  let e = matrix[1][1];
  let f = matrix[1][2];
  let g = matrix[2][0];
  let h = matrix[2][1];
  let i = matrix[2][2];
  let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if determinant.abs() <= f32::EPSILON {
    return fbx_identity_matrix();
  }
  let inverse = 1.0 / determinant;
  let linear = [
    [
      (e * i - f * h) * inverse,
      (c * h - b * i) * inverse,
      (b * f - c * e) * inverse,
    ],
    [
      (f * g - d * i) * inverse,
      (a * i - c * g) * inverse,
      (c * d - a * f) * inverse,
    ],
    [
      (d * h - e * g) * inverse,
      (b * g - a * h) * inverse,
      (a * e - b * d) * inverse,
    ],
  ];
  let translation = [matrix[0][3], matrix[1][3], matrix[2][3]];
  let dot =
    |row: [f32; 3]| row[0] * translation[0] + row[1] * translation[1] + row[2] * translation[2];
  [
    [linear[0][0], linear[0][1], linear[0][2], -dot(linear[0])],
    [linear[1][0], linear[1][1], linear[1][2], -dot(linear[1])],
    [linear[2][0], linear[2][1], linear[2][2], -dot(linear[2])],
    [0.0, 0.0, 0.0, 1.0],
  ]
}

#[derive(Clone)]
struct SceneNode {
  parent:      Option<usize>,
  translation: Vec3,
  rotation:    Quat,
  scale:       Vec3,
  weights:     Vec<f32>,
}

#[derive(Clone)]
struct ScenePrimitive {
  node:            usize,
  positions:       Vec<Vec3>,
  normals:         Option<Vec<Vec3>>,
  indices:         Vec<u32>,
  joints:          Vec<[u16; 4]>,
  weights:         Vec<[f32; 4]>,
  morph_positions: Vec<Vec<Vec3>>,
  skin:            Option<SceneSkin>,
}

#[derive(Clone)]
struct SceneSkin {
  joints:       Vec<usize>,
  inverse_bind: Vec<Mat4>,
}

#[derive(Clone)]
struct AnimationClip {
  info:     AnimationInfo,
  channels: Vec<AnimationChannel>,
}

impl AnimationClip {
  fn info(&self) -> AnimationInfo {
    self.info.clone()
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Interpolation {
  Linear,
  Step,
  CubicSpline,
}

#[derive(Clone)]
struct AnimationChannel {
  node:          usize,
  interpolation: Interpolation,
  data:          ChannelData,
}

#[derive(Clone)]
enum ChannelData {
  Translation(Vec<Key<[f32; 3]>>),
  Rotation(Vec<Key<[f32; 4]>>),
  Scale(Vec<Key<[f32; 3]>>),
  Weights(Vec<Key<Vec<f32>>>),
}

#[derive(Clone)]
struct Key<T> {
  time:        f32,
  value:       T,
  in_tangent:  Option<T>,
  out_tangent: Option<T>,
}

impl AnimatedScene {
  fn from_document(document: gltf::Document, buffers: Vec<Vec<u8>>) -> Result<Self> {
    let get_buffer =
      |buffer: gltf::Buffer<'_>| buffers.get(buffer.index()).map(|data| data.as_slice());
    let document_ref = &document;
    let gltf_nodes: Vec<_> = document_ref.nodes().collect();
    let mut parents = vec![None; gltf_nodes.len()];
    for node in &gltf_nodes {
      for child in node.children() {
        parents[child.index()] = Some(node.index());
      }
    }

    let nodes: Vec<SceneNode> = gltf_nodes
      .iter()
      .map(|node| {
        let (translation, rotation, scale) = node.transform().decomposed();
        let weights = node
          .weights()
          .or_else(|| node.mesh().and_then(|mesh| mesh.weights()))
          .unwrap_or(&[])
          .to_vec();
        SceneNode {
          parent: parents[node.index()],
          translation: Vec3::from_array(translation),
          rotation: Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
          scale: Vec3::from_array(scale),
          weights,
        }
      })
      .collect();

    let mut active = vec![false; gltf_nodes.len()];
    let roots: Vec<_> = document_ref
      .default_scene()
      .or_else(|| document_ref.scenes().next())
      .map(|scene| scene.nodes().map(|node| node.index()).collect())
      .unwrap_or_else(|| {
        gltf_nodes
          .iter()
          .filter(|node| node.mesh().is_some())
          .map(|node| node.index())
          .collect()
      });
    for root in roots {
      mark_active(root, &gltf_nodes, &mut active);
    }

    let skins: Vec<SceneSkin> = document_ref
      .skins()
      .map(|skin| {
        let joints: Vec<_> = skin.joints().map(|joint| joint.index()).collect();
        let inverse_bind = skin
          .reader(get_buffer)
          .read_inverse_bind_matrices()
          .map(|matrices| {
            matrices
              .map(|matrix| Mat4::from_cols_array_2d(&matrix))
              .collect()
          })
          .unwrap_or_else(|| vec![Mat4::IDENTITY; joints.len()]);
        SceneSkin {
          joints,
          inverse_bind,
        }
      })
      .collect();

    let mut primitives = Vec::new();
    for node in gltf_nodes.iter().filter(|node| active[node.index()]) {
      let Some(mesh) = node.mesh() else { continue };
      for primitive in mesh.primitives() {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
          bail!("glTF primitive {} is not triangulated", primitive.index());
        }
        let reader = primitive.reader(get_buffer);
        let positions: Vec<Vec3> = reader
          .read_positions()
          .ok_or_else(|| anyhow!("glTF primitive has no POSITION attribute"))?
          .map(Vec3::from_array)
          .collect();
        let indices = reader
          .read_indices()
          .map(|indices| indices.into_u32().collect())
          .unwrap_or_else(|| (0..positions.len() as u32).collect());
        let joints = reader
          .read_joints(0)
          .map(|values| values.into_u16().collect())
          .unwrap_or_default();
        let weights = reader
          .read_weights(0)
          .map(|values| values.into_f32().collect())
          .unwrap_or_default();
        let normals = reader
          .read_normals()
          .map(|values| values.map(Vec3::from_array).collect());
        let morph_positions = reader
          .read_morph_targets()
          .filter_map(|(positions, _, _)| {
            positions.map(|values| values.map(Vec3::from_array).collect())
          })
          .collect();
        let skin = node
          .skin()
          .and_then(|skin| skins.get(skin.index()).cloned());
        primitives.push(ScenePrimitive {
          node: node.index(),
          positions,
          normals,
          indices,
          joints,
          weights,
          morph_positions,
          skin,
        });
      }
    }
    if primitives.is_empty() {
      bail!("glTF scene contains no triangle meshes")
    }

    let animations = document_ref
      .animations()
      .map(|animation| parse_animation(animation, &buffers))
      .collect::<Result<Vec<_>>>()?;
    Ok(Self {
      nodes,
      primitives,
      animations,
    })
  }
}

/// Reads only geometry/animation buffers. Using this small loader instead of
/// glTF's image-enabled importer keeps image crates and texture decoding out
/// of both native startup and the WASM bundle.
fn load_buffers(
  document: &gltf::Document,
  base: Option<&Path>,
  mut blob: Option<Vec<u8>>,
) -> Result<Vec<Vec<u8>>> {
  document
    .buffers()
    .map(|buffer| match buffer.source() {
      gltf::buffer::Source::Bin => blob
        .take()
        .ok_or_else(|| anyhow!("glTF buffer {} has no GLB binary payload", buffer.index())),
      gltf::buffer::Source::Uri(uri) => load_buffer_uri(uri, base),
    })
    .collect()
}

fn load_buffer_uri(uri: &str, base: Option<&Path>) -> Result<Vec<u8>> {
  #[cfg(target_arch = "wasm32")]
  let _ = base;
  if let Some(data) = uri.strip_prefix("data:") {
    let (_, payload) = data
      .split_once(',')
      .ok_or_else(|| anyhow!("invalid glTF data URI"))?;
    if data[..data.len() - payload.len() - 1].contains(";base64") {
      return decode_base64(payload);
    }
    return percent_decode(payload);
  }

  #[cfg(not(target_arch = "wasm32"))]
  if let Some(base) = base {
    let path = percent_decode(uri)
      .and_then(|bytes| String::from_utf8(bytes).context("glTF buffer URI is not UTF-8"))?;
    return std::fs::read(base.join(path)).context("reading external glTF buffer");
  }

  bail!("external glTF buffer URI requires a native file base path: {uri}")
}

fn percent_decode(value: &str) -> Result<Vec<u8>> {
  let bytes = value.as_bytes();
  let mut result = Vec::with_capacity(bytes.len());
  let mut index = 0;
  while index < bytes.len() {
    if bytes[index] == b'%' {
      if index + 2 >= bytes.len() {
        bail!("invalid percent escape in glTF URI")
      }
      let high = hex_digit(bytes[index + 1])?;
      let low = hex_digit(bytes[index + 2])?;
      result.push(high * 16 + low);
      index += 3;
    } else {
      result.push(bytes[index]);
      index += 1;
    }
  }
  Ok(result)
}

fn decode_base64(value: &str) -> Result<Vec<u8>> {
  let mut result = Vec::with_capacity(value.len() * 3 / 4);
  let mut quartet = [0u8; 4];
  let mut count = 0;
  for byte in value.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
    quartet[count] = match byte {
      b'A'..=b'Z' => byte - b'A',
      b'a'..=b'z' => byte - b'a' + 26,
      b'0'..=b'9' => byte - b'0' + 52,
      b'+' => 62,
      b'/' => 63,
      b'=' => 64,
      _ => bail!("invalid base64 glTF buffer URI"),
    };
    count += 1;
    if count == 4 {
      result.push((quartet[0] << 2) | (quartet[1] >> 4));
      if quartet[2] != 64 {
        result.push((quartet[1] << 4) | (quartet[2] >> 2));
      }
      if quartet[3] != 64 {
        result.push((quartet[2] << 6) | quartet[3]);
      }
      count = 0;
    }
  }
  if count != 0 {
    bail!("incomplete base64 glTF buffer URI")
  }
  Ok(result)
}

fn hex_digit(byte: u8) -> Result<u8> {
  match byte {
    b'0'..=b'9' => Ok(byte - b'0'),
    b'a'..=b'f' => Ok(byte - b'a' + 10),
    b'A'..=b'F' => Ok(byte - b'A' + 10),
    _ => bail!("invalid hexadecimal escape in glTF URI"),
  }
}

fn mark_active(index: usize, nodes: &[gltf::Node<'_>], active: &mut [bool]) {
  if active[index] {
    return;
  }
  active[index] = true;
  for child in nodes[index].children() {
    mark_active(child.index(), nodes, active);
  }
}

fn world_matrix(
  index: usize,
  nodes: &[SceneNode],
  locals: &[Mat4],
  cache: &mut [Option<Mat4>],
) -> Mat4 {
  if let Some(matrix) = cache[index] {
    return matrix;
  }
  let matrix = nodes[index]
    .parent
    .map(|parent| world_matrix(parent, nodes, locals, cache) * locals[index])
    .unwrap_or(locals[index]);
  cache[index] = Some(matrix);
  matrix
}

/// Converts a CPU matrix into the two column-major matrices consumed by WGSL.
fn gpu_transform(matrix: Mat4) -> SkinTransform {
  SkinTransform {
    matrix: matrix.to_cols_array_2d(),
    normal: matrix.inverse().transpose().to_cols_array_2d(),
  }
}

fn parse_animation(animation: gltf::Animation<'_>, buffers: &[Vec<u8>]) -> Result<AnimationClip> {
  let mut channels = Vec::new();
  let mut duration: f32 = 0.0;
  let mut info_channels = Vec::new();
  for channel in animation.channels() {
    let node = channel.target().node().index();
    let property = channel.target().property();
    let interpolation = match channel.sampler().interpolation() {
      animation::Interpolation::Linear => Interpolation::Linear,
      animation::Interpolation::Step => Interpolation::Step,
      animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
    };
    let reader = channel
      .reader(|buffer: gltf::Buffer<'_>| buffers.get(buffer.index()).map(|data| data.as_slice()));
    let inputs: Vec<f32> = reader
      .read_inputs()
      .ok_or_else(|| anyhow!("animation channel has no input sampler"))?
      .collect();
    if let Some(end) = inputs.iter().copied().reduce(f32::max) {
      duration = duration.max(end);
    }
    let outputs = reader
      .read_outputs()
      .ok_or_else(|| anyhow!("animation channel has no output sampler"))?;
    let data = match (property, outputs) {
      (animation::Property::Translation, animation::util::ReadOutputs::Translations(values)) => {
        ChannelData::Translation(make_vec3_keys(inputs, values.collect(), interpolation))
      }
      (animation::Property::Scale, animation::util::ReadOutputs::Scales(values)) => {
        ChannelData::Scale(make_vec3_keys(inputs, values.collect(), interpolation))
      }
      (animation::Property::Rotation, animation::util::ReadOutputs::Rotations(values)) => {
        ChannelData::Rotation(make_quat_keys(
          inputs,
          values.into_f32().collect(),
          interpolation,
        ))
      }
      (
        animation::Property::MorphTargetWeights,
        animation::util::ReadOutputs::MorphTargetWeights(values),
      ) => {
        let values: Vec<f32> = values.into_f32().collect();
        let stride = values.len().checked_div(inputs.len().max(1)).unwrap_or(0);
        let keys = inputs
          .iter()
          .enumerate()
          .map(|(index, time)| {
            let source_index = match interpolation {
              Interpolation::CubicSpline => index * 3 + 1,
              _ => index,
            };
            let start = source_index * stride;
            Key {
              time:        *time,
              value:       values.get(start..start + stride).unwrap_or(&[]).to_vec(),
              in_tangent:  (interpolation == Interpolation::CubicSpline).then(|| {
                let start = index * 3 * stride;
                values.get(start..start + stride).unwrap_or(&[]).to_vec()
              }),
              out_tangent: (interpolation == Interpolation::CubicSpline).then(|| {
                let start = (index * 3 + 2) * stride;
                values.get(start..start + stride).unwrap_or(&[]).to_vec()
              }),
            }
          })
          .collect();
        ChannelData::Weights(keys)
      }
      _ => bail!("glTF animation channel output does not match its target property"),
    };
    info_channels.push(AnimationChannelInfo {
      node,
      property: property_name(property).to_string(),
      interpolation: interpolation_name(interpolation).to_string(),
    });
    channels.push(AnimationChannel {
      node,
      interpolation,
      data,
    });
  }
  Ok(AnimationClip {
    info: AnimationInfo {
      name: animation
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("Animation {}", animation.index())),
      duration,
      channels: info_channels,
    },
    channels,
  })
}

fn make_vec3_keys(
  inputs: Vec<f32>,
  values: Vec<[f32; 3]>,
  interpolation: Interpolation,
) -> Vec<Key<[f32; 3]>> {
  inputs
    .into_iter()
    .enumerate()
    .map(|(index, time)| {
      let value_index = match interpolation {
        Interpolation::CubicSpline => index * 3 + 1,
        _ => index,
      };
      Key {
        time,
        value: values.get(value_index).copied().unwrap_or([0.0; 3]),
        in_tangent: (interpolation == Interpolation::CubicSpline)
          .then(|| values.get(index * 3).copied().unwrap_or([0.0; 3])),
        out_tangent: (interpolation == Interpolation::CubicSpline)
          .then(|| values.get(index * 3 + 2).copied().unwrap_or([0.0; 3])),
      }
    })
    .collect()
}

fn make_quat_keys(
  inputs: Vec<f32>,
  values: Vec<[f32; 4]>,
  interpolation: Interpolation,
) -> Vec<Key<[f32; 4]>> {
  inputs
    .into_iter()
    .enumerate()
    .map(|(index, time)| {
      let value_index = match interpolation {
        Interpolation::CubicSpline => index * 3 + 1,
        _ => index,
      };
      Key {
        time,
        value: values
          .get(value_index)
          .copied()
          .unwrap_or([0.0, 0.0, 0.0, 1.0]),
        in_tangent: (interpolation == Interpolation::CubicSpline)
          .then(|| values.get(index * 3).copied().unwrap_or([0.0; 4])),
        out_tangent: (interpolation == Interpolation::CubicSpline)
          .then(|| values.get(index * 3 + 2).copied().unwrap_or([0.0; 4])),
      }
    })
    .collect()
}

fn sample_vec3(keys: &[Key<[f32; 3]>], interpolation: Interpolation, time: f32) -> Option<Vec3> {
  let (a, b, factor) = key_pair(keys, interpolation, time)?;
  if interpolation != Interpolation::CubicSpline {
    return Some(Vec3::from_array(a.value).lerp(Vec3::from_array(b.value), factor));
  }
  let delta = b.time - a.time;
  let (h00, h10, h01, h11) = hermite_basis(factor);
  Some(
    Vec3::from_array(a.value) * h00
      + Vec3::from_array(a.out_tangent.unwrap_or([0.0; 3])) * (h10 * delta)
      + Vec3::from_array(b.value) * h01
      + Vec3::from_array(b.in_tangent.unwrap_or([0.0; 3])) * (h11 * delta),
  )
}

fn sample_quat(keys: &[Key<[f32; 4]>], interpolation: Interpolation, time: f32) -> Option<Quat> {
  let (a, b, factor) = key_pair(keys, interpolation, time)?;
  if interpolation != Interpolation::CubicSpline {
    return Some(
      Quat::from_xyzw(a.value[0], a.value[1], a.value[2], a.value[3]).slerp(
        Quat::from_xyzw(b.value[0], b.value[1], b.value[2], b.value[3]),
        factor,
      ),
    );
  }
  let delta = b.time - a.time;
  let (h00, h10, h01, h11) = hermite_basis(factor);
  let in_tangent = b.in_tangent.unwrap_or([0.0; 4]);
  let out_tangent = a.out_tangent.unwrap_or([0.0; 4]);
  let value: [f32; 4] = std::array::from_fn(|index| {
    a.value[index] * h00
      + out_tangent[index] * (h10 * delta)
      + b.value[index] * h01
      + in_tangent[index] * (h11 * delta)
  });
  Some(Quat::from_xyzw(value[0], value[1], value[2], value[3]).normalize())
}

fn sample_weights(
  keys: &[Key<Vec<f32>>],
  interpolation: Interpolation,
  time: f32,
) -> Option<Vec<f32>> {
  let (a, b, factor) = key_pair(keys, interpolation, time)?;
  if interpolation != Interpolation::CubicSpline {
    return Some(
      a.value
        .iter()
        .zip(b.value.iter())
        .map(|(a, b)| a + (b - a) * factor)
        .collect(),
    );
  }
  let delta = b.time - a.time;
  let (h00, h10, h01, h11) = hermite_basis(factor);
  let out_tangent = a.out_tangent.as_deref().unwrap_or(&[]);
  let in_tangent = b.in_tangent.as_deref().unwrap_or(&[]);
  Some(
    a.value
      .iter()
      .enumerate()
      .map(|(index, value)| {
        value * h00
          + out_tangent.get(index).copied().unwrap_or(0.0) * (h10 * delta)
          + b.value.get(index).copied().unwrap_or(0.0) * h01
          + in_tangent.get(index).copied().unwrap_or(0.0) * (h11 * delta)
      })
      .collect(),
  )
}

fn hermite_basis(t: f32) -> (f32, f32, f32, f32) {
  let t2 = t * t;
  let t3 = t2 * t;
  (
    2.0 * t3 - 3.0 * t2 + 1.0,
    t3 - 2.0 * t2 + t,
    -2.0 * t3 + 3.0 * t2,
    t3 - t2,
  )
}

fn key_pair<T>(
  keys: &[Key<T>],
  interpolation: Interpolation,
  time: f32,
) -> Option<(&Key<T>, &Key<T>, f32)> {
  let first = keys.first()?;
  let last = keys.last()?;
  if time <= first.time {
    return Some((first, first, 0.0));
  }
  if time >= last.time {
    return Some((last, last, 0.0));
  }
  let next = keys
    .iter()
    .position(|key| key.time >= time)
    .unwrap_or(keys.len() - 1);
  let a = &keys[next - 1];
  let b = &keys[next];
  let factor = if matches!(interpolation, Interpolation::Step) {
    0.0
  } else {
    ((time - a.time) / (b.time - a.time)).clamp(0.0, 1.0)
  };
  Some((a, b, factor))
}

fn wrap_time(time: f32, duration: f32) -> f32 {
  if duration <= 0.0 {
    0.0
  } else {
    let wrapped = time.rem_euclid(duration);
    // Preserve a positive exact clip endpoint. This matters to trajectory
    // extraction: the final uniform sample should be the authored last pose,
    // not an immediate jump back to the first pose. A time of zero remains
    // the first pose as usual.
    if time > 0.0 && wrapped == 0.0 {
      duration
    } else {
      wrapped
    }
  }
}

fn property_name(property: animation::Property) -> &'static str {
  match property {
    animation::Property::Translation => "translation",
    animation::Property::Rotation => "rotation",
    animation::Property::Scale => "scale",
    animation::Property::MorphTargetWeights => "weights",
  }
}

fn interpolation_name(interpolation: Interpolation) -> &'static str {
  match interpolation {
    Interpolation::Linear => "LINEAR",
    Interpolation::Step => "STEP",
    Interpolation::CubicSpline => "CUBICSPLINE",
  }
}

#[cfg(test)]
mod tests {
  use super::AnimatedScene;
  use crate::common::Vec3;

  /// Keep the supplied FBX as a regression fixture: this catches both
  /// decoder-version changes and accidental loss of animation conversion
  /// before a browser build is generated.
  #[test]
  fn supplied_inside_crescent_kick_fbx_loads_with_animation() {
    let bytes = include_bytes!("../../public/models/Inside Crescent Kick.fbx");
    let scene = AnimatedScene::from_bytes(bytes).expect("supplied FBX should load");
    let animations = scene.animations();
    assert!(
      !animations.is_empty(),
      "the supplied FBX should expose animation"
    );
    assert!(
      scene
        .primitives
        .iter()
        .all(|primitive| primitive.normals.is_some()),
      "the supplied FBX should expose authored vertex normals"
    );
    assert!(!scene.initial_mesh().vertices.is_empty());
    let gpu_mesh = scene.gpu_mesh();
    assert_eq!(
      gpu_mesh.vertices.len(),
      scene.initial_mesh().vertices.len(),
      "FBX corners must not be duplicated for GPU skinning"
    );
    assert!(!gpu_mesh.transforms.is_empty());

    // A metadata-only parse is not enough: verify that sampling the clip
    // actually changes the skinned surface away from its rest pose.
    let rest = scene.sample(Some(0), 0.0);
    let animated = scene.sample(Some(0), animations[0].duration * 0.5);
    let mut reused = scene.initial_mesh();
    scene.sample_into(Some(0), animations[0].duration * 0.5, &mut reused);
    assert_eq!(reused.vertices.len(), animated.vertices.len());
    assert!(
      reused
        .vertices
        .iter()
        .zip(animated.vertices.iter())
        .all(|(a, b)| a.position == b.position && a.normal == b.normal)
    );
    assert!(
      rest
        .vertices
        .iter()
        .zip(animated.vertices.iter())
        .any(
          |(a, b)| Vec3::from_array(a.position).distance_squared(Vec3::from_array(b.position))
            > 1.0e-8
        ),
      "sampling the first FBX animation should move at least one vertex"
    );
  }
}
