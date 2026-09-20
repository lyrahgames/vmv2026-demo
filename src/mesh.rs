//! OBJ loading and the CPU-side mesh representation used by wgpu.

use crate::common::*;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
  // These fields deliberately match the WGSL vertex input layout: position
  // at location 0 and normal at location 1.
  pub position: [f32; 3],
  pub normal:   [f32; 3],
}

/// Vertex format used by animated scenes.
///
/// The position and authored normal remain in mesh-local space.  The vertex
/// shader applies the node transform or the four joint transforms, so the
/// CPU never rewrites this array while an animation is playing.  `base_transform`
/// is also used for unskinned primitives and for vertices whose skin weights
/// sum to zero.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SkinnedVertex {
  pub position:          [f32; 3],
  pub normal:            [f32; 3],
  pub base_transform:    u32,
  pub joints:            [u32; 4],
  pub weights:           [f32; 4],
  pub morph_base:        u32,
  pub morph_count:       u32,
  pub morph_weight_base: u32,
}

impl SkinnedVertex {
  /// Vertex attributes consumed by `vs_skinned` in the animated shader.
  pub const ATTRIBUTES: [wgpu::VertexAttribute; 8] = wgpu::vertex_attr_array![
      0 => Float32x3,
      1 => Float32x3,
      2 => Uint32,
      3 => Uint32x4,
      4 => Float32x4,
      5 => Uint32,
      6 => Uint32,
      7 => Uint32,
  ];

  pub fn layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Self>() as u64,
      step_mode:    wgpu::VertexStepMode::Vertex,
      attributes:   &Self::ATTRIBUTES,
    }
  }
}

/// One pose transform uploaded to the animated scene's storage buffer.
///
/// Keeping the inverse-transpose beside the transform avoids a matrix inverse
/// in WGSL and lets normal skinning use the same four weights as position
/// skinning.  The values are updated once per frame, once per node/skin entry,
/// rather than once per vertex.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SkinTransform {
  pub matrix: [[f32; 4]; 4],
  pub normal: [[f32; 4]; 4],
}

impl Vertex {
  // wgpu uses this declaration to interpret each packed Vertex in the GPU
  // vertex buffer.  The order must match the Rust struct and shader inputs.
  pub const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

  /// Describes the stride and attributes of one vertex to wgpu.
  pub fn layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Self>() as u64,
      step_mode:    wgpu::VertexStepMode::Vertex,
      attributes:   &Self::ATTRIBUTES,
    }
  }
}

/// A triangulated mesh plus bounds used to frame the camera.
#[derive(Clone)]
pub struct Mesh {
  /// Interleaved position/normal data uploaded to the vertex buffer.
  pub vertices: Vec<Vertex>,
  /// Triangle indices uploaded to the index buffer.
  pub indices:  Vec<u32>,
  /// Minimum position along each world axis.
  pub min:      Vec3,
  /// Maximum position along each world axis.
  pub max:      Vec3,
}

impl Mesh {
  /// Parses OBJ text and converts every model into one indexed mesh.
  ///
  /// `single_index: false` keeps the source position and normal indices
  /// visible. The renderer has no UV/material seam semantics, so normals are
  /// accumulated onto the shared position vertices instead of causing a
  /// second vertex to be manufactured for each OBJ attribute combination.
  pub fn from_obj_text(text: &str) -> Result<Self> {
    let (models, _) = tobj::load_obj_buf(
      &mut std::io::Cursor::new(text.as_bytes()),
      &tobj::LoadOptions {
        single_index: false,
        triangulate: true,
        ..Default::default()
      },
      |_| Ok((Vec::new(), Default::default())),
    )?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for model in models {
      let m = model.mesh;
      // Each OBJ model has its own local index space.  Offset its indices
      // by the number of vertices already collected.
      let base = vertices.len() as u32;
      let mut normal_sums = vec![Vec3::ZERO; m.positions.len() / 3];
      let mut normal_counts = vec![0u32; m.positions.len() / 3];
      for (corner, &position_index) in m.indices.iter().enumerate() {
        let Some(&normal_index) = m.normal_indices.get(corner) else {
          continue;
        };
        let normal_start = normal_index as usize * 3;
        if normal_start + 2 >= m.normals.len() {
          continue;
        }
        let position_index = position_index as usize;
        if position_index >= normal_sums.len() {
          continue;
        }
        normal_sums[position_index] += Vec3::new(
          m.normals[normal_start],
          m.normals[normal_start + 1],
          m.normals[normal_start + 2],
        );
        normal_counts[position_index] += 1;
      }
      for i in 0..m.positions.len() / 3 {
        let p = Vec3::new(
          m.positions[i * 3],
          m.positions[i * 3 + 1],
          m.positions[i * 3 + 2],
        );
        let n = if normal_counts[i] > 0 {
          (normal_sums[i] / normal_counts[i] as f32).normalize_or_zero()
        } else {
          Vec3::ZERO
        };
        vertices.push(Vertex {
          position: p.to_array(),
          normal:   n.to_array(),
        });
      }
      indices.extend(m.indices.into_iter().map(|i| base + i));
    }
    Self::from_parts(vertices, indices)
  }

  /// Finishes a mesh assembled by any loader.
  ///
  /// OBJ and glTF use different source representations, but the renderer
  /// deliberately consumes the same compact position/normal/index format.
  /// Keeping normal generation and bounds calculation here means animated
  /// glTF frames use exactly the same camera and lighting conventions as OBJ
  /// meshes.
  pub fn from_parts(mut vertices: Vec<Vertex>, indices: Vec<u32>) -> Result<Self> {
    if vertices.is_empty() {
      // A mesh with no vertices would create invalid/meaningless GPU
      // buffers, so fail early with a useful error.
      bail!("mesh contains no vertices")
    }
    Self::fill_missing_normals(&mut vertices, &indices);
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    // Bounds are calculated once here and reused by camera reset logic.
    for v in &vertices {
      let p = Vec3::from_array(v.position);
      min = min.min(p);
      max = max.max(p);
    }
    Ok(Self {
      vertices,
      indices,
      min,
      max,
    })
  }

  /// Fills only missing normals while preserving every authored normal.
  ///
  /// Animated playback uses this helper directly because it reuses the
  /// existing vertex allocation instead of constructing a temporary `Mesh`
  /// for every frame. The source winding is the only orientation authority;
  /// guessing an orientation from the mesh center creates incorrect shading
  /// on concave or disconnected surfaces.
  pub(crate) fn fill_missing_normals(vertices: &mut [Vertex], indices: &[u32]) {
    if !vertices
      .iter()
      .any(|vertex| Vec3::from_array(vertex.normal).length_squared() == 0.0)
    {
      return;
    }
    let mut sums = vec![Vec3::ZERO; vertices.len()];
    for tri in indices.chunks_exact(3) {
      let a = Vec3::from_array(vertices[tri[0] as usize].position);
      let b = Vec3::from_array(vertices[tri[1] as usize].position);
      let c = Vec3::from_array(vertices[tri[2] as usize].position);
      let n = (b - a).cross(c - a);
      sums[tri[0] as usize] += n;
      sums[tri[1] as usize] += n;
      sums[tri[2] as usize] += n;
    }
    for (vertex, sum) in vertices.iter_mut().zip(sums) {
      if Vec3::from_array(vertex.normal).length_squared() == 0.0 {
        vertex.normal = sum.normalize_or_zero().to_array();
      }
    }
  }

  /// Reads a file for the native Lua path, then reuses the text parser.
  pub fn from_file(path: &str) -> Result<Self> {
    Self::from_obj_text(
      &std::fs::read_to_string(path).with_context(|| format!("reading OBJ {path}"))?,
    )
  }
}

#[cfg(test)]
mod tests {
  use super::{Mesh, Vertex};

  #[test]
  fn authored_normals_survive_partial_fallback_generation() {
    // Two normals are authored and one is absent.  The missing entry must
    // receive a fallback, while the two source values must remain exactly
    // as supplied by the asset instead of being replaced by face averages.
    let mesh = Mesh::from_parts(
      vec![
        Vertex {
          position: [0.0, 0.0, 0.0],
          normal:   [0.0, 0.0, 1.0],
        },
        Vertex {
          position: [1.0, 0.0, 0.0],
          normal:   [0.0, 0.0, 1.0],
        },
        Vertex {
          position: [0.0, 1.0, 0.0],
          normal:   [0.0; 3],
        },
      ],
      vec![0, 1, 2],
    )
    .expect("test triangle should be valid");

    assert_eq!(mesh.vertices[0].normal, [0.0, 0.0, 1.0]);
    assert_eq!(mesh.vertices[1].normal, [0.0, 0.0, 1.0]);
    assert_eq!(mesh.vertices[2].normal, [0.0, 0.0, 1.0]);
  }
}
