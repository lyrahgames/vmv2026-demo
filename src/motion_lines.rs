//! Seed selection and GPU-side motion-line sampling.
//!
//! A motion-line bundle is deliberately kept separate from animation playback:
//! the selected animation is sampled over its complete duration once, while
//! the ordinary mesh can continue to play, pause, or seek independently.  The
//! CPU prepares the animation palettes because the existing scene sampler owns
//! the format-specific FBX/glTF interpolation logic; a compute shader then
//! applies those palettes to the selected source vertices and writes the full
//! trajectory buffer on the GPU.

use crate::common::Vec3;

/// Selection mode for importance-based spacetime seeding.
///
/// Stochastic selection uses a deterministic GPU hash as its seeded random
/// source, keeping native and browser output reproducible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportanceSelectionMode {
  Deterministic,
  Stochastic,
}

/// Common options for the shared GPU spacetime selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpacetimeSelectionOptions {
  pub importance: bool,
  pub stochastic: bool,
  pub extended: bool,
}

/// The first extension point for seed selection algorithms.
///
/// Every selector returns indices into the immutable animated vertex buffer.
/// Consequently a seed is always an actual surface vertex, never an
/// interpolated point or an independently invented sample.
#[derive(Clone, Debug, PartialEq)]
pub enum SeedSelectionAlgorithm {
  /// Use every vertex in every loaded surface primitive.
  AllVertices,
  /// Select at most `count` distinct vertices with a deterministic
  /// pseudo-random selector. Determinism keeps native and browser showcases
  /// reproducible while still providing a random vertex subset.
  RandomVertices { count: usize },
  /// Select vertices with greedy farthest-point sampling. The first two
  /// seeds form the farthest pair; later seeds maximize their distance to the
  /// closest already selected seed.
  UniformVertices { count: usize },
  /// Select vertices by applying the same greedy max-min procedure to the
  /// complete animation sampled at a lower rate. The positions are generated
  /// and consumed on the GPU before the final high-rate trace is launched.
  UniformSpacetimeVertices { count: usize, sampling_rate: f32 },
  /// Select from the normalized spacetime max-min score. Deterministic mode
  /// chooses the largest probability; stochastic mode samples from it.
  ImportanceSpacetimeVertices {
    count:         usize,
    sampling_rate: f32,
    mode: ImportanceSelectionMode,
  },
  /// Importance spacetime selection additionally weights candidates by their
  /// travelled distance over the sampled animation.
  ExtendedImportanceSpacetimeVertices {
    count: usize,
    sampling_rate: f32,
    mode: ImportanceSelectionMode,
  },
}

impl SeedSelectionAlgorithm {
  /// Selects valid source-vertex indices for algorithms that do not need
  /// candidate positions.
  pub fn select(&self, vertex_count: usize) -> Vec<u32> {
    match self {
      Self::AllVertices => (0..vertex_count as u32).collect(),
      Self::RandomVertices { count } => {
        let target = (*count).min(vertex_count);
        if target == 0 {
          return Vec::new();
        }

        // Partial Fisher-Yates without pulling a general-purpose RNG into the
        // native and wasm builds. The xorshift state is local to this
        // selection, so no global randomness or platform-specific API can
        // make the bundle differ between the two frontends.
        let mut candidates: Vec<u32> = (0..vertex_count as u32).collect();
        let mut state = 0x9E37_79B9_u64 ^ vertex_count as u64 ^ target as u64;
        for index in 0..target {
          state ^= state << 13;
          state ^= state >> 7;
          state ^= state << 17;
          let remaining = vertex_count - index;
          let offset = (state as usize) % remaining;
          candidates.swap(index, index + offset);
        }
        candidates.truncate(target);
        candidates
      }
      // Spatial selection must go through `select_positions`; returning a
      // prefix here would violate the uniform-covering contract silently.
      Self::UniformVertices { .. }
      | Self::UniformSpacetimeVertices { .. }
      | Self::ImportanceSpacetimeVertices { .. }
      | Self::ExtendedImportanceSpacetimeVertices { .. } => Vec::new(),
    }
  }

  /// Selects source vertices while allowing spatial algorithms to inspect
  /// positions without copying the animated GPU vertex stream. The callback
  /// is evaluated only during this method and is never retained.
  pub fn select_positions<F>(&self, vertex_count: usize, position_at: F) -> Vec<u32>
  where
    F: Fn(usize) -> [f32; 3],
  {
    match self {
      Self::UniformVertices { count } => select_uniform_vertices(vertex_count, *count, position_at),
      Self::AllVertices
      | Self::RandomVertices { .. }
      | Self::UniformSpacetimeVertices { .. }
      | Self::ImportanceSpacetimeVertices { .. }
      | Self::ExtendedImportanceSpacetimeVertices { .. } => self.select(vertex_count),
      }
    }

  /// Returns the common GPU configuration for all spacetime strategies.
  /// Keeping this path shared avoids duplicating tracing and score logic.
  pub(crate) fn spacetime_options(&self) -> Option<(usize, f32, SpacetimeSelectionOptions)> {
    match self {
      Self::UniformSpacetimeVertices {
        count,
        sampling_rate,
      } => Some((
        *count,
        *sampling_rate,
        SpacetimeSelectionOptions {
          importance: false,
          stochastic: false,
          extended: false,
        },
      )),
      Self::ImportanceSpacetimeVertices {
        count,
        sampling_rate,
        mode,
      } => Some((
        *count,
        *sampling_rate,
        SpacetimeSelectionOptions {
          importance: true,
          stochastic: matches!(mode, ImportanceSelectionMode::Stochastic),
          extended: false,
        },
      )),
      Self::ExtendedImportanceSpacetimeVertices {
        count,
        sampling_rate,
        mode,
      } => Some((
        *count,
        *sampling_rate,
        SpacetimeSelectionOptions {
          importance: true,
          stochastic: matches!(mode, ImportanceSelectionMode::Stochastic),
          extended: true,
        },
      )),
      _ => None,
    }
  }
}

/// Small deterministic pseudo-random generator used only for tie-breaking.
/// Determinism makes native and browser showcases reproducible, while the
/// reservoir choice still follows the algorithm's random tie rule.
fn next_selection_random(state: &mut u64) -> u64 {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  *state
}

/// Replaces a current best candidate with probability 1/`ties`.
fn replace_random_tie(state: &mut u64, ties: &mut u64) -> bool {
  *ties = ties.saturating_add(1);
  next_selection_random(state) % *ties == 0
}

/// Greedy max-min/farthest-point vertex sampling from the requested formula.
fn select_uniform_vertices<F>(vertex_count: usize, requested: usize, position_at: F) -> Vec<u32>
where
  F: Fn(usize) -> [f32; 3],
{
  let target = requested.min(vertex_count);
  if target == 0 || vertex_count == 0 {
    return Vec::new();
  }

  // Cache positions only for this CPU selection stage. The temporary vector
  // is released before GPU extraction begins and is never uploaded as a
  // second vertex buffer.
  let positions: Vec<Vec3> = (0..vertex_count)
    .map(|index| Vec3::from_array(position_at(index)))
    .collect();
  let mut random_state = 0x9E37_79B9_u64 ^ vertex_count as u64 ^ target as u64;

  // A single requested seed has no pair to initialize. Pick it randomly,
  // which is the natural degenerate form of the random tie rule.
  if target == 1 {
    return vec![(next_selection_random(&mut random_state) as usize % vertex_count) as u32];
  }

  // Find the farthest pair. Reservoir sampling makes exact distance ties
  // random without storing the O(n²) pairwise distance matrix.
  let mut first = 0usize;
  let mut second = 1usize;
  let mut best_distance = f32::NEG_INFINITY;
  let mut ties = 0_u64;
  for left in 0..vertex_count {
    for right in (left + 1)..vertex_count {
      let distance = positions[left].distance_squared(positions[right]);
      if !distance.is_finite() {
        continue;
      }
      if distance > best_distance {
        best_distance = distance;
        first = left;
        second = right;
        ties = 1;
      } else if distance == best_distance && replace_random_tie(&mut random_state, &mut ties) {
        first = left;
        second = right;
      }
    }
  }
  // Keep a valid result for malformed source data with no finite pair.
  if !best_distance.is_finite() {
    first = 0;
    second = 1;
  }

  let mut selected = vec![false; vertex_count];
  selected[first] = true;
  selected[second] = true;
  let mut seeds = vec![first as u32, second as u32];

  // `nearest_squared[i]` is the distance from candidate i to its closest
  // selected seed. Updating it after each promotion is equivalent to
  // recomputing δ(j) at every iteration, but uses O(n) memory and O(n*k)
  // work instead of storing the full pairwise distance matrix.
  let mut nearest_squared = vec![f32::INFINITY; vertex_count];
  for candidate in 0..vertex_count {
    if !selected[candidate] {
      nearest_squared[candidate] = positions[candidate]
        .distance_squared(positions[first])
        .min(positions[candidate].distance_squared(positions[second]));
    }
  }

  while seeds.len() < target {
    let mut next = None;
    let mut best_nearest = f32::NEG_INFINITY;
    let mut candidate_ties = 0_u64;
    for candidate in 0..vertex_count {
      if selected[candidate] {
        continue;
      }
      let distance = nearest_squared[candidate];
      if distance > best_nearest {
        best_nearest = distance;
        next = Some(candidate);
        candidate_ties = 1;
      } else if distance == best_nearest
        && replace_random_tie(&mut random_state, &mut candidate_ties)
      {
        next = Some(candidate);
      }
    }

    // Non-finite positions are still valid source vertices. If they cannot
    // participate in a distance comparison, promote the first remaining one
    // so the requested count is honored rather than panicking.
    let candidate = next.unwrap_or_else(|| {
      (0..vertex_count)
        .find(|&index| !selected[index])
        .expect("an unselected candidate exists before reaching target")
    });
    selected[candidate] = true;
    seeds.push(candidate as u32);

    for other in 0..vertex_count {
      if !selected[other] {
        nearest_squared[other] =
          nearest_squared[other].min(positions[other].distance_squared(positions[candidate]));
      }
    }
  }

  seeds
}

/// Configuration for extracting one complete speedline bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct MotionLineConfig {
  /// The stage that chooses source vertices before GPU tracing.
  pub seed_selection:    SeedSelectionAlgorithm,
  /// Uniform samples per second along the selected animation clip.
  pub frames_per_second: f32,
}

impl MotionLineConfig {
  /// Validates user-facing settings before any GPU buffer is allocated.
  pub fn validate(&self) -> crate::common::Result<()> {
    if !self.frames_per_second.is_finite() || self.frames_per_second <= 0.0 {
      crate::common::bail!("motion-line FPS must be finite and greater than zero")
    }
    if let SeedSelectionAlgorithm::RandomVertices { count }
    | SeedSelectionAlgorithm::UniformVertices { count }
    | SeedSelectionAlgorithm::UniformSpacetimeVertices { count, .. }
    | SeedSelectionAlgorithm::ImportanceSpacetimeVertices { count, .. }
    | SeedSelectionAlgorithm::ExtendedImportanceSpacetimeVertices { count, .. } =
      self.seed_selection
    {
      if count == 0 {
        crate::common::bail!("motion-line seed count must be greater than zero")
      }
    }
    if let Some((_, sampling_rate, _)) = self.seed_selection.spacetime_options() {
      if !sampling_rate.is_finite() || sampling_rate <= 0.0 {
        crate::common::bail!(
          "uniform spacetime seed sampling rate must be finite and greater than zero"
        )
      }
    }
    Ok(())
  }
}

/// A CPU-side description used while preparing GPU storage buffers.
///
/// `palettes` and `morph_weights` are laid out as sample-major arrays. The
/// compute shader uses the strides to locate the pose belonging to one
/// trajectory sample. `duration` is copied into the GPU parameter block so
/// the final sample remains the exact animation endpoint.
pub(crate) struct MotionLineSamples {
  pub seeds:          Vec<u32>,
  pub palettes:       Vec<crate::mesh::SkinTransform>,
  pub morph_weights:  Vec<f32>,
  pub duration:       f32,
  pub sample_count:   u32,
  pub palette_stride: u32,
  pub morph_stride:   u32,
}

/// Animation pose streams prepared for one uniform temporal sampling rate.
/// The spacetime selector uses this type for its short-lived low-rate pass;
/// the ordinary motion-line extraction uses it again at the requested final
/// line FPS.
pub(crate) struct PoseSamples {
  pub palettes:       Vec<crate::mesh::SkinTransform>,
  pub morph_weights:  Vec<f32>,
  pub duration:       f32,
  pub sample_count:   u32,
  pub palette_stride: u32,
  pub morph_stride:   u32,
}

/// Prepares only the sample-major pose data. Keeping this separate from seed
/// selection lets the spacetime algorithm reuse the same GPU skinning inputs
/// while keeping its temporary position output much smaller than full
/// trajectory records.
pub(crate) fn prepare_pose_samples(
  scene: &crate::scene::AnimatedScene,
  animation: usize,
  palette_stride: usize,
  morph_stride: usize,
  frames_per_second: f32,
) -> crate::common::Result<PoseSamples> {
  if !frames_per_second.is_finite() || frames_per_second <= 0.0 {
    crate::common::bail!("motion-line FPS must be finite and greater than zero")
  }
  let info =
    scene.animations().get(animation).cloned().ok_or_else(|| {
      crate::common::anyhow!("motion-line animation index {animation} is invalid")
    })?;

  // Include both endpoints. Clamping the final time avoids wrapping back to
  // the first key when duration * FPS is not an integer.
  let sample_count = ((info.duration.max(0.0) * frames_per_second).ceil() as usize)
    .saturating_add(1)
    .max(1);
  let morph_stride = morph_stride.max(1);
  let mut palettes = Vec::with_capacity(sample_count.saturating_mul(palette_stride));
  let mut morph_weights = Vec::with_capacity(sample_count.saturating_mul(morph_stride));
  let mut transforms = Vec::new();
  let mut frame_morph_weights = Vec::new();
  for sample in 0..sample_count {
    let time = if sample + 1 == sample_count {
      info.duration.max(0.0)
    } else {
      sample as f32 / frames_per_second
    };
    scene.update_gpu_pose(
      Some(animation),
      time,
      &mut transforms,
      &mut frame_morph_weights,
    );
    if transforms.len() != palette_stride {
      crate::common::bail!("motion-line pose changed its transform palette size")
    }
    palettes.extend_from_slice(&transforms);
    if frame_morph_weights.len() < morph_stride {
      frame_morph_weights.resize(morph_stride, 0.0);
    }
    morph_weights.extend_from_slice(&frame_morph_weights[..morph_stride]);
  }

  Ok(PoseSamples {
    palettes,
    morph_weights,
    duration: info.duration.max(0.0),
    sample_count: sample_count as u32,
    palette_stride: palette_stride as u32,
    morph_stride: morph_stride as u32,
  })
}

/// Builds the sample data for one selected animation.
pub(crate) fn prepare_samples(
  scene: &crate::scene::AnimatedScene,
  animation: usize,
  vertex_count: usize,
  position_at: impl Fn(usize) -> [f32; 3],
  palette_stride: usize,
  morph_stride: usize,
  config: &MotionLineConfig,
) -> crate::common::Result<MotionLineSamples> {
  config.validate()?;
  if matches!(
    config.seed_selection,
    SeedSelectionAlgorithm::UniformSpacetimeVertices { .. }
      | SeedSelectionAlgorithm::ImportanceSpacetimeVertices { .. }
      | SeedSelectionAlgorithm::ExtendedImportanceSpacetimeVertices { .. }
  ) {
    crate::common::bail!("spacetime seeds must be prepared by the GPU motion-line extraction stage")
  }
  let seeds = config
    .seed_selection
    .select_positions(vertex_count, position_at);
  if seeds.is_empty() {
    crate::common::bail!("motion-line seed selection produced no vertices")
  }

  let pose = prepare_pose_samples(
    scene,
    animation,
    palette_stride,
    morph_stride,
    config.frames_per_second,
  )?;

  Ok(MotionLineSamples {
    seeds,
    palettes: pose.palettes,
    morph_weights: pose.morph_weights,
    duration: pose.duration,
    sample_count: pose.sample_count,
    palette_stride: pose.palette_stride,
    morph_stride: pose.morph_stride,
  })
}

#[cfg(test)]
mod tests {
  use super::{SeedSelectionAlgorithm, Vec3};

  #[test]
  fn all_vertices_are_valid_source_indices() {
    assert_eq!(
      SeedSelectionAlgorithm::AllVertices.select(4),
      vec![0, 1, 2, 3]
    );
  }

  #[test]
  fn random_selection_is_unique_and_bounded() {
    let seeds = SeedSelectionAlgorithm::RandomVertices { count: 20 }.select(5);
    assert_eq!(seeds.len(), 5);
    assert!(seeds.iter().all(|&seed| seed < 5));
    let mut sorted = seeds.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), seeds.len());
  }

  #[test]
  fn uniform_selection_starts_with_a_farthest_pair() {
    let positions = [
      [0.0, 0.0, 0.0],
      [2.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [2.0, 1.0, 0.0],
    ];
    let seeds = SeedSelectionAlgorithm::UniformVertices { count: 3 }
      .select_positions(positions.len(), |index| positions[index]);
    assert_eq!(seeds.len(), 3);
    let first = positions[seeds[0] as usize];
    let second = positions[seeds[1] as usize];
    assert_eq!(
      Vec3::from_array(first).distance_squared(Vec3::from_array(second)),
      5.0
    );
  }

  #[test]
  fn uniform_selection_promotes_the_maximin_candidate() {
    let positions = [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [5.0, 0.0, 0.0],
      [9.0, 0.0, 0.0],
      [10.0, 0.0, 0.0],
    ];
    let seeds = SeedSelectionAlgorithm::UniformVertices { count: 3 }
      .select_positions(positions.len(), |index| positions[index]);
    assert_eq!(seeds[2], 2);
  }

  #[test]
  fn uniform_selection_handles_one_vertex_and_clamps_count() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let one = SeedSelectionAlgorithm::UniformVertices { count: 1 }
      .select_positions(positions.len(), |index| positions[index]);
    let all = SeedSelectionAlgorithm::UniformVertices { count: 20 }
      .select_positions(positions.len(), |index| positions[index]);
    assert_eq!(one.len(), 1);
    assert_eq!(all.len(), positions.len());
  }
}
