// Parallel greedy max-min selection over a temporary all-vertex spacetime
// trace.
//
// The old implementation put the complete O(vertices^2 * samples) farthest
// pair search and every greedy round into one invocation. That invocation
// could run for billions of iterations on a character mesh and trigger the
// GPU watchdog. The selector is now a sequence of short, bounded passes:
//
//   0. one invocation per left-hand vertex finds its best right-hand pair;
//   1. one invocation reduces those pair scores and writes the first seeds;
//   2. every candidate initializes its distance-to-the-selected-set stream;
//   3. one invocation selects the next candidate;
//   4. every candidate updates that stream for the newly selected seed.
//
// The reduction passes are O(vertices), while the expensive distance work is
// distributed over the candidate invocations. `candidate_nearest` stores one
// value per candidate and sampled time, so later rounds do not rescan all old
// seeds. Distances are squared; this preserves every min/max comparison and
// avoids an unnecessary square root.

struct PairScore {
  score: f32,
  right: u32,
  _padding0: u32,
  _padding1: u32,
};

struct SelectionParams {
  vertex_count: u32,
  sample_count: u32,
  seed_count: u32,
  stage: u32,
  selected_count: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> pair_scores: array<PairScore>;
@group(0) @binding(2) var<storage, read_write> candidate_nearest: array<f32>;
@group(0) @binding(3) var<storage, read_write> candidate_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> seeds: array<u32>;
@group(0) @binding(5) var<uniform> params: SelectionParams;

fn position(vertex: u32, sample: u32) -> vec3<f32> {
  return positions[sample * params.vertex_count + vertex].xyz;
}

fn distance_squared(first: vec3<f32>, second: vec3<f32>) -> f32 {
  let delta = first - second;
  return dot(delta, delta);
}

// A deterministic hash is used for exact-distance ties. It provides stable
// native/browser output while retaining the intended arbitrary tie choice.
fn tie_hash(a: u32, b: u32, sample: u32, round: u32) -> u32 {
  var value = 0x9e3779b9u ^ a * 0x85ebca6bu ^ b * 0xc2b2ae35u;
  value = value ^ sample * 0x27d4eb2du ^ round * 0x165667b1u;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

// Stage 0: find the best partner for one left-hand vertex. Only the best
// partner is retained, reducing the scratch buffer to one record per vertex.
fn find_pair(left: u32) {
  if (left >= params.vertex_count) {
    return;
  }
  var best_score = -1.0;
  var best_right = left;
  for (var right = left + 1u; right < params.vertex_count; right = right + 1u) {
    var score = 0.0;
    for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
      score = max(score, distance_squared(position(left, sample), position(right, sample)));
    }
    if (score > best_score ||
        (score == best_score && tie_hash(left, right, 0u, 0u) <
          tie_hash(left, best_right, 0u, 0u))) {
      best_score = score;
      best_right = right;
    }
  }
  pair_scores[left] = PairScore(best_score, best_right, 0u, 0u);
}

// Stage 1: reduce the per-left pair winners and write the initial pair.
fn select_pair() {
  if (params.vertex_count == 1u) {
    seeds[0] = 0u;
    return;
  }
  var best_left = 0u;
  var best_right = 1u;
  var best_score = -1.0;
  for (var left = 0u; left + 1u < params.vertex_count; left = left + 1u) {
    let pair = pair_scores[left];
    if (pair.score > best_score ||
        (pair.score == best_score &&
          tie_hash(left, pair.right, 0u, 0u) <
            tie_hash(best_left, best_right, 0u, 0u))) {
      best_score = pair.score;
      best_left = left;
      best_right = pair.right;
    }
  }
  seeds[0] = best_left;
  if (params.seed_count > 1u) {
    seeds[1] = best_right;
  }
}

// Stage 2: initialize the nearest-selected-seed distance for every candidate
// and reduce it over time into the score used by the next selection pass.
fn initialize_candidate(candidate: u32) {
  if (candidate >= params.vertex_count) {
    return;
  }
  var score = -1.0;
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    var nearest = distance_squared(position(candidate, sample), position(seeds[0], sample));
    if (params.seed_count > 1u) {
      nearest = min(nearest,
        distance_squared(position(candidate, sample), position(seeds[1], sample)));
    }
    candidate_nearest[sample * params.vertex_count + candidate] = nearest;
    score = max(score, nearest);
  }
  // A negative score is the selected-candidate marker. All valid squared
  // distances are non-negative, so this avoids scanning the seed list in
  // every later reduction pass.
  if (candidate == seeds[0] || (params.seed_count > 1u && candidate == seeds[1])) {
    candidate_scores[candidate] = -1.0;
  } else {
    candidate_scores[candidate] = score;
  }
}

// Stage 3: promote the candidate with the largest max-over-time nearest
// distance. A selected candidate is ignored, and the shader is dispatched
// only after the previous pass has written the current score buffer.
fn select_candidate() {
  if (params.selected_count >= params.seed_count || params.vertex_count == 0u) {
    return;
  }
  var best_candidate = 0u;
  var best_score = -1.0;
  for (var candidate = 0u; candidate < params.vertex_count; candidate = candidate + 1u) {
    if (candidate_scores[candidate] < 0.0) {
      continue;
    }
    let score = candidate_scores[candidate];
    if (score > best_score ||
        (score == best_score && tie_hash(candidate, best_candidate, 0u,
          params.selected_count) < tie_hash(best_candidate, candidate, 0u,
          params.selected_count))) {
      best_score = score;
      best_candidate = candidate;
    }
  }
  candidate_scores[best_candidate] = -1.0;
  seeds[params.selected_count] = best_candidate;
}

// Stage 4: add the newly selected seed to every candidate's nearest-distance
// stream. This is the incremental form of min(delta(j,t)); it avoids the
// previous O(vertices * samples * selected-seeds) loop in every round.
fn update_candidate(candidate: u32) {
  if (candidate >= params.vertex_count) {
    return;
  }
  // Keep all already selected candidates disabled. This single score test
  // replaces the O(selected_count) membership scan in every invocation.
  if (candidate_scores[candidate] < 0.0) {
    return;
  }
  let selected = seeds[params.selected_count];
  var score = -1.0;
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    let offset = sample * params.vertex_count + candidate;
    let nearest = min(candidate_nearest[offset],
      distance_squared(position(candidate, sample), position(selected, sample)));
    candidate_nearest[offset] = nearest;
    score = max(score, nearest);
  }
  candidate_scores[candidate] = score;
}

@compute @workgroup_size(64, 1, 1)
fn select(@builtin(global_invocation_id) invocation: vec3<u32>) {
  if (params.vertex_count == 0u || params.seed_count == 0u) {
    return;
  }

  switch params.stage {
    case 0u: {
      find_pair(invocation.x);
    }
    case 1u: {
      if (invocation.x == 0u) {
        if (params.seed_count == 1u) {
          seeds[0] = tie_hash(params.vertex_count, params.sample_count, 0u, 0u) %
            params.vertex_count;
        } else {
          select_pair();
        }
      }
    }
    case 2u: {
      initialize_candidate(invocation.x);
    }
    case 3u: {
      if (invocation.x == 0u) {
        select_candidate();
      }
    }
    case 4u: {
      update_candidate(invocation.x);
    }
    default: {}
  }
}
