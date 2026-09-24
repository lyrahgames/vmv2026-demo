// Parallel greedy max-min selection over a temporary all-vertex spacetime
// trace.
//
// The old implementation put the complete O(vertices^2 * samples) farthest
// pair search and every greedy round into one invocation. That invocation
// could run for billions of iterations on a character mesh and trigger the
// GPU watchdog. The selector is now a sequence of short, bounded passes:
//
//   0. one workgroup per left-hand vertex finds its best right-hand pair;
//   1. one invocation reduces those pair scores and writes the first seeds;
//   2. every candidate initializes its distance-to-the-selected-set stream;
//   3. one invocation selects the next candidate (argmax or importance draw);
//   4. every candidate updates that stream for the newly selected seed.
//
// The reduction passes are O(vertices), while the expensive distance work is
// distributed over the candidate invocations. `candidate_nearest` stores one
// value per candidate and sampled time, so later rounds do not rescan all old
// seeds. Spacing distances are squared; this preserves every min/max
// comparison and avoids an unnecessary square root. The extended mode uses
// one square root per consecutive trajectory sample for travelled distance.

var<workgroup> pair_tile_scores: array<f32, 64>;
var<workgroup> pair_tile_rights: array<u32, 64>;

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
  importance: u32,
  stochastic: u32,
  extended: u32,
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

// The shared uniform selector stores phi^2 to preserve argmax ordering. An
// importance distribution must use the Euclidean phi itself, so convert only
// at normalization time without repeating any position-distance work.
fn spacing_score(candidate: u32) -> f32 {
  return sqrt(max(candidate_scores[candidate], 0.0));
}

// A deterministic hash is used for exact-distance ties. It provides stable
// native/browser output while retaining the intended arbitrary tie choice.
fn tie_hash(a: u32, b: u32, sample: u32, round: u32) -> u32 {
  var value = 0x9e3779b9u ^ (a * 0x85ebca6bu) ^ (b * 0xc2b2ae35u);
  value = value ^ (sample * 0x27d4eb2du) ^ (round * 0x165667b1u);
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

// A reproducible [0, 1) variate for stochastic importance selection. The
// selector remains deterministic for a fixed scene/configuration while still
// sampling the requested probability distribution.
fn random_unit(round: u32) -> f32 {
  let value = tie_hash(params.vertex_count, params.seed_count, params.sample_count, round);
  return (f32(value) + 0.5) / 4294967296.0;
}

// Stage 0: find the best partner for one left-hand vertex. The 64 lanes of a
// workgroup split the right-hand scan, so no single invocation has to execute
// the complete O(vertices * samples) inner loop. `selected_count` carries the
// x dimension of this pass; it is otherwise unused by stage 0.
fn find_pair(left: u32, lane: u32) {
  let valid = left < params.vertex_count;
  var best_score = -1.0;
  var best_right = left;
  if (valid) {
    for (var right = left + 1u + lane; right < params.vertex_count; right = right + 64u) {
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
  }
  pair_tile_scores[lane] = best_score;
  pair_tile_rights[lane] = best_right;
  workgroupBarrier();
  if (lane == 0u && valid) {
    for (var other = 1u; other < 64u; other = other + 1u) {
      let score = pair_tile_scores[other];
      let right = pair_tile_rights[other];
      if (score > best_score ||
          (score == best_score && tie_hash(left, right, 0u, 0u) <
            tie_hash(left, best_right, 0u, 0u))) {
        best_score = score;
        best_right = right;
      }
    }
    pair_scores[left] = PairScore(best_score, best_right, 0u, 0u);
  }
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
// The optional travelled distance is accumulated in the score field of the
// pair scratch record after the initial-pair pass has finished. Reusing that
// existing buffer avoids another per-candidate allocation.
fn initialize_candidate(candidate: u32) {
  if (candidate >= params.vertex_count) {
    return;
  }
  var score = -1.0;
  var travelled = 0.0;
  var previous = vec3<f32>(0.0);
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    let current = position(candidate, sample);
    var nearest = distance_squared(current, position(seeds[0], sample));
    if (params.seed_count > 1u) {
      nearest = min(nearest,
        distance_squared(current, position(seeds[1], sample)));
    }
    if (params.extended != 0u && sample > 0u) {
      travelled = travelled + sqrt(distance_squared(previous, current));
    }
    candidate_nearest[sample * params.vertex_count + candidate] = nearest;
    score = max(score, nearest);
    previous = current;
  }
  if (params.extended != 0u) {
    pair_scores[candidate].score = travelled;
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

// Stage 3: promote the next candidate. Uniform mode uses the largest
// max-over-time nearest distance; importance modes normalize that same score
// (and optionally travelled distance) before choosing. A selected candidate
// is ignored, and this pass runs only after the previous score update.
fn select_candidate() {
  if (params.selected_count >= params.seed_count || params.vertex_count == 0u) {
    return;
  }
  var best_candidate = 0u;
  var best_score = -1.0;

  // Preserve the original uniform selector exactly: it selects the maximum
  // max-over-time nearest distance and uses the existing deterministic tie
  // rule. Importance modes use the normalized probabilities below.
  if (params.importance == 0u) {
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
  } else {
    var phi_sum = 0.0;
    var gamma_sum = 0.0;
    var available_count = 0u;
    for (var candidate = 0u; candidate < params.vertex_count; candidate = candidate + 1u) {
      if (candidate_scores[candidate] < 0.0) {
        continue;
      }
      phi_sum = phi_sum + spacing_score(candidate);
      if (params.extended != 0u) {
        gamma_sum = gamma_sum + pair_scores[candidate].score;
      }
      available_count = available_count + 1u;
    }

    let count = max(f32(available_count), 1.0);
    var combined_sum = 0.0;
    for (var candidate = 0u; candidate < params.vertex_count; candidate = candidate + 1u) {
      if (candidate_scores[candidate] < 0.0) {
        continue;
      }
      var p_phi = spacing_score(candidate) / phi_sum;
      if (!(phi_sum > 0.0)) {
        // All spacing scores are zero: use a uniform fallback rather than
        // producing NaNs or starving every candidate.
        p_phi = 1.0 / count;
      }
      var weight = p_phi;
      if (params.extended != 0u) {
        var p_gamma = pair_scores[candidate].score / gamma_sum;
        if (!(gamma_sum > 0.0)) {
          // A rigid/static animation should reduce to ordinary importance
          // selection instead of making every combined probability zero.
          p_gamma = 1.0 / count;
        }
        weight = p_phi * p_gamma;
      }
      combined_sum = combined_sum + weight;
    }

    var random_threshold = 0.0;
    if (params.stochastic != 0u) {
      random_threshold = random_unit(params.selected_count);
    }
    var cumulative = 0.0;
    var found_stochastic = false;
    var stochastic_candidate = 0u;
    var best_probability = -1.0;
    for (var candidate = 0u; candidate < params.vertex_count; candidate = candidate + 1u) {
      if (candidate_scores[candidate] < 0.0) {
        continue;
      }
      var p_phi = spacing_score(candidate) / phi_sum;
      if (!(phi_sum > 0.0)) {
        p_phi = 1.0 / count;
      }
      var weight = p_phi;
      if (params.extended != 0u) {
        var p_gamma = pair_scores[candidate].score / gamma_sum;
        if (!(gamma_sum > 0.0)) {
          p_gamma = 1.0 / count;
        }
        weight = p_phi * p_gamma;
      }
      var probability = weight / combined_sum;
      if (!(combined_sum > 0.0)) {
        probability = 1.0 / count;
      }

      if (params.stochastic != 0u) {
        stochastic_candidate = candidate;
        cumulative = cumulative + probability;
        if (!found_stochastic && random_threshold < cumulative) {
          best_candidate = candidate;
          found_stochastic = true;
        }
      } else if (probability > best_probability ||
          (probability == best_probability && tie_hash(candidate, best_candidate, 0u,
            params.selected_count) < tie_hash(best_candidate, candidate, 0u,
            params.selected_count))) {
        best_probability = probability;
        best_candidate = candidate;
      }
    }
    if (params.stochastic != 0u && !found_stochastic) {
      // Floating-point accumulation can finish just below one. The final
      // available candidate is a safe and deterministic fallback.
      best_candidate = stochastic_candidate;
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
fn select(
  @builtin(global_invocation_id) invocation: vec3<u32>,
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  if (params.vertex_count == 0u || params.seed_count == 0u) {
    return;
  }

  switch params.stage {
    case 0u: {
      find_pair(workgroup.x + workgroup.y * params.selected_count, local.x);
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
