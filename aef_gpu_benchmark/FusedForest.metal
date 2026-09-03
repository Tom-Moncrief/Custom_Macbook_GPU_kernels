#include <metal_stdlib>
using namespace metal;

constant uint CHANNELS = 64;

kernel void forest_regression(device const char *features [[buffer(0)]],
                              device const int *tree_offsets [[buffer(1)]],
                              device const short *feature_index [[buffer(2)]],
                              device const short *threshold_q [[buffer(3)]],
                              device const int *left_child [[buffer(4)]],
                              device const int *right_child [[buffer(5)]],
                              device const float *leaf_value [[buffer(6)]],
                              device float *output [[buffer(7)]],
                              constant uint &tree_count [[buffer(8)]],
                              uint pixel [[thread_position_in_grid]]) {
    const uint base = pixel * CHANNELS;
    float sum = 0.0f;
    for (uint tree = 0; tree < tree_count; ++tree) {
        int node = tree_offsets[tree];
        while (left_child[node] >= 0) {
            const short feature = feature_index[node];
            const char raw = features[base + uint(feature)];
            const short value = raw == char(-128) ? short(0) : short(raw);
            node = value <= threshold_q[node] ? left_child[node] : right_child[node];
        }
        sum += leaf_value[node];
    }
    output[pixel] = sum / float(tree_count);
}

kernel void forest_binary_classification(device const char *features [[buffer(0)]],
                                         device const int *tree_offsets [[buffer(1)]],
                                         device const short *feature_index [[buffer(2)]],
                                         device const short *threshold_q [[buffer(3)]],
                                         device const int *left_child [[buffer(4)]],
                                         device const int *right_child [[buffer(5)]],
                                         device const float *leaf_value [[buffer(6)]],
                                         device float *output [[buffer(7)]],
                                         constant uint &tree_count [[buffer(8)]],
                                         uint pixel [[thread_position_in_grid]]) {
    const uint base = pixel * CHANNELS;
    float positive_votes = 0.0f;
    for (uint tree = 0; tree < tree_count; ++tree) {
        int node = tree_offsets[tree];
        while (left_child[node] >= 0) {
            const short feature = feature_index[node];
            const char raw = features[base + uint(feature)];
            const short value = raw == char(-128) ? short(0) : short(raw);
            node = value <= threshold_q[node] ? left_child[node] : right_child[node];
        }
        positive_votes += leaf_value[node];
    }
    output[pixel] = positive_votes / float(tree_count);
}

kernel void forest_partial(device const char *features [[buffer(0)]],
                           device const int *tree_offsets [[buffer(1)]],
                           device const short *feature_index [[buffer(2)]],
                           device const short *threshold_q [[buffer(3)]],
                           device const int *left_child [[buffer(4)]],
                           device const int *right_child [[buffer(5)]],
                           device const float *leaf_value [[buffer(6)]],
                           device float *partial [[buffer(7)]],
                           constant uint &tree_start [[buffer(8)]],
                           constant uint &trees_in_chunk [[buffer(9)]],
                           constant uint &tree_count [[buffer(10)]],
                           uint job [[thread_position_in_grid]]) {
    const uint pixel = job / trees_in_chunk;
    const uint local_tree = job % trees_in_chunk;
    const uint tree = tree_start + local_tree;
    if (tree >= tree_count) { partial[job] = 0.0f; return; }
    const uint base = pixel * CHANNELS;
    int node = tree_offsets[tree];
    while (left_child[node] >= 0) {
        const short feature = feature_index[node];
        const char raw = features[base + uint(feature)];
        const short value = raw == char(-128) ? short(0) : short(raw);
        node = value <= threshold_q[node] ? left_child[node] : right_child[node];
    }
    partial[job] = leaf_value[node];
}

kernel void forest_accumulate(device const float *partial [[buffer(0)]],
                              device float *output [[buffer(1)]],
                              constant uint &trees_in_chunk [[buffer(2)]],
                              constant uint &tree_count [[buffer(3)]],
                              uint pixel [[thread_position_in_grid]]) {
    float sum = 0.0f;
    for (uint t = 0; t < trees_in_chunk; ++t) sum += partial[pixel * trees_in_chunk + t];
    output[pixel] += sum / float(tree_count);
}

kernel void forest_partial_packed(device const char *features [[buffer(0)]],
                                  device const int *tree_offsets [[buffer(1)]],
                                  device const uchar *feature_index [[buffer(2)]],
                                  device const char *threshold_q [[buffer(3)]],
                                  device const int *left_child [[buffer(4)]],
                                  device const int *right_child [[buffer(5)]],
                                  device const float *leaf_value [[buffer(6)]],
                                  device float *partial [[buffer(7)]],
                                  constant uint &tree_start [[buffer(8)]],
                                  constant uint &trees_in_chunk [[buffer(9)]],
                                  constant uint &tree_count [[buffer(10)]],
                                  uint job [[thread_position_in_grid]]) {
    const uint pixel = job / trees_in_chunk;
    const uint local_tree = job % trees_in_chunk;
    const uint tree = tree_start + local_tree;
    if (tree >= tree_count) { partial[job] = 0.0f; return; }
    const uint base = pixel * CHANNELS;
    int node = tree_offsets[tree];
    while (left_child[node] >= 0) {
        const uint feature = uint(feature_index[node]);
        const char raw = features[base + feature];
        const short value = raw == char(-128) ? short(0) : short(raw);
        node = value <= short(threshold_q[node]) ? left_child[node] : right_child[node];
    }
    partial[job] = leaf_value[node];
}

kernel void forest_partial_packed_cached(device const char *features [[buffer(0)]],
                                         device const int *tree_offsets [[buffer(1)]],
                                         device const uchar *feature_index [[buffer(2)]],
                                         device const char *threshold_q [[buffer(3)]],
                                         device const int *left_child [[buffer(4)]],
                                         device const int *right_child [[buffer(5)]],
                                         device const float *leaf_value [[buffer(6)]],
                                         device float *partial [[buffer(7)]],
                                         constant uint &tree_start [[buffer(8)]],
                                         constant uint &trees_in_chunk [[buffer(9)]],
                                         constant uint &tree_count [[buffer(10)]],
                                         uint gid [[thread_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint threads_per_group [[threads_per_threadgroup]]) {
    threadgroup char tile_features[4096];
    const uint job = gid;
    const uint group = gid / threads_per_group;
    const uint pixels_per_group = threads_per_group / trees_in_chunk;
    const uint group_pixel = group * pixels_per_group;
    for (uint i = tid; i < pixels_per_group * CHANNELS; i += threads_per_group) {
        const uint pixel = group_pixel + i / CHANNELS;
        const char raw = features[pixel * CHANNELS + (i % CHANNELS)];
        tile_features[i] = raw == char(-128) ? char(0) : raw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pixel = job / trees_in_chunk;
    const uint local_tree = job % trees_in_chunk;
    const uint tree = tree_start + local_tree;
    if (tree >= tree_count) { partial[job] = 0.0f; return; }
    const uint local_pixel = pixel - group_pixel;
    int node = tree_offsets[tree];
    while (left_child[node] >= 0) {
        const uint feature = uint(feature_index[node]);
        node = tile_features[local_pixel * CHANNELS + feature] <= threshold_q[node] ? left_child[node] : right_child[node];
    }
    partial[job] = leaf_value[node];
}
