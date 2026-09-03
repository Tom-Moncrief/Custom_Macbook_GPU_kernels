#include <metal_stdlib>
using namespace metal;

constant uint CHANNELS = 64;

kernel void fused_cosine(device const char *input [[buffer(0)]],
                         device const float *query [[buffer(1)]],
                         device float *output [[buffer(2)]],
                         uint pixel [[thread_position_in_grid]]) {
    const uint base = pixel * CHANNELS;
    float dot = 0.0f;
    float norm2 = 0.0f;
    bool any_valid = false;
    for (uint c = 0; c < CHANNELS; ++c) {
        const char q = input[base + c];
        if (q == char(-128)) continue;
        const float raw = float(q);
        const float magnitude = fabs(raw) / 127.5f;
        const float value = copysign(magnitude * magnitude, raw);
        dot += value * query[c];
        norm2 += value * value;
        any_valid = true;
    }
    output[pixel] = any_valid && norm2 > 0.0f ? dot * rsqrt(norm2) : 0.0f;
}
