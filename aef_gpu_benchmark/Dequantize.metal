#include <metal_stdlib>
using namespace metal;

kernel void dequantize(device const char *input [[buffer(0)]],
                       device float *output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    const char q = input[index];
    if (q == char(-128)) {
        output[index] = 0.0f;
        return;
    }
    const float x = float(q);
    const float magnitude = fabs(x) / 127.5f;
    output[index] = copysign(magnitude * magnitude, x);
}
