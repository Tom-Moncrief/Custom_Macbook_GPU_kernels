import Foundation
import Metal

guard CommandLine.arguments.count == 3 else {
    fputs("usage: benchmark.swift input.raw output.f32\n", stderr); exit(2)
}
let inputPath = CommandLine.arguments[1]
let outputPath = CommandLine.arguments[2]
let sourcePath = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("Dequantize.metal")
let source = try String(contentsOf: sourcePath, encoding: .utf8)
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal device unavailable") }
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try queue.device.makeComputePipelineState(function: library.makeFunction(name: "dequantize")!)
let input = try Data(contentsOf: URL(fileURLWithPath: inputPath))
let inputBuffer = device.makeBuffer(bytes: [UInt8](input), length: input.count, options: .storageModeShared)!
let outputBuffer = device.makeBuffer(length: input.count * MemoryLayout<Float>.size, options: .storageModeShared)!
let n = input.count
let grid = MTLSize(width: n, height: 1, depth: 1)
let threads = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)

for _ in 0..<2 {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline); enc.setBuffer(inputBuffer, offset: 0, index: 0); enc.setBuffer(outputBuffer, offset: 0, index: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: threads); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}

var times: [Double] = []
for _ in 0..<5 {
    let start = DispatchTime.now().uptimeNanoseconds
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline); enc.setBuffer(inputBuffer, offset: 0, index: 0); enc.setBuffer(outputBuffer, offset: 0, index: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: threads); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0)
}
let output = Data(bytes: outputBuffer.contents(), count: input.count * MemoryLayout<Float>.size)
try output.write(to: URL(fileURLWithPath: outputPath))
let median = times.sorted()[times.count / 2]
print("{\"device\":\"\(device.name)\",\"bytes\":\(n),\"gpu_seconds\":\(times),\"gpu_median_seconds\":\(median)}")
