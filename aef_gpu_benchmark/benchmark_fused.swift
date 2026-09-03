import Foundation
import Metal

guard CommandLine.arguments.count == 4 else { fputs("usage: benchmark_fused.swift input.raw query.f32 output.f32\n", stderr); exit(2) }
let input = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
let query = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))
let outputPath = CommandLine.arguments[3]
let sourceURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("FusedCosine.metal")
let source = try String(contentsOf: sourceURL, encoding: .utf8)
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "fused_cosine")!)!
let inputBuffer = device.makeBuffer(bytes: [UInt8](input), length: input.count, options: .storageModeShared)!
let queryBuffer = device.makeBuffer(bytes: [UInt8](query), length: query.count, options: .storageModeShared)!
let pixelCount = input.count / 64
let outputBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.size, options: .storageModeShared)!
let grid = MTLSize(width: pixelCount, height: 1, depth: 1)
let threads = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)

func run() -> Double {
    let start = DispatchTime.now().uptimeNanoseconds
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline); enc.setBuffer(inputBuffer, offset: 0, index: 0); enc.setBuffer(queryBuffer, offset: 0, index: 1); enc.setBuffer(outputBuffer, offset: 0, index: 2)
    enc.dispatchThreads(grid, threadsPerThreadgroup: threads); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
}
for _ in 0..<2 { _ = run() }
var times: [Double] = []
for _ in 0..<5 { times.append(run()) }
let output = Data(bytes: outputBuffer.contents(), count: pixelCount * MemoryLayout<Float>.size)
try output.write(to: URL(fileURLWithPath: outputPath))
print("{\"device\":\"\(device.name)\",\"pixels\":\(pixelCount),\"gpu_seconds\":\(times),\"gpu_median_seconds\":\(times.sorted()[times.count / 2])}")
