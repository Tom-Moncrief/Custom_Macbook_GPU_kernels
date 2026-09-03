import Foundation
import Metal

guard CommandLine.arguments.count == 4 else { fputs("usage: benchmark_forest.swift model-prefix input.raw output.f32\n", stderr); exit(2) }
let prefix = CommandLine.arguments[1]
let input = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))
let outputPath = CommandLine.arguments[3]
let modelName = URL(fileURLWithPath: prefix).lastPathComponent
let mode = modelName.contains("classification") ? "forest_binary_classification" : "forest_regression"
let dir = URL(fileURLWithPath: prefix).deletingLastPathComponent().path
func load(_ suffix: String) throws -> Data { try Data(contentsOf: URL(fileURLWithPath: dir + "/" + modelName + suffix)) }
let offsets = try load(".offsets.i32")
let features = try load(".features.i16")
let thresholds = try load(".thresholds.i16")
let left = try load(".left.i32")
let right = try load(".right.i32")
let leaves = try load(".leaves.f32")
let meta = try String(contentsOfFile: dir + "/" + modelName + ".json", encoding: .utf8)
let treeCount = UInt32(meta.components(separatedBy: "tree_count")[1].split(separator: ":")[1].split(separator: ",")[0].trimmingCharacters(in: .whitespacesAndNewlines))!
let sourceURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("FusedForest.metal")
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(source: String(contentsOf: sourceURL, encoding: .utf8), options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: mode)!)
func buffer(_ data: Data) -> MTLBuffer { device.makeBuffer(bytes: [UInt8](data), length: data.count, options: .storageModeShared)! }
let inputBuffer = buffer(input); let offsetsBuffer = buffer(offsets); let featuresBuffer = buffer(features); let thresholdsBuffer = buffer(thresholds); let leftBuffer = buffer(left); let rightBuffer = buffer(right); let leavesBuffer = buffer(leaves)
let outputBuffer = device.makeBuffer(length: (input.count / 64) * MemoryLayout<Float>.size, options: .storageModeShared)!
let countBuffer = device.makeBuffer(bytes: [treeCount], length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
let grid = MTLSize(width: input.count / 64, height: 1, depth: 1)
let threads = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
for _ in 0..<2 {
    let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!; enc.setComputePipelineState(pipeline); enc.setBuffer(inputBuffer, offset: 0, index: 0); enc.setBuffer(offsetsBuffer, offset: 0, index: 1); enc.setBuffer(featuresBuffer, offset: 0, index: 2); enc.setBuffer(thresholdsBuffer, offset: 0, index: 3); enc.setBuffer(leftBuffer, offset: 0, index: 4); enc.setBuffer(rightBuffer, offset: 0, index: 5); enc.setBuffer(leavesBuffer, offset: 0, index: 6); enc.setBuffer(outputBuffer, offset: 0, index: 7); enc.setBuffer(countBuffer, offset: 0, index: 8); enc.dispatchThreads(grid, threadsPerThreadgroup: threads); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}
var times: [Double] = []
for _ in 0..<5 {
    let start = DispatchTime.now().uptimeNanoseconds
    let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!; enc.setComputePipelineState(pipeline); enc.setBuffer(inputBuffer, offset: 0, index: 0); enc.setBuffer(offsetsBuffer, offset: 0, index: 1); enc.setBuffer(featuresBuffer, offset: 0, index: 2); enc.setBuffer(thresholdsBuffer, offset: 0, index: 3); enc.setBuffer(leftBuffer, offset: 0, index: 4); enc.setBuffer(rightBuffer, offset: 0, index: 5); enc.setBuffer(leavesBuffer, offset: 0, index: 6); enc.setBuffer(outputBuffer, offset: 0, index: 7); enc.setBuffer(countBuffer, offset: 0, index: 8); enc.dispatchThreads(grid, threadsPerThreadgroup: threads); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted(); times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0)
}
try Data(bytes: outputBuffer.contents(), count: (input.count / 64) * MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: outputPath))
print("{\"device\":\"\(device.name)\",\"mode\":\"\(mode)\",\"pixels\":\(input.count / 64),\"trees\":\(treeCount),\"gpu_median_seconds\":\(times.sorted()[2]),\"gpu_seconds\":\(times)}")
