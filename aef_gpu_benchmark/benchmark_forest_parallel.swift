import Foundation
import Metal

guard CommandLine.arguments.count >= 4 && CommandLine.arguments.count <= 7 else { fputs("usage: benchmark_forest_parallel.swift model-prefix input.raw output.f32 [chunk-trees] [packed] [cached]\n", stderr); exit(2) }
let prefix = CommandLine.arguments[1]; let input = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])); let outputPath = CommandLine.arguments[3]
let modelName = URL(fileURLWithPath: prefix).lastPathComponent; let dir = URL(fileURLWithPath: prefix).deletingLastPathComponent().path
func load(_ suffix: String) throws -> Data { try Data(contentsOf: URL(fileURLWithPath: dir + "/" + modelName + suffix)) }
let packed = CommandLine.arguments.contains("packed")
let cached = CommandLine.arguments.contains("cached")
let offsets = try load(".offsets.i32"), features = try load(packed ? ".features.u8" : ".features.i16"), thresholds = try load(packed ? ".thresholds.i8" : ".thresholds.i16"), left = try load(".left.i32"), right = try load(".right.i32"), leaves = try load(".leaves.f32")
let meta = try String(contentsOfFile: dir + "/" + modelName + ".json", encoding: .utf8)
let treeCount = UInt32(meta.components(separatedBy: "tree_count")[1].split(separator: ":")[1].split(separator: ",")[0].trimmingCharacters(in: .whitespacesAndNewlines))!
let sourceURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("FusedForest.metal")
let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
let library = try device.makeLibrary(source: String(contentsOf: sourceURL, encoding: .utf8), options: nil)
let partialName = cached ? "forest_partial_packed_cached" : (packed ? "forest_partial_packed" : "forest_partial")
let partialPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: partialName)!)
let accumulatePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "forest_accumulate")!)
func buffer(_ data: Data) -> MTLBuffer { device.makeBuffer(bytes: [UInt8](data), length: data.count, options: .storageModeShared)! }
let inputBuffer = buffer(input), offsetsBuffer = buffer(offsets), featuresBuffer = buffer(features), thresholdsBuffer = buffer(thresholds), leftBuffer = buffer(left), rightBuffer = buffer(right), leavesBuffer = buffer(leaves)
let pixelCount = input.count / 64
let chunkTrees: UInt32 = CommandLine.arguments.dropFirst(4).compactMap { UInt32($0) }.first ?? 4
let partialBuffer = device.makeBuffer(length: pixelCount * Int(chunkTrees) * MemoryLayout<Float>.size, options: .storageModeShared)!
let outputBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.size, options: .storageModeShared)!
let threads = MTLSize(width: min(partialPipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
let pixelGrid = MTLSize(width: pixelCount, height: 1, depth: 1)

func run() -> Double {
    outputBuffer.contents().initializeMemory(as: Float.self, repeating: 0, count: pixelCount)
    let start = DispatchTime.now().uptimeNanoseconds
    let cb = queue.makeCommandBuffer()!
    var treeStart: UInt32 = 0
    while treeStart < treeCount {
        let remaining = treeCount - treeStart; let inChunk = min(chunkTrees, remaining)
        var ts = treeStart, ic = inChunk, tc = treeCount
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(partialPipeline); enc.setBuffer(inputBuffer, offset: 0, index: 0); enc.setBuffer(offsetsBuffer, offset: 0, index: 1); enc.setBuffer(featuresBuffer, offset: 0, index: 2); enc.setBuffer(thresholdsBuffer, offset: 0, index: 3); enc.setBuffer(leftBuffer, offset: 0, index: 4); enc.setBuffer(rightBuffer, offset: 0, index: 5); enc.setBuffer(leavesBuffer, offset: 0, index: 6); enc.setBuffer(partialBuffer, offset: 0, index: 7); enc.setBytes(&ts, length: 4, index: 8); enc.setBytes(&ic, length: 4, index: 9); enc.setBytes(&tc, length: 4, index: 10); enc.dispatchThreads(MTLSize(width: pixelCount * Int(inChunk), height: 1, depth: 1), threadsPerThreadgroup: threads); enc.endEncoding()
        let red = cb.makeComputeCommandEncoder()!
        red.setComputePipelineState(accumulatePipeline); red.setBuffer(partialBuffer, offset: 0, index: 0); red.setBuffer(outputBuffer, offset: 0, index: 1); red.setBytes(&ic, length: 4, index: 2); red.setBytes(&tc, length: 4, index: 3); red.dispatchThreads(pixelGrid, threadsPerThreadgroup: threads); red.endEncoding()
        treeStart += inChunk
    }
    cb.commit(); cb.waitUntilCompleted()
    return Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
}
for _ in 0..<2 { _ = run() }
var times: [Double] = []; for _ in 0..<5 { times.append(run()) }
try Data(bytes: outputBuffer.contents(), count: pixelCount * 4).write(to: URL(fileURLWithPath: outputPath))
print("{\"device\":\"\(device.name)\",\"pixels\":\(pixelCount),\"trees\":\(treeCount),\"chunk_trees\":\(chunkTrees),\"packed\":\(packed),\"cached\":\(cached),\"gpu_median_seconds\":\(times.sorted()[2]),\"gpu_seconds\":\(times)}")
