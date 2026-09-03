import Foundation
import Metal

let sourceURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("FusedCosine.metal")
let source = try String(contentsOf: sourceURL, encoding: .utf8)
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "fused_cosine")!)
let threads = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)

func signal(_ message: String) {
    FileHandle.standardOutput.write(Data((message + "\n").utf8))
}

signal("READY")
while let line = readLine() {
    if line == "QUIT" { break }
    let fields = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
    guard fields.count == 3 else { signal("ERROR\tbad command"); continue }
    let input = try Data(contentsOf: URL(fileURLWithPath: fields[0]))
    let query = try Data(contentsOf: URL(fileURLWithPath: fields[1]))
    let inputBuffer = device.makeBuffer(bytes: [UInt8](input), length: input.count, options: .storageModeShared)!
    let queryBuffer = device.makeBuffer(bytes: [UInt8](query), length: query.count, options: .storageModeShared)!
    let pixelCount = input.count / 64
    let outputBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.size, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(inputBuffer, offset: 0, index: 0)
    enc.setBuffer(queryBuffer, offset: 0, index: 1)
    enc.setBuffer(outputBuffer, offset: 0, index: 2)
    enc.dispatchThreads(MTLSize(width: pixelCount, height: 1, depth: 1), threadsPerThreadgroup: threads)
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let output = Data(bytes: outputBuffer.contents(), count: pixelCount * MemoryLayout<Float>.size)
    try output.write(to: URL(fileURLWithPath: fields[2]))
    signal("DONE\t\(pixelCount)")
}
