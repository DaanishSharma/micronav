// Speak text to an audio file with an installed Apple voice (premium voices included,
// which the `say` command doesn't use properly).
// Usage: swift tts.swift VOICE_ID OUT.caf "text"   (rate: natural default)
//        swift tts.swift --list                    (installed English voices)
import AVFoundation

let args = CommandLine.arguments
if args.count > 1 && args[1] == "--list" {
    for v in AVSpeechSynthesisVoice.speechVoices() where v.language.hasPrefix("en") {
        print(v.identifier, v.name, v.language, "quality:", v.quality.rawValue)
    }
    exit(0)
}
guard args.count >= 4, let voice = AVSpeechSynthesisVoice(identifier: args[1]) else {
    FileHandle.standardError.write("usage: swift tts.swift VOICE_ID OUT.caf TEXT\n".data(using: .utf8)!)
    exit(1)
}
let utterance = AVSpeechUtterance(string: args[3])
utterance.voice = voice
let synth = AVSpeechSynthesizer()
var file: AVAudioFile?
var lastBuffer = Date()
let done = DispatchSemaphore(value: 0)
synth.write(utterance) { buffer in
    guard let pcm = buffer as? AVAudioPCMBuffer else { return }
    lastBuffer = Date()
    if pcm.frameLength == 0 { done.signal(); return }
    do {
        if file == nil {
            file = try AVAudioFile(forWriting: URL(fileURLWithPath: args[2]), settings: pcm.format.settings,
                                   commonFormat: pcm.format.commonFormat, interleaved: pcm.format.isInterleaved)
        }
        try file?.write(from: pcm)
    } catch {
        FileHandle.standardError.write("write failed: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}
// some voices never send the final empty buffer: also stop after 1.5 s without audio
while done.wait(timeout: .now() + 0.05) == .timedOut {
    RunLoop.current.run(until: Date().addingTimeInterval(0.05))
    if file != nil && Date().timeIntervalSince(lastBuffer) > 1.5 { break }
}
