//
//  ExportDetectionsTests.swift
//  ParivartaTests
//
//  M1 (baseline comparison) — export utility, not a pass/fail test.
//  ดัมป์ผลตรวจจับของ Parivarta (และเอกสาร+เฉลยของชุด C/D ที่ฝังใน Swift) เป็น JSONL
//  ในระบบพิกัด Unicode scalar เพื่อให้ตัวให้คะแนน Python เทียบทุกระบบบน input เดียวกัน
//
//  รูปแบบ span:  {"doc_id": "...", "type": "PERSON", "start": 39, "end": 56}
//  รูปแบบ doc:   {"doc_id": "...", "text": "..."}
//
//  ไฟล์ถูกเขียนไปที่ …/Application Support/Parivarta/exports/ (พิมพ์ path ออกมา)
//  แล้วสคริปต์ M1 คัดลอกเข้าที่ tools/eval/exports/
//

import Testing
import Foundation
@testable import Parivarta

private final class AMBBundleToken {}

@MainActor
struct ExportDetections {

    static func exportsDir() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = base.appendingPathComponent("Parivarta/exports", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func scalarSpan(_ r: Range<String.Index>, in text: String) -> (Int, Int) {
        let s = text.unicodeScalars
        return (s.distance(from: s.startIndex, to: r.lowerBound),
                s.distance(from: s.startIndex, to: r.upperBound))
    }

    static func write(_ lines: [String], to name: String) {
        let url = exportsDir().appendingPathComponent(name)
        try? (lines.joined(separator: "\n") + "\n").data(using: .utf8)?.write(to: url)
    }

    /// รันตัวตรวจของเราแล้วคืน span-JSON ต่อ 1 เอกสาร
    static func detect(_ text: String, docID: String, tagger: ThaiNERTagger?, detector: PIIDetector) async -> [String] {
        let thai = tagger != nil ? await tagger!.tag(text) : []
        let entities = detector.detect(in: text, extraEntities: thai)
        return entities.map { e in
            let (a, b) = scalarSpan(e.range, in: text)
            return #"{"doc_id":"\#(docID)","type":"\#(e.type)","start":\#(a),"end":\#(b)}"#
        }
    }

    static func jsonString(_ s: String) -> String {
        // escape สำหรับ text field (มี newline/quote/backslash ได้)
        var out = ""
        for ch in s.unicodeScalars {
            switch ch {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:   out.unicodeScalars.append(ch)
            }
        }
        return out
    }

    @Test func exportForBaselineComparison() async throws {
        let tagger = try? await ThaiNERTagger.load()
        let detector = PIIDetector()

        // ----- Dataset B: มี GEN/Goal อยู่แล้วฝั่ง Python → ส่งออกเฉพาะผลของเรา -----
        var oursB: [String] = []
        let pairs = try BenchmarkEvaluation.loadFixtures()
        for (doc, gold) in pairs {
            oursB += await Self.detect(doc.text, docID: gold.id, tagger: tagger, detector: detector)
        }
        Self.write(oursB, to: "ours_B.jsonl")

        // ----- Dataset C (negatives): ส่งออก docs + ours (gold ว่างโดยนิยาม) -----
        var docsC: [String] = [], oursC: [String] = []
        for (i, s) in NegativeEvaluation.samples.enumerated() where !s.ambiguous {
            let id = String(format: "NEG-%03d", i)
            docsC.append(#"{"doc_id":"\#(id)","text":"\#(Self.jsonString(s.text))"}"#)
            oursC += await Self.detect(s.text, docID: id, tagger: tagger, detector: detector)
        }
        Self.write(docsC, to: "docs_C.jsonl")
        Self.write(oursC, to: "ours_C.jsonl")

        // ----- Dataset D (hard positives): docs + gold (substring→scalar) + ours -----
        var docsD: [String] = [], goldD: [String] = [], oursD: [String] = []
        for (i, s) in RobustnessEvaluation.hardPositives.enumerated() {
            let id = String(format: "HP-%03d", i)
            docsD.append(#"{"doc_id":"\#(id)","text":"\#(Self.jsonString(s.text))","category":"\#(s.category)"}"#)
            let scalars = s.text.unicodeScalars
            for m in s.mentions {
                var start = s.text.startIndex
                while let r = s.text.range(of: m, range: start..<s.text.endIndex) {
                    let a = scalars.distance(from: scalars.startIndex, to: r.lowerBound)
                    let b = scalars.distance(from: scalars.startIndex, to: r.upperBound)
                    goldD.append(#"{"doc_id":"\#(id)","type":"ANY","start":\#(a),"end":\#(b)}"#)
                    start = r.upperBound
                }
            }
            oursD += await Self.detect(s.text, docID: id, tagger: tagger, detector: detector)
        }
        Self.write(docsD, to: "docs_D.jsonl")
        Self.write(goldD, to: "gold_D.jsonl")
        Self.write(oursD, to: "ours_D.jsonl")

        print("[export] wrote to: \(Self.exportsDir().path)")
        print("[export] B ours=\(oursB.count) | C docs=\(docsC.count) ours=\(oursC.count) | D docs=\(docsD.count) gold=\(goldD.count) ours=\(oursD.count)")
        #expect(!oursB.isEmpty)
    }

    // ----- Dataset AMB (minimal-pair ambiguity): docs loaded from the test bundle -----
    // gold_AMB is scored Python-side; here we only emit ours' spans on the AMB documents.
    struct AmbDoc: Decodable { let id: String; let text: String }

    @Test func exportAMBForBaselineComparison() async throws {
        let bundle = Bundle(for: AMBBundleToken.self)
        guard let url = bundle.url(forResource: "docs_AMB", withExtension: "jsonl")
                ?? bundle.url(forResource: "docs_AMB", withExtension: "jsonl", subdirectory: "Fixtures") else {
            throw NSError(domain: "amb", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "ไม่พบ docs_AMB.jsonl ใน test bundle"])
        }
        let dec = JSONDecoder()
        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(whereSeparator: \.isNewline).map(String.init)

        let tagger = try? await ThaiNERTagger.load()
        let detector = PIIDetector()
        var oursAMB: [String] = []
        for l in lines {
            guard let d = try? dec.decode(AmbDoc.self, from: Data(l.utf8)) else { continue }
            oursAMB += await Self.detect(d.text, docID: d.id, tagger: tagger, detector: detector)
        }
        Self.write(oursAMB, to: "ours_AMB.jsonl")
        print("[export] AMB docs=\(lines.count) ours=\(oursAMB.count) → \(Self.exportsDir().path)")
        #expect(!oursAMB.isEmpty)
    }
}
