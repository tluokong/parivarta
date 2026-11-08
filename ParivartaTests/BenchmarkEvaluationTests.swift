//
//  BenchmarkEvaluationTests.swift
//  ParivartaTests
//
//  Eval harness (E0) สำหรับชุด benchmark ภายนอก 2,500 เอกสาร (GEN_Entity + Goal_Entity)
//  ชุดนี้เป็นย่อหน้ายาว มี PII เป้าหมาย 1 ตัว/เอกสาร พร้อม character offset (code-point)
//  5 ชนิด × 500: PERSON / PHONE / EMAIL / ADDRESS / NATIONAL_ID
//
//  วัด per-type recall 3 แบบ:
//    - containment (privacy): ทุกอักขระของ gold ถูก mask ครบไหม  ← ตัวชี้วัดหลักด้านความเป็นส่วนตัว
//    - typed-containment:      ครบ + ชนิดตรง (หลัง map ชื่อชนิด)
//    - exact-span:             มี entity ที่ช่วงตรงเป๊ะ + ชนิดตรง
//  และ over-detection (precision proxy): จำนวน entity ที่ไม่ทับ gold + สัดส่วนอักขระ mask เกิน
//
//  หมายเหตุพิกัด: gold offset เป็น "code point" (Python) ส่วน String.Index เป็น grapheme
//  จึงเทียบทุกอย่างในระบบพิกัด Unicode scalar เพื่อให้ offset ตรงกับเฉลย
//

import Testing
import Foundation
@testable import Parivarta

/// ใช้ระบุ bundle ของ test target เพื่อหาไฟล์ fixture
private final class BundleToken {}

@MainActor
struct BenchmarkEvaluation {

    struct GenDoc: Decodable { let id: String; let text: String }
    struct GoldItem: Decodable {
        let id: String
        let target_entity_type: String
        let entity_text: String
        let annotation_start: Int
        let annotation_end: Int
    }

    /// map ชื่อชนิดของ benchmark → ชนิดที่ detector ของเราปล่อยออกมา
    static func mappedTypes(for goldType: String) -> Set<String> {
        switch goldType {
        case "PHONE":       return ["PHONE_NUMBER"]
        case "NATIONAL_ID": return ["THAI_ID"]
        case "EMAIL":       return ["EMAIL"]
        case "PERSON":      return ["PERSON"]
        case "ADDRESS":     return ["ADDRESS", "LOCATION"]   // ADDRESS ยังไม่มี — LOCATION เผื่ออนาคต
        default:            return [goldType]
        }
    }

    // MARK: - fixture loading

    static func loadFixtures() throws -> [(doc: GenDoc, gold: GoldItem)] {
        let bundle = Bundle(for: BundleToken.self)
        func url(_ name: String) throws -> URL {
            if let u = bundle.url(forResource: name, withExtension: "jsonl") { return u }
            if let u = bundle.url(forResource: name, withExtension: "jsonl", subdirectory: "Fixtures") { return u }
            throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "ไม่พบ \(name).jsonl ใน test bundle"])
        }
        let dec = JSONDecoder()
        func lines(_ u: URL) throws -> [String] {
            // ไฟล์ต้นฉบับใช้ CRLF — split ด้วย isNewline (ครอบคลุม \n, \r, \r\n)
            try String(contentsOf: u, encoding: .utf8)
                .split(whereSeparator: \.isNewline)
                .map(String.init)
        }
        let genLines = try lines(try url("GEN_Entity"))
        let goldLines = try lines(try url("Goal_Entity"))
        var docs: [String: GenDoc] = [:]
        var firstErr: String?
        for l in genLines {
            do { let d = try dec.decode(GenDoc.self, from: Data(l.utf8)); docs[d.id] = d }
            catch { if firstErr == nil { firstErr = "GEN decode: \(error) | line=\(l.prefix(80))" } }
        }
        var pairs: [(GenDoc, GoldItem)] = []
        for l in goldLines {
            do {
                let g = try dec.decode(GoldItem.self, from: Data(l.utf8))
                if let doc = docs[g.id] { pairs.append((doc, g)) }
            } catch { if firstErr == nil { firstErr = "GOLD decode: \(error) | line=\(l.prefix(80))" } }
        }
        print("[bench] genLines=\(genLines.count) goldLines=\(goldLines.count) docs=\(docs.count) pairs=\(pairs.count) firstErr=\(firstErr ?? "none")")
        return pairs
    }

    // MARK: - scalar-coordinate helpers

    /// แปลงช่วง String.Index → ช่วง offset ในหน่วย Unicode scalar (ให้ตรงพิกัดกับ gold แบบ Python)
    static func scalarRange(_ r: Range<String.Index>, in text: String) -> Range<Int> {
        let scalars = text.unicodeScalars
        let lo = scalars.distance(from: scalars.startIndex, to: r.lowerBound)
        let hi = scalars.distance(from: scalars.startIndex, to: r.upperBound)
        return lo..<hi
    }

    // MARK: - E0 baseline

    @Test func evaluateBenchmarkBaseline() async throws {
        let pairs = try Self.loadFixtures()
        #expect(!pairs.isEmpty)

        let tagger = try? await ThaiNERTagger.load()
        let detector = PIIDetector()

        // ตัวนับต่อชนิด
        struct Acc { var total = 0, covered = 0, typedCovered = 0, exact = 0 }
        var byType: [String: Acc] = [:]
        var extraDetections = 0
        var totalScalars = 0, overMaskScalars = 0

        for (doc, gold) in pairs {
            let text = doc.text
            let scalarCount = text.unicodeScalars.count
            let goldRange = gold.annotation_start..<gold.annotation_end
            let wanted = Self.mappedTypes(for: gold.target_entity_type)

            let thai = tagger != nil ? await tagger!.tag(text) : []
            let entities = detector.detect(in: text, extraEntities: thai)
            let detected = entities.map { (type: $0.type, range: Self.scalarRange($0.range, in: text)) }

            var maskedSet = Set<Int>()
            for d in detected { maskedSet.formUnion(d.range) }

            var acc = byType[gold.target_entity_type] ?? Acc()
            acc.total += 1
            let covered = goldRange.allSatisfy { maskedSet.contains($0) }
            if covered { acc.covered += 1 }
            let typedCovered = detected.contains {
                wanted.contains($0.type) &&
                $0.range.lowerBound <= goldRange.lowerBound && $0.range.upperBound >= goldRange.upperBound
            }
            if typedCovered { acc.typedCovered += 1 }
            let exact = detected.contains { wanted.contains($0.type) && $0.range == goldRange }
            if exact { acc.exact += 1 }
            byType[gold.target_entity_type] = acc

            for d in detected where d.range.upperBound <= goldRange.lowerBound || d.range.lowerBound >= goldRange.upperBound {
                extraDetections += 1
            }
            totalScalars += scalarCount
            for i in maskedSet where !goldRange.contains(i) { overMaskScalars += 1 }
        }

        func pct(_ a: Int, _ b: Int) -> String { b > 0 ? String(format: "%5.1f%%", Double(a) / Double(b) * 100) : "  n/a" }

        var out = "\n===== E0 — External Benchmark Baseline (\(pairs.count) docs) =====\n"
        out += "ThaiNER model loaded: \(tagger != nil)\n\n"
        out += "| Type        | N   | Cover(priv)↑ | Typed↑  | Exact↑  |\n"
        out += "|-------------|-----|--------------|---------|---------|\n"
        var tN = 0, tCov = 0, tTyped = 0, tExact = 0
        for key in ["PERSON", "PHONE", "EMAIL", "ADDRESS", "NATIONAL_ID"] {
            guard let a = byType[key] else { continue }
            out += "| \(key.padding(toLength: 11, withPad: " ", startingAt: 0)) | \(String(format: "%3d", a.total)) |     \(pct(a.covered, a.total)) | \(pct(a.typedCovered, a.total)) | \(pct(a.exact, a.total)) |\n"
            tN += a.total; tCov += a.covered; tTyped += a.typedCovered; tExact += a.exact
        }
        out += "|-------------|-----|--------------|---------|---------|\n"
        out += "| OVERALL     | \(String(format: "%3d", tN)) |     \(pct(tCov, tN)) | \(pct(tTyped, tN)) | \(pct(tExact, tN)) |\n"
        out += "\nOver-detection (precision proxy):\n"
        out += "  extra detections (ไม่ทับ gold): \(extraDetections)  (\(String(format: "%.2f", Double(extraDetections)/Double(max(1,pairs.count))))/doc)\n"
        out += "  over-mask scalar ratio:        \(pct(overMaskScalars, totalScalars))\n"
        out += "================================================\n"
        print(out)

    }
}
