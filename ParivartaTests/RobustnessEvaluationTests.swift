//
//  RobustnessEvaluationTests.swift
//  ParivartaTests
//
//  E6 — Hard-positive robustness: positive ที่ยากกว่าชุดแม่แบบ (ที่อยู่ไม่เป็นแม่แบบ,
//       หลาย entity, nested, ชื่อ romanized/ต่างชาติ) วัด coverage รายหมวด
//  E7 — Gazetteer generalization: วัด coverage ของ PERSON ภายใต้ gazetteer 3 โหมด
//       (off / pass1 / full) เพื่อแยกส่วนที่คลังชื่อ 21k ช่วย ออกจาก NER/heuristic ล้วน
//
//  scoring แบบ containment ในพิกัด Unicode scalar (กัน grapheme ของไทยเพี้ยน)
//

import Testing
import Foundation
@testable import Parivarta

@MainActor
struct RobustnessEvaluation {

    struct Sample {
        let category: String
        let text: String
        let mentions: [String]     // ข้อความย่อยที่ต้องถูก mask (ชนิดไม่สำคัญในการวัด coverage)
        init(_ category: String, _ text: String, _ mentions: [String]) {
            self.category = category; self.text = text; self.mentions = mentions
        }
    }

    // MARK: - E6 hard-positive set

    static let hardPositives: [Sample] = [
        // --- ADDR_VAR: ที่อยู่ที่ไม่ขึ้นต้นด้วย "บ้านเลขที่" / มีรหัสไปรษณีย์ / คอนโด ---
        Sample("ADDR_VAR", "ที่อยู่ 99/1 หมู่ 5 ต.บางพลี อ.เมือง จ.ชลบุรี 20000",
               ["99/1 หมู่ 5 ต.บางพลี อ.เมือง จ.ชลบุรี 20000"]),
        Sample("ADDR_VAR", "บ้านเลขที่ 12 ถนนสุขุมวิท กรุงเทพฯ 10110",
               ["บ้านเลขที่ 12 ถนนสุขุมวิท กรุงเทพฯ 10110"]),
        Sample("ADDR_VAR", "คอนโดชั้น 8 ห้อง 802 อาคารเอ ซอยลาดพร้าว 15",
               ["คอนโดชั้น 8 ห้อง 802 อาคารเอ ซอยลาดพร้าว 15"]),
        Sample("ADDR_VAR", "เลขที่ 5 ซอย 3 ถนนพหลโยธิน แขวงจตุจักร เขตจตุจักร",
               ["เลขที่ 5 ซอย 3 ถนนพหลโยธิน แขวงจตุจักร เขตจตุจักร"]),

        // --- MULTI: หลาย entity ในประโยคเดียว ---
        Sample("MULTI", "นายสมชาย ใจดี โทร 081-234-5678 อีเมล somchai@mail.com",
               ["สมชาย ใจดี", "081-234-5678", "somchai@mail.com"]),
        Sample("MULTI", "ติดต่อ นางสาวมาลี หรือ นายวิชัย ที่ 02-111-2222 และ 02-333-4444",
               ["มาลี", "วิชัย", "02-111-2222", "02-333-4444"]),
        Sample("MULTI", "ผู้โอนคือ ธนกร ปลายทาง กสิกร 1234567890123 ยอด 5000 บาท",
               ["ธนกร", "1234567890123"]),

        // --- NESTED: บุคคลซ้อนในองค์กร / honorific เฉพาะทาง ---
        Sample("NESTED", "บริษัท สมชายพาณิชย์ จำกัด มีนายสมชาย เป็นเจ้าของ",
               ["สมชายพาณิชย์", "สมชาย"]),
        Sample("NESTED", "โรงพยาบาลรามาธิบดี โดย นพ. ประเสริฐ สุขใจ",
               ["ประเสริฐ สุขใจ"]),
        Sample("NESTED", "ดร. วิรัช มานะ อาจารย์ประจำมหาวิทยาลัยเชียงใหม่",
               ["วิรัช มานะ", "เชียงใหม่"]),

        // --- ROMANIZED / FOREIGN: ชื่อไทยสะกดอังกฤษ + ชื่อต่างชาติ ---
        Sample("ROMANIZED", "Mr. Somchai Jaidee ส่งเอกสารเรียบร้อยแล้ว",
               ["Somchai Jaidee"]),
        Sample("ROMANIZED", "ผู้จัดการชื่อ Michael Chen จะติดต่อกลับ",
               ["Michael Chen"]),
        Sample("ROMANIZED", "คุณ Somsak Wong เป็นพยานในเหตุการณ์",
               ["Somsak Wong"]),
    ]

    @Test func evaluateHardPositives() async throws {
        let tagger = try? await ThaiNERTagger.load()
        let detector = PIIDetector()

        struct Acc { var total = 0, covered = 0 }
        var byCat: [String: Acc] = [:]
        var leaks: [String] = []
        var gTotal = 0, gCovered = 0

        for s in Self.hardPositives {
            let thai = tagger != nil ? await tagger!.tag(s.text) : []
            let entities = detector.detect(in: s.text, extraEntities: thai)
            let masked = Self.maskedScalars(entities, in: s.text)

            var acc = byCat[s.category] ?? Acc()
            for m in s.mentions {
                gTotal += 1; acc.total += 1
                let covered = Self.occurrences(of: m, in: s.text).contains { $0.allSatisfy { masked.contains($0) } }
                if covered { gCovered += 1; acc.covered += 1 }
                else { leaks.append("[\(s.category)] \(m)") }
            }
            byCat[s.category] = acc
        }

        func pct(_ a: Int, _ b: Int) -> String { b > 0 ? String(format: "%5.1f%%", Double(a)/Double(b)*100) : "  n/a" }
        var out = "\n===== E6 — Hard-Positive Robustness =====\n"
        out += "ThaiNER model loaded: \(tagger != nil)\n\n"
        out += "| Category   | mentions | covered | coverage↑ |\n"
        out += "|------------|----------|---------|-----------|\n"
        for key in ["ADDR_VAR", "MULTI", "NESTED", "ROMANIZED"] {
            guard let a = byCat[key] else { continue }
            out += "| \(key.padding(toLength: 10, withPad: " ", startingAt: 0)) |    \(String(format: "%2d", a.total))    |   \(String(format: "%2d", a.covered))    |   \(pct(a.covered, a.total)) |\n"
        }
        out += "|------------|----------|---------|-----------|\n"
        out += "| OVERALL    |    \(String(format: "%2d", gTotal))    |   \(String(format: "%2d", gCovered))    |   \(pct(gCovered, gTotal)) |\n"
        if !leaks.isEmpty { out += "\nLeaks:\n" + leaks.map { "  • \($0)" }.joined(separator: "\n") + "\n" }
        out += "==========================================\n"
        print(out)
        #expect(!Self.hardPositives.isEmpty)
    }

    // MARK: - E7 gazetteer generalization

    /// PERSON ที่คละความน่าจะอยู่/ไม่อยู่ในคลัง 21k (รวมชื่อหายาก/ต่างชาติ)
    static let personProbe: [Sample] = [
        Sample("PERSON", "รายงานโดยนายจิรวัฒน์ ภูวนารถ เมื่อเช้านี้", ["จิรวัฒน์ ภูวนารถ"]),
        Sample("PERSON", "ผู้ต้องสงสัยชื่อ ณัฐดนัย ธนกฤต", ["ณัฐดนัย ธนกฤต"]),
        Sample("PERSON", "พยานคือ นางสาวฟ้าใหม่ เมฆา", ["ฟ้าใหม่ เมฆา"]),
        Sample("PERSON", "คุณอากิระ ทานากะ เดินทางมาถึงแล้ว", ["อากิระ ทานากะ"]),
        Sample("PERSON", "เจ้าของบัญชีคือ นายเตชินท์ กิตติคุณ", ["เตชินท์ กิตติคุณ"]),
        Sample("PERSON", "ผู้เสียหายชื่อ ปุณณภา", ["ปุณณภา"]),
        Sample("PERSON", "นายกฤตเมธ วงศ์ไพศาล ให้การกับตำรวจ", ["กฤตเมธ วงศ์ไพศาล"]),
        Sample("PERSON", "คุณวิน ดีเซล มาเยือนไทย", ["วิน ดีเซล"]),
        Sample("PERSON", "ผู้จัดการคือ นางพิมพ์มาดา เรืองศรี", ["พิมพ์มาดา เรืองศรี"]),
        Sample("PERSON", "แขกรับเชิญชื่อ ก้องภพ อัศวิน", ["ก้องภพ อัศวิน"]),
    ]

    @Test func evaluateGazetteerGeneralization() async throws {
        let tagger = try? await ThaiNERTagger.load()
        let detector = PIIDetector()

        func coverage(_ mode: ThaiGazetteer.Mode) async -> (covered: Int, total: Int) {
            // ตั้งโหมดในขอบเขต task นี้ (TaskLocal) กันชนกับเทสต์อื่นที่รันขนาน
            await ThaiGazetteer.$mode.withValue(mode) {
                var covered = 0, total = 0
                for s in Self.personProbe {
                    let thai = tagger != nil ? await tagger!.tag(s.text) : []
                    let entities = detector.detect(in: s.text, extraEntities: thai)
                    let masked = Self.maskedScalars(entities, in: s.text)
                    for m in s.mentions {
                        total += 1
                        if Self.occurrences(of: m, in: s.text).contains(where: { $0.allSatisfy { masked.contains($0) } }) { covered += 1 }
                    }
                }
                return (covered, total)
            }
        }

        let off = await coverage(.off)
        let pass1 = await coverage(.pass1)
        let full = await coverage(.full)

        func pct(_ a: Int, _ b: Int) -> String { b > 0 ? String(format: "%5.1f%%", Double(a)/Double(b)*100) : "  n/a" }
        var out = "\n===== E7 — Gazetteer Generalization (\(Self.personProbe.count) PERSON) =====\n"
        out += "ThaiNER model loaded: \(tagger != nil)\n\n"
        out += "| Gazetteer mode          | covered | coverage↑ |\n"
        out += "|-------------------------|---------|-----------|\n"
        out += "| off  (NER + heuristic)  |   \(String(format: "%2d", off.covered))/\(off.total)  |   \(pct(off.covered, off.total)) |\n"
        out += "| pass1 (+ name-run/cue)  |   \(String(format: "%2d", pass1.covered))/\(pass1.total)  |   \(pct(pass1.covered, pass1.total)) |\n"
        out += "| full  (+ run-together)  |   \(String(format: "%2d", full.covered))/\(full.total)  |   \(pct(full.covered, full.total)) |\n"
        out += "|-------------------------|---------|-----------|\n"
        out += "Gazetteer contribution (full − off): +\(full.covered - off.covered) mention(s)\n"
        out += "===================================================\n"
        print(out)
        #expect(!Self.personProbe.isEmpty)
    }

    // MARK: - helpers (scalar coordinate)

    static func maskedScalars(_ entities: [PIIEntity], in text: String) -> Set<Int> {
        let scalars = text.unicodeScalars
        var set = Set<Int>()
        for e in entities {
            let lo = scalars.distance(from: scalars.startIndex, to: e.range.lowerBound)
            let hi = scalars.distance(from: scalars.startIndex, to: e.range.upperBound)
            set.formUnion(lo..<hi)
        }
        return set
    }

    static func occurrences(of sub: String, in text: String) -> [Range<Int>] {
        let scalars = text.unicodeScalars
        var result: [Range<Int>] = []
        var start = text.startIndex
        while let r = text.range(of: sub, range: start..<text.endIndex) {
            let lo = scalars.distance(from: scalars.startIndex, to: r.lowerBound)
            let hi = scalars.distance(from: scalars.startIndex, to: r.upperBound)
            result.append(lo..<hi)
            start = r.upperBound
        }
        return result
    }
}
