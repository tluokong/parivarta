//
//  ThaiNEREvaluationTests.swift
//  ParivartaTests
//
//  Eval harness เชิงวิจัย: วัดคุณภาพการตรวจจับ PII ของทั้ง pipeline
//  (regex + NLTagger + heuristic + Thai NER Core ML + watchlist) เทียบ gold set
//
//  เมตริก:
//   - Privacy (หลัก): coverage-recall / leak-rate  → PII ถูก mask ครบไหม (false negative = ข้อมูลหลุด)
//                     over-mask ratio             → มาสก์เกินแค่ไหน (อักขระที่ mask แต่ไม่ใช่ PII)
//   - Entity-level (typed, containment): Precision / Recall / F1
//
//  หมายเหตุ: ประเมินบน "ข้อความดิบ" (ไม่ผ่าน TextPreprocessor) เพื่อให้ offset ของ gold ตรง
//  — การกรอง URL/mention/hashtag เป็นคนละเรื่องกับคุณภาพ NER
//

import Testing
import Foundation
@testable import Parivarta

@MainActor
struct ThaiNEREvaluation {

    /// หนึ่งตัวอย่าง: ข้อความ + รายการ PII ที่ต้องถูก mask (ข้อความย่อย, ชนิด)
    struct Sample {
        let text: String
        let mentions: [(text: String, type: String)]
        init(_ text: String, _ mentions: [(String, String)]) {
            self.text = text
            self.mentions = mentions.map { (text: $0.0, type: $0.1) }
        }
    }

    /// Gold set — ขยายได้เรื่อย ๆ (ตั้งเป้า 50–100 ประโยค)
    static let gold: [Sample] = [
        Sample("นายสมชาย เลิศวิสัย โทร 02-123-4567 อีเมล support@fake-bank.com",
               [("สมชาย เลิศวิสัย", "PERSON"), ("02-123-4567", "PHONE_NUMBER"), ("support@fake-bank.com", "EMAIL")]),
        Sample("ประเสริฐ ลือโขง โอนเงินให้ วิชัย มั่งมี",
               [("ประเสริฐ ลือโขง", "PERSON"), ("วิชัย มั่งมี", "PERSON")]),
        Sample("พบผู้ต้องสงสัยชื่อ มาลี ใจดี อาศัยอยู่ที่จังหวัดเชียงใหม่",
               [("มาลี ใจดี", "PERSON"), ("เชียงใหม่", "LOCATION")]),
        Sample("บริษัท การบินไทย จำกัด มหาชน ตั้งอยู่ที่กรุงเทพมหานคร",
               [("การบินไทย", "ORGANIZATION"), ("กรุงเทพมหานคร", "LOCATION")]),
        Sample("นาย Somchai Jaidee ส่งข้อมูลไปที่ john@example.com",
               [("Somchai Jaidee", "PERSON"), ("john@example.com", "EMAIL")]),
        Sample("ติดต่อสมหญิงที่เบอร์0891112222",
               [("สมหญิง", "PERSON"), ("0891112222", "PHONE_NUMBER")]),
        Sample("เลขบัตรประชาชน 1234567890123 ของนางสาวสุดา",
               [("1234567890123", "THAI_ID"), ("สุดา", "PERSON")]),
        Sample("เซิร์ฟเวอร์ IP 192.168.1.50 ของบริษัท ทรู คอร์ปอเรชั่น จำกัด",
               [("192.168.1.50", "IP_ADDRESS"), ("ทรู คอร์ปอเรชั่น", "ORGANIZATION")]),
        Sample("โทรหาคุณอนันต์ที่ +66-89-123-4567",
               [("อนันต์", "PERSON"), ("+66-89-123-4567", "PHONE_NUMBER")]),
        Sample("เกิดเหตุที่อำเภอหาดใหญ่ จังหวัดสงขลา",
               [("หาดใหญ่", "LOCATION"), ("สงขลา", "LOCATION")]),
        Sample("ส่งไปที่ a@b.com และ c@d.co.th",
               [("a@b.com", "EMAIL"), ("c@d.co.th", "EMAIL")]),
        Sample("กิตติ วัฒนา เป็นพยานในคดีนี้",
               [("กิตติ วัฒนา", "PERSON")]),
        Sample("สัญญากับบริษัท สยามเทค จำกัด ลงนามแล้ว",
               [("สยามเทค", "ORGANIZATION")]),
        Sample("นาย เอกชัย รายงานเหตุการณ์",
               [("เอกชัย", "PERSON")]),
        Sample("เดินทางจากภูเก็ตไปเชียงราย",
               [("ภูเก็ต", "LOCATION"), ("เชียงราย", "LOCATION")]),
        Sample("แฮกเกอร์ชื่อ John Connor อยู่ที่ Russia",
               [("John Connor", "PERSON"), ("Russia", "LOCATION")]),
        Sample("คุณวิภา จากบริษัท เอไอ จำกัด โทร 02-999-8888",
               [("วิภา", "PERSON"), ("เอไอ", "ORGANIZATION"), ("02-999-8888", "PHONE_NUMBER")]),
        Sample("วันนี้อากาศดีมากเหมาะแก่การเดินเล่น", []),   // negative — ควรไม่ mask อะไร
        Sample("รายงานระบุว่านาย ธนา ศรีสุข และ นางสาว ปนัดดา ทองคำ ทำงานที่บริษัท ไซเบอร์ซีเคียว จำกัด",
               [("ธนา ศรีสุข", "PERSON"), ("ปนัดดา ทองคำ", "PERSON"), ("ไซเบอร์ซีเคียว", "ORGANIZATION")]),
        Sample("ผู้เสียหายชื่อ อรุณี พบเงินโอนเข้าบัญชีจากเบอร์ 081-234-5678",
               [("อรุณี", "PERSON"), ("081-234-5678", "PHONE_NUMBER")]),

        // --- ชุดขยาย (v2): ชื่อคน ---
        Sample("สมชาย ใจกล้า ให้การกับตำรวจ", [("สมชาย ใจกล้า", "PERSON")]),
        Sample("นายวีระ สุขสันต์ เป็นเจ้าของบัญชี", [("วีระ สุขสันต์", "PERSON")]),
        Sample("ผู้ต้องหาคือ นางมาลี ทองสุข", [("มาลี ทองสุข", "PERSON")]),
        Sample("คุณธนพล กิจเจริญ โทรมาแจ้งเหตุ", [("ธนพล กิจเจริญ", "PERSON")]),
        Sample("พยานชื่อ สุดารัตน์ ยืนยันเหตุการณ์", [("สุดารัตน์", "PERSON")]),
        Sample("เด็กชายกิตติศักดิ์ หายตัวไปเมื่อวาน", [("กิตติศักดิ์", "PERSON")]),
        Sample("อนุชา พร้อมด้วย ประภาส เดินทางมาถึง",
               [("อนุชา", "PERSON"), ("ประภาส", "PERSON")]),

        // --- สถานที่ ---
        Sample("สำนักงานตั้งอยู่ที่จังหวัดนนทบุรี", [("นนทบุรี", "LOCATION")]),
        Sample("เดินทางไปอุดรธานีและขอนแก่น",
               [("อุดรธานี", "LOCATION"), ("ขอนแก่น", "LOCATION")]),
        Sample("พบตัวที่จังหวัดชลบุรี", [("ชลบุรี", "LOCATION")]),
        Sample("ระบบล่มที่ศูนย์ข้อมูลจังหวัดปทุมธานี", [("ปทุมธานี", "LOCATION")]),

        // --- องค์กร ---
        Sample("บริษัท ไทยเบฟเวอเรจ จำกัด รายงานผลประกอบการ", [("ไทยเบฟเวอเรจ", "ORGANIZATION")]),
        Sample("ธนาคารกสิกรไทย แจ้งเตือนลูกค้าเรื่องมิจฉาชีพ", [("ธนาคารกสิกรไทย", "ORGANIZATION")]),
        Sample("หน่วยงาน สกมช. ออกประกาศเตือนภัย", [("สกมช.", "ORGANIZATION")]),

        // --- เบอร์โทร ---
        Sample("โทร 081-234-5678 หรือ 02-555-1234",
               [("081-234-5678", "PHONE_NUMBER"), ("02-555-1234", "PHONE_NUMBER")]),
        Sample("เบอร์มือถือ 0899998888 ติดต่อได้", [("0899998888", "PHONE_NUMBER")]),
        Sample("ติดต่อสายด่วน +66 2 123 4567", [("+66 2 123 4567", "PHONE_NUMBER")]),

        // --- อีเมล ---
        Sample("อีเมล somchai.j@gmail.com และ admin@company.co.th",
               [("somchai.j@gmail.com", "EMAIL"), ("admin@company.co.th", "EMAIL")]),
        Sample("ส่งรายงานไปที่ ceo@bank-secure.org ด่วน", [("ceo@bank-secure.org", "EMAIL")]),

        // --- เลขบัตรประชาชน ---
        Sample("บัตรประชาชนเลขที่ 3100600123456", [("3100600123456", "THAI_ID")]),
        Sample("หมายเลข 1234567890123 และ 9876543210987",
               [("1234567890123", "THAI_ID"), ("9876543210987", "THAI_ID")]),

        // --- IP ---
        Sample("การโจมตีมาจาก 203.150.20.5", [("203.150.20.5", "IP_ADDRESS")]),
        Sample("เซิร์ฟเวอร์ภายใน 10.10.10.254 ถูกเจาะ", [("10.10.10.254", "IP_ADDRESS")]),

        // --- code-mixed ---
        Sample("แฮกเกอร์ใช้ชื่อ Mr. David Kim ส่งเมล david@darkweb.io",
               [("David Kim", "PERSON"), ("david@darkweb.io", "EMAIL")]),
        Sample("ผู้ต้องสงสัย Alice จากประเทศ Singapore",
               [("Alice", "PERSON"), ("Singapore", "LOCATION")]),

        // --- negative (ไม่มี PII) ---
        Sample("ระบบทำงานปกติดีทุกอย่างเรียบร้อย", []),
        Sample("โปรดตรวจสอบรายงานประจำเดือนนี้", []),
        Sample("การประชุมเลื่อนออกไปเป็นสัปดาห์หน้า", []),

        // --- OSINT เข้มข้น (หลายชนิดในประโยคเดียว) ---
        Sample("พบว่านายสมศักดิ์ รักชาติ เบอร์ 0812223333 อีเมล somsak@mail.com อยู่จังหวัดเชียงใหม่",
               [("สมศักดิ์ รักชาติ", "PERSON"), ("0812223333", "PHONE_NUMBER"),
                ("somsak@mail.com", "EMAIL"), ("เชียงใหม่", "LOCATION")]),
        Sample("บริษัท ดาต้าเทค จำกัด มี CEO ชื่อ John Smith โทร 02-000-1111",
               [("ดาต้าเทค", "ORGANIZATION"), ("John Smith", "PERSON"), ("02-000-1111", "PHONE_NUMBER")]),
    ]

    /// เมตริกสรุปของหนึ่ง configuration
    struct Metrics {
        let label: String
        let samples: Int
        let goldTotal: Int
        let leakRate: Double
        let coverageRecall: Double
        let typedRecall: Double
        let precision: Double
        let f1: Double
        let overMaskRatio: Double
        let leaks: [String]
    }

    /// รัน pipeline ทั้งชุด gold (โหมด gazetteer ตั้งไว้แล้วโดยผู้เรียกผ่าน TaskLocal)
    /// แล้วคืนเมตริก + log รายประโยค
    private func evaluate(label: String,
                          tagger: ThaiNERTagger?,
                          detector: PIIDetector) async -> (metrics: Metrics, detail: String) {
        var goldTotal = 0, covered = 0, typedTP = 0
        var detectedTotal = 0, detectedFP = 0
        var totalChars = 0, overMaskChars = 0
        var allLeaks: [String] = []

        var detail = "\n----- \(label) — per-sentence -----\n"
        for sample in Self.gold {
            let text = sample.text
            let thai = tagger != nil ? await tagger!.tag(text) : []
            let entities = detector.detect(in: text, extraEntities: thai)

            let detected: [(type: String, range: Range<Int>)] = entities.map {
                (type: $0.type, range: Self.intRange($0.range, in: text))
            }
            var maskedSet = Set<Int>()
            for d in detected { maskedSet.formUnion(d.range) }

            var goldRanges: [(type: String, range: Range<Int>)] = []
            for m in sample.mentions {
                for r in Self.occurrences(of: m.text, in: text) {
                    goldRanges.append((type: m.type, range: r))
                }
            }

            var sampleLeaks: [String] = []
            for g in goldRanges {
                goldTotal += 1
                let isCovered = g.range.allSatisfy { maskedSet.contains($0) }
                if isCovered { covered += 1 } else { sampleLeaks.append(Self.slice(text, g.range)) }
                let typed = detected.contains { $0.type == g.type && $0.range.lowerBound <= g.range.lowerBound && $0.range.upperBound >= g.range.upperBound }
                if typed { typedTP += 1 }
            }
            allLeaks.append(contentsOf: sampleLeaks)

            for d in detected {
                detectedTotal += 1
                let overlapsGold = goldRanges.contains { !(d.range.upperBound <= $0.range.lowerBound || d.range.lowerBound >= $0.range.upperBound) }
                if !overlapsGold { detectedFP += 1 }
            }

            totalChars += text.count
            for i in maskedSet where !goldRanges.contains(where: { $0.range.contains(i) }) {
                overMaskChars += 1
            }

            if !sampleLeaks.isEmpty {
                detail += "  LEAK  \(text.prefix(40)) → \(sampleLeaks.joined(separator: " | "))\n"
            }
        }

        let coverageRecall = goldTotal > 0 ? Double(covered) / Double(goldTotal) : 1
        let typedRecall = goldTotal > 0 ? Double(typedTP) / Double(goldTotal) : 1
        let precision = detectedTotal > 0 ? Double(detectedTotal - detectedFP) / Double(detectedTotal) : 1
        let f1 = (precision + typedRecall) > 0 ? 2 * precision * typedRecall / (precision + typedRecall) : 0
        let overMaskRatio = totalChars > 0 ? Double(overMaskChars) / Double(totalChars) : 0

        let metrics = Metrics(label: label, samples: Self.gold.count, goldTotal: goldTotal,
                              leakRate: 1 - coverageRecall, coverageRecall: coverageRecall,
                              typedRecall: typedRecall, precision: precision, f1: f1,
                              overMaskRatio: overMaskRatio, leaks: allLeaks)
        return (metrics, detail)
    }

    @Test func evaluatePIIDetectionPipeline() async throws {
        let tagger = try? await ThaiNERTagger.load()
        let detector = PIIDetector()

        var detail = ""
        let configs: [(String, ThaiGazetteer.Mode)] = [
            ("Baseline (no gazetteer)", .off),
            ("Exp 2  (gazetteer Pass 1)", .pass1),
            ("Exp 2b (Pass 1 + Pass 2)", .full),
        ]

        var results: [Metrics] = []
        for (label, mode) in configs {
            // ตั้งโหมดในขอบเขต task ของ config นี้ (TaskLocal) → ไม่รบกวนเทสต์อื่นที่รันขนาน
            let (metrics, d) = await ThaiGazetteer.$mode.withValue(mode) {
                await evaluate(label: label, tagger: tagger, detector: detector)
            }
            results.append(metrics)
            detail += d
        }

        func pct(_ v: Double) -> String { String(format: "%5.1f%%", v * 100) }

        var out = "\n===== Thai NER / PII — Comparative Evaluation =====\n"
        out += "ThaiNER model loaded: \(tagger != nil)\n"
        out += "Gold set: \(Self.gold.count) sentences, \(results.first?.goldTotal ?? 0) mentions (same set for all configs)\n"
        out += detail
        out += "\n----- SUMMARY (apples-to-apples on \(Self.gold.count)-sentence set) -----\n"
        out += "| Config                     | Leak↓  | Cover↑ | TypedR↑| Prec↑  | F1↑    | OverM↓ |\n"
        out += "|----------------------------|--------|--------|--------|--------|--------|--------|\n"
        for m in results {
            let name = m.label.padding(toLength: 26, withPad: " ", startingAt: 0)
            out += "| \(name) |\(pct(m.leakRate)) |\(pct(m.coverageRecall)) |\(pct(m.typedRecall)) |\(pct(m.precision)) |\(pct(m.f1)) |\(pct(m.overMaskRatio)) |\n"
        }
        out += "==================================================\n"
        if let last = results.last, !last.leaks.isEmpty {
            out += "Remaining leaks (Exp 2b): \(last.leaks.joined(separator: ", "))\n"
        }
        out += "Note: Exp 1 (NER word-seg port) was a measured negative result on the 20-set and reverted; not in the current codebase.\n"
        print(out)

        #expect(!Self.gold.isEmpty)
    }

    // MARK: - helpers

    private static func intRange(_ r: Range<String.Index>, in text: String) -> Range<Int> {
        let lo = text.distance(from: text.startIndex, to: r.lowerBound)
        let hi = text.distance(from: text.startIndex, to: r.upperBound)
        return lo..<hi
    }

    private static func occurrences(of sub: String, in text: String) -> [Range<Int>] {
        var result: [Range<Int>] = []
        var start = text.startIndex
        while let r = text.range(of: sub, range: start..<text.endIndex) {
            result.append(intRange(r, in: text))
            start = r.upperBound
        }
        return result
    }

    private static func slice(_ text: String, _ r: Range<Int>) -> String {
        let lo = text.index(text.startIndex, offsetBy: r.lowerBound)
        let hi = text.index(text.startIndex, offsetBy: r.upperBound)
        return String(text[lo..<hi])
    }
}
