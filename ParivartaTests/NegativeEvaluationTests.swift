//
//  NegativeEvaluationTests.swift
//  ParivartaTests
//
//  E5 — Precision / false-positive harness ชุด "กับดัก" ภาษาไทย
//
//  Benchmark เดิม (BenchmarkEvaluation) เป็น positive-only จึงวัด precision ไม่ได้
//  ชุดนี้ทุกประโยค "ไม่มี PII จริง" แต่มีตัวลวง — วัดว่า detector ยิงพลาด (false positive)
//  แค่ไหน และ mask เกินกี่ตัวอักษร
//
//  แยก 2 กลุ่ม:
//   - hard-negative (expectClean): ผู้อ่านคนไทยเห็นชัดว่าไม่ใช่ PII → การตรวจพบ = false positive
//   - format-collision (ambiguous): รูปแบบเหมือน PII เป๊ะ (เลข 13 หลัก, เบอร์ 10 หลัก)
//       รายงานแยกเป็น "อัตราการชนรูปแบบ" ไม่นับเป็น FP (เป็น recall/precision tradeoff โดยธรรมชาติ)
//
//  ประเมินบนข้อความดิบ (เหมือน BenchmarkEvaluation) เพื่อเทียบกันได้
//

import Testing
import Foundation
@testable import Parivarta

@MainActor
struct NegativeEvaluation {

    struct Neg {
        let category: String
        let text: String
        let ambiguous: Bool
        init(_ category: String, _ text: String, ambiguous: Bool = false) {
            self.category = category; self.text = text; self.ambiguous = ambiguous
        }
    }

    static let samples: [Neg] = [
        // --- NUMERIC: ตัวเลขที่ไม่ใช่ PII ---
        Neg("NUMERIC", "ยอดขายรวมทั้งปีอยู่ที่ 1234567890123 บาทถ้วน", ambiguous: true),   // 13 หลัก
        Neg("NUMERIC", "รหัสสินค้าในคลังคือ 1000000000001 ครับ", ambiguous: true),          // 13 หลัก
        Neg("NUMERIC", "หมายเลขคำสั่งซื้อของคุณคือ 0899999999", ambiguous: true),           // 10 หลัก เหมือนเบอร์
        Neg("NUMERIC", "เวอร์ชันซอฟต์แวร์ล่าสุดคือ 10.0.19045.1 แล้ว"),                     // ไม่ใช่ IP (octet เกิน)
        Neg("NUMERIC", "ตั้งค่า gateway เป็น 192.168.0.300 ไม่สำเร็จ"),                      // IP-shaped แต่ octet>255
        Neg("NUMERIC", "การประชุมเริ่มเวลา 09:30 ถึง 16:30 น."),
        Neg("NUMERIC", "สินค้าราคา 1,250 บาท ลดเพิ่มอีก 15%"),
        Neg("NUMERIC", "ปีงบประมาณ 2567 ได้รับจัดสรรงบเพิ่ม"),

        // --- NAME_COLLIDE: คำทั่วไปที่พ้องกับชื่อคน ---
        Neg("NAME_COLLIDE", "เขาสะสมสมบัติไว้มากมายหลายอย่าง"),
        Neg("NAME_COLLIDE", "ทุกคนอยากมีความมั่นคงในหน้าที่การงาน"),
        Neg("NAME_COLLIDE", "ผลงานวิจัยชิ้นนี้ประเสริฐเกินคาด"),
        Neg("NAME_COLLIDE", "เธอเป็นคนใจดีและเอื้อเฟื้อเผื่อแผ่"),
        Neg("NAME_COLLIDE", "ราคาทองคำปรับตัวสูงขึ้นอย่างต่อเนื่อง"),
        Neg("NAME_COLLIDE", "ขอให้ทุกท่านได้รับพรอันประเสริฐ"),
        Neg("NAME_COLLIDE", "ธนาคารแห่งนี้ให้บริการรวดเร็วมาก"),
        Neg("NAME_COLLIDE", "ความสามัคคีทำให้องค์กรมั่นคงยิ่งขึ้น"),
        Neg("NAME_COLLIDE", "เขาทำความดีด้วยความบริสุทธิ์ใจ"),

        // --- PLACE_COLLIDE: คำนำหน้าสถานที่ในบริบทที่ไม่ใช่สถานที่ ---
        Neg("PLACE_COLLIDE", "พื้นที่นี้ถูกกำหนดเป็นเขตปลอดอากร"),
        Neg("PLACE_COLLIDE", "โปรดรักษาความสะอาดภายในเขตชุมชน"),
        Neg("PLACE_COLLIDE", "ตำบลและอำเภอเป็นหน่วยปกครองท้องถิ่น"),
        Neg("PLACE_COLLIDE", "ทุกจังหวัดต้องส่งรายงานภายในสิ้นเดือน"),
        Neg("PLACE_COLLIDE", "บ้านเลขที่ดังกล่าวยังไม่ได้ระบุในเอกสาร"),
        Neg("PLACE_COLLIDE", "เขตอุตสาหกรรมแห่งใหม่กำลังก่อสร้าง"),

        // --- EMAIL_URL: สัญลักษณ์/โดเมนที่ไม่ใช่อีเมล ---
        Neg("EMAIL_URL", "อ่านรายละเอียดเพิ่มเติมได้จากเอกสารทั่วไป"),
        Neg("EMAIL_URL", "ใช้ตัวแปร a@b ในสูตรคำนวณของบทเรียน"),
        Neg("EMAIL_URL", "สัญลักษณ์ @ ใช้แทนคำว่าที่อยู่ในภาษาอังกฤษ"),
        Neg("EMAIL_URL", "โดเมน .com เป็นที่นิยมมากที่สุดในโลก"),
        Neg("EMAIL_URL", "อัตราส่วนภาพของจอคือ 16:9 มาตรฐาน"),

        // --- DATE_REF: วันที่ / เลขอ้างอิง / โค้ด ---
        Neg("DATE_REF", "กำหนดส่งงานวันที่ 12/05/2567 ก่อนเที่ยง"),
        Neg("DATE_REF", "อ้างอิงหนังสือเลขที่ กค 0405/ว 123"),
        Neg("DATE_REF", "รหัส OTP ของคุณคือ 482913 ใช้ได้ 5 นาที"),
        Neg("DATE_REF", "เลขพัสดุคือ EA123456789TH ตรวจสอบได้"),
        Neg("DATE_REF", "หมายเลขเที่ยวบิน TG 123 ออกเดินทางเช้านี้"),

        // --- BENIGN: ไม่มีตัวลวงเลย (sanity) ---
        Neg("BENIGN", "วันนี้อากาศดีมากเหมาะแก่การพักผ่อน"),
        Neg("BENIGN", "ระบบทำงานได้ตามปกติทุกอย่างเรียบร้อย"),
        Neg("BENIGN", "โปรดตรวจสอบรายงานประจำเดือนอีกครั้ง"),
        Neg("BENIGN", "การประชุมถูกเลื่อนออกไปเป็นสัปดาห์หน้า"),
        Neg("BENIGN", "ขอบคุณสำหรับความร่วมมือของทุกฝ่าย"),
    ]

    @Test func evaluateFalsePositives() async throws {
        let tagger = try? await ThaiNERTagger.load()
        let detector = PIIDetector()

        struct Acc { var total = 0, fpDocs = 0, falseDetections = 0 }
        var byCategory: [String: Acc] = [:]
        var typeCulprit: [String: Int] = [:]      // ชนิดที่ยิงพลาด (เฉพาะ hard-negative)
        var totalChars = 0, overMaskChars = 0
        var cleanTotal = 0, cleanFPDocs = 0, cleanFalse = 0
        var ambiguousTotal = 0, ambiguousDetectedDocs = 0
        var examples: [String] = []

        for s in Self.samples {
            let thai = tagger != nil ? await tagger!.tag(s.text) : []
            let entities = detector.detect(in: s.text, extraEntities: thai)

            if s.ambiguous {
                ambiguousTotal += 1
                if !entities.isEmpty { ambiguousDetectedDocs += 1 }
                continue
            }

            // hard-negative: ทุกการตรวจพบ = false positive
            cleanTotal += 1
            var acc = byCategory[s.category] ?? Acc()
            acc.total += 1
            if !entities.isEmpty {
                acc.fpDocs += 1; cleanFPDocs += 1
                if examples.count < 12 {
                    let dets = entities.map { "\($0.type):\($0.text)" }.joined(separator: ", ")
                    examples.append("[\(s.category)] \(s.text.prefix(38)) → \(dets)")
                }
            }
            acc.falseDetections += entities.count
            cleanFalse += entities.count
            for e in entities { typeCulprit[e.type, default: 0] += 1 }
            byCategory[s.category] = acc

            totalChars += s.text.count
            var masked = Set<Int>()
            for e in entities { masked.formUnion(Self.intRange(e.range, in: s.text)) }
            overMaskChars += masked.count
        }

        func pct(_ a: Int, _ b: Int) -> String { b > 0 ? String(format: "%5.1f%%", Double(a)/Double(b)*100) : "  n/a" }

        var out = "\n===== E5 — Negative / False-Positive Evaluation =====\n"
        out += "ThaiNER model loaded: \(tagger != nil)\n"
        out += "Hard-negatives: \(cleanTotal) · Format-collision (ambiguous): \(ambiguousTotal)\n\n"
        out += "| Category      | N  | FP docs | FP-doc rate↓ |\n"
        out += "|---------------|----|---------|--------------|\n"
        for key in ["NUMERIC", "NAME_COLLIDE", "PLACE_COLLIDE", "EMAIL_URL", "DATE_REF", "BENIGN"] {
            guard let a = byCategory[key] else { continue }
            out += "| \(key.padding(toLength: 13, withPad: " ", startingAt: 0)) | \(String(format: "%2d", a.total)) |   \(String(format: "%2d", a.fpDocs))    |     \(pct(a.fpDocs, a.total)) |\n"
        }
        out += "|---------------|----|---------|--------------|\n"
        out += "| OVERALL(clean)| \(String(format: "%2d", cleanTotal)) |   \(String(format: "%2d", cleanFPDocs))    |     \(pct(cleanFPDocs, cleanTotal)) |\n"
        out += "\nSpecificity (1 − FP-doc rate): \(pct(cleanTotal - cleanFPDocs, cleanTotal))\n"
        out += "False detections (hard-neg): \(cleanFalse)   Over-mask ratio: \(pct(overMaskChars, totalChars))\n"
        out += "Format-collision detected: \(ambiguousDetectedDocs)/\(ambiguousTotal) (\(pct(ambiguousDetectedDocs, ambiguousTotal))) — คาดว่าสูง (รูปแบบเหมือน PII จริง)\n"
        if !typeCulprit.isEmpty {
            let culprits = typeCulprit.sorted { $0.value > $1.value }.map { "\($0.key)×\($0.value)" }.joined(separator: ", ")
            out += "FP by type: \(culprits)\n"
        }
        if !examples.isEmpty {
            out += "\nFalse-positive examples:\n" + examples.map { "  • \($0)" }.joined(separator: "\n") + "\n"
        }
        out += "===================================================\n"
        print(out)

        #expect(!Self.samples.isEmpty)
    }

    private static func intRange(_ r: Range<String.Index>, in text: String) -> [Int] {
        let lo = text.distance(from: text.startIndex, to: r.lowerBound)
        let hi = text.distance(from: text.startIndex, to: r.upperBound)
        return Array(lo..<hi)
    }
}
