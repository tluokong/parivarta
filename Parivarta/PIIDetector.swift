//
//  PIIDetector.swift
//  Parivarta
//
//  ตรวจจับข้อมูลส่วนบุคคล (PII) ด้วยเครื่องมือ 3 ชั้นทำงานร่วมกัน:
//    1. Apple NaturalLanguage (NER) — ชื่อคน / สถานที่ / องค์กร
//    2. Custom Regex — เบอร์โทร / อีเมล / เลขบัตรประชาชน / IP
//    3. Watchlist — คำต้องห้ามที่กำหนดเอง + องค์กรที่เฝ้าระวัง
//
//  แทนที่ Presidio + spaCy + PyThaiNLP ของระบบเดิมด้วยเฟรมเวิร์กเนทีฟของ Apple
//

import Foundation
import NaturalLanguage

struct PIIDetector {
    /// รูปแบบ regex แบบกำหนดเอง (พอร์ตจากระบบ Python เดิม)
    private let customPatterns: [(type: String, regex: NSRegularExpression)]

    /// องค์กรที่เฝ้าระวังเสมอ
    private let organizationWatchlist = ["RMUTL", "สกมช.", "NCSA"]

    /// ชื่อจังหวัด/เมืองไทยที่พบบ่อย (คัดเฉพาะชื่อเฉพาะที่ไม่ค่อยชนคำทั่วไป เพื่อลด false positive)
    private let thaiPlaceNames = [
        "กรุงเทพมหานคร", "กรุงเทพฯ", "กรุงเทพ", "เชียงใหม่", "เชียงราย", "ภูเก็ต",
        "ชลบุรี", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ", "ขอนแก่น", "นครราชสีมา",
        "อุดรธานี", "หาดใหญ่", "พัทยา", "พระนครศรีอยุธยา", "อยุธยา", "นครศรีธรรมราช",
        "สงขลา", "สุราษฎร์ธานี", "ระยอง", "นครสวรรค์", "อุบลราชธานี", "เชียงตุง",
    ]

    /// regex จับชื่อบุคคลไทย (คำนำหน้า + ชื่อ) — เรียงคำนำหน้ายาวก่อนสั้นเพื่อให้จับได้เต็ม
    private let thaiNameRegex: NSRegularExpression

    init() {
        func rx(_ pattern: String) -> NSRegularExpression {
            try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        }
        // ลำดับสำคัญ: อีเมลก่อนเบอร์โทร เพื่อกันตัวเลขในอีเมลถูกจับเป็นเบอร์
        customPatterns = [
            ("EMAIL", rx(#"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"#)),
            // เลขบัตรประชาชนไทย 13 หลัก จัดกลุ่ม 1-4-5-2-1 รับตัวคั่น "-"/ช่องว่าง หรือไม่มี
            // (คนละ pattern กับเบอร์โทรที่จัดกลุ่ม 3-3-4 จึงไม่ทับกัน)
            ("THAI_ID", rx(#"\b\d[-\s]?\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d\b"#)),
            // กลุ่มกลาง \d{3,4} รองรับทั้ง 3-3-4 (เดิม) และ 2-4-4 (เช่น 06-0000-0301)
            ("PHONE_NUMBER", rx(#"(?:\+?66|0)[\s\-]?\d{1,2}[\s\-]?\d{3,4}[\s\-]?\d{3,4}"#)),
            // IP: ตรวจช่วง octet 0–255 (กัน false positive เช่น 192.168.0.300)
            ("IP_ADDRESS", rx(#"\b(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\b"#)),
            // ที่อยู่ไทย: opener (บ้านเลขที่/ที่อยู่/เลขที่/หมู่บ้าน) + โทเคนบ้านเลขที่
            // ตามด้วย "ฟิลด์" ที่ขึ้นต้นด้วยคำนำหน้า (ซอย/ถนน/หมู่/ตำบล/แขวง/อำเภอ/เขต/จังหวัด
            // รวมตัวย่อ ซ./ถ./ม./ต./อ./จ.) อย่างน้อย 1 ฟิลด์ แล้วต่อท้ายด้วยชื่อกรุงเทพฯ + รหัสไปรษณีย์
            // ทุกส่วนผูกขอบด้วยโทเคน (\S+) จึงไม่งาบข้ามคำ; ต้องมี ≥1 ฟิลด์ จึงไม่จับ opener ที่ลอยในเนื้อความ
            ("ADDRESS", rx(#"(?:บ้านเลขที่|ที่อยู่|เลขที่|หมู่บ้าน)\s*\S+(?:\s+(?:ซอย|ซ\.|ถนน|ถ\.|หมู่|ม\.|ตำบล|ต\.|แขวง|อำเภอ|อ\.|เขต|จังหวัด|จ\.)\s*\S+)+(?:\s+(?:กรุงเทพมหานคร|กรุงเทพฯ|กทม\.?))?(?:\s+\d{5})?"#)),
        ]

        // คำนำหน้า + ชื่อไทย 1–2 พยางค์/คำ (ช่วงอักขระไทยคือ U+0E00–U+0E7F)
        // ชื่อหลังคำนำหน้ารับได้ทั้งไทยและอังกฤษ (ล็อกชื่อ-สกุลให้เป็นสคริปต์เดียวกัน กัน over-capture คำกริยาไทยที่ตามมา)
        // lookbehind (?<![฀-๿]) เฉพาะ "คุณ" เท่านั้น เพื่อกันแมตช์กลางคำ เช่น "ขอบคุณ"/"ของคุณ"
        // (คำนำหน้าอื่นเช่น "นาย" ในภาษาไทยติดกับคำหน้าได้ปกติ เช่น "ว่านาย" จึงไม่ใส่ lookbehind)
        thaiNameRegex = rx(#"(?:นางสาว|นาง|นาย|น\.ส\.|ด\.ช\.|ด\.ญ\.|เด็กชาย|เด็กหญิง|(?<![฀-๿])คุณ)\s*(?:[฀-๿]+(?:\s+[฀-๿]+)?|[A-Za-z]+(?:\s+[A-Za-z]+)?)"#)
    }

    /// ตรวจจับ PII ทั้งหมดในข้อความ แล้วคืนรายการที่ไม่ซ้อนทับกัน เรียงตามตำแหน่ง
    /// - Parameter extraEntities: entity เพิ่มเติมจากภายนอก (เช่น Thai NER Core ML) นำมารวม dedup ด้วย
    func detect(in text: String, userWatchlist: [String] = [], extraEntities: [PIIEntity] = []) -> [PIIEntity] {
        var entities: [PIIEntity] = []
        entities += detectWatchlist(in: text, userWatchlist: userWatchlist)
        entities += detectRegex(in: text)
        entities += detectNamedEntities(in: text)
        entities += detectThaiHeuristics(in: text)
        entities += ThaiGazetteer.shared.detect(in: text)      // union คลังชื่อไทย → เพิ่ม recall
        entities += extraEntities
        return deduplicate(entities, in: text)
    }

    // MARK: - ชั้นที่ 1: NaturalLanguage NER

    private func detectNamedEntities(in text: String) -> [PIIEntity] {
        let tagger = NLTagger(tagSchemes: [.nameType])
        tagger.string = text
        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace, .joinNames]

        var results: [PIIEntity] = []
        tagger.enumerateTags(in: text.startIndex..<text.endIndex,
                             unit: .word,
                             scheme: .nameType,
                             options: options) { tag, range in
            guard let tag else { return true }
            let type: String?
            switch tag {
            case .personalName:     type = "PERSON"
            case .placeName:        type = "LOCATION"
            case .organizationName: type = "ORGANIZATION"
            default:                type = nil
            }
            if let type {
                results.append(PIIEntity(type: type,
                                         text: String(text[range]),
                                         range: range,
                                         confidence: 0.75,
                                         precision: 0.3))   // ชอบจับเกินหาง → ขอบเขตต่ำ
            }
            return true
        }
        return results
    }

    // MARK: - ชั้นที่ 2: Custom Regex

    private func detectRegex(in text: String) -> [PIIEntity] {
        var results: [PIIEntity] = []
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        for (type, regex) in customPatterns {
            for match in regex.matches(in: text, range: fullRange) {
                if let range = Range(match.range, in: text) {
                    results.append(PIIEntity(type: type,
                                             text: String(text[range]),
                                             range: range,
                                             confidence: 1.0,
                                             precision: 1.0))   // regex รูปแบบตายตัว → ขอบเขตแม่นสุด
                }
            }
        }
        return results
    }

    // MARK: - ชั้นเสริม: Heuristic ภาษาไทย

    /// ชดเชยจุดอ่อนของ NLTagger กับภาษาไทย: จับชื่อคน/สถานที่จากคำนำหน้า และชื่อจังหวัดที่รู้จัก
    private func detectThaiHeuristics(in text: String) -> [PIIEntity] {
        var results: [PIIEntity] = []
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)

        // จับเฉพาะ PERSON จากคำนำหน้า — LOCATION ใช้คลังจังหวัด (precise) แทน regex คำนำหน้าที่โลภ
        // (regex จังหวัด/เขต/ตำบล + [฀-๿]+ เคยงาบข้ามขอบคำ เพราะไทยไม่มีช่องว่าง → false positive สูง)
        let patterns: [(type: String, regex: NSRegularExpression)] = [
            ("PERSON", thaiNameRegex),
        ]
        for (type, regex) in patterns {
            for match in regex.matches(in: text, range: fullRange) {
                if let range = Range(match.range, in: text) {
                    results.append(PIIEntity(type: type,
                                             text: String(text[range]),
                                             range: range,
                                             confidence: 0.70,
                                             precision: 0.8))   // ผูกกับคำนำหน้า → ขอบเขตค่อนข้างแม่น
                }
            }
        }

        // ชื่อจังหวัด/เมืองไทยที่รู้จัก
        for place in thaiPlaceNames {
            var searchStart = text.startIndex
            while let found = text.range(of: place, range: searchStart..<text.endIndex) {
                results.append(PIIEntity(type: "LOCATION",
                                         text: place,
                                         range: found,
                                         confidence: 0.80,
                                         precision: 0.85))   // ชื่อเฉพาะตรงตัว
                searchStart = found.upperBound
            }
        }
        return results
    }

    // MARK: - ชั้นที่ 3: Watchlist

    private func detectWatchlist(in text: String, userWatchlist: [String]) -> [PIIEntity] {
        var results: [PIIEntity] = []
        let targets =
            organizationWatchlist.map { (word: $0, type: "ORGANIZATION") } +
            userWatchlist.map { (word: $0, type: "CUSTOM_KEYWORD") }

        for target in targets where !target.word.isEmpty {
            var searchStart = text.startIndex
            while let found = text.range(of: target.word, range: searchStart..<text.endIndex) {
                results.append(PIIEntity(type: target.type,
                                         text: target.word,
                                         range: found,
                                         confidence: 1.0,
                                         precision: 0.9))   // ตรงคำที่เฝ้าระวังพอดี
                searchStart = found.upperBound
            }
        }
        return results
    }

    // MARK: - รวมผล & ตัดส่วนที่ซ้อนทับ

    /// เก็บ entity โดยไล่จากตำแหน่งเริ่มต้นน้อยไปมาก แล้วตัดตัวที่ทับซ้อนออก
    /// เมื่อเริ่มตำแหน่งเดียวกัน: เลือก "ขอบเขตแม่นกว่า" ก่อน (precision) เพื่อกันสแปนที่จับเกินหาง
    /// (NER/NLTagger) มาเบียดสแปนที่กระชับกว่า (regex) — เสมอกันจึงค่อยเลือกยาวกว่า/มั่นใจกว่า
    private func deduplicate(_ entities: [PIIEntity], in text: String) -> [PIIEntity] {
        let sorted = entities.sorted { lhs, rhs in
            if lhs.range.lowerBound != rhs.range.lowerBound {
                return lhs.range.lowerBound < rhs.range.lowerBound
            }
            if lhs.precision != rhs.precision {
                return lhs.precision > rhs.precision                 // แม่นกว่ามาก่อน
            }
            if lhs.range.upperBound != rhs.range.upperBound {
                return lhs.range.upperBound > rhs.range.upperBound   // ยาวกว่ามาก่อน
            }
            return lhs.confidence > rhs.confidence
        }

        var kept: [PIIEntity] = []
        for entity in sorted {
            let overlaps = kept.contains { existing in
                !(entity.range.upperBound <= existing.range.lowerBound ||
                  entity.range.lowerBound >= existing.range.upperBound)
            }
            if !overlaps {
                kept.append(entity)
            }
        }
        let ordered = kept.sorted { $0.range.lowerBound < $1.range.lowerBound }

        // รวมสแปน PERSON ที่ติดกันโดยคั่นด้วย "ช่องว่างล้วน" ให้เป็นสแปนเดียว
        // (ชื่อ-สกุลที่ถูกตัดเป็นคนละสแปน เช่น "ณัฐดนัย"+"ธนกฤต" → "ณัฐดนัย ธนกฤต")
        // → ปิดช่องว่างที่ไม่ถูก mask + ได้รหัสแทนเดียวต่อ 1 คน; ไม่แตะชนิดอื่นเพื่อกันรวมคนละรายการ
        var merged: [PIIEntity] = []
        for entity in ordered {
            if var last = merged.last, last.type == "PERSON", entity.type == "PERSON",
               last.range.upperBound <= entity.range.lowerBound,
               text[last.range.upperBound..<entity.range.lowerBound].allSatisfy(\.isWhitespace) {
                let range = last.range.lowerBound..<entity.range.upperBound
                last = PIIEntity(type: "PERSON", text: String(text[range]), range: range,
                                 confidence: max(last.confidence, entity.confidence),
                                 precision: max(last.precision, entity.precision))
                merged[merged.count - 1] = last
            } else {
                merged.append(entity)
            }
        }
        return merged
    }
}
