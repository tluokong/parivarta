//
//  ThaiGazetteer.swift
//  Parivarta
//
//  ตัวจับชื่อคนไทยแบบ gazetteer (คลังรายชื่อ) — เสริม (union) กับ Core ML NER
//  เพื่อเพิ่ม recall (ลด PII หลุด) โดยไม่แตะโมเดล
//
//  วิธี: ตัดคำแบบ newmm (greedy longest-match ด้วย thai_words.txt) แล้วจับคำที่อยู่ใน
//  คลังชื่อ (thai_person_names.txt) — ตัดคำก่อนเพื่อกัน false positive เช่น "ธนา" ใน "ธนาคาร"
//  ยอมรับเป็น PERSON เมื่อ: ชื่อติดกัน ≥ 2 คำ (ชื่อ+สกุล) หรือ ชื่อเดี่ยวที่มีคำนำหน้า/`ชื่อ` นำ
//

import Foundation

struct ThaiGazetteer {
    static let shared = ThaiGazetteer()

    /// โหมดการทำงาน — ใช้เปรียบเทียบใน eval (ปกติคือ .full)
    ///   .off      → ปิด gazetteer (baseline)
    ///   .pass1    → เฉพาะ Pass 1 (run ชื่อ ≥2 คำ หรือ ชื่อเดี่ยว+cue) = Exp 2
    ///   .full     → Pass 1 + Pass 2 (จับชื่อเขียนติดกัน)            = Exp 2b (ค่าเริ่มต้น)
    enum Mode { case off, pass1, full }
    /// TaskLocal เพื่อให้ eval แต่ละชุด (ที่รันขนานกัน) ตั้งโหมดในขอบเขต task ของตัวเอง
    /// ไม่รบกวนกัน (ค่าปกติ .full = พฤติกรรมจริงของแอป) — ตั้งผ่าน `ThaiGazetteer.$mode.withValue(_:)`
    @TaskLocal static var mode: Mode = .full

    private let words: Set<String>            // พจนานุกรมตัดคำ (newmm)
    private let names: Set<String>            // คลังชื่อ-นามสกุลไทย
    private let maxWordLen: Int
    private let cues: Set<String> = ["นาย", "นาง", "นางสาว", "คุณ", "ชื่อ", "น.ส.", "เด็กชาย", "เด็กหญิง"]

    init() {
        words = ThaiGazetteer.loadSet("thai_words")
        names = ThaiGazetteer.loadSet("thai_person_names")
        maxWordLen = words.map(\.count).max() ?? 1
    }

    private static func loadSet(_ resource: String) -> Set<String> {
        guard let url = Bundle.main.url(forResource: resource, withExtension: "txt"),
              let raw = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        var set = Set<String>()
        for line in raw.split(separator: "\n") {
            let w = line.trimmingCharacters(in: .whitespaces)
            if !w.isEmpty { set.insert(w) }
        }
        return set
    }

    /// จับ PERSON จากคลังชื่อ (คืน [] ถ้าโหลดคลังไม่ได้)
    func detect(in text: String) -> [PIIEntity] {
        guard Self.mode != .off, !names.isEmpty, !words.isEmpty else { return [] }

        // 1) ตัดคำ (greedy newmm) เก็บเฉพาะคำ "เนื้อหา" (ไม่รวมช่องว่าง) พร้อมช่วงตัวอักษร + เป็นชื่อไหม
        let content = segmentContentWords(text)
        if content.isEmpty { return [] }

        var entities: [PIIEntity] = []

        // Pass 1: run ของคำที่เป็นชื่อติดกัน (≥2 คำ) หรือชื่อเดี่ยวที่มีคำนำหน้า/`ชื่อ` นำ
        //         (เช็ก cue จากตัวอักษรก่อนหน้าโดยตรง — ทนต่อการตัดคำที่ไม่ลงตัว)
        var i = 0
        while i < content.count {
            guard content[i].isName else { i += 1; continue }
            var j = i
            while j + 1 < content.count && content[j + 1].isName { j += 1 }
            let runLength = j - i + 1
            if runLength >= 2 || hasCue(before: content[i].range.lowerBound, in: text) {
                let range = content[i].range.lowerBound..<content[j].range.upperBound
                entities.append(PIIEntity(type: "PERSON", text: String(text[range]), range: range, confidence: 0.8, precision: 0.6))
            }
            i = j + 1
        }

        // Pass 2: ชื่อเขียนติดกันที่ newmm ตัดแตก (เช่น "สมหญิง" → "สม"+"หญิง")
        //         รวมคำ "ติดกันไม่มีช่องว่าง" ≥2 คำ ที่ต่อกันแล้วเป็นชื่อในคลัง (ยาว ≥4)
        //         — align ขอบคำ จึงไม่จับเศษกลางคำ (เช่น "ธนา" ใน "ธนาคาร" ซึ่งเป็นคำเดียว)
        guard Self.mode == .full else { return entities }
        var p = 0
        while p < content.count {
            var joined = content[p].surface
            var lastUpper = content[p].range.upperBound
            var matchedEnd = -1
            var q = p + 1
            while q < content.count, content[q].range.lowerBound == lastUpper, q - p <= 4 {
                joined += content[q].surface
                lastUpper = content[q].range.upperBound
                if joined.count >= 4, names.contains(joined) { matchedEnd = q }   // greedy longest
                q += 1
            }
            if matchedEnd >= 0 {
                let range = content[p].range.lowerBound..<content[matchedEnd].range.upperBound
                entities.append(PIIEntity(type: "PERSON", text: String(text[range]), range: range, confidence: 0.8, precision: 0.6))
                p = matchedEnd + 1
            } else {
                p += 1
            }
        }
        return entities
    }

    /// ตรวจว่าตัวอักษรก่อนตำแหน่ง (ตัดช่องว่างท้าย) ลงท้ายด้วยคำนำหน้า/`ชื่อ` ไหม
    private func hasCue(before index: String.Index, in text: String) -> Bool {
        let prefix = text[..<index].trimmingCharacters(in: .whitespaces)
        return cues.contains { prefix.hasSuffix($0) }
    }

    // MARK: - newmm greedy segmentation (เฉพาะคำเนื้อหา)

    private func segmentContentWords(_ text: String) -> [(surface: String, range: Range<String.Index>, isName: Bool)] {
        let chars = Array(text)
        let idx = Array(text.indices) + [text.endIndex]
        func isThai(_ c: Character) -> Bool {
            guard let s = c.unicodeScalars.first else { return false }
            return s.value >= 0x0E00 && s.value <= 0x0E7F
        }

        var result: [(String, Range<String.Index>, Bool)] = []
        var p = 0
        let n = chars.count
        while p < n {
            let c = chars[p]
            if c.isWhitespace {
                p += 1                                          // ข้ามช่องว่าง (แต่ range ยัง "เชื่อม" ผ่านการเว้นคำในการรวม)
            } else if isThai(c) {
                var matched = 0
                var L = min(maxWordLen, n - p)
                while L >= 2 {
                    if isThai(chars[p + L - 1]), words.contains(String(chars[p..<p + L])) { matched = L; break }
                    L -= 1
                }
                let len = matched > 0 ? matched : 1
                let surface = String(chars[p..<p + len])
                result.append((surface, idx[p]..<idx[p + len], names.contains(surface)))
                p += len
            } else {
                var j = p + 1
                while j < n, !chars[j].isWhitespace, !isThai(chars[j]) { j += 1 }
                result.append((String(chars[p..<j]), idx[p]..<idx[j], false))
                p = j
            }
        }
        return result
    }
}
