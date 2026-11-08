//
//  TextPreprocessor.swift
//  Parivarta
//
//  ทำความสะอาดข้อความก่อนตรวจจับ PII (เทียบเท่า CodeMixedPreprocessor เดิม)
//

import Foundation

/// กรอง URL, @mention และ #hashtag ออกจากข้อความ พร้อมจัดระเบียบช่องว่าง
/// (กฎ @mention ผูกขอบซ้ายไว้ จึงไม่ไปกินอีเมล — ดูคำอธิบายที่ตัวแปร `mention`)
struct TextPreprocessor {
    private let url = try! NSRegularExpression(pattern: #"https?://[^\s]+"#, options: [.caseInsensitive])
    // "@" ต้องไม่ถูกนำหน้าด้วยอักขระที่เป็นส่วนของ local-part ของอีเมลได้
    // ของเดิม #"@\w+"# กินส่วนโดเมนของอีเมลไปก่อนที่ PIIDetector จะเห็น
    // ("a@b.com" → "a[MENTION]") ทำให้อีเมลไม่เคยถูกแทนด้วยรหัสแทนในเส้นทางใช้งานจริง
    // — ชุดประเมิน (Dataset B/C/D) มองไม่เห็นบั๊กนี้เพราะให้คะแนนบนข้อความดิบที่ยังไม่ผ่าน preprocess
    // ใช้ช่วง ASCII ไม่ใช่ \w เพื่อให้ mention ที่เขียนติดคำไทย (เช่น "ทัก@ScamBoss") ยังถูกกรอง
    private let mention = try! NSRegularExpression(pattern: #"(?<![A-Za-z0-9_.+%-])@\w+"#)
    private let hashtag = try! NSRegularExpression(pattern: #"#\w+"#)

    /// แปลง URL/mention/hashtag เป็น placeholder และยุบช่องว่างซ้ำ
    func preprocess(_ text: String) -> String {
        var result = text
        result = replaceMatches(url, in: result, with: "[URL]")
        result = replaceMatches(mention, in: result, with: "[MENTION]")
        result = replaceMatches(hashtag, in: result, with: "[HASHTAG]")
        result = result.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func replaceMatches(_ regex: NSRegularExpression, in text: String, with replacement: String) -> String {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.stringByReplacingMatches(in: text, range: range, withTemplate: replacement)
    }
}
