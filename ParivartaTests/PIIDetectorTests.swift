//
//  PIIDetectorTests.swift
//  ParivartaTests
//
//  ทดสอบการตรวจจับ PII: regex, watchlist, คำนำหน้าไทย และการไม่ซ้อนทับ (dedup)
//  หมายเหตุ: ไม่ใช้ Thai NER (Core ML) — ทดสอบเฉพาะ regex/NLTagger/heuristic ที่รันเร็ว
//

import Testing
@testable import Parivarta

@MainActor
struct PIIDetectorTests {
    let detector = PIIDetector()

    /// หา entity ตามชนิดที่ระบุ
    private func texts(_ entities: [PIIEntity], ofType type: String) -> [String] {
        entities.filter { $0.type == type }.map(\.text)
    }

    @Test func detectsThaiPhoneNumber() {
        let e = detector.detect(in: "ติดต่อเบอร์ 02-123-4567 ด่วน")
        #expect(texts(e, ofType: "PHONE_NUMBER").contains("02-123-4567"))
    }

    @Test func detectsInternationalPhone() {
        let e = detector.detect(in: "โทร +6689-123-4567")
        #expect(texts(e, ofType: "PHONE_NUMBER").contains("+6689-123-4567"))
    }

    @Test func detectsEmail() {
        let e = detector.detect(in: "ส่งไปที่ support@fake-bank-alert.com นะ")
        #expect(texts(e, ofType: "EMAIL").contains("support@fake-bank-alert.com"))
    }

    @Test func detectsThaiNationalID() {
        let e = detector.detect(in: "เลขบัตร 1234567890123 นี้")
        #expect(texts(e, ofType: "THAI_ID").contains("1234567890123"))
    }

    @Test func detectsIPAddress() {
        let e = detector.detect(in: "จาก IP 192.168.1.100 เข้ามา")
        #expect(texts(e, ofType: "IP_ADDRESS").contains("192.168.1.100"))
    }

    @Test func detectsThaiPersonByHonorific() {
        let e = detector.detect(in: "เรียนคุณ สมชาย เลิศวิสัย")
        let persons = texts(e, ofType: "PERSON")
        #expect(persons.contains { $0.contains("สมชาย") })
    }

    @Test func detectsCustomWatchlistKeyword() {
        let e = detector.detect(in: "โครงการ ProjectX เป็นความลับ", userWatchlist: ["ProjectX"])
        #expect(texts(e, ofType: "CUSTOM_KEYWORD").contains("ProjectX"))
    }

    @Test func resultsHaveNoOverlappingRanges() {
        let e = detector.detect(in: "คุณ สมชาย เบอร์ 02-123-4567 อีเมล a@b.com IP 10.0.0.1")
        let sorted = e.sorted { $0.range.lowerBound < $1.range.lowerBound }
        for i in 1..<max(sorted.count, 1) where sorted.count > 1 {
            // entity ถัดไปต้องเริ่มหลัง entity ก่อนหน้าจบ (ไม่ทับซ้อน)
            #expect(sorted[i].range.lowerBound >= sorted[i - 1].range.upperBound)
        }
    }

    /// เส้นทางใช้งานจริงคือ preprocess → detect ดังนั้นอีเมลต้องรอดจากขั้น normalize มาให้ตรวจได้
    @Test func detectsEmailAfterPreprocessing() {
        let cleaned = TextPreprocessor().preprocess("ติดต่อ support@fake-bank-alert.com หรือ @ScamBoss")
        let e = detector.detect(in: cleaned)
        #expect(texts(e, ofType: "EMAIL").contains("support@fake-bank-alert.com"))
        #expect(cleaned.contains("[MENTION]"))
    }

    @Test func returnsEmptyForNonPII() {
        let e = detector.detect(in: "วันนี้อากาศดีมาก")
        #expect(texts(e, ofType: "PHONE_NUMBER").isEmpty)
        #expect(texts(e, ofType: "EMAIL").isEmpty)
    }
}
