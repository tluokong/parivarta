//
//  TextPreprocessorTests.swift
//  ParivartaTests
//
//  ทดสอบการกรอง URL / mention / hashtag ก่อนตรวจ PII
//

import Testing
@testable import Parivarta

@MainActor
struct TextPreprocessorTests {
    let pre = TextPreprocessor()

    @Test func redactsURL() {
        let out = pre.preprocess("ดูเว็บ https://fake-crypto-wallet.com ด่วน")
        #expect(out.contains("[URL]"))
        #expect(out.contains("fake-crypto-wallet.com") == false)
    }

    @Test func redactsMentionAndHashtag() {
        let out = pre.preprocess("ผู้ใช้ @ScamBoss โพสต์ #crypto วันนี้")
        #expect(out.contains("[MENTION]"))
        #expect(out.contains("[HASHTAG]"))
        #expect(out.contains("@ScamBoss") == false)
    }

    @Test func collapsesWhitespace() {
        let out = pre.preprocess("a    b\n\nc")
        #expect(out == "a b c")
    }

    @Test func keepsPlainThaiText() {
        let out = pre.preprocess("สวัสดีครับ")
        #expect(out == "สวัสดีครับ")
    }

    /// กฎ @mention ต้องไม่กินส่วนโดเมนของอีเมล
    /// (บั๊กเดิม: "a@b.com" → "a[MENTION]" ทำให้ PIIDetector ไม่เคยเห็นอีเมลเลย
    ///  และชุดประเมินมองไม่เห็นเพราะให้คะแนนบนข้อความดิบที่ยังไม่ผ่าน preprocess)
    @Test func keepsEmailIntact() {
        let out = pre.preprocess("ส่งไปที่ support@fake-bank-alert.com นะ")
        #expect(out.contains("support@fake-bank-alert.com"))
        #expect(out.contains("[MENTION]") == false)
    }

    @Test func keepsEmailWithDotsAndPlusInLocalPart() {
        let out = pre.preprocess("อีเมล first.last+tag@example.co.th ใช้ได้")
        #expect(out.contains("first.last+tag@example.co.th"))
        #expect(out.contains("[MENTION]") == false)
    }

    @Test func stillRedactsMentionAfterPunctuation() {
        let out = pre.preprocess("ผู้ใช้ (@ScamBoss) และ @Another โพสต์")
        #expect(out.contains("@ScamBoss") == false)
        #expect(out.contains("@Another") == false)
    }

    /// mention ที่เขียนติดกับคำไทยต้องยังถูกกรอง (ภาษาไทยไม่เว้นวรรคระหว่างคำ)
    @Test func redactsMentionAttachedToThaiWord() {
        let out = pre.preprocess("ทัก@ScamBoss ด่วน")
        #expect(out.contains("[MENTION]"))
        #expect(out.contains("@ScamBoss") == false)
    }
}
