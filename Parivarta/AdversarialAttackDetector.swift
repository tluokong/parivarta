//
//  AdversarialAttackDetector.swift
//  Parivarta
//
//  เกราะป้องกันชั้นนอก: ตรวจจับความพยายามโจมตีแบบ Prompt Injection / Jailbreak
//

import Foundation

/// ตรวจจับคำสั่งที่พยายามหลอกให้ AI ละเมิดกฎ (เช่น "ignore all previous instructions")
struct AdversarialAttackDetector {
    /// รูปแบบ (regex) ที่บ่งชี้การโจมตี — พอร์ตและขยายจากระบบ Python เดิม
    private let patterns: [String] = [
        #"describing\.\s*\+\s+similarly"#,
        #"write\s+opposite\s+contents"#,
        #"ignore\s+(all\s+|any\s+)?previous"#,
        #"ignore\s+(all\s+)?prior\s+instructions"#,
        #"disregard\s+(all\s+)?(previous|prior)"#,
        #"bypass\s+security"#,
        #"reveal\s+(your\s+)?system\s+prompt"#,
        #"tell\s+me\s+your\s+system\s+prompt"#,
        #"jailbreak"#,
        #"developer\s+mode"#,
    ]

    private let compiled: [NSRegularExpression]

    init() {
        compiled = patterns.compactMap {
            try? NSRegularExpression(pattern: $0, options: [.caseInsensitive])
        }
    }

    /// ตรวจสอบข้อความ คืนค่าว่าปลอดภัยหรือไม่ พร้อมรายการรูปแบบที่ตรวจพบ
    func inspect(_ text: String) -> (isSafe: Bool, matched: [String]) {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        var matched: [String] = []
        for (index, regex) in compiled.enumerated() {
            if regex.firstMatch(in: text, range: range) != nil {
                matched.append(patterns[index])
            }
        }
        return (matched.isEmpty, matched)
    }
}
