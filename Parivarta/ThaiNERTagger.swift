//
//  ThaiNERTagger.swift
//  Parivarta
//
//  ตัวจับ Named Entity ภาษาไทยด้วยโมเดล Core ML (WangchanBERTa NER ที่แปลงจาก
//  pythainlp/thainer-corpus-v2-base-model) ทำงาน on-device ทั้งหมด
//
//  ขั้นตอน:
//    1) tokenize ข้อความด้วย Unigram/SentencePiece (swift-transformers)
//    2) เข้ารหัสเป็น input_ids + attention_mask (padding คงที่ 512)
//    3) รัน Core ML ได้ logits [1, 512, 36]
//    4) argmax → ป้าย BIO ต่อ token
//    5) map token กลับเป็นช่วงตัวอักษรจริง แล้วรวมเป็น entity (PERSON/LOCATION/ORGANIZATION)
//

import Foundation
import CoreML
import Tokenizers
import Hub

actor ThaiNERTagger {
    private let model: MLModel
    private let tokenizer: any Tokenizer
    private let labels: [String]
    private let padTokenId: Int
    private let maxLength = 512

    /// ป้ายของโมเดล → ชนิด PII ที่เราสนใจ (เบอร์/อีเมล/วันที่ ปล่อยให้ regex จัดการเพราะแม่นกว่า)
    private static let typeMap: [String: String] = [
        "PERSON": "PERSON",
        "LOCATION": "LOCATION",
        "ORGANIZATION": "ORGANIZATION",
    ]

    /// เกณฑ์ความมั่นใจขั้นต่ำ (softmax) ที่จะนับ token เป็นส่วนหนึ่งของ entity
    /// ตั้งต่ำ เพราะชื่อไทยเขียนติดกันมัก prob ต่ำ (0.4–0.6) — สูงไปจะตัดชื่อจริงเป็นชิ้น
    private static let minTokenProb = 0.35

    private init(model: MLModel, tokenizer: any Tokenizer, labels: [String], padTokenId: Int) {
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.padTokenId = padTokenId
    }

    // MARK: - โหลดโมเดล + tokenizer (ครั้งเดียว)

    static func load() async throws -> ThaiNERTagger {
        debugLog("load() started")

        // Core ML model (Xcode คอมไพล์ .mlpackage → .mlmodelc ให้อัตโนมัติ)
        guard let modelURL = Bundle.main.url(forResource: "ThaiNER", withExtension: "mlmodelc") else {
            debugLog("ERROR: ThaiNER.mlmodelc not found in bundle")
            throw NSError(domain: "ThaiNER", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "ไม่พบ ThaiNER.mlmodelc ใน bundle"])
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        debugLog("MLModel loaded OK")

        // tokenizer: โหลดจาก Config โดยตรง (offline, ไม่แตะ hub)
        let tokenizer = try loadTokenizer()
        debugLog("tokenizer loaded OK")

        // labels
        guard let labelsURL = Bundle.main.url(forResource: "thai_ner_labels", withExtension: "json"),
              let labelsData = try? Data(contentsOf: labelsURL),
              let labels = try? JSONDecoder().decode([String].self, from: labelsData) else {
            debugLog("ERROR: cannot load thai_ner_labels.json")
            throw NSError(domain: "ThaiNER", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "โหลด thai_ner_labels.json ไม่ได้"])
        }

        let padId = tokenizer.convertTokenToId("<pad>") ?? 1
        debugLog("load() done. labels=\(labels.count) padId=\(padId)")
        return ThaiNERTagger(model: model, tokenizer: tokenizer, labels: labels, padTokenId: padId)
    }

    /// โหลด tokenizer แบบ synchronous จากไฟล์ JSON ที่ bundle ไว้ (ไม่แตะ network/hub)
    private static func loadTokenizer() throws -> any Tokenizer {
        guard let dataURL = Bundle.main.url(forResource: "thai_ner_tokenizer", withExtension: "json"),
              let configURL = Bundle.main.url(forResource: "thai_ner_tokenizer_config", withExtension: "json") else {
            throw NSError(domain: "ThaiNER", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "ไม่พบไฟล์ tokenizer ใน bundle"])
        }
        let tokenizerData = try JSONDecoder().decode(Config.self, from: Data(contentsOf: dataURL))
        let tokenizerConfig = try JSONDecoder().decode(Config.self, from: Data(contentsOf: configURL))
        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: false)
    }

    // MARK: - ตรวจจับ entity

    /// คืน entity ภาษาไทย (PERSON/LOCATION/ORGANIZATION) พร้อมช่วงตัวอักษรใน `text`
    func tag(_ text: String) -> [PIIEntity] {
        let pieces = tokenizer.tokenize(text: text)           // subword พร้อม ▁ (ไม่รวม special)
        let ids = tokenizer.encode(text: text)                // (อาจ) <s> + pieces + </s>
        Self.debugLog("tag() text=\(text.prefix(40)) pieces=\(pieces.count) ids=\(ids.count) idsHead=\(ids.prefix(6))")
        if pieces.isEmpty || ids.isEmpty { return [] }

        guard let logits = try? predict(inputIds: ids) else {
            Self.debugLog("tag() predict FAILED")
            return []
        }

        // จำนวน special token นำหน้า: 1 ถ้า encode เติม <s> ไว้ ไม่งั้น 0
        let leading = (tokenizer.bosTokenId != nil && ids.first == tokenizer.bosTokenId) ? 1 : 0
        let usableCount = min(pieces.count, maxLength - leading - 1)

        // 1) หาช่วงตัวอักษรจริงของแต่ละ token ด้วยการไล่จับ surface ตามลำดับ
        let ranges = characterRanges(for: pieces, in: text)

        // 2) รวม token ที่เป็น entity ชนิดใดก็ได้ (PERSON/LOCATION/ORGANIZATION) ที่ติดกันเป็นก้อนเดียว
        //    หยุดเฉพาะเมื่อเจอ O/prob ต่ำ — ทนต่อการที่โมเดลสลับชนิดกลางชื่อ และลด over-capture คำ O ข้างเคียง
        var entities: [PIIEntity] = []
        var curStart: String.Index?
        var curEnd: String.Index?
        var typeVotes: [String: Int] = [:]

        func flush() {
            if let s = curStart, let e = curEnd, s < e {
                let type = typeVotes.max { $0.value < $1.value }?.key ?? "PERSON"
                entities.append(PIIEntity(type: type, text: String(text[s..<e]), range: s..<e, confidence: 0.9, precision: 0.4))
            }
            curStart = nil; curEnd = nil; typeVotes = [:]
        }

        for k in 0..<usableCount {
            guard let range = ranges[k] else { continue }      // metaspace/ไม่พบ → ข้าม ไม่ตัดก้อน
            let (labelIdx, prob) = labelProbability(logits, position: leading + k)
            let (_, rawType) = splitBIO(labels[labelIdx])
            let mapped = rawType.flatMap { Self.typeMap[$0] }
            Self.debugLog("  tok[\(k)]='\(pieces[k])' -> \(labels[labelIdx]) p=\(String(format: "%.2f", prob))")

            if let mapped, prob >= Self.minTokenProb {
                if curStart == nil { curStart = range.lowerBound }
                curEnd = range.upperBound
                typeVotes[mapped, default: 0] += 1
            } else {
                flush()                                        // O หรือ prob ต่ำ → จบก้อน
            }
        }
        flush()
        Self.debugLog("tag() entities=\(entities.count) \(entities.map { "\($0.type):\($0.text)" })")
        return entities
    }

    // MARK: - Core ML inference

    private func predict(inputIds: [Int]) throws -> MLMultiArray {
        let idsArray = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        for i in 0..<maxLength {
            let hasToken = i < inputIds.count
            idsArray[i] = NSNumber(value: Int32(hasToken ? inputIds[i] : padTokenId))
            maskArray[i] = NSNumber(value: Int32(hasToken ? 1 : 0))
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": idsArray,
            "attention_mask": maskArray,
        ])
        let output = try model.prediction(from: input)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "ThaiNER", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "ไม่พบ output 'logits'"])
        }
        return logits
    }

    /// logits shape = [1, 512, labelCount] → flat index ของ [0, position, j] คือ position*labelCount + j
    /// คืน (index ของป้ายที่มั่นใจสุด, ความน่าจะเป็น softmax ของป้ายนั้น)
    private func labelProbability(_ logits: MLMultiArray, position: Int) -> (label: Int, prob: Double) {
        let labelCount = labels.count
        let base = position * labelCount
        var values = [Double](repeating: 0, count: labelCount)
        var maxLogit = -Double.greatestFiniteMagnitude
        var bestIndex = 0
        for j in 0..<labelCount {
            let value = logits[base + j].doubleValue
            values[j] = value
            if value > maxLogit {
                maxLogit = value
                bestIndex = j
            }
        }
        var sumExp = 0.0
        for j in 0..<labelCount {
            sumExp += exp(values[j] - maxLogit)
        }
        let prob = sumExp > 0 ? 1.0 / sumExp : 0.0   // softmax ของป้ายที่มี logit สูงสุด = exp(0)/Σ
        return (bestIndex, prob)
    }

    // MARK: - Helpers

    /// ไล่จับ surface ของแต่ละ subword ในข้อความจริงตามลำดับ เพื่อได้ช่วงตัวอักษร
    private func characterRanges(for pieces: [String], in text: String) -> [Range<String.Index>?] {
        var result: [Range<String.Index>?] = []
        result.reserveCapacity(pieces.count)
        var cursor = text.startIndex
        for piece in pieces {
            let surface = piece.replacingOccurrences(of: "▁", with: "")
            if surface.isEmpty {
                result.append(nil)                              // token ช่องว่าง (metaspace)
                continue
            }
            if let found = text.range(of: surface, options: [.caseInsensitive], range: cursor..<text.endIndex) {
                result.append(found)
                cursor = found.upperBound
            } else {
                result.append(nil)                              // <unk>/ผลจาก normalize ที่หาไม่เจอ
            }
        }
        return result
    }

    /// แยกป้าย BIO เช่น "B-PERSON" → ("B", "PERSON"), "O" → ("O", nil)
    private func splitBIO(_ label: String) -> (prefix: String, type: String?) {
        if label == "O" { return ("O", nil) }
        if let dash = label.firstIndex(of: "-") {
            return (String(label[..<dash]), String(label[label.index(after: dash)...]))
        }
        return ("O", nil)
    }

    // MARK: - Debug log (เขียนลงไฟล์ใน container เพื่อวินิจฉัย)

    /// เปิด/ปิดการเขียน debug log ลงไฟล์ (ตั้งเป็น true เมื่อต้องวินิจฉัยปัญหา)
    nonisolated static let debugEnabled = false

    nonisolated static func debugLog(_ message: String) {
        guard debugEnabled else { return }
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = base.appendingPathComponent("Parivarta")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("thainer_debug.log")
        let line = "[\(Date())] \(message)\n"
        guard let data = line.data(using: .utf8) else { return }
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: url)
        }
    }
}
