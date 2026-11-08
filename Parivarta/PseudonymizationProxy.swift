//
//  PseudonymizationProxy.swift
//  Parivarta
//
//  ตัวประสานหลักของระบบ ทำงาน 5 ขั้นตอน:
//    1) ตรวจภัยคุกคาม (Prompt Injection) → บล็อกถ้าไม่ปลอดภัย
//    2) ทำความสะอาดข้อความ
//    3) ตรวจจับ & แทนที่ PII ด้วยรหัส (masking)
//    4) ส่งข้อความที่ mask แล้วไปให้ LLM
//    5) ถอดหน้ากาก (de-masking) คำตอบกลับเป็นข้อมูลจริง
//

import Foundation

final class PseudonymizationProxy {
    private let attackDetector = AdversarialAttackDetector()
    private let preprocessor = TextPreprocessor()
    private let piiDetector = PIIDetector()
    private let vault: MappingVault
    private let llm = LLMClient()

    init(vault: MappingVault = MappingVault()) {
        self.vault = vault
    }

    // Thai NER (Core ML) โหลดครั้งเดียวแบบ lazy — ถ้าโหลดไม่ได้จะใช้เฉพาะ regex/NLTagger/heuristic
    private var thaiTagger: ThaiNERTagger?
    private var thaiTaggerLoaded = false

    func process(prompt: String,
                 watchlist: [String],
                 settings: LLMSettings,
                 onStage: @escaping (ProxyStage) -> Void = { _ in },
                 onDelta: @escaping (String) -> Void = { _ in }) async -> ProxyResult {
        // 1) เกราะป้องกัน Prompt Injection
        let safety = attackDetector.inspect(prompt)
        guard safety.isSafe else {
            return .blocked(reason: "ตรวจพบการโจมตีแบบ Prompt Injection!",
                            patterns: safety.matched)
        }

        // 2) ทำความสะอาดข้อความ
        let cleaned = preprocessor.preprocess(prompt)

        // 3) ตรวจจับ PII — รวมผลจาก Thai NER (Core ML) กับ regex/NLTagger/heuristic
        onStage(.analyzing)
        let thaiEntities = await thaiNEREntities(in: cleaned)
        let entities = piiDetector.detect(in: cleaned, userWatchlist: watchlist, extraEntities: thaiEntities)
        let spans = entities.map { DetectedSpan(type: $0.type, text: $0.text, range: $0.range) }

        // 4–5) mask → LLM (stream) → de-mask
        return await complete(original: prompt, cleaned: cleaned, spans: spans,
                              settings: settings, onStage: onStage, onDelta: onDelta)
    }

    /// เฟส 1 ของ "ตรวจก่อนส่ง": ตรวจภัยคุกคาม + ทำความสะอาด + ตรวจจับ PII (ยังไม่เรียก LLM)
    func analyze(prompt: String, watchlist: [String]) async -> AnalysisResult {
        let safety = attackDetector.inspect(prompt)
        guard safety.isSafe else {
            return .blocked(reason: "ตรวจพบการโจมตีแบบ Prompt Injection!",
                            patterns: safety.matched)
        }
        let cleaned = preprocessor.preprocess(prompt)
        let thaiEntities = await thaiNEREntities(in: cleaned)
        let entities = piiDetector.detect(in: cleaned, userWatchlist: watchlist, extraEntities: thaiEntities)
        let spans = entities.map { DetectedSpan(type: $0.type, text: $0.text, range: $0.range) }
        return .safe(cleaned: cleaned, spans: spans)
    }

    /// เฟส 2 ของ "ตรวจก่อนส่ง": mask เฉพาะช่วงที่ผู้ใช้อนุมัติ → LLM → de-mask
    func send(cleaned: String,
              original: String,
              approvedSpans: [DetectedSpan],
              settings: LLMSettings,
              onStage: @escaping (ProxyStage) -> Void = { _ in },
              onDelta: @escaping (String) -> Void = { _ in }) async -> ProxyResult {
        let enabled = approvedSpans.filter(\.isEnabled)
        return await complete(original: original, cleaned: cleaned, spans: enabled,
                              settings: settings, onStage: onStage, onDelta: onDelta)
    }

    // MARK: - แกนกลาง: mask (เฉพาะ span ที่เลือก) → สตรีมจาก LLM → de-mask ทีละส่วน

    private func complete(original: String,
                          cleaned: String,
                          spans: [DetectedSpan],
                          settings: LLMSettings,
                          onStage: @escaping (ProxyStage) -> Void,
                          onDelta: @escaping (String) -> Void) async -> ProxyResult {
        let provider = settings.provider

        // mask ตามช่วงที่เลือก (เรียงตามตำแหน่ง กันช่วงทับซ้อน)
        onStage(.masking)
        var masked = ""
        var mappings: [PseudonymMapping] = []
        var cursor = cleaned.startIndex
        for span in spans.sorted(by: { $0.range.lowerBound < $1.range.lowerBound }) {
            guard span.range.lowerBound >= cursor else { continue }   // ข้ามช่วงที่ทับกัน
            masked += cleaned[cursor..<span.range.lowerBound]
            let reused = vault.existingPseudonym(for: span.text) != nil
            let pseudonym = vault.pseudonym(for: span.text, type: span.type)
            masked += pseudonym
            mappings.append(PseudonymMapping(original: span.text,
                                             pseudonym: pseudonym,
                                             type: span.type,
                                             isReused: reused))
            cursor = span.range.upperBound
        }
        masked += cleaned[cursor...]

        // เรียงรหัสจากยาวไปสั้นไว้ล่วงหน้า (ใช้ซ้ำทุกครั้งที่ถอดรหัส)
        let ordered = mappings.sorted { $0.pseudonym.count > $1.pseudonym.count }
        func demask(_ text: String) -> String {
            var out = text
            for m in ordered { out = out.replacingOccurrences(of: m.pseudonym, with: m.original) }
            return out
        }

        // สตรีมคำตอบ: สะสม raw แล้วถอดรหัสทั้งบัฟเฟอร์ทุกเดลตา (กันรหัสถูกหั่นข้าม chunk)
        onStage(.contacting)
        var rawReply = ""
        var sawFirst = false
        do {
            for try await delta in llm.stream(prompt: masked, settings: settings) {
                if !sawFirst { sawFirst = true; onStage(.receiving) }
                rawReply += delta
                onDelta(demask(rawReply))
            }
        } catch {
            let llmError = LLMClient.map(error)
            let reason = "\(llmError.errorDescription ?? "เรียก LLM ไม่สำเร็จ")\n💡 \(llmError.hint)"
            let partial = ProxyTelemetry(provider: provider,
                                         originalPrompt: original,
                                         maskedPrompt: masked,
                                         rawReply: rawReply,
                                         demaskedReply: "",
                                         mappings: mappings)
            return .failure(reason: reason, telemetry: partial)
        }

        let finalReply = demask(rawReply)
        let telemetry = ProxyTelemetry(provider: provider,
                                       originalPrompt: original,
                                       maskedPrompt: masked,
                                       rawReply: rawReply,
                                       demaskedReply: finalReply,
                                       mappings: mappings)
        return .success(reply: finalReply, telemetry: telemetry)
    }

    /// โหลด Thai NER (ครั้งแรก) แล้วตรวจจับ entity — ถ้าโหลด/รันไม่ได้ คืน [] เพื่อไม่ให้ทั้งระบบล้ม
    private func thaiNEREntities(in text: String) async -> [PIIEntity] {
        if !thaiTaggerLoaded {
            thaiTaggerLoaded = true
            do {
                thaiTagger = try await ThaiNERTagger.load()
            } catch {
                ThaiNERTagger.debugLog("LOAD FAILED: \(error)")
            }
        }
        guard let thaiTagger else { return [] }
        return await thaiTagger.tag(text)
    }
}
