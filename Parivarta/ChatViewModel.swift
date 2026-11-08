//
//  ChatViewModel.swift
//  Parivarta
//
//  สถานะและตรรกะของหน้าแชท เชื่อมระหว่าง UI กับ PseudonymizationProxy
//

import Foundation
import Observation

@Observable
final class ChatViewModel {
    var messages: [ChatMessage] = []
    var provider: LLMProvider = .openAI
    var apiKey: String = ""
    var watchlist: [String] = []
    var isProcessing = false

    /// เปิดโหมด "ตรวจก่อนส่ง" — แสดงพรีวิว PII ให้ยืนยันก่อนออกจากเครื่อง
    var previewEnabled = true
    /// สถานะพรีวิวที่รอผู้ใช้ยืนยัน (nil = ไม่มี)
    var pendingPreview: PendingPreview?
    /// ขั้นตอนปัจจุบันระหว่างประมวลผล (สำหรับ progress เป็นสเต็ป)
    var stage: ProxyStage?

    /// ข้อมูลพรีวิว "ตรวจก่อนส่ง" ที่รอการยืนยัน
    struct PendingPreview: Identifiable {
        let id = UUID()
        let original: String
        let cleaned: String
        var spans: [DetectedSpan]

        var enabledCount: Int { spans.filter(\.isEnabled).count }
        var hasDisabled: Bool { spans.contains { !$0.isEnabled } }
    }

    // การตั้งค่าเพิ่มเติมต่อผู้ให้บริการ
    var customModel: String = ""                          // Ollama / OpenRouter ระบุชื่อโมเดลเอง
    var ollamaBaseURL: String = "http://localhost:11434"
    var azureEndpoint: String = ""
    var azureDeployment: String = ""
    var azureAPIVersion: String = "2024-06-01"

    private let vault = MappingVault()
    private let proxy: PseudonymizationProxy

    init() {
        proxy = PseudonymizationProxy(vault: vault)
    }

    // MARK: - Vault (หน้าจัดการข้อมูลที่เก็บไว้)

    /// ทุก mapping ที่เก็บไว้ (ถอดรหัสแล้ว) — สำหรับหน้าจัดการ vault
    func vaultEntries() -> [MappingVault.Entry] { vault.entries() }

    /// ลืม mapping ทั้งหมด
    func clearVault() { vault.removeAll() }

    /// รหัสแทนที่เคยกำหนดให้ค่านี้ (ถ้ามี) — ใช้โชว์ในพรีวิวเพื่อสื่อความสม่ำเสมอ
    func existingPseudonym(for text: String) -> String? { vault.existingPseudonym(for: text) }

    /// ตรวจว่าพร้อมส่งข้อความไหม (ขึ้นกับผู้ให้บริการ)
    var canSend: Bool {
        guard !isProcessing else { return false }
        if provider.requiresAPIKey && apiKey.trimmingCharacters(in: .whitespaces).isEmpty {
            return false
        }
        if provider == .azureOpenAI {
            if azureEndpoint.trimmingCharacters(in: .whitespaces).isEmpty { return false }
            if azureDeployment.trimmingCharacters(in: .whitespaces).isEmpty { return false }
        }
        return true
    }

    /// รวมการตั้งค่าเป็น LLMSettings เดียว (แก้ค่า model ให้เหมาะกับผู้ให้บริการ)
    private func makeSettings() -> LLMSettings {
        let trimmedModel = customModel.trimmingCharacters(in: .whitespaces)
        let model: String
        switch provider {
        case .ollama, .openRouter:
            model = trimmedModel.isEmpty ? provider.defaultModel : trimmedModel
        case .azureOpenAI:
            model = azureDeployment
        default:
            model = trimmedModel.isEmpty ? provider.defaultModel : trimmedModel
        }
        return LLMSettings(provider: provider,
                           apiKey: apiKey,
                           model: model,
                           ollamaBaseURL: ollamaBaseURL,
                           azureEndpoint: azureEndpoint,
                           azureDeployment: azureDeployment,
                           azureAPIVersion: azureAPIVersion)
    }

    func addWatchlistWord(_ word: String) {
        let trimmed = word.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !watchlist.contains(trimmed) else { return }
        watchlist.append(trimmed)
    }

    func clearWatchlist() {
        watchlist.removeAll()
    }

    /// จุดเข้าหลักจาก UI: ถ้าเปิด preview จะวิเคราะห์แล้วโชว์พรีวิวก่อน ไม่งั้นส่งตรง
    func submit(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, pendingPreview == nil else { return }

        guard previewEnabled else {
            await runFullPipeline(trimmed)
            return
        }

        isProcessing = true
        stage = .analyzing
        let analysis = await proxy.analyze(prompt: trimmed, watchlist: watchlist)
        isProcessing = false
        stage = nil

        switch analysis {
        case .blocked(let reason, let patterns):
            messages.append(ChatMessage(role: .user, text: trimmed))
            appendBlocked(reason: reason, patterns: patterns)
        case .safe(let cleaned, let spans):
            if spans.isEmpty {
                // ไม่มี PII → ไม่ต้องรบกวนให้ยืนยัน ส่งได้เลย
                await sendApproved(PendingPreview(original: trimmed, cleaned: cleaned, spans: []))
            } else {
                pendingPreview = PendingPreview(original: trimmed, cleaned: cleaned, spans: spans)
            }
        }
    }

    /// ผู้ใช้กด "ส่งอย่างปลอดภัย" จากพรีวิว
    func confirmPreview() async {
        guard let preview = pendingPreview else { return }
        pendingPreview = nil
        await sendApproved(preview)
    }

    /// ผู้ใช้ยกเลิกพรีวิว (กลับไปแก้ข้อความ)
    func cancelPreview() {
        pendingPreview = nil
    }

    /// สลับการปกป้องของ span รายตัวในพรีวิว
    func toggleSpan(_ id: DetectedSpan.ID) {
        guard var preview = pendingPreview,
              let idx = preview.spans.firstIndex(where: { $0.id == id }) else { return }
        preview.spans[idx].isEnabled.toggle()
        pendingPreview = preview
    }

    // MARK: - ภายใน

    private func sendApproved(_ preview: PendingPreview) async {
        await streamRun(userText: preview.original) { onStage, onDelta in
            await self.proxy.send(cleaned: preview.cleaned,
                                  original: preview.original,
                                  approvedSpans: preview.spans,
                                  settings: self.makeSettings(),
                                  onStage: onStage, onDelta: onDelta)
        }
    }

    private func runFullPipeline(_ trimmed: String) async {
        await streamRun(userText: trimmed) { onStage, onDelta in
            await self.proxy.process(prompt: trimmed,
                                     watchlist: self.watchlist,
                                     settings: self.makeSettings(),
                                     onStage: onStage, onDelta: onDelta)
        }
    }

    /// รันงานที่สตรีมคำตอบ: วางฟองผู้ใช้ + ฟองผู้ช่วยเปล่า แล้วอัปเดตข้อความทีละเดลตา
    private func streamRun(userText: String,
                          perform: (@escaping (ProxyStage) -> Void, @escaping (String) -> Void) async -> ProxyResult) async {
        messages.append(ChatMessage(role: .user, text: userText))
        let placeholder = ChatMessage(role: .assistant, text: "")
        messages.append(placeholder)
        let assistantID = placeholder.id

        isProcessing = true
        stage = nil
        defer { isProcessing = false; stage = nil }

        let result = await perform(
            { newStage in self.stage = newStage },
            { text in self.update(assistantID, text: text) }
        )

        switch result {
        case .success(let reply, let telemetry):
            update(assistantID, text: reply, telemetry: telemetry)
        case .failure(let reason, let telemetry):
            update(assistantID,
                   text: "เรียก LLM ไม่สำเร็จ\n\n\(reason)\n\n(ข้อมูลถูกแทนที่ด้วยรหัสเรียบร้อยก่อนส่ง — เปิดแถบด้านล่างเพื่อตรวจสอบได้)",
                   telemetry: telemetry, isBlocked: true)
        case .blocked(let reason, let patterns):
            let detail = patterns.isEmpty ? "" : "\n\nรูปแบบที่ตรวจพบ: \(patterns.joined(separator: ", "))"
            update(assistantID, text: "ถูกบล็อก: \(reason)\(detail)", isBlocked: true)
        }
    }

    /// อัปเดตข้อความผู้ช่วยตาม id (ใช้ระหว่างสตรีมและตอนจบ)
    private func update(_ id: ChatMessage.ID, text: String,
                        telemetry: ProxyTelemetry? = nil, isBlocked: Bool = false) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[idx].text = text
        if let telemetry { messages[idx].telemetry = telemetry }
        if isBlocked { messages[idx].isBlocked = true }
    }

    private func appendBlocked(reason: String, patterns: [String]) {
        let detail = patterns.isEmpty ? "" : "\n\nรูปแบบที่ตรวจพบ: \(patterns.joined(separator: ", "))"
        messages.append(ChatMessage(role: .assistant,
                                    text: "ถูกบล็อก: \(reason)\(detail)",
                                    isBlocked: true))
    }
}
