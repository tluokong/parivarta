//
//  Models.swift
//  Parivarta
//
//  โครงสร้างข้อมูลหลักของระบบ Zero-Trust Local Privacy Proxy
//

import Foundation

/// ผู้ให้บริการ LLM ที่รองรับ (ข้อความจะถูก mask ก่อนส่งออกทุกครั้ง)
enum LLMProvider: String, CaseIterable, Identifiable {
    case openAI = "ChatGPT (OpenAI)"
    case gemini = "Gemini (Google)"
    case anthropic = "Claude (Anthropic)"
    case ollama = "Ollama (Local)"
    case azureOpenAI = "Azure OpenAI"
    case openRouter = "OpenRouter"

    var id: String { rawValue }

    /// ชื่อสั้นของผู้ให้บริการ (ใช้ในป้ายกำกับที่กระชับ เช่น ช่องกรอก API Key)
    var shortName: String {
        switch self {
        case .openAI:      return "OpenAI"
        case .gemini:      return "Gemini"
        case .anthropic:   return "Anthropic"
        case .ollama:      return "Ollama"
        case .azureOpenAI: return "Azure OpenAI"
        case .openRouter:  return "OpenRouter"
        }
    }

    /// ชื่อโมเดลเริ่มต้นของแต่ละผู้ให้บริการ
    var defaultModel: String {
        switch self {
        case .openAI: return "gpt-4o-mini"
        case .gemini: return "gemini-2.5-flash"
        case .anthropic: return "claude-opus-4-8"
        case .ollama: return "llama3.1"
        case .azureOpenAI: return ""              // ใช้ชื่อ deployment แทน
        case .openRouter: return "anthropic/claude-3.5-sonnet"
        }
    }

    /// ต้องกรอก API Key ไหม (Ollama รันในเครื่องจึงไม่ต้อง)
    var requiresAPIKey: Bool { self != .ollama }

    /// ให้ผู้ใช้ระบุชื่อโมเดลเองได้ (Ollama/OpenRouter จำเป็นต้องระบุ)
    var allowsCustomModel: Bool {
        switch self {
        case .ollama, .openRouter: return true
        default: return false
        }
    }
}

/// การตั้งค่าการเชื่อมต่อ LLM ที่ส่งผ่านไปยัง LLMClient
struct LLMSettings {
    var provider: LLMProvider
    var apiKey: String = ""
    var model: String = ""                                   // โมเดลที่ใช้จริง (override หรือ default)
    var ollamaBaseURL: String = "http://localhost:11434"
    var azureEndpoint: String = ""                           // เช่น https://my-res.openai.azure.com
    var azureDeployment: String = ""
    var azureAPIVersion: String = "2024-06-01"
}

/// ข้อมูลส่วนบุคคล (PII) ที่ตรวจพบในข้อความ พร้อมตำแหน่งในสตริง
struct PIIEntity: Identifiable {
    let id = UUID()
    let type: String                    // เช่น PERSON, PHONE_NUMBER, EMAIL
    let text: String                    // ข้อความจริงที่ตรวจพบ
    let range: Range<String.Index>      // ตำแหน่งใน "cleaned text"
    let confidence: Double
    /// ความ "แม่นของขอบเขต" ของแหล่งที่ตรวจพบ (0–1) ใช้ตอน dedup:
    /// แหล่งที่ให้สแปนกระชับ/ตรง (regex รูปแบบตายตัว) สูงกว่าแหล่งที่ชอบจับเกินหาง (NER/NLTagger)
    /// — ต่างจาก confidence ที่วัด "ความมั่นใจว่าเป็น PII"
    var precision: Double = 0.5
}

/// การจับคู่ระหว่างข้อมูลจริงกับรหัสแทน (pseudonym)
struct PseudonymMapping: Hashable {
    let original: String
    let pseudonym: String
    let type: String
    /// true = ใช้รหัสเดิมที่เคยกำหนดไว้ (entity เดียวกันข้ามเทิร์น) → แสดงความสม่ำเสมอ
    var isReused: Bool = false
}

/// รายละเอียดการทำงานเบื้องหลังของ proxy สำหรับแสดงในแผง "ดูการทำงานเบื้องหลัง"
struct ProxyTelemetry {
    let provider: LLMProvider
    let originalPrompt: String      // 1️⃣ ข้อความก่อนส่ง
    let maskedPrompt: String        // 2️⃣ ข้อความที่แปลงแล้ว (ส่งให้ LLM)
    let rawReply: String            // 3️⃣ คำตอบดิบจาก LLM (ยังเป็นรหัส)
    let demaskedReply: String       // 4️⃣ คำตอบที่ถอดรหัสกลับแล้ว
    let mappings: [PseudonymMapping]

    var piiCount: Int { mappings.count }
}

/// ช่วง PII ที่ตรวจพบ สำหรับพรีวิว "ตรวจก่อนส่ง" — ผู้ใช้เปิด/ปิดการปกป้องรายตัวได้
struct DetectedSpan: Identifiable {
    let id = UUID()
    let type: String
    let text: String
    let range: Range<String.Index>   // ตำแหน่งใน cleaned text
    var isEnabled: Bool = true        // ปิด = ยอมส่งคำนี้ออกโดยไม่ mask
}

/// ผลการวิเคราะห์ก่อนส่ง (ยังไม่เรียก LLM) — ใช้แสดงพรีวิวให้ผู้ใช้ยืนยัน
enum AnalysisResult {
    case safe(cleaned: String, spans: [DetectedSpan])
    case blocked(reason: String, patterns: [String])
}

/// ขั้นตอนการทำงานของ proxy — ใช้แสดง progress เป็นสเต็ปให้ผู้ใช้เห็นคุณค่า zero-trust
enum ProxyStage: Int, CaseIterable {
    case analyzing   // 1) ตรวจจับ PII (รวม Core ML NER)
    case masking     // 2) แทนที่ PII ด้วยรหัส
    case contacting  // 3) ส่งข้อความที่ปลอดภัยแล้วไปยัง LLM
    case receiving   // 4) รับคำตอบ (สตรีม) + ถอดรหัสกลับ

    var label: String {
        switch self {
        case .analyzing:  return "ตรวจจับ PII"
        case .masking:    return "แทนที่ด้วยรหัส"
        case .contacting: return "ส่งให้ LLM"
        case .receiving:  return "รับคำตอบ & ถอดรหัส"
        }
    }

    var systemImage: String {
        switch self {
        case .analyzing:  return "magnifyingglass"
        case .masking:    return "lock.fill"
        case .contacting: return "paperplane.fill"
        case .receiving:  return "arrow.down.circle.fill"
        }
    }
}

/// ผลลัพธ์ของการประมวลผลข้อความผ่าน proxy
enum ProxyResult {
    case success(reply: String, telemetry: ProxyTelemetry)
    case blocked(reason: String, patterns: [String])
    /// LLM ล้มเหลว แต่ยังแนบ telemetry เพื่อให้ดูผลการ mask PII ได้ (masking ทำเสร็จก่อนเรียก LLM)
    case failure(reason: String, telemetry: ProxyTelemetry?)
}

/// ข้อความหนึ่งบรรทัดในหน้าแชท
struct ChatMessage: Identifiable {
    enum Role {
        case user
        case assistant
    }

    let id = UUID()
    let role: Role
    var text: String
    var telemetry: ProxyTelemetry? = nil
    var isBlocked: Bool = false
}
