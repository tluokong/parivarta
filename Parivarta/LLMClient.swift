//
//  LLMClient.swift
//  Parivarta
//
//  เรียก LLM ผ่าน API (ข้อความที่ส่งถูก mask เรียบร้อยแล้วเสมอ)
//  รองรับ: OpenAI, Gemini, Anthropic (Claude), Ollama (local), Azure OpenAI, OpenRouter
//
//  หมายเหตุ: Swift ยังไม่มี official Anthropic SDK จึงเรียกผ่าน raw HTTP ตามสเปกทางการ
//  (POST /v1/messages, header x-api-key + anthropic-version: 2023-06-01)
//

import Foundation

struct LLMClient {
    enum LLMError: LocalizedError {
        case unauthorized                 // 401/403 — API key
        case rateLimited                  // 429
        case serverError(Int)             // 5xx
        case network(String)              // เชื่อมต่อไม่ได้ / timeout
        case badResponse(String)          // อื่น ๆ + รายละเอียด

        var errorDescription: String? {
            switch self {
            case .unauthorized:      return "การยืนยันตัวตนล้มเหลว — API key ไม่ถูกต้องหรือหมดสิทธิ์"
            case .rateLimited:       return "ถูกจำกัดอัตราการเรียก (rate limit)"
            case .serverError(let c): return "เซิร์ฟเวอร์ผู้ให้บริการขัดข้อง (HTTP \(c))"
            case .network(let m):    return "เชื่อมต่อเครือข่ายไม่สำเร็จ: \(m)"
            case .badResponse(let m): return m
            }
        }

        /// คำแนะนำการแก้ไขที่ผู้ใช้ทำได้จริง (แสดงคู่กับ error)
        var hint: String {
            switch self {
            case .unauthorized:  return "ตรวจสอบ API key ในแถบตั้งค่าด้านซ้าย"
            case .rateLimited:   return "รอสักครู่แล้วส่งใหม่ หรือสลับผู้ให้บริการ/โมเดล"
            case .serverError:   return "ลองใหม่อีกครั้ง หรือสลับผู้ให้บริการ"
            case .network:       return "ตรวจสอบอินเทอร์เน็ต และ URL ของผู้ให้บริการ (เช่น Ollama/Azure)"
            case .badResponse:   return "ตรวจสอบชื่อโมเดลและการตั้งค่าผู้ให้บริการ"
            }
        }
    }

    /// แปลง error ใด ๆ ให้เป็น LLMError (จับ URLError เป็น .network)
    static func map(_ error: Error) -> LLMError {
        if let llm = error as? LLMError { return llm }
        if let urlError = error as? URLError { return .network(urlError.localizedDescription) }
        return .network(error.localizedDescription)
    }

    /// ส่งข้อความ (ที่ mask แล้ว) ไปยังผู้ให้บริการที่เลือก แล้วคืนคำตอบดิบ
    func complete(prompt: String, settings: LLMSettings) async throws -> String {
        switch settings.provider {
        case .openAI:
            return try await openAICompatible(
                prompt: prompt, model: settings.model,
                url: URL(string: "https://api.openai.com/v1/chat/completions")!,
                bearerKey: settings.apiKey)

        case .openRouter:
            return try await openAICompatible(
                prompt: prompt, model: settings.model,
                url: URL(string: "https://openrouter.ai/api/v1/chat/completions")!,
                bearerKey: settings.apiKey)

        case .ollama:
            let base = settings.ollamaBaseURL.trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            return try await openAICompatible(
                prompt: prompt, model: settings.model,
                url: URL(string: "\(base)/v1/chat/completions")!,
                bearerKey: nil)                                  // local — ไม่ต้องใช้ key

        case .azureOpenAI:
            return try await callAzureOpenAI(prompt: prompt, settings: settings)

        case .gemini:
            return try await callGemini(prompt: prompt, model: settings.model, apiKey: settings.apiKey)

        case .anthropic:
            return try await callAnthropic(prompt: prompt, model: settings.model, apiKey: settings.apiKey)
        }
    }

    // MARK: - OpenAI-compatible (OpenAI / OpenRouter / Ollama / Azure ใช้รูปแบบตอบเดียวกัน)

    private func openAICompatible(prompt: String, model: String, url: URL, bearerKey: String?) async throws -> String {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let bearerKey, !bearerKey.isEmpty {
            request.setValue("Bearer \(bearerKey)", forHTTPHeaderField: "Authorization")
        }
        let body: [String: Any] = [
            "model": model,
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.validate(response, data)
        return try Self.extractOpenAIContent(data)
    }

    private func callAzureOpenAI(prompt: String, settings: LLMSettings) async throws -> String {
        let endpoint = settings.azureEndpoint.trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let urlString = "\(endpoint)/openai/deployments/\(settings.azureDeployment)/chat/completions?api-version=\(settings.azureAPIVersion)"
        guard let url = URL(string: urlString) else {
            throw LLMError.badResponse("Azure endpoint/deployment ไม่ถูกต้อง")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(settings.apiKey, forHTTPHeaderField: "api-key")   // Azure ใช้ header api-key
        let body: [String: Any] = [
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.validate(response, data)
        return try Self.extractOpenAIContent(data)
    }

    // MARK: - Gemini (generateContent)

    private func callGemini(prompt: String, model: String, apiKey: String) async throws -> String {
        let url = URL(string: "https://generativelanguage.googleapis.com/v1beta/models/\(model):generateContent")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")

        let body: [String: Any] = ["contents": [["parts": [["text": prompt]]]]]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.validate(response, data)

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let candidates = json?["candidates"] as? [[String: Any]],
              let content = candidates.first?["content"] as? [String: Any],
              let parts = content["parts"] as? [[String: Any]],
              let text = parts.first?["text"] as? String else {
            throw LLMError.badResponse("รูปแบบผลลัพธ์จาก Gemini ไม่ถูกต้อง")
        }
        return text
    }

    // MARK: - Anthropic (Messages API, raw HTTP)

    private func callAnthropic(prompt: String, model: String, apiKey: String) async throws -> String {
        var request = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

        let body: [String: Any] = [
            "model": model,
            "max_tokens": 4096,
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.validate(response, data)

        // content เป็น array ของ block — รวมเฉพาะ block ชนิด "text"
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let content = json?["content"] as? [[String: Any]] else {
            throw LLMError.badResponse("รูปแบบผลลัพธ์จาก Anthropic ไม่ถูกต้อง")
        }
        let text = content
            .filter { ($0["type"] as? String) == "text" }
            .compactMap { $0["text"] as? String }
            .joined()
        guard !text.isEmpty else {
            throw LLMError.badResponse("ไม่พบข้อความในผลลัพธ์จาก Anthropic")
        }
        return text
    }

    // MARK: - Helpers

    private static func extractOpenAIContent(_ data: Data) throws -> String {
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let choices = json?["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw LLMError.badResponse("รูปแบบผลลัพธ์ (OpenAI-compatible) ไม่ถูกต้อง")
        }
        return content
    }

    private static func validate(_ response: URLResponse, _ data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw statusError(http.statusCode, data)
        }
    }

    /// จับ HTTP status ที่ผิดปกติเป็น LLMError ที่มีความหมาย
    private static func statusError(_ status: Int, _ data: Data) -> LLMError {
        switch status {
        case 401, 403: return .unauthorized
        case 429:      return .rateLimited
        case 500...:   return .serverError(status)
        default:
            let detail = String(data: data, encoding: .utf8) ?? "ไม่มีรายละเอียด"
            return .badResponse("HTTP \(status): \(detail)")
        }
    }

    // MARK: - Streaming (SSE) — คืน "เดลตา" ข้อความทีละชิ้น

    /// สตรีมคำตอบจากผู้ให้บริการ (OpenAI-compatible + Anthropic รองรับ SSE จริง,
    /// Gemini fallback เป็นก้อนเดียว) — ยิงข้อความส่วนเพิ่มออกมาเรื่อย ๆ
    func stream(prompt: String, settings: LLMSettings) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    switch settings.provider {
                    case .openAI:
                        try await streamOpenAICompatible(
                            prompt: prompt, model: settings.model,
                            url: URL(string: "https://api.openai.com/v1/chat/completions")!,
                            bearerKey: settings.apiKey, extraHeaders: [:], continuation: continuation)
                    case .openRouter:
                        try await streamOpenAICompatible(
                            prompt: prompt, model: settings.model,
                            url: URL(string: "https://openrouter.ai/api/v1/chat/completions")!,
                            bearerKey: settings.apiKey, extraHeaders: [:], continuation: continuation)
                    case .ollama:
                        let base = settings.ollamaBaseURL.trimmingCharacters(in: .whitespaces)
                            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                        try await streamOpenAICompatible(
                            prompt: prompt, model: settings.model,
                            url: URL(string: "\(base)/v1/chat/completions")!,
                            bearerKey: nil, extraHeaders: [:], continuation: continuation)
                    case .azureOpenAI:
                        let endpoint = settings.azureEndpoint.trimmingCharacters(in: .whitespaces)
                            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                        guard let url = URL(string: "\(endpoint)/openai/deployments/\(settings.azureDeployment)/chat/completions?api-version=\(settings.azureAPIVersion)") else {
                            throw LLMError.badResponse("Azure endpoint/deployment ไม่ถูกต้อง")
                        }
                        try await streamOpenAICompatible(
                            prompt: prompt, model: settings.model, url: url,
                            bearerKey: nil, extraHeaders: ["api-key": settings.apiKey], continuation: continuation)
                    case .anthropic:
                        try await streamAnthropic(prompt: prompt, model: settings.model,
                                                  apiKey: settings.apiKey, continuation: continuation)
                    case .gemini:
                        // ยังไม่ทำ SSE ของ Gemini — ดึงทั้งก้อนแล้วส่งครั้งเดียว
                        let full = try await callGemini(prompt: prompt, model: settings.model, apiKey: settings.apiKey)
                        continuation.yield(full)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: Self.map(error))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func streamOpenAICompatible(prompt: String, model: String, url: URL,
                                        bearerKey: String?, extraHeaders: [String: String],
                                        continuation: AsyncThrowingStream<String, Error>.Continuation) async throws {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let bearerKey, !bearerKey.isEmpty {
            request.setValue("Bearer \(bearerKey)", forHTTPHeaderField: "Authorization")
        }
        for (k, v) in extraHeaders { request.setValue(v, forHTTPHeaderField: k) }
        let body: [String: Any] = [
            "model": model,
            "stream": true,
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let bytes = try await Self.openStream(request)
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" { break }
            guard let d = payload.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let delta = choices.first?["delta"] as? [String: Any],
                  let content = delta["content"] as? String else { continue }
            continuation.yield(content)
        }
    }

    private func streamAnthropic(prompt: String, model: String, apiKey: String,
                                 continuation: AsyncThrowingStream<String, Error>.Continuation) async throws {
        var request = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        let body: [String: Any] = [
            "model": model,
            "max_tokens": 4096,
            "stream": true,
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let bytes = try await Self.openStream(request)
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            guard let d = payload.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { continue }
            if (json["type"] as? String) == "content_block_delta",
               let delta = json["delta"] as? [String: Any],
               (delta["type"] as? String) == "text_delta",
               let text = delta["text"] as? String {
                continuation.yield(text)
            }
        }
    }

    /// เปิด SSE stream + ตรวจ HTTP status (อ่าน body ถ้า error เพื่อรายละเอียด)
    private static func openStream(_ request: URLRequest) async throws -> URLSession.AsyncBytes {
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            var data = Data()
            for try await b in bytes { data.append(b) }
            throw statusError(http.statusCode, data)
        }
        return bytes
    }
}
