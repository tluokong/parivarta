//
//  ContentView.swift
//  Parivarta
//
//  หน้าจอหลัก: Sidebar ตั้งค่า + หน้าแชท Parivarta
//

import SwiftUI
import AppKit

struct ContentView: View {
    @State private var viewModel = ChatViewModel()

    var body: some View {
        NavigationSplitView {
            SettingsSidebar(viewModel: viewModel)
                .navigationSplitViewColumnWidth(min: 260, ideal: 300, max: 360)
        } detail: {
            ChatView(viewModel: viewModel)
        }
    }
}

// MARK: - Sidebar ตั้งค่าระบบ

struct SettingsSidebar: View {
    @Bindable var viewModel: ChatViewModel
    @State private var newWord = ""

    var body: some View {
        Form {
            Section {
                HStack(spacing: 12) {
                    Image("ParivartaLogo")
                        .resizable()
                        .interpolation(.high)
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Parivarta").font(.title3).bold()
                        Text("Zero-Trust Privacy Proxy")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 2)
            }

            Section {
                Picker("ผู้ให้บริการ AI", selection: $viewModel.provider) {
                    ForEach(LLMProvider.allCases) { provider in
                        Text(provider.rawValue).tag(provider)
                    }
                }
                .labelsHidden()

                if viewModel.provider.requiresAPIKey {
                    SettingsField(icon: "key.fill", placeholder: "API Key",
                                  text: $viewModel.apiKey, secure: true, mono: true,
                                  accessibilityLabel: "API Key ของ \(viewModel.provider.shortName)")
                }

                if viewModel.provider.allowsCustomModel {
                    SettingsField(icon: "cube", placeholder: "ชื่อโมเดล (เช่น \(viewModel.provider.defaultModel))",
                                  text: $viewModel.customModel)
                }

                if viewModel.provider == .ollama {
                    SettingsField(icon: "link", placeholder: "Ollama Base URL",
                                  text: $viewModel.ollamaBaseURL, mono: true)
                    Text("รันในเครื่อง — ไม่ต้องใช้ API Key")
                        .font(.caption).foregroundStyle(.secondary)
                }

                if viewModel.provider == .azureOpenAI {
                    SettingsField(icon: "link", placeholder: "Azure Endpoint (https://xxx.openai.azure.com)",
                                  text: $viewModel.azureEndpoint, mono: true)
                    SettingsField(icon: "cube", placeholder: "Deployment name",
                                  text: $viewModel.azureDeployment)
                    SettingsField(icon: "number", placeholder: "API version",
                                  text: $viewModel.azureAPIVersion, mono: true)
                }
            } header: {
                Label("ผู้ให้บริการ AI", systemImage: "sparkles")
            }

            Section {
                HStack {
                    TextField("คำที่ต้องปกป้องเสมอ", text: $newWord)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(addWord)
                    Button(action: addWord) {
                        Image(systemName: "plus")
                    }
                    .disabled(newWord.trimmingCharacters(in: .whitespaces).isEmpty)
                    .accessibilityLabel("เพิ่มคำที่ต้องปกป้อง")
                }

                if viewModel.watchlist.isEmpty {
                    Text("ยังไม่มีคำที่กำหนดเอง")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(viewModel.watchlist, id: \.self) { word in
                        Text(word)
                            .font(.system(.body, design: .monospaced))
                    }
                    Button(role: .destructive) {
                        viewModel.clearWatchlist()
                    } label: {
                        Label("ล้างคำทั้งหมด", systemImage: "trash")
                    }
                }
            } header: {
                Label("คำที่ต้องปกป้องเสมอ", systemImage: "list.bullet.rectangle")
            }

            Section {
                Toggle(isOn: $viewModel.previewEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("ตรวจก่อนส่ง (Preview)")
                        Text("แสดง PII ที่จะถูกปกป้องให้ยืนยันก่อนออกจากเครื่อง")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            } header: {
                Label("ความเป็นส่วนตัว", systemImage: "hand.raised")
            }

            Section {
                Label("ข้อมูลถูกปกป้องในเครื่องก่อนส่งออกทุกครั้ง", systemImage: "lock.shield")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .navigationTitle("การตั้งค่า")
    }

    private func addWord() {
        viewModel.addWatchlistWord(newWord)
        newWord = ""
    }
}

// MARK: - หน้าแชทหลัก

struct ChatView: View {
    @Bindable var viewModel: ChatViewModel
    @State private var input = ""
    @State private var showVault = false

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 16) {
                        if viewModel.messages.isEmpty {
                            EmptyStateView()
                        }
                        ForEach(viewModel.messages) { message in
                            MessageRow(message: message)
                                .id(message.id)
                        }
                        if viewModel.isProcessing {
                            StageIndicator(current: viewModel.stage)
                                .id("processing")
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) {
                    if let last = viewModel.messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }

            Divider()
            if viewModel.pendingPreview != nil {
                PreviewPanel(viewModel: viewModel,
                             onConfirm: { Task { await viewModel.confirmPreview(); input = "" } },
                             onCancel: { viewModel.cancelPreview() })
                Divider()
            }
            inputBar
        }
        .navigationTitle("Parivarta")
        .navigationSubtitle("Zero-Trust Privacy Proxy")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showVault = true
                } label: {
                    Label("จัดการข้อมูลที่เก็บไว้", systemImage: "lock.rectangle.stack")
                }
                .help("ดู/ลบ ตารางการแทนที่ (mapping) ที่เก็บไว้ในเครื่อง")
            }
        }
        .sheet(isPresented: $showVault) {
            VaultManagementView(viewModel: viewModel)
        }
    }

    private var inputBar: some View {
        VStack(spacing: 6) {
            if !viewModel.canSend && !viewModel.isProcessing {
                Label("กรอกการตั้งค่า API Key ที่แถบด้านซ้ายให้ครบก่อนใช้งาน",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack(spacing: 8) {
                TextField("พิมพ์ข้อความ…", text: $input, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...5)
                    .onSubmit(sendMessage)
                    .disabled(viewModel.pendingPreview != nil)
                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "paperplane.fill")
                        .padding(6)
                }
                .buttonStyle(.glassProminent)
                .disabled(!viewModel.canSend || viewModel.pendingPreview != nil || input.trimmingCharacters(in: .whitespaces).isEmpty)
                .accessibilityLabel("ส่งข้อความ")
                .help("ส่งข้อความ (Return) · ขึ้นบรรทัดใหม่ด้วย Shift-Return")
            }
        }
        .padding()
    }

    private func sendMessage() {
        let text = input
        guard viewModel.canSend, viewModel.pendingPreview == nil,
              !text.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        Task {
            await viewModel.submit(text)
            if viewModel.pendingPreview == nil { input = "" }   // ส่ง/บล็อกไปแล้ว; ถ้าเป็นพรีวิวคงข้อความไว้ให้แก้
        }
    }
}

// MARK: - ฟิลด์ตั้งค่าสไตล์เดียวกัน (ไอคอนนำหน้า + พื้นโค้งมน)

struct SettingsField: View {
    let icon: String
    let placeholder: String
    @Binding var text: String
    var secure: Bool = false
    var mono: Bool = false
    var accessibilityLabel: String? = nil

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(.secondary)
                .frame(width: 16)
            Group {
                if secure {
                    SecureField(placeholder, text: $text)
                } else {
                    TextField(placeholder, text: $text)
                }
            }
            .textFieldStyle(.plain)
            .font(mono ? .system(.body, design: .monospaced) : .body)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(maxWidth: .infinity)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 9))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel ?? placeholder)
    }
}

// MARK: - ตัวบอกขั้นตอนการประมวลผล (progress เป็นสเต็ป)

struct StageIndicator: View {
    let current: ProxyStage?

    var body: some View {
        let active = current ?? .analyzing
        HStack(spacing: 6) {
            ForEach(ProxyStage.allCases, id: \.self) { step in
                let isDone = step.rawValue < active.rawValue
                let isActive = step.rawValue == active.rawValue
                HStack(spacing: 4) {
                    if isActive {
                        ProgressView().controlSize(.mini)
                    } else {
                        Image(systemName: isDone ? "checkmark.circle.fill" : step.systemImage)
                            .foregroundStyle(isDone ? Color.green : .secondary)
                    }
                    Text(step.label)
                        .font(.caption2)
                        .foregroundStyle(isActive ? .primary : .secondary)
                        .fontWeight(isActive ? .semibold : .regular)
                }
                if step != ProxyStage.allCases.last {
                    Image(systemName: "chevron.compact.right")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.vertical, 6).padding(.horizontal, 10)
        .glassEffect(.regular, in: .capsule)
    }
}

// MARK: - จานสีของชนิด PII (ใช้ไฮไลต์ให้เห็นว่าอะไรจะถูกปกป้อง)

enum PIIPalette {
    static func color(_ type: String) -> Color {
        switch type {
        case "PERSON":        return .blue
        case "PHONE_NUMBER":  return .green
        case "EMAIL":         return .orange
        case "THAI_ID":       return .red
        case "IP_ADDRESS":    return .purple
        case "ADDRESS":       return .pink
        case "LOCATION":      return .teal
        case "ORGANIZATION":  return .indigo
        default:              return .gray
        }
    }

    /// ป้ายชนิดแบบอ่านง่ายเป็นภาษาไทย
    static func label(_ type: String) -> String {
        switch type {
        case "PERSON":        return "ชื่อบุคคล"
        case "PHONE_NUMBER":  return "เบอร์โทร"
        case "EMAIL":         return "อีเมล"
        case "THAI_ID":       return "เลขบัตร ปชช."
        case "IP_ADDRESS":    return "IP"
        case "ADDRESS":       return "ที่อยู่"
        case "LOCATION":      return "สถานที่"
        case "ORGANIZATION":  return "องค์กร"
        case "CUSTOM_KEYWORD": return "คำต้องห้าม"
        default:              return type
        }
    }
}

// MARK: - แผงพรีวิว "ตรวจก่อนส่ง"

struct PreviewPanel: View {
    @Bindable var viewModel: ChatViewModel
    let onConfirm: () -> Void
    let onCancel: () -> Void

    var body: some View {
        if let preview = viewModel.pendingPreview {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 6) {
                    Image(systemName: "lock.shield.fill").foregroundStyle(.green)
                    Text("ตรวจก่อนส่ง — จะปกป้อง \(preview.enabledCount) รายการ")
                        .font(.subheadline).bold()
                    Spacer()
                    Text("ประมวลผลในเครื่อง")
                        .font(.caption2).foregroundStyle(.secondary)
                }

                // ข้อความที่จะส่งจริง พร้อมไฮไลต์ช่วงที่ถูกปกป้อง
                ScrollView {
                    Text(Self.highlighted(preview.cleaned, spans: preview.spans))
                        .font(.callout)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 120)
                .padding(8)
                .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))

                // ชิปรายการ PII — แตะเพื่อเปิด/ปิดการปกป้องรายตัว
                Text("แตะเพื่อเลือกว่าจะปกป้องรายการไหน:")
                    .font(.caption).foregroundStyle(.secondary)
                FlowChips(spans: preview.spans,
                          existing: { viewModel.existingPseudonym(for: $0) }) { id in
                    viewModel.toggleSpan(id)
                }

                if preview.hasDisabled {
                    Label("รายการที่ปิดไว้จะถูกส่งออกโดยไม่ปกป้อง", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }

                HStack {
                    Button("ยกเลิก / แก้ไข", role: .cancel, action: onCancel)
                        .buttonStyle(.glass)
                        .keyboardShortcut(.escape, modifiers: [])
                    Spacer()
                    Button(action: onConfirm) {
                        Label("ส่งอย่างปลอดภัย", systemImage: "paperplane.fill")
                    }
                    .buttonStyle(.glassProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                }
            }
            .padding()
            .background(.thinMaterial)
        }
    }

    /// สร้างข้อความไฮไลต์: พื้นหลังสีตามชนิด เฉพาะ span ที่เปิดการปกป้อง
    static func highlighted(_ text: String, spans: [DetectedSpan]) -> AttributedString {
        var attr = AttributedString(text)
        for span in spans where span.isEnabled {
            guard let lo = AttributedString.Index(span.range.lowerBound, within: attr),
                  let hi = AttributedString.Index(span.range.upperBound, within: attr) else { continue }
            attr[lo..<hi].backgroundColor = PIIPalette.color(span.type).opacity(0.30)
        }
        return attr
    }
}

// MARK: - ชิป PII แบบไหลบรรทัด

struct FlowChips: View {
    let spans: [DetectedSpan]
    let existing: (String) -> String?
    let onTap: (DetectedSpan.ID) -> Void

    var body: some View {
        let columns = [GridItem(.adaptive(minimum: 140), spacing: 8, alignment: .leading)]
        GlassEffectContainer(spacing: 8) {
            LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                ForEach(spans) { span in
                    Button { onTap(span.id) } label: {
                        HStack(spacing: 5) {
                            Image(systemName: span.isEnabled ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(span.isEnabled ? PIIPalette.color(span.type) : .secondary)
                            VStack(alignment: .leading, spacing: 0) {
                                Text(span.text)
                                    .font(.caption).lineLimit(1)
                                    .strikethrough(!span.isEnabled)
                                // แสดงรหัสเดิมถ้าเคยพบ entity นี้ (ความสม่ำเสมอ), ไม่งั้นแสดงชนิด
                                if let ps = existing(span.text) {
                                    Text("↩︎ \(ps)")
                                        .font(.caption2).foregroundStyle(.blue)
                                } else {
                                    Text(PIIPalette.label(span.type))
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                        }
                        .padding(.vertical, 5).padding(.horizontal, 10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.plain)
                    .glassEffect(
                        span.isEnabled
                            ? .regular.tint(PIIPalette.color(span.type).opacity(0.5)).interactive()
                            : .regular.interactive(),
                        in: .rect(cornerRadius: 10))
                    .opacity(span.isEnabled ? 1 : 0.6)
                    .help(span.isEnabled ? "กำลังปกป้อง — แตะเพื่อไม่ปกป้อง" : "ไม่ปกป้อง — แตะเพื่อปกป้อง")
                    .accessibilityLabel("\(PIIPalette.label(span.type)): \(span.text)")
                    .accessibilityValue(span.isEnabled ? "ปกป้อง" : "ไม่ปกป้อง")
                }
            }
        }
    }
}

// MARK: - แถวข้อความหนึ่งรายการ

struct MessageRow: View {
    let message: ChatMessage

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer(minLength: 60)
                Text(message.text)
                    .textSelection(.enabled)
                    .padding(10)
                    .background(Color.accentColor.opacity(0.15),
                                in: RoundedRectangle(cornerRadius: 10))
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 8) {
                    if message.isBlocked {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                            .accessibilityLabel("คำเตือน")
                    }
                    Text(message.text)
                        .textSelection(.enabled)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background((message.isBlocked ? Color.red : Color.gray).opacity(0.12),
                            in: RoundedRectangle(cornerRadius: 10))

                if let telemetry = message.telemetry {
                    TelemetryView(telemetry: telemetry)
                }
            }
            .padding(.trailing, 60)
        }
    }
}

// MARK: - แผง "ดูการทำงานเบื้องหลัง"

struct TelemetryView: View {
    let telemetry: ProxyTelemetry

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Label(telemetry.provider.rawValue, systemImage: "cpu")
                    Spacer()
                    Text("PII ที่ตรวจพบ: \(telemetry.piiCount) รายการ")
                        .foregroundStyle(.secondary)
                }
                .font(.caption)

                // ต้นฉบับ: ไฮไลต์ PII ที่ถูกปกป้อง (ค้นหาข้อความจริงของ mapping)
                sectionView(1, "ข้อความก่อนส่ง (Original Input)", .blue, copy: telemetry.originalPrompt) {
                    Text(Self.highlightPII(telemetry.originalPrompt, mappings: telemetry.mappings))
                }
                sectionView(2, "ข้อความที่แปลงแล้ว (Masked Prompt)", .orange, copy: telemetry.maskedPrompt) {
                    Text(telemetry.maskedPrompt.isEmpty ? "—" : telemetry.maskedPrompt)
                }
                sectionView(3, "คำตอบดิบจาก LLM (Raw Output)", .red, copy: telemetry.rawReply) {
                    Text(telemetry.rawReply.isEmpty ? "—" : telemetry.rawReply)
                }
                sectionView(4, "ถอดรหัสกลับแล้ว (De-masked Output)", .green, copy: telemetry.demaskedReply) {
                    Text(telemetry.demaskedReply.isEmpty ? "—" : telemetry.demaskedReply)
                }

                if !telemetry.mappings.isEmpty {
                    Label("ตารางการแทนที่ด้วยรหัส", systemImage: "key.fill")
                        .font(.caption).bold()
                    ForEach(telemetry.mappings, id: \.self) { mapping in
                        HStack(spacing: 6) {
                            Circle().fill(PIIPalette.color(mapping.type)).frame(width: 7, height: 7)
                            Text("\(mapping.pseudonym)  ⟶  \(mapping.original)")
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(.secondary)
                            if mapping.isReused {
                                Label("ใช้รหัสเดิม", systemImage: "arrow.triangle.2.circlepath")
                                    .font(.caption2).foregroundStyle(.blue)
                                    .help("entity เดียวกับที่เคยพบ — ได้รหัสเดิมเสมอ (ความสม่ำเสมอข้ามเทิร์น)")
                            }
                        }
                    }
                }
            }
            .padding(.top, 6)
        } label: {
            Label("ดูการทำงานของ Privacy Proxy เบื้องหลัง", systemImage: "eye")
        }
        .font(.callout)
    }

    /// section ที่มีปุ่ม copy และรับ view เนื้อหา (เพื่อไฮไลต์ต้นฉบับได้)
    private func sectionView<Content: View>(_ number: Int, _ title: String, _ color: Color, copy: String,
                                            @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 5) {
                Image(systemName: "\(number).circle.fill").foregroundStyle(color)
                Text(title).font(.caption).bold()
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(copy, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .disabled(copy.isEmpty)
                .help("คัดลอก")
                .accessibilityLabel("คัดลอก \(title)")
            }
            content()
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
                .background(color.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
        }
    }

    /// ไฮไลต์คำที่เป็น PII ในข้อความต้นฉบับ โดยค้นหา substring ของแต่ละ mapping
    static func highlightPII(_ text: String, mappings: [PseudonymMapping]) -> AttributedString {
        var attr = AttributedString(text)
        for mapping in mappings where !mapping.original.isEmpty {
            var searchStart = attr.startIndex
            while let range = attr[searchStart...].range(of: mapping.original) {
                attr[range].backgroundColor = PIIPalette.color(mapping.type).opacity(0.30)
                searchStart = range.upperBound
            }
        }
        return attr
    }
}

// MARK: - หน้าจอเริ่มต้น (ยังไม่มีข้อความ)

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 14) {
            Image("ParivartaLogo")
                .resizable()
                .interpolation(.high)
                .frame(width: 96, height: 96)
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .shadow(color: .black.opacity(0.15), radius: 8, y: 4)
            Text("Parivarta")
                .font(.largeTitle).bold()
            Text("Zero-Trust Privacy Proxy")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("พิมพ์ข้อความที่มีชื่อคน เบอร์โทร หรืออีเมล แล้ว Parivarta จะปกป้องข้อมูลส่วนบุคคลให้อัตโนมัติ\nแทนที่ด้วยรหัสก่อนส่งให้ AI แล้วถอดกลับให้อ่านได้อย่างเป็นธรรมชาติ")
                .font(.callout)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }
}

// MARK: - หน้าจัดการ Vault (ข้อมูลที่เก็บไว้ในเครื่อง)

struct VaultManagementView: View {
    let viewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var entries: [MappingVault.Entry] = []
    @State private var confirmClear = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("ข้อมูลที่เก็บไว้ในเครื่อง", systemImage: "lock.rectangle.stack.fill")
                    .font(.headline)
                Spacer()
                Text("\(entries.count) รายการ").foregroundStyle(.secondary)
            }
            .padding()

            Divider()

            if entries.isEmpty {
                ContentUnavailableView("ยังไม่มีข้อมูลที่เก็บไว้",
                                       systemImage: "lock.open",
                                       description: Text("mapping ระหว่างข้อมูลจริงกับรหัสแทนจะถูกเข้ารหัสและเก็บในเครื่องเมื่อคุณส่งข้อความที่มี PII"))
                    .frame(maxHeight: .infinity)
            } else {
                List(entries) { entry in
                    HStack(spacing: 8) {
                        Circle().fill(PIIPalette.color(entry.type)).frame(width: 8, height: 8)
                        Text(entry.pseudonym)
                            .font(.system(.body, design: .monospaced)).bold()
                        Image(systemName: "arrow.right").foregroundStyle(.tertiary)
                        Text(entry.original)
                            .font(.system(.body, design: .monospaced))
                        Spacer()
                        Text(PIIPalette.label(entry.type))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .textSelection(.enabled)
                }
            }

            Divider()

            HStack {
                Label("เข้ารหัส AES-GCM · คีย์อยู่ใน Keychain", systemImage: "checkmark.shield")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button(role: .destructive) {
                    confirmClear = true
                } label: {
                    Label("ลืมทั้งหมด", systemImage: "trash")
                }
                .disabled(entries.isEmpty)
                Button("เสร็จสิ้น") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding()
        }
        .frame(minWidth: 460, minHeight: 360)
        .onAppear { entries = viewModel.vaultEntries() }
        .confirmationDialog("ลบ mapping ทั้งหมดที่เก็บไว้?",
                            isPresented: $confirmClear, titleVisibility: .visible) {
            Button("ลืมทั้งหมด", role: .destructive) {
                viewModel.clearVault()
                entries = viewModel.vaultEntries()
            }
            Button("ยกเลิก", role: .cancel) {}
        } message: {
            Text("หลังจากนี้ข้อความเดิมที่ถูกแทนที่ด้วยรหัสจะถอดกลับไม่ได้ และข้อมูลเดิมจะได้รหัสใหม่")
        }
    }
}

#Preview {
    ContentView()
}
