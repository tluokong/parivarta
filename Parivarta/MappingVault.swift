//
//  MappingVault.swift
//  Parivarta
//
//  คลังเก็บ mapping ระหว่างข้อมูลจริง (เข้ารหัส) กับรหัสแทน แบบถาวร
//  แทนที่ Fernet + SQLite เดิมด้วย CryptoKit (AES-GCM) + ไฟล์ JSON + Keychain
//

import Foundation
import CryptoKit
import Security

final class MappingVault {
    /// ระเบียนหนึ่งรายการในคลัง
    private struct Record: Codable {
        let pseudonym: String
        let type: String
        let ciphertext: Data      // ข้อมูลจริงที่ถูกเข้ารหัสด้วย AES-GCM
    }

    /// โครงสร้างข้อมูลที่บันทึกลงดิสก์
    private struct Store: Codable {
        var records: [String: Record] = [:]   // key = SHA-256 ของข้อมูลจริง (normalize แล้ว)
        var counters: [String: Int] = [:]     // key = ประเภท entity → เลขลำดับล่าสุด
    }

    private let key: SymmetricKey
    private let fileURL: URL
    private var store: Store

    init() {
        self.key = MappingVault.loadOrCreateKey()
        self.fileURL = MappingVault.storeURL()
        self.store = MappingVault.load(from: fileURL) ?? Store()
    }

    /// รายการหนึ่งใน vault สำหรับแสดง/จัดการ (ถอดรหัสข้อมูลจริงแล้ว)
    struct Entry: Identifiable, Hashable {
        var id: String { pseudonym }
        let pseudonym: String
        let type: String
        let original: String
    }

    private func hashKey(_ value: String) -> String {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return SHA256.hash(data: Data(normalized.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    /// รหัสแทนที่เคยกำหนดไว้แล้วสำหรับค่านี้ (nil = ยังไม่เคยพบ) — อ่านอย่างเดียว ไม่สร้างใหม่
    func existingPseudonym(for value: String) -> String? {
        store.records[hashKey(value)]?.pseudonym
    }

    /// คืนรหัสแทนสำหรับข้อมูล PII (ใช้รหัสเดิมถ้าเคยพบแล้ว ไม่งั้นสร้างใหม่)
    func pseudonym(for value: String, type: String) -> String {
        let hash = hashKey(value)
        if let existing = store.records[hash] {
            return existing.pseudonym
        }

        let next = (store.counters[type] ?? 0) + 1
        store.counters[type] = next
        let pseudonym = String(format: "%@_%03d", type, next)

        store.records[hash] = Record(pseudonym: pseudonym,
                                     type: type,
                                     ciphertext: encrypt(value))
        persist()
        return pseudonym
    }

    /// ทุกรายการที่เก็บไว้ (ถอดรหัสแล้ว) เรียงตามรหัสแทน — สำหรับหน้าจัดการ vault
    func entries() -> [Entry] {
        store.records.values.compactMap { record in
            guard let original = decrypt(record.ciphertext) else { return nil }
            return Entry(pseudonym: record.pseudonym, type: record.type, original: original)
        }
        .sorted { $0.pseudonym < $1.pseudonym }
    }

    /// ลบ mapping ทั้งหมด (ปุ่ม "ลืมทั้งหมด")
    func removeAll() {
        store = Store()
        persist()
    }

    // MARK: - การเข้ารหัส (AES-GCM)

    private func encrypt(_ text: String) -> Data {
        let sealed = try? AES.GCM.seal(Data(text.utf8), using: key)
        return sealed?.combined ?? Data()
    }

    /// ถอดรหัสข้อมูลจริงกลับ (ไว้สำหรับฟีเจอร์ตรวจสอบย้อนหลังในอนาคต)
    func decrypt(_ data: Data) -> String? {
        guard let box = try? AES.GCM.SealedBox(combined: data),
              let plain = try? AES.GCM.open(box, using: key) else {
            return nil
        }
        return String(data: plain, encoding: .utf8)
    }

    // MARK: - การบันทึกลงดิสก์

    private func persist() {
        guard let data = try? JSONEncoder().encode(store) else { return }
        try? FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? data.write(to: fileURL, options: .atomic)
    }

    private static func load(from url: URL) -> Store? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Store.self, from: data)
    }

    private static func storeURL() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return base.appendingPathComponent("Parivarta/vault.json")
    }

    // MARK: - Master Key เก็บใน Keychain

    private static let keychainAccount = "Lanna.Parivarta.vaultKey"

    private static func loadOrCreateKey() -> SymmetricKey {
        if let data = keychainRead(account: keychainAccount) {
            return SymmetricKey(data: data)
        }
        let newKey = SymmetricKey(size: .bits256)
        let raw = newKey.withUnsafeBytes { Data($0) }
        keychainWrite(account: keychainAccount, data: raw)
        return newKey
    }

    private static func keychainRead(account: String) -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        return status == errSecSuccess ? result as? Data : nil
    }

    private static func keychainWrite(account: String, data: Data) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
        ]
        SecItemDelete(query as CFDictionary)
        SecItemAdd(query as CFDictionary, nil)
    }
}
