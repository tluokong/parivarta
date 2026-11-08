// สร้างไอคอนแอป Parivarta ขนาด 1024×1024 (โล่ + รูกุญแจ บนพื้นไล่เฉดคราม→น้ำเงิน)
// รัน: swift tools/make_icon.swift <output.png>
import AppKit
import CoreText

/// สร้าง CGPath ของข้อความ (รวม glyph ทั้งหมด) เพื่อใช้เจาะทะลุโล่
func letterPath(_ text: String, fontName: String, fontSize: CGFloat) -> CGPath {
    let font = NSFont(name: fontName, size: fontSize) ?? NSFont.boldSystemFont(ofSize: fontSize)
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: [.font: font]))
    let path = CGMutablePath()
    for run in (CTLineGetGlyphRuns(line) as! [CTRun]) {
        let attrs = CTRunGetAttributes(run) as NSDictionary
        let runFont = attrs[kCTFontAttributeName as String] as! CTFont
        for i in 0..<CTRunGetGlyphCount(run) {
            var glyph = CGGlyph()
            var pos = CGPoint()
            CTRunGetGlyphs(run, CFRangeMake(i, 1), &glyph)
            CTRunGetPositions(run, CFRangeMake(i, 1), &pos)
            if let gp = CTFontCreatePathForGlyph(runFont, glyph, nil) {
                path.addPath(gp, transform: CGAffineTransform(translationX: pos.x, y: pos.y))
            }
        }
    }
    return path
}

let size = 1024.0
let outPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/parivarta_icon_1024.png"

let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size), pixelsHigh: Int(size),
                           bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                           colorSpaceName: .calibratedRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

// พื้นโปร่งใส
NSColor.clear.set()
NSRect(x: 0, y: 0, width: size, height: size).fill()

// สี่เหลี่ยมมุมมนสไตล์ macOS (มี margin รอบ)
let margin = 100.0
let squareRect = NSRect(x: margin, y: margin, width: size - margin * 2, height: size - margin * 2)
let square = NSBezierPath(roundedRect: squareRect, xRadius: 185, yRadius: 185)
square.addClip()

// ไล่เฉดสีแนวตั้ง คราม → น้ำเงินเข้ม
let top = NSColor(srgbRed: 0.31, green: 0.27, blue: 0.90, alpha: 1)      // #4F46E5
let bottom = NSColor(srgbRed: 0.11, green: 0.31, blue: 0.85, alpha: 1)   // #1D4ED8
NSGradient(starting: bottom, ending: top)!.draw(in: squareRect, angle: 90)

// ไฮไลต์นุ่ม ๆ ไล่จากล่าง (โปร่ง) ขึ้นบน (สว่างจาง) เต็มความสูง — ไม่มีรอยต่อ
let highlight = NSGradient(colors: [NSColor(white: 1, alpha: 0.0), NSColor(white: 1, alpha: 0.16)])!
highlight.draw(in: squareRect, angle: 90)

NSGraphicsContext.current?.restoreGraphicsState()  // ยกเลิก clip
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

// --- โล่ + รูกุญแจ (เจาะทะลุด้วย even-odd) ---
let cx = 512.0
let shield = NSBezierPath()
shield.move(to: NSPoint(x: cx, y: 792))                       // ยอดบนกลาง
shield.curve(to: NSPoint(x: 690, y: 720), controlPoint1: NSPoint(x: 620, y: 792), controlPoint2: NSPoint(x: 690, y: 770))
shield.line(to: NSPoint(x: 690, y: 545))                      // ข้างขวา
shield.curve(to: NSPoint(x: cx, y: 250), controlPoint1: NSPoint(x: 690, y: 400), controlPoint2: NSPoint(x: 610, y: 315))
shield.curve(to: NSPoint(x: 334, y: 545), controlPoint1: NSPoint(x: 414, y: 315), controlPoint2: NSPoint(x: 334, y: 400))
shield.line(to: NSPoint(x: 334, y: 720))                      // ข้างซ้าย
shield.curve(to: NSPoint(x: cx, y: 792), controlPoint1: NSPoint(x: 334, y: 770), controlPoint2: NSPoint(x: 404, y: 792))
shield.close()

// ตัวอักษร "P" เจาะทะลุโล่ (โชว์เฉดสีพื้นผ่านช่องตัวอักษร)
let rawP = letterPath("P", fontName: "Helvetica-Bold", fontSize: 500)
let bb = rawP.boundingBoxOfPath
let targetH = 330.0
let scale = targetH / bb.height
let tx = cx - scale * bb.midX             // จัดกึ่งกลางแนวนอนที่ cx
let ty = 540.0 - scale * bb.midY          // จัดกึ่งกลางแนวตั้งกลางโล่
var tf = CGAffineTransform(a: scale, b: 0, c: 0, d: scale, tx: tx, ty: ty)
let pPath = rawP.copy(using: &tf)!

let combined = NSBezierPath()
combined.append(shield)
combined.append(NSBezierPath(cgPath: pPath))
combined.windingRule = .evenOdd
NSColor.white.set()
combined.fill()

NSGraphicsContext.current?.restoreGraphicsState()

// บันทึกเป็น PNG
guard let data = rep.representation(using: .png, properties: [:]) else {
    FileHandle.standardError.write("PNG encode failed\n".data(using: .utf8)!)
    exit(1)
}
try! data.write(to: URL(fileURLWithPath: outPath))
print("✅ wrote \(outPath)")
