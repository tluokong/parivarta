# tools — สร้างโมเดล Thai NER (Core ML)

โมเดล `Parivarta/ThaiNER.mlpackage` (~210 MB) **ไม่ได้รวมไว้ใน repository** เพราะเกินขนาดไฟล์สูงสุดของ GitHub (100 MB)
ต้องรันสคริปต์นี้หนึ่งครั้งก่อน build แอป (The Core ML model is not committed; run this script once before building.)

## วิธีสร้างโมเดล

```bash
# 1) สร้าง virtualenv + ติดตั้ง dependency (ครั้งเดียว)
python3 -m venv tools/venv
tools/venv/bin/pip install -r tools/requirements.txt

# 2) แปลงโมเดล (ดาวน์โหลด WangchanBERTa NER แล้วแปลงเป็น Core ML)
tools/venv/bin/python tools/convert_thai_ner.py
```

สคริปต์จะเขียนไฟล์พร้อมใช้ลงใน `Parivarta/` โดยตรง:

- `ThaiNER.mlpackage` — โมเดล Core ML (fixed shape 1×512, float16)
- `thai_ner_tokenizer.json`, `thai_ner_tokenizer_config.json`, `thai_ner_labels.json`

## Patch ที่สคริปต์ทำอัตโนมัติ (จำเป็น)

swift-transformers ฝั่ง Swift จะพังถ้าไม่ patch สิ่งเหล่านี้:

1. **`tokenizer_class`: `CamembertTokenizer` → `XLMRobertaTokenizer`** — บังคับให้ใช้ `UnigramTokenizer` (SentencePiece) ให้ตรงกับโมเดล
2. **dedup token `<mask>` ที่ซ้ำใน vocab** — `Dictionary(uniqueKeysWithValues:)` ของ Swift จะ trap ถ้ามี key ซ้ำ (id คงเดิมเพื่อไม่ให้ alignment เพี้ยน)

ที่มา: `pythainlp/thainer-corpus-v2-base-model` (WangchanBERTa token classification)
