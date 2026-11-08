#!/usr/bin/env python3
"""
แปลงโมเดล Thai NER (WangchanBERTa token-classification) เป็น Core ML (.mlpackage)
พร้อม export tokenizer.json และ labels.json สำหรับใช้ฝั่ง Swift

รัน:
    python3 tools/convert_thai_ner.py

ผลลัพธ์อยู่ใน tools/output/
"""

import json
import os
import shutil
import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_ID = "pythainlp/thainer-corpus-v2-base-model"
MAX_LEN = 512          # ความยาวคงที่ (ใช้ padding) เพื่อให้ output shape นิ่ง — Core ML runtime เสถียร
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
# โฟลเดอร์ source ของแอป — สคริปต์จะเขียนไฟล์พร้อมใช้ (patch แล้ว) ลงที่นี่โดยตรง
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Parivarta"))


class TokenClassifierWrapper(torch.nn.Module):
    """ห่อโมเดลให้รับ (input_ids, attention_mask) แล้วคืน logits ตัวเดียว"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"⏬ กำลังโหลดโมเดล: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    model.eval()

    id2label = model.config.id2label
    print(f"🏷️  ป้ายกำกับ ({len(id2label)}): {list(id2label.values())}")

    # trace ด้วย input ตัวอย่าง
    dummy_ids = torch.ones(1, MAX_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long)
    wrapper = TokenClassifierWrapper(model).eval()

    print("🔧 กำลัง trace โมเดล...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask))

    print("🍏 กำลังแปลงเป็น Core ML (fixed shape 1x%d)..." % MAX_LEN)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, MAX_LEN), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, MAX_LEN), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,   # ลดขนาดโมเดล ~ครึ่งหนึ่ง
    )
    mlmodel.short_description = "Thai NER (WangchanBERTa) token classification"

    model_path = os.path.join(OUT_DIR, "ThaiNER.mlpackage")
    mlmodel.save(model_path)
    print(f"✅ บันทึกโมเดล: {model_path}")

    # tokenizer.json (fast tokenizer) + ไฟล์ประกอบ
    tokenizer.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
    print("✅ บันทึก tokenizer")

    # labels.json — แม็ป index → ป้ายกำกับ (เรียงตาม index)
    labels = [id2label[i] for i in range(len(id2label))]
    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"✅ บันทึก labels.json: {labels}")

    # นำไฟล์ (พร้อม patch ที่จำเป็น) ไปวางในโฟลเดอร์แอปให้พร้อม build
    finalize_into_app(model_path, os.path.join(OUT_DIR, "tokenizer"), labels)

    print("\n🎉 แปลงเสร็จ! กำลังทดสอบ prediction ด้วย Core ML...\n")
    verify(model_path, tokenizer, labels)


def finalize_into_app(model_path, tokenizer_dir, labels):
    """คัดลอกโมเดล + tokenizer (patch แล้ว) + labels ลงโฟลเดอร์แอป ให้พร้อม build ทันที

    Patch ที่จำเป็น (ไม่งั้น swift-transformers จะ crash / ใช้ tokenizer ผิดชนิด):
      1. tokenizer_class: CamembertTokenizer -> XLMRobertaTokenizer (บังคับใช้ UnigramTokenizer)
      2. dedup token '<mask>' ที่ซ้ำใน vocab (Dictionary(uniqueKeysWithValues:) ของ Swift จะ trap)
    """
    os.makedirs(APP_DIR, exist_ok=True)

    # 1) โมเดล .mlpackage
    dst_model = os.path.join(APP_DIR, "ThaiNER.mlpackage")
    if os.path.exists(dst_model):
        shutil.rmtree(dst_model)
    shutil.copytree(model_path, dst_model)

    # 2) tokenizer.json + dedup <mask>
    with open(os.path.join(tokenizer_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok_data = json.load(f)
    vocab = tok_data["model"]["vocab"]
    seen = set()
    for i, entry in enumerate(vocab):
        token = entry[0]
        if token in seen:
            entry[0] = f"{token}_dup_{i}"      # เปลี่ยนชื่อตัวซ้ำ (id คงเดิม → alignment ไม่เพี้ยน)
        seen.add(token)
    with open(os.path.join(APP_DIR, "thai_ner_tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False)

    # 3) tokenizer_config.json + เปลี่ยน tokenizer_class
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["tokenizer_class"] = "XLMRobertaTokenizer"
    with open(os.path.join(APP_DIR, "thai_ner_tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)

    # 4) labels
    with open(os.path.join(APP_DIR, "thai_ner_labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)

    print(f"✅ วางไฟล์พร้อมใช้ (patch แล้ว) ลงใน {APP_DIR}")


def verify(model_path, tokenizer, labels):
    """โหลด Core ML model กลับมา รัน prediction จริง แล้วถอด BIO tag เป็น entity"""
    mlmodel = ct.models.MLModel(model_path)
    samples = [
        "นายสมชาย เลิศวิสัย โทร 02-123-4567 อีเมล somchai@example.com",
        "พบผู้ต้องสงสัยชื่อ มาลี ใจดี อาศัยอยู่ที่จังหวัดเชียงใหม่",
        "บริษัท การบินไทย จำกัด มหาชน ตั้งอยู่ที่กรุงเทพมหานคร",
    ]
    for text in samples:
        enc = tokenizer(text, return_tensors="np", return_offsets_mapping=True,
                        padding="max_length", max_length=MAX_LEN, truncation=True)
        offsets = enc["offset_mapping"][0]
        out = mlmodel.predict({
            "input_ids": enc["input_ids"].astype(np.int32),
            "attention_mask": enc["attention_mask"].astype(np.int32),
        })
        logits = out["logits"][0]                 # [seq, num_labels]
        pred_ids = logits.argmax(axis=-1)

        # ถอด BIO → รวม span
        entities = []
        cur_type, cur_start, cur_end = None, None, None
        for i, (pid, (s, e)) in enumerate(zip(pred_ids, offsets)):
            if s == e:                            # special/padding token
                continue
            label = labels[pid]
            if label.startswith("B-"):
                if cur_type:
                    entities.append((cur_type, cur_start, cur_end))
                cur_type, cur_start, cur_end = label[2:], s, e
            elif label.startswith("I-") and cur_type == label[2:]:
                cur_end = e
            else:
                if cur_type:
                    entities.append((cur_type, cur_start, cur_end))
                cur_type, cur_start, cur_end = None, None, None
        if cur_type:
            entities.append((cur_type, cur_start, cur_end))

        print(f"INPUT: {text}")
        for etype, s, e in entities:
            print(f"   [{etype}] '{text[s:e]}'")
        print("---")


if __name__ == "__main__":
    main()
