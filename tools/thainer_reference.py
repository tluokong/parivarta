#!/usr/bin/env python3
"""
ยืนยันสมมติฐาน Experiment 1: รันโมเดล thainer-corpus-v2 ด้วย pipeline ที่ถูกต้อง
ตาม model card (replace(" ","<_>") + newmm word_tokenize + is_split_into_words)
บนประโยค gold ที่เคย leak แล้วดูว่าโมเดลจับ entity ได้ถูกไหม
"""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pythainlp.tokenize import word_tokenize

MODEL = "pythainlp/thainer-corpus-v2-base-model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL)
model.eval()
id2label = model.config.id2label

SAMPLES = [
    "นายสมชาย เลิศวิสัย โทร 02-123-4567",
    "ประเสริฐ ลือโขง โอนเงินให้ วิชัย มั่งมี",
    "ติดต่อสมหญิงที่เบอร์0891112222",     # เคย leak: สมหญิง
    "กิตติ วัฒนา เป็นพยานในคดีนี้",         # เคย leak: กิตติ วัฒนา
    "ผู้เสียหายชื่อ อรุณี พบเงินโอน",        # เคย leak: อรุณี
    "บริษัท การบินไทย จำกัด มหาชน ตั้งอยู่ที่กรุงเทพมหานคร",
    "เดินทางจากภูเก็ตไปเชียงราย",
]

def entities(sentence: str):
    cut = word_tokenize(sentence.replace(" ", "<_>"), engine="newmm")
    enc = tok(cut, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"], attention_mask=enc["attention_mask"])[0]
    pred = torch.argmax(logits, dim=2)[0].tolist()
    word_ids = enc.word_ids(0)
    # ป้ายระดับคำ = ป้ายของ subword แรกของแต่ละคำ
    word_label = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_label:
            word_label[wid] = id2label[pred[pos]]
    # รวม BIO เป็น entity (คำ)
    ents, cur_t, cur_toks = [], None, []
    for i, w in enumerate(cut):
        lab = word_label.get(i, "O")
        base = lab[2:] if "-" in lab else "O"
        if lab.startswith("B-") or (lab.startswith("I-") and base != cur_t):
            if cur_t: ents.append((cur_t, "".join(cur_toks)))
            cur_t, cur_toks = base, [w]
        elif lab.startswith("I-") and base == cur_t:
            cur_toks.append(w)
        else:
            if cur_t: ents.append((cur_t, "".join(cur_toks)))
            cur_t, cur_toks = None, []
    if cur_t: ents.append((cur_t, "".join(cur_toks)))
    return [(t, s.replace("<_>", " ")) for t, s in ents]

for s in SAMPLES:
    print("INPUT:", s)
    for t, txt in entities(s):
        print(f"   [{t}] '{txt}'")
    print("---")
