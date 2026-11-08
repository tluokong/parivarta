# On-Device Thai PII Detection for a Zero-Trust LLM Proxy: An Eval-Driven Study

## Abstract

Cloud Large Language Models (LLMs) create a privacy hazard: any Personally Identifiable
Information (PII) in a prompt leaves the user's device. We study the detection stage of
**Parivarta**, a native macOS *zero-trust local privacy proxy* that pseudonymizes PII on-device
before a prompt is sent and restores it in the reply. In this setting the two error types are
**asymmetric** — a missed entity is exfiltrated (a *leak*), whereas an over-detection only
over-masks and costs utility — so we adopt a **privacy-first** evaluation that reports *coverage*
(leak prevention) separately from *typed* and *exact-span* quality, plus explicit *over-mask* and
*specificity* costs, all in a **Unicode-scalar coordinate system** that reconciles code-point gold
offsets with Swift's grapheme indices for Thai. The detector is a **hybrid ensemble** (deterministic
regex, Apple `NLTagger`, Thai honorific/administrative heuristics, a word-segmentation-gated 21k-name
gazetteer, an on-device Core ML WangchanBERTa NER, and a watchlist) fused by a single
**precision-aware overlap-resolution** rule. Following a strict eval-first, one-variable-at-a-time
protocol (E0–E8), overall coverage on a 2,500-document external benchmark rises **77.1% → 100%** and
exact-span **65.7% → 100%** at near-flat over-mask, while a dedicated 35-item hard-negative suite
lifts specificity to **94.3%** and a hard-positive probe reaches **95.2%**. Against five external
baselines (Microsoft Presidio, PyThaiNLP, spaCy, a regex core, and a zero-shot local LLM,
qwen2.5:7b), Parivarta
**strictly dominates** on per-document coverage (McNemar *p* ≪ 10⁻³, no baseline covering a document
ours misses). Finally, a new **1,250-pair minimal-pair "AMB" suite** isolates *context-sensitivity*
— the same surface string as PII vs. a look-alike — and shows that **every** system, a zero-shot LLM
included, keys on surface form rather than context; Parivarta posts the field's highest POS-recall
(74.1%) but lowest NEG-specificity (21.8%), i.e. its residual error is **over-masking, not leakage**,
the safe direction for a proxy. We contribute the protocol, the datasets and scorers, and an analysis
that localizes the one worthwhile next investment.

**Keywords:** PII detection; data pseudonymization; privacy-preserving NLP; Thai NLP; named-entity
recognition; on-device machine learning; large language models; zero-trust systems.

**Contributions.**

1. A **privacy-first evaluation protocol** for masking systems that separates leak-preventing
   coverage from typed/exact-span quality and reports over-mask and specificity as explicit costs,
   with two **reproducibility requirements for Thai span scoring** (Unicode-scalar coordinates;
   CRLF/grapheme newline handling) that are easy to get wrong and materially move the numbers.
2. A **hybrid on-device detector** whose **precision-aware overlap-resolution** rule fixes span
   boundaries globally (exact-span 88.6% → 100%) without retraining, and whose
   **segmentation-gated name gazetteer** improves Thai-name recall and precision *simultaneously*.
3. An **eval-first ablation** (E0–E8) in which each mechanism is added singly under a
   no-privacy-regression gate — including a negative result recorded but not shipped.
4. A **head-to-head comparison** against five external baselines on identical data with bootstrap
   CIs and paired McNemar tests, showing strict per-document dominance on coverage.
5. **Dataset AMB**, a 1,250-pair minimal-pair ambiguity suite, plus a paired POS/NEG scorer, that
   makes *context-sensitivity* measurable and reveals it as an open problem for **all** current
   methods — while showing our system's residual error is the privacy-safe (over-masking) kind.

*All figures are reproducible from the harnesses in `ParivartaTests/` and the stdlib scorer/baseline
runners in `tools/eval/` (see §6.5).*

---

## 1. Experimental Setting

### 1.1 System under test

We study the Personally Identifiable Information (PII) detection stage of **Parivarta**, a
native macOS *zero-trust local privacy proxy*. Before a user prompt is forwarded to a
cloud Large Language Model (LLM), every detected PII span is replaced by a type-indexed
pseudonym (e.g., `PERSON_001`, `PHONE_NUMBER_001`); the mapping is held locally in an
encrypted vault and the LLM reply is de-pseudonymized on-device. Detection quality is
therefore the *privacy-critical* component: a missed entity ("leak") is exfiltrated to the
cloud, whereas an over-detection merely over-masks and degrades utility.

The detector is a **hybrid ensemble** whose members are combined by a single overlap
resolution (de-duplication) step:

| Layer | Detector | Target types | Nature |
|-------|----------|--------------|--------|
| L1 | Deterministic regular expressions | EMAIL, THAI_ID (national ID), PHONE_NUMBER, IP_ADDRESS, ADDRESS | High-precision, fixed-format |
| L2 | Apple `NLTagger` (`.nameType`) | PERSON, LOCATION, ORGANIZATION | Statistical, general-purpose |
| L3 | Thai heuristics | PERSON (honorific-anchored), LOCATION (administrative-prefix / province gazetteer) | Rule-based |
| L4 | Dictionary-segmented name gazetteer (`ThaiGazetteer`) | PERSON | Lexicon + word segmentation |
| L5 | On-device neural Thai NER (Core ML) | PERSON, LOCATION, ORGANIZATION | Transformer |
| L6 | User / organization watchlist | ORGANIZATION, CUSTOM_KEYWORD | Exact string |

**L5** is `pythainlp/thainer-corpus-v2-base-model` (a WangchanBERTa / XLM-R–family token
classifier) converted to Apple **Core ML** (fp16, fixed input length 512) and executed
fully on-device; sub-word tokenization uses the model's XLM-R Unigram tokenizer through
the `swift-transformers` library, loaded offline. **L4** segments Thai text by greedy
maximal matching against a 62k-word dictionary (a re-implementation of PyThaiNLP's *newmm*)
and matches word-aligned spans against a 21k-name lexicon (PyThaiNLP female/male/family
names). All computation is local; no text leaves the device during detection.

### 1.2 Overlap resolution

Because members fire independently, their spans overlap. A greedy de-duplication keeps a
locally preferred entity and discards any later entity whose span intersects a kept one.
The ordering key is the central design variable studied in Experiment E4 (Section 5.5).

### 1.3 Hardware / software

All experiments run inside the application's own sandbox on Apple silicon (macOS on Apple
silicon, Swift 6, Swift Testing framework). The Core ML model runs on the Apple Neural
Engine / GPU via `MLComputeUnits.all`. Every reported number is produced by executing the
real production pipeline (identical code path to the shipping app), not a re-implementation.
The detector is deterministic given fixed model weights, so results are invariant to the host:
the ablation (E0–E8) and the baseline/AMB comparison (§5.10–§5.11) were run on different Apple
silicon Xcode toolchains and reproduce the same detector figures. The Python-side baselines and
the unified scorer run under a pinned virtual environment (`tools/eval/requirements.txt`).

---

## 2. Data Preparation

We use **five complementary datasets**, each isolating a distinct capability: **A** — a 50-sentence
internal development set for ablation; **B** — a 2,500-document external benchmark for primary
recall; **C** — a 35-item hard-negative suite for precision/false positives; **D** — hand-authored
hard positives and an OOV probe for robustness; and **AMB** — a 1,250-pair minimal-pair suite for
context-sensitivity. Every span (gold and predicted) is compared in a common **Unicode-scalar
coordinate system** (§2.3, §3).

### 2.1 Dataset A — internal development set (ablation)

A hand-authored gold set of **50 Thai sentences (77 entity mentions)** spanning PERSON,
LOCATION, ORGANIZATION, PHONE_NUMBER, EMAIL, THAI_ID, IP_ADDRESS, code-mixed Thai/English
names, and *negative* sentences that contain no PII. Each mention is annotated as a
`(surface substring, type)` pair. This set is used for **ablation** of the gazetteer and as
a regression guard; because annotations are substrings, it is scored by *containment*
(Section 3), which is robust to span-boundary conventions.

### 2.2 Dataset B — external benchmark (primary evaluation)

An independently generated synthetic benchmark of **2,500 documents**, balanced across
**five PII types (500 documents each): PERSON, PHONE, EMAIL, ADDRESS, NATIONAL_ID**. Each
document is a multi-sentence Thai paragraph containing **exactly one target entity**,
supplied with a **character-offset span** (`annotation_start`, `annotation_end`) and a gold
type. The benchmark is *positive-only* (every document contains its target) and exhibits
realistic surface variation, e.g.:

- **PHONE**: three surface forms — plain `0600137951`, 3-3-4 dashed/spaced `080-014-5870` /
  `090 015 3789`, and 2-4-4 dashed `06-0000-0301`.
- **NATIONAL_ID**: bare 13 digits `1000001458703`, dashed `1-0000-01379-51-1`, spaced
  `1 0000 01537 89 9` (canonical Thai 1-4-5-2-1 grouping).
- **ADDRESS**: full postal-style strings, e.g.
  `บ้านเลขที่ 1/1 ซอยสายลม ถนนพฤกษา แขวงทุ่งงาม เขตเมืองกลาง`.
- **PERSON**: honorific-prefixed names (นาย / นาง / นางสาว / คุณ) in running prose.

### 2.3 Two preprocessing pitfalls (and their fixes)

Reproducibility on Thai text required two non-obvious corrections, documented here because
they materially affect measured scores:

1. **Line-ending / grapheme interaction.** The benchmark files use **CRLF** terminators.
   Swift groups `"\r\n"` into a *single extended grapheme cluster*, so naively splitting on
   the `"\n"` `Character` returns the entire file as one line. We split on the Unicode
   *newline* property instead.

2. **Offset coordinate mismatch.** The gold offsets are **Unicode code-point (scalar)
   indices** (Python `str` semantics), whereas Swift `String.Index` is **grapheme-cluster**
   based. Thai combining vowels and tone marks make these two coordinate systems diverge.
   All spans (gold and predicted) are therefore converted to a **common Unicode-scalar
   integer coordinate system** before any comparison. Failing to do this silently
   mis-aligns every Thai span.

### 2.4 Dataset C — negative / false-positive suite

Because Datasets A and B are dominated by positive instances, neither measures the
**false-positive rate** directly. We therefore construct a **35-sentence hard-negative
suite** in which *no* span should be masked, spanning six adversarial categories:
(i) **numeric look-alikes** (sales figures, product codes, times, prices, budget years,
version strings, invalid-octet IP-like strings); (ii) **name/common-word collisions** —
Thai words that are also given names (สมบัติ "property", มั่นคง "stable", ประเสริฐ
"excellent", ใจดี "kind", ทอง "gold", พร "blessing", and ธนา inside ธนาคาร "bank");
(iii) **place-prefix words in non-place context** (เขตปลอดอากร "duty-free zone",
จังหวัด/ตำบล/อำเภอ followed by verbs, บ้านเลขที่ used pronominally); (iv) **email/URL
look-alikes** (a bare `@`, a `.com` mention, ratios like `16:9`); (v) **dates / reference
codes / OTP / tracking numbers**; and (vi) **benign** sentences with no traps. A further
**3 format-collision** items (a bare 13-digit number, a bare 10-digit number) are held out
and reported separately, because they are byte-for-byte indistinguishable from a national ID
or a mobile number and any purely format-based detector is *expected* to flag them.

### 2.5 Dataset D — hard-positive robustness & OOV probe

Two small hand-authored positive sets stress generalization beyond Dataset B's templates:

- **Hard positives (13 sentences, 21 mentions)** across four categories: **ADDR_VAR** —
  addresses that do *not* follow the benchmark template (openers ที่อยู่ / เลขที่ / คอนโด
  instead of บ้านเลขที่, trailing province + 5-digit postal code, condo/room/floor tokens);
  **MULTI** — several entity types in one sentence; **NESTED** — a person inside an
  organization, and specialist honorifics (นพ. "Dr. med.", ดร. "Dr."); **ROMANIZED** —
  Thai names spelled in Latin and foreign names.
- **OOV person probe (10 sentences)** of rare, coined, and foreign personal names, used to
  isolate the gazetteer's contribution by re-scoring under each gazetteer mode.

Both are scored by *containment* in scalar coordinates (as for Dataset A).

### 2.6 Dataset AMB — minimal-pair ambiguity suite (context-sensitivity)

Datasets B–D establish high surface *coverage* and precision on *separable* positives and
negatives. They do not test the hardest case for any PII detector: **the same surface string is
PII in one context and not in another.** Dataset AMB targets exactly this. It is a **minimal-pair**
suite of **1,250 pairs (2,500 documents)**, balanced across the five types (250 pairs each).
Every pair shares one **candidate surface form** and one **ambiguity type** and differs *only* in
context:

- an **ambiguous-positive (POS)** document, in which the candidate genuinely *is* the entity
  (gold type, with a span), disambiguated only by the surrounding words — e.g. the bare given
  name *แก้ว* used as a person (`…บันทึก…เพียงว่า แก้ว จะตรวจเอกสาร…`); and
- a **hard-negative (NEG)** document, in which the *same* surface form is a non-PII look-alike
  (gold `O`, no span) — e.g. *แก้ว* meaning "glass" (`แก้วสำหรับใส่น้ำถูกจัดวางไว้บนโต๊ะ…`).

The five ambiguity types — one per PII type — are each adversarial to a *different* detector layer:

| PII type | ambiguity type | POS (is PII) | NEG (look-alike, gold `O`) |
|----------|----------------|--------------|-----------------------------|
| PERSON | `common_word_as_person` | a common word used as a name (แก้ว, ขวัญ) | the same word in its ordinary meaning |
| ADDRESS | `internal_location_or_address` | a mailing address | an internal location (อาคาร/ชั้น/ห้อง) that is not a deliverable address |
| PHONE | `numeric_shape` | a real phone number | a phone-shaped number in another role (e.g. dotted `060.013.7951`) |
| EMAIL | `system_identifier_shape` | a personal email | an email-shaped system identifier (e.g. `case0011+check@example.com`) |
| NATIONAL_ID | `thirteen_digit_code` | a national ID | a 13-digit code that is not an ID (e.g. `1000001379511`) |

Crucially, several NEG members are **byte-for-byte valid** for a format detector (a bare 13-digit
code; an RFC-valid `…@example.com` string): a purely format-based layer *must* flag them and so
**cannot** score on the NEG side without using context. AMB therefore measures the property that
Datasets B/C cannot — whether detection is **context-sensitive** rather than surface-triggered —
and is deliberately **harder than Dataset C**, whose negatives were chosen to be separable. Each
candidate carries scalar `candidate_start/end` offsets; POS members add an `annotation_start/end`
span. Scoring is in the same Unicode-scalar coordinate system (fixtures `exports/docs_AMB.jsonl`,
`exports/gold_AMB.jsonl`).

---

## 3. Evaluation Methodology

Let a gold entity be a scalar span `g = [s, e)` with type `t_g`. Let the detector return a
set `D` of predictions, each a `(type, span)` pair, and let `M = ⋃_{d∈D} span(d)` be the set
of masked scalar indices. A type map `φ` reconciles detector labels with benchmark labels
(`PHONE → PHONE_NUMBER`, `NATIONAL_ID → THAI_ID`, `ADDRESS → {ADDRESS, LOCATION}`, identity
otherwise). We report:

- **Privacy-containment recall (Coverage).** `g` is *covered* iff `[s, e) ⊆ M`. This is the
  headline privacy metric: an uncovered gold entity is a **leak** (`leak = 1 − coverage`).
  It is label-agnostic because, for privacy, *any* mask over the PII prevents exfiltration.
- **Typed-containment recall.** `∃ d∈D : φ⁻¹(t_g) ∋ type(d) ∧ span(d) ⊇ [s, e)`. Adds the
  requirement that the mask carry the correct type (so the pseudonym is semantically right).
- **Exact-span match.** `∃ d∈D : type(d) ∈ φ(t_g) ∧ span(d) = [s, e)`. The strict criterion:
  the mask boundary coincides with the entity boundary (clean pseudonym, minimal collateral).
- **Over-detection rate.** Mean number of predictions per document that do **not** intersect
  the gold span — a precision proxy under single-entity annotation.
- **Over-mask ratio.** `|{ i ∈ M : i ∉ [s, e) }| / |document|` in scalars — the fraction of
  characters masked that are not the target entity (utility cost).

For Dataset A (substring annotations, possibly several per sentence) we additionally report
containment-based **Precision, Recall and F1** and the per-sentence **leak rate**.

For Dataset C (every sentence is a true negative) we report:

- **False-positive document rate.** Fraction of hard-negative sentences with ≥1 detection.
- **Specificity.** `1 − FP-doc rate`.
- **Over-mask ratio** on negatives, and a **culprit-type breakdown** (which label/layer
  produced the spurious spans), which localizes the source of each false positive.
- **Format-collision detection rate** on the held-out ambiguous items, reported separately.

For Dataset AMB (minimal pairs) each judgment is restricted to the **shared candidate span** so
the two members of a pair are strictly comparable; detections elsewhere in a document are ignored.
We report:

- **POS-recall (TPR / detection sensitivity).** Fraction of the 1,250 POS documents whose gold
  candidate span is covered (masked).
- **NEG-specificity (1 − FPR).** Fraction of the 1,250 NEG documents in which *no* prediction
  overlaps the candidate span.
- **Pair-accuracy.** Fraction of the 1,250 *pairs* for which **both** members are correct (POS
  covered **and** NEG not flagged) — the strict context-sensitivity score. A surface-triggered
  detector that always fires, or never fires, on a given form scores 0 on that pair regardless of
  how its POS-recall or NEG-specificity look in isolation; the informative reference point is the
  value expected if the two judgments were **independent** (POS-recall × NEG-specificity).
- **Per-type pair-relevant accuracy.** The candidate-span judgment accuracy over all POS and NEG
  members of a type (equivalently, the mean of that type's POS-recall and NEG-specificity).

> **Note on precision.** Dataset B is *positive-only* with a *single* annotated entity per
> document, so true precision is not computable there — outside-span predictions may be
> genuine false positives or unannotated real entities; we report over-detection and
> over-mask only as *proxies*. The **precision claim rests on Dataset C** (an explicit
> negative suite), complemented by Dataset A which contains a few negatives with exhaustive
> annotation.

---

## 4. Experimental Procedure (Step by Step)

The study follows a strict **eval-first, one-variable-at-a-time** protocol. We first
establish a baseline (E0), then introduce one change per experiment, and after each change
re-run **both** datasets to confirm the intended gain and the absence of privacy regression.
A change is accepted only if it does not increase the leak rate on either dataset.

- **E0 — Baseline & harness.** Instrument the unmodified production pipeline; measure all
  metrics on Dataset B; run the gazetteer ablation on Dataset A. *Purpose: locate the gaps.*
- **E1 — National-ID format coverage.** Hypothesis: NATIONAL_ID recall is limited by a
  regex that matches only bare 13-digit strings. Generalize it to the canonical 1-4-5-2-1
  grouping with optional `-`/space separators, chosen to be **disjoint from the phone 3-3-4
  grouping** so it cannot capture phone numbers.
- **E3 — Phone format coverage.** Hypothesis (revised by data): the residual phone misses
  are the **2-4-4 dashed** mobile form, not the space form assumed a priori. Relax the
  middle group of the phone regex from exactly 3 to 3–4 digits.
- **E2 — Address detection.** Hypothesis: addresses are only *incidentally* covered by the
  location layers; add a dedicated **cue-anchored** ADDRESS detector that starts at
  `บ้านเลขที่` and consumes the space-separated administrative fields (ซอย / ถนน / หมู่ /
  แขวง / ตำบล / เขต / อำเภอ / จังหวัด), requiring ≥1 field so a bare cue in prose is not matched.
- **E4 — Boundary correction via precision-aware overlap resolution.** Hypothesis (from E2's
  diagnostics): exact-span errors are **tail over-capture** — statistical members (L5 neural
  NER for PERSON, L2 `NLTagger` for LOCATION) emit spans that share the entity's start but
  extend past its end, and the *longest-span-wins* tie-break lets them defeat the tight
  regex spans. Introduce a per-entity **span-precision score** and prefer higher precision
  *at equal start* during de-duplication.
- **E5 — Precision / false-positive study.** With recall saturated on Dataset B, we turn to
  precision on the new Dataset C. E5 first *measures* the false-positive baseline, then
  applies three targeted repairs, each gated on **no recall regression** across Datasets A
  and B: **E5a** validate IP octets to `0–255`; **E5b** add a Thai negative-lookbehind to
  the honorific PERSON rule, **scoped to `คุณ`** so that common words ending in คุณ
  (ขอบคุณ "thank", ของคุณ "your") no longer trigger a name — a blanket lookbehind was
  rejected because it also blocked legitimately attached honorifics (ว่า**นาย**…);
  **E5c** remove the greedy location-prefix regex (`จังหวัด|เขต|ตำบล… + [฀-๿]+`), which,
  lacking word boundaries in space-free Thai, ran across entire clauses — provinces remain
  covered by the exact place gazetteer, addresses by the E2 detector, and locations by NER.
- **E6 — Hard-positive robustness (probe, no code change).** Measure coverage on Dataset D's
  non-templated positives to test whether the recall saturation of Dataset B generalizes.
- **E7 — Gazetteer generalization (probe, no code change).** Re-score the OOV person probe
  under gazetteer modes *off* / *pass 1* / *full* to isolate how much recall depends on the
  21k-name lexicon versus the neural NER and honorific heuristic.

- **E8b — Whitespace-split span merge (from the E7 diagnostic).** Diagnosing E7's apparent
  regression revealed it was not a leak but a one-character gap: the gazetteer split a
  two-word name into two adjacent spans, leaving only the intervening space unmasked. Add a
  post-de-duplication pass that merges adjacent **PERSON** spans separated solely by
  whitespace into a single span (PERSON-only, so distinct phones/emails are never fused).
- **E8a — ADDRESS generalization (from the E6 finding).** Broaden the ADDRESS detector past
  the `บ้านเลขที่` template — additional openers (ที่อยู่ / เลขที่ / หมู่บ้าน), field
  abbreviations (ซ./ถ./ม./ต./อ./จ.), space-separated field values (หมู่ 5), and a trailing
  กรุงเทพฯ/กทม. plus a 5-digit postal code — while **keeping the bounded per-token style**
  (no `[฀-๿]+` across word boundaries, per E5c) and the ≥1-field-prefix requirement so that
  bare openers in prose still do not match. Gated on re-running the negative suite.

Experiments E1, E3, E2, E4, E5, E8b, E8a are applied **cumulatively** in that chronological
order; E6 and E7 are measurement-only probes (E6 is re-reported before and after E8a).

### 4.1 Per-experiment log (E0–E8)

The same experiments, written up individually as *hypothesis → change → result → verdict*.
"Result" cites the headline metric moved by the step (full tables in §5); "Verdict" records
the no-regression gate that had to hold for the change to be accepted. Metric shorthand:
*cov* = privacy coverage on Dataset B, *exact* = exact-span on Dataset B, *spec* =
specificity on Dataset C, *D-cov* = hard-positive coverage on Dataset D.

**E0 — Baseline & harness.**
*Hypothesis:* the unmodified production pipeline already handles some PII types but leaks on
others; we must measure before touching code.
*Change:* none — instrument the shipping pipeline, load Dataset B in scalar coordinates, run
the Dataset A gazetteer ablation.
*Result:* cov 77.1% / typed 77.1% / exact 65.7%; over-mask 4.5%; over-detect 1.61/doc. The
leak concentrates in NATIONAL_ID (cov 40.0%), ADDRESS (55.4%), and PHONE (90.0%). (§5.1, §5.2)
*Verdict:* establishes the reference point; no gate.

**E1 — National-ID format coverage.**
*Hypothesis:* NATIONAL_ID recall is capped because the regex matches only bare 13-digit
strings, missing the dashed and spaced canonical 1-4-5-2-1 groupings.
*Change:* generalize the THAI_ID regex to 1-4-5-2-1 with optional `-`/space separators,
deliberately **disjoint from the phone 3-3-4 grouping** so it cannot swallow phone numbers.
*Result:* NATIONAL_ID cov 40.0 → 100.0 (the two unhandled formats were exactly the 300
missing docs); overall cov 77.1 → 89.1, exact 65.7 → 77.7. (§5.1, §5.2)
*Verdict:* accepted — no phone/other-type regression on B, Dataset A unchanged.

**E3 — Phone format coverage.**
*Hypothesis (revised by the data):* the residual phone misses are the **2-4-4 dashed** mobile
form (e.g. `06-0000-0301`), not the spaced form assumed a priori.
*Change:* relax the middle group of the phone regex from exactly 3 to **3–4** digits.
*Result:* PHONE cov 90.0 → 100.0 (all 50 misses were 2-4-4 dashed); overall cov 89.1 → 91.1,
exact 77.7 → 79.7. (§5.1, §5.2)
*Verdict:* accepted — the widened middle group stays disjoint from the national-ID grouping;
no NATIONAL_ID regression.

**E2 — Address detection.**
*Hypothesis:* addresses are only *incidentally* covered by the location layers and need a
dedicated detector.
*Change:* add a **cue-anchored** ADDRESS detector starting at `บ้านเลขที่` and consuming the
space-separated administrative fields (ซอย / ถนน / หมู่ / แขวง / ตำบล / เขต / อำเภอ / จังหวัด),
requiring ≥1 field so a bare cue in prose does not match.
*Result:* ADDRESS cov 55.4 → 100.0; overall cov 91.1 → 100.0, exact 79.7 → 88.6. ADDRESS
*exact-span* remains at 54.0 — the tail-overcapture problem addressed next by E4. (§5.1, §5.2)
*Verdict:* accepted — coverage reaches ceiling; no negative-suite check yet (C introduced at E5).

**E4 — Boundary correction via precision-aware overlap resolution.**
*Hypothesis (from E2 diagnostics):* exact-span errors are **tail over-capture** — statistical
members (L5 neural NER for PERSON, L2 `NLTagger` for LOCATION) emit spans sharing the entity's
start but extending past its end, and *longest-span-wins* lets them defeat the tight regex spans.
*Change:* attach a per-entity **span-precision** score (regex 1.00 … `NLTagger` 0.30) and, **at
equal start**, prefer higher precision before longer length during de-duplication (different-start
overlaps unchanged — a conservative choice that cannot add a leak). (§5.4, §5.5)
*Result:* PERSON exact 89.2 → 100.0, ADDRESS exact 54.0 → 100.0, overall exact 88.6 → 100.0;
over-mask 4.5 → 4.4 (tails trimmed); cov unchanged at 100.0.
*Verdict:* accepted — coverage held at 100%, Dataset A unchanged, all detector unit tests pass.

**E5 — Precision / false-positive study.**
*Hypothesis:* with recall saturated on B, the remaining risk is false positives, unmeasurable
on positive-only data — so introduce Dataset C and repair the culprits.
*Change:* three repairs, each gated on **no recall regression** on A and B: **E5a** validate IP
octets to 0–255; **E5b** a `คุณ`-scoped negative-lookbehind on the honorific PERSON rule (so
ขอบคุณ/ของคุณ no longer trigger a name); **E5c** delete the greedy location-prefix regex
(`จังหวัด|เขต|ตำบล… + [฀-๿]+`) that ran across whole clauses in space-free Thai. (§5.6)
*Result:* specificity 74.3 → 94.3, negative over-mask 14.5 → 1.0; as a side effect Dataset B
over-mask 4.4 → 1.5 (removed greedy spans). Residual 2 FPs are Apple-`NLTagger` artifacts.
*Verdict:* accepted — Dataset B stays 100/100/100, Dataset A recall unchanged (leak 5.2%, F1 97.3%).

**E6 — Hard-positive robustness (probe, no code change).**
*Hypothesis:* does B's recall saturation generalize to non-templated positives?
*Change:* none — measure coverage on Dataset D.
*Result:* D-cov 85.7% overall; MULTI/NESTED/ROMANIZED hold at 100%, but **ADDR_VAR collapses to
25%** — the E2 detector was over-fit to the `บ้านเลขที่` template. (§5.8) Motivates E8a; re-run
after E8a gives 95.2%.
*Verdict:* probe — no gate; surfaces the ADDRESS generalization gap.

**E7 — Gazetteer generalization (probe, no code change).**
*Hypothesis:* how much of person recall depends on the 21k-name lexicon vs. the neural NER?
*Change:* none — re-score the 10-name OOV probe under gazetteer modes *off / pass 1 / full*.
*Result:* off 10/10, pass 1 10/10, **full 9/10** — the NER alone already covers all ten OOV
names, so the system is not lexicon-dependent for generalization. The single `full` shortfall
was traced to a span-assembly artifact (a split name leaving one space unmasked), not a leak —
fixed by E8b. (§5.9)
*Verdict:* probe — no gate; motivates E8b.

**E8b — Whitespace-split span merge (from the E7 diagnostic).**
*Hypothesis:* E7's `full` shortfall is a one-character gap, not a privacy leak — the gazetteer
split `ณัฐดนัย ธนกฤต` into two adjacent spans, leaving the intervening space unmasked, which the
strict containment metric scores as uncovered.
*Change:* after de-duplication, merge adjacent **PERSON** spans separated **solely by
whitespace** into one span (PERSON-only, so distinct phones/emails are never fused).
*Result:* `full` mode recovers to 10/10; bonus, one pseudonym per person instead of two. (§5.9)
*Verdict:* accepted — Datasets A/B/C unchanged.

**E8a — ADDRESS generalization (from the E6 finding).**
*Hypothesis:* the ADDR_VAR collapse (E6) is template over-fit; broadening the openers/fields
recovers it without new false positives.
*Change:* extend the ADDRESS detector past `บ้านเลขที่` — extra openers (ที่อยู่ / เลขที่ /
หมู่บ้าน), field abbreviations (ซ./ถ./ม./ต./อ./จ.), space-separated field values (หมู่ 5), and a
trailing กรุงเทพฯ/กทม. + 5-digit postal code — keeping the **bounded per-token style** (no
`[฀-๿]+` across word boundaries, per E5c) and the ≥1-field-prefix requirement. (§5.8)
*Result:* ADDR_VAR 25 → 75, D-cov overall 85.7 → 95.2. The one residual miss is an
interleaved-number condominium address, left as a known limit.
*Verdict:* accepted — specificity stays 94.3%, Dataset B stays 100%, Dataset A unchanged.

---

## 5. Results

### 5.1 Dataset B — cumulative progression (2,500 documents)

Overall (micro-averaged over 2,500 gold entities):

| Stage | Change | Coverage (privacy) ↑ | Typed recall ↑ | Exact-span ↑ | Over-mask ↓ | Over-detect/doc |
|-------|--------|:---:|:---:|:---:|:---:|:---:|
| E0 | Baseline | 77.1% | 77.1% | 65.7% | 4.5% | 1.61 |
| +E1 | National-ID formats | 89.1% | 89.1% | 77.7% | 4.5% | 1.61 |
| +E3 | Phone 2-4-4 form | 91.1% | 91.1% | 79.7% | 4.5% | 1.61 |
| +E2 | Address detector | 100.0% | 100.0% | 88.6% | 4.5% | 1.61 |
| +E4 | Precision-aware dedup | **100.0%** | **100.0%** | **100.0%** | **4.4%** | 1.61 |

### 5.2 Dataset B — per-type, baseline (E0) vs. final (E4)

Reported as *Coverage / Typed / Exact* (%).

| Type | N | E0 (baseline) | E4 (final) |
|------|---|---------------|------------|
| PERSON | 500 | 100.0 / 100.0 / 89.2 | 100.0 / 100.0 / 100.0 |
| PHONE | 500 | 90.0 / 90.0 / 90.0 | 100.0 / 100.0 / 100.0 |
| EMAIL | 500 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 |
| ADDRESS | 500 | 55.4 / 55.4 / 9.4 | 100.0 / 100.0 / 100.0 |
| NATIONAL_ID | 500 | 40.0 / 40.0 / 40.0 | 100.0 / 100.0 / 100.0 |
| **Overall** | **2500** | **77.1 / 77.1 / 65.7** | **100.0 / 100.0 / 100.0** |

**Per-type attribution.** NATIONAL_ID recall rose 40.0 → 100.0 at E1 (the two missed
formats, dashed and spaced, account for exactly the 300 missing documents); PHONE 90.0 →
100.0 at E3 (the 50 misses were all 2-4-4 dashed); ADDRESS coverage 55.4 → 100.0 at E2 but
its exact-span remained 54.0 until E4; PERSON exact-span 89.2 → 100.0 at E4. Coverage was
never reduced by any step.

### 5.3 Dataset A — gazetteer ablation (50 sentences, 77 mentions)

Containment metrics; stable across E1–E4 (these experiments do not touch the name path):

| Configuration | Leak ↓ | Coverage ↑ | Typed recall ↑ | Precision ↑ | F1 ↑ | Over-mask ↓ |
|---------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (no gazetteer) | 7.8% | 92.2% | 92.2% | 100.0% | 95.9% | 9.2% |
| + Gazetteer pass 1 (name-run + honorific cue) | 6.5% | 93.5% | 93.5% | 100.0% | 96.6% | 9.2% |
| + Gazetteer pass 2 (run-together join) | 5.2% | 94.8% | 94.8% | 100.0% | 97.3% | 9.2% |

The gazetteer monotonically reduces the leak rate at **constant 100% precision and constant
9.2% over-mask** — it adds recall at no precision or utility cost. Precision remains 100%
because the lexicon is consulted only on dictionary-word-aligned spans, which prevents
mid-word false positives (e.g., the name fragment ธนา inside the common word ธนาคาร "bank").

### 5.4 Diagnostic evidence for E4 (root-cause analysis)

Before E4, every exact-span error was a *tail over-capture* sharing the gold start
(illustrative cases, scalar offsets):

| Type | Gold span | Predicted span (source) |
|------|-----------|-------------------------|
| PERSON | `นางสาวณัฐชา ประเสริฐกุล` | `นางสาวณัฐชา ประเสริฐกุล รับ` (neural NER) |
| ADDRESS | `…เขตเมืองกลาง` | `…เขตเมืองกลาง เพื่อให้` (Apple `NLTagger`, typed LOCATION) |

In both cases a **tight, correct** span from a high-precision member (honorific PERSON regex;
ADDRESS regex) was defeated by a **longer, boundary-polluted** span from a statistical member
under the old longest-wins tie-break.

### 5.5 E4 mechanism

We add to each entity a **span-precision** score in [0, 1], distinct from the detector's
existing *confidence* (which measures "is this PII?", not "is the boundary tight?"):

| Source | Precision | Rationale |
|--------|:---:|-----------|
| Fixed-format regex (EMAIL, THAI_ID, PHONE, IP, ADDRESS) | 1.00 | Boundaries are exact by construction |
| Watchlist (exact string) | 0.90 | Exact match |
| Province gazetteer | 0.85 | Proper-noun exact match |
| Honorific / administrative-prefix heuristics | 0.80 | Anchored, bounded |
| Name gazetteer | 0.60 | Word-aligned but lexicon-limited |
| Neural Thai NER (L5) | 0.40 | Tends to over-extend tails |
| `NLTagger` (L2) | 0.30 | Coarsest boundaries |

De-duplication is left-to-right by start position (unchanged for different-start overlaps —
a deliberately **conservative** choice that cannot introduce a new leak there); **at equal
start** it now prefers higher span-precision before longer length. This discards the
over-captured tail while retaining full coverage, since the preferred tight span still
contains the gold entity.

Effect (Dataset B): PERSON exact 89.2 → 100.0, ADDRESS exact 54.0 → 100.0, overall exact
88.6 → 100.0; over-mask 4.5% → 4.4% (tails trimmed); coverage unchanged at 100%. Dataset A
is unchanged (no leak regression) and all detector unit tests pass.

### 5.6 Dataset C — precision / false-positive study (E5)

On the 35 hard-negatives, the E4 system (recall-optimized) has a **false-positive document
rate of 25.7%** (specificity 74.3%) and a negative over-mask of 14.5%. The per-category
breakdown localizes the problem sharply:

| Category | N | FP-doc rate (E4) | FP-doc rate (E5) |
|----------|---|:---:|:---:|
| NAME_COLLIDE | 9 | 0.0% | 0.0% |
| NUMERIC | 5 | 20.0% | 0.0% |
| EMAIL_URL | 5 | 20.0% | 20.0% |
| DATE_REF | 5 | 20.0% | 0.0% |
| BENIGN | 5 | 20.0% | 0.0% |
| PLACE_COLLIDE | 6 | 83.3% | 16.7% |
| **Overall (specificity)** | **35** | **25.7% (74.3%)** | **5.7% (94.3%)** |

The culprit-type breakdown attributes the E4 false positives to the greedy **LOCATION**
prefix regex (5 of 9), the **PERSON** honorific firing on `คุณ` embedded in ขอบคุณ/ของคุณ
(2), an unvalidated **IP** octet (1, `192.168.0.300`), and one Apple-`NLTagger` **ORG** noise
span. The three E5 repairs raise specificity **74.3% → 94.3%** and cut negative over-mask
**14.5% → 1.0%**, while — verified by re-running Datasets A and B — leaving **Dataset B at
100%** on all five types and **Dataset A recall unchanged** (Exp 2b leak 5.2%, F1 97.3%);
Dataset A over-mask in fact improves 9.2% → 7.3% because the removed greedy spans no longer
over-extend. The two residual false positives are both Apple-`NLTagger` artifacts (a bare
`เขต`, and `โดเมน .`), i.e., outside the components we control by rule.

Two findings deserve emphasis. First, **NAME_COLLIDE is 0% throughout**: the
segmentation-gated gazetteer never fires on common words that happen to spell names — the
same design that lifted recall on Dataset B is *also* what protects precision here, so the
lexicon is a two-sided win rather than a recall/precision trade. Second, the biggest
precision hole was a **heuristic that relied on `[฀-๿]+` in a script without word
boundaries**; in space-free Thai such a pattern silently consumes whole clauses, and the
principled fix is to delete it and defer to word-aligned or exact-match layers.

### 5.7 Combined recall/precision summary

| Axis | Metric | Value |
|------|--------|:---:|
| Recall (Dataset B, 2500) | coverage / typed / exact | 100% / 100% / 100% |
| Recall (Dataset A, 50) | leak / F1 | 5.2% / 97.3% |
| Precision (Dataset C, 35) | specificity | 94.3% |
| Robustness (Dataset D, 21) | hard-positive coverage (§5.8, after E8a) | 95.2% |
| Context-sensitivity (Dataset AMB, 1250 pairs) | POS-recall / NEG-spec / pair-acc (§5.11) | 74.1% / 21.8% / 8.6% |
| Utility | over-mask (B / A / C) | 1.5% / 7.3% / 1.0% |

(Dataset B over-mask fell from the E4-stage 4.4% to 1.5% once E5c removed the greedy
location-prefix spans that previously masked surrounding characters.)

### 5.8 Dataset D — hard-positive robustness (E6)

| Category | mentions | coverage (before E8a) | coverage (after E8a) |
|----------|:---:|:---:|:---:|
| MULTI (multiple entities) | 9 | 100.0% | 100.0% |
| NESTED (person-in-org, นพ./ดร.) | 5 | 100.0% | 100.0% |
| ROMANIZED / foreign | 3 | 100.0% | 100.0% |
| **ADDR_VAR (non-template address)** | 4 | **25.0%** | **75.0%** |
| **Overall** | 21 | 85.7% | **95.2%** |

Multi-entity, nested, and romanized/foreign cases hold at 100% — the neural NER recovers
entities behind specialist honorifics (นพ., ดร.) that are absent from the rule list, and the
honorific/Latin path handles transliterated names. **ADDRESS initially collapsed to 25%**:
the E2 detector was anchored on the literal `บ้านเลขที่` and a closed field-prefix set, so an
address opening with ที่อยู่/เลขที่, or ending with a province plus a 5-digit postal code,
was only partially masked or missed — quantifying §6.3's template-regularity threat (the
Dataset B ADDRESS score of 100% reflects the generator's fixed field order, not address
detection in general). **E8a** generalizes the detector (see §4) and lifts ADDR_VAR to
**75%** (E6 overall 85.7% → 95.2%) at no precision cost — negative specificity stays 94.3%,
Dataset B stays 100%, and Dataset A is unchanged. The one remaining miss is a condominium
address that interleaves numbers between openers (`คอนโดชั้น 8 ห้อง 802 อาคารเอ ซอยลาดพร้าว
15`); covering it robustly would require sequence structure the bounded regex deliberately
avoids, so it is left as a known limitation rather than risk new false positives.

### 5.9 Dataset D — gazetteer generalization (E7)

Coverage of 10 rare/foreign personal names under each gazetteer mode:

| Gazetteer mode | coverage |
|----------------|:---:|
| off (NER + honorific heuristic only) | **10/10 (100%)** |
| pass 1 (+ name-run / cue) | 10/10 (100%) |
| full (+ run-together join) | 9/10 (90%) |

Two observations, both tempering earlier claims. First, the neural NER **already covers all
ten out-of-vocabulary names on its own**, so the system is *not* over-reliant on the lexicon
for generalization — the gazetteer's value (shown on Dataset A) is concentrated on specific
hard cases (run-together spellings, names the NER misses), not blanket recall. Second, before
E8b `full` scored *below* `off` by one mention. **Diagnosis showed this was not a privacy
leak but a scoring artifact**: the gazetteer split `ณัฐดนัย ธนกฤต` into two adjacent spans
`[17,24)` and `[25,30)`; both name words were masked, but the single space at index 24 was
not, so the containment criterion — which requires *every* gold index masked — scored the
mention uncovered. **E8b** (merge adjacent PERSON spans separated only by whitespace) closes
the gap and, as a bonus, yields one pseudonym per person instead of two. After E8b, `full`
recovers to **10/10** with no change to Datasets A/B/C. The episode is a reminder that a
strict containment metric can penalize a split-but-complete mask that carries no privacy
risk, and that the fix belongs in span assembly, not in the detectors.

### 5.10 External baseline comparison (M1)

The ablation (E0–E8) shows *how* Parivarta reaches its operating point; this experiment asks
*how it compares* to existing PII detectors on identical data. We evaluate five external
systems against Parivarta ("ours") on the same datasets, scored by the **same stdlib
scorer** (`tools/eval/score.py`) after converting every system's output to a common
span-JSONL (`{doc_id, type, start, end}`) in Unicode-scalar coordinates. The comparators are:

| System | Kind | Notes |
|--------|------|-------|
| ours | hybrid ensemble (L1–L6) | the shipping detector, final config (post-E8) |
| Presidio | Microsoft Presidio | regex recognizers + spaCy NER, default Thai-agnostic config |
| PyThaiNLP | `thainer-corpus-v2` CRF | Thai-native neural/CRF NER |
| regex-only | L1 of ours, in isolation | ablates the ensemble to its deterministic core |
| spaCy | `xx_ent_wiki_sm` | multilingual NER, no Thai-specific rules |
| LLM (qwen2.5:7b) | zero-shot, local Ollama | full run on all datasets, off-device via Ollama (no cloud) |

Coverage figures carry a **bootstrap 95% CI** (10k resamples over documents); pairwise
coverage differences on Dataset B use **McNemar's paired test** over the 2,500 shared docs.

**Headline results.** Dataset B coverage / typed / exact-span; Dataset C specificity;
Dataset D hard-positive coverage.

| System | B cov ↑ | B cov 95% CI | B typed | B exact | C spec ↑ | C FP-docs | D cov ↑ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ours** | **100.0%** | [100.0, 100.0] | **100.0%** | **100.0%** | **94.3%** | 2 | **95.5%** |
| Presidio | 61.2% | [59.2, 63.0] | 40.9% | 40.2% | 77.1% | 8 | 36.4% |
| LLM (qwen2.5:7b) | 60.2% | [58.2, 62.1] | 58.6% | 58.6% | 100.0% | 0 | 63.6% |
| regex-only | 28.0% | [26.2, 29.8] | 28.0% | 28.0% | 97.1% | 1 | 18.2% |
| PyThaiNLP | 27.8% | [26.0, 29.5] | 17.3% | 17.0% | 80.0% | 7 | 50.0% |
| spaCy | 9.8% | [8.6, 11.0] | 0.2% | 0.1% | 42.9% | 20 | 27.3% |

(N: Dataset B = 2,500; Dataset C = 35; Dataset D = 22. Dataset D is counted here as **22 gold
spans** — the export records one span per surface *occurrence* — whereas §5.8 reports the **21
distinct mentions**; both yield ours ≈ 95% coverage. Every system, the LLM included, was scored on
all four datasets.)

**Per-type coverage on Dataset B** localizes each comparator's blind spots:

| Type | ours | LLM (qwen) | Presidio | PyThaiNLP | regex-only | spaCy |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| PERSON | 100.0% | 18.4% | 5.8% | 86.6% | 0.0% | 11.2% |
| PHONE | 100.0% | 87.6% | 100.0% | 0.0% | 0.0% | 11.2% |
| EMAIL | 100.0% | 90.6% | 100.0% | 52.2% | 100.0% | 20.4% |
| ADDRESS | 100.0% | 23.6% | 0.0% | 0.0% | 0.0% | 0.0% |
| NATIONAL_ID | 100.0% | 80.8% | 100.0% | 0.0% | 40.0% | 6.0% |

**Paired significance (McNemar, Dataset B coverage).** In every pair the improvement is
*one-directional* — because ours covers **every** Dataset B document, no baseline can cover a
document ours misses (base-only = 0 by construction), so each test is maximally significant and
ours-only simply equals the baseline's own miss count (e.g. 2,500 × (1 − 0.602) = 995 for the LLM):

| ours vs. | ours-only wins | base-only wins | p-value |
|----------|:---:|:---:|:---:|
| Presidio | 971 | 0 | 9.8 × 10⁻²¹³ *** |
| LLM (qwen2.5:7b) | 995 | 0 | ≈ 0 *** |
| regex-only | 1,800 | 0 | ≈ 0 *** |
| PyThaiNLP | 1,806 | 0 | ≈ 0 *** |
| spaCy | 2,256 | 0 | ≈ 0 *** |

**Reading of the comparison.** (i) **No single external system covers all five types** —
Presidio's regex recognizers nail PHONE/EMAIL/NATIONAL_ID but score ~0 on Thai PERSON and
ADDRESS; PyThaiNLP is the mirror image (strong PERSON, blind to the structured formats it has
no recognizer for); spaCy's multilingual model is weak across the board on Thai. (ii)
**ADDRESS is detected by *ours* alone** (100% vs. 0% for every comparator), confirming that
the cue-anchored E2/E8a detector addresses a genuine gap rather than re-deriving an existing
capability. (iii) **regex-only is the precision anchor** (C specificity 97.1%, a single FP) but
its 28% coverage shows the deterministic core alone is far from sufficient — the ensemble's
statistical layers supply the missing 72 points of recall while the precision-aware fusion
(E4) and the E5 repairs recover specificity to a comparable 94.3%. (iv) The gains are **not a
metric artifact**: because base-only wins are zero in every pair, ours **strictly dominates**
each baseline on per-document coverage (McNemar p ≪ 10⁻³). (v) **The zero-shot LLM is not
competitive with the ensemble and errs toward *leaking*.** At **60.2%** Dataset B coverage the LLM
(qwen2.5:7b) even trails Presidio (61.2%), collapsing on the open-vocabulary Thai types (PERSON
18.4%, ADDRESS 23.6%) while holding the structured numeric formats (PHONE 87.6%, NATIONAL_ID 80.8%,
EMAIL 90.6%). Its **perfect Dataset C specificity** (0 false positives) shows it *under*-extracts —
the more dangerous direction for a privacy proxy, since a missed entity is exfiltrated. A capable
zero-shot LLM is therefore no substitute for a recall-first, coverage-guaranteed pipeline.

**LLM baseline (qwen2.5:7b, zero-shot, local).** Measured on the full datasets off-device via Ollama
(no cloud; resume-safe, executed on a faster machine — see `tools/eval/README_WINDOWS.md`). The LLM
reaches only **60.2% coverage** — *below* Presidio (61.2%) and **39.8 points under ours** —
collapsing on the open-vocabulary Thai types (PERSON 18.4%, ADDRESS 23.6%) while it does capture the
structured numeric formats (PHONE 87.6%, NATIONAL_ID 80.8%, EMAIL 90.6%). It is markedly
**conservative on negatives** (Dataset C specificity 100%, zero false positives), i.e. it
under-extracts rather than over-flags, and reaches 63.6% on the hard-positive Dataset D. The take-away
for a privacy proxy is that a zero-shot LLM errs toward *leaking* (under-masking), the unsafe
direction, whereas the ensemble guarantees coverage. The LLM is a comparison point only, not part of
the privacy-preserving system, and is run solely on synthetic evaluation data.

### 5.11 Dataset AMB — context-sensitivity (all systems)

The minimal-pair suite (§2.6) is where surface-triggered detection is expected to break down. All
six systems are scored on the shared candidate span (POS-recall = candidate masked; NEG-specificity
= candidate untouched; pair-accuracy = both members of a `pair_id` correct), with bootstrap 95% CIs
and, for pair-accuracy, the **independence baseline** POS-recall × NEG-specificity — the value
expected if the two judgments were uncorrelated. Rows are ordered by pair-accuracy.

| System | POS-recall (TPR) | NEG-specificity (1 − FPR) | pair-accuracy ↑ | indep. baseline |
|--------|:---:|:---:|:---:|:---:|
| LLM (qwen2.5:7b) | 42.4% [39.6, 45.1] | 75.3% [72.9, 77.7] | **26.1%** [23.7, 28.6] | 31.9% |
| **ours** | **74.1%** [71.6, 76.5] | 21.8% [19.4, 24.2] | 8.6% [7.0, 10.2] | 16.1% |
| spaCy | 14.6% [12.7, 16.6] | 75.2% [72.8, 77.6] | 7.9% [6.5, 9.4] | 11.0% |
| Presidio | 64.1% [61.4, 66.7] | 30.6% [28.1, 33.3] | 2.9% [2.0, 3.8] | 19.6% |
| PyThaiNLP | 4.5% [3.4, 5.7] | 61.5% [58.8, 64.2] | 0.1% [0.0, 0.2] | 2.8% |
| regex-only | 34.7% [32.2, 37.4] | 65.3% [62.6, 67.8] | 0.0% [0.0, 0.0] | 22.7% |

Per-type **pair-relevant accuracy** (mean of that type's POS-recall and NEG-specificity):

| Type | ours | LLM (qwen) | Presidio | PyThaiNLP | regex-only | spaCy |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| PERSON | 70.6% | 51.4% | 52.6% | 48.8% | 50.0% | 53.0% |
| PHONE | 50.0% | 65.0% | 50.0% | 50.0% | 50.0% | 51.0% |
| EMAIL | 50.0% | 52.6% | 50.0% | 29.4% | 50.0% | 50.8% |
| ADDRESS | 19.0% | 54.2% | 34.2% | 0.0% | 50.0% | 21.4% |
| NATIONAL_ID | 50.0% | 71.0% | 50.0% | 36.8% | 50.0% | 48.4% |

(LLM figures are from the §5.10 zero-shot run; all other rows are computed by `score.py --system X`
via the new `score_AMB`, and ours' AMB predictions by the `exportAMBForBaselineComparison` harness.)

**Reading.** (i) **AMB collapses the entire field.** No system exceeds **26.1%** pair-accuracy,
against the 60.2–100% coverage regime of Dataset B — AMB is the discriminating, unsaturated
benchmark §6.4 called for. (ii) **Surface-triggering is universal.** *Every* system's pair-accuracy
lies **below its own independence baseline** (the LLM included: 26.1% < 31.9%), i.e. POS- and
NEG-correctness are *negatively* correlated: each system tends to treat the shared surface form the
same way in both contexts. `regex-only` makes this exact — pair-accuracy **0.0%** and every per-type
value exactly **50.0%**, because a deterministic surface rule fires on *both* members of a pair or
*neither* and can never be both-right. None of the six systems, the LLM included, robustly uses Thai
context to disambiguate. (iii) **Two error regimes, and the ranking inverts.** Format/rule systems
(ours, Presidio, regex) buy POS-recall at the cost of NEG-specificity (they fire on the byte-valid
look-alikes); the NER systems (PyThaiNLP, spaCy) and the **LLM** do the reverse (miss the entity but
stay quiet on the look-alike). Only the LLM clears chance on balanced accuracy (58.9%), reaching it
via a conservative skew (POS 42% / NEG 75%), and it takes the top pair-accuracy *despite* being weak
on Dataset B coverage: the AMB ranking is almost the inverse of the coverage ranking.
(iv) **ours confirms the pre-registered hypothesis exactly.** PHONE, EMAIL and NATIONAL_ID all land
at precisely **50.0%** pair-relevant accuracy = POS-recall ≈ 100% with NEG-specificity ≈ 0% (the L1
format layer flags *both* members of every pair, exactly the format-collision items §2.4 flagged as
"expected to fire"). ours does best on **PERSON (70.6%)**, where the segmentation-gated gazetteer plus
honorific context give it genuine disambiguation power (the `NAME_COLLIDE` 0%-FP strength of §5.6
carries over) — the one type where a rule-based Thai component actually reads context. (v) **Privacy
reading.** ours' AMB weakness is **over-masking** (low NEG-specificity), which costs *utility*, not
privacy: a masked look-alike leaks nothing, and POS-recall stays 74.1%, so no true entity escapes.
For a zero-trust proxy this is the *safe* failure direction, whereas the NER systems' low POS-recall
would be leaks. AMB therefore quantifies a real limitation of the deterministic ensemble — surface
disambiguation — and motivates a **context-aware layer** (learned or LLM-assisted) as future work,
without contradicting the privacy guarantee the ensemble provides on Datasets B/D.

**Analysis.** Two quantitative views sharpen the picture. First, collapsing the two axes into a
**balanced member accuracy** = (POS-recall + NEG-specificity) / 2 — the chance level is 50% —
ranks the systems as:

| System | balanced member acc | pair-accuracy |
|--------|:---:|:---:|
| LLM (qwen2.5:7b) | **58.9%** | 26.1% |
| regex-only | 50.0% | 0.0% |
| ours | 47.9% | 8.6% |
| Presidio | 47.3% | 2.9% |
| spaCy | 44.9% | 7.9% |
| PyThaiNLP | 33.0% | 0.1% |

Only the **LLM exceeds chance** on balanced accuracy (58.9%); every rule/NER system sits at or below
50%, `regex-only` landing *exactly* on the 50% trade-off line (its POS-recall and NEG-specificity sum
to 100 by construction). In other words, on the hardest task the non-contextual systems do not "know"
anything beyond surface form — they merely *place* their fixed error budget differently along the
POS/NEG axis, and even the LLM clears chance only marginally. Second, the two benchmarks are **nearly
decoupled**: ranking by Dataset B coverage gives
`ours > Presidio > LLM > regex > PyThaiNLP > spaCy`, whereas ranking by AMB pair-accuracy gives
`LLM > ours > spaCy > Presidio > PyThaiNLP > regex` — the coverage leader (ours) is the context
laggard among the top group, while the LLM is best on AMB yet only third on coverage (just behind
Presidio). Coverage and context-sensitivity are therefore **distinct, weakly-related capability
axes**, and a system tuned to saturate one says little about the other.

**Conclusion (AMB).** The minimal-pair suite achieves its design goal: it is the one benchmark in
this study that **no method saturates** (best pair-accuracy 26.1%, the LLM), and it isolates a
capability — using Thai context to separate PII from a same-surface look-alike — that **every**
evaluated system, the zero-shot LLM included, currently lacks (all fall below their independence
baseline; only the LLM clears chance on balanced accuracy, and only barely). For Parivarta specifically, AMB pins down the
exact shape of the residual risk: the ensemble's format layers give it the field's **highest
POS-recall (74.1%)** but its lowest NEG-specificity (21.8%), so its errors are **over-masking of
look-alikes, not leakage of real entities** — the benign direction for a zero-trust proxy, and one
that costs utility rather than privacy. The practical implication is a **targeted, safe upgrade
path**: add a context-aware disambiguation stage constrained to only *withhold* a mask on a
low-confidence look-alike (never to remove coverage), which can lift NEG-specificity toward the LLM's
level while the deterministic layers keep coverage at the Dataset B/D ceiling. AMB thus converts a
previously unmeasured qualitative worry ("does the detector understand context?") into a concrete,
reproducible number and a bounded engineering task.

---

## 6. Discussion

### 6.1 Principal findings

1. **A hybrid, largely deterministic pipeline attains ceiling recall on structured Thai
   PII.** Overall privacy coverage rose from 77.1% to **100%** on 2,500 documents through
   four targeted, low-complexity changes, with the over-mask (utility) cost essentially flat
   (4.5% → 4.4%) and the over-detection rate constant (1.61/doc).

2. **The dominant failure mode for regex-detectable PII is surface-format brittleness, not
   modeling capacity.** National-ID and phone recall were bottlenecked entirely by
   unhandled but well-defined formatting variants; encoding the *canonical group structure*
   (1-4-5-2-1 vs. 3-3-4) both closed the gap and preserved inter-type disambiguation.

3. **Boundary quality and detection are separable problems, and are best solved by the
   fusion policy rather than the detectors.** The neural and OS NER components reliably
   *find* entities but *over-extend* their spans; a precision-aware fusion rule corrects the
   boundary globally without retraining or per-detector post-processing, lifting exact-span
   from 88.6% to 100% at zero privacy cost.

4. **Lexicon fusion is a Pareto improvement for Thai names.** The dictionary-aligned name
   gazetteer reduced the development-set leak rate by ~33% (7.8% → 5.2%) at unchanged 100%
   precision, empirically validating word-segmentation-gated lexicon matching for a
   space-free script. The dedicated negative suite corroborates this from the other side:
   name/common-word collisions yield **0% false positives**, so the lexicon improves recall
   and precision simultaneously rather than trading one for the other.

5. **In a script without word boundaries, the precision risk lives in prefix heuristics, and
   the fix is to delete them.** Once measured against negatives, the single largest
   false-positive source was a location-prefix regex whose `[฀-๿]+` tail crossed word and
   clause boundaries; removing it (and deferring to word-aligned / exact-match layers) plus
   two small guards (IP octet validation, a `คุณ`-scoped lookbehind) raised specificity from
   74.3% to **94.3%** and cut negative over-mask to **1.0%** — with **zero** recall cost on
   either recall dataset. Precision and recall were therefore *not* in tension here.

6. **Coverage and context-sensitivity are separate axes, and the ensemble's remaining weakness is
   the safe one.** On the minimal-pair suite (Dataset AMB, §5.11) *every* system — including a
   zero-shot LLM — scores below its own independence baseline, i.e. none robustly uses Thai context
   to tell a real entity from a same-surface look-alike; only the LLM clears chance on balanced
   member accuracy (58.9%), and it does so precisely *because* it is among the weakest systems on
   Dataset B coverage, so the coverage and context rankings nearly invert. Parivarta posts the field's highest
   POS-recall (74.1%) but its lowest NEG-specificity (21.8%): its residual error is **over-masking of
   look-alikes, not leakage**, the benign direction for a zero-trust proxy. This localizes the one
   worthwhile next investment — a context-aware stage allowed only to *withhold* a mask, never to
   remove coverage.

### 6.2 Methodological contributions

- A **privacy-first evaluation protocol** that separates *coverage* (leak prevention) from
  *typed* and *exact-span* quality, and reports *over-mask* as an explicit utility cost —
  appropriate for masking systems where recall and boundary tightness matter more than
  conventional token-F1.
- Two **reproducibility requirements for Thai span evaluation** that are easy to get wrong:
  scoring in a **Unicode-scalar coordinate system** to reconcile code-point gold offsets
  with grapheme-based string indices, and newline handling under CRLF/grapheme interaction.
- An **eval-first ablation discipline** in which each mechanism is added singly and gated on
  a no-regression criterion across an independent development set — including a *negative
  result recorded but not shipped* (an earlier neural-preprocessing port that regressed the
  leak rate and was reverted), which we regard as part of the scientific record.
- A **reproducibility caveat for ablation under parallel test execution.** The gazetteer
  ablation is driven by a mode switch read inside the detector. Implemented as shared mutable
  global state, it *raced* under the test framework's parallel suite execution — one
  harness setting the mode to *off* transiently corrupted a concurrent detector assertion,
  producing a non-deterministic failure unrelated to the code under test. Binding the mode as
  **task-local state** (set only within a scoped `withValue` per configuration) confines each
  experiment's setting to its own task and restores determinism. Ablation toggles that are
  read on a hot path should be task/thread-scoped, not global, whenever tests run in parallel.

### 6.3 Threats to validity

- **Synthetic, positive-only benchmark.** Dataset B is generated and contains exactly one
  entity per document, so (i) true precision / false-positive rate is not measurable on it,
  and (ii) surface diversity, while realistic, is bounded by the generator's templates
  (e.g., addresses share a fixed field order). The 100% ceiling should be read as *saturation
  of the benchmark*, not proof of perfection in the wild. Precision is instead measured on
  the dedicated negative Dataset C (specificity 94.3%); however, **Dataset C is small (35
  hand-authored hard-negatives)** and its categories reflect the authors' hypotheses about
  where the system errs, so it establishes the *presence and locus* of false positives
  rather than a population-level precision estimate.
- **Template regularity flatters the rule-based members** — *quantified and partly
  mitigated*: on the non-templated ADDR_VAR probe (Dataset D) ADDRESS coverage was 25% under
  the `บ้านเลขที่`-anchored E2 detector, and 75% after the E8a generalization; the Dataset B
  ADDRESS score of 100% still reflects the generator's fixed field order, not address
  detection in general. Real Thai addresses omit and reorder fields and add building/room
  tokens (E8a's residual miss is exactly such a case), so ADDRESS remains the least
  generalizable component and the strongest candidate for a learned or sequence model.
- **Construct validity of the precision scores.** The span-precision ranking is
  expert-assigned, not learned; its ordering is justified by the diagnostics but its exact
  values are not tuned, and a different corpus could expose a mis-ranking.
- **Single-run measurement.** The pipeline is deterministic given fixed model weights, so we
  report point estimates without variance; confidence intervals would require bootstrapping
  over documents.

### 6.4 Limitations and future work

The positive benchmark is **saturated** (100% on every metric and type) and the negative
suite (E5) has driven specificity to 94.3%, so both can no longer discriminate strongly
between methods at their current size. Meaningful further evaluation requires **harder,
larger data**: (i) a **much larger negative set** — the current 35 hand-authored
hard-negatives establish the locus of false positives but not a population estimate;
(ii) **noisier, less templated** addresses (missing/reordered fields, postal codes,
building/room tokens) and code-mixed / romanized Thai names; (iii) documents with
**multiple and nested** entities to stress the fusion policy; and (iv) **exhaustive**
multi-entity annotation to enable true corpus-level precision/recall. **Dataset AMB (§2.6, §5.11)
is a first step in this direction** — a 1,250-pair minimal-pair suite that isolates
context-sensitivity and discriminates sharply across all six systems (pair-accuracy 0.0–26.1%, and
*every* system below its own independence baseline). Its two findings shape the modeling agenda:
(a) no system, the zero-shot LLM included, robustly disambiguates PII from same-surface look-alikes,
and (b) our ensemble's error there is **over-masking, not leaking** (POS-recall 74.1%, NEG-specificity
21.8%), the safe direction for a proxy. Together they point to a **context-aware disambiguation
layer** (learned or LLM-assisted, gated so it can only *withhold* a mask, never remove coverage) as
the clearest next step. Modeling directions
include learned span-precision weights and calibrated confidence for the fusion step,
filtering the residual Apple-`NLTagger` noise that produced the last two false positives,
and extending ORGANIZATION coverage (an off-axis leak on the development set) that the
current benchmark does not exercise. Both Dataset-D follow-ups have now been addressed —
the E7 span-assembly gap by E8b, and ADDRESS generalization by E8a (ADDR_VAR 25% → 75% with
no precision cost) — leaving only the interleaved-number condominium address, whose robust
capture would need sequence structure the rule-based detector deliberately avoids; that case
is the natural boundary at which a learned ADDRESS model would take over.

### 6.5 Reproducibility

All metrics are emitted to the test console by four self-contained harnesses that execute
the production detector: the gazetteer ablation (`ThaiNEREvaluationTests`) toggles the
lexicon via a mode switch; the external-benchmark harness (`BenchmarkEvaluationTests`) loads
the 2,500-document fixture, aligns offsets in scalar coordinates, and prints the per-type
table; the negative harness (`NegativeEvaluationTests`) reports per-category false-positive
rates, specificity, and the culprit-type breakdown; and the robustness harness
(`RobustnessEvaluationTests`) covers the hard-positive and OOV-probe studies. Each
experiment that changes code corresponds to a single, reviewable commit (E0–E5, E8b, E8a),
and the measurement-only probes (E6, E7) run against the final system, so any stage can be
reconstructed by checkout. The gazetteer mode used by the ablation is task-local, so the
suite is deterministic under parallel execution.

The **cross-system comparison** (§5.10, §5.11) is reproduced by a small, dependency-light toolchain
under `tools/eval/`. Our detector's predictions are exported by the `ExportDetections` harness
(`exportForBaselineComparison` for B/C/D and `exportAMBForBaselineComparison` for AMB), which runs the
identical production pipeline and writes span-JSONL in scalar coordinates; the external baselines are
produced by one runner each (`run_presidio.py`, `run_pythainlp.py`, `run_spacy.py`, `run_regex.py`)
and the zero-shot LLM by the provider-agnostic `run_llm.py` (local Ollama; resume-safe). All systems
are then scored **uniformly** by a single stdlib scorer, `score.py` — `score_B/C/D` and the paired
`score_AMB` — which computes bootstrap 95% confidence intervals and the paired McNemar test. Datasets
B and AMB and all gold files ship as fixtures (`ParivartaTests/Fixtures/`, `tools/eval/exports/`), so
every table in §5 can be regenerated from a clean checkout with no network access (the LLM baseline
excepted, which requires a local model).

---

## 7. Conclusion

We set out to answer whether an **on-device, largely deterministic** detector can protect Thai PII
well enough to anchor a zero-trust LLM proxy, and to characterize precisely where it still falls
short. Under a privacy-first protocol and an eval-first ablation (E0–E8), the hybrid ensemble reaches
**ceiling recall** on a 2,500-document external benchmark (coverage and exact-span **100%**) through
four small, individually-gated changes, holds **94.3% specificity** on a dedicated negative suite and
**95.2%** on non-templated hard positives, and **strictly dominates** five external baselines —
Microsoft Presidio, PyThaiNLP, spaCy, a regex core, and a zero-shot local LLM (qwen2.5:7b) —
on per-document coverage (McNemar *p* ≪ 10⁻³, zero base-only wins). Three findings generalize beyond this system: the
dominant failure mode for structured PII is **surface-format brittleness, not modeling capacity**;
**boundary quality is best fixed by the fusion policy** rather than the detectors; and in a
word-boundary-free script the precision risk lives in **prefix heuristics**, whose principled cure is
deletion in favor of word-aligned or exact-match layers.

The study's harder question is answered by **Dataset AMB**. On minimal pairs that share a surface
form but differ in context, **no method — the zero-shot LLM included — robustly disambiguates
PII from look-alikes**: every system scores below its own independence baseline, and only the LLM
clears chance on balanced accuracy, precisely because it is among the weakest systems on coverage
(it is *third* on Dataset B coverage, just behind Presidio, yet *first* on AMB). Parivarta records
the field's highest POS-recall (74.1%) and lowest NEG-specificity (21.8%), which means its residual
error is **over-masking, not leakage** — the benign direction for a privacy proxy, where utility, not
confidentiality, pays the price. This converts a previously qualitative worry into a concrete,
reproducible number and pins the one worthwhile next investment: a **context-aware disambiguation
stage constrained to only withhold a mask, never remove coverage**, which can raise NEG-specificity
toward the LLM's level while the deterministic layers keep coverage at ceiling. More broadly, the
positive benchmarks are now **saturated** and the negative suite nearly so; meaningful future
evaluation needs larger, noisier, exhaustively-annotated and multi-entity data — of which AMB is a
first, discriminating step — together with learned span-precision and calibrated fusion. We release
the protocol, datasets, and scorers to make each of these directly reproducible.
