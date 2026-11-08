# Methodology

*System design and methods for **Parivarta**, an on-device zero-trust pseudonymization proxy
for Thai/English prompts to cloud LLMs. This section describes the artifact as implemented;
the evaluation protocol and results are reported separately (see `EXPERIMENTS.md`).*

## 3.1 Problem formulation and threat model

We consider a user who wishes to use a remote large language model (LLM) service while
preventing personally identifiable information (PII) in the prompt from being disclosed to,
retained by, or used for training by the provider. Let a prompt be a string `x`. A detector
`D` returns a set of typed spans `E = {(t_i, [a_i, b_i))}` over `x`. A **pseudonymizer** `π`
replaces each detected span with a stable, type-indexed surrogate token (a *pseudonym*),
producing a masked prompt `x̃ = π(x, E)` that is sent to the LLM, together with an invertible
mapping `M : pseudonym → original`. The reply `ỹ` returned by the LLM is restored locally by
the inverse map, `y = π⁻¹(ỹ, M)`, before being shown to the user.

The **trust boundary** is the device: detection, pseudonymization, the mapping store, and
de-pseudonymization all execute locally, and only `x̃` (which by construction contains
surrogates in place of PII) crosses the network. The adversary model is (i) an honest-but-
curious LLM provider that observes every `x̃`, and (ii) prompt-injection content embedded in
`x` that attempts to subvert the downstream model. The design objective is **privacy-first**:
the primary cost metric is *leakage* (PII that reaches the provider unmasked); over-masking
(masking non-PII) is a secondary utility cost, not a safety failure.

Because the target language is Thai — written without inter-word spaces and with combining
vowel/tone marks — the methods below are designed around the absence of orthographic word
boundaries, which is central to several design choices.

## 3.2 Pipeline overview

A prompt is processed by five sequential stages:

1. **Adversarial pre-filter** (§3.3): reject prompts exhibiting prompt-injection patterns.
2. **Normalization** (§3.4): canonicalize non-content tokens and whitespace.
3. **PII detection** (§3.5): a six-member ensemble produces candidate typed spans.
4. **Span fusion** (§3.6): resolve overlaps into a consistent, minimal span set.
5. **Reversible pseudonymization** (§3.7): substitute surrogates, call the LLM, and restore
   the reply (§3.8).

Stages 1–4 are deterministic given fixed model weights. The detection ensemble is the core
contribution and is deliberately *hybrid*: high-precision deterministic matchers for
formatted identifiers, lexicon- and rule-based components for Thai names and places, and a
neural sequence model for open-vocabulary entities, combined by a precision-aware fusion rule.

## 3.3 Adversarial pre-filter

Before any content is processed, the prompt is screened against a set of case-insensitive
regular expressions characterizing common prompt-injection and jailbreak attempts
(e.g., *ignore (all) previous/prior instructions*, *disregard previous*, *bypass security*,
*reveal/tell me your system prompt*, *jailbreak*, *developer mode*, and the
“describing … + similarly” / “write opposite contents” adversarial-suffix templates). If any
pattern matches, the prompt is blocked and never forwarded; the matched patterns are surfaced
to the user for transparency. This is a conservative gate: it precedes detection so that a
malicious prompt cannot reach the network even in masked form.

## 3.4 Normalization

The prompt is normalized to reduce spurious detections and to stabilize span offsets:
URLs (`https?://…`), `@`-mentions, and `#`-hashtags are replaced by the placeholders
`[URL]`, `[MENTION]`, `[HASHTAG]`, and runs of whitespace are collapsed to a single space
with surrounding whitespace trimmed. Detection operates on this normalized string; span
offsets are interpreted consistently in the Unicode **scalar** coordinate system throughout,
which is required for correctness in Thai because a grapheme cluster (Swift's `Character`)
may comprise several scalars.

The mention rule is **left-anchored** — `(?<![A-Za-z0-9_.+%-])@\w+` rather than `@\w+` — so
that it cannot consume the domain of an e-mail address. An unanchored rule rewrites
`a@b.com` to `a[MENTION]` *before* any detector sees it, so the address is never
pseudonymized and the normalizer silently destroys the token instead of masking it. The
guard is restricted to ASCII local-part characters rather than `\w` so that mentions written
flush against Thai text (`ทัก@handle`, since Thai does not delimit words with spaces) are
still normalized. This interaction is invisible to any benchmark that scores detectors on raw
document text rather than on the normalized string the deployed pipeline actually consumes.

## 3.5 PII detection ensemble

Each member emits typed candidate spans with two scalar scores: a **confidence** (belief that
the span is PII) and a **span-precision** weight (belief that the span *boundaries* are
tight); the latter is used by fusion (§3.6) and is the key that lets high-recall, loose
members coexist with high-precision, tight members. Table 1 summarizes the members.

**Table 1. Detection ensemble.**

| # | Member | Target types | Method | Span-precision |
|---|--------|--------------|--------|:---:|
| L6 | Watchlist | ORGANIZATION, CUSTOM_KEYWORD | Exact string match (organizational + user-supplied terms) | 0.90 |
| L1 | Pattern matchers | EMAIL, THAI_ID, PHONE_NUMBER, IP_ADDRESS, ADDRESS | Deterministic regular expressions | 1.00 |
| L2 | OS NER | PERSON, LOCATION, ORGANIZATION | Apple `NaturalLanguage` name-type tagger | 0.30 |
| L3 | Thai heuristics | PERSON, LOCATION | Honorific-anchored name rule; province lexicon | 0.80 / 0.85 |
| L4 | Name gazetteer | PERSON | Dictionary word segmentation + name lexicon | 0.60 |
| L5 | Neural Thai NER | PERSON, LOCATION, ORGANIZATION | Transformer token classifier, on-device | 0.40 |

### 3.5.1 Deterministic pattern matchers (L1)

Formatted identifiers are matched by regular expressions engineered around their canonical
structure rather than their surface separators:

- **EMAIL**: a standard local-part`@`domain`.`TLD pattern.
- **National ID (THAI_ID)**: the 13-digit Thai citizen identifier is matched by its fixed
  1–4–5–2–1 digit grouping with optional `-` or space separators
  (`\d[-\s]?\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d`), so the dashed, spaced, and unseparated
  renderings are all recognized. The grouping is *distinct* from the phone grouping, which
  prevents the two matchers from competing for the same digits.
- **PHONE_NUMBER**: a Thai/international mobile/landline pattern anchored on a `0`/`+66`
  prefix with 1–2 / 3–4 / 3–4 digit groups and optional separators, covering plain, 3-3-4,
  and 2-4-4 dashed renderings.
- **IP_ADDRESS**: four dot-separated octets, each **range-validated to 0–255**, so
  IP-shaped but invalid strings (e.g., `192.168.0.300`) are not matched.
- **ADDRESS**: a Thai postal address is matched as an *opener* token
  (`บ้านเลขที่ | ที่อยู่ | เลขที่ | หมู่บ้าน`) followed by a house token and **at least one**
  administrative field that begins with a recognized prefix (ซอย/ถนน/หมู่/ตำบล/แขวง/อำเภอ/เขต/
  จังหวัด and their abbreviations ซ./ถ./ม./ต./อ./จ.), optionally extended by a city name
  (กรุงเทพฯ/กทม.) and a 5-digit postal code. Crucially, every field is bounded to a single
  non-space token (`\S+`) rather than a greedy Thai-character run: because Thai lacks word
  boundaries, a greedy `[฀-๿]+` after a prefix consumes entire clauses and is a dominant
  source of false positives (§ discussion in `EXPERIMENTS.md`). Requiring ≥1 field also
  ensures a bare opener occurring in prose is not matched.

Pattern matchers are assigned the maximal span-precision (1.00): their boundaries are exact
by construction.

### 3.5.2 Statistical OS NER (L2)

Apple's `NaturalLanguage` name-type tagger provides an on-device, general-purpose NER for
PERSON/LOCATION/ORGANIZATION. It contributes recall on open-vocabulary and code-mixed
entities but produces the loosest boundaries (it frequently over-extends spans over adjacent
tokens); it therefore receives the lowest span-precision (0.30).

### 3.5.3 Rule-based Thai heuristics (L3)

Two rules compensate for weaknesses of general NER on Thai:

- **Honorific-anchored PERSON**: a title (นาย/นาง/นางสาว/คุณ/เด็กชาย/เด็กหญิง and the
  abbreviations น.ส./ด.ช./ด.ญ.) followed by one or two name tokens, of a *single script*
  (Thai *or* Latin) to avoid absorbing a following Thai verb. The token `คุณ` — which also
  occurs inside common words such as *ขอบคุณ* (“thank”) and *ของคุณ* (“your”) — carries a
  negative look-behind so it fires only at a genuine word boundary; other titles are matched
  without this restriction because in Thai they legitimately attach to the preceding word.
- **Province lexicon**: a curated list of Thai province/city proper nouns matched exactly.
  (An earlier prefix-based location rule that matched *administrative-prefix + Thai run* was
  removed after it was found to dominate false positives; standalone locations are covered by
  this exact lexicon, by ADDRESS, and by the neural model.)

### 3.5.4 Dictionary-segmented name gazetteer (L4)

To recover Thai personal names robustly without spaces, L4 first **segments** the text by a
re-implementation of the *newmm* greedy maximal-matching algorithm over a 62k-entry Thai
dictionary, then matches word-aligned segments against a **name lexicon of 21,218 entries**
(PyThaiNLP female, male, and family name lists). Segmentation-before-lookup is essential: it
prevents a name substring from matching inside an unrelated word (e.g., the name *ธนา* inside
*ธนาคาร*, “bank”). A span is accepted as PERSON when either (i) two or more adjacent
name-segments form a run (given + family name), or (ii) a single name-segment is preceded by
a title/`ชื่อ` cue; a second pass additionally recovers run-together spellings by joining
contiguous, boundary-aligned segments whose concatenation is a lexicon name of length ≥4.
The gazetteer is assigned medium span-precision (0.60): boundaries are word-aligned but
lexicon-limited.

### 3.5.5 On-device neural Thai NER (L5)

Open-vocabulary Thai entities are detected by a transformer token classifier
(`pythainlp/thainer-corpus-v2-base-model`, a WangchanBERTa / XLM-R-family model) converted to
Apple **Core ML** at half precision and executed entirely on-device (Neural Engine / GPU).
Sub-word tokenization uses the model's XLM-R Unigram tokenizer, loaded offline; the model runs
at a fixed input length of 512 with an attention mask. For each sub-word the network emits a
BIO tag distribution; the argmax tag is taken with a softmax-probability floor of 0.35 (set
low because run-together Thai names yield low per-token probabilities). Sub-words are mapped
back to character spans by sequential surface alignment, and contiguous non-`O` sub-words are
merged into a single entity whose type is the majority BIO type. The neural member has high
confidence but a tendency to over-extend span tails, and is assigned low span-precision (0.40).

### 3.6 Span fusion and conflict resolution

The union of candidate spans is reconciled into a non-overlapping set by a two-phase rule.

**Phase 1 — precision-aware de-duplication.** Candidates are ordered by
(i) start offset ascending, then (ii) span-precision descending, then (iii) length
descending, then (iv) confidence descending; a greedy left-to-right scan keeps a candidate
iff it does not overlap an already-kept span. The precision-before-length ordering is the
critical design point: when a tight, high-precision span (e.g., a regex ADDRESS or an
honorific-anchored PERSON) shares its start with a longer, boundary-polluted span from a
statistical member (L2/L5 over-capturing a trailing token), the tight span is kept and the
over-captured tail discarded. Because the tight span still contains the true entity, coverage
is preserved while collateral masking is reduced. For different-start overlaps the earlier
span wins (as in the classical interval-scheduling scan), a conservative choice that cannot
introduce a new leak.

**Phase 2 — whitespace-gap merge.** Adjacent kept **PERSON** spans separated *solely by
whitespace* are merged into one span. Word segmentation can split a two-token personal name
into two spans whose only unmasked character is the intervening space; merging closes that
gap and yields a single pseudonym per person. The merge is restricted to PERSON so that
distinct formatted identifiers (e.g., two phone numbers) are never fused.

### 3.7 Reversible pseudonymization and the mapping vault

Given the fused span set, the masked prompt is produced by a single left-to-right splice:
non-PII text is copied verbatim and each PII span is replaced by a pseudonym obtained from a
persistent **mapping vault**. For a value `v` of type `t`, the vault key is
`SHA-256(normalize(v))` with `normalize` = trim + case-fold, so that surface-equivalent
mentions share an entry. On first occurrence the vault allocates the surrogate `t_k` where `k`
is a per-type monotonic counter (e.g., `PERSON_001`, `PHONE_NUMBER_002`) and stores the record
`(pseudonym, type, AES-GCM-256(v))`; on subsequent occurrences it returns the existing
surrogate. This yields **referential consistency**: the same entity receives the same
pseudonym within and across turns, preserving the coreference structure the LLM needs while
revealing nothing about the original. Ciphertext is sealed with a 256-bit key held in the
platform **Keychain**; the record set is persisted as an encrypted-at-rest JSON store. The
vault exposes read-back (for a user-facing audit view) and full erasure (“forget”), so retained
data remains under user control.

### 3.8 De-pseudonymization and streaming

The LLM is invoked with the masked prompt. Its reply is restored by replacing each pseudonym
with its original value, applying substitutions in **descending pseudonym-length order** so
that no surrogate is a prefix of another (e.g., `PERSON_1` vs `PERSON_10`). Under streaming
generation the running raw output is accumulated and the *entire* buffer is de-pseudonymized
on every delta, so a surrogate split across chunk boundaries still resolves correctly once
complete. Only the masked prompt ever leaves the device; the mapping and the restored reply
remain local.

### 3.9 Implementation and reproducibility notes

The system is a single native application; all components (including the Core ML model and
the tokenizer/dictionary/lexicon resources) are bundled and run without network access during
detection. Two Thai-specific correctness requirements are enforced throughout: (i) all span
arithmetic uses Unicode scalar offsets to reconcile code-point annotations with grapheme-based
string indexing, and (ii) the ablation switch controlling the gazetteer is bound as
task-local state so that parallel evaluation runs do not interfere. Each detection component
and every subsequent refinement is covered by an executable harness, enabling exact
reproduction of the reported metrics; the evaluation datasets, metrics, and ablations are
detailed in `EXPERIMENTS.md`.
