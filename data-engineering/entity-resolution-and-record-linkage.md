# Entity Resolution and Record Linkage

Entity resolution identifies records that refer to the same real-world entity even when their fields differ. It is used to deduplicate customer records, connect product catalogs, reconcile organizations, and build consistent datasets from multiple sources.

It is harder than exact duplicate removal because names, addresses, identifiers, and formatting may be incomplete or inconsistent.

## Two related tasks

- **Deduplication:** find multiple records for the same entity inside one dataset.
- **Record linkage:** determine which records across two or more datasets refer to the same entity.

A match is a decision under uncertainty. False matches can be more damaging than missed matches, especially when records affect people, payments, or access.

## A practical pipeline

### 1. Normalize carefully

Apply safe, field-specific normalization:

- Trim whitespace and standardize case.
- Normalize phone numbers with country context.
- Standardize dates and units.
- Preserve meaningful punctuation or tokens when they distinguish entities.
- Keep the original value for auditing.

Do not aggressively remove information. For example, apartment numbers, legal suffixes, or diacritics may matter in a particular domain.

### 2. Generate candidate pairs with blocking

Comparing every pair requires roughly N squared comparisons. Blocking reduces the search space by grouping records that share a likely key, such as postal code, normalized email, or a phonetic surname key.

Blocking improves efficiency but can miss true matches when the chosen key is missing or inconsistent. Use multiple complementary blocking rules and measure candidate-pair recall.

### 3. Compare fields

Useful comparison features include:

- Exact identifier agreement.
- Name similarity.
- Address similarity.
- Date-of-birth agreement where lawful and appropriate.
- Phone or email agreement.
- Contradictory evidence, such as incompatible country or entity type.

Similarity should be field-aware. A high string score on a common name is weaker evidence than an exact match on a verified unique identifier.

### 4. Score and decide

Common approaches:

- Rule-based thresholds for simple, well-understood domains.
- Probabilistic linkage, such as the Fellegi-Sunter framework.
- Supervised pair classification using labeled match and non-match pairs.
- Human review for ambiguous or high-impact cases.

Use at least three outcomes: match, non-match, and uncertain/manual review. A single threshold can hide an important precision-recall trade-off.

## A small fuzzy-matching example

This example scores candidate name pairs. It is a demonstration, not a complete entity-resolution system.

```python
from rapidfuzz import fuzz
import re

def normalize_name(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())

def name_similarity(left: str, right: str) -> float:
    a = normalize_name(left)
    b = normalize_name(right)
    return fuzz.token_sort_ratio(a, b) / 100.0

score = name_similarity("A. Sharma Pvt. Ltd.", "Sharma Private Limited")
print(f"Similarity: {score:.2f}")
```

The example can produce misleading scores because legal suffixes and abbreviations need domain-specific treatment. In production, combine multiple fields, apply blocking, and calibrate decisions on labeled pairs.

## Probabilistic linkage intuition

The Fellegi-Sunter approach compares the evidence for a match against the evidence for a non-match.

For a comparison outcome gamma on a field:

- m estimates how often that outcome appears among true matches.
- u estimates how often it appears among non-matches.
- The log likelihood ratio log(m/u) measures how strongly the outcome supports a match.

Evidence from several fields can be combined under modelling assumptions. Correlated fields and poor estimates can make scores overconfident, so validate the model on representative labeled pairs.

## Evaluate the complete pipeline

- **Candidate recall:** fraction of true matching pairs that blocking generates.
- **Pair precision and recall:** quality of match decisions.
- **F1 or precision-recall curve:** summarize a chosen operating point.
- **Cluster quality:** assess whether final entity groups contain false merges or missed links.
- **Review burden:** number and cost of ambiguous pairs sent to humans.
- **Slice analysis:** performance by source system, language, region, or record completeness.

Pair-level metrics alone can hide cluster errors. One incorrect edge can merge two otherwise separate entity groups.

## Common failure modes

- Using a single blocking key and missing records with inconsistent values.
- Treating fuzzy string similarity as proof of identity.
- Merging records transitively without checking cluster-level consistency.
- Ignoring common names and field-specific rarity.
- Leaking identifiers or sensitive attributes into logs and review tools.
- Evaluating on easy random pairs rather than realistic difficult non-matches.
- Optimizing recall without measuring the cost of false merges.

## Practice

Create a small labeled pair dataset with true matches, near matches, and hard non-matches. Compare exact matching with a fuzzy multi-field scorer. Measure candidate recall, pair precision/recall, and the number of cases requiring manual review.

## References

- [Python Record Linkage Toolkit](https://recordlinkage.readthedocs.io/)
- [RapidFuzz documentation](https://rapidfuzz.github.io/RapidFuzz/)
- [Fellegi and Sunter: A Theory for Record Linkage](https://doi.org/10.1080/01621459.1969.10501049)

**Key takeaway:** Entity resolution is a candidate-generation, evidence-scoring, and decision problem. Measure missed candidates and false merges separately, and retain an auditable path back to source records.
