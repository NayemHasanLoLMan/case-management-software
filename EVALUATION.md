# Evaluation Approach

This document explains how we measured whether the system's outputs are grounded in the source documents and whether the improvement loop produces measurably better drafts.

## Grounding Evaluation

### What grounding means here

A draft output is grounded if every factual claim it makes can be traced back to a retrieved chunk from one of the four source documents. An ungrounded claim is one that the system invented without evidence, a hallucination.

### How we evaluate it

**Source citation tracing.** Every draft output ends with a SOURCES USED section listing each retrieved chunk by source file and relevance score. A reviewer can open the corresponding file, search for the text from the chunk, and verify the claim. This is the primary grounding mechanism; the chain of evidence is explicit and inspectable.

**LLM grounding instruction.** The generation prompt contains a hard rule: "If a fact is not in the evidence, write 'not found in documents' rather than guessing." We check for this phrase in outputs to detect cases where the LLM acknowledged missing information rather than fabricating it.

**Manual spot check methodology.** For each draft generated from the Rodriguez documents:

1. Pick 5 specific factual claims at random from the output (an amount, a date, a name, an instrument number, a deadline).
2. Look up the claim in the SOURCES USED list.
3. Open the corresponding source file.
4. Confirm the claim appears verbatim or is a direct inference from that text.

Applying this to both generated primary drafts (Title Review Summary and Case Status Memo):

| Draft | Claims spot-checked | Grounded | Ungrounded | Notes |
|---|---|---|---|---|
| Title Review Summary | 8 | 8 | 0 | All instrument numbers, dates, and amounts traced to title search files |
| Case Status Memo | 8 | 8 | 0 | Court dates from court_order.txt, payoff from servicer_email.txt |

All checked claims were grounded. The LLM did not fabricate any specific figures or dates.

**Retrieval relevance check.** The retrieval demo output (02_retrieval_demo.txt) shows that a query for "HOA lis pendens amount instrument number" returns title_search_page1.txt as the top result, the file that actually contains that information. This confirms the retriever is directing the generator to the right sources.

## Improvement Loop Evaluation

### What we measure

After building the style guide from two operator edit pairs, we generate a Document Checklist twice, once without the style guide (baseline) and once with it (improved). We then ask the LLM to compare the two versions on three dimensions:

1. Whether specific information was added or reorganized
2. Whether the improved draft is more actionable
3. Whether grounding and citations improved

### Results from the actual run

The improvement evaluation (04_improvement_evaluation.txt) found:

- The improved draft added missing document types (promissory note, mortgage originals) that the baseline omitted, with an explanation of why each is legally required.
- Items were moved from "on file" to "required/missing" based on the distinction between copies and originals, a pattern the operator edits taught the system.
- Specific filing deadlines related to the servicer transfer were added to the deadlines section.
- Each missing document item included a reference to the source that requires it, improving traceability.
- Relevance score references in the improved SOURCES USED section were more specific.

The LLM evaluator described the improved version as "significantly better due to enhanced clarity, actionability, and structural organization" and identified one remaining cosmetic gap (a duplicate sources section).

### Quantitative proxy metrics

Since we don't have a ground truth labeled dataset, we use these proxy signals:

| Metric | Baseline Checklist | Improved Checklist |
|---|---|---|
| Section count | 3 | 3 |
| Items in "Required/Missing" | 3 | 5 |
| Items with explicit legal justification | 0 | 3 |
| Items with deadline tied to source | 1 | 4 |
| Source files cited | 4 | 4 |

More items in the "Required/Missing" section with explicit justification is the clearest signal: the system learned from the operator edits that it needs to distinguish what exists from what is legally required, and to explain why.

### Limitations

The improvement loop is demonstrated on a single case with two edit pairs. With more cases and more varied operator corrections, the style guide would accumulate more patterns, and the improvements would be more pronounced. The current setup is sufficient to demonstrate the mechanism but not to statistically measure improvement across a distribution of cases.

The LLM-as-evaluator approach (compare_drafts) is itself subject to the model's own biases and is not a substitute for human review. The evaluation output should be read as a structured first-pass assessment, not a ground truth measurement.
