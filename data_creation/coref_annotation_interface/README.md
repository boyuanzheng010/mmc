# Coref annotation

This repository contains Amazon Mechanical Turk/Turkle templates and tools for annotating incremental coreference links, that is, links from an entity mention back to previous mentions of the same entity in the document.

## Annotation

For annotation, we generally use a JSON lines file format (one JSON object per line).  Mechanical Turk uses CSV input and output, so a Python script `convert-jsonl-csv.py` is provided for converting between the two formats.  Run `convert-jsonl-csv.py -h` to see the usage.

### Input

For the constrained multi-query annotation protocol (`multi-query-constrained-hit.html`), each input line has the following format (here a single line has been expanded onto multiple lines and indented to illustrate the structure):

```json
{
  "sentences": [
    ["john", "crashed", "his", "car", "."],
    ["the", "crash", "occurred", "in", "catonsville", "."]
  ],
  "querySpans": [
    { "sentenceIndex": 1, "startToken": 0, "endToken": 2 },
    { "sentenceIndex": 1, "startToken": 4, "endToken": 5 }
  ],
  "candidateSpans": [
    { "sentenceIndex": 0, "startToken": 0, "endToken": 1 },
    { "sentenceIndex": 0, "startToken": 1, "endToken": 2 },
    { "sentenceIndex": 0, "startToken": 1, "endToken": 4 },
    { "sentenceIndex": 0, "startToken": 2, "endToken": 4 },
    { "sentenceIndex": 0, "startToken": 3, "endToken": 4 },
    { "sentenceIndex": 1, "startToken": 0, "endToken": 2 },
    { "sentenceIndex": 1, "startToken": 1, "endToken": 2 },
    { "sentenceIndex": 1, "startToken": 4, "endToken": 5 }
  ]
}
```

The contents of `sentences` consist of a list of sentences, each of which is represented by a list of words (strings).

In general, text spans are represented as a three-element dictionaries with the following entries:

* `sentenceIndex`: the zero-based index of the sentence in `sentences`
* `startToken`, `endToken`: a zero-based exclusive range corresponding to the words of the span within the specified sentence 

Each entry of `querySpans` represents a span in the sentences to be annotated.

The content of `candidateSpans` represents the spans in the sentences that can be submitted as answers.

### Output

Each line of the output JSON lines data, extracted and converted from the CSV column `Answer.answer_spans` in the Mechanical Turk output, consists of a list of answer spans:

```json
[
  {
    "querySpan": { "sentenceIndex": 1, "startToken": 0, "endToken": 2 },
    "sentenceIndex": 0,
    "startToken": 1,
    "endToken": 4,
    "notPresent": false
  },
  {
    "querySpan": { "sentenceIndex": 1, "startToken": 4, "endToken": 5 },
    "sentenceIndex": -1,
    "startToken": -1,
    "endToken": -1,
    "notPresent": true
  }
]
```

The value of `querySpan` represents a given answer span's corresponding query span and is copied from the input data; the outer sentence index, start token, and end token represent the answer span.

If there is no answer for a given query span, the answer span indices are all set to `-1` and `notPresent` is set to `true`.

### Toy interface

To make a "toy" version of the interface, a self-contained HTML page for superficial demonstration purposes, use the `make-toy-interface.py` script.  For example:

```bash
python make-toy-interface.py \
    --interface-template-path multi-query-constrained-hit.html \
    --sampler \
    userstudy-20210421-2/day1/session1/unlabeled_data.jsonl \
    > ../public_html/active-learning-hit.html
```

## Evaluation

Download the evaluation code:

```bash
curl -o - http://conll.cemantix.org/download/reference-coreference-scorers.v8.01.tar.gz | tar -xz
```

Print dev data in human-friendly format:

```bash
python print-jsonl.py /export/c01/pxia/data/coref/dev.english.jsonlines
```

For demonstration purposes, create a dummy set of system outputs:

```bash
sed '/^#/!s/^\([^ ]\+ \+\).*$/\1-/' /export/c01/pxia/data/coref/dev.english.v4_gold_conll > predictions.conll
```

The system output file as a CoNLL-like format as in the following example:

```
#begin document (tc/ch/00/ch_0009); part 000
tc/ch/00/ch_0009 (10000)
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 (10002)
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -

tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 (10013
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 (10014)|10013)
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 (10009)
tc/ch/00/ch_0009 -

#end document
#begin document (tc/ch/00/ch_0009); part 001
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 -
tc/ch/00/ch_0009 (10000)
tc/ch/00/ch_0009 -

#end document
```

In the OntoNotes coreference paper and elsewhere, the MUC, B-cubed, and entity-based (phi 4) CEAF metrics are used for evaluation.  The focus is on the F1 computations using each metric, or even the average of all three F1 scores.  Do the following to run the evaluation for each metric:

```bash
for m in muc bcub ceafe
do
    reference-coreference-scorers/v8.01/scorer.pl $m /export/c01/pxia/data/coref/dev.english.v4_gold_conll predictions.conll
done
```
