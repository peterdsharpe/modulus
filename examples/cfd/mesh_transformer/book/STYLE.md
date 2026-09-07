# Style sheet for the polished chapters

This file governs every `*.qmd` chapter except the lab notebook
(`18-notebook.qmd`), which is the dated, append-only record and follows its
own rules. Read it before writing or editing any chapter.

## What a polished chapter is

A polished chapter presents the **current best state of knowledge** to a
technically expert reader who has never seen this project. It is not a
history of how the knowledge was acquired. The notebook holds the history.

Consequences:

1. **Cut, never retract.** If a claim once made is now known to be wrong, it
   does not appear in the chapter at all — not as "we previously believed",
   not as "retired", not as "reframed", not as "superseded". Delete it and its
   references. A reader must never watch the book argue with itself.
2. **State the current model, not the path to it.** The reader is using
   the best current version; a number about an earlier implementation or
   configuration ("was 9.8 GB", "as trained", "before the rewrite") is not a
   fact about the model and does not appear. If an implementation choice
   needs justifying, give the reason in one place (the cost section), not
   the history.
2b. **No dates, no chronology.** No "2026-09-05", no "as of", no "addendum",
   no "the campaign", no "wave", no "round", no "critic review", no "before /
   after the fix". Experiments are described by what they measure, not when
   they ran. (Exception: the tech-tree status appendix, which is explicitly a
   live status ledger, and the notebook.)
3. **No project slang.** Node codes (A35b, V0, B1, S1, P1'', W2-C, G3K) and
   version tags (v3c, v5a4) do not appear in chapter prose. Say what the
   experiment is: "the single-factor ablation at 35 training cases". If an
   identifier is unavoidable in a table, define it in the same table's
   caption.
4. **Architectures by name, every time.** ISLA, GeoTransolver, Transolver. The retired exact-kernel
   design appears only in the interior chapter and the index footnote, as
   "the exact-kernel MeshTransformer, an earlier design of this program";
   never "MT1". Never "the incumbent", "the baseline", "the old model".

## Every number has a home

5. **Define the quantity before the number.** Before any table or figure,
   one or two sentences say: what is measured (e.g. relative L2 error of
   surface pressure, ‖p̂ − p‖/‖p − p_∞‖ over the sampled points of a case,
   geometric-mean over cases), on which data (the split, its size, and why
   that split), under what protocol (tokens per case, epochs, seeds,
   learning rate), and what a ratio in the table means (which is the
   numerator). A table whose cells say "1.08x" with no such sentence is a
   defect.
6. **Resource-matched comparisons only.** Any cross-architecture comparison
   states the peak training memory, the parameter count and the tokens per
   case of both sides. If they differ, say so in the same sentence and say
   whether the difference could buy accuracy (e.g. the same function
   evaluated with a less efficient implementation does not). Prefer to show
   the matched-resource comparison outright.
7. **Splits and datasets are introduced once, in chapter 1**, with their
   sizes and purposes; later chapters name them (e.g. "the 35-case split")
   and cross-reference `@sec-...` for the definition.
8. **Provenance is a footnote, not a paragraph.** Each figure/table caption
   ends with the artifact path it is computed from (`results/....json`). The
   prose does not narrate file names.

## Figures

9. **No jitter or dodge on categorical axes.** Distinguish seeds or
   architectures by marker, fill, or line style, never by a horizontal
   offset that reads as a value.
10. Lines from zero for bars; log axes labelled; every axis has quantity and
    unit in brackets; direct labels rather than legends where space allows;
    one visual channel per variable; grey for context, colour for the claim;
    the same hue means the same architecture in every figure
    (ISLA blue `#2a78d6`, GeoTransolver red `#e34948`,
    Transolver grey `#52514e`, exact-kernel MeshTransformer purple
    `#7b4fb3` where present).
11. Every figure gets a "how to read this" sentence in the caption or
    adjacent callout that states what it does *not* show.
12. Figures are computed live from `results/*.json` in a `{python}` cell
    with `code-fold: true`, using the inline palette pattern already in the
    book (do not import `figures.py`, which switches matplotlib to Agg).

## Prose

13. Lead with the conclusion. First sentence of a chapter and of every
    section states the finding and why it matters.
14. Words → picture → symbols. Physical description first, figure second,
    equation last; sandwich equations between a symbol table and a sentence
    on meaning.
15. Define every term and acronym at first use in each chapter. Assume
    expertise (PDEs, transformers, equivariance), not familiarity.
16. Short paragraphs, one idea each. Bold at most the load-bearing claim of
    a section. No exclamation marks, no "remarkably", no "earth-shattering".
17. State scope and limits plainly in the same section as the claim, not in
    a separate "honesty" chapter (a limits chapter may *collect* them, but
    each claim carries its own).
18. Do not make a result look better than it is. If a comparison is unfair,
    fix it or drop it; do not caption around it.

## Cross-references and mechanics

19. Keep `{#sec-...}` anchors that other chapters or the notebook reference
    (a list is provided with each rewrite brief). New anchors are lowercase
    kebab-case and unique across the book.
20. Every chapter must render: run
    `uv run --no-sync quarto render <file>.qmd` from `book/` and fix errors.
21. Do not edit `18-notebook.qmd` or `results/` from a chapter rewrite.
