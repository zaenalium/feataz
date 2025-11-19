# feataz documentation

`feataz` is a batteries-included collection of feature engineering helpers that sit on top of
[Polars](https://www.pola.rs/). Every transformer follows the familiar `fit` / `transform`
contract and focuses on being light-weight, dependency-aware, and fast enough for everyday
analytics work.

## Highlights

- **Consistent API** – every transformer inherits from a tiny `Transformer` base class so they all
  share `fit`, `transform`, and `fit_transform` methods.
- **Broad coverage** – categorical encoders, discretizers, scaling, variance-stabilizing
  transforms, interaction builders, snapshot aggregations, imputers, outlier guards, diagnostics,
  and even an automated featurizer live in `src/feataz`.
- **Polars-first** – by default everything works on `polars.DataFrame` objects, but key entry
  points accept pandas objects and convert them internally.
- **Incremental adoption** – string similarity, decision-tree helpers, advanced discretizers,
  and scikit-learn based utilities are optional extras; you can install only what you need.

## Directory tour

- `src/feataz`: the implementation modules. Each module is documented on the
  [Transformer catalog](reference/index.md) page.
- `examples/smoke.py`: a runnable walkthrough that exercises most transformers in one go.
- `notebook/feature_engineering_on_titanic_data.ipynb`: a narrative case study.
- `notebook/tutorials/`: ten Jupytext notebooks that mirror the same ideas in different datasets.

## Where to start

1. Follow the [getting started guide](getting-started.md) to install the package and run the quick
   start snippet.
2. Browse the [Transformer catalog](reference/index.md) to understand what each module provides.
3. Reproduce the walkthroughs in [Examples & Tutorials](examples.md) to see real DataFrames being
   transformed end-to-end.

Use the search bar in the generated site to jump directly to a class or method name. Every page
links back to the relevant source file so you can inspect or extend the underlying implementation.
