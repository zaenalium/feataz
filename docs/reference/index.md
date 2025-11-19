# Transformer catalog

Each transformer in `feataz` inherits from the lightweight `Transformer` base class defined in
`src/feataz/base.py`. The base takes care of handling Polars vs. pandas inputs, exposes
`fit_transform`, and provides serialization helpers through `to_dict` / `from_dict`.

Use this catalog to find the module, summary, and the most important constructor arguments for
every public class exported from `feataz.__init__`.

## Categorical encoders (`feataz.encoders`)

| Transformer | Summary | Key options |
| --- | --- | --- |
| `OneHotEncoder` | Expands low-cardinality categoricals into dummy columns and optionally keeps originals. | `drop_original`, `handle_unknown`, `min_frequency`, `top_k_per_column`. |
| `OrdinalEncoder` | Assigns stable integer codes, optionally keeping the original categorical column. | `drop_original`, `handle_unknown`, `encoding_strategy` (`frequency`, `sorted`, or custom map). |
| `CountFrequencyEncoder` | Replaces categories with raw counts or normalized frequencies. | `normalize`, `drop_original`, `handle_missing`. |
| `MeanEncoder` | Target encoding with optional smoothing toward the global mean to prevent leakage. | `target`, `smoothing`, `groupby`, `noise_std`. |
| `WeightOfEvidenceEncoder` | Binary or one-vs-rest weight-of-evidence transformation. | `target`, `multi_class`, `clip`. |
| `DecisionTreeEncoder` | Fits a shallow univariate tree per column (requires scikit-learn) and emits leaf indices. | `problem` (`auto`, `regression`, `classification`), `max_leaf_nodes`, `min_samples_leaf`. |
| `RareLabelEncoder` | Groups infrequent categories into a catch-all bucket while preserving option to drop the original. | `min_frequency`, `rare_label`, `drop_original`. |
| `StringSimilarityEncoder` | Computes similarity scores with respect to anchor values using `difflib.SequenceMatcher`. | `top_k_anchors`, `anchors`, `drop_original`, `normalize`. |
| `HashEncoder` | Applies a feature hashing trick to categorical columns with a stable number of components. | `n_components`, `salt`, `drop_original`. |
| `BinaryEncoder` | Converts categories to binary digits across sufficient bit columns. | `drop_original`, `min_frequency`. |
| `LeaveOneOutEncoder` | Smoothed leave-one-out (cross-validated) target encoding. | `target`, `smoothing`, `randomized`. |

## Discretizers (`feataz.discretize`)

| Transformer | Summary | Key options |
| --- | --- | --- |
| `EqualFrequencyDiscretizer` | Splits numeric columns into quantile buckets with equal row counts. | `n_bins`, `right`, `labels`. |
| `EqualWidthDiscretizer` | Uniform-width bins spanning the numerical min/max range. | `n_bins`, `include_lowest`, `labels`. |
| `ArbitraryDiscretizer` | Uses user-provided bin edges for deterministic mappings. | `bins`, `right`, `labels`. |
| `DecisionTreeDiscretizer` | Learns splits using a univariate decision tree (scikit-learn). | `target`, `max_leaf_nodes`, `min_samples_leaf`. |
| `GeometricWidthDiscretizer` | Builds exponentially growing bin widths for skewed features. | `n_bins`, `ratio`. |
| `KMeansDiscretizer` | Clusters numeric values into `n_bins` centroids (requires scikit-learn). | `n_bins`, `random_state`, `max_iter`. |
| `MDLPDiscretizer` | Supervised Minimum Description Length Principle binning with stopping heuristics. | `target`, `min_samples_bin`, `max_bins`. |
| `ChiMergeDiscretizer` | Chi-square-based supervised binning with optional post-merge pruning. | `target`, `max_bins`, `alpha`, `initial_bins`, `min_samples_bin`. |
| `IsotonicBinningDiscretizer` | Fits a monotonic trend per column using isotonic regression. | `target`, `monotonic`, `max_bins`, `min_samples_bin`. |
| `MonotonicOptimalBinningDiscretizer` | Grid-searches monotonic bins with post-merge optimization. | `target`, `initial_bins`, `min_bin_size`, `max_bins`. |

## Variance stabilizing transforms (`feataz.vst`)

| Transformer | Summary | Key options |
| --- | --- | --- |
| `LogTransformer` | Applies a log transform with automatic epsilon handling. | `columns`, `shift`, `drop_original`. |
| `LogCPTransformer` | `log(x + c)` transform for count-plus-offset inputs. | `c`, `columns`. |
| `ReciprocalTransformer` | Converts values via `1 / x`. Useful for heavy-tailed positive features. | `clip`, `columns`. |
| `ArcsinTransformer` | `arcsin(sqrt(x))` style transform for proportions. | `clip_range`, `columns`. |
| `PowerTransformer` | Raises values to a fixed power (square roots, cubes, etc.). | `power`, `columns`. |
| `BoxCoxTransformer` | Grid-search-based Box-Cox lambda solution without SciPy. | `columns`, `lambdas`, `drop_original`. |
| `YeoJohnsonTransformer` | Yeo-Johnson transform supporting zero and negative values. | `columns`, `lambdas`, `drop_original`. |

## Feature builders & interactions (`feataz.features`, `feataz.interaction`, `feataz.snapshot`)

| Transformer | Module | Summary | Key options |
| --- | --- | --- | --- |
| `MathFeatures` | `features` | Generates unary transforms (log, sqrt, etc.), polynomial powers, and pairwise ops. | `columns`, `unary_ops`, `powers`, `binary_ops`, `keep_original`. |
| `RelativeFeatures` | `features` | Computes ratios, differences, and percentage deltas between target/reference columns. | `target_columns`, `reference_columns`, `operations`. |
| `CyclicalFeatures` | `features` | Converts periodic values (hour of day, day of week) into sine/cosine pairs. | `columns`, `period`, `drop_original`. |
| `DecisionTreeFeatures` | `features` | Extracts predictions or leaf indices from a scikit-learn tree fit on selected columns. | `target`, `problem`, `output`, `max_depth`. |
| `FeatureInteractions` | `interaction` | Group-by aggregations that roll up value columns across aggregations. | `groupby`, `value_columns`, `aggregations`, `suffix`. |
| `TimeSnapshotAggregator` | `snapshot` | Trailing window aggregations over datetime columns. | `time_column`, `groupby`, `value_columns`, `windows`, `aggregations`, `include_current`. |
| `DynamicSnapshotAggregator` | `snapshot` | Calendar-based rolling snapshots that advance by `every` (e.g., weekly). | `every`, `offset`, `value_columns`, `aggregations`. |
| `ToDateSnapshotAggregator` | `snapshot` | Month-to-date / quarter-to-date aggregations with optional exclusion of the current row. | `period`, `include_current`, `groupby`. |
| `EWMAggregator` | `snapshot` | Exponentially weighted moving averages per group. | `alpha`, `adjust`, `groupby`, `value_columns`. |

## Imputation (`feataz.impute`)

| Transformer | Summary | Key options |
| --- | --- | --- |
| `SimpleImputer` | Column-wise strategy maps for numeric and categorical data plus missing indicators. | `numeric_strategy_map`, `categorical_strategy_map`, `fill_values`, `add_indicator`. |
| `GroupImputer` | Imputes within groups defined by one or more columns. | `groupby`, `columns`, `numeric_strategy`, `categorical_strategy`. |
| `KNNImputer` | Delegates imputation to scikit-learn's KNNImputer for the provided numeric columns. | `n_neighbors`, `weights`. |
| `IterativeImputer` | MICE-style iterative imputation powered by scikit-learn. | `max_iter`, `initial_strategy`, `estimator`. |
| `TimeSeriesImputer` | Forward/backward fills ordered by a timestamp per group. | `time_column`, `groupby`, `method` (`forward`, `backward`, `both`). |

## Outlier handling (`feataz.outliers`)

| Transformer | Summary | Key options |
| --- | --- | --- |
| `ClipOutliers` | Detects outliers via quantiles or IQR and either clips, drops, or flags them. | `method` (`quantile`, `iqr`), `action`, `q_low`, `q_high`, `iqr_factor`, `add_indicator`. |
| `IsolationForestOutlierHandler` | Fits scikit-learn's Isolation Forest to numeric columns. | `contamination`, `action`, `add_score`. |

## Scaling and ranks (`feataz.scale`)

| Transformer | Summary | Key options |
| --- | --- | --- |
| `RobustScaler` | Scales values using the median and IQR instead of mean/std. | `quantile_range`, `center`, `scale`, `groupby`. |
| `QuantileRankTransformer` | Replaces values with rank/percentile scores, optionally within a group. | `groupby`, `method` (`dense`, `ordinal`), `normalize`. |

## Feature selection (`feataz.selection`)

| Transformer | Summary | Key options |
| --- | --- | --- |
| `VarianceThresholdSelector` | Drops columns with low variance using a numeric threshold. | `threshold`, `columns`. |
| `MutualInformationSelector` | Keeps the top-k features by MI score (classification or regression). | `target`, `k`, `discrete_features`. |
| `ModelBasedImportanceSelector` | Uses a scikit-learn estimator's `feature_importances_` to retain high-signal columns. | `estimator`, `top_k`, `importance_threshold`. |
| `MRMRSelector` | Implements the Max Relevance, Min Redundancy criterion. | `target`, `k`, `redundancy_weight`. |

## Diagnostics (`feataz.diagnostics`)

| Function | Summary | Usage |
| --- | --- | --- |
| `information_value` | Calculates IV for a binned feature vs. a binary target. | Call after a discretizer to quantify predictive strength. |
| `ks_statistic` | Computes the Kolmogorov–Smirnov statistic for a score column. | Useful for monitoring classifier drift. |
| `psi` | Population Stability Index between expected and actual score arrays. | Monitor distribution shift in production. |

<a id="autofeaturizer-and-suggestions"></a>
## AutoFeaturizer and suggestions (`feataz.auto`, `feataz.advanced`)

| Component | Summary | Key options |
| --- | --- | --- |
| `AutoFeaturizer` | Inspects schema + target column and executes a configurable pipeline (imputation, ranks, encoders). | `target_column`, `include_outliers`, `add_ranks`, `drop_original`. |
| `suggest_methods` | Returns a dict describing recommended transformers for each column. The output can seed `AutoFeaturizer`. | `target_column`, `include_outliers`, `max_cardinality`. |
| `CrossFitTransformer` | Wraps an arbitrary transformer factory and produces cross-fold out-of-fold features to avoid leakage. | `n_splits`, `shuffle`, `random_state`, `transformer_factory`. |

All transformers and helper functions are re-exported from `feataz.__init__`, so you can import
from the package root without referencing submodules explicitly.
