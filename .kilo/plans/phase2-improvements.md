# Phase 2 Improvements Plan

## Overview
This plan focuses on developer experience, code quality tooling, and API polish improvements that provide high value with relatively low effort.

## Priority Items

### 1. Add Pre-commit Hooks and CI/CD (High Priority, Low Effort)

**Files to create/modify:**
- `.pre-commit-config.yaml` (new)
- `.github/workflows/ci.yml` (new)
- `pyproject.toml` (update)

**Why:** Automated code quality checks prevent issues from being committed, ensure consistent formatting, and catch type errors early.

**Pre-commit hooks to include:**
- ruff (linting and formatting)
- mypy (type checking)
- trailing whitespace, end-of-file fixes

**CI/CD workflow:**
- Test matrix: Python 3.9, 3.10, 3.11, 3.12
- Run pytest with coverage
- Run ruff check
- Run mypy (optional, can be warnings-only initially)

### 2. Update pyproject.toml with Dev Tools Configuration (High Priority, Low Effort)

**Changes:**
- Add ruff configuration (line-length, select rules)
- Add mypy configuration
- Update dev dependencies
- Widen Polars version constraint

**Ruff rules to enable:**
- E: pycodestyle errors
- F: pyflakes
- I: isort
- UP: pyupgrade
- B: flake8-bugbear
- SIM: flake8-simplify

### 3. Use pl.selectors for Column Selection (Medium Priority, Low Effort)

**Files to modify:**
- `src/feataz/base.py` - Update ColumnInferenceMixin to use selectors
- `src/feataz/encoders.py` - Update _infer_categorical_columns
- `src/feataz/discretize.py` - Update _infer_numeric_columns
- Other modules as needed

**Why:** `pl.selectors` is the modern Polars way to select columns by type, more readable and maintainable.

**Example:**
```python
# Before
def _infer_numeric_columns(df, columns):
    if columns is not None:
        return list(columns)
    return [n for n, t in zip(df.columns, df.dtypes) if t.is_numeric()]

# After
import polars.selectors as cs

def _infer_numeric_columns(df, columns):
    if columns is not None:
        return list(columns)
    return df.select(cs.numeric()).columns
```

### 4. Add Column Name Collision Prevention (Medium Priority, Medium Effort)

**Files to modify:**
- `src/feataz/base.py` - Add helper method
- All transformers with suffix output

**Approach:**
- Add `if_exists` parameter to transformers: "error" | "append_unique" | "overwrite"
- Default to "error" for safety
- "append_unique" adds a counter suffix if column exists

### 5. Make CrossFitTransformer.oof_train_ Public (Low Priority, Low Effort)

**Files to modify:**
- `src/feataz/advanced.py`

**Change:**
```python
@property
def oof_train(self) -> pl.DataFrame:
    """Out-of-fold transformed training data (only available after fit)."""
    if not self.is_fitted_:
        raise RuntimeError("Call fit() first")
    return self._oof_train_
```

### 6. Add Logging for Edge Cases (Medium Priority, Low Effort)

**Files to modify:**
- All transformer modules

**Approach:**
- Add `import logging` at module level
- Log warnings for edge cases (insufficient unique values, empty bins, constant columns)
- Use DEBUG level for verbose information

### 7. Standardize drop_original Default (Medium Priority, Low Effort)

**Current state:**
- OneHotEncoder: `drop_original=True`
- RareLabelEncoder: `drop_original=False`
- TimeSnapshotAggregator: `drop_original=False`

**Recommendation:** Document the convention or standardize to `drop_original=False` for safety.

### 8. Add CONTRIBUTING.md (Low Priority, Low Effort)

**Content:**
- Development setup instructions
- How to run tests
- Code style guidelines
- How to add new transformers
- Pull request process

## Implementation Order

1. **pyproject.toml updates** (5 min)
2. **Pre-commit hooks** (10 min)
3. **CI/CD workflow** (10 min)
4. **pl.selectors migration** (15 min)
5. **CrossFitTransformer.oof_train property** (2 min)
6. **Logging for edge cases** (15 min)
7. **Column collision prevention** (20 min)
8. **CONTRIBUTING.md** (10 min)

## Expected Outcomes

- **Code Quality:** Automated linting and type checking
- **Developer Experience:** Clear contribution guidelines and consistent code style
- **API Polish:** Better handling of edge cases and column name conflicts
- **Maintainability:** Modern Polars patterns and better logging

## Testing Strategy

- All changes should pass existing 42 tests
- Add tests for column collision prevention
- Verify CI/CD runs successfully on all Python versions
