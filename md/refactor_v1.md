# Refactor and Clean the `infraslow` Python Package

Act as a senior Python developer experienced in scientific computing, EEG signal processing, dependency analysis, and safe codebase refactoring.

Refactor and clean the code under:

```text
infraslow/src/infraslow/
```

The main goal is to remove unused code and improve the package structure **without changing any existing scientific or computational logic**.

---

## 1. Entry Points That Must Continue Working

Determine which modules, classes, functions, constants, and utilities are required by these files:

```text
infraslow/src/run_all_metrics.py
infraslow/src/plot_infraslow.ipynb
infraslow/src/demo_infraslow_yasa.ipynb
infraslow/src/demo_infraslow_yasa_average.ipynb
```

These four files are the authoritative entry points for determining whether code is still used.

Trace both direct and indirect dependencies.

For example:

```text
run_all_metrics.py
    └── imports function A
            └── calls function B
                    └── uses constant C
```

Functions `A`, `B`, and constant `C` are all considered used and must remain.

Do not check only direct imports.

---

## 2. Protected Visualization Package

Do not remove any function, class, constant, or module inside:

```text
infraslow/src/infraslow/viz/
```

The entire `viz` package is protected.

You may make only minimal non-functional changes inside `viz` when required to repair imports caused by moving other modules. However:

* do not delete visualization functions;
* do not simplify visualization functions;
* do not change visualization behavior;
* do not change plotting output;
* do not rename the public visualization API;
* do not remove apparently unused visualization helpers.

Treat all code under `infraslow/src/infraslow/viz/` as externally used.

---

## 3. Primary Refactoring Objectives

Refactor `infraslow/src/infraslow/` to:

1. Remove unused functions.
2. Remove unused classes.
3. Remove unused constants.
4. Remove unused modules when the entire module is unnecessary.
5. Remove unused imports.
6. Remove unreachable or dead code.
7. Remove duplicated code only when it can be consolidated without changing behavior.
8. Move functions into more appropriate modules when the current location is clearly incorrect.
9. Improve module organization and naming where safe.
10. Preserve all currently used public interfaces whenever possible.

Do not perform a major redesign.

The result should be a smaller, clearer, and easier-to-maintain package while preserving current behavior.

---

## 4. Strict No-Logic-Change Requirement

Do not change scientific, signal-processing, statistical, or data-processing logic.

This includes, but is not limited to:

* PSG loading;
* channel mapping;
* hypnogram processing;
* spindle detection;
* spindle-to-bout assignment;
* bout construction;
* sleep-stage definitions;
* spectral analysis;
* infraslow spectrum correction;
* frequency ranges;
* filters;
* thresholds;
* normalization;
* interpolation;
* averaging;
* aggregation;
* sleep-metric calculation;
* missing-data handling;
* NPZ output values;
* metadata calculation;
* multiprocessing behavior;
* plotting calculations.

Do not change:

* formulas;
* numerical thresholds;
* default parameters;
* return values;
* output column names;
* NPZ keys;
* array shapes;
* array dtypes;
* time units;
* channel ordering;
* sleep-stage coding;
* exception behavior relied upon by the entry points.

Refactoring must be behavior-preserving.

---

## 5. Inspect the Notebooks Properly

The three notebooks must be included in the dependency analysis:

```text
infraslow/src/plot_infraslow.ipynb
infraslow/src/demo_infraslow_yasa.ipynb
infraslow/src/demo_infraslow_yasa_average.ipynb
```

Inspect all code cells for:

* normal imports;
* aliased imports;
* wildcard imports;
* direct function calls;
* module-qualified calls;
* imported constants;
* dynamically accessed attributes;
* calls made through intermediate variables;
* commented examples that represent intended public usage.

Examples:

```python
from infraslow.processing import calculate_spectrum
```

```python
import infraslow.processing as processing
processing.calculate_spectrum(...)
```

```python
from infraslow.processing import *
```

Be conservative when wildcard imports or dynamic attribute access make usage uncertain.

Do not remove code merely because static analysis cannot detect notebook usage with complete certainty.

---

## 6. Dependency Analysis

Before modifying code, create an internal dependency map covering:

```text
entry point
    → imported module
        → imported function or class
            → internal helper
                → shared constant or utility
```

Include:

* direct imports;
* relative imports;
* re-exported functions;
* imports from `__init__.py`;
* function-to-function calls;
* class construction;
* method calls;
* decorators;
* callback functions;
* multiprocessing worker functions;
* dynamically selected functions where identifiable;
* notebook dependencies;
* protected `viz` dependencies.

Check every `__init__.py` file because symbols may be re-exported from there.

A function is not unused merely because it is not imported directly by an entry point. It may be called by another required function.

---

## 7. Conservative Removal Policy

Remove a function, class, constant, or module only when you can establish that it is not used by:

1. `infraslow/src/run_all_metrics.py`;
2. `infraslow/src/plot_infraslow.ipynb`;
3. `infraslow/src/demo_infraslow_yasa.ipynb`;
4. `infraslow/src/demo_infraslow_yasa_average.ipynb`;
5. any transitive dependency of those files;
6. any code under `infraslow/src/infraslow/viz/`;
7. package initialization or re-export logic required by those files.

When usage is ambiguous, retain the code and report it as uncertain rather than deleting it.

Do not delete a symbol solely because an IDE, linter, or static-analysis tool marks it as unused.

Validate the result against actual imports and execution paths.

---

## 8. Moving Functions Between Modules

You may move a function when it is clearly located in an inappropriate module.

Examples of potentially appropriate organization:

```text
infraslow/
├── io/
│   ├── psg.py
│   └── annotations.py
├── processing/
│   ├── spindles.py
│   ├── bouts.py
│   ├── spectra.py
│   └── sleep_metrics.py
├── utils/
│   ├── validation.py
│   └── paths.py
├── stats/
└── viz/
```

This is only an example. Adapt to the existing repository structure rather than forcing a new architecture.

When moving a used function:

* update every import;
* update notebook imports;
* update relative imports;
* update `__init__.py` exports;
* avoid circular imports;
* preserve the function signature;
* preserve defaults;
* preserve return type and structure;
* preserve exceptions and side effects.

Where practical, maintain a backward-compatible import alias if moving the function could break external code.

For example:

```python
# Backward-compatible re-export
from .new_module import existing_function
```

Do not create unnecessary compatibility wrappers for functions that are confirmed to be entirely internal.

---

## 9. Duplicate Code

Identify duplicated implementations, especially for:

* channel resolution;
* hypnogram conversion;
* sleep-stage masks;
* contiguous-bout detection;
* spindle filtering;
* spectrum aggregation;
* input validation;
* empty-array creation;
* file-path construction.

Consolidate duplicates only when their behavior is genuinely equivalent.

Before combining duplicated functions, compare:

* parameter defaults;
* accepted input types;
* boundary conditions;
* NaN handling;
* empty-input handling;
* return types;
* time units;
* array dtypes;
* exception behavior.

Do not merge functions that look similar but implement slightly different scientific behavior.

---

## 10. Imports and Module Cleanup

Remove unused imports from modules that remain.

Organize imports in this order:

```python
# Standard library

# Third-party packages

# Local package imports
```

Do not introduce wildcard imports.

Avoid optional imports at module level when they cause the package to fail even though the associated functionality is not being used.

Preserve lazy imports when they are necessary for:

* optional dependencies;
* multiprocessing;
* heavy scientific packages;
* circular-import prevention.

Do not move imports purely for style if doing so changes initialization behavior.

---

## 11. Public API and `__init__.py`

Review all package `__init__.py` files.

Remove stale exports only when the exported symbol has been safely removed.

Preserve exports used by the entry points and notebooks.

Make sure the following continue to work where currently supported:

```python
from infraslow.some_module import some_function
```

and:

```python
from infraslow import some_function
```

Do not silently break an existing public import path used by the specified files.

Avoid eagerly importing heavy modules into the top-level package unless the current code requires it.

---

## 12. Multiprocessing Safety

Pay special attention to functions used by:

```text
infraslow/src/run_all_metrics.py
```

Do not remove or nest functions used as multiprocessing workers.

Multiprocessing worker functions should remain importable at module scope when required for pickling.

Do not convert module-level worker functions into:

* nested functions;
* lambdas;
* local closures;
* non-picklable callables.

Do not change process-pool behavior, worker counts, task ordering, or error handling unless necessary to preserve the current execution.

---

## 13. Scientific Reproducibility

The refactored code must generate equivalent outputs for the same inputs.

Check equivalence for:

* metadata values;
* detected spindle start, stop, and peak times;
* bout start and stop times;
* spindle counts per bout;
* spectral frequency arrays;
* corrected mean spectra;
* NPZ key names;
* NPZ array shapes and dtypes;
* missing-channel behavior;
* subjects with empty results.

Use exact equality where deterministic and appropriate.

For floating-point arrays, use an appropriately strict comparison such as:

```python
np.testing.assert_allclose(
    actual,
    expected,
    rtol=1e-10,
    atol=1e-12,
    equal_nan=True,
)
```

Do not use loose tolerances that could hide a scientific-logic change.

---

## 14. Required Validation

Before refactoring, capture baseline behavior for a small representative sample.

Use at least:

* one subject with all expected EEG channels;
* one subject with one or more missing channels;
* one subject with no valid spindle bouts, when available.

Compare the original and refactored outputs.

Validate:

```text
metadata columns
metadata values
NPZ filenames
NPZ keys
array shapes
array dtypes
spindle start values
spindle stop values
spindle peak values
bout start values
bout stop values
n_spindles values
freqs values
corr_mean values
```

Also verify that the four entry points still import and run through their relevant code paths.

For notebooks, at minimum:

* parse every code cell;
* validate imports;
* execute the notebook when the environment and data permit;
* otherwise provide a clear report of which cells could not be executed and why.

---

## 15. Static Analysis Tools

You may use tools such as:

```text
ruff
pyflakes
vulture
pytest
coverage
```

However, do not automatically delete everything reported as unused.

Static-analysis results are only candidates for manual review.

Pay special attention to false positives caused by:

* notebooks;
* dynamic imports;
* re-exports;
* multiprocessing;
* decorators;
* callbacks;
* protected visualization functions;
* configuration-driven calls.

---

## 16. Tests

Add or update focused tests where practical.

Tests should cover important retained behavior, including:

* channel mapping;
* sleep-stage masking;
* N2 bout detection;
* N3 bout detection;
* combined NREM bout detection;
* spindle assignment using peak time;
* minimum bout duration;
* spindle-count calculation;
* empty-input handling;
* spectral output shape;
* NPZ serialization;
* missing-channel handling.

Do not rewrite scientific tests merely to make changed behavior pass.

Tests must confirm that behavior remains unchanged.

---

## 17. Documentation and Comments

Remove comments that are:

* obsolete;
* misleading;
* duplicated;
* describing code that no longer exists.

Retain comments that explain:

* scientific reasoning;
* non-obvious thresholds;
* frequency ranges;
* time conversions;
* spindle assignment;
* edge-case behavior;
* external-library workarounds.

Improve unclear docstrings without changing the documented behavior.

Use concise NumPy-style or Google-style docstrings consistently with the existing project.

---

## 18. Code-Quality Requirements

The final code should:

* use clear function and variable names;
* avoid unnecessary abstraction;
* avoid oversized multipurpose modules when safe to split;
* avoid circular imports;
* avoid duplicate implementations;
* use `pathlib.Path` consistently where it is already appropriate;
* preserve type hints or add them where useful;
* preserve existing logging behavior;
* avoid broad `except Exception` unless required at a subject or worker boundary;
* include context in raised or logged errors;
* avoid mutable default arguments;
* avoid module-level execution with side effects.

Do not perform cosmetic rewrites that produce a very large diff without improving maintainability.

Prefer focused, reviewable changes.

---

## 19. Files That Must Not Be Modified Unnecessarily

Do not modify these entry points beyond necessary import-path updates:

```text
infraslow/src/run_all_metrics.py
infraslow/src/plot_infraslow.ipynb
infraslow/src/demo_infraslow_yasa.ipynb
infraslow/src/demo_infraslow_yasa_average.ipynb
```

Do not change notebook analysis logic, plotting logic, parameters, or outputs.

Only update imports or module paths when required by the refactor.

Do not remove any code from:

```text
infraslow/src/infraslow/viz/
```

---

## 20. Required Deliverables

After completing the refactor, provide:

1. The complete refactored code.
2. A list of all modified files.
3. A list of all removed files.
4. A list of all removed functions, classes, and constants.
5. For every removed symbol, explain why it was safe to remove.
6. A list of functions or modules that appeared unused but were retained because their usage was uncertain.
7. A dependency summary for each authoritative entry point.
8. A summary of functions moved between modules.
9. A list of updated import paths.
10. A summary of duplicated code that was consolidated.
11. Baseline-versus-refactored validation results.
12. Test results.
13. Static-analysis results before and after refactoring.
14. Confirmation that no code was removed from `infraslow/src/infraslow/viz/`.
15. Confirmation that scientific and computational behavior was not changed.

Use a summary table similar to:

| Item           | Before | After | Notes |
| -------------- | -----: | ----: | ----- |
| Python modules |    ... |   ... | ...   |
| Functions      |    ... |   ... | ...   |
| Classes        |    ... |   ... | ...   |
| Lines of code  |    ... |   ... | ...   |
| Unused imports |    ... |   ... | ...   |
| Tests passing  |    ... |   ... | ...   |

Also provide a removal table:

| Removed symbol or file | Previous location | Evidence it was unused                                                                   |
| ---------------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `example_function`     | `module.py`       | No direct or transitive references from the four entry points or protected `viz` package |

---

## 21. Final Constraints

The following constraints are mandatory:

```text
Do not change scientific logic.
Do not change numerical output.
Do not change function behavior.
Do not remove transitive dependencies.
Do not remove functions from infraslow/src/infraslow/viz/.
Do not break imports used by the four specified entry points.
Do not remove code based only on static-analysis warnings.
Do not provide only recommendations—apply the refactor.
```

Make the smallest safe set of changes that meaningfully removes unused code and improves organization.
