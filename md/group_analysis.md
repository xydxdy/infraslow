# Create N2–C3 Spindle-Rate Group Analysis

Act as a senior Python developer and biostatistician experienced in sleep EEG, spindle detection, infraslow spectral analysis, and reproducible scientific Python development.

## Objective

Use the existing analysis in:

```text
infraslow/src/demo_infraslow_yasa_compare.py
```

to create:

```text
infraslow/src/group_analysis.py
```

The analysis must be restricted to:

```text
Sleep state: N2
EEG channel: C3
```

Use infraslow-py310 conda environment

The new analysis should:

1. Load the subject-level N2–C3 spindle and infraslow results.
2. Validate and prepare the subject-level dataset.
3. Divide subjects into two groups using N2–C3 `spindle_per_min`.
4. Compare N2–C3 infraslow summary parameters between the groups.
5. Plot and compare the N2–C3 infraslow spectra using the same plotting logic and visual components as `demo_infraslow_yasa_compare.py`.

Do not load metadata in this task.

Do not analyze demographic or sleep-architecture variables yet.

Do not change the existing spindle-detection, infraslow calculation, spectral correction, fitting, or plotting logic.

---

## Fixed Analysis Scope

Use only:

```python
sleep_stage = "N2"
channel = "C3"
```

Requirements:

* Filter the input results to N2 before group assignment or statistical analysis.
* Filter the EEG results to C3 before group assignment or statistical analysis.
* Do not combine N2 with N3, NREM, wake, REM, or whole-night results.
* Do not combine C3 with C4, F3, F4, O1, O2, or any other channel.
* Do not average across channels.
* Do not use another channel as a fallback when C3 is unavailable.
* Subjects without valid N2–C3 results must be reported and excluded with a clear reason.
* Include `sleep_stage` and `channel` columns in saved output tables.
* Verify that every included row has:

```text
sleep_stage = N2
channel = C3
```

The script should fail with a clear error if no valid N2–C3 records are available.

---

## Step 1: Inspect the Existing Analysis

Inspect:

```text
infraslow/src/demo_infraslow_yasa_compare.py
```

Determine:

* How subject IDs are represented.
* How the script selects sleep stages.
* How the script selects EEG channels.
* Where subject-level results are generated or stored.
* How the N2 C3 infraslow spectrum is stored.
* How frequency arrays are stored.
* How corrected infraslow power arrays are stored.
* How spindle detections are restricted to N2.
* How the following parameters are calculated:

```text
peak_freq_hz
peak_period_s
bandwidth_hz
auc
chromatogram_peak_area
spindle_per_min
spindle_per_min_SEM
```

Also inspect exactly how `demo_infraslow_yasa_compare.py` creates its comparison plot, including:

* Raw or original spectrum.
* Corrected spectrum.
* Baseline or background component.
* Bi-Gaussian fitted curve.
* Peak-frequency marker.
* Bandwidth representation.
* AUC visualization.
* Chromatogram peak-area visualization.
* Spindle-rate annotation.
* Axis limits.
* Frequency range.
* Labels.
* Legends.
* Figure layout.
* Titles and annotations.

Reuse the same project functions and plotting logic wherever possible.

Do not create a simplified unrelated plot when the existing script already has the required comparison visualization.

Do not copy large blocks of code unnecessarily. Refactor reusable functions only when needed.

---

## Step 2: Load N2–C3 Subject-Level Results

Load the subject-level results produced or used by:

```text
infraslow/src/demo_infraslow_yasa_compare.py
```

Filter the results to:

```text
state = N2
channel = C3
```

Use the actual state and channel column names found in the existing pipeline.

The analysis requires at least:

```text
subject_id
sleep_stage
channel
peak_freq_hz
peak_period_s
bandwidth_hz
auc
chromatogram_peak_area
spindle_per_min
spindle_per_min_SEM
```

It must also load the arrays needed for the spectrum comparison plots, such as:

```text
freqs
raw_power
baseline
corrected
corr_mean
fitted_curve
```

Use the actual names and data structures found in the existing code.

Requirements:

* Do not assume the input format before inspecting the existing script.
* Support the existing result format.
* Standardize subject IDs consistently.
* Confirm that summary values and spectrum arrays refer to the same subject.
* Detect duplicated N2–C3 subject records.
* Do not silently remove conflicting duplicates.
* Report subjects missing N2 data.
* Report subjects missing the C3 channel.
* Report subjects missing summary parameters.
* Report subjects missing spectrum arrays.

The script should accept input paths through command-line arguments matching the existing result format, for example:

```text
--results
```

or, when stored separately:

```text
--summary-results
--spectrum-results
```

---

## Step 3: Validate the N2–C3 Dataset

Validate:

```text
peak_freq_hz
peak_period_s
bandwidth_hz
auc
chromatogram_peak_area
spindle_per_min
spindle_per_min_SEM
```

Checks must include:

* Missing values.
* Infinite values.
* Duplicate subject records.
* Incorrect sleep state.
* Incorrect EEG channel.
* Negative `spindle_per_min`.
* Negative `spindle_per_min_SEM`.
* Nonpositive `peak_freq_hz`.
* Nonpositive `peak_period_s`.
* Negative `bandwidth_hz`.
* Negative `auc`.
* Negative `chromatogram_peak_area`.
* Empty frequency arrays.
* Empty power arrays.
* Unequal frequency and spectrum-array lengths.
* Non-finite frequency or power values.
* Non-increasing frequency arrays.
* Invalid fitted curves.

Check the consistency of:

```python
peak_period_s ≈ 1.0 / peak_freq_hz
```

Use a documented numerical tolerance.

Do not silently recalculate or overwrite inconsistent values.

Save all validation warnings and exclusion reasons.

Make sure there is one independent spindle-rate observation per subject for N2 and C3.

Save:

```text
validated_N2_C3_subject_results.csv
invalid_or_excluded_N2_C3_subjects.csv
validation_report_N2_C3.txt
```

The validation report should include:

* Total subjects loaded.
* Number with N2 data.
* Number with C3 data.
* Number with valid N2–C3 data.
* Number excluded.
* Duplicate count.
* Missingness for every required parameter.
* Number with valid spectrum arrays.
* Reasons for exclusion.

---

## Step 4: Divide Subjects into Two Groups

Use the N2–C3 value:

```text
spindle_per_min
```

to divide subjects into:

```text
low_spindle_rate
high_spindle_rate
```

Use a two-component Gaussian Mixture Model as the primary grouping method.

### GMM Requirements

* Use one valid N2–C3 `spindle_per_min` value per subject.
* Use a fixed random seed.
* Evaluate:

  * Raw `spindle_per_min`
  * `log1p(spindle_per_min)`
* Fit a two-component GMM for each representation.
* Compare the models using BIC.
* Select the representation with the lower BIC.
* Order the final labels using the component centers on the original spindle-rate scale.

The component with the lower spindle-rate center must be:

```text
low_spindle_rate
```

The component with the higher spindle-rate center must be:

```text
high_spindle_rate
```

Save posterior group-assignment probabilities.

Mark a subject as uncertain when:

```text
group_probability < 0.70
```

Make the threshold configurable.

Do not exclude uncertain subjects from the primary analysis.

Save:

```text
N2_C3_subject_group_assignments.csv
```

with at least:

```text
subject_id
sleep_stage
channel
spindle_per_min
spindle_per_min_SEM
spindle_group
group_probability
uncertain_assignment
gmm_input_scale
raw_scale_bic
log1p_scale_bic
```

### Spindle-Rate Distribution Plot

Create:

```text
N2_C3_spindle_rate_group_distribution.png
```

The figure should show:

* Distribution of N2–C3 `spindle_per_min`.
* Two fitted GMM components.
* Group centers.
* Group membership.
* Estimated decision boundary.
* Number of subjects in each group.
* Number of uncertain assignments.

Because the groups are constructed from `spindle_per_min`, do not use a statistical difference in `spindle_per_min` as independent evidence supporting the groups.

---

## Step 5: Compare N2–C3 Infraslow Summary Parameters

Compare the following N2–C3 parameters between the two spindle-rate groups:

```text
power
peak_freq_hz
peak_period_s
bandwidth_hz
auc
chromatogram_peak_area
```

First inspect the existing pipeline and identify the correct subject-level definition of `power`.

Do not invent a new power parameter.

Document whether `power` represents:

* Mean corrected infraslow power.
* Maximum corrected power.
* Integrated infraslow-band power.
* Fitted peak amplitude.
* Another existing output.

### Descriptive Statistics

For each parameter and group, report:

```text
n
missing_n
mean
standard_deviation
median
q1
q3
minimum
maximum
```

### Statistical Testing

Use:

* Welch’s independent-samples t-test for approximately symmetric distributions without severe outlier problems.
* Mann–Whitney U for strongly skewed or outlier-sensitive distributions.

Do not choose the test using only the Shapiro–Wilk p-value.

Consider:

* Distribution plots.
* Q–Q plots.
* Skewness.
* Extreme outliers.
* Group sizes.
* Variance differences.

### Effect Sizes

Report:

* Hedges’ (g) for Welch’s t-test.
* Rank-biserial correlation or Cliff’s delta for Mann–Whitney U.

Define the effect-size direction as:

```text
high_spindle_rate - low_spindle_rate
```

### Multiple-Comparison Correction

Apply Benjamini–Hochberg FDR correction across the tested N2–C3 infraslow parameters.

Save:

```text
p_value
q_value
```

Do not call a result statistically significant unless the FDR-adjusted q-value is below the configured alpha.

Save:

```text
N2_C3_infraslow_group_comparison.csv
```

Recommended columns:

```text
sleep_stage
channel
parameter
low_group_n
high_group_n
low_mean
low_sd
low_median
low_q1
low_q3
high_mean
high_sd
high_median
high_q1
high_q3
test
statistic
p_value
q_value
effect_size_name
effect_size
significant_fdr
```

---

## Step 6: Reproduce the Existing Infraslow Comparison Plot

This is a required part of the task.

Use the plotting logic from:

```text
infraslow/src/demo_infraslow_yasa_compare.py
```

to compare:

```text
N2–C3 low_spindle_rate
versus
N2–C3 high_spindle_rate
```

Do not create only a basic two-line mean spectrum plot.

The output must preserve the scientifically relevant visual components used in the existing comparison plot.

### Required Group-Level Comparison

Create a group comparison plot with:

* Low-spindle-rate N2–C3 spectrum.
* High-spindle-rate N2–C3 spectrum.
* The same corrected spectrum definition used in the existing script.
* The same frequency range.
* The same frequency-axis scaling.
* The same power transformation.
* The same baseline correction.
* The same fitting approach.
* Group uncertainty calculated across subjects.
* Group sample sizes.

Calculate the group curve in this order:

1. Prepare the valid spectrum for each subject.
2. Align spectra to a common frequency grid if required.
3. Keep only N2–C3 observations.
4. Average within subject if duplicate segments exist.
5. Calculate the group mean across subjects.
6. Calculate SEM or a 95% confidence interval across subjects.

Do not pool epochs, frequency samples, or repeated segments as independent subjects.

### Match `demo_infraslow_yasa_compare.py`

Reproduce the comparison elements used by the existing script, where available:

* Original or uncorrected spectrum.
* Background or baseline spectrum.
* Corrected infraslow spectrum.
* Bi-Gaussian fitted curve.
* Peak-frequency vertical line or marker.
* Peak-period annotation.
* Left and right bandwidth representation.
* AUC region or annotation.
* Chromatogram peak-area region or annotation.
* Spindle-rate summary.
* Legend.
* Labels.
* Titles.
* Axis ranges.
* Figure layout.

Adapt the plot from individual or condition comparison to the new group comparison.

Do not remove important plot components merely to simplify implementation.

### Required Figures

Create at least:

```text
N2_C3_infraslow_group_compare.png
N2_C3_infraslow_group_compare.pdf
```

This should be the main comparison figure following the style and logic of:

```text
infraslow/src/demo_infraslow_yasa_compare.py
```

Also create a clean group-level spectrum figure:

```text
N2_C3_infraslow_power_by_spindle_group.png
N2_C3_infraslow_power_by_spindle_group.pdf
```

The clean figure should show:

* Low-spindle-rate mean spectrum.
* High-spindle-rate mean spectrum.
* SEM or 95% confidence intervals.
* N2 and C3 clearly stated in the title.
* Group sample sizes.
* Infraslow frequency range.

### Parameter Comparison Plots

Create comparison plots for:

```text
peak_freq_hz
peak_period_s
bandwidth_hz
auc
chromatogram_peak_area
```

Use box plots or violin plots with individual subject points.

Save:

```text
N2_C3_parameter_group_comparisons.png
N2_C3_parameter_group_comparisons.pdf
```

Annotate the figures with FDR-adjusted q-values where appropriate.

Do not yet implement frequency-wise statistical testing or cluster-based permutation tests.

---

## Code Organization

Create:

```text
infraslow/src/group_analysis.py
```

The script should orchestrate:

1. Loading results.
2. Filtering to N2 and C3.
3. Validating results.
4. Assigning spindle-rate groups.
5. Running statistical comparisons.
6. Reproducing the existing comparison plot.
7. Creating group-level figures.
8. Saving outputs.

Place reusable statistical functions under:

```text
infraslow/src/infraslow/stats/
```

Suggested files:

```text
infraslow/src/infraslow/stats/
├── __init__.py
├── group_assignment.py
├── group_comparison.py
└── effect_sizes.py
```

Place reusable plotting functions under:

```text
infraslow/src/infraslow/viz/
```

For example:

```text
infraslow/src/infraslow/viz/group_analysis.py
```

Whenever possible, reuse or extend plotting functions already used by:

```text
infraslow/src/demo_infraslow_yasa_compare.py
```

Do not duplicate plotting logic in multiple files.

---

## Output Structure

Use a configurable output directory with the default:

```text
infraslow/results/group_analysis/N2_C3/
```

Save:

```text
N2_C3/
├── validated_N2_C3_subject_results.csv
├── invalid_or_excluded_N2_C3_subjects.csv
├── N2_C3_subject_group_assignments.csv
├── N2_C3_infraslow_group_comparison.csv
├── validation_report_N2_C3.txt
├── grouping_report_N2_C3.txt
├── N2_C3_spindle_rate_group_distribution.png
├── N2_C3_parameter_group_comparisons.png
├── N2_C3_parameter_group_comparisons.pdf
├── N2_C3_infraslow_group_compare.png
├── N2_C3_infraslow_group_compare.pdf
├── N2_C3_infraslow_power_by_spindle_group.png
└── N2_C3_infraslow_power_by_spindle_group.pdf
```

---

## Command-Line Interface

The script should run approximately as:

```bash
python infraslow/src/group_analysis.py \
    --results <path-to-subject-level-results> \
    --sleep-stage N2 \
    --channel C3 \
    --output-dir infraslow/results/group_analysis/N2_C3 \
    --group-probability-threshold 0.70 \
    --fdr-alpha 0.05 \
    --random-state 42
```

Add appropriate arguments based on the existing input format:

```text
--results
--summary-results
--spectrum-results
--sleep-stage
--channel
--output-dir
--subject-id-column
--group-probability-threshold
--fdr-alpha
--random-state
--overwrite
--log-level
```

Although `--sleep-stage` and `--channel` may be configurable, the default values must be:

```text
--sleep-stage N2
--channel C3
```

Do not require metadata-related arguments.

---

## Code Quality Requirements

* Use `pathlib.Path`.
* Use type hints.
* Add clear docstrings.
* Use the `logging` module.
* Use a fixed random seed.
* Keep functions small and testable.
* Handle missing files with clear errors.
* Handle missing columns with clear errors.
* Do not silently discard invalid subjects.
* Do not use another state or channel as a fallback.
* Follow the existing repository structure and coding style.
* Reuse existing scientific and plotting functions.
* Do not modify unrelated files.
* Do not change the original spindle or infraslow calculations.

---

## Scope Limit

For this task, do not implement:

* Metadata loading.
* Age, gender, or BMI analysis.
* Sleep-architecture comparisons.
* ANCOVA.
* Adjusted regression.
* Continuous spindle-rate regression.
* Median-split sensitivity analysis.
* Cluster-based permutation testing.
* Frequency-wise significance testing.
* Other sleep states.
* Other EEG channels.

---

## Expected Final Response

After implementation, provide:

1. Files created or modified.
2. A brief explanation of the N2–C3 workflow.
3. The exact command needed to run it.
4. The expected input format.
5. A description of the generated output files.
6. Assumptions about N2 labeling and C3 channel naming.
7. Any limitations caused by the existing result format.
8. Confirmation that only N2 and C3 were analyzed.
9. Confirmation that no metadata were loaded.
10. Confirmation that the comparison plotting logic from `demo_infraslow_yasa_compare.py` was reused.
11. Confirmation that the original spindle and infraslow calculations were not changed.
