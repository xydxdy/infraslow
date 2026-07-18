# Recheck and Update the All-Metrics Infraslow Pipeline

Act as a senior Python developer and EEG/PSG signal-processing researcher with experience in:

* Python
* NumPy and pandas
* YASA spindle detection
* PSG and hypnogram processing
* Spectral analysis
* Multiprocessing
* Slurm
* Scientific pipeline validation

Please inspect, recheck, and update the following files:

* `src/run_all_metrics.py`
* `run_all_metrics.sbatch`

Also inspect any helper functions used by these files, especially code related to:

* PSG loading
* channel mapping
* hypnogram processing
* sleep-metric calculation
* spindle detection
* sleep-bout detection
* infraslow spectral calculation
* NPZ serialization
* multiprocessing

Reuse the existing project functions whenever appropriate.

Do not change unrelated scientific logic. If scientific logic must be changed, clearly explain the reason.

---

# 1. New Output Structure

Replace the existing output path:

```bash
$SCRATCH/results/all_metrics.csv
```

with the following output structure:

```text
$SCRATCH/results_v2/
├── metadata.csv
├── errors.csv
└── npz/
    ├── N2/
    ├── N3/
    └── NREM/
```

The pipeline must automatically create all missing directories.

The expected NPZ output paths are:

```bash
$SCRATCH/results_v2/npz/N2/{ID}.npz
$SCRATCH/results_v2/npz/N3/{ID}.npz
$SCRATCH/results_v2/npz/NREM/{ID}.npz
```

---

# 2. Metadata Output

Save subject-level demographic and sleep metadata to:

```bash
$SCRATCH/results_v2/metadata.csv
```

Keep the subject identifier used by the current project, such as:

```text
ID
```

or:

```text
subject_id
```

as the first column.

The remaining columns must be saved in this exact order:

```text
Age
Gender
BMI
TIB
SPT
WASO
TST
N1
N2
N3
REM
NREM
SOL
Lat_N1
Lat_N2
Lat_N3
Lat_REM
%N1
%N2
%N3
%REM
%NREM
SE
SME
```

Each subject must have only one row in `metadata.csv`.

Do not include spindle or infraslow values in `metadata.csv`. Those results must be saved in the NPZ files.

---

# 3. Overall Processing Flow

For each subject, run the following pipeline:

```text
Load PSG and sleep-stage annotations
    ↓
Calculate subject-level sleep metadata
    ↓
Resolve available EEG channels
    ↓
For each EEG channel:
    Detect NREM spindles using stages 1, 2, and 3
    ↓
    Construct N2, N3, and combined NREM bouts
    ↓
    Keep bouts with duration ≥ 200 seconds
    ↓
    Keep bouts containing at least one spindle
    ↓
    Calculate the infraslow spectrum for valid bouts
    ↓
Save one NPZ file per subject for N2, N3, and NREM
```

---

# 4. PSG Loading

For each subject:

1. Load the PSG recording.
2. Load the sleep-stage annotations or hypnogram.
3. Obtain the PSG sampling frequency.
4. Ensure the hypnogram and signal timelines are correctly aligned.
5. Calculate the required subject-level sleep metadata.
6. Process the following canonical EEG channels independently:

```python
CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2"]
```

Use the project’s existing channel-mapping logic to map canonical channel names to available PSG channel names.

For example, the canonical channel `F3` may map to recordings such as:

```text
F3M2
F3A2
F3-M2
EEG F3-A2
F3:M2
F3
```

Do not duplicate channel-mapping logic if an existing helper already handles this.

If a channel is unavailable:

* skip only that channel;
* continue processing the remaining channels;
* record the missing channel in the log or error output;
* do not fail the entire subject.

---

# 5. Spindle Detection

Run spindle detection independently for each available EEG channel.

Use YASA spindle detection or the existing project wrapper.

The spindle detector must initially include all NREM stages:

```python
include=(1, 2, 3)
```

The spindle detection results must retain at least:

```text
start
stop
peak
```

All spindle times must use the same unit.

Prefer:

```text
seconds from the beginning of the PSG
```

unless the current project consistently uses another documented time system.

Do not mix spindle detections between channels.

For example:

* `F3` results must contain only spindles detected from `F3`;
* `F4` results must contain only spindles detected from `F4`;
* and so on.

If no spindle is detected for one channel, save correctly typed empty arrays or omit that channel according to the selected NPZ convention.

Document the chosen convention.

---

# 6. Sleep-State Analyses

Create three separate analyses:

| Output | Included stages    |
| ------ | ------------------ |
| `N2`   | Stage 2 only       |
| `N3`   | Stage 3 only       |
| `NREM` | Stages 1, 2, and 3 |

The output folders must be:

```text
$SCRATCH/results_v2/npz/N2/
$SCRATCH/results_v2/npz/N3/
$SCRATCH/results_v2/npz/NREM/
```

Do not mix results across these analyses.

---

# 7. Bout Construction

For each channel and each sleep-state analysis, identify continuous bouts.

## N2 bouts

An N2 bout contains continuous Stage 2 sleep only.

An N2 bout ends when the sleep stage changes from Stage 2 to any other stage.

## N3 bouts

An N3 bout contains continuous Stage 3 sleep only.

An N3 bout ends when the sleep stage changes from Stage 3 to any other stage.

## NREM bouts

An NREM bout contains continuous epochs from any of these stages:

```python
{1, 2, 3}
```

Transitions between N1, N2, and N3 must not split the combined NREM bout.

For example:

```text
N1 → N2 → N3 → N2
```

must be treated as one continuous NREM bout.

A combined NREM bout ends when interrupted by:

* Wake
* REM
* an invalid sleep stage
* missing annotation
* a recording gap

---

# 8. Bout Inclusion Criteria

Keep a bout only when both conditions are satisfied:

```text
bout duration ≥ 200 seconds
```

and:

```text
number of spindles in the bout ≥ 1
```

For every retained bout, save:

```text
start
stop
n_spindles
```

The bout duration must be calculated as:

```python
duration = stop - start
```

Use one consistent time convention for spindle and bout values.

---

# 9. Assigning Spindles to Bouts

Use the spindle peak time to determine whether a spindle belongs to a bout.

Use this interval rule:

```python
bout_start <= spindle_peak < bout_stop
```

Apply this rule consistently for:

* N2
* N3
* NREM
* all EEG channels

Do not assign one spindle to multiple bouts.

The value saved as `n_spindles` must exactly match the number of spindle peaks satisfying the bout interval condition.

A detected spindle may start before a bout boundary or stop after a bout boundary. Its assignment must still be determined by its `peak` value.

---

# 10. Infraslow Spectrum Calculation

For every retained bout:

1. Select the spindle events assigned to the bout.
2. Construct the spindle-event time series using the existing project implementation.
3. Calculate the infraslow spectrum using the existing infraslow-processing functions.
4. Save or aggregate the corrected spectrum.

The final channel-level spectrum must save:

```text
freqs
corr_mean
```

Use the existing scientific implementation whenever possible.

Do not rewrite the infraslow algorithm unnecessarily.

Before averaging bout-level spectra:

* verify that the frequency vectors are identical;
* verify that array lengths match;
* verify that values are finite where required.

Do not silently average spectra with incompatible frequency grids.

If the existing implementation already provides interpolation or frequency-grid alignment, reuse it.

If frequency grids differ and no existing method exists, implement a clearly documented interpolation to one common frequency grid.

Handle invalid bouts safely, including:

* insufficient spindle events;
* empty event sequences;
* failed spectral estimation;
* all-NaN results;
* infinite results;
* incompatible frequency vectors;
* zero-length spectra.

One invalid bout must not terminate processing for the entire channel or subject.

---

# 11. NPZ Output Files

Save one file for each subject and each state:

```bash
$SCRATCH/results_v2/npz/N2/{ID}.npz
$SCRATCH/results_v2/npz/N3/{ID}.npz
$SCRATCH/results_v2/npz/NREM/{ID}.npz
```

Each NPZ file must contain results independently for every available EEG channel.

The conceptual structure must be:

```text
F3
├── spectra
│   ├── freqs
│   └── corr_mean
├── spindles
│   ├── start
│   ├── stop
│   └── peak
└── bouts
    ├── start
    ├── stop
    └── n_spindles

F4
├── spectra
│   ├── freqs
│   └── corr_mean
├── spindles
│   ├── start
│   ├── stop
│   └── peak
└── bouts
    ├── start
    ├── stop
    └── n_spindles

C3
├── spectra
│   ├── freqs
│   └── corr_mean
├── spindles
│   ├── start
│   ├── stop
│   └── peak
└── bouts
    ├── start
    ├── stop
    └── n_spindles

C4
├── spectra
│   ├── freqs
│   └── corr_mean
├── spindles
│   ├── start
│   ├── stop
│   └── peak
└── bouts
    ├── start
    ├── stop
    └── n_spindles

O1
├── spectra
│   ├── freqs
│   └── corr_mean
├── spindles
│   ├── start
│   ├── stop
│   └── peak
└── bouts
    ├── start
    ├── stop
    └── n_spindles

O2
├── spectra
│   ├── freqs
│   └── corr_mean
├── spindles
│   ├── start
│   ├── stop
│   └── peak
└── bouts
    ├── start
    ├── stop
    └── n_spindles
```

---

# 12. Flattened NPZ Key Schema

Do not save nested Python dictionaries as object arrays.

The files must be loadable using:

```python
np.load(npz_path, allow_pickle=False)
```

Use flattened NPZ keys.

For `F3`, save:

```text
F3__spectra__freqs
F3__spectra__corr_mean
F3__spindles__start
F3__spindles__stop
F3__spindles__peak
F3__bouts__start
F3__bouts__stop
F3__bouts__n_spindles
```

Repeat the same structure for:

```text
F4
C3
C4
O1
O2
```

The complete expected key pattern is:

```text
{channel}__spectra__freqs
{channel}__spectra__corr_mean
{channel}__spindles__start
{channel}__spindles__stop
{channel}__spindles__peak
{channel}__bouts__start
{channel}__bouts__stop
{channel}__bouts__n_spindles
```

For example:

```text
F4__spectra__freqs
F4__spectra__corr_mean
F4__spindles__start
F4__spindles__stop
F4__spindles__peak
F4__bouts__start
F4__bouts__stop
F4__bouts__n_spindles
```

Save only numeric NumPy arrays where possible.

Recommended dtypes:

```python
freqs: float64
corr_mean: float64
spindle start: float64
spindle stop: float64
spindle peak: float64
bout start: float64
bout stop: float64
bout n_spindles: int64
```

Do not save pandas DataFrames, dictionaries, or arbitrary Python objects inside the NPZ files.

---

# 13. Spindles Saved in Each State File

Each state-specific NPZ file should contain the spindle results relevant to that state analysis.

Use the spindle peak and hypnogram to determine the spindle state.

For example:

## N2 file

```text
$SCRATCH/results_v2/npz/N2/{ID}.npz
```

Save only spindles whose peaks belong to retained N2 bouts.

## N3 file

```text
$SCRATCH/results_v2/npz/N3/{ID}.npz
```

Save only spindles whose peaks belong to retained N3 bouts.

## NREM file

```text
$SCRATCH/results_v2/npz/NREM/{ID}.npz
```

Save only spindles whose peaks belong to retained combined NREM bouts.

Do not save all detected spindles in every state file unless the existing downstream analysis explicitly requires that behavior.

The state-specific spindle arrays should correspond to the bouts and spectra in that NPZ file.

---

# 14. Missing and Empty Results

Choose and document one consistent rule.

The preferred rule is to save all expected channel keys, even if a channel is missing or has no valid result.

For a missing or empty channel, save correctly typed empty arrays:

```python
np.array([], dtype=np.float64)
```

for time and spectrum arrays, and:

```python
np.array([], dtype=np.int64)
```

for `n_spindles`.

For example:

```python
result["F3__spindles__start"] = np.array([], dtype=np.float64)
result["F3__spindles__stop"] = np.array([], dtype=np.float64)
result["F3__spindles__peak"] = np.array([], dtype=np.float64)

result["F3__bouts__start"] = np.array([], dtype=np.float64)
result["F3__bouts__stop"] = np.array([], dtype=np.float64)
result["F3__bouts__n_spindles"] = np.array([], dtype=np.int64)

result["F3__spectra__freqs"] = np.array([], dtype=np.float64)
result["F3__spectra__corr_mean"] = np.array([], dtype=np.float64)
```

This ensures a predictable schema for downstream analysis.

---

# 15. Metadata Multiprocessing Safety

If the script uses multiprocessing, workers must not write directly to the same `metadata.csv`.

Use this design:

1. Each worker processes one subject.
2. Each worker saves only its own NPZ files.
3. Each worker returns:

   * subject ID;
   * metadata row;
   * processing status;
   * warning information;
   * error information.
4. The parent process collects worker results.
5. The parent process writes `metadata.csv`.
6. The parent process writes `errors.csv`.

Do not allow multiple workers to append to the same CSV simultaneously.

Sort the final metadata table by subject ID when practical.

Prevent duplicate metadata rows during reruns.

---

# 16. Rerun and Resume Behavior

Implement safe rerun behavior.

Before skipping a subject-state NPZ file, validate that it:

* exists;
* can be loaded;
* contains the expected keys;
* has matching array lengths;
* is not corrupted.

Do not skip a file only because its path exists.

If a file is missing, incomplete, invalid, or corrupted, rebuild it.

The pipeline must clearly log whether each output was:

```text
created
validated and skipped
rebuilt
failed
```

Do not overwrite valid subject results unnecessarily.

---

# 17. Atomic File Writing

Prevent partially written output files.

For every NPZ output:

1. Save to a temporary path in the same directory.
2. Load and validate the temporary file.
3. Atomically rename the temporary file to the final path.

For example:

```text
12345.npz.tmp
```

then:

```text
12345.npz
```

Make sure the temporary filename works correctly with `np.savez` or `np.savez_compressed` and does not accidentally receive an additional `.npz` extension.

Use a similar safe write-and-replace strategy for:

* `metadata.csv`
* `errors.csv`

where practical.

Remove temporary files when saving or validation fails.

---

# 18. Error Output

Save processing errors to:

```bash
$SCRATCH/results_v2/errors.csv
```

Include useful fields such as:

```text
subject_id
state
channel
error_type
error_message
traceback
```

The pipeline must distinguish between:

* subject-level failure;
* channel-level failure;
* bout-level failure;
* spectrum-level failure;
* missing channel;
* missing annotation;
* corrupted existing output.

A channel-level error must not automatically stop processing other channels.

A state-level error must not automatically stop processing other states.

A subject-level failure must not stop processing other subjects.

---

# 19. Validation Requirements

Add a reusable NPZ validation function.

It must verify the following.

## File validation

* The file can be loaded using:

```python
np.load(npz_path, allow_pickle=False)
```

* No object arrays are present.
* The expected key naming convention is used.

## Spectrum validation

For every channel:

```text
len(freqs) == len(corr_mean)
```

When non-empty:

* `freqs` must be finite;
* `corr_mean` must contain valid finite values according to the project’s scientific requirements;
* frequencies should be ordered;
* frequencies should not contain invalid negative values unless scientifically intended.

## Spindle validation

For every channel:

```text
len(start) == len(stop) == len(peak)
```

Every spindle must satisfy:

```python
start <= peak <= stop
```

All spindle times must be finite.

## Bout validation

For every channel:

```text
len(start) == len(stop) == len(n_spindles)
```

Every bout must satisfy:

```python
stop > start
```

Every retained bout must satisfy:

```python
stop - start >= 200
```

Every retained bout must satisfy:

```python
n_spindles >= 1
```

## Spindle-to-bout validation

For each bout, calculate:

```python
assigned = (
    (spindle_peak >= bout_start)
    & (spindle_peak < bout_stop)
)
```

Then verify:

```python
assigned.sum() == n_spindles
```

A spindle peak must not be assigned to more than one bout.

---

# 20. Scientific Consistency Checks

Recheck the complete pipeline for the following potential issues:

* incorrect conversion between samples, epochs, and seconds;
* using epoch indices as seconds;
* incorrect hypnogram upsampling;
* signal and hypnogram length mismatch;
* incorrect PSG sampling frequency;
* spindle times using a different time origin from bout times;
* incorrect channel mapping;
* duplicated channel processing;
* mixing spindle events between channels;
* mixing N2, N3, and NREM outputs;
* splitting NREM bouts at N1-to-N2 or N2-to-N3 transitions;
* assigning spindles using inconsistent boundary rules;
* including bouts shorter than 200 seconds;
* including bouts with zero spindles;
* incorrect `n_spindles` values;
* averaging incompatible spectral frequency vectors;
* averaging NaN or infinite spectral values;
* overwriting another subject’s NPZ file;
* race conditions in CSV writing;
* memory leaks from loaded PSG projects;
* failing to release Luna or PSG resources;
* nested parallelism from NumPy or BLAS;
* duplicate metadata rows;
* partial output files caused by interrupted jobs;
* silently skipping failed channels or subjects.

Clearly report any bug or ambiguity found in the current implementation.

---

# 21. Update `src/run_all_metrics.py`

Update the Python script so it supports explicit command-line arguments.

A preferred interface is:

```bash
python -u src/run_all_metrics.py \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --metadata-input "${METADATA_INPUT}" \
    --output-root "${SCRATCH}/results_v2" \
    --metadata-output "${SCRATCH}/results_v2/metadata.csv" \
    --error-output "${SCRATCH}/results_v2/errors.csv"
```

Adapt the argument names if necessary to remain consistent with the existing code.

The script should support at least:

```text
--workers
--metadata-input
--output-root
--metadata-output
--error-output
```

Useful optional arguments may include:

```text
--subject-id
--limit
--overwrite
--validate-only
--dry-run
--log-level
```

Do not hard-code environment-specific paths inside the Python code.

Use:

* type hints;
* clear docstrings;
* structured functions;
* useful logging;
* robust exception handling;
* deterministic processing;
* minimal duplicated logic.

---

# 22. Update `run_all_metrics.sbatch`

Update the Slurm script to:

* activate the existing Conda environment;
* create required output and log directories;
* use `$SCRATCH/results_v2`;
* pass explicit arguments to `src/run_all_metrics.py`;
* use `SLURM_CPUS_PER_TASK` as the multiprocessing worker count;
* prevent BLAS libraries from spawning extra threads;
* quote Bash variables;
* terminate when a command fails.

Start the script with:

```bash
#!/bin/bash
```

and include:

```bash
set -euo pipefail
```

Use environment variables such as:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Activate the Conda environment correctly:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310
```

Use a structure similar to:

```bash
OUTPUT_ROOT="${SCRATCH}/results_v2"
METADATA_OUTPUT="${OUTPUT_ROOT}/metadata.csv"
ERROR_OUTPUT="${OUTPUT_ROOT}/errors.csv"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/npz/N2"
mkdir -p "${OUTPUT_ROOT}/npz/N3"
mkdir -p "${OUTPUT_ROOT}/npz/NREM"
mkdir -p logs

python -u src/run_all_metrics.py \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --metadata-input "${METADATA_INPUT}" \
    --output-root "${OUTPUT_ROOT}" \
    --metadata-output "${METADATA_OUTPUT}" \
    --error-output "${ERROR_OUTPUT}"
```

Use the existing metadata-input path from the current script unless it is incorrect.

Do not replace valid project-specific Slurm settings without a reason.

---

# 23. Resource Cleanup

Make sure PSG-related resources are released after every subject.

Use `try`, `except`, and `finally` where appropriate.

Clean up:

* Luna project objects;
* large EEG arrays;
* hypnogram arrays;
* temporary data;
* temporary files.

Do not rely only on garbage collection for external PSG resources.

The code must remain safe when processing many subjects in one worker process.

---

# 24. Logging

Add useful logging at these levels:

## Subject level

```text
Starting subject
Metadata calculated
Subject completed
Subject failed
```

## Channel level

```text
Channel resolved
Channel missing
Spindles detected
Channel failed
```

## State level

```text
N2 processing started
N3 processing started
NREM processing started
Valid bouts found
Spectrum calculated
NPZ saved
Existing NPZ validated and skipped
```

Avoid printing extremely large arrays or full DataFrames.

Include subject ID, channel, and state in relevant log messages.

---

# 25. Test and Dry-Run Procedure

Provide a test procedure before running the full Slurm job.

The test should process one or two subjects and verify:

1. `metadata.csv` is created.
2. Each subject has one metadata row.
3. The N2 NPZ file is created.
4. The N3 NPZ file is created.
5. The NREM NPZ file is created.
6. Files load with `allow_pickle=False`.
7. Expected channel keys exist.
8. Spindle arrays have matching lengths.
9. Bout arrays have matching lengths.
10. Every bout is at least 200 seconds.
11. Every bout has at least one spindle.
12. `n_spindles` matches the spindle-peak count.
13. Spectrum array lengths match.
14. Missing channels are handled correctly.
15. A rerun validates and skips complete files.
16. A corrupted test file is detected and rebuilt.

Provide example commands for:

```text
one-subject dry run
two-subject test
validation-only run
full Slurm submission
```

---

# 26. Example NPZ Loading Code

Provide an example similar to:

```python
from pathlib import Path

import numpy as np

subject_id = "12345"
npz_path = Path(
    f"/path/to/results_v2/npz/N2/{subject_id}.npz"
)

with np.load(npz_path, allow_pickle=False) as data:
    channel = "F3"

    freqs = data[f"{channel}__spectra__freqs"]
    corr_mean = data[f"{channel}__spectra__corr_mean"]

    spindle_start = data[f"{channel}__spindles__start"]
    spindle_stop = data[f"{channel}__spindles__stop"]
    spindle_peak = data[f"{channel}__spindles__peak"]

    bout_start = data[f"{channel}__bouts__start"]
    bout_stop = data[f"{channel}__bouts__stop"]
    bout_n_spindles = data[
        f"{channel}__bouts__n_spindles"
    ]

print("Spectrum:", freqs.shape, corr_mean.shape)
print("Spindles:", spindle_start.shape)
print("Bouts:", bout_start.shape)
```

Also show how to list all keys:

```python
with np.load(npz_path, allow_pickle=False) as data:
    print(data.files)
```

---

# 27. Required Deliverables

After reviewing the existing pipeline, provide:

1. The complete updated `src/run_all_metrics.py`.
2. The complete updated `run_all_metrics.sbatch`.
3. Complete updates to any required helper modules.
4. A concise explanation of the final processing pipeline.
5. The final metadata-column schema.
6. The final NPZ key schema.
7. An example for loading one subject’s NPZ file.
8. A reusable NPZ validation function.
9. A one- or two-subject dry-run procedure.
10. A summary of bugs, inconsistencies, and ambiguous behavior found in the old implementation.
11. A list of any scientific-logic changes that were necessary.
12. Confirmation that NPZ files load using:

```python
np.load(path, allow_pickle=False)
```

Do not provide only pseudocode or partial code.

Return production-ready code that can be directly applied to the repository.

Preserve the existing scientific processing logic unless a modification is required to satisfy these requirements. Clearly explain every meaningful change.
