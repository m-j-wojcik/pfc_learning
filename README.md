# Learning Shapes Neural Geometry in the Prefrontal Cortex

This repository contains the analysis code for the paper:

**"Learning shapes neural geometry in the prefrontal cortex"**  
Michał J. Wójcik, Jake P. Stroud, Dante Wasmuht, Makoto Kusunoki, Mikiko Kadohisa, Mark J. Buckley, Rui Ponte Costa, Nicholas E. Myers, Laurence T. Hunt, John Duncan & Mark G. Stokes

Published in: [to be updated]

## Overview

This project investigates how neural representations in the primate prefrontal cortex (PFC) evolve during learning of an exclusive-or (XOR) rule. We demonstrate that PFC representations progress from high-dimensional, nonlinear, and randomly mixed geometries to low-dimensional, rule-selective formats as learning proceeds. Upon generalizing the learned rule to novel stimuli, these representations further evolve into abstract, stimulus-invariant geometries.

### Key Findings

- Neural representations transition from **high-dimensional** to **low-dimensional** across learning stages
- Early learning uses randomly mixed selectivity patterns supporting exploration
- Late learning shows minimal, task-relevant representations supporting generalisation
- Cross-generalisation to novel stimulus sets reveals abstract, stimulus-invariant neural codes

## Data Requirements

### Download Data

The electrophysiology data used in this analysis can be downloaded from Dryad:  
**https://datadryad.org/stash/dataset/doi:10.5061/dryad.c2fqz61kb**

### Data Files

After downloading, place the following files in the `./data/` directory:

**Required files:**
- `*_Spikes_preprocessed.npy` - Neural spike data (preprocessed)
- `*_meta.npy` - Metadata files containing trial information
- `*_cell_loc.csv` - Cell location information (anatomical coordinates)
- `*Event_re.mat` - Behavioral event data (already included in the repository)

**Data structure:**
```
./data/
├── Wom20200910_Spikes_preprocessed.npy
├── Wom20200910_meta.npy
├── Wom20200910_cell_loc.csv
├── Wom20200910Event_re.mat
├── Wom20200911_Spikes_preprocessed.npy
├── Wom20200911_meta.npy
├── ... (additional Womble sessions)
├── Wil20201020_Spikes_preprocessed.npy
├── Wil20201020_meta.npy
├── Wil20201020_cell_loc.csv
├── Wil20201020Event_re.mat
└── ... (additional Wilfred sessions)
```

Note: Files follow the naming convention `{SubjectPrefix}{Date}_{FileType}` where:
- Subject prefixes: `Wom` (Womble), `Wil` (Wilfred)
- Dates: YYYYMMDD format
- Multiple sessions per subject spanning the learning period

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Scikit-learn
- pandas
- PyYAML
- seaborn
- tqdm
- MNE-Python (for `mne.decoding`)
- joblib
- numba

### Setup

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/pfc_learning.git
cd pfc_learning
```

2. Install required packages:
```bash
pip install numpy scipy matplotlib scikit-learn pandas pyyaml seaborn tqdm mne joblib numba
```

3. Download the data from Dryad (see Data Requirements above) and place in `./data/` directory

## Project Structure

```
.
├── README.md                 # This file
├── config.yml                # Configuration file for analysis parameters
├── fun_lib.py                # Library of plotting and utility functions
├── figure_1.py               # Analysis and plotting for Figure 1
├── figure_2.py               # Analysis and plotting for Figure 2
├── figure_3.py               # Analysis and plotting for Figure 3
├── figure_4.py               # Analysis and plotting for Figure 4
├── supp_fig_2.py            # Supplementary Figure 2
├── supp_fig_3.py            # Supplementary Figure 3
├── supp_fig_4.py            # Supplementary Figure 4 (requires figure_2.py)
├── supp_fig_6.py            # Supplementary Figure 6 
├── supp_fig_7.py            # Supplementary Figure 7 (requires figure_4.py)
├── supp_fig_9.py            # Supplementary Figure 9
├── data/                     # Data directory (populate with downloads)
├── processed_data/           # Cached analysis results (auto-created)
└── figures/                  # Generated figures (auto-created)
```

## Usage

### Configuration

The `config.yml` file controls all analysis parameters. Key sections include:

**Analysis Parameters:**
```yaml
ANALYSIS_PARAMS:
  OBSERVE_OR_RUN_PREPROCESSING: 'observe'  # 'run' or 'observe'; 'run': generates the firing rate files
  OBSERVE_OR_RUN: 'observe'                # 'run' to regenerate, 'observe' to load cached processed files
  N_STAGES: 4                              # Number of learning stages
  N_WINDOWS: 3                             # Number of windows for sliding window analysis
  TIME_WINDOW_EXP1_COL_LOCK: [50, 150]    # Color-locked analysis window (ms)
  TIME_WINDOW_EXP1_SHAPE_LOCK: [100, 150] # Shape-locked analysis window (ms)
  TIME_WINDOW_EXP2_COL_LOCK: [50, 100]    # Experiment 2 color window
  TIME_WINDOW_EXP2_SHAPE_LOCK: [100, 150] # Experiment 2 shape window
  N_REPS: 1000                             # Number of permutations for statistical tests
```

**Paths:**
```yaml
PATHS:
  output_path: './processed_data/'         # Cached analysis outputs
  out_template_figures: './figures/'       # Generated figures
  in_template_beh: './data/{0}Event_re.mat'
  out_template_spks: './data/{0}_Spikes_preprocessed.npy'
  out_template_meta: './data/{0}_meta.npy'
  out_template_loc: './data/{0}_cell_loc.csv'
```

**Session Names:**
The config file includes all session identifiers for both subjects across Experiments 1 and 2 (e.g., `Wom20200910`, `Wil20201020`, etc.)

### Running Analyses

#### Main Figures

Each main figure script can be run independently to regenerate the corresponding figure from the paper:

```bash
# Generate main figures
python figure_1.py
python figure_2.py
python figure_3.py
python figure_4.py
```

#### Supplementary Figures

**Important:** Supplementary figure scripts depend on running the main figure scripts first, as they use cached outputs from the main analyses:

```bash
# After running the corresponding main figures, generate supplementary figures:
python supp_fig_2.py   
python supp_fig_3.py   
python supp_fig_4.py   # Requires: figure_2.py outputs
python supp_fig_6.py   
python supp_fig_7.py   # Requires: figure_4.py outputs
python supp_fig_9.py   
```

**Workflow recommendation:**
1. First, run all main figure scripts (figure_1.py through figure_4.py)
2. Then run supplementary figure scripts as needed

### Run/Observe Mode

The analysis pipeline has two stages, each with its own run/observe control in `config.yml`:

#### Stage 1: Preprocessing (`OBSERVE_OR_RUN_PREPROCESSING`)
- **Run mode** (`'run'`): Generates downsampled, filtered firing rate files and groups sessions into learning stages
- **Observe mode** (`'observe'`): Uses previously generated preprocessed files

#### Stage 2: Analysis (`OBSERVE_OR_RUN`)
- **Run mode** (`'run'`): Runs analyses on the preprocessed files and caches results in `./processed_data/`
- **Observe mode** (`'observe'`): Loads previously cached analysis results for faster figure generation

#### Workflow Guide

**To regenerate the full analysis from scratch:**
```yaml
ANALYSIS_PARAMS:
  OBSERVE_OR_RUN_PREPROCESSING: 'run'  # Generate firing rate files and stage groupings
  OBSERVE_OR_RUN: 'run'                # Run all analyses on preprocessed data
```

**To re-run analyses with different parameters (but same preprocessing):**
```yaml
ANALYSIS_PARAMS:
  OBSERVE_OR_RUN_PREPROCESSING: 'observe'  # Use existing preprocessed files
  OBSERVE_OR_RUN: 'run'                    # Regenerate analyses with new parameters
```

**For fastest figure generation (loading all cached results):**
```yaml
ANALYSIS_PARAMS:
  OBSERVE_OR_RUN_PREPROCESSING: 'observe'  # Use existing preprocessed files
  OBSERVE_OR_RUN: 'observe'                # Load all cached analysis results
```

⚠️ **Important:** When changing analysis parameters (time windows, number of stages, etc.):
- If parameters affect preprocessing (e.g., `N_STAGES`), set both to `'run'`
- If parameters only affect analysis (e.g., specific time windows for decoding), you can keep preprocessing as `'observe'` and only set analysis to `'run'`
- Otherwise, the analysis will use stale cached results that don't reflect your parameter changes

### Output

- **Generated figures** are saved to `./figures/` directory
- **Cached analysis results** are stored in `./processed_data/` directory to speed up subsequent runs

## Analysis Methods

### Key Analyses Implemented

1. **Linear SVM Decoding** - Time-resolved decoding of task variables (color, shape, XOR) using support vector machines

2. **Cross-Generalisation Analysis** - Tests how well decoders trained on one condition generalise to other conditions, revealing the abstract structure of neural representations

3. **Shattering Dimensionality** - Measures the effective dimensionality by testing all possible binary dichotomies in the task space

4. **Selectivity Space Analysis** - Multiple linear regression to characterize single-neuron selectivity patterns and compare to theoretical models (random mixed vs. minimal selectivity)

5. **Cross-Stimulus Set Generalisation** - Examines whether learned representations transfer to novel stimulus sets (Experiment 2)

## Experiments

### Experiment 1: XOR Learning
Animals learned an XOR rule from scratch where the nonlinear combination of color and shape predicted reward. Neural recordings tracked learning dynamics across multiple sessions.

### Experiment 2: Rule Generalization  
After learning the XOR rule in Experiment 1, animals were tested with a novel stimulus set to assess whether the learned rule generalized to new sensory inputs.

## Citation

If you use this code or data, please cite:

```bibtex
@article{wojcik2024learning,
  title={Learning shapes neural geometry in the prefrontal cortex},
  author={W{\'o}jcik, Micha{\l} J and Stroud, Jake P and Wasmuht, Dante and Kusunoki, Makoto and Kadohisa, Mikiko and Buckley, Mark J and Costa, Rui Ponte and Myers, Nicholas E and Hunt, Laurence T and Duncan, John and Stokes, Mark G},
  journal={[BioRxiv]},
  year={2024}
}
```

**Data citation:**  
Wójcik, Michał; Stroud, Jake; Wasmuht, Dante et al. (2024). Electrophysiological recordings of prefrontal activity over learning in non-human primates [Dataset]. Dryad. https://doi.org/10.5061/dryad.c2fqz61kb

## License

CC-BY

## Acknowledgments

This work was funded by the Wellcome Trust, the Medical Research Council UK, the Biotechnology and Biological Sciences Research Council, the Clarendon Fund and Saven European Scholarship, and the James S. McDonnell Foundation. 

We thank all colleagues for feedback and support. Special acknowledgment to Mark Stokes, who passed away on January 13, 2023. His brilliant mind and insights were instrumental to this work.

## Additional Resources

- **Paper**: [will be updated]
- **Data**: https://datadryad.org/stash/dataset/doi:10.5061/dryad.c2fqz61kb
