# Neural Decoding and Representation Analysis Pipeline

Code for analyzing neural encoding during learning tasks, with a focus on decoding task variables and analyzing representational changes across learning stages.

## Key Files

- `fun_lib.py` - Core analysis functions library
- `figure_3.py`, `figure_4and5.py`, `supp_fig_5.py`, `supp_fig_7.py` - Scripts for generating paper figures
- `config.yml` - Configuration parameters and settings

## Main Functions

### Data Processing & Analysis
- `get_data_stages()` - Load and preprocess neural recording data
- `decode()` - Perform decoding analysis on neural data
- `compute_lifetime_sparseness()` - Calculate sparseness of neural representations
- `run_time_resolved_decoding()` - Time-resolved decoding analysis
- `compute_f_rate_norm()` - Compute firing rate norms
- `get_betas_cross_val()` - Cross-validated regression analysis

### Visualization
- `plot_dec_xgen()` - Plot cross-generalization decoding results 
- `plot_distance()` - Plot distance metrics between conditions
- `plot_mean_and_ci_prop()` - Plot means with confidence intervals
- `plot_regression_sup()` - Plot regression analysis results

## Dependencies

- numpy
- pandas 
- matplotlib
- seaborn
- scipy
- scikit-learn
- tqdm
- yaml

## Data Requirements

The data used in this analysis can be downloaded from: https://datadryad.org/stash/dataset/doi:10.5061/dryad.c2fqz61kb

Input data files:
- Neural spike data (`*_Spikes_preprocessed.npy`)
- Metadata (`*_meta.npy`)
- Cell locations (`*_cell_loc.csv`)
- Behavioral data (`*Event_re.mat`)

## Usage

1. Configure parameters in `config.yml`
2. Run analysis scripts (`figure_*.py`) to generate figures
3. Use `fun_lib.py` functions for custom analyses

## Configuration

Key parameters in `config.yml`:
- `ANALYSIS_PARAMS`: Analysis settings (time windows, stages, repetitions)
- `ENCODING_EXP1/2`: Task variable encodings
- `PATHS`: Data and output file paths
- `SESSION_NAMES`: Experimental session identifiers
- `TRIGGER_CODES`: Behavioral event codes
