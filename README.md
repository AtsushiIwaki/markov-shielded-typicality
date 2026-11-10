# Markov-shielded typicality for 1D spin chains

This repository contains the Markov-shielded typical matrix product state
(MST-MPS) implementation for the one-dimensional transverse-field Ising model
with open boundaries. The method and numerical demonstrations are described in
[*Markov-Shielded Typicality for Thermal Quantum States*](https://arxiv.org/abs/2611.10593).

The Julia implementation uses [ITensors.jl](https://github.com/ITensor/ITensors.jl)
and [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) for tensor and MPS
operations and [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) for
single-site tensor optimization with BFGS.

The shared Julia routines are in `src/`. The temperature sweep, entanglement
analysis, and plotting scripts are in `scripts/`.

## Setup

Run the following commands from the repository root. The examples use Julia
1.12.6 and Python 3.12. Python is needed for plotting.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model and parameters

The transverse-field Ising Hamiltonian on an open chain with unit Ising
coupling is

$$
H = -\sum_{i=1}^{N-1} \sigma_i^z \sigma_{i+1}^z
    - g \sum_{i=1}^{N} \sigma_i^x.
$$

Here $\sigma_i^x$ and $\sigma_i^z$ are Pauli operators on physical site $i$.

| Argument | Symbol | Meaning |
| --- | --- | --- |
| `--system_size` | $N$ | Number of physical spins in the chain. |
| `--transverse_field` | $g$ | Transverse field in units of the Ising coupling. Use a multiple of 0.1. |
| `--bond_dimension` | $\chi$ | Target MPS bond dimension. Use a positive power of two. |
| `--markov_shield` | $m$ | Maximum number of preceding physical sites in the Markov shield for each site. Use `0` for an empty shield. |
| `--temperature_interval` | $\Delta T$ | Temperature step. Use a positive multiple of 0.1. |
| `--maximum_temperature` | $T_{\max}$ | Upper limit of the temperature sweep, which starts at $T=0$. |

## Example workflow

The example uses a six-site chain at `T = 0` and `T = 0.5`.
The small iteration count keeps the example short.

### 1. Run the temperature sweep and plot observables

The sweep writes one observables JSON and saves an MPS checkpoint for each
temperature. The plotting command reads the observables JSON.

```sh
julia --project=. scripts/run_mst_mps.jl \
  --system_size 6 --transverse_field 0.5 --bond_dimension 2 --markov_shield 1 \
  --temperature_interval 0.5 --maximum_temperature 0.5 \
  --num_sweeps 1 --num_iterations 2 --output_format json

python scripts/plot_mst_mps_observables.py \
  --system_size 6 --transverse_field 0.5 --bond_dimension 2 --markov_shield 1
```

### 2. Compute and plot entanglement from the saved MPS

Pass the same model and temperature arguments to select the checkpoints from
step 1. The analysis requires a saved state at each selected temperature. The
plotting command compares temperatures for the same $N$, $g$, $\chi$, and $m$.

```sh
julia --project=. scripts/compute_mps_entanglement.jl \
  --system_size 6 --transverse_field 0.5 --bond_dimension 2 --markov_shield 1 \
  --temperature_interval 0.5 --maximum_temperature 0.5

python scripts/plot_mps_entanglement.py \
  results/entanglement/n006_g05_c02_m01_t00.json \
  results/entanglement/n006_g05_c02_m01_t05.json
```

## Generated files

The full example writes these files under `results/`, which is ignored by Git.

```text
results/
├── observables/
│   └── n006_g05_c02_m01.json
├── mps/
│   ├── n006_g05_c02_m01_t00.h5
│   └── n006_g05_c02_m01_t05.h5
├── entanglement/
│   ├── n006_g05_c02_m01_t00.json
│   └── n006_g05_c02_m01_t05.json
└── plot/
    ├── n06_g05.pdf
    └── entanglement_n006_g05_c02_m01.pdf
```

The observables JSON contains temperature and the per-site energy, Markov free
energy, and transverse and longitudinal magnetizations. The observables PDF
also plots the Markov entropy per site as `(E - F_M) / T` at positive temperatures.

Each HDF5 checkpoint stores the MPS at one temperature. Its entanglement JSON
records the entropies plotted on the two pages of the entanglement PDF. The first
page shows the entropy as a function of centered block size, using the leftmost
block when two are equally centered. The second shows the entropy across each
cut. At cut position $i$, the left partition contains the left auxiliary sites
and the first $i$ physical sites.

## Reusing saved MPS states

For each temperature in the sweep, the Julia driver looks for an HDF5 state
whose filename matches the system size (`--system_size`), transverse field
(`--transverse_field`), bond dimension (`--bond_dimension`), Markov shield size
(`--markov_shield`), and temperature. It loads matching states directly and
computes states for temperatures without a matching file.

Matching states are reused even when `--num_sweeps`, `--num_iterations`, or
`--free_energy_tolerance` changes. To recompute all states with different
optimization settings, move the existing `results/` directory aside before
running the calculation again.

## Citation

If you use this code in your research, please cite Atsushi Iwaki and Chisa
Hotta, *Markov-Shielded Typicality for Thermal Quantum States* (2026).
[arXiv:2611.10593](https://arxiv.org/abs/2611.10593)

```bibtex
@misc{iwaki2026markovshielded,
  author        = {Iwaki, Atsushi and Hotta, Chisa},
  title         = {{Markov-Shielded Typicality for Thermal Quantum States}},
  year          = {2026},
  eprint        = {2611.10593},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.stat-mech},
  url           = {https://arxiv.org/abs/2611.10593}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

This software is released under the [MIT License](LICENSE).
