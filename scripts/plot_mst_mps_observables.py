import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def observables_path(system_size, field_tag, bond_dimension, shield):
    name = f"n{system_size:03d}_g{field_tag:02d}_c{bond_dimension:02d}_m{shield:02d}.json"
    current = Path("results/observables") / name
    for path in (
        current,
        Path("results/mst_mps/observables") / name,
        Path("results/mst_mps/raw") / name,
    ):
        if path.is_file():
            return path
    return current


def save_page(pdf, curves, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    for label, x, y in curves:
        ax.plot(x, y, ".", label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    ax.legend()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot 1D MST-MPS temperature-sweep results.")
    parser.add_argument("--system_size", type=int, default=48)
    parser.add_argument(
        "--transverse_field", type=float, default=0.5, help="Transverse field, a multiple of 0.1."
    )
    parser.add_argument("--bond_dimension", type=int, nargs="+", default=[8])
    parser.add_argument("--markov_shield", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    field = args.transverse_field
    scaled_field = 10 * field
    if not math.isfinite(scaled_field) or not math.isclose(
        scaled_field, round(scaled_field), rel_tol=2 * math.ulp(1.0)
    ):
        parser.error("--transverse_field must be a multiple of 0.1.")
    field_tag = round(scaled_field)

    runs = []
    for bond_dimension in dict.fromkeys(args.bond_dimension):
        for shield in dict.fromkeys(args.markov_shield):
            path = observables_path(args.system_size, field_tag, bond_dimension, shield)
            if not path.is_file():
                print(f"{path} does not exist.")
                continue
            with path.open(encoding="utf-8") as file:
                data = json.load(file)
            runs.append((f"chi={bond_dimension}, m={shield}", data))

    if not runs:
        parser.error("no observables JSON files found for the requested parameters.")

    entropy_curves = []
    for label, data in runs:
        temperatures = np.asarray(data["temperature"])
        positive = temperatures > 0
        if positive.any():
            energy = np.asarray(data["energy"])[positive]
            free_energy = np.asarray(data["markov_free_energy"])[positive]
            entropy = (energy - free_energy) / temperatures[positive]
            entropy_curves.append((label, energy, entropy))

    title = f"N={args.system_size}, g={field}"
    pdf_path = Path("results/plot") / f"n{args.system_size:02d}_g{field_tag:02d}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for key, ylabel in (
            ("energy", "Energy per site"),
            ("markov_free_energy", "Markov free energy per site"),
            ("magnetization_x", "X magnetization per site"),
            ("magnetization_z", "Z magnetization per site"),
        ):
            curves = ((label, data["temperature"], data[key]) for label, data in runs)
            save_page(pdf, curves, "Temperature", ylabel, title)

        if entropy_curves:
            save_page(pdf, entropy_curves, "Energy per site", "Markov entropy per site", title)

    print(f"PDF file is saved to {pdf_path}.")


if __name__ == "__main__":
    main()
