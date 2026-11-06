import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

from plot_mst_mps_observables import save_page


def main():
    parser = argparse.ArgumentParser(
        description="Plot entanglement entropy from MST-MPS analysis JSON."
    )
    parser.add_argument(
        "inputs", type=Path, nargs="+", help="JSON files from compute_mps_entanglement.jl."
    )
    args = parser.parse_args()

    runs = []
    for path in args.inputs:
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
        with path.open(encoding="utf-8") as file:
            runs.append(json.load(file))

    parameter_keys = ("system_size", "transverse_field", "bond_dimension", "markov_shield")
    parameters = tuple(runs[0][key] for key in parameter_keys)
    system_size, field, bond_dimension, shield = parameters
    if any(tuple(run[key] for key in parameter_keys) != parameters for run in runs[1:]):
        parser.error(
            "inputs must have the same system size, transverse field, bond dimension, and shield size."
        )

    runs.sort(key=lambda run: run["temperature"])
    title = f"N={system_size}, g={field:g}, chi={bond_dimension}, m={shield}"
    field_tag = round(10 * field)
    stem = f"n{system_size:03d}_g{field_tag:02d}_c{bond_dimension:02d}_m{shield:02d}"
    output = Path("results/plot") / f"entanglement_{stem}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for x_key, y_key, xlabel in (
            ("center_site", "center_ee", "Subsystem size"),
            ("edge_pos", "edge_ee", "Cut position"),
        ):
            curves = ((f"T={run['temperature']:g}", run[x_key], run[y_key]) for run in runs)
            save_page(pdf, curves, xlabel, "Entanglement entropy", title)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
