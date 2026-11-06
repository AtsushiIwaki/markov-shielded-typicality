using ArgParse, HDF5, ITensorMPS, JSON3, Printf

include("../src/markov_free_energy.jl")

function is_tenth_grid(value)
    scaled = 10 * value
    nearest = round(scaled)
    return typemin(Int) <= nearest <= typemax(Int) &&
        isapprox(scaled, nearest; rtol = 2eps(Float64))
end

function parse_commandline()
    settings = ArgParseSettings(description = "Compute entanglement entropy from saved MST-MPS states.")
    @add_arg_table! settings begin
        "--system_size"
        help = "Number of physical spins."
        arg_type = Int
        default = 16
        range_tester = value -> value > 0
        "--transverse_field"
        help = "Transverse field, a multiple of 0.1."
        arg_type = Float64
        default = 1.5
        range_tester = is_tenth_grid
        "--bond_dimension"
        help = "MPS bond dimension, a positive power of two."
        arg_type = Int
        default = 8
        range_tester = ispow2
        "--markov_shield"
        help = "Markov shield size."
        arg_type = Int
        default = 5
        "--temperature_interval"
        help = "Temperature interval, a positive multiple of 0.1."
        arg_type = Float64
        default = 0.1
        range_tester = value -> value > 0 && is_tenth_grid(value)
        "--maximum_temperature"
        help = "Upper limit of the sweep from T=0."
        arg_type = Float64
        default = 0.9
        range_tester = value -> isfinite(value) && value >= 0
    end
    return parse_args(settings)
end

function checkpoint_name(N, field_tag, chi, m, temperature)
    return @sprintf(
        "n%0.3d_g%0.2d_c%0.2d_m%0.2d_t%0.2d.h5",
        N, field_tag, chi, m, round(Int, 10 * temperature)
    )
end

function analyze_checkpoint(path, N, field_tag, chi, m, temperature)
    Naux = trailing_zeros(chi)

    psi = h5open(path, "r") do file
        read(file, "psi", MPS)
    end
    if length(psi) != N + 2 * Naux
        throw(ArgumentError("MPS length in $path does not match the system size and bond dimension"))
    end

    positions = 0:N
    edge_ee = [
        cut == 0 ? 0.0 : entanglement_entropy(psi, 1:cut) for cut in Naux:(Naux + N)
    ]
    center_ee = zeros(N + 1)
    for n in 1:N
        first = Naux + (N - n) ÷ 2 + 1
        center_ee[n + 1] = entanglement_entropy(psi, first:(first + n - 1))
    end

    return Dict(
        "system_size" => N,
        "transverse_field" => field_tag / 10,
        "bond_dimension" => chi,
        "markov_shield" => m,
        "temperature" => round(Int, 10 * temperature) / 10,
        "center_site" => positions,
        "center_ee" => center_ee,
        "edge_pos" => positions,
        "edge_ee" => edge_ee,
    )
end

function main()
    args = parse_commandline()
    N = args["system_size"]
    field_tag = round(Int, 10 * args["transverse_field"])
    chi = args["bond_dimension"]
    m = args["markov_shield"]
    temperatures = 0.0:args["temperature_interval"]:args["maximum_temperature"]

    mps_dir = "results/mps"
    legacy_mps_dir = "results/mst_mps/mps"
    ground_name = checkpoint_name(N, field_tag, chi, m, 0.0)
    reuse_legacy_mps = !isfile(joinpath(mps_dir, ground_name)) &&
        isfile(joinpath(legacy_mps_dir, ground_name))

    checkpoints = map(temperatures) do T
        name = checkpoint_name(N, field_tag, chi, m, T)
        path = joinpath(mps_dir, name)
        if reuse_legacy_mps && !isfile(path)
            path = joinpath(legacy_mps_dir, name)
        end
        if !isfile(path)
            throw(ArgumentError("missing MPS checkpoint for T=$T: $path"))
        end
        return path
    end

    output_dir = "results/entanglement"
    for (T, checkpoint) in zip(temperatures, checkpoints)
        data = analyze_checkpoint(checkpoint, N, field_tag, chi, m, T)
        stem = splitext(basename(checkpoint))[1]
        output_path = joinpath(output_dir, stem * ".json")
        mkpath(output_dir)
        mktemp(output_dir) do temporary_path, file
            JSON3.pretty(file, data)
            close(file)
            mv(temporary_path, output_path; force = true)
        end
        println("Saved: $output_path")
    end
end

main()
