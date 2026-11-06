#=
MST-MPS method, version 3
* Markov-shielded typical matrix product states
* MPS with auxiliary systems
* Transverse field Ising model
* 1-site update of local tensor by optimization
* Numerical cost per sweep: O(N^2 8^m)
* ITensors.jl + Optim.jl
2024/12/31/Tue
=#

using ITensors, ITensorMPS
using Optim
using ArgParse
using Printf
using JSON3
using HDF5

include("../src/mps_auxiliary.jl")

function is_tenth_grid(value)
    scaled = 10 * value
    nearest = round(scaled)
    return typemin(Int) <= nearest <= typemax(Int) &&
        isapprox(scaled, nearest; rtol = 2eps(Float64))
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--system_size"
        arg_type = Int
        default = 16
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
        arg_type = Int
        default = 5
        "--temperature_interval"
        help = "Temperature interval, a positive multiple of 0.1."
        arg_type = Float64
        default = 0.1
        range_tester = value -> value > 0 && is_tenth_grid(value)
        "--maximum_temperature"
        arg_type = Float64
        default = 0.9
        "--num_sweeps"
        arg_type = Int
        default = 10
        "--num_iterations"
        arg_type = Int
        default = 10
        "--free_energy_tolerance"
        arg_type = Float64
        default = 1e-5
        "--output_level"
        arg_type = Int
        default = 1
        "--output_format"
        arg_type = String
        default = "json"
    end
    return parse_args(s)
end

function main()

    parsed_args = parse_commandline()

    N = parsed_args["system_size"]
    g = parsed_args["transverse_field"]
    g_tag = round(Int, 10 * g)
    chi = parsed_args["bond_dimension"]

    m = parsed_args["markov_shield"]
    del_T = parsed_args["temperature_interval"]
    T_max = parsed_args["maximum_temperature"]

    n_sweep = parsed_args["num_sweeps"]
    n_iter = parsed_args["num_iterations"]
    tol_fene = parsed_args["free_energy_tolerance"]

    output_level = parsed_args["output_level"]
    output_format = parsed_args["output_format"]

    Naux = Int(log2(chi))
    Ntot = N + 2 * Naux

    ground_file = @sprintf(
        "results/mps/n%0.3d_g%0.2d_c%0.2d_m%0.2d_t00.h5",
        N,
        g_tag,
        chi,
        m
    )
    legacy_mps_dir = "results/mst_mps/mps"
    legacy_ground_file = joinpath(legacy_mps_dir, basename(ground_file))
    reuse_legacy_mps = !isfile(ground_file) && isfile(legacy_ground_file)
    saved_ground_file = reuse_legacy_mps ? legacy_ground_file : ground_file
    if isfile(saved_ground_file)
        h5open(saved_ground_file, "r") do file
            psi = read(file, "psi", MPS)
        end
        sites = siteinds(psi)
    else
        sites = siteinds("S=1/2", Ntot)
    end

    # MPO of Hamiltonian
    os = OpSum()
    for j = Naux+1:Naux+N-1
        os -= "Z", j, "Z", j + 1
    end
    for j = Naux+1:Naux+N
        os -= g, "X", j
    end
    H = MPO(os, sites)

    # MPO of magnetization x
    os = OpSum()
    for j = Naux+1:Naux+N
        os += "X", j
    end
    Mx = MPO(os, sites)

    # MPO of magnetization z
    os = OpSum()
    for j = Naux+1:Naux+N
        os += "Z", j
    end
    Mz = MPO(os, sites)

    # initialize observables
    Ts = 0.0:del_T:T_max
    enes = Vector{Float64}(undef, length(Ts))
    mfes = Vector{Float64}(undef, length(Ts))
    mgxs = Vector{Float64}(undef, length(Ts))
    mgzs = Vector{Float64}(undef, length(Ts))

    psi = random_mps(sites; linkdims = chi)
    orthogonalize!(psi, Ntot)

    mkpath("results/mps")

    # ground state at T = 0
    # optimization of energy
    T = Ts[1]
    if output_level >= 1
        println("T = $T")
    end

    if isfile(saved_ground_file)

        h5open(saved_ground_file, "r") do file
            psi = read(file, "psi", MPS)
        end
        println("MPS loaded from $saved_ground_file")

    else

        for sweep = 1:n_sweep
            start_time = time()

            for pos in vcat(1:Ntot-1, Ntot:-1:1)
                orthogonalize!(psi, pos)

                eff_ham = effective_hamiltonian(pos, psi, H)
                loc_mat = extract_matrix(psi, pos)

                result = optimize(x -> _1site_energy(x, eff_ham), loc_mat, BFGS())

                loc_mat = Optim.minimizer(result)
                loc_mat /= norm(loc_mat)

                psi = embed_matrix(psi, pos, loc_mat)
            end

            end_time = time()

            if output_level >= 1
                ene = energy(psi, H) / N
                println(
                    "$sweep / $n_sweep sweeps: energy = $ene, time = $(end_time - start_time)",
                )
            end
        end

        # save MPS to HDF5 file
        h5open(ground_file, "w") do file
            write(file, "psi", psi)
        end

    end

    # calculate observables
    ene = energy(psi, H) / N
    mfe = ene
    mgx = real(inner(psi', Mx, psi)) / N
    mgz = real(inner(psi', Mz, psi)) / N
    enes[1] = ene
    mfes[1] = mfe
    mgxs[1] = mgx
    mgzs[1] = mgz


    # finite-temperature MST state
    # optimization of Markov free energy
    for i = 2:length(Ts)
        T = Ts[i]
        if output_level >= 1
            println("T = $T")
        end

        file_name = @sprintf(
            "results/mps/n%0.3d_g%0.2d_c%0.2d_m%0.2d_t%0.2d.h5",
            N,
            g_tag,
            chi,
            m,
            round(Int, 10 * T)
        )

        saved_file = file_name
        if reuse_legacy_mps && !isfile(saved_file)
            saved_file = joinpath(legacy_mps_dir, basename(file_name))
        end
        if isfile(saved_file)

            h5open(saved_file, "r") do file
                psi = read(file, "psi", MPS)
            end
            println("MPS loaded from $saved_file")

        else

            for sweep = 1:n_sweep
                start_time = time()
                mfe0 = markov_free_energy_aux(psi, H, m, T) / N

                for pos in vcat(1:Ntot-1, Ntot:-1:1)
                    orthogonalize!(psi, pos)

                    eff_ham = effective_hamiltonian(pos, psi, H)
                    loc_mat = extract_matrix(psi, pos)

                    result = optimize(
                        x -> _1site_markov_free_energy_aux(x, pos, psi, eff_ham, m, T),
                        loc_mat,
                        BFGS(),
                        Optim.Options(iterations = n_iter),
                    )

                    loc_mat = Optim.minimizer(result)
                    loc_mat /= norm(loc_mat)

                    psi = embed_matrix(psi, pos, loc_mat)

                    conv = Optim.converged(result)

                    if output_level >= 2
                        mfe = markov_free_energy_aux(psi, H, m, T) / N
                        println("$pos / $Ntot site: free energy = $mfe, converged = $conv")
                    end
                end

                end_time = time()
                mfe = markov_free_energy_aux(psi, H, m, T) / N

                if output_level >= 1
                    println(
                        "$sweep / $n_sweep sweeps: free energy = $mfe, time = $(end_time - start_time)",
                    )
                end

                if abs(mfe - mfe0) < tol_fene
                    break
                end
            end

            # save MPS to HDF5 file
            h5open(file_name, "w") do file
                write(file, "psi", psi)
            end

        end

        # calculate observables
        ene = energy(psi, H) / N
        mfe = markov_free_energy_aux(psi, H, m, T) / N
        mgx = inner(psi', Mx, psi) / N
        mgz = inner(psi', Mz, psi) / N
        enes[i] = ene
        mfes[i] = mfe
        mgxs[i] = mgx
        mgzs[i] = mgz

    end

    if isdir("results/observables") == false
        mkpath("results/observables")
    end

    if output_format == "txt" # output observables to txt file
        file_name =
            @sprintf "results/observables/n%0.3d_g%0.2d_c%0.2d_m%0.2d.txt" N g_tag chi m
        open(file_name, "w") do io
            println(io, "# \$1 temperature")
            println(io, "# \$2 energy")
            println(io, "# \$3 Markov free energy")
            println(io, "# \$4 magnetization x")
            println(io, "# \$5 magnetization z")
            for i in eachindex(Ts)
                println(io, "$(Ts[i])\t$(enes[i])\t$(mfes[i])\t$(mgxs[i])\t$(mgzs[i])")
            end
        end
    elseif output_format == "json" # output observables to json file
        file_name =
            @sprintf "results/observables/n%0.3d_g%0.2d_c%0.2d_m%0.2d.json" N g_tag chi m
        json_dict = Dict(
            "temperature" => Ts,
            "energy" => enes,
            "markov_free_energy" => mfes,
            "magnetization_x" => mgxs,
            "magnetization_z" => mgzs,
        )
        open(file_name, "w") do io
            JSON3.pretty(io, json_dict)
        end
    else # output observables to stdout
        println(
            "# 1 temperatur | 2 energy | 3 free energy | 4 magnetization x | 5 magnetization z",
        )
        for i in eachindex(Ts)
            println("# $(Ts[i])\t$(enes[i])\t$(mfes[i])\t$(mgxs[i])\t$(mgzs[i])")
        end
    end

end

main()
