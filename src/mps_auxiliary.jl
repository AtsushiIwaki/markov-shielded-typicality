using ITensors, ITensorMPS
using LinearAlgebra

include("markov_free_energy.jl")

function markov_entropy_aux(
    psi::MPS,
    m::Int64;
    normalize::Bool = true,
    orthogonalize::Bool = true,
)::Float64

    Ntot = length(psi)
    Naux = Int(log2(maxlinkdim(psi)))
    N = Ntot - 2 * Naux

    if normalize
        psi = LinearAlgebra.normalize(psi)
    end

    if orthogonalize
        psi = ITensorMPS.orthogonalize(psi, Naux + 1)
    end

    SM = 0.0
    for k = Naux+1:Naux+N
        if k == Naux + 1
            SM += entanglement_entropy(psi, k:k; normalize = false, orthogonalize = false)
        else
            first = max(k - m, Naux + 1)
            if first > Naux + 1
                psi = ITensorMPS.orthogonalize(psi, first)
            end
            SM +=
                entanglement_entropy(psi, first:k; normalize = false, orthogonalize = false)
            # An empty shield has zero entropy.
            if first < k
                SM -= entanglement_entropy(
                    psi,
                    first:k-1;
                    normalize = false,
                    orthogonalize = false,
                )
            end
        end
    end

    return SM

end

function _1site_markov_free_energy_aux(
    loc_mat::Array{Float64},
    pos::Int64,
    psi::MPS,
    eff_ham::ITensor,
    m::Int64,
    T::Float64;
    normalize::Bool = true,
)::Float64

    if normalize
        loc_mat /= norm(loc_mat)
    end

    ene = _1site_energy(loc_mat, eff_ham; normalize = false)

    psi = embed_matrix(psi, pos, loc_mat)
    ent = markov_entropy_aux(psi, m; normalize = false)

    return ene - T * ent

end

function markov_free_energy_aux(
    psi::MPS,
    H::MPO,
    m::Int64,
    T::Float64;
    normalize::Bool = true,
)::Float64

    if normalize
        psi = LinearAlgebra.normalize(psi)
    end

    ene = energy(psi, H; normalize = false)

    ent = markov_entropy_aux(psi, m; normalize = false)

    return ene - T * ent

end
