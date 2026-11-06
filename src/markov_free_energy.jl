using ITensors, ITensorMPS
using LinearAlgebra

function entanglement_entropy(
    psi::MPS,
    sites::UnitRange{Int64};
    normalize::Bool = true,
    orthogonalize::Bool = true,
)::Float64

    N = length(psi)

    if normalize
        psi = LinearAlgebra.normalize(psi)
    end

    if sites[1] == 1 && sites[end] == N

        SvN = 0.0

    elseif sites[1] == 1

        b = sites[end]
        if orthogonalize
            psi = ITensorMPS.orthogonalize(psi, b)
        end

        l_inds = [linkinds(psi, b - 1)..., siteinds(psi, b)...]
        r_inds = [linkinds(psi, b)...]
        mat = Array(psi[b], l_inds..., r_inds...)
        mat = reshape(mat, (dim(l_inds), dim(r_inds)))

        S = svdvals(mat)
        SvN = 0.0
        for n in eachindex(S)
            p = S[n]^2
            if p > 1e-10
                SvN += -p * log(p)
            end
        end

    elseif sites[end] == N

        b = sites[1]
        if orthogonalize
            psi = ITensorMPS.orthogonalize(psi, b)
        end

        l_inds = [linkinds(psi, b - 1)...]
        r_inds = [linkinds(psi, b)..., siteinds(psi, b)...]
        mat = Array(psi[b], l_inds..., r_inds...)
        mat = reshape(mat, (dim(l_inds), dim(r_inds)))

        S = svdvals(mat)
        SvN = 0.0
        for n in eachindex(S)
            p = S[n]^2
            if p > 1e-10
                SvN += -p * log(p)
            end
        end

    else

        if orthogonalize
            psi = ITensorMPS.orthogonalize(psi, sites[1])
        end

        boundary_dim = dim(linkind(psi, sites[1] - 1)) * dim(linkind(psi, sites[end]))
        # Compare with 2^length(sites) without overflowing for large blocks.
        if ndigits(boundary_dim; base = 2) <= length(sites)

            tensor = ITensor(1.0)
            for s in sites
                tensor *= psi[s] * dag(prime(psi[s], "Link"))
            end

            l_inds = [i for i in inds(tensor) if hasplev(i, 0)]
            r_inds = [i for i in inds(tensor) if hasplev(i, 1)]
            mat = Array(tensor, l_inds..., r_inds...)
            mat = reshape(mat, (dim(l_inds), dim(r_inds)))

            S = eigvals(Hermitian(mat))
            SvN = sum(-p * log(p) for p in S if p > 1e-10)

        else

            tensor = ITensor(1.0)
            for s in sites
                tensor *= psi[s]
            end
            tensor *= dag(prime(tensor, "Site"))

            l_inds = [i for i in inds(tensor) if hasplev(i, 0)]
            r_inds = [i for i in inds(tensor) if hasplev(i, 1)]
            mat = Array(tensor, l_inds..., r_inds...)
            mat = reshape(mat, (dim(l_inds), dim(r_inds)))

            S = eigvals(Hermitian(mat))
            SvN = sum(-p * log(p) for p in S if p > 1e-10)

        end

    end

    return SvN

end

function energy(psi::MPS, H::MPO; normalize::Bool = true)::Float64

    if normalize
        psi = LinearAlgebra.normalize(psi)
    end

    return real(inner(psi', H, psi))

end

function effective_hamiltonian(pos::Int64, psi::MPS, H::MPO)::ITensor

    ltensor = ITensor(1.0)
    for j = 1:pos-1
        ltensor *= psi[j] * H[j] * dag(prime(psi[j]))
    end

    rtensor = ITensor(1.0)
    for j = pos+1:length(psi)
        rtensor *= psi[j] * H[j] * dag(prime(psi[j]))
    end

    return ltensor * H[pos] * rtensor

end

function _1site_energy(
    loc_mat::Array{Float64},
    eff_ham::ITensor;
    normalize::Bool = true,
)::Float64

    if normalize
        loc_mat = loc_mat / norm(loc_mat)
    end

    site_inds = [i for i in inds(eff_ham, "Site") if hasplev(i, 0)]
    link_inds = [i for i in inds(eff_ham, "Link") if hasplev(i, 0)]
    tensor = ITensor(loc_mat, site_inds, link_inds)

    return real(scalar(tensor * eff_ham * dag(prime(tensor))))

end

function extract_matrix(psi::MPS, pos::Int64)::Array{Float64}

    site_inds = inds(psi[pos], "Site")
    if pos == 1
        link_inds = linkinds(psi, pos)
    elseif pos == length(psi)
        link_inds = linkinds(psi, pos - 1)
    else
        link_inds = (linkind(psi, pos - 1), linkind(psi, pos))
    end

    return Array(psi[pos], site_inds..., link_inds...)

end

function embed_matrix(psi::MPS, pos::Int64, loc_mat::Array{Float64})::MPS

    site_inds = inds(psi[pos], "Site")
    if pos == 1
        link_inds = linkinds(psi, pos)
    elseif pos == length(psi)
        link_inds = linkinds(psi, pos - 1)
    else
        link_inds = (linkind(psi, pos - 1), linkind(psi, pos))
    end

    psi0 = copy(psi)
    psi0[pos] = ITensor(loc_mat, site_inds, link_inds)

    return psi0

end
