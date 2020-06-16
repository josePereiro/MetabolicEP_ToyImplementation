# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: jl,ipynb
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

using SpecialFunctions
using LinearAlgebra
using Distributions
using Plots
pyplot();

# ## References
#
# Braunstein, Alfredo, Anna Muntoni, Andrea Pagnani, and Mirko Pieropan. “Compressed Sensing Reconstruction Using Expectation Propagation.” Journal of Physics A: Mathematical and Theoretical, July 9, 2019. https://doi.org/10.1088/1751-8121/ab3065.
#
# Braunstein, Alfredo, Anna Paola Muntoni, and Andrea Pagnani. “An Analytic Approximation of the Feasible Space of Metabolic Networks.” Nature Communications 8, no. 1 (April 6, 2017): 1–9. https://doi.org/10.1038/ncomms14915.

# ### Formulating of the problem

# We are going to formulate an iterative strategy to solve the problem of finding a multivariate probability measure over the set of fluxes $n$ compatible with equations (1 and 2). 
#
# $$  \large \mathbf{Sv} = \mathbf{b} \ \ \ \ \ \ \small (1) $$         
# $$ \large  \mathbf{lb} \le \mathbf{v} \le \mathbf{ub} \ \ \ \ \ \ \small (2) $$         

# +
# Toy Model
# rxns: gt    ferm  resp  ldh   lt   biom    atpm  # mets
const S = [   1.0  -1.0   0.0   0.0   0.0   0.0    0.0;  #  G
        0.0   2.0  18.0   0.0   0.0  -55.0  -5.0;  #  E
        0.0   2.0  -1.0  -1.0   0.0   0.0    0.0;  #  P
        0.0   0.0   0.0   1.0   1.0   0.0    0.0;  #  L
    ]

M,N = size(S)

mets = ["G", "E", "P", "L"]
const b =    [0.0, 0.0, 0.0, 0.0] # demand
metNames = ["Glucose", "Energy", "Intermediate Product" , "Lactate"];

rxns = ["gt"  ,"ferm" ,"resp" , "ldh" ,  "lt" , "biom", "atpm"];
const lb =   [0.0   , 0.0   , 0.0   ,  0.0  , -10.0,   0.0,     0.5];
const ub =   [1.0   , 10.0  , 10.0  , 10.0  ,   0.0,  10.0,    10.0];
rxnNames = ["Glucose transport", "Fermentation", "Respiration", 
    "Lactate DH", "Lactate transport", "Biomass production rate", "atp demand"];
# -

const β = 1e10

# For a vector of fluxes satisfying bounds 2, we can define a quadratic energy function $E(n)$ whose minimum(s) lies on the assignment of variables $v$ satisfying the stoichiometric constraints in equation (1)
#
# $$\large E(\mathbf{v}) = 
#     \frac{1}{2} (\mathbf{Sv} - \mathbf{b})^T(\mathbf{Sv} - \mathbf{b}) 
# \ \ \ \ \ \ \small (3) 
# $$         
#
# We define the likelihood of observing $b$ given a set of fluxes $v$ as a Boltzmann distribution:
#
# $$\large 
# P(\mathbf{b}|\mathbf{v}) = 
#     \Bigg{(} \frac{\beta}{2\pi}\Bigg{)}^{\frac{M}{2}} 
#    exp\Bigg{[}
#        -\frac{\beta}{2} (\mathbf{Sv} - \mathbf{b})^T(\mathbf{Sv} - \mathbf{b})
#     \Bigg{]} 
# \ \ \ \ \ \ \small (4) 
# $$ 
#
# In a Bayesian perspective, one can consider the posterior probability of observing $P(\mathbf{v}|\mathbf{b})$ as:
#
# $$ \large 
#     P(\mathbf{v}|\mathbf{b}) = 
#     \frac{P(\mathbf{b}|\mathbf{v})P(\mathbf{v})}{P(\mathbf{b})}
# \ \ \ \ \ \ \small (5) 
# $$
#
# where the prior
#
# $$ \large 
#     P(\mathbf{v}) = \prod_{n=1}^{N} \psi_{n}(v_{n}) = 
#         \prod_{n=1}^{N} \frac{\mathbb{1} (v_n \in [v_{n}^{inf}, v_{n}^{sup}])} {v_{n}^{inf} - v_{n}^{sup}}
# \ \ \ \ \ \ \small (6) 
# $$
#
# We finally obtain the following relation for the posterior
#
# $$ \large
#     P(\mathbf{v}|\mathbf{b}) = 
#     \frac{1}{P(\mathbf{b})}
#      exp\Bigg{[}
#        -\frac{\beta}{2} (\mathbf{Sv} - \mathbf{b})^T(\mathbf{Sv} - \mathbf{b})
#     \Bigg{]} 
#     \prod_{n=1}^{N} \psi_{n}(v_n)
# \ \ \ \ \ \ \small (7) 
# $$
#
# By marginalization of equation (7), one can determine the marginal posterior $P_n(v_n|\mathbf{b})$ for each flux $n \in \{1, ..., N\}$. However, performing this computation naively would require the calculation of a multiple integral that is in principle computationally very expensive and cannot be performed analytically in an efficient way

# ### A non-adaptative approach

# As a first approximation, one can think of replacing each prior $\psi_n (v_n)$ with a single Gaussian distribution $\phi_n(v_n)$:
#
# $$ \large
#     \phi_n(v_n) = \frac{1}{\sqrt{2\pi \, d_n}} 
#     exp \Bigg{[} -
#         \frac{{(v_n - a_n)}^2}{2d_n} 
#     \Bigg{]}
# \ \ \ \ \ \ \small (8) 
# $$
#
# Note that in this approximation fluxes result unbounded

function ϕ(vn, an = 0.0, dn = 1.0)
    exp(-((vn - an)^2)/(2dn))/sqrt(2π*dn)
end

# whose statistics, that is, the mean and the variance, are constrained to be equal to the one of $\psi_n(v_n)$. That is:
#
# $$ \large
# \left\{
#     \begin{array}\\
#         a_n = {\langle v_n \rangle}_{\psi_n (v_n)} \\
#         d_n = {\langle v_{n}^2 \rangle}_{\psi_n (v_n)} - {\langle v_n \rangle}_{\psi_n (v_n)}^2 \\
#     \end{array}
# \right. 
# \ \ \ \ \ \ \small (9) 
# $$
#
# The exact priors $\psi_n (v_n)$ are just an uniform over the interval $lb_n \le v_n \le ub_n$, so:
#
# $$ \large{
# {\langle v_n \rangle}_{\psi_n (v_n)} = \frac{ub_n + lb_n}{2} \\
# {\langle v_{n}^2 \rangle}_{\psi_n (v_n)} - {\langle v_n \rangle}_{\psi_n (v_n)}^2 = \frac{1}{12} (ub_n - lb_n)^2
# }
# $$

ψ(vn, lb, ub) = lb <= vn <= ub ? 1/(ub - lb) : zero(vn)
ψave(lb, ub) = (ub + lb)/2
ψvar(lb, ub) = ((ub - lb)^2)/12;

_lb, _ub = 0, 10
_m = (_ub - _lb)/10
plot(title = "Priors", xlabel = "v", ylabel = "pdf", size = [500,350])
plot!(vn -> ψ(vn, _lb, _ub), _lb - _m, _ub + _m, label = "ψ", color = :red, lw = 3) 
plot!(vn -> ϕ(vn, ψave(_lb, _ub), ψvar(_lb, _ub)), _lb - _m, _ub + _m, label = "ϕ", color = :blue, lw = 3)
vline!([ψave(_lb, _ub)], label = "ave", color = :black)
vline!([ψave(_lb, _ub) - sqrt(ψvar(_lb, _ub))], color = :black, ls = :dash, label = "std")
vline!([ψave(_lb, _ub) + sqrt(ψvar(_lb, _ub))],  color = :black, ls = :dash, label = "")

# We estimate the **marginal posteriors** from the distribution
#
# $$ \large
#     Q(\mathbf{v}|\mathbf{b}) = 
#     \frac{1}{Z_Q}
#     exp\Bigg{[}
#        -\frac{\beta}{2} (\mathbf{Sv} - \mathbf{b})^T(\mathbf{Sv} - \mathbf{b})
#     \Bigg{]} 
#     \prod_{n=1}^{N} \phi_{n}(v_n; a_n, d_n)
# \ \ \ \ \ \ \small (10) 
# $$
#
#
# $$ \large  
#     =
#     \frac{1}{Z_Q}
#     exp\Bigg{[}
#        -\frac{1}{2} (\mathbf{v} - \mathbf{\mu})^T \, \mathbf\Sigma^{-1} \, (\mathbf{v} - \mathbf{\mu})
#     \Bigg{]} 
# $$
#
# where 
#
# $$ \large
#     \Sigma^{-1} = \beta \, \mathbf{S}^T \mathbf{S} + \mathbf{D}
# \ \ \ \ \ \ \small (11) 
# $$
#
# $$ \large
#     \mathbf{\mu} = \mathbf{\Sigma} \, (\beta \, \mathbf{S}^T \, \mathbf{b} + \mathbf{D} \, \mathbf{a})
# \ \ \ \ \ \ \small (12) 
# $$
#
# $$ \large
#     Z_Q = (2 \, \pi)^{\frac{N}{2}} (\det \mathbf{\Sigma})^{\frac{1}{2}}
# \ \ \ \ \ \ \small (13) 
# $$
#
# and $\mathbf D$ is a diagonal matrix having elements $d_1^{-1}, ... , d_N^{-1}$.

function Q(v, μ, Σ)
    Zq = sqrt(det(Σ))*(2π)^(N/2)
    return exp(-0.5(v - μ)'*inv(Σ)*(v - μ))/Zq
end

a = ψave.(lb, ub)
d = ψvar.(lb, ub)
D = Diagonal(1 ./ a)
Σ = inv(β*S'S + D)
μ = Σ*(β*S'b + D*a);

ps = []
for (i, ider) in rxns |> enumerate
    p = Plots.plot(lb[i], ub[i], xlabel = "v", ylabel = "pdf", label = ider) do v
        ϕ(v, μ[i], Σ[i, i])
    end
    push!(ps, p)
end

Plots.plot(ps..., size = [900,900])

# ### Expectation Propagation (EP)

# EP [26] is an efficient technique to approximate intractable (that is, impossible or impractical to compute analytically) posterior probabilities.
#
# Let us consider the $n$th flux and its corresponding approximate prior $\phi_n(v_n; a_n, d_n)$. We define a **tilted distribution** $Q^{(n)}$ as:
#
# $$ \large
#     Q^{(n)} (\mathbf{v}|\mathbf{b}) \equiv
#     \frac{1}{\mathbf{Z}_{Q^{(n)}}}
#     exp\Bigg{[}
#        -\frac{\beta}{2} (\mathbf{Sv} - \mathbf{b})^T(\mathbf{Sv} - \mathbf{b})
#     \Bigg{]} 
#     \psi_n(v_n)
#     \prod_{m\neq n}^{N} \phi_{m}(v_m; a_m, d_m)
# \ \ \ \ \ \ \small (13) 
# $$
#
# $$ \large
#     \equiv
#     \frac{1}{\mathbf{Z}_{Q^{(n)}}}
#     exp\Bigg{[}
#        -\frac{1}{2} (\mathbf{v} - \mathbf{\mu}^{(n)})^T \, {\mathbf{\Sigma}^{(n)}}^{-1} \, (\mathbf{v} - \mathbf{\mu}^{(n)})
#     \Bigg{]} 
#     \psi_n(v_n)
# $$
#
# where
#     
# $$ \large
#     {\mathbf{\Sigma}^{(n)}}^{-1} = \beta \, \mathbf{S}^T \mathbf{S} + \mathbf{D}^{(n)}
# $$
#
# $$ \large
#     \mathbf{\mu}^{(n)} = \mathbf{\Sigma}^{(n)} \, (\beta \, \mathbf{S}^T \, \mathbf{b} + \mathbf{D}^{(n)} \, \mathbf{a})
# $$
#
# <!-- $$ \large
#     Z_{Q^{(n)}} = (2 \, \pi)^{\frac{N}{2}} (\det \mathbf{\Sigma}^{(n)})^{\frac{1}{2}}
# $$
#  -->
# and, in analogy with equation (10), $D^{(n)}$ is a diagonal matrix of elements $d^{-1}_m$ for all diagonal elements $m \neq n$ and zero for $m = n$.

# The important difference between the **tilted distribution** and the **multivariate Gaussian** $Q(\mathbf{v}|\mathbf{b})$ is that all the intractable priors are
# approximated as Gaussian probability densities except for the $n$th
# prior, which is treated **exactly**. **For this reason, we expect that this distribution will be more accurate than $Q(\mathbf{v}|\mathbf{b})$ regarding the estimate of the statistics of flux $n$ without significantly affecting the computation of expectations**. 
#
# One way of determining the unknown parameters $a_n$ and $d_n$ of $\phi_n(v_n; a_n, d_n)$ is to require that the multivariate Gaussian distribution $Q(\mathbf{v}|\mathbf{b})$ is as close as possible to the auxiliary distribution $Q^{(n)}(\mathbf{v}|\mathbf{b})$. This can be done by matching the first and the second moments of the two distributions. Thus, we aim at imposing the following moment matching conditions:
#
# $$ \large
# \left\{
#     \begin{array}\\
#         {\langle v_n \rangle}_{Q^{(n)}} = {\langle v_n \rangle}_{Q} \\
#         {\langle v_n^2 \rangle}_{Q^{(n)}} = {\langle v_n^2 \rangle}_{Q} \\
#     \end{array}
# \right. 
# \ \ \ \ \ \ \small (14) 
# $$
#
# from which we get a relation for the parameters $a_n$, $d_n$:
#
# $$ \large
#     d_n = \Bigg(
#         \frac{1}{{\langle v_n^2 \rangle}_{Q^{(n)}} - {\langle v_n \rangle}_{Q^{(n)}}^2} -
#         \frac{1}{\Sigma_{nn}^{(n)}}
#     \Bigg)^{-1}
# $$
#
# $$ \large
#     a_n = d_n \Bigg[
#         {\langle v_n \rangle}_{Q^{(n)}}
#         \Bigg(
#             \frac{1}{d_n} + \frac{1}{\Sigma^{(n)}_{nn}} 
#         \Bigg) - \frac{\mu^{(n)}_n}{\Sigma^{(n)}_{nn}} 
#     \Bigg]
# $$
#
# The **moments of the tilted distribution** can be found as:
#
# $$ \large
#     {\langle v_n \rangle}_{Q^{(n)}} = \mu^{(n)}_n + 
#     \frac{\mathcal{N}(A^{(n)}_n) - \mathcal{N}(B^{(n)}_n)}
#         {\Phi(B^{(n)}_n) - \Phi(A^{(n)}_n)}
#         \sqrt{\Sigma^{(n)}_{nn}}
# $$
#
#
# $$ \large
#     {\langle v_n^2 \rangle}_{Q^{(n)}} - {\langle v_n \rangle}_{Q^{(n)}} =
#     \Sigma^{(n)}_{nn}
#     \Bigg[
#         1 + 
#         \frac
#         {
#             A^{(n)}_n\mathcal{N}(A^{(n)}_n) -
#             B^{(n)}_n\mathcal{N}(B^{(n)}_n)
#         }
#         {
#             \Phi(B^{(n)}_n) - \Phi(A^{(n)}_n)
#         } -
#         \Bigg(
#             \frac
#             {\mathcal{N}(A^{(n)}_n) - \mathcal{N}(B^{(n)}_n)}
#             {\Phi(B^{(n)}_n) - \Phi(A^{(n)}_n)}
#         \Bigg)^2 \,
#     \Bigg]
# $$
#
# where
#
# $$ \large
#     A^{(n)}_n = \frac{v^{inf}_n \, - \mu^{(n)}_n}{\sqrt{\Sigma^{(n)}_{nn}}},\:
#     B^{(n)}_n = \frac{v^{sup}_n \, - \mu^{(n)}_n}{\sqrt{\Sigma^{(n)}_{nn}}}
# $$
#
# and
#
# $$ \large 
#     \mathcal{N}(x) = \frac{1}{\sqrt{2\pi}}
#     exp \Bigg(
#         -\frac{x^2}{2}
#     \Bigg)
# $$ is the probability density function of the standard normal distribution and
#
# $$ \large
#     \Phi(x) = \frac{1}{2} \Bigg[1 + erf(\frac{2}{\sqrt{2}})\Bigg]
# $$
# is its comulative.
#

# EP consists in sequentially repeating this update step for all the other fluxes and iterate until we reach a numerical convergence. At the fixed point, we directly estimate the **marginal posteriors** $P_n(v_n|\mathbf{b})$ , for $n \in \{1, ... ,N\}$, from marginalization of the **tilted distribution** $Q^{(n)}$ that turns out to be a **truncated Gaussian** density in the interval $[v_n^{inf} , v_n^{sup}]$ 
#
# At difference from the **non-adaptive approach**, the EP algorithm determines the **approximated prior** density by trying to reproduce the effect that the **true prior density** has on variable $v_n$, including the interaction of this term with the rest of the system. First, the information encoded in the stoichiometric matrix is surely  encompassed in the computation of the means and the variances of the approximation since both the distributions $Q^{(n)}$ and $Q$ contain the exact expression of the **likelihood**. Second, the refinement of each prior also depends on the parameters of all the other fluxes.

Φ(x) = 0.5*(1.0 + erf(x/sqrt(2.0)));

Plots.plot(Φ, -10, 10, label = "Φ", 
    xlabel = "v", ylabel = "pdf", size = [300, 200], legend = :topleft)
Plots.plot!(ϕ, -10, 10, label = "ϕ")

# ### EP Implementation
# This implementation is just for educative propose, for a efficient one see the reference. 

# +
iters = 5000
const β = 1e12

# This are constants
_βSS = β*S'*S
_βSb = β*S'*b

# initialize a and d, just using the parameters of the exact priors
a = ψave.(lb, ub)
d = ψvar.(lb, ub)

# For numerical reasons we normalize all
# nfactor = 1.0 # max(maximum(abs.(ub)), maximum(abs.(lb))) # scale factor
nlb = copy(lb) # ./ nfactor
nub = copy(ub) # ./ nfactor

# track the tilted parameters in each iteration
μss = [] 
σss = [] 

# Last itaration
ave = zeros(Float64, N)
var = zeros(Float64, N)
for it in 1:iters
    for (n, rxn) in enumerate(rxns)
        
        # Tilted parameters
        Dn = Diagonal(1 ./ d)
        Dn[n,n] = 0.0
        
        Σn = inv(_βSS + Dn)
        μn = Σn*(_βSb + Dn * a)
        
        # Tilted moments
        An = (nlb[n] - μn[n])/sqrt(Σn[n,n])
        Bn = (nub[n] - μn[n])/sqrt(Σn[n,n])
        ϕAn = ϕ(An);  ϕBn = ϕ(Bn)
        ΦAn = Φ(An); ΦBn = Φ(Bn)
        
        ave[n] = μn[n] + ((ϕAn - ϕBn)/(ΦBn - ΦAn)) * sqrt(Σn[n,n])
        var[n] = Σn[n,n] * (1 + (An*ϕAn - Bn*ϕBn)/(ΦBn - ΦAn) - ((ϕAn - ϕBn)/(ΦBn - ΦAn))^2)
        
        
        # Updating an and dn
        d[n] = 1/(1/var[n] - 1/Σn[n,n])
        a[n] = d[n]*(ave[n]*(1/d[n] + 1/Σn[n,n]) - μn[n]/Σn[n,n])
    end
end
# -

D = Diagonal(1 ./ a)
Σ = inv(β*S'S + D)
μ = Σ*(β*S'b + D*a);
tϕs = [Truncated(Normal(μ[i], sqrt(Σ[i,i])), lb[i], ub[i]) for i in eachindex(rxns)];

ps = []
for (i, ider) in rxns |> enumerate
    _m = (ub[i] - lb[i])/10
    p = Plots.plot(lb[i] - _m, ub[i] + _m, 
            xlabel = "v", ylabel = "pdf", label = "Q", title = ider) do v
        ϕ(v, μ[i], Σ[i, i])
    end
    
    p = Plots.plot!(v -> pdf(tϕs[i], v), lb[i] - _m, ub[i] + _m, label = "Qn")     
    push!(ps, p)
end

Plots.plot(ps..., size = [900,900])

S * mean.(tϕs)





using Chemostat
Ch = Chemostat;

model = Ch.Utils.MetNet(S, b, lb, ub, rxns);

epout = Ch.MaxEntEP.maxent_ep(model, alpha = β);

tϕs = [Truncated(Normal(epout.μ[i], sqrt(Σ[i,i])), lb[i], ub[i]) for i in eachindex(rxns)];

epout.av 

ave


