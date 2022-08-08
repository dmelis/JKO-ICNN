import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Distribution
from torch.distributions import constraints
import pdb
import math

class PDE():
    def __init__():
        #super(PDE, self).__init__()
        self.name = 'name'
    def __call__(self, x, t):
        pass



#### Porous Medium equation

# class PorousMediumPDE(PDE):
#     """ The porous medium equation:
#                     ∂ₜρ = ∆ρᵐ,   m > 1
#     """
#     def __init__(self, m = 2, t0=0, *args, **kwargs):
#         super(PorousMediumPDE, self).__init__(*args, **kwargs)
#         self.m  = m
#         self.t0 = t0
#
#     def forward(self, x=None, t=None):


###### Solutions of PDEs
#
# class BarenblattProfile2(Distribution):
#     """ Family of solutions to the porous medium equation:
#                     ∂ₜρ = ∆ρᵐ,   m > 1
#
#         They have the form:
#                 log ρ(x,t) = - (t + t0) + log ()    / (m-1)
#
#     """
#     arg_constraints = {
#         'm': constraints.nonnegative_integer,
#         'C': constraints.positive,
#         't0': constraints.positive,
#         'α': constraints.real,
#     }
#
#     def __init__(self, m = 2, t0=0, C=None, α=1, validate_args=None):
#         self.m  = m
#         self.t0 = torch.tensor(t0, dtype=torch.double)
#         self.C  = C
#         #self.α  = α
#         self.t  = self.t0.clone()
#         batch_shape = torch.Size()
#         #super(BarenblattProfile, self).__init__(batch_shape, validate_args=validate_args)
#
#     def log_prob(self, x, t=None):
#         if t is not None:
#             real_t = t + self.t0
#         else:
#             real_t = self.t
#         main_term = torch.relu(self.C - ((self.m-1)/(2*self.m*(self.m+1)))*torch.pow(x,2)*(real_t ** (-2/(self.m+1.0)) ))
#         print(main_term)
#         logp = torch.log(main_term)/(self.m-1.0) - torch.log(real_t)/(self.m+1.0)
#         return logp
#
#     def step(self, step_size=0.01):
#         self.t += step_size

# Rejection sampler

# Other samplers:
# https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture17.pdf
def rejection_sampler(ρ, sample_size, xmin,xmax, proposal=None, eps = 1e-2, plot=False, verbose=False):
    d = ρ.d
    if proposal is None:
        if d == 1:
            proposal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        else:
            proposal = torch.distributions.MultivariateNormal(torch.zeros(d), scale_tril=torch.diag(torch.ones(d)))

    unif = torch.distributions.Uniform(0,1)

    if d == 1:
        xgrid = torch.from_numpy(np.linspace(xmin, xmax, 100)).float()
    else:
        _grid = np.meshgrid(*[np.linspace(xmin,xmax,int(1000**(3/(d+2)))) for i in range(d)], indexing='ij')
        xgrid = torch.from_numpy(np.column_stack((_x.ravel() for _x in _grid))).float()

    f = np.reshape(torch.exp(ρ.log_prob(xgrid)), [xgrid.shape[0]])
    g = np.reshape(torch.exp(proposal.log_prob(xgrid)), [xgrid.shape[0]])
    M = (f/g).max() + eps
    assert torch.all(f<=M*g)

    #unif
    naccept = ntrial = 0
    samples = []
    MAXTRIALS = 1e7
    if verbose: print(f'M=sup(p/q)={M}')
    while (naccept < sample_size[0]) and (ntrial < MAXTRIALS):
        x = proposal.sample((1,))
        u = unif.sample((1,))
        if torch.log(u) < ρ.log_prob(x) - torch.log(M) - proposal.log_prob(x):
            samples.append(x.squeeze())
            naccept += 1
        else:
            RHS = (ρ.log_prob(x) - torch.log(M) - proposal.log_prob(x)).item()
            if verbose: print(f'Failed to accept: log(u) = {torch.log(u).item()} >= {RHS} = log(ρ(x)) - log(M) - log(q(x))')
        ntrial += 1

    samples = torch.stack(samples)
    if verbose:
        print(f'Rejection sampling: {ntrial} trials to get {sample_size} samples')

    if d == 1 and plot:
        fig, ax = plt.subplots()
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.plot(x_grid, f, color='red', ls='-', label='f')
        ax.plot(x_grid, M*g, color='blue', ls='-', label='M*g')
        ax.scatter(samples, 0*samples, c='red', s=0.3)
        ax.legend()
        plt.show()

    return samples.unsqueeze(1)


class DynamicDistribution(Distribution):
    """ Extends pytorch distributions to model dynamic (time-evolving) densities
        of the form ρ(x,t).

    """
    def __init__(self, t0, batch_shape, validate_args, *args, **kwargs):
        self.t0 = torch.tensor(t0, requires_grad=False, dtype=torch.double)
        self.t  = self.t0.clone()
        self.domain = self.get_domain()
        super(DynamicDistribution, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_size=None):
        proposal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        X = rejection_sampler(self, sample_size, -1, 1)
        return X

    def step(self, step_size=0.01):
        self.t += step_size
        self.domain = self.get_domain()

    def reset(self):
        self.t = self.t0.clone()
        self.domain = self.get_domain()

    def get_domain(self):
        return None



class FokkerPlanckNonLinDiff(object):
    """docstring for FokkerPlanckNonLinDiff."""

    def __init__(self, V=None, V_inv=None,
                 m=2, C=1, initial_mass=0, eps_domain = 1e-6):
        super(FokkerPlanckNonLinDiff, self).__init__()
        self.m = m
        # Find C by numerical integration
        #torch.linspace()
        #torch.trapz(y, x, *, dim=-1) → Tensor
        self.C = C
        if V is None:
            self.V = lambda x: torch.pow(x,2)
            self.V_inv = lambda x: math.sqrt(x)
        else:
            self.V = V
            self.V_inv = V_inv
        if self.V_inv is not None:
            thresh = self.V_inv((self.m/(self.m-1))*self.C) - eps_domain
            self.domain = (-thresh, thresh)
        else:
            pdb.set_trace(header='TODO')


    def log_prob(self, x):
        return torch.log(torch.relu(self.C - (self.m - 1)*self.V(x)/(1.0*self.m))/(self.m-1))

    def prob(self, x):
        return torch.pow(torch.relu(self.C - (self.m - 1)*self.V(x)/(1.0*self.m)), 1./(self.m-1))

    def get_domain(self, t=None, eps=1e-6):
        return self.domain

    def sample(self, sample_size=None):
        proposal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        X = rejection_sampler(self, sample_size, -1, 1)
        return X

class AggregationEquilibrium(object):
    def __init__(self, eps_domain = 1e-6):
        super(AggregationEquilibrium, self).__init__()
        self.domain = (-np.sqrt(2).item()+eps_domain, np.sqrt(2).item()-eps_domain)

    def prob(self, x):
        return torch.sqrt(torch.relu(2- x**2))/np.pi

    def log_prob(self, x):
        return 0.5*torch.log(torch.relu(2-x**2)) - np.log(np.pi)

    def get_domain(self, t=None, eps=1e-6):
        return self.domain

    def sample(self, sample_size=None, plot=False):
        proposal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        X = rejection_sampler(self, sample_size, *self.domain, plot=plot)
        return X
# https://pdfs.semanticscholar.org/74a6/84835fba7f2e1febb8012f2aed342f5d81f4.pdf?_ga=2.153272059.2117402532.1620770949-1326845582.1620770949

class HeatEquationSolution(object):
    """docstring for HeatEquationSolution."""

    def __init__(self, arg):
        super(HeatEquationSolution, self).__init__()
        self.arg = arg



class BarenblattProfile(DynamicDistribution):
    """ A.k.a Barenblatt-Pattle. Family of solutions to the porous medium equation:
                    ∂ₜρ = ∆ρᵐ,   m > 1

        They have the form:
                log ρ(x,t) = - (t + t0) + log ()    / (m-1)

        Steady state is ρ(x,∞) = 0, and asymptotically
                ρ(x,t) ∼ t^(-α) * C^(1/(m-1)) * Indicator[ |x| < Ct^{2β}/κ  ]


    """
    arg_constraints = {
        'm': constraints.nonnegative_integer,
        'M': constraints.positive,
        'd': constraints.nonnegative_integer,
        'C': constraints.positive,
        't0': constraints.positive,
    }
    support = constraints.real

    # TODO: fix validate args
    def __init__(self, m = 2, t0=0, C=0.5, d=1, M=2.0, validate_args=False):
        batch_shape = torch.Size()
        self.m = torch.tensor(m, requires_grad=False, dtype=torch.double)
        self.M = torch.tensor(M, requires_grad=False, dtype=torch.double)
        self.C = torch.tensor(C, requires_grad=False, dtype=torch.double)
        self.d = torch.tensor(d, dtype=torch.int32, requires_grad=False)
        self.α = torch.tensor(d/(d*(m-1) + 2), requires_grad=False, dtype=torch.double)
        self.β = self.α/d
        self.κ = self.α*(m-1)/(2*m*d)
        super(BarenblattProfile, self).__init__(t0, batch_shape, validate_args=validate_args)

    def get_domain(self, t=None, eps=1e-6):
        C, κ, α, β, d, m = self.C, self.κ, self.α, self.β, self.d, self.m
        real_t = (t + self.t0) if t is not None else self.t
        # shrink domain by eps to avoid infs close to boundary
        radius = (torch.sqrt(C/κ)*torch.pow(real_t, β)).item() - eps
        if d == 1:
            # core = κ * x.view(-1, d).pow(2).sum(1) * torch.pow(real_t, -2*β)
            bounds = [(-radius, radius)]
        else:
            # Bounding L-infty ball: superset of true domain
            bounds = [(-radius, radius) for i in range(d)]
            # Inscribed L-infty ball: subset of true domain
            bounds = [(-radius/math.sqrt(2), radius/math.sqrt(2)) for i in range(d)]

        return bounds

    def in_domain(self, x, t=None, eps=1e-6):
        C, κ, α, β, d, m = self.C, self.κ, self.α, self.β, self.d, self.m
        real_t = (t + self.t0) if t is not None else self.t
        # shrink domain by eps to avoid infs close to boundary
        radius = (torch.sqrt(C/κ)*torch.pow(real_t, β)).item() - eps
        if x.ndim == 1:
            x = x.view(1,-1)
        correct_dim = torch.ones(1)*(x.shape[1] == d)
        correct_mag = torch.norm(x, dim=1) < radius
        return torch.logical_and(correct_mag, correct_dim)

    def diffusion_value(self, x, t=None):
        C, κ, α, β, d, m = self.C, self.κ, self.α, self.β, self.d, self.m
        real_t = (t + self.t0) if t is not None else self.t
        core = κ * x.view(-1, d).pow(2).sum(1) * torch.pow(real_t, -2*β)
        M = torch.relu(C - core)
        #Delta_ρ_m = M**(1/(m-1))*t**(-2*β - α*m)*(-α + M**(-1)*(2*κ*β*t**(-2*β))** x.view(-1, d).pow(2).sum(1))
        Delta_ρ_m = M**(1/(m-1)) * torch.pow(t,-2*β-α*m) * ( torch.pow(M, -1)*2*core*β/(m-1.0) - α )
        return Delta_ρ_m

    def partial_t(self, x, t=None):
        C, κ, α, β, d, m = self.C, self.κ, self.α, self.β, self.d, self.m
        real_t = torch.tensor(t + self.t0) if t is not None else torch.tensor(self.t)
        core = κ * x.view(-1, d).pow(2).sum(1) * torch.pow(real_t, -2*β)
        #print( torch.any(C-core < 0))
        M = torch.relu(C - core)
        #M = torch.relu(C - κ * x.view(-1, d).pow(2).sum(1) * real_t**(-2*β) )
        #dρ_dt = M**(1./(m-1.))*t**(-α-1)*(2*β*κ*x.view(-1, d).pow(2).sum(1)/(m-1)*t**(-2*β)*M**(-1) - α)
        dρ_dt = M**(1/(m-1)) * torch.pow(t,-α-1.0) * ( torch.pow(M, -1)*2*core*β/(m-1.0) - α )
        return dρ_dt

    def log_prob(self, x, t=None):
        real_t = (t + self.t0) if t is not None else self.t
        M = torch.relu(self.C - self.κ * x.view(-1, self.d).pow(2).sum(1) * real_t**(-2*self.β) )
        logp = torch.log(M)/(self.m-1.0) - torch.log(real_t)*self.α
        return logp.view(-1,)#.view(x.shape)

    def log_prob_asymptotic(self, x, t=None, eps=1e-6):
        """ Log density of asymptotic solution as t->∞:
                ρ(x,t) = 1/Z * t^(-α) * C^(1/(m-1)) * Indicator[ ‖x‖^2 < Ct^{2β}/κ  ]

            Partition function:
                Z = 1/M * int(ρ(x,t)) = M^{-1} * t^(-α) * C^(1/(m-1)) * Vol_d(r=(C/K)^(1/2) * t^{β} )
            where M is the pre-defined total mass of the density (Carrillo et al take M = 2)

            So:
               ρ(x,t) = M/Vol_d(r=(C/K)^(1/2) * t^{β} )
                      = (M * Γ(d/2 + 1)) / (π^(d/2) * r^d)
               log ρ(x,t) = log(M) + log(Γ(d/2 + 1)) - dlog(π)/2 - d(0.5*log(C) - 0.5log(K) + β*log(t))

            In 1D, this can be simplified as:
               ρ(x,t) = M/(2r) = M * 0.5 * (C/K)^(-1/2) * t^{-β}
               log ρ(x,t) = -β*log(t) - 0.5*log(C) + 0.5*log(K) + log(M/2)

        """
        C, κ, α, β, d, m, M = self.C, self.κ, self.α, self.β, self.d, self.m, self.M
        real_t = (t + self.t0) if t is not None else self.t
        if self.d == 1:
            # FIXME: should we use real_t here?
            min, Max = self.get_domain(t) if t is not None else self.domain
            mask = torch.ones_like(x).masked_fill_(((x <= min) | (x >= Max)), float('-inf'))
        else: # THis could be used for 1D too
            b = self.in_domain(x, t)
            mask = torch.ones(x.shape[0]).masked_fill_(~self.in_domain(x, t), float('-inf'))

        #log_Z = (self.α-self.β)*real_t + 2*self.C*(self.m - 1.)/(m+1.) + 0.5*self.κ
        #logp = (-self.α*real_t + self.C/(self.m - 1.0))*mask
        #logp = (-self.β*real_t + self.C*(2*self.m**2-3*self.m+3)/(self.m**2 - 1.0))*mask
        ### I had this before, for submission version in 1D, it's missing a term that is zero when M=2, np.log(0.5):
        logp = (-β*torch.log(real_t) - 0.5*torch.log(C) + 0.5*torch.log(κ))*mask
        #print(logp + np.log(M/2))
        ### But I think this one is correct
        radius = (torch.sqrt(C/κ)*torch.pow(real_t, β)).item() - eps
        vol_dball = ((math.pi**(d/2))/sp.special.gamma(d/2+1)) * (radius**d)
        #print(f'r={radius:4.2f}, vol(r)={vol_dball:4.2f}')
        #logp = (1/vol_dball)*mask
        logp = (np.log(M) + np.log(sp.special.gamma(d/2+1)) - d*np.log(math.pi)/2 - 0.5*d*torch.log(C) + 0.5*d*torch.log(κ) - β*d*torch.log(real_t))*mask
        return logp.view(-1,)#.view(x.shape)

    def sample(self, sample_size=None):
        d = self.d
        if d == 1:
            proposal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            X = rejection_sampler(self, sample_size, -1, 1)
        else:
            proposal = torch.distributions.MultivariateNormal(torch.zeros(d), scale_tril=torch.diag(torch.ones(d)))
            X = rejection_sampler(self, sample_size, -1, 1, proposal=proposal, verbose=False)
        return X.reshape(*sample_size, d)
