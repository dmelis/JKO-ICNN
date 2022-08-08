import numpy as np
import torch
import torch.nn as nn
import pdb
from geomloss import SamplesLoss

from cpflow.lib.logdet_estimators import stochastic_logdet_gradient_estimator, unbiased_logdet, stochastic_lanczos_quadrature
from .utils import sample_rademacher, meshgrid_from

OVERRIDE_NANS_WARNING = True


class Functional(nn.Module):
    """ Parent class for functionals over measures.

        Attributes:
            _requires_x (bool): whether F requires input x for its computation
            _requires_u (bool): whether F requires function u for its computation
            _requires_g (bool): whether F requires gradient g for its computation
            _requires_hvp (bool): whether F requires hessian-vector product callable
                for its computation
            _requires_H (bool): whether F requires full hessian H for its computation

        Methods:
            header_str: produces formatted string for optimization log header
            update_str: produces formatted string for optimization log iterates

    """
    def __init__(self, *args, **kwargs):
        super(Functional, self).__init__()
        self._requires_x   = False
        self._requires_u   = False
        self._requires_g   = False
        self._requires_hvp = False
        self._requires_H   = False
        self._allows_exact = False

    def header_str(self):
        pass

    def update_str(self):
        pass

    def update_composition(self, *args, **kwargs):
        pass

    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=False):
        """ gradient_eval = True triggers special behaviors for the inner-loop
            calls of the functionals, where we're interested in gradients more
            than exact evaluation of F.
        """
        pass

    def exact_eval(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

class FunctionalSum(Functional):
    """ Sum of functionals """
    def __init__(self, funs, weights=None, *args, **kwargs):
        super(FunctionalSum, self).__init__(*args, **kwargs)
        self.funs = funs
        if weights is None:
            weights = [1]*len(self.funs)
        self.weights = weights
        assert len(funs) == len(weights)
        # Add todo: check for weight len
        self._requires_x = any([f._requires_x for f in self.funs])
        self._requires_u = any([f._requires_u for f in self.funs])
        self._requires_g = any([f._requires_g for f in self.funs])
        self._requires_hvp = any([f._requires_hvp for f in self.funs])
        self._requires_H = any([f._requires_H for f in self.funs])
        self._allows_exact = True # will pass on decision to submodules
        self._F_ρ = None

    def header_str(self):
        s = '{:>8}'.format('F(ρ)= ')
        for i,f in enumerate(self.funs):
            s += '   + ' if i > 0 else '     '
            s += f.header_str().replace('F(ρ)=', '')
        return s

    def update_str(self):
        s = f'{self._F_ρ:8.2f}'
        for f in self.funs:
            s += f.update_str()
        return s

    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=True):
        total=0
        for i,f in enumerate(self.funs):
            total += self.weights[i]*f(x,u,g,hvp,H,gradient_eval)
        self._F_ρ = total
        return total

    def exact_eval(self, *args, **kwargs):
        total=0
        for i,f in enumerate(self.funs):
            if f._allows_exact:
                total += self.weights[i]*f.exact_eval(*args, **kwargs)
        return total

    def reset(self, *args, **kwargs):
        for i,f in enumerate(self.funs):
            f.reset(*args, **kwargs)

class PotentialFunctional(Functional):
    """ A functional of the form:
            F(ρ) = ∫V(x)dρ(x)
        The gradient flow of this functional corresponds to an advection PDE:
            ∂ρ/∂t = ∇·(ρ ∇V)
    """
    def __init__(self, V, *args, **kwargs):
        super(PotentialFunctional, self).__init__(*args, **kwargs)
        self.V = V
        self._requires_g = True
        self._F_ρ = None
        self._allows_exact = True

    def header_str(self):
        return '{:>10}'.format('F(ρ)=𝒱(ρ)')

    def update_str(self):
        return f'{self._F_ρ:10.2e}'

    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=True):
        out = self.V(g).mean()  # V(G(X))
        self._F_ρ = out.item()
        return out

    def exact_eval(self, x, ρ=None, ρx=None, **kwargs):
        " Exact evaluation of functional when given the analytic density """
        # manual integration: self.f(torch.exp(ρ.log_prob(x))).mean()
        # torch integration:
        if ρx is not None:
            y = self.V(x) * ρx
        else:
            y = self.V(x) * torch.exp(ρ.log_prob(x))
        integral = torch.trapz(y.squeeze(), x.squeeze())
        return integral

    def reset(self,  *args, **kwargs):
        self._F_ρ = None


class InteractionFunctional(Functional):
    """ A functional of the form:
            F(ρ) = ∫W(x-y)dρ(x)
        The gradient flow of this functional corresponds to an advection PDE:
            ∂ρ/∂t = ∇·(ρ ∇W * ρ)
    """
    def __init__(self, W, *args, **kwargs):
        super(InteractionFunctional, self).__init__(*args, **kwargs)
        self.W = W
        self._requires_g = True
        self._F_ρ = None
        self._allows_exact = True

    def header_str(self):
        return '{:>10}'.format('F(ρ)=𝒲(ρ)')

    def update_str(self):
        return f'{self._F_ρ:10.2e}'

    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=True):
        # Split samples into two batches
        idx = torch.randperm(x.shape[0])
        idx1,idx2 = idx[:int(x.shape[0]/2)], idx[int(x.shape[0]/2):]

        ## Method 1: single avergage - cruder approximation, faster
        # assert x.shape[0] % 2 == 0, 'single expectation method works for even-sixed samples only'
        # out = self.W(g[idx1,:] - g[idx2,:]).mean() * 0.5

        ## Method 2: iterated averages - works even for odd sample size
        G = (g[idx1,:].unsqueeze(1) - g[idx2,:])   # n1 x n2 x d, Gij = xi - xj
        out = self.W(G).squeeze().mean() * 0.5

        self._F_ρ = out.item()
        return out

    def sample_eval(self, X, ρ=None, eval_points=100, **kwargs):
        # Method 1: Double Expectation
        idx = torch.randperm(X.shape[0])
        idx1,idx2 = idx[:int(X.shape[0]/2)], idx[int(X.shape[0]/2):]
        G = (X[idx1,:].unsqueeze(1) - X[idx2,:])   # n1 x n2 x d, Gij = xi - xj
        expectation = self.W(G).squeeze().mean()
        return expectation * 0.5


    def exact_eval(self, x, ρ=None, ρx=None, **kwargs):
        " Exact evaluation via integration, when given the analytic density """
        xv, yv = torch.meshgrid([x.squeeze(),x.squeeze()])
        if ρx is None:
            ρx = torch.exp(ρ.log_prob(x)).squeeze()
        else:
            ρx = ρx.squeeze()

        # Method 1:
        # z = self.W(xv - yv) * ρx
        # integral = torch.trapz((torch.trapz(z, x.squeeze(), dim=0) * ρx).squeeze(), x.squeeze())

        # Method 2: (does seem equivalent)
        Z = self.W((xv - yv).unsqueeze(-1)).fill_diagonal_(0) * (torch.outer(ρx,ρx))
        # Need to replace inf->0 in diagonal for the functional we're using because it thakes log of difference

        integral = torch.trapz(torch.trapz(Z, x.squeeze(), dim=0).squeeze(), x.squeeze())

        return integral * 0.5

class InternalEnergyFunctional(Functional):
    """ A functional of the form:
            F(ρ) = ∫f(ρ(x))dx
    """
    def __init__(self, f, ρ, X=None, multfun = None, cg_iters=50, m_lanczos=10, *args, **kwargs):
        super(InternalEnergyFunctional, self).__init__(*args, **kwargs)
        self._allows_exact = True
        self.f = f
        self.multfun = multfun
        self.ρ0 = ρ
        self.X0 = X
        self.m_lanczos = m_lanczos
        self.cg_iters = cg_iters
        self._requires_x = True
        self._requires_hvp = True
        self.bsz, *dims = X.shape
        self.dim = np.prod(dims)
        #self._requires_H = (self.dim <= 1) # Only needed if using Method 1 for logdet
        self.logdet_composition = torch.zeros(self.bsz)
        if hasattr(self.ρ0, 'reset'): self.ρ0.reset()
        if X is not None:
            self.logprob_X0 = self.ρ0.log_prob(X.detach())
        else:
            self.logprob_X0 = None
        self._F_ρ = None


    def reset(self, X=None, *args, **kwargs):
        if X is not None: #new sample
            print('here')
            self.X0 = X
            self.bsz, *dims = X.shape
        self.logdet_composition = torch.zeros(self.bsz)
        if hasattr(self.ρ0, 'reset'): self.ρ0.reset()
        self.logprob_X0 = self.ρ0.log_prob(self.X0.detach())
        self._F_ρ = None

    def header_str(self):
        return '{:>10}'.format('F(ρ)=ℱ(ρ)')

    def update_str(self):
        return f'{self._F_ρ:10.2f}'

    def update_composition(self, logdet):
        self.logdet_composition += logdet.detach()

    #def eval_update(self, x=None, u=None, g=None, hvp=None, H=None):

    # def exact_eval(self, x, ρ=None, ρx=None, **kwargs):
    #     " Exact evaluation of functional when given the analytic density """
    #     # manual integration: self.f(torch.exp(ρ.log_prob(x))).mean()
    #     # torch integration:
    #     if ρx is not None:
    #         y = self.f(ρx)
    #     else:
    #         y = self.f(torch.exp(ρ.log_prob(x)))
    #     integral = torch.trapz(y.squeeze(), x.squeeze())
    #     return integral

    def exact_eval(self, x=None, ρ=None, ρx=None, **kwargs):
        """ Exact evaluation of functional via integral ∫ρ(x)log ρ(x)dx
            - requires having analytic density ρ or evaluated at grid ρx
            - only feasible in 1D or 2D

            Takes either x,ρ or x,ρx as args. If given x, is assumed to be a
            uniform grid.
        """
        #pdb.set_trace()
        assert not (ρ is None and ρx is None)
        use_deltas=True
        with torch.no_grad():
            if ρx is not None:
                fx = self.f(ρx)
            else:
                if x is None:
                    x, npoints = meshgrid_from(ρ=ρ)

                if x.ndim == 1:
                    x = x.view(-1, 1)
                if hasattr(ρ, 'log_prob'):
                    fx = self.f(torch.exp(ρ.log_prob(x)))
                else: # last ditch attempted - use directly as callable
                    fx = self.f(torch.exp(ρ(x)))

            fx = fx.view(x.shape)
            assert x.ndim == 2, 'x should be n x d'
            assert fx.ndim == 2, 'fx should be n x d'
            ## Filter out nans - these are zero mass points, don't add to integral
            _x  = x[~torch.isinf(fx).squeeze(),:]
            _fx = fx[~torch.isinf(fx).squeeze(),:]

            def get_delta(u):
                a, b = torch.unique(u)[:2]
                return b-a
            dxs = [get_delta(_x[:,i]) for i in range(x.shape[1])]


            if x.shape[1] == 1: #1D
                if use_deltas:
                    integral = torch.trapz(_fx.squeeze(), dx=dxs[0])
                else:
                    pdb.set_trace()
                    integral = torch.trapz(_fx.squeeze(), x.squeeze())
            elif x.shape[1] == 2: #2D
                if use_deltas:
                    integral = torch.trapz(torch.trapz(_fx.reshape(npoints, npoints), dx=dxs[0], dim=0), dx=dxs[1], dim=0)
                else:
                    pdb.set_trace(header='Not implemented yet')
            else:
                raise NotImplementedError()
        return integral

    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=True):
        bsz, *dims = x.shape
        dim = np.prod(dims)
        update_cache = not gradient_eval
        ### Use direct objective for the theta-independent multiplier
        with torch.no_grad():
            #pdb.set_trace()
            if self.dim > 1:
                # vector must be normalized for Qr decomposition in lanczos_tridiagonalization
                v1 = torch.nn.functional.normalize(sample_rademacher(bsz, dim)).to(x)
                #logdet = unbiased_logdet(hvp, v, p=0.1, n_exact_terms=4)
                logdet = stochastic_lanczos_quadrature(hvp, v1, m=max(2,min(self.m_lanczos,dim)))
                #logdet = stochastic_logdet_gradient_estimator(hvp, v, self.cg_iters, rtol=0.0, atol=1e-6)
            else:
                # In 1D, H is a scalar, so lodget(H) = log abs(H), no need to estimate
                # Method 1: via actual hessian
                #logdet = torch.log(H.reshape(bsz,))
                # Method 2: via hvp
                logdet = torch.log(hvp(torch.ones(bsz)).squeeze())
                if torch.isnan(logdet).any():
                    if not OVERRIDE_NANS_WARNING: pdb.set_trace(header='nans in logdet')

            assert self.logprob_X0.shape == logdet.shape, print(self.logprob_X0.shape, logdet.shape)
            logratio = self.logprob_X0 - self.logdet_composition - logdet
            ratio = torch.exp(logratio)

        if self.multfun is None:
            # We use the .sum() trick: d/dx_i f(x_i) = d/d_xi sum_i(f(x))
            _ratio = ratio.detach().clone().requires_grad_(True)
            f_p = torch.autograd.grad(self.f(_ratio).sum(), _ratio, create_graph=True)[0]
            multiplier = f_p - self.f(ratio)/ratio
        else:
            multiplier = self.multfun(logratio)


        if torch.isnan(multiplier).any():
            print('Nans in multiplier')
            if not OVERRIDE_NANS_WARNING: pdb.set_trace()

        ### Now get gradient with conjugate_gradient method on surrogate objective
        if self.dim > 1:
            v = sample_rademacher(bsz, dim).to(x)
            logdet_estim = stochastic_logdet_gradient_estimator(hvp, v, self.cg_iters, rtol=0.0, atol=1e-3)
        else:
            #logdet_estim = torch.log(H.reshape(bsz,1))
            logdet_estim = torch.log(hvp(torch.ones(bsz)).squeeze())

        surrogate_loss = -multiplier*logdet_estim
        F_ρ = surrogate_loss.mean()

        # #ratio = self.density/torch.exp(self.logdet_composition + logdet)
        # logratio = self.logprob_X0 - self.logdet_composition - logdet
        # ratio = torch.exp(logratio)
        # F_ρ = (self.f(ratio)/ratio).mean()

        if torch.isnan(F_ρ).any():
            print('Nans in F_ρ: ')
            if torch.isnan(logdet_estim).any(): print('Nans in logdet estim')
            if torch.isnan(logratio).any(): print('Nans in multiplier')
            if torch.isnan(multiplier).any(): print('Nans in multiplier')
            if not OVERRIDE_NANS_WARNING: pdb.set_trace()

        self._F_ρ = F_ρ.item()

        if update_cache:
            print(f'Updating cache with: {self.logprob_X0[0].item():8.2e}  '
                  f'{self.logdet_composition[0].item():8.2e}  '
                  f'{logdet[0].item():8.2e}  '
                  f'{ratio[0].item():8.2e}  ')
            self.update_composition(logdet)

        return F_ρ


class EntropyFunctional(Functional):
    """ A functional of the form:
            F(ρ) = ∫ρ(x)log ρ(x)dx
        This is a particular case of the internal energy functional with t(y) = y log y
    """
    def __init__(self, ρ0=None, dim=1, cg_iters = 10, m_lanczos=2,
                value_eval_method='logdet', *args, **kwargs):
        super(EntropyFunctional, self).__init__(*args, **kwargs)
        self.cg_iters = cg_iters
        self.m_lanczos = m_lanczos
        self._requires_x = True
        self._requires_hvp = True
        #self.compute_exact = compute_exact
        self.value_eval_method = value_eval_method
        #assert not (compute_exact and ρ0 is None), "exact computation of entropy requires ρ0 be provided"
        self.ρ0 = ρ0
        self._allows_exact = ( ρ0 is not None ) and (dim <= 2)
        assert not (value_eval_method == 'logdet' and ρ0 is None)
        self._F_ρ = None      # Value of true objective
        self._F_hat_ρ = None  # Value of surrogate objective
        if self._allows_exact and value_eval_method == 'exact':
            #domain = ρ0.get_domain() if hasattr(ρ0, 'get_domain') else (-2,2a)
            #x = meshgrid_from(ρ=ρ0)
            self._F_ρ = self.exact_eval(ρ=ρ0)
        else:
            ### FIXME: is there anything we can do here?
            self._F_ρ = 0


    def reset(self, *args, **kwargs):
        self._F_ρ = None      # Value of true objective
        if self._allows_exact and self.value_eval_method == 'exact':
            #domain = ρ0.get_domain() if hasattr(ρ0, 'get_domain') else (-2,2a)
            #x = meshgrid_from(ρ=ρ0)
            self._F_ρ = self.exact_eval(ρ=self.ρ0)
        else:
            ### FIXME: is there anything we can do here?
            self._F_ρ = 0
        self._F_hat_ρ = None  # Value of surrogate objective

    def header_str(self):
        return '{:>12}'.format('F(ρ)=ℱ*(ρ)')

    def update_str(self):
        return f'{self._F_hat_ρ:12.2e}'

    def exact_eval(self, x=None, ρ=None, ρx=None, **kwargs):
        """ Exact evaluation of functional via integral ∫ρ(x)log ρ(x)dx
            - requires having analytic density ρ or evaluated at grid ρx
            - only feasible in 1D or 2D

            Takes either x,ρ or x,ρx as args. If given x, is assumed to be a
            uniform grid.
        """
        assert not (ρ is None and ρx is None)
        use_deltas=True
        with torch.no_grad():
            if ρx is not None:
                fx = ρx*torch.log(ρx)
            else:
                if x is None:
                    x, npoints = meshgrid_from(ρ=ρ)
                logp = ρ.log_prob(x)
                fx = torch.exp(logp) * logp

            def get_delta(u):
                a, b = torch.unique(u)[:2]
                return b-a
            dxs = [get_delta(x[:,i]) for i in range(x.shape[1])]

            if x.shape[1] == 1: #1D
                if use_deltas:
                    integral = torch.trapz(fx.squeeze(), dx=dxs[0])
                else:
                    pdb.set_trace()
                    integral = torch.trapz(fx.squeeze(), x.squeeze())
            elif x.shape[1] == 2: #2D
                if use_deltas:
                    integral = torch.trapz(torch.trapz(fx.reshape(npoints, npoints), dx=dxs[0], dim=0), dx=dxs[1], dim=0)
                else:
                    pdb.set_trace(header='Not implemented yet')
            else:
                raise NotImplementedError()

        return integral


    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=True):
        bsz, *dims = x.shape
        dim = np.prod(dims)

        if gradient_eval:
            assert hvp is not None, 'Need to pass hvp when gradient_eval=True'
            #print('here')
            ## For the loss to backprop, we use the surrogate objective
            v = sample_rademacher(bsz, dim).to(x)
            logdet = stochastic_logdet_gradient_estimator(hvp, v, self.cg_iters, rtol=0.0, atol=1e-3)
            value = -logdet.mean()
            #pdb.set_trace()
            #print(value.item())
        elif self.value_eval_method == 'logdet' or not self._allows_exact:
            ## For debugging, we use the unbiased log dest estimater + complete objective
            # F(ρ_{t+1}) = F(ρ_t) - E[logdet]
            with torch.no_grad():
                if dim > 1:
                    # vector must be normalized for Qr decomposition in lanczos_tridiagonalization
                    v1 = torch.nn.functional.normalize(sample_rademacher(bsz, dim)).to(x)
                    #logdet = unbiased_logdet(hvp, v, p=0.1, n_exact_terms=4)
                    logdet_exact = stochastic_lanczos_quadrature(hvp, v1, m=max(2,min(self.m_lanczos,dim)))
                    #logdet = stochastic_logdet_gradient_estimator(hvp, v, self.cg_iters, rtol=0.0, atol=1e-6)
                else:
                    # In 1D, H is a scalar, so lodget(H) = log abs(H), no need to estimate
                    # Method 1: via actual hessian
                    #logdet = torch.log(H.reshape(bsz,))
                    # Method 2: via hvp
                    logdet_exact = torch.log(hvp(torch.ones(bsz)).squeeze())
                value = self._F_ρ - logdet_exact.mean()
                self._F_ρ = value # Not that we don't update this value in gradient evaluations
        elif self.value_eval_method == 'exact':
            value = self.exact_eval(x, 'ρt') # FIXME: where do we get ρt from? maybe self.flow.ρt?
            # Should we add it as anothar potential arg to Functional forward calls?
        else:
            raise NotImplementedError('Port from InternalEnergyFunctional')

        self._F_hat_ρ = value

        return value


class FokkerPlanckFunctional(Functional):
    """ Functional corresponding to the Fokker-Planck equation:
            ∂ρ/∂t = ∆ρ + ∇·(ρ ∇V)
        This PDE arises as a gradient flow of the functional:
            F(ρ) = ℱ(ρ) + 𝒱(ρ) = ∫ρ(x)log ρ(x)dx + ∫V(x)dρ(x)
    """
    def __init__(self, V, *args, **kwargs):
        super(FokkerPlanckFunctional, self).__init__(*args, **kwargs)
        self.𝒱 = PotentialFunctional(V)
        self.ℱ = EntropyFunctional()
        self._requires_g = True    # needed by 𝒱
        self._requires_x = True    # needed by ℱ
        self._requires_hvp = True  # needed by ℱ
        self._F_ρ = None
        self._cV_ρ = None
        self._cF_ρ = None

    def header_str(self):
        return '{:>8} {:>8} {:>8}'.format('F(ρ)=','ℱ(ρ)','+   𝒱(ρ)')

    def update_str(self):
        return f'{self._F_ρ:8.2f} {self._cF_ρ:8.2f} {self._cV_ρ:8.2f}'

    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=True):
        ℱ_ρ = self.ℱ(x, u, g, hvp, H)
        𝒱_ρ = self.𝒱(x, u, g, hvp, H)
        self._cF_ρ = ℱ_ρ.item()
        self._cV_ρ = 𝒱_ρ.item()
        self._F_ρ  = self._cF_ρ + self._cV_ρ
        #pdb.set_trace()
        return ℱ_ρ + 𝒱_ρ

class DiffAdvIntFunctional(Functional):
    """ Functional corresponding to the general diffusion-advection-interaction PDE:
            ∂ρ/∂t = ∇·(ρ [∇(f'(ρ)) + ∇V + (∇W)*ρ] )
        This PDE arises as a gradient flow of the functional:
            F(ρ) = ℱ(ρ) + 𝒱(ρ) + 𝒲(ρ) = ∫f(ρ(x))dx + ∫V(x)dρ(x) + ½∫∫W(x-y)dρ(x)dρ(y)
    """
    def __init__(self, f, V, W, *args, **kwargs):
        super(DiffAdvIntFunctional, self).__init__(*args, **kwargs)
        self.ℱ = InternalEnergyFunctional(f)
        self.𝒱 = PotentialFunctional(V)
        self.𝒲 = InteractionFunctional(W)
        self._requires_g = True    # needed by 𝒱
        self._requires_x = True    # needed by ℱ
        self._requires_hvp = True  # needed by ℱ
        # TODO: What does W need?


    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=True):
        ℱ_ρ = self.ℱ(x, u, g, hvp, H)
        𝒱_ρ = self.𝒱(x, u, g, hvp, H)
        𝒲_ρ = self.𝒲(x, u, g, hvp, H)
        out = ℱ_ρ + 𝒱_ρ + 𝒲_ρ
        # TODO MAYBE: use FunctionalSum
        return out


class SinkhornDivergenceFunctional(Functional):
    """ A functional of the form:
            F(ρ) = SinkornDivergence(ρ, β)
        where β is a target (fixed) measure.
     """
    def __init__(self, Y, loss='sinkhorn', p=2, blur=.05, debias=True, *args, **kwargs):
        super(SinkhornDivergenceFunctional, self).__init__(*args, **kwargs)
        assert loss in ['sinkhorn', 'energy', 'laplacian', 'gaussian']
        self.p = p
        self.debias = debias
        self.blur = blur
        # TODO: have a switch between SamplesLoss and ImagesLoss once the latter is added to the geomloss repo
        self.loss = SamplesLoss(loss, p=p, blur=blur, debias=debias)
        # blur = (1/e)**p
        ### DECIDE: Can do partial instead to avoid explicitly coping Y
        #self.loss = partial(SamplesLoss(loss, p=p, blur=blur, debias=debias), Y)
        self.Y    = Y
        self._requires_g = True
        self._F_ρ = None

    def header_str(self):
        return '{:>12}'.format('F(ρ)=SD(ρ,β)')

    def update_str(self):
        return f'{self._F_ρ:12.2e}'

    def forward(self, x=None, u=None, g=None, hvp=None, H=None, gradient_eval=True):
        bsz, *dims = x.shape
        dim = np.prod(dims)
        sd = self.loss(g.view(g.shape[0], -1), self.Y)
        self._F_ρ = sd.item()
        return sd
