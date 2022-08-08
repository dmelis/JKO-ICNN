import inspect
import types
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from celluloid import Camera
import pdb
import numpy as np
#from tqdm import tqdm
from tqdm.autonotebook import tqdm
import scipy.stats


from .icnn import FICNN, ISCNN
from .utils import net_batch_gradients, invert_potential
from .utils import PositiveWeightClipper
from .utils import infer_input_type, test_weights, test_hessian, test_convexity
from .utils import hessian, show, density_plot_2d
from .callbacks import PlottingCallback, CallbackList, Callback, FunctionalValueCallback

#from cpflow.lib.icnn import
from cpflow.lib.logdet_estimators import stochastic_logdet_gradient_estimator, unbiased_logdet, stochastic_lanczos_quadrature
from .utils import sample_rademacher


from torchvision.utils import make_grid, save_image

from geomloss import SamplesLoss

OVERRIDE_STEP_WARNING = True

class GradFlow():
    """ Main Gradient Flow class.

    An object of this class is instantiated with an intial dataset X, a functional
    objective F, an potentially a target dataset Y.

    Arguments:
        X (tensor): the initial particle positions.
        F (callable): a functional objective to minimize, it must take a tensor
            X_t as input and retrieve a torch scalar.
        Y (tensor, optional): target particle positions (for plots only).
        τ (scalar): the JKO step size parameter. Larger values will yield faster
            flows but might lead to inaccuracies and numerical instability.
        lr (scalar): the learning rate for the inner (JKO) optimization problem.
        opt_iters (int): number of interations for inner (JKO) optimization problem.
        convex_net (nn.Module, optional): an already instantiated neural net to be
            used in the inner optimization problem (otherwise an FICNN is used).
        f_input_type (str): DEPRECATED
        animate (bool): whether to produce a video animation (othwerise static plots).



    """
    def __init__(self, X, F, Y=None, ρ0=None,
                 ## JKO scheme
                 τ=1e-2,
                 ## Inner optimization
                 optimizer = 'adam',
                 lr=1e-4,
                 momentum = 0.1,
                 opt_iters = 1000,
                 convex_net = None,
                 strongify_icnn=False,
                 warmopt = True,
                 inner_print_freq = 100,
                 ## Misc
                 save_path=None, device='cpu',
                 callbacks=PlottingCallback(),
                 ## Plotting
                 figsize=None,
                 animate=True, save_format='png',
                 show_trajectories=True,
                 trajectory_length=20,
                 show_density=True,
                 density_type='2d',
                 plot_pad = 0.5,
                 xrng = None,
                 yrng = None,
                 check_for_divergence=False
                 ):
        ### Process and Save Inputs
        ### Do we actually need these ones here or only in plotting callback?
        self.X0, self.input_type, self.channels = infer_input_type(X)
        self.n, *dims = self.X0.shape
        self.dims = dims
        self.flat_dim = np.prod(dims)
        self.Y = Y.detach().clone() if Y is not None else None
        self.F = F
        self.ρ0 = ρ0
        self.τ = τ
        ### Args for inner optimization
        self.convex_net = convex_net
        self.strongify_icnn = strongify_icnn
        self.optimizer = optimizer
        self.lr  = lr
        self.momentum = momentum
        self.opt_iters = opt_iters
        self.warmopt = warmopt
        self.inner_print_freq = inner_print_freq
        ### Plotting
        self.animate = animate
        self.show_trajectories = (self.input_type == 'features') and show_trajectories
        self.show_density      = show_density
        self.density_type      = density_type
        self.plot_pad = plot_pad
        self.xrng = xrng
        self.yrng = yrng
        if figsize is None:
            self.figsize = (6,4) if not self.animate else (10,7)
        else:
            self.figsize = figsize
        self.fig, self.ax = None, None
        ### Book-keeping
        self.times = None
        self.collect_composition = True
        self.f_hist = []
        self.X_hist = []
        # if self.show_trajectories:
        #     self.X_hist = self.X0.unsqueeze(-1).cpu().float() # time will be last dim
        self.trajectory_length = trajectory_length
        if self.collect_composition:
            self.u_1_t = torch.nn.Sequential()
            self.u_hist = []
        ### Saving
        self.save_path = save_path
        self.save_format = save_format
        ### GPU-related
        self.device = torch.device(device)
        ### Misc
        self.abort_ = False # will be triggered if something goes awry during flow
        self.check_for_divergence = check_for_divergence  # flag to turn on and off checking for divergence abort
        # Callbacks cannot be empty - if None, replace by dummy Callback
        self.callbacks = callbacks if callbacks is not None else Callback()
        self.callbacks._inherit_from_flow(self)

    def print_step_problem_header(self):
        #print('{:4} {:8} {:8} {:8} {:8} {:8}'.format('It', 'Disp Loss','Func Loss','l+','||X||','||T(X)||'))
        header = '{:4} {:>10}'.format('Iter', 'W(ρ,ρ_t)')
        if hasattr(self.F, 'header_str'):
            header += self.F.header_str()
        else:
            header += '{:>8}'.format('F(ρ)')
        header += '{:>8} {:>8} {:>10}'.format('‖X‖','‖T(X)‖', '‖ΔT(X)‖')
        print('-'*(len(header)+1))
        print(header)
        print('-'*(len(header)+1))

    def solve_step_problem(self, x, print_every=200, warm_u=None, tol = 1e-4):
        """ Inner optimization problem solver.

            For a given set of particles X, seeks to find:

                    argmin_{u}  1/(2τ) * ‖∇u(X)-X‖^2  + F(∇u(X))
        """
        bsz, *dims = x.shape
        dim = np.prod(dims)

        if self.convex_net is None:
            u = FICNN(input_dim=dims[0], hidden_dims=[100, 100], nonlin='leaky_relu', dropout=0).to(x)#.to(self.device)
        elif isinstance(self.convex_net, types.FunctionType):
            # factory function was passed
            u = self.convex_net().to(x)
        elif inspect.isclass(self.convex_net) and issubclass(self.convex_net, torch.nn.Module):
            # net class was passed
            u = self.convex_net(dim=dims[0], dimh=100, num_hidden_layers=2).to(x)

        # FIXME: do we really need to have these be attributes of net?
        u.input_dim  = dims[0] if not hasattr(u, 'input_dim') else u.input_dim
        u.output_dim = 1 if not hasattr(u, 'output_dim') else u.output_dim

        if warm_u is not None: # Maybe just use warm_u directly??
            init_net = warm_u.icnn if hasattr(warm_u, 'icnn') else warm_u
            u.load_state_dict(init_net.state_dict())
            #u.load_state__dict = init_net.state_dict()


        if self.strongify_icnn:
            if type(self.strongify_icnn) is tuple:
                step = len(self.f_hist) * 1.0
                ζ = self.strongify_icnn[0] * (self.strongify_icnn[1]**step)
                #print('here', self.f_hist, (self.strongify_icnn[0]**step))
            else:
                ζ = self.strongify_icnn
            print(f'ζ = {ζ}')
            u = ISCNN(u, strength=ζ)

        def param_stats(module):
            if hasattr(module, 'weight_z') and module.weight_z is not None:
                print(module.weight_z.min().item(), module.weight_z.max().item())

        test_weights(u)
        #test_convexity(u,  convtol = 0.0)

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(u.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(u.parameters(), lr=self.lr)#, momentum=self.momentum)
        else:
            raise ValueError()

        clipper = PositiveWeightClipper()


        self.print_step_problem_header()

        Tx_old = x.clone().detach()

        ####################### Inner Optimization Loop ########################
        for i in range(self.opt_iters+1):
            optimizer.zero_grad()

            ### TODO MAYBE: write a wrapper around input preparation
            def eval_F(self, ):
                Tx = ...

            ### TODO: not sure this is necessary
            x_ = x.detach().clone().requires_grad_(True)


            ################# Compute Gradient T(x) = ∇_x u(x) #################
            ### TODO: Decide between these two. Compare. Check for Nans etc.
            ### Both methods seem to give same result, except one is more prone
            ### to NaNs, and currently Tx1 has a broken computational graph.
            # Method 1.
            Tx1 = net_batch_gradients(u, x_, detach=False)
            #print(x_.requires_grad, u.training)
            # # Method 2.
            # ux = u(x_)
            # Tx2 = torch.autograd.grad(ux.sum(), x_, create_graph=True)[0]
            # try:
            #     assert torch.all(torch.eq(Tx1, Tx2)), 'Not equal'
            # except:
            #     if torch.isnan(Tx).any(): print('NaNs in Tx')
            #     if torch.isnan(Tx2).any(): print('NaNs in Tx2')
            # pdb.set_trace()
            Tx = Tx1

            if torch.isnan(Tx).any():
                print('Warning: nans in Tx, early stopping')
                break
            ####################################################################

            w_loss = (Tx-x_).norm(2,dim=1).pow(2).mean() # Should be equiv to (TX-X).pow(2).sum(dim=1)

            f_inputs = {'x': x_, 'g': Tx, 'u': u}
            if self.F._requires_H:
                Hux = torch.stack([torch.autograd.functional.hessian(u, x_[i], create_graph=False, strict=False) for i in range(x.shape[0])])
                f_inputs['H'] = Hux
            if self.F._requires_hvp:
                ######## Instantiate Hessian Vector Product Funtion ################
                # Hess_u(x)
                def hvp_fun(v):
                    """ Returns <v, Hess_u(x)>  """
                    v = v.reshape(bsz, *dims)
                    hvp = torch.autograd.grad(Tx, x_, v, create_graph=True, retain_graph=True)[0]
                    if not torch.isnan(v).any() and torch.isnan(hvp).any():
                        raise ArithmeticError("v has no nans but hvp has nans.")
                    hvp = hvp.reshape(bsz, dim)
                    return hvp
                ####################################################################
                f_inputs['hvp'] = hvp_fun

            f_loss = self.F(**f_inputs, gradient_eval=True)
            if torch.isnan(f_loss).any():
                print('Nans in f_loss! Aborting flow.')
                self.abort_ = True
                return None, None, None
                #break

            ΔT = (Tx - Tx_old).norm(2, dim=1).mean()
            Tx_old = Tx.clone().detach()

            ## Check relative ratio between objectives
            ratio = (np.abs(f_loss.item())*2.0*self.τ)/w_loss.item()
            # TODO: what to do about negative f_loss here?
            ideal_range = np.array([np.floor(np.log10(self.τ/ratio)), np.ceil(np.log10(self.τ/ratio))])
            if (len(self.f_hist) == 0) and (i == 0) and (ratio > 1e1 or ratio < 1e-1):
                action = 'decreasing' if (ratio > 1e1) else 'increasing'
                print('Relative ratio between F and W terms in JKO is too ' \
                      f'unbalanced...\nconsider choosing τ in 10^[{ideal_range[0]},{ideal_range[1]}] to balance them.')
                if not OVERRIDE_STEP_WARNING: pdb.set_trace()

            ## JKO STEP
            loss =  f_loss + w_loss/(2.0*self.τ)
            loss.backward()
            if i % print_every == 0 or (i == self.opt_iters):
                if hasattr(self.F, 'update_str'):
                    f_str = self.F.update_str()
                else:
                    f_str =  f'{f_loss.item():>8.2f}'
                print(f'{i:4} {w_loss.item():>10.2e} ' + f_str + \
                      f' {x_.mean(0).norm().item():>8.2f}'
                      f' {Tx.mean(0).norm().item():>8.2f}'
                      f' {ΔT.item():>10.2e}'
                      )

            optimizer.step()
            u.apply(clipper)
            test_weights(u)
            #test_hessian(u, strict=False)

        if ΔT > tol:
            print('Warning: inner optim. did not converge. Consider increasing opt_iters.')

        ### Final evaluation of F with 'exact' (expensive) value coomputation
        ### and (whenever needed) caching and updates
        #Tx = net_batch_gradients(u, x)
        # Method 2.
        x_ = x.detach().clone().requires_grad_(True)
        ux = u(x_)
        Tx = torch.autograd.grad(ux.sum(), x_, create_graph=True)[0]
        f_inputs = {'x': x_, 'g': Tx, 'u': u}

        if self.F._requires_H:
            Hux = torch.stack([torch.autograd.functional.hessian(u, x_[i], create_graph=False, strict=False) for i in range(x.shape[0])])
            f_inputs['H'] = Hux
        if self.F._requires_hvp:
            ######## Instantiate Hessian Vector Product Funtion ################
            # Hess_u(x)
            def hvp_fun(v):
                """ Returns <v, Hess_u(x)>  """
                v = v.reshape(bsz, *dims)
                hvp = torch.autograd.grad(Tx, x_, v, create_graph=True, retain_graph=True)[0]
                if not torch.isnan(v).any() and torch.isnan(hvp).any():
                    raise ArithmeticError("v has no nans but hvp has nans.")
                hvp = hvp.reshape(bsz, dim)
                return hvp
            ####################################################################
            f_inputs['hvp'] = hvp_fun

        f_val    = self.F(**f_inputs, gradient_eval=False)

        #Tx = net_batch_gradients(u, x)
        return Tx.detach(), u.eval(), f_val.item()

    def step(self, X, iter, warm_u=None):
        TX, u, f = self.solve_step_problem(X, warm_u=warm_u, print_every=self.inner_print_freq)
        self.f_hist.append(f)
        if (TX is not None) and self.show_trajectories:
            self.X_hist = torch.cat([self.X_hist, TX.clone().cpu().float().unsqueeze(-1)], dim=-1)
        if (u is not None) and self.collect_composition:
            self.u_hist.append(u)

        if f is not None:
            print(f'\n---> After JKO Step: {iter}, F(ρ)={f:8.4e} <---\n')

        if self.check_for_divergence and (f - self.f_hist[0]) > 2*np.abs(self.f_hist[0]):
            print('Flow seems to be diverging, aborting')
            self.abort_ = True

        return TX,u,f

    def concat_mapping(self, x, return_init=False):
        """ Computes the concatenation of Monge maps:
                T_1:t(x) = (∇u_t ○ ... ○ ∇u_1) (x)
        """
        #pdb.set_trace(header='JK')
        x_0 = x.detach().clone().requires_grad_(True)
        Tx_i = x_0.clone()
        for i,u in enumerate(self.u_hist,1):
            # Method 1
            #Tx_i = torch.autograd.grad(u(Tx_i).sum(), Tx_i, create_graph=True)[0]
            # Method 2
            #print(i, Tx_i.requires_grad, u.training)
            #pdb.set_trace()
            Tx_i = net_batch_gradients(u, Tx_i, detach=False)
            #print(Tx_i.requires_grad)

        if return_init:
            return Tx_i, x_0
        else:
            return Tx_i

    def inverse_map(self, y,  return_init=False):
        y0 = y.clone().requires_grad_(True)
        T_inv_yi = y0.clone()
        for t, u in zip(reversed(range(len(self.u_hist))), reversed(self.u_hist)):
            T_inv_yi = invert_potential(u, T_inv_yi, max_iter=1000000, lr=1.0, tol=1e-12, history_size=500, verbose=True)#, x=None, context=None, **kwargs):

        if return_init:
            return T_inv_yi, y0
        else:
            return T_inv_yi


    # TODO MAYBE: Move this to utils? If not, remove the ulist arg.
    def compute_logdet_composition(self, ulist, X):
        bsz, *dims = X.shape
        dim = np.prod(dims)

        TX_t, X_0 = self.concat_mapping(X, return_init=True)

        def hvp(v):
            """ Returns <v, Hess_u(x)>  """
            v = v.reshape(bsz, *dims)
            hvp = torch.autograd.grad(TX_t, X_0, v, create_graph=True, retain_graph=True)[0]
            if not torch.isnan(v).any() and torch.isnan(hvp).any():
                raise ArithmeticError("v has no nans but hvp has nans.")
            hvp = hvp.reshape(bsz, dim)
            return hvp

        with torch.no_grad():
            if dim > 1:
                v = sample_rademacher(bsz, dim).to(X_0)
                #logdet = unbiased_logdet(hvp, v, p=0.01, n_exact_terms=4)
                logdet = stochastic_lanczos_quadrature(hvp, v, m=2, func=torch.log)
            else:
                logdet = torch.log(hvp(torch.ones(bsz)).squeeze())
        return logdet.view(X.shape)

    def estimate_density(self, X_query=None, xrng=None, yrng=None, method='pushforward'):
        """ Estimate density using current particles Xt, evaluated at query
            samples X_query or in a uniform meshgrid.

            The density is estimated as:
                log_p(x) = log(ρ0(x)) - log(|J_T(x)|)
            where J_T is the jacobian of the map:
                T_1:t(x) = (∇u_t ○ ... ○ ∇u_1) (x)
        """
        if X_query is None and self.flat_dim == 2:
            if xrng is None or yrng is None:
                xmin, ymin = [v.item() for v in self.Xt.min(dim=0)[0]]
                xmax, ymax = [v.item() for v in self.Xt.max(dim=0)[0]]
            else:
                xmin, xmax = xrng
                ymin, ymax = yrng
            x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            X_query = torch.from_numpy(np.vstack([x.ravel(), y.ravel()]).T).float()
        elif X_query is None:
            raise ValueError()

        if method == 'pushforward':
            if self.ρ0 is None:
                raise ValueError()
            #self.compute_logdet_composition(self.u_hist, X_query)
            #print(torch.exp(self.ρ0.log_prob(X_query)).mean())
            T_inv_query = self.inverse_map(X_query).detach()
            #pdb.set_trace()
            #print(self.ρ0.log_prob(T_inv_query).shape, self.compute_logdet_composition(self.u_hist, X_query).shape)
            Z = self.ρ0.log_prob(T_inv_query).view(X_query.shape[0], 1) - self.compute_logdet_composition(self.u_hist, X_query).view(X_query.shape[0], 1)
        elif method.lower() == 'kde':
            kernel = scipy.stats.gaussian_kde(self.Xt.T)
            Z = np.reshape(kernel.logpdf(X_query.T), X_query.shape[0])

        return Z

    def reset(self):
        self.f_hist = []
        self.X_hist = []
        self.u_hist = []
        self.Xt = None
        self.ut = None
        # if self.show_trajectories:
        #     self.X_hist = self.X0.unsqueeze(-1).cpu().float() # time will be last dim
        if self.collect_composition:
            self.u_1_t = torch.nn.Sequential()

        self.abort_ = False
        self.callbacks.reset()
        if self.ρ0 is not None and hasattr(self.ρ0, 'reset'):
            self.ρ0.reset()

        self.F.reset()


    def flow(self, steps=10, X0=None):
        """ Steps: outer steps

            X0 can be passed to override initialization sample (useful for multiple
            restarts.)

        """

        X0 = X0 if X0 is not None else self.X0
        Xt = X0.clone()

        if self.show_trajectories:
            self.X_hist = X0.unsqueeze(-1).cpu().float() # time will be last dim

        ## TODO: do we want an initial eval of F at time 0?
        if hasattr(self.F, 'sample_eval'):
            f = self.F.sample_eval(X0.detach().clone())
        else:
            f = None
        self.callbacks.on_flow_begin(Xt, self.Y, f=f, flow=self, steps=steps)

        ut = None
        times =  self.τ*np.arange(1, steps+1)
        pbar = tqdm(times, leave=False)
        for iter,t in enumerate(pbar, 1):
            pbar.set_description(f'JKO Step {iter:3} (time={t:8.4f})')
            self.callbacks.on_step_begin(iter, t)
            Xt, ut, ft = self.step(Xt, iter, warm_u = ut if self.warmopt else None)
            if self.abort_: # This step failed, abort before plotting
                break

            X_hist = self.X_hist[:,:,-self.trajectory_length:] if self.show_trajectories else None
            self.callbacks.on_step_end(Xt, X_hist, self.Y, ft, iter, t)
            self.Xt = Xt
            self.ut = ut
            # TODO: Maybe add stopping condition on tolerance here
            # TODO: Maybe add stopping condition on divergence (e.g. on F estim) here


        self.callbacks.on_flow_end()

        return Xt


    def flow_repeatedly(self, X=None, ρ0=None, samples=100, steps=10, repetitions=5):#, repetitions=10):
        X_joint_hist = []
        # Only valid if FunctionalValueCallback used
        # [isinstance(cb, 'FunctionalValueCallback') cb for cb in self.]

        ### FIXME: this fails when running from notebook because there I import via python path, not installed package
        #bool_f_cb = [isinstance(cb, callbacks.FunctionalValueCallback) for cb in self.callbacks]
        ### Ugly hack for now: check for attribute that only FunctionalValueCallback has
        bool_feval_cb = [hasattr(cb, 'F_estim_hist') for cb in self.callbacks]
        times = []
        if any(bool_feval_cb):
            feval_cb = self.callbacks[np.where(bool_feval_cb)[0][0]]
            F_estim_hist = []
            F_obj_hist = []
            F_exact_hist = []

        for i in range(repetitions):
            print(f'------------- FLOW WITH RANDOM RESTART # {i+1} -------------')
            self.reset()
            if self.ρ0 is not None:
                X0 = self.ρ0.sample((samples,)).reshape(samples, self.flat_dim)
            elif X is not None:
                X0 = X[i]
            self.F.reset(X0) #FIXME: we do double reset for F's, in self.reset() and here

            XT = self.flow(steps=steps, X0=X0)

            #if torch.any(torch.isinf(feval_cb.F_obj_hist)):
            # if np.any(np.isinf(feval_cb.F_obj_hist)):
            #     print(f'Warning: infs in restart {i}, discarding')
            #     continue

            ## TODO: check return code of self.flow, skip rep if aborted
            if self.abort_:
                #pdb.set_trace(header='Aborted restart')
                continue

            #if self.f_hist[-1] > self.f_hist[0]:
            if (feval_cb.F_estim_hist[-1] > feval_cb.F_estim_hist[0]):
                print('Run failed')
                continue

            self.callbacks.stack_run()

            X_joint_hist.append(self.X_hist.clone().detach().cpu().float().unsqueeze(-1))
            F_estim_hist.append(feval_cb.F_estim_hist)
            F_exact_hist.append(feval_cb.F_exact_hist)
            F_obj_hist.append(feval_cb.F_obj_hist)
            times.append(feval_cb.times)
            plt.show()

        pdb.set_trace(header=f'Done with loops, completed {len(times)} succesful ones')

        X_joint_hist  = torch.cat(X_joint_hist, dim = -1)  # n x d x t x reps
        F_obj_hist    = np.vstack(F_obj_hist)
        F_estim_hist  = np.vstack(F_estim_hist)
        #F_exact_hist = feval_cb.F_exact_hist # this should be the same for all ## Actually no, can change - stochasticy in eval of F
        F_exact_hist  = np.vstack(F_exact_hist)
        times         = np.vstack(times)

        feval_cb.F_estim_hist = F_estim_hist
        feval_cb.F_exact_hist = F_exact_hist
        feval_cb.F_obj_hist = F_obj_hist
        feval_cb.times = times

        return X_joint_hist

    def plot(self, X, X_traj=None, Y=None, title=None, iteration=0, **kwargs):
        ## Now we outsource this to plotting callback
        cblist = self.callbacks if isinstance(self.callbacks, CallbackList) else [self.callbacks]
        for cb in cblist:
            #if isinstance(cb, PlottingCallback): #Fails import PlottingCallback imported differently, e.g. realtive vs absolute
            if hasattr(cb, 'plot'):
                cb.plot(X, X_traj, Y, title, iteration, new_fig=True, **kwargs)
