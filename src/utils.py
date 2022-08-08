import pdb
import torch
from .icnn import LinearFConvex

from PIL import Image
import PIL.ImageOps

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.animation import FuncAnimation

import seaborn as sns
import pandas as pd
import numpy as np
import math
import scipy.stats
import gc

def infer_input_type(X):
    n, *dims = X.shape
    # Batch BW images:     (n, h, w) or (n, 1, h, w)
    # Batch Color  images: (n, 3, h, w)
    # Single Color image:  (3, h, w)
    # 1D data: (3,) so dims will be []
    X0 = X.detach().clone()
    if dims == []:
        input_type = 'features'
        channels = None
        X0 = X0.unsqueeze(1)
        dims = [1]
    elif (X.ndim == 3) and (X.shape[0] == 3):
        # Single image, 3 channels, n will be number of pixels
        X0 = X0.permute(1,2,0).reshape(-1, 3)
        input_type = 'pixels'
        channels = 3
    elif (len(dims)==2 and dims[0]==dims[1]) or (len(dims)==3 and dims[0] in [1,3]):
        channels = 1 if len(dims)==2 else dims[0]
        input_type = 'images'
    else:
        input_type = 'features'
        channels = None
    return X0, input_type, channels

def show(img, ax = None):
    npimg = img.numpy()
    if ax is None:
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.axis('off')
    else:
        ax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        ax.axis('off')

def meta_subplots(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show = True
    else:
        show = False
        fig = None # maybe from plt.gcf() if needed?
    return fig, ax, show


class PositiveWeightClipper(object):
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        tol = 1e-18
        # filter the variables to get the ones you want
        if hasattr(module, 'weight_z') and module.weight_z is not None:
            w = module.weight_z.data
            w = w.clamp(tol,None)
            module.weight_z.data = w

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    for i in range(len(flat_y)):
        grad_y = torch.zeros_like(flat_y)
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph,allow_unused=True)
        if grad_x is None:
            print('here')
            grad_x = torch.zeros(x.shape)
        jac.append(grad_x.reshape(x.shape))
    return torch.stack(jac).reshape(y.shape + x.shape)

def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

def gradient_penalty(net, x):
    # ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    '''

    :param input: state[index]
    :param network: actor or critic
    :return: gradient penalty
    '''
    x_ = x.clone().detach().requires_grad_(True)
    fx = net(x_)
    musk = torch.ones_like(fx)
    g = torch.autograd.grad(fx, x_, grad_outputs=musk,
                     retain_graph=True, create_graph=True,
                     allow_unused=True)[0]  # get tensor from tuple
    #g = g.view(-1, 1)
    print(g.shape)
    return (g-x).norm(2, dim=1).pow(2).mean()

def net_batch_gradients(net, x, detach=True):
    if detach:
        x_ = x.clone().detach().requires_grad_(True)
    else:
        x_ = x
    fx = net(x_)
    musk = torch.ones_like(fx)
    g = torch.autograd.grad(fx, x_, grad_outputs=musk,
                     retain_graph=True, create_graph=True,
                     allow_unused=True)[0]  # get tensor from tuple
    return g


def invert_potential(u, y, max_iter=1000000, lr=1.0, tol=1e-12, x=None,
    history_size = 100, verbose=False, **kwargs):
    if x is None:
        x = y.clone().detach().requires_grad_(True)


    optimizer = torch.optim.LBFGS([x], lr=lr, line_search_fn="strong_wolfe",
                                  max_iter=max_iter, tolerance_grad=tol,
                                  history_size=history_size,
                                  tolerance_change=tol)
    def closure():
        # Solves x such that f(x) - y = 0
        # <=> Solves x such that argmin_x F(x) - <x,y>
        optimizer.zero_grad() # CP-flows paper doesn't have this, seems uncessary
        F = u(x)
        loss = torch.sum(F) - torch.sum(x * y)
        x.grad = torch.autograd.grad(loss, x)[0].detach()
        return loss


    optimizer.step(closure)

    Tx = net_batch_gradients(u, x, detach=True)
    error_new = (Tx - y).abs().max().item()

    if verbose and (error_new > math.sqrt(tol)):
        print('inversion error', error_new, flush=True)

    torch.cuda.empty_cache()
    gc.collect()
    return x




def net_batch_hessian(net, x):
    x_ = x.clone().detach().requires_grad_(True)
    fx = net(x_)
    musk = torch.ones_like(fx)
    g = torch.autograd.grad(fx, x_, grad_outputs=musk,
                     retain_graph=True, create_graph=True,
                     allow_unused=True)[0]  # get tensor from tuple


def test_weights(f, tol = 1e-12):
    # Checks that all weights that should be non-negative are indeed non-negative
    for m in f.modules():
        if type(m) is LinearFConvex:
            #if m.weight_z is not None: print(m.weight_z.data.min())
            assert (m.weight_z is None) or (m.weight_z.data.min() >= 0), f'Negative weights detected in {m}: {m.weight_z.data.min():4.2e}'


def test_convexity(f, niter=1000, convtol = 1e-8, strict = True):
    """ Right now only works for f: R^d -> R. """
    assert f.output_dim == 1, "Convexity test currently works only for 1-D output"
    violations = 0
    for i in range(niter):
        x = torch.randn(1,f.input_dim)
        y = torch.randn(1,f.input_dim, requires_grad=True)
        f.zero_grad()
        fy = f(y)
        gy = torch.autograd.grad(fy, y)[0].flatten().detach()
        with torch.no_grad():
            fx = f(x)
            fy = fy.detach()
            #print(fx, fy, torch.dot(gy, (x - y).flatten()))
            curvature = (fx-fy) - torch.dot(gy, (x - y).flatten())
            #print(curvature)
            if curvature < -convtol:
                print('Convexity violated! Curvature: {}'.format(curvature))
                if strict: raise ValueError()
                violations += 1

    x = torch.randn(niter, f.input_dim)
    y = torch.randn(niter,f.input_dim, requires_grad=True)


    print(f"Convexity (via curvature) violated for {violations}/{niter} tested pairs at tolerance {convtol}.")

def test_hessian(f, niter=10, eigtol = 1e-8, strict=True):
    """ Right now only works for f: R^d -> R. """
    assert f.output_dim == 1, "Convexity test currently works only for 1-D output"
    violations = 0
    for i in range(niter):
        x = torch.randn(1,f.input_dim, requires_grad=True)
        f.zero_grad()

        H = hessian(f(x), x).squeeze()

        assert torch.allclose(H, H.T), "Hessian is not symmetric"

        with torch.no_grad():
            eigvals = torch.symeig(H, eigenvectors=False).eigenvalues
            mineig  = torch.min(eigvals)
            #print(eigvals)
            if mineig < -eigtol:
                print('Hessian has negative eigenvalues: {}'.format(mineig))
                if strict: raise ValueError()
                violations += 1

    print(f"Convexity (via Hessian) violated for {violations}/{niter} tested pairs at tolerance {eigtol}.")


from cpflow.lib.logdet_estimators import stochastic_logdet_gradient_estimator

def sample_rademacher(*shape):
    return (torch.rand(*shape) > 0.5).float() * 2 - 1

def logdet_estimate_contained(u,x):
    bsz, *dims = x.shape
    dim = np.prod(dims)
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)
        ux = u(x)
        f = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
        def hvp_fun(v):
            v = v.reshape(bsz, *dims)
            hvp = torch.autograd.grad(f, x, v, create_graph=True, retain_graph=True)[0]
            #print('asd')
            if not torch.isnan(v).any() and torch.isnan(hvp).any():
                raise ArithmeticError("v has no nans but hvp has nans.")
            hvp = hvp.reshape(bsz, dim)
            return hvp

    v = sample_rademacher(bsz, dim).to(x)
    #print('v', v.shape)#, v)
    logdet = stochastic_logdet_gradient_estimator(hvp_fun, v, 10, rtol=0.0, atol=1e-3)
    #print('logdet', logdet.shape)
    return -logdet.mean()

def logdet_estimate(x, hvp_fun):
    """ Stochastic estimate of logdet Hess_x u(x) """
    bsz, *dims = x.shape
    dim = np.prod(dims)
    v = sample_rademacher(bsz, dim).to(x)
    #print('v', v.shape)#, v)
    logdet = stochastic_logdet_gradient_estimator(hvp_fun, v, 10, rtol=0.0, atol=1e-3)
    #print('logdet', logdet.shape)
    return -logdet.mean()


########

def meshgrid_from(X=None, ρ=None, ranges=None, npoints_dim=101, eps=1e-5):
    """ Create a uniform mesh grid in 1 or 2d using x to set boundaries

        For 1d, returns vector of size (npoints, 1)
        For 2d, returns flattened tensor of size (npoints^2, 2)
    """
    if ranges is not None:
        dim = 1 if type(ranges[0]) is float else len(ranges)
    elif ρ is not None:
        if hasattr(ρ, 'domain'):
            ranges = ρ.domain
            dim = 1 if type(ranges[0]) in [int,float] else len(ranges)
            # Move a bit from boundary to avoid infs
            if dim == 1:
                ranges = [(ranges[0]+eps, ranges[1]-eps)]
            else:
                ranges = [(r[0]+eps, r[1]-eps) for r in ranges]
            #pdb.set_trace()
        elif hasattr(ρ, 'sample'):
            _x = ρ.sample((100,))
            dim = 1 if _x.ndim ==1 else _x.shape[1]
            ranges = tuple(zip(-x.min(dim=0)[0],-x.max(dim=0)[0]))
        else:
            ranges = ranges[:dim]
    elif X is not None:
        bsz, *dims = X.shape
        dim = np.prod(dims)
        mins = X.min(dim=0)[0]
        maxs = X.max(dim=0)[0]
        ranges = tuple(zip(mins,maxs))   # [(xmin, xmax), (ymin, ymax)]
    else: # Default is 2D
        ranges = ((-1,1),(-1,1))
        dim = len(ranges)
    assert dim <= 2, "meshgrid_from only works for 1 or 2D inputs"

    if dim == 1:
        grid = safe_linspace(*ranges[0],npoints_dim).unsqueeze(1)
        npoints_dim = len(grid) # effective # points
        #grid
    else:
        xv, yv = torch.meshgrid([safe_linspace(*ranges[0], npoints_dim), safe_linspace(*ranges[1], npoints_dim)])
        # TODO: Maybe go back to np linspace, seems less prone to errors?
        # x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        # positions = np.vstack([x.ravel(), y.ravel()])
        grid = torch.vstack([torch.ravel(xv), torch.ravel(yv)]).T

    #pdb.set_trace()
    if ρ is not None:
        if dim == 1:
            assert (grid.min(dim=0)[0][0] > ρ.domain[0]) and (grid.max(dim=0)[0][0] < ρ.domain[1])
        else:
            for i,(m,M) in enumerate(ranges):
                #print(i, (m,M))
                #pdb.set_trace()
                assert (grid.min(dim=0)[0][i] > ρ.domain[i][0]) and (grid.max(dim=0)[0][i] < ρ.domain[i][1])

    #print(ρ.domain, grid.min(dim=0)[0], grid.max(dim=0)[0])
    return grid, npoints_dim

def safe_linspace(xmin, xmax, n):
    # linspace is prone to floating point errors, need to force compliance with domain
    ls = torch.linspace(xmin, xmax, n)
    ls = ls[(ls >= xmin) & (ls <= xmax)]
    return ls


################################## PLOTTING ####################################

def show_grid(tensor, dataname=None, invert=True, title=None,
             save_path=None, to_pil=False, ax = None,format='png'):
    " Displays image grid. To be used after torchvision's make_grid "
    if dataname and dataname in DATASET_NORMALIZATION:
        # Brings back to [0,1] range
        mean, std = (d[0] for d in DATASET_NORMALIZATION[dataname])
        tensor = tensor.mul(std).add_(mean)
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    ndarr = np.transpose(ndarr, (1,2,0))
    if to_pil:
        im = Image.fromarray(ndarr)
        if invert:
            im = PIL.ImageOps.invert(im)
        im.show(title=title)
        if save_path:
            im.save(save_path, format=format)
    else:
        if not ax: fig, ax  = plt.subplots()
        ax.imshow(ndarr, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if title: ax.set_title(title)


def animate_1d_pde(ρ, steps, xmin,xmax, ρ_inf=None, *args, **kwargs):
    ρ.reset()
    fig, ax = plt.subplots()
    ax.set_xlim(xmin,xmax)
    _x = torch.linspace(xmin, xmax, 100)
    y = torch.exp(ρ.log_prob(_x))

    # ax.plot(sim.grid[0]/nm, steady, color='k', ls='--', alpha=.5)
    line, = ax.plot(_x,y, lw=2, color='C3', ls='--', alpha=0.5)
    line, = ax.plot(_x,y, lw=2, color='C3')
    if ρ_inf is not None:
        yinf = torch.exp(ρinf.log_prob(_x))
        line, = ax.plot(_x,yinf, lw=2, color='C2')

    ax.set(xlabel='x', ylabel='normalized PDF')
    # ax.margins(x=0)
    #plt.show()

    def update(i):
        ρ.step(1e-3)
        y = torch.exp(ρ.log_prob(_x))
        #line, = ax.plot(_x, y, lw=2, color='red')
        line.set_ydata(y)
        return [line]

    anim = FuncAnimation(fig, update, frames=range(steps), *args, **kwargs)
    plt.close()
    ρ.reset()
    return anim


def density_plot_1d(X, kernel=None, xrng=None, yrng=None, ax = None, show=True,
                lognorm = False, histogram=False, use_sns=True, legend=True,
                density_label='Flow-Estimated Density',
                ρ_init=None, ρ_init_label = 'Initial Density', ρ_init_color='C2',
                ρ_target=None, ρ_target_label = 'Steady State Density', ρ_target_color='C0',
                color='C1', rug_color = 'k', bw=1.0,
                cmap=plt.cm.gist_earth_r, alpha=1.0, scatter_color='k'):
    #cmap = 'viridis'
    if cmap == 'viridis':
        scatter_color = 'w'
    elif cmap == 'jet':
        alpha = 0.5

    if xrng is None:# or yrng is None:
        xmin, xmax = X.min().item(), X.max().item()
    else:
        xmin, xmax = xrng

    x = torch.linspace(xmin, xmax, 100).unsqueeze(1)

    print(f'Density BW: {bw}')
    if (kernel is None) and not use_sns:
        kernel = scipy.stats.gaussian_kde(X.flatten(), bw_method = bw)
        Z = kernel.pdf(x.flatten())
        Z_max = Z.max()
    elif kernel is not None:
        #x = torch.from_numpy(x).float()
        Z = np.reshape(kernel(x), x.shape)
        Z_max = Z.max()
    else:
        Z = None
        Z_max = 2

    if yrng is None:
        ymin, ymax = -0.2, Z_max + 0.2
    else:
        ymin, ymax = yrng

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    # if lognorm:
    #     ax.imshow(np.rot90(Z), cmap=cmap, alpha=alpha, extent=[xmin, xmax, ymin, ymax], norm=LogNorm(vmin=1e-10, vmax=Z.max()))#, interpolation='nearest')
    # else:
    #     ax.imshow(np.rot90(Z), cmap=cmap, alpha=alpha, extent=[xmin, xmax, ymin, ymax])


    if ρ_target is not None:
        ax.plot(x, torch.exp(ρ_target.log_prob(x)), label=ρ_target_label,
                color = ρ_target_color, ls = '--')
    if ρ_init is not None:
        ax.plot(x, torch.exp(ρ_init.log_prob(x)), label=ρ_init_label,
                color = ρ_init_color, ls = '--')

    # Via Seaborn:
    if kernel is None and use_sns:
        ax = sns.distplot(X, ax = ax, rug=True, color=color,
                          rug_kws={"color": rug_color},
                          hist = histogram,
                          hist_kws={"alpha": 0.3, "linewidth": 3,},#"histtype": "step",
                          #, "color": "g"}
                          kde_kws={'bw':bw},
                          label=density_label,
                          )
        ax.set_xlim([xmin, xmax])
    else:
        # Manual
        ax.plot(x, Z, color='red', ls='-')
        if histogram:
            ax.hist(X, density=True, alpha=0.5, ec='black')

        ax.scatter(X,np.zeros_like(X), color=scatter_color, s=1)

    if legend: ax.legend()#['Flow', 'Steady State'])

    if show:
        plt.show()




def density_plot_2d(X, kernel=None, xrng=None, yrng=None, ax = None, show=True,
                lognorm = False, bw=None,
                cmap=plt.cm.gist_earth_r, alpha=1.0, scatter_color='k'):
    #cmap = 'viridis'
    if cmap == 'viridis':
        scatter_color = 'w'
    elif cmap == 'jet':
        alpha = 0.5

    if xrng is None or yrng is None:
        xmin, ymin = [v.item() for v in X.min(dim=0)[0]]
        xmax, ymax = [v.item() for v in X.max(dim=0)[0]]
    else:
        xmin, xmax = xrng
        ymin, ymax = yrng
    x, y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([x.ravel(), y.ravel()])

    if kernel is None:
        kernel = scipy.stats.gaussian_kde(X.T, bw_method=bw)
        Z = np.reshape(kernel.pdf(positions).T, x.shape)
    else:
        positions = torch.from_numpy(positions.T).float()
        #pdb.set_trace()
        Z = np.reshape(kernel(positions), x.shape)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    if lognorm:
        norm = LogNorm(vmin=1e-10, vmax=Z.max())
        ax.imshow(np.rot90(Z), cmap=cmap, alpha=alpha, extent=[xmin, xmax, ymin, ymax], norm=norm)#, interpolation='nearest')
    else:
        norm = None
        if cmap in ['coolwarm']:
            norm = TwoSlopeNorm(vcenter=0) # to make sure 0 gets white
        ax.imshow(np.rot90(Z), cmap=cmap, alpha=alpha, extent=[xmin, xmax, ymin, ymax], norm=norm)

    ax.scatter(*X.T, color=scatter_color, s=1)

    if show:
        plt.show()


def density_plot_3d(X, kernel=None, xrng=None, yrng=None, ax = None, show=True, lognorm = False,
                cmap='viridis', bw=0.5, alpha=1.0, scatter_color='k'):
    #cmap = 'viridis'
    if cmap == 'viridis':
        scatter_color = 'w'
    elif cmap == 'jet':
        alpha = 0.5

    if xrng is None or yrng is None:
        xmin, ymin = [v.item() for v in X.min(dim=0)[0]]
        xmax, ymax = [v.item() for v in X.max(dim=0)[0]]
    else:
        xmin, xmax = xrng
        ymin, ymax = yrng
    x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([x.ravel(), y.ravel()])

    if kernel is None:
        kernel = scipy.stats.gaussian_kde(X.T, bw_method=bw)
        Z = np.reshape(kernel.pdf(positions).T, x.shape)
    else:
        positions = torch.from_numpy(positions.T).float()
        #pdb.set_trace()
        Z = np.reshape(kernel(positions), x.shape)

    if ax is None:
        fig = plt.figure()#figsize=)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    surf = ax.plot_surface(x,y, Z, cmap=cmap)#, alpha=alpha)
    ax.set_zlim([0,10])
    ax.autoscale(False)
    ax.set(xlabel='x', ylabel='y', zlabel='normalized PDF')

    # if lognorm:
    #     ax.imshow(np.rot90(Z), cmap=cmap, alpha=alpha, extent=[xmin, xmax, ymin, ymax], norm=LogNorm(vmin=1e-10, vmax=Z.max()))#, interpolation='nearest')
    # else:
    #     ax.imshow(np.rot90(Z), cmap=cmap, alpha=alpha, extent=[xmin, xmax, ymin, ymax])
    #
    # ax.scatter(*X.T, color=scatter_color, s=1)

    if show:
        plt.show()


def averaged_density_plot(X=None, ρ0=None, ρt=None, ρinf=None, step_size=1., freq=5, xrng=None, yrng=None,
    domain=None, npoints=1001,bw=0.5, figsize=(5,5), title=None, plot_flow=True,
    save_path = None):
    """ XT should have shape: nsamples x d x x"""
    n,dim,steps,reps = X.shape

    # 0. Setup
    #x = torch.linspace(-10, 10, 200)#.unsqueeze(1)
    #x = meshgrid_from()
    times = np.arange(steps) * step_size
    x = torch.linspace(*domain, npoints)#.unsqueeze(1)

    plot_steps = np.arange(0, len(times), freq)
    plot_times = times[plot_steps]

    fig, ax = plt.subplots(figsize=figsize)
    labels = []

    # 1. Density estimation for each set of particles
    dfs = []
    for r in range(reps):
        for step, time in zip(plot_steps, plot_times):
            kde = scipy.stats.gaussian_kde(X[:,:,step,r].flatten(), bw_method = bw)
            z = kde(x).flatten()
            df = pd.DataFrame(np.vstack([x.numpy(),z]).T, columns=['x','Density'])
            df['Flow'] = np.round(time, decimals=5) #f't={times[t]:8.2f}' #
            df['rep']  = r
            dfs.append(df)
    df = pd.concat(dfs)

    # 2. Plot flow densities
    if plot_flow:
        ax.plot([],[], lw=0, label='Flow')
        sns.lineplot(data=df, x='x', y = 'Density', hue='Flow', ci = 95, ax = ax, palette='flare', linewidth=0.5)
        labels += [r'Flow $\hat{\rho}_t$'] +[r'$t=${:4.2e}'.format(t) if t>0 else r'$t=0$' for t in plot_times]

    # 3. Additional lines
    if ρ0 is not None or ρt is not None or ρinf is not None:
        ax.plot([],[], lw=0, label='Exact')
        labels += ["Exact"]

    if ρt is not None:
        ρt.reset()
        dfs = []
        for step, time in zip(plot_steps, plot_times):
            ρt.step(time)
            print(ρt.t)
            z = torch.exp(ρt.log_prob(x))
            df = pd.DataFrame(np.vstack([x.numpy(),z]).T, columns=['x','Density'])
            df['Flow'] = np.round(time, decimals=5) #f't={times[t]:8.2f}' #
            dfs.append(df)
        df = pd.concat(dfs)
        sns.lineplot(data=df, x='x', y = 'Density', hue='Flow', ci = 95, ax = ax, palette='flare', linewidth=1)
        labels = [r'Flow $\rho(x,t)$'] +[r'$t=${:4.2e}'.format(t) if t>0 else r'$t=0$' for t in plot_times]
        ρt.reset()

    if ρ0 is not None and ρt is None:
        ax.plot(x, torch.exp(ρ0.log_prob(x)), label = 'initial', color=sns.color_palette("flare",100)[0], ls='--', lw=2.0)
        labels += [r'$\rho_0$ (initial)']
    if ρinf is not None:
        ax.plot(x, torch.exp(ρinf.log_prob(x)), label = 'steady', color=sns.color_palette("flare",100)[-1], ls='--', lw=2.0)
        labels += [r'$\rho_{\infty}$ (steady)']

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper right')
    if title:
        ax.set_title(title)
    ax.set_xlim(*xrng)
    ax.set_ylim(*yrng)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')

    plt.show()
