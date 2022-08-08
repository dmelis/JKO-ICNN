import os
import pdb
import time
from functools import partial
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from celluloid import Camera
from torchvision.utils import make_grid, save_image
try:
    from rdkit import Chem
    from rdkit.Chem import QED
except:
    print('No rdkit')

from tqdm.notebook import tqdm
try:
    import ot
except:
    print('No POT')

from .utils import show, density_plot_1d, density_plot_2d, density_plot_3d
from .utils import meshgrid_from, meta_subplots
from .pde import DynamicDistribution

try:
    from tsnecuda import TSNE
    tsnelib = 'tsnecuda'
except:
    print('tsnecuda not found - will use (slower) cpu implementation')
    #logger.warning("tsnecuda not found - will use (slower) TSNE from sklearn")
    from sklearn.manifold import TSNE
    tsnelib = 'sklearn'
    #
    from openTSNE import TSNE
    from openTSNE.callbacks import ErrorLogger
    tsnelib = 'opentsne'



class Callback():
    # Default Attribs
    store_trajectories = False
    trajectory_freq    = None
    def __init__(self): pass
    def _inherit_from_flow(self, flow, *args, **kwargs):
        self.n = flow.n
        self.input_type = flow.input_type
        self.channels = flow.channels
        self.flat_dim = flow.flat_dim
        # Anything else needed by callbacks?
    def on_flow_begin(self, *args, **kwargs): pass
    def on_step_begin(self, *args, **kwargs): pass
    def on_step_end(self, *args, **kwargs): pass
    def on_flow_end(self, *args, **kwargs): pass
    def reset(self): pass
    def stack_run(self): pass

class CallbackList(Callback):
    def __init__(self, cbs):
        self.cbs = cbs
        ### Aggregate requierements imposed by callbacks on Flow - if at least
        # one of them needs it, ask for it.

        # trajactories need to be stored if *any* callback requires them
        trajectory_attrs = [cb.store_trajectories for cb in cbs]
        self.store_trajectories = np.array(trajectory_attrs).any()
        # snapshot frequency for trajectories by fastest frequency one
        trajfreq_attrs =  [cb.trajectory_freq for cb in cbs if cb.trajectory_freq is not None]
        self.trajectory_freq = np.array(trajfreq_attrs) if len(trajfreq_attrs) >0 else None
        # true time-dependent density (if provided) must be updated only once
        # TODO: do we want to update in here, or in flow?
    def __len__(self):
        return len(self.cbs)
    def __getitem__(self, i):
        return self.cbs[i]
    def _inherit_from_flow(self, flow,  *args, **kwargs):
        for cb in self.cbs: cb._inherit_from_flow(flow)
    def on_flow_begin(self, *args, **kwargs):
        for cb in self.cbs: cb.on_flow_begin(*args, **kwargs)
    def on_flow_end(self, *args, **kwargs):
        for cb in self.cbs: cb.on_flow_end(*args, **kwargs)
    def on_step_begin(self, *args, **kwargs):
        for cb in self.cbs: cb.on_step_begin(*args, **kwargs)
    def on_step_end(self, *args, **kwargs):
        for cb in self.cbs: cb.on_step_end(*args, **kwargs)
    def reset(self):
        for cb in self.cbs: cb.reset()
    def stack_run(self):
        for cb in self.cbs: cb.stack_run()

### Modified from my otdd repo
class PlottingCallback(Callback):
    def __init__(self,
            display_freq=1,
            animate=True,
            same_fig=False,
            show_trajectories=True,
            show_density=True,
            density_type='2d',
            density_method='pushforward',
            density_bw=None,
            ρ0 = None,
            ρinf = None,
            use_sns=False,
            trajectory_length=5,
            show_target = True,
            plot_pad = 0.2,
            figsize=None,
            save_format='pdf',
            ndim=2,
            azim=-80,
            elev=5,
            xrng=None,
            yrng=None,
            density_cmap=None,
            traj_color='k',
            save_path=None):
        self.animate = animate
        #self.display_its = display_its
        self.display_freq = display_freq
        self.same_fig = same_fig
        self.show_trajectories = show_trajectories
        self.show_density = show_density
        self.density_type = density_type
        self.density_method = density_method
        self.density_bw = density_bw
        if ρ0 is not None:
            self.ρ0 = copy.deepcopy(ρ0)
            self.ρt = copy.deepcopy(ρ0)
        else:
            self.ρ0 = self.ρt = None
        self.ρinf = ρinf

        self.use_sns = use_sns
        self.store_trajectories = self.show_trajectories
        self.trajectory_freq = self.display_freq
        self.trajectory_length = trajectory_length
        self.show_target = show_target
        self.ndim = ndim

        ## Low-level plotting args
        if figsize is None:
            self.figsize = (6,4) #if not self.animate else (10,7)
        else:
            self.figsize = figsize
        self.azim = azim
        self.elev = elev
        self.plot_pad = plot_pad
        self.density_cmap=density_cmap
        self.traj_color=traj_color
        self.xrng = xrng
        self.yrng = yrng
        self.save_format = save_format
        self.save_path = save_path
        self.fig = self.ax = None

    def _get_plot_ranges(self, X1, X2=None):
        pad = self.plot_pad
        with torch.no_grad():
            if X2 is None: #LAZY
                X2 = X1
            if self.xrng is not None and self.yrng is not None:
                return self.xrng, self.yrng
            with torch.no_grad():
                xmin = X1[:,0].min().item()
                xmax = X1[:,0].max().item()
                if X1.shape[1] >= 2:
                    ymin = X1[:,1].min().item()
                    ymax = X1[:,1].max().item()
                else:
                    # Set range for denisty plot of 1D data
                    ymin, ymax = -0.1, 3
                ## TODO: cann do this generally over d dimensions with min/max over second axis
                if X2 is not None:
                    xmin = min(xmin, X2[:,0].min().item())
                    xmax = max(xmax, X2[:,0].max().item())
                    if X2.shape[1] >= 2:
                        ymin = min(ymin, X2[:,1].min().item())
                        ymax = max(ymax, X2[:,1].max().item())

                xmin = xmin - pad
                xmax = xmax + pad
                if X1.shape[1] >= 2:
                    ymin = ymin - pad
                    ymax = ymax + pad

            self.xrng, self.yrng = (xmin,xmax), (ymin,ymax)
        return self.xrng, self.yrng

    def plot(self, X, X_traj=None, Y=None, title=None, iteration=0, time=None, new_fig=False, ax=None, **kwargs):
        if self.show_density:
            if self.density_method == 'pushforward':
                #kernel = partial(self.flow.estimate_density, method='pushforward')
                kernel = lambda x: torch.exp(self.flow.estimate_density(x, method='pushforward'))
            else:
                kernel = None

        if new_fig:
            if ax is None:
                fig, ax = plt.subplots()
            if self.input_type == 'features' and self.flat_dim in [1,2,3]:
                xrng, yrng = self._get_plot_ranges(X, Y)
        else: # Either animation, or multiple time steps in same plot
            if self.fig is None: # First time plotting
                if not self.show_density or self.density_type == '2d':
                    self.fig, self.ax = plt.subplots(figsize=self.figsize)
                else:
                    self.fig = plt.figure(figsize=self.figsize)
                    self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
                if self.input_type == 'features' and self.flat_dim in [1,2,3]:
                    xrng, yrng = self._get_plot_ranges(X, Y)
                    self.ax.set_xlim(xrng)
                    self.ax.set_ylim(yrng)
                else:
                    #plt.axis('off')
                    self.ax.axes.xaxis.set_visible(False)
                    self.ax.axes.yaxis.set_visible(False)
                if self.animate:
                    self.camera = Camera(self.fig)
            ax = self.ax
            if self.input_type == 'features' and self.flat_dim in [1,2,3]:
                xrng, yrng = ax.get_xlim(), ax.get_ylim()

        ## Color Palette
        #pdb.set_trace()
        #j = iteration/self.display_freq
        #print(j)
        main_color = self.main_cmap(int(iteration/self.display_freq)) #'C0' if new_fig else f'C{int(j)}'

        if self.input_type == 'images':
            show(make_grid(X[:64].cpu().reshape(64,self.channels,self.dims[-2],self.dims[-1])), ax = ax)
        elif self.input_type == 'pixels':
            ax.imshow(X.reshape(int(np.sqrt(self.n)), int(np.sqrt(self.n)), 3))
        elif self.flat_dim == 2: # 2d
            if not self.show_density:
                ax.scatter(*X.T.cpu(), c='C0', s=1)
            elif self.density_type == '2d':
                density_plot_2d(X.cpu(), xrng=xrng, yrng=yrng, ax=ax,
                            cmap=self.density_cmap, show=False)
            else:
                density_plot_3d(X.cpu(), xrng=xrng, yrng=yrng, ax=ax,
                            cmap=self.density_cmap, show=False)
            if Y is not None:
                ax.scatter(*Y.T.cpu(), c='C1', s=1)
            if self.show_trajectories and X_traj is not None:
                for x in X_traj:
                    ax.plot(*x, color=self.traj_color, alpha=0.2, linewidth=0.5)
            ax.set_xlim(xrng)
            ax.set_ylim(yrng)
            ax.set_title('')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        elif self.flat_dim == 1:
            if not self.show_density:
                ax.scatter(X, np.zeros_like(X), c=main_color, s=1)
            else:
                ρflow_label = r'$\hat{\rho}_t$' + ('' if self.animate else f', t={time:2.2f}')
                ρinf_label = r'$\rho_{\infty}$ (steady)' if not self.same_fig or (iteration==0) else None
                ρtrue_label = r'$\rho(x,t)$' if not self.same_fig or (iteration==0) else None
                ρinit_label = r'$\rho_0$ (initial)' if not self.same_fig or (iteration==0) else None

                density_plot_1d(X.cpu(), xrng=xrng, use_sns=self.use_sns,
                            color=main_color, rug_color=main_color,
                            histogram=self.animate or not self.same_fig,
                            ax=ax, kernel = kernel,
                            bw=self.density_bw,
                            ρ_init = self.ρ0 if self.ρ0 else None,
                            ρ_init_label = ρinit_label if self.ρ0 else None,
                            ρ_target = self.ρinf if self.ρinf else self.ρt,
                            ρ_target_label = ρinf_label if self.ρinf else ρtrue_label,
                            density_label = ρflow_label,
                            cmap=self.density_cmap, legend=False, show=False)

            if Y is not None:
                ax.scatter(*Y.T.cpu(), c='C1', s=1)
            if self.show_trajectories and X_traj is not None:
                for x in X_traj:
                    ax.plot(*x, color=self.traj_color, alpha=0.2, linewidth=0.5)
            ax.set_xlim(xrng)
            ax.set_ylim(yrng)
            ax.set_title('')
            #ax.get_xaxis().set_ticks([])
            #ax.get_yaxis().set_ticks([])

        if (self.flat_dim==1) and (iteration==0 or new_fig):
            ax.legend(fontsize='large')
        # elif not self.animate:
        #     ax.legend(fontsize='large')

        if (title is not None):# and (iteration==0 or new_fig):
            if not (self.show_density and self.density_type == '3d'):
                ax.text(0.5, 1.01, title, transform=ax.transAxes, ha='center',size=14)
            else:
                ax.text2D(0.5, 1.01, title, transform=ax.transAxes, ha='center',size=14)

    def on_flow_begin(self, X, Y=None, f=None, flow=None, steps=None, **kwargs):
        self.flow = flow
        if self.save_path:
            save_dir = os.path.dirname(self.save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        if not self.animate and self.same_fig:
            # Plots will have differnt colors
            total_plots = int(steps/self.display_freq) + 1
            _cmap = sns.color_palette("rocket", 10)[::-1][:total_plots]
            #_cmap = sns.color_palette("rocket", total_plots)
            main_cmap = lambda i: _cmap[i]#, as_cmap=True)
        else:
            main_cmap = lambda i: 'C1'
        self.main_cmap = main_cmap

        if not self.animate and self.same_fig:
            title = 'Flow Evolution'
        elif f is None:
            title = r'Time t=0       '#, $F(\rho_t)$=????'
        else:
            title = r'Time t=0, $F(\rho_t)$={:4.4f}'.format(f)
        self.plot(X, Y=Y, title=title, time=0)
        if self.animate:
            self.camera.snap()
        elif not self.same_fig:
            if self.save_path:
                outpath = self.save_path + '_t_0.0' #+ self.save_format
                #plt.tight_layout()
                plt.savefig(outpath + '.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(outpath + '.png', dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(1)
            plt.close()

    def on_flow_end(self, **kwargs):
        if self.animate:
            animation = self.camera.animate()
            if self.save_path:
                animation.save(self.save_path +'.mp4')
            self.animation = animation
            plt.close(self.fig)
        elif self.same_fig:
            self.ax.legend()
            if self.save_path:
                outpath = self.save_path + '_multi_t' + self.save_format
                plt.tight_layout()
                plt.savefig(outpath, dpi=300) #bbox_inches='tight',
            plt.show(block=False)
            #plt.pause(1)
            #plt.close()
            #plt.close('all')#self.fig) #-> Doesn't work!! Why?

    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        if self.ρt is not None and hasattr(self.ρt, 'step'):
            ## Update the density
            ## TODO: what if we have multiple callbacks that need to update density??
            self.ρt.step(self.flow.τ)
            ## Check that times agree
            assert ((self.ρt.t - self.ρt.t0).item() - t) < 1e-12

        if self.display_freq is None or (iteration % self.display_freq == 0): # display
            if not self.animate and self.same_fig:
                title = ''
            else:
                title = r'Time t={:8.4f}, $F(\rho_t)$={:4.2f}'.format(t, obj)

            X_traj = X_hist[:,:,-self.trajectory_length:] if self.show_trajectories else None
            self.plot(X, X_traj=X_traj, Y=Y, title=title, iteration=iteration, time=t,
                      new_fig=not (self.animate or self.same_fig)) #trajectories=kwargs['trajectories'],

            if self.animate:
                self.camera.snap()
            elif not self.same_fig:
                if self.save_path:
                    outpath = self.save_path + '_t_{}'.format(t)#, self.save_format)
                    #plt.tight_layout()
                    plt.savefig(outpath +'.pdf', dpi=300, bbox_inches='tight')
                    plt.savefig(outpath +'.png', dpi=300, bbox_inches='tight')
                #plt.show()
                plt.show(block=False)
                plt.pause(0.2)
                plt.close()
            else:
                pass

    def reset(self):
        self.fig = None
        self.ax  = None
        self.camera = None
        if self.ρt is not None:
            self.ρt.reset()

### Modified from my otdd repo
class EmbeddingCallback(PlottingCallback):
    """ Callback to embed and plot high dimensional flows

        Inherits from PlottingCallback all the plotting stuff, implements the
        embedding part.

    """
    def __init__(self, method = 'tsne', dimension=2, joint = True, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.joint  = joint
        self.dim    = dimension

    # Override inheritance to replace original input type/dim with embedded one
    def _inherit_from_flow(self, flow, *args, **kwargs):
        self.n = flow.n
        self.input_type = 'features'
        self.channels = None
        self.flat_dim = self.dim

    def _embed(self, X, Y=None):
        with torch.no_grad():
            X_ = X.detach().clone().reshape(self.n, -1).cpu()
            if Y is not None:
                X_ = torch.cat([X_,Y.detach().clone().reshape(self.n, -1).cpu()], dim=0)
            if tsnelib in ['sklearn', 'tsnecuda']:
                # TODO: CHeck if tsnecuda can take cudatensor, to avoid .cpu() above
                X_emb = TSNE(n_components=self.ndim, verbose=0, perplexity=50).fit_transform(X_)
            else:
                if not hasattr(self, 'tsne') or self.tsne is None:
                    X_emb = TSNE(n_components=self.ndim, perplexity=50,
                             n_jobs=8,verbose=2).fit(X_)
                    self.tsne = X_emb
                X_emb = self.tsne.transform(X_).astype(np.float32)
            if isinstance(X_emb, np.ndarray):
                X_emb = torch.from_numpy(X_emb)
            if Y is not None:
                X_emb, Y_emb = X_emb[:X.shape[0],:].to('cpu'), X_emb[X.shape[0]:,:].to('cpu')
            else:
                X_emb, Y_emb = X_emb.to('cpu'), None
        return X_emb, Y_emb

    def on_flow_begin(self, X, Y=None, f=None):
        X_emb, Y_emb = self._embed(X, Y)

        if self.store_trajectories:
            self.Xt = X_emb.detach().clone().cpu().unsqueeze(-1).float() # time will be last dim

        super().on_flow_begin(X_emb, Y_emb, f)

    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        if self.display_freq is None or (iteration % self.display_freq == 0): # display
            X_emb, Y_emb = self._embed(X, Y)

            if self.store_trajectories:
                # Convert to cpu, float (in case it was double) for dumping
                self.Xt = torch.cat([self.Xt, X_emb.detach().clone().cpu().float().unsqueeze(-1)], dim=-1)

            super().on_step_end(X_emb, self.Xt, Y, obj, iteration, t, **kwargs)


class MolCallback(Callback):
    def __init__(self, history_save_file, vae, v_model, v_loss_functional, decoding, display_plots=False):
        super().__init__()
        self.vae = vae
        self.v_model = v_model
        self.v_loss_functional = v_loss_functional
        self.decoding = decoding
        self.history_dict = {'validity': [], 'uniqueness': [], 'qed_calculable': [],
                             'qed_avg': [], 'qed_median': [], 'qed_std': [], 'qed_dist': []
                             }
        self.history_save_file = history_save_file
        self.display_plots = display_plots

    def decode_and_calculate(self, embeds, step):
        print('\n***********************************************************************************')
        print(f'\t\tJKO STEP {step} End Summary:')

        # Get decoded strings
        print('Decoding embeddings to SMILES strings...', end='')
        start = time.time()
        decoded_smiles = self.vae.sample(n_batch=embeds.shape[0], z=embeds, decoding=self.decoding)
        print(f'(Took {time.time() - start:,.2f} seconds.)')

        # Calculate percentage of SMILES that are valid and unique
        valid_smiles = []
        valid_mols = []
        valid_embeds = []
        for smi, emb in zip(decoded_smiles, embeds):
            m = Chem.MolFromSmiles(smi)
            if m:
                valid_smiles.append(smi)
                valid_mols.append(m)
                valid_embeds.append(emb)
        pcnt_valid = 100 * len(valid_smiles) / len(decoded_smiles)
        self.history_dict['validity'].append(pcnt_valid)
        print(f'Valid SMILES percentage: {pcnt_valid:.2f}% ({len(valid_smiles):,d}/{len(decoded_smiles):,d})')
        pcnt_unique = 100 * len(set(valid_smiles)) / len(valid_smiles)
        self.history_dict['uniqueness'].append(pcnt_unique)
        print(f'Percent unique (of valid): {pcnt_unique:.2f}% ({(len(set(valid_smiles))):,d}/{len(valid_smiles):,d})')

        # Calculate new QED distribution
        new_qeds = []
        calc_embeds = []
        for m, e in zip(valid_mols, valid_embeds):
            try:
                new_qeds.append(QED.qed(m))
                calc_embeds.append(e)
            except ValueError:
                continue
        pcnt_calc = 100 * len(new_qeds) / len(valid_smiles)
        self.history_dict['qed_calculable'].append(pcnt_calc)
        print(f'Percent QED-calculable (of valid): {pcnt_calc:.2f}% ({len(new_qeds):,d}/{len(valid_smiles):,d})')
        new_qeds = np.array(new_qeds)
        self.history_dict['qed_avg'].append(new_qeds.mean())
        self.history_dict['qed_median'].append(np.median(new_qeds))
        self.history_dict['qed_std'].append(new_qeds.std())
        print(f'Avg. QED: {new_qeds.mean():.3f}')
        print(f'Median QED: {np.median(new_qeds):.3f}')
        print(f'Std. QED: {new_qeds.std():.3f}')
        self.history_dict['qed_dist'].append(new_qeds)
        if self.display_plots:
            fig, ax = plt.subplots()
            sns.histplot(new_qeds, ax=ax, kde=True, bins=50, color='darkblue', line_kws={'linewidth': 2})
            ax.set_xlabel('QED', fontsize=16)
            ax.set_title(f'QED Distribution for Step # {step}', fontsize=20)
            plt.show()
        print('***********************************************************************************\n')

    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        if iteration % 1 == 0:
            self.decode_and_calculate(X, iteration)

    def on_flow_end(self, *args, **kwargs):
        torch.save(self.history_dict, self.history_save_file)


class EstimateEntropyCallback(Callback):
    def __init__(self, device, eps=1e-7):
        super().__init__()
        self.device = device
        # self.pdist = torch.nn.PairwiseDistance(p=2)
        self.eps = eps

    def estimate_entropy(self, points, step):
        # Normalize point cloud
        # norm_points = torch.nn.functional.normalize(points, p=2, dim=1)

        # Get n x n distance matrix
        dist_matrix = torch.cdist(points, points, p=2)
        # Set diagonal to positive infinity
        mask = torch.eye(points.shape[0], points.shape[0], dtype=torch.bool, device=self.device)
        dist_matrix.masked_fill_(mask, float('inf'))
        # print(dist_matrix.shape)
        # print(dist_matrix)

        # Use this code to check below
        # nearest_neighbors_idx = torch.min(dist_matrix, dim=1).indices
        # print(nearest_neighbors_idx)
        # nearest_neighbors_tensor = torch.index_select(points, dim=0, index=nearest_neighbors_idx)
        # print(torch.nn.PairwiseDistance(p=2)(points, nearest_neighbors_tensor))
        # print(dist_matrix.min(dim=1).values)

        # Estimate entropy
        print(f'Entropy estimate at step {step}: {torch.log(dist_matrix.min(dim=1).values + self.eps).sum()}')

    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        self.estimate_entropy(X, iteration)


class ConvergenceCallback(Callback):
    def __init__(self, ρ_target=None, samples_target=200, distance='sinkhorn',
        modality='first-iter', entreg = 1e-1, figsize=(5,5), save_path=None, **kwargs):
        super().__init__(**kwargs)
        self.modality = modality
        self.distance = distance
        self.entreg   = entreg
        self.dists    = []
        self.times    = []
        self.stack_dists = []
        self.stack_times = np.array([])
        self.figsize  = figsize
        if ρ_target is not None:
            self.X_tgt = ρ_target.sample((samples_target,)).detach()
        else:
            self.X_tgt = None
        self.save_path = None

    #def on_flow_begin(self, *args, **kwargs): pass
    #def on_step_begin(self, *args, **kwargs): pass
    def compute_convergence(self, X, X_hist=None):
        # If convergence in distribution
        if self.X_tgt is not None:
            C = ot.dist(X, self.X_tgt)
        elif self.modality == 'iter-to-iter':
            # We compare two last iterates
            C = ot.dist(X_hist[:,:,-1], X)
        elif self.modality == 'first-iter':
            C = ot.dist(X_hist[:,:,0], X)
        else:
            raise ValueError()

        C_norm = C.max()

        if self.distance == 'sinkhorn':
            # dist = ot.sinkhorn2(ot.unif(C.shape[0]), ot.unif(C.shape[1]),
            #         C / C.max(), self.entreg, numItermax=50)
            Γ = ot.sinkhorn(ot.unif(C.shape[0]), ot.unif(C.shape[1]),
                    C / C_norm, self.entreg, numItermax=50)[0]
            dist = (C*Γ).sum() # Multiply by unnormalized cost to maintian scale
        elif self.distance.lower() in ['emd','wasserstein']:
            dist = ot.emd2(ot.unif(C.shape[0]), ot.unif(C.shape[1]), C/C_norm)*C_norm
                    #method='sinkhorn_epsilon_scaling', verbose=True)
        return dist

    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        dists = self.compute_convergence(X, X_hist)
        steplog = {'Time': t, 'Distance': dists}
        self.dists.append(steplog)
        self.times.append(t)

    def on_flow_end(self, *args, **kwargs):
        self.times = np.array(self.times)
        #self.dists = np.array(self.dists)
        self.plot()

    def on_flow_begin(self, X, Y=None, f=None, flow=None, **kwargs):
        if self.X_tgt is not None:
            dists = self.compute_convergence(X)
            steplog = {'Time': 0.0, 'Distance': dists}
            self.dists.append(steplog)
            self.times.append(0.0)
        elif self.modality == 'iter-on-iter':
            # Nothing to do until we have two iters
            pass

    def plot(self, xlim=None, ylim=None, ax=None, palette=None, stack=False,
             save_path = None, *args, **kwargs):
        #
        # if ax is None:
        #     fig, ax = plt.subplots()
        fig, ax, show = meta_subplots(ax, self.figsize)


        df = pd.DataFrame(self.stack_dists if stack else self.dists)
        df = pd.melt(df, id_vars=['Time'], var_name='Criterion', value_name='Distance')


        if self.modality != 'first-iter':
            ylabel = r'$W_2(\hat{\rho}_t, \hat{\rho}_{t+1})$'
        elif self.X_tgt is not None:
            ylabel = r'$W_2(\hat{\rho}_t, \rho(x, t\rightarrow \infty))$'
        elif self.modality == 'first-iter':
            df['Distance'] = 1 - df['Distance']
            ylabel = r'$1-W_2(\hat{\rho}_0, \hat{\rho}_{t+1})$'

        if stack:
            print(df)
            pdb.set_trace()
        sns.lineplot(x='Time', y = 'Distance', style='Criterion',
                    hue= 'Criterion',data=df, ci=95, #palette=palette, #markers=markers,
                    ax=ax, *args, **kwargs)

        #ax.legend()
        ax.get_legend().remove()

        #ax.plot(self.times, self.dists)
        dist_or_div = ' distance' if self.distance.lower() in ['emd', 'wasserstein'] else ' divergence'
        ax.set_title(r'Convergence of $\hat{\rho}_t$ via ' + self.distance)# + dist_or_div)
        ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        if xlim is not None: ax.set_xlim(*xlim)
        if ylim is not None: ax.set_ylim(*ylim)

        if save_path is not None or self.save_path is not None:
            if save_path is not None:
                _save_path = save_path
            else:
                _save_path = self.save_path

            plt.savefig(_save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(_save_path + '.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            return ax

    def stack_run(self):
        self.stack_times  = np.append(self.stack_times, np.array(self.times))
        self.stack_dists += self.dists

    def reset(self):
        self.times  = []
        self.dists  = []


class DistributionalDistanceCallback(Callback):
    def __init__(self, ρ0=None, ρinf=None, ρasymp = None, criterion='sinkhorn',
                 entreg = 1e-1, figsize=(5,5), save_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ρ0 = copy.deepcopy(ρ0)
        self.ρt = copy.deepcopy(ρ0)
        self.ρinf = ρinf
        self.ρasymp = ρasymp
        self.criterion = criterion
        self.entreg   = entreg
        self.errors   = []
        self.times    = []
        self.stack_errors = []
        self.stack_times = np.array([])
        self.ref_ρs   = None
        self._C_norm  = None
        self.figsize  = figsize
        self.save_path = save_path

    def on_flow_begin(self, X, Y=None, f=None, flow=None, **kwargs):
        self.X0 = X.detach().clone()
        self.flow = flow

    def compute_error(self, X, ρ):
        """ Computes error of empirical sample X w.r.t a reference density ρ """
        with torch.no_grad():
            if self.criterion == 'likelihood':
                # Evaluate likelihood of samples X according to reference density
                if hasattr(ρ, 'log_prob'):
                    error = - ρ.log_prob(X).mean() # + self.ρ0.log_prob(self.X0).mean()
                else:
                    error = - ρ(X).mean() # + self.ρ0.log_prob(self.X0).mean()
            elif self.criterion in ['sinkhorn','emd']:
                # Get sample from reference density
                if hasattr(ρ, 'sample'):
                    X_ref = ρ.sample((X.shape[0],)).reshape(X.shape)
                else:
                    raise NotImplementedError
                    #rejection_sampler(ρ, )
                # Compute distance between the two samples at time t
                C = ot.dist(X, X_ref)
                C_norm = C.max()
                #if self._C_norm is None: self._C_norm = C.max()  # To make costs comparable across iters!
                if self.criterion == 'sinkhorn':
                    # error = ot.sinkhorn2(ot.unif(X.shape[0]), ot.unif(X.shape[0]),
                    #         C / self._C_norm, self.entreg, numItermax=50)[0]
                    Γ = ot.sinkhorn(ot.unif(X.shape[0]), ot.unif(X.shape[0]),
                            C / C_norm, self.entreg, numItermax=100)[0]
                    error = (C*Γ).sum() # Multiply by unnormalized cost to maintian scale
                elif self.criterion == 'emd':
                    error = ot.emd2(ot.unif(X.shape[0]), ot.unif(X.shape[0]), C / C_norm) * C_norm
        return error

    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        errors = {'Time': t}
        if self.ρt is not None and hasattr(self.ρt, 'step'):
            ## Update the density
            self.ρt.step(t)
            ## Check that times agree
            assert ((self.ρt.t - self.ρt.t0).item() - iteration*t) < 1e-10
            errors[r'$d(\hat{\rho}_t, \rho(x,t))$'] = self.compute_error(X, self.ρt)

        if self.ρinf is not None:
            errors[r'$d(\hat{\rho}_t, \rho_{\infty}(x))$'] = self.compute_error(X, self.ρinf)

        if self.ρasymp is not None:
            errors[r'$d(\hat{\rho}_t,\rho(x,t\rightarrow \infty))$'] = self.compute_error(X, self.ρasymp)

        self.errors.append(errors)
        self.times.append(t)

    def on_flow_end(self, *args, **kwargs):
        self.times = np.array(self.times)
        self.plot()

    def plot(self, xlim=None, ylim=None, ax=None, palette=None, stack=False,
             save_path = None, *args, **kwargs):
        # Colors
        #c_ρ0      = sns.color_palette("flare",100)[0]
        #
        #
        # c_ρInf    = sns.color_palette("flare",100)[-1]
        # #c_ρxT      = #sns.color_palette("flare",100)[50]
        # c_ρxT     = sns.color_palette('Set2', 3)[0]
        # #c_ρasymp  = sns.color_palette("flare",100)[-1]
        # c_ρasymp  = sns.color_palette('Set2', 3)[1]

        #palette = [c_ρxT, c_ρInf]
        if palette is None:
            palette = sns.color_palette('flare', len(self.errors[0])-1) # = 'Set2'

        #markers = ['-', '.-', ':'][:len(self.errors[0])-1]

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show = True
        else:
            show = False

        df = pd.DataFrame(self.stack_errors if stack else self.errors)
        df = pd.melt(df, id_vars=['Time'], var_name='Criterion', value_name='Distance')
        sns.lineplot(x='Time', y = 'Distance', style='Criterion',
                    hue= 'Criterion',data=df, ci=95, palette=palette, #markers=markers,
                    ax=ax, *args, **kwargs)
        ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        ax.set_title(r'Approximation Error on $\rho_t$ via ' + self.criterion)

        # if self.criterion == 'likelihood':
        #     ax.set_ylabel(r'NLL$(X_t | \rho)$')
        # elif self.criterion == 'sinkhorn':
        #     ax.set_ylabel(r'Sinkhorn$(\hat{\rho}_t, \cdot)$')
        # elif self.criterion == 'emd':
        #     ax.set_ylabel(r'$\textup{W}_2(\hat{\rho}_t, \cdot)')

        #ax.legend(self.ref_ρs)
        if self.save_path is not None:
            plt.savefig(self.save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(self.save_path + '.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            return ax

    def stack_run(self):
        self.stack_times  = np.append(self.stack_times, np.array(self.times))
        self.stack_errors += self.errors

    def reset(self):
        self.times   = []
        self.errors  = []
        if self.ρt is not None:
            self.ρt.reset()

class FunctionalValueCallback(Callback):
    """ Evaluation callback based on value of objective Functional.

        Currently, only for internal energy functionals. TODO: expand to V, W
        or create new callbacks.

        TODO:
            - Update this to look more like DistributionalDistanceCallback
              (i.e., append dicts, add stack_run)
            - After that, update flow_repeatedly


    """
    def __init__(self, F, ρ0=None, ρinf=None, F_asymp=None,
                 eval_points=1001, eval_type='integral',
                 integral_domain=None, plot_F_obj=False, plot_F_estim=True,
                 plot_F_exact=False,
                 figsize=(5,5), save_path=None, **kwargs):
        super().__init__(**kwargs)
        self.F = F
        self.eval_points = eval_points
        self.eval_type   = eval_type
        self.domain = integral_domain
        self.plot_F_obj = plot_F_obj
        self.plot_F_estim = plot_F_estim
        self.plot_F_exact = plot_F_exact

        self.ρ0 = copy.deepcopy(ρ0) if ρ0 is not None else None
        # Only keep track of ρt if it's time-evolving (DynamicDistribution)
        self.ρt = copy.deepcopy(ρ0) if hasattr(ρ0, 'step') else None
        self.ρinf = ρinf
        self.F_asymp = F_asymp # Should be a function of *relative* time t

        self.F_obj_hist   = [] # Loss objective (might be different from true F)
        self.F_estim_hist = [] # Exact F evaluated in estimated density
        self.F_exact_hist = [] # Exact F evaluated in exact density
        self.F_asymp_hist = [] # Exact F evaluated in asymptotic density

        self.F_ρinf = None
        self.times    = []
        self.figsize  = figsize
        self.save_path = save_path

    def on_flow_begin(self, X, Y=None, f=None, flow=None, steps=None, **kwargs):
        self.X0 = X.detach().clone()
        self.flow = flow

        self.times.append(0.0)
        self.F_obj_hist.append(f.item() if f is not None else None)
        F_exact, F_estim = self.eval_F()
        if F_exact: self.F_exact_hist.append(F_exact)
        if F_estim: self.F_estim_hist.append(F_estim)

        ## Get domain for integration of exat functions
        if self.ρ0 is not None and hasattr(self.ρ0, 'domain'):
            x_range = self.ρ0.domain
        elif self.domain is not None:
            x_range = self.domain
        else:
            raise ValueError()

        x = torch.linspace(*x_range,self.eval_points).unsqueeze(1)

        if self.ρinf is not None:
            self.F_ρinf = self.F.exact_eval(x, self.ρinf).numpy()

        if self.ρ0 is not None:
            self.F_ρ0 = self.F.exact_eval(x, self.ρ0).numpy()

        if self.F_asymp: self.F_asymp_hist.append(self.F_asymp(t=0.0))


    def eval_F(self):
        ### FIXME: This assumes the all points with mass 0 can be ignored,
        ### this is always the case for functionals that can be interpreted as
        ### expectations, like V and W. For F, this implicitly assumes that
        ### f(0)=0 or inf, which might not always hold.
        # if self.ρt is not None and hasattr(self.ρt, 'domain'):
        #     x_range = self.domain
        # else:
        #     raise ValueError()

        #x = torch.linspace(*x_range,self.eval_points).unsqueeze(1)
        if self.ρt is not None:
            ## If we have exact ρ(x,t) density compute its F value + get its domain
            x, npoints = meshgrid_from(ρ=self.ρt, npoints_dim = self.eval_points)
            with torch.no_grad(): F_exact = self.F.exact_eval(x, self.ρt)
        elif self.ρ0 is not None:
            ## If we only have initial ρ(x,0), just get its domain
            x, npoints = meshgrid_from(ρ=self.ρ0, npoints_dim = self.eval_points)
            F_exact = None
        else:
            F_exact = None

        # Compute F on density estimated from flow
        print('Exact query ranges:', x.min(), x.max())
        log_prob_flow_t = self.flow.estimate_density(X_query=x, method='pushforward')

        #F_estim = torch.trapz(self.F.f(torch.exp(log_prob_flow_t.squeeze())), x.squeeze())

        F_estim = self.F.exact_eval(x, ρx=torch.exp(log_prob_flow_t))
        if torch.isnan(F_estim):
            print(x.min(), x.max(), self.ρt.domain)
            pdb.set_trace()
        return F_exact, F_estim


    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        self.times.append(t)
        self.F_obj_hist.append(obj)

        ## If true density was provided: compute value of F at this exact density
        if self.ρt is not None and hasattr(self.ρt, 'step'):
            ## Update the density
            ## TODO: what if we have multiple callbacks that need to update density??
            self.ρt.step(self.flow.τ)
            ## Check that times agree
            assert ((self.ρt.t - self.ρt.t0).item() - t) < 1e-12, f'Time missmatch: {(self.ρt.t - self.ρt.t0).item()} vs {t}'
        F_exact, F_estim = self.eval_F()
        if F_exact: self.F_exact_hist.append(F_exact)
        if F_estim: self.F_estim_hist.append(F_estim)

        pstr = f'F(ρₜ)={F_estim.item():8.4f} (via exact computation)'
        if self.ρt is not None:
            pstr += f', F(ρ(x,t))={F_exact.item():8.4f}, ΔF={(F_exact - F_estim).item():8.2f}'
        print(pstr)

        if self.F_asymp: self.F_asymp_hist.append(self.F_asymp(t))


    def on_flow_end(self, *args, **kwargs):
        self.times  = np.array(self.times)
        self.F_obj_hist = np.array(self.F_obj_hist)
        self.F_estim_hist = np.array(self.F_estim_hist)
        self.F_asymp_hist = np.array(self.F_asymp_hist)

        #self.F_exact_hist = np.array(self.F_exact_hist)
        self.plot()

    def plot(self, xlim=None, ylim=None, ax=None, *args, **kwargs):
        legend = []

        # Colors
        c0   = sns.color_palette("flare",100)[0]
        cInf = sns.color_palette("flare",100)[-1]
        cT   = sns.color_palette("flare",100)[50]
        cTo  = sns.color_palette("flare",100)[20]
        cE   = sns.color_palette("flare",100)[75]

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show = True
        else:
            show = False


        if self.ρ0 is not None:
            ax.axhline(self.F_ρ0, c=c0, ls='--')
            legend.append(r'$F(\rho_0)$')
        if self.ρinf is not None:
            ax.axhline(self.F_ρinf, c=cInf, ls='-.')
            legend.append(r'$F(\rho_{\infty})$')
            plot_gap = False
            if plot_gap:
                ax.plot(self.times, np.abs(self.F_estim_hist - self.F_ρinf), c='gray', ls=':')
                legend.append(r'$|F(\rho_t)- F(\rho_{\infty})|$')

        if len(self.F_asymp_hist) > 0:
            self.F_asymp_hist = np.array(self.F_asymp_hist)
            if self.times.ndim > 1:
                _times = self.times[0]
            else:
                _times = self.times
            ax.plot(_times, self.F_asymp_hist, color = cInf)
            legend.append(r'$F(\rho(x,t\rightarrow \infty))$')


        if len(self.F_exact_hist) > 0 and self.plot_F_exact:
            self.F_exact_hist = np.array(self.F_exact_hist)
            if self.F_exact_hist.ndim == 1:
                ax.plot(self.times, self.F_exact_hist, color = cE)
            else:
                df = pd.DataFrame(np.column_stack([self.times.flatten(), self.F_exact_hist.flatten()]), columns=['Time', 'F'])
                sns.lineplot(x='Time', y='F', data=df, ci=95, ax = ax, color=cE)
            legend.append(r'$F(\rho(x,t))$')

        if len(self.F_estim_hist) > 0 and self.plot_F_estim:
            if self.F_estim_hist.ndim == 1:
                ax.plot(self.times, self.F_estim_hist, color=cT)
            else: ## flow_repeatedly setting
                #pdb.set_trace()
                df = pd.DataFrame(np.column_stack([self.times.flatten(), self.F_estim_hist.flatten()]), columns=['Time', 'F'])
                sns.lineplot(x='Time', y='F', data=df, ci=95, ax = ax, color=cT)
            legend.append(r'$F(\rho_t)$')

        if len(self.F_obj_hist) > 0 and self.plot_F_obj:
            if self.F_obj_hist.ndim == 1:
                ax.plot(self.times, self.F_obj_hist, color=cTo)
            else: ## flow_repeatedly setting
                pdb.set_trace()
                df = pd.DataFrame(np.column_stack([self.times.flatten(), self.F_obj_hist.flatten()]), columns=['Time', 'F'])
                sns.lineplot(x='Time', y='F', data=df, ci=95, ax = ax, color=cTo)
            legend.append(r'$\hat{F}(\rho_t)$')

        #pdb.set_trace()
        ax.legend(legend, loc='upper right')
        ax.set_title('Objective Value of Flow-Evolved Density')
        ax.set_xlabel('Time')
#        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3e}"))
        ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        ax.set_ylabel('F(ρ)')
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if self.save_path is not None:
            plt.savefig(self.save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(self.save_path + '.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            return ax

    def reset(self):
        self.times        = []
        self.F_obj_hist = []
        self.F_exact_hist = []
        self.F_estim_hist = []
        self.F_asymp_hist = []
        self.F_ρinf = None
        if self.ρt is not None:
            self.ρt.reset()

class FunctionalValueCallbackNew(Callback):
    """ Evaluation callback based on value of objective Functional.

        Currently, only for internal energy functionals. TODO: expand to V, W
        or create new callbacks.

        TODO:
            - Update this to look more like DistributionalDistanceCallback
              (i.e., append dicts, add stack_run)
            - After that, update flow_repeatedly


    """
    def __init__(self, F, ρ0=None, ρinf=None, F_asymp=None,
                 eval_points=1001, eval_type='integral',
                 integral_domain=None, plot_F_obj=False, plot_F_estim=True,
                 plot_F_exact=False,
                 figsize=(5,5), save_path=None, **kwargs):
        super().__init__(**kwargs)
        self.F = F
        self.eval_points = eval_points
        self.eval_type   = eval_type
        self.domain = integral_domain
        self.plot_F_obj = plot_F_obj
        self.plot_F_estim = plot_F_estim
        self.plot_F_exact = plot_F_exact

        ### Distributions
        self.ρ0 = copy.deepcopy(ρ0) if ρ0 is not None else None
        # Only keep track of ρt if it's time-evolving (DynamicDistribution)
        self.ρt = copy.deepcopy(ρ0) if hasattr(ρ0, 'step') else None
        self.ρinf = ρinf
        self.F_asymp = F_asymp # Should be a function of *relative* time t

        #self.F_obj_hist   = [] # Loss objective (might be different from true F)
        #self.F_estim_hist = [] # Exact F evaluated in estimated density
        #self.F_exact_hist = [] # Exact F evaluated in exact density
        #self.F_asymp_hist = [] # Exact F evaluated in asymptotic density

        self.steplog = []
        self.times   = []
        self.stacked_steplog = []
        self.stacked_times   = np.array([])

        self.F_ρinf = None
        self.figsize  = figsize
        self.save_path = save_path

    def on_flow_begin(self, X, Y=None, f=None, flow=None, steps=None, **kwargs):
        self.X0 = X.detach().clone()
        self.flow = flow

        steplog = {'Time': 0.0, 'F_obj': f.item() if f is not None else None}
        #self.F_obj_hist.append(f.item() if f is not None else None)

        ### Initial Evaluation of time-varying F values
        F_values = self.eval_F()
        steplog.update(F_values)

        ### Initial (and only) Evaluation of static F valuess

        ## Get domain for integration of exat functions
        if self.ρ0 is not None and hasattr(self.ρ0, 'domain'):
            x_range = self.ρ0.domain
        elif self.domain is not None:
            x_range = self.domain
        else:
            raise ValueError()

        x = torch.linspace(*x_range,self.eval_points).unsqueeze(1)

        if self.ρinf is not None:
            self.F_ρinf = self.F.exact_eval(x, self.ρinf).numpy()

        if self.ρ0 is not None:
            self.F_ρ0 = self.F.exact_eval(x, self.ρ0).numpy()

        if self.F_asymp: self.F_asymp_hist.append(self.F_asymp(t=0.0))

        self.steplog.append(steplog)
        self.times.append(0.0)


    def eval_F(self):
        ### FIXME: This assumes the all points with mass 0 can be ignored,
        ### this is always the case for functionals that can be interpreted as
        ### expectations, like V and W. For F, this implicitly assumes that
        ### f(0)=0 or inf, which might not always hold.
        # if self.ρt is not None and hasattr(self.ρt, 'domain'):
        #     x_range = self.domain
        # else:
        #     raise ValueError()

        F_values = {}

        #x = torch.linspace(*x_range,self.eval_points).unsqueeze(1)
        if self.ρt is not None:
            ## If we have exact ρ(x,t) density compute its F value + get its domain
            x, npoints = meshgrid_from(ρ=self.ρt, npoints_dim = self.eval_points)
            with torch.no_grad(): F_exact = self.F.exact_eval(x, self.ρt)
            F_values['F_exact'] = F_exact
        elif self.ρ0 is not None:
            ## If we only have initial ρ(x,0), just get its domain
            x, npoints = meshgrid_from(ρ=self.ρ0, npoints_dim = self.eval_points)


        ## Compute F(ρ_t): F on density estimated from flow
        print('Exact query ranges:', x.min(), x.max())
        log_prob_flow_t = self.flow.estimate_density(X_query=x, method='pushforward')

        F_estim = self.F.exact_eval(x, ρx=torch.exp(log_prob_flow_t))
        F_values['F_estim'] = F_estim

        ## Compute F(ρ(x,t->infty)): F on asymptotic density (if any)
        if self.F_asymp: F_values['F_asymp'] = self.F_asymp(t)

        if torch.isnan(F_estim):
            print(x.min(), x.max(), self.ρt.domain)
            pdb.set_trace()

        return F_values


    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        steplog = {'Time': t, 'F_obj': obj}

        self.times.append(t)
        self.F_obj_hist.append(obj)

        ## If true density was provided: compute value of F at this exact density
        if self.ρt is not None and hasattr(self.ρt, 'step'):
            ## Update the density
            ## TODO: what if we have multiple callbacks that need to update density??
            self.ρt.step(self.flow.τ)
            ## Check that times agree
            assert ((self.ρt.t - self.ρt.t0).item() - t) < 1e-12, f'Time missmatch: {(self.ρt.t - self.ρt.t0).item()} vs {t}'
        F_values = self.eval_F()

        steplog.update(F_values)

        pstr = f'F(ρₜ)={F_values["F_estim"].item():8.4f} (via exact computation)'
        if self.ρt is not None:
            pstr += f', F(ρ(x,t))={F_values["F_exact"].item():8.4f}, ΔF={(F_values["F_exact"] - F_values["F_estim"]).item():8.2f}'
        print(pstr)

        self.steplog.append(steplog)
        self.times.append(t)

    def on_flow_end(self, *args, **kwargs):
        self.times  = np.array(self.times)
        # self.F_obj_hist = np.array(self.F_obj_hist)
        # self.F_estim_hist = np.array(self.F_estim_hist)
        # self.F_asymp_hist = np.array(self.F_asymp_hist)
        #self.F_exact_hist = np.array(self.F_exact_hist)
        self.plot()

    def plot(self, xlim=None, ylim=None, ax=None, palette=None, stack=False,
             save_path = None, *args, **kwargs):
        legend = []

        # Colors
        c0   = sns.color_palette("flare",100)[0]
        cInf = sns.color_palette("flare",100)[-1]
        cT   = sns.color_palette("flare",100)[50]
        cTo  = sns.color_palette("flare",100)[20]
        cE   = sns.color_palette("flare",100)[75]

        fig, ax, show = meta_subplots(ax, self.figsize)

        pdb.set_trace()
        df = pd.DataFrame(self.stack_results if stack else self.results)
        df = pd.melt(df, id_vars=['Time'], var_name='Eval Type', value_name='Value')


        if self.ρ0 is not None:
            ax.axhline(self.F_ρ0, c=c0, ls='--')
            legend.append(r'$F(\rho_0)$')
        if self.ρinf is not None:
            ax.axhline(self.F_ρinf, c=cInf, ls='-.')
            legend.append(r'$F(\rho_{\infty})$')
            plot_gap = False
            if plot_gap:
                ax.plot(self.times, np.abs(self.F_estim_hist - self.F_ρinf), c='gray', ls=':')
                legend.append(r'$|F(\rho_t)- F(\rho_{\infty})|$')

        if len(self.F_asymp_hist) > 0:
            self.F_asymp_hist = np.array(self.F_asymp_hist)
            if self.times.ndim > 1:
                _times = self.times[0]
            else:
                _times = self.times
            ax.plot(_times, self.F_asymp_hist, color = cInf)
            legend.append(r'$F(\rho(x,t\rightarrow \infty))$')


        if len(self.F_exact_hist) > 0 and self.plot_F_exact:
            self.F_exact_hist = np.array(self.F_exact_hist)
            if self.F_exact_hist.ndim == 1:
                ax.plot(self.times, self.F_exact_hist, color = cE)
            else:
                df = pd.DataFrame(np.column_stack([self.times.flatten(), self.F_exact_hist.flatten()]), columns=['Time', 'F'])
                sns.lineplot(x='Time', y='F', data=df, ci=95, ax = ax, color=cE)
            legend.append(r'$F(\rho(x,t))$')

        if len(self.F_estim_hist) > 0 and self.plot_F_estim:
            if self.F_estim_hist.ndim == 1:
                ax.plot(self.times, self.F_estim_hist, color=cT)
            else: ## flow_repeatedly setting
                #pdb.set_trace()
                df = pd.DataFrame(np.column_stack([self.times.flatten(), self.F_estim_hist.flatten()]), columns=['Time', 'F'])
                sns.lineplot(x='Time', y='F', data=df, ci=95, ax = ax, color=cT)
            legend.append(r'$F(\rho_t)$')

        if len(self.F_obj_hist) > 0 and self.plot_F_obj:
            if self.F_obj_hist.ndim == 1:
                ax.plot(self.times, self.F_obj_hist, color=cTo)
            else: ## flow_repeatedly setting
                pdb.set_trace()
                df = pd.DataFrame(np.column_stack([self.times.flatten(), self.F_obj_hist.flatten()]), columns=['Time', 'F'])
                sns.lineplot(x='Time', y='F', data=df, ci=95, ax = ax, color=cTo)
            legend.append(r'$\hat{F}(\rho_t)$')

        ax.legend(legend, loc='upper right')
        ax.set_title('Objective Value of Flow-Evolved Density')
        ax.set_xlabel('Time')
        ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        ax.set_ylabel('F(ρ)')
        if xlim is not None: ax.set_xlim(*xlim)
        if ylim is not None: ax.set_ylim(*ylim)

        if self.save_path is not None:
            plt.savefig(self.save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(self.save_path + '.png', dpi=300, bbox_inches='tight')
        if show: plt.show()
        else:
            return ax

    def stack_run(self):
        self.stacked_times  = np.append(self.stacked_times, np.array(self.times))
        self.stacked_steplog += self.steplog

    def reset(self):
        self.times        = []
        self.steplog      = []
        self.F_ρinf = None
        if self.ρt is not None: self.ρt.reset()

class EstimateEntropyCallback(Callback):
    def __init__(self, device, eps=1e-7):
        super().__init__()
        self.device = device
        self.eps = eps
    def estimate_entropy(self, points, step):
        # Get n x n distance matrix
        dist_matrix = torch.cdist(points, points, p=2)
        # Set diagonal to positive infinity
        mask = torch.eye(points.shape[0], points.shape[0], dtype=torch.bool, device=self.device)
        dist_matrix.masked_fill_(mask, float('inf'))
        # print(dist_matrix.shape)
        # print(dist_matrix)
        # Estimate entropy
        print(f'Entropy estimate at step {step}: {torch.log(dist_matrix.min(dim=1).values + self.eps).sum()}')
    def on_step_end(self, X, X_hist, Y, obj, iteration, t, **kwargs):
        self.estimate_entropy(X, iteration)
