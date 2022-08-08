import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pdb


from langevin_sampling.samplers import MetropolisAdjustedLangevin as MALA
from langevin_sampling.samplers import LangevinDynamics as SGLD


def get_langevin_samples(nsamples=100, ρ=None, dim=None, sampler_type='mala', lr=1e-2, lr_final=1e-4, iters=1000,
                         inits=10, skip=5, burn_in=200, device='cpu',
                         plot=False, verbose=False):
    hist_samples = []
    loss_log = []
    nl_pdf = lambda x: -ρ.log_prob(x)

    if device == 'cpu':
        torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    device = torch.device(device)
    nsamples_per_init = int(nsamples/inits) + 10 # add slack in case some fail
    for i in tqdm(range(inits)):
        loss_log = []
        x = 0.02*torch.randn(dim, requires_grad=True, device=device)
        x = x.clamp(*ρ.domain[0])
        if sampler_type == 'mala':
            sampler = MALA(x, nl_pdf, lr=lr, lr_final=lr_final, max_itr=iters, device=device)
        for j in tqdm(range(burn_in + nsamples_per_init*skip), leave=False):
            est, loss = sampler.sample()
            loss_log.append(loss)
            if j > burn_in and j%skip == 0 and (not np.isinf(loss)):
                hist_samples.append(est.cpu().numpy())
        if verbose: print(est, nl_pdf(est))
        if plot:
            fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
            plt.plot(loss_log); plt.title("Unnormalized PDF")
            plt.xlabel("Iterations")
            plt.ylabel(r"$- \log \ \mathrm{N}(\mathbf{x} | \mu, \Sigma) + const.$")
            plt.grid()
            plt.show()

    est_samples = np.array(hist_samples)

    print(len(est_samples))
    if len(est_samples) < 1:
        print('No samples')
        sys.exit()
    elif len(est_samples) > nsamples:
        print('Collected too many')
        est_samples = est_samples[np.random.choice(range(len(est_samples)), nsamples)]

    if dim == 2 and plot:
        fig = plt.figure(dpi=150, figsize=(9, 4))
        plt.subplot(111);
        plt.scatter(est_samples[:, 0], est_samples[:, 1], s=.5, color="#db76bf")
        plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
        plt.xlim(-1,1); plt.ylim(-1,1)
        plt.title("Metropolis-adjusted Langevin dynamics")
        plt.show()

    X = torch.from_numpy(est_samples).float().to(device)
    return X
