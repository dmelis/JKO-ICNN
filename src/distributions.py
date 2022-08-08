import pdb
import numpy as np
import torch
import torch.distributions as D
from torch.distributions import Distribution
import PIL.ImageOps
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import scipy.stats

class KDE_Wrapper(Distribution):
    def __init__(self, kde):
        self.kde = kde # a fitted scipy stats objectt
    def log_prob(self, x):
        return torch.from_numpy(self.kde.logpdf(x.T))

def GMM(d=2, k=5, σ=5e-3, eps=1e-2, diagonal=False):
    mix = D.Categorical(torch.ones(k,))
    μs  = torch.rand(k,d)
    if diagonal:
        Σs  = σ*torch.rand(k,d)
        comp = D.Independent(D.Normal(μs, Σs), 1)
    else:
        Σs = eps*torch.rand(k, d, d)
        Σs = torch.bmm(Σs, Σs.permute(0,2,1)) + σ*torch.eye(d).reshape((1, d, d)).repeat(k, 1, 1)
        comp = D.MultivariateNormal(μs, Σs)
    ρ = D.MixtureSameFamily(mix, comp)
    return ρ

def pixels_from_image(path, crop=None, resize=128):
    img = Image.open(path)
    transform_list = []
    if crop:
        transform_list.append(transforms.CenterCrop(crop))
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    X = transforms.Compose(transform_list)(img)
    plt.imshow(X.permute(1,2,0))
    plt.axis('off')
    plt.show()
    return X

def particles_from_image(path, n=1000, thresh=0.5, crop=None, invert=False):
    img = Image.open(path)
    width, height = img.size   # Get dimensions
    if crop:
        img = transforms.CenterCrop(crop)(img)
    img = img.convert(mode="L", dither=1)#Image.NONE)

    #img = PIL.ImageOps.invert(img.convert('L')).convet('1')
    X = (np.asarray(img)) * 1.0
    plt.imshow(X)
    # For BW
    X = ~np.asarray(img.rotate(-90))
    # For Grayscale
    if invert:
        X = np.asarray(img.rotate(-90))/255
    else:
        X = 1 - np.asarray(img.rotate(-90))/255
    print(X[:50,:50], X.shape)

    x,y = np.where(X > thresh)
    Z = np.array(list(zip(x,y)))/X.shape[0]
    Z_s = Z[np.random.choice(Z.shape[0],n, replace=False)]
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(*Z_s.T, s=0.5)
    plt.show()
    Z_s = torch.tensor(Z_s)
    return Z_s



def density_from_image(bw=0.5, *args, **kwargs):
    X = particles_from_image(*args, **kwargs)
    kde = scipy.stats.gaussian_kde(X.float().T, bw_method = bw)
    ρ = KDE_Wrapper(kde)
    ρ.domain = [(0,1), (0,1)]
    return ρ, X
