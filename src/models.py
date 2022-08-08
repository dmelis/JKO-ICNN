import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F

from cpflow.lib.icnn import ICNN, ICNN2, ICNN3, ConvICNN, ConvICNN2, ConvICNN3, ResICNN2

convex_net_dict = {'ICNN': ICNN,
                   'ICNN2': ICNN2,
                   'ICNN3': ICNN3,
                   'ConvICNN': ConvICNN,
                   'ConvICNN2': ConvICNN2,
                   'ConvICNN3': ConvICNN3,
                   'ResICNN2': ResICNN2,
                   }


class MoleculeAnnotationPredictor(nn.Module):
    def __init__(self, convex_net_type, input_dim=128, hidden_dim=128, num_hidden_layers=4):
        super().__init__()
        self.convex_net = convex_net_dict[convex_net_type](dim=input_dim,
                                                           dimh=hidden_dim,
                                                           num_hidden_layers=num_hidden_layers)

    def forward(self, x):
        convex_net_output = self.convex_net(x)
        return convex_net_output


class LitVModel(pl.LightningModule):
    def __init__(self, net, lr, task, loss_functional='mse'):
        super().__init__()
        self.net = net
        self.lr = lr
        self.task = task
        self.loss_functional_name = loss_functional
        assert loss_functional in ['mse', 'ce'], 'You passed an invalid loss functional. Use \'mse\' or \'ce\'.'
        if loss_functional == 'mse':
            self.loss_functional = F.mse_loss
        elif loss_functional == 'ce':
            self.loss_functional = F.binary_cross_entropy_with_logits

    def forward(self, x):
        output = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        if self.task == 'classification':
            with torch.no_grad():
                if self.loss_functional_name == 'mse':
                    y_preds = torch.sign(y_hat)
                else:
                    y_preds = torch.where(torch.sigmoid(y_hat) <= 0.5, 0, 1)
                self.log('train_acc', torch.sum(y_preds == y.unsqueeze(dim=1)) / x.shape[0])
        train_loss = self.loss_functional(y_hat, y.unsqueeze(dim=1))
        self.log('train_loss', train_loss)
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        val_loss = self.loss_functional(y_hat, y.unsqueeze(dim=1))
        self.log('val_loss', val_loss)
        if self.task == 'classification':
            if self.loss_functional_name == 'mse':
                y_preds = torch.sign(y_hat)
            else:
                y_preds = torch.where(torch.sigmoid(y_hat) <= 0.5, 0, 1)
            self.log('val_acc', torch.sum(y_preds == y.unsqueeze(dim=1)) / x.shape[0])
            return {'val_loss': val_loss,
                    'val_acc': torch.sum(y_preds == y.unsqueeze(dim=1)) / x.shape[0],
                    'preds': torch.where(y_preds.squeeze() <= 0, 0, 1),
                    'target': torch.where(y <= 0, 0, 1),
                    }
        else:
            return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        if self.task == 'classification':
            preds = torch.cat([tmp['preds'].int() for tmp in outputs])
            targets = torch.cat([tmp['target'].int() for tmp in outputs])
            confusion_matrix = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=2, normalize='true')

            df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=range(2), columns=range(2))
            plt.figure(figsize=(10, 7))
            sns.set(font_scale=2)
            ax = sns.heatmap(df_cm, annot=True, cmap='Blues')
            ax.set_xlabel('Predicted', fontsize=24)
            ax.set_ylabel('Ground Truth', fontsize=24)
            if self.loss_functional_name == 'mse':
                ax.set_xticklabels(labels=[-1, 1])
                ax.set_yticklabels(labels=[-1, 1])
            fig_ = ax.get_figure()
            plt.close(fig_)
            self.logger.experiment.add_figure(f'Confusion matrix ({self.loss_functional_name.capitalize()})',
                                              fig_,
                                              self.current_epoch)
        else:
            pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        test_loss = self.loss_functional(y_hat, y.unsqueeze(dim=1))
        self.log('test_loss', test_loss)
        if self.task == 'classification':
            if self.loss_functional_name == 'mse':
                y_preds = torch.sign(y_hat)
            else:
                y_preds = torch.where(torch.sigmoid(y_hat) <= 0.5, 0, 1)
            self.log('test_acc', torch.sum(y_preds == y.unsqueeze(dim=1)) / x.shape[0])
        return {'loss': test_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            'monitor': 'val_loss',
        }


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)

class Sampler:

    def __init__(self, model, img_shape, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs

class EBM_CNNModel(nn.Module):

    def __init__(self, hidden_features=32, out_dim=1, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2

        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
                nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4), # [16x16] - Larger padding to get 32x32 image
                Swish(),
                nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1), #  [8x8]
                Swish(),
                nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1), # [4x4]
                Swish(),
                nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1), # [2x2]
                Swish(),
                nn.Flatten(),
                nn.Linear(c_hid3*4, c_hid3),
                Swish(),
                nn.Linear(c_hid3, out_dim)
        )


    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x

class DeepEnergyModel(pl.LightningModule):

    def __init__(self, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = EBM_CNNModel(**CNN_args)
        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)


    def forward(self, x):
        z = self.cnn(x)
        return z


    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, _ = batch
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        # Logging
        self.log('loss', loss)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        return loss


    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log('val_contrastive_divergence', cdiv)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())
