import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sqrt
from torch.special import expm1

from einops import repeat
from tqdm import tqdm

from model.helper import *

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: UViT,
        *,
        image_size,
        channels = 3,
        pred_objective,
        noise_schedule = logsnr_schedule_cosine,
        noise_d = 64,
        noise_d_low = None,
        noise_d_high = None,
        num_sample_steps = 500,
        clip_sample_denoised = True,
    ):
        super().__init__()

        assert pred_objective in {'v', 'eps', 'x0'}, 'whether to predict v-space (progressive distillation paper) or noise'
        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'

        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.pred_objective = pred_objective
        self.log_snr = noise_schedule
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)
        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), 'both noise_d_low and noise_d_high must be set'
            self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        """
        Compute the mean and variance of the posterior for the current and next time step.

        Args:
            x (torch.Tensor): Noised image, shape (N, C, H, W)
            time (torch.Tensor): Current time step, shape (N,)
            time_next (torch.Tensor): Next time step, shape (N,)

        Returns:
            model_mean (torch.Tensor): Mean of the posterior, shape (N, C, H, W)
            posterior_variance (torch.Tensor): Variance of the posterior, shape (N, 1, 1, 1) or broadcastable
        """
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred = self.model(x, batch_log_snr)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred
        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        # x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance


    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        """
        Sample from the posterior at a given time step.

        Args:
            x (torch.Tensor): Current image, shape (N, C, H, W)
            time (float or torch.Tensor): Current time step, shape () or (N,)
            time_next (float or torch.Tensor): Next time step, shape () or (N,)

        Returns:
            torch.Tensor: Sampled image at next time step, shape (N, C, H, W)
        """
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        Run the full reverse diffusion process to sample images.

        Args:
            shape (tuple): Shape of the output image (N, C, H, W)

        Returns:
            torch.Tensor: Sampled images, shape (N, C, H, W)
        """
        batch = shape[0]

        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        # img.clamp_(-1., 1.)
        # img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))


    def q_sample(self, x_start, times, noise = None):
        """
        Diffuse the input image to a given time step.

        Args:
            x_start (torch.Tensor): Original image, shape (N, C, H, W)
            times (torch.Tensor): Time steps, shape (N,)
            noise (torch.Tensor, optional): Noise to add, shape (N, C, H, W)

        Returns:
            x_noised (torch.Tensor): Noised image, shape (N, C, H, W)
            log_snr (torch.Tensor): Log SNR at given times, shape (N,)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    def p_losses(self, x_start, times, noise = None):
        """
        Compute the training loss for a batch.

        Args:
            x_start (torch.Tensor): Original image, shape (N, C, H, W)
            times (torch.Tensor): Time steps, shape (N,)
            noise (torch.Tensor, optional): Noise to add, shape (N, C, H, W)

        Returns:
            torch.Tensor: Scalar loss value
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        model_out = self.model(x, log_snr)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start
        elif self.pred_objective == 'eps':
            target = noise
        return F.mse_loss(model_out, target)

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # img = normalize_to_neg_one_to_one(img)
        times = torch.zeros((img.shape[0],), device = self.device).float().uniform_(0, 1)

        return self.p_losses(img, times, *args, **kwargs)

class CondGaussianDiffusion(GaussianDiffusion):
    def __init__(
            self,
            model,
            *,
            image_size,
            channels,
            extra_channels = 0,
            cond_channels = 1,
            pred_objective,
            loss_type,
            noise_schedule = logsnr_schedule_cosine,
            noise_d = 64,
            noise_d_low = None,
            noise_d_high = None,
            num_sample_steps,
            clip_sample_denoised = True,
    ):
        super(CondGaussianDiffusion, self).__init__(model, image_size=image_size, channels=channels,
                                                    pred_objective=pred_objective, noise_schedule=noise_schedule,
                                                    noise_d=noise_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high,
                                                    num_sample_steps=num_sample_steps,
                                                    clip_sample_denoised=clip_sample_denoised)
        if loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            try:
                from utils.import_utils import get_obj_from_str
                loss_type = get_obj_from_str(loss_type)
            except:
                raise NotImplementedError

        self.loss_type = loss_type
        self.extra_channels = extra_channels
        self.cond_channels = cond_channels

        # This history is used to store the history of the x_start in reverse diffusion process
        # When call the sample function, the history will be reset and store the x_start of each sample
        self.history = []

    def forward(self, img, cond_img, seg=None, extra_cond=None, *args, **kwargs):
        """
        Forward pass for conditional diffusion training.

        Args:
            img (torch.Tensor): Input image, shape (N, C, H, W)
            cond_img (torch.Tensor): Conditioning image, shape (N, C_cond, H, W)
            seg (torch.Tensor, optional): Segmentation mask, shape (N, C, H, W)
            extra_cond (torch.Tensor, optional): Extra conditioning, shape (N, C_extra, H, W)

        Returns:
            torch.Tensor: Scalar loss value
        """
        b, channels, h, w = img.shape
        cond_channels = cond_img.shape[1]
        assert channels == self.channels, f"Channels mismatch: {channels} != {self.channels}"
        assert h == w == self.image_size, f"Image size mismatch: {h} != {self.image_size} or {w} != {self.image_size}"
        assert cond_channels == self.cond_channels, f"Condition channels mismatch: {cond_channels} != {self.cond_channels}"

        # img = normalize_to_neg_one_to_one(img)
        seg = normalize_to_neg_one_to_one(seg) if seg is not None else None
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        times = torch.zeros((img.shape[0],), device=self.device).float().uniform_(0, 1)
        return self.p_losses(img, times, cond_img, seg, extra_cond, *args, **kwargs)

    def p_losses(self, x_start, times, cond_img, seg=None, extra_cond=None, noise=None, *args, **kwargs):
        """
        Compute the training loss for a batch in the conditional setting.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample, if seg is not None, sample from seg.
        x, log_snr = self.q_sample(x_start=seg, times=times, noise=noise) if seg is not None else self.q_sample(x_start=x_start, times=times, noise=noise)

        # predict and take gradient step
        model_out = self.model(torch.cat([x, extra_cond], dim=1), log_snr, cond_img)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start
            # pred_x0 = alpha * x - sigma * model_out
            # true_x0 = alpha * x - sigma * target
        elif self.pred_objective == 'eps':
            target = noise
        elif self.pred_objective == 'x0':
            # Note: x_start in here is from -1 to 1
            target = x_start

        # Apply proper output activation for custom losses
        # if self.loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            # if self.pred_objective == 'x0':
                # For segmentation losses, convert model output to [0,1] range
                # but keep target in same range
                # model_out = torch.sigmoid(model_out)
                # target = (target + 1) / 2  # Convert target from [-1,1] to [0,1]

        if self.loss_type == 'l2':
            return F.mse_loss(model_out, target)
        elif self.loss_type == 'l1':
            return F.l1_loss(model_out, target)
        elif self.loss_type == 'l1+l2':
            return F.mse_loss(model_out, target) + F.l1_loss(model_out, target)
        elif self.loss_type == 'mean(l1, l2)':
            return (F.mse_loss(model_out, target) + F.l1_loss(model_out, target)) / 2
        else:
            return self.loss_type(model_out, target)

    @torch.no_grad()
    def sample(self, cond_img, extra_cond=None, verbose=True):
        """
        Generate samples from the conditional model.
        """
        b, c, h, w = cond_img.shape
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        return self.p_sample_loop((b, self.channels, self.image_size, self.image_size),
                                  cond_img, extra_cond=extra_cond, verbose=verbose)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_img, extra_cond, verbose=True):
        """
        Run the full reverse diffusion process to sample images in the conditional setting.

        Args:
            shape (tuple): Shape of the output image (N, C, H, W)
            cond_img (torch.Tensor): Conditioning image, shape (N, C_cond, H, W)
            extra_cond (torch.Tensor): Extra conditioning, shape (N, C_extra, H, W)
            verbose (bool): Whether to show progress bar.

        Returns:
            torch.Tensor: Sampled images, shape (N, C, H, W)
        """
        self.history = []
        img = torch.randn(shape, device=self.device)
        conditioning_features = self.model.extract_features(cond_img)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps,
                      disable=not verbose):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, conditioning_features, extra_cond, times, times_next)

        # img.clamp_(-1., 1.)
        # img = (img + 1) / 2
        return img

    @torch.no_grad()
    def p_sample(self, x, cond, extra_cond, time, time_next):
        """
        Sample from the posterior at a given time step in the conditional setting.
        """
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, cond=cond, extra_cond=extra_cond, time=time,
                                                          time_next=time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    def p_mean_variance(self, x, cond, extra_cond, time, time_next):
        """
        Compute the mean and variance of the posterior for the current and next time step in the conditional setting.
        """
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.model.sample_unet(torch.cat([x, extra_cond], dim=1),
                                      batch_log_snr, cond)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred
        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha
        elif self.pred_objective == 'x0':
            x_start = pred

        # x_start.clamp_(-1., 1.)
        self.history.append(x_start) # change to pred when generate cam

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance