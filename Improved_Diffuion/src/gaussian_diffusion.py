import torch
import torch.nn as nn
from data.pyg_dataToGraph import DataToGraph
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from notebooks.losses import normal_kl, discretized_gaussian_log_likelihood
#%%
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
#%%
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
#%%

class GaussianDiffusion:
    def __init__(
        self,
        num_steps,
        #model,
        #classifier,
        classifier_scale = 1.0
        #model_mean_type = "epsilon",
        #model_var_type,
        #loss_type,
        #rescale_timesteps=False
        ):
        self.num_steps = num_steps
        #self.model = model
        #self.classifier = classifier
        self.classifier_scale = classifier_scale
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # # 生成时间步的序列
        # self.t = torch.linspace(0, 1, num_steps + 1)  # 主要时间步范围从 0 到 1
        #
        # # 使用余弦调度生成 betas
        # self.betas = torch.cos(torch.pi / 2.0 * self.t) ** 2  # 余弦平方函数
        # self.betas = self.betas / self.betas.max()  # 归一化到 0-1 范围
        # self.betas = torch.flip(self.betas, [0])  # 反转顺序，以确保从小到大递增
        # self.betas = torch.clamp(self.betas, min=1e-5, max=0.5e-2)  # 调整范围到 (1e-5, 0.5e-2)

        # 生成betas序列
        self.betas = self._cosine_beta_schedule_with_offset(
            num_steps=num_steps,
            max_beta=0.999,      # 可调节参数
            offset=0.008         # Improved DDPM 标准偏移量
        )

        # 计算 alpha , alpha_prod , alpha_prod_previous , alpha_bar_sqrt 等变量的值
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 累积连乘
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1]).float(), self.alphas_cumprod[:-1]], 0)  # p means previous
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        # self.posterior_log_variance_clipped = torch.log(
        #     torch.append(self.posterior_variance[1], self.posterior_variance[1:])
        # )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([
                self.posterior_variance[1:2],  # 保持维度一致
                self.posterior_variance[1:]
            ])
        )

        # 计算后验参数coef1和coef2
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # 将超参数也移动到GPU
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(self.device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(self.device)

    def _cosine_beta_schedule_with_offset(self, num_steps, max_beta=0.999, offset=0.008):
        """Improved DDPM 余弦调度核心函数"""
        # 生成连续时间点 (包括端点)
        t = torch.linspace(0, 1, num_steps + 1)

        # 计算 alpha_bar (累积乘积)
        alpha_bars = torch.cos((t + offset) / (1 + offset) * torch.pi / 2) ** 2

        # 计算 beta 值
        betas = []
        for i in range(num_steps):
            beta = 1 - (alpha_bars[i+1] / alpha_bars[i])
            betas.append(min(beta.item(), max_beta))

        # 强制单调递增
        for i in range(1, len(betas)):
            if betas[i] < betas[i-1]:
                betas[i] = betas[i-1]

        return torch.tensor(betas, dtype=torch.float32)

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, batch_labels = None, clip_denoised = True
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, batch_labels, clip_denoised=clip_denoised
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def cond_fn(self, x, t, y):
        """分类器梯度计算函数"""
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale

    def condition_mean(self, p_mean_var, x, t, batch_y):
        gradient = self.cond_fn(x, t, batch_y)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def q_mean_variance(self, x_start, t):
        # 前向计算mean和variance，根据前向扩散公式推导可得mean和variance
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, batch_labels = None, clip_denoised=True, denoised_fn=None,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, batch_labels)

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(torch.log(self.betas), t, x.shape)
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
        model_mean = model_output

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self,
                 model,
                 x,
                 t,
                 batch_labels,
                 clip_denoised = True,
                 denoised_fn = None,
                 ):
        out = self.p_mean_variance(model, x, t, batch_labels, clip_denoised, denoised_fn)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        out["mean"] = self.condition_mean(
             out, x, t, batch_labels
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_progressive(
            self,
            shape,
            batch_labels,
            noise = None,
            clip_denoised=True,
            denoised_fn=None,
            progress=False,
    ):
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=self.device)
        indices = list(range(self.num_steps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                out = self.p_sample(
                    img,
                    t,
                    batch_labels,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                )
                yield out
                img = out["sample"]

    def p_sample_loop(
            self,
            shape,
            batch_labels,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            progress=False,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            shape,
            batch_labels,
            noise = noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def training_losses(self, model, x_start, t, batch_labels, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        model_output = model(x_t, t, batch_labels)  # 完整模型输出 [B, 2C, ...]

        B, C = x_t.shape[:2]  # 修改点1：确保获取正确的通道维度
        assert model_output.shape == (B, C * 2, *x_t.shape[2:])

        # 分割噪声预测和方差预测
        model_output, model_var_values = torch.split(model_output, C, dim=1)

        # 关键修改点2：创建冻结梯度后的拼接输出
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)  # 冻结噪声预测梯度

        # 修改点3：使用 frozen_out 代替原始模型
        terms["vb"] = self._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r,  # 直接返回预计算的 frozen_out
            x_start=x_start,
            x_t=x_t,
            t=t,
            batch_labels=batch_labels,  # 注意：需要确认 _vb_terms_bpd 是否支持这个参数
            clip_denoised=False
        )["output"]

        # 强制使用 PREVIOUS_X 目标
        target, _ = self.q_posterior_mean_variance(x_start, x_t, t)
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"] + terms["vb"]

        return terms


# class GaussianDiffusion:
#     def __init__(
#         self,
#         num_steps,
#         model,
#         classifier,
#         classifier_scale = 1.0
#         #model_mean_type = "epsilon",
#         #model_var_type,
#         #loss_type,
#         #rescale_timesteps=False
#         ):
#         self.num_steps = num_steps
#         self.model = model
#         self.classifier = classifier
#         self.classifier_scale = classifier_scale
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         # # 生成时间步的序列
#         # self.t = torch.linspace(0, 1, num_steps + 1)  # 主要时间步范围从 0 到 1
#         #
#         # # 使用余弦调度生成 betas
#         # self.betas = torch.cos(torch.pi / 2.0 * self.t) ** 2  # 余弦平方函数
#         # self.betas = self.betas / self.betas.max()  # 归一化到 0-1 范围
#         # self.betas = torch.flip(self.betas, [0])  # 反转顺序，以确保从小到大递增
#         # self.betas = torch.clamp(self.betas, min=1e-5, max=0.5e-2)  # 调整范围到 (1e-5, 0.5e-2)
#
#         # 生成betas序列
#         self.betas = self._cosine_beta_schedule_with_offset(
#             num_steps=num_steps,
#             max_beta=0.999,      # 可调节参数
#             offset=0.008         # Improved DDPM 标准偏移量
#         )
#
#         # 计算 alpha , alpha_prod , alpha_prod_previous , alpha_bar_sqrt 等变量的值
#         self.alphas = 1 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 累积连乘
#         self.alphas_cumprod_prev = torch.cat([torch.tensor([1]).float(), self.alphas_cumprod[:-1]], 0)  # p means previous
#         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#         self.posterior_variance = (
#             self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
#         )
#         self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
#         # self.posterior_log_variance_clipped = torch.log(
#         #     torch.append(self.posterior_variance[1], self.posterior_variance[1:])
#         # )
#         self.posterior_log_variance_clipped = torch.log(
#             torch.cat([
#                 self.posterior_variance[1:2],  # 保持维度一致
#                 self.posterior_variance[1:]
#             ])
#         )
#
#         # 计算后验参数coef1和coef2
#         self.posterior_mean_coef1 = (
#             self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
#         )
#         self.posterior_mean_coef2 = (
#             (1.0 - self.alphas_cumprod_prev)
#             * np.sqrt(self.alphas)
#             / (1.0 - self.alphas_cumprod)
#         )
#
#         # 将超参数也移动到GPU
#         self.betas = self.betas.to(self.device)
#         self.alphas = self.alphas.to(self.device)
#         self.alphas_cumprod = self.alphas_cumprod.to(self.device)
#         self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
#         self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
#         self.posterior_variance = self.posterior_variance.to(self.device)
#         self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(self.device)
#         self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
#         self.posterior_mean_coef1 = self.posterior_mean_coef1.to(self.device)
#         self.posterior_mean_coef2 = self.posterior_mean_coef2.to(self.device)
#
#     def _cosine_beta_schedule_with_offset(self, num_steps, max_beta=0.999, offset=0.008):
#         """Improved DDPM 余弦调度核心函数"""
#         # 生成连续时间点 (包括端点)
#         t = torch.linspace(0, 1, num_steps + 1)
#
#         # 计算 alpha_bar (累积乘积)
#         alpha_bars = torch.cos((t + offset) / (1 + offset) * torch.pi / 2) ** 2
#
#         # 计算 beta 值
#         betas = []
#         for i in range(num_steps):
#             beta = 1 - (alpha_bars[i+1] / alpha_bars[i])
#             betas.append(min(beta.item(), max_beta))
#
#         # 强制单调递增
#         for i in range(1, len(betas)):
#             if betas[i] < betas[i-1]:
#                 betas[i] = betas[i-1]
#
#         return torch.tensor(betas, dtype=torch.float32)
#
#     def _predict_xstart_from_xprev(self, x_t, t, xprev):
#         assert x_t.shape == xprev.shape
#         return (  # (xprev - coef2*x_t) / coef1
#             _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
#             - _extract_into_tensor(
#                 self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
#             )
#             * x_t
#         )
#
#     def _vb_terms_bpd(
#             self, model, x_start, x_t, t, batch_labels = None, clip_denoised = True
#     ):
#         true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
#             x_start=x_start, x_t=x_t, t=t
#         )
#         out = self.p_mean_variance(
#             model, x_t, t, batch_labels, clip_denoised=clip_denoised
#         )
#         kl = normal_kl(
#             true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
#         )
#         kl = mean_flat(kl) / np.log(2.0)
#
#         decoder_nll = -discretized_gaussian_log_likelihood(
#             x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
#         )
#         assert decoder_nll.shape == x_start.shape
#         decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
#
#         # At the first timestep return the decoder NLL,
#         # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
#         output = torch.where((t == 0), decoder_nll, kl)
#         return {"output": output, "pred_xstart": out["pred_xstart"]}
#
#     def cond_fn(self, x, t, y):
#         """分类器梯度计算函数"""
#         assert y is not None
#         with torch.enable_grad():
#             x_in = x.detach().requires_grad_(True)
#             logits = self.classifier(x_in, t)
#             log_probs = F.log_softmax(logits, dim=-1)
#             selected = log_probs[range(len(logits)), y.view(-1)]
#             return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale
#
#     def condition_mean(self, p_mean_var, x, t, batch_y):
#         gradient = self.cond_fn(x, t, batch_y)
#         new_mean = (
#             p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
#         )
#         return new_mean
#
#     def q_mean_variance(self, x_start, t):
#         # 前向计算mean和variance，根据前向扩散公式推导可得mean和variance
#         mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
#         variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
#         log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
#         return mean, variance, log_variance
#
#     def q_sample(self, x_start, t, noise = None):
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         assert noise.shape == x_start.shape
#
#         return (
#                 extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
#                 + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
#         )
#
#     def q_posterior_mean_variance(self, x_start, x_t, t):
#         assert x_start.shape == x_t.shape
#         posterior_mean = (
#             extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
#             + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_variance = extract(self.posterior_variance, t, x_t.shape)
#         posterior_log_variance_clipped = extract(
#             self.posterior_log_variance_clipped, t, x_t.shape
#         )
#         assert (
#             posterior_mean.shape[0]
#             == posterior_variance.shape[0]
#             == posterior_log_variance_clipped.shape[0]
#             == x_start.shape[0]
#         )
#         return posterior_mean, posterior_variance, posterior_log_variance_clipped
#
#     def p_mean_variance(
#             self, model, x, t, batch_labels = None, clip_denoised=True, denoised_fn=None,
#     ):
#         B, C = x.shape[:2]
#         assert t.shape == (B,)
#         model_output = model(x, t, batch_labels)
#
#         assert model_output.shape == (B, C * 2, *x.shape[2:])
#         model_output, model_var_values = torch.split(model_output, C, dim=1)
#         min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
#         max_log = extract(torch.log(self.betas), t, x.shape)
#         frac = (model_var_values + 1) / 2
#         model_log_variance = frac * max_log + (1 - frac) * min_log
#         model_variance = torch.exp(model_log_variance)
#
#         def process_xstart(x):
#             if denoised_fn is not None:
#                 x = denoised_fn(x)
#             if clip_denoised:
#                 return x.clamp(-1, 1)
#             return x
#
#         pred_xstart = process_xstart(
#                 self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
#             )
#         model_mean = model_output
#
#         assert (
#             model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
#         )
#         return {
#             "mean": model_mean,
#             "variance": model_variance,
#             "log_variance": model_log_variance,
#             "pred_xstart": pred_xstart,
#         }
#
#     def p_sample(self,
#                  x,
#                  t,
#                  batch_labels,
#                  clip_denoised = True,
#                  denoised_fn = None,
#                  ):
#         out = self.p_mean_variance(self.model, x, t, batch_labels, clip_denoised, denoised_fn)
#         noise = torch.randn_like(x)
#         nonzero_mask = (
#             (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
#         )
#         out["mean"] = self.condition_mean(
#              out, x, t, batch_labels
#         )
#         sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
#         return {"sample": sample, "pred_xstart": out["pred_xstart"]}
#
#     def p_sample_loop_progressive(
#             self,
#             shape,
#             batch_labels,
#             noise = None,
#             clip_denoised=True,
#             denoised_fn=None,
#             progress=False,
#     ):
#         assert isinstance(shape, (tuple, list))
#         if noise is not None:
#             img = noise
#         else:
#             img = torch.randn(*shape, device=self.device)
#         indices = list(range(self.num_steps))[::-1]
#
#         if progress:
#             # Lazy import so that we don't depend on tqdm.
#             from tqdm.auto import tqdm
#
#             indices = tqdm(indices)
#
#         for i in indices:
#             t = torch.tensor([i] * shape[0], device=self.device)
#             with torch.no_grad():
#                 out = self.p_sample(
#                     img,
#                     t,
#                     batch_labels,
#                     clip_denoised=clip_denoised,
#                     denoised_fn=denoised_fn,
#                 )
#                 yield out
#                 img = out["sample"]
#
#     def p_sample_loop(
#             self,
#             shape,
#             batch_labels,
#             noise=None,
#             clip_denoised=True,
#             denoised_fn=None,
#             progress=False,
#     ):
#         final = None
#         for sample in self.p_sample_loop_progressive(
#             shape,
#             batch_labels,
#             noise = noise,
#             clip_denoised=clip_denoised,
#             denoised_fn=denoised_fn,
#             progress=progress,
#         ):
#             final = sample
#         return final["sample"]
#
#     def training_losses(self, model, x_start, t, batch_labels, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         x_t = self.q_sample(x_start, t, noise=noise)
#
#         terms = {}
#         model_output = model(x_t, t, batch_labels)  # 完整模型输出 [B, 2C, ...]
#
#         B, C = x_t.shape[:2]  # 修改点1：确保获取正确的通道维度
#         assert model_output.shape == (B, C * 2, *x_t.shape[2:])
#
#         # 分割噪声预测和方差预测
#         model_output, model_var_values = torch.split(model_output, C, dim=1)
#
#         # 关键修改点2：创建冻结梯度后的拼接输出
#         frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)  # 冻结噪声预测梯度
#
#         # 修改点3：使用 frozen_out 代替原始模型
#         terms["vb"] = self._vb_terms_bpd(
#             model=lambda *args, r=frozen_out: r,  # 直接返回预计算的 frozen_out
#             x_start=x_start,
#             x_t=x_t,
#             t=t,
#             batch_labels=batch_labels,  # 注意：需要确认 _vb_terms_bpd 是否支持这个参数
#             clip_denoised=False
#         )["output"]
#
#         # 强制使用 PREVIOUS_X 目标
#         target, _ = self.q_posterior_mean_variance(x_start, x_t, t)
#         assert model_output.shape == target.shape == x_start.shape
#         terms["mse"] = mean_flat((target - model_output) ** 2)
#         terms["loss"] = terms["mse"] + terms["vb"]
#
#         return terms
