import torch
from torch.optim import Adam  # type: ignore


class PAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, lambda_p=1e-2, p_norm=1, *args, **kwargs):
        super(PAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, *args, **kwargs)
        self.p_norm = p_norm
        self.lambda_p = lambda_p

    @torch.no_grad()
    def step(self, closure=None):
        # Store the old params
        old_params = []
        for group in self.param_groups:
            old_params.append({param: param.data.clone() for param in group['params'] if param.grad is not None})

        # Perform the standard Adam step
        loss = super(PAdam, self).step(closure)

        # Perform the PAdam step
        for group, old_group in zip(self.param_groups, old_params):
            for param in group['params']:
                if param.grad is None:
                    continue

                # Use old parameters in the decay factor
                param_old = old_group[param]
                X = param_old.abs() ** (2 - self.p_norm)
                update_term = X / (X + self.p_norm * group['lr'] * self.lambda_p)

                # Update the parameters
                param.data.mul_(update_term)

        return loss
