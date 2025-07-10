import torch
import transformers


class WarmupThenReduceLROnPlateau(torch.optim.lr_scheduler._LRScheduler):
    """
    1) Linearly warm up LR from init_lr to peak_lr over `warmup_steps` steps.
    2) After warmup, switch to ReduceLROnPlateau with the given kwargs.
    """

    def __init__(self, optimizer, warmup_steps, init_lr, peak_lr, plateau_kwargs, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        # wrapped PyTorch ReduceLROnPlateau
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_kwargs)
        self._warmup_done = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            # Linear interpolation from init_lr to peak_lr
            pct = self.last_epoch / float(max(1, self.warmup_steps))
            return [self.init_lr + pct * (self.peak_lr - self.init_lr) for _ in self.optimizer.param_groups]
        else:
            # After warmup, leave LR to the plateau scheduler
            return [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self.get_lr()

    def step(self, metrics=None):
        if metrics is None:
            self.last_epoch += 1
            if self.last_epoch <= self.warmup_steps:
                for pg, lr in zip(self.optimizer.param_groups, self.get_lr()):
                    pg["lr"] = lr
        else:
            if self.last_epoch > self.warmup_steps:
                if not self._warmup_done:
                    self._warmup_done = True
                self.plateau.step(metrics)

    def state_dict(self):
        sd = super().state_dict()
        sd["plateau"] = self.plateau.state_dict()
        sd["_warmup_done"] = self._warmup_done
        return sd

    def load_state_dict(self, sd):
        self._warmup_done = sd.pop("_warmup_done", False)
        plateau_sd = sd.pop("plateau", None)
        super().load_state_dict(sd)
        if plateau_sd is not None:
            self.plateau.load_state_dict(plateau_sd)


SCHEDULER_DICT = {
    "WarmupThenReduceLROnPlateau": WarmupThenReduceLROnPlateau,
    "WarmupThenCosineDecay": transformers.get_cosine_schedule_with_warmup,
    "WarmupThenLinearDecay": transformers.get_linear_schedule_with_warmup,
    "WarmupThenConstant": transformers.get_constant_schedule_with_warmup,
}
