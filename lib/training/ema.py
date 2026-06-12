from copy import deepcopy

import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        decay=0.99,
        warm_up_iter=100,
        ema_steps=32,
    ):
        super(EMA, self).__init__()
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.n_iter = 0

        self.decay = decay
        self.warm_up_iter = max(1, warm_up_iter)
        self.ema_steps = ema_steps

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update_parameters(self, model: nn.Module):
        if self.n_iter > self.warm_up_iter:
            if (self.n_iter - self.warm_up_iter) % self.ema_steps == 0:
                for p_swa, p_model in zip(
                    self.model.state_dict().values(), model.state_dict().values()
                ):
                    if p_model.requires_grad:
                        p = self.decay * p_swa + (1 - self.decay) * p_model.detach()
                        p_swa.copy_(p)
                    else:
                        p_swa.copy_(p_model.detach())
        elif self.n_iter == self.warm_up_iter:
            for p_swa, p_model in zip(
                self.model.state_dict().values(), model.state_dict().values()
            ):
                p_swa.copy_(p_model.detach())

        self.n_iter = self.n_iter + 1


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    ema = EMA(model, warm_up_iter=100, ema_steps=10, decay=0.9)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    for i in range(10000):
        opt.zero_grad()
        x = torch.rand((64, 1)) * torch.pi * 2
        y = torch.sin(x)

        pred_y = model(x)
        loss = torch.nn.functional.smooth_l1_loss(pred_y, y, beta=0.001)

        loss.backward()
        opt.step()
        ema.update_parameters(model)
        if i % 100 == 0:
            print(f"#i {i}   {loss.item()}")

    model.eval()

    torch.save(ema.model.state_dict(), "test.pt")

    ema2 = EMA(model)
    ema2.model.load_state_dict(torch.load("test.pt"))

    with torch.no_grad():
        x = torch.rand((128, 1)) * torch.pi * 2
        y = torch.sin(x)
        pred_y = model(x)

        pred_y2 = ema(x)
        pred_y3 = ema2(x)
        loss = torch.nn.functional.mse_loss(pred_y, y)
        loss2 = torch.nn.functional.mse_loss(pred_y2, y)
        loss3 = torch.nn.functional.mse_loss(pred_y3, y)
        print(f"{loss.item()}   {loss2.item()}  {loss3.item()}")
