import time
import numpy as np
import torch
import torch.nn as nn
from ema_pytorch import EMA
from scipy import integrate

class VelocityNetwork(nn.Module):
    """MLP used for the conditional velocity field b_t(x | y)."""

    def __init__(
        self,
        n_dim_x=1,
        n_dim_y=1,
        width=256,
        depth=4,
        activation="ReLU",
    ):
        super().__init__()

        activations = {
            "ReLU": nn.ReLU,
            "SiLU": nn.SiLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
            "LeakyReLU": nn.LeakyReLU,
            "ELU": nn.ELU,
            "GELU": nn.GELU,
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        activation_fn = activations[activation]

        layers = [nn.Linear(n_dim_x + n_dim_y + 4, width), activation_fn()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), activation_fn()])
        layers.append(nn.Linear(width, n_dim_x))

        self.fc = nn.Sequential(*layers) # fc := fully connected

    def forward(self, x, y, t):
        t = t.reshape(-1)
        embed = torch.stack(
            [
                t - 0.5,
                torch.cos(2 * torch.pi * t),
                torch.sin(2 * torch.pi * t),
                -torch.cos(4 * torch.pi * t),
            ],
            dim=-1,
        )
        return self.fc(torch.cat([x, y, embed], dim=-1))

class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching that learns """

    def __init__(
        self,
        n_dim_x=1,
        n_dim_y=1,
        width=256,
        depth=4,
        activation="ReLU",
        ema=True,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = VelocityNetwork(
            n_dim_x=n_dim_x,
            n_dim_y=n_dim_y,
            width=width,
            depth=depth,
            activation=activation,
        ).to(self.device)
        self.loss_func = nn.MSELoss()
        self.vel_ema = EMA(self.net, beta=0.99) if ema else None

        self.n_dim_x = n_dim_x
        self.n_dim_y = n_dim_y

    def forward(self, x, y, t, ema=False):
        return self.net(x, y, t) if not ema else self.vel_ema.ema_model(x, y, t)

    @staticmethod
    def x_t(x_0, x_1, t):
        return (1.0 - t) * x_0 + t * x_1

    @staticmethod
    def dot_x_t(x_0, x_1):
        return x_1 - x_0

    def loss(self, x_0, x_1, y, t, ema=False):
        x_t = self.x_t(x_0, x_1, t)
        target = self.dot_x_t(x_0, x_1)
        prediction = self(x_t, y, t, ema=ema)
        return self.loss_func(prediction, target)

    def train(
        self,
        X_train, # Assumes all datasets are torch tensors
        X_test,
        Y_train,
        Y_test,
        n_iters,
        batch_size,
        lr=1e-3,
    ):
        t0 = time.time() 

        X_train = X_train.reshape(-1, self.n_dim_x).to(self.device)
        Y_train = Y_train.reshape(-1, self.n_dim_y).to(self.device)

        X_test = X_test.reshape(-1, self.n_dim_x).to(self.device)
        Y_test = Y_test.reshape(-1, self.n_dim_y).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        train_losses = []
        test_losses = []

        for i in range(n_iters):
            self.net.train(mode=True)

            # Compute train loss
            idx = torch.randint(0, X_train.shape[0], (batch_size,)) # Randomly picks indices
            x_1, y = X_train[idx], Y_train[idx]
            x_0 = torch.randn_like(x_1).to(self.device) # Source = Gaussian
            t = torch.rand(batch_size, 1, device=self.device)

            loss = self.loss(x_0, x_1, y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.vel_ema is not None: self.vel_ema.update()

            train_losses.append(loss.item())

            # Compute test loss
            with torch.no_grad():
                self.net.eval()

                idx_test = torch.randint(0, X_test.shape[0], (batch_size,)) # Randomly picks indices
                x_1_test, y_test = X_test[idx_test], Y_test[idx_test]
                x_0_test = torch.randn_like(x_1_test).to(self.device)
                t_test = torch.rand(batch_size, 1, device=self.device)

                test_loss = self.loss(x_0_test, x_1_test, y_test, t_test, ema=self.vel_ema is not None)
                test_losses.append(test_loss.item())

            # Updating print statements
            if (i + 1) % 100 == 0: print(f"Iteration ({i+1:0{len(str(n_iters))}d}/{n_iters}) Training Loss = {train_losses[-1]:.4f} | Testing Loss = {test_losses[-1]:.4f}")

        # Compute the moving average test loss 
        window_size = min(1000, len(test_losses))
        initial_means = np.cumsum(test_losses[:window_size - 1]) / np.arange(1, window_size)
        moving_means = np.convolve(
            test_losses,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        ma_test_losses = np.concatenate([initial_means, moving_means])

        # delete everything occupying GPU memory
        if self.device == 'cuda': torch.cuda.empty_cache()

        t1 = time.time()
        print(f"\nFinished training in {t1 - t0:.1f} sec.")

        return train_losses, test_losses, ma_test_losses

    def sample(
        self,
        y_cond,
        batch_size,
        ema=True,
        return_path=False, # returns entire path of samples from t: 0 -> 1
    ):
        self.net.eval()
        if self.vel_ema is not None: self.vel_ema.ema_model.eval()

        x_0 = torch.randn(batch_size, self.n_dim_x, device=self.device)
        y_cond = y_cond.view(1, -1).expand(x_0.shape[0], -1).to(self.device)

        def ode_func(t, x):
            x_tensor = torch.as_tensor(
                x,
                device=self.device,
                dtype=torch.float32,
            ).reshape(x_0.shape)

            t_tensor = torch.full(
                (batch_size, 1),
                t,
                device=self.device,
                dtype=x_0.dtype,
            )

            with torch.no_grad():
                velocity = self(x_tensor, y_cond, t_tensor, ema=ema)

            return velocity.cpu().numpy().reshape(-1).astype(np.float64)

        result = integrate.solve_ivp(
            ode_func,
            (0.0, 1.0),
            x_0.cpu().numpy().reshape(-1),
            rtol=1e-5,
            atol=1e-5,
            method="RK45",
        )

        n_steps_used = result.t.shape[0]

        # result.y has shape (batch_size * n_dim_x, n_steps_used)
        path = torch.tensor(
            result.y,
            device=self.device,
            dtype=torch.float32,
        ).reshape(
            x_0.shape[0],
            x_0.shape[1],
            n_steps_used,
        )

        if return_path: 
            # Shape: (n_steps, batch_size, n_dim_x)
            return path.permute(2, 0, 1).detach().cpu().numpy(), n_steps_used

        final_samples = path[:, :, -1]
        return final_samples.detach().cpu().numpy(), n_steps_used 

