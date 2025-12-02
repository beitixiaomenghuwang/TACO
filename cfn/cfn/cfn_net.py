import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(dim)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x


class CoinFlippingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=64, hidden_dim=256, depth=3, drop=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_act = nn.GELU()
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim, hidden_dim * 4, drop=drop)
            for _ in range(depth)
        ])

        self.output = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output.weight);     nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_act(x)
        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)
        return self.output(x)


class cfn_net(nn.Module):
    def __init__(self, input_dim, output_dim=64, cfn_hidden_dim=256):
        super().__init__()
        self.cfn = CoinFlippingNetwork(
            input_dim=input_dim, 
            output_dim=output_dim,
            hidden_dim=cfn_hidden_dim,
        )

        # Deep copy for prior model
        self.prior_cfn = copy.deepcopy(self.cfn)
        for param in self.prior_cfn.parameters():
            param.requires_grad = False
        
        prior_mean = torch.nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)
        prior_var = torch.nn.Parameter(torch.full((1, output_dim), 0.01), requires_grad=False)
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_var", prior_var)

        # Buffer to store prior outputs
        self.prior_outputs = None
        self.iter_count = 0
        self.output_dim = output_dim

    def update_prior(self):
        """Incrementally update the prior mean and variance."""
        for i in range(self.prior_outputs.shape[0]):
            self.iter_count += 1
            
            # Compute delta between the new output and current mean
            delta = self.prior_outputs[i] - self.prior_mean
            
            # Update mean: mean = mean + delta / iter_count
            self.prior_mean.data += delta / self.iter_count
            
            # Update variance: variance = variance + delta^2 / iter_count
            delta_sq = delta ** 2
            self.prior_var.data += (delta_sq - self.prior_var) / self.iter_count

    def forward(self, features):
        # Record prior output (called during forward pass to update the buffer)
        features = features * 10
        with torch.no_grad():
            self.prior_outputs = self.prior_cfn(features)
            prior = (self.prior_outputs - self.prior_mean) / torch.sqrt(self.prior_var + 1e-6)
        coin_flipping = self.cfn(features) + prior
        self.prior = prior
        return coin_flipping


class CFN(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        cfn_output_dim=64,
        cfn_hidden_dim=256
        ):
        super().__init__()

        print(f"cfn network hidden dim is {cfn_hidden_dim}")
        self.cfn = cfn_net(
            input_dim=input_dim,
            output_dim=cfn_output_dim,
            cfn_hidden_dim=cfn_hidden_dim
        )

    
    def compute_loss_feature(self, feature_batch):
        for key in feature_batch:
            if isinstance(feature_batch[key], torch.Tensor):
                feature_batch[key] = feature_batch[key].to(next(self.parameters()).device)
        feature_batch["feature"] = feature_batch["feature"].to(next(self.parameters()).dtype)
        preds = self.cfn(feature_batch["feature"])
        self.cfn.update_prior()
        targets = feature_batch["CoinFlip_target"].float().to(next(self.parameters()).device)
        loss = F.mse_loss(preds, targets)
        return loss, preds



