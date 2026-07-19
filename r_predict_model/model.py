"""
Reward-only ensemble model used by the MBPO training entry point.

The original MBPO template predicts reward plus state deltas for continuous
control tasks.  In this project the SAC observation is a PSD image plus FHSS
metadata, and synthetic rollouts are intentionally limited to one step.  The
model therefore learns only:

    flattened(state_img, hoprate, block_idx, action) -> reward

The ensemble predicts both reward mean and reward variance.  During rollout,
only elite networks selected by holdout loss are sampled, which follows the
standard MBPO practice of using an ensemble to reduce model bias.
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardScaler:
    """
    Feature-wise standardization helper for model inputs.

    Neural dynamics models are sensitive to input scale.  The scaler stores the
    mean and standard deviation computed from the training split, then reuses
    them for holdout evaluation and inference so all data enters the ensemble in
    the same normalized coordinate system.
    """

    def __init__(self):
        self.mu = None
        self.std = None

    def fit(self, data):
        """Estimate feature mean and standard deviation from a 2-D array."""
        data = np.asarray(data, dtype=np.float32)
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # Constant features would divide by zero; keeping their scale at one
        # leaves those dimensions centered without amplifying numerical noise.
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Apply the fitted normalization to input features."""
        if self.mu is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        return (np.asarray(data, dtype=np.float32) - self.mu) / self.std

    def inverse_transform(self, data):
        """Map normalized data back to the original feature scale."""
        if self.mu is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform().")
        return self.std * data + self.mu


def _truncated_normal_(tensor, mean=0.0, std=0.01):
    """
    Fill ``tensor`` with samples from a normal distribution truncated to 2 std.

    PyTorch has native truncation utilities in newer versions, but this local
    implementation keeps the project independent of a specific torch release.
    """
    with torch.no_grad():
        torch.nn.init.normal_(tensor, mean=mean, std=std)
        while True:
            cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
            if not torch.sum(cond):
                break
            tensor[cond] = torch.normal(
                mean=mean,
                std=std,
                size=(int(cond.sum().item()),),
                device=tensor.device,
            )


def init_weights(module):
    """Initialize linear and ensemble-linear layers with MBPO-style weights."""
    if isinstance(module, (nn.Linear, EnsembleFC)):
        input_dim = module.in_features
        _truncated_normal_(module.weight, std=1 / (2 * np.sqrt(input_dim)))
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    """
    Fully connected layer evaluated for all ensemble members in parallel.

    Shapes:
        input_tensor: [ensemble_size, batch_size, in_features]
        weight:       [ensemble_size, in_features, out_features]
        output:       [ensemble_size, batch_size, out_features]

    Each ensemble member owns a separate weight matrix and bias vector.  Batched
    matrix multiplication lets all members process the same minibatch without a
    Python loop.
    """

    def __init__(
        self,
        in_features,
        out_features,
        ensemble_size=5,
        weight_decay=0.0,
        bias=True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.ensemble_size = int(ensemble_size)
        self.weight_decay = float(weight_decay)
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset weights and bias for every ensemble member."""
        _truncated_normal_(self.weight, std=1 / (2 * np.sqrt(self.in_features)))
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, input_tensor):
        """Apply the ensemble-specific affine transform."""
        output = torch.bmm(input_tensor, self.weight)
        if self.bias is not None:
            output = output + self.bias[:, None, :]
        return output


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class EnsembleModel(nn.Module):
    """
    Probabilistic neural ensemble that predicts reward mean and variance.

    The network has five ensemble-aware linear layers.  The final layer emits
    ``2 * output_dim`` values for each model member: the first half is the mean,
    and the second half is a bounded log-variance.  The bounded variance prevents
    unstable confidence estimates from dominating the loss during early
    training.
    """

    def __init__(
        self,
        input_size,
        output_size,
        ensemble_size,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
        device=None,
    ):
        super().__init__()
        self.output_dim = int(output_size)
        self.use_decay = bool(use_decay)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.nn1 = EnsembleFC(input_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        # Non-trainable variance bounds used by the probabilistic MBPO loss.
        self.max_logvar = nn.Parameter(
            torch.ones((1, self.output_dim), device=self.device) / 2,
            requires_grad=False,
        )
        self.min_logvar = nn.Parameter(
            -torch.ones((1, self.output_dim), device=self.device) * 10,
            requires_grad=False,
        )
        self.swish = Swish()
        self.apply(init_weights)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, ret_log_var=False):
        """
        Run a forward pass through every ensemble member.

        Args:
            x: Tensor with shape [network_size, batch_size, input_size].
            ret_log_var: When true, return log-variance instead of variance.

        Returns:
            mean and either log-variance or variance, each with shape
            [network_size, batch_size, output_dim].
        """
        x = self.swish(self.nn1(x))
        x = self.swish(self.nn2(x))
        x = self.swish(self.nn3(x))
        x = self.swish(self.nn4(x))
        output = self.nn5(x)
        mean = output[:, :, :self.output_dim]
        logvar = self.max_logvar - F.softplus(self.max_logvar - output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if ret_log_var:
            return mean, logvar
        return mean, torch.exp(logvar)

    def get_decay_loss(self):
        """Compute ensemble layer weight decay using each layer's configured scale."""
        decay_loss = 0.0
        for module in self.children():
            if isinstance(module, EnsembleFC):
                decay_loss += module.weight_decay * torch.sum(torch.square(module.weight)) / 2.0
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        Compute probabilistic ensemble loss.

        With ``inc_var_loss=True``, errors are weighted by inverse variance and
        a log-variance penalty is added, so the model must explain both mean
        prediction error and predictive uncertainty.  With ``False``, the method
        returns plain per-network MSE, which is used for holdout ranking.
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        if inc_var_loss:
            inv_var = torch.exp(-logvar)
            mse_loss = torch.mean(torch.mean(torch.square(mean - labels) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.square(mean - labels), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def update(self, loss):
        """Backpropagate one optimizer step for the ensemble model."""
        self.optimizer.zero_grad()
        loss = loss + 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss = loss + self.get_decay_loss()
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    """
    User-facing wrapper around ``EnsembleModel`` for MBPO reward prediction.

    Responsibilities:
        - normalize input features;
        - split data into training and holdout sets;
        - train every ensemble member with independent minibatch order;
        - select elite models by holdout MSE;
        - expose deterministic or stochastic reward predictions.
    """

    def __init__(
        self,
        network_size,
        elite_size,
        state_size,
        action_size,
        reward_size=1,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
        device=None,
    ):
        self.network_size = int(network_size)
        self.elite_size = int(elite_size)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.reward_size = int(reward_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.elite_model_idxes = list(range(min(self.elite_size, self.network_size)))
        self.scaler = StandardScaler()
        self.ensemble_model = EnsembleModel(
            input_size=self.state_size + self.action_size,
            output_size=self.reward_size,
            ensemble_size=self.network_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            use_decay=use_decay,
            device=self.device,
        )
        self.last_train_stats = {}

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0.2, max_epochs_since_update=5):
        """
        Train the ensemble on supervised reward targets.

        Args:
            inputs: Array [num_samples, state_size + action_size].
            labels: Array [num_samples] or [num_samples, reward_size].
            batch_size: Minibatch size per ensemble member.
            holdout_ratio: Fraction of samples reserved for validation and
                elite model selection.
            max_epochs_since_update: Early-stop patience measured in epochs
                without meaningful holdout improvement.

        Returns:
            Dictionary containing epoch count, per-model holdout losses, elite
            model indices, and mean holdout loss.
        """
        inputs = np.asarray(inputs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if inputs.shape[0] < 2:
            raise ValueError("Need at least two samples to train the ensemble reward model.")

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        num_holdout = min(max(1, num_holdout), inputs.shape[0] - 1)
        # Shuffle before splitting so the holdout set is not biased by replay
        # insertion order.
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs_t = torch.from_numpy(holdout_inputs).float().to(self.device)
        holdout_labels_t = torch.from_numpy(holdout_labels).float().to(self.device)
        holdout_inputs_t = holdout_inputs_t[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels_t = holdout_labels_t[None, :, :].repeat([self.network_size, 1, 1])

        snapshots = {i: (None, 1e10) for i in range(self.network_size)}
        epochs_since_update = 0
        last_holdout_losses = None
        final_epoch = 0

        for epoch in itertools.count():
            # Each ensemble member receives a different permutation.  This
            # cheap bootstrapping encourages model diversity.
            train_idx = np.vstack([
                np.random.permutation(train_inputs.shape[0])
                for _ in range(self.network_size)
            ])
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos:start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(self.device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(self.device)
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.update(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs_t, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(
                    holdout_mean,
                    holdout_logvar,
                    holdout_labels_t,
                    inc_var_loss=False,
                )
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                last_holdout_losses = holdout_mse_losses
                # Elite networks are the lowest-loss members on held-out real
                # replay data.  Rollouts sample only from this subset.
                self.elite_model_idxes = np.argsort(holdout_mse_losses)[:self.elite_size].tolist()

            updated = False
            for i, current_loss in enumerate(holdout_mse_losses):
                _, best_loss = snapshots[i]
                improvement = (best_loss - current_loss) / best_loss
                # Match the common MBPO early-stop rule: a model has improved
                # only if holdout loss drops by more than one percent.
                if improvement > 0.01:
                    snapshots[i] = (epoch, current_loss)
                    updated = True
            epochs_since_update = 0 if updated else epochs_since_update + 1
            final_epoch = epoch
            if epochs_since_update > max_epochs_since_update:
                break

        self.last_train_stats = {
            "epochs": final_epoch + 1,
            "holdout_losses": last_holdout_losses,
            "elite_model_idxes": list(self.elite_model_idxes),
            "holdout_loss_mean": float(np.mean(last_holdout_losses)),
        }
        return self.last_train_stats

    def predict(self, inputs, batch_size=1024, factored=True):
        """
        Predict reward distribution for a batch of encoded transitions.

        Args:
            inputs: Raw, unnormalized model inputs.
            batch_size: Inference minibatch size.
            factored: If true, keep the ensemble dimension.  If false, return
                the mixture mean and total variance after marginalizing over
                ensemble members.
        """
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            batch = inputs[i:min(i + batch_size, inputs.shape[0])]
            batch_t = torch.from_numpy(batch).float().to(self.device)
            batch_t = batch_t[None, :, :].repeat([self.network_size, 1, 1])
            with torch.no_grad():
                batch_mean, batch_var = self.ensemble_model(batch_t, ret_log_var=False)
            ensemble_mean.append(batch_mean.detach().cpu().numpy())
            ensemble_var.append(batch_var.detach().cpu().numpy())

        ensemble_mean = np.concatenate(ensemble_mean, axis=1)
        ensemble_var = np.concatenate(ensemble_var, axis=1)
        if factored:
            return ensemble_mean, ensemble_var

        mean = np.mean(ensemble_mean, axis=0)
        # Law of total variance: expected model variance plus disagreement
        # among ensemble means.
        var = np.mean(ensemble_var, axis=0) + np.mean(np.square(ensemble_mean - mean[None, :, :]), axis=0)
        return mean, var

    def predict_reward(self, inputs, deterministic=False):
        """
        Return scalar reward predictions from elite ensemble members.

        Stochastic mode samples from each selected model's Gaussian output and
        randomly chooses an elite model per item.  Deterministic mode cycles
        through elite model means, which is useful for reproducible evaluation.
        """
        means, variances = self.predict(inputs, factored=True)
        batch_size = means.shape[1]
        if deterministic:
            model_idxes = np.asarray(self.elite_model_idxes)[
                np.arange(batch_size) % len(self.elite_model_idxes)
            ]
            return means[model_idxes, np.arange(batch_size), 0]

        stds = np.sqrt(np.maximum(variances, 1e-12))
        samples = means + np.random.normal(size=means.shape) * stds
        model_idxes = np.random.choice(self.elite_model_idxes, size=batch_size)
        return samples[model_idxes, np.arange(batch_size), 0]
