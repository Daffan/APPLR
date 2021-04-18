import torch
from torch import nn
import numpy as np
from copy import deepcopy
from typing import Dict, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_layer = [64, 64], cnn = True, feature_layer = [256]):
        super().__init__()
        if cnn:
            self.feature = nn.Sequential(
                nn.Linear(720, feature_layer[0]), nn.ReLU(inplace=True)
                )
            feature_shape = feature_layer[0] + int(np.log2(action_shape-1)) + 1
        else:
            self.feature = lambda x: x.view(x.shape[0], -1)
            feature_shape = state_shape

        layers = [np.prod(feature_shape)] + hidden_layer
        self.value = []
        self.advantage = []
        for i, o in zip(layers[:-1], layers[1:]):
            self.value.append(nn.Linear(i, o))
            self.value.append(nn.ReLU(inplace=True))
            self.advantage.append(nn.Linear(i, o))
            self.advantage.append(nn.ReLU(inplace=True))
        self.advantage.append(nn.Linear(o, np.prod(action_shape)))
        self.value.append(nn.Linear(o, 1))

        self.value = nn.Sequential(*self.value)
        self.advantage = nn.Sequential(*self.advantage)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        laser = obs.view(batch, 1, -1)[:,:,:720]
        params = obs.view(batch, -1)[:, 720:]

        embedding = self.feature(laser).view(batch, -1)
        feature = torch.cat((embedding, params), dim = 1)

        advantage = self.advantage(feature)
        value = self.value(feature)
        logits = value + advantage - advantage.mean(1, keepdim=True)
        return logits, state

class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602

    Implementation of Double Q-Learning. arXiv:1509.06461

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here)

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (``0``
        if you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 discount_factor: float = 0.99,
                 estimation_step: int = 1,
                 target_update_freq: Optional[int] = 0,
                 reward_normalization: bool = False,
                 grad_norm_clipping=10,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0
        assert 0 <= discount_factor <= 1, 'discount_factor should in [0, 1]'
        self._gamma = discount_factor
        assert estimation_step > 0, 'estimation_step should greater than 0'
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self.grad_norm_clipping = grad_norm_clipping
        self._cnt = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode=True) -> torch.nn.Module:
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            a = self(batch, input='obs_next', eps=0).act
            with torch.no_grad():
                target_q = self(
                    batch, model='model_old', input='obs_next').logits
            target_q = target_q[np.arange(len(a)), a]
        else:
            with torch.no_grad():
                target_q = self(batch, input='obs_next').logits.max(dim=1)[0]
        return target_q

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        """Compute the n-step return for Q-learning targets. More details can
        be found at :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                model: str = 'model',
                input: str = 'obs',
                eps: Optional[float] = None,
                **kwargs) -> Batch:
        """Compute action over the given batch data. If you need to mask the
        action, please add a "mask" into batch.obs, for example, if we have an
        environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = getattr(batch, input)
        obs_ = obs.obs if hasattr(obs, 'obs') else obs
        q, h = model(obs_, state=state, info=batch.info)
        act = to_numpy(q.max(dim=1)[1])
        has_mask = hasattr(obs, 'mask')
        if has_mask:
            # some of actions are masked, they cannot be selected
            q_ = to_numpy(q)
            q_[~obs.mask] = -np.inf
            act = q_.argmax(axis=1)
        # add eps to act
        if eps is None:
            eps = self.eps
        if not np.isclose(eps, 0):
            for i in range(len(q)):
                if np.random.rand() < eps:
                    q_ = np.random.rand(*q[i].shape)
                    if has_mask:
                        q_[~obs.mask[i]] = -np.inf
                    act[i] = q_.argmax()
        return Batch(logits=q, act=act, state=h)

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        if self._target and self._cnt % self._freq == 0:
            self.sync_weight()
        weight = batch.pop("weight", 1.0)
        self.optim.zero_grad()
        q = self(batch, eps=0.).logits
        q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns, q).flatten()
        c = torch.nn.SmoothL1Loss(reduction = 'none')
        # c = lambda r, q: (r-q).pow(2)
        td = c(r, q)
        loss = (td * weight).mean()
        batch.weight = loss  # prio-buffer
        loss.backward()
        if self.grad_norm_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clipping)
        self.optim.step()
        self._cnt += 1
        return {'loss': loss.item()}
