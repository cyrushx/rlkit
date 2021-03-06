from collections import OrderedDict, namedtuple
from typing import Tuple

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from external.rlkit.rlkit.core.loss import LossFunction, LossStatistics
import external.rlkit.rlkit.torch.pytorch_util as ptu
from external.rlkit.rlkit.core.eval_util import create_stats_ordered_dict
from external.rlkit.rlkit.torch.torch_rl_algorithm import TorchTrainer
from external.rlkit.rlkit.core.logging import add_prefix
import gtimer as gt
import pdb

SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss rf1_loss rf2_loss',
)

class RiskConditionedSACTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            rf1,
            rf2,
            target_rf1,
            target_rf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            rf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.rf1 = rf1
        self.rf2 = rf2
        self.target_rf1 = target_rf1
        self.target_rf2 = target_rf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.rf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.rf1_optimizer = optimizer_class(
            self.rf1.parameters(),
            lr=rf_lr,
        )
        self.rf2_optimizer = optimizer_class(
            self.rf2.parameters(),
            lr=rf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()
        
        self.rf1_optimizer.zero_grad()
        losses.rf1_loss.backward()
        self.rf1_optimizer.step()

        self.rf2_optimizer.zero_grad()
        losses.rf2_loss.backward()
        self.rf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        collisions = batch['collision']
        risk = batch['risk']
        risk_bound = batch['risk_bound']
        allocated_risk = batch['allocated_risk']
        """
        Policy and Alpha Loss
        """
        obs_concat = torch.cat([obs, allocated_risk, risk_bound], dim=1)
        next_obs_concat = torch.cat([next_obs, allocated_risk, risk_bound], dim=1)
        dist = self.policy(obs_concat)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        ## TODO: finetune the loss function BELOW
        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        
        r_new_actions = torch.min(
            self.rf1(obs, new_obs_actions),
            self.rf2(obs, new_obs_actions),
        )
        ## NOTE: if budget is already <0, it means that at the timestep
        ## the risk bound  constraint is already violated
        # r_new_actions = self.rf1(obs, new_obs_actions)
        ## TODO: softmax for risk critic
        r_loss_coeff = 10.0
        overused_risk = allocated_risk + r_new_actions - risk_bound 
        # overused_risk =  r_new_actions - (risk_budget - risk_bound) # >0 if violate risk, =0 otherwise
        r_policy_loss = torch.nn.functional.relu(overused_risk) * r_loss_coeff # >0 if violate risk, =0 otherwise
        # TODO(cyrushx): Add risk in policy loss.
        # policy_loss = (alpha*log_pi - q_new_actions + 1.*r_new_actions).mean()
        policy_loss = (alpha*log_pi - q_new_actions + r_policy_loss).mean()
        
        # r_bound = 
        # r_left = r_bound - r_new_actions
        # m = nn.Hardtanh(0, 0.01)
        # r_step = m(r_left) * 100
        # policy_loss = (alpha * log_pi - q_new_actions * r_step).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs_concat)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        
        """
        Risk Critic Loss
        """
        # r1_pred = self.rf1(obs, actions)
        # r2_pred = self.rf2(obs, actions)
        # # TODO(cyrushx): Replace target_r_values with ground truth risk values.
        # # target_r_values = self.target_rf1(next_obs, new_next_actions)
        # target_r_values = torch.min(
        #     self.target_rf1(next_obs, new_next_actions),
        #     self.target_rf2(next_obs, new_next_actions),
        # ) - alpha * new_log_pi

        # # r_target = risk
        # # r_target = collisions + (1. - terminals) * (1 - collisions) * target_r_values
        # r_target = self.reward_scale * risk + (1. - terminals) * self.discount * target_r_values
        # rf1_loss = self.rf_criterion(r1_pred, r_target.detach())
        # rf2_loss = self.rf_criterion(r2_pred, r_target.detach())

        r1_pred = self.rf1(obs, actions)
        r2_pred = self.rf2(obs, actions)
        next_dist = self.policy(next_obs_concat)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_r_values = torch.min(
            self.target_rf1(next_obs, new_next_actions),
            self.target_rf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        r_target = self.reward_scale * risk + (1. - terminals) * self.discount * target_r_values
        rf1_loss = self.rf_criterion(r1_pred, r_target.detach())
        rf2_loss = self.rf_criterion(r2_pred, r_target.detach())


        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['RF1 Loss'] = np.mean(ptu.get_numpy(rf1_loss))
            eval_statistics['RF2 Loss'] = np.mean(ptu.get_numpy(rf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'R1 Predictions',
                ptu.get_numpy(r1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'R2 Predictions',
                ptu.get_numpy(r2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'R Targets',
                ptu.get_numpy(r_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Collisions',
                ptu.get_numpy(collisions),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
            rf1_loss=rf1_loss,
            rf2_loss=rf2_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.rf1,
            self.rf2,
            self.target_rf1,
            self.target_rf2,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.rf1_optimizer,
            self.rf2_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            rf1=self.rf1,
            rf2=self.rf2,
            target_rf1=self.target_rf1,
            target_rf2=self.target_rf2,
        )
