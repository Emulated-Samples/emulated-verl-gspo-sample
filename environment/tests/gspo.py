from __future__ import annotations

import sys
import torch
from typing import Optional

from gspo_reference import ground_truth_compute_policy_loss_gspo
from gspo_scoring import compute_scalar_score, compute_tensor_score
from gspo_helpers import make_mask
from gspo_scenarios import SCENARIOS, Scenario, _default_config

IMPORT_ERROR: Optional[Exception] = None
try:
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
except Exception as exc:
    IMPORT_ERROR = exc
    compute_policy_loss_gspo = None
    get_policy_loss_fn = None


class GSPOTestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def _record(self, score: float) -> None:
        if score >= 0.99:
            self.passed += 1
        else:
            self.failed += 1

    def check_registration(self) -> None:
        reg_exists_score = 0.0
        reg_correct_score = 0.0
        reg_exists_details = ""
        reg_correct_details = ""
        try:
            if get_policy_loss_fn is None:
                raise ImportError(str(IMPORT_ERROR))
            fn = get_policy_loss_fn("gspo")
            reg_exists_score = 1.0
            if fn is compute_policy_loss_gspo:
                reg_correct_score = 1.0
            else:
                reg_correct_details = f"resolved={getattr(fn, '__name__', type(fn).__name__)}"
        except (ValueError, Exception) as e:
            reg_exists_details = f"error={_compact(str(e))}"
            reg_correct_details = "policy loss gspo was not resolvable"

        print(f"Registration score: exists {reg_exists_score:.6f}")
        if reg_exists_score < 0.99:
            print(f"Registration details: exists {reg_exists_details}")
        print(f"Registration score: correct {reg_correct_score:.6f}")
        if reg_correct_score < 0.99:
            print(f"Registration details: correct {reg_correct_details}")
        self._record(reg_exists_score)
        self._record(reg_correct_score)

    def run_scenario(self, scenario: Scenario) -> None:
        name = scenario.name
        torch.manual_seed(scenario.seed)

        old_log_prob = scenario.old_log_prob_gen()
        log_prob = scenario.log_prob_gen(old_log_prob)
        advantages = scenario.adv_gen()
        response_mask = make_mask(scenario.mask_mode, scenario.batch_size, scenario.seq_len)
        if scenario.input_transform is not None:
            old_log_prob, log_prob, advantages, response_mask = scenario.input_transform(
                old_log_prob, log_prob, advantages, response_mask,
            )
        config = scenario.config or _default_config()

        gold_loss, gold_clipfrac, gold_kl, gold_clipfrac_lower = ground_truth_compute_policy_loss_gspo(
            old_log_prob=old_log_prob, log_prob=log_prob,
            advantages=advantages, response_mask=response_mask,
            loss_agg_mode="seq-mean-token-mean", config=config,
        )

        # --- Forward validation ---
        forward_score = 0.0
        forward_details = ""
        try:
            if compute_policy_loss_gspo is None:
                raise ImportError(str(IMPORT_ERROR))
            agent_loss, agent_clipfrac, agent_kl, agent_clipfrac_lower = compute_policy_loss_gspo(
                old_log_prob=old_log_prob, log_prob=log_prob,
                advantages=advantages, response_mask=response_mask,
                loss_agg_mode=scenario.loss_agg_mode, config=config,
            )
            forward_score = min(
                compute_scalar_score(agent_loss, gold_loss),
                compute_scalar_score(agent_clipfrac, gold_clipfrac),
                compute_scalar_score(agent_kl, gold_kl),
                compute_scalar_score(agent_clipfrac_lower, gold_clipfrac_lower),
            )
            forward_details = _forward_details(
                agent_loss, gold_loss,
                agent_clipfrac, gold_clipfrac,
                agent_kl, gold_kl,
                agent_clipfrac_lower, gold_clipfrac_lower,
            )
            if scenario.extra_checks is not None:
                scenario.extra_checks(
                    gold_loss, gold_clipfrac, gold_kl, gold_clipfrac_lower,
                    None, response_mask,
                )
        except Exception as e:
            forward_details = f"error={_compact(str(e))}"
            print(f"Forward error: {name}: {e}", file=sys.stderr)

        print(f"Forward score: {name} {forward_score:.6f}")
        if forward_score < 0.99:
            print(f"Forward details: {name} {forward_details}")
        self._record(forward_score)

        # --- Gradient validation ---
        gradient_score = 0.0
        if forward_score < 0.5:
            print(f"Gradient score: {name} 0.000000")
            print(f"Gradient details: {name} skipped=forward_score_below_0.5 forward_score={forward_score:.6f}")
            self.failed += 1
            return

        gradient_details = ""
        try:
            log_prob_gt = log_prob.clone().detach().requires_grad_(True)
            gt_loss, _, _, _ = ground_truth_compute_policy_loss_gspo(
                old_log_prob=old_log_prob.detach(), log_prob=log_prob_gt,
                advantages=advantages.detach(), response_mask=response_mask.detach(),
                loss_agg_mode="seq-mean-token-mean", config=config,
            )
            gt_loss.backward()
            grad_gt = log_prob_gt.grad.detach()

            log_prob_alt = log_prob.clone().detach().requires_grad_(True)
            alt_loss, _, _, _ = compute_policy_loss_gspo(
                old_log_prob=old_log_prob.detach(), log_prob=log_prob_alt,
                advantages=advantages.detach(), response_mask=response_mask.detach(),
                loss_agg_mode=scenario.loss_agg_mode, config=config,
            )
            alt_loss.backward()
            grad_alt = log_prob_alt.grad.detach()

            assert grad_gt.shape == grad_alt.shape, "gradient shape mismatch"
            assert torch.isfinite(grad_gt).all() and torch.isfinite(grad_alt).all(), "non-finite gradients"

            gradient_score = compute_tensor_score(grad_alt, grad_gt, response_mask)
            gradient_details = _gradient_details(grad_alt, grad_gt, response_mask, alt_loss, gt_loss)

            if scenario.extra_checks is not None:
                scenario.extra_checks(
                    gold_loss, gold_clipfrac, gold_kl, gold_clipfrac_lower,
                    grad_alt, response_mask,
                )
        except Exception as e:
            gradient_score = 0.0
            gradient_details = f"error={_compact(str(e))}"
            print(f"Gradient error: {name}: {e}", file=sys.stderr)

        print(f"Gradient score: {name} {gradient_score:.6f}")
        if gradient_score < 0.99:
            print(f"Gradient details: {name} {gradient_details}")
        self._record(gradient_score)

    def run_all(self) -> None:
        if IMPORT_ERROR is not None:
            print(f"ImportError: {IMPORT_ERROR}", file=sys.stderr)
        self.check_registration()
        for scenario in SCENARIOS:
            self.run_scenario(scenario)
        total = self.passed + self.failed
        print(f"\n{self.passed}/{total} checks passed, {self.failed}/{total} failed")
        print("GSPO_TESTS_COMPLETE")


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(()).item())


def _rel_error(agent_val: torch.Tensor, gold_val: torch.Tensor) -> float:
    agent = _scalar(agent_val)
    gold = _scalar(gold_val)
    return abs(agent - gold) / (abs(gold) + 1e-8)


def _scalar_pair(label: str, agent_val: torch.Tensor, gold_val: torch.Tensor) -> str:
    return (
        f"{label}=candidate:{_scalar(agent_val):.6g},"
        f"test_oracle:{_scalar(gold_val):.6g},rel_err:{_rel_error(agent_val, gold_val):.3g}"
    )


def _forward_details(
    agent_loss: torch.Tensor,
    gold_loss: torch.Tensor,
    agent_clipfrac: torch.Tensor,
    gold_clipfrac: torch.Tensor,
    agent_kl: torch.Tensor,
    gold_kl: torch.Tensor,
    agent_clipfrac_lower: torch.Tensor,
    gold_clipfrac_lower: torch.Tensor,
) -> str:
    return " ".join([
        _scalar_pair("loss", agent_loss, gold_loss),
        _scalar_pair("clipfrac", agent_clipfrac, gold_clipfrac),
        _scalar_pair("kl", agent_kl, gold_kl),
        _scalar_pair("clipfrac_lower", agent_clipfrac_lower, gold_clipfrac_lower),
    ])


def _gradient_details(
    agent_grad: torch.Tensor,
    gold_grad: torch.Tensor,
    mask: torch.Tensor,
    agent_loss: torch.Tensor,
    gold_loss: torch.Tensor,
) -> str:
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return "empty_mask=true"

    agent = agent_grad[mask_bool]
    gold = gold_grad[mask_bool]
    diff = agent - gold
    rel = diff.abs() / (gold.abs() + 1e-8)
    denom = agent.norm() * gold.norm() + 1e-12
    cosine = torch.dot(agent.flatten(), gold.flatten()) / denom

    return " ".join([
        _scalar_pair("loss", agent_loss, gold_loss),
        f"grad_mean_abs={diff.abs().mean().item():.6g}",
        f"grad_max_abs={diff.abs().max().item():.6g}",
        f"grad_mean_rel={rel.mean().item():.6g}",
        f"grad_cosine={cosine.item():.6g}",
    ])


def _compact(text: str) -> str:
    return " ".join(text.split())[:500]


if __name__ == "__main__":
    GSPOTestRunner().run_all()
