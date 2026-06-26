# GSPO in verl

> **Notice:** This repository is public for limited research review only. It is
> **not open source**. The data and tasks are proprietary IP of Emulated, Inc.
> and may not be copied, redistributed, used for model training, used to create
> derivative datasets/environments, or used commercially without written
> permission. See [`LICENSE`](./LICENSE). Contact: founders@emulated.so
>
> The `workspace/` tree is third-party verl source under the Apache License 2.0 (`workspace/LICENSE`) and is not covered by the proprietary terms above.

This sample asks an agent to implement Group Sequence Policy Optimization (GSPO) inside verl's training loss framework. The task gives the agent the relevant section of the GSPO paper, then requires a production-shaped implementation that fits verl's existing policy-loss API.

The hard part is not just matching the forward loss value. Strong solutions need the sequence-level importance ratio, length normalization, masking behavior, clipping semantics, and the stop-gradient construction that preserves the right token-level gradients. Models often produce code that looks correct in the forward pass but fails because the detach behavior is wrong.

## Quick start

- [Problem prompt](environment/problems.yaml): the exact agent-facing task and GSPO paper excerpt.
- [Rubric criteria](environment/src/criteria.ts): process and code criteria used to evaluate implementation quality.
- [Deterministic graders](environment/tests): focused tests for GSPO values, registration, scenarios, and gradient behavior.

## Grader walkthrough

This environment is a compact implementation-paper task. It checks whether an agent can read a new RL objective, map it into an existing training framework, and verify both the numerical result and the gradient path.

The graders look for:

- clean integration with verl's existing policy-loss registry and return-shape conventions
- sequence-level geometric-mean importance ratios with response masking
- prompt-required log-ratio clamping and `seq-mean-token-mean` aggregation
- a detach/stop-gradient construction that keeps the forward objective and backward gradients aligned
- focused verification, including registration checks and explicit backward/gradient checks
