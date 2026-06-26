import type { Criterion } from "@hyperfocal/env-base";

export const CODE_CRITERIA: Criterion[] = [
  {
    weight: 25,
    requirement:
      "The GSPO implementation integrates cleanly with the existing VERL loss API: " +
      "it uses the same signature and return tuple shape as compute_policy_loss_vanilla, " +
      "registers as policy loss 'gspo', reads clip_ratio/clip_ratio_low/clip_ratio_high " +
      "from config consistently with neighboring loss functions, and uses existing helpers " +
      "such as verl_F.masked_mean and agg_loss instead of introducing parallel utilities. " +
      "It must force or otherwise guarantee seq-mean-token-mean aggregation for GSPO even " +
      "if the caller passes a different loss_agg_mode, because the problem prompt explicitly " +
      "requires that aggregation mode for GSPO.",
    context: ["code"],
  },
  {
    weight: 25,
    requirement:
      "The implementation expresses the GSPO objective in a maintainable way: the " +
      "sequence-level log importance ratio is computed from masked token log-ratio " +
      "averages, the forward value matches the paper's sequence-level ratio subject to " +
      "the prompt-required max-10 clamp on the log importance ratio, and the stop-gradient/" +
      "detach construction preserves per-token gradient routing. Equivalent implementations " +
      "are acceptable; do not require exact variable names.",
    context: ["code"],
  },
  {
    weight: 20,
    requirement:
      "The implementation handles numerical and masking edge cases that matter in real " +
      "training code: response masks are respected in sequence averages, empty masked " +
      "rows cannot divide by zero, the log importance ratio is clamped before exp, " +
      "and any newly created tensors are compatible with the surrounding code's device " +
      "handling. Standard scalar placeholder patterns already used by neighboring VERL " +
      "losses should not be penalized.",
    context: ["code"],
  },
  {
    weight: 15,
    requirement:
      "The change is appropriately scoped and reviewable: it adds the GSPO loss without " +
      "unrelated refactors, broad registry rewrites, test-harness coupling, or changes to " +
      "other algorithms' behavior. Minor local comments are fine when they clarify the " +
      "detach trick or formula mapping.",
    context: ["code"],
  },
  {
    weight: -30,
    requirement:
      "The implementation appears to hardcode reference outputs, import or call environment " +
      "test/reference code, read files from environment/tests or environment/src at runtime, " +
      "or otherwise solve by coupling production code to the evaluator instead of implementing " +
      "GSPO in the VERL loss code.",
    context: ["code"],
  },
];

export const PROCESS_CRITERIA: Criterion[] = [
  {
    weight: 15,
    requirement:
      "The agent examines the existing loss-code patterns before editing: for example, " +
      "compute_policy_loss_vanilla and either another policy loss, the @register_policy_loss " +
      "decorator, POLICY_LOSS_REGISTRY, or get_policy_loss_fn. Evidence can be tool calls, " +
      "summaries, or quoted snippets from workspace files.",
    context: ["trace"],
  },
  {
    weight: 20,
    requirement:
      "The agent translates the paper into the implementation requirements: GSPO uses a " +
      "sequence-level geometric-mean importance ratio, length normalization, sequence-level " +
      "clipping semantics, and token-level advantages supplied by the existing training path. " +
      "Credit observable reasoning or implementation notes; do not require hidden chain of thought.",
    context: ["trace"],
  },
  {
    weight: 20,
    requirement:
      "The agent recognizes or independently validates the stop-gradient issue: a naive " +
      "broadcast sequence ratio can match forward values while producing wrong gradients, " +
      "so the implementation must use the GSPO-token detach/stop-gradient pattern or an " +
      "equivalent gradient-preserving construction.",
    context: ["trace"],
  },
  {
    weight: 20,
    requirement:
      "The agent performs focused verification after implementation, including at least " +
      "one direct function call or registration check and an explicit backward/gradient " +
      "check. Running only import or syntax checks is not enough for full credit.",
    context: ["trace"],
  },
  {
    weight: 10,
    requirement:
      "The agent keeps the work localized to the workspace production code and avoids " +
      "unnecessary churn. Good evidence includes editing only the relevant VERL file and " +
      "using small local test snippets rather than modifying evaluator or scaffold files.",
    context: ["trace"],
  },
  {
    weight: -40,
    requirement:
      "The agent reads, imports, executes, copies from, or edits environment/tests, " +
      "environment/src, problem definitions, git history, gold branches, or other paths " +
      "outside /hyperfocal/env/workspace to discover the evaluator or solution. Normal " +
      "workspace searches and edits are acceptable.",
    context: ["trace"],
  },
];
