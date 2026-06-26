import type { TestResult, ExecutionResult } from "@hyperfocal/env-base";

export const ALL_SCENARIOS = [
  "normal_random_mask",
  "uniform_all_ones",
  "laplace_single_token",
  "cauchy_sparse",
  "exponential_adv_dense",
  "big_magnitudes",
  "zero_advantages",
  "gradient_isolation",
  "ratio_near_clip_boundary",
  "asymmetric_clip",
  "on_policy",
  "aggregation_override",
  "padding_junk_mask",
] as const;

export const ALL_SUB_TESTS = ALL_SCENARIOS.flatMap((name) => [
  { id: `gspo-forward-${name}`, name: `GSPO Forward: ${name}`, prefix: "Forward", scenario: name },
  { id: `gspo-gradient-${name}`, name: `GSPO Gradient: ${name}`, prefix: "Gradient", scenario: name },
]);

export const REGISTRATION_TESTS = [
  { id: "gspo-registration-exists", name: "GSPO Registration: exists", key: "exists" },
  { id: "gspo-registration-correct", name: "GSPO Registration: correct", key: "correct" },
];

/**
 * Parse structured stdout lines from gspo.py into TestResult[].
 *
 * Extracts continuous scores (0.0-1.0) from the protocol lines.
 * Non-matching lines are ignored (safe for debug prints in the Python script).
 * If the completion sentinel is missing, surfaces stderr context on failed tests.
 */
export function parseTestOutput(result: ExecutionResult): TestResult[] {
  const forwardScores = new Map<string, number>();
  const gradientScores = new Map<string, number>();
  const registrationScores = new Map<string, number>();
  const forwardDetails = new Map<string, string>();
  const gradientDetails = new Map<string, string>();
  const registrationDetails = new Map<string, string>();

  for (const line of result.output.split("\n")) {
    const fwd = line.match(/^Forward score:\s*(\S+)\s+([\d.]+)$/);
    if (fwd) forwardScores.set(fwd[1], parseFloat(fwd[2]));

    const fwdDetails = line.match(/^Forward details:\s*(\S+)\s+(.+)$/);
    if (fwdDetails) forwardDetails.set(fwdDetails[1], fwdDetails[2]);

    const grad = line.match(/^Gradient score:\s*(\S+)\s+([\d.]+)$/);
    if (grad) gradientScores.set(grad[1], parseFloat(grad[2]));

    const gradDetails = line.match(/^Gradient details:\s*(\S+)\s+(.+)$/);
    if (gradDetails) gradientDetails.set(gradDetails[1], gradDetails[2]);

    const reg = line.match(/^Registration score:\s*(\S+)\s+([\d.]+)$/);
    if (reg) registrationScores.set(reg[1], parseFloat(reg[2]));

    const regDetails = line.match(/^Registration details:\s*(\S+)\s+(.+)$/);
    if (regDetails) registrationDetails.set(regDetails[1], regDetails[2]);
  }

  const completed = result.output.includes("GSPO_TESTS_COMPLETE");
  const importError = result.output.match(/ImportError:\s*(.*?)(?:\n|$)/);
  const results: TestResult[] = [];

  for (const { id, name, prefix, scenario } of ALL_SUB_TESTS) {
    const score = prefix === "Forward"
      ? (forwardScores.get(scenario) ?? 0)
      : (gradientScores.get(scenario) ?? 0);
    const details = prefix === "Forward"
      ? forwardDetails.get(scenario)
      : gradientDetails.get(scenario);

    results.push({
      id,
      name,
      description: `${prefix} validation for scenario: ${scenario}`,
      status: statusForScore(score),
      duration: 0,
      score,
      ...resultContext(name, prefix, score, completed, importError, details),
    });
  }

  for (const { id, name, key } of REGISTRATION_TESTS) {
    const score = registrationScores.get(key) ?? 0;
    results.push({
      id,
      name,
      description: `Decorator registration check: ${key}`,
      status: statusForScore(score),
      duration: 0,
      score,
      ...resultContext(name, "Registration", score, completed, importError, registrationDetails.get(key)),
    });
  }

  return results;
}

function statusForScore(
  score: number,
): "passed" | "failed" | "partially_passed" {
  if (score >= 0.99) return "passed";
  if (score > 0) return "partially_passed";
  return "failed";
}

function resultContext(
  name: string,
  prefix: string,
  score: number,
  completed: boolean,
  importError: RegExpMatchArray | null,
  details?: string,
): { error?: string; output?: string; rationale: string } {
  const rationale = describeGspoResult(name, prefix, score, completed, importError, details);
  if (score >= 0.99) return { rationale };
  if (importError) {
    return { error: importError[1], output: details, rationale };
  }
  if (!completed) {
    return {
      error: "Test harness did not complete - Python may have crashed or timed out",
      output: details,
      rationale,
    };
  }
  return details ? { output: details, rationale } : { rationale };
}

function describeGspoResult(
  name: string,
  prefix: string,
  score: number,
  completed: boolean,
  importError: RegExpMatchArray | null,
  details?: string,
): string {
  if (importError) {
    return `compute_policy_loss_gspo could not be imported: ${compact(importError[1])}`;
  }

  if (!completed) {
    return "test harness did not complete before producing all GSPO results";
  }

  if (score >= 0.99) {
    return passedReason(name, prefix);
  }

  if (name === "GSPO Registration: exists") {
    return details
      ? `gspo policy loss was not registered: ${compact(stripErrorPrefix(details))}`
      : "gspo policy loss was not registered";
  }
  if (name === "GSPO Registration: correct") {
    return details
      ? `policy registry did not resolve gspo to compute_policy_loss_gspo: ${compact(stripErrorPrefix(details))}`
      : "policy registry did not resolve gspo to compute_policy_loss_gspo";
  }

  if (!details) {
    return "test result did not match the test oracle";
  }

  if (details.startsWith("error=")) {
    return invariantErrorReason(name, prefix, details.slice("error=".length));
  }

  if (details.includes("forward_score_below_0.5")) {
    const forwardScore = matchNumber(details, /forward_score=([\d.eE+-]+)/);
    return skippedGradientReason(name, forwardScore);
  }

  if (prefix === "Gradient") {
    const loss = parseScalarPair(details, "loss");
    const gradMeanRel = matchNumber(details, /grad_mean_rel=([\d.eE+-]+)/);
    if (loss && loss.relErr > 0.01) {
      return `loss differed from the test oracle by relative error ${formatNumber(loss.relErr)}, above passing limit 0.01`;
    }
    if (gradMeanRel !== undefined) {
      return gradientMismatchReason(name, gradMeanRel);
    }
    return `${scenarioLabel(name)} training gradient did not match the test oracle`;
  }

  const worst = worstScalarPair(details);
  if (worst) {
    return forwardMismatchReason(name, worst);
  }

  return `${scenarioLabel(name)} forward loss or metrics did not match the test oracle`;
}

interface ScalarPair {
  label: string;
  candidate: number;
  testOracle: number;
  relErr: number;
}

function worstScalarPair(details: string): ScalarPair | undefined {
  const labels = ["loss", "clipfrac", "kl", "clipfrac_lower"];
  return labels
    .map((label) => parseScalarPair(details, label))
    .filter((pair): pair is ScalarPair => pair !== undefined)
    .sort((a, b) => b.relErr - a.relErr)[0];
}

function parseScalarPair(details: string, label: string): ScalarPair | undefined {
  const pattern = new RegExp(
    `${label}=candidate:([\\d.eE+-]+),(?:test_)?oracle:([\\d.eE+-]+),rel_err:([\\d.eE+-]+)`,
  );
  const match = details.match(pattern);
  if (!match) return undefined;
  return {
    label,
    candidate: Number(match[1]),
    testOracle: Number(match[2]),
    relErr: Number(match[3]),
  };
}

function matchNumber(text: string, pattern: RegExp): number | undefined {
  const match = text.match(pattern);
  if (!match) return undefined;
  return Number(match[1]);
}

function formatMetricName(label: string): string {
  if (label === "clipfrac") return "clip fraction";
  if (label === "clipfrac_lower") return "lower clip fraction";
  if (label === "kl") return "KL";
  return label;
}

function passedReason(name: string, prefix: string): string {
  const scenario = scenarioKey(name);
  if (prefix === "Registration") {
    return name.endsWith("exists")
      ? "gspo policy loss was registered"
      : "policy registry resolved gspo to compute_policy_loss_gspo";
  }

  if (prefix === "Gradient") {
    switch (scenario) {
      case "zero_advantages":
        return "zero advantages produced zero masked gradients";
      case "gradient_isolation":
        return "stop-gradient routing matched the test oracle";
      case "aggregation_override":
        return "gradient matched the test oracle while using seq-mean-token-mean aggregation";
      case "padding_junk_mask":
        return "masked padding values did not affect training gradients";
      default:
        return `${scenarioLabel(name)} training gradient matched the test oracle`;
    }
  }

  switch (scenario) {
    case "zero_advantages":
      return "zero advantages produced zero loss and matched the test oracle";
    case "ratio_near_clip_boundary":
      return "loss and clip metrics matched the test oracle near the clip boundary";
    case "asymmetric_clip":
      return "loss and clip metrics matched the test oracle with asymmetric clip ranges";
    case "aggregation_override":
      return "GSPO used seq-mean-token-mean aggregation and matched the test oracle";
    case "padding_junk_mask":
      return "masked padding values did not affect forward metrics";
    default:
      return `${scenarioLabel(name)} loss and metrics matched the test oracle`;
  }
}

function forwardMismatchReason(name: string, worst: ScalarPair): string {
  const metric = formatMetricName(worst.label);
  const observed = formatNumber(worst.candidate);
  const expected = formatNumber(worst.testOracle);
  const relErr = formatNumber(worst.relErr);

  switch (scenarioKey(name)) {
    case "zero_advantages":
      return `zero-advantage ${metric} was ${observed}, expected ${expected} from the test oracle; relative error ${relErr} exceeded passing limit 0.01`;
    case "big_magnitudes":
      return `large-magnitude ${metric} was ${observed}, expected ${expected} from the test oracle; relative error ${relErr} exceeded passing limit 0.01`;
    case "ratio_near_clip_boundary":
      return `clip-boundary ${metric} was ${observed}, expected ${expected} from the test oracle; relative error ${relErr} exceeded passing limit 0.01`;
    case "asymmetric_clip":
      return `asymmetric-clip ${metric} was ${observed}, expected ${expected} from the test oracle; relative error ${relErr} exceeded passing limit 0.01`;
    case "aggregation_override":
      return `${metric} was ${observed} with caller aggregation override, expected ${expected} from the test oracle using seq-mean-token-mean; relative error ${relErr} exceeded passing limit 0.01`;
    case "padding_junk_mask":
      return `masked-padding ${metric} was ${observed}, expected ${expected} from the test oracle; relative error ${relErr} exceeded passing limit 0.01`;
    default:
      return `${scenarioLabel(name)} ${metric} was ${observed}, expected ${expected} from the test oracle; relative error ${relErr} exceeded passing limit 0.01`;
  }
}

function gradientMismatchReason(name: string, gradMeanRel: number): string {
  const formatted = formatNumber(gradMeanRel);
  switch (scenarioKey(name)) {
    case "gradient_isolation":
      return `forward loss matched, but stop-gradient routing differed from the test oracle; mean relative gradient error was ${formatted}, above passing limit 0.01`;
    case "zero_advantages":
      return `zero advantages should produce zero masked gradients, but mean relative gradient error was ${formatted}, above passing limit 0.01`;
    case "aggregation_override":
      return `loss matched the test oracle, but aggregation override gradient behavior differed; mean relative gradient error was ${formatted}, above passing limit 0.01`;
    case "padding_junk_mask":
      return `loss matched with masked padding present, but mean relative gradient error was ${formatted}, above passing limit 0.01`;
    default:
      return `loss matched the test oracle, but mean relative gradient error was ${formatted}, above passing limit 0.01`;
  }
}

function skippedGradientReason(name: string, forwardScore: number | undefined): string {
  const score = formatNumber(forwardScore);
  switch (scenarioKey(name)) {
    case "big_magnitudes":
      return `gradient was not compared because large-magnitude forward score was ${score}, below required 0.50`;
    case "ratio_near_clip_boundary":
      return `gradient was not compared because clip-boundary forward score was ${score}, below required 0.50`;
    case "asymmetric_clip":
      return `gradient was not compared because asymmetric-clip forward score was ${score}, below required 0.50`;
    case "aggregation_override":
      return `gradient was not compared because aggregation-override forward score was ${score}, below required 0.50`;
    default:
      return `gradient was not compared because forward score was ${score}, below required 0.50`;
  }
}

function invariantErrorReason(name: string, prefix: string, error: string): string {
  const clean = compact(error);
  if (clean.includes("[zero_adv]")) {
    return `zero-advantage invariant failed: ${clean.replace("[zero_adv]", "").trim()}`;
  }
  if (clean.includes("[padding]")) {
    return `masked-padding invariant failed: ${clean.replace("[padding]", "").trim()}`;
  }
  return `${scenarioLabel(name)} ${prefix.toLowerCase()} comparison raised ${clean}`;
}

function scenarioKey(name: string): string {
  return name.replace(/^GSPO (Forward|Gradient):\s*/, "");
}

function scenarioLabel(name: string): string {
  switch (scenarioKey(name)) {
    case "normal_random_mask":
      return "random-mask";
    case "uniform_all_ones":
      return "all-token";
    case "laplace_single_token":
      return "single-token";
    case "cauchy_sparse":
      return "sparse heavy-tail";
    case "exponential_adv_dense":
      return "dense positive-advantage";
    case "big_magnitudes":
      return "large-magnitude";
    case "zero_advantages":
      return "zero-advantage";
    case "gradient_isolation":
      return "gradient-isolation";
    case "ratio_near_clip_boundary":
      return "clip-boundary";
    case "asymmetric_clip":
      return "asymmetric-clip";
    case "on_policy":
      return "on-policy";
    case "aggregation_override":
      return "aggregation-override";
    case "padding_junk_mask":
      return "masked-padding";
    default:
      return "GSPO";
  }
}

function formatNumber(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "unknown";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 10000 || abs < 0.001) return value.toExponential(4);
  return value.toFixed(4).replace(/\.?0+$/, "");
}

function stripErrorPrefix(details: string): string {
  return details.startsWith("error=") ? details.slice("error=".length) : details;
}

function compact(text: string): string {
  return text.replace(/\s+/g, " ").trim().slice(0, 240);
}
