import {
  EnvironmentDefinition,
  Logger,
  TestResult,
  BatchTest,
  runSimpleTests,
  runSubprocessTests,
  executeWithExitCode,
  loadProblemsFromDirectory,
  createRubricTest,
  getLogsDir,
} from "@hyperfocal/env-base";
import type { SimpleTest } from "@hyperfocal/env-base";
import * as path from "path";
import * as fs from "fs";
import { fileURLToPath } from "url";

import { parseTestOutput } from "./gspo-test-runner.js";
import { CODE_CRITERIA, PROCESS_CRITERIA } from "./criteria.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const problems = loadProblemsFromDirectory(path.join(__dirname, ".."));

const WORKSPACE_PATH =
  process.env.WORKSPACE_PATH ||
  path.join(__dirname, "..", "..", "workspace");
const ENVIRONMENT_DIR = path.join(__dirname, "..");
const RUBRIC_MODEL = "openai/gpt-5.5";

class Environment implements EnvironmentDefinition {
  async listProblems() {
    return problems;
  }

  async setupProblem(problemId?: string, logger?: Logger): Promise<void> {
    const log = (msg: string) => (logger ? logger.info(msg) : console.log(msg));
    const id = problemId ?? problems.find((p) => p.default)?.id ?? problems[0]?.id;

    log(`Setting up ${id}`);
    await resetWorkspace(log);

    log("Checking Python dependencies...");
    const checkResult = await executeWithExitCode(
      'python3 -c "import torch; import omegaconf; import packaging; import tensordict"',
    );

    if (!checkResult.success) {
      log("Installing Python dependencies...");
      await ensurePipAvailable(log);
      const installResult = await executeWithExitCode(
        `python3 -m pip install -r ${shellQuote(path.join(ENVIRONMENT_DIR, "requirements-test.txt"))}`,
        { timeout: 600000 },
      );
      if (!installResult.success) {
        throw new Error(
          `Failed to install Python dependencies: ${installResult.output}`,
        );
      }
      log("Python dependencies installed");
    } else {
      log("Python dependencies available");
    }

    log("Setup completed");
  }

  async runTests(problemId: string, logger: Logger): Promise<TestResult[]> {
    const testScript = path.join(ENVIRONMENT_DIR, "tests", "gspo.py");

    const batchTest: BatchTest = {
      id: "gspo-ground-truth",
      name: "GSPO Ground Truth Validation",
      description:
        "13 scenarios testing forward values and backward gradients, plus decorator registration (28 sub-tests)",
      runBatch: async (logger: Logger): Promise<TestResult[]> => {
        const results = await runSubprocessTests(
          `PYTHONPATH=${WORKSPACE_PATH} python3 ${testScript}`,
          {
            timeout: 120000,
            cwd: WORKSPACE_PATH,
            logger,
            parseResults: parseTestOutput,
          },
        );
        return results;
      },
    };

    const tests: (BatchTest | SimpleTest)[] = [batchTest];

    tests.push(
      createRubricTest({
        id: "rubric-code-quality",
        name: "GSPO Code Quality Rubric",
        description:
          "LLM judge evaluates code style, conventions, and defensive coding",
        criteria: CODE_CRITERIA,
        modelName: RUBRIC_MODEL,
        getContext: async () => {
          const functionCode = await extractPythonFunction(
            path.join(WORKSPACE_PATH, "verl/trainer/ppo/core_algos.py"),
            "compute_policy_loss_gspo",
          );
          const diff = await getWorkspaceDiff();
          return {
            code:
              `# Extracted compute_policy_loss_gspo\n${functionCode}\n\n` +
              `# Workspace diff against rollout baseline\n${diff}`,
          };
        },
        passThreshold: 0.5,
      }),
    );

    const trace = findLatestAgentTrace(problemId);
    if (trace) {
      tests.push(
        createRubricTest({
          id: "rubric-agent-process",
          name: "Agent Process Quality Rubric",
          description:
            "LLM judge evaluates agent reasoning, paper understanding, and testing approach",
          criteria: PROCESS_CRITERIA,
          modelName: RUBRIC_MODEL,
          getContext: async () => ({ trace }),
          passThreshold: 0.5,
          tracePreprocessOptions: {},
        }),
      );
    } else {
      logger.info("No agent trace found -- skipping process rubric test");
    }

    return runSimpleTests(tests, logger);
  }
}

export default new Environment();

async function resetWorkspace(log: (msg: string) => void): Promise<void> {
  log("Resetting workspace to branch state...");

  await runSetupCommand("git rev-parse --is-inside-work-tree", {
    cwd: WORKSPACE_PATH,
    silent: true,
  });
  await runSetupCommand("git restore --source=HEAD --staged --worktree -- .", {
    cwd: WORKSPACE_PATH,
    silent: true,
  });
  await runSetupCommand("git clean -fd -- .", {
    cwd: WORKSPACE_PATH,
    silent: true,
  });
  await runSetupCommand(
    "rm -f .agent-prompt.txt .hyperfocal/manifest.json && " +
      "find . -type d \\( -name __pycache__ -o -name .pytest_cache \\) -prune -exec rm -rf {} + && " +
      "find . -type f \\( -name '*.pyc' -o -name '*.pyo' \\) -delete",
    {
      cwd: WORKSPACE_PATH,
      silent: true,
    },
  );

  log("Workspace reset completed");
}

async function extractPythonFunction(
  filePath: string,
  functionName: string,
): Promise<string> {
  const result = await executeWithExitCode(
    `python3 -c "
import ast, sys
with open('${filePath}') as f:
    source = f.read()
    lines = source.splitlines()
tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == '${functionName}':
        start = node.lineno - 1
        end = node.end_lineno
        while start > 0 and lines[start-1].strip().startswith('@'):
            start -= 1
        print('\\\\n'.join(lines[start:end]))
        sys.exit(0)
print('FUNCTION_NOT_FOUND')
"`,
    { silent: true },
  );

  if (!result.success || result.output.trim() === "FUNCTION_NOT_FOUND") {
    return `# ${functionName} was not found in ${path.basename(filePath)}`;
  }
  return result.output;
}

async function getWorkspaceDiff(): Promise<string> {
  const baseline = await executeWithExitCode(
    "git rev-parse --verify hyperfocal-baseline",
    { cwd: WORKSPACE_PATH, silent: true },
  );
  const diffTarget = baseline.success ? "hyperfocal-baseline" : "";
  const result = await executeWithExitCode(
    `git -c core.fileMode=false diff ${diffTarget} -- verl/trainer/ppo/core_algos.py`,
    { cwd: WORKSPACE_PATH, silent: true },
  );
  if (!result.success) {
    return "# Could not collect workspace diff";
  }
  return result.output.trim() || "# No tracked diff in core_algos.py";
}

function findLatestAgentTrace(problemId: string): string | null {
  const agentDir = path.join(getLogsDir(), problemId, "agent");
  try {
    const traceFiles = fs
      .readdirSync(agentDir)
      .filter((f) => f.endsWith(".jsonl"))
      .sort()
      .reverse();
    if (traceFiles.length === 0) return null;
    return fs.readFileSync(path.join(agentDir, traceFiles[0]), "utf-8");
  } catch {
    return null;
  }
}

async function runSetupCommand(
  command: string,
  options?: { cwd?: string; silent?: boolean; timeout?: number },
): Promise<void> {
  const result = await executeWithExitCode(command, options);
  if (!result.success) {
    throw new Error(`Setup command failed: ${command}\n${result.output}`);
  }
}

async function ensurePipAvailable(log: (msg: string) => void): Promise<void> {
  const checkPip = await executeWithExitCode("python3 -m pip --version", {
    silent: true,
  });
  if (checkPip.success) {
    return;
  }

  log("Bootstrapping pip with ensurepip...");
  const ensurePip = await executeWithExitCode(
    "python3 -m ensurepip --upgrade",
    { timeout: 120000 },
  );
  if (!ensurePip.success) {
    throw new Error(`Failed to bootstrap pip: ${ensurePip.output}`);
  }

  const verifyPip = await executeWithExitCode("python3 -m pip --version", {
    silent: true,
  });
  if (!verifyPip.success) {
    throw new Error(`pip still unavailable after ensurepip: ${verifyPip.output}`);
  }
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}
