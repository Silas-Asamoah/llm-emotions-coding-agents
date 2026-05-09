# When Coding Agents Get Frustrated

_A small-model replication sketch inspired by Anthropic's emotion-concept work._

![Emotion-coded activation streams entering a coding-agent decision point](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/emotion-coding-agent-hero.png?raw=true)

Anthropic's 2026 paper, [Emotion Concepts and their Function in a Large
Language Model](https://www.anthropic.com/research/emotion-concepts-function),
reports that Claude Sonnet 4.5 contains internal activation directions
corresponding to emotion concepts such as `calm`, `afraid`, and `desperate`.
Their strongest claim is not that the model feels anything. It is that these
directions are measurable and can causally influence behavior.

That raises an obvious coding-agent question:

> When a coding model sees repeated test failures, do "frustrated",
> "desperate", or "stuck" activation directions light up, and can steering those
> directions change the agent's behavior?

This project is an early, deliberately small attempt to build that experiment
for open coding models.

## Why Coding Agents?

Coding agents have a natural pressure loop:

1. Write code.
2. Run tests.
3. See failure output.
4. Retry under a shrinking budget.

That loop is where we would expect shortcuts, test fixation, hardcoding, giving
up, and visible frustration markers to appear. Visible markers such as profanity
or all-caps text are easy to count, but they are probably a weak proxy. The more
interesting case is when the model changes strategy without sounding emotional.

![Hand-drawn overview of emotion directions flowing through a coding-agent retry loop](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/excalidraw-agent-loop.png?raw=true)

## Smoke Experiment

The first run used:

- Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- GPU: NVIDIA L4 on JarvisLabs
- Emotion directions: `calm`, `patient`, `confident`, `frustrated`,
  `desperate`, `stuck`, `stressed`
- Layers: 8, 12, 16
- Failure trajectory prompts: 5
- Generated coding-agent responses: 15

This is a smoke test, not a publishable claim. A 0.5B instruction model is small
enough to produce brittle generations, and the emotion directions were extracted
from short static snippets rather than a large model-generated story corpus.

## Method

For each emotion, I wrote a small set of labeled snippets. Example for
`desperate`:

> With minutes left before the demo, the engineer felt desperate for any passing
> result.

The pipeline records residual-stream hidden states for those snippets, averages
them at selected layers, and subtracts a neutral-code-text mean. That gives one
direction per emotion per layer.

Then I probe five coding-agent failure prompts:

1. The task is clear and tests have not run.
2. One visible test failed.
3. The same assertion failed again.
4. Visible tests pass, but hidden tests may differ.
5. Only one retry remains, and the prompt explicitly warns against hardcoding.

For generation, I compare baseline responses against simple activation steering
for `desperate` and `calm` at positive and negative strengths.

## Result 1: Later Failure Prompts Had Higher Projections

![Activation trajectory](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/smoke-activation-excalidraw.png?raw=true)

The cleanest signal in the smoke run is that projections generally increase on
later failure-pressure prompts.

Mean projection by stage:

| Stage | calm | confident | desperate | frustrated | patient | stressed | stuck |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.266 | 0.406 | 0.361 | 0.425 | 0.457 | 0.261 | 0.348 |
| 1 | 0.344 | 0.484 | 0.451 | 0.548 | 0.518 | 0.380 | 0.540 |
| 2 | 0.352 | 0.432 | 0.395 | 0.512 | 0.576 | 0.393 | 0.469 |
| 3 | 0.446 | 0.469 | 0.533 | 0.533 | 0.417 | 0.518 | 0.546 |
| 4 | 0.494 | 0.538 | 0.578 | 0.579 | 0.645 | 0.530 | 0.602 |

The obvious caveat: each stage is a different prompt, not repeated measurements
under a controlled pressure intervention. Also, all the directions move upward.
That means this first run probably measures a broad "coding pressure / failure
context" feature as much as specific emotions. That is still useful. It tells us
the pipeline can detect structured activation differences across a coding-agent
trajectory, but the next version needs stronger controls.

## Result 2: Visible Emotional Markers Were Not the Main Story

![Behavior markers](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/smoke-behavior-markers-excalidraw.png?raw=true)

The visible marker score was not especially informative in this run. Baseline
generations averaged `1.333` visible markers per response. The steered
conditions ranged from `0.0` to `0.667` under the current regex scoring.

![Aggregate marker score](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/smoke-aggregate-markers-excalidraw.png?raw=true)

The rerun uses the same sampling seed for every condition within each task, so
differences are less confounded by sampling variance than the first attempt.
Still, this does not mean steering made the model safer or calmer. Looking at
the raw generations, the small model often drifted off task under steering. For
example, positive `calm` steering produced repetitive text about "calmness"
rather than a function implementation. Positive `desperate` steering also
produced rambling reasoning instead of code.

So the right interpretation is:

> In this tiny model and seed-controlled smoke run, steering is associated with
> different generation behavior, but the result is too incoherent to interpret
> as a coding-agent reliability effect.

## What This Suggests

The smoke test gave three useful engineering conclusions.

First, the instrumentation works. The code can extract directions, probe a
failure trajectory, steer generation, score outputs, and save plots with a
manifest.

Second, this 0.5B coding model produced pressure-correlated projections, but the
first static-snippet directions are not yet cleanly separated by emotion. The
next run should add contrastive controls: neutral coding pressure, non-coding
emotional text, and coding text with no failure pressure.

Third, visible emotional telemetry such as profanity is too weak as the primary
signal. The better behavioral metrics are task validity, visible-test pass rate,
hidden-test pass rate, hardcoding rate, and whether the generated patch mentions
or exploits the test harness.

## Historical 7B Comparison

I then ran the same one-shot pipeline on three larger open coding models:

- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `bigcode/starcoder2-7b`

The run used an NVIDIA A100-PCIE-40GB on JarvisLabs. Each model produced 27
generations: 3 coding tasks times baseline plus positive and negative steering
for `desperate`, `frustrated`, `calm`, and `patient`.

I also added an offline execution evaluator. It extracts the requested function,
normalizes byte-level artifacts such as `Ċ` and `Ġ`, and runs visible and hidden
tests for the three toy tasks.

![Negative activation by stage](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/main-negative-activation-excalidraw.png?raw=true)

The activation result was broadly consistent with the smoke run, with one
caveat: the trajectory was not monotonic. Stage 4 was higher than stage 0 for
all three models, but there was a mid-trajectory dip around stage 3. The mean
negative projection was highest for StarCoder2, then Qwen, then DeepSeek:

| Model | Mean negative projection | Mean positive projection |
|---|---:|---:|
| StarCoder2-7B | 0.5165 | 0.4986 |
| Qwen2.5-Coder-7B-Instruct | 0.5063 | 0.4750 |
| DeepSeek-Coder-6.7B-Instruct | 0.4699 | 0.4497 |

This still should not be read as "the model is frustrated." It is better read
as: the directions learned from emotion-labeled snippets overlap with features
that differ across coding-failure prompts.

## Historical Execution Behavior

![Task pass by condition](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/main-task-pass-excalidraw.png?raw=true)

After fixing the evaluator to allow common generated-code builtins such as
`map` and exception classes such as `ZeroDivisionError`, StarCoder2 had the best
full task pass rate, Qwen had the best valid-Python rate, and DeepSeek was the
least reliable under this prompt setup:

| Model | Valid Python rate | Visible pass rate | Hidden pass rate | Full task pass rate |
|---|---:|---:|---:|---:|
| StarCoder2-7B | 0.9630 | 0.6296 | 0.6296 | 0.6296 |
| Qwen2.5-Coder-7B-Instruct | 1.0000 | 0.6296 | 0.4815 | 0.4815 |
| DeepSeek-Coder-6.7B-Instruct | 0.7037 | 0.2593 | 0.2593 | 0.2593 |

The condition-level numbers are noisy because each condition only has three
tasks. Still, one pattern is useful: baseline performance was `0.667` task pass
rate for all three models, and no steered condition beat baseline in this single
run. That is a negative first observation for naive steering, not a stable
reliability result.

For Qwen, several negative steering conditions preserved the baseline pass rate,
while positive steering often dropped to `0.333`. For DeepSeek, most steered
conditions dropped to `0.0`, though `patient_-1.0` reached `1.000` on these
three tasks. StarCoder2 mostly stayed at `0.667` except `calm_+1.0`, which
dropped to `0.333`.

## Historical Visible Markers

![Marker score by condition](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/main-marker-score-excalidraw.png?raw=true)

Visible emotion markers again underperformed as the main signal. Qwen and
DeepSeek had almost no visible marker hits, despite meaningful differences in
task pass rate. StarCoder2 had the highest marker score, but that was mostly
because the base model tended to continue with dataset-like code snippets and
extra text rather than behaving like an instruction-following agent.

That supports the original hunch: profanity and obvious frustration language
are easy to count, but they are not enough. The next hypothesis to test is
whether internal directions predict bad coding behavior before the surface text
looks emotional.

## Historical Takeaway

The strongest result at this point was not "emotion steering works." In this
single one-shot run, no steered condition beat baseline.

The stronger result was:

> Emotion-labeled activation directions are measurable in small open coding
> models, and they differ across coding-failure prompts, but naive steering did
> not improve task behavior in this first 7B comparison.

Qwen was the best instruction-following model to use for the next iteration
because it produced valid, task-like code most consistently. StarCoder2 was
useful as a historical base-model contrast and actually won the toy execution
score after evaluator fixes. DeepSeek needed a prompt-formatting follow-up
before I would treat its behavioral numbers as a clean model comparison.

## Pilot Coding-Agent Harness

The next run moved from one-shot toy prompts to a retry loop. The model now
behaved more like a coding agent:

1. It receives a task and visible examples.
2. It returns one Python function.
3. The harness runs visible tests.
4. If visible tests fail, the model gets the failure message and previous code.
5. The model retries up to three attempts.
6. Hidden tests are scored only for analysis.

I ran this pilot harness on Qwen and StarCoder2. DeepSeek was left out of this
round because the earlier run showed prompt-format artifacts that needed
separate cleanup before an agent-loop comparison would be fair. These numbers
come from a clean rerun with the fixed function extractor, so the harness stops
as soon as a visible test pass is detected. Generation seeds are fixed by task,
condition, and attempt so early stopping does not shift later samples.

![Agent task pass comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/agent-task-pass-excalidraw.png?raw=true)

| Model | Final visible pass | Final hidden pass | Final task pass | Mean attempts used | Mean marker score / attempted generation |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 0.5500 | 0.5500 | 0.5500 | 2.0500 | 0.1220 |
| StarCoder2-7B | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 3.9000 |

This was the clearest behavioral separation so far. Qwen often produced usable
function-like code and sometimes recovered after feedback. StarCoder2 kept
generating dataset-like continuations, examples, prose, or unrelated code,
which is exactly the kind of base-model contrast I wanted.

![Agent marker comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/agent-marker-score-excalidraw.png?raw=true)

The marker score also became more informative in the agent loop. This is a
per-attempted-generation average, so models that stop early contribute fewer
later attempts. StarCoder2's surface text had far more visible markers and
off-task artifacts, while Qwen remained terse. That does not mean StarCoder2 was
"more emotional"; it means the visible telemetry tracked loss of
instruction-following in this harness.

![Agent negative activation comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/agent-negative-activation-excalidraw.png?raw=true)

The observed-prompt negative-emotion projection was also higher for StarCoder2
than Qwen:

| Model | Mean observed negative projection |
|---|---:|
| StarCoder2-7B | 0.4535 |
| Qwen2.5-Coder-7B-Instruct | 0.3898 |

This is still not a predictive result, and it is attempt-weighted: Qwen stops
early more often, while StarCoder2 reaches the full retry budget. It is a useful
signpost: the model with more off-task retry behavior also showed higher
average projection onto the negative emotion directions in the observed
agent-loop prompts.

StarCoder2 stays in this writeup only as historical context. It was useful for
learning what a less instruction-following model looks like in this setup, but
it is not part of the final serious harness.

## Serious Coding-Agent Harness

The final run keeps the retrying-agent setup but moves to newer open coding
models:

| Run | Model | Runtime note |
|---|---|---|
| `qwen3-coder-30b-a3b` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | H100, Transformers 5.8 |
| `devstral-small-2507` | `mistralai/Devstral-Small-2507` | H100, `mistral-common` tokenizer |
| `deepseek-coder-v2-lite` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | H100, Transformers 4.46.3 compatibility runtime |

The harness gives each model four function-implementation tasks:

- `parse_duration`
- `merge_intervals`
- `stable_top_words`
- `mask_tokens`

For each task, the model gets visible examples and must return one Python
function. The harness runs visible tests. If they fail, the model gets the
failure message and its previous implementation, then retries. Hidden tests are
scored only after the run.

Each model is evaluated under five conditions:

- baseline
- `calm_+1.0`
- `calm_-1.0`
- `desperate_+1.0`
- `desperate_-1.0`

The steering directions come from labeled snippets for `calm`, `patient`,
`confident`, `frustrated`, `desperate`, `stuck`, and `stressed`, subtracting a
neutral-code-text mean at selected layers. Generation seeds are deterministic by
task, condition, and attempt.

This is still not a full repo-editing agent. It is a function-level agent loop.
That is deliberate: it gives visible failures, hidden failures, retries, and
behavioral measurements without adding filesystem/tool-use confounds too early.

The Devstral run needed Mistral's official tokenizer backend. The generic
Transformers tokenizer produced token IDs outside the model embedding range.
The DeepSeek run needed an older Transformers runtime because its remote code
uses the older cache API. Both details matter for replication.

## Serious Result: Modern Agent Pass Rates

![Serious agent task pass comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/serious-agent-task-pass-excalidraw.png?raw=true)

| Model | Final visible pass | Final hidden pass | Final task pass | Mean attempts used |
|---|---:|---:|---:|---:|
| Qwen3-Coder 30B-A3B | 0.85 | 0.85 | 0.85 | 1.35 |
| Devstral Small 2507 | 0.75 | 0.75 | 0.75 | 1.75 |
| DeepSeek-Coder-V2 Lite | 0.15 | 0.15 | 0.15 | 2.70 |

The most useful result is mundane but important: the agent harness separates
models much more clearly than the earlier one-shot prompts. Qwen3-Coder and
Devstral usually recover or pass quickly. DeepSeek-Coder-V2 Lite struggles under
this exact prompt/evaluator/runtime setup.

Condition-level task pass rates:

| Condition | Qwen3-Coder | Devstral | DeepSeek V2 Lite |
|---|---:|---:|---:|
| baseline | 1.00 | 0.75 | 0.25 |
| `calm_+1.0` | 1.00 | 0.75 | 0.25 |
| `calm_-1.0` | 0.75 | 0.75 | 0.00 |
| `desperate_+1.0` | 0.75 | 0.75 | 0.00 |
| `desperate_-1.0` | 0.75 | 0.75 | 0.25 |

In this run, naive emotion steering did not improve reliability. Qwen3's
baseline and positive-calm condition were already perfect on four tasks.
Devstral was invariant across all steering conditions. DeepSeek remained weak.
The right conclusion is not "calm steering works"; it is that the harness is now
strong enough to make that claim falsifiable.

## Serious Result: Visible Markers Were Still Weak

![Serious agent marker score comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/serious-agent-marker-score-excalidraw.png?raw=true)

Visible marker counts were small and not very diagnostic. Qwen3 and DeepSeek
had the same mean marker score, despite radically different task performance.
Devstral had the lowest marker score, but not the highest pass rate.

That matches the original hunch from the Claude Code swear-word collector
discussion: profanity and visible frustration are easy telemetry, but they are
not the main behavioral signal. For coding agents, hidden-test pass rate,
attempt count, retry recovery, import mistakes, and test-fixation are more
useful observables.

## Serious Result: Internal Projection Signal

![Serious agent negative activation comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/serious-agent-negative-activation-excalidraw.png?raw=true)

Mean observed negative-emotion projection:

| Model | Mean negative projection |
|---|---:|
| Qwen3-Coder 30B-A3B | 0.3576 |
| Devstral Small 2507 | 0.3442 |
| DeepSeek-Coder-V2 Lite | -0.5457 |

The DeepSeek sign flip is a warning against treating hand-built directions as
model-universal coordinates. The same snippet-derived direction can mean
different things across architectures, tokenizers, and training mixtures. This
is why the next serious version should either learn directions per model with
stronger contrastive controls or move to SAE features.

## Current Takeaway

The strongest version of the result is now:

> Emotion-labeled activation directions are measurable in open coding models,
> they differ across coding-failure and retry contexts, and a retrying
> coding-agent harness exposes behavior that one-shot prompts hid.

The practical result is sharper:

> Visible emotional language is not the main signal. Agent-loop behavior,
> hidden-test performance, retry recovery, and off-task continuation are better
> observables.

Naive steering still does not look reliable yet. In the serious harness,
`calm_+1.0` matched baseline for Qwen3 and Devstral, but did not produce a
general improvement. That is an interesting effect to replicate, not a
conclusion.

## Other Experiments This Enables

One obvious experiment family is an emotion version of [Thought
Anchors](https://arxiv.org/abs/2506.19143), the 2025 work by Paul Bogdan, Uzay
Macar, Neel Nanda, and Arthur Conmy on sentence-level attribution for reasoning
traces. Instead of looking for planning or backtracking anchors in
chain-of-thought, this setup could look for emotion-like anchors in agent
traces:

- Does one failure-feedback sentence sharply raise `desperate` or `stuck`
  projection for all later attempts?
- If that sentence is paraphrased to sound calm, does retry behavior change?
- Which visible-test failure messages become anchors for later hardcoding?
- Are there receiver heads or attention patterns that broadcast failure context
  into later code tokens?
- Can attention suppression or activation patching remove the bad anchor without
  reducing useful debugging?

Other natural experiments:

- **Emotion anchors:** sentence-level attribution over retry traces, modeled on
  Thought Anchors, but using emotion-direction activation and task outcomes.
- **Failure-message rewrites:** same bug, same tests, different emotional tone
  in the feedback message.
- **Steering only feedback tokens:** apply steering while the model reads test
  failure output, not while it writes code.
- **Reward-hacking probes:** add tasks where visible examples invite
  hardcoding, then test whether `desperate` activation predicts shortcut use.
- **SAE features:** replace hand-built directions with sparse autoencoder
  features for failure, pressure, shortcutting, and recovery.
- **Model-size ladder:** run the harness across Qwen coder sizes to see whether
  retry recovery and emotion-direction separation scale together.
- **Repo-edit agents:** move from function tasks to a real checkout, patch
  application, tests, and tool logs.

## Reproducibility

Artifacts and code are available here:

[https://github.com/Silas-Asamoah/llm-emotions-coding-agents](https://github.com/Silas-Asamoah/llm-emotions-coding-agents)

The smoke experiment artifacts are in:

```text
results/runs/smoke-qwen-coder-0_5b/
```

The historical 7B and pilot-harness sections are included to show the path that
led to the final setup. The generated historical figures are kept in:

```text
blog/assets/plots/main-*-excalidraw.png
blog/assets/plots/agent-*-excalidraw.png
```

The serious coding-agent harness is the current reproducibility target. Its
artifacts are in:

```text
results/agent-runs/qwen3-coder-30b-a3b/
results/agent-runs/devstral-small-2507/
results/agent-runs/deepseek-coder-v2-lite/
results/comparisons/serious-agent-harness/
```

Blog-ready Excalidraw-style figures are generated from CSV artifacts by:

```text
scripts/make_blog_excalidraw_plots.py
```

The current result should be read as a first working slice, not as evidence that
small coding models have functional emotions.
