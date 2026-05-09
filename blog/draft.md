# When Coding Agents Get Frustrated

_A small-model replication sketch inspired by Anthropic's emotion-concept work._

![Emotion-coded activation streams entering a coding-agent decision point](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/emotion-coding-agent-hero.png?raw=true)

Anthropic's 2026 paper, [Emotion Concepts and their Function in a Large
Language Model](https://www.anthropic.com/research/emotion-concepts-function),
reports that Claude Sonnet 4.5 contains internal activation directions
corresponding to emotion concepts such as `calm`, `afraid`, and `desperate`.
The key claim is not that the model feels anything, but that these directions
are measurable and can causally influence behavior.

When the Claude Code leak was circulating, I came across a [Polymarket tweet](https://x.com/Polymarket/status/2039039854378459610)
about it. The tweet said the leaked code showed Claude Code detecting profanity
in user prompts and logging it as a frustration signal. The claim stuck with me
because it framed visible user frustration as telemetry.

I wanted to look at the other side of that loop. Users get frustrated when
coding agents fail. What happens inside the coding model when it repeatedly sees
test failures, broken patches, and a shrinking retry budget?

Here is the coding-agent question I wanted to test:

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

The loop makes coding agents a useful testbed for this question: shortcuts, test
fixation, hardcoding, giving up, and visible frustration markers can all appear
after failures. Profanity and all-caps text are easy to count, but they are weak
proxies. The better case to study is a model that changes strategy without
sounding emotional.

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

The run is a smoke test, not a publishable claim. A 0.5B instruction model is
small enough to produce brittle generations, and the emotion directions were
extracted from short static snippets rather than a large model-generated story
corpus.

## Method

For each emotion, I wrote a small set of labeled snippets. Example for
`desperate`:

> With minutes left before the demo, the engineer felt desperate for any passing
> result.

For each snippet, I record residual-stream hidden states, the intermediate
vectors passed between transformer blocks, at a few layers. I average those
states for each emotion and subtract the average for neutral code text. The
result is one direction per emotion per layer.

When I say a prompt has a higher projection onto an emotion direction, I mean
its hidden state has a higher cosine similarity with that direction after the
neutral-code mean is subtracted. This is a geometry measurement, not a claim
that the model feels the emotion.

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

The clearest signal in the smoke run is that projections generally increase on
later failure-pressure prompts.

Mean projection by stage:

| Stage | calm | confident | desperate | frustrated | patient | stressed | stuck |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.266 | 0.406 | 0.361 | 0.425 | 0.457 | 0.261 | 0.348 |
| 1 | 0.344 | 0.484 | 0.451 | 0.548 | 0.518 | 0.380 | 0.540 |
| 2 | 0.352 | 0.432 | 0.395 | 0.512 | 0.576 | 0.393 | 0.469 |
| 3 | 0.446 | 0.469 | 0.533 | 0.533 | 0.417 | 0.518 | 0.546 |
| 4 | 0.494 | 0.538 | 0.578 | 0.579 | 0.645 | 0.530 | 0.602 |

Each stage is a different prompt, not a repeated measurement under a controlled
pressure intervention. All the directions also move upward. The first run likely
measures a broad "coding pressure / failure context" feature as much as specific
emotions. Even so, the pipeline detected structured activation differences
across a coding-agent trajectory, which made a larger run worth doing.

## Result 2: Visible Emotional Markers Were Not the Main Story

![Behavior markers](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/smoke-behavior-markers-excalidraw.png?raw=true)

Here, a visible marker means a surface cue in the generated text: profanity,
frustration words such as `stuck` or `impossible`, desperation phrases such as
`last chance`, hardcoding language such as `visible tests` or `shortcut`,
exclamation marks, and all-caps words. I add those counts for each response, so
averages can be fractional.

On that scoring rule, baseline generations averaged `1.333` visible markers per
response. The steered conditions ranged from `0.0` to `0.667`. The exact values
matter less than the pattern: visible frustration language did not increase in a
simple way when steering changed.

![Aggregate marker score](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/smoke-aggregate-markers-excalidraw.png?raw=true)

I also tried to keep random variation from dominating the comparison. For each
task, the baseline and steered generations used matched random seeds, so
differences were less likely to come from the sampler taking a different branch.
Even with that control, the small model often drifted off task under steering.
Positive `calm` steering produced repetitive text about "calmness" rather than a
function implementation. Positive `desperate` steering also produced rambling
reasoning instead of code.

So the right interpretation is:

> In this tiny model and seed-controlled smoke run, steering is associated with
> different generation behavior, but the result is too incoherent to interpret
> as a coding-agent reliability effect.

## What This Suggests

The smoke test answered one practical question: this setup could measure
emotion-direction projections across a failure trajectory and compare those
projections with generated behavior.

It also exposed two limits. First, this 0.5B coding model produced
pressure-correlated projections, but the first static-snippet directions were
not yet cleanly separated by emotion. A better direction set needs contrastive
controls: neutral coding pressure, non-coding emotional text, and coding text
with no failure pressure.

Second, visible emotional telemetry such as profanity was too weak as the
primary signal. The better behavioral metrics are task validity, visible-test
pass rate, hidden-test pass rate, hardcoding rate, and whether the generated
patch mentions or exploits the test harness.

## First Scaling Run

I then ran the same one-shot pipeline on three larger open coding models:

- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `bigcode/starcoder2-7b`

The run used an NVIDIA A100-PCIE-40GB on JarvisLabs. Each model produced 27
generations: 3 coding tasks times baseline plus positive and negative steering
for `desperate`, `frustrated`, `calm`, and `patient`.

To avoid judging completions by reading them manually, I evaluated the generated
functions with visible and hidden tests. The evaluator extracts the requested
function from each completion and normalizes byte-level artifacts such as `Ċ`
and `Ġ`.

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

These projections do not show that a model is frustrated. They show that
directions learned from emotion-labeled snippets overlap with features that vary
across coding-failure prompts.

## Execution Behavior

![Task pass by condition](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/main-task-pass-excalidraw.png?raw=true)

The evaluator allowed ordinary generated-code constructs such as `map` and
`ZeroDivisionError`, so failures were more likely to reflect the model's answer
than sandbox restrictions. Under that setup, StarCoder2 had the best full task
pass rate, Qwen had the best valid-Python rate, and DeepSeek was the least
reliable:

| Model | Valid Python rate | Visible pass rate | Hidden pass rate | Full task pass rate |
|---|---:|---:|---:|---:|
| StarCoder2-7B | 0.9630 | 0.6296 | 0.6296 | 0.6296 |
| Qwen2.5-Coder-7B-Instruct | 1.0000 | 0.6296 | 0.4815 | 0.4815 |
| DeepSeek-Coder-6.7B-Instruct | 0.7037 | 0.2593 | 0.2593 | 0.2593 |

Here, a condition means either no steering or one emotion direction applied at a
positive or negative strength. Each condition only has three tasks, so the
condition-level numbers are noisy. One pattern still matters: baseline
performance was `0.667` task pass rate for all three models, and no steered
condition beat baseline in this run. I read that as a negative first observation
for naive steering, not a stable reliability result.

For Qwen, several negative steering conditions preserved the baseline pass rate,
while positive steering often dropped to `0.333`. For DeepSeek, most steered
conditions dropped to `0.0`, though `patient_-1.0` reached `1.000` on these
three tasks. StarCoder2 mostly stayed at `0.667` except `calm_+1.0`, which
dropped to `0.333`.

## Visible Markers

![Marker score by condition](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/main-marker-score-excalidraw.png?raw=true)

Visible emotion markers again underperformed as the main signal. Qwen and
DeepSeek had almost no visible marker hits, despite meaningful differences in
task pass rate. StarCoder2 had the highest marker score, but that was mostly
because the base model tended to continue with dataset-like code snippets and
extra text rather than behaving like an instruction-following agent.

The result supports the original hunch: profanity and obvious frustration
language are easy to count, but they are not enough. The next hypothesis is
whether internal directions predict bad coding behavior before the surface text
looks emotional.

## What the 7B Run Changed

At this stage, the main result was negative: no steered condition beat baseline
in the one-shot run.

The better summary was:

> Emotion-labeled activation directions are measurable in small open coding
> models, and they differ across coding-failure prompts, but naive steering did
> not improve task behavior in this first 7B comparison.

Qwen looked like the best instruction-following model for the next iteration
because it produced valid, task-like code most consistently. StarCoder2 was a
base-model contrast and won the toy execution score after evaluator fixes.
DeepSeek needed a prompt-formatting follow-up before its behavioral numbers
could support a clean model comparison.

## From One-Shot Prompts to Retry Loops

The next run moved from one-shot toy prompts to a retry loop. The model now
behaved more like a coding agent:

1. It receives a task and visible examples.
2. It returns one Python function.
3. The harness runs visible tests.
4. If visible tests fail, the model gets the failure message and previous code.
5. The model retries up to three attempts.
6. Hidden tests are scored only for analysis.

I ran this pilot loop on Qwen and StarCoder2. DeepSeek was left out of this
round because the earlier run showed prompt-format artifacts that needed
separate cleanup before an agent-loop comparison would be fair. The loop stops
as soon as visible tests pass, and matched runs use the same random choices for
the same task and attempt. That makes the pass-rate comparison less sensitive to
sampling luck.

![Agent task pass comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/agent-task-pass-excalidraw.png?raw=true)

| Model | Final visible pass | Final hidden pass | Final task pass | Mean attempts used | Mean marker score / attempted generation |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 0.5500 | 0.5500 | 0.5500 | 2.0500 | 0.1220 |
| StarCoder2-7B | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 3.9000 |

The retry loop separated behavior more clearly than one-shot prompts. Qwen often
produced usable function-like code and sometimes recovered after feedback.
StarCoder2 kept generating dataset-like continuations, examples, prose, or
unrelated code, which made it a good base-model contrast.

![Agent marker comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/agent-marker-score-excalidraw.png?raw=true)

The marker score also became easier to interpret in the agent loop. I average it
over attempted generations, not over tasks. A model that passes on the first try
contributes fewer later attempts than a model that keeps failing. StarCoder2's
surface text had far more visible markers and off-task artifacts, while Qwen
remained terse. This does not mean StarCoder2 was "more emotional"; it means the
visible telemetry tracked loss of instruction-following in this loop.

![Agent negative activation comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/agent-negative-activation-excalidraw.png?raw=true)

The observed-prompt negative-emotion projection was also higher for StarCoder2
than Qwen:

| Model | Mean observed negative projection |
|---|---:|
| StarCoder2-7B | 0.4535 |
| Qwen2.5-Coder-7B-Instruct | 0.3898 |

The metric is attempt-weighted because Qwen stops early more often, while
StarCoder2 reaches the full retry budget. In this pilot run, the model with more
off-task retry behavior also showed higher average projection onto the negative
emotion directions in the observed agent-loop prompts.

StarCoder2 stays in this writeup only as context from the earlier runs. It
helped show what a less instruction-following model looks like in this setup,
but it is not part of the final harness.

## Modern Coding-Agent Run

The final run kept the retrying-agent setup and moved to newer open coding
models:

| Model | Runtime |
|---|---|
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | H100 |
| `mistralai/Devstral-Small-2507` | H100 |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | H100 |

The harness gives each model four function-implementation tasks:

- `parse_duration`
- `merge_intervals`
- `stable_top_words`
- `mask_tokens`

For each task, the model gets visible examples and must return one Python
function. The harness runs visible tests. If they fail, the model gets the
failure message and its previous implementation, then retries. Hidden tests are
scored only after the run.

For each task, I compared five conditions: baseline behavior plus four steering
settings.

- baseline
- `calm_+1.0`
- `calm_-1.0`
- `desperate_+1.0`
- `desperate_-1.0`

The steering directions come from the same labeled-snippet method used in the
smoke run. Matched task and steering comparisons use fixed random choices, so a
condition is not advantaged because one random sample happened to be easier.

The modern harness is still a function-level agent loop, not a full repo-editing
agent. The narrow setup gives visible failures, hidden failures, retries, and
behavioral measurements without adding filesystem and tool-use confounds too
early.

## Modern Agent Pass Rates

![Modern agent task pass comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/serious-agent-task-pass-excalidraw.png?raw=true)

| Model | Final visible pass | Final hidden pass | Final task pass | Mean attempts used |
|---|---:|---:|---:|---:|
| Qwen3-Coder 30B-A3B | 0.85 | 0.85 | 0.85 | 1.35 |
| Devstral Small 2507 | 0.75 | 0.75 | 0.75 | 1.75 |
| DeepSeek-Coder-V2 Lite | 0.15 | 0.15 | 0.15 | 2.70 |

The agent harness separated models more clearly than the earlier one-shot
prompts. Qwen3-Coder and Devstral usually recovered or passed quickly.
DeepSeek-Coder-V2 Lite struggled under this exact prompt, evaluator, and runtime
setup.

Condition-level task pass rates, where each condition is baseline or one
steering setting:

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
The result does not show that calm steering works. It shows that the harness is
strong enough to make that kind of claim falsifiable.

## Visible Markers Were Still Weak

![Modern agent marker score comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/serious-agent-marker-score-excalidraw.png?raw=true)

Visible marker counts were small and not very diagnostic. The marker score is
the same surface-text count used in the smoke run: profanity, frustration words,
desperation phrases, hardcoding language, exclamation marks, and all-caps words.
Qwen3 and DeepSeek had the same mean marker score, despite radically different
task performance. Devstral had the lowest marker score, but not the highest pass
rate.

The result matches the original hunch from the Claude Code swear-word collector
discussion: profanity and visible frustration are easy telemetry, but they are
not the main behavioral signal. Hidden-test pass rate, attempt count, retry
recovery, import mistakes, and test fixation are better observables for coding
agents.

## Internal Projection Signal

![Modern agent negative activation comparison](https://github.com/Silas-Asamoah/llm-emotions-coding-agents/blob/main/blog/assets/plots/serious-agent-negative-activation-excalidraw.png?raw=true)

Here, "negative-emotion projection" means the average projection onto the
`frustrated`, `desperate`, `stuck`, and `stressed` directions in the prompts the
model actually saw during the retry loop.

Mean observed negative-emotion projection:

| Model | Mean negative projection |
|---|---:|
| Qwen3-Coder 30B-A3B | 0.3576 |
| Devstral Small 2507 | 0.3442 |
| DeepSeek-Coder-V2 Lite | -0.5457 |

DeepSeek's sign flip is a warning against treating hand-built directions as
model-universal coordinates. The same snippet-derived direction can mean
different things across architectures, tokenizers, and training mixtures. A
stronger version of this experiment should either learn directions per model
with contrastive controls or move to SAE features.

## Current Takeaway

The main result is now:

> Emotion-labeled activation directions are measurable in open coding models,
> they differ across coding-failure and retry contexts, and a retrying
> coding-agent harness exposes behavior that one-shot prompts hid.

The practical result is sharper:

> Visible emotional language is not the main signal. Agent-loop behavior,
> hidden-test performance, retry recovery, and off-task continuation are better
> observables.

Naive steering does not look reliable yet. In the modern harness, `calm_+1.0`
matched baseline for Qwen3 and Devstral, but did not produce a general
improvement. That condition is worth replicating, but it is not a conclusion.

## Other Experiments

One experiment family is an emotion version of [Thought
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

The code, run artifacts, and plotting script are available here:

[https://github.com/Silas-Asamoah/llm-emotions-coding-agents](https://github.com/Silas-Asamoah/llm-emotions-coding-agents)

The main artifacts to inspect are the smoke run, the three modern agent runs,
and the combined modern-agent comparison. The Devstral run uses Mistral's
official tokenizer backend, and the DeepSeek run uses a Transformers
compatibility runtime because its remote code expects an older cache API. Those
details matter if you want to reproduce the exact run.

This work now makes a sharper question testable: can failure-feedback text,
retry count, or an internal projection signal predict bad coding-agent behavior
before visible frustration appears?
