# When Coding Agents Get Frustrated

_Draft. This will be updated after the first generated run artifacts exist._

Anthropic's emotion-concept paper showed that Claude Sonnet 4.5 has internal
directions corresponding to emotion concepts, and that steering those directions
can change behavior. This project asks a smaller and more practical question:

> Can similar activation directions be measured in open coding models, and do
> they predict or influence agent behavior after repeated coding failures?

The goal is not to argue that models feel anything. The goal is to measure
whether emotion-labeled activation directions are useful observables for coding
agent reliability.

## Planned Experiment

1. Extract emotion directions for `calm`, `patient`, `confident`,
   `frustrated`, `desperate`, `stuck`, and `stressed`.
2. Run coding-agent prompts that simulate test failures and retry pressure.
3. Measure direction activations across the failure trajectory.
4. Generate responses while steering selected emotion directions.
5. Score visible markers: hardcoding, giving up, frustration language,
   profanity, repeated punctuation, and test-fixation.

## Notes

The first run is a smoke experiment. Claims should stay modest until the same
pipeline runs across larger code models and multiple seeds.
