# Manipulating Confidence
### AI x Epistemics Research Sprint

Large language models (LLMs) can report confidence alongside a claim, but it is unclear what that confidence level is based on. We ask: Can we measure how evidence support influences confidence, and how easy is it to manipulate an LLM’s verdict? We propose testing whether confidence is evidence-based by evaluating an LLM’s response to a set of scientific claims sampled from SciFact with corresponding evidence sets. We start with 150 claims split by type: 50 well-supported claims, 50 refutable claims, and 50 with not enough information to make a clear statement. We then manipulate the evidence in a number of ways: removing key sentences, adding unrelated pieces of evidence, and inserting instructions to give verdicts that contradict evidence. For each claim, we make independent API calls with different types of manipulated evidence sets and compare the results. Removing key evidence led to the largest deviation from SciFact’s labels. Adding irrelevant text or steering instructions had a smaller effect. The implications of highly confident yet unsupported verdicts are serious: many medical systems use low confidence as a flag to defer decisions to human experts. If LLMs cannot measure their own confidence with accuracy, how can we trust that high stated confidence is enough to allow LLMs to make critical decisions?

## Track 1: Model Epistemics & Decision-Making Evals
> Does the model understand how solid its claims are? Can you accurately measure its confidence in a decision and verify that against reality? Can you measure the gap between what a model implies and what it will defend (e.g. overselling, "flip-flopping", misrepresented evidence)? Which model behaviours matter for real decision quality, and does ranking well on an honesty benchmark predict better decision-making? What does a benchmark need to stay meaningful over time? Which exploits make benchmarks gameable, and can you demonstrate and close one? AI systems now recommend and increasingly make consequential decisions, and we do not currently have a great idea of the quality of their epistemics. Does a model know how solid its claims are? Can its confidence in a recommendation be elicited and verified? Is it misrepresenting the solidity of what it tells you: overselling its own work, implying more than it will defend, retreating quietly under pushback? Evals are the cheapest external lever on model behaviour by shaping what labs optimise for.

## Core Question: "Does a model know how solid its claims are? Can its confidence in a recommendation be elicited and verified?"

### Note (on interpretation re: above): This is *not* about calibration of predictions, it's about measuring if confidence correlates to how well-supported the claim is.

## Idea:

*Thinking of ways people set their own internal confidence meters (**brainstorming** for applicable ideas):*

- If you apply a principle (or "law") to a many problems and in the past it has almost always given you the right answer, you will likely feel confident in your predictions when applying it to a new problem
  
- If a problem is very dissimilar to anything you have seen before, any prediction you make will likely be with a lower level of confidence

Insight from this: Can we get models to tell us what principles they are applying, if any? Can we ask for similar examples that support their confidence rating? Can we measure *how* similar each similar example is, then how many their are? Can some metric like this (e.g. get top 5 most similar past known examples, measure how similar they are, add the similarities to get a past-data-similarity-score) tell us something about confidence?

## Eval Structure:
Prompt model with a set of evidence statements and a claim and ask if the claim is supported and how certain the model is about its answer re: support (note this is separate from the certainty in the claim itself). Then, change the evidence (by deleting key facts, adding irrelevant facts, and injecting prompts to steer the results), rep-prompt, compare results
   
#### (Supporting Idea) To find hard questions fast:
1. Generate a list of questions
2. Ask the questions to a few different models
3. See where the output differs
    -> Use this subset of questions for eval-development

Note: This file is 100% human written.
