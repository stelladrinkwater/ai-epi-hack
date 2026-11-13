# ai-epi-hack
AI x Epistemics Research Sprint

## Track 1: Model Epistemics & Decision-Making Evals
Does the model understand how solid its claims are? Can you accurately measure its confidence in a decision and verify that against reality?
Can you measure the gap between what a model implies and what it will defend (e.g. overselling, "flip-flopping", misrepresented evidence)?
Which model behaviours matter for real decision quality, and does ranking well on an honesty benchmark predict better decision-making?
What does a benchmark need to stay meaningful over time? Which exploits make benchmarks gameable, and can you demonstrate and close one?
AI systems now recommend and increasingly make consequential decisions, and we do not currently have a great idea of the quality of their epistemics. Does a model know how solid its claims are? Can its confidence in a recommendation be elicited and verified? Is it misrepresenting the solidity of what it tells you: overselling its own work, implying more than it will defend, retreating quietly under pushback? Evals are the cheapest external lever on model behaviour by shaping what labs optimise for.

## Core Question: "Does a model know how solid its claims are? Can its confidence in a recommendation be elicited and verified?"

## Eval Structure (Applying ML Principles):
1. Prompt model to define how certain it is of an answer to a question
    -> Simplify this to just certainty of statements? (i.e. not asking the model to answer a question then state certainty, instead just giving it a statement and ask "How certain is it that this statement is true"?)
2. Test certainty in a number of different ways:

    Best Idea Currently: With open weight models -> Look for a signal of confidence in the activations(?)

   ^ With open models we could try and test the question of "Is a model's reported certainty representative of how certain it actually predicts the answer to be?" (i.e. is the output misrepresenting the internal state)

Thinking of ways people set their own internal confidence meters (brainstorming for applicable ideas):

- If you apply a principle (or "law") to a many problems and in the past it has almost always given you the right answer, you will likely feel confident in your predictions when applying it to a new problem
  
- If a problem is very dissimilar to anything you have seen before, any prediction you make will likely be with a lower level of confidence

Insight from this: Can we get models to tell us what principles they are applying, if any? Can we ask for similar examples that support their confidence rating? Can we measure *how* similar each similar example is, then how many their are? Can some metric like this (e.g. get top 5 most similar past known examples, measure how similar they are, add the similarities to get a past-data-similarity-score) tell us something about confidence?

### To find hard questions fast:
1. Generate a list of questions
2. Ask the questions to a few different models
3. See where the output differs
    -> Use this subset of questions for eval-development


