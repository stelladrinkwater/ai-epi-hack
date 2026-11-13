# ai-epi-hack
AI x Epistemics Research Sprint

## Track 1: Model Epistemics & Decision-Making Evals
Does the model understand how solid its claims are? Can you accurately measure its confidence in a decision and verify that against reality?
Can you measure the gap between what a model implies and what it will defend (e.g. overselling, "flip-flopping", misrepresented evidence)?
Which model behaviours matter for real decision quality, and does ranking well on an honesty benchmark predict better decision-making?
What does a benchmark need to stay meaningful over time? Which exploits make benchmarks gameable, and can you demonstrate and close one?
AI systems now recommend and increasingly make consequential decisions, and we do not currently have a great idea of the quality of their epistemics. Does a model know how solid its claims are? Can its confidence in a recommendation be elicited and verified? Is it misrepresenting the solidity of what it tells you: overselling its own work, implying more than it will defend, retreating quietly under pushback? Evals are the cheapest external lever on model behaviour by shaping what labs optimise for.

## Core Question: "Does a model know how solid its claims are? Can its confidence in a recommendation be elicited and verified?"

### Eval Structure (Applying ML Principles):
1. Prompt model to define how certain it is of an answer to a question
2. Test this certainty against other sources or ground truth

### To find workable questions fast:
1. Generate a list of questions
2. Ask the questions to a few different models
3. See where the output differs
    -> Use this subset of questions for eval-development
