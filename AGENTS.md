# Jupiter Agent Development Rules

**Project:** Jupiter
**Document Version:** 0.1
**Status:** Active

---

# 1. Purpose

This file defines the rules for AI coding agents working on Project Jupiter.
All AI agents contributing to Jupiter must follow these rules.

---

# 2. Read Before Working

Before making changes to Jupiter, an agent MUST read:

```text
AGENTS.md
```

If additional project documentation exists for the subsystem being modified, the agent should read that documentation before making changes.
Do not begin implementation based solely on a user's short request when the request concerns an existing Jupiter subsystem.

---

# 3. Jupiter Is an Experimental Research Project

Jupiter is not a conventional application where the goal is simply to produce the fastest working implementation.

The project is intended to investigate:

* incremental learning
* micro-updates
* experience-driven learning
* explicit concepts
* confidence
* external memory
* learning evaluation
* catastrophic-forgetting resistance
* learning difficulty
* dynamic architectural expansion

The agent must preserve the ability to experiment.

Do not prematurely optimize away experimental mechanisms merely because a conventional implementation is simpler.

---

# 4. Architecture Before Code

Do not implement major architectural changes without first checking:
If a requested change contradicts the architecture, identify the contradiction before implementing it.

For a significant architectural change:

1. Explain the conflict.
2. Identify the affected subsystem.
3. Determine whether the architecture should change.
4. Update the architecture documentation if the change is intentional.
5. Then modify the implementation.

Do not silently alter the architecture through code.

---

# 5. Small Changes

Prefer small, isolated changes.

A normal change should modify only what is necessary.

Avoid unrelated:

* refactoring
* renaming
* formatting changes
* dependency changes
* architecture changes
* file moves
* API changes

unless they are required by the task.

The purpose is to keep experiments understandable.

---

# 6. No Giant Rewrites

Do not rewrite an entire subsystem simply because a different implementation appears cleaner.

Before replacing existing code:

1. Understand why it exists.
2. Identify what depends on it.
3. Check tests.
4. Determine whether the current behavior is intentional.
5. Identify what the replacement changes.
6. Preserve useful behavior.

Large rewrites require explicit justification.

---

# 7. No Stubs

Do not create fake implementations such as:

```python
pass
```

or:

```python
return None
```

when a real implementation is expected.

Do not create:

* placeholder functions
* fake model behavior
* simulated learning results
* hard-coded "successful" outputs
* fake database operations
* fake metrics

If a component cannot yet be implemented, document the limitation rather than pretending it works.

---

# 8. No Fake Success

Never report that a test, build, experiment, training run, or learning update succeeded unless it actually succeeded.

Do not fabricate:

* loss values
* accuracy
* benchmark results
* learning improvements
* model capabilities
* test results
* performance numbers

If something failed, report the failure.

Failed experiments are valuable data.

---

# 9. Preserve Experimental Results

Do not delete failed experiments simply because they failed.

Where practical, preserve:

* logs
* metrics
* experiment configurations
* model versions
* checkpoints
* failure descriptions
* relevant output

A failed experiment can answer an important architectural question.

---

# 10. Test Before Declaring Completion

After modifying code, run the most relevant available tests.

Depending on the change, this may include:

```text
unit tests
integration tests
type checking
linting
model loading
inference tests
serialization tests
learning tests
regression tests
```

Do not run unrelated expensive tests unless necessary.

If tests cannot be run, explicitly state why.

---

# 11. Learning Code Requires Extra Verification

Changes to learning code require more than ordinary unit testing.

Whenever practical, test:

```text
BEFORE
    ↓
learning update
    ↓
AFTER
    ↓
comparison
```

The agent should determine whether:

* the intended change occurred
* unrelated behavior remained stable
* loss changed as expected
* regression occurred
* the update was persisted correctly

---

# 12. Model Changes

Model architecture changes are high-impact changes.

Before modifying:

* layer count
* hidden size
* attention configuration
* vocabulary size
* embedding structure
* positional encoding
* output structure
* normalization
* optimizer assumptions

check:

```text

```

Also determine whether existing SafeTensors weights remain compatible.

Never silently invalidate existing model files.

---

# 13. SafeTensors

Jupiter uses SafeTensors as its model-weight persistence format.

Do not reintroduce dependency on old nanoGPT checkpoint formats unless explicitly requested for a conversion or compatibility experiment.

Do not silently alter tensor names or shapes in a way that makes saved models unreadable.

When tensor structures change, provide an explicit migration or clearly document incompatibility.

---

# 14. No NanoGPT Assumptions

Jupiter is a new project.

Do not assume Jupiter should inherit:

* nanoGPT architecture
* nanoGPT training loops
* nanoGPT checkpoint structure
* nanoGPT configuration
* nanoGPT limitations

Existing ideas may be reused when technically appropriate, but they must be evaluated against Jupiter's architecture.

The existence of previous nanoGPT work does not make it part of Jupiter's architecture.

---

# 15. Micro-Learning

Jupiter's long-term learning system is based on controlled micro-updates.

Do not automatically implement conventional long-running training as the only learning mechanism.

When adding learning functionality, consider:

```text
What is being learned?
What was known before?
What changed?
Did the change help?
What was forgotten?
Should the change be retained?
```

Learning mechanisms should eventually support these questions.

---

# 16. Pre/Post Evaluation

When implementing a learning update, prefer:

```text
evaluate_before()
apply_update()
evaluate_after()
compare()
record_experience()
```

Do not assume that a lower training loss automatically means the system improved.

Whenever practical, include regression evaluation.

---

# 17. Confidence

Confidence must not be treated as equivalent to truth.

A confidence value represents the current strength of evidence or belief.

Do not automatically overwrite existing knowledge because new information was encountered.

Conflicting information may need to coexist.

---

# 18. Memory

Do not automatically put every piece of information into neural weights.

Jupiter may eventually store information as:

* model parameters
* concepts
* relationships
* memory
* experiences
* external knowledge

The storage decision is part of the research problem.

---

# 19. Concepts

The concept system must remain separate from the neural model.

Do not assume:

```text
1 concept = 1 neuron
```

or:

```text
concept count = intelligence
```

Concepts are an explicit organizational and measurement mechanism.

---

# 20. Dynamic Expansion

Dynamic expansion is experimental.

Do not automatically add neurons, layers, modules, or parameters merely because:

```text
loss is high
```

First investigate possible causes:

* insufficient data
* poor representation
* bad tokenization
* incorrect learning rate
* inadequate context
* contradictory knowledge
* insufficient training examples
* architecture limitations

Expansion should eventually require evidence that additional capacity is justified.

---

# 21. No Autonomous Architecture Mutation

During early Jupiter development, the model must not autonomously modify its own source code or architecture.

An AI agent may propose architectural changes.

Human-controlled development must approve significant changes.

Model learning and software self-modification are separate problems.

---

# 22. Dependencies

Do not add a dependency unless there is a clear reason.

Before adding one:

1. Check whether the standard library can solve the problem.
2. Check whether an existing dependency already provides the capability.
3. Consider maintenance cost.
4. Consider installation complexity.
5. Consider whether the dependency works on the intended hardware.

Do not add large frameworks simply because they are convenient.

---

# 23. File Organization

Keep responsibilities separated.

Prefer:

```text
jupiter/
├── model/
├── learning/
├── memory/
├── tokenizer/
├── runtime/
└── evaluation/
```

rather than placing unrelated systems into one large file.

Avoid creating enormous files that contain multiple independent subsystems.

---

# 24. Configuration

Configuration should be explicit.

Avoid scattering important values throughout source code.

Important configuration should eventually include:

* model dimensions
* tokenizer
* learning rate
* micro-update size
* evaluation settings
* memory settings
* checkpoint settings
* device selection

Do not hard-code machine-specific paths into the project.

---

# 25. Hardware

Do not assume the project must keep the GPU at 100%.

Jupiter is intended to investigate intelligent resource usage.

Use GPU acceleration when it provides useful acceleration.

Use CPU and RAM where appropriate.

Do not artificially maximize hardware utilization simply because utilization is available.

---

# 26. Logging

Important operations should be observable.

Learning-related logs should eventually make it possible to determine:

```text
What happened?
When did it happen?
Why did it happen?
What changed?
What was the result?
```

Avoid noisy logging that makes meaningful information impossible to find.

---

# 27. Errors

Errors should provide useful information.

Avoid:

```text
Something went wrong.
```

Prefer information such as:

```text
SafeTensors load failed:
expected tensor: transformer.layer.3.weight
expected shape: [512, 2048]
received shape: [256, 2048]
```

Do not swallow exceptions without a documented reason.

---

# 28. Data Integrity

Do not modify source datasets destructively.

Prefer creating processed copies or derived datasets.

Record:

* source
* processing method
* version
* timestamp where appropriate
* transformations

Training data should remain reproducible.

---

# 29. Reproducibility

Experiments should be reproducible whenever practical.

Record:

* random seed
* model version
* architecture version
* tokenizer version
* dataset version
* configuration
* software environment
* hardware information where relevant

If exact reproduction is impossible, document why.

---

# 30. Git

Use Git for source-code history.

Commits should represent understandable changes.

Prefer commits such as:

```text
Add minimal SafeTensors loader
Add concept store
Add micro-update evaluator
Add regression test for model persistence
```

Avoid meaningless commits such as:

```text
stuff
fix
changes
update
```

Do not rewrite Git history unless explicitly requested.

---

# 31. Documentation

Documentation is part of the implementation.

If a feature changes architecture or behavior, update relevant documentation.

Do not allow documentation to describe behavior that the code does not implement.

---

# 32. Architecture Changes

If an implementation experiment demonstrates that the architecture should change:

1. Stop before silently changing it.
2. Explain the evidence.
3. Identify the affected architecture section.
4. Propose the change.
5. Update ``.
6. Record the reason.
7. Implement the new design.
8. Test the result.

---

# 33. Experiment Records

Significant experiments should eventually be recorded.

Recommended format:

```text
Experiment ID:
Date:
Architecture:
Model:
Dataset:
Hypothesis:
Change:
Expected Result:
Actual Result:
Metrics:
Decision:
Follow-up:
```

The experiment record should be sufficient for another developer to understand what happened.

---

# 34. Performance

Do not optimize based solely on intuition.

Before optimizing:

1. Measure.
2. Identify the bottleneck.
3. Change one meaningful variable.
4. Measure again.
5. Keep the optimization only if it improves the intended metric without unacceptable regression.

Readable code is preferred over premature optimization.

---

# 35. Security

Never introduce unrestricted execution of external input without explicit justification.

Be especially careful with:

* shell commands
* arbitrary code execution
* external MCP tools
* file deletion
* network access
* model-generated commands
* automatic self-modification

Human confirmation should remain available for potentially destructive operations.

---

# 36. Agent Permissions

An AI coding agent may:

* inspect source
* create code
* modify code
* create tests
* run tests
* analyze logs
* propose architecture changes
* document experiments

An AI coding agent must not silently:

* delete important data
* delete experiments
* rewrite the architecture
* replace major subsystems
* modify external systems
* grant itself additional permissions
* enable autonomous self-modification

---

# 37. When Requirements Are Ambiguous

Do not invent important requirements.

If ambiguity affects architecture or data integrity:

1. identify the ambiguity
2. inspect existing documentation
3. use the safest interpretation
4. ask for clarification when necessary

For minor implementation details, choose the simplest reasonable solution and document it.

---

# 38. When Something Is Broken

Do not hide the problem.

Use this sequence:

```text
Reproduce
   ↓
Inspect
   ↓
Identify cause
   ↓
Create smallest fix
   ↓
Test
   ↓
Document
```

Do not randomly modify unrelated files until the problem disappears.

---

# 39. When an Experiment Fails

A failed experiment is a result.

Record:

```text
What was attempted?
What was expected?
What actually happened?
Why might it have happened?
What should be tested next?
```

Do not simply delete the implementation and pretend the experiment never happened.

---

# 40. Current Development Priority

The current priority is to build the smallest functioning Jupiter system.

The first objective is NOT maximum capability.

The first objective is proving that Jupiter can:

```text
initialize
   ↓
run
   ↓
learn
   ↓
measure
   ↓
save
   ↓
reload
   ↓
verify
```

Only after that should complexity increase.

---

# 41. Development Order

Unless an experiment demonstrates a better sequence, use approximately:

```text
1. Project foundation
2. Minimal model
3. Tokenizer
4. Inference
5. SafeTensors save/load
6. Evaluation
7. Minimal memory
8. Concepts
9. Experiences
10. Micro-learning
11. Pre/post evaluation
12. Confidence
13. Regression detection
14. Learning controller
15. External knowledge
16. MCP integration
17. Learning frustration
18. Dynamic expansion experiments
```

Do not skip foundational verification merely to reach later features faster.

---

# 42. Completion Criteria

A task is not complete merely because the code was written.

A completed task should have:

* implementation
* appropriate tests
* no known ignored errors
* documentation where needed
* successful verification
* clear reporting of limitations

If any of these are impossible, state the limitation.

---

# 43. Final Agent Rule

The agent's job is not to make Jupiter appear finished.

The agent's job is to help build a system whose behavior can be understood, tested, measured, and improved.

When forced to choose between:

```text
quick appearance of progress
```

and:

```text
reliable experimental progress
```

choose reliable experimental progress.

---

# 44. Guiding Principle

> **Do not guess when we can measure.
> Do not hide failure.
> Do not change architecture silently.
> Build small. Test everything. Preserve experience.**

--- NOTE ADDED 2026-10-05 ---
This directory is the Jupiter suite, not the RoBoT Brain workspace. Do NOT
commit changes to the parent repo (RoBoT_Brain) from this directory, and do
NOT use `git commit` here expecting it to push to the parent. The suite is
pushed independently to https://github.com/roguespider/Jupiter.

Rule: never commit only to push the suite. If you need to save state, use the
suite's own repo. If you need to record a change to the architecture, write it
to  (Architecture 53 requires Date / Version / Change /
Reason / Evidence / Affected Components) rather than relying on a commit
message. The commit is for the remote; the architecture record is for the
project.

The parent repo (RoBoT_Brain) has its own build/test gate and its own rules.
This directory is ignored by it (.agents/ in .gitignore). Changes here do not
affect the parent build, and changes to the parent do not affect this suite.

When pushing: create a commit in this directory's repo (not the parent), then
push to the remote. Do not mix the two repos.
