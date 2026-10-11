JUPITER_ARCHITECTURE.md

1. Purpose and Philosophy
2. Core Principles
3. System Overview
4. Model Architecture
5. Tokenization
6. Learning System
7. Micro-Updates
8. Concept System
9. Knowledge Representation
10. Memory
11. Experience
12. Confidence
13. Conflict Resolution
14. Evaluation
15. Learning Frustration
16. Dynamic Expansion
17. Architecture Feedback
18. External Knowledge
19. MCP Integration
20. Inference
21. Persistence / SafeTensors
22. Training vs Learning
23. Experimentation
24. Rollback / Recovery
25. Hardware / Compute
26. Data Pipeline
27. Security / Safety
28. Observability
29. Testing
30. Future Research


# Project Jupiter Architecture

**Document:** JUPITER_ARCHITECTURE.md
**Architecture Version:** 0.7
**Project Status:** Experimental / Foundational
**Architecture Type:** Continuously Learning Neural System
**Primary Goal:** Develop a small, observable, incrementally learnable artificial intelligence system that can acquire knowledge through controlled micro-updates, external memory, experience, and eventually dynamic architectural growth.

---

# 1. Purpose

Project Jupiter is an experimental artificial intelligence project designed to investigate an alternative to conventional large-scale language-model training.

Jupiter is not intended to begin as a large pretrained model.

Jupiter will begin as a relatively small neural system and develop through controlled learning.

The central idea is:

> **Jupiter should learn continuously rather than relying exclusively on large, infrequent training runs.**

Learning will occur through small, measurable updates.

Each update should provide information about:

* what Jupiter attempted to learn
* what it already knew
* what changed
* whether the change improved performance
* whether the new information conflicts with previous knowledge
* whether the current architecture has sufficient capacity
* whether the information should be stored externally instead of modifying model parameters

The system must remain observable throughout development.

---

# 2. Core Philosophy

Jupiter follows these principles.

## 2.1 Knowledge Is Not Identical to Weights

Information stored in neural parameters is only one form of knowledge.

Jupiter may maintain information through:

* neural parameters
* explicit concepts
* relationships
* external memory
* experiences
* confidence values
* learned procedures
* contextual information

No assumption should be made that every piece of information belongs inside model weights.

---

## 2.2 Learning Is More Important Than Training

Traditional training generally treats a dataset as something repeatedly presented to a model to minimize loss.

Jupiter treats learning as a process.

The system should be capable of asking, internally or through its learning controller:

```text
What am I trying to learn?
What do I already know?
How confident am I?
What conflicts with existing knowledge?
Did this information improve me?
Should I change?
Where should this information live?
```

Training remains an available mechanism, but it is not the definition of learning.

---

## 2.3 Small Changes Are Preferable to Blind Large Changes

Jupiter should favor controlled micro-updates over unnecessarily large training sessions.

A learning event should ideally modify only what is necessary.

This makes learning:

* observable
* measurable
* reversible
* experimentally useful
* easier to debug

---

## 2.4 Experience Creates Learning

Jupiter should record experiences associated with learning.

An experience may contain:

* input
* interpretation
* attempted response
* expected result
* actual result
* loss
* confidence
* update performed
* result after update
* whether the update helped
* relevant concepts
* relationships discovered

Experience should become a source of future learning.

---

# 3. Development Philosophy

Architecture precedes implementation.

Implementation is allowed to change when experiments demonstrate that an architectural assumption is incorrect.

No implementation should become permanent merely because it was written first.

The development loop is:

```text
Architecture
    ↓
Implementation
    ↓
Experiment
    ↓
Measurement
    ↓
Observation
    ↓
Architectural Revision
    ↓
Implementation
```

The project should favor evidence over assumptions.

---

# 4. Architecture Status System

Architectural ideas must have a status.

## PROPOSED

An idea exists but has not yet been implemented.

## EXPERIMENTAL

The idea has been implemented but has not been sufficiently validated.

## IMPLEMENTED

The feature exists and functions as designed.

## VERIFIED

The feature has passed repeatable tests demonstrating the intended behavior.

## DEPRECATED

The feature or architectural assumption has been replaced.

## REJECTED

An experiment demonstrated that the approach should not be used.

Experimental features must not be represented as proven architecture.

---

# 5. High-Level System

Jupiter consists of several cooperating systems.

```text
                         JUPITER
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           Model         Learning       Memory
              │             │             │
              └─────────────┼─────────────┘
                            │
                       Experience
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           Concepts      Evaluation    Confidence
              │             │             │
              └─────────────┼─────────────┘
                            │
                    Architecture
                     Feedback Loop
                            │
                    Dynamic Expansion
```

The initial implementation does not need to implement every subsystem.

The architecture defines the eventual direction while implementation proceeds incrementally.

---

# 6. Neural Model

Jupiter will begin with a relatively small neural language model.

The initial model size is intentionally modest.

The exact architecture is experimental.

Possible initial architectures include:

* transformer
* hybrid transformer
* recurrent components
* attention-based components
* expandable neural architecture

The initial implementation should prioritize:

1. understandability
2. observability
3. reproducibility
4. stable incremental learning
5. efficient experimentation

Model size should not be increased merely for the sake of increasing parameter count.

---

# 7. Initial Model Size

The first useful production-like Jupiter model may target approximately 128 million parameters.

However, development should begin with an even smaller model if necessary.

The project should prove the learning system before committing to a larger model.

Possible progression:

```text
Tiny Jupiter
    ↓
Small Jupiter
    ↓
~128M Jupiter
    ↓
Expandable Jupiter
```

The 128M target is a milestone, not a requirement.

---

# 8. Tokenization

Jupiter requires a tokenizer capable of representing:

* natural language
* programming languages
* symbols
* punctuation
* malformed input
* slang
* common typing errors
* source-code structures

The tokenizer must not be selected solely because it is common.

It should be evaluated against Jupiter's intended workloads.

Tokenizer experiments must be measurable.

Changing tokenization after substantial training may invalidate existing model weights.

Therefore tokenizer decisions should be made before long-term model development.

---

# 9. Knowledge Representation

Jupiter will use multiple forms of representation.

## Neural Knowledge

Information represented through model parameters.

## Explicit Concepts

Named or identifiable concepts maintained outside the model.

## Relationships

Connections between concepts.

## Experiences

Historical learning events.

## External Knowledge

Information that Jupiter can retrieve without permanently modifying model weights.

The architecture deliberately avoids forcing all information into one representation.

---

# 10. Concepts

A concept represents a meaningful unit of knowledge.

Examples may include:

```text
Python
function
memory
recursion
HTTP
ownership
Rust
gravity
database
error handling
```

A concept may contain:

* identifier
* description
* confidence
* maturity
* relationships
* prerequisites
* supporting experiences
* conflicting information
* usage frequency
* last verification time

Concepts should not automatically correspond one-to-one with neural features.

They are an external representation used to observe and organize learning.

---

# 11. Concept Count

Jupiter should maintain a measurable concept count.

Concept count may become an important signal for determining whether learning is producing meaningful knowledge.

However:

> More concepts does not automatically mean more intelligence.

The system must eventually distinguish between:

* useful concepts
* duplicate concepts
* fragmented concepts
* weak concepts
* strongly connected concepts
* obsolete concepts

Concept count is therefore a measurement, not a score.

---

# 12. Relationships

Concepts may have relationships such as:

```text
IS_A
PART_OF
DEPENDS_ON
USED_BY
SIMILAR_TO
CONTRADICTS
CAUSES
PRECEDES
REQUIRES
IMPLEMENTS
CO_OCCURS
```

Relationships may have their own confidence.

For example:

```text
Rust
    └── uses ──> ownership
                 confidence: 0.97
```

Relationships should not automatically be treated as facts.

They must be evaluated based on evidence.

CO_OCCURS was added under Architecture 53 and is the only type Jupiter derives
for itself. It records that one learning source surfaced two terms together.
That is a fact about the corpus, not a claim about meaning: Jupiter still
cannot say why the terms appear together, let alone whether they are related.

It was added because the list above had no honest way to express the
measurement. Recording co-occurrence as SIMILAR_TO would claim a likeness that
term frequency cannot support, and recording it as IS_A or PART_OF would claim a
structure Jupiter has no evidence for. Naming the measurement is better than
forcing it into a type that means something else. Every other type remains
available for a future system that has real evidence to assert it.

---

# 13. Confidence

Jupiter should maintain confidence independently from simple binary truth values.

Example:

```text
Concept:
    HTTP/2

Confidence:
    0.96

Evidence:
    147 observations

Contradictions:
    2

Last verified:
    recent
```

Conflicting information should not automatically destroy previous information.

Instead, Jupiter should preserve competing representations when evidence warrants doing so.

Example:

```text
Knowledge A
confidence: 0.91

Knowledge B
confidence: 0.63
```

The system can later update these values as additional evidence becomes available.

---

# 14. Knowledge Conflict

When new information conflicts with existing information, Jupiter should not immediately overwrite the old information.

The system should:

1. detect the conflict
2. identify the affected concepts
3. preserve both pieces of information
4. evaluate evidence
5. update confidence
6. determine whether the conflict is contextual
7. record the experience

Some apparent conflicts may actually be context differences.

---

# 15. Memory

Jupiter should eventually use multiple memory classes.

## Working Memory

Information currently relevant to an active task.

## Long-Term Knowledge

Stable information retained for future use.

## Experience Memory

Records of previous learning events and outcomes.

## External Memory

Information stored outside model parameters and retrieved when needed.

The exact storage technology is implementation-dependent.

SQLite is a strong candidate for the initial implementation because it provides a simple, portable, inspectable database.

---

# 16. Memory Versus Weights

Jupiter should develop a mechanism for determining whether information should:

* remain external
* modify model parameters
* become an explicit concept
* become a relationship
* become an experience
* exist in multiple forms

This decision should eventually consider:

* frequency
* importance
* confidence
* generalization
* retrieval cost
* learning cost
* permanence
* usefulness
* contradiction risk

This system is experimental and will evolve through testing.

---

# 17. Learning

Learning is represented as a controlled event.

A simplified learning cycle:

```text
Input
  ↓
Understand
  ↓
Retrieve Existing Knowledge
  ↓
Evaluate Difference
  ↓
Generate Learning Proposal
  ↓
Estimate Confidence
  ↓
Perform Micro-Update
  ↓
Evaluate
  ↓
Accept / Reject / Modify
  ↓
Record Experience
```

The learning controller is responsible for coordinating this process.

---

# 18. Micro-Updates

Jupiter should support small learning updates rather than requiring long conventional training sessions.

A micro-update may consist of:

* one example
* several examples
* one concept
* one relationship
* one short sequence
* one error correction
* one programming pattern
* one verified fact

The update should be evaluated before and after modification.

---

# 19. Pre/Post Evaluation

Every important learning update should ideally have:

```text
BEFORE
  ↓
evaluation
  ↓
UPDATE
  ↓
AFTER
  ↓
comparison
```

Possible measurements:

* loss
* accuracy
* perplexity
* task performance
* concept confidence
* retrieval quality
* regression tests
* generalization
* catastrophic forgetting

The system should determine whether the update actually helped.

---

# 20. Update Acceptance

A proposed update should not automatically be accepted.

The learning controller may classify an update as:

```text
ACCEPT
REJECT
DEFER
REVIEW
ROLLBACK
```

An update can be rejected when:

* improvement is negligible
* regression occurs
* confidence is insufficient
* contradictory evidence is stronger
* catastrophic forgetting occurs
* the update destabilizes the model

---

# 21. Learning Frustration

Jupiter should eventually measure repeated learning difficulty.

Learning frustration is not an emotion.

It is an engineering measurement.

Example:

```text
Concept: recursive ownership pattern

Attempts: 27
Successful generalization: 4
Repeated errors: 23
Loss improvement: minimal

Frustration Index: HIGH
```

Repeated difficulty may indicate:

* insufficient data
* poor representation
* insufficient model capacity
* inadequate context
* conflicting knowledge
* tokenizer limitations
* inappropriate architecture

The frustration signal should therefore feed into architectural analysis.

---

# 22. Dynamic Expansion

Dynamic expansion is an experimental long-term feature.

Jupiter may eventually be capable of increasing its representational capacity.

Possible expansion mechanisms include:

* additional neurons
* additional hidden dimensions
* additional layers
* specialized modules
* memory expansion
* expert modules
* auxiliary networks

Expansion should not occur simply because loss is high.

The system should distinguish between:

```text
"I don't know this."

and

"I cannot represent this effectively with my current architecture."
```

That distinction is central to the research objective.

---

# 23. Expansion Feedback Loop

A possible future system:

```text
Learning difficulty
       ↓
Repeated failure
       ↓
Analyze cause
       ↓
Insufficient capacity?
       ↓
      YES
       ↓
Propose expansion
       ↓
Experiment
       ↓
Evaluate
       ↓
Keep / rollback
```

Expansion proposals must be measurable and reversible.

---

# 24. Training Versus Learning

Traditional training may still be used.

However, Jupiter distinguishes:

### Training

A computational process used to modify model parameters.

### Learning

The larger process of acquiring, evaluating, organizing, remembering, and applying knowledge.

Training is therefore one mechanism inside the learning architecture.

---

# 25. Data Pipeline

Data may originate from:

* curated datasets
* documentation
* programming examples
* conversations
* generated examples
* external knowledge
* verified corrections
* Jupiter's own experiences

Data should be processed before being used for learning when practical.

The pipeline should eventually support:

```text
Raw Data
   ↓
Parsing
   ↓
Cleaning
   ↓
Classification
   ↓
Deduplication
   ↓
Concept Detection
   ↓
Relationship Detection
   ↓
Evaluation
   ↓
Learning Proposal
```

---

# 26. Programming Knowledge

Programming is a major Jupiter capability.

The system should eventually support:

* multiple programming languages
* syntax
* semantics
* translation
* debugging
* error interpretation
* warning interpretation
* code transformation
* code generation
* code explanation
* documentation
* cross-language reasoning

Programming knowledge should be treated as structured knowledge rather than simply memorized text.

---

# 27. Natural Language

Jupiter should eventually support natural-language understanding and generation.

Initial language capability should focus on:

* English
* programming language text
* technical documentation
* informal text
* common errors
* slang
* malformed input

Additional languages can be introduced experimentally.

---

# 28. External Knowledge

Jupiter should be capable of retrieving information from external sources.

External information may come from:

* MCP servers
* databases
* files
* APIs
* documentation
* search systems

External knowledge should not automatically become permanent internal knowledge.

The system should evaluate whether information deserves retention.

---

# 29. MCP Integration

MCP should provide an interface between Jupiter and external tools.

Potential capabilities include:

* retrieving information
* reading files
* writing approved information
* querying databases
* interacting with tools
* accessing RoBoT services

Jupiter must remain capable of operating without MCP during core model testing.

MCP is an integration layer, not a fundamental requirement for the neural model.

---

# 30. Persistence

Model parameters will use **SafeTensors**.

Jupiter must not depend on an old nanoGPT checkpoint format.

The model architecture and tensor metadata must be stored separately from arbitrary training scripts.

A saved Jupiter instance should contain enough information to determine:

* architecture
* tokenizer configuration
* model version
* tensor version
* training state when applicable
* learning state when applicable

---

# 31. Experiment Tracking

Every meaningful experiment should produce a record.

An experiment record should ideally contain:

```text
experiment_id
timestamp
model_version
architecture_version
input/data identifier
learning method
hyperparameters
before metrics
update
after metrics
decision
notes
```

This makes Jupiter's development scientifically useful rather than dependent on memory.

---

# 32. Rollback

Learning must be reversible.

Before significant model changes, Jupiter should be capable of creating a checkpoint.

If an update produces unacceptable regression:

```text
Current Model
     ↓
Regression detected
     ↓
Rollback
     ↓
Restore previous state
     ↓
Record failed experiment
```

Failed experiments are valuable information and should not simply disappear.

---

# 33. Catastrophic Forgetting

Jupiter must monitor for knowledge degradation.

An update that improves one capability but destroys another is not automatically considered successful.

Regression tests should eventually cover previously learned capabilities.

The system should track:

```text
New Capability
+
Existing Capability Preservation
```

rather than measuring only the new task.

---

# 34. Evaluation

Jupiter should eventually have multiple evaluation categories.

## Language

Understanding and generation.

## Programming

Code comprehension and transformation.

## Knowledge

Recall and application.

## Reasoning

Multi-step problem solving.

## Memory

Retrieval and retention.

## Learning

Ability to improve from examples.

## Stability

Ability to learn without destroying existing capability.

## Expansion

Ability to determine when additional capacity is useful.

---

# 35. Observability

Jupiter must be observable during development.

The system should eventually expose:

* current model state
* active learning event
* loss
* confidence
* concepts
* concept count
* memory operations
* update magnitude
* update acceptance
* rejected updates
* regression detection
* expansion proposals
* GPU utilization
* CPU utilization
* RAM utilization
* storage usage

The goal is to avoid the development experience of:

```text
"Something has been training for six hours.
I have no idea what it is doing."
```

---

# 36. Hardware Philosophy

Jupiter should be designed to use available hardware intelligently.

The development environment may contain:

* high-end GPU
* large system RAM
* CPU resources
* local storage

GPU utilization should not be treated as a goal by itself.

A long-running 100% GPU workload is not automatically better than a lower-utilization learning process.

The system should eventually balance:

* CPU
* RAM
* GPU
* storage
* computation time
* learning frequency

---

# 37. Continuous Operation

Long-term Jupiter may operate continuously.

Possible loop:

```text
Observe
  ↓
Receive information
  ↓
Evaluate
  ↓
Learn
  ↓
Measure
  ↓
Store experience
  ↓
Rest
  ↓
Review
  ↓
Continue
```

Continuous operation must not imply uncontrolled self-modification.

Changes must remain observable and governed by the learning system.

This loop is implemented. `autonomy.py` observes, decides, acts and records,
and `run_jupiter.bat autonomous` runs it until a person stops it. What it drops
when it starts is the cycle budget, because a budget is something a person sets
for a run they intend to watch. What it keeps is the wall clock, the idle limit
and every control gate, because an agent that cannot be stopped is unattended
rather than autonomous.

Rest is a real stage rather than the absence of one. A loop with nothing to
learn says so, records why, and waits, up to a bounded number of passes before
it concludes there is nothing coming rather than waiting forever.

Each pass records the decision and the reason for it. An agent that acts on its
own state without recording why it chose that action cannot be audited, and
Architecture 38 asks for human control over what it changes. `run_jupiter.bat
status` prints what the loop has been deciding and what it is waiting on.

---

# 38. Safety and Control

Jupiter must not be permitted to arbitrarily modify its own source code during early development.

Model learning and software self-modification are separate concerns.

Initial development requires explicit human control over:

* architecture changes
* source-code changes
* model expansion
* external tool access
* persistent memory changes
* destructive operations

Autonomous modification may be investigated later as a separate research problem.

---

# 39. Development Environment

Zed is the primary development environment.

Zed may be used for:

* source editing
* AI-assisted development
* project navigation
* terminal operations
* MCP integration
* agent-assisted testing
* Git operations
* collaboration

Delta/DeltaDB may be used for development collaboration and synchronized project work.

DeltaDB is not Jupiter's cognitive memory.

The distinction must remain explicit.

---

# 40. Agent Development Rules

AI coding agents working on Jupiter must:

1. Read `JUPITER_ARCHITECTURE.md`.
2. Read `AGENTS.md`.
3. Understand the current implementation status.
4. Make the smallest reasonable change.
5. Avoid unnecessary rewrites.
6. Run relevant tests.
7. Report failures honestly.
8. Never fabricate successful tests.
9. Never silently change architecture.
10. Document significant architectural changes.
11. Avoid unnecessary dependencies.
12. Avoid placeholder implementations.
13. Avoid dead code.
14. Preserve experimental results.
15. Never delete data merely to make tests pass.

---

# 41. Testing Philosophy

Tests should exist at multiple levels.

## Unit Tests

Individual functions and components.

## Integration Tests

Subsystem interactions.

## Learning Tests

Whether a learning update produces the intended result.

## Regression Tests

Whether existing capabilities survive updates.

## Architecture Tests

Whether architectural assumptions remain valid.

## Long-Running Tests

Whether repeated micro-learning remains stable.

Covered by tests/test_long_running.py, which runs the real controller over
several cycles rather than calling into it. It is the slowest thing in the
suite by an order of magnitude, and it exists because until it was added no test
invoked run_worker at all, so the whole continuous path had never been run end
to end.

---

# 42. Experimental Discipline

Jupiter should not change multiple fundamental variables simultaneously when an experiment is intended to determine causality.

For example:

Bad experiment:

```text
change tokenizer
change architecture
change optimizer
change dataset
change learning rate
```

Good experiment:

```text
change one major variable
measure result
compare against baseline
```

Complex experiments may eventually require multiple variables, but the reason must be documented.

---

# 43. Versioning

Jupiter should maintain separate versions for:

* architecture
* model
* tokenizer
* learning system
* memory schema
* experiments

Example:

```text
Architecture: 0.1
Model: 0.1.0
Tokenizer: 0.1.0
Learning: 0.1.0
Memory: 0.1.0
```

These versions should not be assumed to advance together.

---

# 44. Initial Development Sequence

The initial development order should be approximately:

```text
1. Architecture
2. Project rules
3. Minimal model
4. Tokenizer
5. Inference
6. Save/load
7. Evaluation
8. Minimal memory
9. Concept system
10. Experience system
11. Micro-learning
12. Pre/post evaluation
13. Confidence
14. Regression detection
15. Learning controller
16. External knowledge
17. MCP integration
18. Learning frustration
19. Expansion experiments
```

This order may change when experiments demonstrate a better sequence.

---

# 45. First Milestone

The first meaningful Jupiter milestone is NOT:

> Train a 128M parameter model.

The first milestone is:

> **Create a tiny model that can learn a small piece of information, measure the result, save itself, reload itself, and demonstrate whether the update helped.**

If this cannot be accomplished reliably, increasing model size will not solve the fundamental problem.

---

# 46. Second Milestone

The second milestone is:

> **Teach Jupiter to remember what happened when it learned.**

Jupiter should be able to record:

```text
I attempted X.
I knew Y beforehand.
I changed Z.
Performance changed by N.
The update was accepted/rejected.
```

---

# 47. Third Milestone

The third milestone is:

> **Teach Jupiter to organize what it learns into concepts and relationships.**

At this stage we begin testing whether explicit knowledge structures improve learning and retrieval.

---

# 48. Fourth Milestone

The fourth milestone is:

> **Teach Jupiter to perform repeated micro-learning without catastrophic degradation.**

This is where the project begins to diverge significantly from conventional training workflows.

---

# 49. Fifth Milestone

The fifth milestone is:

> **Investigate whether learning difficulty can provide useful feedback about model capacity.**

Only after the previous systems are stable should dynamic expansion become a serious implementation target.

---

# 50. Long-Term Objective

The long-term objective is to investigate whether a relatively small artificial intelligence can develop useful capabilities through:

```text
Knowledge
    +
Experience
    +
Memory
    +
Micro-learning
    +
Evaluation
    +
Confidence
    +
Architectural feedback
    +
Additional capacity when justified
```

rather than relying exclusively on enormous static training runs.

The project does not assume this approach will outperform conventional large language models.

The purpose is to experimentally determine what is possible.

---

# 51. Fundamental Rule

Jupiter must remain an experiment.

No architectural idea is sacred.

No implementation is permanent merely because it works.

No failed experiment is wasted if its result is recorded.

The system should evolve through evidence.

> **Build small. Learn slowly. Measure everything. Preserve experience. Expand only when evidence says expansion is necessary.**

---

# 52. Current Architecture Status

| Component                               | Status               |
| --------------------------------------- | -------------------- |
| Overall architecture                    | PROPOSED             |
| Neural model                            | IMPLEMENTED          |
| Tokenizer                               | IMPLEMENTED          |
| Inference                               | IMPLEMENTED          |
| SafeTensors persistence                 | IMPLEMENTED          |
| Micro-learning                          | IMPLEMENTED          |
| Pre/post evaluation                     | IMPLEMENTED          |
| Rollback                                | IMPLEMENTED          |
| Experiment tracking                     | IMPLEMENTED          |
| Update acceptance                       | EXPERIMENTAL         |
| Confidence                              | EXPERIMENTAL         |
| Regression detection                    | EXPERIMENTAL         |
| Learning controller                     | EXPERIMENTAL         |
| Continuous operation                    | EXPERIMENTAL         |
| Observability                           | EXPERIMENTAL         |
| Experience system                       | EXPERIMENTAL         |
| Concept system                          | EXPERIMENTAL         |
| Relationship system                     | EXPERIMENTAL         |
| Memory                                  | EXPERIMENTAL         |
| Retrieval                              | EXPERIMENTAL         |
| Evaluation                             | EXPERIMENTAL         |
| Single entry point                     | IMPLEMENTED          |
| Project test suite                     | IMPLEMENTED          |
| Learning frustration                    | EXPERIMENTAL         |
| Dynamic expansion                       | EXPERIMENTAL         |
| MCP integration                         | EXPERIMENTAL         |
| External knowledge                      | EXPERIMENTAL         |
| Consulting the outside world            | EXPERIMENTAL         |
| Safety and control                      | EXPERIMENTAL         |
| Knowledge assertions                    | EXPERIMENTAL         |
| Zed development environment             | SELECTED             |
| Delta/DeltaDB development collaboration | SELECTED             |

EXPERIMENTAL means implemented and demonstrated once, but not yet validated
across enough runs to be treated as architecture. Regression detection, the
learning controller and continuous operation are all EXPERIMENTAL for that
reason: each has been shown to work, and none has survived a long run yet.

Regression detection currently compares a reserved document against itself
across cycles. It cannot yet detect forgetting that affects the reserved
document and the learned document equally, and it has no notion of which
capabilities matter.

run_jupiter.bat is the single entry point for the project. Every module is
reached from it, and a test resolves the import graph from the batch file and
fails if any module cannot be reached, so the project cannot quietly grow a
part that the entry point does not drive. Two files are reached through a mode
rather than the default chain: the telemetry follower never returns, and remote
GPU training costs money, so the chain uses a bounded summary of the first and
an availability check of the second.

The context window is 512 tokens. At the byte tokenizer's one token per byte
that is 512 characters, which is why a learned tokenizer exists alongside it and
is now the default: the BPE vocabulary is what makes a 512-token window able to
hold a question, some retrieved knowledge and an answer. The byte tokenizer
remains available and remains the lossless fallback, because it cannot be
invalidated by a change of corpus.

The figure quoted for the vocabulary needs its method stated, because two
different measurements were being reported as one. prepare_dataset compares
vocabularies on a 4000-character sample taken from the first curated document,
which gives 2.384 bytes per token and a window of roughly 1220 characters.
Measured across all twelve curated documents the same vocabulary gives about
2.14 bytes per token, and a window nearer 1100 characters. The narrower sample is
the optimistic one; the corpus-wide number is the honest one and is the figure to
quote.

The last of the Architecture 34 categories to have a measurement behind it was
Expansion, which could not be measured until dynamic expansion existed to
measure. It exists now in two halves. Memory expansion changes how much Jupiter
extracts from a struggling document and is reversible by lowering a limit.
Capacity growth adds a layer, which is only possible because the shape guard now
tells a shape that got bigger from a shape that does not fit, so a grown model
inherits the weights it had rather than starting over. Growth is refused without
a human, capped at 32 layers, and judged against the holdout loss recorded at the
moment it was authorised, so the report can say whether expanding the model was
worth it rather than only that it happened.

Programming and Reasoning used to be unmeasured too, on the grounds that nothing
existed to evaluate; that was not quite right, since the corpus contains
programming documents and every cycle already records a before and after holdout
loss. Both are measured by whether bounded updates reduce loss on that class of
material, and neither number is a capability claim: a low loss on a code document
means its bytes are predicted, not that the program is correct.

MCP is an HTTP endpoint over JSON-RPC, built on the standard library, exposing
five tools that are all capabilities Jupiter already had. It binds to loopback
only, and it sits behind the Architecture 38 external-access control, because
reaching a model over a socket is external access whatever the caller calls
itself. It is an integration layer: the model, the learning loop and the
tokenizer run identically whether or not it is started.

Architecture 28 is now reached rather than only reported. The decision layer
consults configured external sources when the corpus is dry, gated on the
external domain, and files what passes Architecture 28's retention test into
the raw corpus for curation rather than into the knowledge store. The
question it asks is built from the concepts Jupiter understands least well.
It has no sources configured by default, so a run that has not named any
behaves exactly as it did before: the capability exists and is refused until
both a source and a grant are supplied.

---

# 53. Architecture Change Policy

Any significant change to this document should include:

```text
Date
Architecture Version
Change
Reason
Evidence
Affected Components
```

Architectural changes should be based on implementation results whenever possible.

The architecture document is a living specification.

---

# 54. Final Principle

Jupiter is not being built simply to answer questions.

Jupiter is being built to investigate whether an artificial intelligence can **develop** through controlled experience.

The central research question is:

> **Can a small neural system become increasingly capable through continuous, measured, experience-driven learning without depending entirely on massive conventional retraining?**

Everything else in the architecture exists to help answer that question.

---

# 55. Change Log

Required by Architecture 53. Newest first.

## 2026-10-05 - The world is consulted instead of described

```text
Architecture Version: 0.15
Change: CONSULT added to the decision layer; external egress gated
Reason: Architecture 28 was implemented and never reached
Evidence: 43 test files pass, including a new suite of 19
Affected Components: autonomy, external, config, learning loop
```

Architecture 28's external knowledge module was fully built - files, documentation,
an MCP client over HTTP, a retention test - and nothing ever called it. The only
production use was `evaluation.py` calling `census()` to print a report of sources
that no part of the system consulted. Jupiter could describe a world it never
queried.

The connectivity suite reported `55/55 reachable, orphans: none` while this was
true, because `evaluation.py` imports the module. Reachability of a module is not
reachability of its capability, and a green check over that distinction is exactly
the shape of a stale assumption this project keeps testing for.

`CONSULT` is chosen when the corpus has nothing left to learn from and a source is
configured. That is the one branch that used to concede: a dry corpus produced
REST and a reason saying no curated document was available. The world is the
obvious answer there, and it is now asked. It consults only when material is
missing, so an external access is never spent on a question Jupiter could answer
from what it already has.

Bounded, because a world is an unbounded supply of documents. Three consecutive
consults that file nothing and the loop goes back to resting; a consult that filed
material resets the count, so the bound means three fruitless attempts rather than
three passes. Without it the loop never concludes there is nothing coming and never
reaches its own stopping rule.

The query is built from the concepts Jupiter understands least well, and this
replaced a first attempt that built it from source filenames. A curated document is
filed under a content hash, so that query asked the world about a hex string, found
nothing, and looked exactly like a search that worked. The concept layer is the only
place Jupiter records what it actually knows the name of. Descriptions are included
because Architecture 28's retention floor needs two shared terms, and a one-word
query can never satisfy it, so every consult would have been refused for a reason
that had nothing to do with the material it found.

Retained material is filed into the raw corpus for curation and the acceptance gate,
never into the knowledge store. Architecture 28 is explicit that external
information should not automatically become permanent internal knowledge, and
filing rather than storing is what keeps that true.

Two things arrived with it, and the second was already wrong.

The `external` domain had no gate anywhere in the project, and this is why: nothing
called the functions that leave the process, so nothing needed one. Wiring a
capability in without its gate would have created exactly the ungated external
access Architecture 38 exists to prevent, so fetching and probing an agent now
carry the gate, deny by default, and record. A refusal propagates rather than being
folded into an empty result, because `ControlDenied` is a `PermissionError` which is
an `OSError`, which the per-source handler catches - a refusal would otherwise have
been indistinguishable from a world with nothing to say. The census now reports an
unprobed agent as unknown rather than unreachable, since "never asked" and "asked
and it did not answer" are different facts.

`McpClient` never set `kind`, so `census()` raised `AttributeError` for any
configured MCP source. It was unreachable because nothing configured one. This was
found by the new tests rather than reasoned out in advance.

Separately, `act()` raised `ValueError` for LEARN and REVIEW with no runner, while
its own docstring said the action would be "reported as chosen but not performed",
and four tests had encoded the documented behaviour throughout. `step()` calls
`act()`, so a supervisor with no runner attached died on the pass it was recording.
It reports now. This predates the change above and was found because the new tests
ran the file.

What this does not do: Jupiter still cannot perceive anything nobody offered it.
CONSULT asks the world a question Jupiter already had, built from its own weakest
concepts. It has no goals, and it still does not act on anything except itself.

## 2026-10-02 - Growth on a count rather than a diagnosis, and the suite in parallel

```text
Architecture Version: 0.14
```

Growth now happens every hundred accepted cycles when the expansion domain is
granted, rather than only when a document is diagnosed as short of capacity.

The reason is a measurement rather than a preference. The capacity diagnosis
takes its answer from a frustration score, and over this corpus it was wrong
often enough to be useless in practice: seventy-eight sources were blamed on
capacity and no growth was ever proposed, and the one growth that was proposed,
approved and applied made the holdout loss worse, 4.73 to 4.85, and was rolled
back. The rule that produced those outcomes was not broken; it was answering a
question a heuristic cannot answer reliably. Counting accepted cycles is a
question that cannot be answered wrongly.

The cadence decides when to try and never whether to keep. A cadence proposal
arrives unapproved like any other, the loss at authorisation is recorded the same
way, and the acceptance gate still judges the result against that baseline and
rolls back a growth that did not help. Conflating "when to try" with "whether to
keep" is how a model grows without evidence, so the two are kept apart and the
proposal records `cause: cadence` with no frustration index, because a proposal
that did not say so would be indistinguishable from one made because a document
was hard.

Counted in accepted cycles, not cycles run. A rejected cycle is evidence that did
not work, and a hundred of those is not a hundred attempts.

Refused when there is no room, at either ceiling, and the refusal is reported
rather than proposed anyway: a proposal the growth code cannot apply is worse
than none, because it sits in the state file looking like something waiting for
a human. The counter is kept in its own file rather than on the proposal, because
the proposal is rewritten by the first growth it causes and would take the
counter with it.

JUPITER_GROWTH_CADENCE_CYCLES sets the interval and zero turns it off.

Step one runs the suite in parallel. The suite is thirty-six independent files
sharing nothing but the modules they import, on a machine with thirty-two cores,
so one file per process takes nineteen seconds against thirty-five. Splitting by
module rather than by test is deliberate: several of these files build a
tokenizer, a corpus or a model in setUp, and a whole file per process pays for
that setup once and gives a failure the file it belongs to. The verdict is the
one unittest gives - every file ran, and the chain stops on any failure.
JUPITER_TESTS_SEQUENTIAL=1 runs them one at a time.

Two test defects fixed while measuring this, both of which had been writing to
the project rather than to a fixture:

* Four curation fixtures redirected out_dir but not quarantine_dir, so the
  documents their refusal path quarantined landed in the working directory. That
  is why a folder that had been emptied kept coming back. There is now a check
  that fails when a fixture names one without the other.
* test_suite_integrity was overwritten by a rewrite meant to change only its
  docstring, which left thirty lines where a hundred and fifty had been. It was
  rebuilt from its bytecode cache, and the two assertions whose argument order
  had been reversed in the rebuild were corrected against what the cached module
  actually did.

The connectivity check also learned to read an indented invocation. A script run
inside a conditional block is still run by the entry point, and refusing to see
it reports the parallel test runner as dead code.

```text
Changed:   expansion.py          cadence counter, availability, proposal
           config.py             growth_cadence_cycles
           learning_worker.py    counts accepted cycles, proposes when one is owed
           run_tests.py         new: parallel suite runner
           run_jupiter.bat      step one runs it; JUPITER_TESTS_SEQUENTIAL
           tests/               15 cadence tests, 1 production-write guard
Affected:  the expansion rule. The chapter requires expansion to be justified and
           reversible; a cadence is justified by having done the work rather than
           by a diagnosis, and it remains reversible and still judged.
Verified:  1026 tests, 19s in parallel against 35s sequential, 36 files 0 failed.
           Fires at 100 accepted cycles and not at 99; pays once, refires at 200;
           refuses at either ceiling; the count survives a restart and a corrupt
           counter starts over rather than raising.
```

## 2026-10-02 - Curation stopped quarantining the corpus it had already curated

```text
Architecture Version: 0.16
```

Running the entry point re-curated everything, every time, and on this corpus
that is four and a quarter hours before Jupiter learns anything at all.

Measured on six real documents, 402 MB, seventy-seven seconds:

    dialogue analysis  46.7s   60.9%
    scrub              24.4s   31.8%
    read                3.4s    4.4%
    write               1.7s    2.2%
    sha256              0.6s    0.7%

The write is two per cent. The complaint was that every file is rewritten, and
that is true, but rewriting a byte-identical file is not why it is slow: sixty-
one per cent of the time is dialogue analysis applied to text that has not
changed since the last run. Hashing the same bytes costs a two-hundred-and-fifty-
sixth of the work, which is the difference between four hours and a minute.

So a file whose size and modification time are unchanged since the last run is
not read, scrubbed, classified or written again. It was all of that last time
and it produced a file whose name is already the digest of its own content, so
the second run was producing the same bytes for hours of arithmetic.

What still forces a document to be curated again:

* it changed, detected by size or modification time
* the curated file it produced has since been quarantined or deleted, because a
  cache pointing at a document that is nowhere would report the file as done
* the curation rules changed, detected by a version counter on the cache. Without
  that, a change to what counts as contamination or as a conversation would leave
  every document silently curated by two different versions of the pipeline and
  the report describing neither. The counter is its own rather than the
  architecture version, because most architecture changes do not alter what a
  document curates to and invalidating on all of them would throw the cache away
  for edits that changed nothing about the output.

A refused document is refused again rather than remembered as refused. Refusal
is a decision about the rules in force now, and caching it would let a document
stay refused after the rule that refused it changed.

The report says what it skipped, counted as its own field. A run that skipped
everything and reported zero files examined would be indistinguishable from a run
that examined nothing, and the second is a bug while the first is the feature.

An unreadable cache is ignored rather than fatal, which means the next run does
the work and writes a fresh record. A cache that cannot be trusted is one that is
not used.

```text
Changed:   curate_dataset.py   size/mtime cache, skip, version counter,
                                curate() takes a config
           tests/test_curation_cache.py  new, 12 tests
Affected:  none. This is the pipeline doing less of the same thing, and the
           conditions that force the work are enumerated rather than assumed.
Verified:  1008 tests, 37s. A second run over unchanged files skips all of
           them; editing one file re-curates exactly that one; deleting a
           curated file causes it to be written again.
```

## 2026-10-02 - The suite stopped reading the real corpus, and the GPU is now shared

```text
Architecture Version: 0.15
```

Verifying the project took four minutes, and it was one test.

Ninety-seven per cent of the suite's runtime was a single test,
test_the_whole_report_carries_the_category, at one thousand one hundred and
seventy-nine seconds. Its siblings call evaluate_expansion, which reads a
fixture; this one called the whole report. Its setUp redirected the expansion
state and the output directory but not the knowledge store, so evaluation.report
opened the developer's real twenty-five megabyte store and ran retrieval probes
against it. The whole suite is now twenty-eight seconds, and one module of it is
nought point nine.

Nothing about what the test asserts changed. What changed is that a test reads a
fixture rather than production data, which is the difference between a slow suite
and a test pointed at the wrong thing, and the two are indistinguishable from the
outside until you time the individual tests.

The second half is where the work runs. The device was "cuda whenever a GPU
exists", chosen once and held for the life of a run. A worker left unattended
for a day therefore holds whatever it was given in its first second, and if
somebody else is training on that card Jupiter goes and competes with them.

resources.py now decides, from what the machine is actually doing: the GPU when
there is one and it is not already taken, the CPU when it is. Above eighty per
cent of the GPU's memory the card counts as busy and is left alone. The rule
was described as three tiers - GPU, then RAM, then CPU - and the honest answer is
that there are two. Host memory is not somewhere tensor work runs; it is what
the CPU reads from while computing. A busy GPU means the CPU, and a busy CPU is
reported rather than acted on, because there is nowhere further to go. Saying
otherwise would name a device that does not exist.

Every decision is reported with the number that produced it, because a run that
quietly fell back to the CPU is ten times slower and the log is the only place
that can say so. A driver that will not report its memory is treated as busy,
which is the conservative direction: guessing free means fighting somebody else's
job, guessing busy means running a little slower. Utilisation is reported as
unknown rather than guessed, because torch exposes no portable counter for it and
a threshold nobody can measure would look like a decision without being one.

`run_jupiter.bat resources` prints the picture, and JUPITER_DEVICE still wins
over all of it, because a person who knows their machine should not have to argue
with it.

```text
Changed:   tests/test_evaluation.py  knowledge store and report paths in a fixture
           resources.py             new: choose a device from what is in use
           config.py                 device deferred to a decision, threshold set
           learning_worker.py        the reason is printed beside the device
           run_jupiter.bat           resources mode
           tests/test_resources.py   new, 12 tests
Affected:  the hardware philosophy: a shared machine is the case it describes,
           and a long unattended run is the case where it matters. The chapter
           says a 100% GPU workload is not automatically better than a slower
           learning process; this makes that a decision rather than an aside.
Verified:  996 tests, 31s, previously 236s for 984. A busy GPU and an absent GPU
           both fall to the CPU and say why.
```

## 2026-10-02 - Curation stopped quarantining the corpus it had already curated

```text
Architecture Version: 0.14
```

The owner thinned data/raw to one skill at a time, to stop the corpus being held
three times over, and asked what the three copies actually were. They are two:
data/raw, which curation never writes to, and data/curated, which it names by
the SHA-256 of each document's own rendered content. Re-running curation
therefore rewrites the same files rather than accumulating new ones.

The space was somewhere else. out/quarantine held 1,328 files and 18.5 GB, which
is 97 per cent of everything the project stored. It is not a third copy of
anything: it is an archive of documents curation refused, accumulated across
every run since the project began, and it had become the largest directory in
the project while sitting beside a five-megabyte model.

It grew because of how a deliberately partial corpus interacts with the prune.
A curated file is legitimate exactly when its name is a digest this pipeline
produced. Curating a subset of raw therefore declared every other curated file
stale and moved it out, and putting the material back meant re-curating it from
scratch. Measured: curating a single skill quarantined 664 files totalling 18.5
GB, and the next run with the material restored would have quarantined them all
again. Staging a corpus in batches was not merely wasteful, it was a ratchet.

A digest the pipeline has accepted before is now remembered, in
out/curation_digests.json, and a curated file carrying one is retained rather
than quarantined even when its raw source is not present in this run. A digest
nobody has seen is still quarantined, because that is what "not produced by this
pipeline" means, and an unreadable history degrades to quarantining rather than
retaining, so the failure mode is files being moved aside where they can be
restored rather than files being trusted on the word of a note.

All 664 quarantined files were verified before being restored: each is named by
the SHA-256 of its own content, minus the trailing newline curation appends
when it writes, and all 664 matched. A file that could not certify itself was
left where it was. out/quarantine/curated/ is now empty and the files are back
in data/curated. Re-running curation on the same partial raw now reports
quarantined 0 and retained_without_source 664, which is the point.

Quarantine itself moved out of out/ to a top-level quarantine/, set by
JUPITER_QUARANTINE_DIR. It is a safety archive that nothing reads and nothing
trains on, and in a project that curates in batches it grows without bound;
keeping it inside out/ made the working directory the largest thing in the
project. out/ went from 19.0 GB to 0.5 GB.

A curated digest is over the rendered text without the newline the writer adds,
which was not written down anywhere and cost an incorrect verification when
these files were checked. It is now recorded next to the code that computes it.

```text
Changed:   curate_dataset.py   digest history, retain known files, quarantine path
           config.py           quarantine_dir + JUPITER_QUARANTINE_DIR
Affected:  Architecture 25 (data pipeline), 28 (external knowledge is not
           automatically permanent). Neither changes: this makes curation obey
           them, and stops it doing irreversible-feeling work on a corpus that
           is deliberately being staged.
Verified:  984 tests; quarantined 0 and retained_without_source 664 on a
           partial raw; all 664 restored files digest-verified before the move;
           out/ 19.0 GB -> 0.5 GB.
```

## 2026-10-01 - The corpus teaches a skill, and the loop follows the measurement

```text
Architecture Version: 0.13
```

The corpus was replaced with skill-targeted material: 4,545 files and 135.8 GB
holding conversational instruction records, mathematics, source code in 663
distinct languages, reasoning chains, science and prose. Measured, not assumed:
each group was opened and read.

Two things about it were invisible to the pipeline, and both are fixed here.

**The conversation material was being counted as prose.** The records are held
as structured objects, `{"input": [{"role": "user", "content": ...}],
"output": ...}`, and dialogue.py reads speaker labels at the start of a line.
Every role in that corpus lives inside a quoted JSON key, so the reader saw
none of it: 259 MB of the best conversation material reported zero turns, was
rendered as raw JSON, and would have taught the model to emit braces instead of
answers. dialogue.py now reads both layouts, measures the share per record
rather than per line because a record is often one enormous line, and renders a
structured record as the two turns the model is actually spoken in. A reply is
counted as a turn even though no role key names it, which is what stops every
instruction document from reading as a monologue with one speaker.

**Nothing connected evaluation to document choice.** Architecture 34 measures
named categories and Architecture 25 says what may be learned from, but the two
were separate. Selection ranked by how hard a document had been to learn, which
is the right instinct answering the wrong question: material nobody can learn is
not a capability Jupiter lacks. skills.py now reads the evaluation report,
inverts each category into a weakness, maps that onto the folders data/raw was
sorted into, and adds a capped preference to the queue. A measured conversation
of 0.00 moves data/raw/conversation/ up, and the reason recorded with the choice
says so.

The cap is the design. A bonus of at most 0.5 sits below the need signal, so a
weak capability can express a preference and cannot overrule a document that is
actively failing to learn. This is a preference, not a policy: the acceptance
gate still judges, an unlearnable document is still rejected, and Architecture 38
still governs.

data/raw is now sorted into conversation, math, code, reasoning, science and
prose, with duplicate copies moved to data/raw/duplicates. Nothing was deleted
and nothing was copied; every file was moved, and a duplicate keeps the name of
the document it duplicates so the pair stays legible. Curation reads raw
recursively already, so no pipeline change was needed to follow the folders,
but it now skips duplicates/ and quarantined/, which exist precisely to keep
material that has already been accounted for from being hashed again to reach
the verdict it was set aside for.

Two lessons recorded because both were learned the hard way. The organiser
hashed a hundred and thirty-five gigabytes on every run and took over eight
minutes to find twenty-one duplicates; it now reads the first and last 32 KB of
each file and hashes in full only the handful whose fingerprint collides, which
is twenty-eight seconds for the same answer. And grouping by file size, which
looked like an obvious cheap prefilter, was no prefilter at all here: the large
documents were split into equal-sized parts, so a thousand of them shared one
length and were all read anyway.

Filing by name alone left thirty-one documents in the root, and they were not
obscure: the largest set of examination mathematics in the corpus is named by
hash, as are nine megabytes of programming instruction and Jupiter's own
generated conversations. They are now recognised by reading the head of the
file, which is bounded, and the rule that matches a book header is specific
because "Language: English" is in the Gutenberg header of every translated
Asimov title and an earlier version filed a nuclear-energy book as source code.
Material nothing recognises is still reported rather than guessed at.

Tidying is part of filing rather than a separate chore. A thousand and seven
hundred and forty empty directories were the shells left where documents used to
be, and removing them is safe only because a directory holding any file is left
exactly as it is. The skill folders themselves are kept even when empty, since
the folder a duplicate is set aside in would otherwise vanish the moment it was
emptied and be recreated on the next run.

```text
Changed:   dialogue.py            structured records read, rendered, measured
           skills.py              new: evaluation weakness -> folder preference
           selection.py           folder bonus in rank(), named in the reason
           learning_worker.py      passes the bonus, re-read per cycle
           organize_raw.py        new: sort raw into folders, set duplicates aside
           curate_dataset.py      skips duplicates/ and quarantined/
           run_jupiter.bat        organize and skills modes
           tests/                  50 tests
Affected:  Architecture 25 (data sources), 27 (language), 34 (evaluation),
           21 (frustration as a gate, not a ranking). The architecture itself is
           unchanged; this makes the code obey it.
Verified:  984 tests, a reverted growth offered and approved, 4,393 files
           filed across six skill folders, no file deleted, organizer idempotent.
```

The known-limitation note that said the corpus was prose and Jupiter had never
seen a question is withdrawn. It is now untrue, and the 0.11 entry's framing of
the same gap has been overtaken by the corpus rather than by the code.

## 2026-10-01 - A reverted growth can be approved again, and the entry point shows it

```text
Architecture Version: 0.12
```

A growth was approved on 2026-09-29, applied from five layers to six and from
256 to 320 wide, measured, found to make the holdout loss worse (4.7287 to
4.8487) and rolled back. The rollback was correct and the code says so. What it
left behind was a latch.

The rollback restores the weights and the shape record, so the model returns to
exactly the five layers the failed attempt started from. Two guards were keyed on
whether a recorded verdict matched the shape the model has now, and after a
revert that is permanently true. `should_propose_growth` therefore declined to
propose, and `prompt_for_growth` declined to ask. The proposal sat in the state
file looking unanswered when it had in fact been answered by a measurement rather
than by a person, and the only way to apply it was a command the person watching
the run did not know existed.

That is the same defect the 0.8 entry describes for `approve_growth` having no
caller: a control whose application is unreachable is a control that is off. The
0.11 entry widened the escape hatch to a changed shape, which is right, and a
rollback defeats it by making the shape identical again. Neither guard was
wrong about its own case. Both were wrong to treat a measurement the system took
as a decision a person made, which is what Architecture 38 does not permit.

A reverted growth is now offered again, and the prompt names the previous
attempt: the loss at approval, the loss at review, the change, the verdict, and
the checkpoint it was restored from. Approving measures it the same way again, so
a second verdict is a measurement rather than a repetition, and the snapshot
that makes a second failure reversible is taken again rather than inherited.
Declining changes nothing and leaves the proposal on record, exactly as before.

The default chain now prints the expansion state as its own step. The approval
was reachable from `expansion approve-growth` and nothing else, which meant a
person who runs only `run_jupiter.bat` could be told a growth was proposed and
have no way to answer it. A decision someone cannot see is one they cannot make.

The proposal-generation guard is deliberately unchanged. Architecture 23's loop
should not re-ask the same question on the same evidence every cycle, and that
is a statement about what the loop proposes rather than about what a person may
decide. The distinction is the whole fix: the loop still does not re-propose, and
the person is still asked.

```text
Changed:   learning_worker.py     prompt_for_growth: reverted growth re-offered,
                                prior verdict shown
           run_jupiter.bat       expansion status added to the default chain
           tests/test_autonomy.py 1 test rewritten, 1 added
Affected:  Architecture 38 (human control over expansion), 23 (the loop).
           Architecture itself is unchanged; this makes the code obey it.
Verified:  942 tests, 8-step chain exit 0 equivalent, a reverted growth is
           offered and approved from a sandboxed state, and the real model was
           not approved by the test.
```

## 2026-09-28 - The capability gaps, not just the declaration ones

```text
Architecture Version: 0.11
```

The previous entry closed the declaration gaps and the summary said so in a way
that read as though the sweep was finished. It was not, and this is the rest of
it. Three capabilities were named in the document and unreachable in the system,
and the first of them explained the other two.

Co-occurrence was recorded only on an accepted cycle. Over 124 recorded cycles
that was 21, so the concept graph had almost no evidence to derive from, and the
graded relationship types were correct in code and absent from the store. The
gate was in the wrong place: co-occurrence is a fact about what a document
contains, not a claim that learning it worked, and a document that was read and
rejected still says which terms it holds. Recording it whatever the verdict is
the whole fix. What stays gated is everything that asserts Jupiter stands
behind the content, because that is a different claim and a rejected cycle is a
reason not to make one.

A growth that did not help was recorded as a failure and then stayed in the
model forever. The extra layer was in the checkpoint, a smaller model cannot take
a larger checkpoint, and nothing had been saved beforehand, so the honest options
were a full retrain or accepting a model that had been made worse. The growth
now snapshots the published checkpoint at the moment it is approved, and a
verdict of "did not help" restores those weights, rolls the shape record back
with them, and un-approves the growth so it is not applied again. Restoring the
record matters as much as restoring the weights: leaving it describing the grown
model while the model is the smaller one would make the next run build the grown
shape and immediately archive what had just been restored.

"Insufficient data" was one of three frustration causes nothing could produce,
and the evidence was already on disk. The curation report names the files it
refused as too short, duplicated or too large to read, and a source that never
entered the corpus is not a source that was hard to learn, it is a source there
was not enough of. That cause is now produced, and it names the four files
responsible. Two causes remain unevidenceable and stay named as such: nothing
counts how often the tokenizer is the limit, and nothing compares the current
architecture against what the material needs.

```text
Changed:   learning_worker.py  co-occurrence and derivation leave the accept gate
           expansion.py        snapshot at approval, revert a growth that failed
           frustration.py      insufficient data is now evidenceable
           JUPITER_ARCHITECTURE.md
           tests/              9 tests
Verified:  776 tests, 8-step chain exit 0, a harmful growth reverts the
           weights and the shape record
```

## 2026-09-28 - Every declared capability is either reachable or declared unreachable

```text
Architecture Version: 0.10
```

A sweep for doc-versus-code gaps found a consistent failure mode: things that
were declared, named in a tuple or a constant, and never actually produced. The
declarations read as descriptions of the system when they were lists of
intentions, which is the specific dishonesty this project is supposed to avoid.

Ten of the eleven relationship types Architecture 12 names had no code path that
could create one. The concept graph was therefore nine disconnected fragments of
"these two words appeared together", and the document's own limitation note
described a system that could not link ideas. Two are now derived from evidence
the corpus actually contains: SIMILAR_TO from how similarly two terms are
distributed across sources, and PRECEDES from an order the material states
consistently. Both are refusals as much as additions, and most of the tests are
about what is not recorded: a pair seen in one document, an order that reverses,
two terms that share no sources. An edge drawn where the corpus is silent is
worse than a missing one, because it looks like knowledge.

Getting there took three corrections to my own reasoning. The first version
compared a term's source set by testing membership in its own edge, which
answered the same question for both endpoints and made every pair look
identical. The second sorted the term pair, which is exactly what destroyed the
order PRECEDES depends on: a document saying "one two" and one saying "two one"
produced the same edge. Both are the kind of defect that reads as working.

Three of the six Architecture 16 memory destinations were unreachable. All six
now are, by giving each a case the evidence distinguishes: a disputed but
supported fact is a relationship, because a concept would have to pick one of
two claims; something seen often enough that the repetition is the information
is an experience; something supported and disputed at once is held in both.
Separately, the improvement argument was accepted by route_for_source and then
discarded, so a source that taught nothing and one that taught a great deal
routed identically.

Three smaller ones, each a real bug rather than a tidiness matter. ROLLBACK was
missing from the run review, which named four verdicts as literals and so
reported a run that had undone something as a run that had done nothing. The
retrieval docstring claimed the matching method travelled with the result and it
did not, which is Architecture 31's requirement and a score with no visible
basis. The three frustration causes the system cannot evidence are now named as
unevidenced rather than left looking available, and the one cause it produced
every time a new source appeared is now declared.

The evaluation report grew two lines that make the remaining gaps measurable
rather than rhetorical: how many relationship types are really in use, and what
the programming and register layers hold. Both layers wrote on every accepted
cycle and nothing read them back, which makes a layer that is working
indistinguishable from one that is silently failing.

One flaky test is fixed. It asserted a bounded update reduces loss while seeding
only torch, and run_micro_learning draws its batches with random.randrange, so
it failed roughly one run in twenty. A gate that goes red for no reason is
indistinguishable from a gate that has found a regression, which makes it worse
than no gate at all.

```text
Changed:   relationships.py    SIMILAR_TO, PRECEDES, source and order tracking
           memory.py           three destinations reachable, improvement wired
           experiment.py       ROLLBACK counted, verdicts from the authority
           retrieval.py        the matching method travels with the result
           frustration.py      declared and evidenced causes separated
           code_knowledge.py   diagnostics recorded, kinds checked
           evaluation.py       relationships and structured knowledge reported
           concepts.py         maturity checked against the declared set
           mcp_server.py       internal-error code used where it belongs
           tokenizer.py        removed an unread duplicate constant
           train.py            learning rate read back from the optimizer
           tests/              22 tests
Verified:  752 tests, five consecutive green runs, 8-step chain exit 0,
           zero dead constants and zero dead functions
```

## 2026-09-28 - Continuous operation: the decision layer exists

```text
Architecture Version: 0.9
```

Architecture 37 describes a loop: Observe, Receive, Evaluate, Learn, Measure,
Store experience, Rest, Review, Continue. Every stage existed somewhere in the
project and none of them were connected. Frustration measured difficulty,
selection chose a source, growth could widen the model, acceptance judged the
result and rollback undid a bad update, and the worker ran them in an order
someone had written down in a batch file. What was missing was the thing that
decides, and the batch file exiting after eight steps was the honest symptom.

autonomy.py is that thing. It observes, decides, acts and records, and the four
are separate calls rather than one function, which is what lets a decision be
tested without running a training cycle that takes twenty minutes to answer. A
Decision carries the evidence it was made from and a sentence naming it, because
an agent that acts without recording why cannot be audited.

The policy is deliberately defensive before it is ambitious, and the order is
the design: review past work before proposing new work, and rest before
expansion, because a loop that expands while it is not learning is a loop making
the model larger without making it better. A source short of capacity with the
expansion domain granted grows the model. The same shortage without the grant
does not stop the loop: it keeps learning, and its reason names the variable the
person needs to set. Refusing to expand is a normal outcome of this loop rather
than a failure of it.

`run_jupiter.bat autonomous` starts the loop and returns only when a person
stops it. The cycle budget is what it drops, because a budget is a thing someone
sets for a run they intend to watch. The wall clock stays, and so does the idle
limit, because an agent that cannot be stopped is unattended rather than
autonomous. Ctrl+C sets a flag, the next pass sees it, and the lock is released
on the way out.

The optimizer's moments are now persisted beside the resume weights. They were
rebuilt at the start of every run, so AdamW's estimates were thrown away between
cycles and the opening steps of a bounded micro-update moved on a cold estimate
of every gradient, which for an update of a few dozen steps is most of what the
update is made of. Loading checks the shapes itself, because torch's
load_state_dict does not: it matches by position and succeeds against a model
that has grown, leaving most parameters with no moment at all. That is a run
that looks resumed and is not, and it is now refused with a reason.

```text
Changed:   autonomy.py              the decision layer
           learning_worker.py        --continuous, run_continuously
           train.py                  optimizer state save/load
           config.py                 autonomy limits
           run_jupiter.bat           autonomous and status modes
           tests/test_autonomy.py    28 tests
Verified:  733 tests, 8-step chain exit 0, 55 autonomous passes,
           agent grew the model unprompted when granted and explained
           itself when not
```

## 2026-09-28 - Neurons can grow, and a human can authorise it

```text
Architecture Version: 0.8
```

An audit of all 55 chapters against the code found two defects that made the
system look finished and not be, both of them dead ends an operator could hit,
and neither of them reachable by a test.

n_embd had been fixed at 256 for the entire life of the project. Growth could
add a layer, so Jupiter got deeper and never wider, and Architecture 22's "more
neurons" and "more hidden dimensions" were unavailable. The reason was recorded
and correct: the attention projection fuses Q, K and V into one tensor, so
copying a prefix of its rows would write the learned Q weights into the K slot.
The model would still run and still produce numbers, so nothing downstream could
detect it. The reasoning was right about the prefix and wrong about the
conclusion, because that tensor is three stacked blocks rather than one, and each
can be copied as a block. The MLP up-projection stacks four the same way and is
handled the same way. A growth is now n_layer +1 and n_embd +64, both
configurable, both judged by the acceptance gate that every other change answers
to.

The second defect was the control that authorises all of this. The worker prints
"then approve it", and approve_growth had no caller anywhere outside the test
suite, so the instruction was unactionable from the entry point and a Python
prompt was the only way to comply with it. A control whose application is
unreachable is a control that is off. `run_jupiter.bat expansion` reaches it
now: status, approve-growth, review-growth, grant and rollback, each of them
through the same gate the library uses, so running it from a terminal is exactly
as gated as calling the function.

The audit also found the MCP endpoint printed nothing at all. serve() called
serve_forever() internally, so main() blocked before reaching the banner that
prints the address: the server worked and the operator was told nothing, leaving
no way to connect. Zero bytes became 305. Python block-buffers stdout when it is
not a terminal, so the batch file also passes -u, without which a working server
still looks silent.

Width growth is verified group by group rather than by shape: every learned
value must land in the slot it was learned for, checked across the fused QKV
projection, the four stacked MLP groups, the embeddings, the head, the layer
norms and the square projections. A widened model is not required to reproduce
the narrow one exactly, because it has more degrees of freedom, and the test
says so; what it must never do is produce a number instead of an error.

```text
Changed:   growth.py          block-wise width growth, previously refused
           expansion.py        width proposal, current_embd, operator CLI
           mcp_server.py       banner before serving, previously unreachable
           run_jupiter.bat     expansion mode, -u for the server
           tests/test_evaluation.py   14 tests
Verified:  702 tests, 8-step chain exit 0, MCP reachable, 305 bytes
```

## 2026-09-27 - Evaluation reports how much, not what

```text
Architecture Version: 0.7
```

The evaluation report measured everything Architecture 34 asks for and then
printed the definition of each measurement instead of the measurement. A reader
asking how well Jupiter was doing got a page of sentences explaining what a
holdout loss was, and the numbers were either absent or expressed in a form that
could not be compared between categories.

Two problems, and the second was the one that mattered.

A raw difference in loss is not comparable. -1.12 nats off a starting loss of
7.5 and -1.12 nats off a starting loss of 100 are opposite stories told with the
same number, and every loss category in the report was quoting exactly that.
Each now reports the share of the starting loss the bounded update removed, as a
percentage against the value it started from, with the before and after losses
kept beside it so the percentage can be checked rather than trusted.

A rate is not a change. Reporting "79% of cycles improved" with a plus sign
reads as "it went up by 79%", which is a different claim, so rates are printed
unsigned and changes signed. Loss improvements stay negative because the loss
went down: flipping the sign for tidiness would break the comparison against
every other loss figure in the same report. An unmeasured category prints a dash
rather than 0.0%, because a measured zero is a result and an absent measurement
is the absence of one.

The console now leads with the numbers under "HOW MUCH, NOT WHAT" and keeps the
definitions below under "WHAT EACH NUMBER MEANS", because both are wanted and
they were fighting for the same space. The arithmetic is verified independently
of the display: the reported mean improvement, improvement rate and before/after
losses reproduce exactly from the experiment records.

On the last full run the Expansion line read "did not help" where the previous
run had read "helped", because the grown model's holdout loss went up. That is
the gate working, not a regression in it.

```text
Changed:   evaluation.py        improvement/percent/rate, percentage metrics,
                               headline-first console
           tests/test_evaluation.py   16 tests
Verified:  687 tests both ways, full 8-step chain exit 0, 0 errors
```

## 2026-09-27 - Curation reads a bounded document, and the suite ignores your shell

```text
Architecture Version: 0.7
```

Running the entry point rather than only the tests found two defects that a
green suite had been hiding, which is the argument for running it.

The first ended the run. data/raw held a 77 GB document and a 49 GB one, 124 GB
in total, and curation reads a whole file to judge it. The read raised a
MemoryError from inside a codec, so the traceback named neither the file nor its
size nor anything the owner could act on. There was a size check in the module
already, but it guarded the curated output, not the raw input, so the read that
actually mattered was the one thing unbounded.

Curation now takes the size from the filesystem before reading and refuses
anything over max_raw_bytes, which is configuration rather than a constant
because raising it is a decision about the machine and belongs to whoever runs
it. A refused file is counted and named with its size and the limit, exactly as
an unparseable one already was, and data/raw is left untouched: refusing to read
a document is not permission to remove it. All 124 GB are still on disk.

The second was mine, and it is the reason this entry exists. Verifying the chain
with JUPITER_CONTROL_GRANTS exported made five control tests fail while the other
664 passed, and the shape of that failure is a broken gate rather than a broken
test. The gate was not broken; it was granted, by the environment, and those
tests assert that it refuses by default. A control test that reads the
developer's own shell is not testing the control.

Every test now runs with the grant variable cleared and the previous value put
back afterwards, so the suite answers the same question however it was invoked.
The suite is green both ways now, which is the property worth having: 671 tests,
identical with and without the variable set.

```text
Changed:   config.py         max_raw_bytes
           curate_dataset.py refuse an oversized raw file before reading it
           control.py        document the ambient-environment hazard
           tests/support.py  AmbientControlIsolation
           tests/test_curation.py   6 tests
           tests/test_control.py    isolation tests, 2 control classes
Verified:  full 8-step chain, exit 0, 0 errors, 671 tests both ways
```

## 2026-09-27 - The status table and the code agree again

```text
Architecture Version: 0.7
```

Three places said something the code had overtaken. Each was found by checking a
claim against the source rather than by reading the document, and each is now
corrected at the level where the falsehood was.

MCP integration was listed as FUTURE while mcp_server.py existed, exposed five
working tools, bound to loopback and sat behind the external-access control. The
changelog in this same file already described the endpoint as delivered; only the
status table had not been updated. It is EXPERIMENTAL, not IMPLEMENTED, because
it has been run but not exercised across enough sessions to call it verified.

Dynamic expansion was listed as EXPERIMENTAL CONCEPT, a status the document
defines as not an implementation at all. Expansion has two implemented halves,
one of which changes the model's shape. It is EXPERIMENTAL for the same reason
the other experimental features are: implemented and demonstrated, not yet
validated across enough runs to be architecture.

Learning frustration carried the same status and was corrected for the same
reason. It is measured per concept, reported in the evaluation report, and it is
what decides whether a source short of capacity gets a growth proposal, so it is
a working part of the loop rather than a concept.

Expansion was the last Architecture 34 category reported as unmeasured, on the
ground that dynamic expansion did not exist to measure. It does now, so the
report measures it: the number that matters is the holdout loss change across
the growth decision, not how many growths were made, because a growth that counts
says only that the feature was used. A growth authorised without a baseline is
reported as unjudged rather than scored, which is the same rule the category list
follows and the reason that list is still worth keeping as a test.

All nine categories Architecture 34 names are now measured. The test that
enforced the old claim was rewritten rather than deleted, so a category added
without a measurement still fails.

```text
Changed:   evaluation.py        measure Expansion; docstring corrected
           tests/test_evaluation.py    7 tests, stale assertion rewritten
           JUPITER_ARCHITECTURE.md     status table and two prose sections
Verified:  661 tests, 0 failures, 0 pyflakes warnings, 37/37 modules reachable
```

## 2026-09-27 - A growth is now applied, recorded, and judged

```text
Architecture Version: 0.7
```

Capacity growth existed as four functions and nothing called them. A proposal was
written to disk, an approval was recorded, and no model ever grew, so Architecture
22's "the model can expand" was true of the code and false of the system.

The path is now closed end to end. build_model reads the saved architecture
before growing, so growths accumulate rather than each proposing the same layer
count from the seed configuration. The architecture record is written from the
model that was actually built rather than from the configuration that asked for
it, which was the defect that would have made a growth self-defeating: the config
still says four layers after the model has five, so recording the config would
have recorded a shape the weights do not have, and the next run would have read
that record, built the smaller model, and archived the checkpoint it had just
produced. The worker proposes a growth when frustration blames capacity, and
judges it at the end of the run against the holdout loss recorded when a human
authorised it.

Two gates, deliberately. Approving the growth is an expansion change; recording
the new architecture of a model in use is an architecture change. They are
separate Architecture 38 domains and both ask, because a growth that only needed
one signature would mean the record could be rewritten to match whatever the
model happened to become.

A growth that is judged not to have helped is recorded as such and is not
repeated. Reverting the layer count is not a reload: a smaller model cannot take
a larger checkpoint, so undoing one is a fresh training run, and the report says
so rather than implying a rollback that did not happen.

```text
Changed:   config.py       record the model that was built, not the config
           train.py        apply an approved growth on top of the saved record
           expansion.py    read the current layer count from the record
           learning_worker.py   propose and judge a growth
           tests/test_expansion.py   16 tests
Verified:  654 tests, 0 failures, 0 pyflakes warnings, 37/37 modules reachable
```

## 2026-09-27 - MCP endpoint, so an agent can reach Jupiter on a port

```text
Architecture Version: 0.7

Change:
  mcp_server.py exposes Jupiter over HTTP with JSON-RPC 2.0, which is what "an
  agent has an IP address and a port" means. It speaks initialize, tools/list,
  tools/call and ping, and exposes five tools: status, retrieve, capabilities,
  evaluate and learn_document. Reached as `run_jupiter.bat mcp`. New config:
  mcp_host, mcp_port. Tests drive it over a real socket rather than calling
  handlers, because the thing a client meets is a port.

Reason:
  Architecture 29 lists what MCP should provide and says two things that shape
  the design: Jupiter must remain capable of operating without MCP during core
  model testing, and MCP is an integration layer rather than a requirement of
  the neural model. So this is a separate module the default chain does not run,
  and the model, learning loop and tokenizer are untouched.

  The transport is the standard library. Architecture 22 asks whether a
  dependency is needed before one is added, and http.server, json and
  socketserver already provide the whole protocol, so a framework would have
  been a large dependency for no capability.

Evidence:
  Sixteen tests. Against a live socket: health reports the five tools,
  initialize negotiates 2025-06-18, tools/list returns schemas, a tool call
  returns a result, an unknown tool is -32602, an unknown method is -32601, and
  a tool that raises is reported as isError without the server going down.

Three defaults are deliberate and all are about who may reach the model.

  It binds to loopback, not to every interface. A model with a learned corpus
  and a knowledge store is not something to put on a network because a port was
  configured; widening mcp_host has to be a deliberate act.

  It is refused without the external-access grant, because reaching a model
  over a socket is external tool access whatever the caller calls itself.

  A write from a network caller is treated as a persistent memory change and
  needs the memory grant, not merely the external one. A refused write leaves
  nothing behind.

Affected Components:
  MCP integration (Architecture 29), safety and control (Architecture 38),
  configuration, the single entry point (Architecture 40).
```

## 2026-09-27 - Assertions, so two sources can finally be found to disagree

```text
Architecture Version: 0.7

Change:
  assertions.py records what a source claims: a subject, a relation, an object,
  a polarity and a source. An accepted document is harvested in the learning
  cycle, so the layer fills from real material. Two sources asserting the same
  proposition corroborate each other; two sources asserting opposite ones
  contradict, and that opens a CONTRADICTS edge in both directions. A source
  denying a claim it previously asserted is recorded as a context difference and
  opens no edge. A second source on the denying side counts as corroborating the
  denial rather than as more conflict.

Reason:
  Step 6 of Architecture 14 was built and was correct, and it never fired.
  Across twelve real cycles every promoted term ended up both supporting and
  contradicting under the same document name, because a document accepted in one
  cycle and rejected in another appears on both lists, so the decision read it
  as contextual every time. Nothing was wrong with the decision; the input was.
  A contradiction was a bare count on a term, and term co-occurrence records
  that two words appeared in the same document, which cannot say that one
  source denies what another asserts. A rejected cycle is evidence that learning
  a document damaged the reserved document, not a disagreement about anything.

  The missing thing was not a better decision but a way to record a claim, and
  polarity is what makes a disagreement expressible at all. The patterns are
  narrow and declared a heuristic for the same reason the concept extractor is:
  an assertion layer that invents claims would fabricate disagreements between
  real sources, which is worse than having none.

Evidence:
  Sixteen tests. Two sources denying and asserting "the river is safe" produce
  two CONTRADICTS edges, doc-a to doc-b and doc-b to doc-a, and never an edge
  from a source to itself. The same source asserting then denying the same
  claim produces no edge, which is Architecture 14's context difference. Three
  sources produce one corroboration and one corroborated denial rather than
  three separate conflicts.

Affected Components:
  Knowledge conflict (Architecture 14), concepts and relationships
  (Architecture 10, 12), learning controller (Architecture 37).
```

## 2026-09-27 - Dynamic expansion, built for the one mechanism that is safe

```text
Architecture Version: 0.7

Change:
  expansion.py implements Architecture 23's loop from the point where it was
  missing. The first four steps already worked: frustration measures the
  difficulty, the band says whether it repeated, and likely_cause separates "not
  enough data" from "cannot represent this with the current architecture". What
  was absent was everything after the question - propose, experiment, evaluate,
  keep or roll back. propose() acts on one cause only, insufficient model
  capacity, and returns nothing for the other six. A grant is refused unless a
  human authorises the expansion domain, through the control layer added
  earlier. A grant records the holdout loss at the moment it was made, so the
  cycles that follow are comparable rather than a matter of opinion, and
  rollback returns the extraction limit to what the configuration said.

  learning_worker records proposals on the first cycle of a run and applies
  granted expansions when it extracts concepts, so an authorised expansion
  changes behaviour rather than merely being written down.

Reason:
  Architecture 22 lists seven expansion mechanisms. Six of them - more neurons,
  more hidden dimensions, more layers, specialised modules, expert modules,
  auxiliary networks - are shape changes, and the shape guard archives any
  checkpoint whose shape disagrees, so each would be a full retrain rather than
  a growth. Memory expansion is the seventh, changes no parameters, leaves the
  checkpoint valid, and is genuinely reversible, which is what lets Architecture
  23's "measurable and reversible" be honoured without weakening the guard that
  stops silent weight loss.

  The reasoning for memory rather than weights is Jupiter's own. Its concepts
  live in an external store rather than in the parameters, by Architecture 10,
  so for this system the capacity to represent more of a document is the
  capacity to name more of it.

Evidence:
  Eighteen tests, most of them about refusals, because the properties that
  matter are the ones stopping an expansion happening for the wrong reason or
  being impossible to undo. Verified end to end: a source short of capacity is
  proposed 5 to 10 concepts per cycle, a source failing for any other reason is
  not proposed at all, a grant without authority is refused and leaves the limit
  at 5, an authorised grant raises it to 10, the effect is reported as -1.5
  when the holdout loss improves and +1.5 when it worsens, and rollback returns
  the limit to 5.

Not changed:
  The shape guard, and therefore the six parameter-changing mechanisms. They
  remain unbuilt and are not counted as done.

Affected Components:
  Dynamic expansion (Architecture 22, 23), learning frustration (Architecture
  21), safety and control (Architecture 38), learning controller
  (Architecture 37).
```

## 2026-09-27 - Long-running tests, the last two version domains, and a live rollback

```text
Architecture Version: 0.7

Change:
  Four gaps closed.

  tests/test_long_running.py adds the test level Architecture 41 names and the
  project did not have. It runs the real controller over several cycles on a
  tiny model and a tiny corpus, and covers the whole continuous path: the
  need-scored queue, acceptance and regression, resume persistence, the visit
  history, a rejected cycle restoring the live model, knowledge surviving a
  restart, weights staying finite, and a second run resuming rather than
  restarting.

  A REJECT now restores the live model immediately, to the last accepted state
  or, when a run has earned none, to the state the run started from. REVIEW and
  DEFER do not restore, because Architecture 20 is explicit that REVIEW does not
  roll an update back and a dead-band move is not a measurement of damage.

  The Architecture 16 routing call now supplies eight of its nine factors.
  improvement and the reserved-document gain were computed a few lines above
  the call and simply not handed over; the cycle duration and the source's age
  were in scope and unused.

  learning_system_version and memory_schema_version added, completing the six
  version domains Architecture 43 names, and recorded beside the other four in
  every experiment row.

Reason:
  No test in the suite invoked run_worker at all, so everything written for the
  continuous path had never been run end to end. Architecture 16's own
  docstring claimed its factors were all observable from what the system
  records; that was true of the function and false of the caller. Architecture
  43 asks that the versions not be assumed to advance together, and with two
  domains missing a knowledge store written by a different learning system was
  indistinguishable from one this run produced.

Evidence:
  The long-running suite is 15 tests and runs the real controller rather than
  calling into it, which is why it is the slowest thing in the suite by an
  order of magnitude. 14 of the 15 passed on their first run; the one that did
  not asserted a status vocabulary that does not exist, and was corrected to
  assert what decide_stop can actually return.

  A rejected cycle previously left the damaged weights live until the end of
  the run, so every following cycle measured itself against damage that had
  already been dealt with. The published checkpoint was always safe, because it
  is written only on ACCEPT, so the gap was in the live model rather than in
  persistence.

Affected Components:
  Testing (Architecture 41), routing (Architecture 16), versioning
  (Architecture 43), rollback (Architecture 32), memory versus weights
  (Architecture 15).
```

## 2026-09-27 - Need-scored selection, and material Jupiter writes for itself

```text
Architecture Version: 0.7

Change:
  Two additions, both closing origins and mechanisms Architecture already named
  and never built.

  selection.py ranks the learning corpus by need instead of walking it in the
  order sorted() returns it. Need combines the Architecture 21 frustration band,
  the Architecture 13 confidence of a source, and an age term that reaches its
  maximum after one full rotation. learning_worker records why each document was
  chosen in worker_cycle_start telemetry, and persists the visit history per
  cycle so an interrupted run does not forget it.

  synthesize.py writes material Jupiter produces itself, into
  data/raw/generated/, as ordinary raw text. It is gated on the cause of the
  frustration rather than on its size, and it fires only for the two of
  Architecture 21's seven causes that more material can answer. frustration
  gained for_sources(), a sources-only sweep, so the loop does not walk every
  concept to get the half it acts on.

  New config: selection_state, synthesis_dir, synthesis_per_run,
  synthesis_max_new_tokens, synthesis_seed_chars.

Reason:
  Architecture 25 lists generated examples and Jupiter's own experiences among
  the origins data may come from. Neither existed, which left the corpus closed:
  every training byte had been placed in data/raw by a person, so Jupiter could
  only learn material it had already been handed and could never meet a case
  nobody thought to include.

  Selection was the matching half. The worker took the next document in
  alphabetical order, which is a directory listing rather than a curriculum, and
  nothing in the project asked which material Jupiter was worst at. Now the
  frustration signal has a consumer, which is what Architecture 21 meant by it
  feeding into architectural analysis.

Three properties are deliberate and load-bearing:

  Generated text does not bypass curation. It is raw data like any other, so it
  is parsed, cleaned, classified and deduplicated, and refused as contaminated
  or too short on exactly the same terms. A self-feeding loop is the last place
  that should have a private path.

  The gate is on the cause, not the score. A representation the model cannot
  compress does not become compressible by being shown more of it, and
  Architecture 23's caution about acting on a measurement applies to the cheap
  remedy as much as to expansion. Generating into a capacity or representation
  failure would reinforce the failure rather than fix it.

  Whether the generated material helped is not a separate claim. It becomes a
  normal candidate, so the same acceptance gate, the same reserved regression
  document and the same pre/post measurement apply to it, and that verdict is
  the measurement.

Evidence:
  Against the live corpus of 12 documents and 5 recorded frustration rows, the
  queue now joins 4 judged sources and visits the two HIGH-band ones roughly
  twice as often as the LOW ones, while still visiting everything: 8 cycles
  produced 223d92b4, 5865f7ca, 50040dec, 223d92b4, 5865f7ca, 613bef68,
  223d92b4, 5865f7ca. Under the old round-robin the first three cycles would
  have been 223d92b4, 50040dec, 5865f7ca in that order regardless of anything
  measured.

  Synthesis fired 0 times on the live state, which is the gate working rather
  than the generator failing: both HIGH sources fail for "insufficient model
  capacity" and the rest for "conflicting knowledge", and neither is answered by
  writing more text. Forced open with an "inadequate context" measurement, it
  produced a 3983-character document from the real published checkpoint, which
  the unmodified curation accepted into the corpus (12 documents to 13) with no
  quarantine.

Two defects were found and fixed during this work, both in the new code:

  Sources are recorded under the bare filename while candidates arrive as full
  paths, so the join matched nothing and the queue silently degraded to
  alphabetical order while reporting a healthy need score. selection.key_for
  normalises this in one place, with tests.

  The seed was initially the whole source document, producing a 12.4 million
  character file that was almost entirely a copy of a document already in the
  corpus, which dedup cannot catch because the appended continuation changes
  the digest, and which data/raw would then hold forever because raw is never
  pruned. The seed is now windowed to synthesis_seed_chars.

Affected Components:
  Data pipeline (Architecture 25), learning frustration (Architecture 21),
  confidence (Architecture 13), learning controller (Architecture 37),
  observability (Architecture 35).
```

## 2026-09-27 - Evaluation writes its report instead of printing three thousand lines

```text
Architecture Version: 0.6

Change:
  evaluation.py now writes the complete report to data/evaluation_report.json
  and prints a bounded summary: which categories are covered, a headline note
  per measured category, and the frustration census counts. A --full flag still
  prints the entire report. main() takes an optional config and argv, matching
  prepare_dataset.inspect. New config field evaluation_report.

Reason:
  The report is measured per concept and the frustration census reaches several
  hundred rows once a corpus has been learned. Against the live knowledge store
  the report was 3013 lines, of which 2957 were the frustration section alone,
  and 129 concepts were above the high band. The entry point printed all of it
  in one burst, which on a Windows console is enough to trip the terminal's
  selection mode: the display pauses and a run that had already finished looks
  like one that has hung. Reporting a measurement as a wall of text is the same
  failure as reporting it as a wrong number. Architecture 35 asks for the run to
  be observable while it happens, and three thousand lines is not observable.

Evidence:
  Console output measured before and after on the live store: 3026 lines before,
  22 after, with --full still producing 3038. A test asserts that the summary is
  the same length for a one-concept store and a 165-concept one, so console size
  is bounded by the number of categories rather than by how much has been
  learned.

Not changed:
  Every metric, the category list, and which categories are measured. Only where
  the detail is written and how much reaches the console.

Affected Components:
  Evaluation (Architecture 34), observability (Architecture 35), configuration.
```

## 2026-09-27 - Review counted as evidence, so the review floor stops latching

```text
Architecture Version: 0.6

Change:
  A REVIEW verdict now raises a source's confidence by a quarter of the accept
  gain and counts as one piece of evidence, where before it left confidence
  exactly as it was. DEFER still records nothing. No verdict ordering was
  changed, the review floor itself is unchanged, and REJECT and ROLLBACK still
  subtract.

Reason:
  Architecture 13 states that the system "can later update these values as
  additional evidence becomes available", and its own worked example pairs a
  confidence with 147 observations and 2 contradictions, which models confidence
  as evidence accumulated against contradictions found. The code contradicted
  both. The review floor classified a sub-floor source REVIEW, and REVIEW was
  confidence-neutral, so the only verdict that could raise confidence, ACCEPT,
  was precisely the one the floor withheld. Confidence was frozen for the rest
  of the run. It had stopped being a measurement of evidence and become a latch,
  and a source that had demonstrably stopped damaging the reserved document was
  denied the chance to be trusted again.

Evidence:
  The confidence floor is not prescribed anywhere in this document. It appears
  only in the changelog entry below, recorded as a deliberate ordering choice:
  measured forgetting is checked before the floor, and a low floor means the
  source has not earned trust. That rationale is untouched. What it did not
  provide was any way for a source to earn trust while below the floor, which
  is what left the two rules mutually exclusive.

  Measured on the live corpus: holdout loss fell by 1.5 to 4.3 against a defer
  band of 0.0001, and six consecutive cycles were classified REVIEW because
  confidence sat at 0.258 against a floor of 0.35. No ACCEPT followed, so no
  published checkpoint was written and sample.py could not start. The reserved
  document was improving through those cycles, from about 8.4 to about 4.9, so
  the updates were not the damaging kind the floor exists to catch. Two
  confidence states were at 0.000, having never earned a single ACCEPT.

Affected Components:
  Confidence (Architecture 13), update acceptance (Architecture 20), and the
  per-source confidence state written to out/confidence.json.
```

## 2026-09-27 - Resume state separated from the published checkpoint

```text
Architecture Version: 0.6

Change:
  Added out/Jupiter_resume.safetensors, written after every cycle and after
  the end-of-run rollback, and resumed from at the start of every run. It is
  tried before the published checkpoint and falls back to it when unusable.
  out/Jupiter_latest.safetensors remains written only on ACCEPT and remains
  what the conversation loads. Each file now records a "role" of published or
  resume in its SafeTensors metadata. JUPITER_ARCHITECTURE.md:944 and the
  Architecture 30 persistence flow are unaffected in intent: the published
  model is still the last judged one.

Reason:
  Architecture 20 requires an update to be judged before it is kept, and that
  was applied to both files. The consequence was that a run which never earned
  an ACCEPT persisted no weights at all, so the next run found nothing to
  resume from, printed "Starting Jupiter from random initialization" and
  measured the same starting loss forever. A verdict answers whether a model
  is ready to be shown; it does not answer whether the next run may build on
  that state, and Architecture 30 requires learning to be resumable.

Evidence:
  Over 31 recorded cycles the last 13 had holdout_loss_before between 8.51
  and 8.71 against ln(5256) = 8.5671, the loss of a randomly initialized
  model of that vocabulary. No out/Jupiter_latest.safetensors existed, and the
  final 16 cycles produced zero ACCEPTs, so nothing was ever written to
  resume from. With the change, four consecutive single-cycle runs in
  isolation measured 5.5650 -> 5.4693 -> 5.3210 -> 5.1833 -> 5.0673, each run
  resuming at the previous run's finishing value, while the published
  checkpoint stayed pinned at the last ACCEPTed cycle exactly as intended.

Affected Components:
  Persistence (Architecture 30, 32), update acceptance (Architecture 20),
  learning controller (Architecture 37), SafeTensors metadata.
```

## 2026-09-27 - Context window, learned tokenizer, dialogue, evaluation

```text
Architecture Version: 0.4

Change:
  Raised the context window from 256 to 512 tokens. Added a learned
  byte-level BPE tokenizer alongside the byte tokenizer, with the tokenizer
  measurement Architecture 8 requires. Made the architecture record versioned
  and gave it a migration guard that archives a checkpoint whose shape no
  longer matches instead of failing with a tensor size error.

  Added dialogue support: a turn is a speaker label followed by text, curation
  classifies prose against conversation and counts chat-template
  contamination, and a conversation now speaks in that layout.

  Added an evaluation surface covering five of the eight Architecture 34
  categories, naming the three it cannot measure.

Reason:
  Architecture 8 warns that a byte-level tokenizer at one token per byte makes
  the context window far smaller than it looks, and that the tokenizer decision
  should be made before long-term development. At 256 tokens Jupiter could not
  hold a question, some retrieved knowledge and an answer.

  Architecture 25 lists conversations as a data origin and none existed, so the
  model had no way to learn that a turn exists. Architecture 43 requires
  versions to be tracked separately, and nothing carried a version.

  Architecture 34 asks for eight evaluation categories and loss on held-out
  text is not eight categories.

Evidence:
  208 tests pass. The connectivity test reports 25 of 25 modules reachable
  from run_jupiter.bat with no orphan.

  Parameter count moved from 3,290,624 to 3,356,160, which is exactly the
  65,536 extra position embeddings that 256 additional positions at 256
  dimensions require.

  Tokenizer measured on the current corpus, as Architecture 8 requires: 500
  merges gives 1.639 bytes per token, 2000 gives 2.077, and 5000 gives 2.384
  on the full corpus for under a second of training. Beyond 5000 the corpus
  runs out of pairs. The default is 5000 because it is where the measurement
  stops paying, not because it is a round number.

  Switching tokenizers end to end was verified: the model is rebuilt for the
  new vocabulary, the checkpoint shape disagrees, the guard archives it with a
  note explaining which field changed, and training restarts cleanly on the
  first attempt. An earlier version of this change let the saved record
  override the vocabulary, which built a 256 entry embedding table for a
  tokenizer emitting ids above 255 and failed inside a forward pass instead
  of at the point of configuration. The vocabulary now belongs to the
  tokenizer.

  Curation of the current corpus reports 5 conversational and 7 prose
  documents, and 3 documents containing chat-template markers, one of them
  with 3026. That is contamination in the training data, not dialogue.

  Evaluation on the current state reports a retrieval hit rate of 1.0 over 10
  probes, 2 recorded cycles that both improved holdout loss, and 0 measured
  regressions on the reserved document.

Affected Components:
  Tokenizer (new BPE implementation, measurement, migration), Persistence
  (versioned architecture record, migration guard), Memory (context assembly),
  Data pipeline (dialogue classification, contamination counting), Evaluation
  (new), Architecture 52 status table, Architecture 34 coverage.

Not changed:
  The learning algorithm, acceptance verdicts, regression detection, the model
  family, and every explicit control in Architecture 38. No retraining was
  performed; the checkpoint on disk was trained before this change and is
  archived under out/archive/.
```

## 2026-09-27 - Closing the measurement and honesty gaps

```text
Architecture Version: 0.5

Change:
  train.py no longer runs its own cycle. It delegates to the learning
  controller, so a single cycle is judged by Architecture 20 and checked for
  forgetting by Architecture 33 like every other cycle. modal_train.py, which
  ran train.py directly on remote hardware, now runs the controller's --once
  mode. This removed the last path that saved a checkpoint it had not judged.

  Cycles now record perplexity, next-byte accuracy, the generalization gap
  between seen and held-out loss, and the L2 magnitude of the weight change,
  which Architecture 19 and 35 both ask for and neither of which existed. The
  resource sample now includes disk usage.

  The configured precision is applied. bfloat16 narrows the forward pass on
  CUDA while the parameters stay in float32 for the optimizer. It was
  previously selected and then never used, so a capable machine trained in
  float32 without saying so.

  Working memory now has a lifecycle: the document being learned is held for
  the duration of the cycle and cleared afterwards, because Architecture 15
  makes it task-scoped. The Architecture 16 routing decision is now taken each
  cycle and recorded with its reasons, using the three of the nine factors
  that can be measured today.

  Acceptance now checks measured forgetting before the confidence floor. A
  regression means the update was measured and it damaged earlier learning,
  which is a judgement; a low confidence floor means the source has not earned
  trust, which is a request to observe again. The previous order reported
  REVIEW when both held, hiding a measured regression behind a request for more
  data.

  Removed code nothing called: the nanoGPT compatibility alias, three store
  methods with no caller, an unreferenced constant, three configuration knobs
  that no code read, and five unused imports.

Not changed:
  Model architecture, tokenizer, the learning algorithm, the concept and
  relationship model, and every explicit control in Architecture 38.

Reason:
  Architecture 35 lists update magnitude and storage usage among the things the
  system should expose, and neither existed. Architecture 19 lists accuracy
  and perplexity, and neither existed. Architecture 40 forbids dead code.

  The dead code was removed rather than wired because nothing in the
  architecture requires it. allow_growth and growth_min_improvement were policy
  limits for dynamic expansion, which is unimplemented, so they constrained
  nothing; worker_min_improvement duplicated defer_band. Architecture 38 asks
  for human control over model expansion, and a flag that gates nothing is not a
  control. That control will be added when expansion exists.

Evidence:
  228 tests pass, up from 208. No module has an unused import.

  Running python train.py in a sandbox now produces a governed cycle: a
  verdict, a reserved-document measurement, and an experiment record, which it
  did not before.

Affected Components:
  Learning controller (governs the single-cycle path), Update acceptance
  (verdict ordering), Observability (perplexity, accuracy, generalization,
  update magnitude, storage, precision), Memory (working memory lifecycle,
  Architecture 16 routing), Configuration, Architecture 52 status table, README.
```

## 2026-09-27 - Single entry point, knowledge retrieval, co-occurrence

```text
Architecture Version: 0.3

Change:
  run_jupiter.bat became the single entry point and now drives every module
  in the project. It verifies the test suite, curates, inspects, learns,
  shows recent telemetry, checks remote-training availability, and then opens
  a conversation, with watch and remote reachable as modes.

  Added a test that resolves the import graph from the batch file and fails
  if any module cannot be reached from it, so the project cannot grow a part
  the entry point does not drive.

  Added retrieval. The knowledge layer had been write-only: the learning
  controller accumulated concepts, experiences and memory, and nothing read
  them back, so Jupiter recorded knowledge it could never use. An accepted
  cycle now also copies its source text into external memory, and a
  conversation retrieves against the store and places what it finds ahead of
  the question, keeping knowledge and dropping the oldest conversation.

  Added the CO_OCCURS relationship type, which is the only relationship
  Jupiter derives for itself, and wired it into the learning cycle.

  Made the telemetry follower and the remote GPU launcher reachable from the
  entry point without either blocking the chain or costing money. The
  follower now tracks a byte offset instead of re-reading a file that grows
  without bound.

Reason:
  Architecture 40 asks for one loop -- initialize, run, learn, measure, save,
  reload, verify -- driven from one place. In practice the entry point called
  train.py, which performs one cycle, saves the checkpoint unconditionally,
  and never runs update acceptance or regression detection. The whole
  learning system described in Architecture 17 through 37 was unreachable
  from the command the project is actually run with.

  Architecture 9 says knowledge should not all live in weights, and
  Architecture 16 says information should be able to remain external. Neither
  is observable unless something reads the store back.

  Architecture 12's relationship types had no honest way to record the one
  thing Jupiter can measure about two terms, which is that the same source
  surfaced both.

Evidence:
  132 tests pass. The connectivity test reports 22 of 22 modules reachable
  from run_jupiter.bat with no orphan and no tolerated exception; before this
  change it reported 8 of 21.

  A sandboxed two-cycle run of curate, prepare and the learning controller
  accepted both cycles, wrote concepts, experiences, long-term memory and a
  12000-character external memory record with its origin, and produced a
  relationships collection that did not previously exist. A separate process
  then opened that store and retrieved from it, which is the retention
  property Architecture 16 describes. A conversation against the same store
  recalled the learned document on a question about it, scoring 1.0 on the
  two query terms it shared with the document.

  The generation quality remains poor. This is a 3.29M parameter byte-level
  model whose sandbox run used 32 micro-steps, and nothing here changes that.

Affected Components:
  Relationship system (new type), Retrieval (new), Memory (external memory
  now written and read), Observability (telemetry summary, census in the
  event stream), Learning controller (drives the entry point), Concept
  system, Architecture 52 status table, Architecture 12 type list.

Not changed:
  Model architecture, tokenizer, parameter count, and every explicit control
  in Architecture 38. The checkpoint format is untouched.
```

## 2026-09-26 - Learning controller, regression detection, acceptance

```text
Architecture Version: 0.1 (living specification, first recorded change)

Change:
  Added a continuous learning worker that loops the existing proven
  micro-update cycle with an explicit stopping rule. Added a reserved
  regression document, per-source confidence, Architecture 20 update
  acceptance verdicts, experiment records, a review step, a hard
  per-cycle time limit and a single-writer lock. train.py was refactored
  mechanically so both entry points share one micro-update implementation.

Reason:
  Architecture 37 asks for continuous operation, and Architecture 33 warns
  that an update which improves one capability while destroying another is
  not a success. Repeated learning had no mechanism to detect forgetting:
  every document supplied its own holdout, so improvement on the current
  document was the only signal available and accepting it could silently
  damage earlier learning.

Evidence:
  A four-cycle run produced the deciding observation. Cycle 3 improved its
  own document by +0.704 holdout while the reserved document regressed by
  -0.435, and was classified REJECT instead of being saved. A run with a
  single learnable document, and therefore no yardstick, classified an
  improvement of +1.808 as REVIEW rather than ACCEPT, because forgetting
  could not be ruled out. A 16-cycle run of the previous behaviour accepted
  16 of 16 updates on per-document holdout alone and overshot its 900s
  budget to 934.6s, which is why the per-cycle limit was added.

Affected Components:
  Neural model (unchanged), Tokenizer (unchanged), Micro-learning
  (unchanged logic, extracted), Pre/post evaluation (unchanged),
  Regression detection (new), Confidence (new), Update acceptance (new),
  Learning controller (new), Continuous operation (new), Observability
  (new), Experiment tracking (new), Rollback (unchanged mechanism,
  now verified by re-measurement), Architecture 52 status table.

Not changed:
  Model architecture, tokenizer, parameter count, and every
  explicit control in Architecture 38.
```

## 2026-09-26 - Knowledge layer

```text
Architecture Version: 0.2

Change:
  Added a JSON-backed document store, explicit concepts with maturity and
  census, typed relationships with their own confidence, the four memory
  classes from Architecture 15, and the Architecture 16 routing decision.
  The learning controller now records an experience per cycle and promotes
  candidate terms into concept evidence.

Reason:
  Architecture 9 requires multiple representations rather than everything
  forced into weights, and Architecture 10 insists concepts are an external
  organization and measurement mechanism. Without them, learning produced
  weights and telemetry but nothing Jupiter could reason about or query.

Evidence:
  A three-cycle run produced 15 concepts, 3 experience records and 3
  long-term entries, with the rejected cycle recorded as REJECT alongside the
  accepted ones. Contradiction testing preserved both sides of a conflict and
  kept the contradicted relationship with its evidence rather than deleting
  it. Cyclic prerequisite graphs terminate instead of recursing forever, and
  a corrupt store raises rather than silently starting empty.

Affected Components:
  Concept system, Relationship system, Memory, Experience system.
  Neural model, Tokenizer and architecture unchanged.
```

## Known limitations at this version

```text
Regression detection uses one reserved document. Forgetting that damages
the reserved document and the learned document equally is invisible.

Confidence is an evidence counter, not the competing-representation model
described in Architecture 13.

Concepts are extracted by term frequency with a stopword list, not by
understanding. On prose this still promotes words like "check" and "shows".
Real extraction needs either curated concept lists or genuine language
analysis, and this is labelled as a heuristic wherever it is used. A concept
description therefore records where its evidence came from, never what the
term means.

Retrieval is lexical. A question is matched by shared significant terms, so
a paraphrase that shares no vocabulary is not found, and two unrelated
records that happen to share words score alike. Every result reports the
terms that matched so a recall can be judged rather than trusted.

CO_OCCURS is the only relationship Jupiter derives from a single document: two
terms co-occurred because one document ranked them all highly, which is a fact
about the corpus. Two more are derived from the accumulated evidence.
SIMILAR_TO comes from how similarly two terms are distributed across sources,
and PRECEDES from an order the material states consistently, and both are
refused rather than guessed when the evidence is thin: a pair seen once, or an
order that reverses, produces no edge at all.

Co-occurrence is recorded whatever the verdict, because it is a fact about the
corpus and not a claim that learning worked. It used to be recorded only on an
accepted cycle, which starved the derivation of evidence: 21 accepted cycles out
of 124 left the graph with almost nothing to derive from. A document that was
read and rejected still says which terms it contains. What stays behind the
acceptance gate is everything that asserts Jupiter is standing behind the
content: the retrievable copy of the text, the claims harvested from it, and the
programming and register structure.

Seven of the eleven types - IS_A, PART_OF, DEPENDS_ON, USED_BY, CAUSES,
REQUIRES, IMPLEMENTS - are still named and never created, and the evaluation
report says so by name rather than letting the enumeration imply a diagnostic
power that does not exist. Inventing one of them to make the list look complete
would be a claim the corpus does not support.

Jupiter now holds conversation. The corpus carries structured instruction
records rather than labelled transcripts, dialogue.py reads both layouts, and a
record is rendered as the two turns the model is spoken in. This was prose with
no questions in it when this section was written and the corpus has since
changed; what remains true is the finding underneath it, which is that the
material has to be there. A model that has never seen a question cannot learn
to answer one, and it was the corpus and not the code that was missing.

The context window is 512 tokens, which is roughly 1100 characters at the
corpus-wide compression rate. That is now the binding constraint rather than the
material: the corpus holds 135.8 GB of skill-targeted material and a cycle reads
a bounded number of windows from it.

Two documents in the current corpus carry leaked chat-template markers, one of
them with 3026 of them, and both are refused at curation. A third file was also
being refused, incorrectly: it carried ten lines beginning "user:", "system:" or
"human:" in sixteen million characters, which is writing that discusses users
and systems rather than a dumped conversation. Refusal now needs a marker
*share* as well as a count, for the same reason dialogue classification needs
one: a document that merely contains markers is not a document made of them.

The two refused documents are counted in data/curation_report.json and are
untouched in data/raw. They cannot be trained on whether or not they are
deleted, because curation refuses them and quarantines any curated copy.

All nine reported evaluation categories are measured. The report lists nine rows
rather than Architecture 34's eight because Frustration was added as a measured
category. Expansion was the last to join, once capacity growth could be applied
and judged rather than merely proposed; it reports what the growth decision was
worth, not how many growths were made, because a growth with no baseline loss to
compare against is reported as unjudged rather than scored. The report also
states how many of the eleven relationship types are actually in use, and what
the programming and register layers hold, because both write on every accepted
cycle and a layer nobody reads back is indistinguishable from one that is
silently failing.

The optimizer state is persisted beside the resume weights. It used to be
rebuilt at the start of every run, so AdamW's moments were not carried between
cycles and the first steps of each cycle were less effective than they would be
with a warm optimizer. Loading checks the tensor shapes itself rather than
trusting torch: load_state_dict matches by position and will accept a state
saved for a model that has since grown, leaving most parameters with no moment
at all, which is a run that looks resumed and is not.

Architecture 16's routing decision is made with eight of its nine factors.
The one that is missing is retrieval_cost, and it is passed as zero honestly:
nothing records how long a real retrieval against a source takes, because
nothing counts retrievals per source yet. That means no evidence rather than
evidence of nothing. All six declared destinations are reachable: external,
weights, concept, relationship, experience and multiple. A disputed but
supported fact is held as a relationship rather than a concept, because a concept
would have to pick one of two claims; something seen often enough that the
repetition is itself the information is held as an experience; and something
supported and disputed at once is held in both rather than being forced into
one.

Architecture 14 step 6 is implemented, and step 7 now has a representation it can
act on. A contradiction records which source caused it, and the decision between
a genuine conflict and a context difference is made by asking who is on each
side.

Step 7 is now reachable, through assertions. Term co-occurrence cannot say that
one source denies what another asserts, which is why no CONTRADICTS edge was ever
opened: a rejected cycle is evidence that learning a document damaged the
reserved document, not a disagreement about anything. An assertion records the
claim a source made and whether it is positive or negative, which is what makes
a disagreement expressible at all. Two sources asserting the same thing
corroborate; two sources asserting opposite things contradict, and that opens
the edge in both directions. A source denying a claim it previously asserted is
a context difference and opens none.

What remains is coverage. The patterns read "X is Y", "X is not Y", "X requires
Y", "X causes Y" and "X precedes Y" and are deliberately narrow, so a claim
phrased any other way is invisible. That is a limit of the extraction rather
than of the representation, and a wider set would recognise more wrongly.

The learning controller has not yet run long enough to be called verified.

MCP integration is now a module, mcp_server.py, and it is reached as
`run_jupiter.bat mcp`. It exposes the model over HTTP with JSON-RPC 2.0 so an
agent can connect once Jupiter has an address and a port, and it adds no
dependency: the transport is the standard library. Jupiter continues to work
without it, which is what Architecture 29 asks for. The endpoint binds to
loopback and is refused without a human granting external access, and a write
from a network caller is refused without the memory grant as well.

Learning frustration is implemented, which this section previously denied. It
computes a per-source and per-concept index, bands it, attributes a ranked cause
and reaches the entry point through the evaluation report, with its own test
suite. What is missing is the second half of Architecture 21: the index now
drives which document is learned next and when Jupiter writes more material of
its own, but it drives no architectural change, because dynamic expansion does
not exist. The frustration-to-expansion feedback loop is still open.

Jupiter can now reach outside itself, and the limit on that is worth stating
plainly because it is easy to overstate. The consult is a question Jupiter
already had: it is built from its own weakest concepts and answers nothing it
could not have asked itself. There is no goal behind it, and nothing is
perceived that a source did not already hold. The world is consulted, not
inhabited. Three gaps separate this from perception proper, and none of them
is closed: nothing arrives unprompted, nothing is asked because Jupiter
wants an answer, and nothing that comes back changes what Jupiter is trying
to do.
```
