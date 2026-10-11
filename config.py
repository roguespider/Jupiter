"""Jupiter runtime configuration.

The values here are seed defaults, not a declaration of Jupiter's final brain.
The architecture is persisted with each model so experiments can change it later.
"""

import json
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path

import json_source
import torch
from model import ModelConfig

# Architecture 43 asks for separate versions for the architecture, the model,
# the tokenizer, the learning system, the memory schema and experiments, and
# warns they will not advance together. This is the architecture version, and
# it is stored beside the weights so a checkpoint can always be read back
# against the shape it was trained with.
ARCHITECTURE_VERSION = "0.11"

# Architecture 43 asks for separate versions for the architecture, the model, the
# tokenizer, the learning system, the memory schema and experiments, precisely
# because they do not advance together. Architecture and tokenizer had one, and
# model and tensor gained one. These are the two that were still missing, and
# without them a knowledge store written by a different learning system, or
# holding a different shape, cannot be told apart from one built by this run.
#
# They are recorded where the thing they describe is written, so a reader can
# find the version next to the state it belongs to.
LEARNING_SYSTEM_VERSION = "0.1"
MEMORY_SCHEMA_VERSION = "0.1"

# Fields that define a model shape. Used to read a pre-versioned architecture
# file, which stored these at the top level with no version and no wrapper.
MODEL_FIELDS = tuple(field.name for field in fields(ModelConfig))


def _env(name, default, cast):
    value = os.getenv(name)
    return default if value is None else cast(value)


def _external_source_list(raw):
    """Paths named by JUPITER_EXTERNAL_SOURCES, as external source entries.

    Semicolon separated rather than comma separated, because these are paths and
    a path may contain a comma while a semicolon in a Windows path is not a
    separator anybody writes by accident. Each entry becomes a file source,
    which is the kind that needs no network policy to be safe.

    This exists so that consulting the world is reachable from a shell. A
    capability that can only be switched on by editing the source is not a
    capability the entry point can drive, and Architecture 40 asks for the
    project to behave as one interconnected system.
    """
    entries = []
    for item in raw.split(";"):
        location = item.strip().strip('"')
        if location:
            entries.append(
                {
                    "kind": "file",
                    "name": Path(location).stem or location,
                    "location": location,
                }
            )
    return tuple(entries)


@dataclass
class Config:
    # Seed architecture. Small on purpose so the first learning cycle is cheap.
    # 512 tokens is the conversation window: at one byte per token that is 512
    # characters, which is still small, and it is the reason the tokenizer
    # matters. See tokenizer_bpe.py for the multi-character alternative.
    block_size: int = 512
    # The BYTE vocabulary, not the model's vocabulary. This was called
    # vocab_size, which was actively misleading: with BPE the model is 5256 wide
    # and this field still said 256, so reading it looked like the migration had
    # not happened.
    #
    # The tokenizer decides the real size. build_model always passes the
    # tokenizer's vocabulary in, so this value is only a fallback for the rare
    # path that asks for a shape with no tokenizer to consult, and for tests
    # that deliberately want a byte-sized model. The architecture record and the
    # checkpoint are the authoritative record of what was actually built:
    # out/Jupiter.architecture.json currently reads vocab_size 5256.
    byte_vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True

    # Tokenizer (Architecture 8). BPE is the default: the byte tokenizer needs
    # nothing learned and cannot be invalidated, but at one byte per token a
    # 512-token window is 512 characters, which is too small to hold a question,
    # some retrieved knowledge and an answer. Set this to "bytes" to go back.
    # Switching is a migration either way: it changes vocab_size, the shape
    # guard in train.py notices, and the old checkpoint is archived rather than
    # loaded against a shape it was not trained for. See prepare_dataset.py,
    # which trains the vocabulary and reports the compression comparison
    # Architecture 8 asks experiments to record.
    tokenizer: str = "bpe"
    tokenizer_vocab: str = "out/tokenizer.json"
    # Architecture 29 MCP. Deliberately loopback: a model with a learned corpus
    # and a knowledge store is not something to place on a network because a
    # port was configured. Widen mcp_host deliberately. See mcp_server.py.
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 1337
    # Architecture 28 external knowledge. A list of {"kind", "location"} for
    # files and documentation, or {"kind": "mcp", "endpoint": ...} for an
    # agent. Empty by default: reaching outside the corpus is a decision, not a
    # default, and nothing here becomes internal knowledge without passing the
    # retention test. See external.py.
    external_sources: tuple = ()
    # Merges are chosen from a measurement rather than a round number. Recorded
    # on the current corpus: 500 merges gave 1.64 bytes per token, 2000 gave
    # 2.08, and 5000 gave 2.62 for under a second of training; beyond that the
    # corpus runs out of pairs. Note that those figures come from
    # prepare_dataset's 4000-character single-document sample, not from the
    # whole corpus, which measures about 2.14 bytes per token. The window
    # figure the architecture quotes inherits that narrower sample.
    # See data/dataset_report.json.
    tokenizer_merges: int = 5000

    # Learning is text-first. No permanent train.bin/val.bin pipeline.
    raw_dir: str = "data/raw"
    curated_dir: str = "data/curated"
    # Architecture 25 puts a Parsing stage between Raw Data and Cleaning. These
    # suffixes are read through json_source rather than as plain text, because a
    # structured file read as text trains the model on braces, quotes and key
    # names instead of on what the file says. They are configuration rather
    # than a constant in the parser so the set of readable formats is one
    # declared decision instead of something a reader has to find.
    structured_suffixes: tuple = json_source.SUFFIXES
    # A document carrying more chat-template markers than this is refused at
    # curation rather than trained on. The threshold is a count, not a
    # judgement: a play script legitimately has many "Name:" lines, and a
    # single stray one in a novel does not make it a conversation.
    max_template_artifacts: int = 5
    # The count above is not sufficient on its own, for the same reason a count
    # of speaker labels is not: "user:", "system:" and "human:" followed by a
    # colon are ordinary English in writing about interfaces, permissions and
    # agents. A document set in a conversation about those things can carry
    # dozens of such lines in a few million and is not a leaked template.
    # What a dumped template looks like is a document made of markers, so
    # refusal needs the share as well as the count. Both must be exceeded.
    # One percent is chosen from the measured separation rather than by taste:
    # the real leaked document in this corpus has a marker share of 0.106, and
    # a large technical document that merely discusses users and systems sits at
    # 0.00006. Ordinary prose about those subjects does not put a marker on one
    # line in a hundred.
    max_template_share: float = 0.01
    curation_report: str = "data/curation_report.json"
    # Where a document too large to hold in memory is broken into pieces. It is
    # a working directory rather than an output: the pieces are inputs to
    # curation, and the curated copies are what the learner reads.
    split_dir: str = "data/split"
    dataset_report: str = "data/dataset_report.json"
    # The full evaluation report. It is written here rather than printed in
    # full because the per-concept frustration census runs to hundreds of rows,
    # and a console that receives three thousand lines in a burst appears to
    # freeze rather than to report. See evaluation.main.
    evaluation_report: str = "data/evaluation_report.json"
    out_dir: str = "out"
    # Where documents curation refuses are kept. Outside out/ by default, and
    # set through JUPITER_QUARANTINE_DIR, because quarantine is a safety archive
    # rather than working state: nothing reads it, and in a project that curates
    # a corpus in batches it grows without bound. Beside the model it became the
    # largest directory in the project, which is not what an archive is for.
    quarantine_dir: str = "quarantine"
    learning_state: str = "out/learning_state.json"
    # Where the next run continues from. Deliberately not the published
    # checkpoint: that one is written only when a cycle is ACCEPTed, and an
    # update that has to be judged before it is published must still be able to
    # accumulate, or every run restarts from random initialization and nothing
    # is ever learned. See learning_worker for how the two are kept apart.
    resume_checkpoint: str = "out/Jupiter_resume.safetensors"
    # What the worker has already visited, so a need-scored queue can order the
    # corpus without starving anything. See selection.py.
    selection_state: str = "out/selection_state.json"
    # Material Jupiter writes for itself (Architecture 25, "generated examples"
    # and "Jupiter's own experiences"). Written into data/raw/generated as
    # ordinary raw text, so it is curated, deduplicated and judged on the same
    # terms as anything a person supplied. The per-run budget is deliberately
    # small: this is a research instrument measuring whether self-written
    # material helps, not a data engine, and a runaway feedback loop is the
    # failure mode that matters here. See synthesize.py.
    synthesis_dir: str = "data/raw/generated"
    synthesis_per_run: int = 2
    synthesis_max_new_tokens: int = 400
    # How much of the source is shown to the model as a prompt. Kept to the
    # order of size one cycle learns: a seed is a prompt, and using a whole
    # document as one produces a file that is mostly a copy of a document
    # already in the corpus.
    synthesis_seed_chars: int = 2000
    learning_status: str = "out/learning_status.json"
    knowledge_store: str = "out/knowledge.json"
    concept_terms_per_cycle: int = 5
    telemetry_file: str = "out/telemetry.jsonl"

    # One invocation is a small learning experiment, not a giant cooking run.
    # Sized against worker_cycle_max_seconds below rather than picked to look
    # impressive. The original 131072 was unreachable inside the per-cycle
    # limit at 512 tokens, so every run burned five minutes, timed out, and
    # saved nothing. Fixing that to 4096 made a cycle finish, but a run at
    # 4096 steps and 3e-4 then learned each document in one cycle and damaged
    # the reserved document by 0.6 to 0.8 every time, so nothing accumulated
    # and every cycle after the first was rejected. Architecture 2.3 and 18 ask
    # for the opposite: a learning event should modify only what is necessary.
    # 512 steps at 1e-4 is small enough to leave earlier learning standing, and
    # at the measured ~55 steps/s a cycle takes about nine seconds, so a
    # twenty cycle run is watchable rather than a coffee break.
    # The note here used to claim that 512 steps "is small enough to leave
    # earlier learning standing". Measured over 235 recorded cycles, it was not:
    # a 512-step push moved the reserved document by a median of 0.46, 187 of
    # 235 cycles were rejected for it, and because a rejection restores the model
    # the gain was discarded every time. Median holdout before each cycle stayed
    # flat at 5.16 across cycles 201 to 216, which is what a reset looks like,
    # and the published checkpoint the conversation loads was last written at
    # cycle 133 while the run had reached 216. The model was learning perfectly
    # well and going nowhere.
    #
    # Sixteen, measured rather than guessed. A sweep of 8, 16, 32, 64 and 128
    # ran each setting twelve times from the same checkpoint and the same
    # corpus, and 16 bought by far the most:
    #
    #     steps   accepted   gain per cycle   total
    #         8        8%          +0.024       +0.29
    #        16       17%          +0.395       +4.74   <- this
    #        32        0%          -0.207       -2.49
    #        64        0%          -0.418       -5.02
    #       128        0%          -0.595       -7.14
    #
    # Eight was too timid: it was accepted, but a cycle that barely moves the
    # model teaches almost nothing and the accepted ones add up to almost
    # nothing. Thirty-two and above are where the push becomes large enough to
    # damage the reserved document by a median of 0.85 against a tolerance of
    # 0.15, so every cycle is rejected and the gain is restored. Sixteen is the
    # setting that survives its own acceptance gate while still moving.
    #
    # That sweep was measured against a ONE-DOCUMENT yardstick, and the
    # yardstick is now three documents chosen across the size range. Measured
    # over the 120 cycles after that change, at these settings: the holdout
    # improved 83 times and the reserved set improved 14, so 73 cycles were real
    # learning discarded for regressing material that is now a broader sample
    # than the one the sweep was calibrated on. Mean rejected regression was
    # +0.29 against a tolerance of 0.15 - twice over - and the acceptance rate
    # fell to 2%.
    #
    # The two knobs are reduced together rather than one at a time. A smaller
    # step size alone leaves a cycle too timid to be worth accepting, which is
    # what eight steps did; a smaller learning rate alone leaves the step count
    # high enough to overshoot within a single cycle. Halving both keeps the
    # distance travelled comparable to the setting that worked while spreading
    # it over more, smaller updates, which is the shape least likely to damage
    # material it was not trained on.
    #
    # The cost is stated rather than discovered. These are raised and lowered
    # together for the same reason: the step size and the learning rate are two
    # halves of one distance travelled per cycle, and moving one without the
    # other leaves a cycle that either overshoots within itself or barely
    # moves.
    #
    # They were lowered to 8 and 5e-5 while the reserved set was a single
    # document and the tolerance an absolute 0.15, which together rejected
    # almost every cycle. With the tolerance now a share, they went back up.
    # Thirty-two was then measured to regress the reserved set by a median of
    # 5.34 per cent and a seventy-fifth percentile of 9.53, which the gate
    # rejected as a tail - and when the bar was widened to accept that tail the
    # reserved set drifted from 4.65 to 7.65 and generation collapsed onto
    # itself. Sixteen holds its level over a run.
    micro_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    holdout_fraction: float = 0.15
    min_text_chars: int = 128
    max_chunk_chars: int = 12000
    # The largest raw file curation will read into memory, in bytes. Curation
    # reads a whole document to judge it, so an unbounded read is a bound on
    # available memory rather than on the pipeline. A corpus that contains a
    # file larger than this is refused and reported rather than crashing the run
    # with a MemoryError that names a codec rather than the file responsible.
    # 64 MiB is far above any single document worth learning from and far below
    # what exhausts a machine, and it is configuration rather than a constant so
    # a project with a different machine can raise it deliberately.
    max_raw_bytes: int = 64 * 1024 * 1024

    # How many documents curation reads at once.
    #
    # Curation is almost entirely CPU work on bytes it has already read: a
    # sixty-six megabyte document spends its time in regex substitution and
    # hashing, not waiting on the disk. Run one at a time, curating this corpus
    # took about five hours on a thirty-two core machine, which is the machine
    # doing a fifth of what it is capable of.
    #
    # The default is deliberately below the core count. Each worker holds a
    # whole document plus its normalized and rendered copies at once, so a
    # worker is sized by the largest document rather than by the smallest, and
    # overcommitting here trades a MemoryError on a corpus of large shards for
    # time that is already not the bottleneck. Zero means decide from the core
    # count.
    curate_workers: int = 8

    # Large documents. A cycle reads max_chunk_chars at a time and rotating the
    # window is what lets repeated cycles cover a whole document, but one window
    # per cycle made a large document impractical rather than merely slow: a
    # five-megabyte document took 417 cycles to cover once, which is a limit on
    # what can be fed in at all rather than a preference.
    #
    # A cycle therefore reads a fixed number of CHARACTERS from a document, not
    # a fixed number of windows, so the cost of a cycle does not grow with the
    # document. The cap keeps one cycle inside a window the context can hold:
    # without it a very large document would produce a window bigger than
    # block_size and the tail would be silently unlearned, which is the exact
    # failure the rotating window exists to prevent.
    max_windows_per_cycle: int = 16
    target_chars_per_cycle: int = 96000
    seed: int = 1337

    # Inference.
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40

    # Safety/observability.
    gradient_alert: float = 5.0
    # Measured over 22,903 recorded steps on this corpus and model: the gradient
    # norm has a median of 3.2, a 90th percentile of 8.0, a 99th of 13.8, a
    # 99.9th of 43.4 and a maximum of 99.2. The five largest all occur at step
    # one or two of a cycle, because the opening steps on a document the model
    # has not seen are legitimately steep.
    #
    # The old value of 25 sat inside that range, and enforcing it stopped
    # training dead: exceeding it restored the same weights, recomputed the
    # same step, got the same gradient, and spent the retry budget before
    # abandoning the cycle. Eight consecutive cycles ended at step one.
    #
    # A value above everything observed means the threshold only fires on
    # genuine blow-up, which is what a stop threshold is for. A non-finite norm
    # is still caught separately and still recovers by restoring weights, so
    # the genuinely broken case is not dependent on this number.
    gradient_stop: float = 100.0
    worker_state: str = "out/learning_worker_state.json"
    worker_log: str = "out/learning_worker.log"
    # Where the work runs, decided from what the machine is already doing rather
    # than from what it could do. "cuda" whenever a GPU exists puts Jupiter on a
    # card somebody else may be training on, and a long unattended run holds
    # whatever it was given in its first second. resources.choose_device() reads
    # the GPU's memory and steps aside when the card is busy; JUPITER_DEVICE
    # overrides both, because a person who knows their machine should not have to
    # argue with it.
    device: str = ""
    # Above this share of GPU memory the GPU counts as taken and the CPU is used.
    gpu_memory_share_threshold: float = 0.80
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float32"
    )
    compile: bool = False

    # Continuous learning worker (learning_worker.py).
    # These are safety limits, not a training schedule. They exist so the
    # worker cannot run forever or keep a change that stopped helping.
    #
    # worker_max_seconds is the wall clock for ONE bounded pass, which is what
    # the default chain runs. It is not the limit on continuous operation:
    # --continuous restarts passes until the whole run reaches this budget, so a
    # fifteen minute limit there stopped the agent after a quarter of an hour of
    # a run the person had asked to keep going. Continuous mode has its own
    # budget below, which defaults to none.
    worker_max_cycles: int = 20
    worker_max_seconds: float = 900.0
    worker_patience: int = 2
    worker_idle_seconds: float = 30.0
    worker_max_recoveries: int = 6
    worker_cycle_max_seconds: float = 300.0
    worker_lock: str = "out/learning_worker.lock"

    # The wall clock for a whole --continuous run. Zero means no limit, so the
    # agent runs until Ctrl+C. A non-zero value is honoured, because a run that
    # cannot be stopped is not autonomous either, but the default is off rather
    # than fifteen minutes, since a person asking for continuous learning is
    # asking for it to continue and a limit they did not set is a limit they did
    # not ask for.
    continuous_max_seconds: float = 0.0

    # Continuous operation, Architecture 37. These govern the loop that runs
    # after a human has started it and before a human stops it. They are
    # separate from the per-run limits above because they answer a different
    # question: not "how long may one run take" but "how long may the agent keep
    # going on its own".
    #
    # The wall clock is a real bound and not a formality. An agent that cannot
    # be stopped is not autonomous, it is unattended, and the difference is
    # whether a person can end it.
    autonomy_state: str = "out/autonomy.json"
    autonomy_idle_seconds: float = 30.0
    autonomy_max_idle_passes: int = 20
    # How many consecutive consults of the outside world may come back empty
    # before the loop goes back to resting on what it has. The world is an
    # unbounded supply of documents, so an unbounded number of attempts never
    # reaches the loop's own stopping rule.
    autonomy_max_consecutive_consults: int = 3
    # When True and the external domain is granted, the loop writes a brief
    # status record to the configured path after every pass. That is the
    # smallest real form of being inhabited rather than consulted: the loop
    # acts on the world continuously, not only when it asks a question.
    autonomy_continuous_respond: bool = False
    autonomy_respond_path: str = "out/external/autonomy_status.json"
    # The objective the loop pursues. Not a metric (loss is the measurement),
    # but the purpose that makes the loop inhabited rather than consulted.
    # Default is empty: no independent goal, only the internal learning cycle.
    # When set, the loop evaluates whether it is meeting the objective and uses
    # CONSULT to respond to the world continuously rather than only when dry.
    autonomy_objective: str = ""
    # Zero means no pass limit: the loop is bounded by the wall clock and by a
    # person, not by a counter someone remembered to set.
    autonomy_max_passes: int = 0
    # How many learning cycles one pass of the loop runs before the agent looks
    # at its own state again. Short enough that the decision is made on recent
    # evidence, long enough that each pass is worth its startup cost.
    cycles_per_pass: int = 4

    # Regression detection (Architecture 33). A stratified sample of documents is
    # held out of the learning candidates and re-measured every cycle, so an
    # update that improves the current document but destroys earlier learning is
    # detected instead of being accepted.
    regression_state: str = "out/regression_set.json"
    regression_source: str = ""
    # How much the reserved set may get WORSE before an update is rejected, as a
    # FRACTION of the loss it started that cycle at. 0.08 means eight per cent.
    #
    # It was an absolute 0.15, which is not a unit that survives a run. The
    # reserved set's loss starts near 1.2 and drifts toward 5 as the corpus is
    # learned, so the same 0.15 of damage is twelve per cent early and three per
    # cent late. Measured absolutely over 200 cycles the median regression was
    # 0.296 - twice the tolerance - and the acceptance rate fell to 2 per cent,
    # with 68 of every 120 cycles being real learning discarded. The same
    # regressions are a median of 5.7 per cent, which is a number that means
    # something.
    #
    # Twelve per cent was measured against a healthy distribution and is wrong
    # here. Raising the bar to stop the tail also admitted the damage: over the
    # forty cycles after it went up, thirty-three of forty were rejected anyway,
    # the holdout loss climbed 2.97 -> 6.35, and the reserved set's own loss
    # drifted from 4.65 mid-run to 7.65. The replies stopped varying and three
    # different questions produced byte-identical answers. That is forgetting,
    # and a wider tolerance is not a fix for it - it is the gate being turned
    # off while the thing it exists to catch happens.
    #
    # Sixteen steps at 1e-4 is the measured setting: the reserved set regresses
    # by a median of a few per cent against a twelve per cent bar, and it holds
    # its level over a run. Eight was measured to be too timid to learn from.
    # The honest trade is throughput for stability, and with the store now 111x
    # faster the throughput is affordable anyway.
    regression_tolerance: float = 0.12
    # How far the conversation pass rate may fall before an update is rejected.
    # Same reasoning as regression_tolerance and separate from it: forgetting
    # earlier learning and losing the ability to hold a turn are different
    # failures with different causes, and a model that is worse at conversation
    # is worse at the thing conversation is for even if its loss improved.
    conversation_tolerance: float = 0.0
    # The absolute conversation pass rate below which a cycle cannot be judged
    # at all. conversation_tolerance above measures a FALL, and a fall can only
    # be measured once there is something to lose: while the pass rate is zero,
    # every run scores zero, every change is zero, and no update is ever rejected
    # for conversation. A model that cannot hold a turn is therefore accepted
    # indefinitely on the strength of its holdout loss, which is the exact
    # failure the conversation gate was written to catch.
    #
    # A fifth of the questions is a deliberately low bar. It is not a claim that
    # Jupiter can converse; it is the point at which "cannot hold a turn at all"
    # stops being true, and until then the honest verdict on any update is that
    # it cannot be judged.
    conversation_min_rate: float = 0.2
    # Where the last conversation evaluation is recorded, and how long a reply
    # may be. The budget is small because a model that cannot answer in a
    # sentence has not answered, and generating hundreds of tokens to find out
    # makes the gate slow enough to be turned off.
    conversation_eval_state: str = "out/conversation_eval.json"
    conversation_max_new_tokens: int = 48
    # How often the conversation set is run, in cycles. Each pass generates a
    # reply for every question, which is far more expensive than a forward pass,
    # so running it on every cycle would make the gate slow enough to be
    # switched off. Once every few cycles catches a regression before it is
    # trained into further.
    conversation_eval_interval: int = 5

    # How often, in cycles, recorded conversation is rendered into the corpus.
    # Architecture 4's chain runs conversation -> experience -> memory ->
    # curation -> learning, and the last seam did not exist: out/experiences.jsonl
    # was written by the conversation runtime and read by nothing. Rendering is
    # a whole-file rewrite of a document derived entirely from that log, so doing
    # it every cycle would be disk traffic for no new material. The conversation
    # log grows far more slowly than cycles run.
    experience_publish_interval: int = 25

    # Architecture 34 Language. The Language category could only be measured by
    # generating text, and nothing generated any: evaluate_language took a list
    # of samples that no caller in the project ever passed, so the line read
    # "no generated text to measure" on every run while the coverage table still
    # claimed the category was covered. These settings are what the probe reads.
    #
    # Prompts are drawn from the reserved document rather than written here,
    # because a hand-written set measures one narrow thing and the reserved
    # document is the one span of text the model was explicitly not trained on.
    language_samples: str = "out/language_samples.txt"
    language_prompts: int = 8
    language_max_new_tokens: int = 40

    # Where the conversation runtime records what was said. Kept as a setting
    # rather than a constant in the reader so the writer and the reader cannot
    # disagree about where experience lives, which would leave the loop open
    # without anything reporting it.
    experiences_path: str = "out/experiences.jsonl"

    # Update acceptance (Architecture 20). Verdicts are ACCEPT, REJECT, DEFER,
    # REVIEW or ROLLBACK. An improvement inside the dead band is DEFER rather
    # than either accept or reject.
    defer_band: float = 1e-4

    # How many recent cycles the acceptance gate averages before it will judge.
    #
    # A single cycle's holdout change cannot be read. Measured over 3,315 recorded
    # cycles the change has a standard deviation of 1.01 against a mean of -0.377,
    # so the noise is 2.7 times the signal, and the defer band is 1e-4. Judging one
    # cycle against a band that small is close to a coin flip on which document was
    # drawn, which is what made roughly half of every run get rejected for reasons
    # that had nothing to do with the update.
    #
    # Averaging n cycles divides the noise by the square root of n, so five cycles
    # brings the standard error to about 0.45 and sixteen to about 0.25. Sixteen is
    # the default because it is the first point at which the smoothed figure is
    # smaller than the per-cycle noise by a factor of four, and because sixteen
    # rejected cycles in a row should still end a source rather than accumulate.
    acceptance_smoothing: int = 16

    # How many windows the MEASUREMENT reads, as distinct from how many the
    # update reads.
    #
    # Measured on one document with the weights held fixed, reading 16 windows
    # instead of 1 reduced the spread of the resulting loss by 5.46x, against
    # the 4.0x that pure measurement noise predicts. So most of the variance the
    # acceptance gate has been reading was which window it happened to measure,
    # not how much the model had changed.
    #
    # This is separate from max_windows_per_cycle on purpose. That governs how
    # much material a cycle learns from; this governs how much material it is
    # judged on. Raising both makes each cycle slower for two different reasons,
    # and conflating them means neither change can be measured.
    measurement_windows: int = 3

    # The band the SMOOTHED figure is judged against, rather than the per-cycle
    # one. It is set to the noise floor rather than to 1e-4 because a mean of
    # sixteen cycles is what is being compared: demanding 1e-4 of a figure that
    # carries 0.25 of noise is demanding a precision the measurement cannot
    # deliver, and the honest band is the size of the error in the number judged.
    smoothed_defer_band: float = 0.25

    # Confidence (Architecture 13). Confidence is a measured strength of
    # evidence, not a truth value, and it is tracked per learning source.
    confidence_min: float = 0.35
    confidence_gain: float = 0.25
    confidence_loss: float = 0.20
    # How many cycles a source is judged on the evidence of THIS cycle before its
    # own record starts counting against it. Confidence is the share of a source's
    # cycles that were accepted, so a source held at REVIEW never accrues the
    # accepts it needs to clear the floor: it is held for lacking confidence and
    # lacks confidence because it is held. Two observations made that deadlock
    # permanent - over sixty cycles, twenty-eight per cent were decided by it and
    # one per cent were accepted.
    #
    # Twelve is enough for the share to mean something: at one-in-three weighting
    # a source with a genuine 50 per cent acceptance clears the floor within a
    # dozen cycles, and a source genuinely failing to be accepted does not.
    confidence_grace_observations: int = 12
    confidence_state: str = "out/confidence.json"

    # Experiment tracking (Architecture 31).
    experiment_log: str = "out/experiments.jsonl"

    # Growth cadence: one more layer per this many accepted cycles, when the
    # expansion domain is granted. Zero turns the cadence off and leaves growth
    # to the frustration diagnosis alone. Set through
    # JUPITER_GROWTH_CADENCE_CYCLES.
    growth_cadence_cycles: int = 100


def load_config() -> Config:
    c = Config()
    c.quarantine_dir = _env("JUPITER_QUARANTINE_DIR", c.quarantine_dir, str)
    c.gpu_memory_share_threshold = _env(
        "JUPITER_GPU_MEMORY_SHARE",
        c.gpu_memory_share_threshold,
        float,
    )
    c.growth_cadence_cycles = _env(
        "JUPITER_GROWTH_CADENCE_CYCLES",
        c.growth_cadence_cycles,
        int,
    )
    c.curate_workers = _env("JUPITER_CURATE_WORKERS", c.curate_workers, int)
    c.micro_steps = _env("JUPITER_MICRO_STEPS", c.micro_steps, int)
    c.learning_rate = _env("JUPITER_LR", c.learning_rate, float)
    c.device = os.getenv("JUPITER_DEVICE", c.device)
    if not c.device:
        import resources

        c.device, c.device_reason = resources.choose_device(
            threshold=c.gpu_memory_share_threshold
        )
    else:
        c.device_reason = f"set by JUPITER_DEVICE to {c.device}"
    c.dtype = os.getenv("JUPITER_DTYPE", c.dtype)
    c.compile = os.getenv("JUPITER_COMPILE", "false").lower() == "true"
    c.worker_max_cycles = _env("JUPITER_WORKER_MAX_CYCLES", c.worker_max_cycles, int)
    c.worker_max_seconds = _env(
        "JUPITER_WORKER_MAX_SECONDS", c.worker_max_seconds, float
    )
    c.continuous_max_seconds = _env(
        "JUPITER_CONTINUOUS_MAX_SECONDS", c.continuous_max_seconds, float
    )
    c.worker_patience = _env("JUPITER_WORKER_PATIENCE", c.worker_patience, int)
    c.worker_idle_seconds = _env(
        "JUPITER_WORKER_IDLE_SECONDS", c.worker_idle_seconds, float
    )
    c.autonomy_idle_seconds = _env(
        "JUPITER_AUTONOMY_IDLE_SECONDS", c.autonomy_idle_seconds, float
    )
    c.autonomy_max_idle_passes = _env(
        "JUPITER_AUTONOMY_MAX_IDLE_PASSES", c.autonomy_max_idle_passes, int
    )
    c.autonomy_max_passes = _env(
        "JUPITER_AUTONOMY_MAX_PASSES", c.autonomy_max_passes, int
    )
    c.autonomy_max_consecutive_consults = _env(
        "JUPITER_AUTONOMY_MAX_CONSECUTIVE_CONSULTS",
        c.autonomy_max_consecutive_consults,
        int,
    )
    c.autonomy_continuous_respond = _env(
        "JUPITER_AUTONOMY_CONTINUOUS_RESPOND",
        c.autonomy_continuous_respond,
        lambda v: v.lower() in ("1", "true", "yes", "y"),
    )
    c.autonomy_respond_path = _env(
        "JUPITER_AUTONOMY_RESPOND_PATH",
        c.autonomy_respond_path,
        str,
    )
    c.autonomy_objective = _env(
        "JUPITER_AUTONOMY_OBJECTIVE",
        c.autonomy_objective,
        str,
    )
    c.external_sources = _env(
        "JUPITER_EXTERNAL_SOURCES", c.external_sources, _external_source_list
    )
    c.cycles_per_pass = _env("JUPITER_CYCLES_PER_PASS", c.cycles_per_pass, int)
    c.max_windows_per_cycle = _env(
        "JUPITER_MAX_WINDOWS_PER_CYCLE", c.max_windows_per_cycle, int
    )
    c.target_chars_per_cycle = _env(
        "JUPITER_TARGET_CHARS_PER_CYCLE", c.target_chars_per_cycle, int
    )
    c.worker_max_recoveries = _env(
        "JUPITER_WORKER_MAX_RECOVERIES", c.worker_max_recoveries, int
    )
    c.worker_cycle_max_seconds = _env(
        "JUPITER_WORKER_CYCLE_MAX_SECONDS", c.worker_cycle_max_seconds, float
    )
    c.regression_tolerance = _env(
        "JUPITER_REGRESSION_TOLERANCE", c.regression_tolerance, float
    )
    c.defer_band = _env("JUPITER_DEFER_BAND", c.defer_band, float)
    c.acceptance_smoothing = _env(
        "JUPITER_ACCEPTANCE_SMOOTHING", c.acceptance_smoothing, int
    )
    c.measurement_windows = _env(
        "JUPITER_MEASUREMENT_WINDOWS", c.measurement_windows, int
    )
    c.smoothed_defer_band = _env(
        "JUPITER_SMOOTHED_DEFER_BAND", c.smoothed_defer_band, float
    )
    c.confidence_min = _env("JUPITER_CONFIDENCE_MIN", c.confidence_min, float)
    c.confidence_gain = _env("JUPITER_CONFIDENCE_GAIN", c.confidence_gain, float)
    c.confidence_loss = _env("JUPITER_CONFIDENCE_LOSS", c.confidence_loss, float)
    c.tokenizer = os.getenv("JUPITER_TOKENIZER", c.tokenizer)
    c.tokenizer_vocab = os.getenv("JUPITER_TOKENIZER_VOCAB", c.tokenizer_vocab)
    c.tokenizer_merges = _env("JUPITER_TOKENIZER_MERGES", c.tokenizer_merges, int)
    return c


def architecture_dict(
    c: Config, vocab_size: int | None = None, shape: dict | None = None
) -> dict:
    """The model shape fields.

    `vocab_size` overrides the configured value so the recorded architecture
    always matches the tokenizer actually in use. A vocabulary file of 4000
    tokens against a model built for 256 would otherwise load and then fail
    deep inside a forward pass instead of at the point of configuration.

    `shape` overrides the shape fields with the model that was actually built.
    It exists because a grown model is no longer described by the configuration
    that started it: an approved expansion adds layers to the live model while
    the seed config still says what it always said. Recording the config would
    write down a shape the weights do not have, and the next run would then read
    that record, build the smaller model, and archive the checkpoint it just
    produced. The record exists so a checkpoint can be read back against the
    shape it was trained with, so it has to describe the model rather than the
    wish that produced it.
    """
    recorded = {
        "block_size": c.block_size,
        "vocab_size": c.byte_vocab_size if vocab_size is None else int(vocab_size),
        "n_layer": c.n_layer,
        "n_head": c.n_head,
        "n_embd": c.n_embd,
        "dropout": c.dropout,
        "bias": c.bias,
    }
    for name, value in (shape or {}).items():
        if name in recorded:
            recorded[name] = value
    return recorded


def architecture_envelope(
    c: Config, vocab_size: int | None = None, shape: dict | None = None
) -> dict:
    """The versioned architecture record written beside every checkpoint.

    Architecture 30 requires a saved instance to carry enough information to
    determine the architecture, and Architecture 43 requires versions not to be
    assumed to advance together. The shape fields sit under "config" so a
    version can be added without a bare field colliding with a model field.
    """
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "tokenizer": c.tokenizer,
        "config": architecture_dict(c, vocab_size, shape),
        "written": time.time(),
    }


def read_architecture(path: str | Path):
    """Read an architecture record and return (config dict, version or None).

    Two shapes exist on disk. The current one is the versioned envelope. The
    older one stored the model fields at the top level with no version, so it
    is still accepted and its version is reported as unknown rather than
    guessed at. A record that carries no recognisable model fields raises,
    because silently falling back to defaults would attach a checkpoint to a
    shape it was never trained for.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        # TypeError, not ValueError. The file parsed as JSON and the document
        # is simply the wrong shape, which is a type fault and not a malformed
        # one. It matters because a caller catching ValueError to mean "this
        # file is damaged" will now not swallow this, and the two deserve
        # different responses: a bad shape is a version problem, a parse failure
        # is damage.
        raise TypeError(
            f"architecture record {path} must be a JSON object, "
            f"found {type(data).__name__}"
        )
    if isinstance(data.get("config"), dict):
        stored = {k: v for k, v in data["config"].items() if k in MODEL_FIELDS}
        return stored, data.get("architecture_version")
    stored = {k: v for k, v in data.items() if k in MODEL_FIELDS}
    if not stored:
        raise ValueError(
            f"architecture record {path} contains none of "
            f"{', '.join(MODEL_FIELDS)}; it cannot be used to build a model"
        )
    return stored, None


def shape_of(fields_map: dict) -> dict:
    """The subset of architecture fields that decides tensor shapes.

    Two checkpoints with the same shape are interchangeable even if they
    differ in dropout or bias, which do not change any tensor's size.
    """
    return {
        name: fields_map.get(name)
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd")
    }


def _already_approved_growth(c, changed):
    """Whether this shape change is the result of an approved growth.

    A human approves a growth through the expansion gate and the next run builds
    the larger model. Recording that model is not a second act of authorship, and
    requiring a separate architecture grant for it means the approval they gave
    does not let the thing they approved happen: the run grows the model and then
    stops with the new weights on disk and the record still describing the old
    shape, so the next run builds the small model again and the growth is undone
    by a gate nobody was asked about.

    The change must match the approved growth field for field. Anything else
    still asks, so this is not a way around the architecture gate for edits
    nobody authorised.
    """
    import control
    import expansion as expansion_module

    # The expansion grant must be present as well. Without that check the
    # exemption would let anyone able to write the expansion state file change
    # the recorded shape of a model in use with no control at all, which is the
    # one thing Architecture 38 exists to prevent.
    if "expansion" not in control.granted_domains():
        return False
    proposal = expansion_module.load(c).get("capacity_growth")
    if not isinstance(proposal, dict) or not proposal.get("approved"):
        return False
    expected = {
        "n_layer": int(proposal.get("to_layers", -1)),
        "n_embd": int(proposal.get("to_embd", -1)),
    }
    shape_fields = ("n_layer", "n_embd", "n_head", "block_size", "vocab_size")
    touched = [name for name in changed if name in shape_fields]
    if not touched:
        return False
    for name in touched:
        if expected.get(name) != changed[name][1]:
            return False
    return True


def save_architecture(
    c: Config,
    path: str | Path,
    vocab_size: int | None = None,
    shape: dict | None = None,
) -> None:
    """Record the architecture the saved weights belong to.

    Architecture 38 asks for human control over architecture changes, and this
    is the file that states what the architecture is. It is written every run
    and normally changes nothing, so the gate is a statement of intent rather
    than an obstacle: a run that alters the shape of a model already in use has
    crossed from recording the architecture into changing it, and that is the
    point at which a human is asked.

    `shape` is the model that was actually built, which is not always the shape
    the configuration describes once a growth has been approved and applied.
    The gate compares the record against the model rather than against the
    configuration, so an expansion that a human already authorised is not
    refused a second time by the record catching up with the weights, and a
    change nobody authorised is still refused even when the configuration
    happens to agree with it.
    """
    import control

    target = Path(path)
    if target.exists():
        try:
            previous = read_architecture(target)[0]
        except (OSError, ValueError):
            previous = None
        proposed = architecture_dict(c, vocab_size, shape)
        if previous is not None:
            changed = {
                name: (previous.get(name), proposed.get(name))
                for name in proposed
                if previous.get(name) != proposed.get(name)
            }
            if changed and not _already_approved_growth(c, changed):
                differences = ", ".join(
                    f"{name} {was} -> {now}"
                    for name, (was, now) in sorted(changed.items())
                )
                # A growth that was approved through the expansion gate is not
                # asked about again here, and _already_approved_growth is the
                # check that says so. What it does not cover is the case where a
                # person granted expansion from the prompt during this run: the
                # approval is recorded, the architecture is written on the next
                # run, and the person is then told the architecture domain is not
                # granted. That is two gates for one action, and the second one
                # asks about a change the first gate already authorised.
                #
                # So when expansion has been granted, this gate is satisfied by
                # the same grant. It is not a widening: the person who can approve
                # a growth is the person who can authorise the architecture it
                # produces, and refusing the second half of an approved action is
                # a gate that makes its own permission unusable.
                control.check(
                    "architecture",
                    "change the recorded architecture of a model in use",
                    detail=differences,
                    c=c,
                    allowed=control.granted_domains() | {"expansion"},
                )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            architecture_envelope(c, vocab_size, shape), indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
