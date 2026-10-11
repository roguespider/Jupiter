# Jupiter From Scratch

A small causal transformer that learns through bounded micro-updates and
measures whether each one actually helped.

Current architecture:
- block_size: 512
- vocab_size: 256 (byte) or 256 + merges (BPE)
- n_layer: 4
- n_head: 4
- n_embd: 256
- ~3.36M parameters

This package starts from random initialization. It does NOT use ckpt.pt and
does NOT contain a checkpoint converter.

The governing specification is JUPITER_ARCHITECTURE.md, and the working rules
are in AGENTS.md. Read both before changing anything here.

Quick start:
1. python -m pip install -r requirements.txt
2. Put source documents in data/raw/ (see Data sources below)
3. run_jupiter.bat

run_jupiter.bat is the entry point for the whole project. It verifies the test
suite, curates, inspects and prepares the tokenizer, learns, shows recent
telemetry, reports the evaluation, checks whether remote training is
available, and then opens a conversation. Two more files are reached through
it as modes:

- run_jupiter.bat watch      follow the telemetry stream live
- run_jupiter.bat remote     train on a remote Modal GPU
- run_jupiter.bat mcp        serve the model to an agent over HTTP (MCP)

Individual stages, if you want one on its own:
- python curate_dataset.py
- python prepare_dataset.py
- python learning_worker.py            loop the proven cycle until a stopping rule fires
- python learning_worker.py --once     run exactly one cycle
- python learning_worker.py --detach   run in the background
- python train.py                     one cycle, through the same governed
                                      controller -- it does not save a checkpoint
                                      it has not judged
- python watch_training.py --summary   print the last 20 telemetry events
- python evaluation.py                 Architecture 34 coverage and metrics
- python evaluation.py --full          the same, plus the entire report on screen
- python sample.py                     talk to the model in a persistent session

Tokenizers:
- bytes (default)  one token per byte, lossless, cannot be invalidated
- bpe              learned byte-level BPE, measured at 2.384 bytes per token on
                   the current corpus, so a 512 token window holds roughly 1220
                   characters instead of 512

Switch with JUPITER_TOKENIZER=bpe. The vocabulary is trained automatically by
prepare_dataset.py if it is missing, and saved to out/tokenizer.json. Changing
the tokenizer changes the model shape, so the old checkpoint is archived under
out/archive/ and training restarts: weights cannot be reused across
vocabularies. Nothing was retrained as part of this change, so the checkpoint
on disk predates the current architecture and will be archived on the next run.

Environment overrides worth knowing:
- JUPITER_TOKENIZER        bytes or bpe
- JUPITER_TOKENIZER_MERGES how many merges to learn (default 5000)
- JUPITER_DEVICE           cuda or cpu
- JUPITER_DTYPE            bfloat16 (narrowed forward pass on CUDA) or float32

Tests:
- python -m unittest discover -s tests -t .   the whole suite
- python -m tests.test_connectivity           which modules the entry point reaches

A cycle is only saved when it is ACCEPTed, which means it improved its own
document and did not measurably damage the reserved regression document.

Two checkpoint files exist, and they answer different questions:

- out/Jupiter_latest.safetensors is the published model. It is written only when
  a cycle is ACCEPTed, and it is what the conversation loads, so Architecture 20's
  judgement governs everything anyone ever sees.
- out/Jupiter_resume.safetensors is where the next run continues from, and it is
  written after every cycle. It is working state, never published.

The split exists because a verdict answers "is this model ready to be shown", not
"may the next run build on this". Gating both files on ACCEPT meant a run that
never earned an ACCEPT persisted no weights at all, so every later run found
nothing to resume from, started from random initialization, and measured the
same starting loss forever. The run's final state is written after the
end-of-run rollback, so a run that ended on a rejected cycle hands its restored
state forward rather than its worst one.

What Jupiter can recall:
- Before each turn the question is matched against the knowledge store, and
  whatever is relevant is placed ahead of the conversation, so an answer can
  be grounded in material Jupiter actually learned. Matching is lexical, and
  the terms that matched are printed so a recall can be judged.
- /recall lists everything currently retrievable.
- Knowledge is kept in the context window and the oldest conversation is
  dropped, because a recalled fact answers the current question.

Outputs:
out/Jupiter_latest.safetensors     published model, written only on ACCEPT
out/Jupiter_resume.safetensors      run state, written every cycle, never published
out/Jupiter.architecture.json       the architecture the weights belong to
out/learning_state.json             full per-cycle record (resumable)
out/learning_status.json            small live status snapshot
out/learning_worker_state.json      worker run state and final review
out/experiments.jsonl               one record per cycle, including rejects
out/confidence.json                 per-source confidence and evidence
out/regression_set.json             which document is held out as the yardstick
out/telemetry.jsonl                 append-only event stream
out/learning_worker.log             detached worker output
out/selection_state.json            which documents have been visited, and when
data/raw/generated/                 material Jupiter wrote for itself
data/evaluation_report.json         full Architecture 34 evaluation report

Knowledge store:
out/knowledge.json                  concepts, relationships, memory, experiences
out/tokenizer.json                   learned BPE vocabulary, when trained

SafeTensors contains the model weights and nothing else. There is no
training_state file: the optimizer is rebuilt at the start of every run, so
AdamW's moments are not carried between cycles. That is a real limitation, not
an oversight to be papered over, and it is listed in the known limitations of
JUPITER_ARCHITECTURE.md.

The included tokenizer is a byte-token pipeline tokenizer for testing. Replace it with the exact Jupiter tokenizer before serious language training.

Curation and training are deliberately separate. Preprocessing can clean, deduplicate, score and balance data without training the model. The model still needs optimization to learn the selected data.

Data sources:

Curation reads Architecture 25's Parsing stage, so data/raw takes plain text and
structured records alike:

- .txt, .md, .text are read as text.
- .json, .jsonl and .ndjson are parsed and the text is taken out of the records,
  because a structured file read as characters trains the model on braces, quotes
  and key names instead of on what the file says.

The structured record shapes that are read:

    {"speaker": "Alice", "text": "..."}              one turn
    {"role": "user", "content": "..."}               one turn, role mapped to
                                                    the labels dialogue.py uses
    {"messages": [ ...turns... ]}                    a conversation
    {"conversations": [{"from":..., "value":...}]}   a ShareGPT export
    ["first paragraph", "second paragraph"]          paragraphs

A record whose fields are not in that list is counted as unrecognized and left
out, never guessed at: inventing text for a shape the parser has not read is how
a corpus fills with material nobody wrote. The count in the curation report is
the signal that a shape is worth adding. Turns are rendered through
dialogue.format_turn and flattened to one line, because dialogue.turn_share
measures turns against lines and a turn spread over three lines reads as prose.

The readable suffixes are Config.structured_suffixes, not a constant in the
parser. Any file in data/raw carrying an extension that is neither text nor
structured is counted and named under unsupported_files rather than skipped in
silence, so a file that is doing nothing is visible.

What the worker learns next, and what it writes for itself:

- Documents are ranked by need, not walked in alphabetical order. Need combines
  the Architecture 21 frustration band, the Architecture 13 confidence of a
  source, and an age term that reaches its maximum after one full rotation, so a
  document the model is permanently bad at comes round sooner without starving
  the rest of the corpus. The reason for each choice is in worker_cycle_start
  telemetry, and the visit history is in out/selection_state.json.
- Jupiter writes some material itself, into data/raw/generated/, when a source
  keeps defeating a bounded update for a reason more material could answer. It
  is gated on the cause rather than the score, so it does not fire for a
  representation or capacity failure that more examples would only reinforce.
  Generated files are named generated-<seed>-c<cycle>-<timestamp>.txt, so their
  provenance survives in the filename.
- Generated text is raw data like anything else. It is curated, deduplicated
  and judged on the same terms, and a generated document that does not earn an
  ACCEPT is rejected on the evidence, the same as any other. Whether it helped
  is therefore measured by the ordinary acceptance gate rather than asserted.

modal_train.py is an A100-80GB Modal training scaffold.
watch_training.py displays live telemetry.

MCP, Architecture 29:

    run_jupiter.bat mcp

serves Jupiter over HTTP with JSON-RPC 2.0 so an agent can connect once Jupiter
has an address and a port. It prints the loopback and network addresses on
startup. It adds no dependency: the transport is the standard library.

It binds to 127.0.0.1 by default. A model with a learned corpus and a knowledge
store is not something to place on a network because a port was configured, so
widen mcp_host deliberately.

It is also refused until a human authorises external access, and a write from a
network caller needs the memory grant as well:

    set JUPITER_CONTROL_GRANTS=external
    run_jupiter.bat mcp

Tools exposed: status, retrieve, capabilities, evaluate, learn_document.

Useful environment overrides: JUPITER_MICRO_STEPS, JUPITER_LR,
JUPITER_WORKER_MAX_CYCLES, JUPITER_WORKER_MAX_SECONDS,
JUPITER_WORKER_CYCLE_MAX_SECONDS, JUPITER_WORKER_PATIENCE,
JUPITER_REGRESSION_TOLERANCE, JUPITER_CONFIDENCE_MIN.

Every cycle records more than loss. The cycle record and each experiment row
carry perplexity, next-byte accuracy, the generalization gap between seen and
held-out loss, and the L2 magnitude of the weight change the cycle produced.
The resource sample includes disk usage, not just CPU, RAM and GPU.

Consulting the world (Architecture 28, now reachable):

    set JUPITER_EXTERNAL_SOURCES=E:\path\to\docs
    set JUPITER_CONTROL_GRANTS=expansion;external
    run_jupiter.bat autonomous

Without both the source and the grant, CONSULT is refused and the loop records
why. Without sources configured (the default), the loop behaves exactly as
before: the capability exists and is refused until both are supplied. The
consult only fires when the corpus is dry, so a normal run with material never
reaches it.
