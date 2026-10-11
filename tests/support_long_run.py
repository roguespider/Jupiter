"""The fixture the long-running tests share (Architecture 41).

This is the setup every long run needs: three short documents, a deliberately
tiny model, and a controller that runs for real. It lives here rather than in one
of the test files because the tests that use it now run as three separate files,
and a fixture copied into each of them would be three copies that could drift.

The model is a few hundred thousand parameters and the corpus is three short
documents. What these tests are for is the control flow and the bookkeeping -
need-scored selection, acceptance, regression detection, rollback, resume
persistence - and not the quality of the learning, so the size is chosen to keep
a full multi-cycle run to a few seconds while still exercising every stage.

The cost is real and is why the tests are split. Measured here, one cycle of
this tiny model takes about seven seconds, and the majority of it is the
per-skill probe pass: twenty-eight token-by-token generations on the CPU, which
is about four and a half of those seven seconds. Fifteen tests over three cycles
each is forty-five real cycles, which is over three minutes of the entry point's
two-minute budget if it all runs in one file.
"""

import contextlib
import io
import json
from pathlib import Path

import curate_dataset
import learning_worker
from telemetry import Telemetry

from tests.support import TempDirTestCase, make_config

# Short, self-contained and lexically distinct, so the tokenizer, the concept
# extractor and the regression yardstick all have something to work with. The
# text is deliberately flat rather than good prose: this suite is about whether
# the controller keeps its books straight, not about what Jupiter learns.
DOC_A = (
    "the observatory stood on the granite shelf above the valley. "
    "instruments recorded the river and the long dark weather. "
) * 30
DOC_B = (
    "a second account described the survey in careful ordered detail. "
    "every entry named the instrument the hour and the observer. "
) * 30
DOC_C = (
    "a third document concerns permissions and how a system grants them. "
    "an operator confirms before a change is applied to the record. "
) * 30


class LongRunTestCase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.raw = self.workdir / "raw"
        self.raw.mkdir(parents=True)
        self.out = self.workdir / "out"
        self.out.mkdir(parents=True)
        for name, text in (("a.txt", DOC_A), ("b.txt", DOC_B), ("c.txt", DOC_C)):
            (self.raw / name).write_text(text, encoding="utf-8")

    def config(self, **overrides):
        settings = dict(
            device="cpu",
            dtype="float32",
            compile=False,
            block_size=64,
            n_layer=1,
            n_head=1,
            n_embd=32,
            micro_steps=6,
            learning_rate=1e-3,
            raw_dir=str(self.raw),
            curated_dir=str(self.workdir / "curated"),
            curation_report=str(self.workdir / "curation.json"),
            out_dir=str(self.out),
            telemetry_file=str(self.out / "telemetry.jsonl"),
            knowledge_store=str(self.out / "knowledge.json"),
            learning_state=str(self.out / "learning_state.json"),
            learning_status=str(self.out / "learning_status.json"),
            worker_state=str(self.out / "worker_state.json"),
            worker_lock=str(self.out / "worker.lock"),
            regression_state=str(self.out / "regression_set.json"),
            confidence_state=str(self.out / "confidence.json"),
            experiment_log=str(self.out / "experiments.jsonl"),
            selection_state=str(self.out / "selection.json"),
            resume_checkpoint=str(self.out / "Jupiter_resume.safetensors"),
            synthesis_dir=str(self.raw / "generated"),
            synthesis_per_run=1,
            synthesis_max_new_tokens=32,
            worker_cycle_max_seconds=60.0,
        )
        settings.update(overrides)
        return make_config(**settings)

    def curate_with(self, c):
        original = curate_dataset.load_config
        curate_dataset.load_config = lambda: c
        self.addCleanup(setattr, curate_dataset, "load_config", original)
        with contextlib.redirect_stdout(io.StringIO()):
            # The config is passed rather than left to be found, so a test cannot
            # reach the developer's own corpus even if the patch above is
            # removed. Seven tests were curating eighty gigabytes of real
            # material because curate() fell back to load_config().
            return curate_dataset.curate(c)

    def run_worker(self, c, cycles):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            learning_worker.run_worker(c, Telemetry(c.telemetry_file), cycles)
        return buffer.getvalue()

    def experiments(self, c):
        path = Path(c.experiment_log)
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def telemetry(self, c):
        path = Path(c.telemetry_file)
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
