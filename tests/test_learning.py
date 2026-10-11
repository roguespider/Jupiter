"""Learning, tokenizer and persistence tests.

Covers Architecture 8 (tokenization), 30 (SafeTensors persistence), 35
(observability) and 45 (the first milestone: learn a small piece of
information, measure it, save it, reload it, and know whether the update
helped).

These use a deliberately tiny model and a handful of steps. The point is not
capability; it is that a bounded micro-update measurably improves the material
it was given and that the result survives a save and reload.
"""

import json
import unittest
from pathlib import Path

import tokenizer as tokenizer_module
import torch
import train
from model import Jupiter, ModelConfig
from safetensors.torch import load_model
from telemetry import Telemetry, read_recent
from train import (
    loss_on,
    make_batch,
    measure_on,
    run_micro_learning,
    save_checkpoint,
    seed_everything,
    snapshot_model,
    split_tokens,
    update_magnitude,
    window_for,
)

from config import Config
from tests.support import TempDirTestCase, make_config

REPEATED_TEXT = (
    "the learning worker records what happened during every cycle. "
    "a cycle is only saved when it is accepted. "
) * 40


class TokenizerTests(unittest.TestCase):
    """Architecture 8: the tokenizer must represent more than clean English."""

    def test_round_trip_preserves_text(self):
        for text in ("hello", "recursion", "naive cafe", "tab\tseparated"):
            self.assertEqual(
                tokenizer_module.decode(tokenizer_module.encode(text)),
                text,
            )

    def test_multibyte_characters_survive(self):
        for text in ("naïve café", "日本語", "🦙"):
            self.assertEqual(
                tokenizer_module.decode(tokenizer_module.encode(text)),
                text,
            )

    def test_malformed_input_does_not_crash(self):
        self.assertIsInstance(
            tokenizer_module.decode(tokenizer_module.encode("\ud800")), str
        )

    def test_validate_accepts_the_whole_byte_range(self):
        tokenizer_module.validate(list(range(256)))

    def test_validate_rejects_ids_outside_the_vocabulary(self):
        for bad in (256, -1, 1000):
            with self.assertRaises(ValueError):
                tokenizer_module.validate([bad])


class MicroUpdateTests(TempDirTestCase):
    """Architecture 41 learning test: does the update produce the intent?"""

    def setUp(self):
        super().setUp()
        torch.manual_seed(7)
        # run_micro_learning picks its batches with random.randrange, and only
        # torch was seeded here. The tests in this class assert that a bounded
        # update reduces loss, which is a statement about the update rather than
        # about which batch it happened to draw; with an unseeded sampler that
        # assertion failed roughly one run in twenty, which is worse than no
        # test because a red gate is indistinguishable from a real regression.
        seed_everything(7)
        self.config = make_config(
            device="cpu",
            block_size=32,
            micro_steps=12,
            learning_rate=3e-3,
            holdout_fraction=0.25,
            telemetry_file=str(self.workdir / "telemetry.jsonl"),
        )
        self.model = Jupiter(
            ModelConfig(block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64)
        ).to("cpu")
        self.tokenizer = tokenizer_module.ByteTokenizer()

    def learn_once(self):
        c = self.config
        train_tokens, holdout_tokens = split_tokens(
            REPEATED_TEXT, c.block_size, c.holdout_fraction, self.tokenizer
        )
        telemetry = Telemetry(c.telemetry_file)
        before = measure_on(holdout_tokens, self.model, c)
        optimizer = self.model.configure_optimizer(c.learning_rate, c.weight_decay)
        result, _, _ = run_micro_learning(
            self.model,
            c,
            train_tokens,
            holdout_tokens,
            telemetry,
            cycle=1,
            before_seen=before,
            before_holdout=before,
            learning_rate=c.learning_rate,
            optimizer=optimizer,
        )
        return train_tokens, holdout_tokens, before["loss"], result

    def test_a_bounded_micro_update_reduces_loss(self):
        train_tokens, holdout_tokens, before, result = self.learn_once()
        after = loss_on(holdout_tokens, self.model, self.config)
        self.assertEqual(result["successful_steps"], self.config.micro_steps)
        self.assertEqual(result["gradient_recoveries"], 0)
        self.assertLess(
            after,
            before,
            "a micro-update on repeated text must reduce holdout loss",
        )
        self.assertGreater(result["train_loss_improvement"], 0.0)
        self.assertGreater(result["holdout_loss_improvement"], 0.0)

    def test_the_micro_update_actually_changes_weights(self):
        c = self.config
        train_tokens, holdout_tokens = split_tokens(
            REPEATED_TEXT, c.block_size, c.holdout_fraction, self.tokenizer
        )
        before = self.model.token_embedding.weight.detach().clone()
        optimizer = self.model.configure_optimizer(c.learning_rate, c.weight_decay)
        run_micro_learning(
            self.model,
            c,
            train_tokens,
            holdout_tokens,
            Telemetry(c.telemetry_file),
            cycle=1,
            before_seen=measure_on(train_tokens, self.model, c),
            before_holdout=measure_on(train_tokens, self.model, c),
            learning_rate=c.learning_rate,
            optimizer=optimizer,
        )
        self.assertFalse(
            torch.equal(before, self.model.token_embedding.weight),
            "a micro-update that reports progress must have moved the weights",
        )

    def test_the_cycle_reports_before_and_after_separately(self):
        _, _, before, result = self.learn_once()
        self.assertEqual(result["holdout_loss_before"], before)
        self.assertNotEqual(result["holdout_loss_after"], before)

    def test_a_checkpoint_round_trips_to_identical_weights(self):
        c = self.config
        c.out_dir = str(self.workdir / "out")
        path = save_checkpoint(self.model, c, cycle=3)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "Jupiter_latest.safetensors")
        reloaded = Jupiter(self.model.config)
        load_model(reloaded, str(path), strict=True)
        for name, original in self.model.state_dict().items():
            self.assertTrue(
                torch.equal(original, reloaded.state_dict()[name]),
                f"tensor {name} changed across the save/reload",
            )

    def test_a_cycle_time_limit_stops_the_loop(self):
        from train import CycleTimeLimitExceeded

        c = self.config
        train_tokens, holdout_tokens = split_tokens(
            REPEATED_TEXT, c.block_size, c.holdout_fraction, self.tokenizer
        )
        optimizer = self.model.configure_optimizer(c.learning_rate, c.weight_decay)
        with self.assertRaises(CycleTimeLimitExceeded):
            run_micro_learning(
                self.model,
                c,
                train_tokens,
                holdout_tokens,
                Telemetry(c.telemetry_file),
                cycle=1,
                before_seen=measure_on(train_tokens, self.model, c),
                before_holdout=measure_on(train_tokens, self.model, c),
                learning_rate=c.learning_rate,
                optimizer=optimizer,
                deadline=0.0,
            )


class WindowTests(unittest.TestCase):
    """A cycle must see a different part of a document than the last one.

    Reading truncated every document to its first max_chunk_chars, so a
    twelve-million-character book taught the same twelve thousand characters on
    every cycle and 99.9 percent of the corpus was never seen. No run over the
    corpus is possible without this.
    """

    def test_a_document_shorter_than_the_window_is_unchanged(self):
        text = "abcdefghij"
        self.assertEqual(window_for(text, 0, 100), text)
        self.assertEqual(window_for(text, 7, 100), text)

    def test_consecutive_cycles_take_different_parts(self):
        text = "".join(chr(ord("a") + n % 26) for n in range(1000))
        self.assertNotEqual(window_for(text, 0, 100), window_for(text, 1, 100))

    def test_every_window_has_the_requested_size(self):
        text = "x" * 1000
        for index in range(10):
            self.assertEqual(len(window_for(text, index, 100)), 100)

    def test_cycles_eventually_cover_the_whole_document(self):
        text = "".join(str(n % 10) for n in range(1000))
        covered = "".join(window_for(text, index, 100) for index in range(10))
        self.assertEqual(covered, text)

    def test_the_window_stays_inside_the_document(self):
        text = "y" * 250
        for index in range(12):
            self.assertEqual(len(window_for(text, index, 100)), 100)


class BudgetConsistencyTests(unittest.TestCase):
    """A default cycle has to be able to finish.

    micro_steps and worker_cycle_max_seconds are two knobs that constrain each
    other and nothing connected them. Raising block_size from 256 to 512 doubled
    the cost of a step and made the default micro_steps unreachable inside the
    limit, so every run burned five minutes, timed out and saved nothing. That
    reads as a broken project rather than as a misconfiguration, so the
    relationship is pinned here.
    """

    # Deliberately pessimistic. The measured rate on one 4090 was about 55
    # steps/s; a CPU-only machine is far slower, and a default that only fits
    # on fast hardware is not a default.
    CONSERVATIVE_STEPS_PER_SECOND = 20

    def test_a_default_cycle_fits_inside_the_per_cycle_limit(self):
        c = Config()
        reachable = c.worker_cycle_max_seconds * self.CONSERVATIVE_STEPS_PER_SECOND
        self.assertLessEqual(
            c.micro_steps,
            reachable,
            f"a default cycle needs {c.micro_steps} steps but only about "
            f"{int(reachable)} fit in {c.worker_cycle_max_seconds}s at a "
            f"pessimistic {self.CONSERVATIVE_STEPS_PER_SECOND} steps/s",
        )

    def test_one_cycle_is_what_has_to_fit(self):
        # The run does not need to complete every cycle: Architecture 37's
        # stopping rules end it as soon as any limit fires. What has to fit is
        # one cycle, because a cycle that cannot finish saves nothing and the
        # run looks broken.
        c = Config()
        reachable = c.worker_cycle_max_seconds * self.CONSERVATIVE_STEPS_PER_SECOND
        self.assertLessEqual(c.micro_steps, reachable)

    def test_the_run_has_stopping_rules_that_can_end_it(self):
        # At 500 steps a twenty cycle run fits inside the time budget, so the
        # run no longer depends on a limit firing to finish. The limits still
        # have to exist and be set, because patience, recoveries and the clock
        # are what stop a run that starts going badly.
        c = Config()
        self.assertGreater(c.worker_max_cycles, 0)
        self.assertGreater(c.worker_max_seconds, 0)
        self.assertGreater(c.worker_patience, 0)
        self.assertGreater(c.worker_max_recoveries, 0)

    def test_a_cycle_is_short_enough_to_watch(self):
        # The point of a small update is that you can see what it did.
        c = Config()
        self.assertLessEqual(c.micro_steps, 1000, "a cycle should stay watchable")


class GradientStopTests(TempDirTestCase):
    """A large gradient is damped, and a broken one is recovered from.

    gradient_stop was configured at 25.0, written into every experiment row and
    telemetry event as stop_threshold, and compared against nothing.

    Enforcing it naively then stopped training dead, and this class exists to
    hold the distinction that fixes that. A non-finite gradient means the
    weights are broken; restoring the last safe state genuinely repairs them, so
    recomputing the step can succeed. A merely large gradient is not a fault at
    all: the step is recomputed from the same weights on the same batch, so it
    produces the same gradient, and halving the learning rate does not change
    that. Retrying it spent the budget three times and then abandoned the cycle.
    Eight consecutive cycles ended at step one with "Unable to stabilize
    gradient".

    Measured over 22,903 steps, the norm has a median of 3.2 and a maximum of
    99.2, with the five largest all at step one or two of a cycle. A threshold
    inside that range fires on ordinary learning, so it is set above it and the
    response is to damp and carry on rather than to abort.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(7)
        self.config = make_config(
            device="cpu",
            block_size=32,
            micro_steps=12,
            learning_rate=3e-3,
            holdout_fraction=0.25,
            telemetry_file=str(self.workdir / "telemetry.jsonl"),
        )
        self.model = Jupiter(
            ModelConfig(block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64)
        ).to("cpu")
        self.tokenizer = tokenizer_module.ByteTokenizer()

    def learn(self):
        c = self.config
        train_tokens, holdout_tokens = split_tokens(
            REPEATED_TEXT, c.block_size, c.holdout_fraction, self.tokenizer
        )
        before = measure_on(holdout_tokens, self.model, c)
        optimizer = self.model.configure_optimizer(c.learning_rate, c.weight_decay)
        _result, _train, _holdout = run_micro_learning(
            self.model,
            c,
            train_tokens,
            holdout_tokens,
            Telemetry(c.telemetry_file),
            cycle=1,
            before_seen=before,
            before_holdout=before,
            learning_rate=c.learning_rate,
            optimizer=optimizer,
        )
        return _result

    def events(self):
        return [
            json.loads(line)
            for line in Path(self.config.telemetry_file)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

    def test_a_normal_cycle_takes_no_recovery_and_no_damping(self):
        result = self.learn()
        self.assertEqual(result["gradient_recoveries"], 0)
        self.assertEqual(result["gradient_damped"], 0)

    def test_the_stop_threshold_sits_above_the_measured_operating_range(self):
        # The recorded 99.9th percentile is 43.4 and the maximum seen is 99.2.
        # A threshold inside that range is not a stop threshold, it is a
        # description of normal learning.
        self.assertGreaterEqual(self.config.gradient_stop, 100.0)
        self.assertGreater(self.config.gradient_stop, self.config.gradient_alert)

    def test_a_gradient_past_the_threshold_damps_the_step_and_the_cycle_continues(self):
        # The regression. Forcing the threshold to zero trips every step, and
        # the cycle must still finish rather than raising "Unable to stabilize".
        self.config.gradient_stop = 0.0
        result = self.learn()
        self.assertGreater(result["gradient_damped"], 0)
        self.assertEqual(
            result["gradient_recoveries"],
            0,
            "a large gradient must not be treated as a broken one",
        )
        self.assertEqual(
            result["successful_steps"] + result["gradient_damped"],
            self.config.micro_steps,
            "every step must be either applied or damped, never lost",
        )

    def test_a_damped_step_is_reported(self):
        self.config.gradient_stop = 0.0
        self.learn()
        stops = [e for e in self.events() if e["event"] == "gradient_stop"]
        self.assertTrue(stops, "exceeding the threshold must be recorded")
        self.assertIn("stop_threshold", stops[0])

    def test_a_damped_step_is_not_reported_as_a_recovery(self):
        # The two are different facts. Sharing a label would make a log full of
        # ordinary steep first-steps look like a run full of broken weights.
        self.config.gradient_stop = 0.0
        self.learn()
        self.assertEqual(
            [e for e in self.events() if e["event"] == "micro_recovery"],
            [],
        )


class ShortDocumentTests(unittest.TestCase):
    """A document shorter than the context window must still train.

    This crashed for every document of block_size + 1 tokens or fewer, which is
    the smallest span split_tokens produces. Raising block_size from 256 to 512
    widened that window from 256 bytes to 512, so the bug grew with the feature
    instead of being caught by it.
    """

    def test_the_smallest_span_a_document_can_supply_does_not_crash(self):
        block = 32
        for count in range(1, block * 2):
            tokens = torch.arange(count, dtype=torch.long)
            x, y = make_batch(tokens, block, "cpu")
            self.assertEqual(x.shape, (1, block), f"{count} tokens")
            self.assertEqual(y.shape, (1, block), f"{count} tokens")
            self.assertTrue(torch.equal(y[0][:-1], x[0][1:]), f"{count} tokens")

    def test_a_span_is_always_a_contiguous_slice_of_the_document(self):
        block = 8
        tokens = torch.arange(40, dtype=torch.long)
        for _ in range(20):
            x, y = make_batch(tokens, block, "cpu")
            # The target is the token after each input, so y is x shifted on by
            # one. Drawing anything else would train the model on the wrong
            # next token while still producing a plausible-looking loss.
            self.assertTrue(torch.equal(x[0] + 1, y[0]))

    def test_split_tokens_output_can_always_be_batched(self):
        c = make_config(block_size=64, holdout_fraction=0.25)
        tokenizer = tokenizer_module.ByteTokenizer()
        for length in (128, 200, 300, 1000):
            text = "the observatory counts events. " * (length // 12)
            train_tokens, _ = split_tokens(
                text, c.block_size, c.holdout_fraction, tokenizer
            )
            x, _ = make_batch(train_tokens, c.block_size, "cpu")
            self.assertEqual(x.shape[1], c.block_size, f"{length} chars")


class MeasurementTests(TempDirTestCase):
    """Architecture 19 asks for more than loss, so the record carries more."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(11)
        self.config = make_config(device="cpu", block_size=32, holdout_fraction=0.25)
        self.model = Jupiter(
            ModelConfig(block_size=32, vocab_size=256, n_layer=1, n_head=1, n_embd=32)
        ).to("cpu")
        self.tokenizer = tokenizer_module.ByteTokenizer()
        train_tokens, holdout_tokens = split_tokens(
            REPEATED_TEXT,
            self.config.block_size,
            self.config.holdout_fraction,
            self.tokenizer,
        )
        self.train_tokens = train_tokens
        self.holdout_tokens = holdout_tokens

    def cycle(self, steps=8):
        c = self.config
        c.micro_steps = steps
        c.learning_rate = 3e-3
        before = measure_on(self.holdout_tokens, self.model, c)
        optimizer = self.model.configure_optimizer(c.learning_rate, c.weight_decay)
        result, _, _ = run_micro_learning(
            self.model,
            c,
            self.train_tokens,
            self.holdout_tokens,
            Telemetry(str(self.workdir / "t.jsonl")),
            cycle=1,
            before_seen=before,
            before_holdout=before,
            learning_rate=c.learning_rate,
            optimizer=optimizer,
        )
        return before, result

    def test_measure_reports_loss_perplexity_and_accuracy(self):
        result = measure_on(self.holdout_tokens, self.model, self.config)
        self.assertGreater(result["loss"], 0.0)
        self.assertGreater(result["perplexity"], 1.0)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)
        self.assertGreater(result["positions"], 0)

    def test_accuracy_is_a_share_of_real_positions(self):
        measured = measure_on(self.holdout_tokens, self.model, self.config)
        self.assertGreater(measured["positions"], 0)
        self.assertLessEqual(measured["positions"], self.config.block_size * 3)

    def test_perplexity_follows_the_loss(self):
        result = measure_on(self.holdout_tokens, self.model, self.config)
        self.assertAlmostEqual(
            result["perplexity"],
            round(2.718281828 ** result["loss"], 4),
            places=1,
        )

    def test_loss_on_agrees_with_measure_on(self):
        self.assertAlmostEqual(
            loss_on(self.holdout_tokens, self.model, self.config),
            measure_on(self.holdout_tokens, self.model, self.config)["loss"],
        )

    def test_a_learned_model_gets_more_accurate(self):
        before = measure_on(self.holdout_tokens, self.model, self.config)
        self.cycle()
        after = measure_on(self.holdout_tokens, self.model, self.config)
        self.assertGreaterEqual(after["accuracy"], before["accuracy"])
        self.assertLess(after["loss"], before["loss"])

    def test_untouched_weights_have_zero_magnitude(self):
        before = snapshot_model(self.model)
        self.assertEqual(update_magnitude(before, self.model), 0.0)

    def test_update_magnitude_grows_with_the_change(self):
        before = snapshot_model(self.model)
        for parameter in self.model.parameters():
            with torch.no_grad():
                parameter.add_(0.01)
        magnitude = update_magnitude(before, self.model)
        expected = (0.01**2 * sum(p.numel() for p in self.model.parameters())) ** 0.5
        self.assertGreater(magnitude, 0.0)
        self.assertAlmostEqual(magnitude, expected, places=4)

    def test_the_cycle_record_carries_the_architecture_19_measurements(self):
        _, result = self.cycle()
        for field in (
            "holdout_perplexity_before",
            "holdout_perplexity_after",
            "holdout_accuracy_before",
            "holdout_accuracy_after",
            "generalization_gap",
            "update_magnitude",
        ):
            self.assertIn(field, result, field)
        self.assertGreater(result["update_magnitude"], 0.0)


class PrecisionTests(TempDirTestCase):
    """Architecture 36: the configured precision has to be the one used."""

    def test_bfloat16_on_cuda_is_used(self):
        c = make_config(dtype="bfloat16", device="cuda")
        self.assertTrue(train.use_mixed_precision(c))
        self.assertEqual(train.active_precision(c), "bfloat16")

    def test_float32_is_the_default_path(self):
        c = make_config(dtype="float32", device="cuda")
        self.assertFalse(train.use_mixed_precision(c))
        self.assertEqual(train.active_precision(c), "float32")

    def test_bfloat16_on_cpu_falls_back_to_float32(self):
        # A CPU autocast would be slower, not faster, and pretending otherwise
        # would make the recorded precision a lie.
        c = make_config(dtype="bfloat16", device="cpu")
        self.assertFalse(train.use_mixed_precision(c))
        self.assertEqual(train.active_precision(c), "float32")

    def test_float16_is_not_used_because_it_needs_a_loss_scaler(self):
        c = make_config(dtype="float16", device="cuda")
        self.assertFalse(train.use_mixed_precision(c))

    def test_the_context_is_usable_either_way(self):
        with train.autocast_context(make_config(dtype="float32", device="cpu")):
            pass
        with train.autocast_context(make_config(dtype="bfloat16", device="cpu")):
            pass

    def test_the_cycle_records_the_precision_it_actually_used(self):
        c = make_config(
            device="cpu",
            block_size=32,
            micro_steps=2,
            holdout_fraction=0.25,
            learning_rate=3e-3,
            telemetry_file=str(self.workdir / "t.jsonl"),
        )
        model = Jupiter(
            ModelConfig(block_size=32, vocab_size=256, n_layer=1, n_head=1, n_embd=32)
        ).to("cpu")
        train_tokens, holdout_tokens = split_tokens(
            REPEATED_TEXT,
            c.block_size,
            c.holdout_fraction,
            tokenizer_module.ByteTokenizer(),
        )
        before = train.measure_on(holdout_tokens, model, c)
        optimizer = model.configure_optimizer(c.learning_rate, c.weight_decay)
        result, _, _ = train.run_micro_learning(
            model,
            c,
            train_tokens,
            holdout_tokens,
            Telemetry(c.telemetry_file),
            cycle=1,
            before_seen=before,
            before_holdout=before,
            learning_rate=c.learning_rate,
            optimizer=optimizer,
        )
        self.assertEqual(result["precision"], "float32")


class TelemetryTests(TempDirTestCase):
    """Architecture 35: the run must be observable while it happens."""

    def test_events_are_appended_and_readable(self):
        path = self.workdir / "telemetry.jsonl"
        telemetry = Telemetry(path)
        telemetry.write("cycle_start", learning_cycle=1)
        telemetry.write("cycle_end", learning_cycle=1)
        events = read_recent(path, limit=10)
        self.assertEqual(
            [event["event"] for event in events], ["cycle_start", "cycle_end"]
        )
        self.assertEqual(events[0]["learning_cycle"], 1)

    def test_limit_returns_the_newest_events(self):
        path = self.workdir / "telemetry.jsonl"
        telemetry = Telemetry(path)
        for index in range(5):
            telemetry.write("micro_update", step=index)
        events = read_recent(path, limit=2)
        self.assertEqual([event["step"] for event in events], [3, 4])

    def test_reading_a_missing_or_truncated_stream_is_safe(self):
        path = self.workdir / "telemetry.jsonl"
        self.assertEqual(read_recent(path), [])
        Telemetry(path).write("first", value=1)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{half written\n")
        self.assertEqual([event["event"] for event in read_recent(path)], ["first"])
