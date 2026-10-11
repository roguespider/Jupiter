"""Tests for the versioned architecture record and the migration guard.

Architecture 30 requires a saved instance to carry enough information to
determine the architecture, and Architecture 43 requires separate versions that
are not assumed to advance together. These tests check the record is versioned,
that an older unversioned file is still readable, and -- most importantly --
that a checkpoint trained with a different shape is refused with an explanation
rather than a tensor-size error.
"""

import json
import unittest
from pathlib import Path

import torch
import train
from model import Jupiter, ModelConfig
from safetensors import safe_open

import config
from tests.support import TempDirTestCase, make_config


class ArchitectureRecordTests(TempDirTestCase):
    def test_the_envelope_is_versioned_and_wraps_the_shape(self):
        c = make_config(block_size=512)
        envelope = config.architecture_envelope(c)
        self.assertEqual(envelope["architecture_version"], config.ARCHITECTURE_VERSION)
        self.assertEqual(envelope["config"]["block_size"], 512)
        self.assertIn("written", envelope)

    def test_a_saved_record_reads_back_to_the_same_shape(self):
        c = make_config(block_size=512, n_embd=256)
        path = self.workdir / "Jupiter.architecture.json"
        config.save_architecture(c, path)
        stored, version = config.read_architecture(path)
        self.assertEqual(version, config.ARCHITECTURE_VERSION)
        self.assertEqual(stored["block_size"], 512)
        self.assertEqual(stored["n_embd"], 256)

    def test_an_unversioned_record_is_still_readable(self):
        # Files written before versioning stored the model fields at the top
        # level. They must keep working, with the version reported as unknown
        # rather than guessed at.
        path = self.write(
            "Jupiter.architecture.json",
            json.dumps(
                {
                    "block_size": 256,
                    "vocab_size": 256,
                    "n_layer": 4,
                    "n_head": 4,
                    "n_embd": 256,
                    "dropout": 0.0,
                    "bias": True,
                }
            ),
        )
        stored, version = config.read_architecture(path)
        self.assertIsNone(version)
        self.assertEqual(stored["block_size"], 256)

    def test_a_record_with_no_model_fields_is_refused(self):
        path = self.write("bad.json", json.dumps({"notes": "not an architecture"}))
        with self.assertRaises(ValueError):
            config.read_architecture(path)

    def test_a_non_object_record_is_refused(self):
        # TypeError, not ValueError. The file parsed as JSON and the document is
        # the wrong shape, which is a version problem rather than damage, and a
        # caller catching ValueError to mean "this file is broken" should not
        # swallow it.
        path = self.write("bad.json", "[1, 2, 3]")
        with self.assertRaises(TypeError):
            config.read_architecture(path)

    def test_shape_ignores_fields_that_do_not_change_tensor_sizes(self):
        first = config.shape_of({"block_size": 512, "dropout": 0.0})
        second = config.shape_of({"block_size": 512, "dropout": 0.5})
        self.assertEqual(first, second)
        self.assertNotEqual(
            config.shape_of({"block_size": 512}),
            config.shape_of({"block_size": 256}),
        )


class CheckpointShapeTests(TempDirTestCase):
    def build(self, block_size=64):
        c = make_config(
            device="cpu", block_size=block_size, out_dir=str(self.workdir / "out")
        )
        model = Jupiter(
            ModelConfig(
                block_size=block_size, vocab_size=256, n_layer=1, n_head=1, n_embd=32
            )
        ).to("cpu")
        return c, model

    def test_a_saved_checkpoint_records_its_architecture_and_version(self):
        c, model = self.build()
        path = train.save_checkpoint(model, c, cycle=2)
        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata()
        self.assertEqual(metadata["architecture_version"], config.ARCHITECTURE_VERSION)
        recorded = json.loads(metadata["architecture"])
        self.assertEqual(recorded["block_size"], 64)

    def test_checkpoint_shape_reads_back_what_was_written(self):
        c, model = self.build(block_size=128)
        path = train.save_checkpoint(model, c, cycle=1)
        self.assertEqual(train.checkpoint_shape(path)["block_size"], 128)

    def test_a_matching_checkpoint_loads(self):
        c, model = self.build()
        path = train.save_checkpoint(model, c, cycle=1)
        config.save_architecture(c, Path(c.out_dir) / "Jupiter.architecture.json")
        self.assertTrue(
            train.attach_checkpoint(
                model, [path], Path(c.out_dir) / "Jupiter.architecture.json"
            )
        )

    def test_a_checkpoint_smaller_than_the_model_is_grown_not_archived(self):
        # The migration guard changed when growth was implemented. A shape that
        # got BIGGER is a model that grew: the context window is rows of the
        # position embedding, and a 64-token checkpoint carries real weights
        # into a 512-token model. Archiving it threw away everything it had
        # learned in order to get a shape that did not need restarting for.
        #
        # The architecture record is written with the same hidden width the
        # checkpoint was trained at, because this test is about the context
        # window growing and nothing else. A record that also widened the model
        # would be testing two growths at once and would not say which one broke.
        c, narrow = self.build(block_size=64)
        path = train.save_checkpoint(narrow, c, cycle=1)
        wide = Jupiter(
            ModelConfig(block_size=512, vocab_size=256, n_layer=1, n_head=1, n_embd=32)
        ).to("cpu")
        architecture = Path(c.out_dir) / "Jupiter.architecture.json"
        widened = make_config(device="cpu", block_size=512)
        widened.n_embd = 32
        widened.n_head = 1
        config.save_architecture(widened, architecture)
        self.assertTrue(train.attach_checkpoint(wide, [path], architecture))
        self.assertTrue(path.exists(), "a grown checkpoint must not be archived")
        self.assertTrue(
            torch.equal(
                wide.position_embedding.weight[:64], narrow.position_embedding.weight
            ),
            "the learned positions must survive the growth",
        )

    def test_a_checkpoint_larger_than_the_model_is_still_archived(self):
        # The other direction, which is the case the guard was always for. A
        # checkpoint that is bigger genuinely does not fit, and loading it would
        # either raise or, worse, pair the wrong numbers together silently.
        c, _ = self.build()
        path = train.save_checkpoint(
            Jupiter(
                ModelConfig(
                    block_size=512, vocab_size=256, n_layer=1, n_head=1, n_embd=32
                )
            ).to("cpu"),
            c,
            cycle=1,
        )
        narrow = Jupiter(
            ModelConfig(block_size=64, vocab_size=256, n_layer=1, n_head=1, n_embd=32)
        ).to("cpu")
        architecture = Path(c.out_dir) / "Jupiter.architecture.json"
        config.save_architecture(make_config(device="cpu", block_size=64), architecture)
        self.assertFalse(train.attach_checkpoint(narrow, [path], architecture))
        self.assertFalse(path.exists(), "the incompatible checkpoint must be moved")
        archived = list((Path(c.out_dir) / "archive").glob("incompatible-*"))
        self.assertEqual(len(archived), 1)
        kept = archived[0] / path.name
        self.assertTrue(kept.exists(), "the checkpoint must be preserved, not deleted")
        note = (archived[0] / "reason.txt").read_text(encoding="utf-8")
        self.assertIn("block_size", note)
        self.assertIn("fresh training run", note)

    def test_the_tokenizer_decides_the_vocabulary_not_the_saved_record(self):
        # A saved architecture trained on bytes must not be allowed to build a
        # 256 entry embedding table for a tokenizer that emits ids above 255.
        # The failure that prevents is an out-of-range index inside a forward
        # pass, which says nothing about the real cause.
        c = make_config(device="cpu", out_dir=str(self.workdir / "out"))
        architecture = Path(c.out_dir) / "Jupiter.architecture.json"
        config.save_architecture(make_config(byte_vocab_size=256), architecture)
        model, _ = train.build_model(c, vocab_size=5256)
        self.assertEqual(model.config.vocab_size, 5256)

    def test_the_saved_record_still_decides_the_window(self):
        c = make_config(device="cpu", out_dir=str(self.workdir / "out"))
        architecture = Path(c.out_dir) / "Jupiter.architecture.json"
        config.save_architecture(
            make_config(block_size=256, byte_vocab_size=256), architecture
        )
        model, _ = train.build_model(c, vocab_size=256)
        self.assertEqual(model.config.block_size, 256)

    def test_no_checkpoint_means_a_fresh_start(self):
        c, model = self.build()
        missing = Path(c.out_dir) / "absent.safetensors"
        architecture = Path(c.out_dir) / "Jupiter.architecture.json"
        self.assertFalse(train.attach_checkpoint(model, [missing], architecture))

    def test_an_unreadable_checkpoint_shape_falls_back_to_loading(self):
        # The pre-check is a courtesy; if metadata cannot be read the load
        # itself is still allowed to decide.
        c, model = self.build()
        self.assertIsNone(
            train.checkpoint_shape(Path(c.out_dir) / "absent.safetensors")
        )


class ResumeStateTests(TempDirTestCase):
    """Where a run continues from, as opposed to what it publishes.

    The published checkpoint is written only when a cycle is ACCEPTed, and it is
    what the conversation loads. That is Architecture 20 working correctly, and
    it used to have a consequence nobody intended: a run that never earns an
    ACCEPT writes no weights at all, so the next run finds nothing to resume
    from, starts from random initialization, and repeats. Measured over one
    corpus, the holdout loss before every cycle sat at ln(vocab_size) to within
    a hundredth, which is exactly the loss of a randomly initialized model. The
    resume state exists so that learning accumulates while the verdict that
    would publish a model is still being earned.
    """

    def build(self, block_size=64):
        c = make_config(
            device="cpu",
            block_size=block_size,
            out_dir=str(self.workdir / "out"),
            resume_checkpoint=str(self.workdir / "out" / "Jupiter_resume.safetensors"),
        )
        model = Jupiter(
            ModelConfig(
                block_size=block_size, vocab_size=256, n_layer=1, n_head=1, n_embd=32
            )
        ).to("cpu")
        config.save_architecture(c, Path(c.out_dir) / "Jupiter.architecture.json")
        return c, model

    def published(self, c):
        return Path(c.out_dir) / train.PUBLISHED_CHECKPOINT

    def resume(self, c):
        return Path(c.resume_checkpoint)

    def test_the_published_checkpoint_stays_where_the_conversation_reads_it(self):
        # sample.py loads this exact filename, so it must not move.
        c, model = self.build()
        path = train.save_checkpoint(model, c, cycle=1)
        self.assertEqual(path.name, "Jupiter_latest.safetensors")
        self.assertEqual(path, self.published(c))

    def test_resume_state_is_written_to_its_own_file(self):
        c, model = self.build()
        path = train.save_checkpoint(model, c, cycle=1, path=self.resume(c))
        self.assertEqual(path, self.resume(c))
        self.assertTrue(path.exists())
        self.assertNotEqual(path, self.published(c))

    def test_each_file_records_the_role_it_plays(self):
        # A reader looking at a checkpoint should be able to tell whether it is
        # the model that was judged or the state a run happens to be resuming.
        c, model = self.build()
        train.save_checkpoint(model, c, cycle=1)
        train.save_checkpoint(model, c, cycle=1, path=self.resume(c))
        for path, role in (
            (self.published(c), "published"),
            (self.resume(c), "resume"),
        ):
            with safe_open(str(path), framework="pt") as handle:
                self.assertEqual(handle.metadata()["role"], role)

    def test_writing_resume_state_leaves_the_published_checkpoint_alone(self):
        # The ACCEPT gate is the whole point of the published file. Saving
        # resume state must not quietly publish a model that was not judged.
        c, model = self.build()
        published = train.save_checkpoint(model, c, cycle=1)
        before = published.read_bytes()
        train.save_checkpoint(model, c, cycle=2, path=self.resume(c))
        self.assertEqual(published.read_bytes(), before)

    def test_a_run_resumes_from_the_resume_state_rather_than_starting_over(self):
        c, model = self.build()
        train.save_checkpoint(model, c, cycle=1, path=self.resume(c))
        fresh = Jupiter(model.config)
        self.assertTrue(
            train.attach_checkpoint(
                fresh,
                [self.resume(c), self.published(c)],
                Path(c.out_dir) / "Jupiter.architecture.json",
            )
        )
        for name, original in model.state_dict().items():
            self.assertTrue(
                torch.equal(original, fresh.state_dict()[name]),
                f"tensor {name} was not carried over from the resume state",
            )

    def test_a_missing_resume_state_falls_back_to_the_published_checkpoint(self):
        c, model = self.build()
        train.save_checkpoint(model, c, cycle=1)
        fresh = Jupiter(model.config)
        self.assertTrue(
            train.attach_checkpoint(
                fresh,
                [self.resume(c), self.published(c)],
                Path(c.out_dir) / "Jupiter.architecture.json",
            )
        )

    def test_a_resume_state_smaller_than_the_model_is_grown(self):
        # The resume state is behind the published one, so it is the first thing
        # tried. When the model has grown since, it is a prefix of the model
        # rather than a mismatched shape, and its learned weights are worth more
        # than falling through to the older published model.
        c, _ = self.build()
        train.save_checkpoint(
            Jupiter(
                ModelConfig(
                    block_size=32, vocab_size=256, n_layer=1, n_head=1, n_embd=32
                )
            ).to("cpu"),
            c,
            cycle=1,
            path=self.resume(c),
        )
        grown = Jupiter(
            ModelConfig(block_size=64, vocab_size=256, n_layer=1, n_head=1, n_embd=32)
        ).to("cpu")
        self.assertTrue(
            train.attach_checkpoint(
                grown,
                [self.resume(c)],
                Path(c.out_dir) / "Jupiter.architecture.json",
            )
        )
        self.assertTrue(self.resume(c).exists(), "a grown resume state is not archived")

    def test_a_resume_state_too_large_falls_through_to_the_published_one(self):
        # The other direction. Giving up entirely would throw away a perfectly
        # good published model, so the too-large file is archived with a note
        # and the next candidate is tried.
        c, _ = self.build()
        train.save_checkpoint(
            Jupiter(
                ModelConfig(
                    block_size=128, vocab_size=256, n_layer=1, n_head=1, n_embd=32
                )
            ).to("cpu"),
            c,
            cycle=1,
            path=self.resume(c),
        )
        usable = Jupiter(
            ModelConfig(block_size=64, vocab_size=256, n_layer=1, n_head=1, n_embd=32)
        ).to("cpu")
        train.save_checkpoint(usable, c, cycle=1)
        fresh = Jupiter(usable.config)
        self.assertTrue(
            train.attach_checkpoint(
                fresh,
                [self.resume(c), self.published(c)],
                Path(c.out_dir) / "Jupiter.architecture.json",
            )
        )
        self.assertFalse(
            self.resume(c).exists(), "an unusable resume state must be archived"
        )
        archived = list((Path(c.out_dir) / "archive").glob("incompatible-*"))
        self.assertEqual(len(archived), 1)
        self.assertTrue((archived[0] / self.resume(c).name).exists())
        self.assertTrue(self.published(c).exists())

    def test_with_nothing_usable_it_still_starts_from_random_initialization(self):
        c, model = self.build()
        fresh = Jupiter(model.config)
        self.assertFalse(
            train.attach_checkpoint(
                fresh,
                [self.resume(c), self.published(c)],
                Path(c.out_dir) / "Jupiter.architecture.json",
            )
        )

    def test_the_configured_resume_path_is_not_the_published_path(self):
        # If these ever collide, the ACCEPT gate silently stops meaning anything,
        # because every cycle would overwrite the model the conversation loads.
        c = make_config()
        self.assertNotEqual(Path(c.resume_checkpoint).name, train.PUBLISHED_CHECKPOINT)
        self.assertNotEqual(
            Path(c.resume_checkpoint), Path(c.out_dir) / "Jupiter_latest.safetensors"
        )


class VocabularySourceTests(TempDirTestCase):
    """The tokenizer decides the vocabulary, not the seed config.

    The field was called vocab_size and read 256 while the model was 5256 wide,
    because the tokenizer overrides it at build time. Reading it looked exactly
    like the BPE migration had not happened, which is the opposite of the truth:
    out/Jupiter.architecture.json records 5256 and the checkpoint's lm_head is
    5256 rows. It is now byte_vocab_size, which says what it is, and these tests
    say what it is for: a fallback for a shape built with no tokenizer to
    consult, and nothing else.
    """

    def test_the_config_no_longer_claims_to_be_the_model_vocabulary(self):
        c = make_config()
        self.assertEqual(c.byte_vocab_size, 256)
        self.assertFalse(
            hasattr(c, "vocab_size"),
            "Config.vocab_size was removed so it cannot be read as the model's "
            "vocabulary; only ModelConfig.vocab_size is authoritative",
        )

    def test_the_tokenizer_vocabulary_wins_over_the_byte_fallback(self):
        c = make_config(device="cpu", out_dir=str(self.workdir / "out"))
        model, _ = train.build_model(c, vocab_size=5256)
        self.assertEqual(
            model.config.vocab_size,
            5256,
            "the tokenizer decides the vocabulary, not the config default",
        )

    def test_the_byte_fallback_is_used_only_when_no_vocabulary_is_given(self):
        c = make_config(device="cpu", out_dir=str(self.workdir / "out"))
        self.assertEqual(config.architecture_dict(c)["vocab_size"], 256)

    def test_the_default_configuration_asks_for_bpe(self):
        self.assertEqual(make_config().tokenizer, "bpe")

    def test_the_architecture_record_reports_the_real_vocabulary(self):
        # The record and the checkpoint are the authoritative answer to "how wide
        # is this model", which is the question the old field name invited.
        c = make_config(device="cpu", out_dir=str(self.workdir / "out"))
        path = self.workdir / "out" / "Jupiter.architecture.json"
        config.save_architecture(c, path, vocab_size=5256)
        written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written["config"]["vocab_size"], 5256)
        self.assertNotEqual(written["config"]["vocab_size"], c.byte_vocab_size)


class BlockSizeTests(unittest.TestCase):
    def test_the_conversation_window_is_512_tokens(self):
        self.assertEqual(config.Config().block_size, 512)


class GenerationTests(unittest.TestCase):
    """generate() samples from the last token it was given.

    A zero-length context has no last token. That used to fail as an IndexError
    about tensor dimensions, which names the symptom and not the mistake, and
    left whoever hit it looking at the model rather than at the caller that
    built the empty context. It is refused here, where the reason can be said.
    """

    def model(self):
        return Jupiter(
            ModelConfig(block_size=16, vocab_size=32, n_layer=1, n_head=2, n_embd=8)
        )

    def test_an_empty_context_is_refused_with_a_readable_reason(self):
        model = self.model()
        with self.assertRaises(ValueError) as caught:
            model.generate(torch.zeros((1, 0), dtype=torch.long), max_new_tokens=2)
        message = str(caught.exception)
        self.assertIn("at least one token", message)
        # The shape is named so the caller can see what it actually passed.
        self.assertIn("(1, 0)", message)

    def test_a_sequence_that_is_not_a_batch_of_rows_is_refused(self):
        model = self.model()
        with self.assertRaises(ValueError):
            model.generate(torch.zeros((4,), dtype=torch.long), max_new_tokens=2)

    def test_one_token_of_context_is_enough(self):
        model = self.model()
        result = model.generate(torch.tensor([[7]]), max_new_tokens=3)
        self.assertEqual(result.shape[1], 4)

    def test_generation_extends_the_context_rather_than_replacing_it(self):
        model = self.model()
        start = torch.tensor([[1, 2, 3]])
        result = model.generate(start, max_new_tokens=2)
        self.assertTrue(torch.equal(result[0, :3], start[0]))

    def test_a_failure_does_not_leave_the_model_in_eval_mode(self):
        # The training flag is restored in a finally. Restored after the loop
        # instead, a failure part way through would leave the model silently
        # switched to eval for whatever happened to run next.
        model = self.model()
        model.train()
        original = model.forward

        def explode(*args, **kwargs):
            raise RuntimeError("forward pass failed")

        model.forward = explode
        self.addCleanup(setattr, model, "forward", original)
        with self.assertRaises(RuntimeError):
            model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4)
        self.assertTrue(
            model.training, "training mode must be restored after a failure"
        )


class ApprovedGrowthRecordTests(TempDirTestCase):
    """Recording a shape a human already approved is not a second decision.

    A growth is authorised through the expansion gate, and the next run builds
    the larger model. Writing that model into the architecture record used to
    ask the architecture gate for a separate signature, which meant the approval
    a person had given did not let the thing they approved happen: the run grew
    the model and then stopped, leaving the new weights on disk and the record
    still describing the old shape. The next run then built the small model
    again and the growth was undone by a gate nobody was asked about.
    """

    def setUp(self):
        super().setUp()
        self.config = make_config(
            out_dir=str(self.workdir / "out"),
            expansion_state=str(self.workdir / "out" / "expansion.json"),
            device="cpu",
        )
        self.record = Path(self.config.out_dir) / "Jupiter.architecture.json"

    def grant(self, *domains):
        import control

        original = control.granted_domains
        control.granted_domains = lambda environ=None: set(domains)
        self.addCleanup(setattr, control, "granted_domains", original)

    def approve(self, to_layers=6, to_embd=320):
        import expansion

        expansion.load
        expansion.save(
            self.config,
            {
                "capacity_growth": {
                    "approved": True,
                    "from_layers": 5,
                    "to_layers": to_layers,
                    "from_embd": 256,
                    "to_embd": to_embd,
                }
            },
        )

    def test_a_growth_the_record_matches_does_not_ask_again(self):
        self.grant("expansion")
        self.approve()
        config.save_architecture(
            self.config,
            self.record,
            64,
            shape={"n_layer": 6, "n_embd": 320},
        )
        stored, _version = config.read_architecture(self.record)
        self.assertEqual(stored["n_layer"], 6)
        self.assertEqual(stored["n_embd"], 320)

    def seed_record(self, n_layer=5, n_embd=256):
        """Write the record the growth is about to change.

        The gate only fires on a change, so a test that never wrote a first
        record was testing nothing: there was no previous shape to differ from.
        """
        self.grant("architecture")
        config.save_architecture(
            self.config,
            self.record,
            64,
            shape={"n_layer": n_layer, "n_embd": n_embd},
        )

    def test_a_change_nobody_approved_is_still_refused(self):
        # The exemption is for the approved growth and nothing else.
        import control

        self.seed_record()
        self.grant("expansion")
        self.approve(to_layers=6, to_embd=320)
        with self.assertRaises(control.ControlDenied):
            config.save_architecture(
                self.config,
                self.record,
                64,
                shape={"n_layer": 9, "n_embd": 512},
            )

    def test_an_approved_growth_without_the_grant_is_still_refused(self):
        # Someone able to write the expansion state file must not thereby be able
        # to change a model in use with no control at all. The grant is part of
        # the exemption, not an optional extra.
        import control

        self.seed_record()
        self.grant()
        self.approve(to_layers=6, to_embd=320)
        with self.assertRaises(control.ControlDenied):
            config.save_architecture(
                self.config,
                self.record,
                64,
                shape={"n_layer": 6, "n_embd": 320},
            )

    def test_a_change_without_an_approved_growth_is_still_refused(self):
        import control

        self.seed_record()
        self.grant("expansion")
        with self.assertRaises(control.ControlDenied):
            config.save_architecture(
                self.config,
                self.record,
                64,
                shape={"n_layer": 7, "n_embd": 384},
            )
