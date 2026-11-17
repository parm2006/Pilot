import io
import json
import queue
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock

from pilot_voice import Actuator, Router, VoiceEngine, compile_command, load_commands, parse_combo


class VoiceLoggingTests(unittest.TestCase):
    def setUp(self):
        table = {
            "copy": compile_command("copy", {"keys": "ctrl+c"}),
            "resume listener": compile_command("resume listener", {"builtin": "resume_listener"}),
        }
        self.events = queue.Queue()
        self.engine = VoiceEngine(Router(table), self.events)
        self.engine._transcribe = Mock(return_value="something unrelated")

    def utterance(self, text, paused=False):
        output = io.StringIO()
        with redirect_stdout(output):
            self.engine._on_utterance({"text": text}, b"audio", paused)
        return output.getvalue()

    def test_exact_command_keeps_fast_path(self):
        output = self.utterance("copy")
        self.assertIn("command guess: 'copy'", output)
        self.assertIn("(exact)", output)
        self.engine._transcribe.assert_not_called()
        self.assertEqual(self.events.get_nowait()[1].command.phrase, "copy")

    def test_verbose_logs_free_transcript_for_exact_command(self):
        self.engine.verbose = True
        self.engine._transcribe.return_value = "copy that"
        output = self.utterance("copy")
        self.assertIn("transcript: 'copy that'", output)
        self.engine._transcribe.assert_called_once()
        self.assertEqual(self.events.qsize(), 1)

    def test_unknown_and_empty_grammar_results_are_transcribed(self):
        for text in ("[unk]", ""):
            with self.subTest(text=text):
                self.engine._transcribe.reset_mock()
                output = self.utterance(text)
                self.assertIn("transcript: 'something unrelated'", output)
                self.assertIn("no match", output)
                self.engine._transcribe.assert_called_once()
                self.assertTrue(self.events.empty())

    def test_empty_free_transcript_is_visible(self):
        self.engine._transcribe.return_value = ""
        self.assertIn("<no words recognized>", self.utterance(""))
        self.assertTrue(self.events.empty())

    def test_empty_grammar_can_still_resolve_a_transcribed_command(self):
        self.engine._transcribe.return_value = "copy"
        self.utterance("")
        self.assertEqual(self.events.get_nowait()[1].command.phrase, "copy")

    def test_paused_listener_does_not_transcribe_or_run_normal_commands(self):
        self.engine.verbose = True
        self.assertIn("listener paused", self.utterance("copy", paused=True))
        self.engine._transcribe.assert_not_called()
        self.assertTrue(self.events.empty())

    def test_paused_listener_still_accepts_resume(self):
        self.utterance("resume listener", paused=True)
        self.assertEqual(self.events.get_nowait()[1].command.builtin, "resume_listener")
        self.engine._transcribe.assert_not_called()

    def test_spoken_pilot_controls_use_builtin_actions(self):
        table, _, errors = load_commands(Path(__file__).with_name("commands.json"))
        self.assertEqual(errors, [])
        actuator = Actuator(Mock(), dry=True)
        for phrase, action in {
            "center cursor": "recenter",
            "recalibrate": "recal",
            "calibrate clicks": "wink",
            "switch cursor mode": "control",
            "change tracking input": "input",
            "pause pilot": "pause_pilot",
            "resume pilot": "resume_pilot",
            "pause listener": "pause_listener",
            "resume listener": "resume_listener",
            "quit pilot": "stop_all",
        }.items():
            with self.subTest(phrase=phrase):
                result = Router(table).route(phrase)
                self.assertEqual(actuator.do(result.command), action)
                actuator.pg.hotkey.assert_not_called()

    def test_control_plus_every_letter_and_spoken_letter_names(self):
        for letter in "abcdefghijklmnopqrstuvwxyz":
            self.assertEqual(parse_combo(f"control {letter}"), ["ctrl", letter])
        for phrase, letter in {"control see": "c", "control vee": "v", "ctrl double you": "w"}.items():
            self.assertEqual(parse_combo(phrase), ["ctrl", letter])

    def test_complicated_shortcuts_are_rejected_without_fuzzy_matching(self):
        router = Router({})
        router.fuzzy = Mock()
        for phrase in ("control shift t", "control alt delete", "control one", "control page up", "alt tab"):
            self.assertIsNone(parse_combo(phrase))
            self.assertIsNone(router.route(phrase))
        router.fuzzy.assert_not_called()

    def test_control_c_runs_without_a_free_transcription_or_ollama(self):
        self.utterance("control see")
        self.engine._transcribe.assert_not_called()
        _, result = self.events.get_nowait()
        self.assertEqual(result.how, "combo")
        pg = Mock()
        Actuator(pg).do(result.command)
        pg.hotkey.assert_called_once_with("ctrl", "c", interval=0.0)

    def test_control_letters_are_in_active_grammar_but_not_paused_grammar(self):
        self.assertIn("control c", json.loads(self.engine._grammar()))
        self.assertNotIn("control c", json.loads(self.engine._grammar(only_control=True)))


if __name__ == "__main__":
    unittest.main()
