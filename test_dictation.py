import io
import queue
import unittest
from concurrent.futures import Future
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pilot_dictation import FRAME_BYTES, SpeechBuffer
from pilot_voice import Router, VoiceEngine, VoiceSystem, compile_command

SPEECH = b"\x01\x00" * (FRAME_BYTES // 2)
SILENCE = bytes(FRAME_BYTES)


class SpeechBufferTests(unittest.TestCase):
    def buffer(self, gap=0.8):
        return SpeechBuffer(gap=gap, vad=SimpleNamespace(is_speech=lambda frame, _: frame == SPEECH))

    def test_arbitrary_microphone_blocks_and_pause_boundary(self):
        buffer = self.buffer()
        audio = SPEECH * 20 + SILENCE * 39
        for offset in range(0, len(audio), 4000):
            self.assertIsNone(buffer.feed(audio[offset:offset + 4000]))
        self.assertEqual(buffer.feed(SILENCE), SPEECH * 20 + SILENCE * 40)

    def test_brief_pause_does_not_cut_sentence(self):
        buffer = self.buffer()
        self.assertIsNone(buffer.feed(SPEECH * 10 + SILENCE * 20 + SPEECH * 10))
        self.assertIsNotNone(buffer.feed(SILENCE * 40))

    def test_silence_times_out_without_transcription(self):
        self.assertEqual(self.buffer().feed(SILENCE * 500), b"")

    def test_click_noise_is_not_a_dictated_phrase(self):
        self.assertEqual(self.buffer().feed(SPEECH * 2 + SILENCE * 40), b"")

    def test_invalid_pause_is_rejected(self):
        with self.assertRaises(ValueError):
            self.buffer(gap=0)


class DictationTests(unittest.TestCase):
    def setUp(self):
        self.events = queue.Queue()
        self.engine = VoiceEngine(Router({}), self.events)
        self.engine._reset = Mock()
        self.engine.whisper = Mock()
        self.addCleanup(self.engine.stop)
        self.output = io.StringIO()
        self.logging = redirect_stdout(self.output)
        self.logging.__enter__()
        self.addCleanup(self.logging.__exit__, None, None, None)

    def ready_result(self, text="Copy this paragraph. Save it."):
        self.engine._dictation_target = 123
        self.engine._dictation_future = Future()
        self.engine._dictation_future.set_result(text)
        self.engine._poll_dictation()

    def system(self, dry=False):
        system = object.__new__(VoiceSystem)
        system.engine, system.events, system.hand = self.engine, self.events, None
        system.act = Mock(dry=dry)
        system._poll_config = Mock()
        return system

    @patch("pilot_voice.foreground_window", return_value=123)
    def test_dictate_switches_mode_without_queuing_a_command(self, _):
        cmd = compile_command("dictate", {"builtin": "dictate", "fuzzy": False})
        self.engine.router.table["dictate"] = cmd
        self.engine._on_utterance({"text": "dictate"}, b"trigger audio", False)
        self.assertIsNotNone(self.engine._dictation_buffer)
        self.assertEqual(self.engine.status, "DICTATING")
        self.assertTrue(self.events.empty())
        self.assertEqual(self.engine._dictation_buffer.frames, [])

    def test_dictation_audio_bypasses_command_router(self):
        self.engine._dirty = False
        self.engine._dictation_buffer = Mock(feed=Mock(return_value=None))
        self.engine._on_utterance = Mock()
        self.engine._step(b"speech containing copy or delete")
        self.engine._on_utterance.assert_not_called()

    def test_result_is_delivered_once_as_text(self):
        self.ready_result()
        self.engine._poll_dictation()
        self.assertEqual(self.events.qsize(), 1)
        kind, result = self.events.get_nowait()
        self.assertEqual(kind, "dictation")
        self.assertIn("Copy this paragraph", result[0])

    @patch("pilot_voice.type_text")
    @patch("pilot_voice.foreground_window", return_value=123)
    def test_result_types_literal_unicode_text_and_returns_to_commands(self, _, type_text):
        self.ready_result("Café — copy this.")
        self.system().pump(None, 0)
        type_text.assert_called_once_with("Café — copy this.")
        self.assertIsNone(self.engine._dictation_future)
        self.assertEqual(self.engine.status, "LISTENING")

    @patch("pilot_voice.type_text")
    @patch("pilot_voice.foreground_window", return_value=456)
    def test_focus_change_does_not_type_into_another_window(self, _, type_text):
        self.ready_result()
        self.system().pump(None, 0)
        type_text.assert_not_called()

    @patch("pilot_voice.type_text")
    def test_cancel_discards_already_queued_text(self, type_text):
        self.ready_result()
        self.engine.cancel_dictation()
        self.system().pump(None, 0)
        type_text.assert_not_called()

    def test_calibration_event_reset_cancels_pending_dictation(self):
        self.ready_result()
        self.system().clear_events()
        self.assertTrue(self.events.empty())
        self.assertIsNone(self.engine._dictation_future)

    @patch("pilot_voice.type_text")
    def test_dry_run_only_prints_transcript(self, type_text):
        self.ready_result()
        self.system(dry=True).pump(None, 0)
        type_text.assert_not_called()
        self.assertIn("DRY RUN dictation", self.output.getvalue())

    def test_cancel_during_capture_never_submits_audio(self):
        buffer = Mock()
        buffer.feed.side_effect = lambda data: (self.engine.cancel_dictation(), SPEECH)[1]
        self.engine._dictation_buffer = buffer
        self.engine._record_dictation(SPEECH)
        self.assertIsNone(self.engine._dictation_future)

    def test_whisper_failure_recovers_command_mode(self):
        self.engine._dictation_future = Future()
        self.engine._dictation_future.set_exception(RuntimeError("test inference failure"))
        self.engine._poll_dictation()
        self.assertIsNone(self.engine._dictation_future)
        self.assertIn("transcription failed", self.output.getvalue())

    def test_hand_mute_cancels_captured_audio(self):
        self.engine._dirty = False
        self.engine._dictation_buffer = Mock()
        self.engine.muted = True
        self.engine._step(SPEECH)
        self.assertIsNone(self.engine._dictation_buffer)
        self.assertEqual(self.engine.status, "MUTED (hand)")

    def test_pause_cancels_captured_audio(self):
        self.engine._dirty = False
        self.engine._dictation_buffer = Mock()
        self.engine.paused = True
        self.engine.rec_paused = Mock()
        self.engine.rec_paused.AcceptWaveform.return_value = False
        self.engine.rec_paused.PartialResult.return_value = '{"partial": ""}'
        self.engine._step(SILENCE)
        self.assertIsNone(self.engine._dictation_buffer)
        self.assertEqual(self.engine.status, "PAUSED")


if __name__ == "__main__":
    unittest.main()
