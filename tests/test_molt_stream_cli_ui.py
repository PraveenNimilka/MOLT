import pytest

from molt_stream import __version__
from molt_stream.cli import TerminalUI, _interactive_interrupt_decision, build_parser
from molt_stream.core.contracts import ProgressEvent


def test_cli_reports_package_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"molt {__version__}"


def test_card_uses_rounded_border_and_status_badge(capsys):
    TerminalUI(enabled=True, color=False).card(
        "Workspace info", [("State", "ready"), ("GPU", "RTX 4060")]
    )
    output = capsys.readouterr().out
    assert "╭─ [ Workspace info ]" in output
    assert "[ READY ]" in output
    assert "╰" in output and "╯" in output
    assert len({len(line) for line in output.splitlines()}) == 1


def test_progress_has_slim_bar_and_live_metrics(capsys):
    ui = TerminalUI(enabled=True, color=False)
    ui.progress(ProgressEvent(
        "step", step=5, total_steps=10, elapsed_seconds=2.0,
        tokens_per_second=155_000.0, loss=2.5,
        vram_bytes=2_000_000_000, gpu_temperature_c=59.0,
        cpu_temperature_c=61.5,
        thermal_state="full-speed", thermal_pause_seconds=0.0,
    ))
    ui.finish_progress()
    output = capsys.readouterr().out
    assert "██████████░░░░░░░░░░" in output
    assert "50.0%" in output
    assert "155,000 tok/s" in output
    assert "loss 2.5000" in output
    assert "GPU 59.0°C" in output
    assert "CPU 61.5°C" in output
    assert "[Normal]" in output


def test_progress_shows_live_cooling_pause(capsys):
    ui = TerminalUI(enabled=True, color=False)
    ui.progress(ProgressEvent(
        "step", step=7, total_steps=10, elapsed_seconds=2.0,
        tokens_per_second=88_400.0, vram_bytes=1_600_000_000,
        gpu_temperature_c=68.5, thermal_state="cooling",
        thermal_pause_seconds=0.015,
    ))
    ui.finish_progress()
    output = capsys.readouterr().out
    assert "[Pause: 15ms]" in output


def test_ui_renders_thermal_abort_badge(capsys):
    ui = TerminalUI(enabled=True, color=False)
    ui.progress(ProgressEvent(
        "thermal_abort", message="Thermal boundary reached; saving checkpoint",
        thermal_state="thermal-abort",
    ))
    output = capsys.readouterr().out
    assert "[ THERMAL_ABORT ]" in output
    assert "saving checkpoint" in output


def test_progress_uses_session_steps_for_resume_eta(capsys):
    ui = TerminalUI(enabled=True, color=False)
    # Resumed at step 500, now at step 501 after 2.0 seconds of session time
    # 499 steps remaining. 1 session step in 2s -> 499 * 2s = 998s = 16m 38s
    ui.progress(ProgressEvent(
        "step", step=501, total_steps=1000, elapsed_seconds=2.0,
        tokens_per_second=2100.0, loss=0.5,
        vram_bytes=2_000_000_000, gpu_temperature_c=65.0,
        initial_step=500,
    ))
    ui.finish_progress()
    output = capsys.readouterr().out
    assert "16:38" in output  # Realistic ~16m ETA, NOT 00:01!


def test_selection_has_numbered_non_tty_fallback(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")
    selected = TerminalUI(enabled=True, color=False).select(
        "Choose", [("First", "one"), ("Second", "two")]
    )
    assert selected == 1
    assert "1. First" in capsys.readouterr().out


def test_interrupt_menu_can_continue_without_checkpoint_question(monkeypatch):
    ui = TerminalUI(enabled=True, color=False)
    answers = iter([False])
    monkeypatch.setattr(ui, "confirm", lambda *_args, **_kwargs: next(answers))
    assert _interactive_interrupt_decision(ui) == (False, True)


def test_interrupt_menu_can_stop_without_saving(monkeypatch):
    ui = TerminalUI(enabled=True, color=False)
    answers = iter([True, False])
    monkeypatch.setattr(ui, "confirm", lambda *_args, **_kwargs: next(answers))
    assert _interactive_interrupt_decision(ui) == (True, False)
