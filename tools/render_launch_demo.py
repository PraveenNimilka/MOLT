"""Render the public 75-second MOLT launch demo from its checked-in log."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"
WIDTH, HEIGHT = 1280, 720
SLIDES = (
    (8, "MOLT LOCAL TRAINING", ("Real 75-second workflow demonstration", "Source-available under PolyForm Shield 1.0.0")),
    (9, "HARDWARE", ("NVIDIA RTX 4060 Laptop GPU  |  8,188 MiB", "Driver 610.62  |  Windows 11", "Python 3.12.13  |  PyTorch 2.8.0+cu128")),
    (9, "PINNED INPUTS", ("Qwen/Qwen2.5-0.5B-Instruct", "Immutable model revision: 7ae5576...a775", "30-record project-authored demonstration dataset")),
    (8, "FIT TEST", ("Two real forward / backward / optimizer steps", "Result: COMPLETED", "Only then does the full run start")),
    (12, "TRAINING", ("20 updates  |  context 128  |  batch 1", "Joint static CUDA graph  |  fused FP32 AdamW", "Validation NLL: 3.859000  ->  1.267150", "Committed compute: 2,802.98 token/s")),
    (10, "TELEMETRY", ("Peak GPU temperature: 65 C", "PyTorch allocation peak: 885,637,632 bytes", "NVML board peak: 1,968,504,832 bytes", "GPU board energy: 157.070 J")),
    (9, "EXPORT", ("Hugging Face PEFT safetensors adapter", "Verified adapter SHA-256: b14afe9c...19a73", "Base weights and dataset are not included")),
    (10, "ADAPTER RELOAD", ("Prompt: Return only the MOLT demo status code.", "Response: THERMAL-READY", "Full recording data and sanitized logs are linked")),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows = Path("C:/Windows/Fonts")
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(windows / name), size)


def frame(title: str, lines: tuple[str, ...], destination: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#07111f")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((50, 45, 1230, 675), radius=28, fill="#0d1b2d", outline="#23d18b", width=3)
    draw.text((90, 82), "MOLT", font=font(28, True), fill="#23d18b")
    draw.text((90, 145), title, font=font(48, True), fill="#f6f8fa")
    y = 245
    for line in lines:
        draw.text((96, y), line, font=font(29), fill="#c9d5e4")
        y += 67
    draw.text((90, 625), "Measured smoke run • 2026-09-08 • not a benchmark", font=font(20), fill="#7890aa")
    image.save(destination)


def main() -> int:
    record = json.loads((DEMO / "launch-demo.json").read_text(encoding="utf-8"))
    if record["adapter"]["reload_response"] != "THERMAL-READY":
        raise RuntimeError("demo record does not match the rendered conclusion")
    output = DEMO / "molt-launch-demo-75s.mp4"
    with tempfile.TemporaryDirectory(prefix="molt-demo-") as directory:
        temporary = Path(directory)
        concat = []
        for index, (seconds, title, lines) in enumerate(SLIDES):
            path = temporary / f"slide-{index:02d}.png"
            frame(title, lines, path)
            concat.extend((f"file '{path.as_posix()}'", f"duration {seconds}"))
        concat.append(f"file '{path.as_posix()}'")
        manifest = temporary / "slides.txt"
        manifest.write_text("\n".join(concat) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
                "-t", "75",
                "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "slow",
                "-crf", "25", "-movflags", "+faststart", str(output),
            ],
            check=True,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
