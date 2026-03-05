"""
AI Video Generator Agent
========================
Generates a narrated video from a topic or a custom list of slides.

Each slide is rendered as a 1280×720 image with a gradient background,
title, bullet-point body text, and an optional icon emoji.
Google Text-to-Speech (gTTS) produces a voice-over for each slide.
MoviePy stitches the images and audio clips into a single MP4 file.

Usage (CLI)
-----------
    # Built-in topic (air quality)
    python ai_video_generator.py

    # Custom topic
    python ai_video_generator.py --topic "Climate Change"

    # Custom slides JSON file
    python ai_video_generator.py --slides my_slides.json

    # Choose output path
    python ai_video_generator.py --topic "Space Exploration" --output space.mp4

Slides JSON format
------------------
    [
      {
        "title": "Slide Title",
        "narration": "Text that will be spoken aloud.",
        "bullets": ["Bullet one", "Bullet two"],
        "icon": "🚀",
        "bg_colors": ["#0a1628", "#1e3a5f"]
      },
      ...
    ]
All keys except "title" and "narration" are optional.

Python API
----------
    from ai_video_generator import VideoGeneratorAgent
    agent = VideoGeneratorAgent()
    agent.generate(topic="Renewable Energy", output_path="energy.mp4")
"""

import os
import sys
import json
import math
import argparse
import textwrap
import tempfile
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from moviepy import (
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

# ─────────────────────────────────────────────────────────────────
# Default slide library (topic → slides)
# ─────────────────────────────────────────────────────────────────

DEFAULT_TOPICS: dict[str, list[dict]] = {
    "Air Quality": [
        {
            "title": "What is Air Quality Index (AQI)?",
            "narration": (
                "The Air Quality Index, or AQI, is a standardised scale that tells "
                "us how clean or polluted the air is at any given time and what "
                "health effects might be a concern."
            ),
            "bullets": [
                "Scale from 0 (Good) to 500 (Severe)",
                "Measures PM2.5, PM10, NO₂, CO, SO₂ and O₃",
                "Published daily by government agencies worldwide",
            ],
            "icon": "🌬️",
            "bg_colors": ["#0a1628", "#1e3a5f"],
        },
        {
            "title": "Key Air Pollutants",
            "narration": (
                "Several pollutants drive the AQI. Particulate matter smaller than "
                "2.5 micrometres — called PM2.5 — is the most dangerous because it "
                "penetrates deep into our lungs. Nitrogen dioxide, carbon monoxide "
                "and ozone also play major roles."
            ),
            "bullets": [
                "PM2.5 & PM10 — fine and coarse particles",
                "NO₂ — produced mainly by traffic & industry",
                "CO — colourless, odourless toxic gas",
                "O₃ — ground-level ozone from sunlight reactions",
            ],
            "icon": "🏭",
            "bg_colors": ["#1a0a28", "#3a1e5f"],
        },
        {
            "title": "Health Impacts of Poor Air Quality",
            "narration": (
                "Breathing polluted air has serious health consequences. Short-term "
                "exposure causes irritation of the eyes, nose and throat. Long-term "
                "exposure is linked to chronic respiratory disease, heart disease "
                "and even lung cancer."
            ),
            "bullets": [
                "Short-term: eye/throat irritation, headaches",
                "Long-term: asthma, COPD, cardiovascular disease",
                "Vulnerable groups: children, elderly, outdoor workers",
                "7 million premature deaths per year (WHO estimate)",
            ],
            "icon": "🫁",
            "bg_colors": ["#280a0a", "#5f1e1e"],
        },
        {
            "title": "AI-Powered AQI Prediction",
            "narration": (
                "Machine learning models can predict the Air Quality Index hours "
                "or even days in advance. By learning patterns from historical "
                "pollution and weather data, a Random Forest model can achieve "
                "an R-squared score above 0.95, giving highly accurate forecasts."
            ),
            "bullets": [
                "Trained on PM2.5, PM10, NO₂, CO, SO₂, O₃ & weather",
                "Random Forest achieves R² > 0.95",
                "Enables early health warnings for citizens",
                "Powers smart city environmental dashboards",
            ],
            "icon": "🤖",
            "bg_colors": ["#0a1a28", "#1e3a5a"],
        },
        {
            "title": "How to Protect Yourself",
            "narration": (
                "There are practical steps everyone can take to reduce exposure "
                "to air pollution. Check the AQI forecast before outdoor activities, "
                "wear an N95 mask on high-pollution days, and keep indoor air "
                "clean with proper ventilation or air purifiers."
            ),
            "bullets": [
                "Check AQI before outdoor exercise",
                "Wear N95 / FFP2 mask when AQI > 150",
                "Use HEPA air purifiers indoors",
                "Reduce car trips — choose public transport or cycling",
                "Plant trees — nature's own air filters",
            ],
            "icon": "🛡️",
            "bg_colors": ["#0a2810", "#1e5f2a"],
        },
    ],
    "Climate Change": [
        {
            "title": "What is Climate Change?",
            "narration": (
                "Climate change refers to long-term shifts in global temperatures "
                "and weather patterns. While some of these shifts are natural, "
                "since the 1800s human activities — mainly the burning of fossil "
                "fuels — have been the main driver."
            ),
            "bullets": [
                "Global average temperature up ~1.1 °C since pre-industrial times",
                "CO₂ levels now above 420 ppm (highest in 3 million years)",
                "Driven by fossil fuels, deforestation & industry",
            ],
            "icon": "🌍",
            "bg_colors": ["#0a1628", "#1e3a5f"],
        },
        {
            "title": "Causes of Climate Change",
            "narration": (
                "The burning of coal, oil and natural gas releases carbon dioxide "
                "and methane. These greenhouse gases trap heat in the atmosphere, "
                "causing the planet to warm. Deforestation removes trees that "
                "would otherwise absorb that CO₂."
            ),
            "bullets": [
                "Fossil fuels — 75% of global greenhouse gas emissions",
                "Deforestation — 10% of emissions",
                "Agriculture — methane from livestock & rice paddies",
                "Industry — cement, steel and chemical production",
            ],
            "icon": "🔥",
            "bg_colors": ["#280a0a", "#5f1e1e"],
        },
        {
            "title": "Effects We See Today",
            "narration": (
                "The effects of climate change are already visible. Extreme weather "
                "events are more frequent and severe. Glaciers and ice sheets are "
                "melting, causing sea levels to rise and threatening coastal cities."
            ),
            "bullets": [
                "More frequent heatwaves, floods & droughts",
                "Arctic sea ice declining 13% per decade",
                "Sea level rise of 3.7 mm per year",
                "Coral bleaching — 50% of the Great Barrier Reef lost",
            ],
            "icon": "🌊",
            "bg_colors": ["#0a1a28", "#1e3a5a"],
        },
        {
            "title": "Solutions & Clean Energy",
            "narration": (
                "Transitioning to renewable energy is the single most important "
                "step we can take. Solar and wind power are now the cheapest sources "
                "of electricity in history. Combined with energy efficiency and "
                "reforestation, we can still limit warming to 1.5 degrees Celsius."
            ),
            "bullets": [
                "Solar & wind — cheapest electricity ever",
                "Electric vehicles reducing transport emissions",
                "Reforestation sequesters billions of tonnes of CO₂",
                "Green hydrogen for heavy industry",
            ],
            "icon": "♻️",
            "bg_colors": ["#0a2810", "#1e5f2a"],
        },
    ],
    "Space Exploration": [
        {
            "title": "Why Explore Space?",
            "narration": (
                "Space exploration expands our understanding of the universe, "
                "drives technological innovation, and unites humanity in a shared "
                "quest for knowledge. Technologies developed for space have given "
                "us GPS, memory foam, water purification and much more."
            ),
            "bullets": [
                "Advances science & technology",
                "Inspires future generations of engineers & scientists",
                "Searches for signs of extraterrestrial life",
                "Potential for off-world resources & human settlement",
            ],
            "icon": "🚀",
            "bg_colors": ["#0a0a1a", "#1a1a3f"],
        },
        {
            "title": "Key Milestones",
            "narration": (
                "Humanity has achieved remarkable milestones in space. Sputnik "
                "became the first artificial satellite in 1957. Just twelve years "
                "later, Apollo 11 landed the first humans on the Moon. Today, the "
                "International Space Station has been continuously crewed for "
                "over 24 years."
            ),
            "bullets": [
                "1957 — Sputnik: first artificial satellite",
                "1969 — Apollo 11: humans walk on the Moon",
                "1990 — Hubble Space Telescope launched",
                "2020 — SpaceX Crew Dragon: commercial crew era",
                "2021 — James Webb Space Telescope",
            ],
            "icon": "🌕",
            "bg_colors": ["#0a1628", "#1e3a5f"],
        },
        {
            "title": "Mars — The Next Frontier",
            "narration": (
                "Mars is our most likely destination for human settlement. "
                "Multiple rovers have explored its surface, revealing evidence "
                "of ancient rivers and lakes. NASA's Artemis programme and SpaceX "
                "Starship are both targeting crewed Mars missions in the 2030s."
            ),
            "bullets": [
                "Perseverance rover collecting rock samples",
                "Ingenuity helicopter — first powered flight on another planet",
                "Evidence of ancient liquid water on Mars",
                "Crewed missions targeted for the 2030s",
            ],
            "icon": "🔴",
            "bg_colors": ["#280a0a", "#4a1e1e"],
        },
    ],
}


# ─────────────────────────────────────────────────────────────────
# Colour & style helpers
# ─────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        # Expand shorthand: #abc → #aabbcc
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        raise ValueError(
            f"Invalid hex color '#{hex_color}' — must be 3 or 6 hex digits."
        )
    return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))


def _interpolate_color(
    color1: tuple[int, int, int],
    color2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    return tuple(int(color1[i] + (color2[i] - color1[i]) * t) for i in range(3))


# ─────────────────────────────────────────────────────────────────
# Image rendering
# ─────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
ACCENT = (37, 99, 235)       # blue
TEXT_PRIMARY = (226, 232, 240)
TEXT_SECONDARY = (148, 163, 184)
BULLET_COLOR = (6, 182, 212)   # cyan


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Try to load a system font; fall back to PIL's built-in default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold
        else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_gradient_background(
    draw: ImageDraw.ImageDraw,
    color1: tuple[int, int, int],
    color2: tuple[int, int, int],
) -> None:
    for y in range(HEIGHT):
        t = y / HEIGHT
        r, g, b = _interpolate_color(color1, color2, t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))


def _draw_decorative_elements(draw: ImageDraw.ImageDraw) -> None:
    """Draw subtle geometric accents."""
    # Top-left accent bar
    draw.rectangle([0, 0, 6, HEIGHT], fill=(*ACCENT, 200))
    # Bottom thin line
    draw.rectangle([0, HEIGHT - 3, WIDTH, HEIGHT], fill=(*ACCENT, 150))
    # Faint circle decoration top-right
    cx, cy, r = WIDTH - 80, 80, 140
    for i in range(3):
        offset = i * 30
        draw.ellipse(
            [cx - r - offset, cy - r - offset, cx + r + offset, cy + r + offset],
            outline=(*ACCENT, 20 - i * 6),
            width=2,
        )


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    dummy = Image.new("RGB", (1, 1))
    d = ImageDraw.Draw(dummy)
    for word in words:
        test = (current + " " + word).strip()
        bbox = d.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_slide(slide: dict, slide_number: int, total_slides: int) -> np.ndarray:
    """Render a single slide as a NumPy array (H×W×3, uint8)."""
    title = slide.get("title", "")
    bullets = slide.get("bullets", [])
    icon = slide.get("icon", "")
    raw_colors = slide.get("bg_colors", ["#0a1628", "#152848"])

    color1 = _hex_to_rgb(raw_colors[0])
    color2 = _hex_to_rgb(raw_colors[1])

    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img, "RGBA")

    # Background gradient
    _draw_gradient_background(draw, color1, color2)
    _draw_decorative_elements(draw)

    # Fonts
    font_title = _load_font(52, bold=True)
    font_bullet = _load_font(30)
    font_small = _load_font(22)
    font_icon = _load_font(72, bold=False)

    # Progress bar at bottom
    progress = (slide_number) / total_slides
    bar_width = int(WIDTH * progress)
    draw.rectangle([0, HEIGHT - 3, bar_width, HEIGHT], fill=(*ACCENT, 255))

    # Slide counter
    counter_text = f"{slide_number} / {total_slides}"
    draw.text(
        (WIDTH - 100, HEIGHT - 36),
        counter_text,
        font=font_small,
        fill=(*TEXT_SECONDARY, 180),
    )

    # Icon (top-right area)
    if icon:
        try:
            draw.text((WIDTH - 170, 40), icon, font=font_icon, fill=(255, 255, 255, 200))
        except Exception:
            pass

    # Title
    margin_left = 60
    title_y = 60
    text_max_width = WIDTH - margin_left * 2 - (180 if icon else 0)
    title_lines = _wrap_text(title, font_title, text_max_width)
    for line in title_lines:
        draw.text((margin_left, title_y), line, font=font_title, fill=TEXT_PRIMARY)
        bbox = draw.textbbox((margin_left, title_y), line, font=font_title)
        title_y = bbox[3] + 8

    # Underline below title
    draw.rectangle(
        [margin_left, title_y + 10, margin_left + 80, title_y + 14],
        fill=ACCENT,
    )
    title_y += 36

    # Bullet points
    bullet_x = margin_left
    bullet_y = title_y
    line_height = 52
    for point in bullets:
        if bullet_y + line_height > HEIGHT - 50:
            break
        # Bullet dot
        draw.ellipse(
            [bullet_x, bullet_y + 14, bullet_x + 10, bullet_y + 24],
            fill=BULLET_COLOR,
        )
        # Bullet text (wrap if long)
        wrapped = _wrap_text(point, font_bullet, WIDTH - bullet_x - 90)
        for i, wline in enumerate(wrapped[:2]):  # max 2 sub-lines per bullet
            draw.text(
                (bullet_x + 24, bullet_y + (i * 34)),
                wline,
                font=font_bullet,
                fill=TEXT_PRIMARY,
            )
        bullet_y += line_height + (34 * max(0, len(wrapped[:2]) - 1))

    return np.array(img)


# ─────────────────────────────────────────────────────────────────
# Voice-over generation
# ─────────────────────────────────────────────────────────────────

def generate_voiceover(text: str, output_path: str, lang: str = "en") -> bool:
    """
    Generate an MP3 voice-over using gTTS.
    Returns True on success, False if the network is unavailable.
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return True
    except Exception as exc:
        print(f"    [TTS] Warning: could not generate audio — {exc}")
        return False


# ─────────────────────────────────────────────────────────────────
# Main agent class
# ─────────────────────────────────────────────────────────────────

class VideoGeneratorAgent:
    """
    AI Video Generator Agent.

    Parameters
    ----------
    lang : str
        Language code for gTTS (default 'en').
    fps : int
        Frames per second for the output video (default 24).
    fallback_slide_duration : float
        Seconds to show a slide when TTS is unavailable (default 5.0).
    """

    def __init__(
        self,
        lang: str = "en",
        fps: int = 24,
        fallback_slide_duration: float = 5.0,
    ) -> None:
        self.lang = lang
        self.fps = fps
        self.fallback_slide_duration = fallback_slide_duration

    # ── public API ────────────────────────────────────────────────

    def generate(
        self,
        topic: str | None = None,
        slides: list[dict] | None = None,
        output_path: str = "output_video.mp4",
    ) -> str:
        """
        Generate an MP4 video.

        Provide either *topic* (a built-in topic name) or *slides*
        (a custom list of slide dicts). If both are given, *slides*
        takes precedence. If neither is given, the default 'Air Quality'
        topic is used.

        Returns the absolute path to the generated video.
        """
        if slides is None:
            if topic is None:
                topic = "Air Quality"
            slides = self._get_slides_for_topic(topic)

        print(f"\n{'='*60}")
        print("  AI Video Generator Agent")
        print(f"  Topic : {topic or 'Custom slides'}")
        print(f"  Slides: {len(slides)}")
        print(f"  Output: {output_path}")
        print(f"{'='*60}\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            clips = self._build_clips(slides, tmpdir)
            if not clips:
                raise RuntimeError("No clips were produced — cannot create video.")
            final = concatenate_videoclips(clips, method="compose")
            output_path = os.path.abspath(output_path)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            final.write_videofile(
                output_path,
                fps=self.fps,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )

        print(f"\n✅ Video saved: {output_path}")
        print(
            f"   Duration : {final.duration:.1f}s  |  "
            f"Resolution: {WIDTH}×{HEIGHT}  |  FPS: {self.fps}"
        )
        return output_path

    # ── private helpers ───────────────────────────────────────────

    def _get_slides_for_topic(self, topic: str) -> list[dict]:
        """Look up built-in slides or raise a helpful error."""
        # Case-insensitive lookup
        for key, value in DEFAULT_TOPICS.items():
            if key.lower() == topic.lower():
                return value
        available = ", ".join(f'"{k}"' for k in DEFAULT_TOPICS)
        raise ValueError(
            f'Unknown topic "{topic}". '
            f"Built-in topics: {available}. "
            "Use --slides to supply a custom JSON file."
        )

    def _build_clips(self, slides: list[dict], tmpdir: str) -> list:
        clips = []
        total = len(slides)
        for idx, slide in enumerate(slides, start=1):
            title = slide.get("title", f"Slide {idx}")
            narration = slide.get("narration", title)
            print(f"  [{idx}/{total}] Rendering: {title}")

            # 1. Render image frame
            frame = render_slide(slide, idx, total)
            img_path = os.path.join(tmpdir, f"slide_{idx:03d}.png")
            Image.fromarray(frame).save(img_path)

            # 2. Generate voice-over
            audio_path = os.path.join(tmpdir, f"audio_{idx:03d}.mp3")
            has_audio = generate_voiceover(narration, audio_path, self.lang)

            # 3. Create video clip
            if has_audio and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration + 0.5  # small tail pause
                video_clip = (
                    ImageClip(img_path, duration=duration)
                    .with_audio(audio_clip)
                )
            else:
                duration = self.fallback_slide_duration
                video_clip = ImageClip(img_path, duration=duration)

            clips.append(video_clip)
        return clips


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Video Generator Agent — creates narrated videos from slides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            f"""\
            Built-in topics:
              {chr(10).join(f'  • {t}' for t in DEFAULT_TOPICS)}

            Examples:
              python ai_video_generator.py
              python ai_video_generator.py --topic "Climate Change"
              python ai_video_generator.py --topic "Space Exploration" --output space.mp4
              python ai_video_generator.py --slides my_slides.json --output custom.mp4
            """
        ),
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Built-in topic name (default: Air Quality)",
    )
    parser.add_argument(
        "--slides",
        default=None,
        metavar="FILE",
        help="Path to a JSON file containing a list of slide dicts",
    )
    parser.add_argument(
        "--output",
        default="output_video.mp4",
        help="Output MP4 file path (default: output_video.mp4)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="gTTS language code for voice-over (default: en)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Video frames per second (default: 24)",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="Print available built-in topics and exit",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_topics:
        print("Available built-in topics:")
        for topic in DEFAULT_TOPICS:
            print(f"  • {topic}  ({len(DEFAULT_TOPICS[topic])} slides)")
        return

    slides = None
    if args.slides:
        with open(args.slides, "r", encoding="utf-8") as fh:
            slides = json.load(fh)
        if not isinstance(slides, list):
            print("ERROR: The JSON file must contain a list of slide objects.")
            sys.exit(1)

    agent = VideoGeneratorAgent(lang=args.lang, fps=args.fps)
    try:
        agent.generate(
            topic=args.topic,
            slides=slides,
            output_path=args.output,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
