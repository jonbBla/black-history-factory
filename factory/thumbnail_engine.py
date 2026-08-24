"""Real thumbnail implementation.

Contract:
  input:  a source scene image + the episode title
  output: paths.thumbnail(job_id) -- a PNG used by the dashboard's
          "most recent video" card.

Composites the title text over the chosen frame with a dark gradient behind
the text for legibility. Falls back to a plain copy of the source image if
PIL/font loading fails for any reason.
"""

from __future__ import annotations
import os
import shutil
import textwrap


def _wrap_title(title: str, width_chars: int = 22) -> list:
    return textwrap.wrap(title, width=width_chars) or [title]


def run(paths, job_id: str, source_image: str, title: str = "") -> str:
    out = paths.thumbnail(job_id)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if not source_image or not os.path.exists(source_image):
        with open(out, "wb") as f:
            f.write(b"")
        return out

    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(source_image).convert("RGB")
        w, h = img.size

        gradient = Image.new("L", (1, h), color=0)
        for y in range(h):
            fade_start = int(h * 0.55)
            if y < fade_start:
                gradient.putpixel((0, y), 0)
            else:
                alpha = int(180 * (y - fade_start) / max(1, (h - fade_start)))
                gradient.putpixel((0, y), alpha)
        gradient = gradient.resize((w, h))
        overlay = Image.new("RGB", (w, h), color=(0, 0, 0))
        img = Image.composite(overlay, img, gradient)

        if title:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    size=max(24, w // 20),
                )
            except Exception:
                font = ImageFont.load_default()

            lines = _wrap_title(title)
            line_height = font.size + 8 if hasattr(font, "size") else 30
            total_h = line_height * len(lines)
            y = h - total_h - int(h * 0.06)
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
                x = max(int(w * 0.05), (w - text_w) // 2 - int(w * 0.15))
                draw.text((x, y), line, font=font, fill=(245, 235, 220))
                y += line_height

        img.save(out, "PNG")
        return out

    except Exception:
        shutil.copy(source_image, out)
        return out
