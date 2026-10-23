"""Remove verified title-only pixels from neutral CAD views without cropping.

The original renderer draws Arial 22 at known locations. A title rectangle is
removed only if EVERY pixel in it equals that title rendered on white. Any
unexpected pixel causes the entire original image to be preserved unchanged.
No source file is modified, and no image is resized or cropped.
"""
from pathlib import Path
import argparse
import json
import re
import shutil

from PIL import Image, ImageChops, ImageDraw, ImageFont

VIEWS = ("oblique", "opposite", "profile", "axial", "cut_X", "cut_Y")
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")


def title_plan(original, review_id):
    """Verify whole title rectangles against an independently rendered template."""
    if original.size != (1500, 1000) or not re.fullmatch(r"Z\d{4}", review_id):
        return [], "unexpected_image_size_or_identifier"
    if not FONT_PATH.exists():
        return [], "original_renderer_font_unavailable"
    font = ImageFont.truetype(str(FONT_PATH), 22)
    template = Image.new("RGB", original.size, "white")
    draw = ImageDraw.Draw(template)
    rectangles = []
    for index, view in enumerate(VIEWS):
        xy = (index % 3 * 500 + 8, index // 3 * 500 + 6)
        text = review_id + " / " + view
        box = draw.textbbox(xy, text, font=font)
        box = (max(0, box[0]-2), max(0, box[1]-2),
               min(1500, box[2]+2), min(1000, box[3]+2))
        draw.text(xy, text, font=font, fill="black")
        # Exact template equality is stronger than a blue-pixel threshold: it
        # also rejects unexpected grey or black geometry inside the rectangle.
        if ImageChops.difference(original.crop(box), template.crop(box)).getbbox() is not None:
            return [], "title_rectangle_contains_noncaption_pixels_or_font_mismatch"
        rectangles.append(box)
    return rectangles, None


def expected_neutral(original, review_id):
    rectangles, reason = title_plan(original, review_id)
    result = original.copy()
    for box in rectangles:
        result.paste("white", box)
    return result, rectangles, reason


def neutralize(source, target):
    """Create a lossless neutral asset; return its deterministic audit record."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("Never overwrite an original CAD view")
    with Image.open(source) as image:
        original = image.convert("RGB")
    result, rectangles, reason = expected_neutral(original, source.stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    if reason:
        # Neutral Z identifiers are acceptable if safe removal is unprovable.
        shutil.copyfile(source, target)
    else:
        result.save(target, format="PNG", compress_level=3)
    return dict(source=str(source), target=str(target), size=list(original.size),
                mode="original_preserved" if reason else "verified_titles_removed",
                title_rectangles=[list(box) for box in rectangles], fallback_reason=reason,
                policy="Exact original-caption template; no crop, resize or change outside title-only rectangles")


def verify(source, target):
    """Prove pixel identity to the allowed source transformation."""
    source, target = Path(source), Path(target)
    with Image.open(source) as image:
        original = image.convert("RGB")
    with Image.open(target) as image:
        actual = image.convert("RGB")
    expected, rectangles, reason = expected_neutral(original, source.stem)
    if actual.size != original.size:
        raise AssertionError("Neutral asset changed the original image dimensions")
    if ImageChops.difference(actual, expected).getbbox() is not None:
        raise AssertionError("Neutral asset changed pixels outside the permitted title removal")
    return dict(verified=True, mode="original_preserved" if reason else "verified_titles_removed",
                size=list(original.size), title_rectangles=[list(box) for box in rectangles],
                fallback_reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    record = neutralize(args.source, args.target)
    record["verification"] = verify(args.source, args.target)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
