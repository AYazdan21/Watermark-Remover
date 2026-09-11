"""Generates clean, synthetic Persian/Farsi business-document background
pages for the watermark-compositing pipeline.

Why generation instead of scraping a corpus: the obvious public Persian OCR
dataset (IDPL-PFOD) is single text lines at ~700x50px -- useless as a full
page background. Bulk-scraping real documents is ToS-sensitive, slow, and
non-reproducible. Programmatic generation (in the spirit of the "Persian
Pixel" synthetic-document line of work) gives unlimited volume, zero
licensing risk, and full control over the structural variety that actually
matters for training a segmentation model: bordered notices, dense tables,
form grids, plain letters, spreadsheet grids -- at varying page sizes,
margins, tints, rule colors, font sizes and a "scanned" degradation.

Ground variety: early corpora from this module were uniformly light,
near-white paper with dark ink -- realistic for a scanned form, useless for
teaching a segmenter that a watermark can sit on a dark presentation slide,
a photograph, or blank paper. `PageStyle` therefore also samples dark
grounds (~15%) and strongly tinted coloured papers (~20%) alongside the
original near-white look, inverting ink/rule/accent to light variants
whenever the ground is dark (see `PageStyle.random`); and three archetypes
-- `blank_page`, `photo_report`, `slide_deck` -- cover near-empty pages,
pages with procedural raster imagery, and landscape slide decks, which the
original six (all dense text/tables/forms) never exercised.

Text rendering: Persian is RTL and cursive-joining -- naively drawing a
logical-order Unicode string with a shaping-unaware layout engine produces
disconnected, wrongly-ordered letterforms. This module renders through
PIL's Raqm-backed text layout (``ImageDraw.text(..., direction="rtl",
language="fa")``), which was verified against this Pillow build
(``PIL.features.check("raqm")``) to shape and reorder Persian text
correctly with NO manual pre-shaping needed -- see `_HAS_RAQM` and the
verification notes in the task report. When Raqm isn't available (a
different Pillow build with no libraqm), this degrades to manual shaping
via `arabic_reshaper` + `python-bidi` if those are importable, and as a
last resort draws raw logical text (which WILL look wrong -- disconnected /
reversed -- but at least still produces an image rather than crashing).

Fonts: prefers the bundled Vazirmatn webfont (SIL OFL, fetched into
``fonts/`` alongside this file) for a realistic, modern Persian document
look, and falls back through the Arabic-capable fonts already on this
Windows box (Tahoma, arabtype.ttf, Segoe UI) if Vazirmatn is missing --
so this still works with zero network access, just with system-font
looks instead.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"

_HAS_RAQM = features.check("raqm")
try:
    import arabic_reshaper
    from bidi import get_display as _bidi_get_display

    _HAS_RESHAPER = True
except Exception:
    _HAS_RESHAPER = False


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

# (regular, bold) candidates in priority order. First pair whose files all
# resolve wins as the "primary" family; the rest are kept as alternates so
# individual pages can vary their typeface for visual diversity.
_FONT_CANDIDATES: List[Tuple[str, str]] = [
    (str(FONT_DIR / "Vazirmatn-Regular.ttf"), str(FONT_DIR / "Vazirmatn-Bold.ttf")),
    (str(FONT_DIR / "Vazirmatn-Medium.ttf"), str(FONT_DIR / "Vazirmatn-Bold.ttf")),
    (r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\tahomabd.ttf"),
    (r"C:\Windows\Fonts\arabtype.ttf", r"C:\Windows\Fonts\arabtype.ttf"),
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
]


def _resolve_font_families() -> List[Tuple[str, str]]:
    resolved = [(r, b) for r, b in _FONT_CANDIDATES if os.path.isfile(r) and os.path.isfile(b)]
    if not resolved:
        raise RuntimeError(
            "No Arabic/Persian-capable font found. Expected a bundled Vazirmatn "
            f"font under {FONT_DIR} or a system font (Tahoma/arabtype/Segoe UI)."
        )
    return resolved


_FONT_FAMILIES = _resolve_font_families()
_FONT_CACHE: dict = {}


def get_font(size: int, bold: bool = False, family_idx: int = 0) -> ImageFont.FreeTypeFont:
    family_idx = family_idx % len(_FONT_FAMILIES)
    reg, bd = _FONT_FAMILIES[family_idx]
    path = bd if bold else reg
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def n_font_families() -> int:
    return len(_FONT_FAMILIES)


# ---------------------------------------------------------------------------
# Persian digit / RTL text helpers
# ---------------------------------------------------------------------------

_LATIN_TO_FARSI_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def farsi_digits(s: str) -> str:
    return s.translate(_LATIN_TO_FARSI_DIGITS)


def _shape_for_basic_layout(text: str) -> str:
    """Fallback path when Raqm isn't available: manually shape + visually
    reorder so a plain (non-Raqm) layout engine draws it left-to-right in
    the already-correct visual order."""
    if not _HAS_RESHAPER:
        return text  # last resort -- will render disconnected/reversed
    return _bidi_get_display(arabic_reshaper.reshape(text))


def rtl_text_kwargs() -> dict:
    """Extra kwargs to pass to ImageDraw.text/textlength/textbbox so Persian
    shapes and reorders correctly. Only valid when Raqm is compiled in."""
    return {"direction": "rtl", "language": "fa"} if _HAS_RAQM else {}


def prep_rtl(text: str) -> str:
    """Returns the string ready to hand to ImageDraw.text alongside
    rtl_text_kwargs(): unchanged (logical order) when Raqm will do the
    shaping itself, pre-shaped+reordered otherwise."""
    return text if _HAS_RAQM else _shape_for_basic_layout(text)


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    prepared = prep_rtl(text)
    return draw.textlength(prepared, font=font, **rtl_text_kwargs())


def draw_rtl(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, anchor="ra"):
    prepared = prep_rtl(text)
    draw.text(xy, prepared, font=font, fill=fill, anchor=anchor, **rtl_text_kwargs())


def wrap_rtl(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> List[str]:
    """Greedy word-wrap of a logical-order Persian string to fit max_width,
    breaking only on plain spaces (keeps ZWNJ-joined compounds intact)."""
    words = text.split(" ")
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and text_width(draw, trial, font) > max_width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_rtl_paragraph(draw, text, font, right_x, top_y, max_width, fill, line_gap=1.5):
    lines = wrap_rtl(draw, text, font, max_width)
    y = top_y
    line_h = font.size * line_gap
    for ln in lines:
        draw_rtl(draw, (right_x, y), ln, font, fill, anchor="ra")
        y += line_h
    return y


# ---------------------------------------------------------------------------
# Content bank -- generic, originally-written procurement/tender boilerplate.
# Not copied from any real document; purely for structural realism.
# ---------------------------------------------------------------------------

TITLES = [
    "آگهی مزایده عمومی",
    "آگهی مناقصه عمومی یک مرحله‌ای",
    "فراخوان ارزیابی کیفی مناقصه‌گران",
    "اطلاعیه ثبت‌نام و دریافت اسناد",
    "صورت‌جلسه بازگشایی پاکات مناقصه",
    "آگهی فراخوان عمومی خرید تجهیزات",
    "اطلاعیه مزایده فروش اموال منقول",
]

ORG_NAMES = [
    "شهرداری منطقه یک",
    "سازمان مدیریت و برنامه‌ریزی استان",
    "شرکت مهندسی توسعه ساختمان",
    "اداره کل راه و شهرسازی",
    "سازمان صنعت، معدن و تجارت",
    "شرکت آب و فاضلاب منطقه",
    "دانشگاه علوم پزشکی استان",
]

BODY_SENTENCES = [
    "{org} در نظر دارد نسبت به واگذاری موضوع این آگهی از طریق مزایده عمومی اقدام نماید.",
    "متقاضیان می‌توانند جهت دریافت اسناد و کسب اطلاعات بیشتر به نشانی دبیرخانه سازمان مراجعه و یا با شماره تلفن درج‌شده تماس حاصل فرمایند.",
    "مهلت دریافت اسناد از تاریخ انتشار آگهی به مدت ده روز کاری بوده و به پیشنهادهای فاقد سپرده یا ارسال‌شده خارج از موعد ترتیب اثر داده نخواهد شد.",
    "کلیه هزینه‌های چاپ آگهی در روزنامه‌های کثیرالانتشار بر عهده برنده مزایده خواهد بود.",
    "این سازمان در رد یا قبول هر یک از پیشنهادها مختار است.",
    "سپرده شرکت در مزایده معادل پنج درصد قیمت پایه کارشناسی می‌باشد که می‌بایست به‌صورت ضمانت‌نامه بانکی یا فیش نقدی ارائه گردد.",
    "بازگشایی پاکات پیشنهادها در جلسه‌ای با حضور اعضای کمیسیون معاملات برگزار خواهد شد.",
    "متقاضیان محترم می‌بایست پیش از تکمیل فرم شرکت در مزایده از محل مورد نظر بازدید به عمل آورند.",
    "پرداخت مابه‌التفاوت قیمت پیشنهادی و قیمت پایه کارشناسی می‌بایست ظرف مدت یک هفته پس از اعلام نتیجه صورت پذیرد.",
    "اسناد مزایده به همراه نقشه و کروکی ملک در پیوست این آگهی موجود می‌باشد.",
]

FORM_FIELD_LABELS = [
    "نام و نام خانوادگی",
    "کد ملی",
    "شماره تماس",
    "نشانی پستی",
    "کد پستی",
    "شماره پرونده",
    "تاریخ تولد",
    "نام پدر",
    "شماره شناسنامه",
    "پست الکترونیک",
    "نام شرکت / سازمان",
    "شماره ثبت شرکت",
    "کد اقتصادی",
    "شماره حساب بانکی",
    "مبلغ سپرده (ریال)",
    "تاریخ درخواست",
    "نوع فعالیت",
    "شماره پیگیری",
]

RADIO_GROUPS = [
    ("جنسیت", ["مرد", "زن"]),
    ("نوع متقاضی", ["حقیقی", "حقوقی"]),
    ("وضعیت تأهل", ["مجرد", "متأهل"]),
    ("نحوه پرداخت", ["نقدی", "اقساطی"]),
]

TABLE_HEADERS = ["ردیف", "شرح کالا / خدمات", "تعداد", "واحد", "مبلغ واحد (ریال)", "مبلغ کل (ریال)"]
UNITS = ["عدد", "دستگاه", "بسته", "متر", "کیلوگرم", "سری"]
ITEM_NAMES = [
    "کاغذ A4 هشتاد گرم",
    "تجهیزات شبکه",
    "صندلی اداری",
    "میز کارشناسی",
    "پرینتر لیزری",
    "کابل شبکه",
    "لوازم التحریر",
    "باتری یو پی اس",
    "چاپگر حرارتی",
    "کیس کامپیوتر",
]

LETTER_SUBJECTS = [
    "درخواست تمدید مهلت اجرای پروژه",
    "اعلام نتیجه بررسی پیشنهادات فنی",
    "پیگیری وضعیت پرداخت صورت‌وضعیت",
    "ابلاغ نتایج ارزیابی کیفی",
    "دعوت به جلسه بازگشایی پاکات",
]

SIGNATURE_LABELS = ["امضا و مهر", "محل امضا", "مسئول دبیرخانه"]

BASMALEH = "به نام خدا"


def rand_number(rng: random.Random, low: int, high: int) -> str:
    return farsi_digits(str(rng.randint(low, high)))


def rand_date(rng: random.Random) -> str:
    return farsi_digits(f"140{rng.randint(0,3)}/{rng.randint(1,12):02d}/{rng.randint(1,28):02d}")


def rand_money(rng: random.Random) -> str:
    val = rng.randint(50, 90000) * 1000
    return farsi_digits(f"{val:,}").replace(",", "٬")


# ---------------------------------------------------------------------------
# Page style
# ---------------------------------------------------------------------------

PAGE_SIZES = [
    (1240, 1754, "a4_portrait"),
    (1754, 1240, "a4_landscape"),
    (1100, 1500, "letter_ish"),
    (1300, 1700, "form_tall"),
    (1500, 1000, "spreadsheet_wide"),
]

BG_TINTS = [
    (255, 255, 255),
    (250, 247, 238),
    (247, 246, 242),
    (245, 246, 248),
    (252, 250, 244),
    (240, 240, 236),
]

RULE_COLORS = [
    (20, 20, 20),
    (40, 40, 45),
    (60, 30, 30),
    (25, 40, 30),
]

ACCENT_COLORS = [
    (30, 45, 90),    # navy
    (120, 20, 24),   # maroon/red
    (20, 80, 50),    # dark green
    (90, 60, 20),    # brown
    (25, 70, 80),    # teal
    (60, 60, 60),    # charcoal
]

# Strongly-tinted light papers -- real forms/letterheads use visibly
# coloured stock (pale blue requisition forms, pale-yellow carbon copies,
# pale-pink "urgent" routing sheets), not just the near-white BG_TINTS
# above. Luminance stays high (~230-245) so the existing dark
# RULE_COLORS/ACCENT_COLORS remain legible directly on top -- no ink
# inversion needed for these.
COLOR_BG_TINTS = [
    (225, 235, 250),  # pale blue
    (224, 240, 226),  # pale green
    (250, 244, 214),  # pale yellow
    (250, 228, 235),  # pale pink
    (235, 228, 245),  # pale grey-lavender
]

# Dark grounds -- near-black neutrals and dark navy/charcoal/slate, kept in
# roughly the 18-55 luminance range so they read as "a dark UI/slide", not
# pure black. Ink/rule/accent must invert to light colours whenever one of
# these is chosen (see PageStyle.random) or every archetype renders
# black-on-black.
DARK_BG_GROUNDS = [
    (18, 18, 20),    # near-black neutral
    (26, 28, 32),    # graphite
    (30, 33, 40),    # slate
    (22, 26, 38),    # dark navy
    (35, 30, 30),    # dark charcoal-brown
    (28, 40, 36),    # dark forest
    (50, 52, 58),    # lighter slate, near the top of the range
]

# Light ink for dark grounds -- swapped in for style.rule (the main
# body-text/line colour, drawn straight onto the page background) whenever
# PageStyle.random picks a dark ground. All comfortably above ~210
# luminance so contrast against DARK_BG_GROUNDS is never marginal.
DARK_RULE_COLORS = [
    (235, 235, 235),
    (225, 228, 232),
    (210, 215, 225),  # cool light gray
    (230, 220, 205),  # warm cream
]

# Light accent for dark grounds -- swapped in for style.accent (used for
# borders/underlines/seals/letterhead text drawn straight on the page
# background) whenever the ground is dark. NOT used for the opaque
# header/footer bars -- those keep drawing from ACCENT_COLORS via
# style.bar_fill so the bar's hardcoded white text stays legible regardless
# of what the surrounding page looks like.
DARK_ACCENT_COLORS = [
    (140, 180, 240),  # light blue
    (240, 160, 170),  # light coral/rose
    (150, 215, 175),  # light mint
    (230, 195, 120),  # light gold/tan
    (140, 215, 220),  # light cyan/teal
    (200, 200, 205),  # light graphite/silver
]


@dataclass
class PageStyle:
    w: int
    h: int
    size_name: str
    bg: Tuple[int, int, int]
    rule: Tuple[int, int, int]
    accent: Tuple[int, int, int]
    bar_fill: Tuple[int, int, int]
    font_family: int
    base_font_size: int
    margin: int
    scanned: bool
    dark: bool

    @staticmethod
    def random(rng: random.Random, prefer_landscape: bool = False) -> "PageStyle":
        sizes = PAGE_SIZES
        if prefer_landscape and rng.random() < 0.85:
            landscape_sizes = [s for s in PAGE_SIZES if s[0] > s[1]]
            if landscape_sizes:
                sizes = landscape_sizes
        w, h, name = rng.choice(sizes)
        # +/- 6% jitter on page size for variety without breaking archetype proportions
        w = int(w * rng.uniform(0.94, 1.06))
        h = int(h * rng.uniform(0.94, 1.06))

        # Ground: ~15% dark, ~20% strongly-tinted light, the remaining ~65%
        # the original near-white look. Ink/rule/accent are picked to match
        # centrally, right here, so every archetype below just uses
        # style.rule/style.accent and gets the right contrast for free
        # instead of needing its own "if dark:" branch.
        roll = rng.random()
        if roll < 0.15:
            dark = True
            bg = rng.choice(DARK_BG_GROUNDS)
        elif roll < 0.35:
            dark = False
            bg = rng.choice(COLOR_BG_TINTS)
        else:
            dark = False
            bg = rng.choice(BG_TINTS)

        rule = rng.choice(DARK_RULE_COLORS) if dark else rng.choice(RULE_COLORS)
        accent = rng.choice(DARK_ACCENT_COLORS) if dark else rng.choice(ACCENT_COLORS)
        # Always drawn from the original mid-dark palette regardless of
        # ground: this is the fill behind hardcoded white bar text (header
        # bars, table header rows, ...), so it must stay dark/saturated
        # enough for white to read on it even when the page around it is a
        # dark ground with a light accent everywhere else.
        bar_fill = rng.choice(ACCENT_COLORS)

        return PageStyle(
            w=w, h=h, size_name=name,
            bg=bg, rule=rule, accent=accent, bar_fill=bar_fill,
            font_family=rng.randrange(n_font_families()),
            base_font_size=rng.randint(20, 30),
            margin=rng.randint(50, 110),
            scanned=rng.random() < 0.35,
            dark=dark,
        )


# ---------------------------------------------------------------------------
# Archetype generators -- each returns a PIL RGB Image
# ---------------------------------------------------------------------------

def _new_canvas(style: PageStyle) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (style.w, style.h), style.bg)
    return im, ImageDraw.Draw(im)


def gen_tender_notice(rng: random.Random, style: PageStyle) -> Image.Image:
    """Bordered notice: colored header/footer bars, centered heading, RTL
    body paragraphs, a decorative outer frame, small geometric logo mark."""
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, style.margin

    border_w = rng.randint(3, 7)
    d.rectangle([m // 2, m // 2, w - m // 2, h - m // 2], outline=style.accent, width=border_w)
    inset = m // 2 + border_w + 8
    if rng.random() < 0.5:
        d.rectangle([inset, inset, w - inset, h - inset], outline=style.rule, width=1)

    bar_h = rng.randint(50, 80)
    d.rectangle([m, m, w - m, m + bar_h], fill=style.bar_fill)
    org = rng.choice(ORG_NAMES)
    hdr_font = get_font(rng.randint(24, 30), bold=True, family_idx=style.font_family)
    draw_rtl(d, (w - m - 20, m + bar_h // 2), org, hdr_font, (255, 255, 255), anchor="rm")

    title_font = get_font(style.base_font_size + 12, bold=True, family_idx=style.font_family)
    title = f"{rng.choice(TITLES)} شماره {rand_date(rng)[:9]}"
    tw = text_width(d, title, title_font)
    draw_rtl(d, (w / 2 + tw / 2, m + bar_h + 30), title, title_font, style.rule, anchor="ra")
    d.line([(w / 2 - tw / 2 - 10, m + bar_h + 30 + title_font.size + 12),
            (w / 2 + tw / 2 + 10, m + bar_h + 30 + title_font.size + 12)], fill=style.accent, width=2)

    body_font = get_font(style.base_font_size, family_idx=style.font_family)
    n_para = rng.randint(3, 6)
    chosen = rng.sample(BODY_SENTENCES, k=min(n_para, len(BODY_SENTENCES)))
    y = m + bar_h + 30 + title_font.size + 40
    right_x = w - m - 20
    max_w = w - 2 * m - 40
    for sent in chosen:
        text = sent.format(org=org)
        y = draw_rtl_paragraph(d, text, body_font, right_x, y, max_w, style.rule, line_gap=1.6)
        y += body_font.size * 0.8

    footer_h = rng.randint(40, 60)
    footer_top = h - m - footer_h

    # a small key/value "tender details" block -- fills the page more
    # realistically than paragraphs alone and adds structural variety
    # (label-left / value-right rows, like a real notice's fact box).
    if rng.random() < 0.75 and y + 160 < footer_top:
        detail_font = get_font(style.base_font_size - 2, family_idx=style.font_family)
        detail_label_font = get_font(style.base_font_size - 2, bold=True, family_idx=style.font_family)
        details = [
            ("مبلغ سپرده شرکت در مزایده", rand_money(rng) + " ریال"),
            ("مهلت دریافت اسناد", rand_date(rng)),
            ("مهلت تسلیم پیشنهاد", rand_date(rng)),
            ("محل بازگشایی پاکات", "دبیرخانه " + org),
        ]
        box_top = y + 20
        row_h2 = detail_font.size * 2.0
        box_bottom = min(footer_top - 20, box_top + row_h2 * len(details))
        n_fit = max(1, int((box_bottom - box_top) / row_h2))
        d.rectangle([m, box_top, w - m, box_top + row_h2 * n_fit], outline=style.rule, width=1)
        dy = box_top
        for label, val in details[:n_fit]:
            d.line([(m, dy), (w - m, dy)], fill=style.rule, width=1) if dy != box_top else None
            draw_rtl(d, (w - m - 16, dy + row_h2 / 2), label + ":", detail_label_font, style.rule, anchor="rm")
            draw_rtl(d, (w - m - 16 - 340, dy + row_h2 / 2), val, detail_font, style.rule, anchor="rm")
            dy += row_h2
        y = box_top + row_h2 * n_fit

    # small geometric "logo" mark -- an abstract shield/seal shape, NOT a
    # copy of any real logo. Placed near the bottom like a real stamp/seal
    # next to a signature, well clear of the body-text column above.
    if rng.random() < 0.7:
        seal_y = min(footer_top - 60, y + 60)
        if seal_y > y + 20:
            _draw_geo_seal(d, (w - m - 55, seal_y), 38, style.accent)

    d.rectangle([m, footer_top, w - m, h - m], fill=style.bar_fill)
    foot_font = get_font(style.base_font_size - 4, family_idx=style.font_family)
    footer_text = f"نشانی: خیابان اصلی، پلاک {rand_number(rng,1,300)} - تلفن: {farsi_digits('021-8800'+str(rng.randint(1000,9999)))}"
    draw_rtl(d, (w - m - 20, h - m - footer_h // 2), footer_text, foot_font, (255, 255, 255), anchor="rm")

    return im


def _draw_geo_seal(d: ImageDraw.ImageDraw, center, r, color):
    cx, cy = center
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
    d.ellipse([cx - r * 0.65, cy - r * 0.65, cx + r * 0.65, cy + r * 0.65], outline=color, width=2)
    pts = []
    n = 5
    for i in range(n):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        pts.append((cx + r * 0.4 * math.cos(ang), cy + r * 0.4 * math.sin(ang)))
    d.polygon(pts, fill=color)


def gen_tabular_document(rng: random.Random, style: PageStyle) -> Image.Image:
    """A real table: header row, gridlines, numeric cells -- columns laid
    out right-to-left since 'ردیف' (row #) conventionally sits at the
    right in Persian tables."""
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, style.margin

    title_font = get_font(style.base_font_size + 8, bold=True, family_idx=style.font_family)
    title = f"صورت ریز اقلام {rng.choice(['خریداری‌شده', 'درخواستی', 'مزایده'])}"
    draw_rtl(d, (w - m, m), title, title_font, style.rule, anchor="ra")

    n_rows = rng.randint(8, 16)
    table_top = m + title_font.size + 40
    table_bottom = h - m
    row_h = (table_bottom - table_top) / (n_rows + 1)
    if row_h < 26:
        n_rows = max(4, int((table_bottom - table_top) / 26) - 1)
        row_h = (table_bottom - table_top) / (n_rows + 1)

    col_fracs = [0.07, 0.34, 0.11, 0.11, 0.18, 0.19]
    table_w = w - 2 * m
    col_widths = [f * table_w for f in col_fracs]
    # right-to-left column x boundaries
    col_x = [w - m]
    for cw in col_widths:
        col_x.append(col_x[-1] - cw)

    header_font = get_font(style.base_font_size - 2, bold=True, family_idx=style.font_family)
    cell_font = get_font(style.base_font_size - 4, family_idx=style.font_family)

    d.rectangle([m, table_top, w - m, table_top + row_h], fill=style.bar_fill)
    for i, htext in enumerate(TABLE_HEADERS):
        cx = (col_x[i] + col_x[i + 1]) / 2
        draw_rtl(d, (cx, table_top + row_h / 2), htext, header_font, (255, 255, 255), anchor="mm")

    zebra = rng.random() < 0.5
    for r in range(n_rows):
        y0 = table_top + row_h * (r + 1)
        y1 = y0 + row_h
        if zebra and r % 2 == 1:
            d.rectangle([m, y0, w - m, y1], fill=tuple(min(255, c + 6) if c > 240 else max(0, c - 8) for c in style.bg))
        qty = rng.randint(1, 500)
        unit_price = rng.randint(10, 900) * 1000
        total = qty * unit_price
        row_vals = [
            farsi_digits(str(r + 1)),
            rng.choice(ITEM_NAMES),
            farsi_digits(str(qty)),
            rng.choice(UNITS),
            farsi_digits(f"{unit_price:,}").replace(",", "٬"),
            farsi_digits(f"{total:,}").replace(",", "٬"),
        ]
        for i, val in enumerate(row_vals):
            cx = (col_x[i] + col_x[i + 1]) / 2
            draw_rtl(d, (cx, (y0 + y1) / 2), val, cell_font, style.rule, anchor="mm")

    # gridlines
    total_bottom = table_top + row_h * (n_rows + 1)
    for x in col_x:
        d.line([(x, table_top), (x, total_bottom)], fill=style.rule, width=1)
    for r in range(n_rows + 2):
        y = table_top + row_h * r
        d.line([(m, y), (w - m, y)], fill=style.rule, width=1)

    return im


def gen_form_layout(rng: random.Random, style: PageStyle) -> Image.Image:
    """Dense grid of small bordered field boxes with labels, plus a couple
    of radio-button groups -- mimics a web-form screenshot."""
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, style.margin

    bar_h = rng.randint(55, 75)
    d.rectangle([0, 0, w, bar_h], fill=style.bar_fill)
    hdr_font = get_font(style.base_font_size + 6, bold=True, family_idx=style.font_family)
    draw_rtl(d, (w - m, bar_h / 2), "فرم درخواست ثبت‌نام متقاضی", hdr_font, (255, 255, 255), anchor="rm")

    label_font = get_font(style.base_font_size - 6, family_idx=style.font_family)
    box_h = rng.randint(38, 50)
    n_cols = rng.choice([2, 3])
    gutter = 30
    col_w = (w - 2 * m - gutter * (n_cols - 1)) / n_cols

    row_gap = box_h + 34
    radio_block_h = 150
    avail_h = h - bar_h - 40 - radio_block_h
    n_rows = max(2, int(avail_h / row_gap))
    n_fields = n_rows * n_cols

    # Real web forms repeat/pad past the "natural" field list on a tall
    # page (extra rows, confirmation fields, etc.) -- sample with
    # replacement once the fixed label bank is exhausted so tall pages
    # stay visually dense instead of trailing off into blank space.
    pool = FORM_FIELD_LABELS[:]
    rng.shuffle(pool)
    fields = (pool * (n_fields // len(pool) + 1))[:n_fields]

    y = bar_h + 40
    for idx, label in enumerate(fields):
        col = idx % n_cols
        if col == 0 and idx > 0:
            y += row_gap
        col_right = w - m - col * (col_w + gutter)
        col_left = col_right - col_w
        draw_rtl(d, (col_right, y), label + ":", label_font, style.rule, anchor="ra")
        box_top = y + label_font.size + 6
        d.rectangle([col_left, box_top, col_right, box_top + box_h], outline=style.rule, width=1)
    y += row_gap + box_h + 20

    # radio groups
    for gname, opts in rng.sample(RADIO_GROUPS, k=min(2, len(RADIO_GROUPS))):
        draw_rtl(d, (w - m, y), gname + ":", label_font, style.rule, anchor="ra")
        x = w - m - 140
        for opt in opts:
            r = 8
            d.ellipse([x - r, y + 4, x + r, y + 4 + 2 * r], outline=style.rule, width=2)
            draw_rtl(d, (x - r - 8, y + 4 + r), opt, label_font, style.rule, anchor="rm")
            x -= 140
        y += row_gap * 0.7

    return im


def gen_plain_letter(rng: random.Random, style: PageStyle) -> Image.Image:
    """Letterhead block, basmaleh, subject line, body paragraphs, signature
    line -- a plain official correspondence layout."""
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, style.margin

    org = rng.choice(ORG_NAMES)
    letterhead_font = get_font(style.base_font_size + 4, bold=True, family_idx=style.font_family)
    draw_rtl(d, (w - m, m), org, letterhead_font, style.accent, anchor="ra")
    sub_font = get_font(style.base_font_size - 8, family_idx=style.font_family)
    draw_rtl(d, (w - m, m + letterhead_font.size + 10), "معاونت اداری و مالی", sub_font, style.rule, anchor="ra")
    if rng.random() < 0.6:
        _draw_geo_seal(d, (m + 45, m + 35), 32, style.accent)
    d.line([(m, m + letterhead_font.size + 55), (w - m, m + letterhead_font.size + 55)], fill=style.rule, width=2)

    y = m + letterhead_font.size + 90
    basmaleh_font = get_font(style.base_font_size, family_idx=style.font_family)
    bw = text_width(d, BASMALEH, basmaleh_font)
    draw_rtl(d, (w / 2 + bw / 2, y), BASMALEH, basmaleh_font, style.rule, anchor="ra")
    y += basmaleh_font.size * 2.2

    meta_font = get_font(style.base_font_size - 4, family_idx=style.font_family)
    draw_rtl(d, (w - m, y), f"شماره: {rand_number(rng, 1000, 9999)}/{rand_number(rng,100,999)}", meta_font, style.rule, anchor="ra")
    y += meta_font.size * 1.6
    draw_rtl(d, (w - m, y), f"تاریخ: {rand_date(rng)}", meta_font, style.rule, anchor="ra")
    y += meta_font.size * 1.6
    draw_rtl(d, (w - m, y), f"موضوع: {rng.choice(LETTER_SUBJECTS)}", meta_font, style.rule, anchor="ra")
    y += meta_font.size * 2.4

    body_font = get_font(style.base_font_size, family_idx=style.font_family)
    n_para = rng.randint(2, 4)
    chosen = rng.sample(BODY_SENTENCES, k=min(n_para, len(BODY_SENTENCES)))
    for sent in chosen:
        text = sent.format(org=org)
        y = draw_rtl_paragraph(d, text, body_font, w - m, y, w - 2 * m, style.rule, line_gap=1.7)
        y += body_font.size

    sig_font = get_font(style.base_font_size - 2, family_idx=style.font_family)
    sig_y = h - m - 80
    d.line([(w - m - 220, sig_y), (w - m, sig_y)], fill=style.rule, width=1)
    draw_rtl(d, (w - m - 30, sig_y + 12), rng.choice(SIGNATURE_LABELS), sig_font, style.rule, anchor="ra")

    return im


def gen_spreadsheet(rng: random.Random, style: PageStyle) -> Image.Image:
    """Dense uniform gridlines with numeric data -- mimics an Excel
    screenshot: light gray toolbar strip, lettered columns, numbered rows."""
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, style.margin
    toolbar_h = 40
    d.rectangle([0, 0, w, toolbar_h], fill=(235, 235, 235))
    d.line([(0, toolbar_h), (w, toolbar_h)], fill=(180, 180, 180), width=1)
    for i in range(6):
        d.rectangle([12 + i * 30, 10, 12 + i * 30 + 20, 30], outline=(180, 180, 180))

    grid_top = toolbar_h + 26
    row_h = rng.randint(24, 32)
    col_w = rng.randint(70, 100)
    n_rows = int((h - grid_top - 10) / row_h)
    n_cols = int((w - 40) / col_w)

    row_label_w = 34
    col_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    header_font = get_font(15, bold=True, family_idx=style.font_family)
    cell_font = get_font(15, family_idx=style.font_family)

    grid_left = 10
    grid_right = grid_left + row_label_w + n_cols * col_w
    grid_right = min(grid_right, w - 10)
    n_cols = max(1, int((grid_right - grid_left - row_label_w) / col_w))

    # The interior always renders as a plain white spreadsheet canvas,
    # independent of style.bg/style.dark: a screenshot of a spreadsheet app
    # keeps its own white cell background no matter what desktop/viewer
    # chrome surrounds it, and the hardcoded dark cell-text colors below
    # would be unreadable if this showed through to a dark page ground.
    d.rectangle([grid_left, grid_top, grid_right, grid_top + row_h * (n_rows + 1)], fill=(255, 255, 255))

    # column header row (letters stay LTR -- matches real spreadsheet apps)
    d.rectangle([grid_left, grid_top, grid_right, grid_top + row_h], fill=(240, 240, 240))
    for c in range(n_cols):
        x0 = grid_left + row_label_w + c * col_w
        d.text((x0 + col_w / 2, grid_top + row_h / 2), col_letters[c % 26], font=header_font,
               fill=(60, 60, 60), anchor="mm")

    for r in range(n_rows):
        y0 = grid_top + row_h * (r + 1)
        d.rectangle([grid_left, y0, grid_left + row_label_w, y0 + row_h], fill=(240, 240, 240))
        d.text((grid_left + row_label_w / 2, y0 + row_h / 2), str(r + 1), font=header_font,
               fill=(60, 60, 60), anchor="mm")
        for c in range(n_cols):
            x0 = grid_left + row_label_w + c * col_w
            if rng.random() < 0.82:
                val = farsi_digits(str(rng.randint(0, 999999)))
                draw_rtl(d, (x0 + col_w - 6, y0 + row_h / 2), val, cell_font, (30, 30, 30), anchor="rm")

    grid_bottom = grid_top + row_h * (n_rows + 1)
    for c in range(n_cols + 1):
        x = grid_left + row_label_w + c * col_w
        d.line([(x, grid_top), (x, grid_bottom)], fill=(210, 210, 210), width=1)
    d.line([(grid_left, grid_top), (grid_left, grid_bottom)], fill=(210, 210, 210), width=1)
    for r in range(n_rows + 2):
        y = grid_top + row_h * r
        d.line([(grid_left, y), (grid_right, y)], fill=(210, 210, 210), width=1)

    return im


def gen_dense_text(rng: random.Random, style: "PageStyle") -> Image.Image:
    """A wall of Persian body text -- contract / report / terms-and-conditions
    page, filled edge to edge with minimal whitespace.

    Exists specifically to train the watermark-under-heavy-text case. That is
    the failure mode of the current detector: it finds marks sitting on clear
    paper but misses every instance overlapping dense body text, because the
    text destroys the local contrast the mark would otherwise stand out
    against. A background corpus of mostly-whitespace pages would never teach
    a model to handle it, so this archetype deliberately leaves the mark
    almost nowhere clean to land.
    """
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, style.margin

    two_col = rng.random() < 0.35
    body_size = max(11, style.base_font_size - rng.choice([4, 6, 8]))
    body_font = get_font(body_size, family_idx=style.font_family)
    head_font = get_font(style.base_font_size + 2, bold=True, family_idx=style.font_family)
    line_gap = rng.uniform(1.25, 1.5)  # tight leading -> more text per page

    org = rng.choice(ORG_NAMES)
    y = m
    draw_rtl(d, (w - m, y), org, head_font, style.accent, anchor="ra")
    y += head_font.size * 1.8
    d.line([(m, y), (w - m, y)], fill=style.rule, width=1)
    y += head_font.size * 0.8

    if two_col:
        gutter = int(w * 0.045)
        col_w = (w - 2 * m - gutter) / 2
        columns = [(w - m, col_w), (w - m - col_w - gutter, col_w)]
    else:
        columns = [(w - m, w - 2 * m)]

    sec_font = get_font(body_size + 3, bold=True, family_idx=style.font_family)
    for right_x, col_w in columns:
        cy = y
        # Fill until the column runs out of vertical room, rather than a fixed
        # paragraph count -- page sizes vary, and the point is a full page.
        guard = 0
        while cy < h - m - body_font.size * 2 and guard < 60:
            guard += 1
            if rng.random() < 0.22:
                draw_rtl(d, (right_x, cy), rng.choice(SECTION_TITLES), sec_font, style.accent, anchor="ra")
                cy += sec_font.size * 1.7
                continue
            # BODY_SENTENCES carries an "{org}" placeholder; the other
            # archetypes format it, and skipping that renders a literal
            # "{org}" into the page.
            text = " ".join(rng.choices(BODY_SENTENCES, k=rng.randint(2, 4))).format(org=org)
            cy = draw_rtl_paragraph(d, text, body_font, right_x, cy, col_w, style.rule, line_gap=line_gap)
            cy += body_font.size * rng.uniform(0.3, 0.8)

    return im


SECTION_TITLES = [
    "ماده ۱ - موضوع قرارداد",
    "ماده ۲ - مدت اجرا",
    "ماده ۳ - مبلغ و نحوه پرداخت",
    "ماده ۴ - تعهدات طرفین",
    "ماده ۵ - فسخ قرارداد",
    "تبصره",
    "شرایط عمومی",
    "توضیحات تکمیلی",
]

PHOTO_REPORT_TITLES = [
    "گزارش تصویری پیشرفت پروژه",
    "پیوست مستندات فنی و تصویری",
    "گزارش مستندسازی بازدید میدانی",
    "ضمیمه تصاویر و نمودارهای گزارش",
]

FIGURE_CAPTIONS = [
    "شکل {n}- نمودار روند تغییرات در دوره گزارش",
    "تصویر {n}- نمای کلی از محل اجرای پروژه",
    "شکل {n}- نقشه موقعیت جغرافیایی طرح",
    "تصویر {n}- مستندات تصویری بازدید میدانی",
    "نمودار {n}- مقایسه شاخص‌های عملکردی",
]

SLIDE_TITLES = [
    "گزارش عملکرد سالانه",
    "برنامه راهبردی توسعه",
    "خلاصه نتایج پروژه",
    "چشم‌انداز و اهداف سازمانی",
    "تحلیل بازار و رقبا",
    "جمع‌بندی و پیشنهادها",
    "روند رشد و شاخص‌های کلیدی",
]

SLIDE_BULLETS = [
    "افزایش بهره‌وری در واحدهای عملیاتی",
    "کاهش هزینه‌های جاری به میزان قابل توجه",
    "توسعه زیرساخت فناوری اطلاعات",
    "ارتقای کیفیت خدمات مشتریان",
    "برنامه‌ریزی برای توسعه بازار منطقه‌ای",
    "تقویت تیم‌های تخصصی و آموزش کارکنان",
    "بهبود فرآیندهای داخلی گزارش‌دهی",
    "گسترش همکاری‌های بین‌بخشی",
]


# ---------------------------------------------------------------------------
# Procedural imagery -- for `photo_report`'s figures/photos. No network, no
# external assets: everything below is generated from noise/gradient math,
# the same spirit as the rest of this module's "programmatic generation
# instead of scraping" approach (see module docstring).
# ---------------------------------------------------------------------------

def _value_noise_field(w: int, h: int, rng: random.Random, octaves: int = 4, persistence: float = 0.55) -> np.ndarray:
    """Multi-octave value noise in [0, 1], shape (h, w).

    Standard "value noise" trick: draw a small grid of independent random
    values per octave and let PIL's bicubic resize do the smooth
    interpolation up to full resolution (that's the same role a lattice
    interpolation function plays in classic value/Perlin noise), then sum
    octaves at halving amplitude/doubling frequency. Deterministic given
    `rng`, and cheap since every octave's source grid is tiny.
    """
    field = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total_amp = 0.0
    grid_h = 3
    for _ in range(max(1, octaves)):
        gh = max(2, grid_h)
        gw = max(2, int(round(gh * w / max(1, h))))
        grid = np.array([[rng.random() for _ in range(gw)] for _ in range(gh)], dtype=np.float32)
        small = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        big = small.resize((w, h), resample=Image.BICUBIC)
        field += amp * (np.asarray(big, dtype=np.float32) / 255.0)
        total_amp += amp
        amp *= persistence
        grid_h *= 2
    field /= max(total_amp, 1e-6)
    field -= field.min()
    mx = field.max()
    if mx > 1e-6:
        field /= mx
    return field


def _linear_gradient_array(w: int, h: int, angle_deg: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = math.radians(angle_deg)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    proj -= proj.min()
    mx = proj.max()
    if mx > 1e-6:
        proj /= mx
    return proj


def _radial_gradient_array(w: int, h: int, cx: float, cy: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mx = d.max()
    if mx > 1e-6:
        d /= mx
    return d


def _colorize_field(field: np.ndarray, c0: Tuple[int, int, int], c1: Tuple[int, int, int]) -> np.ndarray:
    """Linearly interpolate an HxW field of [0,1] values between two colors,
    per-pixel, returning an HxWx3 uint8 array."""
    c0a = np.array(c0, dtype=np.float32)
    c1a = np.array(c1, dtype=np.float32)
    out = c0a[None, None, :] + (c1a - c0a)[None, None, :] * field[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _random_image_palette(rng: random.Random) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool]:
    """Picks the two colorize endpoints for a procedural figure, and whether
    it's near-monochrome. Deliberately spans bright / dark / colorful /
    near-monochrome outcomes -- the whole point of `photo_report` is that
    the watermark has to be learned against imagery that varies in
    luminance and saturation, not one washed-out noise texture repeated."""
    kind = rng.choices(
        ["colorful", "bright_duo", "dark_duo", "mono_dark", "mono_bright"],
        weights=[0.35, 0.2, 0.2, 0.125, 0.125],
    )[0]
    if kind == "colorful":
        c0 = tuple(rng.randint(20, 235) for _ in range(3))
        c1 = tuple(rng.randint(20, 235) for _ in range(3))
        return c0, c1, False
    if kind == "bright_duo":
        base = rng.choice(ACCENT_COLORS)
        light = tuple(min(255, int(c * 0.4 + 180)) for c in base)
        c1 = tuple(min(255, c + 40) for c in light)
        return light, c1, False
    if kind == "dark_duo":
        base = rng.choice(ACCENT_COLORS)
        d0 = tuple(int(c * 0.3) for c in base)
        d1 = tuple(int(c * 0.15) for c in base)
        return d0, d1, False
    if kind == "mono_dark":
        g0 = rng.randint(10, 40)
        g1 = min(255, g0 + rng.randint(40, 90))
        return (g0, g0, g0), (g1, g1, g1), True
    g0 = rng.randint(150, 195)  # mono_bright
    g1 = min(255, g0 + rng.randint(40, 90))
    return (g0, g0, g0), (g1, g1, g1), True


def _add_soft_blobs(arr: np.ndarray, rng: random.Random, c0, c1) -> np.ndarray:
    """Composites a handful of heavily-blurred, alpha-masked colored
    ellipses onto an existing image array -- the "soft coloured blob"
    look of an out-of-focus photo or a defocused bokeh background."""
    h, w = arr.shape[:2]
    base = Image.fromarray(arr, "RGB")
    n = rng.randint(3, 7)
    for _ in range(n):
        r = rng.uniform(0.12, 0.35) * max(w, h)
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        color = c1 if rng.random() < 0.5 else c0
        jitter = tuple(int(max(0, min(255, c + rng.randint(-30, 30)))) for c in color)
        layer = Image.new("RGB", (w, h), (0, 0, 0))
        ImageDraw.Draw(layer).ellipse([cx - r, cy - r, cx + r, cy + r], fill=jitter)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=rng.randint(60, 140))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, r * 0.4)))
        base = Image.composite(layer, base, mask)
    return np.asarray(base)


def _draw_synthetic_chart(w: int, h: int, rng: random.Random, mono: bool) -> Image.Image:
    """A simple chart-report look: axes plus either a bar chart or a
    trend polyline with point markers, on its own light or dark panel."""
    dark_panel = rng.random() < 0.4
    panel_bg = (24, 26, 30) if dark_panel else (250, 250, 248)
    ink = (225, 225, 225) if dark_panel else (40, 40, 40)
    im = Image.new("RGB", (w, h), panel_bg)
    d = ImageDraw.Draw(im)
    pad = max(8, int(min(w, h) * 0.1))
    ax_left, ax_bottom = pad, h - pad
    ax_right, ax_top = w - pad, pad
    d.line([(ax_left, ax_top), (ax_left, ax_bottom)], fill=ink, width=2)
    d.line([(ax_left, ax_bottom), (ax_right, ax_bottom)], fill=ink, width=2)
    palette = [(120, 120, 120), (170, 170, 170), (90, 90, 90)] if mono else ACCENT_COLORS

    if rng.random() < 0.55:
        n = rng.randint(4, 8)
        col_w = (ax_right - ax_left) / n
        bw = col_w * 0.6
        for i in range(n):
            val = rng.uniform(0.15, 1.0)
            bh = val * (ax_bottom - ax_top - 4)
            x0 = ax_left + i * col_w + col_w * 0.2
            d.rectangle([x0, ax_bottom - bh, x0 + bw, ax_bottom], fill=rng.choice(palette))
    else:
        n = rng.randint(5, 9)
        pts = []
        for i in range(n):
            x = ax_left + i * (ax_right - ax_left) / (n - 1)
            val = rng.uniform(0.1, 0.95)
            y = ax_bottom - val * (ax_bottom - ax_top - 4)
            pts.append((x, y))
        line_color = rng.choice(palette)
        d.line(pts, fill=line_color, width=3, joint="curve")
        for (px, py) in pts:
            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=rng.choice(palette))
    return im


def _gen_figure_image(w: int, h: int, rng: random.Random, allow_chart: bool = True) -> Image.Image:
    """One procedurally generated 'photo/figure' panel of exactly (w, h):
    multi-octave value noise, a gradient, soft colour blobs, or (if
    `allow_chart`) a bars/line chart -- picked and colour-graded per call so
    `photo_report` spans bright, dark, colorful and near-monochrome imagery
    instead of one recognizable texture repeated everywhere."""
    w, h = max(2, int(w)), max(2, int(h))
    kinds = ["noise", "gradient", "blobs"]
    if allow_chart:
        kinds.append("chart")
    kind = rng.choice(kinds)
    c0, c1, mono = _random_image_palette(rng)

    if kind == "chart":
        return _draw_synthetic_chart(w, h, rng, mono)

    if kind == "gradient":
        if rng.random() < 0.5:
            field = _linear_gradient_array(w, h, rng.uniform(0, 360))
        else:
            field = _radial_gradient_array(w, h, rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h)
    else:
        field = _value_noise_field(w, h, rng, octaves=rng.randint(3, 5))

    arr = _colorize_field(field, c0, c1)
    if kind == "blobs" or (kind == "noise" and rng.random() < 0.4):
        arr = _add_soft_blobs(arr, rng, c0, c1)

    # A touch of sensor-style grain on every figure -- without it, a
    # narrow-contrast mono/gradient pick can render so flat it is
    # indistinguishable from a blank page, which would defeat the whole
    # point of this archetype (imagery for the watermark to sit on).
    grain_sigma = rng.uniform(2.0, 6.0)
    grain = np.random.normal(0, grain_sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def gen_blank_page(rng: random.Random, style: PageStyle) -> Image.Image:
    """A near-empty page -- the "watermark on empty background" case, which
    was previously completely absent from this corpus.

    Every other archetype gives the segmenter dense structure to key off
    of (a rule, a table edge, a text line) right where a mark tends to
    land. Without an archetype at the opposite extreme -- a mark alone on
    bare paper -- the model never sees the easiest geometric case, and easy
    cases still need coverage or a detector tuned for clutter can
    paradoxically stumble on a blank cover sheet or an appendix page. Kept
    genuinely sparse on purpose: at most a letterhead rule, a page number,
    a lone signature block or a single heading, never more than two of
    those four, so ink coverage stays at a few percent at most.
    """
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, style.margin

    n_elems = rng.choices([0, 1, 2], weights=[0.3, 0.45, 0.25])[0]
    elems = rng.sample(["letterhead", "pagenum", "signature", "heading"], k=n_elems)

    if "letterhead" in elems:
        line_y = m + rng.randint(0, 40)
        if rng.random() < 0.5:
            org_font = get_font(style.base_font_size - 6, family_idx=style.font_family)
            draw_rtl(d, (w - m, line_y - org_font.size - 6), rng.choice(ORG_NAMES), org_font, style.rule, anchor="ra")
        d.line([(m, line_y), (w - m, line_y)], fill=style.rule, width=1)

    if "heading" in elems:
        head_font = get_font(style.base_font_size + 6, bold=True, family_idx=style.font_family)
        y0 = m + rng.randint(60, 140)
        title = rng.choice(TITLES + LETTER_SUBJECTS)
        tw = text_width(d, title, head_font)
        draw_rtl(d, (w / 2 + tw / 2, y0), title, head_font, style.rule, anchor="ra")

    if "signature" in elems:
        sig_font = get_font(style.base_font_size - 2, family_idx=style.font_family)
        sig_y = h - m - rng.randint(100, 220)
        d.line([(w - m - 220, sig_y), (w - m, sig_y)], fill=style.rule, width=1)
        draw_rtl(d, (w - m - 30, sig_y + 12), rng.choice(SIGNATURE_LABELS), sig_font, style.rule, anchor="ra")

    if "pagenum" in elems:
        pg_font = get_font(style.base_font_size - 6, family_idx=style.font_family)
        pg = farsi_digits(str(rng.randint(1, 40)))
        draw_rtl(d, (w / 2, h - m - pg_font.size - 6), pg, pg_font, style.rule, anchor="ma")

    return im


def gen_photo_report(rng: random.Random, style: PageStyle) -> Image.Image:
    """A page whose content includes raster IMAGE regions, not just text --
    the "watermark over a photo/figure" case.

    Two sub-modes: most pages frame one or two procedurally generated
    figures (noise/gradient/blob/chart imagery, see `_gen_figure_image`)
    inside an otherwise normal text page, with a Persian caption under
    each; the rest are full-bleed, where the generated imagery covers the
    entire page like a scanned photograph or a screenshot, with little or
    no text. Both matter for the same reason `blank_page` does: a mark
    sitting on a photograph destroys different local cues than one sitting
    on paper, and the segmenter never saw that cue distribution before.
    """
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, style.margin

    if rng.random() < 0.4:
        scene = _gen_figure_image(w, h, rng, allow_chart=False)
        im.paste(scene, (0, 0))
        d = ImageDraw.Draw(im)
        if rng.random() < 0.35:
            bar_h = rng.randint(46, 64)
            d.rectangle([0, h - bar_h, w, h], fill=(15, 15, 18))
            cap_font = get_font(style.base_font_size - 4, family_idx=style.font_family)
            cap = rng.choice(FIGURE_CAPTIONS).format(n=farsi_digits(str(rng.randint(1, 9))))
            draw_rtl(d, (w - m, h - bar_h / 2), cap, cap_font, (235, 235, 235), anchor="rm")
        return im

    title_font = get_font(style.base_font_size + 8, bold=True, family_idx=style.font_family)
    draw_rtl(d, (w - m, m), rng.choice(PHOTO_REPORT_TITLES), title_font, style.rule, anchor="ra")
    y = m + title_font.size + 30

    n_figs = rng.choice([1, 1, 2])
    gutter = 20 if n_figs == 2 else 0
    fig_w = int((w - 2 * m - gutter * (n_figs - 1)) / n_figs)
    fig_h = int(fig_w * rng.uniform(0.55, 0.8))
    cap_font = get_font(style.base_font_size - 6, family_idx=style.font_family)
    x_right = w - m
    max_bottom = y
    for i in range(n_figs):
        x_left = x_right - fig_w
        fig_img = _gen_figure_image(fig_w, fig_h, rng)
        im.paste(fig_img, (int(x_left), int(y)))
        d.rectangle([x_left, y, x_right, y + fig_h], outline=style.rule, width=2)
        cap = rng.choice(FIGURE_CAPTIONS).format(n=farsi_digits(str(i + 1)))
        cap_y = y + fig_h + 8
        draw_rtl(d, ((x_left + x_right) / 2, cap_y), cap, cap_font, style.rule, anchor="ma")
        max_bottom = max(max_bottom, cap_y + cap_font.size * 1.6)
        x_right = x_left - gutter
    y = max_bottom + 20

    org = rng.choice(ORG_NAMES)
    body_font = get_font(style.base_font_size, family_idx=style.font_family)
    n_para = rng.randint(1, 3)
    for sent in rng.sample(BODY_SENTENCES, k=min(n_para, len(BODY_SENTENCES))):
        text = sent.format(org=org)
        y = draw_rtl_paragraph(d, text, body_font, w - m, y, w - 2 * m, style.rule, line_gap=1.6)
        y += body_font.size * 0.8

    return im


def gen_slide_deck(rng: random.Random, style: PageStyle) -> Image.Image:
    """A landscape presentation-slide look: a large title, a short bullet
    list, optionally a gradient wash over the ground, and a footer/logo
    bar -- the archetype `PageStyle.random(prefer_landscape=True)` biases
    towards the landscape entries in PAGE_SIZES for.

    Exists because a watermark on a slide export (a deck screenshot
    attached to an email, a PDF-per-slide report) is a visually distinct
    regime from a business letter: huge sparse type, a handful of bullets,
    and -- unlike every other archetype here -- a dark or saturated ground
    is the norm rather than the rare case, since dark slide themes are
    extremely common in the real corpus this augments.
    """
    im, d = _new_canvas(style)
    w, h, m = style.w, style.h, int(style.margin * 0.8)

    if rng.random() < 0.5:
        if rng.random() < 0.6:
            field = _linear_gradient_array(w, h, rng.uniform(0, 360))
        else:
            field = _radial_gradient_array(w, h, rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h)
        wash = _colorize_field(field * rng.uniform(0.35, 0.75), style.bg, style.accent)
        im = Image.fromarray(wash, "RGB")
        d = ImageDraw.Draw(im)

    title_font = get_font(style.base_font_size + 20, bold=True, family_idx=style.font_family)
    title = rng.choice(SLIDE_TITLES)
    y = m + rng.randint(20, 60)
    draw_rtl(d, (w - m, y), title, title_font, style.rule, anchor="ra")
    y += title_font.size * 1.6
    d.line([(w - m, y), (w - m - rng.randint(200, 420), y)], fill=style.accent, width=4)
    y += 40

    bullet_font = get_font(style.base_font_size + 2, family_idx=style.font_family)
    n_bul = rng.randint(3, 6)
    for text in rng.sample(SLIDE_BULLETS, k=min(n_bul, len(SLIDE_BULLETS))):
        r = 6
        d.ellipse([w - m - r * 2, y + bullet_font.size * 0.35, w - m, y + bullet_font.size * 0.35 + r * 2],
                  fill=style.accent)
        y = draw_rtl_paragraph(d, text, bullet_font, w - m - r * 4 - 14, y, w - 2 * m - r * 4 - 14,
                                style.rule, line_gap=1.5)
        y += bullet_font.size * 0.9

    if rng.random() < 0.55:
        bar_h = rng.randint(30, 46)
        d.rectangle([0, h - bar_h, w, h], fill=style.bar_fill)
        foot_font = get_font(style.base_font_size - 8, family_idx=style.font_family)
        draw_rtl(d, (w - m, h - bar_h / 2), rng.choice(ORG_NAMES), foot_font, (255, 255, 255), anchor="rm")
        _draw_geo_seal(d, (m + 20, h - bar_h / 2), int(bar_h * 0.32), (255, 255, 255))

    return im


ARCHETYPES = {
    "tender_notice": gen_tender_notice,
    "tabular_document": gen_tabular_document,
    "form_layout": gen_form_layout,
    "plain_letter": gen_plain_letter,
    "spreadsheet": gen_spreadsheet,
    "dense_text": gen_dense_text,
    "blank_page": gen_blank_page,
    "photo_report": gen_photo_report,
    "slide_deck": gen_slide_deck,
}

# Relative frequency in a generated corpus, weights sum to 100 so each value
# doubles as its exact percentage. Grouped roughly as:
#   dense/text-heavy   (dense_text, tender_notice, plain_letter)      -> 47%
#   structured/tabular (tabular_document, form_layout, spreadsheet)  -> 21%
#   photo_report                                                     -> 13%
#   blank_page                                                       -> 11%
#   slide_deck                                                       ->  8%
# Text-heavy archetypes still dominate: the watermark-over-dense-text case
# is both the hardest for the model and the one the current detector fails,
# so it needs the most coverage (dense_text highest of the three). The three
# newer archetypes (blank_page/photo_report/slide_deck) exist to cover
# background regimes that were previously entirely absent -- empty pages,
# raster imagery, and slide decks -- so they get real weight even though
# they're individually smaller than the text-heavy group. Dark/coloured
# grounds (see PageStyle.random) cut across ALL of these, not just one
# archetype, so they aren't a separate weighted bucket here.
ARCHETYPE_WEIGHTS = {
    "dense_text": 19,
    "tender_notice": 14,
    "plain_letter": 14,
    "tabular_document": 11,
    "form_layout": 5,
    "spreadsheet": 5,
    "photo_report": 13,
    "blank_page": 11,
    "slide_deck": 8,
}


# ---------------------------------------------------------------------------
# "Scanned" degradation
# ---------------------------------------------------------------------------

def apply_scan_effects(im: Image.Image, rng: random.Random, dark: bool = False) -> Image.Image:
    """Slight rotation, mild noise, soft blur, off-white cast -- approximates
    a phone photo / flatbed scan of a printed page rather than a clean
    digital render.

    `dark` matters here: a scanned/photographed DARK page (a slide, a
    dark-themed screenshot) doesn't get the paper-white cast a scanned
    printed page does. Using the light cast unconditionally would paint
    bright white triangles into whatever corners the rotation exposes and
    wash the whole page towards white, defeating the point of the dark
    grounds in PageStyle -- so both the rotation fill and the color-cast
    tint switch to a dark neutral instead when the page itself is dark.
    """
    angle = rng.uniform(-2.2, 2.2)
    corner_fill = (18, 18, 20) if dark else tuple(int(c * 0.97) for c in (250, 248, 244))
    im = im.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=corner_fill)

    arr = np.asarray(im).astype(np.float32)
    noise_sigma = rng.uniform(2.5, 7.0)
    noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255)

    # gentle paper/dark-ambient color cast + slight contrast pull-in
    tint = np.array([40, 40, 44], dtype=np.float32) if dark else np.array([250, 247, 240], dtype=np.float32)
    mix = rng.uniform(0.03, 0.09)
    arr = arr * (1 - mix) + tint * mix
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")

    if rng.random() < 0.8:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
    return im


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_one(rng: random.Random, archetype: Optional[str] = None) -> Tuple[Image.Image, str, PageStyle]:
    name = archetype or rng.choice(list(ARCHETYPES.keys()))
    style = PageStyle.random(rng, prefer_landscape=(name == "slide_deck"))
    im = ARCHETYPES[name](rng, style)
    if style.scanned:
        im = apply_scan_effects(im, rng, dark=style.dark)
    return im, name, style


def generate_dataset(out_dir: Path, count: int, seed: int = 0) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # Weighted round-robin rather than uniform: deterministic for a given
    # seed, but honours ARCHETYPE_WEIGHTS so text-heavy pages dominate.
    names: List[str] = []
    for nm, wt in ARCHETYPE_WEIGHTS.items():
        names.extend([nm] * wt)
    names = names or list(ARCHETYPES.keys())
    # Shuffle the run-length list before walking it. Without this, `names` is
    # a run of identical entries per archetype in dict order, so
    # `names[i % len(names)]` emits contiguous BLOCKS -- and any count below
    # the first archetype's weight yields that archetype and nothing else
    # (measured: count=18 against a weight of 19 produced 18/18 dense_text,
    # zero coverage of the other eight). Shuffling once, from the same seeded
    # rng, keeps generation deterministic per seed while making any prefix of
    # the sequence an approximately representative sample of the weights.
    rng.shuffle(names)
    written: List[Path] = []
    for i in range(count):
        archetype = names[i % len(names)]
        im, name, style = generate_one(rng, archetype=archetype)
        ext = "jpg" if style.scanned else "png"
        fname = f"bg_{i:03d}_{name}_{style.size_name}.{ext}"
        path = out_dir / fname
        if ext == "jpg":
            im.save(path, quality=rng.randint(70, 92))
        else:
            im.save(path)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "wm_backgrounds"),
                     help="Output directory (default: <repo_root>/wm_backgrounds)")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    written = generate_dataset(out_dir, args.count, seed=args.seed)
    print(f"[gen_persian_docs] wrote {len(written)} backgrounds to {out_dir}")
    print(f"[gen_persian_docs] raqm shaping: {_HAS_RAQM}  reshaper fallback available: {_HAS_RESHAPER}")
    print(f"[gen_persian_docs] font families resolved: {len(_FONT_FAMILIES)} -> {[Path(r).name for r,_ in _FONT_FAMILIES]}")


if __name__ == "__main__":
    main()
