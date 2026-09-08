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


@dataclass
class PageStyle:
    w: int
    h: int
    size_name: str
    bg: Tuple[int, int, int]
    rule: Tuple[int, int, int]
    accent: Tuple[int, int, int]
    font_family: int
    base_font_size: int
    margin: int
    scanned: bool

    @staticmethod
    def random(rng: random.Random) -> "PageStyle":
        w, h, name = rng.choice(PAGE_SIZES)
        # +/- 6% jitter on page size for variety without breaking archetype proportions
        w = int(w * rng.uniform(0.94, 1.06))
        h = int(h * rng.uniform(0.94, 1.06))
        return PageStyle(
            w=w, h=h, size_name=name,
            bg=rng.choice(BG_TINTS),
            rule=rng.choice(RULE_COLORS),
            accent=rng.choice(ACCENT_COLORS),
            font_family=rng.randrange(n_font_families()),
            base_font_size=rng.randint(20, 30),
            margin=rng.randint(50, 110),
            scanned=rng.random() < 0.35,
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
    d.rectangle([m, m, w - m, m + bar_h], fill=style.accent)
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

    d.rectangle([m, footer_top, w - m, h - m], fill=style.accent)
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

    d.rectangle([m, table_top, w - m, table_top + row_h], fill=style.accent)
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
    d.rectangle([0, 0, w, bar_h], fill=style.accent)
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


ARCHETYPES = {
    "tender_notice": gen_tender_notice,
    "tabular_document": gen_tabular_document,
    "form_layout": gen_form_layout,
    "plain_letter": gen_plain_letter,
    "spreadsheet": gen_spreadsheet,
    "dense_text": gen_dense_text,
}

# Relative frequency in a generated corpus. Text-heavy archetypes are
# oversampled on purpose: the watermark-over-dense-text case is both the
# hardest for the model and the one the current detector fails, so it needs
# the most coverage. The near-empty archetypes (form/spreadsheet grids) still
# appear because they are common in the real workload and provide the
# structural-line variety, just at lower weight.
ARCHETYPE_WEIGHTS = {
    "dense_text": 4,
    "tender_notice": 3,
    "plain_letter": 3,
    "tabular_document": 2,
    "form_layout": 1,
    "spreadsheet": 1,
}


# ---------------------------------------------------------------------------
# "Scanned" degradation
# ---------------------------------------------------------------------------

def apply_scan_effects(im: Image.Image, rng: random.Random) -> Image.Image:
    """Slight rotation, mild noise, soft blur, off-white cast -- approximates
    a phone photo / flatbed scan of a printed page rather than a clean
    digital render."""
    angle = rng.uniform(-2.2, 2.2)
    bg = tuple(int(c * 0.97) for c in (250, 248, 244))
    im = im.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg)

    arr = np.asarray(im).astype(np.float32)
    noise_sigma = rng.uniform(2.5, 7.0)
    noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255)

    # gentle off-white paper cast + slight contrast pull-in
    tint = np.array([250, 247, 240], dtype=np.float32)
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
    style = PageStyle.random(rng)
    im = ARCHETYPES[name](rng, style)
    if style.scanned:
        im = apply_scan_effects(im, rng)
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
