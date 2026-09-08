# Bundled fonts

## Vazirmatn

- Files: `Vazirmatn-Regular.ttf`, `Vazirmatn-Medium.ttf`, `Vazirmatn-Bold.ttf`
- Upstream: https://github.com/rastikerdar/vazirmatn
- License: SIL Open Font License 1.1 (OFL) — https://github.com/rastikerdar/vazirmatn/blob/master/OFL.txt

Vazirmatn is a Persian/Arabic-script typeface. It is bundled here so
`gen_persian_docs.py` produces reproducible synthetic Persian document
backgrounds without depending on a network fetch or on which fonts happen to
be installed on a given machine.

The SIL OFL permits bundling and redistribution provided the license
accompanies the fonts and they are not sold on their own. The generator falls
back to system Arabic-capable fonts (Tahoma / arabtype / Segoe UI) when these
files are absent, so removing them degrades output variety but does not break
the pipeline.
