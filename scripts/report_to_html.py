"""Turn a report Markdown file into HTML next to it, styled for A4 printing.

    uv run --with markdown python scripts/report_to_html.py docs/report_ja.md
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
        --no-pdf-header-footer --print-to-pdf="$PWD/docs/report_ja.pdf" "file://$PWD/docs/report_ja.html"

Japanese reports (*_ja.md) use Hiragino Kaku Gothic and Chinese ones Hiragino Sans GB, so each
language gets its own glyphs for the shared characters.
"""

import sys

import markdown

CSS = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: FONT, sans-serif; font-size: 10.5pt; line-height: 1.75; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4pt; line-height: 1.4; }
h1 + p { color: #555; margin-top: 0; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; margin-top: 20pt; }
h3 { font-size: 11.5pt; margin-top: 14pt; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 9.2pt; margin: 8pt 0; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 3pt 6pt; vertical-align: top; }
th { background: #f1f1f1; }
code { font-family: Menlo, monospace; font-size: 8.8pt; background: #f3f3f3; padding: 0 2pt;
       border-radius: 2px; }
img { display: block; max-width: 100%; margin: 8pt auto 2pt; page-break-inside: avoid; }
p:has(> img) { page-break-inside: avoid; page-break-after: avoid; margin-bottom: 0; }
em { font-style: normal; color: #444; font-size: 9.3pt; display: block; text-align: center; }
li { margin: 2pt 0; }
"""


def main():
    src = sys.argv[1]
    dst = src.rsplit('.', 1)[0] + '.html'
    ja = src.endswith('_ja.md')
    if ja:
        lang, title, font = 'ja', 'SO-101 模倣学習レポート', "'Hiragino Kaku Gothic ProN', 'Hiragino Sans'"
    else:
        lang, title, font = 'zh-CN', 'SO-101 模仿学习报告', "'Hiragino Sans GB', 'PingFang SC'"
    with open(src, encoding='utf-8') as f:
        body = markdown.markdown(f.read(), extensions=['tables'])
    css = CSS.replace('FONT', font)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
                f'<title>{title}</title><style>{css}</style></head><body>{body}</body></html>')
    print(dst)


if __name__ == '__main__':
    main()
