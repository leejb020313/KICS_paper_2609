"""Render the JEV paper's math with LaTeX (Tectonic) and rasterise to PNG for the Word draft.

Each entry becomes <name>.tex -> <name>.pdf -> <name>.png (600 dpi). The PNG is inserted at the
PDF's natural size, scaled by 0.95 so 10pt LaTeX math matches the 9.5pt body text.
Usage: python render_equations.py <path to tectonic.exe>
"""
import os
import subprocess
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PREAMBLE = r"""\documentclass[10pt,border=0.6pt]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}
"""
EQUATIONS = {
    "eq1": r"$\displaystyle \hat{y}(x) = \mathbb{I}\left[\sigma(a \cdot \ell(x) + b \cdot s(x) + c) \geq 0.5\right]$",
    "indicator": r"$\mathbb{I}[\cdot]$",
}


def main(tectonic):
    for name, body in EQUATIONS.items():
        tex = os.path.join(HERE, name + ".tex")
        with open(tex, "w", encoding="utf-8") as f:
            f.write(PREAMBLE + body + "\n" + r"\end{document}" + "\n")
        subprocess.run([tectonic, "--chatter", "minimal", tex], check=True, cwd=HERE)
        doc = pymupdf.open(os.path.join(HERE, name + ".pdf"))
        doc[0].get_pixmap(dpi=600, alpha=False).save(os.path.join(HERE, name + ".png"))
        w, h = doc[0].rect.width, doc[0].rect.height
        print(f"{name}: {w:.1f} x {h:.1f} pt")


if __name__ == "__main__":
    main(sys.argv[1])
