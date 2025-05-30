"""A module for the Tikz export of 2-dimensional axes."""

import numpy as np
import re
from . import tikz

TIKZPICTURE_OPTIONS = { # general styling
    "group line/.style": "semithick",
}

AXIS_OPTIONS = { # basic axis options, which don't depend on the diagram values
    "clip": "false",
    "grid": "both",
    "axis line style": "draw=none",
    "tick style": "draw=none",
    "xticklabel pos": "upper",
    "y dir": "reverse",
    "xmin": "0.5",
    "ymin": "0.66",
    "legend style": "draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em,/tikz/every odd column/.append style={column sep=.5em}",
    "legend cell align": "left",
    "title style": "yshift=\\baselineskip",
    "width": "\\axisdefaultwidth",
}

def to_str(average_ranks_and_names, groups, treatment_names_all, diagram_names, *, reverse_x=False, as_document=False, tikzpicture_options=dict(), axis_options=dict(), preamble=None):
    """Return a string with Tikz code."""
    # average_ranks, names = zip(*average_ranks_and_names)
    m = len(diagram_names) # numbers of diagrams
    k = len(treatment_names_all) # max numbers of treatments
    axis_defaults = tikz._merge_dicts(AXIS_OPTIONS, { # diagram-dependent axis options
        "ytick": ",".join((np.arange(m)+1).astype(str)),
        "yticklabels": ",".join([ "{" + tikz._label(n) + "}" for n in diagram_names ]),
        "xmax": str(k + .5),
        "ymax": str(m + .66),
        "height": f"{.5 if m == 2 else m/5 if m < 5 else m/6}*\\axisdefaultheight",
    })
    if reverse_x:
        axis_defaults["x dir"] = "reverse"
    commands = []
    for treatment in treatment_names_all:
        treatment_ranks = []
        for ranks_cur, names_cur in average_ranks_and_names:
            if treatment in names_cur:
                i = names_cur.index(treatment)
                treatment_ranks.append(ranks_cur[i])
            else:
                treatment_ranks.append(None)
        commands.append(_rank_plot(treatment_ranks, treatment))
    # commands = [ _rank_plot(average_ranks[:,i], treatment_names[i]) for i in range(k) ]
    for i in range(m):
        average_ranks, names = average_ranks_and_names[i]
        for (j, g) in enumerate(groups[i]):
            group_ranks = [average_ranks[names.index(t)] for t in g]
            commands.append(_group(
                np.min(group_ranks),
                np.max(group_ranks),
                i + (j+.66) / (1.33 * len(groups[i]) + 1) + 1
            ))
    tikz_str = tikz._tikzpicture(
        tikz._axis(*commands, options=tikz._merge_dicts(axis_defaults, axis_options)),
        options = tikz._merge_dicts(TIKZPICTURE_OPTIONS, tikzpicture_options)
    )
    if as_document:
        tikz_str = tikz._document(tikz_str, preamble=preamble)
    return tikz_str

def _rank_plot(average_ranks, treatment_name):
    points = []
    for i, r in enumerate(average_ranks):
        if r is not None:  # Check if the setting should be skipped
            points.append(f"({r}, {i+1})")

    return "\n".join([
        "\\addplot+[only marks] coordinates {",
        "  " + "\n  ".join(points),
        "};",
        "\\addlegendentry{" + tikz._label(treatment_name) + "}",
    ])
def _group(minrank, maxrank, ypos):
    return f"\\draw[group line] (axis cs:{minrank},{ypos}) -- ++(0pt,-3pt) -- ([yshift=-3pt]axis cs:{maxrank},{ypos}) -- ++(0pt,3pt);"
