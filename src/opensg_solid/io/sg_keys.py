"""sg_keys.py -- the ONE name of the per-element frame block of an SG yaml.

An SG yaml carries the material frame of every element as one top-level
block, 9 direction cosines per element [e1 | e2 | e3], each axis written
with components (x2, x3, x1) -- (x, y, z) of a 2-D SG's mesh plane.  That
block is named `orientation:`.  It is PreVABS' own word
(`<material_orientation>`, "writing local orientation") and the key every
OpenSG reader already stores the frames under in memory
(sg_mesh.load_sg_input, io.sg_input.read_opensg_yaml, the npz sidecar).

`elementOrientations:` -- the legacy spelling of OpenSG 2.1 and earlier
(and of the 1.x FEniCS ShellSegmentMesh / SolidSegmentMesh) -- is still
READ.  Every loader calls normalize_orientation_key on the parsed mapping
right after yaml.load; it renames the legacy key to `orientation` with a
one-line note, printed once per file in a process (the solid command line
parses its yaml twice, for the mesh-size line and for the homogenization).
A file that carries BOTH spellings is refused: which block holds the frames
would be a guess.  Every writer writes `orientation:` only, and the rows
themselves are unchanged.  This module is the one place in src/ that names
the legacy spelling (tests/unit/test_orientation_key.py guards it).

The module itself is stdlib-only: opensg_shell's loaders import it too, and
the detectors that only scan a file's top-level keys (sg_mesh.block_keys,
sg_dialect) call has_orientation_block without numpy or JAX of their own.
Importing it still runs opensg_solid/__init__.py, which loads JAX.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE
# ----------------------------------------------------------------------------
#   ORIENTATION_KEY          "orientation" -- read and written
#   LEGACY_ORIENTATION_KEYS  ("elementOrientations",) -- read only
#   ORIENTATION_KEYS         both, the canonical name first
#   d, path                  a parsed top-level yaml mapping; its file name
#   found                    the spellings d carries
#   _NOTED                   absolute paths of the files whose legacy note
#                            this process has printed
# ----------------------------------------------------------------------------
"""
import os

ORIENTATION_KEY = "orientation"
LEGACY_ORIENTATION_KEYS = ("elementOrientations",)
ORIENTATION_KEYS = (ORIENTATION_KEY,) + LEGACY_ORIENTATION_KEYS
_NOTED = set()


class FrameBlockError(ValueError):
    """A file or SG dict that carries the frame block under both names: the
    refusal of normalize_orientation_key, which the `opensg` command line
    ends with as its one-line reason (opensg/cli.py), any engine."""


def has_orientation_block(keys):
    """True when a set of top-level yaml keys holds the frame block, in
    either spelling.

    In:  keys -- any container of key names (a set, a list, a dict)
    Out: bool."""
    return any(k in keys for k in ORIENTATION_KEYS)


def normalize_orientation_key(d, path="<SG yaml>"):
    """Rename a legacy frame-block key to `orientation`, IN PLACE.

    Called on the parsed top-level mapping right after yaml.load (and on a
    copy of an SG dict handed in through the API) before anything reads
    the frames.  The values are untouched: 9 numbers per element [e1|e2|e3]
    with components (x2, x3, x1); a null block stays "no frames".  The
    note is printed once per file in a process (_NOTED); a path that is not
    a file (an API dict's "<SG dict>" label) gets it every time.

    In:  d -- the parsed yaml mapping (anything else is returned
         unchanged); path str -- the file, named in the note and the error
    Out: d itself, its frames (if any) under "orientation".
    Raises FrameBlockError (a ValueError) when d carries both spellings."""
    if not isinstance(d, dict):
        return d
    found = [k for k in ORIENTATION_KEYS if k in d]
    if len(found) > 1:
        raise FrameBlockError(
            "%s carries the per-element frame block twice (%s): keep "
            "`%s:` only; `%s:` is its legacy spelling"
            % (path, " and ".join(found), ORIENTATION_KEY, found[1]))
    if found and found[0] != ORIENTATION_KEY:
        d[ORIENTATION_KEY] = d.pop(found[0])
        f = os.path.abspath(path) if os.path.isfile(str(path)) else None
        if f not in _NOTED:
            print(" NOTE: %s spells the frame block `%s:` (the legacy name, "
                  "OpenSG 2.1 and earlier); it is read as `%s:`, the one "
                  "name OpenSG writes"
                  % (path, found[0], ORIENTATION_KEY))
            if f is not None:
                _NOTED.add(f)
    return d
