"""Are the wild-type and AVA-silenced recordings comparable conditions?

The intervention contrast uses WT_NoStim as the wild-type arm and AVA_HisCl as
the silenced arm. The Kato repository also contains WT_Stim. If the AVA_HisCl
recordings were made under an oxygen stimulus while WT_NoStim was not, then the
contrast confounds AVA silencing with stimulation, and WT_Stim would be the
correct comparison group.

The behavioural state annotation is the discriminator: the no-stimulus
recordings are labelled with a finer state set than the stimulus recordings.
This reads both files and reports the state alphabet per animal per arm.

    python scripts/check_kato_arms.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

ROOT = Path("Data/celegans")
ARMS = [("WT_NoStim", "WT_NoStim.mat"), ("AVA_HisCl", "AVA_HisCl.mat")]
BAR = "=" * 72


def read_states(h: h5py.File, grp: h5py.Group, i: int):
    """States for animal i, dereferencing the object array."""
    ref = grp["States"][i][0]
    return np.asarray(h[ref][()]).ravel()


def read_key(h: h5py.File, grp: h5py.Group, i: int) -> list[str]:
    if "States_key" not in grp:
        return []
    ref = grp["States_key"][i][0]
    node = h[ref]
    out = []
    for j in range(node.shape[0]):
        try:
            s = h[node[j][0]][()]
            out.append("".join(chr(int(c)) for c in np.asarray(s).ravel() if c))
        except Exception:
            pass
    return out


def main() -> int:
    print(BAR)
    print("BEHAVIOURAL STATE ALPHABET PER ARM")
    print(BAR)
    summary = {}
    for name, fn in ARMS:
        path = ROOT / fn
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        with h5py.File(path, "r") as h:
            g = h[name]
            n = g["States"].shape[0]
            counts = []
            print(f"\n{name}  ({n} animals)")
            for i in range(n):
                st = read_states(h, g, i)
                u = np.unique(st[~np.isnan(st)]) if st.dtype.kind == "f" else np.unique(st)
                counts.append(len(u))
                key = read_key(h, g, i)
                ks = (", ".join(key)) if key else "(no States_key in this arm)"
                print(f"  animal {i}: T={len(st):5d}  states={len(u)}  "
                      f"labels={u.astype(int).tolist()}")
                if i == 0 and key:
                    print(f"            key: {ks}")
            summary[name] = counts
    print("\n" + BAR)
    print("VERDICT")
    print(BAR)
    if len(summary) == 2:
        a, b = [summary[n] for n, _ in ARMS]
        print(f"WT_NoStim distinct states per animal : {a}")
        print(f"AVA_HisCl distinct states per animal : {b}")
        if max(a) != max(b):
            print("\nThe two arms carry DIFFERENT state alphabets "
                  f"({max(a)} vs {max(b)}).")
            print("In the Kato repository the coarser alphabet accompanies the")
            print("stimulus recordings, so this is consistent with AVA_HisCl")
            print("having been recorded under stimulus while WT_NoStim was not.")
            print("If so, the contrast confounds silencing with stimulation and")
            print("WT_Stim is the matched control. This needs resolving against")
            print("the dataset documentation before the contrast is reported.")
        else:
            print("\nSame state alphabet in both arms; no evidence here of a")
            print("stimulus mismatch between the two conditions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
