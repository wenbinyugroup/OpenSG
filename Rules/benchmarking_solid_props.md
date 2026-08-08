# Rule — how to benchmark equivalent 3-D solid properties

**Scope:** any shell-vs-solid or code-vs-paper comparison of an equivalent 3-D law.

## Reference hierarchy

1. **Closed form** where one exists — the only reference that cannot itself be wrong.
   Periodic cross cell `C44 = 0.5·E'·(t/L)^3`; frame lattice `D44 = 2·kv·kh/(kv+kh)`,
   `k = 6·E'·I/span`; single wall `A_mat·C`.
2. **SwiftComp `.K`** — msg-solid reproduces it **digit-for-digit** (7 significant figures on all
   nine entries, 192k and 546k tets). A published "MSG solid" column *is* SwiftComp; label it as
   SwiftComp, not as the paper author's own model.
3. **Another OpenSG route** — useful, but a disagreement between two of our own solvers is not
   evidence about which is right.

## Match the geometry before comparing numbers

Most apparent "bugs" in this area were geometry or normalization mismatches:

- **Thickness.** Recover it from the reference mesh itself — `t = 2V/S_free` (material volume
  over free-surface area) — never guess. A guessed `t = 0.02` against the true `0.036547`
  produced a clean factor ~0.5 across every component that looked exactly like a solver bug.
- **Census.** Cross-check relative density (`area × t` vs the solid's material volume) before
  reading any table; agreement to <1 % means the comparison is meaningful.
- **Normalization.** See `output_format_and_timing.md` — a uniform whole-matrix ratio is a
  convention difference.
- **Cell construction.** Ring cells merge coincident walls under the periodic tie (factor 4 on
  `C44` only); use a cross cell when benchmarking shear.

## Read the error structure, not just its size

A **uniform** offset within a channel class is a model statement; a **scattered** one is a bug.
Post-fix Schwarz-P vs a SwiftComp-exact solid: −3 % on all normals, −6 % on all shears, +0.3 %
on Poisson, cubic symmetry intact to 5 digits in both — that pattern is the thin-shell reduction
at $t/R \approx 0.2$, and is reported as physics, not chased.

Poisson-type ratios are normalization-invariant: if $\nu$ matches but moduli do not, the problem
is $\omega$, not the mechanics.

## Reporting

- Tabulate **every non-zero $C_{ij}$**, never the diagonal alone.
- Quote the reference's own columns verbatim and state which is the reference.
- Report symmetry residuals (cubic/orthotropic terms that should vanish) as a health check on
  the periodic tie — 4–5 orders below the diagonal is the expected level.
- State open items explicitly (e.g. a sign convention still unreconciled) rather than
  suppressing the term.
