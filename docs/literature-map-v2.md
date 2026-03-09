# Literature Map v2 — Deepritz-Enhanced meshfree KAN-RKPM project

## Scope of this note

This note refines v1 by adding:
- a clearer positioning of the project,
- a first-pass neighboring-work analysis,
- a comparison matrix against major nearby directions,
- and a sharper novelty-risk assessment.

This is still not a final literature review. It is a working research note for framing the project before a more complete paper-oriented review.

---

## 1. Refined project positioning

A more precise current interpretation of the project is:

> The project studies whether a neural parameterization (currently KAN-based) can be used to learn structured meshfree shape functions, and therefore a structured meshfree trial space, while preserving or approximately recovering key RKPM/MLS-style consistency and stability properties, and then use that learned space in a variational PDE solver.

This framing has three levels:

1. **Direct object**: learning meshfree shape functions.
2. **Numerical-analysis interpretation**: learning a structured trial space.
3. **Solver context**: using that learned space inside a variational/Deep-Ritz-type PDE solve.

This framing is currently preferred over weaker descriptions such as “just using KAN for PDEs” or stronger but riskier descriptions such as “fully neural RKPM”.

---

## 2. Updated understanding of the main literature clusters

### Cluster 1 — Classical meshfree / RKPM / MLS foundations

This cluster remains indispensable because it defines:
- partition of unity,
- linear/polynomial reproduction,
- correction and moment-matrix machinery,
- conditioning/stability concerns,
- and the implementation burden that originally motivated the present project.

### Current interpretation for this project
The present project is best read as engaging with the **shape-function construction problem** from this cluster, rather than only borrowing a few meshfree ideas superficially.

### Why this matters
The eventual paper introduction can likely be built around the claim that:
- classical meshfree methods offer structured approximation spaces,
- but explicit construction is nontrivial,
- and the current work investigates whether a learned parameterization can recover useful structure without abandoning the variational setting.

---

### Cluster 2 — Deep Ritz / VPINN / variational neural PDE

This cluster is now more clearly understood as the **solver-side language** of the project.

Important point from this round:
- the VPINN literature explicitly uses the language of **trial space** and **test space**.
- therefore, the “learned trial space” interpretation of the present project is well-supported by an existing variational neural PDE literature.

### Representative works already identified
- Deep Ritz Method (2018)
- VPINN (2019)
- hp-VPINN (2020)
- Robust VPINN (2023)
- MF-VPINN (2024)

### What this cluster contributes
This cluster helps say:
- the project is not a standard PINN residual-minimization story,
- it lives in a variational PDE context,
- but its innovation axis is likely **not** test-space design, quadrature design, or variational residual robustness.

### Key difference from the current project
VPINN/MF-VPINN mostly emphasize:
- variational formulation,
- test-space design,
- quadrature,
- robustness,
- adaptive training.

The present project instead emphasizes:
- structured shape-function construction,
- trial-space structure,
- consistency diagnostics,
- and stability of the learned basis family.

So VPINN appears to be an important **contextual neighbor**, but not currently the most dangerous novelty collision.

---

### Cluster 3 — Interpretable neural discretization / learned basis / neural interpolation

This cluster has become even more important after the second round of searching.

The strongest identified neighboring work so far is:
- **Finite Element Neural Network Interpolation. Part I: Interpretable and Adaptive Discretization for Solving PDEs (FENNI, 2024)**

### Why FENNI is important
From the available abstract-level information, FENNI involves:
- interpretable discretization,
- neural interpolation,
- sparse architecture,
- reference-element-based shape functions,
- multigrid/adaptive training,
- variational loss.

This is conceptually close to the present project in the broad sense of **learning an interpretable discretization/basis-like structure for PDE solving**.

### Why it is still not the same problem
At least from the current evidence, FENNI is:
- **mesh-based**,
- rooted in finite-element structure,
- reference-element centered,
- and not focused on RKPM/MLS-style meshfree consistency.

### Current conclusion
FENNI is likely the strongest currently identified **dangerous neighbor**, but it may also be extremely useful for sharpening the boundary:

- FENNI: learned/interpretable **mesh-based** discretization.
- Current project: learned/interpretable **meshfree** shape-function / trial-space construction with RKPM/MLS-inspired consistency goals.

This boundary should be made explicit in future writing.

---

### Cluster 4 — KAN in scientific computing / PDE contexts

This cluster still matters, but the project should not lean on it too heavily.

### Current interpretation
KAN is better treated as:
- a current parameterization choice,
- possibly a good inductive bias for representing local shape responses,
- but probably **not** the full source of the project’s novelty.

### Why this matters
If the paper over-claims novelty via KAN alone, reviewers may respond with:
- “Would MLP/RBF/spline networks also work?”
- “Is the main contribution really KAN, or the structured shape-function learning framework?”

Current answer should likely be:
- KAN is part of the method design,
- but the stronger contribution is the **structured meshfree learned trial-space viewpoint**.

---

### Cluster 5 — Meshfree + neural hybrid directions

This remains the likely **novelty discovery zone**, but also the hardest cluster to search cleanly.

The current search evidence is noisy, which suggests one or both of the following:
- truly close prior art may be sparse,
- or current queries are still too broad.

### What this cluster must answer next
- Has anyone directly learned meshfree shape functions with a neural parameterization?
- Has anyone explicitly enforced or evaluated PU/reproduction in a learned meshfree setting?
- Has anyone built a learned meshfree basis/trial family and used it inside a variational PDE solver?
- Has anyone used teacher-style alignment to classical RKPM/MLS basis constructions?

This cluster remains a top target for the next search round.

---

## 3. First-pass comparison matrix

| Direction / Work | Meshfree? | Learned shape/basis functions? | Explicit consistency goals (PU / reproduction)? | Variational PDE context? | Interpretable discretization emphasis? | Closest relevance to current project |
|---|---:|---:|---:|---:|---:|---|
| Classical RKPM / MLS | Yes | No | Yes | Can be, but classical numerical setting | Yes (classical numerical sense) | Foundational baseline for shape-function structure |
| Deep Ritz | No (not inherently) | No explicit structured basis family | No | Yes | No | Solver framework context |
| VPINN | Not inherently | Trial space is NN, but not structured learned meshfree basis | No explicit RKPM-style consistency | Yes | Limited | Important variational language, not same innovation axis |
| MF-VPINN | Meshfree test-space generation, not meshfree learned shape family | Not in the current sense | No explicit RKPM-style consistency | Yes | Limited | Related meshfree variational neighbor |
| FENNI | No (mesh-based) | Yes | FEM-structured, not RKPM/MLS-style meshfree consistency | Yes | Yes | Strong conceptual neighbor / novelty-risk paper |
| Current project | Yes | Yes | Yes / approximately targeted | Yes | Yes | Meshfree learned shape-function / learned trial-space project |

### Main takeaway from the matrix
The current project does **not** look like a simple variant of Deep Ritz or VPINN.
Its more distinctive axis is:
- meshfree,
- learned shape-function family,
- explicit consistency/stability diagnostics,
- variational PDE use.

This is encouraging, but it still depends on whether close meshfree+neural basis papers exist and have not yet been found.

---

## 4. Emerging novelty hypothesis

The current best hypothesis is:

> The project’s likely novelty is not that every individual ingredient is unprecedented, but that it combines classical meshfree consistency goals, learned shape-function construction, a KAN-based local parameterization, mechanism-guided stabilization (teacher/reg), and variational PDE solving into one structured framework.

This suggests a **combination novelty with a strong structural core**, rather than a novelty claim based on only one ingredient.

### More specifically, the strongest current candidate contribution is:
- learning a **structured meshfree trial space / shape-function family**,
- rather than merely using a neural network as a black-box PDE ansatz.

---

## 5. Main novelty risks identified so far

### Risk 1 — Learned discretization papers such as FENNI
This is currently the clearest conceptual neighbor.
If not read carefully, it could make the present work appear less new than it is.

### Risk 2 — Hidden meshfree+neural basis literature not yet found
This remains unresolved. A more targeted search is required.

### Risk 3 — Over-claiming the RKPM connection
If the paper claims too strongly that the method “is RKPM” rather than “learns a RKPM/MLS-inspired structured family with consistency goals”, that may trigger avoidable reviewer resistance.

### Risk 4 — Over-claiming KAN as the core novelty
This would likely weaken the paper. The current project seems stronger when framed around structured trial-space learning rather than “KAN for PDE” alone.

---

## 6. Updated framing recommendation

The current best framing recommendation is:

> This work investigates a KAN-parameterized framework for learning structured meshfree shape functions — and thus a structured meshfree trial space — with RKPM/MLS-inspired consistency goals, and evaluates that learned space in a variational PDE solving setting.

This framing is preferred because it:
- preserves the original “learning shape functions” motivation,
- connects naturally to Galerkin/variational numerical language,
- avoids overstating equivalence to classical RKPM,
- and places the project in a clearer relation to nearby neural discretization works.

---

## 7. Immediate next reading tasks (for v3 or a focused review pack)

### Priority A — must refine
1. MLS foundational papers (precise canonical set)
2. RKPM foundational/canonical review papers
3. FENNI full reading and extraction of the exact difference points
4. One representative KAN-for-PDE paper to clarify what KAN contributes here

### Priority B — novelty search
5. Targeted search for “learned meshfree basis / learned meshfree shape function / neural meshfree interpolation”
6. Search for partition-of-unity-enforced learned approximation spaces
7. Search for teacher/distillation to classical numerical bases

---

## 8. Working conclusions at the current stage

1. The **trial-space** interpretation is now well-supported and should remain in play.
2. VPINN/MF-VPINN are important contextual neighbors, but currently do not appear to define the main novelty conflict.
3. FENNI and related learned discretization work are the most important currently identified near-neighbor family.
4. The strongest current framing is about **structured meshfree shape-function / trial-space learning**, not merely KAN or generic neural PDE solving.
5. A more targeted search is still needed before making any strong novelty claim.

---

## Status

This is Literature Map v2.
A possible next step is a dedicated note focused on one of the following:
- `docs/novelty-risk-matrix.md`
- `docs/related-work-difference-table.md`
- `docs/fenni-mf-vpinn-rkpm-comparison.md`

