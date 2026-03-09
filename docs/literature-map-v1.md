# Literature Map v1 — Deepritz-Enhanced meshfree KAN-RKPM project

## Purpose

This note records the first-round literature map for the current research direction around meshfree KAN/RKPM shape-function learning and variational PDE solving.

The goal of this document is not to claim novelty yet, but to:
- define the relevant literature clusters,
- identify the most important neighboring areas,
- guide the next round of targeted reading,
- help refine the eventual research question and novelty statement.

## Current project understanding

The current project is not just "using a neural network to solve a PDE".
It is better understood as an attempt to learn structured meshfree shape functions, and therefore a structured trial space, while preserving key consistency properties associated with RKPM/MLS-style constructions.

This places the work at the intersection of:
- classical meshfree shape-function construction,
- variational neural PDE solving,
- interpretable/structured neural discretization,
- KAN-based parameterization,
- meshfree + neural hybrid methods.

## Literature clusters

### Cluster 1 — Classical meshfree / RKPM / MLS foundations

Questions this cluster should answer:
- What are the defining properties of RKPM/MLS shape functions?
- How are PU and linear reproduction established in the classical setting?
- What role do kernel correction, polynomial bases, and moment matrices play?
- Where exactly does the implementation burden come from in classical meshfree methods?

Why it matters here:
- This cluster defines the mathematical baseline for what the current project is trying to emulate, relax, or learn.
- It is essential for writing the introduction and for making precise claims about consistency and structure.

Representative starting points:
- Liu, Jun, Zhang (1995), Reproducing Kernel Particle Methods.
- Chen et al. (2001), stabilized conforming nodal integration / SCNI context.
- MLS foundational papers (to be refined in v2).

### Cluster 2 — Deep Ritz / VPINN / variational neural PDE methods

Questions this cluster should answer:
- How do variational neural methods describe trial spaces and test spaces?
- What is already known about variational neural approximations, quadrature, and test-function design?
- How close is the present project to VPINN-style thinking, and where does it differ?

Why it matters here:
- This cluster provides the solver-side language.
- It is especially important for legitimizing the “trial space” viewpoint.
- It helps distinguish the current project from standard PINN-style residual minimization.

Representative starting points:
- E & Yu (2018), The Deep Ritz Method.
- Kharazmi, Zhang, Karniadakis (2019), VPINN.
- hp-VPINNs (2020).
- Berrone / Canuto / Pintore works on VPINN analysis and MF-VPINN.

### Cluster 3 — Interpretable neural discretization / learned basis / neural interpolation

Questions this cluster should answer:
- Are there already works that learn discretization/interpolation functions in an interpretable way?
- How do such works define basis functions and approximation spaces?
- Are they mesh-based, reference-element-based, or more general?
- Where exactly is the boundary between those works and the current meshfree project?

Why it matters here:
- This is likely the most dangerous neighboring cluster for novelty assessment.
- These works may not be meshfree, but they may be conceptually close in terms of “learned structured approximation spaces”.

Representative starting points:
- Finite Element Neural Network Interpolation (FENNI), 2024.

### Cluster 4 — KAN in scientific computing / PDE contexts

Questions this cluster should answer:
- How is KAN currently used in PDE/scientific ML work?
- Is KAN being used merely as a function approximator, or as a structured basis-like object?
- What, if anything, is genuinely specific about KAN in the current project?

Why it matters here:
- This cluster matters for parameterization choices.
- However, KAN alone probably should not carry the entire novelty claim.
- The likely stronger claim is about structured meshfree shape-function learning, not only about KAN.

Representative starting points:
- To be refined in v2.

### Cluster 5 — Meshfree + neural hybrid directions

Questions this cluster should answer:
- Has anyone directly learned meshfree shape functions with neural parameterizations?
- Has anyone explicitly enforced PU / reproduction in a learned meshfree setting?
- Has anyone combined learned meshfree shape functions with a variational PDE solver?
- Has anyone used teacher-style distillation to classical meshfree basis families?

Why it matters here:
- This cluster is the likely novelty discovery zone.
- If it is sparse, that strengthens the originality of the current project.
- If close prior art exists, this cluster will define the most important differentiation work.

Representative starting points:
- To be refined in v2 through more targeted searches.

## Preliminary priority ranking

### Highest priority
1. Classical meshfree / RKPM / MLS foundations.
2. Interpretable neural discretization / learned basis.
3. Deep Ritz / VPINN / MF-VPINN.

### Second priority
4. Meshfree + neural hybrid directions.
5. KAN-specific PDE/scientific computing literature.

## Initial novelty hypothesis

Current hypothesis (not final):

The likely novelty is not that every individual ingredient is new, but that the project may combine:
- meshfree RKPM/MLS-style consistency goals,
- learned shape-function parameterization,
- a KAN-based local representation,
- a two-stage mechanism-driven training path,
- and a variational PDE solving framework,

in a way that is not obviously already covered by nearby literatures.

This suggests a likely “structured combination novelty” rather than a pure single-axis novelty.

## Immediate next-step reading priorities

### Priority set A
- RKPM foundational literature.
- MLS foundational literature.
- Deep Ritz.
- VPINN.
- MF-VPINN.
- FENNI.

### Priority set B
- KAN-for-PDE/scientific computing papers.
- Targeted searches for learned meshfree shape functions / neural meshfree basis work.

## Open questions for v2

1. Should the project be framed primarily as:
   - learning shape functions,
   - learning a meshfree trial space,
   - or both simultaneously?
2. Is teacher best framed as:
   - representational validation,
   - stabilization mechanism,
   - or both with a clear hierarchy?
3. Which nearby work is most dangerous for novelty collision:
   - FENNI-like learned discretization,
   - meshfree variational neural methods,
   - or an as-yet-unidentified learned meshfree basis paper?
4. How strong a claim is justified regarding preservation of RKPM/MLS structure?

## Status

This is Literature Map v1.
Next step: produce v2 with more concrete representative papers, tighter cluster boundaries, and a first-pass difference matrix against the current project.
