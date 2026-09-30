# Specifications

Two documents describe the framework: what it must do, and how each part is built to do it.

| File | What it is | Standard it follows |
|---|---|---|
| [system-requirements-specification.md](system-requirements-specification.md) | **System Requirements Specification (SyRS).** Every requirement the framework as a whole must meet, each with an ID, a priority, its current status and how it is checked. | ISO/IEC/IEEE 29148:2018, *Requirements engineering* (clause 9.6, SyRS content) |
| [subsystem-design-description.md](subsystem-design-description.md) | **Subsystem Design Description (SDD).** The framework split into subsystems (shared core, Stage 0 to 4, search), and for each one its parts, interfaces, data, behaviour, algorithm and design rationale. | IEEE 1016-2009, *Software Design Descriptions*, with viewpoints in the sense of ISO/IEC/IEEE 42010:2022, *Architecture description* |

## Why these standards

- **ISO/IEC/IEEE 29148:2018** replaced IEEE 830 (the old "SRS" standard) and is the current IEEE standard for
  writing requirements. It asks that every requirement be necessary, unambiguous, verifiable and traceable, and
  that each one say how it will be verified (inspection, analysis, demonstration or test). That fits a thesis
  well: an examiner can follow a claim from the requirement to the test or the report that shows it.
- **IEEE 1016-2009** is the IEEE standard for design documents. Rather than a fixed table of contents, it asks the
  author to pick *viewpoints* (context, composition, interfaces, information, interaction, algorithms,
  resources...) that answer the readers' concerns, and to give each subsystem a *view* from each chosen
  viewpoint. This lets one document hold both implemented and planned subsystems in the same shape.
- **ISO/IEC/IEEE 42010:2022** defines what "viewpoint", "view", "stakeholder" and "concern" mean; IEEE 1016 builds
  on it, so the SDD uses its vocabulary.

Neither standard is followed to the letter where it asks for material a single-author research codebase does
not have (for example contractual, safety or regulatory sections). Those clauses are listed as "not applicable"
rather than dropped, so the mapping to the standard stays visible.

## In plain words

A **requirements specification** is a numbered list of promises: "the framework shall measure memory", "the
search shall never tune on the held-out data". Each promise says how we will prove it was kept. A **design
description** explains how the program is built to keep those promises: which parts exist, what each part
receives and hands on, and why it was built that way. The first document answers *what*; the second answers
*how*.

## Keeping them current

- Requirements are the single source for what a stage must do; the project instructions and the thesis spec
  are their origin. When the spec changes, update the SyRS first, then the SDD, then the code.
- Each requirement has a status (Implemented, Partial, Planned). Update it in the same pull request that changes
  the code.
- Planned subsystems follow the class and sequence diagrams in [`../diagrams/`](../diagrams/README.md); names of
  planned classes are proposals and may change when the stage is written.
