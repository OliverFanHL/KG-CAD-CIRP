# CAD relevance annotation rubric v5

## Review protocol

This rubric was fixed after a development pilot on 2026-10-21 for AI engineering-shape review across eleven tasks. The pilot used 44 validation CADs with no exact STEP hash overlap with the test split. Pilot case IDs below identify examples used to define task boundaries.

The pilot refined annotation criteria and source views while retaining the retrieval model, weights, exemplars, query bounds, and priors. First-pass reviews excluded test grades and method rankings. Because historical test performance was already known and the criteria were refined, comparisons across annotation releases combine grading revisions with interpretive changes.

## Decisions and evidence

Assess every CAD against all eleven tasks using the following grades:

| Grade | Criterion |
|---|---|
| 2 | The named geometric features are visible |
| 1 | Positive task-form evidence is visible, with a specific unresolved requirement or explicitly simplified representation |
| 0 | The form is visibly incompatible, or a required feature is visibly absent under the task's exclusion criteria |
| null (`?`, cannot assess) | Missing surfaces, unresolved views or occlusion, or uninterpretable geometry prevent assessment of a decisive feature |

Keep unresolved assessments as null. A fully visible smooth bore or shank is a simplified representation: a plausible nut or headed fastener can receive a conditional grade when its thread is unmodeled or unresolved.

Record feature evidence with view locators:

- `feature+@view(detail)`: visible.
- `feature-@view(detail)`: visibly absent.
- `feature?@view(detail)`: unresolved within a readable, plausible form.
- `feature!@view(detail)`: unreadable evidence.

Base each record on individual inspection of the identified CAD. Exclude folder labels, predictions, and retrieval scores from the assessment. Include at least one observation from a full-surface view, supplemented by cutaways as needed.

The six panels are oblique, opposite, profile, axial, cut_X, and cut_Y. Full-size images are 1500×1000 and omit catalog categories and rankings. Use contact sheets for navigation and individual images for detailed assessment. Cutaways are uncapped visualization clips; their boundaries do not represent manufactured faces or dimensional measurements. Surface tessellation edges are hidden. Distinguish helical threads from faceting and cut-edge serrations. If enlargement leaves a decisive feature unresolved, assign the appropriate conditional or cannot-assess state.

## Task boundaries

### S — `support_body`

Stationary carrier/housing body + mounting features + a distinguishable circular support-seat/carrier feature. Concentric stepped seat, captured bearing/race, supporting circular housing with fixing ears/feet support 2. A plain bored mounting block is at most 1: a bore alone does not establish its rotary support seat. A bare rolling bearing has no mounting carrier and is 0. A pipe flange is not promoted to support merely because it has a hole. Pilot D002/D003/D020 versus D042.

### C — `shaft_connector`

A complete two-ended coaxial or articulated connection with positive clamping, keying, radial retention, joint or equivalent drive attachment at both regions supports 2. A recognizable coupling half/one positively retained shaft region with the other unresolved is 1. Plain hollow reducers/sleeves are 0. A keyed gear/pulley hub by itself is not a complete connector and should not gain a connector role solely from its keyway. Short all-joint/coupling bodies are not drive shafts. Pilot D004/D009/D025/D031 versus D042.

### B — `belt_wheel`

Continuous circumferential V/rounded belt groove, or fine belt-type axial teeth confined by retaining flanges, supports 2. Generic cylindrical toothed wheels with genuinely unresolved belt-vs-gear interface can be 1 even if cylindrical gear morphology is clear; record that exact profile uncertainty. Positively identifiable chain roller pockets/narrow sprocket teeth, bevel/worm forms, plain bolt flanges and smooth washers are 0. A radial shaft-retention feature in a pulley hub does not remove the belt role. Pilot D012/D019/D021/D039 versus D014/D016.

### T — `threaded_clamp`

Head with bearing surface + external shank + visibly modeled helical thread supports 2. Smooth headed fastener-like shanks with thread explicitly unmodeled/unresolved support 1. Socket/wrench/slotted turning features strengthen head interpretation. Headless set screws/studs, lifting eyes/hooks, forked link rods, nuts and unrelated shapes are 0. Internal bore does not exclude an otherwise qualifying hollow head. Thread proof must appear as actual helical ridges/grooves in profile or cutaway, not triangular mesh shading. Pilot D015/D023/D034 versus smooth headed D035.

### N — `nut_retention`

Standalone wrenchable/turnable bored nut-like body + visible internal helical thread supports 2. Clearly nut-like hex/square/knurled body with observed smooth or unresolved bore thread is 1. Plain sleeves/washers, externally threaded bolts and large pipe flanges are 0. Male/female standoffs may meet N only if a genuinely standalone nut-like internally threaded body is visible; do not automatically label both nut and bolt because a tube has threads. Pilot D006/D024/D028/D029 show explicit smooth-bore conditionals.

### M — `bracket_mount`

Connected fixed rigid body + two angled or offset attachment regions + holes/slots on both supports 2. A plausible fixed mounting body with second region/fixing interpretation unresolved is 1. A single flat plate is not upgraded solely by multiple coplanar holes; shallow channels can remain 1 if the second role is not established. Articulating hinges are 0 for the fixed-bracket task. A circular opening alone is not a flange/support label. Pilot D022/D044 versus D033.

### F — `flanged_passage`

Open central passage + surrounding bolt pattern + raised/recessed coaxial register/neck mating geometry supports 2. A flat open bolt plate without register is 1. An open neck may supply the preliminary coaxial mating geometry; no fluid service is inferred. A blind center is 0. Positive shaft keyway/spline/shaft-clamping interface excludes passage interpretation; if shaft versus passage role genuinely cannot be distinguished, keep a plausible role conditional and name the uncertainty, rather than asserting fluid suitability. A generic bracket hole is not a flange. Pilot D001/D038 versus D013/D043.

### G — `gear_mesh`

Repeated external spur/helical teeth on a cylindrical gear body supports 2. Unresolved cylindrical tooth form can be 1. Positively identified bevel/crown-only/worm gears, chain sprockets and flange-confined timing-belt teeth are 0. A bore is not required: integral shaft pinions may meet G and D simultaneously. Do not infer gear module or mating fit. Pilot D008/D030 versus D014/D016.

### D — `shaft_drive`

An elongated continuous journal/shaft span plus positive rotational transfer feature (keyway, spline, integral gear, crank/eccentric, opposed drive flats/attachments) supports 2. A visibly elongated smooth/simply stepped round journal without such evidence is 1. A short pin/stub with a head/shoulder but no elongated journal span is 0; a headless thread-only linear screw and headed clamping screw are 0. Long shafts with complete end joints may meet C and D, unlike short all-joint connectors. Shape proportions guide the distinction; no new metric cutoff is introduced. Pilot D025/D026 versus short headed D035; D036/D040 are conditional.

### P — `compression_return`

Open-pitch helical coil, cylindrical/conical/barrel form, with hook-free axial seating ends supports 2. Flat-ground ends are not required. Hook-free close coil with travel/seating unresolved or curved/elliptical form with uncertain axial seating is 1. Closed toroidal coils, tensile hooks, torsion arms, flat clips and threaded rods are 0. Pilot D005/D011/D037/D041 versus close D010 and torsion D027.

### E — `tension_return`

Helical extension body + two opposed tensile hooks/loops supports 2, including a connected multi-coil form with clear terminal hooks. One hook or concealed second attachment on a plausible extension form is 1. Long tangential/angular torsion arms, bare open compression coils and flat clips are 0. Pilot D007/D032 versus D027.

## Evidence feature names

Use these feature names for each positive, conditional, or unresolved role, with concrete geometry details and view names:

| Role | Features |
|---|---|
| S | `body`, `seat`, `mounting` |
| C | `connection`, `two_ends`, `retention` |
| B | `wheel`, `belt_contact`, `profile` |
| T | `head`, `shank`, `external_helix` |
| N | `body`, `turning`, `internal_helix` |
| M | `rigid`, `two_faces`, `fixing` |
| F | `passage`, `bolt_pattern`, `register`, `shaft_interface` |
| G | `cylindrical`, `teeth`, `profile` |
| D | `elongated`, `journal`, `drive` |
| P | `coil`, `travel`, `axial_ends` |
| E | `coil`, `hooks`, `tensile_ends` |

For roles rejected by form, describe the observed incompatibility. Assessments concern visible geometry; material, load capacity, thread size, gear module, and actual service require additional evidence. Choose grades from the source evidence independently of their effect on retrieval metrics.
