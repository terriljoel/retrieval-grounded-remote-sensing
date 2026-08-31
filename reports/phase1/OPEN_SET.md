# Leave-one-class-out open-set test

Does `novelty = 1 − sim@1` flag an object class the memory has never seen?

For each NWPU class C, **every case whose verified class is C is removed from the memory** —
accepts, relabels and localisation rejects alike, so no class-C pixels remain under any
decision. Val accepts of class C are then queried against that reduced memory and compared
against val accepts of the nine classes it still knows.

Encoder `remoteclip_l14`, arm `local`, α = 0.5, k = 4 (the sweep's selection, read back from
`alpha_sweep_summary.csv`). Memory = train accepts, queries = val accepts, test untouched.

**Why this rather than a second dataset.** Imagery, sensor, resolution, crop geometry and
encoder are all held fixed; the only thing that changes is whether the memory has seen the
class. Swapping in HRRSD cannot separate "novel class" from "different dataset", and
RemoteCLIP having trained on HRRSD would confound it further. The HRRSD open-set classes
remain the natural complement — but as a second, dirtier test, not this one.

## Results

`thr` is the novelty threshold that flags 95% of the held-out class as novel; `false-flag` is
the fraction of familiar accepts that same threshold wrongly flags. Reported at the operating
point per the standing rule, with AUROC alongside.

| held-out class | n query | memory after removal | mean novelty (held-out) | mean novelty (familiar) | AUROC | thr | false-flag @95% |
|---|---|---|---|---|---|---|---|
| baseball_diamond | 65 | 2573 | 0.166 | 0.054 | 1.000 | 0.143 | 0.0% |
| bridge | 16 | 2750 | 0.222 | 0.052 | 1.000 | 0.181 | 0.0% |
| storage_tank | 60 | 2317 | 0.198 | 0.053 | 1.000 | 0.169 | 0.0% |
| airplane | 144 | 2330 | 0.185 | 0.058 | 1.000 | 0.160 | 0.0% |
| vehicle | 74 | 2400 | 0.198 | 0.051 | 1.000 | 0.166 | 0.0% |
| ship | 53 | 2641 | 0.170 | 0.055 | 1.000 | 0.136 | 0.2% |
| ground_track_field | 24 | 2728 | 0.169 | 0.053 | 0.999 | 0.124 | 0.4% |
| harbor | 17 | 2661 | 0.150 | 0.053 | 0.999 | 0.123 | 0.4% |
| basketball_court | 19 | 2731 | 0.146 | 0.053 | 0.998 | 0.117 | 0.4% |
| tennis_court | 59 | 2447 | 0.140 | 0.052 | 0.997 | 0.109 | 0.6% |

**Macro AUROC 0.999** (min 0.997, max 1.000).
**Macro false-flag at 95% novel detection: 0.2%** (min 0.0%, max 0.6%).

## Reading

**Novelty survives as a gate feature, decisively and for every class.** A held-out class sits
at novelty 0.14–0.22 against a familiar baseline of ~0.05 — a 3–4× gap, and the separation
holds even for the classes with the fewest query examples (bridge n=16, harbor n=17).

This is stronger than the background result from checkpoint 4 (novelty AUROC 0.981 on
background crops, 8.8% cost). Flagging an *unfamiliar object* is easier than flagging *no
object*: an unseen class still produces a coherent, confidently-embedded crop that simply has
no near neighbour, whereas a background patch can accidentally resemble a real object.

**The ranking is interpretable.** The hardest classes to flag as novel are `tennis_court`
(0.997, 0.6%) and `basketball_court` (0.998, 0.4%) — the two that most resemble each other,
so removing one leaves the other as a plausible neighbour. The easiest is `bridge` (novelty
0.222), which has no visual analogue among the remaining nine. That the difficulty ordering
tracks visual similarity rather than class frequency is evidence the measurement is real and
not an artifact of memory size.

## Limitations, stated

1. **The NWPU classes are visually distinct.** Ten well-separated categories is a favourable
   case. Fine-grained novelty — a ship class the memory has not seen against ship classes it
   has — is not tested here and would be harder.
2. **Queries are clean GT accepts.** A novel class arriving from a real detector also carries
   localisation error, and §BASELINES shows localisation is exactly where the gate is blind.
   The two effects compound and are not measured together yet.
3. **HRRSD's three open-set classes are the harder complement** and are not run yet:
   `crossroad`, `parking_lot`, `t_junction` are all road-and-pavement scenes that overlap
   visually with NWPU's `vehicle` and `bridge` contexts, so they will not separate as cleanly
   as a held-out NWPU class does. Their number must also be reported separately —
   RemoteCLIP trained on HRRSD.

Reproduce: `python3 -m scripts.run_open_set`. Output: `open_set_leave_one_class_out.csv`.
