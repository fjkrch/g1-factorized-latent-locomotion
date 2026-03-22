# ICRA 2026 Workshop Reframing Guide
## "Generative Digital Twins for Real2Sim and Sim2Real Transfer in Robotics"

---

## A. Core Repositioning

1. **Foreground: evaluation methodology for transfer-relevant representations.** The paper's strongest asset for this workshop is not DynaMITE itself, but the *evaluation protocol*: factorial ablation, probing, intervention, disentanglement metrics, and multi-axis OOD stress testing. Frame this as "what the sim2real community should be doing before claiming a representation is transfer-ready."

2. **Foreground: challenging a widespread but unverified sim2real assumption.** Auxiliary dynamics supervision (RMA and descendants) is a backbone of modern sim2real locomotion. The community assumes these representations encode dynamics. This paper shows, with controlled evidence, that they do not—at least in one well-controlled setting. This is directly relevant to anyone building adaptive digital twins that rely on learned dynamics estimators.

3. **Foreground: bottleneck compression as the actual mechanism.** The factorial ablation isolates a clean, actionable insight: dimensionality compression through a tanh bottleneck (not explicit dynamics supervision) drives the OOD advantage. This reframes the contribution as a *design guideline for simulation-side policies*—compress your representation, don't assume auxiliary losses do what you think.

4. **Foreground: OOD robustness evaluation as a proxy for transfer readiness.** The combined-shift stress test (simultaneous friction + push + delay perturbation) is a simulation-side proxy for the compound dynamics mismatch encountered during sim2real transfer. Frame this as *transfer-relevant robustness profiling*.

5. **Downplay: the "method paper" framing.** Do not present DynaMITE as a proposed method. Present it as an *experimental vehicle* for testing a common sim2real design assumption. The paper is an evaluation study, not a method contribution.

6. **Downplay: nominal ID performance comparisons.** The LSTM-beats-everything result on ID reward is true but not the workshop story. Mention it briefly; do not make it the headline. The workshop audience cares about what transfers, not what wins in the training distribution.

---

## B. Title Options

1. **Does Auxiliary Dynamics Supervision Produce Transfer-Relevant Representations? Evidence from Simulated Humanoid Locomotion**
2. **Bottleneck Compression, Not Dynamics Supervision: Isolating Representation Mechanisms for Robust Simulated Locomotion**
3. **Evaluating Simulation-Side Inductive Biases for Transfer-Relevant Robustness in Humanoid Locomotion**
4. **What Does Auxiliary Dynamics Supervision Actually Encode? A Probing and Ablation Study for Sim2Real Representation Design**
5. **Transfer-Relevant Robustness Without Decodable Dynamics: A Factorial Study of Representation Design in Simulated Locomotion**
6. **Challenging the Dynamics Estimator Assumption: Probing Auxiliary-Loss Representations in Humanoid Locomotion**
7. **Representation Bottlenecks vs. Dynamics Supervision: Lessons for Robust Policy Transfer from Simulation**
8. **Simulation-Side Representation Evaluation for Transfer: When Auxiliary Dynamics Losses Do Not Encode Dynamics**
9. **How Should We Evaluate Simulation Representations Before Transfer? A Negative-Result Study on Factored Dynamics Supervision**
10. **Compression Beats Supervision: A Controlled Study of Representation Design Choices for OOD-Robust Locomotion Policies**

**Recommended for workshop submission:** Title 1 or 4 (question-form titles signal evaluation/negative-result work clearly and invite engagement). Title 10 is the punchiest if allowed.

---

## C. Abstract Rewrites

### Version 1: Conservative

Sim2real transfer for legged locomotion relies heavily on simulation-trained representations that are assumed to capture dynamics-relevant information. A common approach augments PPO training with per-factor auxiliary losses that supervise a latent space on randomized dynamics parameters—friction, mass, motor strength, contact stiffness, and action delay. Despite widespread adoption, whether these representations actually encode their target dynamics has not been directly verified with probing and intervention analyses.

We evaluate this assumption using DynaMITE, a transformer encoder with a factored 24-d latent trained with per-factor auxiliary dynamics losses, compared against LSTM, Transformer, and MLP baselines on a Unitree G1 humanoid across four Isaac Lab tasks. Under linear and nonlinear probes, clamping interventions, and standard disentanglement metrics (MIG, DCI, SAP), the supervised latent shows no evidence of decodable or functionally separable factor structure ($R^2 \approx 0$, all intervention effects $< 0.05$). An unsupervised LSTM hidden state achieves higher probe accuracy.

A $2 \times 2$ factorial ablation ($n = 10$ seeds) isolates a tanh bottleneck—not the auxiliary supervision—as the component driving observed robustness differences under severe combined perturbation. These findings suggest that simulation-side representation designers should verify dynamics encoding directly rather than assuming auxiliary losses produce transfer-relevant structure, and that information compression may matter more than explicit dynamics supervision for OOD robustness. (213 words)

### Version 2: Strong but Honest

Auxiliary dynamics supervision is a foundational design choice for sim2real locomotion—yet the community routinely assumes these representations encode dynamics without direct verification. We provide the first controlled probing, intervention, and factorial ablation study testing whether per-factor auxiliary dynamics losses produce decodable, functionally separable, or transfer-relevant latent structure in simulated humanoid locomotion.

Using a Unitree G1 humanoid in Isaac Lab, we compare DynaMITE (a transformer encoder with a factored 24-d latent and per-factor auxiliary losses) against LSTM, Transformer, and MLP baselines across four locomotion tasks. The result is a clear negative: probe $R^2 \approx 0$ for all five dynamics factors, clamping any subspace changes reward by $< 0.05$, and disentanglement metrics are near zero. An unsupervised LSTM hidden state encodes more dynamics information by every measure.

A $2 \times 2$ factorial ($n = 10$ seeds) reveals that the tanh information bottleneck—not the auxiliary supervision—drives the observed OOD robustness advantage (bottleneck: $+0.10$ under severe combined perturbation; auxiliary loss: $+0.03$, $p = 0.669$). This finding has direct implications for simulation-side representation design in transfer pipelines: practitioners should invest in compression architectures and multi-axis robustness evaluation protocols rather than assuming auxiliary dynamics losses produce usable estimators. We release a complete evaluation framework for verifying representation quality before transfer. (207 words)

### Version 3: Workshop-Optimized

A common assumption in sim2real locomotion is that auxiliary dynamics losses produce representations encoding environment parameters—enabling adaptive behavior that bridges the simulation-to-reality gap. We directly test this assumption and find it unsupported: in a controlled study of simulated humanoid locomotion with domain randomization, per-factor auxiliary supervision fails to produce decodable or functionally separable latent dynamics structure.

We evaluate DynaMITE, a transformer encoder with a factored 24-d latent trained by per-factor auxiliary losses during PPO, against LSTM, Transformer, and MLP baselines on a Unitree G1 humanoid across four Isaac Lab tasks. Probes yield $R^2 \approx 0$; latent interventions produce negligible behavioral effects; standard disentanglement metrics (MIG, DCI, SAP) are near zero. An unsupervised LSTM hidden state achieves higher probe accuracy across all factors.

Crucially, a $2 \times 2$ factorial ablation ($n = 10$) reveals that a simple tanh information bottleneck—not the explicit dynamics supervision—drives the observed advantage under severe multi-axis perturbation (a simulation-side proxy for compound sim2real mismatch). For the digital-twin and sim2real community, this study provides: (1) evidence that a widely used representation design assumption does not hold in this setting, (2) a multi-axis OOD robustness evaluation protocol relevant to transfer readiness assessment, and (3) the actionable finding that representation compression may matter more than explicit dynamics supervision for robust policy behavior under distribution shift. (214 words)

---

## D. Positioning Paragraph (for Introduction)

> Simulation-based policy training is the primary pathway to real-world deployment for legged robots, and the fidelity of simulation-side representations is a prerequisite for successful transfer. Digital-twin and sim2real pipelines increasingly rely on adaptive policies conditioned on learned dynamics estimators—yet the transferability of these representations is typically assessed only through downstream reward, not through direct verification of what they encode. This paper contributes to the sim2real pipeline by asking a prior question: *does a common supervision strategy actually produce the dynamics representation it is designed to produce?* We argue that rigorous simulation-side evaluation—probing, intervention, factorial ablation, and multi-axis robustness profiling—is a necessary step before transfer, and that negative results at this stage are as informative as positive ones. Even without a physical deployment experiment, our findings directly inform how digital-twin practitioners should design, verify, and trust the adaptive representations embedded in their simulation-trained policies.

---

## E. Contribution Bullets (Rewritten)

1. **A controlled evaluation of a common sim2real representation assumption.** We provide the first direct probing, intervention, and disentanglement analysis testing whether per-factor auxiliary dynamics supervision produces decodable or functionally separable latent structure in locomotion—a design choice adopted by RMA and its descendants. Under six complementary analyses, the answer is negative.

2. **Factorial isolation of the operative mechanism.** A $2 \times 2$ factorial ablation ($n = 10$ seeds) cleanly separates the contributions of information bottleneck compression and auxiliary supervision, finding that compression—not explicit dynamics prediction—drives observed robustness differences. This is an actionable design insight for simulation-side representation engineering.

3. **A multi-axis robustness evaluation protocol for transfer readiness assessment.** We introduce a combined-shift stress test (simultaneous friction, push, and delay perturbation) as a simulation-side proxy for the compound dynamics mismatch characteristic of sim2real transfer, and demonstrate that it reveals failure modes invisible to single-axis sweeps.

4. **Complete evaluation framework and negative-result transparency.** We release all code, configurations, and analysis tools, and report null findings in full—contributing to better calibration of expectations around auxiliary-loss representations in the sim2real locomotion pipeline.

---

## F. Section-by-Section Reframing

### Introduction

**Current framing:** "Does factor-wise auxiliary dynamics supervision produce useful latent structure?" — framed as a representation learning question.

**Workshop reframing:**
- Open with the sim2real pipeline: simulation → representation learning → transfer. Cite the reliance of modern sim2real locomotion on auxiliary dynamics supervision (RMA, etc.).
- Introduce the *verification gap*: these representations are adopted for transfer but never directly tested for what they encode. Frame this as a gap in the digital-twin pipeline.
- Position the paper as filling that gap: "Before transferring simulation-trained representations, we should verify what they contain."
- Move the DynaMITE description later; lead with the question, not the architecture.
- Add the positioning paragraph from Section D above.
- Keep the negative-result preview but frame it as *important for the community*: "Our findings caution against assuming auxiliary losses produce usable dynamics estimators in transfer pipelines."

### Related Work

**Current framing:** DR, history-conditioned policies, latent dynamics identification, representation analysis in RL.

**Workshop reframing:**
- Add a subsection or paragraph on **sim2real transfer pipelines and digital twins for locomotion**. Cite recent work on adaptive digital twins, system identification for transfer, and the role of learned dynamics estimators in closing the sim2real gap.
- Reframe the DR subsection as "Simulation Fidelity and Domain Randomization for Transfer"—emphasize that DR is the standard simulation-side strategy for bridging the reality gap.
- In the RMA/adaptive policies subsection, emphasize that these methods are *designed for transfer* but evaluated only through downstream reward. Position your probing/intervention methodology as what is missing.
- Add 2–3 citations relevant to the workshop: digital twin construction, sim2real benchmarks, evaluation protocols for transfer readiness. (If you don't have them, note them as "to add.")
- Cut or compress the pure representation-analysis-in-RL subsection; keep only what directly connects to the sim2real story.

### Method

**Current framing:** Detailed architecture description.

**Workshop reframing:**
- Present DynaMITE as an *instantiation of a common design pattern* (auxiliary dynamics supervision), not as a proposed method. Opening sentence: "To test whether per-factor auxiliary dynamics supervision produces transfer-relevant representations, we instantiate the design pattern as DynaMITE..."
- Shorten architecture details. The workshop audience cares about what you tested, not the transformer hyperparameters. Move detailed architecture to a figure + short caption.
- Emphasize the **evaluation methodology** as the primary methodological contribution: the probing protocol, the intervention protocol, the factorial design, and the combined-shift stress test.
- Rename or reframe the section: consider "Experimental Setup" or "Evaluation Framework" instead of "Method" to signal this is an evaluation paper.

### Experiments

**Workshop reframing:**
- **Lead with the factorial ablation** (Section 5.6 in current paper). This is the cleanest, most actionable result. Move it up.
- **Follow with the probing/intervention null result.** This is the headline negative finding.
- **Then the combined-shift robustness profile.** Frame it as "transfer-relevant robustness evaluation."
- **Compress or cut:** single-axis OOD sweeps (keep only one representative figure), push recovery (compress to one paragraph + table), Pareto analysis (compress to one sentence), nominal ID comparison (one table, brief discussion).
- **Remove or move to appendix:** detailed tracking error tables, per-factor gradient norms, geometry/SVD analysis, MI estimation details.
- See Section K for the full scope reduction plan.

### Discussion

**Current framing:** Operational regimes, practical takeaways, why LSTM degrades more.

**Workshop reframing:**
- Lead with **implications for sim2real representation design**: "For practitioners building adaptive policies for transfer, our results suggest that..."
- Add a paragraph on **what this means for digital-twin pipelines**: if auxiliary dynamics supervision does not produce decodable estimators, then real-time monitoring and updating of digital twins via learned representations requires different approaches.
- Reframe the ID–OOD tradeoff as a **simulation-side indicator of transfer robustness**: models that degrade less under simulation-defined compound perturbation may be better candidates for transfer, and compression may be the mechanism.
- Add a paragraph on **evaluation protocol recommendations**: what should the sim2real community check before trusting a representation for transfer?
- Cut the detailed LSTM-degradation speculation (Section 5.10 in current paper); compress to one sentence.

### Limitations

**Current framing:** Statistical power, probe family, simulation only.

**Workshop reframing:**
- Keep "simulation only" but reframe it: "All experiments are conducted in simulation, which is the intended scope. We do not claim real-world transfer; we argue that simulation-side evaluation is a necessary antecedent to transfer and is itself under-practiced."
- Keep probe-family limitation (honest, and a sim2real audience will appreciate it).
- Add: "Our combined-shift protocol is a proxy for sim2real mismatch but is not calibrated to any specific real-world platform."
- Cut or compress: sensitivity metric limitation (too methodological for a workshop), deterministic-only evaluation (minor), narrow reward spread (not relevant to workshop story).

---

## G. Reviewer-Facing Framing

### Objection 1: "This is not really about digital twins."

**Rebuttal:** Digital twins for sim2real transfer rely on learned dynamics estimators embedded in simulation-trained policies. If those estimators do not encode what they claim to—as we demonstrate—then the adaptive mechanism assumed by the digital-twin pipeline is broken at the representation level. Ensuring representation fidelity is a prerequisite for functional digital twins. Our evaluation tools (probing, intervention, factorial ablation) are directly applicable to any digital-twin framework that uses learned adaptive policies.

### Objection 2: "This is only simulation—no real Sim2Real experiment."

**Rebuttal:** The workshop scope explicitly includes "benchmarks, datasets, and evaluation protocols for Real2Sim and Sim2Real." Our contribution is precisely an evaluation protocol for verifying simulation-side representation quality before transfer. If a representation fails our simulation-side tests, transferring it to hardware is premature. This is analogous to unit testing before integration testing: finding representation failures in simulation is cheaper and faster than discovering them on hardware. We note that the vast majority of sim2real locomotion papers evaluate their representations only through downstream reward, not through direct verification—our paper fills that gap.

### Objection 3: "This is just a negative result."

**Rebuttal:** Negative results that challenge widely adopted design assumptions are among the most valuable contributions a workshop can publish. Auxiliary dynamics supervision (RMA and descendants) is one of the most common representation design choices in sim2real locomotion. We show, with controlled factorial evidence, that it does not produce what it is assumed to produce. This recalibrates community expectations and redirects effort toward more promising approaches (e.g., compression). The workshop's non-archival format is an ideal venue for precisely this type of finding.

### Objection 4: "No true Sim2Real experiment."

**Rebuttal:** We do not claim sim2real transfer. We claim that simulation-side evaluation of transfer-relevant representations is under-practiced and that our methodology reveals failures invisible to standard reward-only evaluation. The compound perturbation protocol simulates the *type* of mismatch encountered in transfer (simultaneous friction, force, and latency shifts). We explicitly state hardware validation as future work. The workshop scope includes "learning algorithms that use continuously updated twins for robust, scalable, generalizable robot behavior"—our evaluation of robustness under compound perturbation fits directly.

### Objection 5: "The results are not statistically significant."

**Rebuttal:** The *negative* findings (no decodable factor structure) are confirmed: $R^2 \approx 0$, intervention effects $< 0.05$, disentanglement metrics near zero. The factorial main effect of auxiliary losses is confirmed as null ($p = 0.732$ for ID, $p = 0.669$ for OOD). The bottleneck effect ($p \approx 0.2$) is consistent across both ID and OOD regimes with 10 seeds. Most OOD rankings are directional ($n = 5$), which we state transparently. The strength of a negative result is in ruling things out, and our evidence for "auxiliary losses do not help" is statistically clear.

### Objection 6: "The method (DynaMITE) is not novel enough."

**Rebuttal:** The contribution is the evaluation, not the method. DynaMITE is an experimental vehicle—an instantiation of the auxiliary-dynamics-supervision pattern—designed to be tested and dissected. The novelty lies in the evaluation methodology (factorial ablation isolating compression vs. supervision, six-tool representation analysis) and the finding (compression, not supervision, is the operative mechanism).

### Objection 7: "Only one robot / one simulator / one task family."

**Rebuttal:** We test across four distinct locomotion tasks (flat, push, randomized, terrain) and five perturbation axes, with 10-seed factorial and 5-seed main comparison, totaling ~80+ evaluation conditions. The scope is narrower than a multi-platform study but deeper: we provide causal decomposition (factorial ablation) and mechanistic analysis rather than broad but shallow benchmarking. Workshop papers are expected to have focused scope.

### Objection 8: "The paper is too long / has too many experiments for a workshop."

**Rebuttal:** We present a lean version in the workshop submission (see Section K), retaining only the factorial ablation, probe null result, and combined-shift robustness profile. The full analysis is available in the extended version. The workshop submission focuses on one clean argument: auxiliary dynamics supervision does not produce what it claims, and compression is the actual mechanism.

---

## H. Language to Use / Language to Avoid

| **Use** | **Avoid** |
|---|---|
| transfer-relevant robustness evaluation | we bridge the sim-to-real gap |
| simulation-side inductive bias | our digital twin system |
| representation design for generalizable policy learning | we propose a novel method |
| compound perturbation as a proxy for sim2real mismatch | we validate on real hardware |
| evaluation protocol for transfer readiness | our method outperforms |
| representation verification before transfer | state-of-the-art |
| simulation-side representation engineering | we close the reality gap |
| the auxiliary supervision does not produce the intended structure | the method fails |
| information bottleneck as an operative mechanism | our architecture is superior |
| directly testing a common sim2real assumption | we present DynaMITE |
| robustness profiling under distribution shift | we solve sim2real transfer |
| the dynamics estimator assumption | our digital twin learns |
| actionable design insight for sim2real practitioners | breakthrough |
| factorial isolation of causal components | we demonstrate sim2real |
| this finding recalibrates expectations | negative result (as a headline) |
| controlled study of representation design choices | our approach eliminates the need for |
| transfer-relevant representation analysis | we generate a digital twin |
| simulation fidelity evaluation | we reconstruct real environments |
| compound dynamics mismatch | reality gap (overused, vague) |
| representation compression | outperforms all baselines |
| design guideline for simulation policies | novel framework |
| evaluation methodology contribution | end-to-end pipeline |

---

## I. Submission-Ready Workshop Summaries

### 50-Word Summary

We test whether auxiliary dynamics supervision—a common sim2real design choice—produces decodable dynamics representations in simulated humanoid locomotion. It does not. A factorial ablation reveals that information bottleneck compression, not explicit dynamics supervision, drives robustness under compound perturbation. We provide evaluation protocols for verifying simulation-side representation quality before transfer.

### 100-Word Summary

Auxiliary dynamics supervision is widely used in sim2real locomotion to learn adaptive representations, yet whether these representations actually encode dynamics is seldom verified. We directly test this using probing, intervention, and disentanglement analyses on a transformer encoder with per-factor auxiliary losses, trained via PPO on a Unitree G1 humanoid across four Isaac Lab tasks. The supervised latent shows no evidence of decodable factor structure ($R^2 \approx 0$). A $2 \times 2$ factorial ablation ($n = 10$ seeds) isolates information bottleneck compression—not auxiliary supervision—as the mechanism driving OOD robustness. These findings provide actionable design insights for simulation-side representation engineering and a multi-axis robustness evaluation protocol relevant to transfer readiness assessment.

### 150-Word Summary

Sim2real transfer pipelines for legged robots rely on simulation-trained adaptive representations assumed to encode environment dynamics. We directly evaluate this assumption using probing, intervention, factorial ablation, and disentanglement analysis on a Unitree G1 humanoid trained with per-factor auxiliary dynamics losses (DynaMITE) alongside LSTM, Transformer, and MLP baselines across four Isaac Lab locomotion tasks.

The results challenge a foundational design assumption: the supervised factored latent shows $R^2 \approx 0$ under both linear and nonlinear probes across all five dynamics factors, negligible behavioral effects under clamping interventions, and near-zero disentanglement scores. An unsupervised LSTM hidden state achieves higher probe accuracy.

A $2 \times 2$ factorial ablation ($n = 10$) isolates information bottleneck compression—not explicit dynamics supervision—as the operative mechanism for OOD robustness under compound perturbation. We contribute a simulation-side evaluation framework for verifying transfer-relevant representation quality and provide evidence that compression architecture may matter more than dynamics supervision for robust policy behavior under distribution shift.

---

## J. Optional Stretch Version

### Aggressive Reframing: "Auditing the Adaptive Layer of Simulation-to-Reality Transfer Pipelines"

**Pitch:** Modern sim2real locomotion pipelines assume that auxiliary-loss representations form an implicit dynamics model that enables adaptive behavior during transfer. We provide the first *representation audit* of this adaptive layer—a systematic verification of what auxiliary dynamics supervision actually encodes, using tools borrowed from the interpretability and disentanglement literatures. Our audit reveals that the assumed dynamics encoding does not exist in this setting: the representation is not decodable, not functionally separable, and scores near zero on standard disentanglement metrics. The component that *does* drive robustness under compound perturbation—a proxy for sim2real dynamics mismatch—is the information bottleneck, not the explicit supervision.

**Interpretive claims (labeled):**
- *Interpretation:* "The combined-shift stress test serves as a simulation-side proxy for the compound dynamics mismatch encountered during physical deployment." (Not directly demonstrated—no hardware transfer was performed.)
- *Interpretation:* "Information bottleneck compression may be more important than explicit dynamics supervision for transfer-ready representations." (Supported by factorial evidence in simulation; not validated in a real transfer experiment.)
- *Interpretation:* "Representation auditing should become a standard step in sim2real pipelines before hardware deployment." (A recommendation, not a demonstrated workflow.)
- *Demonstrated:* "Auxiliary dynamics supervision does not produce decodable factor structure under the probes and interventions used." (Directly shown.)
- *Demonstrated:* "The tanh bottleneck, not auxiliary losses, drives the observed robustness differences." (Directly shown via factorial.)
- *Demonstrated:* "An unsupervised LSTM hidden state achieves higher probe R² than the explicitly supervised factored latent." (Directly shown.)

**Additional framing for this version:** Position the paper as contributing to the workshop's "safety, robustness, monitoring, uncertainty modeling with digital twins" theme. Argue that if practitioners intend to monitor or update digital twins using learned dynamics estimators, they must first verify that those estimators encode what they claim. Your paper provides the verification tools and demonstrates a case where verification fails.

---

## K. Scope Reduction Plan

### Current Experiment Groups

| # | Experiment Group | Current Sections | Decision | Justification |
|---|---|---|---|---|
| 1 | **Nominal ID comparison** (4 models × 4 tasks × 5 seeds) | §5.1, Table 1–3, Fig 2 | **Compress** | Keep one table (Table 1). Cut paired t-test table and bar chart. LSTM winning is needed as context, but it's not the workshop story. |
| 2 | **Combined-shift stress test** (4 models × 5 severity levels × 5 seeds) | §5.2, Tables 4–6, Fig 3 | **Keep** | This is the core "transfer-relevant robustness" evidence. Keep reward table and figure. Compress tracking error table to one sentence. |
| 3 | **Pareto analysis** (ID vs. severe OOD) | §5.3, Fig 4, Table 7 | **Remove** | Redundant with the combined-shift results. The Pareto framing adds complexity without new insight for the workshop story. |
| 4 | **Push recovery** (4 models × 7 magnitudes × 5 seeds) | §5.4, Tables 8–9 | **Remove** | Interesting but peripheral to the transfer/representation story. The faster-threshold-but-worse-gait nuance is hard to fit into a tight workshop paper. |
| 5 | **Single-axis OOD sweeps** (push, friction, delay × 3 tasks) | §5.5, Tables 10–11, Fig 5 | **Remove** | The combined-shift stress test (Experiment 2) is stronger and more transfer-relevant. Single-axis sweeps are redundant when time/space is tight. |
| 6 | **Factor alignment** (custom metric) | §5.6.1, Fig 6 | **Compress** | Merge into probing section as one sentence: "A custom within-factor correlation metric shows moderate alignment (0.50 vs. 0.20 chance) but this does not indicate decodability." |
| 7 | **Intervention analysis** (5 factors × 3 DR levels × 3 seeds) | §5.6.2, Table 12 | **Keep** | Directly supports the headline negative result. Keep the table; it's small. |
| 8 | **Latent probe analysis** (5 factors × 2 probe types × 2 models) | §5.6.3, Table 13 | **Keep** | Directly supports the headline negative result. R² ≈ 0 is the most quotable finding. |
| 9 | **Ablation study** (5 variants × 10 seeds) | §5.7, Table 14 | **Compress** | Keep only the 2×2 factorial (Experiment 10). The individual ablation table is superseded by the factorial. |
| 10 | **2×2 Factorial: ID** (4 cells × 10 seeds) | §5.7.1, Table 15 | **Keep** | This is the core mechanistic contribution. Bottleneck vs. auxiliary loss decomposition. |
| 11 | **2×2 Factorial: OOD** (4 cells × 10 seeds × severe levels) | §5.7.2, Tables 16–17 | **Keep** | Confirms the bottleneck finding extends to OOD. Keep both tables. |
| 12 | **Evidence summary table** | §5.8, Table 18 | **Keep** | Excellent for workshop readability. Compress slightly. |
| 13 | **Gradient flow analysis** (3 seeds × 10M steps) | §6.1, Table 19, Fig 7 | **Remove** | Mechanistic detail that is interesting but not essential for the workshop argument. Mention orthogonality in one sentence if needed. |
| 14 | **Representation geometry / SVD** (5 seeds × 36k samples) | §6.2, Table 20, Fig 8 | **Remove** | Effective rank ≈ 5 is a nice fact. Mention it in one sentence in the discussion. Full analysis is excessive for workshop scope. |
| 15 | **Mutual information estimation** (5 seeds) | §6.3, Table 21, Fig 9 | **Remove** | MI is low for both models. The result is consistent with probes but adds no new insight for the workshop story. |
| 16 | **Standard disentanglement metrics** (MIG, DCI, SAP) | §6.4, Table 22 | **Compress** | One sentence: "MIG, DCI, SAP scores are near zero for both models, confirming the probe findings." Keep the numbers but not the table. |

### Lean Workshop Version: Minimum Experiments

**Keep (4 experiment blocks):**

1. **2×2 Factorial: ID + OOD** (Tables 15–17) — The core mechanistic contribution. Cleanly isolates compression vs. supervision.
2. **Probing null result** (Table 13) — The headline negative finding. R² ≈ 0 is the single most quotable number.
3. **Intervention null result** (Table 12) — Complements probing. Shows the latent is not functionally separable.
4. **Combined-shift stress test** (Table 4, Fig 3) — The transfer-relevant robustness evaluation.

**Compress to one table + one paragraph:**

5. **Nominal ID comparison** (Table 1 only) — Context for "LSTM wins ID, but that's not the whole story."

**Compress to a single sentence each:**

6. Factor alignment, standard disentanglement metrics, effective rank — Brief mentions, no tables.

**Remove entirely:**

7. Push recovery, single-axis OOD sweeps, Pareto analysis, gradient flow, MI estimation, paired t-test tables, tracking error tables, auxiliary head convergence details.

### Resulting Workshop Paper Structure

```
1. Introduction (1 page)
   - Sim2real relies on simulation-side representations
   - Auxiliary dynamics supervision is standard but unverified
   - We test the assumption and find it unsupported
   - Bottleneck compression is the actual mechanism
   - Contributions (4 bullets from Section E)

2. Background and Related Work (0.5 page)
   - DR for sim2real transfer
   - Auxiliary dynamics supervision (RMA and descendants)
   - Gap: no direct representation verification

3. Experimental Setup (0.75 page)
   - DynaMITE as instantiation of the design pattern (figure)
   - Baselines (table, compressed)
   - Evaluation protocol (brief)

4. Results (1.5 pages)
   4.1 Probing and Intervention: No Decodable Factor Structure
       - Probe R² table
       - Intervention table
       - One sentence on disentanglement metrics
   4.2 Factorial Ablation: Compression vs. Supervision
       - 2×2 ID table
       - 2×2 OOD table
       - Degradation analysis
   4.3 Transfer-Relevant Robustness Profile
       - Combined-shift table + figure
       - Brief ID comparison for context

5. Discussion and Implications (0.5 page)
   - What this means for sim2real representation design
   - Compression > supervision as a design guideline
   - Evaluation recommendations for the community

6. Limitations and Future Work (0.25 page)
   - Simulation only (by design)
   - Probe family
   - Hardware validation as next step

Total: ~4.5 pages + references (fits ICRA workshop 6-page limit)
```

---

## Summary of Key Messaging

**One-sentence pitch:** "We show that auxiliary dynamics supervision—a standard sim2real design choice—does not produce the dynamics representation it is assumed to, and that information bottleneck compression is the actual mechanism driving robustness under compound perturbation."

**Why this workshop:** This paper is a *representation audit* for a design pattern used throughout the sim2real locomotion community. It provides evaluation tools, a clean negative result, and an actionable alternative (compression over supervision) that the digital-twin audience can immediately apply.

**Why negative results matter here:** The workshop invites work on "learning algorithms that use continuously updated twins for robust, scalable, generalizable robot behavior." If the adaptive representations inside those twins don't encode what they claim, the entire pipeline is built on an unverified assumption. Our paper makes that verification possible and demonstrates that it's needed.
