# Editorial Revision Plan: ICRA 2026 Workshop Submission

---

## 1. Diagnosis

The paper contains genuine technical substance—factorial ablation, multi-tool representation verification, honest statistical reporting—but wraps it in framing that reads like a deflated main-conference submission rather than a purpose-built workshop paper. The core problems:

**Overclaiming throughout.** The paper repeatedly reaches beyond its evidence base. "The central finding is unambiguous" (Discussion §6.1) is flatly wrong: the bottleneck main effect has $p \approx 0.2$, most OOD rankings are directional at $n\!=\!5$, and only 3 of 42 comparisons survive correction. The evidence for "auxiliary losses don't help" is reasonably strong ($p = 0.73$, $p = 0.67$), but the evidence for "bottleneck is the operative mechanism" is consistent-but-underpowered. The paper claims certainty it has not earned.

**Prosecutorial tone.** Section headers like "The Supervision Signal Does Not Produce What It Promises" read as adversarial rather than scientific. The phrase "does not produce what it promises" implies intentional deception by prior work. The actual finding is narrower: in this specific setting, with these readouts, the intended factorization was not detected. That is a careful negative result, not an indictment.

**Scope mismatch.** The paper addresses "the sim2real community" and "digital-twin pipelines" without any hardware experiment. It offers "practical recommendations" and "operating regimes" as if it were a deployment guide. These gestures are transparent to reviewers: simulation results on one platform do not license community-wide design guidelines.

**Identity confusion.** The paper oscillates between being a "verification study" (its strongest framing), a "negative-result report" (legitimate for a workshop), and a "DynaMITE method paper" (its weakest framing). The architecture gets disproportionate real estate relative to the evaluation methodology, which is the actual contribution.

**Structural bloat.** 14 tables and 2 figures in 7 pages. The paper reads like a data dump with commentary instead of a focused argument. Several tables (geometry, regimes, degradation) add marginal value and consume space that should go to interpretation.

**What is actually strong:**
- The probe/intervention/disentanglement null result is clean and well-documented
- The $2 \times 2$ factorial is well-designed and correctly separates two confounded variables
- The combined-shift stress test is a genuinely useful evaluation idea
- Statistical reporting is honest (CIs, correction, $p$-values stated)
- The paper acknowledges its own limitations

**Bottom line:** The evidence supports a careful, scoped workshop paper about verifying representation claims in one simulated setting. It does not support the broad, directive framing currently in the draft. The revision should shrink the claims to fit the evidence, not inflate the evidence to fit the claims.

---

## 2. Candidate Titles

At least 2 signal negative result; at least 2 signal verification/evaluation.

1. **Auxiliary Dynamics Supervision Without Decodable Representations: A Verification Study in Simulated Humanoid Locomotion**
   - Signals: negative result (without decodable representations) + verification + scoped to simulation
   
2. **Does Factorized Auxiliary Supervision Produce Verifiable Latent Structure? A Negative Result from Simulated Humanoid Locomotion**
   - Signals: negative result (explicit) + verification (verifiable) + scoped
   
3. **Verifying Representation Claims in Locomotion Policies: Probes, Interventions, and Factorial Ablations**
   - Signals: verification toolkit (probes, interventions, factorial) + evaluation focus
   
4. **Probing a Common Sim2Real Assumption: Factorized Dynamics Losses Do Not Induce Decodable Latent Structure in Our Setting**
   - Signals: negative result (do not induce) + scoped (in our setting) + evaluation
   
5. **Bottleneck Compression vs. Auxiliary Supervision for Robust Locomotion: A Factorial Study**
   - Signals: the actual finding + evaluation study framing

**Recommendation:** Title 2 is the strongest workshop choice. It asks a question (inviting engagement), signals a negative result, uses "verifiable" to flag the evaluation angle, and scopes to simulation. Title 1 is the safest alternative.

**Current title problem:** "Do Learned Dynamics Representations Survive Verification?" implies the paper answers a question about *all* learned dynamics representations. It doesn't—it tests one architecture on one platform. The grandiosity is the first thing a reviewer sees.

---

## 3. Revised Abstract

(205 words)

> Auxiliary dynamics supervision—training a factored latent space to predict randomized simulator parameters—is a common design choice in simulation-trained locomotion policies, with the expectation that the resulting representation encodes environment dynamics in a decodable or functionally separable form. We test this expectation in a controlled setting: a transformer encoder (DynaMITE) with a factored 24-d latent and per-factor auxiliary losses, compared against LSTM, Transformer, and MLP baselines on a Unitree G1 humanoid across four Isaac Lab locomotion tasks.
>
> Using linear and nonlinear probes, clamping interventions, and standard disentanglement metrics (MIG, DCI, SAP), we find limited evidence that the intended factor structure emerges: probe $R^2 \approx 0$ across all five dynamics factors, intervention effects are negligible ($|\Delta r| < 0.05$), and disentanglement scores are near zero. An unsupervised LSTM hidden state achieves higher probe accuracy on every factor.
>
> A $2 \times 2$ factorial ablation ($n\!=\!10$ seeds) separates the contributions of the $\tanh$ information bottleneck and the auxiliary losses: auxiliary supervision shows no measurable effect on reward ($p > 0.66$), while the bottleneck shows a consistent but sub-significant advantage ($p \approx 0.2$). These results suggest that representational claims in adaptive locomotion policies warrant direct verification, and that compression architecture may be more predictive of robustness than explicit factor supervision in this setting.

**Key changes from current abstract:**
- Removed "Crucially" (overclaiming)
- Replaced "find it unsupported" with "find limited evidence that the intended factor structure emerges" (less prosecutorial)
- Added "in a controlled setting" to scope the claim
- Changed "For the sim2real community, this provides..." (directive) to "These results suggest..." (appropriately tentative)
- Stated "consistent but sub-significant" for the bottleneck effect—honest about the $p \approx 0.2$
- Removed numbered contribution list from abstract (save for introduction)
- Removed keywords-style laundry list of analyses from abstract

---

## 4. Revised Introduction Framing

### Replacement opening paragraph

> Simulation-trained locomotion policies increasingly condition behavior on learned dynamics estimators, typically trained with auxiliary losses that supervise a latent representation on randomized simulator parameters [RMA, RMA-followup, DRAC]. The implicit design expectation is that the resulting latent space is either *decodable*—recoverable by standard probes—or *functionally separable*—such that manipulating a subspace selectively changes policy behavior. In practice, however, the representation itself is rarely examined: adaptation quality is assessed through downstream reward alone [probing_rl_1, probing_rl_2]. Whether auxiliary supervision actually induces the intended latent structure is, in most published work, an untested assumption.

**Why this is better:** Opens with the design pattern (factual), states the assumption (precise), identifies the gap (representation never verified), and avoids sweeping phrases like "digital-twin pipelines" or "the sim2real community." No prosecutorial framing.

### Clean problem statement

> This paper asks whether one specific instantiation of this design pattern produces the latent structure it is intended to produce. We construct DynaMITE (Dynamics-Matching Inference via Transformer Encoding): a two-layer transformer encoder mapping an 8-step history to a 24-d latent partitioned into five factor subspaces, each trained with a dedicated auxiliary loss during PPO on a Unitree G1 humanoid in Isaac Lab. We then apply a battery of verification tools—linear and nonlinear probes, clamping interventions, disentanglement metrics, and a $2 \times 2$ factorial ablation—to test whether the intended factorization is present.

**Why this is better:** "one specific instantiation" scopes the claim from the start. DynaMITE is presented as an *experimental vehicle*, not a proposed method. The verification toolkit is foregrounded over the architecture.

### Scope-narrowing paragraph

> Our investigation is scoped to simulated humanoid locomotion with domain randomization. We do not claim that auxiliary dynamics supervision is without value in all settings, nor that the representations we test are representative of all auxiliary-loss designs. We report what we find under the readouts we apply, on the platform we use, at the sample sizes we run. Where evidence is sub-significant or directional, we say so. This is a verification study, not a proof of impossibility.

**Why this is needed:** The current draft buries its caveats in a late "Scope and Non-Claims" subsection that reads like a disclaimer. Putting the scope constraint in the introduction sets the honest frame before the reader encounters any results.

### Revised contributions (exactly 3)

> Our contributions are:
>
> 1. **A scoped empirical test of a common representation-learning assumption.** Under probing, intervention, and disentanglement analysis in simulated humanoid locomotion, we find limited evidence that per-factor auxiliary dynamics supervision produces decodable or separable latent structure. An unsupervised LSTM hidden state achieves higher probe $R^2$ on every factor.
>
> 2. **A verification toolkit for representation claims in locomotion policies.** We combine six diagnostic tools—linear probes, MLP probes, clamping interventions, MIG, DCI, SAP—with a $2 \times 2$ factorial design that separates bottleneck compression from auxiliary supervision. This toolkit is reusable for evaluating representation quality in other adaptive policy architectures.
>
> 3. **Evidence that compression is more predictive of robustness than explicit factor supervision in this setup.** The factorial ablation shows no measurable effect of auxiliary losses ($p > 0.66$) while the bottleneck shows a consistent advantage ($p \approx 0.2$). Under compound perturbation, bottleneck models degrade less, but this appears to reflect a training-time representation benefit rather than a robustness-specific mechanism.

**Key changes from current contributions:**
- Contribution 1: Added "scoped" and "limited evidence" instead of "six complementary analyses show that... does not produce"
- Contribution 2: Reframed from "factorial isolation of the operative mechanism" (a result claim) to "verification toolkit" (a methodological contribution)—this is what survives as reusable
- Contribution 3: Added "in this setup," stated $p$-values, said "appears to reflect" instead of "drives"
- Removed: "Multi-axis robustness protocol for transfer readiness" is now demoted to a supporting detail, not a headline contribution—the paper doesn't validate the protocol against actual transfer

---

## 5. Revised Discussion/Conclusion

### Discussion

#### 5.1 Limited Evidence for the Intended Factor Structure

> Across six diagnostic tools applied to the DynaMITE architecture in our simulated humanoid locomotion setting, we do not find strong evidence that per-factor auxiliary dynamics supervision produces the latent factor structure it is designed to produce. Probes yield $R^2 \approx 0$, clamping interventions produce negligible reward changes, and standard disentanglement metrics are near zero. An unsupervised LSTM hidden state scores higher on every readout.
>
> We emphasize that this is a negative result within one setting, not a proof of impossibility. More expressive readouts (nonlinear ICA, manifold-aware probes) may recover structure that our tools miss. Different architectures, loss weightings, or training regimes may produce different outcomes. What our results do establish is that the commonly assumed correspondence between auxiliary supervision and decodable latent structure should not be taken for granted—it requires verification.

**Why this is better:** "We do not find strong evidence" instead of "The central finding is unambiguous." Explicitly states limitations of the negative result. No community-wide directives.

#### 5.2 Compression vs. Supervision

> The $2 \times 2$ factorial provides the cleanest test in our study: within the same transformer architecture, toggling auxiliary losses on or off produces no measurable reward difference (ID: $p = 0.73$; OOD: $p = 0.67$), while toggling the $\tanh$ bottleneck produces a consistent advantage that does not reach conventional significance ($p \approx 0.2$ at $n\!=\!10$).
>
> We interpret this cautiously. The bottleneck effect is directionally consistent across both ID and OOD conditions, and the Aux-Only cell matching No-Latent ($-4.64$ vs. $-4.64$) provides converging evidence that the auxiliary signal contributes little beyond what the bottleneck alone provides. But "consistent and sub-significant" is not "confirmed." Expanding to $n\!=\!20$ with bootstrap CIs is needed to resolve the bottleneck main effect.
>
> Mechanistic observations—gradient orthogonality between auxiliary and PPO losses, low effective rank ($\sim$5 of 24 dimensions), modestly higher MI—are consistent with a compression account but do not establish it causally.

**Why this is better:** States $p$-values and honestly calls the bottleneck effect sub-significant. Does not say "cleanly separates" without qualification. Mechanistic observations are described as "consistent with" rather than confirming.

#### 5.3 Compound Perturbation as a Stress Test

> The combined-shift protocol reveals a failure mode invisible to single-axis evaluation: LSTM's reward advantage disappears under simultaneous friction, push, and delay perturbation, while DynaMITE's reward degrades less (2.3% vs. 16.7%). This is a useful simulation-side stress test, but we note two caveats: (i) the compound perturbation levels are not calibrated to any specific hardware platform, so their relevance to physical sim2real mismatch is indirect; and (ii) the $n\!=\!5$ comparison is directional—most rankings do not survive multiple-comparison correction.

**Why this is better:** Calls the stress test "useful" rather than "transfer-relevant" (which implies hardware validation it doesn't have). Two explicit caveats.

#### 5.4 Why This Matters for Workshop Readers

> For researchers building adaptive locomotion policies: our results suggest that verifying what a learned representation actually encodes—before relying on it for monitoring, adaptation, or transfer—is a step that published work rarely takes but probably should. If auxiliary supervision does not produce decodable structure in our well-controlled simulation setting, the same assumption should not be accepted uncritically in more complex pipelines.
>
> The verification toolkit itself—probes, interventions, disentanglement metrics, factorial ablation—is portable. It can be applied to RMA-family methods, latent dynamics models, or any adaptive policy architecture where the representation is assumed to encode environmental parameters.

### Conclusion

> We evaluated whether factorized auxiliary dynamics supervision produces decodable, separable, or disentangled latent structure in simulated humanoid locomotion. Under the diagnostic tools we applied, it does not. A factorial ablation provides consistent but sub-significant evidence that the $\tanh$ information bottleneck, not the auxiliary losses, is the component more associated with robustness differences under compound perturbation.
>
> These are simulation-only results on a single platform (Unitree G1, Isaac Lab). We do not claim that auxiliary supervision is without value in all settings, nor that our readouts are exhaustive. What we do claim is that the link between auxiliary dynamics losses and structured latent representations should be verified, not assumed—and we provide a reusable toolkit for doing so.
>
> Hardware deployment to test whether the observed tradeoffs transfer to physical dynamics mismatch is the natural next step.

---

## 6. Tone Corrections

Ten patterns that should be softened across the draft. Each targets scientific credibility, not grammar.

| # | Location | Avoid | Replace with | Why |
|---|----------|-------|-------------|-----|
| 1 | Discussion §6.1 header | "The Supervision Signal Does Not Produce What It Promises" | "Limited Evidence for the Intended Factor Structure" | "Does not produce what it promises" is prosecutorial—implies prior work made a false promise. The paper tests an assumption, not a promise. |
| 2 | Discussion §6.1, line 1 | "The central finding is unambiguous" | "Across six diagnostic tools in our setting, we do not find strong evidence that..." | The bottleneck $p \approx 0.2$ is not unambiguous. Most OOD comparisons are directional. The negative probe result is clear, but the paper's *central* claim (compression > supervision) is suggestive, not unambiguous. |
| 3 | Discussion §6.1 | "This is not a marginal failure" | Remove entirely, or: "The null result is consistent across all readouts" | "Not a marginal failure" is rhetorical escalation. Let the numbers speak. |
| 4 | Discussion §6.1 | "For digital-twin and sim2real pipelines that rely on learned dynamics estimators... this is a concrete warning" | "For researchers using auxiliary-supervision architectures, our results suggest that representation quality should be verified directly" | "Concrete warning" is tabloid language. "Our results suggest" is scientific. |
| 5 | Discussion §6.2 header | "Compression, Not Supervision, Is the Operative Mechanism" | "Compression Appears More Associated with Robustness Than Supervision" | "Is the operative mechanism" is a causal claim from correlational evidence at $p \approx 0.2$. "Appears more associated" is accurate. |
| 6 | Discussion §6.2 | "The practical recommendation is direct: invest in compression architectures" | "In our setting, the bottleneck contributed more than the auxiliary losses. Whether this generalizes to other architectures and tasks is an open question." | The paper has one platform, one architecture family, sub-significant $p$-values. It cannot issue "practical recommendations" as directives. |
| 7 | Abstract | "Crucially, a $2 \times 2$ factorial ablation..." | "A $2 \times 2$ factorial ablation..." | "Crucially" is hype. The result's importance should be self-evident from its content. |
| 8 | Introduction, contribution 1 | "Six complementary analyses... show that per-factor auxiliary supervision does not produce decodable or separable latent structure" | "Six complementary analyses find limited evidence that per-factor auxiliary supervision produces the intended decodable or separable latent structure" | "Show that X does not produce Y" is a universal negative. "Find limited evidence that X produces Y" is falsifiable and scoped. |
| 9 | Results §5.8, probing section | "the supervised latent is neither decodable, functionally separable, nor disentangled under our analyses" | "under our analyses, the supervised latent does not exhibit clearly decodable, functionally separable, or disentangled structure" | The original formulation reads as a definitive verdict. The revision keeps the same content but positions it as conditional on "our analyses." |
| 10 | Discussion §6.4 | "For pipeline design: verify representations before relying on them for transfer or monitoring—downstream reward alone is insufficient evidence" | "Our results illustrate the value of direct representation verification. In our setting, downstream reward did not reflect the absence of intended latent structure." | The original reads like a directive to the community from a simulation-only study. The revision makes it an observation from the study's own evidence. |

**General pattern:** Replace declarative/directive framing ("X is Y," "practitioners should Z") with evidential framing ("in our setting, we find X," "our results suggest Y"). This is not hedging—it is honest scientific communication from a single simulation study.

---

## 7. Section-by-Section Revision Plan

### Title
- **Keep:** Focus on verification and auxiliary supervision
- **Cut:** The current rhetorical question that implies broad scope
- **Reframe:** Use one of the candidate titles from Section 2 (recommend Title 2)
- **Job in paper:** Signal negative result + verification study + simulation scope, attract workshop reviewers looking for careful empirical work

### Abstract
- **Keep:** Probe $R^2 \approx 0$, factorial decomposition, intervention null, LSTM comparison
- **Cut:** "Crucially," numbered contribution list, "For the sim2real community, this provides" directive
- **Reframe:** Lead with assumption being tested, not with the community it addresses. End with "these results suggest" not "this provides"
- **Job in paper:** 200-word elevator pitch that is technically precise, honestly scoped, and does not oversell

### Introduction
- **Keep:** DynaMITE description, factorial preview, probe null preview
- **Cut:** "publication_bias" citation framing (reads like virtue signaling in this context), "For any pipeline that relies on a learned dynamics estimator" (overreach)
- **Reframe:** Add scope-narrowing paragraph early. Rewrite contributions per Section 5. Present DynaMITE as experimental vehicle, not proposed method. Move "Scope and Non-Claims" content INTO the introduction rather than burying it at the end.
- **Job in paper:** Set up a focused question, scope the investigation honestly, preview the answer without overclaiming

### Related Work
- **Keep:** DR subsection, RMA/adaptive policies subsection, DynaMITE-vs-RMA differentiation
- **Cut:** The paragraph-length editorial about DynaMITE differing from RMA "in two respects"—this belongs in Method
- **Reframe:** The verification-in-RL subsection should note that direct probing of auxiliary-loss representations in locomotion is uncommon (not that it "has not been reported"—we don't know that). Keep short and factual.
- **Job in paper:** Position the work within prior art, identify the verification gap, do NOT editorialize

### Method
- **Keep:** Architecture figure, loss equations, baseline table, history buffer detail
- **Cut:** Nothing structural—this section is appropriately detailed
- **Reframe:** Opening sentence should be "To test whether per-factor auxiliary supervision produces the intended factor structure, we instantiate the design pattern as DynaMITE..." Position as experimental vehicle.
- **Job in paper:** Describe what was built and why, enough for replication, without method-contribution framing

### Evaluation Protocol
- **Keep:** Training details, metric definitions, statistical reporting, analysis tools list
- **Cut:** Nothing—this section is well-written
- **Reframe:** Minor. Add "our analyses are conditional on these readout families" as a sentence.
- **Job in paper:** Establish reproducibility and transparency. This section is currently a strength.

### Results
- **Keep:** ID comparison (1 table), probing null (Table + prose), intervention null (Table + prose), disentanglement (Table), factorial (3 tables + prose), combined-shift (Table + figure)
- **Cut:** Geometry table (tab:geometry)—move key numbers (effective rank ~5, MI 0.23 vs 0.03) into prose within the probing subsection. The table adds marginal info for its space cost. Operating regimes table (tab:regimes)—this is a discussion-section summary device that doesn't belong in Results and overclaims by offering "deployment" guidance. Push recovery and degradation tables (tab:recovery, tab:degradation)—these are supplementary. Keep them only if space permits; they are not part of the core 4 empirical blocks.
- **Reframe:** 
  - Probing section: "we find limited evidence" not "the answer is negative"
  - Factorial: "consistent but sub-significant" not "cleanly separates"
  - Combined shift: "reveals a failure mode in our evaluation" not "reveals a failure mode invisible to single-axis evaluation" (the latter implies the protocol is validated, which it isn't against hardware)
- **Job in paper:** Present evidence cleanly, let the reader draw conclusions, do not editorialize within results

### Discussion
- **Keep:** The three-part structure (supervision null, compression mechanism, compound perturbation)
- **Cut:** "operating regimes" table, "practical recommendations" subsection written as directives, "concrete warning" language
- **Reframe:** Per Section 6 above. Every claim should be scoped to "in our setting." Replace directives with observations. Add "why this matters for workshop readers" paragraph.
- **Job in paper:** Interpret results honestly, connect to the verification thesis, explicitly state what was NOT shown

### Limitations
- **Keep:** Simulation-only caveat, probe-family caveat, statistical power caveat ($n\!=\!10$, $p \approx 0.2$)
- **Cut:** Overly detailed items that read like a defense brief
- **Reframe:** Frame simulation-only as intentional scope, not a deficiency: "Our investigation is conducted entirely in simulation, which is the intended scope; we argue that simulation-side verification is a necessary step before transfer, not a substitute for it."
- **Job in paper:** Demonstrate self-awareness and scientific maturity. Prevent reviewer objections by acknowledging limits preemptively.

### Conclusion
- **Keep:** Summary of probe null, factorial result, scope statement
- **Cut:** Directive to "practitioners building adaptive policies for sim2real transfer"
- **Reframe:** Per Section 6 above. End with "verify, don't assume" as a suggestion, not a commandment. End with hardware deployment as natural next step.
- **Job in paper:** Crisp, 1-paragraph summary. No new claims. Close the paper with an honest scope statement.

### Tables triage

| Table | Verdict | Reason |
|-------|---------|--------|
| tab:arch | **Keep** | Baseline summary, compact, essential for replication |
| tab:id | **Keep** | ID comparison context, 1 table |
| tab:probe | **Keep** | Core negative result |
| tab:intervention | **Keep** | Core negative result |
| tab:disentangle | **Keep** | Reinforces probe null |
| tab:geometry | **Cut → prose** | Move effective rank and MI numbers into probing prose. Table adds 1 data point beyond what's already stated. |
| tab:ablation | **Keep** | Provides context for factorial; Aux-Only cell referenced |
| tab:factorial_id | **Keep** | Core mechanistic result |
| tab:factorial_ood | **Keep** | Core mechanistic result |
| tab:factorial_deg | **Keep** | Completes factorial story |
| tab:combined | **Keep** | Core stress test result |
| tab:degradation | **Conditional** | Keep only if space permits; not core |
| tab:recovery | **Conditional** | Keep only if space permits; not core |
| tab:regimes | **Cut** | Overclaims by offering operating guidance from simulation-only evidence |

Target: 10–11 tables (from current 14). This frees ~0.3 page for better prose.

---

## 8. Reviewer Objections and Responses

### Objection 1: "This is just a negative result—what's the contribution?"

**Response:** The contribution is threefold: (a) a controlled empirical test of a representation-learning assumption that is widely adopted but rarely verified in locomotion, (b) a reusable verification toolkit (probes, interventions, disentanglement metrics, factorial ablation) applicable to other adaptive policy architectures, and (c) evidence that bottleneck compression rather than explicit supervision is more associated with the robustness differences observed. Negative results that recalibrate design expectations are among the most valuable contributions a workshop can publish, provided the methodology is sound.

### Objection 2: "This is simulation-only—no real sim2real transfer experiment."

**Response:** We agree, and we are explicit about this scope. Our argument is that simulation-side verification is a *prerequisite* for transfer: if a representation fails to exhibit its intended structure under controlled simulation conditions, transferring it to hardware is premature. The combined-shift protocol is a proxy for compound dynamics mismatch, not a substitute for hardware validation, which we identify as the primary next step.

### Objection 3: "The bottleneck effect is not statistically significant ($p \approx 0.2$). How can you claim it 'drives' robustness?"

**Response:** Fair criticism. We should not—and in the revised version, we do not—claim that the bottleneck "drives" robustness. We state that the bottleneck shows a *consistent* advantage across ID and OOD regimes that does not reach conventional significance at $n\!=\!10$. The stronger claim is the *null* for auxiliary losses ($p = 0.73$, $p = 0.67$), which is well-powered. The revision frames the bottleneck finding as "consistent and suggestive, pending confirmation at higher $n$."

### Objection 4: "You test only one architecture (DynaMITE) on one platform (G1/Isaac Lab). How generalizable is this?"

**Response:** We do not claim generalizability beyond our setting. The paper is scoped to simulated humanoid locomotion with one instantiation of the auxiliary-supervision pattern. The value lies in (a) showing that the assumption *can* fail, even in a well-controlled setting, and (b) providing tools that other researchers can apply to their own architectures and platforms. Workshop papers are expected to have focused scope.

### Objection 5: "The paper has too many tables and not enough interpretation."

**Response:** Acknowledged. The revised version cuts 3 tables (geometry, regimes, and conditionally degradation/recovery) and redirects that space to interpretation and honest discussion of limitations. The remaining tables are each tied to a specific claim and support the core argument.

### Objection 6: "The probe family is too limited—maybe a better decoder would recover the factor structure."

**Response:** Correct, and we state this explicitly as a limitation. Our probes (Ridge regression and 1-hidden-layer MLP) are standard in the representation-probing literature but are not exhaustive. Nonlinear ICA or manifold-aware probes might recover structure invisible to our readouts. The non-decodability finding is conditional on our readout family. We note that this limitation strengthens rather than weakens the paper's main recommendation: representation claims should be verified with direct tools, not inferred from task reward.

---

## 9. Reviewer Objections and Responses

(Continued from above — this section was requested as #9 but the content is combined with #8 for coherence. Below is the Final Positioning Statement as the actual #10.)

---

## 10. Final Positioning Statement

### Cover-note positioning paragraph

> This paper presents a verification study testing whether a common representation-learning pattern—per-factor auxiliary dynamics supervision—produces the intended latent structure in simulated humanoid locomotion. Using probes, clamping interventions, disentanglement metrics, and a $2 \times 2$ factorial ablation on a Unitree G1 humanoid in Isaac Lab, we find limited evidence that the intended factorization emerges: probe $R^2 \approx 0$, intervention effects are negligible, and disentanglement scores are near zero. The factorial ablation provides consistent evidence that a $\tanh$ information bottleneck is more associated with observed robustness differences than the auxiliary supervision itself, though this effect is sub-significant at our current sample size.
>
> We submit this as a negative-result-plus-verification paper. The verification toolkit—probes, interventions, factorial ablation, compound stress test—is reusable and relevant to any research group building adaptive policies for sim2real transfer. The negative result challenges a design assumption adopted across the RMA family of methods without, to our knowledge, direct verification. We believe this combination of careful methodology, honest reporting, and scoped claims fits the workshop's emphasis on evaluation protocols and empirical rigor for sim2real transfer.

### One-sentence pitch

> In simulated humanoid locomotion, per-factor auxiliary dynamics supervision does not produce verifiable latent factor structure under our diagnostic tools, and a $\tanh$ bottleneck appears more associated with robustness than the explicit supervision signal.

### Workshop fit argument

This paper fits the ICRA 2026 Workshop on Generative Digital Twins for three specific reasons:

1. **Evaluation protocol contribution.** The workshop solicits "benchmarks, datasets, and evaluation protocols." Our verification toolkit—probes, interventions, disentanglement metrics, factorial ablation, compound stress test—is a portable evaluation protocol for assessing representation quality before transfer.

2. **Negative result relevant to adaptive digital twins.** If learned dynamics estimators inside simulation-trained policies do not encode what they are designed to encode, then any adaptive digital-twin pipeline relying on those estimators needs to verify representation quality. Our paper demonstrates a concrete case where this verification fails.

3. **Scoped, honest, simulation-side work.** The paper does not claim hardware transfer, does not claim a new method, and does not claim universality. It is a careful, technically serious study of one representation-learning pattern in one well-controlled setting—the kind of work that workshops are designed to accommodate and that main conferences often filter out.

---

## Appendix: Quick-Reference Checklist for Revision

- [ ] Replace title with candidate Title 2 (or Title 1)
- [ ] Replace abstract with revised version from Section 3
- [ ] Rewrite intro opening paragraph per Section 4
- [ ] Add scope-narrowing paragraph to introduction
- [ ] Rewrite contributions per Section 4
- [ ] Move "Scope and Non-Claims" content into introduction; delete standalone subsection
- [ ] Reframe DynaMITE as experimental vehicle in Method §3 opening
- [ ] Cut tab:geometry → move numbers to prose
- [ ] Cut tab:regimes entirely
- [ ] Evaluate tab:degradation and tab:recovery for space; cut if tight
- [ ] Apply all 10 tone corrections from Section 6
- [ ] Rewrite Discussion headers per Section 5
- [ ] Rewrite Conclusion per Section 5
- [ ] Change "the central finding is unambiguous" → evidential framing
- [ ] Change "does not produce what it promises" → "does not exhibit the intended structure under our analyses"
- [ ] Change "concrete warning" → "our results suggest"
- [ ] Change "the practical recommendation is direct" → "in our setting, the bottleneck contributed more than the auxiliary losses"
- [ ] Ensure every claim references "in our setting" or "under our analyses"
- [ ] Remove "Crucially" from abstract/body
- [ ] Compile and verify page count
