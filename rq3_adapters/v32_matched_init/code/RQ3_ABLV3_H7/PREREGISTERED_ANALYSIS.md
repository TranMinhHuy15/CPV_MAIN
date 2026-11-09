# RQ3 matched ablation V3 — analysis plan, fixed before this new training round

## Scope
Exploratory follow-up after inspecting V2 test results. This is not a previously unseen confirmatory test.
All six adapters are trained anew on each frozen base; no historical A7/G3 predictions enter main contrasts.
Two bases: H7 CNN+LSTM2048(history7), RiskProp V6 CNN+fc(no base RNN). Same seeds42/43/44; V6 seed44 rerun.
Output roots MUST be new. V2 results and input weights remain read-only.

## Matrix and recipe
B0 frozen base; C7 current token repeated7; R7 realH7/last state; A7 realH7/attention+age;
A26 realH26/attention+age; U3 realH7/H13/H26 uniformmean; G3 same windows learnedsoftmaxfusion.
Shared window weights, reset LSTM per window, no future token; CNN snippet context remains in C7.
A26 adapter512 is not the old longH26 LSTM2048 baseline.
AdamW lr2e-4,wd1e-4,width512,dropout.1,bound1,microbatch2*accumulation8,50epochs,
milestones20/40 gamma.1,clip40,FP32. No new hyperparameter search. Earliest max validation fullmAUC perrun.
18runs/base=6arms*3seeds,900epochs/base. No test-based model/epoch/seed exclusions.

## Matched initialization and RNG
Final adapters use canonicalG3 CPU initialization at the runseed, in a forked CPU RNG.
Remove unneeded Q/K/decay/fusion without reinitializing common layers. All shared tensors match across six arms.
Final-adapter creation does not advance CPU/CUDA training RNG. The pinned parent constructor still creates
a compatibilityM1 adapter that is replaced; its advanced RNG is not used to choose final adapter weights.
No claim that V3 matches the historical integrated M1 initialization. Sampler/dropout use the runseed;
common dropout shape and zero LSTM internal dropout keep stochastic consumption comparable across arms.

## Gates and preservation
Pinned original source, baselinecheckpoint hashes, cachemembership/hash, frozenweights bitcheck,
24microbatch/3update smoke; native8video parity perpath; no test scoring before all18runs complete.
V6 validation max|dp|<=1e-2 AND mean<=1e-3 inherited/amended threshold, not exact numerical identity.
PlanMD SHA and runtimecode hashes bind before training and are rechecked by everyworker/analysis.
Default compact_completed_trials=false preserves full generated checkpoints/resume snapshots.
Explicit true only compacts completed owned trials after export/frozenchecks; always retains log/history/adapter.

## Diagnostics and pilot
Every validation epoch: gate/raw/tanhraw/correction moments, correction fraction|delta|>.95,
attentionentropy bywindow, G3omega moments; first2validationbatches store gate/raw/correction/omega arrays.
Diagnostics fsync before the original priority90 epoch commit; resume trims rows beyond the committed epoch.
These are passive logs, never a scheduler/selection criterion. Correlated tokens are descriptive, not independentn.
Capture both saturation and variance; correction alone cannot identify gate/raw separately.
Pilot interrupts after committedepoch2 then resumes to3, checking unique history+snapshot commit.
This is not proof of uninterrupted bit-exact CUDA equivalence; CPU fixtures check RNG isolation, diagnostic trimming and pilot orchestration; actual optimizer recovery needs the GPU pilot.

## Statistical analysis
1344testvideos, grouped AP at.5/1.0/1.5s; mAP equalgroup mean, mAUC equalgroup fullROC-AUC mean.
5000pairedvideo bootstrap, RNG20261106, strata(group*label), identicalindices across allarms/seeds/bases.
Perseed delta and mean of3 fixedseeds; CI95percentile on mean paired draws. This is not a seedpopulationCI.
Tail p=2min(P(draw<=0),P(draw>=0)), capped1; approximate, MonteCarlozero is not exactp0.
Mainfamily perbase: A7-C7, A7-R7, G3-U3, U3-A26, G3-A26, eachmAP+AP1.5 =>Holm10tests/base.
A7-R7 differs524289parameters and removesage too; not pureattention. Multiwindow vsA26 is not compute-matched.
Secondary: eacharm-B0; mAUC exploratory. No historicalsidebranch inmain. Crossbase
I=(X-B0)_H7-(X-B0)_V6, same indices, exploratoryuncorrected. No causalceilingclaim from differing effects.
Effectpresent requires adjustedp<.05,CI excludes0 and samepositive(ornegative) direction all3seeds.
Otherwise trend/no detectableeffect, not equality/noninferiority. Report all outcomes, not only winners.

## Follow-up boundaries
No changes to loss/gatebound in response to V2 seed44. Do not drop that seed or pick epoch from test.
Report parameter+MAC+measured cachefit time separately; no online latency claim from cachefit.
Native online perarm latency/OOD/unseenconfirmation and novelty review remain future work.
