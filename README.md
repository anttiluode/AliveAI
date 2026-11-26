# AliveAI

**Frozen knowledge, changing internal activity, experience-dependent reading.**

AliveAI is an exploratory recurrent associative machine built on the exported hidden-state bank from [SweepSol](https://github.com/anttiluode/SweepSol). It inherits the fluid-regime reader from [CombinatoryPlaySol](https://github.com/anttiluode/CombinatoryPlaySol) and the ideas of [Sweep](https://github.com/anttiluode/Sweep). This is a computational instrument, not a claim that AI is alive, conscious, or biologically faithful.

## Play

Open `index.html` locally, or serve the repository with GitHub Pages (Settings → Pages → Deploy from a branch → main / root). On an HTTP(S) host the page tries to load **bank.json** beside itself automatically. Until the bank is uploaded it uses an explicitly labeled synthetic toy. Local file mode uses the bank file picker. No dependencies, GPU or model download for the browser.

```bat
git clone https://github.com/anttiluode/AliveAI
cd AliveAI
start index.html
```

Use the existing SweepSol bank.json (words, vectors, meta). Put it in the repository root to publish the demo. This is a compressed sample of frozen model activations, not the model itself, a J-lens, or full residual states. Check provenance and applicable model license before redistribution. The application does not upload memories or traces. Do not commit personal memory exports. The bank is intentionally not gitignored.

### Try the learning loop

1. Start with bird,shadow,winter and Step or Run. Defaults are the user-discovered fluid regime: fatigue .44, burn .12, recovery .015, tether .18, temperature .09, k=12, feedback .48, coverage .16, spread .55.
2. Nominate one visible concept. The reader pauses. Give Useful +1, Neutral 0, or Unhelpful -1, according to a real task or your judgment.
3. Step again. The contextual value field now affects which concepts can enter the workspace. Positive feedback also restores some active energy and raises patience. Negative prediction error reduces patience.
4. New episode resets fast state while retaining learned memory. Toggle Recall learned memory off and repeat to compare. The same cue context and tag retrieve earlier records. A different tag may suppress them.
5. Inject new cues without wiping the state or fatigue field. Apply tag changes context without resetting the state. Export/import memory explicitly to carry learning between sessions (same bank only).

Without external feedback the system explores, but it does not invent success, change learned memory, or autonomously act. There is no external model call in the browser.

## Architecture

- **Frozen geometry**: normalized word-state bank, optional grain.
- **Fast process**: continuous state, 12 simultaneous weighted observations, cue tether and feedback into the state. No token emission required.
- **Recoverable fatigue**: selected concepts burn energy; correlated neighbors also tire. Rest restores capacity. Fatigue acts before top-k selection.
- **Value**: external reward r in [-1,1], prediction error delta = r - V, and V ← V + .2 delta. V is a simple episode-local scalar expectation, not a neural reward model.
- **Patience**: P ← clip(P + .15 delta, .1,.95), modulating burn by clip(1.5-P,.35,1.5). Positive reward replenishes .2*r energy of the nominated package. Novelty is measured, not treated as reward.
- **External plasticity**: bounded memory (512 recent records) stores cue IDs, context tag, continuous query state, package IDs, nominated action and observed signed reward. Retrieval matches current cues, tag and state, biases the next candidate scores before selection. Negative memories inhibit; positive ones attract. No weight updates or backpropagation. Records are local, not claimed to model human memory.
- **Slow context**: current tag gates memory; sensory cue injection blends .4 new cue state with .6 old state, preserving fatigue, memory, temperature and ongoing dynamics.

The memory field is deliberately small and explicit; stored action bias dominates weaker package bias. Reward applies only to a nominated observed word; this is not temporal credit assignment across an entire trajectory. Patience and value are engineering abstractions, not simulations of serotonin or dopamine. The agent has no self-generated objectives.

## CPU experiments (Node.js)

```bat
node run.js --self-test
node run.js --bank demo --budget 10 --beats 20
node run.js --bank bank.json --cues bird,shadow,winter --ticks 444 --out alive-results-free.json
node run.js --bank bank.json --tasks tasks.json --budget 40 --beats 80 --epochs 2
```

The included task set is small and hand-authored: 6 training cue sets, 10 held-out cue sets, some sharing answer categories deliberately, plus new categories. It is an exploratory common-association task, not a validated RAT benchmark. All listed words are present in the user's original Qwen bank word list. The evaluator alone knows the accepted answers. Each recurrent arm nominates at most one not-yet-evaluated visible candidate per beat; correct gets +1, wrong gets -.15. Train memory only on training tasks, then freeze it for seen and held-out evaluations. Same maximum oracle-call budget for each arm; repeated queries forbidden, and recurrent arms have a beat cap. Static ranks all candidates once and evaluates in order, so computation is NOT matched. Report success, calls and beats.

Arms: static cue blend, fatigue reader with no regulation or memory, regulated reader with no memory, regulated reader with trained memory. Held-out success is the primary transfer measure; seen-task gains alone may be memorization. Correct-answer words may appear across train/test, so this is not answer-disjoint generalization. A negative result is a valid result. No claim of improvement.

Optional full frozen Qwen comparison (local GPU, torch/transformers, one greedy generated answer per question):

```bat
python3.13 compare_qwen.py --device cuda --local-files-only
```

Native Qwen does not receive the oracle or external memories, and has one generated answer, so it is a descriptive reference, **not** a matched-budget comparison. This is not yet a closed-loop integration with the native transformer. The cached reader is the working motor.

## Receipts and limitations

13 synthetic test groups pass, including upstream checks, positive and negative reward, memory serialization, context gating, actual pre-selection bias, cue injection, frozen bank and frozen evaluation memory. In the tiny hand-designed toy (2 training and 3 test tasks, budget 10, beats 20, epochs 2), static got 3/3 held-out; plain, regulated and learned each got 2/3. Learned averaged 4.67 calls vs regulated 5.33; seen tasks 2/2 vs 1/2. These are only synthetic smoke results, not evidence of useful transfer. Browser and CLI JavaScript parsed; full browser end-to-end, actual Qwen bank experiments and Python reference have not been executed here.

The main hypothesis is whether a persistent regulatory reader can learn useful ways to exploit a frozen representation through experience without changing the original weights. The first priority is honest comparison against static retrieval; no further architectural complexity is justified by toy wiring checks alone.

Upstream reader copied from CombinatoryPlaySol; its exact source blob is noted at the top of play.js. Existing repository license is retained.
