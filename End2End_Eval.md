# End-to-end evaluation of the reading agent

Evaluated 2026-11-11 on ten wall photos, from the image to the final English reading.

**In short:** with the true signs in hand the agent names the right king on 10 of 10 photos and passes 8. On the full pipeline, where the signs come from the detector, it names every king on 8 of 10 photos and passes 3, with 5 partial and 2 fail. No reading invented a king. The gap between the two is the cost of vision, and the eval pointed at one concrete bug, which was fixed and re-measured.

## What is tested

The whole chain, as a visitor uses it:

```
photo + layout + direction
  -> YOLO detector -> reading order -> cartouche matcher
  -> Claude Haiku with corpus tools -> router -> Claude Sonnet review
  -> transliteration + English translation
```

Layout (rows or columns) and direction (rtl or ltr) are given per photo, because the pipeline cannot detect them. The optional archaeology form (period, site, reign) is sent as `unknown`, so it cannot leak the answer.

## Test set and ground truth

Ten photos in `chosen/`: nine royal inscriptions from Dynasty XVIII to XXVI and one Pyramid Texts wall. Kings on them: Thutmose III (4 photos), Thutmose IV, Amenhotep III, Horemheb, Ramesses II, Ramesses III, Amasis and Unas.

The ground truth is `reports/agent_golden_chosen.json`. For each photo it holds:
- the Gardiner signs per line, in reading order
- each cartouche with its king and accepted spellings of the name
- the transliteration and an English translation
- a short list of key facts with accepted wordings
- the true signs that are not among the detector's 150 classes

How it was made:
1. Claude drafted every record from the photo **before any agent run**, so the agent's output could not shape the truth.
2. Readings were cross-checked against the corpus tools and the royal titularies.
3. The author, who labelled the detector's training set by hand, reviewed and corrected all ten records.

Seven of the ten photos show text the ground truth does not transcribe (weathered rows, columns cut by the frame). Those records are flagged `truth_incomplete`.

## Two conditions

| Condition | Signs the agent receives | What it measures |
|---|---|---|
| **Pipeline** | the detector's, as in production | the product |
| **Oracle** | the ground-truth signs and cartouche names | the agent alone, with perfect vision |

The difference between the two is the share of the error that belongs to vision.

## How a reading is scored

Four checks, all computed by code against the reviewed truth. No LLM judge.

| Check | Passes when |
|---|---|
| King | every king on the wall is named, by any accepted spelling |
| No invented king | no king is named who is not on the wall |
| Transliteration | similarity to the truth is at least 0.70, after normalising notation |
| Meaning | at least half of the key facts appear in the translation |

- **Pass:** all four.
- **Partial:** kings right and none invented, but transliteration or meaning falls short.
- **Fail:** a king missed or invented, or the read errored.

Transliteration is compared after mapping Unicode to Manuel de Codage and treating *y/j* and *z/s* as the same letters. For the seven incomplete photos the score is coverage: how much of the truth the reading contains. Text the agent writes beyond the truth is not counted against it there. The strict two-way score is reported next to it.

The thresholds (0.70 and one half) were fixed before the first run and were not changed afterwards.

## Results

### Summary

| | Pass | Partial | Fail | All kings named | Transliteration (strict) | Key facts |
|---|---|---|---|---|---|---|
| Pipeline, baseline | 4 | 4 | 2 | 8 of 10 | 0.69 (0.55) | 0.62 |
| Pipeline, after the label fix | 3 | 5 | 2 | 8 of 10 | 0.73 (0.62) | 0.64 |
| Oracle (true signs) | 8 | 2 | 0 | 10 of 10 | 0.83 (0.71) | 0.84 |

Invented kings: 0 in all 30 reads.

The label fix raised the strict transliteration score from 0.55 to 0.62, but the verdict count did not improve: one photo moved up to pass and two moved down to partial. Those two sit at 0.69 and 0.68, just under the 0.70 line, and with one trial per photo that is run-to-run variation, not a measured regression.

### Per photo (pipeline, after the label fix)

| Photo | King(s) | Verdict | Transliteration | Key facts | Oracle verdict |
|---|---|---|---|---|---|
| `bnm,..jpeg` | Ramesses III | pass | 0.94 | 1.00 | partial |
| `color_signs.jpg` | Horemheb | pass | 0.91 | 0.80 | pass |
| `Hatshepsut_temple_wall.jpg` | Thutmose III | pass | 0.83 | 0.71 | pass |
| `grand_glyphs.jpeg` | Amasis | partial | 0.69 | 0.50 | pass |
| `ramsses_the_II_wall.webp` | Ramesses II | partial | 0.68 | 0.67 | pass |
| `asdfghj.jpeg` | Thutmose III | partial | 0.62 | 0.60 | pass |
| `bnnm45.jpeg` | Thutmose III | partial | 0.60 | 0.50 | pass |
| `Pyramid_of_Unas.jpg` | Unas | partial | 0.48 | 0.20 | partial |
| `dfghj.webp` | Thutmose III and IV | fail (1 of 2 kings) | 0.86 | 0.67 | pass |
| `tyuiug.jpg` | Amenhotep III | fail (king not named) | 0.72 | 0.75 | pass |

### Vision alone

- About 78% of the true signs are detected (order ignored).
- 17 of 19 cartouches are found; 15 are matched to the right king, 2 are refused.
- Fourteen distinct true signs across the set are outside the detector's 150 classes (for example `C10` Maat, `G26` the ibis on a standard, `U21` the *stp* adze), so no detector output could be perfect.

### Cost and time

| | Per read | Notes |
|---|---|---|
| Cost | about $0.03 | $0.004 when Haiku reads alone |
| Time | about 50 s | 28 s when Haiku reads alone |
| Escalated to Sonnet | 27 of 30 reads | 25 because Haiku asked for a second opinion |

The full baseline (20 reads) cost $0.57; the 10-read rerun cost $0.35.

## What the eval found

1. **Wrong cartouche label (fixed).** The name lexicon had one transliteration per king, so a throne-name cartouche was sent to the agent labelled with the birth name. The agent then wrote the wrong title, or both names. Each spelling now carries its own name and kind (`vision/data/royal_names.json`, `vision/cartouche_matcher.py`).
   - Ramesses III: 0.61 → 0.94, partial → pass, read by Haiku alone for $0.004 instead of $0.039.
   - Horemheb: 0.67 → 0.88 strict.
2. **Vision decides the two fails.** On `tyuiug.jpg` the cartouche is not detected, so Amenhotep III is never named. On `dfghj.webp` one cartouche's name match is refused. With the true signs both pass.
3. **The agent fills in stock formulas.** On the obelisk (`bnnm45.jpeg`) it skips *jr.n=f m mnw* ("he made it as his monument") and writes "given life forever", which is not on the stone. Not fixed yet.
4. **Almost every read escalates.** On short royal texts Haiku asks for a second opinion nearly every time, so the router saves little there.
5. **Long, weathered walls stay hard** in both conditions. On the Unas wall the agent recognises the king and "he lives", but not the "he is not dead" half of the litany.

## Limits

- **Ten photos, one trial each.** A single verdict is 10 points, and photos near a threshold can flip between runs. This is a regression check, not a benchmark.
- **The ground truth was drafted by a Claude model** and the agent under test is Claude too. The blind draft, the corpus cross-checks and the human review are the mitigations.
- **Seven truths are incomplete.** On those photos, text the agent invents beyond the transcribed part is not penalised. Both the coverage and the strict scores are saved per photo.
- **Three scorer faults were found and fixed after the first run**, with the earlier scores kept:
  - names in a citation note were counted as invented kings
  - *mry/mrj* and *zꜣ/sꜣ* were treated as different
  - the similarity score punished readings longer than an incomplete truth

  They changed 10 verdicts on the baseline: 9 up and 1 down. The thresholds were not touched.
- **The oracle condition was not rerun** after the label fix, because it builds its labels from the ground truth and does not use the matcher.

## Reproduce

Start the server with the daily read limit off (the $3 spend guard stays on):

```bash
TRANSLIT_RUNS_PER_DAY=0 LLM_REQUESTS_PER_WINDOW=0 \
  thoth/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
```

Run and score:

```bash
thoth/bin/python scripts/agent_eval_run.py --run-id my_run
thoth/bin/python scripts/agent_eval_score.py reports/agent_eval/my_run \
  --baseline reports/agent_eval/baseline_2026-11-11/summary.json
```

The scorer makes no LLM calls, so it is free to rerun. `--baseline` lists the photos whose verdict changed.

| File | Role |
|---|---|
| `reports/agent_golden_chosen.json` | ground truth |
| `scripts/agent_eval_run.py` | posts each photo to the server and saves every raw response |
| `scripts/agent_eval_score.py`, `scripts/agent_eval_metrics.py` | the four checks, the verdict, the diagnostics |
| `reports/agent_eval/baseline_2026-11-11/` | baseline run, both conditions |
| `reports/agent_eval/label_fix/` | pipeline rerun after the label fix |

No evaluation framework is used. The agent already returns its trace, tool calls, evidence and cost with every reading, and the references are checkable, so plain scripts and a frozen test set are enough.
