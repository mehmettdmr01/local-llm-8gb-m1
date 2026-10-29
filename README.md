# Local LLMs on an 8 GB MacBook Air (M1): the measurements

Video: (link coming)

Eight local models, run with [Ollama](https://ollama.com) on a 2020 MacBook Air (M1, 8 GB RAM, fanless,
macOS 26), each given the same 14-question exam and timed. Everything here was produced by `bench.py`;
`results.json` holds the numbers and `raw/` every answer (and every thinking trace) the models gave.

## Setup

- **Ollama 0.40.11**, every model as `ollama pull` delivers it, **default settings**: Ollama picked a
  4,096-token context from the GPU memory it detected (`server-log-excerpts.txt`), thinking models think.
  Options sent: `seed: 42` only.
- **Real-world machine state:** VS Code and Chrome stayed open for the whole run; free memory when each
  model started was 54–77 %, and fell to 10–45 % while they ran (`free_pct_*` in `results.json`).
  Models ran one after another (order as in the table), with 60 s between them.
- **Speed** = median of 3 warm runs generating 256 tokens (`gen_tps`), plus one run after a ~2,000-token
  prompt (`long_*`). **GPU share** = `size_vram / size` from `/api/ps`; layer split from the server log.
- **Exam** (`bench.py`): 4 Python functions run against hidden `assert` tests, 4 SQL queries run against a
  seeded SQLite database and compared with a reference query, 6 short questions (arithmetic, a train
  timetable, a riddle, the r's in "strawberry", a JSON object, a reply of exactly three words). The four
  number questions must end with a line `Answer: <number>`, and format counts. Each answer gets
  **4,096 tokens or 300 s**, whichever comes first. **Each question ran once.**

## Results

| model | params, quant | in memory (GB) | layers on GPU | tokens/s | exam score | exam time (s) | timeouts |
|---|---|---|---|---|---|---|---|
| gemma3:1b | 999.89M Q4_K_M | 0.88 | 27/27 | 54.1 | 7/14 | 26 | 0 |
| llama3.2:1b | 1.2B Q8_0 | 1.52 | 17/17 | 44.5 | 4/14 | 26 | 0 |
| qwen3:1.7b | 2.0B Q4_K_M | 1.45 | 29/29 | 45.4 | 11/14 | 528 | 0 |
| llama3.2:3b | 3.2B Q4_K_M | 2.56 | 29/29 | 27.8 | 7/14 | 42 | 0 |
| qwen3:4b | 4.0B Q4_K_M | 3.18 | 37/37 | 20.9 | 11/14 | 2128 | 0 |
| gemma3:4b | 3.9B Q4_K_M | 2.43 | 34/35 | 20.9 | 13/14 | 105 | 0 |
| mistral:7b | 7.2B Q4_K_M | 5.15 | 31/33 | 12.3 | 7/14 | 225 | 0 |
| qwen3:8b | 8.2B Q4_K_M | 6.04 | 30/37 | 6.4 | 11/14 | 2355 | 3 |

GB are decimal (10⁹ bytes), as Ollama reports them; the laptop's "8 GB" is 8 GiB = 8.59 GB.
Not measured: phi4-mini, deepseek-r1:8b, gemma3:12b, qwen3:14b (the network dropped to ~1 KB/s during the run; phi4-mini first hit a
bug in this script, since fixed).

## The bandwidth rule

Generating one token reads (almost) the whole model from memory once, and the M1's memory bandwidth is
68.25 GB/s, so `68.25 / size in memory` is a ceiling for tokens/s:

| model | in memory (GB) | predicted tok/s | measured tok/s | measured / predicted |
|---|---|---|---|---|
| gemma3:1b | 0.88 | 77.6 | 54.1 | 0.70 |
| llama3.2:1b | 1.52 | 44.9 | 44.5 | 0.99 |
| qwen3:1.7b | 1.45 | 47.1 | 45.4 | 0.96 |
| llama3.2:3b | 2.56 | 26.7 | 27.8 | 1.04 |
| qwen3:4b | 3.18 | 21.5 | 20.9 | 0.97 |
| gemma3:4b | 2.43 | 28.1 | 20.9 | 0.74 |
| mistral:7b | 5.15 | 13.3 | 12.3 | 0.93 |
| qwen3:8b | 6.04 | 11.3 | 6.4 | 0.57 |

Five of the eight are within 7 % of the prediction. Both Gemma 3 models come in about a quarter under it
(not explained here). Qwen3 8B is at 57 %: at 6.04 GB it does not fit under the GPU working-set limit Metal
reports on this machine (`recommendedMaxWorkingSetSize = 5726.63 MB`; Ollama keeps another 512 MiB spare),
so 30 of its 37 layers ran on the GPU and 7 on the CPU. It also ran last, with 10 % of memory free, so part
of the slowdown may be memory pressure rather than the split alone.

## Limits

14 questions, each run once, is a small exam: a one-point difference is noise. These are Ollama defaults on
one laptop with apps open; MLX, a larger context, KV-cache quantisation or a raised GPU limit
(`sudo sysctl iogpu.wired_limit_mb=…`) were not tested. Thermal state (`pmset -g therm`, once a minute,
`thermal.log`) never reported a warning; logging started during the last model.

## Run it

```
brew install ollama && ollama serve &
python3 bench.py                 # all models; skips the ones already in results.json
python3 bench.py --only gemma3:4b --keep
```
