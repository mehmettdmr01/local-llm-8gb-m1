#!/usr/bin/env python3
"""Twelve local models on one 8 GB Apple M1, measured with Ollama.

For each model: pull it, measure load time, generation speed (3 warm runs, median) on an empty
context and once after a ~2,000-token prompt, where it runs (GPU / CPU split from /api/ps), memory
(swap and free memory before/during/after), then a 14-question exam graded automatically: 4 Python
tasks run against hidden tests, 4 SQL queries run on a fixed SQLite database and compared with a
reference query, 6 exact-answer questions (final line "Answer: <n>"). Every request uses the
model's own defaults, exactly as `ollama run` would (thinking models think, each model's own
temperature and context size); only the seed is fixed (42) so runs repeat. Per-question cap:
4,096 tokens and 300 s; a timeout counts as a fail and is reported as such. Raw answers go to
raw/<key>.json, the summary to results.json. The model is removed after measuring, unless --keep.
Methodology reviewed by Gemini before the run (../elestiri-gemini-yontem.md).

  python3 bench.py                 # all models not yet in results.json
  python3 bench.py --only qwen3:4b # one model
"""
import argparse, json, re, subprocess, sys, tempfile, textwrap, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "http://127.0.0.1:11434"
MODELS = ["gemma3:1b", "llama3.2:1b", "qwen3:1.7b", "llama3.2:3b", "phi4-mini", "qwen3:4b",
          "gemma3:4b", "mistral:7b", "qwen3:8b", "deepseek-r1:8b", "gemma3:12b", "qwen3:14b"]
OPTS = {"seed": 42}  # everything else: the model's own defaults, as `ollama run`
SPEED_PROMPT = "Explain how a hash map works, step by step, for a junior developer."

def post(path, body, timeout=900):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def get(path):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.load(r)

def swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))

def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else None

def runner_rss_mb():
    out = subprocess.run(["ps", "-axo", "rss,command"], capture_output=True, text=True).stdout
    return round(sum(int(l.split()[0]) for l in out.splitlines()[1:] if "ollama" in l and "runner" in l) / 1024)

# --- the exam --------------------------------------------------------------------------------
CODE_TASKS = {
    "parse_duration": ("Write a Python function parse_duration(s) that converts strings like '1h30m', '45s', '2h5s' or "
                       "'3m' into a total number of seconds (int). Return only the code in one ```python block.",
                       "assert parse_duration('1h30m')==5400\nassert parse_duration('45s')==45\nassert parse_duration('2h5s')==7205\n"
                       "assert parse_duration('3m')==180\nassert parse_duration('1h1m1s')==3661\n"),
    "merge_intervals": ("Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns "
                        "the merged, sorted list of overlapping intervals. Return only the code in one ```python block.",
                        "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\n"
                        "assert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([[5,7],[1,2]])==[[1,2],[5,7]]\n"
                        "assert merge_intervals([])==[]\n"),
    "is_valid_ipv4": ("Write a Python function is_valid_ipv4(s) that returns True only for valid dotted IPv4 addresses "
                      "(four decimal numbers 0-255, no leading zeros, no extra characters). Return only the code in one ```python block.",
                      "assert is_valid_ipv4('192.168.1.1')\nassert is_valid_ipv4('0.0.0.0')\nassert not is_valid_ipv4('256.1.1.1')\n"
                      "assert not is_valid_ipv4('1.2.3')\nassert not is_valid_ipv4('01.2.3.4')\nassert not is_valid_ipv4('1.2.3.4 ')\n"
                      "assert not is_valid_ipv4('a.b.c.d')\n"),
    "roman_to_int": ("Write a Python function roman_to_int(s) that converts a Roman numeral string to an integer. "
                     "Return only the code in one ```python block.",
                     "assert roman_to_int('III')==3\nassert roman_to_int('LVIII')==58\nassert roman_to_int('MCMXCIV')==1994\n"
                     "assert roman_to_int('XLII')==42\nassert roman_to_int('CDXLIV')==444\n"),
}
FINAL = " You may think it through, but end your reply with a final line in exactly this form: Answer: <number>"
EXACT_TASKS = {
    "multiply": ("What is 487 multiplied by 36?" + FINAL, lambda a: final_answer(a) == 17532),
    "train": ("A train leaves at 14:35 and arrives at 18:10 the same day. How many minutes is the trip?" + FINAL,
              lambda a: final_answer(a) == 215),
    "sisters": ("Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?" + FINAL,
                lambda a: final_answer(a) == 1),
    "strawberry": ("How many times does the letter r appear in the word strawberry?" + FINAL,
                   lambda a: final_answer(a) == 3),
    "json": ("Return a JSON object with the keys name (string), age (integer) and languages (list of strings) for this sentence: "
             "'Ada is 36 and writes Python and C.' Return only the JSON.", lambda a: json_ok(a)),
    "three_words": ("Reply with exactly three words, all in lowercase, and nothing else.", lambda a: three_words(a)),
}

SQL_SCHEMA = ("Tables (SQLite): customers(id INTEGER PRIMARY KEY, name TEXT, region TEXT); "
              "orders(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), order_date TEXT 'YYYY-MM-DD', amount REAL). ")
SQL_TAIL = " Return only the SQL query in one ```sql block."
SQL_TASKS = {
    "sql_region_totals": ("Write a query that returns each region and the total order amount in that region (columns region, total), "
                          "highest total first.", "SELECT c.region, SUM(o.amount) FROM orders o JOIN customers c ON c.id=o.customer_id "
                          "GROUP BY c.region ORDER BY 2 DESC"),
    "sql_top2": ("Write a query that returns the names of the 2 customers with the largest total order amount, largest first.",
                 "SELECT c.name FROM orders o JOIN customers c ON c.id=o.customer_id GROUP BY c.id ORDER BY SUM(o.amount) DESC LIMIT 2"),
    "sql_busiest_month": ("Write a query that returns the month (as 'YYYY-MM') with the most orders, and the number of orders "
                          "in it (two columns, one row).",
                          "SELECT substr(order_date,1,7), COUNT(*) FROM orders GROUP BY 1 ORDER BY 2 DESC LIMIT 1"),
    "sql_avg_frequent": ("Write a query that returns, for every customer with more than 2 orders, their name and their average "
                         "order amount rounded to 2 decimals, ordered by name.",
                         "SELECT c.name, ROUND(AVG(o.amount),2) FROM orders o JOIN customers c ON c.id=o.customer_id "
                         "GROUP BY c.id HAVING COUNT(*)>2 ORDER BY c.name"),
}

def sql_db():
    import random, sqlite3
    rnd = random.Random(7)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, region TEXT)")
    db.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, amount REAL)")
    names = ["Ada", "Linus", "Grace", "Ken", "Barbara", "Dennis", "Margaret", "Guido"]
    regions = ["North", "South", "East", "West"]
    for i, n in enumerate(names, 1):
        db.execute("INSERT INTO customers VALUES (?,?,?)", (i, n, regions[i % 4]))
    weights = [6, 1, 4, 2, 5, 1, 3, 2]  # uneven, so the answers are unique
    oid = 1
    for cid, w in enumerate(weights, 1):
        for _ in range(w):
            m = rnd.choice(["2026-01", "2026-02", "2026-03", "2026-03", "2026-04"])
            db.execute("INSERT INTO orders VALUES (?,?,?,?)", (oid, cid, f"{m}-{rnd.randint(1, 28):02d}", round(rnd.uniform(10, 400), 2)))
            oid += 1
    return db

def sql_ok(answer, reference):
    a = strip_think(answer)
    blocks = re.findall(r"```(?:sql|sqlite)?\n(.*?)```", a, re.S | re.I)
    query = (blocks[-1] if blocks else a).strip().rstrip(";")
    db = sql_db()
    try:
        got = [tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in db.execute(query).fetchall()]
    except Exception as e:
        return False, str(e)[:200]
    want = [tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in db.execute(reference).fetchall()]
    return got == want, "" if got == want else f"got {got[:3]}"

def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

def final_answer(a):
    """The number on the last 'Answer: n' line; if the model ignored the format but replied with
    nothing but a number, that number. Anything else is a fail (format is part of the task)."""
    t = strip_think(a)
    m = re.findall(r"answer\W{0,4}[:=]\W{0,4}(-?\d[\d,]*)", t, re.I)
    if m:
        return int(m[-1].replace(",", ""))
    bare = re.fullmatch(r"\W*(-?\d[\d,]*)\W*", t)
    return int(bare.group(1).replace(",", "")) if bare else None

def json_ok(a):
    a = strip_think(a)
    m = re.search(r"\{.*\}", a, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except Exception:
        return False
    return (isinstance(d, dict) and d.get("name") == "Ada" and d.get("age") == 36 and isinstance(d.get("age"), int)
            and isinstance(d.get("languages"), list) and sorted(x.lower() for x in d["languages"]) == ["c", "python"])

def three_words(a):
    t = strip_think(a)
    words = re.findall(r"[A-Za-z']+", t)
    return len(words) == 3 and all(w == w.lower() for w in words) and not re.search(r"\d", t)

def code_ok(answer, tests):
    a = strip_think(answer)
    # The last fenced block; if a model writes ``` inside its own docstring that block is cut short,
    # so also try everything from the first fence to the last one. The answer passes if either runs.
    blocks = re.findall(r"```(?:python)?\n(.*?)```", a, re.S)
    greedy = re.findall(r"```(?:python)?\n(.*)```", a, re.S)
    candidates = ([blocks[-1]] if blocks else []) + greedy + ([] if blocks else [a])
    err = ""
    for code in candidates:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code + "\n\n" + tests + "\nprint('PASS')\n")
        try:
            r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            err = "timeout"; continue
        if "PASS" in r.stdout:
            return True, ""
        err = (r.stderr.strip().splitlines() or [""])[-1][:200]
    return False, err

def ask(model, prompt, cap_tokens=4096, cap_s=300):
    t0 = time.time()
    body = {"model": model, "prompt": prompt, "stream": True, "options": {**OPTS, "num_predict": cap_tokens}}
    req = urllib.request.Request(API + "/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    text, think, last, timed_out = [], [], {}, False
    with urllib.request.urlopen(req, timeout=cap_s + 120) as r:
        for line in r:
            chunk = json.loads(line)
            text.append(chunk.get("response", ""))
            think.append(chunk.get("thinking", "") or "")
            if chunk.get("done"):
                last = chunk
                break
            if time.time() - t0 > cap_s:
                timed_out = True
                break
    full = "".join(text)
    return {"answer": full, "thinking": "".join(think), "wall_s": round(time.time() - t0, 2), "timed_out": timed_out,
            "eval_count": last.get("eval_count"), "eval_s": (last.get("eval_duration") or 0) / 1e9}

def measure(model, keep, key):
    rec = {"model": model, "key": key, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "free_pct_before": free_pct(), "swap_mb_before": swap_mb()}
    t = time.time()
    subprocess.run(["ollama", "pull", model], check=True, capture_output=True)
    rec["pull_s"] = round(time.time() - t, 1)
    names = {model, model if ":" in model else model + ":latest"}  # "phi4-mini" is listed as "phi4-mini:latest"
    tag = next(m for m in get("/api/tags")["models"] if m["name"] in names or m["model"] in names)
    rec["size_gb"] = round(tag["size"] / 1e9, 2)
    rec["details"] = tag.get("details", {})
    subprocess.run(["ollama", "stop", model], capture_output=True)
    # cold load + first answer
    r = post("/api/generate", {"model": model, "prompt": "Say hi.", "stream": False, "options": {**OPTS, "num_predict": 8}})
    rec["load_s"] = round(r.get("load_duration", 0) / 1e9, 2)
    ps = get("/api/ps")["models"]
    p = next((m for m in ps if m["name"] in names or m["model"] in names), {})
    rec["vram_gb"] = round(p.get("size_vram", 0) / 1e9, 2)
    rec["loaded_gb"] = round(p.get("size", 0) / 1e9, 2)
    rec["gpu_share"] = round(p.get("size_vram", 0) / p["size"], 3) if p.get("size") else None
    # speed: 3 warm runs, 256 tokens
    runs = []
    for _ in range(3):
        r = post("/api/generate", {"model": model, "prompt": SPEED_PROMPT, "stream": False,
                                    "options": {**OPTS, "num_predict": 256}})
        runs.append({"gen_tps": r["eval_count"] / (r["eval_duration"] / 1e9), "prompt_tps": r["prompt_eval_count"] / max(r["prompt_eval_duration"] / 1e9, 1e-6),
                     "eval_count": r["eval_count"]})
    runs.sort(key=lambda x: x["gen_tps"])
    rec["gen_tps"] = round(runs[1]["gen_tps"], 2)
    rec["prompt_tps"] = round(sorted(x["prompt_tps"] for x in runs)[1], 1)
    rec["speed_runs"] = runs
    # once more after a ~2,000-token prompt (prefill speed, and generation with a filled context)
    long_prompt = ("Here is some code:\n\n" + (HERE / "bench.py").read_text()[:7000] +
                   "\n\nSummarise what this script measures in five bullet points.")
    r = post("/api/generate", {"model": model, "prompt": long_prompt, "stream": False, "options": {**OPTS, "num_predict": 256}})
    rec["long_prompt_tokens"] = r.get("prompt_eval_count")
    rec["long_prompt_tps"] = round(r["prompt_eval_count"] / max(r["prompt_eval_duration"] / 1e9, 1e-6), 1)
    rec["long_gen_tps"] = round(r["eval_count"] / max(r["eval_duration"] / 1e9, 1e-6), 2)
    rec["swap_mb_during"] = swap_mb()
    rec["free_pct_during"] = free_pct()
    # exam
    exam, raw = {}, {}
    for name, (prompt, tests) in CODE_TASKS.items():
        a = ask(model, prompt)
        ok, err = (False, "timed out") if a["timed_out"] else code_ok(a["answer"], tests)
        exam[name] = {"ok": ok, "wall_s": a["wall_s"], "tokens": a["eval_count"], "timed_out": a["timed_out"], "error": err}
        raw[name] = a
    for name, (prompt, reference) in SQL_TASKS.items():
        a = ask(model, SQL_SCHEMA + prompt + SQL_TAIL)
        ok, err = (False, "timed out") if a["timed_out"] else sql_ok(a["answer"], reference)
        exam[name] = {"ok": ok, "wall_s": a["wall_s"], "tokens": a["eval_count"], "timed_out": a["timed_out"], "error": err}
        raw[name] = a
    for name, (prompt, check) in EXACT_TASKS.items():
        a = ask(model, prompt)
        ok = False if a["timed_out"] else bool(check(a["answer"]))
        exam[name] = {"ok": ok, "wall_s": a["wall_s"], "tokens": a["eval_count"], "timed_out": a["timed_out"]}
        raw[name] = a
    rec["exam"] = exam
    rec["score"] = sum(v["ok"] for v in exam.values())
    rec["exam_wall_s"] = round(sum(v["wall_s"] for v in exam.values()), 1)
    rec["swap_mb_after"] = swap_mb()
    (HERE / "raw").mkdir(exist_ok=True)
    (HERE / "raw" / f"{key.replace(':', '_').replace('|', '__')}.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    subprocess.run(["ollama", "stop", model], capture_output=True)
    if not keep:
        subprocess.run(["ollama", "rm", model], capture_output=True)
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--tag", help="store under '<model>|<tag>' (e.g. a run with different server settings)")
    a = ap.parse_args()
    out = HERE / "results.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    keyof = lambda m: f"{m}|{a.tag}" if a.tag else m
    todo = [a.only] if a.only else [m for m in MODELS if keyof(m) not in results]
    for i, model in enumerate(todo):
        if i:
            time.sleep(60)  # let swap and memory settle between models
        print(f"== {model}", flush=True)
        try:
            rec = measure(model, a.keep, keyof(model))
        except Exception as e:  # record the failure and move on (e.g. a model that cannot load)
            rec = {"model": model, "error": repr(e)[:400], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            subprocess.run(["ollama", "rm", model], capture_output=True)
        results[keyof(model)] = rec
        out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"   {rec.get('size_gb')} GB · load {rec.get('load_s')} s · {rec.get('gen_tps')} tok/s · GPU {rec.get('gpu_share')} · "
              f"prefill {rec.get('prompt_tps')} / long {rec.get('long_prompt_tps')} tok/s · score {rec.get('score')}/14 · exam {rec.get('exam_wall_s')} s · swap {rec.get('swap_mb_during')} MB {rec.get('error', '')}", flush=True)

if __name__ == "__main__":
    main()
