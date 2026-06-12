# Episode 2 — X Thread Draft

> Real numbers from the overnight ladder + ceiling probe. Tighten to your voice, then post.
> Delete this header before posting.

---

**Tweet 1 (hook)**
I tried to find the breaking point of my M5 Max — the model so big it crashes.

I ran a ladder from 3B to 123B parameters locally. Then allocated memory until macOS should've died.

It never broke. A 123-billion-parameter model runs on a laptop with room to spare. 🧵

---

**Tweet 2 (the curve)**
Decode throughput, 4-bit, same harness, idle machine:

3B   → 227 tok/s
7B   → 117 tok/s
14B  → 60 tok/s
32B  → 27 tok/s
70B  → 11 tok/s
123B → 6.3 tok/s

38× the parameters, ~36× slower. Decode is memory-bandwidth-bound — clean, predictable falloff.

---

**Tweet 3 (the real surprise: TTFT)**
Everyone quotes tok/s. The number that actually kills UX is time-to-first-token:

3B   → 89ms
32B  → 326ms
70B  → 1.8s
123B → 4.6s

Prefill on a huge model is brutal. At 123B you wait 4.6 seconds before the first word. That's the real cost of going big, not decode.

---

**Tweet 4 (memory is dead linear)**
Peak RAM scales almost perfectly with size at 4-bit (~0.56GB per billion params):

14B  → 8.5GB
32B  → 18.6GB
70B  → 39.9GB
123B → 69.2GB

The 123B model — the biggest I ran — used 69GB. On a 128GB machine. Half the tank.

---

**Tweet 5 (the breaking point that wasn't)**
So I went looking for the wall. I allocated GPU memory in 4GB chunks until Metal refused.

Apple advertises a 115GB "recommended working set."

It blew past it. I hit 128GB — on a 137GB machine — with no error. I stopped it myself for safety.

The 115GB limit is a suggestion, not a wall.

---

**Tweet 6 (what this means)**
The takeaway: unified memory is the whole game.

A PC with a 24GB GPU can't load a 70B model. This laptop ran a 123B model and had 60GB free. The "ceiling" is basically physical RAM minus the OS — far above what Apple's own API recommends.

The usability cliff for interactive use is ~32B. Above that it's batch work.

---

**Tweet 7 (the ops war story)**
Bonus, because build-in-public means showing the mess:

The 8-bit 72B download failed 3 times — hf-xet hung on resume, then read-timeouts, then a dropped connection. Lesson: `HF_HUB_DISABLE_XET=1` + a longer timeout is more reliable for big resumes, and every restart spawns orphaned partials.

Cut it loose. The curve didn't need it.

---

**Tweet 8 (repo + next)**
Everything reproducible — harness, per-model JSON, the memory probe:
→ github.com/Suryals/mlx-lab

Next: the eval that matters — can a local model pick the right tool for an ops alert? Speed is table stakes; correctness is the product.

Building in public. Follow along.

#MLX #AppleSilicon #LocalLLM #M5Max #AIEval #BuildInPublic
