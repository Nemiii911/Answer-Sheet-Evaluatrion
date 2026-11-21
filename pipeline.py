#!/usr/bin/env python3
"""
Answer Sheet Evaluation pipeline  (CNV AI Labs timed assignment)

  1. EXTRACT    page image/PDF  -> transcript text with inline markers
  2. STRUCTURE  transcripts     -> answers keyed by question number (handles page spillover)
  3. SCORE      answer + rubric -> marks by meaning (LLM judge, N runs, temperature 0)
  4. CONFIDENCE rule-based signals -> high / medium / low + one-line reason

Usage
  # full run on scanned pages (needs OPENAI_API_KEY)
  python pipeline.py --images page1.jpg page2.jpg --key answer_key.json --out-dir out

  # skip OCR, use saved/hand-corrected transcripts
  python pipeline.py --transcripts sample_transcripts/page*.txt --key answer_key.json --out-dir out

  # fully offline smoke test (lexical baseline instead of LLM judge)
  python pipeline.py --transcripts sample_transcripts/page*.txt --key answer_key.json --baseline-judge

Transcript markers produced by the extractor (and relied on by later steps):
  [?word?]          uncertain word            [illegible]       unreadable word
  [crossed-out: x]  struck-through text       [DIAGRAM: desc]   diagram / drawing
  [MARGIN: x]       margin note
"""
import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import statistics
import sys
from pathlib import Path

MODEL = os.environ.get("EVAL_MODEL", "gpt-4.1-mini")

# --------------------------------------------------------------------------- #
# LLM helper
# --------------------------------------------------------------------------- #
_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()  # reads OPENAI_API_KEY
    return _client


def call_llm(system, content, max_tokens=3000):
    """One OpenAI Responses API call, temperature 0 where the model allows it."""
    client = get_client()
    # Text can be passed directly; multimodal parts must be inside a user message.
    request_input = content if isinstance(content, str) else [{"role": "user", "content": content}]
    kwargs = dict(model=MODEL, max_output_tokens=max_tokens, instructions=system, input=request_input)
    try:
        resp = client.responses.create(temperature=0, **kwargs)
    except Exception as e:  # some models reject the temperature parameter
        if "temperature" in str(e).lower():
            resp = client.responses.create(**kwargs)
        else:
            raise
    return resp.output_text


def parse_json(text):
    """Pull the first JSON object out of an LLM reply (tolerates code fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start:end + 1])


# --------------------------------------------------------------------------- #
# STEP 1 - EXTRACT
# --------------------------------------------------------------------------- #
EXTRACT_SYSTEM = """You are a careful handwriting transcriber for exam answer sheets.
Transcribe EVERYTHING written on the page, in reading order, exactly as the student wrote it.
Do NOT correct spelling, grammar or factual mistakes. Do NOT answer the questions yourself.
Keep question numbers exactly as written (e.g. "Q1.", "Ans 2", "3)").
Use these inline markers:
  [?word?]            a word you can read only with doubt (give your best guess inside)
  [illegible]         a word you cannot read at all
  [crossed-out: ...]  text the student struck through (transcribe it if readable)
  [DIAGRAM: ...]      a drawing/figure, with a short description and any labels, placed where it appears
  [MARGIN: ...]       notes written in margins or outside the main answer area
If the page begins mid-answer (continuation of the previous page) just start transcribing; do not invent a question number.
Return ONLY the transcript as plain text. No commentary."""


def encode_file(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    data_url = f"data:{mime};base64,{data}"
    if mime == "application/pdf":
        return {"type": "input_file", "filename": Path(path).name, "file_data": data_url}
    return {"type": "input_image", "image_url": data_url}


def extract_page(path):
    content = [encode_file(path), {"type": "input_text", "text": "Transcribe this answer sheet page."}]
    return call_llm(EXTRACT_SYSTEM, content, max_tokens=4000).strip()


def extract_all(image_paths, out_dir):
    """OCR every page; cache transcripts to disk so reruns are free and hand-fixable."""
    tdir = Path(out_dir) / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    pages = []
    for i, p in enumerate(image_paths, 1):
        cache = tdir / f"page{i}.txt"
        if cache.exists():
            print(f"[extract] page {i}: using cached {cache}")
            pages.append(cache.read_text())
            continue
        print(f"[extract] page {i}: {p}")
        text = extract_page(p)
        cache.write_text(text)
        pages.append(text)
    return pages


# --------------------------------------------------------------------------- #
# STEP 2 - STRUCTURE
# --------------------------------------------------------------------------- #
PREFIXED = re.compile(r"^\s*(?:Question|Ques|Que|Q|Answer|Ans|A)\s*[.:\-]?\s*(?:No\.?\s*)?(\d{1,2})\s*[.):\-]?\s*(.*)$", re.I)
BARE = re.compile(r"^\s*(\d{1,2})\s*[.):]\s+(.*)$")
TAG = re.compile(r"\[(?:\?[^\]]*\?|illegible|crossed-out:[^\]]*|DIAGRAM:[^\]]*|MARGIN:[^\]]*)\]", re.I)


def match_marker(line, qids, seen, allow_bare):
    m = PREFIXED.match(line)
    if m and int(m.group(1)) in qids and int(m.group(1)) not in seen:
        return int(m.group(1)), m.group(2)
    if allow_bare:
        m = BARE.match(line)
        expected = min((q for q in qids if q not in seen), default=None)
        if m and int(m.group(1)) == expected:
            return expected, m.group(2)
    return None


def segment(pages, qids, allow_bare=False):
    """
    Split page transcripts into answers by question marker.
    Content that appears before the first marker on a page (page > 1) is a
    spillover from the previous page and is appended to the open answer.
    """
    ans = {}
    current, seen = None, set()
    for pno, text in enumerate(pages, 1):
        marker_on_page = False
        for line in text.splitlines():
            if not line.strip():
                continue
            hit = match_marker(line, qids, seen, allow_bare)
            if hit:
                qid, rest = hit
                seen.add(qid)
                current = qid
                marker_on_page = True
                ans[qid] = {"lines": [], "pages": {pno}, "spillover": False}
                if rest.strip():
                    ans[qid]["lines"].append(rest.strip())
            elif current is not None:
                a = ans[current]
                a["lines"].append(line.strip())
                a["pages"].add(pno)
                if pno > 1 and not marker_on_page:
                    a["spillover"] = True  # text continued from previous page
    return ans


def analyse_answer(raw_lines):
    """Derive clean text (what the judge sees) and the quality signals used for confidence."""
    raw = "\n".join(raw_lines)
    crossed = re.findall(r"\[crossed-out:[^\]]*\]", raw, flags=re.I)
    diagrams = re.findall(r"\[DIAGRAM:([^\]]*)\]", raw, flags=re.I)
    margins = re.findall(r"\[MARGIN:([^\]]*)\]", raw, flags=re.I)
    unsure = re.findall(r"\[\?[^\]]*\?\]", raw)
    illeg = re.findall(r"\[illegible\]", raw, flags=re.I)

    clean = re.sub(r"\[crossed-out:[^\]]*\]", "", raw, flags=re.I)   # student cancelled it
    clean = re.sub(r"\[MARGIN:[^\]]*\]", "", clean, flags=re.I)       # not part of the answer
    clean = re.sub(r"\[\?([^\]]*?)\?\]", r"\1", clean)                # keep best-guess word
    clean = re.sub(r"\[DIAGRAM:([^\]]*)\]", r"(diagram drawn: \1)", clean, flags=re.I)
    clean = re.sub(r"[ \t]+", " ", clean).strip()

    words = len(re.sub(r"\[[^\]]*\]", "", raw).split())
    bad = len(unsure) + len(illeg)
    return {
        "clean_text": clean,
        "has_diagram": bool(diagrams),
        "has_crossed_out": bool(crossed),
        "has_margin_note": bool(margins),
        "uncertain_ratio": round(bad / max(words + bad, 1), 3),
        "uncertain_count": bad,
    }


# --------------------------------------------------------------------------- #
# STEP 3 - SCORE
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """You are a fair, consistent examiner marking a student's handwritten answer (already transcribed).
Judge MEANING, not wording: a correct idea in different words earns full marks for that rubric point.
Do not penalise spelling, grammar, handwriting-transcription artifacts or word order.
Award marks per rubric point (partial marks in steps of 0.5 are allowed, never above the point's marks).
Do not award a point if the answer contradicts it or states something factually wrong about it.
Do not credit things the student did not write. The text "(diagram drawn: ...)" means a diagram exists but you
cannot verify it; do not give marks for diagram quality.
Return ONLY JSON:
{"points":[{"point":"<rubric point>","awarded":<number>,"evidence":"<short quote or 'not present'>"}],
 "reason":"<one sentence on the overall marking>"}"""


def judge_llm(q, answer_text):
    rubric = "\n".join(f"{i+1}. ({p['marks']} mark) {p['point']}" for i, p in enumerate(q["rubric"]))
    prompt = (f"QUESTION: {q['question']}\nMAX MARKS: {q['max_marks']}\n"
              f"MODEL ANSWER (reference): {q['model_answer']}\n\nRUBRIC:\n{rubric}\n\n"
              f"STUDENT ANSWER:\n{answer_text or '(blank)'}")
    data = parse_json(call_llm(JUDGE_SYSTEM, prompt, max_tokens=1500))
    return normalise_judgement(q, data)


STOP = set("a an the of and or to in on is are was were it its that this with for by as at from be have has had "
           "which who into their they them then than so such can will not no but".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1}


def judge_baseline(q, answer_text):
    """Lexical fallback (NOT semantic). Offline smoke tests and a sanity baseline to compare against the LLM judge."""
    a = toks(answer_text)
    pts = []
    for p in q["rubric"]:
        t = toks(p["point"])
        frac = len(t & a) / max(len(t), 1)
        awarded = p["marks"] if frac >= 0.5 else (p["marks"] / 2 if frac >= 0.3 else 0)
        pts.append({"point": p["point"], "awarded": awarded, "evidence": f"token overlap {frac:.2f}"})
    return normalise_judgement(q, {"points": pts, "reason": "lexical baseline (not semantic)"})


def normalise_judgement(q, data):
    """Clip each point to its max and recompute the total ourselves (never trust the LLM's arithmetic)."""
    rub = q["rubric"]
    out = []
    for i, p in enumerate(rub):
        got = data["points"][i] if i < len(data.get("points", [])) else {}
        awarded = max(0.0, min(float(got.get("awarded", 0) or 0), p["marks"]))
        out.append({"point": p["point"], "max": p["marks"], "awarded": awarded, "evidence": got.get("evidence", "")})
    return {"points": out, "total": round(sum(x["awarded"] for x in out), 2), "reason": data.get("reason", "")}


def score_answer(q, answer_text, runs, baseline):
    if not answer_text.strip():
        j = normalise_judgement(q, {"points": [], "reason": "no answer found"})
        return {"score": 0.0, "spread": 0.0, "judgement": j, "runs": [0.0]}
    judge = judge_baseline if baseline else judge_llm
    results = [judge(q, answer_text) for _ in range(1 if baseline else runs)]
    totals = [r["total"] for r in results]
    # take the median run as the representative judgement
    rep = min(results, key=lambda r: abs(r["total"] - statistics.median(totals)))
    return {"score": round(statistics.median(totals), 2), "spread": round(max(totals) - min(totals), 2),
            "judgement": rep, "runs": totals}


# --------------------------------------------------------------------------- #
# STEP 4 - CONFIDENCE
# --------------------------------------------------------------------------- #
def confidence(q, seg_info, sig, sc, baseline):
    """
    Rule-based. Each signal is (severity, reason). Rules:
      any 'low'          -> low
      >=2 'medium'       -> low
      1 'medium'         -> medium
      otherwise          -> high
    Rule-based signals are used rather than asking the LLM how sure it is, because LLM self-reports are overconfident.
    """
    sigs = []
    mx = q["max_marks"]
    frac = sc["score"] / mx

    if seg_info is None or not sig["clean_text"].strip():
        sigs.append(("low", "no answer text found: skipped by student or missed by segmentation/OCR"))
    else:
        if sig["uncertain_ratio"] >= 0.15:
            sigs.append(("low", f"{int(sig['uncertain_ratio']*100)}% of words unreadable/uncertain in OCR"))
        elif sig["uncertain_count"] >= 1:
            sigs.append(("medium", f"{sig['uncertain_count']} uncertain/illegible word(s) in OCR"))
        if sc["spread"] >= 0.2 * mx:
            sigs.append(("low", f"judge runs disagree by {sc['spread']} marks"))
        elif sc["spread"] >= 0.1 * mx:
            sigs.append(("medium", f"judge runs differ by {sc['spread']} marks"))
        if 0.4 <= frac <= 0.6:
            sigs.append(("medium", "score is borderline (near half marks)"))
        if seg_info["spillover"]:
            sigs.append(("medium", "answer continues across pages; stitching may be wrong"))
        if sig["has_diagram"]:
            sigs.append(("medium", "contains a diagram that is not evaluated and may carry marks"))
        if sig["has_crossed_out"]:
            sigs.append(("note", "crossed-out text ignored"))
        if baseline:
            sigs.append(("medium", "scored with lexical baseline, not semantic judge"))

    sev = [s for s, _ in sigs]
    if "low" in sev or sev.count("medium") >= 2:
        label = "low"
    elif "medium" in sev:
        label = "medium"
    else:
        label = "high"

    order = {"low": 0, "medium": 1, "note": 2}
    reasons = [r for s, r in sorted(sigs, key=lambda x: order[x[0]])]
    reason = "; ".join(reasons[:2]) if reasons else "clean OCR, single page, judge runs agree, score not borderline"
    return label, reason


# --------------------------------------------------------------------------- #
# Orchestration + output
# --------------------------------------------------------------------------- #
def key_comparison(q, sc):
    hit = [p["point"] for p in sc["judgement"]["points"] if p["awarded"] >= p["max"]]
    part = [p["point"] for p in sc["judgement"]["points"] if 0 < p["awarded"] < p["max"]]
    miss = [p["point"] for p in sc["judgement"]["points"] if p["awarded"] == 0]
    s = f"{sc['score']}/{q['max_marks']}. Matched {len(hit)}/{len(q['rubric'])} rubric points fully"
    if part:
        s += f", {len(part)} partially"
    if miss:
        s += f". Missing: " + " | ".join(miss)
    return s


def run(pages, key, judge_runs, baseline, allow_bare):
    qids = {q["id"] for q in key["questions"]}
    segs = segment(pages, qids, allow_bare)
    rows = []
    for q in key["questions"]:
        seg = segs.get(q["id"])
        sig = analyse_answer(seg["lines"]) if seg else analyse_answer([])
        sc = score_answer(q, sig["clean_text"], judge_runs, baseline)
        label, reason = confidence(q, seg, sig, sc, baseline)
        rows.append({
            "question_no": q["id"],
            "extracted_answer": sig["clean_text"],
            "score": sc["score"],
            "max_marks": q["max_marks"],
            "score_vs_key": key_comparison(q, sc),
            "confidence": label,
            "reason": reason,
            "needs_human_review": label != "high",
            "pages": sorted(seg["pages"]) if seg else [],
            "judge_runs": sc["runs"],
            "judge_reason": sc["judgement"]["reason"],
        })
    return rows


def write_outputs(rows, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    print(f"\nWrote {out/'results.json'} and {out/'results.csv'}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", help="page images/PDFs in order")
    src.add_argument("--transcripts", nargs="+", help="skip OCR: transcript .txt files in page order")
    ap.add_argument("--key", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--judge-runs", type=int, default=2, help="LLM judge repeats (disagreement lowers confidence)")
    ap.add_argument("--baseline-judge", action="store_true", help="offline lexical judge instead of the LLM")
    ap.add_argument("--allow-bare-numbers", action="store_true",
                    help="also treat a bare '1.' line as a question marker (risky if answers use numbered lists)")
    a = ap.parse_args()

    key = json.loads(Path(a.key).read_text())
    if a.transcripts:
        pages = [Path(p).read_text() for p in a.transcripts]
    else:
        pages = extract_all(a.images, a.out_dir)

    rows = run(pages, key, a.judge_runs, a.baseline_judge, a.allow_bare_numbers)
    write_outputs(rows, a.out_dir)
    print(f"\n{'Q':<3}{'score':<9}{'conf':<8}reason")
    for r in rows:
        print(f"{r['question_no']:<3}{str(r['score'])+'/'+str(r['max_marks']):<9}{r['confidence']:<8}{r['reason']}")


if __name__ == "__main__":
    sys.exit(main())
