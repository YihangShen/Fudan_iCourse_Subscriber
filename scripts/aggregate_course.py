#!/usr/bin/env python3
"""Aggregate every lecture summary of one course into a single Markdown file.

Usage:
    python scripts/aggregate_course.py <db_path> <course_id> <output_dir>

Writes:
    <output_dir>/<course_title>_课程笔记汇总_<today>.md   (combined document)
    <output_dir>/lectures/<sub_id>_<date>.md              (one file per lecture)
    <output_dir>/manifest.txt                             (coverage report)
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import Database  # noqa: E402

CST = timezone(timedelta(hours=8))

_ANCHOR_DROP = set("!\"#$%&'()*+,./:;<=>?@[\\]^`{|}~")


def _safe(s: str) -> str:
    return re.sub(r"[^\w\-一-鿿]", "_", s)[:80]


_HEADING_RE = re.compile(r"^(#{1,6})(\s+)(.*)$")


def demote(md: str, levels: int = 2) -> str:
    """Push inner headings of a lecture summary down so they nest under the
    section's own ``##`` heading instead of competing with it."""
    out = []
    for line in md.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            new_level = min(6, len(m.group(1)) + levels)
            out.append("#" * new_level + m.group(2) + m.group(3))
        else:
            out.append(line)
    return "\n".join(out)


def anchor(title: str) -> str:
    """Approximate GitHub heading anchor generation."""
    t = title.strip().lower()
    t = "".join(" " if unicodedata.category(c) == "Zs" else c for c in t)
    t = "".join(c for c in t if c not in _ANCHOR_DROP)
    return t.replace(" ", "-")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    db_path = sys.argv[1]
    course_id = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "out"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "lectures"), exist_ok=True)

    db = Database(db_path)
    courses = db.conn.execute(
        "SELECT * FROM courses WHERE course_id = ?", (course_id,)
    ).fetchall()
    if not courses:
        print(f"[aggregate] course {course_id} not found; known course ids:")
        for row in db.conn.execute("SELECT course_id, title FROM courses").fetchall():
            print(f"  {row['course_id']}  {row['title']}")
        return 1
    course = courses[0]
    title = course["title"] or course_id
    teacher = course["teacher"] or ""

    rows = db.conn.execute(
        "SELECT * FROM lectures WHERE course_id = ?", (course_id,)
    ).fetchall()

    def sort_key(r):
        d = (r["date"] or "").strip() or "9999-99-99"
        try:
            sid = int(r["sub_id"])
        except (TypeError, ValueError):
            sid = 1 << 30
        return (d, sid)

    lectures = sorted((dict(r) for r in rows), key=sort_key)
    summarized = [l for l in lectures if (l.get("summary") or "").strip()]
    dates = [l["date"] for l in lectures if (l.get("date") or "").strip()]

    now = datetime.now(CST).strftime("%Y-%m-%d %H:%M")

    manifest = []
    manifest.append(f"course_id={course_id}")
    manifest.append(f"title={title}")
    manifest.append(f"teacher={teacher}")
    manifest.append(f"lectures_total={len(lectures)}")
    manifest.append(f"lectures_summarized={len(summarized)}")
    manifest.append(f"date_range={(min(dates) if dates else 'N/A')} ~ {(max(dates) if dates else 'N/A')}")
    manifest.append("")
    for i, lec in enumerate(lectures, 1):
        tr = len((lec.get("transcript") or "").strip())
        su = len((lec.get("summary") or "").strip())
        manifest.append(
            f"{i:02d} sub_id={lec['sub_id']} date={lec.get('date') or '-'} "
            f"title={lec.get('sub_title') or '-'} transcript_chars={tr} "
            f"summary_chars={su} processed_at={lec.get('processed_at') or '-'} "
            f"error_stage={lec.get('error_stage') or '-'}"
        )
    manifest_text = "\n".join(manifest) + "\n"
    print(manifest_text)
    with open(os.path.join(out_dir, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write(manifest_text)

    # One file per lecture
    for i, lec in enumerate(lectures, 1):
        head = f"# {title} — {lec.get('sub_title') or lec['sub_id']}（{lec.get('date') or '-'}）\n\n"
        body = (lec.get("summary") or "").strip()
        if not body:
            body = f"> 该课次暂未生成笔记（error_stage: {lec.get('error_stage') or '-'}）。"
        fname = f"{i:02d}_{_safe(lec.get('date') or lec['sub_id'])}_{_safe(lec.get('sub_title') or '')}.md"
        with open(os.path.join(out_dir, "lectures", fname), "w", encoding="utf-8") as f:
            f.write(head + body + "\n")

    # Combined document
    parts = [
        f"# {title} — 课程笔记汇总",
        "",
        f"- 课程 ID：{course_id}",
        f"- 任课教师：{teacher}",
        f"- 课次数：{len(lectures)}（其中 {len(summarized)} 节有完整笔记）",
        f"- 时间范围：{(min(dates) if dates else 'N/A')} ~ {(max(dates) if dates else 'N/A')}",
        f"- 汇总生成时间：{now}",
        "- 数据来源：iCourse 录播音频（语音转写）+ 课件 OCR，经大模型整理为课堂讲义",
        "",
        "## 目录",
        "",
    ]
    toc = []
    for lec in lectures:
        label = f"{lec.get('sub_title') or lec['sub_id']}（{lec.get('date') or '-'}）"
        toc.append(f"- [{label}](#{anchor(label)})")
    parts.extend(toc)
    parts.append("")
    parts.append("---")
    parts.append("")

    missing = []
    for lec in lectures:
        label = f"{lec.get('sub_title') or lec['sub_id']}（{lec.get('date') or '-'}）"
        parts.append(f"## {label}")
        parts.append("")
        body = (lec.get("summary") or "").strip()
        if body:
            parts.append(demote(body))
        else:
            reason = lec.get("error_stage") or "尚未处理"
            missing.append((label, lec["sub_id"], reason))
            parts.append(f"> 该课次暂未生成笔记（原因：{reason}）。")
        parts.append("")
        parts.append("---")
        parts.append("")

    if missing:
        parts.append("## 未生成笔记的课次")
        parts.append("")
        for label, sid, reason in missing:
            parts.append(f"- {label}（sub_id={sid}，原因：{reason}）")
        parts.append("")

    parts.append("")
    parts.append(f"*本文档共汇总 {len(summarized)} 节课的课堂讲义，由 iCourse Subscriber 自动生成；"
                 "语音转写与课件 OCR 可能存在识别误差，关键定义与推导请以教材和老师原话为准。*")
    parts.append("")

    today = datetime.now(CST).strftime("%Y-%m-%d")
    combined_name = f"{_safe(title)}_课程笔记汇总_{today}.md"
    combined_path = os.path.join(out_dir, combined_name)
    text = "\n".join(parts)
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[aggregate] wrote {combined_path} ({len(text)} chars, {text.count(chr(10)) + 1} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
