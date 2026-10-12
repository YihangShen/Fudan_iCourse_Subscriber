import re
import smtplib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import BytesIO
from PIL import Image
from collections import OrderedDict
from email.message import EmailMessage
from email.policy import SMTP
from html import escape
from email.utils import formataddr, formatdate, make_msgid
from urllib.parse import quote

import markdown
from pygments.formatters import HtmlFormatter

from src.runtime import config

_MD_EXTENSIONS= ["tables", "fenced_code", "nl2br", "sane_lists", "codehilite"]

_MD_EXTENSION_CONFIGS = {
    "codehilite": {
        "guess_lang": False,
        "linenums": False,
        "css_class": "highlight",
    }
}

_PYGMENTS_CSS = HtmlFormatter(style="friendly").get_style_defs(".highlight")

_EMAIL_CSS = """\
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 15px;
    line-height: 1.7;
    color: #1a1a1a;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
h2 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
    margin-top: 32px;
}
h3 {
    color: #34495e;
    margin-top: 24px;
}
h3 small {
    color: #7f8c8d;
    font-weight: normal;
}
h4 { color: #555; margin-top: 18px; }
hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 28px 0;
}
strong { color: #c0392b; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}
th {
    background: #f5f6fa;
    font-weight: 600;
}
tr:nth-child(even) { background: #fafafa; }
pre {
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px 16px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.5;
}
code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 13px;
}
p code {
    background: #f0f0f0;
    padding: 2px 5px;
    border-radius: 3px;
}
blockquote {
    border-left: 4px solid #3498db;
    margin: 12px 0;
    padding: 8px 16px;
    background: #f8f9fa;
    color: #555;
}
ul, ol { padding-left: 24px; }
li { margin-bottom: 4px; }
"""

_MIN_INLINE_HEIGHT = 13  # minimum logical height for inline formulas (px)

_IMAGE_CACHE: dict[str, tuple] = {}


def _fetch_latex_image(url: str, dpi: int = 300) -> tuple:
    """Fetch rendered LaTeX image, return (width, height, png_bytes).

    Width/height are logical display pixels (DPI-adjusted).
    Returns (None, None, None) on failure.
    """
    if url in _IMAGE_CACHE:
        return _IMAGE_CACHE[url]

    try:
        scale_factor = dpi / 96.0
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        logical_width = max(1, int(img.width / scale_factor))
        logical_height = max(1, int(img.height / scale_factor))

        result = (logical_width, logical_height, response.content)
        _IMAGE_CACHE[url] = result
        return result
    except Exception as e:
        print(f"[LaTeX Render] Image fetch failed: {e}")
        return None, None, None


def _prefetch_latex_images(urls: list[str], dpi: int = 300) -> None:
    """Pre-fetch multiple LaTeX images concurrently.

    Results are stored in ``_IMAGE_CACHE`` so that subsequent calls to
    ``_fetch_latex_image`` become instant cache hits.
    """
    uncached = [u for u in urls if u not in _IMAGE_CACHE]
    if not uncached:
        return
    with ThreadPoolExecutor(max_workers=min(len(uncached), 8)) as pool:
        futures = {pool.submit(_fetch_latex_image, u, dpi): u for u in uncached}
        for future in as_completed(futures):
            future.result()  # trigger any exception logging inside _fetch_latex_image


def _md_to_html(md_text: str, cid_images: dict | None = None) -> str:
    """Convert Markdown to styled HTML, rendering LaTeX math as images.

    Processing order: extract LaTeX → markdown convert → restore as <img>.
    This prevents the markdown engine from corrupting backslash escapes.

    Args:
        md_text: Markdown source text.
        cid_images: When provided (dict), download images and embed via CID
                    references instead of external URLs.  The dict is populated
                    with {cid_name: png_bytes} entries for the caller to attach
                    to the MIME message.
    """
    latex_map: dict[str, str] = {}
    counter = 0

    def _stash(match):
        nonlocal counter
        key = f"\x00LATEX{counter}\x00"
        counter += 1
        latex_map[key] = match.group(0)
        return key

    def _stash_block(match):
        """Stash \\[...\\] as $$...$$ for uniform downstream handling."""
        nonlocal counter
        key = f"\x00LATEX{counter}\x00"
        counter += 1
        latex_map[key] = "$$" + match.group(1) + "$$"
        return key

    def _stash_inline(match):
        """Stash \\(...\\) as $...$ for uniform downstream handling."""
        nonlocal counter
        key = f"\x00LATEX{counter}\x00"
        counter += 1
        latex_map[key] = "$" + match.group(1) + "$"
        return key

    # Extract LaTeX in order: block first, then inline
    # 1) $$...$$ block formulas
    text = re.sub(r"\$\$(.+?)\$\$", _stash, md_text, flags=re.DOTALL)
    # 2) \[...\] block formulas (normalize to $$...$$)
    text = re.sub(r"\\\[(.+?)\\\]", _stash_block, text, flags=re.DOTALL)
    # 3) $...$ inline formulas (not $$)
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _stash, text)
    # 4) \(...\) inline formulas (normalize to $...$)
    text = re.sub(r"\\\((.+?)\\\)", _stash_inline, text)

    html = markdown.markdown(
        text,
        extensions=_MD_EXTENSIONS,
        extension_configs=_MD_EXTENSION_CONFIGS,
    )

    # Build URL list and pre-fetch all LaTeX images concurrently
    # Each entry: (url, latex_content, is_block)
    latex_info: dict[str, tuple[str, str, bool]] = {}
    for key, original in latex_map.items():
        is_block = original.startswith("$$")
        latex_content = original[2:-2] if is_block else original[1:-1]
        prefix = r"\dpi{300}\bg{white}" if is_block else r"\dpi{300}\bg{white}\inline"
        url = f"https://latex.codecogs.com/png.latex?{prefix}%20{quote(latex_content)}"
        latex_info[key] = (url, latex_content, is_block)

    _prefetch_latex_images([info[0] for info in latex_info.values()])

    for key, (url, latex_content, is_block) in latex_info.items():
        w, h, img_data = _fetch_latex_image(url)

        if is_block:
            if w and h:
                src = _resolve_src(url, img_data, cid_images)
                img_tag = (
                    f'<div style="text-align:center;margin:16px 0">'
                    f'<img src="{src}" alt="{escape(latex_content)}" '
                    f'width="{w}" height="{h}" '
                    f'style="width:{w}px;height:{h}px;max-width:none;'
                    f'vertical-align:middle;border:none;display:inline-block;">'
                    f'</div>'
                )
            else:
                img_tag = (
                    f'<div style="text-align:center;margin:16px 0">'
                    f'<code>{escape(latex_content)}</code></div>'
                )
        else:
            if w and h:
                # Enforce minimum height so formulas aren't smaller than text
                if h < _MIN_INLINE_HEIGHT:
                    scale = _MIN_INLINE_HEIGHT / h
                    w = max(1, int(w * scale))
                    h = _MIN_INLINE_HEIGHT

                src = _resolve_src(url, img_data, cid_images)
                img_tag = (
                    f'<img src="{src}" alt="{escape(latex_content)}" '
                    f'width="{w}" height="{h}" '
                    f'style="width:{w}px;height:{h}px;max-width:none;'
                    f'vertical-align:-3px;border:none;margin:0 2px;">'
                )
            else:
                img_tag = f'<code>{escape(latex_content)}</code>'

        html = html.replace(key, img_tag)

    return html


def _resolve_src(url: str, img_data: bytes | None,
                 cid_images: dict | None) -> str:
    """Return a CID reference if embedding, otherwise the original URL."""
    if cid_images is not None and img_data:
        cid = f"latex-{uuid.uuid4().hex[:12]}"
        cid_images[cid] = img_data
        return f"cid:{cid}"
    return url


def _important_notices(summary: str) -> str:
    """Extract explicit notice sections, retaining nested headings and lists.

    Do not infer deadlines or treat absence of this section as absence of
    homework. Old summaries without a notice heading remain intact in the MD.
    """
    sections = []
    active_level = None
    fence = None
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
            if active_level is not None:
                sections.append(line)
            continue
        if fence:
            if active_level is not None:
                sections.append(line)
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", stripped)
        title = heading.group(1) if heading else stripped
        title = title.strip("* _【】[]：:")
        is_notice = title in {"课程事项提醒", "课程事项", "重要事项提醒", "重要事项"}
        if heading:
            level = len(stripped) - len(stripped.lstrip("#"))
            if active_level is not None and level <= active_level:
                active_level = None
            if is_notice:
                active_level = level
                continue
        elif is_notice:
            active_level = 6  # a legacy standalone bold label
            continue
        if active_level is not None:
            sections.append(line)
    text = "\n".join(sections).strip()
    # Plain text only; preserve the actual wording, numbers and URLs.
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return text or "笔记未单独列出课程事项，请查阅附件确认。"


def _header_text(value: str) -> str:
    return " ".join(str(value).splitlines()).strip()


def _attachment_name(item: dict, index: int) -> str:
    label = "_".join(str(item.get(k) or "") for k in ("course_title", "date", "sub_title"))
    label = re.sub(r'[\x00-\x1f\x7f/\\:*?"<>|]', "_", label).strip(" ._")
    return f"{index:03d}_{label[:100] or '课程笔记'}.md"


class Emailer:
    """Send UTF-8 plain-text notices with complete Markdown attachments."""

    def __init__(self):
        self.host = config.SMTP_HOST
        self.port = config.SMTP_PORT
        self.sender = config.SMTP_EMAIL.strip()
        self.password = config.SMTP_PASSWORD.strip()
        self.receiver = config.RECEIVER_EMAIL.strip()

    def _build_message(self, items: list[dict]) -> EmailMessage:
        courses: OrderedDict[str, list[dict]] = OrderedDict()
        for item in items:
            courses.setdefault(item["course_title"], []).append(item)
        subject = "[FiCS] " + ", ".join(
            f"{_header_text(title)} ({len(lectures)})"
            for title, lectures in courses.items()
        )
        if any(item.get("is_update") for item in items):
            subject += "（笔记更新）"

        body = ["课程事项提醒", "完整课程笔记见随信附带的 .md 文件（UTF-8 编码）。",
                "附件保留 Markdown 排版和 LaTeX 公式。", ""]
        attachments = []
        for index, item in enumerate(items, 1):
            filename = _attachment_name(item, index)
            title = _header_text(item["course_title"])
            lecture = _header_text(item.get("sub_title", ""))
            date = _header_text(item.get("date", ""))
            body.extend([f"{index}. {title} — {lecture} ({date})",
                         _important_notices(item["summary"]),
                         f"笔记附件：{filename}", ""])
            document = (f"# {title}\n\n课次：{lecture}\n\n日期：{date}\n\n"
                        + item["summary"] + "\n")
            attachments.append((filename, document.encode("utf-8")))

        msg = EmailMessage(policy=SMTP)
        msg["Subject"] = subject
        msg["From"] = formataddr(("iCourse Subscriber", self.sender))
        msg["To"] = self.receiver
        msg["Date"] = formatdate(localtime=False)
        msg["Message-ID"] = make_msgid()
        msg.set_content("\n".join(body), charset="utf-8", cte="base64")
        for filename, content in attachments:
            msg.add_attachment(content, maintype="text", subtype="markdown",
                               filename=filename, params={"charset": "utf-8"})
        return msg

    def send(self, items: list[dict]) -> bool:
        if not items:
            print("[Emailer] Nothing to send.")
            return True
        if not all((self.sender, self.password, self.receiver)):
            raise ValueError("SMTP_EMAIL, SMTP_PASSWORD and RECEIVER_EMAIL are required.")
        msg = self._build_message(items)
        # Reuse Message-ID on retry; avoid duplicate delivery when QUIT fails
        # after the SMTP server has already accepted the message.
        for attempt in range(3):
            accepted = False
            try:
                with smtplib.SMTP_SSL(self.host, self.port, timeout=60) as server:
                    server.login(self.sender, self.password)
                    refused = server.send_message(
                        msg, from_addr=self.sender, to_addrs=[self.receiver],
                    )
                    if refused:
                        raise smtplib.SMTPRecipientsRefused(refused)
                    accepted = True
                print(f"[Emailer] SMTP accepted {len(items)} Markdown attachment(s).")
                return True
            except Exception as e:
                if accepted:
                    print("[Emailer] SMTP accepted message; connection close failed. "
                          "Not resending.")
                    return True
                print(f"[Emailer] Attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        print("[Emailer] All send attempts failed; notes remain unsent for retry.")
        return False
