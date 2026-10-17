
import re
import pymupdf

HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")

def extract_chapter_map(pdf_path):
    """
    Map each physical PDF page (1-based) to
    its active chapter and section.

    Uses PDF bookmarks, not text-based
    heading detection.

    Important limitation: 
    This implementation assigns metadata at page level. 
    If multiple sections start on the same PDF page, the last bookmarked section on that page is assigned to the entire page. 
    We'll keep this simple initially and improve it later if necessary.
    """

    with pymupdf.open(pdf_path) as pdf:
        toc = pdf.get_toc()

        # Collect numbered chapter/section bookmarks.
        headings = []

        for level, title, page in toc:
            match = HEADING_PATTERN.match(title.strip())

            if not match or page < 1:
                continue

            number = match.group(1)
            depth = len(number.split("."))

            headings.append({
                "number": number,
                "title": title.strip(),
                "page": page,
                "depth": depth})

        # Preserve the PDF's bookmark order.
        # Entries on the same page are applied
        # in that order.
        headings.sort(key=lambda h: h["page"])

        chapter_map = {}
        active = {}
        heading_index = 0

        for page_number in range(1, len(pdf) + 1):
            while (heading_index < len(headings)and headings[heading_index]["page"] <= page_number):

                heading = headings[heading_index]
                depth = heading["depth"]

                # Remove headings at this level
                # and all deeper levels.
                active = {
                    d: value
                    for d, value in active.items()
                    if d < depth}

                active[depth] = heading["title"]
                heading_index += 1

            # Deepest active heading below
            # the chapter level.
            sections = [active[d] for d in sorted(active) if d >= 2]

            chapter_map[page_number] = {
                "chapter": active.get(
                    1, "Unknown Chapter"
                ),
                "section": (
                    sections[-1]
                    if sections
                    else "Unknown Section"
                ),
                "section_path": " > ".join(
                    active[d]
                    for d in sorted(active)
                ) or "Unknown"
            }

    return chapter_map


if __name__ == "__main__":

    from pathlib import Path

    for pdf_path in Path("./materials").glob("*.pdf"):

        print(f"\nPDF: {pdf_path.name}")

        chapter_map = extract_chapter_map(str(pdf_path))

        # Show only pages where the assigned
        # chapter or section changes.
        previous = None

        for page, info in chapter_map.items():
            if info != previous:
                print(
                    f"Page {page}: "
                    f"{info['section_path']}"
                )
            previous = info
