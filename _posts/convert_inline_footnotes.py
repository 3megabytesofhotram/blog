import re
import sys

def convert_inline_footnotes(md_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    footnotes = []
    footnote_map = {}
    
    def footnote_replacer(match):
        inline_note = match.group(1)
        if inline_note not in footnote_map:
            footnote_num = len(footnotes) + 1
            footnote_map[inline_note] = footnote_num
            footnotes.append(f"[^{footnote_num}]: {inline_note}")
        return f"[^{footnote_map[inline_note]}]"
    
    # Replace inline footnotes with reference footnotes
    content = re.sub(r'\^\[(.*?)\]', footnote_replacer, content)
    
    # Append extracted footnotes at the end of the document
    if footnotes:
        content += '\n\n' + '\n'.join(footnotes) + '\n'
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Converted footnotes in {md_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_inline_footnotes.py <markdown_file>")
        sys.exit(1)
    convert_inline_footnotes(sys.argv[1])
