with open('sz_gui.py', 'r') as f:
    lines = f.readlines()

triple_quote_lines = []
for i, line in enumerate(lines):
    if '"""' in line:
        count = line.count('"""')
        triple_quote_lines.append((i+1, count, line.rstrip()))

print("All lines with triple quotes:")
for line_num, count, text in triple_quote_lines:
    print(f'Line {line_num}: {count} occurrences')
    print(f'  {text[:100]}')
