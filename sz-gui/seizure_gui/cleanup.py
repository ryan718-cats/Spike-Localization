with open('sz_gui.py', 'r') as f:
    lines = f.readlines()

# Find the indices of the problematic sections
parsing_idx = None
helpers_idx = None

for i, line in enumerate(lines):
    if '# ── Parsing' in line:
        parsing_idx = i
    if '# ── Helpers' in line and helpers_idx is None:
        helpers_idx = i
        break

if parsing_idx is not None and helpers_idx is not None:
    # Delete all lines between parsing_idx+1 and helpers_idx-1
    new_lines = lines[:parsing_idx+1] + ['\n', '\n'] + lines[helpers_idx:]
    
    with open('sz_gui.py', 'w') as f:
        f.writelines(new_lines)
    print(f'Removed lines {parsing_idx+2} to {helpers_idx-1}')
else:
    print(f'Could not find parsing_idx={parsing_idx}, helpers_idx={helpers_idx}')
