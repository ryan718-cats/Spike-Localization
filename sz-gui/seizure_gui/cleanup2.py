with open('sz_gui.py', 'r') as f:
    lines = f.readlines()

# Find key line numbers
normalize_end = None
match_start = None

for i, line in enumerate(lines):
    if 'return "".join(str(int(p))' in line:
        normalize_end = i + 1
    if line.strip().startswith('def match_channel('):
        match_start = i
        break

if normalize_end and match_start:
    # Keep everything up to normalize_end, skip all garbage, then keep match_start onwards
    new_lines = lines[:normalize_end] + ['\n\n'] + lines[match_start:]
    
    with open('sz_gui.py', 'w') as f:
        f.writelines(new_lines)
    print(f'Cleaned: kept lines 1-{normalize_end}, deleted {normalize_end+1}-{match_start-1}, kept from {match_start} onwards')
else:
    print(f'Could not find anchors: normalize_end={normalize_end}, match_start={match_start}')
