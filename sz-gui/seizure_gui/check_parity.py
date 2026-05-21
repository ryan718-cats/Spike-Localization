with open('sz_gui.py', 'r') as f:
    content = f.read()

# Count triple quotes
total_quotes = content.count('"""')
print(f'Total occurrences of triple quotes: {total_quotes}')
print(f'Even = OK, Odd = ERROR')

if total_quotes % 2 == 1:
    print('ERROR: Odd number of triple quotes found!')
    # Find the line where parity flips
    lines = content.split('\n')
    quote_count = 0
    for i, line in enumerate(lines):
        line_quotes = line.count('"""')
        if line_quotes > 0:
            quote_count += line_quotes
            parity = "OPEN" if (quote_count % 2 == 1) else "CLOSED"
            print(f'Line {i+1}: {line_quotes} quotes, total: {quote_count} ({parity})')
