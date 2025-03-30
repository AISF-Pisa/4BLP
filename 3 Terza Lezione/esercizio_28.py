def hanoi(n, source, auxiliary, target, memo={}):
    '''ljbcwobc
    '''
    if n == 1:
        return [(n, source, target)]
    
    if (n, source, target) in memo:
        return memo[(n, source, target)]
    
    steps = hanoi(n - 1, source, target, auxiliary)
    steps.append((n, source, target))
    steps += hanoi(n - 1, auxiliary, source, target)
    
    memo[(n, source, target)] = steps
    return steps

# Esempio di utilizzo
n = 3
steps = hanoi(n, "A", "B", "C")
print(steps)

for step in steps:
    print(f"Sposta il disco {step[0]} dal piolo", step[1], "al piolo", step[2])
print("Numero totale di mosse:", len(steps))

