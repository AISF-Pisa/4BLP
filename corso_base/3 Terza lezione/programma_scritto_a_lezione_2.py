a = int(input("Inserisci un numero: "))
if a % 2 == 0 and a != 0:
    print(f"{a} e' pari")
elif a==0:
    print(f"{a} non è né pari né dispari")
else:
    print(f"{a} è dispari")

f = lambda x: f"{x} è pari" if x % 2==0 else f"{x} è dispari"

print(f(11))
"""
and
P Q P&Q
V F F
F V F
V V V
F F F

or
P Q P | Q
V F V
F V V
F F F
V V V
"""
