def area(base, altezza):
    area = base * altezza
    return area
    # return base*altezza
base = float(input("Inserisci la base: "))
altezza = float(input("Inserisci l'altezza: "))
print(f"L'area è: {area(base, altezza)}")