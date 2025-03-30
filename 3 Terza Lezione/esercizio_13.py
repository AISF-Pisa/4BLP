def esa_dec(numero_esadecimale):
    '''oufyiyfc
    '''
    numero_decimale = 0
    lunghezza = len(numero_esadecimale)
    for i in range(lunghezza):
        cifra = numero_esadecimale[lunghezza - 1 - i]
        if '0' <= cifra <= '9':
            valore = ord(cifra) - ord('0')
        elif 'A' <= cifra <= 'F':
            valore = ord(cifra) - ord('A') + 10
        elif 'a' <= cifra <= 'f':
            valore = ord(cifra) - ord('a') + 10
        else:
            raise ValueError("Carattere non valido nell'input esadecimale")
        numero_decimale += valore * (16 ** i)
    return numero_decimale

# Esempio di utilizzo
numero_esadecimale = "1A3F"
numero_decimale = esa_dec(numero_esadecimale)
print(f"Il numero esadecimale {numero_esadecimale} in decimale è {numero_decimale}")

def dec_esa(numero_decimale):
    '''oiguiyyv
    '''
    if numero_decimale == 0:
        return "0"
    
    cifre_esadecimali = "0123456789ABCDEF"
    esadecimale = ""
    
    while numero_decimale > 0:
        resto = numero_decimale % 16
        esadecimale = cifre_esadecimali[resto] + esadecimale
        numero_decimale = numero_decimale // 16
        
    return esadecimale

# Esempio di utilizzo
numero_decimale = 158
numero_esadecimale = dec_esa(numero_decimale)
print(f"Il numero decimale {numero_decimale} in esadecimale è {numero_esadecimale}")
