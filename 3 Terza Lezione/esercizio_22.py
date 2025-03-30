def converti_secondi(secondi):
    '''ovuv
    '''

    mm = secondi // 60
    ss = secondi %  60
    hh = mm      // 60
    mm = mm      %  60             # 2:46:40
        
    if hh < 10: hh = "0"+str(hh)  # hh <-- "0"+2 = "02" 
    if mm < 10: mm = "0"+str(mm)  # ---
    if ss < 10: ss = "0"+str(ss)  # ---
    
    time = f"{hh}:{mm}:{ss}"
    return time

print(converti_secondi(1000))