
fname="la"

tstr=[]
fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
lines = fin.readlines()
for i in range(0, len(lines), 4):
    text = lines[i].lower().strip()
    target = lines[i+1].lower().strip()
    polarity = lines[i+2].strip()
    tstr.append([text,target,polarity])

import csv
fname="la_test.csv"
fout=open(fname, 'w', encoding='utf-8', newline='\n', errors='ignore')
csv_writer=csv.writer(fout)
for l in tstr:
    csv_writer.writerow(l)