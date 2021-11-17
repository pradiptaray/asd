import string

kmerlen = 5

prefix = '../newdata/datamaker/'
fd = file(prefix + 'table.count')
fstr1 = ''
fstr2 = ''
i = 0
for line in fd:
    if (i == 0):
      i += 1
      continue
      
    sline = string.split(line)
    kmer = sline[0]
    fstr1 += kmer + '\n'
    hist = string.join(sline[1:], ' ')
    fstr2 += hist + '\n'
fd.close()

prefix = '../result/'
fd = file(prefix + 'kmer' + str(kmerlen) + '.txt', 'w')
fd.write(fstr1)
fd.close()

fd = file(prefix + 'kmer_hist' + str(kmerlen) + '.txt', 'w')
fd.write(fstr2)
fd.close()
    
