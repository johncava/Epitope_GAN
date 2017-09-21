import random

prot_set = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F',
            'P','S','T','W','Y','V']

def create_epitope():
    epitope = ''
    for aa in xrange(20):
        epitope += random.choice(prot_set)
    print epitope

create_epitope()