#!/bin/python

from __future__ import print_function

# command line args
import argparse
parser=argparse.ArgumentParser('produce table of crawl stats for latex paper')
parser.add_argument('--output',type=str,default='paper/table/data_summary.tex')
args = parser.parse_args()

# misc data
countries={
    'ar':'Argentina',
    'bo':'Bolivia',
    'cl':'Chile',
    'co':'Colombia',
    'cr':'Costa Rica',
    'cu':'Cuba',
    #'do':'Dominican Rep.',
    'do':'Dominican Republic',
    'ec':'Ecuador',
    'sv':'El Salvador',
    'gt':'Guatemala',
    'hn':'Honduras',
    'mx':'Mexico',
    'ni':'Nicaragua',
    'pa':'Panama',
    'py':'Paraguay',
    'pe':'Peru',
    'pr':'Puerto Rico',
    'es':'Spain',
    'uy':'Uruguay',
    've':'Venezuela',
    #'gq':'Equat. Guinea',
    'gq':'Equatorial Guinea',

    'us':'United States',
    'pt':'Portugal',
    'bz':'Belize',
    'br':'Brazil',
}

ccTLD_main=['ar','bo','cl','co','cr','cu','do','ec','sv','gt','hn','mx','ni','pa','py','pe','pr','es','uy','ve','gq']
ccTLD_other=['us','pt','bz','br']

#ccTLD_main=['gt']
#ccTLD_other=['bz']

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1000.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def lc_file_with_cache(filename):
    filename_lc=filename+'.lcr'
    try:
        with open(filename_lc,'r') as f:
            return int(f.readline())
    except:
        try:
            import gzip
            count=sum(1 for line in gzip.open(filename))
        except:
            count=sum(1 for line in open(filename))
        with open(filename_lc,'w') as f:
            f.write(str(count)+'\n')
        return count

def slurm_extract(filename):
    ret=[]
    with open(filename,'r') as f:
        for line in f:
            if line[:12] == 'total_vocab=' or line[:12] == 'total_words=':
                ret.append(int(line[12:]))
    return ret[:3]


# write table
with open(args.output,'w') as output:
    output.write('\\begin{tabular}{ll R{0.7in}R{1in}R{1in}}\n')

    output.write('\\textbf{ccTLD}~~ & \\textbf{Country} & \\textbf{Domains} & \\textbf{URLs} & \\textbf{Words} \\\\\n')
    output.write('\\hline\n')
    output.write('\Tstrut')

    total_size=0
    total_seen=0
    total_qualified=0
    total_vocab=0
    total_words=0
    total_domains=0
    def ccTLD2str(ccTLD):
        global total_size
        global total_seen
        global total_qualified
        global total_words
        global total_domains
        import os
        import gzip
        print(' processing '+ccTLD)
        size=os.stat('crawls/ccTLD.'+ccTLD+'.jl.gz').st_size
        total_size+=size
        seen=lc_file_with_cache('crawls/ccTLD.'+ccTLD+'.jobdir/requests.seen')
        total_seen+=seen
        qualified=lc_file_with_cache('crawls/ccTLD.'+ccTLD+'.jl.gz')
        total_qualified+=qualified
        vocab,words,domains=slurm_extract('slurm/make_vocab.ccTLD.'+ccTLD+'.jl.gz.out')
        total_words+=words
        total_domains+=domains
        #return '\\texttt{%s} & %s & %0.2E & %0.2E & %0.2f & $%0.1f \\times10^3$ & $%0.1f \\times10^6$ & %10d & %s\\\\\n'%(
        #return '\\texttt{~%s} & %s & $\\SI{%d}{}$ & $\\SI{%d}{}$ & %0.2f & $\\SI{%d}{}$ & $\\SI{%d}{}$ & %10d \\\\\n'%(
        #return '\\texttt{~%s} & %s & $\\SI{%d}{}$ & %0.2f & $\\SI{%d}{}$ & $\\SI{%d}{}$ & $\\SI{%d}{}$ \\\\\n'%(
        #return '\\texttt{~%s} & %s &  $\\SI{%d}{}$ & $%0.2f\\times10^6$ & $%0.2f\\times10^6$ \\\\\n'%(
        return '\\texttt{~%s} & %s &  $\\SI{%d}{}$ & $\\SI{%d}{}$ & $\SI{%d}{}$ \\\\\n'%(
            ccTLD,
            countries[ccTLD],
            domains,
            #seen/1e6,
            qualified,
            #qualified/float(seen),
            #vocab/1e6,
            words,
            #sizeof_fmt(size),
            )

    for ccTLD in sorted(ccTLD_main):
        output.write(ccTLD2str(ccTLD))

    output.write('\\arrayrulecolor{lightgray}\\hline\n\\arrayrulecolor{black}\Tstrut')

    for ccTLD in sorted(ccTLD_other):
        output.write(ccTLD2str(ccTLD))

    output.write('\\hline\n')

    #output.write('\\Tstrut\\textbf{Total} &&  $\\SI{%d}{}$ & $%0.2f\\times10^6$ & $%0.2f\\times10^6$\n'%(
    output.write('\\Tstrut\\textbf{Total} &&  $\\SI{%d}{}$ & $\\SI{%d}{}$ & $\\SI{%d}{}$\n'%(
        total_domains,
        #total_seen/1e6,
        total_qualified,
        #total_qualified/float(total_seen),
        #total_vocab,
        total_words,
        #sizeof_fmt(total_size),
        ))

    output.write('\\end{tabular}\n')
