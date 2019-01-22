import sys, os, re

def rename():
    ref_fn = 'SUMMARY.md'
    lines = []

    f = open(ref_fn, 'r')

    cnt = 0
    for line in f:
        if line.strip() != '':
            start_p = re.compile(r"^.+([0-9]{2}\-[a-z_]+/cover.md).+$")
            fn_p = re.compile(r"^.+([0-9]{2}\-[a-z_]+)/([a-z0-9\-_]+.md).+$")
            line_p = re.compile(r"^(.+)([0-9]{2}\-[a-z_]+)/([a-z0-9\-_]+.md)(.+)$")

            if start_p.search(line):
                cnt = 0

            if fn_p.search(line):
                from_fn = fn_p.sub(r'\1/\2', line.strip())
                to_fn = fn_p.sub(r'\1/%02d-\2' % cnt, line.strip())
                
                '''
                print('%s\t%s' % (from_fn, to_fn))
                try:
                    os.rename(from_fn, to_fn)
                except:
                    pass
                    '''

                to_line = line_p.sub(r'\1\2/%02d-\3\4' % cnt, line)
                #print(to_line)
                lines += [to_line]
                
                cnt += 1
            else:
                lines += [line]
        else:
            lines += [line]

    f.close()

    lines = ''.join(lines)
    print(lines)

if __name__ == '__main__':
    rename()