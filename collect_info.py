import sys
import glob

def main(input_fn, keyword):
    f = open(input_fn, 'r')

    for line in f:
        if line.strip() != '':
            line = line.strip()
            if keyword in line.strip():
                print('%s\t%s' % (input_fn, line))

    f.close()

if __name__=='__main__':
    path = sys.argv[1]
    keyword = sys.argv[2]

    fns = glob.glob("./*/*.md")
    print('Found %d files.' % len(fns))
    # print('\n'.join(fns))

    for fn in fns:
        main(fn, keyword)
