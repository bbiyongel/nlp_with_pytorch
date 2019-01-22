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

    f = open(ref_fn, 'w')

    f.write(lines)

    f.close()

def download_image(url, fn):
    import urllib.request

    urllib.request.urlretrieve(url, fn)

def rename_images(fn):
    extract_p = re.compile(r"^.*([0-9]{2})\-[a-z_]+/([0-9]{2})\-.+\.md.*$")
    chapter_n = extract_p.sub(r'\1', fn)
    section_n = extract_p.sub(r'\2', fn)

    f = open(fn, 'r')

    cnt = 0
    for line in f:
        if line.strip() != '':
            fn_p = re.compile(r"^.*!\[.+\]\((.+)\).*$")

            if fn_p.search(line):
                img_fn = fn_p.sub(r'\1', line.strip())

                cnt += 1

                to_fn = '%s-%s-%02d' % (chapter_n, section_n, cnt)
                print('%s\t%s\t%s' % (fn, img_fn, to_fn))

    f.close()

    return cnt

def rename_images_all():
    ref_fn = 'SUMMARY.md'

    f = open(ref_fn, 'r')

    cnt = 0
    for line in f:
        if line.strip() != '':
            fn_p = re.compile(r"^.+([0-9]{2}\-[a-z_]+)/([0-9]{2}\-[a-z0-9\-_]+.md).+$")

            if fn_p.search(line):
                fn = fn_p.sub(r'\1/\2', line.strip())

                cnt += rename_images(fn)

    f.close()

    print(cnt)

if __name__ == '__main__':
    rename_images_all()