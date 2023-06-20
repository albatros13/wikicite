import re
import tldextract

sources_mbf = [
    "mediabiasfactcheck/leastBiased.txt",
    "mediabiasfactcheck/leftBiased.txt",
    "mediabiasfactcheck/leftCenterBiased.txt",
    "mediabiasfactcheck/rightBiased.txt",
    "mediabiasfactcheck/rightCenterBiased.txt",
]

sources = [
    "newspaperindex/data-newspapers.txt",
    "newsmedialists/data-newspapers.txt",
    "newsmedialists/data-tv.txt",
    "newsmedialists/data-magazines.txt"
]

domains = set()

for source in sources_mbf:
    f = open(source, "r")
    for x in f:
        res = re.findall(r'\(.*?\)', x)
        url = res[0][1:-1] if len(res) > 0 else x
        if '.' in url:
            url_parsed = tldextract.extract(url)
            domains.add(url_parsed.domain)

for source in sources:
    f = open(source, "r")
    for x in f:
        if x.startswith('***'):
            continue
        url = x.replace('https://', '').replace('www.', '').replace('youtube.com/', '').replace('instagram.com/', '')\
            .replace('facebook.com/', '').replace('twitter.com/', '').replace('pinterest.com/', '')\
            .replace('user/', '').replace('plus.google.com/', '')
        url_parsed = tldextract.extract(url)
        if len(url_parsed.domain) > 0:
            domains.add(url_parsed.domain)

print(len(domains))

f = open("domains.txt", "w")
f.write(str(domains).replace("'", "").replace(" ", ""))
f.close()
