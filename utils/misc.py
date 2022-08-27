import re

def search_between(start, end, string):
    return re.search('%s(.*)%s' % (start, end), string).group(1)