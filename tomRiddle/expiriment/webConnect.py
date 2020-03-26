
from urllib import request

sites = [
    "www.google.com"
]


def openpage(url):
    with request.urlopen(url) as u:
        page = u.open()
        return page
