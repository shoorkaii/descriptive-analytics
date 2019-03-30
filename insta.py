import json
import urllib.request as urllib2

url_recent_media = 'https://www.instagram.com/%s/' % 'reza_ranjkesh'
response = urllib2.urlopen(url_recent_media)

insta_html = response.read()
insta_html_split = insta_html.split('"ProfilePage":[')
if len(insta_html_split) > 1:
    insta_html_split_2 = insta_html_split[1].split(']},"gatekeepers"')
    if len(insta_html_split_2) > 1:
        json_dict = json.loads(insta_html_split_2[0])
