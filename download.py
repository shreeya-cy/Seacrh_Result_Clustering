import requests
from langdetect import detect
import re

def is_english_with_langdetect(text):
    try:
        language = detect(text)

        return language == 'en'
    except:
        return False

c = 0
file_path = "Wikipedia_topics"
topics = []
count = 0
with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        line = line.replace("_", " ")
        if(line.startswith('!') or line == '\n'):
            continue
        line = re.sub('[!,*)@#%(&$_?.^"]-', '', line)
        if(is_english_with_langdetect(line)):
            topics.append(line)
            count+=1
        if(count > 2000):
            break

for topic in topics:
    text_config = {
        'action': 'query',
        'format': 'json',
        'titles': topic,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
    }
    text_response = requests.get('https://en.wikipedia.org/w/api.php',params=text_config).json()
    text_page = next(iter(text_response['query']['pages'].values()))
    if 'extract' not in text_page or text_page['extract'] == '':
        continue
    else:
        try:
            file1 = open(text_page['title']+".txt","w")
            file1.write(text_page['extract'])
            c+=1
            file1.close()
        except:
            continue
print(c)




