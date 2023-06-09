# เก็บข้อมูล

```python
from bs4 import BeautifulSoup
import requests

page = requests.get("https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Lists_of_topics")

soup = BeautifulSoup(page.content, 'html.parser')

topic = [p.find_all('a',class_="mw-redirect") for p in soup.find_all('p')]
topic = [t for top in topic for t in top]
topic = topic[3:]
print(topic)
```

```python
import pandas as pd
def scrape_topic(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  topic = soup.find_all('p')[:3]#.text.strip()
  topic = [t.text.strip() for t in topic]
  return ' '.join(topic)
data = [scrape_topic(f"https://en.wikipedia.org/{url.get('href')}") for url in topic] 
print(data)
```
จากนั้นก็มานั้งทำความสะอาดข้อมูลที่ละตัว 
สามารถดูได้ที่ script [นี้]( 
https://github.com/ohmreborn/question-generation-AIB2023/blob/main/evaluate/scrape_data_wiki.ipynb)
แล้วก็เอามาบันทึกไว้ใน folder [eval_data](https://github.com/ohmreborn/question-generation-AIB2023/tree/main/evaluate/eval_data)
โดยจะแบ่งเป็น 9 ชุด ไว้สำหรับเอาไปให้ Ai สร้างคำถามแล้วเอาไปทำแบบฟอร์ม[นี้](https://forms.gle/DJJUKEpYocycoTpC9) สามารถไปทำเพื่อช่วยผู้พัฒนาได้
