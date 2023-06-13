
# การรวบรวมและทำความสะอาดข้อมูล

ก่อนอื่นติดตั้ง Package ที่ต้องใช้ก่อน
``` python
!pip install -q datasets
```
จากนั้นก็โหลดมา
``` python
from datasets import load_dataset
datafile = {'train':'unified_abstract_infill.jsonl'}
data = load_dataset('laion/OIG',data_files=datafile)
print(data)
```
ทีนี่ก็เลือก feature ที่เราต้องการ สามารถอ่านเพิ่มเติมได้ที่ (Medium)[]
``` python
data = data['train'].filter(lambda x: x['text'].startswith('Background:'))
print(data)
```
จากนั้นก็ทำความสะอาดข้อมูล
``` python
import numpy as np
import re
```
``` python
def setup(data):
  conversation = data['text'].split('<human>:',1)
  conversation[1] = '<human>:' + conversation[1]
  conversation[0] = conversation[0][11:]
  conversation[0] = re.sub('😃🧘😂🌍🍞🚗📞🎉❤🍆👨👩👧','',conversation[0])
  conversation[1] = x = re.sub('http.+?', '',conversation[1] )
  return {'Background:':conversation[0],
          '<human>:_<bot>:':conversation[1]
          }
```
``` python
data = data.map(setup)
print(data)
```
save file เก็บไว้สำหรับเอาไปเทรนทีหลัง
``` python
data.to_json('output.jsonl')
```

