---
jupyter:
  colab:
    authorship_tag: ABX9TyOOfpMnedUUkbL28nlmyFWF
    include_colab_link: true
    private_outputs: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown colab_type="text" id="view-in-github"}
`<a href="https://colab.research.google.com/github/ohmreborn/question-generation-AIB2023/blob/main/dataset/preprocess_dataset.ipynb" target="_parent">`{=html}`<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>`{=html}`</a>`{=html}
:::

::: {.cell .code id="dCLnlLiIxJEh"}
``` python
!pip install -q datasets
```
:::

::: {.cell .code id="iqMt0BVCxNHA"}
``` python
from datasets import load_dataset
datafile = {'train':'unified_abstract_infill.jsonl'}
data = load_dataset('laion/OIG',data_files=datafile)
print(data)
```
:::

::: {.cell .code id="iDvGYOkDsWKP"}
``` python
data = data['train'].filter(lambda x: x['text'].startswith('Background:'))
print(data)
```
:::

::: {.cell .code id="Oqxu3_yoanzO"}
``` python
import numpy as np
import re
```
:::

::: {.cell .code id="Xw27y47eLw-a"}
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
:::

::: {.cell .code id="-7-dEr4itg8u"}
``` python
data = data.map(setup)
print(data)
```
:::

::: {.cell .code id="cG5z56jThbJY"}
``` python
data.to_json('output.jsonl')
```
:::

::: {.cell .code id="zxxdTuaAvR9W"}
``` python
```
:::
