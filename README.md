# NDVC

 
```
/DB/videoCopyDetection(@\\mlsun.sogang.ac.kr\data\VCDB`)
|-- annotation
|   |-- baggio_penalty_1994.txt
|   `-- ...
|-- background_dataset
|   |-- bg_meta.txt
|   |-- frames 
|   `-- videos
`-- core_dataset
    |-- core_meta.txt
    |-- frames
    `-- videos

```

```
NDVC
|-- README.md
|-- dataset
|   `-- vcdb3.py 
|-- eval.py 
|-- models
|   |-- Triplet.py
|   |-- lossses.py
|   |-- nets.py
|   |-- pooling.py
|   `-- utils.py
`-- utils
    |-- Period.py
    |-- TemporalNetwork.py
    |-- __init__.py
    |-- __pycache__
    |-- extract_frames.py
    |-- replace_meta_txt.py
    `-- utils.py

```
| Package | Version | 
| -------|:-------------:|
| torch     |  1.1.0  |
| torchvision |  0.2.2 |
| torchsummary | 1.5.1|



```
eval.py
1. extract DB FingerPrint
2. while Query:
    2.1 extract query FingerPrint 
    2.2 find Similarity query - reference
    2.3 Temporal Network
    2.4 matching with ground Truth
```



