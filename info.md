
 
```
#(wd:/DB/CC_WEB_VIDEO/)
.
|-- GT
|-- Ground.zip
|-- Seed.txt
|-- Shot_Info.txt
|-- Video_Complete.txt
|-- Video_List.txt
|-- frame_1_per_sec         # extract frame 1-fps
|   |-- frames              # vid
|   `-- resnet50            # extract feature with resnet50 (~AdaptiveAvgPool2d-346)
|       |-- frames
|       |-- f-feature       # frame features per video 
|       |-- v-feature       # video feature per video (average frame feature)
|       `-- v-feature.pt    # all video feature (vid order)
`-- videos
```
