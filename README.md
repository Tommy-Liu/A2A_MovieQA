# MovieQA_Contest

## Video Part

### Video data
* Python dictionary:
```python
info = {
    'num_frames': 0,
    'image_size': [],
    'fps': 0.0,
    'duration': 0.0,
}
video_data = \
{
    'video_base_name': 
    {
        'avail': True/False,
        'shot_boundary': [],
        'info': info,
    }
    # ...: {}
}
```
### Subtitle data
* Python dictionary:
```python
subtitle_data = \
{
    'video_base_name': {
        'subtitle':[],
        'subtitle_index': [],
        'frame_time': [],
        'shot_boundary': [],
    }
    # ...: {}
}
```





