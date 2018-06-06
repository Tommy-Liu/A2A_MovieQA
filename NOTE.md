### Video data
* Python dictionary:
```python
video_data = \
{
    'video_base_name': 
    {
        'avail': True/False,
        'num_frames': 0,
        'image_size': [],
        'fps': 0.0,
        'duration': 0.0,
    }
    # ... '': {}
}
```
### Shot boundary
```python
shot_boundary = {
    'video_base_name': [],
    # ... '' : []
}
```
### Subtitle data
* Python dictionary:
```python
video_subtitle = {
    'video_base_name': [[]],
    # ... '': [[]]
}
video_subtitle_shot = {
    'video_base_name': [],
    # ... '': []
}
```
### Frame time
From matidx.
```python
frame_time = {
    'video_base_name':[]
    # ... '': []
}
```
### Total QA data
* Python dictionary:
```python
qa_list = [
    {
        "qid": '',
        "question": '',
        "answers": [],
        "imdb_key": '',
        "correct_index": 0,
        "mv+sub": bool,
        "video_clips": []
    }
    # ...: {}
]
total_qa = {
        'train': qa_list,
        'test': qa_list,
        'val': qa_list,
    }
```
### Tokenize QA data
```python
tokenize_qa = [
    {
        'tokenize_question': [],
        'tokenize_answer': [[]],
        'video_clips': [],
        'correct_index': 0
    }
    # ...: {}
]
```
### Encoded QA data 
```python
encode_qa = [
    {
        'encoded_answer': [[]],
        'encoded_question': [],
        'video_clips': [],
        'correct_index': 0
    }
    # ...: {}
]
encode_sub = {
    "video_base_name":{
        'subtitle': [[]],
        'subtitle_shot': [],
        'shot_boundary': [],
    }
    # ... '': {}
}
```

### Feature resolve 
1. Feature list
    1. 3-d array
    2. [[[]]]
2. Feature
    1. 2-d array
    2. 1-d array
    3. [[]]
    4. []