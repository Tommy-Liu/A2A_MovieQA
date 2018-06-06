# MovieQA_Contest

## Installation Guidance

1. Python Version: 3.6
2. Required Package: tensorflow:1.7.0, imageio, numpy, scipy, pillow
3. Git clone MovieQA_benchmark from <a href="https://github.com/makarandtapaswi/MovieQA_benchmark">github</a>, and change the path of ```MovieQAPath.benchmark_dir``` to the path where you clone to.
4. Download Faster-RCNN pretrained model from <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">model zoo</a> and change the path of ```MovieQAPath.faster_rcnn_graph``` to the path where you download to.
5. Download all data from MovieQA to MovieQA_benchmark. (Note: you have to register first.)

## Running Guidance

* Extract all frames from video clips. It will store all frames into```MovieQAPath.image_dir```.
```
python -m process.video
| [--check] [--no_extract]
```
* Prepare GloVe embedding [<a href="http://nlp.stanford.edu/data/glove.840B.300d.zip">link</a>] to the destination in ```./embed/args.py```, and move current directory to ```./embed```. Then, type: (Note: please refer to ```./embed/args.py``` for more information.)
```
python data.py
| [--debug]
python train.py
python deploy.py
```

* Process all sentences in MovieQA, including tokenizing, generating sentence embedding and sampling frames.
```
python -m process.text_v3 --one
```
* Extract bounding box feature.
```
python extract_bbox.py
```
* You can train now. If you want to use different model or tune with different hyper-parameters, you can follow: (Note: please refer to ```train.py``` to get more information about flags)
```
python train.py --mode subt+feat --mod model_full.1 --hp 02
```

