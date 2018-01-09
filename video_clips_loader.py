from config import MovieQAConfig
from utils.data_utils import json_load
config = MovieQAConfig()

avail_video_metadata = json_load(config.avail_video_metadata_file)

print(avail_video_metadata.keys())
# print(avail_video_metadata['list'])
print(avail_video_metadata['info'])
print(avail_video_metadata['info']['tt0816436.sf-094722.ef-095195.video'].keys())


