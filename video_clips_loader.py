from config import MovieQAConfig
from data_utils import load_json
config = MovieQAConfig()



avail_video_metadata = load_json(config.avail_video_metadata_file)

print(avail_video_metadata.keys())
# print(avail_video_metadata['list'])
print(avail_video_metadata['info'])
print(avail_video_metadata['info']['tt0816436.sf-094722.ef-095195.video'].keys())


