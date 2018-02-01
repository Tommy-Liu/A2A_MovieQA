from time import time

import utils.data_utils as du
from config import MovieQAPath
from data.data_loader import Subtitle, QA, DataLoader
from embed.args import EmbeddingPath

mp = MovieQAPath()
ep = EmbeddingPath()


def main():
    data = DataLoader()
    subt = Subtitle()
    video_data = du.json_load(mp.video_data_file)
    # vocab = du.json_load(ep.gram_vocab_file)
    # embedding = np.load(ep.gram_embedding_vec_file)

    start_time = time()
    qa0 = QA().include(video_clips=True)
    qa = data['qa']
    print(len(qa0.get(qa)), '%.4f s' % (time() - start_time))
    start_time = time()
    qa1 = qa0.include(video_clips=[k + '.mp4' for k in video_data.keys()])
    print(len(qa1.get(qa)), '%.4f s' % (time() - start_time))
    start_time = time()
    qa2 = qa1.exclude(imdb_key=['tt1371111'])
    print(len(qa2.get(qa)), '%.4f s' % (time() - start_time))


if __name__ == '__main__':
    main()
