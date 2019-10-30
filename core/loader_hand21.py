import mxnet as mx
import numpy as np
import minibatch_hand21 as minibatch

class ImageLoader(mx.io.DataIter):
    def __init__(self, imdb, im_size, batch_size, thread_num, flip=True, shuffle=False, ctx=None, work_load_list=None):

        super(ImageLoader, self).__init__()

        self.imdb = imdb
        self.im_size = im_size
        self.batch_size = batch_size
        self.thread_num = thread_num
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        self.cur = 0
        self.image_num = len(imdb)
        self.size = self.image_num
        self.index = np.arange(self.size)

        self.batch = None
        self.data = None
        self.label = None
		
        self.label_names = ['landmark_target','landmark_vis']
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [('data', self.data[0].shape)]
      #  return [(k, v.shape) for k, v in zip(self.data_name, self.data)]


    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)]


    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        #print cur_from,cur_to,self.index[cur_from:cur_to]
        imdb = []
        for i in range(cur_from,cur_to):
            idx = self.index[i]
            annotation = self.imdb[idx]
            imdb.append(annotation)
        
        data, label = minibatch.get_minibatch(imdb, self.im_size, self.thread_num)
        self.data = [mx.nd.array(data['data'])]
        self.label = [mx.nd.array(label[name]) for name in self.label_names]
