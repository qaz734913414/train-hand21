import mxnet as mx
import numpy as np
from config import config


class LANDMARK_MSE(mx.metric.EvalMetric):
    def __init__(self):
        super(LANDMARK_MSE, self).__init__('lmL2')

    def update(self,labels, preds):
        # output: landmark_pred_output, landmark_keep_inds
        # label: landmark_target, landmark_vis
        pred_delta = preds[0].asnumpy()
        landmark_target = labels[0].asnumpy()
        vis = labels[1].asnumpy()
        
        vis = vis > 0
        e = (pred_delta - landmark_target)**2
        e = e*vis
        error = np.sum(e)
        size = np.sum(vis)
        self.sum_metric += error
        self.num_inst += size

class LANDMARK_L1(mx.metric.EvalMetric):
    def __init__(self):
        super(LANDMARK_L1, self).__init__('lmL1')

    def update(self,labels, preds):
        # output: landmark_pred_output, landmark_keep_inds
        # label: landmark_target, landmark_vis
        pred_delta = preds[0].asnumpy()
        landmark_target = labels[0].asnumpy()
        vis = labels[1].asnumpy()
        
        vis = vis > 0
        e = abs(pred_delta - landmark_target)
        e = e*vis
        error = np.sum(e)
        size = np.sum(vis)
        self.sum_metric += error
        self.num_inst += size
		