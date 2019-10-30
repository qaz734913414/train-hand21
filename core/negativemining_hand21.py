import mxnet as mx
import numpy as np
from config import config

class NegativeMiningOperator_Hand21(mx.operator.CustomOp):
    def __init__(self, landmark_ohem=config.LANDMARK_OHEM, landmark_ohem_ratio=config.LANDMARK_OHEM_RATIO):
        super(NegativeMiningOperator_Hand21, self).__init__()
        self.landmark_ohem = landmark_ohem
        self.landmark_ohem_ratio = landmark_ohem_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        landmark_pred = in_data[0].asnumpy() # batchsize x 42
        landmark_target = in_data[1].asnumpy() # batchsize x 42
        landmark_vis = in_data[2].asnumpy() # batchsize x 42
        
        # landmark
        self.assign(out_data[0], req[0], in_data[0])
        landmark_keep = landmark_vis > 0
        self.assign(out_data[1], req[1], mx.nd.array(landmark_keep))

    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        landmark_pred = in_data[0].asnumpy() # batchsize x 42
        landmark_target = in_data[1].asnumpy() # batchsize x 42
        landmark_vis = in_data[2].asnumpy() # batchsize x 42
        
        landmark_keep = out_data[1].asnumpy()
        
        num = landmark_keep.shape[0]
        landmark_grad = 2*(landmark_pred - landmark_target) / num
        landmark_grad *= landmark_keep
        #print(landmark_grad)
        self.assign(in_grad[0], req[0], mx.nd.array(landmark_grad))
    

@mx.operator.register("negativemining_hand21")
class NegativeMiningProp_Hand21(mx.operator.CustomOpProp):
    def __init__(self):
        super(NegativeMiningProp_Hand21, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['landmark_pred', 'landmark_target', 'landmark_vis']

    def list_outputs(self):
        return ['landmark_out', 'landmark_keep']

    def infer_shape(self, in_shape):
        #print(in_shape)
        keep_shape = in_shape[0]
        return in_shape, [in_shape[0], keep_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return NegativeMiningOperator_Human14()
