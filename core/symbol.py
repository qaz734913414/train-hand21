import mxnet as mx
import core.negativemining_hand21
from config import config


bn_mom = 0.9
#bn_mom = 0.9997

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)
    act = Act(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act
    
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)    
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv    

    
def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj
    
def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity=conv+shortcut
    return identity
	
res_base_dim = 32

def L14_Net112(mode="train"):
    """
    #Proposal Network
    #input shape 3 x 112 x 112
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    landmark_vis = mx.symbol.Variable(name="landmark_vis")
    
    # data = 112X112
    # conv1 = 56X56
    conv1 = Conv(data, num_filter=res_base_dim, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1")
    conv2 = Residual(conv1, num_block=1, num_out= res_base_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim, name="res2")
    
	#conv23 = 28X28
    conv23 = DResidual(conv2, num_out=res_base_dim*2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*2, name="dconv23")
    conv3 = Residual(conv23, num_block=2, num_out=res_base_dim*2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*2, name="res3")
    
	#conv34 = 14X14
    conv34 = DResidual(conv3, num_out=res_base_dim*4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*4, name="dconv34")
    conv4 = Residual(conv34, num_block=3, num_out=res_base_dim*4, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*4, name="res4")
    
	#conv45 = 7X7
    conv45 = DResidual(conv4, num_out=res_base_dim*8, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*8, name="dconv45")
    conv5 = Residual(conv45, num_block=2, num_out=res_base_dim*8, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*8, name="res5")
    
	# conv6 = 1x1
    conv6 = Conv(conv5, num_filter=res_base_dim*8, kernel=(7, 7), pad=(0, 0), stride=(1, 1), name="conv6")
    fc1 = Conv(conv6, num_filter=res_base_dim*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="fc1")
    fc2 = Conv(fc1, num_filter=res_base_dim*32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="fc2")	
    conv6_3 = mx.symbol.FullyConnected(data=fc2, num_hidden=42, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
	
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        out = mx.symbol.Custom(landmark_vis = landmark_vis, landmark_pred=bn6_3, landmark_target=landmark_target, 
                            op_type='negativemining_hand21', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	