import os
import os.path as osp

import numpy as np
import tensorflow as tf

FLAGS2 = {}
temps = {}

######################
# General Parameters #
######################
temps['rng_seed']= 3 #"Tensorflow seed for reproducibility")
FLAGS2["pixel_means"] = np.array([[[102.9801, 115.9465, 122.7717]]])

######################
# Network Parameters #
######################
temps['net']= "vgg_16"#"The network to be used as backbone")

#######################
# Training Parameters #
#######################
temps['weight_decay']= 0.0005# "Weight decay, for regularization")
temps['learning_rate']= 0.001#"Learning rate")
temps['momentum']= 0.9#, "Momentum")
temps['gamma']=0.1#, "Factor for reducing the learning rate")

temps['batch_size']= 256#, "Network batch size during training")
temps['max_iters']= 3000#, "Max iteration")
temps['step_size']= 30000#, "Step size for reducing the learning rate, currently only support one step")
temps['display']= 50#, "Iteration intervals for showing the loss during training, on command line interface")

temps['initializer']= "truncated"#, "Network initialization parameters")
temps['pretrained_model']= "./data/imagenet_weights/vgg_16.ckpt"#, "Pretrained network weights")

temps['bias_decay']= False# "Whether to have weight decay on bias as well")
temps['double_bias']= True#, "Whether to double the learning rate for bias")
temps['use_all_gt']= True#, "Whether to use all ground truth bounding boxes for training, " "For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''")
temps['max_size']= 1000#, "Max pixel size of the longest side of a scaled input image")
temps['test_max_size']= 1000#, "Max pixel size of the longest side of a scaled input image")
temps['ims_per_batch']= 1#, "Images to use per minibatch")
temps['snapshot_iterations']= 2000#, "Iteration to take snapshot")

FLAGS2["scales"] = (600,)
FLAGS2["test_scales"] = (600,)

######################
# Testing Parameters #
######################
temps['test_mode']= "top"#, "Test mode for bbox proposal")  # nms, top

##################
# RPN Parameters #
##################
temps['rpn_negative_overlap']= 0.3#, "IOU < thresh: negative example")
temps['rpn_positive_overlap']= 0.7#, "IOU >= thresh: positive example")
temps['rpn_fg_fraction']= 0.5#, "Max number of foreground examples")
temps['rpn_train_nms_thresh']= 0.7#, "NMS threshold used on RPN proposals")
temps['rpn_test_nms_thresh']= 0.7#, "NMS threshold used on RPN proposals")

temps['rpn_train_pre_nms_top_n']= 20000# "Number of top scoring boxes to keep before apply NMS to RPN proposals")
temps['rpn_train_post_nms_top_n']= 8000# "Number of top scoring boxes to keep before apply NMS to RPN proposals")
temps['rpn_test_pre_nms_top_n']= 8000# "Number of top scoring boxes to keep before apply NMS to RPN proposals")
temps['rpn_test_post_nms_top_n']= 500# "Number of top scoring boxes to keep before apply NMS to RPN proposals")
temps['rpn_batchsize']= 256# "Total number of examples")
temps['rpn_positive_weight']= -1#'Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p).'
                            #'Set to -1.0 to use uniform example weighting')
temps['rpn_top_n']= 300# "Only useful when TEST.MODE is 'top', specifies the number of top proposals to select")

temps['rpn_clobber_positives']= False# "If an anchor satisfied by positive and negative conditions set to negative")

#######################
# Proposal Parameters #
#######################
temps['proposal_fg_fraction']= 0.25#, "Fraction of minibatch that is labeled foreground (i.e. class > 0)")
temps['proposal_use_gt']= False#, "Whether to add ground truth boxes to the pool when sampling regions")

###########################
# Bounding Box Parameters #
###########################
temps['roi_fg_threshold']= 0.5#, "Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)")
temps['roi_bg_threshold_high']= 0.5#, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")
temps['roi_bg_threshold_low']= 0.1#, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")

temps['bbox_normalize_targets_precomputed']= True#, "# Normalize the targets using 'precomputed' (or made up) means and stdevs (BBOX_NORMALIZE_TARGETS must also be True)")
temps['test_bbox_reg']=True#, "Test using bounding-box regressors")

FLAGS2["bbox_inside_weights"] = (1.0, 1.0, 1.0, 1.0)
FLAGS2["bbox_normalize_means"] = (0.0, 0.0, 0.0, 0.0)
FLAGS2["bbox_normalize_stds"] = (0.1, 0.1, 0.1, 0.1)


##################
# ROI Parameters #
##################
temps['roi_pooling_size']= 7#, "Size of the pooled region after RoI pooling")

######################
# Dataset Parameters #
######################
FLAGS2["root_dir"] = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
FLAGS2["data_dir"] = osp.abspath(osp.join(FLAGS2["root_dir"], 'data'))

def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(FLAGS2["root_dir"], FLAGS2["root_dir"] , 'default', imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
