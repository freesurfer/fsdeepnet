import os
import glob
import numpy as np
import surfa as sf

class Evaluation:
    """
    This class performs dice evaluation between ground truth and predicted label maps.

    Attributes
    ----------

    Methods
    -------
    evaluate
        Perform dice evaluation between ground truth and segmentation directories

    evaluate_oneseg
        Compute dice score between single predicted segmentation and its ground truth label map
    """
        
    def __init__(self, labels_segmentation):

        """
        Evaluation Constructor.
        """

        self._labels_segmentation, self._unique_idx = np.unique(labels_segmentation, return_index=True)

        
    def evaluate(self, gt_folder, eval_folder, evaluation_labels=None, path_dice=None):

        """
        Perform dice evaluation between ground truth and segmentation directories, 
        where scores will be saved as a numpy array if 'path_dice' is provided.

        This numpy array will be organised as follows: rows correspond to structures, and columns to subjects.
        Importantly, rows are given in a sorted order.

        Example: we segment 2 subjects, where output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
                                    so sorted output_labels = [0, 2, 3, 4, 17, 41, 42, 43, 53]
        dice = [[xxx, xxx],  # scores for label 0
                [xxx, xxx],  # scores for label 2
                [xxx, xxx],  # scores for label 3
                [xxx, xxx],  # scores for label 4
                [xxx, xxx],  # scores for label 17
                [xxx, xxx],  # scores for label 41
                [xxx, xxx],  # scores for label 42
                [xxx, xxx],  # scores for label 43
                [xxx, xxx]]  # scores for label 53
                /       \
          subject 1    subject 2


        Parameters
        ----------
        gt_folder  : string
            path of the ground truth label map directory
        eval_foler : string
            path of the predicted segmentation directory
        evaluation_labels: 1d numpy array
            (optional) a subset of the segmentation labels to be evaluated
                       Default is np.unique(segmentation_labels).
        """

        # check inputs
        assert gt_folder is not None, 'please specify the ground truth (folder)'
        assert eval_folder is not None, 'please specify the segmentation (folder)'

        if evaluation_labels is None:
            evaluation_labels = self._labels_segmentation
        n_labels = len(evaluation_labels)
        
        # convert path to absolute paths
        gt_folder = os.path.abspath(gt_folder)
        eval_folder = os.path.abspath(eval_folder)
        if (os.path.isdir(gt_folder)):
            assert os.path.isdir(eval_folder), 'both %s and %s need to be directory' % (gt_folder, eval_folder)
            # get *.nii.gz, *.nii, *.mgz in the directory
            path_gt_labels = sorted(glob.glob(os.path.join(gt_folder, '*.nii.gz')) +
                                 glob.glob(os.path.join(gt_folder, '*.nii')) +
                                 glob.glob(os.path.join(gt_folder, '*.mgz')))
            path_segs = sorted(glob.glob(os.path.join(eval_folder, '*.nii.gz')) +
                                 glob.glob(os.path.join(eval_folder, '*.nii')) +
                                 glob.glob(os.path.join(eval_folder, '*.mgz')))            
            """
            # get all files in the directory
            path_gt_labels = sorted(glob.glob(os.path.join(gt_folder, '*')))
            path_segs = sorted(glob.glob(os.path.join(eval_folder, '*')))
            path_segs = path_segs = [x for x in path_segs if os.path.isfile(x)]
            """
            assert len(path_gt_labels) == len(path_segs), 'gt and segmentation folders must have the same amount of label maps.'
        else:
            # single ground truth/segmentation
            assert os.path.isfile(gt_folder), 'file does not exist: %s\n' % gt_folder
            assert os.path.isfile(eval_folder), 'file does not exist: %s\n' % eval_folder
            
            path_gt_labels = [gt_folder]
            path_segs = [eval_folder]

        # compute evaluation metrics
        dice_coefs = np.zeros((n_labels, len(path_segs)))            
        for idx, (path_gt, path_seg) in enumerate(zip(path_gt_labels, path_segs)):
            dice_coefs_seg = self.evaluate_oneseg(path_gt, path_seg,
                                                  evaluation_labels=evaluation_labels)
            dice_coefs[:n_labels, idx]   = np.transpose(dice_coefs_seg)
            
        # write results
        if (path_dice is None):
            path_dice = os.path.join(eval_folder, 'evaluation_dices.npy')

        os.makedirs(os.path.dirname(path_dice), exist_ok=True)
        np.save(path_dice, dice_coefs)
        path_dicedat = os.path.join(os.path.dirname(path_dice),os.path.splitext(os.path.basename(path_dice))[0])+'.dat'
        np.savetxt(path_dicedat, dice_coefs)
        print(f"\noutput evaluation dices as {path_dice} and {path_dicedat}")


    # evaluate single segmentation and its ground truth
    def evaluate_oneseg(self, path_gt, path_seg, evaluation_labels=None):

        """
        Compute dice score between ground truth label map and predicted segmentation.

        Parameters
        ----------
        path_gt  : string
            path of single ground truth label map
        path_seg : string
            path of sigle predicted segmentation

        Returns
        -------
        dice_coefs : numpy array (n_labels x 1)
            dice coefs for each label
        """        

        if evaluation_labels is None:
            evaluation_labels = self._labels_segmentation
        n_labels = len(evaluation_labels)
        
        # load gt labels and segmentation
        gt_labels = sf.load_volume(path_gt).reorient('RAS').data
        seg = sf.load_volume(path_seg).reorient('RAS').data

        # initialise result matrices
        dice_coefs = np.zeros(n_labels)
            
        # compute Dice scores
        dice_coefs = fast_dice(gt_labels, seg, evaluation_labels)
        dice_coefs2 = dice_coeffs(gt_labels, seg, evaluation_labels)
        #dice_coefs3 = dice_coeffs2(gt_labels, seg, evaluation_labels)

        return dice_coefs
            


def dice_coeffs(gt, pred, labels):
    """
    calculate dice coefficient between ground truth and prediction segmentations for each label
    """
    
    assert gt.shape == pred.shape, 'both inputs should have same size, had {} and {}'.format(gt.shape, pred.shape)

    nlabels = len(np.unique(labels))
    dice_scores = np.zeros(nlabels)
    
    for idx, label in enumerate(labels):
        # convert ground truth and predicted segmentations to binary masks        
        gt_mask = np.array(gt == label).astype(int)
        pred_mask = np.array(pred == label).astype(int)

        intersect = np.sum(pred_mask * gt_mask)
        total_sum = np.sum(pred_mask) + np.sum(gt_mask)
        dice_scores[idx] = 2 * intersect / (total_sum + 1e-5)
        
    return dice_scores


"""
# not working
def dice_coeffs2(gt, pred, labels):
    #calculate dice coefficient between ground truth and prediction segmentations for each label
    
    assert gt.shape == pred.shape, 'both inputs should have same size, had {} and {}'.format(gt.shape, pred.shape)

    labels_sorted = np.sort(labels)
    #nlabels = len(labels_sorted)
    bins = np.sort(np.concatenate([labels_sorted - 0.1, labels_sorted + 0.1]))

    histo = np.histogram2d(gt.flatten(), pred.flatten(), bins=bins)[0]
    idx = np.arange(start=1, stop=2 * len(labels_sorted), step=2)
    dice_scores = 2 * np.diag(histo)[idx] / (np.sum(histo, 0)[idx] + np.sum(histo, 1)[idx] + 1e-5)
        
    return dice_scores
"""


def fast_dice(x, y, labels):
    """Fast implementation of Dice scores.
    :param x: input label map
    :param y: input label map of the same size as x
    :param labels: numpy array of labels to evaluate on
    :return: numpy array with Dice scores in the same order as labels.
    """

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)

    if len(labels) > 1:
        # sort labels
        labels_sorted = np.sort(labels)

        # build bins for histograms
        label_edges = np.sort(np.concatenate([labels_sorted - 0.1, labels_sorted + 0.1]))
        label_edges = np.insert(label_edges, [0, len(label_edges)], [labels_sorted[0] - 0.1, labels_sorted[-1] + 0.1])

        # compute Dice and re-arrange scores in initial order
        hst = np.histogram2d(x.flatten(), y.flatten(), bins=label_edges)[0]
        idx = np.arange(start=1, stop=2 * len(labels_sorted), step=2)
        dice_score = 2 * np.diag(hst)[idx] / (np.sum(hst, 0)[idx] + np.sum(hst, 1)[idx] + 1e-5)
        dice_score = dice_score[np.searchsorted(labels_sorted, labels)]

    else:
        dice_score = dice(x == labels[0], y == labels[0])

    return dice_score


