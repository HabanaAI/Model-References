import numpy as np

from stereo.data.mask_utils import VDPDMasks
from stereo.data.gtd_mask_utils import GTD_VDPDMasks

class UnionVDPDMasks(object):

    def __init__(self, clip, clip_name, camera_name, itrk_path, enlarge_ratio=1.1, must_have_itrk=True, must_have_gtd=True, speed_thresh_for_moving=2.0):
        try:
            self.gtd_masks = GTD_VDPDMasks(clip, clip_name, enlarge_ratio, speed_thresh_for_moving=speed_thresh_for_moving)
        except:
            if must_have_gtd:
                assert False
            else:
                self.gtd_masks = None
        try:
            self.itrk_masks = VDPDMasks(clip, camera_name, itrk_path, enlarge_ratio)
        except:
            if must_have_itrk:
                assert False
            else:
                self.itrk_masks = None
        assert not (self.gtd_masks is None and self.itrk_masks is None)

    def get_mask_image(self, gi, cam, im_sz, focal, origin, name, rectified):
        itrk_mask = None
        gtd_mask = None
        if self.itrk_masks is not None:
            itrk_mask = self.itrk_masks.get_mask_image(gi, im_sz, focal, origin, name, rectified)
        if self.gtd_masks is not None:
            gtd_mask = self.gtd_masks.get_mask_image(gi, cam, im_sz, focal, origin, name, rectified)
        if itrk_mask is not None and gtd_mask is not None:
            return np.logical_or(itrk_mask, gtd_mask).astype(itrk_mask.dtype)
        if itrk_mask is not None:
            return itrk_mask
        if gtd_mask is not None:
            return gtd_mask
        assert False