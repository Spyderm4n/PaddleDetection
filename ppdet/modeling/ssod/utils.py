#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn.functional as F


def align_weak_strong_shape(data_weak, data_strong):
    max_shape_x = max(data_strong['image'].shape[2],
                      data_weak['image'].shape[2])
    max_shape_y = max(data_strong['image'].shape[3],
                      data_weak['image'].shape[3])

    scale_x_s = max_shape_x / data_strong['image'].shape[2]
    scale_y_s = max_shape_y / data_strong['image'].shape[3]
    scale_x_w = max_shape_x / data_weak['image'].shape[2]
    scale_y_w = max_shape_y / data_weak['image'].shape[3]
    target_size = [max_shape_x, max_shape_y]

    if scale_x_s != 1 or scale_y_s != 1:
        data_strong['image'] = F.interpolate(
            data_strong['image'],
            size=target_size,
            mode='bilinear',
            align_corners=False)
        if 'gt_bbox' in data_strong:
            gt_bboxes = data_strong['gt_bbox'].numpy()
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x_s
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y_s
            data_strong['gt_bbox'] = paddle.to_tensor(gt_bboxes)

    if scale_x_w != 1 or scale_y_w != 1:
        data_weak['image'] = F.interpolate(
            data_weak['image'],
            size=target_size,
            mode='bilinear',
            align_corners=False)
        if 'gt_bbox' in data_weak:
            gt_bboxes = data_weak['gt_bbox'].numpy()
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x_w
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y_w
            data_weak['gt_bbox'] = paddle.to_tensor(gt_bboxes)
    return data_weak, data_strong


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    pt = pred_sigmoid
    zerolabel = paddle.zeros_like(pt)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos],
        reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss


def filter_invalid(bbox, label=None, score=None, thr=0.0, min_size=0):
    if score.numel() > 0:
        soft_score = score.max(-1)
        valid = soft_score >= thr
        bbox = bbox[valid]

        if label is not None:
            label = label[valid]
        score = score[valid]
    if min_size is not None and bbox.shape[0] > 0:
        bw = bbox[:, 2]
        bh = bbox[:, 3]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]

        if label is not None:
            label = label[valid]
            score = score[valid]

    return bbox, label, score


def picodet_pseudo_gt_from_teacher(teacher_out, data_strong, semi_cfg):
    """Convert PicoDet teacher NMS predictions to pseudo-GT for student training.

    The teacher runs in eval mode on weak-augmented unlabeled images and
    produces bbox predictions in [class_id, score, x1, y1, x2, y2] format
    (after NMS, in input-image coordinate space when scale_factor=1).
    This function filters those predictions and injects them as GT into the
    strong-augmented unlabeled batch so the student can be trained with the
    standard PicoDet supervised loss.

    Args:
        teacher_out (dict): Output of teacher model in eval mode.
            - 'bbox': Tensor [N_total, 6], each row: [class_id, score, x1, y1, x2, y2]
            - 'bbox_num': Tensor [B], number of boxes per image
        data_strong (dict): Strong-augmented unlabeled batch (will be copied).
        semi_cfg (dict): Semi-supervised config with keys:
            - pseudo_score_thr (float): Confidence threshold. Default 0.4.
            - min_box_size (int): Minimum box side length in pixels. Default 8.
            - max_pseudo_num (int): Max pseudo boxes per image. Default 20.
            - allowed_target_classes (list|None): Allowed class ids (0-based).

    Returns:
        dict: A new batch dict with pseudo gt_bbox, gt_class, pad_gt_mask
              suitable for PicoDet's get_loss().
    """
    score_thr = semi_cfg.get('pseudo_score_thr', 0.4)
    min_box_size = semi_cfg.get('min_box_size', 8)
    max_pseudo_num = semi_cfg.get('max_pseudo_num', 20)
    allowed_classes = semi_cfg.get('allowed_target_classes', None)

    bboxes = teacher_out['bbox']  # (N_total, 6): [class, score, x1, y1, x2, y2]
    bbox_num = teacher_out['bbox_num']  # (B,)
    batch_size = int(bbox_num.shape[0])

    # Convert bbox_num to a plain list once to avoid repeated CPU-GPU syncs
    bbox_num_list = bbox_num.numpy().tolist()

    all_gt_bbox = []
    all_gt_class = []
    all_valid_num = []
    max_gt_num = 0

    offset = 0
    for i in range(batch_size):
        n = int(bbox_num_list[i])
        if n > 0:
            img_bboxes = bboxes[offset:offset + n]  # (n, 6)
            cls_ids = img_bboxes[:, 0]              # (n,)
            scores = img_bboxes[:, 1]               # (n,)
            boxes = img_bboxes[:, 2:]               # (n, 4): [x1, y1, x2, y2]

            # Filter by confidence score
            valid_mask = scores >= score_thr

            # Filter by minimum box size
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            size_mask = (bw >= min_box_size) & (bh >= min_box_size)
            valid_mask = valid_mask & size_mask

            # Filter by allowed class IDs if specified (vectorised)
            if allowed_classes is not None:
                allowed_t = paddle.to_tensor(
                    allowed_classes, dtype=cls_ids.dtype)
                # cls_ids (n,) vs allowed_t (k,) -> broadcast comparison
                class_mask = (cls_ids.unsqueeze(1) == allowed_t.unsqueeze(0)
                              ).any(axis=1)
                valid_mask = valid_mask & class_mask

            boxes = boxes[valid_mask]
            cls_ids = cls_ids[valid_mask]

            # Limit to max_pseudo_num (keep top-scoring)
            if boxes.shape[0] > max_pseudo_num:
                valid_scores = scores[valid_mask]
                k = min(max_pseudo_num, valid_scores.shape[0])
                _, topk_idx = paddle.topk(valid_scores, k)
                boxes = boxes[topk_idx]
                cls_ids = cls_ids[topk_idx]
        else:
            boxes = paddle.zeros([0, 4], dtype='float32')
            cls_ids = paddle.zeros([0], dtype='float32')

        all_gt_bbox.append(boxes)
        all_gt_class.append(cls_ids)
        num_valid = boxes.shape[0]
        all_valid_num.append(num_valid)
        if num_valid > max_gt_num:
            max_gt_num = num_valid
        offset += n

    # Ensure at least 1 slot to avoid zero-dim tensors when no pseudo-labels
    # are produced for the entire batch.
    if max_gt_num == 0:
        import logging
        logging.getLogger('ppdet.engine').debug(
            "picodet_pseudo_gt_from_teacher: no pseudo-labels in this batch "
            "(score_thr=%.2f). Student will see zero-GT batch.", score_thr)
    max_gt_num = max(max_gt_num, 1)

    # Build padded GT tensors compatible with PicoDet's get_loss()
    gt_bbox = paddle.zeros([batch_size, max_gt_num, 4], dtype='float32')
    gt_class = paddle.zeros([batch_size, max_gt_num, 1], dtype='int32')
    pad_gt_mask = paddle.zeros([batch_size, max_gt_num, 1], dtype='float32')

    for i in range(batch_size):
        n = all_valid_num[i]
        if n > 0:
            gt_bbox[i, :n] = all_gt_bbox[i]
            gt_class[i, :n, 0] = all_gt_class[i].cast('int32')
            pad_gt_mask[i, :n] = 1.0

    # Copy the strong batch and inject pseudo-GT
    pseudo_batch = dict(data_strong)
    pseudo_batch['gt_bbox'] = gt_bbox
    pseudo_batch['gt_class'] = gt_class
    pseudo_batch['pad_gt_mask'] = pad_gt_mask
    # Drop gt_score if present; pseudo labels use hard assignment
    pseudo_batch.pop('gt_score', None)

    return pseudo_batch
