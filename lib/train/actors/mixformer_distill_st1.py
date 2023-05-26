from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixFormerDistillStage1Actor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, net_teacher, run_score_head=False,
                 z_size_teacher=None, x_size_teacher=None, feat_sz=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head

        # distill related
        self.net_teacher = net_teacher.eval()
        self.distill_logits_loss = nn.KLDivLoss(reduction="batchmean")
        self.feat_sz = feat_sz
        self.z_size_teacher = z_size_teacher
        self.x_size_teacher = x_size_teacher

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward student
        out_dict = self.forward_pass(data)  # forward teacher
        with torch.no_grad():
            out_dict_teacher = self.forward_pass_teacher(data)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        labels = None
        if 'pred_scores' in out_dict:
            try:
                labels = data['label'].view(-1)  # (batch, ) 0 or 1
            except:
                raise Exception("Please setting proper labels for score branch.")

        # compute losses
        loss, status = self.compute_losses(out_dict, out_dict_teacher, gt_bboxes[0], labels=labels)

        return loss, status

    def forward_pass(self, data):
        out_dict = self.net(data['template_images'][0], data['template_images'][1], data['search_images'],
                               softmax=False)
        return out_dict

    def forward_pass_teacher(self, data):
        # z0 = F.interpolate(data['template_images'][0], size=(self.z_size_teacher, self.z_size_teacher), mode='bilinear', align_corners=True)
        # z1 = F.interpolate(data['template_images'][1], size=(self.z_size_teacher, self.z_size_teacher), mode='bilinear', align_corners=True)
        # x = F.interpolate(data['search_images'][0], size=(self.x_size_teacher, self.x_size_teacher), mode='bilinear', align_corners=True)
        out_dict = self.net_teacher(data['template_images'][0], data['template_images'][1], data['search_images'], softmax=True)
        # out_dict = self.net_teacher(z0, z1, x, softmax=True)
        return out_dict

    def compute_losses(self, pred_dict, pred_dict_teacher, gt_bbox, return_status=True, labels=None):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        pred_boxes_teacher = pred_dict_teacher['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec_teacher = box_cxcywh_to_xyxy(pred_boxes_teacher).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # compute ciou and iou
        try:
            ciou_loss, iou = self.objective['ciou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            ciou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        try:
            _, iou_teacher = self.objective['ciou'](pred_boxes_vec_teacher, gt_boxes_vec)
        except:
            iou_teacher = torch.tensor(0.0).cuda()

        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # compute distillation loss
        distill_loss_logits = self.compute_losses_distill(pred_dict, pred_dict_teacher)

        # weighted sum
        loss = self.loss_weight['ciou'] * ciou_loss + self.loss_weight['l1'] * l1_loss + \
               self.loss_weight['corner'] * distill_loss_logits

        # compute cls loss if neccessary
        if 'pred_scores' in pred_dict:
            score_loss = self.objective['score'](pred_dict['pred_scores'].view(-1), labels)
            loss = score_loss * self.loss_weight['score']

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            mean_iou_teacher = iou_teacher.detach().mean()
            if 'pred_scores' in pred_dict:
                status = {"Loss/total": loss.item(),
                          "Loss/scores": score_loss.item()}
            else:
                status = {"Loss/total": loss.item(),
                          "Loss/ciou": ciou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "Loss/distill_logits": distill_loss_logits.item(),
                          "IoU": mean_iou.item(),
                          "IoU_teacher": mean_iou_teacher.item()}
            return loss, status
        else:
            return loss

    def compute_losses_distill(self, pred_dict, pred_dict_teacher):
        """
        prob_tl/br: corner logits before softmax for student, prob after softmax for teacher, shape (b, hw)
        distill_feat_list: features, shape (b, hw, c)
        """
        prob_l = pred_dict['prob_l']
        prob_r = pred_dict['prob_r']
        prob_t = pred_dict['prob_t']
        prob_b = pred_dict['prob_b']    # (b, feat_sz)

        feat_sz = self.feat_sz
        assert feat_sz ** 2 == pred_dict_teacher['prob_tl'].size(1)
        ptl_tea, pbr_tea = pred_dict_teacher['prob_tl'].detach(), pred_dict_teacher['prob_br'].detach()
        ptl_tea = ptl_tea.view((-1, feat_sz, feat_sz))
        pbr_tea = pbr_tea.view((-1, feat_sz, feat_sz))
        prob_t_tea = ptl_tea.sum(dim=2)
        prob_l_tea = ptl_tea.sum(dim=1)
        prob_b_tea = pbr_tea.sum(dim=2)
        prob_r_tea = pbr_tea.sum(dim=1) # (b, feat_sz)

        dis_loss_logits = (self.distill_logits_loss(F.log_softmax(prob_t, dim=1), prob_t_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_l, dim=1), prob_l_tea) + 
                           self.distill_logits_loss(F.log_softmax(prob_b, dim=1), prob_b_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_r, dim=1), prob_r_tea)) / 4

        return dis_loss_logits
