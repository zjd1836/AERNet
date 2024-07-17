import numpy as np


class SegEvaluator:
    def __init__(self, class_num=4):
        if class_num == 1:
            class_num = 2
        self.num_class = class_num
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def kappa(self,OA):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = np.sum(self.confusion_matrix)
        pe = np.dot(pe_rows, pe_cols) / (sum_total ** 2)
        #po = self.pixel_oa()
        po = OA
        return (po - pe) / (1 - pe)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix


    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.mat=self.confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def loss_weight(self):
        TN = self.confusion_matrix[0][0]
        FP = self.confusion_matrix[0][1]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[1][1]
        w_00 = TP / (TP + FP + FN) 
        w_11 = TN / (TN + FN + FP)
        return w_00, w_11

    def matrix(self,class_index):
        metric = {}
        recall = 0.0
        precision = 0.0
        for i in range(self.num_class):
            recall += self.confusion_matrix[i, i] / (np.sum(self.confusion_matrix[:, i]) + 1e-8)
            precision += self.confusion_matrix[i, i] / (np.sum(self.confusion_matrix[i, :]) + 1e-8)
        precision_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        recall_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        OA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        iou_per_class = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        metric['0_IoU'] = iou_per_class[0]
        metric['1_IoU'] = iou_per_class[1]
        metric['IoU'] = np.nanmean(iou_per_class)
        metric['Precision'] = precision_cls[class_index]  #precision / self.num_class
        metric['Recall'] = recall_cls[class_index]          #recall / self.num_class
        metric['OA'] = OA
        metric['F1'] = (2 * precision_cls[class_index] * recall_cls[class_index]) / (precision_cls[class_index] + recall_cls[class_index])  
        Kappa = self.kappa(OA)
        metric['Kappa'] = Kappa
        return metric
