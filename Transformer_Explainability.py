"""
@InProceedings{Chefer_2021_CVPR,
    author    = {Chefer, Hila and Gur, Shir and Wolf, Lior},
    title     = {Transformer Interpretability Beyond Attention Visualization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {782-791}
}
"""
import numpy as np
import cv2
import torch

class Transformer_Explainability:
    def __init__(self, model, cls_to_idx):
        self.model = model
        self.cls_to_idx = cls_to_idx

    def avg_heads(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    def apply_self_attention_rules(self, R_ss, cam_ss):
        R_ss_addition = torch.matmul(cam_ss, R_ss)
        return R_ss_addition

    def generate_relevance(self, input, index=None):
        output = self.model(input, register_hook=True)

        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        num_tokens = self.model.models.layers[0].attention.get_attention_map().shape[-1]
        R = torch.eye(num_tokens, num_tokens).cuda()

        for layer in self.model.models.layers:
            grad = layer.attention.get_attn_gradients()
            cam = layer.attention.get_attention_map()
            cam = self.avg_heads(cam, grad)
            R += self.apply_self_attention_rules(R.cuda(), cam.cuda())

        return R[0, 1:]

    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    def generate_visualization(self, image, class_index=None):
        self.image = image
        _, h, w = self.image.shape

        transformer_attribution = self.generate_relevance(self.image.unsqueeze(0).cuda(), index=class_index).detach()
        sqrt_shape = int(transformer_attribution.shape[0] ** 0.5)

        # print("Shape before reshape:", transformer_attribution.shape)
        transformer_attribution = transformer_attribution.reshape(1, 1, sqrt_shape, sqrt_shape)

        scale_factor_h = h / sqrt_shape
        scale_factor_w = w / sqrt_shape

        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=(scale_factor_h, scale_factor_w), mode="bilinear")

        transformer_attribution = transformer_attribution.reshape(h, w).cuda().data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

        image_transformer_attribution = self.image.permute(1, 2, 0).data.cpu().numpy()
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())

        vis = self.show_cam_on_image(image_transformer_attribution, transformer_attribution)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

        return vis

    def print_top_classes(self, predictions):
        # 予測上位3を表示
        prob = torch.softmax(predictions, dim=1)
        class_indices = predictions.data.topk(3, dim=1)[1][0].tolist()
        max_str_len = 0
        class_names = []
        for cls_idx in class_indices:
            class_names.append(self.cls_to_idx[cls_idx])
            if len(self.cls_to_idx[cls_idx]) > max_str_len:
                max_str_len = len(self.cls_to_idx[cls_idx])

        output_strings = []  # このリストにすべての文字列を保存
        for cls_idx in class_indices:
            line = '\t{} : {}'.format(cls_idx, self.cls_to_idx[cls_idx])
            line += ' ' * (max_str_len - len(self.cls_to_idx[cls_idx])) + '\t\t'
            # line += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx]) # value + probを表示
            line += 'prob = {:.1f}%'.format(100 * prob[0, cls_idx]) # probのみ 
            output_strings.append(line)

        # すべての文字列を連結して1つの文字列として返す
        return '\n'.join(output_strings)
    