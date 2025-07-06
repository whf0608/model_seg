'''
Hongruixuan Chen, Chen Wu, Bo Du, Liangpei Zhang, and Le Wang: 
Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network, 
IEEE Trans. Geosci. Remote Sens., 58, 2848–2864, 2020. 
https://github.com/ChenHongruixuan/SiamCRNN/tree/master/FCN_version
'''

import torch
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SiamCRNN(nn.Module):
    def __init__(self,in_channels=3, n_classes=4):
        super(SiamCRNN, self).__init__()
        expansion = 1

        self.encoder_1 = torchvision.models.resnet34(pretrained=True)
        self.encoder_2 = torchvision.models.resnet34(pretrained=True)
        return_nodes = {
            'layer1': 'feat1',
            'layer2': 'feat2',
            'layer3': 'feat3',
            'layer4': 'feat4'
        }
        self.extractor_1 = create_feature_extractor(self.encoder_1, return_nodes=return_nodes)
        self.extractor_2 = create_feature_extractor(self.encoder_2, return_nodes=return_nodes)

        self.convlstm_4 = ConvLSTM(input_dim=512 * expansion , hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_3 = ConvLSTM(input_dim=256 * expansion , hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_2 = ConvLSTM(input_dim=128 * expansion , hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_1 = ConvLSTM(input_dim=64 * expansion , hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        
        self.trans_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=512 * expansion, out_channels=128, padding=1),
                                             nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=256 * expansion, out_channels=128, padding=1),
                                             nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=128 * expansion, out_channels=128, padding=1),
                                             nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=64 * expansion, out_channels=128, padding=1),
                                             nn.BatchNorm2d(128), nn.ReLU())

        self.smooth_layer_13 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_12 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_11 = ResBlock(in_channels=128, out_channels=128, stride=1) 

        self.smooth_layer_23 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_22 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_21 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        

        self.main_clf_loc = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        self.main_clf_clf = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data=None):
        if post_data is None:  post_data = pre_data
        pre_features = self.extractor_1(pre_data)
        post_features = self.extractor_2(post_data)
        pre_low_level_feat_1, pre_low_level_feat_2, pre_low_level_feat_3, pre_output = pre_features["feat1"], pre_features["feat2"], pre_features["feat3"], pre_features["feat4"]
        post_low_level_feat_1, post_low_level_feat_2, post_low_level_feat_3, post_output = post_features["feat1"], post_features["feat2"], post_features["feat3"], post_features["feat4"]

        # Concatenate along the time dimension
        p4_loc = self.trans_layer_4(pre_output)
        combined_4 = torch.stack([pre_output, post_output], dim=1)
        # Apply ConvLSTM
        _, last_state_list_4 = self.convlstm_4(combined_4)
        p4 = last_state_list_4[0][0]

        p3_loc = self.trans_layer_3(pre_low_level_feat_3)
        p3_loc = self._upsample_add(p4_loc, p3_loc)
        p3_loc = self.smooth_layer_13(p3_loc)
        combined_3 = torch.stack([pre_low_level_feat_3, post_low_level_feat_3], dim=1)
        # Apply ConvLSTM
        _, last_state_list_3 = self.convlstm_3(combined_3)
        p3 = last_state_list_3[0][0]
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_23(p3)
        
        
        p2_loc = self.trans_layer_2(pre_low_level_feat_2)
        p2_loc = self._upsample_add(p3_loc, p2_loc)
        p2_loc = self.smooth_layer_12(p2_loc)
        combined_2 = torch.stack([pre_low_level_feat_2, post_low_level_feat_2], dim=1)
        # Apply ConvLSTM
        _, last_state_list_2 = self.convlstm_2(combined_2)
        p2 = last_state_list_2[0][0]
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_22(p2)

        
        p1_loc = self.trans_layer_1(pre_low_level_feat_1)
        p1_loc = self._upsample_add(p2_loc, p1_loc)
        p1_loc = self.smooth_layer_11(p1_loc)

        combined_1 = torch.stack([pre_low_level_feat_1, post_low_level_feat_1], dim=1)
        # Apply ConvLSTM
        _, last_state_list_1 = self.convlstm_1(combined_1)
        p1 = last_state_list_1[0][0]
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_21(p1)

        output_loc = self.main_clf_loc(p1_loc)
        output_loc = F.interpolate(output_loc, size=pre_data.size()[-2:], mode='bilinear')
        output_clf = self.main_clf_clf(p1)
        output_clf = F.interpolate(output_clf, size=pre_data.size()[-2:], mode='bilinear')
        return output_loc, output_clf


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
