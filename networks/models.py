import torch
import torch.nn as nn
import networks.resnet_GN_WS
import networks.layers_WS as L
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
import networks.layers_WS


class MattingModuleBase(nn.Module):
    def __init__(self):
        super(MattingModuleBase, self).__init__()


class MattingModuleSingleGpu(MattingModuleBase):
    def __init__(self, net_enc, net_dec):
        super(MattingModuleSingleGpu, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec

    def forward(self, image, trimap_transformed, prev_alpha, trimap):

        resnet_input = torch.cat((image, trimap_transformed), 1)
        if(self.encoder.use_mask_input):
            resnet_input = torch.cat((image, trimap_transformed, prev_alpha), 1)

        conv_out, indices = self.encoder(resnet_input, return_feature_maps=True)

        return self.decoder(conv_out, indices, trimap_transformed, prev_alpha, trimap)


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname.find('BasicConv') == -1 and classname.find('CoordConv') == -1 and classname.find('ConvGRUCell') == -1 and classname.find('ConvLSTMCell') == -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def build_encoder(self, use_mask_input, weights='', pretrained=True):
        orig_resnet = networks.resnet_GN_WS.__dict__['l_resnet50'](pretrained=False)
        net_encoder = ResnetDilated(orig_resnet, use_mask_input, dilate_scale=8)

        num_channels = 3 + 6
        if(net_encoder.use_mask_input):
            num_channels += 1

        # num_channels+=9
        if(num_channels > 3):
            print('modifying input layer')
            net_encoder_sd = net_encoder.state_dict()
            conv1_weights = net_encoder_sd['conv1.weight']
            c_out, c_in, h, w = conv1_weights.size()
            conv1_mod = torch.zeros(c_out, num_channels, h, w)
            conv1_mod[:, :3, :, :] = conv1_weights
            conv1 = net_encoder.conv1
            conv1.in_channels = num_channels
            conv1.weight = torch.nn.Parameter(conv1_mod)
            net_encoder.conv1 = conv1
            net_encoder_sd['conv1.weight'] = conv1_mod
            net_encoder.load_state_dict(net_encoder_sd)

        return net_encoder

    def build_decoder(self, use_mask_input=True, use_usr_encoder=True, gf=True):
        net_decoder = InteractiveSegNet(use_mask_input=use_mask_input, use_usr_encoder=use_usr_encoder, gf=gf)
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, use_mask_input=False, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
        self.use_mask_input = use_mask_input

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = [x]
        x = self.relu(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out, indices
        return [x]


class usr_encoder2(nn.Module):
    def __init__(self, input_dim=2):
        super(usr_encoder2, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x2))
        x4 = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x4))
        x = F.leaky_relu(self.conv6(x))

        return [x, x4, x2]


class InteractiveSegNet(nn.Module):
    def __init__(self, pool_scales=(1, 2, 3, 6), use_mask_input=True, use_usr_encoder=True, gf=True):
        super(InteractiveSegNet, self).__init__()

        self.use_usr_encoder = use_usr_encoder
        self.gf = gf
        if(self.use_usr_encoder):
            self.use_mask_input = use_mask_input

            usr_inp_dim = 6
            if self.use_mask_input:
                usr_inp_dim += 1

            self.usr_encoder = usr_encoder2(input_dim=usr_inp_dim)

            usr_encoder_dims = [256, 256, 64]

        else:
            usr_encoder_dims = [0, 0, 0]

        conv_2_C = 256

        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                L.Conv2d(2048 + usr_encoder_dims[0], 256, kernel_size=1, bias=True),
                nn.GroupNorm(32, 256),
                nn.LeakyReLU()
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_up1 = nn.Sequential(
            L.Conv2d(2048 + len(pool_scales) * 256 + usr_encoder_dims[0], 256,
                     kernel_size=3, padding=1, bias=True),

            nn.GroupNorm(32, 256),
            nn.LeakyReLU(),
            L.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU()
        )

        self.conv_up2 = nn.Sequential(
            L.Conv2d(conv_2_C + 256 + usr_encoder_dims[1], 256,
                     kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU()
        )

        self.conv_up3 = nn.Sequential(
            L.Conv2d(256 + 64 + usr_encoder_dims[2], 64,
                     kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU()
        )

        self.conv_up4 = nn.Sequential(
            nn.Conv2d(64 + 3 + 6, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),

            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True)
        )

        if(self.gf):
            self.guided_map_conv1 = nn.Conv2d(5, 64, 1)
            self.guided_map_relu1 = nn.ReLU(inplace=True)
            self.guided_map_conv2 = nn.Conv2d(64, 1, 1)
            self.guided_filter = GuidedFilter(2, 1e-8)

    def forward(self, conv_out, indices, trimap_transformed, prev_alpha, trimap):
        conv5 = conv_out[-1]
        img = conv_out[-6][:, :3]
        if(self.gf):
            g0 = self.guided_map_relu1(self.guided_map_conv1(torch.cat((img, trimap_transformed[:, ::3]), 1)))
            g = self.guided_map_conv2(g0)

        if(self.use_usr_encoder):
            if self.use_mask_input:
                usr_x, usr_x4, usr_x2 = self.usr_encoder(torch.cat((trimap_transformed, prev_alpha), 1))
            else:
                usr_x, usr_x4, usr_x2 = self.usr_encoder(trimap_transformed)

            conv5 = torch.cat((conv5, usr_x), 1)

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_up1(ppm_out)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-4]), 1)
        if self.use_usr_encoder:
            x = torch.cat((x, usr_x4), 1)

        x = self.conv_up2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-5]), 1)
        if self.use_usr_encoder:
            x = torch.cat((x, usr_x2), 1)
        x = self.conv_up3(x)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, img, trimap_transformed), 1)

        pred = self.conv_up4(x)

        pred = torch.clamp(pred, 0, 1)

        pred[trimap[:, 0][:, None, :, :] == 1] = 0
        pred[trimap[:, 1][:, None, :, :] == 1] = 1

        if(self.gf):
            pred = self.guided_filter(g, pred)
            pred = pred.clamp(0, 1)

        return pred


def build_model(args):
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(use_mask_input=(not args.use_usr_encoder and args.use_mask_input))

    net_decoder = builder.build_decoder(use_mask_input=args.use_mask_input, use_usr_encoder=args.use_usr_encoder, gf=True)

    model = MattingModuleSingleGpu(net_encoder, net_decoder)

    model.cuda()

    sd = torch.load(args.weights)
    model.load_state_dict(sd, strict=True)
    return model
