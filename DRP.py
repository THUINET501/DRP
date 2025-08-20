from __future__ import print_function
import argparse
import os
import logging
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from models import skip
from torch_radon import RadonFanbeam
import matplotlib.pyplot as plt
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description='Deep Radon Prior')
    parser.add_argument('--input', type=str, default='./data_input/phantom.mat', help='Input .mat file path')
    parser.add_argument('--output', type=str, default='./data_results/phantom-60/', help='Output directory')
    parser.add_argument('--n_angles', type=int, default=60, help='Number of projection angles')
    parser.add_argument('--detector_count', type=int, default=729, help='Detector count')
    parser.add_argument('--epochs', type=int, default=150, help='Number of outer epochs')
    parser.add_argument('--iters', type=int, default=200, help='Number of inner iterations')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_p', type=float, default=0.8, help='Weight p for input update')
    return parser.parse_args()


class Config:
    def __init__(self, args):
        self.input_path = args.input
        self.output_path = args.output
        self.n_angles = args.n_angles
        self.detector_count = args.detector_count
        self.num_epochs = args.epochs
        self.num_iterations = args.iters
        self.learning_rate = args.lr
        self.weight_p = args.weight_p
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model_path = os.path.join(self.output_path, 'best_model.pt')
        self.latest_model_path = os.path.join(self.output_path, 'latest_model.pt')
        self.fbp_path = os.path.join(self.output_path, 'fbp.mat')


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('DRP')


def setup_network(device):
    net = skip(1, 1,
               num_channels_down=[16, 32, 64, 128, 256],
               num_channels_up=[16, 32, 64, 128, 256],
               num_channels_skip=[4, 4, 4, 4, 4],
               upsample_mode='bilinear',
               need_sigmoid=True,
               need_bias=True,
               pad='zero',
               act_fun='LeakyReLU')
    return net.to(device)


def get_radon_transform(image_size, config):
    angles = np.linspace(0, 2 * np.pi, config.n_angles, endpoint=False)
    return RadonFanbeam(image_size, angles, 600, 600, det_count=config.detector_count, det_spacing=1.5, clip_to_circle=False)


def plot_images(sinogram, fbp, output_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sinogram.cpu().numpy(), cmap='gray')
    plt.title('Sinogram')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(fbp.cpu().numpy(), cmap='gray')
    plt.title('FBP')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sinogram_fbp.png'))
    plt.show()


def train(config, logger):
    os.makedirs(config.output_path, exist_ok=True)
    logger.info('Output directory: %s', config.output_path)

    if not os.path.exists(config.input_path):
        logger.error('Input file not found: %s', config.input_path)
        return

    data = loadmat(config.input_path)
    if 'data' not in data:
        logger.error('Input .mat does not contain key "data"')
        return

    img = data['data'].astype(np.float32)
    image_size = img.shape[0]

    radon = get_radon_transform(image_size, config)

    with torch.no_grad():
        x = torch.FloatTensor(img).to(config.device)
        sinogram = radon.forward(x)
        filtered_sinogram = radon.filter_sinogram(sinogram)
        fbp = radon.backprojection(filtered_sinogram)
        fbp[fbp < 0] = 0
        savemat(config.fbp_path, {'fbp': fbp.cpu().numpy()})

        sino_w = radon.forward(x * 0 + 1)
        weight_i = radon.backprojection(sino_w)

        net_input = fbp.unsqueeze(0).unsqueeze(1).to(config.device)

    plot_images(sinogram, fbp, config.output_path)

    net = setup_network(config.device)
    mse = torch.nn.MSELoss(reduction='mean').to(config.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, amsgrad=True)

    params = sum([np.prod(list(p.size())) for p in net.parameters()])
    logger.info('Number of params: %d', params)

    for epoch in range(config.num_epochs):
        out = net(net_input).data
        out_sino = radon.forward(out.squeeze(0).squeeze(0)) if out.dim() == 4 else radon.forward(out)

        err = torch.div(radon.backward(sinogram - out_sino), weight_i)

        net_input = config.weight_p * err / torch.norm(err) + (1 - config.weight_p) * out / torch.norm(out)

        for iteration in range(config.num_iterations):
            optimizer.zero_grad()
            out = net(net_input)
            out_sino = radon.forward(out)
            total_loss = mse(out_sino.squeeze(), sinogram.squeeze())
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            output = torch.squeeze(out).cpu().numpy()
            psnr = compare_psnr(img, output)
            ssim = compare_ssim(img, output, data_range=output.max() - output.min())
            logger.info('Epoch %03d Loss %.6f PSNR: %.4f SSIM: %.4f', epoch, total_loss.item(), psnr, ssim)

            pd.DataFrame({'epoch': [f'{epoch:03d}'], 'PSNR': [f'{psnr:.4f}'], 'SSIM': [f'{ssim:.4f}']}).to_csv(
                os.path.join(config.output_path, f'PSNR_SSIM-{config.n_angles}.csv'), mode='a', index=False, header=False)

            savemat(os.path.join(config.output_path, f'{epoch:03d}.mat'), {'out': output})

            if psnr > config.best_psnr:
                config.best_psnr = psnr
                savemat(os.path.join(config.output_path, 'best_result.mat'), {'out': output})
                torch.save({'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, config.best_model_path)
            if ssim > config.best_ssim:
                config.best_ssim = ssim

            logger.info('Best PSNR: %.4f, Best SSIM: %.4f', config.best_psnr, config.best_ssim)

            torch.save({'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, config.latest_model_path)


def main():
    args = get_args()
    config = Config(args)
    logger = setup_logging()
    train(config, logger)


if __name__ == '__main__':
    main()
