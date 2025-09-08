import nibabel as nib
import numpy as np
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.networks import Generator, Discriminator
from src.data import BCP
# from src.data import create_weighted_sampler_bcp
from src.discriminator_augmentations import disc_augment
from src.losses import discriminator_loss, generator_loss


"""
Config
"""

epochs = 200
batch_size = 1
d_train_steps = 1
g_train_steps = 1
oversample = True
clip_bckgnd = True
reg_loss = 'NCC'
g_ch = 32
d_ch = 64
initialization = 'default'

# Optimizer parameters
lr_g = 1e-4
lr_d = 3e-4
beta1_g = 0.0
beta2_g = 0.9
beta1_d = 0.0
beta2_d = 0.9

# Loss weights
loss_wt_reg = 1.0
loss_wt_gan = 0.1
loss_wt_tv = 0
loss_wts = (loss_wt_gan, loss_wt_reg, loss_wt_tv)
loss_wt_gp = 1e-3
start_step = 0

# save
affine = np.eye(4, dtype=np.float32)
header = None
val_ages = [1, 3, 6, 9, 12, 18, 24]

"""
Setup
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'Ours'
save_dir = MODEL_NAME
if not os.path.exists('logs/' + save_dir):
    os.makedirs('logs/' + save_dir)
if not os.path.exists('models/' + save_dir):
    os.makedirs('models/' + save_dir)
if not os.path.exists('images/' + save_dir):
    os.makedirs('images/' + save_dir)

G = Generator().to(device)
D = Discriminator().to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1_g, beta2_g), eps=1e-7)
D_optimizer = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1_d, beta2_d), eps=1e-7)


"""
Dataset
"""

# Training dataset
dataset = BCP(data_folder='./BCP_train', age_max=2205., mode='default')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# sampler = create_weighted_sampler_bcp(data_folder='./BCP_train')
# dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Validation dataset
val_dataset = BCP(data_folder='./BCP_test', age_max=2205., mode='default')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

rough_template = torch.from_numpy(
    np.load('./BCP_rough_template.npz')['arr_0']
).unsqueeze(0).to(device).float()
# rough_t1 = rough_template[:, 0, :, :, :].reshape(batch_size, 1, 160, 192, 160)
# rough_tissue = rough_template[:, 1:, :, :, :].reshape(batch_size, 3, 160, 192, 160)
# rough_template = torch.cat((rough_t1, rough_tissue), dim=1).to(device)

# temp = nib.load('./rough_temp.nii.gz')
# affine = temp.affine
# header = temp.header

# val_ages = [1, 3, 6, 9, 12, 18, 24]

# continue_training = True

"""
Training
"""

def train():
    writer = SummaryWriter(log_dir='logs/' + save_dir)
    step = 0
    current_epoch = 0

    for epoch in range(current_epoch, epochs):
        print(f"{25 * '='} Epoch:{epoch} {25 * '='}")
        G.train()
        D.train()

        for i, (image, age) in enumerate(tqdm(dataloader)):
            step += 1

            """
            Train discriminator
            """

            # We get two additional samples to train the discriminator.
            registration, registration_age = next(iter(dataloader))
            adversarial, adversarial_age = next(iter(dataloader))
            adversarial_t1 = adversarial[:, 0, :, :, :].reshape(batch_size, 1, 160, 192, 160)

            registration = registration.to(device)
            adversarial = adversarial.to(device)
            registration_age = registration_age.to(device)
            adversarial_age = adversarial_age.to(device)
            adversarial_t1 = adversarial_t1.to(device)

            # Forward pass through the generator
            moved_atlas, _, _, _ = G(registration_age, rough_template, registration)
            moved_atlas_t1 = moved_atlas[:, 0, :, :, :].reshape(batch_size, 1, 160, 192, 160)

            # Apply discriminator augmentation
            moved_atlas_t1 = disc_augment(moved_atlas_t1)
            adversarial_t1 = disc_augment(adversarial_t1)

            # Forward pass through the discriminator
            fake_pred = D(moved_atlas_t1, registration_age)
            real_pred = D(adversarial_t1, adversarial_age)

            # Calculate losses
            d_loss = discriminator_loss(real_pred, fake_pred)

            gp = adversarial.requires_grad_()
            gp_t1 = gp[:, 0, :, :, :].reshape(batch_size, 1, 160, 192, 160)
            gp_age = adversarial_age.requires_grad_()

            # R1 gradient penalty
            gp_pred = D(gp_t1, gp_age)
            gp_grad = autograd.grad(outputs=gp_pred,
                                    inputs=gp_t1,
                                    grad_outputs=torch.ones(1, 1, 10, 12, 10).to(device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True,
                                    allow_unused=True)
            gp_grad_sqr = [torch.square(grad) for grad in gp_grad]
            gp_grad_sqr = torch.sum(torch.cat(gp_grad_sqr))
            gp_loss = loss_wt_gp * gp_grad_sqr / 2.0

            # Calculate total loss
            total_d_loss = d_loss + gp_loss

            # Backprop
            D_optimizer.zero_grad()
            total_d_loss.backward()
            D_optimizer.step()

            # del adversarial_t1, registration, adversarial
            # del moved_atlas, _, moved_atlas_t1
            # del fake_pred, real_pred
            # del gp, gp_t1, gp_age
            # del gp_pred, gp_grad, gp_grad_sqr

            # Log
            writer.add_scalar('Discriminator/d_loss', d_loss.item(), step)
            writer.add_scalar('Discriminator/gp_loss', gp_loss.item(), step)
            writer.add_scalar('Discriminator/total_d_loss', total_d_loss.item(), step)

            """
            Train generator
            """

            image = image.to(device)
            age = age.to(device)

            moved, disp_field_ms, _, disp_field_half = G(age, rough_template, image)

            moved_t1 = moved[:, 0, :, :, :].reshape(batch_size, 1, 160, 192, 160)
            pred = D(moved_t1, age)

            # Calculate losses
            # total_g_loss, g_loss, smoothness_loss, magnitude_loss, similarity_loss, moving_magnitude_loss, dice_loss, ce_loss = \
            # generator_loss(
            #     pred,
            #     disp_field_ms1,
            #     disp_field_half1,
            #     moved_atlas1,
            #     image1,
            #     loss_wts,
            #     reg_loss,
            # )

            total_g_loss, g_loss, smoothness_loss, magnitude_loss, similarity_loss, moving_magnitude_loss, dice_loss = \
                generator_loss(
                    pred,
                    disp_field_ms,
                    disp_field_half,
                    moved,
                    image,
                    loss_wts,
                    reg_loss,
                )

            # Backprop
            G_optimizer.zero_grad()
            total_g_loss.backward()
            G_optimizer.step()

            # del image
            # del moved, disp_field_ms, disp_field_half
            # del moved_t1, pred
            # del _
            # del seg1, seg2, _
            # del _

            # Log
            writer.add_scalar('Generator/g_loss', g_loss.item(), step)
            writer.add_scalar('Generator/smoothness_loss', smoothness_loss.item(), step)
            writer.add_scalar('Generator/magnitude_loss', magnitude_loss.item(), step)
            writer.add_scalar('Generator/similarity_loss', similarity_loss.item(), step)
            writer.add_scalar('Generator/moving_magnitude_loss', moving_magnitude_loss.item(), step)
            writer.add_scalar('Generator/dice_loss', dice_loss.item(), step)
            writer.add_scalar('Generator/total_g_loss', total_g_loss.item(), step)

        if epoch % 5 == 0:
            path = f'./models/{save_dir}/model_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'G_optimizer_state_dict': G_optimizer.state_dict(),
                'D_state_dict': D.state_dict(),
                'D_optimizer_state_dict': D_optimizer.state_dict(),
                'g_loss': total_g_loss,
                'd_loss': total_d_loss,
                'loss_wts': loss_wts,
                'step': step,
            }, path)

        """
        Visualization and saving images
        """

        with torch.no_grad():
            relu = nn.ReLU()

            for i, (image, age) in enumerate(val_dataloader):
                image = image.to(device)
                age = age.to(device)
                moved_atlas, _, atlas, disp_field_half = G(age, rough_template, image)
                moved_atlas_t1 = moved_atlas[:, 0, :, :, :].reshape(batch_size, 1, 160, 192, 160)

                moved_atlas = relu(moved_atlas)
                atlas = relu(atlas)

                # Log images
                writer.add_image('moved_atlas/1', moved_atlas_t1[:, 0, 80, :, :].squeeze(), epoch, dataformats='HW')
                writer.add_image('moved_atlas/2', moved_atlas_t1[:, 0, :, 96, :].squeeze(), epoch, dataformats='HW')
                writer.add_image('moved_atlas/3', moved_atlas_t1[:, 0, :, :, 80].squeeze(), epoch, dataformats='HW')
                writer.add_image('atlas/1', atlas[:, 0, 80, :, :].squeeze(), epoch, dataformats='HW')
                writer.add_image('atlas/2', atlas[:, 0, :, 96, :].squeeze(), epoch, dataformats='HW')
                writer.add_image('atlas/3', atlas[:, 0, :, :, 80].squeeze(), epoch, dataformats='HW')

                # Save images
                moved_atlas = nib.Nifti1Image(moved_atlas.permute(2, 3, 4, 0, 1).cpu().numpy(), affine, header)
                atlas = nib.Nifti1Image(atlas.permute(2, 3, 4, 0, 1).cpu().numpy(), affine, header)
                disp_field_half = nib.Nifti1Image(disp_field_half.permute(2, 3, 4, 0, 1).cpu().numpy(), affine, header)
                nib.save(moved_atlas, f'./images/{save_dir}/moved_atlas_{val_ages[i]}_epoch_{epoch}.nii.gz')
                nib.save(atlas, f'./images/{save_dir}/atlas_{val_ages[i]}_epoch_{epoch}.nii.gz')
                nib.save(disp_field_half, f'./images/{save_dir}/disp_field_half_{val_ages[i]}_epoch_{epoch}.nii.gz')


if __name__ == '__main__':
    """
    GPU configuration
    """
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    print(device)

    """
    Train
    """
    train()
