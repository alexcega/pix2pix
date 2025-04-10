import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
    ):
    loop = tqdm(loader, leave=True)
    
    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        #* Train Discriminator
        with torch.amp.autocast("cuda"):
            y_fake = gen(x)
            disc_real = disc(x, y)
            disc_fake = disc(x, y_fake.detach())
            disc_real_loss = bce(disc_real, torch.ones_like(disc_real))
            disc_fake_loss = bce(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
        
        disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        #* Train Generator
        with torch.amp.autocast("cuda"):
            disc_fake = disc(x, y_fake)
            gen_fake_loss = bce(disc_fake, torch.ones_like(disc_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            gen_loss = gen_fake_loss + L1

        opt_gen.zero_grad()
        d_scaler.scale(gen_loss).backward()
        d_scaler.step(opt_gen)
        d_scaler.update()\
        
        if idx % 10 == 0:
            loop.set_postfix(
                disc_real=torch.sigmoid(disc_real).mean().item(),
                disc_fake=torch.sigmoid(disc_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))  #TODO Check betas
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss() # Binary Cross Entropy with logits #TODO check logits
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir="data/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")
    val_dataset = MapDataset(root_dir="data/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch %5 ==0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")

if __name__ == "__main__":
    main()