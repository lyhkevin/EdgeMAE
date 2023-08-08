import torch
import logging
from utils.maeloader import *
from model.EdgeMAE import *
from utils.mae_visualize import *
import torch.distributed as dist
from utils.visualize import *
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
        self.parser.add_argument('--min_lr', type=float, default=3e-5, help='learning rate')
        self.parser.add_argument('--warmup_epochs', default=1)
        self.parser.add_argument("--epoch", default=100, type=int)
        self.parser.add_argument("--batch_size_per_gpu", default=40, type=int)
        self.parser.add_argument("--patch_size", default=8, type=int)
        self.parser.add_argument("--img_size", default=256, type=int)
        self.parser.add_argument("--l1_loss", default=10, type=int)
        self.parser.add_argument("--augment", default=True) #preform data augmentation
        self.parser.add_argument("--modality", default='all') #using all modalities for pre-training (t1, t2, t1c, flair)
        self.parser.add_argument("--masking_ratio", default=0.7,type=float)
        self.parser.add_argument("--num_workers", default=8, type=int)
        
        self.parser.add_argument('--use_checkpoints', default=False)
        self.parser.add_argument('--img_save_path', type=str,default='./snapshot/ddp/')
        self.parser.add_argument('--weight_save_path', type=str,default='./weight/ddp/')
        self.parser.add_argument("--data_root", default='./data/train/')

        self.parser.add_argument("--depth", default=12, type=int)
        self.parser.add_argument("--use_patchwise_loss", default=True)
        self.parser.add_argument("--decoder_depth", default=8, type=int)
        self.parser.add_argument("--save_output", default=200, type=int)
        self.parser.add_argument("--save_weight", default=1, type=int)
        self.parser.add_argument("--num_heads", default=16, type=int)
        self.parser.add_argument("--decoder_num_heads", default=8, type=int)
        self.parser.add_argument("--mlp_ratio", default=4, type=int)
        self.parser.add_argument("--dim_encoder", default=128, type=int)
        self.parser.add_argument("--dim_decoder", default=64, type=int)
        self.parser.add_argument("--log_path", default='./log/ddp.log')

        #for distributed training
        self.parser.add_argument("--rank", default=0, type=int)
        self.parser.add_argument("--world_size", default=3, type=int)
        self.opt = self.parser.parse_args(args=[])

    def get_opt(self):
        return self.opt
    
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def train(rank, world_size):
    print(f"Start running EdgeMAE on rank {rank}.")
    setup(rank, world_size)
    mae = EdgeMAE(img_size=opt.img_size,patch_size=opt.patch_size, embed_dim=opt.dim_encoder, depth=opt.depth, num_heads=opt.num_heads, in_chans=1,
        decoder_embed_dim=opt.dim_decoder, decoder_depth=opt.decoder_depth, decoder_num_heads=opt.decoder_num_heads,
        mlp_ratio=opt.mlp_ratio,norm_pix_loss=False,patchwise_loss=opt.use_patchwise_loss)
    mae = nn.SyncBatchNorm.convert_sync_batchnorm(mae)
    mae = mae.to(rank)
    mae = DDP(mae, device_ids=[rank],find_unused_parameters=True)
    
    os.makedirs(opt.img_save_path,exist_ok=True)
    os.makedirs(opt.weight_save_path,exist_ok=True)

    dataset, _ = get_maeloader(batchsize=opt.batch_size_per_gpu, shuffle=True,pin_memory=True,img_size=opt.img_size,
                img_root=opt.data_root,num_workers=opt.num_workers,augment=opt.augment,modality=opt.modality)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset,sampler=sampler,batch_size=opt.batch_size_per_gpu,num_workers=opt.num_workers,pin_memory=True,drop_last=True,)
    optimizer = torch.optim.Adam(mae.parameters(), lr=opt.lr,betas=(0.9, 0.95))
    lr_scheduler = cosine_scheduler(opt.lr, opt.min_lr, opt.epoch, len(train_loader), warmup_epochs=opt.warmup_epochs)
        
    logging.basicConfig(filename=opt.log_path,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    
    for epoch in range(0,opt.epoch):
        for i,img in enumerate(train_loader):
            
            it = len(train_loader) * epoch + i
            for id, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler[it]
                
            optimizer.zero_grad()
            img = img.to(rank)
            
            rec_loss, edge_loss,edge_gt,x_edge,x_rec,mask = mae(img,opt.masking_ratio,epoch)
            loss = rec_loss * opt.l1_loss + edge_loss 
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]" % 
                      (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer)))
                
            if i % opt.save_output == 0 and rank == 0:
                y1, im_masked1, im_paste1 = MAE_visualize(img, x_rec, mask)
                y2, im_masked2, im_paste2 = MAE_visualize(edge_gt, x_edge, mask)
                edge_gt,img = edge_gt.cpu(),img.cpu()
                save_image([img[0],im_masked1,im_paste1,edge_gt[0],im_masked2,im_paste2],
                    opt.img_save_path + str(epoch) + ' ' + str(i)+'.png', nrow=3,normalize=False)
                logging.info("[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                    % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer)))
                
        if rank == 0:
            if epoch % opt.save_weight == 0:
                torch.save(mae.state_dict(), opt.weight_save_path + str(epoch) + 'MAE.pth')
    if rank == 0:
        torch.save(mae.state_dict(), opt.weight_save_path + './MAE.pth')
        
opt = Options().get_opt()

if __name__ == '__main__':
    mp.spawn(train, args=(opt.world_size,), nprocs=opt.world_size, join=True)