import logging
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from util.train_utils import Mean, HingeLoss, seed_worker
from util.utils import get_preprocess
from dataset.dataset import TwoAFCDataset
from torch.utils.data import DataLoader
import torch
from peft import get_peft_model, LoraConfig, PeftModel
from dreamsim import PerceptualModel
import os
import configargparse


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')

    ## Run options
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--tag', type=str, default='', help='tag for experiments (ex. experiment name)')
    parser.add_argument('--log_dir', type=str, default="./logs", help='path to save model checkpoints and logs')
    parser.add_argument('--load_dir', type=str, default="./models", help='path to pretrained ViT checkpoints')
    parser.add_argument('--save_mode', type=str, default="all", help='whether to save only LoRA adapter weights, '
                                                                     'entire model, or both. Accepted '
                                                                     'options: [adapter_only, entire_model, all]')

    ## Model options
    parser.add_argument('--model_type', type=str, default='dino_vitb16',
                        help='Which ViT model to finetune. To finetune an ensemble of models, pass a comma-separated'
                             'list of models. Accepted models: [dino_vits8, dino_vits16, dino_vitb8, dino_vitb16, '
                             'clip_vitb16, clip_vitb32, clip_vitl14, mae_vitb16, mae_vitl16, mae_vith14, '
                             'open_clip_vitb16, open_clip_vitb32, open_clip_vitl14]')
    parser.add_argument('--feat_type', type=str, default='cls',
                        help='What type of feature to extract from the model. If finetuning an ensemble, pass a '
                             'comma-separated list of features (same length as model_type). Accepted feature types: '
                             '[cls, embedding, last_layer, cls_patch].')
    parser.add_argument('--stride', type=str, default='16',
                        help='Stride of first convolution layer the model (should match patch size). If finetuning'
                             'an ensemble, pass a comma-separated list (same length as model_type).')
    parser.add_argument('--use_lora', type=bool, default=False,
                        help='Whether to train with LoRA finetuning [True] or with an MLP head [False].')
    parser.add_argument('--hidden_size', type=int, default=1, help='Size of the MLP hidden layer.')

    ## Dataset options
    parser.add_argument('--dataset_root', type=str, default="./dataset/nights", help='path to training dataset.')
    parser.add_argument('--num_workers', type=int, default=4)

    ## Training options
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='Dataset batch size.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--margin', default=0.01, type=float, help='Margin for hinge loss')

    ## LoRA-specific options
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=float, default=0.1, help='Alpha for attention scaling')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout probability for LoRA layers')

    return parser.parse_args()


class LightningPerceptualModel(pl.LightningModule):
    def __init__(self, feat_type: str = "cls", model_type: str = "dino_vitb16", stride: str = "16",
                 hidden_size: int = 1,
                 lr: float = 0.0003, use_lora: bool = False, margin: float = 0.05, lora_r: int = 16,
                 lora_alpha: float = 0.5, lora_dropout: float = 0.3, weight_decay: float = 0.0, train_data_len: int = 1,
                 load_dir: str = "./models", device: str = "cuda", save_mode: str = "all",
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.feat_type = feat_type
        self.model_type = model_type
        self.stride = stride
        self.hidden_size = hidden_size
        self.lr = lr
        self.use_lora = use_lora
        self.margin = margin
        self.weight_decay = weight_decay
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.train_data_len = train_data_len
        self.save_mode = save_mode

        self.__validate_save_mode()

        self.started = False
        self.val_metrics = {'loss': Mean().to(device), 'score': Mean().to(device)}
        self.__reset_val_metrics()

        self.perceptual_model = PerceptualModel(feat_type=self.feat_type, model_type=self.model_type,
                                                stride=self.stride,
                                                hidden_size=self.hidden_size, lora=self.use_lora, load_dir=load_dir,
                                                device=device)
        if self.use_lora:
            self.__prep_lora_model()
        else:
            self.__prep_linear_model()

        pytorch_total_params = sum(p.numel() for p in self.perceptual_model.parameters())
        pytorch_total_trainable_params = sum(p.numel() for p in self.perceptual_model.parameters() if p.requires_grad)
        print(f'Total params: {pytorch_total_params} | Trainable params: {pytorch_total_trainable_params} '
              f'| % Trainable: {pytorch_total_trainable_params / pytorch_total_params * 100}')

        self.criterion = HingeLoss(margin=self.margin, device=device)

        self.epoch_loss_train = 0.0
        self.train_num_correct = 0.0

    def forward(self, img_ref, img_0, img_1):
        dist_0 = self.perceptual_model(img_ref, img_0)
        dist_1 = self.perceptual_model(img_ref, img_1)
        return dist_0, dist_1

    def training_step(self, batch, batch_idx):
        img_ref, img_0, img_1, target, idx = batch
        dist_0, dist_1 = self.forward(img_ref, img_0, img_1)
        decisions = torch.lt(dist_1, dist_0)
        logit = dist_0 - dist_1
        loss = self.criterion(logit.squeeze(), target)
        loss /= target.shape[0]
        self.epoch_loss_train += loss
        self.train_num_correct += ((target >= 0.5) == decisions).sum()
        return loss

    def validation_step(self, batch, batch_idx):
        img_ref, img_0, img_1, target, id = batch
        dist_0, dist_1 = self.forward(img_ref, img_0, img_1)
        decisions = torch.lt(dist_1, dist_0)
        logit = dist_0 - dist_1
        loss = self.criterion(logit.squeeze(), target)
        val_num_correct = ((target >= 0.5) == decisions).sum()
        self.val_metrics['loss'].update(loss, target.shape[0])
        self.val_metrics['score'].update(val_num_correct, target.shape[0])
        return loss

    def on_train_epoch_start(self):
        self.epoch_loss_train = 0.0
        self.train_num_correct = 0.0
        self.started = True

    def on_train_epoch_end(self):
        self.logger.experiment.log({f'train_loss/': self.epoch_loss_train / self.trainer.num_training_batches})
        self.logger.experiment.log({f'train_2afc_acc/': self.train_num_correct / self.train_data_len})
        if self.use_lora:
            self.__save_lora_weights()

    def on_validation_start(self):
        for extractor in self.perceptual_model.extractor_list:
            extractor.model.eval()

    def on_validation_epoch_start(self):
        self.__reset_val_metrics()

    def on_validation_epoch_end(self):
        score = self.val_metrics['score'].compute()
        loss = self.val_metrics['loss'].compute()

        self.log(f'val_acc_ckpt', score, logger=False)
        self.log(f'val_loss_ckpt', loss, logger=False)

        self.logger.experiment.log({f'val_2afc_acc/': score})
        self.logger.experiment.log({f'val_loss/': loss})

        return score

    def configure_optimizers(self):
        params = list(self.perceptual_model.parameters())
        for extractor in self.perceptual_model.extractor_list:
            params += list(extractor.model.parameters())
        for extractor, feat_type in zip(self.perceptual_model.extractor_list, self.perceptual_model.feat_type_list):
            if feat_type == 'embedding':
                params += [extractor.proj]
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        return [optimizer]

    def load_lora_weights(self, checkpoint_root, epoch_load=None):
        if self.save_mode in {'adapter_only', 'all'}:
            if epoch_load is not None:
                checkpoint_root = os.path.join(checkpoint_root, f'epoch_{epoch_load}')
                
            with open(os.path.join(checkpoint_root, 'adapter_config.json'), 'r') as f:
                adapter_config = json.load(f)
            lora_keys = ['r', 'lora_alpha', 'lora_dropout', 'bias', 'target_modules']
            lora_config = LoraConfig(**{k: adapter_config[k] for k in lora_keys})
            self.perceptual_model = get_peft_model(self.perceptual_model, lora_config)
            
            logging.info(f'Loading adapter weights from {checkpoint_root}')
            self.perceptual_model = PeftModel.from_pretrained(self.perceptual_model.base_model.model, checkpoint_root).to(self.device)
        else:
            logging.info(f'Loading entire model from {checkpoint_root}')
            sd = torch.load(os.path.join(checkpoint_root, f'epoch={epoch_load:02d}.ckpt'))['state_dict']
            self.load_state_dict(sd, strict=True)

    def __reset_val_metrics(self):
        for k, v in self.val_metrics.items():
            v.reset()

    def __prep_lora_model(self):
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias='none',
            target_modules=['qkv']
        )
        self.perceptual_model = get_peft_model(self.perceptual_model, config)

    def __prep_linear_model(self):
        for extractor in self.perceptual_model.extractor_list:
            extractor.model.requires_grad_(False)
            if self.feat_type == "embedding":
                extractor.proj.requires_grad_(False)
            self.perceptual_model.mlp.requires_grad_(True)

    def __save_lora_weights(self):
        if self.save_mode != 'entire_model':
            save_dir = os.path.join(self.trainer.callbacks[-1].dirpath, f'epoch_{self.trainer.current_epoch}')
            self.perceptual_model.save_pretrained(save_dir)

    def __validate_save_mode(self):
        save_options = {'adapter_only', 'entire_model', 'all'}
        assert self.save_mode in save_options, f'save_mode must be one of {save_options}, got {self.save_mode}'
        logging.info(f'Using save mode: {self.save_mode}')


def run(args, device):
    tag = args.tag if len(args.tag) > 0 else ""
    training_method = "lora" if args.use_lora else "mlp"
    exp_dir = os.path.join(args.log_dir,
                           f'{tag}_{str(args.model_type)}_{str(args.feat_type)}_{str(training_method)}_' +
                           f'lr_{str(args.lr)}_batchsize_{str(args.batch_size)}_wd_{str(args.weight_decay)}'
                           f'_hiddensize_{str(args.hidden_size)}_margin_{str(args.margin)}'
                           )
    if args.use_lora:
        exp_dir += f'_lorar_{str(args.lora_r)}_loraalpha_{str(args.lora_alpha)}_loradropout_{str(args.lora_dropout)}'

    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = TwoAFCDataset(root_dir=args.dataset_root, split="train", preprocess=get_preprocess(args.model_type))
    val_dataset = TwoAFCDataset(root_dir=args.dataset_root, split="val", preprocess=get_preprocess(args.model_type))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    logger = WandbLogger(save_dir=exp_dir, project='dreamsim_training')
    checkpointer = ModelCheckpoint(monitor='val_loss_ckpt',
                                                save_top_k=-1,
                                                save_last=True,
                                                filename='{epoch:02d}',
                                                mode='min') if args.save_mode != 'adapter_only' else None
    trainer = Trainer(devices=1,
                      accelerator='gpu',
                      log_every_n_steps=10,
                      logger=logger,
                      max_epochs=args.epochs,
                      default_root_dir=exp_dir,
                      callbacks=checkpointer,
                      num_sanity_val_steps=0,
                      )
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)

    logging.basicConfig(filename=os.path.join(exp_dir, 'exp.log'), level=logging.INFO, force=True)
    logging.info("Arguments: ", vars(args))

    model = LightningPerceptualModel(device=device, train_data_len=len(train_dataset), **vars(args))

    logging.info("Validating before training")
    trainer.validate(model, val_loader)
    logging.info("Training")
    trainer.fit(model, train_loader, val_loader)

    print("Done :)")


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args, device)
