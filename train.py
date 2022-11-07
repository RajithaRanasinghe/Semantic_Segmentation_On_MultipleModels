
import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from torch.autograd import Variable
import time
from datetime import datetime
from distutils.dir_util import copy_tree
import logging
import sys
from tensorboardX import SummaryWriter

class Network():
    def __init__(self, parameters):
        super(Network, self).__init__()
        self.MyParameters = parameters



        self.BinaryCrossEntropy = torch.nn.BCELoss()

    def _init_logger(self):
        #self.parameters = {'epoch':200, 'datasetName':'dataset', 'LearningRate':5e-4, 'BatchSize':4, 'ModelInputHeight':256, 'ModelInputWidth':256, 'LogPath':'', 'TrainingImages':'', 'TrainingMasks':'', 'ValidationImages':'', 'ValidationMasks':'', 'OutputPath':'', 'Mode':'Training', 'Model':0, 'ModelName':''}
        
        self.model_name = self.MyParameters['ModelName']
        self.dataset_name = self.MyParameters['datasetName']

        log_dir = 'logs/' + self.model_name + '/' + self.dataset_name + '/train' + '/{}'.format(
            time.strftime('%Y%m%d-%H%M%S'))

        self.logger = self.get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.image_save_path = log_dir + "/saved_images"
        self.create_dir(self.image_save_path)
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)



    def run(self):
        #self.parameters = {'epoch':200, 'datasetName':'dataset', 'LearningRate':5e-4, 'BatchSize':4, 'ModelInputHeight':256, 'ModelInputWidth':256, 'LogPath':'', 'TrainingImages':'', 'TrainingMasks':'', 'ValidationImages':'', 'ValidationMasks':'', 'OutputPath':'', 'Mode':'Training', 'Model':0, 'ModelName':''}
        model = self.MyParameters['Model'](pretrained=False, progress=True, num_classes=1, aux_loss=None)
        
        try:
            model.cuda()
        except:
            model.cpu()

        optimizer = torch.optim.Adam(model.parameters(), self.MyParameters['LearningRate'])

        image_root = self.MyParameters['TrainingImages']
        gt_root = self.MyParameters['TrainingMasks']
        val_image_root = self.MyParameters['ValidationImages']
        val_gt_root = self.MyParameters['ValidationMasks']
        batchsize = self.MyParameters['BatchSize']
        trainsize = (self.MyParameters['ModelInputHeight'], self.MyParameters['ModelInputWidth'])
        NumEpoch = self.MyParameters['epoch']

        train_loader, val_loader = self.get_loader(image_root, gt_root, val_image_root, val_gt_root, batchsize, trainsize)

        total_step = len(train_loader)
        val_total_step = len(val_loader)

        #Training Loop
        print("Let's go!")
        for epoch in range(1, NumEpoch):
            running_dice = 0.0
            running_loss = 0.0

            for i, pack in enumerate(train_loader, start=1):
                optimizer.zero_grad()

                images, gts = pack
                images = Variable(images)
                gts = Variable(gts)
                try:
                    images = images.cuda()
                    gts = gts.cuda()
                except:
                    images = images.cpu()
                    gts = gts.cpu()

                if self.MyParameters['ModelName'] != 'U-net':
                    pred = torch.sigmoid(model(images)['out']) # This is for torch models
                elif self.MyParameters['ModelName'] == 'U-net':
                    pred = torch.sigmoid(model(images)) # This is for U-net

                loss = self.calc_loss(pred, gts)

                loss.backward()
                optimizer.step()

                dice_coe = self.dice(pred, gts)
                running_dice += dice_coe
                running_loss += loss


                if i % 10 == 0 or i == total_step:
                    self.logger.info(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}, dice_coe: {:.4f}'.
                            format(datetime.now(), epoch, NumEpoch, i, total_step, loss.item(), dice_coe))

            epoch_dice = running_dice / len(train_loader)
            epoch_loss = running_loss / len(train_loader)
            self.logger.info('Train dice coeff: {}'.format(epoch_dice))
            self.writer.add_scalar('Train/DSC', epoch_dice, epoch)
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            #Comented to test deeplab
            val_running_dice = 0.0
            val_running_loss = 0.0


            #for i, pack in enumerate(val_loader, start=1):
            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    #images, gts, name = pack
                    images, gts = pack
                    images = Variable(images)
                    gts = Variable(gts)

                    try:
                        images = images.cuda()
                        gts = gts.cuda()
                    except:
                        images = images.cpu()
                        gts = gts.cpu()

                    if self.MyParameters['ModelName'] != 'U-net':
                        pred = torch.sigmoid(model(images)['out']) # This is for torch models
                    elif self.MyParameters['ModelName'] == 'U-net':
                        pred = torch.sigmoid(model(images)) # This is for U-net

                val_loss = self.calc_loss(pred, gts)
                self.visualize_val_gt(gts, 'gt{}'.format(i))
                self.visualize_val_prediction(pred, 'pd{}'.format(i))



                val_dice_coe = self.dice(pred, gts)
                val_running_dice += val_dice_coe
                val_running_loss += val_loss
            

                if i % 10 == 0 or i == total_step:
                    self.logger.info(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Validation loss: {:.4f}, dice_coe: {:.4f}'.
                            format(datetime.now(), epoch, NumEpoch, i, val_total_step, val_loss.item(), val_dice_coe))


            val_epoch_dice = val_running_dice / len(val_loader)
            val_epoch_loss = val_running_loss / len(val_loader)
            self.logger.info('Validation dice coeff: {}'.format(val_epoch_dice))
            self.writer.add_scalar('Validation/DSC', val_epoch_dice, epoch)
            self.writer.add_scalar('Validation/Loss', val_epoch_loss, epoch)

            mdice_coeff = val_epoch_dice

            if self.best_dice_coeff < mdice_coeff:
                self.best_dice_coeff = mdice_coeff
                self.save_best = True

                if not os.path.exists(self.image_save_path):
                    os.makedirs(self.image_save_path)

                copy_tree(self.image_save_path, self.save_path + '/best_model_predictions')
                self.patience = 0
            else:
                self.save_best = False
                self.patience += 1

            # adjust_lr(optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
            Checkpoints_Path = self.save_path + '/Checkpoints'
            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best:
                torch.save(model.state_dict(), Checkpoints_Path + '/Model_{}_{}_{}.pth'.format(self.model_name, self.dataset_name, datetime.now()))

            self.logger.info('current best dice coef {}'.format(self.best_dice_coeff))
            self.logger.info('current patience :{}'.format(self.patience))
        

    def dice_loss(self, pred_mask, true_mask):
        loss = 1 - self.dice(pred_mask, true_mask)

        return loss


    def calc_loss(self,pred, target, bce_weight=0.2):
        bce = self.BinaryCrossEntropy(pred, target)
        dice = self.dice_loss(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)

        return loss


    def dice(self, pred, target):
        intersection = (abs(target - pred) < 0.05).sum()
        cardinality = (target >= 0).sum() + (pred >= 0).sum()

        return 2.0 * intersection / cardinality

    def create_exp_dir(self, path, desc='Experiment dir: {}'):
        if not os.path.exists(path):
            os.makedirs(path)
        print(desc.format(path))

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


    def get_logger(self, log_dir):
        self.create_exp_dir(log_dir)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger('Nas Seg')
        logger.addHandler(fh)
        return logger


    def get_loader(self, image_root, gt_root, val_image_root, val_gt_root, batchsize, trainsize, shuffle=False, num_workers=4, pin_memory=True):

        train_dataset = TrainingDataset(image_root, gt_root, trainsize)
        val_dataset = ValidationDataset(val_image_root, val_gt_root, trainsize)
        # = TestDataset(test_image_root, test_gt_root, trainsize)

        data_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batchsize,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    shuffle=True)

        val_loader = data.DataLoader(dataset=val_dataset,
                                    batch_size=batchsize,
                                    #batch_size=2,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    shuffle=False)

        return data_loader, val_loader


class TrainingDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def img_gt_rotate(self, img,gt):
        angle = transforms.RandomRotation.get_params([-180, 180])
        image_rotate = tf.rotate(img, angle, resample=Image.NEAREST)
        gt_rotate = tf.rotate(gt, angle, resample=Image.NEAREST)
        return image_rotate,gt_rotate

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

class ValidationDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)



        name = self.images[index].split("/")[-1].split(".jpg")[0]

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

class TestDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)



        name = self.images[index].split("/")[-1].split(".jpg")[0]

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size