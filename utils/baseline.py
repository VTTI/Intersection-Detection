import glob
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import dataset_baseline
from utils.model import baseline


class RunBaseline:
    def __init__(self, int_dir, non_int_dir, model_name, optimizer, num_epochs, batch_size, log_step, out_dir,
                 lr, resize_shape, comment, mode, custom_image_path):

        self.comment = f"LR_{lr}_OPT_{optimizer}_BATCH_{batch_size}_SHAPE_{resize_shape}_{comment}"
        self.int_dir = int_dir
        self.custom = custom_image_path
        self.non_int_dir = non_int_dir
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_step = log_step
        self.out_dir = os.path.join(out_dir, model_name)
        self.lr = lr
        self.resize_shape = resize_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = baseline(name=model_name).to(self.device)
        self.opt = self.get_optimizer(self.optimizer, self.model, self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='max',cooldown=10, patience=10,min_lr=1e-6, verbose=True)
        self.scheduler_epoch = torch.optim.lr_scheduler.StepLR(self.opt,step_size=20,gamma=0.1,verbose=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, 'runs', f'{self.comment}'))
        self.train_set, self.test_set, self.val_set, self.weight_int, self.weight_tod = dataset_baseline(
            self.int_dir, self.non_int_dir, out_dir=self.out_dir, resize_shape=self.resize_shape)
        self.criterion_zone = nn.CrossEntropyLoss(weight=self.weight_int).to(self.device)
        self.criterion_tod = nn.CrossEntropyLoss(weight=self.weight_tod).to(self.device)
        # data loaders
        if (mode == "train") or (mode == "test"):
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=8)
            self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

        self.best_model_path = os.path.join(self.out_dir, 'weights', self.comment, f"{self.comment}_best_weight.pth")

    def train(self):

        # make relevant directories
        os.makedirs(os.path.join(self.out_dir, 'weights', self.comment), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'logs', self.comment), exist_ok=True)

        # initialize logger
        filename = os.path.join(self.out_dir, 'logs', self.comment, f'{self.comment}_train.log')
        logging.basicConfig(filename=filename, filemode='w',
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.INFO)
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

        # initializations
        total_step = len(self.train_loader)
        best_f = None

        logging.info(f'Total number of training example: {len(self.train_set)}')
        logging.info(
            f"Training Configurations model backbone: {self.model_name}; optimizer: {self.optimizer}; lr: {self.lr};"
            f" num epochs: {self.num_epochs}; batch size: {self.batch_size}; image shape: {self.resize_shape}, "
            f"device:{self.device}")

        for epoch in range(self.num_epochs):

            
            logging.info(f"--- Epoch {epoch + 1} ---")
            logging.info(f"--- Current Optimizer: {self.optimizer} ---")
            #logging.info(f"--- Learning Rate at previous epoch : {self.scheduler_epoch.get_last_lr()} ---")
            logging.info(f"--- Current Learning Rate : {self.opt.param_groups[0]['lr']} ---")
            self.model.train()

            for itr, data in enumerate(self.train_loader):
                self.opt.zero_grad()
                image_tensor = data["image"].to(self.device)
                zone = data["label"].to(self.device)  # zone
                tod = data["tod"].to(self.device)  # time of day
                
                ## Check if binary of two bit model(D/N included or not)
                if self.model_name == 'resnext50_2':
                    ## prediction
                    predictions = self.model(image_tensor)  # shape:[-1, 4]
                    zone_pred = predictions[:]  # intersection prediction
                    
                    ## loss
                    loss = self.criterion_zone(zone_pred, zone)
                else:
                    ## predictions
                    predictions = self.model(image_tensor)  # shape:[-1, 4]
                    zone_pred = predictions[:, 0:2]  # intersection prediction
                    tod_pred = predictions[:, -2:]  # time of day prediction
                    ## loss
                    loss = self.criterion_zone(zone_pred, zone) + self.criterion_tod(tod_pred, tod)
                    #loss = self.criterion_zone(zone_pred, zone)


                loss.backward()
                self.opt.step()

                # log performance
                if itr % self.log_step == 0:
                    logging.info(f'Running logs for epoch: {epoch + 1} Step: {itr}/{total_step}'
                                 f'---> Loss: {round(loss.item(), 4)}')

            logging.info('---Performing Validation---')
            p_train_zone, p_train_tod, train_loss = self.eval(self.train_loader)
            p_val_zone, p_val_tod, val_loss = self.eval(self.val_loader)

            # logging output
            logging.info(f'Training--->Loss: {round(train_loss, 4)}; '
                         f'F-Score-Zone: {round(p_train_zone[0] * 100, 2)}%;'
                         f'[tp, tn, fp, fn]-Zone: {p_train_zone[1:5]}, accuracy: {round(p_train_zone[-1] * 100, 2)}%')
            logging.info(f'Validation--->Loss: {round(val_loss, 4)};'
                         f'F-Score-Zone: {round(p_val_zone[0] * 100, 2)}%; '
                         f'[tp, tn, fp, fn]-Zone: {p_val_zone[1:5]}, accuracy: {round(p_val_zone[-1] * 100, 2)}%')

            # updating best score
            if best_f is None:
                best_f = p_val_zone[0]
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info('model saved')
            elif p_val_zone[0] > best_f:
                best_f = p_val_zone[0]
                logging.info(f"Updating best F score to: {round(best_f * 100, 2)}%\n")
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info('model saved')

            # scheduler update
            self.scheduler_epoch.step()
            self.scheduler.step(p_val_zone[0])
            # summary writer 
            self.writer.add_scalars('Loss', {'train loss': train_loss, 'val loss': val_loss}, epoch + 1)
            self.writer.add_scalars('Accuracy', {'train accuracy': p_train_zone[-1], 'val accuracy': p_val_zone[-1]}, epoch + 1)
            self.writer.add_scalars('LRDecay', {'lr': self.opt.param_groups[0]['lr']}, epoch + 1)
            self.writer.add_scalars('Fscore', {'train score': p_train_zone[0], 'val score': p_val_zone[0]}, epoch + 1)

            logging.info("-----------------------------------------------------")
        self.writer.close()

    def test(self, weight=None):
        print(f'---Performing testing on {self.model_name} test set---')
        print(f'Total number of testing example: {len(self.test_set)}')

        loader = DataLoader(dataset=self.test_set, batch_size=1, shuffle=False)
        p_test_zone, p_test_tod, test_loss = self.eval(loader, _test_=True,
                                                       path=self.best_model_path if weight is None else weight)
        print('\n'
              f'Testing--->Loss: {round(test_loss, 4)};'
              f'F-Score-Zone: {round(p_test_zone[0] * 100, 2)}%;'
              f'[tp, tn, fp, fn]-Zone: {p_test_zone[1:5]} accuracy: {round(p_test_zone[-1] * 100, 2)}%')

    def eval(self, loader, path=None, _test_=False):
        total_step = len(loader)
        loss = 0

        if _test_ is True:
            print("Loading weights for testing")
            try:
                self.model.load_state_dict(torch.load(path))
            except Exception as e:
                print("Enter valid path of the model weights! exiting program", e)
                exit()

        ys_zone = list()
        ys_preds_zone = list()
        ys_tod = list()
        ys_preds_tod = list()

        for data in tqdm(loader, file=sys.stdout):
            image_tensor = data["image"].to(self.device)
            zone = data["label"].to(self.device)  # zone
            tod = data["tod"].to(self.device)  # time of day

            # predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(image_tensor)  # shape:[-1, 4]
            
            ## Check for binary model
            ## All other models with 2 bit config
            if self.model_name != 'resnext50_2':
                zone_pred = predictions[:, 0:2]  # intersection prediction
                tod_pred = predictions[:, -2:]  # time of day prediction
                loss += F.cross_entropy(zone_pred, zone) + F.cross_entropy(tod_pred, tod)
                #loss += F.cross_entropy(zone_pred, zone) 
                
                ys_zone.append(zone)
                ys_tod.append(tod)
                ys_preds_zone.append(torch.argmax(zone_pred, dim=1))
                ys_preds_tod.append(torch.argmax(tod_pred, dim=1))
            
            ## Binary Classifier 
            else:
                zone_pred = predictions  # intersection prediction
                loss += F.cross_entropy(zone_pred, zone) 
                ys_zone.append(zone)
                ys_preds_zone.append(torch.argmax(zone_pred, dim=1))
        loss = loss.item()
        Ys_zone = torch.cat(ys_zone, dim=0)
        Ys_preds_zone = torch.cat(ys_preds_zone, dim=0)
        if self.model_name != 'resnext50_2':
            Ys_tod = torch.cat(ys_tod, dim=0)
            Ys_preds_tod = torch.cat(ys_preds_tod, dim=0)

        ZONE = self.performance_metric(Ys_zone, Ys_preds_zone, plot=_test_)
        if self.model_name != 'resnext50_2':
            TOD = self.performance_metric(Ys_tod, Ys_preds_tod, plot=_test_)
            return ZONE, TOD, (loss / total_step)
        else:
            TOD = []
            return ZONE,TOD,(loss/total_step) 

    def performance_metric(self, y_true, y_preds, beta=0.5, plot=False):
        # beta = 1 --> F1
        # beta = 0.5 --> F2

        y_true = y_true.detach()
        y_true = y_true.to("cpu")
        y_preds = y_preds.detach()
        y_preds = y_preds.to("cpu")

        y_true = np.asarray(y_true)
        y_preds = np.asarray(y_preds)

        tp = np.sum((y_preds == 1) & (y_true == 1))
        tn = np.sum((y_preds == 0) & (y_true == 0))
        fp = np.sum((y_preds == 1) & (y_true == 0))
        fn = np.sum((y_preds == 0) & (y_true == 1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        score = ((1 + beta ** 2) * precision * recall) / ((beta ** 2) * precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if plot:
            path = os.path.join(self.out_dir, 'confusion-matrix', self.comment)
            os.makedirs(path, exist_ok=True)
            cf_matrix = confusion_matrix(y_true, y_preds)
            cf_matrix = cf_matrix.astype(float)
            for i in range(cf_matrix.shape[0]):
                cf_matrix[i] = cf_matrix[i] / np.sum(cf_matrix[i])
                cf_matrix[i] = np.around(cf_matrix[i], decimals=3)

            print('Confusion Matrix')
            plt.figure(figsize=[10, 8])
            plt.title('Confusion Matrix', fontsize=24)
            sns.heatmap(cf_matrix, cmap='Oranges', linewidths=3, annot_kws={'fontsize': 24},
                        xticklabels=["non-intersection", "intersection"],
                        yticklabels=["non-intersection", "intersection"],
                        square=True, annot=True)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.savefig(os.path.join(path, "confusion-matrix.png"))
            plt.show()

        return [score, tp, tn, fp, fn, accuracy]

    @staticmethod
    def get_optimizer(optimizer, network, lr):
        if optimizer == 'ADAM':
            opt = optim.Adam(network.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
        elif optimizer == 'ASGD':
            opt = optim.ASGD(network.parameters(), lr=lr)
        elif optimizer == 'SGD':
            opt = optim.SGD(network.parameters(), lr=lr, momentum=0.99)
        elif optimizer == 'RMSprop':
            opt = optim.RMSprop(network.parameters(), lr=lr, momentum=0.99)
        elif optimizer == 'AdamW':
            opt = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-5)
        else:
            opt = optim.SGD(network.parameters(), lr=lr, momentum=0.99)
        return opt

    @staticmethod
    def get_transform(resize_shape):

        transform = transforms.Compose([transforms.Resize((resize_shape[0], resize_shape[1])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        return transform

    def test_on_single_images(self, weight=None):
        root = os.path.join(self.custom, "**", "*.jpg")
        test_data = list(glob.iglob(root, recursive=True))
        os.makedirs(os.path.join(self.out_dir, 'predictions', self.comment), exist_ok=True)
        if weight is None:  # no weight given, use default weights
            weight = self.best_model_path
        transform = self.get_transform(self.resize_shape)
        self.model.load_state_dict(torch.load(weight, map_location=self.device))
        for path in tqdm(test_data):
            image = Image.open(path)
            image_copy = image.copy()
            draw = ImageDraw.Draw(image_copy)
            image = transform(image).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(image)  # shape:[-1, 4] or [-1,2] if binary
            zone_pred = torch.argmax(predictions[:, 0:2]).item()  # Intersection prediction
            if zone_pred == 1:  # work zone
                draw.text((10, 10), "intersection", (255, 0, 0))
            elif zone_pred == 0:  # non work zone
                draw.text((10, 10), "non-intersection", (255, 0, 0))
            image_copy.save(os.path.join(self.out_dir, 'predictions', self.comment,
                                         path.split(os.sep)[-1].replace(".jpg", "_predicted_patch.jpg")))
