import argparse
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


class EmbeddingsDataset(Dataset):
    def __init__(self, emb_dir_1, emb_dir_2):
        self.emb_dir_1 = emb_dir_1
        self.emb_dir_2 = emb_dir_2
        self.name_class_pairs = []
        # read emb names
        for cls in sorted(os.listdir(self.emb_dir_1)):
            for emb_name in os.listdir(os.path.join(self.emb_dir_1, cls)):
                self.name_class_pairs.append(
                    (os.path.join(cls, emb_name),
                     int(cls) - 1)
                )

    def __len__(self):
        return len(self.name_class_pairs)

    def __getitem__(self, item):
        pair = self.name_class_pairs[item]
        emb1 = np.squeeze(np.load(
            os.path.join(self.emb_dir_1, pair[0]))).astype(np.float32)
        emb2 = np.squeeze(np.load(
            os.path.join(self.emb_dir_2, pair[0]))).astype(np.float32)
        return np.concatenate((emb1, emb2)), pair[1]


# model
class Model(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_shape, output_shape)
        self.drop_out = nn.Dropout(0.3)

    def forward(self, x):
        x = self.drop_out(x)
        x = self.fc(x)
        return x


# train epoch
def train_epoch(model, criterion, loader, model_optimizer,
                criterion_optimizer, epoch, device):
    model.train().to(device)
    train_loss = []
    bar = tqdm(loader)
    for data, labels in bar:
        data, labels = data.to(device), labels.to(device)
        model_optimizer.zero_grad()
        criterion_optimizer.zero_grad()
        embeddings = model(data)
        loss = criterion(embeddings, labels)
        loss.backward()
        model_optimizer.step()
        criterion_optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('epoch %d loss: %.5f, smth: %.5f'
                            % (epoch, loss_np, smooth_loss))


# val epoch
def val_epoch(train_data, val_data, model, acc_calculator, epoch):
    model.eval().cpu()
    tester = testers.BaseTester(data_device='cpu')

    val_embeddings, val_labels = tester.get_all_embeddings(val_data, model)
    val_labels = val_labels.squeeze(1)
    val_accuracy = acc_calculator.get_accuracy(
        val_embeddings.half(), val_embeddings.half(),
        val_labels.half(), val_labels.half(), True)

    print("epoch {} val set accuracy (mP@5) = {}\n".format(
        epoch, val_accuracy["mean_average_precision"]))

    return val_accuracy["mean_average_precision"]

    # OOM part
    # train_embeddings, train_labels = tester.get_all_embeddings(train_data, model)
    # train_labels = train_labels.squeeze(1)
    # train_accuracy = acc_calculator.get_accuracy(
    #     train_embeddings.half(), train_embeddings.half(), train_labels.half(), train_labels.half(), True)
    # print("Train set accuracy (mP@5) = {}".format(train_accuracy["mean_average_precision"]))
    # del train_accuracy, train_embeddings, train_labels


# main
def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings-1', type=str, default='data/',
                        help='path to first embeddings folder')
    parser.add_argument('--embeddings-2', type=str, default='data/',
                        help='path to second embeddings folder')
    parser.add_argument('--saving-folder', type=str, required=True,
                        help='path to saving model folder')
    parser.add_argument('--input-shape', type=int, default=1536,
                        help='input embeddings shape')
    parser.add_argument('--output-shape', type=int, default=128,
                        help='model output shape')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='target device')
    parser.add_argument('--num-classes', type=int, default=17850,
                        help='classes number in training dataset')
    parser.add_argument('--model-lr', type=float, default=1e-3,
                        help='model learning rate')
    parser.add_argument('--criterion-lr', type=float, default=1e-3,
                        help='model criterion learning rate')
    parser.add_argument('--num-epochs', type=int, default=300,
                        help='training epochs number')
    parser.add_argument('--scheduler-gamma', type=float, default=0.1,
                        help='argument for StepLR scheduler')
    args = parser.parse_args()

    # create saving folder
    shutil.rmtree(args.saving_folder, ignore_errors=True)
    os.mkdir(args.saving_folder)

    # paths to train/val folders
    embeddings_train_dir_1 = os.path.join(args.embeddings_1, 'train')
    embeddings_train_dir_2 = os.path.join(args.embeddings_2, 'train')
    embeddings_val_dir_1 = os.path.join(args.embeddings_1, 'val')
    embeddings_val_dir_2 = os.path.join(args.embeddings_2, 'val')

    # datasets
    train_dataset = EmbeddingsDataset(emb_dir_1=embeddings_train_dir_1,
                                      emb_dir_2=embeddings_train_dir_2)
    val_dataset = EmbeddingsDataset(emb_dir_1=embeddings_val_dir_1,
                                    emb_dir_2=embeddings_val_dir_2)

    # loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8)

    # model
    model = Model(input_shape=args.input_shape,
                  output_shape=args.output_shape).to(args.device)

    # criterion
    criterion = losses.SubCenterArcFaceLoss(
        num_classes=args.num_classes,
        embedding_size=args.output_shape).to(args.device)

    # optimizers
    model_optimizer = optim.Adam(model.parameters(),
                                 lr=args.model_lr)
    criterion_optimizer = torch.optim.Adam(criterion.parameters(),
                                           lr=args.criterion_lr)

    # schedulers
    model_scheduler = optim.lr_scheduler.StepLR(model_optimizer,
                                                step_size=100,
                                                gamma=args.scheduler_gamma)
    criterion_scheduler = optim.lr_scheduler.StepLR(criterion_optimizer,
                                                    step_size=100,
                                                    gamma=args.scheduler_gamma)

    # metric
    mean_average_precision = AccuracyCalculator(
        include=('mean_average_precision', ),
        k=5, device=torch.device('cpu')
    )

    best_acc = 0
    best_epoch = 0
    for epoch in range(args.num_epochs):

        if epoch == 0:
            _ = val_epoch(
                train_data=train_dataset,
                val_data=val_dataset,
                model=model,
                acc_calculator=mean_average_precision,
                epoch=epoch)

        train_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            model_optimizer=model_optimizer,
            criterion_optimizer=criterion_optimizer,
            epoch=epoch,
            device=args.device)

        acc = val_epoch(
            train_data=train_dataset,
            val_data=val_dataset,
            model=model,
            acc_calculator=mean_average_precision,
            epoch=epoch)

        model_scheduler.step()
        criterion_scheduler.step()

        model_name = f'my_model_{epoch}_{acc:.4f}.pth'
        torch.save(model.state_dict(),
                   os.path.join(args.saving_folder, model_name))
        if best_acc < acc:
            print(f'Save best model. Epoch {epoch}')
            torch.save(model.state_dict(),
                       os.path.join(args.saving_folder, 'best_model.pth'))
            best_acc = acc
            best_epoch = epoch
        else:
            print(f'Accuracy din\'t change from epoch {best_epoch}')


if __name__ == '__main__':
    main()
