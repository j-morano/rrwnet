import multiprocessing
import os

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from utils_pytorch import EarlyStopReduceLROnPlateau, save_model, save_to_csv
from factories import ModelFactory, LossesFactory



def learning_curves(training, validation, outfile):
    """Builds learning curves: training and validation losses along
    iterations.
    """
    plt.rcParams["figure.figsize"] = [16, 9]
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    assert isinstance(ax1, Axes)
    x, y1 = zip(*training)
    ax1.plot(x, y1, 'b', label='training')

    x, y1 = zip(*validation)
    ax1.plot(x, y1, 'r', label='validation')

    ax1.set_yscale('log')

    ax1.legend()

    fig.savefig(outfile)
    plt.close(fig)


class R2Vessels:

    def __init__(
        self,
        base_channels=64,
        in_channels=3,
        out_channels=3,
        num_iterations=5,
        model=None,
        gpu_id=None,
        criterion=None,
        base_criterion=None,
        learning_rate=1e-4
    ):
        current = multiprocessing.current_process()
        self.process_id = str(current.pid)

        self.use_cuda = torch.cuda.is_available()

        if gpu_id is None:
            self.device = torch.device('cuda', 0)
            torch.cuda.set_device(0)
        else:
            self.device = torch.device('cuda', gpu_id)
            torch.cuda.set_device(gpu_id)

        ### Loss
        self.criterion_name = criterion
        losses_factory = LossesFactory()
        if base_criterion is not None:
            base_criterion = losses_factory.create_class(base_criterion)
            self.criterion = losses_factory.create_class(criterion, base_criterion=base_criterion)
        else:
            self.criterion = losses_factory.create_class(criterion)

        ### Model
        self.model_name = model
        self.model = ModelFactory().create_class(
            model,
            input_ch=in_channels,
            output_ch=out_channels,
            base_ch=base_channels,
            num_iterations=num_iterations
        )
        if self.use_cuda:
            self.model.cuda()

        ### Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999)
        )

        # Number of images presented to the network
        self.iter = 0


    def train_epoch(self, r2v_loader):
        total_loss = 0.0

        self.model.train()

        for sample in r2v_loader:
            data = sample[1]

            retino = data[0].cuda(non_blocking=True).requires_grad_(True)
            vessels = data[1].cuda(non_blocking=True).requires_grad_(False)
            mask = data[2].cuda(non_blocking=True).requires_grad_(False)

            self.optimizer.zero_grad()

            predictions = self.model(retino)
            loss = self.criterion(predictions, vessels, mask)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            self.iter += 1

        pattern = '\n|{}| [PID: {}, {}, {}] >> training epoch mean loss: {}'
        avg_loss = total_loss / len(r2v_loader)
        print(pattern.format(
            self.iter,
            self.process_id,
            self.model_name,
            self.criterion_name,
            avg_loss
        ))

        return [avg_loss]


    def test(self, r2v_dataloader, prefix_to_save=None):
        with torch.no_grad():
            total_loss = 0.0

            self.model.eval()

            for sample in r2v_dataloader:
                try:
                    k = sample[0].numpy()[0]
                except AttributeError:
                    k = sample[0][0]
                data = sample[1]

                retino = data[0].cuda(non_blocking=True)
                vessels = data[1].cuda(non_blocking=True)
                mask = data[2].cuda(non_blocking=True)

                predictions = self.model(retino)
                loss = self.criterion(predictions, vessels, mask)

                if prefix_to_save is not None:
                    self.criterion.save_predicted(predictions, prefix_to_save + str(k) + '.png')

                total_loss += loss.item()

            pattern = '|{}| [PID: {}, {}, {}] >> validation mean loss: {}'
            avg_loss = total_loss / len(r2v_dataloader)
            print(pattern.format(
                self.iter,
                self.process_id,
                self.model_name,
                self.criterion_name,
                avg_loss
            ))

            return [avg_loss]

    def training(
        self,
        train_loader,
        test_loader,
        path_to_save,
        init_iter=0,
        save_period=25,
        scheduler_patience=25,
        stopping_patience=25
    ):
        # Initialize the csv files
        save_to_csv([['best_loss', 'iter']], os.path.join(path_to_save, 'best_loss.csv'))
        save_to_csv([['loss', 'iter']], os.path.join(path_to_save, 'train_loss.csv'))
        save_to_csv([['loss', 'iter']], os.path.join(path_to_save, 'test_loss.csv'))

        train_loss = list()
        test_loss = list()
        all_train_loss = list()
        all_test_loss = list()

        scheduler = EarlyStopReduceLROnPlateau(
            self.optimizer,
            self.model,
            path_to_save,
            factor=0.1,
            patience=scheduler_patience,
            patience_stopping=stopping_patience,
            verbose=True,
            cooldown=0,
            threshold=0,
            min_lr=1e-8,
            eps=0
        )

        self.iter = init_iter

        test_count = 0
        while scheduler.training():
            save = (test_count % save_period == 0)

            if save:
                save_path = os.path.join(path_to_save, str(self.iter))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                prefix_to_save = save_path+'/'
            else:
                prefix_to_save = None

            train_loss.append([self.iter] + self.train_epoch(train_loader))
            test_loss.append([self.iter] + self.test(test_loader, prefix_to_save))

            is_best = scheduler.step(test_loss[-1][1], self.iter)

            if is_best:
                save_to_csv([[str(test_loss[-1][1]),str(self.iter)]],
                             os.path.join(path_to_save, 'best_loss.csv'))
                save_model(self.model, path_to_save + '/generator_best.pth')

            if save:
                save_to_csv(train_loss, os.path.join(path_to_save, 'train_loss.csv'))
                save_to_csv(test_loss, os.path.join(path_to_save, 'test_loss.csv'))
                all_train_loss += train_loss
                all_test_loss += test_loss
                train_loss = []
                test_loss = []
                learning_curves(all_train_loss, all_test_loss, path_to_save + '/learning_curves.svg')

            test_count += 1

        if len(train_loss) > 0:
            save_to_csv(train_loss, os.path.join(path_to_save, 'train_loss.csv'))
        if len(test_loss) > 0:
            save_to_csv(test_loss, os.path.join(path_to_save, 'test_loss.csv'))
        save_model(self.model, path_to_save + '/generator_last.pth')
        learning_curves(all_train_loss, all_test_loss, path_to_save + '/learning_curves.svg')
