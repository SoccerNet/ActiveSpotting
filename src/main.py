import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting #,SoccerNetClipsOld
from model import Model
from train import trainer, test, testSpotting
from loss import NLLLoss


def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))
    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size, idx=args.idx_train, dataset=args.dataset)
        dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size, idx=args.idx_valid, dataset=args.dataset)
        dataset_Valid_metric  = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
    dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)


    if args.feature_dim is None:
        args.feature_dim = dataset_Test[0][1].shape[-1]
        print("feature_dim found:", args.feature_dim)
    # create model
    model = Model(weights=args.load_weights, input_size=args.feature_dim,
                  num_classes=dataset_Test.num_classes, window_size=args.window_size, 
                  vocab_size = args.vocab_size,
                  framerate=args.framerate, pool=args.pool).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)


    # training parameters
    if not args.test_only:
        criterion = NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=0, amsgrad=False)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience, eps=args.LRe)

        # start training
        trainer(train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # test on multiple splits [test/challenge]
    for split in args.split_test:
        dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=[split], version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)

        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

        results = testSpotting(test_loader, model=model, model_name=args.model_name, NMS_window=args.NMS_window, NMS_threshold=args.NMS_threshold, metric=args.metric)
        if results is None:
            continue

        a_mAP = results["a_mAP"]
        a_mAP_per_class = results["a_mAP_per_class"]
        a_mAP_visible = results["a_mAP_visible"]
        a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
        a_mAP_unshown = results["a_mAP_unshown"]
        a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

        logging.info("Best Performance at end of training ")
        logging.info("a_mAP visibility all: " +  str(a_mAP))
        logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
        logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))


    return 

def main_Active(args):
    from tqdm import tqdm
    import math
    import random

    # get count
    train_size = len(SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, 
                                    framerate=args.framerate, window_size=args.window_size, dataset=args.dataset))
    valid_size = len(SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, 
                                    framerate=args.framerate, window_size=args.window_size, dataset=args.dataset))

    size_train = train_size // args.sampling_epochs # 188 #1000
    size_valid = valid_size // args.sampling_epochs # 116 #300
    args.idx_train = np.random.randint(train_size, size=size_train)
    args.idx_valid = np.random.randint(valid_size, size=size_valid)

    # Loop Actively
    if args.active_scheme == "linespace":
        list_epochs = range(args.sampling_epochs)
    if args.active_scheme == "increasing":
        list_epochs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,21,23,25,30,35,40,50,60,70,80,90,100]

    for i, l in enumerate(list_epochs):

        if i == 0:
            continue
        n_part = l - list_epochs[i-1]
        # print(n_part)


        # if l > 0 and args.training_scheme == "continue":
        #     args.load_weights = f"models/{args.model_name}/model.pth.tar"


        if i > 1 and args.continue_training:
            args.load_weights = f"models/{args.model_name}/model.pth.tar"

        # Fast Training: reduce patience
        if args.training_scheme == "fast":
            args.patience = 5

        # Faster Training: reduce patience, increase LR and only 2 plateaus
        elif args.training_scheme == "faster":
            args.LR = 1e-2
            args.LRe = args.LR / 100
            args.patience = 5

        # # Fixed iteration
        # elif "fixed_iteration" in args.training_scheme:
        #     args.LR = 1e-2
        #     args.LRe = args.LR / 100
        #     args.patience = 5
        #     args.max_epochs = int(args.training_scheme.split("_")[-1])
        #     if i == 1:
        #         args.max_epochs = 20

        # if args.training_scheme == "fixed":
        #     Perform standard training

            

        #Main training
        main(args)


        # Maion active learning method absed on entropy
        if args.sampling_method == "entropy":


            # Load full training dataset to infer the active learning scores
            dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            # print(len(dataset_Test),dataset_Test.num_classes)
            dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            # print(len(dataset_Train),dataset_Train.num_classes)
            dataloader_train = torch.utils.data.DataLoader(dataset_Train,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.max_num_worker, pin_memory=True)
            dataloader_valid = torch.utils.data.DataLoader(dataset_Valid,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.max_num_worker, pin_memory=True)


            # Load model
            if args.feature_dim is None:
                args.feature_dim = dataset_Test[0][1].shape[-1]
                print("feature_dim found:", args.feature_dim)
            model = Model(weights=args.load_weights, input_size=args.feature_dim,
                        num_classes=dataset_Train.num_classes, window_size=args.window_size, 
                        vocab_size = args.vocab_size,
                        framerate=args.framerate, pool=args.pool).cuda()

            checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
            model.load_state_dict(checkpoint['state_dict'])


            criterion = NLLLoss()


            active_scores_training = list()
            with tqdm(enumerate(dataloader_train), total=len(dataloader_train)) as t:
                for i, (feats, labels) in t:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    output = model(feats)
                    active_scores_training += ( -torch.sum(output*torch.log2(output), dim=1) / math.log2(output.numel()) ).tolist()
            for i in args.idx_train:
                active_scores_training[i] = 0
            args.idx_train = np.append(args.idx_train, np.array(sorted(range(len(active_scores_training)), key=lambda x: active_scores_training[x])[-size_train*n_part:]))


            active_scores_validation = list()
            with tqdm(enumerate(dataloader_valid), total=len(dataloader_valid)) as t:
                for i, (feats, labels) in t:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    output = model(feats)
                    active_scores_validation += ( -torch.sum(output*torch.log2(output), dim=1) / math.log2(output.numel()) ).tolist()
            for i in args.idx_valid:
                active_scores_validation[i] = 0
            args.idx_valid = np.append(args.idx_valid, np.array(sorted(range(len(active_scores_validation)), key=lambda x: active_scores_validation[x])[-size_valid*n_part:]))

        # Inverse of entropy score. Not in paper. Worst that Passive learning
        elif args.sampling_method == "entropy_inv":

            # Load full training dataset to infer the active learning scores
            dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            # print(len(dataset_Test),dataset_Test.num_classes)
            dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            # print(len(dataset_Train),dataset_Train.num_classes)
            dataloader_train = torch.utils.data.DataLoader(dataset_Train,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.max_num_worker, pin_memory=True)
            dataloader_valid = torch.utils.data.DataLoader(dataset_Valid,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.max_num_worker, pin_memory=True)


            # Load model
            if args.feature_dim is None:
                args.feature_dim = dataset_Test[0][1].shape[-1]
                print("feature_dim found:", args.feature_dim)
            model = Model(weights=args.load_weights, input_size=args.feature_dim,
                        num_classes=dataset_Train.num_classes, window_size=args.window_size, 
                        vocab_size = args.vocab_size,
                        framerate=args.framerate, pool=args.pool).cuda()

            checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
            model.load_state_dict(checkpoint['state_dict'])


            criterion = NLLLoss()


            active_scores_training = list()
            with tqdm(enumerate(dataloader_train), total=len(dataloader_train)) as t:
                for i, (feats, labels) in t:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    output = model(feats)
                    active_scores_training += ( -torch.sum(output*torch.log2(output), dim=1) / math.log2(output.numel()) ).tolist()
            for i in args.idx_train:
                active_scores_training[i] = 0
            args.idx_train = np.append(args.idx_train, np.array(sorted(range(len(active_scores_training)), key=lambda x: active_scores_training[x])[:size_train*n_part]))


            active_scores_validation = list()
            with tqdm(enumerate(dataloader_valid), total=len(dataloader_valid)) as t:
                for i, (feats, labels) in t:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    output = model(feats)
                    active_scores_validation += ( -torch.sum(output*torch.log2(output), dim=1) / math.log2(output.numel()) ).tolist()
            for i in args.idx_valid:
                active_scores_validation[i] = 0
            args.idx_valid = np.append(args.idx_valid, np.array(sorted(range(len(active_scores_validation)), key=lambda x: active_scores_validation[x])[:size_valid*n_part]))
    
        # Confidence-based active sampling
        elif "confidence" in args.sampling_method:


            # Load full training dataset to infer the active learning scores
            dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            # print(len(dataset_Test),dataset_Test.num_classes)
            dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size, dataset=args.dataset)
            # print(len(dataset_Train),dataset_Train.num_classes)
            dataloader_train = torch.utils.data.DataLoader(dataset_Train,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.max_num_worker, pin_memory=True)
            dataloader_valid = torch.utils.data.DataLoader(dataset_Valid,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.max_num_worker, pin_memory=True)


            # Load model
            if args.feature_dim is None:
                args.feature_dim = dataset_Test[0][1].shape[-1]
                print("feature_dim found:", args.feature_dim)
            model = Model(weights=args.load_weights, input_size=args.feature_dim,
                        num_classes=dataset_Train.num_classes, window_size=args.window_size, 
                        vocab_size = args.vocab_size,
                        framerate=args.framerate, pool=args.pool).cuda()

            checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
            model.load_state_dict(checkpoint['state_dict'])


            criterion = NLLLoss()

            c_ref = float(args.sampling_method[-3:])
            active_scores_training = list()
            with tqdm(enumerate(dataloader_train), total=len(dataloader_train)) as t:
                for i, (feats, labels) in t:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    output = model(feats)
                    # print(output.shape)
                    outp = torch.max(output, dim=1).values
                    # print(outp)
                    outp=outp.tolist()
                    # print(outp)
                    active_scores_training += [1-2*abs(score-c_ref) for score in outp]
                     # active_score = 2*sum([abs(s-0.5) for s in scores])/len(scores)
            for i in args.idx_train:
                active_scores_training[i] = 0
            args.idx_train = np.append(args.idx_train, np.array(sorted(range(len(active_scores_training)), key=lambda x: active_scores_training[x])[-size_train*n_part:]))


            active_scores_validation = list()
            with tqdm(enumerate(dataloader_valid), total=len(dataloader_valid)) as t:
                for i, (feats, labels) in t:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    output = model(feats)
                    # print(output.shape)
                    outp = torch.max(output, dim=1).values
                    # print(outp)
                    outp=outp.tolist()
                    # print(outp)
                    active_scores_validation += [1-2*abs(score-c_ref) for score in outp]
            for i in args.idx_valid:
                active_scores_validation[i] = 0
            args.idx_valid = np.append(args.idx_valid, np.array(sorted(range(len(active_scores_validation)), key=lambda x: active_scores_validation[x])[-size_valid*n_part:]))

        # Random Sampling
        elif args.sampling_method =="random":

            active_scores_training = np.random.uniform(0.0, 1.0, size=train_size)
            active_scores_validation = np.random.uniform(0.0, 1.0, size=valid_size)

            for i in args.idx_train:
                active_scores_training[i] = 0
            args.idx_train = np.append(args.idx_train, np.array(sorted(range(len(active_scores_training)), key=lambda x: active_scores_training[x])[-size_train*n_part:]))


            for i in args.idx_valid:
                active_scores_validation[i] = 0
            args.idx_valid = np.append(args.idx_valid, np.array(sorted(range(len(active_scores_validation)), key=lambda x: active_scores_validation[x])[-size_valid:]))










if __name__ == '__main__':


    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="/path/to/SoccerNet/",     help='Path for SoccerNet' )
    parser.add_argument('--dataset',   required=False, type=str, default="SoccerNet",  help='Type of dataset' )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_TF2_PCA512.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="Active",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--metric',   required=False, type=str,   default="loose",     help='meric to evalute (tight/loose)' )

    parser.add_argument('--sampling_method',   required=False, type=str, default="random",  help='Sampling method for swiping' )
    parser.add_argument('--sampling_epochs',   required=False, type=int, default=100, help='Number of active epochs' )
    parser.add_argument('--continue_training',   required=False, action='store_true', help='continue training from previous weights' )
    parser.add_argument('--training_scheme',   required=False, type=str, default="fixed", help='fast/faster/fixed' )
    parser.add_argument('--active_scheme',   required=False, type=str, default="linespace", help='linespace/increasing' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test"], help='list of split for testing')

    parser.add_argument('--idx_train', nargs='+', type=float, default=None, help='indexes of sample used in training')
    parser.add_argument('--idx_valid', nargs='+', type=float, default=None, help='indexes of sample used in validation')

    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--feature_dim', required=False, type=int,   default=None,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=10,     help='Number of chunks per epoch' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--window_size', required=False, type=int,   default=15,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool' )
    parser.add_argument('--vocab_size',       required=False, type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--NMS_window',       required=False, type=int,   default=30, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.0, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size', required=False, type=int,   default=256,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,   default=0, help='seed for reproducibility')

    # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()
    logging.info('Starting main function')
    
    main_Active(args)

    logging.info(f'Total Execution Time is {time.time()-start} seconds')

