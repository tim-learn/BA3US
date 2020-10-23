import argparse
import os, random, pdb, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import network, my_loss
import lr_schedule, data_list

def image_train(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_test(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()
    return accuracy, hist_tar, mean_ent

def train(args):
    ## prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())
    dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker)

    if "ResNet" in args.net:
        params = {"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.ResNetFc(**params)
    
    if "VGG" in args.net:
        params = {"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.VGGFc(**params)

    base_network = base_network.cuda()

    ad_net = network.AdversarialNetwork(base_network.output_num(), 1024, args.max_iterations).cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    # ad_net = torch.nn.DataParallel(ad_net).cuda()
    # base_network = torch.nn.DataParallel(base_network).cuda() 

    ## set optimizer
    optimizer_config = {"type":torch.optim.SGD, "optim_params":
                        {'lr':args.lr, "momentum":0.9, "weight_decay":5e-4, "nesterov":True}, 
                        "lr_type":"inv", "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}
                    }
    optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    class_weight = None
    best_ent = 1000
    total_epochs = args.max_iterations // args.test_interval

    for i in range(args.max_iterations + 1):
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            # obtain the class-level weight and evalute the current model
            base_network.train(False)
            temp_acc, class_weight, mean_ent = image_classification(dset_loaders, base_network)
            class_weight = class_weight.cuda().detach()

            temp = [round(i, 4) for i in class_weight.cpu().numpy().tolist()]
            log_str = str(temp)
            args.out_file.write(log_str+"\n")
            args.out_file.flush()
            
            print(class_weight)
            if mean_ent < best_ent:
                best_ent, best_acc = mean_ent, temp_acc
                best_model = base_network.state_dict()
            log_str = "iter: {:05d}, precision: {:.5f}, mean_entropy: {:.5f}".format(i, temp_acc, mean_ent)
            args.out_file.write(log_str+"\n")
            args.out_file.flush()
            print(log_str)

        if i % args.test_interval == 0:
            if args.mu > 0:
                epoch = i // args.test_interval
                len_share = int(max(0, (train_bs // args.mu) * (1 - epoch / total_epochs)))
            elif args.mu == 0:
                len_share = 0  # no augmentation
            else:
                len_share = int(train_bs // abs(args.mu))  # fixed augmentation
            log_str = "\n{}, iter: {:05d}, source/ target/ middle: {:02d} / {:02d} / {:02d}\n".format(args.name, i, train_bs, train_bs, len_share)
            args.out_file.write(log_str)
            args.out_file.flush()
            print(log_str)

            dset_loaders["middle"] = None
            if not len_share == 0:
                dset_loaders["middle"] = DataLoader(dsets["source"], batch_size=len_share, shuffle=True, num_workers=args.worker, 
                    drop_last=True)
                iter_middle = iter(dset_loaders["middle"])

        # train one iter
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])
        if dset_loaders["middle"] is not None and i % len(dset_loaders["middle"]) == 0:
            iter_middle = iter(dset_loaders["middle"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, _ = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        if class_weight is not None and args.weight_cls and class_weight[labels_source].sum() == 0:
            continue

        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        if dset_loaders["middle"] is not None:
            inputs_middle, labels_middle = iter_middle.next()
            features_middle, outputs_middle = base_network(inputs_middle.cuda())
            features = torch.cat((features_source, features_target, features_middle), dim=0)
            outputs = torch.cat((outputs_source, outputs_target, outputs_middle), dim=0)
        else:
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)

        cls_weight = torch.ones(outputs.size(0)).cuda()
        if class_weight is not None and args.weight_aug:
            cls_weight[0:train_bs] = class_weight[labels_source]
            if dset_loaders["middle"] is not None:
                cls_weight[2*train_bs::] = class_weight[labels_middle]

        # compute source cross-entropy loss
        if class_weight is not None and args.weight_cls:
            src_ = torch.nn.CrossEntropyLoss(reduction='none')(outputs_source, labels_source)
            weight = class_weight[labels_source].detach()
            src_loss = torch.sum(weight * src_) / (1e-8 + torch.sum(weight).item())
        else:
            src_loss = torch.nn.CrossEntropyLoss()(outputs_source, labels_source)

        softmax_out = torch.nn.Softmax(dim=1)(outputs)
        entropy = my_loss.Entropy(softmax_out)  
        transfer_loss = my_loss.DANN(features, ad_net, entropy, network.calc_coeff(i, 1, 0, 10, args.max_iterations), cls_weight, len_share)       

        softmax_tar_out = torch.nn.Softmax(dim=1)(outputs_target)
        tar_loss = torch.mean(my_loss.Entropy(softmax_tar_out))
        
        total_loss = src_loss + transfer_loss + args.ent_weight * tar_loss
        if args.cot_weight > 0:
            if class_weight is not None and args.weight_cls:
                cot_loss = my_loss.marginloss(outputs_source, labels_source, args.class_num, alpha=args.alpha, weight=class_weight[labels_source].detach())
            else:
                cot_loss = my_loss.marginloss(outputs_source, labels_source, args.class_num, alpha=args.alpha)
            total_loss += cot_loss * args.cot_weight
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    torch.save(best_model, os.path.join(args.output_dir, "best_model.pt"))
    
    log_str = 'Acc: ' + str(np.round(best_acc*100, 2)) + "\n" + 'Mean_ent: ' + str(np.round(best_ent, 3)) + '\n'
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    return best_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='run')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers") 
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50", "VGG16"])
    
    parser.add_argument('--dset', type=str, default='office_home', choices=["office", "office_home", "imagenet_caltech"])
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--mu', type=int, default=4, help="init augmentation size = batch_size//mu")
    parser.add_argument('--ent_weight', type=float, default=0.1)
    parser.add_argument('--cot_weight', type=float, default=1.0, choices=[0, 1, 5, 10])
    parser.add_argument('--weight_aug', type=bool, default=True)
    parser.add_argument('--weight_cls', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=1)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        k = 25
        args.class_num = 65
        args.max_iterations = 5000
        args.test_interval = 500
        args.lr=1e-3

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        k = 10
        args.class_num = 31
        args.max_iterations = 2000
        args.test_interval = 200
        args.lr=1e-4

    if args.dset == 'imagenet_caltech':
        names = ['imagenet', 'caltech']
        k = 84
        args.class_num = 1000
        if args.s == 1:
            args.class_num = 256

        args.max_iterations = 40000
        args.test_interval = 4000
        args.lr=1e-3

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_folder = './data/'
    args.s_dset_path = data_folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = data_folder + args.dset + '/' + names[args.t] + '_' + str(k) + '_list.txt'

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir = os.path.join('ckp/partial', args.net, args.dset, args.name, args.output)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.out_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    args.out_file.write(str(args)+'\n')
    args.out_file.flush()

    train(args)