import torchvision.transforms as T
import argparse
import torchvision
import models as MODEL
from torch.backends import cudnn
import  time
from utils import *
from dct import *
import higher
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import distributed
from tqdm import tqdm
import timm
import random
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=str, default='16/255')
parser.add_argument('--sgm_lambda', type=float, default=1.0)
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--method', type=str, default = 'fgsm')
parser.add_argument('--linbp_layer', type=str, default='3_1')
# parser.add_argument('--linbp_layer', type=int, default=23)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--save_dir', type=str, default = 'data/imagenet/temp')
parser.add_argument('--target_attack', default=False, action='store_true')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--imagenet_val_dir', type=str, default='../data/imagenet/val')
parser.add_argument('--model_name', type=str, default='resnet50')
parser.add_argument('--state_dict_dic', type=str, default='ckpt/')
parser.add_argument('--if_distributed', type=int, default=0)
parser.add_argument('--alpha', type=str, default='1/255')
parser.add_argument('--portion', type=float, default=0.2)
parser.add_argument('--relu_silu_temperature', type=float, default=1.)
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument('--first_order', action='store_true', default=False, help='use allow_unusedxiliary tower')###
parser.add_argument('--attack_exist', action='store_true', default=False, help='use allow_unusedxiliary tower')###
parser.add_argument('--inner_loop', type=int, default=2, help='number of inner loops for IAPTT')
parser.add_argument('--cg_steps', type=int, default=1)
parser.add_argument('--meta_steps', type=int, default=1)
parser.add_argument('--lamb', type=float, default=10.0)
parser.add_argument("--attack_lr", type=float, default=0.2, help="Tuning factor")

args = parser.parse_args()


def vec_to_grad(vec,model):
    pointer = 0
    res = []
    for param in model.parameters():
        num_param = param.numel()
        res.append(vec[pointer:pointer+num_param].view_as(param).data)
        pointer += num_param
    return res

def distance_loss(params, init_prams):
    loss = 0
    for p,ip in zip(params,init_prams):
        loss += torch.norm(p - ip, 2) ** 2
    return loss


def hv_prod(in_grad, x, params):
    hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
    hv = torch.nn.utils.parameters_to_vector(hv).detach()
    # precondition with identity matrix
    return hv/args.lamb + x


def CG(in_grad, outer_grad, params, model, cg_steps):
    x = outer_grad.clone().detach()
    r = outer_grad.clone().detach() - hv_prod(in_grad, x, params)
    p = r.clone().detach()
    for i in range(cg_steps):
        Ap = hv_prod(in_grad, p, params)
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new.clone().detach()
    return vec_to_grad(x,model)


def normalize(x, ms=None):
    if ms == None:
        ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - ms[0][i]) / ms[1][i]
    return x

def grad_inv(grad):
    return grad.neg()


def generate_pert(phi, return_entropy=False):
    adv_mean = phi[:, :3, :, :]
    adv_std = F.softplus(phi[:, 3:, :, :])
    rand_noise = torch.randn_like(adv_std)
    adv = torch.tanh(adv_mean + rand_noise * adv_std)
    # omit the constants in -logp
    negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
    entropy = negative_logp.mean()  # entropy
    if return_entropy:
        return adv, entropy
    else:
        return adv


def trans_incep(x):
    if 'incep' in args.save_dir:
        return normalize(x, ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data
    else:
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
        x = F.interpolate(x, size=(299,299))
        return normalize(x, ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data


def trans_pnas(x):
    x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
    x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
    x = F.interpolate(x, size=(331,331))
    return normalize(x, ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data


def trans_se(x):
    x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
    x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
    return normalize(x, ms = None).data


def trans_ori(x):
    if 'incep' in args.save_dir:
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
        return normalize(x, ms = None).data
    else:
        return normalize(x, ms = None).data


def test(model, trans, if_minus=False):
    target = torch.from_numpy(np.load(args.save_dir + '/labels.npy')).long()
    if args.target_attack:
        label_switch = torch.tensor(list(range(500, 1000)) + list(range(0, 500))).long()
        target = label_switch[target]
    img_num = 0
    count = 0
    advfile_ls = os.listdir(args.save_dir)
    for advfile_ind in range(len(advfile_ls)-1):
        adv_batch = torch.from_numpy(np.load(args.save_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
        if advfile_ind == 0:
            adv_batch_size = adv_batch.shape[0]
        img = adv_batch
        img_num += img.shape[0]
        label = target[advfile_ind * adv_batch_size : advfile_ind*adv_batch_size + adv_batch.shape[0]]
        label = label.to(device)
        img = img.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(trans(img)), dim=1).view(1,-1)
            if if_minus:
                pred = pred -torch.tensor(1, dtype=torch.long)
        count += (label != pred.squeeze(0)).sum().item()
        del pred, img
        del adv_batch
    return round(100. - 100. * count / img_num, 2) if args.target_attack else round(100. * count / img_num, 2)


if __name__ == '__main__':
    print(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(0)

    os.makedirs(args.save_dir, exist_ok=True)
    epsilon = eval(args.epsilon)
    batch_size = args.batch_size
    method = args.method
    save_dir = args.save_dir
    niters = args.niters
    target_attack = args.target_attack
    sgm_lambda = args.sgm_lambda
    device_id = args.device_id
    imagenet_val_dir = args.imagenet_val_dir
    selected_images_csv = 'data/imagenet/selected_imagenet_{}.csv'.format(args.model_name)
    model_name=args.model_name
    state_dict_dic = args.state_dict_dic
    if_distributed = args.if_distributed
    linbp_layer = args.linbp_layer
    alpha = eval(args.alpha)
    portion = args.portion
    ReLU_SiLU_Function.temperature = args.relu_silu_temperature

    if if_distributed:
        args.local_run = 0
        device_id = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(device_id)
        print(f"=> set cuda device = {device_id}")

        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        print(f"local_run:{args.local_run}")
        if args.local_run == 0:
            print("==> start init_process_group")
            dist.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(hours=24)
            )
        rank_id=int(os.environ["RANK"])
        world_size=int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(device_id))
    else:
        device = torch.device('cpu')
    print(device)

    if not args.attack_exist:
        print('Generate Adversarial images!')
        trans = T.Compose([
            T.Resize((256,256)),
            T.CenterCrop((224,224)),
            T.ToTensor()
        ])
        dataset = SelectedImagenet(imagenet_val_dir=imagenet_val_dir,
                                   selected_images_csv=selected_images_csv,
                                   transform=trans
                                   )
        if if_distributed:
            my_sampler=distributed.DistributedSampler(
                    dataset, shuffle=False)
            ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=my_sampler, num_workers = 8, pin_memory = False)
        else:
            ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = False)

        task_model_list=[]
        maml_task_model_1 = MODEL.inceptionv3.Inception3()
        maml_task_model_1.to(device)
        maml_task_model_1.load_state_dict(torch.load(state_dict_dic + 'inception_v3_google-1a9a5a14.pth'))
        maml_task_model_1.eval()

        task_model_list.append(maml_task_model_1)
        # maml_task_model_2 =timm.create_model('inception_resnet_v2', pretrained=True)#torch.nn.Sequential(SSA_Normalize(args.mean, args.std),pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
        # maml_task_model_2.to(device)
        # maml_task_model_2.eval()
        # task_model_list.append(maml_task_model_2)
        if model_name=='vgg19_bn':
            if 'ghost' in method:
                model = MODEL.vgg19_bn(pretrained=True)
            else:
                model = torchvision.models.vgg19_bn(pretrained=True)
        elif model_name=='resnet50':
            if 'ghost' in method:
                model = MODEL.resnet50(state_dict_dir=state_dict_dic+'resnet50-19c8e357.pth')
            else:
                model = torchvision.models.resnet50(pretrained=True)
        else:
            raise ValueError('Non-existent model name!')
        model.eval()

        # maxpool
        if 'max' in method:
            if model_name=='vgg19_bn':
                for i in [6, 13, 26, 39, 52]:
                    model.features[i]=MaxPool2dK2S2()
            elif model_name=='resnet50':
                model.maxpool=MaxPool2dK3S2P1()
                # model.maxpool.register_backward_hook(gaussian_smooth_backward_hook)
            else:
                raise ValueError('Non-existent model name!')
        # relu linear
        if 'relu_linear' in method:
            if model_name=='vgg19_bn':
                for i in [25, 29, 32, 35, 38, 42, 45, 48, 51][-2:]:
                    model.features[i]=ReLU_Linear()
            elif model_name=='resnet50':
                # TODO: self relu_linbp is difficult
                pass
            else:
                raise ValueError('Non-existent model name!')
        # relu silu
        if 'relu_silu' in method:
            if model_name=='vgg19_bn':
                for i in [25, 29, 32, 35, 38, 42, 45, 48, 51][-2:]:
                    model.features[i]=ReLU_SiLU()
            elif model_name=='resnet50':
                for i in range(1,6):
                    model.layer3[i].relu=ReLU_SiLU()
                for i in range(3):
                    model.layer4[i].relu=ReLU_SiLU()
            else:
                raise ValueError('Non-existent model name!')
        if 'sgm' in method:
            if model_name=='vgg19_bn':
                pass
            elif model_name=='resnet50':
                for i in range(1,3):
                    model.layer1[i].conv1.register_full_backward_hook(resnet_weight_backward_hook)
                for i in range(1,4):
                    model.layer2[i].conv1.register_full_backward_hook(resnet_weight_backward_hook)
                for i in range(1,6):
                    model.layer3[i].conv1.register_full_backward_hook(resnet_weight_backward_hook)
                for i in range(1,3):
                    model.layer4[i].conv1.register_full_backward_hook(resnet_weight_backward_hook)
            else:
                raise ValueError('Non-existent model name!')
        # batch norm 2d
        if 'batch_norm_2d' in method:
            model.features[50]=MyBatchNorm2d(model.features[50].weight.to(device), model.features[50].bias, model.features[50].running_mean.to(device), model.features[50].running_var.to(device), model.features[50].eps)

        if 'T' in method:
            kernel = gkern(5, 3).astype(np.float32)
            gaussian_kernel = np.stack([kernel, kernel, kernel])
            gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
            gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)

        model = nn.Sequential(
                Normalize,
                model
            )
        model.to(device)

        if target_attack:
            label_switch = torch.tensor(list(range(500,1000))+list(range(0,500))).long()
        label_ls = []

        for ind, (ori_img, label)in tqdm(enumerate(ori_loader)):
            pmax_list = []
            time_bp_list=[]
            label_ls.append(label)
            if target_attack:
                label = label_switch[label]

            ori_img = ori_img.to(device)
            img = ori_img.clone()
            m = 0
            variance = 0
            number = 20
            beta = 1.5

            for i in range(args.meta_steps):
                if 'mdi2fgsm' in method:
                    if i == 0:
                        img_x = img
                    else:
                        img_x = input_diversity(img)
                elif 'pgd' in method and i==0:
                    img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon)
                else:
                    img_x = img
                img_x.requires_grad_(True)

                img_x_temp=img_x

                if 'relu_linear' in method and model_name=='resnet50':
                    if not ('admix' in method or 'S' in method):
                        att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, img_x_temp, True, linbp_layer)
                        pred = torch.argmax(att_out, dim=1).view(-1)
                        loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                        model.zero_grad()
                        input_grad = linbp_backw_resnet50(img_x_temp, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=sgm_lambda)
                        input_grad=torch.autograd.grad(img_x_temp, img_x, input_grad)[0]
                    elif 'S' in method:
                        input_grad=0
                        for idx in range(5):
                            temp_img = img_x_temp / math.pow(2, idx)
                            att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, temp_img, True, linbp_layer)
                            pred = torch.argmax(att_out, dim=1).view(-1)
                            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                            model.zero_grad()
                            temp_grad = linbp_backw_resnet50(temp_img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=sgm_lambda)
                            input_grad += torch.autograd.grad(temp_img, img_x, temp_grad)[0]
                    elif 'DA' in method:
                        input_grad=0
                        for idx in range(30):
                            temp_img=img_x_temp + torch.normal(0, 0.05, size=img_x_temp.shape)
                            att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, temp_img, True, linbp_layer)
                            pred = torch.argmax(att_out, dim=1).view(-1)
                            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                            model.zero_grad()
                            temp_grad = linbp_backw_resnet50(temp_img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=sgm_lambda)
                            input_grad += torch.autograd.grad(temp_img, img_x, temp_grad)[0].sign()
                    elif 'admix' in method:
                        input_grad=0
                        for _ in range(3):
                            random_indices = list(range(img_x.shape[0]))
                            random.shuffle(random_indices)
                            for gamma in [1., 1./2, 1./4, 1./8, 1./16]:
                                temp_img = ((img_x + portion * img_x[random_indices]) * gamma).detach().requires_grad_()
                                att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, temp_img, True, linbp_layer)
                                pred = torch.argmax(att_out, dim=1).view(-1)
                                loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                                model.zero_grad()
                                input_grad += linbp_backw_resnet50(temp_img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=sgm_lambda)
                        input_grad /= 15
                else:
                    if not ('admix' in method or 'S' in method):
                        input_grad=0
                        img_x.requires_grad_(True)
                        loss_outer_list = []
                        loss_inner_list = []
                        ModelDelta = ModelTensorDelta(img_x).requires_grad_(True)
                        inner_opt = torch.optim.SGD(ModelDelta.parameters(),lr=args.attack_lr)
                        with higher.innerloop_ctx(ModelDelta, inner_opt, copy_initial_weights=True,
                                                  track_higher_grads=False if args.first_order else True) as (
                                delta, diffopt):
                            for j in range(args.inner_loop):
                                output_v3 = model(delta())
                                loss = -nn.CrossEntropyLoss()(output_v3, label.to(device))
                                if torch.isnan(loss):
                                    break
                                else:
                                    loss_inner_list.append(-loss.item())
                                    model.zero_grad()
                                    diffopt.step(loss)
                                    for task, task_model in enumerate(task_model_list):
                                        logits = task_model(delta())
                                        loss_val = nn.CrossEntropyLoss()(logits, label.to(device))
                                        if task == 0:
                                            loss_outer_list.append(loss_val.item())
                                        else:
                                            loss_outer_list[j] += loss_val.item()

                            for i in range(len(loss_outer_list)):
                                if np.isnan(loss_outer_list[i]):
                                    loss_inner_list = loss_inner_list[:i]
                                    loss_outer_list = loss_outer_list[:i]
                                    break
                            # print(loss_inner_list)
                            # print(loss_outer_list)
                            pmax = loss_outer_list.index(min(loss_outer_list)) if len(loss_inner_list) > 0 else -1
                            pmax_list.append(pmax)
                            for task, task_model in enumerate(task_model_list):
                                logits = task_model(
                                    delta(params=delta.parameters(
                                        time=pmax + 1))) if not args.first_order else task_model(
                                    delta(params=delta.parameters()))
                                loss_val = nn.CrossEntropyLoss()(logits, label.to(device))
                                if task == 0:
                                    loss_outer_final = loss_val
                                else:
                                    loss_outer_final += loss_val
                                task_model.zero_grad()
                            time_0 = time.time()
                            grad_z = torch.autograd.grad(loss_outer_final, delta.parameters(time=0), allow_unused=True)
                            time_bp = time.time()-time_0
                            time_bp_list.append(time_bp)
                        input_grad += grad_z[0].data
                    elif 'S' in method:
                        input_grad=0
                        for idx in range(5):
                            temp_img = img_x_temp / math.pow(2, idx)
                            att_out = model(temp_img)
                            pred = torch.argmax(att_out, dim=1).view(-1)
                            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                            model.zero_grad()
                            loss.backward()
                            input_grad += img_x.grad.data
                    elif 'DA' in method:
                        input_grad=0
                        for idx in range(30):
                            temp_img=img_x_temp + torch.normal(0, 0.05, size=img_x_temp.shape)
                            att_out = model(temp_img)
                            pred = torch.argmax(att_out, dim=1).view(-1)
                            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                            model.zero_grad()
                            loss.backward()
                            input_grad += img_x.grad.data.sign()
                    elif 'admix' in method:
                        input_grad=0
                        for _ in range(3):
                            random_indices = list(range(img_x.shape[0]))
                            random.shuffle(random_indices)
                            for gamma in [1., 1./2, 1./4, 1./8, 1./16]:
                                temp_img = ((img_x + portion * img_x[random_indices]) * gamma).detach().requires_grad_()
                                att_out = model(temp_img)
                                pred = torch.argmax(att_out, dim=1).view(-1)
                                loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                                model.zero_grad()
                                loss.backward()
                                input_grad += temp_img.grad.data
                        input_grad /= 15
                model.zero_grad()
                if 'mdi2fgsm' in method or ('mifgsm' in method and not 'vmifgsm' in method) or 'admix' in method:
                    input_grad = 1 * m + input_grad / torch.norm(input_grad, dim=(1, 2, 3), p=1, keepdim=True)
                    m = input_grad
                ########
                if 'vmifgsm' in method:
                    current_grad = input_grad + variance
                    current_grad = 1 * m + (current_grad) / torch.norm(current_grad, dim=(1, 2, 3), p=1,keepdim=True)
                    m = current_grad
                    variance = 0
                    for _ in range(number):
                        temp_img = (img + (torch.rand_like(img) * 2 * beta * epsilon - beta * epsilon).to(
                            device)).detach()
                        temp_img.requires_grad_()
                        if 'relu_linear' in method and model_name == 'resnet50':
                            att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(
                                model, temp_img, True, linbp_layer)
                            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                            model.zero_grad()
                            variance += linbp_backw_resnet50(temp_img, loss, conv_out_ls, ori_mask_ls, relu_out_ls,
                                                             conv_input_ls, xp=sgm_lambda)
                        else:
                            temp_out = model(temp_img)
                            temp_loss = nn.CrossEntropyLoss()(temp_out, label.to(device))
                            model.zero_grad()
                            temp_loss.backward()
                            variance += temp_img.grad.data
                    model.zero_grad()
                    variance = variance / number - input_grad
                    input_grad = m
                if target_attack:
                    input_grad = - input_grad
                if method == 'fgsm' or '_fgsm' in method:
                    img = img.data + 2 * epsilon * torch.sign(input_grad)
                else:
                    img = img.data + alpha * torch.sign(input_grad)
                img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
                img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
                img = torch.clamp(img, min=0, max=1)
            # writer.writerow(['batch'+str(ind)])
            # writer.writerow([i for i in itertools.chain(pmax_list, time_bp_list)])
            # # writer.writerow('batch'+str(ind))
            # writer_mean.writerow([np.mean(pmax_list),np.mean(time_bp_list)])
            if not ('admix' in method or 'S' in method):
                for j in range(niters):
                    if 'mdi2fgsm' in method:
                        if j == 0:
                            img_x = img
                        else:
                            img_x = input_diversity(img)
                    elif 'pgd' in method:
                        img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon)
                    else:
                        img_x = img
                    img_x.requires_grad_(True)

                    img_x_temp = img_x
                    att_out = model(img_x_temp)
                    loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                    print('outer_loss',loss)
                    model.zero_grad()
                    loss.backward()
                    input_grad = img_x.grad.data
                    if 'T' in method:
                        input_grad = F.conv2d(input_grad, gaussian_kernel, bias=None, stride=1,
                                              padding=((5 - 1) // 2, (5 - 1) // 2), groups=3)
                    if 'mdi2fgsm' in method or ('mifgsm' in method and not 'vmifgsm' in method) or 'admix' in method:
                        input_grad = 1 * m + input_grad / torch.norm(input_grad, dim=(1, 2, 3), p=1, keepdim=True)
                        m = input_grad
                    if 'vmifgsm' in method:
                        current_grad = input_grad + variance
                        current_grad = 1 * m + (current_grad) / torch.norm(current_grad, dim=(1, 2, 3), p=1,keepdim=True)
                        m = current_grad
                        variance = 0
                        for _ in range(number):
                            temp_img = (img + (torch.rand_like(img) * 2 * beta * epsilon - beta * epsilon).to(
                                device)).detach()
                            temp_img.requires_grad_()
                            if 'relu_linear' in method and model_name == 'resnet50':
                                att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(
                                    model, temp_img, True, linbp_layer)
                                loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                                model.zero_grad()
                                variance += linbp_backw_resnet50(temp_img, loss, conv_out_ls, ori_mask_ls, relu_out_ls,
                                                                 conv_input_ls, xp=sgm_lambda)
                            else:
                                temp_out = model(temp_img)
                                temp_loss = nn.CrossEntropyLoss()(temp_out, label.to(device))
                                model.zero_grad()
                                temp_loss.backward()
                                variance += temp_img.grad.data
                        model.zero_grad()
                        variance = variance / number - input_grad
                        input_grad = m
                    if target_attack:
                        input_grad = - input_grad
                    if method == 'fgsm' or '_fgsm' in method:
                        img = img.data + 2 * epsilon *torch.sign(input_grad)
                    else:
                        img = img.data + alpha*torch.sign(input_grad)
                    img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
                    img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
                    img = torch.clamp(img, min=0, max=1)
            #######################
            att_out = model(img)
            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
            model.zero_grad()
            print('final outer_loss', loss)
            att_out = task_model_list[0](img)
            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
            task_model_list[0].zero_grad()
            print('final inner_loss', loss)
            np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img.data*255).cpu().numpy().astype(np.uint8()))
            del img, ori_img, input_grad
        label_ls = torch.cat(label_ls)
        np.save(save_dir + '/labels.npy', label_ls.numpy())
        print('images saved')
        del model
    else:
        #--------------------------------------test---------------------------------
        inceptionv3 = MODEL.inceptionv3.Inception3()
        inceptionv3.to(device)
        inceptionv3.load_state_dict(torch.load(state_dict_dic+'inception_v3_google-1a9a5a14.pth'))
        inceptionv3.eval()
        print('inceptionv3:', test(model = inceptionv3, trans = trans_incep))
        del inceptionv3

        inception_resnet_v2 = timm.create_model('inception_resnet_v2', pretrained=True)
        inception_resnet_v2.eval()
        inception_resnet_v2.to(device)
        print('inception_resnet_v2:', test(model=inception_resnet_v2, trans = trans_incep))
        del inception_resnet_v2

        densenet = torchvision.models.densenet121(pretrained=False)
        densenet.to(device)
        import re
        pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(state_dict_dic+'densenet121-a639ec97.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        densenet.load_state_dict(state_dict)
        densenet.eval()
        print('densenet:', test(model = densenet, trans = trans_se))
        del densenet

        mobilenet = torchvision.models.mobilenet_v2(pretrained=False)
        mobilenet.to(device)
        mobilenet.load_state_dict(torch.load(state_dict_dic+'mobilenet_v2-b0353104.pth'))
        mobilenet.eval()
        print('mobilenet:', test(model = mobilenet, trans = trans_se))
        del mobilenet

        pnasnet = MODEL.pnasnet.pnasnet5large(ckpt_dir =state_dict_dic+'pnasnet5large-bf079911.pth', num_classes=1000, pretrained='imagenet')
        pnasnet.to(device)
        pnasnet.eval()
        print('pnasnet:', test(model = pnasnet, trans = trans_pnas))
        del pnasnet

        senet = MODEL.senet.senet154(ckpt_dir =state_dict_dic+'senet154-c7b49a05.pth')
        senet.to(device)
        senet.eval()
        print('senet:', test(model = senet, trans = trans_se))
        del senet

        ens3_adv_inc_v3 = MODEL.ens3_adv_inc_v3.KitModel(state_dict_dic+'tf2torch_ens3_adv_inc_v3.npy', aux_logits=False)
        ens3_adv_inc_v3.eval()
        ens3_adv_inc_v3.to(device)
        print('ens3_adv_inc_v3:', test(model=ens3_adv_inc_v3, trans = trans_incep, if_minus = True))
        del ens3_adv_inc_v3

        ens4_adv_inc_v3 = MODEL.ens4_adv_inc_v3.KitModel(state_dict_dic+'tf2torch_ens4_adv_inc_v3.npy', aux_logits=False)
        ens4_adv_inc_v3.eval()
        ens4_adv_inc_v3.to(device)
        print('ens4_adv_inc_v3:', test(model=ens4_adv_inc_v3, trans = trans_incep, if_minus = True))
        del ens4_adv_inc_v3

        ens_adv_inception_resnet_v2 = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
        ens_adv_inception_resnet_v2.eval()
        ens_adv_inception_resnet_v2.to(device)
        print('ens_adv_inception_resnet_v2:', test(model=ens_adv_inception_resnet_v2, trans = trans_incep))
        del ens_adv_inception_resnet_v2

        if model_name=='vgg19_bn':
            source_model = torchvision.models.vgg19_bn(pretrained=True)
        elif model_name=='resnet50':
            source_model = torchvision.models.resnet50(pretrained=True)
        else:
            raise ValueError('Non-existent model name!')
        source_model.eval()
        source_model.to(device)
        print('source_model:', test(model = source_model, trans = trans_ori))
        del source_model