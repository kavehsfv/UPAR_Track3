"""Offline checkpoint/model/loss/gradient smoke test; not a full retraining run."""
import argparse
import importlib.util
from pathlib import Path
import torch
from transformers import VitPoseConfig, VitPoseForPoseEstimation

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('train_pose', HERE/'train_pose.py')
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path, default=HERE.parent/'weights/pose_model.pt')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = VitPoseConfig.from_pretrained(HERE/'model_config', local_files_only=True)
    model = tp.ProbPoseViTPose(VitPoseForPoseEstimation(config)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True), strict=True)
    x = torch.randn(2,3,256,192,device=device)
    target = torch.rand(2,17,64,48,device=device)
    mask = torch.ones(2,17,device=device)
    loss_fn = tp.OKSHeatmapLoss()
    for frozen in [True,False]:
        model.freeze_backbone = frozen
        for p in model.vitpose.parameters():
            p.requires_grad = not frozen
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            heat,prob = model(x)
            oks = loss_fn(heat.float(),target,mask)
        pres = tp.bce_presence(prob.float(),mask,mask)
        loss = oks + .1*pres
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        backbone_grads = sum(p.grad is not None for p in model.vitpose.parameters())
        assert (backbone_grads == 0) == frozen
        print(f'freeze={frozen} loss={float(loss.detach()):.6f} finite_gradients={len(grads)} backbone_gradients={backbone_grads}')
    print('PASS: strict checkpoint load, both training stages and losses/backward')


if __name__=='__main__':
    main()
