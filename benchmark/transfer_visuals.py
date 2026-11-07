"""First-convolution activations and steering Grad-CAM on prespecified validation frames."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT/"docs/experiments/semester_2026_fall/transfer/donkey-001")
    parser.add_argument("--tree", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    tree, study = args.tree.resolve(), args.study.resolve()
    sys.path.insert(0, str(tree))
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import MyFlows as ms
    from MyFlows.core.device import asnumpy
    from MyFlows.ops.convolution import Conv2D_Op
    from MyFlows.utils.gradcam import compute_gradcam, overlay_heatmap
    from apps.common.image_preprocess import imread_nchw
    from apps.common.gpu_lock import GpuLock
    from apps.train import train_myflows_donkey as train

    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    selected = read(study/"formal/locked.json")
    rows = read(study/"formal/summary.json")
    scratch = next(r for r in rows if r["method"]=="scratch-imagenet" and r["seed"]==selected["seed"])
    choices = [("Scratch",ROOT/scratch["run"]), ("Pretrained",ROOT/selected["run"])]
    rgb, overlays, conv_maps, raw = [], {}, {}, {}
    metadata = {"schema_version": 1, "target": "steering output index 0", "frames": [], "models": {}}
    with GpuLock(label="transfer Grad-CAM",run_dir=study/"visualization"):
        for label, directory in choices:
            config = read(directory/"config.json")
            options = train.build_parser().parse_args([])
            for k, v in config["args"].items():
                if hasattr(options,k):
                    setattr(options,k,v)
            train.resolve_myflows_device(options.device)
            ms.set_default_dtype(np.float32)
            h,w=config["input_h"],config["input_w"]
            x,_,model,logits,_,graph=train._build_model(options,1,h,w,np.float32,options.conv_backend,load_pretrained=False)
            ms.load_checkpoint([model],None,str(directory/"best"))
            for param in model.params:
                param.trainable=False
            model.eval()
            first=next(n for n in graph.nodes if isinstance(n,Conv2D_Op) and n.parents[1] is model.stem_conv.kernel)
            index,splits,_=train.load_rows(options,Path(options.data))
            ids=splits["val"]
            chosen=[ids[0],ids[len(ids)//2],ids[-1]]
            metadata["models"][label]={"run":directory.relative_to(ROOT).as_posix(),"seed":options.seed,
                                       "checkpoint":"best","input_size":[h,w]}
            overlays[label]=[]
            for i, idx in enumerate(chosen):
                rel,_,_=index[idx]
                path=Path(options.data)/rel
                image=cv2.resize(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB),(w,h))
                x.value=imread_nchw(path,(w,h),np.float32)
                graph.forward()
                if i==0:
                    conv_maps[label]=asnumpy(first.value)[0,:8].copy()
                    raw[label+"_conv"]=conv_maps[label]
                cam,score=compute_gradcam(logits,model._last_feature_nodes["layer4"],target_index=0)
                if not np.isfinite(cam).all() or not np.isfinite(score):
                    raise RuntimeError("nonfinite Grad-CAM output")
                raw[f"{label}_cam_{i}"]=cam
                overlays[label].append((overlay_heatmap(image,cam),score))
                if label=="Scratch":
                    rgb.append(image)
                    metadata["frames"].append({"split":"val","row":idx,"image":str(rel)})
    fig,axes=plt.subplots(2,8,figsize=(15,4),constrained_layout=True)
    for ch in range(8):
        bound=max(np.percentile(np.abs(np.concatenate([v[ch].ravel() for v in conv_maps.values()])),99.5),1e-8)
        for j,(label,_) in enumerate(choices):
            axes[j,ch].imshow(conv_maps[label][ch],cmap="coolwarm",vmin=-bound,vmax=bound)
            axes[j,ch].set(title=f"{label} C{ch}",xticks=[],yticks=[])
    fig.savefig(study/"first-convolution.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(9,7.5),constrained_layout=True)
    for i,image in enumerate(rgb):
        axes[i,0].imshow(image);axes[i,0].set(title=f"Validation row {chosen[i]}",xticks=[],yticks=[])
        for j,(label,_) in enumerate(choices,1):
            overlay,score=overlays[label][i];axes[i,j].imshow(overlay)
            axes[i,j].set(title=f"{label}: angle {score:.3f}",xticks=[],yticks=[])
    fig.savefig(study/"gradcam.png",dpi=150);plt.close(fig)
    np.savez(study/"feature-maps.npz",**raw)
    metadata["arrays_sha256"]=hashlib.sha256((study/"feature-maps.npz").read_bytes()).hexdigest()
    (study/"visualization.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    report=study/"report.md"
    section="## 首层响应与转向 Grad-CAM"
    prefix=report.read_text(encoding="utf-8").split(section)[0].rstrip()
    report.write_text(prefix+"\n\n"+section+"\n\n两列模型采用相同训练 seed、torchvision stem 和相同验证帧。"
                      "首个验证帧的前 8 个通道按通道号排列，每对面板共享色标。Grad-CAM 取 layer4 的特征与转向输出梯度，"
                      "三帧依次取验证清单的首、中、末位置。可通过 visualization.json 核对帧号与 checkpoint，原始数值在 feature-maps.npz。"
                      "\n\n![第一层卷积响应](first-convolution.png)\n\n![三个验证帧的转向 Grad-CAM](gradcam.png)\n",encoding="utf-8")


if __name__=="__main__":
    main()
