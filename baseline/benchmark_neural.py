"""Development on 25 images, freeze settings on five, then final 30-image refit."""
import gc
from pathlib import Path
import time

import numpy as np

from baseline.benchmark import connected, evaluate_test, select_configuration
from baseline.common import distance_instances, normalize, read_pair
from voron_matsam.baseline import seed_all
from voron_matsam.baseline_data import load_manifest, sha256, write_json


def split_data(manifest_path):
    manifest, records, root = load_manifest(manifest_path)
    val = manifest['validation']
    train = [sid for sid in manifest['train_order'] if sid not in val]
    if len(train)!=25 or len(val)!=5 or set(train+val)&set(manifest['test']):
        raise ValueError('Need disjoint 25 development / 5 validation / 10 test images')
    pairs = {sid:read_pair(records[sid],root) for sid in train+val}
    return manifest, pairs, train, val


def unet_reconstruct(probability, config, method):
    from scipy import ndimage as ndi
    foreground = probability >= config['threshold']
    if method=='unet':
        if config.get('erosion',0):
            foreground=ndi.binary_erosion(foreground,iterations=config['erosion'],border_value=1)
        return connected(foreground,200)
    return distance_instances(foreground,config['distance'],200)


def fit_unet(data, output, args):
    import torch
    from baseline.models.unet import UNet, random_batch, segmentation_loss
    seed_all(args.seed)
    device=torch.device('cpu' if args.cpu else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('U-Net benchmark requires a GPU allocation')
    output.mkdir()
    model=UNet(base=32,levels=4).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(args.seed)
    cached=[(normalize(image),mask) for image,mask in data]
    history=[]
    started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses=[]
        for _ in range(args.steps):
            x,y=random_batch(cached,2,256,rng,normalized=True)
            optimizer.zero_grad(set_to_none=True)
            loss=segmentation_loss(model(x.to(device)),y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        print(f'U-Net epoch {epoch+1}/{args.epochs}: loss={np.mean(losses):.5f}',flush=True)
    checkpoint=output/'unet.pt'
    torch.save(model.state_dict(),checkpoint)
    write_json(output/'history.json',history)
    # Predictions always use the saved final checkpoint, not an in-memory best.
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    model.eval()
    info=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        epochs=args.epochs,steps_per_epoch=args.steps,optimizer_updates=args.epochs*args.steps,
        batch_size=2,learning_rate=1e-4,weight_decay=1e-4,widths=[32,64,128,256],
        loss='BCE + Dice, equal weights',precision='float32',seed=args.seed,
        training_seconds=time.perf_counter()-started,checkpoint_selection='final_epoch',
        paper_configuration_status='documented approximation; hidden original widths/settings unavailable')
    return model,device,info


def run_unet(args):
    from baseline.models.unet import tiled_probability
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    manifest,pairs,train,val=split_data(args.manifest)
    model,device,development=fit_unet([pairs[sid] for sid in train],out/'development',args)
    probabilities={sid:tiled_probability(model,pairs[sid][0],device) for sid in val}
    settings={}
    for method in ('unet','unet_watershed'):
        configs=([dict(threshold=t,erosion=e) for t in (.4,.5,.6) for e in (0,1)] if method=='unet'
                 else [dict(threshold=t,distance=d) for t in (.4,.5,.6) for d in (8,12,16)])
        selected,history=select_configuration(configs,val,
            lambda sid,c:unet_reconstruct(probabilities[sid],c,method),lambda sid:pairs[sid][1])
        settings[method]=selected
        write_json(out/f'{method}_validation.json',history)
    write_json(out/'frozen_postprocessing.json',settings)
    del model,probabilities
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model,device,final=fit_unet([pairs[sid] for sid in manifest['train_order']],out/'final',args)
    cache=out/'probabilities'
    cache.mkdir()
    info=dict(development_train_ids=train,selection_validation_ids=val,
        final_train_ids=manifest['train_order'],final_refit_validation='none; decisions already frozen',
        development=development,final=final,postprocessing=settings,
        shared_checkpoint_for_both_methods=final['checkpoint_sha256'])
    write_json(out/'training.json',info)
    for method in ('unet','unet_watershed'):
        def predict(image,sid):
            path=cache/f'{sid}.npz'
            if not path.exists():
                probability=tiled_probability(model,image,device)
                np.savez_compressed(path,probability=probability)
            with np.load(path) as raw:
                return unet_reconstruct(raw['probability'],settings[method],method)
        evaluate_test(predict,args.manifest,out/method,method,info)


def run_stardist(args):
    import tensorflow as tf
    from baseline.models.stardist import StarDistAdapter
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    manifest,pairs,train,val=split_data(args.manifest)
    config=dict(seed=args.seed,n_rays=32,patch_size=256,epochs=args.epochs,
        steps_per_epoch=args.steps,batch_size=4,learning_rate=.0003,
        probability_threshold=.5,nms_threshold=.4)
    development_dir=out/'development'
    development_dir.mkdir()
    development=StarDistAdapter(config,development_dir,args.cpu)
    development_info=development.fit([pairs[sid] for sid in train],[pairs[sid] for sid in val])
    candidates=[dict(prob=.5,nms=.4),dict(prob=.4,nms=.4),dict(prob=.6,nms=.4),
                dict(prob=.5,nms=.3),dict(prob=.5,nms=.5)]
    def val_predict(sid,c):
        return development.model.predict_instances(normalize(pairs[sid][0]),
            prob_thresh=c['prob'],nms_thresh=c['nms'],n_tiles=(2,2,1))[0]
    selected,history=select_configuration(candidates,val,val_predict,lambda sid:pairs[sid][1])
    write_json(out/'validation_selection.json',history)
    del development
    tf.keras.backend.clear_session()
    gc.collect()
    seed_all(args.seed)
    final_dir=out/'final'
    final_dir.mkdir()
    final=StarDistAdapter(config,final_dir,args.cpu)
    # The upstream API requires validation_data. At final refit, use a training
    # image solely for logged monitoring. LR is constant, no early stopping,
    # and weights_last is loaded. This is not an independent validation score.
    final_info=final.fit([pairs[sid] for sid in manifest['train_order']],[pairs[train[0]]])
    final.config.update(probability_threshold=selected['prob'],nms_threshold=selected['nms'])
    info=dict(development_train_ids=train,selection_validation_ids=val,
        final_train_ids=manifest['train_order'],development=development_info,final=final_info,
        final_monitoring_ids=[train[0]],final_monitoring_status='training-only logging; not held-out validation',
        selection_frozen_before_final_refit=True,selected_config=selected,
        tensorflow_devices=[str(gpu) for gpu in tf.config.list_physical_devices('GPU')],
        config=config,seed=args.seed,paper_target='none; extended baseline')
    write_json(out/'training.json',info)
    evaluate_test(final.predict,args.manifest,out/'test','stardist',info)
