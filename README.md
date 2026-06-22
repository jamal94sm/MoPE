
mope_convnext.py: Adding LoRA-MoE to the last stage (stage 3) of ConvNeXt foundation model and fine-tune the model on CASIS-MS-ROI using ArcFace, SupCon (supervised Contrastive), and GRL (cross-entropy) loss functions. We want to use FFT-Swapping technique for augmentation along with general augmentations.
For identification accuracy, cross-domain (Reg: training domains, Qry: target domains) similarity measurement is considered instead of ArcFace recognition.

mope_convnext_mkmmd.py: L2-norm consitency loss is added. GRL is replaced by MK-MMD loss.

mope_cnn.py: replace ConvNeXt model with a small custom CNN model.



best setting for continual mode on CASIA-MS: 
python main.py --dataset casia_ms --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
    --tta_method contrastive_nn --nn_lambda 1.0 --nn_temp 0.1

Best setting for episodic mode on CASIA-MS: 
python main.py --dataset casia_ms --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
    --tta_method contrastive

the improvement comes primarily from BN running stats adaptation (safe_bn mode) + Contrastive loss(NT-Xent) + Nearest Neighbers Cosistency loss term (for continual). 


