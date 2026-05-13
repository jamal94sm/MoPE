
mope_convnext.py: Adding LoRA-MoE to the last stage (stage 3) of ConvNeXt foundation model and fine-tune the model on CASIS-MS-ROI using ArcFace, SupCon (supervised Contrastive), and GRL (cross-entropy) loss functions. We want to use FFT-Swapping technique for augmentation along with general augmentations.
For identification accuracy, cross-domain (Reg: training domains, Qry: target domains) similarity measurement is considered instead of ArcFace recognition.

mope_convnext_mkmmd.py: L2-norm consitency loss is added. GRL is replaced by MK-MMD loss.

mope_cnn.py: replace ConvNeXt model with a small custom CNN model.
