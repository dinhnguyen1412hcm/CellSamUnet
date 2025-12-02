# CellSamUnet
A DUAL-STREAM FEATURE EXTRACTION NETWORK BASED SEGMENTATION METHOD FOR CELL IMAGES USING SAM AND U-NET
# Abstract 
Cell image segmentation is a significant task in biomedical studies, especially spatial transcriptomics analysis. Different deep learning based methods have been applied for segmentation of medical images. Among the methods, U-Net architecture and Segment Anything model show many advantages in the analysis process. However, noises and diversity in shape of objects in cell images still pose research challenges. This study proposes a novel segmentation method for cell images, called CellSamUnet, based on a dual-stream feature extraction network that utilizes an effective fusion of SAM2 image and U-Net encoders. Experimental results on well-known benchmarking datasets demonstrates that the proposed method achieves better performance than other baseline methods.
# Training on Datasets:
!gdown --id 1lVVfq_NRv-8ts8uGwzJRgqj6_RMpCnyB -O monuseg_train.zip

# Validating the trained models 
!gdown --id 141UpDydk6tRAS8OJecealaRp9dTOqP2v -O monuseg_val.zip
