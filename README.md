# GAN_project

### Bachelor Thesis project dedicated to optimal hyperparameter search for two GAN variations - DCGAN and WGAN

Both networks are adapted for training on CIFAR-10 dataset and built using PyTorch framework. However, it is possible to modify code in order to use project for the dataset of choice. Requirements file will be uploaded soon. This project also features automatic FID evaluation after each epoch. All logs, results and generated images for particular choice of network and hyperparameters are stored in separate folders. After training, it is possible to aggregate results, visualizee training process, as well as summarize performance of GANs by using *eval.ipynb* notebook.

<details>
<summary>### Full parseable arguments list for hypsearch.py</summary>
<br>
hyp_search:<br/>
  --LR_Ds [LR_DS [LR_DS ...]]<br/>
                        LR_D list 1, usage example "--LR_Ds 0.0001 0.0002
                        0.0004"
                        
  --LR_Gs [LR_GS [LR_GS ...]]<br/>
                        LR_G list 2, usage example "--LR_Gs 0.0001 0.0002
                        0.0004"
                        
  --SEEDS [SEEDS [SEEDS ...]]<br/>
                        SEED list, usage example "--SEEDS 4242 4343 4444"
                       

gan_hyps:<br/>
  --gan_type {GAN,WGAN}<br/>
                        The type of GAN
                        
  --BATCH_SIZE<br/>
                        The size of batch
                        
  --IMAGE_SIZE<br/>
                        The size of input image
                        
  --INPUT_NOISE<br/>
                        Input noise for generator
                        
  --GENERATOR_FILTERS<br/>
                        The size of convolution filters of G
                        
  --DISCRIMINATOR_FILTERS<br/>
                        The size of convolution filters of D
                        
  --KERNEL_SIZE<br/>
                        The size of kernel for convolution layers
                        
  --NUMBER_CHANNELS<br/>
                        The number of input channels
                        
  --N_EPOCHS<br/>   The number of epochs to run
  
  --B1<br/>               Beta 1
  
  --B2<br/>               Beta 2
  
  --VECTOR_LEN<br/>
                        The number of images to pass for FID calculation (both for real and fake samples)
                        
  --C<br/>                 Clipping value
  
  --save_dir<br/>   Directory name to save training checkpoints to
  
  --result_dir<br/>
                        Directory name to save generated images to
                        
  --log_dir<br/>     Directory name to save training logs to
  
  --dataset<br/>     Dataset name

</details>

