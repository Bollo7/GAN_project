# GAN_project

### Bachelor Thesis project dedicated to fine-tuning of learning rates for two GAN variations - DCGAN and WGAN

Both networks are adapted to CIFAR-10 dataset and built in PyTorch. Requirements file will be uploaded soon. Even though search space is limited, it is possible to extend it in future work. Additionally, both networks are evaluated with FID after each epoch.


### Full parseable arguments list for hypsearch.py

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
  
  --save_dir<br/>   Directory name to save pictures to
  
  --result_dir<br/>
                        Directory name to save generated images to
                        
  --log_dir<br/>     Directory name to save training logs to
  
  --dataset<br/>     Dataset name
  
  #### Separate jupyter notebook gan_eval.ipynb for aggregation of logs and visualization of training process is also provided
