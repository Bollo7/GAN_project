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
                        
  --BATCH_SIZE BATCH_SIZE<br/>
                        The size of batch
                        
  --IMAGE_SIZE IMAGE_SIZE<br/>
                        The size of input image
                        
  --INPUT_NOISE INPUT_NOISE<br/>
                        Input noise for generator
                        
  --GENERATOR_FILTERS GENERATOR_FILTERS<br/>
                        The size of convolution filters of G
                        
  --DISCRIMINATOR_FILTERS DISCRIMINATOR_FILTERS<br/>
                        The size of convolution filters of D
                        
  --KERNEL_SIZE KERNEL_SIZE<br/>
                        The size of kernel for convolution layers
                        
  --NUMBER_CHANNELS NUMBER_CHANNELS<br/>
                        The number of input channels
                        
  --N_EPOCHS N_EPOCHS   The number of epochs to run
  
  --B1 B1               Beta 1
  
  --B2 B2               Beta 2
  
  --VECTOR_LEN VECTOR_LEN<br/>
                        The number of epochs to run
                        
  --C C                 Clipping value
  
  --save_dir SAVE_DIR   Directory name to save pictures to
  
  --result_dir RESULT_DIR
                        Directory name to save generated images to
                        
  --log_dir LOG_DIR     Directory name to save training logs to
  
  --dataset DATASET     Dataset name
