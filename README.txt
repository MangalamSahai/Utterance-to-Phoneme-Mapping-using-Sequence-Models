A]Code running instructions:

1) Install Packages for decoding & calculating Levenstein Distance.
2) Import all required Libraries and functions
3) Setup your Kaggle- Install , download and unzip.
4) Run LibriSamples & LibriSamplesTest  Class.
5) Run the val & train loader.
6) Check if your val_loader is working properly.
7) Run Network Class- My Model.
8) Training Configuration to be executed.
9) Run calculate_levenshtein function.
10) Check if your train_loader function is working fine.
11) Run model_eval() function.
12) Import Drive 
13) Run your code using epoch.  
14) You can use save checkpoint to load model that you had saved.
15) Run model_eval() to test the data.
16) Submit file.



B]Experiments you ran?

1) BidirLSTM Model

LSTM input_size= 256 & Batch_size=128 
CUDA Error.

2) Used Batch_first and got in calculating Levenshtein distance and Loss. Used Batch_first=False and only permuted in decoder.decode which made me run the code.

3) Change Beam_width to a large number increases computation time however does not vary the result much.

4) Without Kaiming Initialization, loss values were quite high.

5) Try to not use dropout in middle layers.



C]Architectures you tried?

1) Bidirectional LSTM

self.embedding:
nn.Conv1d(13,64,kernel_size=3,stride=1,bias=False,padding=1),
          nn.BatchNorm1d(64),
          nn.LeakyReLU(),
          nn.MaxPool1d(kernel_size=2,stride=2),
          nn.Dropout(p=0.45),
          nn.Conv1d(64,128,stride=1,kernel_size=1),
          nn.BatchNorm1d(128),
          nn.LeakyReLU(),
          nn.MaxPool1d(kernel_size=2,stride=2),
          nn.Dropout(0.45),
          nn.Conv1d(256,512,stride=1,kernel_size=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.MaxPool1d(kernel_size=2,stride=2),
          nn.Dropout(0.45),
          )

self.lstm = nn.LSTM(input_size=512,hidden_size=512,num_layers=4,bidirectional=True,dropout=0.4).

However Got cuda error, because the model has very high number of parameters.

No.of Epochs you trained for: Not able to train

Hyper-parameters you used: Batch_Size=128, lr= 2e-3 , Scheduler- Cosine Annealing, optimizer-Adam.

2) Cov-Next Model _ Residual:

class StageLayers(nn.Module):
  
  def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
               padding,
               flag,
                ):
        super().__init__() # Just have to do this for all nn.Module classes

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding= padding
        self.flag= flag        
        self.Layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=self.stride,padding=self.padding,groups=in_channels),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels,out_channels=4*out_channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm1d(num_features=4*out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=4*out_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm1d(num_features=out_channels),
            #nn.Conv2d(in_channels=in_channel,out_channels=in_channel*4,kernel_size=1,stride=1,padding=0),
         )
        
  def forward(self, x):
        
        size = (x.shape[2]+2*self.padding-7)//self.stride +1
        y= nn.functional.interpolate(x,size)
        out = self.Layer(x)

        if self.flag==1:
          
           return out
        
        else: 
          
           return out+y
     
class CovNext(nn.Module):
     def __init__(self, num_classes= 256):
         super().__init__() # Already features indented from previous class. 
         #self.drop_block = DropBlock2D(block_size=3, drop_prob=0.3)  
         self.num_classes = num_classes
         
         self.stem = nn.Sequential(
            nn.Conv1d(in_channels=13,out_channels=96,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm1d(num_features=96),
            nn.ReLU(),
            )
         
         self.stage1_maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1), 
            )
        
         self.stage_cfgs = [
              # in_channel, #blocks
            [96, 96, 3],
            [96, 192, 3], 
            [192, 384, 9], 
            [384, 768, 3], 
            ]
                           
         layers = []
         #pdb.set_trace()
         i=0
         for curr_stage in self.stage_cfgs:
             in_channels, out_channels, num_blocks = curr_stage
             for block_idx in range(num_blocks):
                 print(i)
                 print(block_idx)
                 if block_idx==0: 
                       stride=1 if in_channels==13 else 2
                       padding=0 if in_channels==13 else 3
                      
                 else:
                    stride= 1
                    padding= 3
                         
     

                 layers.append(StageLayers(
                 in_channels = in_channels,
                 out_channels = out_channels,
                 padding = padding,  
                 stride= stride,
                 flag= 1 if block_idx==0 else 0
                 ))
                 in_channels = out_channels
             
             i=i+1  
                 
                            
         self.layers = nn.Sequential(*layers) # Done, save them to the class

       
         self.cls_layer = nn.Sequential(
             nn.Dropout(p=0.1),
             
             nn.AdaptiveAvgPool1d(256),
             nn.Flatten(),
             nn.Linear(768,num_classes),
           )

         self._initialize_weights()

     def _initialize_weights(self):
        """
        Usually, I like to use default pytorch initialization for stuff, but
        MobileNetV2 made a point of putting in some custom ones, so let's just
        use them.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                   m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

     def forward(self, x):
        out = self.stem(x)
        out = self.stage1_maxpool(out)
        out = self.layers(out)
        out = self.cls_layer(out)
        
        

        return out

- Model took a lot of epochs to train but saturates at 26 Levenstein distance.
- Epochs trained: 100
- Hyper-parameters you used: Batch_Size=128, lr= 2e-3 , Scheduler- ReducedLRPlateau, optimizer-Adam.
- Ensured Convolution used gives desired shape for executing residual layer.


D]Specify Which Combination resulted in best Score.

The following Model resulted in best Combination.

self.embedding = nn.Sequential(nn.Conv1d(in_channels=input_size,out_channels=hidden_size,kernel_size=3,stride=2,padding=1,bias=False),
                               nn.BatchNorm1d(hidden_size),
                               nn.GELU(),
                               Block(hidden_size),
                               nn.Dropout(0.4),
                              )

self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=4,bidirectional=True,dropout=0.4)

self.classification = nn.Sequential(nn.Linear(hidden_size*2,hidden_size*4),
                                     nn.GELU(),
                                     nn.Linear(hidden_size*4,41)
                                     )

Without using Batch_First yielded me a good result.
No Cuda Error as batches are 64 and less number of convolution layers are being used.



