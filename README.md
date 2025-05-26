# mark1
▪ first version of noise detection model  
▪ 3-class(apartment, daily, background)  
▪ An inductive approach was taken in referencing "Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation" by Xiuye Gu et al. (Google Research & Nvidia), to develop our audio classification framework based on visual-language distillation principles.(ViLD)  
▪ The dataset is evenly composed of 100 samples each for apartment noise, daily noise, and background noise, resulting in a 1:1:1 ratio.    


▪ The original model before refactoring   
▪ We decided to revise the classification approach of the original mark1 model by transitioning from a three-class method to a two-class method. Due to the high similarity among the original data samples - an issue that is inevitable given the nature of the data - the model struggled to achieve reliable performance when attempting to classify all three types simultaneously.  
▪ As a result, we introduce the mark2 series - a set of models that each focus on binary classification tasks derived from the original problem. Each variant is specialized as follows:  
  - mark2.1: Impact sound vs. Others
  - mark2.2: Water-related sounds(shower/toilet) vs. Others
  - mark2.3: Construction drill sounds vs. Others
  - mark2.4: Everyday noise(daily conversations, ambient noise) vs. Others    

▪ We are currently in the process of updating the dataset and modifying the classification strategy accordingly. Additionally, the text embedding component from the mark1 will be removed as it is no longer required in the new design.    
