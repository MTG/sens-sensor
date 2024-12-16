## PLEASANTNESS AND EVENTFULNESS PREDICTION

This section of the repository includes the code to train models that predict the perceptual qualities of Pleasantness and Eventfulness, see a more detailed research in the documents proposed in <a href="#references">References section</a>.

The following bullet points constitute the keys for the development of these models.

- **Dataset** 
  
  We use the augmented soundscapes proposed in <a href="https://github.com/ntudsp/araus-dataset-baseline-models"> ARAUS dataset </a>. 

- **Features** 
  
  <a href="https://github.com/LAION-AI/CLAP">LAION-AI's CLAP model</a> is used to generate the sound representations for the augmented soundscapes. In particular, we use LAION-AI's pre-trained model *630k-fusion-best.pt*.

- **Test set** 
  
  ARAUS dataset provides more than 25k augmented soundscapes organised in 5 folds for cross-validation plus an additional fold ('fold-0') for testing. However, we include an additional fold for testing ('fold-Fs') formed of 25 real audios obtained from the Freesound library. 


## Reproducibility of model creation

In order to reproduce the creation of the models for predicting Pleasantness and Eventfulness, the following steps need to be followed, indicated with the neccessary instructions and code:

1) Download ARAUS dataset and source code from <a href="https://github.com/ntudsp/araus-dataset-baseline-models"> ARAUS Github page </a>. Once installation processed is completed and the necessary files have been downloaded as indicated, run the script below to generate the 25k augmented soundscapes.

   ``` 
   araus-dataset-baselines-models/code/make_augmented_soundscapes.py
   ```
   **From ARAUS dataset we are only making use of these 25k augmented soundscapes WAV files and the file ```responses.csv```. The rest of the code or files is not neccessary from now on.**

2) After some research (see <a href="#references">References section</a>), CLAP embeddings demonstrated strong performance in terms of prediction accuracy and suitability for real-time processing. The steps below determine the guide to obtain the dataset to train the models, which includes the CLAP embeddings for ARAUS augmented audios as well as for Fold-Fs audios too.
   
   *NOTE: Set up your environment following the instructions in <a href="#environment-configuration">Environment configuration section</a>.*

   1) Adapt ARAUS original dataset for extension 
        
        This <a href="development/dataset_Adequate_ARAUS_for_extension.ipynb">script</a> offers a guide of the adaptation needed prior to the CLAP embeddings generation. It recieves as input ```responses.csv``` and outputs ```responses_adapted.csv```.

   2) Generate dataset
        
        This <a href="development/dataset_Generate_features.py">script</a> generates JSON files with the embeddings for the set of audios indicated. 

        It must be run in the command line with
        ```
        python development/dataset_Generate_features.py --data_path path/to/data/folder --type araus
        # for generating CLAP dataset for ARAUS dataset augmented soundscapes, requires responses_adapted.csv in data/files/

        python development/dataset_Generate_features.py --data_path path/to/data/folder --type new
        # for generating CLAP dataset for fold-Fs soundscape audios, requires responses_fold_Fs.csv in data/files/

        python development/dataset_Generate_features.py --data_path path/to/data/folder --type both
        # for both
        ```

## Environment configuration

## References
- Amaia Sagasti, Martín Rocamora, Frederic Font: *Prediction of Pleasantness and Eventfulness Perceptual Sound Qualities in Urban Soundscapes* - DCASE Workshop 2024 <a href="https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Sagasti_12.pdf">Paper link DCASE webpage</a>
- Amaia Sagasti Martínez - MASTER THESIS: *Prediction of Pleasantness and Eventfulness Perceptual Sound Qualities in Urban Soundscapes* - Sound and Music Computing Master (Music Technology Group, Universitat Pompeu Fabra - Barcelona) <a href="https://zenodo.org/records/13861445">Master Thesis Report link Zenodo</a>
