
This is the source for our paper: ForecastTKGQuestions: A Benchmark for Temporal Question Answering and Forecasting over Temporal Knowledge Graphs.

### Overview:

Our constructed TKG ICEWS21 and proposed dataset ForecastTKGQuestions is under the directory *Data/*.

Source code of our proposed QA model ForecastTKGQA is under the directory *Code/ForecastTKGQA/*.

Source code of TANGO pre-training is under the directory *Code/TANGO/*.

Source code of question generation, extracted 2-hop rules and question templates are under the directory *Question_Generation/*.

### Data:
##### ICEWS21 
We generate ICEWS21 from the events collected from the Integrated Crisis Early Warning System (ICEWS) dataverse. One example of the collected raw events file is presented under the directory *Data/raw_ICEWS/*.

We assign ids for the entities, relations and timestamps appearing in ICEWS21. 
The id mappings are presented in the directory *Data/ICEWS21_processed/mapping/*.

*entity2id.txt, relation2id.txt, ts2id.txt* are used for training TANGO on ICEWS21. *wd_id2entity_text.txt, wd_id2relation_text.txt* are used for linking entities and relations between the TKG dataset (ICEWS21) and QA dataset (ForecastTKGQuestions). 

The mapped ICEWS21 dataset is presented in the directory *Data/ICEWS21_processed/*. 
Each line in these files corresponds to a TKG fact. For each line in these files, the first four numbers correspond to a TKG fact's subject entity, relation, object entity and timestamp, respectively (represented with ids). 
The last number -1 is unnecessary and not used throughout our experiments. 
ForecastTKGQuestions 
All three types of questions are stored in *Data/ForecastTKGQuestions*. 

### TANGO pre-training on ICEWS21:
We provide a pre-trained TANGO checkpoint *TANGO_2022_06_17_21_35_50* in the directory *Code/TANGO/checkpoints/*. 

If you want to train your own TANGO model, do as follows: 

Following the official repository of TANGO, create a python 3.8 environment and install pytorch 1.4.0, torch-scatter 2.0.5, torchdiffeq 0.1.0. 

Replace the corresponding files in the installed torchdiffeq with the files from *Code/TANGO/torchdiffeq/* (code borrowed from the official repository of TANGO).

Copy *Data/ICEWS_processed/train.txt, Data/ICEWS_processed/valid.txt, Data/ICEWS_processed/test.txt* to the directory *Code/TANGO/ICEWS21/*. 

Go to the directory *Code/TANGO/ICEWS21/* and pre-process ICEWS21 by running: 

`
python ICEWS21_predicate_preprocess.py 
`

Then you can pre-train TANGO on ICEWS21. 

Please refer to the official repository of TANGO for more details. 

#### One Time Inference 
As mentioned in section 4.1 of the submission paper, we perform a one time inference to get TANGO representations at every timestamp. 
This step is necessary if you want to use TANGO as the TKG model for ForecastTKGQA.

This step is conducted by using line 372-393 of *Code/TANGO.py* after we have a trained TANGO checkpoint. 

To perform one time inference, run: 

`
python TANGO.py --resume --name TANGO_2022_06_17_21_35_50 --test --score_func complex 
`

Then go to the directory *Code/TANGO/checkpoints/submission/*. 
Copy *tango_submission* to the directory *Code/ForecastTKGQA/models/ICEWS21/kg_embeddings/* for QA. 

*tango_submission* contains TANGO representations at every timestamp (around 4GB). 

### ForecastTKGQA:
#### Installation 
Go to *Code/ForecastTKGQA/* and create a conda environment: 

`
conda create --prefix ./forecasttkgqa_env python=3.8 
conda activate ./forecasttkgqa_env 
`

Install ForecastTKGQA requirements: 

`
conda install --file requirements.txt -c conda-forge 
`

Install pytorch (specify your own CUDA version): 

`
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch 
`

#### To Run ForecastTKGQA 
Copy the folder *Data/ForecastTKGQuestions/* to the directory *Code/ForecastTKGQA/data/ICEWS21/*

Entity prediction questions: 

`
python trainer.py --tkg_model_file tango_submission.pkl --model forecasttkgqa --lm_model distilbert --batch_size 512 --max_epochs 200 --valid_freq 1 --save_to forecasttkgqa --question_type entity_prediction --device 0 
`

Yes-unknown questions: 

`
python trainer.py --tkg_model_file tango_submission.pkl --model forecasttkgqa --lm_model distilbert --batch_size 256 --max_epochs 200 --valid_freq 1 --save_to forecasttkgqa --question_type yes_unknown --device 0 
`

Fact reasoning questions: 

`
python trainer.py --tkg_model_file tango_submission.pkl --model forecasttkgqa --lm_model distilbert --batch_size 256 --max_epochs 200 --valid_freq 1 --save_to forecasttkgqa --question_type fact_reasoning --device 0
`

