test

# Generate chemnlp descriptions for entries 0-1000
python generator.py --data_path "path/to/merged_dataset.json" --start 0 --end 1000 --text chemnlp --output_dir output

# Generate all formats with spacegroup analysis
python generator.py --data_path "path/to/merged_dataset.json" --text all --include_spg --output_dir output

# Skip specific topics
python generator.py --data_path "path/to/merged_dataset.json" --text chemnlp --skip_sentence bond



Output Example:

Field	Value
MP ID	mp-1077201
Formula	UAsP
Ordering	FM
Transition Type	AFM
Source	combined_database
Generated Text Description:

The chemical information includes: The chemical has an atomic formula of UAsP with a prototype of ABC; Its molecular weight is 343.92 g/mol; The atomic fractions are U: 0.333, As: 0.333, P: 0.333, and the atomic values X and Z are 1.38, 2.18, 2.19 and 92, 33, 15, respectively. The structure information includes: The lattice parameters are 3.89, 3.89, 8.1 with angles 90.0, 90.0, 90.0 degrees; The top K XRD peaks are found at 21.9, 25.4, 31.9, 32.6, 33.2 degrees; The material has a density of 9.33 g/cm³; The bond distances are as follows: U-U: 3.89, P-U: 2.8, 2.81, As-U: 3.06, As-As: 2.75, 3.89, As-P: 3.51, P-P: 3.55, 3.89.


python train_llm.py --input_csv output/chemnlp_0_210579_skip_none.csv --model_name gpt2 --epochs 10 --batch_size 8 --learning_rate 2e-5 --use_class_weights --save_model --output_dir llm_output


python train_llm_gnn_prepared.py --data_dir llm_gnn_data

python test_llm_gnn.py --data_dir llm_gnn_data --model_dir llm_gnn_output