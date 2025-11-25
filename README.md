# Towards Pre-trained Graph Condensation via Optimal Transport

## Environment Settings

> python=3.11.5
> torch_geometric>=2.5.0
> torch>=2.2.0
> scikit-learn>=1.3.0
> POT>=0.9.5 

## Usage

You can use the following command to process PreGC:

```bash
python emb_init.py
```

Here, we set the dataset as <u>PubMed</u> by default, with <u>0.08%</u> condensation ratio, <u>SGC</u> architecture.

## Other Setting

**1.** You can change other datasets and ratio to obtain different condensed graphs.
For example:

```bash
python emb_init.py --dataset cora --reduction_rate 0.25
```

**2.** Although PreGC does not rely on task labels during the condensation process, the calculation of the condensation ratio still adheres to the configuration established in existing GC methods. For instance, in the case of the <u>PubMed</u> dataset, the ratio is calculated as $0.25×60/19717 ≈ 0.08$%.



> [!NOTE]
>
> 1. In graph diffusion augmentation, we adjust the diffusion interval by varying the terminal time $T$ and the diffusion order $K$.  This procedure is integrated into the `graph_opt.py`.
> 2. If the condensation ratio or dataset is to be altered, careful tuning of the hyperparameters within the `config_init.json` file is paramount, as it is critical for the performance of PreGC.
> 3. The final condensed graph can be found in the `output` directory.


![Pre-trained Graph Condensation](./pregc.png)
