# HYPOTHESIS: Experimental Results & Model Evaluation Report

This report is automatically generated from the saved knowledge graph (`.graphml`), evaluation metrics (`results_*.json`), and prediction rankings (`predictions_*.csv`).

## 1. Knowledge Graph Summary (`kg_v14_expanded` vs `kg_v15_mechanisms`)

| Graph Version | Total Nodes | Total Edges | Candidate Drugs | Genes (`GEN`) | Pathways (`PWY`) | Diseases (`DIS`) | Drugs with Target Gene (`DRG-GEN`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `kg_v14_expanded.graphml` | 11,785 | 74,836 | 9,528 | 417 | 160 | 850 | 31 (0.3%) |
| `kg_v15_mechanisms.graphml` | 13,941 | 73,259 | 6,187 | 2,573 | 160 | 850 | 3,399 (54.9%) |

## 2. Held-Out Validation Accuracy (In-Distribution Disease Link Prediction)

| Run / Model Configuration | Held-Out Positives ($n$) | Candidate Pool ($N$) | MRR | Hits@10 | Hits@50 | Hits@100 | Median Rank | Percentile Rank (Accuracy Proxy) | Gain vs. Popularity MRR | Gain vs. Random H@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`kg_v15_mechanisms_neg0p5_cos_seed0` (GNN)** | 782 | 6,187 | **0.1480** | **27.62%** | **52.81%** | **62.66%** | **42.0** | **99.34%** | **3.81x** | **170.9x** |
| `kg_v15_mechanisms_neg0p5_cos_seed0` (Popularity Baseline) | 782 | 6,187 | 0.0388 | 6.65% | 16.24% | 20.59% | 636.2 | 89.73% | 1.00x | 41.1x |
| **`kg_v15_mechanisms_seed0` (GNN)** | 2387 | 9,528 | **0.0702** | **12.78%** | **25.43%** | **33.64%** | **330.0** | **96.55%** | **2.02x** | **121.7x** |
| `kg_v15_mechanisms_seed0` (Popularity Baseline) | 2387 | 9,528 | 0.0347 | 6.74% | 17.26% | 22.87% | 664.5 | 93.04% | 1.00x | 64.3x |
| **`kg_v14_expanded_seed0` (GNN)** | 2387 | 9,528 | **0.0700** | **12.02%** | **25.22%** | **31.71%** | **414.0** | **95.66%** | **2.01x** | **114.6x** |
| `kg_v14_expanded_seed0` (Popularity Baseline) | 2387 | 9,528 | 0.0347 | 6.74% | 17.26% | 22.87% | 664.5 | 93.04% | 1.00x | 64.3x |

## 3. Zero-Shot Disease Evaluation (`results_kg_v15_mechanisms_neg0p5_cos_seed0.json`)

All drug edges for the 12 diseases in the Cutaneous Leishmaniasis, Tuberculosis, and Dengue families were stripped prior to training.

| Disease & Evidence Tier | Known Drugs ($n$) | Model | MRR | Hits@10 | Hits@50 | Hits@100 | Median Rank | Percentile Rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **Cutaneous Leishmaniasis | strict (INDICATION/OFF_LABEL)** | 2 | **GNN** | **0.0197** | **0.0%** | **50.0%** | **50.0%** | **608.0** | **90.2%** |
| Cutaneous Leishmaniasis | strict (INDICATION/OFF_LABEL) | 2 | Popularity | 0.0031 | 0.0% | 0.0% | 0.0% | 1077.0 | 82.6% |
| **Cutaneous Leishmaniasis | trial-stage** | 19 | **GNN** | **0.0081** | **0.0%** | **15.8%** | **15.8%** | **389.0** | **93.7%** |
| Cutaneous Leishmaniasis | trial-stage | 19 | Popularity | 0.0095 | 0.0% | 21.1% | 31.6% | 637.0 | 89.7% |
| **Cutaneous Leishmaniasis | literature candidates** | 5 | **GNN** | **0.0043** | **0.0%** | **0.0%** | **0.0%** | **224.0** | **96.4%** |
| Cutaneous Leishmaniasis | literature candidates | 5 | Popularity | 0.1059 | 20.0% | 40.0% | 40.0% | 269.5 | 95.7% |
| **Tuberculosis | strict (INDICATION/OFF_LABEL)** | 58 | **GNN** | **0.0133** | **3.4%** | **10.3%** | **19.0%** | **346.5** | **94.4%** |
| Tuberculosis | strict (INDICATION/OFF_LABEL) | 58 | Popularity | 0.0610 | 6.9% | 22.4% | 27.6% | 741.0 | 88.0% |
| **Tuberculosis | trial-stage** | 61 | **GNN** | **0.0251** | **6.6%** | **19.7%** | **27.9%** | **270.0** | **95.7%** |
| Tuberculosis | trial-stage | 61 | Popularity | 0.0543 | 9.8% | 21.3% | 31.1% | 368.5 | 94.1% |
| **Dengue | trial-stage** | 13 | **GNN** | **0.0027** | **0.0%** | **7.7%** | **7.7%** | **1133.0** | **81.7%** |
| Dengue | trial-stage | 13 | Popularity | 0.0038 | 0.0% | 7.7% | 7.7% | 1400.5 | 77.4% |
| **Dengue | literature candidates** | 6 | **GNN** | **0.0017** | **0.0%** | **0.0%** | **0.0%** | **718.0** | **88.4%** |
| Dengue | literature candidates | 6 | Popularity | 0.2239 | 33.3% | 33.3% | 33.3% | 297.2 | 95.2% |

## 4. Degree-Bias Audit (Correlation Between Training Degree and Prediction Score)

| Target Disease | Pearson $r$ | Spearman $\rho$ | Interpretation |
|---|---:|---:|---|
| **Cutaneous Leishmaniasis** | `+0.3951` | `+0.4976` | Moderate degree sensitivity |
| **Tuberculosis** | `+0.4528` | `+0.6291` | Moderate degree sensitivity |
| **Dengue** | `+0.3025` | `+0.3859` | Moderate degree sensitivity |

## 5. Composition of Top Predicted Candidates (Rediscovered vs. Novel Repurposed %)

| Disease | Cutoff ($K$) | Rediscovered Approved (`strict`) | Rediscovered Clinical Trial (`trial`) | Rediscovered Literature (`lit`) | **Total Already Known (%)** | **Novel Repurposing Candidates (%)** | **Mechanistically Explainable (%)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Cutaneous Leishmaniasis** | Top-10 | 0 (0%) | 0 (0%) | 0 (0%) | **0/10 (0.0%)** | **10/10 (100.0%)** | **8/10 (80.0%)** |
| **Cutaneous Leishmaniasis** | Top-25 | 1 (4%) | 0 (0%) | 0 (0%) | **1/25 (4.0%)** | **24/25 (96.0%)** | **21/25 (84.0%)** |
| **Cutaneous Leishmaniasis** | Top-50 | 1 (2%) | 2 (4%) | 0 (0%) | **3/50 (6.0%)** | **47/50 (94.0%)** | **42/50 (84.0%)** |
| **Cutaneous Leishmaniasis** | Top-100 | 1 (1%) | 4 (4%) | 0 (0%) | **5/100 (5.0%)** | **95/100 (95.0%)** | **83/100 (83.0%)** |
| **Tuberculosis** | Top-10 | 1 (10%) | 4 (40%) | 0 (0%) | **5/10 (50.0%)** | **5/10 (50.0%)** | **10/10 (100.0%)** |
| **Tuberculosis** | Top-25 | 1 (4%) | 8 (32%) | 0 (0%) | **9/25 (36.0%)** | **16/25 (64.0%)** | **24/25 (96.0%)** |
| **Tuberculosis** | Top-50 | 5 (10%) | 11 (22%) | 0 (0%) | **16/50 (32.0%)** | **34/50 (68.0%)** | **48/50 (96.0%)** |
| **Tuberculosis** | Top-100 | 8 (8%) | 15 (15%) | 0 (0%) | **23/100 (23.0%)** | **77/100 (77.0%)** | **95/100 (95.0%)** |
| **Dengue** | Top-10 | 0 (0%) | 0 (0%) | 0 (0%) | **0/10 (0.0%)** | **10/10 (100.0%)** | **0/10 (0.0%)** |
| **Dengue** | Top-25 | 0 (0%) | 0 (0%) | 0 (0%) | **0/25 (0.0%)** | **25/25 (100.0%)** | **0/25 (0.0%)** |
| **Dengue** | Top-50 | 0 (0%) | 1 (2%) | 0 (0%) | **1/50 (2.0%)** | **49/50 (98.0%)** | **0/50 (0.0%)** |
| **Dengue** | Top-100 | 0 (0%) | 1 (1%) | 0 (0%) | **1/100 (1.0%)** | **99/100 (99.0%)** | **0/100 (0.0%)** |

## 6. Top-15 Predicted Small-Molecule Candidates per Target Disease

### 6.1 Cutaneous Leishmaniasis

| Rank | ChEMBL ID | Drug Name | Score | Train Deg | Known Status | Mechanistic Evidence Path |
|---:|---|---|---:|---:|---|---|
| 1 | `CHEMBL1229` | **OSELTAMIVIR** | 6.7480 | 4 | `Novel Candidate` | n_targets=1 | disease_bridge=obsolete_pneumonia(4g), obsolete_autoimmune thro(1g) |
| 2 | `CHEMBL1454` | **LEVAMISOLE** | 6.0301 | 3 | `Novel Candidate` | n_targets=2 |
| 3 | `CHEMBL32` | **MOXIFLOXACIN** | 5.6204 | 26 | `Novel Candidate` | n_targets=1 | disease_bridge=obsolete_pneumonia(4g), graft versus host diseas(3g) |
| 4 | `CHEMBL4297503` | **BALOXAVIR MARBOXIL** | 5.6191 | 2 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_pneumonia(4g) |
| 5 | `CHEMBL777` | **CLAVULANIC ACID** | 5.5373 | 20 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_pneumonia(4g), obsolete_head and neck m(1g) |
| 6 | `CHEMBL1082` | **AMOXICILLIN** | 5.3346 | 34 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_pneumonia(4g), obsolete_osteoporosis(3g) |
| 7 | `CHEMBL501122` | **CEFTAROLINE FOSAMIL** | 5.1688 | 5 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_pneumonia(4g), obsolete_chronic kidney (1g) |
| 8 | `CHEMBL4650326` | **TEREVALEFIM** | 5.1388 | 1 | `Novel Candidate` | n_targets=1 | pathways(2)=MET->RAF/MAP kinase cascade<-CSF2RA; MET->RAF/MAP kinase cascade<-FYN |
| 9 | `CHEMBL38` | **TRETINOIN** | 5.1008 | 7 | `Novel Candidate` | n_targets=15 | disease_bridge=myeloproliferative disor(6g), obsolete_osteoporosis(3g) |
| 10 | `CHEMBL4297529` | **RISUTEGANIB** | 5.0434 | 1 | `Novel Candidate` | n_targets=0 |
| 11 | `CHEMBL501259` | **ERITORAN** | 5.0174 | 3 | `Novel Candidate` | n_targets=2 | pathways(4)=LY96->RSV-host interactions<-TLR7; LY96->Regulation of TLR by endogen<-TLR7 | disease_bridge=obsolete_pneumonia(4g) |
| 12 | `CHEMBL1435` | **CEFAZOLIN** | 4.9840 | 13 | `Novel Candidate` | n_targets=3 | pathways(3)=IL2->Interleukin receptor SHC sig<-CSF2RA; IL2->RAF/MAP kinase cascade<-CSF2RA | disease_bridge=prostate cancer(1g), obsolete_blood coagulati(1g) |
| 13 | `CHEMBL847` | **OXAMNIQUINE** | 4.9346 | 2 | `Novel Candidate` | n_targets=0 |
| 14 | `CHEMBL126` | **LINEZOLID** | 4.9295 | 12 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_pneumonia(4g), obsolete_hypertension(2g) |
| 15 | `CHEMBL131` | **PREDNISOLONE** | 4.9279 | 77 | `Novel Candidate` | n_targets=1 | pathways(41)=NR3C1->Potential therapeutics for S<-IGHV1-2; NR3C1->Potential therapeutics for S<-IGHV1-46 | disease_bridge=obsolete_pneumonia(4g), acute lymphoblastic leuk(4g) |

### 6.2 Tuberculosis

| Rank | ChEMBL ID | Drug Name | Score | Train Deg | Known Status | Mechanistic Evidence Path |
|---:|---|---|---:|---:|---|---|
| 1 | `CHEMBL36` | **PYRIMETHAMINE** | 7.4702 | 11 | `Novel Candidate` | n_targets=2 | pathways(4)=HEXB->Neutrophil degranulation<-LTF; HEXB->Neutrophil degranulation<-MAPK1 | disease_bridge=obsolete_malaria(4g), obsolete_epilepsy(2g) |
| 2 | `CHEMBL1201139` | **MEGESTROL ACETATE** | 6.5564 | 6 | `Novel Candidate` | n_targets=2 | shared_genes=NR3C1 | pathways(12)=NR3C1->HSP90 chaperone cycle for st<-NR3C1; NR3C1->Nuclear Receptor transcripti<-NR3C1 | disease_bridge=obsolete_chronic obstruc(5g), obsolete_head and neck m(1g) |
| 3 | `CHEMBL710` | **FINASTERIDE** | 6.4457 | 8 | `Novel Candidate` | n_targets=3 | disease_bridge=obsolete_acne(4g), obsolete_benign prostati(2g) |
| 4 | `CHEMBL223228` | **EFAVIRENZ** | 6.3655 | 6 | `trial-stage` | n_targets=0 | disease_bridge=obsolete_epilepsy(2g), obsolete_HIV infection(2g) |
| 5 | `CHEMBL1042` | **CHOLECALCIFEROL** | 6.2496 | 40 | `trial-stage` | n_targets=1 | shared_genes=VDR | pathways(4)=VDR->Nuclear Receptor transcripti<-NR3C1; VDR->Nuclear Receptor transcripti<-VDR | disease_bridge=obsolete_urinary tract i(5g), obsolete_chronic obstruc(5g) |
| 6 | `CHEMBL137` | **METRONIDAZOLE** | 5.8874 | 17 | `trial-stage` | n_targets=0 | disease_bridge=obsolete_pneumonia(5g), obsolete_rosacea(4g) |
| 7 | `CHEMBL606` | **MISOPROSTOL** | 5.7607 | 14 | `Novel Candidate` | n_targets=4 | disease_bridge=pregnancy(1g), osteoarthritis(1g) |
| 8 | `CHEMBL600` | **ACETYLCYSTEINE** | 5.7250 | 32 | `strict` | n_targets=10 | pathways(14)=CHUK->FCERI mediated NF-kB activat<-RPS27A; CHUK->IKK complex recruitment medi<-RPS27A | disease_bridge=obsolete_chronic obstruc(5g), obsolete_amyloidosis(5g) |
| 9 | `CHEMBL1509` | **DROSPIRENONE** | 5.6919 | 5 | `Novel Candidate` | n_targets=4 | shared_genes=NR3C1 | pathways(24)=AR->HSP90 chaperone cycle for st<-NR3C1; AR->Nuclear Receptor transcripti<-NR3C1 | disease_bridge=obsolete_acne(4g), obsolete_epilepsy(2g) |
| 10 | `CHEMBL127` | **MEROPENEM** | 5.6103 | 13 | `trial-stage` | n_targets=0 | disease_bridge=obsolete_urinary tract i(5g), obsolete_pneumonia(5g) |
| 11 | `CHEMBL702` | **PIPERACILLIN** | 5.5840 | 16 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_urinary tract i(5g), obsolete_pneumonia(5g) |
| 12 | `CHEMBL1539` | **SULFADOXINE** | 5.5648 | 5 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_malaria(4g), obsolete_epilepsy(2g) |
| 13 | `CHEMBL103` | **PROGESTERONE** | 5.5631 | 16 | `Novel Candidate` | n_targets=9 | shared_genes=NR3C1 | pathways(36)=AR->HSP90 chaperone cycle for st<-NR3C1; AR->Nuclear Receptor transcripti<-NR3C1 | disease_bridge=obsolete_nicotine depend(2g), obsolete_epilepsy(2g) |
| 14 | `CHEMBL270190` | **ALVIMOPAN** | 5.5631 | 5 | `Novel Candidate` | n_targets=3 | pathways(8)=OPRD1->Interleukin-4 and Interleuki<-CCL2; OPRD1->Interleukin-4 and Interleuki<-MMP1 |
| 15 | `CHEMBL404` | **TAZOBACTAM** | 5.5363 | 17 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_urinary tract i(5g), obsolete_pneumonia(5g) |

### 6.3 Dengue

| Rank | ChEMBL ID | Drug Name | Score | Train Deg | Known Status | Mechanistic Evidence Path |
|---:|---|---|---:|---:|---|---|
| 1 | `CHEMBL880` | **FAMCICLOVIR** | 7.2115 | 7 | `Novel Candidate` | n_targets=0 |
| 2 | `CHEMBL1401` | **NITAZOXANIDE** | 7.0058 | 10 | `Novel Candidate` | n_targets=0 |
| 3 | `CHEMBL372795` | **STREPTOMYCIN** | 6.9413 | 10 | `Novel Candidate` | n_targets=1 |
| 4 | `CHEMBL184` | **ACYCLOVIR** | 6.7899 | 11 | `Novel Candidate` | n_targets=0 |
| 5 | `CHEMBL104` | **CLOTRIMAZOLE** | 6.7138 | 6 | `Novel Candidate` | n_targets=3 |
| 6 | `CHEMBL455` | **SULFACETAMIDE** | 6.7048 | 7 | `Novel Candidate` | n_targets=0 |
| 7 | `CHEMBL1282` | **IMIQUIMOD** | 6.6173 | 12 | `Novel Candidate` | n_targets=2 |
| 8 | `CHEMBL2104404` | **EMODEPSIDE** | 6.5171 | 2 | `Novel Candidate` | n_targets=0 |
| 9 | `CHEMBL1440` | **TETRACYCLINE** | 6.4927 | 21 | `Novel Candidate` | n_targets=1 |
| 10 | `CHEMBL1200680` | **SELENIUM SULFIDE** | 6.4823 | 10 | `Novel Candidate` | n_targets=0 |
| 11 | `CHEMBL137` | **METRONIDAZOLE** | 6.4262 | 17 | `Novel Candidate` | n_targets=0 |
| 12 | `CHEMBL1331216` | **HYOSCYAMINE** | 6.3423 | 3 | `Novel Candidate` | n_targets=4 |
| 13 | `CHEMBL32` | **MOXIFLOXACIN** | 6.2747 | 26 | `Novel Candidate` | n_targets=1 |
| 14 | `CHEMBL242948` | **CARNOSINE** | 6.1970 | 2 | `Novel Candidate` | n_targets=0 |
| 15 | `CHEMBL91` | **MICONAZOLE** | 6.1178 | 5 | `Novel Candidate` | n_targets=24 |

