# HYPOTHESIS: Experimental Results & Model Evaluation Report

This report is automatically generated from the saved knowledge graph (`.graphml`), evaluation metrics (`results_*.json`), and prediction rankings (`predictions_*.csv`).

## 1. Knowledge Graph Summary (`kg_v14_expanded` vs `kg_v15_mechanisms`)

| Graph Version | Total Nodes | Total Edges | Candidate Drugs | Genes (`GEN`) | Pathways (`PWY`) | Diseases (`DIS`) | Drugs with Target Gene (`DRG-GEN`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `kg_v14_expanded.graphml` | 11,785 | 74,836 | 9,528 | 417 | 160 | 850 | 31 (0.3%) |
| `kg_v15_mechanisms.graphml` | 13,952 | 73,508 | 6,186 | 2,584 | 160 | 850 | 3,401 (55.0%) |

## 2. Held-Out Validation Accuracy (In-Distribution Disease Link Prediction)

| Run / Model Configuration | Held-Out Positives ($n$) | Candidate Pool ($N$) | MRR | Hits@10 | Hits@50 | Hits@100 | Median Rank | Percentile Rank (Accuracy Proxy) | Gain vs. Popularity MRR | Gain vs. Random H@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`kg_v15_mechanisms_neg0p5_cos_seed0` (GNN)** | 781 | 6,186 | **0.1512** | **25.99%** | **48.66%** | **57.62%** | **53.0** | **99.16%** | **3.60x** | **160.8x** |
| `kg_v15_mechanisms_neg0p5_cos_seed0` (Popularity Baseline) | 781 | 6,186 | 0.0420 | 7.17% | 14.98% | 20.10% | 625.0 | 89.91% | 1.00x | 44.4x |
| **`kg_v15_mechanisms_seed0` (GNN)** | 2387 | 9,528 | **0.0702** | **12.78%** | **25.43%** | **33.64%** | **330.0** | **96.55%** | **2.02x** | **121.7x** |
| `kg_v15_mechanisms_seed0` (Popularity Baseline) | 2387 | 9,528 | 0.0347 | 6.74% | 17.26% | 22.87% | 664.5 | 93.04% | 1.00x | 64.3x |
| **`kg_v14_expanded_seed0` (GNN)** | 2387 | 9,528 | **0.0700** | **12.02%** | **25.22%** | **31.71%** | **414.0** | **95.66%** | **2.01x** | **114.6x** |
| `kg_v14_expanded_seed0` (Popularity Baseline) | 2387 | 9,528 | 0.0347 | 6.74% | 17.26% | 22.87% | 664.5 | 93.04% | 1.00x | 64.3x |

## 3. Zero-Shot Disease Evaluation (`results_kg_v15_mechanisms_neg0p5_cos_seed0.json`)

All drug edges for the 12 diseases in the Cutaneous Leishmaniasis, Tuberculosis, and Dengue families were stripped prior to training.

| Disease & Evidence Tier | Known Drugs ($n$) | Model | MRR | Hits@10 | Hits@50 | Hits@100 | Median Rank | Percentile Rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **Cutaneous Leishmaniasis | strict (INDICATION/OFF_LABEL)** | 2 | **GNN** | **0.0069** | **0.0%** | **0.0%** | **50.0%** | **293.5** | **95.3%** |
| Cutaneous Leishmaniasis | strict (INDICATION/OFF_LABEL) | 2 | Popularity | 0.0031 | 0.0% | 0.0% | 0.0% | 1074.2 | 82.6% |
| **Cutaneous Leishmaniasis | trial-stage** | 19 | **GNN** | **0.0027** | **0.0%** | **0.0%** | **5.3%** | **528.0** | **91.5%** |
| Cutaneous Leishmaniasis | trial-stage | 19 | Popularity | 0.0094 | 0.0% | 21.1% | 31.6% | 634.5 | 89.8% |
| **Cutaneous Leishmaniasis | literature candidates** | 5 | **GNN** | **0.0042** | **0.0%** | **0.0%** | **0.0%** | **276.0** | **95.6%** |
| Cutaneous Leishmaniasis | literature candidates | 5 | Popularity | 0.1059 | 20.0% | 40.0% | 40.0% | 272.0 | 95.6% |
| **Tuberculosis | strict (INDICATION/OFF_LABEL)** | 58 | **GNN** | **0.0160** | **6.9%** | **12.1%** | **19.0%** | **491.0** | **92.1%** |
| Tuberculosis | strict (INDICATION/OFF_LABEL) | 58 | Popularity | 0.0607 | 6.9% | 22.4% | 27.6% | 745.0 | 88.0% |
| **Tuberculosis | trial-stage** | 61 | **GNN** | **0.0246** | **4.9%** | **21.3%** | **27.9%** | **275.0** | **95.6%** |
| Tuberculosis | trial-stage | 61 | Popularity | 0.0537 | 9.8% | 21.3% | 31.1% | 364.5 | 94.1% |
| **Dengue | trial-stage** | 13 | **GNN** | **0.0029** | **0.0%** | **7.7%** | **7.7%** | **1358.0** | **78.1%** |
| Dengue | trial-stage | 13 | Popularity | 0.0037 | 0.0% | 7.7% | 7.7% | 1398.0 | 77.4% |
| **Dengue | literature candidates** | 6 | **GNN** | **0.0038** | **0.0%** | **0.0%** | **0.0%** | **320.0** | **94.8%** |
| Dengue | literature candidates | 6 | Popularity | 0.2239 | 33.3% | 33.3% | 33.3% | 295.0 | 95.2% |

## 4. Degree-Bias Audit (Correlation Between Training Degree and Prediction Score)

| Target Disease | Pearson $r$ | Spearman $\rho$ | Interpretation |
|---|---:|---:|---|
| **Cutaneous Leishmaniasis** | `+0.3717` | `+0.4890` | Moderate degree sensitivity |
| **Tuberculosis** | `+0.4806` | `+0.6878` | Moderate degree sensitivity |
| **Dengue** | `+0.3625` | `+0.4901` | Moderate degree sensitivity |

## 5. Composition of Top Predicted Candidates (Rediscovered vs. Novel Repurposed %)

| Disease | Cutoff ($K$) | Rediscovered Approved (`strict`) | Rediscovered Clinical Trial (`trial`) | Rediscovered Literature (`lit`) | **Total Already Known (%)** | **Novel Repurposing Candidates (%)** | **Mechanistically Explainable (%)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Cutaneous Leishmaniasis** | Top-10 | 0 (0%) | 0 (0%) | 0 (0%) | **0/10 (0.0%)** | **10/10 (100.0%)** | **9/10 (90.0%)** |
| **Cutaneous Leishmaniasis** | Top-25 | 0 (0%) | 0 (0%) | 0 (0%) | **0/25 (0.0%)** | **25/25 (100.0%)** | **24/25 (96.0%)** |
| **Cutaneous Leishmaniasis** | Top-50 | 0 (0%) | 0 (0%) | 0 (0%) | **0/50 (0.0%)** | **50/50 (100.0%)** | **49/50 (98.0%)** |
| **Cutaneous Leishmaniasis** | Top-100 | 1 (1%) | 1 (1%) | 0 (0%) | **2/100 (2.0%)** | **98/100 (98.0%)** | **95/100 (95.0%)** |
| **Tuberculosis** | Top-10 | 1 (10%) | 2 (20%) | 0 (0%) | **3/10 (30.0%)** | **7/10 (70.0%)** | **10/10 (100.0%)** |
| **Tuberculosis** | Top-25 | 4 (16%) | 5 (20%) | 0 (0%) | **9/25 (36.0%)** | **16/25 (64.0%)** | **25/25 (100.0%)** |
| **Tuberculosis** | Top-50 | 6 (12%) | 12 (24%) | 0 (0%) | **18/50 (36.0%)** | **32/50 (64.0%)** | **50/50 (100.0%)** |
| **Tuberculosis** | Top-100 | 9 (9%) | 16 (16%) | 0 (0%) | **25/100 (25.0%)** | **75/100 (75.0%)** | **99/100 (99.0%)** |
| **Dengue** | Top-10 | 0 (0%) | 0 (0%) | 0 (0%) | **0/10 (0.0%)** | **10/10 (100.0%)** | **10/10 (100.0%)** |
| **Dengue** | Top-25 | 0 (0%) | 0 (0%) | 0 (0%) | **0/25 (0.0%)** | **25/25 (100.0%)** | **25/25 (100.0%)** |
| **Dengue** | Top-50 | 0 (0%) | 1 (2%) | 0 (0%) | **1/50 (2.0%)** | **49/50 (98.0%)** | **50/50 (100.0%)** |
| **Dengue** | Top-100 | 0 (0%) | 1 (1%) | 0 (0%) | **1/100 (1.0%)** | **99/100 (99.0%)** | **96/100 (96.0%)** |

## 6. Top-15 Predicted Small-Molecule Candidates per Target Disease

### 6.1 Cutaneous Leishmaniasis

| Rank | ChEMBL ID | Drug Name | Score | Train Deg | Known Status | Mechanistic Evidence Path |
|---:|---|---|---:|---:|---|---|
| 1 | `CHEMBL1454` | **LEVAMISOLE** | 7.2705 | 2 | `Novel Candidate` | n_targets=2 | disease_bridge=nephrotic syndrome(4g) |
| 2 | `CHEMBL403989` | **TG100-801** | 6.2636 | 1 | `Novel Candidate` | n_targets=18 | shared_genes=FGR,FYN,HCK,LYN | pathways(1473)=BLK->Antigen activates B Cell Rec<-FYN; BLK->Antigen activates B Cell Rec<-IGHV1-2 |
| 3 | `CHEMBL173929` | **LUTEIN** | 5.9650 | 5 | `Novel Candidate` | n_targets=0 | disease_bridge=retinitis pigmentosa(3g), pregnancy(2g) |
| 4 | `CHEMBL1200666` | **CALCIPOTRIENE** | 5.9365 | 5 | `Novel Candidate` | n_targets=1 | shared_genes=VDR | pathways(2)=VDR->Nuclear Receptor transcripti<-VDR; VDR->SUMOylation of intracellular<-VDR | disease_bridge=obsolete_psoriasis(11g), obsolete_atopic eczema(10g) |
| 5 | `CHEMBL1229` | **OSELTAMIVIR** | 5.7910 | 4 | `Novel Candidate` | n_targets=1 | disease_bridge=obsolete_pneumonia(12g), obsolete_influenza(1g) |
| 6 | `CHEMBL38` | **TRETINOIN** | 5.6116 | 7 | `Novel Candidate` | n_targets=15 | pathways(8)=RARA->Nuclear Receptor transcripti<-VDR; RARA->SUMOylation of intracellular<-VDR | disease_bridge=myeloproliferative disor(11g), obsolete_atopic eczema(10g) |
| 7 | `CHEMBL847` | **OXAMNIQUINE** | 5.4641 | 2 | `Novel Candidate` | n_targets=0 |
| 8 | `CHEMBL846` | **CALCITRIOL** | 5.4633 | 7 | `Novel Candidate` | n_targets=1 | shared_genes=VDR | pathways(2)=VDR->Nuclear Receptor transcripti<-VDR; VDR->SUMOylation of intracellular<-VDR | disease_bridge=obsolete_chronic kidney (13g), obsolete_psoriasis(11g) |
| 9 | `CHEMBL196` | **ASCORBIC ACID** | 5.4379 | 17 | `Novel Candidate` | n_targets=14 | disease_bridge=obsolete_Myocardial Isch(18g), obsolete_chronic kidney (13g) |
| 10 | `CHEMBL497` | **CLIOQUINOL** | 5.0982 | 2 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_peripheral vasc(4g) |
| 11 | `CHEMBL4597200` | **TRICHLOROTELLURATE** | 5.0393 | 5 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_psoriasis(11g), obsolete_atopic eczema(10g) |
| 12 | `CHEMBL1293` | **BETA CAROTENE** | 5.0357 | 4 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_Myocardial Isch(18g), obsolete_myocardial infa(3g) |
| 13 | `CHEMBL2105611` | **TACALCITOL** | 4.9928 | 2 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_psoriasis(11g), psoriasis vulgaris(8g) |
| 14 | `CHEMBL877` | **TRANEXAMIC ACID** | 4.9799 | 21 | `Novel Candidate` | n_targets=1 | disease_bridge=obsolete_stroke(23g), obsolete_coronary artery(19g) |
| 15 | `CHEMBL1201749` | **DIFLUPREDNATE** | 4.9695 | 4 | `Novel Candidate` | n_targets=1 | pathways(47)=NR3C1->Nuclear Receptor transcripti<-VDR; NR3C1->Potential therapeutics for S<-IGHV1-2 | disease_bridge=abnormal inflammatory re(2g) |

### 6.2 Tuberculosis

| Rank | ChEMBL ID | Drug Name | Score | Train Deg | Known Status | Mechanistic Evidence Path |
|---:|---|---|---:|---:|---|---|
| 1 | `CHEMBL1282` | **IMIQUIMOD** | 7.4248 | 12 | `Novel Candidate` | n_targets=2 | pathways(20)=TLR7->Potential therapeutics for S<-FKBP1A; TLR7->Potential therapeutics for S<-IFNGR1 | disease_bridge=obsolete_cirrhosis of li(9g), graft versus host diseas(8g) |
| 2 | `CHEMBL36` | **PYRIMETHAMINE** | 7.0451 | 11 | `Novel Candidate` | n_targets=2 | pathways(10)=HEXB->Neutrophil degranulation<-APRT; HEXB->Neutrophil degranulation<-CD14 | disease_bridge=schizophrenia(8g), depressive disorder(8g) |
| 3 | `CHEMBL1536` | **ERGOCALCIFEROL** | 6.6014 | 31 | `strict` | n_targets=27 | shared_genes=VDR | pathways(12)=VDR->Nuclear Receptor transcripti<-NR3C1; VDR->Nuclear Receptor transcripti<-RARA | disease_bridge=asthma(9g), obsolete_chronic kidney (8g) |
| 4 | `CHEMBL1140` | **NIACINAMIDE** | 6.0940 | 8 | `Novel Candidate` | n_targets=4 | pathways(10)=BST1->Neutrophil degranulation<-APRT; BST1->Neutrophil degranulation<-CD14 | disease_bridge=obsolete_psoriasis(18g), obsolete_chronic kidney (8g) |
| 5 | `CHEMBL1622` | **FOLIC ACID** | 5.7872 | 15 | `trial-stage` | n_targets=3 | pathways(10)=FOLR3->Neutrophil degranulation<-APRT; FOLR3->Neutrophil degranulation<-CD14 | disease_bridge=obsolete_rheumatoid arth(9g), obsolete_chronic kidney (8g) |
| 6 | `CHEMBL196` | **ASCORBIC ACID** | 5.7073 | 17 | `trial-stage` | n_targets=14 | disease_bridge=obsolete_pneumonia(12g), obsolete_nicotine depend(10g) |
| 7 | `CHEMBL184` | **ACYCLOVIR** | 5.6472 | 11 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_stroke(5g), obsolete_obesity(5g) |
| 8 | `CHEMBL506` | **PRIMAQUINE** | 5.6075 | 4 | `Novel Candidate` | n_targets=1 | disease_bridge=obsolete_malaria(4g) |
| 9 | `CHEMBL710` | **FINASTERIDE** | 5.5275 | 8 | `Novel Candidate` | n_targets=3 | disease_bridge=obsolete_acne(7g), obsolete_prostate adenoc(4g) |
| 10 | `CHEMBL1349` | **VALACYCLOVIR** | 5.3210 | 6 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_chronic obstruc(10g), obsolete_schizoaffective(9g) |
| 11 | `CHEMBL1401` | **NITAZOXANIDE** | 5.2537 | 10 | `strict` | n_targets=0 | disease_bridge=obsolete_rheumatoid arth(9g), obsolete_periodontitis(6g) |
| 12 | `CHEMBL600` | **ACETYLCYSTEINE** | 5.2482 | 32 | `strict` | n_targets=10 | pathways(40)=CHUK->FCERI mediated NF-kB activat<-RPS27A; CHUK->IKK complex recruitment medi<-CD14 | disease_bridge=psoriasis vulgaris(15g), obsolete_nicotine depend(10g) |
| 13 | `CHEMBL986` | **RETINOL** | 5.2365 | 8 | `strict` | n_targets=7 | disease_bridge=obsolete_acne(7g), obsolete_gastroenteritis(5g) |
| 14 | `CHEMBL119` | **TRIMETREXATE** | 5.2351 | 4 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_sarcoma(4g), pancreatic carcinoma(3g) |
| 15 | `CHEMBL42336` | **METHIONINE** | 5.1563 | 5 | `Novel Candidate` | n_targets=4 | disease_bridge=hyperlipidemia(4g), obsolete_rhabdomyosarcom(3g) |

### 6.3 Dengue

| Rank | ChEMBL ID | Drug Name | Score | Train Deg | Known Status | Mechanistic Evidence Path |
|---:|---|---|---:|---:|---|---|
| 1 | `CHEMBL137` | **METRONIDAZOLE** | 6.0671 | 17 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_pneumonia(7g), obsolete_blood coagulati(7g) |
| 2 | `CHEMBL1282` | **IMIQUIMOD** | 6.0461 | 12 | `Novel Candidate` | n_targets=2 | pathways(32)=TLR7->Potential therapeutics for S<-CYSLTR1; TLR7->Potential therapeutics for S<-IL1R1 | disease_bridge=obsolete_melanoma(3g), obsolete_cutaneous melan(3g) |
| 3 | `CHEMBL1440` | **TETRACYCLINE** | 5.7583 | 21 | `Novel Candidate` | n_targets=1 | disease_bridge=obsolete_pneumonia(7g), obsolete_blood coagulati(7g) |
| 4 | `CHEMBL112` | **ACETAMINOPHEN** | 5.7454 | 23 | `Novel Candidate` | n_targets=5 | shared_genes=PTGS2 | pathways(28)=PTGES3->Potential therapeutics for S<-CYSLTR1; PTGES3->Potential therapeutics for S<-IL1R1 | disease_bridge=obsolete_rheumatoid arth(7g), obsolete_allergic rhinit(5g) |
| 5 | `CHEMBL493287` | **GLUCOSAMINE** | 5.7379 | 3 | `Novel Candidate` | n_targets=4 | shared_genes=IFNG,MMP9,TNF | pathways(78)=IFNG->IFNG signaling activates MAP<-IFNG; IFNG->IFNG signaling activates MAP<-MAPK1 | disease_bridge=obsolete_irritable bowel(5g), sarcoidosis(4g) |
| 6 | `CHEMBL139` | **DICLOFENAC** | 5.6911 | 20 | `Novel Candidate` | n_targets=2 | shared_genes=PTGS2 | pathways(23)=PTGS2->Interleukin-10 signaling<-CSF2; PTGS2->Interleukin-10 signaling<-CXCL10 | disease_bridge=obsolete_rheumatoid arth(7g), obsolete_blood coagulati(7g) |
| 7 | `CHEMBL294199` | **CAPSAICIN** | 5.6726 | 12 | `Novel Candidate` | n_targets=1 | disease_bridge=obsolete_blood coagulati(7g), obsolete_myocardial infa(6g) |
| 8 | `CHEMBL1433` | **DOXYCYCLINE ANHYDROUS** | 5.6660 | 31 | `Novel Candidate` | n_targets=4 | pathways(77)=MMP1->Activation of Matrix Metallo<-CMA1; MMP1->Activation of Matrix Metallo<-MMP3 | disease_bridge=obsolete_pneumonia(7g), obsolete_myocardial infa(6g) |
| 9 | `CHEMBL184` | **ACYCLOVIR** | 5.6596 | 11 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_peripheral vasc(6g), obsolete_myocardial infa(6g) |
| 10 | `CHEMBL606` | **MISOPROSTOL** | 5.5691 | 14 | `Novel Candidate` | n_targets=4 | pathways(10)=PTGER1->G alpha (q) signalling event<-ANXA1; PTGER1->G alpha (q) signalling event<-CYSLTR1 | disease_bridge=obsolete_blood coagulati(7g), obsolete_myocardial infa(6g) |
| 11 | `CHEMBL3989740` | **DOXYCYCLINE HYCLATE** | 5.5614 | 8 | `Novel Candidate` | n_targets=4 | pathways(77)=MMP1->Activation of Matrix Metallo<-CMA1; MMP1->Activation of Matrix Metallo<-MMP3 | disease_bridge=obsolete_age-related mac(5g), hereditary hemorrhagic t(2g) |
| 12 | `CHEMBL1434` | **MINOCYCLINE** | 5.5338 | 32 | `Novel Candidate` | n_targets=20 | shared_genes=IL1B,MAPK1,MMP9,NOS2 | pathways(236)=ALOX5->Interleukin-4 and Interleuki<-ANXA1; ALOX5->Interleukin-4 and Interleuki<-FASLG | disease_bridge=obsolete_rheumatoid arth(7g), obsolete_pneumonia(7g) |
| 13 | `CHEMBL1200928` | **ZINC ACETATE ANHYDROUS** | 5.5220 | 4 | `Novel Candidate` | n_targets=117 | shared_genes=C5,MMP9 | pathways(658)=A2M->Degradation of the extracell<-MMP3; A2M->Degradation of the extracell<-MMP9 | disease_bridge=obsolete_pneumonia(7g), porphyrin metabolism dis(1g) |
| 14 | `CHEMBL9967` | **PIRENZEPINE** | 5.3853 | 2 | `Novel Candidate` | n_targets=1 | pathways(2)=CHRM1->G alpha (q) signalling event<-ANXA1; CHRM1->G alpha (q) signalling event<-CYSLTR1 | disease_bridge=obsolete_gastroesophagea(3g), peptic ulcer disease(1g) |
| 15 | `CHEMBL4297088` | **CHLORHEXIDINE GLUCONATE** | 5.3378 | 7 | `Novel Candidate` | n_targets=0 | disease_bridge=obsolete_pneumonia(7g), cystic fibrosis(3g) |

