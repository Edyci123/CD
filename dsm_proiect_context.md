# Proiect DSM: detecția metastazelor de cancer mamar în patch-uri histopatologice

**Curs:** Data Science în Medicină (DSM), UPB, Facultatea de Automatică și Calculatoare
**Echipa:** Eduard Bicu, Mihnea Zamfir
**Titlul lucrării (M1):** *Robust Detection of Breast Cancer Lymph Node Metastases in Histopathology Patches: A Review of the State of the Art*
**Status:** M1 (literature review + prezentare) e gata. Urmează M2 (baseline) și M3 (experimentele pentru RQ-uri).

---

## 1. Problema, pe scurt

- Când o pacientă are cancer mamar, medicii verifică **ganglionii limfatici santinelă** ca să vadă dacă s-a răspândit cancerul. Statusul ganglionilor este unul dintre cei mai puternici factori de prognostic și intră direct în stadializare (pN).
- Patologul se uită la secțiuni de țesut colorate **H&E** (hematoxilină și eozină) la microscop. Munca e lentă, iar metastazele mici se ratează ușor.
- Lamele sunt scanate în **whole-slide images (WSI)**, imagini de ordinul gigapixelilor, așa că problema se poate aborda cu deep learning.
- **Formularea aleasă de noi:** clasificare la nivel de **patch**. Modelul primește o regiune mică de țesut (96×96 px) și decide dacă aceasta conține tumoră metastatică (clasificare binară).

## 2. Datasets

| Dataset | Ce conține | De ce contează |
|---|---|---|
| **CAMELYON16** [1] | 399 WSI de ganglioni santinelă din 2 spitale olandeze (270 train, 129 test), adnotări de tumoră la nivel de pixel | Setul de referință pentru detecție. Pe test: 129 lame, dintre care 49 cu metastaze și 80 fără |
| **CAMELYON17** [2] | 1.000 de lame de la 200 de pacienți din 5 centre, etichete la nivel de lamă și de pacient | Design multi-centru: expune diferențele dintre laboratoare, pe care un set dintr-un singur centru le ascunde |
| **PatchCamelyon (PCam)** [3] | 327.680 de patch-uri RGB de 96×96 extrase din CAMELYON16. Un patch e pozitiv dacă **centrul de 32×32** conține cel puțin un pixel de tumoră | **Datasetul nostru.** Transformă o problemă gigapixel într-o clasificare binară care se antrenează pe un singur GPU |

**Detalii PCam:**
- Split-uri: **262.144 train / 32.768 validare / 32.768 test**.
- Split-urile **nu au WSI comune** (fără leakage între lame).
- Clasele sunt echilibrate **50/50**.

**Alte surse din literatură:** TCGA și arhive private de spital cu 10^4–10^5 WSI, folosite pentru supervizare slabă și pentru pretrainingul foundation models.

## 3. Evaluare și limitările ei

**Metrici pe niveluri:**
- **Patch (PCam):** accuracy și AUC, plus precision, recall (sensitivity), specificity și F1 când costul metastazelor ratate contează.
- **Leziune (CAMELYON16):** scorul **FROC**, adică sensibilitatea medie la 0,25–8 fals pozitive per lamă.
- **Lamă (CAMELYON16):** AUC.
- **Pacient (CAMELYON17):** stadializarea pN, evaluată cu **quadratic weighted Cohen's kappa**.
- **Foundation models:** linear probing, few-shot și zero-shot, pe multe task-uri.

**Rezultate de referință (verificate în papers):**
- CAMELYON16, clasificare de lame [1]: cel mai bun CNN are **AUC 0,994**, un patolog fără limită de timp **0,966**, iar un panel de 11 patologi cu limită de timp realistă **0,810** (medie).
- Detecție de leziuni, Liu et al. [4]: detectorul bazat pe Inception atinge **92,4%** sensibilitate la 8 FP/lamă, față de **82,7%** pentru cea mai bună metodă anterioară și **73,2%** pentru un patolog care caută exhaustiv.
  > ⚠️ În prima versiune a raportului, 73,2% fusese atribuit greșit „metodei anterioare". Am corectat în Google Doc și în .tex. Dacă mai apare undeva varianta veche, schimb-o.
- PCam [3]: DenseNet obține **AUC 0,955**, iar P4M-DenseNet (rotation-equivariant) **0,963**.

**Trei limitări ale protocoalelor de evaluare:**
1. Scorurile de la niveluri diferite nu se pot compara: un AUC pe patch-uri din PCam spune puțin despre acuratețea pe lamă sau pe pacient.
2. Echilibrul 50/50 din PCam nu corespunde prevalenței clinice, deci accuracy-ul supraestimează performanța reală, iar precision-ul nu are sens fără recalibrare.
3. Seturile de test oficiale provin din aceleași centre ca datele de train, deci scorurile in-distribution nu măsoară generalizarea la spitale, scanere sau protocoale de colorare noi.

## 4. Arhitecturi și tehnici de antrenare

**CNN-uri la nivel de patch.**
- Primele lucrări au antrenat CNN-uri GoogLeNet/Inception pe patch-uri și au agregat ieșirile în hărți de probabilitate pentru tumoră [1], [4].
- Veeling et al. [3] au observat că țesutul nu are o orientare canonică și au construit CNN-uri **rotation-equivariant (P4M, group convolutions)**. Acestea au depășit CNN-urile standard pe PCam la același număr de parametri. Tot lucrarea lor a introdus PCam.

**Gestionarea colorației (stain).**
- Culoarea diferă între laboratoare din cauza colorării și a scanării.
- **Stain normalization** (Macenko [5]) mapează imaginile la un aspect de referință al colorației.
- Tellez et al. [6] au comparat normalizarea cu augmentarea de culoare pe 4 task-uri multi-centru. Concluzia lor: **augmentarea de culoare HSV/HED a fost cel mai important factor pentru generalizare**, iar normalizarea a adăugat puțin odată ce augmentarea era puternică. ← baza pentru **RQ1**.

**Supervizare slabă cu Multiple Instance Learning (MIL).**
- Adnotările la nivel de pixel sunt scumpe.
- Ilse et al. [7] au propus **attention-based MIL pooling**: modelul învață o pondere pentru fiecare patch dintr-un „bag", iar ponderile arată ce patch-uri au contat.
- Campanella et al. [8] au antrenat MIL pe **44.732 WSI** folosind doar diagnosticul la nivel de lamă și au obținut AUC > 0,98 pentru cancer de prostată, carcinom bazocelular și metastaze de cancer mamar.
- **CLAM** [9] combină attention MIL cu clustering la nivel de instanță, așa că se antrenează cu mult mai puține lame.
- **TransMIL** [10] înlocuiește atenția independentă cu un transformer care modelează corelațiile dintre patch-uri.

**Foundation models.**
- Aceste modele preantrenează vision transformers mari prin self-supervised learning pe lame neetichetate, apoi refolosesc encoderul înghețat.
- **UNI** [11]: ViT-L antrenat cu DINOv2 pe peste 100M patch-uri din peste 100.000 WSI, evaluat pe 34 de task-uri.
- **Prov-GigaPath** [12]: preantrenat pe 1,3 miliarde de patch-uri din 171.189 de lame, cu un slide encoder LongNet.
- **TITAN** [13]: aliniază reprezentările de lamă cu rapoarte de patologie și 423.122 de caption-uri sintetice, ceea ce permite clasificare zero-shot și generare de rapoarte.

## 5. Cum se leagă lucrările

Literatura formează un lanț în care fiecare direcție rezolvă o limitare a celei dinainte:

1. Challenge-urile [1], [2] au definit task-ul și au arătat că CNN-urile pot concura cu patologii. PCam [3] l-a făcut accesibil ca benchmark mic.
2. Rotation equivariance [3] și augmentarea de culoare [6] tratează invarianțe cunoscute ale histologiei, pe care un CNN simplu ar trebui altfel să le învețe din date.
3. Metodele MIL [7]–[10] elimină dependența de adnotările la nivel de pixel. TransMIL [10] relaxează presupunerea de independență din attention MIL [7], [9].
4. Foundation models [11]–[13] rezolvă lipsa datelor etichetate prin pretraining pe lame neetichetate.
5. Kömen et al. [14] (PathoROB) testează presupunerea comună tuturor acestor lucrări, anume că acuratețea mare in-distribution înseamnă că modelul se bazează pe biologie. **Răspunsul lor: nu neapărat.**

## 6. Gap-uri (lipsurile actuale)

- **Robustețe la variație non-biologică (cel mai mare gap).** PathoROB [14] a testat **20 de foundation models** pe 4 dataseturi, cu 28 de clase biologice din **34 de centre medicale**. Toate modelele au avut deficite de robustețe: **centrul medical de proveniență se putea recupera din feature-uri**, iar aceste feature-uri au cauzat erori de diagnostic. Corecțiile post-hoc au redus problema, dar nu au eliminat-o.
- **Validare externă limitată.** Majoritatea metodelor sunt testate pe split-uri din centrele de train. Evaluarea multi-centru, ca în CAMELYON17, rămâne excepția.
- **Cost de adnotare și de compute.** Etichetele la nivel de pixel cer patologi experți, iar foundation models cer sute de mii de lame și bugete mari de GPU [11], [12], pe care puține grupuri le pot reproduce.
- **Designul benchmark-urilor.** Benchmark-urile de patch-uri echilibrate și seturile de test in-distribution umflă rezultatele.

## 7. Probleme specifice CNN-urilor (din deck)

| Problemă | Ce înseamnă |
|---|---|
| **Culoare / stain** | Același țesut arată diferit de la un laborator sau scanner la altul, iar CNN-ul poate învăța culoarea în loc de morfologie |
| **Orientare și context** | Țesutul nu are un „sus", deci un CNN standard nu e invariant la rotații/flip-uri. Pe PCam contează și faptul că eticheta depinde doar de centrul de 32×32 |
| **Texture bias** | CNN-urile se bazează pe textură mai mult decât pe formă |
| **Overconfidence** | Probabilitățile sunt slab calibrate |
| **Dezechilibru de clase** | În practica clinică, patch-urile cu tumoră sunt rare (PCam e artificial 50/50) |
| **Domain gap ImageNet → țesut** | Feature-urile preantrenate pe poze naturale nu se transferă perfect pe histologie |
| **Lipsa de explicabilitate** | E greu de arătat unui medic de ce a decis modelul |

> Imaginile de țesut din deck sunt reale (H&E, din proiectul open-source **torchstain**, licență MIT), dar **nu sunt din PCam și nu au etichetă de tumoră**. Sunt doar ilustrative. Dacă întreabă cineva, așa le explici.

## 8. Direcția propusă

- **Model:** baseline **ResNet-18** pe PCam.
- **Filozofie:** experimente controlate pe care o echipă de 2 oameni le poate reproduce complet, în loc să concurăm cu foundation models.
- **Metrici:** AUC (principala, pentru că nu depinde de prag), accuracy, precision, recall, F1. **Recall-ul** e raportat special, pentru că metastazele ratate sunt eroarea costisitoare.
- **Test de robustețe:** un set de test separat cu **shift-uri sintetice de culoare/colorație**, aplicate în spațiile **HED** și **HSV** la intensități fixe și crescătoare. Raportăm AUC la fiecare nivel.

### RQ1: robustețe prin augmentare de culoare

> **RQ1:** To what extent does stain-aware colour augmentation (HED/HSV jitter) improve the robustness of a CNN classifier on PCam under controlled colour and stain shifts, compared with geometric augmentation alone?

- **Ce adresează:** gap-ul de robustețe identificat de PathoROB [14]. Verifică dacă rezultatele lui Tellez [6] se mențin pe PCam.
- **Design:**
  - Antrenăm același ResNet-18 în două variante: (a) doar augmentare geometrică (rotații, flip-uri) și (b) augmentare geometrică plus jitter de culoare HED/HSV.
  - Testăm ambele variante pe testul PCam curat și pe testul cu shift-uri de colorație la intensități crescătoare.
  - Comparăm cât de mult scade AUC-ul pe măsură ce shift-ul crește. Întrebarea e dacă augmentarea micșorează scăderea.

### RQ2: transfer learning cu buget limitat de etichete

> **RQ2:** How does ImageNet-pretrained transfer learning compare with training from scratch when only 10%, 25% or 100% of the PCam training labels are available?

- **Ce adresează:** gap-ul de cost al adnotărilor, la o scară la care putem rula toate condițiile pe aceleași split-uri.
- **Design:** 2 inițializări (ImageNet-pretrained și from scratch) × 3 bugete de etichete (10%, 25%, 100% din train), adică **6 rulări**, toate cu același split de test.

### Plan pe milestone-uri

| Milestone | Ce facem |
|---|---|
| **M1** ✅ | Literature review IEEE (14 referințe, cele 5 sub-întrebări notate) + prezentare de 5 minute |
| **M2** | Baseline ResNet-18 pe PCam |
| **M3** | Experimentele de augmentare (RQ1) și de buget de etichete (RQ2), cu setul de test cu shift de culoare |
| **Future work** | Compararea unui encoder de foundation model înghețat (ex. UNI [11]) cu CNN-urile noastre, sub aceleași shift-uri |

## 9. Întrebări probabile la prezentare și răspunsuri scurte

- **De ce PCam și nu lame întregi?** Rulează pe un singur GPU, are split-uri publice fixe fără suprapunere de lame și ne lasă să izolăm efectul augmentării și al pretrainingului.
- **De ce nu folosiți un foundation model?** Scopul nostru e un experiment controlat pe care îl putem reproduce complet. Compararea cu un encoder înghețat (UNI) e trecută ca future work.
- **Cum simulați shift-ul de colorație?** Perturbăm imaginile în spațiul HED și în HSV, cu intensități fixe și crescătoare, și raportăm AUC la fiecare nivel.
- **E un shift sintetic același lucru cu un spital nou?** Nu. E un proxy controlat. Testul mai puternic ar fi pe date multi-centru reale, de exemplu CAMELYON17.
- **Care e metrica principală?** AUC, pentru că nu depinde de prag. Raportăm și recall, pentru că metastazele ratate sunt eroarea costisitoare.

## 10. Livrabile existente

- **Google Doc „Calcul cu date"**, cu tab-urile **M1 Report** și **M1 Presentation** (script cu timing pe 5:00, împărțit Speaker A / Speaker B, plus Q&A).
- **m1_review.tex / m1_review.pdf**: același raport în template-ul IEEE two-column (2 pagini de conținut + pagina de referințe).
- **Deck de 9 slide-uri**: text în engleză, speaker notes în română, figuri ilustrative (patch cu cutia de 32×32 a etichetei PCam, variante cu shift de colorație, variante cu rotație/flip, tabelul cu probleme CNN) și slide de metrici cu cifrele reale de mai sus.

## 11. Referințe

1. B. E. Bejnordi et al., "Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer," *JAMA*, vol. 318, no. 22, pp. 2199–2210, 2017. doi: 10.1001/jama.2017.14585
2. P. Bándi et al., "From detection of individual metastases to classification of lymph node status at the patient level: The CAMELYON17 challenge," *IEEE Trans. Med. Imag.*, vol. 38, no. 2, pp. 550–560, 2019. doi: 10.1109/TMI.2018.2867350
3. B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling, "Rotation equivariant CNNs for digital pathology," *Proc. MICCAI*, LNCS 11071, pp. 210–218, 2018. doi: 10.1007/978-3-030-00934-2_24
4. Y. Liu et al., "Detecting cancer metastases on gigapixel pathology images," arXiv:1703.02442, 2017.
5. M. Macenko et al., "A method for normalizing histology slides for quantitative analysis," *Proc. IEEE ISBI*, pp. 1107–1110, 2009. doi: 10.1109/ISBI.2009.5193250
6. D. Tellez et al., "Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology," *Med. Image Anal.*, vol. 58, 101544, 2019. doi: 10.1016/j.media.2019.101544
7. M. Ilse, J. M. Tomczak, M. Welling, "Attention-based deep multiple instance learning," *Proc. ICML*, PMLR 80, pp. 2127–2136, 2018.
8. G. Campanella et al., "Clinical-grade computational pathology using weakly supervised deep learning on whole slide images," *Nat. Med.*, vol. 25, pp. 1301–1309, 2019. doi: 10.1038/s41591-019-0508-1
9. M. Y. Lu et al., "Data-efficient and weakly supervised computational pathology on whole-slide images," *Nat. Biomed. Eng.*, vol. 5, pp. 555–570, 2021. doi: 10.1038/s41551-020-00682-w
10. Z. Shao et al., "TransMIL: Transformer based correlated multiple instance learning for whole slide image classification," *Proc. NeurIPS*, vol. 34, pp. 2136–2147, 2021.
11. R. J. Chen et al., "Towards a general-purpose foundation model for computational pathology," *Nat. Med.*, vol. 30, pp. 850–862, 2024. doi: 10.1038/s41591-024-02857-3
12. H. Xu et al., "A whole-slide foundation model for digital pathology from real-world data," *Nature*, vol. 630, pp. 181–188, 2024. doi: 10.1038/s41586-024-07441-w
13. T. Ding et al., "A multimodal whole-slide foundation model for pathology," *Nat. Med.*, 2025. doi: 10.1038/s41591-025-03982-3
14. J. Kömen et al., "Towards robust foundation models for digital pathology," *Nat. Commun.*, vol. 17, 5218, 2026. doi: 10.1038/s41467-026-73923-2
