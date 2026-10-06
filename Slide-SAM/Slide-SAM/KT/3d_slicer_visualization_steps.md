# View Slide-SAM results in 3D Slicer

3D Slicer lets us view a CT scan, rotate a predicted organ in 3D, and compare it with the reference outline (ground truth).

**Keep Slide-SAM in JupyterLab. Install Slicer on your own computer and open the saved results there.** You do not need to download the whole dataset, copy model weights, or rerun training.

```text
JupyterLab / project server              Your computer
--------------------------              -------------
CT scan + saved prediction  --download--> 3D Slicer
                                             |
                                             +--> Browse CT slices
                                             +--> Rotate the organ
                                             +--> Compare with ground truth
```

## 1. Download and install Slicer

1. Go to [the official Slicer download page](https://download.slicer.org/).
2. Choose the **Stable Release** for your computer's operating system.
3. Install it:
   - **Windows:** open the installer and follow the steps.
   - **macOS:** open the downloaded `.dmg` and drag Slicer into Applications.
   - **Linux:** extract the downloaded archive and open the `Slicer` application inside it. Some systems need extra packages; see the installation guide below.
4. Open 3D Slicer.

[Official installation guide](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html)

## 2. Download one example from JupyterLab

Start with **case 0031 and its saved liver prediction**. The paths below are relative to the Slide-SAM project folder.

| File | Where to find it |
|---|---|
| Original CT scan | `datasets/01_BCV-Abdomen/Training/img/img0031.nii.gz` |
| Predicted liver | `epoch_20_visualizations_copilot/model_latest/case_0031/liver/predicted_mask.nii.gz` |
| Ground-truth organ labels, optional | `datasets/01_BCV-Abdomen/Training/label/label0031.nii.gz` |
| Details of this prediction, useful for reference | `epoch_20_visualizations_copilot/model_latest/case_0031/liver/metadata.json` |

In JupyterLab's file browser:

1. Find each file.
2. Right-click it and select **Download**.
3. Keep the downloaded files together on your computer:

```text
SlideSAM_case0031/
|-- img0031.nii.gz          CT scan
|-- predicted_mask.nii.gz   Model's liver prediction
|-- label0031.nii.gz        Reference organ labels (optional)
`-- metadata.json          Record of the prediction (optional)
```

Keep the `.nii.gz` files compressed. Slicer can open them directly. All three image files must belong to the **same case**.

The CT and reference paths above are recorded in the saved metadata. If they have moved, use the matching case 0031 files from the dataset's current location.

If JupyterLab and Slicer are on the same computer, and the files are accessible there, open them directly instead of downloading them.

## 3. Open the CT scan

1. Drag `img0031.nii.gz` into the Slicer window.
2. Load it as a **Volume** and click **OK**.
3. Choose the **Four-up** layout from the layout selector in the top toolbar.
4. Use the slice sliders to move through the scan.

The layout gives you three slice views and a 3D view:

```text
+-----------------------+-----------------------+
| Slice across the body | 3D view               |
|                       | Organ appears here    |
|                       | after the next step   |
+-----------------------+-----------------------+
| Slice from the front  | Slice from the side   |
|                       |                       |
+-----------------------+-----------------------+
```

## 4. Open the predicted liver in 3D

1. Drag `predicted_mask.nii.gz` into Slicer.
2. In the loading dialog, set **Description** to **Segmentation**.
3. Click **OK**.
4. Open the **Segmentations** module and select the loaded prediction.
5. Click **Show 3D**.
6. In the 3D view, drag with the left mouse button to rotate the organ and use the mouse wheel to zoom.

You should now see the predicted liver in 3D and its outline or colored overlay on the CT slices. If the organ is outside the view, use the view's reset or fit control.

**Check that the mask lines up with the liver in the CT.** If it appears shifted or flipped, check the case and exported image geometry before judging model quality.

[Official segmentation loading and display guide](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmentations.html)

## 5. Compare the prediction with ground truth

1. Drag `label0031.nii.gz` into Slicer and load it as another **Segmentation**.
2. This file contains several organs. For this liver example, show only **label 6**, which is the liver in this project's BTCV mapping. Imported segment names may be generic; identify the segment corresponding to label value 6.
3. Give the reference liver a **green** color and the predicted liver a **red** color.
4. Use the eye icons to show or hide each segmentation. Adjust opacity or use outlines to make both boundaries visible.
5. Scroll through the CT slices and inspect the organ from different directions.

```text
Green = reference liver       Red = predicted liver
              |                         |
              +------------+------------+
                           |
                    Compare the edges
                           |
          +----------------+----------------+
          |                |                |
      Missing tissue   Extra tissue    Matching region
      Model missed     Model included Model agrees
      part of liver    too much       with reference
```

| What you notice | What to record |
|---|---|
| Reference has tissue that prediction misses | Missing tissue |
| Prediction extends outside the reference | Extra tissue |
| Prediction ends before the reference does | Possible early stopping; needs further checking |
| Most differences are near the outline | Boundary differences |

Use the **3D view to spot the overall problem** and the **CT slices to locate it precisely**.

Example observation: “The prediction covers the middle of the liver but misses part of its lower end.” A visual observation alone does not prove what caused the error.

## 6. Use this in our experiments

For a future box-padding or slice-spacing experiment, open both predictions on the same CT:

```text
Same CT scan + same reference + same initial prompt
                         |
             +-----------+-----------+
             |                       |
      Original prediction     Modified prediction
             |                       |
             +-----------+-----------+
                         |
          Did missing tissue decrease?
          Did unwanted tissue increase?
```

Record the case, organ, checkpoint, and what changed. Compare the visual findings with Dice scores across held-out cases. Keep manually edited masks separate from the original model predictions.

**About this example:** the saved case 0031 prediction comes from an earlier visualization run. It is useful for learning Slicer, but should not be presented as a current held-out cross-validation result. The current fold evaluator does not save its prediction masks; that export step would need to be added later.

This first viewing workflow uses already saved files and does not require stopping or changing any running training process.
