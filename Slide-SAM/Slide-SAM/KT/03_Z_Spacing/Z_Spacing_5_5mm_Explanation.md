# Understanding z-spacing and the proposed 5.5 mm threshold

Date: 30 September 2026.

## 1. What is z-spacing?

Z-spacing is the physical distance between the centers of neighboring slices along the volume's z-axis. It is measured in **millimeters (mm)**, not meters.

For example, 3 mm spacing means neighboring slice centers are 3 mm apart; 5.5 mm spacing means they are 5.5 mm apart. Larger spacing can produce larger anatomical changes between adjacent slices, making propagation harder.

Spacing is not necessarily the same as acquisition slice thickness. Physical orientation must also be checked before assuming an array axis corresponds to anatomical z.

## 2. What did the authors experiment with?

The authors tested how segmentation performance changes when slices are farther apart. This was a robustness experiment, not a demonstrated method for improving accuracy.

```text
Original WORD scan, approximately 2.5–3.0 mm spacing
    → artificially increase z-spacing
    → run Slide-SAM
    → measure how segmentation performance changes
```

The author rebuttal supplied in our discussion contains these example results:

| Tested z-spacing | Pancreas Dice (%) |
|------------------|------------------:|
| 2.75 mm          | 71.84             |
| 5.50 mm          | 66.19             |
| 6.87 mm          | 60.66             |
| 11.00 mm         | 48.41             |

For this example, accuracy worsens as spacing increases. The effect depends on anatomy and spacing; it is not a universal failure boundary at exactly 5.5 mm.

The accessible manuscript describes the experiment using spacing ranges: native WORD at 2.5–3.0 mm, increasing to 5.0–6.0 mm at a resampling ratio of 2, and then larger ranges. The numbers above are from the supplied rebuttal and should not be labeled as a verbatim reproduction of the manuscript table. See [Slide-SAM, Appendix D.2 and Table 4](https://arxiv.org/html/2311.10121v3#A4.SS2) and the [OpenReview discussion](https://openreview.net/forum?id=LNdU9RTv3L).

## 3. Where did our 5.5 mm trigger come from?

| Source | Meaning of 5.5 mm |
|--------|------------------|
| Supplied author rebuttal | One spacing tested after resampling WORD scans |
| Authors' manuscript | Reports a related 5.0–6.0 mm experimental range; does not establish a universal corrective threshold |
| Our initial proposal | Uses spacing greater than 5.5 mm as a candidate trigger for corrective resampling |
| Inspected project code | No corrective `z > 5.5 mm → resample to 3 mm` rule is implemented |

**The authors did not establish 5.5 mm as a special cutoff or demonstrate that resampling a thick scan back to 3 mm fixes the problem.** Our trigger is a proposed experimental setting, not an author-validated rule.

## 4. How is our proposed experiment different?

Our hypothesis goes in the opposite direction:

```text
Scan with large through-plane spacing
    → interpolate intermediate samples toward 3 mm spacing
    → run Slide-SAM
    → restore predictions to the original grid
    → compare with segmentation without corrective resampling
```

Interpolation might make successive model inputs smoother and improve propagation. It does not recover anatomy that was never captured, and it can introduce smoothing artifacts. Improvement must be measured rather than assumed.

For implementation, use linear interpolation for image intensities and nearest-neighbor interpolation for discrete labels. Preserve physical geometry and the initial prompt's physical location. Score restored predictions against the unchanged reference labels on the declared evaluation grid.

Changing only z-spacing to 3 mm is through-plane spacing normalization, not necessarily isotropic resampling: isotropic means equal spacing along all three axes.

## 5. Would the proposed rule activate on our current datasets?

Read-only image-header checks performed on 29 September 2026 found:

| Local dataset/split | Volumes checked | Native z-spacing | Volumes above 5.5 mm |
|---------------------|----------------:|------------------|--------------------:|
| BTCV                | 30              | 2.5, 3.0, or 5.0 mm | 0                |
| WORD training       | 100             | 2.5 or 3.0 mm    | 0                   |
| WORD validation     | 20              | 2.5 or 3.0 mm    | 0                   |
| WORD test           | 30              | 2.5 or 3.0 mm    | 0                   |

These are findings from the local files, not a claim about every distribution or version of these datasets. BTCV details are saved in [the basic audit evidence](Basic_Audit_2026-09-29.json). WORD counts come from the header-only inspection recorded in this conversation.

**Enabling the proposed threshold would change none of these native scans.** We cannot measure its benefit on this cohort simply by turning the rule on.

## 6. How could we test the idea meaningfully?

Two options are available:

1. Use a separate cohort with naturally large through-plane spacing.
2. Create explicitly labeled synthetic thick-spacing versions of held-out scans.

For a synthetic experiment, compare:

```text
Original scan
    → construct one degraded, larger-spacing version
        → Branch A: segment directly
        → Branch B: interpolate toward 3 mm, then segment
    → map both predictions to the same declared evaluation grid
    → compare using the same reference labels and physical initial prompt
```

Do not give Branch B access to the original fine-spacing image when reconstructing intermediate samples: it must use the same degraded image available to Branch A. Keep variants from each patient together and do not count them as independent patients. Choose spacing settings using development data, then freeze them before confirmation.

This would test whether interpolation helps segmentation after information loss. It would remain a synthetic robustness study, not evidence from naturally thick-slice clinical acquisitions.

## 7. Decision for our project

Continue with **box padding first**. Treat z-spacing handling as a later hypothesis requiring suitable data and a separate protocol. Keep the distinction explicit:

- **Authors' experiment:** increase spacing to investigate degradation.
- **Our proposed improvement:** reduce the working-grid spacing by interpolation and test for recovery in segmentation performance.
- **Current evidence:** the proposed improvement has not yet been evaluated.

Related documents: [Basic audit and padding pilot plan](Basic_Audit_and_Padding_Pilot_Plan.md), [post-five-fold decision guide](Post_5Fold_Decision_and_Novelty_Plan.md), and [research novelty proposal](SlideSAM_Research_Novelty_Proposal.md).
