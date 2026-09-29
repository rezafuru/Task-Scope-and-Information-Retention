# Pilot: feature preservation across the residual blocks

Stages and the files they retain, in order.

| Stage | Command | External input | Retained output |
| --- | --- | --- | --- |
| CUB sources | `scripts/pilot_data.py cub-development`, `cub-test` | CUB data root, `CUB_200_2011.tgz`, 200-to-36 family mapping | `data/cub_{development,test}/inventory.json`, `source_hash_inventory.json` |
| Exits | `scripts/pilot_data.py cub-features` | the prepared inventories | `features/<population>/{layer3,block1,block2,block3}.npy`, `extraction.json` |
| Inverses | `scripts/pilot_inversion.py fit` | ImageNet training archive with a byte-offset index, official validation directory | `inverse/<exit>/{run.json,inverse.pt}` |
| Inversions | `scripts/pilot_inversion.py sources`, `apply`, `convert` | CUB data root | `inversions/<population>/<exit>/{preview.json,inventory.json,*.png}` |
| Readers | `scripts/pilot_pair.py readouts --observation <name>` | the extracted exits and the inversions of the development population | `pair/readouts/<observation>/{models.json,reader_seed{17,23}.pt}` |
| Measurement | `scripts/pilot_pair.py heldout` | the fitted readers, the test exits and the test inversions | `pair/heldout/{evaluation.json,predictions.npz}` |
| Wing regions | `scripts/pilot_retention.py` | CUB part annotations, the development inversions | `pair/cue_retention.json` |

An observation is one depth read in one way, eight in all: `layer3`, `block1`, `block2`, `block3`
for the complete native feature map, and `inverse_layer3` to `inverse_block3` for the RGB
reconstruction of that map. The `readouts` stage fits one observation per invocation, two readers
each, and `heldout` reads all eight.

Every reader carries a fresh 200-class output and is fitted on the complete development
population, 4,794 fitting photographs across 200 species with six validation photographs each.
The development exits and inversions therefore have to cover every category, so those stages run
with `--fine-labels` and no values. The measurement needs only the two species, which is the
default for the test population. `heldout` restricts each fitted 200-class output to Indigo
Bunting and Blue Grosbeak on the 30 official test photographs per species, and records the two
fitted-reader accuracies and their arithmetic mean. There is no probability ensemble and no
interval.

`scripts/figures/pilot.py` reads
`results/pilot/pair/{asset_evidence.json,heldout_evaluation.json,cue_retention.json,sources.json}`
and the ten images in `assets/pilot/`. `heldout_evaluation.json` is the output of the `heldout`
stage, and the figure reads the two fitted-reader accuracies per observation from its
`cub_test.restricted_200way` block. `asset_evidence.json` pins that file by SHA256 and repeats
the accuracy block it was verified against. `cue_retention.json` supplies the wing boxes only.
`sources.json` holds the bird box, the split and the per-view file hashes of the two illustrated
photographs, assembled by hand from the CUB annotation.

The ten images in `assets/pilot/` are retained rather than regenerated. They were copied by hand
from intermediate inversion renders that no longer exist, and no stage in this chain wrote that
selection. Rerunning the inversion stages produces the same reconstructions for every photograph
in the population, under the observation ids rather than these filenames.

torchvision is an optional dependency, imported only when a stage needs the network. The weight
enum is `ResNet50_Weights.IMAGENET1K_V2`, downloaded into the torch hub cache on first use and
recorded by SHA256 in every stage record. The fitting stages need a CUDA device to finish in
reasonable time.
