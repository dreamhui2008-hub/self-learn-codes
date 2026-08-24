# Cloud CV Learning In Practice

This is a cloud-only computer vision fluency module.

Use:

- `TUTORIAL.md` as the full lesson guide
- `experiments.ipynb` as the cloud notebook where drills are run

Primary runtime: Kaggle Notebook.

Primary data pattern:

```text
/kaggle/input/<dataset>/<class_name>/<image files>
```

Primary output pattern:

```text
/kaggle/working/cloud_cv_runs/<timestamp>_<run_name>/
```

The tutorial emphasizes the practical layers that many compact CV tutorials skip: mounted data discovery, labels from paths, dataset classes, dataloaders, artifacts, checkpoint reload, batch inference, and handoff files.

