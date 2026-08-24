#   Best Fit Before Project 1

  1. Kaggle Learn: Computer Vision (https://www.kaggle.com/learn/computer-vision)
     Best for CNN mechanics: convolution, ReLU, max pooling, stride/padding, custom convnets, augmentation. It has tutorial + exercise format, so it matches your “mechanical fluency” goal.

  2. PyTorch Official Tutorials (https://docs.pytorch.org/tutorials/)
     Use this for PyTorch-specific mechanics: tensors, datasets/loaders, autograd, optimization loops, saving/loading, transfer learning, data loading optimization. It is less “homework-like” than Kaggle, but
     it is the primary source.

  3. Kaggle Learn: Intro to Machine Learning (https://www.kaggle.com/learn/intro-to-machine-learning) and Machine Learning Explainability (https://www.kaggle.com/learn/machine-learning-explainability)
     Good for validation, overfitting, leakage, model interpretation, feature importance, SHAP. Not deep-learning-heavy, but very useful for experimental discipline.

  Big Data / Data Engineering
  4. DataTalks.Club Data Engineering Zoomcamp (https://datatalks.club/docs/courses/data-engineering-zoomcamp/)
  Best free project-oriented path for big data engineering. It covers Docker, Terraform, workflow orchestration, BigQuery, dbt, Spark, Kafka/Flink, and a final project.

  5. Apache Spark Official Quick Start (https://spark.apache.org/docs/latest/quick-start.html) and Spark Examples (https://spark.apache.org/examples)
     Best for low-level Spark fluency: DataFrames, SQL, lazy evaluation, structured operations, local pyspark, and spark-submit.

  Production ML / MLOps
  6. DataTalks.Club MLOps Zoomcamp (https://datatalks.club/docs/courses/mlops-zoomcamp/)
  Good for MLflow, orchestration, deployment, monitoring, testing, CI/CD, Terraform. As of the current docs, it is self-paced in 2026 rather than running a live cohort.

  7. Made With ML (https://github.com/GokuMohandas/Made-With-ML)
     Strong for production ML habits: testing, packaging, experiment tracking, deployment, monitoring, iteration. This is closer to “how ML systems are built” than pure model training.

  8. Full Stack Deep Learning (https://fullstackdeeplearning.com/course/2022/)
     Good for labs around infrastructure, experiment management, testing, data management, and deployment thinking.

  AI / LLM Engineering
  9. Hugging Face Course (https://huggingface.co/docs/course/en/chapter1/1)
  Best hands-on path for Transformers, Datasets, Tokenizers, Accelerate, fine-tuning, and sharing models.

  10. DataTalks.Club LLM Zoomcamp (https://datatalks.club/docs/courses/llm-zoomcamp/)
     Good for RAG, vector search, orchestration, evaluation, monitoring, and end-to-end LLM app projects.

  11. AI Engineering Field Guide: Home Assignments (https://github.com/alexeygrigorev/ai-engineering-field-guide/blob/main/interview/questions/06-home-assignments.md)
     Best if you want real job-style take-home assignments rather than course exercises. It catalogs actual AI/ML engineering assignment patterns like RAG systems, agents, document processing, evals, and LLM-
     as-judge workflows.

My recommendation before Project 1: do Kaggle Computer Vision fully, then 5-10 PyTorch official tutorials/recipes around data loaders, training loops, saving/loading, and transfer learning. If you want big-
  data muscle too, do Spark Quick Start plus one Data Engineering Zoomcamp Spark module.

# Best Fit for DevOps / SRE / cloud operations

Best Human-Led Path
  If you want guided, hands-on, not Codex-made:

  1. KodeKloud DevOps Path
     Probably the best match for your level and frustration. It has structured paths for Linux, Docker, Kubernetes, Terraform, CI/CD, monitoring, and hands-on labs.
     Link: KodeKloud DevOps Learning Path (https://kodekloud.com/learning-path/devops)

  2. Play with Docker
     Browser-based Docker labs. No local setup. Good for container basics.
     Link: Play with Docker Classroom (https://training.play-with-docker.com/)

  3. KillerCoda
     Browser-based real Linux/Kubernetes environments. Good for Kubernetes drills without installing clusters locally.
     Link: KillerCoda (https://killercoda.com/about)

  4. Linux Foundation: Introduction to Kubernetes LFS158
     Free, beginner-friendly Kubernetes course with labs/assignments.
     Link: LFS158 (https://training.linuxfoundation.org/training/introduction-to-kubernetes/)

  5. AWS Cloud Quest or Google Cloud Skills Boost
     Use these when you want real cloud resources, storage, networking, IAM, and managed services.
     Links: AWS Cloud Quest (https://docs.cloudquest.skillbuilder.aws/coming-soon/index.html), Google Dataproc Lab (https://www.cloudskillsboost.google/focuses/672)

  6. Google SRE Books
     Not beginner drills, but essential for production thinking: reliability, monitoring, incidents, toil, automation, rollout discipline.
     Link: Google SRE Books (https://sre.google/books/)

  Good references:

  - AWS S3 resiliency, lifecycle, versioning (https://docs.aws.amazon.com/AmazonS3/latest/userguide/disaster-recovery-resiliency.html)
  - AWS Well-Architected Reliability Pillar (https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html)
  - Google Cloud Storage lifecycle management (https://docs.cloud.google.com/storage/docs/lifecycle)
  - Azure Blob lifecycle management (https://learn.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-overview)

* Start with KodeKloud DevOps path.
* Use Play with Docker only while learning Docker.
* Use KillerCoda only while learning Kubernetes.
* Use AWS docs only when studying storage/retention.
* Use Google SRE books after you know the tools enough to understand the problems.

#   Mindmap
  A researcher training from scratch eventually touches this chain:

  problem / product requirement
  -> data source
  -> data storage and retention
  -> data validation
  -> training environment
  -> experiment tracking
  -> checkpoint/artifact storage
  -> evaluation gates
  -> model registry
  -> packaging/container
  -> serving or batch inference
  -> deployment pipeline
  -> monitoring/logging/tracing
  -> rollback/retraining
  -> cost/security/access control

  For an AI app/agent, add:

  prompt/version management
  -> tool/function permissions
  -> retrieval/data connectors
  -> trace logs
  -> eval sets
  -> human review / guardrails

  So the missing category is best called:

  ML systems + DevOps + MLOps + cloud operations

  Not pure big data. Not pure Python. Not pure ML.

  Best Learning Spine
  What Is Not Fully Free / Has Caveats

  - DeepLearning.AI Machine Learning in Production: certificate/graded assignments are PRO/paid. Useful, but not free end-to-end. Replace with Full Stack Deep Learning + DataTalksClub MLOps Zoomcamp.
  - KodeKloud: mostly paid/freemium. Good structure, but not free-only. Replace with roadmap.sh + official docs/labs.
  - AWS/GCP/Azure labs: docs are free, but real cloud resources can bill you. Use only after you understand billing/free-tier limits.
  - BentoCloud / Grafana Cloud / Evidently Cloud: cloud services may have free tiers, but do not rely on them as “free.” Use OSS/local modes first.

  Free-Only Replacement Path
  Use this instead:

  1. Big Picture ML Systems
      - Full Stack Deep Learning 2022 (https://fullstackdeeplearning.com/course/2022/)
        Free lecture/lab material. Best for seeing the whole ML app lifecycle: infra, data, testing, deployment, monitoring.

      - DataTalksClub MLOps Zoomcamp (https://datatalks.club/docs/courses/mlops-zoomcamp/)
        Free, hands-on MLOps course. It covers tracking, deployment, monitoring, testing, CI/CD. Use selectively.

  2. DevOps Foundations
      - Missing Semester (https://missing.csail.mit.edu/) for shell, Git, debugging, packaging.
      - Linux Journey (https://labex.io/linuxjourney) for beginner Linux.
      - Pro Git Book (https://git-scm.com/book/en/v2) for Git.
      - roadmap.sh DevOps/MLOps (https://roadmap.sh/) as the map, not the course.

  3. Containers
      - Docker Get Started Workshop (https://docs.docker.com/get-started/workshop/)
      - Play with Docker (https://training.play-with-docker.com/) for browser labs.

  4. Kubernetes
      - Kubernetes Official Basics (https://kubernetes.io/docs/tutorials/kubernetes-basics/)
      - Linux Foundation LFS158 (https://training.linuxfoundation.org/training/introduction-to-kubernetes/) is listed as free.
      - KillerCoda (https://killercoda.com/about) for browser-based Linux/Kubernetes scenarios.

  5. Infrastructure As Code
      - Terraform Tutorials (https://developer.hashicorp.com/terraform)
        Free official tutorials. Learn write -> plan -> apply -> state.

  6. CI/CD
      - GitLab CI/CD first pipeline (https://docs.gitlab.com/ci/quick_start/)
      - GitHub Actions deployment docs (https://docs.github.com/en/actions/get-started/continuous-deployment)

  7. MLOps Tools
      - MLflow Tracking / Model Registry (https://www.mlflow.org/docs/latest/ml/model-registry/workflow/)
      - Made With ML: Testing ML Systems (https://madewithml.com/courses/mlops/testing/)

  8. Serving
      - FastAPI Tutorial (https://github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/index.md)
      - BentoML open-source quickstart (https://docs.bentoml.org/en/latest/get-started/hello-world.html)
      - Later only: Ray Serve (https://docs.ray.io/en/latest/serve/getting_started.html), KServe (https://kserve.github.io/website/docs/getting-started)

  9. Monitoring / Reliability / Retention
      - Prometheus first steps (https://prometheus.io/docs/introduction/first_steps/)
      - Grafana first dashboard (https://grafana.com/docs/grafana/latest/fundamentals/getting-started/first-dashboards/)
      - OpenTelemetry getting started (https://opentelemetry.io/docs/getting-started/)
      - Evidently OSS ML checks (https://docs.evidentlyai.com/quickstart_ml)
      - AWS S3 Versioning (https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html), S3 Object Lock (https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html), AWS Reliability
        Pillar (https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html)

  Concrete Order
  Do not try all of this at once.

  Phase 1: understand map
      DeepLearning.AI Machine Learning in Production
      Full Stack Deep Learning lectures 1-6

  Phase 2: learn deployment substrate
      KodeKloud Linux/shell/Git basics
      Docker
      Play with Docker

  Phase 3: learn orchestration
      Kubernetes basics
      KillerCoda Kubernetes drills
      Terraform basics

  Phase 4: learn ML ops layer
      MLflow tracking/model registry
      BentoML or FastAPI serving
      Evidently monitoring

  Phase 5: capstone
      MLOps Zoomcamp or your own Project 1 extended with Docker/K8s/MLflow

  And yes: this is not mainly learnable from Python. Python is the model/training/API glue. The systems layer uses:

  shell
  YAML
  Dockerfiles
  Terraform HCL
  Kubernetes manifests
  CI/CD configs
  cloud IAM/storage/networking concepts

  You do not need SQL first for this path. You need deployment literacy.