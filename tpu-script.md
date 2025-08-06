## TPU VM

### Enviroment variable
```bash
export ZONE=us-east5-b
export PROJECT_ID=leadership-hpc-discovery
export PROJECT=leadership-hpc-discovery
export ACCELERATOR_TYPE=v6e-1
export NUM_SLICES=1
export TPU_NAME=v6e-1-demo
```

### Create TPU VM
```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
--accelerator-type=$ACCELERATOR_TYPE \
--version=v2-alpha-tpuv6e \
--zone=$ZONE \
--project=$PROJECT 
```
### SSH into TPU VM
```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
   --project ${PROJECT} \
   --zone ${ZONE}
```

### Clear ADCS path temporarily
```bash
export PATH=/usr/local/google/home/jackyf/venvp2/bin:/usr/local/google/home/jackyf/.local/bin:/usr/lib/google-golang/bin:/usr/local/buildtools/java/jdk/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/google/home/jackyf/.terraform/bin:/usr/local/google/home/jackyf/google-tool/cluster-toolkit/gcluster:/usr/local/google/home/jackyf/google-tool/cluster-toolkit:/usr/local/google/home/jackyf/.config/Code/User/globalStorage/github.copilot-chat/debugCommand:/usr/local/google/home/jackyf/.terraform/bin:/usr/local/google/home/jackyf/google-tool/cluster-toolkit/gcluster:/usr/local/google/home/jackyf/google-tool/cluster-toolkit:/usr/local/google/home/jackyf/.vscode/extensions/ms-python.debugpy-2025.4.1-linux-x64/bundled/scripts/noConfigScripts:/usr/local/google/home/jackyf/.config/Code/User/globalStorage/github.copilot-chat/debugCommand:/usr/local/google/home/jackyf/.terraform/bin:/usr/local/google/home/jackyf/google-tool/cluster-toolkit/gcluster:/usr/local/google/home/jackyf/google-tool/cluster-toolkit
```

## Create xpk TPU k8s cluster

1. Specify your TPU GKE cluster configs.
```bash
export PROJECT_ID=leadership-hpc-discovery
export PROJECT=leadership-hpc-discovery

# v6e-256

export CLUSTER_NAME=v6e-demo #<your_cluster_name>
export NETWORK_NAME=${CLUSTER_NAME}-only-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-only-fw
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
export TPU_TYPE=v6e-256 #<your TPU Type>
export NUM_SLICES=2 #<number of TPU node-pools you want to create>
export ZONE=europe-west4-a
export REGION=europe-west4
export OUTPUT_DIR=gs://jf-tpu/bm

# v6e-8

export CLUSTER_NAME=v6e-8-demo #<your_cluster_name>
export NETWORK_NAME=${CLUSTER_NAME}-only-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-only-fw
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
export TPU_TYPE=v6e-8 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to create>
export ZONE=us-east5-b
export REGION=us-east5
export OUTPUT_DIR=gs://jf-tpu/benchmarks

# v5p-512

export CLUSTER_NAME=jacky-v5p-512 #<your_cluster_name>
export NETWORK_NAME=${CLUSTER_NAME}-only-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-only-fw
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
export TPU_TYPE=v5p-512 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to create>
export ZONE=europe-west4-b
export REGION=europe-west4

gcloud config set account jackyf@hpc-discovery.com
gcloud config set compute/zone $ZONE
gcloud config set project ${PROJECT_ID}
```

2. Create the network and firewall for this cluster if it doesn’t exist yet.

```bash
NETWORK_NAME_1=${CLUSTER_NAME}-mtu9k-1-${ZONE}
NETWORK_FW_NAME_1=${NETWORK_NAME_1}-fw-1-${ZONE}

# Use a custom network for better performance as well as avoid the default network to be overloaded.
gcloud compute networks create ${NETWORK_NAME_1} --mtu=8896 --project=${PROJECT} --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create ${NETWORK_FW_NAME_1} --network ${NETWORK_NAME_1} --allow tcp,icmp,udp --project=${PROJECT}

# Secondary subnet for multinic experience. Need custom ip routing to be different from first network’s subnet.
export NETWORK_NAME_2=${CLUSTER_NAME}-privatenetwork-2-${ZONE}
export SUBNET_NAME_2=${CLUSTER_NAME}-privatesubnet-2-${ZONE}
export FIREWALL_RULE_NAME=${CLUSTER_NAME}-privatefirewall-2-${ZONE}
export ROUTER_NAME=${CLUSTER_NAME}-network-2-${ZONE}
export NAT_CONFIG=${CLUSTER_NAME}-natconfig-2-${ZONE}

gcloud compute networks create "${NETWORK_NAME_2}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom --project=$PROJECT
gcloud compute networks subnets create "${SUBNET_NAME_2}" --network="${NETWORK_NAME_2}" --range=10.10.0.0/18 --region="${REGION}" --project=$PROJECT
gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" --network "${NETWORK_NAME_2}" --allow tcp,icmp,udp --project="${PROJECT}"
gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT}" \
  --network="${NETWORK_NAME_2}" \
  --region="${REGION}"
gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --project="${PROJECT}" \
  --enable-logging
```

3. Create GKE cluster with TPU node-pools

```bash
export CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias --enable-multi-networking --network=${NETWORK_NAME_1} --subnetwork=${NETWORK_NAME_1}"

export NODE_POOL_ARGUMENTS="--additional-node-network network=${NETWORK_NAME_2},subnetwork=${SUBNET_NAME_2}"

python3 xpk.py cluster create --cluster $CLUSTER_NAME --num-slices=2 --tpu-type=$TPU_TYPE --zone=$ZONE  --project=$PROJECT --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" --custom-nodepool-arguments="${NODE_POOL_ARGUMENTS}" --on-demand 

python3 xpk.py cluster create-pathways --cluster $CLUSTER_NAME --pathways-gce-machine-type=n2-standard-64 --num-slices=2 --tpu-type=$TPU_TYPE --zone=$ZONE  --project=$PROJECT --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" --custom-nodepool-arguments="${NODE_POOL_ARGUMENTS}" --reservation=cloudtpu-20250318132139-1477063367 --gke-version=1.32.3-gke.1440000

xpk cluster create-pathways \
--num-slices=1 \
--tpu-type=v6e-8 \
--pathways-gce-machine-type=n2-standard-64 \
--on-demand \
--project=${PROJECT} \
--zone=${ZONE} \
--cluster=${CLUSTER_NAME} \
--custom-cluster-arguments="--network=${NETWORK} --subnetwork=${SUBNETWORK} --enable-ip-alias"

```

4. Performance Daemonset
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/ai-on-gke/9ff340f07f70be0130454f9e7238551587242b75/scripts/network-setup/v6e-network-optimization.yaml
```

5. Run workload

### Maxtest workload
```bash
# 0.1.1
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="llama2_70b_4096_sc" \
    --base_docker_image=maxtext_base_image \
    --num_steps=20

#0.1.0
python3 benchmarks/benchmark_runner.py xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="llama2_70b_4096_sc" \
    --base_docker_image=maxtext_base_image

#pathway

xpk workload create-pathways \
--workload=WORKLOAD \
--cluster=CLUSTER \
--num-slices=WORKLOAD_NODEPOOL_COUNT \
--tpu-type=TPU_TYPE \
--project=leadership-hpc-discovery \
--zone=ZONE \
--docker-image='gcr.io/leadership-hpc-discovery/USER_runner' \
--command="python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=gs://BUCKET_NAME per_device_batch_size=1 enable_checkpointing=false remat_policy=full global_parameter_scale=1 steps=20 max_target_length=2048 use_iota_embed=true reuse_example_batch=1 dataset_type=synthetic attention=flash gcs_metrics=True enable_single_controller=True run_name=RUN_NAME-pathways-job"

```

### checkpoint workload

```bash
python3 xpk/xpk.py workload create --cluster $CLUSTER_NAME --base-docker-image maxtext_base_image --ramdisk-directory /tmp --workload orbax1 --tpu-type=v6e-256 --num-slices=2 --max-restarts=10 --command "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && python3 MaxText/train.py MaxText/configs/base.yml remat_policy=full global_parameter_scale=256 base_output_directory=/var dataset_type=synthetic steps=1000 per_device_batch_size=4 ici_fsdp_parallelism=-1 ici_tensor_parallelism=16 max_target_length=256 reuse_example_batch=1 enable_emergency_checkpoint=true checkpoint_period=1000 local_checkpoint_directory=/tmp local_checkpoint_period=20 use_replicator_service=True replicator_backup_interval_minutes=30 run_name=orbax enable_goodput_recording=False monitor_goodput=False"
```


6.GKE Cluster Deletion

```bash
python3 xpk.py cluster delete \
--cluster $CLUSTER_NAME --zone=$ZONE  --project=$PROJECT
```
