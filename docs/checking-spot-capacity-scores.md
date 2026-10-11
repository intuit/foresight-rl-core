# Checking GPU Spot Capacity Scores

Before launching a SageMaker Managed Spot Training job (`infra.use_spot_instances: true`),
especially for scarce instance types like `ml.p6-b200.48xlarge`, check EC2's **Spot
Placement Score** for the instance type/region/AZ you're targeting. A job requesting
capacity that doesn't exist can sit in `SecondaryStatus=Starting` for many hours (or
the full `job.max_wait_hours` window) before failing or eventually succeeding.

## What the score means

`ec2:GetSpotPlacementScores` returns a **1-10 relative likelihood score** per
Availability Zone — not a literal machine count. AWS does not expose actual
free-instance counts via any API (this is intentional: capacity is shared,
dynamic, and commercially sensitive).

| Score | Label | Meaning |
|---|---|---|
| 1-3 | 🔴 Near-Zero | Request will almost certainly fail or be interrupted immediately |
| 4-6 | 🟠 Limited | May succeed, but expect interruptions within minutes to hours |
| 7-8 | 🟢 Available | Capacity exists, low interruption risk |
| 9-10 | 🟢 Abundant | Plenty of capacity, safe for longer runs |

**Scores are volatile** — they can swing from 9 to 1 and back within an hour. Treat
a check as a point-in-time snapshot, not a reservation. Re-check immediately before
launching, and be ready to relaunch if a job stalls despite a good score.

## Quick check (ad hoc, no deployment)

Run directly with `boto3` — no infrastructure to deploy, no IAM changes beyond
what's already needed for `aws_training_job` jobs:

```bash
source .venv/bin/activate && python3 << 'EOF'
import boto3
from collections import defaultdict

REGIONS = ['us-west-2', 'us-east-1']
INSTANCE_TYPES = ['p5.48xlarge', 'p5en.48xlarge', 'p6-b200.48xlarge', 'p6-b300.48xlarge', 'p4de.24xlarge']
TARGET_COUNTS = [1, 2, 4]  # match your job's instance_count(s) of interest

results = []

for region in REGIONS:
    ec2 = boto3.client('ec2', region_name=region)
    az_id_to_name = {}
    az_resp = ec2.describe_availability_zones(Filters=[{'Name': 'region-name', 'Values': [region]}])
    for az in az_resp.get('AvailabilityZones', []):
        az_id_to_name[az['ZoneId']] = az['ZoneName']

    for instance_type in INSTANCE_TYPES:
        try:
            offerings = ec2.describe_instance_type_offerings(
                LocationType='availability-zone',
                Filters=[{'Name': 'instance-type', 'Values': [instance_type]}]
            )
            offered_azs = sorted([o['Location'] for o in offerings.get('InstanceTypeOfferings', [])])
        except Exception:
            offered_azs = []
        if not offered_azs:
            continue  # not offered in this region at all
        for target_count in TARGET_COUNTS:
            try:
                resp = ec2.get_spot_placement_scores(
                    InstanceTypes=[instance_type],
                    TargetCapacity=target_count,
                    SingleAvailabilityZone=True,
                    RegionNames=[region]
                )
                for item in resp.get('SpotPlacementScores', []):
                    az_id = item.get('AvailabilityZoneId', '')
                    az_name = az_id_to_name.get(az_id, az_id)
                    results.append((instance_type, region, az_name, target_count, item.get('Score')))
            except Exception as e:
                print(f"score error {instance_type} {region} x{target_count}: {e}")

grouped = defaultdict(dict)
for instance_type, region, az, target_count, score in results:
    grouped[(instance_type, region, az)][target_count] = score

print(f"{'Instance Type':<20}{'Region':<12}{'AZ':<14}{'Score(1)':<10}{'Score(2)':<10}{'Score(4)':<10}")
for (instance_type, region, az), scores in sorted(grouped.items()):
    print(f"{instance_type:<20}{region:<12}{az:<14}{scores.get(1,'-'):<10}{scores.get(2,'-'):<10}{scores.get(4,'-'):<10}")
EOF
```

Adjust `REGIONS`, `INSTANCE_TYPES`, and `TARGET_COUNTS` to match what you're
actually about to launch — check the exact `instance_count` your job requests,
since scores commonly differ a lot between e.g. 1-instance and 4-instance requests
for the same AZ.

### Required IAM permissions

```
ec2:DescribeSpotPriceHistory
ec2:DescribeInstanceTypeOfferings
ec2:DescribeAvailabilityZones
ec2:GetSpotPlacementScores
```

These are read-only EC2 describe/get calls — no changes to any resource, safe to
run from the same role used to submit training jobs.

## Interpreting results for a specific launch

1. Set `TARGET_COUNTS` to match your `infra.instance_count`. A score good for
   1 instance can be near-zero for 4 — check the count you actually need, not just 1.
2. Prefer AZs with score ≥ 7 for the instance count you need. Score 4-6 may still
   succeed but expect interruptions.
3. If every AZ/region/instance-type combination you check is at score 1, that's a
   real signal — no combination is likely to succeed right now. Re-check later
   rather than repeatedly relaunching into a dead pool.
4. Cross-check against `ec2:DescribeInstanceTypeOfferings` — a high score means
   nothing if the instance type isn't even offered in that AZ (the script above
   already filters to only-offered AZs).

## Forcing a specific AZ (once you've picked one)

Spot Placement Scores tell you *where* to aim, but SageMaker's `CreateTrainingJob`
API has no `AvailabilityZone` parameter — the only way to constrain which AZ(s)
a job can land in is via `infra.vpc.subnets` (a subnet is AZ-specific in AWS).
Without a `vpc:` block, SageMaker's service-managed placement decides the AZ
opaquely — you cannot see or influence which AZ it's attempting via
`DescribeTrainingJob` or CloudTrail (SageMaker's internal spot-capacity
requests do not appear as `RunInstances`/`CreateFleet`/`RequestSpotInstances`
events in this account's CloudTrail).

```yaml
infra:
  vpc:
    subnets:
      - subnet-abc123    # in the AZ you want to target
    security_groups:
      - sg-xyz000
```

See [`intuit/docs/account-setup/aws/sagemaker-training-job.md`](../intuit/docs/account-setup/aws/sagemaker-training-job.md#optional--vpc-configuration)
for the full VPC configuration reference. If no subnet exists yet in the AZ you
want, creating one requires `ec2:CreateSubnet` — not included in the standard
`data_engineer_prd`-style caller role; you'll need someone with VPC/networking
admin permissions in the account.
